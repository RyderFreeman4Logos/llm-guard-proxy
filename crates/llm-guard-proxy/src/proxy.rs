use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::State,
    http::{
        HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri,
        header::{ACCEPT, AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST},
    },
    routing::get,
};
use bytes::BytesMut;
use futures_util::{Stream, StreamExt};
use llm_guard_proxy_core::{
    AppConfig, AttemptId, AttemptRecord, AttemptStatus, ConfigHandle, DownstreamMode, Health,
    LICENSE, MetadataConfig, ObservabilityStore, RawPayloads, RequestId, RequestRecord,
    RequestStatus, SERVICE_NAME, UpstreamMode, redact_upstream_base_url,
    validate_upstream_base_url,
};
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

mod model_metadata;

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROXY_BODY_BYTES_U64: u64 = 64 * 1024 * 1024;
const UPSTREAM_REQUEST_TIMEOUT_SECS: u64 = 120;
const HEADER_VALUE_NOT_UTF8: &str = "[non-utf8]";

/// Shared HTTP proxy state.
#[derive(Clone, Debug)]
pub(crate) struct ProxyState {
    config: ConfigHandle,
    config_path: PathBuf,
    store: ObservabilityStore,
    client: Client,
    in_flight_requests: Arc<Semaphore>,
    max_in_flight_requests: usize,
}

impl ProxyState {
    /// Builds cloneable proxy state for axum handlers.
    #[must_use]
    pub(crate) fn new(
        config: ConfigHandle,
        config_path: PathBuf,
        store: ObservabilityStore,
        client: Client,
        max_in_flight_requests: usize,
    ) -> Self {
        Self {
            config,
            config_path,
            store,
            client,
            in_flight_requests: Arc::new(Semaphore::new(max_in_flight_requests)),
            max_in_flight_requests,
        }
    }

    fn try_acquire_in_flight_permit(&self) -> Result<OwnedSemaphorePermit, InFlightLimitExceeded> {
        self.in_flight_requests
            .clone()
            .try_acquire_owned()
            .map_err(|_error| InFlightLimitExceeded {
                max_in_flight_requests: self.max_in_flight_requests,
            })
    }
}

/// Builds the bounded upstream HTTP client used by the proxy.
///
/// # Errors
///
/// Returns a reqwest error if the HTTP client cannot be built.
pub(crate) fn build_http_client() -> Result<Client, reqwest::Error> {
    Client::builder()
        .timeout(Duration::from_secs(UPSTREAM_REQUEST_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Builds the OpenAI-compatible proxy router.
pub(crate) fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/config-summary", get(health_handler))
        .fallback(proxy_handler)
        .with_state(state)
}

/// Renders the compact health/config summary kept from the bootstrap binary.
#[must_use]
pub(crate) fn render_health(config: &AppConfig, path: &Path, request_id: &RequestId) -> String {
    let health = Health::current();
    let name = SERVICE_NAME;
    let license = LICENSE;
    let readiness = health.readiness().as_str();
    let config_path = path.display();
    let heartbeat_mode = config.heartbeat.mode.as_str();
    let heartbeat_interval_secs = config.heartbeat.interval_secs;
    let observability_enabled = config.observability.enabled;

    format!(
        "{name} request_id={request_id} readiness={readiness} license={license} config_path={config_path} heartbeat_mode={heartbeat_mode} heartbeat_interval_secs={heartbeat_interval_secs} observability_enabled={observability_enabled}"
    )
}

async fn health_handler(State(state): State<ProxyState>) -> Response<Body> {
    match state.config.snapshot() {
        Ok(config) => text_response(
            StatusCode::OK,
            render_health(&config, &state.config_path, &RequestId::generate()),
        ),
        Err(error) => proxy_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "config_snapshot_failed",
            &error.to_string(),
        ),
    }
}

async fn proxy_handler(State(state): State<ProxyState>, request: Request<Body>) -> Response<Body> {
    let request_id = RequestId::generate();
    let started_at_unix_ms = unix_time_millis();
    if let Err(error) = validate_openai_path(request.uri().path()) {
        let finished_at_unix_ms = unix_time_millis();
        let error_type = error.error_type();
        let error_reason = error.to_string();
        let response = proxy_error_response(error.status(), error_type, &error_reason);
        let request_metadata = pre_upstream_request_metadata(
            request.method(),
            request.uri(),
            request.headers(),
            config_shielding_enabled(&state.config),
        );
        record_failed_request(
            &state.store,
            FailedRequestRecord {
                request_id,
                started_at_unix_ms,
                finished_at_unix_ms,
                http_status: error.status().as_u16(),
                error_type,
                error_reason,
                request_metadata,
                attempt: None,
            },
        );
        return response;
    }

    let permit = match state.try_acquire_in_flight_permit() {
        Ok(permit) => permit,
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_type = InFlightLimitExceeded::error_type();
            let error_reason = error.to_string();
            let response =
                proxy_error_response(InFlightLimitExceeded::status(), error_type, &error_reason);
            let request_metadata = pre_upstream_request_metadata(
                request.method(),
                request.uri(),
                request.headers(),
                config_shielding_enabled(&state.config),
            );
            record_failed_request(
                &state.store,
                FailedRequestRecord {
                    request_id,
                    started_at_unix_ms,
                    finished_at_unix_ms,
                    http_status: InFlightLimitExceeded::status().as_u16(),
                    error_type,
                    error_reason,
                    request_metadata,
                    attempt: None,
                },
            );
            return response;
        }
    };

    match forward_openai_request(&state, &request_id, started_at_unix_ms, request, permit).await {
        Ok(response) => response,
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_type = error.error_type();
            let error_reason = error.to_string();
            let response = proxy_error_response(error.status(), error_type, &error_reason);
            let request_metadata = error.request_metadata().cloned().unwrap_or_else(|| {
                BTreeMap::from([(String::from("proxy_error"), error_type.to_owned())])
            });
            record_failed_request(
                &state.store,
                FailedRequestRecord {
                    request_id,
                    started_at_unix_ms,
                    finished_at_unix_ms,
                    http_status: error.status().as_u16(),
                    error_type,
                    error_reason,
                    request_metadata,
                    attempt: error.attempt_record(),
                },
            );
            response
        }
    }
}

async fn forward_openai_request(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    request: Request<Body>,
    in_flight_permit: OwnedSemaphorePermit,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = request.into_parts();
    let method = parts.method;
    let uri = parts.uri;
    let downstream_headers = parts.headers;
    let shielding_enabled_hint = config_shielding_enabled(&state.config);
    let pre_body_request_metadata =
        pre_upstream_request_metadata(&method, &uri, &downstream_headers, shielding_enabled_hint);
    let body = read_body_bytes(body)
        .await
        .map_err(|error| error.with_request_metadata(pre_body_request_metadata))?;
    let body_read_request_metadata = base_request_metadata(
        &method,
        &uri,
        &downstream_headers,
        body.len().to_string(),
        shielding_enabled_hint,
    );
    let config = state.config.snapshot().map_err(|error| {
        ProxyError::config_snapshot(error.to_string())
            .with_request_metadata(body_read_request_metadata)
    })?;
    let request_metadata = request_metadata(
        &method,
        &uri,
        &downstream_headers,
        body.len(),
        config.shielding.enabled,
    );
    let upstream_url = build_upstream_url(&config.upstream.base_url, &uri)
        .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    let reqwest_method = upstream_method(&method)
        .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    let model_id = extract_model_id(&body);
    let attempt_id = AttemptId::for_request(request_id, 1);
    let attempt_started_at_unix_ms = unix_time_millis();
    let attempt_request_metadata = attempt_request_metadata(&method, &uri, &downstream_headers);
    let upstream_response = match send_upstream_request(
        &state.client,
        reqwest_method,
        upstream_url,
        &downstream_headers,
        body.clone(),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_reason = error.to_string();
            let attempt_record = failed_attempt_record(
                attempt_id,
                request_id.clone(),
                attempt_started_at_unix_ms,
                finished_at_unix_ms,
                error.error_type(),
                &error_reason,
                attempt_request_metadata,
            );
            return Err(error.with_observability(request_metadata, attempt_record));
        }
    };
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let upstream_mode = upstream_mode_from_headers(&upstream_headers);
    let response_parts = ForwardedResponseParts {
        store: state.store.clone(),
        request_id: request_id.clone(),
        started_at_unix_ms,
        attempt_id,
        attempt_started_at_unix_ms,
        upstream_mode,
        model_id,
        upstream_status,
        upstream_headers: upstream_headers.clone(),
        request_metadata,
        attempt_request_metadata,
    };
    if should_enrich_models_response(&method, &uri, &upstream_headers, &config) {
        return forward_enriched_models_response(
            response_parts,
            upstream_response,
            in_flight_permit,
            &config.upstream.metadata,
        )
        .await;
    }

    let observer = response_parts.into_observer();
    let response_body =
        ObservedUpstreamBody::new(upstream_response.bytes_stream(), observer, in_flight_permit);
    let response = downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    );

    Ok(response)
}

async fn read_body_bytes(body: Body) -> Result<Bytes, ProxyError> {
    to_bytes(body, MAX_PROXY_BODY_BYTES)
        .await
        .map_err(|error| ProxyError::request_body(error.to_string()))
}

async fn read_upstream_body_bytes(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> Result<Bytes, ProxyError> {
    let mut stream = Box::pin(stream);
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            ProxyError::upstream_body(format!(
                "upstream body stream failed: {}",
                sanitized_reqwest_error(&error)
            ))
        })?;
        let next_len = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| ProxyError::upstream_body(String::from("upstream body is too large")))?;
        if next_len > MAX_PROXY_BODY_BYTES {
            return Err(ProxyError::upstream_body(format!(
                "upstream body exceeded proxy limit: max_bytes={MAX_PROXY_BODY_BYTES}"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

fn should_enrich_models_response(
    method: &Method,
    uri: &Uri,
    upstream_headers: &HeaderMap,
    config: &AppConfig,
) -> bool {
    method == Method::GET
        && uri.path() == "/v1/models"
        && config.upstream.metadata.discovery_enabled
        && config.upstream.metadata.enrich_responses
        && response_body_fits_enrichment_limit(upstream_headers)
}

fn response_body_fits_enrichment_limit(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|body_len| body_len <= MAX_PROXY_BODY_BYTES_U64)
}

fn upstream_method(method: &Method) -> Result<reqwest::Method, ProxyError> {
    reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|error| ProxyError::invalid_method(error.to_string()))
}

async fn send_upstream_request(
    client: &Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: &HeaderMap,
    body: Bytes,
) -> Result<reqwest::Response, ProxyError> {
    let headers = forwarded_request_headers(downstream_headers);
    client
        .request(method, upstream_url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|source| {
            let failure = ReqwestFailureKind::from_error(&source);
            ProxyError::UpstreamTransport {
                failure,
                observability: None,
            }
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReqwestFailureKind {
    Timeout,
    Connect,
    Request,
    Body,
    Decode,
    Other,
}

impl ReqwestFailureKind {
    fn from_error(error: &reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout
        } else if error.is_connect() {
            Self::Connect
        } else if error.is_body() {
            Self::Body
        } else if error.is_decode() {
            Self::Decode
        } else if error.is_request() {
            Self::Request
        } else {
            Self::Other
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "timeout_failure",
            Self::Connect => "connect_failure",
            Self::Request => "request_failure",
            Self::Body => "body_failure",
            Self::Decode => "decode_failure",
            Self::Other => "unknown_failure",
        }
    }
}

impl fmt::Display for ReqwestFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

fn sanitized_reqwest_error(error: &reqwest::Error) -> String {
    ReqwestFailureKind::from_error(error).to_string()
}

struct ForwardedResponseParts {
    store: ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    attempt_id: AttemptId,
    attempt_started_at_unix_ms: u64,
    upstream_mode: UpstreamMode,
    model_id: Option<String>,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    attempt_request_metadata: BTreeMap<String, String>,
}

impl ForwardedResponseParts {
    fn into_observer(self) -> ForwardedBodyObserver {
        ForwardedBodyObserver {
            downstream_mode: downstream_mode_from_headers(&self.upstream_headers),
            store: self.store,
            request_id: self.request_id,
            started_at_unix_ms: self.started_at_unix_ms,
            attempt_id: self.attempt_id,
            attempt_started_at_unix_ms: self.attempt_started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            model_id: self.model_id,
            upstream_status: self.upstream_status,
            upstream_headers: self.upstream_headers,
            request_metadata: self.request_metadata,
            attempt_request_metadata: self.attempt_request_metadata,
        }
    }

    fn into_body_read_error(self, error: ProxyError) -> ProxyError {
        let finished_at_unix_ms = unix_time_millis();
        let error_reason = error.to_string();
        let attempt_record = failed_attempt_record(
            self.attempt_id,
            self.request_id,
            self.attempt_started_at_unix_ms,
            finished_at_unix_ms,
            error.error_type(),
            &error_reason,
            self.attempt_request_metadata,
        );
        error.with_observability(self.request_metadata, attempt_record)
    }
}

async fn forward_enriched_models_response(
    response_parts: ForwardedResponseParts,
    upstream_response: reqwest::Response,
    in_flight_permit: OwnedSemaphorePermit,
    metadata_config: &MetadataConfig,
) -> Result<Response<Body>, ProxyError> {
    let upstream_status = response_parts.upstream_status;
    let upstream_headers = response_parts.upstream_headers.clone();
    let body = match read_upstream_body_bytes(upstream_response.bytes_stream()).await {
        Ok(body) => body,
        Err(error) => return Err(response_parts.into_body_read_error(error)),
    };
    let body = model_metadata::enrich_models_body(metadata_config, body);
    let body_len = u64::try_from(body.len()).unwrap_or(u64::MAX);
    let observer = response_parts.into_observer();
    observer.record(body_len, &BodyCompletion::Succeeded);
    drop(in_flight_permit);

    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from(body),
    ))
}

struct ForwardedBodyObserver {
    store: ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    attempt_id: AttemptId,
    attempt_started_at_unix_ms: u64,
    downstream_mode: DownstreamMode,
    upstream_mode: UpstreamMode,
    model_id: Option<String>,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    attempt_request_metadata: BTreeMap<String, String>,
}

impl ForwardedBodyObserver {
    fn record(self, body_bytes: u64, completion: &BodyCompletion) {
        let finished_at_unix_ms = unix_time_millis();
        let response_metadata = response_metadata(
            self.upstream_status,
            &self.upstream_headers,
            body_bytes,
            finished_at_unix_ms.saturating_sub(self.started_at_unix_ms),
        );
        let attempt_response_metadata = response_metadata.clone();
        let request_record = RequestRecord {
            request_id: self.request_id.clone(),
            started_at_unix_ms: self.started_at_unix_ms,
            finished_at_unix_ms: Some(finished_at_unix_ms),
            downstream_mode: self.downstream_mode,
            upstream_mode: self.upstream_mode,
            model_id: self.model_id,
            input_fingerprint: None,
            status: completion.request_status(),
            http_status: Some(self.upstream_status.as_u16()),
            error_reason: completion.error_reason(),
            abort_reason: completion.abort_reason(),
            request_metadata: self.request_metadata,
            response_metadata,
            raw_payloads: RawPayloads::default(),
        };
        let attempt_record = AttemptRecord {
            attempt_id: self.attempt_id,
            request_id: self.request_id,
            attempt_number: 1,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            finished_at_unix_ms: Some(finished_at_unix_ms),
            upstream_mode: self.upstream_mode,
            status: completion.attempt_status(),
            http_status: Some(self.upstream_status.as_u16()),
            error_reason: completion.error_reason(),
            retry_reason: None,
            abort_reason: completion.abort_reason(),
            request_metadata: self.attempt_request_metadata,
            response_metadata: attempt_response_metadata,
            raw_payloads: RawPayloads::default(),
        };
        record_observability(&self.store, &request_record, Some(&attempt_record));
    }
}

enum BodyCompletion {
    Succeeded,
    UpstreamStreamError(String),
    DownstreamDropped,
}

impl BodyCompletion {
    const fn request_status(&self) -> RequestStatus {
        match self {
            Self::Succeeded => RequestStatus::Succeeded,
            Self::UpstreamStreamError(_) => RequestStatus::Failed,
            Self::DownstreamDropped => RequestStatus::Aborted,
        }
    }

    const fn attempt_status(&self) -> AttemptStatus {
        match self {
            Self::Succeeded => AttemptStatus::Succeeded,
            Self::UpstreamStreamError(_) => AttemptStatus::Failed,
            Self::DownstreamDropped => AttemptStatus::Aborted,
        }
    }

    fn error_reason(&self) -> Option<String> {
        match self {
            Self::UpstreamStreamError(error) => Some(format!("upstream_stream_error: {error}")),
            Self::Succeeded | Self::DownstreamDropped => None,
        }
    }

    fn abort_reason(&self) -> Option<String> {
        match self {
            Self::DownstreamDropped => Some(String::from("downstream_body_dropped_before_eof")),
            Self::Succeeded | Self::UpstreamStreamError(_) => None,
        }
    }
}

struct ObservedUpstreamBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: OwnedSemaphorePermit,
    bytes_seen: u64,
}

impl ObservedUpstreamBody {
    fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            bytes_seen: 0,
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion) {
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion);
        }
    }
}

impl Stream for ObservedUpstreamBody {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                let chunk_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                this.bytes_seen = this.bytes_seen.saturating_add(chunk_len);
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => {
                let completion =
                    BodyCompletion::UpstreamStreamError(sanitized_reqwest_error(&error));
                this.record_once(&completion);
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.record_once(&BodyCompletion::Succeeded);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ObservedUpstreamBody {
    fn drop(&mut self) {
        self.record_once(&BodyCompletion::DownstreamDropped);
    }
}

fn build_upstream_url(base_url: &str, uri: &Uri) -> Result<Url, ProxyError> {
    validate_openai_path(uri.path())?;
    validate_upstream_base_url(base_url)
        .map_err(|error| ProxyError::invalid_upstream_url(base_url, error.to_string()))?;

    let mut base = Url::parse(base_url)
        .map_err(|error| ProxyError::invalid_upstream_url(base_url, error.to_string()))?;
    let path = upstream_path(base.path(), uri.path());
    base.set_path("");
    base.set_query(None);
    base.set_fragment(None);

    let mut url = base.as_str().trim_end_matches('/').to_owned();
    url.push_str(&path);
    if let Some(query) = uri.query() {
        url.push('?');
        url.push_str(query);
    }

    Url::parse(&url).map_err(|error| ProxyError::invalid_upstream_url(base_url, error.to_string()))
}

fn upstream_path(base_path: &str, downstream_path: &str) -> String {
    let trimmed_base = base_path.trim_end_matches('/');
    if trimmed_base.is_empty() {
        return downstream_path.to_owned();
    }

    if trimmed_base == "/v1" {
        if downstream_path == "/v1" {
            return String::from("/v1");
        }
        if let Some(suffix) = downstream_path.strip_prefix("/v1/") {
            return format!("/v1/{suffix}");
        }
    }

    format!("{trimmed_base}{downstream_path}")
}

fn downstream_response(
    status: reqwest::StatusCode,
    upstream_headers: &HeaderMap,
    body: Body,
) -> Response<Body> {
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(body);
    *response.status_mut() = status;
    copy_response_headers(upstream_headers, response.headers_mut());
    response
}

fn forwarded_request_headers(headers: &HeaderMap) -> HeaderMap {
    let connection_tokens = connection_header_tokens(headers);
    let mut forwarded = HeaderMap::new();
    for (name, value) in headers {
        if should_skip_request_header(name, &connection_tokens) {
            continue;
        }
        forwarded.append(name.clone(), value.clone());
    }
    forwarded
}

fn copy_response_headers(source: &HeaderMap, target: &mut HeaderMap) {
    let connection_tokens = connection_header_tokens(source);
    for (name, value) in source {
        if should_skip_response_header(name, &connection_tokens) {
            continue;
        }
        target.append(name.clone(), value.clone());
    }
}

fn should_skip_request_header(name: &HeaderName, connection_tokens: &HashSet<HeaderName>) -> bool {
    name == HOST
        || name == CONTENT_LENGTH
        || is_hop_by_hop_header(name)
        || connection_tokens.contains(name)
}

fn should_skip_response_header(name: &HeaderName, connection_tokens: &HashSet<HeaderName>) -> bool {
    name == CONTENT_LENGTH || is_hop_by_hop_header(name) || connection_tokens.contains(name)
}

fn is_hop_by_hop_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn connection_header_tokens(headers: &HeaderMap) -> HashSet<HeaderName> {
    headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|token| HeaderName::from_bytes(token.trim().as_bytes()).ok())
        .collect()
}

fn request_metadata(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    body_len: usize,
    shielding_enabled: bool,
) -> BTreeMap<String, String> {
    base_request_metadata(
        method,
        uri,
        headers,
        body_len.to_string(),
        Some(shielding_enabled),
    )
}

fn pre_upstream_request_metadata(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    shielding_enabled: Option<bool>,
) -> BTreeMap<String, String> {
    base_request_metadata(
        method,
        uri,
        headers,
        request_body_bytes_hint(headers),
        shielding_enabled,
    )
}

fn base_request_metadata(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    request_body_bytes: String,
    shielding_enabled: Option<bool>,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([
        (String::from("method"), method.as_str().to_owned()),
        (String::from("path"), uri.path().to_owned()),
        (
            String::from("query_present"),
            uri.query().is_some().to_string(),
        ),
        (String::from("request_body_bytes"), request_body_bytes),
        (
            String::from("shielding_config_enabled"),
            shielding_enabled
                .map_or_else(|| String::from("unknown"), |enabled| enabled.to_string()),
        ),
        (
            String::from("policy_transform_applied"),
            String::from("false"),
        ),
    ]);
    copy_selected_header_metadata(&mut metadata, headers, "request");
    metadata
}

fn request_body_bytes_hint(headers: &HeaderMap) -> String {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map_or_else(|| String::from("unknown"), |bytes| bytes.to_string())
}

fn attempt_request_metadata(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([
        (String::from("method"), method.as_str().to_owned()),
        (String::from("path"), uri.path().to_owned()),
        (
            String::from("query_present"),
            uri.query().is_some().to_string(),
        ),
        (String::from("attempt_number"), String::from("1")),
    ]);
    copy_selected_header_metadata(&mut metadata, headers, "upstream_request");
    metadata
}

fn response_metadata(
    status: reqwest::StatusCode,
    headers: &HeaderMap,
    body_len: u64,
    latency_ms: u64,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([
        (
            String::from("http_status_success"),
            status.is_success().to_string(),
        ),
        (String::from("response_body_bytes"), body_len.to_string()),
        (String::from("latency_ms"), latency_ms.to_string()),
    ]);
    copy_selected_header_metadata(&mut metadata, headers, "response");
    metadata
}

fn failed_response_metadata(
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    error_type: &str,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (String::from("error_type"), error_type.to_owned()),
        (
            String::from("latency_ms"),
            finished_at_unix_ms
                .saturating_sub(started_at_unix_ms)
                .to_string(),
        ),
        (
            String::from("upstream_response_received"),
            String::from("false"),
        ),
    ])
}

fn copy_selected_header_metadata(
    metadata: &mut BTreeMap<String, String>,
    headers: &HeaderMap,
    prefix: &str,
) {
    for header in [
        CONTENT_TYPE,
        ACCEPT,
        AUTHORIZATION,
        HeaderName::from_static("x-api-key"),
        HeaderName::from_static("user-agent"),
        HeaderName::from_static("x-request-id"),
        HeaderName::from_static("server"),
    ] {
        if let Some(value) = headers.get(&header) {
            metadata.insert(
                format!("{prefix}_header_{}", header.as_str()),
                header_value(value),
            );
        }
    }
}

fn header_value(value: &HeaderValue) -> String {
    value
        .to_str()
        .map_or_else(|_error| HEADER_VALUE_NOT_UTF8.to_owned(), str::to_owned)
}

fn extract_model_id(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("model")
                .and_then(|model| model.as_str())
                .map(str::to_owned)
        })
}

fn downstream_mode_from_headers(headers: &HeaderMap) -> DownstreamMode {
    if is_event_stream(headers) {
        DownstreamMode::Streaming
    } else {
        DownstreamMode::NonStreamJson
    }
}

fn upstream_mode_from_headers(headers: &HeaderMap) -> UpstreamMode {
    if is_event_stream(headers) {
        UpstreamMode::Streaming
    } else {
        UpstreamMode::NonStreamJson
    }
}

fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"))
}

fn validate_openai_path(path: &str) -> Result<(), OpenAiPathError> {
    if path != "/v1" && !path.starts_with("/v1/") {
        return Err(OpenAiPathError::OutsideOpenAiScope);
    }

    if path.split('/').any(path_segment_decodes_to_dot_segment) {
        return Err(OpenAiPathError::DotSegment);
    }

    Ok(())
}

fn path_segment_decodes_to_dot_segment(segment: &str) -> bool {
    let mut decoded = [0_u8; 2];
    let mut decoded_len = 0_usize;
    let bytes = segment.as_bytes();
    let mut index = 0_usize;

    while index < bytes.len() {
        let byte = if let Some((decoded_byte, next_index)) = percent_encoded_byte(bytes, index) {
            index = next_index;
            decoded_byte
        } else {
            let byte = bytes[index];
            index += 1;
            byte
        };

        if decoded_len == decoded.len() {
            return false;
        }
        decoded[decoded_len] = byte;
        decoded_len += 1;
    }

    matches!(&decoded[..decoded_len], b"." | b"..")
}

fn percent_encoded_byte(bytes: &[u8], index: usize) -> Option<(u8, usize)> {
    if bytes.get(index).copied() != Some(b'%') {
        return None;
    }

    let high = hex_value(*bytes.get(index + 1)?)?;
    let low = hex_value(*bytes.get(index + 2)?)?;
    Some(((high << 4) | low, index + 3))
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn config_shielding_enabled(config: &ConfigHandle) -> Option<bool> {
    config
        .snapshot()
        .ok()
        .map(|snapshot| snapshot.shielding.enabled)
}

fn failed_attempt_record(
    attempt_id: AttemptId,
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    error_type: &str,
    error_reason: &str,
    request_metadata: BTreeMap<String, String>,
) -> AttemptRecord {
    AttemptRecord {
        attempt_id,
        request_id,
        attempt_number: 1,
        started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        upstream_mode: UpstreamMode::NotApplicable,
        status: AttemptStatus::Failed,
        http_status: None,
        error_reason: Some(format!("{error_type}: {error_reason}")),
        retry_reason: None,
        abort_reason: None,
        request_metadata,
        response_metadata: failed_response_metadata(
            started_at_unix_ms,
            finished_at_unix_ms,
            error_type,
        ),
        raw_payloads: RawPayloads::default(),
    }
}

struct FailedRequestRecord<'attempt> {
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    http_status: u16,
    error_type: &'static str,
    error_reason: String,
    request_metadata: BTreeMap<String, String>,
    attempt: Option<&'attempt AttemptRecord>,
}

fn record_failed_request(store: &ObservabilityStore, failure: FailedRequestRecord<'_>) {
    let request_record = RequestRecord {
        request_id: failure.request_id,
        started_at_unix_ms: failure.started_at_unix_ms,
        finished_at_unix_ms: Some(failure.finished_at_unix_ms),
        downstream_mode: DownstreamMode::NonStreamJson,
        upstream_mode: UpstreamMode::NotApplicable,
        model_id: None,
        input_fingerprint: None,
        status: RequestStatus::Failed,
        http_status: Some(failure.http_status),
        error_reason: Some(format!("{}: {}", failure.error_type, failure.error_reason)),
        abort_reason: None,
        request_metadata: failure.request_metadata,
        response_metadata: failed_response_metadata(
            failure.started_at_unix_ms,
            failure.finished_at_unix_ms,
            failure.error_type,
        ),
        raw_payloads: RawPayloads::default(),
    };
    record_observability(store, &request_record, failure.attempt);
}

fn record_observability(
    store: &ObservabilityStore,
    request: &RequestRecord,
    attempt: Option<&AttemptRecord>,
) {
    if let Err(error) = store.record_request(request) {
        eprintln!("failed to write request observability: {error}");
        return;
    }
    if let Some(attempt) = attempt {
        if let Err(error) = store.record_attempt(attempt) {
            eprintln!("failed to write attempt observability: {error}");
        }
    }
}

fn text_response(status: StatusCode, text: String) -> Response<Body> {
    let mut response = Response::new(Body::from(text));
    *response.status_mut() = status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}

fn proxy_error_response(status: StatusCode, error_type: &str, message: &str) -> Response<Body> {
    let mut response = Response::new(Body::from(
        json!({
            "error": {
                "type": error_type,
                "message": message,
            }
        })
        .to_string(),
    ));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn unix_time_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[derive(Debug, Error)]
enum ProxyError {
    #[error("failed to read request body within proxy limit: {reason}")]
    RequestBody {
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("failed to read current config: {reason}")]
    ConfigSnapshot {
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("invalid upstream base URL {display_url}: {reason}")]
    InvalidUpstreamUrl {
        display_url: String,
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("{0}")]
    InvalidRequestPath(#[from] OpenAiPathError),
    #[error("invalid HTTP method: {reason}")]
    InvalidMethod {
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("upstream request failed: {failure}")]
    UpstreamTransport {
        failure: ReqwestFailureKind,
        observability: Option<Box<FailedUpstreamObservability>>,
    },
    #[error("failed to read upstream response body within proxy limit: {reason}")]
    UpstreamBody {
        reason: String,
        observability: Option<Box<FailedUpstreamObservability>>,
    },
}

impl ProxyError {
    fn request_body(reason: String) -> Self {
        Self::RequestBody {
            reason,
            request_metadata: None,
        }
    }

    fn config_snapshot(reason: String) -> Self {
        Self::ConfigSnapshot {
            reason,
            request_metadata: None,
        }
    }

    fn invalid_upstream_url(base_url: &str, reason: String) -> Self {
        Self::InvalidUpstreamUrl {
            display_url: redact_upstream_base_url(base_url),
            reason,
            request_metadata: None,
        }
    }

    fn invalid_method(reason: String) -> Self {
        Self::InvalidMethod {
            reason,
            request_metadata: None,
        }
    }

    fn upstream_body(reason: String) -> Self {
        Self::UpstreamBody {
            reason,
            observability: None,
        }
    }

    const fn status(&self) -> StatusCode {
        match self {
            Self::RequestBody { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidMethod { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidRequestPath(error) => error.status(),
            Self::UpstreamTransport { .. } | Self::UpstreamBody { .. } => StatusCode::BAD_GATEWAY,
        }
    }

    const fn error_type(&self) -> &'static str {
        match self {
            Self::RequestBody { .. } => "request_body_error",
            Self::ConfigSnapshot { .. } => "config_snapshot_failed",
            Self::InvalidUpstreamUrl { .. } => "invalid_upstream_url",
            Self::InvalidRequestPath(error) => error.error_type(),
            Self::InvalidMethod { .. } => "invalid_method",
            Self::UpstreamTransport { .. } => "upstream_transport_error",
            Self::UpstreamBody { .. } => "upstream_body_error",
        }
    }

    fn request_metadata(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::RequestBody {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::ConfigSnapshot {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::InvalidUpstreamUrl {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::InvalidMethod {
                request_metadata: Some(request_metadata),
                ..
            } => Some(request_metadata),
            Self::UpstreamTransport {
                observability: Some(observability),
                ..
            }
            | Self::UpstreamBody {
                observability: Some(observability),
                ..
            } => Some(&observability.request_metadata),
            Self::RequestBody {
                request_metadata: None,
                ..
            }
            | Self::ConfigSnapshot {
                request_metadata: None,
                ..
            }
            | Self::InvalidUpstreamUrl {
                request_metadata: None,
                ..
            }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod {
                request_metadata: None,
                ..
            }
            | Self::UpstreamTransport {
                observability: None,
                ..
            }
            | Self::UpstreamBody {
                observability: None,
                ..
            } => None,
        }
    }

    fn attempt_record(&self) -> Option<&AttemptRecord> {
        match self {
            Self::UpstreamTransport {
                observability: Some(observability),
                ..
            }
            | Self::UpstreamBody {
                observability: Some(observability),
                ..
            } => Some(&observability.attempt_record),
            Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::UpstreamTransport {
                observability: None,
                ..
            }
            | Self::UpstreamBody {
                observability: None,
                ..
            } => None,
        }
    }

    fn with_request_metadata(self, request_metadata: BTreeMap<String, String>) -> Self {
        match self {
            Self::RequestBody { reason, .. } => Self::RequestBody {
                reason,
                request_metadata: Some(request_metadata),
            },
            Self::ConfigSnapshot { reason, .. } => Self::ConfigSnapshot {
                reason,
                request_metadata: Some(request_metadata),
            },
            Self::InvalidUpstreamUrl {
                display_url,
                reason,
                ..
            } => Self::InvalidUpstreamUrl {
                display_url,
                reason,
                request_metadata: Some(request_metadata),
            },
            Self::InvalidMethod { reason, .. } => Self::InvalidMethod {
                reason,
                request_metadata: Some(request_metadata),
            },
            error @ (Self::InvalidRequestPath(_)
            | Self::UpstreamTransport { .. }
            | Self::UpstreamBody { .. }) => error,
        }
    }

    fn with_observability(
        self,
        request_metadata: BTreeMap<String, String>,
        attempt_record: AttemptRecord,
    ) -> Self {
        match self {
            Self::UpstreamTransport { failure, .. } => Self::UpstreamTransport {
                failure,
                observability: Some(Box::new(FailedUpstreamObservability {
                    request_metadata,
                    attempt_record,
                })),
            },
            Self::UpstreamBody { reason, .. } => Self::UpstreamBody {
                reason,
                observability: Some(Box::new(FailedUpstreamObservability {
                    request_metadata,
                    attempt_record,
                })),
            },
            error @ (Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }) => error,
        }
    }
}

#[derive(Debug)]
struct FailedUpstreamObservability {
    request_metadata: BTreeMap<String, String>,
    attempt_record: AttemptRecord,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("proxy in-flight request limit exceeded: max_in_flight_requests={max_in_flight_requests}")]
struct InFlightLimitExceeded {
    max_in_flight_requests: usize,
}

impl InFlightLimitExceeded {
    const fn status() -> StatusCode {
        StatusCode::SERVICE_UNAVAILABLE
    }

    const fn error_type() -> &'static str {
        "proxy_in_flight_limit_exceeded"
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
enum OpenAiPathError {
    #[error("only /v1 OpenAI-compatible endpoints are proxied")]
    OutsideOpenAiScope,
    #[error("OpenAI-compatible request path contains a raw or percent-encoded dot segment")]
    DotSegment,
}

impl OpenAiPathError {
    const fn status(self) -> StatusCode {
        match self {
            Self::OutsideOpenAiScope => StatusCode::NOT_FOUND,
            Self::DotSegment => StatusCode::BAD_REQUEST,
        }
    }

    const fn error_type(self) -> &'static str {
        match self {
            Self::OutsideOpenAiScope => "not_found",
            Self::DotSegment => "invalid_request_path",
        }
    }
}

#[cfg(test)]
mod tests;
