use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    pin::Pin,
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
use futures_util::Stream;
use llm_guard_proxy_core::{
    AppConfig, AttemptId, AttemptRecord, AttemptStatus, ConfigHandle, DownstreamMode, Health,
    LICENSE, ObservabilityStore, RawPayloads, RequestId, RequestRecord, RequestStatus,
    SERVICE_NAME, UpstreamMode,
};
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const UPSTREAM_REQUEST_TIMEOUT_SECS: u64 = 120;
const HEADER_VALUE_NOT_UTF8: &str = "[non-utf8]";

/// Shared HTTP proxy state.
#[derive(Clone, Debug)]
pub(crate) struct ProxyState {
    config: ConfigHandle,
    config_path: PathBuf,
    store: ObservabilityStore,
    client: Client,
}

impl ProxyState {
    /// Builds cloneable proxy state for axum handlers.
    #[must_use]
    pub(crate) fn new(
        config: ConfigHandle,
        config_path: PathBuf,
        store: ObservabilityStore,
        client: Client,
    ) -> Self {
        Self {
            config,
            config_path,
            store,
            client,
        }
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

    match forward_openai_request(&state, &request_id, started_at_unix_ms, request).await {
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
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = request.into_parts();
    let method = parts.method;
    let uri = parts.uri;
    let downstream_headers = parts.headers;
    let body = read_body_bytes(body).await?;
    let config = state
        .config
        .snapshot()
        .map_err(|error| ProxyError::ConfigSnapshot(error.to_string()))?;
    let upstream_url = build_upstream_url(&config.upstream.base_url, &uri)?;
    let model_id = extract_model_id(&body);
    let attempt_id = AttemptId::for_request(request_id, 1);
    let attempt_started_at_unix_ms = unix_time_millis();
    let request_metadata = request_metadata(
        &method,
        &uri,
        &downstream_headers,
        body.len(),
        config.shielding.enabled,
    );
    let attempt_request_metadata = attempt_request_metadata(&method, &uri, &downstream_headers);
    let upstream_response = match send_upstream_request(
        &state.client,
        method.clone(),
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
    let observer = ForwardedBodyObserver {
        store: state.store.clone(),
        request_id: request_id.clone(),
        started_at_unix_ms,
        attempt_id,
        attempt_started_at_unix_ms,
        downstream_mode: downstream_mode_from_headers(&upstream_headers),
        upstream_mode,
        model_id,
        upstream_status,
        upstream_headers: upstream_headers.clone(),
        request_metadata,
        attempt_request_metadata,
    };
    let response_body = ObservedUpstreamBody::new(upstream_response.bytes_stream(), observer);
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
        .map_err(|error| ProxyError::RequestBody(error.to_string()))
}

async fn send_upstream_request(
    client: &Client,
    method: Method,
    upstream_url: Url,
    downstream_headers: &HeaderMap,
    body: Bytes,
) -> Result<reqwest::Response, ProxyError> {
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|error| ProxyError::InvalidMethod(error.to_string()))?;
    let headers = forwarded_request_headers(downstream_headers);
    client
        .request(reqwest_method, upstream_url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|source| ProxyError::UpstreamTransport {
            source,
            observability: None,
        })
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
    bytes_seen: u64,
}

impl ObservedUpstreamBody {
    fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            observer: Some(observer),
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
                let completion = BodyCompletion::UpstreamStreamError(error.to_string());
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

    let mut base =
        Url::parse(base_url).map_err(|error| ProxyError::InvalidUpstreamUrl(error.to_string()))?;
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

    Url::parse(&url).map_err(|error| ProxyError::InvalidUpstreamUrl(error.to_string()))
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
    #[error("failed to read request body within proxy limit: {0}")]
    RequestBody(String),
    #[error("failed to read current config: {0}")]
    ConfigSnapshot(String),
    #[error("invalid upstream base URL: {0}")]
    InvalidUpstreamUrl(String),
    #[error("{0}")]
    InvalidRequestPath(#[from] OpenAiPathError),
    #[error("invalid HTTP method: {0}")]
    InvalidMethod(String),
    #[error("upstream request failed: {source}")]
    UpstreamTransport {
        #[source]
        source: reqwest::Error,
        observability: Option<Box<FailedUpstreamObservability>>,
    },
}

impl ProxyError {
    const fn status(&self) -> StatusCode {
        match self {
            Self::RequestBody(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ConfigSnapshot(_) | Self::InvalidUpstreamUrl(_) | Self::InvalidMethod(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::InvalidRequestPath(error) => error.status(),
            Self::UpstreamTransport { .. } => StatusCode::BAD_GATEWAY,
        }
    }

    const fn error_type(&self) -> &'static str {
        match self {
            Self::RequestBody(_) => "request_body_error",
            Self::ConfigSnapshot(_) => "config_snapshot_failed",
            Self::InvalidUpstreamUrl(_) => "invalid_upstream_url",
            Self::InvalidRequestPath(error) => error.error_type(),
            Self::InvalidMethod(_) => "invalid_method",
            Self::UpstreamTransport { .. } => "upstream_transport_error",
        }
    }

    fn request_metadata(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::UpstreamTransport {
                observability: Some(observability),
                ..
            } => Some(&observability.request_metadata),
            Self::RequestBody(_)
            | Self::ConfigSnapshot(_)
            | Self::InvalidUpstreamUrl(_)
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod(_)
            | Self::UpstreamTransport {
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
            } => Some(&observability.attempt_record),
            Self::RequestBody(_)
            | Self::ConfigSnapshot(_)
            | Self::InvalidUpstreamUrl(_)
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod(_)
            | Self::UpstreamTransport {
                observability: None,
                ..
            } => None,
        }
    }

    fn with_observability(
        self,
        request_metadata: BTreeMap<String, String>,
        attempt_record: AttemptRecord,
    ) -> Self {
        match self {
            Self::UpstreamTransport { source, .. } => Self::UpstreamTransport {
                source,
                observability: Some(Box::new(FailedUpstreamObservability {
                    request_metadata,
                    attempt_record,
                })),
            },
            error @ (Self::RequestBody(_)
            | Self::ConfigSnapshot(_)
            | Self::InvalidUpstreamUrl(_)
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod(_)) => error,
        }
    }
}

#[derive(Debug)]
struct FailedUpstreamObservability {
    request_metadata: BTreeMap<String, String>,
    attempt_record: AttemptRecord,
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
