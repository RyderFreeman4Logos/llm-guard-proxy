use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
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
    if !is_openai_path(request.uri().path()) {
        return proxy_error_response(
            StatusCode::NOT_FOUND,
            "not_found",
            "only /v1 OpenAI-compatible endpoints are proxied",
        );
    }

    let request_id = RequestId::generate();
    let started_at_unix_ms = unix_time_millis();
    match forward_openai_request(&state, &request_id, started_at_unix_ms, request).await {
        Ok(response) => response,
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let response =
                proxy_error_response(error.status(), error.error_type(), &error.to_string());
            record_failed_request(
                &state.store,
                request_id,
                started_at_unix_ms,
                finished_at_unix_ms,
                error.status().as_u16(),
                error.error_type(),
                &error.to_string(),
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

    let upstream_response = send_upstream_request(
        &state.client,
        method.clone(),
        upstream_url,
        &downstream_headers,
        body.clone(),
    )
    .await?;
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let upstream_mode = upstream_mode_from_headers(&upstream_headers);
    let response_body = upstream_response
        .bytes()
        .await
        .map_err(ProxyError::UpstreamTransport)?;
    let finished_at_unix_ms = unix_time_millis();

    let response = downstream_response(upstream_status, &upstream_headers, response_body.clone());
    let request_metadata = request_metadata(
        &method,
        &uri,
        &downstream_headers,
        body.len(),
        config.shielding.enabled,
    );
    let response_metadata = response_metadata(
        upstream_status,
        &upstream_headers,
        response_body.len(),
        finished_at_unix_ms.saturating_sub(started_at_unix_ms),
    );
    let attempt_request_metadata = attempt_request_metadata(&method, &uri, &downstream_headers);
    let attempt_response_metadata = response_metadata.clone();
    let request_record = RequestRecord {
        request_id: request_id.clone(),
        started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        downstream_mode: downstream_mode_from_headers(&upstream_headers),
        upstream_mode,
        model_id,
        input_fingerprint: None,
        status: RequestStatus::Succeeded,
        http_status: Some(upstream_status.as_u16()),
        error_reason: None,
        abort_reason: None,
        request_metadata,
        response_metadata,
        raw_payloads: RawPayloads::default(),
    };
    let attempt_record = AttemptRecord {
        attempt_id,
        request_id: request_id.clone(),
        attempt_number: 1,
        started_at_unix_ms: attempt_started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        upstream_mode,
        status: AttemptStatus::Succeeded,
        http_status: Some(upstream_status.as_u16()),
        error_reason: None,
        retry_reason: None,
        abort_reason: None,
        request_metadata: attempt_request_metadata,
        response_metadata: attempt_response_metadata,
        raw_payloads: RawPayloads::default(),
    };
    record_observability(&state.store, &request_record, Some(&attempt_record));

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
        .map_err(ProxyError::UpstreamTransport)
}

fn build_upstream_url(base_url: &str, uri: &Uri) -> Result<Url, ProxyError> {
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
    body: Bytes,
) -> Response<Body> {
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(Body::from(body));
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
    let mut metadata = BTreeMap::from([
        (String::from("method"), method.as_str().to_owned()),
        (String::from("path"), uri.path().to_owned()),
        (
            String::from("query_present"),
            uri.query().is_some().to_string(),
        ),
        (String::from("request_body_bytes"), body_len.to_string()),
        (
            String::from("shielding_config_enabled"),
            shielding_enabled.to_string(),
        ),
        (
            String::from("policy_transform_applied"),
            String::from("false"),
        ),
    ]);
    copy_selected_header_metadata(&mut metadata, headers, "request");
    metadata
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
    body_len: usize,
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

fn is_openai_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

fn record_failed_request(
    store: &ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    http_status: u16,
    error_type: &str,
    error_reason: &str,
) {
    let request_record = RequestRecord {
        request_id,
        started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        downstream_mode: DownstreamMode::NonStreamJson,
        upstream_mode: UpstreamMode::NotApplicable,
        model_id: None,
        input_fingerprint: None,
        status: RequestStatus::Failed,
        http_status: Some(http_status),
        error_reason: Some(format!("{error_type}: {error_reason}")),
        abort_reason: None,
        request_metadata: BTreeMap::from([(String::from("proxy_error"), error_type.to_owned())]),
        response_metadata: BTreeMap::from([(
            String::from("latency_ms"),
            finished_at_unix_ms
                .saturating_sub(started_at_unix_ms)
                .to_string(),
        )]),
        raw_payloads: RawPayloads::default(),
    };
    record_observability(store, &request_record, None);
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
    #[error("invalid HTTP method: {0}")]
    InvalidMethod(String),
    #[error("upstream request failed: {0}")]
    UpstreamTransport(#[source] reqwest::Error),
}

impl ProxyError {
    const fn status(&self) -> StatusCode {
        match self {
            Self::RequestBody(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ConfigSnapshot(_) | Self::InvalidUpstreamUrl(_) | Self::InvalidMethod(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::UpstreamTransport(_) => StatusCode::BAD_GATEWAY,
        }
    }

    const fn error_type(&self) -> &'static str {
        match self {
            Self::RequestBody(_) => "request_body_error",
            Self::ConfigSnapshot(_) => "config_snapshot_failed",
            Self::InvalidUpstreamUrl(_) => "invalid_upstream_url",
            Self::InvalidMethod(_) => "invalid_method",
            Self::UpstreamTransport(_) => "upstream_transport_error",
        }
    }
}

#[cfg(test)]
mod tests;
