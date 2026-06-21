use std::{
    collections::{BTreeMap, HashMap, HashSet, hash_map::DefaultHasher},
    convert::Infallible,
    fmt,
    future::Future,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
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
    HeartbeatMode, LICENSE, MetadataConfig, ObservabilityStore, RawPayloads, RequestId,
    RequestRecord, RequestStatus, RetryConfig, SERVICE_NAME, UpstreamMode,
    redact_upstream_base_url, validate_upstream_base_url,
};
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant, Interval, MissedTickBehavior};

mod model_metadata;
mod shielded_chat;

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_REPEAT_FINGERPRINT_ENTRIES: usize = 1024;
const UPSTREAM_REQUEST_TIMEOUT_SECS: u64 = 120;
const HEADER_VALUE_NOT_UTF8: &str = "[non-utf8]";
const HEADER_VALUE_REDACTED: &str = "[redacted]";

/// Shared HTTP proxy state.
#[derive(Clone, Debug)]
pub(crate) struct ProxyState {
    config: ConfigHandle,
    config_path: PathBuf,
    store: ObservabilityStore,
    client: Client,
    in_flight_requests: Arc<Semaphore>,
    repeat_inputs: Arc<RepeatInputCache>,
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
            repeat_inputs: Arc::new(RepeatInputCache::default()),
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

#[derive(Debug, Default)]
struct RepeatInputCache {
    entries: Mutex<HashMap<String, RepeatInputEntry>>,
}

#[derive(Clone, Copy, Debug)]
struct RepeatInputEntry {
    count: u32,
    last_seen_unix_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RepeatInputObservation {
    repeated: bool,
    prior_count: u32,
}

impl RepeatInputCache {
    fn observe(
        &self,
        fingerprint: &str,
        now_unix_ms: u64,
        window_secs: u64,
        max_repeated_inputs: u32,
    ) -> RepeatInputObservation {
        let window_ms = window_secs.saturating_mul(1_000);
        let mut entries = repeat_input_entries(&self.entries);
        entries.retain(|_fingerprint, entry| {
            now_unix_ms.saturating_sub(entry.last_seen_unix_ms) <= window_ms
        });

        let observation = {
            let entry = entries
                .entry(fingerprint.to_owned())
                .or_insert(RepeatInputEntry {
                    count: 0,
                    last_seen_unix_ms: now_unix_ms,
                });
            let prior_count = if now_unix_ms.saturating_sub(entry.last_seen_unix_ms) <= window_ms {
                entry.count
            } else {
                0
            };
            entry.count = prior_count.saturating_add(1);
            entry.last_seen_unix_ms = now_unix_ms;

            RepeatInputObservation {
                repeated: prior_count >= max_repeated_inputs,
                prior_count,
            }
        };

        prune_repeat_input_entries(&mut entries);
        observation
    }
}

fn repeat_input_entries(
    entries: &Mutex<HashMap<String, RepeatInputEntry>>,
) -> MutexGuard<'_, HashMap<String, RepeatInputEntry>> {
    match entries.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn prune_repeat_input_entries(entries: &mut HashMap<String, RepeatInputEntry>) {
    if entries.len() <= MAX_REPEAT_FINGERPRINT_ENTRIES {
        return;
    }

    let remove_count = entries.len().saturating_sub(MAX_REPEAT_FINGERPRINT_ENTRIES);
    let mut oldest_entries = entries
        .iter()
        .map(|(fingerprint, entry)| (fingerprint.clone(), entry.last_seen_unix_ms))
        .collect::<Vec<_>>();
    oldest_entries.sort_by_key(|(_fingerprint, last_seen_unix_ms)| *last_seen_unix_ms);
    for (fingerprint, _last_seen_unix_ms) in oldest_entries.into_iter().take(remove_count) {
        entries.remove(&fingerprint);
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
    let mut request_metadata = request_metadata(
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
    let shielded_chat_plan = plan_shielded_chat(state, &config, &method, &uri, &body);
    add_shielded_request_metadata(
        &mut request_metadata,
        shielded_chat_plan.intercepted,
        &shielded_chat_plan.liveness,
        &shielded_chat_plan.thinking_metadata,
    );
    let retry_policy = ShieldedRetryPolicy::from_config(&config.retry);
    if shielded_chat_plan.intercepted {
        add_retry_request_metadata(&mut request_metadata, retry_policy);
        return forward_shielded_chat_with_retries(
            ShieldedRetryRuntime {
                client: state.client.clone(),
                method: reqwest_method,
                upstream_url,
                downstream_method: method,
                downstream_uri: uri,
                downstream_headers,
                upstream_body: shielded_chat_plan.upstream_body,
                store: state.store.clone(),
                request_id: request_id.clone(),
                started_at_unix_ms,
                model_id,
                request_metadata,
                liveness: shielded_chat_plan.liveness,
                thinking_metadata: shielded_chat_plan.thinking_metadata,
                loop_context: shielded_chat_plan.loop_context,
                retry_policy,
            },
            in_flight_permit,
        )
        .await;
    }
    forward_generic_openai_request(GenericForwardContext {
        state,
        config: &config,
        method,
        uri,
        downstream_headers,
        reqwest_method,
        upstream_url,
        upstream_body: shielded_chat_plan.upstream_body,
        liveness: shielded_chat_plan.liveness,
        thinking_metadata: shielded_chat_plan.thinking_metadata,
        request_id,
        started_at_unix_ms,
        model_id,
        request_metadata,
        in_flight_permit,
    })
    .await
}

struct GenericForwardContext<'request> {
    state: &'request ProxyState,
    config: &'request AppConfig,
    method: Method,
    uri: Uri,
    downstream_headers: HeaderMap,
    reqwest_method: reqwest::Method,
    upstream_url: Url,
    upstream_body: Bytes,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    request_id: &'request RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    request_metadata: BTreeMap<String, String>,
    in_flight_permit: OwnedSemaphorePermit,
}

async fn forward_generic_openai_request(
    context: GenericForwardContext<'_>,
) -> Result<Response<Body>, ProxyError> {
    let attempt_id = AttemptId::for_request(context.request_id, 1);
    let attempt_started_at_unix_ms = unix_time_millis();
    let mut attempt_request_metadata =
        attempt_request_metadata(&context.method, &context.uri, &context.downstream_headers);
    add_shielded_request_metadata(
        &mut attempt_request_metadata,
        false,
        &context.liveness,
        &context.thinking_metadata,
    );
    let upstream_response = send_first_upstream_attempt(UpstreamAttemptContext {
        client: &context.state.client,
        method: context.reqwest_method,
        upstream_url: context.upstream_url,
        downstream_headers: &context.downstream_headers,
        upstream_body: context.upstream_body,
        attempt_id: attempt_id.clone(),
        request_id: context.request_id,
        attempt_started_at_unix_ms,
        request_metadata: &context.request_metadata,
        attempt_request_metadata: &attempt_request_metadata,
    })
    .await?;
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let response_parts = ForwardedResponseParts {
        store: context.state.store.clone(),
        request_id: context.request_id.clone(),
        started_at_unix_ms: context.started_at_unix_ms,
        attempt_id,
        attempt_started_at_unix_ms,
        upstream_mode: upstream_mode_from_headers(&upstream_headers),
        model_id: context.model_id,
        input_fingerprint: context.liveness.input_fingerprint.clone(),
        upstream_status,
        upstream_headers: upstream_headers.clone(),
        request_metadata: context.request_metadata,
        attempt_request_metadata,
    };
    forward_upstream_response(
        ResponseDispatch {
            method: &context.method,
            uri: &context.uri,
            config: context.config,
        },
        response_parts,
        upstream_response,
        context.in_flight_permit,
    )
    .await
}

struct UpstreamAttemptContext<'request> {
    client: &'request Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: &'request HeaderMap,
    upstream_body: Bytes,
    attempt_id: AttemptId,
    request_id: &'request RequestId,
    attempt_started_at_unix_ms: u64,
    request_metadata: &'request BTreeMap<String, String>,
    attempt_request_metadata: &'request BTreeMap<String, String>,
}

async fn send_first_upstream_attempt(
    context: UpstreamAttemptContext<'_>,
) -> Result<reqwest::Response, ProxyError> {
    match send_upstream_request(
        context.client,
        context.method,
        context.upstream_url,
        context.downstream_headers,
        context.upstream_body,
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_reason = error.to_string();
            let attempt_record = failed_attempt_record(FailedAttemptRecordInput {
                attempt_id: context.attempt_id,
                request_id: context.request_id.clone(),
                started_at_unix_ms: context.attempt_started_at_unix_ms,
                finished_at_unix_ms,
                error_type: error.error_type(),
                error_reason: &error_reason,
                request_metadata: context.attempt_request_metadata.clone(),
                extra_response_metadata: BTreeMap::new(),
            });
            Err(error.with_observability(context.request_metadata.clone(), attempt_record))
        }
    }
}

struct ResponseDispatch<'request> {
    method: &'request Method,
    uri: &'request Uri,
    config: &'request AppConfig,
}

struct ShieldedChatPlan {
    upstream_body: Bytes,
    intercepted: bool,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    loop_context: shielded_chat::LoopInspectionContext,
}

fn plan_shielded_chat(
    state: &ProxyState,
    config: &AppConfig,
    method: &Method,
    uri: &Uri,
    body: &Bytes,
) -> ShieldedChatPlan {
    let request = if should_intercept_non_stream_chat(method, uri, config) {
        shielded_chat::prepare_non_stream_request(body, &config.thinking)
    } else {
        None
    };
    let upstream_body = request.as_ref().map_or_else(
        || body.clone(),
        shielded_chat::PreparedChatRequest::upstream_body,
    );
    let thinking_metadata = request
        .as_ref()
        .map_or_else(BTreeMap::new, |request| request.thinking_metadata().clone());
    let intercepted = request.is_some();
    let liveness = select_shielded_liveness(state, config, body, intercepted, unix_time_millis());
    let loop_context = if intercepted {
        shielded_chat::LoopInspectionContext::from_request_body(&config.loop_guard, body)
    } else {
        shielded_chat::LoopInspectionContext::empty(&config.loop_guard)
    };

    ShieldedChatPlan {
        upstream_body,
        intercepted,
        liveness,
        thinking_metadata,
        loop_context,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShieldedLivenessSelection {
    mode: ShieldedLivenessMode,
    heartbeat_interval_secs: u64,
    input_fingerprint: Option<String>,
    repeat_observation: RepeatInputObservation,
    repeat_window_secs: u64,
    repeat_max_inputs: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShieldedLivenessMode {
    Sse,
    JsonWhitespace,
    Disabled,
}

impl ShieldedLivenessMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Sse => "sse",
            Self::JsonWhitespace => "json-whitespace",
            Self::Disabled => "disabled",
        }
    }

    const fn downstream_mode(self) -> DownstreamMode {
        match self {
            Self::Sse => DownstreamMode::Streaming,
            Self::JsonWhitespace | Self::Disabled => DownstreamMode::NonStreamJson,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShieldedRetryPolicy {
    enabled: bool,
    max_attempts: u32,
    anti_loop_hint_enabled: bool,
}

impl ShieldedRetryPolicy {
    fn from_config(config: &RetryConfig) -> Self {
        Self {
            enabled: config.enabled,
            max_attempts: if config.enabled {
                config.max_attempts
            } else {
                1
            },
            anti_loop_hint_enabled: config.anti_loop_hint_enabled,
        }
    }

    fn allows_retry_after(self, attempt_number: u32) -> bool {
        self.enabled && attempt_number < self.max_attempts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShieldedRetryCause {
    LoopDetected,
    TransientUpstreamStatus,
    TransientTransport,
    TransientStream,
}

impl ShieldedRetryCause {
    const fn retry_reason(self) -> &'static str {
        match self {
            Self::LoopDetected => "loop_detected",
            Self::TransientUpstreamStatus => "transient_upstream_status",
            Self::TransientTransport => "transient_upstream_transport",
            Self::TransientStream => "transient_upstream_stream_failure",
        }
    }

    const fn next_attempt_reason(self) -> &'static str {
        match self {
            Self::LoopDetected => "previous_loop_detected",
            Self::TransientUpstreamStatus => "previous_transient_upstream_status",
            Self::TransientTransport => "previous_transient_upstream_transport",
            Self::TransientStream => "previous_transient_upstream_stream_failure",
        }
    }
}

async fn forward_upstream_response(
    dispatch: ResponseDispatch<'_>,
    response_parts: ForwardedResponseParts,
    upstream_response: reqwest::Response,
    in_flight_permit: OwnedSemaphorePermit,
) -> Result<Response<Body>, ProxyError> {
    let upstream_status = response_parts.upstream_status;
    let upstream_headers = response_parts.upstream_headers.clone();
    if should_enrich_models_response(dispatch.method, dispatch.uri, dispatch.config) {
        return forward_enriched_models_response(
            response_parts,
            upstream_response,
            in_flight_permit,
            &dispatch.config.upstream.metadata,
        )
        .await;
    }

    let observer = response_parts.into_observer();
    let response_body =
        ObservedUpstreamBody::new(upstream_response.bytes_stream(), observer, in_flight_permit);
    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    ))
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

fn should_enrich_models_response(method: &Method, uri: &Uri, config: &AppConfig) -> bool {
    method == Method::GET
        && uri.path() == "/v1/models"
        && config.upstream.metadata.discovery_enabled
        && config.upstream.metadata.enrich_responses
}

fn should_intercept_non_stream_chat(method: &Method, uri: &Uri, config: &AppConfig) -> bool {
    method == Method::POST && uri.path() == "/v1/chat/completions" && config.shielding.enabled
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

    const fn is_transient(self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::Connect | Self::Body | Self::Other
        )
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
    input_fingerprint: Option<String>,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    attempt_request_metadata: BTreeMap<String, String>,
}

impl ForwardedResponseParts {
    fn into_observer(self) -> ForwardedBodyObserver {
        let downstream_mode = downstream_mode_from_headers(&self.upstream_headers);
        self.into_observer_with(downstream_mode, BTreeMap::new(), RawPayloads::default())
    }

    fn into_observer_with(
        self,
        downstream_mode: DownstreamMode,
        extra_response_metadata: BTreeMap<String, String>,
        raw_payloads: RawPayloads,
    ) -> ForwardedBodyObserver {
        let final_attempt = FinalAttemptContext {
            attempt_id: self.attempt_id,
            attempt_number: 1,
            attempt_max_attempts: 1,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            upstream_status: self.upstream_status,
            upstream_headers: self.upstream_headers.clone(),
            request_metadata: self.attempt_request_metadata,
            extra_response_metadata: extra_response_metadata.clone(),
            raw_payloads: raw_payloads.clone(),
        };
        ForwardedBodyObserver {
            downstream_mode,
            store: self.store,
            request_id: self.request_id,
            started_at_unix_ms: self.started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            model_id: self.model_id,
            input_fingerprint: self.input_fingerprint,
            downstream_status: self.upstream_status,
            downstream_headers: self.upstream_headers,
            request_metadata: self.request_metadata,
            extra_response_metadata,
            raw_payloads,
            completed_attempt_records: Vec::new(),
            final_attempt: Some(final_attempt),
            retry_observation: None,
            attempt_progress: None,
        }
    }

    fn into_body_read_error(self, error: ProxyError) -> ProxyError {
        self.into_body_read_error_with_metadata(error, BTreeMap::new())
    }

    fn into_body_read_error_with_metadata(
        self,
        error: ProxyError,
        extra_response_metadata: BTreeMap<String, String>,
    ) -> ProxyError {
        let finished_at_unix_ms = unix_time_millis();
        let error_reason = error.to_string();
        let attempt_record = failed_attempt_record(FailedAttemptRecordInput {
            attempt_id: self.attempt_id,
            request_id: self.request_id,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            finished_at_unix_ms,
            error_type: error.error_type(),
            error_reason: &error_reason,
            request_metadata: self.attempt_request_metadata,
            extra_response_metadata,
        });
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
    let observer = response_parts.into_observer();
    let response_body = ObservedBufferedBody::new(body, observer, in_flight_permit);

    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    ))
}

#[derive(Clone)]
struct ShieldedRetryRuntime {
    client: Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_method: Method,
    downstream_uri: Uri,
    downstream_headers: HeaderMap,
    upstream_body: Bytes,
    store: ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    request_metadata: BTreeMap<String, String>,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    loop_context: shielded_chat::LoopInspectionContext,
    retry_policy: ShieldedRetryPolicy,
}

#[derive(Clone, Debug)]
struct ShieldedAttemptInfo {
    attempt_id: AttemptId,
    request_id: RequestId,
    attempt_number: u32,
    attempt_max_attempts: u32,
    started_at_unix_ms: u64,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    upstream_mode: UpstreamMode,
    request_metadata: BTreeMap<String, String>,
}

struct ShieldedStartedAttempt {
    info: ShieldedAttemptInfo,
    response: reqwest::Response,
}

struct ShieldedAcceptedOutcome {
    body: Bytes,
    raw_payloads: RawPayloads,
    response_metadata: BTreeMap<String, String>,
    prior_attempt_records: Vec<AttemptRecord>,
    final_attempt: FinalAttemptContext,
}

struct ShieldedAggregatedAttempt {
    body: Bytes,
    raw_payloads: RawPayloads,
    response_metadata: BTreeMap<String, String>,
    final_attempt: FinalAttemptContext,
}

struct ShieldedFailureOutcome {
    error_type: &'static str,
    error_message: String,
    response_metadata: BTreeMap<String, String>,
    attempt_records: Vec<AttemptRecord>,
    upstream_mode: UpstreamMode,
    forwarded_response: Option<Box<ShieldedForwardedFailure>>,
}

struct ShieldedForwardedFailure {
    started: ShieldedStartedAttempt,
    final_attempt: FinalAttemptContext,
}

struct ShieldedTerminalForward {
    started: ShieldedStartedAttempt,
    prior_attempt_records: Vec<AttemptRecord>,
}

enum ShieldedRunOutcome {
    Accepted(ShieldedAcceptedOutcome),
    Failed(ShieldedFailureOutcome),
    TerminalForward(ShieldedTerminalForward),
}

enum ShieldedBeginOutcome {
    Aggregatable {
        started: ShieldedStartedAttempt,
        prior_attempt_records: Vec<AttemptRecord>,
    },
    Failed(ShieldedFailureOutcome),
    TerminalForward(ShieldedTerminalForward),
}

struct ShieldedAttemptFailure {
    attempt_id: AttemptId,
    request_id: RequestId,
    attempt_number: u32,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    upstream_mode: UpstreamMode,
    http_status: Option<u16>,
    error_type: &'static str,
    error_message: String,
    retry_cause: Option<ShieldedRetryCause>,
    abort_reason: Option<String>,
    request_metadata: BTreeMap<String, String>,
    response_metadata: BTreeMap<String, String>,
}

async fn forward_shielded_chat_with_retries(
    runtime: ShieldedRetryRuntime,
    in_flight_permit: OwnedSemaphorePermit,
) -> Result<Response<Body>, ProxyError> {
    if runtime.liveness.mode == ShieldedLivenessMode::Disabled {
        return Ok(
            match run_shielded_attempts(runtime.clone(), None, Vec::new(), true, None).await {
                ShieldedRunOutcome::Accepted(outcome) => {
                    shielded_retry_success_response(&runtime, outcome, in_flight_permit)
                }
                ShieldedRunOutcome::Failed(failure) => {
                    shielded_retry_error_response(&runtime, failure, in_flight_permit)
                }
                ShieldedRunOutcome::TerminalForward(terminal) => {
                    shielded_retry_terminal_forward_response(&runtime, terminal, in_flight_permit)
                }
            },
        );
    }

    match begin_shielded_retry(&runtime).await {
        ShieldedBeginOutcome::Aggregatable {
            started,
            prior_attempt_records,
        } => {
            let upstream_status = started.info.upstream_status;
            let upstream_content_type = started
                .info
                .upstream_headers
                .get(CONTENT_TYPE)
                .map(header_value);
            let response_headers = shielded_chat_stream_response_headers(
                &started.info.upstream_headers,
                runtime.liveness.mode,
            );
            let extra_metadata =
                shielded_liveness_response_metadata(&runtime.liveness, upstream_content_type);
            let attempt_progress = Arc::new(Mutex::new(ShieldedAttemptProgress {
                extra_response_metadata: extra_metadata.clone(),
                completed_attempt_records: prior_attempt_records.clone(),
                current_attempt: Some(
                    started
                        .info
                        .clone()
                        .into_final_context(extra_metadata.clone(), RawPayloads::default()),
                ),
            }));
            let observer = shielded_retry_observer(
                &runtime,
                ShieldedRetryObserverInput {
                    downstream_mode: runtime.liveness.mode.downstream_mode(),
                    downstream_status: upstream_status,
                    downstream_headers: response_headers.clone(),
                    upstream_mode: UpstreamMode::Streaming,
                    extra_response_metadata: extra_metadata,
                    raw_payloads: RawPayloads::default(),
                    completed_attempt_records: prior_attempt_records.clone(),
                    final_attempt: None,
                    attempt_progress: Some(attempt_progress.clone()),
                },
            );
            let aggregate_runtime = runtime.clone();
            let aggregate = Box::pin(async move {
                match run_shielded_attempts(
                    aggregate_runtime,
                    Some(started),
                    prior_attempt_records,
                    false,
                    Some(attempt_progress),
                )
                .await
                {
                    ShieldedRunOutcome::Accepted(outcome) => Ok(outcome),
                    ShieldedRunOutcome::Failed(failure) => Err(failure),
                    ShieldedRunOutcome::TerminalForward(terminal) => Err(terminal_forward_failure(
                        terminal,
                        "non-retryable upstream response after shielded retry",
                    )),
                }
            });
            let response_body = ShieldedLivenessBody::new(
                aggregate,
                runtime.liveness.mode,
                runtime.liveness.heartbeat_interval_secs,
                observer,
                in_flight_permit,
            );
            Ok(response_with_headers(
                upstream_status,
                response_headers,
                Body::from_stream(response_body),
            ))
        }
        ShieldedBeginOutcome::Failed(failure) => Ok(shielded_retry_error_response(
            &runtime,
            failure,
            in_flight_permit,
        )),
        ShieldedBeginOutcome::TerminalForward(terminal) => Ok(
            shielded_retry_terminal_forward_response(&runtime, terminal, in_flight_permit),
        ),
    }
}

enum ShieldedAttemptStep {
    Aggregatable(ShieldedStartedAttempt),
    Retry {
        attempt_number: u32,
        retry_cause: Option<ShieldedRetryCause>,
    },
    Failed(ShieldedFailureOutcome),
    TerminalForward(ShieldedTerminalForward),
}

enum ShieldedStartFailureStep {
    Retry {
        attempt_number: u32,
        retry_cause: Option<ShieldedRetryCause>,
    },
    Failed(ShieldedFailureOutcome),
}

fn shielded_start_failure_step(
    runtime: &ShieldedRetryRuntime,
    failure: ShieldedAttemptFailure,
    attempt_records: &mut Vec<AttemptRecord>,
) -> ShieldedStartFailureStep {
    let next_retry_cause = failure.retry_cause;
    let can_retry = next_retry_cause.is_some_and(|_cause| {
        runtime
            .retry_policy
            .allows_retry_after(failure.attempt_number)
    });
    attempt_records.push(attempt_failure_record(
        &failure,
        if can_retry {
            AttemptStatus::Retried
        } else {
            AttemptStatus::Failed
        },
        if can_retry { next_retry_cause } else { None },
        runtime.retry_policy,
    ));
    if can_retry {
        return ShieldedStartFailureStep::Retry {
            attempt_number: failure.attempt_number.saturating_add(1),
            retry_cause: next_retry_cause,
        };
    }
    ShieldedStartFailureStep::Failed(shielded_failure_outcome(
        failure,
        std::mem::take(attempt_records),
        runtime.retry_policy,
    ))
}

fn shielded_started_attempt_step(
    runtime: &ShieldedRetryRuntime,
    started: ShieldedStartedAttempt,
    attempt_records: &mut Vec<AttemptRecord>,
    allow_terminal_forward: bool,
) -> ShieldedAttemptStep {
    if started.info.upstream_status.is_success() && is_event_stream(&started.info.upstream_headers)
    {
        return ShieldedAttemptStep::Aggregatable(started);
    }

    if !started.info.upstream_status.is_success() {
        if let Some(cause) = retry_cause_for_upstream_status(started.info.upstream_status) {
            if runtime
                .retry_policy
                .allows_retry_after(started.info.attempt_number)
            {
                attempt_records.push(started_status_attempt_record(
                    &started.info,
                    AttemptStatus::Retried,
                    Some(cause),
                    runtime.retry_policy,
                    "retryable upstream status before shielded stream",
                ));
                return ShieldedAttemptStep::Retry {
                    attempt_number: started.info.attempt_number.saturating_add(1),
                    retry_cause: Some(cause),
                };
            }
            let failure = status_failure(
                &started.info,
                cause,
                "retryable upstream status attempts exhausted before shielded stream",
            );
            return ShieldedAttemptStep::Failed(shielded_forwarded_status_failure_outcome(
                failure,
                std::mem::take(attempt_records),
                runtime.retry_policy,
                started,
            ));
        }
        if allow_terminal_forward {
            return ShieldedAttemptStep::TerminalForward(ShieldedTerminalForward {
                started,
                prior_attempt_records: std::mem::take(attempt_records),
            });
        }
        let failure = status_failure_without_retry(
            &started.info,
            "non-retryable upstream response after shielded response started",
        );
        attempt_records.push(attempt_failure_record(
            &failure,
            AttemptStatus::Failed,
            None,
            runtime.retry_policy,
        ));
        return ShieldedAttemptStep::Failed(shielded_failure_outcome(
            failure,
            std::mem::take(attempt_records),
            runtime.retry_policy,
        ));
    }

    let failure = status_failure_without_retry(
        &started.info,
        "shielded chat completion expected upstream text/event-stream response",
    );
    attempt_records.push(attempt_failure_record(
        &failure,
        AttemptStatus::Failed,
        None,
        runtime.retry_policy,
    ));
    ShieldedAttemptStep::Failed(shielded_failure_outcome(
        failure,
        std::mem::take(attempt_records),
        runtime.retry_policy,
    ))
}

async fn begin_shielded_retry(runtime: &ShieldedRetryRuntime) -> ShieldedBeginOutcome {
    let mut attempt_number = 1;
    let mut retry_cause = None;
    let mut attempt_records = Vec::new();
    loop {
        let started = match start_shielded_attempt(runtime, attempt_number, retry_cause).await {
            Ok(started) => started,
            Err(failure) => {
                match shielded_start_failure_step(runtime, failure, &mut attempt_records) {
                    ShieldedStartFailureStep::Retry {
                        attempt_number: next_attempt_number,
                        retry_cause: next_retry_cause,
                    } => {
                        attempt_number = next_attempt_number;
                        retry_cause = next_retry_cause;
                        continue;
                    }
                    ShieldedStartFailureStep::Failed(outcome) => {
                        return ShieldedBeginOutcome::Failed(outcome);
                    }
                }
            }
        };

        match shielded_started_attempt_step(runtime, started, &mut attempt_records, true) {
            ShieldedAttemptStep::Aggregatable(started) => {
                return ShieldedBeginOutcome::Aggregatable {
                    started,
                    prior_attempt_records: attempt_records,
                };
            }
            ShieldedAttemptStep::Retry {
                attempt_number: next_attempt_number,
                retry_cause: next_retry_cause,
            } => {
                attempt_number = next_attempt_number;
                retry_cause = next_retry_cause;
            }
            ShieldedAttemptStep::Failed(outcome) => return ShieldedBeginOutcome::Failed(outcome),
            ShieldedAttemptStep::TerminalForward(terminal) => {
                return ShieldedBeginOutcome::TerminalForward(terminal);
            }
        }
    }
}

async fn aggregate_shielded_attempt(
    runtime: &ShieldedRetryRuntime,
    started: ShieldedStartedAttempt,
) -> Result<ShieldedAggregatedAttempt, ShieldedAttemptFailure> {
    let request_id = runtime.request_id.as_str().to_owned();
    let request_model_id = runtime.model_id.clone();
    match shielded_chat::aggregate_stream(
        started.response.bytes_stream(),
        started.info.started_at_unix_ms,
        &request_id,
        request_model_id.as_deref(),
        runtime.loop_context.clone(),
    )
    .await
    {
        Ok(aggregated) => Ok(ShieldedAggregatedAttempt {
            final_attempt: started.info.into_final_context(
                aggregated.response_metadata.clone(),
                aggregated.raw_payloads.clone(),
            ),
            body: aggregated.body,
            raw_payloads: aggregated.raw_payloads,
            response_metadata: aggregated.response_metadata,
        }),
        Err(error) => Err(aggregation_failure(&started.info, &error)),
    }
}

async fn run_shielded_attempts(
    runtime: ShieldedRetryRuntime,
    initial_attempt: Option<ShieldedStartedAttempt>,
    mut attempt_records: Vec<AttemptRecord>,
    allow_terminal_forward: bool,
    attempt_progress: Option<ShieldedAttemptProgressHandle>,
) -> ShieldedRunOutcome {
    let mut current_attempt = initial_attempt;
    let mut attempt_number = current_attempt
        .as_ref()
        .map_or(1, |attempt| attempt.info.attempt_number);
    let mut retry_cause = None;
    loop {
        let started = if let Some(started) = current_attempt.take() {
            started
        } else {
            match start_shielded_attempt(&runtime, attempt_number, retry_cause).await {
                Ok(started) => started,
                Err(failure) => {
                    match shielded_start_failure_step(&runtime, failure, &mut attempt_records) {
                        ShieldedStartFailureStep::Retry {
                            attempt_number: next_attempt_number,
                            retry_cause: next_retry_cause,
                        } => {
                            attempt_number = next_attempt_number;
                            retry_cause = next_retry_cause;
                            continue;
                        }
                        ShieldedStartFailureStep::Failed(outcome) => {
                            return ShieldedRunOutcome::Failed(outcome);
                        }
                    }
                }
            }
        };

        update_shielded_attempt_progress(
            attempt_progress.as_ref(),
            &attempt_records,
            Some(&started.info),
        );
        let started = match shielded_started_attempt_step(
            &runtime,
            started,
            &mut attempt_records,
            allow_terminal_forward,
        ) {
            ShieldedAttemptStep::Aggregatable(started) => started,
            ShieldedAttemptStep::Retry {
                attempt_number: next_attempt_number,
                retry_cause: next_retry_cause,
            } => {
                update_shielded_attempt_progress(attempt_progress.as_ref(), &attempt_records, None);
                attempt_number = next_attempt_number;
                retry_cause = next_retry_cause;
                continue;
            }
            ShieldedAttemptStep::Failed(outcome) => return ShieldedRunOutcome::Failed(outcome),
            ShieldedAttemptStep::TerminalForward(terminal) => {
                return ShieldedRunOutcome::TerminalForward(terminal);
            }
        };

        match aggregate_shielded_attempt(&runtime, started).await {
            Ok(aggregated) => {
                return ShieldedRunOutcome::Accepted(ShieldedAcceptedOutcome {
                    body: aggregated.body,
                    raw_payloads: aggregated.raw_payloads,
                    response_metadata: aggregated.response_metadata,
                    prior_attempt_records: attempt_records,
                    final_attempt: aggregated.final_attempt,
                });
            }
            Err(failure) => {
                let next_retry_cause = failure.retry_cause;
                let can_retry = next_retry_cause.is_some_and(|_cause| {
                    runtime
                        .retry_policy
                        .allows_retry_after(failure.attempt_number)
                });
                attempt_records.push(attempt_failure_record(
                    &failure,
                    if can_retry {
                        AttemptStatus::Retried
                    } else {
                        AttemptStatus::Failed
                    },
                    if can_retry { next_retry_cause } else { None },
                    runtime.retry_policy,
                ));
                update_shielded_attempt_progress(attempt_progress.as_ref(), &attempt_records, None);
                if can_retry {
                    attempt_number = failure.attempt_number.saturating_add(1);
                    retry_cause = next_retry_cause;
                    continue;
                }
                return ShieldedRunOutcome::Failed(shielded_failure_outcome(
                    failure,
                    attempt_records,
                    runtime.retry_policy,
                ));
            }
        }
    }
}

async fn start_shielded_attempt(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
) -> Result<ShieldedStartedAttempt, ShieldedAttemptFailure> {
    let attempt_id = AttemptId::for_request(&runtime.request_id, attempt_number);
    let attempt_started_at_unix_ms = unix_time_millis();
    let (upstream_body, anti_loop_hint_applied) =
        shielded_attempt_body(runtime, attempt_number, retry_cause);
    let request_metadata = shielded_attempt_request_metadata(
        runtime,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
    );
    match send_upstream_request(
        &runtime.client,
        runtime.method.clone(),
        runtime.upstream_url.clone(),
        &runtime.downstream_headers,
        upstream_body,
    )
    .await
    {
        Ok(response) => {
            let upstream_status = response.status();
            let upstream_headers = response.headers().clone();
            let upstream_mode = upstream_mode_from_headers(&upstream_headers);
            Ok(ShieldedStartedAttempt {
                info: ShieldedAttemptInfo {
                    attempt_id,
                    request_id: runtime.request_id.clone(),
                    attempt_number,
                    attempt_max_attempts: runtime.retry_policy.max_attempts,
                    started_at_unix_ms: attempt_started_at_unix_ms,
                    upstream_status,
                    upstream_headers,
                    upstream_mode,
                    request_metadata,
                },
                response,
            })
        }
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let retry_cause = transport_retry_cause(&error);
            let mut response_metadata = failed_response_metadata(
                attempt_started_at_unix_ms,
                finished_at_unix_ms,
                error.error_type(),
            );
            response_metadata.insert(
                String::from("upstream_response_received"),
                String::from("false"),
            );
            Err(ShieldedAttemptFailure {
                attempt_id,
                request_id: runtime.request_id.clone(),
                attempt_number,
                started_at_unix_ms: attempt_started_at_unix_ms,
                finished_at_unix_ms,
                upstream_mode: UpstreamMode::NotApplicable,
                http_status: None,
                error_type: error.error_type(),
                error_message: error.to_string(),
                retry_cause,
                abort_reason: None,
                request_metadata,
                response_metadata,
            })
        }
    }
}

impl ShieldedAttemptInfo {
    fn into_final_context(
        self,
        extra_response_metadata: BTreeMap<String, String>,
        raw_payloads: RawPayloads,
    ) -> FinalAttemptContext {
        FinalAttemptContext {
            attempt_id: self.attempt_id,
            attempt_number: self.attempt_number,
            attempt_max_attempts: self.attempt_max_attempts,
            started_at_unix_ms: self.started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            upstream_status: self.upstream_status,
            upstream_headers: self.upstream_headers,
            request_metadata: self.request_metadata,
            extra_response_metadata,
            raw_payloads,
        }
    }
}

fn shielded_attempt_body(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
) -> (Bytes, bool) {
    if attempt_number > 1
        && runtime.retry_policy.anti_loop_hint_enabled
        && matches!(retry_cause, Some(ShieldedRetryCause::LoopDetected))
    {
        if let Some(body) = shielded_chat::body_with_anti_loop_retry_hint(
            &runtime.upstream_body,
            attempt_number,
            runtime.retry_policy.max_attempts,
        ) {
            return (body, true);
        }
    }
    (runtime.upstream_body.clone(), false)
}

fn shielded_attempt_request_metadata(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    anti_loop_hint_applied: bool,
) -> BTreeMap<String, String> {
    let mut metadata = attempt_request_metadata(
        &runtime.downstream_method,
        &runtime.downstream_uri,
        &runtime.downstream_headers,
    );
    add_shielded_request_metadata(
        &mut metadata,
        true,
        &runtime.liveness,
        &runtime.thinking_metadata,
    );
    add_retry_attempt_metadata(
        &mut metadata,
        runtime.retry_policy,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
    );
    metadata
}

fn add_retry_attempt_metadata(
    metadata: &mut BTreeMap<String, String>,
    policy: ShieldedRetryPolicy,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    anti_loop_hint_applied: bool,
) {
    metadata.insert(String::from("attempt_number"), attempt_number.to_string());
    metadata.insert(
        String::from("retry_policy_enabled"),
        policy.enabled.to_string(),
    );
    metadata.insert(
        String::from("retry_max_attempts"),
        policy.max_attempts.to_string(),
    );
    metadata.insert(
        String::from("retry_anti_loop_hint_enabled"),
        policy.anti_loop_hint_enabled.to_string(),
    );
    metadata.insert(
        String::from("retry_previous_reason"),
        retry_cause.map_or_else(
            || String::from("none"),
            |cause| cause.next_attempt_reason().to_owned(),
        ),
    );
    metadata.insert(
        String::from("retry_anti_loop_hint_applied"),
        anti_loop_hint_applied.to_string(),
    );
}

fn add_retry_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    policy: ShieldedRetryPolicy,
) {
    metadata.insert(
        String::from("retry_policy_enabled"),
        policy.enabled.to_string(),
    );
    metadata.insert(
        String::from("retry_max_attempts"),
        policy.max_attempts.to_string(),
    );
    metadata.insert(
        String::from("retry_anti_loop_hint_enabled"),
        policy.anti_loop_hint_enabled.to_string(),
    );
}

fn retry_cause_for_upstream_status(status: reqwest::StatusCode) -> Option<ShieldedRetryCause> {
    if matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504) {
        Some(ShieldedRetryCause::TransientUpstreamStatus)
    } else {
        None
    }
}

fn transport_retry_cause(error: &ProxyError) -> Option<ShieldedRetryCause> {
    match error {
        ProxyError::UpstreamTransport { failure, .. } if failure.is_transient() => {
            Some(ShieldedRetryCause::TransientTransport)
        }
        _ => None,
    }
}

fn aggregation_failure(
    info: &ShieldedAttemptInfo,
    error: &shielded_chat::AggregationError,
) -> ShieldedAttemptFailure {
    let finished_at_unix_ms = unix_time_millis();
    let retry_cause = if error.is_loop_detected() {
        Some(ShieldedRetryCause::LoopDetected)
    } else {
        error
            .transient_stream_retry_reason()
            .map(|_reason| ShieldedRetryCause::TransientStream)
    };
    let mut response_metadata = failed_response_metadata(
        info.started_at_unix_ms,
        finished_at_unix_ms,
        "upstream_body_error",
    );
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("true"),
    );
    response_metadata.insert(
        String::from("http_status_success"),
        info.upstream_status.is_success().to_string(),
    );
    response_metadata.extend(error.response_metadata().clone());
    ShieldedAttemptFailure {
        attempt_id: info.attempt_id.clone(),
        request_id: info.request_id.clone(),
        attempt_number: info.attempt_number,
        started_at_unix_ms: info.started_at_unix_ms,
        finished_at_unix_ms,
        upstream_mode: info.upstream_mode,
        http_status: Some(info.upstream_status.as_u16()),
        error_type: "upstream_body_error",
        error_message: error.to_string(),
        retry_cause,
        abort_reason: retry_cause
            .filter(|cause| matches!(cause, ShieldedRetryCause::LoopDetected))
            .map(|_cause| String::from("loop_guard")),
        request_metadata: info.request_metadata.clone(),
        response_metadata,
    }
}

fn status_failure(
    info: &ShieldedAttemptInfo,
    cause: ShieldedRetryCause,
    message: &str,
) -> ShieldedAttemptFailure {
    let finished_at_unix_ms = unix_time_millis();
    let mut response_metadata = response_metadata(
        info.upstream_status,
        &info.upstream_headers,
        0,
        finished_at_unix_ms.saturating_sub(info.started_at_unix_ms),
    );
    response_metadata.insert(
        String::from("status_code"),
        info.upstream_status.as_u16().to_string(),
    );
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("true"),
    );
    ShieldedAttemptFailure {
        attempt_id: info.attempt_id.clone(),
        request_id: info.request_id.clone(),
        attempt_number: info.attempt_number,
        started_at_unix_ms: info.started_at_unix_ms,
        finished_at_unix_ms,
        upstream_mode: info.upstream_mode,
        http_status: Some(info.upstream_status.as_u16()),
        error_type: "upstream_status_error",
        error_message: format!("{message}: HTTP {}", info.upstream_status.as_u16()),
        retry_cause: Some(cause),
        abort_reason: None,
        request_metadata: info.request_metadata.clone(),
        response_metadata,
    }
}

fn status_failure_without_retry(
    info: &ShieldedAttemptInfo,
    message: &str,
) -> ShieldedAttemptFailure {
    let finished_at_unix_ms = unix_time_millis();
    let mut response_metadata = response_metadata(
        info.upstream_status,
        &info.upstream_headers,
        0,
        finished_at_unix_ms.saturating_sub(info.started_at_unix_ms),
    );
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("true"),
    );
    ShieldedAttemptFailure {
        attempt_id: info.attempt_id.clone(),
        request_id: info.request_id.clone(),
        attempt_number: info.attempt_number,
        started_at_unix_ms: info.started_at_unix_ms,
        finished_at_unix_ms,
        upstream_mode: info.upstream_mode,
        http_status: Some(info.upstream_status.as_u16()),
        error_type: "upstream_body_error",
        error_message: message.to_owned(),
        retry_cause: None,
        abort_reason: None,
        request_metadata: info.request_metadata.clone(),
        response_metadata,
    }
}

fn attempt_failure_record(
    failure: &ShieldedAttemptFailure,
    status: AttemptStatus,
    retry_cause: Option<ShieldedRetryCause>,
    policy: ShieldedRetryPolicy,
) -> AttemptRecord {
    let mut response_metadata = failure.response_metadata.clone();
    response_metadata.insert(
        String::from("attempt_number"),
        failure.attempt_number.to_string(),
    );
    response_metadata.insert(
        String::from("attempt_max_attempts"),
        policy.max_attempts.to_string(),
    );
    response_metadata.insert(String::from("attempt_outcome"), status.as_str().to_owned());
    response_metadata.insert(
        String::from("attempt_duration_ms"),
        failure
            .finished_at_unix_ms
            .saturating_sub(failure.started_at_unix_ms)
            .to_string(),
    );
    response_metadata.insert(
        String::from("retry_policy_enabled"),
        policy.enabled.to_string(),
    );
    if let Some(cause) = retry_cause {
        response_metadata.insert(
            String::from("retry_reason"),
            cause.retry_reason().to_owned(),
        );
    } else if failure.retry_cause.is_some() {
        response_metadata.insert(String::from("retry_exhausted"), String::from("true"));
    }
    if let Some(abort_reason) = &failure.abort_reason {
        response_metadata.insert(String::from("abort_reason"), abort_reason.clone());
    }
    AttemptRecord {
        attempt_id: failure.attempt_id.clone(),
        request_id: failure.request_id.clone(),
        attempt_number: failure.attempt_number,
        started_at_unix_ms: failure.started_at_unix_ms,
        finished_at_unix_ms: Some(failure.finished_at_unix_ms),
        upstream_mode: failure.upstream_mode,
        status,
        http_status: failure.http_status,
        error_reason: Some(format!("{}: {}", failure.error_type, failure.error_message)),
        retry_reason: retry_cause.map(|cause| cause.retry_reason().to_owned()),
        abort_reason: failure.abort_reason.clone(),
        request_metadata: failure.request_metadata.clone(),
        response_metadata,
        raw_payloads: RawPayloads::default(),
    }
}

fn started_status_attempt_record(
    info: &ShieldedAttemptInfo,
    status: AttemptStatus,
    retry_cause: Option<ShieldedRetryCause>,
    policy: ShieldedRetryPolicy,
    message: &str,
) -> AttemptRecord {
    let failure = status_failure(
        info,
        retry_cause.unwrap_or(ShieldedRetryCause::TransientUpstreamStatus),
        message,
    );
    attempt_failure_record(&failure, status, retry_cause, policy)
}

fn shielded_failure_outcome(
    failure: ShieldedAttemptFailure,
    attempt_records: Vec<AttemptRecord>,
    policy: ShieldedRetryPolicy,
) -> ShieldedFailureOutcome {
    let mut response_metadata = failure.response_metadata.clone();
    response_metadata.extend(retry_chain_metadata(
        &attempt_records,
        policy,
        RequestStatus::Failed.as_str(),
    ));
    ShieldedFailureOutcome {
        error_type: failure.error_type,
        error_message: failure.error_message,
        response_metadata,
        attempt_records,
        upstream_mode: failure.upstream_mode,
        forwarded_response: None,
    }
}

fn shielded_forwarded_status_failure_outcome(
    failure: ShieldedAttemptFailure,
    attempt_records: Vec<AttemptRecord>,
    policy: ShieldedRetryPolicy,
    started: ShieldedStartedAttempt,
) -> ShieldedFailureOutcome {
    let final_attempt = started.info.clone().into_final_context(
        status_failure_final_attempt_metadata(&failure),
        RawPayloads::default(),
    );
    let mut chain_attempts = attempt_records.clone();
    chain_attempts.push(final_attempt_record(
        final_attempt.clone(),
        &failure.request_id,
        failure.finished_at_unix_ms,
        0,
        &BodyCompletion::UpstreamStatusError(failure.error_message.clone()),
    ));
    let mut response_metadata = failure.response_metadata.clone();
    response_metadata.extend(retry_chain_metadata(
        &chain_attempts,
        policy,
        RequestStatus::Failed.as_str(),
    ));
    ShieldedFailureOutcome {
        error_type: failure.error_type,
        error_message: failure.error_message,
        response_metadata,
        attempt_records,
        upstream_mode: failure.upstream_mode,
        forwarded_response: Some(Box::new(ShieldedForwardedFailure {
            started,
            final_attempt,
        })),
    }
}

fn status_failure_final_attempt_metadata(
    failure: &ShieldedAttemptFailure,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    for key in ["status_code", "upstream_response_received"] {
        if let Some(value) = failure.response_metadata.get(key) {
            metadata.insert(key.to_owned(), value.clone());
        }
    }
    if failure.retry_cause.is_some() {
        metadata.insert(String::from("retry_exhausted"), String::from("true"));
    }
    metadata
}

fn terminal_forward_failure(
    terminal: ShieldedTerminalForward,
    message: &str,
) -> ShieldedFailureOutcome {
    let failure = status_failure_without_retry(&terminal.started.info, message);
    let mut attempt_records = terminal.prior_attempt_records;
    attempt_records.push(attempt_failure_record(
        &failure,
        AttemptStatus::Failed,
        None,
        ShieldedRetryPolicy {
            enabled: false,
            max_attempts: 1,
            anti_loop_hint_enabled: false,
        },
    ));
    shielded_failure_outcome(
        failure,
        attempt_records,
        ShieldedRetryPolicy {
            enabled: false,
            max_attempts: 1,
            anti_loop_hint_enabled: false,
        },
    )
}

fn shielded_retry_success_response(
    runtime: &ShieldedRetryRuntime,
    mut outcome: ShieldedAcceptedOutcome,
    in_flight_permit: OwnedSemaphorePermit,
) -> Response<Body> {
    let body_len = outcome.body.len();
    let upstream_headers = outcome.final_attempt.upstream_headers.clone();
    let upstream_status = outcome.final_attempt.upstream_status;
    let upstream_content_type = upstream_headers.get(CONTENT_TYPE).map(header_value);
    let response_headers = shielded_chat_response_headers(&upstream_headers, body_len);
    let mut extra_metadata = outcome.response_metadata.clone();
    extra_metadata.extend(shielded_liveness_response_metadata(
        &runtime.liveness,
        upstream_content_type.clone(),
    ));
    if let Some(content_type) = upstream_content_type {
        extra_metadata.insert(
            String::from("upstream_response_header_content-type"),
            content_type,
        );
    }
    outcome
        .final_attempt
        .extra_response_metadata
        .extend(extra_metadata.clone());
    outcome.final_attempt.raw_payloads = outcome.raw_payloads.clone();
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: DownstreamMode::NonStreamJson,
            downstream_status: upstream_status,
            downstream_headers: response_headers.clone(),
            upstream_mode: outcome.final_attempt.upstream_mode,
            extra_response_metadata: extra_metadata,
            raw_payloads: outcome.raw_payloads,
            completed_attempt_records: outcome.prior_attempt_records,
            final_attempt: Some(outcome.final_attempt),
            attempt_progress: None,
        },
    );
    let response_body = ObservedBufferedBody::new(outcome.body, observer, in_flight_permit);
    response_with_headers(
        upstream_status,
        response_headers,
        Body::from_stream(response_body),
    )
}

fn shielded_retry_error_response(
    runtime: &ShieldedRetryRuntime,
    failure: ShieldedFailureOutcome,
    in_flight_permit: OwnedSemaphorePermit,
) -> Response<Body> {
    if failure.forwarded_response.is_some() {
        return shielded_retry_forwarded_failure_response(runtime, failure, in_flight_permit);
    }

    let body = proxy_error_json_body(failure.error_type, &failure.error_message);
    let response_headers = json_response_headers(body.len());
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: runtime.liveness.mode.downstream_mode(),
            downstream_status: StatusCode::BAD_GATEWAY,
            downstream_headers: response_headers.clone(),
            upstream_mode: failure.upstream_mode,
            extra_response_metadata: failure.response_metadata,
            raw_payloads: RawPayloads::default(),
            completed_attempt_records: failure.attempt_records,
            final_attempt: None,
            attempt_progress: None,
        },
    );
    let completion = BodyCompletion::UpstreamStreamError(failure.error_message);
    let response_body =
        ObservedBufferedBody::new_with_completion(body, observer, in_flight_permit, completion);
    response_with_headers(
        StatusCode::BAD_GATEWAY,
        response_headers,
        Body::from_stream(response_body),
    )
}

fn shielded_retry_forwarded_failure_response(
    runtime: &ShieldedRetryRuntime,
    mut failure: ShieldedFailureOutcome,
    in_flight_permit: OwnedSemaphorePermit,
) -> Response<Body> {
    let forwarded = failure
        .forwarded_response
        .take()
        .expect("forwarded failure response should be present");
    let upstream_status = forwarded.started.info.upstream_status;
    let upstream_headers = forwarded.started.info.upstream_headers.clone();
    let mut response_headers = HeaderMap::new();
    copy_response_headers(&upstream_headers, &mut response_headers);
    let terminal_completion = BodyCompletion::UpstreamStatusError(failure.error_message);
    let mut extra_response_metadata = failure.response_metadata;
    extra_response_metadata.remove("response_body_bytes");
    extra_response_metadata.remove("latency_ms");
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: downstream_mode_from_headers(&upstream_headers),
            downstream_status: upstream_status,
            downstream_headers: response_headers.clone(),
            upstream_mode: forwarded.final_attempt.upstream_mode,
            extra_response_metadata,
            raw_payloads: RawPayloads::default(),
            completed_attempt_records: failure.attempt_records,
            final_attempt: Some(forwarded.final_attempt),
            attempt_progress: None,
        },
    );
    let response_body = ObservedUpstreamBody::new_with_completion(
        forwarded.started.response.bytes_stream(),
        observer,
        in_flight_permit,
        terminal_completion,
    );
    response_with_headers(
        upstream_status,
        response_headers,
        Body::from_stream(response_body),
    )
}

fn shielded_retry_terminal_forward_response(
    runtime: &ShieldedRetryRuntime,
    terminal: ShieldedTerminalForward,
    in_flight_permit: OwnedSemaphorePermit,
) -> Response<Body> {
    let upstream_status = terminal.started.info.upstream_status;
    let upstream_headers = terminal.started.info.upstream_headers.clone();
    let final_attempt = terminal
        .started
        .info
        .clone()
        .into_final_context(BTreeMap::new(), RawPayloads::default());
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: downstream_mode_from_headers(&upstream_headers),
            downstream_status: upstream_status,
            downstream_headers: upstream_headers.clone(),
            upstream_mode: final_attempt.upstream_mode,
            extra_response_metadata: BTreeMap::new(),
            raw_payloads: RawPayloads::default(),
            completed_attempt_records: terminal.prior_attempt_records,
            final_attempt: Some(final_attempt),
            attempt_progress: None,
        },
    );
    let response_body = ObservedUpstreamBody::new(
        terminal.started.response.bytes_stream(),
        observer,
        in_flight_permit,
    );
    downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    )
}

struct ShieldedRetryObserverInput {
    downstream_mode: DownstreamMode,
    downstream_status: reqwest::StatusCode,
    downstream_headers: HeaderMap,
    upstream_mode: UpstreamMode,
    extra_response_metadata: BTreeMap<String, String>,
    raw_payloads: RawPayloads,
    completed_attempt_records: Vec<AttemptRecord>,
    final_attempt: Option<FinalAttemptContext>,
    attempt_progress: Option<ShieldedAttemptProgressHandle>,
}

fn shielded_retry_observer(
    runtime: &ShieldedRetryRuntime,
    input: ShieldedRetryObserverInput,
) -> ForwardedBodyObserver {
    ForwardedBodyObserver {
        store: runtime.store.clone(),
        request_id: runtime.request_id.clone(),
        started_at_unix_ms: runtime.started_at_unix_ms,
        downstream_mode: input.downstream_mode,
        upstream_mode: input.upstream_mode,
        model_id: runtime.model_id.clone(),
        input_fingerprint: runtime.liveness.input_fingerprint.clone(),
        downstream_status: input.downstream_status,
        downstream_headers: input.downstream_headers,
        request_metadata: runtime.request_metadata.clone(),
        extra_response_metadata: input.extra_response_metadata,
        raw_payloads: input.raw_payloads,
        completed_attempt_records: input.completed_attempt_records,
        final_attempt: input.final_attempt,
        retry_observation: Some(RetryObservation {
            policy: runtime.retry_policy,
        }),
        attempt_progress: input.attempt_progress,
    }
}

fn json_response_headers(body_len: usize) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Ok(content_length) = HeaderValue::from_str(&body_len.to_string()) {
        headers.insert(CONTENT_LENGTH, content_length);
    }
    headers
}

#[derive(Clone, Debug)]
struct FinalAttemptContext {
    attempt_id: AttemptId,
    attempt_number: u32,
    attempt_max_attempts: u32,
    started_at_unix_ms: u64,
    upstream_mode: UpstreamMode,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    extra_response_metadata: BTreeMap<String, String>,
    raw_payloads: RawPayloads,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetryObservation {
    policy: ShieldedRetryPolicy,
}

type ShieldedAttemptProgressHandle = Arc<Mutex<ShieldedAttemptProgress>>;

#[derive(Debug)]
struct ShieldedAttemptProgress {
    extra_response_metadata: BTreeMap<String, String>,
    completed_attempt_records: Vec<AttemptRecord>,
    current_attempt: Option<FinalAttemptContext>,
}

fn shielded_attempt_progress(
    progress: &ShieldedAttemptProgressHandle,
) -> MutexGuard<'_, ShieldedAttemptProgress> {
    match progress.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn update_shielded_attempt_progress(
    progress: Option<&ShieldedAttemptProgressHandle>,
    completed_attempt_records: &[AttemptRecord],
    current_attempt: Option<&ShieldedAttemptInfo>,
) {
    if let Some(progress) = progress {
        let mut progress = shielded_attempt_progress(progress);
        progress.completed_attempt_records = completed_attempt_records.to_vec();
        let extra_response_metadata = progress.extra_response_metadata.clone();
        progress.current_attempt = current_attempt.map(|info| {
            info.clone()
                .into_final_context(extra_response_metadata, RawPayloads::default())
        });
    }
}

struct ForwardedBodyObserver {
    store: ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    downstream_mode: DownstreamMode,
    upstream_mode: UpstreamMode,
    model_id: Option<String>,
    input_fingerprint: Option<String>,
    downstream_status: reqwest::StatusCode,
    downstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    extra_response_metadata: BTreeMap<String, String>,
    raw_payloads: RawPayloads,
    completed_attempt_records: Vec<AttemptRecord>,
    final_attempt: Option<FinalAttemptContext>,
    retry_observation: Option<RetryObservation>,
    attempt_progress: Option<ShieldedAttemptProgressHandle>,
}

impl ForwardedBodyObserver {
    fn record(self, body_bytes: u64, completion: &BodyCompletion) {
        let finished_at_unix_ms = unix_time_millis();
        let mut attempts = self.completed_attempt_records;
        let mut final_attempt = self.final_attempt;
        if matches!(completion, BodyCompletion::DownstreamDropped) {
            if let Some(progress) = &self.attempt_progress {
                let progress = shielded_attempt_progress(progress);
                attempts = progress.completed_attempt_records.clone();
                final_attempt.clone_from(&progress.current_attempt);
            }
        }
        let upstream_mode = final_attempt
            .as_ref()
            .map_or(self.upstream_mode, |attempt| attempt.upstream_mode);
        if let Some(final_attempt) = final_attempt {
            attempts.push(final_attempt_record(
                final_attempt,
                &self.request_id,
                finished_at_unix_ms,
                body_bytes,
                completion,
            ));
        }

        let mut response_metadata = response_metadata(
            self.downstream_status,
            &self.downstream_headers,
            body_bytes,
            finished_at_unix_ms.saturating_sub(self.started_at_unix_ms),
        );
        response_metadata.extend(self.extra_response_metadata);
        if let Some(retry_observation) = self.retry_observation {
            response_metadata.extend(retry_chain_metadata(
                &attempts,
                retry_observation.policy,
                completion.request_status().as_str(),
            ));
        }
        let request_record = RequestRecord {
            request_id: self.request_id,
            started_at_unix_ms: self.started_at_unix_ms,
            finished_at_unix_ms: Some(finished_at_unix_ms),
            downstream_mode: self.downstream_mode,
            upstream_mode,
            model_id: self.model_id,
            input_fingerprint: self.input_fingerprint,
            status: completion.request_status(),
            http_status: Some(self.downstream_status.as_u16()),
            error_reason: completion.error_reason(),
            abort_reason: completion.abort_reason(),
            request_metadata: self.request_metadata,
            response_metadata,
            raw_payloads: self.raw_payloads,
        };
        record_observability_many(&self.store, &request_record, &attempts);
    }
}

fn final_attempt_record(
    attempt: FinalAttemptContext,
    request_id: &RequestId,
    finished_at_unix_ms: u64,
    body_bytes: u64,
    completion: &BodyCompletion,
) -> AttemptRecord {
    let mut response_metadata = response_metadata(
        attempt.upstream_status,
        &attempt.upstream_headers,
        body_bytes,
        finished_at_unix_ms.saturating_sub(attempt.started_at_unix_ms),
    );
    response_metadata.extend(attempt.extra_response_metadata);
    response_metadata.insert(
        String::from("attempt_number"),
        attempt.attempt_number.to_string(),
    );
    response_metadata.insert(
        String::from("attempt_max_attempts"),
        attempt.attempt_max_attempts.to_string(),
    );
    response_metadata.insert(
        String::from("attempt_outcome"),
        completion.attempt_status().as_str().to_owned(),
    );
    AttemptRecord {
        attempt_id: attempt.attempt_id,
        request_id: request_id.clone(),
        attempt_number: attempt.attempt_number,
        started_at_unix_ms: attempt.started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        upstream_mode: attempt.upstream_mode,
        status: completion.attempt_status(),
        http_status: Some(attempt.upstream_status.as_u16()),
        error_reason: completion.error_reason(),
        retry_reason: None,
        abort_reason: completion.abort_reason(),
        request_metadata: attempt.request_metadata,
        response_metadata,
        raw_payloads: attempt.raw_payloads,
    }
}

fn retry_chain_metadata(
    attempts: &[AttemptRecord],
    policy: ShieldedRetryPolicy,
    final_outcome: &str,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("retry_policy_enabled"),
            policy.enabled.to_string(),
        ),
        (
            String::from("retry_max_attempts"),
            policy.max_attempts.to_string(),
        ),
        (
            String::from("retry_anti_loop_hint_enabled"),
            policy.anti_loop_hint_enabled.to_string(),
        ),
        (
            String::from("retry_attempt_count"),
            attempts.len().to_string(),
        ),
        (
            String::from("retry_final_outcome"),
            final_outcome.to_owned(),
        ),
        (
            String::from("retry_attempt_chain"),
            attempt_chain_summary(attempts),
        ),
    ])
}

fn attempt_chain_summary(attempts: &[AttemptRecord]) -> String {
    if attempts.is_empty() {
        return String::from("none");
    }
    attempts
        .iter()
        .map(|attempt| {
            format!(
                "{}:{}:{}:{}",
                attempt.attempt_number,
                attempt.status.as_str(),
                attempt.abort_reason.as_deref().unwrap_or("none"),
                attempt.retry_reason.as_deref().unwrap_or("none")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

enum BodyCompletion {
    Succeeded,
    UpstreamStreamError(String),
    UpstreamStatusError(String),
    DownstreamDropped,
}

impl BodyCompletion {
    const fn request_status(&self) -> RequestStatus {
        match self {
            Self::Succeeded => RequestStatus::Succeeded,
            Self::UpstreamStreamError(_) | Self::UpstreamStatusError(_) => RequestStatus::Failed,
            Self::DownstreamDropped => RequestStatus::Aborted,
        }
    }

    const fn attempt_status(&self) -> AttemptStatus {
        match self {
            Self::Succeeded => AttemptStatus::Succeeded,
            Self::UpstreamStreamError(_) | Self::UpstreamStatusError(_) => AttemptStatus::Failed,
            Self::DownstreamDropped => AttemptStatus::Aborted,
        }
    }

    fn error_reason(&self) -> Option<String> {
        match self {
            Self::UpstreamStreamError(error) => Some(format!("upstream_stream_error: {error}")),
            Self::UpstreamStatusError(error) => Some(format!("upstream_status_error: {error}")),
            Self::Succeeded | Self::DownstreamDropped => None,
        }
    }

    fn abort_reason(&self) -> Option<String> {
        match self {
            Self::DownstreamDropped => Some(String::from("downstream_body_dropped_before_eof")),
            Self::Succeeded | Self::UpstreamStreamError(_) | Self::UpstreamStatusError(_) => None,
        }
    }
}

struct ObservedUpstreamBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: OwnedSemaphorePermit,
    bytes_seen: u64,
    terminal_completion: BodyCompletion,
}

impl ObservedUpstreamBody {
    fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
    ) -> Self {
        Self::new_with_completion(
            stream,
            observer,
            in_flight_permit,
            BodyCompletion::Succeeded,
        )
    }

    fn new_with_completion(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
        terminal_completion: BodyCompletion,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            bytes_seen: 0,
            terminal_completion,
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
                let completion =
                    std::mem::replace(&mut this.terminal_completion, BodyCompletion::Succeeded);
                this.record_once(&completion);
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

struct ObservedBufferedBody {
    body: Option<Bytes>,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: OwnedSemaphorePermit,
    bytes_seen: u64,
    terminal_completion: BodyCompletion,
}

impl ObservedBufferedBody {
    fn new(
        body: Bytes,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
    ) -> Self {
        Self::new_with_completion(body, observer, in_flight_permit, BodyCompletion::Succeeded)
    }

    fn new_with_completion(
        body: Bytes,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
        terminal_completion: BodyCompletion,
    ) -> Self {
        Self {
            body: (!body.is_empty()).then_some(body),
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            bytes_seen: 0,
            terminal_completion,
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion) {
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion);
        }
    }
}

impl Stream for ObservedBufferedBody {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(body) = this.body.take() {
            let body_len = u64::try_from(body.len()).unwrap_or(u64::MAX);
            this.bytes_seen = this.bytes_seen.saturating_add(body_len);
            let completion =
                std::mem::replace(&mut this.terminal_completion, BodyCompletion::Succeeded);
            this.record_once(&completion);
            return Poll::Ready(Some(Ok(body)));
        }

        let completion =
            std::mem::replace(&mut this.terminal_completion, BodyCompletion::Succeeded);
        this.record_once(&completion);
        Poll::Ready(None)
    }
}

impl Drop for ObservedBufferedBody {
    fn drop(&mut self) {
        self.record_once(&BodyCompletion::DownstreamDropped);
    }
}

type ShieldedAggregateFuture =
    Pin<Box<dyn Future<Output = Result<ShieldedAcceptedOutcome, ShieldedFailureOutcome>> + Send>>;

struct ShieldedLivenessBody {
    aggregate: ShieldedAggregateFuture,
    interval: Interval,
    mode: ShieldedLivenessMode,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: OwnedSemaphorePermit,
    bytes_seen: u64,
    terminal_completion: Option<BodyCompletion>,
    json_prefix_pending: bool,
}

impl ShieldedLivenessBody {
    fn new(
        aggregate: ShieldedAggregateFuture,
        mode: ShieldedLivenessMode,
        interval_secs: u64,
        observer: ForwardedBodyObserver,
        in_flight_permit: OwnedSemaphorePermit,
    ) -> Self {
        let period = Duration::from_secs(interval_secs);
        let mut interval = tokio::time::interval_at(Instant::now() + period, period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            aggregate,
            interval,
            mode,
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            bytes_seen: 0,
            terminal_completion: None,
            json_prefix_pending: mode == ShieldedLivenessMode::JsonWhitespace,
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion) {
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion);
        }
    }

    fn count_and_emit(&mut self, bytes: Bytes) -> Poll<Option<Result<Bytes, Infallible>>> {
        let chunk_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.bytes_seen = self.bytes_seen.saturating_add(chunk_len);
        Poll::Ready(Some(Ok(bytes)))
    }

    fn accepted_chunk(&self, body: &Bytes) -> Bytes {
        match self.mode {
            ShieldedLivenessMode::Sse => sse_final_frame(body),
            ShieldedLivenessMode::JsonWhitespace | ShieldedLivenessMode::Disabled => body.clone(),
        }
    }

    fn error_chunk(&self, error_type: &str, error: &str) -> Bytes {
        let body = proxy_error_json_body(error_type, error);
        match self.mode {
            ShieldedLivenessMode::Sse => sse_error_frame(&body),
            ShieldedLivenessMode::JsonWhitespace | ShieldedLivenessMode::Disabled => body,
        }
    }
}

impl Stream for ShieldedLivenessBody {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(completion) = this.terminal_completion.take() {
            this.record_once(&completion);
            return Poll::Ready(None);
        }

        if this.json_prefix_pending {
            this.json_prefix_pending = false;
            return this.count_and_emit(json_whitespace_heartbeat());
        }

        match this.aggregate.as_mut().poll(cx) {
            Poll::Ready(Ok(outcome)) => {
                let chunk = this.accepted_chunk(&outcome.body);
                if let Some(observer) = &mut this.observer {
                    observer.completed_attempt_records = outcome.prior_attempt_records;
                    observer
                        .extra_response_metadata
                        .extend(outcome.response_metadata.clone());
                    observer.raw_payloads = outcome.raw_payloads.clone();
                    let mut final_attempt = outcome.final_attempt;
                    final_attempt
                        .extra_response_metadata
                        .extend(observer.extra_response_metadata.clone());
                    final_attempt.raw_payloads = outcome.raw_payloads;
                    observer.final_attempt = Some(final_attempt);
                }
                this.terminal_completion = Some(BodyCompletion::Succeeded);
                return this.count_and_emit(chunk);
            }
            Poll::Ready(Err(failure)) => {
                let forwarded_final_attempt = failure
                    .forwarded_response
                    .map(|forwarded| forwarded.final_attempt);
                if let Some(observer) = &mut this.observer {
                    observer.completed_attempt_records = failure.attempt_records;
                    observer
                        .extra_response_metadata
                        .extend(failure.response_metadata);
                    observer.final_attempt = forwarded_final_attempt;
                }
                let error_type = failure.error_type;
                let error_message = failure.error_message;
                let chunk = this.error_chunk(error_type, &error_message);
                this.terminal_completion = Some(BodyCompletion::UpstreamStreamError(error_message));
                return this.count_and_emit(chunk);
            }
            Poll::Pending => {}
        }

        match Pin::new(&mut this.interval).poll_tick(cx) {
            Poll::Ready(_instant) => this.count_and_emit(heartbeat_chunk(this.mode)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ShieldedLivenessBody {
    fn drop(&mut self) {
        if let Some(completion) = self.terminal_completion.take() {
            self.record_once(&completion);
        } else {
            self.record_once(&BodyCompletion::DownstreamDropped);
        }
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
    let mut headers = HeaderMap::new();
    copy_response_headers(upstream_headers, &mut headers);
    response_with_headers(status, headers, body)
}

fn response_with_headers(
    status: reqwest::StatusCode,
    headers: HeaderMap,
    body: Body,
) -> Response<Body> {
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

fn shielded_chat_response_headers(upstream_headers: &HeaderMap, body_len: usize) -> HeaderMap {
    let mut headers = HeaderMap::new();
    copy_response_headers(upstream_headers, &mut headers);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Ok(content_length) = HeaderValue::from_str(&body_len.to_string()) {
        headers.insert(CONTENT_LENGTH, content_length);
    }
    headers
}

fn shielded_chat_stream_response_headers(
    upstream_headers: &HeaderMap,
    mode: ShieldedLivenessMode,
) -> HeaderMap {
    let mut headers = HeaderMap::new();
    copy_response_headers(upstream_headers, &mut headers);
    let content_type = match mode {
        ShieldedLivenessMode::Sse => "text/event-stream",
        ShieldedLivenessMode::JsonWhitespace | ShieldedLivenessMode::Disabled => "application/json",
    };
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        HeaderName::from_static("cache-control"),
        HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    headers
}

fn heartbeat_chunk(mode: ShieldedLivenessMode) -> Bytes {
    match mode {
        ShieldedLivenessMode::Sse => Bytes::from_static(b": llm-guard-proxy heartbeat\n\n"),
        ShieldedLivenessMode::JsonWhitespace => json_whitespace_heartbeat(),
        ShieldedLivenessMode::Disabled => Bytes::new(),
    }
}

fn json_whitespace_heartbeat() -> Bytes {
    Bytes::from_static(b" \n")
}

fn sse_final_frame(body: &Bytes) -> Bytes {
    let mut frame = BytesMut::with_capacity(body.len().saturating_add(20));
    frame.extend_from_slice(b"event: final\n");
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(body);
    frame.extend_from_slice(b"\n\n");
    frame.freeze()
}

fn sse_error_frame(body: &Bytes) -> Bytes {
    let mut frame = BytesMut::with_capacity(body.len().saturating_add(20));
    frame.extend_from_slice(b"event: error\n");
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(body);
    frame.extend_from_slice(b"\n\n");
    frame.freeze()
}

fn proxy_error_json_body(error_type: &str, message: &str) -> Bytes {
    Bytes::from(
        json!({
            "error": {
                "type": error_type,
                "message": message,
            }
        })
        .to_string(),
    )
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

fn select_shielded_liveness(
    state: &ProxyState,
    config: &AppConfig,
    body: &Bytes,
    shielded_chat: bool,
    now_unix_ms: u64,
) -> ShieldedLivenessSelection {
    let input_fingerprint = shielded_chat
        .then(|| chat_input_fingerprint(body))
        .flatten();
    let repeat_observation = input_fingerprint
        .as_deref()
        .filter(|_fingerprint| config.loop_guard.enabled)
        .map_or_else(RepeatInputObservation::default, |fingerprint| {
            state.repeat_inputs.observe(
                fingerprint,
                now_unix_ms,
                config.loop_guard.normalized_input_window_secs,
                config.loop_guard.max_repeated_inputs,
            )
        });
    let mode = match config.heartbeat.mode {
        HeartbeatMode::Disabled => ShieldedLivenessMode::Disabled,
        HeartbeatMode::JsonWhitespace => ShieldedLivenessMode::JsonWhitespace,
        HeartbeatMode::Sse if repeat_observation.repeated => ShieldedLivenessMode::JsonWhitespace,
        HeartbeatMode::Sse => ShieldedLivenessMode::Sse,
    };

    ShieldedLivenessSelection {
        mode,
        heartbeat_interval_secs: config.heartbeat.interval_secs,
        input_fingerprint,
        repeat_observation,
        repeat_window_secs: config.loop_guard.normalized_input_window_secs,
        repeat_max_inputs: config.loop_guard.max_repeated_inputs,
    }
}

fn chat_input_fingerprint(body: &Bytes) -> Option<String> {
    let value = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    let normalized = normalize_chat_fingerprint_value(value)?;
    let serialized = serde_json::to_vec(&normalized).ok()?;
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    Some(format!("siphash64:{:016x}", hasher.finish()))
}

fn normalize_chat_fingerprint_value(value: serde_json::Value) -> Option<serde_json::Value> {
    let serde_json::Value::Object(object) = value else {
        return None;
    };
    let mut normalized = serde_json::Map::new();
    for (key, value) in object {
        if key == "stream" || is_sensitive_fingerprint_key(&key) {
            continue;
        }
        normalized.insert(key, sanitize_fingerprint_value(value));
    }
    Some(serde_json::Value::Object(normalized))
}

fn sanitize_fingerprint_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => {
            let mut sanitized = serde_json::Map::new();
            for (key, value) in object {
                if is_sensitive_fingerprint_key(&key) {
                    continue;
                }
                sanitized.insert(key, sanitize_fingerprint_value(value));
            }
            serde_json::Value::Object(sanitized)
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sanitize_fingerprint_value).collect())
        }
        value => value,
    }
}

fn is_sensitive_fingerprint_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|character| character.to_ascii_lowercase())
        .collect::<String>();

    if is_known_non_secret_token_fingerprint_key(&normalized) {
        return false;
    }

    let credential_keyword = [
        "authorization",
        "apikey",
        "accesskey",
        "privatekey",
        "secret",
        "password",
        "credential",
        "credentials",
        "bearer",
    ]
    .iter()
    .any(|sensitive| normalized.contains(sensitive));
    credential_keyword || normalized.contains("token")
}

fn is_known_non_secret_token_fingerprint_key(normalized_key: &str) -> bool {
    matches!(
        normalized_key,
        "maxtokens" | "maxcompletiontokens" | "maxoutputtokens" | "budgettokens"
    )
}

fn request_body_bytes_hint(headers: &HeaderMap) -> String {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map_or_else(|| String::from("unknown"), |bytes| bytes.to_string())
}

fn add_shielded_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    shielded_chat: bool,
    liveness: &ShieldedLivenessSelection,
    thinking_metadata: &BTreeMap<String, String>,
) {
    if shielded_chat {
        add_shielded_chat_request_metadata(metadata);
        add_shielded_liveness_request_metadata(metadata, liveness);
        metadata.extend(thinking_metadata.clone());
    }
}

fn add_shielded_chat_request_metadata(metadata: &mut BTreeMap<String, String>) {
    metadata.insert(String::from("shielded_streaming"), String::from("true"));
    metadata.insert(String::from("upstream_stream_forced"), String::from("true"));
    metadata.insert(
        String::from("policy_transform_applied"),
        String::from("true"),
    );
}

fn add_shielded_liveness_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    liveness: &ShieldedLivenessSelection,
) {
    metadata.extend(shielded_liveness_metadata(liveness));
}

fn shielded_liveness_response_metadata(
    liveness: &ShieldedLivenessSelection,
    upstream_content_type: Option<String>,
) -> BTreeMap<String, String> {
    let mut metadata = shielded_liveness_metadata(liveness);
    metadata.insert(
        String::from("shielded_downstream_streaming"),
        (liveness.mode == ShieldedLivenessMode::Sse).to_string(),
    );
    if let Some(content_type) = upstream_content_type {
        metadata.insert(
            String::from("upstream_response_header_content-type"),
            content_type,
        );
    }
    metadata
}

fn shielded_liveness_metadata(liveness: &ShieldedLivenessSelection) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("downstream_liveness_mode"),
            liveness.mode.as_str().to_owned(),
        ),
        (
            String::from("heartbeat_interval_secs"),
            liveness.heartbeat_interval_secs.to_string(),
        ),
        (
            String::from("repeat_input_window_secs"),
            liveness.repeat_window_secs.to_string(),
        ),
        (
            String::from("repeat_input_max_repeated_inputs"),
            liveness.repeat_max_inputs.to_string(),
        ),
        (
            String::from("input_fingerprint_present"),
            liveness.input_fingerprint.is_some().to_string(),
        ),
        (
            String::from("repeat_input_matched"),
            liveness.repeat_observation.repeated.to_string(),
        ),
        (
            String::from("repeat_input_prior_count"),
            liveness.repeat_observation.prior_count.to_string(),
        ),
    ])
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
                selected_header_metadata_value(&header, value),
            );
        }
    }
}

fn selected_header_metadata_value(name: &HeaderName, value: &HeaderValue) -> String {
    if name == AUTHORIZATION || name.as_str() == "x-api-key" {
        return String::from(HEADER_VALUE_REDACTED);
    }
    header_value(value)
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

struct FailedAttemptRecordInput<'error> {
    attempt_id: AttemptId,
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    error_type: &'static str,
    error_reason: &'error str,
    request_metadata: BTreeMap<String, String>,
    extra_response_metadata: BTreeMap<String, String>,
}

fn failed_attempt_record(input: FailedAttemptRecordInput<'_>) -> AttemptRecord {
    let mut response_metadata = failed_response_metadata(
        input.started_at_unix_ms,
        input.finished_at_unix_ms,
        input.error_type,
    );
    response_metadata.extend(input.extra_response_metadata);
    AttemptRecord {
        attempt_id: input.attempt_id,
        request_id: input.request_id,
        attempt_number: 1,
        started_at_unix_ms: input.started_at_unix_ms,
        finished_at_unix_ms: Some(input.finished_at_unix_ms),
        upstream_mode: UpstreamMode::NotApplicable,
        status: AttemptStatus::Failed,
        http_status: None,
        error_reason: Some(format!("{}: {}", input.error_type, input.error_reason)),
        retry_reason: None,
        abort_reason: None,
        request_metadata: input.request_metadata,
        response_metadata,
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
    let mut response_metadata = failed_response_metadata(
        failure.started_at_unix_ms,
        failure.finished_at_unix_ms,
        failure.error_type,
    );
    if let Some(attempt) = failure.attempt {
        copy_loop_response_metadata(&attempt.response_metadata, &mut response_metadata);
    }
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
        response_metadata,
        raw_payloads: RawPayloads::default(),
    };
    record_observability(store, &request_record, failure.attempt);
}

fn copy_loop_response_metadata(
    source: &BTreeMap<String, String>,
    target: &mut BTreeMap<String, String>,
) {
    for (key, value) in source {
        if key.starts_with("loop_") {
            target.insert(key.clone(), value.clone());
        }
    }
}

fn record_observability(
    store: &ObservabilityStore,
    request: &RequestRecord,
    attempt: Option<&AttemptRecord>,
) {
    let attempts = attempt.into_iter().cloned().collect::<Vec<_>>();
    record_observability_many(store, request, &attempts);
}

fn record_observability_many(
    store: &ObservabilityStore,
    request: &RequestRecord,
    attempts: &[AttemptRecord],
) {
    if let Err(error) = store.record_request(request) {
        eprintln!("failed to write request observability: {error}");
        return;
    }
    for attempt in attempts {
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
