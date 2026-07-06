use std::{
    collections::{BTreeMap, HashMap, HashSet, hash_map::DefaultHasher},
    convert::Infallible,
    fmt,
    future::Future,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    pin::Pin,
    process::Stdio,
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
        header::{
            ACCEPT, AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, RETRY_AFTER,
        },
    },
    routing::get,
};
use bytes::BytesMut;
use futures_util::{Stream, StreamExt};
use llm_guard_proxy_core::{
    AppConfig, AttemptId, AttemptRecord, AttemptStatus, ConfigHandle, DEFAULT_PROFILE_NAME,
    DebugRequestSummary, DownstreamDropPolicy, DownstreamMode, EvidenceAttemptRecord,
    EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceGroupRecord, EvidenceStore,
    EvidenceStoreWrite, GuardExecutor, GuardOutcome, Health, HeartbeatMode, LICENSE,
    LatencyHistogram, ListenerConfig, MetadataConfig, ObservabilityMetricsSnapshot,
    ObservabilityStore, ProfileConfig, RawPayloads, RequestId, RequestRecord, RequestStatus,
    RetryConfig, RetryLadderConfig, SERVICE_NAME, SelectedUpstreamProfile, ShadowSkipReason,
    ThinkingConfig, ThinkingMode, UpstreamMode, UpstreamProfileConfig, UpstreamRouteReason,
    UpstreamStallConfig, redact_upstream_base_url, validate_upstream_base_url,
};
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
use tokio::{
    net::TcpListener,
    process::Command,
    sync::{Mutex as AsyncMutex, Notify, oneshot},
    time::{Instant, Interval, MissedTickBehavior, timeout},
};

mod model_metadata;
mod shielded_chat;

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_REPEAT_FINGERPRINT_ENTRIES: usize = 1024;
const HEADER_VALUE_NOT_UTF8: &str = "[non-utf8]";
const HEADER_VALUE_REDACTED: &str = "[redacted]";
const DEBUG_SUMMARY_PATH: &str = "/debug/recent-requests";
const IN_FLIGHT_CAPACITY_RECHECK_INTERVAL: Duration = Duration::from_millis(100);
const ADMISSION_RETRY_AFTER_SECS: u32 = 1;
const RECOVERY_PROCESS_GROUP_TERM_GRACE: Duration = Duration::from_millis(100);
const RECOVERY_PROCESS_GROUP_KILL_REAP_GRACE: Duration = Duration::from_millis(500);
const MAX_DENIED_MODEL_ID_BYTES: usize = 128;

/// Shared HTTP proxy state.
#[derive(Clone, Debug)]
pub(crate) struct ProxyState {
    config: ConfigHandle,
    config_path: PathBuf,
    listener: ListenerConfig,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    client: Client,
    generation_requests: Arc<InFlightLimiter>,
    generation_body_routing_requests: Arc<InFlightLimiter>,
    generation_profile_requests: Arc<Mutex<HashMap<String, Arc<InFlightLimiter>>>>,
    control_plane_requests: Arc<InFlightLimiter>,
    upstream_stall_recovery: Arc<UpstreamStallRecoveryCoordinator>,
    repeat_inputs: Arc<RepeatInputCache>,
    shadow_attempts: Arc<InFlightLimiter>,
}

impl ProxyState {
    /// Builds cloneable proxy state for axum handlers.
    #[must_use]
    pub(crate) fn new(
        config: ConfigHandle,
        config_path: PathBuf,
        store: ObservabilityStore,
        evidence_store: EvidenceStore,
        client: Client,
    ) -> Self {
        Self {
            config,
            config_path,
            listener: ListenerConfig {
                name: String::from("default"),
                bind_host: String::from("0.0.0.0"),
                port: 0,
                allowed_upstreams: None,
            },
            store,
            evidence_store,
            client,
            generation_requests: Arc::new(InFlightLimiter::default()),
            generation_body_routing_requests: Arc::new(InFlightLimiter::default()),
            generation_profile_requests: Arc::new(Mutex::new(HashMap::new())),
            control_plane_requests: Arc::new(InFlightLimiter::default()),
            upstream_stall_recovery: Arc::new(UpstreamStallRecoveryCoordinator::default()),
            repeat_inputs: Arc::new(RepeatInputCache::default()),
            shadow_attempts: Arc::new(InFlightLimiter::default()),
        }
    }

    /// Returns a clone of this shared proxy state scoped to one downstream listener.
    #[must_use]
    pub(crate) fn for_listener(&self, listener: ListenerConfig) -> Self {
        let mut state = self.clone();
        state.listener = listener;
        state
    }

    async fn acquire_generation_permit(
        &self,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        self.acquire_generation_permit_with_limiter(
            Arc::clone(&self.generation_requests),
            record_context,
        )
        .await
    }

    async fn acquire_generation_body_routing_permit(
        &self,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        self.acquire_generation_permit_with_limiter(
            Arc::clone(&self.generation_body_routing_requests),
            record_context,
        )
        .await
    }

    async fn acquire_generation_permit_with_limiter(
        &self,
        limiter: Arc<InFlightLimiter>,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        let config = self
            .config
            .snapshot()
            .map_err(|error| AdmissionFailure::ConfigSnapshot(error.to_string()))?;
        if let Some(permit) = limiter.try_acquire(config.server.max_in_flight_requests) {
            return Ok(GenerationAdmission::acquired(
                config,
                permit,
                Duration::ZERO,
            ));
        }

        let Some(queue_permit) = limiter.try_enqueue(config.server.max_queued_generation_requests)
        else {
            return Err(AdmissionFailure::GenerationQueueFull {
                max_queued_generation_requests: config.server.max_queued_generation_requests,
                status: config.server.generation_queue_full_status,
                retry_after_secs: config.server.generation_queue_retry_after_secs,
            });
        };

        self.wait_for_generation_capacity(
            limiter,
            queue_permit,
            config.server.generation_queue_timeout_ms,
            record_context,
        )
        .await
    }

    async fn wait_for_generation_capacity(
        &self,
        limiter: Arc<InFlightLimiter>,
        _queue_permit: QueuedAdmissionPermit,
        timeout_ms: u64,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        let queued_at = Instant::now();
        let mut cancel_recorder =
            QueuedAdmissionCancelRecorder::new(record_context, queued_at, timeout_ms);
        let deadline = queued_at + Duration::from_millis(timeout_ms);
        loop {
            let config = match self.config.snapshot() {
                Ok(config) => config,
                Err(error) => {
                    cancel_recorder.disarm();
                    return Err(AdmissionFailure::ConfigSnapshot(error.to_string()));
                }
            };
            if let Some(permit) = limiter.try_acquire(config.server.max_in_flight_requests) {
                let queue_wait = queued_at.elapsed();
                cancel_recorder.disarm();
                return Ok(GenerationAdmission::acquired(config, permit, queue_wait));
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let queue_wait_ms = duration_millis_u64(queued_at.elapsed());
                cancel_recorder.disarm();
                return Err(AdmissionFailure::GenerationQueueTimeout {
                    generation_queue_timeout_ms: timeout_ms,
                    queue_wait_ms,
                });
            }

            limiter
                .wait_for_capacity(remaining.min(IN_FLIGHT_CAPACITY_RECHECK_INTERVAL))
                .await;
        }
    }

    async fn acquire_generation_permit_for_model(
        &self,
        model_id: Option<&str>,
        body_routing_permit: InFlightPermit,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        let config = self
            .config
            .snapshot()
            .map_err(|error| AdmissionFailure::ConfigSnapshot(error.to_string()))?;
        let selected_profile = select_allowed_upstream_profile(&config, &self.listener, model_id)
            .map_err(AdmissionFailure::ListenerUpstreamDenied)?;
        let limiter = self.generation_limiter_for_profile(&selected_profile.profile);
        if let Some(permit) = limiter.try_acquire(
            selected_profile
                .profile
                .effective_max_in_flight_requests(&config.server),
        ) {
            drop(body_routing_permit);
            return Ok(GenerationAdmission::acquired(
                config,
                permit,
                Duration::ZERO,
            ));
        }

        let max_queued_generation_requests = selected_profile
            .profile
            .effective_max_queued_generation_requests(&config.server);
        let Some(queue_permit) = limiter.try_enqueue(max_queued_generation_requests) else {
            drop(body_routing_permit);
            return Err(AdmissionFailure::GenerationQueueFull {
                max_queued_generation_requests,
                status: config.server.generation_queue_full_status,
                retry_after_secs: config.server.generation_queue_retry_after_secs,
            });
        };

        drop(body_routing_permit);

        self.wait_for_profile_generation_capacity(
            queue_permit,
            model_id.map(str::to_owned),
            config.server.generation_queue_timeout_ms,
            record_context,
        )
        .await
    }

    async fn wait_for_profile_generation_capacity(
        &self,
        _queue_permit: QueuedAdmissionPermit,
        model_id: Option<String>,
        timeout_ms: u64,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        let queued_at = Instant::now();
        let mut cancel_recorder =
            QueuedAdmissionCancelRecorder::new(record_context, queued_at, timeout_ms);
        let deadline = queued_at + Duration::from_millis(timeout_ms);
        loop {
            let config = match self.config.snapshot() {
                Ok(config) => config,
                Err(error) => {
                    cancel_recorder.disarm();
                    return Err(AdmissionFailure::ConfigSnapshot(error.to_string()));
                }
            };
            let selected_profile =
                match select_allowed_upstream_profile(&config, &self.listener, model_id.as_deref())
                {
                    Ok(selected_profile) => selected_profile,
                    Err(error) => {
                        cancel_recorder.disarm();
                        return Err(AdmissionFailure::ListenerUpstreamDenied(error));
                    }
                };
            let limiter = self.generation_limiter_for_profile(&selected_profile.profile);
            if let Some(permit) = limiter.try_acquire(
                selected_profile
                    .profile
                    .effective_max_in_flight_requests(&config.server),
            ) {
                let queue_wait = queued_at.elapsed();
                cancel_recorder.disarm();
                return Ok(GenerationAdmission::acquired(config, permit, queue_wait));
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let queue_wait_ms = duration_millis_u64(queued_at.elapsed());
                cancel_recorder.disarm();
                return Err(AdmissionFailure::GenerationQueueTimeout {
                    generation_queue_timeout_ms: timeout_ms,
                    queue_wait_ms,
                });
            }

            limiter
                .wait_for_capacity(remaining.min(IN_FLIGHT_CAPACITY_RECHECK_INTERVAL))
                .await;
        }
    }

    fn generation_limiter_for_profile(
        &self,
        profile: &UpstreamProfileConfig,
    ) -> Arc<InFlightLimiter> {
        if !profile.has_generation_limits() {
            return Arc::clone(&self.generation_requests);
        }

        let mut limiters = generation_profile_limiters(&self.generation_profile_requests);
        Arc::clone(
            limiters
                .entry(profile.name.clone())
                .or_insert_with(|| Arc::new(InFlightLimiter::default())),
        )
    }

    fn try_acquire_control_plane_permit(
        &self,
        max_control_plane_in_flight_requests: usize,
    ) -> Result<InFlightPermit, AdmissionFailure> {
        self.control_plane_requests
            .try_acquire(max_control_plane_in_flight_requests)
            .ok_or(AdmissionFailure::ControlPlaneLimitExceeded {
                max_control_plane_in_flight_requests,
            })
    }

    fn admission_metrics_snapshot(&self) -> AdmissionMetricsSnapshot {
        let generation = self.generation_requests.snapshot_counts();
        let mut profiles = generation_profile_limiters(&self.generation_profile_requests)
            .iter()
            .map(|(profile, limiter)| ProfileAdmissionMetrics {
                profile: profile.clone(),
                counts: limiter.snapshot_counts(),
            })
            .collect::<Vec<_>>();
        profiles.sort_by(|left, right| left.profile.cmp(&right.profile));
        AdmissionMetricsSnapshot {
            generation,
            profiles,
        }
    }
}

/// Serves the proxy until the supplied shutdown future resolves.
///
/// Axum stops accepting new connections after shutdown starts and waits for
/// in-flight response bodies to finish or be dropped.
pub(crate) async fn serve_until_shutdown(
    listener: TcpListener,
    state: ProxyState,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

#[derive(Debug, Default)]
struct InFlightLimiter {
    counts: Mutex<AdmissionCounts>,
    notify: Notify,
}

impl InFlightLimiter {
    fn try_acquire(self: &Arc<Self>, max_in_flight_requests: usize) -> Option<InFlightPermit> {
        let mut counts = admission_counts(&self.counts);
        if counts.active >= max_in_flight_requests {
            return None;
        }

        counts.active = counts.active.saturating_add(1);
        Some(InFlightPermit::limited(Arc::clone(self)))
    }

    fn try_enqueue(self: &Arc<Self>, max_queued_requests: usize) -> Option<QueuedAdmissionPermit> {
        let mut counts = admission_counts(&self.counts);
        if counts.queued >= max_queued_requests {
            return None;
        }

        counts.queued = counts.queued.saturating_add(1);
        Some(QueuedAdmissionPermit {
            limiter: Arc::clone(self),
        })
    }

    async fn wait_for_capacity(&self, max_wait: Duration) {
        tokio::select! {
            () = self.notify.notified() => {}
            () = tokio::time::sleep(max_wait) => {}
        }
    }

    fn release(&self) {
        let mut counts = admission_counts(&self.counts);
        counts.active = counts.active.saturating_sub(1);
        self.notify.notify_waiters();
    }

    fn leave_queue(&self) {
        let mut counts = admission_counts(&self.counts);
        counts.queued = counts.queued.saturating_sub(1);
        self.notify.notify_waiters();
    }

    fn snapshot_counts(&self) -> AdmissionCounts {
        *admission_counts(&self.counts)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AdmissionCounts {
    active: usize,
    queued: usize,
}

#[derive(Debug)]
struct AdmissionMetricsSnapshot {
    generation: AdmissionCounts,
    profiles: Vec<ProfileAdmissionMetrics>,
}

#[derive(Debug)]
struct ProfileAdmissionMetrics {
    profile: String,
    counts: AdmissionCounts,
}

#[derive(Debug)]
struct GenerationAdmission {
    config: AppConfig,
    permit: InFlightPermit,
    queue_wait: Duration,
}

impl GenerationAdmission {
    fn acquired(config: AppConfig, permit: InFlightPermit, queue_wait: Duration) -> Self {
        Self {
            config,
            permit,
            queue_wait,
        }
    }
}

#[derive(Clone, Debug)]
struct AdmissionRecordContext {
    store: ObservabilityStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    request_metadata: BTreeMap<String, String>,
}

#[derive(Debug)]
struct QueuedAdmissionCancelRecorder {
    record: Option<QueuedAdmissionCancelRecord>,
}

impl QueuedAdmissionCancelRecorder {
    fn new(context: AdmissionRecordContext, queued_at: Instant, timeout_ms: u64) -> Self {
        Self {
            record: Some(QueuedAdmissionCancelRecord {
                context,
                queued_at,
                timeout_ms,
            }),
        }
    }

    fn disarm(&mut self) {
        self.record = None;
    }
}

impl Drop for QueuedAdmissionCancelRecorder {
    fn drop(&mut self) {
        if let Some(record) = self.record.take() {
            record_queued_admission_cancel(record);
        }
    }
}

#[derive(Debug)]
struct QueuedAdmissionCancelRecord {
    context: AdmissionRecordContext,
    queued_at: Instant,
    timeout_ms: u64,
}

fn admission_counts(current: &Mutex<AdmissionCounts>) -> MutexGuard<'_, AdmissionCounts> {
    match current.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn generation_profile_limiters(
    current: &Mutex<HashMap<String, Arc<InFlightLimiter>>>,
) -> MutexGuard<'_, HashMap<String, Arc<InFlightLimiter>>> {
    match current.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Debug)]
struct InFlightPermit {
    limiter: Option<Arc<InFlightLimiter>>,
}

impl InFlightPermit {
    fn limited(limiter: Arc<InFlightLimiter>) -> Self {
        Self {
            limiter: Some(limiter),
        }
    }
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        if let Some(limiter) = &self.limiter {
            limiter.release();
        }
    }
}

#[derive(Debug)]
struct QueuedAdmissionPermit {
    limiter: Arc<InFlightLimiter>,
}

impl Drop for QueuedAdmissionPermit {
    fn drop(&mut self) {
        self.limiter.leave_queue();
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
enum AdmissionFailure {
    #[error("failed to read proxy config snapshot: {0}")]
    ConfigSnapshot(String),
    #[error(
        "proxy generation request queue is full: max_queued_generation_requests={max_queued_generation_requests}"
    )]
    GenerationQueueFull {
        max_queued_generation_requests: usize,
        status: u16,
        retry_after_secs: Option<u32>,
    },
    #[error(
        "proxy generation request queue wait timed out: generation_queue_timeout_ms={generation_queue_timeout_ms}"
    )]
    GenerationQueueTimeout {
        generation_queue_timeout_ms: u64,
        queue_wait_ms: u64,
    },
    #[error(
        "proxy control-plane request limit exceeded: max_control_plane_in_flight_requests={max_control_plane_in_flight_requests}"
    )]
    ControlPlaneLimitExceeded {
        max_control_plane_in_flight_requests: usize,
    },
    #[error("{0}")]
    ListenerUpstreamDenied(ListenerUpstreamDenied),
}

impl AdmissionFailure {
    fn status(&self) -> StatusCode {
        match self {
            Self::ConfigSnapshot(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ListenerUpstreamDenied(_) => StatusCode::BAD_REQUEST,
            Self::GenerationQueueFull { status, .. } => match StatusCode::from_u16(*status) {
                Ok(status) => status,
                Err(_error) => StatusCode::SERVICE_UNAVAILABLE,
            },
            Self::GenerationQueueTimeout { .. } | Self::ControlPlaneLimitExceeded { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
        }
    }

    const fn error_type(&self) -> &'static str {
        match self {
            Self::ConfigSnapshot(_) => "config_snapshot_failed",
            Self::GenerationQueueFull { .. } => "proxy_generation_queue_full",
            Self::GenerationQueueTimeout { .. } => "proxy_generation_queue_timeout",
            Self::ListenerUpstreamDenied(_) => "listener_upstream_not_allowed",
            Self::ControlPlaneLimitExceeded { .. } => {
                "proxy_control_plane_in_flight_limit_exceeded"
            }
        }
    }

    fn retry_after(&self) -> Option<String> {
        match self {
            Self::ConfigSnapshot(_) | Self::ListenerUpstreamDenied(_) => None,
            Self::GenerationQueueFull {
                retry_after_secs, ..
            } => Some(
                retry_after_secs
                    .unwrap_or(ADMISSION_RETRY_AFTER_SECS)
                    .to_string(),
            ),
            Self::GenerationQueueTimeout { .. } | Self::ControlPlaneLimitExceeded { .. } => {
                Some(ADMISSION_RETRY_AFTER_SECS.to_string())
            }
        }
    }

    fn request_metadata(&self) -> BTreeMap<String, String> {
        match self {
            Self::GenerationQueueFull {
                max_queued_generation_requests,
                ..
            } => BTreeMap::from([
                (
                    String::from("admission_outcome"),
                    String::from("queue_full_rejected"),
                ),
                (String::from("queue_wait_ms"), String::from("0")),
                (
                    String::from("max_queued_generation_requests"),
                    max_queued_generation_requests.to_string(),
                ),
            ]),
            Self::GenerationQueueTimeout {
                generation_queue_timeout_ms,
                queue_wait_ms,
            } => BTreeMap::from([
                (
                    String::from("admission_outcome"),
                    String::from("queue_timeout"),
                ),
                (String::from("queue_wait_ms"), queue_wait_ms.to_string()),
                (
                    String::from("generation_queue_timeout_ms"),
                    generation_queue_timeout_ms.to_string(),
                ),
            ]),
            Self::ConfigSnapshot(_)
            | Self::ListenerUpstreamDenied(_)
            | Self::ControlPlaneLimitExceeded { .. } => BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error(
    "listener {listener_name} on port {listener_port} does not allow upstream profile {upstream_profile} for model {model_id}"
)]
struct ListenerUpstreamDenied {
    listener_name: String,
    listener_port: u16,
    upstream_profile: String,
    model_id: String,
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
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Builds the OpenAI-compatible proxy router.
pub(crate) fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/config-summary", get(config_summary_handler))
        .fallback(proxy_handler)
        .with_state(state)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HealthUpstreamStatus {
    Disabled,
    Ready,
    Unavailable,
}

impl HealthUpstreamStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "not_checked",
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }

    const fn http_status(self) -> StatusCode {
        match self {
            Self::Disabled | Self::Ready => StatusCode::OK,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
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

async fn config_summary_handler(State(state): State<ProxyState>) -> Response<Body> {
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

async fn health_handler(State(state): State<ProxyState>) -> Response<Body> {
    match state.config.snapshot() {
        Ok(config) => {
            let upstream = probe_upstream_readiness(&state, &config).await;
            let status = upstream.http_status();
            json_response(
                status,
                json!({
                    "service": SERVICE_NAME,
                    "process": "alive",
                    "readiness": Health::current().readiness().as_str(),
                    "upstream": upstream.as_str(),
                    "upstream_probe_enabled": config
                        .observability
                        .health_upstream_probe_enabled
                        .is_enabled(),
                    "observability_enabled": config.observability.enabled,
                    "request_id": RequestId::generate().as_str(),
                })
                .to_string(),
            )
        }
        Err(error) => proxy_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "config_snapshot_failed",
            &error.to_string(),
        ),
    }
}

async fn metrics_handler(State(state): State<ProxyState>) -> Response<Body> {
    match state.config.snapshot() {
        Ok(config) if config.observability.metrics_enabled.is_enabled() => {
            match state.store.metrics_snapshot() {
                Ok(snapshot) => {
                    let admission = state.admission_metrics_snapshot();
                    text_response(StatusCode::OK, render_metrics(&snapshot, &admission))
                }
                Err(error) => proxy_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "metrics_snapshot_failed",
                    &error.to_string(),
                ),
            }
        }
        Ok(_config) => proxy_error_response(
            StatusCode::NOT_FOUND,
            "metrics_disabled",
            "metrics endpoint is disabled",
        ),
        Err(error) => proxy_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "config_snapshot_failed",
            &error.to_string(),
        ),
    }
}

async fn probe_upstream_readiness(state: &ProxyState, config: &AppConfig) -> HealthUpstreamStatus {
    if !config
        .observability
        .health_upstream_probe_enabled
        .is_enabled()
    {
        return HealthUpstreamStatus::Disabled;
    }
    let uri = Uri::from_static("/v1/models");
    let Ok(url) = build_upstream_url(&config.upstream.base_url, &uri) else {
        return HealthUpstreamStatus::Unavailable;
    };
    let timeout = Duration::from_millis(config.observability.health_upstream_probe_timeout_ms);
    match tokio::time::timeout(timeout, state.client.get(url).send()).await {
        Ok(Ok(response)) if response.status().is_success() => HealthUpstreamStatus::Ready,
        Ok(Ok(response)) if response.status().as_u16() == StatusCode::UNAUTHORIZED.as_u16() => {
            HealthUpstreamStatus::Ready
        }
        _ => HealthUpstreamStatus::Unavailable,
    }
}

fn is_configured_debug_summary_request(state: &ProxyState, uri: &Uri) -> bool {
    uri.path() == DEBUG_SUMMARY_PATH && state.config.snapshot().is_ok()
}

fn debug_summary_response(state: &ProxyState, request: &Request<Body>) -> Response<Body> {
    let config = match state.config.snapshot() {
        Ok(config) => config,
        Err(error) => {
            return proxy_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "config_snapshot_failed",
                &error.to_string(),
            );
        }
    };
    if !config.observability.debug_summary_enabled.is_enabled() {
        return proxy_error_response(
            StatusCode::NOT_FOUND,
            "debug_summary_disabled",
            "debug summary endpoint is disabled",
        );
    }
    if !debug_summary_authorized(
        request.headers(),
        config.observability.debug_summary_admin_token.as_deref(),
    ) {
        return proxy_error_response(
            StatusCode::UNAUTHORIZED,
            "debug_summary_unauthorized",
            "debug summary authorization failed",
        );
    }
    let limit = debug_summary_limit(
        request.uri(),
        config.observability.debug_summary_max_records,
        config.observability.debug_summary_max_records,
    );
    match state.store.recent_request_summaries(limit) {
        Ok(summaries) => json_response(
            StatusCode::OK,
            render_debug_summary_json(limit, &summaries).to_string(),
        ),
        Err(error) => proxy_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "debug_summary_failed",
            &error.to_string(),
        ),
    }
}

fn debug_summary_authorized(headers: &HeaderMap, token: Option<&str>) -> bool {
    let Some(token) = token.filter(|token| !token.is_empty()) else {
        return true;
    };
    let bearer_matches = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|value| admin_token_matches(value, token));
    let header_matches = headers
        .get(HeaderName::from_static("x-admin-token"))
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| admin_token_matches(value, token));
    bearer_matches || header_matches
}

fn admin_token_matches(candidate: &str, expected: &str) -> bool {
    let candidate = candidate.as_bytes();
    let expected = expected.as_bytes();
    let mut diff = candidate.len() ^ expected.len();

    for (index, expected_byte) in expected.iter().copied().enumerate() {
        let candidate_byte = candidate.get(index).copied().unwrap_or(0);
        diff |= usize::from(candidate_byte ^ expected_byte);
    }

    diff == 0
}

fn debug_summary_limit(uri: &Uri, default_limit: u32, max_limit: u32) -> u32 {
    let bounded_default = default_limit.clamp(1, max_limit.max(1));
    let Some(query) = uri.query() else {
        return bounded_default;
    };
    query
        .split('&')
        .filter_map(|part| part.split_once('='))
        .find_map(|(key, value)| {
            if key == "limit" {
                value.parse::<u32>().ok()
            } else {
                None
            }
        })
        .map_or(bounded_default, |limit| limit.clamp(1, max_limit.max(1)))
}

fn render_debug_summary_json(limit: u32, summaries: &[DebugRequestSummary]) -> serde_json::Value {
    let requests = summaries
        .iter()
        .map(|summary| {
            json!({
                "request_id": summary.request_id.as_str(),
                "started_at_unix_ms": summary.started_at_unix_ms,
                "finished_at_unix_ms": summary.finished_at_unix_ms,
                "duration_ms": summary.duration_ms,
                "downstream_mode": summary.downstream_mode.as_str(),
                "upstream_mode": summary.upstream_mode.as_str(),
                "model_id": summary.model_id.as_deref(),
                "status": summary.status.as_str(),
                "http_status": summary.http_status,
                "error_reason": summary.error_reason.as_deref(),
                "abort_reason": summary.abort_reason.as_deref(),
                "attempt_count": summary.attempt_count,
                "retry_count": summary.retry_count,
                "loop_detected": summary.loop_detected,
                "request_metadata": &summary.request_metadata,
                "response_metadata": &summary.response_metadata,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "limit": limit,
        "request_count": summaries.len(),
        "redaction": "raw prompts, raw outputs, and sensitive headers are omitted or redacted",
        "requests": requests,
    })
}

fn render_metrics(
    snapshot: &ObservabilityMetricsSnapshot,
    admission: &AdmissionMetricsSnapshot,
) -> String {
    let mut output = String::new();
    push_admission_metrics(&mut output, admission);
    push_request_metrics(&mut output, snapshot);
    push_attempt_metrics(&mut output, snapshot);
    push_retry_and_error_metrics(&mut output, snapshot);
    push_latency_metrics(&mut output, snapshot);
    push_heartbeat_metrics(&mut output, snapshot);
    push_storage_metrics(&mut output, snapshot);
    output
}

fn push_admission_metrics(output: &mut String, admission: &AdmissionMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_generation_active",
        "Current generation requests holding an admitted upstream slot.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_generation_active",
        &[],
        usize_to_u64(admission.generation.active),
    );
    push_metric_header(
        output,
        "llm_guard_proxy_generation_queued",
        "Current generation requests waiting for an upstream slot.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_generation_queued",
        &[],
        usize_to_u64(admission.generation.queued),
    );
    push_metric_header(
        output,
        "llm_guard_proxy_generation_profile_active",
        "Current per-profile generation requests holding an admitted upstream slot.",
        "gauge",
    );
    for profile in &admission.profiles {
        push_metric_line(
            output,
            "llm_guard_proxy_generation_profile_active",
            &[("profile", &profile.profile)],
            usize_to_u64(profile.counts.active),
        );
    }
    push_metric_header(
        output,
        "llm_guard_proxy_generation_profile_queued",
        "Current per-profile generation requests waiting for an upstream slot.",
        "gauge",
    );
    for profile in &admission.profiles {
        push_metric_line(
            output,
            "llm_guard_proxy_generation_profile_queued",
            &[("profile", &profile.profile)],
            usize_to_u64(profile.counts.queued),
        );
    }
}

fn push_request_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_requests",
        "Currently retained proxy request rows by bounded lifecycle labels.",
        "gauge",
    );
    for row in &snapshot.request_counts {
        push_metric_line(
            output,
            "llm_guard_proxy_current_retained_requests",
            &[
                ("status", &row.status),
                ("downstream_mode", &row.downstream_mode),
                ("upstream_mode", &row.upstream_mode),
                ("http_status_class", &row.http_status_class),
            ],
            row.count,
        );
    }
}

fn push_attempt_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_attempts",
        "Currently retained proxy upstream attempts by bounded lifecycle labels.",
        "gauge",
    );
    for row in &snapshot.attempt_counts {
        push_metric_line(
            output,
            "llm_guard_proxy_current_retained_attempts",
            &[
                ("status", &row.status),
                ("upstream_mode", &row.upstream_mode),
                ("http_status_class", &row.http_status_class),
            ],
            row.count,
        );
    }
}

fn push_retry_and_error_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_retries",
        "Currently retained attempts retried or marked with retry reasons.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_current_retained_retries",
        &[],
        snapshot.retry_count,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_loop_aborts",
        "Currently retained loop-guard abort observations.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_current_retained_loop_aborts",
        &[],
        snapshot.loop_abort_count,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_upstream_errors",
        "Currently retained upstream error observations by bounded error bucket.",
        "gauge",
    );
    for row in &snapshot.upstream_error_counts {
        push_metric_line(
            output,
            "llm_guard_proxy_current_retained_upstream_errors",
            &[
                ("kind", &row.kind),
                ("http_status_class", &row.http_status_class),
            ],
            row.count,
        );
    }
}

fn push_latency_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_latency_distribution_gauges(
        output,
        "llm_guard_proxy_current_retained_first_token_latency_ms",
        "first-token latency in milliseconds for shielded attempts",
        &snapshot.first_token_latency_ms,
    );
    push_latency_distribution_gauges(
        output,
        "llm_guard_proxy_current_retained_total_latency_ms",
        "end-to-end request latency in milliseconds",
        &snapshot.total_latency_ms,
    );
}

fn push_heartbeat_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_heartbeat_modes",
        "Currently retained downstream heartbeat/liveness mode counts.",
        "gauge",
    );
    for row in &snapshot.heartbeat_mode_counts {
        push_metric_line(
            output,
            "llm_guard_proxy_current_retained_heartbeat_modes",
            &[("mode", &row.mode)],
            row.count,
        );
    }
}

fn push_storage_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_observability_records",
        "Currently retained observability records.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_observability_records",
        &[],
        snapshot.retention_usage.record_count,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_observability_storage_bytes",
        "SQLite bytes used by the observability store.",
        "gauge",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_observability_storage_bytes",
        &[],
        snapshot.retention_usage.observed_bytes,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_storage_pruning_events_total",
        "Retention pruning events that removed rows.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_storage_pruning_events_total",
        &[],
        snapshot.pruning.prune_events,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_storage_pruned_requests_total",
        "Request rows removed by retention pruning.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_storage_pruned_requests_total",
        &[],
        snapshot.pruning.pruned_requests,
    );
    push_metric_header(
        output,
        "llm_guard_proxy_storage_pruned_attempts_total",
        "Attempt rows removed by retention pruning.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_storage_pruned_attempts_total",
        &[],
        snapshot.pruning.pruned_attempts,
    );
}

fn push_latency_distribution_gauges(
    output: &mut String,
    name: &str,
    help: &str,
    histogram: &LatencyHistogram,
) {
    let le_metric = format!("{name}_le");
    push_metric_header(
        output,
        &le_metric,
        &format!("Currently retained {help} observations less than or equal to the bound."),
        "gauge",
    );
    for bucket in &histogram.buckets {
        let le = bucket.le_ms.to_string();
        push_metric_line(output, &le_metric, &[("le", &le)], bucket.count);
    }
    push_metric_line(output, &le_metric, &[("le", "+Inf")], histogram.count);

    let observations_metric = format!("{name}_observations");
    push_metric_header(
        output,
        &observations_metric,
        &format!("Currently retained {help} observation count."),
        "gauge",
    );
    push_metric_line(output, &observations_metric, &[], histogram.count);

    let sum_metric = format!("{name}_sum_ms");
    push_metric_header(
        output,
        &sum_metric,
        &format!("Currently retained {help} sum."),
        "gauge",
    );
    push_metric_line(output, &sum_metric, &[], histogram.sum_ms);
}

fn push_metric_header(output: &mut String, name: &str, help: &str, metric_type: &str) {
    output.push_str("# HELP ");
    output.push_str(name);
    output.push(' ');
    output.push_str(help);
    output.push('\n');
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push(' ');
    output.push_str(metric_type);
    output.push('\n');
}

fn push_metric_line(output: &mut String, name: &str, labels: &[(&str, &str)], value: u64) {
    output.push_str(name);
    if !labels.is_empty() {
        output.push('{');
        for (index, (key, value)) in labels.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            output.push_str(key);
            output.push_str("=\"");
            output.push_str(&prometheus_escape_label(value));
            output.push('"');
        }
        output.push('}');
    }
    output.push(' ');
    output.push_str(&value.to_string());
    output.push('\n');
}

fn prometheus_escape_label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '"' => escaped.push_str("\\\""),
            _ => escaped.push(character),
        }
    }
    escaped
}

async fn proxy_handler(State(state): State<ProxyState>, request: Request<Body>) -> Response<Body> {
    if request.method() == Method::GET && is_configured_debug_summary_request(&state, request.uri())
    {
        return debug_summary_response(&state, &request);
    }

    let request_id = RequestId::generate();
    let started_at_unix_ms = unix_time_millis();
    if let Err(error) = validate_openai_path(request.uri().path()) {
        let finished_at_unix_ms = unix_time_millis();
        let error_type = error.error_type();
        let error_reason = error.to_string();
        let response = proxy_error_response(error.status(), error_type, &error_reason);
        let mut request_metadata = pre_upstream_request_metadata(
            request.method(),
            request.uri(),
            request.headers(),
            config_shielding_enabled(&state.config),
        );
        add_listener_metadata(&mut request_metadata, &state.listener);
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
                attempts: Vec::new(),
            },
        );
        return response;
    }

    let admission_request = AdmissionRequestMetadata::from_request(&request);
    let admission =
        match admit_request(&state, &request_id, started_at_unix_ms, admission_request).await {
            AdmissionOutcome::Accepted(admission) => *admission,
            AdmissionOutcome::Rejected(response) => return response,
        };

    match forward_openai_request(
        &state,
        &request_id,
        started_at_unix_ms,
        request,
        admission.permit,
        admission.permit_kind,
        admission.admission_metadata,
        admission.config.server.max_request_body_bytes,
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_type = error.error_type();
            let error_reason = error.to_string();
            let response = proxy_error_response_from_error(&error);
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
                    attempts: error.attempt_records(),
                },
            );
            response
        }
    }
}

struct RequestAdmission {
    config: AppConfig,
    permit: InFlightPermit,
    permit_kind: AdmissionPermitKind,
    admission_metadata: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdmissionPermitKind {
    ControlPlane,
    Generation,
    BodyRouting,
}

struct AdmissionRequestMetadata {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
}

impl AdmissionRequestMetadata {
    fn from_request(request: &Request<Body>) -> Self {
        Self {
            method: request.method().clone(),
            uri: request.uri().clone(),
            headers: request.headers().clone(),
        }
    }

    fn pre_upstream_metadata(&self, shielding_enabled: Option<bool>) -> BTreeMap<String, String> {
        pre_upstream_request_metadata(&self.method, &self.uri, &self.headers, shielding_enabled)
    }
}

enum AdmissionOutcome {
    Accepted(Box<RequestAdmission>),
    Rejected(Response<Body>),
}

#[allow(clippy::too_many_lines)]
async fn admit_request(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    request: AdmissionRequestMetadata,
) -> AdmissionOutcome {
    let config = match state.config.snapshot() {
        Ok(config) => config,
        Err(error) => {
            let error_type = "config_snapshot_failed";
            let error_reason = error.to_string();
            let response =
                proxy_error_response(StatusCode::INTERNAL_SERVER_ERROR, error_type, &error_reason);
            record_failed_request(
                &state.store,
                FailedRequestRecord {
                    request_id: request_id.clone(),
                    started_at_unix_ms,
                    finished_at_unix_ms: unix_time_millis(),
                    http_status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                    error_type,
                    error_reason,
                    request_metadata: {
                        let mut metadata = request.pre_upstream_metadata(None);
                        add_listener_metadata(&mut metadata, &state.listener);
                        metadata
                    },
                    attempts: Vec::new(),
                },
            );
            return AdmissionOutcome::Rejected(response);
        }
    };

    if is_control_plane_models_request(&request.method, &request.uri) {
        let permit = match state
            .try_acquire_control_plane_permit(config.server.max_control_plane_in_flight_requests)
        {
            Ok(permit) => permit,
            Err(error) => {
                return reject_admission(
                    state,
                    request_id,
                    started_at_unix_ms,
                    &request,
                    Some(config.shielding.enabled),
                    &error,
                );
            }
        };
        return AdmissionOutcome::Accepted(Box::new(RequestAdmission {
            config,
            permit,
            permit_kind: AdmissionPermitKind::ControlPlane,
            admission_metadata: acquired_admission_metadata(Duration::ZERO),
        }));
    }

    if config.has_upstream_profile_generation_limits() {
        let record_context = admission_record_context(
            state,
            request_id,
            started_at_unix_ms,
            &request,
            Some(config.shielding.enabled),
        );
        let admission = match state
            .acquire_generation_body_routing_permit(record_context)
            .await
        {
            Ok(admission) => admission,
            Err(error) => {
                return reject_admission(
                    state,
                    request_id,
                    started_at_unix_ms,
                    &request,
                    Some(config.shielding.enabled),
                    &error,
                );
            }
        };
        return AdmissionOutcome::Accepted(Box::new(RequestAdmission {
            config: admission.config,
            permit: admission.permit,
            permit_kind: AdmissionPermitKind::BodyRouting,
            admission_metadata: prefixed_acquired_admission_metadata(
                "body_routing",
                admission.queue_wait,
            ),
        }));
    }

    let record_context = admission_record_context(
        state,
        request_id,
        started_at_unix_ms,
        &request,
        Some(config.shielding.enabled),
    );
    let admission = match state.acquire_generation_permit(record_context).await {
        Ok(admission) => admission,
        Err(error) => {
            return reject_admission(
                state,
                request_id,
                started_at_unix_ms,
                &request,
                Some(config.shielding.enabled),
                &error,
            );
        }
    };

    AdmissionOutcome::Accepted(Box::new(RequestAdmission {
        config: admission.config,
        permit: admission.permit,
        permit_kind: AdmissionPermitKind::Generation,
        admission_metadata: acquired_admission_metadata(admission.queue_wait),
    }))
}

fn reject_admission(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    request: &AdmissionRequestMetadata,
    shielding_enabled: Option<bool>,
    error: &AdmissionFailure,
) -> AdmissionOutcome {
    let error_type = error.error_type();
    let error_reason = error.to_string();
    let response = admission_error_response(
        error.status(),
        error_type,
        &error_reason,
        error.retry_after(),
    );
    record_failed_request(
        &state.store,
        FailedRequestRecord {
            request_id: request_id.clone(),
            started_at_unix_ms,
            finished_at_unix_ms: unix_time_millis(),
            http_status: error.status().as_u16(),
            error_type,
            error_reason,
            request_metadata: {
                let mut metadata = request.pre_upstream_metadata(shielding_enabled);
                add_listener_metadata(&mut metadata, &state.listener);
                metadata.extend(error.request_metadata());
                metadata
            },
            attempts: Vec::new(),
        },
    );
    AdmissionOutcome::Rejected(response)
}

fn admission_record_context(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    request: &AdmissionRequestMetadata,
    shielding_enabled: Option<bool>,
) -> AdmissionRecordContext {
    let mut request_metadata = request.pre_upstream_metadata(shielding_enabled);
    add_listener_metadata(&mut request_metadata, &state.listener);
    AdmissionRecordContext {
        store: state.store.clone(),
        request_id: request_id.clone(),
        started_at_unix_ms,
        request_metadata,
    }
}

fn acquired_admission_metadata(queue_wait: Duration) -> BTreeMap<String, String> {
    BTreeMap::from([
        (String::from("admission_outcome"), String::from("acquired")),
        (
            String::from("queue_wait_ms"),
            duration_millis_u64(queue_wait).to_string(),
        ),
    ])
}

fn prefixed_acquired_admission_metadata(
    prefix: &str,
    queue_wait: Duration,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            format!("{prefix}_admission_outcome"),
            String::from("acquired"),
        ),
        (
            format!("{prefix}_queue_wait_ms"),
            duration_millis_u64(queue_wait).to_string(),
        ),
    ])
}

fn is_control_plane_models_request(method: &Method, uri: &Uri) -> bool {
    method == Method::GET && uri.path() == "/v1/models"
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn forward_openai_request(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    request: Request<Body>,
    in_flight_permit: InFlightPermit,
    admission_permit_kind: AdmissionPermitKind,
    admission_metadata: BTreeMap<String, String>,
    max_request_body_bytes: usize,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = request.into_parts();
    let method = parts.method;
    let uri = parts.uri;
    let downstream_headers = parts.headers;
    let shielding_enabled_hint = config_shielding_enabled(&state.config);
    let body_admission = read_body_and_admit_generation(
        state,
        body,
        in_flight_permit,
        admission_permit_kind,
        max_request_body_bytes,
        BodyAdmissionContext {
            method: &method,
            uri: &uri,
            downstream_headers: &downstream_headers,
            shielding_enabled_hint,
            admission_metadata,
            request_id,
            started_at_unix_ms,
        },
    )
    .await?;
    let OpenAiBodyAdmission {
        config,
        body,
        in_flight_permit,
        admission_metadata,
    } = body_admission;
    let mut request_metadata = request_metadata(
        &method,
        &uri,
        &downstream_headers,
        body.len(),
        config.shielding.enabled,
    );
    add_listener_metadata(&mut request_metadata, &state.listener);
    request_metadata.extend(admission_metadata);
    let mut prepared_request =
        prepare_openai_forward_request(state, &config, &method, &uri, &body, &mut request_metadata)
            .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    apply_pre_request_guard(&config, request_id, &mut prepared_request)
        .await
        .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    let retry_policy = ShieldedRetryPolicy::from_config(&config.retry);
    let upstream_stall_policy = UpstreamStallPolicy::from_config(&config.upstream_stall);
    let upstream_timeout =
        Duration::from_millis(prepared_request.upstream_profile.request_timeout_ms);
    if prepared_request.shielded_chat_plan.intercepted {
        add_retry_request_metadata(&mut request_metadata, &retry_policy);
        return forward_shielded_chat_with_retries(
            ShieldedRetryRuntime {
                client: state.client.clone(),
                method: prepared_request.reqwest_method,
                upstream_url: prepared_request.upstream_url,
                downstream_method: method,
                downstream_uri: uri,
                downstream_headers,
                upstream_body: prepared_request.shielded_chat_plan.upstream_body,
                downstream_body: prepared_request.shielded_chat_plan.downstream_body,
                chat_kind: prepared_request.shielded_chat_plan.kind,
                upstream_timeout,
                config: state.config.clone(),
                store: state.store.clone(),
                evidence_store: state.evidence_store.clone(),
                request_id: request_id.clone(),
                started_at_unix_ms,
                model_id: prepared_request.model_id,
                request_metadata,
                listener: state.listener.clone(),
                upstream_profile: prepared_request.upstream_profile,
                caller_profile: prepared_request.caller_profile,
                route_reason: prepared_request.route_reason,
                liveness: prepared_request.shielded_chat_plan.liveness,
                thinking_metadata: prepared_request.shielded_chat_plan.thinking_metadata,
                loop_context: prepared_request.shielded_chat_plan.loop_context,
                retry_policy,
                upstream_stall_policy,
                upstream_stall_recovery: state.upstream_stall_recovery.clone(),
                shadow_attempts: state.shadow_attempts.clone(),
                shadow_evidence: ShadowEvidenceState::default(),
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
        reqwest_method: prepared_request.reqwest_method,
        upstream_url: prepared_request.upstream_url,
        upstream_body: prepared_request.shielded_chat_plan.upstream_body,
        upstream_timeout,
        upstream_profile: prepared_request.upstream_profile,
        route_reason: prepared_request.route_reason,
        liveness: prepared_request.shielded_chat_plan.liveness,
        thinking_policy_applied: prepared_request.shielded_chat_plan.thinking_policy_applied,
        thinking_metadata: prepared_request.shielded_chat_plan.thinking_metadata,
        request_id,
        started_at_unix_ms,
        model_id: prepared_request.model_id,
        request_metadata,
        in_flight_permit,
    })
    .await
}

struct OpenAiBodyAdmission {
    config: AppConfig,
    body: Bytes,
    in_flight_permit: InFlightPermit,
    admission_metadata: BTreeMap<String, String>,
}

struct BodyAdmissionContext<'request> {
    method: &'request Method,
    uri: &'request Uri,
    downstream_headers: &'request HeaderMap,
    shielding_enabled_hint: Option<bool>,
    admission_metadata: BTreeMap<String, String>,
    request_id: &'request RequestId,
    started_at_unix_ms: u64,
}

async fn read_body_and_admit_generation(
    state: &ProxyState,
    body: Body,
    in_flight_permit: InFlightPermit,
    admission_permit_kind: AdmissionPermitKind,
    max_request_body_bytes: usize,
    request: BodyAdmissionContext<'_>,
) -> Result<OpenAiBodyAdmission, ProxyError> {
    let mut pre_body_request_metadata = pre_upstream_request_metadata(
        request.method,
        request.uri,
        request.downstream_headers,
        request.shielding_enabled_hint,
    );
    add_listener_metadata(&mut pre_body_request_metadata, &state.listener);
    let body = read_body_bytes(body, max_request_body_bytes)
        .await
        .map_err(|error| error.with_request_metadata(pre_body_request_metadata))?;
    let mut body_read_request_metadata = base_request_metadata(
        request.method,
        request.uri,
        request.downstream_headers,
        body.len().to_string(),
        request.shielding_enabled_hint,
    );
    add_listener_metadata(&mut body_read_request_metadata, &state.listener);
    let config = state.config.snapshot().map_err(|error| {
        ProxyError::config_snapshot(error.to_string())
            .with_request_metadata(body_read_request_metadata.clone())
    })?;
    if is_control_plane_models_request(request.method, request.uri) {
        return Ok(OpenAiBodyAdmission {
            config,
            body,
            in_flight_permit,
            admission_metadata: request.admission_metadata,
        });
    }

    if admission_permit_kind != AdmissionPermitKind::BodyRouting
        && !config.has_upstream_profile_generation_limits()
    {
        return Ok(OpenAiBodyAdmission {
            config,
            body,
            in_flight_permit,
            admission_metadata: request.admission_metadata,
        });
    }

    let model_id_for_admission = extract_model_id(&body);
    select_allowed_upstream_profile(&config, &state.listener, model_id_for_admission.as_deref())
        .map_err(|error| {
            ProxyError::listener_denied(error)
                .with_request_metadata(body_read_request_metadata.clone())
        })?;
    if admission_permit_kind == AdmissionPermitKind::Generation {
        return Ok(OpenAiBodyAdmission {
            config,
            body,
            in_flight_permit,
            admission_metadata: request.admission_metadata,
        });
    }
    let record_context = AdmissionRecordContext {
        store: state.store.clone(),
        request_id: request.request_id.clone(),
        started_at_unix_ms: request.started_at_unix_ms,
        request_metadata: body_read_request_metadata.clone(),
    };
    let admission = state
        .acquire_generation_permit_for_model(
            model_id_for_admission.as_deref(),
            in_flight_permit,
            record_context,
        )
        .await
        .map_err(|error| {
            let mut metadata = body_read_request_metadata;
            metadata.extend(error.request_metadata());
            ProxyError::admission(error).with_request_metadata(metadata)
        })?;
    let mut admission_metadata = request.admission_metadata;
    admission_metadata.extend(acquired_admission_metadata(admission.queue_wait));
    Ok(OpenAiBodyAdmission {
        config: admission.config,
        body,
        in_flight_permit: admission.permit,
        admission_metadata,
    })
}

struct PreparedOpenAiRequest {
    model_id: Option<String>,
    caller_profile: ProfileConfig,
    upstream_profile: UpstreamProfileConfig,
    route_reason: UpstreamRouteReason,
    upstream_url: Url,
    reqwest_method: reqwest::Method,
    shielded_chat_plan: ShieldedChatPlan,
}

fn prepare_openai_forward_request(
    state: &ProxyState,
    config: &AppConfig,
    method: &Method,
    uri: &Uri,
    body: &Bytes,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<PreparedOpenAiRequest, ProxyError> {
    let model_id = extract_model_id(body);
    let caller_profile = config
        .caller_profile_by_name(DEFAULT_PROFILE_NAME)
        .unwrap_or_else(|| config.default_caller_profile());
    let selected_profile =
        select_profile_for_request(config, &state.listener, method, uri, model_id.as_deref())?;
    let upstream_profile = selected_profile.profile;
    let route_reason = selected_profile.route_reason;
    add_upstream_profile_metadata(request_metadata, &upstream_profile, route_reason);
    let upstream_url = build_upstream_url(&upstream_profile.base_url, uri)?;
    let reqwest_method = upstream_method(method)?;
    let shielded_chat_plan =
        plan_shielded_chat(state, config, &upstream_profile.thinking, method, uri, body);
    add_shielded_request_metadata(
        request_metadata,
        shielded_chat_plan.intercepted,
        shielded_chat_plan.thinking_policy_applied,
        &shielded_chat_plan.liveness,
        &shielded_chat_plan.thinking_metadata,
    );
    request_metadata.extend(context_budget_preflight(
        method,
        uri,
        body,
        &shielded_chat_plan.upstream_body,
        &upstream_profile,
    )?);

    Ok(PreparedOpenAiRequest {
        model_id,
        caller_profile,
        upstream_profile,
        route_reason,
        upstream_url,
        reqwest_method,
        shielded_chat_plan,
    })
}

async fn apply_pre_request_guard(
    config: &AppConfig,
    request_id: &RequestId,
    prepared_request: &mut PreparedOpenAiRequest,
) -> Result<(), ProxyError> {
    if config.guard_workflows.pre_request.is_none() {
        return Ok(());
    }
    let Some(messages) =
        chat_messages_from_body(&prepared_request.shielded_chat_plan.downstream_body)
    else {
        return Ok(());
    };
    let outcome = run_pre_request_guard(
        config,
        request_id.as_str(),
        prepared_request.model_id.as_deref().unwrap_or_default(),
        messages,
        prepared_request.caller_profile.clone(),
    )
    .await;
    match outcome {
        GuardOutcome::Allow | GuardOutcome::Skipped => Ok(()),
        GuardOutcome::Block { reason } => Err(ProxyError::guard_blocked(reason)),
        GuardOutcome::Replace { messages } => {
            prepared_request.shielded_chat_plan.downstream_body = replace_chat_messages(
                &prepared_request.shielded_chat_plan.downstream_body,
                &messages,
            )?;
            prepared_request.shielded_chat_plan.upstream_body = replace_chat_messages(
                &prepared_request.shielded_chat_plan.upstream_body,
                &messages,
            )?;
            Ok(())
        }
    }
}

async fn run_pre_request_guard(
    config: &AppConfig,
    request_id: &str,
    model: &str,
    messages: Vec<serde_json::Value>,
    profile: ProfileConfig,
) -> GuardOutcome {
    let guard_config = config.guard_workflows.clone();
    let workflows = config.workflows.clone();
    let request_id = request_id.to_owned();
    let model = model.to_owned();
    let fail_closed_blocks = guard_config.fail_closed_blocks;
    tokio::task::spawn_blocking(move || {
        GuardExecutor::new(guard_config, workflows).pre_request_guard(
            &request_id,
            &model,
            &messages,
            &profile,
        )
    })
    .await
    .unwrap_or_else(|error| {
        if fail_closed_blocks {
            GuardOutcome::Block {
                reason: format!("guard worker failed: {error}"),
            }
        } else {
            GuardOutcome::Allow
        }
    })
}

async fn apply_post_response_guard(
    runtime: &ShieldedRetryRuntime,
    aggregated: &mut ShieldedAggregatedAttempt,
) {
    let Ok(config) = runtime.config.snapshot() else {
        return;
    };
    if config.guard_workflows.post_response.is_none() {
        return;
    }
    let Ok(response) = serde_json::from_slice::<serde_json::Value>(&aggregated.body) else {
        return;
    };
    let outcome = run_post_response_guard(
        &config,
        runtime.request_id.as_str(),
        runtime.model_id.as_deref().unwrap_or_default(),
        response.clone(),
        runtime.caller_profile.clone(),
    )
    .await;
    match outcome {
        GuardOutcome::Allow | GuardOutcome::Skipped => {}
        GuardOutcome::Block { reason } => {
            let body = safe_refusal_response_body(&response, &reason);
            replace_aggregated_response_body(aggregated, &body);
        }
        GuardOutcome::Replace { messages } => {
            if let Some(response) = messages.into_iter().next() {
                let body = Bytes::from(response.to_string());
                replace_aggregated_response_body(aggregated, &body);
            }
        }
    }
}

async fn run_post_response_guard(
    config: &AppConfig,
    request_id: &str,
    model: &str,
    response: serde_json::Value,
    profile: ProfileConfig,
) -> GuardOutcome {
    let guard_config = config.guard_workflows.clone();
    let workflows = config.workflows.clone();
    let request_id = request_id.to_owned();
    let model = model.to_owned();
    let fail_closed_blocks = guard_config.fail_closed_blocks;
    tokio::task::spawn_blocking(move || {
        GuardExecutor::new(guard_config, workflows).post_response_guard(
            &request_id,
            &model,
            &response,
            &profile,
        )
    })
    .await
    .unwrap_or_else(|error| {
        if fail_closed_blocks {
            GuardOutcome::Block {
                reason: format!("guard worker failed: {error}"),
            }
        } else {
            GuardOutcome::Allow
        }
    })
}

fn chat_messages_from_body(body: &Bytes) -> Option<Vec<serde_json::Value>> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("messages")?
        .as_array()
        .cloned()
}

fn replace_chat_messages(
    body: &Bytes,
    messages: &[serde_json::Value],
) -> Result<Bytes, ProxyError> {
    let mut value = serde_json::from_slice::<serde_json::Value>(body).map_err(|error| {
        ProxyError::guard_blocked(format!("guard could not parse request JSON: {error}"))
    })?;
    let Some(object) = value.as_object_mut() else {
        return Err(ProxyError::guard_blocked(String::from(
            "guard could not rewrite a non-object request body",
        )));
    };
    object.insert(
        String::from("messages"),
        serde_json::Value::Array(messages.to_vec()),
    );
    Ok(Bytes::from(value.to_string()))
}

fn replace_aggregated_response_body(aggregated: &mut ShieldedAggregatedAttempt, body: &Bytes) {
    aggregated.body = body.clone();
    aggregated.sse_body = openai_data_sse_body(body);
    aggregated.response_metadata.insert(
        String::from("guard_post_response_replaced"),
        String::from("true"),
    );
}

fn safe_refusal_response_body(original: &serde_json::Value, reason: &str) -> Bytes {
    let refusal = "I can't help with that request.";
    let mut response = original.clone();
    if let Some(message) = response
        .pointer_mut("/choices/0/message")
        .and_then(serde_json::Value::as_object_mut)
    {
        message.insert(String::from("content"), json!(refusal));
        message.insert(String::from("refusal"), json!(reason));
        message.remove("tool_calls");
        if let Some(choice) = response
            .pointer_mut("/choices/0")
            .and_then(serde_json::Value::as_object_mut)
        {
            choice.insert(String::from("finish_reason"), json!("content_filter"));
        }
        return Bytes::from(response.to_string());
    }
    Bytes::from(
        json!({
            "id": "chatcmpl-guard-refusal",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": refusal,
                    "refusal": reason
                },
                "finish_reason": "content_filter"
            }]
        })
        .to_string(),
    )
}

fn openai_data_sse_body(body: &Bytes) -> Bytes {
    let mut frame = BytesMut::with_capacity(body.len().saturating_add(22));
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(body);
    frame.extend_from_slice(b"\n\ndata: [DONE]\n\n");
    frame.freeze()
}

fn select_profile_for_request(
    config: &AppConfig,
    listener: &ListenerConfig,
    method: &Method,
    uri: &Uri,
    model: Option<&str>,
) -> Result<SelectedUpstreamProfile, ProxyError> {
    if is_control_plane_models_request(method, uri)
        && let Some(allowed_upstreams) = listener.allowed_upstreams.as_ref()
        && let Some(profile_name) = allowed_upstreams.first()
        && let Some(profile) = config.upstream_profile_by_name(profile_name)
    {
        return Ok(SelectedUpstreamProfile {
            profile,
            route_reason: UpstreamRouteReason::MatchedModel,
        });
    }
    select_allowed_upstream_profile(config, listener, model).map_err(ProxyError::listener_denied)
}

fn select_allowed_upstream_profile(
    config: &AppConfig,
    listener: &ListenerConfig,
    model: Option<&str>,
) -> Result<SelectedUpstreamProfile, ListenerUpstreamDenied> {
    let selected = config.select_upstream_profile(model);
    if listener.allows_upstream(&selected.profile.name) {
        return Ok(selected);
    }
    Err(ListenerUpstreamDenied {
        listener_name: listener.name.clone(),
        listener_port: listener.port,
        upstream_profile: selected.profile.name,
        model_id: denied_model_id_summary(model),
    })
}

fn denied_model_id_summary(model: Option<&str>) -> String {
    let Some(model) = model else {
        return String::from("[none]");
    };
    if model.len() <= MAX_DENIED_MODEL_ID_BYTES && !looks_sensitive_text(model) {
        return model.to_owned();
    }
    format!(
        "[redacted model: bytes={}, hash={}]",
        model.len(),
        stable_text_hash(model)
    )
}

fn stable_text_hash(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("siphash64:{:016x}", hasher.finish())
}

fn looks_sensitive_text(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("bearer ")
        || normalized.contains("api_key")
        || normalized.contains("api-key")
        || normalized.contains("x-api-key")
        || normalized.contains("authorization")
        || normalized.contains("sk-")
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
    upstream_timeout: Duration,
    upstream_profile: UpstreamProfileConfig,
    route_reason: UpstreamRouteReason,
    liveness: ShieldedLivenessSelection,
    thinking_policy_applied: bool,
    thinking_metadata: BTreeMap<String, String>,
    request_id: &'request RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    request_metadata: BTreeMap<String, String>,
    in_flight_permit: InFlightPermit,
}

async fn forward_generic_openai_request(
    context: GenericForwardContext<'_>,
) -> Result<Response<Body>, ProxyError> {
    if is_control_plane_models_request(&context.method, &context.uri)
        && let Some(groups) =
            listener_models_upstream_groups(context.config, &context.state.listener)
        && groups.len() > 1
    {
        return forward_merged_models_response(context, groups).await;
    }

    let attempt_id = AttemptId::for_request(context.request_id, 1);
    let attempt_started_at_unix_ms = unix_time_millis();
    let mut attempt_request_metadata =
        attempt_request_metadata(&context.method, &context.uri, &context.downstream_headers);
    add_listener_metadata(&mut attempt_request_metadata, &context.state.listener);
    add_upstream_profile_metadata(
        &mut attempt_request_metadata,
        &context.upstream_profile,
        context.route_reason,
    );
    add_shielded_request_metadata(
        &mut attempt_request_metadata,
        false,
        context.thinking_policy_applied,
        &context.liveness,
        &context.thinking_metadata,
    );
    let upstream_response = send_first_upstream_attempt(UpstreamAttemptContext {
        client: &context.state.client,
        method: context.reqwest_method,
        upstream_url: context.upstream_url,
        downstream_headers: &context.downstream_headers,
        upstream_body: context.upstream_body,
        upstream_timeout: context.upstream_timeout,
        attempt_id: attempt_id.clone(),
        attempt_number: 1,
        request_id: context.request_id,
        attempt_started_at_unix_ms,
        request_metadata: &context.request_metadata,
        attempt_request_metadata: &attempt_request_metadata,
    })
    .await?;
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let response_parts = ForwardedResponseParts {
        config: context.state.config.clone(),
        store: context.state.store.clone(),
        evidence_store: context.state.evidence_store.clone(),
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
            listener: &context.state.listener,
            metadata_config: &context.upstream_profile.metadata,
        },
        response_parts,
        upstream_response,
        context.in_flight_permit,
    )
    .await
}

#[derive(Clone)]
struct ModelsUpstreamGroup {
    profile: UpstreamProfileConfig,
    base_url: String,
    request_timeout_ms: u64,
    metadata: MetadataConfig,
}

struct CompletedModelsFetch {
    body: Bytes,
    attempt_record: AttemptRecord,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    upstream_mode: UpstreamMode,
}

async fn forward_merged_models_response(
    context: GenericForwardContext<'_>,
    groups: Vec<ModelsUpstreamGroup>,
) -> Result<Response<Body>, ProxyError> {
    let mut response_status = None;
    let mut response_headers = None;
    let mut response_mode = None;
    let mut filtered_bodies = Vec::with_capacity(groups.len());
    let mut attempt_records = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let attempt_number = u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1);
        let fetch = match fetch_models_upstream_group(
            &context,
            group,
            attempt_number,
            u32::try_from(groups.len()).unwrap_or(u32::MAX),
        )
        .await
        {
            Ok(fetch) => fetch,
            Err(error) => return Err(error.with_completed_attempt_records(attempt_records)),
        };
        if response_status.is_none() {
            response_status = Some(fetch.upstream_status);
            response_headers = Some(fetch.upstream_headers.clone());
            response_mode = Some(fetch.upstream_mode);
        }
        filtered_bodies.push(fetch.body);
        attempt_records.push(fetch.attempt_record);
    }

    let metadata_config = groups
        .first()
        .map_or(&context.upstream_profile.metadata, |group| &group.metadata);
    let merged_body = model_metadata::merge_models_bodies(filtered_bodies);
    let (upstream_status, upstream_headers, upstream_mode) = if merged_body.has_valid_model_list {
        (
            reqwest::StatusCode::OK,
            models_success_response_headers(),
            UpstreamMode::NotApplicable,
        )
    } else {
        (
            response_status.unwrap_or(reqwest::StatusCode::OK),
            response_headers.unwrap_or_default(),
            response_mode.unwrap_or(UpstreamMode::NotApplicable),
        )
    };
    let body =
        model_metadata::enrich_models_body(context.config, metadata_config, merged_body.body);
    let response_parts = ForwardedResponseParts {
        config: context.state.config.clone(),
        store: context.state.store.clone(),
        evidence_store: context.state.evidence_store.clone(),
        request_id: context.request_id.clone(),
        started_at_unix_ms: context.started_at_unix_ms,
        attempt_id: AttemptId::for_request(context.request_id, 1),
        attempt_started_at_unix_ms: context.started_at_unix_ms,
        upstream_mode,
        model_id: context.model_id,
        input_fingerprint: context.liveness.input_fingerprint.clone(),
        upstream_status,
        upstream_headers: upstream_headers.clone(),
        request_metadata: context.request_metadata,
        attempt_request_metadata: BTreeMap::new(),
    };
    let mut observer = response_parts.into_observer();
    observer.completed_attempt_records = attempt_records;
    observer.final_attempt = None;
    let response_body = ObservedBufferedBody::new(body, observer, context.in_flight_permit);
    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    ))
}

fn models_success_response_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers
}

async fn fetch_models_upstream_group(
    context: &GenericForwardContext<'_>,
    group: &ModelsUpstreamGroup,
    attempt_number: u32,
    attempt_max_attempts: u32,
) -> Result<CompletedModelsFetch, ProxyError> {
    let attempt_id = AttemptId::for_request(context.request_id, attempt_number);
    let attempt_started_at_unix_ms = unix_time_millis();
    let mut attempt_request_metadata =
        attempt_request_metadata(&context.method, &context.uri, &context.downstream_headers);
    attempt_request_metadata.insert(String::from("attempt_number"), attempt_number.to_string());
    add_listener_metadata(&mut attempt_request_metadata, &context.state.listener);
    add_upstream_profile_metadata(
        &mut attempt_request_metadata,
        &group.profile,
        UpstreamRouteReason::MatchedModel,
    );
    add_shielded_request_metadata(
        &mut attempt_request_metadata,
        false,
        context.thinking_policy_applied,
        &context.liveness,
        &context.thinking_metadata,
    );

    let upstream_url = build_upstream_url(&group.base_url, &context.uri)?;
    let upstream_response = send_first_upstream_attempt(UpstreamAttemptContext {
        client: &context.state.client,
        method: context.reqwest_method.clone(),
        upstream_url,
        downstream_headers: &context.downstream_headers,
        upstream_body: context.upstream_body.clone(),
        upstream_timeout: Duration::from_millis(group.request_timeout_ms),
        attempt_id: attempt_id.clone(),
        attempt_number,
        request_id: context.request_id,
        attempt_started_at_unix_ms,
        request_metadata: &context.request_metadata,
        attempt_request_metadata: &attempt_request_metadata,
    })
    .await?;
    let upstream_mode = upstream_mode_from_headers(upstream_response.headers());
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let body = match read_upstream_body_bytes(upstream_response.bytes_stream()).await {
        Ok(body) => body,
        Err(error) => {
            return Err(upstream_body_error_with_observability(
                error,
                context.request_metadata.clone(),
                attempt_request_metadata.clone(),
                attempt_id,
                attempt_number,
                context.request_id.clone(),
                attempt_started_at_unix_ms,
            ));
        }
    };
    let body_len = u64::try_from(body.len()).unwrap_or(u64::MAX);
    let body = filter_models_body_for_listener(context.config, &context.state.listener, body);
    let attempt_record = final_attempt_record(
        FinalAttemptContext {
            attempt_id,
            attempt_number,
            attempt_max_attempts,
            started_at_unix_ms: attempt_started_at_unix_ms,
            upstream_mode,
            upstream_status,
            upstream_headers: upstream_headers.clone(),
            request_metadata: attempt_request_metadata,
            extra_response_metadata: BTreeMap::new(),
            raw_payloads: RawPayloads::default(),
        },
        context.request_id,
        unix_time_millis(),
        body_len,
        &BodyCompletion::Succeeded,
    );

    Ok(CompletedModelsFetch {
        body,
        attempt_record,
        upstream_status,
        upstream_headers,
        upstream_mode,
    })
}

fn upstream_body_error_with_observability(
    error: ProxyError,
    request_metadata: BTreeMap<String, String>,
    attempt_request_metadata: BTreeMap<String, String>,
    attempt_id: AttemptId,
    attempt_number: u32,
    request_id: RequestId,
    attempt_started_at_unix_ms: u64,
) -> ProxyError {
    let finished_at_unix_ms = unix_time_millis();
    let error_reason = error.to_string();
    let attempt_record = failed_attempt_record(FailedAttemptRecordInput {
        attempt_id,
        attempt_number,
        request_id,
        started_at_unix_ms: attempt_started_at_unix_ms,
        finished_at_unix_ms,
        error_type: error.error_type(),
        error_reason: &error_reason,
        request_metadata: attempt_request_metadata,
        extra_response_metadata: BTreeMap::new(),
    });
    error.with_observability(request_metadata, attempt_record)
}

struct UpstreamAttemptContext<'request> {
    client: &'request Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: &'request HeaderMap,
    upstream_body: Bytes,
    upstream_timeout: Duration,
    attempt_id: AttemptId,
    attempt_number: u32,
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
        context.upstream_timeout,
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_reason = error.to_string();
            let attempt_record = failed_attempt_record(FailedAttemptRecordInput {
                attempt_id: context.attempt_id,
                attempt_number: context.attempt_number,
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
    listener: &'request ListenerConfig,
    metadata_config: &'request MetadataConfig,
}

struct ShieldedChatPlan {
    downstream_body: Bytes,
    upstream_body: Bytes,
    intercepted: bool,
    kind: ShieldedChatKind,
    thinking_policy_applied: bool,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    loop_context: shielded_chat::LoopInspectionContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShieldedChatKind {
    NonStream,
    Stream,
    Generic,
}

fn plan_shielded_chat(
    state: &ProxyState,
    config: &AppConfig,
    thinking: &ThinkingConfig,
    method: &Method,
    uri: &Uri,
    body: &Bytes,
) -> ShieldedChatPlan {
    let retry_initial_thinking = config
        .retry
        .ladder
        .first()
        .map_or(thinking, |entry| &entry.thinking);
    let (request, intercepted, kind) = if should_intercept_non_stream_chat(method, uri, config) {
        if let Some(non_stream_request) =
            shielded_chat::prepare_non_stream_request(body, retry_initial_thinking)
        {
            (Some(non_stream_request), true, ShieldedChatKind::NonStream)
        } else if let Some(stream_request) = shielded_chat::prepare_stream_request(
            body,
            if config.retry.shielded_streaming_enabled {
                retry_initial_thinking
            } else {
                thinking
            },
        ) {
            (
                Some(stream_request),
                config.retry.shielded_streaming_enabled,
                if config.retry.shielded_streaming_enabled {
                    ShieldedChatKind::Stream
                } else {
                    ShieldedChatKind::Generic
                },
            )
        } else {
            (None, false, ShieldedChatKind::Generic)
        }
    } else {
        (None, false, ShieldedChatKind::Generic)
    };
    let upstream_body = request.as_ref().map_or_else(
        || body.clone(),
        shielded_chat::PreparedChatRequest::upstream_body,
    );
    let thinking_metadata = request
        .as_ref()
        .map_or_else(BTreeMap::new, |request| request.thinking_metadata().clone());
    let thinking_policy_applied = request.is_some();
    let liveness = select_shielded_liveness(state, config, body, kind, unix_time_millis());
    let loop_context = if intercepted {
        shielded_chat::LoopInspectionContext::from_request_body(&config.loop_guard, body)
    } else {
        shielded_chat::LoopInspectionContext::empty(&config.loop_guard)
    };

    ShieldedChatPlan {
        downstream_body: body.clone(),
        upstream_body,
        intercepted,
        kind,
        thinking_policy_applied,
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShieldedRetryPolicy {
    enabled: bool,
    max_attempts: u32,
    anti_loop_hint_enabled: bool,
    shielded_streaming_enabled: bool,
    downstream_drop_policy: DownstreamDropPolicy,
    ladder: Vec<RetryLadderConfig>,
}

impl ShieldedRetryPolicy {
    fn from_config(config: &RetryConfig) -> Self {
        let max_attempts = if config.enabled {
            if config.ladder.is_empty() {
                config.max_attempts
            } else {
                config
                    .max_attempts
                    .min(u32::try_from(config.ladder.len()).unwrap_or(u32::MAX))
            }
        } else {
            1
        };
        Self {
            enabled: config.enabled,
            max_attempts,
            anti_loop_hint_enabled: config.anti_loop_hint_enabled,
            shielded_streaming_enabled: config.shielded_streaming_enabled,
            downstream_drop_policy: config.downstream_drop_policy,
            ladder: config.ladder.clone(),
        }
    }

    fn allows_retry_after(&self, attempt_number: u32) -> bool {
        self.enabled && attempt_number < self.max_attempts
    }

    fn attempt_plan(
        &self,
        attempt_number: u32,
        fallback_thinking: &ThinkingConfig,
    ) -> ShieldedAttemptPlan {
        let index = attempt_number.saturating_sub(1);
        self.ladder
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .map_or_else(
                || ShieldedAttemptPlan {
                    name: format!("attempt-{attempt_number}"),
                    thinking: fallback_thinking.clone(),
                    anti_loop_hint: None,
                },
                |entry| ShieldedAttemptPlan {
                    name: entry.name.clone(),
                    thinking: entry.thinking.clone(),
                    anti_loop_hint: entry.anti_loop_hint.clone(),
                },
            )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShieldedAttemptPlan {
    name: String,
    thinking: ThinkingConfig,
    anti_loop_hint: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UpstreamStallPolicy {
    enabled: bool,
    idle_timeout: Duration,
    recovery_command: Vec<String>,
    recovery_timeout: Duration,
    recovery_cooldown: Duration,
    recovery_budget_window: Duration,
    recovery_max_per_window: u32,
}

impl UpstreamStallPolicy {
    fn from_config(config: &UpstreamStallConfig) -> Self {
        Self {
            enabled: config.enabled,
            idle_timeout: Duration::from_millis(config.idle_timeout_ms),
            recovery_command: config.recovery_command.clone(),
            recovery_timeout: Duration::from_millis(config.recovery_timeout_ms),
            recovery_cooldown: Duration::from_millis(config.recovery_cooldown_ms),
            recovery_budget_window: Duration::from_millis(config.recovery_budget_window_ms),
            recovery_max_per_window: config.recovery_max_per_window,
        }
    }

    const fn idle_timeout(&self) -> Option<Duration> {
        if self.enabled {
            Some(self.idle_timeout)
        } else {
            None
        }
    }
}

#[derive(Debug, Default)]
struct UpstreamStallRecoveryCoordinator {
    state: AsyncMutex<UpstreamStallRecoveryState>,
    notify: Notify,
}

#[derive(Debug, Default)]
struct UpstreamStallRecoveryState {
    running: bool,
    last_finished: Option<Instant>,
    window_started: Option<Instant>,
    runs_in_window: u32,
    last_result: Option<BTreeMap<String, String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShieldedRetryCause {
    LoopDetected,
    TransientUpstreamStatus,
    TransientTransport,
    TransientStream,
    UpstreamStall,
}

impl ShieldedRetryCause {
    const fn retry_reason(self) -> &'static str {
        match self {
            Self::LoopDetected => "loop_detected",
            Self::TransientUpstreamStatus => "transient_upstream_status",
            Self::TransientTransport => "transient_upstream_transport",
            Self::TransientStream => "transient_upstream_stream_failure",
            Self::UpstreamStall => "upstream_stall",
        }
    }

    const fn next_attempt_reason(self) -> &'static str {
        match self {
            Self::LoopDetected => "previous_loop_detected",
            Self::TransientUpstreamStatus => "previous_transient_upstream_status",
            Self::TransientTransport => "previous_transient_upstream_transport",
            Self::TransientStream => "previous_transient_upstream_stream_failure",
            Self::UpstreamStall => "previous_upstream_stall",
        }
    }
}

async fn forward_upstream_response(
    dispatch: ResponseDispatch<'_>,
    response_parts: ForwardedResponseParts,
    upstream_response: reqwest::Response,
    in_flight_permit: InFlightPermit,
) -> Result<Response<Body>, ProxyError> {
    let upstream_status = response_parts.upstream_status;
    let upstream_headers = response_parts.upstream_headers.clone();
    if should_buffer_models_response(
        dispatch.method,
        dispatch.uri,
        dispatch.metadata_config,
        dispatch.listener,
    ) {
        return forward_buffered_models_response(
            response_parts,
            upstream_response,
            in_flight_permit,
            dispatch.config,
            dispatch.listener,
            dispatch.metadata_config,
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

async fn read_body_bytes(body: Body, max_request_body_bytes: usize) -> Result<Bytes, ProxyError> {
    to_bytes(body, max_request_body_bytes)
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

fn should_enrich_models_response(method: &Method, uri: &Uri, metadata: &MetadataConfig) -> bool {
    method == Method::GET
        && uri.path() == "/v1/models"
        && metadata.discovery_enabled
        && metadata.enrich_responses
}

fn should_buffer_models_response(
    method: &Method,
    uri: &Uri,
    metadata: &MetadataConfig,
    listener: &ListenerConfig,
) -> bool {
    method == Method::GET
        && uri.path() == "/v1/models"
        && (listener.allowed_upstreams.is_some()
            || should_enrich_models_response(method, uri, metadata))
}

fn should_intercept_non_stream_chat(method: &Method, uri: &Uri, config: &AppConfig) -> bool {
    method == Method::POST && uri.path() == "/v1/chat/completions" && config.shielding.enabled
}

fn context_budget_preflight(
    method: &Method,
    uri: &Uri,
    original_body: &Bytes,
    upstream_body: &Bytes,
    profile: &UpstreamProfileConfig,
) -> Result<BTreeMap<String, String>, ProxyError> {
    let Some(param) = context_budget_param(method, uri) else {
        return Ok(BTreeMap::from([(
            String::from("context_budget_preflight"),
            String::from("not_applicable"),
        )]));
    };
    let Some(context_window) = profile.metadata.context_window_override().map(u64::from) else {
        return Ok(BTreeMap::from([(
            String::from("context_budget_preflight"),
            String::from("skipped_no_context_window"),
        )]));
    };

    let original_json = serde_json::from_slice::<serde_json::Value>(original_body).ok();
    let upstream_json = serde_json::from_slice::<serde_json::Value>(upstream_body).ok();
    let input_tokens = original_json.as_ref().map_or_else(
        || estimate_text_tokens(original_body),
        |value| estimate_request_input_tokens(param, value),
    );
    let reserved_output_tokens = upstream_json
        .as_ref()
        .map_or(0, estimate_reserved_output_tokens);
    let safety_margin = u64::from(profile.metadata.input_token_safety_margin);
    let total_tokens = input_tokens
        .saturating_add(reserved_output_tokens)
        .saturating_add(safety_margin);
    let estimate = ContextBudgetEstimate {
        param,
        context_window,
        input_tokens,
        reserved_output_tokens,
        safety_margin,
        total_tokens,
    };
    if estimate.exceeds_window() {
        return Err(ProxyError::context_budget_exceeded(estimate));
    }
    Ok(estimate.metadata("allowed"))
}

fn context_budget_param(method: &Method, uri: &Uri) -> Option<&'static str> {
    if method != Method::POST {
        return None;
    }
    match uri.path() {
        "/v1/chat/completions" => Some("messages"),
        "/v1/completions" => Some("prompt"),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
struct ContextBudgetEstimate {
    param: &'static str,
    context_window: u64,
    input_tokens: u64,
    reserved_output_tokens: u64,
    safety_margin: u64,
    total_tokens: u64,
}

impl ContextBudgetEstimate {
    const fn exceeds_window(self) -> bool {
        self.total_tokens > self.context_window
    }

    fn metadata(self, status: &str) -> BTreeMap<String, String> {
        BTreeMap::from([
            (String::from("context_budget_preflight"), status.to_owned()),
            (String::from("context_budget_param"), self.param.to_owned()),
            (
                String::from("context_budget_window_tokens"),
                self.context_window.to_string(),
            ),
            (
                String::from("context_budget_input_estimate_tokens"),
                self.input_tokens.to_string(),
            ),
            (
                String::from("context_budget_reserved_output_tokens"),
                self.reserved_output_tokens.to_string(),
            ),
            (
                String::from("context_budget_safety_margin_tokens"),
                self.safety_margin.to_string(),
            ),
            (
                String::from("context_budget_total_estimate_tokens"),
                self.total_tokens.to_string(),
            ),
        ])
    }

    fn message(self) -> String {
        format!(
            "Input plus reserved output exceeds upstream context window; lower the caller auto-compaction threshold, input tokens, or requested max_tokens. estimated_total_tokens={} context_window_tokens={} input_tokens={} reserved_output_tokens={} safety_margin_tokens={}",
            self.total_tokens,
            self.context_window,
            self.input_tokens,
            self.reserved_output_tokens,
            self.safety_margin
        )
    }
}

fn estimate_request_input_tokens(param: &str, value: &serde_json::Value) -> u64 {
    match param {
        "messages" => estimate_chat_request_input_tokens(value),
        "prompt" => value.get("prompt").map_or(0, estimate_json_input_tokens),
        _ => 0,
    }
}

fn estimate_chat_request_input_tokens(value: &serde_json::Value) -> u64 {
    let message_tokens = value
        .get("messages")
        .map_or(0, estimate_chat_messages_tokens);
    let tool_definition_tokens = estimate_json_fields_tokens(value, &["tools", "functions"]);
    message_tokens.saturating_add(tool_definition_tokens)
}

fn estimate_chat_messages_tokens(value: &serde_json::Value) -> u64 {
    let serde_json::Value::Array(messages) = value else {
        return estimate_json_input_tokens(value);
    };
    messages.iter().map(estimate_chat_message_tokens).sum()
}

fn estimate_chat_message_tokens(value: &serde_json::Value) -> u64 {
    let serde_json::Value::Object(message) = value else {
        return estimate_json_input_tokens(value);
    };
    estimate_object_fields_tokens(
        message,
        &[
            "role",
            "name",
            "content",
            "tool_call_id",
            "tool_calls",
            "function_call",
        ],
    )
}

fn estimate_json_input_tokens(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::String(text) => estimate_text_tokens(text.as_bytes()),
        serde_json::Value::Array(values) => values.iter().map(estimate_json_input_tokens).sum(),
        serde_json::Value::Object(object) => object.values().map(estimate_json_input_tokens).sum(),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => 0,
    }
}

fn estimate_text_tokens(bytes: &[u8]) -> u64 {
    let text = String::from_utf8_lossy(bytes);
    let bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let byte_estimate = bytes.saturating_add(3) / 4;
    let word_count = u64::try_from(text.split_whitespace().count()).unwrap_or(u64::MAX);
    byte_estimate.max(word_count)
}

fn estimate_json_fields_tokens(value: &serde_json::Value, fields: &[&str]) -> u64 {
    let serde_json::Value::Object(object) = value else {
        return 0;
    };
    estimate_object_fields_tokens(object, fields)
}

fn estimate_object_fields_tokens(
    object: &serde_json::Map<String, serde_json::Value>,
    fields: &[&str],
) -> u64 {
    fields
        .iter()
        .filter_map(|field| object.get(*field))
        .map(estimate_json_input_tokens)
        .fold(0_u64, u64::saturating_add)
}

fn estimate_reserved_output_tokens(value: &serde_json::Value) -> u64 {
    let output_cap = ["max_tokens", "max_completion_tokens", "max_output_tokens"]
        .iter()
        .filter_map(|field| value.get(*field).and_then(serde_json::Value::as_u64))
        .max()
        .unwrap_or(0);
    output_cap.max(estimate_thinking_budget_tokens(value))
}

fn estimate_thinking_budget_tokens(value: &serde_json::Value) -> u64 {
    [
        &["thinking", "budget_tokens"][..],
        &["thinking_token_budget"][..],
        &["thinking_budget"][..],
        &["chat_template_kwargs", "thinking_budget"][..],
        &["extra_body", "thinking_budget"][..],
        &["extra_body", "thinking_token_budget"][..],
        &["extra_body", "thinking", "budget_tokens"][..],
        &["extra_body", "chat_template_kwargs", "thinking_budget"][..],
    ]
    .iter()
    .filter_map(|path| numeric_json_path(value, path))
    .max()
    .unwrap_or(0)
}

fn numeric_json_path(value: &serde_json::Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
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
    timeout: Duration,
) -> Result<reqwest::Response, ProxyError> {
    let headers = forwarded_request_headers(downstream_headers);
    client
        .request(method, upstream_url)
        .headers(headers)
        .body(body)
        .timeout(timeout)
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
    config: ConfigHandle,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
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
            config: self.config,
            downstream_mode,
            store: self.store,
            evidence_store: self.evidence_store,
            shadow_evidence: ShadowEvidenceState::default(),
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
            attempt_number: 1,
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

async fn forward_buffered_models_response(
    response_parts: ForwardedResponseParts,
    upstream_response: reqwest::Response,
    in_flight_permit: InFlightPermit,
    config: &AppConfig,
    listener: &ListenerConfig,
    metadata_config: &MetadataConfig,
) -> Result<Response<Body>, ProxyError> {
    let upstream_status = response_parts.upstream_status;
    let upstream_headers = response_parts.upstream_headers.clone();
    let body = match read_upstream_body_bytes(upstream_response.bytes_stream()).await {
        Ok(body) => body,
        Err(error) => return Err(response_parts.into_body_read_error(error)),
    };
    let body = filter_models_body_for_listener(config, listener, body);
    let body = model_metadata::enrich_models_body(config, metadata_config, body);
    let observer = response_parts.into_observer();
    let response_body = ObservedBufferedBody::new(body, observer, in_flight_permit);

    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    ))
}

fn filter_models_body_for_listener(
    config: &AppConfig,
    listener: &ListenerConfig,
    body: Bytes,
) -> Bytes {
    if listener.allowed_upstreams.is_none() {
        return body;
    }
    model_metadata::filter_models_body_by_id(body, |model_id| {
        select_allowed_upstream_profile(config, listener, Some(model_id)).is_ok()
    })
}

fn listener_models_upstream_groups(
    config: &AppConfig,
    listener: &ListenerConfig,
) -> Option<Vec<ModelsUpstreamGroup>> {
    let upstream_profiles = listener_models_upstream_profiles(config, listener);
    let mut groups = Vec::<ModelsUpstreamGroup>::new();
    for profile in upstream_profiles {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.base_url == profile.base_url)
        {
            group.request_timeout_ms = group.request_timeout_ms.max(profile.request_timeout_ms);
            continue;
        }
        groups.push(ModelsUpstreamGroup {
            base_url: profile.base_url.clone(),
            request_timeout_ms: profile.request_timeout_ms,
            metadata: profile.metadata.clone(),
            profile,
        });
    }

    if groups.is_empty() {
        None
    } else {
        Some(groups)
    }
}

fn listener_models_upstream_profiles(
    config: &AppConfig,
    listener: &ListenerConfig,
) -> Vec<UpstreamProfileConfig> {
    if let Some(allowed_upstreams) = listener.allowed_upstreams.as_ref() {
        return allowed_upstreams
            .iter()
            .filter_map(|profile_name| config.upstream_profile_by_name(profile_name))
            .collect();
    }

    let mut profiles = Vec::with_capacity(config.upstream_profiles.len() + 1);
    profiles.push(config.default_upstream_profile());
    profiles.extend(config.upstream_profiles.iter().cloned());
    profiles
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
    downstream_body: Bytes,
    chat_kind: ShieldedChatKind,
    upstream_timeout: Duration,
    config: ConfigHandle,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    request_id: RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    request_metadata: BTreeMap<String, String>,
    listener: ListenerConfig,
    upstream_profile: UpstreamProfileConfig,
    caller_profile: ProfileConfig,
    route_reason: UpstreamRouteReason,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    loop_context: shielded_chat::LoopInspectionContext,
    retry_policy: ShieldedRetryPolicy,
    upstream_stall_policy: UpstreamStallPolicy,
    upstream_stall_recovery: Arc<UpstreamStallRecoveryCoordinator>,
    shadow_attempts: Arc<InFlightLimiter>,
    shadow_evidence: ShadowEvidenceState,
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
    raw_request_body: Option<String>,
    upstream_body: Bytes,
}

#[derive(Clone, Debug, Default)]
struct ShadowEvidenceState {
    inner: Arc<Mutex<ShadowEvidenceInner>>,
}

#[derive(Debug, Default)]
struct ShadowEvidenceInner {
    reserved_attempts: u32,
    records: Vec<EvidenceAttemptRecord>,
}

impl ShadowEvidenceState {
    fn try_reserve_attempt(&self, max_attempts: u32) -> Option<u32> {
        let mut inner = shadow_evidence_inner(&self.inner);
        if inner.reserved_attempts >= max_attempts {
            return None;
        }
        inner.reserved_attempts = inner.reserved_attempts.saturating_add(1);
        Some(inner.reserved_attempts)
    }

    fn push_record(&self, record: EvidenceAttemptRecord) {
        shadow_evidence_inner(&self.inner).records.push(record);
    }

    fn snapshot(&self) -> Vec<EvidenceAttemptRecord> {
        shadow_evidence_inner(&self.inner).records.clone()
    }
}

fn shadow_evidence_inner(
    current: &Mutex<ShadowEvidenceInner>,
) -> MutexGuard<'_, ShadowEvidenceInner> {
    match current.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct ShieldedStartedAttempt {
    info: ShieldedAttemptInfo,
    response: reqwest::Response,
}

struct ShieldedAcceptedOutcome {
    body: Bytes,
    sse_body: Bytes,
    response_metadata: BTreeMap<String, String>,
    prior_attempt_records: Vec<AttemptRecord>,
    final_attempt: FinalAttemptContext,
}

struct ShieldedDirectRelayOutcome {
    started: ShieldedStartedAttempt,
    prior_attempt_records: Vec<AttemptRecord>,
    response_metadata: BTreeMap<String, String>,
}

struct ShieldedAggregatedAttempt {
    body: Bytes,
    sse_body: Bytes,
    response_metadata: BTreeMap<String, String>,
    final_attempt: FinalAttemptContext,
}

struct ShieldedFailureOutcome {
    error_type: &'static str,
    error_message: String,
    response_metadata: BTreeMap<String, String>,
    attempt_records: Vec<AttemptRecord>,
    upstream_mode: UpstreamMode,
}

struct ShieldedTerminalForward {
    started: ShieldedStartedAttempt,
    prior_attempt_records: Vec<AttemptRecord>,
}

enum ShieldedRunOutcome {
    Accepted(ShieldedAcceptedOutcome),
    DirectRelay(ShieldedDirectRelayOutcome),
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
    raw_payloads: RawPayloads,
    upstream_body: Bytes,
}

async fn forward_shielded_chat_with_retries(
    runtime: ShieldedRetryRuntime,
    in_flight_permit: InFlightPermit,
) -> Result<Response<Body>, ProxyError> {
    if runtime.liveness.mode == ShieldedLivenessMode::Disabled
        && runtime.chat_kind != ShieldedChatKind::Stream
    {
        return Ok(immediate_shielded_retry_response(&runtime, in_flight_permit).await);
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
                    ShieldedRunOutcome::Accepted(outcome) => {
                        Ok(ShieldedAggregateOutcome::Accepted(outcome))
                    }
                    ShieldedRunOutcome::DirectRelay(outcome) => {
                        Ok(ShieldedAggregateOutcome::DirectRelay(outcome))
                    }
                    ShieldedRunOutcome::Failed(failure) => Err(failure),
                    ShieldedRunOutcome::TerminalForward(terminal) => Err(terminal_forward_failure(
                        terminal,
                        "non-retryable upstream response after shielded retry",
                    )),
                }
            });
            let aggregate = maybe_detach_shielded_aggregate(aggregate, &runtime.retry_policy);
            let response_body = ShieldedLivenessBody::new(
                aggregate,
                runtime.liveness.mode,
                if runtime.chat_kind == ShieldedChatKind::Stream {
                    ShieldedAcceptedResponseMode::OpenAiSse
                } else {
                    ShieldedAcceptedResponseMode::JsonCompletion
                },
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

async fn immediate_shielded_retry_response(
    runtime: &ShieldedRetryRuntime,
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
    match run_shielded_attempts(runtime.clone(), None, Vec::new(), true, None).await {
        ShieldedRunOutcome::Accepted(outcome) => {
            shielded_retry_success_response(runtime, outcome, in_flight_permit)
        }
        ShieldedRunOutcome::DirectRelay(outcome) => {
            shielded_retry_direct_relay_response(runtime, outcome, in_flight_permit)
        }
        ShieldedRunOutcome::Failed(failure) => {
            shielded_retry_error_response(runtime, failure, in_flight_permit)
        }
        ShieldedRunOutcome::TerminalForward(terminal) => {
            shielded_retry_terminal_forward_response(runtime, terminal, in_flight_permit)
        }
    }
}

fn maybe_detach_shielded_aggregate(
    aggregate: ShieldedAggregateFuture,
    policy: &ShieldedRetryPolicy,
) -> ShieldedAggregateFuture {
    if policy.downstream_drop_policy != DownstreamDropPolicy::Detach {
        return aggregate;
    }

    let (sender, receiver) =
        oneshot::channel::<Result<ShieldedAggregateOutcome, ShieldedFailureOutcome>>();
    tokio::spawn(async move {
        let _ = sender.send(aggregate.await);
    });
    Box::pin(async move {
        receiver.await.unwrap_or_else(|_closed| {
            Err(ShieldedFailureOutcome {
                error_type: "llm_guard_upstream_error",
                error_message: String::from("detached shielded upstream attempt result was lost"),
                response_metadata: BTreeMap::from([(
                    String::from("downstream_drop_policy"),
                    String::from("detach"),
                )]),
                attempt_records: Vec::new(),
                upstream_mode: UpstreamMode::NotApplicable,
            })
        })
    })
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
        &runtime.retry_policy,
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
        &runtime.retry_policy,
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
                    &runtime.retry_policy,
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
            attempt_records.push(attempt_failure_record(
                &failure,
                AttemptStatus::Failed,
                None,
                &runtime.retry_policy,
            ));
            return ShieldedAttemptStep::Failed(shielded_failure_outcome(
                failure,
                std::mem::take(attempt_records),
                &runtime.retry_policy,
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
            &runtime.retry_policy,
        ));
        return ShieldedAttemptStep::Failed(shielded_failure_outcome(
            failure,
            std::mem::take(attempt_records),
            &runtime.retry_policy,
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
        &runtime.retry_policy,
    ));
    ShieldedAttemptStep::Failed(shielded_failure_outcome(
        failure,
        std::mem::take(attempt_records),
        &runtime.retry_policy,
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
        runtime.upstream_stall_policy.idle_timeout(),
    )
    .await
    {
        Ok(aggregated) => Ok(ShieldedAggregatedAttempt {
            final_attempt: started.info.into_final_context(
                aggregated.response_metadata.clone(),
                aggregated.raw_payloads.clone(),
            ),
            body: aggregated.body,
            sse_body: aggregated.sse_body,
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

        if should_direct_relay_no_thinking_stream(&runtime, &started.info, retry_cause) {
            return ShieldedRunOutcome::DirectRelay(direct_relay_no_thinking_stream_outcome(
                started,
                &attempt_records,
            ));
        }

        match aggregate_shielded_attempt(&runtime, started).await {
            Ok(mut aggregated) => {
                apply_post_response_guard(&runtime, &mut aggregated).await;
                return shielded_accepted_outcome(aggregated, attempt_records);
            }
            Err(mut failure) => {
                let next_retry_cause = failure.retry_cause;
                let mut can_retry = should_retry_after_shielded_failure(&runtime, &failure);
                let recovery_gate = recovery_gate_for_retryable_upstream_stall(
                    &runtime,
                    can_retry,
                    next_retry_cause,
                )
                .await;
                can_retry = can_retry && recovery_gate.permits_retry;
                failure.response_metadata.extend(recovery_gate.metadata);
                let attempt_record = attempt_failure_record(
                    &failure,
                    retry_attempt_status(can_retry),
                    retry_cause_for_attempt_record(can_retry, next_retry_cause),
                    &runtime.retry_policy,
                );
                maybe_schedule_shadow_continuation(&runtime, &failure, &attempt_record);
                attempt_records.push(attempt_record);
                update_shielded_attempt_progress(attempt_progress.as_ref(), &attempt_records, None);
                if can_retry {
                    attempt_number = failure.attempt_number.saturating_add(1);
                    retry_cause = next_retry_cause;
                    continue;
                }
                return ShieldedRunOutcome::Failed(shielded_failure_outcome(
                    failure,
                    attempt_records,
                    &runtime.retry_policy,
                ));
            }
        }
    }
}

fn shielded_accepted_outcome(
    aggregated: ShieldedAggregatedAttempt,
    attempt_records: Vec<AttemptRecord>,
) -> ShieldedRunOutcome {
    ShieldedRunOutcome::Accepted(ShieldedAcceptedOutcome {
        body: aggregated.body,
        sse_body: aggregated.sse_body,
        response_metadata: aggregated.response_metadata,
        prior_attempt_records: attempt_records,
        final_attempt: aggregated.final_attempt,
    })
}

fn direct_relay_no_thinking_stream_outcome(
    started: ShieldedStartedAttempt,
    attempt_records: &[AttemptRecord],
) -> ShieldedDirectRelayOutcome {
    ShieldedDirectRelayOutcome {
        started,
        prior_attempt_records: attempt_records.to_vec(),
        response_metadata: no_thinking_direct_relay_metadata(),
    }
}

fn should_direct_relay_no_thinking_stream(
    runtime: &ShieldedRetryRuntime,
    info: &ShieldedAttemptInfo,
    retry_cause: Option<ShieldedRetryCause>,
) -> bool {
    runtime.chat_kind == ShieldedChatKind::Stream
        && info.attempt_number > 1
        && matches!(retry_cause, Some(ShieldedRetryCause::LoopDetected))
        && info
            .request_metadata
            .get("attempt_thinking_mode")
            .is_some_and(|mode| mode == ThinkingMode::ForceDisable.as_str())
}

fn no_thinking_direct_relay_metadata() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("shielded_direct_streaming_relay"),
            String::from("true"),
        ),
        (
            String::from("shielded_loop_inspection_skipped"),
            String::from("no_thinking_direct_streaming_relay"),
        ),
    ])
}

const fn retry_attempt_status(can_retry: bool) -> AttemptStatus {
    if can_retry {
        AttemptStatus::Retried
    } else {
        AttemptStatus::Failed
    }
}

const fn retry_cause_for_attempt_record(
    can_retry: bool,
    retry_cause: Option<ShieldedRetryCause>,
) -> Option<ShieldedRetryCause> {
    if can_retry { retry_cause } else { None }
}

fn should_retry_after_shielded_failure(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
) -> bool {
    failure.retry_cause.is_some()
        && runtime
            .retry_policy
            .allows_retry_after(failure.attempt_number)
}

struct UpstreamStallRecoveryGate {
    metadata: BTreeMap<String, String>,
    permits_retry: bool,
}

async fn recovery_gate_for_retryable_upstream_stall(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
    retry_cause: Option<ShieldedRetryCause>,
) -> UpstreamStallRecoveryGate {
    if !can_retry || !matches!(retry_cause, Some(ShieldedRetryCause::UpstreamStall)) {
        return UpstreamStallRecoveryGate {
            metadata: BTreeMap::new(),
            permits_retry: true,
        };
    }
    let mut metadata = run_upstream_stall_recovery(
        &runtime.upstream_stall_policy,
        &runtime.upstream_stall_recovery,
    )
    .await;
    let permits_retry = upstream_stall_recovery_permits_retry(&metadata);
    metadata.insert(
        String::from("upstream_stall_recovery_permits_retry"),
        permits_retry.to_string(),
    );
    UpstreamStallRecoveryGate {
        metadata,
        permits_retry,
    }
}

fn upstream_stall_recovery_permits_retry(metadata: &BTreeMap<String, String>) -> bool {
    match metadata
        .get("upstream_stall_recovery_status")
        .map(String::as_str)
    {
        Some("skipped_no_command" | "succeeded") => true,
        Some("joined_inflight") => metadata
            .get("upstream_stall_recovery_joined_status")
            .is_some_and(|status| status == "succeeded"),
        _ => false,
    }
}

async fn run_upstream_stall_recovery(
    policy: &UpstreamStallPolicy,
    coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
) -> BTreeMap<String, String> {
    let mut metadata = upstream_stall_recovery_metadata(!policy.recovery_command.is_empty());
    if policy.recovery_command.is_empty() {
        metadata.insert(
            String::from("upstream_stall_recovery_status"),
            String::from("skipped_no_command"),
        );
        return metadata;
    }

    let mut state = coordinator.state.lock().await;
    if state.running {
        drop(state);
        return wait_for_upstream_stall_recovery_result(policy, coordinator, true).await;
    }

    let now = Instant::now();
    if let Some(last_finished) = state.last_finished {
        let elapsed = now.saturating_duration_since(last_finished);
        if elapsed < policy.recovery_cooldown {
            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                String::from("skipped_cooldown"),
            );
            metadata.insert(
                String::from("upstream_stall_recovery_cooldown_remaining_ms"),
                policy
                    .recovery_cooldown
                    .saturating_sub(elapsed)
                    .as_millis()
                    .to_string(),
            );
            return metadata;
        }
    }

    let window_started = state.window_started.unwrap_or(now);
    if now.saturating_duration_since(window_started) >= policy.recovery_budget_window {
        state.window_started = Some(now);
        state.runs_in_window = 0;
    } else if state.runs_in_window >= policy.recovery_max_per_window {
        metadata.insert(
            String::from("upstream_stall_recovery_status"),
            String::from("skipped_budget_exhausted"),
        );
        metadata.insert(
            String::from("upstream_stall_recovery_budget_runs"),
            state.runs_in_window.to_string(),
        );
        metadata.insert(
            String::from("upstream_stall_recovery_budget_max_per_window"),
            policy.recovery_max_per_window.to_string(),
        );
        return metadata;
    } else if state.window_started.is_none() {
        state.window_started = Some(now);
    }

    state.running = true;
    state.runs_in_window = state.runs_in_window.saturating_add(1);
    drop(state);

    let task_policy = policy.clone();
    let task_coordinator = Arc::clone(coordinator);
    tokio::spawn(async move {
        let mut metadata = upstream_stall_recovery_metadata(true);
        metadata.extend(run_upstream_stall_recovery_command(&task_policy).await);
        finish_upstream_stall_recovery(&task_coordinator, metadata).await;
    });

    wait_for_upstream_stall_recovery_result(policy, coordinator, false).await
}

async fn wait_for_upstream_stall_recovery_result(
    policy: &UpstreamStallPolicy,
    coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
    joined_inflight: bool,
) -> BTreeMap<String, String> {
    let mut metadata = upstream_stall_recovery_metadata(true);
    let deadline = Instant::now() + recovery_join_timeout(policy);
    loop {
        let notified = coordinator.notify.notified();
        tokio::pin!(notified);
        let _ = notified.as_mut().enable();

        let state = coordinator.state.lock().await;
        if !state.running {
            return completed_upstream_stall_recovery_metadata(&metadata, &state, joined_inflight);
        }
        drop(state);

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || timeout(remaining, notified).await.is_err() {
            let state = coordinator.state.lock().await;
            if !state.running {
                return completed_upstream_stall_recovery_metadata(
                    &metadata,
                    &state,
                    joined_inflight,
                );
            }
            drop(state);

            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                if joined_inflight {
                    String::from("join_timeout")
                } else {
                    String::from("completion_timeout")
                },
            );
            return metadata;
        }
    }
}

fn completed_upstream_stall_recovery_metadata(
    metadata: &BTreeMap<String, String>,
    state: &UpstreamStallRecoveryState,
    joined_inflight: bool,
) -> BTreeMap<String, String> {
    let Some(last_result) = &state.last_result else {
        let mut missing = metadata.clone();
        missing.insert(
            String::from("upstream_stall_recovery_status"),
            String::from("missing_result"),
        );
        return missing;
    };
    if !joined_inflight {
        return last_result.clone();
    }
    let mut joined = metadata.clone();
    joined.insert(
        String::from("upstream_stall_recovery_status"),
        String::from("joined_inflight"),
    );
    if let Some(status) = last_result.get("upstream_stall_recovery_status") {
        joined.insert(
            String::from("upstream_stall_recovery_joined_status"),
            status.clone(),
        );
    }
    joined
}

const fn recovery_join_timeout(policy: &UpstreamStallPolicy) -> Duration {
    policy
        .recovery_timeout
        .saturating_add(Duration::from_secs(1))
}

async fn finish_upstream_stall_recovery(
    coordinator: &UpstreamStallRecoveryCoordinator,
    metadata: BTreeMap<String, String>,
) {
    let mut state = coordinator.state.lock().await;
    state.running = false;
    state.last_finished = Some(Instant::now());
    state.last_result = Some(metadata);
    drop(state);
    coordinator.notify.notify_waiters();
}

fn upstream_stall_recovery_metadata(configured: bool) -> BTreeMap<String, String> {
    BTreeMap::from([(
        String::from("upstream_stall_recovery_configured"),
        configured.to_string(),
    )])
}

async fn run_upstream_stall_recovery_command(
    policy: &UpstreamStallPolicy,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("upstream_stall_recovery_ran"),
        String::from("true"),
    )]);
    let program = &policy.recovery_command[0];
    let args = &policy.recovery_command[1..];
    let mut command = Command::new(program);
    command
        .args(args)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_recovery_command(&mut command);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                String::from("spawn_failed"),
            );
            metadata.insert(
                String::from("upstream_stall_recovery_error"),
                error.kind().to_string(),
            );
            return metadata;
        }
    };
    match timeout(policy.recovery_timeout, child.wait()).await {
        Ok(Ok(status)) => {
            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                if status.success() {
                    String::from("succeeded")
                } else {
                    String::from("exit_failure")
                },
            );
            if let Some(code) = status.code() {
                metadata.insert(
                    String::from("upstream_stall_recovery_exit_code"),
                    code.to_string(),
                );
            }
        }
        Ok(Err(error)) => {
            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                String::from("wait_failed"),
            );
            metadata.insert(
                String::from("upstream_stall_recovery_error"),
                error.kind().to_string(),
            );
        }
        Err(_elapsed) => {
            metadata.insert(
                String::from("upstream_stall_recovery_status"),
                String::from("timeout_killed"),
            );
            metadata.extend(terminate_timed_out_recovery_child(&mut child).await);
        }
    }
    metadata
}

#[cfg(unix)]
fn configure_recovery_command(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_recovery_command(_command: &mut Command) {}

#[cfg(unix)]
async fn terminate_timed_out_recovery_child(
    child: &mut tokio::process::Child,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("upstream_stall_recovery_timeout_cleanup_scope"),
        String::from("process_group"),
    )]);
    let Some(pid) = child.id() else {
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_cleanup_status"),
            String::from("missing_child_pid"),
        );
        let _kill_result = child.kill().await;
        return metadata;
    };

    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_sent"),
        send_recovery_process_group_signal(pid, "TERM")
            .await
            .to_string(),
    );
    tokio::time::sleep(RECOVERY_PROCESS_GROUP_TERM_GRACE).await;
    let child_reaped_after_term;
    let term_child_wait_status = match child.try_wait() {
        Ok(Some(_status)) => {
            child_reaped_after_term = true;
            "child_reaped_after_term"
        }
        Ok(None) => {
            child_reaped_after_term = false;
            "child_still_running_after_term"
        }
        Err(_error) => {
            child_reaped_after_term = false;
            "child_wait_failed_after_term"
        }
    };
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_child_wait_status"),
        String::from(term_child_wait_status),
    );

    metadata.insert(
        String::from("upstream_stall_recovery_timeout_kill_sent"),
        send_recovery_process_group_signal(pid, "KILL")
            .await
            .to_string(),
    );
    let cleanup_status = if child_reaped_after_term {
        "group_killed_after_child_reaped"
    } else {
        match timeout(RECOVERY_PROCESS_GROUP_KILL_REAP_GRACE, child.wait()).await {
            Ok(Ok(_status)) => "terminated_after_kill",
            Ok(Err(_error)) => "wait_failed_after_kill",
            Err(_elapsed) => "wait_timeout_after_kill",
        }
    };
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_cleanup_status"),
        String::from(cleanup_status),
    );
    metadata
}

#[cfg(not(unix))]
async fn terminate_timed_out_recovery_child(
    child: &mut tokio::process::Child,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("upstream_stall_recovery_timeout_cleanup_scope"),
        String::from("child"),
    )]);
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_cleanup_status"),
        child.kill().await.is_ok().to_string(),
    );
    metadata
}

#[cfg(unix)]
async fn send_recovery_process_group_signal(pid: u32, signal: &str) -> bool {
    Command::new("kill")
        .arg(format!("-{signal}"))
        .arg("--")
        .arg(format!("-{pid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

async fn start_shielded_attempt(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
) -> Result<ShieldedStartedAttempt, ShieldedAttemptFailure> {
    let attempt_id = AttemptId::for_request(&runtime.request_id, attempt_number);
    let attempt_started_at_unix_ms = unix_time_millis();
    let attempt_plan = runtime
        .retry_policy
        .attempt_plan(attempt_number, &runtime.upstream_profile.thinking);
    let (upstream_body, anti_loop_hint_applied) =
        shielded_attempt_body(runtime, attempt_number, retry_cause, &attempt_plan);
    let raw_request_body = raw_payload_text(&upstream_body);
    let evidence_upstream_body = upstream_body.clone();
    let request_metadata = shielded_attempt_request_metadata(
        runtime,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
        &attempt_plan,
    );
    match send_upstream_request(
        &runtime.client,
        runtime.method.clone(),
        runtime.upstream_url.clone(),
        &runtime.downstream_headers,
        upstream_body,
        runtime.upstream_timeout,
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
                    raw_request_body,
                    upstream_body: evidence_upstream_body,
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
                raw_payloads: RawPayloads {
                    input: raw_request_body,
                    ..RawPayloads::default()
                },
                upstream_body: evidence_upstream_body,
            })
        }
    }
}

fn raw_payload_text(bytes: &Bytes) -> Option<String> {
    std::str::from_utf8(bytes)
        .ok()
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
}

impl ShieldedAttemptInfo {
    fn into_final_context(
        self,
        extra_response_metadata: BTreeMap<String, String>,
        raw_payloads: RawPayloads,
    ) -> FinalAttemptContext {
        let mut raw_payloads = raw_payloads;
        if raw_payloads.input.is_none() {
            raw_payloads.input = self.raw_request_body;
        }
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
    attempt_plan: &ShieldedAttemptPlan,
) -> (Bytes, bool) {
    let prepared_body = match runtime.chat_kind {
        ShieldedChatKind::NonStream => shielded_chat::prepare_non_stream_request(
            &runtime.downstream_body,
            &attempt_plan.thinking,
        ),
        ShieldedChatKind::Stream => {
            shielded_chat::prepare_stream_request(&runtime.downstream_body, &attempt_plan.thinking)
        }
        ShieldedChatKind::Generic => None,
    }
    .map_or_else(
        || runtime.upstream_body.clone(),
        |request| request.upstream_body(),
    );

    if attempt_number > 1
        && runtime.retry_policy.anti_loop_hint_enabled
        && matches!(retry_cause, Some(ShieldedRetryCause::LoopDetected))
    {
        if let Some(body) = shielded_chat::body_with_anti_loop_retry_hint(
            &prepared_body,
            attempt_number,
            runtime.retry_policy.max_attempts,
            attempt_plan.anti_loop_hint.as_deref(),
        ) {
            return (body, true);
        }
    }
    (prepared_body, false)
}

fn shielded_attempt_request_metadata(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    anti_loop_hint_applied: bool,
    attempt_plan: &ShieldedAttemptPlan,
) -> BTreeMap<String, String> {
    let mut metadata = attempt_request_metadata(
        &runtime.downstream_method,
        &runtime.downstream_uri,
        &runtime.downstream_headers,
    );
    add_listener_metadata(&mut metadata, &runtime.listener);
    add_upstream_profile_metadata(
        &mut metadata,
        &runtime.upstream_profile,
        runtime.route_reason,
    );
    add_shielded_request_metadata(
        &mut metadata,
        true,
        true,
        &runtime.liveness,
        &runtime.thinking_metadata,
    );
    add_retry_attempt_metadata(
        &mut metadata,
        &runtime.retry_policy,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
        attempt_plan,
    );
    metadata
}

fn add_retry_attempt_metadata(
    metadata: &mut BTreeMap<String, String>,
    policy: &ShieldedRetryPolicy,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    anti_loop_hint_applied: bool,
    attempt_plan: &ShieldedAttemptPlan,
) {
    metadata.insert(String::from("attempt_number"), attempt_number.to_string());
    metadata.insert(
        String::from("attempt_index"),
        attempt_number.saturating_sub(1).to_string(),
    );
    metadata.insert(String::from("attempt_name"), attempt_plan.name.clone());
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
    metadata.insert(
        String::from("retry_shielded_streaming_enabled"),
        policy.shielded_streaming_enabled.to_string(),
    );
    metadata.insert(
        String::from("downstream_drop_policy"),
        policy.downstream_drop_policy.as_str().to_owned(),
    );
    metadata.insert(
        String::from("attempt_thinking_mode"),
        attempt_plan.thinking.effective_mode().as_str().to_owned(),
    );
    metadata.insert(
        String::from("attempt_thinking_budget_tokens"),
        attempt_plan.thinking.budget_tokens.to_string(),
    );
    metadata.insert(
        String::from("attempt_thinking_max_tokens"),
        attempt_plan
            .thinking
            .max_tokens
            .map_or_else(|| String::from("unset"), |value| value.to_string()),
    );
}

fn add_retry_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    policy: &ShieldedRetryPolicy,
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
    metadata.insert(
        String::from("retry_shielded_streaming_enabled"),
        policy.shielded_streaming_enabled.to_string(),
    );
    metadata.insert(
        String::from("downstream_drop_policy"),
        policy.downstream_drop_policy.as_str().to_owned(),
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
    } else if error.is_upstream_stall() {
        Some(ShieldedRetryCause::UpstreamStall)
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
    let mut raw_payloads = error.raw_payloads().clone();
    if raw_payloads.input.is_none() {
        raw_payloads.input.clone_from(&info.raw_request_body);
    }
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
        abort_reason: match retry_cause {
            Some(ShieldedRetryCause::LoopDetected) => Some(String::from("loop_guard")),
            Some(ShieldedRetryCause::UpstreamStall) => Some(String::from("upstream_stall")),
            _ => None,
        },
        request_metadata: info.request_metadata.clone(),
        response_metadata,
        raw_payloads,
        upstream_body: info.upstream_body.clone(),
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
        raw_payloads: RawPayloads {
            input: info.raw_request_body.clone(),
            ..RawPayloads::default()
        },
        upstream_body: info.upstream_body.clone(),
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
        raw_payloads: RawPayloads {
            input: info.raw_request_body.clone(),
            ..RawPayloads::default()
        },
        upstream_body: info.upstream_body.clone(),
    }
}

fn attempt_failure_record(
    failure: &ShieldedAttemptFailure,
    status: AttemptStatus,
    retry_cause: Option<ShieldedRetryCause>,
    policy: &ShieldedRetryPolicy,
) -> AttemptRecord {
    let mut response_metadata = failure.response_metadata.clone();
    copy_attempt_request_metadata(&mut response_metadata, &failure.request_metadata);
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
        raw_payloads: failure.raw_payloads.clone(),
    }
}

fn copy_attempt_request_metadata(
    response_metadata: &mut BTreeMap<String, String>,
    request_metadata: &BTreeMap<String, String>,
) {
    for key in [
        "attempt_index",
        "attempt_name",
        "retry_previous_reason",
        "retry_anti_loop_hint_applied",
        "retry_shielded_streaming_enabled",
        "downstream_drop_policy",
        "attempt_thinking_mode",
        "attempt_thinking_budget_tokens",
        "attempt_thinking_max_tokens",
    ] {
        if let Some(value) = request_metadata.get(key) {
            response_metadata.insert(key.to_owned(), value.clone());
        }
    }
}

fn started_status_attempt_record(
    info: &ShieldedAttemptInfo,
    status: AttemptStatus,
    retry_cause: Option<ShieldedRetryCause>,
    policy: &ShieldedRetryPolicy,
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
    policy: &ShieldedRetryPolicy,
) -> ShieldedFailureOutcome {
    let mut response_metadata = failure.response_metadata.clone();
    response_metadata.extend(retry_chain_metadata(
        &attempt_records,
        policy,
        RequestStatus::Failed.as_str(),
    ));
    let error_type = structured_shielded_error_type(&failure);
    ShieldedFailureOutcome {
        error_type,
        error_message: failure.error_message,
        response_metadata,
        attempt_records,
        upstream_mode: failure.upstream_mode,
    }
}

fn structured_shielded_error_type(failure: &ShieldedAttemptFailure) -> &'static str {
    if matches!(failure.retry_cause, Some(ShieldedRetryCause::LoopDetected))
        || failure.abort_reason.as_deref() == Some("loop_guard")
    {
        return "llm_guard_loop_retry_exhausted";
    }
    if failure
        .response_metadata
        .get("upstream_stall_detected")
        .is_some_and(|value| value == "true")
        || failure.error_message.contains("timeout")
    {
        return "llm_guard_attempt_timeout";
    }
    "llm_guard_upstream_error"
}

fn terminal_forward_failure(
    terminal: ShieldedTerminalForward,
    message: &str,
) -> ShieldedFailureOutcome {
    let failure = status_failure_without_retry(&terminal.started.info, message);
    let mut attempt_records = terminal.prior_attempt_records;
    let disabled_policy = ShieldedRetryPolicy {
        enabled: false,
        max_attempts: 1,
        anti_loop_hint_enabled: false,
        shielded_streaming_enabled: false,
        downstream_drop_policy: DownstreamDropPolicy::Cancel,
        ladder: Vec::new(),
    };
    attempt_records.push(attempt_failure_record(
        &failure,
        AttemptStatus::Failed,
        None,
        &disabled_policy,
    ));
    shielded_failure_outcome(failure, attempt_records, &disabled_policy)
}

fn shielded_retry_success_response(
    runtime: &ShieldedRetryRuntime,
    mut outcome: ShieldedAcceptedOutcome,
    in_flight_permit: InFlightPermit,
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
    let raw_payloads = outcome.final_attempt.raw_payloads.clone();
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: DownstreamMode::NonStreamJson,
            downstream_status: upstream_status,
            downstream_headers: response_headers.clone(),
            upstream_mode: outcome.final_attempt.upstream_mode,
            extra_response_metadata: extra_metadata,
            raw_payloads,
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
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
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

fn shielded_retry_terminal_forward_response(
    runtime: &ShieldedRetryRuntime,
    terminal: ShieldedTerminalForward,
    in_flight_permit: InFlightPermit,
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

fn shielded_retry_direct_relay_response(
    runtime: &ShieldedRetryRuntime,
    outcome: ShieldedDirectRelayOutcome,
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
    let upstream_status = outcome.started.info.upstream_status;
    let upstream_headers = outcome.started.info.upstream_headers.clone();
    let final_attempt = outcome
        .started
        .info
        .clone()
        .into_final_context(outcome.response_metadata.clone(), RawPayloads::default());
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: DownstreamMode::Streaming,
            downstream_status: upstream_status,
            downstream_headers: upstream_headers.clone(),
            upstream_mode: final_attempt.upstream_mode,
            extra_response_metadata: outcome.response_metadata,
            raw_payloads: RawPayloads::default(),
            completed_attempt_records: outcome.prior_attempt_records,
            final_attempt: Some(final_attempt),
            attempt_progress: None,
        },
    );
    let response_body = ObservedUpstreamBody::new(
        outcome.started.response.bytes_stream(),
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
        config: runtime.config.clone(),
        store: runtime.store.clone(),
        evidence_store: runtime.evidence_store.clone(),
        shadow_evidence: runtime.shadow_evidence.clone(),
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
            policy: runtime.retry_policy.clone(),
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
    config: ConfigHandle,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    shadow_evidence: ShadowEvidenceState,
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
                &retry_observation.policy,
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
        record_evidence_many(
            EvidenceRecordContext {
                config: &self.config,
                store: &self.evidence_store,
                shadow_evidence: &self.shadow_evidence,
            },
            &request_record,
            &attempts,
        );
    }
}

fn final_attempt_record(
    attempt: FinalAttemptContext,
    request_id: &RequestId,
    finished_at_unix_ms: u64,
    body_bytes: u64,
    completion: &BodyCompletion,
) -> AttemptRecord {
    let request_metadata = attempt.request_metadata;
    let mut response_metadata = response_metadata(
        attempt.upstream_status,
        &attempt.upstream_headers,
        body_bytes,
        finished_at_unix_ms.saturating_sub(attempt.started_at_unix_ms),
    );
    response_metadata.extend(attempt.extra_response_metadata);
    copy_attempt_request_metadata(&mut response_metadata, &request_metadata);
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
        request_metadata,
        response_metadata,
        raw_payloads: attempt.raw_payloads,
    }
}

fn retry_chain_metadata(
    attempts: &[AttemptRecord],
    policy: &ShieldedRetryPolicy,
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
            String::from("retry_shielded_streaming_enabled"),
            policy.shielded_streaming_enabled.to_string(),
        ),
        (
            String::from("downstream_drop_policy"),
            policy.downstream_drop_policy.as_str().to_owned(),
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
    _in_flight_permit: InFlightPermit,
    bytes_seen: u64,
    terminal_completion: BodyCompletion,
}

impl ObservedUpstreamBody {
    fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
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
        in_flight_permit: InFlightPermit,
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
    _in_flight_permit: InFlightPermit,
    bytes_seen: u64,
    terminal_completion: BodyCompletion,
}

impl ObservedBufferedBody {
    fn new(body: Bytes, observer: ForwardedBodyObserver, in_flight_permit: InFlightPermit) -> Self {
        Self::new_with_completion(body, observer, in_flight_permit, BodyCompletion::Succeeded)
    }

    fn new_with_completion(
        body: Bytes,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
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

enum ShieldedAggregateOutcome {
    Accepted(ShieldedAcceptedOutcome),
    DirectRelay(ShieldedDirectRelayOutcome),
}

type ShieldedAggregateFuture =
    Pin<Box<dyn Future<Output = Result<ShieldedAggregateOutcome, ShieldedFailureOutcome>> + Send>>;
type DirectRelayStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShieldedAcceptedResponseMode {
    JsonCompletion,
    OpenAiSse,
}

struct ShieldedLivenessBody {
    aggregate: ShieldedAggregateFuture,
    direct_stream: Option<DirectRelayStream>,
    interval: Interval,
    mode: ShieldedLivenessMode,
    accepted_response_mode: ShieldedAcceptedResponseMode,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: InFlightPermit,
    bytes_seen: u64,
    terminal_completion: Option<BodyCompletion>,
    json_prefix_pending: bool,
}

impl ShieldedLivenessBody {
    fn new(
        aggregate: ShieldedAggregateFuture,
        mode: ShieldedLivenessMode,
        accepted_response_mode: ShieldedAcceptedResponseMode,
        interval_secs: u64,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
    ) -> Self {
        let period = Duration::from_secs(interval_secs);
        let mut interval = tokio::time::interval_at(Instant::now() + period, period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            aggregate,
            direct_stream: None,
            interval,
            mode,
            accepted_response_mode,
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

    fn accepted_chunk(&self, outcome: &ShieldedAcceptedOutcome) -> Bytes {
        match self.accepted_response_mode {
            ShieldedAcceptedResponseMode::OpenAiSse => outcome.sse_body.clone(),
            ShieldedAcceptedResponseMode::JsonCompletion => match self.mode {
                ShieldedLivenessMode::Sse => sse_final_frame(&outcome.body),
                ShieldedLivenessMode::JsonWhitespace | ShieldedLivenessMode::Disabled => {
                    outcome.body.clone()
                }
            },
        }
    }

    fn error_chunk(&self, error_type: &str, error: &str) -> Bytes {
        let body = proxy_error_json_body(error_type, error);
        match self.mode {
            ShieldedLivenessMode::Sse => sse_error_frame(&body),
            ShieldedLivenessMode::JsonWhitespace | ShieldedLivenessMode::Disabled => body,
        }
    }

    fn start_direct_relay(&mut self, outcome: ShieldedDirectRelayOutcome) {
        let mut final_attempt = outcome
            .started
            .info
            .into_final_context(outcome.response_metadata.clone(), RawPayloads::default());
        if let Some(observer) = &mut self.observer {
            observer
                .completed_attempt_records
                .clone_from(&outcome.prior_attempt_records);
            observer
                .extra_response_metadata
                .extend(outcome.response_metadata);
            final_attempt
                .extra_response_metadata
                .extend(observer.extra_response_metadata.clone());
            if let Some(progress) = &observer.attempt_progress {
                let mut progress = shielded_attempt_progress(progress);
                progress.completed_attempt_records = outcome.prior_attempt_records;
                progress.current_attempt = Some(final_attempt.clone());
            }
            observer.final_attempt = Some(final_attempt);
        }
        self.direct_stream = Some(Box::pin(outcome.started.response.bytes_stream()));
    }

    fn poll_direct_stream(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Bytes, Infallible>>> {
        let Some(stream) = &mut self.direct_stream else {
            return Poll::Pending;
        };
        match stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => self.count_and_emit(bytes),
            Poll::Ready(Some(Err(error))) => {
                let error_message = sanitized_reqwest_error(&error);
                let chunk = self.error_chunk("llm_guard_upstream_error", &error_message);
                self.terminal_completion = Some(BodyCompletion::UpstreamStreamError(error_message));
                self.direct_stream = None;
                self.count_and_emit(chunk)
            }
            Poll::Ready(None) => {
                self.direct_stream = None;
                self.record_once(&BodyCompletion::Succeeded);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
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

        if this.direct_stream.is_some() {
            return this.poll_direct_stream(cx);
        }

        match this.aggregate.as_mut().poll(cx) {
            Poll::Ready(Ok(ShieldedAggregateOutcome::Accepted(outcome))) => {
                let chunk = this.accepted_chunk(&outcome);
                if let Some(observer) = &mut this.observer {
                    observer.completed_attempt_records = outcome.prior_attempt_records;
                    observer
                        .extra_response_metadata
                        .extend(outcome.response_metadata.clone());
                    let mut final_attempt = outcome.final_attempt;
                    final_attempt
                        .extra_response_metadata
                        .extend(observer.extra_response_metadata.clone());
                    observer.raw_payloads = final_attempt.raw_payloads.clone();
                    observer.final_attempt = Some(final_attempt);
                }
                this.terminal_completion = Some(BodyCompletion::Succeeded);
                return this.count_and_emit(chunk);
            }
            Poll::Ready(Ok(ShieldedAggregateOutcome::DirectRelay(outcome))) => {
                this.start_direct_relay(outcome);
                return this.poll_direct_stream(cx);
            }
            Poll::Ready(Err(failure)) => {
                if let Some(observer) = &mut this.observer {
                    observer.completed_attempt_records = failure.attempt_records;
                    observer
                        .extra_response_metadata
                        .extend(failure.response_metadata);
                    observer.final_attempt = None;
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
        || is_admin_only_request_header(name)
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

fn is_admin_only_request_header(name: &HeaderName) -> bool {
    matches!(name.as_str(), "x-admin-token")
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

fn add_listener_metadata(metadata: &mut BTreeMap<String, String>, listener: &ListenerConfig) {
    metadata.insert(String::from("listener_name"), listener.name.clone());
    metadata.insert(
        String::from("listener_bind_host"),
        listener.bind_host.clone(),
    );
    metadata.insert(String::from("listener_port"), listener.port.to_string());
    metadata.insert(
        String::from("listener_restricted"),
        listener.allowed_upstreams.is_some().to_string(),
    );
}

fn select_shielded_liveness(
    state: &ProxyState,
    config: &AppConfig,
    body: &Bytes,
    kind: ShieldedChatKind,
    now_unix_ms: u64,
) -> ShieldedLivenessSelection {
    let shielded_chat = !matches!(kind, ShieldedChatKind::Generic);
    let input_fingerprint = shielded_chat
        .then(|| chat_input_fingerprint(body))
        .flatten();
    let repeat_observation = input_fingerprint
        .as_deref()
        .filter(|_fingerprint| !config.loop_guard.effective_mode().is_disabled())
        .map_or_else(RepeatInputObservation::default, |fingerprint| {
            state.repeat_inputs.observe(
                fingerprint,
                now_unix_ms,
                config.loop_guard.normalized_input_window_secs,
                config.loop_guard.max_repeated_inputs,
            )
        });
    let mode = match kind {
        ShieldedChatKind::Stream => ShieldedLivenessMode::Sse,
        ShieldedChatKind::NonStream | ShieldedChatKind::Generic => {
            // Non-stream OpenAI-compatible clients require JSON framing even when the
            // proxy internally forces upstream SSE for inspection and retry.
            match config.heartbeat.mode {
                HeartbeatMode::JsonWhitespace => ShieldedLivenessMode::JsonWhitespace,
                HeartbeatMode::Sse if repeat_observation.repeated => {
                    ShieldedLivenessMode::JsonWhitespace
                }
                HeartbeatMode::Disabled | HeartbeatMode::Sse => ShieldedLivenessMode::Disabled,
            }
        }
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
        "maxtokens"
            | "maxcompletiontokens"
            | "maxoutputtokens"
            | "budgettokens"
            | "thinkingtokenbudget"
    )
}

fn request_body_bytes_hint(headers: &HeaderMap) -> String {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map_or_else(|| String::from("unknown"), |bytes| bytes.to_string())
}

fn add_upstream_profile_metadata(
    metadata: &mut BTreeMap<String, String>,
    profile: &UpstreamProfileConfig,
    route_reason: UpstreamRouteReason,
) {
    let context_window = profile.metadata.context_window_override();
    metadata.insert(String::from("upstream_profile"), profile.name.clone());
    metadata.insert(
        String::from("upstream_route_reason"),
        route_reason.as_str().to_owned(),
    );
    metadata.insert(
        String::from("upstream_request_timeout_ms"),
        profile.request_timeout_ms.to_string(),
    );
    metadata.insert(
        String::from("upstream_context_window_configured"),
        context_window.is_some().to_string(),
    );
    metadata.insert(
        String::from("upstream_context_window_tokens"),
        context_window.map_or_else(|| String::from("unknown"), |value| value.to_string()),
    );
    metadata.insert(
        String::from("upstream_input_token_safety_margin"),
        profile.metadata.input_token_safety_margin.to_string(),
    );
}

fn add_shielded_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    shielded_chat: bool,
    thinking_policy_applied: bool,
    liveness: &ShieldedLivenessSelection,
    thinking_metadata: &BTreeMap<String, String>,
) {
    if shielded_chat {
        add_shielded_chat_request_metadata(metadata);
        add_shielded_liveness_request_metadata(metadata, liveness);
    }
    if thinking_policy_applied {
        metadata.insert(
            String::from("policy_transform_applied"),
            String::from("true"),
        );
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
    attempt_number: u32,
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
        attempt_number: input.attempt_number,
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

struct FailedRequestRecord {
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    http_status: u16,
    error_type: &'static str,
    error_reason: String,
    request_metadata: BTreeMap<String, String>,
    attempts: Vec<AttemptRecord>,
}

fn record_queued_admission_cancel(record: QueuedAdmissionCancelRecord) {
    let context = record.context;
    let finished_at_unix_ms = unix_time_millis();
    let queue_wait_ms = duration_millis_u64(record.queued_at.elapsed());
    let error_type = "proxy_generation_queue_cancelled";
    let abort_reason = "downstream_disconnected_while_queued";
    let mut request_metadata = context.request_metadata;
    request_metadata.extend(BTreeMap::from([
        (
            String::from("admission_outcome"),
            String::from("queue_cancelled"),
        ),
        (String::from("queue_wait_ms"), queue_wait_ms.to_string()),
        (
            String::from("generation_queue_timeout_ms"),
            record.timeout_ms.to_string(),
        ),
    ]));
    let mut response_metadata =
        failed_response_metadata(context.started_at_unix_ms, finished_at_unix_ms, error_type);
    response_metadata.insert(String::from("abort_reason"), abort_reason.to_owned());
    let request_record = RequestRecord {
        request_id: context.request_id,
        started_at_unix_ms: context.started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        downstream_mode: DownstreamMode::NonStreamJson,
        upstream_mode: UpstreamMode::NotApplicable,
        model_id: None,
        input_fingerprint: None,
        status: RequestStatus::Aborted,
        http_status: None,
        error_reason: Some(format!("{error_type}: {abort_reason}")),
        abort_reason: Some(abort_reason.to_owned()),
        request_metadata,
        response_metadata,
        raw_payloads: RawPayloads::default(),
    };
    record_observability_many(&context.store, &request_record, &[]);
}

fn record_failed_request(store: &ObservabilityStore, failure: FailedRequestRecord) {
    let mut response_metadata = failed_response_metadata(
        failure.started_at_unix_ms,
        failure.finished_at_unix_ms,
        failure.error_type,
    );
    if let Some(attempt) = failure.attempts.last() {
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
    record_observability_many(store, &request_record, &failure.attempts);
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

#[derive(Clone, Copy)]
struct EvidenceRecordContext<'record> {
    config: &'record ConfigHandle,
    store: &'record EvidenceStore,
    shadow_evidence: &'record ShadowEvidenceState,
}

fn record_evidence_many(
    context: EvidenceRecordContext<'_>,
    request: &RequestRecord,
    attempts: &[AttemptRecord],
) {
    let settings = match context.config.snapshot() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("failed to read evidence settings: {error}");
            return;
        }
    };
    if !settings.evidence.enabled {
        return;
    }

    let group_id = request.request_id.as_str().to_owned();
    let group = EvidenceGroupRecord {
        group_id: group_id.clone(),
        request_id: request.request_id.clone(),
        started_at_unix_ms: request.started_at_unix_ms,
        finished_at_unix_ms: request.finished_at_unix_ms,
        model_id: request.model_id.clone(),
        status: request.status.as_str().to_owned(),
        request_metadata: request.request_metadata.clone(),
        response_metadata: request.response_metadata.clone(),
    };
    let mut evidence_attempts = attempts
        .iter()
        .map(|attempt| evidence_attempt_from_observability(&group_id, request, attempt))
        .collect::<Vec<_>>();
    evidence_attempts.extend(context.shadow_evidence.snapshot());

    match context.store.record_group(&group, &evidence_attempts) {
        Ok(EvidenceStoreWrite::Written) => {
            for shadow in context.shadow_evidence.snapshot() {
                match context.store.record_shadow_attempt(&shadow) {
                    Ok(EvidenceStoreWrite::Written | EvidenceStoreWrite::Disabled) | Err(_) => {}
                }
            }
        }
        Ok(EvidenceStoreWrite::Disabled) => {}
        Err(error) => eprintln!("failed to write evidence ledger: {error}"),
    }
}

fn evidence_attempt_from_observability(
    group_id: &str,
    request: &RequestRecord,
    attempt: &AttemptRecord,
) -> EvidenceAttemptRecord {
    let role = evidence_attempt_role(attempt);
    let shown_to_downstream = attempt_shown_to_downstream(request, attempt);
    EvidenceAttemptRecord {
        attempt_id: attempt.attempt_id.clone(),
        group_id: group_id.to_owned(),
        request_id: request.request_id.clone(),
        attempt_number: attempt.attempt_number,
        role,
        shown_to_downstream,
        started_at_unix_ms: attempt.started_at_unix_ms,
        finished_at_unix_ms: attempt.finished_at_unix_ms,
        upstream_profile: metadata_value(&attempt.request_metadata, "upstream_profile")
            .or_else(|| metadata_value(&attempt.response_metadata, "upstream_profile")),
        model_id: request.model_id.clone(),
        thinking_mode: metadata_value(&attempt.request_metadata, "attempt_thinking_mode")
            .or_else(|| metadata_value(&attempt.response_metadata, "attempt_thinking_mode")),
        thinking_budget_tokens: metadata_u32(
            &attempt.request_metadata,
            "attempt_thinking_budget_tokens",
        )
        .or_else(|| metadata_u32(&attempt.response_metadata, "attempt_thinking_budget_tokens")),
        thinking_max_tokens: metadata_u32(&attempt.request_metadata, "attempt_thinking_max_tokens")
            .or_else(|| metadata_u32(&attempt.response_metadata, "attempt_thinking_max_tokens")),
        detector_features: evidence_detector_features(&attempt.response_metadata),
        status: evidence_attempt_status(attempt, shown_to_downstream),
        http_status: attempt.http_status,
        error_reason: attempt.error_reason.clone(),
        retry_reason: attempt.retry_reason.clone(),
        abort_reason: attempt.abort_reason.clone(),
        shadow_skip_reason: None,
        request_metadata: attempt.request_metadata.clone(),
        response_metadata: attempt.response_metadata.clone(),
        raw_payloads: attempt.raw_payloads.clone(),
    }
}

fn evidence_attempt_role(attempt: &AttemptRecord) -> EvidenceAttemptRole {
    if attempt.attempt_number <= 1 {
        EvidenceAttemptRole::Primary
    } else {
        EvidenceAttemptRole::Fallback
    }
}

fn attempt_shown_to_downstream(request: &RequestRecord, attempt: &AttemptRecord) -> bool {
    request.status == RequestStatus::Succeeded
        && attempt.status == AttemptStatus::Succeeded
        && attempt.retry_reason.is_none()
        && attempt.abort_reason.is_none()
}

fn evidence_attempt_status(
    attempt: &AttemptRecord,
    shown_to_downstream: bool,
) -> EvidenceAttemptStatus {
    match attempt.status {
        AttemptStatus::Succeeded if shown_to_downstream => EvidenceAttemptStatus::Accepted,
        AttemptStatus::Succeeded => EvidenceAttemptStatus::Accepted,
        AttemptStatus::Retried => EvidenceAttemptStatus::Rejected,
        AttemptStatus::Failed => EvidenceAttemptStatus::Failed,
        AttemptStatus::Aborted => EvidenceAttemptStatus::Aborted,
    }
}

fn evidence_detector_features(metadata: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    metadata
        .iter()
        .filter(|(key, _value)| {
            key.starts_with("loop_")
                || matches!(
                    key.as_str(),
                    "response_body_bytes"
                        | "delta_count"
                        | "content_delta_count"
                        | "reasoning_delta_count"
                        | "tool_call_delta_count"
                        | "finish_reason"
                )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn maybe_schedule_shadow_continuation(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
    source: &AttemptRecord,
) {
    if failure.retry_cause != Some(ShieldedRetryCause::LoopDetected) {
        return;
    }
    let settings = match runtime.config.snapshot() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("failed to read shadow evidence settings: {error}");
            return;
        }
    };
    let shadow = &settings.evidence.shadow;
    if !settings.evidence.enabled || !shadow.enabled || !shadow.keep_looping_attempt_running {
        return;
    }

    if shadow.max_shadow_attempts_per_request == 0 {
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            &settings,
            ShadowSkipReason::PerRequestLimit,
        );
        return;
    }
    if shadow.max_global_shadow_in_flight == 0 {
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            &settings,
            ShadowSkipReason::GlobalLimit,
        );
        return;
    }
    let Some(permit) = runtime
        .shadow_attempts
        .try_acquire(shadow.max_global_shadow_in_flight)
    else {
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            &settings,
            ShadowSkipReason::GlobalLimit,
        );
        return;
    };
    let Some(sequence) = runtime
        .shadow_evidence
        .try_reserve_attempt(shadow.max_shadow_attempts_per_request)
    else {
        drop(permit);
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            &settings,
            ShadowSkipReason::PerRequestLimit,
        );
        return;
    };

    let Some(attempt_id) = shadow_attempt_id(&runtime.request_id, source.attempt_number, sequence)
    else {
        drop(permit);
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            &settings,
            ShadowSkipReason::ContinuationUnavailable,
        );
        return;
    };
    let task = ShadowContinuationTask {
        client: runtime.client.clone(),
        method: runtime.method.clone(),
        upstream_url: runtime.upstream_url.clone(),
        downstream_headers: runtime.downstream_headers.clone(),
        upstream_body: failure.upstream_body.clone(),
        upstream_timeout: runtime.upstream_timeout,
        evidence_store: runtime.evidence_store.clone(),
        shadow_evidence: runtime.shadow_evidence.clone(),
        request_id: runtime.request_id.clone(),
        group_id: runtime.request_id.as_str().to_owned(),
        attempt_id,
        source: source.clone(),
        model_id: runtime.model_id.clone(),
        loop_context: runtime.loop_context.clone(),
        shadow_attempt_timeout_ms: shadow.shadow_attempt_timeout_ms,
        parallel_downgrade_attempts: shadow.parallel_downgrade_attempts,
        _permit: permit,
    };
    tokio::spawn(async move {
        let record = run_shadow_continuation(task).await;
        record.shadow_evidence.push_record(record.attempt.clone());
        match record.evidence_store.record_shadow_attempt(&record.attempt) {
            Ok(EvidenceStoreWrite::Written | EvidenceStoreWrite::Disabled) | Err(_) => {}
        }
    });
}

struct ShadowContinuationTask {
    client: Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: HeaderMap,
    upstream_body: Bytes,
    upstream_timeout: Duration,
    evidence_store: EvidenceStore,
    shadow_evidence: ShadowEvidenceState,
    request_id: RequestId,
    group_id: String,
    attempt_id: AttemptId,
    source: AttemptRecord,
    model_id: Option<String>,
    loop_context: shielded_chat::LoopInspectionContext,
    shadow_attempt_timeout_ms: u64,
    parallel_downgrade_attempts: bool,
    _permit: InFlightPermit,
}

struct ShadowContinuationRecord {
    evidence_store: EvidenceStore,
    shadow_evidence: ShadowEvidenceState,
    attempt: EvidenceAttemptRecord,
}

async fn run_shadow_continuation(task: ShadowContinuationTask) -> ShadowContinuationRecord {
    let evidence_store = task.evidence_store.clone();
    let shadow_evidence = task.shadow_evidence.clone();
    let started_at_unix_ms = unix_time_millis();
    let timeout_ms = task.shadow_attempt_timeout_ms;
    let outcome = timeout(
        Duration::from_millis(timeout_ms),
        run_shadow_continuation_request(&task, started_at_unix_ms),
    )
    .await;
    let attempt = match outcome {
        Ok(attempt) => attempt,
        Err(_elapsed) => build_shadow_terminal_record(
            &task,
            ShadowTerminalInput {
                started_at_unix_ms,
                finished_at_unix_ms: unix_time_millis(),
                status: EvidenceAttemptStatus::ShadowTimeout,
                http_status: None,
                response_headers: HeaderMap::new(),
                response_metadata: BTreeMap::new(),
                raw_payloads: RawPayloads::default(),
                response_body_bytes: 0,
                error_reason: Some(String::from("shadow continuation timed out")),
                abort_reason: Some(String::from("shadow_timeout")),
            },
        ),
    };
    ShadowContinuationRecord {
        evidence_store,
        shadow_evidence,
        attempt,
    }
}

async fn run_shadow_continuation_request(
    task: &ShadowContinuationTask,
    started_at_unix_ms: u64,
) -> EvidenceAttemptRecord {
    let mut response_headers = HeaderMap::new();
    let mut http_status = None;
    let mut response_metadata = BTreeMap::new();
    let mut raw_payloads = RawPayloads::default();
    let mut response_body_bytes = 0_u64;
    let mut error_reason = None;

    let response = send_upstream_request(
        &task.client,
        task.method.clone(),
        task.upstream_url.clone(),
        &task.downstream_headers,
        task.upstream_body.clone(),
        task.upstream_timeout,
    )
    .await;
    let status = match response {
        Ok(response) => {
            let status = response.status();
            http_status = Some(status.as_u16());
            response_headers = response.headers().clone();
            if status.is_success() && is_event_stream(&response_headers) {
                match shielded_chat::aggregate_stream(
                    response.bytes_stream(),
                    started_at_unix_ms,
                    task.request_id.as_str(),
                    task.model_id.as_deref(),
                    task.loop_context.clone(),
                    None,
                )
                .await
                {
                    Ok(aggregated) => {
                        response_body_bytes =
                            aggregated.sse_body.len().try_into().unwrap_or(u64::MAX);
                        response_metadata = aggregated.response_metadata;
                        raw_payloads = aggregated.raw_payloads;
                        EvidenceAttemptStatus::Accepted
                    }
                    Err(error) => {
                        response_metadata = error.response_metadata().clone();
                        raw_payloads = error.raw_payloads().clone();
                        error_reason = Some(format!(
                            "shadow continuation SSE aggregation failed: {error}"
                        ));
                        EvidenceAttemptStatus::Failed
                    }
                }
            } else {
                match read_shadow_response_body(response).await {
                    Ok(body) => {
                        response_body_bytes = body.len().try_into().unwrap_or(u64::MAX);
                        raw_payloads.output = raw_payload_text(&body);
                        if status.is_success() {
                            EvidenceAttemptStatus::Accepted
                        } else {
                            error_reason =
                                Some(format!("shadow continuation HTTP {}", status.as_u16()));
                            EvidenceAttemptStatus::Failed
                        }
                    }
                    Err(error) => {
                        error_reason = Some(error);
                        EvidenceAttemptStatus::Failed
                    }
                }
            }
        }
        Err(error) => {
            error_reason = Some(error.to_string());
            EvidenceAttemptStatus::Failed
        }
    };

    build_shadow_terminal_record(
        task,
        ShadowTerminalInput {
            started_at_unix_ms,
            finished_at_unix_ms: unix_time_millis(),
            status,
            http_status,
            response_headers,
            response_metadata,
            raw_payloads,
            response_body_bytes,
            error_reason,
            abort_reason: None,
        },
    )
}

async fn read_shadow_response_body(response: reqwest::Response) -> Result<Bytes, String> {
    let mut stream = response.bytes_stream();
    let mut body = BytesMut::new();
    let mut bytes_seen = 0_usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            format!(
                "shadow continuation stream failed: {}",
                sanitized_reqwest_error(&error)
            )
        })?;
        bytes_seen = bytes_seen
            .checked_add(chunk.len())
            .ok_or_else(|| String::from("shadow continuation body is too large"))?;
        if bytes_seen > MAX_PROXY_BODY_BYTES {
            return Err(format!(
                "shadow continuation body exceeded proxy limit: max_bytes={MAX_PROXY_BODY_BYTES}"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

struct ShadowTerminalInput {
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    status: EvidenceAttemptStatus,
    http_status: Option<u16>,
    response_headers: HeaderMap,
    response_metadata: BTreeMap<String, String>,
    raw_payloads: RawPayloads,
    response_body_bytes: u64,
    error_reason: Option<String>,
    abort_reason: Option<String>,
}

fn build_shadow_terminal_record(
    task: &ShadowContinuationTask,
    input: ShadowTerminalInput,
) -> EvidenceAttemptRecord {
    let mut response_metadata = response_metadata(
        reqwest::StatusCode::from_u16(input.http_status.unwrap_or(599))
            .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
        &input.response_headers,
        input.response_body_bytes,
        input
            .finished_at_unix_ms
            .saturating_sub(input.started_at_unix_ms),
    );
    response_metadata.extend(input.response_metadata);
    add_shadow_metadata(
        &mut response_metadata,
        &task.source,
        task.shadow_attempt_timeout_ms,
        task.parallel_downgrade_attempts,
    );
    response_metadata.insert(
        String::from("shadow_terminal_status"),
        input.status.as_str().to_owned(),
    );

    EvidenceAttemptRecord {
        attempt_id: task.attempt_id.clone(),
        group_id: task.group_id.clone(),
        request_id: task.request_id.clone(),
        attempt_number: task.source.attempt_number,
        role: EvidenceAttemptRole::ShadowContinued,
        shown_to_downstream: false,
        started_at_unix_ms: input.started_at_unix_ms,
        finished_at_unix_ms: Some(input.finished_at_unix_ms),
        upstream_profile: metadata_value(&task.source.request_metadata, "upstream_profile"),
        model_id: task.model_id.clone(),
        thinking_mode: metadata_value(&task.source.request_metadata, "attempt_thinking_mode"),
        thinking_budget_tokens: metadata_u32(
            &task.source.request_metadata,
            "attempt_thinking_budget_tokens",
        ),
        thinking_max_tokens: metadata_u32(
            &task.source.request_metadata,
            "attempt_thinking_max_tokens",
        ),
        detector_features: evidence_detector_features(&task.source.response_metadata),
        status: input.status,
        http_status: input.http_status,
        error_reason: input.error_reason,
        retry_reason: None,
        abort_reason: input.abort_reason,
        shadow_skip_reason: None,
        request_metadata: task.source.request_metadata.clone(),
        response_metadata,
        raw_payloads: {
            let mut raw_payloads = input.raw_payloads;
            if raw_payloads.input.is_none() {
                raw_payloads.input = raw_payload_text(&task.upstream_body);
            }
            raw_payloads
        },
    }
}

fn push_shadow_skipped_record(
    state: &ShadowEvidenceState,
    runtime: &ShieldedRetryRuntime,
    source: &AttemptRecord,
    settings: &AppConfig,
    skip_reason: ShadowSkipReason,
) {
    if let Some(record) = build_shadow_skipped_record(runtime, source, settings, skip_reason) {
        state.push_record(record);
    }
}

fn build_shadow_skipped_record(
    runtime: &ShieldedRetryRuntime,
    source: &AttemptRecord,
    settings: &AppConfig,
    skip_reason: ShadowSkipReason,
) -> Option<EvidenceAttemptRecord> {
    let attempt_id = AttemptId::from_string(format!(
        "{}-shadow-{}-skipped",
        runtime.request_id.as_str(),
        source.attempt_number
    ))
    .ok()?;
    let finished_at = source.finished_at_unix_ms;
    let mut response_metadata = BTreeMap::new();
    response_metadata.insert(String::from("shadow_skipped"), String::from("true"));
    response_metadata.insert(
        String::from("shadow_skip_reason"),
        skip_reason.as_str().to_owned(),
    );
    add_shadow_metadata(
        &mut response_metadata,
        source,
        settings.evidence.shadow.shadow_attempt_timeout_ms,
        settings.evidence.shadow.parallel_downgrade_attempts,
    );
    response_metadata.extend(evidence_detector_features(&source.response_metadata));
    Some(EvidenceAttemptRecord {
        attempt_id,
        group_id: runtime.request_id.as_str().to_owned(),
        request_id: runtime.request_id.clone(),
        attempt_number: source.attempt_number,
        role: EvidenceAttemptRole::ShadowContinued,
        shown_to_downstream: false,
        started_at_unix_ms: source
            .finished_at_unix_ms
            .unwrap_or(source.started_at_unix_ms),
        finished_at_unix_ms: finished_at,
        upstream_profile: metadata_value(&source.request_metadata, "upstream_profile"),
        model_id: runtime.model_id.clone(),
        thinking_mode: metadata_value(&source.request_metadata, "attempt_thinking_mode"),
        thinking_budget_tokens: metadata_u32(
            &source.request_metadata,
            "attempt_thinking_budget_tokens",
        ),
        thinking_max_tokens: metadata_u32(&source.request_metadata, "attempt_thinking_max_tokens"),
        detector_features: evidence_detector_features(&source.response_metadata),
        status: EvidenceAttemptStatus::Skipped,
        http_status: source.http_status,
        error_reason: None,
        retry_reason: None,
        abort_reason: None,
        shadow_skip_reason: Some(skip_reason),
        request_metadata: source.request_metadata.clone(),
        response_metadata,
        raw_payloads: RawPayloads::default(),
    })
}

fn shadow_attempt_id(
    request_id: &RequestId,
    source_attempt_number: u32,
    sequence: u32,
) -> Option<AttemptId> {
    AttemptId::from_string(format!(
        "{}-shadow-{}-{sequence}",
        request_id.as_str(),
        source_attempt_number
    ))
    .ok()
}

fn add_shadow_metadata(
    metadata: &mut BTreeMap<String, String>,
    source: &AttemptRecord,
    timeout_ms: u64,
    parallel_downgrade_attempts: bool,
) {
    metadata.insert(String::from("shadow_continuation"), String::from("true"));
    metadata.insert(
        String::from("shadow_source_attempt_id"),
        source.attempt_id.as_str().to_owned(),
    );
    metadata.insert(
        String::from("shadow_attempt_timeout_ms"),
        timeout_ms.to_string(),
    );
    metadata.insert(
        String::from("shadow_parallel_downgrade_attempts"),
        parallel_downgrade_attempts.to_string(),
    );
    metadata.extend(evidence_detector_features(&source.response_metadata));
}

fn metadata_value(metadata: &BTreeMap<String, String>, key: &str) -> Option<String> {
    metadata
        .get(key)
        .filter(|value| value.as_str() != "unset" && value.as_str() != "unknown")
        .cloned()
}

fn metadata_u32(metadata: &BTreeMap<String, String>, key: &str) -> Option<u32> {
    metadata_value(metadata, key).and_then(|value| value.parse::<u32>().ok())
}

fn json_response(status: StatusCode, body: String) -> Response<Body> {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
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
    proxy_error_response_with_code(status, error_type, message, None, None)
}

fn proxy_error_response_with_code(
    status: StatusCode,
    error_type: &str,
    message: &str,
    code: Option<&str>,
    param: Option<&str>,
) -> Response<Body> {
    let mut error = serde_json::Map::from_iter([
        (String::from("type"), json!(error_type)),
        (String::from("message"), json!(message)),
    ]);
    if let Some(code) = code {
        error.insert(String::from("code"), json!(code));
    }
    if let Some(param) = param {
        error.insert(String::from("param"), json!(param));
    }
    let mut response = Response::new(Body::from(
        serde_json::Value::Object(serde_json::Map::from_iter([(
            String::from("error"),
            serde_json::Value::Object(error),
        )]))
        .to_string(),
    ));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn proxy_error_response_from_error(error: &ProxyError) -> Response<Body> {
    match error {
        ProxyError::Admission { failure, .. } => admission_error_response(
            failure.status(),
            failure.error_type(),
            &failure.to_string(),
            failure.retry_after(),
        ),
        ProxyError::ContextBudgetExceeded {
            message,
            param,
            code,
            ..
        } => proxy_error_response_with_code(
            error.status(),
            error.error_type(),
            message,
            Some(code),
            Some(param),
        ),
        _ => proxy_error_response(error.status(), error.error_type(), &error.to_string()),
    }
}

fn admission_error_response(
    status: StatusCode,
    error_type: &str,
    message: &str,
    retry_after: Option<String>,
) -> Response<Body> {
    let mut response = proxy_error_response(status, error_type, message);
    if let Some(retry_after) = retry_after {
        if let Ok(value) = HeaderValue::from_str(&retry_after) {
            response.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    response
}

fn unix_time_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
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
    #[error("{failure}")]
    Admission {
        failure: AdmissionFailure,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("{failure}")]
    ListenerUpstreamDenied {
        failure: ListenerUpstreamDenied,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("guard workflow blocked request: {reason}")]
    GuardBlocked {
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
    #[error("{message}")]
    ContextBudgetExceeded {
        message: String,
        param: &'static str,
        code: &'static str,
        request_metadata: Option<BTreeMap<String, String>>,
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

    fn admission(failure: AdmissionFailure) -> Self {
        Self::Admission {
            failure,
            request_metadata: None,
        }
    }

    fn listener_denied(failure: ListenerUpstreamDenied) -> Self {
        Self::ListenerUpstreamDenied {
            failure,
            request_metadata: None,
        }
    }

    fn upstream_body(reason: String) -> Self {
        Self::UpstreamBody {
            reason,
            observability: None,
        }
    }

    fn guard_blocked(reason: String) -> Self {
        Self::GuardBlocked {
            reason,
            request_metadata: None,
        }
    }

    fn context_budget_exceeded(estimate: ContextBudgetEstimate) -> Self {
        Self::ContextBudgetExceeded {
            message: estimate.message(),
            param: estimate.param,
            code: "context_budget_exceeded",
            request_metadata: Some(estimate.metadata("rejected")),
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::RequestBody { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidMethod { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidRequestPath(error) => error.status(),
            Self::Admission { failure, .. } => failure.status(),
            Self::GuardBlocked { .. } => StatusCode::FORBIDDEN,
            Self::ListenerUpstreamDenied { .. } | Self::ContextBudgetExceeded { .. } => {
                StatusCode::BAD_REQUEST
            }
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
            Self::Admission { failure, .. } => failure.error_type(),
            Self::ListenerUpstreamDenied { .. } => "listener_upstream_not_allowed",
            Self::GuardBlocked { .. } => "guard_blocked",
            Self::ContextBudgetExceeded { .. } => "invalid_request_error",
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
            }
            | Self::Admission {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::ListenerUpstreamDenied {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::GuardBlocked {
                request_metadata: Some(request_metadata),
                ..
            }
            | Self::ContextBudgetExceeded {
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
            | Self::Admission {
                request_metadata: None,
                ..
            }
            | Self::ListenerUpstreamDenied {
                request_metadata: None,
                ..
            }
            | Self::GuardBlocked {
                request_metadata: None,
                ..
            }
            | Self::ContextBudgetExceeded {
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

    fn attempt_records(&self) -> Vec<AttemptRecord> {
        match self {
            Self::UpstreamTransport {
                observability: Some(observability),
                ..
            }
            | Self::UpstreamBody {
                observability: Some(observability),
                ..
            } => {
                let mut attempts = observability.completed_attempt_records.clone();
                attempts.push(observability.attempt_record.clone());
                attempts
            }
            Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::Admission { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::GuardBlocked { .. }
            | Self::ContextBudgetExceeded { .. }
            | Self::UpstreamTransport {
                observability: None,
                ..
            }
            | Self::UpstreamBody {
                observability: None,
                ..
            } => Vec::new(),
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
            Self::Admission { failure, .. } => Self::Admission {
                failure,
                request_metadata: Some(request_metadata),
            },
            Self::ListenerUpstreamDenied { failure, .. } => Self::ListenerUpstreamDenied {
                failure,
                request_metadata: Some(request_metadata),
            },
            Self::GuardBlocked { reason, .. } => Self::GuardBlocked {
                reason,
                request_metadata: Some(request_metadata),
            },
            Self::ContextBudgetExceeded {
                message,
                param,
                code,
                request_metadata: existing_metadata,
            } => {
                let mut merged = existing_metadata.unwrap_or_default();
                merged.extend(request_metadata);
                Self::ContextBudgetExceeded {
                    message,
                    param,
                    code,
                    request_metadata: Some(merged),
                }
            }
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
                    completed_attempt_records: Vec::new(),
                })),
            },
            Self::UpstreamBody { reason, .. } => Self::UpstreamBody {
                reason,
                observability: Some(Box::new(FailedUpstreamObservability {
                    request_metadata,
                    attempt_record,
                    completed_attempt_records: Vec::new(),
                })),
            },
            error @ (Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::Admission { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::ContextBudgetExceeded { .. }
            | Self::GuardBlocked { .. }) => error,
        }
    }

    fn with_completed_attempt_records(self, completed_attempt_records: Vec<AttemptRecord>) -> Self {
        if completed_attempt_records.is_empty() {
            return self;
        }
        match self {
            Self::UpstreamTransport {
                failure,
                observability: Some(mut observability),
            } => {
                observability
                    .completed_attempt_records
                    .splice(0..0, completed_attempt_records);
                Self::UpstreamTransport {
                    failure,
                    observability: Some(observability),
                }
            }
            Self::UpstreamBody {
                reason,
                observability: Some(mut observability),
            } => {
                observability
                    .completed_attempt_records
                    .splice(0..0, completed_attempt_records);
                Self::UpstreamBody {
                    reason,
                    observability: Some(observability),
                }
            }
            error => error,
        }
    }
}

#[derive(Debug)]
struct FailedUpstreamObservability {
    request_metadata: BTreeMap<String, String>,
    attempt_record: AttemptRecord,
    completed_attempt_records: Vec<AttemptRecord>,
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
