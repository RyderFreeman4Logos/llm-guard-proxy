use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque, hash_map::DefaultHasher},
    convert::Infallible,
    fmt,
    future::{Future, IntoFuture},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    pin::Pin,
    process::Stdio,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, Bytes, HttpBody, to_bytes},
    extract::State,
    http::{
        HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri,
        header::{
            ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST,
            RETRY_AFTER,
        },
    },
    routing::get,
};
use bytes::BytesMut;
use futures_util::{Stream, StreamExt};
#[cfg(feature = "upstream-hot-restart")]
use llm_guard_proxy_core::HotRestartConfig;
#[cfg(feature = "param-override")]
use llm_guard_proxy_core::ParamOverrideConfig;
#[cfg(feature = "guard")]
use llm_guard_proxy_core::{
    AliasTarget, BlockReason, DEFAULT_PROFILE_NAME, GWP_PROTOCOL_VERSION, GuardExecutor,
    GuardOutcome, GuardWorkflowExecutor, GwpDecision, GwpHook, GwpInvocation, GwpResult,
    GwpTraceMode, ModelAliasResolver, ProfileCheckResult, ProfileConfig, UnknownKeyPolicy,
};
use llm_guard_proxy_core::{
    AppConfig, ConfigHandle, DefaultInjectionSchema, DownstreamDropPolicy, Health, HeartbeatMode,
    LICENSE, ListenerConfig, LocalRecoveryConfig, LoopFailurePolicy, LoopGuardConfig,
    MetadataConfig, RestartQueueConfig, RetryConfig, RetryLadderConfig, SERVICE_NAME,
    SelectedUpstreamProfile, ShadowComparisonAttempt, ThinkingConfig, ThinkingMode,
    UpstreamEndpointConfig, UpstreamEndpointProtocol, UpstreamPriority, UpstreamProfileConfig,
    UpstreamRouteReason, UpstreamStallConfig, redact_upstream_base_url, validate_upstream_base_url,
};
use llm_guard_proxy_state::{
    AttemptId, AttemptRecord, AttemptStatus, DebugRequestSummary, DownstreamMode,
    EvidenceAttemptRecord, EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceGroupRecord,
    EvidenceStore, EvidenceStoreWrite, LatencyHistogram, LiveRequestEntry, LiveRequestRegistry,
    LiveRequestState, LiveRequestSummary, ObservabilityMetricsSnapshot, ObservabilityStore,
    RawPayloads, RequestId, RequestRecord, RequestStatus, ShadowSkipReason, TokenUsage,
    UpstreamMode,
};
#[cfg(feature = "guard")]
use llm_guard_proxy_state::{BudgetError, BudgetStore, current_budget_date};
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
#[cfg(feature = "upstream-hot-restart")]
use tokio::task::JoinHandle;
use tokio::task::JoinSet;
use tokio::{
    net::TcpListener,
    process::Command,
    sync::{Mutex as AsyncMutex, Notify, Semaphore, futures::OwnedNotified, oneshot},
    time::{Instant, Interval, MissedTickBehavior, Sleep, timeout},
};

#[cfg(feature = "guard")]
use crate::{workflow_execution::WorkflowExecutionLease, workflow_runtime::WorkflowRuntimeAdapter};

mod buffered_adapter;
mod deepinfra_rerank_adapter;
mod model_metadata;
mod recovery;
mod reranker_protocol;
mod score_adapter;
mod shielded_chat;
mod upstream_failover;

use upstream_failover::{
    EndpointSelectionConstraints, EndpointSelectionError, UpstreamHealthRegistry,
};

use buffered_adapter::{
    BufferedResponseAdapter, adapt_openai_request_if_needed,
    rewrite_buffered_adapter_response_from_upstream, sanitize_transformed_request_headers,
};
use reranker_protocol::{CanonicalRerankerRequest, RenderedEndpointRequest};

#[cfg(all(test, unix))]
use recovery::send_recovery_process_group_signal;
use recovery::{
    RecoveryProcessGuard, configure_recovery_command, recovery_join_timeout,
    recovery_result_poll_interval, terminate_timed_out_recovery_child,
};

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_REPEAT_FINGERPRINT_ENTRIES: usize = 1024;
const HEADER_VALUE_NOT_UTF8: &str = "[non-utf8]";
const HEADER_VALUE_REDACTED: &str = "[redacted]";
const DEBUG_SUMMARY_PATH: &str = "/debug/recent-requests";
const IN_FLIGHT_CAPACITY_RECHECK_INTERVAL: Duration = Duration::from_millis(100);
const ADMISSION_RETRY_AFTER_SECS: u32 = 1;
const COT_SALVAGE_PREFIX_MAX_BYTES: usize = 4_096;
const COT_SALVAGE_THINKING_BUDGET_TOKENS: u32 = 1_024;
const TOKEN_USAGE_BODY_CAP: usize = 64 * 1024;
const MAX_DENIED_MODEL_ID_BYTES: usize = 128;
const PAIRED_SAMPLE_DENOMINATOR: u64 = 1_000_000;
const SERVER_SHUTDOWN_ABORT_REASON: &str = "server_shutdown";
const PROXY_SHUTTING_DOWN_ERROR_TYPE: &str = "proxy_shutting_down";
const REQUEST_DEADLINE_ABORT_REASON: &str = "request_deadline_exhausted";
const REQUEST_DEADLINE_ERROR_TYPE: &str = "llm_guard_request_deadline_exhausted";
const FINAL_DIRECT_RELAY_TERMINATED_ABORT_REASON: &str = "final_direct_relay_terminated";
const MAX_PERSISTENCE_TASKS: usize = 64;
const PERSISTENCE_BACKLOG_DROP_LOG_INTERVAL_MS: u64 = 10_000;
const PERSISTENCE_BACKLOG_DROP_LOG_NEVER_EMITTED: u64 = u64::MAX;
/// Per-profile hard cap for watchdog output samples. Upstream chunk cadence is
/// peer-controlled, so this is deliberately independent of request activity.
const STUCK_WATCHDOG_TOKEN_SAMPLE_CAP: usize = 4_096;
#[cfg(feature = "guard")]
const X_VIRTUAL_KEY_HEADER: &str = "x-virtual-key";

/// Shared HTTP proxy state.
#[derive(Clone, Debug)]
pub(crate) struct ProxyState {
    config: ConfigHandle,
    config_path: PathBuf,
    listener: ListenerConfig,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    #[cfg(feature = "guard")]
    budget_store: Arc<BudgetStore>,
    client: Client,
    generation_requests: Arc<InFlightLimiter>,
    generation_body_routing_requests: Arc<InFlightLimiter>,
    generation_profile_requests: Arc<Mutex<HashMap<String, Arc<InFlightLimiter>>>>,
    control_plane_requests: Arc<InFlightLimiter>,
    #[cfg(feature = "guard")]
    workflow_execution_requests: Arc<InFlightLimiter>,
    upstream_stall_recovery: Arc<UpstreamStallRecoveryCoordinator>,
    upstream_health: Arc<UpstreamHealthRegistry>,
    local_recovery: Arc<LocalRecoveryCoordinatorSet>,
    stuck_watchdog_tokens: Arc<StuckWatchdogTokenTracker>,
    #[cfg(feature = "upstream-hot-restart")]
    hot_restart_recovery: Arc<HotRestartCoordinator>,
    repeat_inputs: Arc<RepeatInputCache>,
    shadow_attempts: Arc<InFlightLimiter>,
    shutdown: Arc<ShutdownGate>,
    malformed_response_counter: Arc<AtomicU64>,
    upstream_failure_counters: Arc<UpstreamFailureCounters>,
    live_registry: Arc<LiveRequestRegistry>,
    persistence_tasks: Arc<PersistenceTasks>,
    #[cfg(test)]
    shielded_heartbeat_ticks: Arc<AtomicU64>,
}

/// Cause-bucketed monotonic counters for upstream failures emitted to
/// `/metrics` as `llm_guard_proxy_upstream_failure_total{cause="..."}`.
#[derive(Debug, Default)]
struct UpstreamFailureCounters {
    connect_failed: AtomicU64,
    timeout: AtomicU64,
    body_error: AtomicU64,
    status_error: AtomicU64,
    transport_error: AtomicU64,
}

impl UpstreamFailureCounters {
    fn increment(&self, cause: UpstreamFailureCause) {
        match cause {
            UpstreamFailureCause::ConnectFailed => &self.connect_failed,
            UpstreamFailureCause::Timeout => &self.timeout,
            UpstreamFailureCause::BodyError => &self.body_error,
            UpstreamFailureCause::StatusError => &self.status_error,
            UpstreamFailureCause::TransportError => &self.transport_error,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> UpstreamFailureSnapshot {
        UpstreamFailureSnapshot {
            connect_failed: self.connect_failed.load(Ordering::Relaxed),
            timeout: self.timeout.load(Ordering::Relaxed),
            body_error: self.body_error.load(Ordering::Relaxed),
            status_error: self.status_error.load(Ordering::Relaxed),
            transport_error: self.transport_error.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UpstreamFailureCause {
    ConnectFailed,
    Timeout,
    BodyError,
    StatusError,
    TransportError,
}

impl UpstreamFailureCause {
    const fn from_reqwest_failure(kind: ReqwestFailureKind) -> Self {
        match kind {
            ReqwestFailureKind::Connect => Self::ConnectFailed,
            ReqwestFailureKind::Timeout => Self::Timeout,
            ReqwestFailureKind::Body | ReqwestFailureKind::Decode => Self::BodyError,
            ReqwestFailureKind::Request | ReqwestFailureKind::Other => Self::TransportError,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::ConnectFailed => "upstream_connect_failed",
            Self::Timeout => "upstream_timeout",
            Self::BodyError => "upstream_body_error",
            Self::StatusError => "upstream_status_error",
            Self::TransportError => "upstream_transport_error",
        }
    }

    const fn metric_label(self) -> &'static str {
        match self {
            Self::ConnectFailed => "connect_failed",
            Self::Timeout => "timeout",
            Self::BodyError => "body_error",
            Self::StatusError => "status_error",
            Self::TransportError => "transport_error",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UpstreamFailureSnapshot {
    connect_failed: u64,
    timeout: u64,
    body_error: u64,
    status_error: u64,
    transport_error: u64,
}

impl ProxyState {
    /// Builds cloneable proxy state for axum handlers.
    #[must_use]
    pub(crate) fn new(
        config: ConfigHandle,
        config_path: PathBuf,
        store: ObservabilityStore,
        evidence_store: EvidenceStore,
        #[cfg(feature = "guard")] budget_store: Arc<BudgetStore>,
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
            #[cfg(feature = "guard")]
            budget_store,
            client,
            generation_requests: Arc::new(InFlightLimiter::default()),
            generation_body_routing_requests: Arc::new(InFlightLimiter::default()),
            generation_profile_requests: Arc::new(Mutex::new(HashMap::new())),
            control_plane_requests: Arc::new(InFlightLimiter::default()),
            #[cfg(feature = "guard")]
            workflow_execution_requests: Arc::new(InFlightLimiter::default()),
            upstream_stall_recovery: Arc::new(UpstreamStallRecoveryCoordinator::default()),
            upstream_health: Arc::new(UpstreamHealthRegistry::default()),
            local_recovery: Arc::new(LocalRecoveryCoordinatorSet::default()),
            stuck_watchdog_tokens: Arc::new(StuckWatchdogTokenTracker::default()),
            #[cfg(feature = "upstream-hot-restart")]
            hot_restart_recovery: Arc::new(HotRestartCoordinator::default()),
            repeat_inputs: Arc::new(RepeatInputCache::default()),
            shadow_attempts: Arc::new(InFlightLimiter::default()),
            shutdown: Arc::new(ShutdownGate::new()),
            malformed_response_counter: Arc::new(AtomicU64::new(0)),
            upstream_failure_counters: Arc::new(UpstreamFailureCounters::default()),
            live_registry: Arc::new(LiveRequestRegistry::new()),
            persistence_tasks: {
                #[cfg(test)]
                {
                    Arc::new(PersistenceTasks::synchronous_for_tests())
                }
                #[cfg(not(test))]
                {
                    Arc::new(PersistenceTasks::default())
                }
            },
            #[cfg(test)]
            shielded_heartbeat_ticks: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns a clone of this shared proxy state scoped to one downstream listener.
    #[must_use]
    pub(crate) fn for_listener(&self, listener: ListenerConfig) -> Self {
        let mut state = self.clone();
        state.listener = listener;
        state
    }

    pub(crate) fn begin_shutdown(&self) {
        self.shutdown.begin_shutdown();
    }

    pub(crate) async fn wait_for_shutdown(&self) {
        let mut shutdown = self.shutdown.subscribe();
        shutdown.cancelled().await;
    }

    pub(crate) async fn flush_persistence(&self) {
        self.persistence_tasks
            .flush(current_shutdown_drain_timeout(&self.config))
            .await;
    }

    #[cfg(test)]
    fn reset_shielded_heartbeat_ticks_for_tests(&self) {
        self.shielded_heartbeat_ticks.store(0, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn shielded_heartbeat_ticks_for_tests(&self) -> u64 {
        self.shielded_heartbeat_ticks.load(Ordering::SeqCst)
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
        if self.shutdown.is_shutting_down() {
            return Err(AdmissionFailure::ShuttingDown { queued: None });
        }
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
        let mut shutdown = self.shutdown.subscribe();
        loop {
            if self.shutdown.is_shutting_down() {
                return Err(AdmissionFailure::ShuttingDown {
                    queued: cancel_recorder.shutdown_cancellation(),
                });
            }
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

            tokio::select! {
                () = limiter.wait_for_capacity(remaining.min(IN_FLIGHT_CAPACITY_RECHECK_INTERVAL)) => {}
                () = shutdown.cancelled() => {
                    return Err(AdmissionFailure::ShuttingDown {
                        queued: cancel_recorder.shutdown_cancellation(),
                    });
                }
            }
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
        if self.shutdown.is_shutting_down() {
            return Err(AdmissionFailure::ShuttingDown { queued: None });
        }
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

    async fn reacquire_generation_permit_for_model(
        &self,
        model_id: Option<&str>,
        record_context: AdmissionRecordContext,
    ) -> Result<GenerationAdmission, AdmissionFailure> {
        let config = self
            .config
            .snapshot()
            .map_err(|error| AdmissionFailure::ConfigSnapshot(error.to_string()))?;
        if self.shutdown.is_shutting_down() {
            return Err(AdmissionFailure::ShuttingDown { queued: None });
        }
        let selected_profile = select_allowed_upstream_profile(&config, &self.listener, model_id)
            .map_err(AdmissionFailure::ListenerUpstreamDenied)?;
        let limiter = self.generation_limiter_for_profile(&selected_profile.profile);
        let max_in_flight_requests = selected_profile
            .profile
            .effective_max_in_flight_requests(&config.server);
        if let Some(permit) = limiter.try_acquire(max_in_flight_requests) {
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
            return Err(AdmissionFailure::GenerationQueueFull {
                max_queued_generation_requests,
                status: config.server.generation_queue_full_status,
                retry_after_secs: config.server.generation_queue_retry_after_secs,
            });
        };
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
        let mut shutdown = self.shutdown.subscribe();
        loop {
            if self.shutdown.is_shutting_down() {
                return Err(AdmissionFailure::ShuttingDown {
                    queued: cancel_recorder.shutdown_cancellation(),
                });
            }
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

            tokio::select! {
                () = limiter.wait_for_capacity(remaining.min(IN_FLIGHT_CAPACITY_RECHECK_INTERVAL)) => {}
                () = shutdown.cancelled() => {
                    return Err(AdmissionFailure::ShuttingDown {
                        queued: cancel_recorder.shutdown_cancellation(),
                    });
                }
            }
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

    async fn acquire_restart_queue_permit(
        &self,
        profile: &UpstreamProfileConfig,
        coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
    ) -> Result<Option<RestartQueuePermit>, ProxyError> {
        let config = self
            .config
            .snapshot()
            .map_err(|error| ProxyError::config_snapshot(error.to_string()))?;
        let max_queued = profile.effective_max_queued_generation_requests(&config.server);
        let limiter = self.generation_limiter_for_profile(profile);
        // Recovery may start or finish while requests are admitted. Hold its
        // coordinator state lock through queue registration so every request
        // that observes a running episode owns a bounded queue permit.
        let recovery = coordinator.state.lock().await;
        if !recovery.running {
            return Ok(None);
        }
        let Some(queued) = limiter.try_enqueue(max_queued) else {
            return Err(ProxyError::upstream_unavailable(
                profile.name.clone(),
                config.server.generation_queue_timeout_ms,
            ));
        };
        coordinator
            .restart_queue_depth
            .fetch_add(1, Ordering::Relaxed);
        drop(recovery);
        Ok(Some(RestartQueuePermit {
            _queued: queued,
            coordinator: Arc::clone(coordinator),
        }))
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

#[derive(Debug)]
struct ShutdownGate {
    closed: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl ShutdownGate {
    fn new() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    fn begin_shutdown(&self) {
        if !self.closed.swap(true, Ordering::SeqCst) {
            self.notify.notify_waiters();
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn subscribe(&self) -> ShutdownSubscription {
        let mut notified = Box::pin(Arc::clone(&self.notify).notified_owned());
        let _already_notified = notified.as_mut().enable();
        ShutdownSubscription {
            closed: Arc::clone(&self.closed),
            notified,
        }
    }
}

struct ShutdownSubscription {
    closed: Arc<AtomicBool>,
    notified: Pin<Box<OwnedNotified>>,
}

impl ShutdownSubscription {
    async fn cancelled(&mut self) {
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        self.notified.as_mut().await;
    }

    fn poll_shutdown(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Poll::Ready(());
        }

        match self.notified.as_mut().poll(cx) {
            Poll::Ready(()) => Poll::Ready(()),
            Poll::Pending => Poll::Pending,
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
    state.upstream_health.start_background_polling(
        state.config.clone(),
        state.client.clone(),
        Arc::clone(&state.shutdown),
    );
    let config = state.config.clone();
    let shutdown_gate = Arc::clone(&state.shutdown);
    let (shutdown_started_tx, shutdown_started_rx) = oneshot::channel();
    let server = axum::serve(listener, router(state)).with_graceful_shutdown(async move {
        shutdown.await;
        shutdown_gate.begin_shutdown();
        let _ignored = shutdown_started_tx.send(());
    });
    let mut server = Box::pin(server.into_future());

    tokio::select! {
        result = server.as_mut() => result,
        _ = shutdown_started_rx => {
            let drain_timeout = current_shutdown_drain_timeout(&config);
            match timeout(drain_timeout, server.as_mut()).await {
                Ok(result) => result,
                Err(_elapsed) => {
                    eprintln!(
                        "llm_guard_proxy_shutdown_drain_timeout timeout_ms={}",
                        drain_timeout.as_millis()
                    );
                    Ok(())
                }
            }
        }
    }
}

fn current_shutdown_drain_timeout(config: &ConfigHandle) -> Duration {
    let timeout_ms = config.snapshot().map_or_else(
        |_error| AppConfig::default().server.shutdown_drain_timeout_ms,
        |snapshot| snapshot.server.shutdown_drain_timeout_ms,
    );
    Duration::from_millis(timeout_ms)
}

#[cfg(test)]
#[derive(Debug)]
struct PersistenceFlushWaitHook {
    arrived: std::sync::mpsc::Sender<()>,
    release: Arc<std::sync::Barrier>,
    notification_registered: Arc<AtomicBool>,
}

#[derive(Debug)]
struct PersistenceBacklogDropLog {
    started_at: Instant,
    last_emitted_at_ms: AtomicU64,
    last_emitted_total: AtomicU64,
}

impl Default for PersistenceBacklogDropLog {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            last_emitted_at_ms: AtomicU64::new(PERSISTENCE_BACKLOG_DROP_LOG_NEVER_EMITTED),
            last_emitted_total: AtomicU64::new(0),
        }
    }
}

impl PersistenceBacklogDropLog {
    fn take_report(&self, dropped_total: u64) -> Option<u64> {
        let elapsed_ms = u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        let interval_ms = PERSISTENCE_BACKLOG_DROP_LOG_INTERVAL_MS;

        loop {
            let last_emitted_at_ms = self.last_emitted_at_ms.load(Ordering::Relaxed);
            if last_emitted_at_ms != PERSISTENCE_BACKLOG_DROP_LOG_NEVER_EMITTED
                && elapsed_ms.saturating_sub(last_emitted_at_ms) < interval_ms
            {
                return None;
            }
            if self
                .last_emitted_at_ms
                .compare_exchange_weak(
                    last_emitted_at_ms,
                    elapsed_ms,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                let previous_total = self
                    .last_emitted_total
                    .swap(dropped_total, Ordering::Relaxed);
                return Some(dropped_total.saturating_sub(previous_total));
            }
        }
    }
}

#[derive(Debug)]
struct PersistenceTasks {
    capacity: Arc<Semaphore>,
    in_flight: AtomicUsize,
    idle: Notify,
    dropped: AtomicU64,
    backlog_drop_log: PersistenceBacklogDropLog,
    #[cfg(test)]
    panics: AtomicUsize,
    #[cfg(test)]
    synchronous_for_tests: bool,
    #[cfg(test)]
    flush_wait_hook: Option<PersistenceFlushWaitHook>,
}

impl Default for PersistenceTasks {
    fn default() -> Self {
        Self {
            capacity: Arc::new(Semaphore::new(MAX_PERSISTENCE_TASKS)),
            in_flight: AtomicUsize::new(0),
            idle: Notify::new(),
            dropped: AtomicU64::new(0),
            backlog_drop_log: PersistenceBacklogDropLog::default(),
            #[cfg(test)]
            panics: AtomicUsize::new(0),
            #[cfg(test)]
            synchronous_for_tests: false,
            #[cfg(test)]
            flush_wait_hook: None,
        }
    }
}

impl PersistenceTasks {
    #[cfg(test)]
    fn synchronous_for_tests() -> Self {
        Self {
            synchronous_for_tests: true,
            ..Self::default()
        }
    }

    #[cfg(test)]
    fn with_capacity_for_tests(capacity: usize) -> Self {
        Self {
            capacity: Arc::new(Semaphore::new(capacity)),
            ..Self::default()
        }
    }

    fn track(self: &Arc<Self>) -> PersistenceTaskGuard {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        PersistenceTaskGuard {
            tasks: Arc::clone(self),
        }
    }

    fn spawn_blocking(self: &Arc<Self>, work: impl FnOnce() + Send + 'static) {
        let Ok(capacity_permit) = Arc::clone(&self.capacity).try_acquire_owned() else {
            // Availability-first bound: refuse unbounded Tokio/blocking growth when SQLite
            // lags. Count every drop so operators can alert; full-record durability under
            // backlog saturation remains a best-effort path, not a silent void.
            let dropped_total = self
                .dropped
                .fetch_add(1, Ordering::Relaxed)
                .saturating_add(1);
            if let Some(dropped_since_last_log) = self.backlog_drop_log.take_report(dropped_total) {
                eprintln!(
                    "persistence backlog full, dropping record dropped_total={dropped_total} dropped_since_last_log={dropped_since_last_log} capacity={MAX_PERSISTENCE_TASKS}"
                );
            }
            return;
        };
        let guard = self.track();
        #[cfg(test)]
        if self.synchronous_for_tests {
            let _capacity_permit = capacity_permit;
            let _guard = guard;
            self.run(work);
            return;
        }

        #[cfg(test)]
        let tasks = Arc::clone(self);
        // The handle is intentionally detached: after its bounded shutdown wait expires,
        // persistence must not retain a waiter on a stalled blocking SQLite operation.
        let _detached_task = tokio::task::spawn_blocking(move || {
            let _capacity_permit = capacity_permit;
            let _guard = guard;
            #[cfg(test)]
            {
                tasks.run(work);
            }
            #[cfg(not(test))]
            Self::run(work);
        });
    }

    fn dropped_total(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn run(&self, work: impl FnOnce()) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err() {
            eprintln!("persistence task panicked");
            self.panics.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[cfg(not(test))]
    fn run(work: impl FnOnce()) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err() {
            eprintln!("persistence task panicked");
        }
    }

    async fn flush(&self, flush_timeout: Duration) {
        let flush = async {
            loop {
                let mut notified = Box::pin(self.idle.notified());
                let _already_notified = notified.as_mut().enable();
                #[cfg(test)]
                if let Some(hook) = &self.flush_wait_hook {
                    hook.notification_registered.store(true, Ordering::SeqCst);
                    hook.arrived
                        .send(())
                        .expect("flush test hook receiver should remain open");
                    hook.release.wait();
                }
                if self.in_flight.load(Ordering::SeqCst) == 0 {
                    return;
                }
                notified.as_mut().await;
            }
        };
        if timeout(flush_timeout, flush).await.is_err() {
            let in_flight = self.in_flight.load(Ordering::SeqCst);
            eprintln!(
                "llm_guard_proxy_persistence_flush_timeout timeout_ms={} in_flight={in_flight}",
                flush_timeout.as_millis()
            );
        }
    }
}

struct PersistenceTaskGuard {
    tasks: Arc<PersistenceTasks>,
}

impl Drop for PersistenceTaskGuard {
    fn drop(&mut self) {
        if self.tasks.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.tasks.idle.notify_waiters();
        }
    }
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

#[cfg(feature = "guard")]
#[derive(Clone, Debug, Error, Eq, PartialEq)]
enum WorkflowExecutionTaskError {
    #[error(
        "workflow execution capacity exhausted (max_in_flight_executions={max_in_flight_executions})"
    )]
    AtCapacity { max_in_flight_executions: usize },
    #[error("workflow worker failed: {0}")]
    WorkerFailed(String),
}

#[cfg(feature = "guard")]
async fn run_workflow_execution<T, Execute>(
    limiter: Arc<InFlightLimiter>,
    max_in_flight_executions: usize,
    execute: Execute,
) -> Result<T, WorkflowExecutionTaskError>
where
    T: Send + 'static,
    Execute: FnOnce(WorkflowExecutionLease) -> T + Send + 'static,
{
    let permit = limiter.try_acquire(max_in_flight_executions).ok_or(
        WorkflowExecutionTaskError::AtCapacity {
            max_in_flight_executions,
        },
    )?;
    tokio::task::spawn_blocking(move || execute(WorkflowExecutionLease::new(permit)))
        .await
        .map_err(|error| WorkflowExecutionTaskError::WorkerFailed(error.to_string()))
}

#[cfg(feature = "guard")]
fn guard_outcome_after_workflow_task(
    result: Result<GuardOutcome, WorkflowExecutionTaskError>,
    fail_closed_blocks: bool,
) -> GuardOutcome {
    result.unwrap_or_else(|error| {
        if fail_closed_blocks {
            GuardOutcome::Block {
                reason: error.to_string(),
            }
        } else {
            GuardOutcome::Allow
        }
    })
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
    persistence_tasks: Arc<PersistenceTasks>,
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

    fn shutdown_cancellation(&mut self) -> Option<QueuedAdmissionCancellation> {
        self.record.take().map(|record| {
            QueuedAdmissionCancellation::from_record(
                &record,
                QueueCancellationReason::ServerShutdown,
            )
        })
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueuedAdmissionCancellation {
    reason: QueueCancellationReason,
    queue_wait_ms: u64,
    generation_queue_timeout_ms: u64,
}

impl QueuedAdmissionCancellation {
    fn from_record(record: &QueuedAdmissionCancelRecord, reason: QueueCancellationReason) -> Self {
        Self {
            reason,
            queue_wait_ms: duration_millis_u64(record.queued_at.elapsed()),
            generation_queue_timeout_ms: record.timeout_ms,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueueCancellationReason {
    DownstreamDisconnected,
    ServerShutdown,
}

impl QueueCancellationReason {
    const fn admission_outcome(self) -> &'static str {
        match self {
            Self::DownstreamDisconnected => "queue_cancelled",
            Self::ServerShutdown => "queue_cancelled_shutdown",
        }
    }

    const fn abort_reason(self) -> &'static str {
        match self {
            Self::DownstreamDisconnected => "downstream_disconnected_while_queued",
            Self::ServerShutdown => "server_shutdown_while_queued",
        }
    }
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

#[derive(Debug)]
struct RestartQueuePermit {
    _queued: QueuedAdmissionPermit,
    coordinator: Arc<UpstreamStallRecoveryCoordinator>,
}

impl Drop for RestartQueuePermit {
    fn drop(&mut self) {
        self.coordinator
            .restart_queue_depth
            .fetch_sub(1, Ordering::Relaxed);
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
    #[error("proxy is shutting down")]
    ShuttingDown {
        queued: Option<QueuedAdmissionCancellation>,
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
            Self::GenerationQueueTimeout { .. }
            | Self::ControlPlaneLimitExceeded { .. }
            | Self::ShuttingDown { .. } => StatusCode::SERVICE_UNAVAILABLE,
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
            Self::ShuttingDown {
                queued: Some(_), ..
            } => "proxy_generation_queue_cancelled",
            Self::ShuttingDown { queued: None } => "proxy_shutting_down",
        }
    }

    const fn request_status(&self) -> RequestStatus {
        match self {
            Self::ShuttingDown {
                queued: Some(_), ..
            } => RequestStatus::Aborted,
            Self::ConfigSnapshot(_)
            | Self::GenerationQueueFull { .. }
            | Self::GenerationQueueTimeout { .. }
            | Self::ControlPlaneLimitExceeded { .. }
            | Self::ShuttingDown { queued: None }
            | Self::ListenerUpstreamDenied(_) => RequestStatus::Failed,
        }
    }

    const fn abort_reason(&self) -> Option<&'static str> {
        match self {
            Self::ShuttingDown {
                queued: Some(queued),
            } => Some(queued.reason.abort_reason()),
            Self::ConfigSnapshot(_)
            | Self::GenerationQueueFull { .. }
            | Self::GenerationQueueTimeout { .. }
            | Self::ControlPlaneLimitExceeded { .. }
            | Self::ShuttingDown { queued: None }
            | Self::ListenerUpstreamDenied(_) => None,
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
            Self::GenerationQueueTimeout { .. }
            | Self::ControlPlaneLimitExceeded { .. }
            | Self::ShuttingDown { .. } => Some(ADMISSION_RETRY_AFTER_SECS.to_string()),
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
            Self::ShuttingDown {
                queued: Some(queued),
            } => BTreeMap::from([
                (
                    String::from("admission_outcome"),
                    queued.reason.admission_outcome().to_owned(),
                ),
                (
                    String::from("queue_wait_ms"),
                    queued.queue_wait_ms.to_string(),
                ),
                (
                    String::from("generation_queue_timeout_ms"),
                    queued.generation_queue_timeout_ms.to_string(),
                ),
            ]),
            Self::ShuttingDown { queued: None } => {
                BTreeMap::from([(String::from("admission_outcome"), String::from("shutdown"))])
            }
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
        .route("/debug/requests", get(debug_live_requests_handler))
        .route("/debug/requests/{id}", get(debug_live_request_handler))
        .fallback(proxy_handler)
        .with_state(state)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HealthUpstreamStatus {
    Disabled,
    Ready,
    Degraded,
    Unavailable,
}

impl HealthUpstreamStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "not_checked",
            Self::Ready => "ready",
            Self::Degraded => "degraded",
            Self::Unavailable => "unavailable",
        }
    }

    const fn http_status(self) -> StatusCode {
        match self {
            Self::Disabled | Self::Ready | Self::Degraded => StatusCode::OK,
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
    // Prune leaked live-registry entries older than 5 minutes as a safety net.
    state.live_registry.prune_stale(5 * 60 * 1000);
    match state.config.snapshot() {
        Ok(config) if config.observability.metrics_enabled.is_enabled() => {
            match state.store.metrics_snapshot() {
                Ok(snapshot) => {
                    let admission = state.admission_metrics_snapshot();
                    let malformed_responses =
                        state.malformed_response_counter.load(Ordering::Relaxed);
                    let persistence_dropped = state.persistence_tasks.dropped_total();
                    let upstream_failures = state.upstream_failure_counters.snapshot();
                    let watchdog = state.local_recovery.watchdog_metrics_snapshot();
                    text_response(
                        StatusCode::OK,
                        render_metrics(
                            &snapshot,
                            &admission,
                            malformed_responses,
                            persistence_dropped,
                            upstream_failures,
                            watchdog,
                        ),
                    )
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
    let models_ok = matches!(
        tokio::time::timeout(timeout, state.client.get(url).send()).await,
        Ok(Ok(response)) if response.status().is_success()
            || response.status().as_u16() == StatusCode::UNAUTHORIZED.as_u16()
    );
    if !models_ok {
        return HealthUpstreamStatus::Unavailable;
    }
    if config.observability.health_chat_probe_enabled.is_enabled()
        && !probe_upstream_chat_completion(state, config).await
    {
        return HealthUpstreamStatus::Degraded;
    }
    HealthUpstreamStatus::Ready
}

/// Sends a minimal chat completion request to the upstream to verify that the
/// generation path is healthy. Returns `true` when the upstream responds with a
/// success status, `false` on any transport failure or non-success status.
async fn probe_upstream_chat_completion(state: &ProxyState, config: &AppConfig) -> bool {
    let uri = Uri::from_static("/v1/chat/completions");
    let Ok(url) = build_upstream_url(&config.upstream.base_url, &uri) else {
        return false;
    };
    // Use a minimal "ping" payload. The probe only needs to verify that the
    // upstream chat-completion transport path responds; the upstream may reject
    // an unknown model with a 4xx, which still indicates the endpoint is alive.
    let body = serde_json::json!({
        "model": "ping",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": false,
    });
    let capped_timeout_ms = config
        .observability
        .health_chat_probe_timeout_ms
        .min(10_000);
    let timeout = Duration::from_millis(capped_timeout_ms);
    let request = state
        .client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .build();
    let Ok(request) = request else {
        return false;
    };
    match tokio::time::timeout(timeout, state.client.execute(request)).await {
        Ok(Ok(response)) => response.status().is_success(),
        _ => false,
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

/// `GET /debug/requests` — lists active (in-flight) requests as JSON.
///
/// Gated by `observability.debug_summary_enabled` and `debug_summary_admin_token`,
/// exactly like `/debug/recent-requests`. Accepts an optional `?state=active`
/// query param (no-op filter today; all live entries are active by definition).
/// Output is metadata-only: no raw prompts or responses.
async fn debug_live_requests_handler(
    State(state): State<ProxyState>,
    request: Request<Body>,
) -> Response<Body> {
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
            StatusCode::FORBIDDEN,
            "debug_summary_disabled",
            "live requests endpoint is disabled",
        );
    }
    if !debug_summary_authorized(
        request.headers(),
        config.observability.debug_summary_admin_token.as_deref(),
    ) {
        return proxy_error_response(
            StatusCode::UNAUTHORIZED,
            "debug_summary_unauthorized",
            "live requests authorization failed",
        );
    }
    // Safety-net prune so leaked entries don't accumulate between metrics scrapes.
    state.live_registry.prune_stale(5 * 60 * 1000);
    let summaries = state.live_registry.list_active();
    json_response(
        StatusCode::OK,
        render_live_requests_json(&summaries).to_string(),
    )
}

/// `GET /debug/requests/{id}` — returns detail for a single live request,
/// including its full stage timeline. Returns 404 if the request is unknown or
/// already completed (live entries are removed on completion).
async fn debug_live_request_handler(
    State(state): State<ProxyState>,
    request: Request<Body>,
) -> Response<Body> {
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
            StatusCode::FORBIDDEN,
            "debug_summary_disabled",
            "live requests endpoint is disabled",
        );
    }
    if !debug_summary_authorized(
        request.headers(),
        config.observability.debug_summary_admin_token.as_deref(),
    ) {
        return proxy_error_response(
            StatusCode::UNAUTHORIZED,
            "debug_summary_unauthorized",
            "live requests authorization failed",
        );
    }
    // Extract the request id from the last path segment.
    let request_id = request
        .uri()
        .path()
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_owned();
    if request_id.is_empty() {
        return proxy_error_response(
            StatusCode::NOT_FOUND,
            "live_request_not_found",
            "request id not provided",
        );
    }
    match state.live_registry.get(&request_id) {
        Some(entry) => json_response(
            StatusCode::OK,
            render_live_request_detail_json(&entry).to_string(),
        ),
        None => proxy_error_response(
            StatusCode::NOT_FOUND,
            "live_request_not_found",
            "no active request with that id (it may have completed or been pruned)",
        ),
    }
}

fn render_live_requests_json(summaries: &[LiveRequestSummary]) -> serde_json::Value {
    let requests: Vec<serde_json::Value> = summaries.iter().map(render_live_summary_json).collect();
    json!({
        "redaction": "metadata-only; no raw prompts or responses are exposed",
        "request_count": summaries.len(),
        "requests": requests,
    })
}

fn render_live_summary_json(summary: &LiveRequestSummary) -> serde_json::Value {
    json!({
        "request_id": summary.request_id,
        "state": summary.state,
        "model": summary.model,
        "profile": summary.profile,
        "downstream_mode": summary.downstream_mode,
        "elapsed_ms": summary.elapsed_ms,
        "queue_wait_ms": summary.queue_wait_ms,
        "first_token_latency_ms": summary.first_token_latency_ms,
        "active_ladder_rung": summary.active_ladder_rung,
        "active_attempt_index": summary.active_attempt_index,
        "chunks_downstream": summary.chunks_downstream,
        "bytes_downstream": summary.bytes_downstream,
        "last_progress_at_ms": summary.last_progress_at_ms,
    })
}

fn render_live_request_detail_json(entry: &LiveRequestEntry) -> serde_json::Value {
    let timeline: Vec<serde_json::Value> = entry
        .timeline
        .iter()
        .map(|event| json!({ "at_ms": event.at_ms, "event": event.event }))
        .collect();
    json!({
        "request_id": entry.request_id,
        "listener": entry.listener,
        "profile": entry.profile,
        "model": entry.model,
        "upstream_target": entry.upstream_target,
        "downstream_mode": entry.downstream_mode,
        "state": entry.state.as_str(),
        "created_at_ms": entry.created_at_ms,
        "last_updated_at_ms": entry.last_updated_at_ms,
        "queue_wait_ms": entry.queue_wait_ms,
        "upstream_elapsed_ms": entry.upstream_elapsed_ms,
        "first_token_latency_ms": entry.first_token_latency_ms,
        "active_ladder_rung": entry.active_ladder_rung,
        "active_attempt_index": entry.active_attempt_index,
        "chunks_downstream": entry.chunks_downstream,
        "bytes_downstream": entry.bytes_downstream,
        "chunks_upstream": entry.chunks_upstream,
        "bytes_upstream": entry.bytes_upstream,
        "last_progress_at_ms": entry.last_progress_at_ms,
        "timeline": timeline,
        "redaction": "metadata-only; no raw prompts or responses are exposed",
    })
}

fn render_metrics(
    snapshot: &ObservabilityMetricsSnapshot,
    admission: &AdmissionMetricsSnapshot,
    malformed_responses: u64,
    persistence_dropped: u64,
    upstream_failures: UpstreamFailureSnapshot,
    watchdog: WatchdogMetricsSnapshot,
) -> String {
    let mut output = String::new();
    push_admission_metrics(&mut output, admission);
    push_watchdog_metrics(&mut output, watchdog);
    push_request_metrics(&mut output, snapshot);
    push_request_terminal_metrics(&mut output, snapshot);
    push_attempt_metrics(&mut output, snapshot);
    push_retry_and_error_metrics(&mut output, snapshot);
    push_malformed_response_metrics(&mut output, malformed_responses);
    push_persistence_drop_metrics(&mut output, persistence_dropped);
    push_upstream_failure_metrics(&mut output, upstream_failures);
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

fn push_watchdog_metrics(output: &mut String, watchdog: WatchdogMetricsSnapshot) {
    for (name, help, metric_type, value) in [
        (
            "llm_guard_proxy_stuck_watchdog_detections_total",
            "Stuck-engine watchdog detections.",
            "counter",
            watchdog.detections,
        ),
        (
            "llm_guard_proxy_stuck_watchdog_restarts_total",
            "Stuck-engine watchdog recovery restarts.",
            "counter",
            watchdog.restarts,
        ),
        (
            "llm_guard_proxy_stuck_watchdog_recovery_successes_total",
            "Successful stuck-engine watchdog recoveries.",
            "counter",
            watchdog.recovery_successes,
        ),
        (
            "llm_guard_proxy_stuck_watchdog_recovery_timeouts_total",
            "Timed-out stuck-engine watchdog recoveries.",
            "counter",
            watchdog.recovery_timeouts,
        ),
        (
            "llm_guard_proxy_restart_queue_depth",
            "Requests currently waiting for a recovery restart.",
            "gauge",
            watchdog.restart_queue_depth,
        ),
        (
            "llm_guard_proxy_stuck_watchdog_restart_queue_depth",
            "Requests currently waiting for a watchdog restart.",
            "gauge",
            watchdog.restart_queue_depth,
        ),
    ] {
        push_metric_header(output, name, help, metric_type);
        push_metric_line(output, name, &[], value);
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

fn push_request_terminal_metrics(output: &mut String, snapshot: &ObservabilityMetricsSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_current_retained_request_terminals",
        "Currently retained proxy request rows by bounded terminal reason.",
        "gauge",
    );
    for row in &snapshot.request_terminal_counts {
        push_metric_line(
            output,
            "llm_guard_proxy_current_retained_request_terminals",
            &[
                ("status", &row.status),
                ("terminal_reason", &row.terminal_reason),
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

fn push_malformed_response_metrics(output: &mut String, malformed_responses: u64) {
    push_metric_header(
        output,
        "llm_guard_proxy_malformed_response_total",
        "Non-stream chat completion responses converted to 502 due to a missing or invalid choices field.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_malformed_response_total",
        &[("kind", "missing_choices")],
        malformed_responses,
    );
}

fn push_persistence_drop_metrics(output: &mut String, persistence_dropped: u64) {
    push_metric_header(
        output,
        "llm_guard_proxy_persistence_dropped_total",
        "Observability/evidence persistence closures dropped because the bounded backlog was full.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_persistence_dropped_total",
        &[],
        persistence_dropped,
    );
}

fn push_upstream_failure_metrics(output: &mut String, snapshot: UpstreamFailureSnapshot) {
    push_metric_header(
        output,
        "llm_guard_proxy_upstream_failure_total",
        "Shielded and forwarded upstream failures classified by bounded cause bucket.",
        "counter",
    );
    push_metric_line(
        output,
        "llm_guard_proxy_upstream_failure_total",
        &[("cause", UpstreamFailureCause::ConnectFailed.metric_label())],
        snapshot.connect_failed,
    );
    push_metric_line(
        output,
        "llm_guard_proxy_upstream_failure_total",
        &[("cause", UpstreamFailureCause::Timeout.metric_label())],
        snapshot.timeout,
    );
    push_metric_line(
        output,
        "llm_guard_proxy_upstream_failure_total",
        &[("cause", UpstreamFailureCause::BodyError.metric_label())],
        snapshot.body_error,
    );
    push_metric_line(
        output,
        "llm_guard_proxy_upstream_failure_total",
        &[("cause", UpstreamFailureCause::StatusError.metric_label())],
        snapshot.status_error,
    );
    push_metric_line(
        output,
        "llm_guard_proxy_upstream_failure_total",
        &[("cause", UpstreamFailureCause::TransportError.metric_label())],
        snapshot.transport_error,
    );
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

fn proxy_handler(
    State(state): State<ProxyState>,
    request: Request<Body>,
) -> Pin<Box<dyn Future<Output = Response<Body>> + Send>> {
    Box::pin(proxy_handler_inner(state, request))
}

#[allow(clippy::too_many_lines)]
async fn proxy_handler_inner(state: ProxyState, request: Request<Body>) -> Response<Body> {
    if request.method() == Method::GET && is_configured_debug_summary_request(&state, request.uri())
    {
        return debug_summary_response(&state, &request);
    }

    let request_id = RequestId::generate();
    let started_at_unix_ms = unix_time_millis();
    // Register the request in the live observability registry. We use
    // "streaming" as the default mode; the forwarder will refine it if the
    // request turns out to be non-streaming JSON.
    state
        .live_registry
        .register(request_id.as_str(), "streaming");
    state
        .live_registry
        .update_target(request_id.as_str(), Some(state.listener.name.clone()), None);
    let live_request_id = request_id.clone();
    let live_registry = Arc::clone(&state.live_registry);
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
            &state.persistence_tasks,
            &state.store,
            FailedRequestRecord {
                request_id: request_id.clone(),
                started_at_unix_ms,
                finished_at_unix_ms,
                status: RequestStatus::Failed,
                http_status: error.status().as_u16(),
                error_type,
                error_reason,
                abort_reason: None,
                request_metadata,
                attempts: Vec::new(),
            },
        );
        live_registry.update_state(live_request_id.as_str(), LiveRequestState::Failed);
        live_registry.fail(live_request_id.as_str());
        return finalize_proxy_terminal_response(response, &request_id);
    }

    let admission_request = AdmissionRequestMetadata::from_request(&request);
    let admission =
        match admit_request(&state, &request_id, started_at_unix_ms, admission_request).await {
            AdmissionOutcome::Accepted(admission) => {
                state
                    .live_registry
                    .update_state(request_id.as_str(), LiveRequestState::Admitted);
                *admission
            }
            AdmissionOutcome::Rejected(response) => {
                state
                    .live_registry
                    .update_state(request_id.as_str(), LiveRequestState::Failed);
                state.live_registry.fail(request_id.as_str());
                return finalize_proxy_terminal_response(response, &request_id);
            }
        };

    let forward_result = Box::pin(forward_openai_request(
        &state,
        &request_id,
        started_at_unix_ms,
        request,
        admission.permit,
        admission.permit_kind,
        admission.admission_metadata,
        admission.config.server.max_request_body_bytes,
    ))
    .await;
    let response = match forward_result {
        Ok(response) => {
            state
                .live_registry
                .update_state(request_id.as_str(), LiveRequestState::Completed);
            state.live_registry.complete(request_id.as_str());
            response
        }
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_type = error.error_type();
            let error_reason = error.to_string();
            if let Some(cause) = error.upstream_failure_cause() {
                state.upstream_failure_counters.increment(cause);
            }
            let response =
                proxy_error_response_from_error_with_diagnostics(&error, Some(&request_id));
            let request_metadata = error.request_metadata().cloned().unwrap_or_else(|| {
                BTreeMap::from([(String::from("proxy_error"), error_type.to_owned())])
            });
            record_failed_request(
                &state.persistence_tasks,
                &state.store,
                FailedRequestRecord {
                    request_id: request_id.clone(),
                    started_at_unix_ms,
                    finished_at_unix_ms,
                    status: error.request_status(),
                    http_status: error.status().as_u16(),
                    error_type,
                    error_reason,
                    abort_reason: error.abort_reason(),
                    request_metadata,
                    attempts: error.attempt_records(),
                },
            );
            state
                .live_registry
                .update_state(live_request_id.as_str(), LiveRequestState::Failed);
            live_registry.fail(live_request_id.as_str());
            response
        }
    };
    finalize_proxy_terminal_response(response, &request_id)
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
                &state.persistence_tasks,
                &state.store,
                FailedRequestRecord {
                    request_id: request_id.clone(),
                    started_at_unix_ms,
                    finished_at_unix_ms: unix_time_millis(),
                    status: RequestStatus::Failed,
                    http_status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                    error_type,
                    error_reason,
                    abort_reason: None,
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
        &state.persistence_tasks,
        &state.store,
        FailedRequestRecord {
            request_id: request_id.clone(),
            started_at_unix_ms,
            finished_at_unix_ms: unix_time_millis(),
            status: error.request_status(),
            http_status: error.status().as_u16(),
            error_type,
            error_reason,
            abort_reason: error.abort_reason(),
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
        persistence_tasks: Arc::clone(&state.persistence_tasks),
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
    #[allow(unused_mut)]
    let mut prepared_request = prepare_openai_forward_request(
        state,
        &config,
        &method,
        &uri,
        &downstream_headers,
        &body,
        &mut request_metadata,
    )
    .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    #[cfg(feature = "guard")]
    if let Some(workflow_alias) = prepared_request.workflow_alias.clone() {
        return handle_workflow_alias_request(WorkflowAliasRequestContext {
            state,
            config: &config,
            request_id,
            started_at_unix_ms,
            requested_model: prepared_request.model_id.as_deref().unwrap_or_default(),
            request_body: prepared_request.shielded_chat_plan.downstream_body.clone(),
            caller_profile_name: prepared_request.caller_profile_name.clone(),
            caller_profile: prepared_request.caller_profile.clone(),
            workflow_alias,
            request_metadata,
            in_flight_permit,
        })
        .await;
    }
    #[cfg(feature = "guard")]
    apply_pre_request_guard(
        &config,
        request_id,
        &method,
        &uri,
        &mut prepared_request,
        Arc::clone(&state.workflow_execution_requests),
    )
    .await
    .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
    let stuck_watchdog_request = prepared_request
        .upstream_profile
        .stuck_watchdog
        .enabled
        .then(|| {
            state.stuck_watchdog_tokens.begin_request(
                &prepared_request.upstream_profile.name,
                watchdog_progress_unit(&prepared_request.forward_uri),
                Duration::from_secs(
                    prepared_request
                        .upstream_profile
                        .stuck_watchdog
                        .detection_window_secs,
                ),
            )
        });
    let endpoint_retry_body = prepared_request.shielded_chat_plan.upstream_body.clone();
    let mut _initial_recovery_trial_lease = None;
    if prepared_request.upstream_profile.has_endpoint_failover() {
        let upstream_deadline = Instant::now()
            + Duration::from_millis(prepared_request.upstream_profile.request_timeout_ms);
        let selected = state
            .upstream_health
            .select_endpoint(
                &state.client,
                &prepared_request.upstream_profile,
                &state.shutdown,
                prepared_request.canonical_reranker.as_ref(),
                Some(&downstream_headers),
                Some(upstream_deadline),
            )
            .await
            .map_err(|error| match error {
                EndpointSelectionError::Shutdown => ProxyError::server_shutdown(),
                EndpointSelectionError::Incompatible { profile } => {
                    ProxyError::ContextBudgetExceeded {
                        message: format!(
                            "no configured endpoint can losslessly represent this request for profile {profile}"
                        ),
                        param: "body",
                        code: "unsupported_reranker_endpoint_request",
                        request_metadata: None,
                        attempts: Vec::new(),
                    }
                }
                EndpointSelectionError::Unavailable { profile, waited_ms } => {
                    ProxyError::upstream_unavailable(profile, waited_ms)
                }
            })
            .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
        prepared_request.upstream_url =
            build_upstream_url(&selected.base_url, &prepared_request.forward_uri)
                .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
        prepared_request
            .upstream_profile
            .base_url
            .clone_from(&selected.base_url);
        prepared_request.terminal_endpoint_protocol = selected.endpoint.protocol;
        prepared_request.terminal_endpoint = selected.endpoint.clone();
        prepared_request
            .endpoint_retry_order
            .clone_from(&selected.selection_order);
        prepared_request.upstream_deadline = Some(upstream_deadline);
        let rendered = match prepared_request.canonical_reranker.as_ref() {
            Some(canonical) => {
                reranker_protocol::render(&selected.endpoint, canonical, &downstream_headers)
            }
            None if selected.endpoint.protocol == UpstreamEndpointProtocol::OpenAi => {
                reranker_protocol::render_openai_endpoint(
                    &selected.endpoint,
                    prepared_request.forward_uri.clone(),
                    &endpoint_retry_body,
                    &downstream_headers,
                    prepared_request.transformed_request_headers,
                )
            }
            None => Err(ProxyError::ContextBudgetExceeded {
                message: String::from(
                    "selected DeepInfra endpoint only supports normalized reranker requests",
                ),
                param: "path",
                code: "unsupported_reranker_endpoint_request",
                request_metadata: None,
                attempts: Vec::new(),
            }),
        }
        .map_err(|error| error.with_request_metadata(request_metadata.clone()))?;
        prepared_request.forward_uri = rendered.uri;
        prepared_request.upstream_url = rendered.url;
        prepared_request.shielded_chat_plan.upstream_body = rendered.body;
        prepared_request.upstream_headers = Some(rendered.headers);
        request_metadata.insert(
            String::from("upstream_endpoint_priority"),
            match selected.priority {
                llm_guard_proxy_core::UpstreamPriority::Primary => String::from("primary"),
                llm_guard_proxy_core::UpstreamPriority::Failover => String::from("failover"),
            },
        );
        request_metadata.insert(
            String::from("upstream_failover_selected"),
            String::from("false"),
        );
        request_metadata.insert(
            String::from("upstream_endpoint_base_url"),
            redact_upstream_base_url(&selected.base_url),
        );
        request_metadata.insert(
            String::from("upstream_endpoint_protocol"),
            selected.endpoint.protocol.as_str().to_owned(),
        );
        request_metadata.insert(
            String::from("upstream_endpoint_selection"),
            prepared_request
                .upstream_profile
                .endpoint_selection
                .as_str()
                .to_owned(),
        );
        _initial_recovery_trial_lease = selected.recovery_trial_lease;
    }
    let retry_policy = ShieldedRetryPolicy::from_config(&config.retry, &config.loop_guard);
    let upstream_stall_policy = UpstreamStallPolicy::from_config(&config.upstream_stall);
    let upstream_timeout =
        Duration::from_millis(prepared_request.upstream_profile.request_timeout_ms);
    if prepared_request.shielded_chat_plan.intercepted {
        add_retry_request_metadata(&mut request_metadata, &retry_policy);
        let request_deadline = ShieldedRequestDeadline::new(retry_policy.request_deadline);
        let local_recovery_policy =
            LocalRecoveryPolicy::from_config(&prepared_request.upstream_profile.local_recovery);
        let local_recovery = state
            .local_recovery
            .coordinator_for(&prepared_request.upstream_profile.name);
        return Box::pin(forward_shielded_chat_with_retries(
            ShieldedRetryRuntime {
                client: state.client.clone(),
                method: prepared_request.reqwest_method,
                upstream_url: prepared_request.upstream_url,
                downstream_method: method,
                downstream_uri: uri,
                upstream_headers: prepared_request
                    .upstream_headers
                    .unwrap_or_else(|| downstream_headers.clone()),
                original_downstream_headers: downstream_headers,
                upstream_body: prepared_request.shielded_chat_plan.upstream_body,
                downstream_body: prepared_request.shielded_chat_plan.downstream_body,
                forward_uri: prepared_request.forward_uri,
                transformed_request_headers: prepared_request.transformed_request_headers,
                terminal_endpoint: prepared_request.terminal_endpoint,
                terminal_endpoint_protocol: prepared_request.terminal_endpoint_protocol,
                endpoint_retry_order: prepared_request.endpoint_retry_order,
                chat_kind: prepared_request.shielded_chat_plan.kind,
                upstream_timeout,
                config: state.config.clone(),
                store: state.store.clone(),
                evidence_store: state.evidence_store.clone(),
                persistence_tasks: Arc::clone(&state.persistence_tasks),
                request_id: request_id.clone(),
                started_at_unix_ms,
                model_id: prepared_request.model_id,
                stuck_watchdog_request,
                request_metadata,
                listener: state.listener.clone(),
                upstream_profile: prepared_request.upstream_profile,
                #[cfg(feature = "guard")]
                caller_profile_name: prepared_request.caller_profile_name,
                #[cfg(feature = "guard")]
                caller_profile: prepared_request.caller_profile,
                #[cfg(feature = "guard")]
                workflow_execution_requests: Arc::clone(&state.workflow_execution_requests),
                route_reason: prepared_request.route_reason,
                liveness: prepared_request.shielded_chat_plan.liveness,
                thinking_metadata: prepared_request.shielded_chat_plan.thinking_metadata,
                loop_context: prepared_request.shielded_chat_plan.loop_context,
                retry_policy,
                request_deadline,
                upstream_stall_policy,
                upstream_stall_recovery: state.upstream_stall_recovery.clone(),
                upstream_health: state.upstream_health.clone(),
                local_recovery_policy,
                local_recovery,
                local_recovery_attempts: Arc::new(AtomicU64::new(0)),
                local_recovery_deadline_replay_permits: Arc::new(AtomicU64::new(0)),
                #[cfg(feature = "upstream-hot-restart")]
                hot_restart_recovery: state.hot_restart_recovery.clone(),
                shadow_attempts: state.shadow_attempts.clone(),
                shutdown: Arc::clone(&state.shutdown),
                downstream_drop_signal: DownstreamDropSignal::default(),
                shadow_evidence: ShadowEvidenceState::default(),
                malformed_response_counter: state.malformed_response_counter.clone(),
                upstream_failure_counters: state.upstream_failure_counters.clone(),
                #[cfg(test)]
                shielded_heartbeat_ticks: state.shielded_heartbeat_ticks.clone(),
            },
            in_flight_permit,
        ))
        .await;
    }
    forward_generic_openai_request(GenericForwardContext {
        state,
        config: &config,
        method,
        uri,
        downstream_headers,
        reqwest_method: prepared_request.reqwest_method,
        upstream_uri: prepared_request.forward_uri,
        upstream_url: prepared_request.upstream_url,
        upstream_body: prepared_request.shielded_chat_plan.upstream_body,
        endpoint_retry_body,
        upstream_timeout,
        upstream_profile: prepared_request.upstream_profile,
        route_reason: prepared_request.route_reason,
        liveness: prepared_request.shielded_chat_plan.liveness,
        thinking_policy_applied: prepared_request.shielded_chat_plan.thinking_policy_applied,
        thinking_metadata: prepared_request.shielded_chat_plan.thinking_metadata,
        request_id,
        started_at_unix_ms,
        model_id: prepared_request.model_id,
        stuck_watchdog_request,
        request_metadata,
        in_flight_permit,
        response_adapter: prepared_request.response_adapter,
        canonical_reranker: prepared_request.canonical_reranker,
        transformed_request_headers: prepared_request.transformed_request_headers,
        upstream_headers: prepared_request.upstream_headers,
        terminal_endpoint_protocol: prepared_request.terminal_endpoint_protocol,
        terminal_endpoint: prepared_request.terminal_endpoint,
        upstream_deadline: prepared_request.upstream_deadline,
        endpoint_retry_order: prepared_request.endpoint_retry_order,
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
    let (body, body_read_request_metadata) =
        read_body_for_generation_admission(state, body, max_request_body_bytes, &request).await?;
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

    admit_generation_after_body(
        state,
        body,
        config,
        in_flight_permit,
        admission_permit_kind,
        body_read_request_metadata,
        request,
    )
    .await
}

async fn read_body_for_generation_admission(
    state: &ProxyState,
    body: Body,
    max_request_body_bytes: usize,
    request: &BodyAdmissionContext<'_>,
) -> Result<(Bytes, BTreeMap<String, String>), ProxyError> {
    let mut pre_body_request_metadata = pre_upstream_request_metadata(
        request.method,
        request.uri,
        request.downstream_headers,
        request.shielding_enabled_hint,
    );
    add_listener_metadata(&mut pre_body_request_metadata, &state.listener);
    let body = read_body_with_adapter_limit(
        body,
        max_request_body_bytes,
        request.method,
        request.uri,
        state.shutdown.subscribe(),
        pre_body_request_metadata,
    )
    .await?;
    let mut body_read_request_metadata = base_request_metadata(
        request.method,
        request.uri,
        request.downstream_headers,
        body.len().to_string(),
        request.shielding_enabled_hint,
    );
    add_listener_metadata(&mut body_read_request_metadata, &state.listener);
    Ok((body, body_read_request_metadata))
}

async fn admit_generation_after_body(
    state: &ProxyState,
    body: Bytes,
    config: AppConfig,
    in_flight_permit: InFlightPermit,
    admission_permit_kind: AdmissionPermitKind,
    body_read_request_metadata: BTreeMap<String, String>,
    request: BodyAdmissionContext<'_>,
) -> Result<OpenAiBodyAdmission, ProxyError> {
    let model_id_for_admission = extract_model_id(request.method, request.uri, &body);
    let selected_profile = select_allowed_upstream_profile(
        &config,
        &state.listener,
        model_id_for_admission.as_deref(),
    )
    .map_err(|error| {
        ProxyError::listener_denied(error).with_request_metadata(body_read_request_metadata.clone())
    })?;
    if selected_profile.profile.restart_queue.enabled {
        return wait_for_restart_queue_and_readmit(
            state,
            body,
            in_flight_permit,
            model_id_for_admission,
            selected_profile.profile,
            body_read_request_metadata,
            request,
        )
        .await;
    }
    if admission_permit_kind == AdmissionPermitKind::Generation {
        return Ok(OpenAiBodyAdmission {
            config,
            body,
            in_flight_permit,
            admission_metadata: request.admission_metadata,
        });
    }

    let record_context =
        body_admission_record_context(state, &request, &body_read_request_metadata);
    let admission = state
        .acquire_generation_permit_for_model(
            model_id_for_admission.as_deref(),
            in_flight_permit,
            record_context,
        )
        .await
        .map_err(|error| admission_proxy_error(error, body_read_request_metadata))?;
    let mut admission_metadata = request.admission_metadata;
    admission_metadata.extend(acquired_admission_metadata(admission.queue_wait));
    Ok(OpenAiBodyAdmission {
        config: admission.config,
        body,
        in_flight_permit: admission.permit,
        admission_metadata,
    })
}

async fn wait_for_restart_queue_and_readmit(
    state: &ProxyState,
    body: Bytes,
    in_flight_permit: InFlightPermit,
    model_id: Option<String>,
    profile: UpstreamProfileConfig,
    mut body_read_request_metadata: BTreeMap<String, String>,
    request: BodyAdmissionContext<'_>,
) -> Result<OpenAiBodyAdmission, ProxyError> {
    // Both provisional admission paths use capacity only to route/read the body.
    // A restart waiter must release that capacity before entering the dedicated queue.
    drop(in_flight_permit);
    wait_for_profile_restart_queue(state, &profile, &mut body_read_request_metadata).await?;
    let record_context =
        body_admission_record_context(state, &request, &body_read_request_metadata);
    let admission = state
        .reacquire_generation_permit_for_model(model_id.as_deref(), record_context)
        .await
        .map_err(|error| admission_proxy_error(error, body_read_request_metadata))?;
    let mut admission_metadata = request.admission_metadata;
    admission_metadata.extend(acquired_admission_metadata(admission.queue_wait));
    Ok(OpenAiBodyAdmission {
        config: admission.config,
        body,
        in_flight_permit: admission.permit,
        admission_metadata,
    })
}

fn body_admission_record_context(
    state: &ProxyState,
    request: &BodyAdmissionContext<'_>,
    request_metadata: &BTreeMap<String, String>,
) -> AdmissionRecordContext {
    AdmissionRecordContext {
        store: state.store.clone(),
        persistence_tasks: Arc::clone(&state.persistence_tasks),
        request_id: request.request_id.clone(),
        started_at_unix_ms: request.started_at_unix_ms,
        request_metadata: request_metadata.clone(),
    }
}

fn admission_proxy_error(
    error: AdmissionFailure,
    mut request_metadata: BTreeMap<String, String>,
) -> ProxyError {
    request_metadata.extend(error.request_metadata());
    ProxyError::admission(error).with_request_metadata(request_metadata)
}

async fn read_body_with_adapter_limit(
    body: Body,
    max_request_body_bytes: usize,
    method: &Method,
    uri: &Uri,
    shutdown: ShutdownSubscription,
    request_metadata: BTreeMap<String, String>,
) -> Result<Bytes, ProxyError> {
    let is_score_request = score_adapter::is_score_request(method, uri);
    let adapter_label = if is_score_request {
        Some("score")
    } else if deepinfra_rerank_adapter::is_request(method, uri) {
        Some("DeepInfra rerank")
    } else {
        None
    };
    let adapter_limit_applies =
        adapter_label.filter(|_| max_request_body_bytes >= score_adapter::MAX_SCORE_BODY_BYTES);
    let body_limit = if adapter_label.is_some() {
        max_request_body_bytes.min(score_adapter::MAX_SCORE_BODY_BYTES)
    } else {
        max_request_body_bytes
    };
    read_body_bytes_until_shutdown(body, body_limit, shutdown)
        .await
        .map_err(|error| {
            let adapter_body_limit_exceeded = matches!(
                &error,
                ProxyError::RequestBody { reason, .. } if reason.contains("length limit exceeded")
            );
            let error = if let Some(adapter_label) = adapter_limit_applies
                && adapter_body_limit_exceeded
            {
                ProxyError::request_body(format!(
                    "{adapter_label} request exceeded adapter limit of {} bytes",
                    score_adapter::MAX_SCORE_BODY_BYTES
                ))
            } else {
                error
            };
            error.with_request_metadata(request_metadata)
        })
}

struct PreparedOpenAiRequest {
    model_id: Option<String>,
    #[cfg(feature = "guard")]
    caller_profile_name: String,
    #[cfg(feature = "guard")]
    caller_profile: ProfileConfig,
    #[cfg(feature = "guard")]
    workflow_alias: Option<ResolvedWorkflowAlias>,
    upstream_profile: UpstreamProfileConfig,
    route_reason: UpstreamRouteReason,
    forward_uri: Uri,
    upstream_url: Url,
    reqwest_method: reqwest::Method,
    shielded_chat_plan: ShieldedChatPlan,
    response_adapter: Option<BufferedResponseAdapter>,
    canonical_reranker: Option<CanonicalRerankerRequest>,
    transformed_request_headers: bool,
    upstream_headers: Option<HeaderMap>,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
    terminal_endpoint: UpstreamEndpointConfig,
    upstream_deadline: Option<Instant>,
    endpoint_retry_order: Vec<String>,
}

#[cfg(feature = "guard")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedWorkflowAlias {
    workflow_id: String,
    timeout_ms: u64,
}

#[cfg(feature = "guard")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedCallerProfile {
    name: String,
    config: ProfileConfig,
    resolution: CallerProfileResolution,
    enforce_policy: bool,
}

#[cfg(feature = "guard")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallerProfileResolution {
    VirtualKeyMatched,
    DisabledDefault,
    SingleProfileDefault,
    UnknownUseDefault,
}

fn prepare_openai_forward_request(
    state: &ProxyState,
    config: &AppConfig,
    method: &Method,
    uri: &Uri,
    downstream_headers: &HeaderMap,
    body: &Bytes,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<PreparedOpenAiRequest, ProxyError> {
    #[cfg(feature = "guard")]
    let caller_profile = resolve_caller_profile(config, downstream_headers)?;
    #[cfg(feature = "guard")]
    add_caller_profile_metadata(request_metadata, &caller_profile);
    let model_id = extract_model_id(method, uri, body);
    #[cfg(feature = "guard")]
    enforce_caller_profile_policy(&caller_profile, model_id.as_deref())?;
    #[cfg(feature = "guard")]
    enforce_caller_profile_budget(state, config, &caller_profile)?;
    #[cfg(feature = "guard")]
    let workflow_alias = workflow_alias_for_model(config, model_id.as_deref())?;
    let adapted_request =
        adapt_openai_request_if_needed(method, uri, downstream_headers, body, request_metadata)?;
    let selected_profile =
        select_profile_for_request(config, &state.listener, method, uri, model_id.as_deref())?;
    let upstream_profile = selected_profile.profile;
    let route_reason = selected_profile.route_reason;
    add_upstream_profile_metadata(request_metadata, &upstream_profile, route_reason);
    let canonical_reranker = reranker_protocol::capture_request(
        method,
        uri,
        body,
        &adapted_request.forward_uri,
        &adapted_request.adapted_body,
    );
    let transformed_request_headers = adapted_request.response_adapter.is_some();
    let response_adapter = if upstream_profile
        .endpoints
        .iter()
        .any(|endpoint| endpoint.protocol == UpstreamEndpointProtocol::DeepInfraQwen3Rerank)
        && let Some(canonical) = canonical_reranker.as_ref()
    {
        Some(BufferedResponseAdapter::HeterogeneousReranker {
            request: canonical.clone(),
            terminal_protocol: UpstreamEndpointProtocol::OpenAi,
        })
    } else {
        adapted_request.response_adapter
    };
    let forward_uri = adapted_request.forward_uri;
    let adapted_body = adapted_request.adapted_body;
    let upstream_url = build_upstream_url(&upstream_profile.base_url, &forward_uri)?;
    let reqwest_method = upstream_method(method)?;
    let body = adapted_body;
    validate_vllm_native_request_controls(config, &upstream_profile, method, &forward_uri, &body)?;
    let shielded_chat_plan = plan_shielded_chat(
        state,
        config,
        &upstream_profile,
        method,
        &forward_uri,
        &body,
    );
    #[cfg(feature = "param-override")]
    let shielded_chat_plan = {
        let mut plan = shielded_chat_plan;
        apply_param_override_to_shielded_plan(method, &forward_uri, &mut plan, &upstream_profile)?;
        plan
    };
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
        &body,
        &shielded_chat_plan.upstream_body,
        &upstream_profile,
    )?);

    let (terminal_endpoint, endpoint_retry_order) = initial_endpoint_retry_state(&upstream_profile);
    Ok(PreparedOpenAiRequest {
        model_id,
        #[cfg(feature = "guard")]
        caller_profile_name: caller_profile.name.clone(),
        #[cfg(feature = "guard")]
        caller_profile: caller_profile.config,
        #[cfg(feature = "guard")]
        workflow_alias,
        upstream_profile,
        route_reason,
        forward_uri,
        upstream_url,
        reqwest_method,
        shielded_chat_plan,
        response_adapter,
        canonical_reranker,
        transformed_request_headers,
        upstream_headers: None,
        terminal_endpoint_protocol: UpstreamEndpointProtocol::OpenAi,
        terminal_endpoint,
        upstream_deadline: None,
        endpoint_retry_order,
    })
}

fn initial_endpoint_retry_state(
    upstream_profile: &UpstreamProfileConfig,
) -> (UpstreamEndpointConfig, Vec<String>) {
    (
        UpstreamEndpointConfig {
            base_url: upstream_profile.base_url.clone(),
            priority: UpstreamPriority::Primary,
            ..UpstreamEndpointConfig::default()
        },
        vec![upstream_profile.base_url.clone()],
    )
}

#[cfg(feature = "guard")]
fn enforce_caller_profile_budget(
    state: &ProxyState,
    config: &AppConfig,
    caller_profile: &ResolvedCallerProfile,
) -> Result<(), ProxyError> {
    let limit = caller_profile.config.daily_request_limit;
    if !config.budget.enabled || limit == 0 {
        return Ok(());
    }
    let date = current_budget_date(config.budget.reset_hour_utc);
    let check = state
        .budget_store
        .check_and_increment(&caller_profile.name, &date, limit)
        .map_err(|error| {
            ProxyError::budget_store_failed(&error).with_request_metadata(BTreeMap::from([
                (String::from("caller_profile"), caller_profile.name.clone()),
                (String::from("budget_date"), date.clone()),
                (String::from("budget_limit"), limit.to_string()),
            ]))
        })?;
    if check.allowed {
        return Ok(());
    }
    Err(ProxyError::budget_exceeded(
        caller_profile.name.clone(),
        date,
        check.current_count,
        check.limit,
    ))
}

#[cfg(feature = "guard")]
fn resolve_caller_profile(
    config: &AppConfig,
    headers: &HeaderMap,
) -> Result<ResolvedCallerProfile, ProxyError> {
    if !config.virtual_keys.enabled {
        return Ok(default_caller_profile(
            config,
            CallerProfileResolution::DisabledDefault,
        ));
    }

    let virtual_key = extract_virtual_key(headers);
    if let Some(virtual_key) = virtual_key
        && let Some(profile_name) = virtual_key_profile_name(config, virtual_key)
        && let Some(profile) = config.caller_profile_by_name(&profile_name)
    {
        return Ok(ResolvedCallerProfile {
            name: profile_name,
            config: profile,
            resolution: CallerProfileResolution::VirtualKeyMatched,
            enforce_policy: !config.profiles.is_empty(),
        });
    }

    if virtual_key.is_none()
        && config.profiles.len() == 1
        && let Some((profile_name, profile)) = config.profiles.iter().next()
    {
        return Ok(ResolvedCallerProfile {
            name: profile_name.clone(),
            config: profile.clone(),
            resolution: CallerProfileResolution::SingleProfileDefault,
            enforce_policy: true,
        });
    }

    match config.virtual_keys.unknown_key_policy {
        UnknownKeyPolicy::UseDefaultProfile => Ok(default_caller_profile(
            config,
            CallerProfileResolution::UnknownUseDefault,
        )),
        UnknownKeyPolicy::FailClosed => Err(ProxyError::virtual_key_unauthorized()),
    }
}

#[cfg(feature = "guard")]
fn default_caller_profile(
    config: &AppConfig,
    resolution: CallerProfileResolution,
) -> ResolvedCallerProfile {
    let config_profile = config.caller_profile_by_name(DEFAULT_PROFILE_NAME);
    ResolvedCallerProfile {
        name: DEFAULT_PROFILE_NAME.to_owned(),
        config: config_profile.unwrap_or_else(|| config.default_caller_profile()),
        resolution,
        enforce_policy: !config.profiles.is_empty(),
    }
}

#[cfg(feature = "guard")]
fn extract_virtual_key(headers: &HeaderMap) -> Option<&str> {
    header_trimmed(headers, &HeaderName::from_static(X_VIRTUAL_KEY_HEADER))
        .or_else(|| bearer_token(headers))
}

#[cfg(feature = "guard")]
fn header_trimmed<'headers>(
    headers: &'headers HeaderMap,
    name: &HeaderName,
) -> Option<&'headers str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

#[cfg(feature = "guard")]
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = header_trimmed(headers, &AUTHORIZATION)?;
    value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

#[cfg(feature = "guard")]
fn virtual_key_profile_name(config: &AppConfig, virtual_key: &str) -> Option<String> {
    let mut selected = None;
    for (configured_key, profile_name) in &config.virtual_keys.keys {
        if constant_time_eq(configured_key.as_bytes(), virtual_key.as_bytes()) {
            selected = Some(profile_name.clone());
        }
    }
    selected
}

#[cfg(feature = "guard")]
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_len = left.len().max(right.len());
    let mut diff = left.len() ^ right.len();
    for index in 0..max_len {
        let left_byte = left.get(index).copied().unwrap_or_default();
        let right_byte = right.get(index).copied().unwrap_or_default();
        diff |= usize::from(left_byte ^ right_byte);
    }
    diff == 0
}

#[cfg(feature = "guard")]
fn add_caller_profile_metadata(
    request_metadata: &mut BTreeMap<String, String>,
    caller_profile: &ResolvedCallerProfile,
) {
    request_metadata.insert(String::from("caller_profile"), caller_profile.name.clone());
    request_metadata.insert(
        String::from("virtual_key_resolution"),
        caller_profile.resolution.as_str().to_owned(),
    );
}

#[cfg(feature = "guard")]
impl CallerProfileResolution {
    const fn as_str(self) -> &'static str {
        match self {
            Self::VirtualKeyMatched => "matched",
            Self::DisabledDefault => "disabled_default",
            Self::SingleProfileDefault => "single_profile_default",
            Self::UnknownUseDefault => "unknown_use_default",
        }
    }
}

#[cfg(feature = "guard")]
fn enforce_caller_profile_policy(
    caller_profile: &ResolvedCallerProfile,
    model_id: Option<&str>,
) -> Result<(), ProxyError> {
    if !caller_profile.enforce_policy {
        return Ok(());
    }
    let Some(model_id) = normalized_model_id(model_id) else {
        return Ok(());
    };
    match caller_profile.config.check_request(model_id, 0) {
        ProfileCheckResult::Allow => Ok(()),
        ProfileCheckResult::Block { reason } => Err(ProxyError::guard_blocked(format!(
            "caller profile blocked request: {}",
            profile_block_reason_message(&reason)
        ))),
    }
}

#[cfg(feature = "guard")]
fn profile_block_reason_message(reason: &BlockReason) -> String {
    match reason {
        BlockReason::ModelNotAllowed { model } => {
            format!(
                "model not allowed: {}",
                denied_model_id_summary(Some(model))
            )
        }
        BlockReason::DailyLimitExceeded { limit } => {
            format!("daily request limit exceeded: limit={limit}")
        }
        BlockReason::KindMismatch => String::from("profile kind mismatch"),
    }
}

fn is_chat_completions_request(method: &Method, uri: &Uri) -> bool {
    method == Method::POST && uri.path() == "/v1/chat/completions"
}

fn validate_vllm_native_request_controls(
    config: &AppConfig,
    profile: &UpstreamProfileConfig,
    method: &Method,
    uri: &Uri,
    body: &Bytes,
) -> Result<(), ProxyError> {
    let vllm_native_configured = profile.thinking.default_injection_schema
        == DefaultInjectionSchema::VllmNative
        || config.retry.ladder.iter().any(|entry| {
            entry
                .default_injection_schema
                .unwrap_or(profile.thinking.default_injection_schema)
                == DefaultInjectionSchema::VllmNative
        });
    if !is_chat_completions_request(method, uri)
        || !vllm_native_configured
        || !shielded_chat::has_conflicting_vllm_native_controls(body)
    {
        return Ok(());
    }

    Err(ProxyError::ContextBudgetExceeded {
        message: String::from(
            "positive thinking_token_budget cannot be combined with an explicit no-thinking marker",
        ),
        param: "thinking_token_budget",
        code: "conflicting_thinking_controls",
        request_metadata: Some(BTreeMap::from([
            (
                String::from("thinking_control_validation"),
                String::from("rejected_conflict"),
            ),
            (
                String::from("thinking_default_injection_schema"),
                String::from("vllm_native"),
            ),
        ])),
        attempts: Vec::new(),
    })
}

#[cfg(feature = "param-override")]
fn param_override_applies(method: &Method, uri: &Uri, profile: &UpstreamProfileConfig) -> bool {
    is_chat_completions_request(method, uri)
        && profile.param_override.enabled
        && param_override_has_fields(&profile.param_override)
}

#[cfg(feature = "param-override")]
fn apply_param_override_to_shielded_plan(
    method: &Method,
    uri: &Uri,
    plan: &mut ShieldedChatPlan,
    profile: &UpstreamProfileConfig,
) -> Result<(), ProxyError> {
    if !param_override_applies(method, uri, profile) {
        return Ok(());
    }
    let (upstream_body, cap_decision) = apply_param_override_to_body(&plan.upstream_body, profile)?;
    plan.upstream_body = upstream_body;
    shielded_chat::merge_final_answer_budget_metadata(
        &plan.upstream_body,
        &mut plan.thinking_metadata,
        cap_decision,
    );
    Ok(())
}

#[cfg(feature = "param-override")]
fn apply_param_override_to_body(
    body: &Bytes,
    profile: &UpstreamProfileConfig,
) -> Result<(Bytes, shielded_chat::AnswerBudgetDecision), ProxyError> {
    if !profile.param_override.enabled || !param_override_has_fields(&profile.param_override) {
        return Ok((body.clone(), shielded_chat::AnswerBudgetDecision::default()));
    }
    let mut value = serde_json::from_slice::<serde_json::Value>(body)
        .map_err(|error| ProxyError::request_body(format!("request body is not JSON: {error}")))?;
    let serde_json::Value::Object(object) = &mut value else {
        return Ok((body.clone(), shielded_chat::AnswerBudgetDecision::default()));
    };
    let cap_decision = apply_param_override_object(object, &profile.param_override);
    let rewritten = serde_json::to_vec(&value).map_err(|error| {
        ProxyError::request_body(format!("request body rewrite failed: {error}"))
    })?;
    Ok((Bytes::from(rewritten), cap_decision))
}

#[cfg(feature = "param-override")]
fn param_override_has_fields(config: &ParamOverrideConfig) -> bool {
    config.temperature.is_some()
        || config.top_p.is_some()
        || config.top_k.is_some()
        || config.max_tokens.is_some()
        || config.frequency_penalty.is_some()
        || config.presence_penalty.is_some()
}

#[cfg(feature = "param-override")]
fn apply_param_override_object(
    object: &mut serde_json::Map<String, serde_json::Value>,
    config: &ParamOverrideConfig,
) -> shielded_chat::AnswerBudgetDecision {
    insert_param_override_fields(object, config);
    if let Some(serde_json::Value::Object(parameters)) = object.get_mut("parameters") {
        insert_param_override_fields(parameters, config);
    }
    config
        .max_tokens
        .map_or_else(shielded_chat::AnswerBudgetDecision::default, |max_tokens| {
            shielded_chat::apply_output_token_cap(object, u64::from(max_tokens))
        })
}

#[cfg(feature = "param-override")]
fn insert_param_override_fields(
    object: &mut serde_json::Map<String, serde_json::Value>,
    config: &ParamOverrideConfig,
) {
    insert_f64_override(object, "temperature", config.temperature);
    insert_f64_override(object, "top_p", config.top_p);
    insert_u32_override(object, "top_k", config.top_k);
    insert_f64_override(object, "frequency_penalty", config.frequency_penalty);
    insert_f64_override(object, "presence_penalty", config.presence_penalty);
}

#[cfg(feature = "param-override")]
fn insert_f64_override(
    object: &mut serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    value: Option<f64>,
) {
    if let Some(number) = value.and_then(serde_json::Number::from_f64) {
        object.insert(field.to_owned(), serde_json::Value::Number(number));
    }
}

#[cfg(feature = "param-override")]
fn insert_u32_override(
    object: &mut serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    value: Option<u32>,
) {
    if let Some(value) = value {
        object.insert(field.to_owned(), serde_json::Value::Number(value.into()));
    }
}

#[cfg(feature = "guard")]
struct WorkflowAliasRequestContext<'request> {
    state: &'request ProxyState,
    config: &'request AppConfig,
    request_id: &'request RequestId,
    started_at_unix_ms: u64,
    requested_model: &'request str,
    request_body: Bytes,
    caller_profile_name: String,
    caller_profile: ProfileConfig,
    workflow_alias: ResolvedWorkflowAlias,
    request_metadata: BTreeMap<String, String>,
    in_flight_permit: InFlightPermit,
}

#[cfg(feature = "guard")]
async fn handle_workflow_alias_request(
    context: WorkflowAliasRequestContext<'_>,
) -> Result<Response<Body>, ProxyError> {
    let messages = chat_messages_from_body(&context.request_body).unwrap_or_default();
    let invocation = GwpInvocation {
        protocol_version: GWP_PROTOCOL_VERSION.to_owned(),
        hook: GwpHook::PreRequestGuard,
        request_id: context.request_id.to_string(),
        profile: context
            .caller_profile
            .to_gwp_profile(&context.caller_profile_name),
        model_alias: context.requested_model.to_owned(),
        messages,
        policy: serde_json::Value::Null,
        budgets: json!({
            "timeout_ms": context.workflow_alias.timeout_ms
        }),
        trace_mode: GwpTraceMode::Redacted,
    };
    let workflow_id = context.workflow_alias.workflow_id.clone();
    let workflow_config = workflow_config_for_alias(context.config, &context.workflow_alias)?;
    let execution_workflow_id = workflow_id.clone();
    let runtime_workflow_id = workflow_id.clone();
    let result = run_workflow_execution(
        Arc::clone(&context.state.workflow_execution_requests),
        context.config.guard_workflows.max_in_flight_executions,
        move |execution_lease| {
            WorkflowRuntimeAdapter::new(
                HashMap::from([(runtime_workflow_id, workflow_config)]),
                &execution_lease,
            )
            .execute(&execution_workflow_id, &invocation)
        },
    )
    .await
    .map_err(|error| ProxyError::guard_blocked(error.to_string()))?
    .ok_or_else(|| {
        ProxyError::guard_blocked(format!(
            "workflow alias references unconfigured workflow {workflow_id:?}"
        ))
    })?;

    let workflow_output = workflow_alias_content_from_result(&workflow_id, result)?;
    let body = workflow_alias_chat_completion_body(
        context.request_id,
        context.requested_model,
        &workflow_output,
    );
    record_workflow_alias_success(
        context.state,
        context.request_id,
        context.started_at_unix_ms,
        context.requested_model,
        &context.workflow_alias,
        context.request_metadata,
        body.len(),
    );
    drop(context.in_flight_permit);
    Ok(json_response(StatusCode::OK, body))
}

#[cfg(feature = "guard")]
fn workflow_config_for_alias(
    config: &AppConfig,
    workflow_alias: &ResolvedWorkflowAlias,
) -> Result<llm_guard_proxy_core::WorkflowConfig, ProxyError> {
    let mut workflow_config = config
        .workflows
        .get(&workflow_alias.workflow_id)
        .cloned()
        .ok_or_else(|| {
            ProxyError::guard_blocked(format!(
                "workflow alias references unconfigured workflow {:?}",
                workflow_alias.workflow_id
            ))
        })?;
    workflow_config.timeout_ms = workflow_alias.timeout_ms;
    Ok(workflow_config)
}

#[cfg(feature = "guard")]
fn workflow_alias_content_from_result(
    workflow_id: &str,
    result: GwpResult,
) -> Result<String, ProxyError> {
    match result.decision {
        GwpDecision::Replace => workflow_replacement_content(result.replacement_messages)
            .ok_or_else(|| {
                ProxyError::guard_blocked(format!(
                    "workflow alias {workflow_id:?} returned replace without assistant content"
                ))
            }),
        GwpDecision::Block => Err(ProxyError::guard_blocked(result.summary)),
        GwpDecision::Allow | GwpDecision::DeferToParent => Err(ProxyError::guard_blocked(format!(
            "workflow alias {workflow_id:?} did not return replacement content: {}",
            result.summary
        ))),
        GwpDecision::ErrorFailClosed => Err(ProxyError::guard_blocked(format!(
            "workflow alias {workflow_id:?} failed closed: {}",
            result.summary
        ))),
    }
}

#[cfg(feature = "guard")]
fn workflow_replacement_content(messages: Option<Vec<serde_json::Value>>) -> Option<String> {
    messages?.into_iter().find_map(|message| match message {
        serde_json::Value::String(content) if !content.is_empty() => Some(content),
        serde_json::Value::Object(object) => object
            .get("content")
            .and_then(serde_json::Value::as_str)
            .filter(|content| !content.is_empty())
            .map(str::to_owned),
        _ => None,
    })
}

#[cfg(feature = "guard")]
fn workflow_alias_chat_completion_body(
    request_id: &RequestId,
    requested_model: &str,
    workflow_output: &str,
) -> String {
    json!({
        "id": format!("chatcmpl-{request_id}"),
        "object": "chat.completion",
        "created": unix_time_secs(),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": workflow_output
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })
    .to_string()
}

#[cfg(feature = "guard")]
fn record_workflow_alias_success(
    state: &ProxyState,
    request_id: &RequestId,
    started_at_unix_ms: u64,
    requested_model: &str,
    workflow_alias: &ResolvedWorkflowAlias,
    mut request_metadata: BTreeMap<String, String>,
    body_len: usize,
) {
    let finished_at_unix_ms = unix_time_millis();
    request_metadata.insert(String::from("workflow_alias"), String::from("true"));
    request_metadata.insert(
        String::from("workflow_id"),
        workflow_alias.workflow_id.clone(),
    );
    request_metadata.insert(
        String::from("workflow_timeout_ms"),
        workflow_alias.timeout_ms.to_string(),
    );
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let request_record = RequestRecord {
        request_id: request_id.clone(),
        started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at_unix_ms),
        downstream_mode: DownstreamMode::NonStreamJson,
        upstream_mode: UpstreamMode::NotApplicable,
        model_id: Some(requested_model.to_owned()),
        input_fingerprint: None,
        status: RequestStatus::Succeeded,
        http_status: Some(StatusCode::OK.as_u16()),
        error_reason: None,
        abort_reason: None,
        request_metadata,
        response_metadata: response_metadata(
            StatusCode::OK,
            &headers,
            usize_to_u64(body_len),
            finished_at_unix_ms.saturating_sub(started_at_unix_ms),
        ),
        raw_payloads: RawPayloads::default(),
    };
    let store = state.store.clone();
    state
        .persistence_tasks
        .spawn_blocking(move || record_observability_many(&store, &request_record, &[]));
}

#[cfg(feature = "guard")]
fn workflow_alias_for_model(
    config: &AppConfig,
    model: Option<&str>,
) -> Result<Option<ResolvedWorkflowAlias>, ProxyError> {
    let Some(model) = normalized_model_id(model) else {
        return Ok(None);
    };
    let resolver = ModelAliasResolver::new(config.model_aliases.clone());
    if !resolver.is_alias(model) {
        return Ok(None);
    }
    match resolver.resolve(model) {
        Ok(AliasTarget::Workflow {
            workflow_id,
            timeout_ms,
        }) => Ok(Some(ResolvedWorkflowAlias {
            workflow_id,
            timeout_ms,
        })),
        Ok(AliasTarget::Upstream { .. }) => Ok(None),
        Err(error) => Err(ProxyError::guard_blocked(error.to_string())),
    }
}

#[cfg(feature = "guard")]
async fn apply_pre_request_guard(
    config: &AppConfig,
    request_id: &RequestId,
    method: &Method,
    uri: &Uri,
    prepared_request: &mut PreparedOpenAiRequest,
    workflow_execution_requests: Arc<InFlightLimiter>,
) -> Result<(), ProxyError> {
    if !is_chat_completions_request(method, uri) || config.guard_workflows.pre_request.is_none() {
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
        prepared_request.caller_profile_name.clone(),
        prepared_request.caller_profile.clone(),
        workflow_execution_requests,
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

#[cfg(feature = "guard")]
async fn run_pre_request_guard(
    config: &AppConfig,
    request_id: &str,
    model: &str,
    messages: Vec<serde_json::Value>,
    profile_name: String,
    profile: ProfileConfig,
    workflow_execution_requests: Arc<InFlightLimiter>,
) -> GuardOutcome {
    let guard_config = config.guard_workflows.clone();
    let workflows = config.workflows.clone();
    let request_id = request_id.to_owned();
    let model = model.to_owned();
    let fail_closed_blocks = guard_config.fail_closed_blocks;
    let max_in_flight_executions = guard_config.max_in_flight_executions;
    let result = run_workflow_execution(
        workflow_execution_requests,
        max_in_flight_executions,
        move |execution_lease| {
            let workflow_executor = WorkflowRuntimeAdapter::new(workflows, &execution_lease);
            GuardExecutor::new(guard_config, &workflow_executor).pre_request_guard(
                &request_id,
                &model,
                &messages,
                &profile_name,
                &profile,
            )
        },
    )
    .await;
    guard_outcome_after_workflow_task(result, fail_closed_blocks)
}

#[cfg(feature = "guard")]
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
        runtime.caller_profile_name.clone(),
        runtime.caller_profile.clone(),
        Arc::clone(&runtime.workflow_execution_requests),
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

#[cfg(feature = "guard")]
async fn run_post_response_guard(
    config: &AppConfig,
    request_id: &str,
    model: &str,
    response: serde_json::Value,
    profile_name: String,
    profile: ProfileConfig,
    workflow_execution_requests: Arc<InFlightLimiter>,
) -> GuardOutcome {
    let guard_config = config.guard_workflows.clone();
    let workflows = config.workflows.clone();
    let request_id = request_id.to_owned();
    let model = model.to_owned();
    let fail_closed_blocks = guard_config.fail_closed_blocks;
    let max_in_flight_executions = guard_config.max_in_flight_executions;
    let result = run_workflow_execution(
        workflow_execution_requests,
        max_in_flight_executions,
        move |execution_lease| {
            let workflow_executor = WorkflowRuntimeAdapter::new(workflows, &execution_lease);
            GuardExecutor::new(guard_config, &workflow_executor).post_response_guard(
                &request_id,
                &model,
                &response,
                &profile_name,
                &profile,
            )
        },
    )
    .await;
    guard_outcome_after_workflow_task(result, fail_closed_blocks)
}

#[cfg(feature = "guard")]
fn chat_messages_from_body(body: &Bytes) -> Option<Vec<serde_json::Value>> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("messages")?
        .as_array()
        .cloned()
}

#[cfg(feature = "guard")]
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

#[cfg(feature = "guard")]
fn replace_aggregated_response_body(aggregated: &mut ShieldedAggregatedAttempt, body: &Bytes) {
    aggregated.body = body.clone();
    aggregated.sse_body = openai_data_sse_body(body);
    aggregated.response_metadata.insert(
        String::from("guard_post_response_replaced"),
        String::from("true"),
    );
}

#[cfg(feature = "guard")]
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

#[cfg(feature = "guard")]
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
    #[cfg(feature = "guard")]
    if let Some(selected) = select_profile_from_model_alias(config, listener, model)? {
        return Ok(selected);
    }
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

#[cfg(feature = "guard")]
fn select_profile_from_model_alias(
    config: &AppConfig,
    listener: &ListenerConfig,
    model: Option<&str>,
) -> Result<Option<SelectedUpstreamProfile>, ListenerUpstreamDenied> {
    let Some(model) = normalized_model_id(model) else {
        return Ok(None);
    };
    let resolver = ModelAliasResolver::new(config.model_aliases.clone());
    if !resolver.is_alias(model) {
        return Ok(None);
    }
    let target = match resolver.resolve(model) {
        Ok(target) => target,
        Err(_error) => return Ok(None),
    };
    match target {
        AliasTarget::Upstream { profile_name } => {
            let Some(profile) = config.upstream_profile_by_name(&profile_name) else {
                return Ok(None);
            };
            if listener.allows_upstream(&profile.name) {
                return Ok(Some(SelectedUpstreamProfile {
                    profile,
                    route_reason: UpstreamRouteReason::MatchedModel,
                }));
            }
            Err(ListenerUpstreamDenied {
                listener_name: listener.name.clone(),
                listener_port: listener.port,
                upstream_profile: profile.name,
                model_id: denied_model_id_summary(Some(model)),
            })
        }
        AliasTarget::Workflow { .. } => Ok(Some(SelectedUpstreamProfile {
            profile: config.default_upstream_profile(),
            route_reason: UpstreamRouteReason::MatchedModel,
        })),
    }
}

#[cfg(feature = "guard")]
fn normalized_model_id(model: Option<&str>) -> Option<&str> {
    model.map(str::trim).filter(|model| !model.is_empty())
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
    upstream_uri: Uri,
    upstream_url: Url,
    upstream_body: Bytes,
    endpoint_retry_body: Bytes,
    upstream_timeout: Duration,
    upstream_profile: UpstreamProfileConfig,
    route_reason: UpstreamRouteReason,
    liveness: ShieldedLivenessSelection,
    thinking_policy_applied: bool,
    thinking_metadata: BTreeMap<String, String>,
    request_id: &'request RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    stuck_watchdog_request: Option<StuckWatchdogRequest>,
    request_metadata: BTreeMap<String, String>,
    in_flight_permit: InFlightPermit,
    response_adapter: Option<BufferedResponseAdapter>,
    canonical_reranker: Option<CanonicalRerankerRequest>,
    transformed_request_headers: bool,
    upstream_headers: Option<HeaderMap>,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
    terminal_endpoint: UpstreamEndpointConfig,
    upstream_deadline: Option<Instant>,
    endpoint_retry_order: Vec<String>,
}

fn prepare_generic_attempt_request(
    context: &GenericForwardContext<'_>,
) -> (Option<HeaderMap>, BTreeMap<String, String>) {
    let override_headers = context.upstream_headers.clone().or_else(|| {
        context
            .transformed_request_headers
            .then(|| sanitize_transformed_request_headers(&context.downstream_headers))
    });
    let headers = override_headers
        .as_ref()
        .unwrap_or(&context.downstream_headers);
    let mut metadata = attempt_request_metadata(&context.method, &context.uri, headers);
    metadata.insert(
        String::from("upstream_request_body_bytes"),
        context.upstream_body.len().to_string(),
    );
    if context.response_adapter.is_some() {
        metadata.insert(String::from("path"), context.upstream_url.path().to_owned());
        metadata.insert(
            String::from("query_present"),
            context.upstream_url.query().is_some().to_string(),
        );
    }
    (override_headers, metadata)
}

fn merged_models_profiles(
    context: &GenericForwardContext<'_>,
) -> Option<Vec<UpstreamProfileConfig>> {
    if !is_control_plane_models_request(&context.method, &context.uri) {
        return None;
    }
    let profiles = listener_models_upstream_profiles(context.config, &context.state.listener);
    (profiles.len() > 1).then_some(profiles)
}

async fn begin_models_upstream_group(
    context: &GenericForwardContext<'_>,
    profile: UpstreamProfileConfig,
) -> Result<ModelsUpstreamGroup, ProxyError> {
    let is_current_profile = profile.name == context.upstream_profile.name;
    let request_deadline = if is_current_profile {
        context
            .upstream_deadline
            .unwrap_or_else(|| Instant::now() + Duration::from_millis(profile.request_timeout_ms))
    } else {
        Instant::now() + Duration::from_millis(profile.request_timeout_ms)
    };
    let (base_url, terminal_endpoint, endpoint_retry_order, recovery_trial_lease) =
        if is_current_profile {
            (
                context.terminal_endpoint.base_url.clone(),
                context.terminal_endpoint.clone(),
                context.endpoint_retry_order.clone(),
                None,
            )
        } else {
            let selected = context
            .state
            .upstream_health
            .select_endpoint(
                &context.state.client,
                &profile,
                &context.state.shutdown,
                None,
                None,
                Some(request_deadline),
            )
            .await
            .map_err(|error| match error {
                EndpointSelectionError::Shutdown => ProxyError::server_shutdown(),
                EndpointSelectionError::Incompatible { profile } => {
                    ProxyError::ContextBudgetExceeded {
                        message: format!(
                            "no OpenAI-compatible model discovery endpoint for profile {profile}"
                        ),
                        param: "path",
                        code: "unsupported_models_endpoint",
                        request_metadata: None,
                        attempts: Vec::new(),
                    }
                }
                EndpointSelectionError::Unavailable { profile, waited_ms } => {
                    ProxyError::upstream_unavailable(profile, waited_ms)
                }
            })?;
            (
                selected.base_url,
                selected.endpoint,
                selected.selection_order,
                selected.recovery_trial_lease,
            )
        };
    Ok(ModelsUpstreamGroup {
        base_url,
        request_timeout_ms: profile.request_timeout_ms,
        metadata: profile.metadata.clone(),
        profile,
        terminal_endpoint,
        endpoint_retry_order,
        _recovery_trial_lease: recovery_trial_lease,
        request_deadline,
    })
}

fn copy_endpoint_selection_metadata(
    source: &BTreeMap<String, String>,
    target: &mut BTreeMap<String, String>,
) {
    for key in [
        "upstream_endpoint_priority",
        "upstream_failover_selected",
        "upstream_endpoint_protocol",
        "upstream_endpoint_selection",
    ] {
        if let Some(value) = source.get(key) {
            target.insert(String::from(key), value.clone());
        }
    }
}

async fn forward_generic_openai_request(
    context: GenericForwardContext<'_>,
) -> Result<Response<Body>, ProxyError> {
    if let Some(profiles) = merged_models_profiles(&context) {
        return forward_merged_models_response(context, profiles).await;
    }

    let attempt_id = AttemptId::for_request(context.request_id, 1);
    let attempt_started_at_unix_ms = unix_time_millis();
    let (transformed_request_headers, mut attempt_request_metadata) =
        prepare_generic_attempt_request(&context);
    let downstream_headers = match transformed_request_headers.as_ref() {
        Some(headers) => headers,
        None => &context.downstream_headers,
    };
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
    copy_endpoint_selection_metadata(&context.request_metadata, &mut attempt_request_metadata);
    let sent_upstream_response = send_first_upstream_attempt(UpstreamAttemptContext {
        client: &context.state.client,
        method: context.reqwest_method.clone(),
        upstream_url: context.upstream_url.clone(),
        downstream_headers,
        upstream_body: context.upstream_body.clone(),
        retry_body: context.endpoint_retry_body.clone(),
        upstream_timeout: context.upstream_timeout,
        attempt_id: attempt_id.clone(),
        attempt_number: 1,
        request_id: context.request_id,
        attempt_started_at_unix_ms,
        request_metadata: &context.request_metadata,
        attempt_request_metadata: &attempt_request_metadata,
        shutdown: context.state.shutdown.subscribe(),
        failover_retry: context.upstream_profile.has_endpoint_failover().then_some(
            UpstreamFailoverRetryContext {
                registry: context.state.upstream_health.as_ref(),
                profile: &context.upstream_profile,
                local_forward_uri: context.upstream_uri.clone(),
                original_downstream_headers: &context.downstream_headers,
                canonical_reranker: context.canonical_reranker.as_ref(),
                transformed_request_headers: context.transformed_request_headers,
                initial_endpoint: &context.terminal_endpoint,
                request_deadline: context.upstream_deadline,
                endpoint_retry_order: &context.endpoint_retry_order,
                shutdown: context.state.shutdown.as_ref(),
            },
        ),
        terminal_endpoint_protocol: context.terminal_endpoint_protocol,
        canonical_reranker: context.canonical_reranker.as_ref(),
        decode_heterogeneous_reranker: context.response_adapter.as_ref().is_some_and(|adapter| {
            matches!(
                adapter,
                BufferedResponseAdapter::HeterogeneousReranker { .. }
            )
        }),
        model_id: context.model_id.as_deref(),
        request_deadline: context.upstream_deadline,
    })
    .await?;
    let GenericForwardedResponse {
        response_parts,
        upstream_response,
        terminal_endpoint_protocol,
    } = generic_forwarded_response(&context, sent_upstream_response);
    forward_generic_endpoint_response(
        context,
        response_parts,
        upstream_response,
        terminal_endpoint_protocol,
    )
    .await
}

async fn forward_generic_endpoint_response(
    context: GenericForwardContext<'_>,
    response_parts: ForwardedResponseParts,
    upstream_response: EndpointResponse,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
) -> Result<Response<Body>, ProxyError> {
    match upstream_response {
        EndpointResponse::Rewritten(rewritten) => Ok(forward_rewritten_endpoint_response(
            response_parts,
            context.in_flight_permit,
            rewritten,
        )),
        EndpointResponse::Upstream(upstream_response) => {
            if let Some(watchdog_request) = response_parts.stuck_watchdog_request.as_ref() {
                // The reqwest response exists only after the upstream has sent
                // response headers. Record this before handing the body to any
                // downstream-owned stream, which may be paused by backpressure.
                watchdog_request.record_upstream_response_started();
            }
            if let Some(adapter) = context
                .response_adapter
                .map(|adapter| adapter.with_terminal_protocol(terminal_endpoint_protocol))
            {
                return rewrite_buffered_adapter_response_from_upstream(
                    response_parts,
                    upstream_response,
                    context.in_flight_permit,
                    adapter,
                    context.model_id.as_deref(),
                )
                .await;
            }
            forward_upstream_response(
                ResponseDispatch {
                    method: &context.method,
                    uri: &context.uri,
                    config: context.config,
                    listener: &context.state.listener,
                    metadata_config: &context.upstream_profile.metadata,
                    malformed_response_counter: &context.state.malformed_response_counter,
                },
                response_parts,
                upstream_response,
                context.in_flight_permit,
            )
            .await
        }
    }
}

struct GenericForwardedResponse {
    response_parts: ForwardedResponseParts,
    upstream_response: EndpointResponse,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
}

fn generic_forwarded_response(
    context: &GenericForwardContext<'_>,
    sent_upstream_response: SentUpstreamResponse,
) -> GenericForwardedResponse {
    let terminal_endpoint_protocol = sent_upstream_response.terminal_endpoint_protocol;
    let upstream_status = sent_upstream_response.response.status();
    let upstream_headers = sent_upstream_response.response.headers().clone();
    let mut request_metadata = context.request_metadata.clone();
    if sent_upstream_response.did_failover {
        request_metadata.insert(
            String::from("upstream_failover_selected"),
            String::from("true"),
        );
    }
    let response_parts = ForwardedResponseParts {
        config: context.state.config.clone(),
        store: context.state.store.clone(),
        evidence_store: context.state.evidence_store.clone(),
        persistence_tasks: Arc::clone(&context.state.persistence_tasks),
        request_id: context.request_id.clone(),
        started_at_unix_ms: context.started_at_unix_ms,
        attempt_id: sent_upstream_response.attempt_id,
        attempt_number: sent_upstream_response.attempt_number,
        attempt_max_attempts: u32::try_from(context.upstream_profile.endpoints.len())
            .unwrap_or(u32::MAX)
            .max(1),
        attempt_started_at_unix_ms: sent_upstream_response.attempt_started_at_unix_ms,
        upstream_mode: upstream_mode_from_headers(&upstream_headers),
        model_id: context.model_id.clone(),
        input_fingerprint: context.liveness.input_fingerprint.clone(),
        upstream_status,
        upstream_headers,
        request_metadata,
        attempt_request_metadata: sent_upstream_response.attempt_request_metadata,
        completed_attempt_records: sent_upstream_response.completed_attempt_records,
        shutdown: Arc::clone(&context.state.shutdown),
        stuck_watchdog_request: context.stuck_watchdog_request.clone(),
    };
    GenericForwardedResponse {
        response_parts,
        upstream_response: sent_upstream_response.response,
        terminal_endpoint_protocol,
    }
}

fn forward_rewritten_endpoint_response(
    response_parts: ForwardedResponseParts,
    in_flight_permit: InFlightPermit,
    rewritten: RewrittenEndpointResponse,
) -> Response<Body> {
    let mut response_headers = HeaderMap::new();
    response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let shutdown = response_parts.shutdown_subscription();
    let observer = response_parts.into_observer_with(
        downstream_mode_from_headers(&response_headers),
        response_headers.clone(),
        BTreeMap::from([(
            String::from("response_body_bytes"),
            rewritten.upstream_body_bytes.to_string(),
        )]),
        BTreeMap::new(),
        RawPayloads::default(),
    );
    let response_body =
        ObservedBufferedBody::new(rewritten.body, observer, in_flight_permit, shutdown);
    downstream_response(
        rewritten.status,
        &response_headers,
        Body::from_stream(response_body),
    )
}

struct ModelsUpstreamGroup {
    profile: UpstreamProfileConfig,
    base_url: String,
    request_timeout_ms: u64,
    metadata: MetadataConfig,
    terminal_endpoint: UpstreamEndpointConfig,
    endpoint_retry_order: Vec<String>,
    // Kept alive while this group is fetched and its endpoint response is classified.
    _recovery_trial_lease: Option<upstream_failover::RecoveryTrialLease>,
    request_deadline: Instant,
}

struct CompletedModelsFetch {
    body: Bytes,
    completed_attempt_records: Vec<AttemptRecord>,
    attempt_record: AttemptRecord,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    upstream_mode: UpstreamMode,
}

async fn forward_merged_models_response(
    context: GenericForwardContext<'_>,
    profiles: Vec<UpstreamProfileConfig>,
) -> Result<Response<Body>, ProxyError> {
    let profile_count = profiles.len();
    let mut response_status = None;
    let mut response_headers = None;
    let mut response_mode = None;
    let mut filtered_bodies = Vec::with_capacity(profile_count);
    let mut attempt_records = Vec::with_capacity(profile_count);
    let mut selected_groups = Vec::<ModelsUpstreamGroup>::with_capacity(profile_count);
    let mut next_attempt_number = 1;
    for profile in profiles {
        let group = match begin_models_upstream_group(&context, profile).await {
            Ok(group) => group,
            Err(error) => return Err(error.with_completed_attempt_records(attempt_records)),
        };
        if selected_groups
            .iter()
            .any(|selected| same_models_endpoint_identity(selected, &group))
        {
            continue;
        }
        let fetch = match fetch_models_upstream_group(
            &context,
            &group,
            next_attempt_number,
            u32::try_from(profile_count).unwrap_or(u32::MAX),
        )
        .await
        {
            Ok(fetch) => fetch,
            Err(error) => return Err(error.with_completed_attempt_records(attempt_records)),
        };
        next_attempt_number = fetch.attempt_record.attempt_number.saturating_add(1);
        if response_status.is_none() {
            response_status = Some(fetch.upstream_status);
            response_headers = Some(fetch.upstream_headers.clone());
            response_mode = Some(fetch.upstream_mode);
        }
        filtered_bodies.push(fetch.body);
        attempt_records.extend(fetch.completed_attempt_records);
        attempt_records.push(fetch.attempt_record);
        selected_groups.push(group);
    }

    let metadata_config = selected_groups
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
        persistence_tasks: Arc::clone(&context.state.persistence_tasks),
        request_id: context.request_id.clone(),
        started_at_unix_ms: context.started_at_unix_ms,
        attempt_id: AttemptId::for_request(context.request_id, 1),
        attempt_number: 1,
        attempt_max_attempts: u32::try_from(selected_groups.len())
            .unwrap_or(u32::MAX)
            .max(1),
        attempt_started_at_unix_ms: context.started_at_unix_ms,
        upstream_mode,
        model_id: context.model_id,
        input_fingerprint: context.liveness.input_fingerprint.clone(),
        upstream_status,
        upstream_headers: upstream_headers.clone(),
        request_metadata: context.request_metadata,
        attempt_request_metadata: BTreeMap::new(),
        completed_attempt_records: Vec::new(),
        shutdown: Arc::clone(&context.state.shutdown),
        stuck_watchdog_request: None,
    };
    let shutdown = response_parts.shutdown_subscription();
    let mut observer = response_parts.into_observer();
    observer.completed_attempt_records = attempt_records;
    observer.final_attempt = None;
    let response_body =
        ObservedBufferedBody::new(body, observer, context.in_flight_permit, shutdown);
    Ok(downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    ))
}

fn same_models_endpoint_identity(left: &ModelsUpstreamGroup, right: &ModelsUpstreamGroup) -> bool {
    left.base_url == right.base_url
        && left.terminal_endpoint.protocol == right.terminal_endpoint.protocol
        && left.terminal_endpoint.api_key_env == right.terminal_endpoint.api_key_env
}

fn models_success_response_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers
}

fn models_failover_retry_context<'request>(
    context: &'request GenericForwardContext<'_>,
    group: &'request ModelsUpstreamGroup,
    downstream_headers: &'request HeaderMap,
) -> Option<UpstreamFailoverRetryContext<'request>> {
    group
        .profile
        .has_endpoint_failover()
        .then_some(UpstreamFailoverRetryContext {
            registry: context.state.upstream_health.as_ref(),
            profile: &group.profile,
            local_forward_uri: context.upstream_uri.clone(),
            original_downstream_headers: downstream_headers,
            canonical_reranker: None,
            transformed_request_headers: false,
            initial_endpoint: &group.terminal_endpoint,
            request_deadline: Some(group.request_deadline),
            endpoint_retry_order: &group.endpoint_retry_order,
            shutdown: context.state.shutdown.as_ref(),
        })
}

fn models_attempt_request(
    context: &GenericForwardContext<'_>,
    group: &ModelsUpstreamGroup,
    attempt_number: u32,
) -> (HeaderMap, BTreeMap<String, String>) {
    let downstream_headers = model_discovery_request_headers(&context.downstream_headers);
    let mut metadata = attempt_request_metadata(&context.method, &context.uri, &downstream_headers);
    metadata.insert(String::from("attempt_number"), attempt_number.to_string());
    add_listener_metadata(&mut metadata, &context.state.listener);
    add_upstream_profile_metadata(
        &mut metadata,
        &group.profile,
        UpstreamRouteReason::MatchedModel,
    );
    add_shielded_request_metadata(
        &mut metadata,
        false,
        context.thinking_policy_applied,
        &context.liveness,
        &context.thinking_metadata,
    );
    (downstream_headers, metadata)
}

async fn fetch_models_upstream_group(
    context: &GenericForwardContext<'_>,
    group: &ModelsUpstreamGroup,
    attempt_number: u32,
    attempt_max_attempts: u32,
) -> Result<CompletedModelsFetch, ProxyError> {
    let attempt_id = AttemptId::for_request(context.request_id, attempt_number);
    let attempt_started_at_unix_ms = unix_time_millis();
    let (downstream_headers, attempt_request_metadata) =
        models_attempt_request(context, group, attempt_number);

    let rendered = reranker_protocol::render_openai_endpoint(
        &group.terminal_endpoint,
        context.uri.clone(),
        &context.upstream_body,
        &downstream_headers,
        false,
    )?;
    let sent_upstream_response = send_first_upstream_attempt(UpstreamAttemptContext {
        client: &context.state.client,
        method: context.reqwest_method.clone(),
        upstream_url: rendered.url,
        downstream_headers: &rendered.headers,
        upstream_body: rendered.body,
        retry_body: context.upstream_body.clone(),
        upstream_timeout: Duration::from_millis(group.request_timeout_ms),
        attempt_id: attempt_id.clone(),
        attempt_number,
        request_id: context.request_id,
        attempt_started_at_unix_ms,
        request_metadata: &context.request_metadata,
        attempt_request_metadata: &attempt_request_metadata,
        shutdown: context.state.shutdown.subscribe(),
        failover_retry: models_failover_retry_context(context, group, &downstream_headers),
        terminal_endpoint_protocol: group.terminal_endpoint.protocol,
        canonical_reranker: None,
        decode_heterogeneous_reranker: false,
        model_id: None,
        request_deadline: Some(group.request_deadline),
    })
    .await?;
    let SentUpstreamResponse {
        response,
        attempt_id,
        attempt_number,
        attempt_started_at_unix_ms,
        attempt_request_metadata,
        completed_attempt_records,
        ..
    } = sent_upstream_response;
    let upstream_response = models_upstream_response(response)?;
    let upstream_mode = upstream_mode_from_headers(upstream_response.headers());
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
    let body = match read_upstream_body_bytes_until_shutdown(
        upstream_response.bytes_stream(),
        context.state.shutdown.subscribe(),
    )
    .await
    {
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
            )
            .with_completed_attempt_records(completed_attempt_records));
        }
    };
    let body_len = u64::try_from(body.len()).unwrap_or(u64::MAX);
    let body = filter_models_body_for_listener(context.config, &context.state.listener, body);
    let attempt_record = final_attempt_record(
        FinalAttemptContext {
            attempt_id,
            attempt_number,
            attempt_max_attempts: attempt_max_attempts.max(attempt_number),
            started_at_unix_ms: attempt_started_at_unix_ms,
            upstream_mode,
            upstream_status,
            upstream_headers: upstream_headers.clone(),
            request_metadata: attempt_request_metadata,
            extra_response_metadata: BTreeMap::new(),
            raw_payloads: RawPayloads::default(),
            response_body: body.clone(),
            sse_body: Bytes::new(),
        },
        context.request_id,
        unix_time_millis(),
        body_len,
        &BodyCompletion::Succeeded,
    );

    Ok(CompletedModelsFetch {
        body,
        completed_attempt_records,
        attempt_record,
        upstream_status,
        upstream_headers,
        upstream_mode,
    })
}

fn models_upstream_response(response: EndpointResponse) -> Result<reqwest::Response, ProxyError> {
    match response {
        EndpointResponse::Upstream(response) => Ok(response),
        EndpointResponse::Rewritten(_) => Err(ProxyError::upstream_body(String::from(
            "models request unexpectedly produced a rewritten endpoint response",
        ))),
    }
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

struct UpstreamFailoverRetryContext<'request> {
    registry: &'request UpstreamHealthRegistry,
    profile: &'request UpstreamProfileConfig,
    local_forward_uri: Uri,
    original_downstream_headers: &'request HeaderMap,
    canonical_reranker: Option<&'request CanonicalRerankerRequest>,
    transformed_request_headers: bool,
    initial_endpoint: &'request UpstreamEndpointConfig,
    request_deadline: Option<Instant>,
    endpoint_retry_order: &'request [String],
    shutdown: &'request ShutdownGate,
}

struct UpstreamAttemptContext<'request> {
    client: &'request Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: &'request HeaderMap,
    upstream_body: Bytes,
    retry_body: Bytes,
    upstream_timeout: Duration,
    attempt_id: AttemptId,
    attempt_number: u32,
    request_id: &'request RequestId,
    attempt_started_at_unix_ms: u64,
    request_metadata: &'request BTreeMap<String, String>,
    attempt_request_metadata: &'request BTreeMap<String, String>,
    shutdown: ShutdownSubscription,
    failover_retry: Option<UpstreamFailoverRetryContext<'request>>,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
    canonical_reranker: Option<&'request CanonicalRerankerRequest>,
    decode_heterogeneous_reranker: bool,
    model_id: Option<&'request str>,
    request_deadline: Option<Instant>,
}

struct SentUpstreamResponse {
    response: EndpointResponse,
    attempt_id: AttemptId,
    attempt_number: u32,
    attempt_started_at_unix_ms: u64,
    attempt_request_metadata: BTreeMap<String, String>,
    completed_attempt_records: Vec<AttemptRecord>,
    did_failover: bool,
    terminal_endpoint: Option<UpstreamEndpointConfig>,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
}

struct PhysicalEndpointAttempt {
    attempt_id: AttemptId,
    attempt_number: u32,
    started_at_unix_ms: u64,
    request_metadata: BTreeMap<String, String>,
    endpoint: Option<UpstreamEndpointConfig>,
    protocol: UpstreamEndpointProtocol,
    observed_response: Option<ObservedEndpointResponse>,
}

struct EndpointFailoverOutcome {
    result: Result<EndpointResponse, ProxyError>,
    terminal_attempt: PhysicalEndpointAttempt,
    completed_attempt_records: Vec<AttemptRecord>,
}

struct EndpointFailoverRuntime<'request> {
    client: &'request Client,
    retry: &'request UpstreamFailoverRetryContext<'request>,
    retry_method: reqwest::Method,
    retry_body: &'request Bytes,
    upstream_timeout: Duration,
    request_id: &'request RequestId,
    decode_heterogeneous_reranker: bool,
    model_id: Option<&'request str>,
}

struct EndpointFailoverProgress {
    result: Result<EndpointResponse, ProxyError>,
    terminal_attempt: PhysicalEndpointAttempt,
}

struct ObservedEndpointResponse {
    status: StatusCode,
    headers: HeaderMap,
}

enum EndpointResponse {
    Upstream(reqwest::Response),
    Rewritten(RewrittenEndpointResponse),
}

struct RewrittenEndpointResponse {
    body: Bytes,
    upstream_body_bytes: usize,
    status: StatusCode,
    headers: HeaderMap,
}

impl EndpointResponse {
    fn status(&self) -> StatusCode {
        match self {
            Self::Upstream(response) => response.status(),
            Self::Rewritten(response) => response.status,
        }
    }

    fn headers(&self) -> &HeaderMap {
        match self {
            Self::Upstream(response) => response.headers(),
            Self::Rewritten(response) => &response.headers,
        }
    }
}

fn render_retry_openai_request(
    retry: &UpstreamFailoverRetryContext<'_>,
    endpoint: &UpstreamEndpointConfig,
    body: &Bytes,
) -> Result<RenderedEndpointRequest, ProxyError> {
    reranker_protocol::render_openai_endpoint(
        endpoint,
        retry.local_forward_uri.clone(),
        body,
        retry.original_downstream_headers,
        retry.transformed_request_headers,
    )
}

async fn send_first_upstream_attempt(
    context: UpstreamAttemptContext<'_>,
) -> Result<SentUpstreamResponse, ProxyError> {
    let terminal_endpoint = context
        .failover_retry
        .as_ref()
        .map(|retry| (*retry.initial_endpoint).clone());
    let mut initial_attempt = PhysicalEndpointAttempt {
        attempt_id: context.attempt_id.clone(),
        attempt_number: context.attempt_number,
        started_at_unix_ms: context.attempt_started_at_unix_ms,
        request_metadata: context.attempt_request_metadata.clone(),
        endpoint: terminal_endpoint,
        protocol: context.terminal_endpoint_protocol,
        observed_response: None,
    };
    annotate_physical_endpoint_attempt(
        &mut initial_attempt.request_metadata,
        initial_attempt.endpoint.as_ref(),
        context.terminal_endpoint_protocol,
        &context.upstream_url,
        context.upstream_body.len(),
        false,
    );
    let endpoint_attempts_remaining = context
        .failover_retry
        .as_ref()
        .map_or(1, |retry| retry.endpoint_retry_order.len());
    let initial_timeout = upstream_timeout_within_deadline(
        context.upstream_timeout,
        context.request_deadline,
        endpoint_attempts_remaining,
    );
    let result = match initial_timeout {
        Ok(timeout) => {
            send_upstream_request_until_shutdown(
                context.client,
                context.method.clone(),
                context.upstream_url.clone(),
                context.downstream_headers,
                context.upstream_body.clone(),
                timeout,
                context.shutdown,
            )
            .await
        }
        Err(error) => Err(error),
    };
    let result = finalize_endpoint_response(
        result,
        &mut initial_attempt,
        context.canonical_reranker,
        context.decode_heterogeneous_reranker,
        context.model_id,
        context.failover_retry.as_ref().map(|retry| retry.shutdown),
    )
    .await;

    let EndpointFailoverOutcome {
        result,
        terminal_attempt,
        completed_attempt_records,
    } = match context.failover_retry.as_ref() {
        Some(retry) => {
            continue_endpoint_failover(
                EndpointFailoverRuntime {
                    client: context.client,
                    retry,
                    retry_method: context.method.clone(),
                    retry_body: &context.retry_body,
                    upstream_timeout: context.upstream_timeout,
                    request_id: context.request_id,
                    decode_heterogeneous_reranker: context.decode_heterogeneous_reranker,
                    model_id: context.model_id,
                },
                EndpointFailoverProgress {
                    result,
                    terminal_attempt: initial_attempt,
                },
            )
            .await
        }
        None => EndpointFailoverOutcome {
            result,
            terminal_attempt: initial_attempt,
            completed_attempt_records: Vec::new(),
        },
    };

    finalize_sent_upstream_response(
        result,
        terminal_attempt,
        completed_attempt_records,
        context.request_id,
        context.request_metadata,
    )
}

fn finalize_sent_upstream_response(
    result: Result<EndpointResponse, ProxyError>,
    mut terminal_attempt: PhysicalEndpointAttempt,
    completed_attempt_records: Vec<AttemptRecord>,
    request_id: &RequestId,
    request_metadata: &BTreeMap<String, String>,
) -> Result<SentUpstreamResponse, ProxyError> {
    let disposition = endpoint_disposition(
        &result,
        terminal_attempt.protocol,
        terminal_attempt.endpoint.as_ref(),
    );
    terminal_attempt.request_metadata.insert(
        String::from("endpoint_disposition"),
        String::from(disposition),
    );
    match result {
        Ok(response) => Ok(SentUpstreamResponse {
            response,
            attempt_id: terminal_attempt.attempt_id,
            attempt_number: terminal_attempt.attempt_number,
            attempt_started_at_unix_ms: terminal_attempt.started_at_unix_ms,
            attempt_request_metadata: terminal_attempt.request_metadata,
            did_failover: !completed_attempt_records.is_empty(),
            completed_attempt_records,
            terminal_endpoint: terminal_attempt.endpoint.clone(),
            terminal_endpoint_protocol: terminal_attempt.protocol,
        }),
        Err(error) => {
            let finished_at_unix_ms = unix_time_millis();
            let error_reason = error.to_string();
            let mut attempt_record = failed_attempt_record(FailedAttemptRecordInput {
                attempt_id: terminal_attempt.attempt_id,
                attempt_number: terminal_attempt.attempt_number,
                request_id: request_id.clone(),
                started_at_unix_ms: terminal_attempt.started_at_unix_ms,
                finished_at_unix_ms,
                error_type: error.error_type(),
                error_reason: &error_reason,
                request_metadata: terminal_attempt.request_metadata,
                extra_response_metadata: BTreeMap::from([(
                    String::from("endpoint_disposition"),
                    String::from(disposition),
                )]),
            });
            if let Some(observed) = terminal_attempt.observed_response {
                attempt_record.http_status = Some(observed.status.as_u16());
                attempt_record.upstream_mode = upstream_mode_from_headers(&observed.headers);
                attempt_record.response_metadata.insert(
                    String::from("upstream_response_received"),
                    String::from("true"),
                );
                attempt_record.response_metadata.insert(
                    String::from("http_status_success"),
                    observed.status.is_success().to_string(),
                );
                copy_selected_header_metadata(
                    &mut attempt_record.response_metadata,
                    &observed.headers,
                    "response",
                );
            }
            Err(error
                .with_observability(request_metadata.clone(), attempt_record)
                .with_completed_attempt_records(completed_attempt_records))
        }
    }
}

async fn continue_endpoint_failover(
    runtime: EndpointFailoverRuntime<'_>,
    progress: EndpointFailoverProgress,
) -> EndpointFailoverOutcome {
    let EndpointFailoverProgress {
        mut result,
        mut terminal_attempt,
    } = progress;
    let mut attempted_base_urls = vec![runtime.retry.initial_endpoint.base_url.clone()];
    let mut completed_attempt_records = Vec::new();
    let eligible_endpoint_count = UpstreamHealthRegistry::eligible_endpoint_count(
        runtime.retry.profile,
        runtime.retry.canonical_reranker,
        Some(runtime.retry.original_downstream_headers),
    );
    let mut terminal_recovery_trial_lease = None;
    while is_retryable_endpoint_result(
        &result,
        terminal_attempt.protocol,
        terminal_attempt.endpoint.as_ref(),
    ) {
        mark_retryable_endpoint_failure(runtime.retry.registry, &terminal_attempt, &result);
        if attempted_base_urls.len() >= eligible_endpoint_count {
            break;
        }
        let selected = match runtime
            .retry
            .registry
            .select_endpoint_excluding(
                runtime.client,
                runtime.retry.profile,
                runtime.retry.shutdown,
                EndpointSelectionConstraints {
                    request: runtime.retry.canonical_reranker,
                    request_headers: Some(runtime.retry.original_downstream_headers),
                    request_deadline: runtime.retry.request_deadline,
                    preferred_base_urls: Some(runtime.retry.endpoint_retry_order),
                    excluded_base_urls: &attempted_base_urls,
                },
            )
            .await
        {
            Ok(selected) => selected,
            Err(
                EndpointSelectionError::Shutdown
                | EndpointSelectionError::Incompatible { .. }
                | EndpointSelectionError::Unavailable { .. },
            ) => break,
        };
        completed_attempt_records.push(retried_endpoint_attempt_record(
            &result,
            endpoint_disposition(
                &result,
                terminal_attempt.protocol,
                terminal_attempt.endpoint.as_ref(),
            ),
            terminal_attempt.attempt_id.clone(),
            terminal_attempt.attempt_number,
            runtime.request_id.clone(),
            terminal_attempt.started_at_unix_ms,
            terminal_attempt.request_metadata.clone(),
        ));
        let endpoint_attempts_remaining =
            eligible_endpoint_count.saturating_sub(attempted_base_urls.len());
        attempted_base_urls.push(selected.base_url.clone());
        let next = send_selected_failover_endpoint(
            &runtime,
            terminal_attempt.attempt_number.saturating_add(1),
            endpoint_attempts_remaining,
            context_attempt_metadata(&terminal_attempt),
            selected,
        )
        .await;
        #[cfg(test)]
        runtime
            .retry
            .registry
            .wait_before_endpoint_classification()
            .await;
        result = next.result;
        terminal_attempt = next.attempt;
        terminal_recovery_trial_lease = next.recovery_trial_lease;
    }
    finish_passive_recovery_trial(runtime.retry.registry, &terminal_attempt, &result);
    drop(terminal_recovery_trial_lease);
    EndpointFailoverOutcome {
        result,
        terminal_attempt,
        completed_attempt_records,
    }
}

fn context_attempt_metadata(attempt: &PhysicalEndpointAttempt) -> BTreeMap<String, String> {
    let mut metadata = attempt.request_metadata.clone();
    for key in [
        "upstream_endpoint_protocol",
        "upstream_endpoint_base_url",
        "upstream_endpoint_priority",
        "upstream_failover_selected",
        "path",
        "upstream_request_body_bytes",
    ] {
        metadata.remove(key);
    }
    metadata
}

struct CompletedFailoverEndpointSend {
    result: Result<EndpointResponse, ProxyError>,
    attempt: PhysicalEndpointAttempt,
    recovery_trial_lease: Option<upstream_failover::RecoveryTrialLease>,
}

async fn send_selected_failover_endpoint(
    runtime: &EndpointFailoverRuntime<'_>,
    attempt_number: u32,
    endpoint_attempts_remaining: usize,
    request_metadata: BTreeMap<String, String>,
    mut selected: upstream_failover::SelectedUpstreamEndpoint,
) -> CompletedFailoverEndpointSend {
    let rendered = match runtime.retry.canonical_reranker {
        Some(canonical) => reranker_protocol::render(
            &selected.endpoint,
            canonical,
            runtime.retry.original_downstream_headers,
        ),
        None => render_retry_openai_request(runtime.retry, &selected.endpoint, runtime.retry_body),
    };
    let attempt_id = AttemptId::for_request(runtime.request_id, attempt_number);
    let started_at_unix_ms = unix_time_millis();
    let mut attempt = PhysicalEndpointAttempt {
        attempt_id,
        attempt_number,
        started_at_unix_ms,
        request_metadata,
        endpoint: Some(selected.endpoint.clone()),
        protocol: selected.endpoint.protocol,
        observed_response: None,
    };
    let response = match rendered {
        Ok(rendered) => {
            annotate_physical_endpoint_attempt(
                &mut attempt.request_metadata,
                attempt.endpoint.as_ref(),
                attempt.protocol,
                &rendered.url,
                rendered.body.len(),
                true,
            );
            match upstream_timeout_within_deadline(
                runtime.upstream_timeout,
                runtime.retry.request_deadline,
                endpoint_attempts_remaining,
            ) {
                Ok(timeout) => {
                    send_upstream_request_until_shutdown(
                        runtime.client,
                        runtime.retry_method.clone(),
                        rendered.url,
                        &rendered.headers,
                        rendered.body,
                        timeout,
                        runtime.retry.shutdown.subscribe(),
                    )
                    .await
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    };
    let result = finalize_endpoint_response(
        response,
        &mut attempt,
        runtime.retry.canonical_reranker,
        runtime.decode_heterogeneous_reranker,
        runtime.model_id,
        Some(runtime.retry.shutdown),
    )
    .await;
    CompletedFailoverEndpointSend {
        result,
        attempt,
        recovery_trial_lease: selected.recovery_trial_lease.take(),
    }
}

async fn finalize_endpoint_response(
    response: Result<reqwest::Response, ProxyError>,
    attempt: &mut PhysicalEndpointAttempt,
    canonical_reranker: Option<&CanonicalRerankerRequest>,
    decode_heterogeneous_reranker: bool,
    model_id: Option<&str>,
    shutdown: Option<&ShutdownGate>,
) -> Result<EndpointResponse, ProxyError> {
    let response = response?;
    let status = response.status();
    let headers = response.headers().clone();
    attempt.observed_response = Some(ObservedEndpointResponse {
        status,
        headers: headers.clone(),
    });
    if !decode_heterogeneous_reranker || !status.is_success() {
        return Ok(EndpointResponse::Upstream(response));
    }
    let Some(request) = canonical_reranker else {
        return Ok(EndpointResponse::Upstream(response));
    };
    let Some(shutdown) = shutdown else {
        return Err(ProxyError::upstream_body(String::from(
            "heterogeneous reranker response has no shutdown controller",
        )));
    };
    let body =
        read_upstream_body_bytes_until_shutdown(response.bytes_stream(), shutdown.subscribe())
            .await?;
    let upstream_body_bytes = body.len();
    let body = reranker_protocol::rewrite_response_for_endpoint(
        &body,
        request,
        attempt.protocol,
        model_id,
    )
    .map_err(|error| {
        ProxyError::upstream_body(format!(
            "heterogeneous reranker response rewrite failed: {error}"
        ))
    })?;
    Ok(EndpointResponse::Rewritten(RewrittenEndpointResponse {
        body,
        upstream_body_bytes,
        status,
        headers,
    }))
}

fn upstream_timeout_within_deadline(
    configured_timeout: Duration,
    request_deadline: Option<Instant>,
    endpoint_attempts_remaining: usize,
) -> Result<Duration, ProxyError> {
    let Some(request_deadline) = request_deadline else {
        return Ok(configured_timeout);
    };
    let remaining = request_deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(ProxyError::UpstreamTransport {
            failure: ReqwestFailureKind::Timeout,
            observability: None,
        });
    }
    let endpoint_attempts_remaining = u32::try_from(endpoint_attempts_remaining)
        .unwrap_or(u32::MAX)
        .max(1);
    Ok(configured_timeout.min(remaining / endpoint_attempts_remaining))
}

fn annotate_physical_endpoint_attempt(
    metadata: &mut BTreeMap<String, String>,
    endpoint: Option<&UpstreamEndpointConfig>,
    protocol: UpstreamEndpointProtocol,
    url: &Url,
    body_len: usize,
    failover_selected: bool,
) {
    metadata.insert(
        String::from("upstream_endpoint_protocol"),
        protocol.as_str().to_owned(),
    );
    metadata.insert(
        String::from("upstream_endpoint_base_url"),
        endpoint.map_or_else(
            || redact_upstream_base_url(url.as_str()),
            |endpoint| redact_upstream_base_url(&endpoint.base_url),
        ),
    );
    metadata.insert(
        String::from("upstream_endpoint_priority"),
        endpoint
            .map_or("primary", |endpoint| endpoint.priority.as_str())
            .to_owned(),
    );
    metadata.insert(
        String::from("upstream_failover_selected"),
        failover_selected.to_string(),
    );
    metadata.insert(String::from("path"), url.path().to_owned());
    metadata.insert(
        String::from("upstream_request_body_bytes"),
        body_len.to_string(),
    );
}

fn retried_endpoint_attempt_record(
    result: &Result<EndpointResponse, ProxyError>,
    disposition: &'static str,
    attempt_id: AttemptId,
    attempt_number: u32,
    request_id: RequestId,
    started_at_unix_ms: u64,
    mut request_metadata: BTreeMap<String, String>,
) -> AttemptRecord {
    let finished_at_unix_ms = unix_time_millis();
    request_metadata.insert(
        String::from("endpoint_disposition"),
        String::from(disposition),
    );
    match result {
        Ok(response) => {
            let upstream_mode = upstream_mode_from_headers(response.headers());
            let mut response_metadata = response_metadata(
                response.status(),
                response.headers(),
                0,
                finished_at_unix_ms.saturating_sub(started_at_unix_ms),
            );
            response_metadata.insert(String::from("attempt_outcome"), String::from("retried"));
            response_metadata.insert(
                String::from("endpoint_disposition"),
                String::from("retryable_failure"),
            );
            response_metadata.insert(
                String::from("retry_reason"),
                retry_reason_for_endpoint_result(result),
            );
            AttemptRecord {
                attempt_id,
                request_id,
                attempt_number,
                started_at_unix_ms,
                finished_at_unix_ms: Some(finished_at_unix_ms),
                upstream_mode,
                status: AttemptStatus::Retried,
                http_status: Some(response.status().as_u16()),
                error_reason: Some(format!(
                    "upstream HTTP {} selected for failover",
                    response.status()
                )),
                retry_reason: Some(retry_reason_for_endpoint_result(result)),
                abort_reason: None,
                token_usage: TokenUsage::default(),
                request_metadata,
                response_metadata,
                raw_payloads: RawPayloads::default(),
            }
        }
        Err(error) => {
            let error_reason = error.to_string();
            let mut record = failed_attempt_record(FailedAttemptRecordInput {
                attempt_id,
                attempt_number,
                request_id,
                started_at_unix_ms,
                finished_at_unix_ms,
                error_type: error.error_type(),
                error_reason: &error_reason,
                request_metadata,
                extra_response_metadata: BTreeMap::from([
                    (String::from("attempt_outcome"), String::from("retried")),
                    (
                        String::from("endpoint_disposition"),
                        String::from("retryable_failure"),
                    ),
                ]),
            });
            record.status = AttemptStatus::Retried;
            record.retry_reason = Some(retry_reason_for_endpoint_result(result));
            record
        }
    }
}

fn retry_reason_for_endpoint_result(result: &Result<EndpointResponse, ProxyError>) -> String {
    match result {
        Ok(response) => format!("endpoint_http_{}", response.status().as_u16()),
        Err(ProxyError::UpstreamTransport { failure, .. }) => {
            format!("endpoint_{}", failure.as_str())
        }
        Err(ProxyError::UpstreamBody { .. }) => String::from("endpoint_protocol_response"),
        Err(_) => String::from("endpoint_retry"),
    }
}

fn is_retryable_endpoint_result(
    result: &Result<EndpointResponse, ProxyError>,
    protocol: UpstreamEndpointProtocol,
    endpoint: Option<&UpstreamEndpointConfig>,
) -> bool {
    match result {
        Err(ProxyError::UpstreamTransport { failure, .. }) => {
            matches!(
                failure,
                ReqwestFailureKind::Connect | ReqwestFailureKind::Timeout
            )
        }
        Err(ProxyError::UpstreamBody { .. }) => true,
        Ok(response) => match response.status() {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                protocol == UpstreamEndpointProtocol::DeepInfraQwen3Rerank
                    || endpoint_has_configured_credential(endpoint)
            }
            StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT => true,
            _ => false,
        },
        Err(_) => false,
    }
}

fn endpoint_has_configured_credential(endpoint: Option<&UpstreamEndpointConfig>) -> bool {
    endpoint.is_some_and(|endpoint| endpoint.api_key_env.is_some())
}

fn is_caller_scoped_rate_limit(result: &Result<EndpointResponse, ProxyError>) -> bool {
    matches!(result, Ok(response) if response.status() == StatusCode::TOO_MANY_REQUESTS)
}

fn mark_retryable_endpoint_failure(
    registry: &UpstreamHealthRegistry,
    attempt: &PhysicalEndpointAttempt,
    result: &Result<EndpointResponse, ProxyError>,
) {
    if (attempt.protocol == UpstreamEndpointProtocol::DeepInfraQwen3Rerank
        || endpoint_has_configured_credential(attempt.endpoint.as_ref())
        || !is_caller_scoped_rate_limit(result))
        && let Some(endpoint) = attempt.endpoint.as_ref()
    {
        registry.mark_unhealthy(endpoint);
    }
}

fn finish_passive_recovery_trial(
    registry: &UpstreamHealthRegistry,
    attempt: &PhysicalEndpointAttempt,
    result: &Result<EndpointResponse, ProxyError>,
) {
    let Some(endpoint) = attempt.endpoint.as_ref() else {
        return;
    };
    if !upstream_failover::is_passive_cloud_endpoint(endpoint)
        || is_retryable_endpoint_result(result, attempt.protocol, Some(endpoint))
    {
        return;
    }
    if result.is_ok() {
        registry.mark_healthy(endpoint);
    }
}

fn endpoint_disposition(
    result: &Result<EndpointResponse, ProxyError>,
    protocol: UpstreamEndpointProtocol,
    endpoint: Option<&UpstreamEndpointConfig>,
) -> &'static str {
    if is_retryable_endpoint_result(result, protocol, endpoint) {
        return "retryable_failure";
    }
    match result {
        Ok(response) if response.status().is_success() => "success",
        Ok(_) => "nonretryable_response",
        Err(_) => "terminal_failure",
    }
}

struct ResponseDispatch<'request> {
    method: &'request Method,
    uri: &'request Uri,
    config: &'request AppConfig,
    listener: &'request ListenerConfig,
    metadata_config: &'request MetadataConfig,
    malformed_response_counter: &'request AtomicU64,
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
    upstream_profile: &UpstreamProfileConfig,
    method: &Method,
    uri: &Uri,
    body: &Bytes,
) -> ShieldedChatPlan {
    let thinking = &upstream_profile.thinking;
    let retry_initial_thinking = config.retry.ladder.first().map_or_else(
        || thinking.clone(),
        |entry| retry_ladder_thinking(entry, thinking),
    );
    let (request, intercepted, kind) = if should_intercept_non_stream_chat(method, uri, config) {
        if let Some(non_stream_request) =
            shielded_chat::prepare_non_stream_request(body, &retry_initial_thinking)
        {
            (Some(non_stream_request), true, ShieldedChatKind::NonStream)
        } else if let Some(stream_request) = shielded_chat::prepare_stream_request(
            body,
            if config.retry.shielded_streaming_enabled {
                &retry_initial_thinking
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
    request_deadline: Duration,
    anti_loop_hint_enabled: bool,
    shielded_streaming_enabled: bool,
    downstream_drop_policy: DownstreamDropPolicy,
    loop_failure_policy: LoopFailurePolicy,
    ladder: Vec<RetryLadderConfig>,
}

impl ShieldedRetryPolicy {
    fn from_config(config: &RetryConfig, loop_guard: &LoopGuardConfig) -> Self {
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
            request_deadline: Duration::from_millis(config.request_deadline_ms),
            anti_loop_hint_enabled: config.anti_loop_hint_enabled,
            shielded_streaming_enabled: config.shielded_streaming_enabled,
            downstream_drop_policy: config.downstream_drop_policy,
            loop_failure_policy: loop_guard.on_reasoning_loop,
            ladder: config.ladder.clone(),
        }
    }

    fn allows_retry_after(&self, attempt_number: u32) -> bool {
        self.enabled && attempt_number < self.max_attempts
    }

    fn attempt_plan(
        &self,
        attempt_number: u32,
        upstream_profile: &UpstreamProfileConfig,
        cot_salvage: Option<&CotSalvageContext>,
    ) -> ShieldedAttemptPlan {
        let fallback_thinking = &upstream_profile.thinking;
        let index = attempt_number.saturating_sub(1);
        let mut plan = self
            .ladder
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .map_or_else(
                || ShieldedAttemptPlan {
                    name: format!("attempt-{attempt_number}"),
                    thinking: fallback_thinking.clone(),
                    anti_loop_hint: None,
                },
                |entry| ShieldedAttemptPlan {
                    name: entry.name.clone(),
                    thinking: retry_ladder_thinking(entry, fallback_thinking),
                    anti_loop_hint: entry.anti_loop_hint.clone(),
                },
            );
        if let Some(cot_salvage) = cot_salvage {
            plan.thinking = cot_salvage_thinking(cot_salvage.policy, &plan.thinking);
        }
        plan
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShieldedAttemptPlan {
    name: String,
    thinking: ThinkingConfig,
    anti_loop_hint: Option<String>,
}

fn retry_ladder_thinking(
    entry: &RetryLadderConfig,
    fallback_thinking: &ThinkingConfig,
) -> ThinkingConfig {
    let mut thinking = entry.thinking.clone();
    thinking.default_injection_schema = entry
        .default_injection_schema
        .unwrap_or(fallback_thinking.default_injection_schema);
    if thinking.effective_mode() == ThinkingMode::ForceDisable {
        thinking.budget_tokens = 0;
    }
    thinking
}

fn cot_salvage_thinking(policy: LoopFailurePolicy, current: &ThinkingConfig) -> ThinkingConfig {
    let mut thinking = current.clone();
    match policy {
        LoopFailurePolicy::RetryLadder => {}
        LoopFailurePolicy::TruncateCotThenAnswer => {
            thinking.mode = ThinkingMode::ForceDisable;
            thinking.enabled = false;
            thinking.force_disable = true;
            thinking.budget_tokens = 0;
            thinking.preserve_answer_budget = false;
        }
        LoopFailurePolicy::BoundedAnswerFromCot => {
            thinking.mode = ThinkingMode::BoundedThinking;
            thinking.enabled = true;
            thinking.force_disable = false;
            thinking.budget_tokens = COT_SALVAGE_THINKING_BUDGET_TOKENS;
            thinking.preserve_answer_budget = false;
        }
    }
    thinking
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UpstreamStallPolicy {
    enabled: bool,
    first_chunk_timeout: Duration,
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
            first_chunk_timeout: Duration::from_millis(config.first_chunk_timeout_ms),
            idle_timeout: Duration::from_millis(config.idle_timeout_ms),
            recovery_command: config.recovery_command.clone(),
            recovery_timeout: Duration::from_millis(config.recovery_timeout_ms),
            recovery_cooldown: Duration::from_millis(config.recovery_cooldown_ms),
            recovery_budget_window: Duration::from_millis(config.recovery_budget_window_ms),
            recovery_max_per_window: config.recovery_max_per_window,
        }
    }

    const fn stream_timeouts(&self) -> Option<UpstreamStreamTimeouts> {
        if self.enabled {
            Some(UpstreamStreamTimeouts {
                first_chunk: self.first_chunk_timeout,
                inter_chunk: self.idle_timeout,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UpstreamStreamTimeouts {
    first_chunk: Duration,
    inter_chunk: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LocalRecoveryPolicy {
    enabled: bool,
    restart_command: Vec<String>,
    restart_timeout: Duration,
    readiness_endpoint: String,
    readiness_body: serde_json::Value,
    readiness_request_timeout: Duration,
    readiness_deadline: Duration,
    readiness_interval: Duration,
    max_attempts_per_request: u32,
    cooldown: Duration,
    budget_window: Duration,
    max_per_window: u32,
}

impl LocalRecoveryPolicy {
    fn from_config(config: &LocalRecoveryConfig) -> Self {
        Self {
            enabled: config.enabled,
            restart_command: config.restart_command.clone(),
            restart_timeout: Duration::from_millis(config.restart_timeout_ms),
            readiness_endpoint: config.readiness_endpoint.clone(),
            readiness_body: config.readiness_body.clone(),
            readiness_request_timeout: Duration::from_millis(config.readiness_request_timeout_ms),
            readiness_deadline: Duration::from_millis(config.readiness_deadline_ms),
            readiness_interval: Duration::from_millis(config.readiness_interval_ms),
            max_attempts_per_request: config.max_attempts_per_request,
            cooldown: Duration::from_millis(config.cooldown_ms),
            budget_window: Duration::from_millis(config.budget_window_ms),
            max_per_window: config.max_per_window,
        }
    }

    fn is_configured(&self) -> bool {
        self.enabled && !self.restart_command.is_empty()
    }
}

#[derive(Debug, Default)]
struct UpstreamStallRecoveryCoordinator {
    state: AsyncMutex<UpstreamStallRecoveryState>,
    notify: Notify,
    restart_queue_depth: AtomicU64,
    watchdog_detections: AtomicU64,
    watchdog_restarts: AtomicU64,
    watchdog_recovery_successes: AtomicU64,
    watchdog_recovery_timeouts: AtomicU64,
}

#[derive(Debug, Default)]
struct UpstreamStallRecoveryState {
    running: bool,
    recovery_started: Option<Instant>,
    recovery_deadline: Option<Instant>,
    last_finished: Option<Instant>,
    window_started: Option<Instant>,
    runs_in_window: u32,
    last_result: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Default)]
struct StuckWatchdogTokenTracker {
    windows: Mutex<HashMap<String, StuckWatchdogTokenWindow>>,
    next_attempt_id: AtomicU64,
}

#[derive(Debug, Default)]
struct StuckWatchdogTokenWindow {
    samples: VecDeque<(Instant, u64)>,
    attempts: HashMap<u64, StuckWatchdogAttempt>,
}

/// Upstream liveness is scoped to a concrete attempt rather than a profile-wide
/// aggregate, so an older completion cannot age a newer request into recovery.
#[derive(Debug)]
struct StuckWatchdogAttempt {
    started_at: Instant,
    response_started: Option<Instant>,
    completed: Option<Instant>,
}

#[derive(Clone, Copy, Debug)]
enum WatchdogProgressUnit {
    Chat,
    Embedding,
    Reranker,
}

fn watchdog_progress_unit(uri: &Uri) -> WatchdogProgressUnit {
    match uri.path() {
        path if path.contains("embeddings") => WatchdogProgressUnit::Embedding,
        path if path.contains("rerank") || path.contains("score") => WatchdogProgressUnit::Reranker,
        _ => WatchdogProgressUnit::Chat,
    }
}

#[derive(Debug)]
struct StuckWatchdogRequestInner {
    tracker: Arc<StuckWatchdogTokenTracker>,
    profile: String,
    attempt_id: u64,
    progress_unit: WatchdogProgressUnit,
    detection_window: Duration,
    sse_buffer: Mutex<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct StuckWatchdogRequest {
    inner: Arc<StuckWatchdogRequestInner>,
}

impl StuckWatchdogTokenTracker {
    fn begin_request(
        self: &Arc<Self>,
        profile: &str,
        progress_unit: WatchdogProgressUnit,
        detection_window: Duration,
    ) -> StuckWatchdogRequest {
        let mut windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let window = windows.entry(profile.to_owned()).or_default();
        let attempt_id = self.next_attempt_id.fetch_add(1, Ordering::Relaxed);
        window.attempts.insert(
            attempt_id,
            StuckWatchdogAttempt {
                started_at: Instant::now(),
                response_started: None,
                completed: None,
            },
        );
        StuckWatchdogRequest {
            inner: Arc::new(StuckWatchdogRequestInner {
                tracker: Arc::clone(self),
                profile: profile.to_owned(),
                attempt_id,
                progress_unit,
                detection_window,
                sse_buffer: Mutex::new(Vec::new()),
            }),
        }
    }

    fn record_response(
        &self,
        profile: &str,
        detection_window: Duration,
        response_body: &[u8],
        sse_body: &[u8],
    ) {
        let Some(output_tokens) = parse_token_usage(response_body, sse_body).output_tokens else {
            return;
        };
        self.prune_profile(profile, detection_window);
        let mut windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        record_watchdog_progress(
            windows.entry(profile.to_owned()).or_default(),
            Instant::now(),
            output_tokens,
        );
    }

    fn record_progress(&self, profile: &str, detection_window: Duration, progress: u64) {
        if progress == 0 {
            return;
        }
        self.prune_profile(profile, detection_window);
        let mut windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        record_watchdog_progress(
            windows.entry(profile.to_owned()).or_default(),
            Instant::now(),
            progress,
        );
    }

    #[cfg(test)]
    fn sample_count(&self, profile: &str) -> usize {
        let windows = self
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        windows
            .get(profile)
            .map_or(0, |window| window.samples.len())
    }

    fn prune_profile(&self, profile: &str, detection_window: Duration) {
        let mut windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(window) = windows.get_mut(profile) {
            let now = Instant::now();
            prune_watchdog_progress(window, detection_window, now);
            prune_watchdog_attempts(window, detection_window, now);
        }
    }

    fn has_too_few_output_tokens(
        &self,
        profile: &str,
        detection_window: Duration,
        minimum_output_tokens: u64,
    ) -> bool {
        let mut windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(window) = windows.get_mut(profile) else {
            return true;
        };
        window_has_too_few_output_tokens(
            window,
            detection_window,
            minimum_output_tokens,
            Instant::now(),
        )
    }

    fn has_active_requests(&self, profile: &str) -> bool {
        let windows = match self.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        windows.get(profile).is_some_and(|window| {
            window
                .attempts
                .values()
                .any(|attempt| attempt.completed.is_none())
        })
    }
}

impl StuckWatchdogRequest {
    /// Response headers were received from the upstream before the body is
    /// handed to the client.  This intentionally does not depend on downstream
    /// consumption, which can be paused indefinitely by client backpressure.
    fn record_upstream_response_started(&self) {
        let mut windows = match self.inner.tracker.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(attempt) = windows
            .get_mut(&self.inner.profile)
            .and_then(|window| window.attempts.get_mut(&self.inner.attempt_id))
        {
            attempt.response_started.get_or_insert_with(Instant::now);
        }
    }

    fn record_response(&self, response_body: &[u8], sse_body: &[u8]) {
        self.inner.tracker.record_response(
            &self.inner.profile,
            self.inner.detection_window,
            response_body,
            sse_body,
        );
    }

    fn record_emitted_chunk(&self, chunk: &[u8]) {
        let progress = match self.inner.progress_unit {
            WatchdogProgressUnit::Chat => self.record_sse_content_progress(chunk),
            // Embedding and reranker endpoints generally omit completion_tokens; a
            // result-bearing chunk is their endpoint-appropriate progress unit.
            WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
                u64::from(!chunk.is_empty())
            }
        };
        self.inner.tracker.record_progress(
            &self.inner.profile,
            self.inner.detection_window,
            progress,
        );
    }

    fn record_sse_content_progress(&self, chunk: &[u8]) -> u64 {
        const MAX_PENDING_SSE_BYTES: usize = 64 * 1024;
        let mut pending = self
            .inner
            .sse_buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending.extend(chunk.iter().copied().filter(|byte| *byte != b'\r'));
        if pending.len() > MAX_PENDING_SSE_BYTES {
            pending.clear();
            return 0;
        }

        let mut progress = 0_u64;
        while let Some(frame_end) = pending.windows(2).position(|window| window == b"\n\n") {
            let frame = pending.drain(..frame_end + 2).collect::<Vec<_>>();
            progress = progress.saturating_add(sse_content_progress(&frame));
        }
        progress
    }
}

impl Drop for StuckWatchdogRequestInner {
    fn drop(&mut self) {
        let mut windows = match self.tracker.windows.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(attempt) = windows
            .get_mut(&self.profile)
            .and_then(|window| window.attempts.get_mut(&self.attempt_id))
        {
            attempt.completed = Some(Instant::now());
        }
    }
}

fn window_has_too_few_output_tokens(
    window: &mut StuckWatchdogTokenWindow,
    detection_window: Duration,
    minimum_output_tokens: u64,
    now: Instant,
) -> bool {
    prune_watchdog_attempts(window, detection_window, now);
    let mut active_attempts = window
        .attempts
        .values()
        .filter(|attempt| attempt.completed.is_none());
    if active_attempts
        .clone()
        .any(|attempt| attempt.response_started.is_some())
        || active_attempts
            .any(|attempt| now.saturating_duration_since(attempt.started_at) < detection_window)
    {
        return false;
    }
    prune_watchdog_progress(window, detection_window, now);
    window
        .samples
        .iter()
        .fold(0_u64, |total, (_, output_tokens)| {
            total.saturating_add(*output_tokens)
        })
        < minimum_output_tokens
}

fn sse_content_progress(frame: &[u8]) -> u64 {
    frame
        .split(|byte| *byte == b'\n')
        .filter_map(|line| line.strip_prefix(b"data:"))
        .map(trim_ascii)
        .filter(|data| !data.is_empty() && *data != b"[DONE]")
        .filter_map(|data| serde_json::from_slice::<serde_json::Value>(data).ok())
        .filter(sse_event_has_model_content)
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len().saturating_sub(1)];
    }
    bytes
}

fn sse_event_has_model_content(event: &serde_json::Value) -> bool {
    event
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                let Some(delta) = choice.get("delta").and_then(serde_json::Value::as_object) else {
                    return false;
                };
                ["content", "reasoning_content", "reasoning", "thinking"]
                    .iter()
                    .any(|field| {
                        delta
                            .get(*field)
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                    })
                    || ["tool_calls", "function_call"].iter().any(|field| {
                        delta.get(*field).is_some_and(|value| match value {
                            serde_json::Value::Array(values) => !values.is_empty(),
                            serde_json::Value::Object(values) => !values.is_empty(),
                            serde_json::Value::Null => false,
                            _ => true,
                        })
                    })
            })
        })
}

fn record_watchdog_progress(window: &mut StuckWatchdogTokenWindow, now: Instant, progress: u64) {
    window.samples.push_back((now, progress));
    while window.samples.len() > STUCK_WATCHDOG_TOKEN_SAMPLE_CAP {
        let _ = window.samples.pop_front();
    }
}

fn prune_watchdog_progress(
    window: &mut StuckWatchdogTokenWindow,
    detection_window: Duration,
    now: Instant,
) {
    if let Some(threshold) = now.checked_sub(detection_window) {
        while window
            .samples
            .front()
            .is_some_and(|(recorded_at, _)| *recorded_at < threshold)
        {
            let _ = window.samples.pop_front();
        }
    }
}

fn prune_watchdog_attempts(
    window: &mut StuckWatchdogTokenWindow,
    detection_window: Duration,
    now: Instant,
) {
    let Some(threshold) = now.checked_sub(detection_window) else {
        return;
    };
    window.attempts.retain(|_, attempt| {
        attempt
            .completed
            .is_none_or(|completed_at| completed_at >= threshold)
    });
}

#[derive(Debug, Default)]
struct WatchdogSchedule {
    next_due: HashMap<String, WatchdogScheduleEntry>,
}

#[derive(Debug)]
struct WatchdogScheduleEntry {
    next_due: Instant,
    applied_interval: Duration,
}

impl WatchdogSchedule {
    fn due_profiles(&mut self, now: Instant, profiles: &[(String, Duration)]) -> Vec<String> {
        profiles
            .iter()
            .filter_map(|(name, interval)| {
                let due = self.next_due.get(name).is_none_or(|entry| {
                    entry.next_due <= now || entry.applied_interval != *interval
                });
                due.then(|| {
                    self.next_due.insert(
                        name.clone(),
                        WatchdogScheduleEntry {
                            next_due: now + *interval,
                            applied_interval: *interval,
                        },
                    );
                    name.clone()
                })
            })
            .collect()
    }
}

async fn run_stuck_engine_watchdog(
    config: ConfigHandle,
    client: Client,
    local_recovery: Arc<LocalRecoveryCoordinatorSet>,
    tokens: Arc<StuckWatchdogTokenTracker>,
    shutdown: Arc<ShutdownGate>,
) {
    let mut schedule = WatchdogSchedule::default();
    let mut recovery_tasks = JoinSet::<(String, BTreeMap<String, String>)>::new();
    let mut recovering_profiles = HashSet::new();
    loop {
        collect_finished_watchdog_recoveries(
            &mut recovery_tasks,
            &mut recovering_profiles,
            &local_recovery,
        );
        let Ok(snapshot) = config.snapshot() else {
            let mut shutdown_gate = shutdown.subscribe();
            tokio::select! {
                () = tokio::time::sleep(Duration::from_secs(60)) => {}
                () = shutdown_gate.cancelled() => return,
            }
            continue;
        };
        let profiles = watchdog_upstream_profiles(&snapshot);
        let scheduled_profiles = profiles
            .iter()
            .filter(|profile| profile.stuck_watchdog.enabled)
            .map(|profile| {
                (
                    profile.name.clone(),
                    Duration::from_secs(profile.stuck_watchdog.check_interval_secs),
                )
            })
            .collect::<Vec<_>>();
        // Bound retention every watchdog tick, including idle profiles. Retention
        // must not depend on an active request that might never finish.
        for profile in &profiles {
            if profile.stuck_watchdog.enabled {
                tokens.prune_profile(
                    &profile.name,
                    Duration::from_secs(profile.stuck_watchdog.detection_window_secs),
                );
            }
        }
        let due_profiles = schedule.due_profiles(Instant::now(), &scheduled_profiles);
        for profile in profiles {
            if !due_profiles
                .iter()
                .any(|due_profile| due_profile == &profile.name)
            {
                continue;
            }
            let watchdog = &profile.stuck_watchdog;
            if recovering_profiles.contains(&profile.name) {
                continue;
            }
            let detection_window = Duration::from_secs(watchdog.detection_window_secs);
            if !tokens.has_active_requests(&profile.name)
                || !tokens.has_too_few_output_tokens(
                    &profile.name,
                    detection_window,
                    watchdog.min_output_tokens_in_window,
                )
            {
                continue;
            }

            let policy = LocalRecoveryPolicy::from_config(&profile.local_recovery);
            if !policy.is_configured() {
                continue;
            }
            eprintln!(
                "llm_guard_proxy_stuck_watchdog profile={} event=detected detection_window_secs={} min_output_tokens={}",
                profile.name, watchdog.detection_window_secs, watchdog.min_output_tokens_in_window,
            );
            let coordinator = local_recovery.coordinator_for(&profile.name);
            coordinator
                .watchdog_detections
                .fetch_add(1, Ordering::Relaxed);
            coordinator
                .watchdog_restarts
                .fetch_add(1, Ordering::Relaxed);
            let profile_name = profile.name.clone();
            recovering_profiles.insert(profile_name.clone());
            recovery_tasks.spawn(run_watchdog_recovery(
                profile_name,
                policy,
                Duration::from_secs(profile.restart_queue.restart_timeout_secs),
                coordinator,
                client.clone(),
                profile.base_url.clone(),
                Arc::clone(&shutdown),
            ));
        }
        let mut shutdown_gate = shutdown.subscribe();
        tokio::select! {
            // The loop wakes cheaply, while `schedule` enforces each profile's own
            // cadence instead of collapsing all intervals to a global minimum.
            () = tokio::time::sleep(Duration::from_secs(1)) => {}
            () = shutdown_gate.cancelled() => {
                recovery_tasks.abort_all();
                while recovery_tasks.join_next().await.is_some() {}
                return;
            },
        }
    }
}

fn collect_finished_watchdog_recoveries(
    recovery_tasks: &mut JoinSet<(String, BTreeMap<String, String>)>,
    recovering_profiles: &mut HashSet<String>,
    local_recovery: &LocalRecoveryCoordinatorSet,
) {
    while let Some(result) = recovery_tasks.try_join_next() {
        if let Ok((profile, recovery)) = result {
            recovering_profiles.remove(&profile);
            record_watchdog_recovery_result(
                &local_recovery.coordinator_for(&profile),
                &profile,
                &recovery,
            );
        }
    }
}

async fn run_watchdog_recovery(
    profile_name: String,
    policy: LocalRecoveryPolicy,
    episode_timeout: Duration,
    coordinator: Arc<UpstreamStallRecoveryCoordinator>,
    client: Client,
    base_url: String,
    shutdown: Arc<ShutdownGate>,
) -> (String, BTreeMap<String, String>) {
    let mut shutdown_gate = shutdown.subscribe();
    let recovery = tokio::select! {
        recovery = run_local_recovery_for_profile(
            &policy,
            &coordinator,
            client,
            base_url,
            LocalRecoveryCause::UpstreamStall,
            Some(episode_timeout),
        ) => recovery,
        () = shutdown_gate.cancelled() => BTreeMap::from([(
            String::from("local_recovery_status"),
            String::from("shutdown_cancelled"),
        )]),
    };
    (profile_name, recovery)
}

fn record_watchdog_recovery_result(
    coordinator: &UpstreamStallRecoveryCoordinator,
    profile: &str,
    recovery: &BTreeMap<String, String>,
) {
    match recovery.get("local_recovery_status").map(String::as_str) {
        Some("succeeded") => {
            coordinator
                .watchdog_recovery_successes
                .fetch_add(1, Ordering::Relaxed);
        }
        Some("completion_timeout" | "episode_timeout" | "join_timeout" | "readiness_timeout") => {
            coordinator
                .watchdog_recovery_timeouts
                .fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    eprintln!(
        "llm_guard_proxy_stuck_watchdog profile={profile} event=recovery_finished status={}",
        recovery
            .get("local_recovery_status")
            .map_or("missing", String::as_str),
    );
}

pub(crate) fn spawn_stuck_engine_watchdog(state: &ProxyState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_stuck_engine_watchdog(
        state.config.clone(),
        state.client.clone(),
        Arc::clone(&state.local_recovery),
        Arc::clone(&state.stuck_watchdog_tokens),
        Arc::clone(&state.shutdown),
    ))
}

fn watchdog_upstream_profiles(config: &AppConfig) -> Vec<UpstreamProfileConfig> {
    let mut profiles = Vec::with_capacity(config.upstream_profiles.len().saturating_add(1));
    profiles.push(config.default_upstream_profile());
    profiles.extend(config.upstream_profiles.iter().cloned());
    profiles
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RestartQueueWaitResult {
    NotRecovering,
    Ready,
    Failed,
    TimedOut,
}

async fn wait_for_restart_queue(
    coordinator: &UpstreamStallRecoveryCoordinator,
    queue_deadline: Duration,
) -> RestartQueueWaitResult {
    let deadline = Instant::now() + queue_deadline;
    let mut waited_for_recovery = false;
    loop {
        let notified = coordinator.notify.notified();
        tokio::pin!(notified);
        let _ = notified.as_mut().enable();

        let state = coordinator.state.lock().await;
        let recovery_deadline = state.recovery_deadline.unwrap_or(deadline);
        if !state.running {
            if !waited_for_recovery {
                return RestartQueueWaitResult::NotRecovering;
            }
            return if state
                .last_result
                .as_ref()
                .is_some_and(local_recovery_completed_ready)
            {
                RestartQueueWaitResult::Ready
            } else {
                RestartQueueWaitResult::Failed
            };
        }
        waited_for_recovery = true;
        drop(state);

        let remaining = deadline
            .min(recovery_deadline)
            .saturating_duration_since(Instant::now());
        if remaining.is_zero() || timeout(remaining, notified).await.is_err() {
            return RestartQueueWaitResult::TimedOut;
        }
    }
}

async fn wait_for_profile_restart_queue(
    state: &ProxyState,
    profile: &UpstreamProfileConfig,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<(), ProxyError> {
    let queue = &profile.restart_queue;
    if !queue.enabled {
        return Ok(());
    }

    let queue_deadline = restart_queue_wait_deadline(queue);
    let coordinator = state.local_recovery.coordinator_for(&profile.name);
    // Restart waiters are deliberately accounted in the bounded queue, never in
    // generation in-flight capacity. The caller has released its routing permit
    // before reaching this wait.
    let Some(_restart_queue_permit) = state
        .acquire_restart_queue_permit(profile, &coordinator)
        .await?
    else {
        // This request atomically observed no recovery while registering, so it
        // must not wait for a later episode without a queue admission permit.
        return Ok(());
    };
    match wait_for_restart_queue(&coordinator, queue_deadline).await {
        RestartQueueWaitResult::NotRecovering => Ok(()),
        RestartQueueWaitResult::Ready => {
            request_metadata.insert(
                String::from("restart_queue_outcome"),
                String::from("released_after_recovery"),
            );
            eprintln!(
                "llm_guard_proxy_restart_queue profile={} event=released_after_recovery",
                profile.name
            );
            Ok(())
        }
        result @ (RestartQueueWaitResult::Failed | RestartQueueWaitResult::TimedOut) => {
            let outcome = match result {
                RestartQueueWaitResult::Failed => "recovery_failed",
                RestartQueueWaitResult::TimedOut => "timeout",
                RestartQueueWaitResult::NotRecovering | RestartQueueWaitResult::Ready => {
                    unreachable!("only failed or timed-out restart queues reach this branch")
                }
            };
            request_metadata.insert(String::from("restart_queue_outcome"), String::from(outcome));
            request_metadata.insert(
                String::from("restart_queue_deadline_secs"),
                queue.queue_deadline_secs.to_string(),
            );
            eprintln!(
                "llm_guard_proxy_restart_queue profile={} event={outcome}",
                profile.name
            );
            Err(ProxyError::upstream_unavailable(
                profile.name.clone(),
                u64::try_from(queue_deadline.as_millis()).unwrap_or(u64::MAX),
            )
            .with_request_metadata(request_metadata.clone()))
        }
    }
}

fn restart_queue_wait_deadline(queue: &RestartQueueConfig) -> Duration {
    Duration::from_secs(queue.queue_deadline_secs.min(queue.restart_timeout_secs))
}

#[derive(Clone, Copy, Debug, Default)]
struct WatchdogMetricsSnapshot {
    detections: u64,
    restarts: u64,
    recovery_successes: u64,
    recovery_timeouts: u64,
    restart_queue_depth: u64,
}

#[derive(Debug, Default)]
struct LocalRecoveryCoordinatorSet {
    coordinators: Mutex<HashMap<String, Arc<UpstreamStallRecoveryCoordinator>>>,
}

impl LocalRecoveryCoordinatorSet {
    fn coordinator_for(&self, profile: &str) -> Arc<UpstreamStallRecoveryCoordinator> {
        let mut coordinators = match self.coordinators.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        coordinators
            .entry(profile.to_owned())
            .or_insert_with(|| Arc::new(UpstreamStallRecoveryCoordinator::default()))
            .clone()
    }

    fn watchdog_metrics_snapshot(&self) -> WatchdogMetricsSnapshot {
        let coordinators = match self.coordinators.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        coordinators.values().fold(
            WatchdogMetricsSnapshot::default(),
            |mut metrics, coordinator| {
                metrics.detections = metrics
                    .detections
                    .saturating_add(coordinator.watchdog_detections.load(Ordering::Relaxed));
                metrics.restarts = metrics
                    .restarts
                    .saturating_add(coordinator.watchdog_restarts.load(Ordering::Relaxed));
                metrics.recovery_successes = metrics.recovery_successes.saturating_add(
                    coordinator
                        .watchdog_recovery_successes
                        .load(Ordering::Relaxed),
                );
                metrics.recovery_timeouts = metrics.recovery_timeouts.saturating_add(
                    coordinator
                        .watchdog_recovery_timeouts
                        .load(Ordering::Relaxed),
                );
                metrics.restart_queue_depth = metrics
                    .restart_queue_depth
                    .saturating_add(coordinator.restart_queue_depth.load(Ordering::Relaxed));
                metrics
            },
        )
    }
}

#[cfg(feature = "upstream-hot-restart")]
#[derive(Debug, Default)]
struct HotRestartCoordinator {
    state: AsyncMutex<HotRestartState>,
    notify: Notify,
}

#[cfg(feature = "upstream-hot-restart")]
#[derive(Debug, Default)]
struct HotRestartState {
    in_progress: Option<HotRestartProbeHandle>,
    last_result: Option<HotRestartResult>,
}

#[cfg(feature = "upstream-hot-restart")]
#[derive(Debug)]
struct HotRestartProbeHandle {
    started_at: Instant,
    join_handle: JoinHandle<HotRestartResult>,
}

#[cfg(feature = "upstream-hot-restart")]
#[derive(Clone, Debug, Eq, PartialEq)]
enum HotRestartResult {
    Ready,
    Timeout,
    Error(String),
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

    let request_path = dispatch.uri.path().to_owned();
    let request_id = response_parts.request_id.clone();
    let shutdown = response_parts.shutdown_subscription();
    let observer = response_parts.into_observer();
    let response_body = ObservedUpstreamBody::new(
        upstream_response.bytes_stream(),
        observer,
        in_flight_permit,
        shutdown,
    );
    let response = downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    );
    Ok(validate_non_stream_chat_completion_response(
        response,
        &request_path,
        &request_id,
        dispatch.malformed_response_counter,
    )
    .await)
}

async fn read_body_bytes(body: Body, max_request_body_bytes: usize) -> Result<Bytes, ProxyError> {
    to_bytes(body, max_request_body_bytes)
        .await
        .map_err(|error| ProxyError::request_body(error.to_string()))
}

async fn read_body_bytes_until_shutdown(
    body: Body,
    max_request_body_bytes: usize,
    mut shutdown: ShutdownSubscription,
) -> Result<Bytes, ProxyError> {
    tokio::select! {
        biased;
        () = shutdown.cancelled() => Err(ProxyError::server_shutdown()),
        result = read_body_bytes(body, max_request_body_bytes) => result,
    }
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

async fn read_upstream_body_bytes_until_shutdown(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
    mut shutdown: ShutdownSubscription,
) -> Result<Bytes, ProxyError> {
    tokio::select! {
        biased;
        () = shutdown.cancelled() => Err(ProxyError::server_shutdown()),
        result = read_upstream_body_bytes(stream) => result,
    }
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
    let send = client
        .request(method, upstream_url)
        .headers(headers)
        .body(body)
        .timeout(timeout)
        .send();
    tokio::time::timeout(timeout, send).await.map_or_else(
        |_| {
            Err(ProxyError::UpstreamTransport {
                failure: ReqwestFailureKind::Timeout,
                observability: None,
            })
        },
        |result| {
            result.map_err(|source| {
                let failure = ReqwestFailureKind::from_error(&source);
                ProxyError::UpstreamTransport {
                    failure,
                    observability: None,
                }
            })
        },
    )
}

async fn send_upstream_request_until_shutdown(
    client: &Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: &HeaderMap,
    body: Bytes,
    timeout: Duration,
    mut shutdown: ShutdownSubscription,
) -> Result<reqwest::Response, ProxyError> {
    tokio::select! {
        biased;
        () = shutdown.cancelled() => Err(ProxyError::server_shutdown()),
        result = send_upstream_request(client, method, upstream_url, downstream_headers, body, timeout) => result,
    }
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
    persistence_tasks: Arc<PersistenceTasks>,
    request_id: RequestId,
    started_at_unix_ms: u64,
    attempt_id: AttemptId,
    attempt_number: u32,
    attempt_max_attempts: u32,
    attempt_started_at_unix_ms: u64,
    upstream_mode: UpstreamMode,
    model_id: Option<String>,
    input_fingerprint: Option<String>,
    upstream_status: reqwest::StatusCode,
    upstream_headers: HeaderMap,
    request_metadata: BTreeMap<String, String>,
    attempt_request_metadata: BTreeMap<String, String>,
    completed_attempt_records: Vec<AttemptRecord>,
    shutdown: Arc<ShutdownGate>,
    stuck_watchdog_request: Option<StuckWatchdogRequest>,
}

impl ForwardedResponseParts {
    fn shutdown_subscription(&self) -> ShutdownSubscription {
        self.shutdown.subscribe()
    }

    fn into_observer(self) -> ForwardedBodyObserver {
        let downstream_mode = downstream_mode_from_headers(&self.upstream_headers);
        let downstream_headers = self.upstream_headers.clone();
        self.into_observer_with(
            downstream_mode,
            downstream_headers,
            BTreeMap::new(),
            BTreeMap::new(),
            RawPayloads::default(),
        )
    }

    fn into_observer_with(
        self,
        downstream_mode: DownstreamMode,
        downstream_headers: HeaderMap,
        attempt_response_metadata: BTreeMap<String, String>,
        extra_response_metadata: BTreeMap<String, String>,
        raw_payloads: RawPayloads,
    ) -> ForwardedBodyObserver {
        let final_attempt = FinalAttemptContext {
            attempt_id: self.attempt_id,
            attempt_number: self.attempt_number,
            attempt_max_attempts: self.attempt_max_attempts,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            upstream_status: self.upstream_status,
            upstream_headers: self.upstream_headers.clone(),
            request_metadata: self.attempt_request_metadata,
            extra_response_metadata: attempt_response_metadata,
            raw_payloads: raw_payloads.clone(),
            response_body: Bytes::new(),
            sse_body: Bytes::new(),
        };
        ForwardedBodyObserver {
            config: self.config,
            downstream_mode,
            store: self.store,
            evidence_store: self.evidence_store,
            persistence_tasks: self.persistence_tasks,
            shadow_evidence: ShadowEvidenceState::default(),
            paired_shadow_runtime: None,
            request_id: self.request_id,
            started_at_unix_ms: self.started_at_unix_ms,
            upstream_mode: self.upstream_mode,
            model_id: self.model_id,
            input_fingerprint: self.input_fingerprint,
            downstream_status: self.upstream_status,
            downstream_headers,
            request_metadata: self.request_metadata,
            extra_response_metadata,
            raw_payloads,
            completed_attempt_records: self.completed_attempt_records,
            stuck_watchdog_request: self.stuck_watchdog_request,
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
            attempt_number: self.attempt_number,
            request_id: self.request_id,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            finished_at_unix_ms,
            error_type: error.error_type(),
            error_reason: &error_reason,
            request_metadata: self.attempt_request_metadata,
            extra_response_metadata,
        });
        error
            .with_observability(self.request_metadata, attempt_record)
            .with_completed_attempt_records(self.completed_attempt_records)
    }

    fn into_response_process_error(
        self,
        error: ProxyError,
        mut extra_response_metadata: BTreeMap<String, String>,
    ) -> ProxyError {
        let finished_at_unix_ms = unix_time_millis();
        let error_reason = error.to_string();
        extra_response_metadata
            .insert(String::from("response_process_error"), String::from("true"));
        let mut attempt_record = failed_attempt_record(FailedAttemptRecordInput {
            attempt_id: self.attempt_id,
            attempt_number: self.attempt_number,
            request_id: self.request_id,
            started_at_unix_ms: self.attempt_started_at_unix_ms,
            finished_at_unix_ms,
            error_type: error.error_type(),
            error_reason: &error_reason,
            request_metadata: self.attempt_request_metadata,
            extra_response_metadata,
        });
        // Response body was received and status known; distinguish from transport failures.
        attempt_record.http_status = Some(self.upstream_status.as_u16());
        attempt_record.upstream_mode = self.upstream_mode;
        attempt_record.response_metadata.insert(
            String::from("upstream_response_received"),
            String::from("true"),
        );
        attempt_record.response_metadata.insert(
            String::from("http_status_success"),
            if self.upstream_status.is_success() {
                String::from("true")
            } else {
                String::from("false")
            },
        );
        copy_selected_header_metadata(
            &mut attempt_record.response_metadata,
            &self.upstream_headers,
            "response",
        );
        error
            .with_observability(self.request_metadata, attempt_record)
            .with_completed_attempt_records(self.completed_attempt_records)
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
    let body = match read_upstream_body_bytes_until_shutdown(
        upstream_response.bytes_stream(),
        response_parts.shutdown_subscription(),
    )
    .await
    {
        Ok(body) => body,
        Err(error) => return Err(response_parts.into_body_read_error(error)),
    };
    let body = filter_models_body_for_listener(config, listener, body);
    let body = model_metadata::enrich_models_body(config, metadata_config, body);
    let shutdown = response_parts.shutdown_subscription();
    let observer = response_parts.into_observer();
    let response_body = ObservedBufferedBody::new(body, observer, in_flight_permit, shutdown);

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

fn model_discovery_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut safe_headers = HeaderMap::new();
    if let Some(accept) = headers.get(ACCEPT) {
        safe_headers.insert(ACCEPT, accept.clone());
    }
    safe_headers
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
    upstream_headers: HeaderMap,
    original_downstream_headers: HeaderMap,
    upstream_body: Bytes,
    downstream_body: Bytes,
    forward_uri: Uri,
    transformed_request_headers: bool,
    terminal_endpoint: UpstreamEndpointConfig,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
    endpoint_retry_order: Vec<String>,
    chat_kind: ShieldedChatKind,
    upstream_timeout: Duration,
    config: ConfigHandle,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    persistence_tasks: Arc<PersistenceTasks>,
    request_id: RequestId,
    started_at_unix_ms: u64,
    model_id: Option<String>,
    stuck_watchdog_request: Option<StuckWatchdogRequest>,
    request_metadata: BTreeMap<String, String>,
    listener: ListenerConfig,
    upstream_profile: UpstreamProfileConfig,
    #[cfg(feature = "guard")]
    caller_profile_name: String,
    #[cfg(feature = "guard")]
    caller_profile: ProfileConfig,
    #[cfg(feature = "guard")]
    workflow_execution_requests: Arc<InFlightLimiter>,
    route_reason: UpstreamRouteReason,
    liveness: ShieldedLivenessSelection,
    thinking_metadata: BTreeMap<String, String>,
    loop_context: shielded_chat::LoopInspectionContext,
    retry_policy: ShieldedRetryPolicy,
    request_deadline: ShieldedRequestDeadline,
    upstream_stall_policy: UpstreamStallPolicy,
    upstream_stall_recovery: Arc<UpstreamStallRecoveryCoordinator>,
    upstream_health: Arc<UpstreamHealthRegistry>,
    local_recovery_policy: LocalRecoveryPolicy,
    local_recovery: Arc<UpstreamStallRecoveryCoordinator>,
    local_recovery_attempts: Arc<AtomicU64>,
    local_recovery_deadline_replay_permits: Arc<AtomicU64>,
    #[cfg(feature = "upstream-hot-restart")]
    hot_restart_recovery: Arc<HotRestartCoordinator>,
    shadow_attempts: Arc<InFlightLimiter>,
    shutdown: Arc<ShutdownGate>,
    downstream_drop_signal: DownstreamDropSignal,
    shadow_evidence: ShadowEvidenceState,
    malformed_response_counter: Arc<AtomicU64>,
    upstream_failure_counters: Arc<UpstreamFailureCounters>,
    #[cfg(test)]
    shielded_heartbeat_ticks: Arc<AtomicU64>,
}

#[derive(Clone, Debug, Default)]
struct DownstreamDropSignal {
    dropped: Arc<AtomicBool>,
}

impl DownstreamDropSignal {
    fn mark_dropped(&self) {
        self.dropped.store(true, Ordering::SeqCst);
    }

    fn is_dropped(&self) -> bool {
        self.dropped.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Copy, Debug)]
struct ShieldedRequestDeadline {
    started_at: Instant,
    max_duration: Duration,
}

impl ShieldedRequestDeadline {
    fn new(max_duration: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            max_duration,
        }
    }

    fn remaining(self) -> Option<Duration> {
        self.max_duration.checked_sub(self.started_at.elapsed())
    }

    fn is_exhausted(self) -> bool {
        self.remaining().is_none_or(|remaining| remaining.is_zero())
    }
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
    completed_endpoint_attempt_records: Vec<AttemptRecord>,
    terminal_endpoint: UpstreamEndpointConfig,
    terminal_endpoint_protocol: UpstreamEndpointProtocol,
    endpoint_retry_order: Vec<String>,
    ignores_request_deadline: bool,
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
    request_deadline: ShieldedRequestDeadline,
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
    downstream_status: StatusCode,
    retry_after_secs: Option<u32>,
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
    #[cfg(feature = "upstream-hot-restart")]
    transport_failure: Option<ReqwestFailureKind>,
    error_type: &'static str,
    error_message: String,
    retry_cause: Option<ShieldedRetryCause>,
    abort_reason: Option<String>,
    request_metadata: BTreeMap<String, String>,
    response_metadata: BTreeMap<String, String>,
    raw_payloads: RawPayloads,
    upstream_body: Bytes,
    completed_endpoint_attempt_records: Vec<AttemptRecord>,
}

#[derive(Clone, Debug)]
struct CotSalvageContext {
    policy: LoopFailurePolicy,
    source_attempt_id: AttemptId,
    source_attempt_number: u32,
    source_attempt_duration_ms: u64,
    reasoning_prefix: String,
}

impl CotSalvageContext {
    fn prefix_bytes(&self) -> usize {
        self.reasoning_prefix.len()
    }
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
        } => Ok(shielded_liveness_stream_response(
            &runtime,
            started,
            prior_attempt_records,
            in_flight_permit,
        )),
        ShieldedBeginOutcome::Failed(failure) => Ok(shielded_retry_error_response(
            &runtime,
            failure,
            in_flight_permit,
        )),
        ShieldedBeginOutcome::TerminalForward(terminal) => Ok(
            shielded_retry_terminal_forward_response(&runtime, terminal, in_flight_permit).await,
        ),
    }
}

fn shielded_liveness_stream_response(
    runtime: &ShieldedRetryRuntime,
    started: ShieldedStartedAttempt,
    prior_attempt_records: Vec<AttemptRecord>,
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
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
        current_attempt: Some(started.info.clone().into_final_context(
            extra_metadata.clone(),
            RawPayloads::default(),
            Bytes::new(),
            Bytes::new(),
        )),
    }));
    let observer = shielded_retry_observer(
        runtime,
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
    let liveness_settings = ShieldedLivenessBodySettings {
        mode: runtime.liveness.mode,
        accepted_response_mode: if runtime.chat_kind == ShieldedChatKind::Stream {
            ShieldedAcceptedResponseMode::OpenAiSse
        } else {
            ShieldedAcceptedResponseMode::JsonCompletion
        },
        interval_secs: runtime.liveness.heartbeat_interval_secs,
        upstream_failure_counters: runtime.upstream_failure_counters.clone(),
        #[cfg(test)]
        shielded_heartbeat_ticks: runtime.shielded_heartbeat_ticks.clone(),
    };
    let response_body = ShieldedLivenessBody::new(
        aggregate,
        &liveness_settings,
        observer,
        in_flight_permit,
        runtime.shutdown.subscribe(),
        runtime.downstream_drop_signal.clone(),
    );
    response_with_headers(
        upstream_status,
        response_headers,
        Body::from_stream(response_body),
    )
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
            shielded_retry_direct_relay_response(runtime, outcome, in_flight_permit).await
        }
        ShieldedRunOutcome::Failed(failure) => {
            shielded_retry_error_response(runtime, failure, in_flight_permit)
        }
        ShieldedRunOutcome::TerminalForward(terminal) => {
            shielded_retry_terminal_forward_response(runtime, terminal, in_flight_permit).await
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
                downstream_status: StatusCode::BAD_GATEWAY,
                retry_after_secs: None,
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

async fn shielded_start_failure_step(
    runtime: &ShieldedRetryRuntime,
    mut failure: ShieldedAttemptFailure,
    attempt_records: &mut Vec<AttemptRecord>,
) -> ShieldedStartFailureStep {
    attempt_records.append(&mut failure.completed_endpoint_attempt_records);
    let next_retry_cause = failure.retry_cause;
    let can_retry = should_retry_after_shielded_failure(runtime, &failure);
    let (failure, can_retry) = {
        let mut failure = failure;
        let mut can_retry = can_retry;
        let local_recovery_gate =
            local_recovery_gate_for_attempt_failure(runtime, can_retry, &failure).await;
        if local_recovery_gate.applied {
            can_retry = local_recovery_gate.permits_deadline_replay
                || (can_retry && local_recovery_gate.permits_retry);
            failure
                .response_metadata
                .extend(local_recovery_gate.metadata);
        }
        (failure, can_retry)
    };
    #[cfg(feature = "upstream-hot-restart")]
    let (failure, can_retry, terminal) = {
        let mut failure = failure;
        let mut can_retry = can_retry;
        let mut terminal = None;
        if !failure
            .response_metadata
            .contains_key("local_recovery_status")
        {
            let hot_restart_gate =
                hot_restart_gate_for_attempt_failure(runtime, can_retry, &failure).await;
            can_retry = can_retry && hot_restart_gate.permits_retry;
            failure.response_metadata.extend(hot_restart_gate.metadata);
            if let Some(hot_restart_terminal) = hot_restart_gate.terminal {
                failure.abort_reason = Some(hot_restart_terminal.abort_reason.to_owned());
                terminal = Some(hot_restart_terminal);
            }
        }
        (failure, can_retry, terminal)
    };
    attempt_records.push(attempt_failure_record(
        &failure,
        shielded_failed_attempt_status(can_retry, &failure),
        if can_retry { next_retry_cause } else { None },
        &runtime.retry_policy,
    ));
    if can_retry {
        return ShieldedStartFailureStep::Retry {
            attempt_number: failure.attempt_number.saturating_add(1),
            retry_cause: next_retry_cause,
        };
    }
    let outcome = shielded_failure_outcome(
        failure,
        std::mem::take(attempt_records),
        &runtime.retry_policy,
    );
    #[cfg(feature = "upstream-hot-restart")]
    let outcome = {
        let mut outcome = outcome;
        if let Some(hot_restart_terminal) = terminal {
            outcome.downstream_status = hot_restart_terminal.status;
            outcome.retry_after_secs = hot_restart_terminal.retry_after_secs;
        }
        outcome
    };
    ShieldedStartFailureStep::Failed(outcome)
}

async fn shielded_started_attempt_step(
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
            return shielded_retryable_status_step(runtime, &started.info, cause, attempt_records)
                .await;
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

async fn shielded_retryable_status_step(
    runtime: &ShieldedRetryRuntime,
    info: &ShieldedAttemptInfo,
    cause: ShieldedRetryCause,
    attempt_records: &mut Vec<AttemptRecord>,
) -> ShieldedAttemptStep {
    let failure = status_failure(
        info,
        cause,
        "retryable upstream status before shielded stream",
    );
    let can_retry = should_retry_after_shielded_failure(runtime, &failure);
    let (failure, can_retry) = {
        let mut failure = failure;
        let mut can_retry = can_retry;
        let local_recovery_gate =
            local_recovery_gate_for_status(runtime, can_retry, info.upstream_status).await;
        if local_recovery_gate.applied {
            can_retry = local_recovery_gate.permits_deadline_replay
                || (can_retry && local_recovery_gate.permits_retry);
            failure
                .response_metadata
                .extend(local_recovery_gate.metadata);
        }
        (failure, can_retry)
    };
    #[cfg(feature = "upstream-hot-restart")]
    let (failure, can_retry, terminal) = {
        let mut failure = failure;
        let mut can_retry = can_retry;
        let mut terminal = None;
        if !failure
            .response_metadata
            .contains_key("local_recovery_status")
        {
            let hot_restart_gate =
                hot_restart_gate_for_status(runtime, info.upstream_status, can_retry).await;
            can_retry = can_retry && hot_restart_gate.permits_retry;
            failure.response_metadata.extend(hot_restart_gate.metadata);
            if let Some(hot_restart_terminal) = hot_restart_gate.terminal {
                failure.abort_reason = Some(hot_restart_terminal.abort_reason.to_owned());
                terminal = Some(hot_restart_terminal);
            }
        }
        (failure, can_retry, terminal)
    };
    if can_retry {
        attempt_records.push(attempt_failure_record(
            &failure,
            AttemptStatus::Retried,
            Some(cause),
            &runtime.retry_policy,
        ));
        return ShieldedAttemptStep::Retry {
            attempt_number: info.attempt_number.saturating_add(1),
            retry_cause: Some(cause),
        };
    }
    attempt_records.push(attempt_failure_record(
        &failure,
        AttemptStatus::Failed,
        None,
        &runtime.retry_policy,
    ));
    let outcome = shielded_failure_outcome(
        failure,
        std::mem::take(attempt_records),
        &runtime.retry_policy,
    );
    #[cfg(feature = "upstream-hot-restart")]
    let outcome = {
        let mut outcome = outcome;
        if let Some(hot_restart_terminal) = terminal {
            outcome.downstream_status = hot_restart_terminal.status;
            outcome.retry_after_secs = hot_restart_terminal.retry_after_secs;
        }
        outcome
    };
    ShieldedAttemptStep::Failed(outcome)
}

async fn begin_shielded_retry(runtime: &ShieldedRetryRuntime) -> ShieldedBeginOutcome {
    let mut attempt_number = 1;
    let mut retry_cause = None;
    let mut attempt_records = Vec::new();
    loop {
        let ignores_request_deadline = consume_local_recovery_deadline_replay_permit(runtime);
        if runtime.request_deadline.is_exhausted() && !ignores_request_deadline {
            return ShieldedBeginOutcome::Failed(request_deadline_exhausted_outcome(
                runtime,
                attempt_number,
                std::mem::take(&mut attempt_records),
            ));
        }
        let mut started = match start_shielded_attempt(
            runtime,
            attempt_number,
            retry_cause,
            None,
            ignores_request_deadline,
        )
        .await
        {
            Ok(started) => started,
            Err(failure) => {
                match shielded_start_failure_step(runtime, failure, &mut attempt_records).await {
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
        attempt_records.append(&mut started.completed_endpoint_attempt_records);

        match shielded_started_attempt_step(runtime, started, &mut attempt_records, true).await {
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
    let stuck_watchdog_request = runtime.stuck_watchdog_request.clone();
    let upstream_stream = started.response.bytes_stream().inspect(move |chunk| {
        if let (Some(request), Ok(chunk)) = (&stuck_watchdog_request, chunk) {
            // Shielded calls aggregate upstream SSE internally, so this is the
            // only place their sustained model output reaches the watchdog.
            request.record_emitted_chunk(chunk);
        }
    });
    let aggregate = shielded_chat::aggregate_stream(
        upstream_stream,
        started.info.started_at_unix_ms,
        &request_id,
        request_model_id.as_deref(),
        runtime.loop_context.clone(),
        runtime.upstream_stall_policy.stream_timeouts(),
    );
    let remaining_deadline = if started.ignores_request_deadline {
        runtime.upstream_timeout
    } else {
        let Some(remaining_deadline) = runtime.request_deadline.remaining() else {
            return Err(request_deadline_shielded_attempt_failure(&started.info));
        };
        remaining_deadline
    };
    let mut shutdown = runtime.shutdown.subscribe();
    let result = tokio::select! {
        biased;
        () = shutdown.cancelled() => return Err(shutdown_shielded_attempt_failure(&started.info)),
        () = tokio::time::sleep(remaining_deadline) => return Err(request_deadline_shielded_attempt_failure(&started.info)),
        result = aggregate => result,
    };
    match result {
        Ok(aggregated) => Ok(ShieldedAggregatedAttempt {
            final_attempt: started.info.into_final_context(
                aggregated.response_metadata.clone(),
                aggregated.raw_payloads.clone(),
                aggregated.body.clone(),
                aggregated.sse_body.clone(),
            ),
            body: aggregated.body,
            sse_body: aggregated.sse_body,
            response_metadata: aggregated.response_metadata,
        }),
        Err(error) => Err(aggregation_failure(&started.info, &error)),
    }
}

#[allow(clippy::too_many_lines)]
async fn run_shielded_attempts(
    mut runtime: ShieldedRetryRuntime,
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
    let mut cot_salvage = None;
    let mut cot_salvage_attempted = false;
    loop {
        let mut started = if let Some(started) = current_attempt.take() {
            started
        } else {
            let ignores_request_deadline = consume_local_recovery_deadline_replay_permit(&runtime);
            if runtime.request_deadline.is_exhausted() && !ignores_request_deadline {
                return ShieldedRunOutcome::Failed(request_deadline_exhausted_outcome(
                    &runtime,
                    attempt_number,
                    attempt_records,
                ));
            }
            match start_shielded_attempt(
                &runtime,
                attempt_number,
                retry_cause,
                cot_salvage.as_ref(),
                ignores_request_deadline,
            )
            .await
            {
                Ok(started) => started,
                Err(failure) => {
                    match shielded_start_failure_step(&runtime, failure, &mut attempt_records).await
                    {
                        ShieldedStartFailureStep::Retry {
                            attempt_number: next_attempt_number,
                            retry_cause: next_retry_cause,
                        } => {
                            attempt_number = next_attempt_number;
                            retry_cause = next_retry_cause;
                            cot_salvage = None;
                            continue;
                        }
                        ShieldedStartFailureStep::Failed(outcome) => {
                            return ShieldedRunOutcome::Failed(outcome);
                        }
                    }
                }
            }
        };
        update_shielded_retry_endpoint(&mut runtime, &started);
        attempt_records.append(&mut started.completed_endpoint_attempt_records);

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
        )
        .await
        {
            ShieldedAttemptStep::Aggregatable(started) => started,
            ShieldedAttemptStep::Retry {
                attempt_number: next_attempt_number,
                retry_cause: next_retry_cause,
            } => {
                update_shielded_attempt_progress(attempt_progress.as_ref(), &attempt_records, None);
                attempt_number = next_attempt_number;
                retry_cause = next_retry_cause;
                cot_salvage = None;
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
                runtime.request_deadline,
            ));
        }

        match aggregate_shielded_attempt(&runtime, started).await {
            Ok(aggregated) => {
                #[cfg(feature = "guard")]
                let mut aggregated = aggregated;
                #[cfg(feature = "guard")]
                apply_post_response_guard(&runtime, &mut aggregated).await;
                return shielded_accepted_outcome(aggregated, attempt_records);
            }
            Err(mut failure) => {
                if runtime.request_deadline.is_exhausted() {
                    mark_request_deadline_attempt_failure(&mut failure);
                }
                let next_retry_cause = failure.retry_cause;
                let mut can_retry = should_retry_after_shielded_failure(&runtime, &failure);
                let local_recovery_gate =
                    local_recovery_gate_for_attempt_failure(&runtime, can_retry, &failure).await;
                if local_recovery_gate.applied {
                    can_retry = local_recovery_gate.permits_deadline_replay
                        || (can_retry && local_recovery_gate.permits_retry);
                    failure
                        .response_metadata
                        .extend(local_recovery_gate.metadata);
                }
                let next_cot_salvage =
                    if can_retry && cot_salvage.is_none() && !cot_salvage_attempted {
                        cot_salvage_context_for_failure(&runtime, &failure)
                    } else {
                        None
                    };
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
                    shielded_failed_attempt_status(can_retry, &failure),
                    retry_cause_for_attempt_record(can_retry, next_retry_cause),
                    &runtime.retry_policy,
                );
                maybe_schedule_shadow_continuation(&runtime, &failure, &attempt_record);
                attempt_records.push(attempt_record);
                update_shielded_attempt_progress(attempt_progress.as_ref(), &attempt_records, None);
                if can_retry {
                    attempt_number = failure.attempt_number.saturating_add(1);
                    retry_cause = next_retry_cause;
                    if next_cot_salvage.is_some() {
                        cot_salvage_attempted = true;
                    }
                    cot_salvage = next_cot_salvage;
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
    request_deadline: ShieldedRequestDeadline,
) -> ShieldedDirectRelayOutcome {
    ShieldedDirectRelayOutcome {
        started,
        prior_attempt_records: attempt_records.to_vec(),
        response_metadata: no_thinking_direct_relay_metadata(),
        request_deadline,
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
            .get("cot_salvage_used")
            .is_none_or(|used| used != "true")
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
            String::from("shielded_direct_streaming_relay_deadline_bound"),
            String::from("true"),
        ),
        (
            String::from("shielded_loop_inspection_skipped"),
            String::from("no_thinking_direct_streaming_relay"),
        ),
    ])
}

fn shielded_failed_attempt_status(
    can_retry: bool,
    failure: &ShieldedAttemptFailure,
) -> AttemptStatus {
    if can_retry {
        AttemptStatus::Retried
    } else if is_server_shutdown_failure(failure) {
        AttemptStatus::Aborted
    } else {
        AttemptStatus::Failed
    }
}

fn is_server_shutdown_failure(failure: &ShieldedAttemptFailure) -> bool {
    failure.abort_reason.as_deref() == Some(SERVER_SHUTDOWN_ABORT_REASON)
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
    !is_server_shutdown_failure(failure)
        && failure.retry_cause.is_some()
        && !runtime.downstream_drop_signal.is_dropped()
        && runtime
            .retry_policy
            .allows_retry_after(failure.attempt_number)
}

#[cfg(feature = "upstream-hot-restart")]
struct HotRestartGate {
    metadata: BTreeMap<String, String>,
    permits_retry: bool,
    terminal: Option<HotRestartTerminal>,
}

#[cfg(feature = "upstream-hot-restart")]
struct HotRestartTerminal {
    status: StatusCode,
    retry_after_secs: Option<u32>,
    abort_reason: &'static str,
}

#[cfg(feature = "upstream-hot-restart")]
async fn hot_restart_gate_for_attempt_failure(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
    failure: &ShieldedAttemptFailure,
) -> HotRestartGate {
    let applies = can_retry
        && runtime.upstream_profile.hot_restart.enabled
        && matches!(failure.transport_failure, Some(ReqwestFailureKind::Connect));
    hot_restart_gate(runtime, applies).await
}

#[cfg(feature = "upstream-hot-restart")]
async fn hot_restart_gate_for_status(
    runtime: &ShieldedRetryRuntime,
    status: reqwest::StatusCode,
    can_retry: bool,
) -> HotRestartGate {
    let applies =
        can_retry && runtime.upstream_profile.hot_restart.enabled && is_hot_restart_status(status);
    hot_restart_gate(runtime, applies).await
}

#[cfg(feature = "upstream-hot-restart")]
async fn hot_restart_gate(runtime: &ShieldedRetryRuntime, applies: bool) -> HotRestartGate {
    if !applies {
        return HotRestartGate {
            metadata: BTreeMap::new(),
            permits_retry: true,
            terminal: None,
        };
    }
    let mut metadata = hot_restart_recovery_metadata(true);
    let Some(remaining_deadline) = runtime.request_deadline.remaining() else {
        return request_deadline_hot_restart_gate(metadata);
    };
    let result = match timeout(remaining_deadline, wait_for_hot_restart_recovery(runtime)).await {
        Ok(result) => result,
        Err(_elapsed) => return request_deadline_hot_restart_gate(metadata),
    };
    match result {
        HotRestartResult::Ready => {
            metadata.insert(
                String::from("hot_restart_recovery_status"),
                String::from("ready"),
            );
            HotRestartGate {
                metadata,
                permits_retry: true,
                terminal: None,
            }
        }
        HotRestartResult::Timeout => {
            metadata.insert(
                String::from("hot_restart_recovery_status"),
                String::from("timeout"),
            );
            HotRestartGate {
                metadata,
                permits_retry: false,
                terminal: Some(HotRestartTerminal {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    retry_after_secs: Some(hot_restart_retry_after_secs(
                        &runtime.upstream_profile.hot_restart,
                    )),
                    abort_reason: "hot_restart_timeout",
                }),
            }
        }
        HotRestartResult::Error(error) => {
            metadata.insert(
                String::from("hot_restart_recovery_status"),
                String::from("error"),
            );
            metadata.insert(String::from("hot_restart_recovery_error"), error);
            HotRestartGate {
                metadata,
                permits_retry: false,
                terminal: Some(HotRestartTerminal {
                    status: StatusCode::BAD_GATEWAY,
                    retry_after_secs: None,
                    abort_reason: "hot_restart_error",
                }),
            }
        }
    }
}

#[cfg(feature = "upstream-hot-restart")]
fn request_deadline_hot_restart_gate(mut metadata: BTreeMap<String, String>) -> HotRestartGate {
    metadata.insert(
        String::from("hot_restart_recovery_status"),
        String::from(REQUEST_DEADLINE_ABORT_REASON),
    );
    metadata.insert(
        String::from("request_deadline_exhausted"),
        String::from("true"),
    );
    metadata.insert(
        String::from("shielded_terminal_reason"),
        String::from(REQUEST_DEADLINE_ABORT_REASON),
    );
    HotRestartGate {
        metadata,
        permits_retry: false,
        terminal: Some(HotRestartTerminal {
            status: StatusCode::BAD_GATEWAY,
            retry_after_secs: None,
            abort_reason: REQUEST_DEADLINE_ABORT_REASON,
        }),
    }
}

#[cfg(feature = "upstream-hot-restart")]
const fn is_hot_restart_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 502..=504)
}

#[cfg(feature = "upstream-hot-restart")]
fn hot_restart_retry_after_secs(config: &HotRestartConfig) -> u32 {
    u32::try_from(config.probe_interval_secs).unwrap_or(u32::MAX)
}

#[cfg(feature = "upstream-hot-restart")]
async fn wait_for_hot_restart_recovery(runtime: &ShieldedRetryRuntime) -> HotRestartResult {
    let coordinator = &runtime.hot_restart_recovery;
    let mut state = coordinator.state.lock().await;
    if state.in_progress.is_none() {
        let client = runtime.client.clone();
        let base_url = runtime.upstream_profile.base_url.clone();
        let config = runtime.upstream_profile.hot_restart.clone();
        let model_id = hot_restart_probe_model(runtime);
        let join_handle =
            tokio::spawn(
                async move { run_hot_restart_probe(client, base_url, config, model_id).await },
            );
        state.in_progress = Some(HotRestartProbeHandle {
            started_at: Instant::now(),
            join_handle,
        });
    }
    drop(state);
    wait_for_hot_restart_probe_result(coordinator).await
}

#[cfg(feature = "upstream-hot-restart")]
fn hot_restart_probe_model(runtime: &ShieldedRetryRuntime) -> Option<String> {
    runtime.model_id.clone().or_else(|| {
        runtime
            .upstream_profile
            .match_models
            .first()
            .map(ToOwned::to_owned)
    })
}

#[cfg(feature = "upstream-hot-restart")]
async fn wait_for_hot_restart_probe_result(
    coordinator: &Arc<HotRestartCoordinator>,
) -> HotRestartResult {
    loop {
        let notified = coordinator.notify.notified();
        tokio::pin!(notified);
        let _ = notified.as_mut().enable();

        let mut state = coordinator.state.lock().await;
        if state.in_progress.is_none() {
            return state.last_result.clone().unwrap_or_else(|| {
                HotRestartResult::Error(String::from("hot-restart probe completed without result"))
            });
        }
        if let Some(handle) = state.in_progress.as_mut()
            && handle.join_handle.is_finished()
        {
            let Some(handle) = state.in_progress.take() else {
                continue;
            };
            drop(state);
            let _probe_elapsed = handle.started_at.elapsed();
            let result = match handle.join_handle.await {
                Ok(result) => result,
                Err(error) => HotRestartResult::Error(format!("probe task failed: {error}")),
            };
            finish_hot_restart_recovery(coordinator, result.clone()).await;
            return result;
        }
        drop(state);
        let _ = timeout(Duration::from_millis(100), notified).await;
    }
}

#[cfg(feature = "upstream-hot-restart")]
async fn finish_hot_restart_recovery(
    coordinator: &HotRestartCoordinator,
    result: HotRestartResult,
) {
    let mut state = coordinator.state.lock().await;
    state.in_progress = None;
    state.last_result = Some(result);
    drop(state);
    coordinator.notify.notify_waiters();
}

#[cfg(feature = "upstream-hot-restart")]
async fn run_hot_restart_probe(
    client: Client,
    base_url: String,
    config: HotRestartConfig,
    model_id: Option<String>,
) -> HotRestartResult {
    let deadline = Instant::now() + Duration::from_secs(config.probe_timeout_secs);
    let interval = Duration::from_secs(config.probe_interval_secs);
    loop {
        if Instant::now() >= deadline {
            return HotRestartResult::Timeout;
        }
        match send_hot_restart_probe(&client, &base_url, &config, model_id.as_deref()).await {
            Ok(true) => return HotRestartResult::Ready,
            Ok(false) => {}
            Err(error) => {
                if Instant::now() >= deadline {
                    return HotRestartResult::Error(error);
                }
            }
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return HotRestartResult::Timeout;
        }
        tokio::time::sleep(std::cmp::min(interval, remaining)).await;
    }
}

#[cfg(feature = "upstream-hot-restart")]
async fn send_hot_restart_probe(
    client: &Client,
    base_url: &str,
    config: &HotRestartConfig,
    model_id: Option<&str>,
) -> Result<bool, String> {
    let uri = Uri::from_static("/v1/chat/completions");
    let upstream_url = build_upstream_url(base_url, &uri).map_err(|error| error.to_string())?;
    let body = hot_restart_probe_body(config, model_id);
    let response = client
        .post(upstream_url)
        .header(
            HeaderName::from_static("x-llm-guard-proxy-probe"),
            HeaderValue::from_static("hot-restart"),
        )
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .body(body.to_string())
        .timeout(Duration::from_secs(config.probe_interval_secs))
        .send()
        .await
        .map_err(|error| sanitized_reqwest_error(&error))?;
    if !response.status().is_success() {
        return Ok(false);
    }
    let body = response
        .bytes()
        .await
        .map_err(|error| sanitized_reqwest_error(&error))?;
    let value = serde_json::from_slice::<serde_json::Value>(&body)
        .map_err(|error| format!("probe response JSON decode failed: {error}"))?;
    Ok(is_valid_hot_restart_completion(&value))
}

#[cfg(feature = "upstream-hot-restart")]
fn hot_restart_probe_body(config: &HotRestartConfig, model_id: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::Map::from_iter([
        (String::from("messages"), config.probe_messages.clone()),
        (
            String::from("max_tokens"),
            serde_json::Value::from(config.probe_max_tokens),
        ),
        (String::from("stream"), serde_json::Value::Bool(false)),
    ]);
    if let Some(model_id) = model_id {
        body.insert(
            String::from("model"),
            serde_json::Value::String(model_id.to_owned()),
        );
    }
    if let Some(kwargs) = &config.probe_chat_template_kwargs {
        body.insert(String::from("chat_template_kwargs"), kwargs.clone());
    }
    serde_json::Value::Object(body)
}

#[cfg(feature = "upstream-hot-restart")]
fn is_valid_hot_restart_completion(value: &serde_json::Value) -> bool {
    value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
}

#[cfg(feature = "upstream-hot-restart")]
fn hot_restart_recovery_metadata(configured: bool) -> BTreeMap<String, String> {
    BTreeMap::from([(
        String::from("hot_restart_recovery_configured"),
        configured.to_string(),
    )])
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LocalRecoveryCause {
    UpstreamStall,
    TransientStatus,
    TransientTransport,
    RequestDeadline,
}

impl LocalRecoveryCause {
    const fn as_str(self) -> &'static str {
        match self {
            Self::UpstreamStall => "upstream_stall",
            Self::TransientStatus => "transient_status",
            Self::TransientTransport => "transient_transport",
            Self::RequestDeadline => "request_deadline",
        }
    }
}

struct LocalRecoveryGate {
    metadata: BTreeMap<String, String>,
    permits_retry: bool,
    applied: bool,
    permits_deadline_replay: bool,
}

async fn local_recovery_gate_for_attempt_failure(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
    failure: &ShieldedAttemptFailure,
) -> LocalRecoveryGate {
    let cause = if failure.abort_reason.as_deref() == Some(REQUEST_DEADLINE_ABORT_REASON) {
        Some(LocalRecoveryCause::RequestDeadline)
    } else {
        local_recovery_transport_cause(failure)
    };
    local_recovery_gate(runtime, can_retry, cause).await
}

fn local_recovery_transport_cause(failure: &ShieldedAttemptFailure) -> Option<LocalRecoveryCause> {
    if failure.retry_cause == Some(ShieldedRetryCause::TransientStream) {
        return Some(LocalRecoveryCause::TransientTransport);
    }
    #[cfg(feature = "upstream-hot-restart")]
    {
        if matches!(
            failure.transport_failure,
            Some(ReqwestFailureKind::Timeout | ReqwestFailureKind::Connect)
        ) {
            return Some(LocalRecoveryCause::TransientTransport);
        }
    }
    let _ = failure;
    None
}

async fn local_recovery_gate_for_status(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
    status: reqwest::StatusCode,
) -> LocalRecoveryGate {
    let cause = if matches!(status.as_u16(), 502..=504) {
        Some(LocalRecoveryCause::TransientStatus)
    } else {
        None
    };
    local_recovery_gate(runtime, can_retry, cause).await
}

async fn local_recovery_gate_for_upstream_stall(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
) -> LocalRecoveryGate {
    local_recovery_gate(runtime, can_retry, Some(LocalRecoveryCause::UpstreamStall)).await
}

async fn local_recovery_gate(
    runtime: &ShieldedRetryRuntime,
    can_retry: bool,
    cause: Option<LocalRecoveryCause>,
) -> LocalRecoveryGate {
    let Some(cause) = cause else {
        return unapplied_local_recovery_gate();
    };
    if (!can_retry && cause != LocalRecoveryCause::RequestDeadline)
        || (!runtime.local_recovery_policy.enabled
            && runtime.local_recovery_policy.restart_command.is_empty())
    {
        return unapplied_local_recovery_gate();
    }

    if !runtime.local_recovery_policy.enabled {
        return unapplied_local_recovery_gate();
    }
    if runtime.local_recovery_policy.restart_command.is_empty() {
        return unapplied_local_recovery_gate();
    }
    let mut metadata = local_recovery_metadata(runtime, cause);
    if runtime.downstream_drop_signal.is_dropped() {
        return skipped_local_recovery_gate(metadata, "skipped_downstream_dropped", false);
    }
    let previous_attempts = runtime
        .local_recovery_attempts
        .fetch_add(1, Ordering::SeqCst);
    if previous_attempts >= u64::from(runtime.local_recovery_policy.max_attempts_per_request) {
        metadata.insert(
            String::from("local_recovery_status"),
            String::from("skipped_request_budget_exhausted"),
        );
        metadata.insert(
            String::from("local_recovery_permits_retry"),
            String::from("false"),
        );
        metadata.insert(
            String::from("local_recovery_request_attempts_used"),
            previous_attempts.to_string(),
        );
        return applied_local_recovery_gate(metadata, false, false);
    }
    metadata.insert(
        String::from("local_recovery_request_attempts_used"),
        previous_attempts.saturating_add(1).to_string(),
    );

    let recovery_metadata = run_local_recovery(runtime, cause).await;
    metadata.extend(recovery_metadata);
    let permits_retry = local_recovery_permits_retry(&metadata);
    let permits_deadline_replay =
        cause == LocalRecoveryCause::RequestDeadline && local_recovery_completed_ready(&metadata);
    if permits_deadline_replay {
        runtime
            .local_recovery_deadline_replay_permits
            .fetch_add(1, Ordering::SeqCst);
    }
    metadata.insert(
        String::from("local_recovery_permits_retry"),
        permits_retry.to_string(),
    );
    applied_local_recovery_gate(metadata, permits_retry, permits_deadline_replay)
}

fn unapplied_local_recovery_gate() -> LocalRecoveryGate {
    LocalRecoveryGate {
        metadata: BTreeMap::new(),
        permits_retry: true,
        applied: false,
        permits_deadline_replay: false,
    }
}

fn skipped_local_recovery_gate(
    mut metadata: BTreeMap<String, String>,
    status: &str,
    permits_retry: bool,
) -> LocalRecoveryGate {
    metadata.insert(String::from("local_recovery_status"), status.to_owned());
    metadata.insert(
        String::from("local_recovery_permits_retry"),
        permits_retry.to_string(),
    );
    applied_local_recovery_gate(metadata, permits_retry, false)
}

fn applied_local_recovery_gate(
    metadata: BTreeMap<String, String>,
    permits_retry: bool,
    permits_deadline_replay: bool,
) -> LocalRecoveryGate {
    LocalRecoveryGate {
        metadata,
        permits_retry,
        applied: true,
        permits_deadline_replay,
    }
}

fn local_recovery_metadata(
    runtime: &ShieldedRetryRuntime,
    cause: LocalRecoveryCause,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("local_recovery_configured"),
            runtime.local_recovery_policy.is_configured().to_string(),
        ),
        (
            String::from("local_recovery_cause"),
            cause.as_str().to_owned(),
        ),
        (
            String::from("local_recovery_profile"),
            runtime.upstream_profile.name.clone(),
        ),
    ])
}

fn local_recovery_permits_retry(metadata: &BTreeMap<String, String>) -> bool {
    match metadata.get("local_recovery_status").map(String::as_str) {
        Some("skipped_disabled" | "skipped_no_command" | "succeeded") => true,
        Some("joined_inflight") => metadata
            .get("local_recovery_joined_status")
            .is_some_and(|status| status == "succeeded"),
        _ => false,
    }
}

fn local_recovery_completed_ready(metadata: &BTreeMap<String, String>) -> bool {
    matches!(
        metadata.get("local_recovery_status").map(String::as_str),
        Some("succeeded")
    ) || (metadata
        .get("local_recovery_status")
        .is_some_and(|status| status == "joined_inflight")
        && metadata
            .get("local_recovery_joined_status")
            .is_some_and(|status| status == "succeeded"))
}

fn consume_local_recovery_deadline_replay_permit(runtime: &ShieldedRetryRuntime) -> bool {
    runtime
        .local_recovery_deadline_replay_permits
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |permits| {
            permits.checked_sub(1)
        })
        .is_ok()
}

async fn run_local_recovery(
    runtime: &ShieldedRetryRuntime,
    cause: LocalRecoveryCause,
) -> BTreeMap<String, String> {
    run_local_recovery_for_profile(
        &runtime.local_recovery_policy,
        &runtime.local_recovery,
        runtime.client.clone(),
        runtime.upstream_profile.base_url.clone(),
        cause,
        Some(Duration::from_secs(
            runtime.upstream_profile.restart_queue.restart_timeout_secs,
        )),
    )
    .await
}

async fn run_local_recovery_for_profile(
    policy: &LocalRecoveryPolicy,
    coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
    client: Client,
    base_url: String,
    cause: LocalRecoveryCause,
    episode_timeout: Option<Duration>,
) -> BTreeMap<String, String> {
    let mut state = coordinator.state.lock().await;
    if state.running {
        drop(state);
        return wait_for_local_recovery_result(policy, coordinator, true).await;
    }

    let mut metadata = BTreeMap::new();
    let now = Instant::now();
    if let Some(last_finished) = state.last_finished {
        let elapsed = now.saturating_duration_since(last_finished);
        if elapsed < policy.cooldown {
            metadata.insert(
                String::from("local_recovery_status"),
                String::from("skipped_cooldown"),
            );
            metadata.insert(
                String::from("local_recovery_cooldown_remaining_ms"),
                policy
                    .cooldown
                    .saturating_sub(elapsed)
                    .as_millis()
                    .to_string(),
            );
            return metadata;
        }
    }

    let window_started = state.window_started.unwrap_or(now);
    if now.saturating_duration_since(window_started) >= policy.budget_window {
        state.window_started = Some(now);
        state.runs_in_window = 0;
    } else if state.runs_in_window >= policy.max_per_window {
        metadata.insert(
            String::from("local_recovery_status"),
            String::from("skipped_budget_exhausted"),
        );
        metadata.insert(
            String::from("local_recovery_budget_runs"),
            state.runs_in_window.to_string(),
        );
        metadata.insert(
            String::from("local_recovery_budget_max_per_window"),
            policy.max_per_window.to_string(),
        );
        return metadata;
    } else if state.window_started.is_none() {
        state.window_started = Some(now);
    }

    state.running = true;
    state.recovery_started = Some(now);
    let recovery_timeout = policy
        .restart_timeout
        .saturating_add(policy.readiness_deadline)
        .saturating_add(Duration::from_secs(1));
    state.recovery_deadline = Some(
        now + episode_timeout.map_or(recovery_timeout, |timeout| recovery_timeout.min(timeout)),
    );
    state.runs_in_window = state.runs_in_window.saturating_add(1);
    drop(state);

    spawn_local_recovery_task(
        policy.clone(),
        Arc::clone(coordinator),
        client,
        base_url,
        cause,
        episode_timeout,
    );

    wait_for_local_recovery_result(policy, coordinator, false).await
}

fn spawn_local_recovery_task(
    policy: LocalRecoveryPolicy,
    coordinator: Arc<UpstreamStallRecoveryCoordinator>,
    client: Client,
    base_url: String,
    cause: LocalRecoveryCause,
    episode_timeout: Option<Duration>,
) {
    tokio::spawn(async move {
        let trigger_cause = cause.as_str().to_owned();
        let recovery_trigger_cause = trigger_cause.clone();
        let recovery = async {
            let mut metadata = BTreeMap::from([(
                String::from("local_recovery_trigger_cause"),
                recovery_trigger_cause,
            )]);
            metadata.extend(run_local_recovery_restart_command(&policy).await);
            if metadata
                .get("local_recovery_restart_status")
                .is_some_and(|status| status == "succeeded")
            {
                metadata.extend(run_local_recovery_readiness(client, base_url, &policy).await);
            }
            if !metadata.contains_key("local_recovery_status") {
                let status = match metadata
                    .get("local_recovery_readiness_status")
                    .map(String::as_str)
                {
                    Some("ready") => "succeeded",
                    Some("timeout") => "readiness_timeout",
                    Some("error") => "readiness_error",
                    Some(_) => "readiness_not_ready",
                    None => "restart_failed",
                };
                metadata.insert(String::from("local_recovery_status"), status.to_owned());
            }
            metadata
        };
        let metadata = match episode_timeout {
            Some(timeout_duration) => match timeout(timeout_duration, recovery).await {
                Ok(metadata) => metadata,
                Err(_elapsed) => BTreeMap::from([
                    (String::from("local_recovery_trigger_cause"), trigger_cause),
                    (
                        String::from("local_recovery_status"),
                        String::from("episode_timeout"),
                    ),
                ]),
            },
            None => recovery.await,
        };
        finish_upstream_stall_recovery(&coordinator, metadata).await;
    });
}

async fn wait_for_local_recovery_result(
    policy: &LocalRecoveryPolicy,
    coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
    joined_inflight: bool,
) -> BTreeMap<String, String> {
    let mut state = coordinator.state.lock().await;
    let recovery_started = *state.recovery_started.get_or_insert_with(Instant::now);
    let deadline = *state.recovery_deadline.get_or_insert_with(|| {
        recovery_started
            + policy
                .restart_timeout
                .saturating_add(policy.readiness_deadline)
                .saturating_add(Duration::from_secs(1))
    });
    drop(state);
    loop {
        let notified = coordinator.notify.notified();
        tokio::pin!(notified);
        let _ = notified.as_mut().enable();

        let state = coordinator.state.lock().await;
        if !state.running {
            return completed_local_recovery_metadata(&state, joined_inflight);
        }
        drop(state);

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || timeout(remaining, notified).await.is_err() {
            let state = coordinator.state.lock().await;
            if !state.running {
                return completed_local_recovery_metadata(&state, joined_inflight);
            }
            drop(state);

            return BTreeMap::from([(
                String::from("local_recovery_status"),
                if joined_inflight {
                    String::from("join_timeout")
                } else {
                    String::from("completion_timeout")
                },
            )]);
        }
    }
}

fn completed_local_recovery_metadata(
    state: &UpstreamStallRecoveryState,
    joined_inflight: bool,
) -> BTreeMap<String, String> {
    let Some(last_result) = &state.last_result else {
        return BTreeMap::from([(
            String::from("local_recovery_status"),
            String::from("missing_result"),
        )]);
    };
    if !joined_inflight {
        return last_result.clone();
    }
    let mut joined = BTreeMap::from([(
        String::from("local_recovery_status"),
        String::from("joined_inflight"),
    )]);
    if let Some(status) = last_result.get("local_recovery_status") {
        joined.insert(String::from("local_recovery_joined_status"), status.clone());
    }
    joined
}

async fn run_local_recovery_restart_command(
    policy: &LocalRecoveryPolicy,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("local_recovery_restart_ran"),
        String::from("true"),
    )]);
    let program = &policy.restart_command[0];
    let args = &policy.restart_command[1..];
    let mut command = Command::new(program);
    command
        .args(args)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_recovery_command(&mut command);
    let mut child = match command.spawn() {
        Ok(child) => RecoveryProcessGuard::new(child),
        Err(error) => {
            metadata.insert(
                String::from("local_recovery_restart_status"),
                String::from("spawn_failed"),
            );
            metadata.insert(
                String::from("local_recovery_restart_error"),
                error.kind().to_string(),
            );
            metadata.insert(
                String::from("local_recovery_status"),
                String::from("spawn_failed"),
            );
            return metadata;
        }
    };
    match timeout(policy.restart_timeout, child.wait()).await {
        Ok(Ok(status)) => {
            let restart_status = if status.success() {
                "succeeded"
            } else {
                "exit_failure"
            };
            metadata.insert(
                String::from("local_recovery_restart_status"),
                restart_status.to_owned(),
            );
            if restart_status != "succeeded" {
                metadata.insert(
                    String::from("local_recovery_status"),
                    restart_status.to_owned(),
                );
            }
            if let Some(code) = status.code() {
                metadata.insert(
                    String::from("local_recovery_restart_exit_code"),
                    code.to_string(),
                );
            }
        }
        Ok(Err(error)) => {
            metadata.insert(
                String::from("local_recovery_restart_status"),
                String::from("wait_failed"),
            );
            metadata.insert(
                String::from("local_recovery_restart_error"),
                error.kind().to_string(),
            );
            metadata.insert(
                String::from("local_recovery_status"),
                String::from("wait_failed"),
            );
        }
        Err(_elapsed) => {
            metadata.insert(
                String::from("local_recovery_restart_status"),
                String::from("timeout_killed"),
            );
            metadata.insert(
                String::from("local_recovery_status"),
                String::from("timeout_killed"),
            );
            metadata.extend(terminate_timed_out_recovery_child(&mut child).await);
        }
    }
    metadata
}

async fn run_local_recovery_readiness(
    client: Client,
    base_url: String,
    policy: &LocalRecoveryPolicy,
) -> BTreeMap<String, String> {
    let deadline = Instant::now() + policy.readiness_deadline;
    loop {
        if Instant::now() >= deadline {
            return BTreeMap::from([(
                String::from("local_recovery_readiness_status"),
                String::from("timeout"),
            )]);
        }
        match send_local_recovery_readiness_probe(&client, &base_url, policy).await {
            Ok(true) => {
                return BTreeMap::from([(
                    String::from("local_recovery_readiness_status"),
                    String::from("ready"),
                )]);
            }
            Ok(false) => {}
            Err(error) => {
                if Instant::now() >= deadline {
                    return BTreeMap::from([
                        (
                            String::from("local_recovery_readiness_status"),
                            String::from("error"),
                        ),
                        (String::from("local_recovery_readiness_error"), error),
                    ]);
                }
            }
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return BTreeMap::from([(
                String::from("local_recovery_readiness_status"),
                String::from("timeout"),
            )]);
        }
        tokio::time::sleep(std::cmp::min(policy.readiness_interval, remaining)).await;
    }
}

async fn send_local_recovery_readiness_probe(
    client: &Client,
    base_url: &str,
    policy: &LocalRecoveryPolicy,
) -> Result<bool, String> {
    let uri = policy
        .readiness_endpoint
        .parse::<Uri>()
        .map_err(|error| format!("readiness endpoint parse failed: {error}"))?;
    let upstream_url = build_upstream_url(base_url, &uri).map_err(|error| error.to_string())?;
    let response = client
        .post(upstream_url)
        .header(
            HeaderName::from_static("x-llm-guard-proxy-probe"),
            HeaderValue::from_static("local-recovery"),
        )
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .body(policy.readiness_body.to_string())
        .timeout(policy.readiness_request_timeout)
        .send()
        .await
        .map_err(|error| sanitized_reqwest_error(&error))?;
    if !response.status().is_success() {
        return Ok(false);
    }
    let body = response
        .bytes()
        .await
        .map_err(|error| sanitized_reqwest_error(&error))?;
    let value = serde_json::from_slice::<serde_json::Value>(&body)
        .map_err(|error| format!("readiness response JSON decode failed: {error}"))?;
    Ok(is_valid_chat_completion_probe_response(&value))
}

fn is_valid_chat_completion_probe_response(value: &serde_json::Value) -> bool {
    value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
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
    let local_recovery_gate = local_recovery_gate_for_upstream_stall(runtime, can_retry).await;
    if local_recovery_gate.applied {
        return UpstreamStallRecoveryGate {
            metadata: local_recovery_gate.metadata,
            permits_retry: local_recovery_gate.permits_retry,
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
    let deadline = Instant::now() + recovery_join_timeout(policy.recovery_timeout);
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
        if remaining.is_zero() {
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

        // A result publication can race a waiter that has already prepared its notification.
        // Poll within the budgeted handoff margin so that race cannot defer a completed result
        // until the full public join deadline.
        let poll_interval = recovery_result_poll_interval();
        let _notified = timeout(remaining.min(poll_interval), notified).await;
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
        Ok(child) => RecoveryProcessGuard::new(child),
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
    metadata
        .extend(wait_for_recovery_child_with_timeout(&mut child, policy.recovery_timeout).await);
    metadata
}

async fn wait_for_recovery_child_with_timeout(
    child: &mut RecoveryProcessGuard,
    recovery_timeout: Duration,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    match timeout(recovery_timeout, child.wait()).await {
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
            metadata.extend(terminate_timed_out_recovery_child(child).await);
        }
    }
    metadata
}

/// Shielded attempts buffer the upstream independently of downstream response
/// consumption, so headers must count as progress before aggregation begins.
fn record_shielded_attempt_response_start(runtime: &ShieldedRetryRuntime) {
    if let Some(watchdog_request) = runtime.stuck_watchdog_request.as_ref() {
        watchdog_request.record_upstream_response_started();
    }
}

async fn start_shielded_attempt(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    cot_salvage: Option<&CotSalvageContext>,
    ignores_request_deadline: bool,
) -> Result<ShieldedStartedAttempt, ShieldedAttemptFailure> {
    let attempt_id = AttemptId::for_request(&runtime.request_id, attempt_number);
    let attempt_started_at_unix_ms = unix_time_millis();
    let attempt_plan =
        runtime
            .retry_policy
            .attempt_plan(attempt_number, &runtime.upstream_profile, cot_salvage);
    let (upstream_body, anti_loop_hint_applied, attempt_thinking_metadata) = shielded_attempt_body(
        runtime,
        attempt_number,
        retry_cause,
        &attempt_plan,
        cot_salvage,
    );
    let request_metadata = shielded_attempt_request_metadata(
        runtime,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
        &attempt_plan,
        cot_salvage,
        &attempt_thinking_metadata,
    );
    let rendered = render_shielded_endpoint_body(runtime, &upstream_body).map_err(|error| {
        shielded_start_transport_failure(ShieldedStartFailureInput {
            runtime,
            attempt_id: attempt_id.clone(),
            attempt_number,
            attempt_started_at_unix_ms,
            request_metadata: request_metadata.clone(),
            raw_request_body: None,
            evidence_upstream_body: upstream_body.clone(),
            error,
        })
    })?;
    let raw_request_body = raw_payload_text(&rendered.body);
    let evidence_upstream_body = rendered.body.clone();
    let upstream_timeout = shielded_attempt_upstream_timeout(runtime, ignores_request_deadline);
    let start_failure = |error| ShieldedStartFailureInput {
        runtime,
        attempt_id: attempt_id.clone(),
        attempt_number,
        attempt_started_at_unix_ms,
        request_metadata: request_metadata.clone(),
        raw_request_body: raw_request_body.clone(),
        evidence_upstream_body: evidence_upstream_body.clone(),
        error,
    };
    let sent = match send_shielded_upstream_attempt(
        runtime,
        (
            attempt_id.clone(),
            attempt_number,
            attempt_started_at_unix_ms,
        ),
        &request_metadata,
        rendered,
        upstream_body,
        upstream_timeout,
        ignores_request_deadline,
    )
    .await
    {
        Ok(sent) => sent,
        Err(error) => return Err(shielded_start_transport_failure(start_failure(error))),
    };
    let (terminal_endpoint, endpoint_retry_order) = shielded_retry_endpoint(runtime, &sent);
    let EndpointResponse::Upstream(response) = sent.response else {
        return Err(shielded_start_transport_failure(start_failure(
            ProxyError::upstream_body(String::from(
                "shielded chat request unexpectedly produced a rewritten endpoint response",
            )),
        )));
    };
    record_shielded_attempt_response_start(runtime);
    let upstream_status = response.status();
    let upstream_headers = response.headers().clone();
    let upstream_mode = upstream_mode_from_headers(&upstream_headers);
    Ok(ShieldedStartedAttempt {
        info: ShieldedAttemptInfo {
            attempt_id: sent.attempt_id,
            request_id: runtime.request_id.clone(),
            attempt_number: sent.attempt_number,
            attempt_max_attempts: runtime.retry_policy.max_attempts,
            started_at_unix_ms: sent.attempt_started_at_unix_ms,
            upstream_status,
            upstream_headers,
            upstream_mode,
            request_metadata: sent.attempt_request_metadata,
            raw_request_body,
            upstream_body: evidence_upstream_body,
        },
        response,
        completed_endpoint_attempt_records: sent.completed_attempt_records,
        terminal_endpoint,
        terminal_endpoint_protocol: sent.terminal_endpoint_protocol,
        endpoint_retry_order,
        ignores_request_deadline,
    })
}

fn render_shielded_endpoint_body(
    runtime: &ShieldedRetryRuntime,
    body: &Bytes,
) -> Result<reranker_protocol::RenderedEndpointRequest, ProxyError> {
    reranker_protocol::render_openai_endpoint(
        &runtime.terminal_endpoint,
        runtime.forward_uri.clone(),
        body,
        &runtime.original_downstream_headers,
        runtime.transformed_request_headers,
    )
}

fn shielded_retry_endpoint(
    runtime: &ShieldedRetryRuntime,
    sent: &SentUpstreamResponse,
) -> (UpstreamEndpointConfig, Vec<String>) {
    let terminal_endpoint = sent
        .terminal_endpoint
        .clone()
        .unwrap_or_else(|| runtime.terminal_endpoint.clone());
    let endpoint_retry_order =
        shielded_endpoint_retry_order(&terminal_endpoint, &runtime.endpoint_retry_order);
    (terminal_endpoint, endpoint_retry_order)
}

fn shielded_endpoint_retry_order(
    terminal_endpoint: &UpstreamEndpointConfig,
    endpoint_retry_order: &[String],
) -> Vec<String> {
    let mut retry_order = endpoint_retry_order.to_vec();
    retry_order.retain(|base_url| base_url != &terminal_endpoint.base_url);
    retry_order.insert(0, terminal_endpoint.base_url.clone());
    retry_order
}

fn update_shielded_retry_endpoint(
    runtime: &mut ShieldedRetryRuntime,
    started: &ShieldedStartedAttempt,
) {
    runtime
        .terminal_endpoint
        .clone_from(&started.terminal_endpoint);
    runtime.terminal_endpoint_protocol = started.terminal_endpoint_protocol;
    runtime
        .endpoint_retry_order
        .clone_from(&started.endpoint_retry_order);
}

fn shielded_attempt_upstream_timeout(
    runtime: &ShieldedRetryRuntime,
    ignores_request_deadline: bool,
) -> Duration {
    if ignores_request_deadline {
        runtime.upstream_timeout
    } else {
        runtime
            .request_deadline
            .remaining()
            .map_or(Duration::ZERO, |remaining| {
                runtime.upstream_timeout.min(remaining)
            })
    }
}

async fn send_shielded_upstream_attempt(
    runtime: &ShieldedRetryRuntime,
    attempt: (AttemptId, u32, u64),
    request_metadata: &BTreeMap<String, String>,
    rendered: reranker_protocol::RenderedEndpointRequest,
    retry_body: Bytes,
    upstream_timeout: Duration,
    ignores_request_deadline: bool,
) -> Result<SentUpstreamResponse, ProxyError> {
    let (attempt_id, attempt_number, attempt_started_at_unix_ms) = attempt;
    let request_deadline = (!ignores_request_deadline)
        .then(|| {
            runtime
                .request_deadline
                .remaining()
                .map(|remaining| Instant::now() + remaining)
        })
        .flatten();
    send_first_upstream_attempt(UpstreamAttemptContext {
        client: &runtime.client,
        method: runtime.method.clone(),
        upstream_url: rendered.url,
        downstream_headers: &rendered.headers,
        upstream_body: rendered.body,
        retry_body,
        upstream_timeout,
        attempt_id,
        attempt_number,
        request_id: &runtime.request_id,
        attempt_started_at_unix_ms,
        request_metadata,
        attempt_request_metadata: request_metadata,
        shutdown: runtime.shutdown.subscribe(),
        failover_retry: runtime.upstream_profile.has_endpoint_failover().then_some(
            UpstreamFailoverRetryContext {
                registry: runtime.upstream_health.as_ref(),
                profile: &runtime.upstream_profile,
                local_forward_uri: runtime.forward_uri.clone(),
                original_downstream_headers: &runtime.original_downstream_headers,
                canonical_reranker: None,
                transformed_request_headers: runtime.transformed_request_headers,
                initial_endpoint: &runtime.terminal_endpoint,
                request_deadline,
                endpoint_retry_order: &runtime.endpoint_retry_order,
                shutdown: runtime.shutdown.as_ref(),
            },
        ),
        terminal_endpoint_protocol: runtime.terminal_endpoint_protocol,
        canonical_reranker: None,
        decode_heterogeneous_reranker: false,
        model_id: None,
        request_deadline,
    })
    .await
}

struct ShieldedStartFailureInput<'runtime> {
    runtime: &'runtime ShieldedRetryRuntime,
    attempt_id: AttemptId,
    attempt_number: u32,
    attempt_started_at_unix_ms: u64,
    request_metadata: BTreeMap<String, String>,
    raw_request_body: Option<String>,
    evidence_upstream_body: Bytes,
    error: ProxyError,
}

fn shielded_start_error_type(error: &ProxyError, request_deadline_exhausted: bool) -> &'static str {
    if request_deadline_exhausted {
        REQUEST_DEADLINE_ERROR_TYPE
    } else if matches!(
        error,
        ProxyError::UpstreamTransport {
            failure: ReqwestFailureKind::Connect,
            ..
        }
    ) {
        "upstream_connect_error"
    } else {
        error.error_type()
    }
}

fn shielded_start_transport_failure(
    input: ShieldedStartFailureInput<'_>,
) -> ShieldedAttemptFailure {
    let mut finished_at_unix_ms = unix_time_millis();
    let request_deadline_exhausted = input.runtime.request_deadline.is_exhausted()
        && matches!(
            &input.error,
            ProxyError::UpstreamTransport {
                failure: ReqwestFailureKind::Timeout,
                ..
            }
        );
    let retry_cause =
        if matches!(&input.error, ProxyError::Shutdown { .. }) || request_deadline_exhausted {
            None
        } else {
            transport_retry_cause(&input.error)
        };
    let abort_reason = if request_deadline_exhausted {
        Some(String::from(REQUEST_DEADLINE_ABORT_REASON))
    } else {
        input.error.abort_reason().map(str::to_owned)
    };
    let error_type = shielded_start_error_type(&input.error, request_deadline_exhausted);
    let error_message = if request_deadline_exhausted {
        String::from("shielded request deadline exhausted before upstream response headers")
    } else {
        input.error.to_string()
    };
    let mut completed_endpoint_attempt_records = input.error.attempt_records();
    let terminal_endpoint_attempt = completed_endpoint_attempt_records.pop();
    let mut attempt_id = input.attempt_id;
    let mut request_id = input.runtime.request_id.clone();
    let mut attempt_number = input.attempt_number;
    let mut started_at_unix_ms = input.attempt_started_at_unix_ms;
    let mut upstream_mode = UpstreamMode::NotApplicable;
    let mut http_status = None;
    let mut request_metadata = input.request_metadata;
    let mut response_metadata =
        failed_response_metadata(started_at_unix_ms, finished_at_unix_ms, error_type);
    let mut raw_payloads = RawPayloads::default();
    if let Some(terminal) = terminal_endpoint_attempt {
        attempt_id = terminal.attempt_id;
        request_id = terminal.request_id;
        attempt_number = terminal.attempt_number;
        started_at_unix_ms = terminal.started_at_unix_ms;
        finished_at_unix_ms = terminal.finished_at_unix_ms.unwrap_or(finished_at_unix_ms);
        upstream_mode = terminal.upstream_mode;
        http_status = terminal.http_status;
        request_metadata = terminal.request_metadata;
        response_metadata.extend(terminal.response_metadata);
        raw_payloads = terminal.raw_payloads;
    }
    if raw_payloads.input.is_none() {
        raw_payloads.input = input.raw_request_body;
    }
    response_metadata.insert(String::from("error_type"), error_type.to_owned());
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("false"),
    );
    if let Some(reason) = &abort_reason {
        response_metadata.insert(String::from("abort_reason"), reason.clone());
        response_metadata.insert(String::from("shielded_terminal_reason"), reason.clone());
    }
    if request_deadline_exhausted {
        response_metadata.insert(
            String::from("request_deadline_exhausted"),
            String::from("true"),
        );
    }
    ShieldedAttemptFailure {
        attempt_id,
        request_id,
        attempt_number,
        started_at_unix_ms,
        finished_at_unix_ms,
        upstream_mode,
        http_status,
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: match &input.error {
            ProxyError::UpstreamTransport { failure, .. } => Some(*failure),
            _ => None,
        },
        error_type,
        error_message,
        retry_cause,
        abort_reason,
        request_metadata,
        response_metadata,
        raw_payloads,
        upstream_body: input.evidence_upstream_body,
        completed_endpoint_attempt_records,
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
        response_body: Bytes,
        sse_body: Bytes,
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
            response_body,
            sse_body,
        }
    }
}

fn shielded_attempt_body(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    attempt_plan: &ShieldedAttemptPlan,
    cot_salvage: Option<&CotSalvageContext>,
) -> (Bytes, bool, BTreeMap<String, String>) {
    let prepared_request = match runtime.chat_kind {
        ShieldedChatKind::NonStream => shielded_chat::prepare_non_stream_request(
            &runtime.downstream_body,
            &attempt_plan.thinking,
        ),
        ShieldedChatKind::Stream => {
            shielded_chat::prepare_stream_request(&runtime.downstream_body, &attempt_plan.thinking)
        }
        ShieldedChatKind::Generic => None,
    };
    let (prepared_body, mut thinking_metadata) = prepared_request.map_or_else(
        || {
            (
                runtime.upstream_body.clone(),
                runtime.thinking_metadata.clone(),
            )
        },
        |request| (request.upstream_body(), request.thinking_metadata().clone()),
    );

    if let Some(cot_salvage) = cot_salvage
        && let Some(body) = shielded_chat::body_with_cot_salvage_retry_hint(
            &prepared_body,
            attempt_number,
            runtime.retry_policy.max_attempts,
            cot_salvage.policy.as_str(),
            &cot_salvage.reasoning_prefix,
            attempt_plan.anti_loop_hint.as_deref(),
        )
    {
        let body = apply_shielded_param_override_to_body_or_original(
            body,
            &runtime.upstream_profile,
            &mut thinking_metadata,
        );
        return (body, true, thinking_metadata);
    }

    let (prepared_body, anti_loop_hint_applied) = if attempt_number > 1
        && runtime.retry_policy.anti_loop_hint_enabled
        && matches!(retry_cause, Some(ShieldedRetryCause::LoopDetected))
    {
        if let Some(body) = shielded_chat::body_with_anti_loop_retry_hint(
            &prepared_body,
            attempt_number,
            runtime.retry_policy.max_attempts,
            attempt_plan.anti_loop_hint.as_deref(),
        ) {
            (body, true)
        } else {
            (prepared_body, false)
        }
    } else {
        (prepared_body, false)
    };
    let prepared_body = apply_shielded_param_override_to_body_or_original(
        prepared_body,
        &runtime.upstream_profile,
        &mut thinking_metadata,
    );
    (prepared_body, anti_loop_hint_applied, thinking_metadata)
}

#[cfg(feature = "param-override")]
fn apply_shielded_param_override_to_body_or_original(
    body: Bytes,
    profile: &UpstreamProfileConfig,
    thinking_metadata: &mut BTreeMap<String, String>,
) -> Bytes {
    match apply_param_override_to_body(&body, profile) {
        Ok((rewritten, cap_decision)) => {
            shielded_chat::merge_final_answer_budget_metadata(
                &rewritten,
                thinking_metadata,
                cap_decision,
            );
            rewritten
        }
        Err(_error) => body,
    }
}

#[cfg(not(feature = "param-override"))]
fn apply_shielded_param_override_to_body_or_original(
    body: Bytes,
    _profile: &UpstreamProfileConfig,
    _thinking_metadata: &mut BTreeMap<String, String>,
) -> Bytes {
    body
}

fn shielded_attempt_request_metadata(
    runtime: &ShieldedRetryRuntime,
    attempt_number: u32,
    retry_cause: Option<ShieldedRetryCause>,
    anti_loop_hint_applied: bool,
    attempt_plan: &ShieldedAttemptPlan,
    cot_salvage: Option<&CotSalvageContext>,
    thinking_metadata: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut metadata = attempt_request_metadata(
        &runtime.downstream_method,
        &runtime.downstream_uri,
        &runtime.original_downstream_headers,
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
        thinking_metadata,
    );
    add_retry_attempt_metadata(
        &mut metadata,
        &runtime.retry_policy,
        attempt_number,
        retry_cause,
        anti_loop_hint_applied,
        attempt_plan,
        cot_salvage,
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
    cot_salvage: Option<&CotSalvageContext>,
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
        String::from("retry_request_deadline_ms"),
        policy.request_deadline.as_millis().to_string(),
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
        String::from("loop_failure_policy"),
        policy.loop_failure_policy.as_str().to_owned(),
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
    add_cot_salvage_request_metadata(metadata, cot_salvage, &attempt_plan.thinking);
}

fn add_cot_salvage_request_metadata(
    metadata: &mut BTreeMap<String, String>,
    cot_salvage: Option<&CotSalvageContext>,
    thinking: &ThinkingConfig,
) {
    let Some(cot_salvage) = cot_salvage else {
        metadata.insert(String::from("cot_salvage_used"), String::from("false"));
        return;
    };
    metadata.insert(String::from("cot_salvage_used"), String::from("true"));
    metadata.insert(
        String::from("cot_salvage_policy"),
        cot_salvage.policy.as_str().to_owned(),
    );
    metadata.insert(
        String::from("cot_salvage_source_attempt_id"),
        cot_salvage.source_attempt_id.as_str().to_owned(),
    );
    metadata.insert(
        String::from("cot_salvage_source_attempt_number"),
        cot_salvage.source_attempt_number.to_string(),
    );
    metadata.insert(
        String::from("cot_salvage_source_attempt_duration_ms"),
        cot_salvage.source_attempt_duration_ms.to_string(),
    );
    metadata.insert(
        String::from("cot_salvage_reasoning_prefix_bytes"),
        cot_salvage.prefix_bytes().to_string(),
    );
    metadata.insert(
        String::from("cot_salvage_thinking_budget_tokens"),
        thinking.budget_tokens.to_string(),
    );
    metadata.insert(
        String::from("cot_salvage_thinking_mode"),
        thinking.effective_mode().as_str().to_owned(),
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
        String::from("retry_request_deadline_ms"),
        policy.request_deadline.as_millis().to_string(),
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
        String::from("loop_failure_policy"),
        policy.loop_failure_policy.as_str().to_owned(),
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
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: None,
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
        completed_endpoint_attempt_records: Vec::new(),
    }
}

fn shutdown_shielded_attempt_failure(info: &ShieldedAttemptInfo) -> ShieldedAttemptFailure {
    let finished_at_unix_ms = unix_time_millis();
    let mut response_metadata = failed_response_metadata(
        info.started_at_unix_ms,
        finished_at_unix_ms,
        PROXY_SHUTTING_DOWN_ERROR_TYPE,
    );
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("true"),
    );
    response_metadata.insert(
        String::from("abort_reason"),
        SERVER_SHUTDOWN_ABORT_REASON.to_owned(),
    );
    ShieldedAttemptFailure {
        attempt_id: info.attempt_id.clone(),
        request_id: info.request_id.clone(),
        attempt_number: info.attempt_number,
        started_at_unix_ms: info.started_at_unix_ms,
        finished_at_unix_ms,
        upstream_mode: info.upstream_mode,
        http_status: Some(info.upstream_status.as_u16()),
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: None,
        error_type: PROXY_SHUTTING_DOWN_ERROR_TYPE,
        error_message: String::from("proxy is shutting down"),
        retry_cause: None,
        abort_reason: Some(SERVER_SHUTDOWN_ABORT_REASON.to_owned()),
        request_metadata: info.request_metadata.clone(),
        response_metadata,
        raw_payloads: RawPayloads {
            input: info.raw_request_body.clone(),
            ..RawPayloads::default()
        },
        upstream_body: info.upstream_body.clone(),
        completed_endpoint_attempt_records: Vec::new(),
    }
}

fn request_deadline_shielded_attempt_failure(info: &ShieldedAttemptInfo) -> ShieldedAttemptFailure {
    let finished_at_unix_ms = unix_time_millis();
    let mut response_metadata = failed_response_metadata(
        info.started_at_unix_ms,
        finished_at_unix_ms,
        REQUEST_DEADLINE_ERROR_TYPE,
    );
    response_metadata.insert(
        String::from("upstream_response_received"),
        String::from("true"),
    );
    response_metadata.insert(
        String::from("abort_reason"),
        REQUEST_DEADLINE_ABORT_REASON.to_owned(),
    );
    response_metadata.insert(
        String::from("request_deadline_exhausted"),
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
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: None,
        error_type: REQUEST_DEADLINE_ERROR_TYPE,
        error_message: String::from("shielded request deadline exhausted"),
        retry_cause: None,
        abort_reason: Some(REQUEST_DEADLINE_ABORT_REASON.to_owned()),
        request_metadata: info.request_metadata.clone(),
        response_metadata,
        raw_payloads: RawPayloads {
            input: info.raw_request_body.clone(),
            ..RawPayloads::default()
        },
        upstream_body: info.upstream_body.clone(),
        completed_endpoint_attempt_records: Vec::new(),
    }
}

fn mark_request_deadline_attempt_failure(failure: &mut ShieldedAttemptFailure) {
    failure.error_type = REQUEST_DEADLINE_ERROR_TYPE;
    failure.error_message = String::from("shielded request deadline exhausted");
    failure.retry_cause = None;
    failure.abort_reason = Some(REQUEST_DEADLINE_ABORT_REASON.to_owned());
    failure.response_metadata.insert(
        String::from("abort_reason"),
        REQUEST_DEADLINE_ABORT_REASON.to_owned(),
    );
    failure.response_metadata.insert(
        String::from("request_deadline_exhausted"),
        String::from("true"),
    );
}

fn cot_salvage_context_for_failure(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
) -> Option<CotSalvageContext> {
    if failure.retry_cause != Some(ShieldedRetryCause::LoopDetected)
        || !runtime.retry_policy.loop_failure_policy.uses_cot_salvage()
        || failure
            .response_metadata
            .get("loop_channel")
            .is_none_or(|channel| channel != "reasoning")
    {
        return None;
    }
    let reasoning = failure.raw_payloads.reasoning.as_deref()?;
    let reasoning_prefix = bounded_utf8_prefix(reasoning, COT_SALVAGE_PREFIX_MAX_BYTES);
    if reasoning_prefix.trim().is_empty() {
        return None;
    }
    Some(CotSalvageContext {
        policy: runtime.retry_policy.loop_failure_policy,
        source_attempt_id: failure.attempt_id.clone(),
        source_attempt_number: failure.attempt_number,
        source_attempt_duration_ms: failure
            .finished_at_unix_ms
            .saturating_sub(failure.started_at_unix_ms),
        reasoning_prefix,
    })
}

fn bounded_utf8_prefix(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next_end = index + character.len_utf8();
        if next_end > max_bytes {
            break;
        }
        end = next_end;
    }
    value[..end].to_owned()
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
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: None,
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
        completed_endpoint_attempt_records: Vec::new(),
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
        #[cfg(feature = "upstream-hot-restart")]
        transport_failure: None,
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
        completed_endpoint_attempt_records: Vec::new(),
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
        if failure.retry_cause == Some(ShieldedRetryCause::LoopDetected) {
            response_metadata.insert(
                String::from("retry_exhausted_reason"),
                String::from("loop_retry_exhausted"),
            );
        }
    }
    if let Some(abort_reason) = &failure.abort_reason {
        response_metadata.insert(String::from("abort_reason"), abort_reason.clone());
        response_metadata.insert(
            String::from("attempt_terminal_reason"),
            abort_reason.clone(),
        );
    } else if failure.retry_cause == Some(ShieldedRetryCause::LoopDetected) && retry_cause.is_none()
    {
        response_metadata.insert(
            String::from("attempt_terminal_reason"),
            String::from("loop_retry_exhausted"),
        );
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
        token_usage: TokenUsage::default(),
        request_metadata: failure.request_metadata.clone(),
        response_metadata,
        raw_payloads: failure.raw_payloads.clone(),
    }
}

fn copy_attempt_request_metadata(
    response_metadata: &mut BTreeMap<String, String>,
    request_metadata: &BTreeMap<String, String>,
) {
    for (key, value) in request_metadata
        .iter()
        .filter(|(key, _value)| key.starts_with("thinking_"))
    {
        response_metadata.insert(key.clone(), value.clone());
    }
    for key in [
        "attempt_index",
        "attempt_name",
        "retry_request_deadline_ms",
        "retry_previous_reason",
        "retry_anti_loop_hint_applied",
        "retry_shielded_streaming_enabled",
        "loop_failure_policy",
        "downstream_drop_policy",
        "attempt_thinking_mode",
        "attempt_thinking_budget_tokens",
        "attempt_thinking_max_tokens",
        "cot_salvage_used",
        "cot_salvage_policy",
        "cot_salvage_source_attempt_id",
        "cot_salvage_source_attempt_number",
        "cot_salvage_source_attempt_duration_ms",
        "cot_salvage_reasoning_prefix_bytes",
        "cot_salvage_thinking_budget_tokens",
        "cot_salvage_thinking_mode",
    ] {
        if let Some(value) = request_metadata.get(key) {
            response_metadata.insert(key.to_owned(), value.clone());
        }
    }
}

fn shielded_failure_outcome(
    failure: ShieldedAttemptFailure,
    attempt_records: Vec<AttemptRecord>,
    policy: &ShieldedRetryPolicy,
) -> ShieldedFailureOutcome {
    let mut response_metadata = failure.response_metadata.clone();
    let abort_reason = failure.abort_reason.clone();
    let request_status = if is_server_shutdown_failure(&failure) {
        RequestStatus::Aborted
    } else {
        RequestStatus::Failed
    };
    response_metadata.extend(retry_chain_metadata(
        &attempt_records,
        policy,
        request_status.as_str(),
    ));
    if let Some(reason) = &abort_reason {
        response_metadata.insert(String::from("abort_reason"), reason.clone());
        response_metadata.insert(String::from("shielded_terminal_reason"), reason.clone());
    } else if failure.retry_cause == Some(ShieldedRetryCause::LoopDetected) {
        response_metadata.insert(
            String::from("shielded_terminal_reason"),
            String::from("loop_retry_exhausted"),
        );
    }
    let error_type = structured_shielded_error_type(&failure);
    let downstream_status = if is_server_shutdown_failure(&failure) {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::BAD_GATEWAY
    };
    ShieldedFailureOutcome {
        error_type,
        error_message: failure.error_message,
        response_metadata,
        attempt_records,
        upstream_mode: failure.upstream_mode,
        downstream_status,
        retry_after_secs: None,
    }
}

fn request_deadline_exhausted_outcome(
    runtime: &ShieldedRetryRuntime,
    unstarted_attempt_number: u32,
    attempt_records: Vec<AttemptRecord>,
) -> ShieldedFailureOutcome {
    let mut response_metadata = BTreeMap::from([
        (
            String::from("request_deadline_exhausted"),
            String::from("true"),
        ),
        (
            String::from("retry_unstarted_attempt_number"),
            unstarted_attempt_number.to_string(),
        ),
        (
            String::from("retry_unstarted_reason"),
            String::from(REQUEST_DEADLINE_ABORT_REASON),
        ),
        (
            String::from("abort_reason"),
            String::from(REQUEST_DEADLINE_ABORT_REASON),
        ),
        (
            String::from("shielded_terminal_reason"),
            String::from(REQUEST_DEADLINE_ABORT_REASON),
        ),
    ]);
    response_metadata.extend(retry_chain_metadata(
        &attempt_records,
        &runtime.retry_policy,
        RequestStatus::Failed.as_str(),
    ));
    ShieldedFailureOutcome {
        error_type: REQUEST_DEADLINE_ERROR_TYPE,
        error_message: String::from(
            "shielded request deadline exhausted before next retry attempt",
        ),
        response_metadata,
        attempt_records,
        upstream_mode: UpstreamMode::NotApplicable,
        downstream_status: StatusCode::BAD_GATEWAY,
        retry_after_secs: None,
    }
}

fn structured_shielded_error_type(failure: &ShieldedAttemptFailure) -> &'static str {
    if is_server_shutdown_failure(failure) {
        return PROXY_SHUTTING_DOWN_ERROR_TYPE;
    }
    if failure.abort_reason.as_deref() == Some(REQUEST_DEADLINE_ABORT_REASON)
        || failure.error_type == REQUEST_DEADLINE_ERROR_TYPE
    {
        return REQUEST_DEADLINE_ERROR_TYPE;
    }
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
        request_deadline: Duration::from_millis(RetryConfig::default().request_deadline_ms),
        anti_loop_hint_enabled: false,
        shielded_streaming_enabled: false,
        downstream_drop_policy: DownstreamDropPolicy::Cancel,
        loop_failure_policy: LoopFailurePolicy::RetryLadder,
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
    let request_path = runtime.downstream_uri.path().to_owned();
    if upstream_status.is_success()
        && is_chat_completion_path(&request_path)
        && !response_has_valid_choices(&outcome.body)
    {
        runtime
            .malformed_response_counter
            .fetch_add(1, Ordering::Relaxed);
        return malformed_choices_error_response(&runtime.request_id);
    }
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
    let response_body = ObservedBufferedBody::new(
        outcome.body,
        observer,
        in_flight_permit,
        runtime.shutdown.subscribe(),
    );
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
    let request_id = runtime.request_id.clone();
    let cause = classify_shielded_failure_cause(&failure);
    let cause_code = cause.map(UpstreamFailureCause::code);
    let body = proxy_error_json_body_with_diagnostics(
        failure.error_type,
        &failure.error_message,
        cause_code,
        Some(&request_id),
    );
    if let Some(cause) = cause {
        runtime.upstream_failure_counters.increment(cause);
    }
    let is_shutdown_failure = failure
        .response_metadata
        .get("abort_reason")
        .is_some_and(|reason| reason == SERVER_SHUTDOWN_ABORT_REASON);
    let mut response_headers = json_response_headers(body.len());
    if let Some(retry_after_secs) = failure.retry_after_secs
        && let Ok(value) = HeaderValue::from_str(&retry_after_secs.to_string())
    {
        response_headers.insert(RETRY_AFTER, value);
    }
    let downstream_status = failure.downstream_status;
    let observer = shielded_retry_observer(
        runtime,
        ShieldedRetryObserverInput {
            downstream_mode: runtime.liveness.mode.downstream_mode(),
            downstream_status,
            downstream_headers: response_headers.clone(),
            upstream_mode: failure.upstream_mode,
            extra_response_metadata: failure.response_metadata,
            raw_payloads: RawPayloads::default(),
            completed_attempt_records: failure.attempt_records,
            final_attempt: None,
            attempt_progress: None,
        },
    );
    let completion = if is_shutdown_failure {
        BodyCompletion::Shutdown
    } else {
        BodyCompletion::UpstreamStreamError(failure.error_message)
    };
    let response_body = ObservedBufferedBody::new_with_completion(
        body,
        observer,
        in_flight_permit,
        completion,
        runtime.shutdown.subscribe(),
    );
    let mut response = Response::new(Body::from_stream(response_body));
    *response.status_mut() = downstream_status;
    *response.headers_mut() = response_headers;
    response
}

async fn shielded_retry_terminal_forward_response(
    runtime: &ShieldedRetryRuntime,
    terminal: ShieldedTerminalForward,
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
    let upstream_status = terminal.started.info.upstream_status;
    let upstream_headers = terminal.started.info.upstream_headers.clone();
    let request_path = runtime.downstream_uri.path().to_owned();
    let request_id = runtime.request_id.clone();
    let malformed_counter = runtime.malformed_response_counter.clone();
    let final_attempt = terminal.started.info.clone().into_final_context(
        BTreeMap::new(),
        RawPayloads::default(),
        Bytes::new(),
        Bytes::new(),
    );
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
        runtime.shutdown.subscribe(),
    );
    let response = downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    );
    validate_non_stream_chat_completion_response(
        response,
        &request_path,
        &request_id,
        &malformed_counter,
    )
    .await
}

async fn shielded_retry_direct_relay_response(
    runtime: &ShieldedRetryRuntime,
    outcome: ShieldedDirectRelayOutcome,
    in_flight_permit: InFlightPermit,
) -> Response<Body> {
    let upstream_status = outcome.started.info.upstream_status;
    let upstream_headers = outcome.started.info.upstream_headers.clone();
    let request_path = runtime.downstream_uri.path().to_owned();
    let request_id = runtime.request_id.clone();
    let malformed_counter = runtime.malformed_response_counter.clone();
    let final_attempt = outcome.started.info.clone().into_final_context(
        outcome.response_metadata.clone(),
        RawPayloads::default(),
        Bytes::new(),
        Bytes::new(),
    );
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
    let response_body = ObservedUpstreamBody::new_with_deadline(
        outcome.started.response.bytes_stream(),
        observer,
        in_flight_permit,
        BodyCompletion::Succeeded,
        runtime.shutdown.subscribe(),
        Some(outcome.request_deadline),
    );
    let response = downstream_response(
        upstream_status,
        &upstream_headers,
        Body::from_stream(response_body),
    );
    validate_non_stream_chat_completion_response(
        response,
        &request_path,
        &request_id,
        &malformed_counter,
    )
    .await
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
        persistence_tasks: Arc::clone(&runtime.persistence_tasks),
        shadow_evidence: runtime.shadow_evidence.clone(),
        paired_shadow_runtime: Some(runtime.clone()),
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
        stuck_watchdog_request: runtime.stuck_watchdog_request.clone(),
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
    response_body: Bytes,
    sse_body: Bytes,
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
            info.clone().into_final_context(
                extra_response_metadata,
                RawPayloads::default(),
                Bytes::new(),
                Bytes::new(),
            )
        });
    }
}

struct ForwardedBodyObserver {
    config: ConfigHandle,
    store: ObservabilityStore,
    evidence_store: EvidenceStore,
    persistence_tasks: Arc<PersistenceTasks>,
    shadow_evidence: ShadowEvidenceState,
    paired_shadow_runtime: Option<ShieldedRetryRuntime>,
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
    stuck_watchdog_request: Option<StuckWatchdogRequest>,
    final_attempt: Option<FinalAttemptContext>,
    retry_observation: Option<RetryObservation>,
    attempt_progress: Option<ShieldedAttemptProgressHandle>,
}

impl ForwardedBodyObserver {
    fn record_chunk(&self, chunk: &[u8]) {
        if let Some(stuck_watchdog_request) = &self.stuck_watchdog_request {
            // This is invoked from the body poll that yields the chunk downstream.
            stuck_watchdog_request.record_emitted_chunk(chunk);
        }
    }

    fn record(self, body_bytes: u64, completion: &BodyCompletion, response_body: &Bytes) {
        let finished_at_unix_ms = unix_time_millis();
        let mut attempts = self.completed_attempt_records;
        let mut final_attempt = self.final_attempt;
        let paired_shadow_runtime = self.paired_shadow_runtime;
        if matches!(
            completion,
            BodyCompletion::DownstreamDropped | BodyCompletion::Shutdown
        ) && let Some(progress) = &self.attempt_progress
        {
            let progress = shielded_attempt_progress(progress);
            attempts.clone_from(&progress.completed_attempt_records);
            final_attempt.clone_from(&progress.current_attempt);
        }
        let upstream_mode = final_attempt
            .as_ref()
            .map_or(self.upstream_mode, |attempt| attempt.upstream_mode);
        if let Some(stuck_watchdog_request) = &self.stuck_watchdog_request {
            let sse_body = final_attempt
                .as_ref()
                .map(|attempt| attempt.sse_body.as_ref())
                .unwrap_or_default();
            stuck_watchdog_request.record_response(response_body, sse_body);
        }
        if let Some(final_attempt) = &mut final_attempt
            && final_attempt.response_body.is_empty()
        {
            final_attempt.response_body = response_body.clone();
        }
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
            response_metadata
                .entry(String::from("shielded_terminal_reason"))
                .or_insert_with(|| completion.metadata_reason().to_owned());
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
        let store = self.store.clone();
        let evidence_store = self.evidence_store.clone();
        let config = self.config.clone();
        let shadow_evidence = self.shadow_evidence.clone();
        let terminal_reason: &'static str = completion.terminal_reason();
        self.persistence_tasks.spawn_blocking(move || {
            record_observability_many(&store, &request_record, &attempts);
            let evidence_written = record_evidence_many(
                EvidenceRecordContext {
                    config: &config,
                    store: &evidence_store,
                    shadow_evidence: &shadow_evidence,
                },
                &request_record,
                &attempts,
            );
            log_request_cleanup(
                &request_record,
                terminal_reason,
                unix_time_millis().saturating_sub(finished_at_unix_ms),
                evidence_written,
            );
            if evidence_written && let Some(runtime) = paired_shadow_runtime.as_ref() {
                maybe_schedule_paired_comparison_after_primary(runtime, &request_record, &attempts);
            }
        });
    }
}

fn log_request_cleanup(
    request: &RequestRecord,
    terminal_reason: &'static str,
    cleanup_latency_ms: u64,
    evidence_written: bool,
) {
    eprintln!(
        "{}",
        request_cleanup_log_line(
            request,
            terminal_reason,
            cleanup_latency_ms,
            evidence_written
        )
    );
}

fn request_cleanup_log_line(
    request: &RequestRecord,
    terminal_reason: &'static str,
    cleanup_latency_ms: u64,
    evidence_written: bool,
) -> String {
    format!(
        "llm_guard_proxy_request_cleanup request_id={} status={} terminal_reason={} cleanup_latency_ms={} http_status={} downstream_mode={} upstream_mode={} evidence_written={}",
        request.request_id.as_str(),
        request.status.as_str(),
        terminal_reason,
        cleanup_latency_ms,
        request.http_status.unwrap_or(0),
        request.downstream_mode.as_str(),
        request.upstream_mode.as_str(),
        evidence_written
    )
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
    response_metadata.insert(
        String::from("attempt_terminal_reason"),
        completion.metadata_reason().to_owned(),
    );
    if matches!(completion, BodyCompletion::FinalDirectRelayTerminated) {
        response_metadata.insert(
            String::from("final_direct_relay_terminated"),
            String::from("true"),
        );
        response_metadata.insert(
            String::from("request_deadline_exhausted"),
            String::from("true"),
        );
    }
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
        token_usage: parse_token_usage(&attempt.response_body, &attempt.sse_body),
        request_metadata,
        response_metadata,
        raw_payloads: attempt.raw_payloads,
    }
}

fn parse_token_usage(body: &[u8], sse_body: &[u8]) -> TokenUsage {
    parse_token_usage_json(body)
        .or_else(|| {
            std::str::from_utf8(sse_body).ok().and_then(|sse_body| {
                sse_body.lines().rev().find_map(|line| {
                    line.trim_end_matches('\r')
                        .strip_prefix("data:")
                        .and_then(|data| parse_token_usage_json(data.trim_start().as_bytes()))
                })
            })
        })
        .unwrap_or_default()
}

fn parse_token_usage_json(body: &[u8]) -> Option<TokenUsage> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = value.get("usage")?.as_object()?;
    Some(TokenUsage {
        input_tokens: usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64),
        output_tokens: usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64),
        cached_input_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(serde_json::Value::as_u64),
        reasoning_tokens: usage
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(serde_json::Value::as_u64),
    })
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
            String::from("retry_request_deadline_ms"),
            policy.request_deadline.as_millis().to_string(),
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
            String::from("loop_failure_policy"),
            policy.loop_failure_policy.as_str().to_owned(),
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
    FinalDirectRelayTerminated,
    Shutdown,
}

impl BodyCompletion {
    const fn request_status(&self) -> RequestStatus {
        match self {
            Self::Succeeded => RequestStatus::Succeeded,
            Self::UpstreamStreamError(_) => RequestStatus::Failed,
            Self::DownstreamDropped | Self::FinalDirectRelayTerminated | Self::Shutdown => {
                RequestStatus::Aborted
            }
        }
    }

    const fn attempt_status(&self) -> AttemptStatus {
        match self {
            Self::Succeeded => AttemptStatus::Succeeded,
            Self::UpstreamStreamError(_) => AttemptStatus::Failed,
            Self::DownstreamDropped | Self::FinalDirectRelayTerminated | Self::Shutdown => {
                AttemptStatus::Aborted
            }
        }
    }

    fn error_reason(&self) -> Option<String> {
        match self {
            Self::UpstreamStreamError(error) => Some(format!("upstream_stream_error: {error}")),
            Self::FinalDirectRelayTerminated => Some(String::from(
                "request_deadline_exhausted: final direct relay terminated",
            )),
            Self::Succeeded | Self::DownstreamDropped | Self::Shutdown => None,
        }
    }

    fn abort_reason(&self) -> Option<String> {
        match self {
            Self::DownstreamDropped => Some(String::from("downstream_body_dropped_before_eof")),
            Self::FinalDirectRelayTerminated => {
                Some(String::from(FINAL_DIRECT_RELAY_TERMINATED_ABORT_REASON))
            }
            Self::Shutdown => Some(String::from("server_shutdown")),
            Self::Succeeded | Self::UpstreamStreamError(_) => None,
        }
    }

    const fn terminal_reason(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::UpstreamStreamError(_) => "upstream_stream_error",
            Self::DownstreamDropped => "downstream_disconnect",
            Self::FinalDirectRelayTerminated => "final_direct_relay_terminated",
            Self::Shutdown => "server_shutdown",
        }
    }

    const fn metadata_reason(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::UpstreamStreamError(_) => "upstream_stream_error",
            Self::DownstreamDropped => "downstream_body_dropped_before_eof",
            Self::FinalDirectRelayTerminated => FINAL_DIRECT_RELAY_TERMINATED_ABORT_REASON,
            Self::Shutdown => SERVER_SHUTDOWN_ABORT_REASON,
        }
    }
}

struct ObservedUpstreamBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: InFlightPermit,
    shutdown: ShutdownSubscription,
    bytes_seen: u64,
    body_buffer: BytesMut,
    terminal_completion: BodyCompletion,
    deadline: Option<Pin<Box<Sleep>>>,
}

impl ObservedUpstreamBody {
    fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        shutdown: ShutdownSubscription,
    ) -> Self {
        Self::new_with_completion(
            stream,
            observer,
            in_flight_permit,
            BodyCompletion::Succeeded,
            shutdown,
            None,
        )
    }

    fn new_with_deadline(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        terminal_completion: BodyCompletion,
        shutdown: ShutdownSubscription,
        deadline: Option<ShieldedRequestDeadline>,
    ) -> Self {
        Self::new_with_completion(
            stream,
            observer,
            in_flight_permit,
            terminal_completion,
            shutdown,
            deadline,
        )
    }

    fn new_with_completion(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        terminal_completion: BodyCompletion,
        shutdown: ShutdownSubscription,
        deadline: Option<ShieldedRequestDeadline>,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            shutdown,
            bytes_seen: 0,
            body_buffer: BytesMut::new(),
            terminal_completion,
            deadline: deadline
                .map(|deadline| deadline.remaining().unwrap_or(Duration::ZERO))
                .map(tokio::time::sleep)
                .map(Box::pin),
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion) {
        let response_body = self.body_buffer.split().freeze();
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion, &response_body);
        }
    }
}

impl Stream for ObservedUpstreamBody {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.shutdown.poll_shutdown(cx).is_ready() {
            this.record_once(&BodyCompletion::Shutdown);
            return Poll::Ready(None);
        }
        if let Some(deadline) = &mut this.deadline
            && deadline.as_mut().poll(cx).is_ready()
        {
            this.record_once(&BodyCompletion::FinalDirectRelayTerminated);
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                let chunk_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                this.bytes_seen = this.bytes_seen.saturating_add(chunk_len);
                if let Some(observer) = &this.observer {
                    observer.record_chunk(&bytes);
                }
                if this.body_buffer.len() < TOKEN_USAGE_BODY_CAP {
                    let remaining = TOKEN_USAGE_BODY_CAP - this.body_buffer.len();
                    let take = remaining.min(bytes.len());
                    this.body_buffer.extend_from_slice(&bytes[..take]);
                }
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
    shutdown: ShutdownSubscription,
    bytes_seen: u64,
    terminal_completion: BodyCompletion,
}

impl ObservedBufferedBody {
    fn new(
        body: Bytes,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        shutdown: ShutdownSubscription,
    ) -> Self {
        Self::new_with_completion(
            body,
            observer,
            in_flight_permit,
            BodyCompletion::Succeeded,
            shutdown,
        )
    }

    fn new_with_completion(
        body: Bytes,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        terminal_completion: BodyCompletion,
        shutdown: ShutdownSubscription,
    ) -> Self {
        Self {
            body: (!body.is_empty()).then_some(body),
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            shutdown,
            bytes_seen: 0,
            terminal_completion,
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion, response_body: &Bytes) {
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion, response_body);
        }
    }
}

impl Stream for ObservedBufferedBody {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(body) = this.body.take() {
            let body_len = u64::try_from(body.len()).unwrap_or(u64::MAX);
            this.bytes_seen = this.bytes_seen.saturating_add(body_len);
            let completion =
                std::mem::replace(&mut this.terminal_completion, BodyCompletion::Succeeded);
            this.record_once(&completion, &body);
            return Poll::Ready(Some(Ok(body)));
        }

        if this.shutdown.poll_shutdown(cx).is_ready() {
            this.record_once(&BodyCompletion::Shutdown, &Bytes::new());
            return Poll::Ready(None);
        }

        let completion =
            std::mem::replace(&mut this.terminal_completion, BodyCompletion::Succeeded);
        this.record_once(&completion, &Bytes::new());
        Poll::Ready(None)
    }
}

impl Drop for ObservedBufferedBody {
    fn drop(&mut self) {
        let response_body = self.body.take().unwrap_or_default();
        self.record_once(&BodyCompletion::DownstreamDropped, &response_body);
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

struct ShieldedLivenessBodySettings {
    mode: ShieldedLivenessMode,
    accepted_response_mode: ShieldedAcceptedResponseMode,
    interval_secs: u64,
    upstream_failure_counters: Arc<UpstreamFailureCounters>,
    #[cfg(test)]
    shielded_heartbeat_ticks: Arc<AtomicU64>,
}

struct ShieldedLivenessBody {
    aggregate: ShieldedAggregateFuture,
    direct_stream: Option<DirectRelayStream>,
    direct_deadline: Option<Pin<Box<Sleep>>>,
    interval: Interval,
    mode: ShieldedLivenessMode,
    accepted_response_mode: ShieldedAcceptedResponseMode,
    observer: Option<ForwardedBodyObserver>,
    _in_flight_permit: InFlightPermit,
    shutdown: ShutdownSubscription,
    downstream_drop_signal: DownstreamDropSignal,
    upstream_failure_counters: Arc<UpstreamFailureCounters>,
    #[cfg(test)]
    shielded_heartbeat_ticks: Arc<AtomicU64>,
    bytes_seen: u64,
    terminal_completion: Option<BodyCompletion>,
    json_prefix_pending: bool,
}

impl ShieldedLivenessBody {
    fn new(
        aggregate: ShieldedAggregateFuture,
        settings: &ShieldedLivenessBodySettings,
        observer: ForwardedBodyObserver,
        in_flight_permit: InFlightPermit,
        shutdown: ShutdownSubscription,
        downstream_drop_signal: DownstreamDropSignal,
    ) -> Self {
        let period = Duration::from_secs(settings.interval_secs);
        let mut interval = tokio::time::interval_at(Instant::now() + period, period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            aggregate,
            direct_stream: None,
            direct_deadline: None,
            interval,
            mode: settings.mode,
            accepted_response_mode: settings.accepted_response_mode,
            observer: Some(observer),
            _in_flight_permit: in_flight_permit,
            shutdown,
            downstream_drop_signal,
            upstream_failure_counters: settings.upstream_failure_counters.clone(),
            #[cfg(test)]
            shielded_heartbeat_ticks: settings.shielded_heartbeat_ticks.clone(),
            bytes_seen: 0,
            terminal_completion: None,
            json_prefix_pending: settings.mode == ShieldedLivenessMode::JsonWhitespace,
        }
    }

    fn record_once(&mut self, completion: &BodyCompletion) {
        if let Some(observer) = self.observer.take() {
            observer.record(self.bytes_seen, completion, &Bytes::new());
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

    fn deadline_error_chunk(&self) -> Bytes {
        self.error_chunk(
            REQUEST_DEADLINE_ERROR_TYPE,
            "shielded request deadline exhausted during final direct relay",
        )
    }

    fn terminate_direct_relay_for_deadline(&mut self) -> Bytes {
        self.upstream_failure_counters
            .increment(UpstreamFailureCause::Timeout);
        self.direct_deadline = None;
        self.direct_stream = None;
        self.terminal_completion = Some(BodyCompletion::FinalDirectRelayTerminated);
        self.deadline_error_chunk()
    }

    fn start_direct_relay(&mut self, outcome: ShieldedDirectRelayOutcome) -> Option<Bytes> {
        let mut final_attempt = outcome.started.info.into_final_context(
            outcome.response_metadata.clone(),
            RawPayloads::default(),
            Bytes::new(),
            Bytes::new(),
        );
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
        let Some(remaining_deadline) = outcome.request_deadline.remaining() else {
            return Some(self.terminate_direct_relay_for_deadline());
        };
        self.direct_deadline = Some(Box::pin(tokio::time::sleep(remaining_deadline)));
        self.direct_stream = Some(Box::pin(outcome.started.response.bytes_stream()));
        None
    }

    fn poll_direct_stream(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Bytes, Infallible>>> {
        let Some(stream) = &mut self.direct_stream else {
            if let Some(completion) = self.terminal_completion.take() {
                self.record_once(&completion);
                return Poll::Ready(None);
            }
            return Poll::Pending;
        };
        if let Some(deadline) = &mut self.direct_deadline
            && deadline.as_mut().poll(cx).is_ready()
        {
            let chunk = self.terminate_direct_relay_for_deadline();
            return self.count_and_emit(chunk);
        }
        match stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => self.count_and_emit(bytes),
            Poll::Ready(Some(Err(error))) => {
                self.upstream_failure_counters.increment(
                    UpstreamFailureCause::from_reqwest_failure(ReqwestFailureKind::from_error(
                        &error,
                    )),
                );
                let error_message = sanitized_reqwest_error(&error);
                let chunk = self.error_chunk("llm_guard_upstream_error", &error_message);
                self.terminal_completion = Some(BodyCompletion::UpstreamStreamError(error_message));
                self.direct_stream = None;
                self.count_and_emit(chunk)
            }
            Poll::Ready(None) => {
                self.direct_stream = None;
                self.direct_deadline = None;
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
        if this.shutdown.poll_shutdown(cx).is_ready() {
            this.record_once(&BodyCompletion::Shutdown);
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
                if let Some(chunk) = this.start_direct_relay(outcome) {
                    return this.count_and_emit(chunk);
                }
                return this.poll_direct_stream(cx);
            }
            Poll::Ready(Err(failure)) => {
                if let Some(cause) = classify_shielded_failure_cause(&failure) {
                    this.upstream_failure_counters.increment(cause);
                }
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
            Poll::Ready(_instant) => {
                #[cfg(test)]
                this.shielded_heartbeat_ticks.fetch_add(1, Ordering::SeqCst);
                this.count_and_emit(heartbeat_chunk(this.mode))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ShieldedLivenessBody {
    fn drop(&mut self) {
        if let Some(completion) = self.terminal_completion.take() {
            self.record_once(&completion);
        } else if self.observer.is_some() {
            self.downstream_drop_signal.mark_dropped();
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

/// Adds the stable observability identifier at the single terminal proxy boundary.
///
/// This overwrites any upstream value so every client-visible terminal response
/// correlates with the request record persisted by this proxy.
fn finalize_proxy_terminal_response(
    mut response: Response<Body>,
    request_id: &RequestId,
) -> Response<Body> {
    let request_id = HeaderValue::from_str(request_id.as_str())
        .expect("generated request IDs must be valid HTTP header values");
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-request-id"), request_id);
    response
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

fn proxy_error_json_body_with_diagnostics(
    error_type: &str,
    message: &str,
    cause_code: Option<&str>,
    request_id: Option<&RequestId>,
) -> Bytes {
    let mut error = serde_json::Map::from_iter([
        (String::from("type"), json!(error_type)),
        (String::from("message"), json!(message)),
    ]);
    if let Some(cause) = cause_code {
        error.insert(String::from("cause"), json!(cause));
        error
            .entry(String::from("code"))
            .or_insert_with(|| json!(cause));
    }
    if let Some(rid) = request_id {
        error.insert(String::from("request_id"), json!(rid.as_str()));
    }
    Bytes::from(
        serde_json::Value::Object(serde_json::Map::from_iter([(
            String::from("error"),
            serde_json::Value::Object(error),
        )]))
        .to_string(),
    )
}

fn classify_shielded_failure_cause(
    failure: &ShieldedFailureOutcome,
) -> Option<UpstreamFailureCause> {
    let metadata = &failure.response_metadata;
    if failure.error_type == REQUEST_DEADLINE_ERROR_TYPE
        || metadata_has(metadata, "request_deadline_exhausted", "true")
        || metadata_has(metadata, "abort_reason", REQUEST_DEADLINE_ABORT_REASON)
        || metadata_has(
            metadata,
            "shielded_terminal_reason",
            REQUEST_DEADLINE_ABORT_REASON,
        )
    {
        return Some(UpstreamFailureCause::Timeout);
    }
    if metadata_has(metadata, "upstream_stall_detected", "true")
        || metadata_has(metadata, "abort_reason", "upstream_stall")
        || metadata_has(metadata, "shielded_terminal_reason", "upstream_stall")
    {
        return Some(UpstreamFailureCause::Timeout);
    }
    if metadata.contains_key("status_code") || failure.error_type == "upstream_status_error" {
        return Some(UpstreamFailureCause::StatusError);
    }
    if failure.error_type == "upstream_body_error"
        || metadata_has(metadata, "error_type", "upstream_body_error")
    {
        return Some(UpstreamFailureCause::BodyError);
    }
    classify_shielded_error_message_cause(&failure.error_message)
}

fn metadata_has(metadata: &BTreeMap<String, String>, key: &str, expected: &str) -> bool {
    metadata.get(key).is_some_and(|value| value == expected)
}

fn classify_shielded_error_message_cause(message: &str) -> Option<UpstreamFailureCause> {
    let message = message.to_ascii_lowercase();
    if message.contains("connect_failure")
        || message.contains("connect failed")
        || message.contains("connection")
    {
        Some(UpstreamFailureCause::ConnectFailed)
    } else if message.contains("timeout")
        || message.contains("timed out")
        || message.contains("deadline")
    {
        Some(UpstreamFailureCause::Timeout)
    } else if message.contains("status") && message.contains("http") {
        Some(UpstreamFailureCause::StatusError)
    } else if message.contains("body") || message.contains("decode") {
        Some(UpstreamFailureCause::BodyError)
    } else if message.contains("upstream") {
        Some(UpstreamFailureCause::TransportError)
    } else {
        None
    }
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
        ACCEPT_ENCODING,
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

fn extract_model_id(method: &Method, uri: &Uri, body: &Bytes) -> Option<String> {
    if let Some(model) = deepinfra_rerank_adapter::model_id_from_path(method, uri) {
        return Some(model.to_owned());
    }
    if score_adapter::is_score_request(method, uri) {
        return score_adapter::model_id_from_score_body(body);
    }
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

const MALFORMED_CHOICES_MESSAGE: &str =
    "upstream returned a malformed response: missing or invalid 'choices' field";

fn is_chat_completion_path(path: &str) -> bool {
    path == "/v1/chat/completions" || path == "/chat/completions"
}

fn is_application_json_response(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            let lower = value.to_ascii_lowercase();
            lower.contains("application/json") || lower.starts_with("application/")
        })
}

fn malformed_choices_error_response(request_id: &RequestId) -> Response<Body> {
    let body = json!({
        "error": {
            "message": MALFORMED_CHOICES_MESSAGE,
            "type": "upstream_error",
            "code": "malformed_response"
        }
    })
    .to_string();
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::BAD_GATEWAY;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Ok(value) = HeaderValue::from_str(request_id.as_str()) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-request-id"), value);
    }
    response
}

fn response_has_valid_choices(body: &Bytes) -> bool {
    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) => value
            .get("choices")
            .is_some_and(serde_json::Value::is_array),
        Err(_) => false,
    }
}

/// Validates a non-stream chat completion response at the final downstream
/// boundary. If the response is a 2xx for a non-stream `/v1/chat/completions`
/// (or `/chat/completions`) request with a JSON body that lacks a valid
/// `choices` array, it is replaced with an OpenAI-compatible 502 error.
/// Streaming responses, non-2xx responses, and non-chat endpoints pass through
/// unchanged.
async fn validate_non_stream_chat_completion_response(
    response: Response<Body>,
    request_path: &str,
    request_id: &RequestId,
    malformed_counter: &AtomicU64,
) -> Response<Body> {
    let status = response.status();
    let headers = response.headers().clone();
    if !status.is_success()
        || !is_chat_completion_path(request_path)
        || is_event_stream(&headers)
        || !is_application_json_response(&headers)
    {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(body_bytes) = to_bytes(body, MAX_PROXY_BODY_BYTES).await else {
        malformed_counter.fetch_add(1, Ordering::Relaxed);
        return malformed_choices_error_response(request_id);
    };
    if !response_has_valid_choices(&body_bytes) {
        malformed_counter.fetch_add(1, Ordering::Relaxed);
        return malformed_choices_error_response(request_id);
    }
    let mut reconstructed = Response::from_parts(parts, Body::from(body_bytes));
    if let Ok(content_length) =
        HeaderValue::from_str(&reconstructed.body().size_hint().lower().to_string())
    {
        reconstructed
            .headers_mut()
            .insert(CONTENT_LENGTH, content_length);
    }
    reconstructed
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
        token_usage: TokenUsage::default(),
        request_metadata: input.request_metadata,
        response_metadata,
        raw_payloads: RawPayloads::default(),
    }
}

fn shutdown_attempt_record(mut record: AttemptRecord) -> AttemptRecord {
    record.status = AttemptStatus::Aborted;
    record.retry_reason = None;
    record.abort_reason = Some(SERVER_SHUTDOWN_ABORT_REASON.to_owned());
    record.response_metadata.insert(
        String::from("abort_reason"),
        SERVER_SHUTDOWN_ABORT_REASON.to_owned(),
    );
    record
}

struct FailedRequestRecord {
    request_id: RequestId,
    started_at_unix_ms: u64,
    finished_at_unix_ms: u64,
    status: RequestStatus,
    http_status: u16,
    error_type: &'static str,
    error_reason: String,
    abort_reason: Option<&'static str>,
    request_metadata: BTreeMap<String, String>,
    attempts: Vec<AttemptRecord>,
}

fn record_queued_admission_cancel(record: QueuedAdmissionCancelRecord) {
    let context = record.context;
    let finished_at_unix_ms = unix_time_millis();
    let queue_wait_ms = duration_millis_u64(record.queued_at.elapsed());
    let error_type = "proxy_generation_queue_cancelled";
    let reason = QueueCancellationReason::DownstreamDisconnected;
    let abort_reason = reason.abort_reason();
    let mut request_metadata = context.request_metadata;
    request_metadata.extend(BTreeMap::from([
        (
            String::from("admission_outcome"),
            reason.admission_outcome().to_owned(),
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
    let persistence_tasks = Arc::clone(&context.persistence_tasks);
    persistence_tasks.spawn_blocking(move || {
        record_observability_many(&context.store, &request_record, &[]);
        log_request_cleanup(
            &request_record,
            failed_request_terminal_reason(
                request_record.status,
                request_record.abort_reason.as_deref(),
            ),
            unix_time_millis().saturating_sub(finished_at_unix_ms),
            false,
        );
    });
}

fn record_failed_request(
    persistence_tasks: &Arc<PersistenceTasks>,
    store: &ObservabilityStore,
    failure: FailedRequestRecord,
) {
    let store = store.clone();
    persistence_tasks.spawn_blocking(move || {
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
            status: failure.status,
            http_status: Some(failure.http_status),
            error_reason: Some(format!("{}: {}", failure.error_type, failure.error_reason)),
            abort_reason: failure.abort_reason.map(str::to_owned),
            request_metadata: failure.request_metadata,
            response_metadata,
            raw_payloads: RawPayloads::default(),
        };
        record_observability_many(&store, &request_record, &failure.attempts);
        log_request_cleanup(
            &request_record,
            failed_request_terminal_reason(
                request_record.status,
                request_record.abort_reason.as_deref(),
            ),
            unix_time_millis().saturating_sub(failure.finished_at_unix_ms),
            false,
        );
    });
}

fn failed_request_terminal_reason(
    status: RequestStatus,
    abort_reason: Option<&str>,
) -> &'static str {
    match (status, abort_reason) {
        (RequestStatus::Aborted, Some("server_shutdown" | "server_shutdown_while_queued")) => {
            "server_shutdown"
        }
        (
            RequestStatus::Aborted,
            Some("downstream_body_dropped_before_eof" | "downstream_disconnected_while_queued"),
        ) => "downstream_disconnect",
        (RequestStatus::Aborted, _) => "aborted",
        (RequestStatus::Failed, _) => "failed",
        (RequestStatus::Succeeded, _) => "succeeded",
    }
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
) -> bool {
    let settings = match context.config.snapshot() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("failed to read evidence settings: {error}");
            return false;
        }
    };
    if !settings.evidence.enabled {
        return false;
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
            true
        }
        Ok(EvidenceStoreWrite::Disabled) => false,
        Err(error) => {
            eprintln!("failed to write evidence ledger: {error}");
            false
        }
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
                        | "loop_failure_policy"
                        | "cot_salvage_used"
                        | "cot_salvage_policy"
                        | "cot_salvage_source_attempt_number"
                        | "cot_salvage_reasoning_prefix_bytes"
                        | "cot_salvage_thinking_budget_tokens"
                        | "shadow_compare_attempt"
                        | "shadow_paired_comparison"
                        | "variant_name"
                        | "shadow_terminal_status"
                )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn maybe_schedule_paired_comparison_after_primary(
    runtime: &ShieldedRetryRuntime,
    request: &RequestRecord,
    attempts: &[AttemptRecord],
) {
    if runtime.downstream_drop_signal.is_dropped() {
        return;
    }
    let settings = match runtime.config.snapshot() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("failed to read paired shadow evidence settings: {error}");
            return;
        }
    };
    let paired = &settings.evidence.shadow.paired_comparison;
    if !settings.evidence.enabled
        || !paired.enabled
        || paired.sample_rate <= 0.0
        || paired.variants.is_empty()
    {
        return;
    }
    if !paired_sample_matches(runtime.request_id.as_str(), paired.sample_rate) {
        return;
    }
    let Some(source) = attempts.iter().find(|attempt| {
        attempt.attempt_number == 1
            && attempt.status == AttemptStatus::Succeeded
            && attempt_shown_to_downstream(request, attempt)
    }) else {
        return;
    };
    for variant in &paired.variants {
        if let Some(plan) = paired_shadow_comparison_attempt_plan(runtime, source, *variant) {
            schedule_shadow_attempt(runtime, source, &settings, plan);
        }
    }
}

fn paired_sample_matches(request_id: &str, sample_rate: f64) -> bool {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    if sample_rate >= 1.0 {
        return true;
    }
    if sample_rate <= 0.0 {
        return false;
    }
    let hash = request_id
        .as_bytes()
        .iter()
        .fold(FNV_OFFSET_BASIS, |current, byte| {
            (current ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        });
    let threshold = paired_sample_threshold(sample_rate);
    threshold > 0 && hash % PAIRED_SAMPLE_DENOMINATOR < threshold
}

fn paired_sample_threshold(sample_rate: f64) -> u64 {
    let rendered = format!("{sample_rate:.6}");
    let mut parts = rendered.split('.');
    let Some(whole) = parts.next() else {
        return 0;
    };
    if whole == "1" {
        return PAIRED_SAMPLE_DENOMINATOR;
    }
    if whole != "0" {
        return 0;
    }
    let fraction = parts.next().unwrap_or_default();
    let mut digits = String::with_capacity(6);
    digits.push_str(fraction);
    while digits.len() < 6 {
        digits.push('0');
    }
    digits.truncate(6);
    digits
        .parse::<u64>()
        .map_or(0, |value| value.min(PAIRED_SAMPLE_DENOMINATOR))
}

fn maybe_schedule_shadow_continuation(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
    source: &AttemptRecord,
) {
    if runtime.downstream_drop_signal.is_dropped() {
        return;
    }
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
    if !settings.evidence.enabled || !shadow.enabled {
        return;
    }
    let plans = shadow_attempt_plans(runtime, failure, source, &settings);
    if plans.is_empty() {
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

    for plan in plans {
        schedule_shadow_attempt(runtime, source, &settings, plan);
    }
}

fn schedule_shadow_attempt(
    runtime: &ShieldedRetryRuntime,
    source: &AttemptRecord,
    settings: &AppConfig,
    plan: ShadowAttemptPlan,
) {
    if runtime.shutdown.is_shutting_down() || runtime.downstream_drop_signal.is_dropped() {
        return;
    }
    let shadow = &settings.evidence.shadow;
    if shadow.max_global_shadow_in_flight == 0 {
        push_shadow_skipped_record(
            &runtime.shadow_evidence,
            runtime,
            source,
            settings,
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
            settings,
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
            settings,
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
            settings,
            ShadowSkipReason::ContinuationUnavailable,
        );
        return;
    };
    let task = ShadowContinuationTask {
        client: runtime.client.clone(),
        method: runtime.method.clone(),
        upstream_url: runtime.upstream_url.clone(),
        downstream_headers: runtime.upstream_headers.clone(),
        upstream_body: plan.upstream_body,
        upstream_timeout: runtime.upstream_timeout,
        evidence_store: runtime.evidence_store.clone(),
        persistence_tasks: Arc::clone(&runtime.persistence_tasks),
        shadow_evidence: runtime.shadow_evidence.clone(),
        shutdown: Arc::clone(&runtime.shutdown),
        request_id: runtime.request_id.clone(),
        group_id: runtime.request_id.as_str().to_owned(),
        attempt_id,
        source: source.clone(),
        request_metadata: plan.request_metadata,
        model_id: runtime.model_id.clone(),
        loop_context: runtime.loop_context.clone(),
        shadow_attempt_timeout_ms: shadow.shadow_attempt_timeout_ms,
        parallel_downgrade_attempts: shadow.parallel_downgrade_attempts,
        comparison_attempt: plan.comparison_attempt,
        _permit: permit,
    };
    let task_guard = runtime.persistence_tasks.track();
    tokio::spawn(async move {
        let _task_guard = task_guard;
        let record = run_shadow_continuation(task).await;
        record.shadow_evidence.push_record(record.attempt.clone());
        let persistence_tasks = Arc::clone(&record.persistence_tasks);
        persistence_tasks.spawn_blocking(move || {
            match record.evidence_store.record_shadow_attempt(&record.attempt) {
                Ok(EvidenceStoreWrite::Written | EvidenceStoreWrite::Disabled) | Err(_) => {}
            }
        });
    });
}

struct ShadowAttemptPlan {
    upstream_body: Bytes,
    request_metadata: BTreeMap<String, String>,
    comparison_attempt: Option<ShadowComparisonAttempt>,
}

fn shadow_attempt_plans(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
    source: &AttemptRecord,
    settings: &AppConfig,
) -> Vec<ShadowAttemptPlan> {
    let shadow = &settings.evidence.shadow;
    let mut plans = Vec::new();
    if shadow.keep_looping_attempt_running {
        plans.push(ShadowAttemptPlan {
            upstream_body: failure.upstream_body.clone(),
            request_metadata: source.request_metadata.clone(),
            comparison_attempt: None,
        });
    }
    for comparison in &shadow.compare_attempts {
        if let Some(plan) = shadow_comparison_attempt_plan(runtime, failure, source, *comparison) {
            plans.push(plan);
        }
    }
    plans
}

fn shadow_comparison_attempt_plan(
    runtime: &ShieldedRetryRuntime,
    failure: &ShieldedAttemptFailure,
    source: &AttemptRecord,
    comparison: ShadowComparisonAttempt,
) -> Option<ShadowAttemptPlan> {
    let thinking = shadow_comparison_thinking(comparison, &runtime.upstream_profile.thinking);
    let mut prepared = prepared_shadow_body(runtime, &thinking);
    let mut upstream_body = prepared.upstream_body;
    if comparison == ShadowComparisonAttempt::CotSalvage {
        let reasoning = failure.raw_payloads.reasoning.as_deref()?;
        let reasoning_prefix = bounded_utf8_prefix(reasoning, COT_SALVAGE_PREFIX_MAX_BYTES);
        if reasoning_prefix.trim().is_empty() {
            return None;
        }
        upstream_body = shielded_chat::body_with_cot_salvage_retry_hint(
            &upstream_body,
            source.attempt_number.saturating_add(1),
            runtime.retry_policy.max_attempts,
            comparison.as_str(),
            &reasoning_prefix,
            None,
        )?;
    }
    upstream_body = apply_shielded_param_override_to_body_or_original(
        upstream_body,
        &runtime.upstream_profile,
        &mut prepared.thinking_metadata,
    );
    upstream_body = render_shadow_endpoint_body(runtime, &upstream_body)?;
    let mut request_metadata = source.request_metadata.clone();
    request_metadata.extend(prepared.thinking_metadata);
    request_metadata.insert(
        String::from("shadow_compare_attempt"),
        comparison.as_str().to_owned(),
    );
    request_metadata.insert(
        String::from("attempt_thinking_mode"),
        thinking.effective_mode().as_str().to_owned(),
    );
    request_metadata.insert(
        String::from("attempt_thinking_budget_tokens"),
        thinking.budget_tokens.to_string(),
    );
    request_metadata.insert(
        String::from("attempt_thinking_max_tokens"),
        thinking
            .max_tokens
            .map_or_else(|| String::from("unset"), |value| value.to_string()),
    );
    Some(ShadowAttemptPlan {
        upstream_body,
        request_metadata,
        comparison_attempt: Some(comparison),
    })
}

fn paired_shadow_comparison_attempt_plan(
    runtime: &ShieldedRetryRuntime,
    source: &AttemptRecord,
    comparison: ShadowComparisonAttempt,
) -> Option<ShadowAttemptPlan> {
    if comparison == ShadowComparisonAttempt::CotSalvage {
        return None;
    }
    let thinking = shadow_comparison_thinking(comparison, &runtime.upstream_profile.thinking);
    let mut prepared = prepared_shadow_body(runtime, &thinking);
    let upstream_body = apply_shielded_param_override_to_body_or_original(
        prepared.upstream_body,
        &runtime.upstream_profile,
        &mut prepared.thinking_metadata,
    );
    let upstream_body = render_shadow_endpoint_body(runtime, &upstream_body)?;
    let mut request_metadata = source.request_metadata.clone();
    request_metadata.extend(prepared.thinking_metadata);
    request_metadata.insert(
        String::from("shadow_compare_attempt"),
        comparison.as_str().to_owned(),
    );
    request_metadata.insert(
        String::from("shadow_paired_comparison"),
        String::from("true"),
    );
    request_metadata.insert(String::from("variant_name"), comparison.as_str().to_owned());
    request_metadata.insert(
        String::from("attempt_thinking_mode"),
        thinking.effective_mode().as_str().to_owned(),
    );
    request_metadata.insert(
        String::from("attempt_thinking_budget_tokens"),
        thinking.budget_tokens.to_string(),
    );
    request_metadata.insert(
        String::from("attempt_thinking_max_tokens"),
        thinking
            .max_tokens
            .map_or_else(|| String::from("unset"), |value| value.to_string()),
    );
    Some(ShadowAttemptPlan {
        upstream_body,
        request_metadata,
        comparison_attempt: Some(comparison),
    })
}

struct PreparedShadowBody {
    upstream_body: Bytes,
    thinking_metadata: BTreeMap<String, String>,
}

fn render_shadow_endpoint_body(runtime: &ShieldedRetryRuntime, body: &Bytes) -> Option<Bytes> {
    reranker_protocol::render_openai_endpoint(
        &runtime.terminal_endpoint,
        runtime.forward_uri.clone(),
        body,
        &runtime.original_downstream_headers,
        runtime.transformed_request_headers,
    )
    .ok()
    .map(|rendered| rendered.body)
}

fn prepared_shadow_body(
    runtime: &ShieldedRetryRuntime,
    thinking: &ThinkingConfig,
) -> PreparedShadowBody {
    let prepared = match runtime.chat_kind {
        ShieldedChatKind::NonStream => {
            shielded_chat::prepare_non_stream_request(&runtime.downstream_body, thinking)
        }
        ShieldedChatKind::Stream => {
            shielded_chat::prepare_stream_request(&runtime.downstream_body, thinking)
        }
        ShieldedChatKind::Generic => None,
    };
    prepared.map_or_else(
        || PreparedShadowBody {
            upstream_body: runtime.upstream_body.clone(),
            thinking_metadata: runtime.thinking_metadata.clone(),
        },
        |request| PreparedShadowBody {
            upstream_body: request.upstream_body(),
            thinking_metadata: request.thinking_metadata().clone(),
        },
    )
}

fn shadow_comparison_thinking(
    comparison: ShadowComparisonAttempt,
    current: &ThinkingConfig,
) -> ThinkingConfig {
    let mut thinking = current.clone();
    match comparison {
        ShadowComparisonAttempt::MaxThinking => {
            thinking.mode = ThinkingMode::ForceThinking;
            thinking.enabled = true;
            thinking.force_disable = false;
            thinking.budget_tokens = thinking.budget_tokens.max(32_768);
            thinking.preserve_answer_budget = true;
        }
        ShadowComparisonAttempt::BoundedThinking | ShadowComparisonAttempt::CotSalvage => {
            thinking.mode = ThinkingMode::BoundedThinking;
            thinking.enabled = true;
            thinking.force_disable = false;
            thinking.budget_tokens = COT_SALVAGE_THINKING_BUDGET_TOKENS;
            thinking.preserve_answer_budget = false;
        }
        ShadowComparisonAttempt::NoThinking => {
            thinking.mode = ThinkingMode::ForceDisable;
            thinking.enabled = false;
            thinking.force_disable = true;
            thinking.budget_tokens = 0;
            thinking.preserve_answer_budget = false;
        }
    }
    thinking
}

struct ShadowContinuationTask {
    client: Client,
    method: reqwest::Method,
    upstream_url: Url,
    downstream_headers: HeaderMap,
    upstream_body: Bytes,
    upstream_timeout: Duration,
    evidence_store: EvidenceStore,
    persistence_tasks: Arc<PersistenceTasks>,
    shadow_evidence: ShadowEvidenceState,
    shutdown: Arc<ShutdownGate>,
    request_id: RequestId,
    group_id: String,
    attempt_id: AttemptId,
    source: AttemptRecord,
    request_metadata: BTreeMap<String, String>,
    model_id: Option<String>,
    loop_context: shielded_chat::LoopInspectionContext,
    shadow_attempt_timeout_ms: u64,
    parallel_downgrade_attempts: bool,
    comparison_attempt: Option<ShadowComparisonAttempt>,
    _permit: InFlightPermit,
}

struct ShadowContinuationRecord {
    evidence_store: EvidenceStore,
    persistence_tasks: Arc<PersistenceTasks>,
    shadow_evidence: ShadowEvidenceState,
    attempt: EvidenceAttemptRecord,
}

async fn run_shadow_continuation(task: ShadowContinuationTask) -> ShadowContinuationRecord {
    let evidence_store = task.evidence_store.clone();
    let persistence_tasks = Arc::clone(&task.persistence_tasks);
    let shadow_evidence = task.shadow_evidence.clone();
    let started_at_unix_ms = unix_time_millis();
    let timeout_ms = task.shadow_attempt_timeout_ms;
    let mut shutdown = task.shutdown.subscribe();
    let attempt = tokio::select! {
        biased;
        () = shutdown.cancelled() => build_shadow_terminal_record(
            &task,
            ShadowTerminalInput {
                started_at_unix_ms,
                finished_at_unix_ms: unix_time_millis(),
                status: EvidenceAttemptStatus::Aborted,
                http_status: None,
                response_headers: HeaderMap::new(),
                response_metadata: BTreeMap::from([(
                    String::from("abort_reason"),
                    SERVER_SHUTDOWN_ABORT_REASON.to_owned(),
                )]),
                raw_payloads: RawPayloads::default(),
                response_body_bytes: 0,
                error_reason: Some(String::from("shadow continuation cancelled by server shutdown")),
                abort_reason: Some(SERVER_SHUTDOWN_ABORT_REASON.to_owned()),
            },
        ),
        outcome = timeout(
            Duration::from_millis(timeout_ms),
            run_shadow_continuation_request(&task, started_at_unix_ms),
        ) => match outcome {
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
        },
    };
    ShadowContinuationRecord {
        evidence_store,
        persistence_tasks,
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
        task.comparison_attempt,
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
        upstream_profile: metadata_value(&task.request_metadata, "upstream_profile"),
        model_id: task.model_id.clone(),
        thinking_mode: metadata_value(&task.request_metadata, "attempt_thinking_mode"),
        thinking_budget_tokens: metadata_u32(
            &task.request_metadata,
            "attempt_thinking_budget_tokens",
        ),
        thinking_max_tokens: metadata_u32(&task.request_metadata, "attempt_thinking_max_tokens"),
        detector_features: evidence_detector_features(&task.source.response_metadata),
        status: input.status,
        http_status: input.http_status,
        error_reason: input.error_reason,
        retry_reason: None,
        abort_reason: input.abort_reason,
        shadow_skip_reason: None,
        request_metadata: task.request_metadata.clone(),
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
        None,
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
    comparison_attempt: Option<ShadowComparisonAttempt>,
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
    if let Some(comparison_attempt) = comparison_attempt {
        metadata.insert(
            String::from("shadow_compare_attempt"),
            comparison_attempt.as_str().to_owned(),
        );
    }
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
    proxy_error_response_with_diagnostics(status, error_type, message, code, param, None, None)
}

fn proxy_error_response_with_diagnostics(
    status: StatusCode,
    error_type: &str,
    message: &str,
    code: Option<&str>,
    param: Option<&str>,
    cause_code: Option<&str>,
    request_id: Option<&RequestId>,
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
    if let Some(cause) = cause_code {
        error.insert(String::from("cause"), json!(cause));
        // Ensure a stable `code` is present for OpenAI-style clients that key
        // off `code` rather than `cause`.
        error
            .entry(String::from("code"))
            .or_insert_with(|| json!(cause));
    }
    if let Some(request_id) = request_id {
        error.insert(String::from("request_id"), json!(request_id.as_str()));
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
    if let Some(request_id) = request_id
        && let Ok(value) = HeaderValue::from_str(request_id.as_str())
    {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-request-id"), value);
    }
    response
}

fn proxy_error_response_from_error_with_diagnostics(
    error: &ProxyError,
    request_id: Option<&RequestId>,
) -> Response<Body> {
    let cause_code = error
        .upstream_failure_cause()
        .map(UpstreamFailureCause::code);
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
        } => proxy_error_response_with_diagnostics(
            error.status(),
            error.error_type(),
            message,
            Some(code),
            Some(param),
            cause_code,
            request_id,
        ),
        _ => proxy_error_response_with_diagnostics(
            error.status(),
            error.error_type(),
            &error.to_string(),
            None,
            None,
            cause_code,
            request_id,
        ),
    }
}

fn admission_error_response(
    status: StatusCode,
    error_type: &str,
    message: &str,
    retry_after: Option<String>,
) -> Response<Body> {
    let mut response = proxy_error_response(status, error_type, message);
    if let Some(retry_after) = retry_after
        && let Ok(value) = HeaderValue::from_str(&retry_after)
    {
        response.headers_mut().insert(RETRY_AFTER, value);
    }
    response
}

fn unix_time_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(feature = "guard")]
fn unix_time_secs() -> u64 {
    unix_time_millis() / 1_000
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
    #[error("proxy is shutting down")]
    Shutdown {
        request_metadata: Option<BTreeMap<String, String>>,
        attempts: Vec<AttemptRecord>,
    },
    #[cfg(feature = "guard")]
    #[error("guard workflow blocked request: {reason}")]
    GuardBlocked {
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[cfg(feature = "guard")]
    #[error("virtual key is required or not recognized")]
    VirtualKeyUnauthorized {
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[cfg(feature = "guard")]
    #[error(
        "daily request budget exceeded for profile {profile}: count={current_count} limit={limit} date={date}"
    )]
    BudgetExceeded {
        profile: String,
        date: String,
        current_count: u64,
        limit: u64,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[cfg(feature = "guard")]
    #[error("budget store failed: {reason}")]
    BudgetStoreFailure {
        reason: String,
        request_metadata: Option<BTreeMap<String, String>>,
    },
    #[error("no healthy upstream endpoint for profile {profile} after {waited_ms}ms")]
    UpstreamUnavailable {
        profile: String,
        waited_ms: u64,
        request_metadata: Option<BTreeMap<String, String>>,
        attempts: Vec<AttemptRecord>,
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
        attempts: Vec<AttemptRecord>,
    },
}

fn merge_request_metadata(
    existing: Option<BTreeMap<String, String>>,
    additional: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut merged = existing.unwrap_or_default();
    merged.extend(additional);
    merged
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

    fn server_shutdown() -> Self {
        Self::Shutdown {
            request_metadata: None,
            attempts: Vec::new(),
        }
    }

    fn upstream_unavailable(profile: String, waited_ms: u64) -> Self {
        Self::UpstreamUnavailable {
            profile,
            waited_ms,
            request_metadata: None,
            attempts: Vec::new(),
        }
    }

    fn upstream_body(reason: String) -> Self {
        Self::UpstreamBody {
            reason,
            observability: None,
        }
    }

    #[cfg(feature = "guard")]
    fn guard_blocked(reason: String) -> Self {
        Self::GuardBlocked {
            reason,
            request_metadata: None,
        }
    }

    #[cfg(feature = "guard")]
    fn virtual_key_unauthorized() -> Self {
        Self::VirtualKeyUnauthorized {
            request_metadata: Some(BTreeMap::from([(
                String::from("virtual_key_resolution"),
                String::from("fail_closed"),
            )])),
        }
    }

    #[cfg(feature = "guard")]
    fn budget_exceeded(profile: String, date: String, current_count: u64, limit: u64) -> Self {
        Self::BudgetExceeded {
            request_metadata: Some(BTreeMap::from([
                (String::from("caller_profile"), profile.clone()),
                (String::from("budget_date"), date.clone()),
                (String::from("budget_count"), current_count.to_string()),
                (String::from("budget_limit"), limit.to_string()),
                (
                    String::from("budget_outcome"),
                    String::from("limit_exceeded"),
                ),
            ])),
            profile,
            date,
            current_count,
            limit,
        }
    }

    #[cfg(feature = "guard")]
    fn budget_store_failed(reason: &BudgetError) -> Self {
        Self::BudgetStoreFailure {
            reason: reason.to_string(),
            request_metadata: None,
        }
    }

    fn context_budget_exceeded(estimate: ContextBudgetEstimate) -> Self {
        Self::ContextBudgetExceeded {
            message: estimate.message(),
            param: estimate.param,
            code: "context_budget_exceeded",
            request_metadata: Some(estimate.metadata("rejected")),
            attempts: Vec::new(),
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
            Self::Shutdown { .. } | Self::UpstreamUnavailable { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            #[cfg(feature = "guard")]
            Self::VirtualKeyUnauthorized { .. } => StatusCode::UNAUTHORIZED,
            #[cfg(feature = "guard")]
            Self::BudgetExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
            #[cfg(feature = "guard")]
            Self::BudgetStoreFailure { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            #[cfg(feature = "guard")]
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
            Self::Shutdown { .. } => PROXY_SHUTTING_DOWN_ERROR_TYPE,
            Self::UpstreamUnavailable { .. } => "upstream_unavailable",
            #[cfg(feature = "guard")]
            Self::VirtualKeyUnauthorized { .. } => "virtual_key_unauthorized",
            #[cfg(feature = "guard")]
            Self::BudgetExceeded { .. } => "budget_exhausted",
            #[cfg(feature = "guard")]
            Self::BudgetStoreFailure { .. } => "budget_store_failed",
            #[cfg(feature = "guard")]
            Self::GuardBlocked { .. } => "guard_blocked",
            Self::ContextBudgetExceeded { .. } => "invalid_request_error",
            Self::UpstreamTransport { .. } => "upstream_transport_error",
            Self::UpstreamBody { .. } => "upstream_body_error",
        }
    }

    const fn request_status(&self) -> RequestStatus {
        match self {
            Self::Admission { failure, .. } => failure.request_status(),
            Self::Shutdown { .. } => RequestStatus::Aborted,
            Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::UpstreamUnavailable { .. }
            | Self::UpstreamTransport { .. }
            | Self::UpstreamBody { .. }
            | Self::ContextBudgetExceeded { .. } => RequestStatus::Failed,
            #[cfg(feature = "guard")]
            Self::GuardBlocked { .. }
            | Self::VirtualKeyUnauthorized { .. }
            | Self::BudgetExceeded { .. }
            | Self::BudgetStoreFailure { .. } => RequestStatus::Failed,
        }
    }

    const fn abort_reason(&self) -> Option<&'static str> {
        match self {
            Self::Admission { failure, .. } => failure.abort_reason(),
            Self::Shutdown { .. } => Some(SERVER_SHUTDOWN_ABORT_REASON),
            Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::UpstreamUnavailable { .. }
            | Self::UpstreamTransport { .. }
            | Self::UpstreamBody { .. }
            | Self::ContextBudgetExceeded { .. } => None,
            #[cfg(feature = "guard")]
            Self::GuardBlocked { .. }
            | Self::VirtualKeyUnauthorized { .. }
            | Self::BudgetExceeded { .. }
            | Self::BudgetStoreFailure { .. } => None,
        }
    }

    /// Returns the stable cause-bucket `code` for a 502 upstream failure, or
    /// `None` for non-upstream errors. Used in the error JSON body and to select
    /// the cause label for `llm_guard_proxy_upstream_failure_total`.
    const fn upstream_failure_cause(&self) -> Option<UpstreamFailureCause> {
        match self {
            Self::UpstreamTransport { failure, .. } => {
                Some(UpstreamFailureCause::from_reqwest_failure(*failure))
            }
            Self::UpstreamBody { .. } => Some(UpstreamFailureCause::BodyError),
            _ => None,
        }
    }

    fn request_metadata(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::UpstreamTransport {
                observability: Some(observability),
                ..
            }
            | Self::UpstreamBody {
                observability: Some(observability),
                ..
            } => Some(&observability.request_metadata),
            Self::UpstreamTransport {
                observability: None,
                ..
            }
            | Self::UpstreamBody {
                observability: None,
                ..
            }
            | Self::InvalidRequestPath(_) => None,
            _ => self.direct_request_metadata(),
        }
    }

    fn direct_request_metadata(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::RequestBody {
                request_metadata, ..
            }
            | Self::ConfigSnapshot {
                request_metadata, ..
            }
            | Self::InvalidUpstreamUrl {
                request_metadata, ..
            }
            | Self::InvalidMethod {
                request_metadata, ..
            }
            | Self::Admission {
                request_metadata, ..
            }
            | Self::ListenerUpstreamDenied {
                request_metadata, ..
            }
            | Self::UpstreamUnavailable {
                request_metadata, ..
            }
            | Self::Shutdown {
                request_metadata, ..
            }
            | Self::ContextBudgetExceeded {
                request_metadata, ..
            } => request_metadata.as_ref(),
            #[cfg(feature = "guard")]
            Self::GuardBlocked {
                request_metadata, ..
            } => request_metadata.as_ref(),
            #[cfg(feature = "guard")]
            Self::VirtualKeyUnauthorized { request_metadata } => request_metadata.as_ref(),
            #[cfg(feature = "guard")]
            Self::BudgetExceeded {
                request_metadata, ..
            }
            | Self::BudgetStoreFailure {
                request_metadata, ..
            } => request_metadata.as_ref(),
            Self::InvalidRequestPath(_)
            | Self::UpstreamTransport { .. }
            | Self::UpstreamBody { .. } => None,
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
            Self::Shutdown { attempts, .. }
            | Self::UpstreamUnavailable { attempts, .. }
            | Self::ContextBudgetExceeded { attempts, .. } => attempts.clone(),
            Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::Admission { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::UpstreamTransport {
                observability: None,
                ..
            }
            | Self::UpstreamBody {
                observability: None,
                ..
            } => Vec::new(),
            #[cfg(feature = "guard")]
            Self::GuardBlocked { .. }
            | Self::VirtualKeyUnauthorized { .. }
            | Self::BudgetExceeded { .. }
            | Self::BudgetStoreFailure { .. } => Vec::new(),
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
            Self::UpstreamUnavailable {
                profile,
                waited_ms,
                attempts,
                ..
            } => Self::UpstreamUnavailable {
                profile,
                waited_ms,
                request_metadata: Some(request_metadata),
                attempts,
            },
            Self::Shutdown { attempts, .. } => Self::Shutdown {
                request_metadata: Some(request_metadata),
                attempts,
            },
            #[cfg(feature = "guard")]
            Self::GuardBlocked { reason, .. } => Self::GuardBlocked {
                reason,
                request_metadata: Some(request_metadata),
            },
            #[cfg(feature = "guard")]
            Self::VirtualKeyUnauthorized {
                request_metadata: existing_metadata,
            } => {
                let mut merged = existing_metadata.unwrap_or_default();
                merged.extend(request_metadata);
                Self::VirtualKeyUnauthorized {
                    request_metadata: Some(merged),
                }
            }
            Self::ContextBudgetExceeded {
                message,
                param,
                code,
                request_metadata: existing_metadata,
                attempts,
            } => Self::ContextBudgetExceeded {
                message,
                param,
                code,
                request_metadata: Some(merge_request_metadata(existing_metadata, request_metadata)),
                attempts,
            },
            #[cfg(feature = "guard")]
            Self::BudgetExceeded {
                profile,
                date,
                current_count,
                limit,
                request_metadata: existing_metadata,
            } => {
                let mut merged = existing_metadata.unwrap_or_default();
                merged.extend(request_metadata);
                Self::BudgetExceeded {
                    profile,
                    date,
                    current_count,
                    limit,
                    request_metadata: Some(merged),
                }
            }
            #[cfg(feature = "guard")]
            Self::BudgetStoreFailure { reason, .. } => Self::BudgetStoreFailure {
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
            Self::Shutdown { .. } => Self::Shutdown {
                request_metadata: Some(request_metadata),
                attempts: vec![shutdown_attempt_record(attempt_record)],
            },
            error @ (Self::RequestBody { .. }
            | Self::ConfigSnapshot { .. }
            | Self::InvalidUpstreamUrl { .. }
            | Self::InvalidRequestPath(_)
            | Self::InvalidMethod { .. }
            | Self::Admission { .. }
            | Self::ListenerUpstreamDenied { .. }
            | Self::UpstreamUnavailable { .. }
            | Self::ContextBudgetExceeded { .. }) => error,
            #[cfg(feature = "guard")]
            error @ (Self::GuardBlocked { .. }
            | Self::VirtualKeyUnauthorized { .. }
            | Self::BudgetExceeded { .. }
            | Self::BudgetStoreFailure { .. }) => error,
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
            Self::Shutdown {
                request_metadata,
                mut attempts,
            } => {
                attempts.splice(0..0, completed_attempt_records);
                Self::Shutdown {
                    request_metadata,
                    attempts,
                }
            }
            Self::UpstreamUnavailable {
                profile,
                waited_ms,
                request_metadata,
                mut attempts,
            } => {
                attempts.splice(0..0, completed_attempt_records);
                Self::UpstreamUnavailable {
                    profile,
                    waited_ms,
                    request_metadata,
                    attempts,
                }
            }
            Self::ContextBudgetExceeded {
                message,
                param,
                code,
                request_metadata,
                mut attempts,
            } => {
                attempts.splice(0..0, completed_attempt_records);
                Self::ContextBudgetExceeded {
                    message,
                    param,
                    code,
                    request_metadata,
                    attempts,
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
#[path = "proxy/persistence_flush_tests.rs"]
mod persistence_flush_tests;
#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "guard"))]
#[path = "proxy/workflow_admission_tests.rs"]
mod workflow_admission_tests;
