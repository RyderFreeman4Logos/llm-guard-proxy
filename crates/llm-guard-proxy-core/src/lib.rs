#![forbid(unsafe_code)]
//! Headless core types for the `llm-guard-proxy` service.
//!
//! Issue #1 intentionally kept this crate small. Later issues add proxy,
//! retry, and request-shielding behavior behind core interfaces.

#[cfg(feature = "guard")]
pub mod budget;
pub mod context_rot;
pub mod embedding;
pub mod evidence;
#[cfg(feature = "family")]
pub mod family;
#[cfg(feature = "guard")]
mod guard;
#[cfg(feature = "guard")]
pub mod gwp;
mod loop_detector;
#[cfg(feature = "guard")]
mod model_alias;
pub mod model_judge;
mod observability;
#[cfg(feature = "guard")]
pub mod profile;
pub mod replay;
pub mod risk_combiner;
mod settings;
#[cfg(feature = "guard")]
pub mod workflow;

#[cfg(feature = "guard")]
pub use budget::{BudgetCheck, BudgetError, BudgetStore, current_budget_date};
pub use context_rot::{
    ContextChunk, ContextRotConfig, ContextRotScorer, ContextRotSignal, DEFAULT_CONTENT_WEIGHT,
    DEFAULT_ECHO_REPEAT_COUNT, DEFAULT_ECHO_SIMILARITY_THRESHOLD, DEFAULT_MAX_CONTEXT_CHUNKS,
    DEFAULT_REASONING_WEIGHT, DEFAULT_TOOL_ARGS_WEIGHT, DEFAULT_TOOL_OUTPUT_ECHO_WEIGHT,
};
pub use embedding::{
    CONTENT_SIMILARITY_THRESHOLD, DisabledEmbeddingBackend, EmbeddingBackend, EmbeddingChannel,
    EmbeddingError, EmbeddingFuture, EmbeddingInput, EmbeddingQueue, EmbeddingQueueResult,
    EmbeddingVector, MIN_OBSERVATIONS_FOR_SIGNAL, REASONING_SIMILARITY_THRESHOLD,
    SemanticLoopConfig, SemanticLoopScorer, SemanticLoopSignal, TOOL_ARGS_SIMILARITY_THRESHOLD,
};
pub use evidence::{
    EvidenceAttemptRecord, EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceDatabaseStatus,
    EvidenceError, EvidenceExportArtifact, EvidenceExportPair, EvidenceGroupRecord,
    EvidencePruningStats, EvidenceRawArtifactKind, EvidenceRetentionUsage, EvidenceShadowRecord,
    EvidenceStore, EvidenceStoreWrite, EvidenceSummaryRow, ShadowSkipReason,
};
#[cfg(feature = "family")]
pub use family::{
    CHILD_SAFE_DAILY_REQUEST_LIMIT, CHILD_SAFE_MODEL_ALIAS, CHILD_SAFE_PROFILE_NAME,
    CategoryAction, CategoryConfig, FAMILY_GUARD_PACK_NAME, FamilyCategory, FamilyPolicyConfig,
    FamilyPolicyOutcome, FamilyPolicyWarning,
};
#[cfg(feature = "guard")]
pub use guard::{GuardExecutor, GuardOutcome};
#[cfg(feature = "guard")]
pub use gwp::{
    GWP_PROTOCOL_VERSION, GwpAudit, GwpDecision, GwpHook, GwpInvocation, GwpProfile,
    GwpProfileKind, GwpResult, GwpTraceMode,
};
pub use loop_detector::{
    BoundedFeatureSummary, ChannelizedLoopDetector, DetectorEventKind, DetectorSummary,
    LoopDetector, LoopDetectorInput, LoopInputProfile, LoopReasonCode, LoopSeverity, LoopSignal,
    StreamChannel, ToolCallFingerprintInput, ToolFingerprint, ToolLoopDetector, ToolLoopSignal,
};
#[cfg(feature = "guard")]
pub use model_alias::{
    AliasKind, AliasResolutionError, AliasTarget, DEFAULT_WORKFLOW_TIMEOUT_MS,
    MAX_WORKFLOW_TIMEOUT_MS, ModelAliasConfig, ModelAliasResolver,
};
pub use model_judge::{
    AnswerCandidate, ChannelMetrics, ChannelSnapshot, CleanReasoningState, JudgePromptBuilder,
    JudgeSeverity, JudgeSnapshot, LoopJudgeResult, LoopType, ProvenanceFact, RecommendedAction,
    SnapshotChannels, TaskKind, ToolState, WindowSpan,
};
pub use observability::{
    AttemptId, AttemptMetricCount, AttemptRecord, AttemptStatus, DebugRequestSummary,
    DownstreamMode, HeartbeatModeMetricCount, HistogramBucket, LatencyHistogram, LiveRequestEntry,
    LiveRequestRegistry, LiveRequestState, LiveRequestSummary, ObservabilityError,
    ObservabilityMetricsSnapshot, ObservabilityStore, RawPayloadChunk, RawPayloads, RequestId,
    RequestMetricCount, RequestRecord, RequestStatus, RetentionPruningStats, RetentionUsage,
    StoreWrite, TimelineEvent, UpstreamErrorMetricCount, UpstreamMode,
};
#[cfg(feature = "guard")]
pub use profile::{
    BlockReason, DEFAULT_PROFILE_NAME, ProfileCheckResult, ProfileConfig, ProfileKind,
    ShieldedBuffering,
};
pub use replay::{
    CalibrationResult, RecordDetectionResult, ReplayChannel, ReplayConfig, ReplayRecord,
    ReplayRunner, SEVERITY_HARD, SEVERITY_MILD, SEVERITY_NONE, SourceCalibration, SseEvent,
};
pub use risk_combiner::{CombinedRisk, DetectorKind, DetectorSignal, RiskCombiner};
#[cfg(feature = "param-override")]
pub use settings::ParamOverrideConfig;
pub use settings::{
    AppConfig, CloudflareConfig, ConfigError, ConfigHandle, ConfigManager, ConfigParseError,
    DEFAULT_CONFIG_RELATIVE_PATH, DefaultInjectionSchema, DownstreamDropPolicy, EvidenceConfig,
    EvidencePairedComparisonConfig, EvidenceShadowConfig, HeartbeatConfig, HeartbeatMode,
    HotRestartConfig, ListenerConfig, LoopFailurePolicy, LoopGuardConfig, LoopGuardMode,
    MetadataConfig, MissingConfigPolicy, NoThinkingMarkerPolicy, ObservabilityConfig,
    RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS, ReloadOutcome, ReloadWatcher,
    RestartRequiredChange, RetentionConfig, RetryConfig, RetryLadderConfig,
    SelectedUpstreamProfile, ServerConfig, ShadowComparisonAttempt, ShieldingConfig,
    ThinkingConfig, ThinkingMode, ToolRequestThinkingPolicy, UpstreamConfig, UpstreamProfileConfig,
    UpstreamRouteReason, UpstreamStallConfig, ValidationError, default_config_path,
    redact_upstream_base_url, validate_upstream_base_url,
};
#[cfg(feature = "guard")]
pub use settings::{BudgetConfig, UnknownKeyPolicy, VirtualKeyConfig};
#[cfg(feature = "guard")]
pub use workflow::{StdioRuntime, WorkflowConfig, WorkflowRuntime};

/// Public service name used by the binary and documentation.
pub const SERVICE_NAME: &str = "llm-guard-proxy";

/// Repository license identifier.
pub const LICENSE: &str = "Apache-2.0";

/// Current process readiness state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Readiness {
    /// The placeholder process has started and can answer health checks.
    Ready,
}

impl Readiness {
    /// Returns the stable wire/display label for the readiness state.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
        }
    }
}

/// Minimal health model shared by the service entry point and tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Health {
    readiness: Readiness,
}

impl Health {
    /// Builds the current placeholder health response.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            readiness: Readiness::Ready,
        }
    }

    /// Returns the process readiness value.
    #[must_use]
    pub const fn readiness(self) -> Readiness {
        self.readiness
    }

    /// Returns whether this process is ready to accept health checks.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self.readiness, Readiness::Ready)
    }
}

impl Default for Health {
    fn default() -> Self {
        Self::current()
    }
}

#[cfg(test)]
mod tests {
    use super::{Health, LICENSE, Readiness, SERVICE_NAME};

    #[test]
    fn health_defaults_to_ready() {
        let health = Health::default();

        assert!(health.is_ready());
        assert_eq!(health.readiness(), Readiness::Ready);
        assert_eq!(health.readiness().as_str(), "ready");
    }

    #[test]
    fn service_metadata_matches_repository_contract() {
        assert_eq!(SERVICE_NAME, "llm-guard-proxy");
        assert_eq!(LICENSE, "Apache-2.0");
    }
}
