#![forbid(unsafe_code)]
//! Operational state and persistence for the `llm-guard-proxy` service.
//!
//! This crate owns observability, evidence, and budget state. It depends on
//! validated configuration contracts from `llm-guard-proxy-core`; the core
//! crate does not depend on this crate.

#[cfg(feature = "guard")]
pub mod budget;
mod config_paths;
pub mod evidence;
mod observability;

#[cfg(feature = "guard")]
pub use budget::{BudgetCheck, BudgetError, BudgetStore, current_budget_date};
pub use config_paths::{materialize_evidence_path_defaults, preflight_evidence_paths};
pub use evidence::{
    EvidenceAttemptRecord, EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceDatabaseStatus,
    EvidenceError, EvidenceExportArtifact, EvidenceExportPair, EvidenceGroupRecord,
    EvidencePruningStats, EvidenceRawArtifactKind, EvidenceRetentionUsage, EvidenceShadowRecord,
    EvidenceStore, EvidenceStoreWrite, EvidenceSummaryRow, ShadowSkipReason,
};
pub use observability::{
    AttemptId, AttemptMetricCount, AttemptRecord, AttemptStatus, DebugRequestSummary,
    DownstreamMode, HeartbeatModeMetricCount, HistogramBucket, LatencyHistogram, LiveRequestEntry,
    LiveRequestRegistry, LiveRequestState, LiveRequestSummary, ObservabilityError,
    ObservabilityMetricsSnapshot, ObservabilityStore, RawPayloadChunk, RawPayloads, RequestId,
    RequestMetricCount, RequestRecord, RequestStatus, RequestTerminalMetricCount,
    RetentionPruningStats, RetentionUsage, StoreWrite, TimelineEvent, UpstreamErrorMetricCount,
    UpstreamMode,
};
