use std::collections::BTreeMap;

use serde::Serialize;

use crate::{AttemptId, RawPayloads, RequestId};

/// Role of an upstream attempt inside one correlated evidence group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceAttemptRole {
    /// Initial attempt under the configured primary thinking policy.
    Primary,
    /// Retry ladder attempt that may be shown to downstream.
    Fallback,
    /// Evidence-only record for a looped attempt that would have continued.
    ShadowContinued,
}

impl EvidenceAttemptRole {
    /// Returns the stable storage label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Fallback => "fallback",
            Self::ShadowContinued => "shadow_continued",
        }
    }
}

/// Terminal status for an evidence attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceAttemptStatus {
    /// Attempt completed and was accepted by the shielded policy.
    Accepted,
    /// Attempt was rejected and another attempt was scheduled.
    Rejected,
    /// Attempt failed without producing an accepted response.
    Failed,
    /// Attempt was aborted or cancelled.
    Aborted,
    /// Shadow continuation was intentionally skipped by policy or limits.
    Skipped,
    /// Shadow continuation exceeded its configured timeout.
    ShadowTimeout,
}

impl EvidenceAttemptStatus {
    /// Returns the stable storage label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
            Self::Skipped => "skipped",
            Self::ShadowTimeout => "shadow_timeout",
        }
    }
}

/// Reason an evidence-only shadow continuation was skipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowSkipReason {
    /// Shadow collection is disabled by config.
    Disabled,
    /// The request already used its configured shadow budget.
    PerRequestLimit,
    /// The global in-flight shadow limit was already reached.
    GlobalLimit,
    /// Continuing the original upstream stream is not implemented in this slice.
    ContinuationUnavailable,
}

impl ShadowSkipReason {
    /// Returns the stable storage label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PerRequestLimit => "per_request_limit",
            Self::GlobalLimit => "global_limit",
            Self::ContinuationUnavailable => "continuation_unavailable",
        }
    }
}

/// Correlated evidence group for one downstream request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceGroupRecord {
    /// Stable evidence group id.
    pub group_id: String,
    /// Proxy request id that produced this group.
    pub request_id: RequestId,
    /// Request start time in Unix milliseconds.
    pub started_at_unix_ms: u64,
    /// Request finish time in Unix milliseconds.
    pub finished_at_unix_ms: Option<u64>,
    /// OpenAI-compatible request model id, if known.
    pub model_id: Option<String>,
    /// Final downstream status label.
    pub status: String,
    /// Redacted request-side metadata.
    pub request_metadata: BTreeMap<String, String>,
    /// Redacted response-side metadata.
    pub response_metadata: BTreeMap<String, String>,
}

/// Evidence row for one primary, fallback, or shadow attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceAttemptRecord {
    /// Stable attempt id.
    pub attempt_id: AttemptId,
    /// Parent evidence group id.
    pub group_id: String,
    /// Parent proxy request id.
    pub request_id: RequestId,
    /// One-based upstream attempt number.
    pub attempt_number: u32,
    /// Attempt role in this evidence group.
    pub role: EvidenceAttemptRole,
    /// Whether any bytes from this attempt were shown to downstream.
    pub shown_to_downstream: bool,
    /// Attempt start time in Unix milliseconds.
    pub started_at_unix_ms: u64,
    /// Attempt finish time in Unix milliseconds.
    pub finished_at_unix_ms: Option<u64>,
    /// Selected upstream profile name.
    pub upstream_profile: Option<String>,
    /// Request or response model id.
    pub model_id: Option<String>,
    /// Effective thinking mode.
    pub thinking_mode: Option<String>,
    /// Effective thinking budget in tokens.
    pub thinking_budget_tokens: Option<u32>,
    /// Effective max token cap.
    pub thinking_max_tokens: Option<u32>,
    /// Detector features and bounded loop metadata.
    pub detector_features: BTreeMap<String, String>,
    /// Terminal status for this evidence attempt.
    pub status: EvidenceAttemptStatus,
    /// Upstream HTTP status, if one was received.
    pub http_status: Option<u16>,
    /// Failure reason, if any.
    pub error_reason: Option<String>,
    /// Retry reason, if this attempt scheduled another attempt.
    pub retry_reason: Option<String>,
    /// Abort/cancellation reason, if any.
    pub abort_reason: Option<String>,
    /// Shadow skip reason for evidence-only skeleton records.
    pub shadow_skip_reason: Option<ShadowSkipReason>,
    /// Request-side metadata.
    pub request_metadata: BTreeMap<String, String>,
    /// Response-side metadata.
    pub response_metadata: BTreeMap<String, String>,
    /// Optional raw payload fragments.
    pub raw_payloads: RawPayloads,
}

/// Evidence-only shadow terminal record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceShadowRecord {
    /// Shadow attempt row to persist.
    pub attempt: EvidenceAttemptRecord,
}

/// Result of an evidence store write after checking current settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceStoreWrite {
    /// The row was persisted.
    Written,
    /// `evidence.enabled=false`, so the write was intentionally skipped.
    Disabled,
}

/// Current retention usage tracked by the evidence store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvidenceRetentionUsage {
    /// Number of evidence groups retained.
    pub group_count: u64,
    /// Number of evidence attempts retained.
    pub attempt_count: u64,
    /// Number of raw chunks retained.
    pub chunk_count: u64,
    /// Total logical evidence rows retained.
    pub record_count: u64,
    /// Actual `SQLite` page storage bytes.
    pub observed_bytes: u64,
}

/// Evidence retention pruning counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvidencePruningStats {
    /// Number of pruning passes that deleted at least one group.
    pub prune_events: u64,
    /// Number of evidence groups deleted.
    pub pruned_groups: u64,
    /// Number of evidence attempts deleted.
    pub pruned_attempts: u64,
    /// Number of raw chunks deleted.
    pub pruned_chunks: u64,
    /// Last pruning time in Unix milliseconds, if pruning has happened.
    pub last_pruned_at_unix_ms: Option<u64>,
}

/// Stable raw artifact kind labels used by paired comparison exports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceRawArtifactKind {
    /// Raw prompt/request body.
    Input,
    /// Raw model answer/output.
    Output,
    /// Raw model reasoning payload.
    Reasoning,
    /// Raw tool-call payload.
    ToolCalls,
}

impl EvidenceRawArtifactKind {
    /// Returns the stable storage label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Reasoning => "reasoning",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// Evidence database capability status for CLI diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct EvidenceDatabaseStatus {
    /// Whether the database file exists.
    pub exists: bool,
    /// `SQLite` `user_version`, when readable.
    pub schema_version: Option<i64>,
    /// Whether raw paired comparison tables and legacy raw columns are present.
    pub supports_raw_paired_comparison: bool,
    /// Whether the legacy raw columns exist on evidence attempts.
    pub has_attempt_raw_columns: bool,
    /// Whether the raw artifact metadata table exists.
    pub has_raw_artifact_table: bool,
}

/// Count of raw artifact rows for one role/variant/kind tuple.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceSummaryRow {
    /// Evidence attempt role.
    pub role: String,
    /// Shadow variant name, or `primary` when not a shadow comparison variant.
    pub variant_name: String,
    /// Artifact kind.
    pub artifact_kind: String,
    /// Number of artifact metadata rows.
    pub artifact_count: u64,
    /// Number of rows that still retain raw content.
    pub content_present_count: u64,
    /// Total currently retained content bytes.
    pub bytes_stored: u64,
}

/// One exported raw artifact for a paired-comparison variant.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EvidenceExportArtifact {
    /// Attempt id that produced this artifact.
    pub attempt_id: String,
    /// Evidence attempt role.
    pub role: String,
    /// Redacted and possibly truncated artifact content.
    pub content: String,
    /// Original byte length before truncation.
    pub bytes_original: u64,
    /// Stored byte length after truncation.
    pub bytes_stored: u64,
    /// Whether the stored content was truncated.
    pub truncated: bool,
    /// Whether secret redaction changed the content.
    pub redacted: bool,
    /// `SHA-256` hash of stored content for deduplication.
    pub sha256: String,
}

/// One exported pair row grouped by request and artifact kind.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EvidenceExportPair {
    /// Evidence group id.
    pub group_id: String,
    /// Proxy request id.
    pub request_id: String,
    /// Artifact kind shared by every variant in this row.
    pub artifact_kind: String,
    /// Artifact payloads keyed by configured variant name.
    pub variants: BTreeMap<String, EvidenceExportArtifact>,
}
