//! Opt-in shadow evidence ledger for loop detector tuning.
//!
//! Evidence is separate from observability because it can persist sensitive
//! paired attempt data. The store remains disabled by default and does not
//! create artifacts until an enabled write reaches the storage boundary.

mod error;
mod model;
mod redaction;
mod store;

#[cfg(test)]
mod tests;

pub use error::EvidenceError;
pub use model::{
    EvidenceAttemptRecord, EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceDatabaseStatus,
    EvidenceExportArtifact, EvidenceExportPair, EvidenceGroupRecord, EvidencePruningStats,
    EvidenceRawArtifactKind, EvidenceRetentionUsage, EvidenceShadowRecord, EvidenceStoreWrite,
    EvidenceSummaryRow, ShadowSkipReason,
};
pub use store::EvidenceStore;
