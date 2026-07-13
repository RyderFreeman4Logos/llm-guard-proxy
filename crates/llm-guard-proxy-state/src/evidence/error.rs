use std::path::PathBuf;

use thiserror::Error;

use llm_guard_proxy_core::ConfigError;

/// Evidence ledger storage failures.
#[derive(Debug, Error)]
pub enum EvidenceError {
    /// Current config could not be read.
    #[error("failed to read evidence config: {0}")]
    Config(#[from] ConfigError),
    /// HOME is required to expand a `~/` evidence path.
    #[error("could not determine home directory for evidence path")]
    HomeDirectoryUnavailable,
    /// Creating the evidence parent directory failed.
    #[error("failed to create evidence directory {path}: {source}")]
    CreateDirectory {
        /// Directory path that could not be created.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Inspecting an evidence storage path failed.
    #[error("failed to inspect evidence storage path {path}: {source}")]
    InspectPath {
        /// Path whose metadata could not be inspected.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// The configured evidence storage path is unsafe for sensitive data.
    #[error("unsafe evidence storage path {path}: {reason}")]
    UnsafeStoragePath {
        /// Unsafe configured path.
        path: PathBuf,
        /// Static reason safe to show in config errors.
        reason: &'static str,
    },
    /// Restricting evidence storage permissions failed.
    #[error("failed to restrict evidence storage permissions for {path}: {source}")]
    RestrictPermissions {
        /// Path whose permissions could not be restricted.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// `SQLite` operation failed.
    #[error("failed to {action}: {source}")]
    Sqlite {
        /// Operation being performed.
        action: &'static str,
        /// Source `SQLite` error.
        source: rusqlite::Error,
    },
    /// A future schema version was found.
    #[error("unsupported evidence schema version {version}; supported version is {supported}")]
    UnsupportedSchemaVersion {
        /// Version read from `SQLite`.
        version: i64,
        /// Highest version supported by this binary.
        supported: i64,
    },
    /// Metadata serialization failed.
    #[error("failed to serialize {field} metadata: {source}")]
    SerializeMetadata {
        /// Metadata field being serialized.
        field: &'static str,
        /// Source JSON error.
        source: serde_json::Error,
    },
    /// Shared `SQLite` connection state was poisoned by a panic.
    #[error("evidence store lock is poisoned")]
    LockPoisoned,
    /// Caller supplied an empty typed identifier.
    #[error("evidence {kind} id must not be empty")]
    EmptyIdentifier {
        /// Identifier kind.
        kind: &'static str,
    },
    /// A numeric value did not fit `SQLite`'s signed integer range.
    #[error("{field} does not fit SQLite integer range")]
    IntegerOutOfRange {
        /// Field name.
        field: &'static str,
    },
}
