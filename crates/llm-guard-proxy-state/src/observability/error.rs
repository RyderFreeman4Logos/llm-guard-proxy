use std::path::PathBuf;

use thiserror::Error;

use llm_guard_proxy_core::ConfigError;

/// Observability storage failures.
#[derive(Debug, Error)]
pub enum ObservabilityError {
    /// Current config could not be read.
    #[error("failed to read observability config: {0}")]
    Config(#[from] ConfigError),
    /// HOME is required to expand a `~/` `SQLite` path.
    #[error("could not determine home directory for observability SQLite path")]
    HomeDirectoryUnavailable,
    /// Creating the `SQLite` parent directory failed.
    #[error("failed to create observability directory {path}: {source}")]
    CreateDirectory {
        /// Directory path that could not be created.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Inspecting an observability storage path failed.
    #[error("failed to inspect observability storage path {path}: {source}")]
    InspectPath {
        /// Path whose metadata could not be inspected.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// The configured observability storage path is unsafe for sensitive data.
    #[error("unsafe observability storage path {path}: {reason}")]
    UnsafeStoragePath {
        /// Unsafe configured path.
        path: PathBuf,
        /// Static reason safe to show in config errors.
        reason: &'static str,
    },
    /// Restricting observability storage permissions failed.
    #[error("failed to restrict observability storage permissions for {path}: {source}")]
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
    #[error("unsupported observability schema version {version}; supported version is {supported}")]
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
    #[error("observability store lock is poisoned")]
    LockPoisoned,
    /// Another store or process owns the configured writer path.
    #[error("observability writer ownership is already held for {path}")]
    WriterOwnershipHeld {
        /// Normalized `SQLite` path whose writer is already active.
        path: PathBuf,
    },
    /// Preparing or locking the writer-ownership sidecar failed.
    #[error("failed to acquire observability writer ownership for {path}: {source}")]
    WriterOwnership {
        /// Normalized `SQLite` path whose ownership could not be acquired.
        path: PathBuf,
        /// Source filesystem locking error.
        source: std::io::Error,
    },
    /// A file-backed database cannot be represented by one path-owned writer lock.
    #[error(
        "observability writer ownership requires exactly one filesystem link for {path}; found {link_count}"
    )]
    WriterOwnershipLinkCount {
        /// Normalized `SQLite` path with unsupported filesystem identity.
        path: PathBuf,
        /// Link count read from the securely opened database file descriptor.
        link_count: u64,
    },
    /// Resolving an alias-free `SQLite` storage path failed.
    #[error("failed to normalize observability SQLite path {path}: {source}")]
    NormalizeStoragePath {
        /// Configured or absolute path being normalized.
        path: PathBuf,
        /// Source filesystem resolution error.
        source: std::io::Error,
    },
    /// Cached metrics were invalidated after database recovery failed.
    #[error("observability metrics are unavailable until the store resynchronizes")]
    MetricsUnavailable,
    /// A write failed and the metrics cache could not be resynchronized.
    #[error(
        "observability write failed ({write_error}); metrics recovery also failed ({recovery_error})"
    )]
    MetricsRecoveryFailed {
        /// Original write-path failure.
        write_error: Box<Self>,
        /// Failure while rebuilding metrics from committed database state.
        recovery_error: Box<Self>,
    },
    /// Caller supplied an empty typed identifier.
    #[error("observability {kind} id must not be empty")]
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
