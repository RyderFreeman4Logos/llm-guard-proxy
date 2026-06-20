use std::path::PathBuf;

use thiserror::Error;

use crate::ConfigError;

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
