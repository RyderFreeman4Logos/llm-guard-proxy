//! Bounded local `SQLite` persistence for host telemetry and alert evidence.

use crate::{DiskRate, HostSample, PolicyDecision, StorageConfig, TelemetryEvent, TelemetryState};
use rusqlite::{Connection, params};
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

/// `SQLite` store for fixed-size telemetry rows and bounded alert evidence.
#[derive(Debug)]
pub struct TelemetryStore {
    connection: Connection,
    retention: StorageConfig,
}

impl TelemetryStore {
    /// Opens the configured database and creates the telemetry schema.
    ///
    /// # Errors
    ///
    /// Returns an error when the path cannot be prepared or `SQLite` rejects the
    /// connection, schema, or retention operation.
    pub fn open(retention: StorageConfig) -> Result<Self, TelemetryStoreError> {
        let path = resolve_path(retention.sqlite_path())?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| TelemetryStoreError::Io {
                action: "create telemetry parent directory",
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let connection = Connection::open(&path).map_err(|source| TelemetryStoreError::Sqlite {
            action: "open telemetry database",
            source,
        })?;
        connection
            .execute_batch(
                "PRAGMA auto_vacuum = INCREMENTAL;
                 PRAGMA foreign_keys = ON;
                 CREATE TABLE IF NOT EXISTS samples (
                     id INTEGER PRIMARY KEY,
                     sampled_at_unix_ms INTEGER NOT NULL,
                     mem_total_kib INTEGER NOT NULL,
                     mem_available_kib INTEGER NOT NULL,
                     swap_total_kib INTEGER NOT NULL,
                     swap_used_kib INTEGER NOT NULL,
                     load_one REAL NOT NULL,
                     load_five REAL NOT NULL,
                     load_fifteen REAL NOT NULL,
                     disk_read_bytes_per_sec INTEGER,
                     disk_write_bytes_per_sec INTEGER,
                     disk_io_millis_per_sec INTEGER,
                     gpu_temperature_c REAL,
                     gpu_power_w REAL,
                     gpu_utilization_percent REAL,
                     gpu_clock_mhz REAL,
                     state TEXT NOT NULL,
                     event TEXT
                 );
                 CREATE INDEX IF NOT EXISTS samples_sampled_at_idx
                     ON samples(sampled_at_unix_ms);
                 CREATE TABLE IF NOT EXISTS evidence (
                     id INTEGER PRIMARY KEY,
                     sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
                     reason TEXT NOT NULL,
                     captured_at_unix_ms INTEGER NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS evidence_sample_id_idx
                     ON evidence(sample_id);",
            )
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "create telemetry schema",
                source,
            })?;
        Ok(Self {
            connection,
            retention,
        })
    }

    /// Persists one sample and alert evidence when the policy requests it.
    ///
    /// The stored payload is numeric host state only. It never includes
    /// requests, model prompts, service controls, or command output.
    ///
    /// # Errors
    ///
    /// Returns an error when a value cannot fit `SQLite` or the database write
    /// or bounded-retention pass fails.
    pub fn record(
        &mut self,
        sample: HostSample,
        disk_rate: Option<DiskRate>,
        decision: PolicyDecision,
    ) -> Result<(), TelemetryStoreError> {
        let transaction =
            self.connection
                .transaction()
                .map_err(|source| TelemetryStoreError::Sqlite {
                    action: "start telemetry transaction",
                    source,
                })?;
        transaction
            .execute(
                "INSERT INTO samples (
                    sampled_at_unix_ms, mem_total_kib, mem_available_kib,
                    swap_total_kib, swap_used_kib, load_one, load_five, load_fifteen,
                    disk_read_bytes_per_sec, disk_write_bytes_per_sec, disk_io_millis_per_sec,
                    gpu_temperature_c, gpu_power_w, gpu_utilization_percent, gpu_clock_mhz,
                    state, event
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
                 )",
                params![
                    sqlite_integer(sample.sampled_at_unix_ms, "sample timestamp")?,
                    sqlite_integer(sample.memory.total_kib, "memory total")?,
                    sqlite_integer(sample.memory.available_kib, "memory available")?,
                    sqlite_integer(sample.memory.swap_total_kib, "swap total")?,
                    sqlite_integer(sample.memory.swap_used_kib(), "swap used")?,
                    sample.load.one,
                    sample.load.five,
                    sample.load.fifteen,
                    disk_rate
                        .map(|rate| sqlite_integer(rate.read_bytes_per_sec, "disk read rate"))
                        .transpose()?,
                    disk_rate
                        .map(|rate| sqlite_integer(rate.write_bytes_per_sec, "disk write rate"))
                        .transpose()?,
                    disk_rate
                        .map(|rate| sqlite_integer(rate.io_millis_per_sec, "disk io rate"))
                        .transpose()?,
                    sample.gpu.map(|gpu| gpu.temperature_c),
                    sample.gpu.map(|gpu| gpu.power_w),
                    sample.gpu.map(|gpu| gpu.utilization_percent),
                    sample.gpu.map(|gpu| gpu.clock_mhz),
                    state_name(decision.state),
                    decision.event.map(event_name),
                ],
            )
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "insert telemetry sample",
                source,
            })?;
        if let Some(event) = decision.event.filter(|event| event.collects_evidence()) {
            transaction
                .execute(
                    "INSERT INTO evidence (sample_id, reason, captured_at_unix_ms)
                     VALUES (?1, ?2, ?3)",
                    params![
                        transaction.last_insert_rowid(),
                        event_name(event),
                        sqlite_integer(sample.sampled_at_unix_ms, "evidence timestamp")?,
                    ],
                )
                .map_err(|source| TelemetryStoreError::Sqlite {
                    action: "insert telemetry evidence",
                    source,
                })?;
        }
        transaction
            .commit()
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "commit telemetry transaction",
                source,
            })?;
        self.enforce_retention()
    }

    /// Returns the number of retained samples for tests and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` cannot count the retained rows.
    pub fn sample_count(&self) -> Result<u64, TelemetryStoreError> {
        let count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM samples", [], |row| row.get(0))
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "count telemetry samples",
                source,
            })?;
        u64::try_from(count).map_err(|_error| TelemetryStoreError::InvalidValue {
            field: "stored sample count",
        })
    }

    fn enforce_retention(&mut self) -> Result<(), TelemetryStoreError> {
        let count = self.sample_count()?;
        if count <= self.retention.max_records() {
            return Ok(());
        }
        let delete_count = count.saturating_sub(self.retention.prune_to_records());
        self.connection
            .execute(
                "DELETE FROM samples WHERE id IN (
                    SELECT id FROM samples ORDER BY id ASC LIMIT ?1
                 )",
                [sqlite_integer(delete_count, "retention delete count")?],
            )
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "prune telemetry samples",
                source,
            })?;
        Ok(())
    }

    /// Reclaims a bounded number of free pages during out-of-band maintenance.
    ///
    /// Sampling and retention pruning never invoke this method, so compaction
    /// cannot block the sampling cadence. New databases are initialized for
    /// incremental auto-vacuum; existing databases retain their prior mode.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the incremental maintenance pass.
    pub fn compact_incrementally(&mut self) -> Result<(), TelemetryStoreError> {
        self.connection
            .execute_batch("PRAGMA incremental_vacuum(256)")
            .map_err(|source| TelemetryStoreError::Sqlite {
                action: "incrementally compact telemetry database",
                source,
            })
    }
}

fn resolve_path(path: &Path) -> Result<PathBuf, TelemetryStoreError> {
    let Some(stripped) = path.to_str().and_then(|value| value.strip_prefix("~/")) else {
        return Ok(path.to_path_buf());
    };
    let home = env::var_os("HOME").ok_or(TelemetryStoreError::MissingHome)?;
    Ok(PathBuf::from(home).join(stripped))
}

fn sqlite_integer(value: u64, field: &'static str) -> Result<i64, TelemetryStoreError> {
    i64::try_from(value).map_err(|_error| TelemetryStoreError::InvalidValue { field })
}

const fn state_name(state: TelemetryState) -> &'static str {
    match state {
        TelemetryState::Healthy => "healthy",
        TelemetryState::SwapWarning => "swap_warning",
        TelemetryState::Alert(_) => "alert",
    }
}

const fn event_name(event: TelemetryEvent) -> &'static str {
    match event {
        TelemetryEvent::SwapWarning => "swap_warning",
        TelemetryEvent::Alert(crate::PressureReason::MemoryAvailable) => "memory_available",
        TelemetryEvent::Alert(crate::PressureReason::Swap) => "swap",
        TelemetryEvent::Alert(crate::PressureReason::MemoryAndSwap) => "memory_and_swap",
        TelemetryEvent::Cleared => "cleared",
    }
}

/// Persistence failures for the observer-only telemetry store.
#[derive(Debug, Error)]
pub enum TelemetryStoreError {
    /// A directory could not be created.
    #[error("{action} at {path}: {source}", path = path.display())]
    Io {
        /// Failed filesystem operation.
        action: &'static str,
        /// Path used for the operation.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// `SQLite` rejected an operation.
    #[error("{action}: {source}")]
    Sqlite {
        /// Failed `SQLite` operation.
        action: &'static str,
        /// Underlying `SQLite` error.
        source: rusqlite::Error,
    },
    /// A home-relative database path needs `HOME`.
    #[error("HOME must be set when host telemetry sqlite_path starts with ~/")]
    MissingHome,
    /// A numeric value cannot be safely represented by `SQLite`.
    #[error("{field} cannot be represented by SQLite")]
    InvalidValue {
        /// Rejected field name.
        field: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::TelemetryStore;
    use crate::{
        HostSample, LoadAverage, MemorySample, PolicyDecision, PressureReason, StorageConfig,
        TelemetryEvent, TelemetryState,
    };
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> PathBuf {
        let id = NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("llm-guard-proxy-telemetry-{id}.sqlite3"))
    }

    fn sample(timestamp: u64) -> HostSample {
        HostSample {
            sampled_at_unix_ms: timestamp,
            memory: MemorySample {
                total_kib: 10,
                available_kib: 5,
                swap_total_kib: 4,
                swap_free_kib: 4,
            },
            load: LoadAverage {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            },
            disk: None,
            gpu: None,
        }
    }

    #[test]
    fn retention_prunes_oldest_rows_to_the_lower_target() {
        let path = test_path();
        let config = StorageConfig::new(path.clone(), 3, 2).expect("test retention is valid");
        let mut store = TelemetryStore::open(config).expect("store opens");
        let decision = PolicyDecision {
            state: TelemetryState::Healthy,
            event: None,
        };
        for timestamp in 1..=4 {
            store
                .record(sample(timestamp), None, decision)
                .expect("sample persists");
        }
        assert_eq!(store.sample_count().expect("count reads"), 2);
        drop(store);
        fs::remove_file(path).expect("test database can be removed");
    }

    #[test]
    fn retention_indexes_cascades_and_leaves_compaction_for_explicit_maintenance() {
        let path = test_path();
        let config = StorageConfig::new(path.clone(), 3, 2).expect("test retention is valid");
        let mut store = TelemetryStore::open(config).expect("store opens");
        let indexed: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema
                 WHERE type = 'index' AND name = 'evidence_sample_id_idx'",
                [],
                |row| row.get(0),
            )
            .expect("schema index query succeeds");
        assert_eq!(indexed, 1, "retention cascades need a child-key index");
        let auto_vacuum: i64 = store
            .connection
            .query_row("PRAGMA auto_vacuum", [], |row| row.get(0))
            .expect("auto-vacuum mode reads");
        assert_eq!(auto_vacuum, 2, "new stores use incremental vacuum mode");

        let decision = PolicyDecision {
            state: TelemetryState::Alert(PressureReason::MemoryAvailable),
            event: Some(TelemetryEvent::Alert(PressureReason::MemoryAvailable)),
        };
        for timestamp in 1..=3 {
            store
                .record(sample(timestamp), None, decision)
                .expect("sample persists");
        }
        store
            .connection
            .execute(
                "INSERT INTO evidence (sample_id, reason, captured_at_unix_ms)
                 VALUES (1, ?1, 1)",
                ["x".repeat(1024 * 1024)],
            )
            .expect("large oldest-row evidence fixture inserts");
        store
            .record(sample(4), None, decision)
            .expect("sample persists and triggers pruning");

        let evidence_count: i64 = store
            .connection
            .query_row("SELECT COUNT(*) FROM evidence", [], |row| row.get(0))
            .expect("evidence count reads");
        assert_eq!(evidence_count, 2, "oldest evidence cascades with samples");
        let free_before: i64 = store
            .connection
            .query_row("PRAGMA freelist_count", [], |row| row.get(0))
            .expect("free-page count reads");
        assert!(
            free_before > 0,
            "sampling-path pruning must not run a full compaction"
        );

        store
            .compact_incrementally()
            .expect("explicit incremental maintenance succeeds");
        let free_after: i64 = store
            .connection
            .query_row("PRAGMA freelist_count", [], |row| row.get(0))
            .expect("free-page count reads after maintenance");
        assert!(
            free_after < free_before,
            "out-of-band maintenance should reclaim free pages incrementally"
        );

        drop(store);
        fs::remove_file(path).expect("test database can be removed");
    }
}
