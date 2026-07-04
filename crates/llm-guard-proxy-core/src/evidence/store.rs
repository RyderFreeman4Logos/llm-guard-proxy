use std::{
    collections::BTreeMap,
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

use rusqlite::{Connection, OpenFlags, params};

use super::{
    error::EvidenceError,
    model::{
        EvidenceAttemptRecord, EvidenceGroupRecord, EvidenceRetentionUsage, EvidenceStoreWrite,
        ShadowSkipReason,
    },
    redaction::{evidence_metadata_map, sanitize_raw_payloads, scrub_optional_text},
};
use crate::{ConfigHandle, RawPayloads};

const SCHEMA_VERSION: i64 = 1;
#[cfg(unix)]
const EVIDENCE_DIRECTORY_MODE: u32 = 0o700;
#[cfg(unix)]
const EVIDENCE_SQLITE_MODE: u32 = 0o600;

/// SQLite-backed evidence ledger.
#[derive(Clone, Debug)]
pub struct EvidenceStore {
    config: ConfigHandle,
    connection: Arc<Mutex<Option<Connection>>>,
}

impl EvidenceStore {
    /// Builds a lazy evidence store.
    ///
    /// This does not create any filesystem artifacts. The `SQLite` file is opened
    /// only when `evidence.enabled=true` and the caller records evidence.
    #[must_use]
    pub fn open(config: ConfigHandle) -> Self {
        Self {
            config,
            connection: Arc::new(Mutex::new(None)),
        }
    }

    /// Persists one correlated evidence group and its attempts.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when current settings cannot be read,
    /// metadata cannot serialize, the database lock is poisoned, or `SQLite`
    /// persistence fails.
    pub fn record_group(
        &self,
        group: &EvidenceGroupRecord,
        attempts: &[EvidenceAttemptRecord],
    ) -> Result<EvidenceStoreWrite, EvidenceError> {
        let settings = self.config.snapshot()?;
        if !settings.evidence.enabled {
            return Ok(EvidenceStoreWrite::Disabled);
        }
        let include_headers = settings.evidence.include_request_headers;
        let include_raw_payloads = settings.evidence.include_raw_payloads;
        let prepared_group = PreparedGroup::from_record(group, include_headers)?;
        let prepared_attempts = attempts
            .iter()
            .map(|attempt| {
                PreparedAttempt::from_record(attempt, include_headers, include_raw_payloads)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut connection = self.lock_connection()?;
        let connection =
            open_connection_if_needed(&mut connection, &settings.evidence.sqlite_path)?;
        insert_group_and_attempts(connection, &prepared_group, &prepared_attempts)?;
        let pruning = enforce_retention(connection, &settings.evidence)?;
        record_pruning_outcome(connection, &pruning)?;
        Ok(EvidenceStoreWrite::Written)
    }

    /// Persists one evidence-only shadow attempt in an existing group.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when the write cannot be persisted.
    pub fn record_shadow_attempt(
        &self,
        attempt: &EvidenceAttemptRecord,
    ) -> Result<EvidenceStoreWrite, EvidenceError> {
        let settings = self.config.snapshot()?;
        if !settings.evidence.enabled {
            return Ok(EvidenceStoreWrite::Disabled);
        }
        let prepared_attempt = PreparedAttempt::from_record(
            attempt,
            settings.evidence.include_request_headers,
            settings.evidence.include_raw_payloads,
        )?;
        let mut connection = self.lock_connection()?;
        let connection =
            open_connection_if_needed(&mut connection, &settings.evidence.sqlite_path)?;
        insert_attempt(connection, &prepared_attempt)?;
        let pruning = enforce_retention(connection, &settings.evidence)?;
        record_pruning_outcome(connection, &pruning)?;
        Ok(EvidenceStoreWrite::Written)
    }

    /// Returns logical retention usage for tests and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when the store cannot be queried.
    pub fn retention_usage(&self) -> Result<EvidenceRetentionUsage, EvidenceError> {
        let settings = self.config.snapshot()?;
        let mut connection = self.lock_connection()?;
        let connection =
            open_connection_if_needed(&mut connection, &settings.evidence.sqlite_path)?;
        read_retention_usage(connection)
    }

    pub(super) fn lock_connection(
        &self,
    ) -> Result<MutexGuard<'_, Option<Connection>>, EvidenceError> {
        self.connection
            .lock()
            .map_err(|_error| EvidenceError::LockPoisoned)
    }
}

#[derive(Debug)]
struct PreparedGroup {
    group_id: String,
    request_id: String,
    started_at_unix_ms: i64,
    finished_at_unix_ms: Option<i64>,
    model_id: Option<String>,
    status: String,
    request_metadata_json: String,
    response_metadata_json: String,
    estimated_bytes: i64,
}

impl PreparedGroup {
    fn from_record(
        record: &EvidenceGroupRecord,
        include_headers: bool,
    ) -> Result<Self, EvidenceError> {
        require_nonempty(&record.group_id, "group")?;
        let request_metadata = evidence_metadata_map(&record.request_metadata, include_headers);
        let response_metadata = evidence_metadata_map(&record.response_metadata, include_headers);
        let request_metadata_json = metadata_json(&request_metadata, "evidence group request")?;
        let response_metadata_json = metadata_json(&response_metadata, "evidence group response")?;
        let estimated_bytes =
            estimate_group_bytes(record, &request_metadata_json, &response_metadata_json)?;
        Ok(Self {
            group_id: record.group_id.clone(),
            request_id: record.request_id.as_str().to_owned(),
            started_at_unix_ms: to_sqlite_i64(record.started_at_unix_ms, "started_at_unix_ms")?,
            finished_at_unix_ms: optional_to_sqlite_i64(
                record.finished_at_unix_ms,
                "finished_at_unix_ms",
            )?,
            model_id: scrub_optional_text(record.model_id.as_ref()),
            status: record.status.clone(),
            request_metadata_json,
            response_metadata_json,
            estimated_bytes,
        })
    }
}

#[derive(Debug)]
struct PreparedAttempt {
    attempt_id: String,
    group_id: String,
    request_id: String,
    attempt_number: i64,
    role: &'static str,
    shown_to_downstream: i64,
    started_at_unix_ms: i64,
    finished_at_unix_ms: Option<i64>,
    upstream_profile: Option<String>,
    model_id: Option<String>,
    thinking_mode: Option<String>,
    thinking_budget_tokens: Option<i64>,
    thinking_max_tokens: Option<i64>,
    detector_features_json: String,
    status: &'static str,
    http_status: Option<i64>,
    error_reason: Option<String>,
    retry_reason: Option<String>,
    abort_reason: Option<String>,
    shadow_skip_reason: Option<&'static str>,
    request_metadata_json: String,
    response_metadata_json: String,
    raw_payloads: RawPayloads,
    estimated_bytes: i64,
}

impl PreparedAttempt {
    fn from_record(
        record: &EvidenceAttemptRecord,
        include_headers: bool,
        include_raw_payloads: bool,
    ) -> Result<Self, EvidenceError> {
        require_nonempty(&record.group_id, "group")?;
        let request_metadata = evidence_metadata_map(&record.request_metadata, include_headers);
        let response_metadata = evidence_metadata_map(&record.response_metadata, include_headers);
        let detector_features = evidence_metadata_map(&record.detector_features, false);
        let request_metadata_json = metadata_json(&request_metadata, "evidence attempt request")?;
        let response_metadata_json =
            metadata_json(&response_metadata, "evidence attempt response")?;
        let detector_features_json = metadata_json(&detector_features, "evidence detector")?;
        let raw_payloads = sanitize_raw_payloads(&record.raw_payloads, include_raw_payloads);
        let estimated_bytes = estimate_attempt_bytes(
            record,
            &request_metadata_json,
            &response_metadata_json,
            &detector_features_json,
            &raw_payloads,
        )?;

        Ok(Self {
            attempt_id: record.attempt_id.as_str().to_owned(),
            group_id: record.group_id.clone(),
            request_id: record.request_id.as_str().to_owned(),
            attempt_number: i64::from(record.attempt_number),
            role: record.role.as_str(),
            shown_to_downstream: i64::from(record.shown_to_downstream),
            started_at_unix_ms: to_sqlite_i64(record.started_at_unix_ms, "started_at_unix_ms")?,
            finished_at_unix_ms: optional_to_sqlite_i64(
                record.finished_at_unix_ms,
                "finished_at_unix_ms",
            )?,
            upstream_profile: scrub_optional_text(record.upstream_profile.as_ref()),
            model_id: scrub_optional_text(record.model_id.as_ref()),
            thinking_mode: scrub_optional_text(record.thinking_mode.as_ref()),
            thinking_budget_tokens: record.thinking_budget_tokens.map(i64::from),
            thinking_max_tokens: record.thinking_max_tokens.map(i64::from),
            detector_features_json,
            status: record.status.as_str(),
            http_status: record.http_status.map(i64::from),
            error_reason: scrub_optional_text(record.error_reason.as_ref()),
            retry_reason: scrub_optional_text(record.retry_reason.as_ref()),
            abort_reason: scrub_optional_text(record.abort_reason.as_ref()),
            shadow_skip_reason: record.shadow_skip_reason.map(ShadowSkipReason::as_str),
            request_metadata_json,
            response_metadata_json,
            raw_payloads,
            estimated_bytes,
        })
    }
}

fn open_connection_if_needed<'connection>(
    slot: &'connection mut Option<Connection>,
    configured_path: &Path,
) -> Result<&'connection mut Connection, EvidenceError> {
    if slot.is_none() {
        let sqlite_path = resolve_sqlite_path(configured_path)?;
        prepare_parent_directory(&sqlite_path)?;
        prepare_sqlite_file(&sqlite_path)?;
        let connection = open_sqlite_connection(&sqlite_path)?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(|source| EvidenceError::Sqlite {
                action: "enable SQLite foreign keys",
                source,
            })?;
        migrate(&connection)?;
        *slot = Some(connection);
    }
    slot.as_mut().ok_or(EvidenceError::LockPoisoned)
}

fn migrate(connection: &Connection) -> Result<(), EvidenceError> {
    let version = read_schema_version(connection)?;
    if version > SCHEMA_VERSION {
        return Err(EvidenceError::UnsupportedSchemaVersion {
            version,
            supported: SCHEMA_VERSION,
        });
    }
    if version == SCHEMA_VERSION {
        return Ok(());
    }
    create_schema(connection)
}

fn create_schema(connection: &Connection) -> Result<(), EvidenceError> {
    connection
        .execute_batch(
            r"
CREATE TABLE IF NOT EXISTS evidence_groups (
    group_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    started_at_unix_ms INTEGER NOT NULL,
    finished_at_unix_ms INTEGER,
    model_id TEXT,
    status TEXT NOT NULL,
    request_metadata_json TEXT NOT NULL,
    response_metadata_json TEXT NOT NULL,
    estimated_bytes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS evidence_groups_started_at_idx
    ON evidence_groups(started_at_unix_ms, group_id);
CREATE INDEX IF NOT EXISTS evidence_groups_request_id_idx
    ON evidence_groups(request_id);

CREATE TABLE IF NOT EXISTS evidence_attempts (
    attempt_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL REFERENCES evidence_groups(group_id) ON DELETE CASCADE,
    request_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    shown_to_downstream INTEGER NOT NULL,
    started_at_unix_ms INTEGER NOT NULL,
    finished_at_unix_ms INTEGER,
    upstream_profile TEXT,
    model_id TEXT,
    thinking_mode TEXT,
    thinking_budget_tokens INTEGER,
    thinking_max_tokens INTEGER,
    detector_features_json TEXT NOT NULL,
    status TEXT NOT NULL,
    http_status INTEGER,
    error_reason TEXT,
    retry_reason TEXT,
    abort_reason TEXT,
    shadow_skip_reason TEXT,
    request_metadata_json TEXT NOT NULL,
    response_metadata_json TEXT NOT NULL,
    raw_input TEXT,
    raw_output TEXT,
    raw_reasoning TEXT,
    raw_tool_calls TEXT,
    estimated_bytes INTEGER NOT NULL,
    UNIQUE(group_id, attempt_number, role)
);

CREATE INDEX IF NOT EXISTS evidence_attempts_group_id_idx
    ON evidence_attempts(group_id);
CREATE INDEX IF NOT EXISTS evidence_attempts_role_idx
    ON evidence_attempts(role);
CREATE INDEX IF NOT EXISTS evidence_attempts_status_idx
    ON evidence_attempts(status);

CREATE TABLE IF NOT EXISTS evidence_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES evidence_attempts(attempt_id) ON DELETE CASCADE,
    channel TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_bytes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS evidence_chunks_attempt_id_idx
    ON evidence_chunks(attempt_id);

CREATE TABLE IF NOT EXISTS evidence_pruning_stats (
    stats_key TEXT PRIMARY KEY,
    prune_events INTEGER NOT NULL,
    pruned_groups INTEGER NOT NULL,
    pruned_attempts INTEGER NOT NULL,
    pruned_chunks INTEGER NOT NULL,
    last_pruned_at_unix_ms INTEGER
);

INSERT OR IGNORE INTO evidence_pruning_stats (
    stats_key,
    prune_events,
    pruned_groups,
    pruned_attempts,
    pruned_chunks,
    last_pruned_at_unix_ms
) VALUES ('global', 0, 0, 0, 0, NULL);

PRAGMA user_version = 1;
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "create SQLite evidence schema",
            source,
        })
}

fn read_schema_version(connection: &Connection) -> Result<i64, EvidenceError> {
    connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|source| EvidenceError::Sqlite {
            action: "read SQLite evidence schema version",
            source,
        })
}

fn insert_group_and_attempts(
    connection: &mut Connection,
    group: &PreparedGroup,
    attempts: &[PreparedAttempt],
) -> Result<(), EvidenceError> {
    let transaction = connection
        .transaction()
        .map_err(|source| EvidenceError::Sqlite {
            action: "start evidence group transaction",
            source,
        })?;
    insert_group_in_transaction(&transaction, group)?;
    for attempt in attempts {
        insert_attempt_in_transaction(&transaction, attempt)?;
    }
    transaction
        .commit()
        .map_err(|source| EvidenceError::Sqlite {
            action: "commit evidence group transaction",
            source,
        })
}

fn insert_attempt(
    connection: &mut Connection,
    attempt: &PreparedAttempt,
) -> Result<(), EvidenceError> {
    let transaction = connection
        .transaction()
        .map_err(|source| EvidenceError::Sqlite {
            action: "start evidence attempt transaction",
            source,
        })?;
    insert_attempt_in_transaction(&transaction, attempt)?;
    transaction
        .commit()
        .map_err(|source| EvidenceError::Sqlite {
            action: "commit evidence attempt transaction",
            source,
        })
}

fn insert_group_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    group: &PreparedGroup,
) -> Result<(), EvidenceError> {
    transaction
        .execute(
            r"
INSERT INTO evidence_groups (
    group_id,
    request_id,
    started_at_unix_ms,
    finished_at_unix_ms,
    model_id,
    status,
    request_metadata_json,
    response_metadata_json,
    estimated_bytes
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
ON CONFLICT(group_id) DO UPDATE SET
    request_id = excluded.request_id,
    started_at_unix_ms = excluded.started_at_unix_ms,
    finished_at_unix_ms = excluded.finished_at_unix_ms,
    model_id = excluded.model_id,
    status = excluded.status,
    request_metadata_json = excluded.request_metadata_json,
    response_metadata_json = excluded.response_metadata_json,
    estimated_bytes = excluded.estimated_bytes
",
            params![
                group.group_id,
                group.request_id,
                group.started_at_unix_ms,
                group.finished_at_unix_ms,
                group.model_id,
                group.status,
                group.request_metadata_json,
                group.response_metadata_json,
                group.estimated_bytes,
            ],
        )
        .map(|_updated| ())
        .map_err(|source| EvidenceError::Sqlite {
            action: "write evidence group row",
            source,
        })
}

fn insert_attempt_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    attempt: &PreparedAttempt,
) -> Result<(), EvidenceError> {
    transaction
        .execute(
            r"
INSERT OR REPLACE INTO evidence_attempts (
    attempt_id,
    group_id,
    request_id,
    attempt_number,
    role,
    shown_to_downstream,
    started_at_unix_ms,
    finished_at_unix_ms,
    upstream_profile,
    model_id,
    thinking_mode,
    thinking_budget_tokens,
    thinking_max_tokens,
    detector_features_json,
    status,
    http_status,
    error_reason,
    retry_reason,
    abort_reason,
    shadow_skip_reason,
    request_metadata_json,
    response_metadata_json,
    raw_input,
    raw_output,
    raw_reasoning,
    raw_tool_calls,
    estimated_bytes
) VALUES (
    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
    ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
    ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
)",
            params![
                attempt.attempt_id,
                attempt.group_id,
                attempt.request_id,
                attempt.attempt_number,
                attempt.role,
                attempt.shown_to_downstream,
                attempt.started_at_unix_ms,
                attempt.finished_at_unix_ms,
                attempt.upstream_profile,
                attempt.model_id,
                attempt.thinking_mode,
                attempt.thinking_budget_tokens,
                attempt.thinking_max_tokens,
                attempt.detector_features_json,
                attempt.status,
                attempt.http_status,
                attempt.error_reason,
                attempt.retry_reason,
                attempt.abort_reason,
                attempt.shadow_skip_reason,
                attempt.request_metadata_json,
                attempt.response_metadata_json,
                attempt.raw_payloads.input,
                attempt.raw_payloads.output,
                attempt.raw_payloads.reasoning,
                attempt.raw_payloads.tool_calls,
                attempt.estimated_bytes,
            ],
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "write evidence attempt row",
            source,
        })?;
    insert_raw_chunks_in_transaction(transaction, attempt)
}

fn insert_raw_chunks_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    attempt: &PreparedAttempt,
) -> Result<(), EvidenceError> {
    transaction
        .execute(
            "DELETE FROM evidence_chunks WHERE attempt_id = ?1",
            params![attempt.attempt_id],
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "replace evidence raw chunks",
            source,
        })?;
    let chunks = raw_chunks(&attempt.raw_payloads);
    for (sequence_number, (channel, chunk)) in chunks.into_iter().enumerate() {
        transaction
            .execute(
                r"
INSERT INTO evidence_chunks (attempt_id, channel, sequence_number, chunk_text, chunk_bytes)
VALUES (?1, ?2, ?3, ?4, ?5)
",
                params![
                    attempt.attempt_id,
                    channel,
                    to_sqlite_i64(sequence_number, "sequence_number")?,
                    chunk,
                    to_sqlite_i64(chunk.len(), "chunk_bytes")?,
                ],
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "write evidence raw chunk",
                source,
            })?;
    }
    Ok(())
}

fn raw_chunks(raw_payloads: &RawPayloads) -> Vec<(&str, &str)> {
    let mut chunks = Vec::new();
    if let Some(input) = &raw_payloads.input {
        chunks.push(("input", input.as_str()));
    }
    if !raw_payloads.chunks.is_empty() {
        chunks.extend(
            raw_payloads
                .chunks
                .iter()
                .map(|chunk| (chunk.channel.as_str(), chunk.text.as_str())),
        );
        return chunks;
    }
    if let Some(output) = &raw_payloads.output {
        chunks.push(("output", output.as_str()));
    }
    if let Some(reasoning) = &raw_payloads.reasoning {
        chunks.push(("reasoning", reasoning.as_str()));
    }
    if let Some(tool_calls) = &raw_payloads.tool_calls {
        chunks.push(("tool_calls", tool_calls.as_str()));
    }
    chunks
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RetentionPruneOutcome {
    groups: u64,
    attempts: u64,
    chunks: u64,
}

impl RetentionPruneOutcome {
    const fn deleted_any(self) -> bool {
        self.groups > 0 || self.attempts > 0 || self.chunks > 0
    }

    fn add(&mut self, other: Self) {
        self.groups = self.groups.saturating_add(other.groups);
        self.attempts = self.attempts.saturating_add(other.attempts);
        self.chunks = self.chunks.saturating_add(other.chunks);
    }
}

fn enforce_retention(
    connection: &mut Connection,
    config: &crate::EvidenceConfig,
) -> Result<RetentionPruneOutcome, EvidenceError> {
    let mut usage = read_retention_usage(connection)?;
    let target_bytes = if usage.observed_bytes > config.max_bytes {
        config.prune_to_bytes
    } else {
        config.max_bytes
    };
    let target_records = if usage.record_count > config.max_records {
        config.effective_prune_to_records()
    } else {
        config.max_records
    };
    let mut outcome = RetentionPruneOutcome::default();

    while usage.observed_bytes > target_bytes || usage.record_count > target_records {
        let batch = prune_retained_groups(connection, target_records, target_bytes)?;
        if !batch.deleted_any() {
            break;
        }
        outcome.add(batch);
        vacuum_database(connection)?;
        usage = read_retention_usage(connection)?;
        if usage.record_count == 0 {
            break;
        }
    }
    Ok(outcome)
}

fn prune_retained_groups(
    connection: &mut Connection,
    max_records: u64,
    target_bytes: u64,
) -> Result<RetentionPruneOutcome, EvidenceError> {
    let transaction = connection
        .transaction()
        .map_err(|source| EvidenceError::Sqlite {
            action: "start evidence retention transaction",
            source,
        })?;
    let mut outcome = RetentionPruneOutcome::default();
    let mut usage = read_retention_usage(&transaction)?;
    let mut logical_bytes = read_logical_observed_bytes(&transaction)?;

    while usage.record_count > max_records
        || logical_bytes > target_bytes
        || (!outcome.deleted_any() && usage.group_count > 0)
    {
        let Some(group_id) = oldest_group_id(&transaction)? else {
            break;
        };
        let attempt_count = read_attempt_count_for_group(&transaction, &group_id)?;
        let chunk_count = read_chunk_count_for_group(&transaction, &group_id)?;
        transaction
            .execute(
                "DELETE FROM evidence_groups WHERE group_id = ?1",
                params![group_id],
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "prune oldest evidence group",
                source,
            })?;
        outcome.groups = outcome.groups.saturating_add(1);
        outcome.attempts = outcome.attempts.saturating_add(attempt_count);
        outcome.chunks = outcome.chunks.saturating_add(chunk_count);
        usage = read_retention_usage(&transaction)?;
        logical_bytes = read_logical_observed_bytes(&transaction)?;
    }

    transaction
        .commit()
        .map_err(|source| EvidenceError::Sqlite {
            action: "commit evidence retention transaction",
            source,
        })?;
    Ok(outcome)
}

fn record_pruning_outcome(
    connection: &mut Connection,
    outcome: &RetentionPruneOutcome,
) -> Result<(), EvidenceError> {
    if !outcome.deleted_any() {
        return Ok(());
    }
    let now = to_sqlite_i64(unix_time_millis(), "last_pruned_at_unix_ms")?;
    connection
        .execute(
            r"
UPDATE evidence_pruning_stats
SET prune_events = prune_events + 1,
    pruned_groups = pruned_groups + ?1,
    pruned_attempts = pruned_attempts + ?2,
    pruned_chunks = pruned_chunks + ?3,
    last_pruned_at_unix_ms = ?4
WHERE stats_key = 'global'
",
            params![
                to_sqlite_i64(outcome.groups, "pruned_groups")?,
                to_sqlite_i64(outcome.attempts, "pruned_attempts")?,
                to_sqlite_i64(outcome.chunks, "pruned_chunks")?,
                now,
            ],
        )
        .map(|_updated| ())
        .map_err(|source| EvidenceError::Sqlite {
            action: "record evidence retention pruning stats",
            source,
        })
}

fn vacuum_database(connection: &Connection) -> Result<(), EvidenceError> {
    connection
        .execute_batch("VACUUM")
        .map_err(|source| EvidenceError::Sqlite {
            action: "vacuum SQLite evidence store",
            source,
        })
}

fn oldest_group_id(connection: &Connection) -> Result<Option<String>, EvidenceError> {
    let mut statement = connection
        .prepare(
            r"
SELECT group_id
FROM evidence_groups
ORDER BY started_at_unix_ms ASC, group_id ASC
LIMIT 1
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "prepare oldest evidence group query",
            source,
        })?;
    let mut rows = statement
        .query([])
        .map_err(|source| EvidenceError::Sqlite {
            action: "query oldest evidence group",
            source,
        })?;
    let Some(row) = rows.next().map_err(|source| EvidenceError::Sqlite {
        action: "read oldest evidence group",
        source,
    })?
    else {
        return Ok(None);
    };
    row.get(0)
        .map(Some)
        .map_err(|source| EvidenceError::Sqlite {
            action: "decode oldest evidence group",
            source,
        })
}

fn read_retention_usage(connection: &Connection) -> Result<EvidenceRetentionUsage, EvidenceError> {
    let group_count = read_count(connection, "evidence_groups", "read evidence group count")?;
    let attempt_count = read_count(
        connection,
        "evidence_attempts",
        "read evidence attempt count",
    )?;
    let chunk_count = read_count(connection, "evidence_chunks", "read evidence chunk count")?;
    let record_count = group_count
        .saturating_add(attempt_count)
        .saturating_add(chunk_count);
    let observed_bytes = read_sqlite_storage_bytes(connection)?;
    Ok(EvidenceRetentionUsage {
        group_count,
        attempt_count,
        chunk_count,
        record_count,
        observed_bytes,
    })
}

fn read_count(
    connection: &Connection,
    table: &'static str,
    action: &'static str,
) -> Result<u64, EvidenceError> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let count: i64 = connection
        .query_row(&sql, [], |row| row.get(0))
        .map_err(|source| EvidenceError::Sqlite { action, source })?;
    Ok(nonnegative_i64_to_u64(count))
}

fn read_attempt_count_for_group(
    connection: &Connection,
    group_id: &str,
) -> Result<u64, EvidenceError> {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM evidence_attempts WHERE group_id = ?1",
            params![group_id],
            |row| row.get(0),
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence attempt count for pruned group",
            source,
        })?;
    Ok(nonnegative_i64_to_u64(count))
}

fn read_chunk_count_for_group(
    connection: &Connection,
    group_id: &str,
) -> Result<u64, EvidenceError> {
    let count: i64 = connection
        .query_row(
            r"
SELECT COUNT(*)
FROM evidence_chunks
WHERE attempt_id IN (
    SELECT attempt_id FROM evidence_attempts WHERE group_id = ?1
)
",
            params![group_id],
            |row| row.get(0),
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence chunk count for pruned group",
            source,
        })?;
    Ok(nonnegative_i64_to_u64(count))
}

fn read_logical_observed_bytes(connection: &Connection) -> Result<u64, EvidenceError> {
    let group_bytes = read_sum_estimated_bytes(
        connection,
        "evidence_groups",
        "read evidence group logical bytes",
    )?;
    let attempt_bytes = read_sum_estimated_bytes(
        connection,
        "evidence_attempts",
        "read evidence attempt logical bytes",
    )?;
    let chunk_bytes: i64 = connection
        .query_row(
            "SELECT COALESCE(SUM(chunk_bytes), 0) FROM evidence_chunks",
            [],
            |row| row.get(0),
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence chunk logical bytes",
            source,
        })?;
    Ok(group_bytes
        .saturating_add(attempt_bytes)
        .saturating_add(nonnegative_i64_to_u64(chunk_bytes)))
}

fn read_sum_estimated_bytes(
    connection: &Connection,
    table: &'static str,
    action: &'static str,
) -> Result<u64, EvidenceError> {
    let sql = format!("SELECT COALESCE(SUM(estimated_bytes), 0) FROM {table}");
    let count: i64 = connection
        .query_row(&sql, [], |row| row.get(0))
        .map_err(|source| EvidenceError::Sqlite { action, source })?;
    Ok(nonnegative_i64_to_u64(count))
}

fn read_sqlite_storage_bytes(connection: &Connection) -> Result<u64, EvidenceError> {
    let page_count: i64 = connection
        .query_row("PRAGMA page_count", [], |row| row.get(0))
        .map_err(|source| EvidenceError::Sqlite {
            action: "read SQLite evidence page count",
            source,
        })?;
    let page_size: i64 = connection
        .query_row("PRAGMA page_size", [], |row| row.get(0))
        .map_err(|source| EvidenceError::Sqlite {
            action: "read SQLite evidence page size",
            source,
        })?;
    Ok(nonnegative_i64_to_u64(page_count).saturating_mul(nonnegative_i64_to_u64(page_size)))
}

fn metadata_json(
    metadata: &BTreeMap<String, String>,
    field: &'static str,
) -> Result<String, EvidenceError> {
    serde_json::to_string(metadata)
        .map_err(|source| EvidenceError::SerializeMetadata { field, source })
}

fn estimate_group_bytes(
    record: &EvidenceGroupRecord,
    request_metadata_json: &str,
    response_metadata_json: &str,
) -> Result<i64, EvidenceError> {
    let bytes = record
        .group_id
        .len()
        .saturating_add(record.request_id.as_str().len())
        .saturating_add(record.model_id.as_ref().map_or(0, String::len))
        .saturating_add(record.status.len())
        .saturating_add(request_metadata_json.len())
        .saturating_add(response_metadata_json.len());
    to_sqlite_i64(bytes, "estimated_bytes")
}

fn estimate_attempt_bytes(
    record: &EvidenceAttemptRecord,
    request_metadata_json: &str,
    response_metadata_json: &str,
    detector_features_json: &str,
    raw_payloads: &RawPayloads,
) -> Result<i64, EvidenceError> {
    let mut bytes = record
        .attempt_id
        .as_str()
        .len()
        .saturating_add(record.group_id.len())
        .saturating_add(record.request_id.as_str().len())
        .saturating_add(record.upstream_profile.as_ref().map_or(0, String::len))
        .saturating_add(record.model_id.as_ref().map_or(0, String::len))
        .saturating_add(record.thinking_mode.as_ref().map_or(0, String::len))
        .saturating_add(request_metadata_json.len())
        .saturating_add(response_metadata_json.len())
        .saturating_add(detector_features_json.len());
    for (_channel, chunk) in raw_chunks(raw_payloads) {
        bytes = bytes.saturating_add(chunk.len());
    }
    to_sqlite_i64(bytes, "estimated_bytes")
}

fn resolve_sqlite_path(path: &Path) -> Result<PathBuf, EvidenceError> {
    if path.starts_with("~") {
        let home = env::var_os("HOME").ok_or(EvidenceError::HomeDirectoryUnavailable)?;
        let suffix = path.strip_prefix("~").unwrap_or(path);
        return Ok(PathBuf::from(home).join(suffix));
    }
    Ok(path.to_path_buf())
}

fn prepare_parent_directory(path: &Path) -> Result<(), EvidenceError> {
    let Some(parent) = path.parent() else {
        return Err(EvidenceError::UnsafeStoragePath {
            path: path.to_path_buf(),
            reason: "path must have a parent directory",
        });
    };
    if parent.as_os_str().is_empty() {
        return Err(EvidenceError::UnsafeStoragePath {
            path: path.to_path_buf(),
            reason: "path must have a parent directory",
        });
    }
    if parent.exists() {
        ensure_owner_private_directory(parent)?;
        return Ok(());
    }

    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(EVIDENCE_DIRECTORY_MODE);
    builder
        .create(parent)
        .map_err(|source| EvidenceError::CreateDirectory {
            path: parent.to_path_buf(),
            source,
        })?;
    ensure_owner_private_directory(parent)
}

#[cfg(unix)]
fn ensure_owner_private_directory(path: &Path) -> Result<(), EvidenceError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| EvidenceError::InspectPath {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(EvidenceError::UnsafeStoragePath {
            path: path.to_path_buf(),
            reason: "parent must be a real directory",
        });
    }
    let mode = metadata.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(EvidenceError::UnsafeStoragePath {
            path: path.to_path_buf(),
            reason: "parent directory must not be accessible by group or other users",
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_owner_private_directory(path: &Path) -> Result<(), EvidenceError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| EvidenceError::InspectPath {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_dir() {
        return Err(EvidenceError::UnsafeStoragePath {
            path: path.to_path_buf(),
            reason: "parent must be a directory",
        });
    }
    Ok(())
}

fn prepare_sqlite_file(path: &Path) -> Result<(), EvidenceError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(EvidenceError::UnsafeStoragePath {
                    path: path.to_path_buf(),
                    reason: "SQLite path must be a regular file",
                });
            }
            restrict_sqlite_permissions(path)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let mut options = fs::OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            options.mode(EVIDENCE_SQLITE_MODE);
            drop(
                options
                    .open(path)
                    .map_err(|source| EvidenceError::InspectPath {
                        path: path.to_path_buf(),
                        source,
                    })?,
            );
            restrict_sqlite_permissions(path)
        }
        Err(source) => Err(EvidenceError::InspectPath {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(unix)]
fn restrict_sqlite_permissions(path: &Path) -> Result<(), EvidenceError> {
    let permissions = fs::Permissions::from_mode(EVIDENCE_SQLITE_MODE);
    fs::set_permissions(path, permissions).map_err(|source| EvidenceError::RestrictPermissions {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn restrict_sqlite_permissions(_path: &Path) -> Result<(), EvidenceError> {
    Ok(())
}

fn open_sqlite_connection(path: &Path) -> Result<Connection, EvidenceError> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|source| EvidenceError::Sqlite {
        action: "open SQLite evidence store",
        source,
    })
}

fn require_nonempty(value: &str, kind: &'static str) -> Result<(), EvidenceError> {
    if value.is_empty() {
        Err(EvidenceError::EmptyIdentifier { kind })
    } else {
        Ok(())
    }
}

fn optional_to_sqlite_i64(
    value: Option<u64>,
    field: &'static str,
) -> Result<Option<i64>, EvidenceError> {
    value.map(|value| to_sqlite_i64(value, field)).transpose()
}

fn to_sqlite_i64(value: impl TryInto<i64>, field: &'static str) -> Result<i64, EvidenceError> {
    value
        .try_into()
        .map_err(|_error| EvidenceError::IntegerOutOfRange { field })
}

fn nonnegative_i64_to_u64(value: i64) -> u64 {
    u64::try_from(value.max(0)).unwrap_or(0)
}

fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_millis().try_into().unwrap_or(u64::MAX)
        })
}
