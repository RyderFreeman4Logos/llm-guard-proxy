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
        EvidenceAttemptRecord, EvidenceDatabaseStatus, EvidenceExportArtifact, EvidenceExportPair,
        EvidenceGroupRecord, EvidenceRawArtifactKind, EvidenceRetentionUsage, EvidenceStoreWrite,
        EvidenceSummaryRow, ShadowSkipReason,
    },
    redaction::{evidence_metadata_map, sanitize_raw_payloads, scrub_optional_text},
};
use llm_guard_proxy_core::{ConfigHandle, EvidenceConfig};

use crate::RawPayloads;

const SCHEMA_VERSION: i64 = 2;
const SHA256_HEX_LEN: usize = 64;
const SECONDS_PER_DAY: u64 = 86_400;
const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];
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
        let prepared_group = PreparedGroup::from_record(group, include_headers)?;
        let prepared_attempts = attempts
            .iter()
            .map(|attempt| {
                PreparedAttempt::from_record(attempt, include_headers, &settings.evidence)
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
            &settings.evidence,
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

    /// Returns raw paired-comparison schema support for an evidence database.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when an existing database cannot be read.
    pub fn database_status(path: &Path) -> Result<EvidenceDatabaseStatus, EvidenceError> {
        let sqlite_path = resolve_sqlite_path(path)?;
        if !sqlite_path.exists() {
            return Ok(EvidenceDatabaseStatus {
                exists: false,
                schema_version: None,
                supports_raw_paired_comparison: false,
                has_attempt_raw_columns: false,
                has_raw_artifact_table: false,
            });
        }
        let connection = open_existing_sqlite_connection(&sqlite_path)?;
        let schema_version = read_schema_version(&connection)?;
        let has_attempt_raw_columns = table_has_columns(
            &connection,
            "evidence_attempts",
            &["raw_input", "raw_output", "raw_reasoning", "raw_tool_calls"],
        )?;
        let has_raw_artifact_table = table_exists(&connection, "evidence_raw_artifacts")?;
        Ok(EvidenceDatabaseStatus {
            exists: true,
            schema_version: Some(schema_version),
            supports_raw_paired_comparison: has_attempt_raw_columns && has_raw_artifact_table,
            has_attempt_raw_columns,
            has_raw_artifact_table,
        })
    }

    /// Returns raw artifact counts grouped by role, variant, and artifact kind.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when an existing database cannot be queried.
    pub fn summary(path: &Path) -> Result<Vec<EvidenceSummaryRow>, EvidenceError> {
        let sqlite_path = resolve_sqlite_path(path)?;
        if !sqlite_path.exists() {
            return Ok(Vec::new());
        }
        let connection = open_existing_sqlite_connection(&sqlite_path)?;
        if !table_exists(&connection, "evidence_raw_artifacts")? {
            return Ok(Vec::new());
        }
        read_summary_rows(&connection)
    }

    /// Exports paired raw artifacts for offline scoring.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError`] when an existing database cannot be queried.
    pub fn export_pairs(
        path: &Path,
        variants: &[String],
        include: &[EvidenceRawArtifactKind],
    ) -> Result<Vec<EvidenceExportPair>, EvidenceError> {
        let sqlite_path = resolve_sqlite_path(path)?;
        if !sqlite_path.exists() || variants.is_empty() || include.is_empty() {
            return Ok(Vec::new());
        }
        let connection = open_existing_sqlite_connection(&sqlite_path)?;
        if !table_exists(&connection, "evidence_raw_artifacts")? {
            return Ok(Vec::new());
        }
        read_export_pairs(&connection, variants, include)
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
    raw_artifacts: Vec<PreparedRawArtifact>,
    estimated_bytes: i64,
}

impl PreparedAttempt {
    fn from_record(
        record: &EvidenceAttemptRecord,
        include_headers: bool,
        config: &EvidenceConfig,
    ) -> Result<Self, EvidenceError> {
        require_nonempty(&record.group_id, "group")?;
        let request_metadata = evidence_metadata_map(&record.request_metadata, include_headers);
        let response_metadata = evidence_metadata_map(&record.response_metadata, include_headers);
        let detector_features = evidence_metadata_map(&record.detector_features, false);
        let request_metadata_json = metadata_json(&request_metadata, "evidence attempt request")?;
        let response_metadata_json =
            metadata_json(&response_metadata, "evidence attempt response")?;
        let detector_features_json = metadata_json(&detector_features, "evidence detector")?;
        let raw_policy = RawCapturePolicy::from_record(record, config);
        let prepared_raw = prepare_raw_payloads(&record.raw_payloads, &raw_policy);
        let estimated_bytes = estimate_attempt_bytes(
            record,
            &request_metadata_json,
            &response_metadata_json,
            &detector_features_json,
            &prepared_raw.payloads,
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
            raw_payloads: prepared_raw.payloads,
            raw_artifacts: prepared_raw.artifacts,
            estimated_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawCaptureFieldPolicy {
    enabled: bool,
    max_bytes: Option<u64>,
}

impl RawCaptureFieldPolicy {
    const DISABLED: Self = Self {
        enabled: false,
        max_bytes: None,
    };

    const UNBOUNDED: Self = Self {
        enabled: true,
        max_bytes: None,
    };

    const fn bounded(max_bytes: u64) -> Self {
        Self {
            enabled: true,
            max_bytes: Some(max_bytes),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawCapturePolicy {
    input: RawCaptureFieldPolicy,
    output: RawCaptureFieldPolicy,
    reasoning: RawCaptureFieldPolicy,
    tool_calls: RawCaptureFieldPolicy,
    include_chunks: bool,
}

impl RawCapturePolicy {
    const NONE: Self = Self {
        input: RawCaptureFieldPolicy::DISABLED,
        output: RawCaptureFieldPolicy::DISABLED,
        reasoning: RawCaptureFieldPolicy::DISABLED,
        tool_calls: RawCaptureFieldPolicy::DISABLED,
        include_chunks: false,
    };

    const UNBOUNDED_ALL: Self = Self {
        input: RawCaptureFieldPolicy::UNBOUNDED,
        output: RawCaptureFieldPolicy::UNBOUNDED,
        reasoning: RawCaptureFieldPolicy::UNBOUNDED,
        tool_calls: RawCaptureFieldPolicy::UNBOUNDED,
        include_chunks: true,
    };

    fn from_record(record: &EvidenceAttemptRecord, config: &EvidenceConfig) -> Self {
        if config.include_raw_payloads {
            return Self::UNBOUNDED_ALL;
        }
        if !is_paired_comparison_attempt(record) {
            return Self::NONE;
        }
        let paired = &config.shadow.paired_comparison;
        Self {
            input: if paired.include_raw_input {
                RawCaptureFieldPolicy::bounded(paired.max_raw_input_bytes)
            } else {
                RawCaptureFieldPolicy::DISABLED
            },
            output: if paired.include_raw_output {
                RawCaptureFieldPolicy::bounded(paired.max_raw_output_bytes)
            } else {
                RawCaptureFieldPolicy::DISABLED
            },
            reasoning: if paired.include_raw_reasoning {
                RawCaptureFieldPolicy::bounded(paired.max_raw_reasoning_bytes)
            } else {
                RawCaptureFieldPolicy::DISABLED
            },
            tool_calls: if paired.include_raw_output {
                RawCaptureFieldPolicy::bounded(paired.max_raw_output_bytes)
            } else {
                RawCaptureFieldPolicy::DISABLED
            },
            include_chunks: false,
        }
    }
}

#[derive(Debug)]
struct PreparedRawPayloads {
    payloads: RawPayloads,
    artifacts: Vec<PreparedRawArtifact>,
}

#[derive(Debug)]
struct PreparedRawArtifact {
    kind: EvidenceRawArtifactKind,
    content: String,
    bytes_original: i64,
    bytes_stored: i64,
    truncated: i64,
    redacted: i64,
    sha256: String,
}

fn prepare_raw_payloads(raw: &RawPayloads, policy: &RawCapturePolicy) -> PreparedRawPayloads {
    let sanitized = sanitize_raw_payloads(raw, true);
    let mut payloads = RawPayloads::default();
    let mut artifacts = Vec::new();

    if let Some((content, artifact)) = prepare_raw_artifact(
        raw.input.as_deref(),
        sanitized.input.as_deref(),
        EvidenceRawArtifactKind::Input,
        policy.input,
    ) {
        payloads.input = Some(content);
        artifacts.push(artifact);
    }
    if let Some((content, artifact)) = prepare_raw_artifact(
        raw.output.as_deref(),
        sanitized.output.as_deref(),
        EvidenceRawArtifactKind::Output,
        policy.output,
    ) {
        payloads.output = Some(content);
        artifacts.push(artifact);
    }
    if let Some((content, artifact)) = prepare_raw_artifact(
        raw.reasoning.as_deref(),
        sanitized.reasoning.as_deref(),
        EvidenceRawArtifactKind::Reasoning,
        policy.reasoning,
    ) {
        payloads.reasoning = Some(content);
        artifacts.push(artifact);
    }
    if let Some((content, artifact)) = prepare_raw_artifact(
        raw.tool_calls.as_deref(),
        sanitized.tool_calls.as_deref(),
        EvidenceRawArtifactKind::ToolCalls,
        policy.tool_calls,
    ) {
        payloads.tool_calls = Some(content);
        artifacts.push(artifact);
    }
    if policy.include_chunks {
        payloads.chunks = sanitized.chunks;
    }

    PreparedRawPayloads {
        payloads,
        artifacts,
    }
}

fn prepare_raw_artifact(
    original: Option<&str>,
    sanitized: Option<&str>,
    kind: EvidenceRawArtifactKind,
    policy: RawCaptureFieldPolicy,
) -> Option<(String, PreparedRawArtifact)> {
    if !policy.enabled {
        return None;
    }
    let original = original?;
    let sanitized = sanitized.unwrap_or(original);
    let (content, truncated) = truncate_utf8(sanitized, policy.max_bytes);
    let artifact = PreparedRawArtifact {
        kind,
        bytes_original: usize_to_i64_saturating(original.len()),
        bytes_stored: usize_to_i64_saturating(content.len()),
        truncated: bool_to_sqlite_i64(truncated),
        redacted: bool_to_sqlite_i64(original != sanitized),
        sha256: sha256_hex(content.as_bytes()),
        content: content.clone(),
    };
    Some((content, artifact))
}

fn truncate_utf8(value: &str, max_bytes: Option<u64>) -> (String, bool) {
    let Some(max_bytes) = max_bytes.and_then(|value| usize::try_from(value).ok()) else {
        return (value.to_owned(), false);
    };
    if value.len() <= max_bytes {
        return (value.to_owned(), false);
    }
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next_end = index + character.len_utf8();
        if next_end > max_bytes {
            break;
        }
        end = next_end;
    }
    (value[..end].to_owned(), true)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256_digest(bytes);
    let mut output = String::with_capacity(SHA256_HEX_LEN);
    for byte in digest {
        output.push(hex_digit(byte >> 4));
        output.push(hex_digit(byte & 0x0f));
    }
    output
}

fn sha256_digest(input: &[u8]) -> [u8; 32] {
    let bit_len = u64::try_from(input.len())
        .unwrap_or(u64::MAX / 8)
        .saturating_mul(8);
    let mut message = Vec::with_capacity(input.len().saturating_add(72));
    message.extend_from_slice(input);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = SHA256_INITIAL_STATE;
    for chunk in message.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (word, bytes) in words.iter_mut().take(16).zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let sigma0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let sigma1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(words[index - 7])
                .wrapping_add(sigma1);
        }

        let [
            mut working_a,
            mut working_b,
            mut working_c,
            mut working_d,
            mut working_e,
            mut working_f,
            mut working_g,
            mut working_h,
        ] = state;
        for (index, constant) in SHA256_ROUND_CONSTANTS.iter().enumerate() {
            let big_sigma1 =
                working_e.rotate_right(6) ^ working_e.rotate_right(11) ^ working_e.rotate_right(25);
            let choose = (working_e & working_f) ^ ((!working_e) & working_g);
            let temp1 = working_h
                .wrapping_add(big_sigma1)
                .wrapping_add(choose)
                .wrapping_add(*constant)
                .wrapping_add(words[index]);
            let big_sigma0 =
                working_a.rotate_right(2) ^ working_a.rotate_right(13) ^ working_a.rotate_right(22);
            let majority =
                (working_a & working_b) ^ (working_a & working_c) ^ (working_b & working_c);
            let temp2 = big_sigma0.wrapping_add(majority);
            working_h = working_g;
            working_g = working_f;
            working_f = working_e;
            working_e = working_d.wrapping_add(temp1);
            working_d = working_c;
            working_c = working_b;
            working_b = working_a;
            working_a = temp1.wrapping_add(temp2);
        }

        for (value, addend) in state.iter_mut().zip([
            working_a, working_b, working_c, working_d, working_e, working_f, working_g, working_h,
        ]) {
            *value = value.wrapping_add(addend);
        }
    }

    let mut digest = [0_u8; 32];
    for (index, value) in state.into_iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    digest
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        _ => char::from(b'a' + (value - 10)),
    }
}

fn is_paired_comparison_attempt(record: &EvidenceAttemptRecord) -> bool {
    record
        .request_metadata
        .get("shadow_paired_comparison")
        .or_else(|| record.response_metadata.get("shadow_paired_comparison"))
        .is_some_and(|value| value == "true")
}

fn bool_to_sqlite_i64(value: bool) -> i64 {
    i64::from(value)
}

fn usize_to_i64_saturating(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
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
    match version {
        0 => create_schema(connection),
        1 => migrate_schema_v2(connection),
        _ => Ok(()),
    }
}

#[allow(clippy::too_many_lines)]
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
    estimated_bytes INTEGER NOT NULL
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

CREATE TABLE IF NOT EXISTS evidence_raw_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES evidence_attempts(attempt_id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    role TEXT NOT NULL,
    variant_name TEXT,
    artifact_kind TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    content_text TEXT,
    bytes_original INTEGER NOT NULL,
    bytes_stored INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    redacted INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    pruned_at_unix_ms INTEGER,
    UNIQUE(attempt_id, artifact_kind)
);

CREATE INDEX IF NOT EXISTS evidence_raw_artifacts_group_variant_idx
    ON evidence_raw_artifacts(group_id, variant_name, artifact_kind);
CREATE INDEX IF NOT EXISTS evidence_raw_artifacts_retention_idx
    ON evidence_raw_artifacts(created_at_unix_ms, artifact_id)
    WHERE content_text IS NOT NULL;

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

PRAGMA user_version = 2;
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "create SQLite evidence schema",
            source,
        })
}

#[allow(clippy::too_many_lines)]
fn migrate_schema_v2(connection: &Connection) -> Result<(), EvidenceError> {
    connection
        .execute_batch(
            r"
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE;

DROP TABLE IF EXISTS evidence_attempts_v2;

CREATE TABLE evidence_attempts_v2 (
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
    estimated_bytes INTEGER NOT NULL
);

INSERT INTO evidence_attempts_v2 (
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
)
SELECT
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
FROM evidence_attempts;

DROP TABLE evidence_attempts;
ALTER TABLE evidence_attempts_v2 RENAME TO evidence_attempts;

CREATE INDEX IF NOT EXISTS evidence_attempts_group_id_idx
    ON evidence_attempts(group_id);
CREATE INDEX IF NOT EXISTS evidence_attempts_role_idx
    ON evidence_attempts(role);
CREATE INDEX IF NOT EXISTS evidence_attempts_status_idx
    ON evidence_attempts(status);

CREATE TABLE IF NOT EXISTS evidence_raw_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES evidence_attempts(attempt_id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    role TEXT NOT NULL,
    variant_name TEXT,
    artifact_kind TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    content_text TEXT,
    bytes_original INTEGER NOT NULL,
    bytes_stored INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    redacted INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    pruned_at_unix_ms INTEGER,
    UNIQUE(attempt_id, artifact_kind)
);

CREATE INDEX IF NOT EXISTS evidence_raw_artifacts_group_variant_idx
    ON evidence_raw_artifacts(group_id, variant_name, artifact_kind);
CREATE INDEX IF NOT EXISTS evidence_raw_artifacts_retention_idx
    ON evidence_raw_artifacts(created_at_unix_ms, artifact_id)
    WHERE content_text IS NOT NULL;

PRAGMA user_version = 2;
COMMIT;
PRAGMA foreign_keys = ON;
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "migrate SQLite evidence schema to v2",
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
    insert_raw_chunks_in_transaction(transaction, attempt)?;
    insert_raw_artifacts_in_transaction(transaction, attempt)
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

fn insert_raw_artifacts_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    attempt: &PreparedAttempt,
) -> Result<(), EvidenceError> {
    transaction
        .execute(
            "DELETE FROM evidence_raw_artifacts WHERE attempt_id = ?1",
            params![attempt.attempt_id],
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "replace evidence raw artifacts",
            source,
        })?;
    let created_at_unix_ms = to_sqlite_i64(unix_time_millis(), "created_at_unix_ms")?;
    let variant_name = raw_artifact_variant_name(attempt);
    for artifact in &attempt.raw_artifacts {
        transaction
            .execute(
                r"
INSERT INTO evidence_raw_artifacts (
    attempt_id,
    group_id,
    request_id,
    role,
    variant_name,
    artifact_kind,
    created_at_unix_ms,
    content_text,
    bytes_original,
    bytes_stored,
    truncated,
    redacted,
    sha256,
    pruned_at_unix_ms
) VALUES (
    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, NULL
)
",
                params![
                    attempt.attempt_id,
                    attempt.group_id,
                    attempt.request_id,
                    attempt.role,
                    variant_name.as_deref(),
                    artifact.kind.as_str(),
                    created_at_unix_ms,
                    artifact.content.as_str(),
                    artifact.bytes_original,
                    artifact.bytes_stored,
                    artifact.truncated,
                    artifact.redacted,
                    artifact.sha256.as_str(),
                ],
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "write evidence raw artifact",
                source,
            })?;
    }
    Ok(())
}

fn raw_artifact_variant_name(attempt: &PreparedAttempt) -> Option<String> {
    metadata_map_from_json(&attempt.request_metadata_json)
        .ok()
        .and_then(|metadata| {
            metadata
                .get("variant_name")
                .or_else(|| metadata.get("shadow_compare_attempt"))
                .cloned()
        })
        .or_else(|| {
            metadata_map_from_json(&attempt.response_metadata_json)
                .ok()
                .and_then(|metadata| {
                    metadata
                        .get("variant_name")
                        .or_else(|| metadata.get("shadow_compare_attempt"))
                        .cloned()
                })
        })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RetentionPruneOutcome {
    groups: u64,
    attempts: u64,
    chunks: u64,
    raw_artifacts: u64,
}

impl RetentionPruneOutcome {
    const fn deleted_any(self) -> bool {
        self.groups > 0 || self.attempts > 0 || self.chunks > 0 || self.raw_artifacts > 0
    }

    fn add(&mut self, other: Self) {
        self.groups = self.groups.saturating_add(other.groups);
        self.attempts = self.attempts.saturating_add(other.attempts);
        self.chunks = self.chunks.saturating_add(other.chunks);
        self.raw_artifacts = self.raw_artifacts.saturating_add(other.raw_artifacts);
    }
}

fn enforce_retention(
    connection: &mut Connection,
    config: &EvidenceConfig,
) -> Result<RetentionPruneOutcome, EvidenceError> {
    let mut outcome = enforce_raw_artifact_retention(connection, config)?;
    if outcome.deleted_any() {
        vacuum_database(connection)?;
    }
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawArtifactPointer {
    artifact_id: i64,
    attempt_id: String,
    kind: String,
    created_at_unix_ms: u64,
    bytes_stored: u64,
}

fn enforce_raw_artifact_retention(
    connection: &mut Connection,
    config: &EvidenceConfig,
) -> Result<RetentionPruneOutcome, EvidenceError> {
    if !table_exists(connection, "evidence_raw_artifacts")? {
        return Ok(RetentionPruneOutcome::default());
    }
    let artifacts = read_content_raw_artifacts(connection)?;
    if artifacts.is_empty() {
        return Ok(RetentionPruneOutcome::default());
    }
    let now = unix_time_millis();
    let retention_window_ms = config
        .shadow
        .paired_comparison
        .retention_days
        .saturating_mul(SECONDS_PER_DAY)
        .saturating_mul(1_000);
    let expire_before = now.saturating_sub(retention_window_ms);
    let mut remaining_records = u64::try_from(artifacts.len()).unwrap_or(u64::MAX);
    let mut remaining_bytes = artifacts.iter().fold(0_u64, |total, artifact| {
        total.saturating_add(artifact.bytes_stored)
    });
    let mut prune = Vec::new();

    for artifact in artifacts {
        let expired = artifact.created_at_unix_ms < expire_before;
        let over_records =
            remaining_records > config.shadow.paired_comparison.max_retention_records;
        let over_bytes = remaining_bytes > config.shadow.paired_comparison.max_retention_bytes;
        if expired || over_records || over_bytes {
            remaining_records = remaining_records.saturating_sub(1);
            remaining_bytes = remaining_bytes.saturating_sub(artifact.bytes_stored);
            prune.push(artifact);
        }
    }

    prune_raw_artifact_contents(connection, &prune)
}

fn read_content_raw_artifacts(
    connection: &Connection,
) -> Result<Vec<RawArtifactPointer>, EvidenceError> {
    let mut statement = connection
        .prepare(
            r"
SELECT artifact_id, attempt_id, artifact_kind, created_at_unix_ms, bytes_stored
FROM evidence_raw_artifacts
WHERE content_text IS NOT NULL
ORDER BY created_at_unix_ms, artifact_id
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "prepare raw artifact retention query",
            source,
        })?;
    let rows = statement
        .query_map([], |row| {
            let created_at_unix_ms: i64 = row.get(3)?;
            let bytes_stored: i64 = row.get(4)?;
            Ok(RawArtifactPointer {
                artifact_id: row.get(0)?,
                attempt_id: row.get(1)?,
                kind: row.get(2)?,
                created_at_unix_ms: nonnegative_i64_to_u64(created_at_unix_ms),
                bytes_stored: nonnegative_i64_to_u64(bytes_stored),
            })
        })
        .map_err(|source| EvidenceError::Sqlite {
            action: "read raw artifacts for retention",
            source,
        })?;
    rows.map(|row| {
        row.map_err(|source| EvidenceError::Sqlite {
            action: "decode raw artifact retention row",
            source,
        })
    })
    .collect()
}

fn prune_raw_artifact_contents(
    connection: &mut Connection,
    artifacts: &[RawArtifactPointer],
) -> Result<RetentionPruneOutcome, EvidenceError> {
    if artifacts.is_empty() {
        return Ok(RetentionPruneOutcome::default());
    }
    let transaction = connection
        .transaction()
        .map_err(|source| EvidenceError::Sqlite {
            action: "start raw artifact retention transaction",
            source,
        })?;
    let pruned_at = to_sqlite_i64(unix_time_millis(), "raw_artifact.pruned_at_unix_ms")?;
    let mut pruned_artifacts = 0_u64;
    let mut affected_attempts = Vec::<String>::new();

    for artifact in artifacts {
        let updated = transaction
            .execute(
                r"
UPDATE evidence_raw_artifacts
SET content_text = NULL,
    bytes_stored = 0,
    pruned_at_unix_ms = ?1
WHERE artifact_id = ?2
  AND content_text IS NOT NULL
",
                params![pruned_at, artifact.artifact_id],
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "prune raw artifact content",
                source,
            })?;
        if updated > 0 {
            pruned_artifacts = pruned_artifacts.saturating_add(1);
            affected_attempts.push(artifact.attempt_id.clone());
            null_attempt_raw_column(&transaction, &artifact.attempt_id, &artifact.kind)?;
        }
    }
    affected_attempts.sort();
    affected_attempts.dedup();
    let pruned_chunks = delete_chunks_for_attempts(&transaction, &affected_attempts)?;

    transaction
        .commit()
        .map_err(|source| EvidenceError::Sqlite {
            action: "commit raw artifact retention transaction",
            source,
        })?;
    Ok(RetentionPruneOutcome {
        groups: 0,
        attempts: 0,
        chunks: pruned_chunks,
        raw_artifacts: pruned_artifacts,
    })
}

fn null_attempt_raw_column(
    transaction: &rusqlite::Transaction<'_>,
    attempt_id: &str,
    kind: &str,
) -> Result<(), EvidenceError> {
    let column = match kind {
        "input" => "raw_input",
        "output" => "raw_output",
        "reasoning" => "raw_reasoning",
        "tool_calls" => "raw_tool_calls",
        _ => return Ok(()),
    };
    let sql = format!("UPDATE evidence_attempts SET {column} = NULL WHERE attempt_id = ?1");
    transaction
        .execute(&sql, params![attempt_id])
        .map(|_updated| ())
        .map_err(|source| EvidenceError::Sqlite {
            action: "clear retained raw attempt column",
            source,
        })
}

fn delete_chunks_for_attempts(
    transaction: &rusqlite::Transaction<'_>,
    attempt_ids: &[String],
) -> Result<u64, EvidenceError> {
    let mut deleted = 0_u64;
    for attempt_id in attempt_ids {
        let count: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM evidence_chunks WHERE attempt_id = ?1",
                params![attempt_id],
                |row| row.get(0),
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "count raw chunks before pruning",
                source,
            })?;
        transaction
            .execute(
                "DELETE FROM evidence_chunks WHERE attempt_id = ?1",
                params![attempt_id],
            )
            .map_err(|source| EvidenceError::Sqlite {
                action: "prune raw chunks for attempt",
                source,
            })?;
        deleted = deleted.saturating_add(nonnegative_i64_to_u64(count));
    }
    Ok(deleted)
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

fn table_exists(connection: &Connection, table: &str) -> Result<bool, EvidenceError> {
    let exists: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            params![table],
            |row| row.get(0),
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "check evidence table existence",
            source,
        })?;
    Ok(exists > 0)
}

fn table_has_columns(
    connection: &Connection,
    table: &str,
    required_columns: &[&str],
) -> Result<bool, EvidenceError> {
    if !table_exists(connection, table)? {
        return Ok(false);
    }
    let mut statement = connection
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|source| EvidenceError::Sqlite {
            action: "inspect evidence table columns",
            source,
        })?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence table columns",
            source,
        })?;
    let mut columns = Vec::new();
    for row in rows {
        columns.push(row.map_err(|source| EvidenceError::Sqlite {
            action: "decode evidence table column",
            source,
        })?);
    }
    Ok(required_columns
        .iter()
        .all(|required| columns.iter().any(|column| column.as_str() == *required)))
}

fn read_summary_rows(connection: &Connection) -> Result<Vec<EvidenceSummaryRow>, EvidenceError> {
    let mut statement = connection
        .prepare(
            r"
SELECT
    role,
    COALESCE(variant_name, 'primary') AS variant_name,
    artifact_kind,
    COUNT(*) AS artifact_count,
    SUM(CASE WHEN content_text IS NULL THEN 0 ELSE 1 END) AS content_present_count,
    COALESCE(SUM(bytes_stored), 0) AS bytes_stored
FROM evidence_raw_artifacts
GROUP BY role, COALESCE(variant_name, 'primary'), artifact_kind
ORDER BY role, variant_name, artifact_kind
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "prepare evidence summary query",
            source,
        })?;
    let rows = statement
        .query_map([], |row| {
            let artifact_count: i64 = row.get(3)?;
            let content_present_count: i64 = row.get(4)?;
            let bytes_stored: i64 = row.get(5)?;
            Ok(EvidenceSummaryRow {
                role: row.get(0)?,
                variant_name: row.get(1)?,
                artifact_kind: row.get(2)?,
                artifact_count: nonnegative_i64_to_u64(artifact_count),
                content_present_count: nonnegative_i64_to_u64(content_present_count),
                bytes_stored: nonnegative_i64_to_u64(bytes_stored),
            })
        })
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence summary rows",
            source,
        })?;
    rows.map(|row| {
        row.map_err(|source| EvidenceError::Sqlite {
            action: "decode evidence summary row",
            source,
        })
    })
    .collect()
}

fn read_export_pairs(
    connection: &Connection,
    variants: &[String],
    include: &[EvidenceRawArtifactKind],
) -> Result<Vec<EvidenceExportPair>, EvidenceError> {
    let include_labels = include.iter().map(|kind| kind.as_str()).collect::<Vec<_>>();
    let mut statement = connection
        .prepare(
            r"
SELECT
    group_id,
    request_id,
    artifact_kind,
    variant_name,
    attempt_id,
    role,
    content_text,
    bytes_original,
    bytes_stored,
    truncated,
    redacted,
    sha256
FROM evidence_raw_artifacts
WHERE content_text IS NOT NULL
  AND variant_name IS NOT NULL
ORDER BY group_id, artifact_kind, variant_name
",
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "prepare evidence pair export query",
            source,
        })?;
    let rows = statement
        .query_map([], |row| {
            let bytes_original: i64 = row.get(7)?;
            let bytes_stored: i64 = row.get(8)?;
            let truncated: i64 = row.get(9)?;
            let redacted: i64 = row.get(10)?;
            Ok(RawExportRow {
                group_id: row.get(0)?,
                request_id: row.get(1)?,
                artifact_kind: row.get(2)?,
                variant_name: row.get(3)?,
                artifact: EvidenceExportArtifact {
                    attempt_id: row.get(4)?,
                    role: row.get(5)?,
                    content: row.get(6)?,
                    bytes_original: nonnegative_i64_to_u64(bytes_original),
                    bytes_stored: nonnegative_i64_to_u64(bytes_stored),
                    truncated: truncated != 0,
                    redacted: redacted != 0,
                    sha256: row.get(11)?,
                },
            })
        })
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence pair export rows",
            source,
        })?;
    let mut grouped = BTreeMap::<(String, String, String), EvidenceExportPair>::new();
    for row in rows {
        let row = row.map_err(|source| EvidenceError::Sqlite {
            action: "decode evidence pair export row",
            source,
        })?;
        if !variants.contains(&row.variant_name)
            || !include_labels.contains(&row.artifact_kind.as_str())
        {
            continue;
        }
        let key = (
            row.group_id.clone(),
            row.request_id.clone(),
            row.artifact_kind.clone(),
        );
        grouped
            .entry(key)
            .or_insert_with(|| EvidenceExportPair {
                group_id: row.group_id.clone(),
                request_id: row.request_id.clone(),
                artifact_kind: row.artifact_kind.clone(),
                variants: BTreeMap::new(),
            })
            .variants
            .insert(row.variant_name, row.artifact);
    }
    Ok(grouped
        .into_values()
        .filter(|pair| {
            variants
                .iter()
                .all(|variant| pair.variants.contains_key(variant))
        })
        .collect())
}

struct RawExportRow {
    group_id: String,
    request_id: String,
    artifact_kind: String,
    variant_name: String,
    artifact: EvidenceExportArtifact,
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
    let raw_artifact_bytes = if table_exists(connection, "evidence_raw_artifacts")? {
        read_sum_raw_artifact_bytes(connection)?
    } else {
        0
    };
    Ok(group_bytes
        .saturating_add(attempt_bytes)
        .saturating_add(nonnegative_i64_to_u64(chunk_bytes))
        .saturating_add(raw_artifact_bytes))
}

fn read_sum_raw_artifact_bytes(connection: &Connection) -> Result<u64, EvidenceError> {
    let bytes: i64 = connection
        .query_row(
            "SELECT COALESCE(SUM(bytes_stored), 0) FROM evidence_raw_artifacts",
            [],
            |row| row.get(0),
        )
        .map_err(|source| EvidenceError::Sqlite {
            action: "read evidence raw artifact logical bytes",
            source,
        })?;
    Ok(nonnegative_i64_to_u64(bytes))
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

fn metadata_map_from_json(value: &str) -> Result<BTreeMap<String, String>, EvidenceError> {
    serde_json::from_str(value).map_err(|source| EvidenceError::SerializeMetadata {
        field: "evidence metadata",
        source,
    })
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

fn open_existing_sqlite_connection(path: &Path) -> Result<Connection, EvidenceError> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|source| EvidenceError::Sqlite {
        action: "open existing SQLite evidence store",
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
