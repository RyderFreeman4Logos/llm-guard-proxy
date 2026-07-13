use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use rusqlite::Connection;

use super::{
    EvidenceAttemptRecord, EvidenceAttemptRole, EvidenceAttemptStatus, EvidenceGroupRecord,
    EvidenceRawArtifactKind, EvidenceStore, EvidenceStoreWrite, ShadowSkipReason,
};
use llm_guard_proxy_core::{AppConfig, ConfigHandle};

use crate::{AttemptId, RawPayloadChunk, RawPayloads, RequestId, RequestStatus};

const TEST_MAX_BYTES: u64 = 1_000_000;
const TEST_PRUNE_TO_BYTES: u64 = 800_000;

#[test]
fn disabled_evidence_write_creates_no_artifacts() {
    let fixture = EvidenceFixture::new("disabled");
    let manager = fixture.manager(false, false, false, 100, None);
    let store = EvidenceStore::open(manager.clone());

    let write = store
        .record_group(&group_record("group-disabled", 1_000), &[])
        .expect("disabled write should not fail");

    assert_eq!(write, EvidenceStoreWrite::Disabled);
    assert!(!fixture.sqlite_path.exists());
    assert!(!fixture.blob_cache_dir.exists());
}

#[test]
fn content_free_evidence_records_correlated_attempts_without_raw_payloads() {
    let fixture = EvidenceFixture::new("content-free");
    let manager = fixture.manager(true, false, false, 100, None);
    let store = EvidenceStore::open(manager.clone());
    let group = group_record("group-content-free", 1_000);
    let attempts = vec![
        attempt_record(
            "group-content-free",
            1,
            EvidenceAttemptRole::Primary,
            EvidenceAttemptStatus::Rejected,
            false,
        ),
        attempt_record(
            "group-content-free",
            2,
            EvidenceAttemptRole::Fallback,
            EvidenceAttemptStatus::Accepted,
            true,
        ),
    ];

    let write = store
        .record_group(&group, &attempts)
        .expect("enabled evidence should write");

    assert_eq!(write, EvidenceStoreWrite::Written);
    let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_groups"),
        1
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_attempts"),
        2
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_chunks"),
        0
    );

    let rows = evidence_attempt_roles(&connection);
    assert_eq!(
        rows,
        vec![
            (String::from("primary"), 0, String::from("rejected")),
            (String::from("fallback"), 1, String::from("accepted")),
        ]
    );
    let metadata_json: String = connection
        .query_row(
            "SELECT request_metadata_json FROM evidence_attempts WHERE role = 'primary'",
            [],
            |row| row.get(0),
        )
        .expect("metadata should exist");
    assert!(!metadata_json.contains("request_header_authorization"));
    assert!(!metadata_json.contains("secret"));
    assert!(metadata_json.contains("attempt_thinking_budget_tokens"));

    let raw_count = count_rows(
        &connection,
        "SELECT COUNT(*) FROM evidence_attempts WHERE raw_input IS NOT NULL OR raw_output IS NOT NULL OR raw_reasoning IS NOT NULL OR raw_tool_calls IS NOT NULL",
    );
    assert_eq!(raw_count, 0);
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_raw_artifacts"),
        0
    );
}

#[test]
fn paired_raw_artifacts_record_input_output_without_reasoning() {
    let fixture = EvidenceFixture::new("paired-raw-artifacts");
    let manager = fixture.manager_with_extra(
        true,
        false,
        false,
        100,
        None,
        "
[evidence.shadow.paired_comparison]
enabled = true
include_raw_input = true
include_raw_output = true
include_raw_reasoning = false
sample_rate = 1.0
max_raw_input_bytes = 8
max_raw_output_bytes = 7
max_raw_reasoning_bytes = 8
",
    );
    let store = EvidenceStore::open(manager.clone());
    let group_id = "group-paired-raw";
    let mut max_thinking = paired_shadow_attempt(group_id, "max-thinking");
    max_thinking.raw_payloads = RawPayloads {
        input: Some(String::from("same-prompt")),
        output: Some(String::from("max answer")),
        reasoning: Some(String::from("max reasoning should not persist")),
        tool_calls: None,
        chunks: Vec::new(),
    };
    let mut no_thinking = paired_shadow_attempt(group_id, "no-thinking");
    no_thinking.raw_payloads = RawPayloads {
        input: Some(String::from("same-prompt")),
        output: Some(String::from("no answer")),
        reasoning: Some(String::from("no reasoning should not persist")),
        tool_calls: None,
        chunks: Vec::new(),
    };

    store
        .record_group(&group_record(group_id, 1_000), &[max_thinking, no_thinking])
        .expect("paired raw evidence should write");

    let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
    let rows = read_raw_artifacts(&connection);
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows.iter()
            .filter(|row| row.variant_name == "max-thinking")
            .count(),
        2
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.variant_name == "no-thinking")
            .count(),
        2
    );
    assert!(rows.iter().all(|row| row.artifact_kind != "reasoning"));
    assert!(
        rows.iter()
            .filter(|row| row.artifact_kind == "input")
            .all(|row| row.content_text == "same-pro" && row.truncated == 1)
    );
    assert!(
        rows.iter()
            .filter(|row| row.artifact_kind == "output")
            .all(|row| row.bytes_stored <= 7 && row.sha256.len() == 64)
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE raw_input IS NOT NULL AND raw_output IS NOT NULL AND raw_reasoning IS NULL",
        ),
        2
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_chunks"),
        4
    );
    let status =
        EvidenceStore::database_status(&fixture.sqlite_path).expect("database status should read");
    assert!(status.supports_raw_paired_comparison);
    let summary = EvidenceStore::summary(&fixture.sqlite_path).expect("summary should read");
    assert_eq!(summary.len(), 4);
    let pairs = EvidenceStore::export_pairs(
        &fixture.sqlite_path,
        &[String::from("max-thinking"), String::from("no-thinking")],
        &[
            EvidenceRawArtifactKind::Input,
            EvidenceRawArtifactKind::Output,
        ],
    )
    .expect("paired export should read");
    assert_eq!(pairs.len(), 2);
    assert!(
        pairs
            .iter()
            .all(|pair| pair.variants.contains_key("max-thinking")
                && pair.variants.contains_key("no-thinking"))
    );
}

#[test]
fn raw_evidence_capture_redacts_headers_and_secret_fragments() {
    let fixture = EvidenceFixture::new("raw-redaction");
    let manager = fixture.manager(true, true, true, 100, None);
    let store = EvidenceStore::open(manager.clone());
    let group = group_record("group-raw", 1_000);
    let mut attempt = attempt_record(
        "group-raw",
        1,
        EvidenceAttemptRole::Primary,
        EvidenceAttemptStatus::Accepted,
        true,
    );
    attempt.raw_payloads = RawPayloads {
        input: Some(String::from(
            r#"prompt {"content":"Bearer request-secret"} Bearer:colon-secret Bearer=equals-secret"#,
        )),
        output: Some(String::from(
            r#"answer «redacted:sk-…» useful {"delta":"Bearer out"} tail"#,
        )),
        reasoning: Some(String::from(
            "reasoning token=reasoning-secret still useful",
        )),
        tool_calls: Some(String::from(
            r#"{"arguments":{"api_key":"tool-secret","ok":true}}"#,
        )),
        chunks: vec![
            RawPayloadChunk::new("content", "first"),
            RawPayloadChunk::new("content", "Bearer chunk-secret"),
            RawPayloadChunk::new("reasoning", "second"),
        ],
    };

    store
        .record_group(&group, &[attempt])
        .expect("raw evidence should write");

    let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_chunks"),
        4
    );
    let chunks = read_chunks(&connection);
    assert_eq!(
        chunks,
        vec![
            (
                String::from("input"),
                0,
                String::from(
                    r#"prompt {"content":"Bearer [REDACTED]"} Bearer:[REDACTED] Bearer=[REDACTED]"#
                )
            ),
            (String::from("content"), 1, String::from("first")),
            (
                String::from("content"),
                2,
                String::from("Bearer [REDACTED]")
            ),
            (String::from("reasoning"), 3, String::from("second")),
        ]
    );
    let (
        request_metadata_json,
        raw_input,
        raw_output,
        raw_reasoning,
        raw_tool_calls,
    ): (
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = connection
        .query_row(
            "SELECT request_metadata_json, raw_input, raw_output, raw_reasoning, raw_tool_calls FROM evidence_attempts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .expect("raw attempt should exist");
    let persisted_text = format!(
        "{}{}{}{}{}",
        request_metadata_json,
        raw_input.unwrap_or_default(),
        raw_output.clone().unwrap_or_default(),
        raw_reasoning.unwrap_or_default(),
        raw_tool_calls.unwrap_or_default()
    );
    assert!(!persisted_text.contains("request-secret"));
    assert!(!persisted_text.contains("colon-secret"));
    assert!(!persisted_text.contains("equals-secret"));
    assert!(!persisted_text.contains("sk-live-secret"));
    assert!(!persisted_text.contains("out"));
    assert!(!persisted_text.contains("reasoning-secret"));
    assert!(!persisted_text.contains("tool-secret"));
    assert!(request_metadata_json.contains("request_header_authorization"));
    assert!(request_metadata_json.contains("[REDACTED]"));
    assert!(!request_metadata_json.contains("header-secret"));
    let raw_output = raw_output.expect("raw output should be retained");
    assert!(raw_output.contains("answer"));
    assert!(raw_output.contains("[REDACTED]"));
    assert!(raw_output.contains("tail"));
}

#[test]
fn retention_prunes_complete_oldest_groups_and_chunks() {
    let fixture = EvidenceFixture::new("retention");
    let manager = fixture.manager(true, true, false, 4, Some(2));
    let store = EvidenceStore::open(manager.clone());

    for index in 0..3 {
        let group_id = format!("group-retention-{index}");
        let mut attempt = attempt_record(
            &group_id,
            1,
            EvidenceAttemptRole::Primary,
            EvidenceAttemptStatus::Accepted,
            true,
        );
        attempt.raw_payloads = RawPayloads {
            input: Some(format!("payload-{index}")),
            output: None,
            reasoning: None,
            tool_calls: None,
            chunks: Vec::new(),
        };
        store
            .record_group(&group_record(&group_id, 1_000 + index), &[attempt])
            .expect("retention write should succeed");
    }

    let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_groups"),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_groups WHERE group_id = 'group-retention-2'",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_chunks WHERE attempt_id NOT IN (SELECT attempt_id FROM evidence_attempts)",
        ),
        0
    );
}

#[test]
fn retention_prunes_expired_raw_artifact_content_without_deleting_metadata() {
    let fixture = EvidenceFixture::new("raw-retention");
    let manager = fixture.manager_with_extra(
        true,
        false,
        false,
        100,
        None,
        "
[evidence.shadow.paired_comparison]
enabled = true
include_raw_input = true
include_raw_output = false
include_raw_reasoning = false
sample_rate = 1.0
max_retention_records = 100
max_retention_bytes = 1000000
retention_days = 1
",
    );
    let store = EvidenceStore::open(manager.clone());
    let mut first_attempt = paired_shadow_attempt("group-expired-raw-1", "max-thinking");
    first_attempt.raw_payloads.input = Some(String::from("expired raw input"));
    store
        .record_group(
            &group_record("group-expired-raw-1", 1_000),
            &[first_attempt],
        )
        .expect("first raw evidence should write");
    {
        let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
        connection
            .execute(
                "UPDATE evidence_raw_artifacts SET created_at_unix_ms = 0",
                [],
            )
            .expect("test should age raw artifacts");
    }

    let mut second_attempt = paired_shadow_attempt("group-expired-raw-2", "no-thinking");
    second_attempt.raw_payloads.input = Some(String::from("fresh raw input"));
    store
        .record_group(
            &group_record("group-expired-raw-2", 2_000),
            &[second_attempt],
        )
        .expect("second raw evidence should write and prune expired raw content");

    let connection = Connection::open(&fixture.sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_groups"),
        2
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_attempts"),
        2
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_raw_artifacts"),
        2
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_raw_artifacts WHERE content_text IS NOT NULL",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE group_id = 'group-expired-raw-1' AND raw_input IS NULL",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE group_id = 'group-expired-raw-2' AND raw_input IS NOT NULL",
        ),
        1
    );
}

fn read_chunks(connection: &Connection) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT channel, sequence_number, chunk_text \
             FROM evidence_chunks ORDER BY sequence_number",
        )
        .expect("chunk query should prepare");
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .expect("chunk query should execute")
        .map(|row| row.expect("chunk row should decode"))
        .collect()
}

fn group_record(group_id: &str, started_at_unix_ms: u64) -> EvidenceGroupRecord {
    EvidenceGroupRecord {
        group_id: group_id.to_owned(),
        request_id: RequestId::from_string(format!("req-{group_id}"))
            .expect("request id should build"),
        started_at_unix_ms,
        finished_at_unix_ms: Some(started_at_unix_ms + 100),
        model_id: Some(String::from("test-chat")),
        status: RequestStatus::Succeeded.as_str().to_owned(),
        request_metadata: BTreeMap::from([
            (String::from("path"), String::from("/v1/chat/completions")),
            (
                String::from("request_header_authorization"),
                String::from("Bearer header-secret"),
            ),
        ]),
        response_metadata: BTreeMap::from([(
            String::from("response_body_bytes"),
            String::from("5"),
        )]),
    }
}

fn attempt_record(
    group_id: &str,
    attempt_number: u32,
    role: EvidenceAttemptRole,
    status: EvidenceAttemptStatus,
    shown_to_downstream: bool,
) -> EvidenceAttemptRecord {
    EvidenceAttemptRecord {
        attempt_id: AttemptId::from_string(format!(
            "attempt-{group_id}-{attempt_number}-{}",
            role.as_str()
        ))
        .expect("attempt id should build"),
        group_id: group_id.to_owned(),
        request_id: RequestId::from_string(format!("req-{group_id}"))
            .expect("request id should build"),
        attempt_number,
        role,
        shown_to_downstream,
        started_at_unix_ms: 1_000 + u64::from(attempt_number),
        finished_at_unix_ms: Some(1_100 + u64::from(attempt_number)),
        upstream_profile: Some(String::from("default")),
        model_id: Some(String::from("test-chat")),
        thinking_mode: Some(String::from("force_thinking")),
        thinking_budget_tokens: Some(32_768),
        thinking_max_tokens: Some(50_000),
        detector_features: BTreeMap::from([
            (String::from("loop_detected"), String::from("true")),
            (String::from("loop_signal"), String::from("repeated_line")),
        ]),
        status,
        http_status: Some(200),
        error_reason: None,
        retry_reason: (status == EvidenceAttemptStatus::Rejected)
            .then(|| String::from("loop_detected")),
        abort_reason: (status == EvidenceAttemptStatus::Rejected)
            .then(|| String::from("loop_guard")),
        shadow_skip_reason: (status == EvidenceAttemptStatus::Skipped)
            .then_some(ShadowSkipReason::ContinuationUnavailable),
        request_metadata: BTreeMap::from([
            (
                String::from("request_header_authorization"),
                String::from("Bearer header-secret"),
            ),
            (
                String::from("attempt_thinking_budget_tokens"),
                String::from("32768"),
            ),
        ]),
        response_metadata: BTreeMap::from([(
            String::from("response_body_bytes"),
            String::from("42"),
        )]),
        raw_payloads: RawPayloads::default(),
    }
}

fn paired_shadow_attempt(group_id: &str, variant_name: &str) -> EvidenceAttemptRecord {
    let mut attempt = attempt_record(
        group_id,
        1,
        EvidenceAttemptRole::ShadowContinued,
        EvidenceAttemptStatus::Accepted,
        false,
    );
    attempt.attempt_id = AttemptId::from_string(format!("attempt-{group_id}-{variant_name}"))
        .expect("paired attempt id should build");
    attempt.request_metadata.insert(
        String::from("shadow_paired_comparison"),
        String::from("true"),
    );
    attempt
        .request_metadata
        .insert(String::from("variant_name"), variant_name.to_owned());
    attempt.request_metadata.insert(
        String::from("shadow_compare_attempt"),
        variant_name.to_owned(),
    );
    attempt
}

#[derive(Debug, Eq, PartialEq)]
struct RawArtifactRow {
    variant_name: String,
    artifact_kind: String,
    content_text: String,
    bytes_stored: i64,
    truncated: i64,
    sha256: String,
}

fn read_raw_artifacts(connection: &Connection) -> Vec<RawArtifactRow> {
    let mut statement = connection
        .prepare(
            "SELECT COALESCE(variant_name, ''), artifact_kind, COALESCE(content_text, ''), \
             bytes_stored, truncated, sha256 \
             FROM evidence_raw_artifacts ORDER BY variant_name, artifact_kind",
        )
        .expect("raw artifact query should prepare");
    statement
        .query_map([], |row| {
            Ok(RawArtifactRow {
                variant_name: row.get(0)?,
                artifact_kind: row.get(1)?,
                content_text: row.get(2)?,
                bytes_stored: row.get(3)?,
                truncated: row.get(4)?,
                sha256: row.get(5)?,
            })
        })
        .expect("raw artifact query should execute")
        .map(|row| row.expect("raw artifact row should decode"))
        .collect()
}

fn evidence_attempt_roles(connection: &Connection) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT role, shown_to_downstream, status FROM evidence_attempts ORDER BY attempt_number",
        )
        .expect("attempt query should prepare");
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .expect("attempt query should run")
        .map(|row| row.expect("attempt row should decode"))
        .collect()
}

fn count_rows(connection: &Connection, sql: &str) -> u64 {
    let count: i64 = connection
        .query_row(sql, [], |row| row.get(0))
        .expect("count query should succeed");
    u64::try_from(count).expect("count should be nonnegative")
}

struct EvidenceFixture {
    root: PathBuf,
    sqlite_path: PathBuf,
    blob_cache_dir: PathBuf,
}

impl EvidenceFixture {
    fn new(name: &str) -> Self {
        let root = unique_test_dir(name);
        fs::create_dir_all(root.join("storage")).expect("storage dir should be created");
        set_owner_only_dir(&root);
        set_owner_only_dir(&root.join("storage"));
        Self {
            sqlite_path: root.join("storage").join("evidence.sqlite3"),
            blob_cache_dir: root.join("cache").join("evidence").join("blobs"),
            root,
        }
    }

    fn manager(
        &self,
        evidence_enabled: bool,
        include_raw_payloads: bool,
        include_request_headers: bool,
        max_records: u64,
        prune_to_records: Option<u64>,
    ) -> ConfigHandle {
        self.manager_with_extra(
            evidence_enabled,
            include_raw_payloads,
            include_request_headers,
            max_records,
            prune_to_records,
            "",
        )
    }

    fn manager_with_extra(
        &self,
        evidence_enabled: bool,
        include_raw_payloads: bool,
        include_request_headers: bool,
        max_records: u64,
        prune_to_records: Option<u64>,
        extra_config: &str,
    ) -> ConfigHandle {
        let prune_to_records_entry = prune_to_records
            .map(|value| format!("prune_to_records = {value}\n"))
            .unwrap_or_default();
        let contents = format!(
            r#"
[evidence]
enabled = {evidence_enabled}
sqlite_path = "{sqlite_path}"
blob_cache_dir = "{blob_cache_dir}"
include_raw_payloads = {include_raw_payloads}
include_request_headers = {include_request_headers}
max_bytes = {TEST_MAX_BYTES}
prune_to_bytes = {TEST_PRUNE_TO_BYTES}
max_records = {max_records}
{prune_to_records_entry}
{extra_config}
"#,
            sqlite_path = self.sqlite_path.display(),
            blob_cache_dir = self.blob_cache_dir.display(),
            extra_config = extra_config,
        );
        let config = AppConfig::parse(&contents).expect("config should parse");
        config.validate().expect("config should validate");
        ConfigHandle::new(config)
    }
}

impl Drop for EvidenceFixture {
    fn drop(&mut self) {
        let _ignored = fs::remove_dir_all(&self.root);
    }
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("llm-guard-proxy-evidence-{name}-{nanos}"))
}

fn set_owner_only_dir(path: &Path) {
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .expect("test directory permissions should be restricted");
}
