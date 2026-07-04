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
    EvidenceStore, EvidenceStoreWrite, ShadowSkipReason,
};
use crate::{AttemptId, ConfigManager, RawPayloadChunk, RawPayloads, RequestId, RequestStatus};

const TEST_MAX_BYTES: u64 = 1_000_000;
const TEST_PRUNE_TO_BYTES: u64 = 800_000;

#[test]
fn disabled_evidence_write_creates_no_artifacts() {
    let fixture = EvidenceFixture::new("disabled");
    let manager = fixture.manager(false, false, false, 100, None);
    let store = EvidenceStore::open(manager.handle());

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
    let store = EvidenceStore::open(manager.handle());
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
}

#[test]
fn raw_evidence_capture_redacts_headers_and_secret_fragments() {
    let fixture = EvidenceFixture::new("raw-redaction");
    let manager = fixture.manager(true, true, true, 100, None);
    let store = EvidenceStore::open(manager.handle());
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
    let store = EvidenceStore::open(manager.handle());

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
    config_path: PathBuf,
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
            config_path: root.join("config.toml"),
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
    ) -> ConfigManager {
        let prune_to_records_entry = prune_to_records
            .map(|value| format!("prune_to_records = {value}\n"))
            .unwrap_or_default();
        fs::write(
            &self.config_path,
            format!(
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
"#,
                sqlite_path = self.sqlite_path.display(),
                blob_cache_dir = self.blob_cache_dir.display(),
            ),
        )
        .expect("config should be written");
        ConfigManager::from_explicit_path(&self.config_path).expect("config should load")
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
