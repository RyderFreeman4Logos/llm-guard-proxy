use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::params;

use super::{
    AttemptId, AttemptRecord, AttemptStatus, DownstreamMode, ObservabilityStore, RawPayloads,
    RequestId, RequestRecord, RequestStatus, StoreWrite, UpstreamMode,
};
use crate::ConfigManager;

#[test]
fn creates_sqlite_schema_in_test_temp_directory() {
    let fixture = StoreFixture::new("schema");
    let store = fixture.open_store(true, false, 10_000, 8_000);

    assert_eq!(store.schema_version().expect("schema version"), 1);
    assert!(fixture.sqlite_path.exists());

    let connection = store.lock_connection().expect("connection lock");
    let request_table_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'requests'",
            [],
            |row| row.get(0),
        )
        .expect("requests table should exist");
    let attempt_table_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'attempts'",
            [],
            |row| row.get(0),
        )
        .expect("attempts table should exist");

    assert_eq!(request_table_count, 1);
    assert_eq!(attempt_table_count, 1);
}

#[test]
fn writes_success_and_failure_request_and_attempt_rows() {
    let fixture = StoreFixture::new("success-failure");
    let store = fixture.open_store(true, false, 10_000, 8_000);

    let success = request_record("req-success", RequestStatus::Succeeded, 1_000);
    let success_attempt = attempt_record(
        "attempt-success",
        &success.request_id,
        AttemptStatus::Succeeded,
        1,
        1_010,
    );
    let failure = RequestRecord {
        status: RequestStatus::Failed,
        http_status: Some(502),
        error_reason: Some(String::from("upstream timeout")),
        ..request_record("req-failure", RequestStatus::Failed, 2_000)
    };
    let failure_attempt = AttemptRecord {
        error_reason: Some(String::from("connection reset")),
        retry_reason: Some(String::from("transport error")),
        ..attempt_record(
            "attempt-failure",
            &failure.request_id,
            AttemptStatus::Failed,
            1,
            2_010,
        )
    };

    assert_eq!(
        store
            .record_request(&success)
            .expect("success request write"),
        StoreWrite::Written
    );
    assert_eq!(
        store
            .record_attempt(&success_attempt)
            .expect("success attempt write"),
        StoreWrite::Written
    );
    assert_eq!(
        store
            .record_request(&failure)
            .expect("failure request write"),
        StoreWrite::Written
    );
    assert_eq!(
        store
            .record_attempt(&failure_attempt)
            .expect("failure attempt write"),
        StoreWrite::Written
    );

    let connection = store.lock_connection().expect("connection lock");
    let succeeded_requests = count_rows(
        &connection,
        "SELECT COUNT(*) FROM requests WHERE status = 'succeeded'",
    );
    let failed_requests = count_rows(
        &connection,
        "SELECT COUNT(*) FROM requests WHERE status = 'failed'",
    );
    let succeeded_attempts = count_rows(
        &connection,
        "SELECT COUNT(*) FROM attempts WHERE status = 'succeeded'",
    );
    let failed_attempts = count_rows(
        &connection,
        "SELECT COUNT(*) FROM attempts WHERE status = 'failed'",
    );

    assert_eq!(succeeded_requests, 1);
    assert_eq!(failed_requests, 1);
    assert_eq!(succeeded_attempts, 1);
    assert_eq!(failed_attempts, 1);
}

#[test]
fn updating_request_preserves_existing_attempt_rows() {
    let fixture = StoreFixture::new("request-update-preserves-attempts");
    let store = fixture.open_store(true, false, 10_000, 8_000);
    let initial_request = RequestRecord {
        finished_at_unix_ms: None,
        status: RequestStatus::Failed,
        http_status: Some(500),
        error_reason: Some(String::from("in flight")),
        ..request_record("req-update", RequestStatus::Failed, 1_000)
    };
    let attempt = attempt_record(
        "attempt-update",
        &initial_request.request_id,
        AttemptStatus::Succeeded,
        1,
        1_010,
    );

    store
        .record_request(&initial_request)
        .expect("initial request write");
    store.record_attempt(&attempt).expect("attempt write");

    let final_request = RequestRecord {
        finished_at_unix_ms: Some(1_300),
        status: RequestStatus::Succeeded,
        http_status: Some(200),
        error_reason: None,
        response_metadata: BTreeMap::from([(String::from("server"), String::from("updated"))]),
        ..initial_request
    };
    store
        .record_request(&final_request)
        .expect("final request update");

    let connection = store.lock_connection().expect("connection lock");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM attempts WHERE request_id = 'req-update' AND attempt_number = 1",
        ),
        1
    );

    let (status, finished_at_unix_ms, error_reason, response_metadata_json): (
        String,
        i64,
        Option<String>,
        String,
    ) = connection
        .query_row(
            r"
SELECT status, finished_at_unix_ms, error_reason, response_metadata_json
FROM requests
WHERE request_id = 'req-update'
",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("updated request row should exist");
    assert_eq!(status, "succeeded");
    assert_eq!(finished_at_unix_ms, 1_300);
    assert_eq!(error_reason, None);
    assert!(response_metadata_json.contains("updated"));
}

#[test]
fn redacts_authorization_and_api_key_like_values_before_persistence() {
    let fixture = StoreFixture::new("redaction");
    let store = fixture.open_store(true, true, 10_000, 8_000);
    let mut request = request_record("req-redaction", RequestStatus::Succeeded, 1_000);
    request.request_metadata = BTreeMap::from([
        (
            String::from("authorization"),
            String::from("Bearer sk-fixture-secret"),
        ),
        (
            String::from("x-api-key"),
            String::from("fixture-api-key-value"),
        ),
        (
            String::from("content-type"),
            String::from("application/json"),
        ),
    ]);
    request.raw_payloads = RawPayloads {
        input: Some(String::from(r#"{"api_key":"sk-fixture-secret"}"#)),
        output: Some(String::from("model output without credentials")),
        reasoning: Some(String::from("reasoning with token=fixture-secret")),
        tool_calls: None,
    };

    store
        .record_request(&request)
        .expect("redacted request write");

    let connection = store.lock_connection().expect("connection lock");
    let (metadata_json, raw_input, raw_output, raw_reasoning): (
        String,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = connection
        .query_row(
            r"
SELECT request_metadata_json, raw_input, raw_output, raw_reasoning
FROM requests
WHERE request_id = 'req-redaction'
",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("redacted row should exist");

    assert!(!metadata_json.contains("sk-fixture-secret"));
    assert!(!metadata_json.contains("fixture-api-key-value"));
    assert!(metadata_json.contains("[REDACTED]"));
    assert_eq!(raw_input.as_deref(), Some("[REDACTED]"));
    assert_eq!(
        raw_output.as_deref(),
        Some("model output without credentials")
    );
    assert_eq!(raw_reasoning.as_deref(), Some("[REDACTED]"));
}

#[test]
fn redacts_common_raw_payload_secret_forms_before_persistence() {
    let fixture = StoreFixture::new("raw-secret-forms");
    let store = fixture.open_store(true, true, 10_000, 8_000);
    let mut request = request_record("req-raw-secrets", RequestStatus::Succeeded, 1_000);
    request.raw_payloads = RawPayloads {
        input: Some(String::from(r#"{"password":"fixture-password"}"#)),
        output: Some(String::from("credential = fixture-credential")),
        reasoning: Some(String::from("secret: fixture-secret")),
        tool_calls: Some(String::from(r#"{"arguments":{"passwd":"fixture-passwd"}}"#)),
    };
    let mut attempt = attempt_record(
        "attempt-raw-secrets",
        &request.request_id,
        AttemptStatus::Succeeded,
        1,
        1_010,
    );
    attempt.raw_payloads = RawPayloads {
        input: Some(String::from("authorization: Bearer fixture-bearer")),
        output: Some(String::from("token=fixture-token")),
        reasoning: Some(String::from(r#"{"api_key":"fixture-api-key"}"#)),
        tool_calls: Some(String::from("credential: fixture-tool-credential")),
    };

    store
        .record_request(&request)
        .expect("redacted raw request write");
    store
        .record_attempt(&attempt)
        .expect("redacted raw attempt write");

    let connection = store.lock_connection().expect("connection lock");
    assert_redacted_raw_payloads(
        &connection,
        "SELECT raw_input, raw_output, raw_reasoning, raw_tool_calls FROM requests WHERE request_id = 'req-raw-secrets'",
    );
    assert_redacted_raw_payloads(
        &connection,
        "SELECT raw_input, raw_output, raw_reasoning, raw_tool_calls FROM attempts WHERE attempt_id = 'attempt-raw-secrets'",
    );
}

#[test]
fn retention_deletes_oldest_requests_until_under_prune_target() {
    let fixture = StoreFixture::new("retention");
    let store = fixture.open_store(true, true, 1_000, 800);

    for index in 0..3 {
        let mut request = request_record(
            &format!("req-retention-{index}"),
            RequestStatus::Succeeded,
            1_000 + index,
        );
        request.raw_payloads = RawPayloads {
            input: Some("x".repeat(550)),
            output: None,
            reasoning: None,
            tool_calls: None,
        };
        store
            .record_request(&request)
            .expect("retention request write");
    }

    let usage = store.retention_usage().expect("retention usage");
    assert!(usage.observed_bytes <= 800);

    let connection = store.lock_connection().expect("connection lock");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM requests WHERE request_id = 'req-retention-0'",
        ),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM requests WHERE request_id = 'req-retention-2'",
        ),
        1
    );
}

#[test]
fn hot_reload_disabled_setting_stops_new_writes() {
    let fixture = StoreFixture::new("hot-reload-disable");
    let manager = fixture.manager(true, false, 10_000, 8_000);
    let store = ObservabilityStore::open(manager.handle()).expect("store should open");

    let first = request_record("req-before-disable", RequestStatus::Succeeded, 1_000);
    assert_eq!(
        store.record_request(&first).expect("first request write"),
        StoreWrite::Written
    );

    fixture.write_config(false, false, 10_000, 8_000);
    let outcome = manager.reload().expect("reload should succeed");
    assert!(outcome.applied);

    let second = request_record("req-after-disable", RequestStatus::Succeeded, 2_000);
    assert_eq!(
        store.record_request(&second).expect("disabled write"),
        StoreWrite::Disabled
    );

    let connection = store.lock_connection().expect("connection lock");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM requests"), 1);
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM requests WHERE request_id = 'req-after-disable'",
        ),
        0
    );
}

struct StoreFixture {
    root: PathBuf,
    config_path: PathBuf,
    sqlite_path: PathBuf,
}

impl StoreFixture {
    fn new(name: &str) -> Self {
        let root = unique_test_dir(name);
        fs::create_dir_all(&root).expect("test root should be created");
        Self {
            config_path: root.join("config.toml"),
            sqlite_path: root.join("observability.sqlite3"),
            root,
        }
    }

    fn open_store(
        &self,
        enabled: bool,
        capture_raw_payloads: bool,
        max_bytes: u64,
        prune_to_bytes: u64,
    ) -> ObservabilityStore {
        let manager = self.manager(enabled, capture_raw_payloads, max_bytes, prune_to_bytes);
        ObservabilityStore::open(manager.handle()).expect("store should open")
    }

    fn manager(
        &self,
        enabled: bool,
        capture_raw_payloads: bool,
        max_bytes: u64,
        prune_to_bytes: u64,
    ) -> ConfigManager {
        self.write_config(enabled, capture_raw_payloads, max_bytes, prune_to_bytes);
        ConfigManager::from_explicit_path(&self.config_path).expect("config should load")
    }

    fn write_config(
        &self,
        enabled: bool,
        capture_raw_payloads: bool,
        max_bytes: u64,
        prune_to_bytes: u64,
    ) {
        let sqlite_path = self.sqlite_path.display();
        fs::write(
            &self.config_path,
            format!(
                r#"
[observability]
enabled = {enabled}
sqlite_path = "{sqlite_path}"
capture_raw_payloads = {capture_raw_payloads}

[observability.retention]
max_bytes = {max_bytes}
prune_to_bytes = {prune_to_bytes}
max_records = 100
"#
            ),
        )
        .expect("test config should be written");
    }
}

impl Drop for StoreFixture {
    fn drop(&mut self) {
        remove_dir_all(&self.root);
    }
}

fn request_record(id: &str, status: RequestStatus, started_at_unix_ms: u64) -> RequestRecord {
    RequestRecord {
        request_id: RequestId::from_string(id).expect("test request id should be valid"),
        started_at_unix_ms,
        finished_at_unix_ms: Some(started_at_unix_ms + 100),
        downstream_mode: DownstreamMode::Streaming,
        upstream_mode: UpstreamMode::Streaming,
        model_id: Some(String::from("aeon-ultimate")),
        input_fingerprint: Some(format!("fingerprint-{id}")),
        status,
        http_status: Some(200),
        error_reason: None,
        abort_reason: None,
        request_metadata: BTreeMap::from([(
            String::from("content-type"),
            String::from("application/json"),
        )]),
        response_metadata: BTreeMap::from([(String::from("server"), String::from("vllm"))]),
        raw_payloads: RawPayloads::default(),
    }
}

fn attempt_record(
    id: &str,
    request_id: &RequestId,
    status: AttemptStatus,
    attempt_number: u32,
    started_at_unix_ms: u64,
) -> AttemptRecord {
    AttemptRecord {
        attempt_id: AttemptId::from_string(id).expect("test attempt id should be valid"),
        request_id: request_id.clone(),
        attempt_number,
        started_at_unix_ms,
        finished_at_unix_ms: Some(started_at_unix_ms + 80),
        upstream_mode: UpstreamMode::Streaming,
        status,
        http_status: Some(200),
        error_reason: None,
        retry_reason: None,
        abort_reason: None,
        request_metadata: BTreeMap::from([(
            String::from("accept"),
            String::from("text/event-stream"),
        )]),
        response_metadata: BTreeMap::from([(String::from("server"), String::from("vllm"))]),
        raw_payloads: RawPayloads::default(),
    }
}

fn count_rows(connection: &rusqlite::Connection, sql: &str) -> i64 {
    connection
        .query_row(sql, params![], |row| row.get(0))
        .expect("count query should succeed")
}

fn assert_redacted_raw_payloads(connection: &rusqlite::Connection, sql: &str) {
    let raw_payloads: (
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = connection
        .query_row(sql, [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .expect("raw payload row should exist");

    assert_eq!(raw_payloads.0.as_deref(), Some("[REDACTED]"));
    assert_eq!(raw_payloads.1.as_deref(), Some("[REDACTED]"));
    assert_eq!(raw_payloads.2.as_deref(), Some("[REDACTED]"));
    assert_eq!(raw_payloads.3.as_deref(), Some("[REDACTED]"));
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("llm-guard-proxy-{nanos}-{name}"))
}

fn remove_dir_all(path: &Path) {
    if let Err(error) = fs::remove_dir_all(path) {
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}
