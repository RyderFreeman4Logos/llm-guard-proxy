//! Content-free test failure receipts. Historical GF1/GF2 causes remain UNKNOWN.
use super::*;

#[derive(Clone)]
pub(super) struct ReceiptTrace(Arc<Mutex<TraceData>>);

struct TraceData {
    events: Vec<(&'static str, u128, usize)>,
    ids: Vec<String>,
    overflow: bool,
}

impl Default for TraceData {
    fn default() -> Self {
        Self {
            events: Vec::with_capacity(512),
            ids: Vec::with_capacity(16),
            overflow: false,
        }
    }
}

impl ReceiptTrace {
    pub(super) fn install(store: &ObservabilityStore) -> Self {
        let trace = Self(Arc::new(Mutex::new(TraceData::default())));
        let observed = trace.clone();
        let started = std::time::Instant::now();
        store.set_recovery_receipt_test_hook(move |stage, id| {
            let time = started.elapsed().as_micros();
            observed.record(stage, id, time);
        });
        trace
    }

    fn record(&self, stage: &'static str, id: &str, time: u128) {
        let mut trace = self.0.lock().expect("diagnostic trace");
        if trace.events.len() == 512
            || (!trace.ids.iter().any(|old| old == id) && trace.ids.len() == 16)
        {
            trace.overflow = true;
            return;
        }
        let ordinal = trace
            .ids
            .iter()
            .position(|old| old == id)
            .unwrap_or_else(|| {
                trace.ids.push(id.to_owned());
                trace.ids.len() - 1
            });
        trace.events.push((stage, time, ordinal));
    }

    pub(super) fn snapshot(&self) -> Vec<(&'static str, u128)> {
        self.0
            .lock()
            .expect("diagnostic trace")
            .events
            .iter()
            .map(|(stage, time, _)| (*stage, *time))
            .collect()
    }

    fn summary(&self) -> serde_json::Value {
        let trace = self.0.lock().expect("diagnostic trace");
        let events: Vec<_> = trace
            .events
            .iter()
            .map(|(stage, time, receipt)| {
                let stage = if *stage == "record_ack" {
                    "write_result_observed"
                } else {
                    stage
                };
                json!({"stage":stage, "us":time, "receipt":receipt})
            })
            .collect();
        json!({"events":events, "overflow":trace.overflow, "kernel_cause":"UNKNOWN"})
    }

    fn report(&self, http_status: u16) {
        use std::io::Write;
        // Outside SQLite and its timed worker; expose success too under normal libtest capture.
        let _ = writeln!(
            std::io::stderr(),
            "receipt_phase_measurement http_status={http_status} {}",
            self.summary()
        );
    }
}

// Exact categories only: a label-shaped secret is still not a category.
fn category(value: Option<&str>) -> &str {
    match value {
        Some(
            value @ ("succeeded"
            | "failed"
            | "retried"
            | "aborted"
            | "cancelled"
            | "receipt_failed"
            | "ready"
            | "timeout"
            | "timeout_killed"
            | "write_timeout"
            | "write_failed"
            | "writer_busy"
            | "writer_failed"
            | "store_busy"
            | "durability_disabled"
            | "invalid_receipt"
            | "transient_status"
            | "transient_transport"
            | "request_deadline"
            | "upstream_stall"
            | "stuck_watchdog"
            | "transient_upstream_status"
            | "transient_upstream_transport"
            | "upstream_connect_failed"
            | "upstream_timeout"
            | "upstream_body_error"
            | "upstream_status_error"
            | "upstream_transport_error"
            | "upstream_stream_error"
            | "request_deadline_exceeded"
            | "recovery_failed"
            | "cooldown"
            | "budget_exhausted"
            | "disabled"
            | "shutdown_cancelled"),
        ) => value,
        _ => "UNKNOWN",
    }
}

pub(super) fn recovery_metadata(value: &serde_json::Value) -> serde_json::Value {
    let mut safe = serde_json::Map::new();
    for key in [
        "local_recovery_status",
        "local_recovery_restart_status",
        "local_recovery_readiness_status",
        "local_recovery_receipt_error",
        "local_recovery_cause",
    ] {
        safe.insert(
            key.to_owned(),
            json!(category(value.get(key).and_then(serde_json::Value::as_str))),
        );
    }
    safe.insert(
        "local_recovery_request_attempts_used".to_owned(),
        json!(
            value
                .get("local_recovery_request_attempts_used")
                .and_then(serde_json::Value::as_str)
                .and_then(|v| v.parse::<u64>().ok())
        ),
    );
    serde_json::Value::Object(safe)
}

// SQLite interrupt bounds executing SQL (including hostile views), not just an async wait.
// Read-only, zero lock wait, <=32 ordinal rows, <=16KiB per metadata value.
fn bounded_attempts(path: &Path) -> Option<Vec<serde_json::Value>> {
    use std::sync::mpsc::{RecvTimeoutError, channel};
    let connection =
        Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
    connection.busy_timeout(Duration::ZERO).ok()?;
    let deadline = std::time::Instant::now() + Duration::from_millis(100);
    let interrupt = connection.get_interrupt_handle();
    std::thread::scope(|scope| {
        let (done, receiver) = channel::<()>();
        let watchdog = std::thread::Builder::new()
            .spawn_scoped(scope, move || {
                let mut wait = deadline.saturating_duration_since(std::time::Instant::now());
                loop {
                    match receiver.recv_timeout(wait) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => interrupt.interrupt(),
                    }
                    // An interrupt between prepare/step calls is a no-op; keep it armed
                    // until the query owner signals completion, then join before returning.
                    wait = Duration::from_millis(1);
                }
            })
            .ok()?;
        let rows = (|| -> rusqlite::Result<Vec<serde_json::Value>> {
            let mut statement = connection.prepare(
                "SELECT attempt_number, substr(status,1,64), http_status, substr(retry_reason,1,64), substr(abort_reason,1,64), CAST(substr(CAST(response_metadata_json AS BLOB),1,16384) AS TEXT) FROM attempts ORDER BY rowid LIMIT 32",
            )?;
            if std::time::Instant::now() >= deadline {
                return Err(rusqlite::Error::InvalidQuery);
            }
            statement
                .query_map([], |row| {
                    let metadata: String = row.get(5)?;
                    let metadata =
                        serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null);
                    let status: String = row.get(1)?;
                    let retry: Option<String> = row.get(3)?;
                    let abort: Option<String> = row.get(4)?;
                    Ok(json!({
                        "attempt": row.get::<_, u32>(0)?,
                        "status": category(Some(&status)),
                        "http_status": row.get::<_, Option<u16>>(2)?,
                        "retry_reason": category(retry.as_deref()),
                        "abort_reason": category(abort.as_deref()),
                        "recovery": recovery_metadata(&metadata),
                    }))
                })?
                .collect()
        })();
        drop(done);
        watchdog.join().ok()?;
        rows.ok()
    })
}

pub(super) async fn assert_ok(
    response: reqwest::Response,
    proxy: &ProxyFixture,
    trace: &ReceiptTrace,
) {
    let status = response.status();
    if status == StatusCode::OK {
        response.bytes().await.expect("response should drain");
        trace.report(status.as_u16());
        return;
    }
    let body = timeout(
        Duration::from_millis(100),
        read_upstream_body_bytes_limited(response.bytes_stream(), 16 * 1024),
    )
    .await
    .ok()
    .and_then(Result::ok)
    .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let error = category(
        body.as_ref()
            .and_then(|b| b.get("error"))
            .and_then(|e| e.get("code"))
            .and_then(serde_json::Value::as_str),
    );
    let attempts = bounded_attempts(&proxy.sqlite_path);
    let storage = if attempts.is_some() {
        "AVAILABLE"
    } else {
        "UNKNOWN"
    };
    let attempts = attempts.unwrap_or_default();
    // Observe late real completion after HTTP failure, never extend admission or retry.
    proxy
        .state
        .persistence_tasks
        .flush(Duration::from_secs(1))
        .await;
    trace.report(status.as_u16());
    assert_eq!(
        status,
        StatusCode::OK,
        "error_category={error} attempt_storage={storage} ordered_attempts={attempts:?} receipt_stages_us={:?}; historical_GF1_GF2=UNKNOWN",
        trace.snapshot()
    );
}

// Real HTTP responses at the external boundary; the shared assertion remains the SUT.
async fn failure_response(body: &'static str) -> (reqwest::Response, TestServer) {
    use tokio::io::AsyncWriteExt;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("fixture bind");
    let addr = listener.local_addr().expect("fixture address");
    let server = TestServer::new(tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("fixture accept");
        let payload = match body {
            "oversize" => "private-prompt".repeat(8192),
            "malformed" => "private-prompt".to_owned(),
            _ => r#"{"error":{"code":"upstream_timeout","message":"private-prompt"}}"#.to_owned(),
        };
        let length = if body == "stall" || body == "truncated" {
            payload.len() + 1
        } else {
            payload.len()
        };
        socket.write_all(format!("HTTP/1.1 502 Bad Gateway\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{payload}").as_bytes()).await.expect("fixture response");
        if body == "stall" {
            std::future::pending::<()>().await;
        }
    }));
    let response = Client::new()
        .get(format!("http://{addr}"))
        .send()
        .await
        .expect("fixture headers");
    (response, server)
}

async fn original_failure_survives(proxy: &ProxyFixture, body: &'static str) -> String {
    use futures_util::FutureExt;
    let (response, _server) = failure_response(body).await;
    let trace = ReceiptTrace::install(&proxy.store);
    let started = std::time::Instant::now();
    let assertion = std::panic::AssertUnwindSafe(assert_ok(response, proxy, &trace)).catch_unwind();
    let result = timeout(Duration::from_secs(2), assertion)
        .await
        .expect("original HTTP assertion must not hang");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "blocking diagnostics must respect the deadline too"
    );
    let panic = result.expect_err("non-OK response must still fail");
    let text = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("assertion text");
    for required in [
        "502",
        "200",
        "error_category=",
        "attempt_storage=",
        "ordered_attempts=",
        "receipt_stages_us=",
        "historical_GF1_GF2=UNKNOWN",
    ] {
        assert!(
            text.contains(required),
            "original HTTP assertion and complete diagnostics must survive"
        );
    }
    for secret in ["private-prompt", "private-password", "SELECT", "secrets"] {
        assert!(
            !text.contains(secret),
            "complete failure diagnostic must be content-free"
        );
    }
    text.to_owned()
}

#[tokio::test]
async fn failure_body_diagnostics_cannot_mask_http_status() {
    for body in ["stall", "truncated", "malformed", "oversize"] {
        let proxy = ProxyFixture::spawn("http://127.0.0.1:1", false).await;
        let text = original_failure_survives(&proxy, body).await;
        assert!(
            text.contains("error_category=UNKNOWN"),
            "incomplete or oversized body must be unavailable"
        );
    }
}

#[tokio::test]
async fn failure_database_diagnostics_cannot_mask_http_status() {
    for storage in ["missing", "schema", "locked", "malformed", "expensive"] {
        let mut proxy = ProxyFixture::spawn("http://127.0.0.1:1", false).await;
        proxy.sqlite_path = proxy.root.join("diagnostic.sqlite3");
        let connection = if storage == "missing" {
            None
        } else {
            Some(Connection::open(&proxy.sqlite_path).expect("fixture database"))
        };
        if let Some(connection) = &connection {
            if storage == "locked" {
                connection
                    .execute_batch("BEGIN EXCLUSIVE")
                    .expect("fixture lock");
            }
            if storage == "malformed" {
                connection.execute_batch("CREATE TABLE attempts(attempt_number, status, http_status, retry_reason, abort_reason, response_metadata_json); INSERT INTO attempts VALUES(2, 'private-password', 502, NULL, NULL, 'private-prompt'), (1, 'failed', 429, 'transient_status', NULL, '{}');").expect("fixture malformed row");
            }
            if storage == "expensive" {
                connection.execute_batch("CREATE VIEW attempts AS WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT 0 AS rowid, sum(x) AS attempt_number, 'failed' AS status, 502 AS http_status, NULL AS retry_reason, NULL AS abort_reason, '{}' AS response_metadata_json FROM n;").expect("fixture expensive query");
            }
        }
        let started = std::time::Instant::now();
        let text = original_failure_survives(&proxy, "valid").await;
        assert!(
            text.contains("error_category=upstream_timeout"),
            "valid error category must survive DB failure"
        );
        if storage == "expensive" {
            assert!(
                started.elapsed() >= Duration::from_millis(90),
                "fixture must execute SQL until interrupted, not fail at prepare"
            );
        }
        if storage == "malformed" {
            assert!(
                text.contains("attempt_storage=AVAILABLE"),
                "malformed metadata must retain ordinal rows"
            );
            let second = text.find("Number(2)").expect("first inserted ordinal");
            let first = text.find("Number(1)").expect("second inserted ordinal");
            assert!(second < first, "attempts must retain rowid order");
        } else {
            assert!(
                text.contains("attempt_storage=UNKNOWN"),
                "failed query must be unavailable"
            );
        }
        if storage == "missing" {
            assert!(
                !proxy.sqlite_path.exists(),
                "diagnostics must never create a database"
            );
        }
    }
}

#[test]
fn receipt_trace_is_bounded_and_correlates_ordinals() {
    let trace = ReceiptTrace(Arc::new(Mutex::new(TraceData::default())));
    trace.record("commit_start", "private-first", 1);
    trace.record("commit_end", "private-first", 3);
    trace.record("commit_ok", "private-first", 4);
    trace.record("record_ack", "private-second", 5);
    for time in 0..512 {
        trace.record("record_worker_start", "private-second", time);
    }
    let summary = trace.summary();
    assert_eq!(summary["overflow"], true);
    assert_eq!(summary["events"].as_array().expect("events").len(), 512);
    assert_eq!(summary["events"][0]["receipt"], 0);
    assert_eq!(summary["events"][3]["receipt"], 1);
    assert_eq!(summary["events"][3]["stage"], "write_result_observed");
    assert!(!summary.to_string().contains("private-"));
    assert_eq!(summary["kernel_cause"], "UNKNOWN");
}

#[test]
fn failure_categories_are_content_free() {
    let value = json!({"local_recovery_receipt_error":"write_timeout", "local_recovery_status":"receipt_failed", "local_recovery_cause":"transient_transport", "local_recovery_request_attempts_used":"2", "raw_prompt":"private-prompt", "local_recovery_profile":"private-password"});
    let safe = recovery_metadata(&value);
    assert_eq!(safe["local_recovery_receipt_error"], "write_timeout");
    assert_eq!(safe["local_recovery_request_attempts_used"], 2);
    assert_eq!(category(Some("upstream_timeout")), "upstream_timeout");
    for secret in [
        "private-prompt",
        "private-password",
        "write_timeout-secret",
        "Bearer secret",
        "https://user:password@host",
        "SELECT private FROM secrets",
    ] {
        assert_eq!(category(Some(secret)), "UNKNOWN");
        assert!(!safe.to_string().contains(secret));
    }
}
