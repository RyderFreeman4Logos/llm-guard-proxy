//! Content-free test failure receipts. Historical GF1/GF2 causes remain UNKNOWN.
use super::*;

#[derive(Clone)]
pub(super) struct ReceiptTrace(Arc<Mutex<Vec<(&'static str, u128)>>>);

impl ReceiptTrace {
    pub(super) fn install(store: &ObservabilityStore) -> Self {
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&events);
        let started = std::time::Instant::now();
        store.set_recovery_receipt_test_hook(move |stage, _id| {
            observed
                .lock()
                .expect("diagnostic trace")
                .push((stage, started.elapsed().as_micros()));
        });
        Self(events)
    }

    pub(super) fn snapshot(&self) -> Vec<(&'static str, u128)> {
        self.0.lock().expect("diagnostic trace").clone()
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

pub(super) async fn assert_ok(
    response: reqwest::Response,
    proxy: &ProxyFixture,
    trace: &ReceiptTrace,
) {
    let status = response.status();
    let body = response.bytes().await.expect("response should drain");
    if status == StatusCode::OK {
        return;
    }
    let body = serde_json::from_slice::<serde_json::Value>(&body).ok();
    let error = category(
        body.as_ref()
            .and_then(|b| b.get("error"))
            .and_then(|e| e.get("code"))
            .and_then(serde_json::Value::as_str),
    );
    let attempts: Vec<_> = read_attempt_chain_rows(&proxy.sqlite_path)
        .iter()
        .map(|row| {
            json!({
                "attempt": row.attempt_number,
                "status": category(Some(&row.status)),
                "http_status": row.http_status,
                "retry_reason": category(row.retry_reason.as_deref()),
                "abort_reason": category(row.abort_reason.as_deref()),
                "recovery": recovery_metadata(&row.response_metadata),
            })
        })
        .collect();
    assert_eq!(
        status,
        StatusCode::OK,
        "error_category={error} ordered_attempts={attempts:?} receipt_stages_us={:?}; historical_GF1_GF2=UNKNOWN",
        trace.snapshot()
    );
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
