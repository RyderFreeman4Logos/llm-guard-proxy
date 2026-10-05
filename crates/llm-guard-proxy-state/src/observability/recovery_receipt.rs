//! Content-free receipts for local recovery, independent of completed requests.
use std::time::Instant;

use rusqlite::params;
use serde::Serialize;

use super::{ObservabilityStore, redaction::sanitize_optional_text};

/// Allowlisted pre-action facts; no request content or command arguments.
#[derive(Clone, Debug, Serialize)]
pub struct LocalRecoveryReceipt {
    pub receipt_id: String,
    pub generated_at_unix_ms: u64,
    pub process_id: u32,
    pub episode: u64,
    pub detector: &'static str,
    pub cause: &'static str,
    pub profile: String,
    pub command_id: &'static str,
    pub command_generation: u64,
    pub request_id: Option<String>,
    pub attempt_id: Option<String>,
    pub restart_timeout_ms: u64,
    pub readiness_deadline_ms: u64,
    pub first_chunk_timeout_ms: u64,
    pub idle_timeout_ms: u64,
    pub request_deadline_ms: Option<u64>,
    pub detection_window_secs: u64,
    pub min_output_progress_units: u64,
}

impl ObservabilityStore {
    /// Commits a bounded receipt even when ordinary observability is disabled.
    ///
    /// # Errors
    /// Fails closed on lock contention, expired budget, unsafe labels, or SQLite failure.
    pub fn record_local_recovery_receipt(
        &self,
        receipt: &LocalRecoveryReceipt,
        deadline: Instant,
    ) -> Result<(), &'static str> {
        if !matches!(
            receipt.cause,
            "stuck_watchdog"
                | "upstream_stall"
                | "transient_status"
                | "transient_transport"
                | "request_deadline"
        ) || !matches!(receipt.detector, "stuck_watchdog" | "precommit_recovery")
            || receipt.command_id != "local_recovery.restart_command"
        {
            return Err("invalid_receipt");
        }
        let mut safe = receipt.clone();
        safe.profile = safe_label(&safe.profile);
        safe.request_id = safe.request_id.map(|id| safe_label(&id));
        safe.attempt_id = safe.attempt_id.map(|id| safe_label(&id));
        safe.receipt_id = safe_label(&safe.receipt_id);
        let json = serde_json::to_string(&safe).map_err(|_| "invalid_receipt")?;
        let mut connection = self.connection.try_lock().map_err(|_| "store_busy")?;
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("write_timeout")?;
        // A timer cannot preempt a synchronous SQLite busy wait. Bound that wait too.
        let previous: u64 = connection
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .map_err(|_| "write_failed")?;
        connection
            .busy_timeout(remaining)
            .map_err(|_| "write_failed")?;
        let result = (|| {
            let synchronous: u32 = connection
                .query_row("PRAGMA synchronous", [], |row| row.get(0))
                .map_err(|_| "write_failed")?;
            if synchronous < 2 {
                return Err("durability_disabled");
            }
            let transaction = connection.transaction().map_err(|_| "write_failed")?;
            transaction.execute(
                "INSERT INTO local_recovery_receipts(receipt_id, generated_at_unix_ms, receipt_json) VALUES (?1, ?2, ?3)",
                params![safe.receipt_id, safe.generated_at_unix_ms, json],
            ).map_err(|_| "write_failed")?;
            // ponytail: fixed 1024-row content-free ring; configurable retention only if needed.
            transaction.execute("DELETE FROM local_recovery_receipts WHERE rowid IN (SELECT rowid FROM local_recovery_receipts ORDER BY rowid DESC LIMIT -1 OFFSET 1024)", []).map_err(|_| "write_failed")?;
            transaction.commit().map_err(|_| "write_failed")
        })();
        connection
            .busy_timeout(std::time::Duration::from_millis(previous))
            .map_err(|_| "write_failed")?;
        result
    }

    /// Joins a terminal recovery outcome to its pre-action receipt.
    ///
    /// # Errors
    /// Returns a fixed category on contention or persistence failure. Missing completion
    /// leaves the receipt pending, never invents success after cancellation.
    pub fn finish_local_recovery_receipt(
        &self,
        receipt_id: &str,
        outcome: &str,
        restart_status: Option<&str>,
        readiness_status: Option<&str>,
        deadline: Instant,
    ) -> Result<(), &'static str> {
        let connection = self.connection.try_lock().map_err(|_| "store_busy")?;
        let safe_outcome = safe_label(outcome);
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("write_timeout")?;
        let previous: u64 = connection
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .map_err(|_| "write_failed")?;
        connection
            .busy_timeout(remaining)
            .map_err(|_| "write_failed")?;
        let result = connection
            .execute(
                "UPDATE local_recovery_receipts SET outcome = ?2, restart_status = ?3, readiness_status = ?4 WHERE receipt_id = ?1",
                params![receipt_id, safe_outcome, restart_status.map(safe_label), readiness_status.map(safe_label)],
            )
            .map_err(|_| "write_failed");
        connection
            .busy_timeout(std::time::Duration::from_millis(previous))
            .map_err(|_| "write_failed")?;
        match result {
            Ok(1) => Ok(()),
            Ok(_) => Err("receipt_missing"),
            Err(category) => Err(category),
        }
    }
}

fn safe_label(value: &str) -> String {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-.:".contains(&b))
        || value.contains("://")
        || sanitize_optional_text(Some(&value.to_owned())).as_deref() != Some(value)
    {
        return String::from("unknown");
    }
    value.to_owned()
}
