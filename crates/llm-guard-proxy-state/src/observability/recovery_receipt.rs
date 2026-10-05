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
    /// Guardian-only ownership facts; absent for request/watchdog recovery.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardian: Option<GuardianRecoveryIdentity>,
}

/// Fixed-schema guardian facts, independent of prompts, argv and environment.
#[derive(Clone, Debug, Serialize)]
pub struct GuardianRecoveryIdentity {
    pub boot_id: String,
    pub guardian_episode: u64,
    pub config_revision: u64,
    pub target_device: u64,
    pub target_inode: u64,
    pub container_id: String,
    pub available_bytes: u64,
    pub threshold_bytes: u64,
    pub grace_secs: u64,
    pub binary_device: u64,
    pub binary_inode: u64,
}

#[cfg(feature = "recovery-receipt-test-hooks")]
type Observer = dyn Fn(&'static str, &str) + Send + Sync;
#[cfg(feature = "recovery-receipt-test-hooks")]
#[derive(Clone)]
pub(super) struct TestHook(std::sync::Arc<Observer>);

#[cfg(feature = "recovery-receipt-test-hooks")]
impl std::fmt::Debug for TestHook {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RecoveryReceiptTestHook")
    }
}

// Keep fixture instrumentation outside the admission function's production logic.
macro_rules! receipt_sql {
    ($store:expr, $id:expr, $work:expr) => {{
        #[cfg(feature = "recovery-receipt-test-hooks")]
        $store.observe_recovery_receipt_test("record_sql_start", $id);
        let result = $work;
        #[cfg(feature = "recovery-receipt-test-hooks")]
        $store.observe_recovery_receipt_test("record_sql_end", $id);
        result
    }};
}

impl ObservabilityStore {
    /// Installs a fixture-local observer; never affects another store instance.
    ///
    /// # Panics
    /// Panics if the fixture hook mutex is poisoned.
    #[cfg(feature = "recovery-receipt-test-hooks")]
    pub fn set_recovery_receipt_test_hook(
        &self,
        hook: impl Fn(&'static str, &str) + Send + Sync + 'static,
    ) {
        *self
            .recovery_receipt_test_hook
            .lock()
            .expect("fixture hook lock") = Some(TestHook(std::sync::Arc::new(hook)));
    }

    /// Observes an actual writer boundary without replacing its result.
    ///
    /// # Panics
    /// Panics if the fixture hook mutex is poisoned or the observer panics.
    #[cfg(feature = "recovery-receipt-test-hooks")]
    pub fn observe_recovery_receipt_test(&self, stage: &'static str, id: &str) {
        let hook = self
            .recovery_receipt_test_hook
            .lock()
            .expect("fixture hook lock")
            .clone();
        if let Some(hook) = hook {
            (hook.0)(stage, id);
        }
    }
    /// Commits a bounded receipt even when ordinary observability is disabled.
    ///
    /// Queues behind tracked terminal audit writers. The caller must retain worker
    /// ownership and require a durable acknowledgment within its original deadline.
    ///
    /// # Errors
    /// Fails closed on lock contention, expired budget, unsafe labels, or SQLite failure.
    pub fn record_local_recovery_receipt(
        &self,
        receipt: &LocalRecoveryReceipt,
        deadline: Instant,
    ) -> Result<(), &'static str> {
        let guardian = receipt.guardian.as_ref();
        if let Some(identity) = guardian {
            let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                .map_err(|_| "boot_unavailable")?;
            if identity.boot_id != boot.trim() {
                return Err("invalid_boot");
            }
        }
        let guardian_labels = guardian.is_some()
            && receipt.cause == "memory_pressure"
            && receipt.detector == "memory_guardian"
            && receipt.command_id == "guardian.foreground_recovery"
            && receipt.request_id.is_none()
            && receipt.attempt_id.is_none();
        if guardian.is_some_and(|identity| {
            identity.container_id.len() != 64
                || !identity.container_id.bytes().all(|b| b.is_ascii_hexdigit())
        }) {
            return Err("invalid_receipt");
        }
        if !guardian_labels
            && (guardian.is_some()
                || !matches!(
                    receipt.cause,
                    "stuck_watchdog"
                        | "upstream_stall"
                        | "transient_status"
                        | "transient_transport"
                        | "request_deadline"
                )
                || !matches!(receipt.detector, "stuck_watchdog" | "precommit_recovery")
                || receipt.command_id != "local_recovery.restart_command")
        {
            return Err("invalid_receipt");
        }
        let mut safe = receipt.clone();
        safe.profile = safe_label(&safe.profile);
        safe.request_id = safe.request_id.map(|id| safe_label(&id));
        safe.attempt_id = safe.attempt_id.map(|id| safe_label(&id));
        safe.receipt_id = safe_label(&safe.receipt_id);
        let json = serde_json::to_string(&safe).map_err(|_| "invalid_receipt")?;
        let mut connection = self.connection.lock().map_err(|_| "store_busy")?;
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
        let result = receipt_sql!(
            self,
            &safe.receipt_id,
            (|| {
                let synchronous: u32 = connection
                    .query_row("PRAGMA synchronous", [], |row| row.get(0))
                    .map_err(|_| "write_failed")?;
                // Inspect the opened database: :memory: reports synchronous=FULL too.
                let database_file: String = connection
                    .query_row(
                        "SELECT file FROM pragma_database_list WHERE name = 'main'",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(|_| "write_failed")?;
                let journal_mode: String = connection
                    .query_row("PRAGMA journal_mode", [], |row| row.get(0))
                    .map_err(|_| "write_failed")?;
                if synchronous < 2
                    || database_file.is_empty()
                    || matches!(journal_mode.as_str(), "memory" | "off")
                {
                    return Err("durability_disabled");
                }
                let transaction = connection.transaction().map_err(|_| "write_failed")?;
                if let Some(identity) = guardian {
                    // The singleton survives receipt-ring retention and proxy restarts.
                    transaction.execute_batch("CREATE TABLE IF NOT EXISTS guardian_boot_claim (singleton INTEGER PRIMARY KEY CHECK(singleton = 1), boot_id TEXT NOT NULL, receipt_id TEXT NOT NULL)")
                    .map_err(|_| "write_failed")?;
                    let claimed = transaction.execute(
                    "INSERT INTO guardian_boot_claim(singleton, boot_id, receipt_id) VALUES (1, ?1, ?2) ON CONFLICT(singleton) DO UPDATE SET boot_id=excluded.boot_id, receipt_id=excluded.receipt_id WHERE guardian_boot_claim.boot_id != excluded.boot_id",
                    params![identity.boot_id, safe.receipt_id],
                ).map_err(|_| "write_failed")?;
                    if claimed != 1 {
                        return Err("boot_already_claimed");
                    }
                }
                transaction.execute(
                "INSERT INTO local_recovery_receipts(receipt_id, generated_at_unix_ms, receipt_json) VALUES (?1, ?2, ?3)",
                params![safe.receipt_id, safe.generated_at_unix_ms, json],
            ).map_err(|_| "write_failed")?;
                // ponytail: fixed 1024-row content-free ring; configurable retention only if needed.
                transaction.execute("DELETE FROM local_recovery_receipts WHERE rowid IN (SELECT rowid FROM local_recovery_receipts ORDER BY rowid DESC LIMIT -1 OFFSET 1024)", []).map_err(|_| "write_failed")?;
                transaction.commit().map_err(|_| "write_failed")
            })()
        );
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
        #[cfg(feature = "recovery-receipt-test-hooks")]
        self.observe_recovery_receipt_test("terminal_locked", receipt_id);
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
        #[cfg(feature = "recovery-receipt-test-hooks")]
        self.observe_recovery_receipt_test("terminal_sql_start", receipt_id);
        let result = connection
            .execute(
                "UPDATE local_recovery_receipts SET outcome = ?2, restart_status = ?3, readiness_status = ?4 WHERE receipt_id = ?1",
                params![receipt_id, safe_outcome, restart_status.map(safe_label), readiness_status.map(safe_label)],
            )
            .map_err(|_| "write_failed");
        connection
            .busy_timeout(std::time::Duration::from_millis(previous))
            .map_err(|_| "write_failed")?;
        #[cfg(feature = "recovery-receipt-test-hooks")]
        self.observe_recovery_receipt_test("terminal_sql_end", receipt_id);
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
