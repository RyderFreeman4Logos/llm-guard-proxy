//! Pre-spawn durable attribution shared by request and watchdog recovery.
use super::{
    AttemptId, LocalRecoveryCause, LocalRecoveryPolicy, ObservabilityStore, RequestId,
    UpstreamProfileConfig, duration_millis_u64, unix_time_millis,
};
use llm_guard_proxy_state::LocalRecoveryReceipt;
use std::{hash::BuildHasher, time::Duration};

static COMMAND_KEY: std::sync::OnceLock<std::hash::RandomState> = std::sync::OnceLock::new();

const WRITE_BUDGET: Duration = Duration::from_millis(250);

#[derive(Clone)]
pub(super) struct Context {
    store: ObservabilityStore,
    tasks: super::Arc<super::PersistenceTasks>,
    acknowledged: super::Arc<super::AtomicBool>,
    pub(super) physical_owner: super::Arc<super::AtomicUsize>,
    receipt: LocalRecoveryReceipt,
    owner_deadline: Option<super::Instant>,
    downstream_drop: Option<super::DownstreamDropSignal>,
    shutdown: Option<super::Arc<super::ShutdownGate>>,
}

impl Context {
    pub(super) fn new(
        store: ObservabilityStore,
        profile: &UpstreamProfileConfig,
        tasks: super::Arc<super::PersistenceTasks>,
    ) -> Self {
        Self {
            store,
            tasks,
            owner_deadline: None,
            downstream_drop: None,
            shutdown: None,
            physical_owner: super::Arc::new(super::AtomicUsize::new(0)),
            acknowledged: super::Arc::new(super::AtomicBool::new(false)),
            receipt: LocalRecoveryReceipt {
                guardian: None,
                receipt_id: format!("recovery-{}-{}", std::process::id(), RequestId::generate()),
                generated_at_unix_ms: 0,
                process_id: std::process::id(),
                episode: 0,
                detector: "precommit_recovery",
                cause: "upstream_stall",
                profile: profile.name.clone(),
                command_id: "local_recovery.restart_command",
                command_generation: 0,
                request_id: None,
                attempt_id: None,
                restart_timeout_ms: 0,
                readiness_deadline_ms: 0,
                first_chunk_timeout_ms: 0,
                idle_timeout_ms: 0,
                request_deadline_ms: None,
                detection_window_secs: profile.stuck_watchdog.detection_window_secs,
                min_output_progress_units: profile
                    .stuck_watchdog
                    .min_output_progress_units_in_window,
            },
        }
    }

    pub(super) fn physical_owner(mut self, owner: super::Arc<super::AtomicUsize>) -> Self {
        self.physical_owner = owner;
        self
    }

    pub(super) fn stall(mut self, first_chunk_ms: u64, idle_ms: u64) -> Self {
        self.receipt.first_chunk_timeout_ms = first_chunk_ms;
        self.receipt.idle_timeout_ms = idle_ms;
        self
    }

    pub(super) fn request(mut self, request_id: &RequestId, deadline: Duration) -> Self {
        self.receipt.request_id = Some(request_id.as_str().to_owned());
        self.receipt.request_deadline_ms = Some(duration_millis_u64(deadline));
        self
    }

    pub(super) fn attempt(mut self, id: Option<&AttemptId>) -> Self {
        self.receipt.attempt_id = id.map(|id| id.as_str().to_owned());
        self
    }

    pub(super) fn episode(
        mut self,
        episode: u64,
        cause: LocalRecoveryCause,
        policy: &LocalRecoveryPolicy,
    ) -> Self {
        self.receipt.episode = episode;
        // Process-scoped keyed identity: correlate command generations without retaining argv.
        self.receipt.command_generation = COMMAND_KEY
            .get_or_init(std::hash::RandomState::new)
            .hash_one(&policy.restart_command);
        self.receipt.cause = cause.as_str();
        self.receipt.detector = if cause == LocalRecoveryCause::StuckWatchdog {
            "stuck_watchdog"
        } else {
            "precommit_recovery"
        };
        self.receipt.restart_timeout_ms = duration_millis_u64(policy.restart_timeout);
        self.receipt.readiness_deadline_ms = duration_millis_u64(policy.readiness_deadline);
        self
    }

    #[cfg(feature = "memory-guardian")]
    pub(super) fn guardian(
        mut self,
        identity: llm_guard_proxy_state::GuardianRecoveryIdentity,
    ) -> Self {
        self.receipt.guardian = Some(identity);
        self.receipt.detector = "memory_guardian";
        self.receipt.cause = "memory_pressure";
        self.receipt.command_id = "guardian.foreground_recovery";
        self
    }

    pub(super) fn deadline(mut self, deadline: super::Instant) -> Self {
        self.owner_deadline = Some(
            self.owner_deadline
                .map_or(deadline, |old| old.min(deadline)),
        );
        self
    }

    pub(super) fn downstream_drop(mut self, signal: Option<super::DownstreamDropSignal>) -> Self {
        self.downstream_drop = signal;
        self
    }

    pub(super) fn shutdown(mut self, shutdown: super::Arc<super::ShutdownGate>) -> Self {
        self.shutdown = Some(shutdown);
        self
    }

    pub(super) fn owner_cancelled(&self, metadata: &mut super::BTreeMap<String, String>) -> bool {
        let status = if self
            .shutdown
            .as_ref()
            .is_some_and(|gate| gate.is_shutting_down())
        {
            Some("shutdown_cancelled")
        } else if self
            .downstream_drop
            .as_ref()
            .is_some_and(super::DownstreamDropSignal::is_dropped)
        {
            Some("skipped_downstream_dropped")
        } else if self
            .owner_deadline
            .is_some_and(|deadline| super::Instant::now() >= deadline)
        {
            Some("episode_timeout")
        } else {
            None
        };
        if let Some(status) = status {
            metadata.insert(String::from("local_recovery_status"), status.to_owned());
            metadata.insert(
                String::from("local_recovery_restart_status"),
                status.to_owned(),
            );
            return true;
        }
        false
    }

    pub(super) fn track(&self) -> super::PersistenceTaskGuard {
        self.tasks.track()
    }

    pub(super) fn acknowledged_id(&self) -> Option<&str> {
        self.acknowledged
            .load(super::Ordering::Acquire)
            .then(|| self.id())
    }

    pub(super) fn id(&self) -> &str {
        &self.receipt.receipt_id
    }

    pub(super) async fn pre_spawn(
        &self,
        signal: Option<&super::DownstreamCommitSignal>,
    ) -> Result<super::BTreeMap<String, String>, super::BTreeMap<String, String>> {
        let mut metadata = super::BTreeMap::new();
        if self.owner_cancelled(&mut metadata) {
            return Err(metadata);
        }
        if let Err(category) = self.persist().await {
            return Err(super::BTreeMap::from([
                (
                    String::from("local_recovery_status"),
                    String::from("receipt_failed"),
                ),
                (
                    String::from("local_recovery_restart_status"),
                    String::from("receipt_failed"),
                ),
                (
                    String::from("local_recovery_receipt_error"),
                    category.to_owned(),
                ),
            ]));
        }
        let mut metadata = super::BTreeMap::from([(
            String::from("local_recovery_receipt_id"),
            self.id().to_owned(),
        )]);
        if self.owner_cancelled(&mut metadata)
            || super::local_recovery_downstream_commit_observed(signal, &mut metadata)
        {
            return Err(metadata);
        }
        Ok(metadata)
    }

    async fn write(
        &self,
        deadline: std::time::Instant,
        work: impl FnOnce(std::time::Instant) -> Result<(), &'static str> + Send + 'static,
    ) -> Result<(), &'static str> {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .ok_or("write_timeout")?;
        let permit = super::Arc::clone(&self.tasks.capacity)
            .try_acquire_owned()
            .map_err(|_| "writer_busy")?;
        let guard = self.track();
        // Cancellation abandons the acknowledgment, not ownership. Shutdown's existing
        // persistence drain observes this worker; neither the worker nor its guard can spawn.
        let writer = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _guard = guard;
            work(deadline)
        });
        let result = tokio::time::timeout(remaining, writer)
            .await
            .map_err(|_| "write_timeout")?
            .map_err(|_| "writer_failed")?;
        // Timeout polls a ready writer first; late acknowledgment cannot grant spawn authority.
        if std::time::Instant::now() >= deadline {
            return Err("write_timeout");
        }
        result
    }

    pub(super) async fn persist(&self) -> Result<(), &'static str> {
        let store = self.store.clone();
        let mut receipt = self.receipt.clone();
        receipt.generated_at_unix_ms = unix_time_millis();
        let deadline = std::time::Instant::now() + WRITE_BUDGET;
        let deadline = self
            .owner_deadline
            .map_or(deadline, |owner| deadline.min(owner.into_std()));
        #[cfg(test)]
        self.store
            .observe_recovery_receipt_test("record_submit", self.id());
        let result = self
            .write(deadline, move |deadline| {
                #[cfg(test)]
                store.observe_recovery_receipt_test("record_worker_start", &receipt.receipt_id);
                store.record_local_recovery_receipt(&receipt, deadline)
            })
            .await;
        #[cfg(test)]
        self.store
            .observe_recovery_receipt_test("record_ack", self.id());
        result?;
        self.acknowledged.store(true, super::Ordering::Release);
        Ok(())
    }

    pub(super) async fn finish(
        &self,
        metadata: &std::collections::BTreeMap<String, String>,
    ) -> Result<(), &'static str> {
        let outcome = metadata
            .get("local_recovery_status")
            .cloned()
            .unwrap_or_else(|| String::from("unknown"));
        let restart = metadata.get("local_recovery_restart_status").cloned();
        let readiness = metadata.get("local_recovery_readiness_status").cloned();
        let store = self.store.clone();
        let id = self.id().to_owned();
        #[cfg(test)]
        self.store
            .observe_recovery_receipt_test("terminal_submit", self.id());
        let result = self
            .write(std::time::Instant::now() + WRITE_BUDGET, move |deadline| {
                #[cfg(test)]
                store.observe_recovery_receipt_test("terminal_worker_start", &id);
                store.finish_local_recovery_receipt(
                    &id,
                    &outcome,
                    restart.as_deref(),
                    readiness.as_deref(),
                    deadline,
                )
            })
            .await;
        #[cfg(test)]
        self.store
            .observe_recovery_receipt_test("terminal_ack", self.id());
        result
    }
}
