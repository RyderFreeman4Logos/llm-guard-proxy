#[cfg(unix)]
use std::time::Instant;
use std::{collections::BTreeMap, time::Duration};

#[cfg(any(
    target_os = "android",
    target_os = "freebsd",
    target_os = "haiku",
    target_os = "linux"
))]
use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
#[cfg(unix)]
use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use tokio::process::Command;
#[cfg(unix)]
use tokio::time::timeout;

const RECOVERY_PROCESS_GROUP_TERM_GRACE: Duration = Duration::from_millis(100);
const RECOVERY_PROCESS_GROUP_TERM_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Bounds every cleanup step after a recovery command has timed out.
///
/// The public coordinator starts its join deadline before the background task has been scheduled,
/// so `coordinator_handoff` also reserves a small scheduling and result-publication margin.
#[derive(Clone, Copy, Debug)]
struct RecoveryProcessGroupCleanupBudget {
    term_observation: Duration,
    kill_and_final_reap: Duration,
    coordinator_handoff: Duration,
}

impl RecoveryProcessGroupCleanupBudget {
    const fn bounded_cleanup_time(self) -> Duration {
        self.term_observation
            .saturating_add(self.kill_and_final_reap)
            .saturating_add(self.coordinator_handoff)
    }

    const fn public_join_timeout(self, recovery_timeout: Duration) -> Duration {
        recovery_timeout.saturating_add(self.bounded_cleanup_time())
    }
}

const RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET: RecoveryProcessGroupCleanupBudget =
    RecoveryProcessGroupCleanupBudget {
        term_observation: Duration::from_secs(2),
        kill_and_final_reap: Duration::from_millis(500),
        coordinator_handoff: Duration::from_millis(100),
    };

/// Returns the complete public wait bound for a timed recovery command and its cleanup.
pub(super) const fn recovery_join_timeout(recovery_timeout: Duration) -> Duration {
    RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.public_join_timeout(recovery_timeout)
}

/// Bounds state polling when a recovery-result notification is lost.
pub(super) const fn recovery_result_poll_interval() -> Duration {
    RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.coordinator_handoff
}

/// Owns a recovery child and its process group until the direct child is reaped.
///
/// Dropping an armed guard synchronously kills the group, then transfers direct-child reaping to
/// a bounded OS thread so async executor workers never block during cancellation.
pub(super) struct RecoveryProcessGuard {
    child: Option<tokio::process::Child>,
    #[cfg(unix)]
    process_group_id: Option<u32>,
}

impl RecoveryProcessGuard {
    pub(super) fn new(child: tokio::process::Child) -> Self {
        Self {
            #[cfg(unix)]
            process_group_id: child.id(),
            child: Some(child),
        }
    }

    #[cfg(unix)]
    fn process_group_id(&self) -> Option<u32> {
        self.process_group_id
    }

    pub(super) async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        let Some(child) = self.child.as_mut() else {
            return Err(std::io::Error::other("recovery child was already reaped"));
        };
        let result = child.wait().await;
        if result.is_ok() {
            self.disarm_after_reap();
        }
        result
    }

    /// Kills only the direct child when no validated process-group identity is available.
    ///
    /// The normal Unix timeout path signals the entire process group before waiting. This fallback
    /// is used only for a missing group ID or on platforms without recovery process groups.
    async fn kill_direct_child(&mut self) -> std::io::Result<()> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let result = child.kill().await;
        if result.is_ok() {
            self.disarm_after_reap();
        }
        result
    }

    fn disarm_after_reap(&mut self) {
        #[cfg(unix)]
        {
            self.process_group_id = None;
        }
        let _reaped_child = self.child.take();
    }
}

impl Drop for RecoveryProcessGuard {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        #[cfg(unix)]
        if let Some(process_group_id) = self.process_group_id.take() {
            let _group_kill_sent =
                send_recovery_process_group_signal(process_group_id, Signal::SIGKILL);
        }
        let _child_kill_started = child.start_kill();
        spawn_recovery_child_reaper(child);
    }
}

fn spawn_recovery_child_reaper(mut child: tokio::process::Child) {
    // Tokio documents orphan-queue cleanup as best-effort with no speed or frequency guarantee.
    // Retaining the owned child here gives cancellation a bounded `try_wait` loop; on Unix,
    // `try_wait` reaps an exited child. If thread creation fails, `kill_on_drop` still requests
    // direct-child termination and Tokio's orphan queue remains the best-effort fallback.
    let _reaper = std::thread::Builder::new()
        .name(String::from("llm-guard-recovery-reaper"))
        .spawn(move || {
            let deadline = std::time::Instant::now()
                + RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.kill_and_final_reap;
            loop {
                match child.try_wait() {
                    Ok(Some(_status)) => return,
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error)
                        if error.kind() == std::io::ErrorKind::Interrupted
                            && std::time::Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Ok(None) | Err(_) => return,
                }
            }
        });
}

#[cfg(unix)]
pub(super) fn configure_recovery_command(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
pub(super) fn configure_recovery_command(_command: &mut Command) {}

#[cfg(unix)]
pub(super) async fn terminate_timed_out_recovery_child(
    child: &mut RecoveryProcessGuard,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("upstream_stall_recovery_timeout_cleanup_scope"),
        String::from("process_group"),
    )]);
    let Some(pid) = child.process_group_id() else {
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_cleanup_status"),
            String::from("missing_child_pid"),
        );
        let _kill_result = child.kill_direct_child().await;
        return metadata;
    };

    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_sent"),
        send_recovery_process_group_signal(pid, Signal::SIGTERM).to_string(),
    );
    // WNOWAIT keeps the leader PID reserved until the final group signal, so
    // numeric PID reuse cannot redirect SIGKILL to an unrelated process group.
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_child_wait_status"),
        String::from(
            wait_for_term_child_exit_or_deadline(
                pid,
                RECOVERY_PROCESS_GROUP_TERM_GRACE,
                RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.term_observation,
            )
            .await,
        ),
    );

    metadata.insert(
        String::from("upstream_stall_recovery_timeout_kill_sent"),
        send_recovery_process_group_signal(pid, Signal::SIGKILL).to_string(),
    );
    let cleanup_status = match timeout(
        RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.kill_and_final_reap,
        child.wait(),
    )
    .await
    {
        Ok(Ok(_status)) => "terminated_after_kill",
        Ok(Err(_error)) => "wait_failed_after_kill",
        Err(_elapsed) => "wait_timeout_after_kill",
    };
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_cleanup_status"),
        String::from(cleanup_status),
    );
    metadata
}

#[cfg(not(unix))]
pub(super) async fn terminate_timed_out_recovery_child(
    child: &mut RecoveryProcessGuard,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([(
        String::from("upstream_stall_recovery_timeout_cleanup_scope"),
        String::from("child"),
    )]);
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_cleanup_status"),
        child.kill_direct_child().await.is_ok().to_string(),
    );
    metadata
}

#[cfg(unix)]
pub(super) fn send_recovery_process_group_signal(pid: u32, signal: Signal) -> bool {
    let Ok(process_group_id) = i32::try_from(pid) else {
        return false;
    };
    if process_group_id == 0 {
        return false;
    }
    kill(Pid::from_raw(-process_group_id), signal).is_ok()
}

/// Waits for a TERM-signalled direct child to exit without reaping it.
///
/// The initial grace preserves the normal TERM-only path. Subsequent bounded polling gives the
/// scheduler time to observe an exit under load while keeping the leader PID reserved by WNOWAIT
/// until the final process-group SIGKILL is sent.
#[cfg(unix)]
async fn wait_for_term_child_exit_or_deadline(
    pid: u32,
    minimum_grace: Duration,
    maximum_wait: Duration,
) -> &'static str {
    let deadline = Instant::now() + maximum_wait;
    tokio::time::sleep(minimum_grace.min(maximum_wait)).await;

    loop {
        let status = observe_recovery_child_without_reaping(pid);
        if status != "child_still_running_after_term" || Instant::now() >= deadline {
            return status;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        tokio::time::sleep(RECOVERY_PROCESS_GROUP_TERM_POLL_INTERVAL.min(remaining)).await;
    }
}

#[cfg(any(
    target_os = "android",
    target_os = "freebsd",
    target_os = "haiku",
    target_os = "linux"
))]
fn observe_recovery_child_without_reaping(pid: u32) -> &'static str {
    let Ok(pid) = i32::try_from(pid) else {
        return "invalid_child_pid";
    };
    if pid == 0 {
        return "invalid_child_pid";
    }
    let flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT;
    match waitid(Id::Pid(Pid::from_raw(pid)), flags) {
        Ok(WaitStatus::Exited(..) | WaitStatus::Signaled(..)) => "child_exited_unreaped_after_term",
        Ok(WaitStatus::StillAlive) => "child_still_running_after_term",
        Ok(_) => "child_state_changed_unreaped_after_term",
        Err(_) => "child_wait_failed_after_term",
    }
}

#[cfg(all(
    unix,
    not(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "linux"
    ))
))]
fn observe_recovery_child_without_reaping(_pid: u32) -> &'static str {
    "child_state_unavailable_before_kill"
}

#[cfg(test)]
mod tests {
    use super::{RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET, recovery_join_timeout};
    use std::time::Duration;

    #[test]
    fn public_join_timeout_covers_every_bounded_process_group_cleanup_phase() {
        let recovery_timeout = Duration::from_millis(1);
        let budget = RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET;
        let required = recovery_timeout
            .saturating_add(budget.term_observation)
            .saturating_add(budget.kill_and_final_reap)
            .saturating_add(budget.coordinator_handoff);

        assert!(
            recovery_join_timeout(recovery_timeout) >= required,
            "public join must outlive every bounded cleanup phase"
        );
    }
}
