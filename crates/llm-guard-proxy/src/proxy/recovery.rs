#[cfg(unix)]
use std::time::Instant;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

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

/// Owns a recovery child and its process group until settlement is proven.
///
/// Dropping an armed guard signals only while the captured leader identity still matches, then
/// transfers group census and reaping to a bounded OS thread. Unknown identity keeps ownership
/// fenced and suppresses cached-PID/PGID signals.
pub(super) struct RecoveryProcessGuard {
    child: Option<tokio::process::Child>,
    physical_owner: Arc<AtomicUsize>,
    #[cfg(unix)]
    process_group_id: Option<u32>,
    #[cfg(unix)]
    process_group_identity: Option<RecoveryProcessGroupIdentity>,
}

/// Captured group-leader instance; a numeric PGID alone is never signal authority.
#[derive(Clone, Copy)]
struct RecoveryProcessGroupIdentity {
    process_group_id: u32,
    #[cfg(target_os = "linux")]
    start_time_ticks: u64,
}

#[cfg(unix)]
impl RecoveryProcessGroupIdentity {
    fn capture(process_group_id: u32) -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            if !recovery_child_is_unreaped(process_group_id) {
                return None;
            }
            let stat = read_recovery_stat(std::fs::File::open(format!(
                "/proc/{process_group_id}/stat"
            )))
            .ok()??;
            let (observed_group, start_time_ticks, state) =
                recovery_stat_process_group_identity(&stat).ok()?;
            (observed_group == process_group_id && state.is_ascii_alphabetic()).then_some(Self {
                process_group_id,
                start_time_ticks,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = process_group_id;
            None
        }
    }

    fn is_current(self) -> bool {
        #[cfg(target_os = "linux")]
        {
            if !recovery_child_is_unreaped(self.process_group_id) {
                return false;
            }
            let Ok(Some(stat)) = read_recovery_stat(std::fs::File::open(format!(
                "/proc/{}/stat",
                self.process_group_id
            ))) else {
                return false;
            };
            recovery_stat_process_group_identity(&stat).is_ok_and(
                |(group, start_time_ticks, state)| {
                    group == self.process_group_id
                        && start_time_ticks == self.start_time_ticks
                        && state.is_ascii_alphabetic()
                },
            )
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl RecoveryProcessGuard {
    #[cfg(test)]
    pub(super) fn new(child: tokio::process::Child) -> Self {
        Self::new_owned(child, Arc::new(AtomicUsize::new(0)))
    }

    pub(super) fn new_owned(
        child: tokio::process::Child,
        physical_owner: Arc<AtomicUsize>,
    ) -> Self {
        physical_owner.fetch_add(1, Ordering::AcqRel);
        #[cfg(unix)]
        let process_group_id = child.id();
        #[cfg(unix)]
        let process_group_identity =
            process_group_id.and_then(RecoveryProcessGroupIdentity::capture);
        Self {
            physical_owner,
            #[cfg(unix)]
            process_group_id,
            #[cfg(unix)]
            process_group_identity,
            child: Some(child),
        }
    }

    #[cfg(unix)]
    fn process_group_id(&self) -> Option<u32> {
        self.process_group_id
    }

    #[cfg(unix)]
    fn process_group_identity(&self) -> Option<RecoveryProcessGroupIdentity> {
        self.process_group_identity
    }

    #[cfg(unix)]
    pub(super) fn signal_process_group(&self, signal: Signal) -> bool {
        self.process_group_identity
            .is_some_and(|identity| send_recovery_process_group_signal(identity, signal))
    }

    pub(super) async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        #[cfg(unix)]
        if self.process_group_id.is_some() && self.process_group_identity.is_none() {
            return Err(std::io::Error::other("recovery child identity unavailable"));
        }
        #[cfg(target_os = "linux")]
        if let Some(pid) = self.process_group_id {
            let Some(identity) = self.process_group_identity else {
                return Err(std::io::Error::other("recovery child identity unavailable"));
            };
            // Keep the leader unreaped until its group is stopped: reaping first
            // permits PID reuse and makes any later negative-PID signal unsafe.
            if !identity.is_current() {
                return Err(std::io::Error::other("recovery child identity unavailable"));
            }
            loop {
                match observe_recovery_child_without_reaping(pid) {
                    "child_exited_unreaped_after_term" => break,
                    "child_still_running_after_term" => {
                        tokio::time::sleep(RECOVERY_PROCESS_GROUP_TERM_POLL_INTERVAL).await;
                    }
                    _ => return Err(std::io::Error::other("recovery child identity unavailable")),
                }
            }
            let _sent = self.signal_process_group(Signal::SIGKILL);
            if !identity.is_current() {
                return Err(std::io::Error::other("recovery child identity unavailable"));
            }
            loop {
                if !identity.is_current() {
                    return Err(std::io::Error::other("recovery child identity unavailable"));
                }
                if recovery_group_quiescent(pid)? {
                    break;
                }
                tokio::time::sleep(RECOVERY_PROCESS_GROUP_TERM_POLL_INTERVAL).await;
            }
        }
        let Some(child) = self.child.as_mut() else {
            return Err(std::io::Error::other("recovery child was already reaped"));
        };
        let result = child.wait().await;
        if result.is_ok() {
            self.disarm_after_reap();
        }
        result
    }

    /// Kills and reaps the direct child on platforms without Unix process-group ownership.
    ///
    /// Unix cleanup requires a current leader identity; an unknown identity returns an error and
    /// leaves physical ownership fenced instead of signaling a possibly recycled PID.
    async fn kill_direct_child(&mut self) -> std::io::Result<()> {
        #[cfg(unix)]
        if !self
            .process_group_identity
            .is_some_and(RecoveryProcessGroupIdentity::is_current)
        {
            return Err(std::io::Error::other("recovery child identity unavailable"));
        }
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let result = child.kill().await;
        if result.is_ok() {
            #[cfg(unix)]
            if self.process_group_id.is_some() {
                return result;
            }
            self.disarm_after_reap();
        }
        result
    }

    fn disarm_after_reap(&mut self) {
        #[cfg(unix)]
        {
            self.process_group_id = None;
            self.process_group_identity = None;
        }
        if self.child.take().is_some() {
            self.physical_owner.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Observes group quiescence while its unreaped leader still pins the PGID.
/// Zombies cannot execute; their parents retain responsibility for reaping.
#[cfg(target_os = "linux")]
fn recovery_group_quiescent(group: u32) -> std::io::Result<bool> {
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        if entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
            .is_none()
        {
            continue;
        }
        let Some(stat) = read_recovery_stat(std::fs::File::open(entry.path().join("stat")))? else {
            continue;
        };
        if recovery_stat_is_active_member(&stat, group)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Both pathname-open and already-open procfs reads race ordinary process exit.
#[cfg(target_os = "linux")]
fn read_recovery_stat(file: std::io::Result<std::fs::File>) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read as _;
    let result = file.and_then(|mut file| {
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    match result {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                || error.raw_os_error() == Some(libc::ESRCH) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// `comm` is an uninterpreted byte string; only the control fields must be ASCII.
#[cfg(target_os = "linux")]
fn recovery_stat_is_active_member(stat: &[u8], group: u32) -> std::io::Result<bool> {
    let invalid = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid process stat control fields",
        )
    };
    let offset = stat
        .windows(2)
        .rposition(|pair| pair == b") ")
        .ok_or_else(invalid)?
        + 2;
    let mut fields = stat[offset..]
        .split(u8::is_ascii_whitespace)
        .filter(|field| !field.is_empty());
    let state = fields
        .next()
        .filter(|field| field.len() == 1 && field[0].is_ascii_alphabetic())
        .ok_or_else(invalid)?;
    let _parent = fields.next().ok_or_else(invalid)?;
    let pgid = fields
        .next()
        .filter(|field| !field.is_empty() && field.iter().all(u8::is_ascii_digit))
        .and_then(|field| std::str::from_utf8(field).ok())
        .and_then(|field| field.parse::<u32>().ok())
        .ok_or_else(invalid)?;
    Ok(pgid == group && !matches!(state, b"Z" | b"X"))
}

#[cfg(target_os = "linux")]
/// Returns `(process_group_id, start_time_ticks, state)` from one proc stat row.
/// Keep the validated state in the contract so callers cannot accidentally skip it.
fn recovery_stat_process_group_identity(stat: &[u8]) -> std::io::Result<(u32, u64, u8)> {
    let invalid = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid process stat identity fields",
        )
    };
    let offset = stat
        .windows(2)
        .rposition(|pair| pair == b") ")
        .ok_or_else(invalid)?
        + 2;
    let mut fields = stat[offset..]
        .split(u8::is_ascii_whitespace)
        .filter(|field| !field.is_empty());
    let state = fields
        .next()
        .filter(|field| field.len() == 1 && field[0].is_ascii_alphabetic())
        .ok_or_else(invalid)?;
    let _parent = fields.next().ok_or_else(invalid)?;
    let process_group_id = fields
        .next()
        .filter(|field| !field.is_empty() && field.iter().all(u8::is_ascii_digit))
        .and_then(|field| std::str::from_utf8(field).ok())
        .and_then(|field| field.parse::<u32>().ok())
        .ok_or_else(invalid)?;
    // After field 5 (pgrp), starttime is field 22, with sixteen fields between.
    let start_time_ticks = fields
        .nth(16)
        .filter(|field| !field.is_empty() && field.iter().all(u8::is_ascii_digit))
        .and_then(|field| std::str::from_utf8(field).ok())
        .and_then(|field| field.parse::<u64>().ok())
        .ok_or_else(invalid)?;
    Ok((process_group_id, start_time_ticks, state[0]))
}

impl Drop for RecoveryProcessGuard {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        #[cfg(unix)]
        let process_group_id = self.process_group_id.take();
        #[cfg(unix)]
        let process_group_identity = self.process_group_identity.take();
        #[cfg(unix)]
        if let Some(identity) = process_group_identity {
            let _group_kill_sent = send_recovery_process_group_signal(identity, Signal::SIGKILL);
        }
        #[cfg(unix)]
        let _child_kill_started =
            if process_group_identity.is_some_and(RecoveryProcessGroupIdentity::is_current) {
                child.start_kill()
            } else {
                Ok(())
            };
        #[cfg(not(unix))]
        let process_group_id = None;
        #[cfg(not(unix))]
        let process_group_identity = None;
        #[cfg(not(unix))]
        let _child_kill_started = child.start_kill();
        spawn_recovery_child_reaper(
            child,
            process_group_id,
            process_group_identity,
            Arc::clone(&self.physical_owner),
        );
    }
}

fn spawn_recovery_child_reaper(
    mut child: tokio::process::Child,
    process_group_id: Option<u32>,
    process_group_identity: Option<RecoveryProcessGroupIdentity>,
    physical_owner: Arc<AtomicUsize>,
) {
    if process_group_id.is_some() && process_group_identity.is_none() {
        return;
    }
    // Tokio documents orphan-queue cleanup as best-effort with no speed or frequency guarantee.
    // Retaining the owned child here gives cancellation a bounded `try_wait` loop; on Unix,
    // `try_wait` reaps an exited child. Failed thread creation never releases physical ownership.
    let _reaper = std::thread::Builder::new()
        .name(String::from("llm-guard-recovery-reaper"))
        .spawn(move || {
            let deadline = std::time::Instant::now()
                + RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.kill_and_final_reap;
            loop {
                // Do not reap the leader before confirming its pinned group is
                // quiescent. A failed census/identity or exhausted reaper retains
                // fail-closed ownership independently of waiter/audit completion.
                #[cfg(target_os = "linux")]
                if let Some(group) = process_group_id {
                    let Some(identity) = process_group_identity else {
                        return;
                    };
                    if identity.process_group_id != group || !identity.is_current() {
                        return;
                    }
                    match observe_recovery_child_without_reaping(group) {
                        "child_exited_unreaped_after_term" => match recovery_group_quiescent(group)
                        {
                            Ok(true) => {}
                            Ok(false) if std::time::Instant::now() < deadline => {
                                std::thread::sleep(Duration::from_millis(10));
                                continue;
                            }
                            Ok(false) | Err(_) => return,
                        },
                        "child_still_running_after_term"
                            if std::time::Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        _ => return,
                    }
                }
                match child.try_wait() {
                    Ok(Some(_status)) => {
                        physical_owner.fetch_sub(1, Ordering::AcqRel);
                        return;
                    }
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
    // RecoveryProcessGuard must validate ownership before every signal; Tokio's cached-PID
    // kill_on_drop fallback is unsafe after an external reaper and possible PID reuse.
    command.kill_on_drop(false);
    command.process_group(0);
}

#[cfg(not(unix))]
pub(super) fn configure_recovery_command(command: &mut Command) {
    command.kill_on_drop(false);
}

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
        String::from("upstream_stall_recovery_timeout_cleanup_status"),
        String::from("group_identity_unconfirmed"),
    );
    let Some(_identity) = child.process_group_identity() else {
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_term_sent"),
            String::from("false"),
        );
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_term_child_wait_status"),
            String::from("child_identity_unavailable"),
        );
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_kill_sent"),
            String::from("false"),
        );
        return metadata;
    };

    let term_sent = child.signal_process_group(Signal::SIGTERM);
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_sent"),
        term_sent.to_string(),
    );
    if !term_sent {
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_term_child_wait_status"),
            String::from("child_identity_unavailable"),
        );
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_kill_sent"),
            String::from("false"),
        );
        return metadata;
    }

    let term_wait_status = wait_for_term_child_exit_or_deadline(
        pid,
        RECOVERY_PROCESS_GROUP_TERM_GRACE,
        RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.term_observation,
    )
    .await;
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_term_child_wait_status"),
        String::from(term_wait_status),
    );
    if !matches!(
        term_wait_status,
        "child_still_running_after_term" | "child_exited_unreaped_after_term"
    ) {
        metadata.insert(
            String::from("upstream_stall_recovery_timeout_kill_sent"),
            String::from("false"),
        );
        return metadata;
    }

    let kill_sent = child.signal_process_group(Signal::SIGKILL);
    metadata.insert(
        String::from("upstream_stall_recovery_timeout_kill_sent"),
        kill_sent.to_string(),
    );
    if !kill_sent {
        return metadata;
    }
    let cleanup_status = match timeout(
        RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET.kill_and_final_reap,
        child.wait(),
    )
    .await
    {
        Ok(Ok(_status)) => "terminated_after_kill",
        Ok(Err(error)) => {
            #[cfg(test)]
            eprintln!("recovery cleanup wait error={error:?}");
            drop(error);
            "wait_failed_after_kill"
        }
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
fn send_recovery_process_group_signal(
    identity: RecoveryProcessGroupIdentity,
    signal: Signal,
) -> bool {
    if !identity.is_current() {
        return false;
    }
    let Ok(process_group_id) = i32::try_from(identity.process_group_id) else {
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

#[cfg(unix)]
fn recovery_child_is_unreaped(pid: u32) -> bool {
    matches!(
        observe_recovery_child_without_reaping(pid),
        "child_still_running_after_term" | "child_exited_unreaped_after_term"
    )
}

#[cfg(all(test, target_os = "linux"))]
#[path = "recovery_proc_tests.rs"]
mod proc_tests;

#[cfg(test)]
mod tests {
    use super::{RECOVERY_PROCESS_GROUP_CLEANUP_BUDGET, recovery_join_timeout};
    use std::time::Duration;

    #[tokio::test]
    async fn owned_recovery_completion_cannot_leave_a_running_descendant() {
        if std::env::var_os("GUARDIAN_DESCENDANT_TEST").is_none() {
            let mut child = tokio::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", "proxy::recovery::tests::owned_recovery_completion_cannot_leave_a_running_descendant", "--nocapture"])
                .env("GUARDIAN_DESCENDANT_TEST", "1")
                .kill_on_drop(true).spawn().expect("isolated subreaper fixture");
            let result = tokio::time::timeout(Duration::from_secs(15), child.wait()).await;
            if result.is_err() {
                child.kill().await.expect("kill timed out fixture");
            }
            assert!(
                result
                    .expect("bounded fixture")
                    .expect("fixture wait")
                    .success()
            );
            return;
        }
        nix::sys::prctl::set_child_subreaper(true).expect("isolated fixture subreaper");
        let root =
            std::env::temp_dir().join(format!("owned-recovery-completion-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("fixture directory");
        let marker = root.join("descendant");
        let mut command = tokio::process::Command::new("python3");
        command.args(["-c", "import os,time,sys,pathlib; p=os.fork(); pathlib.Path(sys.argv[1]).write_text(str(p)) if p else time.sleep(30)"]);
        command.arg(&marker).kill_on_drop(true);
        super::configure_recovery_command(&mut command);
        let mut owned = super::RecoveryProcessGuard::new(command.spawn().expect("owned fixture"));
        let status = tokio::time::timeout(Duration::from_secs(5), owned.wait())
            .await
            .expect("bounded completion")
            .expect("reaped leader");
        assert!(status.success());
        let pid: i32 = std::fs::read_to_string(&marker)
            .expect("child identity")
            .parse()
            .expect("pid");
        let identity = std::fs::read_to_string(format!("/proc/{pid}/stat"));
        eprintln!("fixture descendant pid={pid} stat={identity:?}");
        let running =
            identity.is_ok_and(|stat| !stat.rsplit_once(") ").expect("stat").1.starts_with('Z'));
        // The isolated subreaper owns this descendant, including on behavioral RED.
        let _cleanup = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid),
            nix::sys::signal::Signal::SIGKILL,
        );
        let reaped = nix::sys::wait::waitpid(nix::unistd::Pid::from_raw(pid), None)
            .expect("reap descendant");
        eprintln!("fixture descendant reaped={reaped:?}");
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
        std::fs::remove_dir_all(root).expect("fixture cleanup");
        assert!(
            !running,
            "completion acknowledged while owned recovery descendant remained running"
        );
    }

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
