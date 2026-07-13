//! Owned workflow subprocess lifecycle and bounded cleanup.
//!
//! The runtime owns and terminates the process group rooted at the spawned leader. A descendant
//! that calls `setsid(2)` or `setpgid(2)` escapes that signal boundary and is not killed by this
//! module. Guaranteed whole-tree termination requires stronger containment such as a cgroup.
//! Deadline-driven local pipe I/O still closes inherited stdin/stdout/stderr file descriptors on
//! time, so escaped descendants cannot hold a request open past its cleanup grace.
//!
//! This module must be the sole waiter for its private workflow children. Embedders must not use
//! process-global `SIGCHLD = SIG_IGN` or `SA_NOCLDWAIT`, and must not run a competing waiter for
//! workflow PIDs. Unexpected `ECHILD` revokes signal authority and leaves admission fail-closed.

use std::{
    env,
    io::{self},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use llm_guard_proxy_core::WorkflowConfig;

use crate::workflow_execution::WorkflowExecutionLease;

#[cfg(test)]
use self::deferred_reaper::{
    DeferredPollOutcome, DeferredWorkflowProcess, SharedDeferredReaper, defer_workflow_process,
    grow_signal_retry_backoff, next_deferred_poll_delay, poll_deferred_cleanup_with,
    poll_deferred_processes,
};
#[cfg(test)]
pub(crate) use self::signal_authority::{linux_process_is_live, linux_process_start_time};
#[cfg(test)]
use self::signal_authority::{signal_owned_process_group_with, signal_owned_workflow_with};
use self::{
    deferred_reaper::{
        DeferredSignalState, defer_workflow_process_with_execution_lease, shared_deferred_reaper,
    },
    drain::{PipeDrainHandle, spawn_pipe_drain},
    pipe_io::{read_bounded_deadline, write_all_deadline},
    signal_authority::{
        LinuxProcessIdentity, NonReapingChildState, ProcessGroupSignalError, SignalAuthority,
        WorkflowSignalOutcome, signal_owned_workflow,
    },
    startup_guard::startup_error_revokes_provisional_authority,
};

pub(crate) use self::startup_guard::SpawnedChildGuard;

#[cfg(test)]
use self::startup_guard::finalize_provisional_child_with;

#[cfg(not(test))]
use self::drain::cleanup_and_finish_drain_with;
#[cfg(test)]
use self::drain::cleanup_and_finish_drain_with;
#[cfg(test)]
pub(crate) use self::drain::install_stderr_drain_completion_probe;

const PROCESS_REAP_GRACE: Duration = Duration::from_millis(500);
const PROCESS_REAP_POLL: Duration = Duration::from_millis(10);
const IDENTITY_RETRY_POLL: Duration = Duration::from_millis(1);

const ALLOWED_ENV_VARS: [&str; 4] = ["PATH", "LANG", "LC_ALL", "HOME"];

/// One armed workflow subprocess owner.
///
/// Dropping an armed value synchronously performs bounded cleanup or transfers the unique child
/// handle to the shared deferred reaper. Normal completion and abort disarm exactly once.
pub(super) struct WorkflowProcess {
    armed: Option<ArmedWorkflowProcess>,
    timed_out: Arc<AtomicBool>,
}

struct ArmedWorkflowProcess {
    child: Option<Child>,
    signal_authority: SignalAuthority,
    watchdog: Option<Watchdog>,
    stderr_handle: Option<PipeDrainHandle>,
    execution_deadline: Instant,
    execution_lease: WorkflowExecutionLease,
}

pub(super) struct WorkflowProcessCompletion {
    pub(super) status: ExitStatus,
    pub(super) timed_out: bool,
}

pub(super) enum WorkflowProcessStartError {
    CleanupUnavailable,
    Spawn(io::Error),
    IdentityCapture(&'static str),
    IdentityChanged(&'static str),
    Watchdog(io::Error),
    StderrDrain(io::Error),
}

pub(super) enum WorkflowStdinError {
    Missing,
    Write(io::Error),
}

pub(super) enum WorkflowStdoutError {
    Missing,
    Read(io::Error),
}

pub(super) enum WorkflowProcessFinishError {
    OwnershipLost,
    Cleanup(io::Error),
    DeadlineExceeded(io::Error),
    Stderr(io::Error),
}

impl WorkflowProcess {
    #[cfg(test)]
    pub(super) fn start(
        config: &WorkflowConfig,
        timeout: Duration,
    ) -> Result<Self, WorkflowProcessStartError> {
        Self::start_with_identity_capture_and_lease(
            config,
            timeout,
            WorkflowExecutionLease::default(),
            LinuxProcessIdentity::capture,
        )
    }

    pub(super) fn start_with_execution_lease(
        config: &WorkflowConfig,
        timeout: Duration,
        execution_lease: WorkflowExecutionLease,
    ) -> Result<Self, WorkflowProcessStartError> {
        Self::start_with_identity_capture_and_lease(
            config,
            timeout,
            execution_lease,
            LinuxProcessIdentity::capture,
        )
    }

    #[cfg(test)]
    fn start_with_identity_capture<Capture>(
        config: &WorkflowConfig,
        timeout: Duration,
        capture_identity: Capture,
    ) -> Result<Self, WorkflowProcessStartError>
    where
        Capture: FnMut(u32) -> Result<LinuxProcessIdentity, ProcessGroupSignalError>,
    {
        Self::start_with_identity_capture_and_lease(
            config,
            timeout,
            WorkflowExecutionLease::default(),
            capture_identity,
        )
    }

    fn start_with_identity_capture_and_lease<Capture>(
        config: &WorkflowConfig,
        timeout: Duration,
        execution_lease: WorkflowExecutionLease,
        capture_identity: Capture,
    ) -> Result<Self, WorkflowProcessStartError>
    where
        Capture: FnMut(u32) -> Result<LinuxProcessIdentity, ProcessGroupSignalError>,
    {
        // An unresolved cleanup retains its leader to anchor PGID ownership. Refusing new
        // children bounds retained processes to requests that were already running.
        if !cleanup_worker_available() {
            return Err(WorkflowProcessStartError::CleanupUnavailable);
        }
        let execution_deadline = Instant::now() + timeout;
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_allowed_environment(&mut command);
        configure_process_group(&mut command);

        let startup_cleanup_deadline = abort_cleanup_deadline(execution_deadline, Instant::now());
        let child = command.spawn().map_err(WorkflowProcessStartError::Spawn)?;
        let mut spawned = SpawnedChildGuard::new_with_execution_lease(
            child,
            startup_cleanup_deadline,
            execution_lease,
        );
        let process_identity = match capture_process_identity_bounded_with(
            spawned.pid(),
            capture_identity,
            || Instant::now() >= startup_cleanup_deadline,
            || thread::sleep(IDENTITY_RETRY_POLL),
        ) {
            Ok(identity) => identity,
            Err(error) => {
                if startup_error_revokes_provisional_authority(error) {
                    spawned.revoke_provisional_authority();
                }
                return Err(WorkflowProcessStartError::IdentityCapture(error.as_str()));
            }
        };
        let signal_authority = SignalAuthority::new(process_identity);
        if let Err(error) =
            validate_signal_authority_bounded(&signal_authority, startup_cleanup_deadline)
        {
            if startup_error_revokes_provisional_authority(error) {
                spawned.revoke_provisional_authority();
            }
            return Err(WorkflowProcessStartError::IdentityChanged(error.as_str()));
        }
        spawned.set_signal_authority(signal_authority.clone());

        let timed_out = Arc::new(AtomicBool::new(false));
        let watchdog = spawn_timeout_watchdog(
            signal_authority.clone(),
            execution_deadline,
            Arc::clone(&timed_out),
        )
        .map_err(WorkflowProcessStartError::Watchdog)?;
        let stderr_handle = match spawned
            .child_mut()
            .and_then(|child| child.stderr.take())
            .map(|pipe| spawn_pipe_drain(pipe, execution_deadline))
            .transpose()
        {
            Ok(handle) => handle,
            Err(error) => return Err(WorkflowProcessStartError::StderrDrain(error)),
        };
        let Some((child, execution_lease)) = spawned.disarm_with_execution_lease() else {
            return Err(WorkflowProcessStartError::IdentityChanged(
                "spawned_child_missing",
            ));
        };

        Ok(Self {
            armed: Some(ArmedWorkflowProcess {
                child: Some(child),
                signal_authority,
                watchdog: Some(watchdog),
                stderr_handle,
                execution_deadline,
                execution_lease,
            }),
            timed_out,
        })
    }

    pub(super) fn write_stdin(&mut self, payload: &[u8]) -> Result<(), WorkflowStdinError> {
        let Some(armed) = self.armed.as_mut() else {
            return Err(WorkflowStdinError::Missing);
        };
        let stdin = armed
            .child
            .as_mut()
            .and_then(|child| child.stdin.take())
            .ok_or(WorkflowStdinError::Missing)?;
        write_all_deadline(stdin, payload, armed.execution_deadline)
            .map_err(WorkflowStdinError::Write)
    }

    pub(super) fn read_stdout(
        &mut self,
        max_stdout_bytes: usize,
    ) -> Result<Vec<u8>, WorkflowStdoutError> {
        let Some(armed) = self.armed.as_mut() else {
            return Err(WorkflowStdoutError::Missing);
        };
        let stdout = armed
            .child
            .as_mut()
            .and_then(|child| child.stdout.take())
            .ok_or(WorkflowStdoutError::Missing)?;
        read_bounded_deadline(stdout, max_stdout_bytes, armed.execution_deadline)
            .map_err(WorkflowStdoutError::Read)
    }

    pub(super) fn abort(mut self) {
        if let Some(armed) = self.armed.take() {
            armed.abort();
        }
    }

    pub(super) fn complete(
        mut self,
    ) -> Result<WorkflowProcessCompletion, WorkflowProcessFinishError> {
        let Some(mut armed) = self.armed.take() else {
            return Err(WorkflowProcessFinishError::OwnershipLost);
        };
        let cleanup_deadline = armed.execution_deadline + PROCESS_REAP_GRACE;
        let execution_deadline = armed.execution_deadline;
        let observation =
            observe_child_exit_without_reaping(&armed.signal_authority, cleanup_deadline);
        if observation == ExitObservation::OwnershipLost {
            armed.cancel_watchdog();
            if let Some(child) = armed.child.take() {
                defer_workflow_process_with_execution_lease(
                    child,
                    DeferredSignalState::Unresolved,
                    armed.execution_lease.clone(),
                );
            }
            if let Some(stderr_handle) = armed.stderr_handle.take() {
                let _drain = stderr_handle.finish(cleanup_deadline, true);
            }
            return Err(WorkflowProcessFinishError::OwnershipLost);
        }
        let cleanup_result = armed.cleanup_and_finish_drain(cleanup_deadline);
        let timed_out = observation == ExitObservation::DeadlineExceeded
            || self.timed_out.load(Ordering::SeqCst)
            || Instant::now() >= execution_deadline;
        let (status, stderr_result) = match cleanup_result {
            Ok(result) => result,
            Err(error) if timed_out => {
                return Err(WorkflowProcessFinishError::DeadlineExceeded(error));
            }
            Err(error) => return Err(WorkflowProcessFinishError::Cleanup(error)),
        };
        if let Some(stderr_result) = stderr_result {
            match stderr_result {
                Ok(_) => {}
                Err(error) if timed_out => {
                    return Err(WorkflowProcessFinishError::DeadlineExceeded(error));
                }
                Err(error) => return Err(WorkflowProcessFinishError::Stderr(error)),
            }
        }
        Ok(WorkflowProcessCompletion { status, timed_out })
    }

    #[cfg(test)]
    fn pid(&self) -> Option<u32> {
        self.armed
            .as_ref()
            .and_then(|armed| armed.child.as_ref())
            .map(Child::id)
    }
}

impl Drop for WorkflowProcess {
    fn drop(&mut self) {
        if let Some(armed) = self.armed.take() {
            drop(armed);
        }
    }
}

impl ArmedWorkflowProcess {
    fn cancel_watchdog(&mut self) {
        if let Some(watchdog) = self.watchdog.take() {
            watchdog.finish();
        }
    }

    fn cleanup_child(&mut self, cleanup_deadline: Instant) -> io::Result<ExitStatus> {
        self.cancel_watchdog();
        let Some(child) = self.child.take() else {
            return Err(io::Error::other("workflow child was already disarmed"));
        };
        finalize_owned_child(
            child,
            self.signal_authority.clone(),
            cleanup_deadline,
            self.execution_lease.clone(),
        )
    }

    fn cleanup_and_finish_drain(
        mut self,
        cleanup_deadline: Instant,
    ) -> io::Result<(ExitStatus, Option<io::Result<Vec<u8>>>)> {
        let stderr_handle = self.stderr_handle.take();
        cleanup_and_finish_drain_with(
            || self.cleanup_child(cleanup_deadline),
            stderr_handle,
            cleanup_deadline,
            PipeDrainHandle::finish,
        )
    }

    fn abort(mut self) {
        let cleanup_deadline = abort_cleanup_deadline(self.execution_deadline, Instant::now());
        let mut stderr_handle = self.stderr_handle.take();
        if let Some(stderr_handle) = stderr_handle.as_mut() {
            stderr_handle.cancel();
        }
        let _cleanup = self.cleanup_child(cleanup_deadline);
        if let Some(stderr_handle) = stderr_handle {
            let _drain = stderr_handle.finish(cleanup_deadline, false);
        }
    }
}

impl Drop for ArmedWorkflowProcess {
    fn drop(&mut self) {
        let cleanup_deadline = abort_cleanup_deadline(self.execution_deadline, Instant::now());
        let mut stderr_handle = self.stderr_handle.take();
        if let Some(stderr_handle) = stderr_handle.as_mut() {
            stderr_handle.cancel();
        }
        let _cleanup = self.cleanup_child(cleanup_deadline);
        if let Some(stderr_handle) = stderr_handle {
            let _drain = stderr_handle.finish(cleanup_deadline, false);
        }
    }
}

fn abort_cleanup_deadline(execution_deadline: Instant, now: Instant) -> Instant {
    (execution_deadline + PROCESS_REAP_GRACE).min(now + PROCESS_REAP_GRACE)
}

fn capture_process_identity_bounded_with<Capture, DeadlineExpired, Pause>(
    pid: u32,
    mut capture: Capture,
    mut deadline_expired: DeadlineExpired,
    mut pause: Pause,
) -> Result<LinuxProcessIdentity, ProcessGroupSignalError>
where
    Capture: FnMut(u32) -> Result<LinuxProcessIdentity, ProcessGroupSignalError>,
    DeadlineExpired: FnMut() -> bool,
    Pause: FnMut(),
{
    loop {
        match capture(pid) {
            Ok(identity) => return Ok(identity),
            Err(ProcessGroupSignalError::IdentityUnavailable) if !deadline_expired() => pause(),
            Err(error) => return Err(error),
        }
    }
}

fn validate_signal_authority_bounded(
    signal_authority: &SignalAuthority,
    cleanup_deadline: Instant,
) -> Result<(), ProcessGroupSignalError> {
    loop {
        match signal_authority.validate_current() {
            Ok(()) => return Ok(()),
            Err(ProcessGroupSignalError::IdentityUnavailable)
                if Instant::now() < cleanup_deadline =>
            {
                thread::sleep(IDENTITY_RETRY_POLL);
            }
            Err(error) => return Err(error),
        }
    }
}

fn cleanup_permits_spawn(worker_available: bool, pending_cleanups: usize) -> bool {
    worker_available && pending_cleanups == 0
}

fn cleanup_worker_available() -> bool {
    let reaper = shared_deferred_reaper();
    cleanup_permits_spawn(
        reaper.worker_available,
        reaper.pending_cleanups.load(Ordering::SeqCst),
    )
}

fn apply_allowed_environment(command: &mut Command) {
    command.env_clear();
    for key in ALLOWED_ENV_VARS {
        if let Some(value) = env::var_os(key) {
            command.env(key, value);
        }
    }
}

fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn spawn_timeout_watchdog(
    signal_authority: SignalAuthority,
    execution_deadline: Instant,
    timed_out: Arc<AtomicBool>,
) -> io::Result<Watchdog> {
    let (done_sender, done_receiver) = mpsc::channel();
    let handle = thread::Builder::new()
        .name(String::from("llm-guard-workflow-watchdog"))
        .spawn(move || {
            if watchdog_deadline_elapsed_with(
                execution_deadline,
                |remaining| done_receiver.recv_timeout(remaining),
                Instant::now,
            ) {
                timed_out.store(true, Ordering::SeqCst);
                terminate_child_group(&signal_authority);
            }
        })?;
    Ok(Watchdog {
        done_sender: Some(done_sender),
        handle: Some(handle),
        execution_deadline,
    })
}

fn watchdog_deadline_elapsed_with<Wait, Now>(
    execution_deadline: Instant,
    wait: Wait,
    mut now: Now,
) -> bool
where
    Wait: FnOnce(Duration) -> Result<(), mpsc::RecvTimeoutError>,
    Now: FnMut() -> Instant,
{
    let remaining = execution_deadline.saturating_duration_since(now());
    match wait(remaining) {
        Err(mpsc::RecvTimeoutError::Timeout) => true,
        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => now() >= execution_deadline,
    }
}

struct Watchdog {
    done_sender: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<()>>,
    execution_deadline: Instant,
}

impl Watchdog {
    fn finish(mut self) {
        self.cancel_before_deadline();
        self.join();
    }

    fn cancel_before_deadline(&mut self) {
        let Some(done_sender) = self.done_sender.take() else {
            return;
        };
        if Instant::now() < self.execution_deadline {
            let _ignored = done_sender.send(());
        }
    }

    fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ignored = handle.join();
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.cancel_before_deadline();
        self.join();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitIdPoll {
    Exited,
    StillRunning,
    OwnershipLost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExitObservation {
    Exited,
    OwnershipLost,
    DeadlineExceeded,
}

fn observe_child_exit_bounded_with<Poll, DeadlineExpired, Pause>(
    mut poll: Poll,
    mut deadline_expired: DeadlineExpired,
    mut pause: Pause,
) -> io::Result<ExitObservation>
where
    Poll: FnMut() -> io::Result<WaitIdPoll>,
    DeadlineExpired: FnMut() -> bool,
    Pause: FnMut(),
{
    loop {
        match poll() {
            Ok(WaitIdPoll::Exited) => return Ok(ExitObservation::Exited),
            Ok(WaitIdPoll::OwnershipLost) => return Ok(ExitObservation::OwnershipLost),
            Ok(WaitIdPoll::StillRunning) | Err(_) => {
                if deadline_expired() {
                    return Ok(ExitObservation::DeadlineExceeded);
                }
                pause();
            }
        }
    }
}

fn observe_child_exit_without_reaping(
    signal_authority: &SignalAuthority,
    deadline: Instant,
) -> ExitObservation {
    observe_child_exit_bounded_with(
        || match signal_authority.observe_child_nonreaping() {
            Ok(NonReapingChildState::Running) => Ok(WaitIdPoll::StillRunning),
            Ok(NonReapingChildState::Exited) => Ok(WaitIdPoll::Exited),
            Err(
                ProcessGroupSignalError::IdentityUnavailable
                | ProcessGroupSignalError::ObservationUnavailable,
            ) => Err(io::Error::from(io::ErrorKind::WouldBlock)),
            Ok(NonReapingChildState::OwnershipLost) | Err(_) => Ok(WaitIdPoll::OwnershipLost),
        },
        || Instant::now() >= deadline,
        || thread::sleep(PROCESS_REAP_POLL),
    )
    .unwrap_or(ExitObservation::DeadlineExceeded)
}

fn finalize_owned_child(
    child: Child,
    signal_authority: SignalAuthority,
    cleanup_deadline: Instant,
    execution_lease: WorkflowExecutionLease,
) -> io::Result<ExitStatus> {
    match signal_owned_workflow(&signal_authority, nix::sys::signal::Signal::SIGKILL) {
        Ok(WorkflowSignalOutcome::StrictGroup) => reap_child_bounded(
            child,
            cleanup_deadline,
            DeferredSignalState::StrictGroupSignaled,
            execution_lease,
        ),
        Ok(WorkflowSignalOutcome::LeaderOnly) => {
            defer_workflow_process_with_execution_lease(
                child,
                DeferredSignalState::SignalPending(signal_authority),
                execution_lease,
            );
            Err(process_group_signal_io_error(
                ProcessGroupSignalError::SignalFailed,
            ))
        }
        Err(error) => {
            let deferred_state = deferred_state_after_signal_failure(signal_authority);
            defer_workflow_process_with_execution_lease(child, deferred_state, execution_lease);
            Err(process_group_signal_io_error(error))
        }
    }
}

fn deferred_state_after_signal_failure(signal_authority: SignalAuthority) -> DeferredSignalState {
    if signal_authority.is_active() {
        DeferredSignalState::SignalPending(signal_authority)
    } else {
        DeferredSignalState::Unresolved
    }
}

#[derive(Debug)]
enum ReapPollOutcome {
    Reaped(ExitStatus),
    TimedOut,
}

fn reap_child_bounded_with<TryWait, DeadlineExpired, Pause>(
    mut try_wait: TryWait,
    mut deadline_expired: DeadlineExpired,
    mut pause: Pause,
) -> io::Result<ReapPollOutcome>
where
    TryWait: FnMut() -> io::Result<Option<ExitStatus>>,
    DeadlineExpired: FnMut() -> bool,
    Pause: FnMut(),
{
    loop {
        match try_wait() {
            Ok(Some(status)) => return Ok(ReapPollOutcome::Reaped(status)),
            Ok(None) => {
                if deadline_expired() {
                    return Ok(ReapPollOutcome::TimedOut);
                }
                pause();
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                if deadline_expired() {
                    return Ok(ReapPollOutcome::TimedOut);
                }
                pause();
            }
            Err(error) => return Err(error),
        }
    }
}

fn reap_child_bounded(
    mut child: Child,
    cleanup_deadline: Instant,
    deferred_state: DeferredSignalState,
    execution_lease: WorkflowExecutionLease,
) -> io::Result<ExitStatus> {
    let outcome = reap_child_bounded_with(
        || child.try_wait(),
        || Instant::now() >= cleanup_deadline,
        || thread::sleep(PROCESS_REAP_POLL),
    );
    match outcome {
        Ok(ReapPollOutcome::Reaped(status)) => Ok(status),
        Ok(ReapPollOutcome::TimedOut) => {
            defer_workflow_process_with_execution_lease(child, deferred_state, execution_lease);
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "workflow child reap exceeded cleanup deadline",
            ))
        }
        Err(error) => {
            defer_workflow_process_with_execution_lease(child, deferred_state, execution_lease);
            Err(error)
        }
    }
}

fn process_group_signal_io_error(error: ProcessGroupSignalError) -> io::Error {
    io::Error::other(error.as_str())
}

fn terminate_child_group(signal_authority: &SignalAuthority) {
    let _signal = signal_owned_workflow(signal_authority, nix::sys::signal::Signal::SIGKILL);
}

#[cfg(test)]
#[path = "workflow_process/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "workflow_process/lifecycle_tests.rs"]
mod lifecycle_tests;

#[cfg(test)]
#[path = "workflow_process/startup_authority_tests.rs"]
mod startup_authority_tests;

#[cfg(test)]
#[path = "workflow_process/moved_pgid_tests.rs"]
mod moved_pgid_tests;

#[cfg(test)]
#[path = "workflow_process/signal_authority_tests.rs"]
mod signal_authority_tests;

#[cfg(test)]
#[path = "workflow_process/isolated_reaper_tests.rs"]
mod isolated_reaper_tests;

#[path = "workflow_process/pipe_io.rs"]
mod pipe_io;

#[path = "workflow_process/drain.rs"]
mod drain;

#[path = "workflow_process/startup_guard.rs"]
mod startup_guard;

#[path = "workflow_process/deferred_reaper.rs"]
mod deferred_reaper;

#[path = "workflow_process/signal_authority.rs"]
mod signal_authority;

#[cfg(test)]
#[path = "workflow_process/test_raii.rs"]
mod test_raii;

#[cfg(test)]
#[path = "workflow_process/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "workflow_process/execution_lease_tests.rs"]
mod execution_lease_tests;
