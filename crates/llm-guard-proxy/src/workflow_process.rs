//! Owned workflow subprocess lifecycle and bounded cleanup.
//!
//! The runtime always owns and terminates the process group rooted at the spawned leader. On Linux
//! it also attempts to place each execution in a delegated cgroup v2 subtree, allowing descendants
//! that call `setsid(2)` or `setpgid(2)` to be killed authoritatively. When delegation is unavailable,
//! deadline-driven local pipe I/O and process-group cleanup preserve the previous bounded behavior.
//!
//! This module must be the sole waiter for its private workflow children. Embedders must not use
//! process-global `SIGCHLD = SIG_IGN` or `SA_NOCLDWAIT`, and must not run a competing waiter for
//! workflow PIDs. Unexpected `ECHILD` revokes signal authority and leaves admission fail-closed.

use std::{
    env,
    ffi::OsString,
    fs::File,
    io::{self},
    os::fd::OwnedFd,
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

#[cfg(target_os = "linux")]
use crate::workflow_cgroup::{
    AtomicSpawnOutcome, AtomicWorkflowChild, WorkflowCgroup, log_atomic_spawn_fallback_once,
};
use crate::workflow_execution::WorkflowExecutionLease;

#[cfg(target_os = "linux")]
use self::deferred_reaper::defer_workflow_cgroup_with_execution_lease;
#[cfg(test)]
use self::deferred_reaper::{
    DeferredPollOutcome, DeferredWorkflowProcess, SharedDeferredReaper, defer_workflow_process,
    grow_signal_retry_backoff, next_deferred_poll_delay, poll_deferred_cleanup_with,
    poll_deferred_processes,
};
#[cfg(all(test, target_os = "linux"))]
use self::deferred_reaper::{
    DeferredWorkflowCgroup, MAX_CGROUP_CLEANUP_RETRIES, poll_deferred_cgroups_with,
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

pub(crate) struct WorkflowChild {
    inner: WorkflowChildInner,
    stdin: Option<File>,
    stdout: Option<File>,
    stderr: Option<File>,
}

enum WorkflowChildInner {
    Standard(Child),
    #[cfg(target_os = "linux")]
    Atomic(AtomicWorkflowChild),
}

impl WorkflowChild {
    fn id(&self) -> u32 {
        match &self.inner {
            WorkflowChildInner::Standard(child) => child.id(),
            #[cfg(target_os = "linux")]
            WorkflowChildInner::Atomic(child) => child.id(),
        }
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        match &mut self.inner {
            WorkflowChildInner::Standard(child) => child.try_wait(),
            #[cfg(target_os = "linux")]
            WorkflowChildInner::Atomic(child) => child.try_wait(),
        }
    }
}

impl From<Child> for WorkflowChild {
    fn from(mut child: Child) -> Self {
        let stdin = child
            .stdin
            .take()
            .map(|pipe| File::from(OwnedFd::from(pipe)));
        let stdout = child
            .stdout
            .take()
            .map(|pipe| File::from(OwnedFd::from(pipe)));
        let stderr = child
            .stderr
            .take()
            .map(|pipe| File::from(OwnedFd::from(pipe)));
        Self {
            inner: WorkflowChildInner::Standard(child),
            stdin,
            stdout,
            stderr,
        }
    }
}

#[cfg(target_os = "linux")]
impl From<AtomicWorkflowChild> for WorkflowChild {
    fn from(mut child: AtomicWorkflowChild) -> Self {
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        Self {
            inner: WorkflowChildInner::Atomic(child),
            stdin,
            stdout,
            stderr,
        }
    }
}

/// One armed workflow subprocess owner.
///
/// Dropping an armed value synchronously performs bounded cleanup or transfers the unique child
/// handle to the shared deferred reaper. Normal completion and abort disarm exactly once.
pub(super) struct WorkflowProcess {
    armed: Option<ArmedWorkflowProcess>,
    timed_out: Arc<AtomicBool>,
}

struct ArmedWorkflowProcess {
    child: Option<WorkflowChild>,
    #[cfg(target_os = "linux")]
    workflow_cgroup: Option<Arc<WorkflowCgroup>>,
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
    #[cfg(target_os = "linux")]
    Cgroup(io::Error),
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
        #[cfg(target_os = "linux")]
        let workflow_cgroup =
            WorkflowCgroup::prepare().map_err(WorkflowProcessStartError::Cgroup)?;
        let environment = allowed_environment();
        let mut command = workflow_command(config, &environment);

        let startup_cleanup_deadline = abort_cleanup_deadline(execution_deadline, Instant::now());
        #[cfg(target_os = "linux")]
        let (child, atomically_placed) = spawn_workflow_child(
            &mut command,
            config,
            &environment,
            workflow_cgroup.as_ref(),
            startup_cleanup_deadline,
            &execution_lease,
        )?;
        #[cfg(not(target_os = "linux"))]
        let child = WorkflowChild::from(command.spawn().map_err(WorkflowProcessStartError::Spawn)?);
        let mut spawned = SpawnedChildGuard::new_with_execution_lease(
            child,
            startup_cleanup_deadline,
            execution_lease,
        );
        #[cfg(target_os = "linux")]
        if let Some(workflow_cgroup) = workflow_cgroup {
            spawned.set_workflow_cgroup(Arc::clone(&workflow_cgroup));
            if !atomically_placed
                && !workflow_cgroup
                    .attach_or_fallback(spawned.pid(), startup_cleanup_deadline)
                    .map_err(WorkflowProcessStartError::Cgroup)?
            {
                spawned.clear_workflow_cgroup();
            }
        }
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
            #[cfg(target_os = "linux")]
            spawned.workflow_cgroup(),
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
        #[cfg(target_os = "linux")]
        let workflow_cgroup = spawned.take_workflow_cgroup();
        let Some((child, execution_lease)) = spawned.disarm_with_execution_lease() else {
            return Err(WorkflowProcessStartError::IdentityChanged(
                "spawned_child_missing",
            ));
        };

        Ok(Self {
            armed: Some(ArmedWorkflowProcess {
                child: Some(child),
                #[cfg(target_os = "linux")]
                workflow_cgroup,
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
            #[cfg(target_os = "linux")]
            let _cgroup_cleanup = armed.cleanup_workflow_cgroup(cleanup_deadline);
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
            .map(WorkflowChild::id)
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
        #[cfg(target_os = "linux")]
        let cgroup_kill_result = self
            .workflow_cgroup
            .as_ref()
            .map_or(Ok(()), |workflow_cgroup| workflow_cgroup.kill());
        let Some(child) = self.child.take() else {
            return Err(io::Error::other("workflow child was already disarmed"));
        };
        let process_group_result = finalize_owned_child(
            child,
            self.signal_authority.clone(),
            cleanup_deadline,
            self.execution_lease.clone(),
        );
        #[cfg(target_os = "linux")]
        {
            let cgroup_result = self.finish_workflow_cgroup(cgroup_kill_result, cleanup_deadline);
            combine_containment_cleanup(cgroup_result, process_group_result)
        }
        #[cfg(not(target_os = "linux"))]
        process_group_result
    }

    #[cfg(target_os = "linux")]
    fn cleanup_workflow_cgroup(&mut self, cleanup_deadline: Instant) -> io::Result<()> {
        let Some(workflow_cgroup) = self.workflow_cgroup.as_ref() else {
            return Ok(());
        };
        let result = workflow_cgroup.kill_and_remove(cleanup_deadline);
        if result.is_ok() {
            self.workflow_cgroup.take();
        } else {
            self.transfer_workflow_cgroup_cleanup();
        }
        result
    }

    #[cfg(target_os = "linux")]
    fn finish_workflow_cgroup(
        &mut self,
        kill_result: io::Result<()>,
        cleanup_deadline: Instant,
    ) -> io::Result<()> {
        let finish_result = self.workflow_cgroup.as_ref().map_or(Ok(()), |cgroup| {
            cgroup.verify_empty_and_remove(cleanup_deadline)
        });
        let result = combine_cgroup_cleanup(kill_result, finish_result);
        if result.is_ok() {
            self.workflow_cgroup.take();
        } else {
            self.transfer_workflow_cgroup_cleanup();
        }
        result
    }

    #[cfg(target_os = "linux")]
    fn transfer_workflow_cgroup_cleanup(&mut self) {
        transfer_workflow_cgroup_cleanup_with(
            &mut self.workflow_cgroup,
            self.execution_lease.clone(),
            defer_workflow_cgroup_with_execution_lease,
        );
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

fn allowed_environment() -> Vec<(OsString, OsString)> {
    ALLOWED_ENV_VARS
        .into_iter()
        .filter_map(|key| env::var_os(key).map(|value| (OsString::from(key), value)))
        .collect()
}

fn apply_allowed_environment(command: &mut Command, environment: &[(OsString, OsString)]) {
    command.env_clear();
    for (key, value) in environment {
        command.env(key, value);
    }
}

fn workflow_command(config: &WorkflowConfig, environment: &[(OsString, OsString)]) -> Command {
    let mut command = Command::new(&config.command);
    command
        .args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_allowed_environment(&mut command, environment);
    configure_process_group(&mut command);
    command
}

#[cfg(target_os = "linux")]
fn spawn_workflow_child(
    command: &mut Command,
    config: &WorkflowConfig,
    environment: &[(OsString, OsString)],
    workflow_cgroup: Option<&Arc<WorkflowCgroup>>,
    cleanup_deadline: Instant,
    execution_lease: &WorkflowExecutionLease,
) -> Result<(WorkflowChild, bool), WorkflowProcessStartError> {
    let Some(workflow_cgroup) = workflow_cgroup else {
        let child = command.spawn().map_err(WorkflowProcessStartError::Spawn)?;
        return Ok((WorkflowChild::from(child), false));
    };
    let arguments = config.args.iter().map(OsString::from).collect::<Vec<_>>();
    let spawn_result =
        match workflow_cgroup.spawn_atomic(config.command.as_ref(), &arguments, environment) {
            Ok(AtomicSpawnOutcome::Spawned(child)) => {
                return Ok((WorkflowChild::from(child), true));
            }
            Ok(AtomicSpawnOutcome::Fallback(error)) => {
                log_atomic_spawn_fallback_once(&error);
                command
                    .spawn()
                    .map(|child| (WorkflowChild::from(child), false))
            }
            Err(error) => Err(error),
        };
    match spawn_result {
        Ok(child) => Ok(child),
        Err(error) => {
            if workflow_cgroup.kill_and_remove(cleanup_deadline).is_err() {
                defer_workflow_cgroup_with_execution_lease(
                    Arc::clone(workflow_cgroup),
                    execution_lease.clone(),
                );
            }
            Err(WorkflowProcessStartError::Spawn(error))
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
    #[cfg(target_os = "linux")] workflow_cgroup: Option<Arc<WorkflowCgroup>>,
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
                #[cfg(target_os = "linux")]
                if let Some(workflow_cgroup) = workflow_cgroup {
                    let _cgroup_kill = workflow_cgroup.kill();
                }
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
    child: WorkflowChild,
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
    mut child: WorkflowChild,
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

#[cfg(target_os = "linux")]
fn combine_containment_cleanup(
    cgroup_result: io::Result<()>,
    process_group_result: io::Result<ExitStatus>,
) -> io::Result<ExitStatus> {
    match (cgroup_result, process_group_result) {
        (Ok(()), Ok(status)) => Ok(status),
        (Err(error), Ok(_)) | (Ok(()), Err(error)) => Err(error),
        (Err(cgroup_error), Err(process_group_error)) => Err(io::Error::other(format!(
            "workflow cgroup cleanup failed: {cgroup_error}; process-group cleanup failed: {process_group_error}"
        ))),
    }
}

#[cfg(target_os = "linux")]
fn transfer_workflow_cgroup_cleanup_with<Defer>(
    workflow_cgroup: &mut Option<Arc<WorkflowCgroup>>,
    execution_lease: WorkflowExecutionLease,
    defer: Defer,
) where
    Defer: FnOnce(Arc<WorkflowCgroup>, WorkflowExecutionLease),
{
    if let Some(workflow_cgroup) = workflow_cgroup.take() {
        defer(workflow_cgroup, execution_lease);
    }
}

#[cfg(target_os = "linux")]
fn combine_cgroup_cleanup(kill: io::Result<()>, finish: io::Result<()>) -> io::Result<()> {
    match (kill, finish) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(kill_error), Err(finish_error)) => Err(io::Error::other(format!(
            "workflow cgroup kill failed: {kill_error}; cleanup failed: {finish_error}"
        ))),
    }
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
