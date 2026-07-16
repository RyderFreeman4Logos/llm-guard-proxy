use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use nix::errno::Errno;

#[cfg(target_os = "linux")]
use crate::workflow_cgroup::WorkflowCgroup;
use crate::workflow_execution::WorkflowExecutionLease;

use super::{
    NonReapingChildState, PROCESS_REAP_POLL, ProcessGroupSignalError, SignalAuthority,
    WorkflowChild, WorkflowSignalOutcome, signal_authority::ProvisionalGroupAuthority,
    signal_owned_workflow,
};

const MAX_SIGNAL_RETRY_BACKOFF: Duration = Duration::from_secs(1);
const IDLE_REAPER_POLL: Duration = Duration::from_millis(50);
#[cfg(target_os = "linux")]
const CGROUP_CLEANUP_ATTEMPT_GRACE: Duration = Duration::from_millis(500);
#[cfg(target_os = "linux")]
pub(super) const MAX_CGROUP_CLEANUP_RETRIES: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeferredPollOutcome {
    Pending,
    Complete,
}

pub(super) enum DeferredSignalState {
    SignalPending(SignalAuthority),
    ProvisionalSignalPending(ProvisionalGroupAuthority),
    StrictGroupSignaled,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingSignalOutcome {
    StrictGroupSignaled,
    Retry,
    Unresolved,
}

fn poll_pending_signal_with<Authority, Observe, SignalGroup>(
    authority: &mut Authority,
    observe: Observe,
    signal_group: SignalGroup,
) -> PendingSignalOutcome
where
    Observe: FnOnce(&mut Authority) -> Result<NonReapingChildState, ProcessGroupSignalError>,
    SignalGroup: FnOnce(&mut Authority) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
{
    match observe(authority) {
        Ok(NonReapingChildState::Running | NonReapingChildState::Exited) => {}
        Ok(NonReapingChildState::OwnershipLost) => return PendingSignalOutcome::Unresolved,
        Err(error) => return pending_outcome_after_error(error),
    }
    match signal_group(authority) {
        Ok(WorkflowSignalOutcome::StrictGroup) => PendingSignalOutcome::StrictGroupSignaled,
        Ok(WorkflowSignalOutcome::LeaderOnly) => PendingSignalOutcome::Retry,
        Err(error) => pending_outcome_after_error(error),
    }
}

const fn pending_outcome_after_error(error: ProcessGroupSignalError) -> PendingSignalOutcome {
    match error {
        ProcessGroupSignalError::IdentityUnavailable
        | ProcessGroupSignalError::ObservationUnavailable
        | ProcessGroupSignalError::SignalFailed => PendingSignalOutcome::Retry,
        ProcessGroupSignalError::InvalidPid
        | ProcessGroupSignalError::IdentityMismatch
        | ProcessGroupSignalError::OwnershipLost => PendingSignalOutcome::Unresolved,
    }
}

#[cfg(test)]
pub(super) fn poll_deferred_cleanup_with<Observe, SignalGroup, PollReap>(
    signal_state: &mut DeferredSignalState,
    observe: Observe,
    signal_group: SignalGroup,
    poll_reap: PollReap,
) -> DeferredPollOutcome
where
    Observe: FnOnce() -> Result<NonReapingChildState, ProcessGroupSignalError>,
    SignalGroup: FnOnce() -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
    PollReap: FnOnce() -> std::io::Result<bool>,
{
    if matches!(signal_state, DeferredSignalState::Unresolved) {
        return DeferredPollOutcome::Pending;
    }

    if matches!(
        signal_state,
        DeferredSignalState::SignalPending(_) | DeferredSignalState::ProvisionalSignalPending(_)
    ) {
        let signal_outcome =
            poll_pending_signal_with(&mut (), |_authority| observe(), |_authority| signal_group());
        match signal_outcome {
            PendingSignalOutcome::StrictGroupSignaled => {
                *signal_state = DeferredSignalState::StrictGroupSignaled;
            }
            PendingSignalOutcome::Retry => return DeferredPollOutcome::Pending,
            PendingSignalOutcome::Unresolved => {
                *signal_state = DeferredSignalState::Unresolved;
                return DeferredPollOutcome::Pending;
            }
        }
    }

    match poll_reap() {
        Ok(true) => DeferredPollOutcome::Complete,
        Err(error) if error.raw_os_error() == Some(Errno::ECHILD as i32) => {
            *signal_state = DeferredSignalState::Unresolved;
            DeferredPollOutcome::Pending
        }
        Ok(false) | Err(_) => DeferredPollOutcome::Pending,
    }
}

pub(super) struct DeferredWorkflowProcess {
    pub(super) child: WorkflowChild,
    pub(super) signal_state: DeferredSignalState,
    pub(super) next_signal_attempt: Instant,
    pub(super) signal_retry_backoff: Duration,
    _execution_lease: WorkflowExecutionLease,
}

impl DeferredWorkflowProcess {
    #[cfg(test)]
    pub(super) fn new<ChildHandle>(child: ChildHandle, signal_state: DeferredSignalState) -> Self
    where
        ChildHandle: Into<WorkflowChild>,
    {
        Self::new_with_execution_lease(child, signal_state, WorkflowExecutionLease::default())
    }

    pub(super) fn new_with_execution_lease<ChildHandle>(
        child: ChildHandle,
        signal_state: DeferredSignalState,
        execution_lease: WorkflowExecutionLease,
    ) -> Self
    where
        ChildHandle: Into<WorkflowChild>,
    {
        let mut child = child.into();
        child.stdin.take();
        child.stdout.take();
        child.stderr.take();
        Self {
            child,
            signal_state,
            next_signal_attempt: Instant::now() + PROCESS_REAP_POLL,
            signal_retry_backoff: PROCESS_REAP_POLL,
            _execution_lease: execution_lease,
        }
    }

    fn close_child_stdio(&mut self) {
        self.child.stdin.take();
        self.child.stdout.take();
        self.child.stderr.take();
    }

    fn poll(&mut self) -> DeferredPollOutcome {
        let now = Instant::now();
        if now < self.next_signal_attempt {
            return DeferredPollOutcome::Pending;
        }
        if matches!(self.signal_state, DeferredSignalState::Unresolved) {
            self.next_signal_attempt = now + MAX_SIGNAL_RETRY_BACKOFF;
            return DeferredPollOutcome::Pending;
        }
        let signal_state =
            std::mem::replace(&mut self.signal_state, DeferredSignalState::Unresolved);
        self.signal_state = match signal_state {
            DeferredSignalState::SignalPending(mut authority) => {
                match poll_pending_signal_with(
                    &mut authority,
                    |authority| authority.observe_child_nonreaping(),
                    |authority| signal_owned_workflow(authority, nix::sys::signal::Signal::SIGKILL),
                ) {
                    PendingSignalOutcome::StrictGroupSignaled => {
                        DeferredSignalState::StrictGroupSignaled
                    }
                    PendingSignalOutcome::Retry => DeferredSignalState::SignalPending(authority),
                    PendingSignalOutcome::Unresolved => DeferredSignalState::Unresolved,
                }
            }
            DeferredSignalState::ProvisionalSignalPending(mut authority) => {
                match poll_pending_signal_with(
                    &mut authority,
                    ProvisionalGroupAuthority::observe_child_nonreaping,
                    |authority| authority.signal_owned_workflow(nix::sys::signal::Signal::SIGKILL),
                ) {
                    PendingSignalOutcome::StrictGroupSignaled => {
                        DeferredSignalState::StrictGroupSignaled
                    }
                    PendingSignalOutcome::Retry => {
                        DeferredSignalState::ProvisionalSignalPending(authority)
                    }
                    PendingSignalOutcome::Unresolved => DeferredSignalState::Unresolved,
                }
            }
            state => state,
        };
        let outcome = if matches!(self.signal_state, DeferredSignalState::StrictGroupSignaled) {
            match self.child.try_wait() {
                Ok(Some(_)) => DeferredPollOutcome::Complete,
                Err(error) if error.raw_os_error() == Some(Errno::ECHILD as i32) => {
                    self.signal_state = DeferredSignalState::Unresolved;
                    DeferredPollOutcome::Pending
                }
                Ok(None) | Err(_) => DeferredPollOutcome::Pending,
            }
        } else {
            DeferredPollOutcome::Pending
        };
        if outcome == DeferredPollOutcome::Pending {
            self.signal_retry_backoff = grow_signal_retry_backoff(self.signal_retry_backoff);
            self.next_signal_attempt = now + self.signal_retry_backoff;
        }
        outcome
    }
}

#[cfg(target_os = "linux")]
pub(super) struct DeferredWorkflowCgroup {
    cgroup: Arc<WorkflowCgroup>,
    failed_cleanup_attempts: usize,
    next_cleanup_attempt: Instant,
    cleanup_retry_backoff: Duration,
    _execution_lease: WorkflowExecutionLease,
}

#[cfg(target_os = "linux")]
impl DeferredWorkflowCgroup {
    pub(super) fn new(
        cgroup: Arc<WorkflowCgroup>,
        execution_lease: WorkflowExecutionLease,
    ) -> Self {
        Self {
            cgroup,
            failed_cleanup_attempts: 0,
            next_cleanup_attempt: Instant::now() + PROCESS_REAP_POLL,
            cleanup_retry_backoff: PROCESS_REAP_POLL,
            _execution_lease: execution_lease,
        }
    }

    fn poll(&mut self) -> DeferredPollOutcome {
        let now = Instant::now();
        let cgroup = Arc::clone(&self.cgroup);
        self.poll_with(
            now,
            || cgroup.kill_and_remove(Instant::now() + CGROUP_CLEANUP_ATTEMPT_GRACE),
            || cgroup.force_kill_and_remove(Instant::now() + CGROUP_CLEANUP_ATTEMPT_GRACE),
        )
    }

    pub(super) fn poll_with<Cleanup, ForceCleanup>(
        &mut self,
        now: Instant,
        cleanup: Cleanup,
        force_cleanup: ForceCleanup,
    ) -> DeferredPollOutcome
    where
        Cleanup: FnOnce() -> std::io::Result<()>,
        ForceCleanup: FnOnce() -> std::io::Result<()>,
    {
        if now < self.next_cleanup_attempt {
            return DeferredPollOutcome::Pending;
        }
        let forced = self.failed_cleanup_attempts >= MAX_CGROUP_CLEANUP_RETRIES;
        let result = if forced { force_cleanup() } else { cleanup() };
        if result.is_ok() {
            return DeferredPollOutcome::Complete;
        }
        if !forced {
            self.failed_cleanup_attempts += 1;
            if self.failed_cleanup_attempts == MAX_CGROUP_CLEANUP_RETRIES {
                eprintln!(
                    "workflow cgroup cleanup exhausted {MAX_CGROUP_CLEANUP_RETRIES} retries; escalating to per-process SIGKILL cleanup"
                );
            }
        }
        self.cleanup_retry_backoff = grow_signal_retry_backoff(self.cleanup_retry_backoff);
        self.next_cleanup_attempt = now + self.cleanup_retry_backoff;
        DeferredPollOutcome::Pending
    }
}

pub(super) fn grow_signal_retry_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(MAX_SIGNAL_RETRY_BACKOFF)
}

pub(super) fn poll_deferred_processes(
    processes: &mut Vec<DeferredWorkflowProcess>,
    pending_cleanups: &AtomicUsize,
) {
    processes.retain_mut(|process| {
        let outcome = process.poll();
        if outcome == DeferredPollOutcome::Complete {
            pending_cleanups.fetch_sub(1, Ordering::SeqCst);
        }
        outcome == DeferredPollOutcome::Pending
    });
}

#[cfg(target_os = "linux")]
pub(super) fn poll_deferred_cgroups_with<Poll>(
    cgroups: &mut Vec<DeferredWorkflowCgroup>,
    pending_cleanups: &AtomicUsize,
    mut poll: Poll,
) where
    Poll: FnMut(&mut DeferredWorkflowCgroup) -> DeferredPollOutcome,
{
    cgroups.retain_mut(|cgroup| {
        let outcome = poll(cgroup);
        if outcome == DeferredPollOutcome::Complete {
            pending_cleanups.fetch_sub(1, Ordering::SeqCst);
        }
        outcome == DeferredPollOutcome::Pending
    });
}

#[cfg(target_os = "linux")]
fn poll_deferred_cgroups(
    cgroups: &mut Vec<DeferredWorkflowCgroup>,
    pending_cleanups: &AtomicUsize,
) {
    poll_deferred_cgroups_with(cgroups, pending_cleanups, DeferredWorkflowCgroup::poll);
}

pub(super) fn next_deferred_poll_delay(
    processes: &[DeferredWorkflowProcess],
    now: Instant,
) -> Duration {
    processes
        .iter()
        .map(|process| process.next_signal_attempt.saturating_duration_since(now))
        .min()
        .map_or(IDLE_REAPER_POLL, |delay| delay.max(IDLE_REAPER_POLL))
}

#[cfg(target_os = "linux")]
fn next_deferred_cgroup_poll_delay(cgroups: &[DeferredWorkflowCgroup], now: Instant) -> Duration {
    cgroups
        .iter()
        .map(|cgroup| cgroup.next_cleanup_attempt.saturating_duration_since(now))
        .min()
        .map_or(IDLE_REAPER_POLL, |delay| delay.max(IDLE_REAPER_POLL))
}

pub(super) struct SharedDeferredReaper {
    pub(super) processes: Arc<Mutex<Vec<DeferredWorkflowProcess>>>,
    #[cfg(target_os = "linux")]
    pub(super) cgroups: Arc<Mutex<Vec<DeferredWorkflowCgroup>>>,
    pub(super) pending_cleanups: Arc<AtomicUsize>,
    pub(super) worker_available: bool,
}

impl SharedDeferredReaper {
    pub(super) fn start() -> Self {
        let processes = Arc::new(Mutex::new(Vec::<DeferredWorkflowProcess>::new()));
        let worker_processes = Arc::clone(&processes);
        #[cfg(target_os = "linux")]
        let cgroups = Arc::new(Mutex::new(Vec::<DeferredWorkflowCgroup>::new()));
        #[cfg(target_os = "linux")]
        let worker_cgroups = Arc::clone(&cgroups);
        let pending_cleanups = Arc::new(AtomicUsize::new(0));
        let worker_pending_cleanups = Arc::clone(&pending_cleanups);
        let worker_available = thread::Builder::new()
            .name(String::from("llm-guard-workflow-reaper"))
            .spawn(move || {
                loop {
                    let now = Instant::now();
                    let process_sleep = {
                        let mut processes = worker_processes
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        poll_deferred_processes(&mut processes, &worker_pending_cleanups);
                        next_deferred_poll_delay(&processes, now)
                    };
                    #[cfg(target_os = "linux")]
                    let sleep_for = {
                        let mut cgroups = worker_cgroups
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        poll_deferred_cgroups(&mut cgroups, &worker_pending_cleanups);
                        process_sleep.min(next_deferred_cgroup_poll_delay(&cgroups, now))
                    };
                    #[cfg(not(target_os = "linux"))]
                    let sleep_for = process_sleep;
                    thread::sleep(sleep_for);
                }
            })
            .is_ok();
        Self {
            processes,
            #[cfg(target_os = "linux")]
            cgroups,
            pending_cleanups,
            worker_available,
        }
    }

    pub(super) fn submit(&self, mut process: DeferredWorkflowProcess) {
        process.close_child_stdio();
        self.pending_cleanups.fetch_add(1, Ordering::SeqCst);
        self.processes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(process);
    }

    #[cfg(target_os = "linux")]
    pub(super) fn submit_cgroup(&self, cgroup: DeferredWorkflowCgroup) {
        self.pending_cleanups.fetch_add(1, Ordering::SeqCst);
        self.cgroups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(cgroup);
    }
}

#[cfg(test)]
pub(super) fn defer_workflow_process<ChildHandle>(
    child: ChildHandle,
    signal_state: DeferredSignalState,
) where
    ChildHandle: Into<WorkflowChild>,
{
    defer_workflow_process_with_execution_lease(
        child.into(),
        signal_state,
        WorkflowExecutionLease::default(),
    );
}

pub(super) fn defer_workflow_process_with_execution_lease(
    child: WorkflowChild,
    signal_state: DeferredSignalState,
    execution_lease: WorkflowExecutionLease,
) {
    shared_deferred_reaper().submit(DeferredWorkflowProcess::new_with_execution_lease(
        child,
        signal_state,
        execution_lease,
    ));
}

#[cfg(target_os = "linux")]
pub(super) fn defer_workflow_cgroup_with_execution_lease(
    cgroup: Arc<WorkflowCgroup>,
    execution_lease: WorkflowExecutionLease,
) {
    shared_deferred_reaper().submit_cgroup(DeferredWorkflowCgroup::new(cgroup, execution_lease));
}

pub(super) fn shared_deferred_reaper() -> &'static SharedDeferredReaper {
    static DEFERRED_REAPER: std::sync::OnceLock<SharedDeferredReaper> = std::sync::OnceLock::new();
    DEFERRED_REAPER.get_or_init(SharedDeferredReaper::start)
}
