use std::{
    process::Child,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use nix::errno::Errno;

use crate::workflow_execution::WorkflowExecutionLease;

use super::{
    NonReapingChildState, PROCESS_REAP_POLL, ProcessGroupSignalError, SignalAuthority,
    WorkflowSignalOutcome, signal_authority::ProvisionalGroupAuthority, signal_owned_workflow,
};

const MAX_SIGNAL_RETRY_BACKOFF: Duration = Duration::from_secs(1);
const IDLE_REAPER_POLL: Duration = Duration::from_millis(50);

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
    pub(super) child: Child,
    pub(super) signal_state: DeferredSignalState,
    pub(super) next_signal_attempt: Instant,
    pub(super) signal_retry_backoff: Duration,
    _execution_lease: WorkflowExecutionLease,
}

impl DeferredWorkflowProcess {
    #[cfg(test)]
    pub(super) fn new(child: Child, signal_state: DeferredSignalState) -> Self {
        Self::new_with_execution_lease(child, signal_state, WorkflowExecutionLease::default())
    }

    pub(super) fn new_with_execution_lease(
        mut child: Child,
        signal_state: DeferredSignalState,
        execution_lease: WorkflowExecutionLease,
    ) -> Self {
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

pub(super) struct SharedDeferredReaper {
    pub(super) processes: Arc<Mutex<Vec<DeferredWorkflowProcess>>>,
    pub(super) pending_cleanups: Arc<AtomicUsize>,
    pub(super) worker_available: bool,
}

impl SharedDeferredReaper {
    pub(super) fn start() -> Self {
        let processes = Arc::new(Mutex::new(Vec::<DeferredWorkflowProcess>::new()));
        let worker_processes = Arc::clone(&processes);
        let pending_cleanups = Arc::new(AtomicUsize::new(0));
        let worker_pending_cleanups = Arc::clone(&pending_cleanups);
        let worker_available = thread::Builder::new()
            .name(String::from("llm-guard-workflow-reaper"))
            .spawn(move || {
                loop {
                    let sleep_for = {
                        let mut processes = worker_processes
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        poll_deferred_processes(&mut processes, &worker_pending_cleanups);
                        next_deferred_poll_delay(&processes, Instant::now())
                    };
                    thread::sleep(sleep_for);
                }
            })
            .is_ok();
        Self {
            processes,
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
}

#[cfg(test)]
pub(super) fn defer_workflow_process(child: Child, signal_state: DeferredSignalState) {
    defer_workflow_process_with_execution_lease(
        child,
        signal_state,
        WorkflowExecutionLease::default(),
    );
}

pub(super) fn defer_workflow_process_with_execution_lease(
    child: Child,
    signal_state: DeferredSignalState,
    execution_lease: WorkflowExecutionLease,
) {
    shared_deferred_reaper().submit(DeferredWorkflowProcess::new_with_execution_lease(
        child,
        signal_state,
        execution_lease,
    ));
}

pub(super) fn shared_deferred_reaper() -> &'static SharedDeferredReaper {
    static DEFERRED_REAPER: std::sync::OnceLock<SharedDeferredReaper> = std::sync::OnceLock::new();
    DEFERRED_REAPER.get_or_init(SharedDeferredReaper::start)
}
