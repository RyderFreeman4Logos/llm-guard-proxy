//! Serialized PID/start-time signal and non-reaping wait authority.
//!
//! This coordinator assumes this module is the sole waiter for each private
//! workflow child. Embedders must not install process-global `SIGCHLD = SIG_IGN`
//! or `SA_NOCLDWAIT`, and must not run a competing `wait`/`waitpid` for workflow
//! PIDs. An unexpected `ECHILD` revokes authority and keeps admission fail-closed.

use std::{
    fs,
    sync::{Arc, Mutex},
};

use nix::{
    errno::Errno,
    sys::{
        signal::Signal,
        wait::{Id, WaitPidFlag, WaitStatus, waitid},
    },
    unistd::Pid,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LinuxProcessIdentity {
    pub(super) pid: u32,
    pub(super) start_time_ticks: u64,
}

impl LinuxProcessIdentity {
    pub(super) fn capture(pid: u32) -> Result<Self, ProcessGroupSignalError> {
        if pid == 0 {
            return Err(ProcessGroupSignalError::InvalidPid);
        }
        let start_time_ticks = linux_process_start_time(pid)?;
        if start_time_ticks == 0 {
            return Err(ProcessGroupSignalError::IdentityUnavailable);
        }
        Ok(Self {
            pid,
            start_time_ticks,
        })
    }

    fn validate_current(self) -> Result<(), ProcessGroupSignalError> {
        if linux_process_start_time(self.pid)? != self.start_time_ticks {
            return Err(ProcessGroupSignalError::IdentityMismatch);
        }
        Ok(())
    }
}

/// Startup-only authority anchored by the unique, unreaped direct child.
///
/// Before `/proc` identity capture succeeds, retaining sole wait ownership of the
/// private process-group leader prevents its PID and PGID from being reused. This
/// non-cloneable coordinator keeps observation, ownership loss, and strict group
/// signalling serialized until it is upgraded or transferred with the `Child`.
pub(super) struct ProvisionalGroupAuthority {
    pid: u32,
    active: bool,
}

impl ProvisionalGroupAuthority {
    pub(super) const fn new(pid: u32) -> Self {
        Self { pid, active: true }
    }

    pub(super) const fn is_active(&self) -> bool {
        self.active
    }

    pub(super) fn revoke(&mut self) {
        self.active = false;
    }

    pub(super) fn observe_child_nonreaping(
        &mut self,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError> {
        self.observe_child_nonreaping_with(|pid| {
            let flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOWAIT | WaitPidFlag::WNOHANG;
            waitid(Id::Pid(pid), flags).map(wait_status_to_nonreaping_state)
        })
    }

    pub(super) fn observe_child_nonreaping_with<Observe>(
        &mut self,
        observe: Observe,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError>
    where
        Observe: FnOnce(Pid) -> Result<NonReapingChildState, Errno>,
    {
        if !self.active {
            return Ok(NonReapingChildState::OwnershipLost);
        }
        let pid = match process_pid_u32(self.pid) {
            Ok(pid) => Pid::from_raw(pid),
            Err(error) => {
                self.active = false;
                return Err(error);
            }
        };
        match observe(pid) {
            Ok(observation) => Ok(observation),
            Err(Errno::ECHILD) => {
                self.active = false;
                Ok(NonReapingChildState::OwnershipLost)
            }
            Err(_) => Err(ProcessGroupSignalError::ObservationUnavailable),
        }
    }

    pub(super) fn signal_owned_workflow(
        &mut self,
        signal: Signal,
    ) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError> {
        self.signal_owned_workflow_with(signal, nix::sys::signal::kill, nix::sys::signal::kill)
    }

    pub(super) fn signal_owned_workflow_with<SignalGroup, SignalLeader>(
        &mut self,
        signal: Signal,
        signal_group: SignalGroup,
        signal_leader: SignalLeader,
    ) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>
    where
        SignalGroup: FnOnce(Pid, Signal) -> nix::Result<()>,
        SignalLeader: FnOnce(Pid, Signal) -> nix::Result<()>,
    {
        if !self.active {
            return Err(ProcessGroupSignalError::OwnershipLost);
        }
        let pid = match process_pid_u32(self.pid) {
            Ok(pid) => pid,
            Err(error) => {
                self.active = false;
                return Err(error);
            }
        };
        signal_workflow_targets_with(pid, signal, signal_group, signal_leader)
    }
}

#[derive(Clone)]
pub(super) struct SignalAuthority {
    state: Arc<Mutex<SignalAuthorityState>>,
}

struct SignalAuthorityState {
    identity: LinuxProcessIdentity,
    active: bool,
}

impl SignalAuthority {
    pub(super) fn new(identity: LinuxProcessIdentity) -> Self {
        Self {
            state: Arc::new(Mutex::new(SignalAuthorityState {
                identity,
                active: true,
            })),
        }
    }

    pub(super) fn validate_current(&self) -> Result<(), ProcessGroupSignalError> {
        self.with_current_identity(|_identity| Ok(()))
    }

    #[cfg(test)]
    pub(super) fn revoke(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active = false;
    }

    pub(super) fn is_active(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
    }

    fn with_current_identity<T>(
        &self,
        action: impl FnOnce(LinuxProcessIdentity) -> Result<T, ProcessGroupSignalError>,
    ) -> Result<T, ProcessGroupSignalError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.active {
            return Err(ProcessGroupSignalError::OwnershipLost);
        }
        if let Err(error) = state.identity.validate_current() {
            if error.revokes_authority() {
                state.active = false;
            }
            return Err(error);
        }
        let result = action(state.identity);
        if matches!(result, Err(error) if error.revokes_authority()) {
            state.active = false;
        }
        result
    }

    pub(super) fn observe_child_nonreaping(
        &self,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError> {
        self.observe_child_nonreaping_with(|pid| {
            let flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOWAIT | WaitPidFlag::WNOHANG;
            waitid(Id::Pid(pid), flags).map(wait_status_to_nonreaping_state)
        })
    }

    pub(super) fn observe_child_nonreaping_with<Observe>(
        &self,
        observe: Observe,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError>
    where
        Observe: FnOnce(Pid) -> Result<NonReapingChildState, Errno>,
    {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.active {
            return Ok(NonReapingChildState::OwnershipLost);
        }
        if let Err(error) = state.identity.validate_current() {
            if error.revokes_authority() {
                state.active = false;
            }
            return Err(error);
        }
        let pid = match process_pid(state.identity) {
            Ok(pid) => Pid::from_raw(pid),
            Err(error) => {
                state.active = false;
                return Err(error);
            }
        };
        match observe(pid) {
            Ok(observation) => Ok(observation),
            Err(Errno::ECHILD) => {
                state.active = false;
                Ok(NonReapingChildState::OwnershipLost)
            }
            Err(_) => Err(ProcessGroupSignalError::ObservationUnavailable),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NonReapingChildState {
    Running,
    Exited,
    OwnershipLost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WorkflowSignalOutcome {
    StrictGroup,
    LeaderOnly,
}

#[cfg(test)]
pub(super) fn signal_owned_process_group_with<SendSignal>(
    authority: &SignalAuthority,
    signal: Signal,
    send_signal: SendSignal,
) -> Result<(), ProcessGroupSignalError>
where
    SendSignal: FnOnce(Pid, Signal) -> nix::Result<()>,
{
    authority.with_current_identity(|identity| {
        let pid = process_pid(identity)?;
        match send_signal(Pid::from_raw(-pid), signal) {
            Ok(()) | Err(Errno::ESRCH) => Ok(()),
            Err(_) => Err(ProcessGroupSignalError::SignalFailed),
        }
    })
}

pub(super) fn signal_owned_workflow(
    authority: &SignalAuthority,
    signal: Signal,
) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError> {
    signal_owned_workflow_with(
        authority,
        signal,
        nix::sys::signal::kill,
        nix::sys::signal::kill,
    )
}

pub(super) fn signal_owned_workflow_with<SignalGroup, SignalLeader>(
    authority: &SignalAuthority,
    signal: Signal,
    signal_group: SignalGroup,
    signal_leader: SignalLeader,
) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>
where
    SignalGroup: FnOnce(Pid, Signal) -> nix::Result<()>,
    SignalLeader: FnOnce(Pid, Signal) -> nix::Result<()>,
{
    authority.with_current_identity(|identity| {
        let pid = process_pid(identity)?;
        signal_workflow_targets_with(pid, signal, signal_group, signal_leader)
    })
}

fn signal_workflow_targets_with<SignalGroup, SignalLeader>(
    pid: i32,
    signal: Signal,
    signal_group: SignalGroup,
    signal_leader: SignalLeader,
) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>
where
    SignalGroup: FnOnce(Pid, Signal) -> nix::Result<()>,
    SignalLeader: FnOnce(Pid, Signal) -> nix::Result<()>,
{
    let group_complete = matches!(
        signal_group(Pid::from_raw(-pid), signal),
        Ok(()) | Err(Errno::ESRCH)
    );
    match signal_leader(Pid::from_raw(pid), signal) {
        Ok(()) | Err(Errno::ESRCH) if group_complete => Ok(WorkflowSignalOutcome::StrictGroup),
        Ok(()) | Err(Errno::ESRCH) => Ok(WorkflowSignalOutcome::LeaderOnly),
        Err(_) => Err(ProcessGroupSignalError::SignalFailed),
    }
}

fn wait_status_to_nonreaping_state(status: WaitStatus) -> NonReapingChildState {
    match status {
        WaitStatus::StillAlive | WaitStatus::Continued(_) | WaitStatus::Stopped(_, _) => {
            NonReapingChildState::Running
        }
        WaitStatus::Exited(_, _)
        | WaitStatus::Signaled(_, _, _)
        | WaitStatus::PtraceEvent(_, _, _)
        | WaitStatus::PtraceSyscall(_) => NonReapingChildState::Exited,
    }
}

fn process_pid(identity: LinuxProcessIdentity) -> Result<i32, ProcessGroupSignalError> {
    process_pid_u32(identity.pid)
}

fn process_pid_u32(pid: u32) -> Result<i32, ProcessGroupSignalError> {
    let pid = i32::try_from(pid).map_err(|_| ProcessGroupSignalError::InvalidPid)?;
    if pid == 0 {
        return Err(ProcessGroupSignalError::InvalidPid);
    }
    Ok(pid)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessGroupSignalError {
    InvalidPid,
    IdentityUnavailable,
    IdentityMismatch,
    OwnershipLost,
    ObservationUnavailable,
    SignalFailed,
}

impl ProcessGroupSignalError {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidPid => "invalid_pid",
            Self::IdentityUnavailable => "identity_unavailable",
            Self::IdentityMismatch => "identity_mismatch",
            Self::OwnershipLost => "ownership_lost",
            Self::ObservationUnavailable => "observation_unavailable",
            Self::SignalFailed => "signal_failed",
        }
    }

    const fn revokes_authority(self) -> bool {
        matches!(
            self,
            Self::InvalidPid | Self::IdentityMismatch | Self::OwnershipLost
        )
    }
}

pub(crate) fn linux_process_start_time(pid: u32) -> Result<u64, ProcessGroupSignalError> {
    linux_process_state_and_start_time(pid).map(|(_state, start_time_ticks)| start_time_ticks)
}

fn linux_process_state_and_start_time(pid: u32) -> Result<(char, u64), ProcessGroupSignalError> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_err(|_| ProcessGroupSignalError::IdentityUnavailable)?;
    parse_linux_process_state_and_start_time(&stat)
}

fn parse_linux_process_state_and_start_time(
    stat: &str,
) -> Result<(char, u64), ProcessGroupSignalError> {
    let (_prefix, suffix) = stat
        .rsplit_once(") ")
        .ok_or(ProcessGroupSignalError::IdentityUnavailable)?;
    let state = suffix
        .chars()
        .next()
        .ok_or(ProcessGroupSignalError::IdentityUnavailable)?;
    // The suffix starts at field 3 (state); starttime is field 22.
    let start_time_ticks = suffix
        .split_whitespace()
        .nth(19)
        .ok_or(ProcessGroupSignalError::IdentityUnavailable)?
        .parse()
        .map_err(|_| ProcessGroupSignalError::IdentityUnavailable)?;
    Ok((state, start_time_ticks))
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinuxProcessIdentityProbe {
    Current,
    ConfirmedGoneOrMismatch,
    Unavailable,
}

#[cfg(test)]
pub(crate) fn probe_linux_process_identity(
    pid: u32,
    expected_start_time_ticks: u64,
) -> LinuxProcessIdentityProbe {
    match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => match parse_linux_process_state_and_start_time(&stat) {
            Ok((_state, observed)) if observed == expected_start_time_ticks => {
                LinuxProcessIdentityProbe::Current
            }
            Ok(_) => LinuxProcessIdentityProbe::ConfirmedGoneOrMismatch,
            Err(_) => LinuxProcessIdentityProbe::Unavailable,
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            LinuxProcessIdentityProbe::ConfirmedGoneOrMismatch
        }
        Err(_) => LinuxProcessIdentityProbe::Unavailable,
    }
}

#[cfg(test)]
pub(crate) fn linux_process_is_live(pid: u32) -> bool {
    let Ok(raw_pid) = i32::try_from(pid) else {
        return false;
    };
    nix::sys::signal::kill(Pid::from_raw(raw_pid), None).is_ok()
        && linux_process_state_and_start_time(pid)
            .is_ok_and(|(state, _start_time_ticks)| state != 'Z')
}
