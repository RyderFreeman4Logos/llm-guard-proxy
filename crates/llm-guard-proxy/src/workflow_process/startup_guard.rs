#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::{io, process::ExitStatus, time::Instant};

#[cfg(test)]
use std::process::Child;

use nix::sys::signal::Signal;

#[cfg(target_os = "linux")]
use crate::workflow_cgroup::WorkflowCgroup;
use crate::workflow_execution::WorkflowExecutionLease;

use super::{
    WorkflowChild,
    deferred_reaper::{DeferredSignalState, defer_workflow_process_with_execution_lease},
    finalize_owned_child, process_group_signal_io_error, reap_child_bounded,
    signal_authority::{
        NonReapingChildState, ProcessGroupSignalError, ProvisionalGroupAuthority, SignalAuthority,
        WorkflowSignalOutcome,
    },
};

enum StartupSignalAuthority {
    Provisional(ProvisionalGroupAuthority),
    Stable(SignalAuthority),
}

pub(crate) struct SpawnedChildGuard {
    child: Option<WorkflowChild>,
    #[cfg(target_os = "linux")]
    workflow_cgroup: Option<Arc<WorkflowCgroup>>,
    signal_authority: Option<StartupSignalAuthority>,
    cleanup_deadline: Instant,
    execution_lease: Option<WorkflowExecutionLease>,
}

impl SpawnedChildGuard {
    #[cfg(test)]
    pub(crate) fn new(child: Child, cleanup_deadline: Instant) -> Self {
        Self::new_with_execution_lease(
            child.into(),
            cleanup_deadline,
            WorkflowExecutionLease::default(),
        )
    }

    pub(super) fn new_with_execution_lease(
        child: WorkflowChild,
        cleanup_deadline: Instant,
        execution_lease: WorkflowExecutionLease,
    ) -> Self {
        let pid = child.id();
        Self {
            child: Some(child),
            #[cfg(target_os = "linux")]
            workflow_cgroup: None,
            signal_authority: Some(StartupSignalAuthority::Provisional(
                ProvisionalGroupAuthority::new(pid),
            )),
            cleanup_deadline,
            execution_lease: Some(execution_lease),
        }
    }

    pub(crate) fn pid(&self) -> u32 {
        self.child.as_ref().map_or(0, WorkflowChild::id)
    }

    pub(super) fn child_mut(&mut self) -> Option<&mut WorkflowChild> {
        self.child.as_mut()
    }

    pub(super) fn set_signal_authority(&mut self, signal_authority: SignalAuthority) {
        self.signal_authority = Some(StartupSignalAuthority::Stable(signal_authority));
    }

    #[cfg(target_os = "linux")]
    pub(super) fn set_workflow_cgroup(&mut self, workflow_cgroup: Arc<WorkflowCgroup>) {
        self.workflow_cgroup = Some(workflow_cgroup);
    }

    #[cfg(target_os = "linux")]
    pub(super) fn workflow_cgroup(&self) -> Option<Arc<WorkflowCgroup>> {
        self.workflow_cgroup.clone()
    }

    #[cfg(target_os = "linux")]
    pub(super) fn clear_workflow_cgroup(&mut self) {
        self.workflow_cgroup.take();
    }

    #[cfg(target_os = "linux")]
    pub(super) fn take_workflow_cgroup(&mut self) -> Option<Arc<WorkflowCgroup>> {
        self.workflow_cgroup.take()
    }

    pub(super) fn revoke_provisional_authority(&mut self) {
        if let Some(StartupSignalAuthority::Provisional(authority)) = self.signal_authority.as_mut()
        {
            authority.revoke();
        }
    }

    #[cfg(test)]
    pub(crate) fn disarm(&mut self) -> Option<WorkflowChild> {
        self.signal_authority.take();
        self.execution_lease.take();
        self.child.take()
    }

    pub(super) fn disarm_with_execution_lease(
        &mut self,
    ) -> Option<(WorkflowChild, WorkflowExecutionLease)> {
        self.signal_authority.take();
        Some((self.child.take()?, self.execution_lease.take()?))
    }
}

impl Drop for SpawnedChildGuard {
    fn drop(&mut self) {
        let Some(child) = self.child.take() else {
            return;
        };
        #[cfg(target_os = "linux")]
        let workflow_cgroup = self.workflow_cgroup.take();
        #[cfg(target_os = "linux")]
        if let Some(workflow_cgroup) = workflow_cgroup.as_ref() {
            let _cgroup_kill = workflow_cgroup.kill();
        }
        let execution_lease = self.execution_lease.take().unwrap_or_default();
        match self.signal_authority.take() {
            Some(StartupSignalAuthority::Stable(signal_authority)) => {
                let _cleanup = finalize_owned_child(
                    child,
                    signal_authority,
                    self.cleanup_deadline,
                    execution_lease,
                );
            }
            Some(StartupSignalAuthority::Provisional(provisional_authority)) => {
                let _cleanup = finalize_provisional_child(
                    child,
                    provisional_authority,
                    self.cleanup_deadline,
                    execution_lease,
                );
            }
            None => defer_workflow_process_with_execution_lease(
                child,
                DeferredSignalState::Unresolved,
                execution_lease,
            ),
        }
        #[cfg(target_os = "linux")]
        if let Some(workflow_cgroup) = workflow_cgroup {
            let _cgroup_cleanup = workflow_cgroup.verify_empty_and_remove(self.cleanup_deadline);
        }
    }
}

pub(super) fn startup_error_revokes_provisional_authority(error: ProcessGroupSignalError) -> bool {
    matches!(
        error,
        ProcessGroupSignalError::InvalidPid
            | ProcessGroupSignalError::IdentityMismatch
            | ProcessGroupSignalError::OwnershipLost
    )
}

fn finalize_provisional_child(
    child: WorkflowChild,
    provisional_authority: ProvisionalGroupAuthority,
    cleanup_deadline: Instant,
    execution_lease: WorkflowExecutionLease,
) -> io::Result<ExitStatus> {
    finalize_provisional_child_with_execution_lease(
        child,
        provisional_authority,
        cleanup_deadline,
        execution_lease,
        ProvisionalGroupAuthority::observe_child_nonreaping,
        |authority| authority.signal_owned_workflow(Signal::SIGKILL),
        defer_workflow_process_with_execution_lease,
    )
}

#[cfg(test)]
pub(super) fn finalize_provisional_child_with<ChildHandle, Observe, SignalGroup, Defer>(
    child: ChildHandle,
    authority: ProvisionalGroupAuthority,
    cleanup_deadline: Instant,
    observe: Observe,
    signal_group: SignalGroup,
    defer: Defer,
) -> io::Result<ExitStatus>
where
    ChildHandle: Into<WorkflowChild>,
    Observe: FnOnce(
        &mut ProvisionalGroupAuthority,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError>,
    SignalGroup: FnOnce(
        &mut ProvisionalGroupAuthority,
    ) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
    Defer: FnOnce(WorkflowChild, DeferredSignalState),
{
    finalize_provisional_child_with_execution_lease(
        child.into(),
        authority,
        cleanup_deadline,
        WorkflowExecutionLease::default(),
        observe,
        signal_group,
        |child, signal_state, _execution_lease| defer(child, signal_state),
    )
}

fn finalize_provisional_child_with_execution_lease<Observe, SignalGroup, Defer>(
    child: WorkflowChild,
    mut authority: ProvisionalGroupAuthority,
    cleanup_deadline: Instant,
    execution_lease: WorkflowExecutionLease,
    observe: Observe,
    signal_group: SignalGroup,
    defer: Defer,
) -> io::Result<ExitStatus>
where
    Observe: FnOnce(
        &mut ProvisionalGroupAuthority,
    ) -> Result<NonReapingChildState, ProcessGroupSignalError>,
    SignalGroup: FnOnce(
        &mut ProvisionalGroupAuthority,
    ) -> Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
    Defer: FnOnce(WorkflowChild, DeferredSignalState, WorkflowExecutionLease),
{
    if !authority.is_active() {
        defer(child, DeferredSignalState::Unresolved, execution_lease);
        return Err(process_group_signal_io_error(
            ProcessGroupSignalError::OwnershipLost,
        ));
    }

    match observe(&mut authority) {
        Ok(NonReapingChildState::Running | NonReapingChildState::Exited) => {}
        Ok(NonReapingChildState::OwnershipLost) => {
            authority.revoke();
            defer(child, DeferredSignalState::Unresolved, execution_lease);
            return Err(process_group_signal_io_error(
                ProcessGroupSignalError::OwnershipLost,
            ));
        }
        Err(error) => {
            let signal_state =
                if startup_error_revokes_provisional_authority(error) || !authority.is_active() {
                    authority.revoke();
                    DeferredSignalState::Unresolved
                } else {
                    DeferredSignalState::ProvisionalSignalPending(authority)
                };
            defer(child, signal_state, execution_lease);
            return Err(process_group_signal_io_error(error));
        }
    }

    match signal_group(&mut authority) {
        Ok(WorkflowSignalOutcome::StrictGroup) => reap_child_bounded(
            child,
            cleanup_deadline,
            DeferredSignalState::StrictGroupSignaled,
            execution_lease,
        ),
        Ok(WorkflowSignalOutcome::LeaderOnly) => {
            defer(
                child,
                DeferredSignalState::ProvisionalSignalPending(authority),
                execution_lease,
            );
            Err(process_group_signal_io_error(
                ProcessGroupSignalError::SignalFailed,
            ))
        }
        Err(error) => {
            let signal_state =
                if startup_error_revokes_provisional_authority(error) || !authority.is_active() {
                    authority.revoke();
                    DeferredSignalState::Unresolved
                } else {
                    DeferredSignalState::ProvisionalSignalPending(authority)
                };
            defer(child, signal_state, execution_lease);
            Err(process_group_signal_io_error(error))
        }
    }
}
