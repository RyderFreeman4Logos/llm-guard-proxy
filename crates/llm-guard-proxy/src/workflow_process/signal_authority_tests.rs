use std::cell::RefCell;

use nix::{errno::Errno, sys::signal::Signal, unistd::Pid};

use super::signal_authority::{
    LinuxProcessIdentity, ProcessGroupSignalError, ProvisionalGroupAuthority, SignalAuthority,
    WorkflowSignalOutcome, signal_owned_workflow_with,
};

#[test]
fn validated_authority_signals_exact_leader_after_group_success() {
    assert_validated_signal_case(Ok(()), Ok(()), Ok(WorkflowSignalOutcome::StrictGroup), 2);
}

#[test]
fn validated_authority_signals_exact_leader_after_group_esrch() {
    assert_validated_signal_case(
        Err(Errno::ESRCH),
        Ok(()),
        Ok(WorkflowSignalOutcome::StrictGroup),
        2,
    );
}

#[test]
fn validated_authority_keeps_cleanup_pending_when_leader_signal_fails() {
    assert_validated_signal_case(
        Ok(()),
        Err(Errno::EPERM),
        Err(ProcessGroupSignalError::SignalFailed),
        2,
    );
}

#[test]
fn validated_authority_reports_group_failure_leader_success_as_partial() {
    assert_validated_signal_case(
        Err(Errno::EPERM),
        Ok(()),
        Ok(WorkflowSignalOutcome::LeaderOnly),
        2,
    );
}

#[test]
fn validated_ownership_loss_prevents_group_and_leader_signals() {
    let authority = current_authority();
    authority.revoke();
    let calls = RefCell::new(Vec::new());

    let result = signal_owned_workflow_with(
        &authority,
        Signal::SIGKILL,
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            Ok(())
        },
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            Ok(())
        },
    );

    assert_eq!(result, Err(ProcessGroupSignalError::OwnershipLost));
    assert!(calls.borrow().is_empty());
}

#[test]
fn provisional_authority_signals_exact_leader_after_group_success() {
    assert_provisional_signal_case(Ok(()), Ok(()), Ok(WorkflowSignalOutcome::StrictGroup), 2);
}

#[test]
fn provisional_authority_signals_exact_leader_after_group_esrch() {
    assert_provisional_signal_case(
        Err(Errno::ESRCH),
        Ok(()),
        Ok(WorkflowSignalOutcome::StrictGroup),
        2,
    );
}

#[test]
fn provisional_authority_keeps_cleanup_pending_when_leader_signal_fails() {
    assert_provisional_signal_case(
        Ok(()),
        Err(Errno::EPERM),
        Err(ProcessGroupSignalError::SignalFailed),
        2,
    );
}

#[test]
fn provisional_authority_reports_group_failure_leader_success_as_partial() {
    assert_provisional_signal_case(
        Err(Errno::EPERM),
        Ok(()),
        Ok(WorkflowSignalOutcome::LeaderOnly),
        2,
    );
}

#[test]
fn provisional_ownership_loss_prevents_group_and_leader_signals() {
    let mut authority = ProvisionalGroupAuthority::new(std::process::id());
    authority.revoke();
    let calls = RefCell::new(Vec::new());

    let result = authority.signal_owned_workflow_with(
        Signal::SIGKILL,
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            Ok(())
        },
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            Ok(())
        },
    );

    assert_eq!(result, Err(ProcessGroupSignalError::OwnershipLost));
    assert!(calls.borrow().is_empty());
}

#[test]
fn both_authorities_accept_exact_leader_esrch_as_strict_completion() {
    assert_validated_signal_case(
        Ok(()),
        Err(Errno::ESRCH),
        Ok(WorkflowSignalOutcome::StrictGroup),
        2,
    );
    assert_provisional_signal_case(
        Ok(()),
        Err(Errno::ESRCH),
        Ok(WorkflowSignalOutcome::StrictGroup),
        2,
    );
}

fn assert_provisional_signal_case(
    group_result: nix::Result<()>,
    leader_result: nix::Result<()>,
    expected: Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
    expected_calls: usize,
) {
    let identity = current_identity();
    let mut authority = ProvisionalGroupAuthority::new(identity.pid);
    let calls = RefCell::new(Vec::new());

    let result = authority.signal_owned_workflow_with(
        Signal::SIGKILL,
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            group_result
        },
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            leader_result
        },
    );

    assert_eq!(result, expected);
    let calls = calls.borrow();
    assert_eq!(calls.len(), expected_calls);
    let raw_pid = i32::try_from(identity.pid).expect("test PID should fit i32");
    assert_eq!(calls[0], (Pid::from_raw(-raw_pid), Signal::SIGKILL));
    if expected_calls == 2 {
        assert_eq!(calls[1], (Pid::from_raw(raw_pid), Signal::SIGKILL));
    }
}

fn assert_validated_signal_case(
    group_result: nix::Result<()>,
    leader_result: nix::Result<()>,
    expected: Result<WorkflowSignalOutcome, ProcessGroupSignalError>,
    expected_calls: usize,
) {
    let identity = current_identity();
    let authority = SignalAuthority::new(identity);
    let calls = RefCell::new(Vec::new());

    let result = signal_owned_workflow_with(
        &authority,
        Signal::SIGKILL,
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            group_result
        },
        |target, signal| {
            calls.borrow_mut().push((target, signal));
            leader_result
        },
    );

    assert_eq!(result, expected);
    let calls = calls.borrow();
    assert_eq!(calls.len(), expected_calls);
    let raw_pid = i32::try_from(identity.pid).expect("test PID should fit i32");
    assert_eq!(calls[0], (Pid::from_raw(-raw_pid), Signal::SIGKILL));
    if expected_calls == 2 {
        assert_eq!(calls[1], (Pid::from_raw(raw_pid), Signal::SIGKILL));
    }
}

fn current_authority() -> SignalAuthority {
    SignalAuthority::new(current_identity())
}

fn current_identity() -> LinuxProcessIdentity {
    LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured")
}
