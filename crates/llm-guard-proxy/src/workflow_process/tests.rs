use std::{
    cell::Cell,
    io,
    process::{Command, Stdio},
    sync::{Arc, Mutex, atomic::AtomicUsize, mpsc},
    thread,
    time::{Duration, Instant},
};

use nix::{
    errno::Errno,
    sys::signal::{Signal, kill},
    unistd::Pid,
};

use super::{
    DeferredPollOutcome, DeferredSignalState, DeferredWorkflowProcess, ExitObservation,
    LinuxProcessIdentity, NonReapingChildState, ProcessGroupSignalError, ReapPollOutcome,
    SharedDeferredReaper, SignalAuthority, SpawnedChildGuard, WaitIdPoll, WorkflowSignalOutcome,
    abort_cleanup_deadline, capture_process_identity_bounded_with, cleanup_permits_spawn,
    configure_process_group, grow_signal_retry_backoff, linux_process_is_live,
    linux_process_start_time, next_deferred_poll_delay, observe_child_exit_bounded_with,
    poll_deferred_cleanup_with, poll_deferred_processes, reap_child_bounded_with,
    signal_owned_process_group_with, signal_owned_workflow_with,
    test_raii::{TestDeferredProcess, TestLocalDeferredReaper, TestProcessGroup},
    watchdog_deadline_elapsed_with,
};

#[test]
fn cleanup_admission_requires_a_healthy_worker_without_pending_cleanup() {
    assert!(cleanup_permits_spawn(true, 0));
    assert!(!cleanup_permits_spawn(false, 0));
    assert!(!cleanup_permits_spawn(true, 1));
}

#[test]
fn signal_retry_backoff_grows_exponentially_and_caps() {
    let mut backoff = Duration::from_millis(10);
    backoff = grow_signal_retry_backoff(backoff);
    assert_eq!(backoff, Duration::from_millis(20));
    backoff = grow_signal_retry_backoff(backoff);
    assert_eq!(backoff, Duration::from_millis(40));
    assert_eq!(
        grow_signal_retry_backoff(Duration::from_millis(900)),
        Duration::from_secs(1)
    );
    assert_eq!(
        grow_signal_retry_backoff(Duration::from_secs(1)),
        Duration::from_secs(1)
    );
}

#[test]
fn deferred_retention_closes_stdio_for_every_signal_state() {
    for state_kind in [
        DeferredFixtureState::SignalPending,
        DeferredFixtureState::ProvisionalSignalPending,
        DeferredFixtureState::Unresolved,
    ] {
        let mut command = Command::new("/bin/sleep");
        command
            .arg("30")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_process_group(&mut command);
        let child = command
            .spawn()
            .expect("deferred stdio fixture should spawn");
        let mut spawned = SpawnedChildGuard::new(child, Instant::now() + super::PROCESS_REAP_GRACE);
        let identity = LinuxProcessIdentity::capture(spawned.pid())
            .expect("deferred stdio fixture identity should be captured");
        spawned.set_signal_authority(SignalAuthority::new(identity));
        let signal_state = match state_kind {
            DeferredFixtureState::SignalPending => {
                DeferredSignalState::SignalPending(SignalAuthority::new(identity))
            }
            DeferredFixtureState::ProvisionalSignalPending => {
                DeferredSignalState::ProvisionalSignalPending(
                    super::signal_authority::ProvisionalGroupAuthority::new(identity.pid),
                )
            }
            DeferredFixtureState::Unresolved => DeferredSignalState::Unresolved,
        };
        let process = DeferredWorkflowProcess::new(
            spawned
                .disarm()
                .expect("spawn guard should transfer the deferred fixture"),
            signal_state,
        );
        let process = TestDeferredProcess::new(process, identity);

        assert!(process.child.stdin.is_none());
        assert!(process.child.stdout.is_none());
        assert!(process.child.stderr.is_none());
    }
}

#[test]
fn deferred_worker_sleeps_until_backoff_instead_of_polling_at_100hz() {
    let now = Instant::now();
    let mut process = TestDeferredProcess::spawn_true(DeferredSignalState::StrictGroupSignaled);
    process.next_signal_attempt = now + Duration::from_secs(1);
    process.signal_retry_backoff = Duration::from_secs(1);

    assert_eq!(
        next_deferred_poll_delay(std::slice::from_ref(&*process), now),
        Duration::from_secs(1)
    );

    drop(TestLocalDeferredReaper::new(SharedDeferredReaper {
        processes: Arc::new(Mutex::new(vec![process.into_process()])),
        #[cfg(target_os = "linux")]
        cgroups: Arc::new(Mutex::new(Vec::new())),
        pending_cleanups: Arc::new(AtomicUsize::new(1)),
        worker_available: false,
    }));
}

#[test]
fn deferred_cleanup_blocks_admission_until_the_child_is_reaped() {
    let reaper = TestLocalDeferredReaper::new(SharedDeferredReaper {
        processes: Arc::new(Mutex::new(Vec::new())),
        #[cfg(target_os = "linux")]
        cgroups: Arc::new(Mutex::new(Vec::new())),
        pending_cleanups: Arc::default(),
        worker_available: true,
    });
    let process = TestDeferredProcess::spawn_true(DeferredSignalState::StrictGroupSignaled);
    reaper.submit(process.into_process());

    let pending_cleanups = || {
        reaper
            .pending_cleanups
            .load(std::sync::atomic::Ordering::SeqCst)
    };
    assert!(!cleanup_permits_spawn(
        reaper.worker_available,
        pending_cleanups()
    ));

    let deadline = Instant::now() + Duration::from_secs(1);
    while pending_cleanups() != 0 {
        assert!(Instant::now() < deadline, "deferred child was not reaped");
        poll_deferred_processes(
            &mut reaper
                .processes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            &reaper.pending_cleanups,
        );
        thread::sleep(Duration::from_millis(10));
    }

    assert!(cleanup_permits_spawn(
        reaper.worker_available,
        pending_cleanups()
    ));
}

#[test]
fn stale_process_identity_is_rejected_by_the_signal_coordinator() {
    let mut command = Command::new("/bin/sleep");
    command.arg("30");
    configure_process_group(&mut command);
    let mut child = TestProcessGroup::spawn(&mut command);
    let identity = child.identity();
    let stale_identity = LinuxProcessIdentity {
        start_time_ticks: identity.start_time_ticks.saturating_add(1),
        ..identity
    };
    let authority = SignalAuthority::new(stale_identity);
    let signal_calls = Cell::new(0_usize);

    let identity_result =
        signal_owned_process_group_with(&authority, Signal::SIGKILL, |_target, _signal| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(())
        });

    assert_eq!(
        identity_result,
        Err(ProcessGroupSignalError::IdentityMismatch)
    );
    assert_eq!(
        signal_calls.get(),
        0,
        "stale identity must close signal authority"
    );
    assert_eq!(
        child
            .child_mut()
            .try_wait()
            .expect("child state should be readable"),
        None,
        "identity mismatch must not signal the live process"
    );
    signal_owned_process_group_with(&SignalAuthority::new(identity), Signal::SIGKILL, kill)
        .expect("owned process group should accept cleanup signal");
    child.finish_after_signal();
}

#[test]
fn revoked_signal_authority_is_sticky_across_clones() {
    let identity = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let authority = SignalAuthority::new(identity);
    let cloned = authority.clone();
    let signal_calls = Cell::new(0_usize);
    authority.revoke();

    let result = signal_owned_process_group_with(&cloned, Signal::SIGKILL, |_target, _signal| {
        signal_calls.set(signal_calls.get() + 1);
        Ok(())
    });

    assert_eq!(result, Err(ProcessGroupSignalError::OwnershipLost));
    assert_eq!(signal_calls.get(), 0);
}

#[test]
fn owned_workflow_signal_reports_leader_only_fallback() {
    let identity = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let authority = SignalAuthority::new(identity);
    let group_calls = Cell::new(0_usize);
    let leader_calls = Cell::new(0_usize);

    let result = signal_owned_workflow_with(
        &authority,
        Signal::SIGKILL,
        |_target, _signal| {
            group_calls.set(group_calls.get() + 1);
            Err(nix::errno::Errno::EPERM)
        },
        |target, signal| {
            leader_calls.set(leader_calls.get() + 1);
            assert_eq!(
                target,
                Pid::from_raw(i32::try_from(identity.pid).expect("test PID should fit i32"))
            );
            assert_eq!(signal, Signal::SIGKILL);
            Ok(())
        },
    );

    assert_eq!(result, Ok(WorkflowSignalOutcome::LeaderOnly));
    assert_eq!(group_calls.get(), 1);
    assert_eq!(leader_calls.get(), 1);
}

#[test]
fn group_esrch_still_signals_the_exact_leader() {
    let identity = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let authority = SignalAuthority::new(identity);
    let leader_calls = Cell::new(0_usize);

    let result = signal_owned_workflow_with(
        &authority,
        Signal::SIGKILL,
        |_target, _signal| Err(Errno::ESRCH),
        |_target, _signal| {
            leader_calls.set(leader_calls.get() + 1);
            Ok(())
        },
    );

    assert_eq!(result, Ok(WorkflowSignalOutcome::StrictGroup));
    assert_eq!(leader_calls.get(), 1);
}

#[test]
fn transient_identity_unavailability_keeps_signal_authority_active() {
    let authority = SignalAuthority::new(LinuxProcessIdentity {
        pid: u32::MAX,
        start_time_ticks: 1,
    });
    let signal_calls = Cell::new(0_usize);

    let result =
        signal_owned_process_group_with(&authority, Signal::SIGKILL, |_target, _signal| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(())
        });

    assert_eq!(result, Err(ProcessGroupSignalError::IdentityUnavailable));
    assert!(authority.is_active());
    assert_eq!(signal_calls.get(), 0);
}

#[test]
fn nonreaping_echild_revokes_authority_across_clones() {
    let identity = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let authority = SignalAuthority::new(identity);
    let clone = authority.clone();
    let signal_calls = Cell::new(0_usize);

    let observation = authority
        .observe_child_nonreaping_with(|_pid| Err(Errno::ECHILD))
        .expect("ECHILD should become a sticky ownership-loss observation");
    let signal_result =
        signal_owned_process_group_with(&clone, Signal::SIGKILL, |_target, _signal| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(())
        });

    assert_eq!(observation, NonReapingChildState::OwnershipLost);
    assert_eq!(signal_result, Err(ProcessGroupSignalError::OwnershipLost));
    assert_eq!(signal_calls.get(), 0);
}

#[test]
fn nonreaping_observation_and_signal_are_serialized_by_one_authority_lock() {
    let identity = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let authority = SignalAuthority::new(identity);
    let observer_authority = authority.clone();
    let signal_authority = authority.clone();
    let deadline = Instant::now() + Duration::from_secs(1);
    let (observer_entered_tx, observer_entered_rx) = mpsc::sync_channel(1);
    let (release_observer_tx, release_observer_rx) = mpsc::sync_channel(1);
    let (signal_finished_tx, signal_finished_rx) = mpsc::sync_channel(1);
    let mut release_observer = ReleaseOnDrop::new(release_observer_tx);

    thread::scope(|scope| {
        scope.spawn(move || {
            observer_authority
                .observe_child_nonreaping_with(|_pid| {
                    observer_entered_tx
                        .send(())
                        .expect("observer entry should be reported");
                    release_observer_rx
                        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                        .expect("observer release should arrive before the fixture deadline");
                    Ok(NonReapingChildState::Running)
                })
                .expect("synthetic observation should succeed")
        });
        observer_entered_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("observer should acquire the authority lock before the fixture deadline");
        scope.spawn(move || {
            let _result = signal_owned_process_group_with(
                &signal_authority,
                Signal::SIGKILL,
                |_target, _signal| Err(Errno::EPERM),
            );
            let _finished = signal_finished_tx.send(());
        });
        assert_eq!(
            signal_finished_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "signal must wait until non-reaping observation releases the coordinator"
        );
        release_observer.release();
        signal_finished_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("signal should finish after observer release before the fixture deadline");
    });
}

#[test]
fn identity_capture_retries_transient_unavailability_within_one_budget() {
    let expected = LinuxProcessIdentity::capture(std::process::id())
        .expect("current process identity should be captured");
    let attempts = Cell::new(0_usize);
    let pauses = Cell::new(0_usize);

    let captured = capture_process_identity_bounded_with(
        expected.pid,
        |_pid| {
            attempts.set(attempts.get() + 1);
            if attempts.get() < 3 {
                Err(ProcessGroupSignalError::IdentityUnavailable)
            } else {
                Ok(expected)
            }
        },
        || false,
        || pauses.set(pauses.get() + 1),
    )
    .expect("transient identity probe failures should be retried");

    assert_eq!(captured, expected);
    assert_eq!(attempts.get(), 3);
    assert_eq!(pauses.get(), 2);
}

#[test]
fn identity_capture_stops_when_the_startup_cleanup_budget_expires() {
    let attempts = Cell::new(0_usize);
    let pauses = Cell::new(0_usize);

    let result = capture_process_identity_bounded_with(
        123,
        |_pid| {
            attempts.set(attempts.get() + 1);
            Err(ProcessGroupSignalError::IdentityUnavailable)
        },
        || attempts.get() >= 3,
        || pauses.set(pauses.get() + 1),
    );

    assert_eq!(result, Err(ProcessGroupSignalError::IdentityUnavailable));
    assert_eq!(attempts.get(), 3);
    assert_eq!(pauses.get(), 2);
}

#[test]
fn startup_guard_without_signal_identity_kills_and_reaps_the_owned_leader() {
    let mut command = Command::new("/bin/sleep");
    command.arg("30");
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("startup cleanup fixture should spawn");
    let mut child = SpawnedChildGuard::new(child, Instant::now() + Duration::from_secs(1));
    let pid = child.pid();
    let identity = LinuxProcessIdentity::capture(pid)
        .expect("startup cleanup fixture identity should be captured");
    child.set_signal_authority(SignalAuthority::new(identity));

    drop(child);
    let deadline = Instant::now() + Duration::from_secs(1);
    while linux_process_start_time(pid).ok() == Some(identity.start_time_ticks)
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(10));
    }
    assert_ne!(
        linux_process_start_time(pid).ok(),
        Some(identity.start_time_ticks),
        "startup guard should terminate and reap directly owned leader {pid}"
    );
}

#[test]
fn cancellation_observed_after_the_deadline_still_times_out() {
    let deadline = Instant::now();

    let timed_out = watchdog_deadline_elapsed_with(
        deadline,
        |_remaining| Ok(()),
        || deadline + Duration::from_millis(1),
    );

    assert!(timed_out);
}

#[test]
fn early_abort_cleanup_never_extends_the_hard_deadline() {
    let now = Instant::now();
    let execution_deadline = now
        .checked_sub(Duration::from_secs(1))
        .expect("test instant should permit subtracting one second");

    assert_eq!(
        abort_cleanup_deadline(execution_deadline, now),
        execution_deadline + super::PROCESS_REAP_GRACE
    );
}

#[test]
fn in_process_probe_recognizes_the_current_process() {
    assert!(linux_process_is_live(std::process::id()));
}

#[test]
fn final_reap_stops_at_poll_budget() {
    let poll_count = Cell::new(0_usize);

    let outcome = reap_child_bounded_with(
        || {
            poll_count.set(poll_count.get() + 1);
            Ok(None)
        },
        || poll_count.get() >= 3,
        || {},
    )
    .expect("synthetic polling should not fail");

    assert!(matches!(outcome, ReapPollOutcome::TimedOut));
    assert_eq!(poll_count.get(), 3);
}

#[test]
fn exit_observation_stops_at_poll_budget() {
    let poll_count = Cell::new(0_usize);

    let outcome = observe_child_exit_bounded_with(
        || {
            poll_count.set(poll_count.get() + 1);
            Ok(WaitIdPoll::StillRunning)
        },
        || poll_count.get() >= 3,
        || {},
    )
    .expect("synthetic observation should not fail");

    assert_eq!(outcome, ExitObservation::DeadlineExceeded);
    assert_eq!(poll_count.get(), 3);
}

#[test]
fn deferred_cleanup_observes_then_strictly_signals_before_reaping() {
    let authority = current_signal_authority();
    let mut signal_state = DeferredSignalState::SignalPending(authority);
    let calls = std::cell::RefCell::new(Vec::new());

    let outcome = poll_deferred_cleanup_with(
        &mut signal_state,
        || {
            calls.borrow_mut().push("observe");
            Ok(NonReapingChildState::Exited)
        },
        || {
            calls.borrow_mut().push("signal");
            Ok(WorkflowSignalOutcome::StrictGroup)
        },
        || {
            calls.borrow_mut().push("reap");
            Ok(true)
        },
    );

    assert_eq!(outcome, DeferredPollOutcome::Complete);
    assert_eq!(*calls.borrow(), ["observe", "signal", "reap"]);
    assert!(matches!(
        signal_state,
        DeferredSignalState::StrictGroupSignaled
    ));
}

#[test]
fn deferred_cleanup_never_reaps_after_leader_only_fallback() {
    let authority = current_signal_authority();
    let mut signal_state = DeferredSignalState::SignalPending(authority);
    let reap_calls = Cell::new(0_usize);

    let outcome = poll_deferred_cleanup_with(
        &mut signal_state,
        || Ok(NonReapingChildState::Exited),
        || Ok(WorkflowSignalOutcome::LeaderOnly),
        || {
            reap_calls.set(reap_calls.get() + 1);
            Ok(true)
        },
    );

    assert_eq!(outcome, DeferredPollOutcome::Pending);
    assert!(matches!(
        signal_state,
        DeferredSignalState::SignalPending(_)
    ));
    assert_eq!(reap_calls.get(), 0);
}

#[test]
fn deferred_cleanup_echild_is_permanent_fail_closed_backpressure() {
    let authority = current_signal_authority();
    let mut signal_state = DeferredSignalState::SignalPending(authority);
    let signal_calls = Cell::new(0_usize);
    let reap_calls = Cell::new(0_usize);

    let first = poll_deferred_cleanup_with(
        &mut signal_state,
        || Ok(NonReapingChildState::OwnershipLost),
        || {
            signal_calls.set(signal_calls.get() + 1);
            Ok(WorkflowSignalOutcome::StrictGroup)
        },
        || {
            reap_calls.set(reap_calls.get() + 1);
            Ok(true)
        },
    );
    let second = poll_deferred_cleanup_with(
        &mut signal_state,
        || panic!("unresolved ownership must not be observed again"),
        || panic!("unresolved ownership must not signal"),
        || panic!("unresolved ownership must not reap"),
    );

    assert_eq!(first, DeferredPollOutcome::Pending);
    assert_eq!(second, DeferredPollOutcome::Pending);
    assert!(matches!(signal_state, DeferredSignalState::Unresolved));
    assert_eq!(signal_calls.get(), 0);
    assert_eq!(reap_calls.get(), 0);
}

#[test]
fn deferred_reap_echild_becomes_permanent_backpressure() {
    let mut signal_state = DeferredSignalState::StrictGroupSignaled;

    let outcome = poll_deferred_cleanup_with(
        &mut signal_state,
        || panic!("strictly signalled child needs no observation"),
        || panic!("strictly signalled child needs no second signal"),
        || Err(io::Error::from_raw_os_error(Errno::ECHILD as i32)),
    );

    assert_eq!(outcome, DeferredPollOutcome::Pending);
    assert!(matches!(signal_state, DeferredSignalState::Unresolved));
}

fn current_signal_authority() -> SignalAuthority {
    SignalAuthority::new(
        LinuxProcessIdentity::capture(std::process::id())
            .expect("current process identity should be captured"),
    )
}

#[derive(Clone, Copy)]
enum DeferredFixtureState {
    SignalPending,
    ProvisionalSignalPending,
    Unresolved,
}

struct ReleaseOnDrop(Option<mpsc::SyncSender<()>>);

impl ReleaseOnDrop {
    const fn new(sender: mpsc::SyncSender<()>) -> Self {
        Self(Some(sender))
    }

    fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            let _released = sender.send(());
        }
    }
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.release();
    }
}
