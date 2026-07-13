use std::{
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use super::{
    DeferredSignalState, DeferredWorkflowProcess, LinuxProcessIdentity, SharedDeferredReaper,
    SignalAuthority, SpawnedChildGuard, configure_process_group, poll_deferred_processes,
    signal_authority::ProvisionalGroupAuthority, test_raii::TestLocalDeferredReaper,
};
use crate::workflow_execution::WorkflowExecutionLease;

#[test]
fn every_deferred_signal_state_retains_its_execution_lease() {
    for state_kind in [
        DeferredLeaseState::Normal,
        DeferredLeaseState::Provisional,
        DeferredLeaseState::Unresolved,
    ] {
        let drops = Arc::new(AtomicUsize::new(0));
        let (child, identity) = spawn_sleep_group();
        let state = match state_kind {
            DeferredLeaseState::Normal => {
                DeferredSignalState::SignalPending(SignalAuthority::new(identity))
            }
            DeferredLeaseState::Provisional => DeferredSignalState::ProvisionalSignalPending(
                ProvisionalGroupAuthority::new(identity.pid),
            ),
            DeferredLeaseState::Unresolved => DeferredSignalState::Unresolved,
        };
        let process = DeferredWorkflowProcess::new_with_execution_lease(
            child,
            state,
            WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops))),
        );

        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(TestDeferredLeaseProcess::new(process, identity));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn successful_deferred_reap_releases_execution_lease_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (child, _identity) = spawn_true_group();
    let process = DeferredWorkflowProcess::new_with_execution_lease(
        child,
        DeferredSignalState::StrictGroupSignaled,
        WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops))),
    );
    let reaper = TestLocalDeferredReaper::new(SharedDeferredReaper {
        processes: Arc::new(std::sync::Mutex::new(vec![process])),
        pending_cleanups: Arc::new(AtomicUsize::new(1)),
        worker_available: true,
    });
    let deadline = Instant::now() + Duration::from_secs(1);

    while reaper.pending_cleanups.load(Ordering::SeqCst) != 0 {
        assert!(
            Instant::now() < deadline,
            "deferred true child was not reaped"
        );
        poll_deferred_processes(
            &mut reaper
                .processes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            &reaper.pending_cleanups,
        );
        thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(reaper);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
enum DeferredLeaseState {
    Normal,
    Provisional,
    Unresolved,
}

fn spawn_sleep_group() -> (std::process::Child, LinuxProcessIdentity) {
    let mut command = Command::new("/bin/sleep");
    command.arg("30");
    spawn_group(&mut command)
}

fn spawn_true_group() -> (std::process::Child, LinuxProcessIdentity) {
    let mut command = Command::new("/bin/true");
    spawn_group(&mut command)
}

fn spawn_group(command: &mut Command) -> (std::process::Child, LinuxProcessIdentity) {
    configure_process_group(command);
    let child = command.spawn().expect("lease fixture should spawn");
    let mut spawned = SpawnedChildGuard::new(child, Instant::now() + Duration::from_secs(1));
    let identity = LinuxProcessIdentity::capture(spawned.pid())
        .expect("lease fixture identity should be captured");
    spawned.set_signal_authority(SignalAuthority::new(identity));
    let child = spawned
        .disarm()
        .expect("lease fixture should transfer child ownership");
    (child, identity)
}

struct TestDeferredLeaseProcess {
    process: Option<DeferredWorkflowProcess>,
    identity: LinuxProcessIdentity,
}

impl TestDeferredLeaseProcess {
    const fn new(process: DeferredWorkflowProcess, identity: LinuxProcessIdentity) -> Self {
        Self {
            process: Some(process),
            identity,
        }
    }
}

impl Drop for TestDeferredLeaseProcess {
    fn drop(&mut self) {
        let Some(mut process) = self.process.take() else {
            return;
        };
        let authority = SignalAuthority::new(self.identity);
        let _signal = super::signal_owned_workflow(&authority, nix::sys::signal::Signal::SIGKILL);
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match process.child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}
