use std::{
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(target_os = "linux")]
use std::{
    cell::{Cell, RefCell},
    fs, io,
    path::PathBuf,
};

#[cfg(target_os = "linux")]
use super::{
    DeferredPollOutcome, DeferredWorkflowCgroup, MAX_CGROUP_CLEANUP_RETRIES, WorkflowCgroup,
    cleanup_permits_spawn, poll_deferred_cgroups_with, transfer_workflow_cgroup_cleanup_with,
};
use super::{
    DeferredSignalState, DeferredWorkflowProcess, LinuxProcessIdentity, SharedDeferredReaper,
    SignalAuthority, SpawnedChildGuard, WorkflowChild, configure_process_group,
    poll_deferred_processes, signal_authority::ProvisionalGroupAuthority,
    test_raii::TestLocalDeferredReaper,
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
        #[cfg(target_os = "linux")]
        cgroups: Arc::new(std::sync::Mutex::new(Vec::new())),
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

#[cfg(target_os = "linux")]
#[test]
fn pending_cgroup_cleanup_keeps_admission_closed_until_confirmed_complete() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cgroup = Arc::new(WorkflowCgroup::for_test(PathBuf::from(format!(
        "/nonexistent/llm-guard-cgroup-pending-test-{}",
        std::process::id()
    ))));
    let mut cgroups = vec![DeferredWorkflowCgroup::new(
        cgroup,
        WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops))),
    )];
    let pending_cleanups = AtomicUsize::new(1);

    poll_deferred_cgroups_with(&mut cgroups, &pending_cleanups, |_| {
        DeferredPollOutcome::Pending
    });
    assert_eq!(pending_cleanups.load(Ordering::SeqCst), 1);
    assert!(!cleanup_permits_spawn(true, 1));
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    poll_deferred_cgroups_with(&mut cgroups, &pending_cleanups, |_| {
        DeferredPollOutcome::Complete
    });
    assert!(cgroups.is_empty());
    assert_eq!(pending_cleanups.load(Ordering::SeqCst), 0);
    assert!(cleanup_permits_spawn(true, 0));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn deferred_cgroup_cleanup_forces_after_bounded_retries_and_retains_lease() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cgroup = Arc::new(WorkflowCgroup::for_test(PathBuf::from(format!(
        "/nonexistent/llm-guard-cgroup-force-test-{}",
        std::process::id()
    ))));
    let mut cleanup = DeferredWorkflowCgroup::new(
        cgroup,
        WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops))),
    );
    let cleanup_calls = Cell::new(0_usize);
    let forced_calls = Cell::new(0_usize);
    let mut now = Instant::now() + Duration::from_secs(2);

    for _ in 0..MAX_CGROUP_CLEANUP_RETRIES {
        assert_eq!(
            cleanup.poll_with(
                now,
                || {
                    cleanup_calls.set(cleanup_calls.get() + 1);
                    Err(io::Error::other("injected cgroup cleanup failure"))
                },
                || panic!("forced cleanup ran before retry limit"),
            ),
            DeferredPollOutcome::Pending
        );
        now += Duration::from_secs(2);
    }
    assert_eq!(cleanup_calls.get(), MAX_CGROUP_CLEANUP_RETRIES);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    assert_eq!(
        cleanup.poll_with(
            now,
            || panic!("normal cleanup ran after retry limit"),
            || {
                forced_calls.set(forced_calls.get() + 1);
                Err(io::Error::other("injected forced cleanup failure"))
            },
        ),
        DeferredPollOutcome::Pending
    );
    now += Duration::from_secs(2);
    assert_eq!(
        cleanup.poll_with(
            now,
            || panic!("normal cleanup ran after retry limit"),
            || {
                forced_calls.set(forced_calls.get() + 1);
                Ok(())
            },
        ),
        DeferredPollOutcome::Complete
    );
    assert_eq!(forced_calls.get(), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(cleanup);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn unavailable_reaper_retains_pending_cgroup_lease() {
    let drops = Arc::new(AtomicUsize::new(0));
    let reaper = SharedDeferredReaper {
        processes: Arc::new(std::sync::Mutex::new(Vec::new())),
        cgroups: Arc::new(std::sync::Mutex::new(Vec::new())),
        pending_cleanups: Arc::new(AtomicUsize::new(0)),
        worker_available: false,
    };
    let cgroup = Arc::new(WorkflowCgroup::for_test(PathBuf::from(format!(
        "/nonexistent/llm-guard-cgroup-worker-test-{}",
        std::process::id()
    ))));

    reaper.submit_cgroup(DeferredWorkflowCgroup::new(
        cgroup,
        WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops))),
    ));

    assert_eq!(reaper.pending_cleanups.load(Ordering::SeqCst), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(!cleanup_permits_spawn(
        reaper.worker_available,
        reaper.pending_cleanups.load(Ordering::SeqCst)
    ));
    drop(reaper);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn failed_cgroup_cleanup_transfers_handle_and_execution_lease() {
    let drops = Arc::new(AtomicUsize::new(0));
    let execution_lease = WorkflowExecutionLease::new(DropProbe(Arc::clone(&drops)));
    let cgroup_path = std::env::temp_dir().join(format!(
        "llm-guard-cgroup-transfer-test-{}",
        std::process::id()
    ));
    let _stale_fixture = fs::remove_dir_all(&cgroup_path);
    fs::create_dir(&cgroup_path).expect("fake cgroup should be created");
    fs::write(cgroup_path.join("cgroup.kill"), "").expect("fake kill control should be created");
    fs::write(cgroup_path.join("cgroup.events"), "populated 0\n")
        .expect("fake events control should be created");
    fs::write(cgroup_path.join("removal-blocker"), "")
        .expect("fake cgroup removal should be blocked");
    let mut workflow_cgroup = Some(Arc::new(WorkflowCgroup::for_test(cgroup_path.clone())));
    let cleanup_result = workflow_cgroup
        .as_ref()
        .expect("fake cgroup should be owned")
        .kill_and_remove(Instant::now());
    assert!(cleanup_result.is_err(), "directory removal must fail");
    let captured = RefCell::new(None);

    if cleanup_result.is_err() {
        transfer_workflow_cgroup_cleanup_with(
            &mut workflow_cgroup,
            execution_lease.clone(),
            |cgroup, lease| {
                assert!(captured.borrow_mut().replace((cgroup, lease)).is_none());
            },
        );
    }

    assert!(workflow_cgroup.is_none());
    assert!(captured.borrow().is_some());
    drop(execution_lease);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    fs::remove_dir_all(cgroup_path).expect("fake cgroup should be removed before handle drop");
    drop(captured.into_inner());
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

fn spawn_sleep_group() -> (WorkflowChild, LinuxProcessIdentity) {
    let mut command = Command::new("/bin/sleep");
    command.arg("30");
    spawn_group(&mut command)
}

fn spawn_true_group() -> (WorkflowChild, LinuxProcessIdentity) {
    let mut command = Command::new("/bin/true");
    spawn_group(&mut command)
}

fn spawn_group(command: &mut Command) -> (WorkflowChild, LinuxProcessIdentity) {
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
