use std::{
    cell::{Cell, RefCell},
    fs,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use llm_guard_proxy_core::{WorkflowConfig, WorkflowRuntime};
use nix::{errno::Errno, sys::signal::Signal};

use super::{
    DeferredPollOutcome, DeferredSignalState, LinuxProcessIdentity, NonReapingChildState,
    ProcessGroupSignalError, SignalAuthority, WorkflowProcess, WorkflowProcessStartError,
    WorkflowSignalOutcome, cleanup_permits_spawn, cleanup_worker_available,
    configure_process_group, finalize_provisional_child_with, linux_process_is_live,
    linux_process_start_time, poll_deferred_cleanup_with,
    signal_authority::ProvisionalGroupAuthority,
    signal_owned_workflow,
    test_support::{
        PublishedProcessCleanup, TestProcessIdentity, wait_for_current_published_identity,
    },
};

const STARTUP_AUTHORITY_HELPER_TEST: &str =
    "workflow_process::startup_authority_tests::unavailable_startup_identity_subprocess_helper";
const STARTUP_AUTHORITY_HELPER_ENV: &str = "LLM_GUARD_STARTUP_AUTHORITY_HELPER";
const STARTUP_AUTHORITY_CLEANUP_IDENTITY_ENV: &str = "LLM_GUARD_STARTUP_AUTHORITY_CLEANUP_IDENTITY";
const STARTUP_AUTHORITY_CLEANUP_DESCENDANT_ENV: &str =
    "LLM_GUARD_STARTUP_AUTHORITY_CLEANUP_DESCENDANT";
const STARTUP_AUTHORITY_RELEASE_ENV: &str = "LLM_GUARD_STARTUP_AUTHORITY_RELEASE";
const STARTUP_AUTHORITY_HELPER_TIMEOUT: Duration = Duration::from_secs(5);

#[test]
fn unavailable_startup_identity_cleanup_kills_same_group_descendant_before_admission_reopens() {
    let deadline = Instant::now() + STARTUP_AUTHORITY_HELPER_TIMEOUT;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow Unix epoch")
        .as_nanos();
    let marker_path = std::env::temp_dir().join(format!(
        "llm-guard-startup-authority-helper-{}-{nonce}.identity",
        std::process::id()
    ));
    let descendant_marker_path = PathBuf::from(format!("{}.descendant", marker_path.display()));
    let release_path = PathBuf::from(format!("{}.release", marker_path.display()));
    let nested_cleanup = PublishedProcessCleanup::new_process_group_with_descendants(
        marker_path.clone(),
        deadline + Duration::from_secs(2),
        vec![descendant_marker_path.clone()],
    );
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", STARTUP_AUTHORITY_HELPER_TEST, "--nocapture"])
        .env(STARTUP_AUTHORITY_HELPER_ENV, "1")
        .env(STARTUP_AUTHORITY_CLEANUP_IDENTITY_ENV, marker_path)
        .env(
            STARTUP_AUTHORITY_CLEANUP_DESCENDANT_ENV,
            descendant_marker_path,
        )
        .env(STARTUP_AUTHORITY_RELEASE_ENV, &release_path);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("startup authority helper should spawn");
    let mut helper = super::SpawnedChildGuard::new(child, deadline);
    let identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("startup authority helper identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(identity));
    wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
        .expect("nested startup leader should publish before scenario release");
    super::test_support::publish_file_atomically(&release_path, b"release");

    loop {
        match helper
            .child_mut()
            .expect("startup authority helper should remain armed")
            .try_wait()
        {
            Ok(Some(status)) => {
                helper.disarm();
                assert!(
                    status.success(),
                    "startup authority helper failed: {status}"
                );
                nested_cleanup.disarm_after_verified_exit();
                return;
            }
            Ok(None) => {
                assert!(
                    Instant::now() < deadline,
                    "startup authority helper exceeded {STARTUP_AUTHORITY_HELPER_TIMEOUT:?}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("startup authority helper status failed: {error}"),
        }
    }
}

#[test]
fn recursive_startup_fixture_killed_before_release_leaves_no_nested_child() {
    let deadline = Instant::now() + Duration::from_secs(3);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let marker_path = std::env::temp_dir().join(format!(
        "llm-guard-startup-pre-release-{}-{nonce}.identity",
        std::process::id()
    ));
    let descendant_path = PathBuf::from(format!("{}.descendant", marker_path.display()));
    let release_path = PathBuf::from(format!("{}.release", marker_path.display()));
    let nested_cleanup = PublishedProcessCleanup::new_process_group(
        marker_path.clone(),
        deadline + Duration::from_secs(2),
    );
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", STARTUP_AUTHORITY_HELPER_TEST, "--nocapture"])
        .env(STARTUP_AUTHORITY_HELPER_ENV, "1")
        .env(STARTUP_AUTHORITY_CLEANUP_IDENTITY_ENV, marker_path)
        .env(STARTUP_AUTHORITY_CLEANUP_DESCENDANT_ENV, descendant_path)
        .env(STARTUP_AUTHORITY_RELEASE_ENV, release_path);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("pre-release startup scenario should spawn");
    let mut helper = super::SpawnedChildGuard::new(child, deadline);
    let helper_identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("pre-release startup scenario identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(helper_identity));
    let nested_identity =
        wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
            .expect("supervisor should publish startup leader before release");

    drop(helper);
    drop(nested_cleanup);

    assert!(
        nested_identity.wait_until_not_live(Instant::now() + Duration::from_secs(1)),
        "nested startup pre-release child must not remain"
    );
}

#[test]
fn unavailable_startup_identity_subprocess_helper() {
    if std::env::var_os(STARTUP_AUTHORITY_HELPER_ENV).is_none() {
        return;
    }
    let fixture = StartupGroupFixture::new();
    let config = fixture.config();

    let result = WorkflowProcess::start_with_identity_capture(
        &config,
        Duration::from_millis(30),
        |leader_pid| {
            fixture.record_leader(leader_pid);
            fixture.await_and_record_descendant();
            Err(ProcessGroupSignalError::IdentityUnavailable)
        },
    );

    assert!(matches!(
        result,
        Err(WorkflowProcessStartError::IdentityCapture(
            "identity_unavailable"
        ))
    ));
    let leader = fixture
        .leader_identity
        .get()
        .expect("startup fixture leader identity should be recorded");
    let descendant = fixture
        .descendant_identity
        .get()
        .expect("startup fixture descendant identity should be recorded");
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let descendant_live = identity_is_live(descendant);
        let admission_open = cleanup_worker_available();
        assert!(
            !admission_open || !descendant_live,
            "admission reopened while same-PGID descendant {} remained live",
            descendant.pid
        );
        if admission_open {
            assert!(!identity_is_live(leader));
            assert!(!descendant_live);
            fixture.disarm();
            return;
        }
        assert!(
            Instant::now() < deadline,
            "startup cleanup did not restore admission within its deferred bound"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn provisional_non_strict_signal_retains_unreaped_leader_and_blocks_admission() {
    for signal_result in [
        Err(ProcessGroupSignalError::SignalFailed),
        Ok(WorkflowSignalOutcome::LeaderOnly),
    ] {
        let (mut child, identity) = spawn_sleep_group();
        let captured = CapturedDeferred::new(identity);
        let signal_calls = Cell::new(0_usize);

        let result = finalize_provisional_child_with(
            child
                .disarm()
                .expect("startup authority guard should transfer the fixture"),
            ProvisionalGroupAuthority::new(identity.pid),
            Instant::now() + Duration::from_secs(1),
            |_authority| Ok(NonReapingChildState::Running),
            |_authority| {
                signal_calls.set(signal_calls.get() + 1);
                signal_result
            },
            |child, state| captured.store(child, state),
        );

        assert!(result.is_err());
        assert_eq!(signal_calls.get(), 1);
        assert!(matches!(
            captured.signal_state().as_deref(),
            Some(DeferredSignalState::ProvisionalSignalPending(_))
        ));
        assert_eq!(
            captured
                .try_wait()
                .expect("retained child state should be readable"),
            None,
            "non-strict workflow signal must not reap the leader"
        );
        assert!(!cleanup_permits_spawn(true, 1));
    }
}

#[test]
fn deferred_provisional_cleanup_retries_group_signal_before_reaping() {
    let mut signal_state = DeferredSignalState::ProvisionalSignalPending(
        ProvisionalGroupAuthority::new(std::process::id()),
    );
    let reap_calls = Cell::new(0_usize);

    let first = poll_deferred_cleanup_with(
        &mut signal_state,
        || Ok(NonReapingChildState::Running),
        || Err(ProcessGroupSignalError::SignalFailed),
        || {
            reap_calls.set(reap_calls.get() + 1);
            Ok(true)
        },
    );

    assert_eq!(first, DeferredPollOutcome::Pending);
    assert_eq!(reap_calls.get(), 0);
    assert!(matches!(
        signal_state,
        DeferredSignalState::ProvisionalSignalPending(_)
    ));

    let second = poll_deferred_cleanup_with(
        &mut signal_state,
        || Ok(NonReapingChildState::Exited),
        || Ok(WorkflowSignalOutcome::StrictGroup),
        || {
            reap_calls.set(reap_calls.get() + 1);
            Ok(true)
        },
    );

    assert_eq!(second, DeferredPollOutcome::Complete);
    assert_eq!(reap_calls.get(), 1);
    assert!(matches!(
        signal_state,
        DeferredSignalState::StrictGroupSignaled
    ));
}

#[test]
fn provisional_echild_revokes_without_signal_or_direct_leader_kill() {
    assert_provisional_ownership_loss_does_not_signal(false);
}

#[test]
fn provisional_identity_mismatch_revokes_without_signal_or_direct_leader_kill() {
    assert_provisional_ownership_loss_does_not_signal(true);
}

fn assert_provisional_ownership_loss_does_not_signal(revoke_before_observation: bool) {
    let (mut child, identity) = spawn_sleep_group();
    let captured = CapturedDeferred::new(identity);
    let observation_calls = Cell::new(0_usize);
    let signal_calls = Cell::new(0_usize);
    let mut authority = ProvisionalGroupAuthority::new(identity.pid);
    if revoke_before_observation {
        authority.revoke();
    }

    let result = finalize_provisional_child_with(
        child
            .disarm()
            .expect("startup authority guard should transfer the fixture"),
        authority,
        Instant::now() + Duration::from_secs(1),
        |_authority| {
            observation_calls.set(observation_calls.get() + 1);
            Err(if revoke_before_observation {
                ProcessGroupSignalError::IdentityMismatch
            } else {
                ProcessGroupSignalError::OwnershipLost
            })
        },
        |_authority| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(WorkflowSignalOutcome::StrictGroup)
        },
        |child, state| captured.store(child, state),
    );

    assert!(result.is_err());
    assert_eq!(signal_calls.get(), 0);
    assert_eq!(
        observation_calls.get(),
        usize::from(!revoke_before_observation)
    );
    assert!(matches!(
        captured.signal_state().as_deref(),
        Some(DeferredSignalState::Unresolved)
    ));
    assert_eq!(
        captured
            .try_wait()
            .expect("ownership-loss child state should be readable"),
        None,
        "ownership loss must not call Child::kill or reap the leader"
    );
}

fn spawn_sleep_group() -> (super::SpawnedChildGuard, LinuxProcessIdentity) {
    let mut command = Command::new("/bin/sleep");
    command.arg("30");
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("startup authority fixture should spawn");
    let mut child =
        super::SpawnedChildGuard::new(child, Instant::now() + super::PROCESS_REAP_GRACE);
    let identity = LinuxProcessIdentity::capture(child.pid())
        .expect("startup authority fixture identity should be captured");
    child.set_signal_authority(SignalAuthority::new(identity));
    (child, identity)
}

struct CapturedDeferred {
    identity: LinuxProcessIdentity,
    process: RefCell<Option<(Child, DeferredSignalState)>>,
}

impl CapturedDeferred {
    fn new(identity: LinuxProcessIdentity) -> Self {
        Self {
            identity,
            process: RefCell::new(None),
        }
    }

    fn store(&self, child: Child, state: DeferredSignalState) {
        assert!(self.process.borrow_mut().replace((child, state)).is_none());
    }

    fn signal_state(&self) -> Option<std::cell::Ref<'_, DeferredSignalState>> {
        std::cell::Ref::filter_map(self.process.borrow(), |process| {
            process.as_ref().map(|(_child, state)| state)
        })
        .ok()
    }

    fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.process
            .borrow_mut()
            .as_mut()
            .expect("deferred process should be captured")
            .0
            .try_wait()
    }
}

impl Drop for CapturedDeferred {
    fn drop(&mut self) {
        let Some((mut child, _state)) = self.process.get_mut().take() else {
            return;
        };
        let authority = SignalAuthority::new(self.identity);
        let _signal = signal_owned_workflow(&authority, Signal::SIGKILL);
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    }
}

struct StartupGroupFixture {
    directory: PathBuf,
    descendant_identity_path: PathBuf,
    cleanup_identity_path: Option<PathBuf>,
    cleanup_descendant_identity_path: Option<PathBuf>,
    release_path: Option<PathBuf>,
    leader_identity: Cell<Option<LinuxProcessIdentity>>,
    descendant_identity: Cell<Option<LinuxProcessIdentity>>,
}

impl StartupGroupFixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "llm-guard-startup-authority-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("startup fixture directory should be created");
        let descendant_identity_path = directory.join("descendant.identity");
        Self {
            directory,
            descendant_identity_path,
            cleanup_identity_path: std::env::var_os(STARTUP_AUTHORITY_CLEANUP_IDENTITY_ENV)
                .map(PathBuf::from),
            cleanup_descendant_identity_path: std::env::var_os(
                STARTUP_AUTHORITY_CLEANUP_DESCENDANT_ENV,
            )
            .map(PathBuf::from),
            release_path: std::env::var_os(STARTUP_AUTHORITY_RELEASE_ENV).map(PathBuf::from),
            leader_identity: Cell::new(None),
            descendant_identity: Cell::new(None),
        }
    }

    fn config(&self) -> WorkflowConfig {
        WorkflowConfig {
            runtime_kind: WorkflowRuntime::Stdio,
            command: String::from("/bin/sh"),
            args: vec![
                String::from("-c"),
                String::from(
                    "release=$2; attempts=0; while [ ! -f \"$release\" ]; do attempts=$((attempts + 1)); if [ \"$attempts\" -ge 1000 ]; then exit 91; fi; sleep 0.005; done; rm -f \"$release\"; sleep 30 & child=$!; start=$(awk '{print $22}' /proc/$child/stat); marker=$1; tmp=\"${marker}.tmp.$$\"; if [ \"$child\" -le 0 ] || [ -z \"$start\" ] || [ \"$start\" -le 0 ]; then exit 90; fi; printf '%s %s\\n' \"$child\" \"$start\" > \"$tmp\"; mv -f \"$tmp\" \"$marker\"; wait",
                ),
                String::from("workflow-startup-authority"),
                self.descendant_identity_path.display().to_string(),
                self.release_path
                    .as_deref()
                    .expect("recursive startup fixture should configure release path")
                    .display()
                    .to_string(),
            ],
            timeout_ms: 30,
            max_stdout_bytes: 4096,
        }
    }

    fn record_leader(&self, pid: u32) {
        if self.leader_identity.get().is_none() {
            let identity = LinuxProcessIdentity::capture(pid).ok();
            if let (Some(path), Some(test_identity)) = (
                self.cleanup_identity_path.as_deref(),
                TestProcessIdentity::capture(pid),
            ) {
                test_identity.publish(path);
            }
            self.leader_identity.set(identity);
        }
    }

    fn await_and_record_descendant(&self) {
        if self.descendant_identity.get().is_some() {
            return;
        }
        let deadline = Instant::now() + Duration::from_millis(300);
        let published =
            wait_for_current_published_identity(&self.descendant_identity_path, deadline)
                .expect("same-PGID descendant should publish a current exact identity");
        if let Some(path) = self.cleanup_descendant_identity_path.as_deref() {
            published.publish(path);
        }
        self.descendant_identity.set(Some(LinuxProcessIdentity {
            pid: published.pid.get(),
            start_time_ticks: published.start_time_ticks.get(),
        }));
    }

    fn disarm(&self) {
        self.leader_identity.set(None);
        self.descendant_identity.set(None);
    }
}

impl Drop for StartupGroupFixture {
    fn drop(&mut self) {
        for identity in [self.descendant_identity.get(), self.leader_identity.get()]
            .into_iter()
            .flatten()
        {
            let authority = SignalAuthority::new(identity);
            let _signal = signal_owned_workflow(&authority, Signal::SIGKILL);
        }
        let _remove = fs::remove_dir_all(&self.directory);
    }
}

fn identity_is_live(identity: LinuxProcessIdentity) -> bool {
    linux_process_start_time(identity.pid).ok() == Some(identity.start_time_ticks)
        && linux_process_is_live(identity.pid)
}

#[test]
fn provisional_echild_seam_maps_waitid_loss_before_any_signal() {
    let mut authority = ProvisionalGroupAuthority::new(std::process::id());
    let signal_calls = Cell::new(0_usize);

    let observation = authority
        .observe_child_nonreaping_with(|_pid| Err(Errno::ECHILD))
        .expect("ECHILD should map to ownership loss");
    let signal = authority.signal_owned_workflow_with(
        Signal::SIGKILL,
        |_pid, _signal| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(())
        },
        |_pid, _signal| {
            signal_calls.set(signal_calls.get() + 1);
            Ok(())
        },
    );

    assert_eq!(observation, NonReapingChildState::OwnershipLost);
    assert_eq!(signal, Err(ProcessGroupSignalError::OwnershipLost));
    assert_eq!(signal_calls.get(), 0);
}
