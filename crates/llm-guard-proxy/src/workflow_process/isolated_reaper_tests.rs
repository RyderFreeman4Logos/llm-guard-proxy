use std::{
    path::{Path, PathBuf},
    process::{Child, Command},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use nix::unistd::{Pid, setpgid};

use super::{
    DeferredSignalState, LinuxProcessIdentity, SignalAuthority, SpawnedChildGuard,
    configure_process_group, defer_workflow_process, linux_process_start_time,
    test_support::{
        PublishedProcessCleanup, TestProcessIdentity, publish_file_atomically,
        remove_atomic_marker, wait_for_published_identity,
    },
};

const SHARED_REAPER_HELPER_TEST: &str =
    "workflow_process::isolated_reaper_tests::shared_deferred_reaper_subprocess_helper";
const SHARED_REAPER_NESTED_TEST: &str =
    "workflow_process::isolated_reaper_tests::shared_reaper_nested_sleep_helper";
const SHARED_REAPER_HELPER_ENV: &str = "LLM_GUARD_SHARED_REAPER_HELPER";
const SHARED_REAPER_NESTED_ENV: &str = "LLM_GUARD_SHARED_REAPER_NESTED";
const SHARED_REAPER_IDENTITY_PATH_ENV: &str = "LLM_GUARD_SHARED_REAPER_IDENTITY_PATH";
const SHARED_REAPER_READY_PATH_ENV: &str = "LLM_GUARD_SHARED_REAPER_READY_PATH";
const SHARED_REAPER_ACK_PATH_ENV: &str = "LLM_GUARD_SHARED_REAPER_ACK_PATH";
const SHARED_REAPER_CLEANUP_IDENTITY_PATH_ENV: &str =
    "LLM_GUARD_SHARED_REAPER_CLEANUP_IDENTITY_PATH";
const SHARED_REAPER_PREIDENTITY_DELAY_ENV: &str = "LLM_GUARD_SHARED_REAPER_PREIDENTITY_DELAY_MS";
const ISOLATED_HELPER_TIMEOUT: Duration = Duration::from_secs(5);

#[test]
fn shared_deferred_reaper_isolated_subprocess_kills_and_reaps_transferred_process() {
    let deadline = Instant::now() + ISOLATED_HELPER_TIMEOUT;
    let nested_cleanup = PublishedNestedCleanup::new(deadline + super::PROCESS_REAP_GRACE);
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", SHARED_REAPER_HELPER_TEST, "--nocapture"])
        .env(SHARED_REAPER_HELPER_ENV, "1")
        .env(
            SHARED_REAPER_IDENTITY_PATH_ENV,
            &nested_cleanup.identity_path,
        )
        .env(SHARED_REAPER_READY_PATH_ENV, &nested_cleanup.ready_path)
        .env(SHARED_REAPER_ACK_PATH_ENV, &nested_cleanup.ack_path)
        .env(
            SHARED_REAPER_CLEANUP_IDENTITY_PATH_ENV,
            nested_cleanup.cleanup.marker_path(),
        );
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("shared reaper helper subprocess should start");
    let mut helper = SpawnedChildGuard::new(child, deadline + super::PROCESS_REAP_GRACE);
    let identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("isolated helper identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(identity));

    loop {
        let _identity = nested_cleanup.cleanup.refresh();
        match helper
            .child_mut()
            .expect("isolated helper child should remain armed")
            .try_wait()
        {
            Ok(Some(status)) => {
                helper.disarm();
                assert!(
                    status.success(),
                    "shared reaper helper should pass: {status}"
                );
                nested_cleanup.disarm_after_verified_exit();
                return;
            }
            Ok(None) => {
                assert!(
                    Instant::now() < deadline,
                    "shared reaper helper exceeded {ISOLATED_HELPER_TIMEOUT:?}"
                );
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("shared reaper helper status failed: {error}"),
        }
    }
}

#[test]
fn isolated_timeout_before_identity_publication_kills_inherited_nested_process() {
    let execution_deadline = Instant::now() + Duration::from_millis(200);
    let cleanup_deadline = Instant::now() + ISOLATED_HELPER_TIMEOUT;
    let nested_cleanup = PublishedNestedCleanup::new(cleanup_deadline);
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", SHARED_REAPER_HELPER_TEST, "--nocapture"])
        .env(SHARED_REAPER_HELPER_ENV, "1")
        .env(
            SHARED_REAPER_IDENTITY_PATH_ENV,
            &nested_cleanup.identity_path,
        )
        .env(SHARED_REAPER_READY_PATH_ENV, &nested_cleanup.ready_path)
        .env(SHARED_REAPER_ACK_PATH_ENV, &nested_cleanup.ack_path)
        .env(
            SHARED_REAPER_CLEANUP_IDENTITY_PATH_ENV,
            nested_cleanup.cleanup.marker_path(),
        )
        .env(SHARED_REAPER_PREIDENTITY_DELAY_ENV, "1000");
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("delayed shared reaper helper should start");
    let mut helper = SpawnedChildGuard::new(child, execution_deadline + super::PROCESS_REAP_GRACE);
    let identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("delayed helper identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(identity));

    let nested_identity = nested_cleanup
        .cleanup
        .wait_for_identity(cleanup_deadline)
        .expect("nested exact cleanup identity should be published before the delay");
    assert!(nested_identity.is_live());
    while Instant::now() < execution_deadline {
        thread::sleep(super::PROCESS_REAP_POLL);
    }

    drop(helper);

    assert!(nested_identity.wait_until_not_live(cleanup_deadline));
    nested_cleanup.cleanup.disarm_after_verified_exit();
    nested_cleanup.remove_markers();
}

#[test]
fn shared_deferred_reaper_subprocess_helper() {
    if std::env::var_os(SHARED_REAPER_HELPER_ENV).is_none() {
        return;
    }
    let deadline = Instant::now() + ISOLATED_HELPER_TIMEOUT;
    let ready_path = helper_path(SHARED_REAPER_READY_PATH_ENV);
    let ack_path = helper_path(SHARED_REAPER_ACK_PATH_ENV);
    let mut command =
        Command::new(std::env::current_exe().expect("nested test executable should resolve"));
    command
        .args(["--exact", SHARED_REAPER_NESTED_TEST, "--nocapture"])
        .env(SHARED_REAPER_NESTED_ENV, "1")
        .env(SHARED_REAPER_READY_PATH_ENV, &ready_path)
        .env(SHARED_REAPER_ACK_PATH_ENV, &ack_path);
    // Inherit the isolated helper's PGID until exact identity has been published. If the
    // supervisor times out before publication, one strict group signal still kills both.
    let child = command.spawn().expect("nested sleep fixture should start");
    let mut inherited_child = InheritedGroupChildGuard::new(child, deadline);
    let pid = inherited_child.pid();
    let cleanup_identity = TestProcessIdentity::capture(pid)
        .expect("nested exact cleanup identity should be captured");
    cleanup_identity.publish(&helper_path(SHARED_REAPER_CLEANUP_IDENTITY_PATH_ENV));
    if let Some(delay) = std::env::var_os(SHARED_REAPER_PREIDENTITY_DELAY_ENV)
        .and_then(|value| value.to_string_lossy().parse::<u64>().ok())
    {
        thread::sleep(Duration::from_millis(delay));
    }
    let identity = LinuxProcessIdentity::capture(pid)
        .expect("transferred fixture identity should be captured");
    assert_eq!(identity.pid, cleanup_identity.pid.get());
    assert_eq!(
        identity.start_time_ticks,
        cleanup_identity.start_time_ticks.get()
    );
    cleanup_identity.publish(&helper_path(SHARED_REAPER_IDENTITY_PATH_ENV));
    publish_file_atomically(&ready_path, b"ready");
    wait_for_path(&ack_path, deadline);

    let mut child = SpawnedChildGuard::new(
        inherited_child
            .disarm()
            .expect("inherited nested child guard should remain armed"),
        deadline,
    );
    child.set_signal_authority(SignalAuthority::new(identity));
    defer_workflow_process(
        child
            .disarm()
            .expect("nested child guard should remain armed"),
        DeferredSignalState::SignalPending(SignalAuthority::new(identity)),
    );
    while Instant::now() < deadline {
        if linux_process_start_time(pid).is_err() {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }

    panic!("shared deferred reaper did not reap process {pid}");
}

#[test]
fn shared_reaper_nested_sleep_helper() {
    if std::env::var_os(SHARED_REAPER_NESTED_ENV).is_none() {
        return;
    }
    let ready_path = helper_path(SHARED_REAPER_READY_PATH_ENV);
    let ack_path = helper_path(SHARED_REAPER_ACK_PATH_ENV);
    wait_for_path(&ready_path, Instant::now() + ISOLATED_HELPER_TIMEOUT);
    setpgid(Pid::from_raw(0), Pid::from_raw(0))
        .expect("nested process should establish its private process group");
    publish_file_atomically(&ack_path, b"private-group");
    thread::sleep(Duration::from_secs(30));
}

struct PublishedNestedCleanup {
    identity_path: PathBuf,
    ready_path: PathBuf,
    ack_path: PathBuf,
    cleanup: PublishedProcessCleanup,
    cleanup_deadline: Instant,
}

impl PublishedNestedCleanup {
    fn new(cleanup_deadline: Instant) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "llm-guard-shared-reaper-{}-{nonce}",
            std::process::id()
        ));
        let cleanup_identity_path = base.with_extension("cleanup-identity");
        Self {
            identity_path: base.with_extension("transfer-identity"),
            ready_path: base.with_extension("ready"),
            ack_path: base.with_extension("ack"),
            cleanup: PublishedProcessCleanup::new(cleanup_identity_path, cleanup_deadline),
            cleanup_deadline,
        }
    }

    fn disarm_after_verified_exit(&self) {
        let cleanup_identity = self
            .cleanup
            .published_identity()
            .expect("successful helper should publish its nested process identity");
        let transfer_identity =
            wait_for_published_identity(&self.identity_path, self.cleanup_deadline)
                .expect("successful helper should publish its transfer identity atomically");
        assert_eq!(cleanup_identity, transfer_identity);
        self.cleanup.disarm_after_verified_exit();
        self.remove_markers();
    }

    fn remove_markers(&self) {
        for path in [&self.identity_path, &self.ready_path, &self.ack_path] {
            remove_atomic_marker(path);
        }
    }
}

impl Drop for PublishedNestedCleanup {
    fn drop(&mut self) {
        self.remove_markers();
    }
}

struct InheritedGroupChildGuard {
    child: Option<Child>,
    cleanup_deadline: Instant,
}

impl InheritedGroupChildGuard {
    const fn new(child: Child, cleanup_deadline: Instant) -> Self {
        Self {
            child: Some(child),
            cleanup_deadline,
        }
    }

    fn pid(&self) -> u32 {
        self.child.as_ref().map_or(0, Child::id)
    }

    fn disarm(&mut self) -> Option<Child> {
        self.child.take()
    }
}

impl Drop for InheritedGroupChildGuard {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _kill = child.kill();
        loop {
            match child.try_wait() {
                Ok(None) if Instant::now() < self.cleanup_deadline => {
                    thread::sleep(super::PROCESS_REAP_POLL);
                }
                Ok(Some(_) | None) | Err(_) => return,
            }
        }
    }
}

fn helper_path(environment_name: &str) -> PathBuf {
    std::env::var_os(environment_name).map_or_else(
        || panic!("{environment_name} should be configured"),
        PathBuf::from,
    )
}

fn wait_for_path(path: &Path, deadline: Instant) {
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
        thread::sleep(super::PROCESS_REAP_POLL);
    }
}
