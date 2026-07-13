use std::{
    cell::Cell,
    fs,
    os::unix::fs::symlink,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use llm_guard_proxy_core::{WorkflowConfig, WorkflowRuntime};

use super::{
    LinuxProcessIdentity, SignalAuthority, SpawnedChildGuard, WorkflowProcess,
    cleanup_and_finish_drain_with, configure_process_group,
    test_support::{
        PublishedProcessCleanup, TestProcessIdentity, publish_file_atomically,
        wait_for_current_published_identity,
    },
};

const PANIC_HELPER_TEST: &str = "workflow_process::lifecycle_tests::panic_cleanup_helper";
const PANIC_SCENARIO_HELPER_TEST: &str =
    "workflow_process::lifecycle_tests::panic_cleanup_subprocess_helper";
const PANIC_SCENARIO_DIRECTORY_ENV: &str = "LLM_GUARD_PANIC_SCENARIO_DIRECTORY";

#[test]
fn panic_after_start_kills_and_reaps_owned_child() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow Unix epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "llm-guard-workflow-panic-scenario-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&directory).expect("panic scenario directory should be created");
    let _directory_cleanup = TestDirectoryCleanup(directory.clone());
    let nested_cleanup = PublishedProcessCleanup::new_process_group(
        directory.join("panic-child.identity"),
        deadline + Duration::from_secs(2),
    );
    let release_path = directory.join("panic-child.release");

    let mut command =
        Command::new(std::env::current_exe().expect("current test executable should resolve"));
    command
        .args(["--exact", PANIC_SCENARIO_HELPER_TEST, "--nocapture"])
        .env(PANIC_SCENARIO_DIRECTORY_ENV, &directory);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("panic cleanup scenario helper should spawn");
    let mut helper = SpawnedChildGuard::new(child, deadline);
    let helper_identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("panic cleanup scenario helper identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(helper_identity));
    let published = wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
        .expect("nested panic helper should publish before scenario release");
    assert!(published.is_current());
    publish_file_atomically(&release_path, b"release");

    loop {
        match helper
            .child_mut()
            .expect("panic cleanup scenario helper should remain armed")
            .try_wait()
        {
            Ok(Some(status)) => {
                helper.disarm();
                assert!(status.success(), "panic cleanup scenario failed: {status}");
                nested_cleanup.disarm_after_verified_exit();
                return;
            }
            Ok(None) => {
                assert!(
                    Instant::now() < deadline,
                    "panic cleanup scenario exceeded its fixture deadline"
                );
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("panic cleanup scenario status failed: {error}"),
        }
    }
}

#[test]
fn recursive_panic_fixture_killed_before_release_leaves_no_nested_child() {
    let deadline = Instant::now() + Duration::from_secs(3);
    let directory = std::env::temp_dir().join(format!(
        "llm-guard-workflow-panic-pre-release-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir(&directory).expect("pre-release fixture directory should be created");
    let _directory_cleanup = TestDirectoryCleanup(directory.clone());
    let nested_cleanup = PublishedProcessCleanup::new_process_group(
        directory.join("panic-child.identity"),
        deadline + Duration::from_secs(2),
    );
    let mut command =
        Command::new(std::env::current_exe().expect("current test executable should resolve"));
    command
        .args(["--exact", PANIC_SCENARIO_HELPER_TEST, "--nocapture"])
        .env(PANIC_SCENARIO_DIRECTORY_ENV, &directory);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("pre-release panic scenario should spawn");
    let mut helper = SpawnedChildGuard::new(child, deadline);
    let helper_identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("pre-release scenario identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(helper_identity));
    let nested_identity =
        wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
            .expect("supervisor should publish nested identity before release");

    drop(helper);
    drop(nested_cleanup);

    assert!(
        nested_identity.wait_until_not_live(Instant::now() + Duration::from_secs(1)),
        "nested pre-release child must not remain as a 30-second fixture"
    );
}

#[test]
fn panic_cleanup_subprocess_helper() {
    let Some(directory) = std::env::var_os(PANIC_SCENARIO_DIRECTORY_ENV).map(PathBuf::from) else {
        return;
    };
    let identity_path = directory.join("panic-child.identity");
    let fixture = PanicHelperFixture::new(&directory);
    let process_identity = Cell::new(None);
    let deadline = Instant::now() + Duration::from_secs(1);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Ok(process) = WorkflowProcess::start(&fixture.config(), Duration::from_secs(30)) else {
            panic!("workflow helper should start");
        };
        let pid = process.pid().expect("armed workflow should own a child");
        let identity = TestProcessIdentity::capture(pid)
            .expect("panic fixture identity should be captured before unwinding");
        process_identity.set(Some(identity));
        identity.publish(&identity_path);
        let published_identity = wait_for_current_published_identity(&identity_path, deadline)
            .expect("panic helper should publish its exact identity before injected unwind");
        assert_eq!(published_identity, identity);
        panic!("injected unwind after WorkflowProcess::start");
    }));

    assert!(result.is_err());
    let process_identity = process_identity
        .get()
        .expect("panic fixture should record a nonzero process identity");
    assert!(
        process_identity.wait_until_not_live(deadline),
        "WorkflowProcess::drop should reap its exact child identity after unwind"
    );
}

#[test]
fn cleanup_failure_cancels_and_bounds_stderr_drain_finish() {
    let finished = Cell::new(false);
    let cancelled = Cell::new(false);
    let cleanup_deadline = Instant::now() + Duration::from_millis(50);

    let result = cleanup_and_finish_drain_with(
        || Err::<(), _>(std::io::Error::other("injected cleanup transfer")),
        Some(()),
        cleanup_deadline,
        |(), observed_deadline, cancel| {
            finished.set(true);
            cancelled.set(cancel);
            assert_eq!(observed_deadline, cleanup_deadline);
            Ok(Vec::new())
        },
    );

    assert!(result.is_err());
    assert!(
        finished.get(),
        "failed cleanup must finish the drain bounded"
    );
    assert!(
        cancelled.get(),
        "failed cleanup must cancel before finishing"
    );
}

#[test]
fn panic_cleanup_helper() {
    let argv0 = std::env::args_os()
        .next()
        .map(PathBuf::from)
        .expect("panic helper argv[0] should exist");
    let is_fixture = argv0
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("workflow-panic-helper"));
    if !is_fixture {
        return;
    }
    wait_for_path(
        &argv0
            .parent()
            .expect("panic helper should live in its fixture directory")
            .join("panic-child.release"),
        Instant::now() + Duration::from_secs(5),
    );
    let _removed = fs::remove_file(
        argv0
            .parent()
            .expect("panic helper should live in its fixture directory")
            .join("panic-child.release"),
    );
    thread::sleep(Duration::from_secs(30));
}

fn wait_for_path(path: &Path, deadline: Instant) {
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
        thread::sleep(Duration::from_millis(5));
    }
}

struct PanicHelperFixture {
    executable: PathBuf,
}

impl PanicHelperFixture {
    fn new(directory: &Path) -> Self {
        let executable = directory.join("workflow-panic-helper");
        symlink(
            std::env::current_exe().expect("current test executable should resolve"),
            &executable,
        )
        .expect("panic helper symlink should be created");
        Self { executable }
    }

    fn config(&self) -> WorkflowConfig {
        WorkflowConfig {
            runtime_kind: WorkflowRuntime::Stdio,
            command: self.executable.display().to_string(),
            args: vec![
                String::from("--exact"),
                String::from(PANIC_HELPER_TEST),
                String::from("--nocapture"),
            ],
            timeout_ms: 30_000,
            max_stdout_bytes: 4096,
        }
    }
}

struct TestDirectoryCleanup(PathBuf);

impl Drop for TestDirectoryCleanup {
    fn drop(&mut self) {
        let _remove = fs::remove_dir_all(&self.0);
    }
}
