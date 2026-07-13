use std::{
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use llm_guard_proxy_core::{WorkflowConfig, WorkflowRuntime};
use nix::unistd::{Pid, getpgid, getpgrp, setpgid};

use super::{
    LinuxProcessIdentity, SignalAuthority, SpawnedChildGuard, WorkflowProcess,
    WorkflowProcessFinishError, cleanup_worker_available, configure_process_group,
    test_support::{
        PublishedProcessCleanup, TestProcessIdentity, publish_file_atomically,
        wait_for_current_published_identity,
    },
};

const SCENARIO_TEST: &str =
    "workflow_process::moved_pgid_tests::moved_pgid_timeout_scenario_helper";
const LEADER_TEST: &str = "workflow_process::moved_pgid_tests::moved_pgid_leader_helper";
const SCENARIO_ENV: &str = "LLM_GUARD_MOVED_PGID_SCENARIO";
const DIRECTORY_ENV: &str = "LLM_GUARD_MOVED_PGID_DIRECTORY";
const HELPER_TIMEOUT: Duration = Duration::from_secs(5);
const WORKFLOW_TIMEOUT: Duration = Duration::from_millis(500);

#[test]
fn moved_pgid_leader_timeout_kills_exact_leader_and_reopens_admission() {
    let directory = unique_directory();
    fs::create_dir(&directory).expect("moved-PGID fixture directory should be created");
    let _directory_cleanup = TestDirectoryCleanup(directory.clone());
    let deadline = Instant::now() + HELPER_TIMEOUT;
    let nested_cleanup = PublishedProcessCleanup::new(
        directory.join("leader.identity"),
        deadline + Duration::from_secs(2),
    );
    let release_path = directory.join("leader.release");
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", SCENARIO_TEST, "--nocapture"])
        .env(SCENARIO_ENV, "1")
        .env(DIRECTORY_ENV, &directory);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("moved-PGID scenario helper should spawn");
    let mut helper = SpawnedChildGuard::new(child, deadline);
    let helper_identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("moved-PGID scenario identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(helper_identity));
    wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
        .expect("nested moved-PGID leader should publish before scenario release");
    publish_file_atomically(&release_path, b"release");

    loop {
        match helper
            .child_mut()
            .expect("moved-PGID scenario helper should remain armed")
            .try_wait()
        {
            Ok(Some(status)) => {
                helper.disarm();
                assert!(status.success(), "moved-PGID scenario failed: {status}");
                nested_cleanup.disarm_after_verified_exit();
                return;
            }
            Ok(None) => {
                assert!(Instant::now() < deadline, "moved-PGID scenario timed out");
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("moved-PGID scenario status failed: {error}"),
        }
    }
}

#[test]
fn recursive_moved_pgid_fixture_killed_before_release_leaves_no_nested_child() {
    let directory = unique_directory();
    fs::create_dir(&directory).expect("pre-release moved-PGID directory should be created");
    let _directory_cleanup = TestDirectoryCleanup(directory.clone());
    let deadline = Instant::now() + Duration::from_secs(3);
    let nested_cleanup = PublishedProcessCleanup::new(
        directory.join("leader.identity"),
        deadline + Duration::from_secs(2),
    );
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .args(["--exact", SCENARIO_TEST, "--nocapture"])
        .env(SCENARIO_ENV, "1")
        .env(DIRECTORY_ENV, &directory);
    configure_process_group(&mut command);
    let child = command
        .spawn()
        .expect("pre-release moved-PGID scenario should spawn");
    let mut helper = SpawnedChildGuard::new(child, deadline);
    let helper_identity = LinuxProcessIdentity::capture(helper.pid())
        .expect("pre-release moved-PGID scenario identity should be captured");
    helper.set_signal_authority(SignalAuthority::new(helper_identity));
    let nested_identity =
        wait_for_current_published_identity(nested_cleanup.marker_path(), deadline)
            .expect("supervisor should publish moved leader before release");

    drop(helper);
    drop(nested_cleanup);

    assert!(
        nested_identity.wait_until_not_live(Instant::now() + Duration::from_secs(1)),
        "nested moved-PGID pre-release child must not remain"
    );
}

#[test]
fn moved_pgid_timeout_scenario_helper() {
    if std::env::var_os(SCENARIO_ENV).is_none() {
        return;
    }
    let directory = PathBuf::from(
        std::env::var_os(DIRECTORY_ENV).expect("moved-PGID directory should be configured"),
    );
    let target_pgid = getpgrp();
    let executable = directory.join(format!("workflow-moved-pgid-leader-{target_pgid}"));
    symlink(
        std::env::current_exe().expect("test executable should resolve"),
        &executable,
    )
    .expect("moved-PGID leader symlink should be created");
    let config = WorkflowConfig {
        runtime_kind: WorkflowRuntime::Stdio,
        command: executable.display().to_string(),
        args: vec![
            String::from("--exact"),
            String::from(LEADER_TEST),
            String::from("--nocapture"),
        ],
        timeout_ms: u64::try_from(WORKFLOW_TIMEOUT.as_millis())
            .expect("workflow timeout should fit u64"),
        max_stdout_bytes: 4096,
    };

    let Ok(process) = WorkflowProcess::start(&config, WORKFLOW_TIMEOUT) else {
        panic!("moved-PGID workflow should start");
    };
    let deadline = Instant::now() + HELPER_TIMEOUT;
    TestProcessIdentity::capture(process.pid().expect("moved leader should remain armed"))
        .expect("moved leader exact identity should be captured")
        .publish(&directory.join("leader.identity"));
    let identity =
        wait_for_current_published_identity(&directory.join("leader.identity"), deadline)
            .expect("moved leader should publish a current exact identity");
    wait_for_path(&directory.join("leader.moved"), deadline);
    let raw_pid = i32::try_from(identity.pid.get()).expect("test PID should fit i32");
    assert_eq!(
        getpgid(Some(Pid::from_raw(raw_pid))).expect("moved leader PGID should be readable"),
        target_pgid
    );

    match process.complete() {
        Ok(completion) => assert!(completion.timed_out),
        Err(WorkflowProcessFinishError::DeadlineExceeded(_)) => {}
        Err(WorkflowProcessFinishError::Cleanup(error)) => {
            panic!("moved leader cleanup failed: {error}")
        }
        Err(WorkflowProcessFinishError::Stderr(error)) => {
            panic!("moved leader stderr cleanup failed: {error}")
        }
        Err(WorkflowProcessFinishError::OwnershipLost) => {
            panic!("moved leader cleanup lost child ownership")
        }
    }

    assert!(identity.wait_until_not_live(deadline));
    assert!(
        cleanup_worker_available(),
        "exact moved-leader cleanup must reopen admission"
    );
}

#[test]
fn moved_pgid_leader_helper() {
    let argv0 = std::env::args_os()
        .next()
        .map(PathBuf::from)
        .expect("helper argv[0] should exist");
    let Some(target_pgid) = argv0
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("workflow-moved-pgid-leader-"))
        .and_then(|value| value.parse::<i32>().ok())
    else {
        return;
    };
    let directory = argv0
        .parent()
        .expect("moved-PGID helper should have a fixture directory");
    wait_for_path(
        &directory.join("leader.release"),
        Instant::now() + HELPER_TIMEOUT,
    );
    let _removed = fs::remove_file(directory.join("leader.release"));
    setpgid(Pid::from_raw(0), Pid::from_raw(target_pgid))
        .expect("workflow leader should move into the existing supervisor PGID");
    publish_file_atomically(&directory.join("leader.moved"), b"moved");
    thread::sleep(Duration::from_secs(30));
}

struct TestDirectoryCleanup(PathBuf);

impl Drop for TestDirectoryCleanup {
    fn drop(&mut self) {
        let _removed = fs::remove_dir_all(&self.0);
    }
}

fn unique_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "llm-guard-moved-pgid-{}-{nonce}",
        std::process::id()
    ))
}

fn wait_for_path(path: &Path, deadline: Instant) {
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
        thread::sleep(Duration::from_millis(5));
    }
}
