use std::{
    cell::Cell,
    fs,
    os::unix::fs::symlink,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use llm_guard_proxy_core::{
    GWP_PROTOCOL_VERSION, GwpDecision, GwpHook, GwpInvocation, GwpProfile, GwpProfileKind,
    GwpTraceMode, WorkflowConfig, WorkflowRuntime,
};
use nix::unistd::setsid;
use serde_json::json;

use super::StdioRuntime;
use crate::workflow_process::{
    SpawnedChildGuard, install_stderr_drain_completion_probe,
    test_support::{
        TestProcessIdentity, publish_file_atomically, read_published_identity,
        wait_for_published_identity,
    },
};

const ESCAPED_HELPER_TEST: &str = "workflow_runtime::deadline_tests::escaped_descendant_helper";
const VALID_RESULT: &str = r#"{"decision":"allow","risk_level":"low","tags":["ok"],"summary":"ok","replacement_messages":null,"audit":{"evidence_spans":[],"notes":["ok"]}}"#;

#[test]
fn setsid_descendant_retaining_stdout_fails_closed_by_deadline() {
    assert_escaped_pipe_times_out("stdout");
}

#[test]
fn setsid_descendant_retaining_stderr_fails_closed_by_deadline() {
    assert_escaped_pipe_times_out("stderr");
}

#[test]
fn early_stdout_failure_cancels_and_finishes_escaped_stderr_within_one_cleanup_grace() {
    let mut fixture = EscapedFixture::new("stderr-abort");
    let (drain_finished_tx, drain_finished_rx) = mpsc::channel();
    let _drain_probe = install_stderr_drain_completion_probe(drain_finished_tx);
    let started = Instant::now();

    let result = fixture.runtime(30_000).execute(&invocation());
    let escaped = fixture.take_escaped_process();

    assert_eq!(result.decision, GwpDecision::ErrorFailClosed);
    assert_eq!(
        result.tags,
        vec![String::from("error"), String::from("stdout_limit")]
    );
    assert!(
        started.elapsed() <= Duration::from_millis(900),
        "early abort cleanup exceeded one 500 ms cleanup grace: {:?}",
        started.elapsed()
    );
    drain_finished_rx
        .try_recv()
        .expect("abort must wake, close, and finish its stderr drain before returning");
    assert!(
        escaped.is_live(),
        "stderr cancellation must not rely on killing the escaped writer"
    );
    drop(escaped);
}

#[test]
fn setsid_descendant_retaining_stdin_cannot_block_past_deadline() {
    let mut fixture = EscapedFixture::new("stdin");
    let mut request = invocation();
    request.messages = vec![json!({"role": "user", "content": "x".repeat(2 * 1024 * 1024)})];
    let started = Instant::now();

    let result = fixture.runtime(150).execute(&request);
    let escaped = fixture.take_escaped_process();

    assert_eq!(result.decision, GwpDecision::ErrorFailClosed);
    assert!(
        started.elapsed() <= Duration::from_millis(900),
        "blocking stdin exceeded execution deadline plus cleanup grace: {:?}",
        started.elapsed()
    );
    drop(escaped);
}

fn assert_escaped_pipe_times_out(mode: &str) {
    let mut fixture = EscapedFixture::new(mode);
    let started = Instant::now();

    let result = fixture.runtime(150).execute(&invocation());
    let escaped = fixture.take_escaped_process();

    assert_eq!(result.decision, GwpDecision::ErrorFailClosed);
    assert_eq!(
        result.tags,
        vec![String::from("error"), String::from("timeout")]
    );
    assert!(
        started.elapsed() <= Duration::from_millis(900),
        "retained {mode} exceeded execution deadline plus cleanup grace: {:?}",
        started.elapsed()
    );
    drop(escaped);
}

#[test]
fn escaped_fixture_watchdog_remains_armed_during_unwind() {
    let observed_identity = Cell::new(None);
    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let fixture = EscapedFixture::new("stdout");
        let _result = fixture.runtime(150).execute(&invocation());
        observed_identity.set(read_published_identity(&fixture.identity_path));
        panic!("injected panic before escaped ownership transfer");
    }));

    assert!(unwind.is_err());
    let identity = observed_identity
        .get()
        .expect("escaped identity should be captured before unwind");
    assert!(
        !identity.is_live(),
        "armed fixture watchdog should clean the escaped process during unwind"
    );
}

#[test]
fn escaped_identity_records_reject_zero_fields_and_malformed_publication() {
    assert!(TestProcessIdentity::parse("0 123").is_none());
    assert!(TestProcessIdentity::parse("123 0").is_none());
    assert!(TestProcessIdentity::parse("123").is_none());
}

#[test]
fn escaped_descendant_helper() {
    let argv0 = std::env::args_os()
        .next()
        .map(PathBuf::from)
        .expect("helper argv[0] should exist");
    let name = argv0
        .file_name()
        .and_then(|name| name.to_str())
        .expect("helper executable name should be UTF-8");
    let directory = argv0
        .parent()
        .expect("helper executable should have a parent directory");

    if let Some(mode) = name.strip_prefix("workflow-launcher-") {
        launch_escaped_descendant(directory, mode);
        std::process::exit(0);
    }
    if name.starts_with("workflow-escaped-") {
        let deadline = Instant::now() + Duration::from_secs(5);
        let identity = TestProcessIdentity::capture(std::process::id())
            .expect("escaped helper should capture its exact process identity");
        identity.publish(&directory.join("escaped.identity"));
        wait_for_path(&directory.join("escaped.release"), deadline);
        setsid().expect("escaped helper should create a new session");
        publish_file_atomically(&directory.join("escaped.setsid-ack"), b"ready");
        thread::sleep(Duration::from_secs(30));
    }
    // The normal test suite invokes this fixture test through the original test-binary name.
}

fn launch_escaped_descendant(directory: &Path, mode: &str) {
    let escaped_path = directory.join(format!("workflow-escaped-{mode}"));
    let mut command = Command::new(escaped_path);
    command.args(["--exact", ESCAPED_HELPER_TEST, "--nocapture"]);
    match mode {
        "stdout" => {
            command.stdin(Stdio::null()).stderr(Stdio::null());
        }
        "stderr" | "stderr-abort" => {
            command.stdin(Stdio::null()).stdout(Stdio::null());
        }
        "stdin" => {
            command.stdout(Stdio::null()).stderr(Stdio::null());
        }
        _ => panic!("unexpected escaped fixture mode: {mode}"),
    }
    let child = command
        .spawn()
        .expect("escaped descendant helper should spawn");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut child = SpawnedChildGuard::new(child, deadline);
    let identity = wait_for_published_identity(&directory.join("escaped.identity"), deadline)
        .expect("escaped helper should atomically publish a complete exact identity");
    assert_eq!(identity.pid.get(), child.pid());
    assert!(identity.is_live());
    publish_file_atomically(&directory.join("escaped.release"), b"release");
    wait_for_path(&directory.join("escaped.setsid-ack"), deadline);
    assert!(identity.is_live());
    child.disarm();

    if mode == "stderr-abort" {
        println!("{}", "x".repeat(8 * 1024));
    } else if mode != "stdin" {
        println!("{VALID_RESULT}");
    }
}

fn wait_for_path(path: &Path, deadline: Instant) {
    while !path.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        path.exists(),
        "fixture marker {} was not published",
        path.display()
    );
}

struct EscapedFixture {
    directory: PathBuf,
    launcher_path: PathBuf,
    identity_path: PathBuf,
    transition_deadline: Instant,
    cleanup_watchdog: Option<TestEscapeWatchdog>,
}

impl EscapedFixture {
    fn new(mode: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "llm-guard-workflow-deadline-{mode}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("escaped fixture directory should be created");
        let current_exe = std::env::current_exe().expect("test executable path should resolve");
        let launcher_path = directory.join(format!("workflow-launcher-{mode}"));
        let escaped_path = directory.join(format!("workflow-escaped-{mode}"));
        symlink(&current_exe, &launcher_path).expect("launcher symlink should be created");
        symlink(current_exe, escaped_path).expect("escaped helper symlink should be created");
        let identity_path = directory.join("escaped.identity");
        let transition_deadline = Instant::now() + Duration::from_secs(5);
        let cleanup_watchdog = Some(TestEscapeWatchdog::start(
            identity_path.clone(),
            transition_deadline,
        ));
        Self {
            directory,
            launcher_path,
            identity_path,
            transition_deadline,
            cleanup_watchdog,
        }
    }

    fn runtime(&self, timeout_ms: u64) -> StdioRuntime {
        StdioRuntime::new(WorkflowConfig {
            runtime_kind: WorkflowRuntime::Stdio,
            command: self.launcher_path.display().to_string(),
            args: vec![
                String::from("--exact"),
                String::from(ESCAPED_HELPER_TEST),
                String::from("--nocapture"),
            ],
            timeout_ms,
            max_stdout_bytes: 4096,
        })
    }

    fn take_escaped_process(&mut self) -> EscapedProcess {
        let identity = wait_for_published_identity(&self.identity_path, self.transition_deadline)
            .expect("launcher should publish a valid escaped descendant identity");
        wait_for_path(
            &self.directory.join("escaped.setsid-ack"),
            self.transition_deadline,
        );
        self.cleanup_watchdog
            .take()
            .expect("escaped fixture watchdog should remain armed until transfer")
            .disarm();
        EscapedProcess {
            identity,
            cleanup_deadline: self.transition_deadline,
        }
    }
}

impl Drop for EscapedFixture {
    fn drop(&mut self) {
        drop(self.cleanup_watchdog.take());
        let _ignored = fs::remove_dir_all(&self.directory);
    }
}

struct TestEscapeWatchdog {
    cancel: Option<mpsc::SyncSender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl TestEscapeWatchdog {
    fn start(identity_path: PathBuf, deadline: Instant) -> Self {
        let (cancel, receiver) = mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            let mut identity = None;
            loop {
                match receiver.try_recv() {
                    Ok(()) => return,
                    Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                identity = read_published_identity(&identity_path).or(identity);
                if Instant::now() >= deadline {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            let identity =
                identity.or_else(|| wait_for_published_identity(&identity_path, deadline));
            if let Some(identity) = identity {
                identity.signal_if_live();
                let _stopped = identity.wait_until_not_live(deadline);
            }
        });
        Self {
            cancel: Some(cancel),
            handle: Some(handle),
        }
    }

    fn disarm(mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _cancel = cancel.send(());
        }
        self.join();
    }

    fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _join = handle.join();
        }
    }
}

impl Drop for TestEscapeWatchdog {
    fn drop(&mut self) {
        self.cancel.take();
        self.join();
    }
}

struct EscapedProcess {
    identity: TestProcessIdentity,
    cleanup_deadline: Instant,
}

impl EscapedProcess {
    fn is_live(&self) -> bool {
        self.identity.is_live()
    }
}

impl Drop for EscapedProcess {
    fn drop(&mut self) {
        self.identity.signal_if_live();
        let _stopped = self.identity.wait_until_not_live(self.cleanup_deadline);
    }
}

fn invocation() -> GwpInvocation {
    GwpInvocation {
        protocol_version: GWP_PROTOCOL_VERSION.to_owned(),
        hook: GwpHook::PreRequestGuard,
        request_id: String::from("req_deadline"),
        profile: GwpProfile {
            id: String::from("child"),
            kind: GwpProfileKind::Child,
        },
        model_alias: String::from("family/child-safe-general-v1"),
        messages: vec![json!({"role": "user", "content": "hello"})],
        policy: json!({}),
        budgets: json!({}),
        trace_mode: GwpTraceMode::Redacted,
    }
}
