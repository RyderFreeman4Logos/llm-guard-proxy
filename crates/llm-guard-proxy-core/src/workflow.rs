//! Guard workflow runtime execution.

use std::{
    env,
    io::{self, Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::{GwpAudit, GwpDecision, GwpInvocation, GwpResult};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const DEFAULT_MAX_STDOUT_BYTES: usize = 1024 * 1024;
const READ_CHUNK_BYTES: usize = 8 * 1024;
const ALLOWED_ENV_VARS: [&str; 4] = ["PATH", "LANG", "LC_ALL", "HOME"];

/// Workflow runtime backend.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRuntime {
    /// Spawn a process and exchange one JSON request/result over stdio.
    #[default]
    Stdio,
}

impl WorkflowRuntime {
    /// Returns the TOML-compatible runtime kind label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
        }
    }
}

/// Configuration for one guard workflow.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct WorkflowConfig {
    /// Workflow runtime backend.
    pub runtime_kind: WorkflowRuntime,
    /// Executable to spawn.
    pub command: String,
    /// Arguments passed directly to the executable.
    pub args: Vec<String>,
    /// Maximum execution time in milliseconds.
    pub timeout_ms: u64,
    /// Maximum stdout bytes accepted as the result JSON.
    pub max_stdout_bytes: usize,
}

impl WorkflowConfig {
    /// Default workflow timeout in milliseconds.
    #[must_use]
    pub const fn default_timeout_ms() -> u64 {
        DEFAULT_TIMEOUT_MS
    }

    /// Maximum accepted workflow timeout in milliseconds.
    #[must_use]
    pub const fn max_timeout_ms() -> u64 {
        MAX_TIMEOUT_MS
    }

    /// Default maximum stdout bytes accepted from a workflow.
    #[must_use]
    pub const fn default_max_stdout_bytes() -> usize {
        DEFAULT_MAX_STDOUT_BYTES
    }
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            runtime_kind: WorkflowRuntime::Stdio,
            command: String::new(),
            args: Vec::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
        }
    }
}

/// Synchronous stdio workflow runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StdioRuntime {
    config: WorkflowConfig,
}

impl StdioRuntime {
    /// Builds a stdio workflow runtime from validated config.
    #[must_use]
    pub const fn new(config: WorkflowConfig) -> Self {
        Self { config }
    }

    /// Spawn the workflow process, write invocation JSON to stdin, read result JSON from stdout,
    /// and enforce timeout plus size limits.
    ///
    /// Any runtime failure is mapped to a [`GwpDecision::ErrorFailClosed`] result.
    #[must_use]
    pub fn execute(&self, invocation: &GwpInvocation) -> GwpResult {
        self.execute_inner(invocation)
            .unwrap_or_else(WorkflowExecutionError::into_result)
    }

    fn execute_inner(
        &self,
        invocation: &GwpInvocation,
    ) -> Result<GwpResult, WorkflowExecutionError> {
        validate_runtime_config(&self.config)?;

        let mut command = Command::new(&self.config.command);
        command
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_allowed_environment(&mut command);
        configure_process_group(&mut command);

        let mut child = command.spawn().map_err(|error| {
            WorkflowExecutionError::new(
                "spawn",
                "failed to spawn workflow process",
                format!("failed to spawn workflow process: {error}"),
            )
        })?;

        let pid = child.id();
        let timed_out = Arc::new(AtomicBool::new(false));
        let watchdog = spawn_timeout_watchdog(pid, self.config.timeout_ms, Arc::clone(&timed_out));
        let stderr_handle = child.stderr.take().map(spawn_pipe_drain);

        let Some(stdout) = child.stdout.take() else {
            terminate_child_group(pid);
            let _ignored = child.wait();
            finish_watchdog(watchdog);
            finish_stderr_drain(stderr_handle);
            return Err(WorkflowExecutionError::new(
                "spawn",
                "workflow stdout was not captured",
                "workflow child missing stdout pipe",
            ));
        };

        let output = match write_invocation(&mut child, invocation)
            .and_then(|()| read_stdout_limited(stdout, self.config.max_stdout_bytes))
        {
            Ok(output) => output,
            Err(error) => {
                terminate_child_group(pid);
                let _ignored = child.wait();
                finish_watchdog(watchdog);
                finish_stderr_drain(stderr_handle);
                return Err(error);
            }
        };

        let status = match child.wait() {
            Ok(status) => status,
            Err(error) => {
                terminate_child_group(pid);
                finish_watchdog(watchdog);
                finish_stderr_drain(stderr_handle);
                return Err(WorkflowExecutionError::new(
                    "wait",
                    "failed to wait for workflow process",
                    format!("failed to wait for workflow process: {error}"),
                ));
            }
        };
        // Always terminate the process group to clean up any descendant
        // processes the workflow may have spawned, even on the success path.
        terminate_child_group(pid);
        finish_watchdog(watchdog);
        finish_stderr_drain(stderr_handle);

        if timed_out.load(Ordering::SeqCst) {
            return Err(WorkflowExecutionError::new(
                "timeout",
                "workflow process timed out",
                format!(
                    "workflow exceeded timeout_ms={} and was killed",
                    self.config.timeout_ms
                ),
            ));
        }
        if !status.success() {
            return Err(non_zero_exit_error(status));
        }
        parse_workflow_result(&output)
    }
}

struct WorkflowExecutionError {
    category: &'static str,
    summary: String,
    detail: String,
}

impl WorkflowExecutionError {
    fn new(category: &'static str, summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            category,
            summary: summary.into(),
            detail: detail.into(),
        }
    }

    fn into_result(self) -> GwpResult {
        GwpResult {
            decision: GwpDecision::ErrorFailClosed,
            risk_level: String::from("error"),
            tags: vec![String::from("error"), self.category.to_owned()],
            summary: self.summary,
            replacement_messages: None,
            audit: GwpAudit {
                evidence_spans: Vec::new(),
                notes: vec![self.detail],
            },
        }
    }
}

fn validate_runtime_config(config: &WorkflowConfig) -> Result<(), WorkflowExecutionError> {
    if config.command.trim().is_empty() {
        return Err(WorkflowExecutionError::new(
            "config",
            "workflow command is empty",
            "workflow command must not be empty",
        ));
    }
    if config.timeout_ms == 0 || config.timeout_ms > MAX_TIMEOUT_MS {
        return Err(WorkflowExecutionError::new(
            "config",
            "workflow timeout is outside the allowed range",
            format!(
                "workflow timeout_ms must be between 1 and {MAX_TIMEOUT_MS}; got {}",
                config.timeout_ms
            ),
        ));
    }
    if config.max_stdout_bytes == 0 {
        return Err(WorkflowExecutionError::new(
            "config",
            "workflow stdout byte limit is zero",
            "workflow max_stdout_bytes must be greater than zero",
        ));
    }
    Ok(())
}

fn apply_allowed_environment(command: &mut Command) {
    command.env_clear();
    for key in ALLOWED_ENV_VARS {
        if let Some(value) = env::var_os(key) {
            command.env(key, value);
        }
    }
}

fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn spawn_timeout_watchdog(pid: u32, timeout_ms: u64, timed_out: Arc<AtomicBool>) -> Watchdog {
    let (done_sender, done_receiver) = mpsc::channel();
    let timeout = Duration::from_millis(timeout_ms);
    let handle = thread::spawn(move || {
        if matches!(
            done_receiver.recv_timeout(timeout),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            timed_out.store(true, Ordering::SeqCst);
            terminate_child_group(pid);
        }
    });
    Watchdog {
        done_sender,
        handle,
    }
}

struct Watchdog {
    done_sender: mpsc::Sender<()>,
    handle: JoinHandle<()>,
}

fn finish_watchdog(watchdog: Watchdog) {
    let _ignored = watchdog.done_sender.send(());
    let _ignored = watchdog.handle.join();
}

fn spawn_pipe_drain<R>(pipe: R) -> JoinHandle<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || read_bounded(pipe, DEFAULT_MAX_STDOUT_BYTES))
}

fn finish_stderr_drain(handle: Option<JoinHandle<io::Result<Vec<u8>>>>) {
    if let Some(handle) = handle {
        let _ignored = handle.join();
    }
}

fn write_invocation(
    child: &mut Child,
    invocation: &GwpInvocation,
) -> Result<(), WorkflowExecutionError> {
    let mut payload = serde_json::to_vec(invocation).map_err(|error| {
        WorkflowExecutionError::new(
            "serialize_invocation",
            "failed to serialize workflow invocation",
            format!("failed to serialize workflow invocation: {error}"),
        )
    })?;
    payload.push(b'\n');

    let mut stdin = child.stdin.take().ok_or_else(|| {
        WorkflowExecutionError::new(
            "stdin",
            "workflow stdin was not captured",
            "workflow child missing stdin pipe",
        )
    })?;
    stdin.write_all(&payload).map_err(|error| {
        WorkflowExecutionError::new(
            "stdin",
            "failed to write workflow invocation",
            format!("failed to write workflow invocation to stdin: {error}"),
        )
    })
}

fn read_stdout_limited<R>(
    stdout: R,
    max_stdout_bytes: usize,
) -> Result<Vec<u8>, WorkflowExecutionError>
where
    R: Read,
{
    read_bounded(stdout, max_stdout_bytes).map_err(|error| {
        let (category, summary) = if error.kind() == io::ErrorKind::InvalidData {
            (
                "stdout_limit",
                "workflow stdout exceeded the configured byte limit",
            )
        } else {
            ("stdout", "failed to read workflow stdout")
        };
        WorkflowExecutionError::new(
            category,
            summary,
            format!("failed to read workflow stdout: {error}"),
        )
    })
}

fn read_bounded<R>(mut reader: R, max_bytes: usize) -> io::Result<Vec<u8>>
where
    R: Read,
{
    let mut output = Vec::new();
    let mut buffer = [0_u8; READ_CHUNK_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(read) > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("stdout exceeded max_stdout_bytes={max_bytes}"),
            ));
        }
        output.extend_from_slice(&buffer[..read]);
    }
}

fn parse_workflow_result(output: &[u8]) -> Result<GwpResult, WorkflowExecutionError> {
    serde_json::from_slice(output).map_err(|error| {
        WorkflowExecutionError::new(
            "malformed_json",
            "workflow returned malformed JSON",
            format!("failed to parse workflow stdout as GWP result JSON: {error}"),
        )
    })
}

fn non_zero_exit_error(status: ExitStatus) -> WorkflowExecutionError {
    WorkflowExecutionError::new(
        "exit_status",
        "workflow process exited unsuccessfully",
        format!("workflow process exited with status {status}"),
    )
}

fn terminate_child_group(pid: u32) {
    // Send SIGKILL to the entire process group, then to the process itself.
    // The group kill catches descendant processes the workflow may have
    // spawned; the direct kill handles the case where the group leader has
    // already exited and the PGID was recycled.
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let pid_signed = i32::try_from(pid).unwrap_or(-1);
    let group_pid = Pid::from_raw(-pid_signed);
    let direct_pid = Pid::from_raw(pid_signed);
    let _ignored = kill(group_pid, Signal::SIGKILL);
    let _ignored2 = kill(direct_pid, Signal::SIGKILL);
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process::Command,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::{StdioRuntime, WorkflowConfig, WorkflowRuntime};
    use crate::{
        GWP_PROTOCOL_VERSION, GwpDecision, GwpHook, GwpInvocation, GwpProfile, GwpProfileKind,
        GwpTraceMode,
    };

    #[test]
    fn executes_allow_script() {
        let temp = TempWorkflowDir::new("allow");
        let script = temp.write_script(
            "allow.sh",
            r#"read ignored
printf '%s\n' '{"decision":"allow","risk_level":"low","tags":["ok"],"summary":"allowed","replacement_messages":null,"audit":{"evidence_spans":[],"notes":["ok"]}}'
"#,
        );

        let result = runtime(&script, Vec::new(), 1_000, 4096).execute(&invocation());

        assert_eq!(result.decision, GwpDecision::Allow);
        assert_eq!(result.summary, "allowed");
    }

    #[test]
    fn executes_block_script() {
        let temp = TempWorkflowDir::new("block");
        let script = temp.write_script(
            "block.sh",
            r#"read ignored
printf '%s\n' '{"decision":"block","risk_level":"high","tags":["policy"],"summary":"blocked","replacement_messages":null,"audit":{"evidence_spans":[],"notes":["blocked"]}}'
"#,
        );

        let result = runtime(&script, Vec::new(), 1_000, 4096).execute(&invocation());

        assert_eq!(result.decision, GwpDecision::Block);
        assert_eq!(result.summary, "blocked");
    }

    #[test]
    fn malformed_json_fails_closed() {
        let temp = TempWorkflowDir::new("malformed");
        let script = temp.write_script("malformed.sh", "read ignored\nprintf '%s\n' 'not json'\n");

        let result = runtime(&script, Vec::new(), 1_000, 4096).execute(&invocation());

        assert_error(&result, "malformed_json");
    }

    #[test]
    fn non_zero_exit_fails_closed() {
        let temp = TempWorkflowDir::new("nonzero");
        let script = temp.write_script("nonzero.sh", "read ignored\nexit 42\n");

        let result = runtime(&script, Vec::new(), 1_000, 4096).execute(&invocation());

        assert_error(&result, "exit_status");
    }

    #[test]
    fn timeout_fails_closed_and_kills_process_group() {
        let temp = TempWorkflowDir::new("timeout");
        let child_pid_path = temp.path.join("child.pid");
        let script = temp.write_script(
            "timeout.sh",
            &format!(
                "read ignored\nsleep 30 &\necho $! > '{}'\nwait\n",
                child_pid_path.display()
            ),
        );

        let result = runtime(&script, Vec::new(), 100, 4096).execute(&invocation());

        assert_error(&result, "timeout");
        let child_pid = fs::read_to_string(&child_pid_path)
            .expect("script should write child pid")
            .trim()
            .to_owned();
        assert_process_exits(&child_pid);
    }

    #[test]
    fn successful_execution_kills_descendant_processes() {
        let temp = TempWorkflowDir::new("descendant");
        let child_pid_path = temp.path.join("child.pid");
        // The background sleep must redirect its stdout/stderr so the
        // main script's stdout pipe closes promptly, allowing the runtime
        // to read the result and exit. The process group kill then cleans
        // up the lingering sleep.
        let script = temp.write_script(
            "descendant.sh",
            &format!(
                "read ignored\nsleep 30 >'{idle}' 2>&1 &\necho $! > '{pid}'\nprintf '%s\\n' '{{\"decision\":\"allow\",\"risk_level\":\"low\",\"tags\":[\"ok\"],\"summary\":\"ok\",\"replacement_messages\":null,\"audit\":{{\"evidence_spans\":[],\"notes\":[\"ok\"]}}}}'\n",
                idle = temp.path.join("idle.log").display(),
                pid = child_pid_path.display()
            ),
        );

        let result = runtime(&script, Vec::new(), 5_000, 4096).execute(&invocation());

        assert_eq!(result.decision, GwpDecision::Allow);
        let child_pid = fs::read_to_string(&child_pid_path)
            .expect("script should write child pid")
            .trim()
            .to_owned();
        assert_process_exits(&child_pid);
    }

    #[test]
    fn stdout_limit_fails_closed() {
        let temp = TempWorkflowDir::new("stdout-limit");
        let script = temp.write_script("large.sh", "read ignored\nprintf '%s' '1234567890'\n");

        let result = runtime(&script, Vec::new(), 1_000, 4).execute(&invocation());

        assert_error(&result, "stdout_limit");
    }

    #[test]
    fn unexpected_environment_is_not_inherited() {
        if std::env::var_os("SHELL").is_none() {
            return;
        }
        let temp = TempWorkflowDir::new("env");
        let script = temp.write_script(
            "env.sh",
            r#"read ignored
if [ -n "${SHELL:-}" ]; then
  printf '%s\n' '{"decision":"block","risk_level":"high","tags":["env"],"summary":"leaked","replacement_messages":null,"audit":{"evidence_spans":[],"notes":["pwd leaked"]}}'
else
  printf '%s\n' '{"decision":"allow","risk_level":"low","tags":["env"],"summary":"clean","replacement_messages":null,"audit":{"evidence_spans":[],"notes":["clean"]}}'
fi
"#,
        );

        let result = runtime(&script, Vec::new(), 1_000, 4096).execute(&invocation());

        assert_eq!(result.decision, GwpDecision::Allow);
        assert_eq!(result.summary, "clean");
    }

    fn runtime(
        script: &Path,
        args: Vec<String>,
        timeout_ms: u64,
        max_stdout_bytes: usize,
    ) -> StdioRuntime {
        let mut command_args = vec![script.display().to_string()];
        command_args.extend(args);
        StdioRuntime::new(WorkflowConfig {
            runtime_kind: WorkflowRuntime::Stdio,
            command: String::from("/bin/sh"),
            args: command_args,
            timeout_ms,
            max_stdout_bytes,
        })
    }

    fn invocation() -> GwpInvocation {
        GwpInvocation {
            protocol_version: GWP_PROTOCOL_VERSION.to_owned(),
            hook: GwpHook::PreRequestGuard,
            request_id: String::from("req_test"),
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

    fn assert_error(result: &crate::GwpResult, category: &str) {
        assert_eq!(result.decision, GwpDecision::ErrorFailClosed);
        assert_eq!(result.risk_level, "error");
        assert_eq!(
            result.tags,
            vec![String::from("error"), category.to_owned()]
        );
        assert_eq!(result.replacement_messages, None);
        assert!(!result.audit.notes.is_empty());
    }

    fn assert_process_exits(pid: &str) {
        for _attempt in 0..20 {
            if !process_exists(pid) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("process {pid} should not survive workflow timeout");
    }

    #[cfg(unix)]
    fn process_exists(pid: &str) -> bool {
        Command::new("kill")
            .args(["-0", pid])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(windows)]
    fn process_exists(_pid: &str) -> bool {
        false
    }

    struct TempWorkflowDir {
        path: PathBuf,
    }

    impl TempWorkflowDir {
        fn new(name: &str) -> Self {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_millis();
            let path = std::env::temp_dir().join(format!(
                "llm-guard-proxy-workflow-{name}-{millis}-{}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("temp workflow directory should be created");
            Self { path }
        }

        fn write_script(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, contents).expect("test script should be written");
            path
        }
    }

    impl Drop for TempWorkflowDir {
        fn drop(&mut self) {
            remove_dir_if_exists(&self.path);
        }
    }

    fn remove_dir_if_exists(path: &Path) {
        let _ignored = fs::remove_dir_all(path);
    }
}
