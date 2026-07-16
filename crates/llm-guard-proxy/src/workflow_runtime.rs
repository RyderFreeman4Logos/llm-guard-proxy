//! Service-owned guard workflow process execution.

use std::{
    collections::HashMap,
    io::{self},
    process::ExitStatus,
    time::Duration,
};

use llm_guard_proxy_core::{
    GuardWorkflowExecutor, GwpAudit, GwpDecision, GwpInvocation, GwpResult, WorkflowConfig,
};

use crate::{
    workflow_execution::WorkflowExecutionLease,
    workflow_process::{
        WorkflowProcess, WorkflowProcessFinishError, WorkflowProcessStartError, WorkflowStdinError,
        WorkflowStdoutError,
    },
};

/// Configured service adapter for the core workflow-execution port.
pub(crate) struct WorkflowRuntimeAdapter {
    workflows: HashMap<String, StdioRuntime>,
}

impl WorkflowRuntimeAdapter {
    /// Builds an adapter from one immutable configuration snapshot.
    pub(crate) fn new(
        workflows: HashMap<String, WorkflowConfig>,
        execution_lease: &WorkflowExecutionLease,
    ) -> Self {
        let workflows = workflows
            .into_iter()
            .map(|(id, config)| {
                (
                    id,
                    StdioRuntime::with_execution_lease(config, execution_lease.clone()),
                )
            })
            .collect();
        Self { workflows }
    }
}

impl GuardWorkflowExecutor for WorkflowRuntimeAdapter {
    fn execute(&self, workflow_id: &str, invocation: &GwpInvocation) -> Option<GwpResult> {
        self.workflows
            .get(workflow_id)
            .map(|runtime| runtime.execute(invocation))
    }
}

/// Synchronous stdio workflow runtime.
#[derive(Clone, Debug)]
struct StdioRuntime {
    config: WorkflowConfig,
    execution_lease: WorkflowExecutionLease,
}

impl StdioRuntime {
    /// Builds a stdio workflow runtime from validated config.
    #[must_use]
    #[cfg(test)]
    fn new(config: WorkflowConfig) -> Self {
        Self::with_execution_lease(config, WorkflowExecutionLease::default())
    }

    fn with_execution_lease(
        config: WorkflowConfig,
        execution_lease: WorkflowExecutionLease,
    ) -> Self {
        Self {
            config,
            execution_lease,
        }
    }

    /// Spawn the workflow process, write invocation JSON to stdin, read result JSON from stdout,
    /// and enforce timeout plus size limits.
    ///
    /// Any runtime failure is mapped to a [`GwpDecision::ErrorFailClosed`] result.
    #[must_use]
    fn execute(&self, invocation: &GwpInvocation) -> GwpResult {
        self.execute_inner(invocation)
            .unwrap_or_else(WorkflowExecutionError::into_result)
    }

    fn execute_inner(
        &self,
        invocation: &GwpInvocation,
    ) -> Result<GwpResult, WorkflowExecutionError> {
        validate_runtime_config(&self.config)?;
        let mut process = WorkflowProcess::start_with_execution_lease(
            &self.config,
            Duration::from_millis(self.config.timeout_ms),
            self.execution_lease.clone(),
        )
        .map_err(workflow_process_start_error)?;

        let output = match write_invocation(&mut process, invocation)
            .and_then(|()| read_stdout_limited(&mut process, self.config.max_stdout_bytes))
        {
            Ok(output) => output,
            Err(error) => {
                process.abort();
                return Err(error);
            }
        };

        let completion = process.complete().map_err(workflow_process_finish_error)?;

        if completion.timed_out {
            return Err(WorkflowExecutionError::new(
                "timeout",
                "workflow process timed out",
                format!(
                    "workflow exceeded timeout_ms={} and was killed",
                    self.config.timeout_ms
                ),
            ));
        }
        if !completion.status.success() {
            return Err(non_zero_exit_error(completion.status));
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
    if config.timeout_ms == 0 || config.timeout_ms > WorkflowConfig::max_timeout_ms() {
        return Err(WorkflowExecutionError::new(
            "config",
            "workflow timeout is outside the allowed range",
            format!(
                "workflow timeout_ms must be between 1 and {}; got {}",
                WorkflowConfig::max_timeout_ms(),
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

fn workflow_process_start_error(error: WorkflowProcessStartError) -> WorkflowExecutionError {
    match error {
        WorkflowProcessStartError::CleanupUnavailable => WorkflowExecutionError::new(
            "runtime",
            "workflow cleanup worker is unavailable",
            "workflow cleanup worker is unavailable or still owns a group awaiting termination",
        ),
        #[cfg(target_os = "linux")]
        WorkflowProcessStartError::Cgroup(error) => WorkflowExecutionError::new(
            "runtime",
            "failed to initialize workflow cgroup containment",
            format!("failed to initialize workflow cgroup containment: {error}"),
        ),
        WorkflowProcessStartError::Spawn(error) => WorkflowExecutionError::new(
            "spawn",
            "failed to spawn workflow process",
            format!("failed to spawn workflow process: {error}"),
        ),
        WorkflowProcessStartError::IdentityCapture(error) => WorkflowExecutionError::new(
            "spawn",
            "failed to capture workflow process identity",
            format!("failed to capture workflow process identity: {error}"),
        ),
        WorkflowProcessStartError::IdentityChanged(error) => WorkflowExecutionError::new(
            "spawn",
            "workflow process identity changed during startup",
            format!("workflow process identity changed during startup: {error}"),
        ),
        WorkflowProcessStartError::Watchdog(error) => WorkflowExecutionError::new(
            "runtime",
            "failed to start workflow timeout watchdog",
            format!("failed to start workflow timeout watchdog: {error}"),
        ),
        WorkflowProcessStartError::StderrDrain(error) => WorkflowExecutionError::new(
            "runtime",
            "failed to start workflow stderr drain",
            format!("failed to start workflow stderr drain: {error}"),
        ),
    }
}

fn workflow_process_finish_error(error: WorkflowProcessFinishError) -> WorkflowExecutionError {
    match error {
        WorkflowProcessFinishError::OwnershipLost => WorkflowExecutionError::new(
            "wait",
            "workflow process ownership was lost before cleanup",
            "waitid reported ECHILD before workflow cleanup",
        ),
        WorkflowProcessFinishError::Cleanup(error) => {
            let (category, summary) = if error.kind() == io::ErrorKind::TimedOut {
                ("timeout", "workflow process timed out")
            } else {
                ("wait", "failed to finalize workflow process")
            };
            WorkflowExecutionError::new(
                category,
                summary,
                format!("failed to signal or reap workflow process: {error}"),
            )
        }
        WorkflowProcessFinishError::DeadlineExceeded(error) => WorkflowExecutionError::new(
            "timeout",
            "workflow process timed out",
            format!("workflow deadline elapsed before cleanup completed: {error}"),
        ),
        WorkflowProcessFinishError::Stderr(error) => {
            let (category, summary) = if error.kind() == io::ErrorKind::TimedOut {
                ("timeout", "workflow process timed out")
            } else {
                ("stderr", "failed to drain workflow stderr")
            };
            WorkflowExecutionError::new(
                category,
                summary,
                format!("failed to drain workflow stderr: {error}"),
            )
        }
    }
}

fn write_invocation(
    process: &mut WorkflowProcess,
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

    process.write_stdin(&payload).map_err(|error| match error {
        WorkflowStdinError::Missing => WorkflowExecutionError::new(
            "stdin",
            "workflow stdin was not captured",
            "workflow child missing stdin pipe",
        ),
        WorkflowStdinError::Write(error) => {
            let (category, summary) = if error.kind() == io::ErrorKind::TimedOut {
                ("timeout", "workflow process timed out")
            } else {
                ("stdin", "failed to write workflow invocation")
            };
            WorkflowExecutionError::new(
                category,
                summary,
                format!("failed to write workflow invocation to stdin: {error}"),
            )
        }
    })
}

fn read_stdout_limited(
    process: &mut WorkflowProcess,
    max_stdout_bytes: usize,
) -> Result<Vec<u8>, WorkflowExecutionError> {
    process.read_stdout(max_stdout_bytes).map_err(|error| {
        let error = match error {
            WorkflowStdoutError::Missing => {
                return WorkflowExecutionError::new(
                    "spawn",
                    "workflow stdout was not captured",
                    "workflow child missing stdout pipe",
                );
            }
            WorkflowStdoutError::Read(error) => error,
        };
        let (category, summary) = if error.kind() == io::ErrorKind::InvalidData {
            (
                "stdout_limit",
                "workflow stdout exceeded the configured byte limit",
            )
        } else if error.kind() == io::ErrorKind::TimedOut {
            ("timeout", "workflow process timed out")
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

#[cfg(test)]
#[path = "workflow_runtime/deadline_tests.rs"]
mod deadline_tests;

#[cfg(test)]
mod tests {
    use std::{
        fs, io,
        path::{Path, PathBuf},
        process::{Child, Command},
        thread,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::{StdioRuntime, workflow_process_finish_error};
    use crate::workflow_process::{
        WorkflowProcessFinishError,
        test_support::{TestProcessIdentity, wait_for_current_published_identity},
    };
    use llm_guard_proxy_core::{
        GWP_PROTOCOL_VERSION, GwpDecision, GwpHook, GwpInvocation, GwpProfile, GwpProfileKind,
        GwpTraceMode, WorkflowConfig, WorkflowRuntime,
    };

    const PID_FILE_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

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
    fn cleanup_deadline_exhaustion_is_classified_as_timeout() {
        let error = workflow_process_finish_error(WorkflowProcessFinishError::Cleanup(
            io::Error::new(io::ErrorKind::TimedOut, "injected cleanup deadline"),
        ));

        assert_eq!(error.category, "timeout");
        assert_eq!(error.summary, "workflow process timed out");
    }

    #[test]
    fn observed_deadline_has_priority_over_later_signal_failure() {
        let result = workflow_process_finish_error(WorkflowProcessFinishError::DeadlineExceeded(
            io::Error::other("signal_failed"),
        ))
        .into_result();

        assert_eq!(
            result.tags,
            vec![String::from("error"), String::from("timeout")]
        );
    }

    #[test]
    fn timeout_fails_closed_and_kills_process_group() {
        const TIMEOUT_MS: u64 = 2_000;
        let temp = TempWorkflowDir::new("timeout");
        let child_pid_path = temp.path.join("child.pid");
        let script = temp.write_script(
            "timeout.sh",
            &format!(
                "read ignored\nsleep 30 &\nchild=$!\nstart=$(awk '{{print $22}}' /proc/$child/stat)\nmarker='{}'\ntmp=\"${{marker}}.tmp.$$\"\nif [ \"$child\" -le 0 ] || [ -z \"$start\" ] || [ \"$start\" -le 0 ]; then exit 90; fi\nprintf '%s %s\\n' \"$child\" \"$start\" > \"$tmp\"\nmv -f \"$tmp\" \"$marker\"\nwait\n",
                child_pid_path.display()
            ),
        );
        let cleanup = TestPidFileCleanup::new(child_pid_path.clone());
        let runtime = runtime(&script, Vec::new(), TIMEOUT_MS, 4096);
        let invocation = invocation();

        thread::scope(|scope| {
            let execution = scope.spawn(|| runtime.execute(&invocation));
            let child_identity = cleanup
                .wait_for_identity(Instant::now() + Duration::from_secs(1))
                .expect("script should publish its child identity before the timeout");
            let result = execution.join().expect("workflow execution should join");

            assert_error(&result, "timeout");
            assert_process_exits(child_identity);
        });
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
                "read ignored\nsleep 30 >'{idle}' 2>&1 &\nchild=$!\nstart=$(awk '{{print $22}}' /proc/$child/stat)\nmarker='{pid}'\ntmp=\"${{marker}}.tmp.$$\"\nif [ \"$child\" -le 0 ] || [ -z \"$start\" ] || [ \"$start\" -le 0 ]; then exit 90; fi\nprintf '%s %s\\n' \"$child\" \"$start\" > \"$tmp\"\nmv -f \"$tmp\" \"$marker\"\nprintf '%s\\n' '{{\"decision\":\"allow\",\"risk_level\":\"low\",\"tags\":[\"ok\"],\"summary\":\"ok\",\"replacement_messages\":null,\"audit\":{{\"evidence_spans\":[],\"notes\":[\"ok\"]}}}}'\n",
                idle = temp.path.join("idle.log").display(),
                pid = child_pid_path.display()
            ),
        );
        let cleanup = TestPidFileCleanup::new(child_pid_path.clone());

        let result = runtime(&script, Vec::new(), 5_000, 4096).execute(&invocation());

        assert_eq!(result.decision, GwpDecision::Allow);
        let child_identity = cleanup
            .identity()
            .expect("script should persist child identity");
        assert_process_exits(child_identity);
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

    #[test]
    fn pid_file_cleanup_retries_malformed_marker_until_atomic_identity_arrives() {
        let temp = TempWorkflowDir::new("pid-cleanup-retry");
        let pid_file = temp.path.join("process.identity");
        fs::write(&pid_file, b"truncated\n").expect("malformed fixture marker should be written");
        let child = Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("cleanup fixture should spawn");
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut child = TestChildGuard::new(child, deadline);
        let published = TestProcessIdentity::capture(child.pid())
            .expect("cleanup fixture identity should be captured");
        let cleanup = TestPidFileCleanup::new(pid_file.clone());
        let staging_file = pid_file.with_extension("identity.tmp");

        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(20));
                fs::write(
                    &staging_file,
                    format!("{} {}\n", published.pid, published.start_time_ticks),
                )
                .expect("complete identity should be staged");
                fs::rename(&staging_file, &pid_file)
                    .expect("complete identity should be published atomically");
            });
            drop(cleanup);
        });

        loop {
            match child
                .child_mut()
                .expect("cleanup fixture should remain armed")
                .try_wait()
            {
                Ok(Some(_)) => {
                    child.disarm();
                    break;
                }
                Ok(None) => {
                    assert!(Instant::now() < deadline, "cleanup fixture remained alive");
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("cleanup fixture status failed: {error}"),
            }
        }
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

    fn assert_error(result: &llm_guard_proxy_core::GwpResult, category: &str) {
        assert_eq!(result.decision, GwpDecision::ErrorFailClosed);
        assert_eq!(result.risk_level, "error");
        assert_eq!(
            result.tags,
            vec![String::from("error"), category.to_owned()]
        );
        assert_eq!(result.replacement_messages, None);
        assert!(!result.audit.notes.is_empty());
    }

    fn assert_process_exits(identity: TestProcessIdentity) {
        for _attempt in 0..20 {
            if !process_exists(identity) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!(
            "process {} should not survive workflow timeout",
            identity.pid
        );
    }

    fn process_exists(identity: TestProcessIdentity) -> bool {
        identity.is_live()
    }

    struct TestChildGuard {
        child: Option<Child>,
        cleanup_deadline: Instant,
    }

    impl TestChildGuard {
        fn new(child: Child, cleanup_deadline: Instant) -> Self {
            Self {
                child: Some(child),
                cleanup_deadline,
            }
        }

        fn pid(&self) -> u32 {
            self.child.as_ref().expect("child guard is armed").id()
        }

        fn child_mut(&mut self) -> Option<&mut Child> {
            self.child.as_mut()
        }

        fn disarm(&mut self) {
            let _child = self.child.take();
        }
    }

    impl Drop for TestChildGuard {
        fn drop(&mut self) {
            let Some(child) = self.child.as_mut() else {
                return;
            };
            let _killed = child.kill();
            loop {
                if child.try_wait().ok().flatten().is_some()
                    || Instant::now() >= self.cleanup_deadline
                {
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
        }
    }

    struct TestPidFileCleanup {
        path: PathBuf,
    }

    impl TestPidFileCleanup {
        const fn new(path: PathBuf) -> Self {
            Self { path }
        }

        fn identity(&self) -> Option<TestProcessIdentity> {
            self.wait_for_identity(Instant::now() + PID_FILE_CLEANUP_TIMEOUT)
        }

        fn wait_for_identity(&self, deadline: Instant) -> Option<TestProcessIdentity> {
            wait_for_current_published_identity(&self.path, deadline)
        }
    }

    impl Drop for TestPidFileCleanup {
        fn drop(&mut self) {
            let deadline = Instant::now() + PID_FILE_CLEANUP_TIMEOUT;
            if let Some(identity) = self.wait_for_identity(deadline) {
                identity.signal_if_live();
            }
        }
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
