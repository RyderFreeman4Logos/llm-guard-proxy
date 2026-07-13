//! Pure guard workflow configuration and execution contracts.

use crate::{GwpInvocation, GwpResult};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const DEFAULT_MAX_STDOUT_BYTES: usize = 1024 * 1024;

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

/// Executes configured guard workflows without exposing runtime or transport details.
///
/// Implementations must be safe to share across service worker threads. Returning `None`
/// means that `workflow_id` is not configured and callers must fail safely. Runtime failures
/// must instead return `Some` with a fail-closed [`GwpResult`].
pub trait GuardWorkflowExecutor: Send + Sync {
    /// Executes one configured workflow for the supplied invocation.
    #[must_use]
    fn execute(&self, workflow_id: &str, invocation: &GwpInvocation) -> Option<GwpResult>;
}
