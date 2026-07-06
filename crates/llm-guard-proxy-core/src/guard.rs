//! Guard workflow hook execution.

use std::collections::HashMap;

use crate::settings::GuardWorkflowConfig;
use crate::{
    GWP_PROTOCOL_VERSION, GwpDecision, GwpHook, GwpInvocation, GwpResult, GwpTraceMode,
    ProfileConfig, StdioRuntime, WorkflowConfig,
};

/// Outcome of a guard check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GuardOutcome {
    /// Request or response is allowed unchanged.
    Allow,
    /// Request or response is blocked with a safe reason.
    Block {
        /// Safe operational block reason.
        reason: String,
    },
    /// Content is replaced by workflow-provided payloads.
    Replace {
        /// Replacement OpenAI-compatible payloads.
        messages: Vec<serde_json::Value>,
    },
    /// No guard is configured for this hook.
    Skipped,
}

/// Executes configured guard workflow hooks for a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuardExecutor {
    config: GuardWorkflowConfig,
    workflows: HashMap<String, StdioRuntime>,
}

impl GuardExecutor {
    /// Builds a guard executor from hook configuration and workflow definitions.
    #[must_use]
    pub fn new(config: GuardWorkflowConfig, workflows: HashMap<String, WorkflowConfig>) -> Self {
        let workflows = workflows
            .into_iter()
            .map(|(id, config)| (id, StdioRuntime::new(config)))
            .collect();
        Self { config, workflows }
    }

    /// Run the `pre_request_guard` hook.
    #[must_use]
    pub fn pre_request_guard(
        &self,
        request_id: &str,
        model: &str,
        messages: &[serde_json::Value],
        profile_id: &str,
        profile: &ProfileConfig,
    ) -> GuardOutcome {
        self.run_guard(GuardRunInput {
            workflow_id: self.config.pre_request.as_deref(),
            hook: GwpHook::PreRequestGuard,
            request_id,
            model,
            messages,
            profile_id,
            profile,
        })
    }

    /// Run the `post_response_guard` hook.
    #[must_use]
    pub fn post_response_guard(
        &self,
        request_id: &str,
        model: &str,
        response: &serde_json::Value,
        profile_id: &str,
        profile: &ProfileConfig,
    ) -> GuardOutcome {
        self.run_guard(GuardRunInput {
            workflow_id: self.config.post_response.as_deref(),
            hook: GwpHook::PostResponseGuard,
            request_id,
            model,
            messages: std::slice::from_ref(response),
            profile_id,
            profile,
        })
    }

    fn run_guard(&self, input: GuardRunInput<'_>) -> GuardOutcome {
        let GuardRunInput {
            workflow_id,
            hook,
            request_id,
            model,
            messages,
            profile_id,
            profile,
        } = input;
        let Some(workflow_id) = workflow_id else {
            return GuardOutcome::Skipped;
        };
        let Some(runtime) = self.workflows.get(workflow_id) else {
            return GuardOutcome::Block {
                reason: format!("guard workflow {workflow_id:?} is not configured"),
            };
        };
        let invocation = GwpInvocation {
            protocol_version: GWP_PROTOCOL_VERSION.to_owned(),
            hook,
            request_id: request_id.to_owned(),
            profile: profile.to_gwp_profile(profile_id),
            model_alias: model.to_owned(),
            messages: messages.to_vec(),
            policy: serde_json::Value::Null,
            budgets: serde_json::Value::Null,
            trace_mode: GwpTraceMode::Redacted,
        };
        self.outcome_from_result(workflow_id, &invocation, runtime.execute(&invocation))
    }

    fn outcome_from_result(
        &self,
        workflow_id: &str,
        invocation: &GwpInvocation,
        result: GwpResult,
    ) -> GuardOutcome {
        eprintln!(
            "guard_decision request_id={} guard_workflow_id={} hook={:?} decision={:?} risk_level={} summary={}",
            invocation.request_id,
            workflow_id,
            invocation.hook,
            result.decision,
            result.risk_level,
            result.summary
        );
        match result.decision {
            GwpDecision::Allow | GwpDecision::DeferToParent => GuardOutcome::Allow,
            GwpDecision::Block => GuardOutcome::Block {
                reason: result.summary,
            },
            GwpDecision::Replace => {
                let messages = result.replacement_messages.unwrap_or_default();
                if messages.is_empty() {
                    GuardOutcome::Block {
                        reason: String::from("guard replacement was empty"),
                    }
                } else {
                    GuardOutcome::Replace { messages }
                }
            }
            GwpDecision::ErrorFailClosed => {
                if self.config.fail_closed_blocks {
                    GuardOutcome::Block {
                        reason: result.summary,
                    }
                } else {
                    GuardOutcome::Allow
                }
            }
        }
    }
}

struct GuardRunInput<'request> {
    workflow_id: Option<&'request str>,
    hook: GwpHook,
    request_id: &'request str,
    model: &'request str,
    messages: &'request [serde_json::Value],
    profile_id: &'request str,
    profile: &'request ProfileConfig,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{GuardExecutor, GuardOutcome};
    use crate::settings::GuardWorkflowConfig;
    use crate::{GwpAudit, GwpDecision, GwpResult, ProfileConfig};

    #[test]
    fn skipped_when_no_hook_is_configured() {
        let executor = GuardExecutor::new(GuardWorkflowConfig::default(), HashMap::new());

        let outcome = executor.pre_request_guard(
            "req-1",
            "model",
            &[json!({"role": "user", "content": "hello"})],
            "default",
            &ProfileConfig::default(),
        );

        assert_eq!(outcome, GuardOutcome::Skipped);
    }

    #[test]
    fn missing_configured_workflow_blocks() {
        let executor = GuardExecutor::new(
            GuardWorkflowConfig {
                pre_request: Some(String::from("guard")),
                post_response: None,
                fail_closed_blocks: false,
            },
            HashMap::new(),
        );

        let outcome = executor.pre_request_guard(
            "req-1",
            "model",
            &[json!({"role": "user", "content": "hello"})],
            "default",
            &ProfileConfig::default(),
        );

        assert_eq!(
            outcome,
            GuardOutcome::Block {
                reason: String::from("guard workflow \"guard\" is not configured")
            }
        );
    }

    #[test]
    fn error_fail_closed_can_allow_when_configured_fail_open() {
        let executor = GuardExecutor::new(
            GuardWorkflowConfig {
                pre_request: Some(String::from("guard")),
                post_response: None,
                fail_closed_blocks: false,
            },
            HashMap::new(),
        );
        let invocation = crate::GwpInvocation {
            protocol_version: crate::GWP_PROTOCOL_VERSION.to_owned(),
            hook: crate::GwpHook::PreRequestGuard,
            request_id: String::from("req-1"),
            profile: ProfileConfig::default().to_gwp_profile("default"),
            model_alias: String::from("model"),
            messages: vec![json!({"role": "user", "content": "hello"})],
            policy: serde_json::Value::Null,
            budgets: serde_json::Value::Null,
            trace_mode: crate::GwpTraceMode::Redacted,
        };

        let outcome = executor.outcome_from_result(
            "guard",
            &invocation,
            GwpResult {
                decision: GwpDecision::ErrorFailClosed,
                risk_level: String::from("error"),
                tags: Vec::new(),
                summary: String::from("runtime failed"),
                replacement_messages: None,
                audit: GwpAudit {
                    evidence_spans: Vec::new(),
                    notes: Vec::new(),
                },
            },
        );

        assert_eq!(outcome, GuardOutcome::Allow);
    }
}
