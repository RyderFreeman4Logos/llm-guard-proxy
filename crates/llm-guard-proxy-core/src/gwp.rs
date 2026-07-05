//! Guard Workflow Protocol invocation and result envelopes.

/// Current Guard Workflow Protocol version emitted by this crate.
pub const GWP_PROTOCOL_VERSION: &str = "gwp-0.1";

/// Invocation envelope sent to a guard workflow hook.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GwpInvocation {
    /// Wire protocol version for the envelope.
    pub protocol_version: String,
    /// Guard hook being invoked.
    pub hook: GwpHook,
    /// Stable request id used for trace correlation.
    pub request_id: String,
    /// Profile attached to the guarded request.
    pub profile: GwpProfile,
    /// Model alias selected for the request.
    pub model_alias: String,
    /// OpenAI-compatible message payloads.
    pub messages: Vec<serde_json::Value>,
    /// Guard policy payload for the selected hook/profile.
    pub policy: serde_json::Value,
    /// Runtime budget payload for the workflow.
    pub budgets: serde_json::Value,
    /// Trace disclosure mode for the invocation payload.
    pub trace_mode: GwpTraceMode,
}

/// Guard workflow hook names supported by phase 1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GwpHook {
    /// Guard runs before forwarding the request upstream.
    PreRequestGuard,
    /// Guard runs after receiving or constructing the response.
    PostResponseGuard,
}

/// Profile metadata attached to a guard invocation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GwpProfile {
    /// Stable profile id.
    pub id: String,
    /// Profile category used by guard policies.
    pub kind: GwpProfileKind,
}

/// Profile categories understood by guard policies.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GwpProfileKind {
    /// Child profile.
    Child,
    /// Adult profile.
    Adult,
}

/// Result envelope returned by a guard workflow hook.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GwpResult {
    /// Guard decision for the attempted request or response.
    pub decision: GwpDecision,
    /// Policy-defined risk level.
    pub risk_level: String,
    /// Policy-defined result tags.
    pub tags: Vec<String>,
    /// Human-readable result summary.
    pub summary: String,
    /// Replacement messages when the decision is `replace`.
    pub replacement_messages: Option<Vec<serde_json::Value>>,
    /// Audit metadata emitted by the workflow.
    pub audit: GwpAudit,
}

/// Guard decision returned by a workflow hook.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GwpDecision {
    /// Allow the request or response unchanged.
    Allow,
    /// Block the request or response.
    Block,
    /// Replace the request messages before forwarding.
    Replace,
    /// Defer the decision to a parent or supervisory policy.
    DeferToParent,
    /// Fail closed after a runtime failure such as timeout, malformed output, or non-zero exit.
    ErrorFailClosed,
}

/// Audit metadata returned by a guard workflow.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GwpAudit {
    /// Policy-defined evidence spans.
    pub evidence_spans: Vec<serde_json::Value>,
    /// Policy-defined audit notes.
    pub notes: Vec<String>,
}

/// Trace disclosure mode for a guard invocation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GwpTraceMode {
    /// Sensitive fields are redacted before workflow invocation.
    Redacted,
    /// Full payload is provided to the workflow.
    Full,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        GWP_PROTOCOL_VERSION, GwpAudit, GwpDecision, GwpHook, GwpInvocation, GwpProfile,
        GwpProfileKind, GwpResult, GwpTraceMode,
    };

    #[test]
    fn invocation_round_trips_through_json() {
        let invocation = GwpInvocation {
            protocol_version: GWP_PROTOCOL_VERSION.to_owned(),
            hook: GwpHook::PreRequestGuard,
            request_id: "req_123".to_owned(),
            profile: GwpProfile {
                id: "profile_child".to_owned(),
                kind: GwpProfileKind::Child,
            },
            model_alias: "aeon-ultimate".to_owned(),
            messages: vec![json!({
                "role": "user",
                "content": "hello"
            })],
            policy: json!({
                "mode": "strict"
            }),
            budgets: json!({
                "timeout_ms": 1000
            }),
            trace_mode: GwpTraceMode::Redacted,
        };

        let encoded = serde_json::to_string(&invocation).expect("invocation should serialize");
        let decoded: GwpInvocation =
            serde_json::from_str(&encoded).expect("invocation should deserialize");

        assert_eq!(decoded, invocation);
        assert_eq!(decoded.protocol_version, GWP_PROTOCOL_VERSION);
    }

    #[test]
    fn result_round_trips_through_json() {
        let result = GwpResult {
            decision: GwpDecision::Replace,
            risk_level: "medium".to_owned(),
            tags: vec!["pii".to_owned(), "rewrite".to_owned()],
            summary: "rewrote sensitive content".to_owned(),
            replacement_messages: Some(vec![json!({
                "role": "user",
                "content": "redacted"
            })]),
            audit: GwpAudit {
                evidence_spans: vec![json!({
                    "path": "/messages/0/content",
                    "kind": "pii"
                })],
                notes: vec!["matched policy rule pii.redact".to_owned()],
            },
        };

        let encoded = serde_json::to_string(&result).expect("result should serialize");
        let decoded: GwpResult = serde_json::from_str(&encoded).expect("result should deserialize");

        assert_eq!(decoded, result);
    }

    #[test]
    fn error_fail_closed_serializes_as_snake_case() {
        let encoded = serde_json::to_string(&GwpDecision::ErrorFailClosed)
            .expect("decision should serialize");

        assert_eq!(encoded, "\"error_fail_closed\"");
    }
}
