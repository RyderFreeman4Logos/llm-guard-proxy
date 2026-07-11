use std::{collections::BTreeMap, pin::Pin, time::Duration};

use axum::body::Bytes;
use bytes::BytesMut;
use futures_util::{Stream, StreamExt};
use llm_guard_proxy_core::{
    DefaultInjectionSchema, NoThinkingMarkerPolicy, RawPayloadChunk, RawPayloads, StreamChannel,
    ThinkingConfig, ThinkingMode, ToolRequestThinkingPolicy,
};
use serde_json::{Map, Number, Value, json};
use tokio::time::timeout;

use super::{
    MAX_PROXY_BODY_BYTES, UpstreamStreamTimeouts, sanitized_reqwest_error, unix_time_millis,
};

mod loop_guard;
pub(super) use loop_guard::{AggregationError, LoopInspectionContext};
use loop_guard::{LoopDetector, observe_completed_tool_call, observe_fragment};

const RAW_STREAM_CHUNK_LIMIT: usize = 2_048;

/// Prepared upstream request body for shielded non-stream chat completion handling.
pub(super) struct PreparedChatRequest {
    upstream_body: Bytes,
    thinking_metadata: BTreeMap<String, String>,
}

impl PreparedChatRequest {
    pub(super) fn upstream_body(&self) -> Bytes {
        self.upstream_body.clone()
    }

    pub(super) fn thinking_metadata(&self) -> &BTreeMap<String, String> {
        &self.thinking_metadata
    }
}

/// Forces upstream streaming and usage frames for non-stream chat requests parsed as JSON.
pub(super) fn prepare_non_stream_request(
    body: &Bytes,
    thinking: &ThinkingConfig,
) -> Option<PreparedChatRequest> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let object = value.as_object_mut()?;
    if let Some(stream) = object.get("stream") {
        if !matches!(stream, Value::Bool(false)) {
            return None;
        }
    }
    if object
        .get("stream_options")
        .is_some_and(|stream_options| !stream_options.is_object())
    {
        return None;
    }

    let thinking_metadata = apply_thinking_policy(object, thinking);
    object.insert(String::from("stream"), Value::Bool(true));
    let stream_options = object
        .entry(String::from("stream_options"))
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(stream_options) = stream_options {
        stream_options.insert(String::from("include_usage"), Value::Bool(true));
    } else {
        *stream_options = json!({ "include_usage": true });
    }
    let upstream_body = serde_json::to_vec(&value).ok()?;

    Some(PreparedChatRequest {
        upstream_body: Bytes::from(upstream_body),
        thinking_metadata,
    })
}

/// Applies thinking-budget shielding to downstream streaming chat requests.
///
/// Streaming requests keep their streaming response contract; unlike
/// [`prepare_non_stream_request`], this does not force upstream usage frames or
/// route the response through non-stream aggregation.
pub(super) fn prepare_stream_request(
    body: &Bytes,
    thinking: &ThinkingConfig,
) -> Option<PreparedChatRequest> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let object = value.as_object_mut()?;
    if !matches!(object.get("stream"), Some(Value::Bool(true))) {
        return None;
    }

    let thinking_metadata = apply_thinking_policy(object, thinking);
    let upstream_body = serde_json::to_vec(&value).ok()?;

    Some(PreparedChatRequest {
        upstream_body: Bytes::from(upstream_body),
        thinking_metadata,
    })
}

/// Returns a retry request body with a bounded anti-loop system hint.
///
/// The hint is deterministic and contains only proxy retry metadata; it never
/// copies raw prompt, output, reasoning, or upstream error text.
///
/// If the request already has a leading `system` message, the hint is merged
/// into that message's content. Otherwise a new system message is inserted at
/// index 0. This preserves the `OpenAI` chat-template invariant that `system`
/// messages must only appear at the beginning of the message array — models
/// such as Qwen3 reject non-leading system messages at the template level.
pub(super) fn body_with_anti_loop_retry_hint(
    body: &Bytes,
    attempt_number: u32,
    max_attempts: u32,
    configured_hint: Option<&str>,
) -> Option<Bytes> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let object = value.as_object_mut()?;
    let messages = object.get_mut("messages")?.as_array_mut()?;
    let hint = configured_hint.map_or_else(
        || anti_loop_retry_hint(attempt_number, max_attempts),
        str::to_owned,
    );
    if messages
        .first()
        .and_then(|msg| msg.get("role"))
        .and_then(Value::as_str)
        .is_some_and(|role| role == "system")
    {
        merge_hint_into_existing_system_message(&mut messages[0], &hint);
    } else {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": hint,
            }),
        );
    }
    serde_json::to_vec(&value).ok().map(Bytes::from)
}

/// Returns a retry request body with bounded private reasoning salvage context.
///
/// The salvage context is sent only to the upstream retry attempt. It is not
/// released downstream by this function, and the prompt explicitly instructs
/// the model not to quote or continue the private notes.
pub(super) fn body_with_cot_salvage_retry_hint(
    body: &Bytes,
    attempt_number: u32,
    max_attempts: u32,
    policy: &str,
    reasoning_prefix: &str,
    configured_hint: Option<&str>,
) -> Option<Bytes> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let object = value.as_object_mut()?;
    let messages = object.get_mut("messages")?.as_array_mut()?;
    let hint = cot_salvage_retry_hint(
        attempt_number,
        max_attempts,
        policy,
        reasoning_prefix,
        configured_hint,
    );
    if messages
        .first()
        .and_then(|msg| msg.get("role"))
        .and_then(Value::as_str)
        .is_some_and(|role| role == "system")
    {
        merge_hint_into_existing_system_message(&mut messages[0], &hint);
    } else {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": hint,
            }),
        );
    }
    serde_json::to_vec(&value).ok().map(Bytes::from)
}

/// Appends the anti-loop retry hint to the `content` field of an existing
/// system message, separated by a blank line. Handles both string and
/// multi-part (array) content shapes.
fn merge_hint_into_existing_system_message(system_message: &mut Value, hint: &str) {
    let Some(content) = system_message.get_mut("content") else {
        return;
    };
    match content {
        Value::String(existing) => {
            if !existing.is_empty() {
                existing.push_str("\n\n");
            }
            existing.push_str(hint);
        }
        Value::Array(parts) => {
            // OpenAI multi-part content: append a text part.
            parts.push(json!({ "type": "text", "text": hint }));
        }
        // Non-standard content shape — replace with a string to guarantee the
        // hint reaches the model.
        _ => {
            *content = Value::String(hint.to_owned());
        }
    }
}

fn anti_loop_retry_hint(attempt_number: u32, max_attempts: u32) -> String {
    format!(
        "llm-guard-proxy retry hint: a prior shielded upstream attempt was aborted by loop protection. Avoid repeating the same output pattern and provide a concise fresh answer. retry_attempt={attempt_number}/{max_attempts}."
    )
}

fn cot_salvage_retry_hint(
    attempt_number: u32,
    max_attempts: u32,
    policy: &str,
    reasoning_prefix: &str,
    configured_hint: Option<&str>,
) -> String {
    let operator_hint = configured_hint.unwrap_or(
        "Use the private notes only to preserve useful intermediate work. Do not quote, reveal, or continue the notes. Answer the original user request directly.",
    );
    format!(
        "llm-guard-proxy CoT salvage retry hint: a prior shielded upstream attempt was aborted by reasoning-loop protection. policy={policy} retry_attempt={attempt_number}/{max_attempts}.\n\n{operator_hint}\n\nPrivate bounded pre-loop reasoning notes:\n{reasoning_prefix}"
    )
}

#[derive(Clone, Copy, Debug)]
struct JsonPath {
    path: &'static [&'static str],
    variant: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct BudgetObservation {
    path: JsonPath,
    state: BudgetState,
}

#[derive(Clone, Debug, Default)]
struct BudgetObservations {
    entries: Vec<BudgetObservation>,
}

#[derive(Clone, Copy, Debug)]
enum BudgetState {
    Numeric(u64),
    Malformed,
}

#[derive(Clone, Copy, Debug)]
enum DisableMarker {
    None,
    Disabled(JsonPath),
    Malformed(JsonPath),
}

#[derive(Clone, Copy, Debug)]
struct NoThinkingMarkerPath {
    path: JsonPath,
    source: &'static str,
}

#[derive(Clone, Debug, Default)]
struct NoThinkingMarkerDetection {
    detected: bool,
    source: &'static str,
    is_escape_hatch: bool,
}

#[derive(Clone, Debug, Default)]
struct AnswerBudgetDecision {
    applied: bool,
    adjusted_fields: Vec<&'static str>,
    preserved_fields: Vec<&'static str>,
    malformed_fields: Vec<&'static str>,
    overflow_fields: Vec<&'static str>,
}

const CANONICAL_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["thinking", "budget_tokens"],
    variant: "canonical",
};

const ROOT_THINKING_TOKEN_BUDGET: JsonPath = JsonPath {
    path: &["thinking_token_budget"],
    variant: "root-thinking-token-budget",
};

const ROOT_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["thinking_budget"],
    variant: "root-thinking-budget",
};

const ROOT_CHAT_TEMPLATE_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["chat_template_kwargs", "thinking_budget"],
    variant: "chat-template-kwargs",
};

const EXTRA_BODY_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["extra_body", "thinking_budget"],
    variant: "extra-body-thinking-budget",
};

const EXTRA_BODY_THINKING_TOKEN_BUDGET: JsonPath = JsonPath {
    path: &["extra_body", "thinking_token_budget"],
    variant: "extra-body-thinking-token-budget",
};

const EXTRA_BODY_CANONICAL_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["extra_body", "thinking", "budget_tokens"],
    variant: "extra-body-canonical",
};

const EXTRA_BODY_CHAT_TEMPLATE_THINKING_BUDGET: JsonPath = JsonPath {
    path: &["extra_body", "chat_template_kwargs", "thinking_budget"],
    variant: "extra-body-chat-template-kwargs",
};

const BUDGET_PATHS: &[JsonPath] = &[
    CANONICAL_THINKING_BUDGET,
    ROOT_THINKING_TOKEN_BUDGET,
    ROOT_THINKING_BUDGET,
    ROOT_CHAT_TEMPLATE_THINKING_BUDGET,
    EXTRA_BODY_THINKING_TOKEN_BUDGET,
    EXTRA_BODY_THINKING_BUDGET,
    EXTRA_BODY_CANONICAL_THINKING_BUDGET,
    EXTRA_BODY_CHAT_TEMPLATE_THINKING_BUDGET,
];

const ROOT_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["enable_thinking"],
    variant: "root-enable-thinking",
};

const THINKING_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["thinking", "enable_thinking"],
    variant: "canonical-enable-thinking",
};

const THINKING_ENABLED: JsonPath = JsonPath {
    path: &["thinking", "enabled"],
    variant: "canonical-thinking-enabled",
};

const ROOT_CHAT_TEMPLATE_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["chat_template_kwargs", "enable_thinking"],
    variant: "chat-template-kwargs-enable-thinking",
};

const EXTRA_BODY_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["extra_body", "enable_thinking"],
    variant: "extra-body-enable-thinking",
};

const EXTRA_BODY_THINKING_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["extra_body", "thinking", "enable_thinking"],
    variant: "extra-body-canonical-enable-thinking",
};

const EXTRA_BODY_THINKING_ENABLED: JsonPath = JsonPath {
    path: &["extra_body", "thinking", "enabled"],
    variant: "extra-body-canonical-thinking-enabled",
};

const EXTRA_BODY_CHAT_TEMPLATE_ENABLE_THINKING: JsonPath = JsonPath {
    path: &["extra_body", "chat_template_kwargs", "enable_thinking"],
    variant: "extra-body-chat-template-kwargs-enable-thinking",
};

const DISABLE_MARKER_PATHS: &[JsonPath] = &[
    ROOT_ENABLE_THINKING,
    THINKING_ENABLE_THINKING,
    THINKING_ENABLED,
    ROOT_CHAT_TEMPLATE_ENABLE_THINKING,
    EXTRA_BODY_ENABLE_THINKING,
    EXTRA_BODY_THINKING_ENABLE_THINKING,
    EXTRA_BODY_THINKING_ENABLED,
    EXTRA_BODY_CHAT_TEMPLATE_ENABLE_THINKING,
];

const ROOT_REASONING_EFFORT: JsonPath = JsonPath {
    path: &["reasoning_effort"],
    variant: "root-reasoning-effort",
};

const EXTRA_BODY_REASONING_EFFORT: JsonPath = JsonPath {
    path: &["extra_body", "reasoning_effort"],
    variant: "extra-body-reasoning-effort",
};

const ROOT_MODEL_REASONING_EFFORT: JsonPath = JsonPath {
    path: &["model_reasoning_effort"],
    variant: "root-model-reasoning-effort",
};

const EXTRA_BODY_MODEL_REASONING_EFFORT: JsonPath = JsonPath {
    path: &["extra_body", "model_reasoning_effort"],
    variant: "extra-body-model-reasoning-effort",
};

const ROOT_DISABLE_THINKING_ESCAPE_HATCH: JsonPath = JsonPath {
    path: &["llm_guard_proxy_disable_thinking"],
    variant: "root-disable-thinking-escape-hatch",
};

const EXTRA_BODY_DISABLE_THINKING_ESCAPE_HATCH: JsonPath = JsonPath {
    path: &["extra_body", "llm_guard_proxy_disable_thinking"],
    variant: "extra-body-disable-thinking-escape-hatch",
};

const ESCAPE_HATCH_NO_THINKING_MARKERS: &[NoThinkingMarkerPath] = &[
    NoThinkingMarkerPath {
        path: ROOT_DISABLE_THINKING_ESCAPE_HATCH,
        source: "llm_guard_proxy_disable_thinking",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_DISABLE_THINKING_ESCAPE_HATCH,
        source: "extra_body.llm_guard_proxy_disable_thinking",
    },
];

const BOOLEAN_NO_THINKING_MARKERS: &[NoThinkingMarkerPath] = &[
    NoThinkingMarkerPath {
        path: ROOT_ENABLE_THINKING,
        source: "enable_thinking",
    },
    NoThinkingMarkerPath {
        path: THINKING_ENABLE_THINKING,
        source: "thinking.enable_thinking",
    },
    NoThinkingMarkerPath {
        path: THINKING_ENABLED,
        source: "thinking.enabled",
    },
    NoThinkingMarkerPath {
        path: ROOT_CHAT_TEMPLATE_ENABLE_THINKING,
        source: "chat_template_kwargs.enable_thinking",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_ENABLE_THINKING,
        source: "extra_body.enable_thinking",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_THINKING_ENABLE_THINKING,
        source: "extra_body.thinking.enable_thinking",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_THINKING_ENABLED,
        source: "extra_body.thinking.enabled",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_CHAT_TEMPLATE_ENABLE_THINKING,
        source: "extra_body.chat_template_kwargs.enable_thinking",
    },
];

const REASONING_EFFORT_NO_THINKING_MARKERS: &[NoThinkingMarkerPath] = &[
    NoThinkingMarkerPath {
        path: ROOT_REASONING_EFFORT,
        source: "reasoning_effort.none",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_REASONING_EFFORT,
        source: "extra_body.reasoning_effort.none",
    },
    NoThinkingMarkerPath {
        path: ROOT_MODEL_REASONING_EFFORT,
        source: "model_reasoning_effort.none",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_MODEL_REASONING_EFFORT,
        source: "extra_body.model_reasoning_effort.none",
    },
];

const BUDGET_NO_THINKING_MARKERS: &[NoThinkingMarkerPath] = &[
    NoThinkingMarkerPath {
        path: CANONICAL_THINKING_BUDGET,
        source: "thinking.budget_tokens.zero",
    },
    NoThinkingMarkerPath {
        path: ROOT_THINKING_TOKEN_BUDGET,
        source: "thinking_token_budget.zero",
    },
    NoThinkingMarkerPath {
        path: ROOT_THINKING_BUDGET,
        source: "thinking_budget.zero",
    },
    NoThinkingMarkerPath {
        path: ROOT_CHAT_TEMPLATE_THINKING_BUDGET,
        source: "chat_template_kwargs.thinking_budget.zero",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_THINKING_TOKEN_BUDGET,
        source: "extra_body.thinking_token_budget.zero",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_THINKING_BUDGET,
        source: "extra_body.thinking_budget.zero",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_CANONICAL_THINKING_BUDGET,
        source: "extra_body.thinking.budget_tokens.zero",
    },
    NoThinkingMarkerPath {
        path: EXTRA_BODY_CHAT_TEMPLATE_THINKING_BUDGET,
        source: "extra_body.chat_template_kwargs.thinking_budget.zero",
    },
];

const ANSWER_BUDGET_FIELDS: &[&str] = &["max_tokens", "max_completion_tokens", "max_output_tokens"];

#[derive(Debug)]
struct ThinkingPolicyOutcome {
    rewrite_applied: bool,
    reason: &'static str,
    final_budget: String,
    answer_budget_delta: u64,
    rewritten_paths: Vec<JsonPath>,
    preserved_paths: Vec<JsonPath>,
    malformed_paths: Vec<JsonPath>,
    zero_paths: Vec<JsonPath>,
}

fn apply_thinking_policy(
    object: &mut Map<String, Value>,
    thinking: &ThinkingConfig,
) -> BTreeMap<String, String> {
    let configured_budget = u64::from(thinking.budget_tokens);
    let budget_observations = find_budget_observations(object);
    let disable_marker = find_disable_marker(object);
    let is_tool_request = is_tool_use_request(object);
    let mut metadata = initial_thinking_metadata(thinking, configured_budget, &budget_observations);
    metadata.insert(
        String::from("thinking_tool_request_detected"),
        is_tool_request.to_string(),
    );
    if !thinking.enabled && !thinking.force_disable {
        let outcome = thinking_noop("policy_disabled", &budget_observations);
        metadata.extend(thinking_outcome_metadata(
            &outcome,
            &AnswerBudgetDecision::default(),
        ));
        return metadata;
    }
    match thinking.effective_mode() {
        ThinkingMode::Passthrough => {
            let outcome = thinking_noop("mode_passthrough", &budget_observations);
            metadata.extend(thinking_outcome_metadata(
                &outcome,
                &AnswerBudgetDecision::default(),
            ));
            return metadata;
        }
        ThinkingMode::ForceDisable => {
            let outcome = apply_force_disable_thinking(
                object,
                &budget_observations,
                &mut metadata,
                thinking.default_injection_schema,
            );
            let answer_budget =
                apply_answer_budget_preservation(object, false, outcome.answer_budget_delta);
            metadata.extend(thinking_outcome_metadata(&outcome, &answer_budget));
            return metadata;
        }
        ThinkingMode::ForceThinking | ThinkingMode::BoundedThinking => {}
    }
    if is_tool_request
        && matches!(
            thinking.tool_request_policy,
            ToolRequestThinkingPolicy::Passthrough
        )
    {
        let outcome = thinking_noop("tool_request_passthrough", &budget_observations);
        metadata.extend(thinking_outcome_metadata(
            &outcome,
            &AnswerBudgetDecision::default(),
        ));
        return metadata;
    }
    if record_force_thinking_marker_decision(object, thinking, &budget_observations, &mut metadata)
    {
        return metadata;
    }
    let outcome = match thinking.effective_mode() {
        ThinkingMode::ForceThinking => apply_force_thinking_policy(
            object,
            configured_budget,
            &budget_observations,
            &mut metadata,
            thinking.default_injection_schema,
        ),
        ThinkingMode::BoundedThinking => apply_thinking_budget_policy(
            object,
            thinking,
            configured_budget,
            &budget_observations,
            disable_marker,
            &mut metadata,
        ),
        ThinkingMode::Passthrough | ThinkingMode::ForceDisable => {
            unreachable!("passthrough and force-disable modes return before budget rewriting")
        }
    };
    let answer_budget = apply_answer_budget_preservation(
        object,
        thinking.preserve_answer_budget,
        outcome.answer_budget_delta,
    );
    let answer_budget = apply_configured_total_cap(object, thinking, answer_budget);

    metadata.extend(thinking_outcome_metadata(&outcome, &answer_budget));
    metadata
}

fn is_tool_use_request(object: &Map<String, Value>) -> bool {
    non_empty_array(object.get("tools"))
        || non_empty_array(object.get("functions"))
        || present_tool_selector(object.get("tool_choice"))
        || present_tool_selector(object.get("function_call"))
}

fn non_empty_array(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Array(values)) if !values.is_empty())
}

fn present_tool_selector(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(selector)) => !selector.trim().eq_ignore_ascii_case("none"),
        Some(Value::Null | Value::Bool(false)) | None => false,
        Some(_) => true,
    }
}

fn initial_thinking_metadata(
    thinking: &ThinkingConfig,
    configured_budget: u64,
    budget_observations: &BudgetObservations,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("thinking_policy_enabled"),
            thinking.enabled.to_string(),
        ),
        (
            String::from("thinking_policy_mode"),
            thinking.effective_mode().as_str().to_owned(),
        ),
        (
            String::from("thinking_force_disable_enabled"),
            thinking.force_disable.to_string(),
        ),
        (
            String::from("thinking_policy_max_tokens"),
            thinking
                .max_tokens
                .map_or_else(|| String::from("none"), |max_tokens| max_tokens.to_string()),
        ),
        (
            String::from("thinking_policy_budget_tokens"),
            configured_budget.to_string(),
        ),
        (
            String::from("thinking_preserve_answer_budget_enabled"),
            thinking.preserve_answer_budget.to_string(),
        ),
        (
            String::from("thinking_budget_accounting"),
            thinking.budget_accounting().to_owned(),
        ),
        (
            String::from("thinking_tool_request_policy"),
            thinking.tool_request_policy.as_str().to_owned(),
        ),
        (
            String::from("thinking_no_thinking_marker_policy"),
            thinking.no_thinking_marker_policy.as_str().to_owned(),
        ),
        (
            String::from("thinking_default_injection_schema"),
            thinking.default_injection_schema.as_str().to_owned(),
        ),
        (
            String::from("thinking_budget_previous_state"),
            previous_budget_state(budget_observations, configured_budget),
        ),
        (
            String::from("thinking_budget_previous_tokens"),
            previous_budget_tokens(budget_observations),
        ),
        (
            String::from("thinking_schema_path"),
            budget_observations.single_path().map_or_else(
                || {
                    if budget_observations.is_empty() {
                        String::from("none")
                    } else {
                        String::from("multiple")
                    }
                },
                JsonPath::display_path,
            ),
        ),
        (
            String::from("thinking_schema_variant"),
            budget_observations.single_path().map_or_else(
                || {
                    if budget_observations.is_empty() {
                        String::from("none")
                    } else {
                        String::from("multiple")
                    }
                },
                |path| path.variant.to_owned(),
            ),
        ),
        (
            String::from("thinking_budget_observed_paths"),
            observed_budget_paths(budget_observations, configured_budget),
        ),
    ])
}

/// Evaluates the no-thinking marker policy for `ForceThinking` mode.
///
/// Returns `true` if the request bypasses force-thinking (caller marker
/// passthrough), in which case the caller should return immediately.
/// Returns `false` if force-thinking should proceed. When a marker was
/// detected but overridden, records observability metadata and returns
/// `false`.
fn record_force_thinking_marker_decision(
    object: &Map<String, Value>,
    thinking: &ThinkingConfig,
    budget_observations: &BudgetObservations,
    metadata: &mut BTreeMap<String, String>,
) -> bool {
    if thinking.effective_mode() != ThinkingMode::ForceThinking {
        return false;
    }
    let marker = detect_no_thinking_markers(object);
    let should_bypass = match thinking.no_thinking_marker_policy {
        NoThinkingMarkerPolicy::Force => false,
        NoThinkingMarkerPolicy::RespectNoThinkingMarkers => marker.detected,
        NoThinkingMarkerPolicy::EscapeHatchOnly => marker.is_escape_hatch,
    };
    if should_bypass {
        metadata.insert(
            String::from("thinking_no_thinking_marker_detected"),
            marker.detected.to_string(),
        );
        metadata.insert(
            String::from("thinking_no_thinking_marker_source"),
            marker.source.to_owned(),
        );
        metadata.insert(
            String::from("thinking_no_thinking_marker_escape_hatch"),
            marker.is_escape_hatch.to_string(),
        );
        let outcome = thinking_noop("caller_no_thinking_marker_passthrough", budget_observations);
        metadata.extend(thinking_outcome_metadata(
            &outcome,
            &AnswerBudgetDecision::default(),
        ));
        return true;
    }
    if marker.detected {
        metadata.insert(
            String::from("thinking_no_thinking_marker_detected"),
            String::from("true"),
        );
        metadata.insert(
            String::from("thinking_no_thinking_marker_source"),
            marker.source.to_owned(),
        );
        metadata.insert(
            String::from("thinking_no_thinking_marker_escape_hatch"),
            marker.is_escape_hatch.to_string(),
        );
        metadata.insert(
            String::from("thinking_no_thinking_marker_overridden"),
            String::from("true"),
        );
    }
    false
}

fn apply_force_thinking_policy(
    object: &mut Map<String, Value>,
    configured_budget: u64,
    budget_observations: &BudgetObservations,
    metadata: &mut BTreeMap<String, String>,
    default_schema: DefaultInjectionSchema,
) -> ThinkingPolicyOutcome {
    if configured_budget == 0 {
        return thinking_noop("configured_budget_zero", budget_observations);
    }

    let enable_marker_paths = normalize_force_enable_markers(object, metadata);
    metadata.insert(
        String::from("thinking_enable_marker_rewritten_paths"),
        join_paths(&enable_marker_paths),
    );

    if budget_observations.is_empty() {
        let mut outcome =
            inject_missing_budget(object, configured_budget, metadata, default_schema);
        outcome.reason = "forced_configured_budget";
        outcome.rewrite_applied = outcome.rewrite_applied || !enable_marker_paths.is_empty();
        return outcome;
    }

    let mut rewritten_paths = Vec::new();
    let mut preserved_paths = Vec::new();
    let mut malformed_paths = Vec::new();
    let mut answer_budget_delta = 0;
    for observation in &budget_observations.entries {
        match observation.state {
            BudgetState::Numeric(existing_budget) if existing_budget == configured_budget => {
                preserved_paths.push(observation.path);
            }
            BudgetState::Numeric(existing_budget) => {
                if set_budget_at_path(object, observation.path, configured_budget) {
                    rewritten_paths.push(observation.path);
                    answer_budget_delta =
                        answer_budget_delta.max(configured_budget.saturating_sub(existing_budget));
                } else {
                    malformed_paths.push(observation.path);
                }
            }
            BudgetState::Malformed => {
                if set_budget_at_path(object, observation.path, configured_budget) {
                    rewritten_paths.push(observation.path);
                    answer_budget_delta = answer_budget_delta.max(configured_budget);
                } else {
                    malformed_paths.push(observation.path);
                }
            }
        }
    }

    ThinkingPolicyOutcome {
        rewrite_applied: !rewritten_paths.is_empty() || !enable_marker_paths.is_empty(),
        reason: "forced_configured_budget",
        final_budget: configured_budget.to_string(),
        answer_budget_delta,
        rewritten_paths,
        preserved_paths,
        malformed_paths,
        zero_paths: Vec::new(),
    }
}

fn apply_thinking_budget_policy(
    object: &mut Map<String, Value>,
    thinking: &ThinkingConfig,
    configured_budget: u64,
    budget_observations: &BudgetObservations,
    disable_marker: DisableMarker,
    metadata: &mut BTreeMap<String, String>,
) -> ThinkingPolicyOutcome {
    if !thinking.enabled {
        return thinking_noop("policy_disabled", budget_observations);
    }
    if configured_budget == 0 {
        return thinking_noop("configured_budget_zero", budget_observations);
    }
    match disable_marker {
        DisableMarker::Disabled(path) => {
            insert_disable_marker_metadata(metadata, path);
            thinking_noop("caller_disabled_thinking", budget_observations)
        }
        DisableMarker::Malformed(path) => {
            insert_disable_marker_metadata(metadata, path);
            thinking_noop("malformed_disable_marker", budget_observations)
        }
        DisableMarker::None => apply_budget_observations(
            object,
            configured_budget,
            budget_observations,
            metadata,
            thinking.default_injection_schema,
        ),
    }
}

fn apply_force_disable_thinking(
    object: &mut Map<String, Value>,
    budget_observations: &BudgetObservations,
    metadata: &mut BTreeMap<String, String>,
    default_schema: DefaultInjectionSchema,
) -> ThinkingPolicyOutcome {
    let disable_marker_paths = normalize_force_disable_markers(object, metadata, default_schema);
    metadata.insert(
        String::from("thinking_disable_marker_rewritten_paths"),
        join_paths(&disable_marker_paths),
    );

    if budget_observations.is_empty() {
        let path = injection_path(object, default_schema);
        metadata.insert(String::from("thinking_schema_path"), path.display_path());
        metadata.insert(
            String::from("thinking_schema_variant"),
            path.variant.to_owned(),
        );
        if set_budget_at_path(object, path, 0) {
            return ThinkingPolicyOutcome {
                rewrite_applied: true,
                reason: "force_disabled_thinking",
                final_budget: String::from("0"),
                answer_budget_delta: 0,
                rewritten_paths: vec![path],
                preserved_paths: Vec::new(),
                malformed_paths: Vec::new(),
                zero_paths: Vec::new(),
            };
        }
        return ThinkingPolicyOutcome {
            rewrite_applied: false,
            reason: "malformed_budget_container",
            final_budget: String::from("unknown"),
            answer_budget_delta: 0,
            rewritten_paths: Vec::new(),
            preserved_paths: Vec::new(),
            malformed_paths: Vec::new(),
            zero_paths: Vec::new(),
        };
    }

    let mut rewritten_paths = Vec::new();
    let mut preserved_paths = Vec::new();
    let mut malformed_paths = Vec::new();
    let mut zero_paths = Vec::new();
    for observation in &budget_observations.entries {
        match observation.state {
            BudgetState::Numeric(0) => {
                preserved_paths.push(observation.path);
                zero_paths.push(observation.path);
            }
            BudgetState::Numeric(_) | BudgetState::Malformed => {
                if set_budget_at_path(object, observation.path, 0) {
                    rewritten_paths.push(observation.path);
                } else {
                    malformed_paths.push(observation.path);
                }
            }
        }
    }

    let final_budget = if rewritten_paths.is_empty() && preserved_paths.is_empty() {
        String::from("unknown")
    } else {
        String::from("0")
    };
    ThinkingPolicyOutcome {
        rewrite_applied: !rewritten_paths.is_empty() || !disable_marker_paths.is_empty(),
        reason: "force_disabled_thinking",
        final_budget,
        answer_budget_delta: 0,
        rewritten_paths,
        preserved_paths,
        malformed_paths,
        zero_paths,
    }
}

fn apply_budget_observations(
    object: &mut Map<String, Value>,
    configured_budget: u64,
    budget_observations: &BudgetObservations,
    metadata: &mut BTreeMap<String, String>,
    default_schema: DefaultInjectionSchema,
) -> ThinkingPolicyOutcome {
    if budget_observations.is_empty() {
        return inject_missing_budget(object, configured_budget, metadata, default_schema);
    }

    let zero_paths = budget_observations.zero_paths();
    if !zero_paths.is_empty() {
        return ThinkingPolicyOutcome {
            rewrite_applied: false,
            reason: "existing_budget_zero",
            final_budget: final_budget_tokens(budget_observations, configured_budget),
            answer_budget_delta: 0,
            rewritten_paths: Vec::new(),
            preserved_paths: budget_observations.non_malformed_paths(),
            malformed_paths: budget_observations.malformed_paths(),
            zero_paths,
        };
    }

    let mut rewritten_paths = Vec::new();
    let mut preserved_paths = Vec::new();
    let mut malformed_paths = Vec::new();
    let mut answer_budget_delta = 0;
    for observation in &budget_observations.entries {
        match observation.state {
            BudgetState::Malformed => malformed_paths.push(observation.path),
            BudgetState::Numeric(existing_budget) if existing_budget < configured_budget => {
                if !set_budget_at_path(object, observation.path, configured_budget) {
                    return malformed_budget_container_outcome(
                        budget_observations,
                        configured_budget,
                    );
                }
                rewritten_paths.push(observation.path);
                answer_budget_delta =
                    answer_budget_delta.max(configured_budget.saturating_sub(existing_budget));
            }
            BudgetState::Numeric(_existing_budget) => preserved_paths.push(observation.path),
        }
    }

    if !rewritten_paths.is_empty() {
        return ThinkingPolicyOutcome {
            rewrite_applied: true,
            reason: "raised_smaller_budget",
            final_budget: final_budget_tokens(budget_observations, configured_budget),
            answer_budget_delta,
            rewritten_paths,
            preserved_paths,
            malformed_paths,
            zero_paths: Vec::new(),
        };
    }

    if !preserved_paths.is_empty() {
        return ThinkingPolicyOutcome {
            rewrite_applied: false,
            reason: "preserved_equal_or_larger_budget",
            final_budget: final_budget_tokens(budget_observations, configured_budget),
            answer_budget_delta: 0,
            rewritten_paths,
            preserved_paths,
            malformed_paths,
            zero_paths: Vec::new(),
        };
    }

    ThinkingPolicyOutcome {
        rewrite_applied: false,
        reason: "malformed_existing_budget",
        final_budget: final_budget_tokens(budget_observations, configured_budget),
        answer_budget_delta: 0,
        rewritten_paths,
        preserved_paths,
        malformed_paths,
        zero_paths: Vec::new(),
    }
}

fn inject_missing_budget(
    object: &mut Map<String, Value>,
    configured_budget: u64,
    metadata: &mut BTreeMap<String, String>,
    default_schema: DefaultInjectionSchema,
) -> ThinkingPolicyOutcome {
    let path = injection_path(object, default_schema);
    metadata.insert(String::from("thinking_schema_path"), path.display_path());
    metadata.insert(
        String::from("thinking_schema_variant"),
        path.variant.to_owned(),
    );
    if set_budget_at_path(object, path, configured_budget) {
        if let Some(enable_path) = chat_template_enable_marker_path(path) {
            if matches!(value_at_path(object, enable_path.path), PathValue::Missing) {
                let _enabled = set_bool_at_path(object, enable_path, true);
            }
        }
        return ThinkingPolicyOutcome {
            rewrite_applied: true,
            reason: "injected_missing_budget",
            final_budget: configured_budget.to_string(),
            answer_budget_delta: configured_budget,
            rewritten_paths: vec![path],
            preserved_paths: Vec::new(),
            malformed_paths: Vec::new(),
            zero_paths: Vec::new(),
        };
    }
    ThinkingPolicyOutcome {
        rewrite_applied: false,
        reason: "malformed_budget_container",
        final_budget: String::from("unknown"),
        answer_budget_delta: 0,
        rewritten_paths: Vec::new(),
        preserved_paths: Vec::new(),
        malformed_paths: Vec::new(),
        zero_paths: Vec::new(),
    }
}

fn malformed_budget_container_outcome(
    budget_observations: &BudgetObservations,
    configured_budget: u64,
) -> ThinkingPolicyOutcome {
    ThinkingPolicyOutcome {
        rewrite_applied: false,
        reason: "malformed_budget_container",
        final_budget: final_budget_tokens(budget_observations, configured_budget),
        answer_budget_delta: 0,
        rewritten_paths: Vec::new(),
        preserved_paths: Vec::new(),
        malformed_paths: budget_observations.malformed_paths(),
        zero_paths: budget_observations.zero_paths(),
    }
}

fn thinking_noop(
    reason: &'static str,
    budget_observations: &BudgetObservations,
) -> ThinkingPolicyOutcome {
    ThinkingPolicyOutcome {
        rewrite_applied: false,
        reason,
        final_budget: current_budget_tokens(budget_observations),
        answer_budget_delta: 0,
        rewritten_paths: Vec::new(),
        preserved_paths: budget_observations.non_malformed_paths(),
        malformed_paths: budget_observations.malformed_paths(),
        zero_paths: budget_observations.zero_paths(),
    }
}

fn insert_disable_marker_metadata(metadata: &mut BTreeMap<String, String>, path: JsonPath) {
    metadata.insert(
        String::from("thinking_disable_marker_path"),
        path.display_path(),
    );
    metadata.insert(
        String::from("thinking_disable_marker_variant"),
        path.variant.to_owned(),
    );
}

fn insert_multiple_disable_marker_metadata(metadata: &mut BTreeMap<String, String>) {
    metadata.insert(
        String::from("thinking_disable_marker_path"),
        String::from("multiple"),
    );
    metadata.insert(
        String::from("thinking_disable_marker_variant"),
        String::from("multiple"),
    );
}

fn thinking_outcome_metadata(
    outcome: &ThinkingPolicyOutcome,
    answer_budget: &AnswerBudgetDecision,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("thinking_rewrite_applied"),
            outcome.rewrite_applied.to_string(),
        ),
        (
            String::from("thinking_rewrite_reason"),
            outcome.reason.to_owned(),
        ),
        (
            String::from("thinking_budget_final_tokens"),
            outcome.final_budget.clone(),
        ),
        (
            String::from("thinking_answer_budget_delta_tokens"),
            outcome.answer_budget_delta.to_string(),
        ),
        (
            String::from("thinking_answer_budget_preservation_applied"),
            answer_budget.applied.to_string(),
        ),
        (
            String::from("thinking_answer_budget_adjusted_fields"),
            join_fields(&answer_budget.adjusted_fields),
        ),
        (
            String::from("thinking_answer_budget_preserved_fields"),
            join_fields(&answer_budget.preserved_fields),
        ),
        (
            String::from("thinking_answer_budget_malformed_fields"),
            join_fields(&answer_budget.malformed_fields),
        ),
        (
            String::from("thinking_answer_budget_overflow_fields"),
            join_fields(&answer_budget.overflow_fields),
        ),
        (
            String::from("thinking_budget_rewritten_paths"),
            join_paths(&outcome.rewritten_paths),
        ),
        (
            String::from("thinking_budget_preserved_paths"),
            join_paths(&outcome.preserved_paths),
        ),
        (
            String::from("thinking_budget_malformed_paths"),
            join_paths(&outcome.malformed_paths),
        ),
        (
            String::from("thinking_budget_zero_paths"),
            join_paths(&outcome.zero_paths),
        ),
    ])
}

fn find_budget_observations(object: &Map<String, Value>) -> BudgetObservations {
    let mut entries = Vec::new();
    for path in BUDGET_PATHS {
        match value_at_path(object, path.path) {
            PathValue::Missing => {}
            PathValue::Malformed => {
                entries.push(BudgetObservation {
                    path: *path,
                    state: BudgetState::Malformed,
                });
            }
            PathValue::Value(value) => {
                entries.push(BudgetObservation {
                    path: *path,
                    state: token_budget_value(value),
                });
            }
        }
    }
    BudgetObservations { entries }
}

fn find_disable_marker(object: &Map<String, Value>) -> DisableMarker {
    let mut malformed = None;
    for path in DISABLE_MARKER_PATHS {
        match value_at_path(object, path.path) {
            PathValue::Malformed => {
                malformed.get_or_insert(*path);
            }
            PathValue::Value(Value::Bool(false)) => return DisableMarker::Disabled(*path),
            PathValue::Missing | PathValue::Value(Value::Bool(true)) => {}
            PathValue::Value(_value) => {
                malformed.get_or_insert(*path);
            }
        }
    }

    malformed.map_or(DisableMarker::None, DisableMarker::Malformed)
}

fn detect_no_thinking_markers(object: &Map<String, Value>) -> NoThinkingMarkerDetection {
    for marker in ESCAPE_HATCH_NO_THINKING_MARKERS {
        if matches!(
            value_at_path(object, marker.path.path),
            PathValue::Value(Value::Bool(true))
        ) {
            return NoThinkingMarkerDetection {
                detected: true,
                source: marker.source,
                is_escape_hatch: true,
            };
        }
    }

    for marker in BOOLEAN_NO_THINKING_MARKERS {
        if matches!(
            value_at_path(object, marker.path.path),
            PathValue::Value(Value::Bool(false))
        ) {
            return NoThinkingMarkerDetection {
                detected: true,
                source: marker.source,
                is_escape_hatch: false,
            };
        }
    }

    for marker in REASONING_EFFORT_NO_THINKING_MARKERS {
        if matches!(
            value_at_path(object, marker.path.path),
            PathValue::Value(Value::String(value)) if value.trim().eq_ignore_ascii_case("none")
        ) {
            return NoThinkingMarkerDetection {
                detected: true,
                source: marker.source,
                is_escape_hatch: false,
            };
        }
    }

    for marker in BUDGET_NO_THINKING_MARKERS {
        if matches!(
            value_at_path(object, marker.path.path),
            PathValue::Value(value) if matches!(token_budget_value(value), BudgetState::Numeric(0))
        ) {
            return NoThinkingMarkerDetection {
                detected: true,
                source: marker.source,
                is_escape_hatch: false,
            };
        }
    }

    NoThinkingMarkerDetection::default()
}

enum PathValue<'a> {
    Missing,
    Malformed,
    Value(&'a Value),
}

fn value_at_path<'a>(object: &'a Map<String, Value>, path: &[&str]) -> PathValue<'a> {
    let Some((&last, parents)) = path.split_last() else {
        return PathValue::Missing;
    };
    let mut current = object;
    for key in parents {
        match current.get(*key) {
            Some(Value::Object(next)) => current = next,
            Some(_value) => return PathValue::Malformed,
            None => return PathValue::Missing,
        }
    }
    current
        .get(last)
        .map_or(PathValue::Missing, PathValue::Value)
}

fn token_budget_value(value: &Value) -> BudgetState {
    match value {
        Value::Number(number) => number
            .as_u64()
            .map_or(BudgetState::Malformed, BudgetState::Numeric),
        _ => BudgetState::Malformed,
    }
}

fn injection_path(object: &Map<String, Value>, default_schema: DefaultInjectionSchema) -> JsonPath {
    if object_at_path(object, &["chat_template_kwargs"]) {
        return ROOT_CHAT_TEMPLATE_THINKING_BUDGET;
    }
    if object_at_path(object, &["extra_body", "chat_template_kwargs"]) {
        return EXTRA_BODY_CHAT_TEMPLATE_THINKING_BUDGET;
    }
    if object_at_path(object, &["extra_body", "thinking"]) {
        return EXTRA_BODY_CANONICAL_THINKING_BUDGET;
    }
    match default_schema {
        DefaultInjectionSchema::Canonical => CANONICAL_THINKING_BUDGET,
        DefaultInjectionSchema::ChatTemplateKwargs => ROOT_CHAT_TEMPLATE_THINKING_BUDGET,
    }
}

fn force_disable_marker_path(
    object: &Map<String, Value>,
    default_schema: DefaultInjectionSchema,
) -> JsonPath {
    if object_at_path(object, &["chat_template_kwargs"]) {
        return ROOT_CHAT_TEMPLATE_ENABLE_THINKING;
    }
    if object_at_path(object, &["extra_body", "chat_template_kwargs"]) {
        return EXTRA_BODY_CHAT_TEMPLATE_ENABLE_THINKING;
    }
    if object_at_path(object, &["extra_body", "thinking"]) {
        return EXTRA_BODY_THINKING_ENABLED;
    }
    if object_at_path(object, &["thinking"]) {
        return THINKING_ENABLED;
    }
    if default_schema == DefaultInjectionSchema::ChatTemplateKwargs {
        return ROOT_CHAT_TEMPLATE_ENABLE_THINKING;
    }
    if object_at_path(object, &["extra_body"]) {
        return EXTRA_BODY_ENABLE_THINKING;
    }
    THINKING_ENABLED
}

fn chat_template_enable_marker_path(budget_path: JsonPath) -> Option<JsonPath> {
    if budget_path.path == ROOT_CHAT_TEMPLATE_THINKING_BUDGET.path {
        return Some(ROOT_CHAT_TEMPLATE_ENABLE_THINKING);
    }
    if budget_path.path == EXTRA_BODY_CHAT_TEMPLATE_THINKING_BUDGET.path {
        return Some(EXTRA_BODY_CHAT_TEMPLATE_ENABLE_THINKING);
    }
    None
}

fn object_at_path(object: &Map<String, Value>, path: &[&str]) -> bool {
    let mut current = object;
    for key in path {
        match current.get(*key) {
            Some(Value::Object(next)) => current = next,
            Some(_) | None => return false,
        }
    }
    true
}

fn normalize_force_disable_markers(
    object: &mut Map<String, Value>,
    metadata: &mut BTreeMap<String, String>,
    default_schema: DefaultInjectionSchema,
) -> Vec<JsonPath> {
    let mut rewritten_paths = Vec::new();
    let mut disabled_marker_present = false;
    for path in DISABLE_MARKER_PATHS {
        match value_at_path(object, path.path) {
            PathValue::Value(Value::Bool(false)) => disabled_marker_present = true,
            PathValue::Value(Value::Bool(true)) => {
                if set_bool_at_path(object, *path, false) {
                    rewritten_paths.push(*path);
                }
            }
            PathValue::Missing | PathValue::Malformed | PathValue::Value(_) => {}
        }
    }

    if rewritten_paths.is_empty() && !disabled_marker_present {
        let path = force_disable_marker_path(object, default_schema);
        if set_bool_at_path(object, path, false) {
            rewritten_paths.push(path);
        }
    }

    match rewritten_paths.as_slice() {
        [] => {}
        [path] => insert_disable_marker_metadata(metadata, *path),
        [_first, ..] => insert_multiple_disable_marker_metadata(metadata),
    }

    rewritten_paths
}

fn normalize_force_enable_markers(
    object: &mut Map<String, Value>,
    metadata: &mut BTreeMap<String, String>,
) -> Vec<JsonPath> {
    let mut rewritten_paths = Vec::new();
    for path in DISABLE_MARKER_PATHS {
        if matches!(
            value_at_path(object, path.path),
            PathValue::Value(Value::Bool(false))
        ) && set_bool_at_path(object, *path, true)
        {
            rewritten_paths.push(*path);
        }
    }

    match rewritten_paths.as_slice() {
        [] => {}
        [path] => insert_disable_marker_metadata(metadata, *path),
        [_first, ..] => insert_multiple_disable_marker_metadata(metadata),
    }

    rewritten_paths
}

fn set_budget_at_path(object: &mut Map<String, Value>, path: JsonPath, budget: u64) -> bool {
    let Some((&last, parents)) = path.path.split_last() else {
        return false;
    };
    let mut current = object;
    for key in parents {
        let entry = current
            .entry((*key).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(next) = entry else {
            return false;
        };
        current = next;
    }
    current.insert(last.to_owned(), Value::Number(Number::from(budget)));
    true
}

fn set_bool_at_path(object: &mut Map<String, Value>, path: JsonPath, value: bool) -> bool {
    let Some((&last, parents)) = path.path.split_last() else {
        return false;
    };
    let mut current = object;
    for key in parents {
        let entry = current
            .entry((*key).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(next) = entry else {
            return false;
        };
        current = next;
    }
    current.insert(last.to_owned(), Value::Bool(value));
    true
}

fn apply_answer_budget_preservation(
    object: &mut Map<String, Value>,
    preserve_answer_budget: bool,
    delta: u64,
) -> AnswerBudgetDecision {
    let mut decision = AnswerBudgetDecision::default();
    for field in ANSWER_BUDGET_FIELDS {
        let Some(value) = object.get_mut(*field) else {
            continue;
        };
        if delta == 0 || !preserve_answer_budget {
            decision.preserved_fields.push(field);
            continue;
        }
        let Some(existing) = value.as_u64() else {
            decision.malformed_fields.push(field);
            continue;
        };
        let Some(adjusted) = existing.checked_add(delta) else {
            decision.overflow_fields.push(field);
            continue;
        };
        *value = Value::Number(Number::from(adjusted));
        decision.adjusted_fields.push(field);
        decision.applied = true;
    }
    decision
}

fn apply_configured_total_cap(
    object: &mut Map<String, Value>,
    thinking: &ThinkingConfig,
    mut decision: AnswerBudgetDecision,
) -> AnswerBudgetDecision {
    let Some(max_tokens) = thinking.max_tokens else {
        return decision;
    };
    if thinking.preserve_answer_budget {
        return decision;
    }
    let max_tokens = u64::from(max_tokens);
    for field in ANSWER_BUDGET_FIELDS {
        match object.get_mut(*field) {
            Some(value) if value.as_u64() == Some(max_tokens) => {
                decision.preserved_fields.push(field);
            }
            Some(value) if value.as_u64().is_some() => {
                *value = Value::Number(Number::from(max_tokens));
                decision.adjusted_fields.push(field);
                decision.applied = true;
            }
            Some(_value) => {
                decision.malformed_fields.push(field);
            }
            None => {}
        }
    }
    if decision.adjusted_fields.is_empty()
        && decision.preserved_fields.is_empty()
        && decision.malformed_fields.is_empty()
    {
        object.insert(
            String::from("max_tokens"),
            Value::Number(Number::from(max_tokens)),
        );
        decision.adjusted_fields.push("max_tokens");
        decision.applied = true;
    }
    decision
}

fn previous_budget_state(observations: &BudgetObservations, configured_budget: u64) -> String {
    if observations.is_empty() {
        return String::from("absent");
    }
    let mut states = observations
        .entries
        .iter()
        .map(|observation| budget_state_label(observation.state, configured_budget))
        .collect::<Vec<_>>();
    states.sort_unstable();
    states.dedup();
    if states.len() == 1 {
        states[0].to_owned()
    } else {
        String::from("mixed")
    }
}

fn previous_budget_tokens(observations: &BudgetObservations) -> String {
    if observations.is_empty() {
        return String::from("absent");
    }
    if observations.entries.len() > 1 {
        return String::from("multiple");
    }
    match observations.entries[0].state {
        BudgetState::Malformed => String::from("malformed"),
        BudgetState::Numeric(value) => value.to_string(),
    }
}

fn final_budget_tokens(observations: &BudgetObservations, configured_budget: u64) -> String {
    budget_tokens_summary(observations, |value| {
        if value > 0 && value < configured_budget {
            configured_budget
        } else {
            value
        }
    })
}

fn current_budget_tokens(observations: &BudgetObservations) -> String {
    budget_tokens_summary(observations, |value| value)
}

fn budget_tokens_summary(
    observations: &BudgetObservations,
    map_numeric: impl Fn(u64) -> u64,
) -> String {
    if observations.is_empty() {
        return String::from("absent");
    }
    let mut values = observations
        .entries
        .iter()
        .filter_map(|observation| match observation.state {
            BudgetState::Malformed => None,
            BudgetState::Numeric(value) => Some(map_numeric(value)),
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    match values.len() {
        0 => String::from("unknown"),
        1 => values[0].to_string(),
        _ => String::from("multiple"),
    }
}

fn budget_state_label(state: BudgetState, configured_budget: u64) -> &'static str {
    match state {
        BudgetState::Malformed => "malformed",
        BudgetState::Numeric(0) => "zero",
        BudgetState::Numeric(existing) if existing < configured_budget => "smaller",
        BudgetState::Numeric(existing) if existing == configured_budget => "equal",
        BudgetState::Numeric(_existing) => "larger",
    }
}

fn join_fields(fields: &[&str]) -> String {
    if fields.is_empty() {
        String::from("none")
    } else {
        fields.join(",")
    }
}

fn join_paths(paths: &[JsonPath]) -> String {
    if paths.is_empty() {
        String::from("none")
    } else {
        paths
            .iter()
            .map(|path| path.display_path())
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn observed_budget_paths(observations: &BudgetObservations, configured_budget: u64) -> String {
    if observations.is_empty() {
        return String::from("none");
    }
    observations
        .entries
        .iter()
        .map(|observation| {
            format!(
                "{}={}",
                observation.path.display_path(),
                budget_state_label(observation.state, configured_budget)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

impl JsonPath {
    fn display_path(self) -> String {
        self.path.join(".")
    }
}

impl BudgetObservations {
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn single_path(&self) -> Option<JsonPath> {
        if self.entries.len() == 1 {
            Some(self.entries[0].path)
        } else {
            None
        }
    }

    fn zero_paths(&self) -> Vec<JsonPath> {
        self.entries
            .iter()
            .filter_map(|observation| match observation.state {
                BudgetState::Numeric(0) => Some(observation.path),
                BudgetState::Numeric(_) | BudgetState::Malformed => None,
            })
            .collect()
    }

    fn malformed_paths(&self) -> Vec<JsonPath> {
        self.entries
            .iter()
            .filter_map(|observation| match observation.state {
                BudgetState::Malformed => Some(observation.path),
                BudgetState::Numeric(_) => None,
            })
            .collect()
    }

    fn non_malformed_paths(&self) -> Vec<JsonPath> {
        self.entries
            .iter()
            .filter_map(|observation| match observation.state {
                BudgetState::Numeric(_) => Some(observation.path),
                BudgetState::Malformed => None,
            })
            .collect()
    }
}

/// Accepted, aggregated OpenAI-compatible chat completion response.
pub(super) struct AggregatedChatCompletion {
    pub(super) body: Bytes,
    pub(super) sse_body: Bytes,
    pub(super) response_metadata: BTreeMap<String, String>,
    pub(super) raw_payloads: RawPayloads,
}

/// Consumes upstream OpenAI-compatible chat completion SSE and aggregates it into JSON.
pub(super) async fn aggregate_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
    attempt_started_at_unix_ms: u64,
    request_id: &str,
    request_model_id: Option<&str>,
    loop_context: LoopInspectionContext,
    upstream_stream_timeouts: Option<UpstreamStreamTimeouts>,
) -> Result<AggregatedChatCompletion, AggregationError> {
    let mut stream = Box::pin(stream);
    let mut buffer = BytesMut::new();
    let mut bytes_seen = 0_usize;
    let mut state = ChatAggregation::new(attempt_started_at_unix_ms, &loop_context);
    let mut saw_first_chunk = false;

    while let Some(chunk) = next_stream_chunk(
        &mut stream,
        stream_chunk_timeout(upstream_stream_timeouts, saw_first_chunk),
    )
    .await?
    {
        let chunk = chunk.map_err(|error| {
            AggregationError::plain(format!(
                "upstream SSE stream failed: {}",
                sanitized_reqwest_error(&error)
            ))
        })?;
        saw_first_chunk = true;
        if state.first_byte_latency_ms.is_none() {
            state.first_byte_latency_ms =
                Some(unix_time_millis().saturating_sub(attempt_started_at_unix_ms));
        }
        bytes_seen = bytes_seen
            .checked_add(chunk.len())
            .ok_or_else(|| AggregationError::plain("upstream SSE body is too large"))?;
        if bytes_seen > MAX_PROXY_BODY_BYTES {
            return Err(AggregationError::plain(format!(
                "upstream SSE body exceeded proxy limit: max_bytes={MAX_PROXY_BODY_BYTES}"
            )));
        }
        buffer.extend_from_slice(&chunk);

        while let Some(event) = next_sse_event(&mut buffer) {
            state.apply_event(&event)?;
        }
    }

    if buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
        return Err(AggregationError::plain(
            "upstream SSE ended with an unterminated event",
        ));
    }

    state.finish(request_id, request_model_id)
}

const fn stream_chunk_timeout(
    timeouts: Option<UpstreamStreamTimeouts>,
    saw_first_chunk: bool,
) -> Option<Duration> {
    match timeouts {
        Some(timeouts) if saw_first_chunk => Some(timeouts.inter_chunk),
        Some(timeouts) => Some(timeouts.first_chunk),
        None => None,
    }
}

async fn next_stream_chunk(
    stream: &mut Pin<Box<impl Stream<Item = Result<Bytes, reqwest::Error>>>>,
    upstream_idle_timeout: Option<Duration>,
) -> Result<Option<Result<Bytes, reqwest::Error>>, AggregationError> {
    let Some(upstream_idle_timeout) = upstream_idle_timeout else {
        return Ok(stream.next().await);
    };
    match timeout(upstream_idle_timeout, stream.next()).await {
        Ok(next_chunk) => Ok(next_chunk),
        Err(_elapsed) => Err(AggregationError::upstream_stall(
            upstream_idle_timeout
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        )),
    }
}

fn next_sse_event(buffer: &mut BytesMut) -> Option<Vec<u8>> {
    let lf_lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, position + 2));
    let crlf_crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, position + 4));
    let (event_end, drain_to) = match (lf_lf, crlf_crlf) {
        (Some(lf), Some(crlf)) => {
            if lf.0 < crlf.0 {
                lf
            } else {
                crlf
            }
        }
        (Some(lf), None) => lf,
        (None, Some(crlf)) => crlf,
        (None, None) => return None,
    };

    let frame = buffer.split_to(drain_to);
    Some(frame[..event_end].to_vec())
}

#[derive(Default)]
struct ChatAggregation {
    attempt_started_at_unix_ms: u64,
    loop_detector: Option<LoopDetector>,
    id: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    service_tier: Option<Value>,
    system_fingerprint: Option<String>,
    usage: Option<Value>,
    extension_fields: Map<String, Value>,
    choices: BTreeMap<u64, ChoiceBuilder>,
    saw_done: bool,
    saw_finish_reason: bool,
    first_byte_latency_ms: Option<u64>,
    first_token_latency_ms: Option<u64>,
    stats: DeltaStats,
    raw_stream_chunks: Vec<RawPayloadChunk>,
}

impl ChatAggregation {
    fn new(attempt_started_at_unix_ms: u64, loop_context: &LoopInspectionContext) -> Self {
        Self {
            attempt_started_at_unix_ms,
            loop_detector: loop_context.detector(),
            ..Self::default()
        }
    }

    fn apply_event(&mut self, event: &[u8]) -> Result<(), AggregationError> {
        let event = std::str::from_utf8(event).map_err(|error| {
            AggregationError::plain(format!("upstream SSE event was not valid UTF-8: {error}"))
        })?;
        let data = sse_data(event);
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }

        let chunk = serde_json::from_str::<Value>(&data).map_err(|error| {
            AggregationError::plain(format!("upstream SSE data was not valid JSON: {error}"))
        })?;
        self.apply_chunk(&chunk)?;
        Ok(())
    }

    fn apply_chunk(&mut self, chunk: &Value) -> Result<(), AggregationError> {
        let Some(chunk_object) = chunk.as_object() else {
            return Ok(());
        };
        copy_extension_fields(
            chunk_object,
            &mut self.extension_fields,
            is_completion_owned_field,
        );

        if let Some(id) = string_field(chunk, "id") {
            self.id.get_or_insert_with(|| id.to_owned());
        }
        if let Some(created) = chunk.get("created").and_then(Value::as_u64) {
            self.created.get_or_insert(created);
        }
        if let Some(model) = string_field(chunk, "model") {
            self.model.get_or_insert_with(|| model.to_owned());
        }
        if let Some(service_tier) = chunk.get("service_tier") {
            self.service_tier
                .get_or_insert_with(|| service_tier.clone());
        }
        if let Some(system_fingerprint) = string_field(chunk, "system_fingerprint") {
            self.system_fingerprint
                .get_or_insert_with(|| system_fingerprint.to_owned());
        }
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(usage.clone());
        }

        let Some(choices) = chunk_object.get("choices").and_then(Value::as_array) else {
            return Ok(());
        };

        for choice in choices {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
            let builder = self.choices.entry(index).or_insert_with(|| ChoiceBuilder {
                index,
                ..ChoiceBuilder::default()
            });
            if let Some(choice_object) = choice.as_object() {
                copy_extension_fields(
                    choice_object,
                    &mut builder.extension_fields,
                    is_choice_owned_field,
                );
            }
            if let Some(finish_reason) =
                choice.get("finish_reason").filter(|value| !value.is_null())
            {
                builder.finish_reason = Some(finish_reason.clone());
                if self.stats.finish_reason.is_none() {
                    self.stats.finish_reason = finish_reason.as_str().map(str::to_owned);
                }
                self.saw_finish_reason = true;
            }
            if let Some(logprobs) = choice.get("logprobs").filter(|value| !value.is_null()) {
                builder.merge_logprobs(logprobs);
            }
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                self.stats.delta_count = self.stats.delta_count.saturating_add(1);
                let apply_result = builder.apply_delta(
                    delta,
                    &mut self.stats,
                    &mut self.first_token_latency_ms,
                    self.attempt_started_at_unix_ms,
                    &mut self.loop_detector,
                    &mut self.raw_stream_chunks,
                );
                if let Err(error) = apply_result {
                    return Err(error.with_raw_payloads(raw_payloads_from_choices(
                        &self.choices,
                        &self.raw_stream_chunks,
                    )));
                }
            }
        }
        Ok(())
    }

    fn finish(
        mut self,
        request_id: &str,
        request_model_id: Option<&str>,
    ) -> Result<AggregatedChatCompletion, AggregationError> {
        self.ensure_usable()?;
        if let Err(error) = self.observe_completed_tool_calls() {
            return Err(error.with_raw_payloads(raw_payloads_from_choices(
                &self.choices,
                &self.raw_stream_chunks,
            )));
        }
        let response_metadata = response_metadata(&self);
        let completion_fields =
            CompletionFields::from_aggregation(&self, request_id, request_model_id);
        let raw_stream_chunks = std::mem::take(&mut self.raw_stream_chunks);
        let finalized_choices = finalize_choices(self.choices);
        let mut raw_payloads = finalized_choices.raw_payloads();
        raw_payloads.chunks = raw_stream_chunks;
        let choices = finalized_choices.choices;
        let sse_body = completion_sse_body(
            &completion_fields,
            finalized_choices.sse_delta_choices,
            &choices,
        )
        .map_err(AggregationError::plain)?;
        let body = completion_body(completion_fields, choices).map_err(AggregationError::plain)?;

        Ok(AggregatedChatCompletion {
            body: Bytes::from(body),
            sse_body: Bytes::from(sse_body),
            response_metadata,
            raw_payloads,
        })
    }

    fn ensure_usable(&self) -> Result<(), AggregationError> {
        if self.choices.is_empty() {
            return Err(AggregationError::plain(
                "upstream SSE ended without chat completion choices",
            ));
        }
        if !self.saw_done && !self.saw_finish_reason {
            return Err(AggregationError::plain(
                "upstream SSE ended before a final chat completion marker",
            ));
        }
        Ok(())
    }

    fn observe_completed_tool_calls(&mut self) -> Result<(), AggregationError> {
        for choice in self.choices.values() {
            choice.observe_completed_tool_calls(&mut self.loop_detector)?;
        }
        Ok(())
    }
}

struct CompletionFields {
    id: String,
    created: u64,
    model: String,
    service_tier: Option<Value>,
    system_fingerprint: Option<String>,
    usage: Option<Value>,
    extension_fields: Map<String, Value>,
}

impl CompletionFields {
    fn from_aggregation(
        aggregation: &ChatAggregation,
        request_id: &str,
        request_model_id: Option<&str>,
    ) -> Self {
        Self {
            id: aggregation
                .id
                .clone()
                .unwrap_or_else(|| format!("chatcmpl-{request_id}")),
            created: aggregation.created.unwrap_or_else(unix_time_secs),
            model: aggregation
                .model
                .clone()
                .or_else(|| request_model_id.map(str::to_owned))
                .unwrap_or_else(|| String::from("unknown")),
            service_tier: aggregation.service_tier.clone(),
            system_fingerprint: aggregation.system_fingerprint.clone(),
            usage: aggregation.usage.clone(),
            extension_fields: aggregation.extension_fields.clone(),
        }
    }
}

struct FinalizedChoices {
    choices: Vec<Value>,
    sse_delta_choices: Vec<Value>,
    raw_output: String,
    raw_reasoning: String,
    raw_tool_calls: Vec<Value>,
}

impl FinalizedChoices {
    fn raw_payloads(&self) -> RawPayloads {
        let raw_tool_calls = (!self.raw_tool_calls.is_empty())
            .then(|| serde_json::to_string(&self.raw_tool_calls).ok())
            .flatten();
        RawPayloads {
            input: None,
            output: (!self.raw_output.is_empty()).then(|| self.raw_output.clone()),
            reasoning: (!self.raw_reasoning.is_empty()).then(|| self.raw_reasoning.clone()),
            tool_calls: raw_tool_calls,
            chunks: Vec::new(),
        }
    }
}

fn finalize_choices(choices: BTreeMap<u64, ChoiceBuilder>) -> FinalizedChoices {
    let mut raw_output = String::new();
    let mut raw_reasoning = String::new();
    let mut raw_tool_calls = Vec::new();
    let mut final_choices = Vec::with_capacity(choices.len());
    let mut sse_delta_choices = Vec::with_capacity(choices.len());

    for choice in choices.into_values() {
        raw_output.push_str(&choice.content);
        raw_reasoning.push_str(&choice.reasoning);
        sse_delta_choices.push(choice.sse_delta_choice());
        let choice = choice.into_value();
        if let Some(tool_calls) = choice
            .get("message")
            .and_then(Value::as_object)
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
        {
            raw_tool_calls.extend(tool_calls.iter().cloned());
        }
        final_choices.push(choice);
    }

    FinalizedChoices {
        choices: final_choices,
        sse_delta_choices,
        raw_output,
        raw_reasoning,
        raw_tool_calls,
    }
}

fn raw_payloads_from_choices(
    choices: &BTreeMap<u64, ChoiceBuilder>,
    raw_stream_chunks: &[RawPayloadChunk],
) -> RawPayloads {
    let mut raw_output = String::new();
    let mut raw_reasoning = String::new();
    let mut raw_tool_calls = Vec::new();

    for choice in choices.values() {
        raw_output.push_str(&choice.content);
        raw_reasoning.push_str(&choice.reasoning);
        if let Some(function_call) = &choice.function_call {
            raw_tool_calls.push(json!({
                "function_call": function_call.as_value(),
            }));
        }
        raw_tool_calls.extend(choice.tool_calls.values().map(ToolCallBuilder::as_value));
    }

    RawPayloads {
        input: None,
        output: (!raw_output.is_empty()).then_some(raw_output),
        reasoning: (!raw_reasoning.is_empty()).then_some(raw_reasoning),
        tool_calls: (!raw_tool_calls.is_empty())
            .then(|| serde_json::to_string(&raw_tool_calls).ok())
            .flatten(),
        chunks: raw_stream_chunks.to_vec(),
    }
}

fn completion_body(fields: CompletionFields, choices: Vec<Value>) -> Result<Vec<u8>, String> {
    let mut response = Map::from_iter([
        (String::from("id"), Value::String(fields.id)),
        (
            String::from("object"),
            Value::String(String::from("chat.completion")),
        ),
        (
            String::from("created"),
            Value::Number(Number::from(fields.created)),
        ),
        (String::from("model"), Value::String(fields.model)),
        (String::from("choices"), Value::Array(choices)),
    ]);
    insert_extension_fields(&mut response, fields.extension_fields);
    if let Some(service_tier) = fields.service_tier {
        response.insert(String::from("service_tier"), service_tier);
    }
    if let Some(system_fingerprint) = fields.system_fingerprint {
        response.insert(
            String::from("system_fingerprint"),
            Value::String(system_fingerprint),
        );
    }
    if let Some(usage) = fields.usage {
        response.insert(String::from("usage"), usage);
    }

    serde_json::to_vec(&Value::Object(response))
        .map_err(|error| format!("failed to serialize aggregated chat completion: {error}"))
}

fn completion_sse_body(
    fields: &CompletionFields,
    delta_choices: Vec<Value>,
    choices: &[Value],
) -> Result<Vec<u8>, String> {
    let mut body = BytesMut::new();
    append_completion_sse_chunk(&mut body, fields, delta_choices, false)?;
    append_completion_sse_chunk(&mut body, fields, finish_sse_choices(choices), true)?;
    body.extend_from_slice(b"data: [DONE]\n\n");
    Ok(body.to_vec())
}

fn append_completion_sse_chunk(
    body: &mut BytesMut,
    fields: &CompletionFields,
    choices: Vec<Value>,
    include_usage: bool,
) -> Result<(), String> {
    let mut chunk = Map::from_iter([
        (String::from("id"), Value::String(fields.id.clone())),
        (
            String::from("object"),
            Value::String(String::from("chat.completion.chunk")),
        ),
        (
            String::from("created"),
            Value::Number(Number::from(fields.created)),
        ),
        (String::from("model"), Value::String(fields.model.clone())),
        (String::from("choices"), Value::Array(choices)),
    ]);
    insert_extension_fields(&mut chunk, fields.extension_fields.clone());
    if let Some(service_tier) = &fields.service_tier {
        chunk.insert(String::from("service_tier"), service_tier.clone());
    }
    if let Some(system_fingerprint) = &fields.system_fingerprint {
        chunk.insert(
            String::from("system_fingerprint"),
            Value::String(system_fingerprint.clone()),
        );
    }
    if include_usage {
        if let Some(usage) = &fields.usage {
            chunk.insert(String::from("usage"), usage.clone());
        }
    }

    let serialized = serde_json::to_vec(&Value::Object(chunk))
        .map_err(|error| format!("failed to serialize aggregated chat completion SSE: {error}"))?;
    body.extend_from_slice(b"data: ");
    body.extend_from_slice(&serialized);
    body.extend_from_slice(b"\n\n");
    Ok(())
}

fn finish_sse_choices(choices: &[Value]) -> Vec<Value> {
    choices
        .iter()
        .map(|choice| {
            let index = choice
                .get("index")
                .cloned()
                .unwrap_or_else(|| Value::Number(Number::from(0)));
            let finish_reason = choice.get("finish_reason").cloned().unwrap_or(Value::Null);
            let stream_choice = Map::from_iter([
                (String::from("index"), index),
                (String::from("delta"), Value::Object(Map::new())),
                (String::from("finish_reason"), finish_reason),
            ]);
            Value::Object(stream_choice)
        })
        .collect()
}

fn copy_extension_fields(
    source: &Map<String, Value>,
    target: &mut Map<String, Value>,
    is_owned_field: fn(&str) -> bool,
) {
    for (key, value) in source {
        if is_owned_field(key) {
            continue;
        }
        target.insert(key.clone(), value.clone());
    }
}

fn insert_extension_fields(target: &mut Map<String, Value>, extension_fields: Map<String, Value>) {
    for (key, value) in extension_fields {
        target.insert(key, value);
    }
}

fn is_completion_owned_field(key: &str) -> bool {
    matches!(
        key,
        "id" | "object"
            | "created"
            | "model"
            | "choices"
            | "service_tier"
            | "system_fingerprint"
            | "usage"
    )
}

fn is_choice_owned_field(key: &str) -> bool {
    matches!(
        key,
        "index" | "delta" | "message" | "finish_reason" | "logprobs"
    )
}

fn is_message_owned_field(key: &str) -> bool {
    matches!(
        key,
        "role"
            | "content"
            | "reasoning_content"
            | "reasoning"
            | "thinking"
            | "tool_calls"
            | "function_call"
            | "refusal"
    )
}

fn response_metadata(aggregation: &ChatAggregation) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::from([
        (String::from("shielded_streaming"), String::from("true")),
        (String::from("upstream_stream_forced"), String::from("true")),
        (
            String::from("first_byte_latency_ms"),
            latency_metadata(aggregation.first_byte_latency_ms),
        ),
        (
            String::from("first_token_latency_ms"),
            latency_metadata(aggregation.first_token_latency_ms),
        ),
        (
            String::from("finish_reason"),
            aggregation
                .stats
                .finish_reason
                .clone()
                .unwrap_or_else(|| String::from("unknown")),
        ),
        (
            String::from("delta_count"),
            aggregation.stats.delta_count.to_string(),
        ),
        (
            String::from("content_delta_count"),
            aggregation.stats.content_delta_count.to_string(),
        ),
        (
            String::from("reasoning_delta_count"),
            aggregation.stats.reasoning_delta_count.to_string(),
        ),
        (
            String::from("tool_call_delta_count"),
            aggregation.stats.tool_call_delta_count.to_string(),
        ),
    ]);
    metadata.insert(
        String::from("response_header_content-type"),
        String::from("application/json"),
    );
    if let Some(loop_detector) = &aggregation.loop_detector {
        metadata.extend(loop_detector.summary().metadata(loop_detector.mode()));
    }
    metadata
}

fn latency_metadata(latency_ms: Option<u64>) -> String {
    latency_ms.map_or_else(|| String::from("unknown"), |latency| latency.to_string())
}

fn sse_data(event: &str) -> String {
    event
        .lines()
        .filter_map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            line.strip_prefix("data:")
                .map(|value| value.strip_prefix(' ').unwrap_or(value))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Default)]
struct ChoiceBuilder {
    index: u64,
    role: Option<String>,
    content: String,
    reasoning: String,
    finish_reason: Option<Value>,
    logprobs: Option<Value>,
    extension_fields: Map<String, Value>,
    message_extension_fields: Map<String, Value>,
    function_call: Option<FunctionCallBuilder>,
    refusal: String,
    saw_refusal: bool,
    tool_calls: BTreeMap<u64, ToolCallBuilder>,
}

impl ChoiceBuilder {
    fn merge_logprobs(&mut self, next: &Value) {
        let Some(existing) = self.logprobs.as_mut() else {
            self.logprobs = Some(next.clone());
            return;
        };
        merge_json_value(existing, next);
    }

    fn sse_delta_choice(&self) -> Value {
        let mut delta = Map::new();
        let has_reasoning = !self.reasoning.is_empty();
        let content = normalize_final_content_after_reasoning(self.content.clone(), has_reasoning);

        delta.insert(
            String::from("role"),
            Value::String(
                self.role
                    .clone()
                    .unwrap_or_else(|| String::from("assistant")),
            ),
        );
        if !(content.is_empty() && (!self.tool_calls.is_empty() || self.function_call.is_some())) {
            delta.insert(String::from("content"), Value::String(content));
        }
        if has_reasoning {
            delta.insert(
                String::from("reasoning_content"),
                Value::String(self.reasoning.clone()),
            );
        }
        if let Some(function_call) = &self.function_call {
            delta.insert(String::from("function_call"), function_call.as_value());
        }
        if self.saw_refusal && !self.refusal.is_empty() {
            delta.insert(String::from("refusal"), Value::String(self.refusal.clone()));
        }
        insert_extension_fields(&mut delta, self.message_extension_fields.clone());
        if !self.tool_calls.is_empty() {
            delta.insert(
                String::from("tool_calls"),
                Value::Array(
                    self.tool_calls
                        .values()
                        .map(ToolCallBuilder::as_sse_delta_value)
                        .collect(),
                ),
            );
        }

        let mut stream_choice = Map::from_iter([
            (
                String::from("index"),
                Value::Number(Number::from(self.index)),
            ),
            (String::from("delta"), Value::Object(delta)),
            (String::from("finish_reason"), Value::Null),
        ]);
        if let Some(logprobs) = &self.logprobs {
            stream_choice.insert(String::from("logprobs"), logprobs.clone());
        }
        insert_extension_fields(&mut stream_choice, self.extension_fields.clone());
        Value::Object(stream_choice)
    }

    fn apply_delta(
        &mut self,
        delta: &Map<String, Value>,
        stats: &mut DeltaStats,
        first_token_latency_ms: &mut Option<u64>,
        attempt_started_at_unix_ms: u64,
        loop_detector: &mut Option<LoopDetector>,
        raw_stream_chunks: &mut Vec<RawPayloadChunk>,
    ) -> Result<(), AggregationError> {
        copy_extension_fields(
            delta,
            &mut self.message_extension_fields,
            is_message_owned_field,
        );
        if let Some(role) = delta.get("role").and_then(Value::as_str) {
            self.role.get_or_insert_with(|| role.to_owned());
        }
        if let Some(content) = delta.get("content").and_then(Value::as_str) {
            if !content.is_empty() {
                self.content.push_str(content);
                stats.content_delta_count = stats.content_delta_count.saturating_add(1);
                mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                observe_fragment(loop_detector, StreamChannel::Content, content)?;
                push_raw_stream_chunk(raw_stream_chunks, StreamChannel::Content, content);
            }
        }
        for field in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(reasoning) = delta.get(field).and_then(Value::as_str) {
                if !reasoning.is_empty() {
                    observe_fragment(loop_detector, StreamChannel::Reasoning, reasoning)?;
                    self.reasoning.push_str(reasoning);
                    stats.reasoning_delta_count = stats.reasoning_delta_count.saturating_add(1);
                    mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                    push_raw_stream_chunk(raw_stream_chunks, StreamChannel::Reasoning, reasoning);
                }
            }
        }
        if let Some(function_call) = delta.get("function_call").and_then(Value::as_object) {
            self.function_call
                .get_or_insert_with(FunctionCallBuilder::default)
                .apply_delta(function_call, loop_detector, raw_stream_chunks)?;
            mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
        }
        if let Some(refusal) = delta.get("refusal") {
            self.saw_refusal = true;
            if let Some(refusal) = refusal.as_str() {
                if !refusal.is_empty() {
                    self.refusal.push_str(refusal);
                    mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                }
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                if let Some(tool_call) = tool_call.as_object() {
                    let index = tool_call
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or(self.tool_calls.len() as u64);
                    self.tool_calls
                        .entry(index)
                        .or_insert_with(|| ToolCallBuilder {
                            index,
                            ..ToolCallBuilder::default()
                        })
                        .apply_delta(tool_call, loop_detector, raw_stream_chunks)?;
                    stats.tool_call_delta_count = stats.tool_call_delta_count.saturating_add(1);
                    mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                }
            }
        }
        Ok(())
    }

    fn into_value(self) -> Value {
        let mut message = Map::new();
        let has_reasoning = !self.reasoning.is_empty();
        let content = normalize_final_content_after_reasoning(self.content, has_reasoning);
        message.insert(
            String::from("role"),
            Value::String(self.role.unwrap_or_else(|| String::from("assistant"))),
        );
        if content.is_empty() && (!self.tool_calls.is_empty() || self.function_call.is_some()) {
            message.insert(String::from("content"), Value::Null);
        } else {
            message.insert(String::from("content"), Value::String(content));
        }
        if has_reasoning {
            message.insert(
                String::from("reasoning_content"),
                Value::String(self.reasoning),
            );
        }
        if let Some(function_call) = self.function_call {
            message.insert(String::from("function_call"), function_call.into_value());
        }
        if self.saw_refusal {
            let refusal = if self.refusal.is_empty() {
                Value::Null
            } else {
                Value::String(self.refusal)
            };
            message.insert(String::from("refusal"), refusal);
        }
        insert_extension_fields(&mut message, self.message_extension_fields);
        if !self.tool_calls.is_empty() {
            message.insert(
                String::from("tool_calls"),
                Value::Array(
                    self.tool_calls
                        .into_values()
                        .map(ToolCallBuilder::into_value)
                        .collect(),
                ),
            );
        }

        let mut choice = Map::from_iter([
            (
                String::from("index"),
                Value::Number(Number::from(self.index)),
            ),
            (String::from("message"), Value::Object(message)),
            (
                String::from("finish_reason"),
                self.finish_reason.unwrap_or(Value::Null),
            ),
        ]);
        if let Some(logprobs) = self.logprobs {
            choice.insert(String::from("logprobs"), logprobs);
        }
        insert_extension_fields(&mut choice, self.extension_fields);
        Value::Object(choice)
    }

    fn observe_completed_tool_calls(
        &self,
        loop_detector: &mut Option<LoopDetector>,
    ) -> Result<(), AggregationError> {
        if let Some(function_call) = &self.function_call {
            function_call.observe_completed(loop_detector)?;
        }
        for tool_call in self.tool_calls.values() {
            tool_call.observe_completed(loop_detector)?;
        }
        Ok(())
    }
}

fn normalize_final_content_after_reasoning(content: String, has_reasoning: bool) -> String {
    if !has_reasoning {
        return content;
    }

    let mut trim_end = 0;
    let mut saw_line_break = false;
    for (index, character) in content.char_indices() {
        if matches!(character, '\n' | '\r' | ' ' | '\t') {
            if matches!(character, '\n' | '\r') {
                saw_line_break = true;
            }
            trim_end = index + character.len_utf8();
            continue;
        }
        break;
    }

    if saw_line_break {
        content[trim_end..].to_owned()
    } else {
        content
    }
}

#[derive(Default)]
struct FunctionCallBuilder {
    name: Option<String>,
    arguments: String,
    saw_arguments: bool,
}

impl FunctionCallBuilder {
    fn apply_delta(
        &mut self,
        function_call: &Map<String, Value>,
        loop_detector: &mut Option<LoopDetector>,
        raw_stream_chunks: &mut Vec<RawPayloadChunk>,
    ) -> Result<(), AggregationError> {
        if let Some(name) = function_call.get("name").and_then(Value::as_str) {
            self.name.get_or_insert_with(|| name.to_owned());
        }
        if let Some(arguments) = function_call.get("arguments").and_then(Value::as_str) {
            self.saw_arguments = true;
            self.arguments.push_str(arguments);
            observe_fragment(loop_detector, StreamChannel::ToolArguments, arguments)?;
            push_raw_stream_chunk(raw_stream_chunks, StreamChannel::ToolArguments, arguments);
        }
        Ok(())
    }

    fn observe_completed(
        &self,
        loop_detector: &mut Option<LoopDetector>,
    ) -> Result<(), AggregationError> {
        if self.saw_arguments {
            observe_completed_tool_call(
                loop_detector,
                self.name.as_deref().unwrap_or(""),
                &self.arguments,
            )?;
        }
        Ok(())
    }

    fn into_value(self) -> Value {
        let mut function_call = Map::new();
        if let Some(name) = self.name {
            function_call.insert(String::from("name"), Value::String(name));
        }
        if self.saw_arguments {
            function_call.insert(String::from("arguments"), Value::String(self.arguments));
        }
        Value::Object(function_call)
    }

    fn as_value(&self) -> Value {
        let mut function_call = Map::new();
        if let Some(name) = &self.name {
            function_call.insert(String::from("name"), Value::String(name.clone()));
        }
        if self.saw_arguments {
            function_call.insert(
                String::from("arguments"),
                Value::String(self.arguments.clone()),
            );
        }
        Value::Object(function_call)
    }
}

fn merge_json_value(existing: &mut Value, next: &Value) {
    match (existing, next) {
        (Value::Object(existing), Value::Object(next)) => {
            for (key, next_value) in next {
                match existing.get_mut(key) {
                    Some(existing_value) => merge_json_value(existing_value, next_value),
                    None => {
                        existing.insert(key.clone(), next_value.clone());
                    }
                }
            }
        }
        (Value::Array(existing), Value::Array(next)) => {
            existing.extend(next.iter().cloned());
        }
        (existing, next) => {
            *existing = next.clone();
        }
    }
}

#[derive(Default)]
struct ToolCallBuilder {
    index: u64,
    id: Option<String>,
    type_name: Option<String>,
    function_name: Option<String>,
    function_arguments: String,
}

impl ToolCallBuilder {
    fn apply_delta(
        &mut self,
        tool_call: &Map<String, Value>,
        loop_detector: &mut Option<LoopDetector>,
        raw_stream_chunks: &mut Vec<RawPayloadChunk>,
    ) -> Result<(), AggregationError> {
        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            self.id.get_or_insert_with(|| id.to_owned());
        }
        if let Some(type_name) = tool_call.get("type").and_then(Value::as_str) {
            self.type_name.get_or_insert_with(|| type_name.to_owned());
        }
        if let Some(function) = tool_call.get("function").and_then(Value::as_object) {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                self.function_name.get_or_insert_with(|| name.to_owned());
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                self.function_arguments.push_str(arguments);
                observe_fragment(loop_detector, StreamChannel::ToolArguments, arguments)?;
                push_raw_stream_chunk(raw_stream_chunks, StreamChannel::ToolArguments, arguments);
            }
        }
        Ok(())
    }

    fn into_value(self) -> Value {
        json!({
            "id": self.id.unwrap_or_else(|| format!("call_{}", self.index)),
            "type": self.type_name.unwrap_or_else(|| String::from("function")),
            "function": {
                "name": self.function_name.unwrap_or_default(),
                "arguments": self.function_arguments,
            },
        })
    }

    fn as_value(&self) -> Value {
        json!({
            "id": self.id.clone().unwrap_or_else(|| format!("call_{}", self.index)),
            "type": self.type_name.clone().unwrap_or_else(|| String::from("function")),
            "function": {
                "name": self.function_name.clone().unwrap_or_default(),
                "arguments": self.function_arguments.clone(),
            },
        })
    }

    fn as_sse_delta_value(&self) -> Value {
        let mut tool_call = self.as_value();
        if let Some(tool_call) = tool_call.as_object_mut() {
            tool_call.insert(
                String::from("index"),
                Value::Number(Number::from(self.index)),
            );
        }
        tool_call
    }

    fn observe_completed(
        &self,
        loop_detector: &mut Option<LoopDetector>,
    ) -> Result<(), AggregationError> {
        if self.function_arguments.is_empty() {
            return Ok(());
        }
        observe_completed_tool_call(
            loop_detector,
            self.function_name.as_deref().unwrap_or(""),
            &self.function_arguments,
        )
    }
}

#[derive(Default)]
struct DeltaStats {
    delta_count: u64,
    content_delta_count: u64,
    reasoning_delta_count: u64,
    tool_call_delta_count: u64,
    finish_reason: Option<String>,
}

fn mark_first_token(first_token_latency_ms: &mut Option<u64>, attempt_started_at_unix_ms: u64) {
    if first_token_latency_ms.is_none() {
        *first_token_latency_ms =
            Some(unix_time_millis().saturating_sub(attempt_started_at_unix_ms));
    }
}

fn push_raw_stream_chunk(chunks: &mut Vec<RawPayloadChunk>, channel: StreamChannel, text: &str) {
    if text.is_empty() || chunks.len() >= RAW_STREAM_CHUNK_LIMIT {
        return;
    }
    chunks.push(RawPayloadChunk::new(channel.as_str(), text));
}

fn string_field<'value>(value: &'value Value, key: &str) -> Option<&'value str> {
    value.get(key).and_then(Value::as_str)
}

fn unix_time_secs() -> u64 {
    unix_time_millis() / 1_000
}
