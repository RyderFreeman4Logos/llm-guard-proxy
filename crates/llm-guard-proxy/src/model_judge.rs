//! No-thinking model judge HTTP client (issue #110).
//!
//! [`ModelJudgeClient`] is the slow-path LLM judge that runs when the risk
//! combiner (issue #109) reports non-trivial risk. It calls the SAME local
//! model that produced the suspicious output, but with thinking DISABLED, and
//! asks it to evaluate a bounded snapshot ([`JudgeSnapshot`]) and return a
//! structured JSON verdict ([`LoopJudgeResult`]).
//!
//! When the verdict requests it (`clean_state_needed`), the client can also
//! produce a [`CleanReasoningState`] for a bounded retry via [`salvage`](ModelJudgeClient::salvage).
//!
//! All data types live in `llm-guard-proxy-core`; this module only owns the
//! async HTTP transport and JSON envelope handling.

#![allow(dead_code)]

use std::time::Duration;

use llm_guard_proxy_core::model_judge::{
    CleanReasoningState, JudgePromptBuilder, JudgeSnapshot, LoopJudgeResult,
};
use reqwest::Client;
use serde::Deserialize;
use thiserror::Error;

/// Errors produced by the model judge client.
#[derive(Debug, Error)]
pub enum JudgeError {
    /// Network or HTTP-layer failure.
    #[error("judge http request failed: {0}")]
    Http(String),
    /// The model returned content that could not be parsed as the expected JSON.
    #[error("judge json parse failed: {0}")]
    JsonParse(String),
    /// The request exceeded the configured timeout.
    #[error("judge request timed out")]
    Timeout,
    /// The endpoint responded with an unexpected shape or status.
    #[error("invalid judge response: {0}")]
    InvalidResponse(String),
}

/// Async HTTP client that calls the local model as a no-thinking loop judge.
///
/// Construct one per evaluation site and reuse it. The client is cheap to
/// clone (it shares an internal `reqwest::Client`).
pub struct ModelJudgeClient {
    client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl ModelJudgeClient {
    /// Create a new judge client.
    ///
    /// - `endpoint`: full chat-completions URL (e.g. `http://host/v1/chat/completions`).
    /// - `model`: model id to route to (the SAME local model, with thinking off).
    /// - `api_key`: optional bearer token for the endpoint.
    /// - `timeout`: per-request timeout; exceeded timeouts map to [`JudgeError::Timeout`].
    #[must_use]
    pub fn new(
        endpoint: String,
        model: String,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            client,
            endpoint,
            model,
            api_key,
            timeout,
        }
    }

    /// Ask the model judge to evaluate the snapshot and return a verdict.
    ///
    /// Builds the system + user prompts, sends a chat completion request with
    /// thinking DISABLED, and parses the JSON response into a [`LoopJudgeResult`].
    ///
    /// # Errors
    ///
    /// Returns [`JudgeError::Timeout`] if the request exceeds the configured
    /// timeout, [`JudgeError::Http`] for network failures, and
    /// [`JudgeError::JsonParse`] if the model's output cannot be parsed.
    pub async fn judge(&self, snapshot: &JudgeSnapshot) -> Result<LoopJudgeResult, JudgeError> {
        let system = JudgePromptBuilder::system_prompt();
        let user = JudgePromptBuilder::user_prompt(snapshot);
        let raw = self.chat_completion(&system, &user).await?;
        parse_json_content(&raw)
    }

    /// Ask the model judge to salvage a clean reasoning state for retry.
    ///
    /// Should only be called when `result.clean_state_needed` is true. The
    /// judge is given the verdict and the original snapshot and asked to emit a
    /// [`CleanReasoningState`].
    ///
    /// # Errors
    ///
    /// Same error variants as [`judge`](Self::judge).
    pub async fn salvage(
        &self,
        result: &LoopJudgeResult,
        snapshot: &JudgeSnapshot,
    ) -> Result<CleanReasoningState, JudgeError> {
        let system = JudgePromptBuilder::system_prompt();
        let user = build_salvage_prompt(result, snapshot);
        let raw = self.chat_completion(&system, &user).await?;
        parse_json_content(&raw)
    }

    /// Send a chat completion request with thinking disabled and return the
    /// raw text content of the first choice.
    async fn chat_completion(&self, system: &str, user: &str) -> Result<String, JudgeError> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": false,
            // Disable thinking for the judge call. Providers that do not
            // recognise this field will simply ignore it.
            "enable_thinking": false,
            "thinking": false,
            "chat_template_kwargs": {"enable_thinking": false},
        });

        let mut request = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .timeout(self.timeout)
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        if let Some(ref api_key) = self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().await.map_err(|error| {
            if error.is_timeout() {
                JudgeError::Timeout
            } else {
                JudgeError::Http(error.to_string())
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(JudgeError::InvalidResponse(format!(
                "endpoint returned status {status}: {body_text}"
            )));
        }

        let envelope: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|error| JudgeError::InvalidResponse(format!("decode envelope: {error}")))?;

        envelope
            .first_content()
            .ok_or_else(|| JudgeError::InvalidResponse(String::from("response had no content")))
    }
}

/// Build the salvage user prompt embedding both the verdict and the snapshot.
fn build_salvage_prompt(result: &LoopJudgeResult, snapshot: &JudgeSnapshot) -> String {
    let verdict = serde_json::to_string_pretty(result)
        .unwrap_or_else(|error| format!("{{\"verdict_error\": \"{error}\"}}"));
    let evidence = serde_json::to_string_pretty(snapshot)
        .unwrap_or_else(|error| format!("{{\"evidence_error\": \"{error}\"}}"));
    format!(
        "<verdict>\n{verdict}\n</verdict>\n<evidence>\n{evidence}\n</evidence>\n\n\
         Produce a clean reasoning state for a bounded retry. \
         Return only a JSON object matching the clean reasoning state schema."
    )
}

/// Parse the model's raw text output as JSON into `T`.
///
/// Tolerates content that wraps the JSON in prose or markdown fences by
/// extracting the first balanced `{...}` object.
fn parse_json_content<T: for<'de> Deserialize<'de>>(raw: &str) -> Result<T, JudgeError> {
    // Fast path: the whole string is valid JSON.
    if let Ok(value) = serde_json::from_str::<T>(raw) {
        return Ok(value);
    }
    // Slow path: extract the first balanced JSON object.
    let extracted = extract_first_json_object(raw)
        .ok_or_else(|| JudgeError::JsonParse(format!("no JSON object found in response: {raw}")))?;
    serde_json::from_str(&extracted)
        .map_err(|error| JudgeError::JsonParse(format!("{error} (raw: {extracted})")))
}

/// Extract the first balanced `{ ... }` object from a string, handling nested
/// braces. Returns `None` if no balanced object is found.
fn extract_first_json_object(raw: &str) -> Option<String> {
    let start = raw.find('{')?;
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    let bytes = raw.as_bytes();
    for (idx, &byte) in bytes.iter().enumerate().skip(start) {
        let ch = char::from(byte);
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(raw[start..=idx].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// The subset of the `OpenAI` chat completion response we need.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    /// Content of the first choice's message, if present.
    fn first_content(&self) -> Option<String> {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.clone())
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: ChatMessage,
}

#[derive(Debug, Default, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_guard_proxy_core::model_judge::TaskKind;

    #[test]
    fn client_can_be_constructed() {
        let client = ModelJudgeClient::new(
            String::from("http://localhost:8080/v1/chat/completions"),
            String::from("test-model"),
            Some(String::from("secret")),
            Duration::from_secs(5),
        );
        assert_eq!(client.endpoint, "http://localhost:8080/v1/chat/completions");
        assert_eq!(client.model, "test-model");
        assert_eq!(client.api_key.as_deref(), Some("secret"));
        assert_eq!(client.timeout, Duration::from_secs(5));
    }

    #[test]
    fn client_can_be_constructed_without_api_key() {
        let client = ModelJudgeClient::new(
            String::from("http://localhost:8080/v1/chat/completions"),
            String::from("test-model"),
            None,
            Duration::from_secs(10),
        );
        assert!(client.api_key.is_none());
    }

    #[test]
    fn judge_error_display_works() {
        let cases = [
            (
                JudgeError::Http(String::from("conn refused")),
                "judge http request failed: conn refused",
            ),
            (
                JudgeError::JsonParse(String::from("bad token")),
                "judge json parse failed: bad token",
            ),
            (JudgeError::Timeout, "judge request timed out"),
            (
                JudgeError::InvalidResponse(String::from("500")),
                "invalid judge response: 500",
            ),
        ];
        for (error, expected) in cases {
            let rendered = format!("{error}");
            assert_eq!(rendered, expected, "Display mismatch for {error:?}");
        }
    }

    #[test]
    fn extract_first_json_object_handles_plain_json() {
        let raw = r#"{"is_loop": true, "confidence": 0.5}"#;
        let extracted = extract_first_json_object(raw).expect("should find object");
        assert!(extracted.contains("\"is_loop\""));
    }

    #[test]
    fn extract_first_json_object_handles_markdown_fence() {
        let raw = "Here is the verdict:\n```json\n{\"is_loop\": false}\n```\nDone.";
        let extracted = extract_first_json_object(raw).expect("should find object");
        assert_eq!(extracted, "{\"is_loop\": false}");
    }

    #[test]
    fn extract_first_json_object_handles_nested_braces_in_strings() {
        // Braces inside strings must not affect depth counting.
        let raw = r#"prefix {"short_reason": "loop at {span} end"} suffix"#;
        let extracted = extract_first_json_object(raw).expect("should find object");
        assert!(extracted.contains("loop at"));
    }

    #[test]
    fn extract_first_json_object_returns_none_when_no_object() {
        assert!(extract_first_json_object("no json here").is_none());
    }

    #[test]
    fn parse_json_content_fast_path() {
        let raw = r#"{"is_loop": true, "severity": "hard", "confidence": 0.9, "loop_types": [], "context_rot_risk": 0.5, "abort_now": true, "recommended_action": "abort_and_salvage", "keep_span_ids": [], "drop_span_ids": [], "clean_state_needed": true, "short_reason": "x"}"#;
        let result: LoopJudgeResult = parse_json_content(raw).expect("should parse");
        assert!(result.is_loop);
        assert!(result.clean_state_needed);
    }

    #[test]
    fn parse_json_content_slow_path_with_prose() {
        let raw = "The verdict is:\n```json\n{\"is_loop\": false, \"severity\": \"none\", \"confidence\": 0.1, \"loop_types\": [], \"context_rot_risk\": 0.0, \"abort_now\": false, \"recommended_action\": \"continue\", \"keep_span_ids\": [], \"drop_span_ids\": [], \"clean_state_needed\": false, \"short_reason\": \"ok\"}\n```";
        let result: LoopJudgeResult = parse_json_content(raw).expect("should parse");
        assert!(!result.is_loop);
    }

    #[test]
    fn build_salvage_prompt_includes_verdict_and_evidence() {
        let snapshot = JudgeSnapshot {
            request_id_hash: String::from("hash"),
            task_kind_hint: TaskKind::Math,
            elapsed_ms: 1000,
            generated_tokens: 100,
            channels: llm_guard_proxy_core::model_judge::SnapshotChannels::default(),
            current_answer_candidate: None,
            known_prompt_constraints: Vec::new(),
            deterministic_signals: Vec::new(),
        };
        let result = LoopJudgeResult {
            is_loop: true,
            severity: llm_guard_proxy_core::model_judge::JudgeSeverity::Hard,
            confidence: 0.9,
            loop_types: Vec::new(),
            context_rot_risk: 0.5,
            abort_now: true,
            recommended_action:
                llm_guard_proxy_core::model_judge::RecommendedAction::AbortAndSalvage,
            loop_start_span_id: None,
            keep_span_ids: Vec::new(),
            drop_span_ids: Vec::new(),
            clean_state_needed: true,
            short_reason: String::from("loop"),
        };
        let prompt = build_salvage_prompt(&result, &snapshot);
        assert!(
            prompt.contains("<verdict>"),
            "prompt must contain verdict tag"
        );
        assert!(
            prompt.contains("<evidence>"),
            "prompt must contain evidence tag"
        );
        assert!(prompt.contains("hash"), "prompt must contain snapshot hash");
    }
}
