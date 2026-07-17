//! Adapt `DeepInfra`'s native Qwen3 reranker contract to vLLM's score API.

use axum::{
    body::Bytes,
    http::{Method, Uri},
};
use serde_json::{Value, json};
use thiserror::Error;

/// `DeepInfra` path model identity used by policy and upstream-profile routing.
pub(crate) const MODEL_ID: &str = "Qwen/Qwen3-Reranker-8B";
const UPSTREAM_MODEL_ID: &str = "qwen3-reranker-8b";
/// Exact `DeepInfra` native inference path supported by this adapter.
pub(crate) const INFERENCE_PATH: &str = "/v1/inference/Qwen/Qwen3-Reranker-8B";
/// `DeepInfra` and Qwen's documented default retrieval instruction.
pub(crate) const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";
const MAX_PAIR_COUNT: usize = 1_024;
const MAX_INSTRUCTION_CHARS: usize = 2_048;
// The converted scalar head is evaluated in f32. Eight ULPs at |s|=1 tolerate
// representation noise without accepting a materially invalid scalar score.
const SCORE_DOMAIN_TOLERANCE: f64 = f32::EPSILON as f64 * 8.0;

/// Validated `DeepInfra` scheduling hint; local inference has one scheduling tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServiceTier {
    /// Normal cloud scheduling.
    Default,
    /// Cloud priority scheduling hint.
    Priority,
    /// Cloud spare-capacity scheduling hint.
    Flex,
}

impl ServiceTier {
    /// Return the canonical `DeepInfra` wire value for observability.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Priority => "priority",
            Self::Flex => "flex",
        }
    }
}

/// Request-derived invariants required to validate the upstream score response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResponseExpectations {
    /// Exact number of pairwise scores required from vLLM.
    pub(crate) result_count: usize,
}

/// Fully validated request adaptation ready for generic proxy forwarding.
#[derive(Debug)]
pub(crate) struct AdaptedRequest {
    /// vLLM score endpoint URI, preserving the downstream query string.
    pub(crate) forward_uri: Uri,
    /// Serialized N:N vLLM score request.
    pub(crate) body: Bytes,
    /// Invariants for fail-closed response conversion.
    pub(crate) response_expectations: ResponseExpectations,
    /// Validated scheduling hint retained in observability metadata.
    pub(crate) service_tier: ServiceTier,
}

/// Bounded client-facing failures produced before any upstream request.
#[derive(Debug, Error)]
pub(crate) enum RequestError {
    /// The request does not satisfy the documented `DeepInfra` schema.
    #[error("{0}")]
    Invalid(String),
    /// The request asks for semantics the synchronous vLLM adapter cannot provide.
    #[error("{0}")]
    Unsupported(String),
}

impl RequestError {
    /// Stable OpenAI-style error code emitted by the proxy error boundary.
    pub(crate) const fn code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid_deepinfra_rerank_request",
            Self::Unsupported(_) => "unsupported_deepinfra_rerank_feature",
        }
    }
}

#[must_use]
/// Return whether the request targets this exact `DeepInfra` model contract.
pub(crate) fn is_request(method: &Method, uri: &Uri) -> bool {
    *method == Method::POST && uri.path() == INFERENCE_PATH
}

#[must_use]
/// Extract the path model identity without requiring a body `model` field.
pub(crate) fn model_id_from_path(method: &Method, uri: &Uri) -> Option<&'static str> {
    is_request(method, uri).then_some(MODEL_ID)
}

/// Validate and convert one synchronous `DeepInfra` request to one vLLM N:N score batch.
///
/// vLLM's score protocol has no instruction field. This adapter requires the target
/// template to implement `DeepInfra`'s documented default; the deployment canary must
/// prove that precondition. Any custom instruction fails closed until the live server
/// exposes an exact per-request template input.
pub(crate) fn adapt_request(uri: &Uri, body: &Bytes) -> Result<AdaptedRequest, RequestError> {
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        RequestError::Invalid(format!("invalid DeepInfra rerank JSON: {error}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        RequestError::Invalid(String::from("DeepInfra rerank body must be a JSON object"))
    })?;
    reject_unknown_fields(object).map_err(RequestError::Invalid)?;
    let queries = required_string_array(object, "queries").map_err(RequestError::Invalid)?;
    let documents = required_string_array(object, "documents").map_err(RequestError::Invalid)?;
    if queries.len() != documents.len() {
        return Err(RequestError::Invalid(String::from(
            "DeepInfra rerank queries and documents must have the same length",
        )));
    }
    validate_instruction(object.get("instruction"))?;
    validate_webhook(object.get("webhook"))?;
    let service_tier =
        parse_service_tier(object.get("service_tier")).map_err(RequestError::Invalid)?;
    let forward_uri = score_uri(uri).map_err(RequestError::Invalid)?;
    let result_count = queries.len();
    let body = serde_json::to_vec(&json!({
        "model": UPSTREAM_MODEL_ID,
        "text_1": queries,
        "text_2": documents,
    }))
    .map(Bytes::from)
    .map_err(|error| {
        RequestError::Invalid(format!("serialize vLLM score request failed: {error}"))
    })?;

    Ok(AdaptedRequest {
        forward_uri,
        body,
        response_expectations: ResponseExpectations { result_count },
        service_tier,
    })
}

fn reject_unknown_fields(object: &serde_json::Map<String, Value>) -> Result<(), String> {
    const ALLOWED_FIELDS: [&str; 5] = [
        "queries",
        "documents",
        "instruction",
        "service_tier",
        "webhook",
    ];
    if let Some(field) = object
        .keys()
        .find(|field| !ALLOWED_FIELDS.contains(&field.as_str()))
    {
        return Err(format!(
            "DeepInfra rerank body contains unsupported field {field:?}"
        ));
    }
    Ok(())
}

fn required_string_array(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, String> {
    let items = object
        .get(field)
        .ok_or_else(|| format!("DeepInfra rerank body missing {field}"))?
        .as_array()
        .ok_or_else(|| format!("DeepInfra rerank {field} must be an array of strings"))?;
    if items.is_empty() {
        return Err(format!(
            "DeepInfra rerank {field} must be a non-empty array"
        ));
    }
    if items.len() > MAX_PAIR_COUNT {
        return Err(format!(
            "DeepInfra rerank {field} must contain at most {MAX_PAIR_COUNT} items"
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("DeepInfra rerank {field}[{index}] must be a string"))
        })
        .collect()
}

fn validate_instruction(instruction: Option<&Value>) -> Result<(), RequestError> {
    let Some(instruction) = instruction else {
        return Ok(());
    };
    let instruction = instruction.as_str().ok_or_else(|| {
        RequestError::Invalid(String::from(
            "DeepInfra rerank instruction must be a string",
        ))
    })?;
    if instruction.chars().count() > MAX_INSTRUCTION_CHARS {
        return Err(RequestError::Invalid(format!(
            "DeepInfra rerank instruction must not exceed {MAX_INSTRUCTION_CHARS} characters"
        )));
    }
    if instruction != DEFAULT_INSTRUCTION {
        return Err(RequestError::Unsupported(String::from(
            "custom instruction is unsupported because vLLM /v1/score has no per-request instruction field",
        )));
    }
    Ok(())
}

fn validate_webhook(webhook: Option<&Value>) -> Result<(), RequestError> {
    match webhook {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(webhook)) if webhook.is_empty() => Err(RequestError::Invalid(
            String::from("DeepInfra rerank webhook must not be empty"),
        )),
        Some(Value::String(_)) => Err(RequestError::Unsupported(String::from(
            "DeepInfra rerank webhook is unsupported by the synchronous local adapter",
        ))),
        Some(_) => Err(RequestError::Invalid(String::from(
            "DeepInfra rerank webhook must be a string or null",
        ))),
    }
}

fn parse_service_tier(service_tier: Option<&Value>) -> Result<ServiceTier, String> {
    match service_tier {
        None => Ok(ServiceTier::Default),
        Some(Value::String(tier)) => match tier.as_str() {
            "default" => Ok(ServiceTier::Default),
            "priority" => Ok(ServiceTier::Priority),
            "flex" => Ok(ServiceTier::Flex),
            _ => Err(String::from(
                "DeepInfra rerank service_tier must be default, priority, or flex",
            )),
        },
        Some(_) => Err(String::from(
            "DeepInfra rerank service_tier must be default, priority, or flex",
        )),
    }
}

fn score_uri(uri: &Uri) -> Result<Uri, String> {
    let rewritten = match uri.query() {
        Some(query) => format!("/v1/score?{query}"),
        None => String::from("/v1/score"),
    };
    rewritten
        .parse()
        .map_err(|error| format!("rewrite DeepInfra rerank URI failed: {error}"))
}

/// Convert a trusted vLLM scalar-head score response to `DeepInfra` probability semantics.
pub(crate) fn score_response_to_deepinfra_response(
    body: &Bytes,
    expected: ResponseExpectations,
) -> Result<Bytes, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| format!("invalid vLLM score JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| String::from("vLLM score response must be a JSON object"))?;
    let data = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| String::from("vLLM score response data must be an array"))?;
    if data.len() != expected.result_count {
        return Err(format!(
            "vLLM score response count {} does not match expected {}",
            data.len(),
            expected.result_count
        ));
    }

    let mut probabilities = vec![None; expected.result_count];
    for (entry_index, item) in data.iter().enumerate() {
        let item = item
            .as_object()
            .ok_or_else(|| format!("vLLM score response data[{entry_index}] must be an object"))?;
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                format!(
                    "vLLM score response data[{entry_index}].index must be a non-negative integer"
                )
            })?;
        if index >= expected.result_count {
            return Err(format!(
                "vLLM score response index {index} is outside expected range 0..{}",
                expected.result_count
            ));
        }
        if probabilities[index].is_some() {
            return Err(format!(
                "vLLM score response contains duplicate index {index}"
            ));
        }
        let score = item
            .get("score")
            .and_then(Value::as_f64)
            .filter(|score| score.is_finite())
            .ok_or_else(|| {
                format!("vLLM score response data[{entry_index}].score must be finite")
            })?;
        probabilities[index] = Some(scalar_score_to_probability(score)?);
    }
    let probabilities = probabilities
        .into_iter()
        .enumerate()
        .map(|(index, score)| {
            score.ok_or_else(|| format!("vLLM score response missing index {index}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_tokens = trusted_prompt_tokens(object)?;
    let request_id = optional_request_id(object)?;
    let mut output = serde_json::Map::new();
    output.insert(String::from("scores"), json!(probabilities));
    output.insert(String::from("input_tokens"), Value::from(input_tokens));
    if let Some(request_id) = request_id {
        output.insert(String::from("request_id"), Value::String(request_id));
    }
    output.insert(
        String::from("inference_status"),
        json!({
            "status": "succeeded",
            "runtime_ms": 0,
            "cost": 0.0,
            "tokens_generated": 0,
            "tokens_input": input_tokens,
            "output_length": 0,
        }),
    );
    serde_json::to_vec(&Value::Object(output))
        .map(Bytes::from)
        .map_err(|error| format!("serialize DeepInfra rerank response failed: {error}"))
}

fn scalar_score_to_probability(score: f64) -> Result<f64, String> {
    if !(-1.0 - SCORE_DOMAIN_TOLERANCE..=1.0 + SCORE_DOMAIN_TOLERANCE).contains(&score) {
        return Err(format!(
            "vLLM scalar score {score} is outside expected [-1, 1] domain"
        ));
    }
    // This midpoint is the overflow-safe equivalent of p_yes = (s + 1) / 2.
    Ok(f64::midpoint(score.clamp(-1.0, 1.0), 1.0))
}

fn trusted_prompt_tokens(object: &serde_json::Map<String, Value>) -> Result<u64, String> {
    let usage = object
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| String::from("vLLM score response usage must be an object"))?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            String::from("vLLM score response usage.prompt_tokens must be a non-negative integer")
        })?;
    for field in ["total_tokens", "completion_tokens"] {
        if let Some(value) = usage.get(field)
            && !value.is_null()
            && value.as_u64().is_none()
        {
            return Err(format!(
                "vLLM score response usage.{field} must be a non-negative integer or null"
            ));
        }
    }
    Ok(prompt_tokens)
}

fn optional_request_id(object: &serde_json::Map<String, Value>) -> Result<Option<String>, String> {
    match object.get("id") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if id.trim().is_empty() => Ok(None),
        Some(Value::String(id)) => Ok(Some(id.clone())),
        Some(_) => Err(String::from(
            "vLLM score response id must be a string or null",
        )),
    }
}

#[cfg(test)]
mod tests;
