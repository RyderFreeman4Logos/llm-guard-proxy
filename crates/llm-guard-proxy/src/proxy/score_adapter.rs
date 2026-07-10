//! Adapt vLLM-style `POST /v1/score` to upstream `POST /v1/rerank`.
//!
//! Background: healthcheck and some clients use `/v1/score` with `text_1`/`text_2`.
//! Querit-4B (and other transformers rerank servers) only implement `/v1/rerank`.
//! The proxy rewrites request and response so clients keep the score contract.

use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Bytes,
    http::{Method, Uri},
};
use serde::{Serializer as _, ser::SerializeMap};
use serde_json::{Value, json, value::RawValue};

mod request;

/// Bound score parsing amplification before model extraction or shape classification.
pub(crate) const MAX_SCORE_BODY_BYTES: usize = 1024 * 1024;

pub(crate) use request::model_id_from_score_body;
#[cfg(test)]
use request::{contains_non_serde_integer, parse_lax_top_n_string};
use request::{has_valid_lax_top_n, parse_lax_positive_top_n, parse_score_value};

/// Whether this request is a score endpoint that should be rewritten to rerank.
#[must_use]
pub(crate) fn is_score_request(method: &Method, uri: &Uri) -> bool {
    *method == Method::POST && uri.path() == "/v1/score"
}

/// Whether this score body can be rewritten as a single `/v1/rerank` call.
///
/// Adaptable shapes: scalar text-only `text_1` with string/array `text_2`, or an
/// already-rerank `query`+`documents` body. Valid unsupported canonical batch or
/// multimodal shapes and complete future shapes are forwarded unchanged;
/// known-invalid canonical types fail locally.
pub(crate) fn can_adapt_score_body_to_rerank(body: &Bytes) -> Result<bool, String> {
    let parsed = parse_score_value(body)?;
    let object = parsed
        .value
        .as_object()
        .ok_or_else(|| String::from("score body must be a JSON object"))?;
    match (object.get("text_1"), object.get("text_2")) {
        (Some(_), None) => Err(String::from("score body missing text_2")),
        (None, Some(_)) => Err(String::from("score body missing text_1")),
        (Some(text_1), Some(text_2)) => {
            let len_1 = validate_score_input_shape("text_1", text_1)?;
            let len_2 = validate_score_input_shape("text_2", text_2)?;
            validate_score_input_lengths(len_1, len_2)?;
            Ok(matches!(text_1, Value::String(_))
                && (matches!(text_2, Value::String(_))
                    || matches!(text_2, Value::Array(items) if items.iter().all(Value::is_string))))
        }
        (None, None) => classify_noncanonical_score_shape(object),
    }
}

fn classify_noncanonical_score_shape(
    object: &serde_json::Map<String, Value>,
) -> Result<bool, String> {
    match (object.get("query"), object.get("documents")) {
        (Some(query), Some(documents)) => {
            let query_len = validate_rerank_query_shape(query)?;
            let document_len = validate_rerank_documents_shape(documents)?;
            validate_score_input_lengths(query_len, document_len)?;
            if object
                .get("top_n")
                .is_some_and(|top_n| !has_valid_lax_top_n(top_n))
            {
                return Err(String::from("score top_n must be a valid integer"));
            }
            Ok(true)
        }
        _ if has_complete_future_score_shape(object) => Ok(false),
        (Some(_), None) => Err(String::from("score body missing documents")),
        (None, Some(_)) => Err(String::from("score body missing query")),
        (None, None) if has_known_future_score_field(object) => Err(String::from(
            "score body contains an incomplete known future score shape",
        )),
        (None, None) if has_complete_opaque_future_score_shape(object) => Ok(false),
        (None, None) => Err(String::from(
            "score body requires a complete score input pair",
        )),
    }
}

fn has_known_future_score_field(object: &serde_json::Map<String, Value>) -> bool {
    ["queries", "items", "data_1", "data_2"]
        .iter()
        .any(|key| object.contains_key(*key))
}

fn has_complete_opaque_future_score_shape(object: &serde_json::Map<String, Value>) -> bool {
    const NON_INPUT_FIELDS: [&str; 9] = [
        "model",
        "top_n",
        "priority",
        "truncate_prompt_tokens",
        "mm_processor_kwargs",
        "additional_data",
        "softmax",
        "activation",
        "use_activation",
    ];
    object
        .keys()
        .filter(|key| !NON_INPUT_FIELDS.contains(&key.as_str()))
        .take(2)
        .count()
        == 2
}

fn has_complete_future_score_shape(object: &serde_json::Map<String, Value>) -> bool {
    (object.contains_key("queries")
        && (object.contains_key("documents") || object.contains_key("items")))
        || (object.contains_key("query") && object.contains_key("items"))
        || (object.contains_key("data_1") && object.contains_key("data_2"))
}

fn validate_rerank_query_shape(value: &Value) -> Result<usize, String> {
    match value {
        Value::String(_) => Ok(1),
        Value::Object(_) => validate_score_input_shape("query", value),
        _ => Err(String::from(
            "score query must be a string or multimodal content object",
        )),
    }
}

fn validate_rerank_documents_shape(value: &Value) -> Result<usize, String> {
    match value {
        Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_string) => {
            Ok(items.len())
        }
        Value::Object(_) => validate_score_input_shape("documents", value),
        _ => Err(String::from(
            "score documents must be a non-empty string array or multimodal content object",
        )),
    }
}

fn validate_score_input_shape(field: &str, value: &Value) -> Result<usize, String> {
    match value {
        Value::String(_) => Ok(1),
        Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_string) => {
            Ok(items.len())
        }
        Value::Object(object) => {
            let Some(Value::Array(items)) = object.get("content") else {
                return Err(format!(
                    "score {field} multimodal object requires a content array"
                ));
            };
            if items.is_empty() || !items.iter().all(is_valid_score_content_part) {
                return Err(format!(
                    "score {field} multimodal content must contain valid content parts"
                ));
            }
            Ok(items.len())
        }
        _ => Err(format!(
            "score {field} must be a string, non-empty string array, or multimodal content object"
        )),
    }
}

fn validate_score_input_lengths(len_1: usize, len_2: usize) -> Result<(), String> {
    if len_1 > 1 && len_1 != len_2 {
        Err(String::from(
            "score input lengths must be either 1:1, 1:N, or N:N",
        ))
    } else {
        Ok(())
    }
}

fn is_valid_score_content_part(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    match object.get("type").and_then(Value::as_str) {
        Some("text") => object.get("text").is_some_and(Value::is_string),
        Some("image_url") => object
            .get("image_url")
            .and_then(Value::as_object)
            .is_some_and(|image| {
                image.get("url").is_some_and(Value::is_string)
                    && image.get("detail").is_none_or(|detail| {
                        matches!(detail.as_str(), Some("auto" | "low" | "high"))
                    })
            }),
        Some("image_embeds") => {
            object
                .get("image_embeds")
                .is_none_or(|embeds| match embeds {
                    Value::Null | Value::String(_) => true,
                    Value::Object(values) => values.values().all(Value::is_string),
                    _ => false,
                })
                && object
                    .get("uuid")
                    .is_none_or(|uuid| uuid.is_null() || uuid.is_string())
        }
        Some("video_url") => object
            .get("video_url")
            .and_then(Value::as_object)
            .and_then(|video| video.get("url"))
            .is_some_and(Value::is_string),
        _ => false,
    }
}

/// Rewrite a vLLM `/v1/score` JSON body into a `/v1/rerank` body.
///
/// Supported score shapes:
/// - `text_1: string`, `text_2: string` → single document
/// - `text_1: string`, `text_2: [string, ...]` → multi document
/// - already-rerank shape with `query` + `documents` is passed through (path still rewritten)
pub(crate) fn score_body_to_rerank_body(body: &Bytes) -> Result<Bytes, String> {
    let parsed = parse_score_value(body)?;
    let object = parsed
        .value
        .as_object()
        .ok_or_else(|| String::from("score body must be a JSON object"))?;

    // A legacy rerank-shaped score body is accepted only when canonical score fields
    // are absent. vLLM permits extra fields, so `text_1`/`text_2` must win collisions.
    if !object.contains_key("text_1")
        && !object.contains_key("text_2")
        && object.contains_key("query")
        && object.contains_key("documents")
    {
        return Ok(body.clone());
    }

    let model = match object.get("model") {
        Some(Value::String(model)) => Some(Value::String(model.clone())),
        Some(Value::Null) | None => None,
        Some(_) => return Err(String::from("score model must be a string or null")),
    };
    let text_1 = object
        .get("text_1")
        .ok_or_else(|| String::from("score body missing text_1"))?;
    let text_2 = object
        .get("text_2")
        .ok_or_else(|| String::from("score body missing text_2"))?;

    let query = text_1
        .as_str()
        .ok_or_else(|| String::from("score text_1 must be a string"))?
        .to_owned();

    let documents: Vec<Value> = match text_2 {
        Value::String(doc) => vec![Value::String(doc.clone())],
        Value::Array(items) => {
            if items.is_empty() {
                return Err(String::from("score text_2 array must be non-empty"));
            }
            for item in items {
                if !item.is_string() {
                    return Err(String::from("score text_2 array items must be strings"));
                }
            }
            items.clone()
        }
        _ => {
            return Err(String::from(
                "score text_2 must be a string or array of strings",
            ));
        }
    };

    let mut out = serde_json::Map::new();
    let mut preserved_passthrough_fields = BTreeMap::new();
    // Preserve non-mapped score options (priority, truncate_prompt_tokens, additional_data, ...).
    for (key, value) in object {
        if matches!(
            key.as_str(),
            "text_1" | "text_2" | "query" | "documents" | "model" | "top_n"
        ) {
            continue;
        }
        out.insert(key.clone(), value.clone());
        if let Some(raw) = parsed.preserved_fields.get(key) {
            preserved_passthrough_fields.insert(key.clone(), *raw);
        }
    }
    if let Some(model) = model {
        out.insert(String::from("model"), model);
    }
    out.insert(String::from("query"), Value::String(query));
    let top_n = documents.len();
    out.insert(String::from("documents"), Value::Array(documents));
    // Preserve original order for score mapping (index aligns with documents).
    // `top_n` is not a ScoreRequest field, so ignore any caller extra and request all scores.
    out.insert(String::from("top_n"), json!(top_n));

    serialize_rerank_object(&out, &preserved_passthrough_fields)
}

fn serialize_rerank_object(
    object: &serde_json::Map<String, Value>,
    preserved_fields: &BTreeMap<String, &RawValue>,
) -> Result<Bytes, String> {
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut output);
    let mut map = serializer
        .serialize_map(Some(object.len()))
        .map_err(|error| format!("serialize rerank body failed: {error}"))?;
    for (key, value) in object {
        if let Some(raw) = preserved_fields.get(key) {
            map.serialize_entry(key, *raw)
        } else {
            map.serialize_entry(key, value)
        }
        .map_err(|error| format!("serialize rerank body failed: {error}"))?;
    }
    map.end()
        .map_err(|error| format!("serialize rerank body failed: {error}"))?;
    Ok(Bytes::from(output))
}

/// Rewrite a `/v1/rerank` response into a vLLM-compatible `/v1/score` response.
///
/// Expected score shape for healthcheck:
/// `{ "data": [ { "index": 0, "object": "score", "score": <f64> }, ... ] }`
///
/// Fail-closed: any malformed entry or duplicate index rejects the whole body.
/// When `expected` is set, validate result cardinality and document-index domain.
pub(crate) fn rerank_response_to_score_response(
    body: &Bytes,
    model: Option<&str>,
    expected: Option<ScoreExpectations>,
) -> Result<Bytes, String> {
    let value: Value =
        serde_json::from_slice(body).map_err(|error| format!("invalid rerank JSON: {error}"))?;

    let items = match value.get("results") {
        Some(results) => results.as_array().ok_or_else(|| {
            String::from("rerank response results must be an array for score adapter")
        })?,
        None => value.get("data").and_then(Value::as_array).ok_or_else(|| {
            String::from("rerank response missing results/data scores for score adapter")
        })?,
    };
    if items.is_empty() {
        return Err(String::from(
            "rerank response results/data is empty for score adapter",
        ));
    }

    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(items.len());
    let mut seen = std::collections::BTreeSet::new();
    for (entry_i, item) in items.iter().enumerate() {
        let index = item
            .get("index")
            .or_else(|| item.get("document_index"))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                format!("rerank result entry {entry_i} missing or invalid index for score adapter")
            })?;
        let score = item
            .get("relevance_score")
            .or_else(|| item.get("score"))
            .or_else(|| item.get("rerank_score"))
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!("rerank result entry {entry_i} missing or invalid score for score adapter")
            })?;
        if !seen.insert(index) {
            return Err(format!(
                "rerank result has duplicate index {index} for score adapter"
            ));
        }
        scored.push((index, score));
    }

    // Stable score list ordered by original document index.
    scored.sort_by_key(|(index, _)| *index);
    if let Some(expected) = expected {
        if scored.len() != expected.result_count {
            return Err(format!(
                "rerank result count {} != expected top_n {} for score adapter",
                scored.len(),
                expected.result_count
            ));
        }
        for (index, _) in &scored {
            if *index >= expected.document_count {
                return Err(format!(
                    "rerank result index {index} out of range for document_count {}",
                    expected.document_count
                ));
            }
        }
        // When all documents are requested, require complete coverage of 0..n.
        if expected.result_count == expected.document_count {
            for expect_i in 0..expected.document_count {
                if !seen.contains(&expect_i) {
                    return Err(format!(
                        "rerank result missing index {expect_i} for score adapter"
                    ));
                }
            }
        }
    }
    let data: Vec<Value> = scored
        .into_iter()
        .map(|(index, score)| {
            json!({
                "index": index,
                "object": "score",
                "score": score,
            })
        })
        .collect();

    let (id, model_value, created) = score_response_identity(&value, model)?;
    let usage = score_usage_from_rerank_response(&value)?;

    let out = json!({
        "id": id,
        "object": "list",
        "created": created,
        "model": model_value,
        "data": data,
        "usage": usage,
    });

    serde_json::to_vec(&out)
        .map(Bytes::from)
        .map_err(|error| format!("serialize score response failed: {error}"))
}

fn score_response_identity(
    value: &Value,
    fallback_model: Option<&str>,
) -> Result<(String, String, u64), String> {
    let id = match value.get("id") {
        Some(Value::String(id)) => id.clone(),
        Some(_) => return Err(String::from("rerank response id must be a string")),
        None => String::from("score-adapted"),
    };
    let model = match value.get("model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => return Err(String::from("rerank response model must be a string")),
        None => fallback_model
            .map(str::to_owned)
            .ok_or_else(|| String::from("rerank response missing model for score adapter"))?,
    };
    let created = match value.get("created") {
        Some(created) => created.as_u64().ok_or_else(|| {
            String::from("rerank response created must be a non-negative integer")
        })?,
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs()),
    };
    Ok((id, model, created))
}

fn score_usage_from_rerank_response(value: &Value) -> Result<Value, String> {
    let usage = match value.get("usage") {
        Some(Value::Object(usage)) => Some(usage),
        Some(_) => return Err(String::from("rerank response usage must be an object")),
        None => None,
    };
    let token_count = |field: &str| -> Result<Option<u64>, String> {
        usage
            .and_then(|usage| usage.get(field))
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    format!("rerank response usage.{field} must be a non-negative integer")
                })
            })
            .transpose()
    };
    let prompt_tokens = token_count("prompt_tokens")?;
    let total_tokens = token_count("total_tokens")?;
    let prompt_tokens = prompt_tokens.or(total_tokens).unwrap_or(0);
    let total_tokens = total_tokens.unwrap_or(prompt_tokens);
    let completion_tokens = match usage.and_then(|usage| usage.get("completion_tokens")) {
        Some(Value::Null) => Value::Null,
        Some(value) => Value::from(value.as_u64().ok_or_else(|| {
            String::from(
                "rerank response usage.completion_tokens must be a non-negative integer or null",
            )
        })?),
        None => Value::from(0),
    };
    let prompt_tokens_details = score_prompt_tokens_details(usage)?;
    Ok(json!({
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_details": prompt_tokens_details,
    }))
}

fn score_prompt_tokens_details(
    usage: Option<&serde_json::Map<String, Value>>,
) -> Result<Value, String> {
    let details = match usage.and_then(|usage| usage.get("prompt_tokens_details")) {
        Some(Value::Object(details)) => details,
        Some(Value::Null) | None => return Ok(Value::Null),
        Some(_) => {
            return Err(String::from(
                "rerank response usage.prompt_tokens_details must be an object or null",
            ));
        }
    };
    let cached_tokens = match details.get("cached_tokens") {
        Some(Value::Null) | None => Value::Null,
        Some(value) => Value::from(value.as_u64().ok_or_else(|| {
            String::from(
                "rerank response usage.prompt_tokens_details.cached_tokens must be a non-negative integer or null",
            )
        })?),
    };
    Ok(json!({"cached_tokens": cached_tokens}))
}

/// Expectations derived from a rewritten rerank request body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScoreExpectations {
    /// Number of results emitted by vLLM after its request-shape-specific truncation.
    pub result_count: usize,
    /// Logical document count; valid indices are `0..document_count`.
    pub document_count: usize,
}

/// Parse score/rerank request expectations for fail-closed response validation.
pub(crate) fn score_expectations_from_rerank_body(body: &Bytes) -> Option<ScoreExpectations> {
    let parsed = parse_score_value(body).ok()?;
    let value = &parsed.value;
    let documents = value.get("documents")?;
    let document_count = validate_rerank_documents_shape(documents).ok()?;
    if document_count == 0 {
        return None;
    }
    // vLLM v0.14.0rc2 compares top_n with len(request.documents). For a
    // multimodal dictionary this is the outer key count (including retained
    // extras), not the logical content count used to produce scoring outputs.
    let result_count = if let Value::Object(object) = documents {
        value
            .get("top_n")
            .and_then(parse_lax_positive_top_n)
            .filter(|top_n| *top_n < object.len())
            .map_or(document_count, |top_n| top_n.min(document_count))
    } else {
        value
            .get("top_n")
            .and_then(parse_lax_positive_top_n)
            .map_or(document_count, |n| n.min(document_count))
    };
    Some(ScoreExpectations {
        result_count,
        document_count,
    })
}

/// Rewrite request URI path `/v1/score` → `/v1/rerank` while keeping query string.
pub(crate) fn score_uri_to_rerank_uri(uri: &Uri) -> Result<Uri, String> {
    let path = uri.path();
    if path != "/v1/score" {
        return Ok(uri.clone());
    }
    let rewritten = match uri.query() {
        Some(query) => format!("/v1/rerank?{query}"),
        None => String::from("/v1/rerank"),
    };
    rewritten
        .parse::<Uri>()
        .map_err(|error| format!("rewrite score uri failed: {error}"))
}

#[cfg(test)]
mod tests;
