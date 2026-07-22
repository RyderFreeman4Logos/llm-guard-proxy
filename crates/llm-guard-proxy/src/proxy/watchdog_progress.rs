use axum::http::Uri;
use serde_json::Value;

/// Incomplete SSE residual after complete frames are drained.
const MAX_PENDING_SSE_RESIDUAL_BYTES: usize = 64 * 1024;
/// Incomplete non-SSE JSON documents (embeddings/rerank) may exceed one SSE frame
/// and arrive across many TCP chunks; keep a larger bound so progress is only
/// recognized once the full document parses.
const MAX_PENDING_RESULT_DOCUMENT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WatchdogProgressUnit {
    Chat,
    Completion,
    Embedding,
    Reranker,
}

/// Maps a request path to an explicit progress protocol.
///
/// Unknown routes return `None` so the stuck-engine watchdog excludes them
/// rather than defaulting to Chat deltas (e.g. Responses API events would never
/// match Chat parsing and would falsely look stalled).
pub(super) fn watchdog_progress_unit(uri: &Uri) -> Option<WatchdogProgressUnit> {
    match uri.path() {
        "/v1/chat/completions" | "/chat/completions" => Some(WatchdogProgressUnit::Chat),
        "/v1/completions" | "/completions" => Some(WatchdogProgressUnit::Completion),
        "/v1/embeddings" | "/embeddings" => Some(WatchdogProgressUnit::Embedding),
        "/v1/rerank" | "/v1/score" | "/rerank" | "/score" => Some(WatchdogProgressUnit::Reranker),
        _ => None,
    }
}

pub(super) fn emitted_progress(
    progress_unit: WatchdogProgressUnit,
    pending: &mut Vec<u8>,
    chunk: &[u8],
) -> u64 {
    // Append first, then parse complete protocol units. The incomplete residual is
    // the only buffer that may grow; its size is capped after draining complete frames.
    pending.extend(chunk.iter().copied().filter(|byte| *byte != b'\r'));
    let progress = match progress_unit {
        WatchdogProgressUnit::Chat => complete_sse_progress(pending, sse_event_has_model_content),
        WatchdogProgressUnit::Completion => {
            complete_sse_progress(pending, sse_event_has_completion_text)
        }
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            complete_result_progress(progress_unit, pending)
        }
    };
    let residual_cap = match progress_unit {
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker
            if !has_sse_framing(pending) =>
        {
            MAX_PENDING_RESULT_DOCUMENT_BYTES
        }
        _ => MAX_PENDING_SSE_RESIDUAL_BYTES,
    };
    if pending.len() > residual_cap {
        // Incomplete residual only: discarding an oversized residual avoids unbounded
        // growth without dropping already-recognized complete progress frames.
        pending.clear();
    }
    progress
}

fn complete_sse_progress<F>(pending: &mut Vec<u8>, event_has_progress: F) -> u64
where
    F: Fn(&Value) -> bool,
{
    let mut progress = 0_u64;
    while let Some(frame_end) = pending.windows(2).position(|window| window == b"\n\n") {
        let frame = pending.drain(..frame_end + 2).collect::<Vec<_>>();
        progress = progress.saturating_add(sse_progress(&frame, &event_has_progress));
    }
    progress
}

fn complete_result_progress(progress_unit: WatchdogProgressUnit, pending: &mut Vec<u8>) -> u64 {
    if has_sse_framing(pending) {
        return complete_sse_progress(pending, |event| {
            result_event_has_progress(progress_unit, event)
        });
    }

    let Ok(event) = serde_json::from_slice::<Value>(pending) else {
        return 0;
    };
    pending.clear();
    u64::from(result_event_has_progress(progress_unit, &event))
}

fn has_sse_framing(pending: &[u8]) -> bool {
    pending.split(|byte| *byte == b'\n').any(|line| {
        [b"data:".as_slice(), b"event:", b"id:", b"retry:", b":"]
            .iter()
            .any(|prefix| line.starts_with(prefix))
    })
}

fn sse_progress<F>(frame: &[u8], event_has_progress: &F) -> u64
where
    F: Fn(&Value) -> bool,
{
    frame
        .split(|byte| *byte == b'\n')
        .filter_map(|line| line.strip_prefix(b"data:"))
        .map(trim_ascii)
        .filter(|data| !data.is_empty() && *data != b"[DONE]")
        .filter_map(|data| serde_json::from_slice::<Value>(data).ok())
        .filter(|event| event_has_progress(event))
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len().saturating_sub(1)];
    }
    bytes
}

fn result_event_has_progress(progress_unit: WatchdogProgressUnit, event: &Value) -> bool {
    match progress_unit {
        WatchdogProgressUnit::Embedding => event
            .get("data")
            .and_then(Value::as_array)
            .is_some_and(|results| results.iter().any(embedding_result_has_content)),
        WatchdogProgressUnit::Reranker => ["results", "data"].iter().any(|field| {
            event
                .get(*field)
                .and_then(Value::as_array)
                .is_some_and(|results| results.iter().any(reranker_result_has_content))
        }),
        WatchdogProgressUnit::Chat | WatchdogProgressUnit::Completion => false,
    }
}

fn embedding_result_has_content(result: &Value) -> bool {
    result.get("embedding").is_some_and(non_empty_json_value)
}

fn reranker_result_has_content(result: &Value) -> bool {
    result.as_object().is_some_and(|fields| {
        ["relevance_score", "rerank_score", "score"]
            .iter()
            .any(|field| fields.get(*field).is_some_and(Value::is_number))
    })
}

fn non_empty_json_value(value: &Value) -> bool {
    match value {
        Value::Array(values) => !values.is_empty(),
        Value::String(value) => !value.is_empty(),
        _ => false,
    }
}

fn sse_event_has_completion_text(event: &Value) -> bool {
    event
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.is_empty())
            })
        })
}

fn sse_event_has_model_content(event: &Value) -> bool {
    event
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
                    return false;
                };
                ["content", "reasoning_content", "reasoning", "thinking"]
                    .iter()
                    .any(|field| {
                        delta
                            .get(*field)
                            .and_then(Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                    })
                    || delta
                        .get("tool_calls")
                        .is_some_and(tool_calls_have_model_content)
                    || delta
                        .get("function_call")
                        .is_some_and(function_call_has_model_content)
            })
        })
}

fn tool_calls_have_model_content(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|calls| calls.iter().any(tool_call_has_model_content))
}

fn tool_call_has_model_content(value: &Value) -> bool {
    value
        .get("function")
        .and_then(Value::as_object)
        .is_some_and(function_fields_have_model_content)
}

fn function_call_has_model_content(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(function_fields_have_model_content)
}

fn function_fields_have_model_content(fields: &serde_json::Map<String, Value>) -> bool {
    ["name", "arguments"].iter().any(|field| {
        fields
            .get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    })
}
