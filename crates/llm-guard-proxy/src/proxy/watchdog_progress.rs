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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WatchdogProgressState {
    Progress(u64),
    Incomplete,
    UnobservableOversize,
}

#[derive(Debug)]
pub(super) struct WatchdogProgressParser {
    pending: Vec<u8>,
    state: WatchdogProgressState,
}

impl Default for WatchdogProgressParser {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            state: WatchdogProgressState::Incomplete,
        }
    }
}

pub(super) fn emitted_progress(
    progress_unit: WatchdogProgressUnit,
    parser: &mut WatchdogProgressParser,
    chunk: &[u8],
) -> WatchdogProgressState {
    if parser.state == WatchdogProgressState::UnobservableOversize {
        return parser.state;
    }

    let pending_cap = residual_cap(progress_unit, &parser.pending, chunk);
    let retained_chunk_len = chunk.iter().filter(|byte| **byte != b'\r').count();
    if retained_chunk_len > pending_cap.saturating_sub(parser.pending.len()) {
        if parser.pending.is_empty()
            && let Some(progress) = complete_progress_without_buffering(progress_unit, chunk)
        {
            parser.state = progress_state(progress);
            return parser.state;
        }
        parser.state = WatchdogProgressState::UnobservableOversize;
        return parser.state;
    }

    parser
        .pending
        .extend(chunk.iter().copied().filter(|byte| *byte != b'\r'));
    let progress = match progress_unit {
        WatchdogProgressUnit::Chat => {
            complete_sse_progress(&mut parser.pending, sse_event_has_model_content)
        }
        WatchdogProgressUnit::Completion => {
            complete_sse_progress(&mut parser.pending, sse_event_has_completion_text)
        }
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            complete_result_progress(progress_unit, &mut parser.pending)
        }
    };
    let pending_cap = residual_cap(progress_unit, &parser.pending, &[]);
    if parser.pending.len() > pending_cap {
        parser.pending.truncate(pending_cap);
        parser.state = WatchdogProgressState::UnobservableOversize;
    } else {
        parser.state = progress_state(progress);
    }
    parser.state
}

/// Records a single progress unit for a complete non-SSE Chat or Completion
/// response once the upstream has reached EOF. Unlike SSE, raw JSON has no frame
/// boundary before EOF, so recognizing it while chunks arrive could record an
/// incomplete document or couple liveness to downstream consumption.
pub(super) fn non_sse_progress_at_eof(
    progress_unit: WatchdogProgressUnit,
    parser: &mut WatchdogProgressParser,
) -> WatchdogProgressState {
    if parser.state == WatchdogProgressState::UnobservableOversize
        || !matches!(
            progress_unit,
            WatchdogProgressUnit::Chat | WatchdogProgressUnit::Completion
        )
        || has_sse_framing(&parser.pending)
    {
        return parser.state;
    }

    let progress = u64::from(
        serde_json::from_slice::<Value>(&parser.pending)
            .ok()
            .is_some_and(|response| complete_chat_or_completion_response(&response)),
    );
    parser.pending.clear();
    parser.state = progress_state(progress);
    parser.state
}

const fn progress_state(progress: u64) -> WatchdogProgressState {
    if progress == 0 {
        WatchdogProgressState::Incomplete
    } else {
        WatchdogProgressState::Progress(progress)
    }
}

fn residual_cap(progress_unit: WatchdogProgressUnit, pending: &[u8], chunk: &[u8]) -> usize {
    match progress_unit {
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker
            if !has_sse_framing(pending) && !has_sse_framing(chunk) =>
        {
            MAX_PENDING_RESULT_DOCUMENT_BYTES
        }
        _ => MAX_PENDING_SSE_RESIDUAL_BYTES,
    }
}

fn complete_progress_without_buffering(
    progress_unit: WatchdogProgressUnit,
    chunk: &[u8],
) -> Option<u64> {
    match progress_unit {
        WatchdogProgressUnit::Chat => {
            complete_sse_progress_without_buffering(chunk, sse_event_has_model_content)
        }
        WatchdogProgressUnit::Completion => {
            complete_sse_progress_without_buffering(chunk, sse_event_has_completion_text)
        }
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker
            if has_sse_framing(chunk) =>
        {
            complete_sse_progress_without_buffering(chunk, |event| {
                result_event_has_progress(progress_unit, event)
            })
        }
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            serde_json::from_slice::<Value>(chunk)
                .ok()
                .map(|event| u64::from(result_event_has_progress(progress_unit, &event)))
        }
    }
}

fn complete_sse_progress_without_buffering<F>(chunk: &[u8], event_has_progress: F) -> Option<u64>
where
    F: Fn(&Value) -> bool,
{
    if chunk.contains(&b'\r') {
        return None;
    }
    let mut remaining = chunk;
    let mut progress = 0_u64;
    while let Some(frame_end) = remaining.windows(2).position(|window| window == b"\n\n") {
        let frame_end = frame_end.saturating_add(2);
        progress =
            progress.saturating_add(sse_progress(&remaining[..frame_end], &event_has_progress));
        remaining = &remaining[frame_end..];
    }
    remaining.is_empty().then_some(progress)
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

fn complete_chat_or_completion_response(response: &Value) -> bool {
    response
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
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
