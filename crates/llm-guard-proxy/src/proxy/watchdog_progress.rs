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
    sse: SseEventParser,
    sse_framing: bool,
    state: WatchdogProgressState,
}

/// Incrementally accumulates one SSE event without retaining prior dispatched
/// events. `line` owns the field currently crossing a chunk boundary and
/// `data` owns only the current event's joined `data:` values.
#[derive(Debug, Default)]
struct SseEventParser {
    line: Vec<u8>,
    data: Vec<u8>,
    has_data: bool,
    skip_lf_after_cr: bool,
}

/// Outcome of parsing an oversized SSE source chunk without retaining its
/// already-dispatched or non-data bytes.
#[derive(Debug)]
enum FastSseProgress {
    Drained {
        progress: u64,
    },
    ProgressWithResidual {
        progress: u64,
        residual: SseEventParser,
    },
    Incomplete {
        residual: SseEventParser,
    },
    UnobservableOversize,
}

impl SseEventParser {
    fn is_empty(&self) -> bool {
        self.line.is_empty() && self.data.is_empty() && !self.has_data && !self.skip_lf_after_cr
    }

    fn clear(&mut self) {
        self.line.clear();
        self.data.clear();
        self.has_data = false;
        self.skip_lf_after_cr = false;
    }

    fn retained_len(&self) -> usize {
        self.line.len().saturating_add(self.data.len())
    }
}

impl Default for WatchdogProgressParser {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            sse: SseEventParser::default(),
            sse_framing: false,
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

    if parser.sse_framing {
        if parser.sse.is_empty() && chunk.len() > MAX_PENDING_SSE_RESIDUAL_BYTES {
            return record_fast_sse_progress(
                parser,
                complete_sse_progress_without_buffering(progress_unit, chunk),
            );
        }
        let progress = parser.sse.consume(chunk, progress_unit);
        return record_sse_progress(parser, progress);
    }

    let pending_cap = residual_cap(progress_unit, &parser.pending, chunk);
    if chunk.len() > pending_cap.saturating_sub(parser.pending.len()) {
        if parser.pending.is_empty() {
            if has_sse_framing(chunk) {
                parser.sse_framing = true;
                return record_fast_sse_progress(
                    parser,
                    complete_sse_progress_without_buffering(progress_unit, chunk),
                );
            }
            if let Some(progress) = complete_progress_without_buffering(progress_unit, chunk) {
                parser.state = progress_state(progress);
                return parser.state;
            }
        }
        return mark_unobservable_oversize(parser);
    }

    parser.pending.extend_from_slice(chunk);
    if has_sse_framing(&parser.pending) {
        parser.sse_framing = true;
        let buffered = std::mem::take(&mut parser.pending);
        let progress = parser.sse.consume(&buffered, progress_unit);
        return record_sse_progress(parser, progress);
    }

    let progress = match progress_unit {
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            let Ok(event) = serde_json::from_slice::<Value>(&parser.pending) else {
                parser.state = WatchdogProgressState::Incomplete;
                return parser.state;
            };
            parser.pending.clear();
            u64::from(result_event_has_progress(progress_unit, &event))
        }
        WatchdogProgressUnit::Chat | WatchdogProgressUnit::Completion => 0,
    };
    parser.state = progress_state(progress);
    parser.state
}

fn record_sse_progress(
    parser: &mut WatchdogProgressParser,
    progress: Result<u64, ()>,
) -> WatchdogProgressState {
    match progress {
        Ok(progress) => {
            parser.state = progress_state(progress);
            parser.state
        }
        Err(()) => mark_unobservable_oversize(parser),
    }
}

fn record_fast_sse_progress(
    parser: &mut WatchdogProgressParser,
    progress: FastSseProgress,
) -> WatchdogProgressState {
    match progress {
        FastSseProgress::Drained { progress } => {
            parser.sse.clear();
            parser.state = progress_state(progress);
        }
        FastSseProgress::ProgressWithResidual { progress, residual } => {
            parser.sse = residual;
            parser.state = progress_state(progress);
        }
        FastSseProgress::Incomplete { residual } => {
            parser.sse = residual;
            parser.state = WatchdogProgressState::Incomplete;
        }
        FastSseProgress::UnobservableOversize => return mark_unobservable_oversize(parser),
    }
    parser.state
}

fn mark_unobservable_oversize(parser: &mut WatchdogProgressParser) -> WatchdogProgressState {
    parser.pending.clear();
    parser.sse.clear();
    parser.state = WatchdogProgressState::UnobservableOversize;
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
        || parser.sse_framing
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

fn complete_chat_or_completion_response(response: &Value) -> bool {
    response
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
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
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            serde_json::from_slice::<Value>(chunk)
                .ok()
                .map(|event| u64::from(result_event_has_progress(progress_unit, &event)))
        }
        WatchdogProgressUnit::Chat | WatchdogProgressUnit::Completion => None,
    }
}

/// Parses an oversized SSE chunk while retaining only the current event's joined
/// data and partial line. A single oversized data field is parsed from its
/// borrowed input slice only when its event is complete; otherwise observation
/// becomes explicitly unobservable.
fn complete_sse_progress_without_buffering(
    progress_unit: WatchdogProgressUnit,
    chunk: &[u8],
) -> FastSseProgress {
    let mut offset = 0;
    let mut progress = 0_u64;
    let mut data = Vec::new();
    let mut large_single_data: Option<&[u8]> = None;
    let mut has_data = false;
    let mut skip_lf_after_cr = false;

    while offset < chunk.len() {
        let Some((line, next_offset)) = next_sse_line(chunk, offset) else {
            if large_single_data.is_some() {
                return FastSseProgress::UnobservableOversize;
            }
            let residual = SseEventParser {
                line: chunk[offset..].to_vec(),
                data,
                has_data,
                skip_lf_after_cr: false,
            };
            if residual.retained_len() > MAX_PENDING_SSE_RESIDUAL_BYTES {
                return FastSseProgress::UnobservableOversize;
            }
            return fast_sse_progress_outcome(progress, residual);
        };
        offset = next_offset;
        if line.is_empty() {
            if has_data {
                let event_data = large_single_data.unwrap_or(&data);
                progress = progress
                    .saturating_add(u64::from(sse_data_has_progress(progress_unit, event_data)));
            }
            data.clear();
            large_single_data = None;
            has_data = false;
        } else if let Some(field_data) = line.strip_prefix(b"data:") {
            let field_data = field_data.strip_prefix(b" ").unwrap_or(field_data);
            if large_single_data.is_some() {
                return FastSseProgress::UnobservableOversize;
            }
            if !has_data && field_data.len() > MAX_PENDING_SSE_RESIDUAL_BYTES {
                large_single_data = Some(field_data);
                has_data = true;
                continue;
            }
            let joined_len = data
                .len()
                .saturating_add(usize::from(has_data))
                .saturating_add(field_data.len());
            if joined_len > MAX_PENDING_SSE_RESIDUAL_BYTES {
                return FastSseProgress::UnobservableOversize;
            }
            if has_data {
                data.push(b'\n');
            }
            data.extend_from_slice(field_data);
            has_data = true;
        }
        skip_lf_after_cr =
            offset == chunk.len() && chunk.get(offset.saturating_sub(1)) == Some(&b'\r');
    }

    if large_single_data.is_some() {
        return FastSseProgress::UnobservableOversize;
    }
    fast_sse_progress_outcome(
        progress,
        SseEventParser {
            line: Vec::new(),
            data,
            has_data,
            skip_lf_after_cr,
        },
    )
}

fn fast_sse_progress_outcome(progress: u64, residual: SseEventParser) -> FastSseProgress {
    if residual.is_empty() {
        FastSseProgress::Drained { progress }
    } else if progress > 0 {
        FastSseProgress::ProgressWithResidual { progress, residual }
    } else {
        FastSseProgress::Incomplete { residual }
    }
}

fn next_sse_line(input: &[u8], offset: usize) -> Option<(&[u8], usize)> {
    let line_end = input[offset..]
        .iter()
        .position(|byte| matches!(byte, b'\n' | b'\r'))?
        .saturating_add(offset);
    let after_line_end = if input[line_end] == b'\r' && input.get(line_end + 1) == Some(&b'\n') {
        line_end.saturating_add(2)
    } else {
        line_end.saturating_add(1)
    };
    Some((&input[offset..line_end], after_line_end))
}

impl SseEventParser {
    fn consume(&mut self, chunk: &[u8], progress_unit: WatchdogProgressUnit) -> Result<u64, ()> {
        let mut progress = 0_u64;
        for byte in chunk {
            if self.skip_lf_after_cr {
                self.skip_lf_after_cr = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\r' => {
                    progress = progress.saturating_add(self.finish_line(progress_unit)?);
                    self.skip_lf_after_cr = true;
                }
                b'\n' => {
                    progress = progress.saturating_add(self.finish_line(progress_unit)?);
                }
                byte => {
                    self.line.push(byte);
                    if self.retained_len() > MAX_PENDING_SSE_RESIDUAL_BYTES {
                        return Err(());
                    }
                }
            }
        }
        Ok(progress)
    }

    fn finish_line(&mut self, progress_unit: WatchdogProgressUnit) -> Result<u64, ()> {
        let line = std::mem::take(&mut self.line);
        if line.is_empty() {
            let progress = if self.has_data {
                u64::from(sse_data_has_progress(progress_unit, &self.data))
            } else {
                0
            };
            self.data.clear();
            self.has_data = false;
            return Ok(progress);
        }

        if let Some(data) = line.strip_prefix(b"data:") {
            let data = data.strip_prefix(b" ").unwrap_or(data);
            let joined_len = self
                .data
                .len()
                .saturating_add(usize::from(self.has_data))
                .saturating_add(data.len());
            if joined_len > MAX_PENDING_SSE_RESIDUAL_BYTES {
                return Err(());
            }
            if self.has_data {
                self.data.push(b'\n');
            }
            self.data.extend_from_slice(data);
            self.has_data = true;
        }
        Ok(0)
    }
}

fn has_sse_framing(bytes: &[u8]) -> bool {
    let mut line_start = 0;
    while line_start < bytes.len() {
        if [b"data:".as_slice(), b"event:", b"id:", b"retry:", b":"]
            .iter()
            .any(|prefix| bytes[line_start..].starts_with(prefix))
        {
            return true;
        }
        let Some(line_end) = bytes[line_start..]
            .iter()
            .position(|byte| matches!(byte, b'\n' | b'\r'))
        else {
            return false;
        };
        line_start = line_start.saturating_add(line_end + 1);
        if bytes[line_start.saturating_sub(1)] == b'\r' && bytes.get(line_start) == Some(&b'\n') {
            line_start = line_start.saturating_add(1);
        }
    }
    false
}

fn sse_data_has_progress(progress_unit: WatchdogProgressUnit, data: &[u8]) -> bool {
    let data = trim_ascii(data);
    !data.is_empty()
        && data != b"[DONE]"
        && serde_json::from_slice::<Value>(data)
            .ok()
            .is_some_and(|event| sse_event_has_progress(progress_unit, &event))
}

fn sse_event_has_progress(progress_unit: WatchdogProgressUnit, event: &Value) -> bool {
    match progress_unit {
        WatchdogProgressUnit::Chat => sse_event_has_model_content(event),
        WatchdogProgressUnit::Completion => sse_event_has_completion_text(event),
        WatchdogProgressUnit::Embedding | WatchdogProgressUnit::Reranker => {
            result_event_has_progress(progress_unit, event)
        }
    }
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
