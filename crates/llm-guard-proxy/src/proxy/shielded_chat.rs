use std::collections::BTreeMap;

use axum::body::Bytes;
use bytes::BytesMut;
use futures_util::{Stream, StreamExt};
use llm_guard_proxy_core::RawPayloads;
use serde_json::{Map, Number, Value, json};

use super::{MAX_PROXY_BODY_BYTES, sanitized_reqwest_error, unix_time_millis};

/// Prepared upstream request body for shielded non-stream chat completion handling.
pub(super) struct PreparedChatRequest {
    upstream_body: Bytes,
}

impl PreparedChatRequest {
    pub(super) fn upstream_body(&self) -> Bytes {
        self.upstream_body.clone()
    }
}

/// Forces upstream streaming and usage frames for non-stream chat requests parsed as JSON.
pub(super) fn prepare_non_stream_request(body: &Bytes) -> Option<PreparedChatRequest> {
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
    })
}

/// Accepted, aggregated OpenAI-compatible chat completion response.
pub(super) struct AggregatedChatCompletion {
    pub(super) body: Bytes,
    pub(super) response_metadata: BTreeMap<String, String>,
    pub(super) raw_payloads: RawPayloads,
}

/// Consumes upstream OpenAI-compatible chat completion SSE and aggregates it into JSON.
pub(super) async fn aggregate_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
    attempt_started_at_unix_ms: u64,
    request_id: &str,
    request_model_id: Option<&str>,
) -> Result<AggregatedChatCompletion, String> {
    let mut stream = Box::pin(stream);
    let mut buffer = BytesMut::new();
    let mut bytes_seen = 0_usize;
    let mut state = ChatAggregation::new(attempt_started_at_unix_ms);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            format!(
                "upstream SSE stream failed: {}",
                sanitized_reqwest_error(&error)
            )
        })?;
        if state.first_byte_latency_ms.is_none() {
            state.first_byte_latency_ms =
                Some(unix_time_millis().saturating_sub(attempt_started_at_unix_ms));
        }
        bytes_seen = bytes_seen
            .checked_add(chunk.len())
            .ok_or_else(|| String::from("upstream SSE body is too large"))?;
        if bytes_seen > MAX_PROXY_BODY_BYTES {
            return Err(format!(
                "upstream SSE body exceeded proxy limit: max_bytes={MAX_PROXY_BODY_BYTES}"
            ));
        }
        buffer.extend_from_slice(&chunk);

        while let Some(event) = next_sse_event(&mut buffer) {
            state.apply_event(&event)?;
        }
    }

    if buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
        return Err(String::from(
            "upstream SSE ended with an unterminated event",
        ));
    }

    state.finish(request_id, request_model_id)
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
}

impl ChatAggregation {
    fn new(attempt_started_at_unix_ms: u64) -> Self {
        Self {
            attempt_started_at_unix_ms,
            ..Self::default()
        }
    }

    fn apply_event(&mut self, event: &[u8]) -> Result<(), String> {
        let event = std::str::from_utf8(event)
            .map_err(|error| format!("upstream SSE event was not valid UTF-8: {error}"))?;
        let data = sse_data(event);
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }

        let chunk = serde_json::from_str::<Value>(&data)
            .map_err(|error| format!("upstream SSE data was not valid JSON: {error}"))?;
        self.apply_chunk(&chunk);
        Ok(())
    }

    fn apply_chunk(&mut self, chunk: &Value) {
        let Some(chunk_object) = chunk.as_object() else {
            return;
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
            return;
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
                builder.apply_delta(
                    delta,
                    &mut self.stats,
                    &mut self.first_token_latency_ms,
                    self.attempt_started_at_unix_ms,
                );
            }
        }
    }

    fn finish(
        self,
        request_id: &str,
        request_model_id: Option<&str>,
    ) -> Result<AggregatedChatCompletion, String> {
        self.ensure_usable()?;
        let response_metadata = response_metadata(&self);
        let completion_fields =
            CompletionFields::from_aggregation(&self, request_id, request_model_id);
        let finalized_choices = finalize_choices(self.choices);
        let raw_payloads = finalized_choices.raw_payloads();
        let body = completion_body(completion_fields, finalized_choices.choices)?;

        Ok(AggregatedChatCompletion {
            body: Bytes::from(body),
            response_metadata,
            raw_payloads,
        })
    }

    fn ensure_usable(&self) -> Result<(), String> {
        if self.choices.is_empty() {
            return Err(String::from(
                "upstream SSE ended without chat completion choices",
            ));
        }
        if !self.saw_done && !self.saw_finish_reason {
            return Err(String::from(
                "upstream SSE ended before a final chat completion marker",
            ));
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
        }
    }
}

fn finalize_choices(choices: BTreeMap<u64, ChoiceBuilder>) -> FinalizedChoices {
    let mut raw_output = String::new();
    let mut raw_reasoning = String::new();
    let mut raw_tool_calls = Vec::new();
    let mut final_choices = Vec::with_capacity(choices.len());

    for choice in choices.into_values() {
        raw_output.push_str(&choice.content);
        raw_reasoning.push_str(&choice.reasoning);
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
        raw_output,
        raw_reasoning,
        raw_tool_calls,
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

    fn apply_delta(
        &mut self,
        delta: &Map<String, Value>,
        stats: &mut DeltaStats,
        first_token_latency_ms: &mut Option<u64>,
        attempt_started_at_unix_ms: u64,
    ) {
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
            }
        }
        for field in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(reasoning) = delta.get(field).and_then(Value::as_str) {
                if !reasoning.is_empty() {
                    self.reasoning.push_str(reasoning);
                    stats.reasoning_delta_count = stats.reasoning_delta_count.saturating_add(1);
                    mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                }
            }
        }
        if let Some(function_call) = delta.get("function_call").and_then(Value::as_object) {
            self.function_call
                .get_or_insert_with(FunctionCallBuilder::default)
                .apply_delta(function_call);
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
                        .apply_delta(tool_call);
                    stats.tool_call_delta_count = stats.tool_call_delta_count.saturating_add(1);
                    mark_first_token(first_token_latency_ms, attempt_started_at_unix_ms);
                }
            }
        }
    }

    fn into_value(self) -> Value {
        let mut message = Map::new();
        message.insert(
            String::from("role"),
            Value::String(self.role.unwrap_or_else(|| String::from("assistant"))),
        );
        if self.content.is_empty() && (!self.tool_calls.is_empty() || self.function_call.is_some())
        {
            message.insert(String::from("content"), Value::Null);
        } else {
            message.insert(String::from("content"), Value::String(self.content));
        }
        if !self.reasoning.is_empty() {
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
}

#[derive(Default)]
struct FunctionCallBuilder {
    name: Option<String>,
    arguments: String,
    saw_arguments: bool,
}

impl FunctionCallBuilder {
    fn apply_delta(&mut self, function_call: &Map<String, Value>) {
        if let Some(name) = function_call.get("name").and_then(Value::as_str) {
            self.name.get_or_insert_with(|| name.to_owned());
        }
        if let Some(arguments) = function_call.get("arguments").and_then(Value::as_str) {
            self.saw_arguments = true;
            self.arguments.push_str(arguments);
        }
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
    fn apply_delta(&mut self, tool_call: &Map<String, Value>) {
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
            }
        }
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

fn string_field<'value>(value: &'value Value, key: &str) -> Option<&'value str> {
    value.get(key).and_then(Value::as_str)
}

fn unix_time_secs() -> u64 {
    unix_time_millis() / 1_000
}
