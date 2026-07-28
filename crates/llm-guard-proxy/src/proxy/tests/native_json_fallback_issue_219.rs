use std::{fs, io, path::Path, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::State,
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::Response,
    routing::post,
};
use futures_util::{StreamExt, stream};
use tokio::{
    net::TcpListener,
    sync::mpsc,
    time::{sleep, timeout},
};

use super::*;

#[tokio::test]
async fn native_json_fallback_retries_a_truncated_forced_sse_body_once() {
    let mut upstream = NativeJsonFallbackUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        native_json_fallback_config(2),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"fallback"}],"stream":false}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "native fallback"
    );

    let first = upstream.recv_request().await;
    let second = upstream.recv_request().await;
    assert_eq!(first["stream"], true);
    assert_eq!(
        first["stream_options"]["include_usage"], true,
        "the protected first attempt must request usage in SSE"
    );
    assert_eq!(body_thinking_budget_json(&first), Some(32_768));
    assert_eq!(second["stream"], false);
    assert!(
        second.get("stream_options").is_none(),
        "the native JSON fallback must not inject stream_options.include_usage"
    );
    assert_eq!(body_thinking_budget_json(&second), Some(8_192));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].response_metadata["upstream_wire_mode"],
        "shielded_sse"
    );
    assert_eq!(
        attempts[0].response_metadata["upstream_stream_forced"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["sse_failure_class"],
        "body_failure"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_eligible"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_used"],
        "false"
    );
    assert_eq!(attempts[0].response_metadata["retry_budget_remaining"], "1");
    assert_eq!(
        attempts[0].response_metadata["loop_guard_coverage"],
        "full_sse"
    );
    assert_eq!(
        attempts[1].response_metadata["upstream_wire_mode"],
        "native_json_fallback"
    );
    assert_eq!(
        attempts[1].response_metadata["upstream_stream_forced"],
        "false"
    );
    assert_eq!(attempts[1].response_metadata["sse_failure_class"], "none");
    assert_eq!(
        attempts[1].response_metadata["native_json_fallback_eligible"],
        "false"
    );
    assert_eq!(
        attempts[1].response_metadata["native_json_fallback_used"],
        "true"
    );
    assert_eq!(attempts[1].response_metadata["retry_budget_remaining"], "0");
    assert_eq!(
        attempts[1].response_metadata["loop_guard_coverage"],
        "unavailable_native_json"
    );
}

#[tokio::test]
async fn native_json_fallback_requires_remaining_attempt_budget() {
    let mut upstream = NativeJsonFallbackUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        native_json_fallback_config(1),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"no budget"}],"stream":false}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _error_body = response
        .text()
        .await
        .expect("error response should be readable");
    assert_eq!(upstream.recv_request().await["stream"], true);
    assert!(
        upstream
            .recv_request_within(Duration::from_millis(100))
            .await
            .is_none(),
        "a single-attempt policy must not send native JSON"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].response_metadata["sse_failure_class"],
        "body_failure"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_eligible"],
        "false"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_used"],
        "false"
    );
    assert_eq!(attempts[0].response_metadata["retry_budget_remaining"], "0");
    assert_eq!(attempts[0].response_metadata["retry_exhausted"], "true");
}

#[tokio::test]
async fn native_json_fallback_does_not_follow_request_deadline_recovery() {
    // Delay past request_deadline_ms so the body failure is reclassified as a deadline
    // abort while still retaining sse_failure_class=body_failure (the F1 regression path).
    let mut upstream =
        NativeJsonFallbackUpstream::spawn_with_forced_sse_error_delay(Duration::from_millis(250))
            .await;
    let recovery_root = unique_test_dir("native-json-fallback-deadline-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let recovery_marker = recovery_root.join("recovered");
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &native_json_fallback_deadline_recovery_config(&recovery_marker),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"deadline body failure"}],"stream":false}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(
        response.status(),
        StatusCode::BAD_GATEWAY,
        "deadline recovery may replay protected SSE, but must not select native JSON"
    );
    let _error_body = response
        .text()
        .await
        .expect("deadline failure response should be readable");
    assert!(
        recovery_marker.exists(),
        "the deadline recovery hook must run before the protected replay"
    );

    let mut requests = Vec::new();
    while let Some(request) = upstream
        .recv_request_within(Duration::from_millis(250))
        .await
    {
        requests.push(request);
    }
    assert!(
        requests.len() >= 2,
        "expected at least first SSE plus readiness/replay; got {}",
        requests.len()
    );
    assert_eq!(requests[0]["stream"], true);
    assert!(
        requests
            .iter()
            .all(|request| request["stream"].as_bool() != Some(false)),
        "request-deadline recovery must never send a native JSON stream:false request"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert!(!attempts.is_empty());
    assert_eq!(
        attempts[0].response_metadata["request_deadline_exhausted"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_eligible"],
        "false"
    );
    assert_eq!(
        attempts[0].response_metadata["native_json_fallback_used"],
        "false"
    );
    assert!(
        attempts.iter().all(|attempt| {
            attempt
                .response_metadata
                .get("upstream_wire_mode")
                .and_then(serde_json::Value::as_str)
                != Some("native_json_fallback")
                && attempt
                    .response_metadata
                    .get("native_json_fallback_used")
                    .and_then(serde_json::Value::as_str)
                    != Some("true")
        }),
        "deadline recovery must not select native JSON wire mode"
    );

    remove_dir_all(&recovery_root);
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn loop_stall_and_protocol_failures_do_not_switch_to_native_json() {
    let mut loop_upstream = FakeUpstream::spawn().await;
    let loop_proxy = ProxyFixture::spawn_with_options(
        &loop_upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 2
anti_loop_hint_enabled = false
"#,
    )
    .await;
    let loop_response = loop_proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            loop_proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"loop"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("loop request should complete");
    assert_eq!(loop_response.status(), StatusCode::BAD_GATEWAY);
    let _loop_error = loop_response
        .text()
        .await
        .expect("loop response should be readable");
    assert_stream_flag(&loop_upstream.recv_next().await, true);
    assert_stream_flag(&loop_upstream.recv_next().await, true);
    assert_no_native_fallback(&read_attempt_chain_rows(&loop_proxy.sqlite_path));

    let mut malformed_upstream = FakeUpstream::spawn().await;
    let malformed_proxy = ProxyFixture::spawn_with_options(
        &malformed_upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        native_json_fallback_config(2),
    )
    .await;
    let malformed_response = malformed_proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=malformed-sse-invalid-json",
            malformed_proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"malformed"}],"stream":false}"#)
        .send()
        .await
        .expect("malformed request should complete");
    assert_eq!(malformed_response.status(), StatusCode::BAD_GATEWAY);
    let _malformed_error = malformed_response
        .text()
        .await
        .expect("malformed response should be readable");
    assert_stream_flag(&malformed_upstream.recv_next().await, true);
    assert!(
        malformed_upstream
            .recv_within(Duration::from_millis(100))
            .await
            .is_none(),
        "malformed SSE must not fall back to native JSON"
    );
    assert_no_native_fallback(&read_attempt_chain_rows(&malformed_proxy.sqlite_path));

    let mut stalled_upstream = FakeUpstream::spawn().await;
    let stalled_proxy = ProxyFixture::spawn_with_options(
        &stalled_upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50
"#,
    )
    .await;
    let stalled_response = stalled_proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            stalled_proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"stall"}],"stream":false}"#)
        .send()
        .await
        .expect("stalled request should complete");
    assert_eq!(stalled_response.status(), StatusCode::BAD_GATEWAY);
    let _stalled_error = stalled_response
        .text()
        .await
        .expect("stalled response should be readable");
    assert_stream_flag(&stalled_upstream.recv_next().await, true);
    assert!(
        stalled_upstream
            .recv_within(Duration::from_millis(100))
            .await
            .is_none(),
        "an upstream stall must not fall back to native JSON"
    );
    assert_no_native_fallback(&read_attempt_chain_rows(&stalled_proxy.sqlite_path));
}

fn native_json_fallback_config(max_attempts: u32) -> &'static str {
    match max_attempts {
        1 => {
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 1
anti_loop_hint_enabled = false
"#
        }
        2 => {
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "forced-sse"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "native-json"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#
        }
        _ => panic!("test config only supports one or two attempts"),
    }
}

fn native_json_fallback_deadline_recovery_config(recovery_marker: &Path) -> String {
    format!(
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 1
request_deadline_ms = 100
anti_loop_hint_enabled = false

[upstream.local_recovery]
enabled = true
trigger_on_request_deadline = true
restart_command = ["/usr/bin/touch", "{recovery_marker}"]
restart_timeout_ms = 1000
readiness_body = {{"model":"test-chat","messages":[{{"role":"user","content":"deadline recovery ready"}}],"max_tokens":1}}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 100
cooldown_ms = 1000
budget_window_ms = 10000
max_per_window = 1
"#,
        recovery_marker = recovery_marker.display()
    )
}

fn body_thinking_budget_json(body: &serde_json::Value) -> Option<u64> {
    body.get("thinking")
        .and_then(|value| value.get("budget_tokens"))
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            body.get("chat_template_kwargs")
                .and_then(|value| value.get("enable_thinking"))
                .and_then(serde_json::Value::as_bool)
                .filter(|enabled| *enabled)
                .and_then(|_| body.get("chat_template_kwargs"))
                .and_then(|value| value.get("thinking_budget"))
                .and_then(serde_json::Value::as_u64)
        })
}

fn assert_stream_flag(request: &ObservedRequest, expected: bool) {
    let body: serde_json::Value =
        serde_json::from_slice(&request.body).expect("upstream request should be JSON");
    assert_eq!(body["stream"], expected);
}

fn assert_no_native_fallback(attempts: &[AttemptChainRow]) {
    assert!(
        attempts.iter().all(|attempt| {
            attempt
                .response_metadata
                .get("upstream_wire_mode")
                .and_then(serde_json::Value::as_str)
                == Some("shielded_sse")
                && attempt
                    .response_metadata
                    .get("native_json_fallback_used")
                    .and_then(serde_json::Value::as_str)
                    == Some("false")
        }),
        "loop, stall, and malformed SSE failures must retain protected SSE attempts"
    );
}

struct NativeJsonFallbackUpstream {
    base_url: String,
    receiver: mpsc::Receiver<serde_json::Value>,
}

#[derive(Clone)]
struct NativeJsonFallbackState {
    sender: mpsc::Sender<serde_json::Value>,
    forced_sse_error_delay: Duration,
}

impl NativeJsonFallbackUpstream {
    async fn spawn() -> Self {
        Self::spawn_with_forced_sse_error_delay(Duration::ZERO).await
    }

    async fn spawn_with_forced_sse_error_delay(forced_sse_error_delay: Duration) -> Self {
        let (sender, receiver) = mpsc::channel(4);
        let app = Router::new()
            .route("/v1/chat/completions", post(native_json_fallback_handler))
            .with_state(NativeJsonFallbackState {
                sender,
                forced_sse_error_delay,
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("native JSON fallback upstream should bind");
        let address = listener
            .local_addr()
            .expect("native JSON fallback upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("native JSON fallback upstream server failed: {error}");
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            receiver,
        }
    }

    async fn recv_request(&mut self) -> serde_json::Value {
        self.receiver
            .recv()
            .await
            .expect("native JSON fallback upstream should receive a request")
    }

    async fn recv_request_within(&mut self, wait: Duration) -> Option<serde_json::Value> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }
}

async fn native_json_fallback_handler(
    State(state): State<NativeJsonFallbackState>,
    request: axum::http::Request<Body>,
) -> Response<Body> {
    let body = to_bytes(request.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("native JSON fallback request body should be readable");
    let request: serde_json::Value =
        serde_json::from_slice(&body).expect("native JSON fallback request should be JSON");
    let upstream_streaming = request["stream"].as_bool().unwrap_or(false);
    state
        .sender
        .send(request)
        .await
        .expect("native JSON fallback upstream observation should send");

    if upstream_streaming {
        let forced_sse_error_delay = state.forced_sse_error_delay;
        let stream = stream::iter([Ok::<Bytes, io::Error>(Bytes::from_static(
            b"data: {\"id\":\"partial\"}\n\n",
        ))])
        .chain(stream::once(async move {
            if !forced_sse_error_delay.is_zero() {
                sleep(forced_sse_error_delay).await;
            }
            Err(io::Error::other("synthetic truncated forced SSE body"))
        }));
        let mut response = Response::new(Body::from_stream(stream));
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        return response;
    }

    let mut response = Response::new(Body::from(
        r#"{"id":"chatcmpl-native-fallback","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"native fallback"},"finish_reason":"stop"}]}"#,
    ));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}
