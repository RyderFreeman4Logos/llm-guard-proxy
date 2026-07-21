use super::*;

#[derive(Clone, Copy)]
enum PrimaryChatScript {
    AlwaysUnavailable,
    LoopThenUnavailable,
}

#[derive(Clone)]
struct ScriptedPrimaryState {
    sender: mpsc::Sender<ObservedRequest>,
    chat_attempts: Arc<AtomicU64>,
    script: PrimaryChatScript,
}

#[tokio::test]
async fn shielded_streaming_endpoint_failover_rerenders_caller_model_for_unaliased_fallback() {
    let mut primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let mut fallback = FakeUpstream::spawn().await;
    let config = shielded_failover_config(
        &primary.base_url,
        &fallback.base_url,
        Some("vendor-primary-alias"),
        2,
    );
    let proxy = spawn_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "preserve the caller model"}],
        }))
        .send()
        .await
        .expect("shielded endpoint failover should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("shielded fallback stream should drain");

    assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
    let primary_request = primary.recv_next().await;
    assert_eq!(primary_request.path_and_query, "/v1/chat/completions");
    assert_eq!(request_model(&primary_request), "vendor-primary-alias");

    assert_eq!(fallback.recv_next().await.path_and_query, "/v1/models");
    let fallback_request = fallback.recv_next().await;
    assert_eq!(fallback_request.path_and_query, "/v1/chat/completions");
    assert_eq!(request_model(&fallback_request), "same-model");
}

#[tokio::test]
async fn shielded_physical_attempts_preserve_begin_terminal_error_chain() {
    let mut primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let (fallback_base_url, fallback_probe_seen) = spawn_probe_then_stop_upstream().await;
    let config = shielded_failover_config(&primary.base_url, &fallback_base_url, None, 1);
    let proxy = spawn_observed_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "preserve terminal endpoint errors"}],
        }))
        .send()
        .await
        .expect("shielded terminal error should return a proxy response");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    response
        .bytes()
        .await
        .expect("shielded terminal error response should drain");
    fallback_probe_seen
        .await
        .expect("fallback should become ready before its terminal connect failure");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_attempt_numbers(&attempts, &[1, 2]);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("endpoint_http_503")
    );
    assert_eq!(attempts[1].status, "failed");
    assert_eq!(
        attempts[1].response_metadata["error_type"],
        "upstream_connect_error"
    );
    assert_endpoint_attribution(&metadata, 0, "primary", false);
    assert_endpoint_attribution(&metadata, 1, "failover", true);
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        primary.recv_next().await.path_and_query,
        "/v1/chat/completions"
    );
}

#[tokio::test]
async fn shielded_physical_attempts_preserve_immediate_nonstream_failover_chain() {
    let primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let fallback = FakeUpstream::spawn().await;
    let config = shielded_failover_config(&primary.base_url, &fallback.base_url, None, 2);
    let proxy = spawn_observed_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "stream": false,
            "messages": [{"role": "user", "content": "preserve immediate attempts"}],
        }))
        .send()
        .await
        .expect("shielded nonstream endpoint failover should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("shielded nonstream response should drain");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_attempt_numbers(&attempts, &[1, 2]);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("endpoint_http_503")
    );
    assert_eq!(attempts[1].status, "succeeded");
    assert_endpoint_attribution(&metadata, 0, "primary", false);
    assert_endpoint_attribution(&metadata, 1, "failover", true);
}

#[tokio::test]
async fn shielded_physical_attempts_preserve_later_retry_failover_chain() {
    let primary = spawn_scripted_primary(PrimaryChatScript::LoopThenUnavailable).await;
    let fallback = FakeUpstream::spawn().await;
    let config = shielded_failover_config(&primary.base_url, &fallback.base_url, None, 3);
    let proxy = spawn_observed_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "preserve later attempts"}],
        }))
        .send()
        .await
        .expect("shielded retry endpoint failover should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("shielded retry stream should drain");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_attempt_numbers(&attempts, &[1, 2, 3]);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(
        attempts[1].retry_reason.as_deref(),
        Some("endpoint_http_503")
    );
    assert_eq!(attempts[2].status, "succeeded");
    assert_endpoint_attribution(&metadata, 0, "primary", false);
    assert_endpoint_attribution(&metadata, 1, "primary", false);
    assert_endpoint_attribution(&metadata, 2, "failover", true);
}

fn request_model(request: &ObservedRequest) -> String {
    let value: serde_json::Value =
        serde_json::from_slice(&request.body).expect("forwarded chat body should be JSON");
    value["model"]
        .as_str()
        .expect("forwarded chat body should contain a string model")
        .to_owned()
}

fn assert_attempt_numbers(attempts: &[AttemptChainRow], expected: &[u32]) {
    assert_eq!(
        attempts
            .iter()
            .map(|attempt| attempt.attempt_number)
            .collect::<Vec<_>>(),
        expected,
        "physical endpoint attempts must remain ordered and continuous"
    );
}

fn assert_endpoint_attribution(
    metadata: &[AttemptRequestMetadataRow],
    index: usize,
    priority: &str,
    failover_selected: bool,
) {
    assert_eq!(
        metadata[index].request_metadata["upstream_endpoint_priority"],
        priority
    );
    assert_eq!(
        metadata[index].request_metadata["upstream_failover_selected"],
        failover_selected.to_string()
    );
}

async fn spawn_scripted_primary(script: PrimaryChatScript) -> FakeUpstream {
    let (sender, receiver) = mpsc::channel(16);
    let app = Router::new()
        .fallback(scripted_primary_handler)
        .with_state(ScriptedPrimaryState {
            sender,
            chat_attempts: Arc::new(AtomicU64::new(0)),
            script,
        });
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("scripted shielded primary should bind");
    let addr = listener
        .local_addr()
        .expect("scripted shielded primary address should be available");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("scripted shielded primary failed: {error}");
        }
    });
    FakeUpstream {
        base_url: format!("http://{addr}/v1"),
        receiver,
    }
}

async fn scripted_primary_handler(
    State(state): State<ScriptedPrimaryState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    let path = observed
        .path_and_query
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    state
        .sender
        .send(observed)
        .await
        .expect("scripted primary observation should send");
    if path == "/v1/models" {
        return json_response("models", r#"{"object":"list","data":[]}"#.to_owned());
    }
    let attempt = state.chat_attempts.fetch_add(1, Ordering::SeqCst);
    if matches!(state.script, PrimaryChatScript::LoopThenUnavailable) && attempt == 0 {
        return repeated_reasoning_line_sse_response(200);
    }
    upstream_status_json_response(StatusCode::SERVICE_UNAVAILABLE)
}

fn shielded_failover_config(
    primary_base_url: &str,
    fallback_base_url: &str,
    primary_model: Option<&str>,
    max_attempts: u32,
) -> String {
    let primary_model =
        primary_model.map_or_else(String::new, |model| format!("model = \"{model}\"\n"));
    format!(
        r#"
[[profile]]
model = "same-model"
request_timeout_ms = 1_000
health_probe_interval = "200ms"
health_probe_timeout = "20ms"
health_probe_max_wait = "400ms"

[[profile.upstream]]
base_url = "{primary_base_url}"
priority = "primary"
protocol = "openai"
{primary_model}
[[profile.upstream]]
base_url = "{fallback_base_url}"
priority = "failover"
protocol = "openai"

[shielding]
enabled = true

[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[upstream.hot_restart]
enabled = false

[retry]
enabled = true
max_attempts = {max_attempts}
shielded_streaming_enabled = true
"#
    )
}
