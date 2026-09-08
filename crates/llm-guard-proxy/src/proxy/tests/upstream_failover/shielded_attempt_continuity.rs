use super::*;

#[derive(Clone, Copy)]
enum PrimaryChatScript {
    AlwaysUnavailable,
    LoopThenUnavailable,
    LoopThenSuccess,
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
async fn forced_alias_policy_survives_each_endpoint_failover_physical_attempt() {
    let mut primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let mut fallback = FakeUpstream::spawn().await;
    let config = forced_alias_failover_config(&primary.base_url, &fallback.base_url);

    for (alias, stream, thinking, budget, temperature, top_p, presence_penalty) in [
        (
            "abliterated-qwen-latest-27b-nvfp4-none",
            false,
            false,
            None,
            0.7,
            0.8,
            1.5,
        ),
        (
            "abliterated-qwen-latest-27b-nvfp4-none",
            true,
            false,
            None,
            0.7,
            0.8,
            1.5,
        ),
        (
            "abliterated-qwen-latest-27b-nvfp4-low",
            false,
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
        (
            "abliterated-qwen-latest-27b-nvfp4-low",
            true,
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
        (
            "abliterated-qwen-latest-27b-nvfp4-medium",
            false,
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
        (
            "abliterated-qwen-latest-27b-nvfp4-medium",
            true,
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
    ] {
        assert_forced_alias_failover_attempts(
            &mut primary,
            &mut fallback,
            &config,
            (
                alias,
                stream,
                thinking,
                budget,
                temperature,
                top_p,
                presence_penalty,
            ),
        )
        .await;
    }
}

#[tokio::test]
async fn canonical_reranker_failover_reapplies_forced_alias_policy_and_metadata() {
    let mut primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let mut fallback = FakeUpstream::spawn().await;
    let config = forced_alias_failover_config(&primary.base_url, &fallback.base_url);
    let proxy = spawn_observed_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .json(&json!({
            "model": "abliterated-qwen-latest-27b-nvfp4-low",
            "query": "forced canonical reranker failover",
            "documents": ["document"],
        }))
        .send()
        .await
        .expect("canonical reranker request should fail over");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("reranker response should drain");

    for request in [
        recv_chat_request(&mut primary).await,
        recv_chat_request(&mut fallback).await,
    ] {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("reranker attempt body should be JSON");
        assert_eq!(body["model"], "abliterated-qwen-latest-27b-nvfp4");
    }

    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(metadata.len(), 2);
    for attempt in metadata {
        let metadata = attempt.request_metadata;
        assert_eq!(
            metadata["forced_alias"],
            "abliterated-qwen-latest-27b-nvfp4-low"
        );
        assert_eq!(
            metadata["forced_upstream_model"],
            "abliterated-qwen-latest-27b-nvfp4"
        );
        assert_eq!(metadata["forced_thinking_mode"], "");
        assert_eq!(metadata["forced_thinking_budget"], "none");
        assert_eq!(metadata["forced_answer_headroom"], "");
        assert_eq!(metadata["forced_wire_total_cap"], "unset");
        assert_eq!(metadata["attempt_thinking_mode"], "");
        assert_eq!(metadata["attempt_thinking_budget_tokens"], "none");
        assert_eq!(metadata["attempt_thinking_max_tokens"], "unset");
    }
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

#[tokio::test]
async fn shielded_retry_stays_on_the_successful_failover_endpoint() {
    let mut primary = spawn_scripted_primary(PrimaryChatScript::AlwaysUnavailable).await;
    let fallback = spawn_scripted_primary(PrimaryChatScript::LoopThenSuccess).await;
    let config = shielded_failover_config(&primary.base_url, &fallback.base_url, None, 3);
    let proxy = spawn_observed_failover_proxy(&primary.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "keep the terminal failover endpoint"}],
        }))
        .send()
        .await
        .expect("shielded retry should complete through the fallback endpoint");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("shielded fallback retry stream should drain");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_attempt_numbers(&attempts, &[1, 2, 3]);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("endpoint_http_503")
    );
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].status, "succeeded");
    assert_endpoint_attribution(&metadata, 0, "primary", false);
    assert_endpoint_attribution(&metadata, 1, "failover", true);
    assert_endpoint_attribution(&metadata, 2, "failover", false);
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        primary.recv_next().await.path_and_query,
        "/v1/chat/completions"
    );
    assert!(
        primary
            .recv_within(Duration::from_millis(100))
            .await
            .is_none(),
        "the unhealthy primary must not receive a second logical shielded retry"
    );
}

async fn assert_forced_alias_failover_attempts(
    primary: &mut FakeUpstream,
    fallback: &mut FakeUpstream,
    config: &str,
    (alias, stream, thinking, budget, temperature, top_p, presence_penalty): (
        &str,
        bool,
        bool,
        Option<u64>,
        f64,
        f64,
        f64,
    ),
) {
    let proxy = spawn_failover_proxy(&primary.base_url, config).await;
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": alias, "stream": stream, "messages": [], "temperature": 9, "top_p": 9,
            "top_k": 9, "min_p": 9, "presence_penalty": 9, "frequency_penalty": 9,
            "repetition_penalty": 9, "max_completion_tokens": 9,
            "extra_body": {"temperature": 8, "thinking": {"enabled": false}},
            "array": [{"frequency_penalty": 8}],
        }))
        .send()
        .await
        .expect("endpoint failover request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");
    for request in [
        recv_chat_request(primary).await,
        recv_chat_request(fallback).await,
    ] {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("physical attempt body should be JSON");
        assert_eq!(body["model"], "abliterated-qwen-latest-27b-nvfp4");
        assert_eq!(body["temperature"], temperature);
        assert_eq!(body["top_p"], top_p);
        assert_eq!(body["top_k"], 20);
        assert_eq!(body["min_p"], 0.0);
        assert_eq!(body["presence_penalty"], presence_penalty);
        assert_eq!(body["repetition_penalty"], 1.0);
        assert_eq!(body["max_tokens"], if thinking { 81920 } else { 16384 });
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], thinking);
        assert_eq!(
            body.get("thinking_token_budget")
                .and_then(serde_json::Value::as_u64),
            budget
        );
        for pointer in [
            "/frequency_penalty",
            "/max_completion_tokens",
            "/extra_body/temperature",
            "/extra_body/thinking/enabled",
        ] {
            assert!(body.pointer(pointer).is_none(), "must strip {pointer}");
        }
        assert_eq!(body["array"][0]["frequency_penalty"], 8);
    }
}

fn forced_alias_failover_config(primary_base_url: &str, fallback_base_url: &str) -> String {
    format!(
        r#"
[shielding]
enabled = true
[upstream.hot_restart]
enabled = false
[retry]
enabled = true
max_attempts = 2
shielded_streaming_enabled = true
[[upstreams]]
name = "forced-failover"
base_url = "{primary_base_url}"
match_models = ["abliterated-qwen-latest-27b-nvfp4-none", "abliterated-qwen-latest-27b-nvfp4-low", "abliterated-qwen-latest-27b-nvfp4-medium"]
request_timeout_ms = 1000
health_probe_interval_ms = 200
health_probe_timeout_ms = 400
health_probe_max_wait_ms = 2000
endpoint_selection = "priority_failover"
[[upstreams.endpoints]]
base_url = "{primary_base_url}"
priority = "primary"
protocol = "openai"
[[upstreams.endpoints]]
base_url = "{fallback_base_url}"
priority = "failover"
protocol = "openai"
{FORCED_ALIAS_PROFILES}
"#
    )
}

const FORCED_ALIAS_PROFILES: &str = r#"
[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-nvfp4-none"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_disable"
output_cap = 16384
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-nvfp4-low"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_thinking"
thinking_budget = 65536
output_cap = 16384
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-nvfp4-medium"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_thinking"
thinking_budget = 65536
output_cap = 16384
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
"#;

fn request_model(request: &ObservedRequest) -> String {
    let value: serde_json::Value =
        serde_json::from_slice(&request.body).expect("forwarded chat body should be JSON");
    value["model"]
        .as_str()
        .expect("forwarded chat body should contain a string model")
        .to_owned()
}

async fn recv_chat_request(fake: &mut FakeUpstream) -> ObservedRequest {
    for _ in 0..4 {
        let request = fake
            .recv_within(Duration::from_secs(2))
            .await
            .expect("endpoint should receive a bounded request sequence");
        if request.path_and_query != "/v1/models" {
            return request;
        }
    }
    panic!("endpoint only received health probes, not a chat request");
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
    let server = TestServer::new(tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("scripted shielded primary failed: {error}");
        }
    }));
    FakeUpstream {
        base_url: format!("http://{addr}/v1"),
        receiver,
        _server: server,
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
    if matches!(
        state.script,
        PrimaryChatScript::LoopThenUnavailable | PrimaryChatScript::LoopThenSuccess
    ) && attempt == 0
    {
        return repeated_reasoning_line_sse_response(200);
    }
    if matches!(state.script, PrimaryChatScript::LoopThenSuccess) {
        return completed_chat_sse_response();
    }
    upstream_status_json_response(StatusCode::SERVICE_UNAVAILABLE)
}

fn completed_chat_sse_response() -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"fallback success\"}}]}\n\ndata: [DONE]\n\n",
        ))
        .expect("scripted successful SSE response should build")
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
request_timeout_ms = 3_000
health_probe_interval = "200ms"
health_probe_timeout = "400ms"
health_probe_max_wait = "2s"

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
