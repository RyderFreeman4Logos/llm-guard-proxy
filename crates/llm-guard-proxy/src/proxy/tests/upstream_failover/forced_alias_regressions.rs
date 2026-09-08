use super::*;

const FORCED_RERANK_ALIAS: &str = "forced-reranker-alias";
const FORCED_RERANK_UPSTREAM_MODEL: &str = "aeon-ultimate";

#[tokio::test]
async fn forced_alias_deepinfra_initial_request_preserves_native_body_and_metadata() {
    let mut deepinfra =
        FakeUpstream::spawn_with_deepinfra_response_body(r#"{"scores":[0.25],"input_tokens":1}"#)
            .await;
    let config = forced_alias_config(&single_deepinfra_reranker_profile_config(
        &deepinfra.base_url,
    ));
    let proxy = spawn_observed_failover_proxy(&deepinfra.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .json(&json!({
            "model": FORCED_RERANK_ALIAS,
            "query": "native DeepInfra body",
            "documents": ["document"],
        }))
        .send()
        .await
        .expect("forced DeepInfra request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    let request = deepinfra.recv_next().await;
    assert_eq!(
        request.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );
    assert_deepinfra_native_body(&request);

    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(metadata.len(), 1);
    assert_forced_deepinfra_attempt_metadata(&metadata[0], "primary", false, &request);
    let evidence = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].status, "accepted");
}

#[tokio::test]
async fn forced_alias_openai_to_deepinfra_failover_preserves_native_body_and_metadata() {
    let mut openai = FakeUpstream::spawn().await;
    let mut deepinfra =
        FakeUpstream::spawn_with_deepinfra_response_body(r#"{"scores":[0.25],"input_tokens":1}"#)
            .await;
    let config = forced_alias_config(&openai_to_deepinfra_reranker_failover_profile_config(
        &openai.base_url,
        &deepinfra.base_url,
    ));
    let proxy = spawn_observed_failover_proxy(&openai.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .json(&json!({
            "model": FORCED_RERANK_ALIAS,
            "text_1": "malformed-openai-failover",
            "text_2": ["document"],
        }))
        .send()
        .await
        .expect("forced request should fail over to DeepInfra");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    let openai_probe = openai
        .recv_within(Duration::from_secs(2))
        .await
        .expect("OpenAI primary should receive its readiness probe");
    assert_eq!(openai_probe.path_and_query, "/v1/models");
    let openai_request = openai
        .recv_within(Duration::from_secs(2))
        .await
        .expect("OpenAI primary should receive the rerank request");
    assert_eq!(openai_request.path_and_query, "/v1/rerank");
    let request = deepinfra
        .recv_within(Duration::from_secs(2))
        .await
        .expect("DeepInfra failover should receive the rerank request");
    assert_eq!(
        request.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );
    assert_deepinfra_native_body(&request);

    let metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(metadata.len(), 2);
    assert_forced_deepinfra_attempt_metadata(&metadata[1], "failover", true, &request);
    let evidence = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(evidence.len(), 2);
    assert_eq!(evidence[0].status, "rejected");
    assert_eq!(evidence[1].status, "accepted");
}

#[tokio::test]
async fn forced_alias_in_flight_hot_reload_keeps_retry_snapshot() {
    let (primary_base_url, _primary) = spawn_shielded_503_upstream().await;
    let (backup_base_url, mut receiver, release, _backup) = spawn_retry_boundary_upstream().await;
    let config = forced_alias_config(&shielded_openai_failover_profile_config(
        &primary_base_url,
        &backup_base_url,
    ));
    let proxy = spawn_observed_failover_proxy(&backup_base_url, &config).await;
    let client = proxy.client.clone();
    let url = format!("{}/v1/chat/completions", proxy.base_url);
    let request = tokio::spawn(async move {
        client
            .post(url)
            .json(&json!({
                "model": FORCED_RERANK_ALIAS,
                "stream": true,
                "messages": [{"role": "user", "content": "retain profile A"}],
            }))
            .send()
            .await
    });

    let probe = receiver
        .recv()
        .await
        .expect("failover readiness probe should reach the boundary");
    assert_eq!(probe.path_and_query, "/v1/models");
    let current = fs::read_to_string(proxy.root.join("config.toml"))
        .expect("current config should be readable");
    let reloaded = current
        .replace(
            "upstream_model = \"aeon-ultimate\"",
            "upstream_model = \"reloaded-canonical\"",
        )
        .replace("output_cap = 16", "output_cap = 8")
        .replace("temperature = 0.7", "temperature = 0.2");
    let replacement = proxy.root.join("config.next");
    fs::write(&replacement, reloaded).expect("profile B should be written");
    fs::rename(&replacement, proxy.root.join("config.toml"))
        .expect("profile B should be published");
    proxy.manager.reload().expect("profile B should hot reload");
    let live = proxy
        .manager
        .handle()
        .snapshot()
        .expect("reloaded snapshot should be readable");
    let live_profile = &live.forced_model_alias_profiles[0];
    assert_eq!(live_profile.upstream_model, "reloaded-canonical");
    assert_eq!(live_profile.output_cap, Some(8));
    assert_eq!(live_profile.temperature, Some(0.2));
    release
        .send(())
        .expect("failover readiness should still be paused");

    let response = timeout(STREAM_COMPLETION_TIMEOUT, request)
        .await
        .expect("retried request should complete")
        .expect("retried request task should not panic")
        .expect("retried request should succeed");
    let status = response.status();
    let response_body = response.text().await.expect("response should be readable");
    assert_eq!(status, StatusCode::OK, "{response_body}");
    assert!(
        openai_sse_json_chunks(&response_body)
            .iter()
            .filter_map(|chunk| chunk.get("model"))
            .all(|model| model == FORCED_RERANK_ALIAS),
        "the client-facing response must retain the requested public alias"
    );
    let retry = receiver
        .recv()
        .await
        .expect("failover attempt should reach the backup");
    assert_eq!(retry.path_and_query, "/v1/chat/completions");
    let body: serde_json::Value =
        serde_json::from_slice(&retry.body).expect("retry body should be JSON");
    assert_eq!(body["model"], "aeon-ultimate");
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
}

async fn spawn_retry_boundary_upstream() -> (
    String,
    mpsc::Receiver<ObservedRequest>,
    oneshot::Sender<()>,
    TestServer,
) {
    let (sender, receiver) = mpsc::channel(2);
    let (release, release_rx) = oneshot::channel();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    let app = Router::new().fallback({
        let release_rx = Arc::clone(&release_rx);
        move |request: Request<Body>| {
            let sender = sender.clone();
            let release_rx = Arc::clone(&release_rx);
            async move {
                let observed = observe_request(request).await;
                let path = observed.path_and_query.clone();
                let body = observed.body.clone();
                sender
                    .send(observed)
                    .await
                    .expect("retry-boundary observation should send");
                if path == "/v1/models" {
                    let release_rx = release_rx
                        .lock()
                        .expect("retry-boundary release should lock")
                        .take()
                        .expect("failover readiness should pause once");
                    release_rx
                        .await
                        .expect("test should release failover readiness");
                    json_response("models", r#"{"object":"list","data":[]}"#.to_owned())
                } else {
                    chat_completion_sse_response(&body)
                }
            }
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("retry-boundary upstream should bind");
    let addr = listener
        .local_addr()
        .expect("retry-boundary upstream address should be available");
    let server = TestServer::new(tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("retry-boundary upstream should serve");
    }));
    (format!("http://{addr}/v1"), receiver, release, server)
}

#[tokio::test]
async fn heterogeneous_openai_terminal_restores_forced_public_model_on_success_and_failover() {
    let mut openai = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn_with_deepinfra_response(
        StatusCode::SERVICE_UNAVAILABLE,
        r#"{"error":"deepinfra unavailable"}"#,
    )
    .await;
    let primary_openai_config =
        forced_alias_config(&openai_to_deepinfra_reranker_failover_profile_config(
            &openai.base_url,
            &deepinfra.base_url,
        ));
    let primary_openai =
        spawn_observed_failover_proxy(&openai.base_url, &primary_openai_config).await;
    let failover_openai_config = forced_alias_config(
        &heterogeneous_reranker_failover_profile_config(&deepinfra.base_url, &openai.base_url),
    );
    let failover_openai =
        spawn_observed_failover_proxy(&openai.base_url, &failover_openai_config).await;

    for proxy in [&primary_openai, &failover_openai] {
        let response = proxy
            .client
            .post(format!("{}/v1/rerank", proxy.base_url))
            .json(&json!({
                "model": FORCED_RERANK_ALIAS,
                "query": "public model must be restored",
                "documents": ["document"],
            }))
            .send()
            .await
            .expect("heterogeneous OpenAI response should complete");
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = response.json().await.expect("response should be JSON");
        assert_eq!(body["model"], FORCED_RERANK_ALIAS);
        assert_eq!(body["id"], "rerank-test");
        assert_eq!(body["results"][0]["index"], 0);
    }

    assert_eq!(openai.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/rerank");
    let deepinfra_request = deepinfra.recv_next().await;
    assert_eq!(
        deepinfra_request.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/rerank");
    let attempts = read_attempt_chain_rows(&failover_openai.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[1].status, "succeeded");
}

#[tokio::test]
async fn heterogeneous_openai_terminal_restores_json_errors_without_touching_opaque_errors() {
    let openai = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn().await;
    let config = forced_alias_config(&openai_to_deepinfra_reranker_failover_profile_config(
        &openai.base_url,
        &deepinfra.base_url,
    ));
    let proxy = spawn_observed_failover_proxy(&openai.base_url, &config).await;

    let json_error = proxy
        .client
        .post(format!(
            "{}/v1/rerank?test=heterogeneous-forced-json-error",
            proxy.base_url
        ))
        .json(&json!({
            "model": FORCED_RERANK_ALIAS,
            "query": "json error",
            "documents": ["document"],
        }))
        .send()
        .await
        .expect("JSON upstream error should complete");
    assert_eq!(json_error.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_error.headers().get(RETRY_AFTER).unwrap(), "13");
    assert_eq!(
        json_error.headers().get(CONTENT_TYPE).unwrap(),
        "application/json"
    );
    assert!(json_error.headers().get("server").is_none());
    assert!(json_error.headers().get("x-upstream-only").is_none());
    let json_error: serde_json::Value = json_error.json().await.expect("JSON error body");
    assert_eq!(json_error["model"], FORCED_RERANK_ALIAS);
    assert_eq!(json_error["error"]["message"], "bad rerank request");

    for (test, expected) in [
        ("heterogeneous-forced-plain-error", b"not JSON".as_slice()),
        (
            "heterogeneous-forced-malformed-error",
            b"{\"model\":".as_slice(),
        ),
    ] {
        let response = proxy
            .client
            .post(format!("{}/v1/rerank?test={test}", proxy.base_url))
            .json(&json!({
                "model": FORCED_RERANK_ALIAS,
                "query": "opaque error",
                "documents": ["document"],
            }))
            .send()
            .await
            .expect("opaque upstream error should complete");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.bytes().await.expect("opaque body"), expected);
    }
    assert!(
        deepinfra
            .recv_within(Duration::from_millis(50))
            .await
            .is_none()
    );
}

fn forced_alias_config(config: &str) -> String {
    format!(
        "{}\n{}",
        config.replacen(
            "model = \"same-model\"",
            &format!("model = \"{FORCED_RERANK_ALIAS}\""),
            1,
        ),
        r#"
[[forced_model_alias_profiles]]
alias = "forced-reranker-alias"
upstream_model = "aeon-ultimate"
thinking_mode = "force_disable"
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0

[evidence]
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false
"#
    )
}

fn assert_deepinfra_native_body(request: &ObservedRequest) {
    let body: serde_json::Value =
        serde_json::from_slice(&request.body).expect("DeepInfra request body should be JSON");
    assert!(body.get("model").is_none());
    assert!(body.get("queries").is_some());
    assert!(body.get("documents").is_some());
    assert!(body.get("instruction").is_some());
}

fn assert_forced_deepinfra_attempt_metadata(
    attempt: &AttemptRequestMetadataRow,
    priority: &str,
    selected_failover: bool,
    request: &ObservedRequest,
) {
    let metadata = &attempt.request_metadata;
    assert_eq!(metadata["forced_alias"], FORCED_RERANK_ALIAS);
    assert_eq!(
        metadata["forced_upstream_model"],
        FORCED_RERANK_UPSTREAM_MODEL
    );
    assert_eq!(
        metadata["upstream_endpoint_protocol"],
        "deepinfra_qwen3_rerank"
    );
    assert_eq!(metadata["upstream_endpoint_priority"], priority);
    assert_eq!(
        metadata["upstream_failover_selected"],
        selected_failover.to_string()
    );
    assert_eq!(metadata["path"], "/v1/inference/Qwen/Qwen3-Reranker-8B");
    assert_eq!(
        metadata["upstream_request_body_bytes"],
        request.body.len().to_string()
    );
}
