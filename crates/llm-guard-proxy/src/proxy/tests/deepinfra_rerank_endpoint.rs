use super::*;

const DEEPINFRA_PATH: &str = "/v1/inference/Qwen/Qwen3-Reranker-8B";

#[tokio::test]
async fn deepinfra_rerank_routes_one_nn_batch_and_returns_native_shape() {
    let mut default = FakeUpstream::spawn().await;
    let mut reranker = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[upstream.metadata]
discovery_enabled = false

[[upstreams]]
name = "deepinfra-reranker"
base_url = "{}"
match_models = ["Qwen/Qwen3-Reranker-8B"]

[upstreams.metadata]
discovery_enabled = false
"#,
            reranker.base_url
        ),
    )
    .await;
    let request_body = json!({
        "queries": ["q1", "q2", "q3"],
        "documents": ["d1", "d2", "d3"],
        "instruction": deepinfra_rerank_adapter::DEFAULT_INSTRUCTION,
        "service_tier": "priority",
    });

    let response = proxy
        .client
        .post(format!(
            "{}{DEEPINFRA_PATH}?test=deepinfra-rerank-ok",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer local-reranker-key")
        .json(&request_body)
        .send()
        .await
        .expect("DeepInfra rerank request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert!(response.headers().get("server").is_none());
    assert!(response.headers().get("x-upstream-endpoint").is_none());
    assert!(response.headers().get("x-upstream-only").is_none());
    assert_ne!(
        response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("private-vllm-request-id")
    );
    let value: serde_json::Value = response.json().await.expect("DeepInfra response JSON");
    assert_eq!(value["scores"], json!([0.0, 1.0, 0.5]));
    assert_eq!(value["input_tokens"], 19);
    assert_eq!(value["request_id"], "score-native-123");
    assert_eq!(value["inference_status"]["status"], "succeeded");
    assert_eq!(value["inference_status"]["tokens_input"], 19);

    let observed = reranker
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("named reranker should receive one request");
    assert_eq!(observed.method, Method::POST);
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=deepinfra-rerank-ok"
    );
    let upstream_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("vLLM score body");
    assert_eq!(upstream_body["model"], "qwen3-reranker-8b");
    assert_eq!(upstream_body["text_1"], json!(["q1", "q2", "q3"]));
    assert_eq!(upstream_body["text_2"], json!(["d1", "d2", "d3"]));
    assert!(upstream_body.get("queries").is_none());
    assert!(upstream_body.get("documents").is_none());
    assert!(upstream_body.get("instruction").is_none());
    assert!(upstream_body.get("service_tier").is_none());
    assert_no_upstream_request(&mut default).await;

    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    let request_metadata: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request metadata");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_metadata).expect("request metadata JSON");
    assert_eq!(request_metadata["deepinfra_rerank_adapter"], "true");
    assert_eq!(request_metadata["deepinfra_service_tier"], "priority");
    assert_eq!(
        request_metadata["deepinfra_service_tier_local_behavior"],
        "single_tier"
    );
    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].request_metadata["path"], "/v1/score");
}

#[tokio::test]
async fn deepinfra_rerank_rejects_unsupported_instruction_before_upstream() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}{DEEPINFRA_PATH}", proxy.base_url))
        .json(&json!({
            "queries": ["q"],
            "documents": ["d"],
            "instruction": "Rank passages for a legal brief",
        }))
        .send()
        .await
        .expect("unsupported instruction should receive a local response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = response_json(response).await;
    assert_eq!(
        value["error"]["code"],
        "unsupported_deepinfra_rerank_feature"
    );
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("custom instruction"))
    );
    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
}

#[tokio::test]
async fn deepinfra_rerank_fails_closed_on_malformed_vllm_scores() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}{DEEPINFRA_PATH}?test=deepinfra-rerank-malformed",
            proxy.base_url
        ))
        .json(&json!({
            "queries": ["q1", "q2", "q3"],
            "documents": ["d1", "d2", "d3"],
        }))
        .send()
        .await
        .expect("malformed upstream response should complete locally");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response.headers().get("server").is_none());
    assert!(response.headers().get("x-upstream-only").is_none());
    let value = response_json(response).await;
    assert_eq!(value["error"]["type"], "upstream_body_error");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("duplicate index 0"))
    );

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("request should reach vLLM score endpoint");
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=deepinfra-rerank-malformed"
    );
}
