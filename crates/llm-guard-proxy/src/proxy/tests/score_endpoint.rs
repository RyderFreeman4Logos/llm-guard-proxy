use super::*;

#[tokio::test]
async fn score_endpoint_passthrough_batched_text1() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-batch", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":["q1","q2"],"text_2":["d1","d2"],"query":"ignored-extra","documents":["ignored-extra"]}"#)
        .send()
        .await
        .expect("batch score should complete");
    // Fake upstream has no /v1/score → 404, but path must remain score (not rewritten).
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("batch score should reach upstream as /v1/score");
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/score?test=score-batch");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("body json");
    assert_eq!(observed_body["text_1"][0], "q1");
    assert_eq!(observed_body["text_2"][1], "d2");
    assert_eq!(observed_body["query"], "ignored-extra");
    assert_eq!(observed_body["documents"][0], "ignored-extra");
    let _ = response.status();
}

#[tokio::test]
async fn score_endpoint_passthrough_unknown_complete_shape() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","left_input":"q","right_input":"d","future":{}}}"#,
        "9".repeat(1_000)
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-future-shape",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(request_body.clone())
        .send()
        .await
        .expect("future score shape should complete");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("future score shape should reach upstream unchanged");
    assert_eq!(observed.path_and_query, "/v1/score?test=score-future-shape");
    assert_eq!(observed.body.as_ref(), request_body.as_bytes());
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response.bytes().await.expect("response body should drain");

    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    let request_meta: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request metadata");
    let metadata: serde_json::Value =
        serde_json::from_str(&request_meta).expect("metadata should be JSON");
    assert_eq!(metadata["score_via_rerank"], "false");
    assert_eq!(metadata["score_passthrough"], "true");
    assert!(metadata.get("score_batch_passthrough").is_none());
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn score_endpoint_ignores_chat_param_overrides() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
temperature = 0.6
",
        ),
    )
    .await;
    let digits = "9".repeat(1_000);
    let canonical =
        format!(r#"{{"model":"test-chat","text_1":"q","text_2":"d","future":{digits}}}"#);

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-param-override",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(canonical)
        .send()
        .await
        .expect("canonical score should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("score response should drain");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("adapted score should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_text = std::str::from_utf8(&observed.body).expect("body should be utf-8");
    assert!(observed_text.contains(&digits));
    assert!(!observed_text.contains("temperature"));

    let future = r#"{"model":"test-chat","left_input":"q","right_input":"d"}"#;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-param-override-future",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(future)
        .send()
        .await
        .expect("future score should complete");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response
        .bytes()
        .await
        .expect("future response should drain");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("future score should reach score upstream");
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=score-param-override-future"
    );
    assert_eq!(observed.body.as_ref(), future.as_bytes());
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn score_endpoint_ignores_blocking_chat_pre_request_guard() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("score-guard-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "block", guard_result("block", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;
    let body = r#"{"model":"test-chat","text_1":"q","text_2":"d","messages":[{"role":"user","content":"ignored score extra"}]}"#;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-guard-block",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header("content-digest", "sha-256=:stale-original-body:")
        .body(body)
        .send()
        .await
        .expect("score request should bypass chat guard");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    let observed = fake.recv_next().await;
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    assert!(observed.headers.get("content-digest").is_none());
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn score_endpoint_ignores_replacing_chat_pre_request_guard() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("score-guard-replace");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let replacement = r#"[{"role":"user","content":"guard replacement"}]"#;
    let script = write_guard_script(
        &guard_root,
        "replace",
        guard_result("replace", Some(replacement)),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;
    let body = r#"{"model":"test-chat","left_input":"q","right_input":"d","messages":[{"role":"user","content":"ignored score extra"}]}"#;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-guard-replace",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header("content-digest", "sha-256=:original-body:")
        .body(body)
        .send()
        .await
        .expect("future score should bypass chat guard");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response.bytes().await.expect("response body should drain");
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=score-guard-replace"
    );
    assert_eq!(observed.body.as_ref(), body.as_bytes());
    assert_eq!(
        observed
            .headers
            .get("content-digest")
            .and_then(|value| value.to_str().ok()),
        Some("sha-256=:original-body:")
    );
}

#[tokio::test]
async fn score_endpoint_passthrough_non_success_rerank_status() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-upstream-500",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/problem+json")
    );
    let body = response.text().await.expect("body");
    assert!(body.contains("upstream boom"), "{body}");

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("should forward to rerank");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
}

#[tokio::test]
async fn score_endpoint_rejects_oversized_body_before_upstream_routing() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d","padding":"{}"}}"#,
        "x".repeat(1024 * 1024)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-oversized", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await
        .expect("oversized score request should receive an error response");
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = response.text().await.expect("error body should drain");
    assert!(
        body.contains("score request exceeded adapter limit"),
        "{body}"
    );

    assert!(
        tokio::time::timeout(Duration::from_millis(100), fake.recv_next())
            .await
            .is_err(),
        "oversized score request must fail before upstream routing"
    );
}

#[tokio::test]
async fn score_endpoint_profile_limit_mode_caps_body_before_model_routing() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "allowed"
base_url = "{0}"
match_models = ["allowed-model"]
max_in_flight_requests = 2

[[upstreams]]
name = "forbidden"
base_url = "{0}"
match_models = ["qwen3-reranker-8b"]
max_in_flight_requests = 2

[[listeners]]
name = "score-restricted"
bind_host = "127.0.0.1"
port = 18003
allowed_upstreams = ["allowed"]
"#,
            fake.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "score-restricted");
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d","padding":"{}"}}"#,
        "x".repeat(1024 * 1024)
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/score?test=score-profile-limit-oversized")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(request_body))
        .expect("score request should build");

    let response = proxy_handler(State(proxy.state.for_listener(listener)), request).await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("error body should drain");
    assert!(
        std::str::from_utf8(&body)
            .expect("error body should be UTF-8")
            .contains("score request exceeded adapter limit")
    );
}

#[tokio::test]
async fn score_endpoint_body_read_error_preserves_received_response_observability() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-body-read-error",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should receive a proxy error");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    response
        .bytes()
        .await
        .expect("proxy error response body should drain");

    fake.recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("score request should reach rerank upstream");
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(attempt_row.status, "failed");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.response_metadata["upstream_response_received"],
        "true"
    );
    assert_eq!(attempt_row.response_metadata["http_status_success"], "true");
    assert_eq!(
        attempt_row.response_metadata["response_header_content-type"],
        "application/vnd.rerank+json"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_x-request-id"],
        "rerank-body-error-123"
    );
}

#[tokio::test]
async fn score_endpoint_negotiates_identity_encoding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-identity", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(axum::http::header::ACCEPT_ENCODING, "gzip, deflate")
        .header("content-encoding", "gzip")
        .header("content-md5", "stale-md5")
        .header("digest", "SHA-256=stale")
        .header("content-digest", "sha-256=:stale:")
        .header("repr-digest", "sha-256=:stale:")
        .header("etag", "stale-etag")
        .header("if-match", "stale-etag")
        .header("if-none-match", "stale-etag")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::OK);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("should reach upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let accept_encoding = observed
        .headers
        .get(axum::http::header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok());
    assert_eq!(accept_encoding, Some("identity"));
    assert!(observed.headers.get("content-encoding").is_none());
    assert!(observed.headers.get("content-md5").is_none());
    assert!(observed.headers.get("digest").is_none());
    assert!(observed.headers.get("content-digest").is_none());
    assert!(observed.headers.get("repr-digest").is_none());
    assert!(observed.headers.get("etag").is_none());
    assert!(observed.headers.get("if-match").is_none());
    assert!(observed.headers.get("if-none-match").is_none());
    response
        .bytes()
        .await
        .expect("score response body should drain");

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    let metadata = &attempts[0].request_metadata;
    assert_eq!(metadata["path"], "/v1/rerank");
    assert_eq!(metadata["query_present"], "true");
    assert_eq!(
        metadata["upstream_request_header_accept-encoding"],
        "identity"
    );
}

#[tokio::test]
async fn score_endpoint_adapts_to_rerank_and_records_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-adapter-ok", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: serde_json::Value = response.json().await.expect("score json");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "score");
    assert_eq!(body["data"][0]["index"], 0);
    assert!(body["data"][0]["score"].as_f64().is_some());
    assert_eq!(body["model"], "qwen3-reranker-8b");
    assert!(body["created"].as_u64().is_some_and(|created| created > 0));
    assert_eq!(body["usage"]["prompt_tokens"], 0);
    assert_eq!(body["usage"]["total_tokens"], 0);
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert!(body["usage"]["prompt_tokens_details"].is_null());

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("score should forward as rerank");
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/rerank?test=score-adapter-ok");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["query"], "q");
    assert_eq!(observed_body["documents"], serde_json::json!(["d"]));
    assert_eq!(observed_body["top_n"], 1);

    // Drain EOF so buffered observer finalizes.
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        request_row.response_metadata["response_header_content-type"],
        "application/json"
    );
    assert!(
        request_row
            .response_metadata
            .get("response_header_server")
            .is_none()
    );
    assert!(
        request_row
            .response_metadata
            .get("response_header_x-request-id")
            .is_none()
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_content-type"],
        "application/vnd.rerank+json"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_server"],
        "fake-rerank"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_x-request-id"],
        "rerank-request-123"
    );
    let upstream_body =
        r#"{"id":"rerank-test","model":"qwen3-reranker-8b","results":[{"index":0,"score":1}]}"#;
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        upstream_body.len().to_string()
    );
    assert_ne!(
        request_row.response_metadata["response_body_bytes"],
        attempt_row.response_metadata["response_body_bytes"]
    );
    assert_score_request_body_provenance(
        &proxy.sqlite_path,
        request_body.len(),
        observed.body.len(),
    );
}

fn assert_score_request_body_provenance(
    sqlite_path: &std::path::Path,
    downstream_body_bytes: usize,
    upstream_body_bytes: usize,
) {
    let connection = rusqlite::Connection::open(sqlite_path).expect("sqlite open");
    let request_meta: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request metadata");
    let metadata: serde_json::Value = serde_json::from_str(&request_meta).expect("meta json");
    assert_eq!(metadata["score_via_rerank"], "true");
    assert_eq!(
        metadata["request_body_bytes"],
        downstream_body_bytes.to_string()
    );

    let attempts = read_attempt_request_metadata_rows(sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].request_metadata["upstream_request_body_bytes"],
        upstream_body_bytes.to_string()
    );
    assert_ne!(
        metadata["request_body_bytes"],
        attempts[0].request_metadata["upstream_request_body_bytes"]
    );
}

#[tokio::test]
async fn score_endpoint_rejects_invalid_client_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for request_body in [
        r"{}",
        r#"{"foo":1}"#,
        r#"{"model":"qwen3-reranker-8b","left_input":"q","softmax":true}"#,
        r#"{"softmax":true,"activation":false}"#,
        r#"{"right_input":"d","use_activation":true}"#,
        r#"{"left_input":"q","priority":1}"#,
        r#"{"left_input":"q","truncate_prompt_tokens":8}"#,
        r#"{"left_input":"q","mm_processor_kwargs":{}}"#,
        r#"{"left_input":"q","top_n":1}"#,
        r#"{"left_input":"q","additional_data":{}}"#,
        r#"{"queries":["q"],"typo":true}"#,
        r#"{"items":["d"],"typo":true}"#,
        r#"{"data_1":"q","typo":true}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q"}"#,
        r#"{"query":"q","typo":true}"#,
        r#"{"query":"q","documents":["a","b"],"top_n":"1."}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":42}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":["d",42]}"#,
        r#"{"text_1":["q1","q2"],"text_2":"d"}"#,
        r#"{"text_1":"q","text_2":{"content":[{}]}}"#,
    ] {
        let response = proxy
            .client
            .post(format!("{}/v1/score", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
            .send()
            .await
            .expect("score request should complete");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    let oversized_negative_string = format!(
        r#"{{"query":"q","documents":["d"],"top_n":"-{}"}}"#,
        "9".repeat(4_300)
    );
    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(oversized_negative_string)
        .send()
        .await
        .expect("oversized negative top_n string should complete");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn score_endpoint_fails_closed_on_partial_rerank_results() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":["a","b"],"top_n":1}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("partial score still reaches upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["top_n"], 2);

    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(attempt_row.status, "failed");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.response_metadata["upstream_response_received"],
        "true"
    );
    assert_eq!(attempt_row.response_metadata["http_status_success"], "true");
    assert_eq!(
        attempt_row.response_metadata["response_process_error"],
        "true"
    );
    let upstream_body = r#"{"id":"rerank-partial","model":"qwen3-reranker-8b","results":[{"index":0,"score":0.9}]}"#;
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        upstream_body.len().to_string()
    );
}

#[tokio::test]
async fn score_endpoint_fails_closed_on_partial_legacy_multimodal_results() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}"#,
        )
        .send()
        .await
        .expect("legacy multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("legacy multimodal score should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(
        observed_body["documents"]["content"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn score_endpoint_accepts_complete_legacy_multimodal_results_with_positive_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-multimodal-top-n",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},"top_n":1}"#,
        )
        .send()
        .await
        .expect("legacy multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body: serde_json::Value = response
        .json()
        .await
        .expect("score response should be valid JSON");
    assert_eq!(response_body["data"].as_array().unwrap().len(), 2);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("legacy multimodal score should reach rerank upstream");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["top_n"], 1);
}

#[tokio::test]
async fn score_endpoint_accepts_truncated_multimodal_results_when_extra_keys_trigger_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"future_extension":true},"top_n":1}"#,
        )
        .send()
        .await
        .expect("extended multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body: serde_json::Value = response
        .json()
        .await
        .expect("score response should be valid JSON");
    assert_eq!(response_body["data"].as_array().unwrap().len(), 1);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("extended multimodal score should reach rerank upstream");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["documents"]["future_extension"], true);
    assert_eq!(observed_body["top_n"], 1);
}

#[tokio::test]
async fn score_endpoint_accepts_arbitrary_precision_legacy_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"query":"q","documents":["a","b"],"top_n":18446744073709551616}"#)
        .send()
        .await
        .expect("arbitrary-precision score request should complete");
    assert_eq!(response.status(), StatusCode::OK);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("arbitrary-precision score request should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    assert_eq!(
        std::str::from_utf8(&observed.body).expect("rerank body UTF-8"),
        r#"{"query":"q","documents":["a","b"],"top_n":18446744073709551616}"#
    );
}

#[tokio::test]
async fn score_endpoint_routes_raw_fallback_model_to_named_upstream() {
    let mut default = FakeUpstream::spawn().await;
    let mut named = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[upstream.metadata]
discovery_enabled = false

[[upstreams]]
name = "score-route"
base_url = "{}"
match_models = ["score-model"]

[upstreams.metadata]
discovery_enabled = false
"#,
            named.base_url
        ),
    )
    .await;
    let body = format!(
        r#"{{"model":"score-model","text_1":"q","text_2":"d","future":{}}}"#,
        "9".repeat(1_000)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("named score route should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");

    let observed = named
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("named upstream should receive adapted score request");
    assert_eq!(observed.path_and_query, "/v1/rerank");
    let observed_body = std::str::from_utf8(&observed.body).expect("rerank body UTF-8");
    assert!(observed_body.contains(r#""model":"score-model""#));
    assert!(observed_body.contains(&format!(r#""future":{}"#, "9".repeat(1_000))));
    assert_no_upstream_request(&mut default).await;
}

#[test]
fn score_model_extraction_uses_raw_fallback_before_policy_and_routing() {
    let body = Bytes::from(format!(
        r#"{{"model":"forbidden-model","text_1":"q","text_2":"d","top_n":{}}}"#,
        "9".repeat(1_000)
    ));
    assert_eq!(
        extract_model_id(&Method::POST, &Uri::from_static("/v1/score"), &body).as_deref(),
        Some("forbidden-model")
    );
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn arbitrary_precision_score_top_n_cannot_bypass_model_policy() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["allowed-model"]
"#,
    )
    .await;
    let body = format!(
        r#"{{"model":"forbidden-model","text_1":"q","text_2":"d","top_n":{}}}"#,
        "9".repeat(1_000)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("score policy rejection should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("model not allowed"))
    );
    assert_no_upstream_request(&mut fake).await;
}
