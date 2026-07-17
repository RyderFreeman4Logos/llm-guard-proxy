use super::*;

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
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("13")
    );
    assert!(response.headers().get("server").is_none());
    assert!(response.headers().get("x-upstream-endpoint").is_none());
    let body = response.text().await.expect("body");
    assert!(body.contains("upstream boom"), "{body}");

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("should forward to rerank");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
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
