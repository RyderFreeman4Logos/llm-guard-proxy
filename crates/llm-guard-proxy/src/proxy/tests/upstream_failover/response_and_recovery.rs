use super::*;

#[tokio::test]
async fn malformed_openai_2xx_retries_using_deepinfra_backup() {
    let mut openai = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn_with_deepinfra_response_body(
        r#"{"scores":[0.25,0.75],"input_tokens":1}"#,
    )
    .await;
    let extra_config =
        openai_to_deepinfra_reranker_failover_profile_config(&openai.base_url, &deepinfra.base_url);
    let proxy = spawn_observed_failover_proxy(&openai.base_url, &extra_config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "text_1": "malformed-openai-failover",
            "text_2": ["first", "second"],
        }))
        .send()
        .await
        .expect("malformed OpenAI response should retry DeepInfra");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/rerank");
    assert_eq!(
        deepinfra.recv_next().await.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("endpoint_protocol_response")
    );
    assert_eq!(
        attempts[0].response_metadata["endpoint_disposition"],
        "retryable_failure"
    );
    let request_metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(request_metadata.len(), 2);
    assert_eq!(
        request_metadata[0].request_metadata["endpoint_disposition"],
        "retryable_failure"
    );
    assert_eq!(
        request_metadata[1].request_metadata["endpoint_disposition"],
        "success"
    );
}
