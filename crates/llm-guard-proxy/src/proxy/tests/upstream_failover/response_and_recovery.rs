use std::sync::Arc;

use tokio::sync::Barrier;

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

#[tokio::test]
async fn deepinfra_cooldown_recovery_admits_exactly_one_concurrent_trial() {
    let openai = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn_with_options(
        None,
        StatusCode::OK,
        "models",
        None,
        Some(Duration::from_millis(150)),
        None,
        Some((
            StatusCode::SERVICE_UNAVAILABLE,
            r#"{"error":"deepinfra unavailable"}"#,
        )),
    )
    .await;
    let extra_config =
        heterogeneous_reranker_failover_profile_config(&deepinfra.base_url, &openai.base_url)
            .replace("request_timeout_ms = 400", "request_timeout_ms = 1000");
    let proxy = spawn_failover_proxy(&openai.base_url, &extra_config).await;

    let first = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .json(&json!({
            "model": "same-model",
            "query": "prime passive cooldown",
            "documents": ["document"],
        }))
        .send()
        .await
        .expect("initial request should fail over after DeepInfra failure");
    assert_eq!(first.status(), StatusCode::OK);
    first.bytes().await.expect("initial body should drain");
    assert_eq!(
        deepinfra.recv_next().await.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );

    sleep(Duration::from_millis(225)).await;
    let request_count = 8;
    let barrier = Arc::new(Barrier::new(request_count + 1));
    let mut requests = Vec::with_capacity(request_count);
    for index in 0..request_count {
        let client = proxy.client.clone();
        let url = format!("{}/v1/rerank", proxy.base_url);
        let barrier = Arc::clone(&barrier);
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            client
                .post(url)
                .json(&json!({
                    "model": "same-model",
                    "query": format!("recovery request {index}"),
                    "documents": ["document"],
                }))
                .send()
                .await
                .expect("concurrent recovery request should complete")
        }));
    }
    barrier.wait().await;
    let mut failed_trials = 0;
    for request in requests {
        let response = request.await.expect("recovery request task should join");
        if response.status() == StatusCode::BAD_GATEWAY {
            failed_trials += 1;
        } else {
            assert_eq!(response.status(), StatusCode::OK);
        }
        response.bytes().await.expect("recovery body should drain");
    }
    assert!(
        failed_trials <= 1,
        "only the single recovery trial may fail while healthy failover requests succeed"
    );

    assert_eq!(
        deepinfra.recv_next().await.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );
    assert!(
        deepinfra
            .recv_within(Duration::from_millis(50))
            .await
            .is_none(),
        "passive cooldown recovery must admit only one concurrent DeepInfra trial"
    );
}

#[tokio::test]
async fn generic_openai_passive_recovery_restores_full_concurrent_admission() {
    let mut openai = FakeUpstream::spawn_with_options(
        None,
        StatusCode::OK,
        "models",
        None,
        Some(Duration::from_millis(150)),
        None,
        None,
    )
    .await;
    let extra_config = openai_single_endpoint_profile_config(
        &openai.base_url,
        Some(THIRD_PARTY_TEST_KEY_ENV),
        None,
    );
    let proxy = spawn_failover_proxy(&openai.base_url, &extra_config).await;
    let url = format!(
        "{}/v1/chat/completions?test=transient-503-then-success",
        proxy.base_url
    );

    let initial = proxy
        .client
        .post(&url)
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "prime passive cooldown"}],
        }))
        .send()
        .await
        .expect("initial OpenAI request should complete");
    assert_eq!(initial.status(), StatusCode::SERVICE_UNAVAILABLE);
    initial.bytes().await.expect("initial body should drain");
    assert_eq!(
        openai.recv_next().await.path_and_query,
        "/v1/chat/completions?test=transient-503-then-success"
    );

    sleep(Duration::from_millis(225)).await;
    let recovery = proxy
        .client
        .post(&url)
        .json(&json!({
            "model": "same-model",
            "stream": true,
            "messages": [{"role": "user", "content": "recover passive endpoint"}],
        }))
        .send()
        .await
        .expect("passive OpenAI recovery trial should complete");
    assert_eq!(recovery.status(), StatusCode::OK);
    recovery.bytes().await.expect("recovery body should drain");
    assert_eq!(
        openai.recv_next().await.path_and_query,
        "/v1/chat/completions?test=transient-503-then-success"
    );

    let request_count = 8;
    let barrier = Arc::new(Barrier::new(request_count + 1));
    let mut requests = Vec::with_capacity(request_count);
    for index in 0..request_count {
        let client = proxy.client.clone();
        let url = url.clone();
        let barrier = Arc::clone(&barrier);
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            client
                .post(url)
                .json(&json!({
                    "model": "same-model",
                    "stream": true,
                    "messages": [{"role": "user", "content": format!("post-recovery request {index}")}],
                }))
                .send()
                .await
                .expect("post-recovery request should complete")
        }));
    }
    barrier.wait().await;
    for request in requests {
        let response = request
            .await
            .expect("post-recovery request task should join");
        assert_eq!(response.status(), StatusCode::OK);
        response
            .bytes()
            .await
            .expect("post-recovery body should drain");
    }
    for _ in 0..request_count {
        assert_eq!(
            openai.recv_next().await.path_and_query,
            "/v1/chat/completions?test=transient-503-then-success"
        );
    }
}

#[tokio::test]
async fn deepinfra_recovery_lease_remains_held_until_retry_classification() {
    let mut openai = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn_with_deepinfra_response(
        StatusCode::SERVICE_UNAVAILABLE,
        r#"{"error":"deepinfra unavailable"}"#,
    )
    .await;
    let extra_config =
        openai_to_deepinfra_reranker_failover_profile_config(&openai.base_url, &deepinfra.base_url);
    let proxy = spawn_observed_failover_proxy(&openai.base_url, &extra_config).await;

    prime_failed_deepinfra_recovery(&proxy, &mut openai, &mut deepinfra).await;

    sleep(Duration::from_millis(225)).await;
    let gate = Arc::new(crate::proxy::upstream_failover::EndpointClassificationGate::new());
    proxy
        .state
        .upstream_health
        .block_next_endpoint_classification(Arc::clone(&gate));

    let first_client = proxy.client.clone();
    let first_url = format!("{}/v1/score", proxy.base_url);
    let first = tokio::spawn(async move {
        first_client
            .post(first_url)
            .json(&json!({
                "model": "same-model",
                "text_1": "malformed-openai-failover",
                "text_2": ["first", "second"],
            }))
            .send()
            .await
            .expect("held recovery request should complete")
    });
    timeout(Duration::from_secs(1), gate.wait_until_arrived())
        .await
        .expect("recovery request should reach the post-send classification gate");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/rerank");
    assert_eq!(
        deepinfra.recv_next().await.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );

    let competing_client = proxy.client.clone();
    let competing_url = format!("{}/v1/score", proxy.base_url);
    let competing = tokio::spawn(async move {
        competing_client
            .post(competing_url)
            .json(&json!({
                "model": "same-model",
                "text_1": "malformed-openai-failover",
                "text_2": ["first", "second"],
            }))
            .send()
            .await
            .expect("competing recovery request should complete")
    });
    let overlapping_trial = deepinfra.recv_within(Duration::from_millis(100)).await;

    gate.release().await;
    assert!(
        overlapping_trial.is_none(),
        "a completed recovery send must retain exclusivity until its health result is classified"
    );
    let first_response = timeout(Duration::from_secs(2), first)
        .await
        .expect("held recovery task should finish after classification resumes")
        .expect("held recovery task should join");
    let competing_response = timeout(Duration::from_secs(2), competing)
        .await
        .expect("competing recovery task should finish")
        .expect("competing recovery task should join");
    first_response
        .bytes()
        .await
        .expect("held recovery body should drain");
    competing_response
        .bytes()
        .await
        .expect("competing recovery body should drain");
}

async fn prime_failed_deepinfra_recovery(
    proxy: &ProxyFixture,
    openai: &mut FakeUpstream,
    deepinfra: &mut FakeUpstream,
) {
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
        .expect("priming request should complete");
    response.bytes().await.expect("priming body should drain");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(openai.recv_next().await.path_and_query, "/v1/rerank");
    assert_eq!(
        deepinfra.recv_next().await.path_and_query,
        "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db"
    );
}
