use super::*;

const MODEL_DISCOVERY_SENSITIVE_HEADERS: [&str; 12] = [
    "authorization",
    "x-api-key",
    "x-virtual-key",
    "cookie",
    "proxy-authorization",
    "signature",
    "signature-input",
    "digest",
    "content-digest",
    "forwarded",
    "x-forwarded-for",
    "x-real-ip",
];

fn assert_model_discovery_headers_are_safe(headers: &HeaderMap) {
    assert_eq!(
        headers.get("accept").and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    for name in MODEL_DISCOVERY_SENSITIVE_HEADERS {
        assert!(
            !headers.contains_key(name),
            "model discovery leaked sensitive caller header {name}"
        );
    }
}

#[tokio::test]
async fn aggregate_models_uses_openai_failover_without_forwarding_caller_credentials() {
    let sensitive_headers = [
        ("authorization", "Bearer aggregate-authorization-unique"),
        ("x-api-key", "aggregate-api-key-unique"),
        ("x-virtual-key", "aggregate-virtual-key-unique"),
        ("cookie", "aggregate-cookie-unique"),
        ("proxy-authorization", "aggregate-proxy-auth-unique"),
        ("signature", "aggregate-signature-unique"),
        ("signature-input", "aggregate-signature-input-unique"),
        ("digest", "aggregate-digest-unique"),
        ("content-digest", "aggregate-content-digest-unique"),
        ("forwarded", "for=aggregate-forwarded-unique"),
        ("x-forwarded-for", "aggregate-forwarded-for-unique"),
        ("x-real-ip", "aggregate-real-ip-unique"),
    ];
    let mut chat = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn().await;
    let mut openai = FakeUpstream::spawn().await;
    let extra_config =
        heterogeneous_reranker_failover_profile_config(&deepinfra.base_url, &openai.base_url);
    let proxy = spawn_failover_proxy(&chat.base_url, &extra_config).await;

    let mut request = proxy
        .client
        .get(format!(
            "{}/v1/models?test=aggregate-credential-boundary",
            proxy.base_url
        ))
        .header("accept", "application/json");
    for (name, value) in sensitive_headers {
        request = request.header(name, value);
    }
    let response = request
        .send()
        .await
        .expect("aggregate model discovery should use the OpenAI failover");

    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("aggregate models response body should drain");
    assert!(
        deepinfra
            .recv_within(Duration::from_millis(100))
            .await
            .is_none(),
        "DeepInfra must not receive model discovery or caller credentials"
    );

    let chat_request = chat.recv_next().await;
    assert_eq!(
        chat_request.path_and_query,
        "/v1/models?test=aggregate-credential-boundary"
    );
    assert_model_discovery_headers_are_safe(&chat_request.headers);

    let openai_probe = openai.recv_next().await;
    assert_eq!(openai_probe.path_and_query, "/v1/models");
    for name in MODEL_DISCOVERY_SENSITIVE_HEADERS {
        assert!(!openai_probe.headers.contains_key(name));
    }
    let openai_request = openai.recv_next().await;
    assert_eq!(
        openai_request.path_and_query,
        "/v1/models?test=aggregate-credential-boundary"
    );
    assert_model_discovery_headers_are_safe(&openai_request.headers);
}

#[tokio::test]
async fn aggregate_models_succeeds_when_deepinfra_primary_is_unreachable() {
    let mut chat = FakeUpstream::spawn().await;
    let deepinfra_base_url = closed_upstream_base_url().await;
    let mut openai = FakeUpstream::spawn().await;
    let extra_config =
        heterogeneous_reranker_failover_profile_config(&deepinfra_base_url, &openai.base_url);
    let proxy = spawn_failover_proxy(&chat.base_url, &extra_config).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=aggregate-unreachable-deepinfra",
            proxy.base_url
        ))
        .bearer_auth("aggregate-models-unreachable-token-unique")
        .send()
        .await
        .expect("aggregate model discovery should bypass unreachable DeepInfra");

    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("aggregate models response body should drain");
    assert!(
        chat.recv_next().await.headers.get(AUTHORIZATION).is_none(),
        "caller credential must not reach the default OpenAI endpoint"
    );
    assert!(
        openai
            .recv_next()
            .await
            .headers
            .get(AUTHORIZATION)
            .is_none(),
        "caller credential must not reach the OpenAI readiness probe"
    );
    assert!(
        openai
            .recv_next()
            .await
            .headers
            .get(AUTHORIZATION)
            .is_none(),
        "caller credential must not reach the selected OpenAI endpoint"
    );
}

#[tokio::test]
async fn openai_caller_auth_errors_do_not_fail_over_or_cool_down_shared_endpoint() {
    for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
        let mut primary = FakeUpstream::spawn_with_rerank_status(status).await;
        let mut cloud = FakeUpstream::spawn().await;
        let extra_config = openai_to_deepinfra_reranker_failover_profile_config(
            &primary.base_url,
            &cloud.base_url,
        );
        let proxy = spawn_failover_proxy(&primary.base_url, &extra_config).await;

        let first = proxy
            .client
            .post(format!("{}/v1/rerank", proxy.base_url))
            .bearer_auth("caller-a")
            .json(&json!({"model": "same-model", "query": "auth", "documents": ["d"]}))
            .send()
            .await
            .expect("first caller auth response should return directly");
        assert_eq!(first.status(), status);
        first
            .bytes()
            .await
            .expect("first caller auth response body should drain");
        assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
        assert_eq!(primary.recv_next().await.path_and_query, "/v1/rerank");
        assert!(cloud.recv_within(Duration::from_millis(30)).await.is_none());

        let second = proxy
            .client
            .post(format!("{}/v1/rerank", proxy.base_url))
            .bearer_auth("caller-b")
            .json(&json!({"model": "same-model", "query": "auth", "documents": ["d"]}))
            .send()
            .await
            .expect("second caller should still reach the shared primary");
        assert_eq!(second.status(), status);
        second
            .bytes()
            .await
            .expect("second caller auth response body should drain");
        assert_eq!(primary.recv_next().await.path_and_query, "/v1/rerank");
        assert!(cloud.recv_within(Duration::from_millis(30)).await.is_none());
    }
}
