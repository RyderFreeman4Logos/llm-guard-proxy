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
async fn signed_rerank_skips_transforming_endpoint_and_preserves_openai_headers() {
    let chat = FakeUpstream::spawn().await;
    let mut deepinfra = FakeUpstream::spawn().await;
    let mut openai = FakeUpstream::spawn_with_rerank_status(StatusCode::OK).await;
    let extra_config =
        heterogeneous_reranker_failover_profile_config(&deepinfra.base_url, &openai.base_url);
    let proxy = spawn_failover_proxy(&chat.base_url, &extra_config).await;
    let body = serde_json::json!({
        "model": "same-model",
        "query": "signed query",
        "documents": ["signed document"],
        "top_n": 1,
    })
    .to_string();

    let response = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header("accept-encoding", "gzip")
        .header("content-digest", "sha-256=:signed-content-digest:")
        .header("signature-input", "sig=(\"content-digest\")")
        .header("signature", "sig=:signed-signature:")
        .body(body)
        .send()
        .await
        .expect("signed rerank request should use the OpenAI endpoint");
    assert_eq!(response.status(), StatusCode::OK);

    let transformed = deepinfra.recv_within(Duration::from_millis(100)).await;
    assert!(
        transformed.is_none(),
        "integrity-bound request must not be transformed for DeepInfra: {transformed:?}"
    );
    let probe = openai
        .recv_within(Duration::from_secs(1))
        .await
        .expect("OpenAI endpoint should receive a health probe");
    assert_eq!(probe.path_and_query, "/v1/models");
    let request = openai
        .recv_within(Duration::from_secs(1))
        .await
        .expect("OpenAI endpoint should receive the signed rerank request");
    assert_eq!(request.path_and_query, "/v1/rerank");
    assert_eq!(request.headers["accept-encoding"], "gzip");
    assert_eq!(
        request.headers["content-digest"],
        "sha-256=:signed-content-digest:"
    );
    assert_eq!(
        request.headers["signature-input"],
        "sig=(\"content-digest\")"
    );
    assert_eq!(request.headers["signature"], "sig=:signed-signature:");
}

#[tokio::test]
async fn incompatible_failover_endpoint_does_not_consume_probe_deadline() {
    let mut primary = FakeUpstream::spawn_with_rerank_status(StatusCode::SERVICE_UNAVAILABLE).await;
    let mut incompatible = FakeUpstream::spawn().await;
    let extra_config = openai_to_deepinfra_reranker_failover_profile_config(
        &primary.base_url,
        &incompatible.base_url,
    );
    let proxy = spawn_failover_proxy(&primary.base_url, &extra_config).await;
    let started_at = Instant::now();

    let response = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .json(&serde_json::json!({
            "model": "same-model",
            "query": "opaque extension",
            "documents": ["document"],
            "instruction": "cannot be represented by DeepInfra",
        }))
        .send()
        .await
        .expect(
            "terminal OpenAI response should return without waiting for an ineligible endpoint",
        );

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        started_at.elapsed() < Duration::from_millis(250),
        "protocol-incompatible failover must not burn the 400ms probe deadline"
    );
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/rerank");
    assert!(
        incompatible
            .recv_within(Duration::from_millis(30))
            .await
            .is_none()
    );
}

#[tokio::test]
async fn credentialless_failover_endpoint_does_not_consume_probe_deadline() {
    const MISSING_KEY_ENV: &str = "LLM_GUARD_PROXY_T4_MISSING_DEEPINFRA_KEY_01KXXEM1";
    assert!(std::env::var_os(MISSING_KEY_ENV).is_none());
    let mut primary = FakeUpstream::spawn_with_rerank_status(StatusCode::SERVICE_UNAVAILABLE).await;
    let mut credentialless = FakeUpstream::spawn().await;
    let extra_config = openai_to_deepinfra_config_with_key_env(
        &primary.base_url,
        &credentialless.base_url,
        MISSING_KEY_ENV,
    );
    let proxy = spawn_failover_proxy(&primary.base_url, &extra_config).await;
    let started_at = Instant::now();

    let response = proxy
        .client
        .post(format!("{}/v1/rerank", proxy.base_url))
        .json(&serde_json::json!({
            "model": "same-model",
            "query": "credential boundary",
            "documents": ["document"],
        }))
        .send()
        .await
        .expect("terminal OpenAI response should return without waiting for credentials");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        started_at.elapsed() < Duration::from_millis(250),
        "credentialless failover must not burn the 400ms probe deadline"
    );
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(primary.recv_next().await.path_and_query, "/v1/rerank");
    assert!(
        credentialless
            .recv_within(Duration::from_millis(30))
            .await
            .is_none()
    );
}

fn openai_to_deepinfra_config_with_key_env(
    primary_base_url: &str,
    deepinfra_base_url: &str,
    api_key_env: &str,
) -> String {
    format!(
        r#"
[[profile]]
model = "same-model"
request_timeout_ms = 400
health_probe_interval = "200ms"
health_probe_timeout = "20ms"
health_probe_max_wait = "400ms"

[[profile.upstream]]
base_url = "{primary_base_url}"
priority = "primary"
protocol = "openai"

[[profile.upstream]]
base_url = "{deepinfra_base_url}"
priority = "failover"
protocol = "deepinfra_qwen3_rerank"
model = "Qwen/Qwen3-Reranker-8B"
model_revision = "5fa94080caafeaa45a15d11f969d7978e087a3db"
api_key_env = "{api_key_env}"
"#
    )
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
