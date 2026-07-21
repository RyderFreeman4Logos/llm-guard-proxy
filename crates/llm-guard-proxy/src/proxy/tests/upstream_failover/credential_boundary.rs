use super::*;

const MODEL_DISCOVERY_SENSITIVE_HEADERS: [&str; 13] = [
    "authorization",
    "x-api-key",
    "x-access-key",
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
        ("x-access-key", "aggregate-access-key-unique"),
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
async fn aggregate_models_persists_each_physical_failover_attempt() {
    let mut chat = FakeUpstream::spawn().await;
    let (primary_base_url, primary_probe_seen) = spawn_probe_then_stop_upstream().await;
    let mut backup = FakeUpstream::spawn().await;
    let extra_config = failover_profile_config(
        &primary_base_url,
        Some(&backup.base_url),
        "200ms",
        "20ms",
        "400ms",
    );
    let proxy = spawn_observed_failover_proxy(&chat.base_url, &extra_config).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=aggregate-physical-attempts",
            proxy.base_url
        ))
        .send()
        .await
        .expect("model discovery should fail over after the selected endpoint disconnects");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("aggregate model response should drain");
    primary_probe_seen
        .await
        .expect("primary should receive the readiness probe before disconnecting");
    assert_eq!(
        chat.recv_next().await.path_and_query,
        "/v1/models?test=aggregate-physical-attempts"
    );
    assert_eq!(backup.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        backup.recv_next().await.path_and_query,
        "/v1/models?test=aggregate-physical-attempts"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(
        attempts.len(),
        3,
        "every physical attempt must be persisted"
    );
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(
        attempts[1].retry_reason.as_deref(),
        Some("endpoint_connect_failure")
    );
    let request_metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(request_metadata.len(), 3);
    assert_eq!(
        request_metadata[1].request_metadata["upstream_failover_selected"],
        "false"
    );
    assert_eq!(
        request_metadata[2].request_metadata["upstream_failover_selected"],
        "true"
    );
    assert_ne!(
        request_metadata[1].request_metadata["upstream_endpoint_base_url"],
        request_metadata[2].request_metadata["upstream_endpoint_base_url"],
        "terminal success must be attributed to the failover endpoint"
    );
}

#[tokio::test]
async fn aggregate_models_allocates_attempts_across_failover_before_later_group() {
    let (primary_base_url, primary_probe_seen) = spawn_probe_then_stop_upstream().await;
    let mut backup = FakeUpstream::spawn().await;
    let mut second = FakeUpstream::spawn().await;
    let extra_config = format!(
        r#"
[[upstreams]]
name = "first"
base_url = "{primary_base_url}"
match_models = ["first-model"]
request_timeout_ms = 400
health_probe_interval_ms = 200
health_probe_timeout_ms = 20
health_probe_max_wait_ms = 400

[[upstreams.endpoints]]
base_url = "{primary_base_url}"
priority = "primary"

[[upstreams.endpoints]]
base_url = "{backup_base_url}"
priority = "failover"

[[upstreams]]
name = "second"
base_url = "{second_base_url}"
match_models = ["second-model"]

[[listeners]]
name = "failover-first-models"
bind_host = "127.0.0.1"
port = 18018
allowed_upstreams = ["first", "second"]
"#,
        backup_base_url = backup.base_url,
        second_base_url = second.base_url,
    );
    let proxy = spawn_observed_failover_proxy(&second.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "failover-first-models");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=request-wide-physical-attempts"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("aggregate model response should drain");
    primary_probe_seen
        .await
        .expect("first profile primary should pass readiness before disconnecting");
    assert_eq!(backup.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        backup.recv_next().await.path_and_query,
        "/v1/models?test=request-wide-physical-attempts"
    );
    assert_eq!(
        second.recv_next().await.path_and_query,
        "/v1/models?test=request-wide-physical-attempts"
    );

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    let attempt_profiles = attempts
        .iter()
        .map(|attempt| {
            (
                attempt.attempt_number,
                attempt.request_metadata["upstream_profile"]
                    .as_str()
                    .expect("attempt should identify its upstream profile"),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        attempt_profiles,
        [(1, "first"), (2, "first"), (3, "second")]
    );
}

#[tokio::test]
async fn aggregate_models_preserves_tail_budget_after_prior_group_and_slow_middle_endpoint() {
    let mut first = FakeUpstream::spawn().await;
    let (second_primary_base_url, primary_probe_seen) = spawn_probe_then_stop_upstream().await;
    let mut slow_middle = FakeUpstream::spawn_with_models_body_and_delay(
        r#"{"object":"list","data":[{"id":"slow-middle","object":"model"}]}"#,
        Duration::from_millis(300),
    )
    .await;
    let mut healthy_tail = FakeUpstream::spawn_with_models_body(
        r#"{"object":"list","data":[{"id":"healthy-tail","object":"model"}]}"#,
    )
    .await;
    let extra_config = format!(
        r#"
[[upstreams]]
name = "first"
base_url = "{first_base_url}"
match_models = ["first-model"]

[[upstreams]]
name = "second"
base_url = "{second_primary_base_url}"
match_models = ["second-model"]
request_timeout_ms = 400
health_probe_interval_ms = 200
health_probe_timeout_ms = 20
health_probe_max_wait_ms = 400

[[upstreams.endpoints]]
base_url = "{second_primary_base_url}"
priority = "primary"

[[upstreams.endpoints]]
base_url = "{slow_middle_base_url}"
priority = "failover"

[[upstreams.endpoints]]
base_url = "{healthy_tail_base_url}"
priority = "failover"

[[listeners]]
name = "tail-budget-models"
bind_host = "127.0.0.1"
port = 18019
allowed_upstreams = ["first", "second"]
"#,
        first_base_url = first.base_url,
        slow_middle_base_url = slow_middle.base_url,
        healthy_tail_base_url = healthy_tail.base_url,
    );
    let proxy = spawn_observed_failover_proxy(&first.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "tail-budget-models");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models&budget=tail"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("aggregate model response should drain");
    primary_probe_seen
        .await
        .expect("second profile primary should pass readiness before disconnecting");
    assert_eq!(
        first.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models&budget=tail"
    );
    assert_eq!(slow_middle.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        slow_middle.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models&budget=tail"
    );
    let mut tail_paths = Vec::new();
    for _ in 0..3 {
        let Some(request) = healthy_tail.recv_within(Duration::from_millis(500)).await else {
            break;
        };
        tail_paths.push(request.path_and_query);
        if tail_paths
            .iter()
            .any(|path| path == "/v1/models?test=distinct-multi-upstream-models&budget=tail")
        {
            break;
        }
    }
    assert!(
        tail_paths.iter().any(|path| path == "/v1/models"),
        "the final healthy endpoint must retain budget for its readiness probe; observed {tail_paths:?}"
    );
    assert!(
        tail_paths
            .iter()
            .any(|path| path == "/v1/models?test=distinct-multi-upstream-models&budget=tail"),
        "the final healthy endpoint must retain budget for the models request; observed {tail_paths:?}"
    );
}

#[tokio::test]
async fn model_discovery_round_robin_advances_once_per_request() {
    let mut first = FakeUpstream::spawn().await;
    let mut second = FakeUpstream::spawn().await;
    let extra_config = round_robin_models_profile_config(&first.base_url, &second.base_url);
    let proxy = ProxyFixture::spawn_with_admission_config(
        &first.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &extra_config,
    )
    .await;
    let listener = listener_config(&proxy, "round-robin-models");
    let state = proxy.state.for_listener(listener);

    for path in ["/v1/models?sequence=first", "/v1/models?sequence=second"] {
        let response = proxy_handler(State(state.clone()), empty_get_request(path)).await;
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("round-robin model response should drain");
    }

    let first_control_requests = drain_matching_requests(&mut first, "sequence=");
    let second_control_requests = drain_matching_requests(&mut second, "sequence=");
    assert_eq!(
        first_control_requests.len(),
        1,
        "the first endpoint should receive one physical models request"
    );
    assert_eq!(
        second_control_requests.len(),
        1,
        "the second endpoint should receive one physical models request"
    );
    assert_ne!(first_control_requests, second_control_requests);
}

fn drain_matching_requests(upstream: &mut FakeUpstream, marker: &str) -> Vec<String> {
    let mut paths = Vec::new();
    while let Ok(request) = upstream.receiver.try_recv() {
        if request.path_and_query.contains(marker) {
            paths.push(request.path_and_query);
        }
    }
    paths
}

fn round_robin_models_profile_config(first_base_url: &str, second_base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "round-robin-models"
base_url = "{first_base_url}"
match_models = ["round-robin-model"]
endpoint_selection = "round_robin"

[[upstreams.endpoints]]
base_url = "{first_base_url}"
priority = "primary"

[[upstreams.endpoints]]
base_url = "{second_base_url}"
priority = "failover"

[[listeners]]
name = "round-robin-models"
bind_host = "127.0.0.1"
port = 18017
allowed_upstreams = ["round-robin-models"]
"#
    )
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
