use super::*;

#[tokio::test]
async fn aggregate_models_preserves_first_profile_attempt_when_later_profile_is_cooling_down() {
    let mut first = FakeUpstream::spawn().await;
    let mut cooling = FakeUpstream::spawn_with_rerank_status(StatusCode::SERVICE_UNAVAILABLE).await;
    let extra_config = aggregate_models_cooldown_config(&first.base_url, &cooling.base_url);
    let proxy = spawn_observed_failover_proxy(&first.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "attempt-preservation-cooldown");
    let state = proxy.state.for_listener(listener);

    let prime_response = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/rerank?test=prime-later-cooldown",
            br#"{"model":"later-model","query":"q","documents":["d"]}"#,
        ),
    )
    .await;
    assert_eq!(prime_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    to_bytes(prime_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("cooldown-prime response should drain");
    assert_eq!(cooling.recv_next().await.path_and_query, "/v1/models");
    assert_eq!(
        cooling.recv_next().await.path_and_query,
        "/v1/rerank?test=prime-later-cooldown"
    );
    let attempts_before_aggregate = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts_before_aggregate.len(), 1);

    let response = proxy_handler(
        State(state),
        empty_get_request("/v1/models?test=attempts-survive-later-cooldown"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("aggregate cooldown response should drain");

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(
        attempts.len(),
        attempts_before_aggregate.len() + 1,
        "the completed first-profile models attempt must survive later selection failure"
    );
    let aggregate_attempt = attempts.last().expect("aggregate attempt should persist");
    assert_eq!(aggregate_attempt.status, "succeeded");
    assert_eq!(
        aggregate_attempt.request_metadata["upstream_profile"],
        "first"
    );
    assert_eq!(
        first.recv_next().await.path_and_query,
        "/v1/models?test=attempts-survive-later-cooldown"
    );
    assert!(
        cooling
            .recv_within(Duration::from_millis(30))
            .await
            .is_none(),
        "cooling endpoint must not receive the aggregate models request"
    );
}

#[tokio::test]
async fn aggregate_models_preserves_first_profile_attempt_when_later_profile_is_incompatible() {
    let mut first = FakeUpstream::spawn().await;
    let mut incompatible = FakeUpstream::spawn().await;
    let extra_config =
        aggregate_models_incompatible_config(&first.base_url, &incompatible.base_url);
    let proxy = spawn_observed_failover_proxy(&first.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "attempt-preservation-incompatible");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=attempts-survive-incompatible-profile"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("aggregate incompatible response should drain");

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(
        attempts.len(),
        1,
        "the completed first-profile models attempt must survive incompatible later selection"
    );
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[0].request_metadata["upstream_profile"], "first");
    assert_eq!(
        first.recv_next().await.path_and_query,
        "/v1/models?test=attempts-survive-incompatible-profile"
    );
    assert!(
        incompatible
            .recv_within(Duration::from_millis(30))
            .await
            .is_none(),
        "protocol-incompatible endpoint must not receive OpenAI model discovery"
    );
}

fn aggregate_models_cooldown_config(first_base_url: &str, later_base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "first"
base_url = "{first_base_url}"
match_models = ["first-model"]

[[upstreams]]
name = "later"
base_url = "{later_base_url}"
match_models = ["later-model"]
request_timeout_ms = 80
health_probe_interval_ms = 5000
health_probe_timeout_ms = 20
health_probe_max_wait_ms = 5000

[[upstreams.endpoints]]
base_url = "{later_base_url}"
priority = "primary"
protocol = "openai"

[[listeners]]
name = "attempt-preservation-cooldown"
bind_host = "127.0.0.1"
port = 18020
allowed_upstreams = ["first", "later"]
"#
    )
}

fn aggregate_models_incompatible_config(
    first_base_url: &str,
    incompatible_base_url: &str,
) -> String {
    format!(
        r#"
[[upstreams]]
name = "first"
base_url = "{first_base_url}"
match_models = ["first-model"]

[[upstreams]]
name = "incompatible"
base_url = "{incompatible_base_url}"
match_models = ["incompatible-model"]

[[upstreams.endpoints]]
base_url = "{incompatible_base_url}"
priority = "primary"
protocol = "deepinfra_qwen3_rerank"
model = "Qwen/Qwen3-Reranker-8B"
model_revision = "5fa94080caafeaa45a15d11f969d7978e087a3db"
api_key_env = "PATH"

[[listeners]]
name = "attempt-preservation-incompatible"
bind_host = "127.0.0.1"
port = 18021
allowed_upstreams = ["first", "incompatible"]
"#
    )
}
