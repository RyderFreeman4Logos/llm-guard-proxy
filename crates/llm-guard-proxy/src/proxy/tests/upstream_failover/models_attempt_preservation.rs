use super::*;
use std::sync::Arc;

use tokio::sync::Barrier;

#[tokio::test]
async fn aggregate_models_holds_credentialed_openai_recovery_lease_until_response_classification() {
    let first = FakeUpstream::spawn().await;
    let mut recovering = FakeUpstream::spawn_with_options(
        Some(r#"{"object":"list","data":[{"id":"recovered-model","object":"model"}]}"#),
        StatusCode::OK,
        "recovering-models",
        Some(Duration::from_millis(300)),
        None,
        Some(StatusCode::SERVICE_UNAVAILABLE),
        None,
    )
    .await;
    let extra_config =
        aggregate_models_credentialed_openai_recovery_config(&first.base_url, &recovering.base_url);
    let proxy = spawn_observed_failover_proxy(&first.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "credentialed-openai-recovery-models");
    let state = proxy.state.for_listener(listener);

    let prime = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/rerank",
            br#"{"model":"recovering-model","query":"prime","documents":["document"]}"#,
        ),
    )
    .await;
    assert_eq!(prime.status(), StatusCode::SERVICE_UNAVAILABLE);
    to_bytes(prime.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("priming response should drain");
    assert_eq!(recovering.recv_next().await.path_and_query, "/v1/rerank");

    sleep(Duration::from_millis(225)).await;
    let request_count = 4;
    let barrier = Arc::new(Barrier::new(request_count + 1));
    let mut requests = Vec::with_capacity(request_count);
    for _ in 0..request_count {
        let barrier = Arc::clone(&barrier);
        let state = state.clone();
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            let response = proxy_handler(
                State(state),
                empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
            )
            .await;
            to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
                .await
                .expect("concurrent models response should drain");
        }));
    }
    barrier.wait().await;

    assert_eq!(
        recovering
            .recv_within(Duration::from_secs(1))
            .await
            .expect("one credentialed OpenAI recovery request should be sent")
            .path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert!(
        recovering
            .recv_within(Duration::from_millis(100))
            .await
            .is_none(),
        "aggregate models must not send an overlapping credentialed OpenAI recovery trial"
    );
    for request in requests {
        timeout(Duration::from_secs(2), request)
            .await
            .expect("concurrent models request should complete")
            .expect("concurrent models task should join");
    }
}

#[tokio::test]
async fn aggregate_models_fetches_same_origin_profiles_with_distinct_credentials() {
    let mut shared = FakeUpstream::spawn().await;
    let extra_config = aggregate_models_credential_distinct_origin_config(&shared.base_url);
    let proxy = spawn_observed_failover_proxy(&shared.base_url, &extra_config).await;
    let listener = listener_config(&proxy, "credential-distinct-origin-models");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=credential-distinct-origin"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("credential-distinct aggregate response should drain");

    let expected_authorizations = ["PATH", "HOME"]
        .into_iter()
        .map(|name| {
            format!(
                "Bearer {}",
                std::env::var(name).expect("configured test credential environment should exist")
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut observed_authorizations = std::collections::BTreeSet::new();
    for _ in 0..2 {
        let request = shared
            .recv_within(Duration::from_millis(150))
            .await
            .expect("each credential-distinct profile must fetch its models");
        assert_eq!(
            request.path_and_query,
            "/v1/models?test=credential-distinct-origin"
        );
        observed_authorizations.insert(
            request
                .headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .expect("credentialed endpoint request should carry authorization")
                .to_owned(),
        );
    }
    assert_eq!(observed_authorizations, expected_authorizations);
}

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

fn aggregate_models_credentialed_openai_recovery_config(
    first_base_url: &str,
    recovering_base_url: &str,
) -> String {
    format!(
        r#"
[[upstreams]]
name = "first"
base_url = "{first_base_url}"
match_models = ["first-model"]

[[upstreams]]
name = "recovering"
base_url = "{recovering_base_url}"
match_models = ["recovering-model"]
request_timeout_ms = 1000
health_probe_interval_ms = 200
health_probe_timeout_ms = 20
health_probe_max_wait_ms = 1000

[[upstreams.endpoints]]
base_url = "{recovering_base_url}"
priority = "primary"
protocol = "openai"
api_key_env = "PATH"

[[listeners]]
name = "credentialed-openai-recovery-models"
bind_host = "127.0.0.1"
port = 18022
allowed_upstreams = ["first", "recovering"]
"#
    )
}

fn aggregate_models_credential_distinct_origin_config(shared_base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "path-credential"
base_url = "{shared_base_url}"
match_models = ["path-model"]

[[upstreams.endpoints]]
base_url = "{shared_base_url}"
priority = "primary"
protocol = "openai"
api_key_env = "PATH"

[[upstreams]]
name = "home-credential"
base_url = "{shared_base_url}"
match_models = ["home-model"]

[[upstreams.endpoints]]
base_url = "{shared_base_url}"
priority = "primary"
protocol = "openai"
api_key_env = "HOME"

[[listeners]]
name = "credential-distinct-origin-models"
bind_host = "127.0.0.1"
port = 18023
allowed_upstreams = ["path-credential", "home-credential"]
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
