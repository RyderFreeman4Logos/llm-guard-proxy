use super::*;

#[tokio::test]
async fn forced_models_listener_queries_and_returns_only_its_forced_profile() {
    let mut alpha =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut beta =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &alpha.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "alpha"
base_url = "{0}"
match_models = ["chat-model"]

[[upstreams]]
name = "beta"
base_url = "{1}"
match_models = ["beta-model"]

[[listeners]]
name = "forced-alpha"
bind_host = "127.0.0.1"
port = 18006
allowed_upstreams = ["alpha", "beta"]
upstream_profile = "alpha"
"#,
            alpha.base_url, beta.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "forced-alpha");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models response should read");
    let models: serde_json::Value =
        serde_json::from_slice(&body).expect("models response should be JSON");
    let model_ids = models["data"]
        .as_array()
        .expect("models response should contain data")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be a string"))
        .collect::<Vec<_>>();
    assert_eq!(model_ids, vec!["chat-model"]);
    assert_eq!(
        alpha.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert!(
        beta.recv_within(Duration::from_millis(100)).await.is_none(),
        "forced listener must not query another allowed profile"
    );

    let (request_metadata, _) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["listener_restricted"], "true");
}

#[tokio::test]
async fn profile_retry_ladder_native_schema_rejects_conflicting_controls_before_upstream_io() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[thinking]
default_injection_schema = "canonical"

[[retry.ladder]]
name = "global-canonical"
default_injection_schema = "canonical"

[[upstreams]]
name = "native-profile"
base_url = "{}"
match_models = ["profile-model"]

[upstreams.thinking]
default_injection_schema = "canonical"

[[upstreams.retry.ladder]]
name = "profile-native"
default_injection_schema = "vllm_native"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"profile-model","messages":[{"role":"user","content":"private-prompt"}],"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .send()
        .await
        .expect("conflicting request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["code"], "conflicting_thinking_controls");
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn profile_retry_ladder_canonical_schema_allows_conflicting_controls_despite_global_native_schema()
 {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[thinking]
default_injection_schema = "vllm_native"

[[retry.ladder]]
name = "global-native"
default_injection_schema = "vllm_native"

[[upstreams]]
name = "canonical-profile"
base_url = "{}"
match_models = ["profile-model"]

[upstreams.thinking]
default_injection_schema = "canonical"

[[upstreams.retry.ladder]]
name = "profile-canonical"
default_injection_schema = "canonical"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"profile-model","messages":[{"role":"user","content":"private-prompt"}],"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .send()
        .await
        .expect("canonical profile request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("response should be readable");
    assert_eq!(
        fake.recv_next().await.path_and_query,
        "/v1/chat/completions"
    );
}

#[test]
fn repeat_input_cache_keeps_long_window_records_when_short_window_profile_observes() {
    let repeat_inputs = RepeatInputCache::default();
    let fingerprint = "siphash64:interleaved-windows";

    assert_eq!(
        repeat_inputs.observe("alpha", fingerprint, 1_000, 120, 1),
        RepeatInputObservation::default()
    );
    assert_eq!(
        repeat_inputs.observe("beta", "siphash64:beta", 3_000, 1, 1),
        RepeatInputObservation::default()
    );
    assert_eq!(
        repeat_inputs.observe("alpha", fingerprint, 4_000, 120, 1),
        RepeatInputObservation {
            repeated: true,
            prior_count: 1,
        }
    );
}

#[test]
fn repeat_input_cache_expires_entry_against_hot_reloaded_shorter_window() {
    let repeat_inputs = RepeatInputCache::default();
    let fingerprint = "siphash64:hot-reloaded-window";

    assert_eq!(
        repeat_inputs.observe("alpha", fingerprint, 1_000, 120, 1),
        RepeatInputObservation::default()
    );
    assert_eq!(
        repeat_inputs.observe("alpha", fingerprint, 2_001, 1, 1),
        RepeatInputObservation::default(),
        "an entry older than the hot-reloaded one-second window must be fresh"
    );
}
