use super::*;

#[tokio::test]
async fn sglang_first_evict_priority_coerces_yaml_string() {
    let numeric = forward_priority_hint("sglang", "-1000").await;
    let observed = forward_priority_hint("sglang", r#""-1000""#).await;

    assert_eq!(numeric["priority"], -1000);
    assert_eq!(observed["priority"], -1000);
}

#[tokio::test]
async fn vllm_priority_hint_uses_openai_scheduling_priority() {
    let observed = forward_priority_hint("vllm", r#""-1000""#).await;

    assert_eq!(observed["priority"], -1000);
}

#[tokio::test]
async fn configured_priority_engine_preserves_request_without_hint() {
    let mut upstream = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &priority_engine_profile_config(&upstream.base_url, "sglang"),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"no priority"}],"temperature":0.7}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response
        .bytes()
        .await
        .expect("response body should be readable");

    assert_eq!(upstream.recv_next().await.body, body);
}

async fn forward_priority_hint(engine: &str, priority: &str) -> serde_json::Value {
    let mut upstream = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &priority_engine_profile_config(&upstream.base_url, engine),
    )
    .await;
    let body = format!(
        r#"{{"model":"test-chat","messages":[{{"role":"user","content":"priority"}}],"priority":{priority}}}"#
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response
        .bytes()
        .await
        .expect("response body should be readable");

    serde_json::from_slice(&upstream.recv_next().await.body)
        .expect("upstream request should be JSON")
}

fn priority_engine_profile_config(base_url: &str, engine: &str) -> String {
    format!(
        r#"
[shielding]
enabled = false

[[upstreams]]
name = "first-evict"
base_url = "{base_url}"
match_models = ["test-chat"]
cache_priority_engine = "{engine}"
"#,
    )
}
