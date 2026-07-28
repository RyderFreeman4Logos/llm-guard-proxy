#![cfg(feature = "guard")]

use super::*;

#[tokio::test]
async fn upstream_model_rewrites_request_and_response_model_names() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
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
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("rewritten model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(json["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        json["model"], "alias-chat",
        "client-facing response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["model"], "aeon-ultimate",
        "upstream must receive the configured upstream_model rewrite"
    );
}

#[tokio::test]
async fn absent_upstream_model_keeps_request_model_passthrough() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "passthrough-profile"
base_url = "{}"
match_models = ["alias-chat"]
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
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("passthrough model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["model"], "alias-chat",
        "without upstream_model the client model name must pass through"
    );
}
