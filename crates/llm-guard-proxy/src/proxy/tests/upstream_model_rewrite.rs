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
async fn upstream_model_rewrites_shielded_streaming_response_model_name() {
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
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("rewritten streaming model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .bytes()
        .await
        .expect("rewritten streaming response should be readable");
    let body = std::str::from_utf8(&body).expect("rewritten streaming response should be UTF-8");
    let chunks = openai_sse_json_chunks(body);
    assert!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.get("model"))
            .all(|model| model == "alias-chat"),
        "client-facing SSE response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "aeon-ultimate");
}

#[tokio::test]
async fn upstream_model_rewrites_final_direct_relay_response_model_name() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000

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
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("direct relay request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .bytes()
        .await
        .expect("direct relay response should be readable");
    let body = std::str::from_utf8(&body).expect("direct relay response should be UTF-8");
    let chunks = openai_sse_json_chunks(body);
    assert!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.get("model"))
            .all(|model| model == "alias-chat"),
        "direct relay must not expose the configured upstream model"
    );

    for _ in 0..3 {
        let observed = fake.recv_next().await;
        let observed_body: serde_json::Value =
            serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
        assert_eq!(observed_body["model"], "aeon-ultimate");
    }
}

#[tokio::test]
async fn upstream_model_rewrites_generic_response_model_name() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-completion"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"alias-completion","prompt":"ping"}"#)
        .send()
        .await
        .expect("rewritten completion request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response_json(response).await["model"],
        "alias-completion",
        "generic client-facing response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "aeon-ultimate");
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

#[tokio::test]
async fn upstream_model_rewrite_sse_buffer_overflows_with_controlled_error() {
    // Feed the rewriter a stream of bytes that exceeds the frame cap without
    // any SSE delimiter. The body must terminate with a controlled error
    // rather than buffering indefinitely.
    let cap = SSE_REWRITE_FRAME_BYTE_LIMIT;
    let oversized: Vec<Result<Bytes, std::io::Error>> = vec![
        Ok(Bytes::from(vec![b'A'; cap])),
        Ok(Bytes::from(vec![b'B'; cap])),
    ];
    let input = futures_util::stream::iter(oversized);
    let mut body = ResponseModelRewriteBody::new(
        input,
        ResponseModelRewriteMode::OpenAiSse,
        String::from("alias-chat"),
    );

    let mut saw_overflow = false;
    while let Some(result) = body.next().await {
        if let Err(ResponseModelRewriteError::FrameOverflow { .. }) = result {
            saw_overflow = true;
        }
    }
    assert!(
        saw_overflow,
        "stream must surface a controlled FrameOverflow error when the SSE \
         frame exceeds the byte cap without a delimiter"
    );
}

#[tokio::test]
async fn upstream_model_rewrite_sse_buffer_releases_after_overflow() {
    // After an overflow, the buffered bytes must be released (not retained
    // for future scans) and the stream must terminate quickly.
    let cap = SSE_REWRITE_FRAME_BYTE_LIMIT;
    let oversized: Vec<Result<Bytes, std::io::Error>> = vec![Ok(Bytes::from(vec![b'X'; cap + 1]))];
    let input = futures_util::stream::iter(oversized);
    let mut body = ResponseModelRewriteBody::new(
        input,
        ResponseModelRewriteMode::OpenAiSse,
        String::from("alias-chat"),
    );

    let mut count = 0;
    while let Some(result) = body.next().await {
        let _ = result;
        count += 1;
        assert!(
            count <= 10,
            "stream should terminate after bounded overflow, not loop"
        );
    }
    // The overflow should emit exactly one error then None — at most 2 polls.
    assert!(
        count <= 2,
        "overflowed stream must terminate quickly, got {count} items"
    );
}

#[tokio::test]
async fn configured_upstream_model_missing_from_models_list_does_not_synthesize_alias() {
    // upstream_model = "aeon-ultimate" but the upstream /v1/models response
    // only contains unrelated models. The proxy must NOT synthesize an alias
    // from the first unrelated model's metadata. We verify by checking that
    // no alias record appears in the response, and specifically that no record
    // copies "unrelated-model"'s metadata with the alias id.
    //
    // We use two upstream profiles so the default listener (no upstream_profile)
    // falls through to allowed_upstreams / first-profile routing, which keeps
    // the original model visible. The key assertion: no "alias-chat" record
    // synthesized from unrelated metadata.
    let models_body = r#"{"object":"list","data":[{"id":"unrelated-model","object":"model","max_model_len":128000,"owned_by":"vllm"}]}"#;
    let fake = FakeUpstream::spawn_with_models_body(models_body).await;
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
        .get(format!("{}/v1/models", proxy.base_url))
        .send()
        .await
        .expect("models request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .text()
        .await
        .expect("models body should be readable");
    let value: serde_json::Value = serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("models body should parse as JSON: {error}; body={body}"));
    let models = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .expect("models body should have a data array");

    let model_ids: Vec<&str> = models
        .iter()
        .filter_map(|model| model.get("id").and_then(serde_json::Value::as_str))
        .collect();
    assert!(
        !model_ids.contains(&"alias-chat"),
        "must not synthesize alias-chat from an unrelated model when the configured \
         upstream_model is missing from the upstream models list; got ids: {model_ids:?}"
    );
    // Additionally verify no synthesized alias records exist at all — if the
    // fix worked, the rewriter should not have copied any metadata.
    assert!(
        !models.iter().any(|model| {
            model
                .get("llm_guard_proxy_alias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        }),
        "no alias records should be synthesized when upstream_model is missing upstream"
    );
}
