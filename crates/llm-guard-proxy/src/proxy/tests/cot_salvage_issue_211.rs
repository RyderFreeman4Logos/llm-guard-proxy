use super::*;

#[tokio::test]
async fn bounded_cot_salvage_uses_configured_limits() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "bounded_answer_from_cot"
output_repeated_line_threshold = 4
cot_salvage_prefix_max_bytes = 16
cot_salvage_retry_thinking_budget = 2048

[retry]
max_attempts = 3
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Hello"
    );

    let first_attempt = fake.recv_next().await;
    let salvage_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&salvage_attempt.body), Some(2_048));

    let salvage_body = String::from_utf8_lossy(&salvage_attempt.body);
    assert!(salvage_body.contains("llm-guard-proxy CoT salvage retry hint"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].response_metadata["loop_channel"], "reasoning");
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_source_attempt_number"],
        "1"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_reasoning_prefix_bytes"],
        "16"
    );
}
