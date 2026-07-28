use super::*;

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn bounded_cot_salvage_uses_configured_note_limit_and_thinking_budget() {
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
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-with-prelude-then-success",
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

    let salvage_request: serde_json::Value =
        serde_json::from_slice(&salvage_attempt.body).expect("salvage body should be JSON");
    assert_eq!(salvage_request["max_tokens"], 50_000);

    let salvage_body = String::from_utf8_lossy(&salvage_attempt.body);
    assert!(salvage_body.contains("llm-guard-proxy CoT salvage retry hint"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].response_metadata["loop_channel"], "reasoning");
    assert_eq!(attempts[0].response_metadata["loop_detected"], "true");
    assert_eq!(attempts[0].response_metadata["loop_detector_class"], "line");
    assert_eq!(attempts[0].response_metadata["ladder_rung"], "max-thinking");
    assert_eq!(attempts[0].response_metadata["salvage_used"], "false");
    assert_eq!(attempts[0].response_metadata["thinking_budget"], "32768");
    assert_eq!(attempts[0].response_metadata["max_tokens"], "50000");
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert!(
        attempts[1].response_metadata.get("loop_detected").is_none()
            || attempts[1].response_metadata["loop_detected"] == "false",
        "non-loop attempt should not report loop_detected=true"
    );
    assert!(
        attempts[1]
            .response_metadata
            .get("loop_detector_class")
            .is_none()
            || attempts[1].response_metadata["loop_detector_class"] == "none",
    );
    assert_eq!(
        attempts[1].response_metadata["ladder_rung"],
        "bounded-thinking"
    );
    assert_eq!(attempts[1].response_metadata["salvage_used"], "true");
    assert_eq!(attempts[1].response_metadata["thinking_budget"], "2048");
    assert_eq!(attempts[1].response_metadata["max_tokens"], "50000");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_source_attempt_number"],
        "1"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_pre_loop_bytes_retained"],
        "16"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_thinking_mode"],
        "bounded_thinking"
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_metric_type(
        &metrics,
        "llm_guard_proxy_current_retained_loop_guard_attempts",
        "gauge",
    );
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_current_retained_loop_guard_attempts",
            &[
                ("loop_detected", "true"),
                ("loop_detector_class", "line"),
                ("ladder_rung", "max-thinking"),
                ("salvage_used", "false"),
                ("thinking_budget", "32768"),
                ("max_tokens", "50000"),
            ],
        ),
        1
    );
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_current_retained_loop_guard_attempts",
            &[
                ("loop_detected", "false"),
                ("loop_detector_class", "none"),
                ("ladder_rung", "bounded-thinking"),
                ("salvage_used", "true"),
                ("thinking_budget", "2048"),
                ("max_tokens", "50000"),
            ],
        ),
        1
    );
    assert!(!metrics.contains("llm-guard-proxy CoT salvage retry hint"));
}

#[tokio::test]
async fn bounded_cot_salvage_keeps_pre_loop_reasoning_and_discards_repeated_tail() {
    const PRE_LOOP_REASONING: &str = "derive the invariant before answering\n";
    const LOOPING_TAIL: &str = "repeat the broken branch\n";

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
cot_salvage_prefix_max_bytes = 1024

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "salvage-answer"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-with-prelude-then-success",
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

    let _first_attempt = fake.recv_next().await;
    let salvage_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&salvage_attempt.body), Some(1_024));
    let salvage_request: serde_json::Value =
        serde_json::from_slice(&salvage_attempt.body).expect("salvage body should be JSON");
    let salvage_messages = salvage_request["messages"]
        .as_array()
        .expect("salvage request should include messages");
    assert_eq!(salvage_messages[0]["role"], "system");
    let salvage_system_content = salvage_messages[0]["content"]
        .as_str()
        .expect("salvage system message should contain text");
    assert!(salvage_system_content.contains(PRE_LOOP_REASONING));
    assert!(!salvage_system_content.contains(LOOPING_TAIL));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].response_metadata["loop_detected"], "true");
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_pre_loop_bytes_retained"],
        PRE_LOOP_REASONING.len().to_string()
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_post_loop_bytes_discarded"],
        (LOOPING_TAIL.len() * 4).to_string()
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_boundary"],
        "repeated_line"
    );
}

#[tokio::test]
async fn bounded_cot_salvage_uses_pre_abort_reasoning_when_repeated_line_tail_is_unavailable() {
    const PRE_LOOP_REASONING: &str = "derive the invariant before answering\n";
    const LOOPING_TAIL: &str = "repeat the broken branch\n";

    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "bounded_answer_from_cot"
output_repeated_line_threshold = 2
cot_salvage_prefix_max_bytes = 1024

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "salvage-answer"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-with-prelude-then-success",
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

    let _first_attempt = fake.recv_next().await;
    let salvage_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&salvage_attempt.body), Some(1_024));
    let salvage_request: serde_json::Value =
        serde_json::from_slice(&salvage_attempt.body).expect("salvage body should be JSON");
    let salvage_messages = salvage_request["messages"]
        .as_array()
        .expect("salvage request should include messages");
    let salvage_system_content = salvage_messages[0]["content"]
        .as_str()
        .expect("salvage system message should contain text");
    assert!(salvage_system_content.contains(PRE_LOOP_REASONING));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_boundary"],
        "repeated_line"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_post_loop_bytes_discarded"],
        (LOOPING_TAIL.len() * 2).to_string()
    );
}

#[tokio::test]
async fn bounded_cot_salvage_keeps_pre_loop_reasoning_when_the_boundary_is_in_the_abort_fragment() {
    const PRE_LOOP_REASONING: &str = "derive the invariant before answering\n";
    const LOOPING_TAIL: &str = "repeat the broken branch\n";

    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "bounded_answer_from_cot"
output_repeated_line_threshold = 2
cot_salvage_prefix_max_bytes = 1024

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "salvage-answer"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-with-intra-fragment-tail-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _first_attempt = fake.recv_next().await;
    let salvage_attempt = fake.recv_next().await;
    let salvage_request: serde_json::Value =
        serde_json::from_slice(&salvage_attempt.body).expect("salvage body should be JSON");
    let salvage_system_content = salvage_request["messages"]
        .as_array()
        .expect("salvage request should include messages")[0]["content"]
        .as_str()
        .expect("salvage system message should contain text");
    assert!(salvage_system_content.contains(PRE_LOOP_REASONING));
    assert!(!salvage_system_content.contains(LOOPING_TAIL));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_boundary"],
        "repeated_line"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_post_loop_bytes_discarded"],
        (LOOPING_TAIL.len() * 2).to_string()
    );
}

#[tokio::test]
async fn bounded_cot_salvage_preserves_reasoning_before_the_detector_selected_repeated_line() {
    const EARLY_REPEAT: &str = "harmless repeated framing\n";
    const USEFUL_ONE: &str = "derive the first useful invariant\n";
    const USEFUL_TWO: &str = "apply the second useful invariant\n";
    const LOOPING_TAIL: &str = "actual repeated loop tail\n";

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
input_overlap_threshold_multiplier = 2
cot_salvage_prefix_max_bytes = 4096

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "salvage-answer"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-with-early-repeat-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"actual repeated loop tail\nactual repeated loop tail\n"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Hello"
    );

    let _first_attempt = fake.recv_next().await;
    let salvage_attempt = fake.recv_next().await;
    let salvage_request: serde_json::Value =
        serde_json::from_slice(&salvage_attempt.body).expect("salvage body should be JSON");
    let salvage_system_content = salvage_request["messages"]
        .as_array()
        .expect("salvage request should include messages")[0]["content"]
        .as_str()
        .expect("salvage system message should contain text");
    assert!(salvage_system_content.contains(EARLY_REPEAT));
    assert!(salvage_system_content.contains(USEFUL_ONE));
    assert!(salvage_system_content.contains(USEFUL_TWO));
    assert!(!salvage_system_content.contains(LOOPING_TAIL));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].response_metadata["loop_signal_0_feature_threshold"],
        "8"
    );
    assert_eq!(
        attempts[0].response_metadata["loop_signal_0_feature_input_overlap_applied"],
        "true"
    );
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_boundary"],
        "repeated_line"
    );
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_post_loop_bytes_discarded"],
        (LOOPING_TAIL.len() * 8).to_string()
    );
}

#[test]
fn evidence_features_keep_cot_salvage_pre_loop_metadata() {
    let metadata = BTreeMap::from([
        (
            String::from("cot_salvage_pre_loop_bytes_available"),
            String::from("82"),
        ),
        (
            String::from("cot_salvage_pre_loop_bytes_retained"),
            String::from("64"),
        ),
        (
            String::from("cot_salvage_pre_loop_bytes_capped"),
            String::from("18"),
        ),
        (
            String::from("cot_salvage_post_loop_bytes_discarded"),
            String::from("256"),
        ),
        (
            String::from("cot_salvage_boundary"),
            String::from("repeated_line"),
        ),
        (
            String::from("cot_salvage_reasoning_prefix_bytes"),
            String::from("stale"),
        ),
    ]);

    let features = evidence_detector_features(&metadata);

    assert_eq!(
        features,
        BTreeMap::from([
            (
                String::from("cot_salvage_pre_loop_bytes_available"),
                String::from("82"),
            ),
            (
                String::from("cot_salvage_pre_loop_bytes_retained"),
                String::from("64"),
            ),
            (
                String::from("cot_salvage_pre_loop_bytes_capped"),
                String::from("18"),
            ),
            (
                String::from("cot_salvage_post_loop_bytes_discarded"),
                String::from("256"),
            ),
            (
                String::from("cot_salvage_boundary"),
                String::from("repeated_line"),
            ),
        ])
    );
}
