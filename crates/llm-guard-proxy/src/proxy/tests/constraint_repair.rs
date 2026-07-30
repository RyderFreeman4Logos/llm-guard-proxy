use super::*;

const ACROSTIC_PROMPT: &str = "Write a 5-line poem that is an acrostic spelling STORM down the first letters of the lines (line 1 begins with S, then T, O, R, M). EVERY line must contain EXACTLY five words. The poem must include the word 'thunder' at least once. Do NOT use any semicolons anywhere in the poem. Output only the five lines, one line each, with no extra text.";
const LIPOGRAM_PROMPT: &str = "Write a 6-line poem with two simultaneous constraints. First, anaphora: every line must begin with the word 'Never'. Second, lipogram: the entire poem must NOT contain the letter 's' anywhere (neither uppercase nor lowercase). The poem must contain the word 'moon'. Output only the six lines, one per line, with no title and no commentary.";
const MULTI_CHOICE_PROMPT: &str = "Write exactly 2 lines.";

#[tokio::test]
async fn constraint_repair_retries_invalid_acrostic_with_max_thinking() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = constraint_repair_proxy(&fake.base_url).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=constraint-repair-acrostic",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(chat_request(ACROSTIC_PROMPT))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Storm thunder shakes distant cedars\nTrees bend beneath silver rain\nOceans roar over dark cliffs\nRavens wheel above wet fields\nMist blankets silent village paths"
    );

    let first_attempt = fake.recv_next().await;
    let repair_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&repair_attempt.body), Some(32_768));
    assert!(!body_contains_text(
        &first_attempt.body,
        "llm-guard-proxy constraint-repair retry hint"
    ));
    assert!(body_contains_text(
        &repair_attempt.body,
        "llm-guard-proxy constraint-repair retry hint"
    ));
    assert!(body_contains_text(
        &repair_attempt.body,
        "Choice 0 violates:"
    ));
    assert!(body_contains_text(
        &repair_attempt.body,
        "Re-read the original user message for literal targets."
    ));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("constraint_violation")
    );
    assert_eq!(
        attempts[1].response_metadata["retry_previous_reason"],
        "previous_constraint_violation"
    );
    assert_eq!(
        attempts[1].response_metadata["constraint_repair_used"],
        "true"
    );
}

#[tokio::test]
async fn constraint_repair_retries_viable_lipogram_with_explicit_feedback() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = constraint_repair_proxy(&fake.base_url).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=constraint-repair-lipogram",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(chat_request(LIPOGRAM_PROMPT))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Never moon amid fog\nNever dim moon afloat\nNever calm moon hum\nNever moon on rim\nNever mild moon aglow\nNever moon in air"
    );

    let _first_attempt = fake.recv_next().await;
    let repair_attempt = fake.recv_next().await;
    assert!(body_contains_text(
        &repair_attempt.body,
        "prohibited letter"
    ));
}

#[tokio::test]
async fn constraint_repair_retries_when_a_later_choice_violates() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = constraint_repair_proxy(&fake.base_url).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=constraint-repair-multi-choice-valid-first",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(multi_choice_chat_request(MULTI_CHOICE_PROMPT))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let final_json = shielded_final_json(response).await;
    assert_eq!(final_json["choices"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        final_json["choices"][1]["message"]["content"],
        "Third corrected line\nFourth corrected line"
    );

    let first_attempt = fake.recv_next().await;
    let repair_attempt = fake.recv_next().await;
    assert!(!body_contains_text(
        &first_attempt.body,
        "llm-guard-proxy constraint-repair retry hint"
    ));
    assert!(body_contains_text(
        &repair_attempt.body,
        "Choice 1 violates: answer must contain exactly two non-empty lines"
    ));
}

#[tokio::test]
async fn constraint_repair_retries_when_the_first_choice_violates() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = constraint_repair_proxy(&fake.base_url).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=constraint-repair-multi-choice-invalid-first",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(multi_choice_chat_request(MULTI_CHOICE_PROMPT))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "First corrected line\nSecond corrected line"
    );

    let _first_attempt = fake.recv_next().await;
    let repair_attempt = fake.recv_next().await;
    assert!(body_contains_text(
        &repair_attempt.body,
        "Choice 0 violates: answer must contain exactly two non-empty lines"
    ));
}

#[tokio::test]
async fn valid_acrostic_does_not_add_a_constraint_repair_retry() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = constraint_repair_proxy(&fake.base_url).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=constraint-repair-acrostic-valid",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(chat_request(ACROSTIC_PROMPT))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let first_attempt = fake.recv_next().await;
    assert!(!body_contains_text(
        &first_attempt.body,
        "llm-guard-proxy constraint-repair retry hint"
    ));
    assert_eq!(read_attempt_chain_rows(&proxy.sqlite_path).len(), 1);
}

async fn constraint_repair_proxy(upstream_url: &str) -> ProxyFixture {
    ProxyFixture::spawn_with_options(
        upstream_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"

[retry]
max_attempts = 2
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
"#,
    )
    .await
}

fn chat_request(prompt: &str) -> String {
    serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": prompt}],
    })
    .to_string()
}

fn multi_choice_chat_request(prompt: &str) -> String {
    serde_json::json!({
        "model": "test-chat",
        "n": 2,
        "messages": [{"role": "user", "content": prompt}],
    })
    .to_string()
}
