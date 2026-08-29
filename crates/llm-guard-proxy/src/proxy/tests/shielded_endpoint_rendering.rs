use super::*;

#[tokio::test]
async fn paired_shadow_requests_apply_configured_endpoint_model_alias() {
    let mut upstream = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_endpoint_model_proxy(
        &upstream.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 2000

[evidence.shadow.paired_comparison]
enabled = true
variants = ["no-thinking"]
sample_rate = 1.0
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=paired-shadow",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"caller-chat","messages":[{"role":"user","content":"compare aliases"}]}"#)
        .send()
        .await
        .expect("paired shadow request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Hello"
    );
    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        1,
    )
    .await;

    let requests = recv_n_upstream_requests(&mut upstream, 3).await;
    let chat_requests = requests
        .iter()
        .filter(|request| request.path_and_query.starts_with("/v1/chat/completions"))
        .collect::<Vec<_>>();
    assert_eq!(
        chat_requests.len(),
        2,
        "primary and paired shadow must both be sent"
    );
    for request in chat_requests {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("rendered chat body should be JSON");
        assert_eq!(body["model"], "vendor-chat");
    }
}

#[tokio::test]
async fn paired_shadow_metadata_tracks_the_final_forced_wire_request() {
    let mut upstream = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_endpoint_model_proxy(
        &upstream.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 2000

[evidence.shadow.paired_comparison]
enabled = true
variants = ["no-thinking"]
sample_rate = 1.0

[[forced_model_alias_profiles]]
alias = "caller-chat"
upstream_model = "forced-canonical"
thinking_mode = "force_thinking"
thinking_budget = 256
output_cap = 128
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions?test=forced-paired-shadow", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"caller-chat","messages":[{"role":"user","content":"compare forced policy"}]}"#)
        .send()
        .await
        .expect("forced paired shadow request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("primary response should drain");
    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        1,
    )
    .await;

    let requests = recv_n_upstream_requests(&mut upstream, 3).await;
    for request in requests
        .iter()
        .filter(|request| request.path_and_query.starts_with("/v1/chat/completions"))
    {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("forced wire body should be JSON");
        assert_eq!(body["model"], "forced-canonical");
        assert_eq!(body["thinking_token_budget"], 256);
        assert_eq!(body["max_tokens"], 384);
    }

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let (thinking_mode, thinking_budget_tokens, thinking_max_tokens, request_metadata_json):
        (Option<String>, Option<u32>, Option<u32>, String) = connection
        .query_row(
            "SELECT thinking_mode, thinking_budget_tokens, thinking_max_tokens, request_metadata_json \
             FROM evidence_attempts WHERE role = 'shadow_continued'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("shadow evidence should exist");
    assert_eq!(thinking_mode.as_deref(), Some("force_thinking"));
    assert_eq!(thinking_budget_tokens, Some(256));
    assert_eq!(thinking_max_tokens, Some(384));
    let metadata: serde_json::Value =
        serde_json::from_str(&request_metadata_json).expect("shadow metadata should be JSON");
    assert_eq!(metadata["forced_alias"], "caller-chat");
    assert_eq!(metadata["forced_upstream_model"], "forced-canonical");
    assert_eq!(metadata["forced_answer_headroom"], "128");
    assert_eq!(metadata["forced_wire_total_cap"], "384");
}

#[tokio::test]
async fn shielded_raw_input_records_rendered_endpoint_model_alias() {
    let mut upstream = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_endpoint_model_proxy(
        &upstream.base_url,
        r"
enabled = true
include_raw_payloads = true
",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"caller-chat","messages":[{"role":"user","content":"record physical request"}]}"#)
        .send()
        .await
        .expect("shielded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Hello"
    );
    let _requests = recv_n_upstream_requests(&mut upstream, 2).await;
    wait_for_evidence_role_status_count(&proxy.evidence_sqlite_path, "primary", "accepted", 1)
        .await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let raw_input: Option<String> = connection
        .query_row(
            "SELECT raw_input FROM evidence_attempts WHERE role = 'primary'",
            [],
            |row| row.get(0),
        )
        .expect("primary evidence should exist");
    let raw_input = raw_input.expect("primary raw input should be captured");
    let raw_body: serde_json::Value =
        serde_json::from_str(&raw_input).expect("primary raw input should be rendered JSON");
    assert_eq!(raw_body["model"], "vendor-chat");
}

async fn spawn_shielded_endpoint_model_proxy(
    upstream_base_url: &str,
    evidence_config: &str,
) -> ProxyFixture {
    let endpoint_profile = format!(
        r#"
[[profile]]
model = "caller-chat"
request_timeout_ms = 1000
health_probe_interval = "200ms"
health_probe_timeout = "20ms"
health_probe_max_wait = "400ms"

[[profile.upstream]]
base_url = "{upstream_base_url}"
priority = "primary"
protocol = "openai"
model = "vendor-chat"

[shielding]
enabled = true
"#
    );
    ProxyFixture::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
        upstream_base_url,
        observability_enabled: true,
        max_in_flight_requests: AppConfig::default().server.max_in_flight_requests,
        server_config: "",
        metadata_config: "",
        observability_config: "",
        evidence_config,
        extra_config: &endpoint_profile,
    })
    .await
}
