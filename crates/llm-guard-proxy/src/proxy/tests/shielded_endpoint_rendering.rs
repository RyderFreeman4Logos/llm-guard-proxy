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
