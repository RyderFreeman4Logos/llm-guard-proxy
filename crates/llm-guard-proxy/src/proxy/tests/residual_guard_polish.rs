use std::{sync::Arc, time::Duration};

use axum::http::{HeaderValue, StatusCode, header::RETRY_AFTER};
use serde_json::json;
use tokio::sync::Barrier;

use super::*;

const GENERIC_RECOVERY_CONFIG: &str = r#"
[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 3000
anti_loop_hint_enabled = false

[upstream.local_recovery]
enabled = true
restart_command = ["/bin/true"]
restart_timeout_ms = 1000
readiness_body = {"model":"test-chat","messages":[],"max_tokens":1}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 25
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#;

#[derive(Clone, Copy)]
enum GenericEntry {
    ShieldingDisabled,
    UnparseableChat,
    ShieldedStreamingDisabled,
}

impl GenericEntry {
    const fn name(self) -> &'static str {
        match self {
            Self::ShieldingDisabled => "shielding-disabled",
            Self::UnparseableChat => "unparseable-chat",
            Self::ShieldedStreamingDisabled => "shielded-streaming-disabled",
        }
    }

    fn config(self) -> String {
        let entry_config = match self {
            Self::ShieldingDisabled => "[shielding]\nenabled = false\n",
            Self::UnparseableChat => "",
            Self::ShieldedStreamingDisabled => "[retry]\nshielded_streaming_enabled = false\n",
        };
        format!("{GENERIC_RECOVERY_CONFIG}\n{entry_config}")
    }

    const fn body(self) -> &'static str {
        match self {
            Self::ShieldingDisabled => {
                r#"{"model":"test-chat","messages":[{"role":"user","content":"generic"}]}"#
            }
            Self::UnparseableChat => "{not-json",
            Self::ShieldedStreamingDisabled => {
                r#"{"model":"test-chat","stream":true,"messages":[{"role":"user","content":"generic"}]}"#
            }
        }
    }
}

async fn drain_profile_health_checks(fake: &mut FakeUpstream) {
    while let Some(observed) = fake.recv_within(Duration::from_millis(20)).await {
        assert_eq!(observed.path_and_query, "/v1/models");
    }
}

async fn recv_non_health_request(fake: &mut FakeUpstream) -> ObservedRequest {
    loop {
        let observed = fake.recv_next().await;
        if observed.path_and_query != "/v1/models" {
            return observed;
        }
    }
}

async fn assert_no_non_health_request(fake: &mut FakeUpstream) {
    let deadline = Instant::now() + Duration::from_millis(100);
    loop {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return;
        };
        let Some(observed) = fake.recv_within(remaining).await else {
            return;
        };
        assert_eq!(observed.path_and_query, "/v1/models");
    }
}

#[tokio::test]
async fn generic_entry_matrix_recovers_502_503_504_before_commit() {
    for entry in [
        GenericEntry::ShieldingDisabled,
        GenericEntry::UnparseableChat,
        GenericEntry::ShieldedStreamingDisabled,
    ] {
        for status in [502_u16, 503, 504] {
            let mut fake = FakeUpstream::spawn().await;
            let proxy = ProxyFixture::spawn_with_options(
                &fake.base_url,
                true,
                AppConfig::default().server.max_in_flight_requests,
                &entry.config(),
            )
            .await;
            let response = proxy
                .client
                .post(format!(
                    "{}/v1/chat/completions?test=generic-{status}-once-then-success&entry={}",
                    proxy.base_url,
                    entry.name()
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(entry.body())
                .send()
                .await
                .expect("generic recovery request should complete");

            assert_eq!(
                response.status(),
                StatusCode::OK,
                "entry={} status={status}",
                entry.name()
            );
            let body: serde_json::Value = response
                .json()
                .await
                .expect("generic replay response should be JSON");
            assert_eq!(body["choices"][0]["message"]["content"], "recovered");

            let first = fake.recv_next().await;
            let probe = fake.recv_next().await;
            let replay = fake.recv_next().await;
            assert_eq!(probe.path_and_query, "/v1/chat/completions");
            assert_eq!(first.path_and_query, replay.path_and_query);
            assert_eq!(first.body, replay.body);
            assert!(fake.recv_within(Duration::from_millis(50)).await.is_none());

            let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0].status, "retried");
            assert_eq!(attempts[0].retry_reason.as_deref(), Some("local_recovery"));
            assert_eq!(
                attempts[0].response_metadata["local_recovery_status"],
                "succeeded"
            );
            assert_eq!(attempts[1].status, "succeeded");
        }
    }
}

#[tokio::test]
async fn generic_transport_timeout_recovers_and_replays_once() {
    let mut fake = FakeUpstream::spawn().await;
    let config = GENERIC_RECOVERY_CONFIG
        .replace("request_deadline_ms = 3000", "request_deadline_ms = 2000")
        + "\n[upstream]\nrequest_timeout_ms = 50\n"
        + "\n[shielding]\nenabled = false\n";
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=generic-timeout-once-then-success",
            proxy.base_url
        ))
        .json(&json!({
            "model": "test-chat",
            "messages": [{"role": "user", "content": "timeout"}],
        }))
        .send()
        .await
        .expect("timeout recovery request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .json::<serde_json::Value>()
            .await
            .expect("replay body should be JSON")["choices"][0]["message"]["content"],
        "recovered"
    );

    let first = fake.recv_next().await;
    let probe = fake.recv_next().await;
    let replay = fake.recv_next().await;
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert_eq!(first.path_and_query, replay.path_and_query);
    assert!(fake.recv_within(Duration::from_millis(50)).await.is_none());
}

#[derive(Clone, Copy)]
enum GenericFirstBodyFailure {
    Timeout,
    Reset,
}

impl GenericFirstBodyFailure {
    const fn name(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Reset => "reset",
        }
    }
}

#[tokio::test]
async fn generic_entry_matrix_recovers_first_body_failure_before_commit() {
    for entry in [
        GenericEntry::ShieldingDisabled,
        GenericEntry::UnparseableChat,
        GenericEntry::ShieldedStreamingDisabled,
    ] {
        for failure in [
            GenericFirstBodyFailure::Timeout,
            GenericFirstBodyFailure::Reset,
        ] {
            let mut fake = FakeUpstream::spawn().await;
            let config = format!("{}\n[upstream]\nrequest_timeout_ms = 75\n", entry.config());
            let proxy = ProxyFixture::spawn_with_options(
                &fake.base_url,
                true,
                AppConfig::default().server.max_in_flight_requests,
                &config,
            )
            .await;
            let response = proxy
                .client
                .post(format!(
                    "{}/v1/chat/completions?test=generic-first-body-{}&entry={}",
                    proxy.base_url,
                    failure.name(),
                    entry.name()
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(entry.body())
                .send()
                .await
                .expect("generic first-body recovery request should complete");

            if response.status() != StatusCode::OK {
                let status = response.status();
                let body = response
                    .text()
                    .await
                    .expect("failed generic response should be readable");
                panic!(
                    "entry={} failure={} returned status={status} body={body}",
                    entry.name(),
                    failure.name()
                );
            }
            let body: serde_json::Value = response
                .json()
                .await
                .expect("generic recovery replay should return complete JSON");
            assert_eq!(body["choices"][0]["message"]["content"], "recovered");

            let first = recv_non_health_request(&mut fake).await;
            let readiness = recv_non_health_request(&mut fake).await;
            let replay = recv_non_health_request(&mut fake).await;
            assert_eq!(readiness.path_and_query, "/v1/chat/completions");
            assert_eq!(first.path_and_query, replay.path_and_query);
            assert_eq!(first.body, replay.body);
            assert_no_non_health_request(&mut fake).await;

            let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0].status, "retried");
            assert_eq!(attempts[0].retry_reason.as_deref(), Some("local_recovery"));
            assert_eq!(
                attempts[0].response_metadata["precommit_body_failure"],
                "true"
            );
            assert_eq!(attempts[1].status, "succeeded");
        }
    }
}

#[tokio::test]
async fn generic_entry_matrix_never_replays_after_first_nonempty_body_byte() {
    for entry in [
        GenericEntry::ShieldingDisabled,
        GenericEntry::UnparseableChat,
        GenericEntry::ShieldedStreamingDisabled,
    ] {
        let recovery_root = unique_test_dir(&format!("generic-post-byte-{}", entry.name()));
        fs::create_dir_all(&recovery_root).expect("recovery root should be created");
        let marker = recovery_root.join("restart-ran");
        let mut fake = FakeUpstream::spawn().await;
        let config = entry.config().replace(
            "restart_command = [\"/bin/true\"]",
            &format!(
                "restart_command = [\"/usr/bin/touch\", \"{}\"]",
                marker.display()
            ),
        );
        let proxy = ProxyFixture::spawn_with_options(
            &fake.base_url,
            true,
            AppConfig::default().server.max_in_flight_requests,
            &config,
        )
        .await;
        let response = proxy
            .client
            .post(format!(
                "{}/v1/chat/completions?test=generic-post-byte-error&entry={}",
                proxy.base_url,
                entry.name()
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(entry.body())
            .send()
            .await
            .expect("generic post-byte response should return headers");
        if response.status() == StatusCode::OK {
            let body_error = response
                .bytes()
                .await
                .expect_err("body reset after a committed byte must reach the client");
            assert!(!body_error.to_string().is_empty());
        } else {
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            response
                .bytes()
                .await
                .expect("buffered post-byte failure should return a proxy error");
        }

        let _first = recv_non_health_request(&mut fake).await;
        assert_no_non_health_request(&mut fake).await;
        assert!(!marker.exists());
        remove_dir_all(&recovery_root);
    }
}

#[tokio::test]
async fn generic_sibling_exhaustion_recovers_then_replays_from_primary() {
    let mut primary = FakeUpstream::spawn().await;
    let mut sibling = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[shielding]
enabled = false

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 3000

[[upstreams]]
name = "recovering-chat"
base_url = "{}"
match_models = ["test-chat"]
request_timeout_ms = 1000
endpoint_selection = "priority_failover"

[[upstreams.endpoints]]
base_url = "{}"
priority = "primary"
protocol = "openai"

[[upstreams.endpoints]]
base_url = "{}"
priority = "failover"
protocol = "openai"

[upstreams.metadata]
discovery_enabled = false

[upstreams.local_recovery]
enabled = true
restart_command = ["/bin/true"]
restart_timeout_ms = 1000
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 25
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        primary.base_url, primary.base_url, sibling.base_url
    );
    let proxy = ProxyFixture::spawn_with_options(
        &primary.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;
    drain_profile_health_checks(&mut primary).await;
    drain_profile_health_checks(&mut sibling).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=generic-503-once-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("sibling exhaustion recovery request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");

    let primary_initial = recv_non_health_request(&mut primary).await;
    let sibling_initial = recv_non_health_request(&mut sibling).await;
    let readiness = recv_non_health_request(&mut primary).await;
    let replay = recv_non_health_request(&mut primary).await;
    assert_eq!(
        primary_initial.path_and_query,
        sibling_initial.path_and_query
    );
    assert_eq!(primary_initial.path_and_query, replay.path_and_query);
    assert_eq!(readiness.path_and_query, "/v1/chat/completions");
    assert_no_non_health_request(&mut primary).await;
    assert_no_non_health_request(&mut sibling).await;

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("local_recovery"));
    assert_eq!(attempts[2].status, "succeeded");
}

#[tokio::test]
async fn generic_429_max_attempts_one_retries_once_without_failover_or_restart() {
    let recovery_root = unique_test_dir("generic-429-no-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let mut primary = FakeUpstream::spawn().await;
    let mut sibling = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[shielding]
enabled = false

[retry]
max_attempts = 1
max_retry_after_secs = 1

[[upstreams]]
name = "rate-limited-chat"
base_url = "{}"
match_models = ["test-chat"]
endpoint_selection = "priority_failover"

[[upstreams.endpoints]]
base_url = "{}"
priority = "primary"
protocol = "openai"

[[upstreams.endpoints]]
base_url = "{}"
priority = "failover"
protocol = "openai"

[upstreams.metadata]
discovery_enabled = false

[upstreams.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 1000
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 25
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        primary.base_url,
        primary.base_url,
        sibling.base_url,
        marker.display()
    );
    let proxy = ProxyFixture::spawn_with_options(
        &primary.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;
    drain_profile_health_checks(&mut primary).await;
    drain_profile_health_checks(&mut sibling).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=always-429",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("rate-limited request should complete");
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    response.bytes().await.expect("429 response should drain");
    let _primary_request = recv_non_health_request(&mut primary).await;
    let _rate_limit_retry = recv_non_health_request(&mut primary).await;
    assert_no_non_health_request(&mut primary).await;
    assert_no_non_health_request(&mut sibling).await;
    assert!(!marker.exists());
    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn generic_429_then_transport_timeout_recovers_and_replays_once() {
    let recovery_root = unique_test_dir("generic-429-transport-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let mut fake = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[shielding]
enabled = false

[upstream]
request_timeout_ms = 50

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 3000
max_retry_after_secs = 1

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 500
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 500
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        marker.display()
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;
    let client = proxy.client.clone();
    let request_url = format!(
        "{}/v1/chat/completions?test=generic-429-then-timeout-then-success",
        proxy.base_url
    );
    let request = tokio::spawn(async move {
        client
            .post(request_url)
            .json(&json!({"model":"test-chat","messages":[]}))
            .send()
            .await
    });

    let first = fake.recv_next().await;
    sleep(Duration::from_millis(100)).await;
    assert!(
        !marker.exists(),
        "the valid bounded 429 itself must not start local recovery"
    );
    let response = request
        .await
        .expect("request task should join")
        .expect("recovered request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");

    let timed_out = fake.recv_next().await;
    let readiness = fake.recv_next().await;
    let replay = fake.recv_next().await;
    assert_eq!(first.path_and_query, timed_out.path_and_query);
    assert_eq!(readiness.path_and_query, "/v1/chat/completions");
    assert_eq!(replay.path_and_query, first.path_and_query);
    assert!(marker.exists());
    assert!(fake.recv_within(Duration::from_millis(200)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].http_status, Some(429));
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("local_recovery"));
    assert_eq!(
        attempts[1].response_metadata["local_recovery_cause"],
        "transient_transport"
    );
    assert_eq!(
        attempts[1].response_metadata["local_recovery_request_attempts_used"],
        "1"
    );
    assert_eq!(attempts[2].status, "succeeded");
    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn shielded_max_attempts_one_preserves_recovery_after_rate_limit_retry() {
    let recovery_root = unique_test_dir("shielded-429-503-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let fake = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[heartbeat]
mode = "disabled"

[evidence]
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 4000
max_retry_after_secs = 1
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "original-policy"
thinking_mode = "force_thinking"
thinking_token_budget = 12345

[[retry.ladder]]
name = "ordinary-retry-policy"
thinking_mode = "force_disable"

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 500
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 500
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        marker.display()
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=shielded-429-then-503-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("shielded rate-limit recovery request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");
    assert!(marker.exists());

    assert_recovery_keeps_ordinary_rung_one(&proxy);
    remove_dir_all(&recovery_root);
}

fn multi_recovery_max_attempts_two_config(marker: &std::path::Path) -> String {
    format!(
        r#"
[heartbeat]
mode = "disabled"

[evidence]
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 6000
max_retry_after_secs = 1
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "original-policy"
thinking_mode = "force_thinking"
thinking_token_budget = 12345

[[retry.ladder]]
name = "ordinary-retry-policy"
thinking_mode = "force_disable"

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 500
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 500
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 2
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        marker.display()
    )
}

fn assert_two_recovery_replays_keep_ordinary_flat(proxy: &ProxyFixture) {
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 4);
    assert_eq!(attempts[0].http_status, Some(429));
    assert_eq!(attempts[1].http_status, Some(503));
    assert_eq!(attempts[2].http_status, Some(503));
    assert_eq!(
        attempts[1].response_metadata["local_recovery_request_attempts_used"],
        "1"
    );
    assert_eq!(
        attempts[2].response_metadata["local_recovery_request_attempts_used"],
        "2"
    );
    let request_metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(
        request_metadata
            .iter()
            .map(|attempt| {
                attempt.request_metadata["ordinary_attempt_number"]
                    .as_str()
                    .expect("ordinary attempt number should be a string")
            })
            .collect::<Vec<_>>(),
        vec!["1", "1", "1", "1"]
    );
    assert_eq!(
        request_metadata
            .iter()
            .map(|attempt| {
                attempt.request_metadata["attempt_name"]
                    .as_str()
                    .expect("attempt name should be a string")
            })
            .collect::<Vec<_>>(),
        vec![
            "original-policy",
            "original-policy",
            "original-policy",
            "original-policy"
        ]
    );
    assert_eq!(attempts[3].status, "succeeded");
}

#[tokio::test]
async fn shielded_max_attempts_one_keeps_ordinary_flat_across_two_recovery_replays() {
    let recovery_root = unique_test_dir("shielded-429-two-503-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_recovery_max_attempts_two_config(&marker),
    )
    .await;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=shielded-429-then-two-503-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("multi-recovery shielded request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");
    assert!(marker.exists());
    assert_two_recovery_replays_keep_ordinary_flat(&proxy);
    remove_dir_all(&recovery_root);
}

fn assert_recovery_keeps_ordinary_rung_one(proxy: &ProxyFixture) {
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].http_status, Some(429));
    assert_eq!(attempts[1].http_status, Some(503));
    let request_metadata = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(
        request_metadata[1].request_metadata["ordinary_attempt_number"],
        "1"
    );
    assert_eq!(
        request_metadata[2].request_metadata["ordinary_attempt_number"],
        "1"
    );
    assert_eq!(
        request_metadata[2].request_metadata["attempt_name"],
        "original-policy"
    );
    assert_eq!(
        request_metadata[2].request_metadata["attempt_thinking_mode"],
        "force_thinking"
    );
    assert_eq!(
        request_metadata[2].request_metadata["attempt_thinking_budget_tokens"],
        "12345"
    );
    assert_eq!(
        attempts[1].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(
        attempts[1].response_metadata["local_recovery_status"],
        "succeeded"
    );
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(
        read_evidence_attempt_rows(&proxy.evidence_sqlite_path)
            .iter()
            .map(|attempt| attempt.role.as_str())
            .collect::<Vec<_>>(),
        vec!["primary", "primary", "primary"]
    );
}

#[test]
fn persisted_ordinary_number_drives_evidence_and_shadow_ordinals() {
    let metadata = BTreeMap::from([(String::from("ordinary_attempt_number"), String::from("1"))]);

    assert_eq!(ordinary_attempt_number_from_metadata(3, &metadata), 1);
    assert_eq!(next_ordinary_attempt_number_from_metadata(3, &metadata), 2);
}

#[tokio::test]
async fn generic_first_attempt_uses_route_timeout_not_remaining_deadline() {
    let mut fake = FakeUpstream::spawn().await;
    let config = r"
[shielding]
enabled = false

[upstream]
request_timeout_ms = 250

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 80
";
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        config,
    )
    .await;
    let started = Instant::now();
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=generic-delayed-503-once-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("generic first attempt should complete under route timeout");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    response.bytes().await.expect("response should drain");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(120),
        "first generic attempt must wait for the upstream delay even when route timeout exceeds remaining request deadline; elapsed={elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(500),
        "first generic attempt must still complete promptly under the route timeout; elapsed={elapsed:?}"
    );
    let _first = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(50)).await.is_none());
}

#[tokio::test]
async fn shielded_max_attempts_four_rate_limit_does_not_reduce_ordinary_retries() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 4
request_deadline_ms = 5000
max_retry_after_secs = 1
anti_loop_hint_enabled = false
"#,
    )
    .await;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=shielded-429-then-three-503-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("shielded retry ladder request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 5);
    assert_eq!(
        attempts
            .iter()
            .map(|attempt| attempt.attempt_number)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );
    assert_eq!(
        read_attempt_request_metadata_rows(&proxy.sqlite_path)
            .iter()
            .map(|attempt| {
                attempt.request_metadata["ordinary_attempt_number"]
                    .as_str()
                    .expect("ordinary attempt number should be a string")
            })
            .collect::<Vec<_>>(),
        vec!["1", "1", "2", "3", "4"]
    );
    assert_eq!(attempts[4].status, "succeeded");
}

#[tokio::test]
async fn held_non_stream_liveness_reports_configured_mode_and_json_framing() {
    for (configured, framing) in [
        ("sse", "disabled"),
        ("json-whitespace", "json-whitespace"),
        ("disabled", "disabled"),
    ] {
        let mut fake = FakeUpstream::spawn().await;
        let proxy = ProxyFixture::spawn_with_options(
            &fake.base_url,
            true,
            AppConfig::default().server.max_in_flight_requests,
            &format!(
                r#"
[heartbeat]
mode = "{configured}"

[loop_guard]
enabled = false
"#
            ),
        )
        .await;
        let summary = proxy
            .client
            .get(format!("{}/config-summary", proxy.base_url))
            .send()
            .await
            .expect("config summary should complete")
            .text()
            .await
            .expect("config summary should be text");
        assert!(summary.contains(&format!("heartbeat_configured_mode={configured}")));

        let response = proxy
            .client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .json(&json!({"model":"test-chat","messages":[]}))
            .send()
            .await
            .expect("held non-stream request should complete");
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        response.bytes().await.expect("response should drain");
        let _business = recv_non_health_request(&mut fake).await;

        let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
        assert_eq!(
            request_metadata["downstream_liveness_configured_mode"],
            configured
        );
        assert_eq!(
            request_metadata["downstream_liveness_framing_mode"],
            framing
        );
        let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
        assert_eq!(attempts.len(), 1);
        assert_eq!(
            attempts[0].response_metadata["downstream_liveness_configured_mode"],
            configured
        );
        assert_eq!(
            attempts[0].response_metadata["downstream_liveness_framing_mode"],
            framing
        );
    }
}

#[tokio::test]
async fn generic_connect_failure_enters_recovery_but_failed_readiness_prevents_replay() {
    let recovery_root = unique_test_dir("generic-connect-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let unbound_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("temporary listener should bind");
    let unbound_addr = unbound_listener
        .local_addr()
        .expect("temporary listener address should be available");
    drop(unbound_listener);
    let unbound_base_url = format!("http://{unbound_addr}/v1");
    let config = format!(
        r#"
[shielding]
enabled = false

[upstream]
request_timeout_ms = 100

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 1000

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 500
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 50
readiness_deadline_ms = 100
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        marker.display()
    );
    let proxy = ProxyFixture::spawn_with_options(
        &unbound_base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    let response = timeout(
        Duration::from_secs(2),
        proxy
            .client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .json(&json!({"model":"test-chat","messages":[]}))
            .send(),
    )
    .await
    .expect("connect failure must remain bounded")
    .expect("connect failure response should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    response.bytes().await.expect("error response should drain");
    assert!(
        marker.exists(),
        "transient connect failure must enter recovery"
    );
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1, "failed readiness must prevent replay");
    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn generic_recovery_cannot_outlive_total_request_deadline() {
    let mut fake = FakeUpstream::spawn().await;
    let config = r#"
[shielding]
enabled = false

[upstream]
request_timeout_ms = 1000

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 80

[upstream.local_recovery]
enabled = true
restart_command = ["/bin/sleep", "0.4"]
restart_timeout_ms = 1000
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        config,
    )
    .await;
    let started = Instant::now();
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=generic-503-once-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("deadline-bounded generic request should complete");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let _deadline_terminated_body = response.bytes().await;
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "generic recovery must be bounded by the original total deadline"
    );
    let _first = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(450)).await.is_none());
}

#[tokio::test]
async fn shielded_recovery_cannot_replay_after_total_request_deadline() {
    let mut fake = FakeUpstream::spawn().await;
    let config = r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 80
anti_loop_hint_enabled = false

[upstream.local_recovery]
enabled = true
restart_command = ["/bin/sleep", "0.4"]
restart_timeout_ms = 1000
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        config,
    )
    .await;
    let started = Instant::now();
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=transient-503-then-success",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("deadline-bounded shielded request should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    response.bytes().await.expect("response should drain");
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "shielded recovery must be bounded by the original total deadline"
    );
    let _first = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(450)).await.is_none());
}

#[tokio::test]
async fn singleflight_joiner_deadline_expires_without_late_replay_permit() {
    let fake = FakeUpstream::spawn().await;
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let recovery_episode_id = {
        let mut state = coordinator.state.lock().await;
        state.running = true;
        let now = Instant::now();
        state.recovery_started = Some(now);
        state.recovery_deadline = Some(now + Duration::from_secs(1));
        state
            .ensure_active_recovery_episode()
            .expect("running recovery must have an episode")
    };
    let completing_coordinator = Arc::clone(&coordinator);
    let completion = tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        finish_local_recovery_episode(
            &completing_coordinator,
            recovery_episode_id,
            BTreeMap::from([(
                String::from("local_recovery_status"),
                String::from("succeeded"),
            )]),
        )
        .await
    });
    let policy = LocalRecoveryPolicy {
        enabled: true,
        trigger_on_request_deadline: false,
        restart_command: vec![String::from("/bin/true")],
        restart_timeout: Duration::from_secs(1),
        readiness_endpoint: String::from("/v1/chat/completions"),
        readiness_body: json!({"model":"test-chat","messages":[],"max_tokens":1}),
        readiness_request_timeout: Duration::from_secs(1),
        readiness_deadline: Duration::from_secs(1),
        readiness_interval: Duration::from_millis(10),
        max_attempts_per_request: 1,
        cooldown: Duration::from_millis(1),
        budget_window: Duration::from_secs(10),
        max_per_window: 20,
    };
    let attempts = AtomicU64::new(0);
    let request_deadline = RequestDeadline::from_started_at(
        Instant::now()
            .checked_sub(Duration::from_millis(30))
            .expect("test instant should support a short adjustment"),
        Duration::from_millis(50),
    );
    let started = Instant::now();
    let gate = precommit_recovery::gate(
        precommit_recovery::Context {
            policy: &policy,
            coordinator: &coordinator,
            client: build_http_client().expect("test client should build"),
            base_url: &fake.base_url,
            profile_name: "default",
            attempts: &attempts,
            downstream_commit_signal: None,
            downstream_drop_signal: None,
            post_await_self_test: None,
            request_deadline,
            episode_timeout: None,
        },
        false,
        LocalRecoveryCause::TransientTransport,
    )
    .await;
    assert!(started.elapsed() < Duration::from_millis(80));
    assert!(gate.applied);
    assert!(!gate.permits_replay);
    assert_eq!(gate.metadata["local_recovery_status"], "join_timeout");
    assert!(
        completion.await.expect("completion task should join"),
        "the shared episode should still publish its later success"
    );
    assert!(
        !gate.permits_replay,
        "later singleflight completion must not mutate an expired joiner's replay decision"
    );
}

#[tokio::test]
async fn generic_connect_failure_recovers_then_replays_original_request() {
    let recovery_root = unique_test_dir("generic-connect-replay");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("temporary listener should bind");
    let address = listener
        .local_addr()
        .expect("temporary listener address should be available");
    drop(listener);
    let (shutdown_sender, server, mut observed) =
        spawn_upstream_after_marker(address, marker.clone());
    let config = format!(
        r#"
[shielding]
enabled = false

[upstream]
request_timeout_ms = 100

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
request_deadline_ms = 2000

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{}"]
restart_timeout_ms = 500
readiness_body = {{"model":"test-chat","messages":[],"max_tokens":1}}
readiness_request_timeout_ms = 500
readiness_deadline_ms = 1000
readiness_interval_ms = 10
max_attempts_per_request = 1
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 20
"#,
        marker.display()
    );
    let base_url = format!("http://{address}/v1");
    let proxy = ProxyFixture::spawn_with_options(
        &base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=connect-replay",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","messages":[]}))
        .send()
        .await
        .expect("connect recovery request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .bytes()
        .await
        .expect("replayed response should drain");

    let probe = observed
        .recv()
        .await
        .expect("readiness probe should arrive");
    let replay = observed
        .recv()
        .await
        .expect("business replay should arrive");
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert_eq!(
        replay.path_and_query,
        "/v1/chat/completions?test=connect-replay"
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&replay.body)
            .expect("replayed body should remain JSON"),
        json!({"model":"test-chat","messages":[]})
    );
    let _ = shutdown_sender.send(());
    server.await.expect("late upstream should stop cleanly");

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("local_recovery"));
    assert_eq!(attempts[1].status, "succeeded");
    remove_dir_all(&recovery_root);
}

fn spawn_upstream_after_marker(
    address: std::net::SocketAddr,
    marker: PathBuf,
) -> (
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
    mpsc::Receiver<ObservedRequest>,
) {
    let (observed_sender, observed_receiver) = mpsc::channel(4);
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let server = tokio::spawn(async move {
        timeout(Duration::from_secs(1), async {
            while !marker.exists() {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("restart command should publish marker");
        let listener = TcpListener::bind(address)
            .await
            .expect("late upstream should bind");
        let app = axum::Router::new().fallback(axum::routing::any(
            move |request: Request<Body>| {
                let observed_sender = observed_sender.clone();
                async move {
                    let observed = observe_request(request).await;
                    let _ = observed_sender.send(observed).await;
                    json_response(
                        "late-upstream",
                        r#"{"id":"chatcmpl-late","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}]}"#
                            .to_owned(),
                    )
                }
            },
        ));
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_receiver.await;
            })
            .await
            .expect("late upstream should serve");
    });
    (shutdown_sender, server, observed_receiver)
}

#[tokio::test]
async fn generic_deadline_and_caller_scoped_statuses_do_not_restart() {
    for (case, extra_config, expected_status, expected_retry_after, retries_rate_limit) in [
        (
            "deadline",
            "\n[shielding]\nenabled = false\n",
            StatusCode::SERVICE_UNAVAILABLE,
            None,
            false,
        ),
        (
            "bad-request",
            "\n[shielding]\nenabled = false\n",
            StatusCode::BAD_REQUEST,
            None,
            false,
        ),
        (
            "always-429",
            "\n[shielding]\nenabled = false\n",
            StatusCode::TOO_MANY_REQUESTS,
            Some("1"),
            true,
        ),
    ] {
        let recovery_root = unique_test_dir(&format!("generic-no-restart-{case}"));
        fs::create_dir_all(&recovery_root).expect("recovery root should be created");
        let marker = recovery_root.join("restart-ran");
        let base_config = if case == "deadline" {
            GENERIC_RECOVERY_CONFIG
                .replace("request_deadline_ms = 3000", "request_deadline_ms = 25")
        } else {
            GENERIC_RECOVERY_CONFIG.to_owned()
        };
        let config = format!(
            "{}\n{extra_config}\n[upstream.local_recovery]\nrestart_command = [\"/usr/bin/touch\", \"{}\"]\n",
            base_config,
            marker.display()
        );
        let mut fake = FakeUpstream::spawn().await;
        let proxy = ProxyFixture::spawn_with_options(
            &fake.base_url,
            true,
            AppConfig::default().server.max_in_flight_requests,
            &config,
        )
        .await;
        let test = if case == "deadline" {
            "generic-delayed-503-once-then-success"
        } else {
            case
        };
        let response = proxy
            .client
            .post(format!(
                "{}/v1/chat/completions?test={test}",
                proxy.base_url
            ))
            .json(&json!({"model":"test-chat","messages":[]}))
            .send()
            .await
            .expect("non-recovery request should complete");
        assert_eq!(response.status(), expected_status, "case={case}");
        assert_eq!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            expected_retry_after,
            "case={case}"
        );
        response.bytes().await.expect("response should drain");
        assert!(!marker.exists(), "case={case} must not restart");
        let _business_request = fake.recv_next().await;
        if retries_rate_limit {
            let _rate_limit_retry = fake.recv_next().await;
        }
        assert!(fake.recv_within(Duration::from_millis(50)).await.is_none());
        remove_dir_all(&recovery_root);
    }
}

#[tokio::test]
async fn generic_post_byte_drop_never_starts_recovery_or_replay() {
    let recovery_root = unique_test_dir("generic-post-byte");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let marker = recovery_root.join("restart-ran");
    let mut upstream = CancellableUpstream::spawn().await;
    let config = format!(
        "{GENERIC_RECOVERY_CONFIG}\n[shielding]\nenabled = false\n\
         [upstream.local_recovery]\nrestart_command = [\"/usr/bin/touch\", \"{}\"]\n",
        marker.display()
    );
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=generic-post-byte",
            proxy.base_url
        ))
        .json(&json!({"model":"test-chat","stream":true,"messages":[]}))
        .send()
        .await
        .expect("generic stream should start");
    assert_eq!(response.status(), StatusCode::OK);
    let _business = upstream.recv_request().await;
    let mut body = response.bytes_stream();
    assert!(
        body.next()
            .await
            .expect("first body item should exist")
            .is_ok()
    );
    drop(body);
    let _drop = upstream.recv_drop_within(Duration::from_secs(1)).await;
    assert!(!marker.exists());
    assert!(
        upstream
            .recv_request_optional_within(Duration::from_millis(100))
            .await
            .is_none()
    );
    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn dropping_generic_request_during_recovery_prevents_replay() {
    let mut fake = FakeUpstream::spawn().await;
    let config = GENERIC_RECOVERY_CONFIG.replace(
        "restart_command = [\"/bin/true\"]",
        "restart_command = [\"/bin/sleep\", \"0.3\"]",
    ) + "\n[shielding]\nenabled = false\n";
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;
    let request = shielded_chat_request(
        "/v1/chat/completions?test=generic-503-once-then-success",
        r#"{"model":"test-chat","messages":[]}"#,
    );
    let request_task = tokio::spawn(proxy_handler(State(proxy.state.clone()), request));
    let _initial = fake.recv_next().await;
    sleep(Duration::from_millis(30)).await;
    request_task.abort();
    let _ = request_task.await;
    sleep(Duration::from_millis(350)).await;

    if let Some(probe) = fake.recv_within(Duration::from_millis(100)).await {
        assert_eq!(
            probe.path_and_query, "/v1/chat/completions",
            "a shared recovery probe may outlive one caller, but replay must not"
        );
    }
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "dropped handler must not replay the business request"
    );
}

#[tokio::test]
async fn max_attempts_one_c8_502_and_stall_share_recovery_without_heartbeat() {
    let recovery_root = unique_test_dir("local-recovery-c8");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let (script_path, count_path) = write_singleflight_restart_script(&recovery_root);
    let mut fake = FakeUpstream::spawn().await;
    let config = format!(
        "{GENERIC_RECOVERY_CONFIG}\n\
         [upstream.stall]\n\
         enabled = true\n\
         first_chunk_timeout_ms = 50\n\
         idle_timeout_ms = 50\n\
         [upstream.local_recovery]\n\
         restart_command = [\"{}\"]\n",
        script_path.display()
    );
    let proxy = ProxyFixture::spawn_with_options(&fake.base_url, true, 8, &config).await;
    let requests = spawn_c8_requests(&proxy).await;
    let observed = collect_c8_upstream_requests(&mut fake).await;
    assert_c8_responses(requests).await;

    let restart_count = fs::read_to_string(&count_path).expect("restart count should be readable");
    assert_eq!(restart_count.lines().count(), 1);
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
    assert_c8_upstream_shape(&observed);
    assert_c8_attempts(&proxy.sqlite_path);
    remove_dir_all(&recovery_root);
}

async fn spawn_c8_requests(
    proxy: &ProxyFixture,
) -> Vec<tokio::task::JoinHandle<reqwest::Response>> {
    const REQUEST_COUNT: usize = 8;
    let barrier = Arc::new(Barrier::new(REQUEST_COUNT + 1));
    let mut requests = Vec::with_capacity(REQUEST_COUNT);
    for index in 0..REQUEST_COUNT {
        let barrier = Arc::clone(&barrier);
        let client = proxy.client.clone();
        let url = format!(
            "{}/v1/chat/completions?test=local-recovery-c8&request={index}",
            proxy.base_url
        );
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            timeout(
                Duration::from_secs(3),
                client
                    .post(url)
                    .json(&json!({
                        "model": "test-chat",
                        "messages": [{"role": "user", "content": format!("request-{index}")}],
                    }))
                    .send(),
            )
            .await
            .expect("operator idle timeout must exceed bounded recovery")
            .expect("c8 request should complete")
        }));
    }
    barrier.wait().await;
    requests
}

async fn collect_c8_upstream_requests(fake: &mut FakeUpstream) -> Vec<ObservedRequest> {
    let mut observed = Vec::with_capacity(17);
    for _ in 0..17 {
        observed.push(
            timeout(Duration::from_secs(3), fake.recv_next())
                .await
                .expect("c8 upstream request set should be bounded"),
        );
    }
    observed
}

async fn assert_c8_responses(requests: Vec<tokio::task::JoinHandle<reqwest::Response>>) {
    for request in requests {
        let response = request.await.expect("c8 task should join");
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.bytes().await.expect("c8 response should drain");
        assert!(!body.as_ref().starts_with(b":"));
        assert!(!body.as_ref().starts_with(b" \n"));
        assert!(
            !body
                .windows(b"heartbeat".len())
                .any(|window| window == b"heartbeat")
        );
    }
}

fn assert_c8_upstream_shape(observed: &[ObservedRequest]) {
    let readiness_count = observed
        .iter()
        .filter(|request| {
            request.path_and_query == "/v1/chat/completions"
                && request
                    .headers
                    .get("x-llm-guard-proxy-probe")
                    .is_some_and(|value| value == HeaderValue::from_static("local-recovery"))
        })
        .count();
    let business_count = observed
        .iter()
        .filter(|request| request.path_and_query.contains("test=local-recovery-c8"))
        .count();
    assert_eq!(readiness_count, 1);
    assert_eq!(business_count, 16);
}

fn assert_c8_attempts(sqlite_path: &Path) {
    let attempts = read_attempt_chain_rows(sqlite_path);
    assert_eq!(attempts.len(), 16);
    assert_eq!(
        attempts
            .iter()
            .filter(|attempt| attempt.status == "retried")
            .count(),
        8
    );
    assert_eq!(
        attempts
            .iter()
            .filter(|attempt| attempt.status == "succeeded")
            .count(),
        8
    );
    assert!(
        attempts
            .iter()
            .filter(|attempt| attempt.status == "succeeded")
            .all(|attempt| {
                attempt
                    .response_metadata
                    .get("downstream_heartbeat_emitted_count")
                    .is_some_and(|count| count == "0")
            })
    );
}
