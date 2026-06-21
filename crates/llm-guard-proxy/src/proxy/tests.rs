use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use axum::http::header::{AUTHORIZATION, CONNECTION, LOCATION};
use futures_util::{Stream, StreamExt, stream};
use llm_guard_proxy_core::ConfigManager;
use rusqlite::Connection;
use tokio::{
    net::TcpListener,
    sync::mpsc,
    time::{sleep, timeout},
};

use super::*;

const TEST_MAX_BYTES: u64 = 1_000_000;
const TEST_PRUNE_TO_BYTES: u64 = 800_000;
const TEST_MAX_RECORDS: u64 = 100;
const STREAM_DELAY: Duration = Duration::from_millis(800);
const STREAM_HEADER_TIMEOUT: Duration = Duration::from_millis(500);
const STREAM_FIRST_CHUNK_TIMEOUT: Duration = Duration::from_millis(250);
const STREAM_SECOND_CHUNK_GUARD: Duration = Duration::from_millis(150);
const STREAM_COMPLETION_TIMEOUT: Duration = Duration::from_secs(2);
const SHIELDED_SLOW_DELAY: Duration = Duration::from_millis(2_500);
const SHIELDED_HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(1_500);
const SHIELDED_RELOAD_GUARD: Duration = Duration::from_millis(1_500);
const SHIELDED_RELOAD_TIMEOUT: Duration = Duration::from_millis(2_500);
const SSE_FIRST_CHUNK: &[u8] = b"data: first\n\n";
const SSE_SECOND_CHUNK: &[u8] = b"data: second\n\n";
const LONG_JSON_FIRST_CHUNK: &[u8] = br#"{"object":"list","data":["#;
const LONG_JSON_SECOND_CHUNK: &[u8] = br"]}";
const MODEL_METADATA_BODY: &str = r#"{"object":"list","data":[{"id":"aeon-ultimate","object":"model","max_model_len":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_CHUNKED_FIRST: &[u8] =
    br#"{"object":"list","data":[{"id":"chunked-model","object":"model","#;
const MODEL_METADATA_CHUNKED_SECOND: &[u8] =
    br#""max_model_len":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_NO_CONTEXT_BODY: &str = r#"{"object":"list","data":[{"id":"fallback-model","object":"model","owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_CONTEXT_LENGTH_BODY: &str = r#"{"object":"list","data":[{"id":"context-length-model","object":"model","context_length":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_MAX_CONTEXT_LENGTH_BODY: &str = r#"{"object":"list","data":[{"id":"max-context-length-model","object":"model","max_context_length":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const LARGE_MODEL_METADATA_EXTRA_BYTES: usize = 1024 * 1024;
static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[tokio::test]
async fn get_models_forwards_method_path_query_and_headers() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/v1/models?limit=2", proxy.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .header(HOST, "downstream.example")
        .header("x-custom-proxy-test", "keep-me")
        .header(CONNECTION, "x-drop-me")
        .header("x-drop-me", "drop-me")
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("upstream header should be forwarded"),
        "models"
    );
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"object":"list","data":[]}"#
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/models?limit=2");
    assert_eq!(observed.body, Bytes::new());
    assert_eq!(
        observed
            .headers
            .get(AUTHORIZATION)
            .expect("authorization should be forwarded"),
        "Bearer test-token"
    );
    assert_eq!(
        observed
            .headers
            .get("x-custom-proxy-test")
            .expect("custom header should be forwarded"),
        "keep-me"
    );
    assert!(
        observed.headers.get("x-drop-me").is_none(),
        "Connection-nominated hop-by-hop header must not be forwarded"
    );
    assert!(
        observed
            .headers
            .get(HOST)
            .is_some_and(|value| value != "downstream.example"),
        "proxy must let the upstream client set Host instead of forwarding the downstream Host"
    );
}

#[tokio::test]
async fn get_models_enriches_context_metadata_and_preserves_unknown_fields() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "aeon-ultimate");
    assert_eq!(model["owned_by"], "vllm");
    assert_eq!(model["extra"], "keep");
    assert_eq!(model["max_model_len"].as_u64(), Some(256_000));
    assert_eq!(model["context_length"].as_u64(), Some(256_000));
    assert_eq!(model["max_context_length"].as_u64(), Some(256_000));

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/models?test=model-metadata");
}

#[tokio::test]
async fn get_models_enriches_chunked_context_metadata_without_content_length() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-chunked",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "chunked-model");
    assert_eq!(model["owned_by"], "vllm");
    assert_eq!(model["extra"], "keep");
    assert_eq!(model["max_model_len"].as_u64(), Some(256_000));
    assert_eq!(model["context_length"].as_u64(), Some(256_000));
    assert_eq!(model["max_context_length"].as_u64(), Some(256_000));

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-chunked"
    );
}

#[tokio::test]
async fn upstream_context_length_overrides_stale_max_model_len_fallback() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-context-length",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "context-length-model");
    assert_normalized_context_fields(&model, 256_000);

    let observed = fake.recv().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-context-length"
    );
}

#[tokio::test]
async fn upstream_max_context_length_overrides_stale_max_model_len_fallback() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-max-context-length",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "max-context-length-model");
    assert_normalized_context_fields(&model, 256_000);

    let observed = fake.recv().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-max-context-length"
    );
}

#[tokio::test]
async fn enriched_models_response_holds_in_flight_permit_until_downstream_body_finishes() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_request = empty_get_request("/v1/models?test=model-metadata-large");

    let first_response = proxy_handler(State(proxy.state.clone()), first_request).await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first models request should reach upstream and hold the only permit");
    assert_eq!(first_observed.method, Method::GET);
    assert_eq!(
        first_observed.path_and_query,
        "/v1/models?test=model-metadata-large"
    );
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "enriched model responses must not be recorded before downstream body completion"
    );

    let second_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(second_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let second_body = to_bytes(second_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("limit response body should read");
    let second_body =
        String::from_utf8(second_body.to_vec()).expect("limit response should be utf-8");
    assert!(
        second_body.contains("proxy_in_flight_limit_exceeded"),
        "second request should be rejected while first model body is undrained: {second_body}"
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "permit rejection must happen before a second upstream request is sent"
    );

    let first_body = to_bytes(first_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("first enriched model body should read");
    let first_body =
        String::from_utf8(first_body.to_vec()).expect("first enriched model body should be utf-8");
    let first_model_record = first_model(&first_body);
    assert_eq!(first_model_record["context_length"].as_u64(), Some(256_000));
    assert_eq!(
        first_model_record["extra"]
            .as_str()
            .expect("large extra field should stay present")
            .len(),
        LARGE_MODEL_METADATA_EXTRA_BYTES
    );

    let third_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(third_response.status(), StatusCode::OK);
    let third_body = to_bytes(third_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("third model body should read after capacity is released");
    let third_body =
        String::from_utf8(third_body.to_vec()).expect("third model body should be utf-8");
    assert_eq!(
        first_model(&third_body)["context_length"].as_u64(),
        Some(256_000)
    );
    let third_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("third request should reach upstream after first body completion");
    assert_eq!(
        third_observed.path_and_query,
        "/v1/models?test=model-metadata"
    );
}

#[tokio::test]
async fn enriched_models_observability_records_success_after_body_consumption() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream");
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "success must wait until the enriched body reaches EOF"
    );

    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("enriched model body should read");
    let expected_body_len = body.len().to_string();
    let body = String::from_utf8(body.to_vec()).expect("enriched model body should be utf-8");
    assert_eq!(first_model(&body)["context_length"].as_u64(), Some(256_000));

    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        2
    );
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(request_row.abort_reason, None);
    assert_eq!(
        request_row.response_metadata["response_body_bytes"],
        expected_body_len.as_str()
    );
    assert_eq!(request_row.response_metadata["http_status_success"], "true");
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(attempt_row.abort_reason, None);
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        expected_body_len.as_str()
    );
}

#[tokio::test]
async fn enriched_models_observability_records_abort_when_body_is_dropped() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream");
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "droppable response body should own the pending observability record"
    );

    drop(response);

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(request_row.response_metadata["response_body_bytes"], "0");
    assert_eq!(attempt_row.status, "aborted");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempt_row.response_metadata["response_body_bytes"], "0");
}

#[tokio::test]
async fn get_models_reflects_upstream_metadata_changes_between_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let first = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-changing",
            proxy.base_url
        ))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;
    let second = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-changing",
            proxy.base_url
        ))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert_eq!(
        first_model(&first)["context_length"].as_u64(),
        Some(128_000)
    );
    assert_eq!(
        first_model(&second)["context_length"].as_u64(),
        Some(256_000)
    );
}

#[tokio::test]
async fn disabled_model_metadata_discovery_or_enrichment_returns_upstream_body_unchanged() {
    for metadata_config in [
        r"
[upstream.metadata]
discovery_enabled = false
enrich_responses = true
",
        r"
[upstream.metadata]
discovery_enabled = true
enrich_responses = false
",
    ] {
        let fake = FakeUpstream::spawn().await;
        let proxy =
            ProxyFixture::spawn_with_metadata_config(&fake.base_url, true, metadata_config).await;

        let response = proxy
            .client
            .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
            .send()
            .await
            .expect("proxy request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.text().await.expect("body should be text"),
            MODEL_METADATA_BODY
        );
        let _observed = fake.recv().await;
    }
}

#[tokio::test]
async fn config_fallback_context_metadata_is_hot_reloadable() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 4096
",
    )
    .await;

    let first = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("metadata reload should succeed");

    let second = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert!(outcome.applied);
    assert_eq!(first_model(&first)["context_length"].as_u64(), Some(4_096));
    assert_eq!(first_model(&first)["max_model_len"].as_u64(), Some(4_096));
    assert_eq!(first_model(&second)["context_length"].as_u64(), Some(8_192));
    assert_eq!(first_model(&second)["max_model_len"].as_u64(), Some(8_192));
}

#[tokio::test]
async fn hot_reloaded_disabled_discovery_stops_model_metadata_enrichment() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let enriched = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
discovery_enabled = false
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("metadata reload should succeed");

    let disabled = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert!(outcome.applied);
    assert_eq!(
        first_model(&enriched)["context_length"].as_u64(),
        Some(256_000)
    );
    assert_eq!(disabled, MODEL_METADATA_BODY);
}

#[tokio::test]
async fn hermes_like_context_extraction_reads_enriched_model_length() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let body = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("proxy request should complete")
        .text()
        .await
        .expect("body should be text");

    let model = first_model(&body);
    assert_eq!(hermes_like_context_length(&model), Some(256_000));
    let _observed = fake.recv().await;
}

#[tokio::test]
async fn shielded_non_stream_chat_forces_upstream_sse_and_aggregates_json() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"stream":false}"#,
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
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be rewritten for downstream SSE"),
        "text/event-stream"
    );
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("shielded fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["id"], "chatcmpl-shielded");
    assert_eq!(aggregated["object"], "chat.completion");
    assert_eq!(aggregated["created"], 1_710_000_000);
    assert_eq!(aggregated["model"], "test-chat");
    assert_eq!(aggregated["choices"][0]["index"], 0);
    assert_eq!(aggregated["choices"][0]["message"]["role"], "assistant");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        aggregated["choices"][0]["message"]["reasoning_content"],
        "think"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        r#"{"q":"x"}"#
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "stop");
    assert_eq!(aggregated["usage"]["prompt_tokens"], 3);
    assert_eq!(aggregated["usage"]["completion_tokens"], 2);
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "test-chat");
    assert_eq!(observed_body["messages"][0]["content"], "ping");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_stream_options_while_forcing_usage() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream_options":{"include_usage":false,"include_obfuscation":true,"vendor_hint":{"mode":"keep"}}}"#,
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
    assert_eq!(observed_body["stream_options"]["include_obfuscation"], true);
    assert_eq!(
        observed_body["stream_options"]["vendor_hint"]["mode"],
        "keep"
    );
}

#[tokio::test]
async fn shielded_thinking_policy_injects_missing_budget_and_preserves_answer_reserve() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_policy_enabled"], "true");
        assert_eq!(metadata["thinking_policy_budget_tokens"], "32768");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "injected_missing_budget"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "absent");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_schema_path"], "thinking.budget_tokens");
        assert_eq!(metadata["thinking_schema_variant"], "canonical");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32768");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "true"
        );
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens"
        );
    }
}

#[tokio::test]
async fn shielded_thinking_policy_respects_explicit_disable_marker() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"api_key":"sk-secret"}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("thinking").is_none());
    assert!(observed_body.get("thinking_budget").is_none());
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "caller_disabled_thinking"
        );
        assert_eq!(
            metadata["thinking_disable_marker_path"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(
            metadata["thinking_answer_budget_preserved_fields"],
            "max_tokens"
        );
        assert_text_excludes_values(&metadata.to_string(), &["sk-secret", "api_key"]);
    }
}

#[tokio::test]
async fn shielded_thinking_policy_preserves_zero_existing_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":0},"max_tokens":64}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["thinking_rewrite_applied"], "false");
    assert_eq!(
        request_metadata["thinking_rewrite_reason"],
        "existing_budget_zero"
    );
    assert_eq!(request_metadata["thinking_budget_previous_state"], "zero");
    assert_eq!(request_metadata["thinking_budget_final_tokens"], "0");
}

#[tokio::test]
async fn shielded_thinking_policy_raises_smaller_budget_and_adjusts_answer_fields() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"max_tokens":64,"max_completion_tokens":32,"max_output_tokens":16}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_831);
    assert_eq!(observed_body["max_completion_tokens"], 32_799);
    assert_eq!(observed_body["max_output_tokens"], 32_783);
    assert_eq!(
        observed_body["max_tokens"].as_u64().expect("max_tokens")
            - observed_body["thinking"]["budget_tokens"]
                .as_u64()
                .expect("thinking budget"),
        63
    );

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(metadata["thinking_rewrite_reason"], "raised_smaller_budget");
        assert_eq!(metadata["thinking_budget_previous_state"], "smaller");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "1");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32767");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "true"
        );
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens,max_completion_tokens,max_output_tokens"
        );
    }
}

#[tokio::test]
async fn shielded_thinking_policy_updates_extra_body_chat_template_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"extra_body":{"chat_template_kwargs":{"enable_thinking":true,"thinking_budget":8}},"max_completion_tokens":20}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_completion_tokens"], 32_780);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(
        request_metadata["thinking_schema_path"],
        "extra_body.chat_template_kwargs.thinking_budget"
    );
    assert_eq!(
        request_metadata["thinking_schema_variant"],
        "extra-body-chat-template-kwargs"
    );
}

#[tokio::test]
async fn shielded_thinking_policy_disabled_leaves_budget_unchanged_except_streaming() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = false
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
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
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["thinking_policy_enabled"], "false");
    assert_eq!(request_metadata["thinking_rewrite_applied"], "false");
    assert_eq!(
        request_metadata["thinking_rewrite_reason"],
        "policy_disabled"
    );
}

#[tokio::test]
async fn hot_reloaded_thinking_policy_changes_subsequent_rewrites() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
budget_tokens = 1024

[loop_guard]
enabled = false
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"reload"}]}"#,
    );

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    let _first_json = shielded_final_json(first).await;
    let first_observed = fake.recv_next().await;
    let first_body: serde_json::Value =
        serde_json::from_slice(&first_observed.body).expect("first body should be JSON");
    assert_eq!(first_body["thinking"]["budget_tokens"], 1024);

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = false
budget_tokens = 1024

[loop_guard]
enabled = false
",
    );
    let disabled_outcome = proxy
        .manager
        .reload()
        .expect("disabled thinking reload should succeed");

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(second.status(), StatusCode::OK);
    let _second_json = shielded_final_json(second).await;
    let second_observed = fake.recv_next().await;
    let second_body: serde_json::Value =
        serde_json::from_slice(&second_observed.body).expect("second body should be JSON");
    assert!(second_body.get("thinking").is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = true
budget_tokens = 2048

[loop_guard]
enabled = false
",
    );
    let enabled_outcome = proxy
        .manager
        .reload()
        .expect("enabled thinking reload should succeed");

    let third = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("third proxy request should complete");
    assert_eq!(third.status(), StatusCode::OK);
    let _third_json = shielded_final_json(third).await;
    let third_observed = fake.recv_next().await;
    let third_body: serde_json::Value =
        serde_json::from_slice(&third_observed.body).expect("third body should be JSON");
    assert_eq!(third_body["thinking"]["budget_tokens"], 2048);

    assert!(disabled_outcome.applied);
    assert!(enabled_outcome.applied);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_compat_function_call_fields_from_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-function-call",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["service_tier"], "flex");
    assert_eq!(aggregated["choices"][0]["message"]["role"], "assistant");
    assert!(aggregated["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        aggregated["choices"][0]["message"]["function_call"]["name"],
        "legacy_lookup"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["function_call"]["arguments"],
        r#"{"q":"x"}"#
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "function_call");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_compat_refusal_fields_from_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-refusal",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["service_tier"], "flex");
    assert_eq!(
        aggregated["choices"][0]["message"]["refusal"],
        "I cannot help with that"
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "stop");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_chat_preserves_malformed_stream_for_upstream_validation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":"false"}"#,
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
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"chatcmpl-test","object":"chat.completion"}"#
    );
    let observed = fake.recv_next().await;
    assert_eq!(observed.body, body);
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], "false");
    assert!(observed_body.get("stream_options").is_none());
}

#[tokio::test]
async fn shielded_chat_preserves_malformed_stream_options_for_upstream_validation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream_options":"bad"}"#,
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
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"chatcmpl-test","object":"chat.completion"}"#
    );
    let observed = fake.recv_next().await;
    assert_eq!(observed.body, body);
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("stream").is_none());
    assert_eq!(observed_body["stream_options"], "bad");
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_choice_logprobs_from_sse_chunks() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"logprobs":true}"#,
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(
        aggregated["choices"][0]["logprobs"]["content"][0]["token"],
        "Hello"
    );
    assert_eq!(
        aggregated["choices"][0]["logprobs"]["content"][1]["token"],
        "!"
    );
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
    assert_eq!(observed_body["logprobs"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_extension_fields_from_sse_chunks() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-extensions",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["object"], "chat.completion");
    assert_eq!(aggregated["provider_metadata"]["phase"], "final");
    assert_eq!(aggregated["x_provider_trace"], "trace-first");
    assert_eq!(
        aggregated["choices"][0]["provider_choice"]["phase"],
        "final"
    );
    assert_eq!(aggregated["choices"][0]["x_choice_trace"], "choice-final");
    assert!(aggregated["choices"][0].get("delta").is_none());
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        aggregated["choices"][0]["message"]["provider_message"]["phase"],
        "final"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["x_message_trace"],
        "trace-first"
    );
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_chat_attempt_metadata_records_stream_timings_and_delta_counts() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv().await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let metadata_json: String = connection
        .query_row("SELECT response_metadata_json FROM attempts", [], |row| {
            row.get(0)
        })
        .expect("attempt row should exist");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("attempt metadata should be JSON");

    assert_eq!(metadata["shielded_streaming"], "true");
    assert_eq!(metadata["upstream_stream_forced"], "true");
    assert_eq!(
        metadata["upstream_response_header_content-type"],
        "text/event-stream"
    );
    assert_eq!(metadata["finish_reason"], "stop");
    assert_eq!(metadata["delta_count"], "3");
    assert_eq!(metadata["content_delta_count"], "2");
    assert_eq!(metadata["reasoning_delta_count"], "1");
    assert_eq!(metadata["tool_call_delta_count"], "2");
    assert_metadata_latency(&metadata, "first_byte_latency_ms");
    assert_metadata_latency(&metadata, "first_token_latency_ms");
}

#[tokio::test]
async fn default_sse_heartbeat_emits_progress_before_slow_shielded_content() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1
",
    )
    .await;

    let response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .post(format!(
                "{}/v1/chat/completions?test=slow-shielded",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
            .send(),
    )
    .await
    .expect("shielded SSE headers should arrive before upstream content")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("shielded default response should be SSE"),
        "text/event-stream"
    );

    let mut body = response.bytes_stream();
    let heartbeat = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "shielded heartbeat").await;
    let heartbeat_text = std::str::from_utf8(&heartbeat).expect("heartbeat should be UTF-8");
    assert!(heartbeat_text.starts_with(": llm-guard-proxy heartbeat"));
    assert!(!heartbeat_text.contains("Hello"));
    assert!(!heartbeat_text.contains("event: final"));

    let final_body = timeout(Duration::from_secs(4), async {
        let mut collected = BytesMut::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.expect("shielded SSE body should not fail");
            collected.extend_from_slice(&chunk);
            if std::str::from_utf8(&collected)
                .expect("shielded SSE should remain UTF-8")
                .contains("event: final")
            {
                return collected.freeze();
            }
        }
        panic!("shielded SSE should emit a final event");
    })
    .await
    .expect("shielded final event should arrive after heartbeat");
    let final_json = final_json_from_sse_body(&final_body);
    assert_eq!(final_json["choices"][0]["message"]["content"], "Hello");

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
}

#[tokio::test]
async fn repeated_input_selects_json_whitespace_and_body_stays_parseable() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1
",
    )
    .await;
    let body =
        r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":0.2}"#;

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first shielded request should complete");
    assert_eq!(
        first
            .headers()
            .get(CONTENT_TYPE)
            .expect("first request should use default SSE"),
        "text/event-stream"
    );
    let first_json = shielded_final_json(first).await;
    assert_eq!(first_json["id"], "chatcmpl-shielded");
    let _first_observed = fake.recv_next().await;

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second shielded request should complete");
    assert_eq!(
        second
            .headers()
            .get(CONTENT_TYPE)
            .expect("repeated request should switch to JSON"),
        "application/json"
    );
    let second_body = second.text().await.expect("second body should be text");
    assert!(
        second_body.chars().next().is_some_and(char::is_whitespace),
        "JSON whitespace mode should prefix heartbeat whitespace: {second_body:?}"
    );
    let second_json: serde_json::Value =
        serde_json::from_str(&second_body).expect("leading whitespace JSON should parse");
    assert_eq!(second_json["id"], "chatcmpl-shielded");
    let _second_observed = fake.recv_next().await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let rows = connection
        .prepare(
            "SELECT input_fingerprint, downstream_mode, request_metadata_json \
             FROM requests ORDER BY started_at_unix_ms, request_id",
        )
        .expect("request query should prepare")
        .query_map([], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .expect("request query should run")
        .collect::<Result<Vec<_>, _>>()
        .expect("request rows should decode");
    assert_eq!(rows.len(), 2);
    let first_fingerprint = rows[0]
        .0
        .as_ref()
        .expect("first request fingerprint should be recorded");
    let second_fingerprint = rows[1]
        .0
        .as_ref()
        .expect("second request fingerprint should be recorded");
    assert_eq!(first_fingerprint, second_fingerprint);
    assert_eq!(rows[0].1, "streaming");
    assert_eq!(rows[1].1, "non-stream-json");
    let first_metadata: serde_json::Value =
        serde_json::from_str(&rows[0].2).expect("first metadata should parse");
    let second_metadata: serde_json::Value =
        serde_json::from_str(&rows[1].2).expect("second metadata should parse");
    assert_eq!(first_metadata["repeat_input_matched"], "false");
    assert_eq!(second_metadata["repeat_input_matched"], "true");
    assert_eq!(
        second_metadata["downstream_liveness_mode"],
        "json-whitespace"
    );
}

#[tokio::test]
async fn hot_reloaded_heartbeat_interval_changes_without_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[loop_guard]
enabled = false
",
    )
    .await;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=slow-shielded",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"first"}]}"#)
        .send()
        .await
        .expect("first shielded request should complete");
    let mut first_body = first.bytes_stream();
    let first_heartbeat = next_chunk(
        &mut first_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "first interval heartbeat",
    )
    .await;
    assert!(
        std::str::from_utf8(&first_heartbeat)
            .expect("heartbeat should be UTF-8")
            .starts_with(": llm-guard-proxy heartbeat")
    );
    drop(first_body);
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 2

[loop_guard]
enabled = false
",
    );
    proxy
        .manager
        .reload()
        .expect("heartbeat interval reload should succeed");

    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=slow-shielded",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"second"}]}"#)
        .send()
        .await
        .expect("second shielded request should complete");
    let mut second_body = second.bytes_stream();
    assert!(
        timeout(SHIELDED_RELOAD_GUARD, second_body.next())
            .await
            .is_err(),
        "reloaded two-second heartbeat should not arrive within the old interval"
    );
    let second_heartbeat = next_chunk(
        &mut second_body,
        SHIELDED_RELOAD_TIMEOUT,
        "second interval heartbeat",
    )
    .await;
    assert!(
        std::str::from_utf8(&second_heartbeat)
            .expect("heartbeat should be UTF-8")
            .starts_with(": llm-guard-proxy heartbeat")
    );
    let _second_observed = fake.recv_next().await;
}

#[tokio::test]
async fn hot_reloaded_repeat_window_changes_repeated_detection_without_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[loop_guard]
normalized_input_window_secs = 1
max_repeated_inputs = 1
",
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"reload-window"}]}"#;

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first request should complete");
    assert_eq!(
        first
            .headers()
            .get(CONTENT_TYPE)
            .expect("first request should use SSE"),
        "text/event-stream"
    );
    let _first_json = shielded_final_json(first).await;
    let _first_observed = fake.recv_next().await;

    sleep(Duration::from_millis(1_200)).await;

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second request should complete");
    assert_eq!(
        second
            .headers()
            .get(CONTENT_TYPE)
            .expect("expired repeat should stay SSE"),
        "text/event-stream"
    );
    let _second_json = shielded_final_json(second).await;
    let _second_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[loop_guard]
normalized_input_window_secs = 120
max_repeated_inputs = 1
",
    );
    proxy
        .manager
        .reload()
        .expect("repeat window reload should succeed");

    let third = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("third request should complete");
    assert_eq!(
        third
            .headers()
            .get(CONTENT_TYPE)
            .expect("reloaded repeat window should switch to JSON"),
        "application/json"
    );
    let third_body = third.text().await.expect("third body should be text");
    let third_json: serde_json::Value =
        serde_json::from_str(&third_body).expect("third JSON should parse");
    assert_eq!(third_json["id"], "chatcmpl-shielded");
    let _third_observed = fake.recv_next().await;
}

#[test]
fn normalized_chat_fingerprint_excludes_secrets_and_includes_output_parameters() {
    let base_value = chat_body_with_secret_values("one", false);
    let secret_changed_value = chat_body_with_secret_values("two", true);
    let base = Bytes::from(base_value.to_string().into_bytes());
    let secret_changed = Bytes::from(secret_changed_value.to_string().into_bytes());
    let temperature_changed = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":0.7,"max_tokens":16,"max_completion_tokens":32,"max_output_tokens":48,"api_key":"sk-one","access_token":"access-one","metadata":{"authorization":"Bearer one","id_token":"id-one"},"stream":false}"#,
    );
    let message_changed = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"pong"}],"temperature":0.2,"max_tokens":16,"max_completion_tokens":32,"max_output_tokens":48,"api_key":"sk-one","access_token":"access-one","metadata":{"authorization":"Bearer one","id_token":"id-one"},"stream":false}"#,
    );

    let normalized =
        normalize_chat_fingerprint_value(base_value).expect("base body should normalize");
    assert_eq!(normalized["max_tokens"], 16);
    assert_eq!(normalized["max_completion_tokens"], 32);
    assert_eq!(normalized["max_output_tokens"], 48);
    assert_eq!(normalized["thinking"]["budget_tokens"], 24);
    assert_normalized_excludes_secret_fields(&normalized);
    assert_text_excludes_values(
        &normalized.to_string(),
        &[
            "sk-one",
            "access-one",
            "refresh-one",
            "api-token-one",
            "auth-token-one",
            "Bearer one",
            "id-one",
            "session-one",
            "bearer-credential-one",
            "password-one",
            "secret-one",
            "credentials-one",
        ],
    );

    let base_fingerprint = chat_input_fingerprint(&base).expect("base fingerprint should compute");
    let secret_fingerprint =
        chat_input_fingerprint(&secret_changed).expect("secret fingerprint should compute");
    let temperature_fingerprint = chat_input_fingerprint(&temperature_changed)
        .expect("temperature fingerprint should compute");
    let message_fingerprint =
        chat_input_fingerprint(&message_changed).expect("message fingerprint should compute");

    assert_eq!(base_fingerprint, secret_fingerprint);
    assert_ne!(base_fingerprint, temperature_fingerprint);
    assert_ne!(base_fingerprint, message_fingerprint);
    assert_text_excludes_values(
        &base_fingerprint,
        &[
            "sk-one",
            "access-one",
            "id-one",
            "Bearer",
            "refresh-one",
            "api-token-one",
            "auth-token-one",
        ],
    );
}

fn chat_body_with_secret_values(suffix: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.2,
        "max_tokens": 16,
        "max_completion_tokens": 32,
        "max_output_tokens": 48,
        "thinking": {
            "budget_tokens": 24
        },
        "api_key": format!("sk-{suffix}"),
        "access_token": format!("access-{suffix}"),
        "refresh_token": format!("refresh-{suffix}"),
        "api_token": format!("api-token-{suffix}"),
        "auth_token": format!("auth-token-{suffix}"),
        "metadata": {
            "authorization": format!("Bearer {suffix}"),
            "id_token": format!("id-{suffix}"),
            "session_token": format!("session-{suffix}"),
            "bearer_credentials": format!("bearer-credential-{suffix}"),
            "password": format!("password-{suffix}"),
            "secret": format!("secret-{suffix}"),
            "credentials": format!("credentials-{suffix}")
        },
        "stream": stream
    })
}

fn assert_normalized_excludes_secret_fields(normalized: &serde_json::Value) {
    assert!(normalized.get("api_key").is_none());
    assert!(normalized.get("access_token").is_none());
    assert!(normalized.get("refresh_token").is_none());
    assert!(normalized.get("api_token").is_none());
    assert!(normalized.get("auth_token").is_none());
    let metadata = normalized
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .expect("metadata should remain after secret fields are stripped");
    for secret_key in [
        "authorization",
        "id_token",
        "session_token",
        "bearer_credentials",
        "password",
        "secret",
        "credentials",
    ] {
        assert!(!metadata.contains_key(secret_key));
    }
}

fn assert_text_excludes_values(text: &str, values: &[&str]) {
    for value in values {
        assert!(!text.contains(value));
    }
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_completion_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_completion_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_output_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_output_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_thinking_budget_tokens_for_repeat_detection() {
    let base_body = chat_body_with_thinking_budget(16);
    let changed_body = chat_body_with_thinking_budget(32);
    assert_budget_change_is_not_repeated(&base_body, &changed_body);
}

fn assert_token_budget_change_is_not_repeated(field_name: &str) {
    let base_body = chat_body_with_token_budget(field_name, 16);
    let changed_body = chat_body_with_token_budget(field_name, 32);
    assert_budget_change_is_not_repeated(&base_body, &changed_body);
}

fn assert_budget_change_is_not_repeated(base_body: &Bytes, changed_body: &Bytes) {
    let base_fingerprint =
        chat_input_fingerprint(base_body).expect("base fingerprint should compute");
    let changed_fingerprint =
        chat_input_fingerprint(changed_body).expect("changed fingerprint should compute");
    assert_ne!(base_fingerprint, changed_fingerprint);

    let repeat_inputs = RepeatInputCache::default();
    let first_observation = repeat_inputs.observe(&base_fingerprint, 1_000, 120, 1);
    let changed_observation = repeat_inputs.observe(&changed_fingerprint, 2_000, 120, 1);
    let repeated_base_observation = repeat_inputs.observe(&base_fingerprint, 3_000, 120, 1);

    assert_eq!(first_observation, RepeatInputObservation::default());
    assert_eq!(changed_observation, RepeatInputObservation::default());
    assert_eq!(
        repeated_base_observation,
        RepeatInputObservation {
            repeated: true,
            prior_count: 1
        }
    );
}

fn chat_body_with_token_budget(field_name: &str, value: u64) -> Bytes {
    let mut body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.2,
        "stream": false
    });
    body.as_object_mut()
        .expect("test body should be an object")
        .insert(field_name.to_owned(), serde_json::Value::from(value));
    Bytes::from(body.to_string().into_bytes())
}

fn chat_body_with_thinking_budget(value: u64) -> Bytes {
    Bytes::from(
        serde_json::json!({
            "model": "test-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.2,
            "thinking": {
                "budget_tokens": value
            },
            "stream": false
        })
        .to_string()
        .into_bytes(),
    )
}

#[tokio::test]
async fn hot_reloaded_disabled_shielding_falls_back_to_generic_chat_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    let _first_body = first.text().await.expect("first body should be text");
    let first_observed = fake.recv_next().await;
    let first_body: serde_json::Value =
        serde_json::from_slice(&first_observed.body).expect("first upstream body should be JSON");
    assert_eq!(first_body["stream"], true);

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[shielding]
enabled = false
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("shielding reload should succeed");

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("second proxy request should complete");

    assert!(outcome.applied);
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(
        second.text().await.expect("second body should be text"),
        r#"{"id":"chatcmpl-test","object":"chat.completion"}"#
    );
    let second_observed = fake.recv_next().await;
    assert_eq!(second_observed.body, body);
}

#[tokio::test]
async fn completions_forwards_body_without_policy_rewrite() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body =
        Bytes::from_static(br#"{"model":"test-completion","prompt":"hello","max_tokens":1}"#);

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"cmpl-test","object":"text_completion"}"#
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/completions");
    assert_eq!(observed.body, body);
}

#[tokio::test]
async fn non_chat_embeddings_pass_through_without_policy_rewrite() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"embedding-model","input":"abc","thinking":{"budget_tokens":32768},"loop_guard":"unchanged"}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/embeddings", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"object":"list","data":[{"embedding":[0.0]}]}"#
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/embeddings");
    assert_eq!(observed.body, body);
}

#[tokio::test]
async fn upstream_redirects_are_forwarded_without_following() {
    for redirect_status in [
        StatusCode::TEMPORARY_REDIRECT,
        StatusCode::PERMANENT_REDIRECT,
    ] {
        let mut target = RedirectTarget::spawn().await;
        let upstream =
            RedirectingUpstream::spawn(redirect_status, target.capture_url.clone()).await;
        let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;
        let body = Bytes::from_static(
            br#"{"model":"test-chat","messages":[{"role":"user","content":"secret prompt"}]}"#,
        );

        let response = proxy
            .client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(body.clone())
            .send()
            .await
            .expect("proxy request should complete");

        assert_eq!(response.status(), redirect_status);
        assert_eq!(
            response
                .headers()
                .get(LOCATION)
                .expect("redirect location should be forwarded"),
            target.capture_url.as_str()
        );
        assert_eq!(
            response.text().await.expect("body should be text"),
            "redirected"
        );

        let observed = upstream.recv().await;
        assert_eq!(observed.method, Method::POST);
        assert_eq!(observed.path_and_query, "/v1/chat/completions");
        let observed_body: serde_json::Value = serde_json::from_slice(&observed.body)
            .expect("redirected upstream body should be JSON");
        assert_eq!(observed_body["model"], "test-chat");
        assert_eq!(observed_body["messages"][0]["content"], "secret prompt");
        assert_eq!(observed_body["stream"], true);
        assert!(
            target
                .recv_within(Duration::from_millis(100))
                .await
                .is_none(),
            "proxy must not follow upstream redirects or replay the prompt body"
        );
    }
}

#[tokio::test]
async fn sse_response_streams_first_chunk_before_upstream_completion() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .post(format!("{}/v1/chat/completions?test=sse", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"test-chat","messages":[],"stream":true}"#)
            .send(),
    )
    .await
    .expect("proxy should return SSE headers before delayed upstream completion")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be forwarded"),
        "text/event-stream"
    );

    let mut body = response.bytes_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first SSE chunk").await;
    assert_eq!(first, Bytes::from_static(SSE_FIRST_CHUNK));
    assert!(
        timeout(STREAM_SECOND_CHUNK_GUARD, body.next())
            .await
            .is_err(),
        "second SSE chunk arrived before the upstream delay elapsed"
    );
    let second = next_chunk(&mut body, STREAM_COMPLETION_TIMEOUT, "second SSE chunk").await;
    assert_eq!(second, Bytes::from_static(SSE_SECOND_CHUNK));
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("SSE stream end should arrive after delayed chunk")
            .is_none()
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions?test=sse");
}

#[tokio::test]
async fn long_json_response_streams_first_chunk_while_upstream_remains_open() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .get(format!("{}/v1/embeddings?test=long-json", proxy.base_url))
            .send(),
    )
    .await
    .expect("proxy should return JSON headers before delayed upstream completion")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be forwarded"),
        "application/json"
    );

    let mut body = response.bytes_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first JSON chunk").await;
    assert_eq!(first, Bytes::from_static(LONG_JSON_FIRST_CHUNK));
    assert!(
        timeout(STREAM_SECOND_CHUNK_GUARD, body.next())
            .await
            .is_err(),
        "second JSON chunk arrived before the upstream delay elapsed"
    );
    let second = next_chunk(&mut body, STREAM_COMPLETION_TIMEOUT, "second JSON chunk").await;
    assert_eq!(second, Bytes::from_static(LONG_JSON_SECOND_CHUNK));
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("JSON stream end should arrive after delayed chunk")
            .is_none()
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");
}

#[tokio::test]
async fn forwarded_call_writes_observability_metadata() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"observed-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"cmpl-test","object":"text_completion"}"#
    );
    let _observed = fake.recv().await;
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        2
    );

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String, String) = connection
        .query_row(
            "SELECT status, http_status, model_id, request_metadata_json, response_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .expect("request row should exist");
    let attempt_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, request_metadata_json, response_metadata_json FROM attempts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("attempt row should exist");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");
    let response_metadata: serde_json::Value =
        serde_json::from_str(&request_row.4).expect("response metadata should be json");
    let attempt_metadata: serde_json::Value =
        serde_json::from_str(&attempt_row.2).expect("attempt metadata should be json");

    assert_eq!(request_row.0, "succeeded");
    assert_eq!(request_row.1, 200);
    assert_eq!(request_row.2, "observed-model");
    assert_eq!(request_metadata["method"], "POST");
    assert_eq!(request_metadata["path"], "/v1/completions");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(response_metadata["http_status_success"], "true");
    assert_eq!(attempt_row.0, "succeeded");
    assert_eq!(attempt_row.1, 200);
    assert_eq!(attempt_metadata["attempt_number"], "1");
}

#[tokio::test]
async fn observability_disabled_skips_new_forwarded_records() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, false).await;

    let response = proxy
        .client
        .get(format!("{}/v1/models", proxy.base_url))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"object":"list","data":[]}"#
    );
    let _observed = fake.recv().await;
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0
    );
}

#[tokio::test]
async fn invalid_openai_path_writes_failed_request_without_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = send_raw_proxy_get(&proxy.base_url, "/v1/../admin").await;

    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request"),
        "dot-segment target should be rejected: {response}"
    );
    assert!(
        response.contains("invalid_request_path"),
        "error body should identify the path validation failure: {response}"
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 400);
    assert!(request_row.2.contains("invalid_request_path"));
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/../admin");
    assert_eq!(request_metadata["query_present"], "false");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(request_metadata["request_body_bytes"], "unknown");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn upstream_transport_failure_writes_failed_request_and_attempt() {
    let upstream_base_url = closed_upstream_base_url().await;
    let proxy = ProxyFixture::spawn(&upstream_base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live&safe=ok",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete with gateway error");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("body should be text");
    assert!(
        body.contains("upstream_transport_error"),
        "gateway error should identify upstream transport failure: {body}"
    );
    assert_sensitive_query_absent("transport failure response body", &body);

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM requests", [], |row| row.get(0))
        .expect("request count should be readable");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_row: (String, i64, String, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json, response_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .expect("failed request row should exist");
    let attempt_row: (
        String,
        Option<i64>,
        String,
        String,
        String,
        Option<i64>,
        i64,
        i64,
    ) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json, response_metadata_json, duration_ms, started_at_unix_ms, finished_at_unix_ms FROM attempts",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )
        .expect("failed attempt row should exist");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");
    let request_response_metadata: serde_json::Value =
        serde_json::from_str(&request_row.4).expect("request response metadata should be json");
    let attempt_metadata: serde_json::Value =
        serde_json::from_str(&attempt_row.3).expect("attempt metadata should be json");
    let attempt_response_metadata: serde_json::Value =
        serde_json::from_str(&attempt_row.4).expect("attempt response metadata should be json");

    assert_eq!(request_count, 1);
    assert_eq!(attempt_count, 1);
    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 502);
    assert!(request_row.2.contains("upstream_transport_error"));
    assert_sensitive_query_absent("request error_reason", &request_row.2);
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/models");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], "0");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(
        request_response_metadata["error_type"],
        "upstream_transport_error"
    );
    assert_eq!(attempt_row.0, "failed");
    assert_eq!(attempt_row.1, None);
    assert!(attempt_row.2.contains("upstream_transport_error"));
    assert_sensitive_query_absent("attempt error_reason", &attempt_row.2);
    assert_eq!(attempt_metadata["method"], "GET");
    assert_eq!(attempt_metadata["path"], "/v1/models");
    assert_eq!(attempt_metadata["query_present"], "true");
    assert_eq!(attempt_metadata["attempt_number"], "1");
    assert_eq!(
        attempt_response_metadata["upstream_response_received"],
        "false"
    );
    assert!(attempt_row.5.is_some());
    assert!(attempt_row.7 >= attempt_row.6);
}

#[tokio::test]
async fn oversized_body_failure_writes_failed_request_without_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body_len = MAX_PROXY_BODY_BYTES + 1;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?oversize=true")
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, body_len.to_string())
        .body(Body::from(vec![b'a'; body_len]))
        .expect("oversized request should build");

    let response = proxy_handler(State(proxy.state.clone()), request).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let response_body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("error response body should read");
    let response_body =
        String::from_utf8(response_body.to_vec()).expect("error response should be utf-8");
    assert!(
        response_body.contains("request_body_error"),
        "error should identify body read failure: {response_body}"
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");
    let body_len = body_len.to_string();

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 413);
    assert!(request_row.2.contains("request_body_error"));
    assert_eq!(request_metadata["method"], "POST");
    assert_eq!(request_metadata["path"], "/v1/completions");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], body_len.as_str());
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn in_flight_limit_rejects_before_body_buffering() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_request = Request::builder()
        .method(Method::GET)
        .uri("/v1/embeddings?test=long-json")
        .body(Body::empty())
        .expect("first request should build");

    let first_response = proxy_handler(State(proxy.state.clone()), first_request).await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream and hold the only permit");
    assert_eq!(first_observed.method, Method::GET);
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let second_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"large"}"#))
        }
    }));
    let second_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?blocked=true")
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, MAX_PROXY_BODY_BYTES.to_string())
        .body(second_body)
        .expect("second request should build");

    let second_response = proxy_handler(State(proxy.state.clone()), second_request).await;

    assert_eq!(second_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "rejected requests must not be body-buffered before permit admission"
    );
    let response_body = to_bytes(second_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("limit response body should read");
    let response_body =
        String::from_utf8(response_body.to_vec()).expect("limit response should be utf-8");
    assert!(
        response_body.contains("proxy_in_flight_limit_exceeded"),
        "limit response should identify capacity rejection: {response_body}"
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("limit rejection request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 503);
    assert!(request_row.2.contains("proxy_in_flight_limit_exceeded"));
    assert_eq!(request_metadata["method"], "POST");
    assert_eq!(request_metadata["path"], "/v1/completions");
    assert_eq!(request_metadata["query_present"], "true");
    let max_body_bytes = MAX_PROXY_BODY_BYTES.to_string();
    assert_eq!(
        request_metadata["request_body_bytes"],
        max_body_bytes.as_str()
    );
    assert_eq!(attempt_count, 0);
    drop(first_response);
}

#[tokio::test]
async fn invalid_upstream_url_failure_writes_metadata_without_secret() {
    let proxy = ProxyFixture::spawn("http://127.0.0.1:1/v1", true).await;
    let uri = Uri::from_static("/v1/models?limit=2");
    let headers = HeaderMap::new();
    let request_id =
        RequestId::from_string("req-invalid-upstream").expect("request id should be valid");
    let metadata = request_metadata(&Method::GET, &uri, &headers, 0, true);
    let error = ProxyError::invalid_upstream_url(
        "https://user:secret@example.test/v1?x-api-key=sk-test#token=sk-test",
        String::from("must not contain query parameters"),
    )
    .with_request_metadata(metadata);
    let error_type = error.error_type();
    let error_reason = error.to_string();
    let request_metadata = error
        .request_metadata()
        .cloned()
        .expect("invalid upstream URL should carry request metadata");

    record_failed_request(
        &proxy.store,
        FailedRequestRecord {
            request_id,
            started_at_unix_ms: 1_000,
            finished_at_unix_ms: 1_050,
            http_status: error.status().as_u16(),
            error_type,
            error_reason,
            request_metadata,
            attempt: None,
        },
    );

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 500);
    assert!(request_row.2.contains("invalid_upstream_url"));
    assert!(
        request_row
            .2
            .contains("https://redacted:redacted@example.test/v1?redacted")
    );
    assert!(!request_row.2.contains("user:secret"));
    assert!(!request_row.2.contains("secret"));
    assert!(!request_row.2.contains("sk-test"));
    assert!(!request_row.2.contains("x-api-key"));
    assert!(!request_row.2.contains("token=sk-test"));
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/models");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], "0");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn dot_segment_paths_are_rejected_without_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for request_target in ["/v1/../admin", "/v1/%2e%2e/admin", "/v1/%2E/admin"] {
        let response = send_raw_proxy_get(&proxy.base_url, request_target).await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "dot-segment target should be rejected: {response}"
        );
        assert_no_upstream_request(&mut fake).await;
    }
}

#[test]
fn upstream_url_uses_v1_base_without_duplicating_path() {
    let uri = Uri::from_static("/v1/models?limit=2");
    let url = build_upstream_url("http://upstream.example/v1", &uri).expect("url should build");

    assert_eq!(url.as_str(), "http://upstream.example/v1/models?limit=2");
}

#[test]
fn upstream_url_preserves_encoded_path_and_query() {
    let uri = Uri::from_static("/v1/files/a%2Fb?cursor=a%2Fb");
    let url = build_upstream_url("http://upstream.example/v1", &uri).expect("url should build");

    assert_eq!(
        url.as_str(),
        "http://upstream.example/v1/files/a%2Fb?cursor=a%2Fb"
    );
}

#[test]
fn upstream_url_rejects_raw_dot_segment_paths() {
    let uri = Uri::from_static("/v1/../admin");
    let error = build_upstream_url("http://upstream.example/v1", &uri)
        .expect_err("path should be rejected");

    assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    assert_eq!(error.error_type(), "invalid_request_path");
}

#[test]
fn upstream_url_rejects_percent_encoded_dot_segment_paths() {
    for path in [
        "/v1/%2e/admin",
        "/v1/%2E/admin",
        "/v1/%2e%2e/admin",
        "/v1/%2E%2E/admin",
        "/v1/.%2e/admin",
        "/v1/%2e./admin",
    ] {
        let uri = Uri::try_from(path).expect("test URI should be valid");
        let error = match build_upstream_url("http://upstream.example/v1", &uri) {
            Ok(url) => panic!("{path} should be rejected, got {url}"),
            Err(error) => error,
        };

        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.error_type(), "invalid_request_path");
    }
}

#[test]
fn upstream_url_rejects_and_redacts_credential_bearing_base_url() {
    let uri = Uri::from_static("/v1/models");
    let error = build_upstream_url(
        "https://user:secret@example.test/v1?x-api-key=sk-test#token=sk-test",
        &uri,
    )
    .expect_err("credential-bearing upstream URL should be rejected");
    let error = error.to_string();

    assert!(error.contains("invalid upstream base URL"));
    assert!(error.contains("https://redacted:redacted@example.test/v1?redacted"));
    assert!(!error.contains("user:secret"));
    assert!(!error.contains("secret"));
    assert!(!error.contains("sk-test"));
    assert!(!error.contains("x-api-key"));
    assert!(!error.contains("token=sk-test"));
}

#[test]
fn upstream_url_rejects_and_redacts_fragment_base_url() {
    let uri = Uri::from_static("/v1/models");
    let error = build_upstream_url("https://example.test/v1#token=sk-test", &uri)
        .expect_err("fragment-bearing upstream URL should be rejected");
    let error = error.to_string();

    assert!(error.contains("invalid upstream base URL"));
    assert!(error.contains("https://example.test/v1"));
    assert!(!error.contains("sk-test"));
    assert!(!error.contains("token=sk-test"));
}

async fn next_chunk<S>(body: &mut S, wait: Duration, label: &str) -> Bytes
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    timeout(wait, body.next())
        .await
        .unwrap_or_else(|_| panic!("{label} should arrive before timeout"))
        .unwrap_or_else(|| panic!("{label} should not end the stream"))
        .unwrap_or_else(|error| panic!("{label} should not fail: {error}"))
}

async fn shielded_final_json(response: reqwest::Response) -> serde_json::Value {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let body = response.bytes().await.expect("body should be readable");
    if content_type.contains("text/event-stream") {
        final_json_from_sse_body(&body)
    } else {
        serde_json::from_slice(&body).unwrap_or_else(|error| {
            panic!("shielded JSON body should parse: {error}; body={body:?}")
        })
    }
}

fn final_json_from_sse_body(body: &[u8]) -> serde_json::Value {
    let text = std::str::from_utf8(body)
        .unwrap_or_else(|error| panic!("SSE body should be UTF-8: {error}; body={body:?}"));
    for event in text.split("\n\n") {
        let mut event_name = "";
        let mut data = String::new();
        for line in event.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(value) = line.strip_prefix("event:") {
                event_name = value.trim();
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.trim_start());
            }
        }
        if event_name == "final" {
            return serde_json::from_str(&data).unwrap_or_else(|error| {
                panic!("final SSE data should parse as JSON: {error}; data={data}")
            });
        }
    }
    panic!("SSE body should include a final event: {text}");
}

fn first_model(body: &str) -> serde_json::Value {
    let value = serde_json::from_str::<serde_json::Value>(body)
        .unwrap_or_else(|error| panic!("model list should parse as JSON: {error}; body={body}"));
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .and_then(|models| models.first())
        .cloned()
        .unwrap_or_else(|| panic!("model list should contain at least one model: {body}"))
}

fn hermes_like_context_length(model: &serde_json::Value) -> Option<u64> {
    ["context_length", "max_model_len", "max_context_length"]
        .into_iter()
        .find_map(|key| model.get(key).and_then(serde_json::Value::as_u64))
}

fn assert_normalized_context_fields(model: &serde_json::Value, expected: u64) {
    assert_eq!(model["context_length"].as_u64(), Some(expected));
    assert_eq!(model["max_context_length"].as_u64(), Some(expected));
    assert_eq!(model["max_model_len"].as_u64(), Some(expected));
}

fn assert_metadata_latency(metadata: &serde_json::Value, key: &str) {
    let value = metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("{key} should be present"));
    value
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("{key} should be a u64 latency: {error}; value={value}"));
}

fn read_single_request_and_attempt_metadata(
    proxy: &ProxyFixture,
) -> (serde_json::Value, serde_json::Value) {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_metadata_json: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request row should exist");
    let attempt_metadata_json: String = connection
        .query_row("SELECT request_metadata_json FROM attempts", [], |row| {
            row.get(0)
        })
        .expect("attempt row should exist");
    let request_metadata =
        serde_json::from_str(&request_metadata_json).expect("request metadata should parse");
    let attempt_metadata =
        serde_json::from_str(&attempt_metadata_json).expect("attempt metadata should parse");
    (request_metadata, attempt_metadata)
}

fn empty_get_request(uri: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .expect("GET request should build")
}

#[derive(Debug)]
struct ForwardedRecordRow {
    status: String,
    http_status: i64,
    abort_reason: Option<String>,
    response_metadata: serde_json::Value,
}

fn read_single_forwarded_request_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, abort_reason, response_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("request row should exist");
    let response_metadata =
        serde_json::from_str(&row.3).expect("request response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        abort_reason: row.2,
        response_metadata,
    }
}

fn read_single_forwarded_attempt_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, abort_reason, response_metadata_json FROM attempts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("attempt row should exist");
    let response_metadata =
        serde_json::from_str(&row.3).expect("attempt response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        abort_reason: row.2,
        response_metadata,
    }
}

#[derive(Debug)]
struct ObservedRequest {
    method: Method,
    path_and_query: String,
    headers: HeaderMap,
    body: Bytes,
}

struct FakeUpstream {
    base_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
}

#[derive(Clone)]
struct FakeUpstreamState {
    sender: mpsc::Sender<ObservedRequest>,
    changing_model_len: Arc<AtomicU64>,
}

impl FakeUpstream {
    async fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(fake_upstream_handler)
            .with_state(FakeUpstreamState {
                sender,
                changing_model_len: Arc::new(AtomicU64::new(128_000)),
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("fake upstream should bind");
        let addr = listener
            .local_addr()
            .expect("fake upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("fake upstream server failed: {error}");
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            receiver,
        }
    }

    async fn recv(mut self) -> ObservedRequest {
        self.recv_next().await
    }

    async fn recv_next(&mut self) -> ObservedRequest {
        self.receiver
            .recv()
            .await
            .expect("fake upstream should capture a request")
    }

    async fn recv_within(&mut self, wait: Duration) -> Option<ObservedRequest> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }
}

struct RedirectingUpstream {
    base_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
}

impl RedirectingUpstream {
    async fn spawn(status: StatusCode, location: String) -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(redirecting_upstream_handler)
            .with_state(RedirectingUpstreamState {
                sender,
                status,
                location,
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("redirecting upstream should bind");
        let addr = listener
            .local_addr()
            .expect("redirecting upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("redirecting upstream server failed: {error}");
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            receiver,
        }
    }

    async fn recv(mut self) -> ObservedRequest {
        self.receiver
            .recv()
            .await
            .expect("redirecting upstream should capture a request")
    }
}

#[derive(Clone)]
struct RedirectingUpstreamState {
    sender: mpsc::Sender<ObservedRequest>,
    status: StatusCode,
    location: String,
}

struct RedirectTarget {
    capture_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
}

impl RedirectTarget {
    async fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(capture_request_handler)
            .with_state(sender);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("redirect target should bind");
        let addr = listener
            .local_addr()
            .expect("redirect target address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("redirect target server failed: {error}");
            }
        });

        Self {
            capture_url: format!("http://{addr}/v1/redirect-target"),
            receiver,
        }
    }

    async fn recv_within(&mut self, wait: Duration) -> Option<ObservedRequest> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }
}

async fn closed_upstream_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("closed upstream listener should bind");
    let addr = listener
        .local_addr()
        .expect("closed upstream address should be available");
    drop(listener);
    format!("http://{addr}/v1")
}

async fn redirecting_upstream_handler(
    State(state): State<RedirectingUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    state
        .sender
        .send(observed)
        .await
        .expect("redirecting upstream observation should send");

    let mut response = Response::new(Body::from("redirected"));
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        LOCATION,
        HeaderValue::from_str(&state.location).expect("redirect location should be valid"),
    );
    response
}

async fn capture_request_handler(
    State(sender): State<mpsc::Sender<ObservedRequest>>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    sender
        .send(observed)
        .await
        .expect("redirect target observation should send");
    Response::new(Body::from("captured"))
}

async fn fake_upstream_handler(
    State(state): State<FakeUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    let path_and_query = observed.path_and_query.clone();
    let body = observed.body.clone();
    let endpoint = observed
        .path_and_query
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let is_sse_stream = observed.path_and_query.contains("test=sse");
    let is_long_json_stream = observed.path_and_query.contains("test=long-json");
    state
        .sender
        .send(observed)
        .await
        .expect("fake upstream observation should send");

    if is_sse_stream {
        return delayed_stream_response(
            "sse",
            "text/event-stream",
            SSE_FIRST_CHUNK,
            SSE_SECOND_CHUNK,
        );
    }
    if is_long_json_stream {
        return delayed_stream_response(
            "long-json",
            "application/json",
            LONG_JSON_FIRST_CHUNK,
            LONG_JSON_SECOND_CHUNK,
        );
    }

    fake_upstream_endpoint_response(&endpoint, &path_and_query, &state, &body)
}

async fn observe_request(request: Request<Body>) -> ObservedRequest {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_PROXY_BODY_BYTES)
        .await
        .expect("fake upstream body should be readable");
    let path_and_query = parts.uri.path_and_query().map_or_else(
        || parts.uri.path().to_owned(),
        |value| value.as_str().to_owned(),
    );
    ObservedRequest {
        method: parts.method,
        path_and_query,
        headers: parts.headers,
        body,
    }
}

fn fake_upstream_endpoint_response(
    endpoint: &str,
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Response<Body> {
    if endpoint == "/v1/models" {
        if path_and_query.contains("test=model-metadata-chunked") {
            return chunked_json_response(
                "models",
                MODEL_METADATA_CHUNKED_FIRST,
                MODEL_METADATA_CHUNKED_SECOND,
            );
        }
        if path_and_query.contains("test=model-metadata-large") {
            return json_response("models", large_model_metadata_body());
        }
        if path_and_query.contains("test=model-metadata-changing") {
            let max_model_len = state
                .changing_model_len
                .fetch_add(128_000, Ordering::SeqCst);
            return json_response("models", model_metadata_body(max_model_len));
        }
        if path_and_query.contains("test=model-metadata-no-context") {
            return json_response("models", MODEL_METADATA_NO_CONTEXT_BODY.to_owned());
        }
        if path_and_query.contains("test=model-metadata-context-length") {
            return json_response("models", MODEL_METADATA_CONTEXT_LENGTH_BODY.to_owned());
        }
        if path_and_query.contains("test=model-metadata-max-context-length") {
            return json_response("models", MODEL_METADATA_MAX_CONTEXT_LENGTH_BODY.to_owned());
        }
        if path_and_query.contains("test=model-metadata") {
            return json_response("models", MODEL_METADATA_BODY.to_owned());
        }
    }

    let (label, body) = match endpoint {
        "/v1/models" => ("models", r#"{"object":"list","data":[]}"#),
        "/v1/chat/completions"
            if path_and_query.contains("test=compat-function-call")
                && body_requests_stream(body) =>
        {
            return chat_completion_compat_function_call_sse_response(body);
        }
        "/v1/chat/completions"
            if path_and_query.contains("test=compat-refusal") && body_requests_stream(body) =>
        {
            return chat_completion_compat_refusal_sse_response(body);
        }
        "/v1/chat/completions"
            if path_and_query.contains("test=compat-extensions") && body_requests_stream(body) =>
        {
            return chat_completion_extension_fields_sse_response(body);
        }
        "/v1/chat/completions"
            if path_and_query.contains("test=slow-shielded") && body_requests_stream(body) =>
        {
            return slow_chat_completion_sse_response(body);
        }
        "/v1/chat/completions" if body_requests_stream(body) => {
            return chat_completion_sse_response(body);
        }
        "/v1/chat/completions" => (
            "chat-completions",
            r#"{"id":"chatcmpl-test","object":"chat.completion"}"#,
        ),
        "/v1/completions" => (
            "completions",
            r#"{"id":"cmpl-test","object":"text_completion"}"#,
        ),
        "/v1/embeddings" => (
            "embeddings",
            r#"{"object":"list","data":[{"embedding":[0.0]}]}"#,
        ),
        _ => ("unknown", r#"{"error":"unsupported"}"#),
    };
    let status = if label == "unknown" {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::OK
    };
    let mut response = json_response(label, body.to_owned());
    *response.status_mut() = status;
    response
}

fn model_metadata_body(max_model_len: u64) -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"aeon-ultimate","object":"model","max_model_len":{max_model_len},"owned_by":"vllm","extra":"keep"}}]}}"#
    )
}

fn large_model_metadata_body() -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"large-model","object":"model","max_model_len":256000,"owned_by":"vllm","extra":"{}"}}]}}"#,
        "x".repeat(LARGE_MODEL_METADATA_EXTRA_BYTES)
    )
}

fn json_response(label: &'static str, body: String) -> Response<Body> {
    let content_length = body.len().to_string();
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&content_length).expect("content length should be valid"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn chat_completion_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let include_logprobs = body_requests_logprobs(body);
    let first_chunk = chat_completion_first_chunk();
    let second_chunk = chat_completion_second_chunk(include_logprobs);
    let final_chunk = chat_completion_final_chunk(include_usage, include_logprobs);
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&second_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-sse"),
    );
    response
}

fn slow_chat_completion_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let include_logprobs = body_requests_logprobs(body);
    let chunks = vec![
        sse_json(&chat_completion_first_chunk()),
        sse_json(&chat_completion_second_chunk(include_logprobs)),
        sse_json(&chat_completion_final_chunk(
            include_usage,
            include_logprobs,
        )),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_delayed_start_stream_response("chat-completions-slow-sse", chunks)
}

fn chat_completion_delayed_start_stream_response(
    label: &'static str,
    chunks: Vec<Bytes>,
) -> Response<Body> {
    let body = Body::from_stream(stream::unfold(
        (0_usize, chunks),
        |(index, chunks)| async move {
            if index >= chunks.len() {
                return None;
            }
            if index == 0 {
                sleep(SHIELDED_SLOW_DELAY).await;
            }
            let chunk = chunks[index].clone();
            Some((
                Ok::<_, std::convert::Infallible>(chunk),
                (index + 1, chunks),
            ))
        },
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn chat_completion_compat_function_call_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "function_call": {
                    "name": "legacy_lookup",
                    "arguments": "{\"q\""
                }
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "function_call": {
                    "arguments": ":\"x\"}"
                }
            },
            "finish_reason": "function_call"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-compat-function-call-sse", chunks)
}

fn chat_completion_compat_refusal_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "refusal": "I cannot"
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "refusal": " help with that"
            },
            "finish_reason": "stop"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-compat-refusal-sse", chunks)
}

fn chat_completion_extension_fields_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "provider_metadata": {
            "phase": "first"
        },
        "x_provider_trace": "trace-first",
        "choices": [{
            "index": 0,
            "provider_choice": {
                "phase": "first"
            },
            "delta": {
                "role": "assistant",
                "content": "Hel",
                "provider_message": {
                    "phase": "first"
                },
                "x_message_trace": "trace-first"
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "provider_metadata": {
            "phase": "final"
        },
        "choices": [{
            "index": 0,
            "provider_choice": {
                "phase": "final"
            },
            "x_choice_trace": "choice-final",
            "delta": {
                "content": "lo",
                "provider_message": {
                    "phase": "final"
                }
            },
            "finish_reason": "stop"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-extension-fields-sse", chunks)
}

fn chat_completion_stream_response<const N: usize>(
    label: &'static str,
    chunks: [Bytes; N],
) -> Response<Body> {
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

fn chat_completion_first_chunk() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hel"
            },
            "finish_reason": null
        }]
    })
}

fn chat_completion_second_chunk(include_logprobs: bool) -> serde_json::Value {
    let mut chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "lo",
                "reasoning_content": "think",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\""
                    }
                }]
            },
            "finish_reason": null
        }]
    });
    if include_logprobs {
        insert_first_choice_field(
            &mut chunk,
            "logprobs",
            serde_json::json!({
                "content": [{
                    "token": "Hello",
                    "bytes": [72, 101, 108, 108, 111],
                    "logprob": -0.01,
                    "top_logprobs": []
                }]
            }),
        );
    }
    chunk
}

fn chat_completion_final_chunk(include_usage: bool, include_logprobs: bool) -> serde_json::Value {
    let mut chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": ":\"x\"}"
                    }
                }]
            },
            "finish_reason": "stop"
        }]
    });
    if include_logprobs {
        insert_first_choice_field(
            &mut chunk,
            "logprobs",
            serde_json::json!({
                "content": [{
                    "token": "!",
                    "bytes": [33],
                    "logprob": -0.02,
                    "top_logprobs": []
                }]
            }),
        );
    }
    if include_usage {
        chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    chunk
}

fn sse_json(value: &serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn insert_first_choice_field(chunk: &mut serde_json::Value, key: &str, field: serde_json::Value) {
    if let Some(choice) = chunk
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(serde_json::Value::as_object_mut)
    {
        choice.insert(key.to_owned(), field);
    }
}

fn body_requests_stream(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

fn body_requests_stream_usage(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("stream_options")
                .and_then(|stream_options| stream_options.get("include_usage"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

fn body_requests_logprobs(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("logprobs").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

fn chunked_json_response(
    label: &'static str,
    first: &'static [u8],
    second: &'static [u8],
) -> Response<Body> {
    let body = Body::from_stream(stream::iter([
        Ok::<_, std::convert::Infallible>(Bytes::from_static(first)),
        Ok::<_, std::convert::Infallible>(Bytes::from_static(second)),
    ]));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

fn delayed_stream_response(
    label: &'static str,
    content_type: &'static str,
    first: &'static [u8],
    second: &'static [u8],
) -> Response<Body> {
    let body = Body::from_stream(stream::unfold(0_u8, move |state| async move {
        match state {
            0 => Some((
                Ok::<_, std::convert::Infallible>(Bytes::from_static(first)),
                1,
            )),
            1 => {
                sleep(STREAM_DELAY).await;
                Some((
                    Ok::<_, std::convert::Infallible>(Bytes::from_static(second)),
                    2,
                ))
            }
            _ => None,
        }
    }));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

struct ProxyFixture {
    base_url: String,
    client: Client,
    manager: ConfigManager,
    state: ProxyState,
    store: ObservabilityStore,
    sqlite_path: PathBuf,
    root: PathBuf,
}

impl ProxyFixture {
    async fn spawn(upstream_base_url: &str, observability_enabled: bool) -> Self {
        Self::spawn_with_max_in_flight_requests(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
        )
        .await
    }

    async fn spawn_with_max_in_flight_requests(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
    ) -> Self {
        Self::spawn_with_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            "",
        )
        .await
    }

    async fn spawn_with_metadata_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        metadata_config: &str,
    ) -> Self {
        Self::spawn_with_options(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
            metadata_config,
        )
        .await
    }

    async fn spawn_with_options(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        metadata_config: &str,
    ) -> Self {
        let root = unique_test_dir("proxy");
        fs::create_dir_all(&root).expect("test root should be created");
        set_owner_only_dir(&root);
        let config_path = root.join("config.toml");
        let sqlite_path = root.join("storage").join("observability.sqlite3");
        write_proxy_config(
            &config_path,
            upstream_base_url,
            &sqlite_path,
            observability_enabled,
            max_in_flight_requests,
            metadata_config,
        );
        let manager =
            ConfigManager::from_explicit_path(&config_path).expect("proxy config should load");
        let config = manager
            .handle()
            .snapshot()
            .expect("proxy config snapshot should load");
        let store = ObservabilityStore::open(manager.handle()).expect("store should open");
        let state = ProxyState::new(
            manager.handle(),
            manager.path().to_path_buf(),
            store.clone(),
            build_http_client().expect("client should build"),
            config.server.max_in_flight_requests,
        );
        let app = router(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("proxy should bind");
        let addr = listener
            .local_addr()
            .expect("proxy addr should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("proxy test server failed: {error}");
            }
        });

        Self {
            base_url: format!("http://{addr}"),
            client: build_http_client().expect("client should build"),
            manager,
            state,
            store,
            sqlite_path,
            root,
        }
    }
}

impl Drop for ProxyFixture {
    fn drop(&mut self) {
        remove_dir_all(&self.root);
    }
}

fn write_proxy_config(
    config_path: &Path,
    upstream_base_url: &str,
    sqlite_path: &Path,
    observability_enabled: bool,
    max_in_flight_requests: usize,
    metadata_config: &str,
) {
    fs::write(
        config_path,
        format!(
            r#"
[server]
max_in_flight_requests = {max_in_flight_requests}

[upstream]
base_url = "{upstream_base_url}"
{metadata_config}

[observability]
enabled = {observability_enabled}
sqlite_path = "{sqlite_path}"
capture_raw_payloads = false

[observability.retention]
max_bytes = {TEST_MAX_BYTES}
prune_to_bytes = {TEST_PRUNE_TO_BYTES}
max_records = {TEST_MAX_RECORDS}
"#,
            sqlite_path = sqlite_path.display(),
        ),
    )
    .expect("test config should be written");
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let counter = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "llm-guard-proxy-{}-{nanos}-{counter}-{name}",
        std::process::id()
    ))
}

fn set_owner_only_dir(path: &Path) {
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .expect("test root should be owner-only");
}

fn remove_dir_all(path: &Path) {
    if let Err(error) = fs::remove_dir_all(path) {
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

async fn assert_no_upstream_request(fake: &mut FakeUpstream) {
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "invalid proxy path must not be forwarded upstream"
    );
}

fn assert_sensitive_query_absent(label: &str, text: &str) {
    for sensitive in [
        "sk-live",
        "api_key",
        "safe=ok",
        "?api_key=sk-live",
        "?api_key=sk-live&safe=ok",
    ] {
        assert!(
            !text.contains(sensitive),
            "{label} leaked sensitive query fragment {sensitive:?}: {text}"
        );
    }
}

async fn send_raw_proxy_get(base_url: &str, request_target: &str) -> String {
    let base_url = base_url.to_owned();
    let request_target = request_target.to_owned();
    tokio::task::spawn_blocking(move || {
        let url = Url::parse(&base_url).expect("proxy base URL should parse");
        let host = url.host_str().expect("proxy base URL should have a host");
        let port = url.port().expect("proxy base URL should have a port");
        let addr = format!("{host}:{port}");
        let mut stream = std::net::TcpStream::connect(&addr)
            .unwrap_or_else(|error| panic!("proxy TCP connection should open: {error}"));
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout should be set");
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("write timeout should be set");
        write!(
            stream,
            "GET {request_target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
        )
        .expect("raw proxy request should write");

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("raw proxy response should read");
        response
    })
    .await
    .expect("blocking raw proxy request should finish")
}
