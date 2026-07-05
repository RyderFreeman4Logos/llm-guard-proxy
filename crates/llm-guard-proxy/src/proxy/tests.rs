use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use axum::http::header::{AUTHORIZATION, CONNECTION, LOCATION};
use futures_util::{Stream, StreamExt, stream};
use llm_guard_proxy_core::ConfigManager;
use rusqlite::{Connection, params};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
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
const MULTI_LISTENER_MODEL_METADATA_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"embedding-model","object":"model","owned_by":"vllm"},{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_CHAT_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"embedding-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"embedding-model","object":"model","owned_by":"vllm","first":"embedding"},{"id":"embedding-model","object":"model","owned_by":"vllm","first":"duplicate"}]}"#;
const DISTINCT_UPSTREAM_RERANK_MODELS_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_SLOW_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"slow-model","object":"model","owned_by":"vllm"}]}"#;
const REPEATED_INPUT_LOOP_LINE: &str = "legitimate repeated input line for issue ten";
const LARGE_MODEL_METADATA_EXTRA_BYTES: usize = 1024 * 1024;
static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[tokio::test]
async fn health_reports_process_and_upstream_readiness() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        "health_upstream_probe_timeout_ms = 100\n",
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    assert_eq!(body["process"], "alive");
    assert_eq!(body["upstream"], "ready");

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/models");

    let broken = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &broken.base_url,
        true,
        "health_upstream_probe_timeout_ms = 20\n",
    )
    .await;
    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    assert_eq!(body["process"], "alive");
    assert_eq!(body["upstream"], "unavailable");
}

#[tokio::test]
async fn metrics_expose_retained_gauges_without_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live-secret&safe=ok",
            proxy.base_url
        ))
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .header("x-api-key", "sk-header-secret")
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv_next().await;

    let response = proxy
        .client
        .get(format!("{}/metrics", proxy.base_url))
        .send()
        .await
        .expect("metrics request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("metrics should be text");
    assert_metric_type(&body, "llm_guard_proxy_generation_active", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_queued", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_profile_active", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_profile_queued", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_current_retained_requests", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_current_retained_attempts", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_current_retained_retries", "gauge");
    assert_metric_type(
        &body,
        "llm_guard_proxy_current_retained_first_token_latency_ms_le",
        "gauge",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_current_retained_total_latency_ms_le",
        "gauge",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_storage_pruning_events_total",
        "counter",
    );
    assert_legacy_retained_counter_metrics_absent(&body);
    assert_safe_operational_text("metrics", &body);
}

#[tokio::test]
async fn retained_metrics_stay_prometheus_safe_after_pruning() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for index in 0..50 {
        send_metrics_chat_request(&proxy, &mut fake, index).await;
    }
    let before = fetch_metrics(&proxy).await;
    assert_metric_type(
        &before,
        "llm_guard_proxy_current_retained_requests",
        "gauge",
    );
    assert_metric_type(
        &before,
        "llm_guard_proxy_current_retained_first_token_latency_ms_le",
        "gauge",
    );
    assert_legacy_retained_counter_metrics_absent(&before);
    assert!(
        metric_value(
            &before,
            "llm_guard_proxy_current_retained_first_token_latency_ms_observations"
        ) > 0
    );
    assert!(
        metric_value(
            &before,
            "llm_guard_proxy_current_retained_total_latency_ms_observations"
        ) > 0
    );

    let before_prune_events = metric_value(&before, "llm_guard_proxy_storage_pruning_events_total");
    let before_pruned_requests =
        metric_value(&before, "llm_guard_proxy_storage_pruned_requests_total");
    let before_pruned_attempts =
        metric_value(&before, "llm_guard_proxy_storage_pruned_attempts_total");

    for index in 50..52 {
        send_metrics_chat_request(&proxy, &mut fake, index).await;
    }
    let after = fetch_metrics(&proxy).await;
    assert_metric_type(&after, "llm_guard_proxy_current_retained_requests", "gauge");
    assert_metric_type(
        &after,
        "llm_guard_proxy_current_retained_total_latency_ms_le",
        "gauge",
    );
    assert_legacy_retained_counter_metrics_absent(&after);

    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruning_events_total") >= before_prune_events
    );
    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruned_requests_total")
            > before_pruned_requests
    );
    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruned_attempts_total")
            > before_pruned_attempts
    );
}

#[tokio::test]
async fn metrics_expose_generation_active_and_queued_gauges() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=metrics-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should reach upstream");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=metrics-active"
    );

    let (queued_request, queued_body_polled) = tracked_json_request(
        "/v1/completions?slot=metrics-queued",
        br#"{"prompt":"queued"}"#,
    );
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));
    sleep(Duration::from_millis(50)).await;
    assert!(!queued_body_polled.load(Ordering::SeqCst));
    assert!(!queued.is_finished());

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_active"),
        1
    );
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_queued"),
        1
    );

    queued.abort();
    assert!(
        queued
            .await
            .expect_err("queued metrics request should be aborted")
            .is_cancelled()
    );
    drop(active_response);
}

#[tokio::test]
async fn debug_summary_is_disabled_by_default() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .send()
        .await
        .expect("debug request should complete");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[test]
fn admin_token_matcher_accepts_only_exact_values() {
    assert!(admin_token_matches("admin-token", "admin-token"));
    assert!(!admin_token_matches("admin-tokeo", "admin-token"));
    assert!(!admin_token_matches("admin-token-extra", "admin-token"));
    assert!(!admin_token_matches("admin-toke", "admin-token"));
    assert!(!admin_token_matches("", "admin-token"));
}

#[tokio::test]
async fn debug_summary_is_gated_bounded_and_redacted() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 2
"#,
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live-secret",
            proxy.base_url
        ))
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .header("x-api-key", "sk-header-secret")
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv_next().await;

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .send()
        .await
        .expect("unauthorized debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-tokeo")
        .send()
        .await
        .expect("bearer near-miss debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header("x-admin-token", "admin-token-extra")
        .send()
        .await
        .expect("admin-token near-miss debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests?limit=50", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("authorized debug request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("debug body should be text");
    assert!(body.contains("\"limit\":2"));
    assert!(body.contains("\"request_count\":1"));
    assert!(body.contains("\"status\":"));
    assert_safe_operational_text("debug summary", &body);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header("x-admin-token", "admin-token")
        .send()
        .await
        .expect("x-admin-token debug request should complete");
    assert_eq!(response.status(), StatusCode::OK);
}

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
        .header("x-admin-token", "admin-secret")
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
        observed.headers.get("x-admin-token").is_none(),
        "debug/admin token headers must not be forwarded upstream"
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
async fn enriched_models_response_bypasses_generation_in_flight_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_request = empty_get_request("/v1/models?test=model-metadata-large");

    let first_response = proxy_handler(State(proxy.state.clone()), first_request).await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first models request should reach upstream");
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

    assert_eq!(second_response.status(), StatusCode::OK);
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("control-plane models request should bypass generation capacity");
    assert_eq!(
        second_observed.path_and_query,
        "/v1/models?test=model-metadata"
    );
    let second_body = to_bytes(second_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("second model body should read");
    let second_body =
        String::from_utf8(second_body.to_vec()).expect("second model body should be utf-8");
    assert_eq!(
        first_model(&second_body)["context_length"].as_u64(),
        Some(256_000)
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
        .expect("third model body should read");
    let third_body =
        String::from_utf8(third_body.to_vec()).expect("third model body should be utf-8");
    assert_eq!(
        first_model(&third_body)["context_length"].as_u64(),
        Some(256_000)
    );
    let third_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("third request should reach upstream without waiting on generation capacity");
    assert_eq!(
        third_observed.path_and_query,
        "/v1/models?test=model-metadata"
    );
}

#[tokio::test]
async fn models_burst_above_old_control_plane_cap_succeeds_and_health_stays_responsive() {
    const BURST_SIZE_ABOVE_OLD_CAP: usize = 8;

    let default_control_plane_cap = AppConfig::default()
        .server
        .max_control_plane_in_flight_requests;
    assert!(default_control_plane_cap >= BURST_SIZE_ABOVE_OLD_CAP);

    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let mut active_model_responses = Vec::with_capacity(BURST_SIZE_ABOVE_OLD_CAP);

    for _ in 0..BURST_SIZE_ABOVE_OLD_CAP {
        let response = proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        active_model_responses.push(response);
        let observed = fake
            .recv_within(STREAM_HEADER_TIMEOUT)
            .await
            .expect("model burst request should reach upstream");
        assert_eq!(observed.path_and_query, "/v1/models?test=model-metadata");
    }

    let health_response = timeout(
        STREAM_COMPLETION_TIMEOUT,
        proxy
            .client
            .get(format!("{}/health", proxy.base_url))
            .send(),
    )
    .await
    .expect("health should stay responsive during model burst")
    .expect("health request should complete");
    assert_eq!(health_response.status(), StatusCode::OK);
    let health_body = health_response
        .text()
        .await
        .expect("health body should be text");
    let health: serde_json::Value =
        serde_json::from_str(&health_body).expect("health should be JSON");
    assert_eq!(health["process"], "alive");
    assert_eq!(health["upstream"], "ready");

    let health_probe = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("health probe should reach upstream during model burst");
    assert_eq!(health_probe.path_and_query, "/v1/models");

    drop(active_model_responses);
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
async fn model_metadata_uses_named_profile_context_override_for_matching_record() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fallback-route"
base_url = "{base_url}"
match_models = ["fallback-model"]

[upstreams.metadata]
context_length_override = 12345
"#,
            base_url = fake.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "fallback-model");
    assert_eq!(model["extra"], "keep");
    assert_normalized_context_fields(&model, 12_345);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-no-context"
    );
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
            .expect("content type should be JSON for non-stream downstream clients"),
        "application/json"
    );
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("shielded fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let response_body = response.text().await.expect("response body should be text");
    assert!(
        !response_body.starts_with(": llm-guard-proxy heartbeat"),
        "non-stream response must not start with SSE heartbeat: {response_body:?}"
    );
    assert!(
        !response_body.contains("event: final"),
        "non-stream response must not contain SSE final framing: {response_body:?}"
    );
    let aggregated: serde_json::Value =
        serde_json::from_str(&response_body).expect("non-stream response should be JSON");
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
    assert!(
        aggregated["choices"][0]["message"]["tool_calls"][0]
            .get("index")
            .is_none()
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
async fn shielded_non_stream_chat_trims_reasoning_separator_from_final_content() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=reasoning-leading-newlines",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"Say OK"}],"stream":false}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "OK");
    assert_eq!(
        aggregated["choices"][0]["message"]["reasoning_content"],
        "think before answering"
    );

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=reasoning-leading-newlines"
    );
}

#[tokio::test]
async fn shielded_loop_guard_catches_reasoning_line_repeated_hundreds_of_times() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("reasoning loop line"));

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    for metadata in [
        &request_row.response_metadata,
        &attempt_row.response_metadata,
    ] {
        assert_eq!(metadata["loop_detected"], "true");
        assert_eq!(metadata["loop_signal"], "repeated_line");
        assert_eq!(metadata["loop_channel"], "reasoning");
        assert!(
            metadata["loop_sample_hash"]
                .as_str()
                .expect("hash should be a string")
                .starts_with("fnv64:")
        );
        assert!(!metadata.to_string().contains("reasoning loop line"));
    }
}

#[tokio::test]
async fn shielded_loop_guard_catches_semantic_reasoning_repetition_with_varied_wording() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 100
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
reasoning_semantic_similarity_threshold_percent = 45
reasoning_semantic_window_token_count = 24
reasoning_semantic_minimum_token_count = 8
reasoning_semantic_history_window_count = 4

[retry]
enabled = false
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=semantic-reasoning-varied",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("bsdtar"));
    assert!(!body.contains("zipfile"));

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    for metadata in [
        &request_row.response_metadata,
        &attempt_row.response_metadata,
    ] {
        assert_eq!(metadata["loop_detected"], "true");
        assert_eq!(metadata["loop_signal"], "semantic_jaccard");
        assert_eq!(metadata["loop_channel"], "reasoning");
        assert!(
            metadata["loop_semantic_similarity_percent"]
                .as_str()
                .and_then(|value| value.parse::<u64>().ok())
                .expect("semantic similarity should be numeric")
                >= 45
        );
        assert!(!metadata.to_string().contains("bsdtar"));
        assert!(!metadata.to_string().contains("zipfile"));
    }
}

#[tokio::test]
async fn shielded_loop_guard_monitor_records_reasoning_signal_without_abort() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
output_repeated_line_threshold = 4
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    let metadata = &attempt_row.response_metadata;
    assert_eq!(metadata["loop_detector_mode"], "monitor");
    assert_eq!(metadata["loop_signal_0_channel"], "reasoning");
    assert_eq!(metadata["loop_signal_0_reason_code"], "repeated_line");
    assert_eq!(metadata["loop_signal_0_severity"], "abort_candidate");
    assert!(metadata.get("loop_detected").is_none());
    assert!(!metadata.to_string().contains("reasoning loop line"));
}

#[tokio::test]
async fn shielded_loop_guard_disabled_skips_detector_metadata() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "disabled"
output_repeated_line_threshold = 4
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    assert!(
        attempt_row
            .response_metadata
            .get("loop_detector_mode")
            .is_none()
    );
    assert!(
        attempt_row
            .response_metadata
            .get("loop_signal_count")
            .is_none()
    );
    assert!(attempt_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn shielded_loop_guard_monitor_records_tool_argument_and_fingerprint_signals() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=repeated-tool-fingerprint",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    let metadata = &attempt_row.response_metadata;
    let metadata_text = metadata.to_string();
    assert_eq!(metadata["loop_detector_mode"], "monitor");
    assert!(metadata_text.contains("tool_arguments"));
    assert!(metadata_text.contains("tool_arguments_json_completed"));
    assert!(metadata_text.contains("tool_fingerprint"));
    assert!(metadata_text.contains("tool_fingerprint_repeated"));
    assert!(metadata_text.contains("fingerprint_hash"));
    assert!(!metadata_text.contains(r#""q":"#));
    assert!(!metadata_text.contains("limit"));
}

#[tokio::test]
async fn debug_summary_exposes_bounded_loop_detector_metadata_without_raw_payloads() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
"#,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 5
"#,
        "",
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=repeated-tool-fingerprint",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"debug-summary-prompt-secret"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests?limit=5", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("debug summary request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("debug summary should be text");
    let summary: serde_json::Value =
        serde_json::from_str(&body).expect("debug summary should be JSON");
    let request = summary["requests"]
        .as_array()
        .and_then(|requests| {
            requests.iter().find(|request| {
                request["response_metadata"]["loop_detector_mode"].as_str() == Some("monitor")
            })
        })
        .expect("debug summary should include the loop request");
    let metadata = request["response_metadata"]
        .as_object()
        .expect("debug summary response metadata should be an object");
    let metadata_text = request["response_metadata"].to_string();

    assert_eq!(
        metadata
            .get("loop_detector_mode")
            .and_then(serde_json::Value::as_str),
        Some("monitor")
    );
    assert!(
        metadata_text.contains("tool_arguments_json_completed"),
        "debug summary should include bounded completed-JSON detector signal: {metadata_text}"
    );
    assert!(
        metadata_text.contains("tool_fingerprint_repeated"),
        "debug summary should include bounded fingerprint detector signal: {metadata_text}"
    );
    assert!(metadata_text.contains("fingerprint_hash"));
    assert!(metadata.len() < 200);
    assert!(!body.contains("debug-summary-prompt-secret"));
    assert!(!metadata_text.contains(r#""q":"#));
    assert!(!metadata_text.contains(r#""limit":1"#));
    assert!(!metadata_text.contains("lookup"));

    let metrics = fetch_metrics(&proxy).await;
    assert_metric_type(
        &metrics,
        "llm_guard_proxy_current_retained_requests",
        "gauge",
    );
    assert!(!metrics.contains("debug-summary-prompt-secret"));
    assert!(!metrics.contains(r#""q":"#));
    assert!(!metrics.contains(r#""limit":1"#));
    assert!(!metrics.contains("lookup"));
}

#[tokio::test]
async fn shielded_loop_guard_does_not_flag_repeated_input_without_output_loop() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
output_repeated_line_threshold = 4
"#,
    )
    .await;
    let repeated_input = format!("{REPEATED_INPUT_LOOP_LINE}\n{REPEATED_INPUT_LOOP_LINE}\n");
    let body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": repeated_input}],
    });

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert!(request_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn shielded_loop_guard_records_suspect_for_output_copying_repeated_input() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
output_repeated_line_threshold = 4
output_token_window_size = 8
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
input_overlap_threshold_multiplier = 3
"#,
    )
    .await;
    let body = repeated_input_chat_body();

    let under_threshold = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=copy-input-under-threshold",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("under-threshold request should complete");
    assert_eq!(under_threshold.status(), StatusCode::OK);
    let _under_body = under_threshold
        .text()
        .await
        .expect("under-threshold body should be text");

    let over_threshold = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=copy-input-over-threshold",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("over-threshold request should complete");
    assert_eq!(over_threshold.status(), StatusCode::OK);
    let _over_body = over_threshold
        .text()
        .await
        .expect("over-threshold body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_reason_code"],
        "repeated_line"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_channel"],
        "content"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_severity"],
        "suspect"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_feature_threshold"],
        "12"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_feature_input_overlap_applied"],
        "true"
    );
    assert!(attempt_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn hot_reloaded_loop_threshold_changes_subsequent_requests() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 10
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
"#,
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#;

    let before_reload = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-six",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(before_reload.status(), StatusCode::OK);
    let _before_body = before_reload
        .text()
        .await
        .expect("first body should be text");

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("loop threshold reload should succeed");
    assert!(outcome.applied);

    let after_reload = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-six",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(after_reload.status(), StatusCode::BAD_GATEWAY);

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.response_metadata["loop_detected"], "true");
    assert_eq!(attempt_row.response_metadata["loop_threshold"], "4");
}

#[tokio::test]
async fn shielded_retry_loops_once_then_succeeds_without_emitting_loop() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(!aggregated.to_string().contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&first_attempt.body));
    assert!(body_contains_retry_hint(&second_attempt.body));
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "successful retry should stop after the second upstream attempt"
    );

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_final_outcome"],
        "succeeded"
    );
    assert_eq!(request_row.response_metadata["retry_max_attempts"], "5");
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("loop_guard"));
    assert_eq!(attempts[0].response_metadata["loop_detected"], "true");
    assert_eq!(attempts[0].response_metadata["attempt_max_attempts"], "5");
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "succeeded");
    assert_eq!(attempts[1].response_metadata["attempt_max_attempts"], "5");
}

#[tokio::test]
async fn evidence_disabled_creates_no_evidence_artifacts_after_proxy_request() {
    let mut fake = FakeUpstream::spawn().await;
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
    let _json = shielded_final_json(response).await;
    let _observed = fake.recv_next().await;
    assert!(!proxy.evidence_sqlite_path.exists());
}

#[tokio::test]
async fn evidence_enabled_records_loop_primary_and_fallback_without_raw_payloads() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false
include_request_headers = false

[evidence.shadow]
enabled = false

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].role, "primary");
    assert_eq!(rows[0].shown_to_downstream, 0);
    assert_eq!(rows[0].status, "rejected");
    assert_eq!(rows[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(rows[0].detector_features["loop_detected"], "true");
    assert_eq!(rows[1].role, "fallback");
    assert_eq!(rows[1].shown_to_downstream, 1);
    assert_eq!(rows[1].status, "accepted");
    assert_eq!(rows[1].thinking_budget_tokens, Some(32_768));

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_groups"),
        1
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_chunks"),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE raw_input IS NOT NULL OR raw_output IS NOT NULL OR raw_reasoning IS NOT NULL OR raw_tool_calls IS NOT NULL",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_shadow_keep_false_does_not_record_shadow_or_extra_upstream_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = false
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 50

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let _primary = fake.recv_next().await;
    let _fallback = fake.recv_next().await;
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "keep_looping_attempt_running=false must abort the looped primary instead of issuing a shadow upstream request"
    );

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].role, "primary");
    assert_eq!(rows[0].status, "rejected");
    assert_eq!(rows[0].shown_to_downstream, 0);
    assert_eq!(rows[1].role, "fallback");
    assert_eq!(rows[1].status, "accepted");
    assert_eq!(rows[1].shown_to_downstream, 1);
    assert!(!rows.iter().any(|row| row.role == "shadow_continued"));

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_raw_capture_redacts_headers_and_payload_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r"
enabled = true
include_raw_payloads = true
include_request_headers = true
",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Bearer qb secret «redacted:sk-…»"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let _observed = fake.recv_next().await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let chunks = read_evidence_chunks(&connection);
    assert_eq!(chunks.len(), 6);
    assert_eq!(chunks[0].0, "input");
    assert_eq!(chunks[0].1, 0);
    assert!(chunks[0].2.contains("[REDACTED]"));
    assert!(!chunks[0].2.contains("sk-"));
    assert_eq!(
        &chunks[1..],
        &[
            (String::from("content"), 1, String::from("Hel")),
            (String::from("content"), 2, String::from("lo")),
            (String::from("reasoning"), 3, String::from("think")),
            (String::from("tool_arguments"), 4, String::from(r#"{"q""#)),
            (String::from("tool_arguments"), 5, String::from(r#":"x"}"#)),
        ]
    );
    let (request_metadata_json, raw_input, raw_output): (String, Option<String>, Option<String>) =
        connection
            .query_row(
                "SELECT request_metadata_json, raw_input, raw_output FROM evidence_attempts",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("raw evidence attempt should exist");
    assert!(request_metadata_json.contains("request_header_authorization"));
    assert!(request_metadata_json.contains("[REDACTED]"));
    assert!(!request_metadata_json.contains("downstream-secret"));
    assert!(
        !raw_input
            .as_deref()
            .unwrap_or_default()
            .contains("sk-live-secret")
    );
    assert!(!raw_input.as_deref().unwrap_or_default().contains("qb"));
    assert!(
        raw_input
            .as_deref()
            .unwrap_or_default()
            .contains("[REDACTED]")
    );
    assert_eq!(raw_output.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn evidence_raw_capture_preserves_loop_rejected_primary_reasoning() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = true

[evidence.shadow]
enabled = false

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let raw_reasoning: Option<String> = connection
        .query_row(
            "SELECT raw_reasoning FROM evidence_attempts WHERE role = 'primary'",
            [],
            |row| row.get(0),
        )
        .expect("primary raw reasoning should query");
    let raw_reasoning = raw_reasoning.expect("looped primary should keep raw reasoning");
    assert!(raw_reasoning.contains("reasoning loop line"));
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_chunks c \
             JOIN evidence_attempts a ON a.attempt_id = c.attempt_id \
             WHERE a.role = 'primary' AND c.channel = 'reasoning'",
        ),
        3
    );
}

#[tokio::test]
async fn evidence_shadow_raw_capture_records_stream_channels_and_redacts_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = true
include_request_headers = true

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 2000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-shadow-raw-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer tiny-header")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Bearer tiny-token sk-t"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let upstream_requests = recv_n_upstream_requests(&mut fake, 3).await;
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| !body_contains_retry_hint(&request.body))
            .count(),
        2
    );
    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        1,
    )
    .await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_shadow_raw_attempt_redacts_and_preserves_stream_payloads(&connection);
    assert_shadow_raw_chunks_redacted(&connection);
}

fn assert_shadow_raw_attempt_redacts_and_preserves_stream_payloads(connection: &Connection) {
    let (request_metadata_json, raw_input, raw_output, raw_reasoning, raw_tool_calls): (
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = connection
        .query_row(
            "SELECT request_metadata_json, raw_input, raw_output, raw_reasoning, raw_tool_calls \
             FROM evidence_attempts WHERE role = 'shadow_continued'",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("shadow raw evidence attempt should exist");
    assert!(request_metadata_json.contains("request_header_authorization"));
    assert!(request_metadata_json.contains("[REDACTED]"));
    assert!(!request_metadata_json.contains("tiny-header"));
    let raw_input = raw_input.expect("shadow raw input should be captured");
    assert!(raw_input.contains("[REDACTED]"));
    assert!(!raw_input.contains("tiny-token"));
    assert!(!raw_input.contains("sk-t"));
    assert_eq!(raw_output.as_deref(), Some("Hello"));
    assert_eq!(raw_reasoning.as_deref(), Some("think"));
    let tool_calls: serde_json::Value = serde_json::from_str(
        raw_tool_calls
            .as_deref()
            .expect("shadow raw tool calls should be captured"),
    )
    .expect("shadow raw tool calls should be JSON");
    assert_eq!(tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(tool_calls[0]["function"]["arguments"], r#"{"q":"x"}"#);
}

fn assert_shadow_raw_chunks_redacted(connection: &Connection) {
    let shadow_chunks = read_evidence_chunks_for_role(connection, "shadow_continued");
    assert_eq!(shadow_chunks.len(), 6);
    assert_eq!(shadow_chunks[0].0, "input");
    assert_eq!(shadow_chunks[0].1, 0);
    assert!(shadow_chunks[0].2.contains("[REDACTED]"));
    assert!(!shadow_chunks[0].2.contains("tiny-token"));
    assert!(!shadow_chunks[0].2.contains("sk-t"));
    assert_eq!(
        &shadow_chunks[1..],
        &[
            (String::from("content"), 1, String::from("Hel")),
            (String::from("content"), 2, String::from("lo")),
            (String::from("reasoning"), 3, String::from("think")),
            (String::from("tool_arguments"), 4, String::from(r#"{"q""#)),
            (String::from("tool_arguments"), 5, String::from(r#":"x"}"#)),
        ]
    );
}

#[tokio::test]
async fn evidence_shadow_skeleton_records_skipped_shadow_without_affecting_fallback() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 0
shadow_attempt_timeout_ms = 10

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2].role, "shadow_continued");
    assert_eq!(rows[2].shown_to_downstream, 0);
    assert_eq!(rows[2].status, "skipped");
    assert_eq!(rows[2].shadow_skip_reason.as_deref(), Some("global_limit"));
}

#[tokio::test]
async fn evidence_shadow_per_request_limit_records_skip_without_extra_upstream_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 0
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 50

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "per-request shadow limit should not issue a shadow upstream request"
    );

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2].role, "shadow_continued");
    assert_eq!(rows[2].status, "skipped");
    assert_eq!(
        rows[2].shadow_skip_reason.as_deref(),
        Some("per_request_limit")
    );
}

#[tokio::test]
async fn evidence_shadow_timeout_releases_global_permit_for_next_request() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 20

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    send_shadow_timeout_request(&proxy, 1).await;
    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_shadow_global_limit_skips_concurrent_request_and_releases_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 2000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let first_client = proxy.client.clone();
    let first_base_url = proxy.base_url.clone();
    let first_request = tokio::spawn(async move {
        send_shadow_timeout_request_parts(&first_client, &first_base_url, 1).await;
    });
    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_n_upstream_requests(&mut fake, 2).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "global shadow limit should skip the concurrent shadow request"
    );
    first_request
        .await
        .expect("first concurrent shadow request task should finish");

    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;
    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        1
    );

    send_shadow_timeout_request(&proxy, 3).await;
    let third_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        third_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;
}

#[tokio::test]
async fn evidence_shadow_downstream_drop_records_terminal_status_and_releases_global_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 20

[heartbeat]
mode = "json-whitespace"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
downstream_drop_policy = "detach"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-shadow-timeout-then-success&id=drop",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should start");
    assert_eq!(response.status(), StatusCode::OK);
    let mut downstream = response.bytes_stream();
    let heartbeat = next_chunk(
        &mut downstream,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "drop test shielded heartbeat",
    )
    .await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(downstream);

    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;
    assert_shadow_timeout_count_stays(&proxy.evidence_sqlite_path, 1).await;

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;

    assert_shadow_timeout_summary(&proxy.evidence_sqlite_path, 2, 2);
}

#[tokio::test]
async fn retry_ladder_advances_from_max_thinking_loop_to_bounded_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true

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
anti_loop_hint = "Previous attempt became repetitive. Answer directly."

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000
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
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(!aggregated.to_string().contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert!(!body_contains_retry_hint(&first_attempt.body));
    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    assert!(second_body_text.contains("Previous attempt became repetitive. Answer directly."));
    assert!(!second_body_text.contains("reasoning loop line"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].response_metadata["attempt_name"],
        "max-thinking"
    );
    assert_eq!(attempts[0].response_metadata["attempt_index"], "0");
    assert_eq!(
        attempts[0].response_metadata["attempt_thinking_mode"],
        "force_thinking"
    );
    assert_eq!(
        attempts[0].response_metadata["attempt_thinking_budget_tokens"],
        "32768"
    );
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(
        attempts[1].response_metadata["attempt_name"],
        "bounded-thinking"
    );
    assert_eq!(attempts[1].response_metadata["attempt_index"], "1");
    assert_eq!(
        attempts[1].response_metadata["retry_previous_reason"],
        "previous_loop_detected"
    );
    assert_eq!(
        attempts[1].response_metadata["attempt_thinking_budget_tokens"],
        "8192"
    );
}

#[tokio::test]
async fn retry_ladder_advances_to_no_thinking_after_two_loop_rejections() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

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
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));
    assert!(!body_contains_retry_hint(&second_attempt.body));
    assert!(!body_contains_retry_hint(&third_attempt.body));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(
        attempts[0].response_metadata["attempt_name"],
        "max-thinking"
    );
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[1].response_metadata["attempt_name"],
        "bounded-thinking"
    );
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(
        attempts[2].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
}

#[tokio::test]
async fn retry_anti_loop_hint_stays_single_message_across_repeated_loop_retries() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true

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
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(retry_hint_count(&first_attempt.body), 0);
    assert_eq!(retry_hint_count(&second_attempt.body), 1);
    assert_eq!(retry_hint_count(&third_attempt.body), 1);

    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    assert!(second_body_text.contains("retry_attempt=2/3"));
    let third_body_text = String::from_utf8_lossy(&third_attempt.body);
    assert!(third_body_text.contains("retry_attempt=3/3"));
    assert!(
        !third_body_text.contains("retry_attempt=2/3"),
        "retry bodies must be rebuilt from the original downstream body, not from the previous generated retry body"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].status, "succeeded");
}

#[tokio::test]
async fn shielded_retry_runs_recovery_command_after_upstream_stall_then_succeeds() {
    let mut fake = FakeUpstream::spawn().await;
    let recovery_root = unique_test_dir("stall-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let recovery_marker = recovery_root.join("recovered");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
idle_timeout_ms = 50
recovery_command = ["/usr/bin/touch", "{recovery_marker}"]
recovery_timeout_ms = 1000
recovery_cooldown_ms = 1000
recovery_budget_window_ms = 10000
recovery_max_per_window = 1
"#,
            recovery_marker = recovery_marker.display()
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(recovery_marker.exists());

    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_detected"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_status"],
        "succeeded"
    );
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "succeeded");

    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn shielded_retry_does_not_replay_when_recovery_command_fails() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
idle_timeout_ms = 50
recovery_command = ["/bin/false"]
recovery_timeout_ms = 1000
recovery_cooldown_ms = 1000
recovery_budget_window_ms = 10000
recovery_max_per_window = 1
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _body = response
        .text()
        .await
        .expect("error body should be consumed");
    let _first_attempt = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(attempts[0].retry_reason, None);
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_status"],
        "exit_failure"
    );
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_permits_retry"],
        "false"
    );
}

#[tokio::test]
async fn upstream_stall_recovery_is_single_flight_and_budget_limited() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/sleep"), String::from("0.2")],
        recovery_timeout: Duration::from_secs(2),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 1,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let first_recovery = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { run_upstream_stall_recovery(&policy, &coordinator).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let joined = run_upstream_stall_recovery(&policy, &coordinator).await;
    let first = first_recovery
        .await
        .expect("first recovery task should join");

    assert_eq!(first["upstream_stall_recovery_status"], "succeeded");
    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");

    tokio::time::sleep(Duration::from_millis(5)).await;
    let budget_limited = run_upstream_stall_recovery(&policy, &coordinator).await;
    assert_eq!(
        budget_limited["upstream_stall_recovery_status"],
        "skipped_budget_exhausted"
    );
    assert_eq!(budget_limited["upstream_stall_recovery_budget_runs"], "1");
}

#[tokio::test]
async fn upstream_stall_recovery_joiners_do_not_hang_after_leader_cancellation() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/sleep"), String::from("0.2")],
        recovery_timeout: Duration::from_secs(2),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let leader = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { run_upstream_stall_recovery(&policy, &coordinator).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    leader.abort();
    assert!(
        leader
            .await
            .expect_err("leader should be cancelled")
            .is_cancelled()
    );

    let joined = timeout(
        Duration::from_millis(500),
        run_upstream_stall_recovery(&policy, &coordinator),
    )
    .await
    .expect("later stall recovery should not wait forever after leader cancellation");

    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");
}

#[tokio::test]
async fn upstream_stall_recovery_joiner_uses_completed_state_after_lost_notification() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/true")],
        recovery_timeout: Duration::from_millis(1),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    {
        let mut state = coordinator.state.lock().await;
        state.running = true;
    }
    let joined = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { wait_for_upstream_stall_recovery_result(&policy, &coordinator, true).await }
    });
    sleep(Duration::from_millis(50)).await;
    {
        let mut state = coordinator.state.lock().await;
        state.running = false;
        state.last_finished = Some(Instant::now());
        state.last_result = Some(BTreeMap::from([
            (
                String::from("upstream_stall_recovery_configured"),
                String::from("true"),
            ),
            (
                String::from("upstream_stall_recovery_status"),
                String::from("succeeded"),
            ),
        ]));
    }

    let joined = timeout(Duration::from_millis(1_500), joined)
        .await
        .expect("lost notification simulation should not hang until the test timeout")
        .expect("joiner task should complete");

    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");
}

#[cfg(unix)]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_descendant_process_group() {
    let test_dir = unique_test_dir("recovery-process-group");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let child_pid_path = test_dir.join("child.pid");
    let script_path = test_dir.join("spawn-descendant.sh");
    fs::write(
        &script_path,
        format!(
            "#!/bin/sh\nsleep 30 &\necho \"$!\" > {}\nsleep 30\n",
            child_pid_path.display()
        ),
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![script_path.display().to_string()],
        recovery_timeout: Duration::from_millis(100),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 1,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let metadata = run_upstream_stall_recovery(&policy, &coordinator).await;
    let child_pid = read_pid_file(&child_pid_path).await;

    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_scope"],
        "process_group"
    );
    assert_process_not_running(child_pid).await;
    remove_dir_all(&test_dir);
}

#[cfg(unix)]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_term_resistant_descendant_process_group() {
    let test_dir = unique_test_dir("recovery-term-resistant-process-group");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let child_pid_path = test_dir.join("child.pid");
    let script_path = test_dir.join("spawn-term-resistant-descendant.sh");
    fs::write(
        &script_path,
        format!(
            "#!/bin/sh\nsh -c 'trap \"\" TERM; echo \"$$\" > {}; while :; do sleep 1; done' &\nsleep 30\n",
            child_pid_path.display()
        ),
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![script_path.display().to_string()],
        recovery_timeout: Duration::from_millis(100),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 1,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let metadata = run_upstream_stall_recovery(&policy, &coordinator).await;
    let child_pid = read_pid_file(&child_pid_path).await;

    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_scope"],
        "process_group"
    );
    assert_process_not_running(child_pid).await;
    remove_dir_all(&test_dir);
}

#[cfg(unix)]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_term_resistant_group_leader_before_join_timeout() {
    let test_dir = unique_test_dir("recovery-term-resistant-group-leader");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let child_pid_path = test_dir.join("child.pid");
    let script_path = test_dir.join("term-resistant-leader.sh");
    fs::write(
        &script_path,
        format!(
            "#!/bin/sh\ntrap '' TERM\necho \"$$\" > {}\nwhile :; do sleep 1; done\n",
            child_pid_path.display()
        ),
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let policy = UpstreamStallPolicy {
        enabled: true,
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![script_path.display().to_string()],
        recovery_timeout: Duration::from_millis(100),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 1,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let metadata = run_upstream_stall_recovery(&policy, &coordinator).await;
    let child_pid = read_pid_file(&child_pid_path).await;

    if metadata
        .get("upstream_stall_recovery_status")
        .map(String::as_str)
        != Some("timeout_killed")
    {
        kill_process_if_running(child_pid).await;
    }
    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_kill_sent"],
        "true"
    );
    assert_process_not_running(child_pid).await;
    remove_dir_all(&test_dir);
}

#[tokio::test]
async fn shielded_retry_all_loop_attempts_returns_error_and_records_chain() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("reasoning loop line"));
    for _ in 0..3 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "3");
    assert_eq!(
        request_row.response_metadata["retry_final_outcome"],
        "failed"
    );
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].status, "failed");
    for attempt in &attempts {
        assert_eq!(attempt.abort_reason.as_deref(), Some("loop_guard"));
        assert_eq!(attempt.response_metadata["loop_detected"], "true");
        assert_eq!(attempt.response_metadata["attempt_max_attempts"], "3");
    }
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert!(attempts[2].retry_reason.is_none());
}

#[tokio::test]
async fn shielded_retry_policy_can_be_disabled_for_single_attempt_behavior() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
enabled = false
max_attempts = 5
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

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let observed = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&observed.body));
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
}

#[tokio::test]
async fn shielded_retry_transient_upstream_status_then_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=transient-503-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first = fake.recv_next().await;
    let _second = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[0].response_metadata["status_code"], "503");
    assert_eq!(attempts[1].status, "succeeded");
}

#[tokio::test]
async fn shielded_retry_exhausted_upstream_status_returns_structured_proxy_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=always-429",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"rate-limit"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body = response.text().await.expect("body should be text");
    assert!(body.contains("llm_guard_upstream_error"));
    assert!(!body.contains("rate-limit"));

    let _first = fake.recv_next().await;
    let _second = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:none:transient_upstream_status,2:failed:none:none"
    );
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[0].response_metadata["status_code"], "429");
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "failed");
    assert_eq!(attempts[1].response_metadata["status_code"], "429");
    assert_eq!(attempts[1].response_metadata["retry_exhausted"], "true");
}

#[tokio::test]
async fn hot_reloaded_retry_max_attempts_reduces_subsequent_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
"#,
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::BAD_GATEWAY);
    let _ = first.text().await.expect("first body should be text");
    for _ in 0..4 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 2
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("retry max attempts reload should succeed");
    assert!(outcome.applied);

    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(second.status(), StatusCode::BAD_GATEWAY);
    let _ = second.text().await.expect("second body should be text");
    for _ in 0..2 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(request_row.response_metadata["retry_max_attempts"], "2");
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
async fn force_thinking_canonical_default_injects_thinking_budget_tokens() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert!(observed_body.get("chat_template_kwargs").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_default_injection_schema"], "canonical");
        assert_eq!(metadata["thinking_schema_path"], "thinking.budget_tokens");
        assert_eq!(metadata["thinking_schema_variant"], "canonical");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_chat_template_kwargs_schema_injects_enable_thinking_and_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_default_injection_schema"],
            "chat_template_kwargs"
        );
        assert_eq!(
            metadata["thinking_schema_path"],
            "chat_template_kwargs.thinking_budget"
        );
        assert_eq!(metadata["thinking_schema_variant"], "chat-template-kwargs");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_chat_template_kwargs_schema_preserves_existing_containers() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"extra_body":{"thinking":{}},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["extra_body"]["thinking"]["budget_tokens"],
        32_768
    );
    assert!(observed_body.get("chat_template_kwargs").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_default_injection_schema"],
            "chat_template_kwargs"
        );
        assert_eq!(
            metadata["thinking_schema_path"],
            "extra_body.thinking.budget_tokens"
        );
        assert_eq!(metadata["thinking_schema_variant"], "extra-body-canonical");
    }
}

#[tokio::test]
async fn hot_reloaded_default_injection_schema_changes_injection_path() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"

[loop_guard]
enabled = false
"#,
    )
    .await;

    let canonical_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"schema-before"}],"max_tokens":64}"#,
    )
    .await;
    assert_eq!(canonical_body["thinking"]["budget_tokens"], 32_768);
    assert!(canonical_body.get("chat_template_kwargs").is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"

[loop_guard]
enabled = false
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("default injection schema reload should succeed");
    assert!(outcome.applied);

    let chat_template_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"schema-after"}],"max_tokens":64}"#,
    )
    .await;
    assert!(chat_template_body.get("thinking").is_none());
    assert_eq!(
        chat_template_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        chat_template_body["chat_template_kwargs"]["thinking_budget"],
        32_768
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
async fn force_thinking_default_overrides_enable_thinking_false() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_no_thinking_marker_policy"], "force");
        assert_eq!(metadata["thinking_no_thinking_marker_detected"], "true");
        assert_eq!(
            metadata["thinking_no_thinking_marker_source"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(metadata["thinking_no_thinking_marker_overridden"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_respect_markers_preserves_enable_thinking_false() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert!(
        observed_body["chat_template_kwargs"]
            .get("thinking_budget")
            .is_none()
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_respect_markers_preserves_reasoning_effort_none() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"reasoning_effort":"none","max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["reasoning_effort"], "none");
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_escape_hatch_only_honors_disable_thinking_escape_hatch() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "escape_hatch_only"
"#,
    )
    .await;

    let normal_marker_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        normal_marker_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        normal_marker_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(normal_marker_body["max_tokens"], 32_832);

    let escape_hatch_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"llm_guard_proxy_disable_thinking":true,"max_tokens":64}"#,
    )
    .await;
    assert_eq!(escape_hatch_body["llm_guard_proxy_disable_thinking"], true);
    assert!(escape_hatch_body.get("thinking").is_none());
    assert_eq!(escape_hatch_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_respect_markers_records_observability() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_no_thinking_marker_policy"],
            "respect_no_thinking_markers"
        );
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "caller_no_thinking_marker_passthrough"
        );
        assert_eq!(metadata["thinking_no_thinking_marker_detected"], "true");
        assert_eq!(
            metadata["thinking_no_thinking_marker_source"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(
            metadata["thinking_no_thinking_marker_escape_hatch"],
            "false"
        );
        assert!(
            metadata
                .get("thinking_no_thinking_marker_overridden")
                .is_none()
        );
    }
}

#[tokio::test]
async fn hot_reloaded_no_thinking_marker_policy_changes_force_thinking_behavior() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let forced_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(forced_body["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(
        forced_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("marker policy reload should succeed");
    assert!(outcome.applied);

    let passthrough_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        passthrough_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert!(
        passthrough_body["chat_template_kwargs"]
            .get("thinking_budget")
            .is_none()
    );
    assert_eq!(passthrough_body["max_tokens"], 64);
}

#[tokio::test]
async fn tool_request_passthrough_leaves_thinking_and_answer_budget_untouched() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
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
    assert_eq!(observed_body["thinking"]["budget_tokens"], 1);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_policy_enabled"], "true");
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "tool_request_passthrough"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "smaller");
        assert_eq!(metadata["thinking_budget_final_tokens"], "1");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "false"
        );
    }
}

#[tokio::test]
async fn force_disable_thinking_zeroes_existing_budget_paths_without_answer_budget_raise() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
force_disable = true
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false,"thinking_token_budget":123,"thinking":{"budget_tokens":456,"enabled":true},"chat_template_kwargs":{"thinking_budget":789,"enable_thinking":true},"extra_body":{"thinking_token_budget":321},"max_tokens":64,"max_completion_tokens":32,"max_output_tokens":16}"#,
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
    assert_eq!(observed_body["thinking_token_budget"], 0);
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(observed_body["thinking"]["enabled"], false);
    assert_eq!(observed_body["chat_template_kwargs"]["thinking_budget"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["extra_body"]["thinking_token_budget"], 0);
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["max_completion_tokens"], 32);
    assert_eq!(observed_body["max_output_tokens"], 16);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "false"
        );

        let rewritten_paths = metadata["thinking_budget_rewritten_paths"]
            .as_str()
            .expect("rewritten paths should be a string");
        for expected_path in [
            "thinking_token_budget",
            "thinking.budget_tokens",
            "chat_template_kwargs.thinking_budget",
            "extra_body.thinking_token_budget",
        ] {
            assert!(
                rewritten_paths.split(',').any(|path| path == expected_path),
                "missing rewritten path {expected_path} in {rewritten_paths}"
            );
        }
    }
}

#[tokio::test]
async fn force_disable_thinking_overrides_tool_request_passthrough() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
force_disable = true
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
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
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
    }
}

#[tokio::test]
async fn tool_request_passthrough_policy_still_injects_non_tool_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
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
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "false");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "injected_missing_budget"
        );
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_legacy_functions_and_preserves_budgets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"functions":[{"name":"lookup","parameters":{"type":"object","properties":{}}}],"thinking":{"budget_tokens":200},"max_completion_tokens":50}"#,
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
    assert_eq!(observed_body["thinking"]["budget_tokens"], 200);
    assert_eq!(observed_body["max_completion_tokens"], 50);
    assert!(observed_body.get("max_tokens").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "tool_request_passthrough"
        );
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_tool_choice_selector_only() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    // tool_choice="auto" without tools array should still be treated as a tool request.
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tool_choice":"auto","thinking":{"budget_tokens":77},"max_output_tokens":40}"#,
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
    assert_eq!(observed_body["thinking"]["budget_tokens"], 77);
    assert_eq!(observed_body["max_output_tokens"], 40);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn tool_request_passthrough_ignores_tool_choice_none() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    // tool_choice="none" should NOT trigger passthrough; regular thinking policy applies.
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tool_choice":"none","max_tokens":64}"#,
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
    // Regular policy injected the default budget and adjusted max_tokens.
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "false");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_legacy_function_call_selector() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"function_call":"auto","thinking":{"budget_tokens":99},"max_tokens":30}"#,
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
    assert_eq!(observed_body["thinking"]["budget_tokens"], 99);
    assert_eq!(observed_body["max_tokens"], 30);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn streaming_chat_applies_thinking_policy_without_downstream_aggregation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64,"stream":true}"#,
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
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("streaming fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let response_body = response.text().await.expect("stream body should be text");
    assert!(response_body.contains("chat.completion.chunk"));
    assert!(!response_body.contains("event: final"));

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);
    assert!(observed_body.get("stream_options").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["policy_transform_applied"], "true");
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
        assert!(metadata.get("shielded_streaming").is_none());
        assert!(metadata.get("upstream_stream_forced").is_none());
    }
}

#[tokio::test]
async fn streaming_chat_force_disable_thinking_injects_zero_without_downstream_aggregation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
force_disable = true
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true,"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
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
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("streaming fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let response_body = response.text().await.expect("stream body should be text");
    assert!(response_body.contains("chat.completion.chunk"));
    assert!(!response_body.contains("event: final"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["chat_template_kwargs"]["thinking_budget"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert!(observed_body.get("stream_options").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["policy_transform_applied"], "true");
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "absent");
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(
            metadata["thinking_schema_path"],
            "chat_template_kwargs.thinking_budget"
        );
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert!(metadata.get("shielded_streaming").is_none());
        assert!(metadata.get("upstream_stream_forced").is_none());
    }
}

#[tokio::test]
async fn shielded_streaming_commit_gate_sends_heartbeat_before_openai_sse_release() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=slow-shielded",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let heartbeat = next_chunk(
        &mut body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "shielded stream heartbeat",
    )
    .await;
    assert_eq!(
        heartbeat,
        Bytes::from_static(b": llm-guard-proxy heartbeat\n\n")
    );
    assert!(!String::from_utf8_lossy(&heartbeat).contains("content"));
    assert!(!String::from_utf8_lossy(&heartbeat).contains("tool_calls"));

    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("data:"));
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains("Hel"));
    assert!(!released.contains("event: final"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].response_metadata["retry_shielded_streaming_enabled"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["downstream_liveness_mode"],
        "sse"
    );
}

#[tokio::test]
async fn shielded_streaming_trims_reasoning_separator_from_released_openai_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=reasoning-leading-newlines",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Say OK"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains(r#""content":"OK""#), "{released}");
    assert!(!released.contains(r#""content":"\n\nOK""#), "{released}");
    assert!(released.contains("data: [DONE]"));
    assert!(!released.contains("event: final"));

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=reasoning-leading-newlines"
    );
}

#[tokio::test]
async fn shielded_streaming_emits_aggregated_logprobs_once_in_openai_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true,"logprobs":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 2, "{released}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "Hello");
    let logprobs = chunks[0]["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("delta chunk should carry aggregated logprobs once");
    assert_eq!(logprobs.len(), 2);
    assert_eq!(logprobs[0]["token"], "Hello");
    assert_eq!(logprobs[1]["token"], "!");
    assert!(
        chunks[1]["choices"][0].get("logprobs").is_none(),
        "{released}"
    );
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
    assert!(released.contains("data: [DONE]"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["logprobs"], true);
}

#[tokio::test]
async fn shielded_streaming_emits_tool_calls_in_openai_delta_shape() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"tool"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 2, "{released}");
    let tool_calls = chunks[0]["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .expect("delta chunk should carry tool calls");
    assert_eq!(tool_calls[0]["index"], 0);
    assert_eq!(tool_calls[0]["id"], "call_1");
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(tool_calls[0]["function"]["arguments"], r#"{"q":"x"}"#);
    assert!(
        chunks[1]["choices"][0]["delta"].get("tool_calls").is_none(),
        "{released}"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
}

#[tokio::test]
async fn shielded_streaming_discards_rejected_tool_call_buffer_before_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
interval_secs = 1

[loop_guard]
mode = "enforce"

[retry]
max_attempts = 2
shielded_streaming_enabled = true
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=tool-loop-then-content-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"tool"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let first = next_chunk(
        &mut body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "shielded stream heartbeat",
    )
    .await;
    let mut released = String::from_utf8_lossy(&first).into_owned();
    if first == Bytes::from_static(b": llm-guard-proxy heartbeat\n\n") {
        assert!(!released.contains("tool_calls"));
        released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    } else {
        released.push_str(&collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await);
    }
    assert!(released.contains("Safe"));
    assert!(!released.contains("lookup"));
    assert!(!released.contains("arguments"));
    assert!(!released.contains("tool_calls"));

    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].status, "succeeded");
}

#[tokio::test]
async fn shielded_streaming_direct_relays_final_no_thinking_retry_after_loop_downgrades() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
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
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=loop-twice-then-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains("Hel"));
    assert!(released.contains("data: [DONE]"));
    assert!(!released.contains("event: final"));
    assert!(!released.contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(attempts[2].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(
        attempts[2].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_loop_inspection_skipped"],
        "no_thinking_direct_streaming_relay"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    assert_eq!(
        request_row.response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
}

#[tokio::test]
async fn shielded_streaming_direct_no_thinking_drop_cancels_upstream_relay() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        1,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true
downstream_drop_policy = "cancel"

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
            "{}/v1/chat/completions?test=loop-twice-then-cancellable-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#)
        .send()
        .await
        .expect("streaming direct relay request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    let mut downstream = response.bytes_stream();
    let first = next_chunk(
        &mut downstream,
        STREAM_COMPLETION_TIMEOUT,
        "first direct no-thinking SSE chunk",
    )
    .await;
    assert!(first.starts_with(b"data: "));
    drop(downstream);

    let first_attempt = upstream.recv_request().await;
    let second_attempt = upstream.recv_request().await;
    let third_attempt = upstream.recv_request().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[2].status, "aborted");
    assert_eq!(
        attempts[2].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
}

#[tokio::test]
async fn per_model_routing_selects_named_upstream_and_records_bounded_metadata() {
    let mut default = FakeUpstream::spawn().await;
    let mut aeon = FakeUpstream::spawn().await;
    let mut fast = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{aeon_base_url}"
match_models = ["aeon-ultimate"]
request_timeout_ms = 90000

[upstreams.metadata]
context_length_override = 4096
input_token_safety_margin = 64

[upstreams.thinking]
mode = "force_thinking"
thinking_token_budget = 128

[[upstreams]]
name = "fast-no-think"
base_url = "{fast_base_url}"
match_models = ["fast-local"]

[upstreams.thinking]
mode = "force_disable"
"#,
            aeon_base_url = aeon.base_url,
            fast_base_url = fast.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"aeon-ultimate","prompt":"hello","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"cmpl-test","object":"text_completion"}"#
    );
    let observed = aeon.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/completions");
    assert!(
        default
            .recv_within(Duration::from_millis(100))
            .await
            .is_none()
    );
    assert!(fast.recv_within(Duration::from_millis(100)).await.is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "aeon-chat");
        assert_eq!(metadata["upstream_route_reason"], "matched_model");
        assert_eq!(metadata["upstream_request_timeout_ms"], "90000");
        assert_eq!(metadata["upstream_context_window_tokens"], "4096");
        assert_eq!(metadata["upstream_input_token_safety_margin"], "64");
        assert!(!metadata.to_string().contains("hello"));
    }
}

#[tokio::test]
async fn unmatched_or_missing_model_routes_to_default_profile() {
    let mut default = FakeUpstream::spawn().await;
    let mut aeon = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{aeon_base_url}"
match_models = ["aeon-ultimate"]
"#,
            aeon_base_url = aeon.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"unmatched-model","prompt":"hello","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let unmatched = default.recv_next().await;
    assert_eq!(unmatched.path_and_query, "/v1/completions");

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"prompt":"missing model","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let missing = default.recv_next().await;
    assert_eq!(missing.path_and_query, "/v1/completions");
    assert!(aeon.recv_within(Duration::from_millis(100)).await.is_none());
}

#[tokio::test]
async fn per_profile_thinking_force_thinking_overrides_disable_marker_and_total_cap() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{base_url}"
match_models = ["aeon-ultimate"]

[upstreams.thinking]
mode = "force_thinking"
max_tokens = 10
thinking_token_budget = 4
budget_accounting = "total_cap"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"aeon-ultimate","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2}"#,
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
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 4);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(observed_body["max_tokens"], 10);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "aeon-chat");
        assert_eq!(metadata["thinking_policy_mode"], "force_thinking");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "4");
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens"
        );
    }
}

#[tokio::test]
async fn per_profile_thinking_force_disable_writes_zero_budget_and_disable_marker() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fast-no-think"
base_url = "{base_url}"
match_models = ["fast-local"]

[upstreams.thinking]
mode = "force_disable"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"fast-local","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":9},"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
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
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "fast-no-think");
        assert_eq!(metadata["thinking_policy_mode"], "force_disable");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
    }
}

#[tokio::test]
async fn per_profile_thinking_passthrough_leaves_caller_thinking_fields_unchanged() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "tool-route"
base_url = "{base_url}"
match_models = ["tool-model"]

[upstreams.thinking]
mode = "passthrough"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"tool-model","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":7},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2}"#,
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
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 7);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 2);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "tool-route");
        assert_eq!(metadata["thinking_policy_mode"], "passthrough");
        assert_eq!(metadata["thinking_rewrite_reason"], "mode_passthrough");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn streaming_chat_downstream_drop_cancels_upstream_relay() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("streaming chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("cancellable SSE upstream should be used"),
        "cancellable-chat-sse"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);

    let mut downstream = response.bytes_stream();
    let first = next_chunk(
        &mut downstream,
        STREAM_FIRST_CHUNK_TIMEOUT,
        "first cancellable SSE chunk",
    )
    .await;
    assert!(first.starts_with(b"data: "));
    drop(downstream);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn non_stream_chat_downstream_drop_cancels_upstream_body() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r"
[shielding]
enabled = false
",
        "",
        "",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("non-stream chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("cancellable JSON upstream should be used"),
        "cancellable-chat-json"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_ne!(observed_body["stream"], true);

    drop(response);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-json");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn shielded_non_stream_chat_downstream_drop_cancels_upstream_aggregation() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("shielded chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("shielded non-stream response should advertise JSON"),
        "application/json"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    drop(response);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn shielded_non_stream_detach_drop_allows_upstream_attempt_to_continue_until_timeout() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"

[retry]
downstream_drop_policy = "detach"

[upstream.stall]
enabled = true
idle_timeout_ms = 200
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("shielded chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    let observed = upstream.recv_request().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);

    let mut downstream = response.bytes_stream();
    let heartbeat = next_chunk(
        &mut downstream,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "detach JSON prefix",
    )
    .await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(downstream);

    assert!(
        upstream
            .recv_drop_optional_within(Duration::from_millis(50))
            .await
            .is_none(),
        "detach mode should not cancel upstream immediately on downstream drop"
    );
    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        request_row.response_metadata["downstream_drop_policy"],
        "detach"
    );
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "aborted");
    assert_eq!(
        attempts[0].response_metadata["downstream_drop_policy"],
        "detach"
    );
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
async fn shielded_thinking_policy_raises_all_known_non_zero_budget_paths_once() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":32768},"extra_body":{"chat_template_kwargs":{"thinking_budget":8}},"max_tokens":64}"#,
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
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_824);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(metadata["thinking_rewrite_reason"], "raised_smaller_budget");
        assert_eq!(metadata["thinking_budget_previous_state"], "mixed");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "multiple");
        assert_eq!(metadata["thinking_schema_path"], "multiple");
        assert_eq!(metadata["thinking_schema_variant"], "multiple");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32760");
        assert_eq!(
            metadata["thinking_budget_observed_paths"],
            "thinking.budget_tokens=equal,extra_body.chat_template_kwargs.thinking_budget=smaller"
        );
        assert_eq!(
            metadata["thinking_budget_rewritten_paths"],
            "extra_body.chat_template_kwargs.thinking_budget"
        );
        assert_eq!(
            metadata["thinking_budget_preserved_paths"],
            "thinking.budget_tokens"
        );
        assert_eq!(metadata["thinking_budget_zero_paths"], "none");
    }
}

#[tokio::test]
async fn shielded_thinking_policy_zero_budget_in_any_known_path_opts_out() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":32768},"extra_body":{"chat_template_kwargs":{"thinking_budget":0}},"max_tokens":64}"#,
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
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        0
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(metadata["thinking_rewrite_reason"], "existing_budget_zero");
        assert_eq!(metadata["thinking_budget_previous_state"], "mixed");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "multiple");
        assert_eq!(metadata["thinking_budget_final_tokens"], "multiple");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_budget_observed_paths"],
            "thinking.budget_tokens=equal,extra_body.chat_template_kwargs.thinking_budget=zero"
        );
        assert_eq!(metadata["thinking_budget_rewritten_paths"], "none");
        assert_eq!(
            metadata["thinking_budget_zero_paths"],
            "extra_body.chat_template_kwargs.thinking_budget"
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
async fn default_sse_mode_buffers_non_stream_json_without_sse_framing() {
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
        Duration::from_secs(4),
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
    .expect("shielded JSON response should arrive after upstream aggregation")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("shielded default non-stream response should be JSON"),
        "application/json"
    );

    let body = response.text().await.expect("response body should be text");
    assert!(
        !body.starts_with(": llm-guard-proxy heartbeat"),
        "non-stream body must not start with SSE heartbeat: {body:?}"
    );
    assert!(
        !body.contains("event: final"),
        "non-stream body must not contain SSE final event: {body:?}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&body).expect("non-stream body should parse as JSON");
    assert_eq!(json["choices"][0]["message"]["content"], "Hello");

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
}

#[tokio::test]
async fn shielded_liveness_drop_records_current_attempt_abort() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=slow-shielded",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"drop-current"}]}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let heartbeat = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "shielded heartbeat").await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(body);

    let _observed = fake.recv_next().await;
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:aborted:downstream_body_dropped_before_eof:none"
    );
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "aborted");
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
}

#[tokio::test]
async fn shielded_liveness_drop_after_prior_retry_records_current_chain() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=loop-once-then-slow-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"drop-after-retry"}]}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let prefix = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "retry JSON prefix").await;
    assert_eq!(prefix, Bytes::from_static(b" \n"));
    let heartbeat = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "retry heartbeat").await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(body);

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&first_attempt.body));
    assert!(body_contains_retry_hint(&second_attempt.body));

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:loop_guard:loop_detected,2:aborted:downstream_body_dropped_before_eof:none"
    );
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("loop_guard"));
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "aborted");
    assert_eq!(
        attempts[1].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
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
            .expect("first request should use JSON"),
        "application/json"
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
    assert_eq!(rows[0].1, "non-stream-json");
    assert_eq!(rows[1].1, "non-stream-json");
    let first_metadata: serde_json::Value =
        serde_json::from_str(&rows[0].2).expect("first metadata should parse");
    let second_metadata: serde_json::Value =
        serde_json::from_str(&rows[1].2).expect("second metadata should parse");
    assert_eq!(first_metadata["repeat_input_matched"], "false");
    assert_eq!(first_metadata["downstream_liveness_mode"], "disabled");
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
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1

[loop_guard]
enabled = false
"#,
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
    let first_prefix = next_chunk(
        &mut first_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "first JSON prefix",
    )
    .await;
    assert_eq!(first_prefix, Bytes::from_static(b" \n"));
    let first_heartbeat = next_chunk(
        &mut first_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "first interval heartbeat",
    )
    .await;
    assert!(
        first_heartbeat == Bytes::from_static(b" \n"),
        "first JSON whitespace heartbeat should be a JSON-safe whitespace chunk"
    );
    drop(first_body);
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 2

[loop_guard]
enabled = false
"#,
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
    let second_prefix = next_chunk(
        &mut second_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "second JSON prefix",
    )
    .await;
    assert_eq!(second_prefix, Bytes::from_static(b" \n"));
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
        second_heartbeat == Bytes::from_static(b" \n"),
        "second JSON whitespace heartbeat should be a JSON-safe whitespace chunk"
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
            .expect("first request should use JSON"),
        "application/json"
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
            .expect("expired repeat should stay JSON"),
        "application/json"
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

#[test]
fn normalized_chat_fingerprint_distinguishes_root_thinking_token_budget_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("thinking_token_budget");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_extra_body_thinking_token_budget_for_repeat_detection()
{
    let base_body = chat_body_with_extra_body_thinking_token_budget(16);
    let changed_body = chat_body_with_extra_body_thinking_token_budget(32);
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

fn chat_body_with_extra_body_thinking_token_budget(value: u64) -> Bytes {
    Bytes::from(
        serde_json::json!({
            "model": "test-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.2,
            "extra_body": {
                "thinking_token_budget": value
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
async fn context_budget_preflight_allows_equal_window_and_rejects_chat_overflow() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[upstream.metadata]
context_length_override = 6
input_token_safety_margin = 1

[thinking]
mode = "passthrough"
"#,
    )
    .await;

    let allowed = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"a b c"}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("allowed proxy request should complete");
    assert_eq!(allowed.status(), StatusCode::OK);
    let _allowed_json = shielded_final_json(allowed).await;
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");

    let rejected = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"a b c d"}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("rejected proxy request should complete");
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    let error = response_json(rejected).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "messages");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message should be string")
            .contains("auto-compaction")
    );
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    assert_eq!(attempt_count, 1);
    let rejected_metadata_json: String = connection
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("rejected request metadata should be readable");
    let rejected_metadata: serde_json::Value =
        serde_json::from_str(&rejected_metadata_json).expect("metadata should parse");
    assert_eq!(rejected_metadata["context_budget_preflight"], "rejected");
    assert_eq!(rejected_metadata["context_budget_param"], "messages");
    assert_eq!(rejected_metadata["context_budget_window_tokens"], "6");
    assert_eq!(
        rejected_metadata["context_budget_total_estimate_tokens"],
        "7"
    );
    assert_eq!(rejected_metadata["upstream_profile"], "default");
    assert!(!rejected_metadata.to_string().contains("a b c d"));
}

#[tokio::test]
async fn context_budget_preflight_counts_chat_tool_definitions_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[upstream.metadata]
context_length_override = 6

[thinking]
mode = "passthrough"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ok"}],"tools":[{"type":"function","function":{"name":"lookup","description":"one two three four five six","parameters":{"type":"object","properties":{"city":{"type":"string","description":"target city"}},"required":["city"]}}}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "messages");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let rejected_metadata_json: String = connection
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("rejected request metadata should be readable");
    let rejected_metadata: serde_json::Value =
        serde_json::from_str(&rejected_metadata_json).expect("metadata should parse");
    assert_eq!(rejected_metadata["context_budget_preflight"], "rejected");
    assert_eq!(rejected_metadata["context_budget_param"], "messages");
    assert!(
        rejected_metadata["context_budget_total_estimate_tokens"]
            .as_str()
            .and_then(|tokens| tokens.parse::<u64>().ok())
            .is_some_and(|tokens| tokens > 6)
    );
    assert!(!rejected_metadata_json.contains("one two three four five six"));
    assert!(!rejected_metadata_json.contains("target city"));
}

#[tokio::test]
async fn context_budget_preflight_rejects_completions_prompt_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
context_length_override = 3
",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-completion","prompt":"one two three","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "prompt");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn context_budget_preflight_rejects_unbroken_prompt_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
context_length_override = 100
",
    )
    .await;
    let long_prompt = "x".repeat(1_000);
    let body = format!(r#"{{"model":"test-completion","prompt":"{long_prompt}","max_tokens":1}}"#);

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "prompt");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
}

#[tokio::test]
async fn hot_reloaded_profile_safety_margin_changes_context_preflight() {
    let mut fake = FakeUpstream::spawn().await;
    let fake_base_url = fake.base_url.clone();
    let profile_config = |safety_margin: u32| {
        format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{fake_base_url}"
match_models = ["aeon-ultimate"]

[upstreams.metadata]
context_length_override = 6
input_token_safety_margin = {safety_margin}

[upstreams.thinking]
mode = "passthrough"
"#,
        )
    };
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &profile_config(0),
    )
    .await;
    let body = r#"{"model":"aeon-ultimate","messages":[{"role":"user","content":"a b c d"}],"max_tokens":1}"#;

    let allowed = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("allowed proxy request should complete");
    assert_eq!(allowed.status(), StatusCode::OK);
    let _allowed_json = shielded_final_json(allowed).await;
    let _observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &profile_config(1),
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("profile safety margin reload should succeed");
    assert!(outcome.applied);
    assert!(outcome.restart_required_changes.is_empty());

    let rejected = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("rejected proxy request should complete");
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    let error = response_json(rejected).await;
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
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
async fn generic_stream_timeout_records_failed_request_and_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "request_timeout_ms = 100\n",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");

    let mut body = response.into_body().into_data_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first JSON chunk").await;
    assert_eq!(first, Bytes::from_static(LONG_JSON_FIRST_CHUNK));
    let timeout_item = timeout(Duration::from_secs(1), body.next())
        .await
        .expect("upstream timeout should surface before the delayed second chunk")
        .expect("body stream should yield an upstream timeout item");
    assert!(
        timeout_item.is_err(),
        "delayed upstream body should fail under the configured timeout"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure")),
        "request error should use bounded timeout kind: {:?}",
        request_row.error_reason
    );
    assert!(
        attempt_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure")),
        "attempt error should use bounded timeout kind: {:?}",
        attempt_row.error_reason
    );
}

#[tokio::test]
async fn shielded_upstream_timeout_returns_bounded_gateway_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
request_timeout_ms = 100

[heartbeat]
mode = "disabled"

[retry]
enabled = false
"#,
    )
    .await;

    let response = timeout(
        Duration::from_secs(2),
        proxy_handler(
            State(proxy.state.clone()),
            shielded_chat_request(
                "/v1/chat/completions?test=slow-shielded",
                r#"{"model":"test-chat","messages":[{"role":"user","content":"timeout"}]}"#,
            ),
        ),
    )
    .await
    .expect("shielded timeout response should be bounded");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("timeout error body should read");
    let body = String::from_utf8(body.to_vec()).expect("timeout error body should be UTF-8");
    assert!(body.contains("llm_guard_attempt_timeout"));
    assert!(body.contains("timeout_failure"));
    assert_safe_operational_text("shielded timeout body", &body);

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure"))
    );
    assert!(
        attempt_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure"))
    );
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
    let upstream = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;

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
async fn queued_generation_request_cancellation_does_not_buffer_or_forward_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json"),
    )
    .await;

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
    let second = tokio::spawn(proxy_handler(State(proxy.state.clone()), second_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queued requests must not be body-buffered before permit admission"
    );
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second request should still be waiting for capacity"
    );

    second.abort();
    match second.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "queued request should be cancelled before upstream dispatch, got {}",
            response.status()
        ),
    }
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "cancelled queued request must not poll its body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(first_response);
}

#[tokio::test]
async fn saturated_generation_requests_wait_for_in_flight_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=one"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream and hold the only permit");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=one"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let second_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"queued"}"#))
        }
    }));
    let second_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?slot=queued")
        .header(CONTENT_TYPE, "application/json")
        .body(second_body)
        .expect("second request should build");
    let second = tokio::spawn(proxy_handler(State(proxy.state.clone()), second_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queued requests must not be body-buffered before capacity is available"
    );
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second request should wait for capacity instead of returning a 503"
    );

    drop(first_response);

    let second_response = second
        .await
        .expect("queued request task should complete after capacity is released");
    assert_eq!(second_response.status(), StatusCode::OK);
    assert!(body_polled.load(Ordering::SeqCst));
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("queued request should reach upstream after capacity is available");
    assert_eq!(second_observed.method, Method::POST);
    assert_eq!(
        second_observed.path_and_query,
        "/v1/completions?slot=queued"
    );
}

#[tokio::test]
async fn restricted_embedding_listener_accepts_embedding_and_rejects_other_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let state = proxy.state.for_listener(listener);

    let accepted = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/embeddings",
            br#"{"model":"embedding-model","input":"hello"}"#,
        ),
    )
    .await;
    assert_eq!(accepted.status(), StatusCode::OK);
    let _body = to_bytes(accepted.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("accepted body should read");
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/embeddings");

    let rejected_chat = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/chat/completions",
            br#"{"model":"chat-model","messages":[{"role":"user","content":"hello"}]}"#,
        ),
    )
    .await;
    assert_eq!(rejected_chat.status(), StatusCode::BAD_REQUEST);
    let rejected_chat_body = to_bytes(rejected_chat.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rejection body should read");
    let rejected_chat_json: serde_json::Value =
        serde_json::from_slice(&rejected_chat_body).expect("rejection should be JSON");
    assert_eq!(
        rejected_chat_json["error"]["type"],
        "listener_upstream_not_allowed"
    );

    let rejected_rerank = proxy_handler(
        State(state),
        json_post_request(
            "/v1/rerank",
            br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
        ),
    )
    .await;
    assert_eq!(rejected_rerank.status(), StatusCode::BAD_REQUEST);
    let _body = to_bytes(rejected_rerank.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rerank rejection body should read");
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn restricted_listener_denial_bounds_untrusted_model_in_response_and_metadata() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let oversized_model = format!("chat-model-{}", "x".repeat(4096));
    let body = format!(
        r#"{{"model":"{oversized_model}","messages":[{{"role":"user","content":"hello"}}]}}"#
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("chat request should build");

    let response = proxy_handler(State(proxy.state.for_listener(listener)), request).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rejection body should read");
    let response_body =
        String::from_utf8(response_body.to_vec()).expect("rejection body should be utf-8");
    let rejection: serde_json::Value =
        serde_json::from_str(&response_body).expect("rejection should be JSON");
    assert_eq!(rejection["error"]["type"], "listener_upstream_not_allowed");
    assert!(response_body.len() < 1024);
    assert!(!response_body.contains(&oversized_model));
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (Option<String>, String, String) = connection
        .query_row(
            "SELECT model_id, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("denied request row should exist");
    let persisted = format!("{} {}", request_row.0.unwrap_or_default(), request_row.1);
    assert!(persisted.len() < 1024);
    assert!(!persisted.contains(&oversized_model));
    assert!(!request_row.2.contains(&oversized_model));
}

#[tokio::test]
async fn aggregate_listener_accepts_chat_embeddings_and_rerank_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "aggregate");
    let state = proxy.state.for_listener(listener);

    for (request, expected_path) in [
        (
            json_post_request(
                "/v1/chat/completions",
                br#"{"model":"chat-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            ),
            "/v1/chat/completions",
        ),
        (
            json_post_request(
                "/v1/embeddings",
                br#"{"model":"embedding-model","input":"hello"}"#,
            ),
            "/v1/embeddings",
        ),
        (
            json_post_request(
                "/v1/rerank",
                br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
            ),
            "/v1/rerank",
        ),
    ] {
        let response = proxy_handler(State(state.clone()), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("response body should read");
        let observed = fake.recv_next().await;
        assert_eq!(observed.path_and_query, expected_path);
    }
}

#[tokio::test]
async fn restricted_models_request_filters_to_listener_reachable_models() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model"]);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_filters_when_metadata_enrichment_is_disabled() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[upstreams.metadata]
discovery_enabled = false
enrich_responses = false

[[upstreams]]
name = "rerank"
base_url = "{0}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["embedding"]
"#,
            fake.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model"]);
    assert_eq!(json["data"][0].get("context_length"), None);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_filters_to_all_allowed_listener_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");
    let state = proxy.state.for_listener(listener);

    let response = proxy_handler(
        State(state.clone()),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );

    for (request, expected_path) in [
        (
            json_post_request(
                "/v1/embeddings",
                br#"{"model":"embedding-model","input":"hello"}"#,
            ),
            "/v1/embeddings",
        ),
        (
            json_post_request(
                "/v1/rerank",
                br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
            ),
            "/v1/rerank",
        ),
    ] {
        let response = proxy_handler(State(state.clone()), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("response body should read");
        let observed = fake.recv_next().await;
        assert_eq!(observed.path_and_query, expected_path);
    }
}

#[tokio::test]
async fn aggregate_models_request_fetches_all_configured_upstream_profiles() {
    let mut chat = FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_CHAT_MODELS_BODY).await;
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &chat.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "aggregate");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert!(model_ids.contains(&"chat-model"));
    assert!(model_ids.contains(&"embedding-model"));
    assert!(model_ids.contains(&"rerank-model"));
    let observed = chat.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let observed = embedding.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let observed = rerank.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_fetches_and_merges_distinct_allowed_upstreams() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let models = json["data"].as_array().expect("data should be an array");
    let model_ids = models
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    assert_eq!(models[0]["first"], "embedding");
    assert!(model_ids.iter().all(|model_id| *model_id != "chat-model"));
    let embedding_request = embedding.recv_next().await;
    assert_eq!(
        embedding_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let rerank_request = rerank.recv_next().await;
    assert_eq!(
        rerank_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn merged_models_enrichment_uses_each_allowed_profile_metadata_config() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[upstreams.metadata]
discovery_enabled = false
enrich_responses = false

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[upstreams.metadata]
context_length_override = 12345

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let models = json["data"].as_array().expect("data should be an array");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["id"], "embedding-model");
    assert_eq!(models[0].get("context_length"), None);
    assert_eq!(models[1]["id"], "rerank-model");
    assert_normalized_context_fields(&models[1], 12_345);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_records_distinct_observability_attempts_per_upstream() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_ne!(attempts[0].attempt_id, attempts[1].attempt_id);
    assert_eq!(attempts[0].request_id, attempts[1].request_id);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[1].status, "succeeded");
    assert_eq!(
        attempts[0].request_metadata["upstream_profile"],
        "embedding"
    );
    assert_eq!(attempts[1].request_metadata["upstream_profile"], "rerank");
}

#[tokio::test]
async fn restricted_models_request_skips_invalid_first_body_when_merging_distinct_upstreams() {
    let mut invalid = FakeUpstream::spawn_with_models_body(r#"{"error":"not a model list"}"#).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &invalid.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "invalid"
base_url = "{0}"
match_models = ["invalid-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "invalid-rerank"
bind_host = "127.0.0.1"
port = 18006
allowed_upstreams = ["invalid", "rerank"]
"#,
            invalid.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "invalid-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["rerank-model"]);
    let invalid_request = invalid.recv_next().await;
    assert_eq!(
        invalid_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let rerank_request = rerank.recv_next().await;
    assert_eq!(
        rerank_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn merged_models_response_uses_proxy_owned_success_headers_for_valid_body() {
    let mut rate_limited = FakeUpstream::spawn_with_models_response(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{"type":"rate_limit","message":"slow down"}}"#,
        "rate-limited-models",
    )
    .await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &rate_limited.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "rate-limited"
base_url = "{0}"
match_models = ["rate-limited-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "mixed-models"
bind_host = "127.0.0.1"
port = 18006
allowed_upstreams = ["rate-limited", "rerank"]
"#,
            rate_limited.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "mixed-models");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(response.headers().get(RETRY_AFTER), None);
    assert_eq!(response.headers().get("x-upstream-endpoint"), None);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["rerank-model"]);
    assert_eq!(
        rate_limited.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_merges_implicit_default_and_named_allowed_upstream() {
    let mut shared = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &shared.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[listeners]]
name = "default-embedding"
bind_host = "127.0.0.1"
port = 18007
allowed_upstreams = ["default", "embedding"]
"#,
            shared.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "default-embedding");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(
        model_ids,
        vec!["chat-model", "embedding-model", "rerank-model"]
    );
    let default_request = shared.recv_next().await;
    assert_eq!(
        default_request.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_excludes_same_base_url_models_routed_to_disallowed_profiles() {
    let mut shared = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &shared.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{0}"
match_models = ["rerank-model"]

[[listeners]]
name = "default-embedding"
bind_host = "127.0.0.1"
port = 18007
allowed_upstreams = ["default", "embedding"]
"#,
            shared.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "default-embedding");
    let state = proxy.state.for_listener(listener);

    let response = proxy_handler(
        State(state.clone()),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["chat-model", "embedding-model"]);
    let response = proxy_handler(
        State(state),
        json_post_request(
            "/v1/rerank",
            br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let default_request = shared.recv_next().await;
    assert_eq!(
        default_request.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_uses_each_upstream_profile_timeout_when_merging() {
    let mut fast =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut slow = FakeUpstream::spawn_with_models_body_and_delay(
        DISTINCT_UPSTREAM_SLOW_MODELS_BODY,
        Duration::from_millis(150),
    )
    .await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fast.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fast"
base_url = "{0}"
match_models = ["embedding-model"]
request_timeout_ms = 50

[[upstreams]]
name = "slow"
base_url = "{1}"
match_models = ["slow-model"]
request_timeout_ms = 1000

[[listeners]]
name = "fast-slow"
bind_host = "127.0.0.1"
port = 18008
allowed_upstreams = ["fast", "slow"]
"#,
            fast.base_url, slow.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "fast-slow");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "slow-model"]);
    let fast_request = fast.recv_next().await;
    assert_eq!(
        fast_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let slow_request = slow.recv_next().await;
    assert_eq!(
        slow_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn observability_records_listener_and_selected_upstream_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        json_post_request(
            "/v1/embeddings",
            br#"{"model":"embedding-model","input":"hello"}"#,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("response body should read");
    let _observed = fake.recv_next().await;

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["listener_name"], "embedding-legacy");
    assert_eq!(request_metadata["listener_port"], "18002");
    assert_eq!(request_metadata["upstream_profile"], "embedding");
    assert_eq!(attempt_metadata["listener_name"], "embedding-legacy");
    assert_eq!(attempt_metadata["upstream_profile"], "embedding");
}

#[tokio::test]
async fn matched_upstream_profile_has_independent_generation_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 0
generation_queue_timeout_ms = 50

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let default_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=default"),
    )
    .await;
    assert_eq!(default_response.status(), StatusCode::OK);
    let default_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should hold the default generation capacity");
    assert_eq!(
        default_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=default"
    );

    let first_embedding = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=embedding-one"),
    )
    .await;
    let second_embedding = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=embedding-two"),
    )
    .await;

    assert_eq!(first_embedding.status(), StatusCode::OK);
    assert_eq!(second_embedding.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first matched-profile request should reach upstream despite default saturation");
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("second matched-profile request should use profile capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding-one"
    );
    assert_eq!(
        second_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding-two"
    );

    drop(first_embedding);
    drop(second_embedding);
    drop(default_response);
}

#[tokio::test]
async fn profile_limit_mode_bounds_body_routing_before_buffering() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let (active_request, active_body_polled) =
        tracked_pending_json_request("/v1/completions?slot=active");
    let active = tokio::spawn(proxy_handler(State(proxy.state.clone()), active_request));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        active_body_polled.load(Ordering::SeqCst),
        "active routing request should hold the body-routing permit while reading its body"
    );

    let (queued_request, queued_body_polled) =
        tracked_pending_json_request("/v1/completions?slot=queued");
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "queued routing request must not read its body before routing capacity is available"
    );
    assert!(
        !queued.is_finished(),
        "queued routing request should occupy the bounded routing queue"
    );

    let (overflow_request, overflow_body_polled) =
        tracked_json_request("/v1/completions?slot=overflow", br#"{"prompt":"overflow"}"#);
    let overflow_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    )
    .await
    .expect("routing queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let overflow_body = to_bytes(overflow_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("routing queue-full body should read");
    let overflow_body =
        String::from_utf8(overflow_body.to_vec()).expect("routing queue-full body should be utf-8");
    assert!(
        overflow_body.contains("proxy_generation_queue_full"),
        "routing queue-full error should identify admission failure: {overflow_body}"
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "routing queue-full rejection must not read the overflow body"
    );
    assert_no_upstream_request(&mut fake).await;

    queued.abort();
    active.abort();
    assert!(
        queued
            .await
            .expect_err("queued request should be aborted")
            .is_cancelled()
    );
    assert!(
        active
            .await
            .expect_err("active request should be aborted")
            .is_cancelled()
    );
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "aborted queued routing request must not read its body"
    );
}

#[tokio::test]
async fn merged_models_request_records_successful_prior_attempt_when_later_fetch_fails() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let broken = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "broken"
base_url = "{1}"
match_models = ["broken-model"]
request_timeout_ms = 50

[[listeners]]
name = "embedding-broken"
bind_host = "127.0.0.1"
port = 18010
allowed_upstreams = ["embedding", "broken"]
"#,
            embedding.base_url, broken.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-broken");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[1].status, "failed");
    assert_ne!(attempts[0].attempt_id, attempts[1].attempt_id);
    assert_eq!(attempts[0].request_id, attempts[1].request_id);
    assert_eq!(
        attempts[0].request_metadata["upstream_profile"],
        "embedding"
    );
    assert_eq!(attempts[1].request_metadata["upstream_profile"], "broken");
}

#[tokio::test]
async fn profile_limit_models_bypass_generation_saturation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_control_plane_in_flight_requests = 2

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let generation_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;
    assert_eq!(generation_response.status(), StatusCode::OK);
    let generation_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("generation request should hold default generation capacity");
    assert_eq!(
        generation_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let model_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata&slot=profile-limits"),
        ),
    )
    .await
    .expect("models request should bypass generation saturation in profile-limit mode");
    assert_eq!(model_response.status(), StatusCode::OK);
    let model_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream in profile-limit mode");
    assert_eq!(
        model_observed.path_and_query,
        "/v1/models?test=model-metadata&slot=profile-limits"
    );

    drop(model_response);
    drop(generation_response);
}

#[tokio::test]
async fn profile_wait_releases_body_routing_capacity_for_other_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let default_active = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=default-active"),
    )
    .await;
    assert_eq!(default_active.status(), StatusCode::OK);
    let default_active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should hold default generation capacity");
    assert_eq!(
        default_active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=default-active"
    );

    let (default_queued_request, default_queued_body_polled) = tracked_json_request(
        "/v1/completions?slot=default-queued",
        br#"{"prompt":"queued default"}"#,
    );
    let default_queued = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        default_queued_request,
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        default_queued_body_polled.load(Ordering::SeqCst),
        "default queued request should read its body before occupying the default profile queue"
    );
    assert!(
        !default_queued.is_finished(),
        "default queued request should wait on default generation capacity"
    );

    let embedding_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            embedding_request("/v1/embeddings?test=long-json&slot=embedding"),
        ),
    )
    .await
    .expect("matched profile request must not be blocked by default profile admission wait");
    assert_eq!(embedding_response.status(), StatusCode::OK);
    let embedding_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("embedding request should reach upstream while default profile waits");
    assert_eq!(
        embedding_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding"
    );

    default_queued.abort();
    assert!(
        default_queued
            .await
            .expect_err("default queued request should be aborted")
            .is_cancelled()
    );
    drop(embedding_response);
    drop(default_active);
}

#[tokio::test]
async fn profile_limit_reload_off_exchanges_body_routing_for_global_generation_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let initial_upstream_config = format!(
        r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
        fake.base_url
    );
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &initial_upstream_config,
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/embeddings?test=long-json&slot=reload-first",
        br#"{"model":"embedding-model","input":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "first request body should start reading",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 1,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit removal reload should succeed");
    assert!(outcome.applied);

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("first request should finish admission after reload")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream after reload");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=reload-first"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/embeddings?slot=reload-overflow",
        br#"{"model":"embedding-model","input":"overflow"}"#,
    );
    let overflow_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    )
    .await
    .expect("overflow response should be bounded");
    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "overflow request should be rejected by global generation admission before body read"
    );
    assert_no_upstream_request(&mut fake).await;

    drop(overflow_response);
    drop(first_response);
}

#[tokio::test]
async fn profile_limit_reload_on_keeps_global_generation_permit_for_default_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/completions?test=long-json&slot=reload-on-default",
        br#"{"prompt":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "default request body should start reading before reload",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 1,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit addition reload should succeed");
    assert!(outcome.applied);

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("first request should finish admission after reload")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should keep its global permit and reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/completions?test=long-json&slot=reload-on-default"
    );

    drop(first_response);
}

#[tokio::test]
async fn profile_limit_reload_on_keeps_global_generation_permit_for_newly_limited_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        2,
        &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/embeddings?test=long-json&slot=reload-on-matched-first",
        br#"{"model":"embedding-model","input":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "matched profile request body should start reading before reload",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 2,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit addition reload should succeed");
    assert!(outcome.applied);

    let active_profile_response = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-active"),
    )
    .await;
    assert_eq!(active_profile_response.status(), StatusCode::OK);
    let active_profile_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("new profile-limited request should hold profile capacity");
    assert_eq!(
        active_profile_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-active"
    );

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("pre-reload request should not be re-admitted into the new profile limiter")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("pre-reload matched request should keep its global permit and reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=reload-on-matched-first"
    );

    drop(active_profile_response);
    drop(first_response);
}

#[tokio::test]
async fn generation_queue_full_fails_without_body_buffering_or_upstream_forward() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 1000\n",
    )
    .await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should hold generation capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let (queued_request, queued_body_polled) =
        tracked_json_request("/v1/completions?slot=queued", br#"{"prompt":"queued"}"#);
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "queued request must not read its body before capacity is available"
    );
    assert!(
        !queued.is_finished(),
        "first queued request should occupy the bounded queue"
    );

    let (overflow_request, overflow_body_polled) =
        tracked_json_request("/v1/completions?slot=overflow", br#"{"prompt":"overflow"}"#);
    let overflow_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    )
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let overflow_body = to_bytes(overflow_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("queue-full body should read");
    let overflow_body =
        String::from_utf8(overflow_body.to_vec()).expect("queue-full body should be utf-8");
    assert!(
        overflow_body.contains("proxy_generation_queue_full"),
        "queue-full error should identify admission failure: {overflow_body}"
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "queue-full rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;

    queued.abort();
    match queued.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "queued request should still be waiting before active response drops, got {}",
            response.status()
        ),
    }
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "aborted queued request must not read its body"
    );
    drop(first_response);
}

#[tokio::test]
async fn generation_queue_full_returns_configured_429_status() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 0\ngeneration_queue_timeout_ms = 1000\ngeneration_queue_full_status = 429\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=configured-status-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=configured-status-active"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/completions?slot=configured-status-overflow",
        br#"{"prompt":"overflow"}"#,
    );
    let overflow_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    )
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "configured queue-full rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(active_response);
}

#[tokio::test]
async fn generation_queue_full_returns_configured_retry_after() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 0\ngeneration_queue_timeout_ms = 1000\ngeneration_queue_retry_after_secs = 30\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=configured-retry-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=configured-retry-active"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/completions?slot=configured-retry-overflow",
        br#"{"prompt":"overflow"}"#,
    );
    let overflow_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    )
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("30")
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "configured Retry-After queue-full rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(active_response);
}

#[tokio::test]
async fn queued_request_cancelled_on_downstream_disconnect_never_reaches_upstream() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=cancel-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=cancel-active"
    );

    let (cancelled_request, cancelled_body_polled) = tracked_json_request(
        "/v1/completions?slot=cancelled-queued",
        br#"{"prompt":"cancelled"}"#,
    );
    let cancelled = tokio::spawn(proxy_handler(State(proxy.state.clone()), cancelled_request));
    sleep(Duration::from_millis(50)).await;
    assert!(!cancelled_body_polled.load(Ordering::SeqCst));
    assert!(!cancelled.is_finished());

    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("queued request future should be cancelled")
            .is_cancelled()
    );
    assert!(!cancelled_body_polled.load(Ordering::SeqCst));
    let cancel_record = read_latest_aborted_request_metadata(&proxy);
    assert_eq!(
        cancel_record.abort_reason.as_deref(),
        Some("downstream_disconnected_while_queued")
    );
    assert_eq!(
        cancel_record.request_metadata["admission_outcome"],
        "queue_cancelled"
    );
    assert_eq!(cancel_record.request_metadata["path"], "/v1/completions");
    assert_no_upstream_request(&mut fake).await;

    let (replacement_request, replacement_body_polled) = tracked_json_request(
        "/v1/completions?slot=replacement-queued",
        br#"{"prompt":"replacement"}"#,
    );
    let replacement = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        replacement_request,
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(
        !replacement_body_polled.load(Ordering::SeqCst),
        "replacement should be queued before capacity is released"
    );
    assert!(
        !replacement.is_finished(),
        "replacement should queue, proving the cancelled request left the queue"
    );

    drop(active_response);
    let replacement_response = timeout(STREAM_COMPLETION_TIMEOUT, replacement)
        .await
        .expect("replacement should complete after capacity is released")
        .expect("replacement task should not panic");
    assert_eq!(replacement_response.status(), StatusCode::OK);
    let _body = to_bytes(replacement_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("replacement body should read");
    let replacement_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("replacement request should reach upstream after release");
    assert_eq!(
        replacement_observed.path_and_query,
        "/v1/completions?slot=replacement-queued"
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn queued_request_cancelled_on_downstream_disconnect_per_profile_limiter() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        16,
        &format!(
            r#"max_queued_generation_requests = 0
generation_queue_timeout_ms = 5000

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 1
"#,
            fake.base_url
        ),
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-cancel-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active profile request should hold profile capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-cancel-active"
    );

    let cancelled = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-cancelled-queued"),
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(!cancelled.is_finished());
    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("queued profile request future should be cancelled")
            .is_cancelled()
    );
    let cancel_record = read_latest_aborted_request_metadata(&proxy);
    assert_eq!(
        cancel_record.request_metadata["admission_outcome"],
        "queue_cancelled"
    );
    assert_eq!(cancel_record.request_metadata["path"], "/v1/embeddings");
    assert_no_upstream_request(&mut fake).await;

    let replacement = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-replacement-queued"),
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(
        !replacement.is_finished(),
        "replacement should queue in the profile limiter"
    );

    drop(active_response);
    let replacement_response = timeout(STREAM_COMPLETION_TIMEOUT, replacement)
        .await
        .expect("profile replacement should complete after capacity is released")
        .expect("profile replacement task should not panic");
    assert_eq!(replacement_response.status(), StatusCode::OK);
    let _body = to_bytes(replacement_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("profile replacement body should read");
    let replacement_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("profile replacement request should reach upstream after release");
    assert_eq!(
        replacement_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-replacement-queued"
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn high_queue_capacity_allows_c32_without_immediate_rejection() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 256\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=c32-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=c32-active"
    );

    let mut queued = Vec::new();
    for index in 0..32 {
        let uri = format!("/v1/completions?slot=c32-{index}");
        let (request, body_polled) = tracked_json_request(&uri, br#"{"prompt":"queued"}"#);
        let handle = tokio::spawn(proxy_handler(State(proxy.state.clone()), request));
        queued.push((handle, body_polled));
    }
    sleep(Duration::from_millis(50)).await;

    for (index, (handle, body_polled)) in queued.iter().enumerate() {
        assert!(
            !handle.is_finished(),
            "queued request {index} should wait instead of being rejected"
        );
        assert!(
            !body_polled.load(Ordering::SeqCst),
            "queued request {index} must not read its body before admission"
        );
    }
    assert_no_upstream_request(&mut fake).await;

    for (handle, _body_polled) in queued {
        handle.abort();
        assert!(
            handle
                .await
                .expect_err("queued c32 request should be aborted")
                .is_cancelled()
        );
    }
    drop(active_response);
}

fn tracked_json_request(uri: &str, body: &'static [u8]) -> (Request<Body>, Arc<AtomicBool>) {
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(body))
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("tracked json request should build");
    (request, polled)
}

fn tracked_pending_json_request(uri: &str) -> (Request<Body>, Arc<AtomicBool>) {
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            std::future::pending::<Result<Bytes, std::convert::Infallible>>().await
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("tracked pending json request should build");
    (request, polled)
}

fn controlled_json_request(
    uri: &str,
    body: &'static [u8],
) -> (Request<Body>, oneshot::Sender<()>, Arc<AtomicBool>) {
    let (release, released) = oneshot::channel();
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            let _ = released.await;
            Ok::<_, std::convert::Infallible>(Bytes::from_static(body))
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("controlled json request should build");
    (request, release, polled)
}

async fn wait_for_flag(flag: &AtomicBool, label: &str) {
    timeout(Duration::from_secs(1), async {
        while !flag.load(Ordering::SeqCst) {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
}

#[tokio::test]
async fn generation_queue_timeout_fails_without_body_buffering_or_upstream_forward() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 20\n",
    )
    .await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should hold generation capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let queued_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"timeout"}"#))
        }
    }));
    let queued_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?slot=timeout")
        .header(CONTENT_TYPE, "application/json")
        .body(queued_body)
        .expect("queued request should build");
    let queued_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), queued_request),
    )
    .await
    .expect("queue-timeout response should be bounded");

    assert_eq!(queued_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        queued_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let queued_body = to_bytes(queued_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("queue-timeout body should read");
    let queued_body =
        String::from_utf8(queued_body.to_vec()).expect("queue-timeout body should be utf-8");
    assert!(
        queued_body.contains("proxy_generation_queue_timeout"),
        "queue-timeout error should identify admission failure: {queued_body}"
    );
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queue-timeout rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(first_response);
}

#[tokio::test]
async fn models_bypass_generation_saturation_but_keep_control_plane_bound() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_control_plane_in_flight_requests = 1\n",
    )
    .await;
    let generation_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(generation_response.status(), StatusCode::OK);
    let generation_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("generation request should hold generation capacity");
    assert_eq!(
        generation_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let first_model_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata-large&slot=one"),
    )
    .await;
    assert_eq!(first_model_response.status(), StatusCode::OK);
    let first_model_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should bypass generation capacity");
    assert_eq!(
        first_model_observed.path_and_query,
        "/v1/models?test=model-metadata-large&slot=one"
    );

    let second_model_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata&slot=two"),
        ),
    )
    .await
    .expect("control-plane limit response should be bounded");
    assert_eq!(
        second_model_response.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        second_model_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let second_model_body = to_bytes(second_model_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("control-plane limit body should read");
    let second_model_body = String::from_utf8(second_model_body.to_vec())
        .expect("control-plane limit body should be utf-8");
    assert!(
        second_model_body.contains("proxy_control_plane_in_flight_limit_exceeded"),
        "control-plane error should identify admission failure: {second_model_body}"
    );
    assert_no_upstream_request(&mut fake).await;

    drop(first_model_response);
    drop(generation_response);
}

#[tokio::test]
async fn in_flight_limit_hot_reload_updates_admission_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 2).await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        1,
        "",
    );
    let outcome = proxy.manager.reload().expect("limit reload should succeed");
    assert!(outcome.applied);
    assert!(
        outcome.restart_required_changes.is_empty(),
        "in-flight limit should be safe to hot reload"
    );

    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=one"),
    )
    .await;
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=one"
    );

    let second = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=two"),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second generation request should wait while live limit is one"
    );

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        2,
        "",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("limit increase should reload");
    assert!(outcome.applied);
    assert!(
        outcome.restart_required_changes.is_empty(),
        "limit increase should not require process restart"
    );

    let second_response = timeout(STREAM_HEADER_TIMEOUT, second)
        .await
        .expect("queued request should finish after limit increase")
        .expect("queued request task should join");
    assert_eq!(second_response.status(), StatusCode::OK);
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("second request should reach upstream after limit increase");
    assert_eq!(
        second_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=two"
    );

    let third = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=three"),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !third.is_finished(),
        "third generation request should wait while both live slots are held"
    );
    third.abort();
    match third.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "third request should still be queued while both slots are held, got {}",
            response.status()
        ),
    }

    drop(first_response);
    drop(second_response);
}

#[tokio::test]
async fn graceful_shutdown_waits_for_in_flight_response_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("shutdown test listener should bind");
    let addr = listener
        .local_addr()
        .expect("shutdown test address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = proxy.state.clone();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });

    let response = proxy
        .client
        .get(format!("http://{addr}/v1/embeddings?test=long-json"))
        .send()
        .await
        .expect("long response request should get headers");
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("long response should reach upstream before shutdown");
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    sleep(Duration::from_millis(100)).await;
    assert!(
        !server.is_finished(),
        "graceful shutdown should wait for the in-flight response body"
    );

    drop(response);
    timeout(Duration::from_secs(2), server)
        .await
        .expect("server should exit after response is dropped")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");
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
            attempts: Vec::new(),
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

async fn next_chunk<S, E>(body: &mut S, wait: Duration, label: &str) -> Bytes
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    timeout(wait, body.next())
        .await
        .unwrap_or_else(|_| panic!("{label} should arrive before timeout"))
        .unwrap_or_else(|| panic!("{label} should not end the stream"))
        .unwrap_or_else(|error| panic!("{label} should not fail: {error}"))
}

async fn collect_stream_text<S, E>(body: &mut S, wait: Duration) -> String
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    let mut bytes = Vec::new();
    loop {
        match timeout(wait, body.next()).await {
            Ok(Some(Ok(chunk))) => bytes.extend_from_slice(&chunk),
            Ok(Some(Err(error))) => panic!("stream should not fail: {error}"),
            Ok(None) => break,
            Err(error) => panic!("stream should finish before timeout: {error}"),
        }
    }
    String::from_utf8(bytes).expect("stream body should be UTF-8")
}

fn openai_sse_json_chunks(text: &str) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();
    for event in text.split("\n\n") {
        let mut data = String::new();
        for line in event.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.trim_start());
            }
        }
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        chunks.push(serde_json::from_str(data).unwrap_or_else(|error| {
            panic!("OpenAI SSE data should parse as JSON: {error}; data={data}")
        }));
    }
    chunks
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

async fn response_json(response: reqwest::Response) -> serde_json::Value {
    let body = response.text().await.expect("body should be readable");
    serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("response body should parse as JSON: {error}; body={body}"))
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

struct AbortedRequestMetadata {
    abort_reason: Option<String>,
    request_metadata: serde_json::Value,
}

fn read_latest_aborted_request_metadata(proxy: &ProxyFixture) -> AbortedRequestMetadata {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let (http_status, abort_reason, request_metadata_json): (Option<i64>, Option<String>, String) =
        connection
            .query_row(
                "SELECT http_status, abort_reason, request_metadata_json \
             FROM requests WHERE status = 'aborted' ORDER BY rowid DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("aborted request row should exist");
    assert_eq!(
        http_status, None,
        "queued cancellation should not invent an HTTP response status"
    );
    let request_metadata =
        serde_json::from_str(&request_metadata_json).expect("request metadata should parse");
    AbortedRequestMetadata {
        abort_reason,
        request_metadata,
    }
}

async fn post_chat_and_observe_body(
    proxy: &ProxyFixture,
    fake: &mut FakeUpstream,
    body: &'static [u8],
) -> serde_json::Value {
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(Bytes::from_static(body))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    serde_json::from_slice(&observed.body).expect("upstream body should be JSON")
}

fn empty_get_request(uri: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .expect("GET request should build")
}

fn json_post_request(uri: &'static str, body: &'static [u8]) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("JSON request should build")
}

fn embedding_request(uri: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"embedding-model","input":"capacity test"}"#,
        ))
        .expect("embedding request should build")
}

fn listener_config(proxy: &ProxyFixture, name: &str) -> ListenerConfig {
    proxy
        .manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed")
        .listeners
        .into_iter()
        .find(|listener| listener.name == name)
        .unwrap_or_else(|| panic!("listener {name} should exist"))
}

fn multi_listener_profile_config(upstream_base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "embedding"
base_url = "{upstream_base_url}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{upstream_base_url}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["embedding"]

[[listeners]]
name = "reranker-legacy"
bind_host = "127.0.0.1"
port = 18003
allowed_upstreams = ["rerank"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005
"#
    )
}

fn shielded_chat_request(uri: &'static str, body: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("shielded chat request should build")
}

#[derive(Debug)]
struct ForwardedRecordRow {
    status: String,
    http_status: i64,
    error_reason: Option<String>,
    abort_reason: Option<String>,
    response_metadata: serde_json::Value,
}

fn read_single_forwarded_request_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, abort_reason, response_metadata_json FROM requests",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("request row should exist");
    let response_metadata =
        serde_json::from_str(&row.4).expect("request response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        error_reason: row.2,
        abort_reason: row.3,
        response_metadata,
    }
}

fn read_single_forwarded_attempt_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, abort_reason, response_metadata_json FROM attempts",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("attempt row should exist");
    let response_metadata =
        serde_json::from_str(&row.4).expect("attempt response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        error_reason: row.2,
        abort_reason: row.3,
        response_metadata,
    }
}

fn assert_forwarded_abort_recorded(proxy: &ProxyFixture) {
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempt_row.status, "aborted");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
}

#[derive(Debug)]
struct ObservabilityRow {
    status: String,
    response_metadata: serde_json::Value,
}

#[derive(Debug)]
struct AttemptChainRow {
    attempt_number: u32,
    status: String,
    retry_reason: Option<String>,
    abort_reason: Option<String>,
    response_metadata: serde_json::Value,
}

#[derive(Debug)]
struct AttemptRequestMetadataRow {
    attempt_id: String,
    request_id: String,
    attempt_number: u32,
    status: String,
    request_metadata: serde_json::Value,
}

#[derive(Debug)]
struct EvidenceAttemptRow {
    role: String,
    shown_to_downstream: i64,
    status: String,
    retry_reason: Option<String>,
    shadow_skip_reason: Option<String>,
    thinking_budget_tokens: Option<u32>,
    detector_features: serde_json::Value,
}

fn read_attempt_request_metadata_rows(sqlite_path: &Path) -> Vec<AttemptRequestMetadataRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT attempt_id, request_id, attempt_number, status, request_metadata_json \
             FROM attempts ORDER BY rowid",
        )
        .expect("attempt metadata query should prepare");
    statement
        .query_map([], |row| {
            let metadata_json: String = row.get(4)?;
            let request_metadata = serde_json::from_str(&metadata_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            Ok(AttemptRequestMetadataRow {
                attempt_id: row.get(0)?,
                request_id: row.get(1)?,
                attempt_number: row.get(2)?,
                status: row.get(3)?,
                request_metadata,
            })
        })
        .expect("attempt metadata query should execute")
        .map(|row| row.expect("attempt metadata row should decode"))
        .collect()
}

fn read_attempt_chain_rows(sqlite_path: &Path) -> Vec<AttemptChainRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT attempt_number, status, retry_reason, abort_reason, response_metadata_json \
             FROM attempts ORDER BY rowid",
        )
        .expect("attempt chain query should prepare");
    statement
        .query_map([], |row| {
            let metadata_json: String = row.get(4)?;
            let response_metadata = serde_json::from_str(&metadata_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            Ok(AttemptChainRow {
                attempt_number: row.get(0)?,
                status: row.get(1)?,
                retry_reason: row.get(2)?,
                abort_reason: row.get(3)?,
                response_metadata,
            })
        })
        .expect("attempt chain query should execute")
        .map(|row| row.expect("attempt chain row should decode"))
        .collect()
}

fn read_evidence_attempt_rows(sqlite_path: &Path) -> Vec<EvidenceAttemptRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT role, shown_to_downstream, status, retry_reason, shadow_skip_reason, \
             thinking_budget_tokens, detector_features_json FROM evidence_attempts ORDER BY rowid",
        )
        .expect("evidence attempt query should prepare");
    statement
        .query_map([], |row| {
            let detector_features_json: String = row.get(6)?;
            let detector_features =
                serde_json::from_str(&detector_features_json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?;
            Ok(EvidenceAttemptRow {
                role: row.get(0)?,
                shown_to_downstream: row.get(1)?,
                status: row.get(2)?,
                retry_reason: row.get(3)?,
                shadow_skip_reason: row.get(4)?,
                thinking_budget_tokens: row.get(5)?,
                detector_features,
            })
        })
        .expect("evidence attempt query should execute")
        .map(|row| row.expect("evidence attempt row should decode"))
        .collect()
}

fn count_rows(connection: &Connection, sql: &str) -> u64 {
    let count: i64 = connection
        .query_row(sql, [], |row| row.get(0))
        .expect("count query should succeed");
    u64::try_from(count).expect("count should be nonnegative")
}

async fn assert_shadow_timeout_count_stays(sqlite_path: &Path, expected_count: u64) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        expected_count
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        expected_count
    );
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        expected_count
    );
}

fn assert_shadow_timeout_summary(
    sqlite_path: &Path,
    expected_timeout_count: u64,
    expected_request_count: u64,
) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        expected_timeout_count
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(DISTINCT request_id) FROM evidence_attempts \
             WHERE role = 'shadow_continued'",
        ),
        expected_request_count
    );
}

fn read_evidence_chunks(connection: &Connection) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT channel, sequence_number, chunk_text \
             FROM evidence_chunks ORDER BY sequence_number",
        )
        .expect("evidence chunks query should prepare");
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .expect("evidence chunks query should execute")
        .map(|row| row.expect("evidence chunk row should decode"))
        .collect()
}

fn read_evidence_chunks_for_role(
    connection: &Connection,
    role: &str,
) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT c.channel, c.sequence_number, c.chunk_text \
             FROM evidence_chunks c \
             JOIN evidence_attempts a ON a.attempt_id = c.attempt_id \
             WHERE a.role = ?1 \
             ORDER BY c.sequence_number",
        )
        .expect("role evidence chunks query should prepare");
    statement
        .query_map(params![role], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .expect("role evidence chunks query should execute")
        .map(|row| row.expect("role evidence chunk row should decode"))
        .collect()
}

async fn send_shadow_timeout_request(proxy: &ProxyFixture, request_index: u32) {
    send_shadow_timeout_request_parts(&proxy.client, &proxy.base_url, request_index).await;
}

async fn send_shadow_timeout_request_parts(
    client: &reqwest::Client,
    base_url: &str,
    request_index: u32,
) {
    let response = client
        .post(format!(
            "{base_url}/v1/chat/completions?test=loop-once-shadow-timeout-then-success&id={request_index}",
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
}

async fn recv_shadow_timeout_upstream_requests(fake: &mut FakeUpstream) -> Vec<ObservedRequest> {
    recv_n_upstream_requests(fake, 3).await
}

async fn recv_n_upstream_requests(
    fake: &mut FakeUpstream,
    expected_count: usize,
) -> Vec<ObservedRequest> {
    let mut requests = Vec::new();
    for _ in 0..expected_count {
        requests.push(
            timeout(Duration::from_secs(2), fake.recv_next())
                .await
                .expect("expected request should reach upstream"),
        );
    }
    requests
}

async fn wait_for_evidence_status_count(sqlite_path: &Path, status: &str, expected: u64) {
    timeout(Duration::from_secs(5), async {
        loop {
            if sqlite_path.exists() {
                let connection = Connection::open(sqlite_path).expect("sqlite should open");
                let query =
                    format!("SELECT COUNT(*) FROM evidence_attempts WHERE status = '{status}'");
                if count_rows(&connection, &query) >= expected {
                    break;
                }
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("evidence status count should reach expected value");
}

async fn wait_for_evidence_role_status_count(
    sqlite_path: &Path,
    role: &str,
    status: &str,
    expected: u64,
) {
    timeout(Duration::from_secs(5), async {
        loop {
            if sqlite_path.exists() {
                let connection = Connection::open(sqlite_path).expect("sqlite should open");
                let count: i64 = connection
                    .query_row(
                        "SELECT COUNT(*) FROM evidence_attempts WHERE role = ?1 AND status = ?2",
                        params![role, status],
                        |row| row.get(0),
                    )
                    .expect("evidence role status count query should succeed");
                if u64::try_from(count).expect("count should be nonnegative") >= expected {
                    break;
                }
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("evidence role status count should reach expected value");
}

fn read_last_observability_row(sqlite_path: &Path, table: &str) -> ObservabilityRow {
    assert!(matches!(table, "requests" | "attempts"));
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let sql =
        format!("SELECT status, response_metadata_json FROM {table} ORDER BY rowid DESC LIMIT 1");
    let row: (String, String) = connection
        .query_row(&sql, [], |row| Ok((row.get(0)?, row.get(1)?)))
        .expect("observability row should exist");
    let response_metadata = serde_json::from_str(&row.1).expect("response metadata should be json");
    ObservabilityRow {
        status: row.0,
        response_metadata,
    }
}

fn repeated_input_chat_body() -> String {
    let repeated_input = format!("{REPEATED_INPUT_LOOP_LINE}\n{REPEATED_INPUT_LOOP_LINE}\n");
    serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": repeated_input}],
    })
    .to_string()
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

#[derive(Debug)]
struct UpstreamDropEvent {
    label: &'static str,
}

struct CancellableUpstream {
    base_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
    drop_receiver: mpsc::Receiver<UpstreamDropEvent>,
}

#[derive(Clone)]
struct CancellableUpstreamState {
    request_sender: mpsc::Sender<ObservedRequest>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    attempt_counts: Arc<Mutex<HashMap<String, u64>>>,
}

impl CancellableUpstream {
    async fn spawn() -> Self {
        let (request_sender, receiver) = mpsc::channel(10);
        let (drop_sender, drop_receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(cancellable_upstream_handler)
            .with_state(CancellableUpstreamState {
                request_sender,
                drop_sender,
                attempt_counts: Arc::new(Mutex::new(HashMap::new())),
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("cancellable upstream should bind");
        let addr = listener
            .local_addr()
            .expect("cancellable upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("cancellable upstream server failed: {error}");
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            receiver,
            drop_receiver,
        }
    }

    async fn recv_request(&mut self) -> ObservedRequest {
        self.receiver
            .recv()
            .await
            .expect("cancellable upstream should capture a request")
    }

    async fn recv_drop_within(&mut self, wait: Duration) -> UpstreamDropEvent {
        timeout(wait, self.drop_receiver.recv())
            .await
            .expect("upstream response body should be dropped before timeout")
            .expect("upstream drop channel should stay open")
    }

    async fn recv_drop_optional_within(&mut self, wait: Duration) -> Option<UpstreamDropEvent> {
        timeout(wait, self.drop_receiver.recv())
            .await
            .ok()
            .flatten()
    }
}

#[derive(Clone)]
struct FakeUpstreamState {
    sender: mpsc::Sender<ObservedRequest>,
    changing_model_len: Arc<AtomicU64>,
    attempt_counts: Arc<Mutex<HashMap<String, u64>>>,
    models_body: Option<&'static str>,
    models_status: StatusCode,
    models_label: &'static str,
    models_delay: Option<Duration>,
}

impl FakeUpstream {
    async fn spawn() -> Self {
        Self::spawn_with_optional_models_body(None).await
    }

    async fn spawn_with_models_body(models_body: &'static str) -> Self {
        Self::spawn_with_models_options(Some(models_body), None).await
    }

    async fn spawn_with_models_response(
        models_status: StatusCode,
        models_body: &'static str,
        models_label: &'static str,
    ) -> Self {
        Self::spawn_with_models_response_options(
            Some(models_body),
            models_status,
            models_label,
            None,
        )
        .await
    }

    async fn spawn_with_models_body_and_delay(
        models_body: &'static str,
        models_delay: Duration,
    ) -> Self {
        Self::spawn_with_models_options(Some(models_body), Some(models_delay)).await
    }

    async fn spawn_with_optional_models_body(models_body: Option<&'static str>) -> Self {
        Self::spawn_with_models_options(models_body, None).await
    }

    async fn spawn_with_models_options(
        models_body: Option<&'static str>,
        models_delay: Option<Duration>,
    ) -> Self {
        Self::spawn_with_models_response_options(
            models_body,
            StatusCode::OK,
            "models",
            models_delay,
        )
        .await
    }

    async fn spawn_with_models_response_options(
        models_body: Option<&'static str>,
        models_status: StatusCode,
        models_label: &'static str,
        models_delay: Option<Duration>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(fake_upstream_handler)
            .with_state(FakeUpstreamState {
                sender,
                changing_model_len: Arc::new(AtomicU64::new(128_000)),
                attempt_counts: Arc::new(Mutex::new(HashMap::new())),
                models_body,
                models_status,
                models_label,
                models_delay,
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

struct BrokenUpstream {
    base_url: String,
}

impl BrokenUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("broken upstream listener should bind");
        let addr = listener
            .local_addr()
            .expect("broken upstream address should be available");
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((_stream, _addr)) => {}
                    Err(error) => {
                        eprintln!("broken upstream listener failed: {error}");
                        break;
                    }
                }
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
        }
    }
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

async fn cancellable_upstream_handler(
    State(state): State<CancellableUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    let body = observed.body.clone();
    let path_and_query = observed.path_and_query.clone();
    state
        .request_sender
        .send(observed)
        .await
        .expect("cancellable upstream observation should send");

    if path_and_query.contains("test=loop-twice-then-cancellable-success")
        && body_requests_stream(&body)
        && next_cancellable_attempt_count(&state, &path_and_query) <= 2
    {
        return repeated_reasoning_line_sse_response(200);
    }

    if body_requests_stream(&body) {
        cancellable_chat_sse_response(state.drop_sender)
    } else {
        cancellable_chat_json_response(state.drop_sender)
    }
}

fn next_cancellable_attempt_count(state: &CancellableUpstreamState, key: &str) -> u64 {
    let mut counts = state
        .attempt_counts
        .lock()
        .expect("cancellable attempt counts should lock");
    let entry = counts.entry(key.to_owned()).or_insert(0);
    *entry = entry.saturating_add(1);
    *entry
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
    if endpoint == "/v1/models"
        && path_and_query.contains("test=distinct-multi-upstream-models")
        && state.models_body.is_some()
        && let Some(models_delay) = state.models_delay
    {
        sleep(models_delay).await;
    }

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
        if path_and_query.contains("test=multi-listener-models") {
            return json_response("models", MULTI_LISTENER_MODEL_METADATA_BODY.to_owned());
        }
        if path_and_query.contains("test=distinct-multi-upstream-models")
            && let Some(models_body) = state.models_body
        {
            let mut response = json_response(state.models_label, models_body.to_owned());
            *response.status_mut() = state.models_status;
            if state.models_status == StatusCode::TOO_MANY_REQUESTS {
                response
                    .headers_mut()
                    .insert(RETRY_AFTER, HeaderValue::from_static("11"));
            }
            return response;
        }
        if path_and_query.contains("test=model-metadata") {
            return json_response("models", MODEL_METADATA_BODY.to_owned());
        }
    }

    if endpoint == "/v1/chat/completions" {
        if let Some(response) = fake_chat_completion_response(path_and_query, state, body) {
            return response;
        }
    }

    let (label, body) = match endpoint {
        "/v1/models" => ("models", r#"{"object":"list","data":[]}"#),
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
        "/v1/rerank" => (
            "rerank",
            r#"{"object":"list","results":[{"index":0,"score":1.0}]}"#,
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

fn fake_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if !body_requests_stream(body) {
        return None;
    }
    fake_streaming_chat_completion_response(path_and_query, state, body)
        .or_else(|| Some(chat_completion_sse_response(body)))
}

fn fake_streaming_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if path_and_query.contains("test=compat-function-call") {
        return Some(chat_completion_compat_function_call_sse_response(body));
    }
    if path_and_query.contains("test=compat-refusal") {
        return Some(chat_completion_compat_refusal_sse_response(body));
    }
    if path_and_query.contains("test=compat-extensions") {
        return Some(chat_completion_extension_fields_sse_response(body));
    }
    if path_and_query.contains("test=slow-shielded") {
        return Some(slow_chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=loop-once-then-slow-success") {
        return Some(if body_contains_retry_hint(body) {
            slow_chat_completion_sse_response(body)
        } else {
            repeated_reasoning_line_sse_response(200)
        });
    }
    if path_and_query.contains("test=loop-once-shadow-raw-then-success") {
        if body_contains_retry_hint(body) {
            return Some(chat_completion_sse_response(body));
        }
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_reasoning_line_sse_response(200));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=loop-once-shadow-timeout-then-success") {
        if body_contains_retry_hint(body) {
            return Some(chat_completion_sse_response(body));
        }
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_reasoning_line_sse_response(200));
        }
        return Some(stalled_chat_completion_sse_response());
    }
    if path_and_query.contains("test=loop-once-then-success") {
        return Some(if body_contains_retry_hint(body) {
            chat_completion_sse_response(body)
        } else {
            repeated_reasoning_line_sse_response(200)
        });
    }
    if path_and_query.contains("test=loop-twice-then-success") {
        if next_fake_attempt_count(state, path_and_query) <= 2 {
            return Some(repeated_reasoning_line_sse_response(200));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=tool-loop-then-content-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_tool_fingerprint_sse_response());
        }
        return Some(content_only_chat_completion_sse_response());
    }
    if path_and_query.contains("test=always-429") {
        return Some(upstream_status_json_response(StatusCode::TOO_MANY_REQUESTS));
    }
    if path_and_query.contains("test=bad-request") {
        return Some(upstream_status_json_response(StatusCode::BAD_REQUEST));
    }
    if path_and_query.contains("test=transient-503-then-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(upstream_status_json_response(
                StatusCode::SERVICE_UNAVAILABLE,
            ));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=stall-once-then-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(stalled_chat_completion_sse_response());
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=loop-reasoning-hundreds") {
        return Some(repeated_reasoning_line_sse_response(200));
    }
    if path_and_query.contains("test=reasoning-leading-newlines") {
        return Some(reasoning_then_leading_newline_content_sse_response());
    }
    if path_and_query.contains("test=loop-reasoning-six") {
        return Some(repeated_reasoning_line_sse_response(6));
    }
    if path_and_query.contains("test=semantic-reasoning-varied") {
        return Some(semantic_reasoning_repetition_sse_response());
    }
    if path_and_query.contains("test=repeated-tool-fingerprint") {
        return Some(repeated_tool_fingerprint_sse_response());
    }
    if path_and_query.contains("test=copy-input-under-threshold") {
        return Some(repeated_input_copy_sse_response(11));
    }
    if path_and_query.contains("test=copy-input-over-threshold") {
        return Some(repeated_input_copy_sse_response(12));
    }
    None
}

fn next_fake_attempt_count(state: &FakeUpstreamState, key: &str) -> u64 {
    let mut counts = state
        .attempt_counts
        .lock()
        .expect("fake upstream attempt counts should not be poisoned");
    let count = counts.entry(key.to_owned()).or_insert(0);
    *count = count.saturating_add(1);
    *count
}

fn body_contains_retry_hint(body: &Bytes) -> bool {
    retry_hint_count(body) > 0
}

fn retry_hint_count(body: &Bytes) -> usize {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("messages")
                .and_then(serde_json::Value::as_array)
                .cloned()
        })
        .map_or(0, |messages| {
            messages
                .iter()
                .filter(|message| {
                    message
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|content| {
                            content.contains("llm-guard-proxy retry hint")
                                || content.contains("Previous attempt became repetitive")
                        })
                })
                .count()
        })
}

fn body_thinking_budget(body: &Bytes) -> Option<u64> {
    let value = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    [
        &["thinking", "budget_tokens"][..],
        &["thinking_token_budget"][..],
        &["thinking_budget"][..],
        &["chat_template_kwargs", "thinking_budget"][..],
        &["extra_body", "thinking_token_budget"][..],
        &["extra_body", "thinking_budget"][..],
        &["extra_body", "thinking", "budget_tokens"][..],
        &["extra_body", "chat_template_kwargs", "thinking_budget"][..],
    ]
    .into_iter()
    .find_map(|path| json_path(&value, path).and_then(serde_json::Value::as_u64))
}

fn json_path<'value>(
    mut value: &'value serde_json::Value,
    path: &[&str],
) -> Option<&'value serde_json::Value> {
    for key in path {
        value = value.get(*key)?;
    }
    Some(value)
}

fn upstream_status_json_response(status: StatusCode) -> Response<Body> {
    let mut response = json_response(
        "chat-completions-transient-error",
        r#"{"error":{"type":"upstream_test_error","message":"try again"}}"#.to_owned(),
    );
    *response.status_mut() = status;
    if status == StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("7"));
    }
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

fn cancellable_chat_sse_response(drop_sender: mpsc::Sender<UpstreamDropEvent>) -> Response<Body> {
    let chunks = vec![
        sse_json(&chat_completion_first_chunk()),
        sse_json(&chat_completion_second_chunk(false)),
        sse_json(&chat_completion_final_chunk(true, false)),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    cancellable_stream_response(
        "cancellable-chat-sse",
        "text/event-stream",
        chunks,
        drop_sender,
    )
}

fn cancellable_chat_json_response(drop_sender: mpsc::Sender<UpstreamDropEvent>) -> Response<Body> {
    let chunks = vec![
        Bytes::from_static(br#"{"id":"chatcmpl-cancellable","#),
        Bytes::from_static(br#""object":"chat.completion"}"#),
    ];
    cancellable_stream_response(
        "cancellable-chat-json",
        "application/json",
        chunks,
        drop_sender,
    )
}

fn cancellable_stream_response(
    label: &'static str,
    content_type: &'static str,
    chunks: Vec<Bytes>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
) -> Response<Body> {
    let body = Body::from_stream(CancellableResponseStream::new(
        label,
        chunks,
        drop_sender,
        STREAM_DELAY,
    ));
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

struct CancellableResponseStream {
    label: &'static str,
    chunks: Vec<Bytes>,
    next_index: usize,
    delay_after_first: Option<Pin<Box<tokio::time::Sleep>>>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    completed: bool,
}

impl CancellableResponseStream {
    fn new(
        label: &'static str,
        chunks: Vec<Bytes>,
        drop_sender: mpsc::Sender<UpstreamDropEvent>,
        delay_after_first: Duration,
    ) -> Self {
        Self {
            label,
            chunks,
            next_index: 0,
            delay_after_first: Some(Box::pin(sleep(delay_after_first))),
            drop_sender,
            completed: false,
        }
    }
}

impl Stream for CancellableResponseStream {
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.next_index >= this.chunks.len() {
            this.completed = true;
            return Poll::Ready(None);
        }

        if this.next_index > 0 {
            if let Some(delay) = &mut this.delay_after_first {
                match delay.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        this.delay_after_first = None;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }

        let chunk = this.chunks[this.next_index].clone();
        this.next_index = this.next_index.saturating_add(1);
        Poll::Ready(Some(Ok(chunk)))
    }
}

impl Drop for CancellableResponseStream {
    fn drop(&mut self) {
        if !self.completed {
            let _send_result = self
                .drop_sender
                .try_send(UpstreamDropEvent { label: self.label });
        }
    }
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

fn content_only_chat_completion_sse_response() -> Response<Body> {
    let chunks = [
        sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Safe"
                },
                "finish_reason": null
            }]
        })),
        sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": " answer"
                },
                "finish_reason": "stop"
            }]
        })),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-content-only-sse", chunks)
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

fn stalled_chat_completion_sse_response() -> Response<Body> {
    let body = Body::from_stream(stream::pending::<Result<Bytes, std::convert::Infallible>>());
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-stalled-sse"),
    );
    response
}

fn repeated_reasoning_line_sse_response(repetitions: usize) -> Response<Body> {
    repeated_delta_sse_response(
        "chat-completions-loop-reasoning-sse",
        repetitions,
        |line| {
            serde_json::json!({
                "reasoning_content": line,
            })
        },
        "reasoning loop line\n",
    )
}

fn reasoning_then_leading_newline_content_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-reasoning-leading-newlines-sse",
        [
            serde_json::json!({
                "reasoning_content": "think before answering",
            }),
            serde_json::json!({
                "content": "\n\nOK",
            }),
        ],
    )
}

fn semantic_reasoning_repetition_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-semantic-reasoning-sse",
        [
            serde_json::json!({
                "reasoning_content": "Use bsdtar to extract the archive into /dev/shm, then check unzip in a temporary directory and inspect members with python zipfile.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Try unzip into a tmpdir, but keep bsdtar available for archive extraction and use Python's zipfile module to inspect entries.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Python zipfile can read the archive listing; if that stalls, extract with bsdtar or unzip into a temporary directory.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Return to the unzip tmpdir plan, with bsdtar as the extractor fallback and python zipfile for inspection.\n",
            }),
        ],
    )
}

fn repeated_tool_fingerprint_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-repeated-tool-fingerprint-sse",
        [
            serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\",\"limit\":1}"
                    }
                }]
            }),
            serde_json::json!({
                "tool_calls": [{
                    "index": 1,
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"limit\":1,\"q\":\"x\"}"
                    }
                }]
            }),
        ],
    )
}

fn repeated_input_copy_sse_response(repetitions: usize) -> Response<Body> {
    repeated_delta_sse_response(
        "chat-completions-copy-input-sse",
        repetitions,
        |line| {
            serde_json::json!({
                "content": line,
            })
        },
        &format!("{REPEATED_INPUT_LOOP_LINE}\n"),
    )
}

fn delta_fragments_sse_response<const N: usize>(
    label: &'static str,
    deltas: [serde_json::Value; N],
) -> Response<Body> {
    delta_vec_sse_response(label, Vec::from(deltas))
}

fn delta_vec_sse_response(label: &'static str, deltas: Vec<serde_json::Value>) -> Response<Body> {
    let mut chunks = Vec::with_capacity(deltas.len().saturating_add(3));
    chunks.push(sse_json(&serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant"
            },
            "finish_reason": null
        }]
    })));
    for delta in deltas {
        chunks.push(sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null
            }]
        })));
    }
    chunks.push(sse_json(&serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    })));
    chunks.push(Bytes::from_static(b"data: [DONE]\n\n"));
    chat_completion_vec_stream_response(label, chunks)
}

fn repeated_delta_sse_response(
    label: &'static str,
    repetitions: usize,
    delta: impl Fn(&str) -> serde_json::Value,
    line: &str,
) -> Response<Body> {
    let mut deltas = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        deltas.push(delta(line));
    }
    delta_vec_sse_response(label, deltas)
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

fn chat_completion_vec_stream_response(label: &'static str, chunks: Vec<Bytes>) -> Response<Body> {
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
    evidence_sqlite_path: PathBuf,
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

    async fn spawn_with_admission_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        server_config: &str,
    ) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            server_config,
            "",
            "",
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
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            "",
            metadata_config,
            "",
            "",
        )
        .await
    }

    async fn spawn_with_observability_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        observability_config: &str,
    ) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
            "",
            "",
            observability_config,
            "",
        )
        .await
    }

    async fn spawn_with_evidence_config(upstream_base_url: &str, evidence_config: &str) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            true,
            AppConfig::default().server.max_in_flight_requests,
            "",
            "",
            "",
            evidence_config,
        )
        .await
    }

    async fn spawn_with_full_options(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        server_config: &str,
        metadata_config: &str,
        observability_config: &str,
        evidence_config: &str,
    ) -> Self {
        let root = unique_test_dir("proxy");
        fs::create_dir_all(&root).expect("test root should be created");
        set_owner_only_dir(&root);
        let config_path = root.join("config.toml");
        let sqlite_path = root.join("storage").join("observability.sqlite3");
        let evidence_sqlite_path = root.join("storage").join("evidence.sqlite3");
        write_proxy_config_with_observability(ProxyConfigWriteOptions {
            config_path: &config_path,
            upstream_base_url,
            sqlite_path: &sqlite_path,
            evidence_sqlite_path: &evidence_sqlite_path,
            observability_enabled,
            max_in_flight_requests,
            server_config,
            metadata_config,
            observability_config,
            evidence_config,
        });
        let manager =
            ConfigManager::from_explicit_path(&config_path).expect("proxy config should load");
        let store = ObservabilityStore::open(manager.handle()).expect("store should open");
        let evidence_store = EvidenceStore::open(manager.handle());
        let state = ProxyState::new(
            manager.handle(),
            manager.path().to_path_buf(),
            store.clone(),
            evidence_store,
            build_http_client().expect("client should build"),
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
            evidence_sqlite_path,
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
    let evidence_sqlite_path = sqlite_path.with_file_name("evidence.sqlite3");
    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path,
        upstream_base_url,
        sqlite_path,
        evidence_sqlite_path: &evidence_sqlite_path,
        observability_enabled,
        max_in_flight_requests,
        server_config: "",
        metadata_config,
        observability_config: "",
        evidence_config: "",
    });
}

#[derive(Clone, Copy)]
struct ProxyConfigWriteOptions<'a> {
    config_path: &'a Path,
    upstream_base_url: &'a str,
    sqlite_path: &'a Path,
    evidence_sqlite_path: &'a Path,
    observability_enabled: bool,
    max_in_flight_requests: usize,
    server_config: &'a str,
    metadata_config: &'a str,
    observability_config: &'a str,
    evidence_config: &'a str,
}

fn write_proxy_config_with_observability(options: ProxyConfigWriteOptions<'_>) {
    fs::write(
        options.config_path,
        format!(
            r#"
[server]
max_in_flight_requests = {max_in_flight_requests}
{server_config}

[upstream]
base_url = "{upstream_base_url}"
{metadata_config}

[observability]
enabled = {observability_enabled}
sqlite_path = "{sqlite_path}"
capture_raw_payloads = false
{observability_config}

[observability.retention]
max_bytes = {TEST_MAX_BYTES}
prune_to_bytes = {TEST_PRUNE_TO_BYTES}
max_records = {TEST_MAX_RECORDS}

[evidence]
sqlite_path = "{evidence_sqlite_path}"
blob_cache_dir = "{blob_cache_dir}"
{evidence_config}
"#,
            max_in_flight_requests = options.max_in_flight_requests,
            server_config = options.server_config,
            upstream_base_url = options.upstream_base_url,
            metadata_config = options.metadata_config,
            observability_enabled = options.observability_enabled,
            sqlite_path = options.sqlite_path.display(),
            evidence_sqlite_path = options.evidence_sqlite_path.display(),
            blob_cache_dir = options
                .evidence_sqlite_path
                .parent()
                .expect("evidence sqlite path should have parent")
                .join("evidence-blobs")
                .display(),
            evidence_config = options.evidence_config,
            observability_config = options.observability_config,
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

#[cfg(unix)]
async fn read_pid_file(path: &Path) -> u32 {
    for _ in 0..20 {
        if let Ok(text) = fs::read_to_string(path) {
            return text
                .trim()
                .parse::<u32>()
                .expect("pid file should contain a child pid");
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("pid file was not written: {}", path.display());
}

#[cfg(unix)]
async fn assert_process_not_running(pid: u32) {
    for _ in 0..20 {
        match linux_process_state(pid) {
            None | Some('Z') => return,
            Some(_) => sleep(Duration::from_millis(50)).await,
        }
    }
    let _ = tokio::process::Command::new("kill")
        .arg("-KILL")
        .arg(pid.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
    panic!("process {pid} still appears to be running");
}

#[cfg(unix)]
async fn kill_process_if_running(pid: u32) {
    if matches!(linux_process_state(pid), None | Some('Z')) {
        return;
    }
    let _ = tokio::process::Command::new("kill")
        .arg("-KILL")
        .arg(pid.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
}

#[cfg(unix)]
fn linux_process_state(pid: u32) -> Option<char> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_prefix, suffix) = stat.rsplit_once(") ")?;
    suffix.chars().next()
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

fn assert_safe_operational_text(label: &str, text: &str) {
    for sensitive in [
        "sk-live-secret",
        "sk-header-secret",
        "downstream-secret",
        "Bearer downstream-secret",
    ] {
        assert!(
            !text.contains(sensitive),
            "{label} leaked sensitive value {sensitive:?}: {text}"
        );
    }
    let lowercase = text.to_ascii_lowercase();
    for sensitive_key in ["authorization", "x-api-key"] {
        assert!(
            !lowercase.contains(sensitive_key),
            "{label} leaked sensitive key {sensitive_key:?}: {text}"
        );
    }
}

async fn send_metrics_chat_request(proxy: &ProxyFixture, fake: &mut FakeUpstream, index: usize) {
    let body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": format!("metrics pruning {index}")}],
    })
    .to_string();
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=metrics-pruning-{index}",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("metrics chat request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response
        .text()
        .await
        .expect("metrics chat body should be consumed");
    let _observed = fake.recv_next().await;
}

async fn fetch_metrics(proxy: &ProxyFixture) -> String {
    let response = proxy
        .client
        .get(format!("{}/metrics", proxy.base_url))
        .send()
        .await
        .expect("metrics request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.text().await.expect("metrics should be text")
}

fn assert_metric_type(body: &str, metric_name: &str, metric_type: &str) {
    let expected = format!("# TYPE {metric_name} {metric_type}");
    assert!(
        body.contains(&expected),
        "metrics body missing expected type line {expected:?}: {body}"
    );
}

fn assert_legacy_retained_counter_metrics_absent(body: &str) {
    for metric_name in [
        "llm_guard_proxy_requests_total",
        "llm_guard_proxy_attempts_total",
        "llm_guard_proxy_retries_total",
        "llm_guard_proxy_loop_aborts_total",
        "llm_guard_proxy_upstream_errors_total",
        "llm_guard_proxy_heartbeat_mode_total",
        "llm_guard_proxy_first_token_latency_ms_bucket",
        "llm_guard_proxy_first_token_latency_ms_count",
        "llm_guard_proxy_first_token_latency_ms_sum",
        "llm_guard_proxy_total_latency_ms_bucket",
        "llm_guard_proxy_total_latency_ms_count",
        "llm_guard_proxy_total_latency_ms_sum",
    ] {
        assert!(
            !body.contains(metric_name),
            "metrics body still exposes legacy retained metric {metric_name:?}: {body}"
        );
    }
}

fn metric_value(body: &str, metric_name: &str) -> u64 {
    let prefix = format!("{metric_name} ");
    body.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("metrics body missing numeric metric {metric_name:?}: {body}"))
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
