use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use axum::http::header::{AUTHORIZATION, CONNECTION};
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
const SSE_FIRST_CHUNK: &[u8] = b"data: first\n\n";
const SSE_SECOND_CHUNK: &[u8] = b"data: second\n\n";
const LONG_JSON_FIRST_CHUNK: &[u8] = br#"{"object":"list","data":["#;
const LONG_JSON_SECOND_CHUNK: &[u8] = br"]}";
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
async fn chat_completions_forwards_body_without_policy_rewrite() {
    let fake = FakeUpstream::spawn().await;
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
        response.text().await.expect("body should be text"),
        r#"{"id":"chatcmpl-test","object":"chat.completion"}"#
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    assert_eq!(observed.body, body);
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
            .get(format!("{}/v1/models?test=long-json", proxy.base_url))
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
    assert_eq!(observed.path_and_query, "/v1/models?test=long-json");
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
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"transport-failure-model","prompt":"ping"}"#)
        .send()
        .await
        .expect("proxy request should complete with gateway error");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("body should be text");
    assert!(
        body.contains("upstream_transport_error"),
        "gateway error should identify upstream transport failure: {body}"
    );

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
    assert_eq!(request_metadata["method"], "POST");
    assert_eq!(request_metadata["path"], "/v1/completions");
    assert_eq!(request_metadata["request_body_bytes"], "51");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(
        request_response_metadata["error_type"],
        "upstream_transport_error"
    );
    assert_eq!(attempt_row.0, "failed");
    assert_eq!(attempt_row.1, None);
    assert!(attempt_row.2.contains("upstream_transport_error"));
    assert_eq!(attempt_metadata["method"], "POST");
    assert_eq!(attempt_metadata["path"], "/v1/completions");
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
async fn invalid_upstream_url_failure_writes_metadata_without_secret() {
    let proxy = ProxyFixture::spawn("http://127.0.0.1:1/v1", true).await;
    let uri = Uri::from_static("/v1/models?limit=2");
    let headers = HeaderMap::new();
    let request_id =
        RequestId::from_string("req-invalid-upstream").expect("request id should be valid");
    let metadata = request_metadata(&Method::GET, &uri, &headers, 0, true);
    let error = ProxyError::invalid_upstream_url(
        "https://user:secret@example.test/v1?api_key=sk-test",
        String::from("must not contain sensitive query parameters"),
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
            .contains("https://redacted:redacted@example.test/v1?redacted=redacted")
    );
    assert!(!request_row.2.contains("user:secret"));
    assert!(!request_row.2.contains("secret"));
    assert!(!request_row.2.contains("sk-test"));
    assert!(!request_row.2.contains("api_key"));
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
    let error = build_upstream_url("https://user:secret@example.test/v1?api_key=sk-test", &uri)
        .expect_err("credential-bearing upstream URL should be rejected");
    let error = error.to_string();

    assert!(error.contains("invalid upstream base URL"));
    assert!(error.contains("https://redacted:redacted@example.test/v1?redacted=redacted"));
    assert!(!error.contains("user:secret"));
    assert!(!error.contains("secret"));
    assert!(!error.contains("sk-test"));
    assert!(!error.contains("api_key"));
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

impl FakeUpstream {
    async fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(fake_upstream_handler)
            .with_state(sender);
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
        self.receiver
            .recv()
            .await
            .expect("fake upstream should capture a request")
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

async fn fake_upstream_handler(
    State(sender): State<mpsc::Sender<ObservedRequest>>,
    request: Request<Body>,
) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_PROXY_BODY_BYTES)
        .await
        .expect("fake upstream body should be readable");
    let path_and_query = parts.uri.path_and_query().map_or_else(
        || parts.uri.path().to_owned(),
        |value| value.as_str().to_owned(),
    );
    let observed = ObservedRequest {
        method: parts.method,
        path_and_query,
        headers: parts.headers,
        body,
    };
    let endpoint = observed
        .path_and_query
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let is_sse_stream = observed.path_and_query.contains("test=sse");
    let is_long_json_stream = observed.path_and_query.contains("test=long-json");
    sender
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

    let (label, body) = match endpoint.as_str() {
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
        _ => ("unknown", r#"{"error":"unsupported"}"#),
    };
    let status = if label == "unknown" {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::OK
    };
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
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
    state: ProxyState,
    store: ObservabilityStore,
    sqlite_path: PathBuf,
    root: PathBuf,
}

impl ProxyFixture {
    async fn spawn(upstream_base_url: &str, observability_enabled: bool) -> Self {
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
        );
        let manager =
            ConfigManager::from_explicit_path(&config_path).expect("proxy config should load");
        let store = ObservabilityStore::open(manager.handle()).expect("store should open");
        let state = ProxyState::new(
            manager.handle(),
            manager.path().to_path_buf(),
            store.clone(),
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
) {
    fs::write(
        config_path,
        format!(
            r#"
[upstream]
base_url = "{upstream_base_url}"

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
