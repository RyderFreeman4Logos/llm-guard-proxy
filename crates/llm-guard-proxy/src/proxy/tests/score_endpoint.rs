use super::*;

async fn in_process_response_json(response: axum::response::Response) -> serde_json::Value {
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("in-process response body should be readable");
    serde_json::from_slice(&body).unwrap_or_else(|error| {
        panic!("in-process response body should parse as JSON: {error}; body={body:?}")
    })
}

#[tokio::test]
async fn score_endpoint_passthrough_batched_text1() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-batch", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":["q1","q2"],"text_2":["d1","d2"],"query":"ignored-extra","documents":["ignored-extra"]}"#)
        .send()
        .await
        .expect("batch score should complete");
    // Fake upstream has no /v1/score → 404, but path must remain score (not rewritten).
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("batch score should reach upstream as /v1/score");
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/score?test=score-batch");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("body json");
    assert_eq!(observed_body["text_1"][0], "q1");
    assert_eq!(observed_body["text_2"][1], "d2");
    assert_eq!(observed_body["query"], "ignored-extra");
    assert_eq!(observed_body["documents"][0], "ignored-extra");
    let _ = response.status();
}

#[tokio::test]
async fn score_endpoint_passthrough_unknown_complete_shape() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","left_input":"q","right_input":"d","future":{}}}"#,
        "9".repeat(1_000)
    );
    let content_digest = "sha-256=:O+FZtxxalNcUkGDg7JYhQ3D8GlUsdTlWARLltzwfRPo=:";
    let signature_input = r#"sig1=("@method" "@path" "content-digest")"#;
    let authorization = r#"Signature keyId="score-future",signature="unchanged""#;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-future-shape",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header("Signature", "sig1=:ZnV0dXJlLXNoYXBl:")
        .header("Signature-Input", signature_input)
        .header("Content-Digest", content_digest)
        .header(AUTHORIZATION, authorization)
        .body(request_body.clone())
        .send()
        .await
        .expect("future score shape should complete");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("future score shape should reach upstream unchanged");
    assert_eq!(observed.path_and_query, "/v1/score?test=score-future-shape");
    assert_eq!(observed.body.as_ref(), request_body.as_bytes());
    assert_eq!(
        observed
            .headers
            .get("signature")
            .and_then(|value| value.to_str().ok()),
        Some("sig1=:ZnV0dXJlLXNoYXBl:")
    );
    assert_eq!(
        observed
            .headers
            .get("signature-input")
            .and_then(|value| value.to_str().ok()),
        Some(signature_input)
    );
    assert_eq!(
        observed
            .headers
            .get("content-digest")
            .and_then(|value| value.to_str().ok()),
        Some(content_digest)
    );
    assert_eq!(
        observed
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some(authorization)
    );
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response.bytes().await.expect("response body should drain");

    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    let request_meta: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request metadata");
    let metadata: serde_json::Value =
        serde_json::from_str(&request_meta).expect("metadata should be JSON");
    assert_eq!(metadata["score_via_rerank"], "false");
    assert_eq!(metadata["score_passthrough"], "true");
    assert!(metadata.get("score_batch_passthrough").is_none());
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn score_endpoint_ignores_chat_param_overrides() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
temperature = 0.6
",
        ),
    )
    .await;
    let digits = "9".repeat(1_000);
    let canonical =
        format!(r#"{{"model":"test-chat","text_1":"q","text_2":"d","future":{digits}}}"#);

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-param-override",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(canonical)
        .send()
        .await
        .expect("canonical score should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("score response should drain");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("adapted score should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_text = std::str::from_utf8(&observed.body).expect("body should be utf-8");
    assert!(observed_text.contains(&digits));
    assert!(!observed_text.contains("temperature"));

    let future = r#"{"model":"test-chat","left_input":"q","right_input":"d"}"#;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-param-override-future",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(future)
        .send()
        .await
        .expect("future score should complete");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response
        .bytes()
        .await
        .expect("future response should drain");
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("future score should reach score upstream");
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=score-param-override-future"
    );
    assert_eq!(observed.body.as_ref(), future.as_bytes());
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn score_endpoint_ignores_blocking_chat_pre_request_guard() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("score-guard-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "block", guard_result("block", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;
    let body = r#"{"model":"test-chat","text_1":"q","text_2":"d","messages":[{"role":"user","content":"ignored score extra"}]}"#;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-guard-block",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header("content-digest", "sha-256=:stale-original-body:")
        .body(body)
        .send()
        .await
        .expect("score request should bypass chat guard");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    let observed = fake.recv_next().await;
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    assert!(observed.headers.get("content-digest").is_none());
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn score_endpoint_ignores_replacing_chat_pre_request_guard() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("score-guard-replace");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let replacement = r#"[{"role":"user","content":"guard replacement"}]"#;
    let script = write_guard_script(
        &guard_root,
        "replace",
        guard_result("replace", Some(replacement)),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;
    let body = r#"{"model":"test-chat","left_input":"q","right_input":"d","messages":[{"role":"user","content":"ignored score extra"}]}"#;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-guard-replace",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header("content-digest", "sha-256=:original-body:")
        .body(body)
        .send()
        .await
        .expect("future score should bypass chat guard");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    response.bytes().await.expect("response body should drain");
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/score?test=score-guard-replace"
    );
    assert_eq!(observed.body.as_ref(), body.as_bytes());
    assert_eq!(
        observed
            .headers
            .get("content-digest")
            .and_then(|value| value.to_str().ok()),
        Some("sha-256=:original-body:")
    );
}

#[tokio::test]
async fn score_endpoint_passthrough_non_success_rerank_status() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-upstream-500",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/problem+json")
    );
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("13")
    );
    assert!(response.headers().get("server").is_none());
    assert!(response.headers().get("x-upstream-endpoint").is_none());
    let body = response.text().await.expect("body");
    assert!(body.contains("upstream boom"), "{body}");

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("should forward to rerank");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
}

#[tokio::test]
async fn score_endpoint_rejects_oversized_body_before_upstream_routing() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d","padding":"{}"}}"#,
        "x".repeat(1024 * 1024)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-oversized", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await
        .expect("oversized score request should receive an error response");
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = response.text().await.expect("error body should drain");
    assert!(
        body.contains("score request exceeded adapter limit"),
        "{body}"
    );

    assert!(
        tokio::time::timeout(Duration::from_millis(100), fake.recv_next())
            .await
            .is_err(),
        "oversized score request must fail before upstream routing"
    );
}

#[tokio::test]
async fn score_endpoint_stops_polling_chunked_body_at_adapter_limit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "allowed"
base_url = "{0}"
match_models = ["allowed-model"]
max_in_flight_requests = 2

[[upstreams]]
name = "forbidden"
base_url = "{0}"
match_models = ["qwen3-reranker-8b"]
max_in_flight_requests = 2

[[listeners]]
name = "score-stream-restricted"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["allowed"]
"#,
            fake.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "score-stream-restricted");
    let polls = Arc::new(AtomicUsize::new(0));
    let stream_polls = Arc::clone(&polls);
    let body_stream = stream::poll_fn(move |_| {
        let poll = stream_polls.fetch_add(1, Ordering::SeqCst);
        match poll {
            0..=3 => Poll::Ready(Some(Ok::<_, std::io::Error>(Bytes::from(vec![
                b'x';
                256 * 1_024
            ])))),
            4 => Poll::Ready(Some(Ok(Bytes::from_static(b"x")))),
            _ => panic!("score body was polled after the first limit-crossing chunk"),
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/score?test=score-stream-limit")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from_stream(body_stream))
        .expect("score request should build");
    assert!(request.headers().get(CONTENT_LENGTH).is_none());
    score_adapter::reset_pydantic_value_parse_count();

    let response = proxy_handler(State(proxy.state.for_listener(listener)), request).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(polls.load(Ordering::SeqCst), 5);
    assert_eq!(score_adapter::pydantic_value_parse_count(), 0);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("error body should drain");
    assert!(
        std::str::from_utf8(&body)
            .expect("error body should be UTF-8")
            .contains("score request exceeded adapter limit")
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM requests"), 1);
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
    let metadata: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("failed request metadata");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata).expect("request metadata should be JSON");
    for forbidden in [
        "model",
        "selected_upstream",
        "upstream_profile",
        "score_via_rerank",
        "guard_action",
    ] {
        assert!(
            metadata.get(forbidden).is_none(),
            "body rejection must precede {forbidden}: {metadata}"
        );
    }
}

#[tokio::test]
async fn score_endpoint_profile_limit_mode_caps_body_before_model_routing() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "allowed"
base_url = "{0}"
match_models = ["allowed-model"]
max_in_flight_requests = 2

[[upstreams]]
name = "forbidden"
base_url = "{0}"
match_models = ["qwen3-reranker-8b"]
max_in_flight_requests = 2

[[listeners]]
name = "score-restricted"
bind_host = "127.0.0.1"
port = 18003
allowed_upstreams = ["allowed"]
"#,
            fake.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "score-restricted");
    let request_body = format!(
        r#"{{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d","padding":"{}"}}"#,
        "x".repeat(1024 * 1024)
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/score?test=score-profile-limit-oversized")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(request_body))
        .expect("score request should build");

    let response = proxy_handler(State(proxy.state.for_listener(listener)), request).await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("error body should drain");
    assert!(
        std::str::from_utf8(&body)
            .expect("error body should be UTF-8")
            .contains("score request exceeded adapter limit")
    );
}

#[tokio::test]
async fn score_endpoint_body_read_error_preserves_received_response_observability() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-body-read-error",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should receive a proxy error");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    response
        .bytes()
        .await
        .expect("proxy error response body should drain");

    fake.recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("score request should reach rerank upstream");
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(attempt_row.status, "failed");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.response_metadata["upstream_response_received"],
        "true"
    );
    assert_eq!(attempt_row.response_metadata["http_status_success"], "true");
    assert_eq!(
        attempt_row.response_metadata["response_header_content-type"],
        "application/vnd.rerank+json"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_x-request-id"],
        "rerank-body-error-123"
    );
}

#[tokio::test]
async fn score_endpoint_negotiates_identity_encoding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-identity", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(axum::http::header::ACCEPT_ENCODING, "gzip, deflate")
        .header("content-encoding", "gzip")
        .header("content-md5", "stale-md5")
        .header("digest", "SHA-256=stale")
        .header("content-digest", "sha-256=:stale:")
        .header("repr-digest", "sha-256=:stale:")
        .header("etag", "stale-etag")
        .header("if-match", "stale-etag")
        .header("if-none-match", "stale-etag")
        .header(AUTHORIZATION, "Bearer score-auth-token")
        .header("x-api-key", "score-api-key")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::OK);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("should reach upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let accept_encoding = observed
        .headers
        .get(axum::http::header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok());
    assert_eq!(accept_encoding, Some("identity"));
    assert!(observed.headers.get("content-encoding").is_none());
    assert!(observed.headers.get("content-md5").is_none());
    assert!(observed.headers.get("digest").is_none());
    assert!(observed.headers.get("content-digest").is_none());
    assert!(observed.headers.get("repr-digest").is_none());
    assert!(observed.headers.get("etag").is_none());
    assert!(observed.headers.get("if-match").is_none());
    assert!(observed.headers.get("if-none-match").is_none());
    assert_eq!(
        observed
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer score-auth-token")
    );
    assert_eq!(
        observed
            .headers
            .get("x-api-key")
            .and_then(|value| value.to_str().ok()),
        Some("score-api-key")
    );
    response
        .bytes()
        .await
        .expect("score response body should drain");

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    let metadata = &attempts[0].request_metadata;
    assert_eq!(metadata["path"], "/v1/rerank");
    assert_eq!(metadata["query_present"], "true");
    assert_eq!(
        metadata["upstream_request_header_accept-encoding"],
        "identity"
    );
}

#[tokio::test]
async fn score_endpoint_rejects_all_signed_transform_header_forms_locally() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let mut cases = Vec::new();

    let mut complete = HeaderMap::new();
    complete.insert("signature", HeaderValue::from_static("sig1=:complete:"));
    complete.insert(
        "signature-input",
        HeaderValue::from_static(r#"sig1=("@method" "@path" "content-digest")"#),
    );
    complete.insert(
        "content-digest",
        HeaderValue::from_static("sha-256=:original:"),
    );
    cases.push(("complete", complete));

    for (name, header) in [
        ("signature-only", "signature"),
        ("signature-input-only", "signature-input"),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(header, HeaderValue::from_static("partial"));
        cases.push((name, headers));
    }

    let mut empty = HeaderMap::new();
    empty.insert("signature", HeaderValue::from_static(""));
    cases.push(("empty", empty));

    let mut malformed = HeaderMap::new();
    malformed.insert(
        "signature-input",
        HeaderValue::from_bytes(&[0xff]).expect("opaque header value should build"),
    );
    cases.push(("malformed", malformed));

    let mut repeated = HeaderMap::new();
    repeated.append("signature", HeaderValue::from_static("sig1=:one:"));
    repeated.append("signature", HeaderValue::from_static("sig2=:two:"));
    cases.push(("repeated", repeated));

    let mut mixed_case = HeaderMap::new();
    mixed_case.insert(
        HeaderName::from_bytes(b"SiGnAtUrE-InPuT").expect("mixed-case name should build"),
        HeaderValue::from_static("mixed"),
    );
    cases.push(("mixed-case", mixed_case));

    for (name, mut headers) in cases {
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("/v1/score?test=signed-{name}"))
            .body(Body::from(
                r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#,
            ))
            .expect("signed score request should build");
        let (mut parts, body) = request.into_parts();
        parts.headers = headers;
        let response =
            proxy_handler(State(proxy.state.clone()), Request::from_parts(parts, body)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "case={name}");
        assert!(response.headers().contains_key("x-request-id"));
        let json = in_process_response_json(response).await;
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "headers");
        assert_eq!(
            json["error"]["code"],
            "signed_request_transformation_unsupported"
        );
        assert!(json["error"]["request_id"].is_string());
    }

    let malformed_body = Request::builder()
        .method(Method::POST)
        .uri("/v1/score?test=signed-malformed-body")
        .header(CONTENT_TYPE, "application/json")
        .header("signature", "sig1=:must-not-win:")
        .body(Body::from("{"))
        .expect("malformed request should build");
    let response = proxy_handler(State(proxy.state.clone()), malformed_body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = in_process_response_json(response).await;
    assert_eq!(json["error"]["code"], "invalid_score_request");
    assert_eq!(json["error"]["param"], "body");

    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM requests"), 8);
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
    let metadata_rows: Vec<String> = connection
        .prepare("SELECT request_metadata_json FROM requests ORDER BY rowid")
        .expect("metadata query should prepare")
        .query_map([], |row| row.get(0))
        .expect("metadata query should execute")
        .map(|row| row.expect("metadata row should decode"))
        .collect();
    assert_eq!(metadata_rows.len(), 8);
    assert!(
        metadata_rows
            .iter()
            .take(7)
            .all(|metadata| metadata.contains("signed_request_transformation_rejected"))
    );
    assert!(metadata_rows.iter().all(|metadata| {
        !metadata.contains("complete")
            && !metadata.contains("partial")
            && !metadata.contains("must-not-win")
    }));
}

#[tokio::test]
async fn score_endpoint_rejects_unsafe_authorization_on_transformed_requests_locally() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let cases = vec![
        (
            "http-message-signature",
            vec![HeaderValue::from_static(
                r#"Signature keyId="score",signature="abc""#,
            )],
        ),
        (
            "aws-sigv4",
            vec![HeaderValue::from_static(
                "AWS4-HMAC-SHA256 Credential=AKIA/20260710/us-east-1/service/aws4_request,SignedHeaders=host,Signature=abc",
            )],
        ),
        (
            "digest",
            vec![HeaderValue::from_static(
                r#"Digest username="score",response="abc""#,
            )],
        ),
        ("hmac", vec![HeaderValue::from_static("HMAC key:abc")]),
        (
            "unknown",
            vec![HeaderValue::from_static("Custom signed-value")],
        ),
        ("malformed", vec![HeaderValue::from_static("Bearer")]),
        (
            "non-utf8",
            vec![HeaderValue::from_bytes(&[0xff]).expect("opaque header value should build")],
        ),
        (
            "mixed-duplicate",
            vec![
                HeaderValue::from_static("Bearer safe-token"),
                HeaderValue::from_static("AWS4-HMAC-SHA256 Credential=AKIA"),
            ],
        ),
    ];

    for (name, values) in cases {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        for value in values {
            headers.append(AUTHORIZATION, value);
        }
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("/v1/score?test=unsafe-authorization-{name}"))
            .body(Body::from(
                r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#,
            ))
            .expect("score request should build");
        let (mut parts, body) = request.into_parts();
        parts.headers = headers;
        let response =
            proxy_handler(State(proxy.state.clone()), Request::from_parts(parts, body)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "case={name}");
        let json = in_process_response_json(response).await;
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "headers");
        assert_eq!(
            json["error"]["code"],
            "signed_request_transformation_unsupported"
        );
    }

    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
    let metadata_rows: Vec<String> = connection
        .prepare("SELECT request_metadata_json FROM requests ORDER BY rowid")
        .expect("metadata query should prepare")
        .query_map([], |row| row.get(0))
        .expect("metadata query should execute")
        .map(|row| row.expect("metadata row should decode"))
        .collect();
    assert_eq!(metadata_rows.len(), 8);
    assert!(
        metadata_rows
            .iter()
            .all(|metadata| metadata.contains("signed_request_transformation_rejected"))
    );
    assert!(metadata_rows.iter().all(|metadata| {
        !metadata.contains("AWS4")
            && !metadata.contains("Digest username")
            && !metadata.contains("signed-value")
            && !metadata.contains("safe-token")
    }));
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn score_endpoint_virtual_key_auth_precedes_transform_errors() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("fail_closed"),
    )
    .await;

    for (name, body, signed) in [
        (
            "signed-transformable",
            r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#,
            true,
        ),
        ("malformed-json", "{", false),
    ] {
        let mut request = proxy
            .client
            .post(format!(
                "{}/v1/score?test=unknown-key-before-{name}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .header("x-virtual-key", "vk_unknown");
        if signed {
            request = request.header("signature", "sig1=:must-not-win:");
        }
        let response = request
            .body(body)
            .send()
            .await
            .expect("unknown virtual key score request should complete");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "case={name}");
        let json: serde_json::Value = response.json().await.expect("error response JSON");
        assert_eq!(json["error"]["type"], "virtual_key_unauthorized");
    }

    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM requests"), 2);
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
    let metadata_rows: Vec<serde_json::Value> = connection
        .prepare("SELECT request_metadata_json FROM requests ORDER BY rowid")
        .expect("metadata query should prepare")
        .query_map([], |row| row.get::<_, String>(0))
        .expect("metadata query should execute")
        .map(|row| {
            serde_json::from_str(&row.expect("metadata row should decode"))
                .expect("metadata should be JSON")
        })
        .collect();
    assert!(metadata_rows.iter().all(|metadata| {
        metadata["virtual_key_resolution"] == "fail_closed"
            && metadata
                .get("signed_request_transformation_rejected")
                .is_none()
            && metadata.get("score_via_rerank").is_none()
    }));
}

#[tokio::test]
async fn score_endpoint_preserves_safe_authorization_on_transformed_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for (name, authorization) in [
        ("bearer", "Bearer score-token._~+/="),
        ("basic", "Basic c2NvcmU6c2VjcmV0"),
    ] {
        let response = proxy
            .client
            .post(format!("{}/v1/score?test=safe-auth-{name}", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, authorization)
            .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#)
            .send()
            .await
            .expect("safe authorization score request should complete");
        assert_eq!(response.status(), StatusCode::OK, "case={name}");
        response.bytes().await.expect("response body should drain");

        let observed = fake.recv_next().await;
        assert_eq!(
            observed.path_and_query,
            format!("/v1/rerank?test=safe-auth-{name}")
        );
        assert_eq!(
            observed
                .headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some(authorization)
        );
    }
}

#[tokio::test]
async fn direct_rerank_preserves_signature_headers() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = r#"{"model":"qwen3-reranker-8b","query":"q","documents":["d"]}"#;
    let content_digest = "sha-256=:/Z1EncffKsW96BDup1K0MFJGC2lRs+AyssoLq0Zog5Q=:";
    let signature_input = r#"sig1=("@method" "@path" "content-digest")"#;
    let authorization = "AWS4-HMAC-SHA256 Credential=AKIA/20260710/us-east-1/service/aws4_request,SignedHeaders=host,Signature=direct";

    let response = proxy
        .client
        .post(format!("{}/v1/rerank?test=signed-direct", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header("signature", "sig1=:ZGlyZWN0:")
        .header("signature-input", signature_input)
        .header("content-digest", content_digest)
        .header(AUTHORIZATION, authorization)
        .body(body)
        .send()
        .await
        .expect("direct rerank should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");

    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/rerank?test=signed-direct");
    assert_eq!(observed.body.as_ref(), body.as_bytes());
    assert_eq!(
        observed
            .headers
            .get("signature")
            .and_then(|value| value.to_str().ok()),
        Some("sig1=:ZGlyZWN0:")
    );
    assert_eq!(
        observed
            .headers
            .get("signature-input")
            .and_then(|value| value.to_str().ok()),
        Some(signature_input)
    );
    assert_eq!(
        observed
            .headers
            .get("content-digest")
            .and_then(|value| value.to_str().ok()),
        Some(content_digest)
    );
    assert_eq!(
        observed
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some(authorization)
    );
}

#[tokio::test]
async fn score_endpoint_adapts_to_rerank_and_records_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let request_body = r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-adapter-ok", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: serde_json::Value = response.json().await.expect("score json");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "score");
    assert_eq!(body["data"][0]["index"], 0);
    assert!(body["data"][0]["score"].as_f64().is_some());
    assert_eq!(body["model"], "qwen3-reranker-8b");
    assert!(body["created"].as_u64().is_some_and(|created| created > 0));
    assert_eq!(body["usage"]["prompt_tokens"], 0);
    assert_eq!(body["usage"]["total_tokens"], 0);
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert!(body["usage"]["prompt_tokens_details"].is_null());

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("score should forward as rerank");
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/rerank?test=score-adapter-ok");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["query"], "q");
    assert_eq!(observed_body["documents"], serde_json::json!(["d"]));
    assert_eq!(observed_body["top_n"], 1);

    // Drain EOF so buffered observer finalizes.
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        request_row.response_metadata["response_header_content-type"],
        "application/json"
    );
    assert!(
        request_row
            .response_metadata
            .get("response_header_server")
            .is_none()
    );
    assert!(
        request_row
            .response_metadata
            .get("response_header_x-request-id")
            .is_none()
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_content-type"],
        "application/vnd.rerank+json"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_server"],
        "fake-rerank"
    );
    assert_eq!(
        attempt_row.response_metadata["response_header_x-request-id"],
        "rerank-request-123"
    );
    let upstream_body =
        r#"{"id":"rerank-test","model":"qwen3-reranker-8b","results":[{"index":0,"score":1}]}"#;
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        upstream_body.len().to_string()
    );
    assert_ne!(
        request_row.response_metadata["response_body_bytes"],
        attempt_row.response_metadata["response_body_bytes"]
    );
    assert_score_request_body_provenance(
        &proxy.sqlite_path,
        request_body.len(),
        observed.body.len(),
    );
}

fn assert_score_request_body_provenance(
    sqlite_path: &std::path::Path,
    downstream_body_bytes: usize,
    upstream_body_bytes: usize,
) {
    let connection = rusqlite::Connection::open(sqlite_path).expect("sqlite open");
    let request_meta: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request metadata");
    let metadata: serde_json::Value = serde_json::from_str(&request_meta).expect("meta json");
    assert_eq!(metadata["score_via_rerank"], "true");
    assert_eq!(
        metadata["request_body_bytes"],
        downstream_body_bytes.to_string()
    );

    let attempts = read_attempt_request_metadata_rows(sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].request_metadata["upstream_request_body_bytes"],
        upstream_body_bytes.to_string()
    );
    assert_ne!(
        metadata["request_body_bytes"],
        attempts[0].request_metadata["upstream_request_body_bytes"]
    );
}

#[tokio::test]
async fn score_endpoint_rejects_invalid_client_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for request_body in [
        r"{}",
        r#"{"foo":1}"#,
        r#"{"model":"qwen3-reranker-8b","left_input":"q","softmax":true}"#,
        r#"{"softmax":true,"activation":false}"#,
        r#"{"right_input":"d","use_activation":true}"#,
        r#"{"left_input":"q","priority":1}"#,
        r#"{"left_input":"q","truncate_prompt_tokens":8}"#,
        r#"{"left_input":"q","mm_processor_kwargs":{}}"#,
        r#"{"left_input":"q","top_n":1}"#,
        r#"{"left_input":"q","additional_data":{}}"#,
        r#"{"queries":["q"],"typo":true}"#,
        r#"{"items":["d"],"typo":true}"#,
        r#"{"data_1":"q","typo":true}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q"}"#,
        r#"{"query":"q","typo":true}"#,
        r#"{"query":"q","documents":["a","b"],"top_n":"1."}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":42}"#,
        r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":["d",42]}"#,
        r#"{"text_1":["q1","q2"],"text_2":"d"}"#,
        r#"{"text_1":"q","text_2":{"content":[{}]}}"#,
    ] {
        let response = proxy
            .client
            .post(format!("{}/v1/score", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
            .send()
            .await
            .expect("score request should complete");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    let oversized_negative_string = format!(
        r#"{{"query":"q","documents":["d"],"top_n":"-{}"}}"#,
        "9".repeat(4_300)
    );
    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(oversized_negative_string)
        .send()
        .await
        .expect("oversized negative top_n string should complete");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
}

#[tokio::test]
async fn score_endpoint_rejects_malformed_json_numbers_locally() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let request_bodies = [
        format!(
            r#"{{"text_1":"q","text_2":"d","future":{}}}"#,
            "0".repeat(1_000)
        ),
        String::from(r#"{"text_1":"q","text_2":"d","top_n":-+1}"#),
        String::from(r#"{"text_1":"q","text_2":"d","future":1e+}"#),
        String::from(r#"{"text_1":"q","text_2":"d","future":1..2}"#),
    ];
    for (index, request_body) in request_bodies.into_iter().enumerate() {
        let response = proxy
            .client
            .post(format!(
                "{}/v1/score?test=score-malformed-number-{index}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
            .send()
            .await
            .expect("malformed score number should receive a local error");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: serde_json::Value = response.json().await.expect("error response JSON");
        assert_eq!(error["error"]["code"], "invalid_score_request");
    }

    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
}

#[tokio::test]
async fn score_endpoint_rejects_overdepth_known_fields_locally() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let padding = "x".repeat(980 * 1_024);
    let nested_value = format!(
        "{}0.0{}",
        "[".repeat(score_adapter::MAX_SCORE_JSON_DEPTH),
        "]".repeat(score_adapter::MAX_SCORE_JSON_DEPTH)
    );
    let model_body =
        format!(r#"{{"padding":"{padding}","model":{nested_value},"text_1":"q","text_2":"d"}}"#);
    let mut image_embeds_body = String::with_capacity(model_body.len() + 96);
    image_embeds_body.push_str(r#"{"padding":""#);
    image_embeds_body.push_str(&padding);
    image_embeds_body.push_str(
        r#"","text_1":"q","text_2":{"content":[{"type":"image_embeds","image_embeds":{"vector":"#,
    );
    image_embeds_body.push_str(&nested_value);
    image_embeds_body.push_str("}}]}}");
    let overdepth_bodies = [model_body, image_embeds_body];
    for (index, request_body) in overdepth_bodies.into_iter().enumerate() {
        assert!(
            request_body.len() > 950 * 1_024
                && request_body.len() < score_adapter::MAX_SCORE_BODY_BYTES,
            "overdepth fixture {index} must remain near the score body limit"
        );
        let response = proxy
            .client
            .post(format!(
                "{}/v1/score?test=score-overdepth-{index}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
            .send()
            .await
            .expect("overdepth score request should receive a local error");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: serde_json::Value = response.json().await.expect("error response JSON");
        assert_eq!(error["error"]["code"], "invalid_score_request");
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("maximum structure depth")),
            "unexpected overdepth response: {error}"
        );
    }
    assert_no_upstream_request(&mut fake).await;
    let connection = rusqlite::Connection::open(&proxy.sqlite_path).expect("sqlite open");
    assert_eq!(count_rows(&connection, "SELECT COUNT(*) FROM attempts"), 0);
}

#[tokio::test]
async fn score_endpoint_fails_closed_on_partial_rerank_results() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":["a","b"],"top_n":1}"#)
        .send()
        .await
        .expect("score request should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("partial score still reaches upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["top_n"], 2);

    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(attempt_row.status, "failed");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.response_metadata["upstream_response_received"],
        "true"
    );
    assert_eq!(attempt_row.response_metadata["http_status_success"], "true");
    assert_eq!(
        attempt_row.response_metadata["response_process_error"],
        "true"
    );
    let upstream_body = r#"{"id":"rerank-partial","model":"qwen3-reranker-8b","results":[{"index":0,"score":0.9}]}"#;
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        upstream_body.len().to_string()
    );
}

#[tokio::test]
async fn score_endpoint_fails_closed_on_partial_legacy_multimodal_results() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}"#,
        )
        .send()
        .await
        .expect("legacy multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("legacy multimodal score should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(
        observed_body["documents"]["content"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn score_endpoint_accepts_complete_legacy_multimodal_results_with_positive_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/score?test=score-multimodal-top-n",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},"top_n":1}"#,
        )
        .send()
        .await
        .expect("legacy multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body: serde_json::Value = response
        .json()
        .await
        .expect("score response should be valid JSON");
    assert_eq!(response_body["data"].as_array().unwrap().len(), 2);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("legacy multimodal score should reach rerank upstream");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["top_n"], 1);
}

#[tokio::test]
async fn score_endpoint_accepts_truncated_multimodal_results_when_extra_keys_trigger_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score?test=score-partial", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"qwen3-reranker-8b","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"future_extension":true},"top_n":1}"#,
        )
        .send()
        .await
        .expect("extended multimodal score request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body: serde_json::Value = response
        .json()
        .await
        .expect("score response should be valid JSON");
    assert_eq!(response_body["data"].as_array().unwrap().len(), 1);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("extended multimodal score should reach rerank upstream");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("rerank body json");
    assert_eq!(observed_body["documents"]["future_extension"], true);
    assert_eq!(observed_body["top_n"], 1);
}

#[tokio::test]
async fn score_endpoint_accepts_arbitrary_precision_legacy_top_n() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"query":"q","documents":["a","b"],"top_n":18446744073709551616}"#)
        .send()
        .await
        .expect("arbitrary-precision score request should complete");
    assert_eq!(response.status(), StatusCode::OK);

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("arbitrary-precision score request should reach rerank upstream");
    assert!(observed.path_and_query.starts_with("/v1/rerank"));
    assert_eq!(
        std::str::from_utf8(&observed.body).expect("rerank body UTF-8"),
        r#"{"query":"q","documents":["a","b"],"top_n":18446744073709551616}"#
    );
}

#[tokio::test]
async fn score_endpoint_routes_raw_fallback_model_to_named_upstream() {
    let mut default = FakeUpstream::spawn().await;
    let mut named = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[upstream.metadata]
discovery_enabled = false

[[upstreams]]
name = "score-route"
base_url = "{}"
match_models = ["score-model"]

[upstreams.metadata]
discovery_enabled = false
"#,
            named.base_url
        ),
    )
    .await;
    let body = format!(
        r#"{{"model":"score-model","text_1":"q","text_2":"d","future":{}}}"#,
        "9".repeat(1_000)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("named score route should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response should drain");

    let observed = named
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("named upstream should receive adapted score request");
    assert_eq!(observed.path_and_query, "/v1/rerank");
    let observed_body = std::str::from_utf8(&observed.body).expect("rerank body UTF-8");
    assert!(observed_body.contains(r#""model":"score-model""#));
    assert!(observed_body.contains(&format!(r#""future":{}"#, "9".repeat(1_000))));
    assert_no_upstream_request(&mut default).await;
}

#[test]
fn score_model_extraction_uses_raw_fallback_before_policy_and_routing() {
    let body = Bytes::from(format!(
        r#"{{"model":"forbidden-model","text_1":"q","text_2":"d","top_n":{}}}"#,
        "9".repeat(1_000)
    ));
    assert_eq!(
        extract_model_id(&Method::POST, &Uri::from_static("/v1/score"), &body).as_deref(),
        Some("forbidden-model")
    );
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn arbitrary_precision_score_top_n_cannot_bypass_model_policy() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["allowed-model"]
"#,
    )
    .await;
    let body = format!(
        r#"{{"model":"forbidden-model","text_1":"q","text_2":"d","top_n":{}}}"#,
        "9".repeat(1_000)
    );

    let response = proxy
        .client
        .post(format!("{}/v1/score", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("score policy rejection should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("model not allowed"))
    );
    assert_no_upstream_request(&mut fake).await;
}
