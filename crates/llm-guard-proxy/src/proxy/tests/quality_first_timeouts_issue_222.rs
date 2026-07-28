use super::*;

#[tokio::test]
async fn shielded_chat_request_deadline_bounds_body_routing_queue_wait() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000

[retry]
request_deadline_ms = 20

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

    let (active_request, active_body_polled) =
        tracked_pending_json_request("/v1/completions?slot=active");
    let active = tokio::spawn(proxy_handler(State(proxy.state.clone()), active_request));
    wait_for_flag(
        &active_body_polled,
        "active request to occupy the body-routing permit",
    )
    .await;

    let queued_body_polled = Arc::new(AtomicBool::new(false));
    let queued_body = Body::from_stream(stream::once({
        let queued_body_polled = Arc::clone(&queued_body_polled);
        async move {
            queued_body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(
                br#"{"model":"test-chat","messages":[{"role":"user","content":"queued"}]}"#,
            ))
        }
    }));
    let queued_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions?slot=deadline")
        .header(CONTENT_TYPE, "application/json")
        .body(queued_body)
        .expect("queued shielded chat request should build");
    let queued_response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), queued_request),
    )
    .await
    .expect("shielded request deadline should bound body-routing queue wait");

    assert_eq!(queued_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let queued_body = to_bytes(queued_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("admission timeout response body should read");
    let queued_body = String::from_utf8(queued_body.to_vec())
        .expect("admission timeout response body should be utf-8");
    assert!(
        queued_body.contains("proxy_generation_queue_timeout"),
        "admission timeout should be reported by the proxy: {queued_body}"
    );
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "a shielded request whose admission budget expires must not read its body"
    );
    assert_no_upstream_request(&mut fake).await;

    active.abort();
    assert!(
        active
            .await
            .expect_err("active request should be cancelled")
            .is_cancelled()
    );
}
