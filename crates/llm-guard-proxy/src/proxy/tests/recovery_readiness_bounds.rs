use super::*;

async fn probe_body(body: Body) -> Result<bool, String> {
    let body = Arc::new(Mutex::new(Some(body)));
    let app = Router::new().fallback(move || {
        let body = Arc::clone(&body);
        async move { Response::new(body.lock().expect("body lock").take().expect("one probe")) }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let policy = LocalRecoveryPolicy::from_config(&LocalRecoveryConfig::default());
    let result = send_local_recovery_readiness_probe(
        &Client::new(),
        &format!("http://{address}/v1"),
        &policy,
    )
    .await;
    server.abort();
    let _ = server.await;
    result
}

#[tokio::test]
async fn readiness_rejects_chunked_overflow_without_content_length() {
    // 64 KiB plus one byte, not a large-allocation/OOM experiment.
    let prefix =
        Bytes::from_static(b"{\"choices\":[{\"message\":{\"content\":\"ready\"}}],\"padding\":\"");
    let chunks = vec![
        prefix,
        Bytes::from(vec![b'x'; 64 * 1024]),
        Bytes::from_static(b"\"}"),
    ];
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::io::Error>),
    ));
    assert!(
        probe_body(body).await.is_err(),
        "chunked readiness must reject at its hard byte bound"
    );
}

#[tokio::test]
async fn readiness_rejects_unusable_choices_and_preserves_supported_variants() {
    for choice in [
        json!(null),
        json!(1),
        json!("scalar"),
        json!({}),
        json!({"message":{}}),
    ] {
        let body = Body::from(json!({"choices":[choice]}).to_string());
        assert!(
            !probe_body(body).await.expect("valid JSON transport"),
            "unusable choice admitted"
        );
    }
    for choice in [
        json!({"message":{"role":"assistant","content":"ready"},"vendor_extension":true}),
        json!({"text":"ready"}),
        json!({"message":{"content":null,"tool_calls":[{"id":"call","type":"function","function":{"name":"ready","arguments":"{}"}}]}}),
        json!({"message":{"function_call":{"name":"ready","arguments":"{}"}}}),
    ] {
        assert!(
            probe_body(Body::from(json!({"choices":[choice]}).to_string()))
                .await
                .expect("probe")
        );
    }
}
