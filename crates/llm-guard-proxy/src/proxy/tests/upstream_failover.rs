use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::oneshot,
};

#[tokio::test]
async fn same_model_request_fails_over_when_primary_is_down() {
    let primary_base_url = closed_upstream_base_url().await;
    let mut backup = FakeUpstream::spawn().await;
    let extra_config = failover_profile_config(
        &primary_base_url,
        Some(&backup.base_url),
        "20ms",
        "10ms",
        "400ms",
    );
    let proxy = spawn_failover_proxy(&backup.base_url, &extra_config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/embeddings", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"same-model","input":"ping"}"#)
        .send()
        .await
        .expect("failover request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    response.bytes().await.expect("response body should drain");
    let probe = backup.recv_next().await;
    assert_eq!(probe.method, Method::GET);
    assert_eq!(probe.path_and_query, "/v1/models");
    let forwarded = backup.recv_next().await;
    assert_eq!(forwarded.method, Method::POST);
    assert_eq!(forwarded.path_and_query, "/v1/embeddings");
}

#[tokio::test]
async fn same_model_request_waits_for_primary_to_recover() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("recovering upstream should bind");
    let addr = listener
        .local_addr()
        .expect("recovering upstream address should be available");
    let primary_base_url = format!("http://{addr}/v1");
    let extra_config = failover_profile_config(&primary_base_url, None, "20ms", "10ms", "600ms");
    let proxy = spawn_failover_proxy(&primary_base_url, &extra_config).await;
    let client = proxy.client.clone();
    let proxy_base_url = proxy.base_url.clone();
    let started_at = Instant::now();
    let request = tokio::spawn(async move {
        client
            .post(format!("{proxy_base_url}/v1/embeddings"))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"same-model","input":"ping"}"#)
            .send()
            .await
            .expect("keep-alive request should complete")
    });

    sleep(Duration::from_millis(80)).await;
    let mut recovered = spawn_fake_upstream_on_listener(listener);
    let response = timeout(Duration::from_secs(2), request)
        .await
        .expect("keep-alive request should finish")
        .expect("keep-alive task should not panic");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(started_at.elapsed() >= Duration::from_millis(80));
    response.bytes().await.expect("response body should drain");

    let mut saw_probe = false;
    let mut saw_forwarded_request = false;
    while let Some(observed) = recovered.recv_within(Duration::from_millis(200)).await {
        saw_probe |= observed.method == Method::GET && observed.path_and_query == "/v1/models";
        saw_forwarded_request |=
            observed.method == Method::POST && observed.path_and_query == "/v1/embeddings";
        if saw_probe && saw_forwarded_request {
            break;
        }
    }
    assert!(
        saw_probe,
        "recovered upstream should receive a readiness probe"
    );
    assert!(
        saw_forwarded_request,
        "held request should be forwarded after recovery"
    );
}

#[tokio::test]
async fn connection_refused_after_ready_probe_retries_on_failover() {
    let (primary_base_url, primary_probe_seen) = spawn_probe_then_stop_upstream().await;
    let mut backup = FakeUpstream::spawn().await;
    let extra_config = failover_profile_config(
        &primary_base_url,
        Some(&backup.base_url),
        "20ms",
        "50ms",
        "1s",
    );
    let proxy = spawn_failover_proxy(&backup.base_url, &extra_config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/embeddings", proxy.base_url))
        .json(&json!({"model": "same-model", "input": "retry"}))
        .send()
        .await
        .expect("request should retry after the primary connection is refused");

    assert_eq!(response.status(), StatusCode::OK);
    primary_probe_seen
        .await
        .expect("primary should receive the initial readiness probe");
    let probe = backup.recv_next().await;
    assert_eq!(probe.path_and_query, "/v1/models");
    let forwarded = backup.recv_next().await;
    assert_eq!(forwarded.path_and_query, "/v1/embeddings");
}

#[tokio::test]
async fn burst_requests_share_cached_health_probe() {
    let primary_base_url = closed_upstream_base_url().await;
    let mut backup = FakeUpstream::spawn().await;
    let extra_config = failover_profile_config(
        &primary_base_url,
        Some(&backup.base_url),
        "200ms",
        "20ms",
        "1s",
    );
    let proxy = spawn_failover_proxy(&backup.base_url, &extra_config).await;

    let mut requests = Vec::new();
    for _ in 0..8 {
        let client = proxy.client.clone();
        let url = format!("{}/v1/embeddings", proxy.base_url);
        requests.push(tokio::spawn(async move {
            client
                .post(url)
                .json(&json!({"model": "same-model", "input": "burst"}))
                .send()
                .await
                .expect("burst request should complete")
                .status()
        }));
    }
    for request in requests {
        assert_eq!(
            request.await.expect("request task should join"),
            StatusCode::OK
        );
    }

    let mut health_probes = 0;
    let mut forwarded_requests = 0;
    while let Ok(observed) = backup.receiver.try_recv() {
        if observed.path_and_query == "/v1/models" {
            health_probes += 1;
        } else if observed.path_and_query == "/v1/embeddings" {
            forwarded_requests += 1;
        }
    }
    assert_eq!(health_probes, 1, "concurrent requests must coalesce probes");
    assert_eq!(forwarded_requests, 8);
}

#[tokio::test]
async fn same_model_request_returns_service_unavailable_after_probe_deadline() {
    let primary_base_url = closed_upstream_base_url().await;
    let extra_config = failover_profile_config(&primary_base_url, None, "20ms", "1s", "80ms");
    let proxy = spawn_failover_proxy(&primary_base_url, &extra_config).await;
    let started_at = Instant::now();

    let response = proxy
        .client
        .post(format!("{}/v1/embeddings", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"same-model","input":"ping"}"#)
        .send()
        .await
        .expect("bounded keep-alive request should complete");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(started_at.elapsed() >= Duration::from_millis(60));
    assert!(started_at.elapsed() < Duration::from_secs(1));
    let error: serde_json::Value = response
        .json()
        .await
        .expect("unavailable response should be JSON");
    assert_eq!(error["error"]["type"], "upstream_unavailable");
}

async fn spawn_probe_then_stop_upstream() -> (String, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe-only upstream should bind");
    let addr = listener
        .local_addr()
        .expect("probe-only upstream address should be available");
    let (probe_seen_tx, probe_seen_rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener
            .accept()
            .await
            .expect("probe-only upstream should accept the health probe");
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        loop {
            let bytes_read = socket
                .read(&mut chunk)
                .await
                .expect("probe-only upstream should read request");
            if bytes_read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..bytes_read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        assert!(
            request.starts_with(b"GET /v1/models "),
            "the first request must be the readiness probe"
        );
        let body = r#"{"object":"list","data":[{"id":"same-model","object":"model"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("probe-only upstream should write response");
        socket
            .shutdown()
            .await
            .expect("probe-only upstream should close after readiness");
        let _ = probe_seen_tx.send(());
    });
    (format!("http://{addr}/v1"), probe_seen_rx)
}

async fn closed_upstream_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("closed upstream address should bind");
    let addr = listener
        .local_addr()
        .expect("closed upstream address should be available");
    drop(listener);
    format!("http://{addr}/v1")
}

fn failover_profile_config(
    primary_base_url: &str,
    backup_base_url: Option<&str>,
    interval: &str,
    probe_timeout: &str,
    max_wait: &str,
) -> String {
    let backup = backup_base_url.map_or_else(String::new, |base_url| {
        format!(
            r#"
[[profile.upstream]]
base_url = "{base_url}"
priority = "failover"
"#
        )
    });
    format!(
        r#"
[[profile]]
model = "same-model"
health_probe_interval = "{interval}"
health_probe_timeout = "{probe_timeout}"
health_probe_max_wait = "{max_wait}"

[[profile.upstream]]
base_url = "{primary_base_url}"
priority = "primary"
{backup}
"#
    )
}

async fn spawn_failover_proxy(default_upstream: &str, extra_config: &str) -> ProxyFixture {
    ProxyFixture::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
        upstream_base_url: default_upstream,
        observability_enabled: false,
        max_in_flight_requests: AppConfig::default().server.max_in_flight_requests,
        server_config: "",
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config,
    })
    .await
}

fn spawn_fake_upstream_on_listener(listener: TcpListener) -> FakeUpstream {
    let (sender, receiver) = mpsc::channel(10);
    let app = Router::new()
        .fallback(fake_upstream_handler)
        .with_state(FakeUpstreamState {
            sender,
            changing_model_len: Arc::new(AtomicU64::new(128_000)),
            attempt_counts: Arc::new(Mutex::new(HashMap::new())),
            models_body: None,
            models_status: StatusCode::OK,
            models_label: "models",
            models_delay: None,
        });
    let addr = listener
        .local_addr()
        .expect("recovering upstream address should be available");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("recovering upstream server failed: {error}");
        }
    });
    FakeUpstream {
        base_url: format!("http://{addr}/v1"),
        receiver,
    }
}
