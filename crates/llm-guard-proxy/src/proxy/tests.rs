use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use axum::http::header::{AUTHORIZATION, CONNECTION};
use llm_guard_proxy_core::ConfigManager;
use rusqlite::Connection;
use tokio::{net::TcpListener, sync::mpsc};

use super::*;

const TEST_MAX_BYTES: u64 = 1_000_000;
const TEST_PRUNE_TO_BYTES: u64 = 800_000;
const TEST_MAX_RECORDS: u64 = 100;

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
    sender
        .send(observed)
        .await
        .expect("fake upstream observation should send");

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

struct ProxyFixture {
    base_url: String,
    client: Client,
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
        let app = router(state);
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
    std::env::temp_dir().join(format!("llm-guard-proxy-{nanos}-{name}"))
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
