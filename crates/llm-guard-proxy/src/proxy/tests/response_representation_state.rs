#[cfg(feature = "guard")]
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[cfg(feature = "guard")]
use axum::{Router, body::Body, extract::State, http::header::CONTENT_TYPE, response::Response};
#[cfg(feature = "guard")]
use futures_util::{StreamExt, stream};
#[cfg(feature = "guard")]
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

use super::*;

const BODY_BOUND_HEADERS: [&str; 13] = [
    "content-encoding",
    "content-md5",
    "digest",
    "content-digest",
    "repr-digest",
    "etag",
    "last-modified",
    "signature",
    "signature-input",
    "if-match",
    "if-none-match",
    "if-modified-since",
    "if-unmodified-since",
];
#[cfg(feature = "guard")]
const FORCED_ALIAS_CONFIG: &str = r#"
[[forced_model_alias_profiles]]
alias = "public-forced-alias"
upstream_model = "canonical-target"
thinking_mode = "force_disable"
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#;

#[cfg(feature = "guard")]
const FORCED_THINKING_ALIAS_LOOP_GUARD_OFF_CONFIG: &str = r#"
[loop_guard]
mode = "disabled"

[retry]
max_attempts = 3
shielded_streaming_enabled = true

[[forced_model_alias_profiles]]
alias = "public-forced-thinking-alias"
upstream_model = "canonical-target"
thinking_mode = "force_thinking"
thinking_budget = 16
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#;

#[cfg(feature = "guard")]
#[derive(Clone)]
struct DelayedAliasSseState {
    first: Bytes,
    second: Bytes,
    first_sent: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    release: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    second_started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[cfg(feature = "guard")]
struct DelayedAliasSseUpstream {
    base_url: String,
    first_sent: oneshot::Receiver<()>,
    release: Option<oneshot::Sender<()>>,
    second_started: oneshot::Receiver<()>,
    server: JoinHandle<()>,
}

#[cfg(feature = "guard")]
impl DelayedAliasSseUpstream {
    async fn spawn(first: &'static [u8], second: &'static [u8]) -> Self {
        let (first_sent_tx, first_sent) = oneshot::channel();
        let (release, release_rx) = oneshot::channel();
        let (second_started_tx, second_started) = oneshot::channel();
        let app = Router::new()
            .fallback(delayed_alias_sse_handler)
            .with_state(DelayedAliasSseState {
                first: Bytes::from_static(first),
                second: Bytes::from_static(second),
                first_sent: Arc::new(Mutex::new(Some(first_sent_tx))),
                release: Arc::new(Mutex::new(Some(release_rx))),
                second_started: Arc::new(Mutex::new(Some(second_started_tx))),
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("delayed SSE upstream should bind");
        let addr = listener
            .local_addr()
            .expect("delayed SSE upstream address should be available");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self {
            base_url: format!("http://{addr}/v1"),
            first_sent,
            release: Some(release),
            second_started,
            server,
        }
    }

    async fn first_frame_sent(&mut self) {
        timeout(STREAM_COMPLETION_TIMEOUT, &mut self.first_sent)
            .await
            .expect("upstream must send the first complete SSE frame")
            .expect("first-frame signal receiver must remain live");
    }

    fn assert_second_frame_is_blocked(&mut self) {
        assert!(
            matches!(
                self.second_started.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ),
            "the upstream must remain blocked before the explicit EOF release"
        );
    }

    fn release(&mut self) {
        self.release
            .take()
            .expect("test may release delayed SSE upstream exactly once")
            .send(())
            .expect("delayed SSE upstream must still await release");
    }

    async fn second_frame_started(&mut self) {
        timeout(STREAM_COMPLETION_TIMEOUT, &mut self.second_started)
            .await
            .expect("released upstream must begin its second SSE frame")
            .expect("second-frame signal receiver must remain live");
    }
}

#[cfg(feature = "guard")]
impl Drop for DelayedAliasSseUpstream {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[cfg(feature = "guard")]
async fn delayed_alias_sse_handler(State(state): State<DelayedAliasSseState>) -> Response {
    let content_length = state.first.len().saturating_add(state.second.len());
    let first_sent = state
        .first_sent
        .lock()
        .expect("first-frame sender lock should not be poisoned")
        .take()
        .expect("delayed SSE upstream should receive one request");
    let release = state
        .release
        .lock()
        .expect("release receiver lock should not be poisoned")
        .take()
        .expect("delayed SSE upstream should receive one request");
    let second_started = state
        .second_started
        .lock()
        .expect("second-frame sender lock should not be poisoned")
        .take()
        .expect("delayed SSE upstream should receive one request");
    let stream = stream::unfold(
        (
            0_u8,
            Some(first_sent),
            Some(release),
            Some(second_started),
            state.first,
            state.second,
        ),
        |(step, first_sent, release, second_started, first, second)| async move {
            match step {
                0 => {
                    first_sent
                        .expect("first-frame sender must remain present")
                        .send(())
                        .expect("test must wait for the first upstream SSE frame");
                    Some((
                        Ok::<Bytes, Infallible>(first),
                        (1, None, release, second_started, Bytes::new(), second),
                    ))
                }
                1 => {
                    release
                        .expect("release receiver must remain present")
                        .await
                        .expect("test must explicitly release delayed upstream EOF");
                    second_started
                        .expect("second-frame sender must remain present")
                        .send(())
                        .expect("test must wait for the released second SSE frame");
                    Some((
                        Ok::<Bytes, Infallible>(second),
                        (2, None, None, None, Bytes::new(), Bytes::new()),
                    ))
                }
                _ => None,
            }
        },
    );
    let mut response = Response::new(Body::from_stream(stream));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    add_stale_body_bound_response_headers(&mut response);
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&content_length.to_string()).expect("test content length is valid"),
    );
    response
}

#[tokio::test]
async fn shielded_non_alias_aggregation_sanitizes_body_bound_headers() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_watchdog_proxy(&fake.base_url).await;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=shielded-aggregate-integrity-headers",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"stream":false}"#,
        )
        .send()
        .await
        .expect("shielded aggregation request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/json"))
    );
    assert_eq!(
        response.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            response.headers().get(header).is_none(),
            "aggregated response must not retain {header}"
        );
    }
    let body = response
        .text()
        .await
        .expect("aggregated response body should be readable");
    let body: serde_json::Value =
        serde_json::from_str(&body).expect("aggregated response should be JSON");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    let _observed = fake.recv_next().await;
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn alias_sse_rewrite_forwards_first_frame_before_upstream_eof() {
    let mut upstream = DelayedAliasSseUpstream::spawn(
        b"data: {\"model\":\"canonical-target\"}\r\r",
        b"data: {\"model\":\"canonical-target\",\"done\":true}\n\n",
    )
    .await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&upstream.base_url, FORCED_ALIAS_CONFIG).await;
    let client = proxy.client.clone();
    let request = tokio::spawn(async move {
        client
            .post(format!("{}/v1/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"public-forced-alias","prompt":"ping","stream":true}"#)
            .send()
            .await
    });

    upstream.first_frame_sent().await;
    let response = timeout(STREAM_COMPLETION_TIMEOUT, request)
        .await
        .expect("downstream headers must not wait for upstream EOF")
        .expect("downstream request task must not panic")
        .expect("alias SSE response should complete its headers");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            response.headers().get(header).is_none(),
            "rewritten SSE must not retain {header}"
        );
    }
    assert_eq!(
        response.headers().get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("text/event-stream"))
    );
    assert!(
        response.headers().get(CONTENT_LENGTH).is_none(),
        "rewritten SSE must strip stale Content-Length"
    );

    let mut body = response.bytes_stream();
    let first = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("rewritten first SSE frame must arrive before upstream EOF")
        .expect("rewritten SSE body must contain a first frame")
        .expect("rewritten first SSE frame must be readable");
    assert_eq!(
        first.as_ref(),
        b"data: {\"model\":\"public-forced-alias\"}\r\r"
    );
    upstream.assert_second_frame_is_blocked();

    upstream.release();
    upstream.second_frame_started().await;
    let second = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("released second SSE frame must reach the client")
        .expect("SSE body must contain the terminal frame")
        .expect("terminal SSE frame must be readable");
    assert_eq!(
        second.as_ref(),
        b"data: {\"done\":true,\"model\":\"public-forced-alias\"}\n\n"
    );
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("SSE body must reach EOF after the released terminal frame")
            .is_none()
    );
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn loop_guard_off_shielded_thinking_chat_relays_first_frame_before_upstream_eof() {
    let mut upstream = DelayedAliasSseUpstream::spawn(
        b"data: {\"model\":\"canonical-target\"}\r\r",
        b"data: {\"model\":\"canonical-target\",\"done\":true}\n\n",
    )
    .await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &upstream.base_url,
        FORCED_THINKING_ALIAS_LOOP_GUARD_OFF_CONFIG,
    )
    .await;
    let client = proxy.client.clone();
    let request = tokio::spawn(async move {
        client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(
                r#"{"model":"public-forced-thinking-alias","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
            )
            .send()
            .await
    });

    upstream.first_frame_sent().await;
    let response = timeout(STREAM_COMPLETION_TIMEOUT, request)
        .await
        .expect("downstream headers must not wait for upstream EOF")
        .expect("downstream request task must not panic")
        .expect("shielded chat response should complete its headers");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.bytes_stream();
    let first = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("loop-guard-off shielded chat must relay before upstream EOF")
        .expect("shielded chat body must contain the first frame")
        .expect("shielded chat first frame must be readable");
    assert_eq!(
        first.as_ref(),
        b"data: {\"model\":\"public-forced-thinking-alias\"}\r\r"
    );
    upstream.assert_second_frame_is_blocked();

    upstream.release();
    upstream.second_frame_started().await;
    let _second = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("released terminal frame must reach the client")
        .expect("shielded chat body must contain the terminal frame")
        .expect("shielded chat terminal frame must be readable");
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn opaque_alias_sse_forwards_first_frame_before_upstream_eof_without_header_changes() {
    let mut upstream =
        DelayedAliasSseUpstream::spawn(b"data: \xFF\r\n\r\n", b"data: opaque-tail\n\n").await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&upstream.base_url, FORCED_ALIAS_CONFIG).await;
    let client = proxy.client.clone();
    let request = tokio::spawn(async move {
        client
            .post(format!("{}/v1/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"public-forced-alias","prompt":"ping","stream":true}"#)
            .send()
            .await
    });

    upstream.first_frame_sent().await;
    let response = timeout(STREAM_COMPLETION_TIMEOUT, request)
        .await
        .expect("opaque downstream headers must not wait for upstream EOF")
        .expect("opaque downstream request task must not panic")
        .expect("opaque alias SSE response should complete its headers");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    assert_eq!(
        response.headers().get(CONTENT_LENGTH),
        Some(&HeaderValue::from_static("30")),
        "opaque streaming bytes must preserve the valid upstream Content-Length"
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            response.headers().get(header).is_some(),
            "opaque no-op SSE must preserve {header}"
        );
    }

    let mut body = response.bytes_stream();
    let first = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("opaque first SSE frame must arrive before upstream EOF")
        .expect("opaque SSE body must contain a first frame")
        .expect("opaque first SSE frame must be readable");
    assert_eq!(first.as_ref(), b"data: \xFF\r\n\r\n");
    upstream.assert_second_frame_is_blocked();

    upstream.release();
    upstream.second_frame_started().await;
    let second = timeout(STREAM_COMPLETION_TIMEOUT, body.next())
        .await
        .expect("released opaque terminal frame must reach the client")
        .expect("opaque SSE body must contain the terminal frame")
        .expect("opaque terminal frame must be readable");
    assert_eq!(second.as_ref(), b"data: opaque-tail\n\n");
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("opaque SSE body must reach EOF after terminal frame")
            .is_none()
    );
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn alias_response_noop_preserves_body_bound_headers_and_bytes() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(&fake.base_url, FORCED_ALIAS_CONFIG).await;

    let cases = [
        (
            "malformed-json",
            r#"{"model":"public-forced-alias","prompt":"ping"}"#,
            "application/json",
            b"{malformed-json".as_slice(),
        ),
        (
            "non-object-json",
            r#"{"model":"public-forced-alias","prompt":"ping"}"#,
            "application/json",
            b"[\"unchanged\"]".as_slice(),
        ),
        (
            "non-utf8-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b"data: \xFF\n\n".as_slice(),
        ),
        (
            "empty-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b"".as_slice(),
        ),
        (
            "comment-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b": keepalive\n\n".as_slice(),
        ),
        (
            "done-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b"data: [DONE]\n\n".as_slice(),
        ),
        (
            "same-model-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b"data: {\"model\":\"public-forced-alias\"}\n\n".as_slice(),
        ),
    ];
    for (case, request_body, content_type, expected_body) in cases {
        let response = proxy
            .client
            .post(format!(
                "{}/v1/completions?test=forced-response-noop-{case}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, content_type)
            .body(request_body)
            .send()
            .await
            .expect("no-op alias response should complete");

        assert_eq!(response.status(), StatusCode::OK, "case={case}");
        assert_eq!(
            response.headers().get("x-safe-custom"),
            Some(&HeaderValue::from_static("preserve-me")),
            "case={case}"
        );
        for header in BODY_BOUND_HEADERS {
            assert!(
                response.headers().get(header).is_some(),
                "no-op alias response must preserve {header}; case={case}"
            );
        }
        assert_eq!(
            response
                .bytes()
                .await
                .expect("body should be readable")
                .as_ref(),
            expected_body,
            "case={case}"
        );
    }
}

#[tokio::test]
async fn buffered_models_noop_preserves_body_bound_headers_but_changed_body_does_not() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
discovery_enabled = true
enrich_responses = true
",
    )
    .await;

    let changed = proxy
        .client
        .get(format!(
            "{}/v1/models?test=models-changed-integrity-headers",
            proxy.base_url
        ))
        .send()
        .await
        .expect("changed models request should complete");
    assert_eq!(changed.status(), StatusCode::OK);
    assert_eq!(
        changed.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            changed.headers().get(header).is_none(),
            "changed models response must not retain {header}"
        );
    }
    let changed_body = changed
        .text()
        .await
        .expect("changed models body should be readable");
    let changed_body: serde_json::Value =
        serde_json::from_str(&changed_body).expect("changed models body should be JSON");
    assert_eq!(changed_body["data"][0]["context_length"], 256_000);
    let _changed_observed = fake.recv_next().await;

    let noop = proxy
        .client
        .get(format!(
            "{}/v1/models?test=models-noop-integrity-headers",
            proxy.base_url
        ))
        .send()
        .await
        .expect("no-op models request should complete");
    assert_eq!(noop.status(), StatusCode::OK);
    assert_eq!(
        noop.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert_eq!(
            noop.headers()
                .get(header)
                .and_then(|value| value.to_str().ok()),
            Some(match header {
                "content-encoding" => "identity",
                "content-md5" => "stale-md5",
                "digest" => "sha-256=stale",
                "content-digest" => "sha-256=:stale:",
                "repr-digest" => "sha-256=:stale-repr:",
                "etag" => "\"stale-etag\"",
                "signature" => "stale-signature",
                "signature-input" => "stale-signature-input",
                "if-match" => "\"stale-match\"",
                "if-none-match" => "\"stale-none-match\"",
                "if-modified-since" | "if-unmodified-since" | "last-modified" => {
                    "Wed, 21 Oct 2015 07:28:00 GMT"
                }
                _ => unreachable!(),
            }),
            "no-op models response must preserve {header}"
        );
    }
    let noop_body = noop
        .text()
        .await
        .expect("no-op models body should be readable");
    assert_eq!(
        noop_body,
        r#"{"object":"list","data":[{"id":"stable-model","object":"model","max_model_len":256000,"context_length":256000,"max_context_length":256000,"owned_by":"vllm"}]}"#
    );
    let _noop_observed = fake.recv_next().await;
}
