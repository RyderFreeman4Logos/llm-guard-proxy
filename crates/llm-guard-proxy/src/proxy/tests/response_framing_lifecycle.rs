use std::{collections::BTreeMap, sync::Arc, time::Duration};

use axum::http::header::{CONNECTION, CONTENT_LENGTH};
use futures_util::{StreamExt, stream};
use tokio::{sync::oneshot, time::timeout};

use super::*;

#[tokio::test]
async fn connection_nominated_content_length_cannot_complete_dropped_body() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(CONNECTION, HeaderValue::from_static("content-length"));
    upstream_headers.insert(CONTENT_LENGTH, HeaderValue::from_static("3"));
    let final_headers = downstream_response_headers(&upstream_headers, false, None);
    assert!(
        final_headers.get(CONTENT_LENGTH).is_none(),
        "Connection-nominated Content-Length must not reach the downstream response"
    );
    let known_content_length =
        response_lifecycle_content_length(&Method::GET, reqwest::StatusCode::OK, &final_headers);
    assert_eq!(known_content_length, None);

    let (extra_staged_tx, extra_staged_rx) = oneshot::channel();
    let (allow_extra_tx, allow_extra_rx) = oneshot::channel();
    let upstream = stream::unfold(
        (0_u8, Some(extra_staged_tx), Some(allow_extra_rx)),
        |(step, extra_staged, allow_extra)| async move {
            match step {
                0 => Some((
                    Ok::<Bytes, reqwest::Error>(Bytes::from_static(b"abc")),
                    (1, extra_staged, allow_extra),
                )),
                1 => {
                    extra_staged
                        .expect("extra-frame marker must remain present")
                        .send(())
                        .expect("test must wait for the causally blocked extra frame");
                    allow_extra
                        .expect("extra-frame release must remain present")
                        .await
                        .expect("test cleanup must release the extra frame");
                    Some((Ok(Bytes::from_static(b"!")), (2, None, None)))
                }
                _ => None,
            }
        },
    );
    let request_id = RequestId::generate();
    let response_parts = ForwardedResponseParts {
        config: proxy.state.config.clone(),
        store: proxy.state.store.clone(),
        evidence_store: proxy.state.evidence_store.clone(),
        persistence_tasks: Arc::clone(&proxy.state.persistence_tasks),
        request_id: request_id.clone(),
        started_at_unix_ms: unix_time_millis(),
        attempt_id: AttemptId::for_request(&request_id, 1),
        attempt_number: 1,
        attempt_max_attempts: 1,
        attempt_started_at_unix_ms: unix_time_millis(),
        upstream_mode: upstream_mode_from_headers(&upstream_headers),
        model_id: Some(String::from("test-model")),
        input_fingerprint: None,
        upstream_status: reqwest::StatusCode::OK,
        upstream_headers,
        request_metadata: BTreeMap::new(),
        attempt_request_metadata: BTreeMap::new(),
        completed_attempt_records: Vec::new(),
        shutdown: Arc::clone(&proxy.state.shutdown),
        stuck_watchdog_attempt: None,
    };
    let mut downstream = ObservedUpstreamBody::new(
        upstream,
        response_parts.into_observer(),
        InFlightPermit { limiter: None },
        proxy.state.shutdown.subscribe(),
        known_content_length,
    );
    assert_eq!(
        downstream
            .next()
            .await
            .expect("first upstream chunk should reach downstream")
            .expect("first upstream chunk should be readable"),
        Bytes::from_static(b"abc")
    );
    timeout(Duration::from_secs(1), extra_staged_rx)
        .await
        .expect("relay must reach the later, blocked extra frame")
        .expect("extra-frame marker sender must remain live");
    drop(downstream);
    let _released = allow_extra_tx.send(());

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(attempt_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        attempt_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
}
