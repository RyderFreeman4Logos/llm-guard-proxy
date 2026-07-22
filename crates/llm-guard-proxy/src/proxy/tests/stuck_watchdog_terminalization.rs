use super::*;

#[tokio::test]
async fn independent_relay_ends_watchdog_lease_at_upstream_eof_while_downstream_is_unread() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let detection_window = Duration::from_millis(50);
    let request = tracker.watch_request(
        "finite-under-capacity",
        WatchdogProgressUnit::Chat,
        detection_window,
    );
    let lease = request.begin_attempt();
    let _unread_downstream = observe_upstream_body_independently(
        stream::iter([Ok::<Bytes, reqwest::Error>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n",
        ))]),
        Some(lease.progress_request()),
    );

    // The finite upstream fits entirely in the relay. Keep the downstream body
    // unread past several detection windows to prove upstream EOF owns the lease.
    sleep(Duration::from_millis(200)).await;

    assert!(
        !tracker.has_active_requests("finite-under-capacity"),
        "upstream EOF must end watchdog activity even while downstream retains buffered bytes"
    );
    assert!(
        !tracker.has_too_few_output_progress_units("finite-under-capacity", detection_window, 1),
        "a completed finite upstream must not become a stale active watchdog attempt"
    );
    drop(lease);
}

#[tokio::test]
async fn independent_relay_records_one_non_sse_chat_or_completion_progress_at_upstream_eof() {
    for (profile, progress_unit, response) in [
        (
            "non-sse-chat",
            WatchdogProgressUnit::Chat,
            br#"{"id":"chat","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}"#
                .as_slice(),
        ),
        (
            "non-sse-completion",
            WatchdogProgressUnit::Completion,
            br#"{"id":"completion","object":"text_completion","choices":[{"index":0,"text":"done","finish_reason":"stop"}]}"#
                .as_slice(),
        ),
    ] {
        let tracker = Arc::new(StuckWatchdogTokenTracker::default());
        let request = tracker.watch_request(profile, progress_unit, Duration::from_secs(1));
        let lease = request.begin_attempt();
        let _unread_downstream = observe_upstream_body_independently(
            stream::iter([Ok::<Bytes, reqwest::Error>(Bytes::copy_from_slice(response))]),
            Some(lease.progress_request()),
        );

        timeout(Duration::from_secs(1), async {
            while tracker.sample_count(profile) < 1 {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("non-SSE progress must be recorded at upstream EOF before downstream draining");
        assert_eq!(
            tracker.sample_count(profile),
            1,
            "a complete non-SSE {progress_unit:?} response must contribute exactly one upstream-time progress sample"
        );
        assert!(
            !tracker.has_active_requests(profile),
            "upstream EOF must terminalize the non-SSE attempt while downstream remains unread"
        );
        drop(lease);
    }
}

#[tokio::test]
async fn buffered_adapter_ends_watchdog_lease_before_constructing_unread_downstream_body() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let tracker = Arc::clone(&proxy.state.stuck_watchdog_tokens);
    let watchdog_request = tracker.watch_request(
        "buffered-adapter",
        WatchdogProgressUnit::Reranker,
        Duration::from_millis(50),
    );
    let request_id = RequestId::generate();
    let upstream_response = proxy
        .state
        .client
        .post(format!("{}/v1/rerank", fake.base_url))
        .body("{}")
        .send()
        .await
        .expect("fake reranker response should be available");
    let upstream_status = upstream_response.status();
    let upstream_headers = upstream_response.headers().clone();
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
        model_id: Some(String::from("qwen3-reranker-8b")),
        input_fingerprint: None,
        upstream_status,
        upstream_headers,
        request_metadata: BTreeMap::new(),
        attempt_request_metadata: BTreeMap::new(),
        completed_attempt_records: Vec::new(),
        shutdown: Arc::clone(&proxy.state.shutdown),
        stuck_watchdog_attempt: Some(watchdog_request.begin_attempt()),
    };

    let unread_downstream = rewrite_buffered_adapter_response_from_upstream(
        response_parts,
        upstream_response,
        InFlightPermit { limiter: None },
        BufferedResponseAdapter::ScoreFromRerank(None),
        Some("qwen3-reranker-8b"),
    )
    .await
    .expect("buffered adapter should rewrite the complete reranker response");

    assert!(
        !tracker.has_active_requests("buffered-adapter"),
        "buffered adapters must end their lease once their upstream body is fully read, before downstream body consumption"
    );
    drop(unread_downstream);
}
