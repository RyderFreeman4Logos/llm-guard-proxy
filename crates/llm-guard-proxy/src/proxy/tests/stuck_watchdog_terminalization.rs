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
async fn independent_relay_stops_at_the_shared_retained_byte_budget_then_resumes_in_order() {
    // This crosses the channel boundary: sixteen chunks fill the channel and the
    // remaining eight fill the pending queue before the twenty-fifth must wait
    // for a downstream receive to release the one shared byte budget.
    const RETAINED_BYTE_BUDGET: usize = 384 * 1024;
    const CHUNK_BYTES: usize = 16 * 1024;
    const CHUNKS_AT_BUDGET: usize = RETAINED_BYTE_BUDGET / CHUNK_BYTES;

    let (budget_reached_tx, budget_reached_rx) = oneshot::channel();
    let (polled_past_budget_tx, mut polled_past_budget_rx) = oneshot::channel();
    let upstream = stream::unfold(
        (
            0_usize,
            Some(budget_reached_tx),
            Some(polled_past_budget_tx),
        ),
        |(index, budget_reached, polled_past_budget)| async move {
            if index > CHUNKS_AT_BUDGET {
                return None;
            }
            let mut budget_reached = budget_reached;
            let mut polled_past_budget = polled_past_budget;
            if index + 1 == CHUNKS_AT_BUDGET {
                let _sent = budget_reached
                    .take()
                    .expect("budget marker must be sent once")
                    .send(());
            }
            if index == CHUNKS_AT_BUDGET {
                let _sent = polled_past_budget
                    .take()
                    .expect("post-budget poll marker must be sent once")
                    .send(());
            }
            Some((
                Ok::<Bytes, reqwest::Error>(Bytes::from(vec![
                    u8::try_from(index).expect(
                        "numbered test chunk index must fit in u8"
                    );
                    CHUNK_BYTES
                ])),
                (index + 1, budget_reached, polled_past_budget),
            ))
        },
    );
    let downstream = observe_upstream_body_independently(upstream, None);
    futures_util::pin_mut!(downstream);

    timeout(Duration::from_secs(1), budget_reached_rx)
        .await
        .expect("upstream must reach the aggregate byte budget")
        .expect("aggregate byte-budget marker sender must remain live");
    assert!(
        timeout(Duration::from_millis(100), &mut polled_past_budget_rx)
            .await
            .is_err(),
        "the relay must not poll byte {CHUNK_BYTES}-sized chunk {} before downstream frees the shared {RETAINED_BYTE_BUDGET}-byte budget",
        CHUNKS_AT_BUDGET + 1
    );

    let first = timeout(Duration::from_secs(1), downstream.next())
        .await
        .expect("first retained chunk should be readable after the pause")
        .expect("relay should keep the first retained chunk")
        .expect("retained upstream chunk should stay successful");
    assert_eq!(first.as_ref(), vec![0_u8; CHUNK_BYTES]);
    timeout(Duration::from_secs(1), &mut polled_past_budget_rx)
        .await
        .expect("a downstream receive must release byte budget for the next upstream poll")
        .expect("post-budget poll marker sender must remain live");

    for expected_index in 1..=CHUNKS_AT_BUDGET {
        let chunk = timeout(Duration::from_secs(1), downstream.next())
            .await
            .expect("retained chunk should drain in order")
            .expect("relay should retain every chunk")
            .expect("retained upstream chunk should stay successful");
        assert_eq!(
            chunk.as_ref(),
            vec![
                u8::try_from(expected_index).expect("numbered test chunk index must fit in u8");
                CHUNK_BYTES
            ]
        );
    }
    assert!(
        downstream.next().await.is_none(),
        "finite upstream must close after the retained chunks drain"
    );
}

#[tokio::test]
async fn independent_relay_cancels_upstream_when_paused_downstream_drops_at_shared_budget() {
    const RETAINED_BYTE_BUDGET: usize = 384 * 1024;
    const CHUNK_BYTES: usize = 16 * 1024;
    const CHUNKS_AT_BUDGET: usize = RETAINED_BYTE_BUDGET / CHUNK_BYTES;

    let (budget_reached_tx, budget_reached_rx) = oneshot::channel();
    let (upstream_drop_tx, upstream_drop_rx) = oneshot::channel::<()>();
    let upstream = stream::unfold(
        (0_usize, Some(budget_reached_tx), upstream_drop_tx),
        |(index, budget_reached, upstream_drop)| async move {
            let mut budget_reached = budget_reached;
            if index + 1 == CHUNKS_AT_BUDGET {
                let _sent = budget_reached
                    .take()
                    .expect("budget marker must be sent once")
                    .send(());
            }
            Some((
                Ok::<Bytes, reqwest::Error>(Bytes::from(vec![
                    u8::try_from(index).expect(
                        "numbered test chunk index must fit in u8"
                    );
                    CHUNK_BYTES
                ])),
                (index + 1, budget_reached, upstream_drop),
            ))
        },
    );
    let downstream = observe_upstream_body_independently(upstream, None);

    timeout(Duration::from_secs(1), budget_reached_rx)
        .await
        .expect("relay must fill the shared byte budget before downstream cancellation")
        .expect("aggregate byte-budget marker sender must remain live");
    drop(downstream);

    let upstream_drop = timeout(Duration::from_secs(1), upstream_drop_rx)
        .await
        .expect("dropping the paused downstream body must cancel upstream polling");
    assert!(
        upstream_drop.is_err(),
        "upstream drop sender must never send a value"
    );
}

#[tokio::test]
async fn independent_relay_admits_one_oversized_chunk_then_waits_for_downstream_release() {
    const RETAINED_BYTE_BUDGET: usize = 384 * 1024;
    let oversized = Bytes::from(vec![b'o'; RETAINED_BYTE_BUDGET + 1]);
    let trailing = Bytes::from_static(b"after-oversized");
    let (oversized_polled_tx, oversized_polled_rx) = oneshot::channel();
    let (trailing_polled_tx, mut trailing_polled_rx) = oneshot::channel();
    let upstream = stream::unfold(
        (
            0_u8,
            Some(oversized_polled_tx),
            Some(trailing_polled_tx),
            oversized.clone(),
            trailing.clone(),
        ),
        |(step, oversized_polled, trailing_polled, oversized, trailing)| async move {
            let mut oversized_polled = oversized_polled;
            let mut trailing_polled = trailing_polled;
            match step {
                0 => {
                    let _sent = oversized_polled
                        .take()
                        .expect("oversized marker must be sent once")
                        .send(());
                    Some((
                        Ok::<Bytes, reqwest::Error>(oversized),
                        (1, oversized_polled, trailing_polled, Bytes::new(), trailing),
                    ))
                }
                1 => {
                    let _sent = trailing_polled
                        .take()
                        .expect("trailing marker must be sent once")
                        .send(());
                    Some((
                        Ok::<Bytes, reqwest::Error>(trailing),
                        (
                            2,
                            oversized_polled,
                            trailing_polled,
                            Bytes::new(),
                            Bytes::new(),
                        ),
                    ))
                }
                _ => None,
            }
        },
    );
    let downstream = observe_upstream_body_independently(upstream, None);
    futures_util::pin_mut!(downstream);

    timeout(Duration::from_secs(1), oversized_polled_rx)
        .await
        .expect("oversized upstream chunk must be observed")
        .expect("oversized marker sender must remain live");
    assert!(
        timeout(Duration::from_millis(100), &mut trailing_polled_rx)
            .await
            .is_err(),
        "one oversized retained chunk must consume the entire budget until downstream receives it"
    );

    let received_oversized = timeout(Duration::from_secs(1), downstream.next())
        .await
        .expect("oversized chunk should remain available to the client")
        .expect("relay must not drop the oversized chunk")
        .expect("oversized chunk should remain successful");
    assert_eq!(received_oversized, oversized);
    timeout(Duration::from_secs(1), &mut trailing_polled_rx)
        .await
        .expect("receiving the oversized chunk must release the relay budget")
        .expect("trailing marker sender must remain live");
    assert_eq!(
        downstream
            .next()
            .await
            .expect("trailing chunk should remain ordered")
            .expect("trailing chunk should stay successful"),
        trailing
    );
}

#[tokio::test]
async fn terminal_json_progress_has_one_owner_when_downstream_drop_races_staged_upstream_eof() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let tracker = Arc::clone(&proxy.state.stuck_watchdog_tokens);
    let profile = "terminal-progress-race";
    let request =
        tracker.watch_request(profile, WatchdogProgressUnit::Chat, Duration::from_secs(1));
    let response = Bytes::from_static(
        br#"{"id":"complete","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}"#,
    );
    let (eof_staged_tx, eof_staged_rx) = oneshot::channel();
    let (allow_eof_tx, allow_eof_rx) = oneshot::channel();
    let upstream = stream::unfold(
        (
            0_u8,
            Some(eof_staged_tx),
            Some(allow_eof_rx),
            response.clone(),
        ),
        |(step, eof_staged, allow_eof, response)| async move {
            match step {
                0 => Some((
                    Ok::<Bytes, reqwest::Error>(response.clone()),
                    (1, eof_staged, allow_eof, response),
                )),
                1 => {
                    eof_staged
                        .expect("EOF staging sender must remain present")
                        .send(())
                        .expect("test must still be waiting for the staged EOF");
                    allow_eof
                        .expect("EOF release receiver must remain present")
                        .await
                        .expect("test must release the staged upstream EOF");
                    None
                }
                _ => None,
            }
        },
    );
    let request_id = RequestId::generate();
    let upstream_headers = HeaderMap::new();
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
        stuck_watchdog_attempt: Some(request.begin_attempt()),
    };
    let mut downstream = ObservedUpstreamBody::new(
        upstream,
        response_parts.into_observer(),
        InFlightPermit { limiter: None },
        proxy.state.shutdown.subscribe(),
    );
    let delivered = timeout(Duration::from_secs(1), downstream.next())
        .await
        .expect("final JSON bytes should reach the downstream body")
        .expect("final JSON chunk should be retained")
        .expect("final JSON chunk should remain successful");
    assert_eq!(delivered, response);

    timeout(Duration::from_secs(1), eof_staged_rx)
        .await
        .expect("relay must stage the upstream EOF after delivering the final JSON")
        .expect("EOF staging sender must remain live");
    // Both real terminal paths are now runnable over the same lease: release EOF
    // immediately before dropping the downstream body. `Drop` records the body,
    // while the relay records the upstream EOF on its dedicated task.
    let _released = allow_eof_tx.send(());
    drop(downstream);
    timeout(Duration::from_secs(1), async {
        while tracker.has_active_requests(profile) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("upstream EOF must terminalize the staged request");
    assert_eq!(
        tracker.sample_count(profile),
        1,
        "downstream-drop and upstream-EOF terminal paths must own exactly one progress sample"
    );
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
