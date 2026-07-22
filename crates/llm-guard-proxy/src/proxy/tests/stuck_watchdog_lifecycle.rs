use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use axum::{Router, extract::State, routing::post};

use super::*;

const WATCHDOG_WINDOW: Duration = Duration::from_secs(1);
const WATCHDOG_TASK_TIMEOUT: Duration = Duration::from_secs(3);

#[test]
fn watchdog_recognizes_every_supported_chat_delta_progress_field() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let deltas = [
        serde_json::json!({"content": "answer"}),
        serde_json::json!({"reasoning_content": "reasoning"}),
        serde_json::json!({"reasoning": "reasoning"}),
        serde_json::json!({"thinking": "thinking"}),
        serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }]
        }),
        serde_json::json!({
            "function_call": {"name": "legacy_lookup", "arguments": "{}"}
        }),
    ];

    for (index, delta) in deltas.into_iter().enumerate() {
        let profile = format!("delta-{index}");
        let request = tracker.begin_request(&profile, WatchdogProgressUnit::Chat, WATCHDOG_WINDOW);
        request.record_emitted_chunk(&chat_delta_sse(&delta));
        assert_eq!(
            tracker.sample_count(&profile),
            1,
            "supported delta field at index {index} must count as upstream progress"
        );
    }
}

#[test]
fn watchdog_maps_completions_and_records_text_progress() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let completions_uri: Uri = "/v1/completions".parse().expect("URI should parse");
    let progress_unit = watchdog_progress_unit(&completions_uri);
    let request = tracker.watch_request("completions", progress_unit, WATCHDOG_WINDOW);

    assert_eq!(progress_unit, WatchdogProgressUnit::Completion);
    assert!(
        request.record_emitted_chunk(
            b"data: {\"choices\":[{\"text\":\"healthy completion text\"}]}\n\n",
        )
    );
    assert_eq!(
        tracker.sample_count("completions"),
        1,
        "a completion text chunk must count as model progress"
    );
}

#[test]
fn watchdog_non_chat_sse_after_comment_records_later_result_event() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.watch_request(
        "reranker-heartbeat",
        WatchdogProgressUnit::Reranker,
        WATCHDOG_WINDOW,
    );

    assert!(request.record_emitted_chunk(
        b": ping\n\ndata: {\"results\":[{\"index\":0,\"relevance_score\":0.9}]}\n\n",
    ));
    assert_eq!(
        tracker.sample_count("reranker-heartbeat"),
        1,
        "an SSE heartbeat must not prevent a later result event from counting as progress"
    );
}

#[test]
fn watchdog_request_snapshot_does_not_prune_shared_samples_after_reload() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let profile = "current-window";
    let current_config_request = tracker.watch_request(
        profile,
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1_800),
    );
    assert!(current_config_request.record_emitted_chunk(
        br#"data: {"choices":[{"delta":{"content":"kept"}}]}

    "#,
    ));

    let stale_config_request =
        tracker.watch_request(profile, WatchdogProgressUnit::Chat, Duration::ZERO);
    assert!(stale_config_request.record_emitted_chunk(
        br#"data: {"choices":[{"delta":{"content":"new"}}]}

    "#,
    ));

    assert_eq!(
        tracker.sample_count(profile),
        2,
        "an old request's detection-window snapshot must not prune shared profile samples"
    );
    assert!(
        !tracker.has_too_few_output_progress_units(profile, Duration::from_secs(1_800), 2),
        "the current watchdog window must retain both currently valid samples"
    );
}

#[tokio::test]
async fn watchdog_completed_attempt_lease_ends_before_failover_selection_wait() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let profile = "physical-attempt";
    let request =
        tracker.watch_request(profile, WatchdogProgressUnit::Chat, Duration::from_secs(1));
    let mut completed_attempt = Some(request.begin_attempt());
    assert!(tracker.has_active_requests(profile));

    // The retryable status or transport failure completed this physical attempt.
    end_stuck_watchdog_attempt(&mut completed_attempt);
    assert!(completed_attempt.is_none());
    assert!(!tracker.has_active_requests(profile));

    // The successor selection can outlast a watchdog window without reviving this lease.
    let selection_wait = Duration::from_millis(40);
    tokio::time::sleep(selection_wait).await;
    assert!(
        !tracker.has_too_few_output_progress_units(profile, selection_wait, 1),
        "a completed attempt must not trigger recovery while failover selects its successor"
    );
}

#[test]
fn watchdog_recovery_has_distinct_telemetry_cause() {
    assert_eq!(
        watchdog_recovery_cause().as_str(),
        "stuck_watchdog",
        "watchdog-triggered recovery must not be attributed to a request upstream stall"
    );
}

#[test]
fn watchdog_non_chat_requires_complete_result_bearing_json() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let cases = [
        (
            WatchdogProgressUnit::Embedding,
            br#"{"data":[{"embedding":[0.1,0.2]}]}"#.as_slice(),
        ),
        (
            WatchdogProgressUnit::Reranker,
            br#"{"results":[{"index":0,"relevance_score":0.9}]}"#.as_slice(),
        ),
    ];

    for (index, (unit, result)) in cases.into_iter().enumerate() {
        let profile = format!("non-chat-invalid-{index}");
        for invalid in [
            b" ".as_slice(),
            b"{".as_slice(),
            b"}".as_slice(),
            b"data: {\n\n".as_slice(),
        ] {
            let request = tracker.watch_request(&profile, unit, WATCHDOG_WINDOW);
            assert!(
                !request.record_emitted_chunk(invalid),
                "whitespace, JSON punctuation, and incomplete JSON are not result progress"
            );
            assert_eq!(tracker.sample_count(&profile), 0);
        }

        let request = tracker.watch_request(&profile, unit, WATCHDOG_WINDOW);
        assert!(
            request.record_emitted_chunk(result),
            "a complete response with the endpoint's result field is progress"
        );
        assert_eq!(tracker.sample_count(&profile), 1);
    }
}

#[test]
fn watchdog_chat_routing_and_tool_calls_fail_closed() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.watch_request(
        "malformed-tool-call",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );

    assert!(
        !request.record_emitted_chunk(
            br#"data: {"choices":[{"delta":{"tool_calls":[false]}}]}

"#,
        ),
        "unrecognized tool-call primitives must not be progress"
    );
    assert_eq!(tracker.sample_count("malformed-tool-call"), 0);

    let score_substring_path: Uri = "/v1/models/scorecard".parse().expect("URI should parse");
    assert_eq!(
        watchdog_progress_unit(&score_substring_path),
        WatchdogProgressUnit::Chat,
        "only registered endpoint paths may select non-chat progress parsing"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_does_not_restart_for_tool_call_only_sse() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("tool-call-only");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = touch_recovery_watchdog_config(&marker, 1);
    let proxy = spawn_watchdog_proxy(&fake.base_url, &config).await;
    let request = proxy.state.stuck_watchdog_tokens.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );
    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);

    sleep(Duration::from_millis(600)).await;
    request.record_emitted_chunk(&chat_delta_sse(&serde_json::json!({
        "tool_calls": [{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]
    })));
    sleep(Duration::from_millis(650)).await;
    let restarted = marker.exists();

    drop(request);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        !restarted,
        "tool-call-only model output is healthy and must not trigger recovery"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_records_progress_while_rewriting_a_heterogeneous_reranker_body() {
    let upstream = SlowHeterogeneousRerankerUpstream::spawn().await;
    let test_root = create_watchdog_test_root("heterogeneous-reranker-progress");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = format!(
        "{}\n\n[[profile]]\nmodel = \"same-model\"\nrequest_timeout_ms = 3000\n\n[[profile.upstream]]\nbase_url = \"{}\"\npriority = \"primary\"\nprotocol = \"deepinfra_qwen3_rerank\"\nmodel = \"Qwen/Qwen3-Reranker-8B\"\nmodel_revision = \"5fa94080caafeaa45a15d11f969d7978e087a3db\"\napi_key_env = \"PATH\"\n",
        recovery_watchdog_config(&marker, 1, 1, 1),
        upstream.base_url,
    );
    let proxy = spawn_watchdog_proxy(&upstream.base_url, &config).await;
    let peer = proxy.state.stuck_watchdog_tokens.begin_request(
        "same-model",
        WatchdogProgressUnit::Reranker,
        WATCHDOG_WINDOW,
    );
    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);
    let client = proxy.client.clone();
    let base_url = proxy.base_url.clone();
    let request = tokio::spawn(async move {
        client
            .post(format!("{base_url}/v1/rerank"))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"same-model","query":"rank this","documents":["document"]}"#)
            .send()
            .await
            .expect("heterogeneous reranker request should complete")
    });

    sleep(Duration::from_millis(1_300)).await;
    let restarted = marker.exists();
    let response = request
        .await
        .expect("heterogeneous reranker request task should join");
    assert_eq!(response.status(), StatusCode::OK);
    let _response_body = response
        .text()
        .await
        .expect("rewritten reranker response body should drain");
    drop(peer);
    stop_watchdog(&proxy, watchdog).await;

    assert!(
        !restarted,
        "a valid result read during heterogeneous reranker rewriting must count as watchdog progress"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_restarts_when_backpressure_stops_upstream_body_progress() {
    let upstream = BackpressureUpstream::spawn().await;
    let test_root = create_watchdog_test_root("downstream-backpressure");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = touch_recovery_watchdog_config(&marker, 1);
    let proxy = spawn_watchdog_proxy(&upstream.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("streaming request should receive upstream headers");
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_upstream_backpressure(&upstream.chunks_pulled).await;

    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);
    sleep(Duration::from_millis(2_200)).await;
    let restarted = marker.exists();

    drop(response);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        restarted,
        "an active response without recent upstream body progress must trigger watchdog recovery"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_restarts_after_headers_without_body_progress() {
    let upstream = PendingSseUpstream::spawn().await;
    let test_root = create_watchdog_test_root("headers-without-body");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = touch_recovery_watchdog_config(&marker, 1);
    let proxy = spawn_watchdog_proxy(&upstream.base_url, &config).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("pending SSE request should receive upstream headers");
    assert_eq!(response.status(), StatusCode::OK);

    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);
    let restarted = wait_for_path(&marker, Duration::from_millis(3_500)).await;

    drop(response);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        restarted,
        "headers without a later upstream SSE body delta must trigger watchdog recovery"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_rolls_attempt_age_to_the_remaining_attempt() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("attempt-rollover");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = touch_recovery_watchdog_config(&marker, 1);
    let proxy = spawn_watchdog_proxy(&fake.base_url, &config).await;

    let older = proxy.state.stuck_watchdog_tokens.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );
    sleep(Duration::from_millis(1_100)).await;
    let newer = proxy.state.stuck_watchdog_tokens.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );
    drop(older);

    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);
    sleep(Duration::from_millis(300)).await;
    let restarted = marker.exists();

    drop(newer);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        !restarted,
        "completing the older overlap must give the newer attempt its own detection window"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_never_waits_after_recovery_starts_without_a_queue_permit() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("restart-queue-start-race");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let config = touch_recovery_watchdog_config(&marker, 1);
    let proxy = spawn_watchdog_proxy(&fake.base_url, &config).await;
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("watchdog config should snapshot")
        .default_upstream_profile();
    let coordinator = proxy.state.local_recovery.coordinator_for(&profile.name);
    let state_guard = coordinator.state.lock().await;
    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);

    let (waiter_started_tx, waiter_started_rx) = oneshot::channel();
    let waiting_state = proxy.state.clone();
    let waiting_profile = profile.clone();
    let mut waiting = tokio::spawn(async move {
        let _started = waiter_started_tx.send(());
        let mut metadata = BTreeMap::new();
        wait_for_profile_restart_queue(&waiting_state, &waiting_profile, &mut metadata).await
    });
    waiter_started_rx
        .await
        .expect("restart queue waiter should start");
    tokio::task::yield_now().await;

    let (starter_queued_tx, starter_queued_rx) = oneshot::channel();
    let starter_coordinator = Arc::clone(&coordinator);
    let starter = tokio::spawn(async move {
        let _queued = starter_queued_tx.send(());
        let mut recovery = starter_coordinator.state.lock().await;
        let now = Instant::now();
        recovery.running = true;
        recovery.recovery_started = Some(now);
        recovery.recovery_deadline = Some(now + Duration::from_secs(1));
    });
    starter_queued_rx
        .await
        .expect("recovery starter should queue behind the waiter");
    drop(state_guard);
    starter.await.expect("recovery starter should join");

    let returned_before_recovery_finished = timeout(Duration::from_millis(100), &mut waiting)
        .await
        .is_ok();
    finish_upstream_stall_recovery(
        &coordinator,
        BTreeMap::from([(
            String::from("local_recovery_status"),
            String::from("succeeded"),
        )]),
    )
    .await;
    if !waiting.is_finished() {
        let result = waiting.await.expect("restart queue waiter should join");
        assert!(result.is_ok());
    }
    stop_watchdog(&proxy, watchdog).await;

    assert!(
        returned_before_recovery_finished,
        "a request that observed no recovery while registering must not wait for a later episode without a queue permit"
    );
    assert_eq!(proxy.state.generation_requests.snapshot_counts().queued, 0);
}

#[tokio::test]
async fn restart_queue_recovery_success_metadata_persists_on_the_completed_request() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("restart-queue-success-metadata");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    let proxy =
        spawn_watchdog_proxy(&fake.base_url, &recovery_watchdog_config(&marker, 1, 1, 1)).await;
    let coordinator = proxy.state.local_recovery.coordinator_for("default");
    {
        let mut recovery = coordinator.state.lock().await;
        let now = Instant::now();
        recovery.running = true;
        recovery.recovery_started = Some(now);
        recovery.recovery_deadline = Some(now + Duration::from_secs(2));
    }

    let client = proxy.client.clone();
    let base_url = proxy.base_url.clone();
    let request = tokio::spawn(async move {
        client
            .post(format!(
                "{base_url}/v1/chat/completions?test=restart-queue-success-metadata"
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"queue me"}]}"#)
            .send()
            .await
            .expect("queued proxy request should complete")
    });
    sleep(Duration::from_millis(40)).await;
    finish_upstream_stall_recovery(
        &coordinator,
        BTreeMap::from([(
            String::from("local_recovery_status"),
            String::from("succeeded"),
        )]),
    )
    .await;

    let response = request.await.expect("queued request task should join");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response
        .text()
        .await
        .expect("response body should be readable");
    proxy.state.flush_persistence().await;

    let request_metadata_json: String = Connection::open(&proxy.sqlite_path)
        .expect("sqlite should open")
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("completed request row should exist");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_metadata_json).expect("request metadata should be json");
    assert_eq!(
        request_metadata["restart_queue_outcome"],
        "released_after_recovery"
    );
    assert!(
        request_metadata["restart_queue_wait_ms"]
            .as_str()
            .is_some_and(|elapsed| elapsed.parse::<u64>().is_ok_and(|elapsed| elapsed > 0)),
        "successful restart-queue metadata must include elapsed wait time"
    );
}

#[tokio::test]
async fn watchdog_lifecycle_applies_reloaded_interval_without_old_interval_delay() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("interval-reload");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let marker = test_root.join("restart.marker");
    // Let the initial cadence run twice without restarting, then make the
    // request eligible before applying the one-second cadence. This proves the
    // reload replaces an already-scheduled, still-distant due time.
    let initial_config = recovery_watchdog_config(&marker, 3, 3, 0);
    let proxy = spawn_watchdog_proxy(&fake.base_url, &initial_config).await;
    let request = proxy.state.stuck_watchdog_tokens.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );
    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);

    sleep(Duration::from_millis(3_200)).await;
    let reloaded_config = recovery_watchdog_config(&marker, 1, 3, 1);
    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &reloaded_config,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("watchdog interval reload should succeed");
    assert!(outcome.applied);
    let restarted = wait_for_path(&marker, Duration::from_millis(3_500)).await;

    drop(request);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        restarted,
        "the reloaded one-second interval must replace the previously scheduled three-second due time"
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn watchdog_lifecycle_restart_timeout_bounds_and_cancels_the_real_episode() {
    let fake = FakeUpstream::spawn().await;
    let test_root = create_watchdog_test_root("episode-timeout");
    let _cleanup = TestDirectoryCleanup::new(&test_root);
    let script_path = test_root.join("slow-restart.sh");
    let pid_path = test_root.join("restart.pid");
    let ready_path = test_root.join("restart.ready");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$$\" > \"$1\"\n: > \"$2\"\nexec sleep 30\n",
    )
    .expect("fake restart script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("fake restart script should be executable");
    let config = slow_recovery_watchdog_config(&script_path, &pid_path, &ready_path);
    let proxy = spawn_watchdog_proxy(&fake.base_url, &config).await;
    let request = proxy.state.stuck_watchdog_tokens.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        WATCHDOG_WINDOW,
    );
    sleep(Duration::from_millis(1_100)).await;
    let watchdog = spawn_stuck_engine_watchdog(&proxy.state);

    let process = read_pid_file_after_ready(&pid_path, &ready_path).await;
    let stopped = wait_for_process_stop(process, Duration::from_millis(2_500)).await;
    if !stopped {
        kill_process_if_running(process);
    }
    let timeout_recorded = wait_for_watchdog_timeout_metric(&proxy, Duration::from_secs(2)).await;
    let restart_recorded = wait_for_watchdog_restart_metric(&proxy, Duration::from_secs(2)).await;

    drop(request);
    stop_watchdog(&proxy, watchdog).await;
    assert!(
        stopped,
        "restart_timeout_secs must cancel the spawned restart process, not only its waiter"
    );
    assert!(
        timeout_recorded,
        "a bounded watchdog episode must finish and record a timeout result"
    );
    assert!(
        restart_recorded,
        "a restart command that began before episode timeout must increment the restart metric"
    );
}

fn chat_delta_sse(delta: &serde_json::Value) -> Bytes {
    sse_json(&serde_json::json!({
        "id": "chatcmpl-watchdog",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": null
        }]
    }))
}

fn create_watchdog_test_root(name: &str) -> PathBuf {
    let root = unique_test_dir(name);
    fs::create_dir_all(&root).expect("watchdog test root should be created");
    root
}

async fn spawn_watchdog_proxy(upstream_base_url: &str, config: &str) -> ProxyFixture {
    ProxyFixture::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
        upstream_base_url,
        observability_enabled: true,
        max_in_flight_requests: AppConfig::default().server.max_in_flight_requests,
        server_config: "",
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: config,
    })
    .await
}

fn touch_recovery_watchdog_config(marker: &Path, check_interval_secs: u64) -> String {
    recovery_watchdog_config(marker, check_interval_secs, 1, 1)
}

fn recovery_watchdog_config(
    marker: &Path,
    check_interval_secs: u64,
    detection_window_secs: u64,
    min_output_progress_units_in_window: u64,
) -> String {
    format!(
        r#"
[retry]
enabled = false

[upstream.stuck_watchdog]
enabled = true
detection_window_secs = {detection_window_secs}
min_output_progress_units_in_window = {min_output_progress_units_in_window}
check_interval_secs = {check_interval_secs}

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{marker}"]
restart_timeout_ms = 1000
readiness_request_timeout_ms = 100
readiness_deadline_ms = 500
readiness_interval_ms = 25
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 10

[upstream.restart_queue]
enabled = true
queue_deadline_secs = 2
restart_timeout_secs = 2
"#,
        marker = marker.display(),
    )
}

#[cfg(target_os = "linux")]
fn slow_recovery_watchdog_config(script: &Path, pid: &Path, ready: &Path) -> String {
    format!(
        r#"
[retry]
enabled = false

[upstream.stuck_watchdog]
enabled = true
detection_window_secs = 1
min_output_progress_units_in_window = 1
check_interval_secs = 1

[upstream.local_recovery]
enabled = true
restart_command = ["{script}", "{pid}", "{ready}"]
restart_timeout_ms = 30000
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 30000
readiness_interval_ms = 25
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 10

[upstream.restart_queue]
enabled = true
queue_deadline_secs = 5
restart_timeout_secs = 1
"#,
        script = script.display(),
        pid = pid.display(),
        ready = ready.display(),
    )
}

async fn stop_watchdog(proxy: &ProxyFixture, watchdog: tokio::task::JoinHandle<()>) {
    proxy.state.begin_shutdown();
    timeout(WATCHDOG_TASK_TIMEOUT, watchdog)
        .await
        .expect("watchdog task should stop after shutdown")
        .expect("watchdog task should join cleanly");
}

async fn wait_for_path(path: &Path, wait: Duration) -> bool {
    timeout(wait, async {
        while !path.exists() {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .is_ok()
}

#[cfg(target_os = "linux")]
async fn wait_for_process_stop(identity: LinuxProcessIdentity, wait: Duration) -> bool {
    timeout(wait, async {
        while identity.is_running() {
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .is_ok()
}

#[cfg(target_os = "linux")]
async fn wait_for_watchdog_timeout_metric(proxy: &ProxyFixture, wait: Duration) -> bool {
    let coordinator = proxy.state.local_recovery.coordinator_for("default");
    timeout(wait, async {
        while coordinator
            .watchdog_recovery_timeouts
            .load(Ordering::Relaxed)
            == 0
        {
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .is_ok()
}

#[cfg(target_os = "linux")]
async fn wait_for_watchdog_restart_metric(proxy: &ProxyFixture, wait: Duration) -> bool {
    let coordinator = proxy.state.local_recovery.coordinator_for("default");
    timeout(wait, async {
        while coordinator.watchdog_restarts.load(Ordering::Relaxed) == 0 {
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .is_ok()
}

struct SlowHeterogeneousRerankerUpstream {
    base_url: String,
    server: tokio::task::JoinHandle<()>,
}

impl SlowHeterogeneousRerankerUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("slow heterogeneous reranker upstream should bind");
        let address = listener
            .local_addr()
            .expect("slow heterogeneous reranker upstream address should resolve");
        let app = Router::new().fallback(slow_heterogeneous_reranker_handler);
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("slow heterogeneous reranker upstream should serve");
        });
        Self {
            base_url: format!("http://{address}"),
            server,
        }
    }
}

impl Drop for SlowHeterogeneousRerankerUpstream {
    fn drop(&mut self) {
        self.server.abort();
    }
}

struct BackpressureUpstream {
    base_url: String,
    chunks_pulled: Arc<AtomicU64>,
    server: tokio::task::JoinHandle<()>,
}

impl BackpressureUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("backpressure upstream should bind");
        let address = listener
            .local_addr()
            .expect("backpressure upstream address should resolve");
        let chunks_pulled = Arc::new(AtomicU64::new(0));
        let app = Router::new()
            .route("/v1/chat/completions", post(backpressure_stream_handler))
            .with_state(Arc::clone(&chunks_pulled));
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("backpressure upstream should serve");
        });
        Self {
            base_url: format!("http://{address}"),
            chunks_pulled,
            server,
        }
    }
}

impl Drop for BackpressureUpstream {
    fn drop(&mut self) {
        self.server.abort();
    }
}

struct PendingSseUpstream {
    base_url: String,
    server: tokio::task::JoinHandle<()>,
}

impl PendingSseUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("pending SSE upstream should bind");
        let address = listener
            .local_addr()
            .expect("pending SSE upstream address should resolve");
        let app = Router::new().route("/v1/chat/completions", post(pending_sse_handler));
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("pending SSE upstream should serve");
        });
        Self {
            base_url: format!("http://{address}"),
            server,
        }
    }
}

impl Drop for PendingSseUpstream {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn slow_heterogeneous_reranker_handler() -> Response<Body> {
    let body = Body::from_stream(stream::unfold(0_u8, |step| async move {
        match step {
            0 => Some((Ok::<Bytes, Infallible>(Bytes::from_static(b"{")), 1)),
            1 => {
                sleep(Duration::from_millis(600)).await;
                Some((
                    Ok::<Bytes, Infallible>(Bytes::from_static(
                        br#""scores":[0.9],"input_tokens":1}"#,
                    )),
                    2,
                ))
            }
            _ => None,
        }
    }));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/vnd.vllm.score+json"),
    );
    response
}

async fn backpressure_stream_handler(
    State(chunks_pulled): State<Arc<AtomicU64>>,
) -> Response<Body> {
    let content = "x".repeat(64 * 1024);
    let frame = chat_delta_sse(&serde_json::json!({"content": content}));
    let body = Body::from_stream(stream::unfold(
        (chunks_pulled, frame),
        |(chunks_pulled, frame)| async move {
            chunks_pulled.fetch_add(1, Ordering::Relaxed);
            Some((
                Ok::<Bytes, Infallible>(frame.clone()),
                (chunks_pulled, frame),
            ))
        },
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

async fn pending_sse_handler() -> Response<Body> {
    let mut response = Response::new(Body::from_stream(stream::pending::<
        Result<Bytes, Infallible>,
    >()));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

async fn wait_for_upstream_backpressure(chunks_pulled: &AtomicU64) {
    timeout(Duration::from_secs(3), async {
        while chunks_pulled.load(Ordering::Relaxed) < 8 {
            sleep(Duration::from_millis(10)).await;
        }
        let mut stable_checks = 0_u8;
        let mut previous = chunks_pulled.load(Ordering::Relaxed);
        while stable_checks < 5 {
            sleep(Duration::from_millis(25)).await;
            let current = chunks_pulled.load(Ordering::Relaxed);
            if current == previous {
                stable_checks = stable_checks.saturating_add(1);
            } else {
                stable_checks = 0;
                previous = current;
            }
        }
    })
    .await
    .expect("unconsumed downstream response should apply bounded backpressure");
}
