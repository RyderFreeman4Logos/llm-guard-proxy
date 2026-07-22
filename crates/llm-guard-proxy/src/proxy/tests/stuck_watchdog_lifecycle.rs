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
