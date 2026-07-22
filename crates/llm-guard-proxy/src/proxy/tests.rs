use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[cfg(unix)]
use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};

use crate::config_reload::ConfigManager;
use axum::http::header::{AUTHORIZATION, CONNECTION, LOCATION};
use futures_util::{Stream, StreamExt, stream};
use rusqlite::{Connection, params};
use tokio::{
    io::AsyncReadExt,
    net::TcpListener,
    sync::{mpsc, oneshot},
    time::{sleep, timeout},
};

use super::*;

#[path = "tests/shielded_endpoint_rendering.rs"]
mod shielded_endpoint_rendering;
#[cfg(unix)]
#[path = "tests/stuck_watchdog_lifecycle.rs"]
mod stuck_watchdog_lifecycle;

const TEST_MAX_BYTES: u64 = 1_000_000;
const TEST_PRUNE_TO_BYTES: u64 = 800_000;
const TEST_MAX_RECORDS: u64 = 100;
const STREAM_DELAY: Duration = Duration::from_millis(800);
const STREAM_HEADER_TIMEOUT: Duration = Duration::from_millis(500);
const STREAM_FIRST_CHUNK_TIMEOUT: Duration = Duration::from_millis(250);
const STREAM_SECOND_CHUNK_GUARD: Duration = Duration::from_millis(150);
const STREAM_COMPLETION_TIMEOUT: Duration = Duration::from_secs(2);
const SHIELDED_SLOW_DELAY: Duration = Duration::from_millis(2_500);
const SHIELDED_HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(1_500);
const SHIELDED_RELOAD_GUARD: Duration = Duration::from_millis(1_500);
const SHIELDED_RELOAD_TIMEOUT: Duration = Duration::from_millis(2_500);
const SSE_FIRST_CHUNK: &[u8] = b"data: first\n\n";
const SSE_SECOND_CHUNK: &[u8] = b"data: second\n\n";
const LONG_JSON_FIRST_CHUNK: &[u8] = br#"{"object":"list","data":["#;
const LONG_JSON_SECOND_CHUNK: &[u8] = br"]}";
const MODEL_METADATA_BODY: &str = r#"{"object":"list","data":[{"id":"aeon-ultimate","object":"model","max_model_len":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_CHUNKED_FIRST: &[u8] =
    br#"{"object":"list","data":[{"id":"chunked-model","object":"model","#;
const MODEL_METADATA_CHUNKED_SECOND: &[u8] =
    br#""max_model_len":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_NO_CONTEXT_BODY: &str = r#"{"object":"list","data":[{"id":"fallback-model","object":"model","owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_CONTEXT_LENGTH_BODY: &str = r#"{"object":"list","data":[{"id":"context-length-model","object":"model","context_length":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MODEL_METADATA_MAX_CONTEXT_LENGTH_BODY: &str = r#"{"object":"list","data":[{"id":"max-context-length-model","object":"model","max_context_length":256000,"owned_by":"vllm","extra":"keep"}]}"#;
const MULTI_LISTENER_MODEL_METADATA_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"embedding-model","object":"model","owned_by":"vllm"},{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_CHAT_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"embedding-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"embedding-model","object":"model","owned_by":"vllm","first":"embedding"},{"id":"embedding-model","object":"model","owned_by":"vllm","first":"duplicate"}]}"#;
const DISTINCT_UPSTREAM_RERANK_MODELS_BODY: &str = r#"{"object":"list","data":[{"id":"chat-model","object":"model","owned_by":"vllm"},{"id":"rerank-model","object":"model","owned_by":"vllm"}]}"#;
const DISTINCT_UPSTREAM_SLOW_MODELS_BODY: &str =
    r#"{"object":"list","data":[{"id":"slow-model","object":"model","owned_by":"vllm"}]}"#;
const REPEATED_INPUT_LOOP_LINE: &str = "legitimate repeated input line for issue ten";
const LARGE_MODEL_METADATA_EXTRA_BYTES: usize = 1024 * 1024;
static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn parse_token_usage_reads_non_streaming_completion_usage() {
    let usage = parse_token_usage(
        br#"{
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 5}
            }
        }"#,
        b"",
    );

    assert_eq!(
        usage,
        TokenUsage {
            input_tokens: Some(11),
            output_tokens: Some(7),
            cached_input_tokens: Some(3),
            reasoning_tokens: Some(5),
        }
    );
}

#[test]
fn parse_token_usage_reads_last_usage_sse_data_line() {
    let usage = parse_token_usage(
        b"not JSON",
        br#"data: {"choices":[{"delta":{"content":"answer"}}]}

data: {"usage":{"prompt_tokens":13,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":4}}}

data: [DONE]
"#,
    );

    assert_eq!(
        usage,
        TokenUsage {
            input_tokens: Some(13),
            output_tokens: Some(8),
            cached_input_tokens: Some(2),
            reasoning_tokens: Some(4),
        }
    );
}

#[test]
fn stuck_watchdog_counts_non_stream_and_sse_output_progress_units_in_trailing_window() {
    let tracker = StuckWatchdogTokenTracker::default();
    let tracker = Arc::new(tracker);
    let request = tracker.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );
    let mut windows = tracker
        .windows
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let attempt = windows
        .get_mut("default")
        .and_then(|window| window.attempts.get_mut(&request.attempt_id))
        .expect("active test request must be tracked");
    attempt.started_at = Instant::now()
        .checked_sub(Duration::from_secs(2))
        .expect("test Instant supports a two-second adjustment");
    drop(windows);
    tracker.record_response(
        "default",
        Duration::from_secs(1),
        br#"{"usage":{"completion_tokens":3}}"#,
        b"",
    );
    tracker.record_response(
        "default",
        Duration::from_secs(1),
        b"",
        br#"data: {"usage":{"completion_tokens":2}}

data: [DONE]
"#,
    );

    assert!(!tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 2));
    assert!(tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 3));
}

#[test]
fn stuck_watchdog_waits_for_a_complete_detection_window_before_declaring_a_new_request_stuck() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    assert!(!tracker.has_active_requests("default"));

    let request = tracker.begin_request(
        "default",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );
    assert!(tracker.has_active_requests("default"));
    assert!(
        !tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 1),
        "a request that just began has not yet had a full detection window to produce output"
    );

    request.record_response(br#"{"usage":{"completion_tokens":1}}"#, b"");
    assert!(!tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 1));

    drop(request);
    assert!(!tracker.has_active_requests("default"));
    assert!(
        !tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 2),
        "a completed-only window must not be considered an active stuck request"
    );
}

#[test]
fn shielded_watchdog_counts_a_stream_delta_and_terminal_usage_once() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.begin_request(
        "shielded",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );

    request.record_upstream_emitted_chunk(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
    );
    request.record_response(
        b"",
        b"data: {\"usage\":{\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
    );

    assert_eq!(
        tracker.sample_count("shielded"),
        1,
        "a shielded upstream delta must not be counted again from terminal usage"
    );
}

#[test]
fn watchdog_metrics_count_only_started_commands_and_timeout_kills() {
    let coordinator = UpstreamStallRecoveryCoordinator::default();
    let skipped = BTreeMap::from([(
        String::from("local_recovery_status"),
        String::from("skipped_cooldown"),
    )]);
    record_watchdog_recovery_result(&coordinator, "default", &skipped);
    assert_eq!(coordinator.watchdog_restarts.load(Ordering::Relaxed), 0);
    assert_eq!(
        coordinator
            .watchdog_recovery_timeouts
            .load(Ordering::Relaxed),
        0
    );

    let timeout_killed = BTreeMap::from([
        (
            String::from("local_recovery_restart_ran"),
            String::from("true"),
        ),
        (
            String::from("local_recovery_status"),
            String::from("timeout_killed"),
        ),
    ]);
    record_watchdog_recovery_result(&coordinator, "default", &timeout_killed);

    assert_eq!(
        coordinator.watchdog_restarts.load(Ordering::Relaxed),
        1,
        "only a recovery command that actually started is a watchdog restart"
    );
    assert_eq!(
        coordinator
            .watchdog_recovery_timeouts
            .load(Ordering::Relaxed),
        1,
        "a killed restart command is a timed-out watchdog recovery"
    );
}

#[test]
fn restart_queue_wait_deadline_caps_recovery_episode_timeout() {
    let queue = RestartQueueConfig {
        enabled: true,
        queue_deadline_secs: 60,
        restart_timeout_secs: 30,
    };

    assert_eq!(restart_queue_wait_deadline(&queue), Duration::from_secs(30));
}

#[tokio::test]
async fn restart_queue_waits_for_readiness_result_and_times_out_without_it() {
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    assert_eq!(
        wait_for_restart_queue(&coordinator, Duration::from_millis(5)).await,
        RestartQueueWaitResult::NotRecovering
    );

    coordinator.state.lock().await.running = true;
    let waiting = {
        let coordinator = Arc::clone(&coordinator);
        tokio::spawn(async move {
            wait_for_restart_queue(&coordinator, Duration::from_millis(200)).await
        })
    };

    sleep(Duration::from_millis(20)).await;
    assert!(
        !waiting.is_finished(),
        "request must remain queued during recovery"
    );
    finish_upstream_stall_recovery(
        &coordinator,
        BTreeMap::from([(
            String::from("local_recovery_status"),
            String::from("succeeded"),
        )]),
    )
    .await;
    assert_eq!(
        waiting.await.expect("queued request task should join"),
        RestartQueueWaitResult::Ready
    );

    coordinator.state.lock().await.running = true;
    assert_eq!(
        wait_for_restart_queue(&coordinator, Duration::from_millis(5)).await,
        RestartQueueWaitResult::TimedOut
    );
}

#[test]
fn watchdog_records_streaming_and_non_chat_progress_when_emitted() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let chat = tracker.begin_request("chat", WatchdogProgressUnit::Chat, Duration::from_secs(1));
    let embedding = tracker.begin_request(
        "embedding",
        WatchdogProgressUnit::Embedding,
        Duration::from_secs(1),
    );
    let reranker = tracker.begin_request(
        "reranker",
        WatchdogProgressUnit::Reranker,
        Duration::from_secs(1),
    );

    chat.record_emitted_chunk(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"healthy stream\"}}]}\n\n",
    );
    embedding.record_emitted_chunk(br#"{"data":[{"embedding":[0.1,0.2]}]}"#);
    reranker.record_emitted_chunk(br#"{"results":[{"index":0,"relevance_score":0.9}]}"#);

    assert!(!tracker.has_too_few_output_progress_units("chat", Duration::from_secs(1), 1));
    assert!(!tracker.has_too_few_output_progress_units("embedding", Duration::from_secs(1), 1));
    assert!(!tracker.has_too_few_output_progress_units("reranker", Duration::from_secs(1), 1));
}

#[test]
fn watchdog_ignores_sse_heartbeats_and_control_frames_but_records_content() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let chat = tracker.begin_request("chat", WatchdogProgressUnit::Chat, Duration::from_secs(1));

    chat.record_emitted_chunk(b": llm-guard-proxy heartbeat\n\n");
    chat.record_emitted_chunk(b"event: ping\ndata: {}\n\n");
    chat.record_emitted_chunk(b"data: [DONE]\n\n");
    assert_eq!(
        tracker.sample_count("chat"),
        0,
        "transport heartbeats and terminal control frames are not model output"
    );

    chat.record_emitted_chunk(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"healthy stream\"}}]}\n\n",
    );
    assert_eq!(tracker.sample_count("chat"), 1);
}

#[test]
fn watchdog_token_samples_are_capped_and_maintained_without_active_requests() {
    let tracker = StuckWatchdogTokenTracker::default();
    for _ in 0..=STUCK_WATCHDOG_PROGRESS_SAMPLE_CAP {
        tracker.record_progress("default", Duration::from_secs(1), 1);
    }
    assert_eq!(
        tracker.sample_count("default"),
        STUCK_WATCHDOG_PROGRESS_SAMPLE_CAP,
        "insertion must cap remotely supplied output samples"
    );

    tracker.prune_profile("default", Duration::ZERO);
    assert_eq!(
        tracker.sample_count("default"),
        0,
        "maintenance must prune samples even after every request has completed"
    );
}

#[test]
fn watchdog_does_not_prune_shared_samples_using_request_snapshots() {
    let tracker = StuckWatchdogTokenTracker::default();

    tracker.record_progress("default", Duration::ZERO, 1);
    tracker.record_progress("default", Duration::ZERO, 1);

    assert_eq!(
        tracker.sample_count("default"),
        2,
        "per-request snapshot windows must not prune shared samples"
    );
    tracker.prune_profile("default", Duration::ZERO);
    assert_eq!(
        tracker.sample_count("default"),
        0,
        "the current watchdog window owns shared-sample retention"
    );
}

#[test]
fn watchdog_physical_attempt_lease_resets_age_and_ends_before_persistence() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.watch_request(
        "default",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );
    let first_attempt = request.begin_attempt();
    {
        let mut windows = tracker
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = windows
            .get_mut("default")
            .and_then(|window| window.attempts.get_mut(&first_attempt.attempt_id))
            .expect("first physical attempt must be active");
        first.started_at = Instant::now()
            .checked_sub(Duration::from_secs(2))
            .expect("test Instant supports a two-second adjustment");
    }

    first_attempt.end();
    let retry_attempt = request.begin_attempt();
    let persistence_runtime = request.clone();

    assert!(
        !tracker.has_too_few_output_progress_units("default", Duration::from_secs(1), 1),
        "a retry is a new physical attempt with a full output window"
    );
    retry_attempt.end();
    assert!(
        !tracker.has_active_requests("default"),
        "ending the network attempt must not wait for persistence runtime clones"
    );
    drop(persistence_runtime);
}

#[test]
fn watchdog_prune_and_insert_are_atomic_against_detection() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.watch_request(
        "default",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );
    let attempt = request.begin_attempt();
    {
        let mut windows = tracker
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let window = windows
            .get_mut("default")
            .expect("attempt window must exist");
        let active = window
            .attempts
            .get_mut(&attempt.attempt_id)
            .expect("attempt must be active");
        active.started_at = Instant::now()
            .checked_sub(Duration::from_secs(2))
            .expect("test Instant supports a two-second adjustment");
        window.samples.push_back((
            Instant::now()
                .checked_sub(Duration::from_secs(2))
                .expect("test Instant supports a two-second adjustment"),
            1,
        ));
    }

    let producer_tracker = Arc::clone(&tracker);
    let entered_pruned_state = Arc::new(Barrier::new(2));
    let release_producer = Arc::new(Barrier::new(2));
    let producer_entered = Arc::clone(&entered_pruned_state);
    let producer_release = Arc::clone(&release_producer);
    let producer = std::thread::spawn(move || {
        producer_tracker.record_progress_with_hook("default", Duration::from_secs(1), 1, || {
            producer_entered.wait();
            producer_release.wait();
        });
    });
    entered_pruned_state.wait();

    let detector_tracker = Arc::clone(&tracker);
    let (detector_tx, detector_rx) = std::sync::mpsc::channel();
    let detector = std::thread::spawn(move || {
        let _ignored = detector_tx.send(detector_tracker.has_too_few_output_progress_units(
            "default",
            Duration::from_secs(1),
            1,
        ));
    });
    assert!(
        detector_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "the detector must remain blocked while producer prunes and inserts under one lock"
    );

    release_producer.wait();
    producer.join().expect("producer must join");
    assert!(
        !detector_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("detector must observe the completed transition"),
        "the detector must see the fresh progress sample, never a pruned empty window"
    );
    detector.join().expect("detector must join");
    attempt.end();
}

#[test]
fn watchdog_progress_units_count_one_multi_token_chat_delta() {
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request = tracker.watch_request(
        "default",
        WatchdogProgressUnit::Chat,
        Duration::from_secs(1),
    );
    let attempt = request.begin_attempt();
    attempt.record_emitted_chunk(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"many distinct model tokens\"}}]}\n\n",
    );

    assert_eq!(tracker.sample_count("default"), 1);
    attempt.end();
}

#[tokio::test]
async fn restart_queue_uses_episode_deadline_not_a_fresh_wait_per_request() {
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    {
        let mut recovery = coordinator.state.lock().await;
        recovery.running = true;
        recovery.recovery_started = Some(Instant::now());
        recovery.recovery_deadline = Some(Instant::now());
    }

    assert_eq!(
        timeout(
            Duration::from_millis(20),
            wait_for_restart_queue(&coordinator, Duration::from_secs(1)),
        )
        .await
        .expect("an expired episode deadline must not start a fresh per-request wait"),
        RestartQueueWaitResult::TimedOut,
        "an already-running recovery has one deadline shared by every queued request"
    );
}

#[tokio::test]
async fn restart_queue_waiters_consume_queue_capacity_not_in_flight_capacity() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\n",
    )
    .await;
    let mut profile = AppConfig::default().default_upstream_profile();
    profile.restart_queue.enabled = true;
    let coordinator = proxy.state.local_recovery.coordinator_for(&profile.name);
    coordinator.state.lock().await.running = true;

    let permit = proxy
        .state
        .acquire_restart_queue_permit(&profile, &coordinator)
        .await
        .expect("restart queue admission should succeed")
        .expect("active recovery should acquire a queue permit");
    let counts = proxy.state.generation_requests.snapshot_counts();
    assert_eq!(counts.active, 0);
    assert_eq!(counts.queued, 1);

    drop(permit);
    assert_eq!(proxy.state.generation_requests.snapshot_counts().queued, 0);
}

#[tokio::test]
async fn real_default_profile_requests_wait_in_the_restart_queue_until_recovery_finishes() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
        upstream_base_url: &fake.base_url,
        observability_enabled: true,
        max_in_flight_requests: 1,
        server_config: "",
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: r"
[upstream.restart_queue]
enabled = true
queue_deadline_secs = 1
restart_timeout_secs = 1
",
    })
    .await;
    let coordinator = proxy.state.local_recovery.coordinator_for("default");
    {
        let mut recovery = coordinator.state.lock().await;
        recovery.running = true;
        recovery.recovery_started = Some(Instant::now());
        recovery.recovery_deadline = Some(Instant::now() + Duration::from_secs(1));
    }

    let request = {
        let client = proxy.client.clone();
        let base_url = proxy.base_url.clone();
        tokio::spawn(async move {
            client
                .post(format!("{base_url}/v1/chat/completions?test=restart-queue"))
                .header(CONTENT_TYPE, "application/json")
                .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"queue me"}]}"#)
                .send()
                .await
                .expect("queued proxy request should complete")
        })
    };

    sleep(Duration::from_millis(40)).await;
    assert!(
        !request.is_finished(),
        "the actual default-profile request must wait during the recovery episode"
    );
    assert!(
        fake.recv_within(Duration::from_millis(40)).await.is_none(),
        "a queued request must not reach the upstream before recovery is ready"
    );

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
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=restart-queue"
    );
}

#[test]
fn watchdog_schedule_keeps_profile_intervals_independent() {
    let now = Instant::now();
    let mut schedule = WatchdogSchedule::default();
    let profiles = vec![
        (String::from("fast"), Duration::from_secs(30)),
        (String::from("slow"), Duration::from_secs(600)),
    ];
    assert_eq!(
        schedule.due_profiles(now, &profiles),
        vec![String::from("fast"), String::from("slow")]
    );
    assert_eq!(
        schedule.due_profiles(now + Duration::from_secs(30), &profiles),
        vec![String::from("fast")],
        "a fast profile must not make a 600-second profile due every 30 seconds"
    );
}

#[test]
fn parse_token_usage_reads_embedding_prompt_tokens() {
    let usage = parse_token_usage(
        br#"{"data":[],"usage":{"prompt_tokens":17,"total_tokens":17}}"#,
        b"",
    );

    assert_eq!(
        usage,
        TokenUsage {
            input_tokens: Some(17),
            ..TokenUsage::default()
        }
    );
}

#[test]
fn parse_token_usage_returns_default_when_usage_is_missing() {
    let usage = parse_token_usage(br#"{"id":"chatcmpl-without-usage"}"#, b"data: [DONE]\n\n");

    assert_eq!(usage, TokenUsage::default());
}

#[test]
fn parse_token_usage_returns_default_for_malformed_json() {
    let usage = parse_token_usage(b"{\"usage\":", b"data: {\"usage\":\n\n");

    assert_eq!(usage, TokenUsage::default());
}

#[tokio::test]
async fn health_reports_process_and_upstream_readiness() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        "health_upstream_probe_timeout_ms = 100\n",
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    assert_eq!(body["process"], "alive");
    assert_eq!(body["upstream"], "ready");

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/models");

    let broken = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &broken.base_url,
        true,
        "health_upstream_probe_timeout_ms = 20\n",
    )
    .await;
    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    assert_eq!(body["process"], "alive");
    assert_eq!(body["upstream"], "unavailable");
}

#[tokio::test]
async fn metrics_expose_retained_gauges_without_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live-secret&safe=ok",
            proxy.base_url
        ))
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .header("x-api-key", "sk-header-secret")
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv_next().await;

    let response = proxy
        .client
        .get(format!("{}/metrics", proxy.base_url))
        .send()
        .await
        .expect("metrics request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("metrics should be text");
    assert_metric_type(&body, "llm_guard_proxy_generation_active", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_queued", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_profile_active", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_generation_profile_queued", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_restart_queue_depth", "gauge");
    assert_metric_type(
        &body,
        "llm_guard_proxy_stuck_watchdog_detections_total",
        "counter",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_stuck_watchdog_restarts_total",
        "counter",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_stuck_watchdog_recovery_successes_total",
        "counter",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_stuck_watchdog_recovery_timeouts_total",
        "counter",
    );
    assert_metric_type(&body, "llm_guard_proxy_current_retained_requests", "gauge");
    assert_metric_type(
        &body,
        "llm_guard_proxy_current_retained_request_terminals",
        "gauge",
    );
    assert_metric_type(&body, "llm_guard_proxy_current_retained_attempts", "gauge");
    assert_metric_type(&body, "llm_guard_proxy_current_retained_retries", "gauge");
    assert_metric_type(
        &body,
        "llm_guard_proxy_current_retained_first_token_latency_ms_le",
        "gauge",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_current_retained_total_latency_ms_le",
        "gauge",
    );
    assert_metric_type(
        &body,
        "llm_guard_proxy_storage_pruning_events_total",
        "counter",
    );
    assert_legacy_retained_counter_metrics_absent(&body);
    assert_safe_operational_text("metrics", &body);
}

#[tokio::test]
async fn metrics_render_unknown_liveness_values_only_as_other() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let adversarial_values = [
        "raw prompt: private patient record",
        "raw response: confidential account balance",
        "reasoning trace: hidden model analysis",
        "x-private-header: confidential-header-value",
        "SSE",
    ];

    for (index, value) in adversarial_values.iter().enumerate() {
        let request = RequestRecord {
            request_id: RequestId::from_string(format!("req-rendered-label-{index}"))
                .expect("test request id should be valid"),
            started_at_unix_ms: 1_000 + u64::try_from(index).expect("small test index"),
            finished_at_unix_ms: Some(1_100 + u64::try_from(index).expect("small test index")),
            downstream_mode: DownstreamMode::NonStreamJson,
            upstream_mode: UpstreamMode::Streaming,
            model_id: None,
            input_fingerprint: None,
            status: RequestStatus::Succeeded,
            http_status: Some(200),
            error_reason: None,
            abort_reason: None,
            request_metadata: BTreeMap::new(),
            response_metadata: BTreeMap::from([(
                String::from("downstream_liveness_mode"),
                (*value).to_owned(),
            )]),
            raw_payloads: RawPayloads::default(),
        };
        proxy
            .store
            .record_request(&request)
            .expect("request should be recorded");
    }

    let body = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &body,
            "llm_guard_proxy_current_retained_heartbeat_modes",
            &[("mode", "other")],
        ),
        5
    );
    for raw_value in adversarial_values {
        assert!(!body.contains(raw_value));
    }
}

#[tokio::test]
async fn retained_metrics_stay_prometheus_safe_after_pruning() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for index in 0..50 {
        send_metrics_chat_request(&proxy, &mut fake, index).await;
    }
    let before = fetch_metrics(&proxy).await;
    assert_metric_type(
        &before,
        "llm_guard_proxy_current_retained_requests",
        "gauge",
    );
    assert_metric_type(
        &before,
        "llm_guard_proxy_current_retained_first_token_latency_ms_le",
        "gauge",
    );
    assert_legacy_retained_counter_metrics_absent(&before);
    assert!(
        metric_value(
            &before,
            "llm_guard_proxy_current_retained_first_token_latency_ms_observations"
        ) > 0
    );
    assert!(
        metric_value(
            &before,
            "llm_guard_proxy_current_retained_total_latency_ms_observations"
        ) > 0
    );

    let before_prune_events = metric_value(&before, "llm_guard_proxy_storage_pruning_events_total");
    let before_pruned_requests =
        metric_value(&before, "llm_guard_proxy_storage_pruned_requests_total");
    let before_pruned_attempts =
        metric_value(&before, "llm_guard_proxy_storage_pruned_attempts_total");

    for index in 50..52 {
        send_metrics_chat_request(&proxy, &mut fake, index).await;
    }
    let after = fetch_metrics(&proxy).await;
    assert_metric_type(&after, "llm_guard_proxy_current_retained_requests", "gauge");
    assert_metric_type(
        &after,
        "llm_guard_proxy_current_retained_total_latency_ms_le",
        "gauge",
    );
    assert_legacy_retained_counter_metrics_absent(&after);

    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruning_events_total") >= before_prune_events
    );
    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruned_requests_total")
            > before_pruned_requests
    );
    assert!(
        metric_value(&after, "llm_guard_proxy_storage_pruned_attempts_total")
            > before_pruned_attempts
    );
}

#[tokio::test]
async fn metrics_expose_generation_active_and_queued_gauges() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=metrics-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should reach upstream");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=metrics-active"
    );

    let (queued_request, queued_body_polled) = tracked_json_request(
        "/v1/completions?slot=metrics-queued",
        br#"{"prompt":"queued"}"#,
    );
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));
    sleep(Duration::from_millis(50)).await;
    assert!(!queued_body_polled.load(Ordering::SeqCst));
    assert!(!queued.is_finished());

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_active"),
        1
    );
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_queued"),
        1
    );

    queued.abort();
    assert!(
        queued
            .await
            .expect_err("queued metrics request should be aborted")
            .is_cancelled()
    );
    drop(active_response);
}

#[tokio::test]
async fn connection_storm_disconnects_do_not_hide_control_plane() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &upstream.base_url,
        true,
        4,
        "max_queued_generation_requests = 8\ngeneration_queue_timeout_ms = 5000\n",
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1
"#,
        "",
        "",
    )
    .await;

    let mut downstreams = Vec::new();
    for index in 0..4 {
        let response = timeout(
            STREAM_COMPLETION_TIMEOUT,
            proxy
                .client
                .post(format!(
                    "{}/v1/chat/completions?test=connection-storm-{index}",
                    proxy.base_url
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"storm"}]}"#)
                .send(),
        )
        .await
        .expect("storm response headers should be bounded")
        .expect("storm response should receive headers");
        assert_eq!(response.status(), StatusCode::OK);
        downstreams.push(response.bytes_stream());
    }

    for downstream in &mut downstreams {
        let prefix = next_chunk(downstream, SHIELDED_HEARTBEAT_TIMEOUT, "storm JSON prefix").await;
        assert_eq!(prefix, Bytes::from_static(b" \n"));
    }

    let metrics = timeout(STREAM_HEADER_TIMEOUT, fetch_metrics(&proxy))
        .await
        .expect("metrics should not hide behind generation storm");
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_active"),
        4
    );

    let health = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .get(format!("{}/health", proxy.base_url))
            .send(),
    )
    .await
    .expect("health should not time out behind generation storm")
    .expect("health request should complete");
    assert!(health.status().is_success());
    let health_body = timeout(STREAM_HEADER_TIMEOUT, health.text())
        .await
        .expect("health body should be bounded")
        .expect("health body should read");
    assert!(!health_body.is_empty(), "health returned 0 bytes");

    let models = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .get(format!("{}/v1/models", proxy.base_url))
            .send(),
    )
    .await
    .expect("models should not time out behind generation storm")
    .expect("models request should complete");
    assert!(models.status().is_success());
    let models_body = timeout(STREAM_HEADER_TIMEOUT, models.text())
        .await
        .expect("models body should be bounded")
        .expect("models body should read");
    assert!(!models_body.is_empty(), "models returned 0 bytes");

    drop(downstreams);
    for _ in 0..4 {
        let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
        assert_eq!(drop_event.label, "cancellable-chat-sse");
    }
    wait_for_generation_metrics(&proxy, 0, 0, STREAM_COMPLETION_TIMEOUT).await;
    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_current_retained_request_terminals",
            &[
                ("status", "aborted"),
                ("terminal_reason", "downstream_disconnect"),
                ("http_status_class", "2xx"),
            ],
        ),
        4
    );
}

#[tokio::test]
async fn debug_summary_is_disabled_by_default() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .send()
        .await
        .expect("debug request should complete");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[test]
fn admin_token_matcher_accepts_only_exact_values() {
    assert!(admin_token_matches("admin-token", "admin-token"));
    assert!(!admin_token_matches("admin-tokeo", "admin-token"));
    assert!(!admin_token_matches("admin-token-extra", "admin-token"));
    assert!(!admin_token_matches("admin-toke", "admin-token"));
    assert!(!admin_token_matches("", "admin-token"));
}

#[tokio::test(flavor = "current_thread")]
async fn persistence_tasks_contain_spawn_blocking_panics() {
    let tasks = Arc::new(PersistenceTasks::default());

    tasks.spawn_blocking(|| panic!("simulated persistence store teardown failure"));

    timeout(
        STREAM_COMPLETION_TIMEOUT,
        tasks.flush(STREAM_COMPLETION_TIMEOUT),
    )
    .await
    .expect("panic-safe persistence task should finish");
    assert_eq!(tasks.panics.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_tasks_drop_work_when_the_bounded_backlog_is_full() {
    let tasks = Arc::new(PersistenceTasks::with_capacity_for_tests(1));
    let (first_started_tx, first_started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let (second_started_tx, second_started_rx) = std::sync::mpsc::channel();

    tasks.spawn_blocking(move || {
        first_started_tx
            .send(())
            .expect("first persistence task receiver should remain open");
        release_rx
            .recv()
            .expect("first persistence task should be released");
    });
    first_started_rx
        .recv_timeout(STREAM_COMPLETION_TIMEOUT)
        .expect("first persistence task should start");

    tasks.spawn_blocking(move || {
        second_started_tx
            .send(())
            .expect("dropped persistence task must not execute");
    });

    assert!(
        second_started_rx
            .recv_timeout(Duration::from_millis(250))
            .is_err(),
        "overflow persistence work must not execute while the backlog is saturated"
    );
    assert_eq!(
        tasks.dropped_total(),
        1,
        "backlog saturation must increment the dropped_total metric counter"
    );
    release_tx
        .send(())
        .expect("first persistence task should still be waiting");
    timeout(
        STREAM_COMPLETION_TIMEOUT,
        tasks.flush(STREAM_COMPLETION_TIMEOUT),
    )
    .await
    .expect("retained persistence task should drain");
}

async fn terminate_and_reap_test_child(
    child: &mut tokio::process::Child,
    process_group_id: u32,
) -> (String, std::io::Result<std::process::ExitStatus>) {
    #[cfg(unix)]
    let termination = kill(
        Pid::from_raw(-process_group_id.cast_signed()),
        Signal::SIGKILL,
    )
    .map_or_else(|error| error.to_string(), |()| String::from("SIGKILL sent"));
    #[cfg(not(unix))]
    let termination = child
        .start_kill()
        .map_or_else(|error| error.to_string(), |()| String::from("SIGKILL sent"));
    let reaped = child.wait().await;
    (termination, reaped)
}

async fn drain_test_child_output(
    stdout_reader: tokio::task::JoinHandle<Vec<u8>>,
    stderr_reader: tokio::task::JoinHandle<Vec<u8>>,
) -> (Vec<u8>, Vec<u8>) {
    let stdout = stdout_reader
        .await
        .expect("backlog-drop child stdout reader should not panic");
    let stderr = stderr_reader
        .await
        .expect("backlog-drop child stderr reader should not panic");
    (stdout, stderr)
}

async fn run_bounded_test_child(test_name: &str, child_env: &str) -> std::process::Output {
    let mut command = tokio::process::Command::new(
        std::env::current_exe().expect("test binary path should be available"),
    );
    command
        .args(["--exact", test_name, "--nocapture"])
        .env(child_env, "1")
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command
        .spawn()
        .expect("backlog-drop child test should start");
    let process_group_id = child.id().expect("backlog-drop child process group id");
    let mut child_stdout = child
        .stdout
        .take()
        .expect("backlog-drop child stdout should be captured");
    let mut child_stderr = child
        .stderr
        .take()
        .expect("backlog-drop child stderr should be captured");
    let stdout_reader = tokio::spawn(async move {
        let mut output = Vec::new();
        child_stdout
            .read_to_end(&mut output)
            .await
            .expect("backlog-drop child stdout should drain");
        output
    });
    let stderr_reader = tokio::spawn(async move {
        let mut output = Vec::new();
        child_stderr
            .read_to_end(&mut output)
            .await
            .expect("backlog-drop child stderr should drain");
        output
    });
    let status = match timeout(Duration::from_secs(30), child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            let (termination, reaped) =
                terminate_and_reap_test_child(&mut child, process_group_id).await;
            let (stdout, stderr) = drain_test_child_output(stdout_reader, stderr_reader).await;
            panic!(
                "backlog-drop child test wait failed: {error}; termination={termination}; reaped={reaped:?}; stdout={}; stderr={}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr),
            );
        }
        Err(_) => {
            let (termination, reaped) =
                terminate_and_reap_test_child(&mut child, process_group_id).await;
            let (stdout, stderr) = drain_test_child_output(stdout_reader, stderr_reader).await;
            panic!(
                "backlog-drop child test timed out after 30 seconds; termination={termination}; reaped={reaped:?}; stdout={}; stderr={}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr),
            );
        }
    };
    let (stdout, stderr) = drain_test_child_output(stdout_reader, stderr_reader).await;
    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_tasks_rate_limit_backlog_drop_logs_during_a_burst() {
    const CHILD_ENV: &str = "LLM_GUARD_PROXY_PERSISTENCE_DROP_LOG_TEST_CHILD";
    const HANG_CHILD_ENV: &str = "LLM_GUARD_PROXY_PERSISTENCE_DROP_LOG_TEST_HANG_CHILD";
    const TEST_NAME: &str =
        "proxy::tests::persistence_tasks_rate_limit_backlog_drop_logs_during_a_burst";
    const OVERFLOW_BURST: usize = 128;

    if std::env::var_os(CHILD_ENV).is_none() {
        let output = run_bounded_test_child(TEST_NAME, CHILD_ENV).await;
        assert!(
            output.status.success(),
            "backlog-drop child test failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
            "backlog-drop child test must run exactly one test: {}",
            String::from_utf8_lossy(&output.stdout)
        );

        let stderr = String::from_utf8_lossy(&output.stderr);
        let log_emissions = stderr
            .matches("persistence backlog full, dropping record")
            .count();
        assert!(
            log_emissions <= 1,
            "a saturated overflow burst must emit at most one backlog-drop log, emitted {log_emissions}"
        );
        assert!(
            stderr.contains("dropped_since_last_log=1"),
            "the first aggregated backlog-drop log must report its dropped delta: {stderr}"
        );
        return;
    }

    if std::env::var_os(HANG_CHILD_ENV).is_some() {
        sleep(Duration::from_secs(60)).await;
    }

    let tasks = Arc::new(PersistenceTasks::with_capacity_for_tests(1));
    let (first_started_tx, first_started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    tasks.spawn_blocking(move || {
        first_started_tx
            .send(())
            .expect("first persistence task receiver should remain open");
        release_rx
            .recv()
            .expect("first persistence task should be released");
    });
    first_started_rx
        .recv_timeout(STREAM_COMPLETION_TIMEOUT)
        .expect("first persistence task should start");

    for _ in 0..OVERFLOW_BURST {
        tasks.spawn_blocking(|| {});
    }
    assert_eq!(
        tasks.dropped_total(),
        OVERFLOW_BURST as u64,
        "every saturated overflow must remain visible in the dropped counter"
    );

    release_tx
        .send(())
        .expect("first persistence task should still be waiting");
    timeout(
        STREAM_COMPLETION_TIMEOUT,
        tasks.flush(STREAM_COMPLETION_TIMEOUT),
    )
    .await
    .expect("retained persistence task should drain");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_tasks_timeout_and_reap_a_hung_backlog_drop_child() {
    const HANG_CHILD_ENV: &str = "LLM_GUARD_PROXY_PERSISTENCE_DROP_LOG_TEST_HANG_CHILD";
    const TEST_NAME: &str =
        "proxy::tests::persistence_tasks_rate_limit_backlog_drop_logs_during_a_burst";

    let mut command = tokio::process::Command::new(
        std::env::current_exe().expect("test binary path should be available"),
    );
    command
        .args(["--exact", TEST_NAME, "--nocapture"])
        .env(HANG_CHILD_ENV, "1")
        .kill_on_drop(true)
        .stderr(Stdio::null());
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command
        .spawn()
        .expect("hung backlog-drop parent test should start");
    let process_group_id = child.id().expect("hung backlog-drop parent child pid");

    // The nested child has a 30-second deadline; leave startup scheduling headroom
    // when the full suite is running concurrently.
    let status = match timeout(Duration::from_secs(60), child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            let (termination, reaped) =
                terminate_and_reap_test_child(&mut child, process_group_id).await;
            panic!(
                "hung backlog-drop parent wait failed: {error}; termination={termination}; reaped={reaped:?}"
            );
        }
        Err(_) => {
            let (termination, reaped) =
                terminate_and_reap_test_child(&mut child, process_group_id).await;
            panic!(
                "backlog-drop parent must fail after terminating its hung child; termination={termination}; reaped={reaped:?}"
            );
        }
    };
    assert!(
        !status.success(),
        "backlog-drop parent should fail after its child timeout"
    );
}

#[tokio::test]
async fn error_shutdown_signals_persistence_tracked_shadow_work_before_flushing() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let persistence_tasks = Arc::clone(&proxy.state.persistence_tasks);
    let task_state = proxy.state.clone();
    let (started_tx, started_rx) = oneshot::channel();

    let shadow_task = tokio::spawn(async move {
        let _task_guard = persistence_tasks.track();
        let mut shutdown = task_state.shutdown.subscribe();
        started_tx
            .send(())
            .expect("shadow task startup receiver should remain open");
        shutdown.cancelled().await;
    });
    timeout(STREAM_COMPLETION_TIMEOUT, started_rx)
        .await
        .expect("shadow task should start before error shutdown")
        .expect("shadow task must not stop before error shutdown");

    proxy.state.begin_shutdown();
    timeout(STREAM_COMPLETION_TIMEOUT, proxy.state.flush_persistence())
        .await
        .expect("error shutdown should let persistence-tracked shadow work drain");
    timeout(STREAM_COMPLETION_TIMEOUT, shadow_task)
        .await
        .expect("shadow task should observe the error shutdown signal")
        .expect("shadow task should not panic");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_flush_does_not_miss_a_zero_transition_after_observing_work() {
    let (arrived_tx, arrived_rx) = std::sync::mpsc::channel();
    let release = Arc::new(std::sync::Barrier::new(2));
    let notification_registered = Arc::new(AtomicBool::new(false));
    let tasks = Arc::new(PersistenceTasks {
        flush_wait_hook: Some(PersistenceFlushWaitHook {
            arrived: arrived_tx,
            release: Arc::clone(&release),
            notification_registered: Arc::clone(&notification_registered),
        }),
        ..PersistenceTasks::default()
    });
    let guard = tasks.track();
    let flush_tasks = Arc::clone(&tasks);
    let mut flush = tokio::spawn(async move {
        flush_tasks.flush(STREAM_COMPLETION_TIMEOUT).await;
    });

    if arrived_rx.recv_timeout(STREAM_COMPLETION_TIMEOUT).is_err() {
        flush.abort();
        drop(guard);
        panic!("flush must register for completion before observing in-flight work");
    }
    if !notification_registered.load(Ordering::SeqCst) {
        release.wait();
        flush.abort();
        drop(guard);
        panic!("flush must register for completion before observing in-flight work");
    }
    drop(guard);
    release.wait();

    match timeout(STREAM_COMPLETION_TIMEOUT, &mut flush).await {
        Ok(result) => result.expect("flush task should not panic"),
        Err(_elapsed) => {
            flush.abort();
            panic!("flush must complete after the last persistence task finishes");
        }
    }
}

#[test]
fn request_cleanup_log_line_is_bounded_and_payload_free() {
    let mut request_metadata = BTreeMap::new();
    request_metadata.insert(
        String::from("authorization"),
        String::from("opaque-auth-marker"),
    );
    let mut response_metadata = BTreeMap::new();
    response_metadata.insert(
        String::from("upstream_error"),
        String::from("contains sk-live-secret"),
    );
    let request = RequestRecord {
        request_id: RequestId::from_string("req-cleanup-log")
            .expect("static request id should be valid"),
        started_at_unix_ms: 1_000,
        finished_at_unix_ms: Some(1_050),
        downstream_mode: DownstreamMode::NonStreamJson,
        upstream_mode: UpstreamMode::Streaming,
        model_id: Some(String::from("secret-model-sk-live")),
        input_fingerprint: Some(String::from("fingerprint-secret")),
        status: RequestStatus::Aborted,
        http_status: Some(200),
        error_reason: Some(String::from("upstream_stream_error: sk-live-secret")),
        abort_reason: Some(String::from("downstream_body_dropped_before_eof")),
        request_metadata,
        response_metadata,
        raw_payloads: RawPayloads::default(),
    };

    let line = request_cleanup_log_line(&request, "downstream_disconnect", 7, false);

    assert!(line.contains("request_id=req-cleanup-log"));
    assert!(line.contains("terminal_reason=downstream_disconnect"));
    assert!(line.contains("cleanup_latency_ms=7"));
    assert!(line.contains("evidence_written=false"));
    assert_safe_operational_text("request cleanup log", &line);
    assert!(!line.contains("secret-model"));
    assert!(!line.contains("fingerprint-secret"));
}

#[tokio::test]
async fn debug_summary_is_gated_bounded_and_redacted() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 2
"#,
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live-secret",
            proxy.base_url
        ))
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .header("x-api-key", "sk-header-secret")
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv_next().await;

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .send()
        .await
        .expect("unauthorized debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-tokeo")
        .send()
        .await
        .expect("bearer near-miss debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header("x-admin-token", "admin-token-extra")
        .send()
        .await
        .expect("admin-token near-miss debug request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests?limit=50", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("authorized debug request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("debug body should be text");
    assert!(body.contains("\"limit\":2"));
    assert!(body.contains("\"request_count\":1"));
    assert!(body.contains("\"status\":"));
    assert_safe_operational_text("debug summary", &body);

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests", proxy.base_url))
        .header("x-admin-token", "admin-token")
        .send()
        .await
        .expect("x-admin-token debug request should complete");
    assert_eq!(response.status(), StatusCode::OK);
}

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
        .header("x-admin-token", "admin-secret")
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
        observed.headers.get("x-admin-token").is_none(),
        "debug/admin token headers must not be forwarded upstream"
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
async fn get_models_enriches_context_metadata_and_preserves_unknown_fields() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "aeon-ultimate");
    assert_eq!(model["owned_by"], "vllm");
    assert_eq!(model["extra"], "keep");
    assert_eq!(model["max_model_len"].as_u64(), Some(256_000));
    assert_eq!(model["context_length"].as_u64(), Some(256_000));
    assert_eq!(model["max_context_length"].as_u64(), Some(256_000));

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/models?test=model-metadata");
}

#[tokio::test]
async fn get_models_enriches_chunked_context_metadata_without_content_length() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-chunked",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "chunked-model");
    assert_eq!(model["owned_by"], "vllm");
    assert_eq!(model["extra"], "keep");
    assert_eq!(model["max_model_len"].as_u64(), Some(256_000));
    assert_eq!(model["context_length"].as_u64(), Some(256_000));
    assert_eq!(model["max_context_length"].as_u64(), Some(256_000));

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-chunked"
    );
}

#[tokio::test]
async fn upstream_context_length_overrides_stale_max_model_len_fallback() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-context-length",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "context-length-model");
    assert_normalized_context_fields(&model, 256_000);

    let observed = fake.recv().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-context-length"
    );
}

#[tokio::test]
async fn upstream_max_context_length_overrides_stale_max_model_len_fallback() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-max-context-length",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "max-context-length-model");
    assert_normalized_context_fields(&model, 256_000);

    let observed = fake.recv().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-max-context-length"
    );
}

#[tokio::test]
async fn enriched_models_response_bypasses_generation_in_flight_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_request = empty_get_request("/v1/models?test=model-metadata-large");

    let first_response = proxy_handler(State(proxy.state.clone()), first_request).await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first models request should reach upstream");
    assert_eq!(first_observed.method, Method::GET);
    assert_eq!(
        first_observed.path_and_query,
        "/v1/models?test=model-metadata-large"
    );
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "enriched model responses must not be recorded before downstream body completion"
    );

    let second_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(second_response.status(), StatusCode::OK);
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("control-plane models request should bypass generation capacity");
    assert_eq!(
        second_observed.path_and_query,
        "/v1/models?test=model-metadata"
    );
    let second_body = to_bytes(second_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("second model body should read");
    let second_body =
        String::from_utf8(second_body.to_vec()).expect("second model body should be utf-8");
    assert_eq!(
        first_model(&second_body)["context_length"].as_u64(),
        Some(256_000)
    );

    let first_body = to_bytes(first_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("first enriched model body should read");
    let first_body =
        String::from_utf8(first_body.to_vec()).expect("first enriched model body should be utf-8");
    let first_model_record = first_model(&first_body);
    assert_eq!(first_model_record["context_length"].as_u64(), Some(256_000));
    assert_eq!(
        first_model_record["extra"]
            .as_str()
            .expect("large extra field should stay present")
            .len(),
        LARGE_MODEL_METADATA_EXTRA_BYTES
    );

    let third_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(third_response.status(), StatusCode::OK);
    let third_body = to_bytes(third_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("third model body should read");
    let third_body =
        String::from_utf8(third_body.to_vec()).expect("third model body should be utf-8");
    assert_eq!(
        first_model(&third_body)["context_length"].as_u64(),
        Some(256_000)
    );
    let third_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("third request should reach upstream without waiting on generation capacity");
    assert_eq!(
        third_observed.path_and_query,
        "/v1/models?test=model-metadata"
    );
}

#[tokio::test]
async fn models_burst_above_old_control_plane_cap_succeeds_and_health_stays_responsive() {
    const BURST_SIZE_ABOVE_OLD_CAP: usize = 8;

    let default_control_plane_cap = AppConfig::default()
        .server
        .max_control_plane_in_flight_requests;
    assert!(default_control_plane_cap >= BURST_SIZE_ABOVE_OLD_CAP);

    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let mut active_model_responses = Vec::with_capacity(BURST_SIZE_ABOVE_OLD_CAP);

    for _ in 0..BURST_SIZE_ABOVE_OLD_CAP {
        let response = proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        active_model_responses.push(response);
        let observed = fake
            .recv_within(STREAM_HEADER_TIMEOUT)
            .await
            .expect("model burst request should reach upstream");
        assert_eq!(observed.path_and_query, "/v1/models?test=model-metadata");
    }

    let health_response = timeout(
        STREAM_COMPLETION_TIMEOUT,
        proxy
            .client
            .get(format!("{}/health", proxy.base_url))
            .send(),
    )
    .await
    .expect("health should stay responsive during model burst")
    .expect("health request should complete");
    assert_eq!(health_response.status(), StatusCode::OK);
    let health_body = health_response
        .text()
        .await
        .expect("health body should be text");
    let health: serde_json::Value =
        serde_json::from_str(&health_body).expect("health should be JSON");
    assert_eq!(health["process"], "alive");
    assert_eq!(health["upstream"], "ready");

    let health_probe = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("health probe should reach upstream during model burst");
    assert_eq!(health_probe.path_and_query, "/v1/models");

    drop(active_model_responses);
}

mod adapter_response_fixtures;
mod deepinfra_rerank_endpoint;
mod score_endpoint;
mod upstream_failover;

use adapter_response_fixtures::{fake_deepinfra_score_response, fake_rerank_response};
#[tokio::test]
async fn enriched_models_observability_records_success_after_body_consumption() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream");
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "success must wait until the enriched body reaches EOF"
    );

    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("enriched model body should read");
    let expected_body_len = body.len().to_string();
    let body = String::from_utf8(body.to_vec()).expect("enriched model body should be utf-8");
    assert_eq!(first_model(&body)["context_length"].as_u64(), Some(256_000));

    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        2
    );
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(request_row.abort_reason, None);
    assert_eq!(
        request_row.response_metadata["response_body_bytes"],
        expected_body_len.as_str()
    );
    assert_eq!(request_row.response_metadata["http_status_success"], "true");
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(attempt_row.abort_reason, None);
    assert_eq!(
        attempt_row.response_metadata["response_body_bytes"],
        expected_body_len.as_str()
    );
}

#[tokio::test]
async fn forwarded_embedding_observability_records_upstream_token_usage() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/embeddings?test=token-usage", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-embedding","input":"ping"}"#)
        .send()
        .await
        .expect("embedding request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let response_body = response
        .bytes()
        .await
        .expect("embedding response body should be readable");
    assert_eq!(
        response_body.as_ref(),
        br#"{"object":"list","data":[{"embedding":[0.0]}],"usage":{"prompt_tokens":17,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}"#
    );
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=token-usage");

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let token_usage: (Option<i64>, Option<i64>, Option<i64>, Option<i64>) = connection
        .query_row(
            "SELECT input_tokens, output_tokens, cached_input_tokens, reasoning_tokens FROM attempts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("attempt token usage should be stored");
    assert_eq!(token_usage, (Some(17), Some(4), Some(3), Some(2)));
}

#[tokio::test]
async fn enriched_models_observability_records_abort_when_body_is_dropped() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream");
    assert_eq!(
        proxy
            .store
            .retention_usage()
            .expect("usage should be readable")
            .record_count,
        0,
        "droppable response body should own the pending observability record"
    );

    drop(response);

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(request_row.response_metadata["response_body_bytes"], "0");
    assert_eq!(attempt_row.status, "aborted");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempt_row.response_metadata["response_body_bytes"], "0");
}

#[tokio::test]
async fn get_models_reflects_upstream_metadata_changes_between_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let first = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-changing",
            proxy.base_url
        ))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;
    let second = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-changing",
            proxy.base_url
        ))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert_eq!(
        first_model(&first)["context_length"].as_u64(),
        Some(128_000)
    );
    assert_eq!(
        first_model(&second)["context_length"].as_u64(),
        Some(256_000)
    );
}

#[tokio::test]
async fn disabled_model_metadata_discovery_or_enrichment_returns_upstream_body_unchanged() {
    for metadata_config in [
        r"
[upstream.metadata]
discovery_enabled = false
enrich_responses = true
",
        r"
[upstream.metadata]
discovery_enabled = true
enrich_responses = false
",
    ] {
        let fake = FakeUpstream::spawn().await;
        let proxy =
            ProxyFixture::spawn_with_metadata_config(&fake.base_url, true, metadata_config).await;

        let response = proxy
            .client
            .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
            .send()
            .await
            .expect("proxy request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.text().await.expect("body should be text"),
            MODEL_METADATA_BODY
        );
        let _observed = fake.recv().await;
    }
}

#[tokio::test]
async fn config_fallback_context_metadata_is_hot_reloadable() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
max_model_len_override = 4096
",
    )
    .await;

    let first = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
max_model_len_override = 8192
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("metadata reload should succeed");

    let second = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert!(outcome.applied);
    assert_eq!(first_model(&first)["context_length"].as_u64(), Some(4_096));
    assert_eq!(first_model(&first)["max_model_len"].as_u64(), Some(4_096));
    assert_eq!(first_model(&second)["context_length"].as_u64(), Some(8_192));
    assert_eq!(first_model(&second)["max_model_len"].as_u64(), Some(8_192));
}

#[tokio::test]
async fn model_metadata_uses_named_profile_context_override_for_matching_record() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fallback-route"
base_url = "{base_url}"
match_models = ["fallback-model"]

[upstreams.metadata]
context_length_override = 12345
"#,
            base_url = fake.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?test=model-metadata-no-context",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("body should be text");
    let model = first_model(&body);
    assert_eq!(model["id"], "fallback-model");
    assert_eq!(model["extra"], "keep");
    assert_normalized_context_fields(&model, 12_345);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=model-metadata-no-context"
    );
}

#[tokio::test]
async fn hot_reloaded_disabled_discovery_stops_model_metadata_enrichment() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let enriched = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("first proxy request should complete")
        .text()
        .await
        .expect("first body should be text");
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
discovery_enabled = false
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("metadata reload should succeed");

    let disabled = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("second proxy request should complete")
        .text()
        .await
        .expect("second body should be text");
    let _second_observed = fake.recv_next().await;

    assert!(outcome.applied);
    assert_eq!(
        first_model(&enriched)["context_length"].as_u64(),
        Some(256_000)
    );
    assert_eq!(disabled, MODEL_METADATA_BODY);
}

#[tokio::test]
async fn hermes_like_context_extraction_reads_enriched_model_length() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let body = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("proxy request should complete")
        .text()
        .await
        .expect("body should be text");

    let model = first_model(&body);
    assert_eq!(hermes_like_context_length(&model), Some(256_000));
    let _observed = fake.recv().await;
}

async fn spawn_shielded_watchdog_proxy(upstream_base_url: &str) -> ProxyFixture {
    ProxyFixture::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
        upstream_base_url,
        observability_enabled: true,
        max_in_flight_requests: 8,
        metadata_config: "",
        server_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: r"
[upstream.stuck_watchdog]
enabled = true
detection_window_secs = 60
min_output_progress_units_in_window = 1
check_interval_secs = 1
",
    })
    .await
}

fn assert_shielded_response_headers(response: &reqwest::Response) -> String {
    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response
        .headers()
        .get("x-request-id")
        .expect("terminal response should include x-request-id")
        .to_str()
        .expect("x-request-id should be valid header text")
        .to_owned();
    assert_ne!(
        request_id, "upstream-request-id-collision",
        "proxy terminal response must overwrite the upstream request ID"
    );
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be JSON for non-stream downstream clients"),
        "application/json"
    );
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("shielded fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    request_id
}

fn assert_shielded_watchdog_progress(proxy: &ProxyFixture) {
    assert!(
        proxy.state.stuck_watchdog_tokens.sample_count("default") > 0,
        "shielded upstream SSE content must reach the watchdog before aggregation completes"
    );
}

#[tokio::test]
async fn shielded_non_stream_chat_forces_upstream_sse_and_aggregates_json() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_watchdog_proxy(&fake.base_url).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"stream":false}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=request-id-collision",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("proxy request should complete");

    let response_request_id = assert_shielded_response_headers(&response);
    assert_shielded_watchdog_progress(&proxy);
    let response_body = response.text().await.expect("response body should be text");
    assert!(
        !response_body.starts_with(": llm-guard-proxy heartbeat"),
        "non-stream response must not start with SSE heartbeat: {response_body:?}"
    );
    assert!(
        !response_body.contains("event: final"),
        "non-stream response must not contain SSE final framing: {response_body:?}"
    );
    let aggregated: serde_json::Value =
        serde_json::from_str(&response_body).expect("non-stream response should be JSON");
    assert_eq!(aggregated["id"], "chatcmpl-shielded");
    assert_eq!(aggregated["object"], "chat.completion");
    assert_eq!(aggregated["created"], 1_710_000_000);
    assert_eq!(aggregated["model"], "test-chat");
    assert_eq!(aggregated["choices"][0]["index"], 0);
    assert_eq!(aggregated["choices"][0]["message"]["role"], "assistant");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        aggregated["choices"][0]["message"]["reasoning_content"],
        "think"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        r#"{"q":"x"}"#
    );
    assert!(
        aggregated["choices"][0]["message"]["tool_calls"][0]
            .get("index")
            .is_none()
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "stop");
    assert_eq!(aggregated["usage"]["prompt_tokens"], 3);
    assert_eq!(aggregated["usage"]["completion_tokens"], 2);
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let persisted_request = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(response_request_id, persisted_request.request_id);

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=request-id-collision"
    );
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "test-chat");
    assert_eq!(observed_body["messages"][0]["content"], "ping");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_trims_reasoning_separator_from_final_content() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=reasoning-leading-newlines",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"Say OK"}],"stream":false}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "OK");
    assert_eq!(
        aggregated["choices"][0]["message"]["reasoning_content"],
        "think before answering"
    );

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=reasoning-leading-newlines"
    );
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn pre_request_guard_allow_proceeds_to_upstream() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-allow");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "allow", guard_result("allow", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn pre_request_guard_block_returns_forbidden_without_upstream_call() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "block", guard_result("block", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"blocked"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(fake.recv_within(STREAM_SECOND_CHUNK_GUARD).await.is_none());
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn pre_request_guard_replace_swaps_messages_before_upstream_call() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-replace");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let replacement = r#"[{"role":"user","content":"redacted prompt"}]"#;
    let script = write_guard_script(
        &guard_root,
        "replace",
        guard_result("replace", Some(replacement)),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"secret prompt"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["messages"][0]["content"], "redacted prompt");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn pre_request_guard_error_fail_closed_blocks_or_allows_by_config() {
    let mut blocked_fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-error-fail-closed");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(
        &guard_root,
        "error-fail-closed",
        guard_result("error_fail_closed", None),
    );
    let blocked_proxy = ProxyFixture::spawn_with_options(
        &blocked_fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let blocked = blocked_proxy
        .client
        .post(format!("{}/v1/chat/completions", blocked_proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");
    assert_eq!(blocked.status(), StatusCode::FORBIDDEN);
    assert!(
        blocked_fake
            .recv_within(STREAM_SECOND_CHUNK_GUARD)
            .await
            .is_none()
    );

    let mut allowed_fake = FakeUpstream::spawn().await;
    let allowed_proxy = ProxyFixture::spawn_with_options(
        &allowed_fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, false),
    )
    .await;

    let allowed = allowed_proxy
        .client
        .post(format!("{}/v1/chat/completions", allowed_proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");
    assert_eq!(allowed.status(), StatusCode::OK);
    let _json = shielded_final_json(allowed).await;
    let _observed = allowed_fake.recv_next().await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn post_response_guard_block_returns_safe_refusal() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-post-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "post-block", guard_result("block", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(None, Some(&script), true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "I can't help with that request."
    );
    assert_eq!(json["choices"][0]["finish_reason"], "content_filter");
    let _observed = fake.recv_next().await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn post_response_guard_replace_changes_client_response() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("guard-post-replace");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let replacement = r#"[{"id":"chatcmpl-guarded","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"guarded answer"},"finish_reason":"stop"}]}]"#;
    let script = write_guard_script(
        &guard_root,
        "post-replace",
        guard_result("replace", Some(replacement)),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(None, Some(&script), true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(json["id"], "chatcmpl-guarded");
    assert_eq!(json["choices"][0]["message"]["content"], "guarded answer");
    let _observed = fake.recv_next().await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_alias_returns_chat_completion() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-alias-success");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let replacement = r#"[{"role":"assistant","content":"workflow answer"}]"#;
    let script = write_guard_script(
        &guard_root,
        "workflow-alias-success",
        guard_result("replace", Some(replacement)),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &workflow_alias_config(&script, 120_000),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"ping"}]}"#,
        )
        .send()
        .await
        .expect("workflow alias request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "family/child-safe-general-v1");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "workflow answer");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["total_tokens"], 0);
    assert!(fake.recv_within(STREAM_SECOND_CHUNK_GUARD).await.is_none());
}

#[test]
#[cfg(feature = "guard")]
fn unknown_workflow_id_returns_error() {
    let config = AppConfig::default();
    let workflow_alias = ResolvedWorkflowAlias {
        workflow_id: String::from("missing_workflow"),
        timeout_ms: 120_000,
    };

    let error = workflow_config_for_alias(&config, &workflow_alias)
        .expect_err("missing workflow should fail closed");

    assert!(error.to_string().contains("unconfigured workflow"));
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_timeout_returns_error() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-alias-timeout");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_slow_guard_script(&guard_root, "workflow-alias-timeout");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &workflow_alias_config(&script, 20),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"ping"}]}"#,
        )
        .send()
        .await
        .expect("workflow alias request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("timed out"))
    );
    assert!(fake.recv_within(STREAM_SECOND_CHUNK_GUARD).await.is_none());
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_block_returns_error() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-alias-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(
        &guard_root,
        "workflow-alias-block",
        guard_result("block", None),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &workflow_alias_config(&script, 120_000),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"blocked"}]}"#,
        )
        .send()
        .await
        .expect("workflow alias request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("guard summary"))
    );
    assert!(fake.recv_within(STREAM_SECOND_CHUNK_GUARD).await.is_none());
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn models_endpoint_includes_aliases() {
    let fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("models-aliases");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(
        &guard_root,
        "models-aliases",
        guard_result("replace", Some(r#"[{"role":"assistant","content":"ok"}]"#)),
    );
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[model_aliases]]
id = "alias-chat"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "family/child-safe-general-v1"
kind = "workflow"
workflow_id = "child_safe_general_v1"

{}
"#,
            workflow_config("child_safe_general_v1", &script)
        ),
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/v1/models?test=model-metadata", proxy.base_url))
        .send()
        .await
        .expect("models request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("models body should be text");
    let json: serde_json::Value = serde_json::from_str(&body).expect("models should be JSON");
    let models = json["data"]
        .as_array()
        .expect("models data should be array");
    let model_ids = models
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();
    assert!(model_ids.contains(&"aeon-ultimate"));
    assert!(model_ids.contains(&"alias-chat"));
    assert!(model_ids.contains(&"family/child-safe-general-v1"));
    let alias_record = models
        .iter()
        .find(|model| model["id"] == "alias-chat")
        .expect("alias model record should exist");
    assert_eq!(alias_record["owned_by"], "llm-guard-proxy");
    assert_eq!(alias_record["llm_guard_proxy_alias"], true);
    assert_eq!(alias_record["alias_kind"], "upstream");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn non_streaming_completion_with_fake_upstream() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("chat request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(json["choices"][0]["message"]["content"], "Hello");
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn streaming_with_shielded_buffering() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("streaming chat request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(content_type.contains("text/event-stream"));
    let body = response.text().await.expect("stream body should be text");
    let chunks = openai_sse_json_chunks(&body);
    let content = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(content, "Hello");
    assert!(body.contains("data: [DONE]"));
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn unknown_model_returns_structured_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"missing-model","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("unknown model request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("model not allowed"))
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn child_profile_blocked_from_adult_alias() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["adult-model"]

[profiles.child]
kind = "child"
allowed_models = ["child-model"]

[virtual_keys]
enabled = true
unknown_key_policy = "fail_closed"

[virtual_keys.keys]
child-key = "child"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header("x-virtual-key", "child-key")
        .body(r#"{"model":"adult-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("child profile request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("model not allowed"))
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn budget_exhausted_returns_429() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 1

[budget]
enabled = true
"#,
    )
    .await;

    let allowed = send_budget_chat_request(&proxy, None).await;
    assert_eq!(allowed.status(), StatusCode::OK);
    let _body = allowed.text().await.expect("body should be text");
    let _observed = fake.recv_next().await;

    let blocked = send_budget_chat_request(&proxy, None).await;
    assert_eq!(blocked.status(), StatusCode::TOO_MANY_REQUESTS);
    let json = response_json(blocked).await;
    assert_eq!(json["error"]["type"], "budget_exhausted");
    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "default"), 1);
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn unknown_virtual_key_fails_closed() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("fail_closed"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header("x-virtual-key", "unknown-key")
        .body(r#"{"model":"gpt-default","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("unknown virtual key request should complete");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "virtual_key_unauthorized");
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_guard_returns_allow_passes_request() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-guard-allow");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(
        &guard_root,
        "workflow-guard-allow",
        guard_result("allow", None),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_guard_returns_block_rejects_request() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-guard-block");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(
        &guard_root,
        "workflow-guard-block",
        guard_result("block", None),
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"blocked"}]}"#)
        .send()
        .await
        .expect("guarded request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn malformed_guard_output_fails_closed() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-guard-malformed");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_literal_guard_script(&guard_root, "workflow-guard-malformed", "not json");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("malformed guard request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("malformed JSON"))
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn workflow_timeout_fails_closed() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-timeout-fails-closed");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_slow_guard_script(&guard_root, "workflow-timeout-fails-closed");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &workflow_alias_config(&script, 20),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("timeout workflow request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("timed out"))
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn large_stdout_capped() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("workflow-large-stdout");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_large_stdout_guard_script(&guard_root, "workflow-large-stdout");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &workflow_alias_config_with_stdout_limit(&script, 120_000, 1_048_576),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("large stdout workflow request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "guard_blocked");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("stdout exceeded"))
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn allowed_request_logged_in_audit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"audit"}]}"#)
        .send()
        .await
        .expect("audited request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let _observed = fake.recv_next().await;
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.http_status, 200);
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn blocked_request_logged_in_audit() {
    let mut fake = FakeUpstream::spawn().await;
    let guard_root = unique_test_dir("blocked-audit");
    fs::create_dir_all(&guard_root).expect("guard root should be created");
    let script = write_guard_script(&guard_root, "blocked-audit", guard_result("block", None));
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &guard_workflow_config(Some(&script), None, true),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"blocked"}]}"#)
        .send()
        .await
        .expect("blocked audited request should complete");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let _json = response_json(response).await;
    assert_no_upstream_request(&mut fake).await;
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 403);
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with("guard_blocked"))
    );
    assert_eq!(audit_row_count(&proxy.sqlite_path, "attempts"), 0);
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn api_keys_never_logged() {
    let secret_virtual_key = "vk_child_def456";
    let secret_authorization = "sk-auth-secret-issue-79";
    let secret_query_key = "sk-query-secret-issue-79";
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("fail_closed"),
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/completions?api_key={secret_query_key}",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, format!("Bearer {secret_authorization}"))
        .header("x-api-key", "sk-header-secret-issue-79")
        .header("x-virtual-key", secret_virtual_key)
        .body(r#"{"model":"child-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("secret audit request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");
    let _observed = fake.recv_next().await;
    let audit_text = read_audit_text(&proxy.sqlite_path);
    for secret in [
        secret_virtual_key,
        secret_authorization,
        secret_query_key,
        "sk-header-secret-issue-79",
    ] {
        assert!(
            !audit_text.contains(secret),
            "audit storage must not contain secret value {secret}"
        );
    }
    assert!(audit_text.contains("child_safe"));
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn request_counted_against_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 2

[budget]
enabled = true
"#,
    )
    .await;

    let response = send_budget_chat_request(&proxy, None).await;

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");
    let _observed = fake.recv_next().await;
    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "default"), 1);
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn budget_exhausted_blocks_request() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 2

[budget]
enabled = true
"#,
    )
    .await;

    for _ in 0..2 {
        let response = send_budget_chat_request(&proxy, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.text().await.expect("body should be text");
        let _observed = fake.recv_next().await;
    }
    let blocked = send_budget_chat_request(&proxy, None).await;

    assert_eq!(blocked.status(), StatusCode::TOO_MANY_REQUESTS);
    let json = response_json(blocked).await;
    assert_eq!(json["error"]["type"], "budget_exhausted");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("limit=2"))
    );
    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "default"), 2);
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn different_profiles_independent() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.adult]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 1

[profiles.child]
kind = "child"
allowed_models = ["test-chat"]
daily_request_limit = 1

[virtual_keys]
enabled = true
unknown_key_policy = "fail_closed"

[virtual_keys.keys]
adult-key = "adult"
child-key = "child"

[budget]
enabled = true
"#,
    )
    .await;

    let adult = send_budget_chat_request(&proxy, Some("adult-key")).await;
    assert_eq!(adult.status(), StatusCode::OK);
    let _body = adult.text().await.expect("body should be text");
    let _adult_observed = fake.recv_next().await;
    let child = send_budget_chat_request(&proxy, Some("child-key")).await;
    assert_eq!(child.status(), StatusCode::OK);
    let _body = child.text().await.expect("body should be text");
    let _child_observed = fake.recv_next().await;
    let adult_blocked = send_budget_chat_request(&proxy, Some("adult-key")).await;

    assert_eq!(adult_blocked.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "adult"), 1);
    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "child"), 1);
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn unlimited_profile_not_counted() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 0

[budget]
enabled = true
"#,
    )
    .await;

    for _ in 0..2 {
        let response = send_budget_chat_request(&proxy, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.text().await.expect("body should be text");
        let _observed = fake.recv_next().await;
    }

    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "default"), 0);
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn budget_disabled_no_check() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["test-chat"]
daily_request_limit = 1

[budget]
enabled = false
"#,
    )
    .await;

    for _ in 0..2 {
        let response = send_budget_chat_request(&proxy, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.text().await.expect("body should be text");
        let _observed = fake.recv_next().await;
    }

    assert_eq!(read_budget_count(&proxy.budget_sqlite_path, "default"), 0);
}

#[tokio::test]
async fn shielded_loop_guard_catches_reasoning_line_repeated_hundreds_of_times() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("reasoning loop line"));

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    for metadata in [
        &request_row.response_metadata,
        &attempt_row.response_metadata,
    ] {
        assert_eq!(metadata["loop_detected"], "true");
        assert_eq!(metadata["loop_signal"], "repeated_line");
        assert_eq!(metadata["loop_channel"], "reasoning");
        assert!(
            metadata["loop_sample_hash"]
                .as_str()
                .expect("hash should be a string")
                .starts_with("fnv64:")
        );
        assert!(!metadata.to_string().contains("reasoning loop line"));
    }
}

#[tokio::test]
async fn shielded_loop_guard_catches_semantic_reasoning_repetition_with_varied_wording() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 100
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
reasoning_semantic_similarity_threshold_percent = 45
reasoning_semantic_window_token_count = 24
reasoning_semantic_minimum_token_count = 8
reasoning_semantic_history_window_count = 4

[retry]
enabled = false
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=semantic-reasoning-varied",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("bsdtar"));
    assert!(!body.contains("zipfile"));

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    for metadata in [
        &request_row.response_metadata,
        &attempt_row.response_metadata,
    ] {
        assert_eq!(metadata["loop_detected"], "true");
        assert_eq!(metadata["loop_signal"], "semantic_jaccard");
        assert_eq!(metadata["loop_channel"], "reasoning");
        assert!(
            metadata["loop_semantic_similarity_percent"]
                .as_str()
                .and_then(|value| value.parse::<u64>().ok())
                .expect("semantic similarity should be numeric")
                >= 45
        );
        assert!(!metadata.to_string().contains("bsdtar"));
        assert!(!metadata.to_string().contains("zipfile"));
    }
}

#[tokio::test]
async fn shielded_loop_guard_monitor_records_reasoning_signal_without_abort() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
output_repeated_line_threshold = 4
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    let metadata = &attempt_row.response_metadata;
    assert_eq!(metadata["loop_detector_mode"], "monitor");
    assert_eq!(metadata["loop_signal_0_channel"], "reasoning");
    assert_eq!(metadata["loop_signal_0_reason_code"], "repeated_line");
    assert_eq!(metadata["loop_signal_0_severity"], "abort_candidate");
    assert!(metadata.get("loop_detected").is_none());
    assert!(!metadata.to_string().contains("reasoning loop line"));
}

#[tokio::test]
async fn shielded_loop_guard_disabled_skips_detector_metadata() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "disabled"
output_repeated_line_threshold = 4
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    assert!(
        attempt_row
            .response_metadata
            .get("loop_detector_mode")
            .is_none()
    );
    assert!(
        attempt_row
            .response_metadata
            .get("loop_signal_count")
            .is_none()
    );
    assert!(attempt_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn shielded_loop_guard_monitor_records_tool_argument_and_fingerprint_signals() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=repeated-tool-fingerprint",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    let metadata = &attempt_row.response_metadata;
    let metadata_text = metadata.to_string();
    assert_eq!(metadata["loop_detector_mode"], "monitor");
    assert!(metadata_text.contains("tool_arguments"));
    assert!(metadata_text.contains("tool_arguments_json_completed"));
    assert!(metadata_text.contains("tool_fingerprint"));
    assert!(metadata_text.contains("tool_fingerprint_repeated"));
    assert!(metadata_text.contains("fingerprint_hash"));
    assert!(!metadata_text.contains(r#""q":"#));
    assert!(!metadata_text.contains("limit"));
}

#[tokio::test]
async fn debug_summary_exposes_bounded_loop_detector_metadata_without_raw_payloads() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "monitor"
"#,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 5
"#,
        "",
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=repeated-tool-fingerprint",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"debug-summary-prompt-secret"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be text");

    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests?limit=5", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("debug summary request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("debug summary should be text");
    let summary: serde_json::Value =
        serde_json::from_str(&body).expect("debug summary should be JSON");
    let request = summary["requests"]
        .as_array()
        .and_then(|requests| {
            requests.iter().find(|request| {
                request["response_metadata"]["loop_detector_mode"].as_str() == Some("monitor")
            })
        })
        .expect("debug summary should include the loop request");
    let metadata = request["response_metadata"]
        .as_object()
        .expect("debug summary response metadata should be an object");
    let metadata_text = request["response_metadata"].to_string();

    assert_eq!(
        metadata
            .get("loop_detector_mode")
            .and_then(serde_json::Value::as_str),
        Some("monitor")
    );
    assert!(
        metadata_text.contains("tool_arguments_json_completed"),
        "debug summary should include bounded completed-JSON detector signal: {metadata_text}"
    );
    assert!(
        metadata_text.contains("tool_fingerprint_repeated"),
        "debug summary should include bounded fingerprint detector signal: {metadata_text}"
    );
    assert!(metadata_text.contains("fingerprint_hash"));
    assert!(metadata.len() < 200);
    assert!(!body.contains("debug-summary-prompt-secret"));
    assert!(!metadata_text.contains(r#""q":"#));
    assert!(!metadata_text.contains(r#""limit":1"#));
    assert!(!metadata_text.contains("lookup"));

    let metrics = fetch_metrics(&proxy).await;
    assert_metric_type(
        &metrics,
        "llm_guard_proxy_current_retained_requests",
        "gauge",
    );
    assert!(!metrics.contains("debug-summary-prompt-secret"));
    assert!(!metrics.contains(r#""q":"#));
    assert!(!metrics.contains(r#""limit":1"#));
    assert!(!metrics.contains("lookup"));
}

#[tokio::test]
async fn shielded_loop_guard_does_not_flag_repeated_input_without_output_loop() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
output_repeated_line_threshold = 4
"#,
    )
    .await;
    let repeated_input = format!("{REPEATED_INPUT_LOOP_LINE}\n{REPEATED_INPUT_LOOP_LINE}\n");
    let body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": repeated_input}],
    });

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert!(request_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn shielded_loop_guard_records_suspect_for_output_copying_repeated_input() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
output_repeated_line_threshold = 4
output_token_window_size = 8
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
input_overlap_threshold_multiplier = 3
"#,
    )
    .await;
    let body = repeated_input_chat_body();

    let under_threshold = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=copy-input-under-threshold",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("under-threshold request should complete");
    assert_eq!(under_threshold.status(), StatusCode::OK);
    let _under_body = under_threshold
        .text()
        .await
        .expect("under-threshold body should be text");

    let over_threshold = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=copy-input-over-threshold",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("over-threshold request should complete");
    assert_eq!(over_threshold.status(), StatusCode::OK);
    let _over_body = over_threshold
        .text()
        .await
        .expect("over-threshold body should be text");

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_reason_code"],
        "repeated_line"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_channel"],
        "content"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_severity"],
        "suspect"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_abort_candidate_count"],
        "0"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_residual_signal_count"],
        "1"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_content_signal_count"],
        "1"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_reasoning_signal_count"],
        "0"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_feature_threshold"],
        "12"
    );
    assert_eq!(
        attempt_row.response_metadata["loop_signal_0_feature_input_overlap_applied"],
        "true"
    );
    assert!(attempt_row.response_metadata.get("loop_detected").is_none());
}

#[tokio::test]
async fn hot_reloaded_loop_threshold_changes_subsequent_requests() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 10
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
"#,
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#;

    let before_reload = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-six",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(before_reload.status(), StatusCode::OK);
    let _before_body = before_reload
        .text()
        .await
        .expect("first body should be text");

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4
output_repeated_token_window_threshold = 100
output_suffix_cycle_threshold = 100
output_low_progress_min_bytes = 1000000
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("loop threshold reload should succeed");
    assert!(outcome.applied);

    let after_reload = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-six",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(after_reload.status(), StatusCode::BAD_GATEWAY);

    let attempt_row = read_last_observability_row(&proxy.sqlite_path, "attempts");
    assert_eq!(attempt_row.response_metadata["loop_detected"], "true");
    assert_eq!(attempt_row.response_metadata["loop_threshold"], "4");
}

#[tokio::test]
async fn shielded_retry_loops_once_then_succeeds_without_emitting_loop() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(!aggregated.to_string().contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&first_attempt.body));
    assert!(body_contains_retry_hint(&second_attempt.body));
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "successful retry should stop after the second upstream attempt"
    );

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_final_outcome"],
        "succeeded"
    );
    assert_eq!(request_row.response_metadata["retry_max_attempts"], "5");
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("loop_guard"));
    assert_eq!(attempts[0].response_metadata["loop_detected"], "true");
    assert_eq!(attempts[0].response_metadata["attempt_max_attempts"], "5");
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "succeeded");
    assert_eq!(attempts[1].response_metadata["attempt_max_attempts"], "5");
}

/// Regression test for issue #99: the anti-loop retry hint must not create a
/// non-leading `system` message when the original request already has a leading
/// `system` message. Qwen-style chat templates reject any non-leading system
/// message with a 400 error.
#[tokio::test]
async fn shielded_retry_preserves_leading_system_message_on_loop_retry() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"ping"}
            ]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;

    // First attempt has no retry hint.
    assert!(!body_contains_retry_hint(&first_attempt.body));

    // Second attempt has the retry hint merged into the existing system message.
    assert!(body_contains_retry_hint(&second_attempt.body));

    // The upstream must receive at most one system message, and it must be at
    // index 0. No non-leading system messages are allowed.
    let second_messages = serde_json::from_slice::<serde_json::Value>(&second_attempt.body)
        .expect("second attempt body should be valid JSON")
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .expect("messages should be an array")
        .clone();

    let system_count = second_messages
        .iter()
        .filter(|msg| msg.get("role").and_then(serde_json::Value::as_str) == Some("system"))
        .count();
    assert_eq!(
        system_count, 1,
        "exactly one system message should exist after retry hint injection"
    );

    let first_role = second_messages[0]
        .get("role")
        .and_then(serde_json::Value::as_str);
    assert_eq!(
        first_role,
        Some("system"),
        "the system message must remain at index 0"
    );

    // The merged content should contain both the original instruction and the hint.
    let merged_content = second_messages[0]
        .get("content")
        .and_then(serde_json::Value::as_str)
        .expect("merged system content should be a string");
    assert!(merged_content.contains("You are a helpful assistant."));
    assert!(merged_content.contains("llm-guard-proxy retry hint"));

    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "successful retry should stop after the second upstream attempt"
    );
}

/// Regression test for issue #99: when there is no existing system message,
/// the retry hint creates a new leading system message as before. This verifies
/// the existing behavior is preserved for requests without a system message.
#[tokio::test]
async fn shielded_retry_inserts_new_system_message_when_none_exists() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;

    let _first = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;

    assert!(body_contains_retry_hint(&second_attempt.body));

    let second_messages = serde_json::from_slice::<serde_json::Value>(&second_attempt.body)
        .expect("second attempt body should be valid JSON")
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .expect("messages should be an array")
        .clone();

    let first_role = second_messages[0]
        .get("role")
        .and_then(serde_json::Value::as_str);
    assert_eq!(
        first_role,
        Some("system"),
        "a new system message should be inserted at index 0"
    );
}

#[tokio::test]
async fn evidence_disabled_creates_no_evidence_artifacts_after_proxy_request() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let _observed = fake.recv_next().await;
    assert!(!proxy.evidence_sqlite_path.exists());
}

#[tokio::test]
async fn evidence_enabled_records_loop_primary_and_fallback_without_raw_payloads() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false
include_request_headers = false

[evidence.shadow]
enabled = false

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].role, "primary");
    assert_eq!(rows[0].shown_to_downstream, 0);
    assert_eq!(rows[0].status, "rejected");
    assert_eq!(rows[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(rows[0].detector_features["loop_detected"], "true");
    assert_eq!(rows[1].role, "fallback");
    assert_eq!(rows[1].shown_to_downstream, 1);
    assert_eq!(rows[1].status, "accepted");
    assert_eq!(rows[1].thinking_budget_tokens, Some(32_768));

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_groups"),
        1
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_chunks"),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE raw_input IS NOT NULL OR raw_output IS NOT NULL OR raw_reasoning IS NOT NULL OR raw_tool_calls IS NOT NULL",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_shadow_keep_false_does_not_record_shadow_or_extra_upstream_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = false
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 50

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let _primary = fake.recv_next().await;
    let _fallback = fake.recv_next().await;
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "keep_looping_attempt_running=false must abort the looped primary instead of issuing a shadow upstream request"
    );

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].role, "primary");
    assert_eq!(rows[0].status, "rejected");
    assert_eq!(rows[0].shown_to_downstream, 0);
    assert_eq!(rows[1].role, "fallback");
    assert_eq!(rows[1].status, "accepted");
    assert_eq!(rows[1].shown_to_downstream, 1);
    assert!(!rows.iter().any(|row| row.role == "shadow_continued"));

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_raw_capture_redacts_headers_and_payload_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r"
enabled = true
include_raw_payloads = true
include_request_headers = true
",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer downstream-secret")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Bearer qb secret «redacted:sk-…»"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _json = shielded_final_json(response).await;
    let _observed = fake.recv_next().await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let chunks = read_evidence_chunks(&connection);
    assert_eq!(chunks.len(), 6);
    assert_eq!(chunks[0].0, "input");
    assert_eq!(chunks[0].1, 0);
    assert!(chunks[0].2.contains("[REDACTED]"));
    assert!(!chunks[0].2.contains("sk-"));
    assert_eq!(
        &chunks[1..],
        &[
            (String::from("content"), 1, String::from("Hel")),
            (String::from("content"), 2, String::from("lo")),
            (String::from("reasoning"), 3, String::from("think")),
            (String::from("tool_arguments"), 4, String::from(r#"{"q""#)),
            (String::from("tool_arguments"), 5, String::from(r#":"x"}"#)),
        ]
    );
    let (request_metadata_json, raw_input, raw_output): (String, Option<String>, Option<String>) =
        connection
            .query_row(
                "SELECT request_metadata_json, raw_input, raw_output FROM evidence_attempts",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("raw evidence attempt should exist");
    assert!(request_metadata_json.contains("request_header_authorization"));
    assert!(request_metadata_json.contains("[REDACTED]"));
    assert!(!request_metadata_json.contains("downstream-secret"));
    assert!(
        !raw_input
            .as_deref()
            .unwrap_or_default()
            .contains("sk-live-secret")
    );
    assert!(!raw_input.as_deref().unwrap_or_default().contains("qb"));
    assert!(
        raw_input
            .as_deref()
            .unwrap_or_default()
            .contains("[REDACTED]")
    );
    assert_eq!(raw_output.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn evidence_raw_capture_preserves_loop_rejected_primary_reasoning() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = true

[evidence.shadow]
enabled = false

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    let raw_reasoning: Option<String> = connection
        .query_row(
            "SELECT raw_reasoning FROM evidence_attempts WHERE role = 'primary'",
            [],
            |row| row.get(0),
        )
        .expect("primary raw reasoning should query");
    let raw_reasoning = raw_reasoning.expect("looped primary should keep raw reasoning");
    assert!(raw_reasoning.contains("reasoning loop line"));
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_chunks c \
             JOIN evidence_attempts a ON a.attempt_id = c.attempt_id \
             WHERE a.role = 'primary' AND c.channel = 'reasoning'",
        ),
        3
    );
}

#[tokio::test]
async fn evidence_shadow_raw_capture_records_stream_channels_and_redacts_secrets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = true
include_request_headers = true

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 2000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-shadow-raw-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer tiny-header")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Bearer tiny-token sk-t"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let upstream_requests = recv_n_upstream_requests(&mut fake, 3).await;
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| !body_contains_retry_hint(&request.body))
            .count(),
        2
    );
    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        1,
    )
    .await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_shadow_raw_attempt_redacts_and_preserves_stream_payloads(&connection);
    assert_shadow_raw_chunks_redacted(&connection);
}

#[tokio::test]
async fn paired_shadow_sample_zero_records_no_shadow_attempts() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 2000

[evidence.shadow.paired_comparison]
enabled = true
variants = ["max-thinking", "no-thinking"]
include_raw_input = true
include_raw_output = true
sample_rate = 0.0
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _primary = fake.recv_next().await;
    assert!(
        fake.recv_within(Duration::from_millis(150)).await.is_none(),
        "sample_rate=0 must not issue paired shadow attempts"
    );

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        0
    );
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM evidence_raw_artifacts"),
        0
    );
}

#[tokio::test]
async fn paired_shadow_records_raw_input_output_without_changing_client_response() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = false
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 2000

[evidence.shadow.paired_comparison]
enabled = true
variants = ["max-thinking", "no-thinking"]
include_raw_input = true
include_raw_output = true
include_raw_reasoning = false
sample_rate = 1.0
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=paired-shadow",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let upstream_requests = recv_n_upstream_requests(&mut fake, 3).await;
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| body_thinking_budget(&request.body) == Some(0))
            .count(),
        1
    );
    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        2,
    )
    .await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'primary' AND shown_to_downstream = 1",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued' AND shown_to_downstream = 0",
        ),
        2
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_raw_artifacts WHERE variant_name IN ('max-thinking', 'no-thinking') AND artifact_kind IN ('input', 'output') AND content_text IS NOT NULL",
        ),
        4
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_raw_artifacts WHERE artifact_kind = 'reasoning'",
        ),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued' AND raw_reasoning IS NOT NULL",
        ),
        0
    );
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn shadow_and_paired_comparisons_do_not_reexpand_thinking_caps() {
    let mut fake = FakeUpstream::spawn().await;
    let profile_config = shadow_param_override_profile_config(&fake.base_url);
    let proxy = ProxyFixture::spawn_with_full_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        &profile_config,
        "",
        shadow_and_paired_comparison_config(),
    )
    .await;

    let loop_response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-shadow-raw-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(loop_response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(loop_response).await["choices"][0]["message"]["content"],
        "Hello"
    );
    let mut upstream_requests = recv_n_upstream_requests(&mut fake, 3).await;

    let paired_response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=paired-shadow",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"paired"}],"max_tokens":64}"#,
        )
        .send()
        .await
        .expect("paired proxy request should complete");
    assert_eq!(paired_response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(paired_response).await["choices"][0]["message"]["content"],
        "Hello"
    );
    upstream_requests.extend(recv_n_upstream_requests(&mut fake, 3).await);

    assert_shadow_param_override_bodies(&upstream_requests);

    wait_for_evidence_role_status_count(
        &proxy.evidence_sqlite_path,
        "shadow_continued",
        "accepted",
        3,
    )
    .await;
    assert_shadow_param_override_metadata(&proxy.evidence_sqlite_path);
}

#[cfg(feature = "param-override")]
fn shadow_param_override_profile_config(base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "shadow-param-override"
base_url = "{base_url}"
match_models = ["test-chat"]

[upstreams.thinking]
mode = "bounded_thinking"
max_tokens = 4096
default_injection_schema = "vllm_native"

[upstreams.param_override]
enabled = true
temperature = 0.6
max_tokens = 50000
"#,
    )
}

#[cfg(feature = "param-override")]
fn shadow_and_paired_comparison_config() -> &'static str {
    r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = false
compare_attempts = ["no-thinking"]
max_shadow_attempts_per_request = 4
max_global_shadow_in_flight = 4
shadow_attempt_timeout_ms = 2000

[evidence.shadow.paired_comparison]
enabled = true
variants = ["max-thinking", "no-thinking"]
include_raw_input = true
include_raw_output = true
sample_rate = 1.0

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 2
anti_loop_hint_enabled = true
"#
}

#[cfg(feature = "param-override")]
fn assert_shadow_param_override_bodies(upstream_requests: &[ObservedRequest]) {
    let bodies = upstream_requests.iter().map(|request| {
        serde_json::from_slice::<serde_json::Value>(&request.body)
            .expect("shadow request body should be JSON")
    });
    assert!(
        bodies
            .into_iter()
            .all(|body| body["max_tokens"] == 4_096 && body["temperature"] == 0.6)
    );
    assert_eq!(
        upstream_requests
            .iter()
            .filter(|request| body_thinking_budget(&request.body) == Some(0))
            .count(),
        2
    );
}

#[cfg(feature = "param-override")]
fn assert_shadow_param_override_metadata(sqlite_path: &Path) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT request_metadata_json FROM evidence_attempts \
             WHERE role = 'shadow_continued' ORDER BY rowid",
        )
        .expect("shadow metadata query should prepare");
    let metadata = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("shadow metadata query should execute")
        .map(|row| {
            serde_json::from_str::<serde_json::Value>(
                &row.expect("shadow metadata row should decode"),
            )
            .expect("shadow metadata should be JSON")
        })
        .collect::<Vec<_>>();
    assert_eq!(metadata.len(), 3);
    for metadata in metadata {
        assert_eq!(metadata["attempt_thinking_max_tokens"], "4096");
        assert_eq!(metadata["thinking_answer_budget_final_max_tokens"], "4096");
    }
}

fn assert_shadow_raw_attempt_redacts_and_preserves_stream_payloads(connection: &Connection) {
    let (request_metadata_json, raw_input, raw_output, raw_reasoning, raw_tool_calls): (
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = connection
        .query_row(
            "SELECT request_metadata_json, raw_input, raw_output, raw_reasoning, raw_tool_calls \
             FROM evidence_attempts WHERE role = 'shadow_continued'",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("shadow raw evidence attempt should exist");
    assert!(request_metadata_json.contains("request_header_authorization"));
    assert!(request_metadata_json.contains("[REDACTED]"));
    assert!(!request_metadata_json.contains("tiny-header"));
    let raw_input = raw_input.expect("shadow raw input should be captured");
    assert!(raw_input.contains("[REDACTED]"));
    assert!(!raw_input.contains("tiny-token"));
    assert!(!raw_input.contains("sk-t"));
    assert_eq!(raw_output.as_deref(), Some("Hello"));
    assert_eq!(raw_reasoning.as_deref(), Some("think"));
    let tool_calls: serde_json::Value = serde_json::from_str(
        raw_tool_calls
            .as_deref()
            .expect("shadow raw tool calls should be captured"),
    )
    .expect("shadow raw tool calls should be JSON");
    assert_eq!(tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(tool_calls[0]["function"]["arguments"], r#"{"q":"x"}"#);
}

fn assert_shadow_raw_chunks_redacted(connection: &Connection) {
    let shadow_chunks = read_evidence_chunks_for_role(connection, "shadow_continued");
    assert_eq!(shadow_chunks.len(), 6);
    assert_eq!(shadow_chunks[0].0, "input");
    assert_eq!(shadow_chunks[0].1, 0);
    assert!(shadow_chunks[0].2.contains("[REDACTED]"));
    assert!(!shadow_chunks[0].2.contains("tiny-token"));
    assert!(!shadow_chunks[0].2.contains("sk-t"));
    assert_eq!(
        &shadow_chunks[1..],
        &[
            (String::from("content"), 1, String::from("Hel")),
            (String::from("content"), 2, String::from("lo")),
            (String::from("reasoning"), 3, String::from("think")),
            (String::from("tool_arguments"), 4, String::from(r#"{"q""#)),
            (String::from("tool_arguments"), 5, String::from(r#":"x"}"#)),
        ]
    );
}

#[tokio::test]
async fn evidence_shadow_skeleton_records_skipped_shadow_without_affecting_fallback() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 0
shadow_attempt_timeout_ms = 10

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2].role, "shadow_continued");
    assert_eq!(rows[2].shown_to_downstream, 0);
    assert_eq!(rows[2].status, "skipped");
    assert_eq!(rows[2].shadow_skip_reason.as_deref(), Some("global_limit"));
}

#[tokio::test]
async fn evidence_shadow_per_request_limit_records_skip_without_extra_upstream_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 0
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 50

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "per-request shadow limit should not issue a shadow upstream request"
    );

    let rows = read_evidence_attempt_rows(&proxy.evidence_sqlite_path);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2].role, "shadow_continued");
    assert_eq!(rows[2].status, "skipped");
    assert_eq!(
        rows[2].shadow_skip_reason.as_deref(),
        Some("per_request_limit")
    );
}

#[tokio::test]
async fn evidence_shadow_timeout_releases_global_permit_for_next_request() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 20

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    send_shadow_timeout_request(&proxy, 1).await;
    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;

    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        0
    );
}

#[tokio::test]
async fn evidence_shadow_global_limit_skips_concurrent_request_and_releases_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 2000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
"#,
    )
    .await;

    let first_client = proxy.client.clone();
    let first_base_url = proxy.base_url.clone();
    let first_request = tokio::spawn(async move {
        send_shadow_timeout_request_parts(&first_client, &first_base_url, 1).await;
    });
    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_n_upstream_requests(&mut fake, 2).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "global shadow limit should skip the concurrent shadow request"
    );
    first_request
        .await
        .expect("first concurrent shadow request task should finish");

    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;
    let connection = Connection::open(&proxy.evidence_sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        1
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        1
    );

    send_shadow_timeout_request(&proxy, 3).await;
    let third_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        third_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;
}

#[tokio::test]
async fn evidence_shadow_downstream_drop_records_terminal_status_and_releases_global_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_evidence_config(
        &fake.base_url,
        r#"
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 1
shadow_attempt_timeout_ms = 20

[heartbeat]
mode = "json-whitespace"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 5
anti_loop_hint_enabled = true
downstream_drop_policy = "detach"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-shadow-timeout-then-success&id=drop",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should start");
    assert_eq!(response.status(), StatusCode::OK);
    let mut downstream = response.bytes_stream();
    let heartbeat = next_chunk(
        &mut downstream,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "drop test shielded heartbeat",
    )
    .await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(downstream);

    let first_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        first_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 1).await;
    assert_shadow_timeout_count_stays(&proxy.evidence_sqlite_path, 1).await;

    send_shadow_timeout_request(&proxy, 2).await;
    let second_requests = recv_shadow_timeout_upstream_requests(&mut fake).await;
    assert_eq!(
        second_requests
            .iter()
            .filter(|request| body_contains_retry_hint(&request.body))
            .count(),
        1
    );
    wait_for_evidence_status_count(&proxy.evidence_sqlite_path, "shadow_timeout", 2).await;

    assert_shadow_timeout_summary(&proxy.evidence_sqlite_path, 2, 2);
}

#[tokio::test]
async fn retry_ladder_advances_from_max_thinking_loop_to_bounded_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
anti_loop_hint = "Previous attempt became repetitive. Answer directly."

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(!aggregated.to_string().contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert!(!body_contains_retry_hint(&first_attempt.body));
    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    assert!(second_body_text.contains("Previous attempt became repetitive. Answer directly."));
    assert!(!second_body_text.contains("reasoning loop line"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].response_metadata["attempt_name"],
        "max-thinking"
    );
    assert_eq!(attempts[0].response_metadata["attempt_index"], "0");
    assert_eq!(
        attempts[0].response_metadata["attempt_thinking_mode"],
        "force_thinking"
    );
    assert_eq!(
        attempts[0].response_metadata["attempt_thinking_budget_tokens"],
        "32768"
    );
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(
        attempts[1].response_metadata["attempt_name"],
        "bounded-thinking"
    );
    assert_eq!(attempts[1].response_metadata["attempt_index"], "1");
    assert_eq!(
        attempts[1].response_metadata["retry_previous_reason"],
        "previous_loop_detected"
    );
    assert_eq!(
        attempts[1].response_metadata["attempt_thinking_budget_tokens"],
        "8192"
    );
}

#[tokio::test]
async fn truncate_cot_then_answer_uses_private_pre_loop_reasoning_without_downstream_exposure() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "truncate_cot_then_answer"
output_repeated_line_threshold = 4

[retry]
max_attempts = 2
anti_loop_hint_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-salvage"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(!aggregated.to_string().contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(0));
    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    assert!(second_body_text.contains("llm-guard-proxy CoT salvage retry hint"));
    assert!(second_body_text.contains("Private bounded pre-loop reasoning notes"));
    assert!(second_body_text.contains("reasoning loop line"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(
        attempts[0].response_metadata["loop_failure_policy"],
        "truncate_cot_then_answer"
    );
    assert_eq!(
        attempts[0].response_metadata["loop_hard_abort_candidate"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["loop_abort_channel"],
        "reasoning"
    );
    assert_eq!(
        attempts[0].response_metadata["loop_abort_candidate_count"],
        "1"
    );
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["cot_salvage_policy"],
        "truncate_cot_then_answer"
    );
    assert_eq!(
        attempts[1].response_metadata["attempt_thinking_budget_tokens"],
        "0"
    );
    assert_eq!(
        attempts[1].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
    assert_eq!(attempts[1].response_metadata["finish_reason"], "stop");
}

#[tokio::test]
async fn retry_ladder_advances_to_no_thinking_after_two_loop_rejections() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));
    assert!(!body_contains_retry_hint(&second_attempt.body));
    assert!(!body_contains_retry_hint(&third_attempt.body));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(
        attempts[0].response_metadata["attempt_name"],
        "max-thinking"
    );
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[1].response_metadata["attempt_name"],
        "bounded-thinking"
    );
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(
        attempts[2].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
}

#[tokio::test]
#[cfg(feature = "param-override")]
async fn vllm_native_retry_ladder_uses_each_budget_and_original_answer_headroom() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{base_url}"
match_models = ["test-chat"]

[upstreams.thinking]
mode = "bounded_thinking"
max_tokens = 50000
thinking_token_budget = 32768
default_injection_schema = "vllm_native"

[upstreams.param_override]
enabled = true
temperature = 0.6
max_tokens = 50000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking-deep"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 16384

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 1024
"#,
            base_url = fake.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-three-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":10000,"max_completion_tokens":200,"max_output_tokens":18446744073709551615}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        shielded_final_json(response).await["choices"][0]["message"]["content"],
        "Hello"
    );

    let mut observed_bodies = Vec::new();
    for _ in 0..4 {
        let observed = fake.recv_next().await;
        observed_bodies.push(
            serde_json::from_slice::<serde_json::Value>(&observed.body)
                .expect("retry body should be JSON"),
        );
    }
    assert_vllm_native_retry_bodies(&observed_bodies);

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_vllm_native_retry_metadata(&attempts, &observed_bodies);
}

#[cfg(feature = "param-override")]
fn assert_vllm_native_retry_bodies(observed_bodies: &[serde_json::Value]) {
    assert_eq!(observed_bodies.len(), 4);
    let budgets = [32_768_u64, 16_384, 8_192];
    let expected_max_tokens = [42_768_u64, 26_384, 18_192];
    for ((body, budget), expected_max_tokens) in observed_bodies
        .iter()
        .take(3)
        .zip(budgets)
        .zip(expected_max_tokens)
    {
        assert_eq!(body["thinking_token_budget"], budget);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(body["max_tokens"], expected_max_tokens);
        assert_eq!(body["max_completion_tokens"], 200 + budget);
        assert_eq!(body["max_output_tokens"], 50_000);
        assert_eq!(body["temperature"], 0.6);
        assert!(body.get("thinking").is_none());
    }
    let no_thinking = &observed_bodies[3];
    assert_eq!(no_thinking["thinking_token_budget"], 0);
    assert_eq!(
        no_thinking["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(no_thinking["max_tokens"], 1_024);
    assert_eq!(no_thinking["max_completion_tokens"], 200);
    assert_eq!(no_thinking["max_output_tokens"], 1_024);
    assert_eq!(no_thinking["temperature"], 0.6);
}

#[cfg(feature = "param-override")]
fn assert_vllm_native_retry_metadata(
    attempts: &[AttemptChainRow],
    observed_bodies: &[serde_json::Value],
) {
    assert_eq!(attempts.len(), 4);
    let budgets = [32_768_u64, 16_384, 8_192];
    for ((attempt, body), budget) in attempts.iter().zip(observed_bodies).take(3).zip(budgets) {
        assert_eq!(
            attempt.response_metadata["attempt_thinking_budget_tokens"],
            budget.to_string()
        );
        assert_eq!(
            attempt.response_metadata["thinking_default_injection_schema"],
            "vllm_native"
        );
        assert_eq!(
            attempt.response_metadata["thinking_schema_path"],
            "thinking_token_budget"
        );
        assert_eq!(
            attempt.response_metadata["thinking_budget_final_tokens"],
            budget.to_string()
        );
        assert_eq!(
            attempt.response_metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens,max_completion_tokens,max_output_tokens"
        );
        assert_eq!(
            attempt.response_metadata["thinking_policy_max_tokens"],
            "50000"
        );
        assert_eq!(
            attempt.response_metadata["attempt_thinking_max_tokens"],
            "50000"
        );
        assert_eq!(
            attempt.response_metadata["thinking_answer_budget_overflow_fields"],
            "max_output_tokens"
        );
        assert_eq!(
            attempt.response_metadata["thinking_answer_budget_final_max_tokens"],
            body["max_tokens"]
                .as_u64()
                .expect("max_tokens should be numeric")
                .to_string()
        );
        assert_eq!(
            attempt.response_metadata["thinking_answer_budget_final_max_completion_tokens"],
            body["max_completion_tokens"]
                .as_u64()
                .expect("max_completion_tokens should be numeric")
                .to_string()
        );
        assert_eq!(
            attempt.response_metadata["thinking_answer_budget_final_max_output_tokens"],
            body["max_output_tokens"]
                .as_u64()
                .expect("max_output_tokens should be numeric")
                .to_string()
        );
    }
    assert_eq!(
        attempts[3].response_metadata["attempt_thinking_budget_tokens"],
        "0"
    );
    assert_eq!(
        attempts[3].response_metadata["thinking_budget_final_tokens"],
        "0"
    );
    assert_eq!(
        attempts[3].response_metadata["thinking_answer_budget_adjusted_fields"],
        "max_tokens,max_output_tokens"
    );
    assert_eq!(
        attempts[3].response_metadata["attempt_thinking_max_tokens"],
        "1024"
    );
    assert_eq!(
        attempts[3].response_metadata["thinking_answer_budget_final_max_tokens"],
        observed_bodies[3]["max_tokens"].to_string()
    );
    assert_eq!(
        attempts[3].response_metadata["thinking_answer_budget_final_max_output_tokens"],
        observed_bodies[3]["max_output_tokens"].to_string()
    );
}

#[tokio::test]
async fn bounded_cot_salvage_falls_back_to_no_thinking_after_salvage_loop() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "bounded_answer_from_cot"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(1_024));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));
    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    let third_body_text = String::from_utf8_lossy(&third_attempt.body);
    assert!(second_body_text.contains("llm-guard-proxy CoT salvage retry hint"));
    assert!(!third_body_text.contains("llm-guard-proxy CoT salvage retry hint"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(
        attempts[1].response_metadata["attempt_thinking_budget_tokens"],
        "1024"
    );
    assert_eq!(attempts[2].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(attempts[2].response_metadata["cot_salvage_used"], "false");
    assert_eq!(
        attempts[2].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
}

#[tokio::test]
async fn bounded_cot_salvage_is_one_shot_before_final_no_thinking_direct_relay() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
on_reasoning_loop = "bounded_answer_from_cot"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
anti_loop_hint_enabled = true
shielded_streaming_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking-deep"
thinking_mode = "force_thinking"
thinking_token_budget = 16384

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("stream body should be text");
    assert!(!body.contains("event: error"));
    assert!(!body.contains("llm_guard_loop_retry_exhausted"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    let fourth_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(1_024));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&fourth_attempt.body), Some(0));

    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    let third_body_text = String::from_utf8_lossy(&third_attempt.body);
    let fourth_body_text = String::from_utf8_lossy(&fourth_attempt.body);
    assert!(second_body_text.contains("llm-guard-proxy CoT salvage retry hint"));
    assert!(!third_body_text.contains("llm-guard-proxy CoT salvage retry hint"));
    assert!(!fourth_body_text.contains("llm-guard-proxy CoT salvage retry hint"));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 4);
    assert_eq!(attempts[1].response_metadata["cot_salvage_used"], "true");
    assert_eq!(attempts[2].response_metadata["cot_salvage_used"], "false");
    assert_eq!(attempts[3].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(attempts[3].response_metadata["cot_salvage_used"], "false");
    assert_eq!(
        attempts[3].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
    assert_eq!(
        attempts[3].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
}

#[tokio::test]
async fn retry_anti_loop_hint_stays_single_message_across_repeated_loop_retries() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(retry_hint_count(&first_attempt.body), 0);
    assert_eq!(retry_hint_count(&second_attempt.body), 1);
    assert_eq!(retry_hint_count(&third_attempt.body), 1);

    let second_body_text = String::from_utf8_lossy(&second_attempt.body);
    assert!(second_body_text.contains("retry_attempt=2/3"));
    let third_body_text = String::from_utf8_lossy(&third_attempt.body);
    assert!(third_body_text.contains("retry_attempt=3/3"));
    assert!(
        !third_body_text.contains("retry_attempt=2/3"),
        "retry bodies must be rebuilt from the original downstream body, not from the previous generated retry body"
    );

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].status, "succeeded");
}

#[tokio::test]
async fn shielded_retry_runs_recovery_command_after_upstream_stall_then_succeeds() {
    let mut fake = FakeUpstream::spawn().await;
    let recovery_root = unique_test_dir("stall-recovery");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let recovery_marker = recovery_root.join("recovered");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50
recovery_command = ["/usr/bin/touch", "{recovery_marker}"]
recovery_timeout_ms = 1000
recovery_cooldown_ms = 1000
recovery_budget_window_ms = 10000
recovery_max_per_window = 1
"#,
            recovery_marker = recovery_marker.display()
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(recovery_marker.exists());

    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_detected"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_status"],
        "succeeded"
    );
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "succeeded");

    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn shielded_retry_does_not_replay_when_recovery_command_fails() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50
recovery_command = ["/bin/false"]
recovery_timeout_ms = 1000
recovery_cooldown_ms = 1000
recovery_budget_window_ms = 10000
recovery_max_per_window = 1
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _body = response
        .text()
        .await
        .expect("error body should be consumed");
    let _first_attempt = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(attempts[0].retry_reason, None);
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_status"],
        "exit_failure"
    );
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_recovery_permits_retry"],
        "false"
    );
}

#[tokio::test]
async fn local_recovery_restart_command() {
    assert!(!LocalRecoveryPolicy::from_config(&LocalRecoveryConfig::default()).is_configured());

    let success_policy = LocalRecoveryPolicy {
        enabled: true,
        restart_command: vec![String::from("/bin/true")],
        restart_timeout: Duration::from_secs(1),
        readiness_endpoint: String::from("/v1/chat/completions"),
        readiness_body: serde_json::json!({"messages":[{"role":"user","content":"ready"}],"max_tokens":1}),
        readiness_request_timeout: Duration::from_millis(50),
        readiness_deadline: Duration::from_millis(50),
        readiness_interval: Duration::from_millis(10),
        max_attempts_per_request: 1,
        cooldown: Duration::from_millis(1),
        budget_window: Duration::from_secs(60),
        max_per_window: 1,
    };
    let success_ran = AtomicBool::new(false);
    let success = run_local_recovery_restart_command(&success_policy, &success_ran).await;
    assert_eq!(success["local_recovery_restart_status"], "succeeded");

    let timeout_policy = LocalRecoveryPolicy {
        restart_command: vec![String::from("/bin/sleep"), String::from("30")],
        restart_timeout: Duration::from_millis(50),
        ..success_policy
    };
    let timeout_ran = AtomicBool::new(false);
    let timeout = run_local_recovery_restart_command(&timeout_policy, &timeout_ran).await;
    assert_eq!(timeout["local_recovery_restart_status"], "timeout_killed");
    assert_eq!(timeout["local_recovery_status"], "timeout_killed");
}

#[tokio::test]
async fn local_recovery_replays_original_request() {
    let mut fake = FakeUpstream::spawn().await;
    let recovery_root = unique_test_dir("local-recovery-replay");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let recovery_marker = recovery_root.join("recovered");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{recovery_marker}"]
restart_timeout_ms = 1000
readiness_body = {{"model":"test-chat","messages":[{{"role":"user","content":"local recovery ready"}}],"max_tokens":1}}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 100
cooldown_ms = 1000
budget_window_ms = 10000
max_per_window = 1
"#,
            recovery_marker = recovery_marker.display()
        ),
    )
    .await;

    let original_body = r#"{"model":"test-chat","messages":[{"role":"user","content":"original business request"}]}"#;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(original_body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(recovery_marker.exists());

    let first = fake.recv_next().await;
    let probe = fake.recv_next().await;
    let replay = fake.recv_next().await;
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert!(body_contains_text(&probe.body, "local recovery ready"));
    assert_eq!(
        replay.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert_eq!(first.body, replay.body);
    assert!(!body_contains_text(&replay.body, "local recovery ready"));
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["local_recovery_status"],
        "succeeded"
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_readiness_status"],
        "ready"
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_permits_retry"],
        "true"
    );
    assert_eq!(attempts[1].status, "succeeded");

    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn local_recovery_chat_readiness() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50

[upstream.local_recovery]
enabled = true
restart_command = ["/bin/true"]
restart_timeout_ms = 1000
readiness_body = {"model":"test-chat","messages":[{"role":"user","content":"hot-restart-never-ready"}],"max_tokens":1}
readiness_request_timeout_ms = 100
readiness_deadline_ms = 100
readiness_interval_ms = 50
cooldown_ms = 1000
budget_window_ms = 10000
max_per_window = 1
"#,
    )
    .await;

    let models = proxy
        .client
        .get(format!("{}/models", fake.base_url))
        .send()
        .await
        .expect("models request should complete");
    assert_eq!(models.status(), StatusCode::OK);
    let _models_body = models.text().await.expect("models body should read");
    let models_probe = fake.recv_next().await;
    assert_eq!(models_probe.path_and_query, "/v1/models");

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _body = response.text().await.expect("error body should be text");
    let first = fake.recv_next().await;
    let readiness = fake.recv_next().await;
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert_eq!(readiness.path_and_query, "/v1/chat/completions");
    while let Some(extra_probe) = fake.recv_within(Duration::from_millis(100)).await {
        assert_eq!(extra_probe.path_and_query, "/v1/chat/completions");
        assert!(body_contains_text(
            &extra_probe.body,
            "hot-restart-never-ready"
        ));
    }

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(
        attempts[0].response_metadata["local_recovery_status"],
        "readiness_timeout"
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_permits_retry"],
        "false"
    );
}

#[tokio::test]
async fn local_recovery_replays_after_request_deadline_recovery() {
    let mut fake = FakeUpstream::spawn().await;
    let recovery_root = unique_test_dir("local-recovery-deadline-replay");
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let recovery_marker = recovery_root.join("recovered");
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
request_deadline_ms = 100
anti_loop_hint_enabled = false

[upstream.local_recovery]
enabled = true
restart_command = ["/usr/bin/touch", "{recovery_marker}"]
restart_timeout_ms = 1000
readiness_body = {{"model":"test-chat","messages":[{{"role":"user","content":"deadline recovery ready"}}],"max_tokens":1}}
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 100
cooldown_ms = 1000
budget_window_ms = 10000
max_per_window = 1
"#,
            recovery_marker = recovery_marker.display()
        ),
    )
    .await;

    let original_body = r#"{"model":"test-chat","messages":[{"role":"user","content":"deadline business request"}]}"#;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(original_body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert!(recovery_marker.exists());

    let first = fake.recv_next().await;
    let probe = fake.recv_next().await;
    let replay = fake.recv_next().await;
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert!(body_contains_text(&probe.body, "deadline recovery ready"));
    assert_eq!(
        replay.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert_eq!(first.body, replay.body);
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some(REQUEST_DEADLINE_ABORT_REASON)
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_cause"],
        "request_deadline"
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_status"],
        "succeeded"
    );
    assert_eq!(
        attempts[0].response_metadata["local_recovery_permits_retry"],
        "true"
    );
    assert_eq!(attempts[1].status, "succeeded");

    remove_dir_all(&recovery_root);
}

#[tokio::test]
async fn local_recovery_failure_observability() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
anti_loop_hint_enabled = false

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50

[upstream.local_recovery]
enabled = true
restart_command = ["/bin/false"]
restart_timeout_ms = 1000
readiness_body = {"model":"test-chat","messages":[{"role":"user","content":"secret readiness prompt"}],"max_tokens":1}
readiness_request_timeout_ms = 100
readiness_deadline_ms = 100
readiness_interval_ms = 50
cooldown_ms = 1000
budget_window_ms = 10000
max_per_window = 1
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=stall-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer sk-local-recovery-secret")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"secret business prompt"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _body = response.text().await.expect("error body should be text");
    let _first = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    let metadata = &attempts[0].response_metadata;
    assert_eq!(metadata["local_recovery_configured"], "true");
    assert_eq!(metadata["local_recovery_status"], "exit_failure");
    assert_eq!(metadata["local_recovery_cause"], "upstream_stall");
    assert_eq!(metadata["local_recovery_profile"], "default");
    assert_eq!(metadata["local_recovery_permits_retry"], "false");
    let serialized_metadata =
        serde_json::to_string(metadata).expect("metadata should serialize for leakage check");
    assert!(!serialized_metadata.contains("secret business prompt"));
    assert!(!serialized_metadata.contains("secret readiness prompt"));
    assert!(!serialized_metadata.contains("sk-local-recovery-secret"));
}

#[tokio::test]
async fn local_recovery_profile_singleflight() {
    assert_local_recovery_coordinator_profiles_are_isolated();
    let mut fake = FakeUpstream::spawn().await;
    let recovery_root = unique_test_dir("local-recovery-singleflight");
    remove_dir_all(&recovery_root);
    fs::create_dir_all(&recovery_root).expect("recovery root should be created");
    let (script_path, count_path) = write_singleflight_restart_script(&recovery_root);

    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 3
anti_loop_hint_enabled = false

[upstream.local_recovery]
enabled = true
restart_command = ["{script_path}"]
restart_timeout_ms = 1000
readiness_request_timeout_ms = 1000
readiness_deadline_ms = 1000
readiness_interval_ms = 100
cooldown_ms = 1
budget_window_ms = 10000
max_per_window = 1
"#,
            script_path = script_path.display()
        ),
    )
    .await;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-concurrent",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"one"}]}"#)
        .send();
    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-concurrent",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"two"}]}"#)
        .send();

    let (first, second) = tokio::join!(first, second);
    let first = first.expect("first request should complete");
    let second = second.expect("second request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let _first_body = first.text().await.expect("first body should read");
    let _second_body = second.text().await.expect("second body should read");

    assert_singleflight_upstream_requests(&mut fake).await;

    let count = fs::read_to_string(&count_path).expect("restart count should be readable");
    assert_eq!(count.lines().count(), 1);

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let recovery_statuses = attempts
        .iter()
        .filter_map(|attempt| attempt.response_metadata.get("local_recovery_status"))
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();
    assert!(recovery_statuses.contains(&"succeeded"));
    assert!(recovery_statuses.contains(&"joined_inflight"));

    remove_dir_all(&recovery_root);
}

fn assert_local_recovery_coordinator_profiles_are_isolated() {
    let coordinators = LocalRecoveryCoordinatorSet::default();
    let first_default = coordinators.coordinator_for("default");
    let second_default = coordinators.coordinator_for("default");
    let other_profile = coordinators.coordinator_for("other");
    assert!(Arc::ptr_eq(&first_default, &second_default));
    assert!(!Arc::ptr_eq(&first_default, &other_profile));
}

fn write_singleflight_restart_script(recovery_root: &Path) -> (PathBuf, PathBuf) {
    let count_path = recovery_root.join("count.txt");
    let script_path = recovery_root.join("restart-once.sh");
    fs::write(
        &script_path,
        format!(
            "#!/bin/sh\necho run >> {}\nsleep 0.2\n",
            count_path.display()
        ),
    )
    .expect("restart script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("restart script should be executable");
    (script_path, count_path)
}

async fn assert_singleflight_upstream_requests(fake: &mut FakeUpstream) {
    let mut observed = Vec::new();
    for _ in 0..5 {
        observed.push(fake.recv_next().await);
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
    let readiness_count = observed
        .iter()
        .filter(|request| {
            request.path_and_query == "/v1/chat/completions"
                && request
                    .headers
                    .get("x-llm-guard-proxy-probe")
                    .is_some_and(|value| value == "local-recovery")
        })
        .count();
    let business_count = observed
        .iter()
        .filter(|request| {
            request.path_and_query == "/v1/chat/completions?test=hot-restart-concurrent"
        })
        .count();
    assert_eq!(readiness_count, 1);
    assert_eq!(business_count, 4);
}

#[tokio::test]
async fn upstream_stall_recovery_is_single_flight_and_budget_limited() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        first_chunk_timeout: Duration::from_millis(50),
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/sleep"), String::from("0.2")],
        recovery_timeout: Duration::from_secs(2),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 1,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let first_recovery = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { run_upstream_stall_recovery(&policy, &coordinator).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let joined = run_upstream_stall_recovery(&policy, &coordinator).await;
    let first = first_recovery
        .await
        .expect("first recovery task should join");

    assert_eq!(first["upstream_stall_recovery_status"], "succeeded");
    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");

    tokio::time::sleep(Duration::from_millis(5)).await;
    let budget_limited = run_upstream_stall_recovery(&policy, &coordinator).await;
    assert_eq!(
        budget_limited["upstream_stall_recovery_status"],
        "skipped_budget_exhausted"
    );
    assert_eq!(budget_limited["upstream_stall_recovery_budget_runs"], "1");
}

#[tokio::test]
async fn upstream_stall_recovery_joiners_do_not_hang_after_leader_cancellation() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        first_chunk_timeout: Duration::from_millis(50),
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/sleep"), String::from("0.2")],
        recovery_timeout: Duration::from_secs(2),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    let leader = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { run_upstream_stall_recovery(&policy, &coordinator).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    leader.abort();
    assert!(
        leader
            .await
            .expect_err("leader should be cancelled")
            .is_cancelled()
    );

    let joined = timeout(
        Duration::from_millis(500),
        run_upstream_stall_recovery(&policy, &coordinator),
    )
    .await
    .expect("later stall recovery should not wait forever after leader cancellation");

    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");
}

#[tokio::test]
async fn upstream_stall_recovery_joiner_uses_completed_state_after_lost_notification() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        first_chunk_timeout: Duration::from_millis(50),
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![String::from("/bin/true")],
        recovery_timeout: Duration::from_millis(1),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());

    {
        let mut state = coordinator.state.lock().await;
        state.running = true;
    }
    let joined = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        let policy = policy.clone();
        async move { wait_for_upstream_stall_recovery_result(&policy, &coordinator, true).await }
    });
    sleep(Duration::from_millis(50)).await;
    {
        let mut state = coordinator.state.lock().await;
        state.running = false;
        state.last_finished = Some(Instant::now());
        state.last_result = Some(BTreeMap::from([
            (
                String::from("upstream_stall_recovery_configured"),
                String::from("true"),
            ),
            (
                String::from("upstream_stall_recovery_status"),
                String::from("succeeded"),
            ),
        ]));
    }

    let joined = timeout(Duration::from_millis(1_500), joined)
        .await
        .expect("lost notification simulation should not hang until the test timeout")
        .expect("joiner task should complete");

    assert_eq!(joined["upstream_stall_recovery_status"], "joined_inflight");
    assert_eq!(joined["upstream_stall_recovery_joined_status"], "succeeded");
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn upstream_stall_recovery_public_path_returns_term_resistant_leader_cleanup_result() {
    let test_dir = unique_test_dir("public recovery term-resistant leader");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let leader_pid_path = test_dir.join("leader.pid");
    let ready_path = test_dir.join("leader.ready");
    let script_path = test_dir.join("term-resistant-leader.sh");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\ntrap '' TERM\nprintf '%s\\n' \"$$\" > \"$1\"\n: > \"$2\"\nwhile :; do sleep 1; done\n",
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let policy = UpstreamStallPolicy {
        enabled: true,
        first_chunk_timeout: Duration::from_millis(50),
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![
            String::from("/bin/sh"),
            script_path.display().to_string(),
            leader_pid_path.display().to_string(),
            ready_path.display().to_string(),
        ],
        // The public waiter must observe a real leader before the timeout path
        // tears it down; cleanup behavior is independent of a one-millisecond race.
        recovery_timeout: Duration::from_secs(1),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let recovery = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        async move { run_upstream_stall_recovery(&policy, &coordinator).await }
    });
    let leader = read_pid_file_after_ready(&leader_pid_path, &ready_path).await;
    let metadata = recovery.await.expect("public recovery task should join");

    // The legacy early public return leaves the background cleanup running.
    // Let that bounded cleanup finish before asserting so this RED test cannot leak its leader.
    sleep(Duration::from_millis(1_250)).await;
    assert_process_reaped(leader).await;
    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_term_sent"],
        "true"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_kill_sent"],
        "true"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_status"],
        "terminated_after_kill"
    );
    remove_dir_all(&test_dir);
}

#[cfg(unix)]
#[tokio::test]
async fn upstream_stall_recovery_command_wiring_times_out_and_cleans_process_group() {
    let policy = UpstreamStallPolicy {
        enabled: true,
        first_chunk_timeout: Duration::from_millis(50),
        idle_timeout: Duration::from_millis(50),
        recovery_command: vec![
            String::from("/bin/sh"),
            String::from("-c"),
            String::from("while :; do sleep 1; done"),
        ],
        recovery_timeout: Duration::from_millis(1),
        recovery_cooldown: Duration::from_millis(1),
        recovery_budget_window: Duration::from_secs(60),
        recovery_max_per_window: 2,
    };

    let metadata = timeout(
        Duration::from_secs(2),
        run_upstream_stall_recovery_command(&policy),
    )
    .await
    .expect("production recovery command cleanup should complete within its bounded grace");

    assert_eq!(metadata["upstream_stall_recovery_ran"], "true");
    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_term_sent"],
        "true"
    );
    assert_ne!(
        metadata["upstream_stall_recovery_timeout_cleanup_status"],
        "wait_timeout_after_kill"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_scope"],
        "process_group"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn recovery_fixture_helpers_are_raii_bounded() {
    let fixture = RecoveryProcessFixture::spawn_shell("while :; do sleep 1; done");
    drop(fixture);
}

#[cfg(unix)]
#[tokio::test]
async fn recovery_process_group_signal_uses_typed_in_process_api() {
    let fixture = RecoveryProcessFixture::spawn_shell("while :; do sleep 1; done");

    assert!(send_recovery_process_group_signal(
        fixture.process_group_id,
        Signal::SIGTERM,
    ));
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn process_cleanup_refuses_changed_start_time_identity() {
    let fixture = RecoveryProcessFixture::spawn_shell("while :; do sleep 1; done");
    let identity = LinuxProcessIdentity::capture(fixture.process_group_id)
        .expect("fixture process identity should be readable");
    let stale_identity = LinuxProcessIdentity {
        start_time_ticks: identity.start_time_ticks ^ 1,
        ..identity
    };

    kill_process_if_running(stale_identity);

    assert!(identity.is_running());
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn recovery_wait_abort_after_readiness_reaps_group_leader() {
    let test_dir = unique_test_dir("recovery abort leader's path");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let leader_pid_path = test_dir.join("leader.pid");
    let ready_path = test_dir.join("leader.ready");
    let script_path = test_dir.join("leader.sh");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$$\" > \"$1\"\n: > \"$2\"\nwhile :; do sleep 1; done\n",
    )
    .expect("test recovery script should be written");

    let fixture = RecoveryProcessFixture::spawn_with_args(
        &script_path,
        &[leader_pid_path.as_path(), ready_path.as_path()],
    );
    let leader = read_pid_file_after_ready(&leader_pid_path, &ready_path).await;
    assert_eq!(leader.pid, fixture.process_group_id);
    let waiting = tokio::spawn(fixture.wait_with_timeout(Duration::from_secs(30)));
    tokio::task::yield_now().await;

    waiting.abort();
    assert!(
        waiting
            .await
            .expect_err("aborted recovery wait should be cancelled")
            .is_cancelled()
    );

    assert_process_reaped(leader).await;
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn recovery_wait_abort_after_readiness_kills_descendant() {
    let test_dir = unique_test_dir("recovery abort descendant's path");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let descendant_pid_path = test_dir.join("descendant.pid");
    let ready_path = test_dir.join("descendant.ready");
    let script_path = test_dir.join("descendant.sh");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\nsleep 30 &\ndescendant_pid=$!\nprintf '%s\\n' \"$descendant_pid\" > \"$1\"\n: > \"$2\"\nwait \"$descendant_pid\"\n",
    )
    .expect("test recovery script should be written");

    let fixture = RecoveryProcessFixture::spawn_with_args(
        &script_path,
        &[descendant_pid_path.as_path(), ready_path.as_path()],
    );
    let leader = LinuxProcessIdentity::capture(fixture.process_group_id)
        .expect("fixture process identity should be readable");
    let descendant = read_pid_file_after_ready(&descendant_pid_path, &ready_path).await;
    let waiting = tokio::spawn(fixture.wait_with_timeout(Duration::from_secs(30)));
    tokio::task::yield_now().await;

    waiting.abort();
    assert!(
        waiting
            .await
            .expect_err("aborted recovery wait should be cancelled")
            .is_cancelled()
    );

    assert_process_reaped(leader).await;
    assert_process_not_running(descendant).await;
}

#[cfg(unix)]
struct RecoveryProcessFixture {
    child: RecoveryProcessGuard,
    process_group_id: u32,
}

#[cfg(unix)]
struct TestDirectoryCleanup(PathBuf);

#[cfg(unix)]
impl TestDirectoryCleanup {
    fn new(path: &Path) -> Self {
        Self(path.to_path_buf())
    }
}

#[cfg(unix)]
impl Drop for TestDirectoryCleanup {
    fn drop(&mut self) {
        remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
impl RecoveryProcessFixture {
    fn spawn_with_args(program: &Path, args: &[&Path]) -> Self {
        let mut command = Command::new("/bin/sh");
        command.arg(program).args(args);
        Self::spawn_command(&mut command)
    }

    fn spawn_shell(script: &str) -> Self {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", script]);
        Self::spawn_command(&mut command)
    }

    fn spawn_command(command: &mut Command) -> Self {
        command
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_recovery_command(command);
        let child = command.spawn().expect("fixture child should spawn");
        let process_group_id = child.id().expect("fixture child should have a PID");
        Self {
            child: RecoveryProcessGuard::new(child),
            process_group_id,
        }
    }

    async fn wait_with_timeout(mut self, recovery_timeout: Duration) -> BTreeMap<String, String> {
        wait_for_recovery_child_with_timeout(&mut self.child, recovery_timeout).await
    }
}

#[cfg(target_os = "linux")]
async fn assert_recovery_process_group_cleanup(
    metadata: &BTreeMap<String, String>,
    leader: LinuxProcessIdentity,
    descendant: LinuxProcessIdentity,
) {
    assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_scope"],
        "process_group"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_term_sent"],
        "true"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_kill_sent"],
        "true"
    );
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_cleanup_status"],
        "terminated_after_kill"
    );
    assert_process_reaped(leader).await;
    assert_process_not_running(descendant).await;
}

#[cfg(unix)]
#[tokio::test]
async fn recovery_child_timeout_cleanup_can_be_exercised_after_fixture_readiness() {
    for _ in 0..8 {
        let fixture = RecoveryProcessFixture::spawn_shell("while :; do sleep 1; done");
        let metadata = fixture.wait_with_timeout(Duration::from_millis(1)).await;

        assert_eq!(metadata["upstream_stall_recovery_status"], "timeout_killed");
        assert_eq!(
            metadata["upstream_stall_recovery_timeout_cleanup_scope"],
            "process_group"
        );
    }
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_descendant_process_group() {
    let test_dir = unique_test_dir("recovery descendant's process group");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let child_pid_path = test_dir.join("child.pid");
    let ready_path = test_dir.join("child.ready");
    let script_path = test_dir.join("spawn-descendant.sh");
    // Publish the descendant PID and ready marker before the long sleep so
    // full-suite scheduler delay cannot race the recovery timeout.
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\nsleep 30 &\nchild_pid=$!\nprintf '%s\\n' \"$child_pid\" > \"$1\"\n: > \"$2\"\nsleep 30\n",
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let fixture = RecoveryProcessFixture::spawn_with_args(
        &script_path,
        &[child_pid_path.as_path(), ready_path.as_path()],
    );
    let child_pid = read_pid_file_after_ready(&child_pid_path, &ready_path).await;
    let leader = LinuxProcessIdentity::capture(fixture.process_group_id)
        .expect("fixture leader identity should be readable");
    let metadata = fixture.wait_with_timeout(Duration::from_millis(1)).await;

    assert_recovery_process_group_cleanup(&metadata, leader, child_pid).await;
    remove_dir_all(&test_dir);
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_term_resistant_descendant_process_group() {
    let test_dir = unique_test_dir("recovery term-resistant descendant's path");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let child_pid_path = test_dir.join("child.pid");
    let ready_path = test_dir.join("child.ready");
    let script_path = test_dir.join("spawn-term-resistant-descendant.sh");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\ntrap 'exit 0' TERM\nsh -c 'trap \"\" TERM; printf \"%s\\n\" \"$$\" > \"$1\"; : > \"$2\"; while :; do sleep 1; done' _ \"$1\" \"$2\" &\n# Parent waits for the descendant readiness handshake before sleeping so suite\n# load cannot kill the group before the PID under test exists.\nwhile [ ! -f \"$2\" ]; do sleep 0.01; done\nsleep 30\n",
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let fixture = RecoveryProcessFixture::spawn_with_args(
        &script_path,
        &[child_pid_path.as_path(), ready_path.as_path()],
    );
    let child_pid = read_pid_file_after_ready(&child_pid_path, &ready_path).await;
    let leader = LinuxProcessIdentity::capture(fixture.process_group_id)
        .expect("fixture leader identity should be readable");
    let metadata = fixture.wait_with_timeout(Duration::from_millis(1)).await;

    assert_recovery_process_group_cleanup(&metadata, leader, child_pid).await;
    assert!(
        matches!(
            metadata["upstream_stall_recovery_timeout_term_child_wait_status"].as_str(),
            "child_still_running_after_term" | "child_exited_unreaped_after_term"
        ),
        "TERM observation must retain a documented direct-leader state"
    );
    remove_dir_all(&test_dir);
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn upstream_stall_recovery_timeout_kills_term_resistant_group_leader_before_join_timeout() {
    let test_dir = unique_test_dir("recovery term-resistant leader's path");
    remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("test directory should be created");
    let _test_dir_cleanup = TestDirectoryCleanup::new(&test_dir);
    let child_pid_path = test_dir.join("child.pid");
    let ready_path = test_dir.join("child.ready");
    let script_path = test_dir.join("term-resistant-leader.sh");
    fs::write(
        &script_path,
        "#!/bin/sh\nset -eu\ntrap '' TERM\nprintf '%s\\n' \"$$\" > \"$1\"\n: > \"$2\"\nwhile :; do sleep 1; done\n",
    )
    .expect("test recovery script should be written");
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))
        .expect("test recovery script should be executable");

    let fixture = RecoveryProcessFixture::spawn_with_args(
        &script_path,
        &[child_pid_path.as_path(), ready_path.as_path()],
    );
    let child_pid = read_pid_file_after_ready(&child_pid_path, &ready_path).await;
    assert_eq!(child_pid.pid, fixture.process_group_id);
    let metadata = fixture.wait_with_timeout(Duration::from_millis(1)).await;

    assert_recovery_process_group_cleanup(&metadata, child_pid, child_pid).await;
    assert_eq!(
        metadata["upstream_stall_recovery_timeout_term_child_wait_status"],
        "child_still_running_after_term"
    );
    remove_dir_all(&test_dir);
}

#[tokio::test]
async fn shielded_retry_all_loop_attempts_returns_error_and_records_chain() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("error body should be text");
    assert!(body.contains("llm_guard_loop_retry_exhausted"));
    assert!(!body.contains("reasoning loop line"));
    for _ in 0..3 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "3");
    assert_eq!(
        request_row.response_metadata["retry_final_outcome"],
        "failed"
    );
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[1].status, "retried");
    assert_eq!(attempts[2].status, "failed");
    for attempt in &attempts {
        assert_eq!(attempt.abort_reason.as_deref(), Some("loop_guard"));
        assert_eq!(attempt.response_metadata["loop_detected"], "true");
        assert_eq!(attempt.response_metadata["attempt_max_attempts"], "3");
    }
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert!(attempts[2].retry_reason.is_none());
}

#[tokio::test]
async fn shielded_retry_policy_can_be_disabled_for_single_attempt_behavior() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
enabled = false
max_attempts = 5
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-once-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let observed = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&observed.body));
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
}

#[tokio::test]
async fn shielded_retry_transient_upstream_status_then_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=transient-503-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    let _first = fake.recv_next().await;
    let _second = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[0].response_metadata["status_code"], "503");
    assert_eq!(attempts[1].status, "succeeded");
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn hot_restart_recovers_after_transient_failure() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
probe_interval_secs = 1
probe_timeout_secs = 5

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-503-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first = fake.recv_next().await;
    let probe = fake.recv_next().await;
    let retry = fake.recv_next().await;
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=hot-restart-503-then-success"
    );
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert!(body_contains_text(&probe.body, "1+1=?"));
    assert_eq!(
        retry.path_and_query,
        "/v1/chat/completions?test=hot-restart-503-then-success"
    );
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].response_metadata["status_code"], "503");
    assert_eq!(
        attempts[0].response_metadata["hot_restart_recovery_status"],
        "ready"
    );
    assert_eq!(attempts[1].status, "succeeded");
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn disabled_local_recovery_does_not_suppress_hot_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
probe_interval_secs = 1
probe_timeout_secs = 5

[upstream.local_recovery]
enabled = false
restart_command = ["/bin/false"]

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-503-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated: serde_json::Value =
        serde_json::from_str(&response.text().await.expect("body should be text"))
            .expect("body should be JSON");
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");

    let first = fake.recv_next().await;
    let probe = fake
        .recv_within(Duration::from_millis(500))
        .await
        .expect("hot-restart probe should run when local recovery is disabled");
    let retry = fake
        .recv_within(Duration::from_millis(500))
        .await
        .expect("original request should replay after hot-restart readiness");
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=hot-restart-503-then-success"
    );
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert_eq!(
        retry.path_and_query,
        "/v1/chat/completions?test=hot-restart-503-then-success"
    );
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].response_metadata["hot_restart_recovery_status"],
        "ready"
    );
    assert!(
        attempts[0]
            .response_metadata
            .get("local_recovery_status")
            .is_none()
    );
    assert_eq!(attempts[1].status, "succeeded");
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn hot_restart_times_out() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
probe_interval_secs = 1
probe_timeout_secs = 1
probe_messages = [{"role":"user","content":"hot-restart-never-ready"}]

[retry]
max_attempts = 3
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-always-503",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let _body = response.text().await.expect("body should be text");

    let first = fake.recv_next().await;
    let probe = fake.recv_next().await;
    assert_eq!(
        first.path_and_query,
        "/v1/chat/completions?test=hot-restart-always-503"
    );
    assert_eq!(probe.path_and_query, "/v1/chat/completions");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some("hot_restart_timeout")
    );
    assert_eq!(
        attempts[0].response_metadata["hot_restart_recovery_status"],
        "timeout"
    );
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn concurrent_requests_share_single_probe() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
probe_interval_secs = 1
probe_timeout_secs = 5
probe_messages = [{"role":"user","content":"hot-restart-shared-probe"}]

[retry]
max_attempts = 3
"#,
    )
    .await;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-concurrent",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"one"}]}"#)
        .send();
    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-concurrent",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"two"}]}"#)
        .send();

    let (first, second) = tokio::join!(first, second);
    let first = first.expect("first proxy request should complete");
    let second = second.expect("second proxy request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let _first_body = first.text().await.expect("first body should be text");
    let _second_body = second.text().await.expect("second body should be text");

    let mut observed = Vec::new();
    for _ in 0..5 {
        observed.push(fake.recv_next().await);
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
    let probe_count = observed
        .iter()
        .filter(|request| {
            request.path_and_query == "/v1/chat/completions"
                && body_contains_text(&request.body, "hot-restart-shared-probe")
        })
        .count();
    let original_count = observed
        .iter()
        .filter(|request| {
            request.path_and_query == "/v1/chat/completions?test=hot-restart-concurrent"
        })
        .count();
    assert_eq!(probe_count, 1);
    assert_eq!(original_count, 4);
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn hot_restart_disabled_passes_through_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[upstream.hot_restart]
enabled = false

[retry]
max_attempts = 1
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=hot-restart-503-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _body = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=hot-restart-503-then-success"
    );
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert!(
        attempts[0]
            .response_metadata
            .get("hot_restart_recovery_status")
            .is_none()
    );
}

#[tokio::test]
async fn shielded_retry_exhausted_upstream_status_returns_structured_proxy_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=always-429",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"rate-limit"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body = response.text().await.expect("body should be text");
    assert!(body.contains("llm_guard_upstream_error"));
    assert!(!body.contains("rate-limit"));

    let _first = fake.recv_next().await;
    let _second = fake.recv_next().await;
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:none:transient_upstream_status,2:failed:none:none"
    );
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[0].response_metadata["status_code"], "429");
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "failed");
    assert_eq!(attempts[1].response_metadata["status_code"], "429");
    assert_eq!(attempts[1].response_metadata["retry_exhausted"], "true");
}

#[tokio::test]
async fn shielded_retry_exhausted_5xx_records_status_failure_surface() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=always-502",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"status-secret"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body = response.text().await.expect("body should be text");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("error body should be valid JSON");
    assert_eq!(parsed["error"]["type"], "llm_guard_upstream_error");
    assert_eq!(parsed["error"]["cause"], "upstream_status_error");
    assert!(parsed.get("choices").is_none());
    assert!(!body.contains("status-secret"));

    let observed = drain_upstream_requests(&mut fake, Duration::from_millis(100)).await;
    assert_eq!(
        observed
            .iter()
            .filter(|request| request.path_and_query == "/v1/chat/completions?test=always-502")
            .count(),
        2
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:none:transient_upstream_status,2:failed:none:none"
    );
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_status")
    );
    assert_eq!(attempts[0].response_metadata["status_code"], "502");
    assert_eq!(attempts[1].status, "failed");
    assert_eq!(attempts[1].response_metadata["status_code"], "502");
    assert_eq!(attempts[1].response_metadata["retry_exhausted"], "true");

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "status_error")]
        ),
        1
    );
}

#[tokio::test]
async fn shielded_retry_connection_refused_returns_structured_proxy_error() {
    let upstream_base_url = "http://127.0.0.1:1/v1";
    let proxy = ProxyFixture::spawn_with_options(
        upstream_base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
request_timeout_ms = 100

[upstream.hot_restart]
enabled = false

[heartbeat]
mode = "disabled"

[retry]
max_attempts = 2
request_deadline_ms = 300
"#,
    )
    .await;

    let response = timeout(
        Duration::from_secs(2),
        proxy_handler(
            State(proxy.state.clone()),
            shielded_chat_request(
                "/v1/chat/completions",
                r#"{"model":"test-chat","messages":[{"role":"user","content":"connect-secret"}]}"#,
            ),
        ),
    )
    .await
    .expect("proxy handler should complete before request deadline");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body_bytes = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let body = String::from_utf8(body_bytes.to_vec()).expect("body should be UTF-8");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("error body should be valid JSON");
    assert_eq!(parsed["error"]["type"], "llm_guard_upstream_error");
    assert_eq!(parsed["error"]["cause"], "upstream_connect_failed");
    assert!(parsed.get("choices").is_none());
    assert!(!body.contains("connect-secret"));

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(
        attempts[0].retry_reason.as_deref(),
        Some("transient_upstream_transport")
    );
    assert_eq!(attempts[1].status, "failed");
    assert_eq!(attempts[1].response_metadata["retry_exhausted"], "true");

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "connect_failed")]
        ),
        1
    );
}

#[tokio::test]
async fn shielded_malformed_sse_body_decode_failure_is_body_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r#"
[heartbeat]
mode = "disabled"

[retry]
max_attempts = 1
anti_loop_hint_enabled = false
"#,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 5
"#,
        "",
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=malformed-sse-invalid-json",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"body-decode-secret"}]}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.text().await.expect("body should be text");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("error body should be valid JSON");
    assert_eq!(parsed["error"]["type"], "llm_guard_upstream_error");
    assert_eq!(parsed["error"]["cause"], "upstream_body_error");
    assert_eq!(parsed["error"]["code"], "upstream_body_error");
    assert!(parsed.get("choices").is_none());
    assert!(!body.contains("body-decode-secret"));

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=malformed-sse-invalid-json"
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "non-retryable malformed SSE must not start another upstream attempt"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("upstream SSE data was not valid JSON")),
        "request error reason should identify malformed upstream SSE body: {request_row:?}"
    );
    assert_eq!(
        request_row.response_metadata["error_type"],
        "upstream_body_error"
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(attempt_row.status, "failed");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.response_metadata["error_type"],
        "upstream_body_error"
    );
    assert_eq!(
        attempt_row.response_metadata["upstream_response_received"],
        "true"
    );
    assert_eq!(attempt_row.response_metadata["http_status_success"], "true");

    assert_body_error_debug_summary(&proxy).await;

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "body_error")]
        ),
        1
    );
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "transport_error")]
        ),
        0
    );
}

#[cfg(feature = "upstream-hot-restart")]
#[tokio::test]
async fn shielded_retry_connect_hot_restart_wait_respects_request_deadline() {
    let upstream_base_url = "http://127.0.0.1:1/v1";
    let proxy = ProxyFixture::spawn_with_options(
        upstream_base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
request_timeout_ms = 100

[heartbeat]
mode = "disabled"

[retry]
max_attempts = 4
request_deadline_ms = 200
"#,
    )
    .await;

    let response = timeout(
        Duration::from_secs(2),
        proxy_handler(
            State(proxy.state.clone()),
            shielded_chat_request(
                "/v1/chat/completions",
                r#"{"model":"test-chat","messages":[{"role":"user","content":"hot-restart-secret"}]}"#,
            ),
        ),
    )
    .await
    .expect("proxy handler should complete before the test timeout");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body_bytes = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let body = String::from_utf8(body_bytes.to_vec()).expect("body should be UTF-8");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("error body should be valid JSON");
    assert_eq!(
        parsed["error"]["type"],
        "llm_guard_request_deadline_exhausted"
    );
    assert_eq!(parsed["error"]["cause"], "upstream_timeout");
    assert!(!body.contains("hot-restart-secret"));

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(request_row.http_status, 502);
    assert_eq!(
        request_row.response_metadata["request_deadline_exhausted"],
        "true"
    );
    assert_eq!(
        request_row.response_metadata["hot_restart_recovery_status"],
        "request_deadline_exhausted"
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some("request_deadline_exhausted")
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "timeout")]
        ),
        1
    );
}

#[tokio::test]
async fn hot_reloaded_retry_max_attempts_reduces_subsequent_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
"#,
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::BAD_GATEWAY);
    let _ = first.text().await.expect("first body should be text");
    for _ in 0..4 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "disabled"

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 2
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("retry max attempts reload should succeed");
    assert!(outcome.applied);

    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-reasoning-hundreds",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(second.status(), StatusCode::BAD_GATEWAY);
    let _ = second.text().await.expect("second body should be text");
    for _ in 0..2 {
        let _ = fake.recv_next().await;
    }
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let request_row = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(request_row.response_metadata["retry_max_attempts"], "2");
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_stream_options_while_forcing_usage() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream_options":{"include_usage":false,"include_obfuscation":true,"vendor_hint":{"mode":"keep"}}}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
    assert_eq!(observed_body["stream_options"]["include_obfuscation"], true);
    assert_eq!(
        observed_body["stream_options"]["vendor_hint"]["mode"],
        "keep"
    );
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_temperature_replaces_client_value() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
temperature = 0.6
",
        ),
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":1.5,"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["temperature"], json!(0.6));
    assert_eq!(observed_body["max_tokens"], 64);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_adds_missing_parameter() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
top_p = 0.95
",
        ),
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["top_p"], json!(0.95));
    assert_eq!(observed_body["max_tokens"], 64);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_disabled_passes_through() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
enabled = false
temperature = 0.6
",
        ),
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":1.5,"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["temperature"], json!(1.5));
    assert_eq!(observed_body["max_tokens"], 64);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_only_set_fields_affected() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
temperature = 0.6
",
        ),
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":1.5,"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["temperature"], json!(0.6));
    assert_eq!(observed_body["max_tokens"], 64);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_max_tokens_caps_thinking_policy_without_replacing_caller_headroom() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "param-override-test"
base_url = "{base_url}"
match_models = ["test-chat"]

[upstreams.thinking]
mode = "force_thinking"

[upstreams.param_override]
max_tokens = 50000
"#,
            base_url = fake.base_url,
        ),
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_max_tokens_is_a_default_cap_for_passthrough_thinking() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &param_override_profile_config(
            &fake.base_url,
            r"
temperature = 0.6
max_tokens = 128
",
        ),
    )
    .await;

    let caller_limited = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"caller-limited"}],"max_tokens":64}"#,
    )
    .await;
    assert_eq!(caller_limited["max_tokens"], 64);
    assert_eq!(caller_limited["temperature"], 0.6);

    let defaulted = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"defaulted"}]}"#,
    )
    .await;
    assert_eq!(defaulted["max_tokens"], 128);
    assert_eq!(defaulted["temperature"], 0.6);
}

#[cfg(feature = "param-override")]
#[tokio::test]
async fn override_max_tokens_cap_matrix_covers_every_chat_forwarding_path() {
    let mut case_index = 0_u64;
    for shielding_enabled in [false, true] {
        for shielded_streaming_enabled in [false, true] {
            let mut fake = FakeUpstream::spawn().await;
            let config = format!(
                r"
{}

[shielding]
enabled = {shielding_enabled}

[retry]
shielded_streaming_enabled = {shielded_streaming_enabled}
",
                param_override_profile_config(
                    &fake.base_url,
                    r"
temperature = 0.6
max_tokens = 128
",
                ),
            );
            let proxy = ProxyFixture::spawn_with_options(
                &fake.base_url,
                true,
                AppConfig::default().server.max_in_flight_requests,
                &config,
            )
            .await;

            for stream in [false, true] {
                for location in [
                    OutputLimitLocation::Root,
                    OutputLimitLocation::Nested,
                    OutputLimitLocation::Absent,
                ] {
                    for input_value in [None, Some(64_u64), Some(100_000_u64)] {
                        case_index = case_index.saturating_add(1);
                        assert_param_override_cap_case(
                            &proxy,
                            &mut fake,
                            ParamOverrideCapCase {
                                case_index,
                                shielding_enabled,
                                shielded_streaming_enabled,
                                stream,
                                location,
                                input_value,
                            },
                        )
                        .await;
                    }
                }
            }
        }
    }
    assert_eq!(case_index, 72);
}

#[cfg(feature = "param-override")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputLimitLocation {
    Root,
    Nested,
    Absent,
}

#[cfg(feature = "param-override")]
#[derive(Clone, Copy, Debug)]
struct ParamOverrideCapCase {
    case_index: u64,
    shielding_enabled: bool,
    shielded_streaming_enabled: bool,
    stream: bool,
    location: OutputLimitLocation,
    input_value: Option<u64>,
}

#[cfg(feature = "param-override")]
async fn assert_param_override_cap_case(
    proxy: &ProxyFixture,
    fake: &mut FakeUpstream,
    case: ParamOverrideCapCase,
) {
    const CAP: u64 = 128;
    let mut request = json!({
        "model": "test-chat",
        "messages": [{
            "role": "user",
            "content": format!("param-cap-matrix-{}", case.case_index),
        }],
        "stream": case.stream,
        "temperature": 1.5,
    });
    let request_object = request
        .as_object_mut()
        .expect("matrix request should be an object");
    match case.location {
        OutputLimitLocation::Root => {
            if let Some(value) = case.input_value {
                request_object.insert(String::from("max_tokens"), json!(value));
            }
        }
        OutputLimitLocation::Nested => {
            let mut parameters = serde_json::Map::new();
            if let Some(value) = case.input_value {
                parameters.insert(String::from("max_tokens"), json!(value));
            }
            request_object.insert(
                String::from("parameters"),
                serde_json::Value::Object(parameters),
            );
        }
        OutputLimitLocation::Absent => {}
    }

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(request.to_string())
        .send()
        .await
        .expect("matrix proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body = response
        .bytes()
        .await
        .expect("matrix response body should be readable");
    assert!(!response_body.is_empty());

    let observed = fake.recv_next().await;
    let observed_body = serde_json::from_slice::<serde_json::Value>(&observed.body)
        .expect("matrix upstream body should be JSON");
    let expected_root = if case.location == OutputLimitLocation::Root {
        case.input_value.map_or(CAP, |value| value.min(CAP))
    } else {
        CAP
    };
    let label = format!(
        "shielding={} shielded_streaming={} stream={} location={:?} input={:?}",
        case.shielding_enabled,
        case.shielded_streaming_enabled,
        case.stream,
        case.location,
        case.input_value,
    );
    assert_eq!(observed_body["max_tokens"], expected_root, "{label}");
    assert_eq!(observed_body["temperature"], 0.6, "{label}");

    let expected_nested = (case.location == OutputLimitLocation::Nested)
        .then(|| case.input_value.map_or(CAP, |value| value.min(CAP)));
    assert_eq!(
        observed_body
            .get("parameters")
            .and_then(serde_json::Value::as_object)
            .and_then(|parameters| parameters.get("max_tokens"))
            .and_then(serde_json::Value::as_u64),
        expected_nested,
        "{label}",
    );

    if case.shielding_enabled {
        let metadata = read_latest_attempt_request_metadata(proxy);
        assert_eq!(
            metadata["thinking_answer_budget_final_max_tokens"],
            expected_root.to_string(),
            "{label}",
        );
        assert_eq!(
            metadata["thinking_answer_budget_final_parameters_max_tokens"],
            expected_nested.map_or_else(|| String::from("absent"), |value| value.to_string()),
            "{label}",
        );
    }
}

#[tokio::test]
async fn force_thinking_canonical_default_injects_thinking_budget_tokens() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert!(observed_body.get("chat_template_kwargs").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_default_injection_schema"], "canonical");
        assert_eq!(metadata["thinking_schema_path"], "thinking.budget_tokens");
        assert_eq!(metadata["thinking_schema_variant"], "canonical");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_chat_template_kwargs_schema_injects_enable_thinking_and_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_default_injection_schema"],
            "chat_template_kwargs"
        );
        assert_eq!(
            metadata["thinking_schema_path"],
            "chat_template_kwargs.thinking_budget"
        );
        assert_eq!(metadata["thinking_schema_variant"], "chat-template-kwargs");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_vllm_native_schema_uses_native_budget_and_template_enablement() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "vllm_native"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"sensitive-prompt"}],"chat_template_kwargs":{"custom_flag":"preserve-me","thinking_budget":8},"extra_body":{"custom_object":{"value":7}},"unknown_top_level":"preserve-me-too","max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking_token_budget"], 32_768);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["custom_flag"],
        "preserve-me"
    );
    assert_eq!(observed_body["extra_body"]["custom_object"]["value"], 7);
    assert_eq!(observed_body["unknown_top_level"], "preserve-me-too");
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_default_injection_schema"], "vllm_native");
        assert_eq!(metadata["thinking_schema_path"], "thinking_token_budget");
        assert_eq!(
            metadata["thinking_enable_marker_path"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_text_excludes_values(
            &metadata.to_string(),
            &["sensitive-prompt", "preserve-me", "preserve-me-too"],
        );
    }
}

#[tokio::test]
async fn bounded_thinking_vllm_native_schema_uses_native_budget_and_template_enablement() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "bounded_thinking"
default_injection_schema = "vllm_native"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"thinking_budget":8192},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking_token_budget"], 32_768);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_832);
}

#[tokio::test]
async fn force_thinking_vllm_native_ignores_legacy_budgets_when_preserving_answer_headroom() {
    assert_vllm_native_legacy_budget_matrix("force_thinking", [32_768, 8_192, 65_536]).await;
}

#[tokio::test]
async fn bounded_thinking_vllm_native_ignores_legacy_budgets_when_preserving_answer_headroom() {
    assert_vllm_native_legacy_budget_matrix("bounded_thinking", [32_768, 8_192, 65_536]).await;
}

async fn assert_vllm_native_legacy_budget_matrix(mode: &str, legacy_budgets: [u64; 3]) {
    let mut fake = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[thinking]
mode = "{mode}"
default_injection_schema = "vllm_native"
"#,
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    for (index, legacy_budget) in legacy_budgets.into_iter().enumerate() {
        let body = format!(
            r#"{{"model":"test-chat","messages":[{{"role":"user","content":"case-{index}"}}],"chat_template_kwargs":{{"thinking_budget":{legacy_budget}}},"max_tokens":64}}"#,
        );
        let observed_body =
            post_chat_and_observe_owned_body(&proxy, &mut fake, Bytes::from(body)).await;

        assert_eq!(observed_body["thinking_token_budget"], 32_768);
        assert_eq!(
            observed_body["chat_template_kwargs"]["enable_thinking"],
            true
        );
        assert_eq!(observed_body["max_tokens"], 32_832);
    }
}

#[tokio::test]
async fn bounded_thinking_vllm_native_rejects_positive_budget_with_explicit_opt_out() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
mode = "bounded_thinking"
default_injection_schema = "vllm_native"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"chat_template_kwargs":{"enable_thinking":false},"thinking_token_budget":8192,"max_tokens":64}"#,
    )
    .await;
}

#[tokio::test]
async fn bounded_thinking_vllm_native_zero_config_normalizes_zero_native_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "bounded_thinking"
budget_tokens = 0
default_injection_schema = "vllm_native"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"zero-budget"}],"thinking_token_budget":0,"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["thinking_token_budget"], 0);
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn bounded_thinking_vllm_native_normalizes_all_explicit_opt_outs() {
    let cases = [
        r#""enable_thinking":false"#,
        r#""thinking":{"enabled":false}"#,
        r#""chat_template_kwargs":{"enable_thinking":false}"#,
        r#""thinking_token_budget":0"#,
    ];
    assert_vllm_native_disabled_cases("bounded_thinking", &cases).await;
}

#[tokio::test]
async fn force_disable_vllm_native_normalizes_markers_and_malformed_containers() {
    let cases = [
        r#""enable_thinking":false"#,
        r#""thinking":{"enabled":false}"#,
        r#""chat_template_kwargs":{"enable_thinking":false}"#,
        r#""chat_template_kwargs":"malformed","thinking_token_budget":8192"#,
    ];
    assert_vllm_native_disabled_cases("force_disable", &cases).await;
}

async fn assert_vllm_native_disabled_cases(mode: &str, cases: &[&str]) {
    let mut fake = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[thinking]
mode = "{mode}"
default_injection_schema = "vllm_native"
"#,
    );
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &config,
    )
    .await;

    for (index, fields) in cases.iter().enumerate() {
        let body = format!(
            r#"{{"model":"test-chat","messages":[{{"role":"user","content":"disable-case-{index}"}}],{fields},"max_tokens":64}}"#,
        );
        let observed_body =
            post_chat_and_observe_owned_body(&proxy, &mut fake, Bytes::from(body)).await;

        assert_eq!(
            observed_body["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert!(
            observed_body.get("thinking_token_budget").is_none()
                || observed_body["thinking_token_budget"] == 0
        );
        assert_eq!(observed_body["max_tokens"], 64);
    }
}

#[tokio::test]
async fn force_thinking_chat_template_kwargs_schema_preserves_existing_containers() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"extra_body":{"thinking":{}},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["extra_body"]["thinking"]["budget_tokens"],
        32_768
    );
    assert!(observed_body.get("chat_template_kwargs").is_none());
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_default_injection_schema"],
            "chat_template_kwargs"
        );
        assert_eq!(
            metadata["thinking_schema_path"],
            "extra_body.thinking.budget_tokens"
        );
        assert_eq!(metadata["thinking_schema_variant"], "extra-body-canonical");
    }
}

#[tokio::test]
async fn hot_reloaded_default_injection_schema_switches_legacy_and_native_without_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"

[loop_guard]
enabled = false
"#,
    )
    .await;

    let legacy_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"legacy-before"}],"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        legacy_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert!(legacy_body.get("thinking_token_budget").is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "vllm_native"

[loop_guard]
enabled = false
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("default injection schema reload should succeed");
    assert!(outcome.applied);

    let native_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"native-middle"}],"max_tokens":64}"#,
    )
    .await;
    assert_eq!(native_body["thinking_token_budget"], 32_768);
    assert_eq!(native_body["chat_template_kwargs"]["enable_thinking"], true);
    assert!(
        native_body["chat_template_kwargs"]
            .get("thinking_budget")
            .is_none()
    );

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
default_injection_schema = "chat_template_kwargs"

[loop_guard]
enabled = false
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("legacy schema reload should succeed");
    assert!(outcome.applied);

    let legacy_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"legacy-after"}],"max_tokens":64}"#,
    )
    .await;
    assert!(legacy_body.get("thinking_token_budget").is_none());
    assert_eq!(
        legacy_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
}

#[tokio::test]
async fn shielded_thinking_policy_injects_missing_budget_and_preserves_answer_reserve() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_policy_enabled"], "true");
        assert_eq!(metadata["thinking_policy_budget_tokens"], "32768");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "injected_missing_budget"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "absent");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_schema_path"], "thinking.budget_tokens");
        assert_eq!(metadata["thinking_schema_variant"], "canonical");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32768");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "true"
        );
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens"
        );
    }
}

#[tokio::test]
async fn force_thinking_default_overrides_enable_thinking_false() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_no_thinking_marker_policy"], "force");
        assert_eq!(metadata["thinking_no_thinking_marker_detected"], "true");
        assert_eq!(
            metadata["thinking_no_thinking_marker_source"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(metadata["thinking_no_thinking_marker_overridden"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
    }
}

#[tokio::test]
async fn force_thinking_respect_markers_preserves_enable_thinking_false() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert!(
        observed_body["chat_template_kwargs"]
            .get("thinking_budget")
            .is_none()
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_vllm_native_respect_marker_rejects_positive_native_budget() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
default_injection_schema = "vllm_native"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"chat_template_kwargs":{"enable_thinking":false},"thinking_token_budget":8192,"max_tokens":64}"#,
    )
    .await;
}

#[tokio::test]
async fn force_thinking_respect_markers_preserves_reasoning_effort_none() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"reasoning_effort":"none","max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["reasoning_effort"], "none");
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_escape_hatch_only_honors_disable_thinking_escape_hatch() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "escape_hatch_only"
"#,
    )
    .await;

    let normal_marker_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        normal_marker_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        normal_marker_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(normal_marker_body["max_tokens"], 32_832);

    let escape_hatch_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"llm_guard_proxy_disable_thinking":true,"max_tokens":64}"#,
    )
    .await;
    assert_eq!(escape_hatch_body["llm_guard_proxy_disable_thinking"], true);
    assert!(escape_hatch_body.get("thinking").is_none());
    assert_eq!(escape_hatch_body["max_tokens"], 64);
}

#[tokio::test]
async fn force_thinking_vllm_native_escape_hatch_rejects_positive_native_budget() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "escape_hatch_only"
default_injection_schema = "vllm_native"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"llm_guard_proxy_disable_thinking":true,"thinking_token_budget":8192,"max_tokens":64}"#,
    )
    .await;
}

#[tokio::test]
async fn force_thinking_respect_markers_records_observability() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_no_thinking_marker_policy"],
            "respect_no_thinking_markers"
        );
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "caller_no_thinking_marker_passthrough"
        );
        assert_eq!(metadata["thinking_no_thinking_marker_detected"], "true");
        assert_eq!(
            metadata["thinking_no_thinking_marker_source"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(
            metadata["thinking_no_thinking_marker_escape_hatch"],
            "false"
        );
        assert!(
            metadata
                .get("thinking_no_thinking_marker_overridden")
                .is_none()
        );
    }
}

#[tokio::test]
async fn hot_reloaded_no_thinking_marker_policy_changes_force_thinking_behavior() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
"#,
    )
    .await;

    let forced_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(forced_body["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(
        forced_body["chat_template_kwargs"]["thinking_budget"],
        32_768
    );

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_thinking"
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("marker policy reload should succeed");
    assert!(outcome.applied);

    let passthrough_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
    assert_eq!(
        passthrough_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert!(
        passthrough_body["chat_template_kwargs"]
            .get("thinking_budget")
            .is_none()
    );
    assert_eq!(passthrough_body["max_tokens"], 64);
}

#[tokio::test]
async fn tool_request_passthrough_leaves_thinking_and_answer_budget_untouched() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 1);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_policy_enabled"], "true");
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "tool_request_passthrough"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "smaller");
        assert_eq!(metadata["thinking_budget_final_tokens"], "1");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "false"
        );
    }
}

#[tokio::test]
async fn vllm_native_policy_disabled_rejects_conflicting_no_thinking_controls() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
enabled = false
default_injection_schema = "vllm_native"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
}

#[tokio::test]
async fn vllm_native_mode_passthrough_rejects_conflicting_no_thinking_controls() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
mode = "passthrough"
default_injection_schema = "vllm_native"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}"#,
    )
    .await;
}

#[tokio::test]
async fn vllm_native_tool_passthrough_rejects_conflicting_no_thinking_controls() {
    assert_vllm_native_conflict_rejected(
        r#"
[thinking]
default_injection_schema = "vllm_native"
tool_request_policy = "passthrough"
"#,
        r#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"tools":[{"type":"function","function":{"name":"sensitive-tool","parameters":{"type":"object","properties":{"secret-property":{"type":"string"}}}}}],"tool_choice":{"type":"function","function":{"name":"sensitive-tool"}}}"#,
    )
    .await;
}

async fn assert_vllm_native_conflict_rejected(config: &str, body: &str) {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        config,
    )
    .await;
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_owned())
        .send()
        .await
        .expect("proxy rejection should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "conflicting_thinking_controls");
    assert_eq!(error["error"]["param"], "thinking_token_budget");
    assert_eq!(
        error["error"]["message"],
        "positive thinking_token_budget cannot be combined with an explicit no-thinking marker"
    );
    assert_text_excludes_values(
        &error.to_string(),
        &["private-prompt", "sensitive-tool", "secret-property"],
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn vllm_native_tool_passthrough_preserves_non_conflicting_budget_and_private_payload() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
default_injection_schema = "vllm_native"
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"private-prompt"}],"tools":[{"type":"function","function":{"name":"sensitive-tool","parameters":{"type":"object","properties":{"secret-property":{"type":"string"}}}}}],"tool_choice":{"type":"function","function":{"name":"sensitive-tool"}},"thinking_token_budget":7,"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking_token_budget"], 7);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(
        observed_body["tools"][0]["function"]["name"],
        "sensitive-tool"
    );
    assert_eq!(
        observed_body["tool_choice"]["function"]["name"],
        "sensitive-tool"
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "tool_request_passthrough"
        );
        assert_text_excludes_values(
            &metadata.to_string(),
            &["private-prompt", "sensitive-tool", "secret-property"],
        );
    }
}

#[tokio::test]
async fn force_disable_thinking_zeroes_existing_budget_paths_without_answer_budget_raise() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
force_disable = true
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false,"thinking_token_budget":123,"thinking":{"budget_tokens":456,"enabled":true},"chat_template_kwargs":{"thinking_budget":789,"enable_thinking":true},"extra_body":{"thinking_token_budget":321},"max_tokens":64,"max_completion_tokens":32,"max_output_tokens":16}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking_token_budget"], 0);
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(observed_body["thinking"]["enabled"], false);
    assert_eq!(observed_body["chat_template_kwargs"]["thinking_budget"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["extra_body"]["thinking_token_budget"], 0);
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["max_completion_tokens"], 32);
    assert_eq!(observed_body["max_output_tokens"], 16);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "false"
        );

        let rewritten_paths = metadata["thinking_budget_rewritten_paths"]
            .as_str()
            .expect("rewritten paths should be a string");
        for expected_path in [
            "thinking_token_budget",
            "thinking.budget_tokens",
            "chat_template_kwargs.thinking_budget",
            "extra_body.thinking_token_budget",
        ] {
            assert!(
                rewritten_paths.split(',').any(|path| path == expected_path),
                "missing rewritten path {expected_path} in {rewritten_paths}"
            );
        }
    }
}

#[tokio::test]
async fn force_disable_thinking_overrides_tool_request_passthrough() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
force_disable = true
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
    }
}

#[tokio::test]
async fn force_disable_vllm_native_overrides_tool_passthrough_and_clears_native_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
mode = "force_disable"
default_injection_schema = "vllm_native"
tool_request_policy = "passthrough"
"#,
    )
    .await;

    let observed_body = post_chat_and_observe_body(
        &proxy,
        &mut fake,
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"thinking_token_budget":8192,"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
    )
    .await;

    assert_eq!(observed_body["thinking_token_budget"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
}

#[tokio::test]
async fn tool_request_passthrough_policy_still_injects_non_tool_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_policy"], "passthrough");
        assert_eq!(metadata["thinking_tool_request_detected"], "false");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "injected_missing_budget"
        );
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_legacy_functions_and_preserves_budgets() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"functions":[{"name":"lookup","parameters":{"type":"object","properties":{}}}],"thinking":{"budget_tokens":200},"max_completion_tokens":50}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 200);
    assert_eq!(observed_body["max_completion_tokens"], 50);
    assert!(observed_body.get("max_tokens").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "tool_request_passthrough"
        );
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_tool_choice_selector_only() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    // tool_choice="auto" without tools array should still be treated as a tool request.
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tool_choice":"auto","thinking":{"budget_tokens":77},"max_output_tokens":40}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 77);
    assert_eq!(observed_body["max_output_tokens"], 40);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn tool_request_passthrough_ignores_tool_choice_none() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    // tool_choice="none" should NOT trigger passthrough; regular thinking policy applies.
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"tool_choice":"none","max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    // Regular policy injected the default budget and adjusted max_tokens.
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "false");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
    }
}

#[tokio::test]
async fn tool_request_passthrough_detects_legacy_function_call_selector() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[thinking]
tool_request_policy = "passthrough"
"#,
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"function_call":"auto","thinking":{"budget_tokens":99},"max_tokens":30}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 99);
    assert_eq!(observed_body["max_tokens"], 30);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_tool_request_detected"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn streaming_chat_applies_thinking_policy_without_downstream_aggregation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64,"stream":true}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("streaming fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let response_body = response.text().await.expect("stream body should be text");
    assert!(response_body.contains("chat.completion.chunk"));
    assert!(!response_body.contains("event: final"));

    let observed = fake.recv_next().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_832);
    assert!(observed_body.get("stream_options").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["policy_transform_applied"], "true");
        assert_eq!(metadata["thinking_policy_enabled"], "true");
        assert_eq!(metadata["thinking_policy_budget_tokens"], "32768");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "injected_missing_budget"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "absent");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_schema_path"], "thinking.budget_tokens");
        assert_eq!(metadata["thinking_schema_variant"], "canonical");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32768");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "true"
        );
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens"
        );
        assert!(metadata.get("shielded_streaming").is_none());
        assert!(metadata.get("upstream_stream_forced").is_none());
    }
}

#[tokio::test]
async fn streaming_chat_force_disable_thinking_injects_zero_without_downstream_aggregation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
force_disable = true
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true,"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("streaming fake upstream SSE should be used"),
        "chat-completions-sse"
    );
    let response_body = response.text().await.expect("stream body should be text");
    assert!(response_body.contains("chat.completion.chunk"));
    assert!(!response_body.contains("event: final"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["chat_template_kwargs"]["thinking_budget"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert!(observed_body.get("stream_options").is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["policy_transform_applied"], "true");
        assert_eq!(metadata["thinking_force_disable_enabled"], "true");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_previous_state"], "absent");
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(
            metadata["thinking_schema_path"],
            "chat_template_kwargs.thinking_budget"
        );
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert!(metadata.get("shielded_streaming").is_none());
        assert!(metadata.get("upstream_stream_forced").is_none());
    }
}

#[tokio::test]
async fn shielded_streaming_commit_gate_sends_heartbeat_before_openai_sse_release() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=slow-shielded",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let heartbeat = next_chunk(
        &mut body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "shielded stream heartbeat",
    )
    .await;
    assert_eq!(
        heartbeat,
        Bytes::from_static(b": llm-guard-proxy heartbeat\n\n")
    );
    assert!(!String::from_utf8_lossy(&heartbeat).contains("content"));
    assert!(!String::from_utf8_lossy(&heartbeat).contains("tool_calls"));

    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("data:"));
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains("Hel"));
    assert!(!released.contains("event: final"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].response_metadata["retry_shielded_streaming_enabled"],
        "true"
    );
    assert_eq!(
        attempts[0].response_metadata["downstream_liveness_mode"],
        "sse"
    );
}

#[tokio::test]
async fn shielded_streaming_trims_reasoning_separator_from_released_openai_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=reasoning-leading-newlines",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"Say OK"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains(r#""content":"OK""#), "{released}");
    assert!(!released.contains(r#""content":"\n\nOK""#), "{released}");
    assert!(released.contains("data: [DONE]"));
    assert!(!released.contains("event: final"));

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=reasoning-leading-newlines"
    );
}

#[tokio::test]
async fn shielded_streaming_emits_aggregated_logprobs_once_in_openai_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true,"logprobs":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 2, "{released}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "Hello");
    let logprobs = chunks[0]["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("delta chunk should carry aggregated logprobs once");
    assert_eq!(logprobs.len(), 2);
    assert_eq!(logprobs[0]["token"], "Hello");
    assert_eq!(logprobs[1]["token"], "!");
    assert!(
        chunks[1]["choices"][0].get("logprobs").is_none(),
        "{released}"
    );
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
    assert!(released.contains("data: [DONE]"));

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["logprobs"], true);
}

#[tokio::test]
async fn shielded_streaming_emits_tool_calls_in_openai_delta_shape() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[retry]
shielded_streaming_enabled = true
",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"tool"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 2, "{released}");
    let tool_calls = chunks[0]["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .expect("delta chunk should carry tool calls");
    assert_eq!(tool_calls[0]["index"], 0);
    assert_eq!(tool_calls[0]["id"], "call_1");
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(tool_calls[0]["function"]["arguments"], r#"{"q":"x"}"#);
    assert!(
        chunks[1]["choices"][0]["delta"].get("tool_calls").is_none(),
        "{released}"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
}

#[tokio::test]
async fn shielded_streaming_discards_rejected_tool_call_buffer_before_success() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
interval_secs = 1

[loop_guard]
mode = "enforce"

[retry]
max_attempts = 2
shielded_streaming_enabled = true
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=tool-loop-then-content-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"tool"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let first = next_chunk(
        &mut body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "shielded stream heartbeat",
    )
    .await;
    let mut released = String::from_utf8_lossy(&first).into_owned();
    if first == Bytes::from_static(b": llm-guard-proxy heartbeat\n\n") {
        assert!(!released.contains("tool_calls"));
        released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    } else {
        released.push_str(&collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await);
    }
    assert!(released.contains("Safe"));
    assert!(!released.contains("lookup"));
    assert!(!released.contains("arguments"));
    assert!(!released.contains("tool_calls"));

    let _first_attempt = fake.recv_next().await;
    let _second_attempt = fake.recv_next().await;
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].status, "succeeded");
}

#[tokio::test]
async fn shielded_streaming_direct_relays_final_no_thinking_retry_after_loop_downgrades() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=loop-twice-then-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("chat.completion.chunk"));
    assert!(released.contains("Hel"));
    assert!(released.contains("data: [DONE]"));
    assert!(!released.contains("event: final"));
    assert!(!released.contains("reasoning loop line"));

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));

    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].status, "succeeded");
    assert_eq!(attempts[2].response_metadata["attempt_name"], "no-thinking");
    assert_eq!(
        attempts[2].response_metadata["attempt_thinking_mode"],
        "force_disable"
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_loop_inspection_skipped"],
        "no_thinking_direct_streaming_relay"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    assert_eq!(
        request_row.response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
}

#[tokio::test]
async fn shielded_streaming_direct_no_thinking_drop_cancels_upstream_relay() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        1,
        r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true
downstream_drop_policy = "cancel"

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-cancellable-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#)
        .send()
        .await
        .expect("streaming direct relay request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    let mut downstream = response.bytes_stream();
    let first = next_chunk(
        &mut downstream,
        STREAM_COMPLETION_TIMEOUT,
        "first direct no-thinking SSE chunk",
    )
    .await;
    assert!(first.starts_with(b"data: "));
    drop(downstream);

    let first_attempt = upstream.recv_request().await;
    let second_attempt = upstream.recv_request().await;
    let third_attempt = upstream.recv_request().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(0));

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempts.len(), 3);
    assert_eq!(attempts[2].status, "aborted");
    assert_eq!(
        attempts[2].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        attempts[2].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
}

#[tokio::test]
async fn shielded_retry_streaming_final_no_thinking_direct_relay_deadline_terminates() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        FINAL_DIRECT_RELAY_DEADLINE_CONFIG,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=loop-three-then-slow-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stream"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(
        !released.trim().is_empty(),
        "deadline-terminated final relay must emit an SSE error instead of an empty 200 body"
    );
    assert!(released.contains("event: error"));
    assert!(released.contains("llm_guard_request_deadline_exhausted"));
    assert!(
        !released.contains("data: [DONE]"),
        "deadline-terminated final relay must not masquerade as a complete SSE answer"
    );
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0]["error"]["type"],
        "llm_guard_request_deadline_exhausted"
    );

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    let third_attempt = fake.recv_next().await;
    let fourth_attempt = fake.recv_next().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    assert_eq!(body_thinking_budget(&second_attempt.body), Some(1_024));
    assert_eq!(body_thinking_budget(&third_attempt.body), Some(8_192));
    assert_eq!(body_thinking_budget(&fourth_attempt.body), Some(0));
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "request-level deadline must not start a fifth attempt"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("final_direct_relay_terminated")
    );
    assert_eq!(
        request_row.response_metadata["shielded_terminal_reason"],
        "final_direct_relay_terminated"
    );
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:loop_guard:loop_detected,2:retried:loop_guard:loop_detected,3:retried:loop_guard:loop_detected,4:aborted:final_direct_relay_terminated:none"
    );
    assert_eq!(attempts.len(), 4);
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[1].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[2].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[3].status, "aborted");
    assert_eq!(
        attempts[3].abort_reason.as_deref(),
        Some("final_direct_relay_terminated")
    );
    assert_eq!(
        attempts[3].response_metadata["shielded_direct_streaming_relay"],
        "true"
    );
    assert_eq!(
        attempts[3].response_metadata["shielded_direct_streaming_relay_deadline_bound"],
        "true"
    );
    assert_eq!(
        attempts[3].response_metadata["final_direct_relay_terminated"],
        "true"
    );
    assert_eq!(
        attempts[3].response_metadata["request_deadline_exhausted"],
        "true"
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "timeout")]
        ),
        1
    );
}

const FINAL_DIRECT_RELAY_DEADLINE_CONFIG: &str = r#"
[heartbeat]
mode = "sse"
interval_secs = 15

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
request_deadline_ms = 200
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking-deep"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 1024

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 128
"#;

#[tokio::test]
async fn shielded_streaming_stall_failure_emits_valid_sse_error_and_metrics() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "sse"
interval_secs = 15

[retry]
max_attempts = 1
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 50
idle_timeout_ms = 50
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=stall-once-then-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"stall-secret"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("event: error"));
    assert!(released.contains("llm_guard_attempt_timeout"));
    assert!(!released.contains("data: [DONE]"));
    assert!(!released.contains("stall-secret"));
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0]["error"]["type"], "llm_guard_attempt_timeout");

    let first_attempt = fake.recv_next().await;
    assert_eq!(
        first_attempt.path_and_query,
        "/v1/chat/completions?test=stall-once-then-success"
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "retry-exhausted stall failure must not start another upstream attempt"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(
        request_row.response_metadata["upstream_stall_detected"],
        "true"
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("upstream_stall"));
    assert_eq!(
        attempts[0].response_metadata["upstream_stall_detected"],
        "true"
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "timeout")]
        ),
        1
    );
}

#[tokio::test]
async fn delayed_first_chunk_within_first_chunk_timeout_succeeds() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "sse"
interval_secs = 15

[retry]
max_attempts = 1
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 1000
idle_timeout_ms = 50
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=delayed-first-chunk-then-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"delayed-first"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("Hello") || released.contains("Hel"));
    assert!(!released.contains("event: error"));
    assert!(released.contains("data: [DONE]"));

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "succeeded");
    assert_ne!(
        request_row.response_metadata["upstream_stall_detected"],
        "true"
    );

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=delayed-first-chunk-then-success"
    );
}

#[tokio::test]
async fn inter_chunk_stall_after_first_chunk_returns_stall_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "sse"
interval_secs = 15

[retry]
max_attempts = 1
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 1000
idle_timeout_ms = 50
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=inter-chunk-stall-after-first",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"inter-chunk-secret"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("event: error"));
    assert!(released.contains("llm_guard_attempt_timeout"));
    assert!(!released.contains("data: [DONE]"));

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(
        request_row.response_metadata["upstream_stall_detected"],
        "true"
    );
    assert_eq!(
        request_row.response_metadata["upstream_stall_idle_timeout_ms"],
        "50"
    );

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=inter-chunk-stall-after-first"
    );
}

#[tokio::test]
async fn shielded_retry_streaming_request_deadline_exhaustion_stops_waiting() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r#"
[heartbeat]
mode = "sse"
interval_secs = 15

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 4
request_deadline_ms = 100
anti_loop_hint_enabled = false
shielded_streaming_enabled = true
"#,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
debug_summary_max_records = 5
"#,
        "",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=slow-shielded",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"deadline-secret"}],"stream":true}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let released = collect_stream_text(&mut body, STREAM_COMPLETION_TIMEOUT).await;
    assert!(released.contains("llm_guard_request_deadline_exhausted"));
    assert!(!released.contains("data: [DONE]"));
    assert!(!released.contains("deadline-secret"));
    let chunks = openai_sse_json_chunks(&released);
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0]["error"]["type"],
        "llm_guard_request_deadline_exhausted"
    );

    let first_attempt = fake.recv_next().await;
    assert_eq!(
        first_attempt.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "deadline exhaustion must not start another upstream attempt"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_ne!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        request_row.response_metadata["shielded_terminal_reason"],
        "request_deadline_exhausted"
    );
    assert_eq!(
        request_row.response_metadata["request_deadline_exhausted"],
        "true"
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some("request_deadline_exhausted")
    );
    assert_eq!(
        attempts[0].response_metadata["request_deadline_exhausted"],
        "true"
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_upstream_failure_total",
            &[("cause", "timeout")]
        ),
        1
    );

    assert_deadline_debug_summary(&proxy).await;
}

#[tokio::test]
async fn per_model_routing_selects_named_upstream_and_records_bounded_metadata() {
    let mut default = FakeUpstream::spawn().await;
    let mut aeon = FakeUpstream::spawn().await;
    let mut fast = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{aeon_base_url}"
match_models = ["aeon-ultimate"]
request_timeout_ms = 90000

[upstreams.metadata]
context_length_override = 4096
input_token_safety_margin = 64

[upstreams.thinking]
mode = "force_thinking"
thinking_token_budget = 128

[[upstreams]]
name = "fast-no-think"
base_url = "{fast_base_url}"
match_models = ["fast-local"]

[upstreams.thinking]
mode = "force_disable"
"#,
            aeon_base_url = aeon.base_url,
            fast_base_url = fast.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"aeon-ultimate","prompt":"hello","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"cmpl-test","object":"text_completion"}"#
    );
    let observed = aeon.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/completions");
    assert!(
        default
            .recv_within(Duration::from_millis(100))
            .await
            .is_none()
    );
    assert!(fast.recv_within(Duration::from_millis(100)).await.is_none());

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "aeon-chat");
        assert_eq!(metadata["upstream_route_reason"], "matched_model");
        assert_eq!(metadata["upstream_request_timeout_ms"], "90000");
        assert_eq!(metadata["upstream_context_window_tokens"], "4096");
        assert_eq!(metadata["upstream_input_token_safety_margin"], "64");
        assert!(!metadata.to_string().contains("hello"));
    }
}

#[tokio::test]
async fn unmatched_or_missing_model_routes_to_default_profile() {
    let mut default = FakeUpstream::spawn().await;
    let mut aeon = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &default.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{aeon_base_url}"
match_models = ["aeon-ultimate"]
"#,
            aeon_base_url = aeon.base_url,
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"unmatched-model","prompt":"hello","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let unmatched = default.recv_next().await;
    assert_eq!(unmatched.path_and_query, "/v1/completions");

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"prompt":"missing model","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let missing = default.recv_next().await;
    assert_eq!(missing.path_and_query, "/v1/completions");
    assert!(aeon.recv_within(Duration::from_millis(100)).await.is_none());
}

#[tokio::test]
async fn per_profile_thinking_force_thinking_overrides_disable_marker_and_total_cap() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{base_url}"
match_models = ["aeon-ultimate"]

[upstreams.thinking]
mode = "force_thinking"
max_tokens = 10
thinking_token_budget = 4
budget_accounting = "total_cap"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"aeon-ultimate","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":1},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 4);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        true
    );
    assert_eq!(observed_body["max_tokens"], 10);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "aeon-chat");
        assert_eq!(metadata["thinking_policy_mode"], "force_thinking");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "forced_configured_budget"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "4");
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens"
        );
    }
}

#[tokio::test]
async fn per_profile_thinking_force_disable_writes_zero_budget_and_disable_marker() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fast-no-think"
base_url = "{base_url}"
match_models = ["fast-local"]

[upstreams.thinking]
mode = "force_disable"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"fast-local","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":9},"chat_template_kwargs":{"enable_thinking":true},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "fast-no-think");
        assert_eq!(metadata["thinking_policy_mode"], "force_disable");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "force_disabled_thinking"
        );
        assert_eq!(metadata["thinking_budget_final_tokens"], "0");
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
    }
}

#[tokio::test]
async fn per_profile_thinking_passthrough_leaves_caller_thinking_fields_unchanged() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "tool-route"
base_url = "{base_url}"
match_models = ["tool-model"]

[upstreams.thinking]
mode = "passthrough"
"#,
            base_url = fake.base_url,
        ),
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"tool-model","messages":[{"role":"user","content":"ping"}],"stream":true,"thinking":{"budget_tokens":7},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.text().await.expect("body should be text");
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 7);
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 2);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["upstream_profile"], "tool-route");
        assert_eq!(metadata["thinking_policy_mode"], "passthrough");
        assert_eq!(metadata["thinking_rewrite_reason"], "mode_passthrough");
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
    }
}

#[tokio::test]
async fn streaming_chat_downstream_drop_cancels_upstream_relay() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("streaming chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("cancellable SSE upstream should be used"),
        "cancellable-chat-sse"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);

    let mut downstream = response.bytes_stream();
    let first = next_chunk(
        &mut downstream,
        STREAM_FIRST_CHUNK_TIMEOUT,
        "first cancellable SSE chunk",
    )
    .await;
    assert!(first.starts_with(b"data: "));
    drop(downstream);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn non_stream_chat_downstream_drop_cancels_upstream_body() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "",
        r"
[shielding]
enabled = false
",
        "",
        "",
    )
    .await;

    // Use /v1/completions (not subject to choices validation, which buffers
    // the body) so the downstream-drop cancellation path is exercised.
    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-completion","prompt":"ping"}"#)
        .send()
        .await
        .expect("non-stream completion request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-endpoint")
            .expect("cancellable JSON upstream should be used"),
        "cancellable-chat-json"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_ne!(observed_body["stream"], true);

    drop(response);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-json");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn shielded_non_stream_chat_downstream_drop_cancels_upstream_aggregation() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("shielded chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("shielded non-stream response should advertise JSON"),
        "application/json"
    );
    let observed = upstream.recv_request().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    drop(response);

    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");
    assert_forwarded_abort_recorded(&proxy);
}

#[tokio::test]
async fn shielded_non_stream_detach_drop_allows_upstream_attempt_to_continue_until_timeout() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"

[retry]
downstream_drop_policy = "detach"

[upstream.stall]
enabled = true
first_chunk_timeout_ms = 200
idle_timeout_ms = 200
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("shielded chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    let observed = upstream.recv_request().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);

    let mut downstream = response.bytes_stream();
    let heartbeat = next_chunk(
        &mut downstream,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "detach JSON prefix",
    )
    .await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(downstream);

    assert!(
        upstream
            .recv_drop_optional_within(Duration::from_millis(50))
            .await
            .is_none(),
        "detach mode should not cancel upstream immediately on downstream drop"
    );
    let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
    assert_eq!(drop_event.label, "cancellable-chat-sse");

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(request_row.status, "aborted");
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(
        request_row.response_metadata["downstream_drop_policy"],
        "detach"
    );
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "aborted");
    assert_eq!(
        attempts[0].response_metadata["downstream_drop_policy"],
        "detach"
    );
}

#[tokio::test]
async fn shielded_non_stream_detach_drop_does_not_start_retry_after_disconnect() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &upstream.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"

[evidence]
enabled = true
include_raw_payloads = false

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 1000

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true
downstream_drop_policy = "detach"

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
thinking_token_budget = 8192
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=delayed-loop-then-cancellable-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("shielded chat request should receive response headers");

    assert_eq!(response.status(), StatusCode::OK);
    let first_attempt = upstream.recv_request().await;
    assert_eq!(body_thinking_budget(&first_attempt.body), Some(32_768));
    let mut downstream = response.bytes_stream();
    let heartbeat = next_chunk(
        &mut downstream,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "detach JSON prefix before retry cancellation",
    )
    .await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(downstream);

    assert!(
        upstream
            .recv_request_optional_within(Duration::from_millis(500))
            .await
            .is_none(),
        "detached downstream drops may let the current attempt finish, but must not advance the retry ladder or shadow evidence attempts"
    );
}

#[tokio::test]
async fn shielded_thinking_policy_respects_explicit_disable_marker() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"api_key":"sk-secret"}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("thinking").is_none());
    assert!(observed_body.get("thinking_budget").is_none());
    assert_eq!(
        observed_body["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(
            metadata["thinking_rewrite_reason"],
            "caller_disabled_thinking"
        );
        assert_eq!(
            metadata["thinking_disable_marker_path"],
            "chat_template_kwargs.enable_thinking"
        );
        assert_eq!(
            metadata["thinking_answer_budget_preserved_fields"],
            "max_tokens"
        );
        assert_text_excludes_values(&metadata.to_string(), &["sk-secret", "api_key"]);
    }
}

#[tokio::test]
async fn shielded_thinking_policy_preserves_zero_existing_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":0},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 0);
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["thinking_rewrite_applied"], "false");
    assert_eq!(
        request_metadata["thinking_rewrite_reason"],
        "existing_budget_zero"
    );
    assert_eq!(request_metadata["thinking_budget_previous_state"], "zero");
    assert_eq!(request_metadata["thinking_budget_final_tokens"], "0");
}

#[tokio::test]
async fn shielded_thinking_policy_raises_smaller_budget_and_adjusts_answer_fields() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"max_tokens":64,"max_completion_tokens":32,"max_output_tokens":16}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(observed_body["max_tokens"], 32_831);
    assert_eq!(observed_body["max_completion_tokens"], 32_799);
    assert_eq!(observed_body["max_output_tokens"], 32_783);
    assert_eq!(
        observed_body["max_tokens"].as_u64().expect("max_tokens")
            - observed_body["thinking"]["budget_tokens"]
                .as_u64()
                .expect("thinking budget"),
        63
    );

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(metadata["thinking_rewrite_reason"], "raised_smaller_budget");
        assert_eq!(metadata["thinking_budget_previous_state"], "smaller");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "1");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32767");
        assert_eq!(
            metadata["thinking_answer_budget_preservation_applied"],
            "true"
        );
        assert_eq!(
            metadata["thinking_answer_budget_adjusted_fields"],
            "max_tokens,max_completion_tokens,max_output_tokens"
        );
    }
}

#[tokio::test]
async fn shielded_thinking_policy_raises_all_known_non_zero_budget_paths_once() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":32768},"extra_body":{"chat_template_kwargs":{"thinking_budget":8}},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert_eq!(observed_body["max_tokens"], 32_824);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "true");
        assert_eq!(metadata["thinking_rewrite_reason"], "raised_smaller_budget");
        assert_eq!(metadata["thinking_budget_previous_state"], "mixed");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "multiple");
        assert_eq!(metadata["thinking_schema_path"], "multiple");
        assert_eq!(metadata["thinking_schema_variant"], "multiple");
        assert_eq!(metadata["thinking_budget_final_tokens"], "32768");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "32760");
        assert_eq!(
            metadata["thinking_budget_observed_paths"],
            "thinking.budget_tokens=equal,extra_body.chat_template_kwargs.thinking_budget=smaller"
        );
        assert_eq!(
            metadata["thinking_budget_rewritten_paths"],
            "extra_body.chat_template_kwargs.thinking_budget"
        );
        assert_eq!(
            metadata["thinking_budget_preserved_paths"],
            "thinking.budget_tokens"
        );
        assert_eq!(metadata["thinking_budget_zero_paths"], "none");
    }
}

#[tokio::test]
async fn shielded_thinking_policy_zero_budget_in_any_known_path_opts_out() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":32768},"extra_body":{"chat_template_kwargs":{"thinking_budget":0}},"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["thinking"]["budget_tokens"], 32_768);
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        0
    );
    assert_eq!(observed_body["max_tokens"], 64);

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    for metadata in [&request_metadata, &attempt_metadata] {
        assert_eq!(metadata["thinking_rewrite_applied"], "false");
        assert_eq!(metadata["thinking_rewrite_reason"], "existing_budget_zero");
        assert_eq!(metadata["thinking_budget_previous_state"], "mixed");
        assert_eq!(metadata["thinking_budget_previous_tokens"], "multiple");
        assert_eq!(metadata["thinking_budget_final_tokens"], "multiple");
        assert_eq!(metadata["thinking_answer_budget_delta_tokens"], "0");
        assert_eq!(
            metadata["thinking_budget_observed_paths"],
            "thinking.budget_tokens=equal,extra_body.chat_template_kwargs.thinking_budget=zero"
        );
        assert_eq!(metadata["thinking_budget_rewritten_paths"], "none");
        assert_eq!(
            metadata["thinking_budget_zero_paths"],
            "extra_body.chat_template_kwargs.thinking_budget"
        );
    }
}

#[tokio::test]
async fn shielded_thinking_policy_updates_extra_body_chat_template_budget() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"extra_body":{"chat_template_kwargs":{"enable_thinking":true,"thinking_budget":8}},"max_completion_tokens":20}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["extra_body"]["chat_template_kwargs"]["thinking_budget"],
        32_768
    );
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_completion_tokens"], 32_780);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(
        request_metadata["thinking_schema_path"],
        "extra_body.chat_template_kwargs.thinking_budget"
    );
    assert_eq!(
        request_metadata["thinking_schema_variant"],
        "extra-body-chat-template-kwargs"
    );
}

#[tokio::test]
async fn shielded_thinking_policy_disabled_leaves_budget_unchanged_except_streaming() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = false
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":64}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("thinking").is_none());
    assert_eq!(observed_body["max_tokens"], 64);
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);

    let (request_metadata, _attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["thinking_policy_enabled"], "false");
    assert_eq!(request_metadata["thinking_rewrite_applied"], "false");
    assert_eq!(
        request_metadata["thinking_rewrite_reason"],
        "policy_disabled"
    );
}

#[tokio::test]
async fn hot_reloaded_thinking_policy_changes_subsequent_rewrites() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
budget_tokens = 1024

[loop_guard]
enabled = false
",
    )
    .await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"reload"}]}"#,
    );

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    let _first_json = shielded_final_json(first).await;
    let first_observed = fake.recv_next().await;
    let first_body: serde_json::Value =
        serde_json::from_slice(&first_observed.body).expect("first body should be JSON");
    assert_eq!(first_body["thinking"]["budget_tokens"], 1024);

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = false
budget_tokens = 1024

[loop_guard]
enabled = false
",
    );
    let disabled_outcome = proxy
        .manager
        .reload()
        .expect("disabled thinking reload should succeed");

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("second proxy request should complete");
    assert_eq!(second.status(), StatusCode::OK);
    let _second_json = shielded_final_json(second).await;
    let second_observed = fake.recv_next().await;
    let second_body: serde_json::Value =
        serde_json::from_slice(&second_observed.body).expect("second body should be JSON");
    assert!(second_body.get("thinking").is_none());

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[thinking]
enabled = true
budget_tokens = 2048

[loop_guard]
enabled = false
",
    );
    let enabled_outcome = proxy
        .manager
        .reload()
        .expect("enabled thinking reload should succeed");

    let third = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("third proxy request should complete");
    assert_eq!(third.status(), StatusCode::OK);
    let _third_json = shielded_final_json(third).await;
    let third_observed = fake.recv_next().await;
    let third_body: serde_json::Value =
        serde_json::from_slice(&third_observed.body).expect("third body should be JSON");
    assert_eq!(third_body["thinking"]["budget_tokens"], 2048);

    assert!(disabled_outcome.applied);
    assert!(enabled_outcome.applied);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_compat_function_call_fields_from_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-function-call",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["service_tier"], "flex");
    assert_eq!(aggregated["choices"][0]["message"]["role"], "assistant");
    assert!(aggregated["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        aggregated["choices"][0]["message"]["function_call"]["name"],
        "legacy_lookup"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["function_call"]["arguments"],
        r#"{"q":"x"}"#
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "function_call");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_compat_refusal_fields_from_sse() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-refusal",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["service_tier"], "flex");
    assert_eq!(
        aggregated["choices"][0]["message"]["refusal"],
        "I cannot help with that"
    );
    assert_eq!(aggregated["choices"][0]["finish_reason"], "stop");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_chat_preserves_malformed_stream_for_upstream_validation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":"false"}"#,
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
        r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#
    );
    let observed = fake.recv_next().await;
    assert_eq!(observed.body, body);
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], "false");
    assert!(observed_body.get("stream_options").is_none());
}

#[tokio::test]
async fn shielded_chat_preserves_malformed_stream_options_for_upstream_validation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream_options":"bad"}"#,
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
        r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#
    );
    let observed = fake.recv_next().await;
    assert_eq!(observed.body, body);
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert!(observed_body.get("stream").is_none());
    assert_eq!(observed_body["stream_options"], "bad");
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_choice_logprobs_from_sse_chunks() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"logprobs":true}"#,
    );

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(
        aggregated["choices"][0]["logprobs"]["content"][0]["token"],
        "Hello"
    );
    assert_eq!(
        aggregated["choices"][0]["logprobs"]["content"][1]["token"],
        "!"
    );
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
    assert_eq!(observed_body["logprobs"], true);
}

#[tokio::test]
async fn shielded_non_stream_chat_preserves_extension_fields_from_sse_chunks() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=compat-extensions",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["object"], "chat.completion");
    assert_eq!(aggregated["provider_metadata"]["phase"], "final");
    assert_eq!(aggregated["x_provider_trace"], "trace-first");
    assert_eq!(
        aggregated["choices"][0]["provider_choice"]["phase"],
        "final"
    );
    assert_eq!(aggregated["choices"][0]["x_choice_trace"], "choice-final");
    assert!(aggregated["choices"][0].get("delta").is_none());
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        aggregated["choices"][0]["message"]["provider_message"]["phase"],
        "final"
    );
    assert_eq!(
        aggregated["choices"][0]["message"]["x_message_trace"],
        "trace-first"
    );
    assert_eq!(aggregated["usage"]["total_tokens"], 5);

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["stream"], true);
    assert_eq!(observed_body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn shielded_chat_attempt_metadata_records_stream_timings_and_delta_counts() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _body = response.text().await.expect("body should be consumed");
    let _observed = fake.recv().await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let metadata_json: String = connection
        .query_row("SELECT response_metadata_json FROM attempts", [], |row| {
            row.get(0)
        })
        .expect("attempt row should exist");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("attempt metadata should be JSON");

    assert_eq!(metadata["shielded_streaming"], "true");
    assert_eq!(metadata["upstream_stream_forced"], "true");
    assert_eq!(
        metadata["upstream_response_header_content-type"],
        "text/event-stream"
    );
    assert_eq!(metadata["finish_reason"], "stop");
    assert_eq!(metadata["delta_count"], "3");
    assert_eq!(metadata["content_delta_count"], "2");
    assert_eq!(metadata["reasoning_delta_count"], "1");
    assert_eq!(metadata["tool_call_delta_count"], "2");
    assert_metadata_latency(&metadata, "first_byte_latency_ms");
    assert_metadata_latency(&metadata, "first_token_latency_ms");
}

#[tokio::test]
async fn default_sse_mode_buffers_non_stream_json_without_sse_framing() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1
",
    )
    .await;

    let response = timeout(
        Duration::from_secs(4),
        proxy
            .client
            .post(format!(
                "{}/v1/chat/completions?test=slow-shielded",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
            .send(),
    )
    .await
    .expect("shielded JSON response should arrive after upstream aggregation")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("shielded default non-stream response should be JSON"),
        "application/json"
    );

    let body = response.text().await.expect("response body should be text");
    assert!(
        !body.starts_with(": llm-guard-proxy heartbeat"),
        "non-stream body must not start with SSE heartbeat: {body:?}"
    );
    assert!(
        !body.contains("event: final"),
        "non-stream body must not contain SSE final event: {body:?}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&body).expect("non-stream body should parse as JSON");
    assert_eq!(json["choices"][0]["message"]["content"], "Hello");

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
}

#[tokio::test]
async fn shielded_liveness_drop_records_current_attempt_abort() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=slow-shielded",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"drop-current"}]}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let heartbeat = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "shielded heartbeat").await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    drop(body);

    let _observed = fake.recv_next().await;
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "1");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:aborted:downstream_body_dropped_before_eof:none"
    );
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "aborted");
    assert_eq!(
        attempts[0].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
}

#[tokio::test]
async fn shielded_liveness_drop_after_prior_retry_records_current_chain() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1

[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = true
"#,
    )
    .await;
    proxy.state.reset_shielded_heartbeat_ticks_for_tests();

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions?test=loop-once-then-slow-success",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"drop-after-retry"}]}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let prefix = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "retry JSON prefix").await;
    assert_eq!(prefix, Bytes::from_static(b" \n"));
    let heartbeat = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "retry heartbeat").await;
    assert_eq!(heartbeat, Bytes::from_static(b" \n"));
    let heartbeat_ticks_before_drop = proxy.state.shielded_heartbeat_ticks_for_tests();
    assert!(
        heartbeat_ticks_before_drop > 0,
        "test must observe at least one shielded heartbeat tick before drop"
    );
    drop(body);
    sleep(Duration::from_millis(1_100)).await;
    assert_eq!(
        proxy.state.shielded_heartbeat_ticks_for_tests(),
        heartbeat_ticks_before_drop,
        "closed shielded liveness body must not keep ticking heartbeats"
    );

    let first_attempt = fake.recv_next().await;
    let second_attempt = fake.recv_next().await;
    assert!(!body_contains_retry_hint(&first_attempt.body));
    assert!(body_contains_retry_hint(&second_attempt.body));
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "cancelled shielded request must not issue a third retry"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.response_metadata["retry_attempt_count"], "2");
    assert_eq!(
        request_row.response_metadata["retry_attempt_chain"],
        "1:retried:loop_guard:loop_detected,2:aborted:downstream_body_dropped_before_eof:none"
    );
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[0].status, "retried");
    assert_eq!(attempts[0].retry_reason.as_deref(), Some("loop_detected"));
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("loop_guard"));
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[1].status, "aborted");
    assert_eq!(
        attempts[1].abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_forwarded_attempt_count_stays(&proxy.sqlite_path, 2).await;
}

#[tokio::test]
async fn shielded_liveness_shutdown_after_terminal_chunk_preserves_success_completion() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        shielded_chat_request(
            "/v1/chat/completions",
            r#"{"model":"test-chat","messages":[{"role":"user","content":"shutdown-after-terminal"}]}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let prefix = next_chunk(
        &mut body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "terminal JSON prefix",
    )
    .await;
    assert_eq!(prefix, Bytes::from_static(b" \n"));
    let final_chunk = next_chunk(&mut body, SHIELDED_HEARTBEAT_TIMEOUT, "terminal JSON body").await;
    let final_json: serde_json::Value =
        serde_json::from_slice(&final_chunk).expect("terminal JSON body should parse");
    assert_eq!(final_json["id"], "chatcmpl-shielded");

    proxy.state.shutdown.begin_shutdown();
    let stream_end = timeout(SHIELDED_HEARTBEAT_TIMEOUT, body.next())
        .await
        .expect("body EOF should arrive before timeout");
    assert!(stream_end.is_none());

    let _observed = fake.recv_next().await;
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "succeeded");
    assert_eq!(request_row.abort_reason, None);
    assert_eq!(attempt_row.status, "succeeded");
    assert_eq!(attempt_row.abort_reason, None);
}

#[tokio::test]
async fn repeated_input_selects_json_whitespace_and_body_stays_parseable() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1
",
    )
    .await;
    let body =
        r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":0.2}"#;

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first shielded request should complete");
    assert_eq!(
        first
            .headers()
            .get(CONTENT_TYPE)
            .expect("first request should use JSON"),
        "application/json"
    );
    let first_json = shielded_final_json(first).await;
    assert_eq!(first_json["id"], "chatcmpl-shielded");
    let _first_observed = fake.recv_next().await;

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second shielded request should complete");
    assert_eq!(
        second
            .headers()
            .get(CONTENT_TYPE)
            .expect("repeated request should switch to JSON"),
        "application/json"
    );
    let second_body = second.text().await.expect("second body should be text");
    assert!(
        second_body.chars().next().is_some_and(char::is_whitespace),
        "JSON whitespace mode should prefix heartbeat whitespace: {second_body:?}"
    );
    let second_json: serde_json::Value =
        serde_json::from_str(&second_body).expect("leading whitespace JSON should parse");
    assert_eq!(second_json["id"], "chatcmpl-shielded");
    let _second_observed = fake.recv_next().await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let rows = connection
        .prepare(
            "SELECT input_fingerprint, downstream_mode, request_metadata_json \
             FROM requests ORDER BY started_at_unix_ms, request_id",
        )
        .expect("request query should prepare")
        .query_map([], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .expect("request query should run")
        .collect::<Result<Vec<_>, _>>()
        .expect("request rows should decode");
    assert_eq!(rows.len(), 2);
    let first_fingerprint = rows[0]
        .0
        .as_ref()
        .expect("first request fingerprint should be recorded");
    let second_fingerprint = rows[1]
        .0
        .as_ref()
        .expect("second request fingerprint should be recorded");
    assert_eq!(first_fingerprint, second_fingerprint);
    assert_eq!(rows[0].1, "non-stream-json");
    assert_eq!(rows[1].1, "non-stream-json");
    let first_metadata: serde_json::Value =
        serde_json::from_str(&rows[0].2).expect("first metadata should parse");
    let second_metadata: serde_json::Value =
        serde_json::from_str(&rows[1].2).expect("second metadata should parse");
    assert_eq!(first_metadata["repeat_input_matched"], "false");
    assert_eq!(first_metadata["downstream_liveness_mode"], "disabled");
    assert_eq!(second_metadata["repeat_input_matched"], "true");
    assert_eq!(
        second_metadata["downstream_liveness_mode"],
        "json-whitespace"
    );
}

#[tokio::test]
async fn hot_reloaded_heartbeat_interval_changes_without_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1

[loop_guard]
enabled = false
"#,
    )
    .await;

    let first = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=slow-shielded",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"first"}]}"#)
        .send()
        .await
        .expect("first shielded request should complete");
    let mut first_body = first.bytes_stream();
    let first_prefix = next_chunk(
        &mut first_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "first JSON prefix",
    )
    .await;
    assert_eq!(first_prefix, Bytes::from_static(b" \n"));
    let first_heartbeat = next_chunk(
        &mut first_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "first interval heartbeat",
    )
    .await;
    assert!(
        first_heartbeat == Bytes::from_static(b" \n"),
        "first JSON whitespace heartbeat should be a JSON-safe whitespace chunk"
    );
    drop(first_body);
    let _first_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 2

[loop_guard]
enabled = false
"#,
    );
    proxy
        .manager
        .reload()
        .expect("heartbeat interval reload should succeed");

    let second = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=slow-shielded",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"second"}]}"#)
        .send()
        .await
        .expect("second shielded request should complete");
    let mut second_body = second.bytes_stream();
    let second_prefix = next_chunk(
        &mut second_body,
        SHIELDED_HEARTBEAT_TIMEOUT,
        "second JSON prefix",
    )
    .await;
    assert_eq!(second_prefix, Bytes::from_static(b" \n"));
    assert!(
        timeout(SHIELDED_RELOAD_GUARD, second_body.next())
            .await
            .is_err(),
        "reloaded two-second heartbeat should not arrive within the old interval"
    );
    let second_heartbeat = next_chunk(
        &mut second_body,
        SHIELDED_RELOAD_TIMEOUT,
        "second interval heartbeat",
    )
    .await;
    assert!(
        second_heartbeat == Bytes::from_static(b" \n"),
        "second JSON whitespace heartbeat should be a JSON-safe whitespace chunk"
    );
    let _second_observed = fake.recv_next().await;
}

#[tokio::test]
async fn hot_reloaded_repeat_window_changes_repeated_detection_without_restart() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[loop_guard]
normalized_input_window_secs = 1
max_repeated_inputs = 1
",
    )
    .await;
    let body = r#"{"model":"test-chat","messages":[{"role":"user","content":"reload-window"}]}"#;

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("first request should complete");
    assert_eq!(
        first
            .headers()
            .get(CONTENT_TYPE)
            .expect("first request should use JSON"),
        "application/json"
    );
    let _first_json = shielded_final_json(first).await;
    let _first_observed = fake.recv_next().await;

    sleep(Duration::from_millis(1_200)).await;

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("second request should complete");
    assert_eq!(
        second
            .headers()
            .get(CONTENT_TYPE)
            .expect("expired repeat should stay JSON"),
        "application/json"
    );
    let _second_json = shielded_final_json(second).await;
    let _second_observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[heartbeat]
interval_secs = 1

[loop_guard]
normalized_input_window_secs = 120
max_repeated_inputs = 1
",
    );
    proxy
        .manager
        .reload()
        .expect("repeat window reload should succeed");

    let third = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("third request should complete");
    assert_eq!(
        third
            .headers()
            .get(CONTENT_TYPE)
            .expect("reloaded repeat window should switch to JSON"),
        "application/json"
    );
    let third_body = third.text().await.expect("third body should be text");
    let third_json: serde_json::Value =
        serde_json::from_str(&third_body).expect("third JSON should parse");
    assert_eq!(third_json["id"], "chatcmpl-shielded");
    let _third_observed = fake.recv_next().await;
}

#[test]
fn normalized_chat_fingerprint_excludes_secrets_and_includes_output_parameters() {
    let base_value = chat_body_with_secret_values("one", false);
    let secret_changed_value = chat_body_with_secret_values("two", true);
    let base = Bytes::from(base_value.to_string().into_bytes());
    let secret_changed = Bytes::from(secret_changed_value.to_string().into_bytes());
    let temperature_changed = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"temperature":0.7,"max_tokens":16,"max_completion_tokens":32,"max_output_tokens":48,"api_key":"sk-one","access_token":"access-one","metadata":{"authorization":"Bearer one","id_token":"id-one"},"stream":false}"#,
    );
    let message_changed = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"pong"}],"temperature":0.2,"max_tokens":16,"max_completion_tokens":32,"max_output_tokens":48,"api_key":"sk-one","access_token":"access-one","metadata":{"authorization":"Bearer one","id_token":"id-one"},"stream":false}"#,
    );

    let normalized =
        normalize_chat_fingerprint_value(base_value).expect("base body should normalize");
    assert_eq!(normalized["max_tokens"], 16);
    assert_eq!(normalized["max_completion_tokens"], 32);
    assert_eq!(normalized["max_output_tokens"], 48);
    assert_eq!(normalized["thinking"]["budget_tokens"], 24);
    assert_normalized_excludes_secret_fields(&normalized);
    assert_text_excludes_values(
        &normalized.to_string(),
        &[
            "sk-one",
            "access-one",
            "refresh-one",
            "api-token-one",
            "auth-token-one",
            "Bearer one",
            "id-one",
            "session-one",
            "bearer-credential-one",
            "password-one",
            "secret-one",
            "credentials-one",
        ],
    );

    let base_fingerprint = chat_input_fingerprint(&base).expect("base fingerprint should compute");
    let secret_fingerprint =
        chat_input_fingerprint(&secret_changed).expect("secret fingerprint should compute");
    let temperature_fingerprint = chat_input_fingerprint(&temperature_changed)
        .expect("temperature fingerprint should compute");
    let message_fingerprint =
        chat_input_fingerprint(&message_changed).expect("message fingerprint should compute");

    assert_eq!(base_fingerprint, secret_fingerprint);
    assert_ne!(base_fingerprint, temperature_fingerprint);
    assert_ne!(base_fingerprint, message_fingerprint);
    assert_text_excludes_values(
        &base_fingerprint,
        &[
            "sk-one",
            "access-one",
            "id-one",
            "Bearer",
            "refresh-one",
            "api-token-one",
            "auth-token-one",
        ],
    );
}

fn chat_body_with_secret_values(suffix: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.2,
        "max_tokens": 16,
        "max_completion_tokens": 32,
        "max_output_tokens": 48,
        "thinking": {
            "budget_tokens": 24
        },
        "api_key": format!("sk-{suffix}"),
        "access_token": format!("access-{suffix}"),
        "refresh_token": format!("refresh-{suffix}"),
        "api_token": format!("api-token-{suffix}"),
        "auth_token": format!("auth-token-{suffix}"),
        "metadata": {
            "authorization": format!("Bearer {suffix}"),
            "id_token": format!("id-{suffix}"),
            "session_token": format!("session-{suffix}"),
            "bearer_credentials": format!("bearer-credential-{suffix}"),
            "password": format!("password-{suffix}"),
            "secret": format!("secret-{suffix}"),
            "credentials": format!("credentials-{suffix}")
        },
        "stream": stream
    })
}

fn assert_normalized_excludes_secret_fields(normalized: &serde_json::Value) {
    assert!(normalized.get("api_key").is_none());
    assert!(normalized.get("access_token").is_none());
    assert!(normalized.get("refresh_token").is_none());
    assert!(normalized.get("api_token").is_none());
    assert!(normalized.get("auth_token").is_none());
    let metadata = normalized
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .expect("metadata should remain after secret fields are stripped");
    for secret_key in [
        "authorization",
        "id_token",
        "session_token",
        "bearer_credentials",
        "password",
        "secret",
        "credentials",
    ] {
        assert!(!metadata.contains_key(secret_key));
    }
}

fn assert_text_excludes_values(text: &str, values: &[&str]) {
    for value in values {
        assert!(!text.contains(value));
    }
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_completion_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_completion_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_max_output_tokens_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("max_output_tokens");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_thinking_budget_tokens_for_repeat_detection() {
    let base_body = chat_body_with_thinking_budget(16);
    let changed_body = chat_body_with_thinking_budget(32);
    assert_budget_change_is_not_repeated(&base_body, &changed_body);
}

#[test]
fn normalized_chat_fingerprint_distinguishes_root_thinking_token_budget_for_repeat_detection() {
    assert_token_budget_change_is_not_repeated("thinking_token_budget");
}

#[test]
fn normalized_chat_fingerprint_distinguishes_extra_body_thinking_token_budget_for_repeat_detection()
{
    let base_body = chat_body_with_extra_body_thinking_token_budget(16);
    let changed_body = chat_body_with_extra_body_thinking_token_budget(32);
    assert_budget_change_is_not_repeated(&base_body, &changed_body);
}

fn assert_token_budget_change_is_not_repeated(field_name: &str) {
    let base_body = chat_body_with_token_budget(field_name, 16);
    let changed_body = chat_body_with_token_budget(field_name, 32);
    assert_budget_change_is_not_repeated(&base_body, &changed_body);
}

fn assert_budget_change_is_not_repeated(base_body: &Bytes, changed_body: &Bytes) {
    let base_fingerprint =
        chat_input_fingerprint(base_body).expect("base fingerprint should compute");
    let changed_fingerprint =
        chat_input_fingerprint(changed_body).expect("changed fingerprint should compute");
    assert_ne!(base_fingerprint, changed_fingerprint);

    let repeat_inputs = RepeatInputCache::default();
    let first_observation = repeat_inputs.observe(&base_fingerprint, 1_000, 120, 1);
    let changed_observation = repeat_inputs.observe(&changed_fingerprint, 2_000, 120, 1);
    let repeated_base_observation = repeat_inputs.observe(&base_fingerprint, 3_000, 120, 1);

    assert_eq!(first_observation, RepeatInputObservation::default());
    assert_eq!(changed_observation, RepeatInputObservation::default());
    assert_eq!(
        repeated_base_observation,
        RepeatInputObservation {
            repeated: true,
            prior_count: 1
        }
    );
}

fn chat_body_with_token_budget(field_name: &str, value: u64) -> Bytes {
    let mut body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.2,
        "stream": false
    });
    body.as_object_mut()
        .expect("test body should be an object")
        .insert(field_name.to_owned(), serde_json::Value::from(value));
    Bytes::from(body.to_string().into_bytes())
}

fn chat_body_with_thinking_budget(value: u64) -> Bytes {
    Bytes::from(
        serde_json::json!({
            "model": "test-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.2,
            "thinking": {
                "budget_tokens": value
            },
            "stream": false
        })
        .to_string()
        .into_bytes(),
    )
}

fn chat_body_with_extra_body_thinking_token_budget(value: u64) -> Bytes {
    Bytes::from(
        serde_json::json!({
            "model": "test-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.2,
            "extra_body": {
                "thinking_token_budget": value
            },
            "stream": false
        })
        .to_string()
        .into_bytes(),
    )
}

#[tokio::test]
async fn hot_reloaded_disabled_shielding_falls_back_to_generic_chat_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body = Bytes::from_static(
        br#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
    );

    let first = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("first proxy request should complete");
    assert_eq!(first.status(), StatusCode::OK);
    let _first_body = first.text().await.expect("first body should be text");
    let first_observed = fake.recv_next().await;
    let first_body: serde_json::Value =
        serde_json::from_slice(&first_observed.body).expect("first upstream body should be JSON");
    assert_eq!(first_body["stream"], true);

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[shielding]
enabled = false
",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("shielding reload should succeed");

    let second = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .expect("second proxy request should complete");

    assert!(outcome.applied);
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(
        second.text().await.expect("second body should be text"),
        r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#
    );
    let second_observed = fake.recv_next().await;
    assert_eq!(second_observed.body, body);
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
async fn context_budget_preflight_allows_equal_window_and_rejects_chat_overflow() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[upstream.metadata]
context_length_override = 6
input_token_safety_margin = 1

[thinking]
mode = "passthrough"
"#,
    )
    .await;

    let allowed = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"a b c"}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("allowed proxy request should complete");
    assert_eq!(allowed.status(), StatusCode::OK);
    let _allowed_json = shielded_final_json(allowed).await;
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");

    let rejected = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"a b c d"}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("rejected proxy request should complete");
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    let error = response_json(rejected).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "messages");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message should be string")
            .contains("auto-compaction")
    );
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    assert_eq!(attempt_count, 1);
    let rejected_metadata_json: String = connection
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("rejected request metadata should be readable");
    let rejected_metadata: serde_json::Value =
        serde_json::from_str(&rejected_metadata_json).expect("metadata should parse");
    assert_eq!(rejected_metadata["context_budget_preflight"], "rejected");
    assert_eq!(rejected_metadata["context_budget_param"], "messages");
    assert_eq!(rejected_metadata["context_budget_window_tokens"], "6");
    assert_eq!(
        rejected_metadata["context_budget_total_estimate_tokens"],
        "7"
    );
    assert_eq!(rejected_metadata["upstream_profile"], "default");
    assert!(!rejected_metadata.to_string().contains("a b c d"));
}

#[tokio::test]
async fn context_budget_preflight_counts_chat_tool_definitions_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
[upstream.metadata]
context_length_override = 6

[thinking]
mode = "passthrough"
"#,
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ok"}],"tools":[{"type":"function","function":{"name":"lookup","description":"one two three four five six","parameters":{"type":"object","properties":{"city":{"type":"string","description":"target city"}},"required":["city"]}}}],"max_tokens":1}"#,
        )
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "messages");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let rejected_metadata_json: String = connection
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("rejected request metadata should be readable");
    let rejected_metadata: serde_json::Value =
        serde_json::from_str(&rejected_metadata_json).expect("metadata should parse");
    assert_eq!(rejected_metadata["context_budget_preflight"], "rejected");
    assert_eq!(rejected_metadata["context_budget_param"], "messages");
    assert!(
        rejected_metadata["context_budget_total_estimate_tokens"]
            .as_str()
            .and_then(|tokens| tokens.parse::<u64>().ok())
            .is_some_and(|tokens| tokens > 6)
    );
    assert!(!rejected_metadata_json.contains("one two three four five six"));
    assert!(!rejected_metadata_json.contains("target city"));
}

#[tokio::test]
async fn context_budget_preflight_rejects_completions_prompt_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
context_length_override = 3
",
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-completion","prompt":"one two three","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "prompt");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn context_budget_preflight_rejects_unbroken_prompt_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r"
[upstream.metadata]
context_length_override = 100
",
    )
    .await;
    let long_prompt = "x".repeat(1_000);
    let body = format!(r#"{{"model":"test-completion","prompt":"{long_prompt}","max_tokens":1}}"#);

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = response_json(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert_eq!(error["error"]["param"], "prompt");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
}

#[tokio::test]
async fn hot_reloaded_profile_safety_margin_changes_context_preflight() {
    let mut fake = FakeUpstream::spawn().await;
    let fake_base_url = fake.base_url.clone();
    let profile_config = |safety_margin: u32| {
        format!(
            r#"
[[upstreams]]
name = "aeon-chat"
base_url = "{fake_base_url}"
match_models = ["aeon-ultimate"]

[upstreams.metadata]
context_length_override = 6
input_token_safety_margin = {safety_margin}

[upstreams.thinking]
mode = "passthrough"
"#,
        )
    };
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &profile_config(0),
    )
    .await;
    let body = r#"{"model":"aeon-ultimate","messages":[{"role":"user","content":"a b c d"}],"max_tokens":1}"#;

    let allowed = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("allowed proxy request should complete");
    assert_eq!(allowed.status(), StatusCode::OK);
    let _allowed_json = shielded_final_json(allowed).await;
    let _observed = fake.recv_next().await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &profile_config(1),
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("profile safety margin reload should succeed");
    assert!(outcome.applied);
    assert!(outcome.restart_required_changes.is_empty());

    let rejected = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("rejected proxy request should complete");
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    let error = response_json(rejected).await;
    assert_eq!(error["error"]["code"], "context_budget_exceeded");
    assert!(fake.recv_within(Duration::from_millis(100)).await.is_none());
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
async fn upstream_redirects_are_forwarded_without_following() {
    for redirect_status in [
        StatusCode::TEMPORARY_REDIRECT,
        StatusCode::PERMANENT_REDIRECT,
    ] {
        let mut target = RedirectTarget::spawn().await;
        let upstream =
            RedirectingUpstream::spawn(redirect_status, target.capture_url.clone()).await;
        let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;
        let body = Bytes::from_static(
            br#"{"model":"test-chat","messages":[{"role":"user","content":"secret prompt"}]}"#,
        );

        let response = proxy
            .client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(body.clone())
            .send()
            .await
            .expect("proxy request should complete");

        assert_eq!(response.status(), redirect_status);
        assert_eq!(
            response
                .headers()
                .get(LOCATION)
                .expect("redirect location should be forwarded"),
            target.capture_url.as_str()
        );
        assert_eq!(
            response.text().await.expect("body should be text"),
            "redirected"
        );

        let observed = upstream.recv().await;
        assert_eq!(observed.method, Method::POST);
        assert_eq!(observed.path_and_query, "/v1/chat/completions");
        let observed_body: serde_json::Value = serde_json::from_slice(&observed.body)
            .expect("redirected upstream body should be JSON");
        assert_eq!(observed_body["model"], "test-chat");
        assert_eq!(observed_body["messages"][0]["content"], "secret prompt");
        assert_eq!(observed_body["stream"], true);
        assert!(
            target
                .recv_within(Duration::from_millis(100))
                .await
                .is_none(),
            "proxy must not follow upstream redirects or replay the prompt body"
        );
    }
}

#[tokio::test]
async fn sse_response_streams_first_chunk_before_upstream_completion() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .post(format!("{}/v1/chat/completions?test=sse", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":"test-chat","messages":[],"stream":true}"#)
            .send(),
    )
    .await
    .expect("proxy should return SSE headers before delayed upstream completion")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be forwarded"),
        "text/event-stream"
    );

    let mut body = response.bytes_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first SSE chunk").await;
    assert_eq!(first, Bytes::from_static(SSE_FIRST_CHUNK));
    assert!(
        timeout(STREAM_SECOND_CHUNK_GUARD, body.next())
            .await
            .is_err(),
        "second SSE chunk arrived before the upstream delay elapsed"
    );
    let second = next_chunk(&mut body, STREAM_COMPLETION_TIMEOUT, "second SSE chunk").await;
    assert_eq!(second, Bytes::from_static(SSE_SECOND_CHUNK));
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("SSE stream end should arrive after delayed chunk")
            .is_none()
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::POST);
    assert_eq!(observed.path_and_query, "/v1/chat/completions?test=sse");
}

#[tokio::test]
async fn long_json_response_streams_first_chunk_while_upstream_remains_open() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = timeout(
        STREAM_HEADER_TIMEOUT,
        proxy
            .client
            .get(format!("{}/v1/embeddings?test=long-json", proxy.base_url))
            .send(),
    )
    .await
    .expect("proxy should return JSON headers before delayed upstream completion")
    .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .expect("content type should be forwarded"),
        "application/json"
    );

    let mut body = response.bytes_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first JSON chunk").await;
    assert_eq!(first, Bytes::from_static(LONG_JSON_FIRST_CHUNK));
    assert!(
        timeout(STREAM_SECOND_CHUNK_GUARD, body.next())
            .await
            .is_err(),
        "second JSON chunk arrived before the upstream delay elapsed"
    );
    let second = next_chunk(&mut body, STREAM_COMPLETION_TIMEOUT, "second JSON chunk").await;
    assert_eq!(second, Bytes::from_static(LONG_JSON_SECOND_CHUNK));
    assert!(
        timeout(STREAM_COMPLETION_TIMEOUT, body.next())
            .await
            .expect("JSON stream end should arrive after delayed chunk")
            .is_none()
    );

    let observed = fake.recv().await;
    assert_eq!(observed.method, Method::GET);
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");
}

#[tokio::test]
async fn generic_stream_timeout_records_failed_request_and_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        "request_timeout_ms = 100\n",
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");

    let mut body = response.into_body().into_data_stream();
    let first = next_chunk(&mut body, STREAM_FIRST_CHUNK_TIMEOUT, "first JSON chunk").await;
    assert_eq!(first, Bytes::from_static(LONG_JSON_FIRST_CHUNK));
    let timeout_item = timeout(Duration::from_secs(1), body.next())
        .await
        .expect("upstream timeout should surface before the delayed second chunk")
        .expect("body stream should yield an upstream timeout item");
    assert!(
        timeout_item.is_err(),
        "delayed upstream body should fail under the configured timeout"
    );

    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure")),
        "request error should use bounded timeout kind: {:?}",
        request_row.error_reason
    );
    assert!(
        attempt_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure")),
        "attempt error should use bounded timeout kind: {:?}",
        attempt_row.error_reason
    );
}

#[tokio::test]
async fn shielded_upstream_timeout_returns_bounded_gateway_error() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        r#"
request_timeout_ms = 100

[heartbeat]
mode = "disabled"

[retry]
enabled = false
"#,
    )
    .await;

    let response = Box::pin(timeout(
        Duration::from_secs(2),
        proxy_handler(
            State(proxy.state.clone()),
            shielded_chat_request(
                "/v1/chat/completions?test=slow-shielded",
                r#"{"model":"test-chat","messages":[{"role":"user","content":"timeout"}]}"#,
            ),
        ),
    ))
    .await
    .expect("shielded timeout response should be bounded");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("timeout error body should read");
    let body = String::from_utf8(body.to_vec()).expect("timeout error body should be UTF-8");
    assert!(body.contains("llm_guard_attempt_timeout"));
    assert!(body.contains("timeout_failure"));
    assert_safe_operational_text("shielded timeout body", &body);

    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=slow-shielded"
    );
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);
    assert_eq!(request_row.status, "failed");
    assert_eq!(attempt_row.status, "failed");
    assert!(
        request_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure"))
    );
    assert!(
        attempt_row
            .error_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("timeout_failure"))
    );
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
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"id":"cmpl-test","object":"text_completion"}"#
    );
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
#[cfg(feature = "guard")]
async fn valid_key_resolves_to_correct_profile() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("use_default_profile"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer vk_child_def456")
        .body(r#"{"model":"child-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake.recv().await;
    let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
    assert_eq!(request_metadata["caller_profile"], "child_safe");
    assert_eq!(request_metadata["virtual_key_resolution"], "matched");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn unknown_key_uses_default() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("use_default_profile"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer vk_unknown")
        .body(r#"{"model":"gpt-default","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake.recv().await;
    let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
    assert_eq!(request_metadata["caller_profile"], "default");
    assert_eq!(
        request_metadata["virtual_key_resolution"],
        "unknown_use_default"
    );
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn unknown_key_fails_closed() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("fail_closed"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer vk_unknown")
        .body(r#"{"model":"gpt-default","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_no_upstream_request(&mut fake).await;
    let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
    assert_eq!(request_metadata["virtual_key_resolution"], "fail_closed");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn no_key_single_profile_uses_default() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        single_profile_virtual_key_config(),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"solo-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake.recv().await;
    let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
    assert_eq!(request_metadata["caller_profile"], "solo");
    assert_eq!(
        request_metadata["virtual_key_resolution"],
        "single_profile_default"
    );
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn x_virtual_key_header_precedence() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("fail_closed"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer vk_child_def456")
        .header("X-Virtual-Key", "vk_adult_abc123")
        .body(r#"{"model":"adult-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake.recv().await;
    let request_metadata = read_last_request_metadata(&proxy.sqlite_path);
    assert_eq!(request_metadata["caller_profile"], "default");
    assert_eq!(request_metadata["virtual_key_resolution"], "matched");
}

#[tokio::test]
#[cfg(feature = "guard")]
async fn key_not_logged_in_audit() {
    let secret_key = "vk_child_def456";
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &virtual_key_config("use_default_profile"),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .header("X-Virtual-Key", secret_key)
        .body(r#"{"model":"child-model","prompt":"ping","max_tokens":1}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _observed = fake.recv().await;
    let persisted_metadata = read_last_request_metadata_json(&proxy.sqlite_path);
    assert!(!persisted_metadata.contains(secret_key));
    assert!(persisted_metadata.contains("child_safe"));
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
    assert_eq!(
        response.text().await.expect("body should be text"),
        r#"{"object":"list","data":[]}"#
    );
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

#[tokio::test]
async fn invalid_openai_path_writes_failed_request_without_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = send_raw_proxy_get(&proxy.base_url, "/v1/../admin").await;
    let response_request_id = raw_response_header(&response, "x-request-id")
        .expect("terminal response should include x-request-id");

    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request"),
        "dot-segment target should be rejected: {response}"
    );
    assert!(
        response.contains("invalid_request_path"),
        "error body should identify the path validation failure: {response}"
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");

    assert_response_request_id_matches_persisted_request(response_request_id, &proxy);
    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 400);
    assert!(request_row.2.contains("invalid_request_path"));
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/../admin");
    assert_eq!(request_metadata["query_present"], "false");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(request_metadata["request_body_bytes"], "unknown");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn upstream_transport_failure_writes_failed_request_and_attempt() {
    let upstream = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, true).await;

    let response = proxy
        .client
        .get(format!(
            "{}/v1/models?api_key=sk-live&safe=ok",
            proxy.base_url
        ))
        .send()
        .await
        .expect("proxy request should complete with gateway error");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_request_id = terminal_response_request_id(response.headers());
    let body = response.text().await.expect("body should be text");
    assert!(
        body.contains("upstream_transport_error"),
        "gateway error should identify upstream transport failure: {body}"
    );
    assert_sensitive_query_absent("transport failure response body", &body);

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM requests", [], |row| row.get(0))
        .expect("request count should be readable");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_row: (String, i64, String, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json, response_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .expect("failed request row should exist");
    let attempt_row: (
        String,
        Option<i64>,
        String,
        String,
        String,
        Option<i64>,
        i64,
        i64,
    ) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json, response_metadata_json, duration_ms, started_at_unix_ms, finished_at_unix_ms FROM attempts",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )
        .expect("failed attempt row should exist");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");
    let request_response_metadata: serde_json::Value =
        serde_json::from_str(&request_row.4).expect("request response metadata should be json");
    let attempt_metadata: serde_json::Value =
        serde_json::from_str(&attempt_row.3).expect("attempt metadata should be json");
    let attempt_response_metadata: serde_json::Value =
        serde_json::from_str(&attempt_row.4).expect("attempt response metadata should be json");

    assert_eq!(request_count, 1);
    assert_eq!(attempt_count, 1);
    assert_response_request_id_matches_persisted_request(&response_request_id, &proxy);
    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 502);
    assert!(request_row.2.contains("upstream_transport_error"));
    assert_sensitive_query_absent("request error_reason", &request_row.2);
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/models");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], "0");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(
        request_response_metadata["error_type"],
        "upstream_transport_error"
    );
    assert_eq!(attempt_row.0, "failed");
    assert_eq!(attempt_row.1, None);
    assert!(attempt_row.2.contains("upstream_transport_error"));
    assert_sensitive_query_absent("attempt error_reason", &attempt_row.2);
    assert_eq!(attempt_metadata["method"], "GET");
    assert_eq!(attempt_metadata["path"], "/v1/models");
    assert_eq!(attempt_metadata["query_present"], "true");
    assert_eq!(attempt_metadata["attempt_number"], "1");
    assert_eq!(
        attempt_response_metadata["upstream_response_received"],
        "false"
    );
    assert!(attempt_row.5.is_some());
    assert!(attempt_row.7 >= attempt_row.6);
}

#[tokio::test]
async fn oversized_body_failure_writes_failed_request_without_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;
    let body_len = MAX_PROXY_BODY_BYTES + 1;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?oversize=true")
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, body_len.to_string())
        .body(Body::from(vec![b'a'; body_len]))
        .expect("oversized request should build");

    let response = proxy_handler(State(proxy.state.clone()), request).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let response_body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("error response body should read");
    let response_body =
        String::from_utf8(response_body.to_vec()).expect("error response should be utf-8");
    assert!(
        response_body.contains("request_body_error"),
        "error should identify body read failure: {response_body}"
    );
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");
    let body_len = body_len.to_string();

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 413);
    assert!(request_row.2.contains("request_body_error"));
    assert_eq!(request_metadata["method"], "POST");
    assert_eq!(request_metadata["path"], "/v1/completions");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], body_len.as_str());
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn queued_generation_request_cancellation_does_not_buffer_or_forward_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream and hold the only permit");
    assert_eq!(first_observed.method, Method::GET);
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let second_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"large"}"#))
        }
    }));
    let second_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?blocked=true")
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, MAX_PROXY_BODY_BYTES.to_string())
        .body(second_body)
        .expect("second request should build");
    let second = tokio::spawn(proxy_handler(State(proxy.state.clone()), second_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queued requests must not be body-buffered before permit admission"
    );
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second request should still be waiting for capacity"
    );

    second.abort();
    match second.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "queued request should be cancelled before upstream dispatch, got {}",
            response.status()
        ),
    }
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "cancelled queued request must not poll its body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(first_response);
}

#[tokio::test]
async fn saturated_generation_requests_wait_for_in_flight_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=one"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream and hold the only permit");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=one"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let second_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"queued"}"#))
        }
    }));
    let second_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?slot=queued")
        .header(CONTENT_TYPE, "application/json")
        .body(second_body)
        .expect("second request should build");
    let second = tokio::spawn(proxy_handler(State(proxy.state.clone()), second_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queued requests must not be body-buffered before capacity is available"
    );
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second request should wait for capacity instead of returning a 503"
    );

    drop(first_response);

    let second_response = second
        .await
        .expect("queued request task should complete after capacity is released");
    assert_eq!(second_response.status(), StatusCode::OK);
    assert!(body_polled.load(Ordering::SeqCst));
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("queued request should reach upstream after capacity is available");
    assert_eq!(second_observed.method, Method::POST);
    assert_eq!(
        second_observed.path_and_query,
        "/v1/completions?slot=queued"
    );
}

#[tokio::test]
async fn restricted_embedding_listener_accepts_embedding_and_rejects_other_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let state = proxy.state.for_listener(listener);

    let accepted = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/embeddings",
            br#"{"model":"embedding-model","input":"hello"}"#,
        ),
    )
    .await;
    assert_eq!(accepted.status(), StatusCode::OK);
    let _body = to_bytes(accepted.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("accepted body should read");
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/embeddings");

    let rejected_chat = proxy_handler(
        State(state.clone()),
        json_post_request(
            "/v1/chat/completions",
            br#"{"model":"chat-model","messages":[{"role":"user","content":"hello"}]}"#,
        ),
    )
    .await;
    assert_eq!(rejected_chat.status(), StatusCode::BAD_REQUEST);
    let rejected_chat_body = to_bytes(rejected_chat.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rejection body should read");
    let rejected_chat_json: serde_json::Value =
        serde_json::from_slice(&rejected_chat_body).expect("rejection should be JSON");
    assert_eq!(
        rejected_chat_json["error"]["type"],
        "listener_upstream_not_allowed"
    );

    let rejected_rerank = proxy_handler(
        State(state),
        json_post_request(
            "/v1/rerank",
            br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
        ),
    )
    .await;
    assert_eq!(rejected_rerank.status(), StatusCode::BAD_REQUEST);
    let _body = to_bytes(rejected_rerank.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rerank rejection body should read");
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn restricted_listener_denial_bounds_untrusted_model_in_response_and_metadata() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let oversized_model = format!("chat-model-{}", "x".repeat(4096));
    let body = format!(
        r#"{{"model":"{oversized_model}","messages":[{{"role":"user","content":"hello"}}]}}"#
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("chat request should build");

    let response = proxy_handler(State(proxy.state.for_listener(listener)), request).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rejection body should read");
    let response_body =
        String::from_utf8(response_body.to_vec()).expect("rejection body should be utf-8");
    let rejection: serde_json::Value =
        serde_json::from_str(&response_body).expect("rejection should be JSON");
    assert_eq!(rejection["error"]["type"], "listener_upstream_not_allowed");
    assert!(response_body.len() < 1024);
    assert!(!response_body.contains(&oversized_model));
    assert_no_upstream_request(&mut fake).await;

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (Option<String>, String, String) = connection
        .query_row(
            "SELECT model_id, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("denied request row should exist");
    let persisted = format!("{} {}", request_row.0.unwrap_or_default(), request_row.1);
    assert!(persisted.len() < 1024);
    assert!(!persisted.contains(&oversized_model));
    assert!(!request_row.2.contains(&oversized_model));
}

#[tokio::test]
async fn aggregate_listener_accepts_chat_embeddings_and_rerank_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "aggregate");
    let state = proxy.state.for_listener(listener);

    for (request, expected_path) in [
        (
            json_post_request(
                "/v1/chat/completions",
                br#"{"model":"chat-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            ),
            "/v1/chat/completions",
        ),
        (
            json_post_request(
                "/v1/embeddings",
                br#"{"model":"embedding-model","input":"hello"}"#,
            ),
            "/v1/embeddings",
        ),
        (
            json_post_request(
                "/v1/rerank",
                br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
            ),
            "/v1/rerank",
        ),
    ] {
        let response = proxy_handler(State(state.clone()), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("response body should read");
        let observed = fake.recv_next().await;
        assert_eq!(observed.path_and_query, expected_path);
    }
}

#[tokio::test]
async fn restricted_models_request_filters_to_listener_reachable_models() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model"]);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_filters_when_metadata_enrichment_is_disabled() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[upstreams.metadata]
discovery_enabled = false
enrich_responses = false

[[upstreams]]
name = "rerank"
base_url = "{0}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["embedding"]
"#,
            fake.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model"]);
    assert_eq!(json["data"][0].get("context_length"), None);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_filters_to_all_allowed_listener_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");
    let state = proxy.state.for_listener(listener);

    let response = proxy_handler(
        State(state.clone()),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    let observed = fake.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=multi-listener-models"
    );

    for (request, expected_path) in [
        (
            json_post_request(
                "/v1/embeddings",
                br#"{"model":"embedding-model","input":"hello"}"#,
            ),
            "/v1/embeddings",
        ),
        (
            json_post_request(
                "/v1/rerank",
                br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
            ),
            "/v1/rerank",
        ),
    ] {
        let response = proxy_handler(State(state.clone()), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("response body should read");
        let observed = fake.recv_next().await;
        assert_eq!(observed.path_and_query, expected_path);
    }
}

#[tokio::test]
async fn aggregate_models_request_fetches_all_configured_upstream_profiles() {
    let mut chat = FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_CHAT_MODELS_BODY).await;
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &chat.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "aggregate");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert!(model_ids.contains(&"chat-model"));
    assert!(model_ids.contains(&"embedding-model"));
    assert!(model_ids.contains(&"rerank-model"));
    let observed = chat.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let observed = embedding.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let observed = rerank.recv_next().await;
    assert_eq!(
        observed.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_fetches_and_merges_distinct_allowed_upstreams() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let models = json["data"].as_array().expect("data should be an array");
    let model_ids = models
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    assert_eq!(models[0]["first"], "embedding");
    assert!(model_ids.iter().all(|model_id| *model_id != "chat-model"));
    let embedding_request = embedding.recv_next().await;
    assert_eq!(
        embedding_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let rerank_request = rerank.recv_next().await;
    assert_eq!(
        rerank_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn merged_models_enrichment_uses_each_allowed_profile_metadata_config() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_ONLY_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_ONLY_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[upstreams.metadata]
discovery_enabled = false
enrich_responses = false

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[upstreams.metadata]
context_length_override = 12345

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let models = json["data"].as_array().expect("data should be an array");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["id"], "embedding-model");
    assert_eq!(models[0].get("context_length"), None);
    assert_eq!(models[1]["id"], "rerank-model");
    assert_normalized_context_fields(&models[1], 12_345);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_records_distinct_observability_attempts_per_upstream() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]
"#,
            embedding.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "rerank-model"]);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_ne!(attempts[0].attempt_id, attempts[1].attempt_id);
    assert_eq!(attempts[0].request_id, attempts[1].request_id);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[1].status, "succeeded");
    assert_eq!(
        attempts[0].request_metadata["upstream_profile"],
        "embedding"
    );
    assert_eq!(attempts[1].request_metadata["upstream_profile"], "rerank");
}

#[tokio::test]
async fn restricted_models_request_skips_invalid_first_body_when_merging_distinct_upstreams() {
    let mut invalid = FakeUpstream::spawn_with_models_body(r#"{"error":"not a model list"}"#).await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &invalid.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "invalid"
base_url = "{0}"
match_models = ["invalid-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "invalid-rerank"
bind_host = "127.0.0.1"
port = 18006
allowed_upstreams = ["invalid", "rerank"]
"#,
            invalid.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "invalid-rerank");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["rerank-model"]);
    let invalid_request = invalid.recv_next().await;
    assert_eq!(
        invalid_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let rerank_request = rerank.recv_next().await;
    assert_eq!(
        rerank_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn merged_models_response_uses_proxy_owned_success_headers_for_valid_body() {
    let mut rate_limited = FakeUpstream::spawn_with_models_response(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{"type":"rate_limit","message":"slow down"}}"#,
        "rate-limited-models",
    )
    .await;
    let mut rerank =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_RERANK_MODELS_BODY).await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &rate_limited.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "rate-limited"
base_url = "{0}"
match_models = ["rate-limited-model"]

[[upstreams]]
name = "rerank"
base_url = "{1}"
match_models = ["rerank-model"]

[[listeners]]
name = "mixed-models"
bind_host = "127.0.0.1"
port = 18006
allowed_upstreams = ["rate-limited", "rerank"]
"#,
            rate_limited.base_url, rerank.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "mixed-models");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(response.headers().get(RETRY_AFTER), None);
    assert_eq!(response.headers().get("x-upstream-endpoint"), None);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["rerank-model"]);
    assert_eq!(
        rate_limited.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    assert_eq!(
        rerank.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn restricted_models_request_merges_implicit_default_and_named_allowed_upstream() {
    let mut shared = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &shared.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[listeners]]
name = "default-embedding"
bind_host = "127.0.0.1"
port = 18007
allowed_upstreams = ["default", "embedding"]
"#,
            shared.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "default-embedding");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(
        model_ids,
        vec!["chat-model", "embedding-model", "rerank-model"]
    );
    let default_request = shared.recv_next().await;
    assert_eq!(
        default_request.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_excludes_same_base_url_models_routed_to_disallowed_profiles() {
    let mut shared = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &shared.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{0}"
match_models = ["rerank-model"]

[[listeners]]
name = "default-embedding"
bind_host = "127.0.0.1"
port = 18007
allowed_upstreams = ["default", "embedding"]
"#,
            shared.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "default-embedding");
    let state = proxy.state.for_listener(listener);

    let response = proxy_handler(
        State(state.clone()),
        empty_get_request("/v1/models?test=multi-listener-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["chat-model", "embedding-model"]);
    let response = proxy_handler(
        State(state),
        json_post_request(
            "/v1/rerank",
            br#"{"model":"rerank-model","query":"hello","documents":["hello"]}"#,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let default_request = shared.recv_next().await;
    assert_eq!(
        default_request.path_and_query,
        "/v1/models?test=multi-listener-models"
    );
}

#[tokio::test]
async fn restricted_models_request_uses_each_upstream_profile_timeout_when_merging() {
    let mut fast =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let mut slow = FakeUpstream::spawn_with_models_body_and_delay(
        DISTINCT_UPSTREAM_SLOW_MODELS_BODY,
        Duration::from_millis(150),
    )
    .await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fast.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "fast"
base_url = "{0}"
match_models = ["embedding-model"]
request_timeout_ms = 50

[[upstreams]]
name = "slow"
base_url = "{1}"
match_models = ["slow-model"]
request_timeout_ms = 1000

[[listeners]]
name = "fast-slow"
bind_host = "127.0.0.1"
port = 18008
allowed_upstreams = ["fast", "slow"]
"#,
            fast.base_url, slow.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "fast-slow");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models body should read");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("models should be JSON");
    let model_ids = json["data"]
        .as_array()
        .expect("data should be an array")
        .iter()
        .map(|model| model["id"].as_str().expect("model id should be string"))
        .collect::<Vec<_>>();

    assert_eq!(model_ids, vec!["embedding-model", "slow-model"]);
    let fast_request = fast.recv_next().await;
    assert_eq!(
        fast_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
    let slow_request = slow.recv_next().await;
    assert_eq!(
        slow_request.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );
}

#[tokio::test]
async fn observability_records_listener_and_selected_upstream_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &multi_listener_profile_config(&fake.base_url),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-legacy");
    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        json_post_request(
            "/v1/embeddings",
            br#"{"model":"embedding-model","input":"hello"}"#,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("response body should read");
    let _observed = fake.recv_next().await;

    let (request_metadata, attempt_metadata) = read_single_request_and_attempt_metadata(&proxy);
    assert_eq!(request_metadata["listener_name"], "embedding-legacy");
    assert_eq!(request_metadata["listener_port"], "18002");
    assert_eq!(request_metadata["upstream_profile"], "embedding");
    assert_eq!(attempt_metadata["listener_name"], "embedding-legacy");
    assert_eq!(attempt_metadata["upstream_profile"], "embedding");
}

#[tokio::test]
async fn matched_upstream_profile_has_independent_generation_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 0
generation_queue_timeout_ms = 50

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let default_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=default"),
    )
    .await;
    assert_eq!(default_response.status(), StatusCode::OK);
    let default_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should hold the default generation capacity");
    assert_eq!(
        default_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=default"
    );

    let first_embedding = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=embedding-one"),
    )
    .await;
    let second_embedding = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=embedding-two"),
    )
    .await;

    assert_eq!(first_embedding.status(), StatusCode::OK);
    assert_eq!(second_embedding.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first matched-profile request should reach upstream despite default saturation");
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("second matched-profile request should use profile capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding-one"
    );
    assert_eq!(
        second_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding-two"
    );

    drop(first_embedding);
    drop(second_embedding);
    drop(default_response);
}

#[tokio::test]
async fn profile_limit_mode_bounds_body_routing_before_buffering() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let (active_request, active_body_polled) =
        tracked_pending_json_request("/v1/completions?slot=active");
    let active = tokio::spawn(proxy_handler(State(proxy.state.clone()), active_request));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        active_body_polled.load(Ordering::SeqCst),
        "active routing request should hold the body-routing permit while reading its body"
    );

    let (queued_request, queued_body_polled) =
        tracked_pending_json_request("/v1/completions?slot=queued");
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "queued routing request must not read its body before routing capacity is available"
    );
    assert!(
        !queued.is_finished(),
        "queued routing request should occupy the bounded routing queue"
    );

    let (overflow_request, overflow_body_polled) =
        tracked_json_request("/v1/completions?slot=overflow", br#"{"prompt":"overflow"}"#);
    let overflow_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    ))
    .await
    .expect("routing queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let overflow_body = to_bytes(overflow_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("routing queue-full body should read");
    let overflow_body =
        String::from_utf8(overflow_body.to_vec()).expect("routing queue-full body should be utf-8");
    assert!(
        overflow_body.contains("proxy_generation_queue_full"),
        "routing queue-full error should identify admission failure: {overflow_body}"
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "routing queue-full rejection must not read the overflow body"
    );
    assert_no_upstream_request(&mut fake).await;

    queued.abort();
    active.abort();
    assert!(
        queued
            .await
            .expect_err("queued request should be aborted")
            .is_cancelled()
    );
    assert!(
        active
            .await
            .expect_err("active request should be aborted")
            .is_cancelled()
    );
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "aborted queued routing request must not read its body"
    );
}

#[tokio::test]
async fn merged_models_request_records_successful_prior_attempt_when_later_fetch_fails() {
    let mut embedding =
        FakeUpstream::spawn_with_models_body(DISTINCT_UPSTREAM_EMBEDDING_MODELS_BODY).await;
    let broken = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &embedding.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[[upstreams]]
name = "embedding"
base_url = "{0}"
match_models = ["embedding-model"]

[[upstreams]]
name = "broken"
base_url = "{1}"
match_models = ["broken-model"]
request_timeout_ms = 50

[[listeners]]
name = "embedding-broken"
bind_host = "127.0.0.1"
port = 18010
allowed_upstreams = ["embedding", "broken"]
"#,
            embedding.base_url, broken.base_url
        ),
    )
    .await;
    let listener = listener_config(&proxy, "embedding-broken");

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        embedding.recv_next().await.path_and_query,
        "/v1/models?test=distinct-multi-upstream-models"
    );

    let attempts = read_attempt_request_metadata_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].attempt_number, 1);
    assert_eq!(attempts[1].attempt_number, 2);
    assert_eq!(attempts[0].status, "succeeded");
    assert_eq!(attempts[1].status, "failed");
    assert_ne!(attempts[0].attempt_id, attempts[1].attempt_id);
    assert_eq!(attempts[0].request_id, attempts[1].request_id);
    assert_eq!(
        attempts[0].request_metadata["upstream_profile"],
        "embedding"
    );
    assert_eq!(attempts[1].request_metadata["upstream_profile"], "broken");
}

#[tokio::test]
async fn profile_limit_models_bypass_generation_saturation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_control_plane_in_flight_requests = 2

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 2
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
    )
    .await;

    let generation_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;
    assert_eq!(generation_response.status(), StatusCode::OK);
    let generation_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("generation request should hold default generation capacity");
    assert_eq!(
        generation_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let model_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata&slot=profile-limits"),
        ),
    ))
    .await
    .expect("models request should bypass generation saturation in profile-limit mode");
    assert_eq!(model_response.status(), StatusCode::OK);
    let model_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should reach upstream in profile-limit mode");
    assert_eq!(
        model_observed.path_and_query,
        "/v1/models?test=model-metadata&slot=profile-limits"
    );

    drop(model_response);
    drop(generation_response);
}

#[tokio::test]
async fn profile_wait_releases_body_routing_capacity_for_other_profiles() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000

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

    let default_active = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=default-active"),
    )
    .await;
    assert_eq!(default_active.status(), StatusCode::OK);
    let default_active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should hold default generation capacity");
    assert_eq!(
        default_active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=default-active"
    );

    let (default_queued_request, default_queued_body_polled) = tracked_json_request(
        "/v1/completions?slot=default-queued",
        br#"{"prompt":"queued default"}"#,
    );
    let default_queued = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        default_queued_request,
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        default_queued_body_polled.load(Ordering::SeqCst),
        "default queued request should read its body before occupying the default profile queue"
    );
    assert!(
        !default_queued.is_finished(),
        "default queued request should wait on default generation capacity"
    );

    let embedding_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            embedding_request("/v1/embeddings?test=long-json&slot=embedding"),
        ),
    ))
    .await
    .expect("matched profile request must not be blocked by default profile admission wait");
    assert_eq!(embedding_response.status(), StatusCode::OK);
    let embedding_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("embedding request should reach upstream while default profile waits");
    assert_eq!(
        embedding_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=embedding"
    );

    default_queued.abort();
    assert!(
        default_queued
            .await
            .expect_err("default queued request should be aborted")
            .is_cancelled()
    );
    drop(embedding_response);
    drop(default_active);
}

#[tokio::test]
async fn profile_limit_reload_off_exchanges_body_routing_for_global_generation_permit() {
    let mut fake = FakeUpstream::spawn().await;
    let initial_upstream_config = format!(
        r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
        fake.base_url
    );
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &initial_upstream_config,
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/embeddings?test=long-json&slot=reload-first",
        br#"{"model":"embedding-model","input":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "first request body should start reading",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        #[cfg(feature = "guard")]
        budget_sqlite_path: &proxy.budget_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 1,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit removal reload should succeed");
    assert!(outcome.applied);

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("first request should finish admission after reload")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream after reload");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=reload-first"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/embeddings?slot=reload-overflow",
        br#"{"model":"embedding-model","input":"overflow"}"#,
    );
    let overflow_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    ))
    .await
    .expect("overflow response should be bounded");
    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "overflow request should be rejected by global generation admission before body read"
    );
    assert_no_upstream_request(&mut fake).await;

    drop(overflow_response);
    drop(first_response);
}

#[tokio::test]
async fn profile_limit_reload_on_keeps_global_generation_permit_for_default_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/completions?test=long-json&slot=reload-on-default",
        br#"{"prompt":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "default request body should start reading before reload",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        #[cfg(feature = "guard")]
        budget_sqlite_path: &proxy.budget_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 1,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit addition reload should succeed");
    assert!(outcome.applied);

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("first request should finish admission after reload")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("default request should keep its global permit and reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/completions?test=long-json&slot=reload-on-default"
    );

    drop(first_response);
}

#[tokio::test]
async fn profile_limit_reload_on_keeps_global_generation_permit_for_newly_limited_profile() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        2,
        &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
"#,
            fake.base_url
        ),
    )
    .await;

    let (first_request, release_first_body, first_body_polled) = controlled_json_request(
        "/v1/embeddings?test=long-json&slot=reload-on-matched-first",
        br#"{"model":"embedding-model","input":"first"}"#,
    );
    let first = tokio::spawn(proxy_handler(State(proxy.state.clone()), first_request));
    wait_for_flag(
        &first_body_polled,
        "matched profile request body should start reading before reload",
    )
    .await;

    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path: proxy.manager.path(),
        upstream_base_url: &fake.base_url,
        sqlite_path: &proxy.sqlite_path,
        evidence_sqlite_path: &proxy.evidence_sqlite_path,
        #[cfg(feature = "guard")]
        budget_sqlite_path: &proxy.budget_sqlite_path,
        observability_enabled: true,
        max_in_flight_requests: 2,
        server_config: &format!(
            r#"max_queued_generation_requests = 0

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 0
"#,
            fake.base_url
        ),
        metadata_config: "",
        observability_config: "",
        evidence_config: "",
        extra_config: "",
    });
    let outcome = proxy
        .manager
        .reload()
        .expect("profile limit addition reload should succeed");
    assert!(outcome.applied);

    let active_profile_response = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-active"),
    )
    .await;
    assert_eq!(active_profile_response.status(), StatusCode::OK);
    let active_profile_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("new profile-limited request should hold profile capacity");
    assert_eq!(
        active_profile_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-active"
    );

    release_first_body
        .send(())
        .expect("first body release should be delivered");
    let first_response = timeout(STREAM_HEADER_TIMEOUT, first)
        .await
        .expect("pre-reload request should not be re-admitted into the new profile limiter")
        .expect("first request task should not panic");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("pre-reload matched request should keep its global permit and reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=reload-on-matched-first"
    );

    drop(active_profile_response);
    drop(first_response);
}

#[tokio::test]
async fn generation_queue_full_fails_without_body_buffering_or_upstream_forward() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 1000\n",
    )
    .await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should hold generation capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let (queued_request, queued_body_polled) =
        tracked_json_request("/v1/completions?slot=queued", br#"{"prompt":"queued"}"#);
    let queued = tokio::spawn(proxy_handler(State(proxy.state.clone()), queued_request));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "queued request must not read its body before capacity is available"
    );
    assert!(
        !queued.is_finished(),
        "first queued request should occupy the bounded queue"
    );

    let (overflow_request, overflow_body_polled) =
        tracked_json_request("/v1/completions?slot=overflow", br#"{"prompt":"overflow"}"#);
    let overflow_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    ))
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let response_request_id = terminal_response_request_id(overflow_response.headers());
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let overflow_body = to_bytes(overflow_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("queue-full body should read");
    let overflow_body =
        String::from_utf8(overflow_body.to_vec()).expect("queue-full body should be utf-8");
    assert!(
        overflow_body.contains("proxy_generation_queue_full"),
        "queue-full error should identify admission failure: {overflow_body}"
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "queue-full rejection must not read the request body"
    );
    let persisted_request = read_last_observability_row(&proxy.sqlite_path, "requests");
    assert_eq!(response_request_id, persisted_request.request_id);
    assert_eq!(persisted_request.status, "failed");
    assert_no_upstream_request(&mut fake).await;

    queued.abort();
    match queued.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "queued request should still be waiting before active response drops, got {}",
            response.status()
        ),
    }
    assert!(
        !queued_body_polled.load(Ordering::SeqCst),
        "aborted queued request must not read its body"
    );
    drop(first_response);
}

#[tokio::test]
async fn generation_queue_full_returns_configured_429_status() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 0\ngeneration_queue_timeout_ms = 1000\ngeneration_queue_full_status = 429\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=configured-status-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=configured-status-active"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/completions?slot=configured-status-overflow",
        br#"{"prompt":"overflow"}"#,
    );
    let overflow_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    ))
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "configured queue-full rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(active_response);
}

#[tokio::test]
async fn generation_queue_full_returns_configured_retry_after() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 0\ngeneration_queue_timeout_ms = 1000\ngeneration_queue_retry_after_secs = 30\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=configured-retry-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=configured-retry-active"
    );

    let (overflow_request, overflow_body_polled) = tracked_json_request(
        "/v1/completions?slot=configured-retry-overflow",
        br#"{"prompt":"overflow"}"#,
    );
    let overflow_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), overflow_request),
    ))
    .await
    .expect("queue-full response should be bounded");

    assert_eq!(overflow_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        overflow_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("30")
    );
    assert!(
        !overflow_body_polled.load(Ordering::SeqCst),
        "configured Retry-After queue-full rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(active_response);
}

#[tokio::test]
async fn queued_request_cancelled_on_downstream_disconnect_never_reaches_upstream() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=cancel-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=cancel-active"
    );

    let (cancelled_request, cancelled_body_polled) = tracked_json_request(
        "/v1/completions?slot=cancelled-queued",
        br#"{"prompt":"cancelled"}"#,
    );
    let cancelled = tokio::spawn(proxy_handler(State(proxy.state.clone()), cancelled_request));
    sleep(Duration::from_millis(50)).await;
    assert!(!cancelled_body_polled.load(Ordering::SeqCst));
    assert!(!cancelled.is_finished());

    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("queued request future should be cancelled")
            .is_cancelled()
    );
    assert!(!cancelled_body_polled.load(Ordering::SeqCst));
    let cancel_record = read_latest_aborted_request_metadata(&proxy);
    assert_eq!(
        cancel_record.abort_reason.as_deref(),
        Some("downstream_disconnected_while_queued")
    );
    assert_eq!(
        cancel_record.request_metadata["admission_outcome"],
        "queue_cancelled"
    );
    assert_eq!(cancel_record.request_metadata["path"], "/v1/completions");
    assert_no_upstream_request(&mut fake).await;

    let (replacement_request, replacement_body_polled) = tracked_json_request(
        "/v1/completions?slot=replacement-queued",
        br#"{"prompt":"replacement"}"#,
    );
    let replacement = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        replacement_request,
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(
        !replacement_body_polled.load(Ordering::SeqCst),
        "replacement should be queued before capacity is released"
    );
    assert!(
        !replacement.is_finished(),
        "replacement should queue, proving the cancelled request left the queue"
    );

    drop(active_response);
    let replacement_response = timeout(STREAM_COMPLETION_TIMEOUT, replacement)
        .await
        .expect("replacement should complete after capacity is released")
        .expect("replacement task should not panic");
    assert_eq!(replacement_response.status(), StatusCode::OK);
    let _body = to_bytes(replacement_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("replacement body should read");
    let replacement_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("replacement request should reach upstream after release");
    assert_eq!(
        replacement_observed.path_and_query,
        "/v1/completions?slot=replacement-queued"
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn queued_request_cancelled_on_downstream_disconnect_per_profile_limiter() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        16,
        &format!(
            r#"max_queued_generation_requests = 0
generation_queue_timeout_ms = 5000

[[upstreams]]
name = "embedding"
base_url = "{}"
match_models = ["embedding-model"]
max_in_flight_requests = 1
max_queued_generation_requests = 1
"#,
            fake.base_url
        ),
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-cancel-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active profile request should hold profile capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-cancel-active"
    );

    let cancelled = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-cancelled-queued"),
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(!cancelled.is_finished());
    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("queued profile request future should be cancelled")
            .is_cancelled()
    );
    let cancel_record = read_latest_aborted_request_metadata(&proxy);
    assert_eq!(
        cancel_record.request_metadata["admission_outcome"],
        "queue_cancelled"
    );
    assert_eq!(cancel_record.request_metadata["path"], "/v1/embeddings");
    assert_no_upstream_request(&mut fake).await;

    let replacement = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        embedding_request("/v1/embeddings?test=long-json&slot=profile-replacement-queued"),
    ));
    sleep(Duration::from_millis(50)).await;
    assert!(
        !replacement.is_finished(),
        "replacement should queue in the profile limiter"
    );

    drop(active_response);
    let replacement_response = timeout(STREAM_COMPLETION_TIMEOUT, replacement)
        .await
        .expect("profile replacement should complete after capacity is released")
        .expect("profile replacement task should not panic");
    assert_eq!(replacement_response.status(), StatusCode::OK);
    let _body = to_bytes(replacement_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("profile replacement body should read");
    let replacement_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("profile replacement request should reach upstream after release");
    assert_eq!(
        replacement_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=profile-replacement-queued"
    );
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn high_queue_capacity_allows_c32_without_immediate_rejection() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 256\ngeneration_queue_timeout_ms = 5000\n",
    )
    .await;
    let active_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=c32-active"),
    )
    .await;
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active request should hold generation capacity");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=c32-active"
    );

    let mut queued = Vec::new();
    for index in 0..32 {
        let uri = format!("/v1/completions?slot=c32-{index}");
        let (request, body_polled) = tracked_json_request(&uri, br#"{"prompt":"queued"}"#);
        let handle = tokio::spawn(proxy_handler(State(proxy.state.clone()), request));
        queued.push((handle, body_polled));
    }
    sleep(Duration::from_millis(50)).await;

    for (index, (handle, body_polled)) in queued.iter().enumerate() {
        assert!(
            !handle.is_finished(),
            "queued request {index} should wait instead of being rejected"
        );
        assert!(
            !body_polled.load(Ordering::SeqCst),
            "queued request {index} must not read its body before admission"
        );
    }
    assert_no_upstream_request(&mut fake).await;

    for (handle, _body_polled) in queued {
        handle.abort();
        assert!(
            handle
                .await
                .expect_err("queued c32 request should be aborted")
                .is_cancelled()
        );
    }
    drop(active_response);
}

fn tracked_json_request(uri: &str, body: &'static [u8]) -> (Request<Body>, Arc<AtomicBool>) {
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(body))
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("tracked json request should build");
    (request, polled)
}

fn tracked_pending_json_request(uri: &str) -> (Request<Body>, Arc<AtomicBool>) {
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            std::future::pending::<Result<Bytes, std::convert::Infallible>>().await
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("tracked pending json request should build");
    (request, polled)
}

fn controlled_json_request(
    uri: &str,
    body: &'static [u8],
) -> (Request<Body>, oneshot::Sender<()>, Arc<AtomicBool>) {
    let (release, released) = oneshot::channel();
    let polled = Arc::new(AtomicBool::new(false));
    let request_body = Body::from_stream(stream::once({
        let polled = Arc::clone(&polled);
        async move {
            polled.store(true, Ordering::SeqCst);
            let _ = released.await;
            Ok::<_, std::convert::Infallible>(Bytes::from_static(body))
        }
    }));
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(request_body)
        .expect("controlled json request should build");
    (request, release, polled)
}

async fn wait_for_flag(flag: &AtomicBool, label: &str) {
    timeout(Duration::from_secs(1), async {
        while !flag.load(Ordering::SeqCst) {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
}

#[tokio::test]
async fn generation_queue_timeout_fails_without_body_buffering_or_upstream_forward() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 20\n",
    )
    .await;
    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should hold generation capacity");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let body_polled = Arc::new(AtomicBool::new(false));
    let queued_body = Body::from_stream(stream::once({
        let body_polled = Arc::clone(&body_polled);
        async move {
            body_polled.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(br#"{"prompt":"timeout"}"#))
        }
    }));
    let queued_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions?slot=timeout")
        .header(CONTENT_TYPE, "application/json")
        .body(queued_body)
        .expect("queued request should build");
    let queued_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(State(proxy.state.clone()), queued_request),
    ))
    .await
    .expect("queue-timeout response should be bounded");

    assert_eq!(queued_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        queued_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let queued_body = to_bytes(queued_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("queue-timeout body should read");
    let queued_body =
        String::from_utf8(queued_body.to_vec()).expect("queue-timeout body should be utf-8");
    assert!(
        queued_body.contains("proxy_generation_queue_timeout"),
        "queue-timeout error should identify admission failure: {queued_body}"
    );
    assert!(
        !body_polled.load(Ordering::SeqCst),
        "queue-timeout rejection must not read the request body"
    );
    assert_no_upstream_request(&mut fake).await;
    drop(first_response);
}

#[tokio::test]
async fn models_bypass_generation_saturation_but_keep_control_plane_bound() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_control_plane_in_flight_requests = 1\n",
    )
    .await;
    let generation_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=active"),
    )
    .await;

    assert_eq!(generation_response.status(), StatusCode::OK);
    let generation_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("generation request should hold generation capacity");
    assert_eq!(
        generation_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=active"
    );

    let first_model_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/models?test=model-metadata-large&slot=one"),
    )
    .await;
    assert_eq!(first_model_response.status(), StatusCode::OK);
    let first_model_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("models request should bypass generation capacity");
    assert_eq!(
        first_model_observed.path_and_query,
        "/v1/models?test=model-metadata-large&slot=one"
    );

    let second_model_response = Box::pin(timeout(
        STREAM_HEADER_TIMEOUT,
        proxy_handler(
            State(proxy.state.clone()),
            empty_get_request("/v1/models?test=model-metadata&slot=two"),
        ),
    ))
    .await
    .expect("control-plane limit response should be bounded");
    assert_eq!(
        second_model_response.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        second_model_response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let second_model_body = to_bytes(second_model_response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("control-plane limit body should read");
    let second_model_body = String::from_utf8(second_model_body.to_vec())
        .expect("control-plane limit body should be utf-8");
    assert!(
        second_model_body.contains("proxy_control_plane_in_flight_limit_exceeded"),
        "control-plane error should identify admission failure: {second_model_body}"
    );
    assert_no_upstream_request(&mut fake).await;

    drop(first_model_response);
    drop(generation_response);
}

#[tokio::test]
async fn in_flight_limit_hot_reload_updates_admission_capacity() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 2).await;

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        1,
        "",
    );
    let outcome = proxy.manager.reload().expect("limit reload should succeed");
    assert!(outcome.applied);
    assert!(
        outcome.restart_required_changes.is_empty(),
        "in-flight limit should be safe to hot reload"
    );

    let first_response = proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=one"),
    )
    .await;
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("first request should reach upstream");
    assert_eq!(
        first_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=one"
    );

    let second = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=two"),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !second.is_finished(),
        "second generation request should wait while live limit is one"
    );

    write_proxy_config(
        proxy.manager.path(),
        &fake.base_url,
        &proxy.sqlite_path,
        true,
        2,
        "",
    );
    let outcome = proxy
        .manager
        .reload()
        .expect("limit increase should reload");
    assert!(outcome.applied);
    assert!(
        outcome.restart_required_changes.is_empty(),
        "limit increase should not require process restart"
    );

    let second_response = timeout(STREAM_HEADER_TIMEOUT, second)
        .await
        .expect("queued request should finish after limit increase")
        .expect("queued request task should join");
    assert_eq!(second_response.status(), StatusCode::OK);
    let second_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("second request should reach upstream after limit increase");
    assert_eq!(
        second_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=two"
    );

    let third = tokio::spawn(proxy_handler(
        State(proxy.state.clone()),
        empty_get_request("/v1/embeddings?test=long-json&slot=three"),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_no_upstream_request(&mut fake).await;
    assert!(
        !third.is_finished(),
        "third generation request should wait while both live slots are held"
    );
    third.abort();
    match third.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(response) => panic!(
            "third request should still be queued while both slots are held, got {}",
            response.status()
        ),
    }

    drop(first_response);
    drop(second_response);
}

#[tokio::test]
async fn graceful_shutdown_aborts_in_flight_response_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("shutdown test listener should bind");
    let addr = listener
        .local_addr()
        .expect("shutdown test address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = proxy.state.clone();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });

    let response = proxy
        .client
        .get(format!("http://{addr}/v1/embeddings?test=long-json"))
        .send()
        .await
        .expect("long response request should get headers");
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("long response should reach upstream before shutdown");
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=long-json");

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    timeout(STREAM_COMPLETION_TIMEOUT, server)
        .await
        .expect("server should exit after shutdown cancels the in-flight body")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");

    let admission = proxy.state.admission_metrics_snapshot();
    assert_eq!(admission.generation.active, 0);
    assert_eq!(admission.generation.queued, 0);
    drop(response);
}

#[tokio::test]
async fn shutdown_wakes_parked_poll_based_response_body() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("parked body shutdown listener should bind");
    let addr = listener
        .local_addr()
        .expect("parked body shutdown address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = proxy.state.clone();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });

    let response = proxy
        .client
        .get(format!("http://{addr}/v1/embeddings?test=parked-body"))
        .send()
        .await
        .expect("parked response request should get headers");
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("parked response should reach upstream before shutdown");
    assert_eq!(observed.path_and_query, "/v1/embeddings?test=parked-body");

    let mut body = response.bytes_stream();
    let first = next_chunk(&mut body, STREAM_HEADER_TIMEOUT, "parked body first chunk").await;
    assert_eq!(first, Bytes::from_static(LONG_JSON_FIRST_CHUNK));

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    timeout(STREAM_COMPLETION_TIMEOUT, server)
        .await
        .expect("server should exit even though upstream body never wakes again")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");

    wait_for_generation_metrics(&proxy, 0, 0, STREAM_COMPLETION_TIMEOUT).await;
    drop(body);
}

#[tokio::test]
async fn shutdown_cancels_pre_response_upstream_work() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_max_in_flight_requests(&fake.base_url, true, 1).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("pre-response shutdown listener should bind");
    let addr = listener
        .local_addr()
        .expect("pre-response shutdown address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = proxy.state.clone();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });

    let request = tokio::spawn({
        let client = proxy.client.clone();
        async move { send_pre_response_shutdown_request(client, addr).await }
    });

    let observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("pre-response request should reach upstream before shutdown");
    assert_eq!(
        observed.path_and_query,
        "/v1/chat/completions?test=pre-response-hang"
    );

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    let response_result = timeout(STREAM_COMPLETION_TIMEOUT, request)
        .await
        .expect("pre-response request should complete after shutdown")
        .expect("pre-response request task should join");
    match response_result {
        Ok((status, body)) => {
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert!(
                !body.is_empty(),
                "pre-response shutdown 503 body should not be empty"
            );
            assert!(
                body.contains("proxy_shutting_down"),
                "pre-response shutdown response should identify cancellation: {body}"
            );
        }
        Err(error) => assert!(
            !error.is_empty(),
            "pre-response shutdown should fail fast or return 503"
        ),
    }
    timeout(STREAM_COMPLETION_TIMEOUT, server)
        .await
        .expect("server should exit after pre-response shutdown cancellation")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");

    wait_for_generation_metrics(&proxy, 0, 0, STREAM_COMPLETION_TIMEOUT).await;
    let request_row = read_aborted_request_metadata_by_path(&proxy, "/v1/chat/completions");
    assert_eq!(request_row.http_status, Some(503));
    assert_eq!(request_row.abort_reason.as_deref(), Some("server_shutdown"));
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "aborted");
    assert_eq!(attempts[0].retry_reason, None);
    assert_eq!(attempts[0].abort_reason.as_deref(), Some("server_shutdown"));
    assert_eq!(
        attempts[0].response_metadata["abort_reason"].as_str(),
        Some("server_shutdown")
    );

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_active"),
        0
    );
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_queued"),
        0
    );
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_current_retained_request_terminals",
            &[
                ("status", "aborted"),
                ("terminal_reason", "server_shutdown"),
                ("http_status_class", "5xx"),
            ],
        ),
        1
    );
}

async fn send_pre_response_shutdown_request(
    client: Client,
    addr: std::net::SocketAddr,
) -> Result<(StatusCode, String), String> {
    let response = match client
        .post(format!(
            "http://{addr}/v1/chat/completions?test=pre-response-hang"
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"shutdown"}]}"#)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => return Err(error.to_string()),
    };
    let status = response.status();
    let body = response
        .text()
        .await
        .expect("pre-response shutdown response body should read");
    Ok((status, body))
}

#[tokio::test]
async fn shutdown_completes_after_disconnect_storm() {
    let mut upstream = CancellableUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_full_options(
        &upstream.base_url,
        true,
        4,
        "max_queued_generation_requests = 8\ngeneration_queue_timeout_ms = 5000\nshutdown_drain_timeout_ms = 1000\n",
        r#"
[heartbeat]
mode = "json-whitespace"
interval_secs = 1
"#,
        "",
        "",
    )
    .await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("shutdown storm listener should bind");
    let addr = listener
        .local_addr()
        .expect("shutdown storm address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = proxy.state.clone();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });

    let mut downstreams = Vec::new();
    for index in 0..4 {
        let response = timeout(
            STREAM_COMPLETION_TIMEOUT,
            proxy
                .client
                .post(format!(
                    "http://{addr}/v1/chat/completions?test=connection-storm-{index}"
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"storm"}]}"#)
                .send(),
        )
        .await
        .expect("shutdown storm response headers should be bounded")
        .expect("shutdown storm response should receive headers");
        assert_eq!(response.status(), StatusCode::OK);
        downstreams.push(response.bytes_stream());
    }

    for downstream in &mut downstreams {
        let prefix = next_chunk(
            downstream,
            SHIELDED_HEARTBEAT_TIMEOUT,
            "shutdown storm prefix",
        )
        .await;
        assert_eq!(prefix, Bytes::from_static(b" \n"));
    }

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    let shutdown_started = tokio::time::Instant::now();
    timeout(STREAM_COMPLETION_TIMEOUT, server)
        .await
        .expect("server should exit after shutdown starts")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");
    assert!(
        shutdown_started.elapsed() < Duration::from_millis(1_500),
        "server should finish within the configured drain budget plus scheduler margin"
    );

    for _ in 0..4 {
        let drop_event = upstream.recv_drop_within(STREAM_COMPLETION_TIMEOUT).await;
        assert_eq!(drop_event.label, "cancellable-chat-sse");
    }
    let admission = proxy.state.admission_metrics_snapshot();
    assert_eq!(admission.generation.active, 0);
    assert_eq!(admission.generation.queued, 0);
    let rejected = proxy_handler(
        State(proxy.state.clone()),
        json_post_request("/v1/completions", br#"{"model":"test","prompt":"late"}"#),
    )
    .await;
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let rejected_body = to_bytes(rejected.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("shutdown rejection body should read");
    let rejected_body =
        String::from_utf8(rejected_body.to_vec()).expect("shutdown body should be utf-8");
    assert!(
        rejected_body.contains("proxy_shutting_down"),
        "shutdown rejection should identify admission failure: {rejected_body}"
    );
    drop(downstreams);
}

#[tokio::test]
async fn queued_generation_requests_cancelled_by_shutdown_are_observable() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_admission_config(
        &fake.base_url,
        true,
        1,
        "max_queued_generation_requests = 1\ngeneration_queue_timeout_ms = 5000\nshutdown_drain_timeout_ms = 25\n",
    )
    .await;
    let (addr, shutdown_tx, server) = spawn_shutdown_server(proxy.state.clone()).await;
    let active_response = start_active_shutdown_request(&proxy, &mut fake, addr).await;

    let queued = tokio::spawn({
        let client = proxy.client.clone();
        async move {
            client
                .post(format!("http://{addr}/v1/completions?slot=shutdown-queued"))
                .header(CONTENT_TYPE, "application/json")
                .body(r#"{"model":"test","prompt":"queued shutdown"}"#)
                .send()
                .await
        }
    });
    sleep(Duration::from_millis(50)).await;
    let admission = proxy.state.admission_metrics_snapshot();
    assert_eq!(admission.generation.active, 1);
    assert_eq!(admission.generation.queued, 1);

    shutdown_tx
        .send(())
        .expect("shutdown signal should be delivered");
    timeout(STREAM_COMPLETION_TIMEOUT, server)
        .await
        .expect("server should exit after queued shutdown cancellation")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");

    let queued_response = timeout(STREAM_COMPLETION_TIMEOUT, queued)
        .await
        .expect("queued request task should finish after shutdown")
        .expect("queued request task should join")
        .expect("queued request should receive shutdown response");
    assert_eq!(queued_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let queued_body = queued_response
        .text()
        .await
        .expect("queued shutdown response body should read");
    assert!(
        queued_body.contains("proxy_generation_queue_cancelled"),
        "queued shutdown response should identify cancellation: {queued_body}"
    );

    wait_for_generation_metrics(&proxy, 0, 0, STREAM_COMPLETION_TIMEOUT).await;
    let cancel_record = read_aborted_request_metadata_by_path(&proxy, "/v1/completions");
    assert_eq!(cancel_record.http_status, Some(503));
    assert_eq!(
        cancel_record.abort_reason.as_deref(),
        Some("server_shutdown_while_queued")
    );
    assert_eq!(
        cancel_record.request_metadata["admission_outcome"],
        "queue_cancelled_shutdown"
    );
    assert_eq!(cancel_record.request_metadata["path"], "/v1/completions");

    let metrics = fetch_metrics(&proxy).await;
    assert_eq!(
        labelled_metric_value(
            &metrics,
            "llm_guard_proxy_current_retained_request_terminals",
            &[
                ("status", "aborted"),
                ("terminal_reason", "server_shutdown"),
                ("http_status_class", "5xx"),
            ],
        ),
        1
    );
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_active"),
        0
    );
    assert_eq!(
        metric_value(&metrics, "llm_guard_proxy_generation_queued"),
        0
    );
    assert_no_upstream_request(&mut fake).await;
    drop(active_response);
}

async fn spawn_shutdown_server(
    state: ProxyState,
) -> (
    std::net::SocketAddr,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("shutdown listener should bind");
    let addr = listener
        .local_addr()
        .expect("shutdown address should be readable");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_until_shutdown(listener, state, async {
            let _received = shutdown_rx.await;
        })
        .await
    });
    (addr, shutdown_tx, server)
}

async fn start_active_shutdown_request(
    proxy: &ProxyFixture,
    fake: &mut FakeUpstream,
    addr: std::net::SocketAddr,
) -> reqwest::Response {
    let active_response = proxy
        .client
        .get(format!(
            "http://{addr}/v1/embeddings?test=long-json&slot=shutdown-active"
        ))
        .send()
        .await
        .expect("active shutdown request should get headers");
    assert_eq!(active_response.status(), StatusCode::OK);
    let active_observed = fake
        .recv_within(STREAM_HEADER_TIMEOUT)
        .await
        .expect("active shutdown request should reach upstream");
    assert_eq!(
        active_observed.path_and_query,
        "/v1/embeddings?test=long-json&slot=shutdown-active"
    );
    active_response
}

#[tokio::test]
async fn invalid_upstream_url_failure_writes_metadata_without_secret() {
    let proxy = ProxyFixture::spawn("http://127.0.0.1:1/v1", true).await;
    let uri = Uri::from_static("/v1/models?limit=2");
    let headers = HeaderMap::new();
    let request_id =
        RequestId::from_string("req-invalid-upstream").expect("request id should be valid");
    let metadata = request_metadata(&Method::GET, &uri, &headers, 0, true);
    let error = ProxyError::invalid_upstream_url(
        "https://user:secret@example.test/v1?x-api-key=sk-test#token=sk-test",
        String::from("must not contain query parameters"),
    )
    .with_request_metadata(metadata);
    let error_type = error.error_type();
    let error_reason = error.to_string();
    let request_metadata = error
        .request_metadata()
        .cloned()
        .expect("invalid upstream URL should carry request metadata");

    record_failed_request(
        &proxy.state.persistence_tasks,
        &proxy.store,
        FailedRequestRecord {
            request_id,
            started_at_unix_ms: 1_000,
            finished_at_unix_ms: 1_050,
            status: RequestStatus::Failed,
            http_status: error.status().as_u16(),
            error_type,
            error_reason,
            abort_reason: None,
            request_metadata,
            attempts: Vec::new(),
        },
    );

    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_row: (String, i64, String, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, request_metadata_json FROM requests",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("failed request row should exist");
    let attempt_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .expect("attempt count should be readable");
    let request_metadata: serde_json::Value =
        serde_json::from_str(&request_row.3).expect("request metadata should be json");

    assert_eq!(request_row.0, "failed");
    assert_eq!(request_row.1, 500);
    assert!(request_row.2.contains("invalid_upstream_url"));
    assert!(
        request_row
            .2
            .contains("https://redacted:redacted@example.test/v1?redacted")
    );
    assert!(!request_row.2.contains("user:secret"));
    assert!(!request_row.2.contains("secret"));
    assert!(!request_row.2.contains("sk-test"));
    assert!(!request_row.2.contains("x-api-key"));
    assert!(!request_row.2.contains("token=sk-test"));
    assert_eq!(request_metadata["method"], "GET");
    assert_eq!(request_metadata["path"], "/v1/models");
    assert_eq!(request_metadata["query_present"], "true");
    assert_eq!(request_metadata["request_body_bytes"], "0");
    assert_eq!(request_metadata["policy_transform_applied"], "false");
    assert_eq!(attempt_count, 0);
}

#[tokio::test]
async fn dot_segment_paths_are_rejected_without_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    for request_target in ["/v1/../admin", "/v1/%2e%2e/admin", "/v1/%2E/admin"] {
        let response = send_raw_proxy_get(&proxy.base_url, request_target).await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "dot-segment target should be rejected: {response}"
        );
        assert_no_upstream_request(&mut fake).await;
    }
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

#[test]
fn upstream_url_rejects_raw_dot_segment_paths() {
    let uri = Uri::from_static("/v1/../admin");
    let error = build_upstream_url("http://upstream.example/v1", &uri)
        .expect_err("path should be rejected");

    assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    assert_eq!(error.error_type(), "invalid_request_path");
}

#[test]
fn upstream_url_rejects_percent_encoded_dot_segment_paths() {
    for path in [
        "/v1/%2e/admin",
        "/v1/%2E/admin",
        "/v1/%2e%2e/admin",
        "/v1/%2E%2E/admin",
        "/v1/.%2e/admin",
        "/v1/%2e./admin",
    ] {
        let uri = Uri::try_from(path).expect("test URI should be valid");
        let error = match build_upstream_url("http://upstream.example/v1", &uri) {
            Ok(url) => panic!("{path} should be rejected, got {url}"),
            Err(error) => error,
        };

        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.error_type(), "invalid_request_path");
    }
}

#[test]
fn upstream_url_rejects_and_redacts_credential_bearing_base_url() {
    let uri = Uri::from_static("/v1/models");
    let error = build_upstream_url(
        "https://user:secret@example.test/v1?x-api-key=sk-test#token=sk-test",
        &uri,
    )
    .expect_err("credential-bearing upstream URL should be rejected");
    let error = error.to_string();

    assert!(error.contains("invalid upstream base URL"));
    assert!(error.contains("https://redacted:redacted@example.test/v1?redacted"));
    assert!(!error.contains("user:secret"));
    assert!(!error.contains("secret"));
    assert!(!error.contains("sk-test"));
    assert!(!error.contains("x-api-key"));
    assert!(!error.contains("token=sk-test"));
}

#[test]
fn upstream_url_rejects_and_redacts_fragment_base_url() {
    let uri = Uri::from_static("/v1/models");
    let error = build_upstream_url("https://example.test/v1#token=sk-test", &uri)
        .expect_err("fragment-bearing upstream URL should be rejected");
    let error = error.to_string();

    assert!(error.contains("invalid upstream base URL"));
    assert!(error.contains("https://example.test/v1"));
    assert!(!error.contains("sk-test"));
    assert!(!error.contains("token=sk-test"));
}

async fn next_chunk<S, E>(body: &mut S, wait: Duration, label: &str) -> Bytes
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    timeout(wait, body.next())
        .await
        .unwrap_or_else(|_| panic!("{label} should arrive before timeout"))
        .unwrap_or_else(|| panic!("{label} should not end the stream"))
        .unwrap_or_else(|error| panic!("{label} should not fail: {error}"))
}

async fn collect_stream_text<S, E>(body: &mut S, wait: Duration) -> String
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    let mut bytes = Vec::new();
    loop {
        match timeout(wait, body.next()).await {
            Ok(Some(Ok(chunk))) => bytes.extend_from_slice(&chunk),
            Ok(Some(Err(error))) => panic!("stream should not fail: {error}"),
            Ok(None) => break,
            Err(error) => panic!("stream should finish before timeout: {error}"),
        }
    }
    String::from_utf8(bytes).expect("stream body should be UTF-8")
}

fn openai_sse_json_chunks(text: &str) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();
    for event in text.split("\n\n") {
        let mut data = String::new();
        for line in event.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.trim_start());
            }
        }
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        chunks.push(serde_json::from_str(data).unwrap_or_else(|error| {
            panic!("OpenAI SSE data should parse as JSON: {error}; data={data}")
        }));
    }
    chunks
}

async fn shielded_final_json(response: reqwest::Response) -> serde_json::Value {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let body = response.bytes().await.expect("body should be readable");
    if content_type.contains("text/event-stream") {
        final_json_from_sse_body(&body)
    } else {
        serde_json::from_slice(&body).unwrap_or_else(|error| {
            panic!("shielded JSON body should parse: {error}; body={body:?}")
        })
    }
}

async fn response_json(response: reqwest::Response) -> serde_json::Value {
    let body = response.text().await.expect("body should be readable");
    serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("response body should parse as JSON: {error}; body={body}"))
}

fn final_json_from_sse_body(body: &[u8]) -> serde_json::Value {
    let text = std::str::from_utf8(body)
        .unwrap_or_else(|error| panic!("SSE body should be UTF-8: {error}; body={body:?}"));
    for event in text.split("\n\n") {
        let mut event_name = "";
        let mut data = String::new();
        for line in event.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(value) = line.strip_prefix("event:") {
                event_name = value.trim();
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.trim_start());
            }
        }
        if event_name == "final" {
            return serde_json::from_str(&data).unwrap_or_else(|error| {
                panic!("final SSE data should parse as JSON: {error}; data={data}")
            });
        }
    }
    panic!("SSE body should include a final event: {text}");
}

fn first_model(body: &str) -> serde_json::Value {
    let value = serde_json::from_str::<serde_json::Value>(body)
        .unwrap_or_else(|error| panic!("model list should parse as JSON: {error}; body={body}"));
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .and_then(|models| models.first())
        .cloned()
        .unwrap_or_else(|| panic!("model list should contain at least one model: {body}"))
}

fn hermes_like_context_length(model: &serde_json::Value) -> Option<u64> {
    ["context_length", "max_model_len", "max_context_length"]
        .into_iter()
        .find_map(|key| model.get(key).and_then(serde_json::Value::as_u64))
}

fn assert_normalized_context_fields(model: &serde_json::Value, expected: u64) {
    assert_eq!(model["context_length"].as_u64(), Some(expected));
    assert_eq!(model["max_context_length"].as_u64(), Some(expected));
    assert_eq!(model["max_model_len"].as_u64(), Some(expected));
}

fn assert_metadata_latency(metadata: &serde_json::Value, key: &str) {
    let value = metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("{key} should be present"));
    value
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("{key} should be a u64 latency: {error}; value={value}"));
}

fn read_single_request_and_attempt_metadata(
    proxy: &ProxyFixture,
) -> (serde_json::Value, serde_json::Value) {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let request_metadata_json: String = connection
        .query_row("SELECT request_metadata_json FROM requests", [], |row| {
            row.get(0)
        })
        .expect("request row should exist");
    let attempt_metadata_json: String = connection
        .query_row("SELECT request_metadata_json FROM attempts", [], |row| {
            row.get(0)
        })
        .expect("attempt row should exist");
    let request_metadata =
        serde_json::from_str(&request_metadata_json).expect("request metadata should parse");
    let attempt_metadata =
        serde_json::from_str(&attempt_metadata_json).expect("attempt metadata should parse");
    (request_metadata, attempt_metadata)
}

#[cfg(feature = "param-override")]
fn read_latest_attempt_request_metadata(proxy: &ProxyFixture) -> serde_json::Value {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let metadata_json = connection
        .query_row(
            "SELECT request_metadata_json FROM attempts ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .expect("latest attempt row should exist");
    serde_json::from_str(&metadata_json).expect("latest attempt metadata should parse")
}

struct AbortedRequestMetadata {
    http_status: Option<i64>,
    abort_reason: Option<String>,
    request_metadata: serde_json::Value,
}

fn read_latest_aborted_request_metadata(proxy: &ProxyFixture) -> AbortedRequestMetadata {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let (http_status, abort_reason, request_metadata_json): (Option<i64>, Option<String>, String) =
        connection
            .query_row(
                "SELECT http_status, abort_reason, request_metadata_json \
             FROM requests WHERE status = 'aborted' ORDER BY rowid DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("aborted request row should exist");
    assert_eq!(
        http_status, None,
        "queued cancellation should not invent an HTTP response status"
    );
    let request_metadata =
        serde_json::from_str(&request_metadata_json).expect("request metadata should parse");
    AbortedRequestMetadata {
        http_status,
        abort_reason,
        request_metadata,
    }
}

fn read_aborted_request_metadata_by_path(
    proxy: &ProxyFixture,
    path: &str,
) -> AbortedRequestMetadata {
    let connection = Connection::open(&proxy.sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT http_status, abort_reason, request_metadata_json \
             FROM requests WHERE status = 'aborted' ORDER BY rowid DESC",
        )
        .expect("aborted request query should prepare");
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .expect("aborted request query should run");
    for row in rows {
        let (http_status, abort_reason, request_metadata_json) =
            row.expect("aborted request row should decode");
        let request_metadata: serde_json::Value =
            serde_json::from_str(&request_metadata_json).expect("request metadata should parse");
        if request_metadata["path"].as_str() == Some(path) {
            return AbortedRequestMetadata {
                http_status,
                abort_reason,
                request_metadata,
            };
        }
    }
    panic!("aborted request row for path {path:?} should exist");
}

async fn post_chat_and_observe_body(
    proxy: &ProxyFixture,
    fake: &mut FakeUpstream,
    body: &'static [u8],
) -> serde_json::Value {
    post_chat_and_observe_owned_body(proxy, fake, Bytes::from_static(body)).await
}

async fn post_chat_and_observe_owned_body(
    proxy: &ProxyFixture,
    fake: &mut FakeUpstream,
    body: Bytes,
) -> serde_json::Value {
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let _aggregated = shielded_final_json(response).await;
    let observed = fake.recv_next().await;
    serde_json::from_slice(&observed.body).expect("upstream body should be JSON")
}

#[cfg(feature = "param-override")]
fn param_override_profile_config(base_url: &str, param_override_body: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "param-override-test"
base_url = "{base_url}"
match_models = ["test-chat"]

[upstreams.thinking]
mode = "passthrough"

[upstreams.param_override]
{param_override_body}
"#,
    )
}

fn empty_get_request(uri: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .expect("GET request should build")
}

fn json_post_request(uri: &'static str, body: &'static [u8]) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("JSON request should build")
}

fn embedding_request(uri: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"embedding-model","input":"capacity test"}"#,
        ))
        .expect("embedding request should build")
}

fn listener_config(proxy: &ProxyFixture, name: &str) -> ListenerConfig {
    proxy
        .manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed")
        .listeners
        .into_iter()
        .find(|listener| listener.name == name)
        .unwrap_or_else(|| panic!("listener {name} should exist"))
}

fn multi_listener_profile_config(upstream_base_url: &str) -> String {
    format!(
        r#"
[[upstreams]]
name = "embedding"
base_url = "{upstream_base_url}"
match_models = ["embedding-model"]

[[upstreams]]
name = "rerank"
base_url = "{upstream_base_url}"
match_models = ["rerank-model"]

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["embedding"]

[[listeners]]
name = "reranker-legacy"
bind_host = "127.0.0.1"
port = 18003
allowed_upstreams = ["rerank"]

[[listeners]]
name = "embedding-rerank"
bind_host = "127.0.0.1"
port = 18004
allowed_upstreams = ["embedding", "rerank"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005
"#
    )
}

fn shielded_chat_request(uri: &'static str, body: &'static str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("shielded chat request should build")
}

#[cfg(feature = "guard")]
async fn send_budget_chat_request(
    proxy: &ProxyFixture,
    virtual_key: Option<&str>,
) -> reqwest::Response {
    let mut request = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"budget"}]}"#);
    if let Some(virtual_key) = virtual_key {
        request = request.header("x-virtual-key", virtual_key);
    }
    request
        .send()
        .await
        .expect("budget request should complete")
}

#[cfg(feature = "guard")]
fn read_budget_count(sqlite_path: &Path, profile: &str) -> u64 {
    let connection = Connection::open(sqlite_path).expect("budget sqlite should open");
    let count = connection
        .query_row(
            "SELECT count FROM budget_counts WHERE profile = ?1",
            params![profile],
            |row| row.get::<_, i64>(0),
        )
        .or_else(|source| match source {
            rusqlite::Error::QueryReturnedNoRows => Ok(0),
            source => Err(source),
        })
        .expect("budget count should read");
    u64::try_from(count).expect("budget count should be non-negative")
}

#[allow(clippy::needless_pass_by_value)]
#[cfg(feature = "guard")]
fn write_guard_script(root: &Path, name: &str, result: String) -> PathBuf {
    let path = root.join(format!("{name}.sh"));
    let script = format!("#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{result}'\n");
    fs::write(&path, script).expect("guard script should be written");
    path
}

#[cfg(feature = "guard")]
fn write_slow_guard_script(root: &Path, name: &str) -> PathBuf {
    let path = root.join(format!("{name}.sh"));
    fs::write(&path, "#!/bin/sh\nsleep 2\n").expect("slow guard script should be written");
    path
}

#[cfg(feature = "guard")]
fn write_literal_guard_script(root: &Path, name: &str, output: &str) -> PathBuf {
    let path = root.join(format!("{name}.sh"));
    let script = format!("#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{output}'\n");
    fs::write(&path, script).expect("literal guard script should be written");
    path
}

#[cfg(feature = "guard")]
fn write_large_stdout_guard_script(root: &Path, name: &str) -> PathBuf {
    let path = root.join(format!("{name}.sh"));
    fs::write(
        &path,
        "#!/bin/sh\ncat >/dev/null\nhead -c 1048577 /dev/zero | tr '\\000' x\n",
    )
    .expect("large stdout guard script should be written");
    path
}

#[cfg(feature = "guard")]
fn guard_result(decision: &str, replacement_messages: Option<&str>) -> String {
    format!(
        r#"{{"decision":"{decision}","risk_level":"test","tags":[],"summary":"guard summary","replacement_messages":{replacement_messages},"audit":{{"evidence_spans":[],"notes":[]}}}}"#,
        replacement_messages = replacement_messages.unwrap_or("null")
    )
}

#[cfg(feature = "guard")]
fn guard_workflow_config(
    pre_request_script: Option<&Path>,
    post_response_script: Option<&Path>,
    fail_closed_blocks: bool,
) -> String {
    let mut config = String::from("\n[guard_workflows]\n");
    if pre_request_script.is_some() {
        config.push_str("pre_request = \"pre_guard\"\n");
    }
    if post_response_script.is_some() {
        config.push_str("post_response = \"post_guard\"\n");
    }
    {
        let fail_line = format!("fail_closed_blocks = {fail_closed_blocks}\n");
        config.push_str(&fail_line);
    }
    if let Some(script) = pre_request_script {
        config.push_str(&workflow_config("pre_guard", script));
    }
    if let Some(script) = post_response_script {
        config.push_str(&workflow_config("post_guard", script));
    }
    config
}

#[cfg(feature = "guard")]
fn workflow_alias_config(script: &Path, alias_timeout_ms: u64) -> String {
    format!(
        r#"
[[model_aliases]]
id = "family/child-safe-general-v1"
kind = "workflow"
workflow_id = "child_safe_general_v1"
workflow_timeout_ms = {alias_timeout_ms}

{}
"#,
        workflow_config("child_safe_general_v1", script)
    )
}

#[cfg(feature = "guard")]
fn workflow_alias_config_with_stdout_limit(
    script: &Path,
    alias_timeout_ms: u64,
    max_stdout_bytes: usize,
) -> String {
    format!(
        r#"
[[model_aliases]]
id = "family/child-safe-general-v1"
kind = "workflow"
workflow_id = "child_safe_general_v1"
workflow_timeout_ms = {alias_timeout_ms}

[workflows.child_safe_general_v1]
runtime_kind = "stdio"
command = "sh"
args = ["{script}"]
timeout_ms = 10000
max_stdout_bytes = {max_stdout_bytes}
"#,
        script = script.display()
    )
}

#[cfg(feature = "guard")]
fn workflow_config(id: &str, script: &Path) -> String {
    format!(
        r#"
[workflows.{id}]
runtime_kind = "stdio"
command = "sh"
args = ["{script}"]
timeout_ms = 10000
max_stdout_bytes = 65536
"#,
        script = script.display()
    )
}

#[cfg(feature = "guard")]
fn virtual_key_config(unknown_key_policy: &str) -> String {
    format!(
        r#"
[profiles.default]
kind = "adult"
allowed_models = ["gpt-default", "adult-model"]

[profiles.child_safe]
kind = "child"
allowed_models = ["child-model"]

[virtual_keys]
enabled = true
unknown_key_policy = "{unknown_key_policy}"

[virtual_keys.keys]
vk_adult_abc123 = "default"
vk_child_def456 = "child_safe"
"#
    )
}

#[cfg(feature = "guard")]
fn single_profile_virtual_key_config() -> &'static str {
    r#"
[profiles.solo]
kind = "adult"
allowed_models = ["solo-model"]

[virtual_keys]
enabled = true
unknown_key_policy = "fail_closed"
"#
}

#[derive(Debug)]
struct ForwardedRecordRow {
    status: String,
    http_status: i64,
    error_reason: Option<String>,
    abort_reason: Option<String>,
    response_metadata: serde_json::Value,
}

fn read_single_forwarded_request_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, abort_reason, response_metadata_json FROM requests",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("request row should exist");
    let response_metadata =
        serde_json::from_str(&row.4).expect("request response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        error_reason: row.2,
        abort_reason: row.3,
        response_metadata,
    }
}

fn read_single_forwarded_attempt_row(sqlite_path: &Path) -> ForwardedRecordRow {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let row: (String, i64, Option<String>, Option<String>, String) = connection
        .query_row(
            "SELECT status, http_status, error_reason, abort_reason, response_metadata_json FROM attempts",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("attempt row should exist");
    let response_metadata =
        serde_json::from_str(&row.4).expect("attempt response metadata should be json");

    ForwardedRecordRow {
        status: row.0,
        http_status: row.1,
        error_reason: row.2,
        abort_reason: row.3,
        response_metadata,
    }
}

fn assert_forwarded_abort_recorded(proxy: &ProxyFixture) {
    let request_row = read_single_forwarded_request_row(&proxy.sqlite_path);
    let attempt_row = read_single_forwarded_attempt_row(&proxy.sqlite_path);

    assert_eq!(request_row.status, "aborted");
    assert_eq!(request_row.http_status, 200);
    assert_eq!(
        request_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
    assert_eq!(attempt_row.status, "aborted");
    assert_eq!(attempt_row.http_status, 200);
    assert_eq!(
        attempt_row.abort_reason.as_deref(),
        Some("downstream_body_dropped_before_eof")
    );
}

#[derive(Debug)]
struct ObservabilityRow {
    request_id: String,
    status: String,
    response_metadata: serde_json::Value,
}

#[derive(Debug)]
struct AttemptChainRow {
    attempt_number: u32,
    status: String,
    retry_reason: Option<String>,
    abort_reason: Option<String>,
    response_metadata: serde_json::Value,
}

#[derive(Debug)]
struct AttemptRequestMetadataRow {
    attempt_id: String,
    request_id: String,
    attempt_number: u32,
    status: String,
    request_metadata: serde_json::Value,
}

#[derive(Debug)]
struct EvidenceAttemptRow {
    role: String,
    shown_to_downstream: i64,
    status: String,
    retry_reason: Option<String>,
    shadow_skip_reason: Option<String>,
    thinking_budget_tokens: Option<u32>,
    detector_features: serde_json::Value,
}

fn read_attempt_request_metadata_rows(sqlite_path: &Path) -> Vec<AttemptRequestMetadataRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT attempt_id, request_id, attempt_number, status, request_metadata_json \
             FROM attempts ORDER BY rowid",
        )
        .expect("attempt metadata query should prepare");
    statement
        .query_map([], |row| {
            let metadata_json: String = row.get(4)?;
            let request_metadata = serde_json::from_str(&metadata_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            Ok(AttemptRequestMetadataRow {
                attempt_id: row.get(0)?,
                request_id: row.get(1)?,
                attempt_number: row.get(2)?,
                status: row.get(3)?,
                request_metadata,
            })
        })
        .expect("attempt metadata query should execute")
        .map(|row| row.expect("attempt metadata row should decode"))
        .collect()
}

fn read_attempt_chain_rows(sqlite_path: &Path) -> Vec<AttemptChainRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT attempt_number, status, retry_reason, abort_reason, response_metadata_json \
             FROM attempts ORDER BY rowid",
        )
        .expect("attempt chain query should prepare");
    statement
        .query_map([], |row| {
            let metadata_json: String = row.get(4)?;
            let response_metadata = serde_json::from_str(&metadata_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            Ok(AttemptChainRow {
                attempt_number: row.get(0)?,
                status: row.get(1)?,
                retry_reason: row.get(2)?,
                abort_reason: row.get(3)?,
                response_metadata,
            })
        })
        .expect("attempt chain query should execute")
        .map(|row| row.expect("attempt chain row should decode"))
        .collect()
}

fn read_evidence_attempt_rows(sqlite_path: &Path) -> Vec<EvidenceAttemptRow> {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut statement = connection
        .prepare(
            "SELECT role, shown_to_downstream, status, retry_reason, shadow_skip_reason, \
             thinking_budget_tokens, detector_features_json FROM evidence_attempts ORDER BY rowid",
        )
        .expect("evidence attempt query should prepare");
    statement
        .query_map([], |row| {
            let detector_features_json: String = row.get(6)?;
            let detector_features =
                serde_json::from_str(&detector_features_json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?;
            Ok(EvidenceAttemptRow {
                role: row.get(0)?,
                shown_to_downstream: row.get(1)?,
                status: row.get(2)?,
                retry_reason: row.get(3)?,
                shadow_skip_reason: row.get(4)?,
                thinking_budget_tokens: row.get(5)?,
                detector_features,
            })
        })
        .expect("evidence attempt query should execute")
        .map(|row| row.expect("evidence attempt row should decode"))
        .collect()
}

fn count_rows(connection: &Connection, sql: &str) -> u64 {
    let count: i64 = connection
        .query_row(sql, [], |row| row.get(0))
        .expect("count query should succeed");
    u64::try_from(count).expect("count should be nonnegative")
}

async fn assert_forwarded_attempt_count_stays(sqlite_path: &Path, expected_count: u64) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM attempts"),
        expected_count
    );
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        count_rows(&connection, "SELECT COUNT(*) FROM attempts"),
        expected_count
    );
}

async fn assert_shadow_timeout_count_stays(sqlite_path: &Path, expected_count: u64) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        expected_count
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        expected_count
    );
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts WHERE role = 'shadow_continued'",
        ),
        expected_count
    );
}

fn assert_shadow_timeout_summary(
    sqlite_path: &Path,
    expected_timeout_count: u64,
    expected_request_count: u64,
) {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'shadow_timeout'",
        ),
        expected_timeout_count
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(*) FROM evidence_attempts \
             WHERE role = 'shadow_continued' AND status = 'skipped' \
             AND shadow_skip_reason = 'global_limit'",
        ),
        0
    );
    assert_eq!(
        count_rows(
            &connection,
            "SELECT COUNT(DISTINCT request_id) FROM evidence_attempts \
             WHERE role = 'shadow_continued'",
        ),
        expected_request_count
    );
}

fn read_evidence_chunks(connection: &Connection) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT channel, sequence_number, chunk_text \
             FROM evidence_chunks ORDER BY sequence_number",
        )
        .expect("evidence chunks query should prepare");
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .expect("evidence chunks query should execute")
        .map(|row| row.expect("evidence chunk row should decode"))
        .collect()
}

fn read_evidence_chunks_for_role(
    connection: &Connection,
    role: &str,
) -> Vec<(String, i64, String)> {
    let mut statement = connection
        .prepare(
            "SELECT c.channel, c.sequence_number, c.chunk_text \
             FROM evidence_chunks c \
             JOIN evidence_attempts a ON a.attempt_id = c.attempt_id \
             WHERE a.role = ?1 \
             ORDER BY c.sequence_number",
        )
        .expect("role evidence chunks query should prepare");
    statement
        .query_map(params![role], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .expect("role evidence chunks query should execute")
        .map(|row| row.expect("role evidence chunk row should decode"))
        .collect()
}

async fn send_shadow_timeout_request(proxy: &ProxyFixture, request_index: u32) {
    send_shadow_timeout_request_parts(&proxy.client, &proxy.base_url, request_index).await;
}

async fn send_shadow_timeout_request_parts(
    client: &reqwest::Client,
    base_url: &str,
    request_index: u32,
) {
    let response = client
        .post(format!(
            "{base_url}/v1/chat/completions?test=loop-once-shadow-timeout-then-success&id={request_index}",
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}]}"#)
        .send()
        .await
        .expect("proxy request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let aggregated = shielded_final_json(response).await;
    assert_eq!(aggregated["choices"][0]["message"]["content"], "Hello");
}

async fn recv_shadow_timeout_upstream_requests(fake: &mut FakeUpstream) -> Vec<ObservedRequest> {
    recv_n_upstream_requests(fake, 3).await
}

async fn recv_n_upstream_requests(
    fake: &mut FakeUpstream,
    expected_count: usize,
) -> Vec<ObservedRequest> {
    let mut requests = Vec::new();
    for _ in 0..expected_count {
        requests.push(
            timeout(Duration::from_secs(2), fake.recv_next())
                .await
                .expect("expected request should reach upstream"),
        );
    }
    requests
}

async fn wait_for_evidence_status_count(sqlite_path: &Path, status: &str, expected: u64) {
    timeout(Duration::from_secs(5), async {
        loop {
            if sqlite_path.exists() {
                let connection = Connection::open(sqlite_path).expect("sqlite should open");
                let query =
                    format!("SELECT COUNT(*) FROM evidence_attempts WHERE status = '{status}'");
                if count_rows(&connection, &query) >= expected {
                    break;
                }
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("evidence status count should reach expected value");
}

async fn wait_for_evidence_role_status_count(
    sqlite_path: &Path,
    role: &str,
    status: &str,
    expected: u64,
) {
    timeout(Duration::from_secs(5), async {
        loop {
            if sqlite_path.exists() {
                let connection = Connection::open(sqlite_path).expect("sqlite should open");
                let count: i64 = connection
                    .query_row(
                        "SELECT COUNT(*) FROM evidence_attempts WHERE role = ?1 AND status = ?2",
                        params![role, status],
                        |row| row.get(0),
                    )
                    .expect("evidence role status count query should succeed");
                if u64::try_from(count).expect("count should be nonnegative") >= expected {
                    break;
                }
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("evidence role status count should reach expected value");
}

fn read_last_observability_row(sqlite_path: &Path, table: &str) -> ObservabilityRow {
    assert!(matches!(table, "requests" | "attempts"));
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let sql = format!(
        "SELECT request_id, status, response_metadata_json FROM {table} ORDER BY rowid DESC LIMIT 1"
    );
    let row: (String, String, String) = connection
        .query_row(&sql, [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .expect("observability row should exist");
    let response_metadata = serde_json::from_str(&row.2).expect("response metadata should be json");
    ObservabilityRow {
        request_id: row.0,
        status: row.1,
        response_metadata,
    }
}

#[cfg(feature = "guard")]
fn read_last_request_metadata(sqlite_path: &Path) -> serde_json::Value {
    serde_json::from_str(&read_last_request_metadata_json(sqlite_path))
        .expect("request metadata should be json")
}

#[cfg(feature = "guard")]
fn read_last_request_metadata_json(sqlite_path: &Path) -> String {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    connection
        .query_row(
            "SELECT request_metadata_json FROM requests ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("request row should exist")
}

#[cfg(feature = "guard")]
fn audit_row_count(sqlite_path: &Path, table: &str) -> u64 {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let count: i64 = connection
        .query_row(&sql, [], |row| row.get(0))
        .expect("audit count query should succeed");
    u64::try_from(count).expect("audit count should be nonnegative")
}

#[cfg(feature = "guard")]
fn read_audit_text(sqlite_path: &Path) -> String {
    let connection = Connection::open(sqlite_path).expect("sqlite should open");
    let mut text = String::new();
    append_audit_table_text(&connection, "requests", &mut text);
    append_audit_table_text(&connection, "attempts", &mut text);
    text
}

#[cfg(feature = "guard")]
fn append_audit_table_text(connection: &Connection, table: &str, text: &mut String) {
    let sql = format!(
        "SELECT request_metadata_json, response_metadata_json, COALESCE(error_reason, '') \
         FROM {table} ORDER BY rowid"
    );
    let mut statement = connection
        .prepare(&sql)
        .expect("audit text query should prepare");
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .expect("audit text query should execute");
    for row in rows {
        let (request_metadata, response_metadata, error_reason) =
            row.expect("audit text row should decode");
        text.push_str(&request_metadata);
        text.push('\n');
        text.push_str(&response_metadata);
        text.push('\n');
        text.push_str(&error_reason);
        text.push('\n');
    }
}

fn repeated_input_chat_body() -> String {
    let repeated_input = format!("{REPEATED_INPUT_LOOP_LINE}\n{REPEATED_INPUT_LOOP_LINE}\n");
    serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": repeated_input}],
    })
    .to_string()
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

#[derive(Debug)]
struct UpstreamDropEvent {
    label: &'static str,
}

struct CancellableUpstream {
    base_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
    drop_receiver: mpsc::Receiver<UpstreamDropEvent>,
}

#[derive(Clone)]
struct CancellableUpstreamState {
    request_sender: mpsc::Sender<ObservedRequest>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    attempt_counts: Arc<Mutex<HashMap<String, u64>>>,
}

impl CancellableUpstream {
    async fn spawn() -> Self {
        let (request_sender, receiver) = mpsc::channel(10);
        let (drop_sender, drop_receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(cancellable_upstream_handler)
            .with_state(CancellableUpstreamState {
                request_sender,
                drop_sender,
                attempt_counts: Arc::new(Mutex::new(HashMap::new())),
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("cancellable upstream should bind");
        let addr = listener
            .local_addr()
            .expect("cancellable upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("cancellable upstream server failed: {error}");
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            receiver,
            drop_receiver,
        }
    }

    async fn recv_request(&mut self) -> ObservedRequest {
        self.receiver
            .recv()
            .await
            .expect("cancellable upstream should capture a request")
    }

    async fn recv_request_optional_within(&mut self, wait: Duration) -> Option<ObservedRequest> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }

    async fn recv_drop_within(&mut self, wait: Duration) -> UpstreamDropEvent {
        timeout(wait, self.drop_receiver.recv())
            .await
            .expect("upstream response body should be dropped before timeout")
            .expect("upstream drop channel should stay open")
    }

    async fn recv_drop_optional_within(&mut self, wait: Duration) -> Option<UpstreamDropEvent> {
        timeout(wait, self.drop_receiver.recv())
            .await
            .ok()
            .flatten()
    }
}

#[derive(Clone)]
struct FakeUpstreamState {
    sender: mpsc::Sender<ObservedRequest>,
    changing_model_len: Arc<AtomicU64>,
    attempt_counts: Arc<Mutex<HashMap<String, u64>>>,
    models_body: Option<&'static str>,
    models_status: StatusCode,
    models_label: &'static str,
    models_delay: Option<Duration>,
    pre_response_delay: Option<Duration>,
    rerank_status: Option<StatusCode>,
    deepinfra_response: Option<(StatusCode, &'static str)>,
}

impl FakeUpstream {
    async fn spawn() -> Self {
        Self::spawn_with_optional_models_body(None).await
    }

    async fn spawn_with_models_body(models_body: &'static str) -> Self {
        Self::spawn_with_models_options(Some(models_body), None).await
    }

    async fn spawn_with_models_response(
        models_status: StatusCode,
        models_body: &'static str,
        models_label: &'static str,
    ) -> Self {
        Self::spawn_with_models_response_options(
            Some(models_body),
            models_status,
            models_label,
            None,
        )
        .await
    }

    async fn spawn_with_models_body_and_delay(
        models_body: &'static str,
        models_delay: Duration,
    ) -> Self {
        Self::spawn_with_models_options(Some(models_body), Some(models_delay)).await
    }

    async fn spawn_with_optional_models_body(models_body: Option<&'static str>) -> Self {
        Self::spawn_with_models_options(models_body, None).await
    }

    async fn spawn_with_models_options(
        models_body: Option<&'static str>,
        models_delay: Option<Duration>,
    ) -> Self {
        Self::spawn_with_models_response_options(
            models_body,
            StatusCode::OK,
            "models",
            models_delay,
        )
        .await
    }

    async fn spawn_with_models_response_options(
        models_body: Option<&'static str>,
        models_status: StatusCode,
        models_label: &'static str,
        models_delay: Option<Duration>,
    ) -> Self {
        Self::spawn_with_options(
            models_body,
            models_status,
            models_label,
            models_delay,
            None,
            None,
            None,
        )
        .await
    }

    async fn spawn_with_pre_response_delay(pre_response_delay: Duration) -> Self {
        Self::spawn_with_options(
            None,
            StatusCode::OK,
            "models",
            None,
            Some(pre_response_delay),
            None,
            None,
        )
        .await
    }

    async fn spawn_with_rerank_status(rerank_status: StatusCode) -> Self {
        Self::spawn_with_options(
            None,
            StatusCode::OK,
            "models",
            None,
            None,
            Some(rerank_status),
            None,
        )
        .await
    }

    async fn spawn_with_deepinfra_response_body(deepinfra_response_body: &'static str) -> Self {
        Self::spawn_with_deepinfra_response(StatusCode::OK, deepinfra_response_body).await
    }

    async fn spawn_with_deepinfra_response(
        deepinfra_response_status: StatusCode,
        deepinfra_response_body: &'static str,
    ) -> Self {
        Self::spawn_with_options(
            None,
            StatusCode::OK,
            "models",
            None,
            None,
            None,
            Some((deepinfra_response_status, deepinfra_response_body)),
        )
        .await
    }

    async fn spawn_with_options(
        models_body: Option<&'static str>,
        models_status: StatusCode,
        models_label: &'static str,
        models_delay: Option<Duration>,
        pre_response_delay: Option<Duration>,
        rerank_status: Option<StatusCode>,
        deepinfra_response: Option<(StatusCode, &'static str)>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(fake_upstream_handler)
            .with_state(FakeUpstreamState {
                sender,
                changing_model_len: Arc::new(AtomicU64::new(128_000)),
                attempt_counts: Arc::new(Mutex::new(HashMap::new())),
                models_body,
                models_status,
                models_label,
                models_delay,
                pre_response_delay,
                rerank_status,
                deepinfra_response,
            });
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
        self.recv_next().await
    }

    async fn recv_next(&mut self) -> ObservedRequest {
        self.receiver
            .recv()
            .await
            .expect("fake upstream should capture a request")
    }

    async fn recv_within(&mut self, wait: Duration) -> Option<ObservedRequest> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }
}

struct RedirectingUpstream {
    base_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
}

impl RedirectingUpstream {
    async fn spawn(status: StatusCode, location: String) -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(redirecting_upstream_handler)
            .with_state(RedirectingUpstreamState {
                sender,
                status,
                location,
            });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("redirecting upstream should bind");
        let addr = listener
            .local_addr()
            .expect("redirecting upstream address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("redirecting upstream server failed: {error}");
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
            .expect("redirecting upstream should capture a request")
    }
}

#[derive(Clone)]
struct RedirectingUpstreamState {
    sender: mpsc::Sender<ObservedRequest>,
    status: StatusCode,
    location: String,
}

struct RedirectTarget {
    capture_url: String,
    receiver: mpsc::Receiver<ObservedRequest>,
}

impl RedirectTarget {
    async fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel(10);
        let app = Router::new()
            .fallback(capture_request_handler)
            .with_state(sender);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("redirect target should bind");
        let addr = listener
            .local_addr()
            .expect("redirect target address should be available");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("redirect target server failed: {error}");
            }
        });

        Self {
            capture_url: format!("http://{addr}/v1/redirect-target"),
            receiver,
        }
    }

    async fn recv_within(&mut self, wait: Duration) -> Option<ObservedRequest> {
        timeout(wait, self.receiver.recv()).await.ok().flatten()
    }
}

struct BrokenUpstream {
    base_url: String,
}

impl BrokenUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("broken upstream listener should bind");
        let addr = listener
            .local_addr()
            .expect("broken upstream address should be available");
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((_stream, _addr)) => {}
                    Err(error) => {
                        eprintln!("broken upstream listener failed: {error}");
                        break;
                    }
                }
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
        }
    }
}

async fn redirecting_upstream_handler(
    State(state): State<RedirectingUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    state
        .sender
        .send(observed)
        .await
        .expect("redirecting upstream observation should send");

    let mut response = Response::new(Body::from("redirected"));
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        LOCATION,
        HeaderValue::from_str(&state.location).expect("redirect location should be valid"),
    );
    response
}

async fn capture_request_handler(
    State(sender): State<mpsc::Sender<ObservedRequest>>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    sender
        .send(observed)
        .await
        .expect("redirect target observation should send");
    Response::new(Body::from("captured"))
}

async fn cancellable_upstream_handler(
    State(state): State<CancellableUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    let body = observed.body.clone();
    let path_and_query = observed.path_and_query.clone();
    state
        .request_sender
        .send(observed)
        .await
        .expect("cancellable upstream observation should send");

    if path_and_query.starts_with("/v1/models") {
        return json_response("models", MODEL_METADATA_BODY.to_owned());
    }

    if path_and_query.contains("test=loop-twice-then-cancellable-success")
        && body_requests_stream(&body)
        && next_cancellable_attempt_count(&state, &path_and_query) <= 2
    {
        return repeated_reasoning_line_sse_response(200);
    }

    if path_and_query.contains("test=delayed-loop-then-cancellable-success")
        && body_requests_stream(&body)
        && next_cancellable_attempt_count(&state, &path_and_query) == 1
    {
        return cancellable_repeated_reasoning_line_sse_response(
            state.drop_sender,
            200,
            Duration::from_millis(100),
        );
    }

    if body_requests_stream(&body) {
        if path_and_query.contains("test=connection-storm") {
            return cancellable_chat_sse_response_with_delay(
                state.drop_sender,
                STREAM_COMPLETION_TIMEOUT,
            );
        }
        cancellable_chat_sse_response(state.drop_sender)
    } else {
        cancellable_chat_json_response(state.drop_sender)
    }
}

fn next_cancellable_attempt_count(state: &CancellableUpstreamState, key: &str) -> u64 {
    let mut counts = state
        .attempt_counts
        .lock()
        .expect("cancellable attempt counts should lock");
    let entry = counts.entry(key.to_owned()).or_insert(0);
    *entry = entry.saturating_add(1);
    *entry
}

async fn fake_upstream_handler(
    State(state): State<FakeUpstreamState>,
    request: Request<Body>,
) -> Response<Body> {
    let observed = observe_request(request).await;
    let path_and_query = observed.path_and_query.clone();
    let body = observed.body.clone();
    let is_hot_restart_probe = observed
        .headers
        .get("x-llm-guard-proxy-probe")
        .is_some_and(|value| value == "hot-restart");
    let endpoint = observed
        .path_and_query
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let is_sse_stream = observed.path_and_query.contains("test=sse");
    let is_long_json_stream = observed.path_and_query.contains("test=long-json");
    state
        .sender
        .send(observed)
        .await
        .expect("fake upstream observation should send");
    if endpoint == "/v1/models"
        && path_and_query.contains("test=distinct-multi-upstream-models")
        && state.models_body.is_some()
        && let Some(models_delay) = state.models_delay
    {
        sleep(models_delay).await;
    }

    if is_sse_stream {
        return delayed_stream_response(
            "sse",
            "text/event-stream",
            SSE_FIRST_CHUNK,
            SSE_SECOND_CHUNK,
        );
    }
    if is_long_json_stream {
        return delayed_stream_response(
            "long-json",
            "application/json",
            LONG_JSON_FIRST_CHUNK,
            LONG_JSON_SECOND_CHUNK,
        );
    }
    if path_and_query.contains("test=parked-body") {
        return parked_stream_response(
            "parked-body",
            "application/json",
            Bytes::from_static(LONG_JSON_FIRST_CHUNK),
        );
    }
    if endpoint == "/v1/chat/completions" && !body_requests_stream(&body) {
        if body_contains_text(&body, "hot-restart-never-ready") {
            return upstream_status_json_response(StatusCode::SERVICE_UNAVAILABLE);
        }
        if body_contains_text(&body, "hot-restart-shared-probe") {
            sleep(Duration::from_millis(150)).await;
        }
        if is_hot_restart_probe {
            return json_response(
                "hot-restart-probe",
                r#"{"id":"chatcmpl-probe","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#
                    .to_owned(),
            );
        }
    }
    if endpoint != "/v1/models"
        && let Some(pre_response_delay) = state.pre_response_delay
    {
        sleep(pre_response_delay).await;
    } else if path_and_query.contains("test=pre-response-hang") {
        sleep(STREAM_COMPLETION_TIMEOUT.saturating_mul(5)).await;
    }

    let mut response = fake_upstream_endpoint_response(&endpoint, &path_and_query, &state, &body);
    if path_and_query.contains("test=request-id-collision") {
        response.headers_mut().insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_static("upstream-request-id-collision"),
        );
    }
    response
}

async fn observe_request(request: Request<Body>) -> ObservedRequest {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_PROXY_BODY_BYTES)
        .await
        .expect("fake upstream body should be readable");
    let path_and_query = parts.uri.path_and_query().map_or_else(
        || parts.uri.path().to_owned(),
        |value| value.as_str().to_owned(),
    );
    ObservedRequest {
        method: parts.method,
        path_and_query,
        headers: parts.headers,
        body,
    }
}

fn fake_upstream_endpoint_response(
    endpoint: &str,
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Response<Body> {
    if endpoint == "/v1/models" {
        if path_and_query.contains("test=model-metadata-chunked") {
            return chunked_json_response(
                "models",
                MODEL_METADATA_CHUNKED_FIRST,
                MODEL_METADATA_CHUNKED_SECOND,
            );
        }
        if path_and_query.contains("test=model-metadata-large") {
            return json_response("models", large_model_metadata_body());
        }
        if path_and_query.contains("test=model-metadata-changing") {
            let max_model_len = state
                .changing_model_len
                .fetch_add(128_000, Ordering::SeqCst);
            return json_response("models", model_metadata_body(max_model_len));
        }
        if path_and_query.contains("test=model-metadata-no-context") {
            return json_response("models", MODEL_METADATA_NO_CONTEXT_BODY.to_owned());
        }
        if path_and_query.contains("test=model-metadata-context-length") {
            return json_response("models", MODEL_METADATA_CONTEXT_LENGTH_BODY.to_owned());
        }
        if path_and_query.contains("test=model-metadata-max-context-length") {
            return json_response("models", MODEL_METADATA_MAX_CONTEXT_LENGTH_BODY.to_owned());
        }
        if path_and_query.contains("test=multi-listener-models") {
            return json_response("models", MULTI_LISTENER_MODEL_METADATA_BODY.to_owned());
        }
        if path_and_query.contains("test=distinct-multi-upstream-models")
            && let Some(models_body) = state.models_body
        {
            let mut response = json_response(state.models_label, models_body.to_owned());
            *response.status_mut() = state.models_status;
            if state.models_status == StatusCode::TOO_MANY_REQUESTS {
                response
                    .headers_mut()
                    .insert(RETRY_AFTER, HeaderValue::from_static("11"));
            }
            return response;
        }
        if path_and_query.contains("test=model-metadata") {
            return json_response("models", MODEL_METADATA_BODY.to_owned());
        }
    }

    if endpoint == "/v1/chat/completions"
        && let Some(response) = fake_chat_completion_response(path_and_query, state, body)
    {
        return response;
    }

    if endpoint == "/v1/embeddings" && path_and_query.contains("test=token-usage") {
        return json_response(
            "embeddings-token-usage",
            String::from(
                r#"{"object":"list","data":[{"embedding":[0.0]}],"usage":{"prompt_tokens":17,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}"#,
            ),
        );
    }

    if endpoint == "/v1/score" && path_and_query.contains("test=deepinfra-rerank") {
        return fake_deepinfra_score_response(path_and_query);
    }

    if endpoint == "/v1/inference/Qwen/Qwen3-Reranker-8B"
        && let Some((status, body)) = state.deepinfra_response
    {
        let mut response = json_response("deepinfra-inference", body.to_owned());
        *response.status_mut() = status;
        return response;
    }

    if endpoint == "/v1/rerank"
        && let Some(status) = state.rerank_status
    {
        return upstream_status_json_response(status);
    }

    let (label, body) = match endpoint {
        "/v1/models" => ("models", r#"{"object":"list","data":[]}"#),
        "/v1/chat/completions" => (
            "chat-completions",
            r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        ),
        "/v1/completions" => (
            "completions",
            r#"{"id":"cmpl-test","object":"text_completion"}"#,
        ),
        "/v1/embeddings" => (
            "embeddings",
            r#"{"object":"list","data":[{"embedding":[0.0]}]}"#,
        ),
        "/v1/rerank" => return fake_rerank_response(path_and_query, body),
        _ => ("unknown", r#"{"error":"unsupported"}"#),
    };
    let status = if label == "unknown" {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::OK
    };
    let mut response = json_response(label, body.to_owned());
    *response.status_mut() = status;
    response
}

fn fake_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if !body_requests_stream(body) {
        return None;
    }
    fake_streaming_chat_completion_response(path_and_query, state, body)
        .or_else(|| fake_malformed_chat_completion_response(path_and_query))
        .or_else(|| Some(chat_completion_sse_response(body)))
}

fn fake_malformed_chat_completion_response(path_and_query: &str) -> Option<Response<Body>> {
    if path_and_query.contains("test=malformed-sse-invalid-json") {
        return Some(malformed_json_chat_completion_sse_response());
    }
    None
}

fn fake_streaming_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if path_and_query.contains("test=watchdog-tool-call-only") {
        return Some(repeated_tool_fingerprint_sse_response());
    }
    if let Some(response) =
        fake_compat_and_loop_once_chat_completion_response(path_and_query, state, body)
    {
        return Some(response);
    }
    if let Some(response) = fake_loop_twice_then_success_response(path_and_query, state, body) {
        return Some(response);
    }
    if let Some(response) = fake_loop_three_then_success_response(path_and_query, state, body) {
        return Some(response);
    }
    if let Some(response) = fake_loop_three_then_slow_success_response(path_and_query, state, body)
    {
        return Some(response);
    }
    if path_and_query.contains("test=tool-loop-then-content-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_tool_fingerprint_sse_response());
        }
        return Some(content_only_chat_completion_sse_response());
    }
    if let Some(response) = fake_fixed_status_chat_completion_response(path_and_query) {
        return Some(response);
    }
    if path_and_query.contains("test=transient-503-then-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(upstream_status_json_response(
                StatusCode::SERVICE_UNAVAILABLE,
            ));
        }
        return Some(chat_completion_sse_response(body));
    }
    if let Some(response) = fake_hot_restart_chat_completion_response(path_and_query, state, body) {
        return Some(response);
    }
    if let Some(response) =
        fake_stall_recovery_chat_completion_response(path_and_query, state, body)
    {
        return Some(response);
    }
    if let Some(response) = fake_loop_fixture_chat_completion_response(path_and_query) {
        return Some(response);
    }
    None
}

fn fake_loop_three_then_slow_success_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if !path_and_query.contains("test=loop-three-then-slow-success") {
        return None;
    }
    if next_fake_attempt_count(state, path_and_query) <= 3 {
        return Some(repeated_reasoning_line_sse_response(200));
    }
    Some(slow_chat_completion_sse_response(body))
}

fn fake_loop_three_then_success_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if !path_and_query.contains("test=loop-three-then-success") {
        return None;
    }
    if next_fake_attempt_count(state, path_and_query) <= 3 {
        return Some(repeated_reasoning_line_sse_response(200));
    }
    Some(chat_completion_sse_response(body))
}

fn fake_loop_twice_then_success_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if !path_and_query.contains("test=loop-twice-then-success") {
        return None;
    }
    if next_fake_attempt_count(state, path_and_query) <= 2 {
        return Some(repeated_reasoning_line_sse_response(200));
    }
    Some(chat_completion_sse_response(body))
}

fn fake_fixed_status_chat_completion_response(path_and_query: &str) -> Option<Response<Body>> {
    if path_and_query.contains("test=always-502") {
        return Some(upstream_status_json_response(StatusCode::BAD_GATEWAY));
    }
    if path_and_query.contains("test=always-429") {
        return Some(upstream_status_json_response(StatusCode::TOO_MANY_REQUESTS));
    }
    if path_and_query.contains("test=bad-request") {
        return Some(upstream_status_json_response(StatusCode::BAD_REQUEST));
    }
    None
}

fn fake_hot_restart_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if path_and_query.contains("test=hot-restart-503-then-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(upstream_status_json_response(
                StatusCode::SERVICE_UNAVAILABLE,
            ));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=hot-restart-concurrent") {
        if next_fake_attempt_count(state, path_and_query) <= 2 {
            return Some(upstream_status_json_response(
                StatusCode::SERVICE_UNAVAILABLE,
            ));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=hot-restart-always-503") {
        return Some(upstream_status_json_response(
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    None
}

fn next_fake_attempt_count(state: &FakeUpstreamState, key: &str) -> u64 {
    let mut counts = state
        .attempt_counts
        .lock()
        .expect("fake upstream attempt counts should not be poisoned");
    let count = counts.entry(key.to_owned()).or_insert(0);
    *count = count.saturating_add(1);
    *count
}

fn body_contains_retry_hint(body: &Bytes) -> bool {
    retry_hint_count(body) > 0
}

fn body_contains_text(body: &Bytes, needle: &str) -> bool {
    std::str::from_utf8(body).is_ok_and(|text| text.contains(needle))
}

fn retry_hint_count(body: &Bytes) -> usize {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("messages")
                .and_then(serde_json::Value::as_array)
                .cloned()
        })
        .map_or(0, |messages| {
            messages
                .iter()
                .filter(|message| {
                    message
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|content| {
                            content.contains("llm-guard-proxy retry hint")
                                || content.contains("llm-guard-proxy CoT salvage retry hint")
                                || content.contains("Previous attempt became repetitive")
                        })
                })
                .count()
        })
}

fn body_thinking_budget(body: &Bytes) -> Option<u64> {
    let value = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    [
        &["thinking", "budget_tokens"][..],
        &["thinking_token_budget"][..],
        &["thinking_budget"][..],
        &["chat_template_kwargs", "thinking_budget"][..],
        &["extra_body", "thinking_token_budget"][..],
        &["extra_body", "thinking_budget"][..],
        &["extra_body", "thinking", "budget_tokens"][..],
        &["extra_body", "chat_template_kwargs", "thinking_budget"][..],
    ]
    .into_iter()
    .find_map(|path| json_path(&value, path).and_then(serde_json::Value::as_u64))
}

fn json_path<'value>(
    mut value: &'value serde_json::Value,
    path: &[&str],
) -> Option<&'value serde_json::Value> {
    for key in path {
        value = value.get(*key)?;
    }
    Some(value)
}

fn upstream_status_json_response(status: StatusCode) -> Response<Body> {
    let mut response = json_response(
        "chat-completions-transient-error",
        r#"{"error":{"type":"upstream_test_error","message":"try again"}}"#.to_owned(),
    );
    *response.status_mut() = status;
    if status == StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("7"));
    }
    response
}

fn model_metadata_body(max_model_len: u64) -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"aeon-ultimate","object":"model","max_model_len":{max_model_len},"owned_by":"vllm","extra":"keep"}}]}}"#
    )
}

fn large_model_metadata_body() -> String {
    format!(
        r#"{{"object":"list","data":[{{"id":"large-model","object":"model","max_model_len":256000,"owned_by":"vllm","extra":"{}"}}]}}"#,
        "x".repeat(LARGE_MODEL_METADATA_EXTRA_BYTES)
    )
}

fn json_response(label: &'static str, body: String) -> Response<Body> {
    let content_length = body.len().to_string();
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&content_length).expect("content length should be valid"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn parked_stream_response(
    label: &'static str,
    content_type: &'static str,
    first: Bytes,
) -> Response<Body> {
    let body = Body::from_stream(
        stream::once(async move { Ok::<Bytes, Infallible>(first) })
            .chain(stream::pending::<Result<Bytes, Infallible>>()),
    );
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

fn cancellable_chat_sse_response(drop_sender: mpsc::Sender<UpstreamDropEvent>) -> Response<Body> {
    cancellable_chat_sse_response_with_delay(drop_sender, STREAM_DELAY)
}

fn cancellable_chat_sse_response_with_delay(
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    delay_after_first: Duration,
) -> Response<Body> {
    let chunks = vec![
        sse_json(&chat_completion_first_chunk()),
        sse_json(&chat_completion_second_chunk(false)),
        sse_json(&chat_completion_final_chunk(true, false)),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    cancellable_stream_response(
        "cancellable-chat-sse",
        "text/event-stream",
        chunks,
        drop_sender,
        delay_after_first,
    )
}

fn cancellable_chat_json_response(drop_sender: mpsc::Sender<UpstreamDropEvent>) -> Response<Body> {
    let chunks = vec![
        Bytes::from_static(br#"{"id":"chatcmpl-cancellable","#),
        Bytes::from_static(
            br#""object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        ),
    ];
    cancellable_stream_response(
        "cancellable-chat-json",
        "application/json",
        chunks,
        drop_sender,
        STREAM_DELAY,
    )
}

fn cancellable_stream_response(
    label: &'static str,
    content_type: &'static str,
    chunks: Vec<Bytes>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    delay_after_first: Duration,
) -> Response<Body> {
    let body = Body::from_stream(CancellableResponseStream::new(
        label,
        chunks,
        drop_sender,
        delay_after_first,
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

struct CancellableResponseStream {
    label: &'static str,
    chunks: Vec<Bytes>,
    next_index: usize,
    delay_after_first: Option<Pin<Box<tokio::time::Sleep>>>,
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    completed: bool,
}

impl CancellableResponseStream {
    fn new(
        label: &'static str,
        chunks: Vec<Bytes>,
        drop_sender: mpsc::Sender<UpstreamDropEvent>,
        delay_after_first: Duration,
    ) -> Self {
        Self {
            label,
            chunks,
            next_index: 0,
            delay_after_first: Some(Box::pin(sleep(delay_after_first))),
            drop_sender,
            completed: false,
        }
    }
}

impl Stream for CancellableResponseStream {
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.next_index >= this.chunks.len() {
            this.completed = true;
            return Poll::Ready(None);
        }

        if this.next_index > 0
            && let Some(delay) = &mut this.delay_after_first
        {
            match delay.as_mut().poll(cx) {
                Poll::Ready(()) => {
                    this.delay_after_first = None;
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        let chunk = this.chunks[this.next_index].clone();
        this.next_index = this.next_index.saturating_add(1);
        Poll::Ready(Some(Ok(chunk)))
    }
}

impl Drop for CancellableResponseStream {
    fn drop(&mut self) {
        if !self.completed {
            let _send_result = self
                .drop_sender
                .try_send(UpstreamDropEvent { label: self.label });
        }
    }
}

fn chat_completion_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let include_logprobs = body_requests_logprobs(body);
    let first_chunk = chat_completion_first_chunk();
    let second_chunk = chat_completion_second_chunk(include_logprobs);
    let final_chunk = chat_completion_final_chunk(include_usage, include_logprobs);
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&second_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-sse"),
    );
    response
}

fn content_only_chat_completion_sse_response() -> Response<Body> {
    let chunks = [
        sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Safe"
                },
                "finish_reason": null
            }]
        })),
        sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": " answer"
                },
                "finish_reason": "stop"
            }]
        })),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-content-only-sse", chunks)
}

fn slow_chat_completion_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let include_logprobs = body_requests_logprobs(body);
    let chunks = vec![
        sse_json(&chat_completion_first_chunk()),
        sse_json(&chat_completion_second_chunk(include_logprobs)),
        sse_json(&chat_completion_final_chunk(
            include_usage,
            include_logprobs,
        )),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_delayed_start_stream_response("chat-completions-slow-sse", chunks)
}

fn stalled_chat_completion_sse_response() -> Response<Body> {
    let body = Body::from_stream(stream::pending::<Result<Bytes, std::convert::Infallible>>());
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-stalled-sse"),
    );
    response
}

fn fake_compat_and_loop_once_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if path_and_query.contains("test=compat-function-call") {
        return Some(chat_completion_compat_function_call_sse_response(body));
    }
    if path_and_query.contains("test=compat-refusal") {
        return Some(chat_completion_compat_refusal_sse_response(body));
    }
    if path_and_query.contains("test=compat-extensions") {
        return Some(chat_completion_extension_fields_sse_response(body));
    }
    if path_and_query.contains("test=slow-shielded") {
        return Some(slow_chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=loop-once-then-slow-success") {
        return Some(if body_contains_retry_hint(body) {
            slow_chat_completion_sse_response(body)
        } else {
            repeated_reasoning_line_sse_response(200)
        });
    }
    if path_and_query.contains("test=loop-once-shadow-raw-then-success") {
        if body_contains_retry_hint(body) {
            return Some(chat_completion_sse_response(body));
        }
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_reasoning_line_sse_response(200));
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=loop-once-shadow-timeout-then-success") {
        if body_contains_retry_hint(body) {
            return Some(chat_completion_sse_response(body));
        }
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(repeated_reasoning_line_sse_response(200));
        }
        return Some(stalled_chat_completion_sse_response());
    }
    if path_and_query.contains("test=loop-once-then-success") {
        return Some(if body_contains_retry_hint(body) {
            chat_completion_sse_response(body)
        } else {
            repeated_reasoning_line_sse_response(200)
        });
    }
    None
}

fn fake_loop_fixture_chat_completion_response(path_and_query: &str) -> Option<Response<Body>> {
    if path_and_query.contains("test=loop-reasoning-hundreds") {
        return Some(repeated_reasoning_line_sse_response(200));
    }
    if path_and_query.contains("test=reasoning-leading-newlines") {
        return Some(reasoning_then_leading_newline_content_sse_response());
    }
    if path_and_query.contains("test=loop-reasoning-six") {
        return Some(repeated_reasoning_line_sse_response(6));
    }
    if path_and_query.contains("test=semantic-reasoning-varied") {
        return Some(semantic_reasoning_repetition_sse_response());
    }
    if path_and_query.contains("test=repeated-tool-fingerprint") {
        return Some(repeated_tool_fingerprint_sse_response());
    }
    if path_and_query.contains("test=copy-input-under-threshold") {
        return Some(repeated_input_copy_sse_response(11));
    }
    if path_and_query.contains("test=copy-input-over-threshold") {
        return Some(repeated_input_copy_sse_response(12));
    }
    None
}

fn fake_stall_recovery_chat_completion_response(
    path_and_query: &str,
    state: &FakeUpstreamState,
    body: &Bytes,
) -> Option<Response<Body>> {
    if path_and_query.contains("test=stall-once-then-success") {
        if next_fake_attempt_count(state, path_and_query) == 1 {
            return Some(stalled_chat_completion_sse_response());
        }
        return Some(chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=delayed-first-chunk-then-success") {
        return Some(delayed_first_chunk_chat_completion_sse_response(body));
    }
    if path_and_query.contains("test=inter-chunk-stall-after-first") {
        return Some(inter_chunk_stalled_chat_completion_sse_response());
    }
    None
}

fn delayed_first_chunk_chat_completion_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let include_logprobs = body_requests_logprobs(body);
    let chunks = vec![
        sse_json(&chat_completion_first_chunk()),
        sse_json(&chat_completion_second_chunk(include_logprobs)),
        sse_json(&chat_completion_final_chunk(
            include_usage,
            include_logprobs,
        )),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    // First chunk arrives after the inter-chunk timeout but before the first-chunk timeout.
    chat_completion_delayed_start_stream_response_with_delay(
        "chat-completions-delayed-first-chunk-sse",
        chunks,
        Duration::from_millis(150),
    )
}

fn inter_chunk_stalled_chat_completion_sse_response() -> Response<Body> {
    let first = sse_json(&chat_completion_first_chunk());
    let body = Body::from_stream(stream::unfold(Some(first), |state| async move {
        if let Some(first) = state {
            // Emit first chunk immediately, then hang between chunks.
            Some((Ok::<_, std::convert::Infallible>(first), None))
        } else {
            let () = std::future::pending().await;
            None
        }
    }));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-inter-chunk-stalled-sse"),
    );
    response
}

fn chat_completion_delayed_start_stream_response_with_delay(
    label: &'static str,
    chunks: Vec<Bytes>,
    first_chunk_delay: Duration,
) -> Response<Body> {
    let body = Body::from_stream(stream::unfold(
        (0_usize, chunks, first_chunk_delay),
        |(index, chunks, first_chunk_delay)| async move {
            if index >= chunks.len() {
                return None;
            }
            if index == 0 {
                sleep(first_chunk_delay).await;
            }
            let chunk = chunks[index].clone();
            Some((
                Ok::<_, std::convert::Infallible>(chunk),
                (index + 1, chunks, first_chunk_delay),
            ))
        },
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn malformed_json_chat_completion_sse_response() -> Response<Body> {
    let mut response = Response::new(Body::from("data: {\"choices\": [\n\n"));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static("chat-completions-malformed-json-sse"),
    );
    response
}

fn repeated_reasoning_line_sse_response(repetitions: usize) -> Response<Body> {
    repeated_delta_sse_response(
        "chat-completions-loop-reasoning-sse",
        repetitions,
        |line| {
            serde_json::json!({
                "reasoning_content": line,
            })
        },
        "reasoning loop line\n",
    )
}

fn cancellable_repeated_reasoning_line_sse_response(
    drop_sender: mpsc::Sender<UpstreamDropEvent>,
    repetitions: usize,
    delay_after_first: Duration,
) -> Response<Body> {
    let mut deltas = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        deltas.push(serde_json::json!({
            "reasoning_content": "reasoning loop line\n",
        }));
    }
    cancellable_stream_response(
        "cancellable-loop-reasoning-sse",
        "text/event-stream",
        delta_vec_sse_chunks(deltas),
        drop_sender,
        delay_after_first,
    )
}

fn reasoning_then_leading_newline_content_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-reasoning-leading-newlines-sse",
        [
            serde_json::json!({
                "reasoning_content": "think before answering",
            }),
            serde_json::json!({
                "content": "\n\nOK",
            }),
        ],
    )
}

fn semantic_reasoning_repetition_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-semantic-reasoning-sse",
        [
            serde_json::json!({
                "reasoning_content": "Use bsdtar to extract the archive into /dev/shm, then check unzip in a temporary directory and inspect members with python zipfile.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Try unzip into a tmpdir, but keep bsdtar available for archive extraction and use Python's zipfile module to inspect entries.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Python zipfile can read the archive listing; if that stalls, extract with bsdtar or unzip into a temporary directory.\n",
            }),
            serde_json::json!({
                "reasoning_content": "Return to the unzip tmpdir plan, with bsdtar as the extractor fallback and python zipfile for inspection.\n",
            }),
        ],
    )
}

fn repeated_tool_fingerprint_sse_response() -> Response<Body> {
    delta_fragments_sse_response(
        "chat-completions-repeated-tool-fingerprint-sse",
        [
            serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\",\"limit\":1}"
                    }
                }]
            }),
            serde_json::json!({
                "tool_calls": [{
                    "index": 1,
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"limit\":1,\"q\":\"x\"}"
                    }
                }]
            }),
        ],
    )
}

fn repeated_input_copy_sse_response(repetitions: usize) -> Response<Body> {
    repeated_delta_sse_response(
        "chat-completions-copy-input-sse",
        repetitions,
        |line| {
            serde_json::json!({
                "content": line,
            })
        },
        &format!("{REPEATED_INPUT_LOOP_LINE}\n"),
    )
}

fn delta_fragments_sse_response<const N: usize>(
    label: &'static str,
    deltas: [serde_json::Value; N],
) -> Response<Body> {
    delta_vec_sse_response(label, Vec::from(deltas))
}

fn delta_vec_sse_response(label: &'static str, deltas: Vec<serde_json::Value>) -> Response<Body> {
    chat_completion_vec_stream_response(label, delta_vec_sse_chunks(deltas))
}

fn delta_vec_sse_chunks(deltas: Vec<serde_json::Value>) -> Vec<Bytes> {
    let mut chunks = Vec::with_capacity(deltas.len().saturating_add(3));
    chunks.push(sse_json(&serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant"
            },
            "finish_reason": null
        }]
    })));
    for delta in deltas {
        chunks.push(sse_json(&serde_json::json!({
            "id": "chatcmpl-shielded",
            "object": "chat.completion.chunk",
            "created": 1_710_000_000_u64,
            "model": "test-chat",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null
            }]
        })));
    }
    chunks.push(sse_json(&serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    })));
    chunks.push(Bytes::from_static(b"data: [DONE]\n\n"));
    chunks
}

fn repeated_delta_sse_response(
    label: &'static str,
    repetitions: usize,
    delta: impl Fn(&str) -> serde_json::Value,
    line: &str,
) -> Response<Body> {
    let mut deltas = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        deltas.push(delta(line));
    }
    delta_vec_sse_response(label, deltas)
}

fn chat_completion_delayed_start_stream_response(
    label: &'static str,
    chunks: Vec<Bytes>,
) -> Response<Body> {
    let body = Body::from_stream(stream::unfold(
        (0_usize, chunks),
        |(index, chunks)| async move {
            if index >= chunks.len() {
                return None;
            }
            if index == 0 {
                sleep(SHIELDED_SLOW_DELAY).await;
            }
            let chunk = chunks[index].clone();
            Some((
                Ok::<_, std::convert::Infallible>(chunk),
                (index + 1, chunks),
            ))
        },
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn chat_completion_vec_stream_response(label: &'static str, chunks: Vec<Bytes>) -> Response<Body> {
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_str(label).expect("static label should be a valid header"),
    );
    response
}

fn chat_completion_compat_function_call_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "function_call": {
                    "name": "legacy_lookup",
                    "arguments": "{\"q\""
                }
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "function_call": {
                    "arguments": ":\"x\"}"
                }
            },
            "finish_reason": "function_call"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-compat-function-call-sse", chunks)
}

fn chat_completion_compat_refusal_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "refusal": "I cannot"
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "service_tier": "flex",
        "choices": [{
            "index": 0,
            "delta": {
                "refusal": " help with that"
            },
            "finish_reason": "stop"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-compat-refusal-sse", chunks)
}

fn chat_completion_extension_fields_sse_response(body: &Bytes) -> Response<Body> {
    let include_usage = body_requests_stream_usage(body);
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "provider_metadata": {
            "phase": "first"
        },
        "x_provider_trace": "trace-first",
        "choices": [{
            "index": 0,
            "provider_choice": {
                "phase": "first"
            },
            "delta": {
                "role": "assistant",
                "content": "Hel",
                "provider_message": {
                    "phase": "first"
                },
                "x_message_trace": "trace-first"
            },
            "finish_reason": null
        }]
    });
    let mut final_chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "provider_metadata": {
            "phase": "final"
        },
        "choices": [{
            "index": 0,
            "provider_choice": {
                "phase": "final"
            },
            "x_choice_trace": "choice-final",
            "delta": {
                "content": "lo",
                "provider_message": {
                    "phase": "final"
                }
            },
            "finish_reason": "stop"
        }]
    });
    if include_usage {
        final_chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    let chunks = [
        sse_json(&first_chunk),
        sse_json(&final_chunk),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    chat_completion_stream_response("chat-completions-extension-fields-sse", chunks)
}

fn chat_completion_stream_response<const N: usize>(
    label: &'static str,
    chunks: [Bytes; N],
) -> Response<Body> {
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

fn chat_completion_first_chunk() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hel"
            },
            "finish_reason": null
        }]
    })
}

fn chat_completion_second_chunk(include_logprobs: bool) -> serde_json::Value {
    let mut chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "lo",
                "reasoning_content": "think",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\""
                    }
                }]
            },
            "finish_reason": null
        }]
    });
    if include_logprobs {
        insert_first_choice_field(
            &mut chunk,
            "logprobs",
            serde_json::json!({
                "content": [{
                    "token": "Hello",
                    "bytes": [72, 101, 108, 108, 111],
                    "logprob": -0.01,
                    "top_logprobs": []
                }]
            }),
        );
    }
    chunk
}

fn chat_completion_final_chunk(include_usage: bool, include_logprobs: bool) -> serde_json::Value {
    let mut chunk = serde_json::json!({
        "id": "chatcmpl-shielded",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000_u64,
        "model": "test-chat",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": ":\"x\"}"
                    }
                }]
            },
            "finish_reason": "stop"
        }]
    });
    if include_logprobs {
        insert_first_choice_field(
            &mut chunk,
            "logprobs",
            serde_json::json!({
                "content": [{
                    "token": "!",
                    "bytes": [33],
                    "logprob": -0.02,
                    "top_logprobs": []
                }]
            }),
        );
    }
    if include_usage {
        chunk
            .as_object_mut()
            .expect("final chunk should be a JSON object")
            .insert(
                String::from("usage"),
                serde_json::json!({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }),
            );
    }
    chunk
}

fn sse_json(value: &serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn insert_first_choice_field(chunk: &mut serde_json::Value, key: &str, field: serde_json::Value) {
    if let Some(choice) = chunk
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(serde_json::Value::as_object_mut)
    {
        choice.insert(key.to_owned(), field);
    }
}

fn body_requests_stream(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

fn body_requests_stream_usage(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("stream_options")
                .and_then(|stream_options| stream_options.get("include_usage"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

fn body_requests_logprobs(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("logprobs").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

fn chunked_json_response(
    label: &'static str,
    first: &'static [u8],
    second: &'static [u8],
) -> Response<Body> {
    let body = Body::from_stream(stream::iter([
        Ok::<_, std::convert::Infallible>(Bytes::from_static(first)),
        Ok::<_, std::convert::Infallible>(Bytes::from_static(second)),
    ]));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

fn delayed_stream_response(
    label: &'static str,
    content_type: &'static str,
    first: &'static [u8],
    second: &'static [u8],
) -> Response<Body> {
    let body = Body::from_stream(stream::unfold(0_u8, move |state| async move {
        match state {
            0 => Some((
                Ok::<_, std::convert::Infallible>(Bytes::from_static(first)),
                1,
            )),
            1 => {
                sleep(STREAM_DELAY).await;
                Some((
                    Ok::<_, std::convert::Infallible>(Bytes::from_static(second)),
                    2,
                ))
            }
            _ => None,
        }
    }));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-endpoint"),
        HeaderValue::from_static(label),
    );
    response
}

struct ProxyFixture {
    base_url: String,
    client: Client,
    manager: ConfigManager,
    state: ProxyState,
    store: ObservabilityStore,
    sqlite_path: PathBuf,
    evidence_sqlite_path: PathBuf,
    #[cfg(feature = "guard")]
    budget_sqlite_path: PathBuf,
    root: PathBuf,
}

impl ProxyFixture {
    async fn spawn(upstream_base_url: &str, observability_enabled: bool) -> Self {
        Self::spawn_with_max_in_flight_requests(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
        )
        .await
    }

    async fn spawn_with_max_in_flight_requests(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
    ) -> Self {
        Self::spawn_with_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            "",
        )
        .await
    }

    async fn spawn_with_admission_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        server_config: &str,
    ) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            server_config,
            "",
            "",
            "",
        )
        .await
    }

    async fn spawn_with_metadata_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        metadata_config: &str,
    ) -> Self {
        Self::spawn_with_options(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
            metadata_config,
        )
        .await
    }

    async fn spawn_with_options(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        metadata_config: &str,
    ) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            "",
            metadata_config,
            "",
            "",
        )
        .await
    }

    async fn spawn_with_observability_config(
        upstream_base_url: &str,
        observability_enabled: bool,
        observability_config: &str,
    ) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            observability_enabled,
            AppConfig::default().server.max_in_flight_requests,
            "",
            "",
            observability_config,
            "",
        )
        .await
    }

    async fn spawn_with_evidence_config(upstream_base_url: &str, evidence_config: &str) -> Self {
        Self::spawn_with_full_options(
            upstream_base_url,
            true,
            AppConfig::default().server.max_in_flight_requests,
            "",
            "",
            "",
            evidence_config,
        )
        .await
    }

    #[cfg(feature = "guard")]
    async fn spawn_with_extra_config(upstream_base_url: &str, extra_config: &str) -> Self {
        Self::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
            upstream_base_url,
            observability_enabled: true,
            max_in_flight_requests: AppConfig::default().server.max_in_flight_requests,
            server_config: "",
            metadata_config: "",
            observability_config: "",
            evidence_config: "",
            extra_config,
        })
        .await
    }

    async fn spawn_with_full_options(
        upstream_base_url: &str,
        observability_enabled: bool,
        max_in_flight_requests: usize,
        server_config: &str,
        metadata_config: &str,
        observability_config: &str,
        evidence_config: &str,
    ) -> Self {
        Self::spawn_with_full_options_and_extra(ProxyFixtureSpawnOptions {
            upstream_base_url,
            observability_enabled,
            max_in_flight_requests,
            server_config,
            metadata_config,
            observability_config,
            evidence_config,
            extra_config: "",
        })
        .await
    }

    async fn spawn_with_full_options_and_extra(options: ProxyFixtureSpawnOptions<'_>) -> Self {
        let root = unique_test_dir("proxy");
        fs::create_dir_all(&root).expect("test root should be created");
        set_owner_only_dir(&root);
        let config_path = root.join("config.toml");
        let sqlite_path = root.join("storage").join("observability.sqlite3");
        let evidence_sqlite_path = root.join("storage").join("evidence.sqlite3");
        #[cfg(feature = "guard")]
        let budget_sqlite_path = root.join("storage").join("budget.sqlite3");
        write_proxy_config_with_observability(ProxyConfigWriteOptions {
            config_path: &config_path,
            upstream_base_url: options.upstream_base_url,
            sqlite_path: &sqlite_path,
            evidence_sqlite_path: &evidence_sqlite_path,
            #[cfg(feature = "guard")]
            budget_sqlite_path: &budget_sqlite_path,
            observability_enabled: options.observability_enabled,
            max_in_flight_requests: options.max_in_flight_requests,
            server_config: options.server_config,
            metadata_config: options.metadata_config,
            observability_config: options.observability_config,
            evidence_config: options.evidence_config,
            extra_config: options.extra_config,
        });
        let manager =
            ConfigManager::from_explicit_path(&config_path).expect("proxy config should load");
        let store = ObservabilityStore::open(manager.handle()).expect("store should open");
        let evidence_store = EvidenceStore::open(manager.handle());
        #[cfg(feature = "guard")]
        let budget_store = Arc::new(
            BudgetStore::open(&budget_sqlite_path.display().to_string())
                .expect("budget store should open"),
        );
        let state = ProxyState::new(
            manager.handle(),
            manager.path().to_path_buf(),
            store.clone(),
            evidence_store,
            #[cfg(feature = "guard")]
            budget_store,
            build_http_client().expect("client should build"),
        );
        let app = router(state.clone());
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
            manager,
            state,
            store,
            sqlite_path,
            evidence_sqlite_path,
            #[cfg(feature = "guard")]
            budget_sqlite_path,
            root,
        }
    }
}

#[derive(Clone, Copy)]
struct ProxyFixtureSpawnOptions<'a> {
    upstream_base_url: &'a str,
    observability_enabled: bool,
    max_in_flight_requests: usize,
    server_config: &'a str,
    metadata_config: &'a str,
    observability_config: &'a str,
    evidence_config: &'a str,
    extra_config: &'a str,
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
    max_in_flight_requests: usize,
    metadata_config: &str,
) {
    let evidence_sqlite_path = sqlite_path.with_file_name("evidence.sqlite3");
    #[cfg(feature = "guard")]
    let budget_sqlite_path = sqlite_path.with_file_name("budget.sqlite3");
    write_proxy_config_with_observability(ProxyConfigWriteOptions {
        config_path,
        upstream_base_url,
        sqlite_path,
        evidence_sqlite_path: &evidence_sqlite_path,
        #[cfg(feature = "guard")]
        budget_sqlite_path: &budget_sqlite_path,
        observability_enabled,
        max_in_flight_requests,
        server_config: "",
        metadata_config,
        observability_config: "",
        evidence_config: "",
        extra_config: "",
    });
}

#[derive(Clone, Copy)]
struct ProxyConfigWriteOptions<'a> {
    config_path: &'a Path,
    upstream_base_url: &'a str,
    sqlite_path: &'a Path,
    evidence_sqlite_path: &'a Path,
    #[cfg(feature = "guard")]
    budget_sqlite_path: &'a Path,
    observability_enabled: bool,
    max_in_flight_requests: usize,
    server_config: &'a str,
    metadata_config: &'a str,
    observability_config: &'a str,
    evidence_config: &'a str,
    extra_config: &'a str,
}

fn write_proxy_config_with_observability(options: ProxyConfigWriteOptions<'_>) {
    #[cfg(feature = "guard")]
    let budget_section = format!(
        r#"
[budget]
sqlite_path = "{budget_sqlite_path}"
"#,
        budget_sqlite_path = options.budget_sqlite_path.display()
    );
    #[cfg(not(feature = "guard"))]
    let budget_section = String::new();
    fs::write(
        options.config_path,
        format!(
            r#"
[server]
max_in_flight_requests = {max_in_flight_requests}
{server_config}

[upstream]
base_url = "{upstream_base_url}"
{metadata_config}

[observability]
enabled = {observability_enabled}
sqlite_path = "{sqlite_path}"
capture_raw_payloads = false
{observability_config}

[observability.retention]
max_bytes = {TEST_MAX_BYTES}
prune_to_bytes = {TEST_PRUNE_TO_BYTES}
max_records = {TEST_MAX_RECORDS}

[evidence]
sqlite_path = "{evidence_sqlite_path}"
blob_cache_dir = "{blob_cache_dir}"
{evidence_config}
{budget_section}
{extra_config}
"#,
            max_in_flight_requests = options.max_in_flight_requests,
            server_config = options.server_config,
            upstream_base_url = options.upstream_base_url,
            metadata_config = options.metadata_config,
            observability_enabled = options.observability_enabled,
            sqlite_path = options.sqlite_path.display(),
            evidence_sqlite_path = options.evidence_sqlite_path.display(),
            blob_cache_dir = options
                .evidence_sqlite_path
                .parent()
                .expect("evidence sqlite path should have parent")
                .join("evidence-blobs")
                .display(),
            evidence_config = options.evidence_config,
            observability_config = options.observability_config,
            budget_section = budget_section,
            extra_config = options.extra_config,
        ),
    )
    .expect("test config should be written");
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let counter = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "llm-guard-proxy-{}-{nanos}-{counter}-{name}",
        std::process::id()
    ))
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

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct LinuxProcessIdentity {
    pid: u32,
    start_time_ticks: u64,
}

#[cfg(target_os = "linux")]
impl LinuxProcessIdentity {
    fn capture(pid: u32) -> Option<Self> {
        let (_state, start_time_ticks) = linux_process_state_and_start_time(pid)?;
        if start_time_ticks == 0 {
            return None;
        }
        Some(Self {
            pid,
            start_time_ticks,
        })
    }

    fn is_running(self) -> bool {
        linux_process_state_and_start_time(self.pid).is_some_and(|(state, start_time_ticks)| {
            state != 'Z' && start_time_ticks == self.start_time_ticks
        })
    }
}

#[cfg(target_os = "linux")]
async fn read_pid_file_after_ready(pid_path: &Path, ready_path: &Path) -> LinuxProcessIdentity {
    // Suite-load scheduling can delay shell startup well beyond 200ms.
    // Wait long enough for the readiness handshake, but fail closed with
    // cleanup-friendly diagnostics if the fixture still never appears.
    for _ in 0..500 {
        if ready_path.exists()
            && let Ok(text) = fs::read_to_string(pid_path)
            && let Ok(pid) = text.trim().parse::<u32>()
            && let Some(identity) = LinuxProcessIdentity::capture(pid)
        {
            return identity;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "pid readiness handshake failed: ready={} pid={}",
        ready_path.display(),
        pid_path.display()
    );
}

#[cfg(target_os = "linux")]
async fn assert_process_not_running(identity: LinuxProcessIdentity) {
    for _ in 0..20 {
        if !identity.is_running() {
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }
    kill_process_if_running(identity);
    panic!("process {} still appears to be running", identity.pid);
}

#[cfg(target_os = "linux")]
async fn assert_process_reaped(identity: LinuxProcessIdentity) {
    for _ in 0..20 {
        if linux_process_state_and_start_time(identity.pid).is_none() {
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }
    kill_process_if_running(identity);
    panic!("process {} was not reaped", identity.pid);
}

#[cfg(target_os = "linux")]
fn kill_process_if_running(identity: LinuxProcessIdentity) {
    if !identity.is_running() {
        return;
    }
    let Ok(pid) = i32::try_from(identity.pid) else {
        return;
    };
    let _signal_result = kill(Pid::from_raw(pid), Signal::SIGKILL);
}

#[cfg(target_os = "linux")]
fn linux_process_state_and_start_time(pid: u32) -> Option<(char, u64)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_prefix, suffix) = stat.rsplit_once(") ")?;
    let state = suffix.chars().next()?;
    // The suffix starts at field 3 (state); starttime is field 22.
    let start_time_ticks = suffix.split_whitespace().nth(19)?.parse().ok()?;
    Some((state, start_time_ticks))
}

async fn assert_no_upstream_request(fake: &mut FakeUpstream) {
    assert!(
        fake.recv_within(Duration::from_millis(100)).await.is_none(),
        "invalid proxy path must not be forwarded upstream"
    );
}

async fn drain_upstream_requests(fake: &mut FakeUpstream, wait: Duration) -> Vec<ObservedRequest> {
    let mut observed = Vec::new();
    while let Some(request) = fake.recv_within(wait).await {
        observed.push(request);
    }
    observed
}

fn assert_sensitive_query_absent(label: &str, text: &str) {
    for sensitive in [
        "sk-live",
        "api_key",
        "safe=ok",
        "?api_key=sk-live",
        "?api_key=sk-live&safe=ok",
    ] {
        assert!(
            !text.contains(sensitive),
            "{label} leaked sensitive query fragment {sensitive:?}: {text}"
        );
    }
}

fn assert_safe_operational_text(label: &str, text: &str) {
    for sensitive in [
        "sk-live-secret",
        "sk-header-secret",
        "downstream-secret",
        "Bearer downstream-secret",
    ] {
        assert!(
            !text.contains(sensitive),
            "{label} leaked sensitive value {sensitive:?}: {text}"
        );
    }
    let lowercase = text.to_ascii_lowercase();
    for sensitive_key in ["authorization", "x-api-key"] {
        assert!(
            !lowercase.contains(sensitive_key),
            "{label} leaked sensitive key {sensitive_key:?}: {text}"
        );
    }
}

async fn send_metrics_chat_request(proxy: &ProxyFixture, fake: &mut FakeUpstream, index: usize) {
    let body = serde_json::json!({
        "model": "test-chat",
        "messages": [{"role": "user", "content": format!("metrics pruning {index}")}],
    })
    .to_string();
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=metrics-pruning-{index}",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .expect("metrics chat request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let _body = response
        .text()
        .await
        .expect("metrics chat body should be consumed");
    let _observed = fake.recv_next().await;
}

async fn fetch_metrics(proxy: &ProxyFixture) -> String {
    let response = proxy
        .client
        .get(format!("{}/metrics", proxy.base_url))
        .send()
        .await
        .expect("metrics request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    response.text().await.expect("metrics should be text")
}

async fn fetch_debug_summary(proxy: &ProxyFixture) -> (String, serde_json::Value) {
    let response = proxy
        .client
        .get(format!("{}/debug/recent-requests?limit=5", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("debug summary request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("debug summary should be text");
    let summary: serde_json::Value =
        serde_json::from_str(&body).expect("debug summary should be JSON");
    (body, summary)
}

async fn assert_deadline_debug_summary(proxy: &ProxyFixture) {
    let (body, summary) = fetch_debug_summary(proxy).await;
    let request = summary["requests"]
        .as_array()
        .and_then(|requests| {
            requests.iter().find(|request| {
                request["response_metadata"]["request_deadline_exhausted"].as_str() == Some("true")
            })
        })
        .expect("debug summary should include the deadline request");

    assert_eq!(
        request["response_metadata"]["shielded_terminal_reason"].as_str(),
        Some("request_deadline_exhausted")
    );
    assert_eq!(
        request["response_metadata"]["retry_attempt_count"].as_str(),
        Some("1")
    );
    assert!(!body.contains("deadline-secret"));
    assert!(!body.contains("admin-token"));
}

async fn assert_body_error_debug_summary(proxy: &ProxyFixture) {
    let (body, summary) = fetch_debug_summary(proxy).await;
    let request = summary["requests"]
        .as_array()
        .and_then(|requests| {
            requests.iter().find(|request| {
                request["response_metadata"]["error_type"].as_str() == Some("upstream_body_error")
            })
        })
        .expect("debug summary should include the body error request");

    assert_eq!(
        request["response_metadata"]["retry_attempt_count"].as_str(),
        Some("1")
    );
    assert_eq!(
        request["response_metadata"]["upstream_response_received"].as_str(),
        Some("true")
    );
    assert_eq!(
        request["response_metadata"]["http_status_success"].as_str(),
        Some("true")
    );
    assert!(!body.contains("body-decode-secret"));
    assert!(!body.contains("admin-token"));
}

async fn wait_for_generation_metrics(
    proxy: &ProxyFixture,
    expected_active: u64,
    expected_queued: u64,
    wait: Duration,
) {
    timeout(wait, async {
        loop {
            let metrics = fetch_metrics(proxy).await;
            let active = metric_value(&metrics, "llm_guard_proxy_generation_active");
            let queued = metric_value(&metrics, "llm_guard_proxy_generation_queued");
            if active == expected_active && queued == expected_queued {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!("generation metrics did not reach active={expected_active} queued={expected_queued}")
    });
}

fn assert_metric_type(body: &str, metric_name: &str, metric_type: &str) {
    let expected = format!("# TYPE {metric_name} {metric_type}");
    assert!(
        body.contains(&expected),
        "metrics body missing expected type line {expected:?}: {body}"
    );
}

fn assert_legacy_retained_counter_metrics_absent(body: &str) {
    for metric_name in [
        "llm_guard_proxy_requests_total",
        "llm_guard_proxy_attempts_total",
        "llm_guard_proxy_retries_total",
        "llm_guard_proxy_loop_aborts_total",
        "llm_guard_proxy_upstream_errors_total",
        "llm_guard_proxy_heartbeat_mode_total",
        "llm_guard_proxy_first_token_latency_ms_bucket",
        "llm_guard_proxy_first_token_latency_ms_count",
        "llm_guard_proxy_first_token_latency_ms_sum",
        "llm_guard_proxy_total_latency_ms_bucket",
        "llm_guard_proxy_total_latency_ms_count",
        "llm_guard_proxy_total_latency_ms_sum",
    ] {
        assert!(
            !body.contains(metric_name),
            "metrics body still exposes legacy retained metric {metric_name:?}: {body}"
        );
    }
}

fn metric_value(body: &str, metric_name: &str) -> u64 {
    let prefix = format!("{metric_name} ");
    body.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("metrics body missing numeric metric {metric_name:?}: {body}"))
}

fn labelled_metric_value(body: &str, metric_name: &str, labels: &[(&str, &str)]) -> u64 {
    let prefix = format!("{metric_name}{{");
    body.lines()
        .find(|line| {
            line.starts_with(&prefix)
                && labels
                    .iter()
                    .all(|(name, value)| line.contains(&format!(r#"{name}="{value}""#)))
        })
        .and_then(|line| {
            line.rsplit_once(' ')
                .and_then(|(_labels, value)| value.parse().ok())
        })
        .unwrap_or_else(|| {
            panic!("metrics body missing labelled metric {metric_name:?} {labels:?}: {body}")
        })
}

async fn send_raw_proxy_get(base_url: &str, request_target: &str) -> String {
    let base_url = base_url.to_owned();
    let request_target = request_target.to_owned();
    tokio::task::spawn_blocking(move || {
        let url = Url::parse(&base_url).expect("proxy base URL should parse");
        let host = url.host_str().expect("proxy base URL should have a host");
        let port = url.port().expect("proxy base URL should have a port");
        let addr = format!("{host}:{port}");
        let mut stream = std::net::TcpStream::connect(&addr)
            .unwrap_or_else(|error| panic!("proxy TCP connection should open: {error}"));
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout should be set");
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("write timeout should be set");
        write!(
            stream,
            "GET {request_target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
        )
        .expect("raw proxy request should write");

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("raw proxy response should read");
        response
    })
    .await
    .expect("blocking raw proxy request should finish")
}

fn raw_response_header<'response>(response: &'response str, name: &str) -> Option<&'response str> {
    response.split("\r\n").find_map(|line| {
        let (header_name, value) = line.split_once(':')?;
        header_name
            .eq_ignore_ascii_case(name)
            .then_some(value.trim())
    })
}

fn terminal_response_request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .expect("terminal response should include x-request-id")
        .to_str()
        .expect("x-request-id should be valid header text")
        .to_owned()
}

fn assert_response_request_id_matches_persisted_request(
    response_request_id: &str,
    proxy: &ProxyFixture,
) {
    assert_eq!(
        response_request_id,
        read_last_observability_row(&proxy.sqlite_path, "requests").request_id
    );
}

// ---------------------------------------------------------------------------
// Issue #124: non-stream chat completion response shape validation
// ---------------------------------------------------------------------------

fn build_json_chat_response(status: StatusCode, body: &str) -> Response<Body> {
    let mut response = Response::new(Body::from(body.to_owned()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn build_streaming_chat_response(status: StatusCode) -> Response<Body> {
    let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    let mut response = Response::new(Body::from(body.to_owned()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

fn request_id_for_test() -> RequestId {
    RequestId::from_string("req-test-124").expect("request id should be valid")
}

#[tokio::test]
async fn validation_passes_through_valid_choices() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hello"}}]}"#;
    let response = build_json_chat_response(StatusCode::OK, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::OK);
    let body_bytes = to_bytes(result.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    assert!(body_bytes.windows(7).any(|w| w == b"choices"));
    assert_eq!(counter.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn validation_converts_missing_choices_to_502() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"id":"chatcmpl-1","object":"chat.completion"}"#;
    let response = build_json_chat_response(StatusCode::OK, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(counter.load(Ordering::Relaxed), 1);
    let body_bytes = to_bytes(result.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let parsed: serde_json::Value =
        serde_json::from_slice(&body_bytes).expect("error body should be valid JSON");
    assert_eq!(
        parsed["error"]["code"],
        serde_json::json!("malformed_response")
    );
    assert_eq!(parsed["error"]["type"], serde_json::json!("upstream_error"));
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("choices"))
    );
}

#[tokio::test]
async fn validation_converts_null_choices_to_502() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"id":"chatcmpl-1","choices":null}"#;
    let response = build_json_chat_response(StatusCode::OK, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn validation_converts_invalid_json_to_502() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = "not valid json at all {{{";
    let mut response = Response::new(Body::from(body.to_owned()));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn validation_passes_through_non_2xx() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"error":"rate limited"}"#;
    let response = build_json_chat_response(StatusCode::TOO_MANY_REQUESTS, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(counter.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn validation_passes_through_streaming_response() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let response = build_streaming_chat_response(StatusCode::OK);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::OK);
    assert_eq!(counter.load(Ordering::Relaxed), 0);
    assert_eq!(
        result.headers().get(CONTENT_TYPE).unwrap(),
        HeaderValue::from_static("text/event-stream")
    );
}

#[tokio::test]
async fn validation_passes_through_non_chat_endpoint() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"object":"list","data":[]}"#;
    let response = build_json_chat_response(StatusCode::OK, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/v1/embeddings",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::OK);
    assert_eq!(counter.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn validation_error_response_includes_request_id() {
    let counter = AtomicU64::new(0);
    let request_id = request_id_for_test();
    let body = r#"{"id":"chatcmpl-1"}"#;
    let response = build_json_chat_response(StatusCode::OK, body);
    let result = validate_non_stream_chat_completion_response(
        response,
        "/chat/completions",
        &request_id,
        &counter,
    )
    .await;
    assert_eq!(result.status(), StatusCode::BAD_GATEWAY);
    let request_id_header = result
        .headers()
        .get("x-request-id")
        .expect("x-request-id header should be present");
    assert_eq!(request_id_header, "req-test-124");
}

#[tokio::test]
async fn upstream_transport_error_response_includes_cause_and_request_id() {
    let request_id = request_id_for_test();
    let error = ProxyError::UpstreamTransport {
        failure: ReqwestFailureKind::Connect,
        observability: None,
    };
    let response = proxy_error_response_from_error_with_diagnostics(&error, Some(&request_id));
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body_bytes = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let parsed: serde_json::Value =
        serde_json::from_slice(&body_bytes).expect("error body should be valid JSON");
    assert_eq!(
        parsed["error"]["cause"],
        serde_json::json!("upstream_connect_failed")
    );
    assert_eq!(
        parsed["error"]["code"],
        serde_json::json!("upstream_connect_failed")
    );
    assert_eq!(
        parsed["error"]["request_id"],
        serde_json::json!(request_id.as_str())
    );
}

#[tokio::test]
async fn upstream_timeout_error_response_classifies_as_timeout() {
    let error = ProxyError::UpstreamTransport {
        failure: ReqwestFailureKind::Timeout,
        observability: None,
    };
    let response = proxy_error_response_from_error_with_diagnostics(&error, None);
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body_bytes = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let parsed: serde_json::Value =
        serde_json::from_slice(&body_bytes).expect("error body should be valid JSON");
    assert_eq!(
        parsed["error"]["cause"],
        serde_json::json!("upstream_timeout")
    );
}

#[tokio::test]
async fn upstream_body_error_response_classifies_as_body_error() {
    let error = ProxyError::UpstreamBody {
        reason: String::from("connection reset"),
        observability: None,
    };
    let response = proxy_error_response_from_error_with_diagnostics(&error, None);
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body_bytes = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("body should read");
    let parsed: serde_json::Value =
        serde_json::from_slice(&body_bytes).expect("error body should be valid JSON");
    assert_eq!(
        parsed["error"]["cause"],
        serde_json::json!("upstream_body_error")
    );
}

#[test]
fn shielded_failure_cause_prefers_structured_body_error_over_upstream_message() {
    let failure = ShieldedFailureOutcome {
        error_type: "llm_guard_upstream_error",
        error_message: String::from("upstream SSE ended without chat completion choices"),
        response_metadata: BTreeMap::from([(
            String::from("error_type"),
            String::from("upstream_body_error"),
        )]),
        attempt_records: Vec::new(),
        upstream_mode: UpstreamMode::Streaming,
        downstream_status: StatusCode::BAD_GATEWAY,
        retry_after_secs: None,
    };

    assert_eq!(
        classify_shielded_failure_cause(&failure),
        Some(UpstreamFailureCause::BodyError)
    );
}

#[tokio::test]
async fn upstream_failure_counters_increment_by_cause() {
    let counters = UpstreamFailureCounters::default();
    counters.increment(UpstreamFailureCause::ConnectFailed);
    counters.increment(UpstreamFailureCause::Timeout);
    counters.increment(UpstreamFailureCause::Timeout);
    counters.increment(UpstreamFailureCause::StatusError);
    let snapshot = counters.snapshot();
    assert_eq!(snapshot.connect_failed, 1);
    assert_eq!(snapshot.timeout, 2);
    assert_eq!(snapshot.body_error, 0);
    assert_eq!(snapshot.status_error, 1);
    assert_eq!(snapshot.transport_error, 0);
}

#[test]
fn upstream_failure_metrics_render_all_cause_labels() {
    let snapshot = UpstreamFailureSnapshot {
        connect_failed: 1,
        timeout: 2,
        body_error: 3,
        status_error: 4,
        transport_error: 5,
    };
    let mut output = String::new();
    push_upstream_failure_metrics(&mut output, snapshot);
    assert!(output.contains("llm_guard_proxy_upstream_failure_total{cause=\"connect_failed\"} 1"));
    assert!(output.contains("llm_guard_proxy_upstream_failure_total{cause=\"timeout\"} 2"));
    assert!(output.contains("llm_guard_proxy_upstream_failure_total{cause=\"body_error\"} 3"));
    assert!(output.contains("llm_guard_proxy_upstream_failure_total{cause=\"status_error\"} 4"));
    assert!(output.contains("llm_guard_proxy_upstream_failure_total{cause=\"transport_error\"} 5"));
}

#[tokio::test]
async fn health_chat_probe_reports_ready_when_upstream_healthy() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        "health_upstream_probe_timeout_ms = 100\nhealth_chat_probe_enabled = true\nhealth_chat_probe_timeout_ms = 100\n",
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    assert_eq!(body["upstream"], "ready");

    // The models probe should have been observed.
    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/models");
}

#[tokio::test]
async fn health_chat_probe_reports_degraded_when_chat_fails() {
    // BrokenUpstream accepts connections but immediately drops them, so the
    // lightweight /v1/models probe may succeed (axum returns 200 for the
    // listener) but the chat probe will fail with a transport error.
    let broken = BrokenUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &broken.base_url,
        true,
        "health_upstream_probe_timeout_ms = 100\nhealth_chat_probe_enabled = true\nhealth_chat_probe_timeout_ms = 100\n",
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/health", proxy.base_url))
        .send()
        .await
        .expect("health request should complete");
    // Either unavailable (models probe failed) or degraded (models ok, chat
    // failed). Both are non-ready states; the key assertion is that we do not
    // report "ready" when chat completions are broken.
    let body_text = response.text().await.expect("health body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("health should be JSON");
    let upstream = body["upstream"].as_str().unwrap_or_default();
    assert!(
        upstream == "degraded" || upstream == "unavailable",
        "expected degraded or unavailable, got {upstream}"
    );
}

#[tokio::test]
async fn debug_live_requests_disabled_by_default() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&fake.base_url, true).await;

    let response = proxy
        .client
        .get(format!("{}/debug/requests", proxy.base_url))
        .send()
        .await
        .expect("debug live requests request should complete");
    // When debug_summary_enabled is disabled, the endpoint returns 403 Forbidden.
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn debug_live_requests_returns_empty_array_when_no_active_requests() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
"#,
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/debug/requests", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("debug live requests request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let body_text = response.text().await.expect("body should be text");
    let body: serde_json::Value = serde_json::from_str(&body_text).expect("body should be JSON");
    assert_eq!(body["request_count"], 0);
    assert!(body["requests"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn debug_live_request_detail_returns_404_for_unknown_id() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
"#,
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/debug/requests/nonexistent-id", proxy.base_url))
        .header(AUTHORIZATION, "Bearer admin-token")
        .send()
        .await
        .expect("debug live request detail request should complete");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn debug_live_requests_unauthorized_without_token() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_observability_config(
        &fake.base_url,
        true,
        r#"debug_summary_enabled = true
debug_summary_admin_token = "admin-token"
"#,
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/debug/requests", proxy.base_url))
        .send()
        .await
        .expect("debug live requests request should complete");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
