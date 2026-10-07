use super::*;

fn fixture() -> (
    ObservabilityStore,
    PathBuf,
    UpstreamProfileConfig,
    Arc<PersistenceTasks>,
) {
    let root = unique_test_dir("pre-action-receipt");
    fs::create_dir_all(&root).expect("receipt root");
    set_owner_only_dir(&root);
    let mut config = AppConfig::default();
    config.observability.enabled = false;
    config.observability.capture_raw_payloads = true;
    config.observability.sqlite_path = root.join("receipt.sqlite3");
    let store = ObservabilityStore::open(ConfigHandle::new(config.clone())).expect("receipt store");
    (
        store,
        config.observability.sqlite_path.clone(),
        config.default_upstream_profile(),
        Arc::new(PersistenceTasks::default()),
    )
}

fn policy(command: Vec<String>) -> LocalRecoveryPolicy {
    LocalRecoveryPolicy {
        enabled: true,
        restart_command: command,
        restart_timeout: Duration::from_secs(2),
        readiness_deadline: Duration::from_secs(1),
        ..LocalRecoveryPolicy::from_config(&LocalRecoveryConfig::default())
    }
}

fn receipt_row(path: &Path, id: &str) -> (serde_json::Value, Option<String>) {
    let connection = Connection::open(path).expect("read receipt");
    let (json, outcome): (String, Option<String>) = connection
        .query_row(
            "SELECT receipt_json, outcome FROM local_recovery_receipts WHERE receipt_id = ?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("durable receipt exists");
    (serde_json::from_str(&json).expect("receipt JSON"), outcome)
}

#[tokio::test]
async fn local_recovery_receipt_precedes_child_and_keeps_causes_private() {
    let (store, path, mut profile, tasks) = fixture();
    let trace = recovery_diagnostics::ReceiptTrace::install(&store);
    profile.name = String::from("https://private-user:private-password@host");
    let marker = path.with_extension("accepted");
    let child = "import sqlite3,sys,pathlib,json; c=sqlite3.connect(sys.argv[1]); r=c.execute('SELECT receipt_json FROM local_recovery_receipts WHERE receipt_id=?',(sys.argv[2],)).fetchone(); assert r is not None; assert json.loads(r[0])['cause']==sys.argv[4]; pathlib.Path(sys.argv[3]).write_text('accepted')";
    for cause in [
        LocalRecoveryCause::RequestDeadline,
        LocalRecoveryCause::TransientTransport,
        LocalRecoveryCause::TransientStatus,
        LocalRecoveryCause::UpstreamStall,
        LocalRecoveryCause::StuckWatchdog,
    ] {
        let context = recovery_receipt::Context::new(store.clone(), &profile, Arc::clone(&tasks))
            .stall(123, 456);
        let policy = policy(vec![
            String::from("python3"),
            String::from("-c"),
            child.to_owned(),
            path.display().to_string(),
            context.id().to_owned(),
            marker.display().to_string(),
            cause.as_str().to_owned(),
            String::from("raw-command-secret-sentinel"),
        ]);
        let context = context.episode(7, cause, &policy);
        let ran = AtomicBool::new(false);
        let result = run_local_recovery_restart_command(&policy, &ran, &context, None).await;
        assert_eq!(
            result["local_recovery_restart_status"],
            "succeeded",
            "recovery={} receipt_stages_us={:?}; historical_GF1=UNKNOWN",
            recovery_diagnostics::recovery_metadata(&json!(result)),
            trace.snapshot()
        );
        assert!(marker.exists());
        fs::remove_file(&marker).expect("remove accepted marker");
        let (receipt, outcome) = receipt_row(&path, context.id());
        assert_eq!(receipt["cause"], cause.as_str());
        assert_eq!(
            receipt["detector"],
            if cause == LocalRecoveryCause::StuckWatchdog {
                "stuck_watchdog"
            } else {
                "precommit_recovery"
            }
        );
        assert_eq!(receipt["profile"], "unknown");
        assert_eq!(receipt["episode"], 7);
        assert_eq!(receipt["first_chunk_timeout_ms"], 123);
        assert_eq!(receipt["idle_timeout_ms"], 456);
        assert!(receipt["generated_at_unix_ms"].as_u64().expect("timestamp") > 0);
        assert!(receipt["command_generation"].as_u64().expect("generation") > 0);
        assert_eq!(outcome, None);
        let serialized = receipt.to_string();
        for secret in [
            "private-user",
            "private-password",
            "raw-command-secret-sentinel",
            child,
        ] {
            assert!(!serialized.contains(secret));
        }
    }
    let connection = Connection::open(path).expect("read independent receipt");
    let requests: u32 = connection
        .query_row("SELECT COUNT(*) FROM requests", [], |row| row.get(0))
        .expect("request count");
    assert_eq!(
        requests, 0,
        "pre-action receipts must not invent completed requests"
    );
}

#[tokio::test]
async fn local_recovery_receipt_volatile_store_never_spawns() {
    let (isolated_store, path, profile, tasks) = fixture();
    let isolated_trace = recovery_diagnostics::ReceiptTrace::install(&isolated_store);
    let marker = path.with_extension("volatile-spawn-forbidden");
    let mut config = AppConfig::default();
    config.observability.enabled = false;
    config.observability.sqlite_path = PathBuf::from(":memory:");
    let store = ObservabilityStore::open(ConfigHandle::new(config)).expect("memory store");
    let trace = recovery_diagnostics::ReceiptTrace::install(&store);
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks)).episode(
        1,
        LocalRecoveryCause::TransientTransport,
        &policy,
    );
    let ran = AtomicBool::new(false);
    let result = run_local_recovery_restart_command(&policy, &ran, &context, None).await;
    assert_eq!(result["local_recovery_restart_status"], "receipt_failed");
    assert_eq!(
        result["local_recovery_receipt_error"],
        "durability_disabled"
    );
    assert_eq!(context.acknowledged_id(), None);
    assert!(!ran.load(Ordering::Relaxed));
    assert!(!marker.exists());
    assert_eq!(
        recovery_diagnostics::recovery_metadata(&json!(result))["local_recovery_receipt_error"],
        "durability_disabled"
    );
    let stages = trace.snapshot();
    assert_eq!(
        stages.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
        vec![
            "record_submit",
            "record_worker_start",
            "record_sql_start",
            "record_sql_end",
            "record_ack"
        ]
    );
    assert!(stages.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    assert!(
        isolated_trace.snapshot().is_empty(),
        "observer is store-local"
    );
    tasks.flush(Duration::from_secs(1)).await;
    assert_eq!(tasks.in_flight.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn local_recovery_receipt_write_failure_and_cancel_never_spawn() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("forbidden");
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let lock = Connection::open(&path).expect("external writer");
    lock.execute_batch("BEGIN EXCLUSIVE")
        .expect("hold SQLite lock");
    let context = recovery_receipt::Context::new(store.clone(), &profile, Arc::clone(&tasks))
        .episode(1, LocalRecoveryCause::TransientTransport, &policy);
    let ran = AtomicBool::new(false);
    let responsive = Arc::new(AtomicBool::new(false));
    let tick = Arc::clone(&responsive);
    let heartbeat = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        tick.store(true, Ordering::Relaxed);
    });
    let result = timeout(
        Duration::from_secs(1),
        run_local_recovery_restart_command(&policy, &ran, &context, None),
    )
    .await
    .expect("bounded write gate");
    assert!(
        responsive.load(Ordering::Relaxed),
        "SQLite contention must not block the runtime"
    );
    heartbeat.await.expect("runtime heartbeat");
    assert_eq!(result["local_recovery_restart_status"], "receipt_failed");
    assert!(!ran.load(Ordering::Relaxed));
    assert!(!marker.exists());
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks)).episode(
        2,
        LocalRecoveryCause::RequestDeadline,
        &policy,
    );
    let cancelled = timeout(
        Duration::from_millis(20),
        run_local_recovery_restart_command(&policy, &ran, &context, None),
    )
    .await;
    assert!(cancelled.is_err());
    lock.execute_batch("ROLLBACK").expect("release SQLite lock");
    // The abandoned writer may still commit; it owns no restart authority.
    tasks.flush(Duration::from_secs(1)).await;
    assert_eq!(tasks.in_flight.load(Ordering::SeqCst), 0);
    assert!(!ran.load(Ordering::Relaxed));
    assert!(!marker.exists());
}

#[tokio::test]
async fn local_recovery_receipt_child_cancellation_keeps_durable_attribution() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("child-pid");
    let forbidden = path.with_extension("after-sleep");
    let child = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30); pathlib.Path(sys.argv[2]).write_text('forbidden')";
    let mut policy = policy(vec![
        String::from("python3"),
        String::from("-c"),
        child.to_owned(),
        marker.display().to_string(),
        forbidden.display().to_string(),
    ]);
    policy.restart_timeout = Duration::from_secs(40);
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks));
    let id = context.id().to_owned();
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let task_coordinator = Arc::clone(&coordinator);
    let waiter = tokio::spawn(async move {
        run_local_recovery_for_profile_observing(
            &policy,
            &task_coordinator,
            build_http_client().expect("client"),
            String::from("http://127.0.0.1:1/v1"),
            LocalRecoveryCause::TransientTransport,
            LocalRecoveryRunOptions {
                receipt: context,
                episode_timeout: None,
                caller_timeout: None,
                recovery_episode_observer: None,
                downstream_commit_signal: None,
                post_await_self_test: None,
            },
        )
        .await
    });
    timeout(Duration::from_secs(2), async {
        while !marker.exists() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("synthetic child started");
    let (receipt, pending) = receipt_row(&path, &id);
    assert_eq!(receipt["cause"], "transient_transport");
    assert_eq!(pending, None);
    let pid: i32 = fs::read_to_string(&marker)
        .expect("child pid")
        .parse()
        .expect("synthetic pid");
    coordinator
        .state
        .lock()
        .await
        .active_local_recovery_task
        .as_ref()
        .expect("owned recovery task")
        .abort();
    let result = timeout(Duration::from_secs(2), waiter)
        .await
        .expect("bounded cancel")
        .expect("recovery waiter");
    assert_eq!(result["local_recovery_status"], "cancelled");
    assert_eq!(result["local_recovery_receipt_id"], id);
    tasks.flush(Duration::from_secs(1)).await;
    assert_eq!(tasks.in_flight.load(Ordering::SeqCst), 0);
    let (_, outcome) = receipt_row(&path, &id);
    assert_eq!(outcome.as_deref(), Some("cancelled"));
    timeout(Duration::from_secs(2), async {
        while kill(Pid::from_raw(pid), None).is_ok() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("synthetic child reaped");
    assert!(!forbidden.exists());
}

#[tokio::test]
async fn local_recovery_receipt_late_acknowledgment_never_spawns() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("late-spawn-forbidden");
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks)).episode(
        1,
        LocalRecoveryCause::TransientTransport,
        &policy,
    );
    let ran = AtomicBool::new(false);
    let lock = Connection::open(&path).expect("late-ack fixture lock");
    lock.execute_batch("BEGIN EXCLUSIVE")
        .expect("hold first poll");
    let mut restart = Box::pin(run_local_recovery_restart_command(
        &policy, &ran, &context, None,
    ));
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(restart.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    lock.execute_batch("ROLLBACK")
        .expect("release first-poll lock");
    // Deliberately stop this current-thread executor from receiving the acknowledgment.
    // The tracked blocking writer completes independently, making its result ready first.
    let worker_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while tasks.in_flight.load(Ordering::SeqCst) != 0 {
        assert!(
            std::time::Instant::now() < worker_deadline,
            "bounded synthetic writer"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(receipt_row(&path, context.id()).1.is_none());
    std::thread::sleep(Duration::from_millis(300));
    let result = restart.await;
    assert_eq!(result["local_recovery_restart_status"], "receipt_failed");
    assert_eq!(result["local_recovery_receipt_error"], "write_timeout");
    assert_eq!(context.acknowledged_id(), None);
    assert!(!ran.load(Ordering::Relaxed));
    assert!(!marker.exists());
}

#[tokio::test]
async fn local_recovery_receipt_shutdown_during_write_never_spawns() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("shutdown-during-write-forbidden");
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let shutdown = Arc::new(ShutdownGate::new());
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks))
        .episode(1, LocalRecoveryCause::StuckWatchdog, &policy)
        .shutdown(Arc::clone(&shutdown));
    let ran = AtomicBool::new(false);
    let lock = Connection::open(&path).expect("shutdown fixture lock");
    lock.execute_batch("BEGIN EXCLUSIVE")
        .expect("hold first poll");
    let mut restart = Box::pin(run_local_recovery_restart_command(
        &policy, &ran, &context, None,
    ));
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(restart.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    lock.execute_batch("ROLLBACK")
        .expect("release first-poll lock");
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while tasks.in_flight.load(Ordering::SeqCst) != 0 {
        assert!(std::time::Instant::now() < deadline, "bounded writer");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(receipt_row(&path, context.id()).1.is_none());
    shutdown.begin_shutdown();
    let result = restart.await;
    assert!(
        !ran.load(Ordering::Relaxed),
        "shutdown owner must not launch a child"
    );
    assert!(!marker.exists());
    assert_eq!(
        result.get("local_recovery_status").map(String::as_str),
        Some("shutdown_cancelled")
    );
}

#[tokio::test]
async fn local_recovery_receipt_downstream_drop_during_write_never_spawns() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("drop-during-write-forbidden");
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let dropped = DownstreamDropSignal::default();
    let attempts = AtomicU64::new(0);
    let lock = Connection::open(&path).expect("drop fixture lock");
    lock.execute_batch("BEGIN EXCLUSIVE").expect("hold writer");
    let mut recovery = Box::pin(precommit_recovery::gate(
        precommit_recovery::Context {
            receipt: recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks)),
            policy: &policy,
            coordinator: &coordinator,
            client: build_http_client().expect("client"),
            base_url: "http://127.0.0.1:1/v1",
            profile_name: "default",
            attempts: &attempts,
            downstream_commit_signal: None,
            downstream_drop_signal: Some(&dropped),
            request_deadline: RequestDeadline::from_started_at(
                Instant::now(),
                Duration::from_secs(2),
            ),
            post_await_self_test: None,
            episode_timeout: None,
        },
        true,
        LocalRecoveryCause::TransientTransport,
    ));
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(recovery.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    timeout(Duration::from_secs(1), async {
        while tasks.in_flight.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("writer plus terminalizer pending");
    dropped.mark_dropped();
    lock.execute_batch("ROLLBACK").expect("release writer");
    let _ = recovery.await;
    tasks.flush(Duration::from_secs(1)).await;
    assert!(!marker.exists(), "dropped owner must not launch restart");
}

#[tokio::test]
async fn local_recovery_receipt_short_owner_deadline_never_spawns() {
    let (store, path, profile, tasks) = fixture();
    let marker = path.with_extension("owner-deadline-forbidden");
    let policy = policy(vec![
        String::from("/usr/bin/touch"),
        marker.display().to_string(),
    ]);
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks)).episode(
        1,
        LocalRecoveryCause::TransientTransport,
        &policy,
    );
    let lock = Connection::open(&path).expect("owner-deadline fixture lock");
    lock.execute_batch("BEGIN EXCLUSIVE")
        .expect("hold first poll");
    let mut recovery = Box::pin(run_local_recovery_task(
        policy,
        build_http_client().expect("client"),
        String::from("http://127.0.0.1:1/v1"),
        LocalRecoveryCause::TransientTransport,
        Some(Duration::from_millis(80)),
        LocalRecoveryTaskContext {
            downstream_commit_signal: None,
            post_await_self_test: None,
            receipt: context.clone(),
        },
    ));
    let started = std::time::Instant::now();
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(recovery.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    lock.execute_batch("ROLLBACK")
        .expect("release first-poll lock");
    while tasks.in_flight.load(Ordering::SeqCst) != 0 {
        assert!(started.elapsed() < Duration::from_secs(2), "bounded writer");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(receipt_row(&path, context.id()).1.is_none());
    std::thread::sleep(Duration::from_millis(120));
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "resume before independent receipt deadline"
    );
    let result = recovery.await;
    assert_ne!(
        result.get("local_recovery_restart_ran").map(String::as_str),
        Some("true"),
        "expired owner must not launch a child"
    );
    assert!(!marker.exists());
}

#[tokio::test]
async fn local_recovery_receipt_terminalizer_joins_command_and_readiness() {
    let (store, path, profile, tasks) = fixture();
    let fake = FakeUpstream::spawn().await;
    let policy = policy(vec![String::from("/bin/true")]);
    let context = recovery_receipt::Context::new(store, &profile, Arc::clone(&tasks));
    let id = context.id().to_owned();
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let result = run_local_recovery_for_profile_observing(
        &policy,
        &coordinator,
        build_http_client().expect("client"),
        fake.base_url.clone(),
        LocalRecoveryCause::StuckWatchdog,
        LocalRecoveryRunOptions {
            receipt: context,
            episode_timeout: None,
            caller_timeout: None,
            recovery_episode_observer: None,
            downstream_commit_signal: None,
            post_await_self_test: None,
        },
    )
    .await;
    assert_eq!(result["local_recovery_status"], "succeeded");
    assert_eq!(result["local_recovery_receipt_id"], id);
    tasks.flush(Duration::from_secs(1)).await;
    assert_eq!(tasks.in_flight.load(Ordering::SeqCst), 0);
    let (_, outcome) = receipt_row(&path, &id);
    assert_eq!(outcome.as_deref(), Some("succeeded"));
    let joined = completed_local_recovery_metadata(Some(&result), true);
    assert_eq!(joined["local_recovery_receipt_id"], id);
}
