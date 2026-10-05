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
        assert_eq!(result["local_recovery_restart_status"], "succeeded");
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
