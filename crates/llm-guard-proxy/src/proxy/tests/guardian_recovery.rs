use super::{Arc, ConfigHandle, Duration, FakeUpstream, ProxyFixture, fs};
use llm_guard_proxy_host_guardian::{
    CgroupTarget,
    monitor::tier2::{RecoveryAuthority, RecoveryOutcome, RecoveryRequest},
};
use std::{os::unix::fs::PermissionsExt, path::PathBuf};
use tokio::sync::oneshot;

async fn fixture(command: Vec<String>) -> (FakeUpstream, ProxyFixture, PathBuf) {
    let upstream = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, false).await;
    let mut config = proxy.state.config.snapshot().expect("config");
    config.guardian.enabled = true;
    config.guardian.target_label = String::from("test");
    config.guardian.registration_file = Some(String::from("target.v1"));
    config.guardian.escalation_enabled = true;
    config.guardian.escalation_profile = config.default_upstream_profile().name;
    config.guardian.cgroup_root = proxy.root.join("cgroups");
    config.upstream.local_recovery.enabled = true;
    config.upstream.local_recovery.restart_command = command;
    config.upstream.local_recovery.cooldown_ms = 100;
    config.upstream.local_recovery.restart_timeout_ms = 3000;
    config.upstream.local_recovery.readiness_deadline_ms = 1000;
    config.upstream.local_recovery.readiness_interval_ms = 20;
    config.validate().expect("valid fixture policy");
    proxy
        .state
        .config
        .apply_reloadable(&config)
        .expect("policy");
    let runtime = proxy.root.join("runtime");
    fs::create_dir_all(&runtime).expect("runtime");
    let uid = nix::unistd::Uid::effective().as_raw();
    let id = "a".repeat(64);
    let group =
        format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{id}.scope");
    let cgroup = config
        .guardian
        .cgroup_root
        .join(group.trim_start_matches('/'));
    fs::create_dir_all(&cgroup).expect("cgroup fixture");
    fs::write(cgroup.join("cgroup.kill"), b"1").expect("kill fixture");
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("events fixture");
    let registration = runtime.join("target.v1");
    fs::write(
        &registration,
        format!("version=1\ncontainer_id={id}\nscope=docker-{id}.scope\ncontrol_group={group}\n"),
    )
    .expect("registration");
    fs::set_permissions(registration, fs::Permissions::from_mode(0o600))
        .expect("registration mode");
    fs::write(runtime.join("meminfo"), b"MemAvailable: 0 kB\n").expect("pressure input");
    (upstream, proxy, runtime)
}

fn make_request(
    config: ConfigHandle,
    runtime: &std::path::Path,
    timeout: Duration,
) -> (RecoveryRequest, oneshot::Receiver<RecoveryOutcome>) {
    let policy = config.guardian_snapshot().expect("policy");
    let cgroup = policy.cgroup_root.join(format!(
        "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope",
        nix::unistd::Uid::effective(),
        nix::unistd::Uid::effective(),
        "a".repeat(64)
    ));
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("new isolated episode");
    let target = CgroupTarget::open_registered(&runtime.join("target.v1"), &policy.cgroup_root)
        .expect("retained generation");
    fs::write(cgroup.join("cgroup.events"), b"populated 0\n").expect("Tier 1 empty fixture");
    let profile = policy.escalation_profile.clone();
    let authority = Arc::new(RecoveryAuthority::new(
        target,
        config,
        policy,
        runtime.to_path_buf(),
        std::fs::File::open(runtime.join("meminfo")).expect("pressure fd"),
        std::time::Instant::now() + timeout,
    ));
    let (completion, receiver) = oneshot::channel();
    (
        RecoveryRequest {
            authority,
            episode: 1,
            profile,
            available_bytes: 0,
            threshold_bytes: 1024 * 1024 * 1024,
            grace_secs: 60,
            completion,
        },
        receiver,
    )
}

async fn run(
    proxy: &ProxyFixture,
    request: RecoveryRequest,
    receiver: oneshot::Receiver<RecoveryOutcome>,
) -> RecoveryOutcome {
    let (sender, worker) = proxy.state.spawn_guardian_recovery();
    sender.send(request).await.expect("handoff");
    let outcome = tokio::time::timeout(Duration::from_secs(6), receiver)
        .await
        .expect("bounded owner")
        .expect("completion ack");
    drop(sender);
    worker.await.expect("joined worker");
    outcome
}

#[tokio::test]
async fn tier2_owned_success_has_durable_native_receipt_and_boot_claim() {
    let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    let (request, receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    assert_eq!(
        run(&proxy, request, receiver).await,
        RecoveryOutcome::Succeeded
    );
    let database = rusqlite::Connection::open(&proxy.sqlite_path).expect("database");
    let (json, outcome): (String, String) = database
        .query_row(
            "SELECT receipt_json, outcome FROM local_recovery_receipts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("receipt");
    let json: serde_json::Value = serde_json::from_str(&json).expect("json");
    assert_eq!(json["detector"], "memory_guardian");
    assert_eq!(json["cause"], "memory_pressure");
    assert_eq!(json["command_id"], "guardian.foreground_recovery");
    assert_eq!(json["guardian"]["guardian_episode"], 1);
    assert!(json["request_id"].is_null());
    assert_eq!(outcome, "succeeded");
    assert!(!json.to_string().contains("/usr/bin/true"));
    database
        .execute("DELETE FROM local_recovery_receipts", [])
        .expect("simulate retention");
    let claims: u32 = database
        .query_row("SELECT count(*) FROM guardian_boot_claim", [], |row| {
            row.get(0)
        })
        .expect("durable boot claim");
    assert_eq!(claims, 1);
    let rows: u32 = database
        .query_row("SELECT count(*) FROM requests", [], |row| row.get(0))
        .expect("no fake completed request");
    assert_eq!(rows, 0);
    // A fresh coordinator cannot reset the durable claim; the live store retains
    // its exclusive writer lease. The independent SQLite read above proves persistence.
    let mut restarted = proxy.state.clone();
    restarted.local_recovery = Arc::new(super::LocalRecoveryCoordinatorSet::default());
    let (duplicate, receiver) =
        make_request(restarted.config.clone(), &runtime, Duration::from_secs(3));
    let (sender, worker) = restarted.spawn_guardian_recovery();
    sender.send(duplicate).await.expect("second episode");
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), receiver)
            .await
            .expect("bounded claim")
            .expect("ack"),
        RecoveryOutcome::NotAdmitted
    );
    drop(sender);
    worker.await.expect("worker joined");
    let receipts: u32 = database
        .query_row("SELECT count(*) FROM local_recovery_receipts", [], |row| {
            row.get(0)
        })
        .expect("no second preaction");
    assert_eq!(receipts, 0);
}

#[tokio::test]
async fn tier2_cancel_ack_follows_actual_owned_process_reap() {
    for case in ["cancel", "reload", "deadline"] {
        owned_cancellation_case(case).await;
    }
}

async fn owned_cancellation_case(case: &str) {
    let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    let marker = runtime.join("pid");
    let mut config = proxy.state.config.snapshot().expect("config");
    config.upstream.local_recovery.restart_command = vec![
        String::from("/usr/bin/python3"),
        String::from("-c"),
        String::from(
            "import signal,time,os,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
        ),
        marker.display().to_string(),
        String::from("secret-argv-marker"),
    ];
    proxy
        .state
        .config
        .apply_reloadable(&config)
        .expect("policy");
    let timeout = if case == "deadline" {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(5)
    };
    let (request, receiver) = make_request(proxy.state.config.clone(), &runtime, timeout);
    let authority = Arc::clone(&request.authority);
    let (sender, worker) = proxy.state.spawn_guardian_recovery();
    sender.send(request).await.expect("handoff");
    tokio::time::timeout(Duration::from_secs(3), async {
        while !marker.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("child started");
    let pid = fs::read_to_string(marker).expect("pid");
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).expect("owned child instance");
    eprintln!("tier2 owned {case} pid={pid} stat={stat}");
    match case {
        "cancel" => authority.cancel(),
        "reload" => {
            config.guardian.escalation_timeout_secs += 1;
            proxy
                .state
                .config
                .apply_reloadable(&config)
                .expect("invalidate generation");
        }
        _ => {}
    }
    let expected = if case == "deadline" {
        RecoveryOutcome::TimedOut
    } else {
        RecoveryOutcome::Cancelled
    };
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(4), receiver)
            .await
            .expect("bounded cleanup")
            .expect("ack"),
        expected
    );
    assert!(
        !std::path::Path::new(&format!("/proc/{pid}")).exists(),
        "done means direct child already reaped"
    );
    drop(sender);
    worker.await.expect("worker joined");
    let database = rusqlite::Connection::open(&proxy.sqlite_path).expect("database");
    let (json, outcome): (String, String) = database
        .query_row(
            "SELECT receipt_json, outcome FROM local_recovery_receipts",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("receipt");
    assert_eq!(
        outcome,
        if case == "deadline" {
            "episode_timeout"
        } else {
            "cancelled"
        }
    );
    assert!(!json.contains("secret-argv-marker"));
    assert!(
        !proxy
            .state
            .local_recovery
            .coordinator_for(&config.guardian.escalation_profile)
            .state
            .lock()
            .await
            .running
    );
}

#[tokio::test]
async fn tier2_stale_expired_busy_and_receipt_failure_do_not_act() {
    for case in ["stale", "expired", "busy", "receipt"] {
        let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
        let marker = runtime.join("must-not-exist");
        let mut config = proxy.state.config.snapshot().expect("config");
        config.upstream.local_recovery.restart_command =
            vec![String::from("/usr/bin/touch"), marker.display().to_string()];
        proxy
            .state
            .config
            .apply_reloadable(&config)
            .expect("policy");
        let timeout = if case == "expired" {
            Duration::ZERO
        } else {
            Duration::from_secs(3)
        };
        let (request, receiver) = make_request(proxy.state.config.clone(), &runtime, timeout);
        if case == "stale" {
            fs::write(runtime.join("target.v1"), b"version=2\n").expect("stale");
        }
        if case == "busy" {
            proxy
                .state
                .local_recovery
                .coordinator_for(&request.profile)
                .state
                .lock()
                .await
                .running = true;
        }
        if case == "receipt" {
            rusqlite::Connection::open(&proxy.sqlite_path)
                .expect("db")
                .execute("DROP TABLE local_recovery_receipts", [])
                .expect("fail receipt");
        }
        assert_eq!(
            run(&proxy, request, receiver).await,
            RecoveryOutcome::NotAdmitted,
            "{case}"
        );
        assert!(!marker.exists(), "{case} dispatched an action");
    }
}
