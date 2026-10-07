//! Opt-in joint acceptance: synthetic memory input, real kernel/action/SQLite paths.
#[path = "../../../../llm-guard-proxy-host-guardian/tests/support/owned_cgroup.rs"]
mod owned_cgroup;
use super::{Arc, Duration, FakeUpstream, ProxyFixture, RecoveryOutcome, fs, settle_owned_fixture};
use llm_guard_proxy_host_guardian::{GuardianIteration, MemoryGuardian};
use owned_cgroup::{
    CgroupObservation, OwnedCgroup, cgroup_members, populated, process_command, process_identity,
    random_id, wait_for_exit, wait_until_populated,
};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        process::ExitStatusExt,
    },
    path::Path,
};
use tokio::sync::mpsc;

#[tokio::test]
#[ignore = "requires writable user delegated app.slice; run just test-real-tier2"]
async fn tier2_real_kernel_admission_action_durable_receipt() {
    let id = random_id().expect("random owned identity");
    let uid = nix::unistd::Uid::effective().as_raw();
    let group =
        format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{id}.scope");
    let path = Path::new("/sys/fs/cgroup").join(group.trim_start_matches('/'));
    let (mut owned, observation, identity) = owned_kernel_target(&path, uid, &group);
    let target_metadata = observation.directory.metadata().expect("target identity");
    assert_eq!(target_metadata.uid(), uid);

    let upstream = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn(&upstream.base_url, false).await;
    let (mut guardian, marker, meminfo) = open_kernel_guardian(&proxy, &id, &group);
    let (sender, mut requests) = mpsc::channel(1);
    guardian.set_recovery_sender(sender);
    assert_eq!(
        guardian.tick().expect("open populated registration"),
        GuardianIteration::Healthy
    );
    assert!(!guardian.is_latched());
    assert_eq!(
        observation.members().expect("armed populated members"),
        [identity.pid]
    );
    // No real host pressure: both Tier 1 and the retained Tier-2 authority read this file.
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("synthetic low input");
    assert_eq!(
        guardian.tick().expect("guardian Tier-1 action"),
        GuardianIteration::Shed
    );
    observation
        .wait_until_empty()
        .expect("real populated=0 and empty procs");
    let status = wait_for_exit(owned.child.as_mut().expect("child"), Duration::from_secs(5))
        .expect("actual reap");
    assert_eq!(status.signal(), Some(9));
    assert_eq!(
        process_identity(identity.pid)
            .expect_err("exact child reaped")
            .kind(),
        std::io::ErrorKind::NotFound
    );
    let request = admit_kernel_request(&mut guardian, &mut requests, &marker).await;
    assert_eq!(
        request
            .authority
            .target_identity()
            .expect("retained generation"),
        (target_metadata.dev(), target_metadata.ino(), id.as_str())
    );
    assert_eq!(
        request.authority.config_revision(),
        proxy.state.config.revision()
    );
    assert!(request.authority.dispatch(|| ()).is_some());
    let proxy = run_kernel_worker(proxy, upstream, guardian, requests, request, &marker).await;
    assert_kernel_receipt(&proxy, &target_metadata, &id, &marker);
    owned.remove().expect("remove only parent-owned leaf");
    assert_eq!(
        observation
            .is_populated()
            .expect_err("removed kernfs is not empty")
            .raw_os_error(),
        Some(nix::errno::Errno::ENODEV as i32)
    );
    assert!(!path.exists());
    let fixture_root = proxy.root.clone();
    drop(proxy);
    assert!(
        !fixture_root.exists(),
        "private SQLite/runtime fixture removed after checked drain"
    );
    println!(
        "SYNTHETIC pressure; production guardian populated registration -> own real Tier-1 kill + child reap -> one same-generation Tier-2 -> worker action reap + loopback readiness -> terminal SQLite ACK/boot claim; owned leaf removed"
    );
}

fn owned_kernel_target(
    path: &Path,
    uid: u32,
    group: &str,
) -> (
    OwnedCgroup,
    CgroupObservation,
    owned_cgroup::ProcessIdentity,
) {
    let parent = path.parent().expect("app.slice");
    let metadata = fs::symlink_metadata(parent).expect("delegated app.slice");
    assert!(metadata.is_dir() && metadata.uid() == uid && metadata.mode() & 0o022 == 0);
    for ancestor in parent.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).expect("cgroup authority ancestry");
        assert!(metadata.is_dir() && (metadata.uid() == 0 || metadata.uid() == uid));
        assert_eq!(
            metadata.mode() & 0o022,
            0,
            "cgroup authority is not foreign-writable"
        );
    }
    assert!(fs::read_to_string("/proc/self/cgroup").expect("runner delegation").lines()
        .any(|line| line.starts_with(&format!("0::/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/llm-guard-real-tier2-test-"))));
    // Exclusive mkdir refuses existing names. Only this parent collects the leaf.
    let mut owned = OwnedCgroup::create(path.to_path_buf()).expect("owned kernel cgroup");
    let child = owned.child.as_mut().expect("owned child");
    let identity = wait_until_populated(path, child).expect("genuinely populated");
    assert_eq!(identity.pid, child.id());
    assert_eq!(cgroup_members(path).expect("members"), [identity.pid]);
    assert!(populated(path).expect("populated"));
    assert_eq!(
        process_command(identity.pid).expect("owned argv"),
        ["/usr/bin/sleep", "120"]
    );
    assert!(
        fs::read_to_string(format!("/proc/{}/cgroup", identity.pid))
            .expect("membership")
            .lines()
            .any(|line| line == format!("0::{group}"))
    );
    let observation = CgroupObservation::open(path).expect("pinned kernel readback");
    (owned, observation, identity)
}

fn open_kernel_guardian(
    proxy: &ProxyFixture,
    id: &str,
    group: &str,
) -> (MemoryGuardian, std::path::PathBuf, std::path::PathBuf) {
    let runtime = proxy.root.join("kernel-runtime");
    fs::create_dir(&runtime).expect("private runtime");
    fs::set_permissions(&runtime, fs::Permissions::from_mode(0o700)).expect("private runtime mode");
    let registration = runtime.join("target.v1");
    let mut record = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&registration)
        .expect("private registration");
    write!(
        record,
        "version=1\ncontainer_id={id}\nscope=docker-{id}.scope\ncontrol_group={group}\n"
    )
    .expect("registration");
    drop(record);
    let meminfo = runtime.join("synthetic-meminfo");
    fs::write(&meminfo, b"MemAvailable: 2097152 kB\n").expect("synthetic healthy input");
    let marker = runtime.join("action.pid");
    let mut config = proxy.state.config.snapshot().expect("config");
    config.guardian.enabled = true;
    config.guardian.target_label = String::from("isolated-kernel-test");
    config.guardian.registration_file = Some(String::from("target.v1"));
    config.guardian.cgroup_root = Path::new("/sys/fs/cgroup").to_path_buf();
    config.guardian.mem_threshold_gib = 1;
    config.guardian.reserve_mib = 1;
    config.guardian.escalation_enabled = true;
    config.guardian.escalation_profile = config.default_upstream_profile().name;
    config.guardian.escalation_mem_threshold_gib = 1;
    config.guardian.escalation_grace_secs = 30;
    config.guardian.escalation_timeout_secs = 10;
    config.upstream.local_recovery.enabled = true;
    config.upstream.local_recovery.restart_command = vec![
        String::from("/usr/bin/python3"),
        String::from("-c"),
        String::from(
            "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(0.1)",
        ),
        marker.display().to_string(),
        String::from("secret-argv-marker"),
    ];
    config.upstream.local_recovery.cooldown_ms = 100;
    config.upstream.local_recovery.restart_timeout_ms = 3000;
    config.upstream.local_recovery.readiness_deadline_ms = 1000;
    config.upstream.local_recovery.readiness_interval_ms = 20;
    config.validate().expect("unrelaxed policy validation");
    proxy
        .state
        .config
        .apply_reloadable(&config)
        .expect("private fixture policy");
    let guardian = MemoryGuardian::open_with_meminfo_for_test(
        proxy.state.config.clone(),
        &runtime,
        File::open(&meminfo).expect("synthetic input fd"),
    )
    .expect("production guardian");
    (guardian, marker, meminfo)
}

async fn admit_kernel_request(
    guardian: &mut MemoryGuardian,
    requests: &mut mpsc::Receiver<super::RecoveryRequest>,
    marker: &Path,
) -> super::RecoveryRequest {
    let grace_started = std::time::Instant::now();
    assert_eq!(
        guardian.tick().expect("guardian verifies own write"),
        GuardianIteration::Verified
    );
    assert!(requests.try_recv().is_err(), "no request before grace");
    assert!(!marker.exists(), "no worker action before admission");
    let request = tokio::time::timeout(Duration::from_secs(35), async {
        loop {
            guardian.tick().expect("same-generation pressure tick");
            if let Ok(request) = requests.try_recv() {
                break request;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("bounded real grace");
    assert!(grace_started.elapsed() >= Duration::from_secs(30));
    assert_eq!(request.episode, 1);
    assert_eq!(request.available_bytes, 0);
    assert_eq!(request.threshold_bytes, 1024 * 1024 * 1024);
    assert_eq!(request.grace_secs, 30);
    request
}

fn assert_kernel_receipt(
    proxy: &ProxyFixture,
    target_metadata: &std::fs::Metadata,
    id: &str,
    marker: &Path,
) {
    let database =
        rusqlite::Connection::open(&proxy.sqlite_path).expect("independent SQLite read after ACK");
    let (receipt_id, json, outcome, boot, claim_id): (String, String, String, String, String) = database.query_row(
        "SELECT r.receipt_id, r.receipt_json, r.outcome, c.boot_id, c.receipt_id FROM local_recovery_receipts r JOIN guardian_boot_claim c ON c.singleton=1",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))).expect("terminal + boot claim");
    assert_eq!(outcome, "succeeded");
    assert_eq!(receipt_id, claim_id);
    assert_eq!(
        boot,
        fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .expect("boot id")
            .trim()
    );
    for (table, expected) in [
        ("local_recovery_receipts", 1),
        ("guardian_boot_claim", 1),
        ("requests", 0),
    ] {
        let count: u32 = database
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .expect("row count");
        assert_eq!(count, expected, "{table}");
    }
    for forbidden in [
        "secret-argv-marker",
        "/usr/bin/python3",
        "pathlib",
        marker.to_str().expect("marker path"),
    ] {
        assert!(!json.contains(forbidden), "payload must not leak");
    }
    let json: serde_json::Value = serde_json::from_str(&json).expect("typed receipt");
    assert_eq!(json["detector"], "memory_guardian");
    assert_eq!(json["cause"], "memory_pressure");
    assert!(json["request_id"].is_null());
    let native = &json["guardian"];
    assert_eq!(native["boot_id"], boot);
    assert_eq!(native["guardian_episode"], 1);
    assert_eq!(native["target_device"], target_metadata.dev());
    assert_eq!(native["target_inode"], target_metadata.ino());
    assert_eq!(native["container_id"], id);
    assert_eq!(native["config_revision"], proxy.state.config.revision());
    assert_eq!(native["available_bytes"], 0);
    assert_eq!(native["threshold_bytes"], 1024_u64 * 1024 * 1024);
    assert_eq!(native["grace_secs"], 30);
}

async fn run_kernel_worker(
    proxy: ProxyFixture,
    mut upstream: FakeUpstream,
    mut guardian: MemoryGuardian,
    mut requests: mpsc::Receiver<super::RecoveryRequest>,
    request: super::RecoveryRequest,
    marker: &Path,
) -> ProxyFixture {
    let terminal_after_reap = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = Arc::clone(&terminal_after_reap);
    let action_marker = marker.to_path_buf();
    proxy
        .state
        .store
        .set_recovery_receipt_test_hook(move |stage, _| {
            if stage == "terminal_sql_start" {
                observed.store(
                    fs::read_to_string(&action_marker)
                        .is_ok_and(|pid| !Path::new(&format!("/proc/{pid}")).exists()),
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        });
    let (worker_sender, worker) = proxy.state.spawn_guardian_recovery();
    worker_sender
        .send(request)
        .await
        .expect("forward guardian's original request, never manufacture one");
    let ack = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            guardian.tick().expect("completion tick");
            if let Some(outcome) = guardian.escalation_outcome() {
                break outcome;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    let readiness = tokio::time::timeout(Duration::from_secs(1), upstream.receiver.recv()).await;
    let reaped_at_ack =
        fs::read_to_string(marker).is_ok_and(|pid| !Path::new(&format!("/proc/{pid}")).exists());
    guardian.tick().expect("same episode remains fenced");
    let duplicate = requests.try_recv();
    drop(guardian);
    drop(worker_sender);
    let proxy = settle_owned_fixture(proxy, upstream, worker).await;
    assert_eq!(
        ack.expect("bounded terminal ACK"),
        RecoveryOutcome::Succeeded
    );
    assert!(
        readiness.is_ok_and(|request| request
            .is_some_and(|request| request.path_and_query == "/v1/chat/completions")),
        "real loopback readiness request"
    );
    assert!(reaped_at_ack && terminal_after_reap.load(std::sync::atomic::Ordering::SeqCst));
    assert!(
        duplicate.is_err(),
        "exactly one guardian emission for the episode"
    );
    proxy
}
