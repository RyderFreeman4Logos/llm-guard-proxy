use super::{
    RecoveryProcessGroupIdentity, RecoveryProcessGuard, configure_recovery_command,
    read_recovery_stat, recovery_child_is_unreaped, recovery_stat_is_active_member,
    send_recovery_process_group_signal, terminate_timed_out_recovery_child,
};
use nix::{
    errno::Errno,
    sys::{
        prctl,
        signal::{Signal, kill},
        wait::{WaitPidFlag, WaitStatus, waitpid},
    },
    unistd::Pid,
};
use std::{
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    process::Command,
    time::timeout,
};

const UNVERIFIED_GROUP_WORKER: &str = "GUARD_282_UNVERIFIED_GROUP_WORKER";

#[tokio::test]
async fn owned_recovery_proc_open_then_exit_is_a_vanished_entry() {
    let mut child = tokio::process::Command::new("/usr/bin/cat")
        .stdin(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("owned fixture");
    let pid = child.id().expect("fixture pid");
    let opened = std::fs::File::open(format!("/proc/{pid}/stat"));
    tokio::time::timeout(std::time::Duration::from_secs(2), child.kill())
        .await
        .expect("bounded fixture reap")
        .expect("reaped fixture");
    assert!(
        read_recovery_stat(opened)
            .expect("vanished stat is not unavailable ownership")
            .is_none()
    );
    assert!(
        read_recovery_stat(Err(std::io::Error::from_raw_os_error(libc::ENOENT)))
            .expect("vanished path")
            .is_none()
    );
    assert!(
        read_recovery_stat(Err(std::io::Error::from_raw_os_error(libc::EACCES))).is_err(),
        "unreadable evidence stays fail closed"
    );
}

#[test]
fn owned_recovery_proc_control_fields_stay_strict() {
    assert!(
        recovery_stat_is_active_member(b"1 (nonutf-\xff ) name) S 1 42", 42).expect("byte name")
    );
    assert!(!recovery_stat_is_active_member(b"1 (\xff) Z 1 42", 42).expect("quiescent"));
    assert!(!recovery_stat_is_active_member(b"1 (\xff) S 1 7", 42).expect("other group"));
    for stat in [
        b"1 (x) S 1".as_slice(),
        b"1 (x) S 1 +42",
        b"1 (x) S 1 99999999999999",
        b"1 (x) S 1 \xff",
        b"1 (x) \xff 1 42",
        b"broken",
    ] {
        assert!(
            recovery_stat_is_active_member(stat, 42).is_err(),
            "invalid control data must not appear quiescent"
        );
    }
}

#[tokio::test]
async fn owned_recovery_ignores_unrelated_non_utf8_process_name() {
    let mut noise = Command::new("/usr/bin/python3")
        .args(["-c", "import ctypes,time; assert ctypes.CDLL(None).prctl(15,b'noise-\\xff',0,0,0)==0; print('ready',flush=True); time.sleep(30)"])
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn().expect("isolated unrelated process");
    let mut ready = [0_u8; 6];
    let started = timeout(
        Duration::from_secs(2),
        noise.stdout.as_mut().expect("pipe").read_exact(&mut ready),
    )
    .await;
    let mut command = Command::new("/usr/bin/true");
    command.kill_on_drop(true);
    configure_recovery_command(&mut command);
    let mut owned = RecoveryProcessGuard::new(command.spawn().expect("owned action"));
    let result = timeout(Duration::from_secs(2), owned.wait()).await;
    // Always reap both fixtures before the behavioral assertion, including RED.
    if let Some(child) = owned.child.as_mut() {
        timeout(Duration::from_secs(2), child.kill())
            .await
            .expect("bounded direct cleanup")
            .expect("reap direct action");
    }
    owned.disarm_after_reap();
    timeout(Duration::from_secs(2), noise.kill())
        .await
        .expect("bounded noise cleanup")
        .expect("reap unrelated fixture");
    assert!(
        started.is_ok_and(|ready| ready.is_ok()),
        "noise process was ready"
    );
    assert_eq!(&ready, b"ready\n");
    assert!(
        matches!(&result, Ok(Ok(status)) if status.success()),
        "unrelated process name blocked owned completion: {result:?}"
    );
}

#[tokio::test]
async fn unverified_recovery_leader_cannot_authorize_cached_group_signal() {
    if std::env::var_os(UNVERIFIED_GROUP_WORKER).is_none() {
        run_unverified_group_worker().await;
        return;
    }

    prctl::set_child_subreaper(true).expect("isolated fixture subreaper");
    let mut fixture_cleanup = OwnedRecoveryCleanup {
        leader: None,
        descendant: None,
    };
    let (mut guard, owner_count, descendant) =
        spawn_owned_recovery_group(&mut fixture_cleanup).await;
    assert_stale_recovery_identity_is_refused(&guard, descendant);
    let (term_sent, kill_sent, cleanup_status, descendant_state, descendant_running) =
        reap_leader_and_check_timeout_cleanup(&mut guard, &owner_count, descendant).await;

    // Retire the guard first; the exact-PID cleanup then handles only fixture children.
    drop(guard);
    drop(fixture_cleanup);
    assert!(
        !term_sent
            && !kill_sent
            && cleanup_status == "group_identity_unconfirmed"
            && descendant_running,
        "timeout cleanup authorized a cached PGID without its original leader: term={term_sent}, kill={kill_sent}, status={cleanup_status}, descendant_state={descendant_state:?}"
    );
}

async fn run_unverified_group_worker() {
    let mut worker = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "proxy::recovery::proc_tests::unverified_recovery_leader_cannot_authorize_cached_group_signal",
            "--nocapture",
        ])
        .env(UNVERIFIED_GROUP_WORKER, "1")
        .kill_on_drop(true)
        .spawn()
        .expect("isolated subprocess fixture");
    let result = timeout(Duration::from_secs(15), worker.wait()).await;
    if result.is_err() {
        let _cleanup = timeout(Duration::from_secs(2), worker.kill()).await;
    }
    assert!(
        matches!(result, Ok(Ok(status)) if status.success()),
        "isolated fixture must complete cleanly: {result:?}"
    );
}

async fn spawn_owned_recovery_group(
    cleanup: &mut OwnedRecoveryCleanup,
) -> (RecoveryProcessGuard, Arc<AtomicUsize>, u32) {
    let mut command = Command::new("python3");
    command
        .args([
            "-c",
            "import os,time; child=os.fork(); print(child,flush=True) if child else None; time.sleep(30)",
        ])
        .stdout(Stdio::piped())
        .kill_on_drop(true);
    configure_recovery_command(&mut command);
    let mut child = command.spawn().expect("owned process-group fixture");
    let leader_pid = child.id().expect("fixture leader PID");
    cleanup.leader = Some(Pid::from_raw(
        i32::try_from(leader_pid).expect("leader PID fits pid_t"),
    ));
    let mut stdout = BufReader::new(child.stdout.take().expect("fixture stdout"));
    let owner_count = Arc::new(AtomicUsize::new(0));
    let guard = RecoveryProcessGuard::new_owned(child, Arc::clone(&owner_count));
    let leader = guard.process_group_id().expect("fixture leader PID");
    assert_eq!(leader, leader_pid, "captured leader PID must stay stable");
    let mut descendant_line = String::new();
    timeout(
        Duration::from_secs(2),
        stdout.read_line(&mut descendant_line),
    )
    .await
    .expect("descendant readiness")
    .expect("read descendant PID");
    drop(stdout);
    let descendant = descendant_line
        .trim()
        .parse::<u32>()
        .expect("descendant PID");
    assert!(
        recovery_stat_is_active_member(
            &std::fs::read(format!("/proc/{descendant}/stat")).expect("descendant stat"),
            leader,
        )
        .expect("fixture group membership"),
        "the fixture descendant must be in the isolated leader group"
    );
    cleanup.descendant = Some(Pid::from_raw(
        i32::try_from(descendant).expect("fixture PID fits pid_t"),
    ));
    (guard, owner_count, descendant)
}

fn assert_stale_recovery_identity_is_refused(guard: &RecoveryProcessGuard, descendant: u32) {
    let identity = guard
        .process_group_identity()
        .expect("owned fixture identity");
    let stale_identity = RecoveryProcessGroupIdentity {
        start_time_ticks: identity.start_time_ticks ^ 1,
        ..identity
    };
    let stale_signal_sent = send_recovery_process_group_signal(stale_identity, Signal::SIGTERM);
    let descendant_still_running =
        recovery_proc_state(descendant).is_some_and(|state| !matches!(state, b'Z' | b'X'));
    assert!(
        !stale_signal_sent && descendant_still_running,
        "a mismatched leader start time authorized the cached PGID: sent={stale_signal_sent}"
    );
}

async fn reap_leader_and_check_timeout_cleanup(
    guard: &mut RecoveryProcessGuard,
    owner_count: &AtomicUsize,
    descendant: u32,
) -> (bool, bool, String, Option<u8>, bool) {
    guard
        .child
        .as_mut()
        .expect("owned child")
        .start_kill()
        .expect("kill only the owned leader");
    let leader_status = timeout(
        Duration::from_secs(2),
        guard.child.as_mut().expect("owned child").wait(),
    )
    .await
    .expect("bounded leader reap")
    .expect("reap owned leader outside the group guard");
    assert!(
        !leader_status.success(),
        "fixture leader was killed directly"
    );

    assert!(
        guard.wait().await.is_err(),
        "reaped leader identity is unconfirmed"
    );
    assert_eq!(
        owner_count.load(Ordering::Acquire),
        1,
        "unconfirmed cleanup retains physical ownership"
    );
    let cleanup = terminate_timed_out_recovery_child(guard).await;
    let term_sent = cleanup["upstream_stall_recovery_timeout_term_sent"] == "true";
    let kill_sent = cleanup["upstream_stall_recovery_timeout_kill_sent"] == "true";
    let cleanup_status = cleanup["upstream_stall_recovery_timeout_cleanup_status"].as_str();
    let descendant_state = recovery_proc_state(descendant);
    let descendant_still_running_after_reap =
        descendant_state.is_some_and(|state| !matches!(state, b'Z' | b'X'));
    (
        term_sent,
        kill_sent,
        cleanup_status.to_owned(),
        descendant_state,
        descendant_still_running_after_reap,
    )
}

struct OwnedRecoveryCleanup {
    leader: Option<Pid>,
    descendant: Option<Pid>,
}

impl Drop for OwnedRecoveryCleanup {
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        if let Some(leader) = self.leader
            && u32::try_from(leader.as_raw()).is_ok_and(recovery_child_is_unreaped)
        {
            let _signal = kill(leader, Signal::SIGKILL);
            reap_fixture_child(leader, deadline);
        }
        if let Some(descendant) = self.descendant {
            let _signal = kill(descendant, Signal::SIGKILL);
            reap_fixture_child(descendant, deadline);
        }
    }
}

fn reap_fixture_child(pid: Pid, deadline: Instant) {
    loop {
        match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
            Err(Errno::ECHILD) | Ok(WaitStatus::StillAlive) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(5));
            }
            _ => return,
        }
    }
}

fn recovery_proc_state(pid: u32) -> Option<u8> {
    let stat = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
    let state_offset = stat.windows(2).rposition(|pair| pair == b") ")? + 2;
    stat[state_offset..]
        .split(u8::is_ascii_whitespace)
        .find(|field| !field.is_empty())
        .and_then(|field| field.first().copied())
}
