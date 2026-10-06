use super::{
    RecoveryProcessGuard, configure_recovery_command, read_recovery_stat,
    recovery_stat_is_active_member,
};
use std::{process::Stdio, time::Duration};
use tokio::{io::AsyncReadExt, process::Command, time::timeout};

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
