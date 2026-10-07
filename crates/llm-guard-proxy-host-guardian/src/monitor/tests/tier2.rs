use super::{File, fs, guardian_config, target_tree};
use crate::monitor::tier2::RecoveryOutcome;
use crate::{EscalationConfig, GuardianIteration, MemoryGuardian};
use llm_guard_proxy_core::ConfigHandle;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

fn fixture() -> (
    PathBuf,
    MemoryGuardian,
    mpsc::Receiver<crate::monitor::tier2::RecoveryRequest>,
) {
    let (root, _) = target_tree();
    let mut config = guardian_config("target.v1", &root);
    config.guardian.escalation_enabled = true;
    config.guardian.escalation_profile = String::from("default");
    config.upstream.local_recovery.enabled = true;
    config.upstream.local_recovery.restart_command = vec![String::from("/usr/bin/true")];
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("synthetic pressure");
    let mut guardian =
        MemoryGuardian::open(ConfigHandle::new(config), root.join("runtime")).expect("guardian");
    guardian.proc_meminfo = File::open(meminfo).expect("synthetic input");
    let (sender, receiver) = mpsc::channel(1);
    guardian.set_recovery_sender(sender);
    (root, guardian, receiver)
}

fn mark_empty(guardian: &MemoryGuardian) {
    let Some(crate::monitor::RecoveryTarget::Cgroup(target)) = &guardian.target else {
        panic!("armed");
    };
    let path = guardian
        .active_policy
        .cgroup_root
        .join(target.registration.control_group.trim_start_matches('/'));
    fs::write(path.join("cgroup.events"), b"populated 0\n").expect("empty");
}

fn elapse_grace(guardian: &mut MemoryGuardian) {
    guardian.escalation.episode.clear();
    guardian.escalation.episode.arm(
        &EscalationConfig {
            enabled: true,
            unit: None,
            grace_period: Duration::from_secs(60),
        },
        Instant::now()
            .checked_sub(Duration::from_secs(61))
            .expect("elapsed fixture grace"),
    );
}

#[test]
fn tier2_reload_requires_executor_and_remains_retryable() {
    let (root, _) = target_tree();
    let handle = ConfigHandle::new(guardian_config("target.v1", &root));
    let mut guardian =
        MemoryGuardian::open(handle.clone(), root.join("runtime")).expect("open guardian");
    guardian.reconcile_healthy_target();
    let original_policy = guardian.active_policy().clone();

    let mut requested = guardian_config("target.v1", &root);
    requested.guardian.escalation_enabled = true;
    requested.guardian.escalation_profile = String::from("default");
    requested.upstream.local_recovery.enabled = true;
    requested.upstream.local_recovery.restart_command = vec![String::from("/usr/bin/true")];
    let reload = handle
        .apply_reloadable(&requested)
        .expect("valid policy reload should return an outcome");
    let requested_snapshot = handle.guardian_snapshot();

    let applied_without_executor = guardian.reconcile_healthy_target();
    let policy_without_executor = guardian.active_policy().clone();
    let rejected_policy = guardian.last_rejected_policy.clone();

    let (sender, receiver) = mpsc::channel(1);
    guardian.set_recovery_sender(sender);
    let applied_with_executor = guardian.reconcile_healthy_target();
    let policy_with_executor = guardian.active_policy().clone();
    drop(receiver);

    fs::remove_dir_all(root).expect("cleanup fixture");
    assert!(reload.applied, "valid requested policy should be published");
    assert_eq!(
        requested_snapshot.expect("snapshot should succeed"),
        requested.guardian
    );
    assert!(
        !applied_without_executor,
        "standalone mode must reject Tier 2"
    );
    assert_eq!(policy_without_executor, original_policy);
    assert_eq!(rejected_policy, Some(requested.guardian.clone()));
    assert!(applied_with_executor, "combined mode should admit Tier 2");
    assert_eq!(policy_with_executor, requested.guardian);
}

#[test]
fn tier2_requires_same_generation_verified_write_and_continuous_pressure() {
    let (root, mut guardian, mut receiver) = fixture();
    assert_eq!(guardian.tick().expect("kill"), GuardianIteration::Shed);
    mark_empty(&guardian);
    assert_eq!(
        guardian.tick().expect("verify"),
        GuardianIteration::Verified
    );
    assert!(receiver.try_recv().is_err(), "grace must not dispatch");
    elapse_grace(&mut guardian);
    guardian.tick().expect("escalate");
    let request = receiver.try_recv().expect("real integrated handoff");
    assert!(request.authority.dispatch(|| ()).is_some());
    assert_eq!(request.episode, 1);
    assert_eq!(request.available_bytes, 0);
    request
        .completion
        .send(RecoveryOutcome::Succeeded)
        .expect("ack");
    guardian.tick().expect("terminal tick");
    assert_eq!(
        guardian.escalation_outcome(),
        Some(RecoveryOutcome::Succeeded)
    );
    assert!(receiver.try_recv().is_err(), "same episode does not loop");
    fs::write(root.join("meminfo"), b"MemAvailable: 2097152 kB\n").expect("recovery");
    assert_eq!(
        guardian.tick().expect("recover"),
        GuardianIteration::Rearmed
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn tier2_empty_without_action_and_pressure_recovery_never_dispatch() {
    let (root, mut guardian, mut receiver) = fixture();
    guardian.reconcile_healthy_target();
    mark_empty(&guardian);
    assert_eq!(
        guardian.tick().expect("no action"),
        GuardianIteration::AlreadyEmpty
    );
    elapse_grace(&mut guardian);
    guardian.tick().expect("no false escalation");
    assert!(receiver.try_recv().is_err());
    fs::write(root.join("meminfo"), b"MemAvailable: 2097152 kB\n").expect("recovery");
    assert_eq!(guardian.tick().expect("rearm"), GuardianIteration::Rearmed);
    assert!(receiver.try_recv().is_err());
    fs::remove_dir_all(root).expect("cleanup");
}

#[tokio::test]
async fn tier2_stale_registration_and_owner_shutdown_cancel_exact_invocation() {
    let (root, mut guardian, mut receiver) = fixture();
    guardian.tick().expect("kill");
    mark_empty(&guardian);
    guardian.tick().expect("verify");
    elapse_grace(&mut guardian);
    guardian.tick().expect("escalate");
    let request = receiver.try_recv().expect("request");
    fs::write(root.join("runtime/target.v1"), b"version=2\n").expect("replace registration");
    assert!(
        request
            .authority
            .dispatch(|| panic!("stale action"))
            .is_none()
    );
    tokio::time::timeout(Duration::from_secs(1), request.authority.cancelled())
        .await
        .expect("stale cancellation");
    let completion = tokio::spawn(async move {
        request
            .completion
            .send(RecoveryOutcome::Cancelled)
            .expect("ack");
    });
    guardian
        .run_until(async {})
        .await
        .expect("shutdown joins acknowledgment");
    completion.await.expect("executor joined");
    assert_eq!(
        guardian.escalation_outcome(),
        Some(RecoveryOutcome::Cancelled)
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn tier2_failed_direct_write_never_becomes_verified_escalation() {
    let (root, mut guardian, mut receiver) = fixture();
    guardian.reconcile_healthy_target();
    let Some(crate::monitor::RecoveryTarget::Cgroup(target)) = guardian.target.as_mut() else {
        panic!("armed target");
    };
    target.kill = File::open(root.join("meminfo")).expect("read-only kill descriptor");
    assert_eq!(
        guardian.tick().expect("failed write"),
        GuardianIteration::KillFailed(libc::EBADF)
    );
    mark_empty(&guardian);
    elapse_grace(&mut guardian);
    assert_eq!(
        guardian.tick().expect("independent empty"),
        GuardianIteration::KillFailed(libc::EBADF)
    );
    assert!(
        receiver.try_recv().is_err(),
        "failed action cannot admit Tier 2"
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn tier2_config_aba_invalidates_previously_admitted_owner() {
    let (root, mut guardian, mut receiver) = fixture();
    guardian.tick().expect("kill");
    mark_empty(&guardian);
    guardian.tick().expect("verify");
    elapse_grace(&mut guardian);
    guardian.tick().expect("escalate");
    let request = receiver.try_recv().expect("request");
    let original = guardian.config_handle.snapshot().expect("snapshot");
    let mut changed = original.clone();
    changed.guardian.escalation_timeout_secs += 1;
    guardian
        .config_handle
        .apply_reloadable(&changed)
        .expect("change");
    guardian
        .config_handle
        .apply_reloadable(&original)
        .expect("return");
    assert!(request.authority.dispatch(|| panic!("ABA owner")).is_none());
    request
        .completion
        .send(RecoveryOutcome::Cancelled)
        .expect("ack");
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn supplied_meminfo_drives_observer_without_host_pressure() {
    let root = super::temporary_tree();
    let meminfo = root.join("synthetic-meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("synthetic input");
    let mut config = guardian_config("missing.v1", &root);
    config.guardian.enabled = false;
    let mut guardian = MemoryGuardian::open_with_meminfo_for_test(
        ConfigHandle::new(config),
        &root,
        File::open(meminfo).expect("input descriptor"),
    )
    .expect("guardian");
    assert_eq!(
        guardian.tick().expect("tick"),
        GuardianIteration::ObserverOnly
    );
    assert!(!guardian.is_latched());
    fs::remove_dir_all(root).expect("cleanup");
}
