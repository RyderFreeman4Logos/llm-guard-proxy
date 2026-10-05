use super::super::{CgroupTarget, GuardianIteration, MemoryGuardian, RecoveryTarget};
use super::{target_tree, temporary_tree};
use crate::{
    EmergencyReserve,
    emergency::{AttemptOutcome, EmergencyController},
};
use nix::unistd::Uid;
use std::fs::{self, File};

#[test]
fn failed_write_to_an_empty_target_is_not_verified_action_success() {
    let (root, registration) = target_tree();
    let mut target =
        CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
            .expect("open populated fixture target");
    // Unprivileged descriptor boundary: the baseline attempts a real failing write.
    // The repaired path must detect no action is needed, not report kill success.
    // This is not proof of privileged kernel cgroup.kill behavior.
    target.kill = File::open(registration).expect("read-only descriptor");
    let group = root.join(target.registration.control_group.trim_start_matches('/'));
    fs::write(group.join("cgroup.events"), b"populated 0\n").expect("target exits independently");
    let mut controller = EmergencyController::new(
        EmergencyReserve::with_page_size(4096, 4096).expect("reserve"),
        1000,
    );
    assert_ne!(
        controller.attempt(0, &target),
        AttemptOutcome::Verified,
        "a failed write plus independently empty target must not claim a verified action"
    );
    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn write_errno_survives_empty_observation_and_resets_only_for_new_generation() {
    let (root, registration) = target_tree();
    let mut target =
        CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
            .expect("open fixture target");
    let group = root.join(target.registration.control_group.trim_start_matches('/'));
    target.kill = File::open(&registration).expect("read-only descriptor");
    let mut controller = EmergencyController::new(
        EmergencyReserve::with_page_size(4096, 4096).expect("reserve"),
        1000,
    );
    assert_eq!(
        controller.attempt(0, &target),
        AttemptOutcome::WriteFailed(libc::EBADF)
    );
    assert_eq!(controller.attempt(999, &target), AttemptOutcome::Waiting);
    assert_eq!(
        controller.attempt(1000, &target),
        AttemptOutcome::WriteFailed(libc::EBADF)
    );
    fs::write(group.join("cgroup.events"), b"populated 0\n").expect("independent empty");
    assert_eq!(
        controller.attempt(1001, &target),
        AttemptOutcome::WriteFailed(libc::EBADF)
    );
    assert!(!controller.target_is_verified());
    controller.reset_for_target_generation();
    assert_eq!(
        controller.attempt(1002, &target),
        AttemptOutcome::AlreadyEmpty
    );
    assert_eq!(fs::read(group.join("cgroup.kill")).expect("untouched"), b"");
    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn guardian_reports_the_direct_write_errno_without_claiming_shed() {
    let (root, _registration) = target_tree();
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("meminfo");
    let mut guardian = MemoryGuardian::open(
        super::guardian_handle("target.v1", &root),
        root.join("runtime"),
    )
    .expect("guardian");
    guardian.proc_meminfo = File::open(&meminfo).expect("open meminfo");
    guardian.reconcile_healthy_target();
    let RecoveryTarget::Cgroup(target) = guardian.target.as_mut().expect("target") else {
        panic!("expected cgroup");
    };
    target.kill = File::open(&meminfo).expect("read-only descriptor");
    assert_eq!(
        guardian.tick().expect("tick"),
        GuardianIteration::KillFailed(libc::EBADF)
    );
    assert!(guardian.is_latched());
    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn no_target_under_pressure_does_not_claim_action_success() {
    let root = temporary_tree();
    let runtime = root.join("runtime");
    fs::create_dir(&runtime).expect("runtime");
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("meminfo");
    let mut guardian = MemoryGuardian::open(super::guardian_handle("target.v1", &root), &runtime)
        .expect("guardian");
    guardian.proc_meminfo = File::open(meminfo).expect("open meminfo");
    assert_eq!(guardian.tick().expect("tick"), GuardianIteration::Unarmed);
    assert!(guardian.is_latched());
    assert!(guardian.target.is_none());
    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn successful_write_empty_target_and_memory_recovery_complete_mode_one() {
    let (root, _registration) = target_tree();
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("meminfo");
    let mut guardian = MemoryGuardian::open(
        super::guardian_handle("target.v1", &root),
        root.join("runtime"),
    )
    .expect("guardian");
    guardian.proc_meminfo = File::open(&meminfo).expect("open meminfo");
    assert_eq!(guardian.tick().expect("shed"), GuardianIteration::Shed);
    let RecoveryTarget::Cgroup(target) = guardian.target.as_ref().expect("target") else {
        panic!("expected cgroup");
    };
    let group = root.join(target.registration.control_group.trim_start_matches('/'));
    assert_eq!(fs::read(group.join("cgroup.kill")).expect("write"), b"1");
    fs::write(group.join("cgroup.events"), b"populated 0\n").expect("empty target");
    guardian.started -= std::time::Duration::from_secs(2);
    assert_eq!(
        guardian.tick().expect("verify"),
        GuardianIteration::Verified
    );
    assert!(guardian.is_latched(), "empty is not recovered memory");
    fs::write(&meminfo, b"MemAvailable: 2097152 kB\n").expect("recovered memory fixture");
    assert_eq!(
        guardian.tick().expect("recover"),
        GuardianIteration::Rearmed
    );
    assert!(!guardian.is_latched());
    fs::remove_dir_all(root).expect("remove fixture");
}
