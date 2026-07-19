use super::{guardian_config, guardian_handle, target_tree, temporary_tree};
use crate::{EmergencyReserve, emergency::AttemptOutcome, kill_direct};
use nix::unistd::Uid;
use std::{fs, os::unix::fs::PermissionsExt};

use super::super::{CgroupTarget, GuardianIteration, MemoryGuardian, RecoveryTarget};

#[test]
fn latched_tick_arms_registration_published_during_pressure() {
    let root = temporary_tree();
    let runtime = root.join("runtime");
    fs::create_dir(&runtime).expect("create runtime");
    let handle = guardian_handle("target.v1", &root);
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("write pressure meminfo");
    let mut guardian = MemoryGuardian::open(handle, &runtime).expect("open guardian");
    guardian.proc_meminfo = fs::File::open(&meminfo).expect("open pressure meminfo");

    assert_eq!(
        guardian.tick().expect("latch without a target"),
        GuardianIteration::Unarmed
    );
    assert!(guardian.is_latched());

    let uid = Uid::effective().as_raw();
    let id = "a".repeat(64);
    let scope = format!("docker-{id}.scope");
    let cgroup = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}"
    ));
    fs::create_dir_all(&cgroup).expect("create cgroup");
    fs::write(cgroup.join("cgroup.kill"), b"").expect("create kill");
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("create events");
    let registration = runtime.join("target.v1");
    fs::write(
        &registration,
        format!(
            "version=1\ncontainer_id={id}\nscope={scope}\ncontrol_group=/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}\n"
        ),
    )
    .expect("publish registration");
    fs::set_permissions(&registration, fs::Permissions::from_mode(0o600))
        .expect("secure registration");

    assert_eq!(
        guardian.tick().expect("latched target acquisition"),
        GuardianIteration::Waiting
    );
    assert_eq!(
        fs::read(cgroup.join("cgroup.kill")).expect("read cgroup kill"),
        b"1"
    );
    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn latched_tick_replaces_retired_cgroup_generation_without_applying_reload() {
    let (root, _registration) = target_tree();
    let runtime = root.join("runtime");
    let handle = guardian_handle("target.v1", &root);
    let meminfo = root.join("meminfo");
    fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("write pressure meminfo");
    let mut guardian = MemoryGuardian::open(handle.clone(), &runtime).expect("open guardian");
    guardian.proc_meminfo = fs::File::open(&meminfo).expect("open pressure meminfo");

    assert_eq!(
        guardian.tick().expect("latch original target"),
        GuardianIteration::Shed
    );
    assert!(guardian.is_latched());

    let mut requested = guardian_config("target.v1", &root);
    requested.guardian.target_label = String::from("hot-reloaded");
    handle
        .apply_reloadable(&requested)
        .expect("publish hot-reloaded policy");

    let uid = Uid::effective().as_raw();
    let id = "a".repeat(64);
    let cgroup = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{id}.scope"
    ));
    fs::write(cgroup.join("cgroup.kill"), b"").expect("clear original kill fixture");
    let retired = cgroup.with_extension("retired");
    fs::rename(&cgroup, &retired).expect("retire original generation");
    fs::create_dir(&cgroup).expect("create replacement generation");
    fs::write(cgroup.join("cgroup.kill"), b"").expect("create replacement kill");
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("create replacement events");

    assert_eq!(
        guardian.tick().expect("latched generation replacement"),
        GuardianIteration::Waiting
    );
    assert_eq!(guardian.active_policy().target_label, "test");
    assert_eq!(
        fs::read(cgroup.join("cgroup.kill")).expect("read replacement kill"),
        b"1"
    );
    assert_eq!(
        fs::read(retired.join("cgroup.kill")).expect("read retired kill"),
        b""
    );
    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn opens_and_kills_a_recreated_cgroup_after_the_original_becomes_empty() {
    let (root, registration) = target_tree();
    let uid = Uid::effective().as_raw();
    let original_id = "a".repeat(64);
    let original_events = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{original_id}.scope/cgroup.events"
    ));
    fs::write(original_events, b"populated 0\n").expect("mark original empty");

    let replacement_id = "b".repeat(64);
    let replacement_scope = format!("docker-{replacement_id}.scope");
    let replacement = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{replacement_scope}"
    ));
    fs::create_dir_all(&replacement).expect("create replacement cgroup");
    fs::write(replacement.join("cgroup.kill"), b"").expect("create replacement kill");
    fs::write(replacement.join("cgroup.events"), b"populated 1\n")
        .expect("create replacement events");
    fs::write(
        &registration,
        format!(
            "version=1\ncontainer_id={replacement_id}\nscope={replacement_scope}\ncontrol_group=/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{replacement_scope}\n"
        ),
    )
    .expect("publish replacement registration");

    let target =
        CgroupTarget::from_registration(&registration, &root, uid).expect("open replacement");
    let mut reserve = EmergencyReserve::with_page_size(4096, 4096).expect("reserve");
    kill_direct(&mut reserve, &target).expect("kill replacement");
    assert_eq!(
        fs::read(replacement.join("cgroup.kill")).expect("read replacement kill"),
        b"1"
    );
    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn startup_waits_until_initial_registration_is_available() {
    let root = temporary_tree();
    let runtime = root.join("runtime");
    fs::create_dir_all(&runtime).expect("create runtime");
    let handle = guardian_handle("target.v1", &root);

    let mut guardian = MemoryGuardian::open(handle, &runtime)
        .expect("missing registration must not abort startup");
    guardian.reconcile_healthy_target();
    assert!(guardian.target.is_none());

    let id = "a".repeat(64);
    let scope = format!("docker-{id}.scope");
    let cgroup = root.join(format!(
        "user.slice/user-{}.slice/user@{}.service/app.slice/{scope}",
        Uid::effective().as_raw(),
        Uid::effective().as_raw()
    ));
    fs::create_dir_all(&cgroup).expect("create cgroup");
    fs::write(cgroup.join("cgroup.kill"), b"").expect("create kill");
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("create events");
    let registration = format!(
        "version=1\ncontainer_id={id}\nscope={scope}\ncontrol_group=/user.slice/user-{}.slice/user@{}.service/app.slice/{scope}\n",
        Uid::effective().as_raw(),
        Uid::effective().as_raw()
    );
    let registration_path = runtime.join("target.v1");
    fs::write(&registration_path, registration).expect("publish registration");
    fs::set_permissions(&registration_path, fs::Permissions::from_mode(0o600))
        .expect("secure registration");

    guardian.reconcile_healthy_target();
    assert!(guardian.target.is_some());
    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn published_registration_replacement_rearms_the_new_cgroup_generation() {
    let (root, registration) = target_tree();
    let runtime = root.join("runtime");
    let handle = guardian_handle("target.v1", &root);
    let mut guardian = MemoryGuardian::open(handle, &runtime).expect("open guardian");
    guardian.reconcile_healthy_target();

    let uid = Uid::effective().as_raw();
    let replacement_id = "b".repeat(64);
    let replacement_scope = format!("docker-{replacement_id}.scope");
    let replacement = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{replacement_scope}"
    ));
    fs::create_dir_all(&replacement).expect("create replacement cgroup");
    fs::write(replacement.join("cgroup.kill"), b"").expect("create replacement kill");
    fs::write(replacement.join("cgroup.events"), b"populated 1\n")
        .expect("create replacement events");
    fs::write(
        &registration,
        format!(
            "version=1\ncontainer_id={replacement_id}\nscope={replacement_scope}\ncontrol_group=/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{replacement_scope}\n"
        ),
    )
    .expect("publish replacement registration");
    fs::set_permissions(&registration, fs::Permissions::from_mode(0o600))
        .expect("secure replacement registration");

    assert!(guardian.reconcile_healthy_target());
    let RecoveryTarget::Cgroup(target) = guardian.target.as_ref().expect("replacement target")
    else {
        panic!("expected cgroup target");
    };
    let mut reserve = EmergencyReserve::with_page_size(4096, 4096).expect("reserve");
    kill_direct(&mut reserve, target).expect("kill replacement");

    assert_eq!(
        fs::read(replacement.join("cgroup.kill")).expect("read replacement kill"),
        b"1"
    );
    let original = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{}.scope/cgroup.kill",
        "a".repeat(64)
    ));
    assert_eq!(fs::read(original).expect("read original kill"), b"");
    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn same_registration_recreation_reopens_and_rearms_new_cgroup_generation() {
    let (root, _registration) = target_tree();
    let runtime = root.join("runtime");
    let handle = guardian_handle("target.v1", &root);
    let mut guardian = MemoryGuardian::open(handle, &runtime).expect("open guardian");
    assert!(guardian.reconcile_healthy_target());

    let uid = Uid::effective().as_raw();
    let id = "a".repeat(64);
    let cgroup = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{id}.scope"
    ));
    fs::write(cgroup.join("cgroup.events"), b"populated 0\n").expect("mark original empty");
    assert_eq!(
        guardian.attempt_emergency(false),
        GuardianIteration::Verified
    );
    fs::write(cgroup.join("cgroup.kill"), b"").expect("clear original kill fixture");

    let retired = cgroup.with_extension("retired");
    fs::rename(&cgroup, &retired).expect("retire original cgroup generation");
    fs::create_dir(&cgroup).expect("recreate cgroup at the registered path");
    fs::write(cgroup.join("cgroup.kill"), b"").expect("create replacement kill");
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("create replacement events");

    assert!(
        guardian.reconcile_healthy_target(),
        "same registration must not hide a new cgroup object"
    );
    assert_eq!(
        guardian.attempt_emergency(false),
        GuardianIteration::Waiting
    );
    assert_eq!(
        fs::read(cgroup.join("cgroup.kill")).expect("read replacement kill"),
        b"1"
    );
    assert_eq!(
        fs::read(retired.join("cgroup.kill")).expect("read retired kill"),
        b""
    );

    fs::write(cgroup.join("cgroup.events"), b"populated 0\n").expect("mark replacement empty");
    let MemoryGuardian {
        target, controller, ..
    } = &mut guardian;
    let RecoveryTarget::Cgroup(target) = target.as_ref().expect("replacement target") else {
        panic!("expected cgroup target");
    };
    assert_eq!(
        controller
            .as_mut()
            .expect("emergency controller")
            .attempt(u64::MAX, target),
        AttemptOutcome::Verified
    );

    fs::remove_dir_all(root).expect("remove root");
}

#[test]
fn same_cgroup_inode_repopulation_rearms_verified_controller() {
    let (root, _registration) = target_tree();
    let runtime = root.join("runtime");
    let handle = guardian_handle("target.v1", &root);
    let mut guardian = MemoryGuardian::open(handle, &runtime).expect("open guardian");
    assert!(guardian.reconcile_healthy_target());

    let uid = Uid::effective().as_raw();
    let id = "a".repeat(64);
    let cgroup = root.join(format!(
        "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/docker-{id}.scope"
    ));
    fs::write(cgroup.join("cgroup.events"), b"populated 0\n").expect("mark target empty");
    assert_eq!(
        guardian.attempt_emergency(false),
        GuardianIteration::Verified
    );
    fs::write(cgroup.join("cgroup.events"), b"populated 1\n")
        .expect("repopulate the same cgroup object");
    assert!(
        guardian.reconcile_healthy_target(),
        "a verified target becoming populated on the same inode must re-arm the controller"
    );
    assert_eq!(
        guardian.attempt_emergency(false),
        GuardianIteration::Waiting
    );
    assert_eq!(
        fs::read(cgroup.join("cgroup.kill")).expect("read repopulated target kill"),
        b"11"
    );

    fs::remove_dir_all(root).expect("remove root");
}
