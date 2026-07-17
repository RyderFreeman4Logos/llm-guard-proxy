use super::{guardian_handle, target_tree, temporary_tree};
use crate::{EmergencyReserve, kill_direct};
use nix::unistd::Uid;
use std::{fs, os::unix::fs::PermissionsExt};

use super::super::{CgroupTarget, MemoryGuardian, RecoveryTarget};

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
