#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![expect(
    unsafe_code,
    reason = "the test instruments the process allocator to prove the Tier-1 contract"
)]

use llm_guard_proxy_host_guardian::{CgroupTarget, EmergencyReserve, kill_direct};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every allocator operation delegates to the process-wide System
// allocator and only records the allocation entry point atomically.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: GlobalAlloc callers provide a valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originated from the delegated allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: GlobalAlloc callers provide a valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: pointer and layout originated from the delegated allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn temporary_tree() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "llm-guard-proxy-guardian-no-alloc-{}-{nonce}",
        std::process::id()
    ))
}

fn create_target(root: &Path, registration: &Path) {
    let uid = nix::unistd::Uid::effective().as_raw();
    let id = "a".repeat(64);
    let scope = format!("docker-{id}.scope");
    let control_group =
        format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}");
    let directory = root.join(control_group.trim_start_matches('/'));
    fs::create_dir_all(&directory).expect("create cgroup");
    fs::write(directory.join("cgroup.kill"), b"").expect("create kill");
    fs::write(directory.join("cgroup.events"), b"populated 1\n").expect("create events");
    fs::write(
        registration,
        format!("version=1\ncontainer_id={id}\nscope={scope}\ncontrol_group={control_group}\n"),
    )
    .expect("write registration");
    fs::set_permissions(registration, fs::Permissions::from_mode(0o600))
        .expect("secure registration");
}

#[test]
fn reserve_release_and_direct_write_allocate_nothing() {
    let root = temporary_tree();
    fs::create_dir_all(&root).expect("create root");
    let registration = root.join("target-cgroup.v1");
    create_target(&root, &registration);
    let target = CgroupTarget::open_registered(&registration, &root).expect("open target");
    let mut reserve = EmergencyReserve::with_page_size(16 * 1024, 4096).expect("reserve");

    ALLOCATIONS.store(0, Ordering::SeqCst);
    let result = kill_direct(&mut reserve, &target);
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);

    assert!(result.is_ok());
    assert_eq!(allocations, 0, "direct emergency function allocated");
    assert!(!reserve.is_allocated());
    fs::remove_dir_all(root).expect("remove fixture");
}
