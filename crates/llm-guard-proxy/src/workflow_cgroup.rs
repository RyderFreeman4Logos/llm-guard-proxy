//! Optional delegated cgroup v2 containment for one workflow execution.
//!
//! The module only creates and removes direct children of the proxy's current cgroup. A missing
//! unified hierarchy, missing `cgroup.kill`, or insufficient delegation falls back to the existing
//! process-group cleanup. Malformed kernel metadata and other unexpected I/O failures remain fatal.

use std::{
    ffi::OsString,
    fs, io,
    os::unix::ffi::OsStringExt,
    path::{Component, Path, PathBuf},
    process,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const CGROUP_MOUNTINFO: &str = "/proc/self/mountinfo";
const CURRENT_CGROUP: &str = "/proc/self/cgroup";
const CLEANUP_POLL: Duration = Duration::from_millis(10);
const DROP_CLEANUP_GRACE: Duration = Duration::from_millis(500);
const MAX_CREATE_ATTEMPTS: u64 = 64;
const MAX_OWNED_CGROUPS: usize = 1_024;

static NEXT_CGROUP_ID: AtomicU64 = AtomicU64::new(0);
static FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnavailableReason {
    UnifiedHierarchyMissing,
    CurrentMembershipMissing,
    KillInterfaceMissing,
    DelegationMissing,
    PermissionDenied,
}

impl UnavailableReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::UnifiedHierarchyMissing => "unified_hierarchy_missing",
            Self::CurrentMembershipMissing => "current_membership_missing",
            Self::KillInterfaceMissing => "cgroup_kill_missing",
            Self::DelegationMissing => "delegation_missing",
            Self::PermissionDenied => "permission_denied",
        }
    }
}

#[derive(Debug)]
enum SetupFailure {
    Unavailable(UnavailableReason),
    Fatal(io::Error),
}

#[derive(Debug)]
struct CgroupMount {
    root: PathBuf,
    mount_point: PathBuf,
}

/// An owned cgroup directory created for exactly one workflow execution.
pub(crate) struct WorkflowCgroup {
    path: PathBuf,
}

impl WorkflowCgroup {
    /// Creates an execution cgroup when the current process has a usable delegated cgroup v2 tree.
    pub(crate) fn prepare() -> io::Result<Option<Arc<Self>>> {
        optional_containment(Self::prepare_inner()).map(|cgroup| cgroup.map(Arc::new))
    }

    fn prepare_inner() -> Result<Self, SetupFailure> {
        let mountinfo = read_kernel_text(CGROUP_MOUNTINFO)?;
        let membership = read_kernel_text(CURRENT_CGROUP)?;
        let mounts = parse_cgroup2_mounts(&mountinfo).map_err(SetupFailure::Fatal)?;
        if mounts.is_empty() {
            return Err(SetupFailure::Unavailable(
                UnavailableReason::UnifiedHierarchyMissing,
            ));
        }
        let Some(current_path) =
            parse_current_cgroup_path(&membership).map_err(SetupFailure::Fatal)?
        else {
            return Err(SetupFailure::Unavailable(
                UnavailableReason::CurrentMembershipMissing,
            ));
        };
        let Some(parent) = resolve_current_cgroup_dir(&mounts, &current_path) else {
            return Err(SetupFailure::Unavailable(
                UnavailableReason::CurrentMembershipMissing,
            ));
        };

        for _ in 0..MAX_CREATE_ATTEMPTS {
            let sequence = NEXT_CGROUP_ID.fetch_add(1, Ordering::Relaxed);
            let path = workflow_cgroup_path(&parent, process::id(), sequence);
            match fs::create_dir(&path) {
                Ok(()) => {
                    let cgroup = Self { path };
                    cgroup.require_control_file("cgroup.procs")?;
                    cgroup.require_control_file("cgroup.events")?;
                    cgroup.require_control_file("cgroup.kill")?;
                    return Ok(cgroup);
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(classify_setup_io("create workflow cgroup", &error)),
            }
        }

        Err(SetupFailure::Fatal(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "failed to allocate a unique workflow cgroup name",
        )))
    }

    fn require_control_file(&self, name: &'static str) -> Result<(), SetupFailure> {
        match fs::metadata(self.path.join(name)) {
            Ok(metadata) if metadata.is_file() => Ok(()),
            Ok(_) => Err(SetupFailure::Fatal(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("workflow cgroup control path {name} is not a file"),
            ))),
            Err(error) if error.kind() == io::ErrorKind::NotFound && name == "cgroup.kill" => Err(
                SetupFailure::Unavailable(UnavailableReason::KillInterfaceMissing),
            ),
            Err(error) => Err(classify_setup_io(
                "inspect workflow cgroup controls",
                &error,
            )),
        }
    }

    /// Attaches the child before any other fallible post-spawn setup.
    ///
    /// `false` means delegation disappeared and the caller must keep process-group cleanup only.
    pub(crate) fn attach_or_fallback(
        &self,
        pid: u32,
        cleanup_deadline: Instant,
    ) -> io::Result<bool> {
        match fs::write(self.path.join("cgroup.procs"), format!("{pid}\n")) {
            Ok(()) => Ok(true),
            Err(error) => match classify_setup_io("attach workflow child to cgroup", &error) {
                SetupFailure::Unavailable(reason) => {
                    log_fallback_once(reason);
                    let _cleanup = self.kill_and_remove(cleanup_deadline);
                    Ok(false)
                }
                SetupFailure::Fatal(error) => Err(error),
            },
        }
    }

    /// Atomically sends `SIGKILL` to every process in the owned cgroup subtree.
    pub(crate) fn kill(&self) -> io::Result<()> {
        match fs::write(self.path.join("cgroup.kill"), b"1\n") {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound && !self.path.exists() => Ok(()),
            Err(error) => Err(io_with_context("kill workflow cgroup", &error)),
        }
    }

    /// Kills the subtree, verifies recursive emptiness, and removes only the owned directory tree.
    pub(crate) fn kill_and_remove(&self, deadline: Instant) -> io::Result<()> {
        let kill_error = self.kill().err();
        let finish_error = self.verify_empty_and_remove(deadline).err();

        match (kill_error, finish_error) {
            (None, None) => Ok(()),
            (kill_error, finish_error) => Err(combine_cleanup_errors([kill_error, finish_error])),
        }
    }

    /// Verifies recursive emptiness and removes the owned subtree after termination signals fire.
    pub(crate) fn verify_empty_and_remove(&self, deadline: Instant) -> io::Result<()> {
        let empty_result = self.wait_until_empty(deadline);
        let remove_result = if empty_result.is_ok() {
            self.remove_owned_tree(deadline)
        } else {
            Ok(())
        };

        match (empty_result.err(), remove_result.err()) {
            (None, None) => Ok(()),
            (empty_error, remove_error) => Err(combine_cleanup_errors([empty_error, remove_error])),
        }
    }

    fn wait_until_empty(&self, deadline: Instant) -> io::Result<()> {
        loop {
            match fs::read_to_string(self.path.join("cgroup.events")) {
                Ok(events) if cgroup_is_empty(&events)? => return Ok(()),
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound && !self.path.exists() => {
                    return Ok(());
                }
                Err(error) => return Err(io_with_context("read workflow cgroup events", &error)),
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "workflow cgroup remained populated past cleanup deadline",
                ));
            }
            thread::sleep(CLEANUP_POLL.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    fn remove_owned_tree(&self, deadline: Instant) -> io::Result<()> {
        loop {
            let mut directories = descendant_cgroup_dirs(&self.path)?;
            directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
            let mut rescan = false;
            for directory in directories {
                match fs::remove_dir(&directory) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) if directory_not_empty(&error) => {
                        rescan = true;
                        break;
                    }
                    Err(error) => {
                        return Err(io_with_context(
                            "remove workflow descendant cgroup directory",
                            &error,
                        ));
                    }
                }
            }
            if rescan {
                if Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "workflow descendant cgroup removal exceeded cleanup deadline",
                    ));
                }
                thread::sleep(CLEANUP_POLL.min(deadline.saturating_duration_since(Instant::now())));
                continue;
            }
            match fs::remove_dir(&self.path) {
                Ok(()) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(error) if directory_not_empty(&error) && Instant::now() < deadline => {
                    thread::sleep(
                        CLEANUP_POLL.min(deadline.saturating_duration_since(Instant::now())),
                    );
                }
                Err(error) => {
                    return Err(io_with_context("remove workflow cgroup directory", &error));
                }
            }
        }
    }
}

impl Drop for WorkflowCgroup {
    fn drop(&mut self) {
        let _cleanup = self.kill_and_remove(Instant::now() + DROP_CLEANUP_GRACE);
    }
}

fn read_kernel_text(path: &'static str) -> Result<String, SetupFailure> {
    fs::read_to_string(path).map_err(|error| classify_setup_io("read cgroup metadata", &error))
}

fn optional_containment<T>(result: Result<T, SetupFailure>) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(SetupFailure::Unavailable(reason)) => {
            log_fallback_once(reason);
            Ok(None)
        }
        Err(SetupFailure::Fatal(error)) => Err(error),
    }
}

fn log_fallback_once(reason: UnavailableReason) {
    if FALLBACK_LOGGED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        eprintln!(
            "workflow cgroup v2 containment unavailable ({}); using process-group cleanup fallback",
            reason.as_str()
        );
    }
}

fn classify_setup_io(operation: &'static str, error: &io::Error) -> SetupFailure {
    let reason = match error.kind() {
        io::ErrorKind::PermissionDenied => Some(UnavailableReason::PermissionDenied),
        io::ErrorKind::ReadOnlyFilesystem | io::ErrorKind::Unsupported => {
            Some(UnavailableReason::DelegationMissing)
        }
        io::ErrorKind::NotFound => Some(UnavailableReason::UnifiedHierarchyMissing),
        _ if matches!(error.raw_os_error(), Some(1 | 13)) => {
            Some(UnavailableReason::PermissionDenied)
        }
        _ if matches!(error.raw_os_error(), Some(30 | 95)) => {
            Some(UnavailableReason::DelegationMissing)
        }
        _ => None,
    };
    reason.map_or_else(
        || SetupFailure::Fatal(io_with_context(operation, error)),
        SetupFailure::Unavailable,
    )
}

fn parse_current_cgroup_path(contents: &str) -> io::Result<Option<PathBuf>> {
    let mut current = None;
    for line in contents.lines() {
        let Some(path) = line.strip_prefix("0::") else {
            continue;
        };
        let path = PathBuf::from(path);
        validate_absolute_path(&path, "current cgroup path")?;
        if current.replace(path).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "multiple cgroup v2 memberships found for current process",
            ));
        }
    }
    Ok(current)
}

fn parse_cgroup2_mounts(contents: &str) -> io::Result<Vec<CgroupMount>> {
    let mut mounts = Vec::new();
    for line in contents.lines() {
        let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
        let Some(separator) = fields.iter().position(|field| *field == "-") else {
            continue;
        };
        if fields.get(separator + 1) != Some(&"cgroup2") {
            continue;
        }
        let (Some(root), Some(mount_point)) = (fields.get(3), fields.get(4)) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed cgroup v2 mountinfo entry",
            ));
        };
        let root = decode_mountinfo_path(root)?;
        let mount_point = decode_mountinfo_path(mount_point)?;
        validate_absolute_path(&root, "cgroup v2 mount root")?;
        validate_absolute_path(&mount_point, "cgroup v2 mount point")?;
        mounts.push(CgroupMount { root, mount_point });
    }
    Ok(mounts)
}

fn decode_mountinfo_path(encoded: &str) -> io::Result<PathBuf> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let Some(octal) = bytes.get(index + 1..index + 4) else {
            return Err(invalid_mount_escape());
        };
        if !octal.iter().all(u8::is_ascii_digit) || octal.iter().any(|digit| *digit > b'7') {
            return Err(invalid_mount_escape());
        }
        decoded.push((octal[0] - b'0') * 64 + (octal[1] - b'0') * 8 + octal[2] - b'0');
        index += 4;
    }
    Ok(PathBuf::from(OsString::from_vec(decoded)))
}

fn invalid_mount_escape() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "malformed octal escape in cgroup mountinfo path",
    )
}

fn validate_absolute_path(path: &Path, label: &'static str) -> io::Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} is not a normalized absolute path"),
        ));
    }
    Ok(())
}

fn resolve_current_cgroup_dir(mounts: &[CgroupMount], current: &Path) -> Option<PathBuf> {
    mounts
        .iter()
        .filter_map(|mount| {
            current.strip_prefix(&mount.root).ok().map(|relative| {
                (
                    mount.root.components().count(),
                    mount.mount_point.join(relative),
                )
            })
        })
        .max_by_key(|(root_depth, _)| *root_depth)
        .map(|(_, path)| path)
}

fn workflow_cgroup_path(parent: &Path, pid: u32, sequence: u64) -> PathBuf {
    parent.join(format!("_llm_guard_proxy_workflow_{pid}_{sequence}"))
}

fn cgroup_is_empty(events: &str) -> io::Result<bool> {
    let value = events
        .lines()
        .find_map(|line| line.strip_prefix("populated "))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "workflow cgroup events omitted populated state",
            )
        })?;
    match value {
        "0" => Ok(true),
        "1" => Ok(false),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "workflow cgroup events contained invalid populated state",
        )),
    }
}

fn descendant_cgroup_dirs(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut pending = vec![root.to_path_buf()];
    let mut descendants = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| io_with_context("read workflow cgroup directory", &error))?
        {
            let entry = entry
                .map_err(|error| io_with_context("read workflow cgroup directory entry", &error))?;
            let file_type = entry
                .file_type()
                .map_err(|error| io_with_context("inspect workflow cgroup entry", &error))?;
            if file_type.is_dir() {
                let path = entry.path();
                descendants.push(path.clone());
                pending.push(path);
                if descendants.len() > MAX_OWNED_CGROUPS {
                    return Err(io::Error::other(
                        "workflow cgroup subtree exceeded cleanup directory limit",
                    ));
                }
            }
        }
    }
    Ok(descendants)
}

fn directory_not_empty(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::DirectoryNotEmpty
        || matches!(error.raw_os_error(), Some(16 | 39))
}

fn io_with_context(operation: &'static str, error: &io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("failed to {operation}: {error}"))
}

fn combine_cleanup_errors<const N: usize>(errors: [Option<io::Error>; N]) -> io::Error {
    let detail = errors
        .into_iter()
        .flatten()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    io::Error::other(detail)
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        process::{Command, Stdio},
        time::Duration,
    };

    use super::*;

    #[test]
    fn parses_mount_and_builds_child_below_current_cgroup() {
        let mounts = parse_cgroup2_mounts(
            "20 1 0:19 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n\
             21 20 0:19 /user.slice /run/cgroup\\040delegated rw - cgroup2 cgroup rw\n",
        )
        .expect("static mountinfo should parse");
        let current = parse_current_cgroup_path("0::/user.slice/app.service\n")
            .expect("static membership should parse")
            .expect("v2 membership should exist");
        let parent = resolve_current_cgroup_dir(&mounts, &current)
            .expect("the most specific mount should resolve");
        assert_eq!(parent, Path::new("/run/cgroup delegated/app.service"));

        let child = workflow_cgroup_path(&parent, 42, 7);
        assert_eq!(child.parent(), Some(parent.as_path()));
        assert_eq!(
            child.file_name(),
            Some(std::ffi::OsStr::new("_llm_guard_proxy_workflow_42_7"))
        );
    }

    #[test]
    fn unavailable_and_delegation_failures_choose_fallback() {
        for reason in [
            UnavailableReason::UnifiedHierarchyMissing,
            UnavailableReason::CurrentMembershipMissing,
            UnavailableReason::KillInterfaceMissing,
            UnavailableReason::DelegationMissing,
            UnavailableReason::PermissionDenied,
        ] {
            assert_eq!(
                optional_containment::<()>(Err(SetupFailure::Unavailable(reason)))
                    .expect("unavailability should not be fatal"),
                None
            );
        }

        let fatal = optional_containment::<()>(Err(SetupFailure::Fatal(io::Error::new(
            io::ErrorKind::InvalidData,
            "malformed metadata",
        ))));
        assert_eq!(
            fatal
                .expect_err("invalid metadata must remain fatal")
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    struct TestChild(Option<std::process::Child>);

    impl Drop for TestChild {
        fn drop(&mut self) {
            if let Some(child) = self.0.as_mut() {
                let _kill = child.kill();
                let _wait = child.wait();
            }
        }
    }

    #[test]
    fn delegated_cgroup_lifecycle_kills_setsid_descendant() {
        let Some(cgroup) = WorkflowCgroup::prepare().expect("cgroup discovery should be valid")
        else {
            eprintln!("skipped: cgroup v2 delegation is unavailable");
            return;
        };
        let child = Command::new("/bin/sh")
            .args([
                "-c",
                "IFS= read -r ready || exit 1; /usr/bin/setsid /bin/sleep 30 & wait",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("test child should spawn");
        let mut child = TestChild(Some(child));
        let deadline = Instant::now() + Duration::from_secs(2);
        if !cgroup
            .attach_or_fallback(
                child.0.as_ref().expect("test child should be owned").id(),
                deadline,
            )
            .expect("attach should either succeed or fall back")
        {
            eprintln!("skipped: cgroup v2 process migration is not delegated");
            return;
        }
        child
            .0
            .as_mut()
            .and_then(|child| child.stdin.take())
            .expect("attached child should have piped stdin")
            .write_all(b"ready\n")
            .expect("attached child should receive release handshake");
        let descendant_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let process_count = fs::read_to_string(cgroup.path.join("cgroup.procs"))
                .expect("owned cgroup process list should be readable")
                .lines()
                .count();
            if process_count >= 2 {
                break;
            }
            assert!(
                Instant::now() < descendant_deadline,
                "setsid descendant did not enter the owned cgroup"
            );
            thread::sleep(CLEANUP_POLL);
        }

        cgroup
            .kill_and_remove(deadline)
            .expect("owned cgroup should become empty and be removed");
        let status = child
            .0
            .as_mut()
            .expect("test child should be owned")
            .wait()
            .expect("killed test child should be reaped");
        assert!(!status.success());
        child.0.take();
    }
}
