//! Optional delegated cgroup v2 containment for one workflow execution.
//!
//! The module only creates and removes direct children of the proxy's current cgroup. A missing
//! unified hierarchy, missing `cgroup.kill`, or insufficient delegation falls back to the existing
//! process-group cleanup. Malformed kernel metadata and other unexpected I/O failures remain fatal.

use std::{
    ffi::{CStr, CString, OsStr, OsString},
    fs::{self, File},
    io::{self, Read},
    mem,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::{
            ffi::{OsStrExt, OsStringExt},
            process::ExitStatusExt,
        },
    },
    path::{Component, Path, PathBuf},
    process::{self, ExitStatus},
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
const CLONE_INTO_CGROUP: u64 = 0x2_0000_0000;
const SHELL_PATH: &CStr = c"/bin/sh";
const SHELL_FLAG: &CStr = c"-c";
const SHELL_COMMAND: &CStr = c"exec /bin/sh \"$0\" \"$@\"";

static NEXT_CGROUP_ID: AtomicU64 = AtomicU64::new(0);
static FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);
static ATOMIC_SPAWN_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

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
    directory: File,
}

/// Result of attempting the Linux atomic cgroup spawn path.
pub(crate) enum AtomicSpawnOutcome {
    Spawned(AtomicWorkflowChild),
    /// A compatibility or policy error rejected atomic placement; use legacy spawn+attach.
    Fallback(io::Error),
}

enum CloneFailure {
    Fallback(io::Error),
    Fatal(io::Error),
}

/// Child handle returned by `clone3`, with the pipe and wait operations used by the runtime.
pub(crate) struct AtomicWorkflowChild {
    pid: libc::pid_t,
    status: Option<ExitStatus>,
    pub(crate) stdin: Option<File>,
    pub(crate) stdout: Option<File>,
    pub(crate) stderr: Option<File>,
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
                    let directory = match File::open(&path) {
                        Ok(directory) => directory,
                        Err(error) => {
                            let _removed = fs::remove_dir(&path);
                            return Err(classify_setup_io(
                                "open workflow cgroup directory",
                                &error,
                            ));
                        }
                    };
                    let cgroup = Self { path, directory };
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

    /// Spawns a process directly into this cgroup with `clone3(CLONE_INTO_CGROUP)`.
    ///
    /// The cgroup directory was opened during preparation, before this function creates the child.
    /// Allowlisted compatibility and policy rejections return [`AtomicSpawnOutcome::Fallback`] so
    /// the caller can retain the legacy `Command::spawn` plus `cgroup.procs` migration path.
    /// Resource, implementation, preparation, and `execve` failures remain ordinary spawn errors.
    pub(crate) fn spawn_atomic(
        &self,
        program: &OsStr,
        arguments: &[OsString],
        environment: &[(OsString, OsString)],
    ) -> io::Result<AtomicSpawnOutcome> {
        let executable = ExecData::new(program, arguments, environment)?;
        let stdin_pipe = Pipe::new()?;
        let stdout_pipe = Pipe::new()?;
        let stderr_pipe = Pipe::new()?;
        let exec_error_pipe = Pipe::new()?;
        let mut clone_arguments = libc::clone_args {
            flags: CLONE_INTO_CGROUP,
            pidfd: 0,
            child_tid: 0,
            parent_tid: 0,
            exit_signal: u64::try_from(libc::SIGCHLD).expect("SIGCHLD should be non-negative"),
            stack: 0,
            stack_size: 0,
            tls: 0,
            set_tid: 0,
            set_tid_size: 0,
            cgroup: u64::try_from(self.directory.as_raw_fd())
                .expect("an open cgroup descriptor should be non-negative"),
        };

        let result = match classify_clone_result(clone_into_cgroup(&mut clone_arguments)) {
            Ok(result) => result,
            Err(CloneFailure::Fallback(error)) => {
                return Ok(AtomicSpawnOutcome::Fallback(error));
            }
            Err(CloneFailure::Fatal(error)) => return Err(error),
        };
        if result == 0 {
            // SAFETY: this is the post-clone child branch. The referenced descriptors and all
            // precomputed C strings remain live in its copied address space.
            unsafe {
                child_exec(
                    &executable,
                    stdin_pipe.read.as_raw_fd(),
                    stdout_pipe.write.as_raw_fd(),
                    stderr_pipe.write.as_raw_fd(),
                    exec_error_pipe.write.as_raw_fd(),
                )
            }
        }

        let pid = libc::pid_t::try_from(result)
            .map_err(|_| io::Error::other("clone3 returned an invalid workflow child PID"))?;
        drop(stdin_pipe.read);
        drop(stdout_pipe.write);
        drop(stderr_pipe.write);
        drop(exec_error_pipe.write);

        match read_exec_error(exec_error_pipe.read) {
            Ok(None) => Ok(AtomicSpawnOutcome::Spawned(AtomicWorkflowChild {
                pid,
                status: None,
                stdin: Some(File::from(stdin_pipe.write)),
                stdout: Some(File::from(stdout_pipe.read)),
                stderr: Some(File::from(stderr_pipe.read)),
            })),
            Ok(Some(errno)) => {
                let _reaped = wait_for_pid(pid, 0);
                Err(io::Error::from_raw_os_error(errno))
            }
            Err(error) => {
                // The child may have executed, so terminate and reap it before returning no handle.
                // SAFETY: `pid` is the positive PID returned by clone3 and is still our child.
                let _killed = unsafe { libc::kill(pid, libc::SIGKILL) };
                let _reaped = wait_for_pid(pid, 0);
                Err(io::Error::new(
                    error.kind(),
                    format!("failed to confirm workflow exec: {error}"),
                ))
            }
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

impl AtomicWorkflowChild {
    pub(crate) fn id(&self) -> u32 {
        u32::try_from(self.pid).expect("clone3 returned a positive PID")
    }

    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status));
        }
        let status = wait_for_pid(self.pid, libc::WNOHANG)?;
        if let Some(status) = status {
            self.status = Some(status);
        }
        Ok(status)
    }

    #[cfg(test)]
    pub(crate) fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        let status = wait_for_pid(self.pid, 0)?.ok_or_else(|| {
            io::Error::other("blocking waitpid unexpectedly reported a running child")
        })?;
        self.status = Some(status);
        Ok(status)
    }
}

struct Pipe {
    read: OwnedFd,
    write: OwnedFd,
}

impl Pipe {
    fn new() -> io::Result<Self> {
        let mut descriptors = [-1; 2];
        // SAFETY: `descriptors` points to writable storage for the two descriptors from pipe2.
        if unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) } == -1 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: pipe2 succeeded and returned two newly owned descriptors.
        let read = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
        // SAFETY: pipe2 succeeded and returned two newly owned descriptors.
        let write = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
        Ok(Self {
            read: move_above_standard_streams(read)?,
            write: move_above_standard_streams(write)?,
        })
    }
}

fn move_above_standard_streams(descriptor: OwnedFd) -> io::Result<OwnedFd> {
    if descriptor.as_raw_fd() > libc::STDERR_FILENO {
        return Ok(descriptor);
    }
    // SAFETY: `descriptor` is open and F_DUPFD_CLOEXEC returns a distinct owned descriptor.
    let duplicate = unsafe {
        libc::fcntl(
            descriptor.as_raw_fd(),
            libc::F_DUPFD_CLOEXEC,
            libc::STDERR_FILENO + 1,
        )
    };
    if duplicate == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fcntl succeeded and returned a newly owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicate) })
}

struct ExecData {
    executable_paths: Vec<CString>,
    shell_argument_pointers: Vec<Vec<*const libc::c_char>>,
    _arguments: Vec<CString>,
    argument_pointers: Vec<*const libc::c_char>,
    _environment: Vec<CString>,
    environment_pointers: Vec<*const libc::c_char>,
}

impl ExecData {
    fn new(
        program: &OsStr,
        arguments: &[OsString],
        environment: &[(OsString, OsString)],
    ) -> io::Result<Self> {
        let executable_paths = executable_paths(program, environment)?;
        let mut argument_strings = Vec::with_capacity(arguments.len().saturating_add(1));
        argument_strings.push(os_string_to_c_string(program, "workflow command")?);
        for argument in arguments {
            argument_strings.push(os_string_to_c_string(argument, "workflow argument")?);
        }
        let mut argument_pointers = argument_strings
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        argument_pointers.push(std::ptr::null());
        let shell_argument_pointers = executable_paths
            .iter()
            .map(|path| {
                let mut pointers = Vec::with_capacity(argument_strings.len().saturating_add(4));
                pointers.extend([
                    SHELL_PATH.as_ptr(),
                    SHELL_FLAG.as_ptr(),
                    SHELL_COMMAND.as_ptr(),
                    path.as_ptr(),
                ]);
                pointers.extend(argument_strings.iter().skip(1).map(|value| value.as_ptr()));
                pointers.push(std::ptr::null());
                pointers
            })
            .collect();

        let mut environment_strings = Vec::with_capacity(environment.len());
        for (key, value) in environment {
            let mut entry = Vec::with_capacity(
                key.as_bytes()
                    .len()
                    .saturating_add(value.as_bytes().len())
                    .saturating_add(1),
            );
            entry.extend_from_slice(key.as_bytes());
            entry.push(b'=');
            entry.extend_from_slice(value.as_bytes());
            environment_strings.push(CString::new(entry).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "workflow environment contains a NUL byte",
                )
            })?);
        }
        let mut environment_pointers = environment_strings
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        environment_pointers.push(std::ptr::null());

        Ok(Self {
            executable_paths,
            shell_argument_pointers,
            _arguments: argument_strings,
            argument_pointers,
            _environment: environment_strings,
            environment_pointers,
        })
    }
}

fn executable_paths(
    program: &OsStr,
    environment: &[(OsString, OsString)],
) -> io::Result<Vec<CString>> {
    if program.as_bytes().contains(&b'/') {
        return Ok(vec![os_string_to_c_string(program, "workflow command")?]);
    }
    let path = environment
        .iter()
        .find_map(|(key, value)| (key == OsStr::new("PATH")).then_some(value.as_os_str()))
        .unwrap_or_else(|| OsStr::new("/bin:/usr/bin"));
    std::env::split_paths(path)
        .map(|directory| {
            os_string_to_c_string(
                directory.join(program).as_os_str(),
                "resolved workflow command",
            )
        })
        .collect()
}

fn os_string_to_c_string(value: &OsStr, label: &'static str) -> io::Result<CString> {
    CString::new(value.as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} contains a NUL byte"),
        )
    })
}

unsafe fn child_exec(
    executable: &ExecData,
    stdin_fd: RawFd,
    stdout_fd: RawFd,
    stderr_fd: RawFd,
    exec_error_fd: RawFd,
) -> ! {
    for (source, target) in [
        (stdin_fd, libc::STDIN_FILENO),
        (stdout_fd, libc::STDOUT_FILENO),
        (stderr_fd, libc::STDERR_FILENO),
    ] {
        // SAFETY: called only in the clone3 child; both descriptors are valid and dup2 is
        // async-signal-safe.
        if unsafe { libc::dup2(source, target) } == -1 {
            // SAFETY: the exec-error descriptor is valid in this child.
            unsafe { write_errno_and_exit(exec_error_fd, current_errno()) }
        }
    }
    // Match `std::process::Command` by undoing Rust's inherited SIGPIPE ignore before exec.
    // SAFETY: called only in the post-clone child and sigaction is async-signal-safe.
    if let Err(errno) = unsafe { reset_sigpipe() } {
        // SAFETY: the exec-error descriptor is valid in this child.
        unsafe { write_errno_and_exit(exec_error_fd, errno) }
    }
    // SAFETY: called only in the child before user code; setpgid is async-signal-safe.
    if unsafe { libc::setpgid(0, 0) } == -1 {
        // SAFETY: the exec-error descriptor is valid in this child.
        unsafe { write_errno_and_exit(exec_error_fd, current_errno()) }
    }

    let mut permission_denied = false;
    for (executable_path, shell_arguments) in executable
        .executable_paths
        .iter()
        .zip(&executable.shell_argument_pointers)
    {
        // SAFETY: every pointer references a live NUL-terminated CString or the required final
        // null pointer. execve is async-signal-safe and returns only on failure.
        unsafe {
            libc::execve(
                executable_path.as_ptr(),
                executable.argument_pointers.as_ptr(),
                executable.environment_pointers.as_ptr(),
            )
        };
        let errno = current_errno();
        if errno == libc::ENOEXEC {
            // Match execvp for executable text without a shebang. The complete `sh -c` argv was
            // built in the parent, and its positional parameters avoid interpolating user input.
            // SAFETY: all pointers reference precomputed live CStrings, and execve is
            // async-signal-safe.
            unsafe {
                libc::execve(
                    SHELL_PATH.as_ptr(),
                    shell_arguments.as_ptr(),
                    executable.environment_pointers.as_ptr(),
                )
            };
            // SAFETY: the exec-error descriptor is valid in this child.
            unsafe { write_errno_and_exit(exec_error_fd, current_errno()) }
        } else if errno == libc::EACCES {
            permission_denied = true;
        } else if errno != libc::ENOENT && errno != libc::ENOTDIR {
            // SAFETY: the exec-error descriptor is valid in this child.
            unsafe { write_errno_and_exit(exec_error_fd, errno) }
        }
    }
    let errno = if permission_denied {
        libc::EACCES
    } else {
        libc::ENOENT
    };
    // SAFETY: the exec-error descriptor is valid in this child.
    unsafe { write_errno_and_exit(exec_error_fd, errno) }
}

unsafe fn reset_sigpipe() -> Result<(), libc::c_int> {
    // SAFETY: all-zero is a valid starting representation for Linux `sigaction`.
    let mut action = unsafe { mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = libc::SIG_DFL;
    // SAFETY: `action.sa_mask` is writable storage for a signal set.
    if unsafe { libc::sigemptyset(&raw mut action.sa_mask) } == -1 {
        return Err(current_errno());
    }
    // SAFETY: `action` is fully initialized and SIGPIPE is a valid signal number.
    if unsafe { libc::sigaction(libc::SIGPIPE, &raw const action, std::ptr::null_mut()) } == -1 {
        return Err(current_errno());
    }
    Ok(())
}

fn current_errno() -> libc::c_int {
    // SAFETY: Linux libc exposes a thread-local errno pointer valid for the calling thread.
    unsafe { *libc::__errno_location() }
}

unsafe fn write_errno_and_exit(exec_error_fd: RawFd, errno: libc::c_int) -> ! {
    let bytes = errno.to_ne_bytes();
    // SAFETY: `bytes` points to four initialized bytes and the descriptor is the inherited
    // exec-error pipe. A short/failed write still must terminate this post-clone child.
    let _written = unsafe {
        libc::write(
            exec_error_fd,
            bytes.as_ptr().cast::<libc::c_void>(),
            bytes.len(),
        )
    };
    // SAFETY: this branch must not run Rust destructors after clone3.
    unsafe { libc::_exit(127) }
}

fn read_exec_error(descriptor: OwnedFd) -> io::Result<Option<libc::c_int>> {
    let mut file = File::from(descriptor);
    let mut bytes = [0_u8; mem::size_of::<libc::c_int>()];
    let mut read = 0;
    loop {
        match file.read(&mut bytes[read..]) {
            Ok(0) if read == 0 => return Ok(None),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "workflow exec error pipe closed with a partial errno",
                ));
            }
            Ok(count) => {
                read = read.saturating_add(count);
                if read == bytes.len() {
                    return Ok(Some(libc::c_int::from_ne_bytes(bytes)));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

fn clone_into_cgroup(arguments: &mut libc::clone_args) -> io::Result<libc::c_long> {
    // SAFETY: this is fork-like clone3 (no shared VM/files/thread flags). In the child branch,
    // `child_exec` calls only async-signal-safe libc operations before `execve` or `_exit`.
    let result = unsafe {
        libc::syscall(
            libc::SYS_clone3,
            &raw mut *arguments,
            mem::size_of::<libc::clone_args>(),
        )
    };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result)
    }
}

fn classify_clone_result(result: io::Result<libc::c_long>) -> Result<libc::c_long, CloneFailure> {
    result.map_err(|error| {
        if clone_error_allows_fallback(&error) {
            CloneFailure::Fallback(error)
        } else {
            CloneFailure::Fatal(error)
        }
    })
}

fn clone_error_allows_fallback(error: &io::Error) -> bool {
    // ENOSYS covers old kernels, while EPERM/EACCES cover seccomp and cgroup placement policy.
    // Resource exhaustion and invalid syscall state must never downgrade atomic containment.
    matches!(
        error.raw_os_error(),
        Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
    )
}

fn wait_for_pid(pid: libc::pid_t, options: libc::c_int) -> io::Result<Option<ExitStatus>> {
    loop {
        let mut status = 0;
        // SAFETY: `pid` is a child returned by clone3 and `status` is writable storage.
        let result = unsafe { libc::waitpid(pid, &raw mut status, options) };
        if result == pid {
            return Ok(Some(ExitStatus::from_raw(status)));
        }
        if result == 0 {
            return Ok(None);
        }
        if result == -1 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        return Err(io::Error::other(
            "waitpid returned an unexpected workflow PID",
        ));
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

pub(crate) fn log_atomic_spawn_fallback_once(error: &io::Error) {
    if ATOMIC_SPAWN_FALLBACK_LOGGED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        eprintln!(
            "clone3(CLONE_INTO_CGROUP) unavailable ({error}); using spawn+attach cgroup fallback"
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
        os::unix::{ffi::OsStrExt, fs::PermissionsExt},
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

    #[test]
    fn atomic_spawn_places_child_in_expected_cgroup_before_exec() {
        let Some(cgroup) = WorkflowCgroup::prepare().expect("cgroup discovery should be valid")
        else {
            eprintln!("skipped: cgroup v2 delegation is unavailable");
            return;
        };
        let expected_parent = parse_current_cgroup_path(
            &fs::read_to_string(CURRENT_CGROUP).expect("current cgroup should be readable"),
        )
        .expect("current cgroup membership should parse")
        .expect("current cgroup v2 membership should exist");
        let expected = expected_parent.join(
            cgroup
                .path
                .file_name()
                .expect("workflow cgroup should have a file name"),
        );

        let mut child = match cgroup
            .spawn_atomic(
                std::ffi::OsStr::new("/bin/sleep"),
                &[std::ffi::OsString::from("30")],
                &[],
            )
            .expect("atomic spawn preparation should succeed")
        {
            AtomicSpawnOutcome::Spawned(child) => child,
            AtomicSpawnOutcome::Fallback(error) => {
                eprintln!("skipped: clone3(CLONE_INTO_CGROUP) is unavailable: {error}");
                return;
            }
        };
        let membership = fs::read_to_string(format!("/proc/{}/cgroup", child.id()))
            .expect("spawned child cgroup membership should be readable");
        let actual = parse_current_cgroup_path(&membership)
            .expect("spawned child cgroup membership should parse")
            .expect("spawned child cgroup v2 membership should exist");

        assert_eq!(actual, expected);

        let deadline = Instant::now() + Duration::from_secs(2);
        cgroup
            .kill_and_remove(deadline)
            .expect("owned cgroup should become empty and be removed");
        let status = child.wait().expect("killed child should be reaped");
        assert!(!status.success());
    }

    #[test]
    fn child_exec_uses_shell_fallback_for_shebangless_scripts() {
        let script = std::env::temp_dir().join(format!(
            "llm_guard_proxy_shebangless_{}_{}",
            std::process::id(),
            NEXT_CGROUP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&script, b"printf '%s\\n' \"$0\" \"$1\"\n")
            .expect("shebang-less test script should be written");
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700))
            .expect("shebang-less test script should be executable");

        let executable = ExecData::new(
            script.as_os_str(),
            &[OsString::from("argument with spaces")],
            &[],
        )
        .expect("shell fallback inputs should be precomputed");
        let stdin_pipe = Pipe::new().expect("stdin pipe should be created");
        let stdout_pipe = Pipe::new().expect("stdout pipe should be created");
        let stderr_pipe = Pipe::new().expect("stderr pipe should be created");
        let exec_error_pipe = Pipe::new().expect("exec error pipe should be created");

        // SAFETY: the child immediately enters `child_exec`, which is the production post-clone
        // async-signal-safe path; the parent retains ownership of its independent pipe ends.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // SAFETY: every descriptor and all precomputed exec data remain live in this child.
            unsafe {
                child_exec(
                    &executable,
                    stdin_pipe.read.as_raw_fd(),
                    stdout_pipe.write.as_raw_fd(),
                    stderr_pipe.write.as_raw_fd(),
                    exec_error_pipe.write.as_raw_fd(),
                )
            }
        }
        if pid == -1 {
            let error = io::Error::last_os_error();
            let _removed = fs::remove_file(&script);
            panic!("fork for shell-fallback test failed: {error}");
        }

        drop(stdin_pipe.read);
        drop(stdin_pipe.write);
        drop(stdout_pipe.write);
        drop(stderr_pipe.write);
        drop(exec_error_pipe.write);
        let exec_error =
            read_exec_error(exec_error_pipe.read).expect("exec error handshake should be readable");
        let mut stdout = Vec::new();
        File::from(stdout_pipe.read)
            .read_to_end(&mut stdout)
            .expect("shell-fallback stdout should be readable");
        let mut stderr = Vec::new();
        File::from(stderr_pipe.read)
            .read_to_end(&mut stderr)
            .expect("shell-fallback stderr should be readable");
        let status = wait_for_pid(pid, 0)
            .expect("shell-fallback child should be waitable")
            .expect("blocking wait should return shell-fallback status");
        fs::remove_file(&script).expect("shebang-less test script should be removed");

        assert_eq!(exec_error, None, "atomic exec reported errno");
        assert!(
            status.success(),
            "shell fallback exited {status}; stderr: {}",
            String::from_utf8_lossy(&stderr)
        );
        let mut expected = script.as_os_str().as_bytes().to_vec();
        expected.extend_from_slice(b"\nargument with spaces\n");
        assert_eq!(stdout, expected);
    }

    #[test]
    fn clone3_compatibility_failures_select_legacy_spawn_and_attach_fallback() {
        for errno in [libc::ENOSYS, libc::EPERM, libc::EACCES] {
            match classify_clone_result(Err(io::Error::from_raw_os_error(errno))) {
                Err(CloneFailure::Fallback(error)) => {
                    assert_eq!(error.raw_os_error(), Some(errno));
                }
                Err(CloneFailure::Fatal(error)) => {
                    panic!("errno {errno} must select fallback, not fail: {error}");
                }
                Ok(_) => panic!("an injected clone3 error cannot be successful"),
            }
        }
    }

    #[test]
    fn clone3_unexpected_failures_are_propagated() {
        for errno in [libc::EAGAIN, libc::ENOMEM, libc::EBADF, libc::EFAULT] {
            match classify_clone_result(Err(io::Error::from_raw_os_error(errno))) {
                Err(CloneFailure::Fatal(error)) => {
                    assert_eq!(error.raw_os_error(), Some(errno));
                }
                Err(CloneFailure::Fallback(error)) => {
                    panic!("errno {errno} was silently downgraded to fallback: {error}");
                }
                Ok(_) => panic!("an injected clone3 error cannot be successful"),
            }
        }
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
