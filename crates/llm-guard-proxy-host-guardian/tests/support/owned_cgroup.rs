//! Private helpers shared by the opt-in kernel acceptance leaves.
use nix::{
    fcntl::OFlag,
    poll::{PollFd, PollFlags, PollTimeout, poll},
};
use std::{
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{self, Read},
    os::{
        fd::{AsFd, AsRawFd},
        unix::{
            fs::OpenOptionsExt,
            fs::{FileExt, MetadataExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const CHILD_SECONDS: &str = "120";
const DEADLINE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProcessIdentity {
    pub(crate) pid: u32,
    pub(crate) pgid: u32,
    pub(crate) starttime: u64,
}

// The parent owns collection: an empty kernel cgroup is not a transient systemd unit.
// Held FDs alone cannot fence systemd's removal of an empty scope.
pub(crate) struct OwnedCgroup {
    pub(crate) path: PathBuf,
    pub(crate) directory: File,
    pub(crate) child: Option<Child>,
}

impl OwnedCgroup {
    pub(crate) fn create(path: PathBuf) -> io::Result<Self> {
        fs::create_dir(&path)?;
        let directory = match OpenOptions::new()
            .read(true)
            .custom_flags((OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC).bits())
            .open(&path)
        {
            Ok(directory) => directory,
            Err(error) => {
                let _ = fs::remove_dir(&path);
                return Err(error);
            }
        };
        let mut owned = Self {
            path,
            directory,
            child: None,
        };
        owned.child = Some(
            Command::new("/bin/sh")
                .args([
                    "-ec",
                    "printf '%s\\n' $$ > \"$1/cgroup.procs\"; exec /usr/bin/sleep \"$2\"",
                    "owned-cgroup",
                ])
                .arg(&owned.path)
                .arg(CHILD_SECONDS)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()?,
        );
        Ok(owned)
    }

    pub(crate) fn remove(&self) -> io::Result<()> {
        let original = self.directory.metadata()?;
        let current = fs::symlink_metadata(&self.path)?;
        if (original.dev(), original.ino()) != (current.dev(), current.ino()) {
            return Err(io::Error::other("owned cgroup identity changed"));
        }
        fs::remove_dir(&self.path)
    }
}

impl Drop for OwnedCgroup {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = wait_for_exit(child, DEADLINE);
        }
        let _ = self.remove();
    }
}

pub(crate) fn random_id() -> io::Result<String> {
    let mut bytes = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let mut id = String::with_capacity(64);
    for byte in bytes {
        write!(&mut id, "{byte:02x}").expect("formatting into String cannot fail");
    }
    Ok(id)
}

pub(crate) fn process_identity(pid: u32) -> io::Result<ProcessIdentity> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    process_identity_from(pid, &stat)
}

pub(crate) fn process_identity_from(pid: u32, stat: &str) -> io::Result<ProcessIdentity> {
    let fields = stat
        .rsplit_once(')')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed proc stat"))?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    if fields.len() <= 19 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "short proc stat",
        ));
    }
    let parse = |field: &str| {
        field
            .parse::<u64>()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    };
    Ok(ProcessIdentity {
        pid,
        pgid: u32::try_from(parse(fields[2])?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        starttime: parse(fields[19])?,
    })
}

pub(crate) fn process_command(pid: u32) -> io::Result<Vec<String>> {
    Ok(fs::read(format!("/proc/{pid}/cmdline"))?
        .split(|byte| *byte == 0)
        .filter(|argument| !argument.is_empty())
        .map(|argument| String::from_utf8_lossy(argument).into_owned())
        .collect())
}

pub(crate) fn cgroup_members(path: &Path) -> io::Result<Vec<u32>> {
    cgroup_members_from(&fs::read_to_string(path.join("cgroup.procs"))?)
}

pub(crate) fn cgroup_members_from(contents: &str) -> io::Result<Vec<u32>> {
    contents
        .lines()
        .map(|pid| {
            pid.parse::<u32>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        })
        .collect()
}

pub(crate) fn populated(path: &Path) -> io::Result<bool> {
    populated_from(&fs::read_to_string(path.join("cgroup.events"))?)
}

pub(crate) fn populated_from(contents: &str) -> io::Result<bool> {
    match contents
        .lines()
        .find_map(|line| line.strip_prefix("populated "))
    {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid populated field",
        )),
    }
}

fn read_pinned_cgroup_file(file: &File) -> io::Result<String> {
    let mut bytes = [0_u8; 256];
    let length = file.read_at(&mut bytes, 0)?;
    if length == bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "pinned cgroup file exceeded buffer",
        ));
    }
    String::from_utf8(bytes[..length].to_vec())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub(crate) struct CgroupObservation {
    pub(crate) directory: File,
    events: File,
    procs: File,
}

impl CgroupObservation {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags((OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC).bits())
            .open(path)?;
        let cgroup_directory = Path::new("/proc/self/fd").join(directory.as_raw_fd().to_string());
        let flags = (OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC).bits();
        let events = OpenOptions::new()
            .read(true)
            .custom_flags(flags)
            .open(cgroup_directory.join("cgroup.events"))?;
        let procs = OpenOptions::new()
            .read(true)
            .custom_flags(flags)
            .open(cgroup_directory.join("cgroup.procs"))?;
        Ok(Self {
            directory,
            events,
            procs,
        })
    }

    pub(crate) fn is_populated(&self) -> io::Result<bool> {
        populated_from(&read_pinned_cgroup_file(&self.events)?)
    }

    pub(crate) fn members(&self) -> io::Result<Vec<u32>> {
        cgroup_members_from(&read_pinned_cgroup_file(&self.procs)?)
    }

    fn wait_for_population_change(&self, timeout: Duration) -> io::Result<()> {
        let timeout = PollTimeout::try_from(timeout).unwrap_or(PollTimeout::MAX);
        let mut descriptor = [PollFd::new(self.events.as_fd(), PollFlags::POLLPRI)];
        let ready = match poll(&mut descriptor, timeout) {
            Ok(ready) => ready,
            Err(nix::errno::Errno::EINTR) => return Ok(()),
            Err(error) => return Err(io::Error::other(error)),
        };
        if ready > 0 {
            let events = descriptor[0].revents().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown cgroup.events poll flags",
                )
            })?;
            if events.contains(PollFlags::POLLNVAL) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "pinned cgroup.events descriptor is invalid",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn wait_until_empty(&self) -> io::Result<()> {
        let deadline = Instant::now() + DEADLINE;
        loop {
            let is_populated = self.is_populated().map_err(|error| {
                io::Error::new(error.kind(), format!("read pinned cgroup.events: {error}"))
            })?;
            let members = self.members().map_err(|error| {
                io::Error::new(error.kind(), format!("read pinned cgroup.procs: {error}"))
            })?;
            if !is_populated && members.is_empty() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "cgroup did not empty",
                ));
            }
            self.wait_for_population_change(deadline.saturating_duration_since(Instant::now()))?;
        }
    }
}

pub(crate) fn wait_until_populated(
    path: &Path,
    launcher: &mut Child,
) -> io::Result<ProcessIdentity> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(status) = launcher.try_wait()? {
            return Err(io::Error::other(format!(
                "owned child exited before cgroup population: {status}"
            )));
        }
        let is_populated = match populated(path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(error),
        };
        if is_populated {
            let members = cgroup_members(path)?;
            if members.len() == 1
                && process_command(members[0])? == ["/usr/bin/sleep", CHILD_SECONDS]
            {
                return process_identity(members[0]);
            }
            if members.len() > 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "task-owned scope contains more than one process",
                ));
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "scope did not populate",
            ));
        }
        thread::sleep(POLL);
    }
}

pub(crate) fn wait_for_exit(child: &mut Child, timeout: Duration) -> io::Result<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "owned child did not exit",
            ));
        }
        thread::sleep(POLL);
    }
}
