use llm_guard_proxy_host_guardian::{CgroupTarget, EmergencyReserve, kill_direct};
use nix::{
    fcntl::OFlag,
    poll::{PollFd, PollFlags, PollTimeout, poll},
    unistd::Uid,
};
use std::{
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsFd, AsRawFd},
        unix::fs::{FileExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{SyncSender, sync_channel},
    thread,
    time::{Duration, Instant},
};

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const CHILD_SECONDS: &str = "120";
const DEADLINE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);
const CHILD_REAP_GRACE: Duration = Duration::from_millis(250);
const SYSTEMCTL_OUTPUT_LIMIT: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessIdentity {
    pid: u32,
    pgid: u32,
    starttime: u64,
}

struct Scratch(PathBuf);

impl Scratch {
    fn create(id: &str) -> io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("llm-guard-real-cgroup-{}-{id}", std::process::id()));
        fs::create_dir(&path)?;
        let directory = Self(path);
        fs::set_permissions(&directory.0, fs::Permissions::from_mode(0o700))?;
        Ok(directory)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct OwnedScope {
    unit: String,
    control_group: String,
    cgroup_path: PathBuf,
    launcher: Child,
    target: Option<ProcessIdentity>,
}

impl OwnedScope {
    fn wait(&mut self) -> io::Result<ExitStatus> {
        wait_for_exit(&mut self.launcher, DEADLINE)
    }

    fn act_on_scope(&self, arguments: &[&str]) {
        if !self.contains_only_target()
            || systemd_property(&self.unit, "ControlGroup").ok().as_deref()
                != Some(self.control_group.as_str())
        {
            return;
        }
        let Ok(mut action) = Command::new("/usr/bin/systemctl")
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            return;
        };
        if wait_for_exit(&mut action, DEADLINE.saturating_sub(CHILD_REAP_GRACE)).is_err() {
            let _ = action.kill();
            let _ = wait_for_exit(&mut action, CHILD_REAP_GRACE);
        }
    }

    fn contains_only_target(&self) -> bool {
        let Ok(members) = cgroup_members(&self.cgroup_path) else {
            return false;
        };
        if members.len() != 1 {
            return false;
        }
        let pid = members[0];
        let Ok(current) = process_identity(pid) else {
            return false;
        };
        let Ok(cgroup) = fs::read_to_string(format!("/proc/{pid}/cgroup")) else {
            return false;
        };
        let Ok(command) = process_command(pid) else {
            return false;
        };
        let identity_matches = match self.target {
            Some(expected) => current == expected,
            None => true,
        };
        identity_matches
            && cgroup
                .lines()
                .any(|line| line == format!("0::{}", self.control_group))
            && command == ["/usr/bin/sleep", CHILD_SECONDS]
    }
}

impl Drop for OwnedScope {
    fn drop(&mut self) {
        self.act_on_scope(&["--user", "stop", &self.unit]);
        let needs_kill = wait_for_exit(&mut self.launcher, DEADLINE).is_err();
        if needs_kill {
            self.act_on_scope(&["--user", "kill", "--signal=KILL", &self.unit]);
        }
        if wait_for_exit(&mut self.launcher, DEADLINE).is_err() {
            let _ = self.launcher.kill();
            let _ = wait_for_exit(&mut self.launcher, CHILD_REAP_GRACE);
        }
    }
}

fn random_id() -> io::Result<String> {
    let mut bytes = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let mut id = String::with_capacity(64);
    for byte in bytes {
        write!(&mut id, "{byte:02x}").expect("formatting into String cannot fail");
    }
    Ok(id)
}

fn process_identity(pid: u32) -> io::Result<ProcessIdentity> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
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

fn process_command(pid: u32) -> io::Result<Vec<String>> {
    Ok(fs::read(format!("/proc/{pid}/cmdline"))?
        .split(|byte| *byte == 0)
        .filter(|argument| !argument.is_empty())
        .map(|argument| String::from_utf8_lossy(argument).into_owned())
        .collect())
}

fn cgroup_members(path: &Path) -> io::Result<Vec<u32>> {
    cgroup_members_from(&fs::read_to_string(path.join("cgroup.procs"))?)
}

fn cgroup_members_from(contents: &str) -> io::Result<Vec<u32>> {
    contents
        .lines()
        .map(|pid| {
            pid.parse::<u32>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        })
        .collect()
}

fn populated(path: &Path) -> io::Result<bool> {
    populated_from(&fs::read_to_string(path.join("cgroup.events"))?)
}

fn populated_from(contents: &str) -> io::Result<bool> {
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

struct CgroupObservation {
    directory: File,
    events: File,
    procs: File,
}

impl CgroupObservation {
    fn open(path: &Path) -> io::Result<Self> {
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

    fn is_populated(&self) -> io::Result<bool> {
        populated_from(&read_pinned_cgroup_file(&self.events)?)
    }

    fn members(&self) -> io::Result<Vec<u32>> {
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

    fn wait_until_empty(&self, mut ready: Option<SyncSender<()>>) -> io::Result<()> {
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
            if let Some(ready) = ready.take() {
                ready.send(()).map_err(|error| {
                    io::Error::new(io::ErrorKind::BrokenPipe, error.to_string())
                })?;
            }
            self.wait_for_population_change(deadline.saturating_duration_since(Instant::now()))?;
        }
    }
}

fn wait_until_empty(path: &Path) -> io::Result<()> {
    CgroupObservation::open(path)?.wait_until_empty(None)
}

fn wait_until_populated(path: &Path, launcher: &mut Child) -> io::Result<ProcessIdentity> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(status) = launcher.try_wait()? {
            return Err(io::Error::other(format!(
                "systemd-run exited before cgroup population: {status}"
            )));
        }
        let is_populated = match populated(path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(error),
        };
        if is_populated {
            let members = cgroup_members(path)?;
            if members.len() == 1 {
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

fn wait_for_exit(child: &mut Child, timeout: Duration) -> io::Result<ExitStatus> {
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

fn wait_until_removed(path: &Path) -> io::Result<()> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
            Ok(_) if Instant::now() >= deadline => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "scope cgroup was not removed",
                ));
            }
            Ok(_) => thread::sleep(POLL),
        }
    }
}

#[test]
fn missing_cgroup_is_not_an_empty_observation() {
    let scratch = Scratch::create("missing-cgroup").expect("scratch directory should be created");
    let missing_events = scratch.0.join("removed-cgroup");
    fs::create_dir(&missing_events).expect("fake cgroup directory should be created");
    let error = wait_until_empty(&missing_events)
        .expect_err("missing cgroup.events was never observed empty");
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}

#[test]
fn missing_cgroup_procs_is_not_an_empty_observation() {
    let scratch =
        Scratch::create("missing-cgroup-procs").expect("scratch directory should be created");
    let missing_procs = scratch.0.join("removed-cgroup");
    fs::create_dir(&missing_procs).expect("fake cgroup directory should be created");
    fs::write(missing_procs.join("cgroup.events"), b"populated 0\n")
        .expect("fake cgroup.events should be written");
    let error = wait_until_empty(&missing_procs)
        .expect_err("missing cgroup.procs was never observed empty");
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}

#[test]
fn systemd_property_timeout_reaps_its_owned_child() {
    let scratch = Scratch::create("systemd-timeout").expect("scratch directory should be created");
    let pid_file = scratch.0.join("systemctl.pid");
    let fake_systemctl = scratch.0.join("systemctl");
    fs::write(
        &fake_systemctl,
        "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$3\"\nexec /usr/bin/sleep 2\n",
    )
    .expect("fake systemctl should be written");
    fs::set_permissions(&fake_systemctl, fs::Permissions::from_mode(0o700))
        .expect("fake systemctl should be executable");

    let started = Instant::now();
    let result = systemd_property_with(
        &fake_systemctl,
        &pid_file.to_string_lossy(),
        "ControlGroup",
        Duration::from_millis(500),
    );
    let elapsed = started.elapsed();
    let error = result.expect_err("stalled systemctl must time out");
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(
        elapsed < Duration::from_millis(1500),
        "query took {elapsed:?}"
    );
    let pid = fs::read_to_string(pid_file)
        .expect("fake systemctl should publish its pid")
        .trim()
        .parse::<u32>()
        .expect("fake systemctl pid should parse");
    assert_eq!(
        process_identity(pid)
            .expect_err("timed-out systemctl child must be reaped")
            .kind(),
        io::ErrorKind::NotFound
    );
}

#[test]
fn systemd_property_rejects_oversized_output() {
    let scratch =
        Scratch::create("systemd-output-limit").expect("scratch directory should be created");
    let fake_systemctl = scratch.0.join("systemctl");
    fs::write(&fake_systemctl, "#!/bin/sh\nprintf '%4097s' ''\n")
        .expect("fake systemctl should be written");
    fs::set_permissions(&fake_systemctl, fs::Permissions::from_mode(0o700))
        .expect("fake systemctl should be executable");

    let error = systemd_property_with(
        &fake_systemctl,
        "ignored.scope",
        "ControlGroup",
        Duration::from_secs(1),
    )
    .expect_err("oversized systemctl output must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

fn systemd_property(unit: &str, property: &str) -> io::Result<String> {
    systemd_property_with(Path::new("/usr/bin/systemctl"), unit, property, DEADLINE)
}

fn systemd_property_with(
    systemctl: &Path,
    unit: &str,
    property: &str,
    timeout: Duration,
) -> io::Result<String> {
    let deadline = Instant::now() + timeout;
    let mut child = Command::new(systemctl)
        .args(["--user", "show", unit, "--property", property, "--value"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let status = match wait_for_exit(&mut child, timeout.saturating_sub(CHILD_REAP_GRACE)) {
        Ok(status) => status,
        Err(error) => {
            let _ = child.kill();
            let remaining = deadline.saturating_duration_since(Instant::now());
            wait_for_exit(&mut child, remaining)?;
            return Err(error);
        }
    };
    if !status.success() {
        return Err(io::Error::other("systemctl show failed"));
    }
    let mut output = Vec::new();
    child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("systemctl stdout was not captured"))?
        .take((SYSTEMCTL_OUTPUT_LIMIT + 1) as u64)
        .read_to_end(&mut output)?;
    if output.len() > SYSTEMCTL_OUTPUT_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "systemctl output exceeded limit",
        ));
    }
    Ok(String::from_utf8_lossy(&output).trim().to_owned())
}

#[test]
#[ignore = "requires a disposable user-systemd delegated scope; run just test-real-cgroup"]
fn registered_scope_kill_reaps_only_the_task_owned_child() -> Result<(), Box<dyn std::error::Error>>
{
    let id = random_id()?;
    let uid = Uid::effective().as_raw();
    let unit = format!("docker-{id}.scope");
    let control_group = format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}");
    let cgroup_path = Path::new(CGROUP_ROOT).join(control_group.trim_start_matches('/'));
    assert_eq!(systemd_property(&unit, "LoadState")?, "not-found");

    let launcher = Command::new("/usr/bin/systemd-run")
        .args(["--user", "--scope", "--quiet", "--collect"])
        .arg(format!("--unit={unit}"))
        .args([
            "--property=Delegate=yes",
            "--",
            "/usr/bin/sleep",
            CHILD_SECONDS,
        ])
        .stdout(Stdio::null())
        .spawn()?;
    let mut scope = OwnedScope {
        unit: unit.clone(),
        control_group: control_group.clone(),
        cgroup_path: cgroup_path.clone(),
        launcher,
        target: None,
    };
    let identity = wait_until_populated(&cgroup_path, &mut scope.launcher)?;
    scope.target = Some(identity);

    assert_eq!(systemd_property(&unit, "ControlGroup")?, control_group);
    assert_eq!(systemd_property(&unit, "Delegate")?, "yes");
    assert_eq!(cgroup_members(&cgroup_path)?, [identity.pid]);
    assert!(populated(&cgroup_path)?);
    assert_eq!(
        process_command(identity.pid)?,
        ["/usr/bin/sleep", CHILD_SECONDS]
    );
    assert!(
        fs::read_to_string(format!("/proc/{}/cgroup", identity.pid))?
            .lines()
            .any(|line| line == format!("0::{control_group}"))
    );
    let observation = CgroupObservation::open(&cgroup_path)?;
    assert_eq!(observation.directory.metadata()?.uid(), uid);
    println!(
        "real-cgroup fixture unit={unit} launcher_pid={} target_pid={} pgid={} starttime={} populated_before=1",
        scope.launcher.id(),
        identity.pid,
        identity.pgid,
        identity.starttime
    );

    let scratch = Scratch::create(&id)?;
    let registration = scratch.0.join("target-cgroup.v1");
    let contents =
        format!("version=1\ncontainer_id={id}\nscope={unit}\ncontrol_group={control_group}\n");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&registration)?;
    file.write_all(contents.as_bytes())?;
    drop(file);
    fs::set_permissions(&registration, fs::Permissions::from_mode(0o600))?;

    let target = CgroupTarget::open_registered(&registration, Path::new(CGROUP_ROOT))?;
    assert_eq!(cgroup_members(&cgroup_path)?, [identity.pid]);
    assert!(populated(&cgroup_path)?);
    let mut reserve = EmergencyReserve::with_page_size(4096, 4096)?;
    let (observer_ready, observer_started) = sync_channel(1);
    thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let observation_for_waiter = &observation;
        let observer =
            scope.spawn(move || observation_for_waiter.wait_until_empty(Some(observer_ready)));
        observer_started.recv_timeout(DEADLINE).map_err(|error| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("empty-state observer did not start: {error}"),
            )
        })?;
        kill_direct(&mut reserve, &target)?;
        println!("real-cgroup fixture cgroup.kill issued");
        observer
            .join()
            .map_err(|_| io::Error::other("empty-state observer panicked"))??;
        Ok(())
    })?;
    drop(observation);
    let _launcher_status = scope.wait()?;
    drop(target);

    match process_identity(identity.pid) {
        Ok(current) => assert_ne!(current.starttime, identity.starttime),
        Err(error) => assert_eq!(error.kind(), io::ErrorKind::NotFound),
    }
    wait_until_removed(&cgroup_path)?;
    assert_eq!(systemd_property(&unit, "LoadState")?, "not-found");
    let scratch_path = scratch.0.clone();
    drop(scratch);
    assert!(!scratch_path.exists());
    println!(
        "registration opened; populated=1 -> cgroup.kill -> observed populated=0 and empty cgroup.procs; exact child reaped; scope removed"
    );
    Ok(())
}
