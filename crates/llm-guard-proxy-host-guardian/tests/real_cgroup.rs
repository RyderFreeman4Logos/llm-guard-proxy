use llm_guard_proxy_host_guardian::{CgroupTarget, EmergencyReserve, kill_direct};
use nix::unistd::Uid;
use std::{
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const CHILD_SECONDS: &str = "120";
const DEADLINE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);

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
        if wait_for_exit(&mut action, DEADLINE).is_err() {
            let _ = action.kill();
            let _ = action.wait();
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
        }
        let _ = self.launcher.wait();
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
    fs::read_to_string(path.join("cgroup.procs"))?
        .lines()
        .map(|pid| {
            pid.parse::<u32>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        })
        .collect()
}

fn populated(path: &Path) -> io::Result<bool> {
    match fs::read_to_string(path.join("cgroup.events"))?
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

fn wait_until_empty(path: &Path) -> io::Result<()> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        let is_populated = match populated(path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let members = match cgroup_members(path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        if !is_populated && members.is_empty() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "cgroup did not empty",
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

fn systemd_property(unit: &str, property: &str) -> io::Result<String> {
    let output = Command::new("/usr/bin/systemctl")
        .args(["--user", "show", unit, "--property", property, "--value"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other("systemctl show failed"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
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
    assert_eq!(fs::metadata(&cgroup_path)?.uid(), uid);
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
    kill_direct(&mut reserve, &target)?;
    wait_until_empty(&cgroup_path)?;
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
        "registration opened; populated 1 -> cgroup.kill -> empty; exact child reaped; scope removed"
    );
    Ok(())
}
