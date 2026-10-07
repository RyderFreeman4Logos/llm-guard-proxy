#[path = "support/owned_cgroup.rs"]
mod owned_cgroup;
use owned_cgroup::{
    CgroupObservation, OwnedCgroup, ProcessIdentity, cgroup_members, populated, process_command,
    process_identity, process_identity_from, random_id, wait_for_exit, wait_until_populated,
};

use llm_guard_proxy_host_guardian::{CgroupTarget, EmergencyReserve, kill_direct};
use nix::{
    fcntl::{FcntlArg, OFlag, fcntl},
    poll::{PollFd, PollFlags, PollTimeout, poll},
    sys::{
        prctl,
        signal::{Signal, kill, killpg},
        wait::{WaitPidFlag, WaitStatus, waitpid},
    },
    unistd::{Pid, Uid},
};
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsFd, AsRawFd},
        unix::{
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    sync::mpsc::sync_channel,
    thread,
    time::{Duration, Instant},
};

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const CHILD_SECONDS: &str = "120";
const DEADLINE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);
const CHILD_REAP_GRACE: Duration = Duration::from_millis(250);
const SYSTEMCTL_OUTPUT_LIMIT: usize = 4096;

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

fn wait_until_empty(path: &Path) -> io::Result<()> {
    CgroupObservation::open(path)?.wait_until_empty()
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

#[test]
fn systemd_property_deadline_cleans_retained_stdout_descendant() {
    const WORKER: &str = "LLM_GUARD_RETAINED_STDOUT_WORKER";
    if std::env::var_os(WORKER).is_none() {
        let mut worker = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "systemd_property_deadline_cleans_retained_stdout_descendant",
                "--nocapture",
            ])
            .env(WORKER, "1")
            .spawn()
            .expect("isolated fixture worker");
        let result = wait_for_exit(&mut worker, Duration::from_secs(15));
        if result.is_err() {
            let _ = worker.kill();
            let _ = wait_for_exit(&mut worker, CHILD_REAP_GRACE);
        }
        assert!(
            result.is_ok_and(|status| status.success()),
            "isolated fixture failed"
        );
        return;
    }
    prctl::set_child_subreaper(true).expect("isolated worker owns adopted descendant");
    let scratch = Scratch::create("systemd-retained-stdout").expect("scratch directory");
    let descendant = OwnedDescendant(scratch.0.join("descendant.stat"));
    let fake_systemctl = scratch.0.join("systemctl");
    fs::write(
        &fake_systemctl,
        format!(
            "#!/bin/sh\n/usr/bin/sleep 120 &\n/usr/bin/cat /proc/$!/stat > '{}'\nexit 0\n",
            descendant.0.display()
        ),
    )
    .expect("fake systemctl should be written");
    fs::set_permissions(&fake_systemctl, fs::Permissions::from_mode(0o700))
        .expect("fake systemctl should be executable");
    let started = Instant::now();
    let result = systemd_property_with(
        &fake_systemctl,
        "retained.stdout.scope",
        "ControlGroup",
        Duration::from_millis(500),
    );
    let elapsed = started.elapsed();
    let identity = descendant
        .identity()
        .expect("published descendant identity");
    let pid = Pid::from_raw(i32::try_from(identity.pid).expect("valid descendant PID"));
    let at_return = process_identity(identity.pid);
    let immediate_status = waitpid(pid, Some(WaitPidFlag::WNOHANG));
    let status = match immediate_status {
        Ok(WaitStatus::StillAlive) => OwnedDescendant::reap(pid, CHILD_REAP_GRACE),
        result => result.map_err(io::Error::other),
    };
    // Capture the oracle before emergency Drop cleanup; cleanup cannot turn RED into GREEN.
    println!(
        "retained-stdout helper_return identity={identity:?} observed={at_return:?} immediate_status={immediate_status:?} status={status:?} elapsed={elapsed:?}"
    );
    assert_eq!(
        at_return.expect("owned descendant remains waitable"),
        identity
    );
    assert_eq!(
        status.expect("exact descendant must be reaped"),
        WaitStatus::Signaled(pid, Signal::SIGKILL, false)
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "query took {elapsed:?}"
    );
    assert_eq!(
        result.expect_err("retained stdout must time out").kind(),
        io::ErrorKind::TimedOut
    );
    assert_eq!(
        process_identity(identity.pid)
            .expect_err("descendant reaped")
            .kind(),
        io::ErrorKind::NotFound
    );
}

struct OwnedDescendant(PathBuf);

impl OwnedDescendant {
    fn identity(&self) -> io::Result<ProcessIdentity> {
        let stat = fs::read_to_string(&self.0)?;
        let pid = stat
            .split_whitespace()
            .next()
            .ok_or_else(|| io::Error::other("missing descendant pid"))?
            .parse::<u32>()
            .map_err(io::Error::other)?;
        process_identity_from(pid, &stat)
    }

    fn reap(pid: Pid, timeout: Duration) -> io::Result<WaitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::StillAlive) if Instant::now() < deadline => thread::sleep(POLL),
                Ok(WaitStatus::StillAlive) => {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "descendant still alive",
                    ));
                }
                Ok(status) => return Ok(status),
                Err(nix::errno::Errno::EINTR) if Instant::now() < deadline => {}
                Err(error) => return Err(io::Error::other(error)),
            }
        }
    }
}

impl Drop for OwnedDescendant {
    fn drop(&mut self) {
        if let Ok(identity) = self.identity()
            && process_identity(identity.pid).ok() == Some(identity)
            && let Ok(pid) = i32::try_from(identity.pid)
        {
            let pid = Pid::from_raw(pid);
            let _ = kill(pid, Signal::SIGKILL);
            let _ = Self::reap(pid, CHILD_REAP_GRACE);
        }
    }
}

fn systemd_property(unit: &str, property: &str) -> io::Result<String> {
    systemd_property_with(Path::new("/usr/bin/systemctl"), unit, property, DEADLINE)
}

fn read_systemd_stdout(stdout: &mut ChildStdout, deadline: Instant) -> io::Result<Vec<u8>> {
    let flags = fcntl(stdout.as_raw_fd(), FcntlArg::F_GETFL).map_err(io::Error::other)?;
    fcntl(
        stdout.as_raw_fd(),
        FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
    )
    .map_err(io::Error::other)?;

    let mut output = Vec::new();
    let mut buffer = [0_u8; SYSTEMCTL_OUTPUT_LIMIT + 1];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "systemctl stdout did not close before deadline",
            ));
        }
        let mut descriptor = [PollFd::new(
            stdout.as_fd(),
            PollFlags::POLLIN | PollFlags::POLLHUP,
        )];
        let ready = match poll(
            &mut descriptor,
            PollTimeout::try_from(remaining).unwrap_or(PollTimeout::MAX),
        ) {
            Ok(ready) => ready,
            Err(nix::errno::Errno::EINTR) => continue,
            Err(error) => return Err(io::Error::other(error)),
        };
        if ready == 0 {
            continue;
        }
        let events = descriptor[0].revents().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown systemctl stdout poll flags",
            )
        })?;
        if events.contains(PollFlags::POLLNVAL) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "systemctl stdout descriptor is invalid",
            ));
        }
        match stdout.read(&mut buffer[..SYSTEMCTL_OUTPUT_LIMIT + 1 - output.len()]) {
            Ok(0) => return Ok(output),
            Ok(length) => {
                output.extend_from_slice(&buffer[..length]);
                if output.len() > SYSTEMCTL_OUTPUT_LIMIT {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "systemctl output exceeded limit",
                    ));
                }
            }
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

fn terminate_owned_systemctl(
    child: &mut Child,
    identity: ProcessIdentity,
    deadline: Instant,
) -> io::Result<()> {
    let group_kill = match process_identity(identity.pid) {
        Ok(current)
            if current == identity && identity.pgid == identity.pid && identity.pgid > 1 =>
        {
            i32::try_from(identity.pgid)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
                .and_then(|pgid| match killpg(Pid::from_raw(pgid), Signal::SIGKILL) {
                    Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
                    Err(error) => Err(io::Error::other(error)),
                })
        }
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "owned systemctl process identity changed",
        )),
        Err(error) => Err(error),
    };
    let _ = child.kill();
    wait_for_exit(child, deadline.saturating_duration_since(Instant::now()))?;
    group_kill
}

fn systemd_property_with(
    systemctl: &Path,
    unit: &str,
    property: &str,
    timeout: Duration,
) -> io::Result<String> {
    let started = Instant::now();
    let deadline = started + timeout;
    let work_deadline = started + timeout.saturating_sub(CHILD_REAP_GRACE);
    let mut command = Command::new(systemctl);
    command
        .args(["--user", "show", unit, "--property", property, "--value"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0);
    let mut child = command.spawn()?;
    let identity = match process_identity(child.id()) {
        Ok(identity) if identity.pgid == child.id() => identity,
        Ok(_) => {
            let error = io::Error::new(
                io::ErrorKind::InvalidData,
                "systemctl did not enter its owned process group",
            );
            let _ = child.kill();
            wait_for_exit(
                &mut child,
                deadline.saturating_duration_since(Instant::now()),
            )?;
            return Err(error);
        }
        Err(error) => {
            let _ = child.kill();
            wait_for_exit(
                &mut child,
                deadline.saturating_duration_since(Instant::now()),
            )?;
            return Err(error);
        }
    };
    let Some(mut stdout) = child.stdout.take() else {
        let error = io::Error::other("systemctl stdout was not captured");
        terminate_owned_systemctl(&mut child, identity, deadline)?;
        return Err(error);
    };
    let output = read_systemd_stdout(&mut stdout, work_deadline);
    drop(stdout);
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            terminate_owned_systemctl(&mut child, identity, deadline)?;
            return Err(error);
        }
    };
    let status = match wait_for_exit(
        &mut child,
        work_deadline.saturating_duration_since(Instant::now()),
    ) {
        Ok(status) => status,
        Err(error) => {
            terminate_owned_systemctl(&mut child, identity, deadline)?;
            return Err(error);
        }
    };
    if !status.success() {
        return Err(io::Error::other("systemctl show failed"));
    }
    Ok(String::from_utf8_lossy(&output).trim().to_owned())
}

#[test]
#[ignore = "requires writable user-systemd delegated app.slice; run just test-real-cgroup"]
fn registered_cgroup_kill_reaps_only_the_task_owned_child() -> Result<(), Box<dyn std::error::Error>>
{
    let id = random_id()?;
    let uid = Uid::effective().as_raw();
    let unit = format!("docker-{id}.scope");
    let control_group = format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}");
    let cgroup_path = Path::new(CGROUP_ROOT).join(control_group.trim_start_matches('/'));
    assert_eq!(systemd_property(&unit, "LoadState")?, "not-found");

    // Use the user's delegated app.slice, but do not ask systemd to own this leaf.
    // Keeping collection in this parent permits an arbitrarily delayed empty observer.
    let mut owned = OwnedCgroup::create(cgroup_path.clone())?;
    let child = owned.child.as_mut().expect("owned child was spawned");
    let identity = wait_until_populated(&cgroup_path, child)?;
    assert_eq!(identity.pid, child.id());
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
        "kernel-cgroup fixture name={unit} launcher_pid={} target_pid={} pgid={} starttime={} populated_before=1",
        identity.pid, identity.pid, identity.pgid, identity.starttime
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

    let target = CgroupTarget::open_registered(&registration, Path::new(CGROUP_ROOT))?;
    assert_eq!(cgroup_members(&cgroup_path)?, [identity.pid]);
    assert!(populated(&cgroup_path)?);
    let mut reserve = EmergencyReserve::with_page_size(4096, 4096)?;
    let (observer_ready, observer_started) = sync_channel(1);
    let (observer_release, observer_released) = sync_channel(1);
    thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let observation_for_waiter = &observation;
        let observer = scope.spawn(move || {
            assert!(observation_for_waiter.is_populated()?);
            observer_ready.send(()).map_err(io::Error::other)?;
            observer_released
                .recv_timeout(DEADLINE)
                .map_err(io::Error::other)?;
            observation_for_waiter.wait_until_empty()
        });
        observer_started.recv_timeout(DEADLINE).map_err(|error| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("empty-state observer did not start: {error}"),
            )
        })?;
        kill_direct(&mut reserve, &target)?;
        println!("real-cgroup fixture cgroup.kill issued");
        let status = wait_for_exit(owned.child.as_mut().expect("owned child"), DEADLINE)?;
        assert_eq!(
            std::os::unix::process::ExitStatusExt::signal(&status),
            Some(9)
        );
        assert!(
            cgroup_path.exists(),
            "parent must retain empty cgroup until observer joins"
        );
        observer_release.send(()).map_err(io::Error::other)?;
        observer
            .join()
            .map_err(|_| io::Error::other("empty-state observer panicked"))??;
        Ok(())
    })?;
    drop(target);

    match process_identity(identity.pid) {
        Ok(current) => assert_ne!(current.starttime, identity.starttime),
        Err(error) => assert_eq!(error.kind(), io::ErrorKind::NotFound),
    }
    owned.remove()?;
    assert_eq!(
        observation
            .is_populated()
            .expect_err("removed kernfs node must not mean empty")
            .raw_os_error(),
        Some(nix::errno::Errno::ENODEV as i32)
    );
    drop(observation);
    wait_until_removed(&cgroup_path)?;
    assert_eq!(systemd_property(&unit, "LoadState")?, "not-found");
    let scratch_path = scratch.0.clone();
    drop(scratch);
    assert!(!scratch_path.exists());
    println!(
        "registration opened; populated=1 -> cgroup.kill -> observed populated=0 and empty cgroup.procs; exact child reaped; parent-owned kernel cgroup removed (not a systemd unit)"
    );
    Ok(())
}
