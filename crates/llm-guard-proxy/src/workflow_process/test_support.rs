use std::{
    cell::Cell,
    fs,
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};

use super::{
    linux_process_is_live, linux_process_start_time,
    signal_authority::{LinuxProcessIdentityProbe as IdentityProbe, probe_linux_process_identity},
};

const MARKER_POLL: Duration = Duration::from_millis(5);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TestProcessIdentity {
    pub(crate) pid: NonZeroU32,
    pub(crate) start_time_ticks: NonZeroU64,
}

impl TestProcessIdentity {
    pub(crate) fn capture(pid: u32) -> Option<Self> {
        Some(Self {
            pid: NonZeroU32::new(pid)?,
            start_time_ticks: NonZeroU64::new(linux_process_start_time(pid).ok()?)?,
        })
    }

    pub(crate) fn parse(contents: &str) -> Option<Self> {
        let mut fields = contents.split_whitespace();
        let identity = Self {
            pid: NonZeroU32::new(fields.next()?.parse().ok()?)?,
            start_time_ticks: NonZeroU64::new(fields.next()?.parse().ok()?)?,
        };
        (fields.next().is_none()).then_some(identity)
    }

    pub(crate) fn read_from(path: &Path) -> Option<Self> {
        Self::parse(&fs::read_to_string(path).ok()?)
    }

    pub(crate) fn publish(self, path: &Path) {
        publish_file_atomically(
            path,
            format!("{} {}", self.pid, self.start_time_ticks).as_bytes(),
        );
    }

    fn probe(self) -> IdentityProbe {
        probe_linux_process_identity(self.pid.get(), self.start_time_ticks.get())
    }

    pub(crate) fn is_current(self) -> bool {
        self.probe() == IdentityProbe::Current
    }

    pub(crate) fn is_live(self) -> bool {
        self.is_current() && linux_process_is_live(self.pid.get())
    }

    pub(crate) fn signal_if_live(self) {
        self.signal_target_if_anchored(false, Instant::now() + Duration::from_secs(1));
    }

    pub(crate) fn signal_group_if_live(self) {
        self.signal_target_if_anchored(true, Instant::now() + Duration::from_secs(1));
    }

    fn signal_target_if_anchored(self, process_group: bool, deadline: Instant) {
        let Ok(raw_pid) = i32::try_from(self.pid.get()) else {
            return;
        };
        loop {
            match self.probe() {
                IdentityProbe::Current => {
                    let target = if process_group { -raw_pid } else { raw_pid };
                    match self.probe() {
                        IdentityProbe::Current => {
                            let _signal = kill(Pid::from_raw(target), Signal::SIGKILL);
                            return;
                        }
                        IdentityProbe::Unavailable if Instant::now() < deadline => {
                            thread::sleep(MARKER_POLL);
                        }
                        IdentityProbe::ConfirmedGoneOrMismatch | IdentityProbe::Unavailable => {
                            return;
                        }
                    }
                }
                IdentityProbe::Unavailable if Instant::now() < deadline => {
                    thread::sleep(MARKER_POLL);
                }
                IdentityProbe::ConfirmedGoneOrMismatch | IdentityProbe::Unavailable => return,
            }
        }
    }

    pub(crate) fn wait_until_not_live(self, deadline: Instant) -> bool {
        loop {
            match self.probe() {
                IdentityProbe::ConfirmedGoneOrMismatch => return true,
                IdentityProbe::Current | IdentityProbe::Unavailable
                    if Instant::now() < deadline =>
                {
                    thread::sleep(MARKER_POLL);
                }
                IdentityProbe::Current | IdentityProbe::Unavailable => return false,
            }
        }
    }
}

pub(crate) struct PublishedProcessCleanup {
    marker_path: PathBuf,
    identity: Cell<Option<TestProcessIdentity>>,
    known_descendant_markers: Vec<PathBuf>,
    cleanup_deadline: Instant,
    armed: Cell<bool>,
    signal_process_group: bool,
}

impl PublishedProcessCleanup {
    pub(crate) fn new(marker_path: PathBuf, cleanup_deadline: Instant) -> Self {
        Self::new_with_signal_scope(marker_path, cleanup_deadline, false, Vec::new())
    }

    pub(crate) fn new_process_group(marker_path: PathBuf, cleanup_deadline: Instant) -> Self {
        Self::new_with_signal_scope(marker_path, cleanup_deadline, true, Vec::new())
    }

    pub(crate) fn new_process_group_with_descendants(
        marker_path: PathBuf,
        cleanup_deadline: Instant,
        known_descendant_markers: Vec<PathBuf>,
    ) -> Self {
        Self::new_with_signal_scope(
            marker_path,
            cleanup_deadline,
            true,
            known_descendant_markers,
        )
    }

    fn new_with_signal_scope(
        marker_path: PathBuf,
        cleanup_deadline: Instant,
        signal_process_group: bool,
        known_descendant_markers: Vec<PathBuf>,
    ) -> Self {
        Self {
            marker_path,
            identity: Cell::new(None),
            known_descendant_markers,
            cleanup_deadline,
            armed: Cell::new(true),
            signal_process_group,
        }
    }

    pub(crate) fn marker_path(&self) -> &Path {
        &self.marker_path
    }

    pub(crate) fn refresh(&self) -> Option<TestProcessIdentity> {
        if let Some(identity) = self.identity.get() {
            return (identity.probe() == IdentityProbe::Current).then_some(identity);
        }
        let identity = TestProcessIdentity::read_from(&self.marker_path)?;
        if identity.probe() != IdentityProbe::Current {
            return None;
        }
        self.identity.set(Some(identity));
        Some(identity)
    }

    pub(crate) fn published_identity(&self) -> Option<TestProcessIdentity> {
        self.identity
            .get()
            .or_else(|| TestProcessIdentity::read_from(&self.marker_path))
    }

    pub(crate) fn wait_for_identity(&self, deadline: Instant) -> Option<TestProcessIdentity> {
        loop {
            if let Some(identity) = self.refresh() {
                return Some(identity);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(MARKER_POLL);
        }
    }

    pub(crate) fn disarm_after_verified_exit(&self) {
        let identity = self
            .published_identity()
            .expect("successful fixture should publish its exact cleanup identity");
        assert!(
            !identity.is_live(),
            "successful fixture should finish exact process cleanup"
        );
        for marker in &self.known_descendant_markers {
            let descendant = TestProcessIdentity::read_from(marker)
                .expect("successful fixture should publish each descendant identity");
            assert!(
                !descendant.is_live(),
                "successful fixture should finish each exact descendant"
            );
        }
        self.armed.set(false);
        self.remove_markers();
    }

    fn remove_markers(&self) {
        remove_atomic_marker(&self.marker_path);
        for marker in &self.known_descendant_markers {
            remove_atomic_marker(marker);
        }
    }
}

impl Drop for PublishedProcessCleanup {
    fn drop(&mut self) {
        if self.armed.get() {
            let identity = self.wait_for_identity(self.cleanup_deadline);
            let descendants = self
                .known_descendant_markers
                .iter()
                .filter_map(|marker| {
                    wait_for_current_published_identity(marker, self.cleanup_deadline)
                })
                .collect::<Vec<_>>();
            if let Some(identity) = identity {
                if self.signal_process_group {
                    identity.signal_target_if_anchored(true, self.cleanup_deadline);
                } else {
                    identity.signal_target_if_anchored(false, self.cleanup_deadline);
                }
            }
            for descendant in &descendants {
                descendant.signal_target_if_anchored(false, self.cleanup_deadline);
            }
            if let Some(identity) = identity {
                let _stopped = identity.wait_until_not_live(self.cleanup_deadline);
            }
            for descendant in descendants {
                let _stopped = descendant.wait_until_not_live(self.cleanup_deadline);
            }
        }
        self.remove_markers();
    }
}

pub(crate) fn read_published_identity(path: &Path) -> Option<TestProcessIdentity> {
    TestProcessIdentity::read_from(path)
}

pub(crate) fn wait_for_published_identity(
    path: &Path,
    deadline: Instant,
) -> Option<TestProcessIdentity> {
    loop {
        if let Some(identity) = read_published_identity(path) {
            return Some(identity);
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(MARKER_POLL);
    }
}

pub(crate) fn wait_for_current_published_identity(
    path: &Path,
    deadline: Instant,
) -> Option<TestProcessIdentity> {
    wait_for_validated_identity_with(
        deadline,
        || read_published_identity(path),
        TestProcessIdentity::probe,
        || thread::sleep(MARKER_POLL),
    )
}

fn wait_for_validated_identity_with<Read, Probe, Sleep>(
    deadline: Instant,
    mut read: Read,
    mut probe: Probe,
    mut sleep: Sleep,
) -> Option<TestProcessIdentity>
where
    Read: FnMut() -> Option<TestProcessIdentity>,
    Probe: FnMut(TestProcessIdentity) -> IdentityProbe,
    Sleep: FnMut(),
{
    loop {
        if let Some(identity) = read()
            && probe(identity) == IdentityProbe::Current
        {
            return Some(identity);
        }
        if Instant::now() >= deadline {
            return None;
        }
        sleep();
    }
}

pub(crate) fn publish_file_atomically(path: &Path, contents: &[u8]) {
    let staging_path = atomic_staging_path(path);
    fs::write(&staging_path, contents).expect("process fixture marker should be staged");
    fs::rename(staging_path, path).expect("process fixture marker should publish atomically");
}

pub(crate) fn remove_atomic_marker(path: &Path) {
    let _removed = fs::remove_file(path);
    let _removed_staging = fs::remove_file(atomic_staging_path(path));
}

pub(crate) fn atomic_staging_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.partial", path.display()))
}

#[test]
fn validated_marker_wait_retries_absent_stale_and_unavailable_observations() {
    let identity = TestProcessIdentity {
        pid: NonZeroU32::new(17).expect("fixture PID is nonzero"),
        start_time_ticks: NonZeroU64::new(23).expect("fixture start time is nonzero"),
    };
    let mut reads = [None, None, Some(identity), Some(identity), Some(identity)].into_iter();
    let mut probes = [
        IdentityProbe::ConfirmedGoneOrMismatch,
        IdentityProbe::Unavailable,
        IdentityProbe::Current,
    ]
    .into_iter();

    let result = wait_for_validated_identity_with(
        Instant::now() + Duration::from_secs(1),
        || reads.next().flatten(),
        |_| {
            probes
                .next()
                .expect("each parseable marker should receive one probe")
        },
        || {},
    );

    assert_eq!(result, Some(identity));
}

#[test]
fn group_cleanup_signals_descendant_while_exact_leader_is_zombie() {
    use std::process::{Command, Stdio};

    let directory = std::env::temp_dir().join(format!(
        "llm-guard-zombie-group-cleanup-{}",
        std::process::id()
    ));
    let _removed = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).expect("zombie group fixture directory should be created");
    let marker = directory.join("descendant.identity");
    let script = format!(
        "sleep 30 & child=$!; start=$(awk '{{print $22}}' /proc/$child/stat); printf '%s %s\\n' \"$child\" \"$start\" > '{}'; exit 0",
        marker.display()
    );
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    super::configure_process_group(&mut command);
    let mut leader = command.spawn().expect("zombie group leader should spawn");
    let leader_identity =
        TestProcessIdentity::capture(leader.id()).expect("leader identity should be captured");
    let deadline = Instant::now() + Duration::from_secs(2);
    let descendant = wait_for_current_published_identity(&marker, deadline)
        .expect("descendant should publish an exact identity");
    while linux_process_is_live(leader_identity.pid.get()) {
        assert!(Instant::now() < deadline, "leader should become a zombie");
        thread::sleep(MARKER_POLL);
    }
    assert!(
        leader_identity.is_current(),
        "zombie must remain an exact anchor"
    );

    leader_identity.signal_group_if_live();
    assert!(
        descendant.wait_until_not_live(deadline),
        "negative-PGID cleanup should kill the known descendant"
    );
    let _reaped = leader.wait();
    remove_atomic_marker(&marker);
    let _removed = fs::remove_dir_all(directory);
}
