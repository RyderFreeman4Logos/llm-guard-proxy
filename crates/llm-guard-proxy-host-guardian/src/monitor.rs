//! The allocation-aware Tier-1 memory monitoring loop.

use crate::{
    config::{ConfigError, GuardianConfig, Thresholds},
    escalation::EscalationEpisode,
    hot_reload::HotReloadableConfig,
};
use nix::unistd::Uid;
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    os::unix::{fs::FileExt, fs::MetadataExt, fs::OpenOptionsExt},
    path::{Path, PathBuf},
    time::Instant,
};
use thiserror::Error;

const MEMINFO_BUFFER_BYTES: usize = 8 * 1024;
const REGISTRATION_MAX_BYTES: usize = 1024;
const EVENTS_BUFFER_BYTES: usize = 512;

/// Parses the exact `MemAvailable: <integer> kB` field from `/proc/meminfo`.
///
/// # Errors
///
/// Returns [`MemInfoError`] when the field is absent, duplicated, malformed,
/// or cannot be converted to bytes.
pub fn parse_mem_available(meminfo: &[u8]) -> Result<u64, MemInfoError> {
    let mut found = None;
    for line in meminfo.split(|byte| *byte == b'\n') {
        let Some(rest) = line.strip_prefix(b"MemAvailable:") else {
            continue;
        };
        if found.is_some() {
            return Err(MemInfoError::Duplicate);
        }
        let rest = trim_ascii(rest);
        let Some(split) = rest.iter().position(u8::is_ascii_whitespace) else {
            return Err(MemInfoError::Malformed);
        };
        let (digits, unit) = rest.split_at(split);
        if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) || trim_ascii(unit) != b"kB"
        {
            return Err(MemInfoError::Malformed);
        }
        let mut kib = 0_u64;
        for digit in digits {
            kib = kib
                .checked_mul(10)
                .and_then(|value| value.checked_add(u64::from(*digit - b'0')))
                .ok_or(MemInfoError::Overflow)?;
        }
        found = Some(kib.checked_mul(1024).ok_or(MemInfoError::Overflow)?);
    }
    found.ok_or(MemInfoError::Missing)
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

/// Returns whether Tier 1 must shed the registered cgroup.
#[must_use]
pub const fn should_shed(mem_available_bytes: u64, thresholds: Thresholds) -> bool {
    mem_available_bytes < thresholds.threshold_bytes()
}

/// Returns whether the reserve may be reallocated and the latch cleared.
#[must_use]
pub const fn should_rearm(mem_available_bytes: u64, thresholds: Thresholds) -> bool {
    mem_available_bytes
        >= thresholds
            .threshold_bytes()
            .saturating_add(thresholds.reserve_bytes() as u64)
}

/// Errors decoding the kernel-provided memory information.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MemInfoError {
    /// `MemAvailable` was not present.
    #[error("MemAvailable is missing")]
    Missing,
    /// The field appeared more than once.
    #[error("MemAvailable appeared more than once")]
    Duplicate,
    /// The field was not an exact integer-kB value.
    #[error("MemAvailable is malformed")]
    Malformed,
    /// Unit conversion overflowed `u64`.
    #[error("MemAvailable overflows bytes")]
    Overflow,
}

/// A fully validated registration record published by the protected workload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Registration {
    /// The 64-character lower-case hex container identifier.
    pub container_id: String,
    /// The matching systemd scope name.
    pub scope: String,
    /// The matching rootless cgroup-v2 path.
    pub control_group: String,
}

/// Registration content failures fail closed before a cgroup descriptor is opened.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RegistrationError {
    /// The file was not UTF-8.
    #[error("registration is not UTF-8")]
    InvalidUtf8,
    /// A line was malformed.
    #[error("registration contains a malformed line")]
    MalformedLine,
    /// An unrecognized field was present.
    #[error("registration contains an unknown field")]
    UnknownField,
    /// A required field was absent.
    #[error("registration is missing a required field")]
    MissingField,
    /// A field was repeated.
    #[error("registration contains a duplicate field")]
    DuplicateField,
    /// Registration versions are not compatible.
    #[error("registration version is invalid")]
    WrongVersion,
    /// The container id is not lower-case hexadecimal.
    #[error("registration has an invalid container id")]
    InvalidContainerId,
    /// The scope does not match the container id.
    #[error("registration has an invalid scope")]
    InvalidScope,
    /// The cgroup path does not match the current effective user and scope.
    #[error("registration has an invalid control group")]
    InvalidControlGroup,
}

/// Parses the stable, line-oriented rootless Docker registration format.
///
/// # Errors
///
/// Returns [`RegistrationError`] when any field is absent, repeated, malformed,
/// or inconsistent with `uid`.
pub fn parse_registration(input: &[u8], uid: u32) -> Result<Registration, RegistrationError> {
    let text = std::str::from_utf8(input).map_err(|_error| RegistrationError::InvalidUtf8)?;
    let mut version = None;
    let mut container_id = None;
    let mut scope = None;
    let mut control_group = None;
    for line in text.split_terminator('\n') {
        if line.is_empty() || line.contains('\r') || line.contains('\0') {
            return Err(RegistrationError::MalformedLine);
        }
        let (key, value) = line
            .split_once('=')
            .ok_or(RegistrationError::MalformedLine)?;
        if value.is_empty() || value.contains('=') {
            return Err(RegistrationError::MalformedLine);
        }
        let slot = match key {
            "version" => &mut version,
            "container_id" => &mut container_id,
            "scope" => &mut scope,
            "control_group" => &mut control_group,
            _ => return Err(RegistrationError::UnknownField),
        };
        if slot.replace(value).is_some() {
            return Err(RegistrationError::DuplicateField);
        }
    }
    if version != Some("1") {
        return Err(if version.is_some() {
            RegistrationError::WrongVersion
        } else {
            RegistrationError::MissingField
        });
    }
    let container_id = container_id.ok_or(RegistrationError::MissingField)?;
    if container_id.len() != 64
        || !container_id
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(RegistrationError::InvalidContainerId);
    }
    let scope = scope.ok_or(RegistrationError::MissingField)?;
    let expected_scope = format!("docker-{container_id}.scope");
    if scope != expected_scope {
        return Err(RegistrationError::InvalidScope);
    }
    let control_group = control_group.ok_or(RegistrationError::MissingField)?;
    let expected_group =
        format!("/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}");
    if control_group != expected_group {
        return Err(RegistrationError::InvalidControlGroup);
    }
    Ok(Registration {
        container_id: container_id.to_owned(),
        scope: scope.to_owned(),
        control_group: control_group.to_owned(),
    })
}

/// Owns pre-opened descriptors used by the allocation-free Tier-1 action.
#[derive(Debug)]
pub struct CgroupTarget {
    kill: File,
    events: File,
}

impl CgroupTarget {
    fn from_registration(
        registration_path: &Path,
        cgroup_root: &Path,
        expected_uid: u32,
    ) -> Result<Self, GuardianError> {
        let mut registration_file = OpenOptions::new()
            .read(true)
            .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits())
            .open(registration_path)
            .map_err(|source| GuardianError::Io {
                operation: "open registration",
                source,
            })?;
        validate_registration_metadata(&registration_file, expected_uid)?;
        let mut bytes = [0_u8; REGISTRATION_MAX_BYTES];
        let length = registration_file
            .read(&mut bytes)
            .map_err(|source| GuardianError::Io {
                operation: "read registration",
                source,
            })?;
        if length == bytes.len() {
            return Err(GuardianError::InvalidRegistration(String::from(
                "registration exceeds fixed buffer",
            )));
        }
        let registration = parse_registration(&bytes[..length], expected_uid)
            .map_err(|error| GuardianError::InvalidRegistration(error.to_string()))?;
        let cgroup_path = cgroup_root.join(registration.control_group.trim_start_matches('/'));
        let kill = OpenOptions::new()
            .write(true)
            .custom_flags(0)
            .open(cgroup_path.join("cgroup.kill"))
            .map_err(|source| GuardianError::Io {
                operation: "open cgroup.kill",
                source,
            })?;
        let events =
            File::open(cgroup_path.join("cgroup.events")).map_err(|source| GuardianError::Io {
                operation: "open cgroup.events",
                source,
            })?;
        let target = Self { kill, events };
        if !target.is_populated().map_err(|source| GuardianError::Io {
            operation: "read cgroup.events while arming",
            source,
        })? {
            return Err(GuardianError::InvalidRegistration(String::from(
                "registered cgroup is not populated",
            )));
        }
        Ok(target)
    }

    /// Releases the reserve and writes one byte to the already-open kill fd.
    ///
    /// This is the emergency path: it never opens files, parses strings, spawns
    /// processes, sends IPC, or allocates. `Write::write_all` handles EINTR.
    ///
    /// # Errors
    ///
    /// Returns the kernel write error from the already-open `cgroup.kill` fd.
    pub fn kill_direct(&mut self, reserve: &mut Option<Vec<u8>>) -> Result<(), io::Error> {
        drop(reserve.take());
        self.kill.write_all(b"1")
    }

    fn is_empty(&self) -> Result<bool, io::Error> {
        Ok(!self.is_populated()?)
    }

    fn is_populated(&self) -> Result<bool, io::Error> {
        let mut bytes = [0_u8; EVENTS_BUFFER_BYTES];
        let length = self.events.read_at(&mut bytes, 0)?;
        let text = std::str::from_utf8(&bytes[..length]).map_err(|_error| {
            io::Error::new(io::ErrorKind::InvalidData, "cgroup.events is not UTF-8")
        })?;
        match text.lines().map(str::trim).find_map(|line| match line {
            "populated 0" => Some(false),
            "populated 1" => Some(true),
            _ => None,
        }) {
            Some(populated) => Ok(populated),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cgroup.events does not contain populated state",
            )),
        }
    }
}

fn validate_registration_metadata(file: &File, expected_uid: u32) -> Result<(), GuardianError> {
    let metadata = file.metadata().map_err(|source| GuardianError::Io {
        operation: "stat registration",
        source,
    })?;
    if !metadata.file_type().is_file()
        || metadata.uid() != expected_uid
        || metadata.nlink() != 1
        || metadata.mode() & 0o7777 != 0o600
    {
        return Err(GuardianError::InvalidRegistration(String::from(
            "registration must be a single-link 0600 regular file owned by the effective user",
        )));
    }
    Ok(())
}

/// Status reported by a single guardian iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardianIteration {
    /// Healthy memory; no action was required.
    Healthy,
    /// The registration or cgroup target is temporarily unavailable; Tier 1 is
    /// disarmed until a later healthy iteration validates and opens a target.
    Unarmed,
    /// The cgroup.kill fast path completed and the guardian is now latched.
    Shed,
    /// The target has not yet become empty after Tier 1.
    Waiting,
    /// The target became empty after Tier 1, but memory has not rearmed yet.
    Verified,
    /// Memory recovered and a new reserve was allocated.
    Rearmed,
}

/// Guardian runtime failures.
#[derive(Debug, Error)]
pub enum GuardianError {
    /// Configuration load or hot-reload failure.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Tier-2 was explicitly enabled but failed after the Tier-1 action.
    #[error(transparent)]
    Escalation(#[from] crate::escalation::EscalationError),
    /// A filesystem operation failed.
    #[error("{operation}: {source}")]
    Io {
        operation: &'static str,
        source: io::Error,
    },
    /// A registration was syntactically or operationally invalid.
    #[error("invalid guardian registration: {0}")]
    InvalidRegistration(String),
    /// No validated target is currently armed.
    #[error("no guardian target is armed")]
    NoTarget,
}

/// Runs Tier 1 continuously and owns all state which may allocate while healthy.
#[derive(Debug)]
pub struct MemoryGuardian {
    reloader: HotReloadableConfig,
    runtime_dir: PathBuf,
    registration_path: PathBuf,
    cgroup_root: PathBuf,
    target: Option<CgroupTarget>,
    proc_meminfo: File,
    reserve: Option<Vec<u8>>,
    latched: bool,
    verified: bool,
    escalation: EscalationEpisode,
}

impl MemoryGuardian {
    /// Arms a guardian using registrations from `runtime_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error when the config, registration, cgroup descriptors, or
    /// `/proc/meminfo` cannot be safely opened.
    pub fn open(
        config_path: impl Into<PathBuf>,
        runtime_dir: impl Into<PathBuf>,
    ) -> Result<Self, GuardianError> {
        let reloader = HotReloadableConfig::new(config_path.into())?;
        let runtime_dir = runtime_dir.into();
        let config = reloader.current();
        let registration_path = registration_path(&config, &runtime_dir);
        let cgroup_root = config.runtime().cgroup_root().to_path_buf();
        let target = open_target_at(&registration_path, &cgroup_root)?;
        let reserve = allocate_reserve(config.thresholds().reserve_bytes());
        let proc_meminfo = File::open("/proc/meminfo").map_err(|source| GuardianError::Io {
            operation: "open /proc/meminfo",
            source,
        })?;
        Ok(Self {
            reloader,
            runtime_dir,
            registration_path,
            cgroup_root,
            target: Some(target),
            proc_meminfo,
            reserve: Some(reserve),
            latched: false,
            verified: false,
            escalation: EscalationEpisode::default(),
        })
    }

    /// Returns whether the emergency path is currently latched.
    #[must_use]
    pub const fn is_latched(&self) -> bool {
        self.latched
    }

    /// Runs one loop iteration. Invoke only from the guardian task.
    ///
    /// # Errors
    ///
    /// Returns errors from the healthy reload/arming path, the pre-opened
    /// kernel descriptors, or explicitly enabled Tier-2 escalation.
    pub fn tick(&mut self) -> Result<GuardianIteration, GuardianError> {
        if self.latched {
            return self.latched_iteration();
        }
        self.healthy_iteration()
    }

    /// Runs until the supplied shutdown future resolves.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::tick`] or optional Tier-2 escalation.
    pub async fn run_until<F>(&mut self, shutdown: F) -> Result<(), GuardianError>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(shutdown);
        loop {
            let iteration = self.tick()?;
            let config = self.reloader.current();
            let delay = if self.latched {
                config.runtime().retry_interval()
            } else {
                config.runtime().poll_interval()
            };
            if matches!(iteration, GuardianIteration::Shed) {
                self.escalation.arm(config.escalation(), Instant::now());
            }
            if self.latched {
                self.escalation
                    .maybe_run(config.escalation(), Instant::now())?;
            }
            tokio::select! {
                () = &mut shutdown => return Ok(()),
                () = tokio::time::sleep(delay) => {}
            }
        }
    }

    fn healthy_iteration(&mut self) -> Result<GuardianIteration, GuardianError> {
        let available = read_mem_available(&self.proc_meminfo)?;
        let thresholds = self.reloader.thresholds();
        if should_shed(available, thresholds) {
            self.target
                .as_mut()
                .ok_or(GuardianError::NoTarget)?
                .kill_direct(&mut self.reserve)
                .map_err(|source| GuardianError::Io {
                    operation: "write cgroup.kill",
                    source,
                })?;
            self.latched = true;
            self.verified = false;
            return Ok(GuardianIteration::Shed);
        }
        // A malformed replacement must retain and keep enforcing the last-good
        // snapshot. The watcher reports the error once, but cannot turn a live
        // guardian into an unprotected stopped process.
        let _reload_error = self.reloader.reload_if_changed().err();
        let config = self.reloader.current();
        self.update_target_location(&config);
        match self.open_configured_target() {
            Ok(target) => self.target = Some(target),
            Err(_error) => {
                self.target = None;
                return Ok(GuardianIteration::Unarmed);
            }
        }
        Ok(GuardianIteration::Healthy)
    }

    fn latched_iteration(&mut self) -> Result<GuardianIteration, GuardianError> {
        let available = read_mem_available(&self.proc_meminfo)?;
        let thresholds = self.reloader.thresholds();
        if should_rearm(available, thresholds) {
            self.reserve = Some(allocate_reserve(thresholds.reserve_bytes()));
            self.latched = false;
            self.verified = false;
            self.escalation.clear();
            return Ok(GuardianIteration::Rearmed);
        }
        self.verified = self
            .target
            .as_ref()
            .ok_or(GuardianError::NoTarget)?
            .is_empty()
            .map_err(|source| GuardianError::Io {
                operation: "read cgroup.events",
                source,
            })?;
        if !self.verified {
            self.target
                .as_mut()
                .ok_or(GuardianError::NoTarget)?
                .kill_direct(&mut self.reserve)
                .map_err(|source| GuardianError::Io {
                    operation: "retry cgroup.kill",
                    source,
                })?;
            return Ok(GuardianIteration::Waiting);
        }
        if should_shed(available, thresholds) {
            // A terminated container may be recreated while the host remains
            // under pressure. This slow path is intentionally outside
            // `kill_direct`: it reopens and validates a fresh descriptor only
            // after the old cgroup has become empty.
            if let Ok(replacement) = self.open_configured_target() {
                self.target = Some(replacement);
                self.target
                    .as_mut()
                    .ok_or(GuardianError::NoTarget)?
                    .kill_direct(&mut self.reserve)
                    .map_err(|source| GuardianError::Io {
                        operation: "write replacement cgroup.kill",
                        source,
                    })?;
                self.verified = false;
                return Ok(GuardianIteration::Shed);
            }
        }
        Ok(if self.verified {
            GuardianIteration::Verified
        } else {
            GuardianIteration::Waiting
        })
    }

    fn update_target_location(&mut self, config: &GuardianConfig) {
        self.registration_path = registration_path(config, &self.runtime_dir);
        self.cgroup_root = config.runtime().cgroup_root().to_path_buf();
    }

    fn open_configured_target(&self) -> Result<CgroupTarget, GuardianError> {
        open_target_at(&self.registration_path, &self.cgroup_root)
    }
}

fn registration_path(config: &GuardianConfig, runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(config.target().registration_file())
}

fn open_target_at(registration: &Path, cgroup_root: &Path) -> Result<CgroupTarget, GuardianError> {
    let uid = Uid::effective().as_raw();
    CgroupTarget::from_registration(registration, cgroup_root, uid)
}

fn read_mem_available(file: &File) -> Result<u64, GuardianError> {
    let mut bytes = [0_u8; MEMINFO_BUFFER_BYTES];
    let length = file
        .read_at(&mut bytes, 0)
        .map_err(|source| GuardianError::Io {
            operation: "read /proc/meminfo",
            source,
        })?;
    if length == bytes.len() {
        return Err(GuardianError::InvalidRegistration(String::from(
            "/proc/meminfo exceeds fixed buffer",
        )));
    }
    parse_mem_available(&bytes[..length]).map_err(|error| {
        GuardianError::InvalidRegistration(format!("invalid /proc/meminfo: {error}"))
    })
}

fn allocate_reserve(bytes: usize) -> Vec<u8> {
    let mut reserve = vec![0_u8; bytes];
    let page_size = 4096;
    for offset in (0..reserve.len()).step_by(page_size) {
        reserve[offset] = 1;
    }
    reserve
}

#[cfg(test)]
mod tests {
    use super::{
        CgroupTarget, MemInfoError, parse_mem_available, parse_registration, should_rearm,
        should_shed,
    };
    use crate::Thresholds;
    use nix::unistd::Uid;
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    const UID: u32 = 1001;
    const ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn registration() -> String {
        format!(
            "version=1\ncontainer_id={ID}\nscope=docker-{ID}.scope\ncontrol_group=/user.slice/user-{UID}.slice/user@{UID}.service/app.slice/docker-{ID}.scope\n"
        )
    }

    fn temporary_tree() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("guardian-target-{nonce}"));
        fs::create_dir_all(&root).expect("create root");
        root
    }

    fn target_tree() -> (PathBuf, PathBuf) {
        let root = temporary_tree();
        let uid = Uid::effective().as_raw();
        let id = "a".repeat(64);
        let scope = format!("docker-{id}.scope");
        let cgroup = root.join(format!(
            "user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}"
        ));
        fs::create_dir_all(&cgroup).expect("create cgroup");
        fs::write(cgroup.join("cgroup.kill"), b"").expect("create kill");
        fs::write(cgroup.join("cgroup.events"), b"populated 1\n").expect("create events");
        let runtime = root.join("runtime");
        fs::create_dir(&runtime).expect("create runtime");
        let registration = format!(
            "version=1\ncontainer_id={id}\nscope={scope}\ncontrol_group=/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}\n"
        );
        let registration_path = runtime.join("target.v1");
        fs::write(&registration_path, registration).expect("write registration");
        fs::set_permissions(&registration_path, fs::Permissions::from_mode(0o600))
            .expect("secure registration");
        (root, registration_path)
    }

    #[test]
    fn parses_mem_available() {
        assert_eq!(
            parse_mem_available(b"MemTotal: 1 kB\nMemAvailable: 42 kB\n"),
            Ok(43_008)
        );
    }

    #[test]
    fn ignores_similarly_named_meminfo_fields() {
        assert_eq!(
            parse_mem_available(b"NotMemAvailable: 5 kB\nMemAvailable: 1 kB\n"),
            Ok(1024)
        );
    }

    #[test]
    fn rejects_missing_mem_available() {
        assert_eq!(
            parse_mem_available(b"MemFree: 1 kB\n"),
            Err(MemInfoError::Missing)
        );
    }

    #[test]
    fn rejects_duplicate_mem_available() {
        assert_eq!(
            parse_mem_available(b"MemAvailable: 1 kB\nMemAvailable: 2 kB\n"),
            Err(MemInfoError::Duplicate)
        );
    }

    #[test]
    fn rejects_bad_unit() {
        assert_eq!(
            parse_mem_available(b"MemAvailable: 1 MB\n"),
            Err(MemInfoError::Malformed)
        );
    }

    #[test]
    fn rejects_missing_unit() {
        assert_eq!(
            parse_mem_available(b"MemAvailable: 1\n"),
            Err(MemInfoError::Malformed)
        );
    }

    #[test]
    fn rejects_non_numeric_value() {
        assert_eq!(
            parse_mem_available(b"MemAvailable: one kB\n"),
            Err(MemInfoError::Malformed)
        );
    }

    #[test]
    fn rejects_overflow() {
        assert_eq!(
            parse_mem_available(b"MemAvailable: 18446744073709551615 kB\n"),
            Err(MemInfoError::Overflow)
        );
    }

    #[test]
    fn sheds_only_below_threshold() {
        let thresholds = Thresholds::new(1, 64).expect("thresholds");
        assert!(should_shed(thresholds.threshold_bytes() - 1, thresholds));
        assert!(!should_shed(thresholds.threshold_bytes(), thresholds));
    }

    #[test]
    fn rearms_only_after_reserve_headroom() {
        let thresholds = Thresholds::new(1, 64).expect("thresholds");
        assert!(!should_rearm(thresholds.threshold_bytes(), thresholds));
        assert!(should_rearm(
            thresholds.threshold_bytes() + thresholds.reserve_bytes() as u64,
            thresholds
        ));
    }

    #[test]
    fn parses_valid_registration() {
        let parsed = parse_registration(registration().as_bytes(), UID).expect("registration");
        assert_eq!(parsed.container_id, ID);
    }

    #[test]
    fn rejects_registration_with_missing_version() {
        let error = parse_registration(registration().replace("version=1\n", "").as_bytes(), UID)
            .expect_err("missing version");
        assert_eq!(error, super::RegistrationError::MissingField);
    }

    #[test]
    fn rejects_registration_with_wrong_version() {
        let error = parse_registration(
            registration().replace("version=1", "version=2").as_bytes(),
            UID,
        )
        .expect_err("wrong version");
        assert_eq!(error, super::RegistrationError::WrongVersion);
    }

    #[test]
    fn rejects_registration_with_unknown_field() {
        let error = parse_registration(format!("{}extra=x\n", registration()).as_bytes(), UID)
            .expect_err("unknown field");
        assert_eq!(error, super::RegistrationError::UnknownField);
    }

    #[test]
    fn rejects_registration_with_duplicate_field() {
        let error = parse_registration(format!("{}scope=x\n", registration()).as_bytes(), UID)
            .expect_err("duplicate field");
        assert_eq!(error, super::RegistrationError::DuplicateField);
    }

    #[test]
    fn rejects_uppercase_container_id() {
        let error = parse_registration(
            registration().replace(ID, &ID.to_uppercase()).as_bytes(),
            UID,
        )
        .expect_err("invalid id");
        assert_eq!(error, super::RegistrationError::InvalidContainerId);
    }

    #[test]
    fn rejects_mismatched_scope() {
        let error =
            parse_registration(registration().replace("docker-", "podman-").as_bytes(), UID)
                .expect_err("invalid scope");
        assert_eq!(error, super::RegistrationError::InvalidScope);
    }

    #[test]
    fn rejects_mismatched_cgroup() {
        let error = parse_registration(
            registration()
                .replace("app.slice", "other.slice")
                .as_bytes(),
            UID,
        )
        .expect_err("invalid cgroup");
        assert_eq!(error, super::RegistrationError::InvalidControlGroup);
    }

    #[test]
    fn accepts_trailing_whitespace_in_meminfo_field() {
        assert_eq!(parse_mem_available(b"MemAvailable: 2 kB  \t\n"), Ok(2048));
    }

    #[test]
    fn registration_requires_newline_terminated_lines() {
        let without_final_newline = registration().trim_end().to_owned();
        assert!(parse_registration(without_final_newline.as_bytes(), UID).is_ok());
    }

    #[test]
    fn cgroup_kill_releases_the_reserve_before_writing() {
        let (root, registration) = target_tree();
        let mut target =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect("open target");
        let mut reserve = Some(vec![1_u8; 4096]);
        target.kill_direct(&mut reserve).expect("kill");
        assert!(reserve.is_none());
        let kill_path = root.join(format!(
            "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope/cgroup.kill",
            Uid::effective().as_raw(),
            Uid::effective().as_raw(),
            "a".repeat(64)
        ));
        assert_eq!(fs::read(kill_path).expect("read kill"), b"1");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn cgroup_kill_can_be_retried_without_rearming_the_reserve() {
        let (root, registration) = target_tree();
        let mut target =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect("open target");
        let mut reserve = Some(vec![1_u8; 4096]);
        target.kill_direct(&mut reserve).expect("first kill");
        target.kill_direct(&mut reserve).expect("retry kill");
        assert!(reserve.is_none());
        let kill_path = root.join(format!(
            "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope/cgroup.kill",
            Uid::effective().as_raw(),
            Uid::effective().as_raw(),
            "a".repeat(64)
        ));
        assert_eq!(fs::read(kill_path).expect("read kill"), b"11");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn cgroup_events_report_empty_target() {
        let (root, registration) = target_tree();
        let target =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect("open target");
        let events = root.join(format!(
            "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope/cgroup.events",
            Uid::effective().as_raw(),
            Uid::effective().as_raw(),
            "a".repeat(64)
        ));
        fs::write(events, b"populated 0\n").expect("mark empty");
        assert!(target.is_empty().expect("events"));
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn empty_cgroup_is_rejected_while_arming() {
        let (root, registration) = target_tree();
        let events = root.join(format!(
            "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope/cgroup.events",
            Uid::effective().as_raw(),
            Uid::effective().as_raw(),
            "a".repeat(64)
        ));
        fs::write(events, b"populated 0\n").expect("mark empty");
        let error =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect_err("empty target must not arm");
        assert!(error.to_string().contains("not populated"));
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn insecure_registration_is_rejected_before_opening_cgroup() {
        let (root, registration) = target_tree();
        fs::set_permissions(&registration, fs::Permissions::from_mode(0o644))
            .expect("make registration insecure");
        assert!(
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .is_err()
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

        let mut target =
            CgroupTarget::from_registration(&registration, &root, uid).expect("open replacement");
        let mut reserve = None;
        target.kill_direct(&mut reserve).expect("kill replacement");
        assert_eq!(
            fs::read(replacement.join("cgroup.kill")).expect("read replacement kill"),
            b"1"
        );
        fs::remove_dir_all(root).expect("remove root");
    }
}
