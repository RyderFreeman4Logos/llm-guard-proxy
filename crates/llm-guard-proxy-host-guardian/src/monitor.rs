//! The allocation-aware Tier-1 memory monitoring loop.

use crate::{
    config::{ConfigError, Thresholds},
    emergency::{AttemptOutcome, EmergencyController, EmergencyReserve},
    escalation::restart_user_unit,
};
use llm_guard_proxy_core::{
    ConfigHandle, ConfigHandleError, GuardianConfig, GuardianKillAction, ValidationError,
};
use nix::fcntl::OFlag;
use nix::unistd::Uid;
use std::{
    fs::{File, OpenOptions},
    io::{self, Read},
    os::{
        fd::AsRawFd,
        unix::{fs::FileExt, fs::MetadataExt, fs::OpenOptionsExt},
    },
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use thiserror::Error;

const MEMINFO_BUFFER_BYTES: usize = 8 * 1024;
const REGISTRATION_MAX_BYTES: usize = 1024;
const EVENTS_BUFFER_BYTES: usize = 512;
const CONFIG_SYNC_INTERVAL: Duration = Duration::from_secs(1);

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
    mem_available_bytes <= thresholds.threshold_bytes()
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
    /// The pre-opened meminfo descriptor could not be read.
    #[error("MemAvailable could not be read")]
    Read,
    /// The fixed emergency buffer was insufficient.
    #[error("MemAvailable exceeds the fixed buffer")]
    TooLarge,
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
    directory: File,
    kill: File,
    events: File,
    registration: Registration,
}

/// Compact cgroup state failures for the allocation-free emergency loop.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum CgroupStateError {
    #[error("cgroup.events could not be read")]
    Read,
    #[error("cgroup.events is malformed")]
    Invalid,
}

impl CgroupTarget {
    /// Opens one validated registration with descriptors retained for Tier 1.
    ///
    /// This is a healthy-path operation; it parses untrusted registration data
    /// and must complete before the target replaces the last-good descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`GuardianError`] when the registration or its target cgroup
    /// cannot be validated and opened.
    pub fn open_registered(
        registration_path: &Path,
        cgroup_root: &Path,
    ) -> Result<Self, GuardianError> {
        Self::from_registration(registration_path, cgroup_root, Uid::effective().as_raw())
    }

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
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags((OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC).bits())
            .open(cgroup_path)
            .map_err(|source| GuardianError::Io {
                operation: "open cgroup directory",
                source,
            })?;
        let kill = open_cgroup_control(&directory, "cgroup.kill", true, "open cgroup.kill")?;
        let events = open_cgroup_control(&directory, "cgroup.events", false, "open cgroup.events")?;
        let target = Self {
            directory,
            kill,
            events,
            registration,
        };
        if !target.is_populated().map_err(|_error| {
            GuardianError::InvalidRegistration(String::from(
                "registered cgroup could not be verified as populated",
            ))
        })? {
            return Err(GuardianError::InvalidRegistration(String::from(
                "registered cgroup is not populated",
            )));
        }
        Ok(target)
    }

    pub(crate) fn kill_fd(&self) -> std::os::fd::RawFd {
        self.kill.as_raw_fd()
    }

    pub(crate) fn is_empty(&self) -> Result<bool, CgroupStateError> {
        Ok(!self.is_populated()?)
    }

    fn has_same_target_generation(&self, other: &Self) -> bool {
        if self.registration != other.registration {
            return false;
        }
        let (Ok(active), Ok(candidate)) = (self.directory.metadata(), other.directory.metadata())
        else {
            return false;
        };
        active.dev() == candidate.dev() && active.ino() == candidate.ino()
    }

    fn is_populated(&self) -> Result<bool, CgroupStateError> {
        let mut bytes = [0_u8; EVENTS_BUFFER_BYTES];
        let length = self
            .events
            .read_at(&mut bytes, 0)
            .map_err(|_error| CgroupStateError::Read)?;
        let text =
            std::str::from_utf8(&bytes[..length]).map_err(|_error| CgroupStateError::Invalid)?;
        match text.lines().map(str::trim).find_map(|line| match line {
            "populated 0" => Some(false),
            "populated 1" => Some(true),
            _ => None,
        }) {
            Some(populated) => Ok(populated),
            None => Err(CgroupStateError::Invalid),
        }
    }
}

fn open_cgroup_control(
    directory: &File,
    name: &'static str,
    write: bool,
    operation: &'static str,
) -> Result<File, GuardianError> {
    // The retained directory descriptor pins one cgroup object even when its
    // registered path is concurrently replaced by a new generation.
    let descriptor_path = PathBuf::from("/proc/self/fd")
        .join(directory.as_raw_fd().to_string())
        .join(name);
    OpenOptions::new()
        .read(!write)
        .write(write)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(descriptor_path)
        .map_err(|source| GuardianError::Io { operation, source })
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
    /// Memory pressure crossed the configured threshold, but recovery actions
    /// are disabled and the guardian only reported the event.
    ObserverOnly,
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
    /// The shared proxy configuration could not be read.
    #[error(transparent)]
    ConfigHandle(#[from] ConfigHandleError),
    /// The shared proxy configuration is not safe to install.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// The pre-touched emergency reserve could not be allocated at startup.
    #[error(transparent)]
    Reserve(#[from] crate::emergency::ReserveError),
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

#[derive(Debug)]
enum RecoveryTarget {
    Cgroup(CgroupTarget),
    Systemd { unit: String },
}

/// Runs Tier 1 continuously and owns all state which may allocate while healthy.
#[derive(Debug)]
pub struct MemoryGuardian {
    config_handle: ConfigHandle,
    active_policy: GuardianConfig,
    thresholds: Thresholds,
    runtime_dir: PathBuf,
    target: Option<RecoveryTarget>,
    proc_meminfo: File,
    controller: Option<EmergencyController>,
    poll_interval: Duration,
    retry_interval: Duration,
    started: Instant,
    latched: bool,
    systemd_next_attempt_millis: u64,
    systemd_verified: bool,
    observer_pressure_reported: bool,
    last_rejected_policy: Option<GuardianConfig>,
}

impl MemoryGuardian {
    /// Opens a guardian backed by the proxy's shared validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the shared config, an enabled reserve mapping, or
    /// `/proc/meminfo` cannot be opened. Missing cgroup registrations are
    /// retried by the healthy loop without taking down the proxy.
    pub fn open(
        config_handle: ConfigHandle,
        runtime_dir: impl Into<PathBuf>,
    ) -> Result<Self, GuardianError> {
        let snapshot = config_handle.snapshot()?;
        snapshot.validate()?;
        let active_policy = snapshot.guardian;
        let thresholds = thresholds_from_policy(&active_policy)?;
        let runtime_dir = runtime_dir.into();
        let controller = if active_policy.enabled {
            Some(EmergencyController::new(
                EmergencyReserve::new(thresholds.reserve_bytes())?,
                active_policy.retry_interval_secs.saturating_mul(1000),
            ))
        } else {
            None
        };
        let proc_meminfo = File::open("/proc/meminfo").map_err(|source| GuardianError::Io {
            operation: "open /proc/meminfo",
            source,
        })?;
        Ok(Self {
            config_handle,
            poll_interval: Duration::from_secs(active_policy.poll_interval_secs),
            retry_interval: Duration::from_secs(active_policy.retry_interval_secs),
            active_policy,
            thresholds,
            runtime_dir,
            target: None,
            proc_meminfo,
            controller,
            started: Instant::now(),
            latched: false,
            systemd_next_attempt_millis: 0,
            systemd_verified: false,
            observer_pressure_reported: false,
            last_rejected_policy: None,
        })
    }

    /// Returns the policy currently armed by the guardian.
    ///
    /// A shared configuration update is exposed here only after any required
    /// cgroup target and emergency reserve have been prepared successfully.
    #[must_use]
    pub const fn active_policy(&self) -> &GuardianConfig {
        &self.active_policy
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
    /// Recoverable meminfo, cgroup, and reload failures are retained in the
    /// guardian state and never terminate the monitoring task.
    pub fn tick(&mut self) -> Result<GuardianIteration, GuardianError> {
        if self.latched {
            return Ok(self.latched_iteration());
        }
        Ok(self.healthy_iteration())
    }

    /// Runs until the supplied shutdown future resolves.
    ///
    /// # Errors
    ///
    /// Only shutdown completes this loop; recoverable protection failures are
    /// logged and retried without taking down combined proxy mode.
    pub async fn run_until<F>(&mut self, shutdown: F) -> Result<(), GuardianError>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(shutdown);
        loop {
            self.tick()?;
            let delay = if self.latched {
                self.retry_interval
            } else {
                self.poll_interval
            };
            let mut remaining = delay;
            while !remaining.is_zero() {
                let slice = if self.latched {
                    remaining
                } else {
                    remaining.min(CONFIG_SYNC_INTERVAL)
                };
                let wait_started = Instant::now();
                tokio::select! {
                    () = &mut shutdown => return Ok(()),
                    () = tokio::time::sleep(slice) => {}
                }
                remaining = remaining.saturating_sub(wait_started.elapsed());
                if !self.latched && self.reconcile_healthy_target() {
                    break;
                }
            }
        }
    }

    fn healthy_iteration(&mut self) -> GuardianIteration {
        let _changed = self.reconcile_healthy_target();
        let available = match read_mem_available(&self.proc_meminfo) {
            Ok(available) => available,
            Err(_error) => return GuardianIteration::Healthy,
        };
        if should_shed(available, self.thresholds) {
            if !self.active_policy.enabled {
                if !self.observer_pressure_reported {
                    eprintln!(
                        "memory guardian: MemAvailable crossed the configured threshold; observer-only mode leaves target {:?} untouched",
                        self.active_policy.target_label
                    );
                    self.observer_pressure_reported = true;
                }
                return GuardianIteration::ObserverOnly;
            }
            self.latched = true;
            if let Some(controller) = self.controller.as_mut() {
                controller.enter_emergency();
            }
            return self.attempt_emergency(true);
        }
        self.observer_pressure_reported = false;
        if !self.active_policy.enabled || self.target.is_some() {
            GuardianIteration::Healthy
        } else {
            GuardianIteration::Unarmed
        }
    }

    fn latched_iteration(&mut self) -> GuardianIteration {
        let available = read_mem_available_compact(&self.proc_meminfo);
        if available.is_ok_and(|value| should_rearm(value, self.thresholds)) {
            let reserve = self
                .controller
                .as_mut()
                .map(|controller| controller.ensure_reserve(self.thresholds.reserve_bytes()));
            match reserve {
                Some(Ok(_)) => {
                    self.latched = false;
                    self.systemd_next_attempt_millis = 0;
                    self.systemd_verified = false;
                    return GuardianIteration::Rearmed;
                }
                None => {
                    self.latched = false;
                    return GuardianIteration::Rearmed;
                }
                Some(Err(_error)) => {}
            }
        }
        let _changed = self.reconcile_active_target();
        self.attempt_emergency(false)
    }

    fn attempt_emergency(&mut self, entering: bool) -> GuardianIteration {
        let Some(target) = self.target.as_ref() else {
            return GuardianIteration::Unarmed;
        };
        let now_millis = elapsed_millis(self.started);
        let outcome = match target {
            RecoveryTarget::Cgroup(target) => self
                .controller
                .as_mut()
                .map_or(AttemptOutcome::Waiting, |controller| {
                    controller.attempt(now_millis, target)
                }),
            RecoveryTarget::Systemd { unit } => {
                let unit = unit.clone();
                self.attempt_systemd_restart(now_millis, &unit)
            }
        };
        match outcome {
            AttemptOutcome::Verified => GuardianIteration::Verified,
            AttemptOutcome::Waiting | AttemptOutcome::Retry if entering => GuardianIteration::Shed,
            AttemptOutcome::Waiting | AttemptOutcome::Retry => GuardianIteration::Waiting,
        }
    }

    fn attempt_systemd_restart(&mut self, now_millis: u64, unit: &str) -> AttemptOutcome {
        if self.systemd_verified || now_millis < self.systemd_next_attempt_millis {
            return AttemptOutcome::Waiting;
        }
        if restart_user_unit(unit, Duration::from_secs(15)).is_ok() {
            self.systemd_verified = true;
            return AttemptOutcome::Verified;
        }
        self.systemd_next_attempt_millis = now_millis
            .saturating_add(u64::try_from(self.retry_interval.as_millis()).unwrap_or(u64::MAX));
        AttemptOutcome::Retry
    }

    fn reconcile_healthy_target(&mut self) -> bool {
        let Ok(requested) = self.config_handle.guardian_snapshot() else {
            return false;
        };
        if requested != self.active_policy {
            return self.try_apply_policy(requested);
        }
        self.reconcile_active_target()
    }

    /// Reopens only the target selected by the installed policy.
    ///
    /// Latched retries use this path so a registration or cgroup generation
    /// can appear under pressure without applying a pending hot-reload.
    fn reconcile_active_target(&mut self) -> bool {
        let target_needs_reconciliation = self.active_policy.enabled
            && (self.target.is_none()
                || self.active_policy.kill_action == GuardianKillAction::CgroupKill);
        if target_needs_reconciliation {
            match open_recovery_target(&self.active_policy, &self.runtime_dir) {
                Ok(target) => {
                    if matches!(
                        (&self.target, &target),
                        (
                            Some(RecoveryTarget::Cgroup(active)),
                            RecoveryTarget::Cgroup(candidate)
                        ) if active.has_same_target_generation(candidate)
                    ) {
                        self.last_rejected_policy = None;
                        return false;
                    }
                    self.target = Some(target);
                    self.last_rejected_policy = None;
                    if let Some(controller) = self.controller.as_mut() {
                        controller.reset_for_target_generation();
                    }
                    return true;
                }
                Err(error) => {
                    let policy = self.active_policy.clone();
                    self.note_policy_rejection(&policy, &error);
                }
            }
        }
        false
    }

    fn try_apply_policy(&mut self, requested: GuardianConfig) -> bool {
        let thresholds = match thresholds_from_policy(&requested) {
            Ok(thresholds) => thresholds,
            Err(error) => {
                self.note_policy_rejection(&requested, &error);
                return false;
            }
        };
        if !requested.enabled {
            self.active_policy = requested;
            self.thresholds = thresholds;
            self.target = None;
            self.controller = None;
            self.poll_interval = Duration::from_secs(self.active_policy.poll_interval_secs);
            self.retry_interval = Duration::from_secs(self.active_policy.retry_interval_secs);
            self.systemd_next_attempt_millis = 0;
            self.systemd_verified = false;
            self.observer_pressure_reported = false;
            self.last_rejected_policy = None;
            return true;
        }

        let target = match open_recovery_target(&requested, &self.runtime_dir) {
            Ok(target) => target,
            Err(error) => {
                self.note_policy_rejection(&requested, &error);
                return false;
            }
        };
        let reserve = match EmergencyReserve::new(thresholds.reserve_bytes()) {
            Ok(reserve) => reserve,
            Err(error) => {
                self.note_policy_rejection(&requested, &error);
                return false;
            }
        };
        let retry_millis = requested.retry_interval_secs.saturating_mul(1000);
        self.poll_interval = Duration::from_secs(requested.poll_interval_secs);
        self.retry_interval = Duration::from_secs(requested.retry_interval_secs);
        self.active_policy = requested;
        self.thresholds = thresholds;
        self.target = Some(target);
        self.controller = Some(EmergencyController::new(reserve, retry_millis));
        self.systemd_next_attempt_millis = 0;
        self.systemd_verified = false;
        self.observer_pressure_reported = false;
        self.last_rejected_policy = None;
        true
    }

    fn note_policy_rejection(
        &mut self,
        requested: &GuardianConfig,
        error: &impl std::fmt::Display,
    ) {
        if self.last_rejected_policy.as_ref() != Some(requested) {
            eprintln!(
                "memory guardian: retaining target {:?}; requested policy for {:?} could not be armed: {error}",
                self.active_policy.target_label, requested.target_label
            );
            self.last_rejected_policy = Some(requested.clone());
        }
    }
}

/// Returns the default runtime registration directory for the effective user.
#[must_use]
pub fn default_runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", Uid::effective().as_raw())))
        .join("gb10-memory-guardian")
}

fn thresholds_from_policy(policy: &GuardianConfig) -> Result<Thresholds, ConfigError> {
    Thresholds::new(policy.mem_threshold_gib, policy.reserve_mib)
}

fn open_recovery_target(
    policy: &GuardianConfig,
    runtime_dir: &Path,
) -> Result<RecoveryTarget, GuardianError> {
    match policy.kill_action {
        GuardianKillAction::CgroupKill => {
            let registration = runtime_dir.join(policy.effective_registration_file());
            CgroupTarget::open_registered(&registration, &policy.cgroup_root)
                .map(RecoveryTarget::Cgroup)
        }
        GuardianKillAction::SystemctlRestart => Ok(RecoveryTarget::Systemd {
            unit: policy.effective_systemd_unit(),
        }),
    }
}

fn read_mem_available(file: &File) -> Result<u64, GuardianError> {
    read_mem_available_compact(file).map_err(|error| {
        GuardianError::InvalidRegistration(format!("invalid /proc/meminfo: {error}"))
    })
}

fn read_mem_available_compact(file: &File) -> Result<u64, MemInfoError> {
    let mut bytes = [0_u8; MEMINFO_BUFFER_BYTES];
    let length = file
        .read_at(&mut bytes, 0)
        .map_err(|_error| MemInfoError::Read)?;
    if length == bytes.len() {
        return Err(MemInfoError::TooLarge);
    }
    parse_mem_available(&bytes[..length])
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        CgroupTarget, MemInfoError, parse_mem_available, parse_registration, should_rearm,
        should_shed,
    };
    use crate::{EmergencyReserve, Thresholds, kill_direct};
    use llm_guard_proxy_core::{AppConfig, ConfigHandle, GuardianKillAction};
    use nix::unistd::Uid;
    use std::{
        fs::{self, File},
        os::unix::fs::PermissionsExt,
        path::{Path, PathBuf},
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

    fn guardian_config(registration_file: &str, cgroup_root: &Path) -> AppConfig {
        let mut config = AppConfig::default();
        config.guardian.enabled = true;
        config.guardian.target_label = String::from("test");
        config.guardian.registration_file = Some(registration_file.to_owned());
        config.guardian.mem_threshold_gib = 1;
        config.guardian.reserve_mib = 1;
        config.guardian.poll_interval_secs = 1;
        config.guardian.retry_interval_secs = 1;
        config.guardian.cgroup_root = cgroup_root.to_path_buf();
        config
    }

    fn guardian_handle(registration_file: &str, cgroup_root: &Path) -> ConfigHandle {
        ConfigHandle::new(guardian_config(registration_file, cgroup_root))
    }

    mod target_reconciliation;

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
    fn sheds_at_or_below_threshold() {
        let thresholds = Thresholds::new(1, 64).expect("thresholds");
        assert!(should_shed(thresholds.threshold_bytes() - 1, thresholds));
        assert!(should_shed(thresholds.threshold_bytes(), thresholds));
        assert!(!should_shed(thresholds.threshold_bytes() + 1, thresholds));
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
        let target =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect("open target");
        let mut reserve = EmergencyReserve::with_page_size(4096, 4096).expect("reserve");
        kill_direct(&mut reserve, &target).expect("kill");
        assert!(!reserve.is_allocated());
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
        let target =
            CgroupTarget::from_registration(&registration, &root, Uid::effective().as_raw())
                .expect("open target");
        let mut reserve = EmergencyReserve::with_page_size(4096, 4096).expect("reserve");
        kill_direct(&mut reserve, &target).expect("first kill");
        kill_direct(&mut reserve, &target).expect("retry kill");
        assert!(!reserve.is_allocated());
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
    fn disabled_policy_observes_pressure_without_latching_or_arming() {
        let root = temporary_tree();
        let meminfo = root.join("meminfo");
        fs::write(&meminfo, b"MemAvailable: 0 kB\n").expect("write meminfo");
        let mut config = AppConfig::default();
        config.guardian.mem_threshold_gib = 1;
        config.guardian.reserve_mib = 1;
        let mut guardian = super::MemoryGuardian::open(ConfigHandle::new(config), &root)
            .expect("open observer guardian");
        guardian.proc_meminfo = File::open(&meminfo).expect("open meminfo");

        assert_eq!(
            guardian.tick().expect("observer tick"),
            super::GuardianIteration::ObserverOnly
        );
        assert!(!guardian.latched);
        assert!(guardian.controller.is_none());
        assert!(guardian.target.is_none());
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn invalid_hot_reload_candidate_retains_the_last_good_target() {
        let (root, _registration) = target_tree();
        let runtime = root.join("runtime");
        let handle = guardian_handle("target.v1", &root);
        let mut guardian =
            super::MemoryGuardian::open(handle.clone(), &runtime).expect("open guardian");
        guardian.reconcile_healthy_target();
        assert!(guardian.target.is_some());

        let requested = guardian_config("missing.v1", &root);
        handle
            .apply_reloadable(&requested)
            .expect("publish requested policy");
        guardian.reconcile_healthy_target();

        assert!(guardian.target.is_some());
        assert_eq!(
            guardian.active_policy.registration_file.as_deref(),
            Some("target.v1")
        );
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn hot_reload_applies_a_complete_systemctl_policy_without_restart() {
        let (root, _registration) = target_tree();
        let runtime = root.join("runtime");
        let handle = guardian_handle("target.v1", &root);
        let mut guardian =
            super::MemoryGuardian::open(handle.clone(), &runtime).expect("open guardian");
        guardian.reconcile_healthy_target();

        let mut requested = guardian_config("unused.v1", &root);
        requested.guardian.target_label = String::from("replacement");
        requested.guardian.mem_threshold_gib = 3;
        requested.guardian.kill_action = GuardianKillAction::SystemctlRestart;
        requested.guardian.poll_interval_secs = 4;
        requested.guardian.systemd_unit = Some(String::from("replacement.service"));
        handle
            .apply_reloadable(&requested)
            .expect("publish requested policy");
        guardian.reconcile_healthy_target();

        assert_eq!(guardian.active_policy(), &requested.guardian);
        assert!(matches!(
            guardian.target,
            Some(super::RecoveryTarget::Systemd { .. })
        ));
        assert_eq!(guardian.poll_interval, std::time::Duration::from_secs(4));
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn latched_emergency_retries_after_meminfo_and_events_errors() {
        let (root, registration) = target_tree();
        let runtime = root.join("runtime");
        let handle = guardian_handle("target.v1", &root);
        let mut guardian = super::MemoryGuardian::open(handle, &runtime).expect("open guardian");
        guardian.reconcile_healthy_target();

        let malformed_meminfo = root.join("meminfo");
        fs::write(&malformed_meminfo, b"malformed\n").expect("write meminfo");
        guardian.proc_meminfo = File::open(&malformed_meminfo).expect("open meminfo");
        let events = root.join(format!(
            "user.slice/user-{}.slice/user@{}.service/app.slice/docker-{}.scope/cgroup.events",
            Uid::effective().as_raw(),
            Uid::effective().as_raw(),
            "a".repeat(64)
        ));
        fs::write(events, b"malformed\n").expect("break events");
        guardian.latched = true;
        guardian
            .controller
            .as_mut()
            .expect("enabled policy controller")
            .enter_emergency();

        assert_eq!(
            guardian.tick().expect("recoverable failure"),
            super::GuardianIteration::Waiting
        );
        assert!(registration.exists());
        fs::remove_dir_all(root).expect("remove root");
    }
}
