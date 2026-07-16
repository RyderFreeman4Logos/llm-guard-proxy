//! Configuration parsing for the host-memory guardian.

use serde::Deserialize;
use std::{
    fs::{File, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::Duration,
};
use thiserror::Error;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const CONFIG_SCHEMA_VERSION: u32 = 1;
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_RETRY_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_ESCALATION_GRACE: Duration = Duration::from_secs(60);

/// A validated pair of memory thresholds published together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Thresholds {
    mem_avail_stop_gib: u64,
    reserve_mib: u64,
    threshold_bytes: u64,
    reserve_bytes: usize,
}

impl Thresholds {
    /// Creates thresholds and precomputes the allocation-free loop values.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Invalid`] when either unit is zero or the byte
    /// conversion cannot be represented by the target platform.
    pub fn new(mem_avail_stop_gib: u64, reserve_mib: u64) -> Result<Self, ConfigError> {
        if mem_avail_stop_gib == 0 {
            return Err(ConfigError::Invalid(String::from(
                "thresholds.mem_avail_stop_gib must be positive",
            )));
        }
        if reserve_mib == 0 {
            return Err(ConfigError::Invalid(String::from(
                "thresholds.reserve_mib must be positive",
            )));
        }
        let threshold_bytes = mem_avail_stop_gib.checked_mul(GIB).ok_or_else(|| {
            ConfigError::Invalid(String::from(
                "thresholds.mem_avail_stop_gib overflows bytes",
            ))
        })?;
        let reserve_bytes = reserve_mib
            .checked_mul(MIB)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                ConfigError::Invalid(String::from("thresholds.reserve_mib overflows bytes"))
            })?;
        Ok(Self {
            mem_avail_stop_gib,
            reserve_mib,
            threshold_bytes,
            reserve_bytes,
        })
    }

    /// Returns the configured stop threshold in GiB.
    #[must_use]
    pub const fn mem_avail_stop_gib(self) -> u64 {
        self.mem_avail_stop_gib
    }

    /// Returns the configured emergency reserve in MiB.
    #[must_use]
    pub const fn reserve_mib(self) -> u64 {
        self.reserve_mib
    }

    /// Returns the stop threshold in bytes.
    #[must_use]
    pub const fn threshold_bytes(self) -> u64 {
        self.threshold_bytes
    }

    /// Returns the pre-touched emergency reserve size in bytes.
    #[must_use]
    pub const fn reserve_bytes(self) -> usize {
        self.reserve_bytes
    }
}

/// A validated target registration used to derive an already-open cgroup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetConfig {
    label: String,
    registration_file: String,
}

impl TargetConfig {
    /// Returns the operator-facing target label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the safe, single-component registration filename.
    #[must_use]
    pub fn registration_file(&self) -> &str {
        &self.registration_file
    }
}

/// Runtime controls which do not alter the emergency fast path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    cgroup_root: PathBuf,
    poll_interval: Duration,
    retry_interval: Duration,
}

impl RuntimeConfig {
    /// Returns the cgroup-v2 hierarchy root.
    #[must_use]
    pub fn cgroup_root(&self) -> &Path {
        &self.cgroup_root
    }

    /// Returns the healthy-loop polling interval.
    #[must_use]
    pub const fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Returns the latched retry interval.
    #[must_use]
    pub const fn retry_interval(&self) -> Duration {
        self.retry_interval
    }
}

/// Tier-2 escalation remains opt-in; Tier 1 is the only default action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EscalationConfig {
    pub(crate) enabled: bool,
    pub(crate) unit: Option<String>,
    pub(crate) grace_period: Duration,
}

impl EscalationConfig {
    /// Whether the optional systemd escalation may run.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the validated unit if escalation is enabled.
    #[must_use]
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Returns the post-Tier-1 grace period.
    #[must_use]
    pub const fn grace_period(&self) -> Duration {
        self.grace_period
    }
}

/// Coherent configuration snapshot used by each healthy monitoring iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuardianConfig {
    thresholds: Thresholds,
    target: TargetConfig,
    runtime: RuntimeConfig,
    escalation: EscalationConfig,
}

impl GuardianConfig {
    /// Parses a guardian configuration file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, parsed, or validated.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits())
            .open(path)
            .map_err(|source| ConfigError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        validate_config_file(&file, path)?;
        let mut text = String::new();
        file.read_to_string(&mut text)
            .map_err(|source| ConfigError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        let file: FileConfig = toml::from_str(&text).map_err(ConfigError::Parse)?;
        file.resolve()
    }

    /// Returns the current threshold pair.
    #[must_use]
    pub const fn thresholds(&self) -> Thresholds {
        self.thresholds
    }

    /// Returns the registered cgroup target configuration.
    #[must_use]
    pub fn target(&self) -> &TargetConfig {
        &self.target
    }

    /// Returns runtime polling and cgroup controls.
    #[must_use]
    pub fn runtime(&self) -> &RuntimeConfig {
        &self.runtime
    }

    /// Returns optional Tier-2 escalation controls.
    #[must_use]
    pub fn escalation(&self) -> &EscalationConfig {
        &self.escalation
    }
}

fn validate_config_file(file: &File, path: &Path) -> Result<(), ConfigError> {
    let metadata = file.metadata().map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file()
        || metadata.uid() != nix::unistd::Uid::effective().as_raw()
        || metadata.nlink() != 1
        || metadata.mode() & 0o7777 != 0o600
    {
        return Err(ConfigError::Invalid(String::from(
            "guardian config must be a single-link 0600 regular file owned by the effective user",
        )));
    }
    Ok(())
}

/// Configuration failures leave the last known-good live snapshot in place.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("read guardian config {path}: {source}", path = path.display())]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// TOML was invalid or included unknown fields.
    #[error("parse guardian config: {0}")]
    Parse(toml::de::Error),
    /// The configuration watcher could not be created or reported an error.
    #[error("watch guardian config: {0}")]
    Watch(String),
    /// A syntactically valid value broke the guardian contract.
    #[error("invalid guardian config: {0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    schema_version: u32,
    target: FileTarget,
    #[serde(default)]
    thresholds: FileThresholds,
    #[serde(default)]
    runtime: FileRuntime,
    #[serde(default)]
    escalation: FileEscalation,
}

impl FileConfig {
    fn resolve(self) -> Result<GuardianConfig, ConfigError> {
        if self.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::Invalid(format!(
                "schema_version must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        let target = self.target.resolve()?;
        let thresholds = self.thresholds.resolve()?;
        let runtime = self.runtime.resolve()?;
        let escalation = self.escalation.resolve()?;
        Ok(GuardianConfig {
            thresholds,
            target,
            runtime,
            escalation,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileTarget {
    label: String,
    registration_file: String,
}

impl FileTarget {
    fn resolve(self) -> Result<TargetConfig, ConfigError> {
        if self.label.is_empty()
            || self.label.len() > 64
            || !self
                .label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(ConfigError::Invalid(String::from(
                "target.label must contain 1 to 64 ASCII letters, digits, '.', '_' or '-'",
            )));
        }
        let mut registration_bytes = self.registration_file.bytes();
        let registration_valid = self.registration_file.len() <= 128
            && registration_bytes
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && registration_bytes
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
        if !registration_valid {
            return Err(ConfigError::Invalid(String::from(
                "target.registration_file must be one safe 1 to 128 byte ASCII filename",
            )));
        }
        Ok(TargetConfig {
            label: self.label,
            registration_file: self.registration_file,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileThresholds {
    mem_avail_stop_gib: Option<u64>,
    reserve_mib: Option<u64>,
}

impl FileThresholds {
    fn resolve(self) -> Result<Thresholds, ConfigError> {
        Thresholds::new(
            self.mem_avail_stop_gib.unwrap_or(1),
            self.reserve_mib.unwrap_or(64),
        )
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRuntime {
    cgroup_root: Option<PathBuf>,
    poll_interval_secs: Option<u64>,
    retry_interval_secs: Option<u64>,
}

impl FileRuntime {
    fn resolve(self) -> Result<RuntimeConfig, ConfigError> {
        let poll_interval = Duration::from_secs(
            self.poll_interval_secs
                .unwrap_or(DEFAULT_POLL_INTERVAL.as_secs()),
        );
        let retry_interval = Duration::from_secs(
            self.retry_interval_secs
                .unwrap_or(DEFAULT_RETRY_INTERVAL.as_secs()),
        );
        if poll_interval.is_zero() || retry_interval.is_zero() {
            return Err(ConfigError::Invalid(String::from(
                "runtime intervals must be positive",
            )));
        }
        Ok(RuntimeConfig {
            cgroup_root: self
                .cgroup_root
                .unwrap_or_else(|| PathBuf::from("/sys/fs/cgroup")),
            poll_interval,
            retry_interval,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEscalation {
    enabled: Option<bool>,
    unit: Option<String>,
    grace_period_secs: Option<u64>,
}

impl FileEscalation {
    fn resolve(self) -> Result<EscalationConfig, ConfigError> {
        let enabled = self.enabled.unwrap_or(false);
        let unit = self.unit.filter(|unit| !unit.is_empty());
        if enabled && unit.is_none() {
            return Err(ConfigError::Invalid(String::from(
                "escalation.unit is required when escalation.enabled is true",
            )));
        }
        if let Some(unit) = unit.as_deref() {
            crate::escalation::validate_unit_name(unit)
                .map_err(|error| ConfigError::Invalid(error.to_string()))?;
        }
        Ok(EscalationConfig {
            enabled,
            unit,
            grace_period: Duration::from_secs(
                self.grace_period_secs
                    .unwrap_or(DEFAULT_ESCALATION_GRACE.as_secs()),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigError, GuardianConfig, Thresholds};
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn parse(text: &str) -> Result<GuardianConfig, ConfigError> {
        let file = toml::from_str(text).map_err(ConfigError::Parse)?;
        super::FileConfig::resolve(file)
    }

    const CONFIG: &str = r#"
schema_version = 1
[target]
label = "aeon-text"
registration_file = "text-cgroup.v1"
[thresholds]
mem_avail_stop_gib = 2
reserve_mib = 128
"#;

    #[test]
    fn parses_configured_thresholds() {
        let config = parse(CONFIG).expect("config parses");
        assert_eq!(
            config.thresholds(),
            Thresholds::new(2, 128).expect("thresholds")
        );
    }

    #[test]
    fn defaults_thresholds_independently() {
        let config = parse(
            "schema_version = 1\n[target]\nlabel = \"x\"\nregistration_file = \"x.v1\"\n[thresholds]\nreserve_mib = 32\n",
        )
        .expect("config parses");
        assert_eq!(config.thresholds().mem_avail_stop_gib(), 1);
        assert_eq!(config.thresholds().reserve_mib(), 32);
    }

    #[test]
    fn rejects_zero_threshold() {
        let error = parse(&CONFIG.replace("mem_avail_stop_gib = 2", "mem_avail_stop_gib = 0"))
            .expect_err("zero threshold fails");
        assert!(error.to_string().contains("mem_avail_stop_gib"));
    }

    #[test]
    fn rejects_zero_reserve() {
        let error = parse(&CONFIG.replace("reserve_mib = 128", "reserve_mib = 0"))
            .expect_err("zero reserve fails");
        assert!(error.to_string().contains("reserve_mib"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let error =
            parse(&format!("{CONFIG}\nunexpected = true\n")).expect_err("unknown field fails");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_wrong_schema() {
        let error = parse(&CONFIG.replacen("schema_version = 1", "schema_version = 2", 1))
            .expect_err("schema fails");
        assert!(error.to_string().contains("schema_version"));
    }

    #[test]
    fn rejects_unsafe_registration_name() {
        let error =
            parse(&CONFIG.replace("text-cgroup.v1", "../target")).expect_err("unsafe name fails");
        assert!(error.to_string().contains("registration_file"));
    }

    #[test]
    fn rejects_unsafe_target_label() {
        let error = parse(&CONFIG.replace("aeon-text", "aeon text"))
            .expect_err("unsafe target label fails");
        assert!(error.to_string().contains("target.label"));
    }

    #[test]
    fn escalation_is_disabled_by_default() {
        let config = parse(CONFIG).expect("config parses");
        assert!(!config.escalation().enabled());
    }

    #[test]
    fn rejects_an_unsafe_escalation_unit_while_healthy() {
        let error = parse(&format!(
            "{CONFIG}\n[escalation]\nenabled = true\nunit = \"../recovery.service\"\n"
        ))
        .expect_err("unsafe unit must fail before emergency mode");
        assert!(error.to_string().contains("systemd service unit"));
    }

    #[test]
    fn rejects_a_symlinked_config_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("guardian-config-{nonce}"));
        fs::create_dir(&directory).expect("create directory");
        let real_path = directory.join("real.toml");
        fs::write(&real_path, CONFIG).expect("write config");
        fs::set_permissions(&real_path, fs::Permissions::from_mode(0o600)).expect("secure config");
        let link_path = directory.join("config.toml");
        symlink(&real_path, &link_path).expect("create symlink");

        assert!(GuardianConfig::load(&link_path).is_err());

        fs::remove_dir_all(directory).expect("remove directory");
    }
}
