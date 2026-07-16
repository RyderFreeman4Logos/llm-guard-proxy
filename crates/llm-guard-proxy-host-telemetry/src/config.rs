//! Configuration for observer-only host telemetry.

use serde::Deserialize;
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use thiserror::Error;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_GPU_TIMEOUT: Duration = Duration::from_secs(2);
const DEFAULT_MAX_RECORDS: u64 = 172_800;
const DEFAULT_PRUNE_TO_RECORDS: u64 = 144_000;
const DEFAULT_WARN_SWAP_MIB: u64 = 7_680;
const DEFAULT_ALERT_SWAP_MIB: u64 = 12_288;
const DEFAULT_ALERT_MEM_AVAILABLE_MIB: u64 = 1_024;
const DEFAULT_ALERT_REPEAT: Duration = Duration::from_secs(60);

/// Validated configuration for the telemetry sampler and its local store.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryConfig {
    storage: StorageConfig,
    sampler: SamplerConfig,
    swap_guard: SwapGuardConfig,
}

impl TelemetryConfig {
    /// Reads and validates a telemetry configuration file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, parsed, or validated.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let file = toml::from_str::<FileConfig>(&text).map_err(ConfigError::Parse)?;
        file.resolve()
    }

    /// Returns the bounded local-storage policy.
    #[must_use]
    pub const fn storage(&self) -> &StorageConfig {
        &self.storage
    }

    /// Returns Linux sampler controls.
    #[must_use]
    pub const fn sampler(&self) -> &SamplerConfig {
        &self.sampler
    }

    /// Returns the observer-only swap-alert policy.
    #[must_use]
    pub const fn swap_guard(&self) -> &SwapGuardConfig {
        &self.swap_guard
    }
}

/// Bounded `SQLite` retention configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    sqlite_path: PathBuf,
    max_records: u64,
    prune_to_records: u64,
}

impl StorageConfig {
    /// Builds a bounded local-storage policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the retention targets are not positive or the
    /// lower target exceeds the upper target.
    pub fn new(
        sqlite_path: PathBuf,
        max_records: u64,
        prune_to_records: u64,
    ) -> Result<Self, ConfigError> {
        if max_records == 0 {
            return Err(ConfigError::Invalid(String::from(
                "storage.max_records must be positive",
            )));
        }
        if prune_to_records == 0 || prune_to_records > max_records {
            return Err(ConfigError::Invalid(String::from(
                "storage.prune_to_records must be positive and at most storage.max_records",
            )));
        }
        Ok(Self {
            sqlite_path,
            max_records,
            prune_to_records,
        })
    }

    /// Returns the `SQLite` database path.
    #[must_use]
    pub fn sqlite_path(&self) -> &Path {
        &self.sqlite_path
    }

    /// Returns the maximum retained telemetry rows.
    #[must_use]
    pub const fn max_records(&self) -> u64 {
        self.max_records
    }

    /// Returns the lower record count reached after pruning.
    #[must_use]
    pub const fn prune_to_records(&self) -> u64 {
        self.prune_to_records
    }
}

/// Linux data-source and cadence controls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplerConfig {
    proc_root: PathBuf,
    disk_device: Option<String>,
    gpu_command: Option<PathBuf>,
    gpu_timeout: Duration,
    interval: Duration,
}

impl SamplerConfig {
    /// Returns the procfs root used for memory, load, and disk reads.
    #[must_use]
    pub fn proc_root(&self) -> &Path {
        &self.proc_root
    }

    /// Returns the optional disk device selected from `/proc/diskstats`.
    #[must_use]
    pub fn disk_device(&self) -> Option<&str> {
        self.disk_device.as_deref()
    }

    /// Returns the optional `nvidia-smi` executable.
    #[must_use]
    pub fn gpu_command(&self) -> Option<&Path> {
        self.gpu_command.as_deref()
    }

    /// Returns the bounded duration allowed for an optional GPU probe.
    #[must_use]
    pub const fn gpu_timeout(&self) -> Duration {
        self.gpu_timeout
    }

    /// Returns the target sampling cadence.
    #[must_use]
    pub const fn interval(&self) -> Duration {
        self.interval
    }
}

/// Thresholds and repeat timing for the read-only swap-pressure policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SwapGuardConfig {
    warn_swap_kib: u64,
    alert_swap_kib: u64,
    alert_mem_available_kib: u64,
    alert_repeat: Duration,
}

impl SwapGuardConfig {
    /// Builds a policy from MiB thresholds.
    ///
    /// # Errors
    ///
    /// Returns an error when a threshold is zero, when warning exceeds alert,
    /// or when converting a value to KiB would overflow.
    pub fn new(
        warn_swap_mib: u64,
        alert_swap_mib: u64,
        alert_mem_available_mib: u64,
        alert_repeat: Duration,
    ) -> Result<Self, ConfigError> {
        if warn_swap_mib == 0 || alert_swap_mib == 0 || alert_mem_available_mib == 0 {
            return Err(ConfigError::Invalid(String::from(
                "swap_guard thresholds must be positive",
            )));
        }
        if warn_swap_mib > alert_swap_mib {
            return Err(ConfigError::Invalid(String::from(
                "swap_guard.warn_swap_mib must be at most swap_guard.alert_swap_mib",
            )));
        }
        let to_kib = |value: u64, field: &str| {
            value
                .checked_mul(1024)
                .ok_or_else(|| ConfigError::Invalid(format!("swap_guard.{field} overflows KiB")))
        };
        Ok(Self {
            warn_swap_kib: to_kib(warn_swap_mib, "warn_swap_mib")?,
            alert_swap_kib: to_kib(alert_swap_mib, "alert_swap_mib")?,
            alert_mem_available_kib: to_kib(alert_mem_available_mib, "alert_mem_available_mib")?,
            alert_repeat,
        })
    }

    /// Returns the swap level at which a warning state begins.
    #[must_use]
    pub const fn warn_swap_kib(&self) -> u64 {
        self.warn_swap_kib
    }

    /// Returns the swap level at which an alert begins.
    #[must_use]
    pub const fn alert_swap_kib(&self) -> u64 {
        self.alert_swap_kib
    }

    /// Returns the `MemAvailable` level below which an alert begins.
    #[must_use]
    pub const fn alert_mem_available_kib(&self) -> u64 {
        self.alert_mem_available_kib
    }

    /// Returns the minimum interval between repeated alert evidence records.
    #[must_use]
    pub const fn alert_repeat(&self) -> Duration {
        self.alert_repeat
    }
}

/// Configuration errors are recoverable and identify the rejected contract.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Reading the configuration file failed.
    #[error("read host telemetry config {path}: {source}", path = path.display())]
    Read {
        /// Config file path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// TOML did not match the supported schema.
    #[error("parse host telemetry config: {0}")]
    Parse(toml::de::Error),
    /// A syntactically valid value violates the telemetry contract.
    #[error("invalid host telemetry config: {0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    schema_version: u32,
    #[serde(default)]
    storage: FileStorageConfig,
    #[serde(default)]
    sampler: FileSamplerConfig,
    #[serde(default)]
    swap_guard: FileSwapGuardConfig,
}

impl FileConfig {
    fn resolve(self) -> Result<TelemetryConfig, ConfigError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ConfigError::Invalid(format!(
                "schema_version must be {SCHEMA_VERSION}"
            )));
        }
        Ok(TelemetryConfig {
            storage: self.storage.resolve()?,
            sampler: self.sampler.resolve()?,
            swap_guard: self.swap_guard.resolve()?,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileStorageConfig {
    sqlite_path: Option<PathBuf>,
    max_records: Option<u64>,
    prune_to_records: Option<u64>,
}

impl FileStorageConfig {
    fn resolve(self) -> Result<StorageConfig, ConfigError> {
        StorageConfig::new(
            self.sqlite_path.unwrap_or_else(|| {
                PathBuf::from("~/.local/state/llm-guard-proxy/host-telemetry.sqlite3")
            }),
            self.max_records.unwrap_or(DEFAULT_MAX_RECORDS),
            self.prune_to_records.unwrap_or(DEFAULT_PRUNE_TO_RECORDS),
        )
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSamplerConfig {
    proc_root: Option<PathBuf>,
    disk_device: Option<String>,
    gpu_command: Option<PathBuf>,
    gpu_timeout_secs: Option<u64>,
    interval_secs: Option<u64>,
}

impl FileSamplerConfig {
    fn resolve(self) -> Result<SamplerConfig, ConfigError> {
        let interval = self
            .interval_secs
            .map_or(DEFAULT_INTERVAL, Duration::from_secs);
        if interval.is_zero() {
            return Err(ConfigError::Invalid(String::from(
                "sampler.interval_secs must be positive",
            )));
        }
        if self.disk_device.as_deref().is_some_and(str::is_empty) {
            return Err(ConfigError::Invalid(String::from(
                "sampler.disk_device must not be empty when set",
            )));
        }
        if self.gpu_command.as_ref().is_some_and(|command| {
            !command.is_absolute() || command.file_name() != Some(OsStr::new("nvidia-smi"))
        }) {
            return Err(ConfigError::Invalid(String::from(
                "sampler.gpu_command must be an absolute nvidia-smi path",
            )));
        }
        let gpu_timeout = self
            .gpu_timeout_secs
            .map_or(DEFAULT_GPU_TIMEOUT, Duration::from_secs);
        if gpu_timeout.is_zero() {
            return Err(ConfigError::Invalid(String::from(
                "sampler.gpu_timeout_secs must be positive",
            )));
        }
        Ok(SamplerConfig {
            proc_root: self.proc_root.unwrap_or_else(|| PathBuf::from("/proc")),
            disk_device: self.disk_device.or_else(|| Some(String::from("nvme0n1"))),
            gpu_command: self.gpu_command,
            gpu_timeout,
            interval,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSwapGuardConfig {
    warn_swap_mib: Option<u64>,
    alert_swap_mib: Option<u64>,
    alert_mem_available_mib: Option<u64>,
    alert_repeat_secs: Option<u64>,
}

impl FileSwapGuardConfig {
    fn resolve(self) -> Result<SwapGuardConfig, ConfigError> {
        let alert_repeat = self
            .alert_repeat_secs
            .map_or(DEFAULT_ALERT_REPEAT, Duration::from_secs);
        if alert_repeat.is_zero() {
            return Err(ConfigError::Invalid(String::from(
                "swap_guard.alert_repeat_secs must be positive",
            )));
        }
        SwapGuardConfig::new(
            self.warn_swap_mib.unwrap_or(DEFAULT_WARN_SWAP_MIB),
            self.alert_swap_mib.unwrap_or(DEFAULT_ALERT_SWAP_MIB),
            self.alert_mem_available_mib
                .unwrap_or(DEFAULT_ALERT_MEM_AVAILABLE_MIB),
            alert_repeat,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigError, StorageConfig, SwapGuardConfig};
    use std::{path::PathBuf, time::Duration};

    #[test]
    fn retention_requires_a_positive_lower_target() {
        let error = StorageConfig::new(PathBuf::from("telemetry.sqlite3"), 10, 0)
            .expect_err("zero prune target must fail");
        assert!(matches!(error, ConfigError::Invalid(_)));
    }

    #[test]
    fn swap_warning_must_not_exceed_alert() {
        let error = SwapGuardConfig::new(2, 1, 1, Duration::from_secs(1))
            .expect_err("warning above alert must fail");
        assert!(matches!(error, ConfigError::Invalid(_)));
    }
}
