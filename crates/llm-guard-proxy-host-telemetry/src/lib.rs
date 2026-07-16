#![forbid(unsafe_code)]

//! Bounded observer-only host telemetry for GB10-compatible Linux hosts.
//!
//! The Linux implementation reads procfs and an optional closed GPU backend,
//! then persists numeric samples and alert evidence. It has no process, cgroup,
//! systemd, or service-control API.

pub mod config;
mod model;
mod policy;
mod store;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(not(target_os = "linux"))]
mod stub;

pub use config::{
    ConfigError, GpuBackend, SamplerConfig, StorageConfig, SwapGuardConfig, TelemetryConfig,
};
#[cfg(target_os = "linux")]
pub use linux::HostTelemetry;
#[cfg(target_os = "linux")]
pub use linux::{
    SamplerError, derive_disk_rate, parse_diskstats, parse_gpu_csv, parse_loadavg, parse_meminfo,
};
pub use model::{
    DiskCounters, DiskRate, GpuSample, HostSample, LoadAverage, MemorySample, PolicyDecision,
    PressureReason, TelemetryEvent, TelemetryIteration, TelemetryState,
};
pub use policy::SwapGuard;
pub use store::{TelemetryStore, TelemetryStoreError};
#[cfg(not(target_os = "linux"))]
pub use stub::HostTelemetry;

/// Top-level host telemetry errors.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// Configuration is unavailable or invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The host sampler could not read a mandatory source.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Sample(#[from] SamplerError),
    /// Bounded local persistence failed.
    #[error(transparent)]
    Store(#[from] TelemetryStoreError),
}
