#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

//! GB10-derived host-memory guardian.
//!
//! The runtime policy comes from the proxy's shared, hot-reloadable config.
//! Deployments may select the pre-opened, allocation-free `cgroup.kill` path
//! or a bounded `systemctl --user restart` action.

pub mod config;
pub mod emergency;
pub mod escalation;
pub mod monitor;

pub use config::{
    ConfigError, EscalationConfig, GuardianConfig, RuntimeConfig, TargetConfig, Thresholds,
};
pub use emergency::{EmergencyController, EmergencyReserve, kill_direct};
pub use escalation::{EscalationEpisode, EscalationError};
pub use monitor::{
    CgroupTarget, GuardianError, GuardianIteration, MemInfoError, MemoryGuardian, Registration,
    RegistrationError, default_runtime_dir, parse_mem_available, parse_registration, should_rearm,
    should_shed,
};
