//! GB10-derived two-tier host-memory guardian.
//!
//! Tier 1 is always the pre-opened, allocation-free `cgroup.kill` action.
//! Tier 2 systemd escalation is compiled in but disabled unless operators
//! explicitly enable it in the guardian configuration.

pub mod config;
pub mod escalation;
pub mod hot_reload;
pub mod monitor;

pub use config::{
    ConfigError, EscalationConfig, GuardianConfig, RuntimeConfig, TargetConfig, Thresholds,
};
pub use escalation::{EscalationEpisode, EscalationError};
pub use hot_reload::HotReloadableConfig;
pub use monitor::{
    CgroupTarget, GuardianError, GuardianIteration, MemInfoError, MemoryGuardian, Registration,
    RegistrationError, parse_mem_available, parse_registration, should_rearm, should_shed,
};
