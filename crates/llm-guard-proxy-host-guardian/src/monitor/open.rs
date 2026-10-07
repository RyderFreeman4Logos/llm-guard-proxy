//! Guardian descriptor initialization; synthetic input is available only to tests.
use super::{
    ConfigHandle, Duration, EmergencyController, EmergencyReserve, File, GuardianError, Instant,
    MemoryGuardian, PathBuf, thresholds_from_policy, tier2,
};

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
            escalation: tier2::GuardianEscalation::default(),
        })
    }

    /// Opens the normal guardian with an explicitly synthetic memory-information descriptor.
    /// This dev-only seam changes neither validation nor cgroup action/admission.
    ///
    /// # Errors
    /// Returns the same initialization errors as [`Self::open`].
    #[cfg(any(test, feature = "synthetic-pressure-test-inputs"))]
    pub fn open_with_meminfo_for_test(
        config_handle: ConfigHandle,
        runtime_dir: impl Into<PathBuf>,
        meminfo: File,
    ) -> Result<Self, GuardianError> {
        let mut guardian = Self::open(config_handle, runtime_dir)?;
        guardian.proc_meminfo = meminfo;
        Ok(guardian)
    }
}
