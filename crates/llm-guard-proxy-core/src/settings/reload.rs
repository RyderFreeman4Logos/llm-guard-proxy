use std::sync::{Arc, RwLock};

use super::{AppConfig, GuardianConfig, RestartRequiredChange};

/// Failure to access the shared configuration snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ConfigHandleError {
    /// Shared config state was poisoned by a panic.
    #[error("config state lock is poisoned")]
    LockPoisoned,
}

/// Thread-safe handle used by request-serving code to read current settings.
#[derive(Clone, Debug)]
pub struct ConfigHandle {
    current: Arc<RwLock<AppConfig>>,
}

impl ConfigHandle {
    /// Creates a shared handle from a validated configuration.
    #[must_use]
    pub fn new(config: AppConfig) -> Self {
        Self {
            current: Arc::new(RwLock::new(config)),
        }
    }

    /// Returns a point-in-time copy of the validated config.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigHandleError::LockPoisoned`] if another thread panicked
    /// while mutating the config.
    pub fn snapshot(&self) -> Result<AppConfig, ConfigHandleError> {
        let guard = self
            .current
            .read()
            .map_err(|_error| ConfigHandleError::LockPoisoned)?;
        Ok(guard.clone())
    }

    /// Returns only the host guardian policy from the current coherent snapshot.
    ///
    /// This avoids cloning unrelated routing, evidence, and workflow state from
    /// the guardian's one-second healthy-path configuration check.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigHandleError::LockPoisoned`] if another thread panicked while
    /// holding the configuration lock.
    pub fn guardian_snapshot(&self) -> Result<GuardianConfig, ConfigHandleError> {
        let guard = self
            .current
            .read()
            .map_err(|_error| ConfigHandleError::LockPoisoned)?;
        Ok(guard.guardian.clone())
    }

    /// Atomically applies the reloadable portion of a validated config.
    ///
    /// Restart-required changes are reported but remain unchanged in the live
    /// snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigHandleError::LockPoisoned`] if another thread panicked
    /// while mutating the config.
    pub fn apply_reloadable(
        &self,
        requested: &AppConfig,
    ) -> Result<ReloadOutcome, ConfigHandleError> {
        let mut current = self
            .current
            .write()
            .map_err(|_error| ConfigHandleError::LockPoisoned)?;
        let (next, outcome) = apply_reloadable(&current, requested);
        *current = next;
        Ok(outcome)
    }
}

/// Applies only live-reloadable settings without performing I/O.
///
/// The returned configuration preserves every restart-required value from
/// `current`; the outcome reports those requested changes to the service.
#[must_use]
pub fn apply_reloadable(current: &AppConfig, requested: &AppConfig) -> (AppConfig, ReloadOutcome) {
    let restart_required_changes = current.restart_required_changes(requested);
    let mut next = current.clone();
    next.apply_reloadable_from(requested);
    let applied = next != *current;
    (
        next,
        ReloadOutcome {
            applied,
            restart_required_changes,
        },
    )
}

/// Result of applying one validated hot-reload candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReloadOutcome {
    /// True when at least one live reloadable setting changed.
    pub applied: bool,
    /// Restart-required changes detected but not applied.
    pub restart_required_changes: Vec<RestartRequiredChange>,
}
