use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

use super::{AppConfig, GuardianConfig, RestartRequiredChange, ValidationError};

/// Failure to access the shared configuration snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ConfigHandleError {
    /// Shared config state was poisoned by a panic.
    #[error("config state lock is poisoned")]
    LockPoisoned,
    /// Configuration ownership revisions must never wrap.
    #[error("config revision exhausted")]
    RevisionExhausted,
}

/// Thread-safe handle used by request-serving code to read current settings.
#[derive(Clone, Debug)]
pub struct ConfigHandle {
    current: Arc<RwLock<AppConfig>>,
    revision: Arc<AtomicU64>,
}

impl ConfigHandle {
    /// Creates a shared handle from a validated configuration.
    #[must_use]
    pub fn new(config: AppConfig) -> Self {
        Self {
            current: Arc::new(RwLock::new(config)),
            revision: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the monotonic installed-policy revision (including A→B→A changes).
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// Dispatches synchronously under the config read lock only at this revision.
    /// The callback must not reload configuration or await.
    pub fn at_revision<T>(&self, revision: u64, action: impl FnOnce() -> T) -> Option<T> {
        let _snapshot = self.current.read().ok()?;
        (self.revision() == revision).then(action)
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

    /// Atomically applies the reloadable portion of a config candidate.
    ///
    /// Restart-required changes are reported but remain unchanged in the live
    /// snapshot. Invalid projected snapshots are rejected without publishing.
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
        if outcome.applied {
            let next_revision = self
                .revision()
                .checked_add(1)
                .ok_or(ConfigHandleError::RevisionExhausted)?;
            self.revision.store(next_revision, Ordering::Release);
        }
        *current = next;
        Ok(outcome)
    }
}

/// Applies only live-reloadable settings without performing I/O.
///
/// The returned configuration preserves every restart-required value from
/// `current`; the outcome reports those requested changes to the service.
/// Invalid projected snapshots are not published.
#[must_use]
pub fn apply_reloadable(current: &AppConfig, requested: &AppConfig) -> (AppConfig, ReloadOutcome) {
    let restart_required_changes = current.restart_required_changes(requested);
    let mut next = current.clone();
    next.apply_reloadable_from(requested);
    if let Err(rejection) = next.validate() {
        return (
            current.clone(),
            ReloadOutcome {
                applied: false,
                restart_required_changes,
                rejection: Some(rejection),
            },
        );
    }
    let applied = next != *current;
    (
        next,
        ReloadOutcome {
            applied,
            restart_required_changes,
            rejection: None,
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
    /// Validation failure in the projected live snapshot.
    pub rejection: Option<ValidationError>,
}
