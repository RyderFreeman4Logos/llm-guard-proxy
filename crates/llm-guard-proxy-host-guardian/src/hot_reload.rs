//! Notify-backed coherent configuration publication.

use crate::config::{ConfigError, GuardianConfig, Thresholds};
use notify::{RecursiveMode, Watcher};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Default)]
struct ReloadSignal {
    changed: AtomicBool,
    error: Mutex<Option<String>>,
}

/// Owns a filesystem watcher and atomically publishes last-good configurations.
///
/// The notify callback never parses, opens a target, or takes emergency action.
/// It only records that the healthy monitoring loop should reload the file.
pub struct HotReloadableConfig {
    config_path: PathBuf,
    live: RwLock<GuardianConfig>,
    signal: Arc<ReloadSignal>,
    _watcher: notify::RecommendedWatcher,
}

impl std::fmt::Debug for HotReloadableConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HotReloadableConfig")
            .field("config_path", &self.config_path)
            .field("thresholds", &self.current())
            .finish_non_exhaustive()
    }
}

impl HotReloadableConfig {
    /// Loads a complete initial snapshot and watches its parent directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the initial configuration is invalid or the
    /// parent directory cannot be watched.
    pub fn new(config_path: impl Into<PathBuf>) -> Result<Self, ConfigError> {
        let config_path = config_path.into();
        let initial = GuardianConfig::load(&config_path)?;
        let parent = config_path.parent().ok_or_else(|| {
            ConfigError::Invalid(format!(
                "guardian config path has no parent: {}",
                config_path.display()
            ))
        })?;
        let signal = Arc::new(ReloadSignal::default());
        let callback_signal = Arc::clone(&signal);
        let watched_path = config_path.clone();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
                Ok(event) if event.paths.iter().any(|path| path == &watched_path) => {
                    callback_signal.changed.store(true, Ordering::Release);
                }
                Ok(_) => {}
                Err(error) => {
                    let mut slot = callback_signal
                        .error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *slot = Some(error.to_string());
                }
            })
            .map_err(|error| ConfigError::Watch(error.to_string()))?;
        watcher
            .watch(parent, RecursiveMode::NonRecursive)
            .map_err(|error| ConfigError::Watch(error.to_string()))?;
        Ok(Self {
            config_path,
            live: RwLock::new(initial),
            signal,
            _watcher: watcher,
        })
    }

    /// Returns a coherent snapshot without filesystem access.
    #[must_use]
    pub fn current(&self) -> GuardianConfig {
        self.live
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Returns the coherent threshold pair without filesystem access.
    #[must_use]
    pub fn thresholds(&self) -> Thresholds {
        self.live
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .thresholds()
    }

    /// Reads and validates a pending candidate in the healthy loop.
    ///
    /// Invalid candidates are returned to the caller without changing live
    /// values, so a partially written or invalid file cannot disarm Tier 1.
    ///
    /// # Errors
    ///
    /// Returns watcher or candidate-validation failures while retaining the
    /// last known-good live configuration.
    pub fn reload_if_changed(&self) -> Result<Option<GuardianConfig>, ConfigError> {
        let error = self
            .signal
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(error) = error {
            return Err(ConfigError::Watch(error));
        }
        if !self.signal.changed.swap(false, Ordering::AcqRel) {
            return Ok(None);
        }
        let candidate = GuardianConfig::load(&self.config_path)?;
        let mut live = self
            .live
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *live == candidate {
            return Ok(None);
        }
        *live = candidate.clone();
        Ok(Some(candidate))
    }

    #[cfg(test)]
    pub(crate) fn mark_changed_for_test(&self) {
        self.signal.changed.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::HotReloadableConfig;
    use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, time::SystemTime};

    fn temp_config() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("guardian-hot-reload-{nonce}.toml"))
    }

    fn config(threshold: u64) -> String {
        format!(
            "schema_version = 1\n[target]\nlabel = \"test\"\nregistration_file = \"target.v1\"\n[thresholds]\nmem_avail_stop_gib = {threshold}\nreserve_mib = 64\n"
        )
    }

    fn secure(path: &std::path::Path) {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).expect("secure config");
    }

    #[test]
    fn publishes_new_coherent_snapshot() {
        let path = temp_config();
        fs::write(&path, config(1)).expect("write config");
        secure(&path);
        let watcher = HotReloadableConfig::new(&path).expect("watcher");
        fs::write(&path, config(3)).expect("replace config");
        secure(&path);
        watcher.mark_changed_for_test();
        assert_eq!(
            watcher
                .reload_if_changed()
                .expect("reload")
                .expect("changed")
                .thresholds()
                .mem_avail_stop_gib(),
            3
        );
        fs::remove_file(path).expect("remove config");
    }

    #[test]
    fn leaves_last_good_snapshot_after_invalid_reload() {
        let path = temp_config();
        fs::write(&path, config(1)).expect("write config");
        secure(&path);
        let watcher = HotReloadableConfig::new(&path).expect("watcher");
        fs::write(&path, "schema_version = 2").expect("replace config");
        watcher.mark_changed_for_test();
        assert!(watcher.reload_if_changed().is_err());
        assert_eq!(watcher.thresholds().mem_avail_stop_gib(), 1);
        fs::remove_file(path).expect("remove config");
    }

    #[test]
    fn unchanged_candidate_is_not_republished() {
        let path = temp_config();
        fs::write(&path, config(1)).expect("write config");
        secure(&path);
        let watcher = HotReloadableConfig::new(&path).expect("watcher");
        watcher.mark_changed_for_test();
        assert!(watcher.reload_if_changed().expect("reload").is_none());
        fs::remove_file(path).expect("remove config");
    }
}
