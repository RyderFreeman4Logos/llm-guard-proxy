//! Filesystem-backed configuration loading and polling for the service.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
    sync::{Arc, RwLock, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

use llm_guard_proxy_core::{
    AppConfig, ConfigHandle, ConfigHandleError, ConfigParseError, ReloadOutcome, ValidationError,
};
use llm_guard_proxy_state::{materialize_evidence_path_defaults, preflight_evidence_paths};

const DEFAULT_CONFIG_RELATIVE_PATH: &str = ".config/llm-guard-proxy/config.toml";

/// Missing-file behavior for a configuration source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingConfigPolicy {
    /// Missing config files are treated as the built-in defaults.
    UseDefaults,
    /// Missing config files are errors.
    RequireFile,
}

/// Failure while loading or polling a service configuration source.
#[derive(Debug, thiserror::Error)]
pub enum ConfigReloadError {
    /// The default path could not be resolved.
    #[error("could not determine home directory for default config path")]
    HomeDirectoryUnavailable,
    /// Reading the config file failed.
    #[error("failed to read config {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    /// Parsing TOML failed.
    #[error("failed to parse config {path}: {source}")]
    Parse {
        path: PathBuf,
        source: ConfigParseError,
    },
    /// Parsed config failed validation.
    #[error("invalid config {path}: {source}")]
    Invalid {
        path: PathBuf,
        source: ValidationError,
    },
    /// Shared config or reload-health state was poisoned by a panic.
    #[error("config state lock is poisoned")]
    LockPoisoned,
    /// The hot reload poll interval was zero.
    #[error("reload poll interval must be greater than zero")]
    EmptyReloadInterval,
    /// The hot reload thread could not start.
    #[error("failed to start config reload watcher for {path}: {source}")]
    WatcherStart { path: PathBuf, source: io::Error },
}

impl From<ConfigHandleError> for ConfigReloadError {
    fn from(_error: ConfigHandleError) -> Self {
        Self::LockPoisoned
    }
}

/// Filesystem-backed config source with reload health tracking.
#[derive(Clone, Debug)]
pub struct ConfigManager {
    path: PathBuf,
    missing_policy: MissingConfigPolicy,
    handle: ConfigHandle,
    last_error: Arc<RwLock<Option<String>>>,
}

impl ConfigManager {
    /// Loads the default config path, using built-in defaults when absent.
    pub(crate) fn from_default_path() -> Result<Self, ConfigReloadError> {
        let path = default_config_path()?;
        Self::from_path_with_policy(path, MissingConfigPolicy::UseDefaults)
    }

    /// Loads an explicit config file path, requiring the file to exist.
    pub(crate) fn from_explicit_path(path: impl Into<PathBuf>) -> Result<Self, ConfigReloadError> {
        Self::from_path_with_policy(path, MissingConfigPolicy::RequireFile)
    }

    fn from_path_with_policy(
        path: impl Into<PathBuf>,
        missing_policy: MissingConfigPolicy,
    ) -> Result<Self, ConfigReloadError> {
        let path = path.into();
        let config = load_config(&path, missing_policy)?;
        Ok(Self {
            path,
            missing_policy,
            handle: ConfigHandle::new(config),
            last_error: Arc::new(RwLock::new(None)),
        })
    }

    /// Returns the source path used by this manager.
    #[must_use]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the shared live configuration handle.
    #[must_use]
    pub(crate) fn handle(&self) -> ConfigHandle {
        self.handle.clone()
    }

    /// Returns the most recent background reload error, if any.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigReloadError::LockPoisoned`] when the reload-health lock
    /// was poisoned by a panic.
    pub fn last_error(&self) -> Result<Option<String>, ConfigReloadError> {
        let guard = self
            .last_error
            .read()
            .map_err(|_error| ConfigReloadError::LockPoisoned)?;
        Ok(guard.clone())
    }

    /// Reloads the source and atomically applies its reloadable settings.
    pub(crate) fn reload(&self) -> Result<ReloadOutcome, ConfigReloadError> {
        let requested = load_config(&self.path, self.missing_policy).inspect_err(|error| {
            self.set_last_error(Some(error.to_string()));
        })?;
        let outcome = self.handle.apply_reloadable(&requested)?;
        self.set_last_error(None);
        Ok(outcome)
    }

    /// Starts a background polling watcher for this source.
    pub(crate) fn spawn_polling(
        &self,
        interval: Duration,
    ) -> Result<ReloadWatcher, ConfigReloadError> {
        if interval.is_zero() {
            return Err(ConfigReloadError::EmptyReloadInterval);
        }

        let manager = self.clone();
        let path = self.path.clone();
        let (stop_tx, stop_rx) = mpsc::channel();
        let thread = thread::Builder::new()
            .name(String::from("llm-guard-proxy-config-reload"))
            .spawn(move || poll_reloads(&manager, &stop_rx, interval))
            .map_err(|source| ConfigReloadError::WatcherStart { path, source })?;

        Ok(ReloadWatcher {
            stop_tx: Some(stop_tx),
            thread: Some(thread),
        })
    }

    fn set_last_error(&self, value: Option<String>) {
        if let Ok(mut guard) = self.last_error.write() {
            *guard = value;
        }
    }
}

/// RAII guard for the service config polling thread.
#[derive(Debug)]
pub struct ReloadWatcher {
    stop_tx: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ReloadWatcher {
    /// Stops the watcher and waits for its thread to exit.
    ///
    /// # Errors
    ///
    /// Returns the panic payload if the polling thread panicked.
    pub fn stop(mut self) -> thread::Result<()> {
        self.request_stop();
        self.join_thread()
    }

    fn request_stop(&mut self) {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _sent = stop_tx.send(());
        }
    }

    fn join_thread(&mut self) -> thread::Result<()> {
        match self.thread.take() {
            Some(thread) => thread.join(),
            None => Ok(()),
        }
    }
}

impl Drop for ReloadWatcher {
    fn drop(&mut self) {
        self.request_stop();
        let _ = self.join_thread();
    }
}

fn default_config_path() -> Result<PathBuf, ConfigReloadError> {
    default_config_path_from_home(env::var_os("HOME"))
}

fn default_config_path_from_home(
    home: Option<std::ffi::OsString>,
) -> Result<PathBuf, ConfigReloadError> {
    let Some(home) = home else {
        return Err(ConfigReloadError::HomeDirectoryUnavailable);
    };
    Ok(PathBuf::from(home).join(DEFAULT_CONFIG_RELATIVE_PATH))
}

fn load_config(
    path: &Path,
    missing_policy: MissingConfigPolicy,
) -> Result<AppConfig, ConfigReloadError> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(source)
            if source.kind() == io::ErrorKind::NotFound
                && missing_policy == MissingConfigPolicy::UseDefaults =>
        {
            let mut config = AppConfig::default();
            materialize_evidence_path_defaults(&mut config);
            validate_config(path, &config)?;
            return Ok(config);
        }
        Err(source) => {
            return Err(ConfigReloadError::Read {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    let mut defaults = AppConfig::default();
    materialize_evidence_path_defaults(&mut defaults);
    let config = AppConfig::parse_with_defaults(&contents, defaults).map_err(|source| {
        ConfigReloadError::Parse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    validate_config(path, &config)?;
    Ok(config)
}

fn validate_config(path: &Path, config: &AppConfig) -> Result<(), ConfigReloadError> {
    config
        .validate()
        .and_then(|()| preflight_evidence_paths(config))
        .map_err(|source| ConfigReloadError::Invalid {
            path: path.to_path_buf(),
            source,
        })
}

fn poll_reloads(manager: &ConfigManager, stop_rx: &mpsc::Receiver<()>, interval: Duration) {
    loop {
        match stop_rx.recv_timeout(interval) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Err(error) = manager.reload() {
                    manager.set_last_error(Some(error.to_string()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::{
        ffi::OsString,
        fs,
        path::{Path, PathBuf},
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use llm_guard_proxy_core::{AppConfig, GuardianKillAction, HeartbeatMode};
    use llm_guard_proxy_state::materialize_evidence_path_defaults;

    use super::{
        ConfigManager, ConfigReloadError, MissingConfigPolicy, default_config_path_from_home,
    };

    #[test]
    fn resolves_default_path_from_home() {
        let path = default_config_path_from_home(Some(OsString::from("/home/test")))
            .expect("home path should resolve");
        assert_eq!(
            path,
            Path::new("/home/test/.config/llm-guard-proxy/config.toml")
        );
        assert!(matches!(
            default_config_path_from_home(None),
            Err(ConfigReloadError::HomeDirectoryUnavailable)
        ));
    }

    #[test]
    fn missing_file_policy_distinguishes_default_and_explicit_sources() {
        let path = unique_test_path("missing.toml");
        let manager = ConfigManager::from_path_with_policy(&path, MissingConfigPolicy::UseDefaults)
            .expect("missing default source should use defaults");
        let mut expected = AppConfig::default();
        materialize_evidence_path_defaults(&mut expected);
        assert_eq!(manager.handle().snapshot().expect("snapshot"), expected);

        let error = ConfigManager::from_explicit_path(&path)
            .expect_err("missing explicit source should fail");
        assert!(matches!(error, ConfigReloadError::Read { .. }));
    }

    #[test]
    fn source_errors_retain_path_and_failure_kind() {
        let parse_path = unique_test_path("parse.toml");
        fs::write(&parse_path, "not toml").expect("write parse fixture");
        let parse_error =
            ConfigManager::from_explicit_path(&parse_path).expect_err("invalid syntax should fail");
        assert!(matches!(parse_error, ConfigReloadError::Parse { .. }));

        let invalid_path = unique_test_path("invalid.toml");
        fs::write(&invalid_path, "[server]\nport = 0\n").expect("write validation fixture");
        let invalid_error = ConfigManager::from_explicit_path(&invalid_path)
            .expect_err("invalid config should fail");
        assert!(matches!(invalid_error, ConfigReloadError::Invalid { .. }));

        remove_file(&parse_path);
        remove_file(&invalid_path);
    }

    #[test]
    fn reload_preserves_restart_fields_and_tracks_health() {
        let path = unique_test_path("reload.toml");
        fs::write(
            &path,
            "[server]\nport = 18009\nmax_in_flight_requests = 4\n",
        )
        .expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");

        fs::write(
            &path,
            "[server]\nport = 19000\nmax_in_flight_requests = 2\n",
        )
        .expect("write reload config");
        let outcome = manager.reload().expect("reload should succeed");
        let snapshot = manager.handle().snapshot().expect("snapshot");
        assert!(outcome.applied);
        assert_eq!(outcome.restart_required_changes.len(), 1);
        assert_eq!(snapshot.server.port, 18_009);
        assert_eq!(snapshot.server.max_in_flight_requests, 2);
        assert_eq!(manager.last_error().expect("reload health"), None);

        fs::write(&path, "not toml").expect("write broken reload");
        manager.reload().expect_err("broken reload should fail");
        assert!(manager.last_error().expect("reload health").is_some());
        assert_eq!(manager.handle().snapshot().expect("snapshot"), snapshot);

        fs::write(
            &path,
            "[server]\nport = 18009\nmax_in_flight_requests = 3\n",
        )
        .expect("write recovered reload");
        manager.reload().expect("recovered reload should succeed");
        assert_eq!(manager.last_error().expect("reload health"), None);
        remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn initial_load_rejects_unsafe_evidence_path() {
        let root = unique_test_path("initial-unsafe-evidence");
        fs::create_dir_all(&root).expect("create unsafe evidence parent");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))
            .expect("set unsafe evidence permissions");
        let path = unique_test_path("initial-unsafe.toml");
        fs::write(
            &path,
            format!(
                "[evidence]\nsqlite_path = \"{}\"\nblob_cache_dir = \"{}\"\n",
                root.join("evidence.sqlite3").display(),
                root.join("blobs").display()
            ),
        )
        .expect("write unsafe config");

        let error = ConfigManager::from_explicit_path(&path)
            .expect_err("initial preflight should reject unsafe parent");
        assert!(matches!(error, ConfigReloadError::Invalid { .. }));

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("restore safe evidence permissions");
        remove_file(&path);
        remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn reload_preflight_failure_preserves_snapshot_and_health_recovers() {
        let path = unique_test_path("preflight-reload.toml");
        fs::write(&path, "[heartbeat]\ninterval_secs = 15\n").expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        let root = unique_test_path("reload-unsafe-evidence");
        fs::create_dir_all(&root).expect("create unsafe evidence parent");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))
            .expect("set unsafe evidence permissions");
        fs::write(
            &path,
            format!(
                "[heartbeat]\ninterval_secs = 4\n[evidence]\nsqlite_path = \"{}\"\nblob_cache_dir = \"{}\"\n",
                root.join("evidence.sqlite3").display(),
                root.join("blobs").display()
            ),
        )
        .expect("write unsafe reload config");

        manager.reload().expect_err("unsafe reload should fail");
        assert_eq!(manager.handle().snapshot().expect("snapshot"), before);
        assert!(manager.last_error().expect("reload health").is_some());

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("make evidence parent safe");
        let outcome = manager.reload().expect("safe reload should recover");
        assert!(outcome.applied);
        assert_eq!(
            manager
                .handle()
                .snapshot()
                .expect("recovered snapshot")
                .heartbeat
                .interval_secs,
            4
        );
        assert_eq!(manager.last_error().expect("reload health"), None);

        remove_file(&path);
        remove_dir_all(&root);
    }

    #[test]
    fn polling_watcher_applies_reloadable_changes() {
        let path = unique_test_path("polling.toml");
        fs::write(&path, "[heartbeat]\nmode = \"sse\"\ninterval_secs = 15\n")
            .expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let handle = manager.handle();
        let watcher = manager
            .spawn_polling(Duration::from_millis(10))
            .expect("start watcher");

        fs::write(
            &path,
            "[heartbeat]\nmode = \"disabled\"\ninterval_secs = 4\n",
        )
        .expect("write reload config");
        let mut observed = false;
        for _attempt in 0..50 {
            let snapshot = handle.snapshot().expect("snapshot");
            if snapshot.heartbeat.mode == HeartbeatMode::Disabled
                && snapshot.heartbeat.interval_secs == 4
            {
                observed = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(observed, "polling watcher should apply reload");

        fs::write(&path, "not toml").expect("write broken polling config");
        let mut observed_error = false;
        for _attempt in 0..50 {
            if manager.last_error().expect("reload health").is_some() {
                observed_error = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            observed_error,
            "polling failure should update reload health"
        );

        fs::write(&path, "[heartbeat]\nmode = \"sse\"\ninterval_secs = 3\n")
            .expect("write recovered polling config");
        let mut recovered = false;
        for _attempt in 0..50 {
            let snapshot = handle.snapshot().expect("snapshot");
            if snapshot.heartbeat.interval_secs == 3
                && manager.last_error().expect("reload health").is_none()
            {
                recovered = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            recovered,
            "successful polling reload should clear health error"
        );

        watcher.stop().expect("watcher should stop");
        remove_file(&path);
    }

    #[test]
    fn polling_watcher_hot_reloads_guardian_policy_and_retains_last_good() {
        let path = unique_test_path("guardian-polling.toml");
        fs::write(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"aeon-text\"\nmem_threshold_gib = 2\nkill_action = \"cgroup.kill\"\npoll_interval_secs = 1\nregistration_file = \"text-cgroup.v1\"\n",
        )
        .expect("write initial guardian config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let handle = manager.handle();
        let watcher = manager
            .spawn_polling(Duration::from_millis(10))
            .expect("start watcher");

        fs::write(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"replacement\"\nmem_threshold_gib = 5\nkill_action = \"systemctl_restart\"\npoll_interval_secs = 4\nsystemd_unit = \"replacement.service\"\n",
        )
        .expect("write replacement guardian config");
        let mut observed = false;
        for _attempt in 0..50 {
            let snapshot = handle.snapshot().expect("snapshot");
            if snapshot.guardian.target_label == "replacement"
                && snapshot.guardian.mem_threshold_gib == 5
                && snapshot.guardian.kill_action == GuardianKillAction::SystemctlRestart
                && snapshot.guardian.poll_interval_secs == 4
            {
                observed = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(observed, "guardian policy should be hot reloaded");
        let last_good = handle.snapshot().expect("last-good snapshot");

        fs::write(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"\"\nmem_threshold_gib = 0\n",
        )
        .expect("write invalid guardian config");
        let mut observed_error = false;
        for _attempt in 0..50 {
            if manager.last_error().expect("reload health").is_some() {
                observed_error = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            observed_error,
            "invalid policy should surface reload health"
        );
        assert_eq!(
            handle.snapshot().expect("retained snapshot").guardian,
            last_good.guardian
        );

        watcher.stop().expect("watcher should stop");
        remove_file(&path);
    }

    #[test]
    fn rejects_zero_poll_interval() {
        let path = unique_test_path("zero-interval.toml");
        fs::write(&path, "").expect("write config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load config");
        assert!(matches!(
            manager.spawn_polling(Duration::ZERO),
            Err(ConfigReloadError::EmptyReloadInterval)
        ));
        remove_file(&path);
    }

    fn unique_test_path(file_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("llm-guard-proxy-config-reload-{nanos}-{file_name}"))
    }

    fn remove_file(path: &Path) {
        if let Err(error) = fs::remove_file(path) {
            assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        }
    }

    fn remove_dir_all(path: &Path) {
        if let Err(error) = fs::remove_dir_all(path) {
            assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        }
    }
}
