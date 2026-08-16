//! Filesystem-backed configuration loading and polling for the service.

use std::{
    env,
    fs::{self, File, Metadata},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use llm_guard_proxy_core::{
    AppConfig, ConfigHandle, ConfigHandleError, ConfigParseError, ReloadOutcome,
    RestartRequiredChange, ValidationError, apply_reloadable,
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
    /// A reload source was empty after a config had already been loaded.
    #[error("empty config generation at {path}")]
    EmptyGeneration { path: PathBuf },
    /// A reload modified the previously accepted file in place.
    #[error("in-place config update rejected at {path}; publish changes by atomic replacement")]
    InPlaceUpdate { path: PathBuf },
    /// The source identity or metadata changed while it was being read.
    #[error("unstable config generation rejected at {path}")]
    UnstableGeneration { path: PathBuf },
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

/// Terminal result of one config reload attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReloadTerminalState {
    Applied,
    NoChange,
    RestartPending,
    Rejected(String),
}

/// Generation-bound reload result paired with its exact live snapshot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReloadStatus {
    pub(crate) attempt_generation: u64,
    pub(crate) snapshot_generation: u64,
    pub(crate) snapshot: AppConfig,
    pub(crate) terminal: ReloadTerminalState,
    pub(crate) restart_required_changes: Vec<RestartRequiredChange>,
}

/// Filesystem-backed config source with reload health tracking.
#[derive(Clone, Debug)]
pub struct ConfigManager {
    path: PathBuf,
    handle: ConfigHandle,
    reload_status: Arc<RwLock<Option<ReloadStatus>>>,
    source_generation: Arc<Mutex<Option<SourceGeneration>>>,
    observed_generation: Arc<Mutex<Option<SourceGeneration>>>,
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
        let (config, source_generation) = load_initial_config(&path, missing_policy)?;
        Ok(Self {
            path,
            handle: ConfigHandle::new(config),
            reload_status: Arc::new(RwLock::new(None)),
            source_generation: Arc::new(Mutex::new(source_generation)),
            observed_generation: Arc::new(Mutex::new(source_generation)),
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
        Ok(self
            .reload_status()?
            .and_then(|status| match status.terminal {
                ReloadTerminalState::Rejected(reason) => Some(reason),
                ReloadTerminalState::Applied
                | ReloadTerminalState::NoChange
                | ReloadTerminalState::RestartPending => None,
            }))
    }

    /// Returns the latest reload result and the exact snapshot it describes.
    pub(crate) fn reload_status(&self) -> Result<Option<ReloadStatus>, ConfigReloadError> {
        let guard = self
            .reload_status
            .read()
            .map_err(|_error| ConfigReloadError::LockPoisoned)?;
        Ok(guard.clone())
    }

    /// Reloads the source and atomically applies its reloadable settings.
    pub(crate) fn reload(&self) -> Result<ReloadOutcome, ConfigReloadError> {
        let mut source_generation = self
            .source_generation
            .lock()
            .map_err(|_error| ConfigReloadError::LockPoisoned)?;
        let mut observed_generation = self
            .observed_generation
            .lock()
            .map_err(|_error| ConfigReloadError::LockPoisoned)?;
        let (requested, requested_generation) =
            match load_reload_config(&self.path, &mut observed_generation) {
                Ok(candidate) => candidate,
                Err(error) => {
                    self.record_status(
                        self.handle.snapshot()?,
                        ReloadTerminalState::Rejected(error.to_string()),
                        Vec::new(),
                    )?;
                    return Err(error);
                }
            };
        let current = self.handle.snapshot()?;
        let (projected, mut outcome) = apply_reloadable(&current, &requested);
        if outcome.rejection.is_none() {
            outcome.rejection = preflight_evidence_paths(&projected).err();
        }
        if outcome.rejection.is_none() {
            outcome = self.handle.apply_reloadable(&requested)?;
            *source_generation = Some(requested_generation);
        }
        let snapshot = self.handle.snapshot()?;
        let terminal = if let Some(rejection) = &outcome.rejection {
            ReloadTerminalState::Rejected(rejection.to_string())
        } else if outcome.applied {
            ReloadTerminalState::Applied
        } else if outcome.restart_required_changes.is_empty() {
            ReloadTerminalState::NoChange
        } else {
            ReloadTerminalState::RestartPending
        };
        self.record_status(snapshot, terminal, outcome.restart_required_changes.clone())?;
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

    fn record_status(
        &self,
        snapshot: AppConfig,
        terminal: ReloadTerminalState,
        restart_required_changes: Vec<RestartRequiredChange>,
    ) -> Result<(), ConfigReloadError> {
        let mut guard = self
            .reload_status
            .write()
            .map_err(|_error| ConfigReloadError::LockPoisoned)?;
        let attempt_generation = guard
            .as_ref()
            .map_or(1, |status| status.attempt_generation + 1);
        let snapshot_generation = if matches!(terminal, ReloadTerminalState::Rejected(_)) {
            guard
                .as_ref()
                .map_or(0, |status| status.snapshot_generation)
        } else {
            attempt_generation
        };
        *guard = Some(ReloadStatus {
            attempt_generation,
            snapshot_generation,
            snapshot,
            terminal,
            restart_required_changes,
        });
        Ok(())
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

fn load_initial_config(
    path: &Path,
    missing_policy: MissingConfigPolicy,
) -> Result<(AppConfig, Option<SourceGeneration>), ConfigReloadError> {
    let source = match File::open(path) {
        Ok(source) => source,
        Err(source)
            if source.kind() == io::ErrorKind::NotFound
                && missing_policy == MissingConfigPolicy::UseDefaults =>
        {
            let mut config = AppConfig::default();
            materialize_evidence_path_defaults(&mut config);
            validate_config(path, &config)?;
            return Ok((config, None));
        }
        Err(source) => {
            return Err(ConfigReloadError::Read {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let (contents, generation) = read_stable_source(path, source, None)?;
    Ok((
        parse_config(path, &decode_config_source(path, contents)?)?,
        Some(generation),
    ))
}

fn load_reload_config(
    path: &Path,
    previous: &mut Option<SourceGeneration>,
) -> Result<(AppConfig, SourceGeneration), ConfigReloadError> {
    let source = File::open(path).map_err(|source| ConfigReloadError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let (contents, generation) = read_stable_source(path, source, Some(previous))?;
    let in_place_update = previous
        .as_ref()
        .is_some_and(|observed| observed.same_identity(&generation) && observed != &generation);
    if in_place_update {
        return Err(ConfigReloadError::InPlaceUpdate {
            path: path.to_path_buf(),
        });
    }
    if previous
        .as_ref()
        .is_none_or(|observed| !observed.same_identity(&generation))
    {
        *previous = Some(generation);
    }
    let contents = decode_config_source(path, contents)?;
    if contents.trim().is_empty() {
        return Err(ConfigReloadError::EmptyGeneration {
            path: path.to_path_buf(),
        });
    }
    Ok((parse_config(path, &contents)?, generation))
}
fn read_stable_source(
    path: &Path,
    mut source: File,
    observed: Option<&mut Option<SourceGeneration>>,
) -> Result<(Vec<u8>, SourceGeneration), ConfigReloadError> {
    let before = source
        .metadata()
        .map(|metadata| SourceGeneration::from(&metadata))
        .map_err(|source| ConfigReloadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let mut contents = Vec::new();
    source
        .read_to_end(&mut contents)
        .map_err(|source| ConfigReloadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let after = source
        .metadata()
        .map(|metadata| SourceGeneration::from(&metadata))
        .map_err(|source| ConfigReloadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let published = fs::metadata(path)
        .map(|metadata| SourceGeneration::from(&metadata))
        .map_err(|source| ConfigReloadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    if before != after || after != published {
        if let Some(observed) = observed {
            *observed = Some(if before.same_identity(&published) {
                before
            } else {
                published
            });
        }
        return Err(ConfigReloadError::UnstableGeneration {
            path: path.to_path_buf(),
        });
    }
    Ok((contents, published))
}

fn decode_config_source(path: &Path, contents: Vec<u8>) -> Result<String, ConfigReloadError> {
    String::from_utf8(contents).map_err(|source| ConfigReloadError::Read {
        path: path.to_path_buf(),
        source: io::Error::new(io::ErrorKind::InvalidData, source),
    })
}

fn parse_config(path: &Path, contents: &str) -> Result<AppConfig, ConfigReloadError> {
    let mut defaults = AppConfig::default();
    materialize_evidence_path_defaults(&mut defaults);
    let config = AppConfig::parse_with_defaults(contents, defaults).map_err(|source| {
        ConfigReloadError::Parse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    validate_config(path, &config)?;
    Ok(config)
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceGeneration {
    device: u64,
    inode: u64,
    mode: u32,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(unix)]
impl From<&Metadata> for SourceGeneration {
    fn from(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            size: metadata.size(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

#[cfg(unix)]
impl SourceGeneration {
    fn same_identity(&self, other: &Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}

#[cfg(not(unix))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceGeneration {
    created: Option<std::time::SystemTime>,
    modified: Option<std::time::SystemTime>,
    size: u64,
}

#[cfg(not(unix))]
impl From<&Metadata> for SourceGeneration {
    fn from(metadata: &Metadata) -> Self {
        Self {
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
            size: metadata.len(),
        }
    }
}

#[cfg(not(unix))]
impl SourceGeneration {
    fn same_identity(&self, other: &Self) -> bool {
        self.created == other.created
    }
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
                let _outcome = manager.reload();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::{
        ffi::OsString,
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use llm_guard_proxy_core::{AppConfig, GuardianKillAction, HeartbeatMode};
    use llm_guard_proxy_state::materialize_evidence_path_defaults;

    use super::{
        ConfigManager, ConfigReloadError, MissingConfigPolicy, ReloadTerminalState,
        default_config_path_from_home,
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

        replace_config_atomically(
            &path,
            "[server]\nport = 19000\nmax_in_flight_requests = 2\n",
        );
        let outcome = manager.reload().expect("reload should succeed");
        let snapshot = manager.handle().snapshot().expect("snapshot");
        assert!(outcome.applied);
        assert_eq!(outcome.restart_required_changes.len(), 1);
        assert_eq!(snapshot.server.port, 18_009);
        assert_eq!(snapshot.server.max_in_flight_requests, 2);
        assert_eq!(manager.last_error().expect("reload health"), None);

        replace_config_atomically(&path, "not toml");
        manager.reload().expect_err("broken reload should fail");
        assert!(manager.last_error().expect("reload health").is_some());
        assert_eq!(manager.handle().snapshot().expect("snapshot"), snapshot);

        replace_config_atomically(
            &path,
            "[server]\nport = 18009\nmax_in_flight_requests = 3\n",
        );
        manager.reload().expect("recovered reload should succeed");
        assert_eq!(manager.last_error().expect("reload health"), None);
        remove_file(&path);
    }

    #[test]
    fn missing_reload_retains_last_good_until_atomic_recovery() {
        let path = unique_test_path("missing-reload.toml");
        fs::write(&path, "[heartbeat]\ninterval_secs = 4\n").expect("write initial config");
        let manager = ConfigManager::from_path_with_policy(&path, MissingConfigPolicy::UseDefaults)
            .expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        fs::remove_file(&path).expect("remove config");
        let error = manager
            .reload()
            .expect_err("missing reload must be rejected");
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        assert_eq!(
            manager.last_error().expect("reload health"),
            Some(error.to_string())
        );

        replace_config_atomically(&path, "[heartbeat]\ninterval_secs = 6\n");
        let outcome = manager.reload().expect("atomic recovery should reload");
        assert!(outcome.applied);
        assert_eq!(
            manager
                .handle()
                .snapshot()
                .expect("recovered snapshot")
                .heartbeat
                .interval_secs,
            6
        );
        assert_eq!(manager.last_error().expect("reload health"), None);
        remove_file(&path);
    }

    #[test]
    fn empty_reload_retains_last_good() {
        let path = unique_test_path("empty-reload.toml");
        fs::write(&path, "[heartbeat]\ninterval_secs = 4\n").expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        replace_config_atomically(&path, "");
        let error = manager.reload().expect_err("empty reload must be rejected");
        assert!(error.to_string().contains("empty config"));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        assert_eq!(
            manager.last_error().expect("reload health"),
            Some(error.to_string())
        );
        remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn rejected_replacement_in_place_rewrite_retains_last_good_until_atomic_recovery() {
        let path = unique_test_path("rejected-replacement-rewrite.toml");
        fs::write(
            &path,
            "[shielding]\nenabled = false\n[heartbeat]\ninterval_secs = 4\n",
        )
        .expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        replace_config_atomically(&path, "");
        let error = manager
            .reload()
            .expect_err("empty replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::EmptyGeneration { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        let replacement_inode = fs::metadata(&path).expect("replacement metadata").ino();
        fs::write(&path, "[heartbeat]\ninterval_secs = 6\n")
            .expect("rewrite rejected replacement in place");
        assert_eq!(
            fs::metadata(&path).expect("rewritten metadata").ino(),
            replacement_inode
        );
        let error = manager
            .reload()
            .expect_err("in-place rewrite of rejected replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        let error = manager
            .reload()
            .expect_err("unchanged in-place rewrite must remain rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        replace_config_atomically(&path, "not toml");
        let error = manager
            .reload()
            .expect_err("invalid replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::Parse { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        let invalid_replacement_inode = fs::metadata(&path).expect("invalid metadata").ino();
        fs::write(&path, "[heartbeat]\ninterval_secs = 7\n")
            .expect("rewrite invalid replacement in place");
        assert_eq!(
            fs::metadata(&path)
                .expect("rewritten invalid metadata")
                .ino(),
            invalid_replacement_inode
        );
        let error = manager
            .reload()
            .expect_err("in-place rewrite of invalid replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        let error = manager
            .reload()
            .expect_err("unchanged invalid replacement rewrite must remain rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        replace_config_atomically(
            &path,
            "[shielding]\nenabled = false\n[heartbeat]\ninterval_secs = 8\n",
        );
        let outcome = manager.reload().expect("atomic recovery should reload");
        assert!(outcome.applied);
        assert_eq!(
            manager
                .handle()
                .snapshot()
                .expect("recovered snapshot")
                .heartbeat
                .interval_secs,
            8
        );
        assert_eq!(manager.last_error().expect("reload health"), None);
        remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn rejected_invalid_utf8_replacement_rewrite_retains_last_good_until_atomic_recovery() {
        let path = unique_test_path("rejected-invalid-utf8-rewrite.toml");
        fs::write(
            &path,
            "[shielding]\nenabled = false\n[heartbeat]\ninterval_secs = 4\n",
        )
        .expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        replace_config_atomically(&path, b"\xff");
        let error = manager
            .reload()
            .expect_err("invalid UTF-8 replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::Read { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        let invalid_utf8_replacement_inode =
            fs::metadata(&path).expect("invalid UTF-8 metadata").ino();
        fs::write(&path, "[heartbeat]\ninterval_secs = 7\n")
            .expect("rewrite invalid UTF-8 replacement in place");
        assert_eq!(
            fs::metadata(&path)
                .expect("rewritten invalid UTF-8 metadata")
                .ino(),
            invalid_utf8_replacement_inode
        );
        let error = manager
            .reload()
            .expect_err("in-place rewrite of invalid UTF-8 replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        let error = manager
            .reload()
            .expect_err("unchanged invalid UTF-8 rewrite must remain rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        replace_config_atomically(
            &path,
            "[shielding]\nenabled = false\n[heartbeat]\ninterval_secs = 8\n",
        );
        let outcome = manager.reload().expect("atomic recovery should reload");
        assert!(outcome.applied);
        assert_eq!(
            manager
                .handle()
                .snapshot()
                .expect("recovered snapshot")
                .heartbeat
                .interval_secs,
            8
        );
        assert_eq!(manager.last_error().expect("reload health"), None);
        remove_file(&path);
    }

    #[test]
    fn in_place_partial_reload_retains_last_good() {
        let path = unique_test_path("in-place-partial.toml");
        fs::write(
            &path,
            "[shielding]\nenabled = false\n[heartbeat]\ninterval_secs = 4\n",
        )
        .expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");

        fs::write(&path, "[heartbeat]\ninterval_secs = 6\n")
            .expect("write syntactically complete partial config");
        let error = manager
            .reload()
            .expect_err("in-place partial reload must be rejected");
        assert!(error.to_string().contains("in-place config update"));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        assert_eq!(
            manager.last_error().expect("reload health"),
            Some(error.to_string())
        );
        let error = manager
            .reload()
            .expect_err("unchanged in-place partial reload must remain rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn same_identity_change_during_capture_remains_rejected() {
        let path = unique_test_path("same-identity-change.toml");
        fs::write(&path, "[heartbeat]\ninterval_secs = 4\n").expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");
        let fifo = path.with_extension("fifo");
        let status = Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo should start");
        assert!(status.success(), "mkfifo should create the capture fixture");
        fs::rename(&fifo, &path).expect("replace source with fifo");

        let writer_path = path.clone();
        let writer = thread::spawn(move || {
            let mut open_fifo = OpenOptions::new()
                .write(true)
                .open(&writer_path)
                .expect("open fifo after reload reader");
            open_fifo
                .write_all(b"[heartbeat]\n")
                .expect("start source generation");
            let mode = fs::metadata(&writer_path)
                .expect("capture source metadata")
                .permissions()
                .mode();
            fs::set_permissions(&writer_path, fs::Permissions::from_mode(mode ^ 0o100))
                .expect("mutate capture source identity generation");
            open_fifo
                .write_all(b"interval_secs = 6\n")
                .expect("finish mutated source generation");
        });

        let error = manager
            .reload()
            .expect_err("same-identity change during capture must be rejected");
        writer.join().expect("capture writer should finish");
        assert!(matches!(
            error,
            ConfigReloadError::UnstableGeneration { .. }
        ));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        let writer_path = path.clone();
        let writer = thread::spawn(move || {
            let mut open_fifo = OpenOptions::new()
                .write(true)
                .open(&writer_path)
                .expect("open fifo after second reload reader");
            open_fifo
                .write_all(b"[heartbeat]\ninterval_secs = 6\n")
                .expect("write unchanged mutated source generation");
        });
        let error = manager
            .reload()
            .expect_err("unchanged same-identity mutation must remain rejected");
        writer.join().expect("second capture writer should finish");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn identity_change_during_capture_retains_last_good() {
        let path = unique_test_path("identity-change.toml");
        fs::write(&path, "[heartbeat]\ninterval_secs = 4\n").expect("write initial config");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let before = manager.handle().snapshot().expect("initial snapshot");
        let fifo = path.with_extension("fifo");
        let status = Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo should start");
        assert!(status.success(), "mkfifo should create the capture fixture");
        fs::rename(&fifo, &path).expect("replace source with fifo");

        let writer_path = path.clone();
        let writer = thread::spawn(move || {
            let mut open_fifo = OpenOptions::new()
                .write(true)
                .open(&writer_path)
                .expect("open fifo after reload reader");
            replace_config_atomically(&writer_path, "[heartbeat]\ninterval_secs = 8\n");
            open_fifo
                .write_all(b"[heartbeat]\ninterval_secs = 6\n")
                .expect("finish old source generation");
        });

        let error = manager
            .reload()
            .expect_err("identity change during capture must be rejected");
        writer.join().expect("capture writer should finish");
        assert!(error.to_string().contains("unstable config generation"));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        assert_eq!(
            manager.last_error().expect("reload health"),
            Some(error.to_string())
        );
        let replacement_inode = fs::metadata(&path).expect("replacement metadata").ino();
        fs::write(&path, "[heartbeat]\ninterval_secs = 70\n")
            .expect("rewrite published replacement in place");
        assert_eq!(
            fs::metadata(&path)
                .expect("rewritten replacement metadata")
                .ino(),
            replacement_inode
        );
        let error = manager
            .reload()
            .expect_err("in-place rewrite of published replacement must be rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );
        let error = manager
            .reload()
            .expect_err("unchanged published replacement rewrite must remain rejected");
        assert!(matches!(error, ConfigReloadError::InPlaceUpdate { .. }));
        assert_eq!(
            manager.handle().snapshot().expect("retained snapshot"),
            before
        );

        replace_config_atomically(&path, "[heartbeat]\ninterval_secs = 9\n");
        let outcome = manager.reload().expect("atomic recovery should reload");
        assert!(outcome.applied);
        assert_eq!(
            manager
                .handle()
                .snapshot()
                .expect("recovered snapshot")
                .heartbeat
                .interval_secs,
            9
        );
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
        replace_config_atomically(
            &path,
            format!(
                "[heartbeat]\ninterval_secs = 4\n[evidence]\nsqlite_path = \"{}\"\nblob_cache_dir = \"{}\"\n",
                root.join("evidence.sqlite3").display(),
                root.join("blobs").display()
            )
            .as_str(),
        );

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

    #[cfg(unix)]
    #[test]
    fn terminal_reload_status_tracks_generations() {
        let active_root = unique_test_path("reload-status-active");
        let candidate_root = unique_test_path("reload-status-candidate");
        for root in [&active_root, &candidate_root] {
            fs::create_dir_all(root).expect("create evidence parent");
            fs::set_permissions(root, fs::Permissions::from_mode(0o700))
                .expect("secure evidence parent");
        }
        let path = unique_test_path("reload-status.toml");
        let config = |root: &Path, port: u16, interval_secs: u64| {
            format!(
                "[server]\nport = {port}\n[heartbeat]\ninterval_secs = {interval_secs}\n[evidence]\nsqlite_path = \"{}\"\nblob_cache_dir = \"{}\"\n",
                root.join("evidence.sqlite3").display(),
                root.join("blobs").display()
            )
        };
        let initial = config(&active_root, 18_009, 15);
        replace_config_atomically(&path, &initial);
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let initial_snapshot = manager.handle().snapshot().expect("initial snapshot");

        replace_config_atomically(&path, "not toml");
        manager.reload().expect_err("malformed input should fail");
        let malformed = manager
            .reload_status()
            .expect("reload status")
            .expect("malformed terminal status");
        assert_eq!(malformed.attempt_generation, 1);
        assert_eq!(malformed.snapshot_generation, 0);
        assert_eq!(malformed.snapshot, initial_snapshot);
        assert!(matches!(
            malformed.terminal,
            ReloadTerminalState::Rejected(_)
        ));

        replace_config_atomically(&path, &initial);
        manager
            .reload()
            .expect("unchanged config should be accepted");
        let no_change = manager
            .reload_status()
            .expect("reload status")
            .expect("no-change terminal status");
        assert_eq!(no_change.attempt_generation, 2);
        assert_eq!(no_change.snapshot_generation, 2);
        assert_eq!(no_change.snapshot, initial_snapshot);
        assert_eq!(no_change.terminal, ReloadTerminalState::NoChange);

        fs::set_permissions(&active_root, fs::Permissions::from_mode(0o755))
            .expect("make projected evidence parent unsafe");
        replace_config_atomically(&path, config(&candidate_root, 18_009, 15));
        let outcome = manager
            .reload()
            .expect("valid candidate with invalid projection should return an outcome");
        assert!(!outcome.applied);
        assert!(outcome.rejection.is_some());
        let rejected = manager
            .reload_status()
            .expect("reload status")
            .expect("projected-rejection terminal status");
        assert_eq!(rejected.attempt_generation, 3);
        assert_eq!(rejected.snapshot_generation, 2);
        assert_eq!(rejected.snapshot, initial_snapshot);
        assert!(matches!(
            rejected.terminal,
            ReloadTerminalState::Rejected(_)
        ));
        assert!(manager.last_error().expect("reload health").is_some());
        fs::set_permissions(&active_root, fs::Permissions::from_mode(0o700))
            .expect("restore projected evidence parent");

        replace_config_atomically(&path, config(&active_root, 19_000, 15));
        manager
            .reload()
            .expect("restart-only recovery should be accepted");
        let restart_pending = manager
            .reload_status()
            .expect("reload status")
            .expect("restart-pending terminal status");
        assert_eq!(restart_pending.attempt_generation, 4);
        assert_eq!(restart_pending.snapshot_generation, 4);
        assert_eq!(restart_pending.snapshot, initial_snapshot);
        assert_eq!(
            restart_pending.terminal,
            ReloadTerminalState::RestartPending
        );
        assert_eq!(restart_pending.restart_required_changes.len(), 1);
        assert_eq!(manager.last_error().expect("reload health"), None);

        replace_config_atomically(&path, config(&active_root, 19_000, 4));
        manager.reload().expect("coherent reload should apply");
        let applied = manager
            .reload_status()
            .expect("reload status")
            .expect("applied terminal status");
        assert_eq!(applied.attempt_generation, 5);
        assert_eq!(applied.snapshot_generation, 5);
        assert_eq!(applied.terminal, ReloadTerminalState::Applied);
        assert_eq!(applied.snapshot.heartbeat.interval_secs, 4);
        assert_eq!(
            applied.snapshot,
            manager.handle().snapshot().expect("current snapshot")
        );

        remove_file(&path);
        remove_dir_all(&active_root);
        remove_dir_all(&candidate_root);
    }

    #[test]
    fn polling_watcher_applies_reloadable_changes() {
        let path = unique_test_path("polling.toml");
        replace_config_atomically(&path, "[heartbeat]\nmode = \"sse\"\ninterval_secs = 15\n");
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let handle = manager.handle();
        let watcher = manager
            .spawn_polling(Duration::from_millis(10))
            .expect("start watcher");

        replace_config_atomically(
            &path,
            "[heartbeat]\nmode = \"disabled\"\ninterval_secs = 4\n",
        );
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

        replace_config_atomically(&path, "not toml");
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
        assert!(matches!(
            manager
                .reload_status()
                .expect("reload status")
                .expect("polling rejection status")
                .terminal,
            ReloadTerminalState::Rejected(_)
        ));

        replace_config_atomically(&path, "[heartbeat]\nmode = \"sse\"\ninterval_secs = 3\n");
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

        replace_config_atomically(
            &path,
            "[server]\nport = 19000\n[heartbeat]\nmode = \"sse\"\ninterval_secs = 3\n",
        );
        let mut restart_pending = false;
        for _attempt in 0..50 {
            if let Some(status) = manager.reload_status().expect("reload status")
                && status.terminal == ReloadTerminalState::RestartPending
                && status.restart_required_changes.len() == 1
            {
                restart_pending = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            restart_pending,
            "polling should retain restart-required metadata"
        );

        watcher.stop().expect("watcher should stop");
        remove_file(&path);
    }

    #[test]
    fn polling_watcher_hot_reloads_guardian_policy_and_retains_last_good() {
        let path = unique_test_path("guardian-polling.toml");
        replace_config_atomically(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"aeon-text\"\nmem_threshold_gib = 2\nkill_action = \"cgroup.kill\"\npoll_interval_secs = 1\nregistration_file = \"text-cgroup.v1\"\n",
        );
        let manager = ConfigManager::from_explicit_path(&path).expect("load initial config");
        let handle = manager.handle();
        let watcher = manager
            .spawn_polling(Duration::from_millis(10))
            .expect("start watcher");

        replace_config_atomically(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"replacement\"\nmem_threshold_gib = 5\nkill_action = \"systemctl_restart\"\npoll_interval_secs = 4\nsystemd_unit = \"replacement.service\"\n",
        );
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

        replace_config_atomically(
            &path,
            "[guardian]\nenabled = true\ntarget_label = \"\"\nmem_threshold_gib = 0\n",
        );
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

    static TEST_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn replace_config_atomically(path: &Path, contents: impl AsRef<[u8]>) {
        let replacement = path.with_extension(format!(
            "replacement-{}",
            TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&replacement, contents).expect("write replacement config");
        fs::rename(&replacement, path).expect("atomically replace config");
    }

    fn unique_test_path(file_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let counter = TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "llm-guard-proxy-config-reload-{}-{nanos}-{counter}-{file_name}",
            std::process::id()
        ))
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
