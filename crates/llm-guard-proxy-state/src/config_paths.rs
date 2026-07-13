//! Environment-backed evidence path defaults and filesystem preflight.
//!
//! The core configuration model is deterministic. This module owns the
//! operational evidence needed to materialize process-specific defaults and
//! reject unsafe storage paths before a config snapshot is installed.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use llm_guard_proxy_core::{AppConfig, EvidenceConfig, ValidationError};

const EVIDENCE_SQLITE_FIELD: &str = "evidence.sqlite_path";
const EVIDENCE_BLOB_CACHE_FIELD: &str = "evidence.blob_cache_dir";

/// Replaces deterministic evidence path fallbacks with XDG defaults.
///
/// Call this on an otherwise-default configuration before parsing user input,
/// so explicit path values remain distinguishable from omitted values.
pub fn materialize_evidence_path_defaults(config: &mut AppConfig) {
    materialize_evidence_path_defaults_from(
        &mut config.evidence,
        env::var_os("XDG_STATE_HOME").map(PathBuf::from),
        env::var_os("XDG_CACHE_HOME").map(PathBuf::from),
    );
}

/// Checks evidence storage paths against current filesystem and HOME state.
///
/// Disabled evidence still checks custom paths. Disabled paths that equal the
/// effective defaults are skipped, matching the historical configuration
/// behavior.
///
/// # Errors
///
/// Returns [`ValidationError`] when a path cannot be resolved or inspected,
/// contains a symlink, has the wrong existing type, or uses an existing
/// directory accessible by group or other users.
pub fn preflight_evidence_paths(config: &AppConfig) -> Result<(), ValidationError> {
    preflight_evidence_paths_from(
        &config.evidence,
        env::var_os("XDG_STATE_HOME").map(PathBuf::from),
        env::var_os("XDG_CACHE_HOME").map(PathBuf::from),
        env::var_os("HOME").map(PathBuf::from).as_deref(),
    )
}

fn materialize_evidence_path_defaults_from(
    evidence: &mut EvidenceConfig,
    xdg_state_home: Option<PathBuf>,
    xdg_cache_home: Option<PathBuf>,
) {
    evidence.sqlite_path = evidence_sqlite_path_from_xdg_state_home(xdg_state_home);
    evidence.blob_cache_dir = evidence_blob_cache_dir_from_xdg_cache_home(xdg_cache_home);
}

fn evidence_sqlite_path_from_xdg_state_home(xdg_state_home: Option<PathBuf>) -> PathBuf {
    xdg_state_home
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("~/.local/state"))
        .join("llm-guard-proxy")
        .join("evidence.sqlite3")
}

fn evidence_blob_cache_dir_from_xdg_cache_home(xdg_cache_home: Option<PathBuf>) -> PathBuf {
    xdg_cache_home
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("~/.cache"))
        .join("llm-guard-proxy")
        .join("evidence")
        .join("blobs")
}

fn preflight_evidence_paths_from(
    evidence: &EvidenceConfig,
    xdg_state_home: Option<PathBuf>,
    xdg_cache_home: Option<PathBuf>,
    home: Option<&Path>,
) -> Result<(), ValidationError> {
    let default_sqlite = evidence_sqlite_path_from_xdg_state_home(xdg_state_home);
    if evidence.enabled || evidence.sqlite_path != default_sqlite {
        validate_evidence_sqlite_path(&evidence.sqlite_path, home)?;
    }

    let default_blob_cache = evidence_blob_cache_dir_from_xdg_cache_home(xdg_cache_home);
    if evidence.enabled || evidence.blob_cache_dir != default_blob_cache {
        validate_evidence_blob_cache_dir(&evidence.blob_cache_dir, home)?;
    }
    Ok(())
}

fn validate_evidence_sqlite_path(path: &Path, home: Option<&Path>) -> Result<(), ValidationError> {
    let resolved = resolve_evidence_validation_path(path, EVIDENCE_SQLITE_FIELD, home)?;
    reject_symlink_path_components(&resolved, EVIDENCE_SQLITE_FIELD)?;
    if let Some(metadata) = path_metadata_if_exists(&resolved, EVIDENCE_SQLITE_FIELD)? {
        require(
            !metadata.file_type().is_symlink() && metadata.is_file(),
            EVIDENCE_SQLITE_FIELD,
            "must be a regular file when it already exists",
        )?;
    }
    if let Some(parent) = resolved.parent() {
        validate_existing_owner_private_directory(parent, EVIDENCE_SQLITE_FIELD)?;
    }
    Ok(())
}

fn validate_evidence_blob_cache_dir(
    path: &Path,
    home: Option<&Path>,
) -> Result<(), ValidationError> {
    let resolved = resolve_evidence_validation_path(path, EVIDENCE_BLOB_CACHE_FIELD, home)?;
    reject_symlink_path_components(&resolved, EVIDENCE_BLOB_CACHE_FIELD)?;
    validate_existing_owner_private_directory(&resolved, EVIDENCE_BLOB_CACHE_FIELD)?;
    if let Some(parent) = resolved.parent() {
        validate_existing_owner_private_directory(parent, EVIDENCE_BLOB_CACHE_FIELD)?;
    }
    Ok(())
}

fn resolve_evidence_validation_path(
    path: &Path,
    field: &'static str,
    home: Option<&Path>,
) -> Result<PathBuf, ValidationError> {
    if path.starts_with("~") {
        let home = home.ok_or_else(|| {
            ValidationError::new(field, "HOME must be set when evidence path starts with ~")
        })?;
        let suffix = path.strip_prefix("~").unwrap_or(path);
        return Ok(home.join(suffix));
    }
    Ok(path.to_path_buf())
}

fn reject_symlink_path_components(path: &Path, field: &'static str) -> Result<(), ValidationError> {
    let mut inspected = PathBuf::new();
    for component in path.components() {
        inspected.push(component.as_os_str());
        match fs::symlink_metadata(&inspected) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ValidationError::new(
                    field,
                    format!(
                        "must not contain symlink path component {}",
                        inspected.display()
                    ),
                ));
            }
            Ok(_metadata) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => break,
            Err(error) => {
                return Err(ValidationError::new(
                    field,
                    format!("must be inspectable: {error}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_existing_owner_private_directory(
    path: &Path,
    field: &'static str,
) -> Result<(), ValidationError> {
    let Some(metadata) = path_metadata_if_exists(path, field)? else {
        return Ok(());
    };
    require(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        field,
        "existing storage parent must be a real directory",
    )?;
    #[cfg(unix)]
    {
        let mode = metadata.permissions().mode() & 0o777;
        require(
            mode.trailing_zeros() >= 6,
            field,
            "existing storage parent must not be accessible by group or other users",
        )?;
    }
    Ok(())
}

fn path_metadata_if_exists(
    path: &Path,
    field: &'static str,
) -> Result<Option<fs::Metadata>, ValidationError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(ValidationError::new(
            field,
            format!("must be inspectable: {error}"),
        )),
    }
}

fn require(
    condition: bool,
    field: &'static str,
    message: &'static str,
) -> Result<(), ValidationError> {
    if condition {
        Ok(())
    } else {
        Err(ValidationError::new(field, message))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use llm_guard_proxy_core::EvidenceConfig;

    use super::{materialize_evidence_path_defaults_from, preflight_evidence_paths_from};

    #[test]
    fn materializes_xdg_defaults_and_empty_value_fallbacks() {
        let mut evidence = EvidenceConfig::default();
        materialize_evidence_path_defaults_from(
            &mut evidence,
            Some(PathBuf::from("/tmp/xdg-state")),
            Some(PathBuf::from("/tmp/xdg-cache")),
        );
        assert_eq!(
            evidence.sqlite_path,
            PathBuf::from("/tmp/xdg-state/llm-guard-proxy/evidence.sqlite3")
        );
        assert_eq!(
            evidence.blob_cache_dir,
            PathBuf::from("/tmp/xdg-cache/llm-guard-proxy/evidence/blobs")
        );

        materialize_evidence_path_defaults_from(
            &mut evidence,
            Some(PathBuf::new()),
            Some(PathBuf::new()),
        );
        assert_eq!(
            evidence.sqlite_path,
            PathBuf::from("~/.local/state/llm-guard-proxy/evidence.sqlite3")
        );
        assert_eq!(
            evidence.blob_cache_dir,
            PathBuf::from("~/.cache/llm-guard-proxy/evidence/blobs")
        );
    }

    #[test]
    fn disabled_default_paths_do_not_require_home() {
        let evidence = EvidenceConfig::default();
        preflight_evidence_paths_from(&evidence, None, None, None)
            .expect("disabled defaults should skip filesystem preflight");
    }

    #[test]
    fn enabled_home_relative_path_requires_home() {
        let evidence = EvidenceConfig {
            enabled: true,
            ..EvidenceConfig::default()
        };
        let error = preflight_evidence_paths_from(&evidence, None, None, None)
            .expect_err("enabled default should require HOME");
        assert_eq!(error.field(), "evidence.sqlite_path");
        assert_eq!(
            error.message(),
            "HOME must be set when evidence path starts with ~"
        );
    }

    #[cfg(unix)]
    #[test]
    fn disabled_custom_paths_reject_unsafe_parent_permissions() {
        let root = unique_test_path("unsafe-parent");
        fs::create_dir_all(&root).expect("create test root");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))
            .expect("set unsafe test permissions");
        let evidence = EvidenceConfig {
            sqlite_path: root.join("evidence.sqlite3"),
            blob_cache_dir: root.join("blobs"),
            ..EvidenceConfig::default()
        };

        let error = preflight_evidence_paths_from(&evidence, None, None, None)
            .expect_err("disabled custom path should still be checked");
        assert_eq!(error.field(), "evidence.sqlite_path");
        assert!(error.message().contains("group or other users"));

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("restore safe permissions");
        remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn disabled_custom_paths_reject_symlink_components() {
        let root = unique_test_path("symlink");
        let real = root.join("real");
        let link = root.join("link");
        fs::create_dir_all(&real).expect("create real directory");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("restrict root permissions");
        fs::set_permissions(&real, fs::Permissions::from_mode(0o700))
            .expect("restrict real permissions");
        symlink(&real, &link).expect("create symlink");

        let mut evidence = EvidenceConfig {
            sqlite_path: link.join("evidence.sqlite3"),
            blob_cache_dir: real.join("blobs"),
            ..EvidenceConfig::default()
        };
        let sqlite_error = preflight_evidence_paths_from(&evidence, None, None, None)
            .expect_err("sqlite symlink should fail");
        assert_eq!(sqlite_error.field(), "evidence.sqlite_path");
        assert!(sqlite_error.message().contains("symlink"));

        evidence.sqlite_path = real.join("evidence.sqlite3");
        evidence.blob_cache_dir = link.join("blobs");
        let blob_error = preflight_evidence_paths_from(&evidence, None, None, None)
            .expect_err("blob cache symlink should fail");
        assert_eq!(blob_error.field(), "evidence.blob_cache_dir");
        assert!(blob_error.message().contains("symlink"));

        remove_dir_all(&root);
    }

    fn unique_test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should follow Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("llm-guard-proxy-config-paths-{name}-{nanos}"))
    }

    fn remove_dir_all(path: &Path) {
        if let Err(error) = fs::remove_dir_all(path) {
            assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        }
    }
}
