use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use super::{
    AppConfig, ConfigManager, ConfigParseError, HeartbeatMode, MissingConfigPolicy,
    RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS, ValidationError, parse::parse_config_text,
};

#[test]
fn defaults_match_issue_contract() {
    let config = AppConfig::default();

    config.validate().expect("default config should validate");
    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_009);
    assert_eq!(config.upstream.base_url, "http://gb10:18009/v1");
    assert!(config.upstream.metadata.discovery_enabled);
    assert!(config.upstream.metadata.enrich_responses);
    assert!(config.shielding.enabled);
    assert!(config.observability.enabled);
    assert!(!config.observability.capture_raw_payloads);
    assert!(config.thinking.enabled);
    assert_eq!(config.thinking.budget_tokens, 32_768);
    assert!(config.loop_guard.enabled);
    assert!(config.retry.enabled);
    assert_eq!(config.heartbeat.mode, HeartbeatMode::Sse);
    assert!(config.cloudflare.enabled);
}

#[test]
fn parses_toml_with_defaults_and_overrides() {
    let config = parse_config_text(
        r#"
[server]
port = 18100

[upstream.metadata]
context_length_override = 256000
max_model_len_override = 256000

[heartbeat]
mode = "json-whitespace"
interval_secs = 5

[cloudflare]
enabled = false
"#,
    )
    .expect("config should parse");

    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_100);
    assert_eq!(config.upstream.base_url, "http://gb10:18009/v1");
    assert_eq!(
        config.upstream.metadata.context_length_override,
        Some(256_000)
    );
    assert_eq!(
        config.upstream.metadata.max_model_len_override,
        Some(256_000)
    );
    assert_eq!(config.heartbeat.mode, HeartbeatMode::JsonWhitespace);
    assert_eq!(config.heartbeat.interval_secs, 5);
    assert!(!config.cloudflare.enabled);
}

#[test]
fn rejects_unknown_toml_fields() {
    let error = parse_config_text(
        r"
[thinking]
unknown = true
",
    )
    .expect_err("unknown fields should fail");

    assert_eq!(error.line(), 3);
    assert!(error.message().contains("unknown config key"));
}

#[test]
fn validates_retention_hysteresis() {
    let mut config = AppConfig::default();
    config.observability.retention.max_bytes = 10;
    config.observability.retention.prune_to_bytes = 11;

    let error = config
        .validate()
        .expect_err("retention relation should fail");
    assert_eq!(error.field(), "observability.retention.prune_to_bytes");
}

#[test]
fn default_path_uses_defaults_when_file_is_absent() {
    let path = unique_test_path("missing-default.toml");
    let manager = ConfigManager::from_path_with_policy(&path, MissingConfigPolicy::UseDefaults)
        .expect("missing default config should use built-in defaults");

    assert_eq!(
        manager
            .handle()
            .snapshot()
            .expect("snapshot should succeed"),
        AppConfig::default()
    );
}

#[test]
fn explicit_path_requires_existing_file() {
    let path = unique_test_path("missing-explicit.toml");
    let error = ConfigManager::from_path_with_policy(&path, MissingConfigPolicy::RequireFile)
        .expect_err("explicit config should require a file");

    assert!(error.to_string().contains("failed to read config"));
}

#[test]
fn reload_applies_only_reloadable_fields() {
    let path = unique_test_path("reload.toml");
    write_config(
        &path,
        r#"
[server]
port = 18009

[heartbeat]
mode = "sse"
interval_secs = 15
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[server]
port = 19000

[heartbeat]
mode = "disabled"
interval_secs = 3
"#,
    );
    let outcome = manager.reload().expect("reload should succeed");
    let snapshot = manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed");

    assert!(outcome.applied);
    assert_eq!(outcome.restart_required_changes.len(), 1);
    assert_eq!(outcome.restart_required_changes[0].field, "server.port");
    assert_eq!(snapshot.server.port, 18_009);
    assert_eq!(snapshot.heartbeat.mode, HeartbeatMode::Disabled);
    assert_eq!(snapshot.heartbeat.interval_secs, 3);

    remove_file(&path);
}

#[test]
fn polling_watcher_applies_reloadable_file_changes() {
    let path = unique_test_path("polling-reload.toml");
    write_config(
        &path,
        r#"
[server]
port = 18009

[heartbeat]
mode = "sse"
interval_secs = 15
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");
    let handle = manager.handle();
    let watcher = manager
        .spawn_polling(Duration::from_millis(10))
        .expect("polling watcher should start");

    write_config(
        &path,
        r#"
[server]
port = 19000

[heartbeat]
mode = "disabled"
interval_secs = 4
"#,
    );

    let mut observed = None;
    for _attempt in 0..50 {
        let snapshot = handle.snapshot().expect("snapshot should succeed");
        if snapshot.heartbeat.mode == HeartbeatMode::Disabled
            && snapshot.heartbeat.interval_secs == 4
        {
            observed = Some(snapshot);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let snapshot = observed.expect("polling watcher should apply reload");
    assert_eq!(snapshot.server.port, 18_009);
    assert_eq!(snapshot.heartbeat.mode, HeartbeatMode::Disabled);
    assert_eq!(snapshot.heartbeat.interval_secs, 4);

    watcher.stop().expect("watcher should stop cleanly");
    remove_file(&path);
}

#[test]
fn reload_metadata_lists_cover_expected_fields() {
    assert!(RELOADABLE_FIELDS.contains(&"thinking.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"cloudflare.enabled"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"upstream.base_url"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"observability.sqlite_path"));
}

fn write_config(path: &Path, contents: &str) {
    fs::write(path, contents).expect("test config should be writable");
}

fn unique_test_path(file_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("llm-guard-proxy-{nanos}-{file_name}"))
}

fn remove_file(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

fn _assert_error_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ConfigParseError>();
    assert_send_sync::<ValidationError>();
}
