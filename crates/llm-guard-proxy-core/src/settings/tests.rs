use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use super::{
    AppConfig, ConfigManager, ConfigParseError, HeartbeatMode, MissingConfigPolicy,
    RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS, ValidationError, parse::parse_config_text,
    redact_upstream_base_url,
};

#[test]
fn defaults_match_issue_contract() {
    let config = AppConfig::default();

    config.validate().expect("default config should validate");
    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_009);
    assert_eq!(config.server.max_in_flight_requests, 16);
    assert_eq!(config.upstream.base_url, "http://gb10:18009/v1");
    assert!(config.upstream.metadata.discovery_enabled);
    assert!(config.upstream.metadata.enrich_responses);
    assert!(config.shielding.enabled);
    assert!(config.observability.enabled);
    assert!(!config.observability.capture_raw_payloads);
    assert!(config.thinking.enabled);
    assert_eq!(config.thinking.budget_tokens, 32_768);
    assert!(config.loop_guard.enabled);
    assert_eq!(config.loop_guard.normalized_input_window_secs, 120);
    assert_eq!(config.loop_guard.max_repeated_inputs, 1);
    assert_eq!(config.loop_guard.output_repeated_line_threshold, 24);
    assert_eq!(config.loop_guard.output_token_window_size, 12);
    assert_eq!(config.loop_guard.output_repeated_token_window_threshold, 32);
    assert_eq!(config.loop_guard.output_suffix_cycle_threshold, 32);
    assert_eq!(config.loop_guard.output_low_progress_min_bytes, 4_096);
    assert_eq!(
        config.loop_guard.output_low_progress_unique_ratio_percent,
        15
    );
    assert_eq!(config.loop_guard.input_overlap_threshold_multiplier, 4);
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
max_in_flight_requests = 2

[upstream.metadata]
context_length_override = 256000
max_model_len_override = 256000

[heartbeat]
mode = "json-whitespace"
interval_secs = 5

[loop_guard]
output_repeated_line_threshold = 40
output_token_window_size = 8
output_repeated_token_window_threshold = 9
output_suffix_cycle_threshold = 10
output_low_progress_min_bytes = 2048
output_low_progress_unique_ratio_percent = 25
input_overlap_threshold_multiplier = 5

[cloudflare]
enabled = false
"#,
    )
    .expect("config should parse");

    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_100);
    assert_eq!(config.server.max_in_flight_requests, 2);
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
    assert_eq!(config.loop_guard.output_repeated_line_threshold, 40);
    assert_eq!(config.loop_guard.output_token_window_size, 8);
    assert_eq!(config.loop_guard.output_repeated_token_window_threshold, 9);
    assert_eq!(config.loop_guard.output_suffix_cycle_threshold, 10);
    assert_eq!(config.loop_guard.output_low_progress_min_bytes, 2_048);
    assert_eq!(
        config.loop_guard.output_low_progress_unique_ratio_percent,
        25
    );
    assert_eq!(config.loop_guard.input_overlap_threshold_multiplier, 5);
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
fn validates_server_in_flight_limit() {
    let mut config = AppConfig::default();
    config.server.max_in_flight_requests = 0;

    let error = config
        .validate()
        .expect_err("zero in-flight request limit should fail");

    assert_eq!(error.field(), "server.max_in_flight_requests");
}

#[test]
fn validates_loop_guard_ratio_limit() {
    let mut config = AppConfig::default();
    config.loop_guard.output_low_progress_unique_ratio_percent = 101;

    let error = config
        .validate()
        .expect_err("low-progress ratio should be bounded");
    assert_eq!(
        error.field(),
        "loop_guard.output_low_progress_unique_ratio_percent"
    );
}

#[test]
fn validates_normal_upstream_base_urls() {
    for base_url in ["http://gb10:18009/v1", "https://host.example/v1"] {
        let mut config = AppConfig::default();
        config.upstream.base_url = base_url.to_owned();

        config
            .validate()
            .expect("normal upstream URL should validate");
    }
}

#[test]
fn rejects_upstream_base_url_with_userinfo() {
    let mut config = AppConfig::default();
    config.upstream.base_url = String::from("https://user:secret@example.test/v1");

    let error = config
        .validate()
        .expect_err("credential-bearing upstream URL should be rejected");

    assert_eq!(error.field(), "upstream.base_url");
    assert!(error.message().contains("userinfo"));
    assert!(!error.to_string().contains("secret"));
}

#[test]
fn rejects_upstream_base_url_with_any_query_string() {
    for base_url in [
        "https://example.test/v1?safe=sk-test",
        "https://example.test/v1?q=Bearer%20sk-test",
        "https://example.test/v1?safe=ok",
    ] {
        let mut config = AppConfig::default();
        config.upstream.base_url = base_url.to_owned();

        let error = config
            .validate()
            .expect_err("upstream base URL query strings should be rejected");

        assert_eq!(error.field(), "upstream.base_url");
        assert!(error.message().contains("query parameters"));
        assert!(!error.to_string().contains("sk-test"));
        assert!(!error.to_string().contains("Bearer"));
        assert!(!error.to_string().contains("safe=sk-test"));
        assert!(!error.to_string().contains("q=Bearer%20sk-test"));
    }
}

#[test]
fn rejects_upstream_base_url_with_sensitive_query_key_variants() {
    for base_url in [
        "https://example.test/v1?x-api-key=sk-test",
        "https://example.test/v1?client_secret=sk-test",
        "https://example.test/v1?refresh_token=sk-test",
        "https://example.test/v1?secret_key=sk-test",
    ] {
        let mut config = AppConfig::default();
        config.upstream.base_url = base_url.to_owned();

        let error = config
            .validate()
            .expect_err("upstream base URL query strings should be rejected");

        assert_eq!(error.field(), "upstream.base_url");
        assert!(error.message().contains("query parameters"));
        assert!(!error.to_string().contains("sk-test"));
    }
}

#[test]
fn rejects_upstream_base_url_with_fragment() {
    for base_url in [
        "https://example.test/v1#token=sk-test",
        "https://example.test/v1#section",
    ] {
        let mut config = AppConfig::default();
        config.upstream.base_url = base_url.to_owned();

        let error = config
            .validate()
            .expect_err("upstream URL fragments should be rejected");

        assert_eq!(error.field(), "upstream.base_url");
        assert!(error.message().contains("fragments"));
        assert!(!error.to_string().contains("sk-test"));
        assert!(!error.to_string().contains("token=sk-test"));
    }
}

#[test]
fn redacts_upstream_base_url_for_display() {
    let mut config = AppConfig::default();
    config.upstream.base_url =
        String::from("https://user:secret@example.test/v1?api_key=sk-test&safe=ok");

    let redacted = config.upstream.redacted_base_url();

    assert_eq!(
        redacted,
        "https://redacted:redacted@example.test/v1?redacted"
    );
    assert!(!redacted.contains("user"));
    assert!(!redacted.contains("secret"));
    assert!(!redacted.contains("sk-test"));
    assert!(!redacted.contains("api_key"));
    assert!(!redacted.contains("safe=ok"));
}

#[test]
fn redacts_sensitive_upstream_query_variants_and_fragments_for_display() {
    for base_url in [
        "https://example.test/v1?x-api-key=sk-test&safe=ok",
        "https://example.test/v1?client_secret=sk-test&safe=ok",
        "https://example.test/v1?safe=sk-test",
        "https://example.test/v1?q=Bearer%20sk-test",
        "https://example.test/v1?safe=ok#token=sk-test",
    ] {
        let redacted = redact_upstream_base_url(base_url);

        assert!(redacted.ends_with("/v1?redacted"));
        assert!(!redacted.contains("sk-test"));
        assert!(!redacted.contains("Bearer"));
        assert!(!redacted.contains("client_secret=sk-test"));
        assert!(!redacted.contains("x-api-key=sk-test"));
        assert!(!redacted.contains("safe=sk-test"));
        assert!(!redacted.contains("q=Bearer%20sk-test"));
        assert!(!redacted.contains("safe=ok"));
        assert!(!redacted.contains("token=sk-test"));
    }
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

[loop_guard]
output_repeated_line_threshold = 24
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

[loop_guard]
output_repeated_line_threshold = 7
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
    assert_eq!(snapshot.loop_guard.output_repeated_line_threshold, 7);

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
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.output_repeated_line_threshold"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.input_overlap_threshold_multiplier"));
    assert!(RELOADABLE_FIELDS.contains(&"cloudflare.enabled"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"server.max_in_flight_requests"));
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
