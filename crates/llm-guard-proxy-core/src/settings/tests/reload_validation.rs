use super::super::{AppConfig, ConfigHandle, apply_reloadable};

#[test]
fn low_level_reload_rejects_unvalidated_candidate_without_publishing() {
    let current = AppConfig::default();
    current.validate().expect("default config should validate");
    let handle = ConfigHandle::new(current.clone());
    let mut requested = current.clone();
    requested.server.max_in_flight_requests = 0;

    let (next, outcome) = apply_reloadable(&current, &requested);

    assert!(!outcome.applied);
    let rejection = outcome
        .rejection
        .expect("invalid candidate must populate structured rejection");
    assert_eq!(rejection.field(), "server.max_in_flight_requests");
    assert_eq!(next, current);

    let handle_outcome = handle
        .apply_reloadable(&requested)
        .expect("handle reload should return an outcome");
    assert!(!handle_outcome.applied);
    let handle_rejection = handle_outcome
        .rejection
        .expect("direct handle callers must not publish invalid candidates");
    assert_eq!(handle_rejection.field(), "server.max_in_flight_requests");
    assert_eq!(handle.snapshot().expect("snapshot should succeed"), current);
}

#[test]
fn apply_reloadable_rejects_projected_invalid_snapshot_without_publishing() {
    let current = AppConfig::parse(
        r#"
[[upstreams]]
name = "named"
base_url = "http://127.0.0.1:19000/v1"
match_models = ["named-chat"]
request_timeout_ms = 5000
"#,
    )
    .expect("current named-profile config should parse");
    current
        .validate()
        .expect("short profile timeout is valid while stall recovery is disabled");

    let requested = AppConfig::parse(
        r"
[upstream.stall]
enabled = true
",
    )
    .expect("requested stall-enabled config should parse");
    requested
        .validate()
        .expect("stall-enabled config is valid without named profiles");

    let handle = ConfigHandle::new(current.clone());
    let (next, outcome) = apply_reloadable(&current, &requested);

    assert!(!outcome.applied);
    let rejection = outcome
        .rejection
        .expect("projected snapshot must populate structured rejection");
    assert_eq!(rejection.field(), "upstream.stall.idle_timeout_ms");
    assert_eq!(next, current);

    let handle_outcome = handle
        .apply_reloadable(&requested)
        .expect("handle reload should return an outcome");
    assert!(!handle_outcome.applied);
    let handle_rejection = handle_outcome
        .rejection
        .expect("invalid projection must not publish over last-good");
    assert_eq!(handle_rejection.field(), "upstream.stall.idle_timeout_ms");
    assert_eq!(handle.snapshot().expect("snapshot should succeed"), current);
}

const TIER2_CONFIG: &str = r#"
[guardian]
enabled = true
target_label = "test"
escalation_enabled = true
escalation_grace_secs = 30
escalation_timeout_secs = 10
escalation_mem_threshold_gib = 1
"#;

fn tier2_config(profile: &str, recovery_config: &str) -> AppConfig {
    let contents = format!("{TIER2_CONFIG}escalation_profile = \"{profile}\"\n{recovery_config}");
    AppConfig::parse(&contents).expect("Tier 2 config should parse")
}

#[test]
fn tier2_requires_existing_enabled_usable_recovery_profile_at_startup_and_reload() {
    let invalid = [
        ("default recovery missing", "default", ""),
        ("named profile missing", "missing", ""),
        (
            "recovery disabled",
            "default",
            "[upstream.local_recovery]\nenabled = false\nrestart_command = [\"/usr/bin/true\"]",
        ),
        (
            "recovery command empty",
            "default",
            "[upstream.local_recovery]\nenabled = true",
        ),
        (
            "foreground command unsupported",
            "default",
            "[upstream.local_recovery]\nenabled = true\nrestart_command = [\"/bin/sh\", \"-c\", \"true\"]",
        ),
    ];
    for (case, profile, recovery_config) in invalid {
        let error = tier2_config(profile, recovery_config)
            .validate()
            .expect_err(case);
        assert_eq!(error.field(), "guardian.escalation_profile", "{case}");
    }

    let configured_default = tier2_config(
        "default",
        "[upstream.local_recovery]\nenabled = true\nrestart_command = [\"/usr/bin/true\"]",
    );
    configured_default
        .validate()
        .expect("configured default recovery profile should validate");

    let configured_named = tier2_config(
        "named",
        "[[upstreams]]\nname = \"named\"\nbase_url = \"http://127.0.0.1:19000/v1\"\nmatch_models = [\"named-chat\"]\n[upstreams.local_recovery]\nenabled = true\nrestart_command = [\"/usr/bin/true\"]",
    );
    configured_named
        .validate()
        .expect("configured named recovery profile should validate");

    let current = AppConfig::default();
    let requested = tier2_config("default", "");
    let (projected, outcome) = apply_reloadable(&current, &requested);
    assert!(!outcome.applied);
    assert_eq!(
        outcome
            .rejection
            .expect("invalid projected policy should be rejected")
            .field(),
        "guardian.escalation_profile"
    );
    assert_eq!(projected, current);

    let handle = ConfigHandle::new(current.clone());
    let handle_outcome = handle
        .apply_reloadable(&requested)
        .expect("reload should return a validation result");
    assert!(!handle_outcome.applied);
    assert_eq!(
        handle_outcome
            .rejection
            .expect("invalid projected policy should not publish")
            .field(),
        "guardian.escalation_profile"
    );
    assert_eq!(handle.snapshot().expect("snapshot should succeed"), current);
}
