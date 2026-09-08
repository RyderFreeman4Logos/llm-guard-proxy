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
