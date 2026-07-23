use super::*;

#[test]
#[cfg(feature = "param-override")]
fn gb10_deploy_documents_listener_and_profile_policy_overrides() {
    let example = deploy_config_example("listener-profile-policy");
    let config = parse_config_text(&example).expect("listener-profile-policy example should parse");
    config
        .validate()
        .expect("listener-profile-policy example should validate");
    assert_eq!(
        config.listeners[0].upstream_profile.as_deref(),
        Some("profile-policy")
    );
    assert_eq!(
        config.upstream_profiles[0]
            .loop_guard
            .as_ref()
            .map(|guard| guard.max_repeated_inputs),
        Some(2)
    );
    assert_eq!(
        config.upstream_profiles[0]
            .retry_ladder
            .as_ref()
            .map(|ladder| ladder[0].name.as_str()),
        Some("profile-first")
    );
}

#[test]
fn profile_policy_reload_metadata_is_complete() {
    for field in ["upstreams.loop_guard", "upstreams.retry.ladder"] {
        assert!(RELOADABLE_FIELDS.contains(&field), "missing {field}");
    }
}

#[test]
fn profile_policy_validation_errors_use_upstreams_paths() {
    let loop_guard_error = parse_config_text(
        r#"
[[upstreams]]
name = "profile"
base_url = "http://profile.example/v1"
match_models = ["profile-model"]

[upstreams.loop_guard]
max_repeated_inputs = 0
"#,
    )
    .expect("profile loop guard config should parse before validation")
    .validate()
    .expect_err("profile loop guard should reject zero repeat threshold");
    assert_eq!(
        loop_guard_error.field(),
        "upstreams.loop_guard.max_repeated_inputs"
    );

    let retry_ladder_error = parse_config_text(
        r#"
[[upstreams]]
name = "profile"
base_url = "http://profile.example/v1"
match_models = ["profile-model"]

[[upstreams.retry.ladder]]
name = " "
"#,
    )
    .expect("profile retry ladder config should parse before validation")
    .validate()
    .expect_err("profile retry ladder should reject a blank name");
    assert_eq!(retry_ladder_error.field(), "upstreams.retry.ladder.name");
}
