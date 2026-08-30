use super::super::{AppConfig, apply_reloadable};

fn endpoint_reload_config(endpoints: &str) -> AppConfig {
    let config = AppConfig::parse(&format!(
        r#"
[[upstreams]]
name = "reranker"
model = "reranker-model"
health_probe_interval = "1ms"
health_probe_timeout = "1ms"
health_probe_max_wait = "2ms"
{endpoints}
"#
    ))
    .expect("endpoint reload fixture should parse");
    config
        .validate()
        .expect("endpoint reload fixture should validate");
    config
}

fn assert_endpoint_reload_applied(current: &AppConfig, requested: &AppConfig) {
    let (next, outcome) = apply_reloadable(current, requested);
    let active = &next.upstream_profiles[0];
    let expected = &requested.upstream_profiles[0];

    assert!(outcome.applied);
    assert!(
        outcome
            .restart_required_changes
            .iter()
            .all(|change| change.field != "upstreams.topology"),
        "reloadable endpoint changes must not be classified as topology changes"
    );
    assert_eq!(active.base_url, expected.base_url);
    assert_eq!(active.endpoints, expected.endpoints);
}

#[test]
fn primary_endpoint_url_replacement_hot_reloads() {
    let current = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "http://old-primary.example/v1"
priority = "primary"

[[profile.upstream]]
base_url = "http://backup.example/v1"
priority = "failover"
"#,
    );
    let requested = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "http://new-primary.example/v1"
priority = "primary"

[[profile.upstream]]
base_url = "http://backup.example/v1"
priority = "failover"
"#,
    );

    assert_endpoint_reload_applied(&current, &requested);
}

#[test]
fn primary_and_failover_priority_swap_hot_reloads() {
    let current = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "http://first.example/v1"
priority = "primary"

[[profile.upstream]]
base_url = "http://second.example/v1"
priority = "failover"
"#,
    );
    let requested = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "http://first.example/v1"
priority = "failover"

[[profile.upstream]]
base_url = "http://second.example/v1"
priority = "primary"
"#,
    );

    assert_endpoint_reload_applied(&current, &requested);
}

#[test]
fn combined_endpoint_identity_fields_hot_reload_coherently() {
    let current = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "http://old-primary.example/v1"
priority = "primary"
protocol = "openai"

[[profile.upstream]]
base_url = "http://backup.example/v1"
priority = "failover"
protocol = "openai"
"#,
    );
    let requested = endpoint_reload_config(
        r#"
[[profile.upstream]]
base_url = "https://api.deepinfra.example/v1/inference"
priority = "primary"
protocol = "deepinfra_qwen3_rerank"
model = "Qwen/Qwen3-Reranker-8B"
model_revision = "2222222222222222222222222222222222222222"
api_key_env = "LLM_GUARD_PROXY_RELOADED_DEEPINFRA_KEY"

[[profile.upstream]]
base_url = "http://backup.example/v1"
priority = "failover"
protocol = "openai"
"#,
    );

    assert_endpoint_reload_applied(&current, &requested);
}

fn forced_alias_reload_config(match_model: &str, forced_upstream_model: &str) -> AppConfig {
    let config = AppConfig::parse(&format!(
        r#"
[[upstreams]]
name = "forced-target"
base_url = "http://profile.example/v1"
match_models = ["{match_model}"]

[[forced_model_alias_profiles]]
alias = "forced-public-alias"
upstream_model = "{forced_upstream_model}"
thinking_mode = "force_disable"
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#
    ))
    .expect("forced alias reload fixture should parse");
    config
        .validate()
        .expect("forced alias reload fixture should validate");
    config
}

#[test]
fn forced_aliases_follow_retained_upstream_routing_generation() {
    let current = forced_alias_reload_config("current-canonical", "current-canonical");
    let requested = forced_alias_reload_config("requested-canonical", "requested-canonical");

    let (next, outcome) = apply_reloadable(&current, &requested);

    assert!(
        outcome
            .restart_required_changes
            .iter()
            .any(|change| change.field == "upstreams.topology")
    );
    assert_eq!(next.upstream_profiles, current.upstream_profiles);
    assert_eq!(
        next.forced_model_alias_profiles, current.forced_model_alias_profiles,
        "forced aliases must stay in the routing generation retained for restart"
    );

    let alias_only_requested =
        forced_alias_reload_config("current-canonical", "reloaded-canonical");
    let (next, outcome) = apply_reloadable(&current, &alias_only_requested);
    assert!(outcome.applied);
    assert_eq!(
        next.forced_model_alias_profiles, alias_only_requested.forced_model_alias_profiles,
        "forced-alias-only reloads remain live when routing topology is unchanged"
    );
}
