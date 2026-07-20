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
