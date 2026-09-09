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
    forced_alias_reload_config_with_extra(match_model, forced_upstream_model, "")
}

fn forced_alias_reload_config_with_extra(
    match_model: &str,
    forced_upstream_model: &str,
    extra: &str,
) -> AppConfig {
    forced_alias_reload_config_with_routing(
        match_model,
        forced_upstream_model,
        "http://current-legacy.example/v1",
        None,
        extra,
    )
}

fn forced_alias_reload_config_with_routing(
    match_model: &str,
    forced_upstream_model: &str,
    legacy_upstream_base_url: &str,
    reserved_model: Option<&str>,
    extra: &str,
) -> AppConfig {
    let reserved_model = reserved_model.map_or(String::new(), |model| {
        format!("reserved_ingress_model_ids = [\"{model}\"]\n")
    });
    let config = AppConfig::parse(&format!(
        r#"
[upstream]
base_url = "{legacy_upstream_base_url}"
{reserved_model}

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
{extra}
"#
    ))
    .expect("forced alias reload fixture should parse");
    config
        .validate()
        .expect("forced alias reload fixture should validate");
    config
}

#[test]
fn forced_aliases_follow_retained_routing_topology_generation() {
    let current = forced_alias_reload_config("current-canonical", "current-canonical");
    let mut renamed_profile =
        forced_alias_reload_config("current-canonical", "requested-canonical");
    renamed_profile.upstream_profiles[0].name = String::from("renamed-forced-target");
    let cases = [
        (
            "named upstream profile topology",
            renamed_profile,
            Some("upstreams.topology"),
        ),
        (
            "match models",
            forced_alias_reload_config("requested-canonical", "requested-canonical"),
            None,
        ),
        (
            "legacy upstream",
            forced_alias_reload_config_with_routing(
                "current-canonical",
                "requested-canonical",
                "http://requested-legacy.example/v1",
                None,
                "",
            ),
            Some("upstream.base_url"),
        ),
        (
            "listener policy",
            forced_alias_reload_config_with_extra(
                "current-canonical",
                "requested-canonical",
                "\n[[listeners]]\nname = \"routing-listener\"\nbind_host = \"127.0.0.1\"\nport = 18010\nallowed_upstreams = [\"default\"]\n",
            ),
            Some("listeners.topology"),
        ),
        (
            "public model aliases",
            forced_alias_reload_config_with_extra(
                "current-canonical",
                "requested-canonical",
                "\n[[model_aliases]]\nid = \"public-routing-alias\"\nkind = \"upstream\"\nupstream_profile = \"default\"\n",
            ),
            Some("model_aliases.topology"),
        ),
        (
            "alias only",
            forced_alias_reload_config("current-canonical", "reloaded-canonical"),
            None,
        ),
    ];

    for (name, requested, restart_field) in cases {
        requested
            .validate()
            .expect("requested routing fixture should validate");
        let (next, outcome) = apply_reloadable(&current, &requested);
        assert_eq!(
            &next.forced_model_alias_profiles,
            restart_field.map_or(&requested.forced_model_alias_profiles, |_| {
                &current.forced_model_alias_profiles
            }),
            "{name}: forced aliases must stay in the retained routing generation"
        );
        assert_eq!(
            outcome
                .restart_required_changes
                .iter()
                .any(|change| Some(change.field) == restart_field),
            restart_field.is_some(),
            "{name}: expected restart-required routing field"
        );
    }
}

#[test]
fn forced_alias_and_reserved_ids_follow_retained_routing_generation() {
    let current = forced_alias_reload_config_with_routing(
        "current-canonical",
        "current-canonical",
        "http://current-legacy.example/v1",
        Some("current-canonical"),
        "",
    );
    let requested = forced_alias_reload_config_with_routing(
        "current-canonical",
        "requested-canonical",
        "http://current-legacy.example/v1",
        Some("requested-canonical"),
        "\n[[listeners]]\nname = \"routing-listener\"\nbind_host = \"127.0.0.1\"\nport = 18010\nallowed_upstreams = [\"default\"]\n",
    );

    let (next, outcome) = apply_reloadable(&current, &requested);

    assert_eq!(
        outcome
            .restart_required_changes
            .iter()
            .map(|change| change.field)
            .collect::<Vec<_>>(),
        vec!["listeners.topology"]
    );
    assert_eq!(
        next.forced_model_alias_profiles,
        current.forced_model_alias_profiles
    );
    assert_eq!(
        next.upstream.reserved_ingress_model_ids,
        current.upstream.reserved_ingress_model_ids
    );
}
