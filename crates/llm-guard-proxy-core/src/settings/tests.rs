#[cfg(unix)]
use std::os::unix::fs::{PermissionsExt, symlink};
use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use super::model::{
    evidence_blob_cache_dir_from_xdg_cache_home, evidence_sqlite_path_from_xdg_state_home,
};
use super::{
    AppConfig, ConfigManager, ConfigParseError, DefaultInjectionSchema, DownstreamDropPolicy,
    HeartbeatMode, LoopGuardMode, MissingConfigPolicy, NoThinkingMarkerPolicy, RELOADABLE_FIELDS,
    RESTART_REQUIRED_FIELDS, ThinkingMode, ToolRequestThinkingPolicy, UpstreamRouteReason,
    ValidationError, parse::parse_config_text, redact_upstream_base_url,
};
#[cfg(feature = "guard")]
use crate::{
    AliasKind, AliasTarget, DEFAULT_PROFILE_NAME, ModelAliasResolver, ProfileConfig, ProfileKind,
    ShieldedBuffering, UnknownKeyPolicy, WorkflowRuntime,
};
#[cfg(feature = "family")]
use crate::{
    CHILD_SAFE_DAILY_REQUEST_LIMIT, CHILD_SAFE_MODEL_ALIAS, CHILD_SAFE_PROFILE_NAME,
    CategoryAction, FAMILY_GUARD_PACK_NAME, FamilyCategory, FamilyPolicyOutcome,
};

#[cfg(feature = "guard")]
const FULL_OVERRIDE_CONFIG: &str = r#"
[server]
port = 18100
max_in_flight_requests = 2
max_queued_generation_requests = 3
generation_queue_timeout_ms = 4000
generation_queue_full_status = 429
generation_queue_retry_after_secs = 30
max_control_plane_in_flight_requests = 5
max_request_body_bytes = 1048576

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["qwen3-embedding-8b"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005

[upstream.metadata]
context_length_override = 256000
max_model_len_override = 256000

[upstream]
request_timeout_ms = 90000

[[upstreams]]
name = "qwen3-embedding-8b"
base_url = "http://embedding.example/v1"
match_models = ["embedding-model"]

[[model_aliases]]
id = "gpt-default"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "family/child-safe-general-v1"
kind = "workflow"
workflow_id = "family.child_safe_general.v1"
workflow_timeout_ms = 120000

[workflows.family.child_safe_general.v1]
runtime_kind = "stdio"
command = "python"
args = ["workflows/content_review.py"]
timeout_ms = 120000
max_stdout_bytes = 1048576

[guard_workflows]
pre_request = "family.child_safe_general.v1"
post_response = "family.child_safe_general.v1"
fail_closed_blocks = false

[profiles.child_default]
kind = "child"
allowed_models = ["family/child-safe-general-v1"]
daily_request_limit = 50
shielded_buffering = "buffered_sse"
guard_pack = "family_basic"

[profiles.adult_default]
kind = "adult"
allowed_models = ["gpt-default", "family/child-safe-general-v1"]
shielded_buffering = "off"

[virtual_keys]
enabled = true
unknown_key_policy = "use_default_profile"

[virtual_keys.keys]
vk_adult_abc123 = "adult_default"
vk_child_def456 = "child_default"

[budget]
enabled = true
sqlite_path = "state/llm-guard-proxy-test-budget.sqlite3"
reset_timezone = "UTC"
reset_hour_utc = 4

[observability]
metrics_enabled = false
health_upstream_probe_enabled = false
health_upstream_probe_timeout_ms = 250
debug_summary_enabled = true
debug_summary_admin_token = "test-admin-token"
debug_summary_max_records = 7

[observability.retention]
max_records = 50
prune_to_records = 40

[evidence]
enabled = true
sqlite_path = "state/llm-guard-proxy-test-evidence.sqlite3"
blob_cache_dir = "cache/llm-guard-proxy-test-evidence-blobs"
include_raw_payloads = true
include_request_headers = true
max_bytes = 1000
prune_to_bytes = 800
max_records = 50
prune_to_records = 40

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
parallel_downgrade_attempts = false
max_shadow_attempts_per_request = 1
max_global_shadow_in_flight = 3
shadow_attempt_timeout_ms = 100

[thinking]
force_disable = true
tool_request_policy = "passthrough"

[heartbeat]
mode = "json-whitespace"
interval_secs = 5

[loop_guard]
mode = "monitor"
output_repeated_line_threshold = 40
output_token_window_size = 8
output_repeated_token_window_threshold = 9
output_suffix_cycle_threshold = 10
output_low_progress_min_bytes = 2048
output_low_progress_unique_ratio_percent = 25
input_overlap_threshold_multiplier = 5

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true
downstream_drop_policy = "detach"

[upstream.stall]
enabled = true
idle_timeout_ms = 5000
recovery_command = ["/usr/bin/systemctl", "--user", "restart", "vllm-aeon-27b-dflash-n12.service"]
recovery_timeout_ms = 60000
recovery_cooldown_ms = 45000
recovery_budget_window_ms = 180000
recovery_max_per_window = 1

[cloudflare]
enabled = false
"#;

#[test]
fn defaults_match_issue_contract() {
    let config = AppConfig::default();

    config.validate().expect("default config should validate");
    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_009);
    assert_eq!(config.server.max_in_flight_requests, 16);
    assert_eq!(config.server.max_queued_generation_requests, 64);
    assert_eq!(config.server.generation_queue_timeout_ms, 30_000);
    assert_eq!(config.server.generation_queue_full_status, 503);
    assert_eq!(config.server.generation_queue_retry_after_secs, None);
    assert_eq!(config.server.max_control_plane_in_flight_requests, 128);
    assert_eq!(config.server.max_request_body_bytes, 67_108_864);
    assert_eq!(config.upstream.base_url, "http://gb10:18009/v1");
    assert_eq!(config.upstream.request_timeout_ms, 120_000);
    assert!(config.upstream.metadata.discovery_enabled);
    assert!(config.upstream.metadata.enrich_responses);
    assert!(config.shielding.enabled);
    assert!(config.observability.enabled);
    assert!(!config.observability.capture_raw_payloads);
    assert!(config.observability.metrics_enabled.is_enabled());
    assert!(
        config
            .observability
            .health_upstream_probe_enabled
            .is_enabled()
    );
    assert_eq!(config.observability.health_upstream_probe_timeout_ms, 500);
    assert!(!config.observability.debug_summary_enabled.is_enabled());
    assert_eq!(config.observability.debug_summary_admin_token, None);
    assert_eq!(config.observability.debug_summary_max_records, 20);
    assert_eq!(config.observability.retention.max_records, 100_000);
    assert_eq!(config.observability.retention.prune_to_records, None);
    assert_eq!(
        config.observability.retention.effective_prune_to_records(),
        80_000
    );
    assert_default_evidence_config(&config);
    #[cfg(feature = "guard")]
    {
        assert!(!config.budget.enabled);
        assert_eq!(
            config.budget.sqlite_path,
            "~/.local/state/llm-guard-proxy/budget.sqlite3"
        );
        assert_eq!(config.budget.reset_timezone, "UTC");
        assert_eq!(config.budget.reset_hour_utc, 0);
    }
    assert!(config.thinking.enabled);
    assert_eq!(config.thinking.mode, ThinkingMode::BoundedThinking);
    assert!(!config.thinking.force_disable);
    assert_eq!(config.thinking.max_tokens, None);
    assert_eq!(config.thinking.budget_tokens, 32_768);
    assert_eq!(
        config.thinking.tool_request_policy,
        ToolRequestThinkingPolicy::Apply
    );
    assert_eq!(
        config.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::Force
    );
    assert_eq!(
        config.thinking.default_injection_schema,
        DefaultInjectionSchema::Canonical
    );
    assert!(config.loop_guard.enabled);
    assert_eq!(config.loop_guard.mode, LoopGuardMode::Monitor);
    assert_eq!(config.loop_guard.effective_mode(), LoopGuardMode::Monitor);
    assert_default_loop_guard_fields(&config);
    assert!(config.retry.enabled);
    assert_eq!(config.retry.max_attempts, 5);
    assert!(config.retry.anti_loop_hint_enabled);
    assert!(!config.retry.shielded_streaming_enabled);
    assert_eq!(
        config.retry.downstream_drop_policy,
        DownstreamDropPolicy::Cancel
    );
    assert!(config.retry.ladder.is_empty());
    assert!(!config.upstream_stall.enabled);
    assert_eq!(config.upstream_stall.idle_timeout_ms, 30_000);
    assert!(config.upstream_stall.recovery_command.is_empty());
    assert_eq!(config.upstream_stall.recovery_timeout_ms, 300_000);
    assert_eq!(config.upstream_stall.recovery_cooldown_ms, 300_000);
    assert_eq!(config.upstream_stall.recovery_budget_window_ms, 900_000);
    assert_eq!(config.upstream_stall.recovery_max_per_window, 2);
    assert_eq!(config.heartbeat.mode, HeartbeatMode::Sse);
    assert!(config.cloudflare.enabled);
    assert!(config.upstream_profiles.is_empty());
    #[cfg(feature = "guard")]
    {
        assert!(config.model_aliases.is_empty());
        assert!(config.workflows.is_empty());
    }
    assert_eq!(config.default_upstream_profile().name, "default");
}

fn assert_default_loop_guard_fields(config: &AppConfig) {
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
    assert!(config.loop_guard.reasoning_semantic_detection_enabled);
    assert_eq!(
        config
            .loop_guard
            .reasoning_semantic_similarity_threshold_percent,
        55
    );
    assert_eq!(config.loop_guard.reasoning_semantic_window_token_count, 24);
    assert_eq!(config.loop_guard.reasoning_semantic_minimum_token_count, 8);
    assert_eq!(
        config.loop_guard.reasoning_semantic_history_window_count,
        16
    );
}

#[test]
fn parses_no_thinking_marker_policy_values() {
    let respect = parse_config_text(
        r#"
[thinking]
no_thinking_marker_policy = "respect_no_thinking_markers"
"#,
    )
    .expect("respect marker policy should parse");
    assert_eq!(
        respect.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::RespectNoThinkingMarkers
    );

    let escape_hatch = parse_config_text(
        r#"
[thinking]
thinking.no_thinking_marker_policy = "escape_hatch_only"
"#,
    )
    .expect("escape hatch marker policy should parse");
    assert_eq!(
        escape_hatch.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::EscapeHatchOnly
    );
}

#[test]
fn parses_default_injection_schema_values() {
    let chat_template_kwargs = parse_config_text(
        r#"
[thinking]
default_injection_schema = "chat_template_kwargs"
"#,
    )
    .expect("chat template kwargs schema should parse");
    assert_eq!(
        chat_template_kwargs.thinking.default_injection_schema,
        DefaultInjectionSchema::ChatTemplateKwargs
    );

    let canonical = parse_config_text(
        r#"
[thinking]
thinking.default_injection_schema = "canonical"
"#,
    )
    .expect("canonical schema alias should parse");
    assert_eq!(
        canonical.thinking.default_injection_schema,
        DefaultInjectionSchema::Canonical
    );
}

#[test]
fn rejects_invalid_default_injection_schema() {
    let error = parse_config_text(
        r#"
[thinking]
default_injection_schema = "qwen"
"#,
    )
    .expect_err("invalid default injection schema should fail");

    assert_eq!(error.line(), 3);
    assert!(
        error
            .message()
            .contains("invalid thinking.default_injection_schema")
    );
    assert!(error.message().contains("canonical"));
    assert!(error.message().contains("chat_template_kwargs"));
}

#[test]
fn rejects_invalid_no_thinking_marker_policy() {
    let error = parse_config_text(
        r#"
[thinking]
no_thinking_marker_policy = "sometimes"
"#,
    )
    .expect_err("invalid marker policy should fail");

    assert_eq!(error.line(), 3);
    assert!(
        error
            .message()
            .contains("invalid thinking.no_thinking_marker_policy")
    );
    assert!(error.message().contains("force"));
    assert!(error.message().contains("respect_no_thinking_markers"));
    assert!(error.message().contains("escape_hatch_only"));
}

fn assert_default_evidence_config(config: &AppConfig) {
    assert!(!config.evidence.enabled);
    assert_eq!(
        config.evidence.sqlite_path,
        PathBuf::from("~/.local/state/llm-guard-proxy/evidence.sqlite3")
    );
    assert_eq!(
        config.evidence.blob_cache_dir,
        PathBuf::from("~/.cache/llm-guard-proxy/evidence/blobs")
    );
    assert!(!config.evidence.include_raw_payloads);
    assert!(!config.evidence.include_request_headers);
    assert_eq!(config.evidence.max_bytes, 10_737_418_240);
    assert_eq!(config.evidence.prune_to_bytes, 8_589_934_592);
    assert_eq!(config.evidence.max_records, 100_000);
    assert_eq!(config.evidence.effective_prune_to_records(), 80_000);
    assert!(!config.evidence.shadow.enabled);
    assert!(!config.evidence.shadow.keep_looping_attempt_running);
    assert!(config.evidence.shadow.parallel_downgrade_attempts);
    assert_eq!(config.evidence.shadow.max_shadow_attempts_per_request, 2);
    assert_eq!(config.evidence.shadow.max_global_shadow_in_flight, 2);
    assert_eq!(config.evidence.shadow.shadow_attempt_timeout_ms, 7_200_000);
}

#[test]
fn evidence_default_paths_follow_xdg_inputs_and_home_fallbacks() {
    assert_eq!(
        evidence_sqlite_path_from_xdg_state_home(Some(PathBuf::from("/tmp/xdg-state"))),
        PathBuf::from("/tmp/xdg-state/llm-guard-proxy/evidence.sqlite3")
    );
    assert_eq!(
        evidence_blob_cache_dir_from_xdg_cache_home(Some(PathBuf::from("/tmp/xdg-cache"))),
        PathBuf::from("/tmp/xdg-cache/llm-guard-proxy/evidence/blobs")
    );
    assert_eq!(
        evidence_sqlite_path_from_xdg_state_home(None),
        PathBuf::from("~/.local/state/llm-guard-proxy/evidence.sqlite3")
    );
    assert_eq!(
        evidence_blob_cache_dir_from_xdg_cache_home(None),
        PathBuf::from("~/.cache/llm-guard-proxy/evidence/blobs")
    );
}

#[test]
fn parses_retry_ladder_stream_shielding_and_drop_policy() {
    let config = parse_config_text(
        r#"
[retry]
enabled = true
max_attempts = 3
anti_loop_hint_enabled = true
shielded_streaming_enabled = true
downstream_drop_policy = "detach"

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768
no_thinking_marker_policy = "respect_no_thinking_markers"

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
anti_loop_hint = "Previous attempt became repetitive. Answer directly."

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000
"#,
    )
    .expect("retry ladder config should parse");

    config.validate().expect("retry ladder should validate");
    assert!(config.retry.shielded_streaming_enabled);
    assert_eq!(
        config.retry.downstream_drop_policy,
        DownstreamDropPolicy::Detach
    );
    assert_eq!(config.retry.ladder.len(), 3);
    assert_eq!(config.retry.ladder[0].name, "max-thinking");
    assert_eq!(
        config.retry.ladder[0].thinking.mode,
        ThinkingMode::ForceThinking
    );
    assert_eq!(config.retry.ladder[0].thinking.max_tokens, Some(50_000));
    assert_eq!(config.retry.ladder[0].thinking.budget_tokens, 32_768);
    assert_eq!(
        config.retry.ladder[0].thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::RespectNoThinkingMarkers
    );
    assert_eq!(config.retry.ladder[1].name, "bounded-thinking");
    assert_eq!(config.retry.ladder[1].thinking.budget_tokens, 8_192);
    assert_eq!(
        config.retry.ladder[1].anti_loop_hint.as_deref(),
        Some("Previous attempt became repetitive. Answer directly.")
    );
    assert_eq!(
        config.retry.ladder[2].thinking.mode,
        ThinkingMode::ForceDisable
    );
}

#[test]
fn validates_enabled_upstream_stall_idle_timeout_precedes_request_timeout() {
    let mut config = AppConfig::default();
    config.upstream_stall.enabled = true;
    config.upstream_stall.idle_timeout_ms = config.upstream.request_timeout_ms;

    let error = config
        .validate()
        .expect_err("stall idle timeout should beat total upstream request timeout");

    assert_eq!(error.field(), "upstream.stall.idle_timeout_ms");
    assert!(
        error
            .message()
            .contains("less than upstream.request_timeout_ms")
    );
}

#[test]
fn parses_named_upstream_profiles_and_preserves_legacy_default_profile() {
    let config = parse_config_text(
        r#"
[upstream]
base_url = "http://default.example/v1"
request_timeout_ms = 120000

[upstream.metadata]
context_length_override = 4096
input_token_safety_margin = 64

[thinking]
mode = "bounded_thinking"
thinking_token_budget = 1234
budget_accounting = "preserve_answer_budget"

[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate", "qwen3.6-27b-decensor-by-aeon"]
request_timeout_ms = 7200000
max_in_flight_requests = 8
max_queued_generation_requests = 16

[upstreams.metadata]
discovery_enabled = true
context_length_override = 262144
input_token_safety_margin = 2048

[upstreams.thinking]
mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768
budget_accounting = "total_cap"
apply_to_tool_requests = true
no_thinking_marker_policy = "escape_hatch_only"
default_injection_schema = "chat_template_kwargs"

[[upstreams]]
name = "fast-no-think"
base_url = "http://127.0.0.1:18100/v1"
match_models = ["fast-local"]

[upstreams.thinking]
mode = "force_disable"
"#,
    )
    .expect("profile config should parse");

    assert_eq!(config.upstream.base_url, "http://default.example/v1");
    assert_eq!(config.thinking.budget_tokens, 1_234);
    assert_eq!(
        config.upstream.metadata.context_length_override,
        Some(4_096)
    );
    assert_eq!(config.upstream.metadata.input_token_safety_margin, 64);

    assert_eq!(config.upstream_profiles.len(), 2);
    let aeon = &config.upstream_profiles[0];
    assert_eq!(aeon.name, "aeon-chat");
    assert_eq!(aeon.base_url, "http://aeon.example/v1");
    assert_eq!(
        aeon.match_models,
        vec![
            String::from("aeon-ultimate"),
            String::from("qwen3.6-27b-decensor-by-aeon"),
        ]
    );
    assert_eq!(aeon.request_timeout_ms, 7_200_000);
    assert_eq!(aeon.max_in_flight_requests, Some(8));
    assert_eq!(aeon.max_queued_generation_requests, Some(16));
    assert_eq!(aeon.metadata.context_length_override, Some(262_144));
    assert_eq!(aeon.metadata.input_token_safety_margin, 2_048);
    assert_eq!(aeon.thinking.mode, ThinkingMode::ForceThinking);
    assert_eq!(aeon.thinking.max_tokens, Some(50_000));
    assert_eq!(aeon.thinking.budget_tokens, 32_768);
    assert!(!aeon.thinking.preserve_answer_budget);
    assert_eq!(
        aeon.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::EscapeHatchOnly
    );
    assert_eq!(
        aeon.thinking.default_injection_schema,
        DefaultInjectionSchema::ChatTemplateKwargs
    );

    let fast = &config.upstream_profiles[1];
    assert_eq!(fast.name, "fast-no-think");
    assert_eq!(fast.match_models, vec![String::from("fast-local")]);
    assert_eq!(fast.thinking.mode, ThinkingMode::ForceDisable);

    config.validate().expect("profile config should validate");
}

#[cfg(feature = "param-override")]
#[test]
fn parses_upstream_profile_param_override() {
    let config = parse_config_text(
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]

[upstreams.param_override]
enabled = true
temperature = 0.6
top_p = 0.95
top_k = 40
max_tokens = 2048
frequency_penalty = 0.1
presence_penalty = -0.2
"#,
    )
    .expect("param override config should parse");

    let override_config = &config.upstream_profiles[0].param_override;
    assert!(override_config.enabled);
    assert_eq!(override_config.temperature, Some(0.6));
    assert_eq!(override_config.top_p, Some(0.95));
    assert_eq!(override_config.top_k, Some(40));
    assert_eq!(override_config.max_tokens, Some(2_048));
    assert_eq!(override_config.frequency_penalty, Some(0.1));
    assert_eq!(override_config.presence_penalty, Some(-0.2));
    config
        .validate()
        .expect("param override config should validate");
}

#[test]
fn parses_hot_restart_overrides_for_default_and_named_upstreams() {
    let config = parse_config_text(
        r#"
[upstream.hot_restart]
probe_interval_secs = 3
probe_timeout_secs = 30
probe_messages = [{"role":"user","content":"ready?"}]
probe_chat_template_kwargs = {"enable_thinking":false}

[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]

[upstreams.hot_restart]
enabled = false
probe_max_tokens = 2
probe_interval_secs = 2
probe_timeout_secs = 20
probe_messages = [{"role":"user","content":"probe"}]
probe_chat_template_kwargs = null
"#,
    )
    .expect("hot restart config should parse");

    assert_eq!(config.upstream.hot_restart.probe_interval_secs, 3);
    assert_eq!(
        config.upstream.hot_restart.probe_messages,
        serde_json::json!([{"role":"user","content":"ready?"}])
    );
    let aeon = &config.upstream_profiles[0];
    assert!(!aeon.hot_restart.enabled);
    assert_eq!(aeon.hot_restart.probe_max_tokens, 2);
    assert_eq!(aeon.hot_restart.probe_chat_template_kwargs, None);
    config
        .validate()
        .expect("hot restart config should validate");
}

#[test]
fn validates_hot_restart_probe_bounds_and_shape() {
    let config = parse_config_text(
        r"
[upstream.hot_restart]
probe_interval_secs = 10
probe_timeout_secs = 5
",
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("probe interval above timeout should fail");
    assert_eq!(error.field(), "upstream.hot_restart.probe_interval_secs");

    let config = parse_config_text(
        r#"
[[upstreams]]
name = "bad-probe"
base_url = "http://example.test/v1"
match_models = ["bad-probe"]

[upstreams.hot_restart]
probe_messages = {}
"#,
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("probe messages must be an array");
    assert_eq!(error.field(), "upstreams.hot_restart.probe_messages");
}

#[test]
fn selects_named_profile_by_model_and_defaults_without_match() {
    let config = parse_config_text(
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]

[[upstreams]]
name = "fast-no-think"
base_url = "http://fast.example/v1"
match_models = ["fast-local"]
"#,
    )
    .expect("profile config should parse");

    let aeon = config.select_upstream_profile(Some("aeon-ultimate"));
    assert_eq!(aeon.profile.name, "aeon-chat");
    assert_eq!(aeon.route_reason, UpstreamRouteReason::MatchedModel);

    let fallback = config.select_upstream_profile(Some("unknown"));
    assert_eq!(fallback.profile.name, "default");
    assert_eq!(
        fallback.route_reason,
        UpstreamRouteReason::DefaultUnmatchedModel
    );

    let no_model = config.select_upstream_profile(None);
    assert_eq!(no_model.profile.name, "default");
    assert_eq!(no_model.route_reason, UpstreamRouteReason::DefaultNoModel);
}

#[test]
#[cfg(feature = "guard")]
fn parses_and_resolves_model_aliases_from_config() {
    let config = parse_config_text(
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]

[[model_aliases]]
id = "gpt-default"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "family/child-safe-general-v1"
kind = "workflow"
workflow_id = "family.child_safe_general.v1"
workflow_timeout_ms = 120000

[workflows."family.child_safe_general.v1"]
runtime_kind = "stdio"
command = "python"
args = ["workflows/content_review.py"]
"#,
    )
    .expect("alias config should parse");

    config.validate().expect("alias config should validate");

    let resolver = ModelAliasResolver::new(config.model_aliases.clone());
    assert_eq!(
        resolver.resolve("gpt-default"),
        Ok(AliasTarget::Upstream {
            profile_name: String::from("default"),
        })
    );
    assert_eq!(
        resolver.resolve("family/child-safe-general-v1"),
        Ok(AliasTarget::Workflow {
            workflow_id: String::from("family.child_safe_general.v1"),
            timeout_ms: 120_000,
        })
    );
}

#[test]
#[cfg(feature = "guard")]
fn validates_model_alias_requirements() {
    for (contents, field) in [
        (
            r#"
[[model_aliases]]
id = ""
kind = "upstream"
upstream_profile = "default"
"#,
            "model_aliases.id",
        ),
        (
            r#"
[[model_aliases]]
id = "dup"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "dup"
kind = "workflow"
workflow_id = "workflow.dup"
"#,
            "model_aliases.id",
        ),
        (
            r#"
[[model_aliases]]
id = "missing-profile"
kind = "upstream"
"#,
            "model_aliases.upstream_profile",
        ),
        (
            r#"
[[model_aliases]]
id = "unknown-profile"
kind = "upstream"
upstream_profile = "missing"
"#,
            "model_aliases.upstream_profile",
        ),
        (
            r#"
[[model_aliases]]
id = "missing-workflow"
kind = "workflow"
"#,
            "model_aliases.workflow_id",
        ),
    ] {
        let config = parse_config_text(contents).expect("config syntax should parse");
        let error = config.validate().expect_err("alias config should fail");
        assert_eq!(error.field(), field);
    }
}

#[test]
#[cfg(feature = "guard")]
fn parses_workflow_config_with_defaults_and_overrides() {
    let config = parse_config_text(
        r#"
[workflows.family.content_review.v1]
runtime_kind = "stdio"
command = "python"
args = ["workflows/content_review.py"]
"#,
    )
    .expect("workflow config should parse");

    config.validate().expect("workflow config should validate");
    let workflow = config
        .workflows
        .get("family.content_review.v1")
        .expect("workflow should be present");
    assert_eq!(workflow.runtime_kind, WorkflowRuntime::Stdio);
    assert_eq!(workflow.command, "python");
    assert_eq!(
        workflow.args,
        vec![String::from("workflows/content_review.py")]
    );
    assert_eq!(workflow.timeout_ms, 120_000);
    assert_eq!(workflow.max_stdout_bytes, 1_048_576);

    let config = parse_config_text(
        r#"
[workflows.family.content_review.v1]
runtime_kind = "stdio"
command = "/bin/sh"
args = ["review.sh"]
timeout_ms = 600000
max_stdout_bytes = 2048
"#,
    )
    .expect("workflow config overrides should parse");

    config
        .validate()
        .expect("workflow override config should validate");
    let workflow = config
        .workflows
        .get("family.content_review.v1")
        .expect("workflow should be present");
    assert_eq!(workflow.command, "/bin/sh");
    assert_eq!(workflow.args, vec![String::from("review.sh")]);
    assert_eq!(workflow.timeout_ms, 600_000);
    assert_eq!(workflow.max_stdout_bytes, 2_048);
}

#[test]
#[cfg(feature = "guard")]
fn rejects_invalid_workflow_config() {
    for (contents, field) in [
        (
            r#"
[workflows.bad]
runtime_kind = "stdio"
command = ""
"#,
            "workflows.command",
        ),
        (
            r#"
[workflows.bad]
runtime_kind = "stdio"
command = "python"
timeout_ms = 0
"#,
            "workflows.timeout_ms",
        ),
        (
            r#"
[workflows.bad]
runtime_kind = "stdio"
command = "python"
timeout_ms = 600001
"#,
            "workflows.timeout_ms",
        ),
        (
            r#"
[workflows.bad]
runtime_kind = "stdio"
command = "python"
max_stdout_bytes = 0
"#,
            "workflows.max_stdout_bytes",
        ),
    ] {
        let config = parse_config_text(contents).expect("config syntax should parse");
        let error = config
            .validate()
            .expect_err("workflow config should fail validation");
        assert_eq!(error.field(), field);
    }
}

#[test]
#[cfg(feature = "guard")]
fn rejects_invalid_workflow_runtime_kind() {
    let error = parse_config_text(
        r#"
[workflows.bad]
runtime_kind = "http"
command = "python"
"#,
    )
    .expect_err("invalid workflow runtime should fail");

    assert_eq!(error.line(), 3);
    assert!(error.message().contains("invalid workflows.runtime_kind"));
    assert!(error.message().contains("stdio"));
}

#[test]
#[cfg(feature = "guard")]
fn rejects_invalid_model_alias_kind() {
    let error = parse_config_text(
        r#"
[[model_aliases]]
id = "bad-kind"
kind = "external"
upstream_profile = "default"
"#,
    )
    .expect_err("invalid alias kind should fail");

    assert_eq!(error.line(), 4);
    assert!(error.message().contains("invalid model_aliases.kind"));
    assert!(error.message().contains("upstream"));
    assert!(error.message().contains("workflow"));
}

#[test]
fn validates_named_upstream_profile_uniqueness_and_required_metadata() {
    for (contents, field) in [
        (
            r#"
[[upstreams]]
name = "dup"
base_url = "http://one.example/v1"
match_models = ["one"]

[[upstreams]]
name = "dup"
base_url = "http://two.example/v1"
match_models = ["two"]
"#,
            "upstreams.name",
        ),
        (
            r#"
[[upstreams]]
name = "one"
base_url = "http://one.example/v1"
match_models = ["same"]

[[upstreams]]
name = "two"
base_url = "http://two.example/v1"
match_models = ["same"]
"#,
            "upstreams.match_models",
        ),
        (
            r#"
[[upstreams]]
name = ""
base_url = "http://one.example/v1"
match_models = ["one"]
"#,
            "upstreams.name",
        ),
        (
            r#"
[[upstreams]]
name = "bad-url"
base_url = "ftp://one.example/v1"
match_models = ["one"]
"#,
            "upstream.base_url",
        ),
        (
            r#"
[[upstreams]]
name = "bad-timeout"
base_url = "http://one.example/v1"
match_models = ["one"]
request_timeout_ms = 0
"#,
            "upstreams.request_timeout_ms",
        ),
        (
            r#"
[[upstreams]]
name = "bad-in-flight"
base_url = "http://one.example/v1"
match_models = ["one"]
max_in_flight_requests = 0
"#,
            "upstreams.max_in_flight_requests",
        ),
        (
            r#"
[[upstreams]]
name = "bad-queue"
base_url = "http://one.example/v1"
match_models = ["one"]
max_queued_generation_requests = 10001
"#,
            "upstreams.max_queued_generation_requests",
        ),
        (
            r#"
[[upstreams]]
name = "bad-metadata"
base_url = "http://one.example/v1"
match_models = ["one"]

[upstreams.metadata]
context_length_override = 0
"#,
            "upstream.metadata.context_length_override",
        ),
    ] {
        let config = parse_config_text(contents).expect("config syntax should parse");
        let error = config.validate().expect_err("profile config should fail");
        assert_eq!(error.field(), field);
    }
}

#[test]
#[allow(clippy::too_many_lines)]
#[cfg(feature = "guard")]
fn parses_toml_with_defaults_and_overrides() {
    let config = parse_config_text(FULL_OVERRIDE_CONFIG).expect("config should parse");

    assert_eq!(config.server.bind_host, "127.0.0.1");
    assert_eq!(config.server.port, 18_100);
    assert_eq!(config.server.max_in_flight_requests, 2);
    assert_eq!(config.server.max_queued_generation_requests, 3);
    assert_eq!(config.server.generation_queue_timeout_ms, 4_000);
    assert_eq!(config.server.generation_queue_full_status, 429);
    assert_eq!(config.server.generation_queue_retry_after_secs, Some(30));
    assert_eq!(config.server.max_control_plane_in_flight_requests, 5);
    assert_eq!(config.server.max_request_body_bytes, 1_048_576);
    assert_eq!(config.listeners.len(), 2);
    assert_eq!(config.listeners[0].name, "embedding-legacy");
    assert_eq!(config.listeners[0].bind_host, "127.0.0.1");
    assert_eq!(config.listeners[0].port, 18_002);
    assert_eq!(
        config.listeners[0].allowed_upstreams.as_deref(),
        Some(&[String::from("qwen3-embedding-8b")][..])
    );
    assert_eq!(config.listeners[1].name, "aggregate");
    assert_eq!(config.listeners[1].port, 18_005);
    assert_eq!(config.listeners[1].allowed_upstreams, None);
    let effective_listeners = config.effective_listeners();
    assert_eq!(effective_listeners.len(), 3);
    assert_eq!(effective_listeners[0].name, "default");
    assert_eq!(effective_listeners[0].port, 18_100);
    assert_eq!(
        config.effective_listener_addresses(),
        vec![
            String::from("127.0.0.1:18100"),
            String::from("127.0.0.1:18002"),
            String::from("127.0.0.1:18005"),
        ]
    );
    assert_eq!(config.upstream.base_url, "http://gb10:18009/v1");
    assert_eq!(config.upstream.request_timeout_ms, 90_000);
    assert!(config.thinking.force_disable);
    assert_eq!(
        config.thinking.tool_request_policy,
        ToolRequestThinkingPolicy::Passthrough
    );
    assert_eq!(
        config.upstream.metadata.context_length_override,
        Some(256_000)
    );
    assert_eq!(
        config.upstream.metadata.max_model_len_override,
        Some(256_000)
    );
    #[cfg(feature = "guard")]
    assert_parsed_model_aliases(&config);
    #[cfg(feature = "guard")]
    {
        assert_eq!(
            config.guard_workflows.pre_request.as_deref(),
            Some("family.child_safe_general.v1")
        );
        assert_eq!(
            config.guard_workflows.post_response.as_deref(),
            Some("family.child_safe_general.v1")
        );
        assert!(!config.guard_workflows.fail_closed_blocks);
    }
    #[cfg(feature = "guard")]
    assert_parsed_profiles(&config);
    #[cfg(feature = "guard")]
    assert_parsed_virtual_keys(&config);
    #[cfg(feature = "guard")]
    {
        assert!(config.budget.enabled);
        assert_eq!(
            config.budget.sqlite_path,
            "state/llm-guard-proxy-test-budget.sqlite3"
        );
        assert_eq!(config.budget.reset_timezone, "UTC");
        assert_eq!(config.budget.reset_hour_utc, 4);
    }
    assert_parsed_observability_overrides(&config);
    assert!(config.evidence.enabled);
    assert_eq!(
        config.evidence.sqlite_path,
        PathBuf::from("state/llm-guard-proxy-test-evidence.sqlite3")
    );
    assert_eq!(
        config.evidence.blob_cache_dir,
        PathBuf::from("cache/llm-guard-proxy-test-evidence-blobs")
    );
    assert!(config.evidence.include_raw_payloads);
    assert!(config.evidence.include_request_headers);
    assert_eq!(config.evidence.max_bytes, 1_000);
    assert_eq!(config.evidence.prune_to_bytes, 800);
    assert_eq!(config.evidence.max_records, 50);
    assert_eq!(config.evidence.prune_to_records, Some(40));
    assert_eq!(config.evidence.effective_prune_to_records(), 40);
    assert!(config.evidence.shadow.enabled);
    assert!(config.evidence.shadow.keep_looping_attempt_running);
    assert!(!config.evidence.shadow.parallel_downgrade_attempts);
    assert_eq!(config.evidence.shadow.max_shadow_attempts_per_request, 1);
    assert_eq!(config.evidence.shadow.max_global_shadow_in_flight, 3);
    assert_eq!(config.evidence.shadow.shadow_attempt_timeout_ms, 100);
    assert_eq!(config.heartbeat.mode, HeartbeatMode::JsonWhitespace);
    assert_eq!(config.heartbeat.interval_secs, 5);
    assert_eq!(config.loop_guard.mode, LoopGuardMode::Monitor);
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
    assert_parsed_retry_overrides(&config);
    assert_parsed_upstream_stall_overrides(&config);
    assert!(!config.cloudflare.enabled);
}

#[cfg(feature = "guard")]
fn assert_parsed_profiles(config: &AppConfig) {
    assert_eq!(config.profiles.len(), 2);
    let child = config
        .profiles
        .get("child_default")
        .expect("child profile should parse");
    assert_eq!(child.kind, ProfileKind::Child);
    assert_eq!(
        child.allowed_models,
        vec![String::from("family/child-safe-general-v1")]
    );
    assert_eq!(child.daily_request_limit, 50);
    assert_eq!(child.shielded_buffering, ShieldedBuffering::BufferedSse);
    assert_eq!(child.guard_pack.as_deref(), Some("family_basic"));

    let adult = config
        .profiles
        .get("adult_default")
        .expect("adult profile should parse");
    assert_eq!(adult.kind, ProfileKind::Adult);
    assert_eq!(
        adult.allowed_models,
        vec![
            String::from("gpt-default"),
            String::from("family/child-safe-general-v1"),
        ]
    );
    assert_eq!(adult.daily_request_limit, 0);
    assert_eq!(adult.shielded_buffering, ShieldedBuffering::Off);
    assert_eq!(adult.guard_pack, None);
}

#[cfg(feature = "guard")]
fn assert_parsed_virtual_keys(config: &AppConfig) {
    assert!(config.virtual_keys.enabled);
    assert_eq!(
        config.virtual_keys.unknown_key_policy,
        UnknownKeyPolicy::UseDefaultProfile
    );
    assert_eq!(config.virtual_keys.keys.len(), 2);
    assert_eq!(
        config
            .virtual_keys
            .keys
            .get("vk_adult_abc123")
            .map(String::as_str),
        Some("adult_default")
    );
    assert_eq!(
        config
            .virtual_keys
            .keys
            .get("vk_child_def456")
            .map(String::as_str),
        Some("child_default")
    );
}

#[test]
#[cfg(feature = "guard")]
fn implicit_default_profile_exists_when_none_configured() {
    let config = AppConfig::default();

    config.validate().expect("default config should validate");
    let profile = config
        .caller_profile_by_name(DEFAULT_PROFILE_NAME)
        .expect("implicit default profile should exist");

    assert_eq!(profile.kind, ProfileKind::Adult);
    assert_eq!(profile.daily_request_limit, 0);
}

#[test]
#[cfg(feature = "guard")]
fn validates_caller_profile_requirements() {
    let config = parse_config_text(
        r#"
[[model_aliases]]
id = "gpt-default"
kind = "upstream"
upstream_profile = "default"

[profiles.child_default]
kind = "child"
allowed_models = ["missing-model"]
"#,
    )
    .expect("config syntax should parse");
    let error = config.validate().expect_err("profile config should fail");
    assert_eq!(error.field(), "profiles.allowed_models");
}

#[test]
#[cfg(feature = "guard")]
fn zero_daily_limit_is_unlimited() {
    let config = parse_config_text(
        r#"
[profiles.child_default]
kind = "child"
daily_request_limit = 0
"#,
    )
    .expect("config syntax should parse");

    config
        .validate()
        .expect("zero profile daily limit should validate");
    assert_eq!(
        config
            .profiles
            .get("child_default")
            .expect("profile should parse")
            .daily_request_limit,
        0
    );
}

#[test]
#[cfg(feature = "guard")]
fn validates_budget_requirements() {
    for (contents, field) in [
        (
            r#"
[budget]
sqlite_path = ""
"#,
            "budget.sqlite_path",
        ),
        (
            r#"
[budget]
reset_timezone = "America/Los_Angeles"
"#,
            "budget.reset_timezone",
        ),
        (
            r"
[budget]
reset_hour_utc = 24
",
            "budget.reset_hour_utc",
        ),
    ] {
        let config = parse_config_text(contents).expect("config syntax should parse");
        let error = config.validate().expect_err("budget config should fail");
        assert_eq!(error.field(), field);
    }
}

#[test]
#[cfg(feature = "guard")]
fn validates_empty_caller_profile_name() {
    let mut config = AppConfig::default();
    config
        .profiles
        .insert(String::new(), ProfileConfig::default());

    let error = config
        .validate()
        .expect_err("empty profile name should fail validation");

    assert_eq!(error.field(), "profiles.name");
}

#[test]
#[cfg(feature = "guard")]
fn duplicate_caller_profile_sections_fail_to_parse() {
    let error = parse_config_text(
        r#"
[profiles.child_default]
kind = "child"

[profiles.child_default]
kind = "adult"
"#,
    )
    .expect_err("duplicate profile sections should fail");

    assert!(
        error.message().contains("duplicate profile section"),
        "unexpected error: {error}"
    );
}

#[test]
#[cfg(feature = "guard")]
fn validates_virtual_key_requirements() {
    for (contents, field) in [
        (
            r#"
[profiles.default]
kind = "adult"

[virtual_keys.keys]
vk_known = "missing"
"#,
            "virtual_keys.keys",
        ),
        (
            r#"
[profiles.adult]
kind = "adult"

[profiles.child]
kind = "child"

[virtual_keys]
enabled = true
unknown_key_policy = "use_default_profile"
"#,
            "virtual_keys.unknown_key_policy",
        ),
    ] {
        let config = parse_config_text(contents).expect("config syntax should parse");
        let error = config
            .validate()
            .expect_err("virtual key config should fail");
        assert_eq!(error.field(), field);
    }
}

#[test]
#[cfg(feature = "guard")]
fn validates_guard_workflow_references() {
    let config = parse_config_text(
        r#"
[guard_workflows]
pre_request = "missing.guard"
"#,
    )
    .expect("config syntax should parse");

    let error = config
        .validate()
        .expect_err("missing guard workflow should fail validation");

    assert_eq!(error.field(), "guard_workflows.pre_request");
}

#[test]
#[cfg(feature = "family")]
fn parses_family_policy_and_creates_default_child_safe_profile() {
    let config = parse_config_text(
        r#"
[family]
enabled = true

[family.categories.self_harm]
enabled = true
action = "defer"

[family.categories.sexual_content]
enabled = true
action = "block"

[family.categories.violence]
enabled = true
action = "block"

[family.categories.drugs]
enabled = true
action = "block"

[family.categories.pii_disclosure]
enabled = true
action = "block"

[family.categories.emotional_dependency]
enabled = true
action = "replace"
replacement = "I'm here to help, but let's also talk to a trusted adult."

[family.categories.direct_homework_answer]
enabled = true
action = "replace"
replacement = "Let me help you understand the concept instead of giving the direct answer."

[family.categories.prompt_attack]
enabled = true
action = "block"
"#,
    )
    .expect("family config should parse");

    assert!(config.family.enabled);
    assert_eq!(
        config
            .family
            .category_config(FamilyCategory::SelfHarm)
            .action,
        CategoryAction::Defer
    );
    assert_eq!(
        config
            .family
            .category_config(FamilyCategory::DirectHomeworkAnswer)
            .replacement
            .as_deref(),
        Some("Let me help you understand the concept instead of giving the direct answer.")
    );

    let child_safe = config
        .profiles
        .get(CHILD_SAFE_PROFILE_NAME)
        .expect("family defaults should create child_safe profile");
    assert_eq!(child_safe.kind, ProfileKind::Child);
    assert_eq!(
        child_safe.allowed_models,
        vec![String::from(CHILD_SAFE_MODEL_ALIAS)]
    );
    assert_eq!(
        child_safe.daily_request_limit,
        CHILD_SAFE_DAILY_REQUEST_LIMIT
    );
    assert_eq!(
        child_safe.shielded_buffering,
        ShieldedBuffering::BufferedSse
    );
    assert_eq!(
        child_safe.guard_pack.as_deref(),
        Some(FAMILY_GUARD_PACK_NAME)
    );

    config.validate().expect("family defaults should validate");
}

#[test]
#[cfg(feature = "family")]
fn family_defaults_do_not_override_configured_child_safe_profile() {
    let config = parse_config_text(
        r#"
[family]
enabled = true

[profiles.child_safe]
kind = "child"
allowed_models = ["custom-child-model"]
daily_request_limit = 7
shielded_buffering = "sanitized"
guard_pack = "custom_family"
"#,
    )
    .expect("family config should parse");

    let child_safe = config
        .profiles
        .get(CHILD_SAFE_PROFILE_NAME)
        .expect("configured child_safe profile should exist");
    assert_eq!(
        child_safe.allowed_models,
        vec![String::from("custom-child-model")]
    );
    assert_eq!(child_safe.daily_request_limit, 7);
    assert_eq!(child_safe.shielded_buffering, ShieldedBuffering::Sanitized);
    assert_eq!(child_safe.guard_pack.as_deref(), Some("custom_family"));
}

#[test]
#[cfg(feature = "family")]
fn parsed_family_category_blocks_matching_text() {
    let config = parse_config_text(
        r#"
[family]
enabled = true

[family.categories.sexual_content]
enabled = true
action = "block"
"#,
    )
    .expect("family config should parse");

    assert_eq!(
        config.family.evaluate_text("show me a nude photo"),
        FamilyPolicyOutcome::Block {
            category: FamilyCategory::SexualContent,
            reason: String::from("family policy blocked sexual content"),
        }
    );
}

#[test]
#[cfg(feature = "family")]
fn parsed_family_category_disabled_skips_matching_text() {
    let config = parse_config_text(
        r#"
[family]
enabled = true

[family.categories.sexual_content]
enabled = false
action = "block"
"#,
    )
    .expect("family config should parse");

    assert_eq!(
        config.family.evaluate_text("show me a nude photo"),
        FamilyPolicyOutcome::Allow {
            warnings: Vec::new()
        }
    );
}

#[test]
#[cfg(feature = "family")]
fn validates_family_replacement_text() {
    let config = parse_config_text(
        r#"
[family]
enabled = true

[family.categories.direct_homework_answer]
enabled = true
action = "replace"
replacement = ""
"#,
    )
    .expect("family config should parse");

    let error = config
        .validate()
        .expect_err("empty replacement should fail validation");
    assert_eq!(error.field(), "family.categories.replacement");
}

#[cfg(feature = "guard")]
fn assert_parsed_model_aliases(config: &AppConfig) {
    assert_eq!(config.model_aliases.len(), 2);
    assert_eq!(config.model_aliases[0].id, "gpt-default");
    assert_eq!(config.model_aliases[0].kind, AliasKind::Upstream);
    assert_eq!(
        config.model_aliases[0].upstream_profile.as_deref(),
        Some("default")
    );
    assert_eq!(config.model_aliases[1].id, "family/child-safe-general-v1");
    assert_eq!(config.model_aliases[1].kind, AliasKind::Workflow);
    assert_eq!(
        config.model_aliases[1].workflow_id.as_deref(),
        Some("family.child_safe_general.v1")
    );
    assert_eq!(config.model_aliases[1].workflow_timeout_ms, Some(120_000));
    let workflow = config
        .workflows
        .get("family.child_safe_general.v1")
        .expect("workflow config should parse");
    assert_eq!(workflow.runtime_kind, WorkflowRuntime::Stdio);
    assert_eq!(workflow.command, "python");
    assert_eq!(
        workflow.args,
        vec![String::from("workflows/content_review.py")]
    );
    assert_eq!(workflow.timeout_ms, 120_000);
    assert_eq!(workflow.max_stdout_bytes, 1_048_576);
}

#[test]
fn validates_listener_allowed_upstreams_and_socket_uniqueness() {
    let config = parse_config_text(
        r#"
[[upstreams]]
name = "embedding"
base_url = "http://embedding.example/v1"
match_models = ["embedding-model"]

[[listeners]]
name = "bad"
port = 18002
allowed_upstreams = ["missing"]
"#,
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("unknown listener upstream should fail");
    assert_eq!(error.field(), "listeners.allowed_upstreams");

    let config = parse_config_text(
        r#"
[[upstreams]]
name = "embedding"
base_url = "http://embedding.example/v1"
match_models = ["embedding-model"]

[[listeners]]
name = "duplicate-socket"
bind_host = "127.0.0.1"
port = 18009
allowed_upstreams = ["embedding"]
"#,
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("duplicate listener socket should fail");
    assert_eq!(error.field(), "listeners.port");

    let config = parse_config_text(
        r#"
[[listeners]]
name = "wildcard"
bind_host = "0.0.0.0"
port = 18009
"#,
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("wildcard and default listener on the same port should fail");
    assert_eq!(error.field(), "listeners.port");

    let config = parse_config_text(
        r#"
[server]
port = 18010

[[listeners]]
name = "loopback-one"
bind_host = "127.0.0.1"
port = 18009

[[listeners]]
name = "loopback-two"
bind_host = "127.0.0.2"
port = 18009
"#,
    )
    .expect("config syntax should parse");
    let error = config
        .validate()
        .expect_err("fail-closed same-port specific listener bindings should fail");
    assert_eq!(error.field(), "listeners.port");

    let config = parse_config_text(
        r#"
[[listeners]]
name = "different-port"
bind_host = "0.0.0.0"
port = 18010
"#,
    )
    .expect("config syntax should parse");
    config
        .validate()
        .expect("different listener ports should validate");
}

#[test]
fn effective_listener_addresses_include_default_and_extra_sockets_in_order() {
    let config = parse_config_text(
        r#"
[server]
bind_host = "0.0.0.0"
port = 18009

[[listeners]]
name = "embedding"
bind_host = "127.0.0.1"
port = 18002

[[listeners]]
name = "aggregate"
bind_host = "::1"
port = 18005
"#,
    )
    .expect("config syntax should parse");
    config.validate().expect("listener sockets should validate");

    assert_eq!(
        config.effective_listener_addresses(),
        vec![
            String::from("0.0.0.0:18009"),
            String::from("127.0.0.1:18002"),
            String::from("[::1]:18005"),
        ]
    );
    assert_eq!(config.default_listener().bind_address(), "0.0.0.0:18009");
}

#[test]
fn parses_loop_guard_semantic_overrides() {
    let config = parse_config_text(
        r"
[loop_guard]
reasoning_semantic_detection_enabled = false
reasoning_semantic_similarity_threshold_percent = 70
reasoning_semantic_window_token_count = 32
reasoning_semantic_minimum_token_count = 16
reasoning_semantic_history_window_count = 20
",
    )
    .expect("semantic loop guard config should parse");

    assert!(!config.loop_guard.reasoning_semantic_detection_enabled);
    assert_eq!(
        config
            .loop_guard
            .reasoning_semantic_similarity_threshold_percent,
        70
    );
    assert_eq!(config.loop_guard.reasoning_semantic_window_token_count, 32);
    assert_eq!(config.loop_guard.reasoning_semantic_minimum_token_count, 16);
    assert_eq!(
        config.loop_guard.reasoning_semantic_history_window_count,
        20
    );
}

#[test]
fn parses_loop_guard_modes_and_legacy_enabled_switch() {
    for (mode, expected) in [
        ("disabled", LoopGuardMode::Disabled),
        ("monitor", LoopGuardMode::Monitor),
        ("enforce", LoopGuardMode::Enforce),
    ] {
        let config = parse_config_text(&format!(
            r#"
[loop_guard]
mode = "{mode}"
"#
        ))
        .expect("loop guard mode should parse");

        assert_eq!(config.loop_guard.mode, expected);
        assert_eq!(config.loop_guard.effective_mode(), expected);
    }

    let config = parse_config_text(
        r#"
[loop_guard]
enabled = false
mode = "enforce"
"#,
    )
    .expect("legacy loop guard switch should parse");

    assert_eq!(config.loop_guard.mode, LoopGuardMode::Enforce);
    assert_eq!(config.loop_guard.effective_mode(), LoopGuardMode::Disabled);
}

#[test]
fn rejects_invalid_loop_guard_mode() {
    let error = parse_config_text(
        r#"
[loop_guard]
mode = "observe-everything"
"#,
    )
    .expect_err("invalid detector mode should fail");

    assert_eq!(error.line(), 3);
    assert!(error.message().contains("invalid loop_guard.mode"));
    assert!(error.message().contains("disabled"));
    assert!(error.message().contains("monitor"));
    assert!(error.message().contains("enforce"));
}

#[cfg(feature = "guard")]
fn assert_parsed_upstream_stall_overrides(config: &AppConfig) {
    assert!(config.upstream_stall.enabled);
    assert_eq!(config.upstream_stall.idle_timeout_ms, 5_000);
    assert_eq!(
        config.upstream_stall.recovery_command,
        vec![
            String::from("/usr/bin/systemctl"),
            String::from("--user"),
            String::from("restart"),
            String::from("vllm-aeon-27b-dflash-n12.service"),
        ]
    );
    assert_eq!(config.upstream_stall.recovery_timeout_ms, 60_000);
    assert_eq!(config.upstream_stall.recovery_cooldown_ms, 45_000);
    assert_eq!(config.upstream_stall.recovery_budget_window_ms, 180_000);
    assert_eq!(config.upstream_stall.recovery_max_per_window, 1);
}

#[cfg(feature = "guard")]
fn assert_parsed_retry_overrides(config: &AppConfig) {
    assert_eq!(config.retry.max_attempts, 3);
    assert!(!config.retry.anti_loop_hint_enabled);
    assert!(config.retry.shielded_streaming_enabled);
    assert_eq!(
        config.retry.downstream_drop_policy,
        DownstreamDropPolicy::Detach
    );
}

#[cfg(feature = "guard")]
fn assert_parsed_observability_overrides(config: &AppConfig) {
    assert!(!config.observability.metrics_enabled.is_enabled());
    assert!(
        !config
            .observability
            .health_upstream_probe_enabled
            .is_enabled()
    );
    assert_eq!(config.observability.health_upstream_probe_timeout_ms, 250);
    assert!(config.observability.debug_summary_enabled.is_enabled());
    assert_eq!(
        config.observability.debug_summary_admin_token.as_deref(),
        Some("test-admin-token")
    );
    assert_eq!(config.observability.debug_summary_max_records, 7);
    assert_eq!(config.observability.retention.max_records, 50);
    assert_eq!(config.observability.retention.prune_to_records, Some(40));
    assert_eq!(
        config.observability.retention.effective_prune_to_records(),
        40
    );
}

#[test]
fn derives_retention_record_hysteresis_from_overridden_max_records() {
    let config = parse_config_text(
        r"
[observability.retention]
max_records = 10
",
    )
    .expect("config should parse");

    assert_eq!(config.observability.retention.max_records, 10);
    assert_eq!(config.observability.retention.prune_to_records, None);
    assert_eq!(
        config.observability.retention.effective_prune_to_records(),
        8
    );
}

#[test]
fn validates_evidence_paths_and_retention() {
    let mut config = AppConfig::default();
    config.evidence.sqlite_path = PathBuf::new();

    let error = config
        .validate()
        .expect_err("empty evidence sqlite path should fail");
    assert_eq!(error.field(), "evidence.sqlite_path");

    config.evidence.sqlite_path = PathBuf::from("evidence.sqlite3");
    let error = config
        .validate()
        .expect_err("bare evidence sqlite path should fail");
    assert_eq!(error.field(), "evidence.sqlite_path");

    config.evidence.sqlite_path = PathBuf::from("storage/evidence.sqlite3");
    config.evidence.blob_cache_dir = PathBuf::new();
    let error = config
        .validate()
        .expect_err("empty evidence blob cache path should fail");
    assert_eq!(error.field(), "evidence.blob_cache_dir");

    config.evidence.blob_cache_dir = PathBuf::from("blobs");
    let error = config
        .validate()
        .expect_err("bare evidence blob cache path should fail");
    assert_eq!(error.field(), "evidence.blob_cache_dir");

    config.evidence.blob_cache_dir = PathBuf::from("cache/blobs");
    config.evidence.max_bytes = 10;
    config.evidence.prune_to_bytes = 11;
    let error = config
        .validate()
        .expect_err("evidence byte hysteresis should fail");
    assert_eq!(error.field(), "evidence.prune_to_bytes");

    config.evidence.prune_to_bytes = 10;
    config.evidence.max_records = 10;
    config.evidence.prune_to_records = Some(11);
    let error = config
        .validate()
        .expect_err("evidence record hysteresis should fail");
    assert_eq!(error.field(), "evidence.prune_to_records");
}

#[cfg(unix)]
#[test]
fn validates_evidence_rejects_unsafe_parent_permissions_when_disabled() {
    let root = unique_test_path("unsafe-evidence-parent");
    fs::create_dir_all(&root).expect("test evidence parent should be created");
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755))
        .expect("test evidence parent permissions should be configurable");

    let mut config = AppConfig::default();
    config.evidence.enabled = false;
    config.evidence.sqlite_path = root.join("evidence.sqlite3");
    config.evidence.blob_cache_dir = root.join("blobs");

    let error = config
        .validate()
        .expect_err("disabled evidence should still reject unsafe parent permissions");

    assert_eq!(error.field(), "evidence.sqlite_path");
    assert!(error.message().contains("group or other users"));

    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
        .expect("test evidence parent permissions should be restorable");
    remove_dir_all(&root);
}

#[cfg(unix)]
#[test]
fn validates_evidence_rejects_symlink_path_components_when_disabled() {
    let root = unique_test_path("symlink-evidence-path");
    let real = root.join("real");
    let link = root.join("link");
    fs::create_dir_all(&real).expect("real evidence directory should be created");
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
        .expect("root permissions should be owner-private");
    fs::set_permissions(&real, fs::Permissions::from_mode(0o700))
        .expect("real evidence permissions should be owner-private");
    symlink(&real, &link).expect("test symlink should be created");

    let mut config = AppConfig::default();
    config.evidence.enabled = false;
    config.evidence.sqlite_path = link.join("evidence.sqlite3");
    config.evidence.blob_cache_dir = real.join("blobs");
    let error = config
        .validate()
        .expect_err("disabled evidence should still reject sqlite symlink components");
    assert_eq!(error.field(), "evidence.sqlite_path");
    assert!(error.message().contains("symlink"));

    config.evidence.sqlite_path = real.join("evidence.sqlite3");
    config.evidence.blob_cache_dir = link.join("blobs");
    let error = config
        .validate()
        .expect_err("disabled evidence should still reject blob cache symlink components");
    assert_eq!(error.field(), "evidence.blob_cache_dir");
    assert!(error.message().contains("symlink"));

    remove_dir_all(&root);
}

#[test]
fn validates_retry_attempt_bounds() {
    let mut config = AppConfig::default();
    config.retry.max_attempts = 11;

    let error = config
        .validate()
        .expect_err("retry attempts should be bounded");

    assert_eq!(error.field(), "retry.max_attempts");
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

    config.observability.retention.prune_to_bytes = 10;
    config.observability.retention.max_records = 10;
    config.observability.retention.prune_to_records = Some(11);
    let error = config
        .validate()
        .expect_err("record retention relation should fail");
    assert_eq!(error.field(), "observability.retention.prune_to_records");
}

#[test]
fn validates_operational_endpoint_bounds() {
    let mut config = AppConfig::default();
    config.observability.debug_summary_max_records = 101;

    let error = config
        .validate()
        .expect_err("debug summary limit should be bounded");
    assert_eq!(error.field(), "observability.debug_summary_max_records");

    config.observability.debug_summary_max_records = 20;
    config.observability.health_upstream_probe_timeout_ms = 0;
    let error = config
        .validate()
        .expect_err("health probe timeout should be nonzero");
    assert_eq!(
        error.field(),
        "observability.health_upstream_probe_timeout_ms"
    );
}

#[test]
fn empty_debug_summary_admin_token_disables_token_requirement() {
    let config = parse_config_text(
        r#"
[observability]
debug_summary_admin_token = ""
"#,
    )
    .expect("empty optional token config should parse");

    assert_eq!(config.observability.debug_summary_admin_token, None);
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
fn validates_server_admission_queue_bounds() {
    let mut config = AppConfig::default();
    config.server.max_queued_generation_requests = 10_001;

    let error = config
        .validate()
        .expect_err("excessive generation queue limit should fail");

    assert_eq!(error.field(), "server.max_queued_generation_requests");

    config = AppConfig::default();
    config.server.generation_queue_timeout_ms = 0;

    let error = config
        .validate()
        .expect_err("zero generation queue timeout should fail");

    assert_eq!(error.field(), "server.generation_queue_timeout_ms");

    config = AppConfig::default();
    config.server.generation_queue_full_status = 200;

    let error = config
        .validate()
        .expect_err("non-error generation queue-full status should fail");

    assert_eq!(error.field(), "server.generation_queue_full_status");
}

#[test]
fn rejects_invalid_generation_queue_full_status_config() {
    for status in [200_u16, 399, 600] {
        let error = parse_config_text(&format!(
            r"
[server]
generation_queue_full_status = {status}
"
        ))
        .expect_err("invalid queue-full status should fail during parsing");

        assert!(
            error
                .to_string()
                .contains("server.generation_queue_full_status")
        );
    }
}

#[test]
fn validates_control_plane_in_flight_limit() {
    let mut config = AppConfig::default();
    config.server.max_control_plane_in_flight_requests = 0;

    let error = config
        .validate()
        .expect_err("zero control-plane request limit should fail");

    assert_eq!(error.field(), "server.max_control_plane_in_flight_requests");

    config.server.max_control_plane_in_flight_requests = 1_025;
    let error = config
        .validate()
        .expect_err("excessive control-plane request limit should fail");

    assert_eq!(error.field(), "server.max_control_plane_in_flight_requests");
}

#[test]
fn validates_request_body_limit_bounds() {
    let mut config = AppConfig::default();
    config.server.max_request_body_bytes = 0;

    let error = config
        .validate()
        .expect_err("zero request body limit should fail");
    assert_eq!(error.field(), "server.max_request_body_bytes");

    config.server.max_request_body_bytes = 1_073_741_825;
    let error = config
        .validate()
        .expect_err("excessive request body limit should fail");
    assert_eq!(error.field(), "server.max_request_body_bytes");
}

#[test]
fn validates_upstream_request_timeout_bounds() {
    let mut config = AppConfig::default();
    config.upstream.request_timeout_ms = 0;

    let error = config
        .validate()
        .expect_err("zero upstream timeout should fail");
    assert_eq!(error.field(), "upstream.request_timeout_ms");
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
fn validates_loop_guard_semantic_bounds() {
    let mut config = AppConfig::default();
    config
        .loop_guard
        .reasoning_semantic_similarity_threshold_percent = 0;

    let error = config
        .validate()
        .expect_err("zero semantic similarity threshold should fail");
    assert_eq!(
        error.field(),
        "loop_guard.reasoning_semantic_similarity_threshold_percent"
    );

    config = AppConfig::default();
    config.loop_guard.reasoning_semantic_window_token_count = 257;
    let error = config
        .validate()
        .expect_err("semantic window token count should be capped");
    assert_eq!(
        error.field(),
        "loop_guard.reasoning_semantic_window_token_count"
    );

    config = AppConfig::default();
    config.loop_guard.reasoning_semantic_minimum_token_count = 25;
    config.loop_guard.reasoning_semantic_window_token_count = 24;
    let error = config
        .validate()
        .expect_err("semantic minimum token count should fit in window");
    assert_eq!(
        error.field(),
        "loop_guard.reasoning_semantic_minimum_token_count"
    );

    config = AppConfig::default();
    config.loop_guard.reasoning_semantic_history_window_count = 257;
    let error = config
        .validate()
        .expect_err("semantic history window count should be capped");
    assert_eq!(
        error.field(),
        "loop_guard.reasoning_semantic_history_window_count"
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
max_in_flight_requests = 4
max_queued_generation_requests = 8
generation_queue_timeout_ms = 2000
generation_queue_full_status = 503
max_control_plane_in_flight_requests = 3
max_request_body_bytes = 1048576

[heartbeat]
mode = "sse"
interval_secs = 15

	[loop_guard]
	mode = "enforce"
	output_repeated_line_threshold = 24
reasoning_semantic_detection_enabled = true
reasoning_semantic_similarity_threshold_percent = 55
reasoning_semantic_window_token_count = 24
reasoning_semantic_minimum_token_count = 12
reasoning_semantic_history_window_count = 16
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[server]
port = 19000
max_in_flight_requests = 2
max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000
generation_queue_full_status = 429
generation_queue_retry_after_secs = 30
max_control_plane_in_flight_requests = 2
max_request_body_bytes = 512

[thinking]
force_disable = true
no_thinking_marker_policy = "respect_no_thinking_markers"
default_injection_schema = "chat_template_kwargs"

[upstream]
request_timeout_ms = 90000

[heartbeat]
mode = "disabled"
interval_secs = 3

	[loop_guard]
	mode = "monitor"
	output_repeated_line_threshold = 7
reasoning_semantic_detection_enabled = false
reasoning_semantic_similarity_threshold_percent = 70
reasoning_semantic_window_token_count = 32
reasoning_semantic_minimum_token_count = 16
reasoning_semantic_history_window_count = 20
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
    assert_eq!(snapshot.server.max_in_flight_requests, 2);
    assert_eq!(snapshot.server.max_queued_generation_requests, 1);
    assert_eq!(snapshot.server.generation_queue_timeout_ms, 1_000);
    assert_eq!(snapshot.server.generation_queue_full_status, 429);
    assert_eq!(snapshot.server.generation_queue_retry_after_secs, Some(30));
    assert_eq!(snapshot.server.max_control_plane_in_flight_requests, 2);
    assert_eq!(snapshot.server.max_request_body_bytes, 512);
    assert!(snapshot.thinking.force_disable);
    assert_eq!(
        snapshot.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::RespectNoThinkingMarkers
    );
    assert_eq!(
        snapshot.thinking.default_injection_schema,
        DefaultInjectionSchema::ChatTemplateKwargs
    );
    assert_eq!(snapshot.upstream.request_timeout_ms, 90_000);
    assert_eq!(snapshot.heartbeat.mode, HeartbeatMode::Disabled);
    assert_eq!(snapshot.heartbeat.interval_secs, 3);
    assert_reloaded_loop_guard_fields(&snapshot);

    remove_file(&path);
}

fn assert_reloaded_loop_guard_fields(snapshot: &AppConfig) {
    assert_eq!(snapshot.loop_guard.mode, LoopGuardMode::Monitor);
    assert_eq!(snapshot.loop_guard.output_repeated_line_threshold, 7);
    assert!(!snapshot.loop_guard.reasoning_semantic_detection_enabled);
    assert_eq!(
        snapshot
            .loop_guard
            .reasoning_semantic_similarity_threshold_percent,
        70
    );
    assert_eq!(
        snapshot.loop_guard.reasoning_semantic_window_token_count,
        32
    );
    assert_eq!(
        snapshot.loop_guard.reasoning_semantic_minimum_token_count,
        16
    );
    assert_eq!(
        snapshot.loop_guard.reasoning_semantic_history_window_count,
        20
    );
}

#[test]
fn reload_applies_safe_named_profile_fields_without_changing_topology() {
    let path = unique_test_path("profile-reload.toml");
    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]
request_timeout_ms = 120000
max_in_flight_requests = 4
max_queued_generation_requests = 4

[upstreams.metadata]
context_length_override = 4096
input_token_safety_margin = 64

[upstreams.thinking]
mode = "bounded_thinking"
thinking_token_budget = 1024
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]
request_timeout_ms = 90000
max_in_flight_requests = 8
max_queued_generation_requests = 16

[upstreams.metadata]
context_length_override = 8192
input_token_safety_margin = 128

[upstreams.thinking]
mode = "force_thinking"
thinking_token_budget = 2048
no_thinking_marker_policy = "escape_hatch_only"
default_injection_schema = "chat_template_kwargs"
"#,
    );
    let outcome = manager.reload().expect("profile reload should succeed");
    let snapshot = manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed");
    let profile = snapshot.select_upstream_profile(Some("aeon-ultimate"));

    assert!(outcome.applied);
    assert!(outcome.restart_required_changes.is_empty());
    assert_eq!(profile.profile.request_timeout_ms, 90_000);
    assert_eq!(profile.profile.max_in_flight_requests, Some(8));
    assert_eq!(profile.profile.max_queued_generation_requests, Some(16));
    assert_eq!(
        profile.profile.metadata.context_length_override,
        Some(8_192)
    );
    assert_eq!(profile.profile.metadata.input_token_safety_margin, 128);
    assert_eq!(profile.profile.thinking.mode, ThinkingMode::ForceThinking);
    assert_eq!(profile.profile.thinking.budget_tokens, 2_048);
    assert_eq!(
        profile.profile.thinking.no_thinking_marker_policy,
        NoThinkingMarkerPolicy::EscapeHatchOnly
    );
    assert_eq!(
        profile.profile.thinking.default_injection_schema,
        DefaultInjectionSchema::ChatTemplateKwargs
    );

    remove_file(&path);
}

#[test]
fn reload_reports_named_profile_topology_changes_without_half_updating_routes() {
    let path = unique_test_path("profile-topology-reload.toml");
    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["aeon-ultimate"]
request_timeout_ms = 120000

[upstreams.thinking]
thinking_token_budget = 1024
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://new-aeon.example/v1"
match_models = ["renamed-model"]
request_timeout_ms = 90000

[upstreams.thinking]
thinking_token_budget = 2048
"#,
    );
    let outcome = manager
        .reload()
        .expect("topology reload should report restart requirement");
    let snapshot = manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed");
    let old_route = snapshot.select_upstream_profile(Some("aeon-ultimate"));
    let renamed_route = snapshot.select_upstream_profile(Some("renamed-model"));

    assert!(!outcome.applied);
    assert_eq!(outcome.restart_required_changes.len(), 1);
    assert_eq!(
        outcome.restart_required_changes[0].field,
        "upstreams.topology"
    );
    assert_eq!(old_route.profile.name, "aeon-chat");
    assert_eq!(old_route.profile.base_url, "http://aeon.example/v1");
    assert_eq!(old_route.profile.thinking.budget_tokens, 1_024);
    assert_eq!(renamed_route.profile.name, "default");

    remove_file(&path);
}

#[test]
fn reload_reports_listener_topology_changes_without_half_updating_listeners() {
    let path = unique_test_path("listener-topology-reload.toml");
    write_config(
        &path,
        r#"
[[upstreams]]
name = "embedding"
base_url = "http://embedding.example/v1"
match_models = ["embedding-model"]

[[listeners]]
name = "embedding-legacy"
port = 18002
allowed_upstreams = ["embedding"]
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[[upstreams]]
name = "embedding"
base_url = "http://embedding.example/v1"
match_models = ["embedding-model"]

[[listeners]]
name = "embedding-legacy"
port = 18003
allowed_upstreams = ["embedding"]
"#,
    );
    let outcome = manager
        .reload()
        .expect("listener topology reload should report restart requirement");
    let snapshot = manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed");

    assert!(!outcome.applied);
    assert_eq!(outcome.restart_required_changes.len(), 1);
    assert_eq!(
        outcome.restart_required_changes[0].field,
        "listeners.topology"
    );
    assert_eq!(snapshot.listeners[0].port, 18_002);

    remove_file(&path);
}

#[test]
fn reload_reports_match_model_topology_changes_with_delimiter_like_aliases() {
    let path = unique_test_path("profile-topology-delimiter-reload.toml");
    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["a,b"]

[upstreams.thinking]
thinking_token_budget = 1024
"#,
    );
    let manager = ConfigManager::from_explicit_path(&path).expect("initial config should load");

    write_config(
        &path,
        r#"
[[upstreams]]
name = "aeon-chat"
base_url = "http://aeon.example/v1"
match_models = ["a", "b"]

[upstreams.thinking]
thinking_token_budget = 1024
"#,
    );
    let outcome = manager
        .reload()
        .expect("topology reload should report restart requirement");
    let snapshot = manager
        .handle()
        .snapshot()
        .expect("snapshot should succeed");
    let old_route = snapshot.select_upstream_profile(Some("a,b"));
    let split_route = snapshot.select_upstream_profile(Some("a"));

    assert!(!outcome.applied);
    assert_eq!(outcome.restart_required_changes.len(), 1);
    assert_eq!(
        outcome.restart_required_changes[0].field,
        "upstreams.topology"
    );
    assert_eq!(old_route.profile.name, "aeon-chat");
    assert_eq!(split_route.profile.name, "default");

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
max_in_flight_requests = 4
max_queued_generation_requests = 8
generation_queue_timeout_ms = 2000
generation_queue_full_status = 503
max_control_plane_in_flight_requests = 3
max_request_body_bytes = 1048576

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
max_in_flight_requests = 2
max_queued_generation_requests = 1
generation_queue_timeout_ms = 1000
generation_queue_full_status = 429
generation_queue_retry_after_secs = 30
max_control_plane_in_flight_requests = 2
max_request_body_bytes = 512

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
    assert_eq!(snapshot.server.max_in_flight_requests, 2);
    assert_eq!(snapshot.server.max_queued_generation_requests, 1);
    assert_eq!(snapshot.server.generation_queue_timeout_ms, 1_000);
    assert_eq!(snapshot.server.generation_queue_full_status, 429);
    assert_eq!(snapshot.server.generation_queue_retry_after_secs, Some(30));
    assert_eq!(snapshot.server.max_control_plane_in_flight_requests, 2);
    assert_eq!(snapshot.server.max_request_body_bytes, 512);
    assert_eq!(snapshot.heartbeat.mode, HeartbeatMode::Disabled);
    assert_eq!(snapshot.heartbeat.interval_secs, 4);

    watcher.stop().expect("watcher should stop cleanly");
    remove_file(&path);
}

#[test]
fn reload_metadata_lists_cover_expected_fields() {
    assert!(RELOADABLE_FIELDS.contains(&"thinking.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.force_disable"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.mode"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.max_tokens"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.tool_request_policy"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.no_thinking_marker_policy"));
    assert!(RELOADABLE_FIELDS.contains(&"thinking.default_injection_schema"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.thinking.mode"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.thinking.no_thinking_marker_policy"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.thinking.default_injection_schema"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.metadata.input_token_safety_margin"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.hot_restart.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.hot_restart.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"upstreams.hot_restart.probe_messages"));
    assert!(RELOADABLE_FIELDS.contains(&"server.max_in_flight_requests"));
    assert!(RELOADABLE_FIELDS.contains(&"server.max_queued_generation_requests"));
    assert!(RELOADABLE_FIELDS.contains(&"server.generation_queue_timeout_ms"));
    assert!(RELOADABLE_FIELDS.contains(&"server.generation_queue_full_status"));
    assert!(RELOADABLE_FIELDS.contains(&"server.generation_queue_retry_after_secs"));
    assert!(RELOADABLE_FIELDS.contains(&"server.max_control_plane_in_flight_requests"));
    assert!(RELOADABLE_FIELDS.contains(&"server.max_request_body_bytes"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.mode"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.output_repeated_line_threshold"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.input_overlap_threshold_multiplier"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.reasoning_semantic_detection_enabled"));
    assert!(
        RELOADABLE_FIELDS.contains(&"loop_guard.reasoning_semantic_similarity_threshold_percent")
    );
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.reasoning_semantic_window_token_count"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.reasoning_semantic_minimum_token_count"));
    assert!(RELOADABLE_FIELDS.contains(&"loop_guard.reasoning_semantic_history_window_count"));
    assert!(RELOADABLE_FIELDS.contains(&"retry.anti_loop_hint_enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"retry.shielded_streaming_enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"retry.downstream_drop_policy"));
    assert!(RELOADABLE_FIELDS.contains(&"retry.ladder"));
    #[cfg(feature = "guard")]
    assert!(RELOADABLE_FIELDS.contains(&"profiles"));
    #[cfg(feature = "guard")]
    assert!(RELOADABLE_FIELDS.contains(&"virtual_keys"));
    #[cfg(feature = "guard")]
    assert!(RELOADABLE_FIELDS.contains(&"budget.enabled"));
    #[cfg(feature = "guard")]
    assert!(RELOADABLE_FIELDS.contains(&"budget.reset_timezone"));
    #[cfg(feature = "guard")]
    assert!(RELOADABLE_FIELDS.contains(&"budget.reset_hour_utc"));
    assert!(RELOADABLE_FIELDS.contains(&"observability.metrics_enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"observability.debug_summary_enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"observability.debug_summary_admin_token"));
    assert!(RELOADABLE_FIELDS.contains(&"observability.retention.prune_to_records"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.include_raw_payloads"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.include_request_headers"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.prune_to_records"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.shadow.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.shadow.keep_looping_attempt_running"));
    assert!(RELOADABLE_FIELDS.contains(&"evidence.shadow.max_global_shadow_in_flight"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.recovery_command"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.recovery_timeout_ms"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.recovery_cooldown_ms"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.recovery_budget_window_ms"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.stall.recovery_max_per_window"));
    assert!(RELOADABLE_FIELDS.contains(&"cloudflare.enabled"));
    assert!(RELOADABLE_FIELDS.contains(&"upstream.request_timeout_ms"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.max_in_flight_requests"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.max_queued_generation_requests"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.generation_queue_timeout_ms"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.generation_queue_full_status"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.generation_queue_retry_after_secs"));
    assert!(!RESTART_REQUIRED_FIELDS.contains(&"server.max_control_plane_in_flight_requests"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"upstream.base_url"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"upstreams.topology"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"listeners.topology"));
    #[cfg(feature = "guard")]
    assert!(RESTART_REQUIRED_FIELDS.contains(&"model_aliases.topology"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"observability.sqlite_path"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"evidence.sqlite_path"));
    assert!(RESTART_REQUIRED_FIELDS.contains(&"evidence.blob_cache_dir"));
    #[cfg(feature = "guard")]
    assert!(RESTART_REQUIRED_FIELDS.contains(&"budget.sqlite_path"));
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

fn remove_dir_all(path: &Path) {
    if let Err(error) = fs::remove_dir_all(path) {
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

fn _assert_error_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ConfigParseError>();
    assert_send_sync::<ValidationError>();
}
