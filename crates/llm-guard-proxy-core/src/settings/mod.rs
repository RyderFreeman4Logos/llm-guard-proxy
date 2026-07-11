//! Typed configuration loading and hot reload support.
//!
//! This module keeps configuration policy in the headless core so the binary,
//! future proxy server, and tests all consume the same validated model.

mod error;
mod model;
mod parse;
mod reload;

#[cfg(test)]
mod tests;

pub use error::{ConfigError, ConfigParseError, ValidationError};
#[cfg(feature = "param-override")]
pub use model::ParamOverrideConfig;
pub use model::{
    AppConfig, CloudflareConfig, ConfigToggle, DefaultInjectionSchema, DownstreamDropPolicy,
    EmbeddingProvider, EmbeddingQueuePolicy, EvidenceConfig, EvidencePairedComparisonConfig,
    EvidenceShadowConfig, HeartbeatConfig, HeartbeatMode, HotRestartConfig, ListenerConfig,
    LocalRecoveryConfig, LoopFailurePolicy, LoopGuardConfig, LoopGuardEmbeddingConfig,
    LoopGuardMode, MetadataConfig, NoThinkingMarkerPolicy, ObservabilityConfig,
    RestartRequiredChange, RetentionConfig, RetryConfig, RetryLadderConfig,
    SelectedUpstreamProfile, ServerConfig, ShadowComparisonAttempt, ShieldingConfig,
    ThinkingConfig, ThinkingMode, ToolRequestThinkingPolicy, UpstreamConfig, UpstreamProfileConfig,
    UpstreamRouteReason, UpstreamStallConfig, redact_upstream_base_url, validate_upstream_base_url,
};
#[cfg(feature = "guard")]
pub use model::{BudgetConfig, GuardWorkflowConfig};
#[cfg(feature = "guard")]
pub use model::{UnknownKeyPolicy, VirtualKeyConfig};
pub use reload::{
    ConfigHandle, ConfigManager, MissingConfigPolicy, ReloadOutcome, ReloadWatcher,
    default_config_path,
};

/// Default config location relative to the user's home directory.
pub const DEFAULT_CONFIG_RELATIVE_PATH: &str = ".config/llm-guard-proxy/config.toml";

/// Fields that can be changed by reloading the config file.
pub const RELOADABLE_FIELDS: &[&str] = &[
    "server.max_in_flight_requests",
    "server.max_queued_generation_requests",
    "server.generation_queue_timeout_ms",
    "server.generation_queue_full_status",
    "server.generation_queue_retry_after_secs",
    "server.max_control_plane_in_flight_requests",
    "server.max_request_body_bytes",
    "server.shutdown_drain_timeout_ms",
    "shielding.enabled",
    "observability.enabled",
    "observability.capture_raw_payloads",
    "observability.metrics_enabled",
    "observability.health_upstream_probe_enabled",
    "observability.health_upstream_probe_timeout_ms",
    "observability.debug_summary_enabled",
    "observability.debug_summary_admin_token",
    "observability.debug_summary_max_records",
    "observability.retention.max_bytes",
    "observability.retention.prune_to_bytes",
    "observability.retention.max_records",
    "observability.retention.prune_to_records",
    "evidence.enabled",
    "evidence.include_raw_payloads",
    "evidence.include_request_headers",
    "evidence.max_bytes",
    "evidence.prune_to_bytes",
    "evidence.max_records",
    "evidence.prune_to_records",
    "evidence.shadow.enabled",
    "evidence.shadow.keep_looping_attempt_running",
    "evidence.shadow.parallel_downgrade_attempts",
    "evidence.shadow.max_shadow_attempts_per_request",
    "evidence.shadow.max_global_shadow_in_flight",
    "evidence.shadow.shadow_attempt_timeout_ms",
    "evidence.shadow.compare_attempts",
    "evidence.shadow.paired_comparison.enabled",
    "evidence.shadow.paired_comparison.variants",
    "evidence.shadow.paired_comparison.include_raw_input",
    "evidence.shadow.paired_comparison.include_raw_output",
    "evidence.shadow.paired_comparison.include_raw_reasoning",
    "evidence.shadow.paired_comparison.sample_rate",
    "evidence.shadow.paired_comparison.max_raw_input_bytes",
    "evidence.shadow.paired_comparison.max_raw_output_bytes",
    "evidence.shadow.paired_comparison.max_raw_reasoning_bytes",
    "evidence.shadow.paired_comparison.max_retention_records",
    "evidence.shadow.paired_comparison.max_retention_bytes",
    "evidence.shadow.paired_comparison.retention_days",
    "thinking.enabled",
    "thinking.force_disable",
    "thinking.mode",
    "thinking.max_tokens",
    "thinking.budget_tokens",
    "thinking.thinking_token_budget",
    "thinking.budget_accounting",
    "thinking.preserve_answer_budget",
    "thinking.tool_request_policy",
    "thinking.no_thinking_marker_policy",
    "thinking.default_injection_schema",
    "thinking.apply_to_tool_requests",
    "loop_guard.enabled",
    "loop_guard.mode",
    "loop_guard.on_reasoning_loop",
    "loop_guard.normalized_input_window_secs",
    "loop_guard.max_repeated_inputs",
    "loop_guard.output_repeated_line_threshold",
    "loop_guard.output_token_window_size",
    "loop_guard.output_repeated_token_window_threshold",
    "loop_guard.output_suffix_cycle_threshold",
    "loop_guard.output_low_progress_min_bytes",
    "loop_guard.output_low_progress_unique_ratio_percent",
    "loop_guard.input_overlap_threshold_multiplier",
    "loop_guard.reasoning_semantic_detection_enabled",
    "loop_guard.reasoning_semantic_similarity_threshold_percent",
    "loop_guard.reasoning_semantic_window_token_count",
    "loop_guard.reasoning_semantic_minimum_token_count",
    "loop_guard.reasoning_semantic_history_window_count",
    "loop_guard.embedding.provider",
    "loop_guard.embedding.endpoint",
    "loop_guard.embedding.model",
    "loop_guard.embedding.api_key",
    "loop_guard.embedding.window_token_count",
    "loop_guard.embedding.window_stride_tokens",
    "loop_guard.embedding.minimum_token_count",
    "loop_guard.embedding.history_window_count",
    "loop_guard.embedding.batch_max_windows",
    "loop_guard.embedding.batch_max_wait_ms",
    "loop_guard.embedding.queue_max_windows",
    "loop_guard.embedding.on_queue_full",
    "loop_guard.embedding.vector_dim",
    "retry.enabled",
    "retry.max_attempts",
    "retry.request_deadline_ms",
    "retry.anti_loop_hint_enabled",
    "retry.shielded_streaming_enabled",
    "retry.downstream_drop_policy",
    "retry.ladder",
    "upstream.stall.enabled",
    "upstream.stall.first_chunk_timeout_ms",
    "upstream.stall.idle_timeout_ms",
    "upstream.stall.recovery_command",
    "upstream.stall.recovery_timeout_ms",
    "upstream.stall.recovery_cooldown_ms",
    "upstream.stall.recovery_budget_window_ms",
    "upstream.stall.recovery_max_per_window",
    "heartbeat.mode",
    "heartbeat.interval_secs",
    "cloudflare.enabled",
    "upstream.request_timeout_ms",
    "upstream.metadata.discovery_enabled",
    "upstream.metadata.enrich_responses",
    "upstream.metadata.refresh_interval_secs",
    "upstream.metadata.context_length_override",
    "upstream.metadata.max_model_len_override",
    "upstream.metadata.input_token_safety_margin",
    "upstream.hot_restart.enabled",
    "upstream.hot_restart.probe_max_tokens",
    "upstream.hot_restart.probe_interval_secs",
    "upstream.hot_restart.probe_timeout_secs",
    "upstream.hot_restart.probe_messages",
    "upstream.hot_restart.probe_chat_template_kwargs",
    "upstream.local_recovery.enabled",
    "upstream.local_recovery.restart_command",
    "upstream.local_recovery.restart_timeout_ms",
    "upstream.local_recovery.readiness_endpoint",
    "upstream.local_recovery.readiness_body",
    "upstream.local_recovery.readiness_request_timeout_ms",
    "upstream.local_recovery.readiness_deadline_ms",
    "upstream.local_recovery.readiness_interval_ms",
    "upstream.local_recovery.max_attempts_per_request",
    "upstream.local_recovery.cooldown_ms",
    "upstream.local_recovery.budget_window_ms",
    "upstream.local_recovery.max_per_window",
    "upstreams.request_timeout_ms",
    "upstreams.max_in_flight_requests",
    "upstreams.max_queued_generation_requests",
    "upstreams.metadata.discovery_enabled",
    "upstreams.metadata.enrich_responses",
    "upstreams.metadata.refresh_interval_secs",
    "upstreams.metadata.context_length_override",
    "upstreams.metadata.max_model_len_override",
    "upstreams.metadata.input_token_safety_margin",
    "upstreams.hot_restart.enabled",
    "upstreams.hot_restart.probe_max_tokens",
    "upstreams.hot_restart.probe_interval_secs",
    "upstreams.hot_restart.probe_timeout_secs",
    "upstreams.hot_restart.probe_messages",
    "upstreams.hot_restart.probe_chat_template_kwargs",
    "upstreams.local_recovery.enabled",
    "upstreams.local_recovery.restart_command",
    "upstreams.local_recovery.restart_timeout_ms",
    "upstreams.local_recovery.readiness_endpoint",
    "upstreams.local_recovery.readiness_body",
    "upstreams.local_recovery.readiness_request_timeout_ms",
    "upstreams.local_recovery.readiness_deadline_ms",
    "upstreams.local_recovery.readiness_interval_ms",
    "upstreams.local_recovery.max_attempts_per_request",
    "upstreams.local_recovery.cooldown_ms",
    "upstreams.local_recovery.budget_window_ms",
    "upstreams.local_recovery.max_per_window",
    "upstreams.thinking.enabled",
    "upstreams.thinking.force_disable",
    "upstreams.thinking.mode",
    "upstreams.thinking.max_tokens",
    "upstreams.thinking.budget_tokens",
    "upstreams.thinking.thinking_token_budget",
    "upstreams.thinking.budget_accounting",
    "upstreams.thinking.preserve_answer_budget",
    "upstreams.thinking.tool_request_policy",
    "upstreams.thinking.no_thinking_marker_policy",
    "upstreams.thinking.default_injection_schema",
    "upstreams.thinking.apply_to_tool_requests",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.enabled",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.temperature",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.top_p",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.top_k",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.max_tokens",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.frequency_penalty",
    #[cfg(feature = "param-override")]
    "upstreams.param_override.presence_penalty",
    #[cfg(feature = "guard")]
    "profiles",
    #[cfg(feature = "guard")]
    "virtual_keys",
    #[cfg(feature = "guard")]
    "budget.enabled",
    #[cfg(feature = "guard")]
    "budget.reset_timezone",
    #[cfg(feature = "guard")]
    "budget.reset_hour_utc",
    #[cfg(feature = "guard")]
    "model_aliases",
    #[cfg(feature = "guard")]
    "workflows",
    #[cfg(feature = "guard")]
    "guard_workflows.pre_request",
    #[cfg(feature = "guard")]
    "guard_workflows.post_response",
    #[cfg(feature = "family")]
    "family",
];

/// Fields read at process startup that require a restart when changed.
pub const RESTART_REQUIRED_FIELDS: &[&str] = &[
    "server.bind_host",
    "server.port",
    "listeners.topology",
    "upstream.base_url",
    "upstreams.topology",
    #[cfg(feature = "guard")]
    "model_aliases.topology",
    "observability.sqlite_path",
    "evidence.sqlite_path",
    "evidence.blob_cache_dir",
    #[cfg(feature = "guard")]
    "budget.sqlite_path",
];
