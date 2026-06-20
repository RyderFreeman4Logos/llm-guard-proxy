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
pub use model::{
    AppConfig, CloudflareConfig, HeartbeatConfig, HeartbeatMode, LoopGuardConfig, MetadataConfig,
    ObservabilityConfig, RestartRequiredChange, RetentionConfig, RetryConfig, ServerConfig,
    ShieldingConfig, ThinkingConfig, UpstreamConfig,
};
pub use reload::{
    ConfigHandle, ConfigManager, MissingConfigPolicy, ReloadOutcome, ReloadWatcher,
    default_config_path,
};

/// Default config location relative to the user's home directory.
pub const DEFAULT_CONFIG_RELATIVE_PATH: &str = ".config/llm-guard-proxy/config.toml";

/// Fields that can be changed by reloading the config file.
pub const RELOADABLE_FIELDS: &[&str] = &[
    "shielding.enabled",
    "observability.enabled",
    "observability.capture_raw_payloads",
    "observability.retention.max_bytes",
    "observability.retention.prune_to_bytes",
    "observability.retention.max_records",
    "thinking.enabled",
    "thinking.budget_tokens",
    "thinking.preserve_answer_budget",
    "loop_guard.enabled",
    "loop_guard.normalized_input_window_secs",
    "loop_guard.max_repeated_inputs",
    "retry.enabled",
    "retry.max_attempts",
    "heartbeat.mode",
    "heartbeat.interval_secs",
    "cloudflare.enabled",
    "upstream.metadata.discovery_enabled",
    "upstream.metadata.enrich_responses",
    "upstream.metadata.refresh_interval_secs",
    "upstream.metadata.context_length_override",
    "upstream.metadata.max_model_len_override",
];

/// Fields read at process startup that require a restart when changed.
pub const RESTART_REQUIRED_FIELDS: &[&str] = &[
    "server.bind_host",
    "server.port",
    "upstream.base_url",
    "observability.sqlite_path",
];
