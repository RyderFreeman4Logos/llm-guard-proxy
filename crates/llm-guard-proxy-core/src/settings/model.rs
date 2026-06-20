use std::path::PathBuf;

use super::ValidationError;

/// Complete application configuration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AppConfig {
    /// Process listener settings. These are restart-required.
    pub server: ServerConfig,
    /// Upstream OpenAI-compatible service settings.
    pub upstream: UpstreamConfig,
    /// Client shielding behavior flags.
    pub shielding: ShieldingConfig,
    /// Observability storage and retention settings.
    pub observability: ObservabilityConfig,
    /// Thinking budget policy for later request rewriting.
    pub thinking: ThinkingConfig,
    /// Loop detection policy.
    pub loop_guard: LoopGuardConfig,
    /// Retry policy for shielded upstream attempts.
    pub retry: RetryConfig,
    /// Downstream liveness policy.
    pub heartbeat: HeartbeatConfig,
    /// Cloudflare compatibility policy for later timeout shielding.
    pub cloudflare: CloudflareConfig,
}

impl AppConfig {
    /// Validates cross-field and range constraints.
    ///
    /// # Errors
    ///
    /// Returns a [`ValidationError`] when a field is empty, zero where zero is
    /// not meaningful, or violates a cross-field relation.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.server.validate()?;
        self.upstream.validate()?;
        self.observability.validate()?;
        self.loop_guard.validate()?;
        self.retry.validate()?;
        self.heartbeat.validate()
    }

    pub(crate) fn apply_reloadable_from(&mut self, requested: &Self) {
        self.shielding = requested.shielding.clone();
        self.observability.enabled = requested.observability.enabled;
        self.observability.capture_raw_payloads = requested.observability.capture_raw_payloads;
        self.observability.retention = requested.observability.retention.clone();
        self.thinking = requested.thinking.clone();
        self.loop_guard = requested.loop_guard.clone();
        self.retry = requested.retry.clone();
        self.heartbeat = requested.heartbeat.clone();
        self.cloudflare = requested.cloudflare.clone();
        self.upstream.metadata = requested.upstream.metadata.clone();
    }

    pub(crate) fn restart_required_changes(&self, requested: &Self) -> Vec<RestartRequiredChange> {
        let mut changes = Vec::new();
        push_change(
            &mut changes,
            "server.bind_host",
            self.server.bind_host.clone(),
            requested.server.bind_host.clone(),
        );
        push_change(
            &mut changes,
            "server.port",
            self.server.port.to_string(),
            requested.server.port.to_string(),
        );
        push_change(
            &mut changes,
            "upstream.base_url",
            self.upstream.base_url.clone(),
            requested.upstream.base_url.clone(),
        );
        push_change(
            &mut changes,
            "observability.sqlite_path",
            self.observability.sqlite_path.display().to_string(),
            requested.observability.sqlite_path.display().to_string(),
        );
        changes
    }
}

/// Listener settings read during process startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConfig {
    /// Interface or hostname to bind.
    pub bind_host: String,
    /// TCP port for the proxy listener.
    pub port: u16,
}

impl ServerConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.bind_host.trim().is_empty(),
            "server.bind_host",
            "must not be empty",
        )?;
        require(self.port > 0, "server.port", "must be between 1 and 65535")
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_host: String::from("127.0.0.1"),
            port: 18_009,
        }
    }
}

/// Upstream OpenAI-compatible service settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamConfig {
    /// Base URL for OpenAI-compatible requests.
    pub base_url: String,
    /// Metadata discovery and model context enrichment policy.
    pub metadata: MetadataConfig,
}

impl UpstreamConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        let has_supported_scheme =
            self.base_url.starts_with("http://") || self.base_url.starts_with("https://");
        require(
            has_supported_scheme,
            "upstream.base_url",
            "must start with http:// or https://",
        )?;
        self.metadata.validate()
    }
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            base_url: String::from("http://gb10:18009/v1"),
            metadata: MetadataConfig::default(),
        }
    }
}

/// Upstream model metadata discovery policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetadataConfig {
    /// Enables polling `/v1/models` for model metadata.
    pub discovery_enabled: bool,
    /// Adds normalized context metadata to downstream model records.
    pub enrich_responses: bool,
    /// Refresh interval for metadata discovery.
    pub refresh_interval_secs: u64,
    /// Optional context length override used when upstream metadata is absent.
    pub context_length_override: Option<u32>,
    /// Optional `max_model_len` override used when upstream metadata is absent.
    pub max_model_len_override: Option<u32>,
}

impl MetadataConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.refresh_interval_secs > 0,
            "upstream.metadata.refresh_interval_secs",
            "must be greater than zero",
        )?;
        require_optional_positive(
            self.context_length_override,
            "upstream.metadata.context_length_override",
        )?;
        require_optional_positive(
            self.max_model_len_override,
            "upstream.metadata.max_model_len_override",
        )
    }
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            discovery_enabled: true,
            enrich_responses: true,
            refresh_interval_secs: 60,
            context_length_override: None,
            max_model_len_override: None,
        }
    }
}

/// Shielded-response behavior flags.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShieldingConfig {
    /// Enables internal shielding before content is released downstream.
    pub enabled: bool,
}

impl Default for ShieldingConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Observability policy and storage settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityConfig {
    /// Enables indexed observability metadata.
    pub enabled: bool,
    /// `SQLite` metadata path. This is restart-required when changed.
    pub sqlite_path: PathBuf,
    /// Enables raw prompt/output sidecars for explicitly configured debugging.
    pub capture_raw_payloads: bool,
    /// Retention limits for metadata and artifacts.
    pub retention: RetentionConfig,
}

impl ObservabilityConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.sqlite_path.as_os_str().is_empty(),
            "observability.sqlite_path",
            "must not be empty",
        )?;
        self.retention.validate()
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sqlite_path: PathBuf::from("~/.local/state/llm-guard-proxy/observability.sqlite3"),
            capture_raw_payloads: false,
            retention: RetentionConfig::default(),
        }
    }
}

/// Retention limits for observability records and artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionConfig {
    /// Hard maximum actual storage budget in bytes.
    ///
    /// `SQLite` stores schema and page metadata, so a database cannot shrink
    /// below its empty schema footprint.
    pub max_bytes: u64,
    /// Hysteresis target after pruning.
    ///
    /// Values below the empty `SQLite` footprint prune rows but cannot reduce
    /// the database file below that storage floor.
    pub prune_to_bytes: u64,
    /// Maximum indexed record count.
    pub max_records: u64,
}

impl RetentionConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.max_bytes > 0,
            "observability.retention.max_bytes",
            "must be greater than zero",
        )?;
        require(
            self.prune_to_bytes > 0,
            "observability.retention.prune_to_bytes",
            "must be greater than zero",
        )?;
        require(
            self.prune_to_bytes <= self.max_bytes,
            "observability.retention.prune_to_bytes",
            "must be less than or equal to max_bytes",
        )?;
        require(
            self.max_records > 0,
            "observability.retention.max_records",
            "must be greater than zero",
        )
    }
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            max_bytes: 1_073_741_824,
            prune_to_bytes: 805_306_368,
            max_records: 100_000,
        }
    }
}

/// Thinking budget policy for later request rewriting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThinkingConfig {
    /// Enables default thinking budget injection.
    pub enabled: bool,
    /// Thinking token budget. A zero budget disables injection.
    pub budget_tokens: u32,
    /// Adjusts `max_tokens` so callers keep their apparent answer budget.
    pub preserve_answer_budget: bool,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            budget_tokens: 32_768,
            preserve_answer_budget: true,
        }
    }
}

/// Loop detection policy for repeated normalized inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopGuardConfig {
    /// Enables normalized repeated-input detection.
    pub enabled: bool,
    /// Time window used to compare normalized inputs.
    pub normalized_input_window_secs: u64,
    /// Repeat threshold that triggers loop-protection behavior.
    pub max_repeated_inputs: u32,
}

impl LoopGuardConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.normalized_input_window_secs > 0,
            "loop_guard.normalized_input_window_secs",
            "must be greater than zero",
        )?;
        require(
            self.max_repeated_inputs > 0,
            "loop_guard.max_repeated_inputs",
            "must be greater than zero",
        )
    }
}

impl Default for LoopGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            normalized_input_window_secs: 120,
            max_repeated_inputs: 1,
        }
    }
}

/// Retry policy for shielded upstream attempts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetryConfig {
    /// Enables retry after a bad or failed shielded attempt.
    pub enabled: bool,
    /// Total upstream attempts, including the first attempt.
    pub max_attempts: u32,
}

impl RetryConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.max_attempts > 0,
            "retry.max_attempts",
            "must be greater than zero",
        )
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 2,
        }
    }
}

/// Downstream heartbeat strategy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeartbeatConfig {
    /// Liveness mode used while the proxy shields upstream attempts.
    pub mode: HeartbeatMode,
    /// Heartbeat interval for streaming or whitespace progress.
    pub interval_secs: u64,
}

impl HeartbeatConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.interval_secs > 0,
            "heartbeat.interval_secs",
            "must be greater than zero",
        )
    }
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            mode: HeartbeatMode::Sse,
            interval_secs: 15,
        }
    }
}

/// Supported downstream heartbeat modes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HeartbeatMode {
    /// Server-sent-event heartbeat/progress frames.
    #[default]
    Sse,
    /// Leading whitespace heartbeat for non-stream JSON responses.
    JsonWhitespace,
    /// No heartbeat.
    Disabled,
}

impl HeartbeatMode {
    /// Returns the TOML-compatible mode label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sse => "sse",
            Self::JsonWhitespace => "json-whitespace",
            Self::Disabled => "disabled",
        }
    }
}

/// Cloudflare compatibility policy reserved for timeout-sensitive deployments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CloudflareConfig {
    /// Enables Cloudflare-aware timeout shielding behavior in future service code.
    pub enabled: bool,
}

impl Default for CloudflareConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// A restart-required field change detected during reload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestartRequiredChange {
    /// Config field name.
    pub field: &'static str,
    /// Currently active value.
    pub active: String,
    /// Requested value from the reloaded file.
    pub requested: String,
}

fn push_change(
    changes: &mut Vec<RestartRequiredChange>,
    field: &'static str,
    active: String,
    requested: String,
) {
    if active != requested {
        changes.push(RestartRequiredChange {
            field,
            active,
            requested,
        });
    }
}

fn require(
    condition: bool,
    field: &'static str,
    message: &'static str,
) -> Result<(), ValidationError> {
    if condition {
        Ok(())
    } else {
        Err(ValidationError::new(field, message))
    }
}

fn require_optional_positive(
    value: Option<u32>,
    field: &'static str,
) -> Result<(), ValidationError> {
    require(value.unwrap_or(1) > 0, field, "must be greater than zero")
}
