#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::HashSet,
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use super::ValidationError;
use url::Url;

const REDACTED_URL_PART: &str = "redacted";
const INVALID_URL_DISPLAY: &str = "[invalid URL]";
const LOOP_GUARD_MAX_SEMANTIC_WINDOW_TOKENS: u32 = 256;
const LOOP_GUARD_MAX_SEMANTIC_HISTORY_WINDOWS: u32 = 256;
const DEFAULT_UPSTREAM_PROFILE_NAME: &str = "default";
const MAX_UPSTREAM_PROFILE_NAME_BYTES: usize = 128;
const MAX_UPSTREAM_MODEL_ALIAS_BYTES: usize = 256;

/// Complete application configuration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AppConfig {
    /// Process listener settings. These are restart-required.
    pub server: ServerConfig,
    /// Additional downstream listener sockets. These are restart-required.
    pub listeners: Vec<ListenerConfig>,
    /// Upstream OpenAI-compatible service settings.
    pub upstream: UpstreamConfig,
    /// Additional named upstream profiles matched by request model.
    pub upstream_profiles: Vec<UpstreamProfileConfig>,
    /// Client shielding behavior flags.
    pub shielding: ShieldingConfig,
    /// Observability storage and retention settings.
    pub observability: ObservabilityConfig,
    /// Opt-in shadow evidence ledger settings.
    pub evidence: EvidenceConfig,
    /// Thinking budget policy for later request rewriting.
    pub thinking: ThinkingConfig,
    /// Loop detection policy.
    pub loop_guard: LoopGuardConfig,
    /// Retry policy for shielded upstream attempts.
    pub retry: RetryConfig,
    /// Upstream no-progress detection and recovery policy.
    pub upstream_stall: UpstreamStallConfig,
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
        self.validate_upstream_profiles()?;
        self.validate_listeners()?;
        self.observability.validate()?;
        self.evidence.validate()?;
        self.thinking.validate("thinking.max_tokens")?;
        self.loop_guard.validate()?;
        self.retry.validate()?;
        self.upstream_stall.validate()?;
        self.validate_upstream_stall_timeout_order()?;
        self.heartbeat.validate()
    }

    fn validate_upstream_profiles(&self) -> Result<(), ValidationError> {
        let mut names = HashSet::from([DEFAULT_UPSTREAM_PROFILE_NAME.to_owned()]);
        let mut match_models = HashSet::new();

        for profile in &self.upstream_profiles {
            profile.validate()?;
            require(
                names.insert(profile.name.clone()),
                "upstreams.name",
                "must be unique and must not duplicate the implicit default profile",
            )?;
            for model in &profile.match_models {
                require(
                    !model.trim().is_empty(),
                    "upstreams.match_models",
                    "must not contain empty model aliases",
                )?;
                require(
                    match_models.insert(model.clone()),
                    "upstreams.match_models",
                    "model aliases must be unique across upstream profiles",
                )?;
            }
        }

        Ok(())
    }

    fn validate_listeners(&self) -> Result<(), ValidationError> {
        let allowed_profile_names = self.upstream_profile_names();
        let mut names = HashSet::from([self.default_listener().name]);
        let mut ports = HashSet::from([self.server.port]);

        for listener in &self.listeners {
            listener.validate()?;
            require(
                names.insert(listener.name.clone()),
                "listeners.name",
                "must be unique and must not duplicate the implicit default listener",
            )?;
            // Fail closed for same-port listeners. Wildcard/specific conflicts and
            // same-port behavior across address families depend on OS socket options.
            require(
                ports.insert(listener.port),
                "listeners.port",
                "listener ports must be unique to avoid startup bind conflicts",
            )?;
            if let Some(allowed_upstreams) = &listener.allowed_upstreams {
                require(
                    !allowed_upstreams.is_empty(),
                    "listeners.allowed_upstreams",
                    "must not be empty when set",
                )?;
                let mut names = HashSet::new();
                for upstream in allowed_upstreams {
                    require(
                        names.insert(upstream.clone()),
                        "listeners.allowed_upstreams",
                        "must not contain duplicate upstream profile names",
                    )?;
                    require(
                        allowed_profile_names.contains(upstream),
                        "listeners.allowed_upstreams",
                        "must reference default or a configured upstream profile name",
                    )?;
                }
            }
        }

        Ok(())
    }

    fn validate_upstream_stall_timeout_order(&self) -> Result<(), ValidationError> {
        if self.upstream_stall.enabled {
            require(
                self.upstream_stall.idle_timeout_ms < self.upstream.request_timeout_ms,
                "upstream.stall.idle_timeout_ms",
                "must be less than upstream.request_timeout_ms when upstream stall recovery is enabled",
            )?;
            for profile in &self.upstream_profiles {
                require(
                    self.upstream_stall.idle_timeout_ms < profile.request_timeout_ms,
                    "upstream.stall.idle_timeout_ms",
                    "must be less than every upstream profile request_timeout_ms when upstream stall recovery is enabled",
                )?;
            }
        }
        Ok(())
    }

    fn upstream_profile_names(&self) -> HashSet<String> {
        let mut names = HashSet::from([DEFAULT_UPSTREAM_PROFILE_NAME.to_owned()]);
        names.extend(
            self.upstream_profiles
                .iter()
                .map(|profile| profile.name.clone()),
        );
        names
    }

    /// Selects the effective upstream profile for a request model.
    #[must_use]
    pub fn select_upstream_profile(&self, model: Option<&str>) -> SelectedUpstreamProfile {
        let model = model.and_then(|model| {
            let trimmed = model.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        });
        if let Some(model) = model {
            if let Some(profile) = self
                .upstream_profiles
                .iter()
                .find(|profile| profile.matches_model(model))
            {
                return SelectedUpstreamProfile {
                    profile: profile.clone(),
                    route_reason: UpstreamRouteReason::MatchedModel,
                };
            }
            return SelectedUpstreamProfile {
                profile: self.default_upstream_profile(),
                route_reason: UpstreamRouteReason::DefaultUnmatchedModel,
            };
        }

        SelectedUpstreamProfile {
            profile: self.default_upstream_profile(),
            route_reason: UpstreamRouteReason::DefaultNoModel,
        }
    }

    /// Selects an upstream profile by configured profile name.
    #[must_use]
    pub fn upstream_profile_by_name(&self, name: &str) -> Option<UpstreamProfileConfig> {
        if name == DEFAULT_UPSTREAM_PROFILE_NAME {
            return Some(self.default_upstream_profile());
        }
        self.upstream_profiles
            .iter()
            .find(|profile| profile.name == name)
            .cloned()
    }

    /// Builds the implicit default profile from legacy `[upstream]` and `[thinking]`.
    #[must_use]
    pub fn default_upstream_profile(&self) -> UpstreamProfileConfig {
        UpstreamProfileConfig {
            name: DEFAULT_UPSTREAM_PROFILE_NAME.to_owned(),
            match_models: Vec::new(),
            base_url: self.upstream.base_url.clone(),
            request_timeout_ms: self.upstream.request_timeout_ms,
            max_in_flight_requests: None,
            max_queued_generation_requests: None,
            metadata: self.upstream.metadata.clone(),
            thinking: self.thinking.clone(),
        }
    }

    /// Builds the implicit listener from legacy `[server]` settings.
    #[must_use]
    pub fn default_listener(&self) -> ListenerConfig {
        ListenerConfig {
            name: String::from("default"),
            bind_host: self.server.bind_host.clone(),
            port: self.server.port,
            allowed_upstreams: None,
        }
    }

    /// Returns the legacy listener followed by all configured extra listeners.
    #[must_use]
    pub fn effective_listeners(&self) -> Vec<ListenerConfig> {
        let mut listeners = Vec::with_capacity(self.listeners.len().saturating_add(1));
        listeners.push(self.default_listener());
        listeners.extend(self.listeners.clone());
        listeners
    }

    /// Returns bind addresses for every effective downstream listener.
    ///
    /// The first address is always the legacy `[server]` listener so callers
    /// that only display one listener can keep using `default_listener()`.
    #[must_use]
    pub fn effective_listener_addresses(&self) -> Vec<String> {
        self.effective_listeners()
            .into_iter()
            .map(|listener| listener.bind_address())
            .collect()
    }

    /// Returns true when any named upstream declares independent generation admission limits.
    #[must_use]
    pub fn has_upstream_profile_generation_limits(&self) -> bool {
        self.upstream_profiles
            .iter()
            .any(UpstreamProfileConfig::has_generation_limits)
    }

    pub(crate) fn apply_reloadable_from(&mut self, requested: &Self) {
        self.server.max_in_flight_requests = requested.server.max_in_flight_requests;
        self.server.max_queued_generation_requests =
            requested.server.max_queued_generation_requests;
        self.server.generation_queue_timeout_ms = requested.server.generation_queue_timeout_ms;
        self.server.max_control_plane_in_flight_requests =
            requested.server.max_control_plane_in_flight_requests;
        self.server.max_request_body_bytes = requested.server.max_request_body_bytes;
        self.shielding = requested.shielding.clone();
        self.observability.enabled = requested.observability.enabled;
        self.observability.capture_raw_payloads = requested.observability.capture_raw_payloads;
        self.observability.metrics_enabled = requested.observability.metrics_enabled;
        self.observability.health_upstream_probe_enabled =
            requested.observability.health_upstream_probe_enabled;
        self.observability.health_upstream_probe_timeout_ms =
            requested.observability.health_upstream_probe_timeout_ms;
        self.observability.debug_summary_enabled = requested.observability.debug_summary_enabled;
        self.observability
            .debug_summary_admin_token
            .clone_from(&requested.observability.debug_summary_admin_token);
        self.observability.debug_summary_max_records =
            requested.observability.debug_summary_max_records;
        self.observability.retention = requested.observability.retention.clone();
        self.evidence.enabled = requested.evidence.enabled;
        self.evidence.include_raw_payloads = requested.evidence.include_raw_payloads;
        self.evidence.include_request_headers = requested.evidence.include_request_headers;
        self.evidence.max_bytes = requested.evidence.max_bytes;
        self.evidence.prune_to_bytes = requested.evidence.prune_to_bytes;
        self.evidence.max_records = requested.evidence.max_records;
        self.evidence.prune_to_records = requested.evidence.prune_to_records;
        self.evidence.shadow = requested.evidence.shadow.clone();
        self.thinking = requested.thinking.clone();
        self.loop_guard = requested.loop_guard.clone();
        self.retry = requested.retry.clone();
        self.upstream_stall = requested.upstream_stall.clone();
        self.heartbeat = requested.heartbeat.clone();
        self.cloudflare = requested.cloudflare.clone();
        self.upstream.request_timeout_ms = requested.upstream.request_timeout_ms;
        self.upstream.metadata = requested.upstream.metadata.clone();
        if self.upstream_profiles_topology_matches(requested) {
            self.apply_reloadable_upstream_profile_fields(requested);
        }
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
        push_structural_change(
            &mut changes,
            "listeners.topology",
            &self.listener_topology(),
            &requested.listener_topology(),
        );
        push_change(
            &mut changes,
            "upstream.base_url",
            self.upstream.base_url.clone(),
            requested.upstream.base_url.clone(),
        );
        push_structural_change(
            &mut changes,
            "upstreams.topology",
            &self.upstream_profile_topology(),
            &requested.upstream_profile_topology(),
        );
        push_change(
            &mut changes,
            "observability.sqlite_path",
            self.observability.sqlite_path.display().to_string(),
            requested.observability.sqlite_path.display().to_string(),
        );
        push_change(
            &mut changes,
            "evidence.sqlite_path",
            self.evidence.sqlite_path.display().to_string(),
            requested.evidence.sqlite_path.display().to_string(),
        );
        push_change(
            &mut changes,
            "evidence.blob_cache_dir",
            self.evidence.blob_cache_dir.display().to_string(),
            requested.evidence.blob_cache_dir.display().to_string(),
        );
        changes
    }

    fn listener_topology(&self) -> Vec<ListenerTopology> {
        self.listeners.iter().map(ListenerTopology::from).collect()
    }

    fn upstream_profiles_topology_matches(&self, requested: &Self) -> bool {
        self.upstream_profile_topology() == requested.upstream_profile_topology()
    }

    fn upstream_profile_topology(&self) -> Vec<UpstreamProfileTopology> {
        self.upstream_profiles
            .iter()
            .map(UpstreamProfileTopology::from)
            .collect()
    }

    fn apply_reloadable_upstream_profile_fields(&mut self, requested: &Self) {
        for (active, requested) in self
            .upstream_profiles
            .iter_mut()
            .zip(requested.upstream_profiles.iter())
        {
            active.request_timeout_ms = requested.request_timeout_ms;
            active.max_in_flight_requests = requested.max_in_flight_requests;
            active.max_queued_generation_requests = requested.max_queued_generation_requests;
            active.metadata = requested.metadata.clone();
            active.thinking = requested.thinking.clone();
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ListenerTopology {
    name: String,
    bind_host: String,
    port: u16,
    allowed_upstreams: Option<Vec<String>>,
}

impl From<&ListenerConfig> for ListenerTopology {
    fn from(listener: &ListenerConfig) -> Self {
        Self {
            name: listener.name.clone(),
            bind_host: listener.bind_host.clone(),
            port: listener.port,
            allowed_upstreams: listener.allowed_upstreams.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UpstreamProfileTopology {
    name: String,
    base_url: String,
    match_models: Vec<String>,
}

impl From<&UpstreamProfileConfig> for UpstreamProfileTopology {
    fn from(profile: &UpstreamProfileConfig) -> Self {
        Self {
            name: profile.name.clone(),
            base_url: profile.base_url.clone(),
            match_models: profile.match_models.clone(),
        }
    }
}

/// Listener settings read during process startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConfig {
    /// Interface or hostname to bind.
    pub bind_host: String,
    /// TCP port for the proxy listener.
    pub port: u16,
    /// Maximum proxied requests admitted into body buffering and upstream forwarding.
    pub max_in_flight_requests: usize,
    /// Maximum generation requests allowed to wait for an in-flight slot.
    pub max_queued_generation_requests: usize,
    /// Maximum milliseconds a queued generation request may wait for capacity.
    pub generation_queue_timeout_ms: u64,
    /// Maximum `/v1/models` requests admitted into upstream forwarding.
    pub max_control_plane_in_flight_requests: usize,
    /// Maximum downstream request body bytes buffered before forwarding.
    pub max_request_body_bytes: usize,
}

impl ServerConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.bind_host.trim().is_empty(),
            "server.bind_host",
            "must not be empty",
        )?;
        require(self.port > 0, "server.port", "must be between 1 and 65535")?;
        require(
            self.max_in_flight_requests > 0,
            "server.max_in_flight_requests",
            "must be greater than zero",
        )?;
        require(
            self.max_queued_generation_requests <= 10_000,
            "server.max_queued_generation_requests",
            "must be less than or equal to 10000",
        )?;
        require(
            self.generation_queue_timeout_ms > 0,
            "server.generation_queue_timeout_ms",
            "must be greater than zero",
        )?;
        require(
            self.max_control_plane_in_flight_requests > 0,
            "server.max_control_plane_in_flight_requests",
            "must be greater than zero",
        )?;
        require(
            self.max_control_plane_in_flight_requests <= 1_024,
            "server.max_control_plane_in_flight_requests",
            "must be less than or equal to 1024",
        )?;
        require(
            self.max_request_body_bytes > 0,
            "server.max_request_body_bytes",
            "must be greater than zero",
        )?;
        require(
            self.max_request_body_bytes <= 1_073_741_824,
            "server.max_request_body_bytes",
            "must be less than or equal to 1073741824",
        )
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_host: String::from("127.0.0.1"),
            port: 18_009,
            max_in_flight_requests: 16,
            max_queued_generation_requests: 64,
            generation_queue_timeout_ms: 30_000,
            max_control_plane_in_flight_requests: 128,
            max_request_body_bytes: 67_108_864,
        }
    }
}

/// One downstream TCP listener and optional upstream profile allow-list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListenerConfig {
    /// Stable listener identity stored in observability metadata.
    pub name: String,
    /// Interface or hostname to bind.
    pub bind_host: String,
    /// TCP port for this listener.
    pub port: u16,
    /// Allowed upstream profile names. `None` means all configured profiles.
    pub allowed_upstreams: Option<Vec<String>>,
}

impl ListenerConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.name.trim().is_empty(),
            "listeners.name",
            "must not be empty",
        )?;
        require(
            self.name == self.name.trim(),
            "listeners.name",
            "must not have leading or trailing whitespace",
        )?;
        require(
            self.name.len() <= MAX_UPSTREAM_PROFILE_NAME_BYTES,
            "listeners.name",
            "must be at most 128 bytes",
        )?;
        require(
            !self.bind_host.trim().is_empty(),
            "listeners.bind_host",
            "must not be empty",
        )?;
        require(
            self.port > 0,
            "listeners.port",
            "must be between 1 and 65535",
        )?;
        if let Some(allowed_upstreams) = &self.allowed_upstreams {
            for upstream in allowed_upstreams {
                require(
                    !upstream.trim().is_empty(),
                    "listeners.allowed_upstreams",
                    "must not contain empty upstream profile names",
                )?;
                require(
                    upstream == upstream.trim(),
                    "listeners.allowed_upstreams",
                    "upstream profile names must not have leading or trailing whitespace",
                )?;
                require(
                    upstream.len() <= MAX_UPSTREAM_PROFILE_NAME_BYTES,
                    "listeners.allowed_upstreams",
                    "upstream profile names must be at most 128 bytes",
                )?;
            }
        }
        Ok(())
    }

    /// Returns true when this listener may route to the selected upstream profile.
    #[must_use]
    pub fn allows_upstream(&self, profile_name: &str) -> bool {
        self.allowed_upstreams
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|upstream| upstream == profile_name))
    }

    /// Returns this listener's bind address in host:port form.
    #[must_use]
    pub fn bind_address(&self) -> String {
        if self.bind_host.contains(':')
            && !(self.bind_host.starts_with('[') && self.bind_host.ends_with(']'))
        {
            return format!("[{}]:{}", self.bind_host, self.port);
        }
        format!("{}:{}", self.bind_host, self.port)
    }
}

impl Default for ListenerConfig {
    fn default() -> Self {
        let server = ServerConfig::default();
        Self {
            name: String::new(),
            bind_host: server.bind_host,
            port: server.port,
            allowed_upstreams: None,
        }
    }
}

/// Upstream OpenAI-compatible service settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamConfig {
    /// Base URL for OpenAI-compatible requests.
    pub base_url: String,
    /// Total upstream request timeout, including streamed response body reads.
    pub request_timeout_ms: u64,
    /// Metadata discovery and model context enrichment policy.
    pub metadata: MetadataConfig,
}

impl UpstreamConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_upstream_base_url(&self.base_url)?;
        require(
            self.request_timeout_ms > 0,
            "upstream.request_timeout_ms",
            "must be greater than zero",
        )?;
        self.metadata.validate()
    }

    /// Returns a display-safe upstream base URL.
    ///
    /// Credentials and query strings are replaced and fragments are removed
    /// before the string is suitable for logs or client-visible diagnostics.
    #[must_use]
    pub fn redacted_base_url(&self) -> String {
        redact_upstream_base_url(&self.base_url)
    }
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            base_url: String::from("http://gb10:18009/v1"),
            request_timeout_ms: 120_000,
            metadata: MetadataConfig::default(),
        }
    }
}

/// Named upstream profile with model routing and per-upstream policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamProfileConfig {
    /// Unique profile name used in observability metadata.
    pub name: String,
    /// Request JSON `model` aliases routed to this profile.
    pub match_models: Vec<String>,
    /// Base URL for OpenAI-compatible requests.
    pub base_url: String,
    /// Total upstream request timeout, including streamed response body reads.
    pub request_timeout_ms: u64,
    /// Optional per-profile in-flight generation request limit.
    pub max_in_flight_requests: Option<usize>,
    /// Optional per-profile queue length for generation requests waiting on this profile.
    pub max_queued_generation_requests: Option<usize>,
    /// Metadata discovery and model context enrichment policy.
    pub metadata: MetadataConfig,
    /// Thinking budget policy for this profile.
    pub thinking: ThinkingConfig,
}

impl UpstreamProfileConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.name.trim().is_empty(),
            "upstreams.name",
            "must not be empty",
        )?;
        require(
            self.name == self.name.trim(),
            "upstreams.name",
            "must not have leading or trailing whitespace",
        )?;
        require(
            self.name.len() <= MAX_UPSTREAM_PROFILE_NAME_BYTES,
            "upstreams.name",
            "must be at most 128 bytes",
        )?;
        require(
            !self.match_models.is_empty(),
            "upstreams.match_models",
            "must contain at least one model alias",
        )?;
        for model in &self.match_models {
            require(
                model == model.trim(),
                "upstreams.match_models",
                "model aliases must not have leading or trailing whitespace",
            )?;
            require(
                model.len() <= MAX_UPSTREAM_MODEL_ALIAS_BYTES,
                "upstreams.match_models",
                "model aliases must be at most 256 bytes",
            )?;
        }
        validate_upstream_base_url(&self.base_url)?;
        require(
            self.request_timeout_ms > 0,
            "upstreams.request_timeout_ms",
            "must be greater than zero",
        )?;
        if let Some(max_in_flight_requests) = self.max_in_flight_requests {
            require(
                max_in_flight_requests > 0,
                "upstreams.max_in_flight_requests",
                "must be greater than zero",
            )?;
        }
        if let Some(max_queued_generation_requests) = self.max_queued_generation_requests {
            require(
                max_queued_generation_requests <= 10_000,
                "upstreams.max_queued_generation_requests",
                "must be less than or equal to 10000",
            )?;
        }
        self.metadata.validate()?;
        self.thinking.validate("upstreams.thinking.max_tokens")
    }

    /// Returns true when this profile uses an independent generation admission limiter.
    #[must_use]
    pub const fn has_generation_limits(&self) -> bool {
        self.max_in_flight_requests.is_some() || self.max_queued_generation_requests.is_some()
    }

    /// Effective in-flight limit for this profile, inheriting the server default when omitted.
    #[must_use]
    pub fn effective_max_in_flight_requests(&self, server: &ServerConfig) -> usize {
        self.max_in_flight_requests
            .unwrap_or(server.max_in_flight_requests)
    }

    /// Effective queue limit for this profile, inheriting the server default when omitted.
    #[must_use]
    pub fn effective_max_queued_generation_requests(&self, server: &ServerConfig) -> usize {
        self.max_queued_generation_requests
            .unwrap_or(server.max_queued_generation_requests)
    }

    /// Returns true when the request model selects this profile.
    #[must_use]
    pub fn matches_model(&self, model: &str) -> bool {
        self.match_models.iter().any(|alias| alias == model)
    }

    /// Returns a display-safe upstream base URL.
    #[must_use]
    pub fn redacted_base_url(&self) -> String {
        redact_upstream_base_url(&self.base_url)
    }
}

impl Default for UpstreamProfileConfig {
    fn default() -> Self {
        let upstream = UpstreamConfig::default();
        Self {
            name: String::new(),
            match_models: Vec::new(),
            base_url: upstream.base_url,
            request_timeout_ms: upstream.request_timeout_ms,
            max_in_flight_requests: None,
            max_queued_generation_requests: None,
            metadata: upstream.metadata,
            thinking: ThinkingConfig::default(),
        }
    }
}

/// Effective upstream profile selected for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedUpstreamProfile {
    /// Selected profile settings.
    pub profile: UpstreamProfileConfig,
    /// Why this profile was selected.
    pub route_reason: UpstreamRouteReason,
}

/// Bounded route reason stored in observability metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpstreamRouteReason {
    /// Request model matched a named profile alias.
    MatchedModel,
    /// Request had no usable model, so the implicit default profile was used.
    DefaultNoModel,
    /// Request model did not match any named profile, so the implicit default was used.
    DefaultUnmatchedModel,
}

impl UpstreamRouteReason {
    /// Returns the stable metadata label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MatchedModel => "matched_model",
            Self::DefaultNoModel => "default_no_model",
            Self::DefaultUnmatchedModel => "default_unmatched_model",
        }
    }
}

/// Validates the configured upstream base URL.
///
/// # Errors
///
/// Returns a [`ValidationError`] when the URL is not absolute HTTP(S), includes
/// userinfo, contains query parameters, or includes a fragment.
pub fn validate_upstream_base_url(base_url: &str) -> Result<(), ValidationError> {
    let url = Url::parse(base_url).map_err(|_error| {
        ValidationError::new(
            "upstream.base_url",
            "must be a valid http:// or https:// URL",
        )
    })?;
    require(
        matches!(url.scheme(), "http" | "https"),
        "upstream.base_url",
        "must start with http:// or https://",
    )?;
    require(
        url.username().is_empty() && url.password().is_none(),
        "upstream.base_url",
        "must not contain username, password, or userinfo",
    )?;
    if url.query().is_some() {
        return Err(ValidationError::new(
            "upstream.base_url",
            "must not contain query parameters",
        ));
    }
    require(
        url.fragment().is_none(),
        "upstream.base_url",
        "must not contain URL fragments",
    )?;
    Ok(())
}

/// Returns a display-safe URL string for logs and diagnostics.
///
/// Invalid URLs are rendered as a fixed marker because preserving fragments of
/// malformed input risks echoing secrets embedded in an unparsable string.
#[must_use]
pub fn redact_upstream_base_url(base_url: &str) -> String {
    let Ok(mut url) = Url::parse(base_url) else {
        return INVALID_URL_DISPLAY.to_owned();
    };

    if !url.username().is_empty() {
        let _ignored = url.set_username(REDACTED_URL_PART);
    }
    if url.password().is_some() {
        let _ignored = url.set_password(Some(REDACTED_URL_PART));
    }
    if url.query().is_some() {
        url.set_query(Some(REDACTED_URL_PART));
    }
    url.set_fragment(None);

    url.to_string()
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
    /// Reserved input-token margin subtracted during context-budget preflight.
    pub input_token_safety_margin: u32,
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
        )?;
        Ok(())
    }

    /// Returns a configured fallback context window when discovery is absent.
    #[must_use]
    pub fn context_window_override(&self) -> Option<u32> {
        self.context_length_override.or(self.max_model_len_override)
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
            input_token_safety_margin: 0,
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

/// Two-state config toggle used to keep endpoint switches explicit in the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigToggle {
    Disabled,
    Enabled,
}

impl ConfigToggle {
    #[must_use]
    pub const fn from_bool(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }

    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Metadata capture and retention settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityConfig {
    /// Enables indexed observability metadata.
    pub enabled: bool,
    /// `SQLite` metadata path. This is restart-required when changed.
    pub sqlite_path: PathBuf,
    /// Enables raw prompt/output sidecars for explicitly configured debugging.
    pub capture_raw_payloads: bool,
    /// Enables the Prometheus-compatible `/metrics` endpoint.
    pub metrics_enabled: ConfigToggle,
    /// Enables bounded upstream probing from `/health`.
    pub health_upstream_probe_enabled: ConfigToggle,
    /// Maximum time spent probing upstream readiness from `/health`.
    pub health_upstream_probe_timeout_ms: u64,
    /// Enables the gated recent-request debug summary endpoint.
    pub debug_summary_enabled: ConfigToggle,
    /// Optional bearer/admin token required for the debug summary endpoint.
    pub debug_summary_admin_token: Option<String>,
    /// Maximum recent request summaries returned by one debug response.
    pub debug_summary_max_records: u32,
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
        require(
            self.health_upstream_probe_timeout_ms > 0,
            "observability.health_upstream_probe_timeout_ms",
            "must be greater than zero",
        )?;
        require(
            self.health_upstream_probe_timeout_ms <= 30_000,
            "observability.health_upstream_probe_timeout_ms",
            "must be less than or equal to 30000",
        )?;
        require(
            self.debug_summary_max_records > 0,
            "observability.debug_summary_max_records",
            "must be greater than zero",
        )?;
        require(
            self.debug_summary_max_records <= 100,
            "observability.debug_summary_max_records",
            "must be less than or equal to 100",
        )?;
        if let Some(token) = &self.debug_summary_admin_token {
            require(
                !token.trim().is_empty(),
                "observability.debug_summary_admin_token",
                "must not be empty when set",
            )?;
        }
        self.retention.validate()
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sqlite_path: PathBuf::from("~/.local/state/llm-guard-proxy/observability.sqlite3"),
            capture_raw_payloads: false,
            metrics_enabled: ConfigToggle::Enabled,
            health_upstream_probe_enabled: ConfigToggle::Enabled,
            health_upstream_probe_timeout_ms: 500,
            debug_summary_enabled: ConfigToggle::Disabled,
            debug_summary_admin_token: None,
            debug_summary_max_records: 20,
            retention: RetentionConfig::default(),
        }
    }
}

/// Opt-in evidence storage and raw payload capture settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceConfig {
    /// Enables the shadow evidence ledger.
    pub enabled: bool,
    /// `SQLite` evidence ledger path. This is restart-required when changed.
    pub sqlite_path: PathBuf,
    /// Reserved cache directory for future bounded raw payload sidecars.
    pub blob_cache_dir: PathBuf,
    /// Enables sensitive raw prompt/output/reasoning/tool payload capture.
    pub include_raw_payloads: bool,
    /// Includes non-secret request/header metadata. Secrets remain redacted.
    pub include_request_headers: bool,
    /// Hard maximum actual evidence storage budget in bytes.
    pub max_bytes: u64,
    /// Hysteresis target after byte pruning.
    pub prune_to_bytes: u64,
    /// Maximum retained evidence rows across groups, attempts, and chunks.
    pub max_records: u64,
    /// Optional hysteresis target after record-count pruning.
    pub prune_to_records: Option<u64>,
    /// Shadow continuation limits.
    pub shadow: EvidenceShadowConfig,
}

impl EvidenceConfig {
    /// Effective evidence record-count pruning target.
    #[must_use]
    pub fn effective_prune_to_records(&self) -> u64 {
        self.prune_to_records
            .unwrap_or_else(|| default_prune_to_records(self.max_records))
    }

    fn validate(&self) -> Result<(), ValidationError> {
        require(
            !self.sqlite_path.as_os_str().is_empty(),
            "evidence.sqlite_path",
            "must not be empty",
        )?;
        require(
            path_has_explicit_parent(&self.sqlite_path),
            "evidence.sqlite_path",
            "must include an explicit parent directory",
        )?;
        require(
            !self.blob_cache_dir.as_os_str().is_empty(),
            "evidence.blob_cache_dir",
            "must not be empty",
        )?;
        require(
            path_has_explicit_parent(&self.blob_cache_dir),
            "evidence.blob_cache_dir",
            "must include an explicit parent directory",
        )?;
        if self.enabled || self.sqlite_path != default_evidence_sqlite_path() {
            validate_evidence_sqlite_path(&self.sqlite_path)?;
        }
        if self.enabled || self.blob_cache_dir != default_evidence_blob_cache_dir() {
            validate_evidence_blob_cache_dir(&self.blob_cache_dir)?;
        }
        require(
            self.max_bytes > 0,
            "evidence.max_bytes",
            "must be greater than zero",
        )?;
        require(
            self.prune_to_bytes > 0,
            "evidence.prune_to_bytes",
            "must be greater than zero",
        )?;
        require(
            self.prune_to_bytes <= self.max_bytes,
            "evidence.prune_to_bytes",
            "must be less than or equal to max_bytes",
        )?;
        require(
            self.max_records > 0,
            "evidence.max_records",
            "must be greater than zero",
        )?;
        if let Some(prune_to_records) = self.prune_to_records {
            require(
                prune_to_records > 0,
                "evidence.prune_to_records",
                "must be greater than zero",
            )?;
            require(
                prune_to_records <= self.max_records,
                "evidence.prune_to_records",
                "must be less than or equal to max_records",
            )?;
        }
        self.shadow.validate()
    }
}

impl Default for EvidenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sqlite_path: default_evidence_sqlite_path(),
            blob_cache_dir: default_evidence_blob_cache_dir(),
            include_raw_payloads: false,
            include_request_headers: false,
            max_bytes: 10_737_418_240,
            prune_to_bytes: 8_589_934_592,
            max_records: 100_000,
            prune_to_records: None,
            shadow: EvidenceShadowConfig::default(),
        }
    }
}

/// Shadow continuation resource limits for evidence collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceShadowConfig {
    /// Enables evidence-only shadow bookkeeping after loop signals.
    pub enabled: bool,
    /// Requests continuing the looped high-thinking attempt for evidence.
    pub keep_looping_attempt_running: bool,
    /// Allows the fallback/downgrade ladder to run while shadow evidence is collected.
    pub parallel_downgrade_attempts: bool,
    /// Maximum evidence-only shadow attempts recorded for one downstream request.
    pub max_shadow_attempts_per_request: u32,
    /// Maximum evidence-only shadow attempts in flight across all requests.
    pub max_global_shadow_in_flight: usize,
    /// Terminal timeout for one evidence-only shadow attempt.
    pub shadow_attempt_timeout_ms: u64,
}

impl EvidenceShadowConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.max_shadow_attempts_per_request <= 10,
            "evidence.shadow.max_shadow_attempts_per_request",
            "must be less than or equal to 10",
        )?;
        require(
            self.max_global_shadow_in_flight <= 1_024,
            "evidence.shadow.max_global_shadow_in_flight",
            "must be less than or equal to 1024",
        )?;
        require(
            self.shadow_attempt_timeout_ms > 0,
            "evidence.shadow.shadow_attempt_timeout_ms",
            "must be greater than zero",
        )
    }
}

impl Default for EvidenceShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            keep_looping_attempt_running: false,
            parallel_downgrade_attempts: true,
            max_shadow_attempts_per_request: 2,
            max_global_shadow_in_flight: 2,
            shadow_attempt_timeout_ms: 7_200_000,
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
    /// Hysteresis target after record-count pruning.
    ///
    /// When omitted, the effective target is derived from `max_records`.
    pub prune_to_records: Option<u64>,
}

impl RetentionConfig {
    /// Effective record-count pruning target.
    ///
    /// Omitted config defaults to 80% of `max_records`, with a minimum target
    /// of one retained record for very small test configurations.
    #[must_use]
    pub fn effective_prune_to_records(&self) -> u64 {
        self.prune_to_records
            .unwrap_or_else(|| default_prune_to_records(self.max_records))
    }

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
        )?;
        if let Some(prune_to_records) = self.prune_to_records {
            require(
                prune_to_records > 0,
                "observability.retention.prune_to_records",
                "must be greater than zero",
            )?;
            require(
                prune_to_records <= self.max_records,
                "observability.retention.prune_to_records",
                "must be less than or equal to max_records",
            )?;
        }
        Ok(())
    }
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            max_bytes: 1_073_741_824,
            prune_to_bytes: 805_306_368,
            max_records: 100_000,
            prune_to_records: None,
        }
    }
}

const fn default_prune_to_records(max_records: u64) -> u64 {
    let gap = max_records / 5;
    let gap = if gap == 0 { 1 } else { gap };
    let target = max_records.saturating_sub(gap);
    if target == 0 { 1 } else { target }
}

fn path_has_explicit_parent(path: &Path) -> bool {
    path.parent()
        .is_some_and(|parent| !parent.as_os_str().is_empty())
}

fn default_evidence_sqlite_path() -> PathBuf {
    evidence_sqlite_path_from_xdg_state_home(env::var_os("XDG_STATE_HOME").map(PathBuf::from))
}

fn default_evidence_blob_cache_dir() -> PathBuf {
    evidence_blob_cache_dir_from_xdg_cache_home(env::var_os("XDG_CACHE_HOME").map(PathBuf::from))
}

pub(super) fn evidence_sqlite_path_from_xdg_state_home(xdg_state_home: Option<PathBuf>) -> PathBuf {
    xdg_state_home
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("~/.local/state"))
        .join("llm-guard-proxy")
        .join("evidence.sqlite3")
}

pub(super) fn evidence_blob_cache_dir_from_xdg_cache_home(
    xdg_cache_home: Option<PathBuf>,
) -> PathBuf {
    xdg_cache_home
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("~/.cache"))
        .join("llm-guard-proxy")
        .join("evidence")
        .join("blobs")
}

fn validate_evidence_sqlite_path(path: &Path) -> Result<(), ValidationError> {
    const FIELD: &str = "evidence.sqlite_path";
    let resolved = resolve_evidence_validation_path(path, FIELD)?;
    reject_symlink_path_components(&resolved, FIELD)?;
    if let Some(metadata) = path_metadata_if_exists(&resolved, FIELD)? {
        require(
            !metadata.file_type().is_symlink() && metadata.is_file(),
            FIELD,
            "must be a regular file when it already exists",
        )?;
    }
    if let Some(parent) = resolved.parent() {
        validate_existing_owner_private_directory(parent, FIELD)?;
    }
    Ok(())
}

fn validate_evidence_blob_cache_dir(path: &Path) -> Result<(), ValidationError> {
    const FIELD: &str = "evidence.blob_cache_dir";
    let resolved = resolve_evidence_validation_path(path, FIELD)?;
    reject_symlink_path_components(&resolved, FIELD)?;
    validate_existing_owner_private_directory(&resolved, FIELD)?;
    if let Some(parent) = resolved.parent() {
        validate_existing_owner_private_directory(parent, FIELD)?;
    }
    Ok(())
}

fn resolve_evidence_validation_path(
    path: &Path,
    field: &'static str,
) -> Result<PathBuf, ValidationError> {
    if path.starts_with("~") {
        let home = env::var_os("HOME").ok_or_else(|| {
            ValidationError::new(field, "HOME must be set when evidence path starts with ~")
        })?;
        let suffix = path.strip_prefix("~").unwrap_or(path);
        return Ok(PathBuf::from(home).join(suffix));
    }
    Ok(path.to_path_buf())
}

fn reject_symlink_path_components(path: &Path, field: &'static str) -> Result<(), ValidationError> {
    let mut inspected = PathBuf::new();
    for component in path.components() {
        inspected.push(component.as_os_str());
        match fs::symlink_metadata(&inspected) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ValidationError::new(
                    field,
                    format!(
                        "must not contain symlink path component {}",
                        inspected.display()
                    ),
                ));
            }
            Ok(_metadata) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => break,
            Err(error) => {
                return Err(ValidationError::new(
                    field,
                    format!("must be inspectable: {error}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_existing_owner_private_directory(
    path: &Path,
    field: &'static str,
) -> Result<(), ValidationError> {
    let Some(metadata) = path_metadata_if_exists(path, field)? else {
        return Ok(());
    };
    require(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        field,
        "existing storage parent must be a real directory",
    )?;
    #[cfg(unix)]
    {
        let mode = metadata.permissions().mode() & 0o777;
        require(
            mode.trailing_zeros() >= 6,
            field,
            "existing storage parent must not be accessible by group or other users",
        )?;
    }
    Ok(())
}

fn path_metadata_if_exists(
    path: &Path,
    field: &'static str,
) -> Result<Option<fs::Metadata>, ValidationError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(ValidationError::new(
            field,
            format!("must be inspectable: {error}"),
        )),
    }
}

/// Thinking-budget behavior for requests that carry tool/function-calling hints.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolRequestThinkingPolicy {
    /// Apply the regular thinking policy to every chat request, including tool-use.
    #[default]
    Apply,
    /// Leave caller-provided thinking fields untouched for tool-use requests.
    Passthrough,
}

impl ToolRequestThinkingPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Apply => "apply",
            Self::Passthrough => "passthrough",
        }
    }
}

/// How force-thinking mode treats explicit caller no-thinking markers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NoThinkingMarkerPolicy {
    /// Force thinking even when callers explicitly ask for no-thinking.
    #[default]
    Force,
    /// Leave explicit caller no-thinking requests untouched.
    RespectNoThinkingMarkers,
    /// Force normal no-thinking markers but honor the proxy escape hatch.
    EscapeHatchOnly,
}

impl NoThinkingMarkerPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Force => "force",
            Self::RespectNoThinkingMarkers => "respect_no_thinking_markers",
            Self::EscapeHatchOnly => "escape_hatch_only",
        }
    }
}

/// Profile thinking rewrite mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ThinkingMode {
    /// Leave caller thinking fields untouched.
    Passthrough,
    /// Force upstream thinking off.
    ForceDisable,
    /// Force the configured thinking budget even when callers disabled thinking.
    ForceThinking,
    /// Inject or raise thinking up to the configured budget while respecting caller disablement.
    #[default]
    BoundedThinking,
}

impl ThinkingMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::ForceDisable => "force_disable",
            Self::ForceThinking => "force_thinking",
            Self::BoundedThinking => "bounded_thinking",
        }
    }
}

/// Default schema for injecting a thinking budget when no existing thinking markers are present.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DefaultInjectionSchema {
    /// Use the canonical `thinking.budget_tokens` schema.
    #[default]
    Canonical,
    /// Use `chat_template_kwargs.enable_thinking` and `chat_template_kwargs.thinking_budget`.
    ///
    /// Required by AEON/Qwen vLLM backends that ignore the canonical thinking schema.
    ChatTemplateKwargs,
}

impl DefaultInjectionSchema {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::ChatTemplateKwargs => "chat_template_kwargs",
        }
    }
}

/// Thinking budget policy for later request rewriting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThinkingConfig {
    /// Profile rewrite mode.
    pub mode: ThinkingMode,
    /// Enables default thinking budget injection.
    pub enabled: bool,
    /// Forces all recognized upstream thinking budgets to zero.
    pub force_disable: bool,
    /// Optional output cap written to OpenAI-compatible token limit fields.
    pub max_tokens: Option<u32>,
    /// Thinking token budget. A zero budget disables injection.
    pub budget_tokens: u32,
    /// Adjusts `max_tokens` so callers keep their apparent answer budget.
    pub preserve_answer_budget: bool,
    /// Request-class policy for tool/function-calling requests.
    pub tool_request_policy: ToolRequestThinkingPolicy,
    /// How force-thinking mode treats explicit caller no-thinking markers.
    pub no_thinking_marker_policy: NoThinkingMarkerPolicy,
    /// Default schema for injecting thinking budget into requests without existing markers.
    pub default_injection_schema: DefaultInjectionSchema,
}

impl ThinkingConfig {
    /// Returns the mode after applying legacy switches.
    #[must_use]
    pub const fn effective_mode(&self) -> ThinkingMode {
        if self.force_disable {
            ThinkingMode::ForceDisable
        } else if !self.enabled {
            ThinkingMode::Passthrough
        } else {
            self.mode
        }
    }

    /// Returns the TOML-compatible output budget accounting label.
    #[must_use]
    pub const fn budget_accounting(&self) -> &'static str {
        if self.preserve_answer_budget {
            "preserve_answer_budget"
        } else {
            "total_cap"
        }
    }

    fn validate(&self, max_tokens_field: &'static str) -> Result<(), ValidationError> {
        require_optional_positive(self.max_tokens, max_tokens_field)
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            mode: ThinkingMode::BoundedThinking,
            enabled: true,
            force_disable: false,
            max_tokens: None,
            budget_tokens: 32_768,
            preserve_answer_budget: true,
            tool_request_policy: ToolRequestThinkingPolicy::Apply,
            no_thinking_marker_policy: NoThinkingMarkerPolicy::Force,
            default_injection_schema: DefaultInjectionSchema::Canonical,
        }
    }
}

/// Loop detector decision mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LoopGuardMode {
    /// Skip detector construction and feature calculation.
    Disabled,
    /// Record bounded detector signals but never abort or retry.
    #[default]
    Monitor,
    /// Abort on high-confidence abort candidates.
    Enforce,
}

impl LoopGuardMode {
    /// Returns the TOML-compatible mode label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Monitor => "monitor",
            Self::Enforce => "enforce",
        }
    }

    /// Returns true when detector work should be skipped.
    #[must_use]
    pub const fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }
}

/// Loop detection policy for repeated normalized inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopGuardConfig {
    /// Enables loop-guard behavior. `false` is equivalent to mode `disabled`.
    pub enabled: bool,
    /// Channelized detector decision mode.
    pub mode: LoopGuardMode,
    /// Time window used to compare normalized inputs.
    pub normalized_input_window_secs: u64,
    /// Repeat threshold that triggers loop-protection behavior.
    pub max_repeated_inputs: u32,
    /// Repeated complete output lines required before aborting.
    pub output_repeated_line_threshold: u32,
    /// Normalized output token window size used for repetition detection.
    pub output_token_window_size: u32,
    /// Repeated normalized token windows required before aborting.
    pub output_repeated_token_window_threshold: u32,
    /// Suffix cycle repetitions required before aborting.
    pub output_suffix_cycle_threshold: u32,
    /// Minimum channel bytes before low-progress detection can abort.
    pub output_low_progress_min_bytes: u64,
    /// Maximum unique token-window ratio allowed for low-progress detection.
    pub output_low_progress_unique_ratio_percent: u32,
    /// Threshold multiplier applied when output repetition overlaps repeated input.
    pub input_overlap_threshold_multiplier: u32,
    /// Enables semantic Jaccard detection for reasoning/thinking streams.
    pub reasoning_semantic_detection_enabled: bool,
    /// Minimum Jaccard similarity required to flag a semantic reasoning loop.
    pub reasoning_semantic_similarity_threshold_percent: u32,
    /// Significant reasoning tokens kept in each semantic comparison window.
    pub reasoning_semantic_window_token_count: u32,
    /// Significant tokens required before a partial semantic window can be compared.
    pub reasoning_semantic_minimum_token_count: u32,
    /// Maximum completed semantic windows kept for bounded history comparison.
    pub reasoning_semantic_history_window_count: u32,
}

impl LoopGuardConfig {
    /// Returns the detector mode after applying the legacy `enabled` switch.
    #[must_use]
    pub const fn effective_mode(&self) -> LoopGuardMode {
        if self.enabled {
            self.mode
        } else {
            LoopGuardMode::Disabled
        }
    }

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
        )?;
        require(
            self.output_repeated_line_threshold > 0,
            "loop_guard.output_repeated_line_threshold",
            "must be greater than zero",
        )?;
        require(
            self.output_token_window_size > 0,
            "loop_guard.output_token_window_size",
            "must be greater than zero",
        )?;
        require(
            self.output_repeated_token_window_threshold > 0,
            "loop_guard.output_repeated_token_window_threshold",
            "must be greater than zero",
        )?;
        require(
            self.output_suffix_cycle_threshold > 0,
            "loop_guard.output_suffix_cycle_threshold",
            "must be greater than zero",
        )?;
        require(
            self.output_low_progress_min_bytes > 0,
            "loop_guard.output_low_progress_min_bytes",
            "must be greater than zero",
        )?;
        require(
            self.output_low_progress_unique_ratio_percent <= 100,
            "loop_guard.output_low_progress_unique_ratio_percent",
            "must be between 0 and 100",
        )?;
        require(
            self.input_overlap_threshold_multiplier > 0,
            "loop_guard.input_overlap_threshold_multiplier",
            "must be greater than zero",
        )?;
        require(
            (1..=100).contains(&self.reasoning_semantic_similarity_threshold_percent),
            "loop_guard.reasoning_semantic_similarity_threshold_percent",
            "must be between 1 and 100",
        )?;
        require(
            self.reasoning_semantic_window_token_count > 0
                && self.reasoning_semantic_window_token_count
                    <= LOOP_GUARD_MAX_SEMANTIC_WINDOW_TOKENS,
            "loop_guard.reasoning_semantic_window_token_count",
            "must be between 1 and 256",
        )?;
        require(
            self.reasoning_semantic_minimum_token_count > 0,
            "loop_guard.reasoning_semantic_minimum_token_count",
            "must be greater than zero",
        )?;
        require(
            self.reasoning_semantic_minimum_token_count
                <= self.reasoning_semantic_window_token_count,
            "loop_guard.reasoning_semantic_minimum_token_count",
            "must be less than or equal to loop_guard.reasoning_semantic_window_token_count",
        )?;
        require(
            self.reasoning_semantic_history_window_count > 0
                && self.reasoning_semantic_history_window_count
                    <= LOOP_GUARD_MAX_SEMANTIC_HISTORY_WINDOWS,
            "loop_guard.reasoning_semantic_history_window_count",
            "must be between 1 and 256",
        )
    }
}

impl Default for LoopGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: LoopGuardMode::Monitor,
            normalized_input_window_secs: 120,
            max_repeated_inputs: 1,
            output_repeated_line_threshold: 24,
            output_token_window_size: 12,
            output_repeated_token_window_threshold: 32,
            output_suffix_cycle_threshold: 32,
            output_low_progress_min_bytes: 4_096,
            output_low_progress_unique_ratio_percent: 15,
            input_overlap_threshold_multiplier: 4,
            reasoning_semantic_detection_enabled: true,
            reasoning_semantic_similarity_threshold_percent: 55,
            reasoning_semantic_window_token_count: 24,
            reasoning_semantic_minimum_token_count: 8,
            reasoning_semantic_history_window_count: 16,
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
    /// Adds a bounded deterministic anti-loop hint to retries after loop aborts.
    pub anti_loop_hint_enabled: bool,
    /// Routes downstream `stream=true` chat completions through shielded retry.
    pub shielded_streaming_enabled: bool,
    /// Controls upstream attempt lifetime after a downstream response body drop.
    pub downstream_drop_policy: DownstreamDropPolicy,
    /// Optional named retry ladder entries. Empty preserves legacy repeated attempts.
    pub ladder: Vec<RetryLadderConfig>,
}

impl RetryConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.max_attempts > 0,
            "retry.max_attempts",
            "must be greater than zero",
        )?;
        require(
            self.max_attempts <= 10,
            "retry.max_attempts",
            "must be less than or equal to 10",
        )?;
        require(
            self.ladder.len() <= 10,
            "retry.ladder",
            "must contain at most 10 entries",
        )?;
        for (index, entry) in self.ladder.iter().enumerate() {
            entry.validate(index)?;
        }
        Ok(())
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 5,
            anti_loop_hint_enabled: true,
            shielded_streaming_enabled: false,
            downstream_drop_policy: DownstreamDropPolicy::Cancel,
            ladder: Vec::new(),
        }
    }
}

/// Upstream attempt behavior after a downstream body is dropped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DownstreamDropPolicy {
    /// Dropping the downstream body cancels the in-progress upstream attempt.
    #[default]
    Cancel,
    /// Dropping the downstream body detaches the in-progress upstream attempt.
    Detach,
}

impl DownstreamDropPolicy {
    /// Returns the TOML-compatible label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancel => "cancel",
            Self::Detach => "detach",
        }
    }
}

/// One named rung in the shielded retry ladder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RetryLadderConfig {
    /// Bounded name recorded in observability.
    pub name: String,
    /// Thinking rewrite policy selected for this attempt.
    pub thinking: ThinkingConfig,
    /// Optional bounded behavioral hint for loop-triggered retries.
    pub anti_loop_hint: Option<String>,
}

impl RetryLadderConfig {
    fn validate(&self, _index: usize) -> Result<(), ValidationError> {
        require(
            !self.name.trim().is_empty(),
            "retry.ladder.name",
            "must not be empty",
        )?;
        require(
            self.name == self.name.trim(),
            "retry.ladder.name",
            "must not have leading or trailing whitespace",
        )?;
        require(
            self.name.len() <= 64,
            "retry.ladder.name",
            "must be at most 64 bytes",
        )?;
        self.thinking.validate("retry.ladder.max_tokens")?;
        if let Some(hint) = &self.anti_loop_hint {
            require(
                !hint.trim().is_empty(),
                "retry.ladder.anti_loop_hint",
                "must not be empty when set",
            )?;
            require(
                hint.len() <= 512,
                "retry.ladder.anti_loop_hint",
                "must be at most 512 bytes",
            )?;
        }
        Ok(())
    }
}

/// Upstream no-progress detection and optional recovery hook.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamStallConfig {
    /// Enables shielded chat aggregation idle-timeout detection.
    pub enabled: bool,
    /// Maximum milliseconds to wait for the next upstream SSE chunk.
    pub idle_timeout_ms: u64,
    /// Optional argv command run after a stall before retrying.
    ///
    /// Empty means recovery is disabled. Each element is passed as a single
    /// argv item; the proxy never invokes a shell. A configured command must
    /// perform the complete recovery procedure, including any restart and
    /// post-restart readiness or smoke checks, because retries are allowed
    /// only after this command exits successfully.
    pub recovery_command: Vec<String>,
    /// Maximum milliseconds to wait for the recovery command.
    pub recovery_timeout_ms: u64,
    /// Minimum milliseconds between completed recovery command executions.
    pub recovery_cooldown_ms: u64,
    /// Rolling window used by the recovery execution budget.
    pub recovery_budget_window_ms: u64,
    /// Maximum recovery command executions inside one budget window.
    pub recovery_max_per_window: u32,
}

impl UpstreamStallConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        require(
            self.idle_timeout_ms > 0,
            "upstream.stall.idle_timeout_ms",
            "must be greater than zero",
        )?;
        require(
            self.recovery_timeout_ms > 0,
            "upstream.stall.recovery_timeout_ms",
            "must be greater than zero",
        )?;
        require(
            self.recovery_command
                .iter()
                .all(|argument| !argument.trim().is_empty()),
            "upstream.stall.recovery_command",
            "must not contain empty argv entries",
        )?;
        require(
            self.recovery_cooldown_ms > 0,
            "upstream.stall.recovery_cooldown_ms",
            "must be greater than zero",
        )?;
        require(
            self.recovery_budget_window_ms > 0,
            "upstream.stall.recovery_budget_window_ms",
            "must be greater than zero",
        )?;
        require(
            self.recovery_budget_window_ms <= 86_400_000,
            "upstream.stall.recovery_budget_window_ms",
            "must be less than or equal to 86400000",
        )?;
        require(
            self.recovery_max_per_window > 0,
            "upstream.stall.recovery_max_per_window",
            "must be greater than zero",
        )?;
        require(
            self.recovery_max_per_window <= 100,
            "upstream.stall.recovery_max_per_window",
            "must be less than or equal to 100",
        )
    }
}

impl Default for UpstreamStallConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_timeout_ms: 30_000,
            recovery_command: Vec::new(),
            recovery_timeout_ms: 300_000,
            recovery_cooldown_ms: 300_000,
            recovery_budget_window_ms: 900_000,
            recovery_max_per_window: 2,
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

fn push_structural_change<T>(
    changes: &mut Vec<RestartRequiredChange>,
    field: &'static str,
    active: &T,
    requested: &T,
) where
    T: Eq + std::fmt::Debug,
{
    if active != requested {
        changes.push(RestartRequiredChange {
            field,
            active: format!("{active:?}"),
            requested: format!("{requested:?}"),
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
