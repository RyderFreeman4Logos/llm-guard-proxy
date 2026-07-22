use std::path::PathBuf;

#[cfg(feature = "family")]
use crate::family::{CategoryAction, CategoryConfig, FamilyCategory};
#[cfg(feature = "guard")]
use crate::model_alias::{AliasKind, ModelAliasConfig};
#[cfg(feature = "guard")]
use crate::profile::{ProfileConfig, ProfileKind, ShieldedBuffering};
#[cfg(feature = "guard")]
use crate::workflow::{WorkflowConfig, WorkflowRuntime};

#[cfg(feature = "param-override")]
use super::ParamOverrideConfig;
use super::{
    AppConfig, CloudflareConfig, ConfigParseError, ConfigToggle, DefaultInjectionSchema,
    DownstreamDropPolicy, EndpointSelectionMode, GuardianConfig, GuardianKillAction,
    HeartbeatConfig, HeartbeatMode, HotRestartConfig, ListenerConfig, LocalRecoveryConfig,
    LoopFailurePolicy, LoopGuardConfig, LoopGuardMode, MetadataConfig, NoThinkingMarkerPolicy,
    ObservabilityConfig, RestartQueueConfig, RetentionConfig, RetryConfig, RetryLadderConfig,
    ServerConfig, ShadowComparisonAttempt, ShieldingConfig, StuckWatchdogConfig, ThinkingConfig,
    ThinkingMode, ToolRequestThinkingPolicy, UpstreamConfig, UpstreamEndpointConfig,
    UpstreamEndpointProtocol, UpstreamPriority, UpstreamProfileConfig, UpstreamStallConfig,
};
#[cfg(feature = "guard")]
use super::{UnknownKeyPolicy, VirtualKeyConfig};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Section {
    Root,
    Server,
    Listener(usize),
    Upstream,
    UpstreamMetadata,
    UpstreamHotRestart,
    UpstreamLocalRecovery,
    UpstreamStuckWatchdog,
    UpstreamRestartQueue,
    UpstreamProfile(usize),
    UpstreamProfileEndpoint {
        profile: usize,
        endpoint: usize,
    },
    UpstreamProfileMetadata(usize),
    UpstreamProfileHotRestart(usize),
    UpstreamProfileLocalRecovery(usize),
    UpstreamProfileStuckWatchdog(usize),
    UpstreamProfileRestartQueue(usize),
    UpstreamProfileThinking(usize),
    #[cfg(feature = "param-override")]
    UpstreamProfileParamOverride(usize),
    #[cfg(feature = "guard")]
    ModelAlias(usize),
    #[cfg(feature = "guard")]
    Profile(String),
    #[cfg(feature = "guard")]
    Workflow(String),
    #[cfg(feature = "guard")]
    GuardWorkflows,
    #[cfg(feature = "guard")]
    VirtualKeys,
    #[cfg(feature = "guard")]
    VirtualKeyMap,
    #[cfg(feature = "guard")]
    Budget,
    #[cfg(feature = "family")]
    Family,
    #[cfg(feature = "family")]
    FamilyCategory(FamilyCategory),
    Shielding,
    Observability,
    ObservabilityRetention,
    Evidence,
    EvidenceShadow,
    EvidenceShadowPairedComparison,
    Thinking,
    LoopGuard,
    LoopGuardEmbedding,
    Retry,
    RetryLadder(usize),
    UpstreamStall,
    Heartbeat,
    Cloudflare,
    Guardian,
}

pub(crate) fn parse_config_text(contents: &str) -> Result<AppConfig, ConfigParseError> {
    parse_config_text_with_defaults(contents, AppConfig::default())
}

pub(crate) fn parse_config_text_with_defaults(
    contents: &str,
    mut config: AppConfig,
) -> Result<AppConfig, ConfigParseError> {
    let mut section = Section::Root;
    let mut current_upstream_profile = None;

    for (index, raw_line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line_without_comment = strip_comment(raw_line, line_number)?;
        let line = line_without_comment.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            section = parse_section(
                &mut config,
                &mut current_upstream_profile,
                line,
                line_number,
            )?;
            continue;
        }
        let (key, value) = split_key_value(line, line_number)?;
        assign_value(&mut config, &section, key.trim(), value.trim(), line_number)?;
    }

    #[cfg(feature = "family")]
    config.apply_family_defaults();

    Ok(config)
}

fn strip_comment(line: &str, line_number: usize) -> Result<&str, ConfigParseError> {
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if character == '"' && !escaped {
            in_string = !in_string;
        }
        if character == '#' && !in_string {
            return Ok(&line[..index]);
        }
        escaped = character == '\\' && !escaped;
        if character != '\\' {
            escaped = false;
        }
    }
    if in_string {
        Err(ConfigParseError::new(line_number, "unterminated string"))
    } else {
        Ok(line)
    }
}

#[allow(clippy::too_many_lines)]
fn parse_section(
    config: &mut AppConfig,
    current_upstream_profile: &mut Option<usize>,
    line: &str,
    line_number: usize,
) -> Result<Section, ConfigParseError> {
    if !line.ends_with(']') {
        return Err(ConfigParseError::new(
            line_number,
            "section header must end with ]",
        ));
    }
    if line == "[[upstreams]]" || line == "[[profile]]" {
        config
            .upstream_profiles
            .push(UpstreamProfileConfig::default());
        let index = config.upstream_profiles.len() - 1;
        *current_upstream_profile = Some(index);
        return Ok(Section::UpstreamProfile(index));
    }
    if matches!(
        line,
        "[[profile.upstream]]"
            | "[[upstreams.upstream]]"
            | "[[profile.endpoints]]"
            | "[[upstreams.endpoints]]"
    ) {
        let profile = current_upstream_profile.ok_or_else(|| {
            ConfigParseError::new(
                line_number,
                "[[profile.upstream]] must follow a [[profile]] or [[upstreams]] profile",
            )
        })?;
        config.upstream_profiles[profile]
            .endpoints
            .push(UpstreamEndpointConfig::default());
        let endpoint = config.upstream_profiles[profile].endpoints.len() - 1;
        return Ok(Section::UpstreamProfileEndpoint { profile, endpoint });
    }
    if line == "[[listeners]]" {
        config.listeners.push(ListenerConfig::default());
        let index = config.listeners.len() - 1;
        return Ok(Section::Listener(index));
    }
    #[cfg(feature = "guard")]
    if line == "[[model_aliases]]" {
        config.model_aliases.push(ModelAliasConfig::default());
        let index = config.model_aliases.len() - 1;
        return Ok(Section::ModelAlias(index));
    }
    if line == "[[retry.ladder]]" {
        config.retry.ladder.push(RetryLadderConfig::default());
        return Ok(Section::RetryLadder(config.retry.ladder.len() - 1));
    }
    let section = &line[1..line.len() - 1];
    #[cfg(feature = "guard")]
    if let Some(raw_workflow_id) = section.strip_prefix("workflows.") {
        let workflow_id = raw_workflow_id.trim_matches('"');
        if workflow_id.trim().is_empty() {
            return Err(ConfigParseError::new(
                line_number,
                "workflow section id must not be empty",
            ));
        }
        if workflow_id != workflow_id.trim() || workflow_id.contains(char::is_whitespace) {
            return Err(ConfigParseError::new(
                line_number,
                "workflow section id must not contain whitespace",
            ));
        }
        if config
            .workflows
            .insert(workflow_id.to_owned(), WorkflowConfig::default())
            .is_some()
        {
            return Err(ConfigParseError::new(
                line_number,
                format!("duplicate workflow section [{section}]"),
            ));
        }
        return Ok(Section::Workflow(workflow_id.to_owned()));
    }
    #[cfg(feature = "guard")]
    if let Some(raw_profile_name) = section.strip_prefix("profiles.") {
        let profile_name = raw_profile_name.trim_matches('"');
        if profile_name.trim().is_empty() {
            return Err(ConfigParseError::new(
                line_number,
                "profile section name must not be empty",
            ));
        }
        if profile_name != profile_name.trim() {
            return Err(ConfigParseError::new(
                line_number,
                "profile section name must not have leading or trailing whitespace",
            ));
        }
        if config
            .profiles
            .insert(profile_name.to_owned(), ProfileConfig::default())
            .is_some()
        {
            return Err(ConfigParseError::new(
                line_number,
                format!("duplicate profile section [{section}]"),
            ));
        }
        return Ok(Section::Profile(profile_name.to_owned()));
    }
    #[cfg(feature = "family")]
    if let Some(raw_category) = section.strip_prefix("family.categories.") {
        let category_key = raw_category.trim_matches('"');
        let Some(category) = FamilyCategory::from_key(category_key) else {
            return Err(ConfigParseError::new(
                line_number,
                format!("unknown family category {category_key:?}"),
            ));
        };
        return Ok(Section::FamilyCategory(category));
    }
    match section {
        "server" => Ok(Section::Server),
        "upstream" => Ok(Section::Upstream),
        "upstream.metadata" => Ok(Section::UpstreamMetadata),
        "upstream.hot_restart" => Ok(Section::UpstreamHotRestart),
        "upstream.local_recovery" => Ok(Section::UpstreamLocalRecovery),
        "upstream.stuck_watchdog" => Ok(Section::UpstreamStuckWatchdog),
        "upstream.restart_queue" => Ok(Section::UpstreamRestartQueue),
        "upstreams.metadata" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.metadata] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileMetadata(index)),
        ),
        "upstreams.hot_restart" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.hot_restart] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileHotRestart(index)),
        ),
        "upstreams.local_recovery" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.local_recovery] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileLocalRecovery(index)),
        ),
        "upstreams.stuck_watchdog" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.stuck_watchdog] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileStuckWatchdog(index)),
        ),
        "upstreams.restart_queue" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.restart_queue] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileRestartQueue(index)),
        ),
        "upstreams.thinking" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.thinking] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileThinking(index)),
        ),
        #[cfg(feature = "param-override")]
        "upstreams.param_override" => current_upstream_profile.map_or_else(
            || {
                Err(ConfigParseError::new(
                    line_number,
                    "[upstreams.param_override] must follow a [[upstreams]] profile",
                ))
            },
            |index| Ok(Section::UpstreamProfileParamOverride(index)),
        ),
        #[cfg(feature = "guard")]
        "guard_workflows" => Ok(Section::GuardWorkflows),
        #[cfg(feature = "guard")]
        "virtual_keys" => Ok(Section::VirtualKeys),
        #[cfg(feature = "guard")]
        "virtual_keys.keys" => Ok(Section::VirtualKeyMap),
        #[cfg(feature = "guard")]
        "budget" => Ok(Section::Budget),
        #[cfg(feature = "family")]
        "family" => Ok(Section::Family),
        "shielding" => Ok(Section::Shielding),
        "observability" => Ok(Section::Observability),
        "observability.retention" => Ok(Section::ObservabilityRetention),
        "evidence" => Ok(Section::Evidence),
        "evidence.shadow" => Ok(Section::EvidenceShadow),
        "evidence.shadow.paired_comparison" => Ok(Section::EvidenceShadowPairedComparison),
        "thinking" => Ok(Section::Thinking),
        "loop_guard" => Ok(Section::LoopGuard),
        "loop_guard.embedding" => Ok(Section::LoopGuardEmbedding),
        "retry" => Ok(Section::Retry),
        "upstream.stall" => Ok(Section::UpstreamStall),
        "heartbeat" => Ok(Section::Heartbeat),
        "cloudflare" => Ok(Section::Cloudflare),
        "guardian" => Ok(Section::Guardian),
        _ => Err(ConfigParseError::new(
            line_number,
            format!("unknown config section [{section}]"),
        )),
    }
}

fn split_key_value(line: &str, line_number: usize) -> Result<(&str, &str), ConfigParseError> {
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if character == '"' && !escaped {
            in_string = !in_string;
        }
        if character == '=' && !in_string {
            return Ok((&line[..index], &line[index + 1..]));
        }
        escaped = character == '\\' && !escaped;
        if character != '\\' {
            escaped = false;
        }
    }
    Err(ConfigParseError::new(
        line_number,
        "expected key = value entry",
    ))
}

fn assign_value(
    config: &mut AppConfig,
    section: &Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    #[cfg(feature = "guard")]
    if let Some(result) = assign_guard_value(config, section, key, value, line_number) {
        return result;
    }
    #[cfg(feature = "family")]
    if let Some(result) = assign_family_value(config, section, key, value, line_number) {
        return result;
    }
    if let Some(result) = assign_upstream_value(config, section, key, value, line_number) {
        return result;
    }

    match section {
        Section::Root => Err(ConfigParseError::new(
            line_number,
            "config keys must be inside a section",
        )),
        Section::Server => assign_server(&mut config.server, key, value, line_number),
        Section::Listener(index) => {
            assign_listener(&mut config.listeners[*index], key, value, line_number)
        }
        Section::Shielding => assign_shielding(&mut config.shielding, key, value, line_number),
        Section::Observability => {
            assign_observability(&mut config.observability, key, value, line_number)
        }
        Section::ObservabilityRetention => {
            assign_retention(&mut config.observability.retention, key, value, line_number)
        }
        Section::Evidence => assign_evidence(&mut config.evidence, key, value, line_number),
        Section::EvidenceShadow => {
            assign_evidence_shadow(&mut config.evidence.shadow, key, value, line_number)
        }
        Section::EvidenceShadowPairedComparison => assign_evidence_paired_comparison(
            &mut config.evidence.shadow.paired_comparison,
            key,
            value,
            line_number,
        ),
        Section::Thinking => assign_thinking(&mut config.thinking, key, value, line_number),
        Section::LoopGuard => assign_loop_guard(&mut config.loop_guard, key, value, line_number),
        Section::LoopGuardEmbedding => {
            assign_loop_guard_embedding(&mut config.loop_guard.embedding, key, value, line_number)
        }
        Section::Retry => assign_retry(&mut config.retry, key, value, line_number),
        Section::RetryLadder(index) => {
            assign_retry_ladder(&mut config.retry.ladder[*index], key, value, line_number)
        }
        Section::UpstreamStall => {
            assign_upstream_stall(&mut config.upstream_stall, key, value, line_number)
        }
        Section::Heartbeat => assign_heartbeat(&mut config.heartbeat, key, value, line_number),
        Section::Cloudflare => assign_cloudflare(&mut config.cloudflare, key, value, line_number),
        Section::Guardian => assign_guardian(&mut config.guardian, key, value, line_number),
        #[cfg(feature = "guard")]
        Section::ModelAlias(_)
        | Section::Profile(_)
        | Section::Workflow(_)
        | Section::GuardWorkflows
        | Section::VirtualKeys
        | Section::VirtualKeyMap
        | Section::Budget => unreachable!("guard sections are handled before this match"),
        #[cfg(feature = "family")]
        Section::Family | Section::FamilyCategory(_) => {
            unreachable!("family sections are handled before this match")
        }
        Section::Upstream
        | Section::UpstreamMetadata
        | Section::UpstreamHotRestart
        | Section::UpstreamLocalRecovery
        | Section::UpstreamStuckWatchdog
        | Section::UpstreamRestartQueue
        | Section::UpstreamProfile(_)
        | Section::UpstreamProfileEndpoint { .. }
        | Section::UpstreamProfileMetadata(_)
        | Section::UpstreamProfileHotRestart(_)
        | Section::UpstreamProfileLocalRecovery(_)
        | Section::UpstreamProfileStuckWatchdog(_)
        | Section::UpstreamProfileRestartQueue(_)
        | Section::UpstreamProfileThinking(_) => {
            unreachable!("upstream sections are handled before this match")
        }
        #[cfg(feature = "param-override")]
        Section::UpstreamProfileParamOverride(_) => {
            unreachable!("upstream profile param override is handled before this match")
        }
    }
}

fn assign_upstream_value(
    config: &mut AppConfig,
    section: &Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Option<Result<(), ConfigParseError>> {
    if let Some(result) = assign_default_upstream_value(config, section, key, value, line_number) {
        return Some(result);
    }
    match section {
        Section::UpstreamProfile(index) => Some(assign_upstream_profile(
            &mut config.upstream_profiles[*index],
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileEndpoint { profile, endpoint } => {
            let profile = &mut config.upstream_profiles[*profile];
            let result = if matches!(
                key,
                "health_probe_interval"
                    | "health_probe_interval_ms"
                    | "health_probe_timeout"
                    | "health_probe_timeout_ms"
                    | "health_probe_max_wait"
                    | "health_probe_max_wait_ms"
            ) {
                assign_upstream_profile(profile, key, value, line_number)
            } else {
                assign_upstream_endpoint(&mut profile.endpoints[*endpoint], key, value, line_number)
            };
            if result.is_ok() {
                synchronize_primary_base_url(profile);
            }
            Some(result)
        }
        Section::UpstreamProfileMetadata(index) => Some(assign_metadata(
            &mut config.upstream_profiles[*index].metadata,
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileHotRestart(index) => Some(assign_hot_restart(
            &mut config.upstream_profiles[*index].hot_restart,
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileLocalRecovery(index) => Some(assign_local_recovery(
            &mut config.upstream_profiles[*index].local_recovery,
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileStuckWatchdog(index) => Some(assign_stuck_watchdog(
            &mut config.upstream_profiles[*index].stuck_watchdog,
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileRestartQueue(index) => Some(assign_restart_queue(
            &mut config.upstream_profiles[*index].restart_queue,
            key,
            value,
            line_number,
        )),
        Section::UpstreamProfileThinking(index) => Some(assign_thinking(
            &mut config.upstream_profiles[*index].thinking,
            key,
            value,
            line_number,
        )),
        #[cfg(feature = "param-override")]
        Section::UpstreamProfileParamOverride(index) => Some(assign_param_override(
            &mut config.upstream_profiles[*index].param_override,
            key,
            value,
            line_number,
        )),
        _ => None,
    }
}

fn assign_default_upstream_value(
    config: &mut AppConfig,
    section: &Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Option<Result<(), ConfigParseError>> {
    match section {
        Section::Upstream => Some(assign_upstream(
            &mut config.upstream,
            key,
            value,
            line_number,
        )),
        Section::UpstreamMetadata => Some(assign_metadata(
            &mut config.upstream.metadata,
            key,
            value,
            line_number,
        )),
        Section::UpstreamHotRestart => Some(assign_hot_restart(
            &mut config.upstream.hot_restart,
            key,
            value,
            line_number,
        )),
        Section::UpstreamLocalRecovery => Some(assign_local_recovery(
            &mut config.upstream.local_recovery,
            key,
            value,
            line_number,
        )),
        Section::UpstreamStuckWatchdog => Some(assign_stuck_watchdog(
            &mut config.upstream.stuck_watchdog,
            key,
            value,
            line_number,
        )),
        Section::UpstreamRestartQueue => Some(assign_restart_queue(
            &mut config.upstream.restart_queue,
            key,
            value,
            line_number,
        )),
        _ => None,
    }
}

#[cfg(feature = "guard")]
fn assign_guard_value(
    config: &mut AppConfig,
    section: &Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Option<Result<(), ConfigParseError>> {
    match section {
        Section::ModelAlias(index) => Some(assign_model_alias(
            &mut config.model_aliases[*index],
            key,
            value,
            line_number,
        )),
        Section::Profile(profile_name) => Some(
            config
                .profiles
                .get_mut(profile_name)
                .ok_or_else(|| {
                    ConfigParseError::new(
                        line_number,
                        format!("profile section {profile_name:?} was not initialized"),
                    )
                })
                .and_then(|profile| assign_profile(profile, key, value, line_number)),
        ),
        Section::Workflow(workflow_id) => Some(
            config
                .workflows
                .get_mut(workflow_id)
                .ok_or_else(|| {
                    ConfigParseError::new(
                        line_number,
                        format!("workflow section {workflow_id:?} was not initialized"),
                    )
                })
                .and_then(|workflow| assign_workflow(workflow, key, value, line_number)),
        ),
        Section::GuardWorkflows => Some(assign_guard_workflows(
            &mut config.guard_workflows,
            key,
            value,
            line_number,
        )),
        Section::VirtualKeys => Some(assign_virtual_keys(
            &mut config.virtual_keys,
            key,
            value,
            line_number,
        )),
        Section::VirtualKeyMap => Some(assign_virtual_key_map(
            &mut config.virtual_keys,
            key,
            value,
            line_number,
        )),
        Section::Budget => Some(assign_budget(&mut config.budget, key, value, line_number)),
        _ => None,
    }
}

#[cfg(feature = "family")]
fn assign_family_value(
    config: &mut AppConfig,
    section: &Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Option<Result<(), ConfigParseError>> {
    match section {
        Section::Family => Some(assign_family(&mut config.family, key, value, line_number)),
        Section::FamilyCategory(category) => {
            let default_category_config = config.family.category_config(*category);
            let category_config = config
                .family
                .categories
                .entry(*category)
                .or_insert(default_category_config);
            Some(assign_family_category(
                category_config,
                key,
                value,
                line_number,
            ))
        }
        _ => None,
    }
}

#[cfg(feature = "family")]
fn assign_family(
    config: &mut crate::FamilyPolicyConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        _ => return unknown_key("family", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "family")]
fn assign_family_category(
    config: &mut CategoryConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "action" => config.action = parse_category_action(value, line_number)?,
        "replacement" => config.replacement = Some(parse_string(value, line_number)?),
        _ => return unknown_key("family.categories", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "family")]
fn parse_category_action(
    value: &str,
    line_number: usize,
) -> Result<CategoryAction, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "block" => Ok(CategoryAction::Block),
        "replace" => Ok(CategoryAction::Replace),
        "defer" => Ok(CategoryAction::Defer),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid family.categories.action {other:?}; expected \"block\", \"replace\", or \"defer\""
            ),
        )),
    }
}

#[cfg(feature = "guard")]
fn assign_profile(
    config: &mut ProfileConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "kind" => config.kind = parse_profile_kind(value, line_number)?,
        "allowed_models" => config.allowed_models = parse_string_array(value, line_number)?,
        "daily_request_limit" => {
            config.daily_request_limit =
                parse_u64(value, line_number, "profiles.daily_request_limit")?;
        }
        "shielded_buffering" => {
            config.shielded_buffering = parse_shielded_buffering(value, line_number)?;
        }
        "guard_pack" => config.guard_pack = Some(parse_string(value, line_number)?),
        _ => return unknown_key("profiles", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn parse_profile_kind(value: &str, line_number: usize) -> Result<ProfileKind, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "child" => Ok(ProfileKind::Child),
        "adult" => Ok(ProfileKind::Adult),
        other => Err(ConfigParseError::new(
            line_number,
            format!("invalid profiles.kind {other:?}; expected \"child\" or \"adult\""),
        )),
    }
}

#[cfg(feature = "guard")]
fn parse_shielded_buffering(
    value: &str,
    line_number: usize,
) -> Result<ShieldedBuffering, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "off" => Ok(ShieldedBuffering::Off),
        "buffered_sse" => Ok(ShieldedBuffering::BufferedSse),
        "sanitized" => Ok(ShieldedBuffering::Sanitized),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid profiles.shielded_buffering {other:?}; expected \"off\", \"buffered_sse\", or \"sanitized\""
            ),
        )),
    }
}

#[cfg(feature = "guard")]
fn assign_workflow(
    config: &mut WorkflowConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "runtime_kind" => config.runtime_kind = parse_workflow_runtime(value, line_number)?,
        "command" => config.command = parse_string(value, line_number)?,
        "args" => config.args = parse_string_array(value, line_number)?,
        "timeout_ms" => {
            config.timeout_ms = parse_u64(value, line_number, "workflows.timeout_ms")?;
        }
        "max_stdout_bytes" => {
            config.max_stdout_bytes =
                parse_usize(value, line_number, "workflows.max_stdout_bytes")?;
        }
        _ => return unknown_key("workflows", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn assign_guard_workflows(
    config: &mut super::GuardWorkflowConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "pre_request" => config.pre_request = Some(parse_string(value, line_number)?),
        "post_response" => config.post_response = Some(parse_string(value, line_number)?),
        "fail_closed_blocks" => config.fail_closed_blocks = parse_bool(value, line_number)?,
        "max_in_flight_executions" => {
            config.max_in_flight_executions = parse_usize(
                value,
                line_number,
                "guard_workflows.max_in_flight_executions",
            )?;
        }
        _ => return unknown_key("guard_workflows", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn assign_virtual_keys(
    config: &mut VirtualKeyConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "unknown_key_policy" => {
            config.unknown_key_policy = parse_unknown_key_policy(value, line_number)?;
        }
        _ => return unknown_key("virtual_keys", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn assign_virtual_key_map(
    config: &mut VirtualKeyConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    let profile_name = parse_string(value, line_number)?;
    config
        .keys
        .insert(key.trim_matches('"').to_owned(), profile_name);
    Ok(())
}

#[cfg(feature = "guard")]
fn assign_budget(
    config: &mut super::BudgetConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "sqlite_path" => config.sqlite_path = parse_string(value, line_number)?,
        "reset_timezone" => config.reset_timezone = parse_string(value, line_number)?,
        "reset_hour_utc" => {
            config.reset_hour_utc = parse_u32(value, line_number, "budget.reset_hour_utc")?;
        }
        _ => return unknown_key("budget", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn parse_unknown_key_policy(
    value: &str,
    line_number: usize,
) -> Result<UnknownKeyPolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "use_default_profile" => Ok(UnknownKeyPolicy::UseDefaultProfile),
        "fail_closed" => Ok(UnknownKeyPolicy::FailClosed),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid virtual_keys.unknown_key_policy {other:?}; expected \"use_default_profile\" or \"fail_closed\""
            ),
        )),
    }
}

#[cfg(feature = "guard")]
fn parse_workflow_runtime(
    value: &str,
    line_number: usize,
) -> Result<WorkflowRuntime, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "stdio" => Ok(WorkflowRuntime::Stdio),
        other => Err(ConfigParseError::new(
            line_number,
            format!("invalid workflows.runtime_kind {other:?}; expected \"stdio\""),
        )),
    }
}

#[cfg(feature = "guard")]
fn assign_model_alias(
    config: &mut ModelAliasConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "id" => config.id = parse_string(value, line_number)?,
        "kind" => config.kind = parse_alias_kind(value, line_number)?,
        "upstream_profile" => config.upstream_profile = Some(parse_string(value, line_number)?),
        "workflow_id" => config.workflow_id = Some(parse_string(value, line_number)?),
        "workflow_timeout_ms" => {
            config.workflow_timeout_ms = Some(parse_u64(
                value,
                line_number,
                "model_aliases.workflow_timeout_ms",
            )?);
        }
        _ => return unknown_key("model_aliases", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "guard")]
fn parse_alias_kind(value: &str, line_number: usize) -> Result<AliasKind, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "upstream" => Ok(AliasKind::Upstream),
        "workflow" => Ok(AliasKind::Workflow),
        other => Err(ConfigParseError::new(
            line_number,
            format!("invalid model_aliases.kind {other:?}; expected \"upstream\" or \"workflow\""),
        )),
    }
}

fn assign_listener(
    config: &mut ListenerConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "name" => config.name = parse_string(value, line_number)?,
        "bind_host" => config.bind_host = parse_string(value, line_number)?,
        "port" => config.port = parse_u16(value, line_number, "listeners.port")?,
        "allowed_upstreams" => {
            config.allowed_upstreams = Some(parse_string_array(value, line_number)?);
        }
        _ => return unknown_key("listeners", key, line_number),
    }
    Ok(())
}

fn assign_upstream_profile(
    config: &mut UpstreamProfileConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "name" => config.name = parse_string(value, line_number)?,
        "model" => {
            let model = parse_string(value, line_number)?;
            config.name.clone_from(&model);
            config.match_models = vec![model];
        }
        "base_url" => config.base_url = parse_string(value, line_number)?,
        "match_models" => config.match_models = parse_string_array(value, line_number)?,
        "endpoint_selection" => {
            config.endpoint_selection = parse_endpoint_selection_mode(value, line_number)?;
        }
        "health_probe_interval" => {
            config.health_probe_interval_ms =
                parse_duration_ms(value, line_number, "profile.health_probe_interval")?;
        }
        "health_probe_timeout" => {
            config.health_probe_timeout_ms =
                parse_duration_ms(value, line_number, "profile.health_probe_timeout")?;
        }
        "health_probe_max_wait" => {
            config.health_probe_max_wait_ms =
                parse_duration_ms(value, line_number, "profile.health_probe_max_wait")?;
        }
        "health_probe_interval_ms" => {
            config.health_probe_interval_ms =
                parse_u64(value, line_number, "profile.health_probe_interval_ms")?;
        }
        "health_probe_timeout_ms" => {
            config.health_probe_timeout_ms =
                parse_u64(value, line_number, "profile.health_probe_timeout_ms")?;
        }
        "health_probe_max_wait_ms" => {
            config.health_probe_max_wait_ms =
                parse_u64(value, line_number, "profile.health_probe_max_wait_ms")?;
        }
        "request_timeout_ms" => {
            config.request_timeout_ms =
                parse_u64(value, line_number, "upstreams.request_timeout_ms")?;
        }
        "max_in_flight_requests" => {
            config.max_in_flight_requests = Some(parse_usize(
                value,
                line_number,
                "upstreams.max_in_flight_requests",
            )?);
        }
        "max_queued_generation_requests" => {
            config.max_queued_generation_requests = Some(parse_usize(
                value,
                line_number,
                "upstreams.max_queued_generation_requests",
            )?);
        }
        _ => return unknown_key("upstreams", key, line_number),
    }
    Ok(())
}

fn assign_upstream_endpoint(
    config: &mut UpstreamEndpointConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "base_url" => config.base_url = parse_string(value, line_number)?,
        "priority" => config.priority = parse_upstream_priority(value, line_number)?,
        "protocol" => config.protocol = parse_upstream_endpoint_protocol(value, line_number)?,
        "model" => config.model = parse_optional_string(value, line_number)?,
        "model_revision" => config.model_revision = parse_optional_string(value, line_number)?,
        "api_key_env" => config.api_key_env = parse_optional_string(value, line_number)?,
        _ => return unknown_key("profile.upstream", key, line_number),
    }
    Ok(())
}

fn parse_endpoint_selection_mode(
    value: &str,
    line_number: usize,
) -> Result<EndpointSelectionMode, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "priority_failover" => Ok(EndpointSelectionMode::PriorityFailover),
        "round_robin" => Ok(EndpointSelectionMode::RoundRobin),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid upstreams.endpoint_selection {other:?}; expected \"priority_failover\" or \"round_robin\""
            ),
        )),
    }
}

fn parse_upstream_endpoint_protocol(
    value: &str,
    line_number: usize,
) -> Result<UpstreamEndpointProtocol, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "openai" => Ok(UpstreamEndpointProtocol::OpenAi),
        "deepinfra_qwen3_rerank" => Ok(UpstreamEndpointProtocol::DeepInfraQwen3Rerank),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid profile.upstream.protocol {other:?}; expected \"openai\" or \"deepinfra_qwen3_rerank\""
            ),
        )),
    }
}

fn parse_upstream_priority(
    value: &str,
    line_number: usize,
) -> Result<UpstreamPriority, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "primary" => Ok(UpstreamPriority::Primary),
        "failover" | "backup" => Ok(UpstreamPriority::Failover),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid profile.upstream.priority {other:?}; expected \"primary\" or \"failover\""
            ),
        )),
    }
}

fn synchronize_primary_base_url(profile: &mut UpstreamProfileConfig) {
    if let Some(primary) = profile
        .endpoints
        .iter()
        .find(|endpoint| endpoint.priority == UpstreamPriority::Primary)
    {
        profile.base_url.clone_from(&primary.base_url);
    }
}

fn assign_server(
    config: &mut ServerConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "bind_host" => config.bind_host = parse_string(value, line_number)?,
        "port" => config.port = parse_u16(value, line_number, "server.port")?,
        "max_in_flight_requests" => {
            config.max_in_flight_requests =
                parse_usize(value, line_number, "server.max_in_flight_requests")?;
        }
        "max_queued_generation_requests" => {
            config.max_queued_generation_requests =
                parse_usize(value, line_number, "server.max_queued_generation_requests")?;
        }
        "generation_queue_timeout_ms" => {
            config.generation_queue_timeout_ms =
                parse_u64(value, line_number, "server.generation_queue_timeout_ms")?;
        }
        "generation_queue_full_status" => {
            config.generation_queue_full_status =
                parse_http_error_status(value, line_number, "server.generation_queue_full_status")?;
        }
        "generation_queue_retry_after_secs" => {
            config.generation_queue_retry_after_secs = Some(parse_u32(
                value,
                line_number,
                "server.generation_queue_retry_after_secs",
            )?);
        }
        "max_control_plane_in_flight_requests" => {
            config.max_control_plane_in_flight_requests = parse_usize(
                value,
                line_number,
                "server.max_control_plane_in_flight_requests",
            )?;
        }
        "max_request_body_bytes" => {
            config.max_request_body_bytes =
                parse_usize(value, line_number, "server.max_request_body_bytes")?;
        }
        "shutdown_drain_timeout_ms" => {
            config.shutdown_drain_timeout_ms =
                parse_u64(value, line_number, "server.shutdown_drain_timeout_ms")?;
        }
        _ => return unknown_key("server", key, line_number),
    }
    Ok(())
}

fn assign_upstream(
    config: &mut UpstreamConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "base_url" => config.base_url = parse_string(value, line_number)?,
        "request_timeout_ms" => {
            config.request_timeout_ms =
                parse_u64(value, line_number, "upstream.request_timeout_ms")?;
        }
        _ => return unknown_key("upstream", key, line_number),
    }
    Ok(())
}

fn assign_metadata(
    config: &mut MetadataConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "discovery_enabled" => config.discovery_enabled = parse_bool(value, line_number)?,
        "enrich_responses" => config.enrich_responses = parse_bool(value, line_number)?,
        "refresh_interval_secs" => {
            config.refresh_interval_secs = parse_u64(
                value,
                line_number,
                "upstream.metadata.refresh_interval_secs",
            )?;
        }
        "context_length_override" => {
            config.context_length_override = Some(parse_u32(
                value,
                line_number,
                "upstream.metadata.context_length_override",
            )?);
        }
        "max_model_len_override" => {
            config.max_model_len_override = Some(parse_u32(
                value,
                line_number,
                "upstream.metadata.max_model_len_override",
            )?);
        }
        "input_token_safety_margin" => {
            config.input_token_safety_margin = parse_u32(
                value,
                line_number,
                "upstream.metadata.input_token_safety_margin",
            )?;
        }
        _ => return unknown_key("upstream.metadata", key, line_number),
    }
    Ok(())
}

fn assign_hot_restart(
    config: &mut HotRestartConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "probe_max_tokens" => {
            config.probe_max_tokens =
                parse_u32(value, line_number, "hot_restart.probe_max_tokens")?;
        }
        "probe_interval_secs" => {
            config.probe_interval_secs =
                parse_u64(value, line_number, "hot_restart.probe_interval_secs")?;
        }
        "probe_timeout_secs" => {
            config.probe_timeout_secs =
                parse_u64(value, line_number, "hot_restart.probe_timeout_secs")?;
        }
        "probe_messages" => {
            config.probe_messages =
                parse_json_value(value, line_number, "hot_restart.probe_messages")?;
        }
        "probe_chat_template_kwargs" => {
            config.probe_chat_template_kwargs = parse_optional_json_value(
                value,
                line_number,
                "hot_restart.probe_chat_template_kwargs",
            )?;
        }
        _ => return unknown_key("hot_restart", key, line_number),
    }
    Ok(())
}

fn assign_local_recovery(
    config: &mut LocalRecoveryConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "restart_command" => {
            config.restart_command = parse_string_array(value, line_number)?;
        }
        "restart_timeout_ms" => {
            config.restart_timeout_ms =
                parse_u64(value, line_number, "local_recovery.restart_timeout_ms")?;
        }
        "readiness_endpoint" => {
            config.readiness_endpoint = parse_string(value, line_number)?;
        }
        "readiness_body" => {
            config.readiness_body =
                parse_json_value(value, line_number, "local_recovery.readiness_body")?;
        }
        "readiness_request_timeout_ms" => {
            config.readiness_request_timeout_ms = parse_u64(
                value,
                line_number,
                "local_recovery.readiness_request_timeout_ms",
            )?;
        }
        "readiness_deadline_ms" => {
            config.readiness_deadline_ms =
                parse_u64(value, line_number, "local_recovery.readiness_deadline_ms")?;
        }
        "readiness_interval_ms" => {
            config.readiness_interval_ms =
                parse_u64(value, line_number, "local_recovery.readiness_interval_ms")?;
        }
        "max_attempts_per_request" => {
            config.max_attempts_per_request = parse_u32(
                value,
                line_number,
                "local_recovery.max_attempts_per_request",
            )?;
        }
        "cooldown_ms" => {
            config.cooldown_ms = parse_u64(value, line_number, "local_recovery.cooldown_ms")?;
        }
        "budget_window_ms" => {
            config.budget_window_ms =
                parse_u64(value, line_number, "local_recovery.budget_window_ms")?;
        }
        "max_per_window" => {
            config.max_per_window = parse_u32(value, line_number, "local_recovery.max_per_window")?;
        }
        _ => return unknown_key("local_recovery", key, line_number),
    }
    Ok(())
}

fn assign_stuck_watchdog(
    config: &mut StuckWatchdogConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "detection_window_secs" => {
            config.detection_window_secs =
                parse_u64(value, line_number, "stuck_watchdog.detection_window_secs")?;
        }
        "min_output_progress_units_in_window" => {
            config.min_output_progress_units_in_window = parse_u64(
                value,
                line_number,
                "stuck_watchdog.min_output_progress_units_in_window",
            )?;
        }
        "check_interval_secs" => {
            config.check_interval_secs =
                parse_u64(value, line_number, "stuck_watchdog.check_interval_secs")?;
        }
        _ => return unknown_key("stuck_watchdog", key, line_number),
    }
    Ok(())
}

fn assign_restart_queue(
    config: &mut RestartQueueConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "queue_deadline_secs" => {
            config.queue_deadline_secs =
                parse_u64(value, line_number, "restart_queue.queue_deadline_secs")?;
        }
        "restart_timeout_secs" => {
            config.restart_timeout_secs =
                parse_u64(value, line_number, "restart_queue.restart_timeout_secs")?;
        }
        _ => return unknown_key("restart_queue", key, line_number),
    }
    Ok(())
}

#[cfg(feature = "param-override")]
fn assign_param_override(
    config: &mut ParamOverrideConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "temperature" => {
            config.temperature = Some(parse_f64(value, line_number, "param_override.temperature")?);
        }
        "top_p" => {
            config.top_p = Some(parse_f64(value, line_number, "param_override.top_p")?);
        }
        "top_k" => {
            config.top_k = Some(parse_u32(value, line_number, "param_override.top_k")?);
        }
        "max_tokens" => {
            config.max_tokens = Some(parse_u32(value, line_number, "param_override.max_tokens")?);
        }
        "frequency_penalty" => {
            config.frequency_penalty = Some(parse_f64(
                value,
                line_number,
                "param_override.frequency_penalty",
            )?);
        }
        "presence_penalty" => {
            config.presence_penalty = Some(parse_f64(
                value,
                line_number,
                "param_override.presence_penalty",
            )?);
        }
        _ => return unknown_key("upstreams.param_override", key, line_number),
    }
    Ok(())
}

fn assign_shielding(
    config: &mut ShieldingConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        _ => return unknown_key("shielding", key, line_number),
    }
    Ok(())
}

fn assign_observability(
    config: &mut ObservabilityConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "sqlite_path" => config.sqlite_path = PathBuf::from(parse_string(value, line_number)?),
        "capture_raw_payloads" => config.capture_raw_payloads = parse_bool(value, line_number)?,
        "metrics_enabled" => {
            config.metrics_enabled = ConfigToggle::from_bool(parse_bool(value, line_number)?);
        }
        "health_upstream_probe_enabled" => {
            config.health_upstream_probe_enabled =
                ConfigToggle::from_bool(parse_bool(value, line_number)?);
        }
        "health_upstream_probe_timeout_ms" => {
            config.health_upstream_probe_timeout_ms = parse_u64(
                value,
                line_number,
                "observability.health_upstream_probe_timeout_ms",
            )?;
        }
        "health_chat_probe_enabled" => {
            config.health_chat_probe_enabled =
                ConfigToggle::from_bool(parse_bool(value, line_number)?);
        }
        "health_chat_probe_timeout_ms" => {
            config.health_chat_probe_timeout_ms = parse_u64(
                value,
                line_number,
                "observability.health_chat_probe_timeout_ms",
            )?;
        }
        "debug_summary_enabled" => {
            config.debug_summary_enabled = ConfigToggle::from_bool(parse_bool(value, line_number)?);
        }
        "debug_summary_admin_token" => {
            config.debug_summary_admin_token = parse_optional_string(value, line_number)?;
        }
        "debug_summary_max_records" => {
            config.debug_summary_max_records = parse_u32(
                value,
                line_number,
                "observability.debug_summary_max_records",
            )?;
        }
        _ => return unknown_key("observability", key, line_number),
    }
    Ok(())
}

fn assign_retention(
    config: &mut RetentionConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "max_bytes" => {
            config.max_bytes = parse_u64(value, line_number, "observability.retention.max_bytes")?;
        }
        "prune_to_bytes" => {
            config.prune_to_bytes =
                parse_u64(value, line_number, "observability.retention.prune_to_bytes")?;
        }
        "max_records" => {
            config.max_records =
                parse_u64(value, line_number, "observability.retention.max_records")?;
        }
        "prune_to_records" => {
            config.prune_to_records = Some(parse_u64(
                value,
                line_number,
                "observability.retention.prune_to_records",
            )?);
        }
        _ => return unknown_key("observability.retention", key, line_number),
    }
    Ok(())
}

fn assign_evidence(
    config: &mut super::EvidenceConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "sqlite_path" => config.sqlite_path = PathBuf::from(parse_string(value, line_number)?),
        "blob_cache_dir" => {
            config.blob_cache_dir = PathBuf::from(parse_string(value, line_number)?);
        }
        "include_raw_payloads" => config.include_raw_payloads = parse_bool(value, line_number)?,
        "include_request_headers" => {
            config.include_request_headers = parse_bool(value, line_number)?;
        }
        "max_bytes" => config.max_bytes = parse_u64(value, line_number, "evidence.max_bytes")?,
        "prune_to_bytes" => {
            config.prune_to_bytes = parse_u64(value, line_number, "evidence.prune_to_bytes")?;
        }
        "max_records" => {
            config.max_records = parse_u64(value, line_number, "evidence.max_records")?;
        }
        "prune_to_records" => {
            config.prune_to_records =
                Some(parse_u64(value, line_number, "evidence.prune_to_records")?);
        }
        _ => return unknown_key("evidence", key, line_number),
    }
    Ok(())
}

fn assign_evidence_shadow(
    config: &mut super::EvidenceShadowConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "keep_looping_attempt_running" => {
            config.keep_looping_attempt_running = parse_bool(value, line_number)?;
        }
        "parallel_downgrade_attempts" => {
            config.parallel_downgrade_attempts = parse_bool(value, line_number)?;
        }
        "max_shadow_attempts_per_request" => {
            config.max_shadow_attempts_per_request = parse_u32(
                value,
                line_number,
                "evidence.shadow.max_shadow_attempts_per_request",
            )?;
        }
        "max_global_shadow_in_flight" => {
            config.max_global_shadow_in_flight = parse_usize(
                value,
                line_number,
                "evidence.shadow.max_global_shadow_in_flight",
            )?;
        }
        "shadow_attempt_timeout_ms" => {
            config.shadow_attempt_timeout_ms = parse_u64(
                value,
                line_number,
                "evidence.shadow.shadow_attempt_timeout_ms",
            )?;
        }
        "compare_attempts" => {
            config.compare_attempts = parse_shadow_compare_attempts(
                value,
                line_number,
                "evidence.shadow.compare_attempts",
            )?;
        }
        _ => return unknown_key("evidence.shadow", key, line_number),
    }
    Ok(())
}

fn assign_evidence_paired_comparison(
    config: &mut super::EvidencePairedComparisonConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "variants" => {
            config.variants = parse_shadow_compare_attempts(
                value,
                line_number,
                "evidence.shadow.paired_comparison.variants",
            )?;
        }
        "include_raw_input" => config.include_raw_input = parse_bool(value, line_number)?,
        "include_raw_output" => config.include_raw_output = parse_bool(value, line_number)?,
        "include_raw_reasoning" => config.include_raw_reasoning = parse_bool(value, line_number)?,
        "sample_rate" => {
            let rate = parse_f64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.sample_rate",
            )?;
            if !(0.0..=1.0).contains(&rate) || !rate.is_finite() {
                return Err(ConfigParseError::new(
                    line_number,
                    "validation failed for evidence.shadow.paired_comparison.sample_rate: must be a finite number between 0.0 and 1.0",
                ));
            }
            config.sample_rate = rate;
        }
        "max_raw_input_bytes" => {
            config.max_raw_input_bytes = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.max_raw_input_bytes",
            )?;
        }
        "max_raw_output_bytes" => {
            config.max_raw_output_bytes = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.max_raw_output_bytes",
            )?;
        }
        "max_raw_reasoning_bytes" => {
            config.max_raw_reasoning_bytes = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.max_raw_reasoning_bytes",
            )?;
        }
        "max_retention_records" => {
            config.max_retention_records = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.max_retention_records",
            )?;
        }
        "max_retention_bytes" => {
            config.max_retention_bytes = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.max_retention_bytes",
            )?;
        }
        "retention_days" => {
            config.retention_days = parse_u64(
                value,
                line_number,
                "evidence.shadow.paired_comparison.retention_days",
            )?;
        }
        _ => return unknown_key("evidence.shadow.paired_comparison", key, line_number),
    }
    Ok(())
}

fn parse_shadow_compare_attempts(
    value: &str,
    line_number: usize,
    field: &'static str,
) -> Result<Vec<ShadowComparisonAttempt>, ConfigParseError> {
    parse_string_array(value, line_number)?
        .into_iter()
        .map(|value| parse_shadow_compare_attempt(&value, line_number, field))
        .collect()
}

fn parse_shadow_compare_attempt(
    value: &str,
    line_number: usize,
    field: &'static str,
) -> Result<ShadowComparisonAttempt, ConfigParseError> {
    match value.trim() {
        "max-thinking" => Ok(ShadowComparisonAttempt::MaxThinking),
        "bounded-thinking" => Ok(ShadowComparisonAttempt::BoundedThinking),
        "no-thinking" => Ok(ShadowComparisonAttempt::NoThinking),
        "cot-salvage" => Ok(ShadowComparisonAttempt::CotSalvage),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid {field} entry {other:?}; expected \"max-thinking\", \"bounded-thinking\", \"no-thinking\", or \"cot-salvage\""
            ),
        )),
    }
}

fn assign_thinking(
    config: &mut ThinkingConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "mode" => config.mode = parse_thinking_mode(value, line_number)?,
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "force_disable" => config.force_disable = parse_bool(value, line_number)?,
        "max_tokens" => {
            config.max_tokens = Some(parse_u32(value, line_number, "thinking.max_tokens")?);
        }
        "budget_tokens" => {
            config.budget_tokens = parse_u32(value, line_number, "thinking.budget_tokens")?;
        }
        "thinking_token_budget" => {
            config.budget_tokens = parse_u32(value, line_number, "thinking.thinking_token_budget")?;
        }
        "preserve_answer_budget" => config.preserve_answer_budget = parse_bool(value, line_number)?,
        "budget_accounting" => {
            config.preserve_answer_budget = parse_budget_accounting(value, line_number)?;
        }
        "tool_request_policy" => {
            config.tool_request_policy = parse_tool_request_thinking_policy(value, line_number)?;
        }
        "no_thinking_marker_policy" | "thinking.no_thinking_marker_policy" => {
            config.no_thinking_marker_policy = parse_no_thinking_marker_policy(value, line_number)?;
        }
        "default_injection_schema" | "thinking.default_injection_schema" => {
            config.default_injection_schema = parse_default_injection_schema(value, line_number)?;
        }
        "apply_to_tool_requests" => {
            config.tool_request_policy = if parse_bool(value, line_number)? {
                ToolRequestThinkingPolicy::Apply
            } else {
                ToolRequestThinkingPolicy::Passthrough
            };
        }
        _ => return unknown_key("thinking", key, line_number),
    }
    Ok(())
}

fn parse_thinking_mode(value: &str, line_number: usize) -> Result<ThinkingMode, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "passthrough" => Ok(ThinkingMode::Passthrough),
        "force_disable" => Ok(ThinkingMode::ForceDisable),
        "force_thinking" => Ok(ThinkingMode::ForceThinking),
        "bounded_thinking" => Ok(ThinkingMode::BoundedThinking),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid thinking.mode {other:?}; expected \"passthrough\", \"force_disable\", \"force_thinking\", or \"bounded_thinking\""
            ),
        )),
    }
}

fn parse_default_injection_schema(
    value: &str,
    line_number: usize,
) -> Result<DefaultInjectionSchema, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "canonical" => Ok(DefaultInjectionSchema::Canonical),
        "chat_template_kwargs" => Ok(DefaultInjectionSchema::ChatTemplateKwargs),
        "vllm_native" => Ok(DefaultInjectionSchema::VllmNative),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid thinking.default_injection_schema {other:?}; expected \"canonical\", \"chat_template_kwargs\", or \"vllm_native\""
            ),
        )),
    }
}

fn parse_budget_accounting(value: &str, line_number: usize) -> Result<bool, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "preserve_answer_budget" => Ok(true),
        "total_cap" => Ok(false),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid thinking.budget_accounting {other:?}; expected \"total_cap\" or \"preserve_answer_budget\""
            ),
        )),
    }
}

fn parse_tool_request_thinking_policy(
    value: &str,
    line_number: usize,
) -> Result<ToolRequestThinkingPolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "apply" => Ok(ToolRequestThinkingPolicy::Apply),
        "passthrough" => Ok(ToolRequestThinkingPolicy::Passthrough),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid thinking.tool_request_policy {other:?}; expected \"apply\" or \"passthrough\""
            ),
        )),
    }
}

fn parse_no_thinking_marker_policy(
    value: &str,
    line_number: usize,
) -> Result<NoThinkingMarkerPolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "force" => Ok(NoThinkingMarkerPolicy::Force),
        "respect_no_thinking_markers" => Ok(NoThinkingMarkerPolicy::RespectNoThinkingMarkers),
        "escape_hatch_only" => Ok(NoThinkingMarkerPolicy::EscapeHatchOnly),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid thinking.no_thinking_marker_policy {other:?}; expected \"force\", \"respect_no_thinking_markers\", or \"escape_hatch_only\""
            ),
        )),
    }
}

fn assign_loop_guard(
    config: &mut LoopGuardConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "mode" => config.mode = parse_loop_guard_mode(value, line_number)?,
        "on_reasoning_loop" => {
            config.on_reasoning_loop = parse_loop_failure_policy(value, line_number)?;
        }
        "normalized_input_window_secs" => {
            config.normalized_input_window_secs = parse_u64(
                value,
                line_number,
                "loop_guard.normalized_input_window_secs",
            )?;
        }
        "max_repeated_inputs" => {
            config.max_repeated_inputs =
                parse_u32(value, line_number, "loop_guard.max_repeated_inputs")?;
        }
        "output_repeated_line_threshold" => {
            config.output_repeated_line_threshold = parse_u32(
                value,
                line_number,
                "loop_guard.output_repeated_line_threshold",
            )?;
        }
        "output_token_window_size" => {
            config.output_token_window_size =
                parse_u32(value, line_number, "loop_guard.output_token_window_size")?;
        }
        "output_repeated_token_window_threshold" => {
            config.output_repeated_token_window_threshold = parse_u32(
                value,
                line_number,
                "loop_guard.output_repeated_token_window_threshold",
            )?;
        }
        "output_suffix_cycle_threshold" => {
            config.output_suffix_cycle_threshold = parse_u32(
                value,
                line_number,
                "loop_guard.output_suffix_cycle_threshold",
            )?;
        }
        "output_low_progress_min_bytes" => {
            config.output_low_progress_min_bytes = parse_u64(
                value,
                line_number,
                "loop_guard.output_low_progress_min_bytes",
            )?;
        }
        "output_low_progress_unique_ratio_percent" => {
            config.output_low_progress_unique_ratio_percent = parse_u32(
                value,
                line_number,
                "loop_guard.output_low_progress_unique_ratio_percent",
            )?;
        }
        "input_overlap_threshold_multiplier" => {
            config.input_overlap_threshold_multiplier = parse_u32(
                value,
                line_number,
                "loop_guard.input_overlap_threshold_multiplier",
            )?;
        }
        "reasoning_semantic_detection_enabled" => {
            config.reasoning_semantic_detection_enabled = parse_bool(value, line_number)?;
        }
        "reasoning_semantic_similarity_threshold_percent" => {
            config.reasoning_semantic_similarity_threshold_percent = parse_u32(
                value,
                line_number,
                "loop_guard.reasoning_semantic_similarity_threshold_percent",
            )?;
        }
        "reasoning_semantic_window_token_count" => {
            config.reasoning_semantic_window_token_count = parse_u32(
                value,
                line_number,
                "loop_guard.reasoning_semantic_window_token_count",
            )?;
        }
        "reasoning_semantic_minimum_token_count" => {
            config.reasoning_semantic_minimum_token_count = parse_u32(
                value,
                line_number,
                "loop_guard.reasoning_semantic_minimum_token_count",
            )?;
        }
        "reasoning_semantic_history_window_count" => {
            config.reasoning_semantic_history_window_count = parse_u32(
                value,
                line_number,
                "loop_guard.reasoning_semantic_history_window_count",
            )?;
        }
        _ => return unknown_key("loop_guard", key, line_number),
    }
    Ok(())
}

fn assign_loop_guard_embedding(
    config: &mut super::LoopGuardEmbeddingConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "provider" => {
            config.provider = parse_embedding_provider(value, line_number)?;
        }
        "endpoint" => config.endpoint = parse_string(value, line_number)?,
        "model" => config.model = parse_string(value, line_number)?,
        "api_key" => {
            config.api_key = Some(parse_string(value, line_number)?);
        }
        "window_token_count" => {
            config.window_token_count = parse_u32(
                value,
                line_number,
                "loop_guard.embedding.window_token_count",
            )?;
        }
        "window_stride_tokens" => {
            config.window_stride_tokens = parse_u32(
                value,
                line_number,
                "loop_guard.embedding.window_stride_tokens",
            )?;
        }
        "minimum_token_count" => {
            config.minimum_token_count = parse_u32(
                value,
                line_number,
                "loop_guard.embedding.minimum_token_count",
            )?;
        }
        "history_window_count" => {
            config.history_window_count = parse_u32(
                value,
                line_number,
                "loop_guard.embedding.history_window_count",
            )?;
        }
        "batch_max_windows" => {
            config.batch_max_windows =
                parse_u32(value, line_number, "loop_guard.embedding.batch_max_windows")?;
        }
        "batch_max_wait_ms" => {
            config.batch_max_wait_ms =
                parse_u32(value, line_number, "loop_guard.embedding.batch_max_wait_ms")?;
        }
        "queue_max_windows" => {
            config.queue_max_windows =
                parse_u32(value, line_number, "loop_guard.embedding.queue_max_windows")?;
        }
        "on_queue_full" => {
            config.on_queue_full = parse_embedding_queue_policy(value, line_number)?;
        }
        "vector_dim" => {
            config.vector_dim = parse_u32(value, line_number, "loop_guard.embedding.vector_dim")?;
        }
        _ => return unknown_key("loop_guard.embedding", key, line_number),
    }
    Ok(())
}

fn parse_embedding_provider(
    value: &str,
    line_number: usize,
) -> Result<super::EmbeddingProvider, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "disabled" => Ok(super::EmbeddingProvider::Disabled),
        "openai_compatible" => Ok(super::EmbeddingProvider::OpenAiCompatible),
        "tei" => Ok(super::EmbeddingProvider::Tei),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid loop_guard.embedding.provider {other:?}; expected \"disabled\", \"openai_compatible\", or \"tei\""
            ),
        )),
    }
}

fn parse_embedding_queue_policy(
    value: &str,
    line_number: usize,
) -> Result<super::EmbeddingQueuePolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "skip" => Ok(super::EmbeddingQueuePolicy::Skip),
        "deterministic_only" => Ok(super::EmbeddingQueuePolicy::DeterministicOnly),
        "block" => Ok(super::EmbeddingQueuePolicy::Block),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid loop_guard.embedding.on_queue_full {other:?}; expected \"skip\", \"deterministic_only\", or \"block\""
            ),
        )),
    }
}

fn parse_loop_guard_mode(
    value: &str,
    line_number: usize,
) -> Result<LoopGuardMode, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "disabled" => Ok(LoopGuardMode::Disabled),
        "monitor" => Ok(LoopGuardMode::Monitor),
        "enforce" => Ok(LoopGuardMode::Enforce),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid loop_guard.mode {other:?}; expected \"disabled\", \"monitor\", or \"enforce\""
            ),
        )),
    }
}

fn parse_loop_failure_policy(
    value: &str,
    line_number: usize,
) -> Result<LoopFailurePolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "retry_ladder" => Ok(LoopFailurePolicy::RetryLadder),
        "truncate_cot_then_answer" => Ok(LoopFailurePolicy::TruncateCotThenAnswer),
        "bounded_answer_from_cot" => Ok(LoopFailurePolicy::BoundedAnswerFromCot),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid loop_guard.on_reasoning_loop {other:?}; expected \"retry_ladder\", \"truncate_cot_then_answer\", or \"bounded_answer_from_cot\""
            ),
        )),
    }
}

fn assign_retry(
    config: &mut RetryConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "max_attempts" => {
            config.max_attempts = parse_u32(value, line_number, "retry.max_attempts")?;
        }
        "request_deadline_ms" => {
            config.request_deadline_ms =
                parse_u64(value, line_number, "retry.request_deadline_ms")?;
        }
        "anti_loop_hint_enabled" => {
            config.anti_loop_hint_enabled = parse_bool(value, line_number)?;
        }
        "shielded_streaming_enabled" => {
            config.shielded_streaming_enabled = parse_bool(value, line_number)?;
        }
        "downstream_drop_policy" => {
            config.downstream_drop_policy = parse_downstream_drop_policy(value, line_number)?;
        }
        _ => return unknown_key("retry", key, line_number),
    }
    Ok(())
}

fn assign_retry_ladder(
    config: &mut RetryLadderConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "name" => config.name = parse_string(value, line_number)?,
        "mode" | "thinking_mode" => {
            config.thinking.mode = parse_thinking_mode(value, line_number)?;
        }
        "enabled" => config.thinking.enabled = parse_bool(value, line_number)?,
        "force_disable" => config.thinking.force_disable = parse_bool(value, line_number)?,
        "max_tokens" => {
            config.thinking.max_tokens =
                Some(parse_u32(value, line_number, "retry.ladder.max_tokens")?);
        }
        "budget_tokens" => {
            config.thinking.budget_tokens =
                parse_u32(value, line_number, "retry.ladder.budget_tokens")?;
        }
        "thinking_token_budget" => {
            config.thinking.budget_tokens =
                parse_u32(value, line_number, "retry.ladder.thinking_token_budget")?;
        }
        "preserve_answer_budget" => {
            config.thinking.preserve_answer_budget = parse_bool(value, line_number)?;
        }
        "budget_accounting" => {
            config.thinking.preserve_answer_budget = parse_budget_accounting(value, line_number)?;
        }
        "tool_request_policy" => {
            config.thinking.tool_request_policy =
                parse_tool_request_thinking_policy(value, line_number)?;
        }
        "no_thinking_marker_policy" | "thinking.no_thinking_marker_policy" => {
            config.thinking.no_thinking_marker_policy =
                parse_no_thinking_marker_policy(value, line_number)?;
        }
        "default_injection_schema" | "thinking.default_injection_schema" => {
            let schema = parse_default_injection_schema(value, line_number)?;
            config.thinking.default_injection_schema = schema;
            config.default_injection_schema = Some(schema);
        }
        "apply_to_tool_requests" => {
            config.thinking.tool_request_policy = if parse_bool(value, line_number)? {
                ToolRequestThinkingPolicy::Apply
            } else {
                ToolRequestThinkingPolicy::Passthrough
            };
        }
        "anti_loop_hint" => {
            config.anti_loop_hint = parse_optional_string(value, line_number)?;
        }
        _ => return unknown_key("retry.ladder", key, line_number),
    }
    Ok(())
}

fn parse_downstream_drop_policy(
    value: &str,
    line_number: usize,
) -> Result<DownstreamDropPolicy, ConfigParseError> {
    match parse_string(value, line_number)?.trim() {
        "cancel" => Ok(DownstreamDropPolicy::Cancel),
        "detach" => Ok(DownstreamDropPolicy::Detach),
        other => Err(ConfigParseError::new(
            line_number,
            format!(
                "invalid retry.downstream_drop_policy {other:?}; expected \"cancel\" or \"detach\""
            ),
        )),
    }
}

fn assign_upstream_stall(
    config: &mut UpstreamStallConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "first_chunk_timeout_ms" => {
            config.first_chunk_timeout_ms =
                parse_u64(value, line_number, "upstream.stall.first_chunk_timeout_ms")?;
        }
        "idle_timeout_ms" => {
            config.idle_timeout_ms =
                parse_u64(value, line_number, "upstream.stall.idle_timeout_ms")?;
        }
        "recovery_command" => {
            config.recovery_command = parse_string_array(value, line_number)?;
        }
        "recovery_timeout_ms" => {
            config.recovery_timeout_ms =
                parse_u64(value, line_number, "upstream.stall.recovery_timeout_ms")?;
        }
        "recovery_cooldown_ms" => {
            config.recovery_cooldown_ms =
                parse_u64(value, line_number, "upstream.stall.recovery_cooldown_ms")?;
        }
        "recovery_budget_window_ms" => {
            config.recovery_budget_window_ms = parse_u64(
                value,
                line_number,
                "upstream.stall.recovery_budget_window_ms",
            )?;
        }
        "recovery_max_per_window" => {
            config.recovery_max_per_window =
                parse_u32(value, line_number, "upstream.stall.recovery_max_per_window")?;
        }
        _ => return unknown_key("upstream.stall", key, line_number),
    }
    Ok(())
}

fn assign_heartbeat(
    config: &mut HeartbeatConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "mode" => config.mode = parse_heartbeat_mode(value, line_number)?,
        "interval_secs" => {
            config.interval_secs = parse_u64(value, line_number, "heartbeat.interval_secs")?;
        }
        _ => return unknown_key("heartbeat", key, line_number),
    }
    Ok(())
}

fn assign_cloudflare(
    config: &mut CloudflareConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        _ => return unknown_key("cloudflare", key, line_number),
    }
    Ok(())
}

fn assign_guardian(
    config: &mut GuardianConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "target_label" => config.target_label = parse_string(value, line_number)?,
        "mem_threshold_gib" => {
            config.mem_threshold_gib = parse_u64(value, line_number, "guardian.mem_threshold_gib")?;
        }
        "kill_action" => config.kill_action = parse_guardian_kill_action(value, line_number)?,
        "poll_interval_secs" => {
            config.poll_interval_secs =
                parse_u64(value, line_number, "guardian.poll_interval_secs")?;
        }
        "registration_file" => {
            config.registration_file = parse_optional_string(value, line_number)?;
        }
        "systemd_unit" => config.systemd_unit = parse_optional_string(value, line_number)?,
        "reserve_mib" => {
            config.reserve_mib = parse_u64(value, line_number, "guardian.reserve_mib")?;
        }
        "retry_interval_secs" => {
            config.retry_interval_secs =
                parse_u64(value, line_number, "guardian.retry_interval_secs")?;
        }
        "cgroup_root" => {
            config.cgroup_root = PathBuf::from(parse_string(value, line_number)?);
        }
        _ => return unknown_key("guardian", key, line_number),
    }
    Ok(())
}

fn parse_guardian_kill_action(
    value: &str,
    line_number: usize,
) -> Result<GuardianKillAction, ConfigParseError> {
    let action = parse_string(value, line_number)?;
    GuardianKillAction::from_label(&action).ok_or_else(|| {
        ConfigParseError::new(
            line_number,
            "guardian.kill_action must be cgroup.kill or systemctl_restart",
        )
    })
}

fn parse_string(value: &str, line_number: usize) -> Result<String, ConfigParseError> {
    if !value.starts_with('"') || !value.ends_with('"') || value.len() < 2 {
        return Err(ConfigParseError::new(
            line_number,
            "expected a quoted string value",
        ));
    }
    let inner = &value[1..value.len() - 1];
    let mut parsed = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            parsed.push(character);
            continue;
        }
        let Some(escaped) = chars.next() else {
            return Err(ConfigParseError::new(
                line_number,
                "string escape sequence is incomplete",
            ));
        };
        match escaped {
            '"' => parsed.push('"'),
            '\\' => parsed.push('\\'),
            'n' => parsed.push('\n'),
            'r' => parsed.push('\r'),
            't' => parsed.push('\t'),
            _ => {
                return Err(ConfigParseError::new(
                    line_number,
                    format!("unsupported string escape \\{escaped}"),
                ));
            }
        }
    }
    Ok(parsed)
}

fn parse_optional_string(
    value: &str,
    line_number: usize,
) -> Result<Option<String>, ConfigParseError> {
    let parsed = parse_string(value, line_number)?;
    Ok((!parsed.is_empty()).then_some(parsed))
}

fn parse_json_value(
    value: &str,
    line_number: usize,
    field: &str,
) -> Result<serde_json::Value, ConfigParseError> {
    serde_json::from_str(value).map_err(|error| {
        ConfigParseError::new(
            line_number,
            format!("invalid JSON value for {field}: {error}"),
        )
    })
}

fn parse_optional_json_value(
    value: &str,
    line_number: usize,
    field: &str,
) -> Result<Option<serde_json::Value>, ConfigParseError> {
    if value == "null" {
        return Ok(None);
    }
    parse_json_value(value, line_number, field).map(Some)
}

fn parse_string_array(value: &str, line_number: usize) -> Result<Vec<String>, ConfigParseError> {
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(ConfigParseError::new(
            line_number,
            "expected an array of quoted string values",
        ));
    }
    let inner = &value[1..value.len() - 1];
    let mut parsed = Vec::new();
    let mut cursor = 0;
    while cursor < inner.len() {
        let remaining = &inner[cursor..];
        let trimmed = remaining.trim_start();
        cursor += remaining.len() - trimmed.len();
        if cursor >= inner.len() {
            break;
        }
        if !inner[cursor..].starts_with('"') {
            return Err(ConfigParseError::new(
                line_number,
                "array entries must be quoted strings",
            ));
        }
        let mut escaped = false;
        let mut end = None;
        for (relative_index, character) in inner[cursor + 1..].char_indices() {
            if character == '"' && !escaped {
                end = Some(cursor + 1 + relative_index);
                break;
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
        }
        let Some(end_index) = end else {
            return Err(ConfigParseError::new(line_number, "unterminated string"));
        };
        parsed.push(parse_string(&inner[cursor..=end_index], line_number)?);
        cursor = end_index + 1;
        let remaining = inner[cursor..].trim_start();
        cursor = inner.len() - remaining.len();
        if cursor >= inner.len() {
            break;
        }
        if !inner[cursor..].starts_with(',') {
            return Err(ConfigParseError::new(
                line_number,
                "array entries must be separated by commas",
            ));
        }
        cursor += 1;
    }
    Ok(parsed)
}

fn parse_bool(value: &str, line_number: usize) -> Result<bool, ConfigParseError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ConfigParseError::new(
            line_number,
            "expected boolean value true or false",
        )),
    }
}

fn parse_heartbeat_mode(
    value: &str,
    line_number: usize,
) -> Result<HeartbeatMode, ConfigParseError> {
    let mode = parse_string(value, line_number)?;
    HeartbeatMode::from_label(&mode).ok_or_else(|| {
        ConfigParseError::new(
            line_number,
            "heartbeat.mode must be sse, json-whitespace, or disabled",
        )
    })
}

fn parse_duration_ms(
    value: &str,
    line_number: usize,
    field: &str,
) -> Result<u64, ConfigParseError> {
    let duration = parse_string(value, line_number)?;
    let (number, multiplier) = if let Some(number) = duration.strip_suffix("ms") {
        (number, 1_u64)
    } else if let Some(number) = duration.strip_suffix('s') {
        (number, 1_000_u64)
    } else if let Some(number) = duration.strip_suffix('m') {
        (number, 60_000_u64)
    } else {
        return Err(ConfigParseError::new(
            line_number,
            format!("{field} must use a duration suffix: ms, s, or m"),
        ));
    };
    let amount = parse_u64(number.trim(), line_number, field)?;
    amount.checked_mul(multiplier).ok_or_else(|| {
        ConfigParseError::new(
            line_number,
            format!("{field} is outside the supported duration range"),
        )
    })
}

fn parse_u16(value: &str, line_number: usize, field: &str) -> Result<u16, ConfigParseError> {
    let number = parse_u64(value, line_number, field)?;
    u16::try_from(number).map_err(|_error| {
        ConfigParseError::new(
            line_number,
            format!("{field} must fit in an unsigned 16-bit port"),
        )
    })
}

fn parse_http_error_status(
    value: &str,
    line_number: usize,
    field: &str,
) -> Result<u16, ConfigParseError> {
    let number = parse_u64(value, line_number, field)?;
    let status = u16::try_from(number).map_err(|_error| {
        ConfigParseError::new(
            line_number,
            format!("{field} must fit in an unsigned 16-bit HTTP status"),
        )
    })?;
    if (400..=599).contains(&status) {
        Ok(status)
    } else {
        Err(ConfigParseError::new(
            line_number,
            format!("{field} must be an HTTP error status between 400 and 599"),
        ))
    }
}

fn parse_u32(value: &str, line_number: usize, field: &str) -> Result<u32, ConfigParseError> {
    let number = parse_u64(value, line_number, field)?;
    u32::try_from(number).map_err(|_error| {
        ConfigParseError::new(
            line_number,
            format!("{field} must fit in an unsigned 32-bit integer"),
        )
    })
}

fn parse_usize(value: &str, line_number: usize, field: &str) -> Result<usize, ConfigParseError> {
    let number = parse_u64(value, line_number, field)?;
    usize::try_from(number).map_err(|_error| {
        ConfigParseError::new(
            line_number,
            format!("{field} must fit in an unsigned pointer-sized integer"),
        )
    })
}

fn parse_u64(value: &str, line_number: usize, field: &str) -> Result<u64, ConfigParseError> {
    if value.starts_with('-') || value.starts_with('+') {
        return Err(ConfigParseError::new(
            line_number,
            format!("{field} must be an unsigned integer"),
        ));
    }
    let normalized = value.replace('_', "");
    if normalized.is_empty()
        || !normalized
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        return Err(ConfigParseError::new(
            line_number,
            format!("{field} must be an unsigned integer"),
        ));
    }
    normalized.parse::<u64>().map_err(|error| {
        ConfigParseError::new(
            line_number,
            format!("{field} is outside the supported range: {error}"),
        )
    })
}

fn parse_f64(value: &str, line_number: usize, field: &str) -> Result<f64, ConfigParseError> {
    let normalized = value.replace('_', "");
    let number = normalized.parse::<f64>().map_err(|error| {
        ConfigParseError::new(line_number, format!("{field} must be a number: {error}"))
    })?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(ConfigParseError::new(
            line_number,
            format!("{field} must be a finite number"),
        ))
    }
}

fn unknown_key<T>(section: &str, key: &str, line_number: usize) -> Result<T, ConfigParseError> {
    Err(ConfigParseError::new(
        line_number,
        format!("unknown config key {section}.{key}"),
    ))
}
