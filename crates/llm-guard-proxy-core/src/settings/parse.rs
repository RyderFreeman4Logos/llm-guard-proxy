use std::path::PathBuf;

use super::{
    AppConfig, CloudflareConfig, ConfigParseError, HeartbeatConfig, HeartbeatMode, LoopGuardConfig,
    MetadataConfig, ObservabilityConfig, RetentionConfig, RetryConfig, ServerConfig,
    ShieldingConfig, ThinkingConfig, UpstreamConfig,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Section {
    Root,
    Server,
    Upstream,
    UpstreamMetadata,
    Shielding,
    Observability,
    ObservabilityRetention,
    Thinking,
    LoopGuard,
    Retry,
    Heartbeat,
    Cloudflare,
}

pub(crate) fn parse_config_text(contents: &str) -> Result<AppConfig, ConfigParseError> {
    let mut config = AppConfig::default();
    let mut section = Section::Root;

    for (index, raw_line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line_without_comment = strip_comment(raw_line, line_number)?;
        let line = line_without_comment.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            section = parse_section(line, line_number)?;
            continue;
        }
        let (key, value) = split_key_value(line, line_number)?;
        assign_value(&mut config, section, key.trim(), value.trim(), line_number)?;
    }

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

fn parse_section(line: &str, line_number: usize) -> Result<Section, ConfigParseError> {
    if !line.ends_with(']') {
        return Err(ConfigParseError::new(
            line_number,
            "section header must end with ]",
        ));
    }
    let section = &line[1..line.len() - 1];
    match section {
        "server" => Ok(Section::Server),
        "upstream" => Ok(Section::Upstream),
        "upstream.metadata" => Ok(Section::UpstreamMetadata),
        "shielding" => Ok(Section::Shielding),
        "observability" => Ok(Section::Observability),
        "observability.retention" => Ok(Section::ObservabilityRetention),
        "thinking" => Ok(Section::Thinking),
        "loop_guard" => Ok(Section::LoopGuard),
        "retry" => Ok(Section::Retry),
        "heartbeat" => Ok(Section::Heartbeat),
        "cloudflare" => Ok(Section::Cloudflare),
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
    section: Section,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match section {
        Section::Root => Err(ConfigParseError::new(
            line_number,
            "config keys must be inside a section",
        )),
        Section::Server => assign_server(&mut config.server, key, value, line_number),
        Section::Upstream => assign_upstream(&mut config.upstream, key, value, line_number),
        Section::UpstreamMetadata => {
            assign_metadata(&mut config.upstream.metadata, key, value, line_number)
        }
        Section::Shielding => assign_shielding(&mut config.shielding, key, value, line_number),
        Section::Observability => {
            assign_observability(&mut config.observability, key, value, line_number)
        }
        Section::ObservabilityRetention => {
            assign_retention(&mut config.observability.retention, key, value, line_number)
        }
        Section::Thinking => assign_thinking(&mut config.thinking, key, value, line_number),
        Section::LoopGuard => assign_loop_guard(&mut config.loop_guard, key, value, line_number),
        Section::Retry => assign_retry(&mut config.retry, key, value, line_number),
        Section::Heartbeat => assign_heartbeat(&mut config.heartbeat, key, value, line_number),
        Section::Cloudflare => assign_cloudflare(&mut config.cloudflare, key, value, line_number),
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
        _ => return unknown_key("upstream.metadata", key, line_number),
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
        _ => return unknown_key("observability.retention", key, line_number),
    }
    Ok(())
}

fn assign_thinking(
    config: &mut ThinkingConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
        "budget_tokens" => {
            config.budget_tokens = parse_u32(value, line_number, "thinking.budget_tokens")?;
        }
        "preserve_answer_budget" => config.preserve_answer_budget = parse_bool(value, line_number)?,
        _ => return unknown_key("thinking", key, line_number),
    }
    Ok(())
}

fn assign_loop_guard(
    config: &mut LoopGuardConfig,
    key: &str,
    value: &str,
    line_number: usize,
) -> Result<(), ConfigParseError> {
    match key {
        "enabled" => config.enabled = parse_bool(value, line_number)?,
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
        _ => return unknown_key("loop_guard", key, line_number),
    }
    Ok(())
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
        _ => return unknown_key("retry", key, line_number),
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
    match mode.as_str() {
        "sse" => Ok(HeartbeatMode::Sse),
        "json-whitespace" => Ok(HeartbeatMode::JsonWhitespace),
        "disabled" => Ok(HeartbeatMode::Disabled),
        _ => Err(ConfigParseError::new(
            line_number,
            "heartbeat.mode must be sse, json-whitespace, or disabled",
        )),
    }
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

fn unknown_key<T>(section: &str, key: &str, line_number: usize) -> Result<T, ConfigParseError> {
    Err(ConfigParseError::new(
        line_number,
        format!("unknown config key {section}.{key}"),
    ))
}
