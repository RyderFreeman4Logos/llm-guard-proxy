#![forbid(unsafe_code)]

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::ExitCode,
};

use llm_guard_proxy_core::{AppConfig, ConfigManager, Health, LICENSE, RequestId, SERVICE_NAME};

fn main() -> ExitCode {
    match run(std::env::args_os()) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: impl IntoIterator<Item = OsString>) -> Result<String, String> {
    let config_path = parse_config_path(args)?;
    let manager = match config_path {
        Some(path) => ConfigManager::from_explicit_path(path),
        None => ConfigManager::from_default_path(),
    }
    .map_err(|error| error.to_string())?;
    let config = manager
        .handle()
        .snapshot()
        .map_err(|error| error.to_string())?;
    let request_id = RequestId::generate();
    Ok(render_health(&config, manager.path(), &request_id))
}

fn parse_config_path(args: impl IntoIterator<Item = OsString>) -> Result<Option<PathBuf>, String> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut config_path = None;

    while let Some(arg) = args.next() {
        if arg == "--config" {
            let Some(path) = args.next() else {
                return Err(String::from("--config requires a path"));
            };
            config_path = Some(PathBuf::from(path));
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
        {
            if value.is_empty() {
                return Err(String::from("--config requires a path"));
            }
            config_path = Some(PathBuf::from(value));
        } else {
            return Err(format!("unknown argument: {}", arg.to_string_lossy()));
        }
    }

    Ok(config_path)
}

#[must_use]
fn render_health(config: &AppConfig, path: &Path, request_id: &RequestId) -> String {
    let health = Health::current();
    let name = SERVICE_NAME;
    let license = LICENSE;
    let readiness = health.readiness().as_str();
    let config_path = path.display();
    let heartbeat_mode = config.heartbeat.mode.as_str();
    let heartbeat_interval_secs = config.heartbeat.interval_secs;
    let observability_enabled = config.observability.enabled;

    format!(
        "{name} request_id={request_id} readiness={readiness} license={license} config_path={config_path} heartbeat_mode={heartbeat_mode} heartbeat_interval_secs={heartbeat_interval_secs} observability_enabled={observability_enabled}"
    )
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, path::Path};

    use llm_guard_proxy_core::{AppConfig, HeartbeatMode, RequestId};

    use super::{parse_config_path, render_health};

    #[test]
    fn renders_health_with_config_summary() {
        let mut config = AppConfig::default();
        config.heartbeat.mode = HeartbeatMode::JsonWhitespace;
        config.heartbeat.interval_secs = 7;
        config.observability.enabled = false;

        let request_id =
            RequestId::from_string("req-health").expect("test request id should be valid");

        assert_eq!(
            render_health(&config, Path::new("/tmp/config.toml"), &request_id),
            "llm-guard-proxy request_id=req-health readiness=ready license=Apache-2.0 config_path=/tmp/config.toml heartbeat_mode=json-whitespace heartbeat_interval_secs=7 observability_enabled=false"
        );
    }

    #[test]
    fn parses_explicit_config_argument() {
        let args = [
            OsString::from("llm-guard-proxy"),
            OsString::from("--config"),
            OsString::from("dev.toml"),
        ];
        assert_eq!(
            parse_config_path(args).expect("args should parse"),
            Some("dev.toml".into()),
        );
    }
}
