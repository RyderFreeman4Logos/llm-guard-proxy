#![forbid(unsafe_code)]

pub mod config_reload;
mod embedding_backend;
mod model_judge;
mod proxy;
mod replay_calibrate;
#[cfg(feature = "guard")]
mod workflow_execution;
#[cfg(feature = "guard")]
mod workflow_process;
#[cfg(feature = "guard")]
mod workflow_runtime;

use std::{ffi::OsString, fs, future::pending, path::PathBuf, process::ExitCode, time::Duration};

use config_reload::ConfigManager;
use llm_guard_proxy_core::redact_upstream_base_url;
#[cfg(feature = "guard")]
use llm_guard_proxy_state::BudgetStore;
use llm_guard_proxy_state::{
    EvidenceRawArtifactKind, EvidenceStore, ObservabilityStore, RequestId,
};
use tokio::{net::TcpListener, sync::watch, task::JoinSet};

#[tokio::main]
async fn main() -> ExitCode {
    match run(std::env::args_os()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let args = args.into_iter().collect::<Vec<_>>();
    if args.get(1).and_then(|arg| arg.to_str()) == Some("evidence") {
        return run_evidence_command(parse_evidence_command(&args[2..])?);
    }
    let config_path = parse_config_path(args)?;
    run_server(config_path).await
}

async fn run_server(config_path: Option<PathBuf>) -> Result<(), String> {
    let manager = match config_path {
        Some(path) => ConfigManager::from_explicit_path(path),
        None => ConfigManager::from_default_path(),
    }
    .map_err(|error| error.to_string())?;
    let config = manager
        .handle()
        .snapshot()
        .map_err(|error| error.to_string())?;
    let store = ObservabilityStore::open(manager.handle()).map_err(|error| error.to_string())?;
    let evidence_store = EvidenceStore::open(manager.handle());
    #[cfg(feature = "guard")]
    let budget_store = std::sync::Arc::new(
        BudgetStore::open(&config.budget.sqlite_path).map_err(|error| error.to_string())?,
    );
    let _watcher = manager
        .spawn_polling(Duration::from_secs(1))
        .map_err(|error| error.to_string())?;
    let mut bound_listeners = Vec::new();
    for listener_config in config.effective_listeners() {
        let bind_address = listener_config.bind_address();
        let listener = TcpListener::bind(&bind_address)
            .await
            .map_err(|error| format!("failed to bind {bind_address}: {error}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|error| format!("failed to read listener address: {error}"))?;
        bound_listeners.push((listener_config, listener, local_addr));
    }
    let request_id = RequestId::generate();
    eprintln!(
        "{}",
        proxy::render_health(&config, manager.path(), &request_id)
    );
    for (listener_config, _listener, local_addr) in &bound_listeners {
        eprintln!(
            "{}",
            render_listening(
                listener_config.name.as_str(),
                local_addr,
                &config.upstream.base_url
            )
        );
    }

    let state = proxy::ProxyState::new(
        manager.handle(),
        manager.path().to_path_buf(),
        store,
        evidence_store,
        #[cfg(feature = "guard")]
        budget_store,
        proxy::build_http_client().map_err(|error| error.to_string())?,
    );
    serve_bound_listeners(bound_listeners, state).await
}

#[derive(Debug, Eq, PartialEq)]
enum EvidenceCommand {
    Status {
        db_path: Option<PathBuf>,
        config_path: Option<PathBuf>,
    },
    Summary {
        db_path: PathBuf,
    },
    ExportPairs {
        db_path: PathBuf,
        variants: Vec<String>,
        include: Vec<EvidenceRawArtifactKind>,
        output_path: PathBuf,
    },
}

fn run_evidence_command(command: EvidenceCommand) -> Result<(), String> {
    match command {
        EvidenceCommand::Status {
            db_path,
            config_path,
        } => {
            let db_path = match db_path {
                Some(path) => path,
                None => configured_evidence_db_path(config_path)?,
            };
            let status = EvidenceStore::database_status(&db_path)
                .map_err(|error| format!("failed to read evidence database status: {error}"))?;
            println!("db={}", db_path.display());
            println!("exists={}", status.exists);
            println!(
                "schema_version={}",
                status
                    .schema_version
                    .map_or_else(|| String::from("none"), |value| value.to_string())
            );
            println!(
                "supports_raw_paired_comparison={}",
                status.supports_raw_paired_comparison
            );
            println!("has_attempt_raw_columns={}", status.has_attempt_raw_columns);
            println!("has_raw_artifact_table={}", status.has_raw_artifact_table);
            Ok(())
        }
        EvidenceCommand::Summary { db_path } => {
            let rows = EvidenceStore::summary(&db_path)
                .map_err(|error| format!("failed to summarize evidence database: {error}"))?;
            println!("role\tvariant\tartifact_kind\tartifacts\tcontent_present\tbytes_stored");
            for row in rows {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    row.role,
                    row.variant_name,
                    row.artifact_kind,
                    row.artifact_count,
                    row.content_present_count,
                    row.bytes_stored
                );
            }
            Ok(())
        }
        EvidenceCommand::ExportPairs {
            db_path,
            variants,
            include,
            output_path,
        } => {
            let pairs = EvidenceStore::export_pairs(&db_path, &variants, &include)
                .map_err(|error| format!("failed to export paired evidence: {error}"))?;
            let mut output = String::new();
            for pair in &pairs {
                output.push_str(
                    &serde_json::to_string(pair)
                        .map_err(|error| format!("failed to encode JSONL pair: {error}"))?,
                );
                output.push('\n');
            }
            fs::write(&output_path, output).map_err(|error| {
                format!(
                    "failed to write paired evidence export {}: {error}",
                    output_path.display()
                )
            })?;
            println!(
                "exported_pairs={} output={}",
                pairs.len(),
                output_path.display()
            );
            Ok(())
        }
    }
}

fn configured_evidence_db_path(config_path: Option<PathBuf>) -> Result<PathBuf, String> {
    let manager = match config_path {
        Some(path) => ConfigManager::from_explicit_path(path),
        None => ConfigManager::from_default_path(),
    }
    .map_err(|error| error.to_string())?;
    let config = manager
        .handle()
        .snapshot()
        .map_err(|error| error.to_string())?;
    Ok(config.evidence.sqlite_path)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("failed to install Ctrl-C handler: {error}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                eprintln!("failed to install SIGTERM handler: {error}");
                pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

async fn serve_bound_listeners(
    bound_listeners: Vec<(
        llm_guard_proxy_core::ListenerConfig,
        TcpListener,
        std::net::SocketAddr,
    )>,
    state: proxy::ProxyState,
) -> Result<(), String> {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ignored = shutdown_tx.send(true);
    });

    let mut servers = JoinSet::new();
    for (listener_config, listener, _local_addr) in bound_listeners {
        let listener_state = state.for_listener(listener_config);
        let mut shutdown_rx = shutdown_rx.clone();
        servers.spawn(async move {
            proxy::serve_until_shutdown(listener, listener_state, async move {
                if !*shutdown_rx.borrow() {
                    let _ignored = shutdown_rx.changed().await;
                }
            })
            .await
        });
    }

    while let Some(result) = servers.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(format!("server failed: {error}")),
            Err(error) => return Err(format!("server task failed: {error}")),
        }
    }

    Ok(())
}

fn render_listening(
    listener_name: &str,
    local_addr: impl std::fmt::Display,
    upstream_base_url: &str,
) -> String {
    format!(
        "llm-guard-proxy listener={listener_name} listening={local_addr} upstream_base_url={}",
        redact_upstream_base_url(upstream_base_url)
    )
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

fn parse_evidence_command(args: &[OsString]) -> Result<EvidenceCommand, String> {
    let Some(command) = args.first().and_then(|arg| arg.to_str()) else {
        return Err(String::from("evidence requires a subcommand"));
    };
    match command {
        "status" => parse_evidence_status_command(&args[1..]),
        "summary" => parse_evidence_summary_command(&args[1..]),
        "export-pairs" => parse_evidence_export_pairs_command(&args[1..]),
        other => Err(format!("unknown evidence subcommand: {other}")),
    }
}

fn parse_evidence_status_command(args: &[OsString]) -> Result<EvidenceCommand, String> {
    let mut db_path = None;
    let mut config_path = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--db" {
            index = index.saturating_add(1);
            db_path = Some(required_path_arg(args, index, "--db")?);
        } else if let Some(value) = arg.to_str().and_then(|value| value.strip_prefix("--db=")) {
            db_path = Some(nonempty_path_value(value, "--db")?);
        } else if arg == "--config" {
            index = index.saturating_add(1);
            config_path = Some(required_path_arg(args, index, "--config")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
        {
            config_path = Some(nonempty_path_value(value, "--config")?);
        } else {
            return Err(format!(
                "unknown evidence status argument: {}",
                arg.to_string_lossy()
            ));
        }
        index = index.saturating_add(1);
    }
    Ok(EvidenceCommand::Status {
        db_path,
        config_path,
    })
}

fn parse_evidence_summary_command(args: &[OsString]) -> Result<EvidenceCommand, String> {
    let mut db_path = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--db" {
            index = index.saturating_add(1);
            db_path = Some(required_path_arg(args, index, "--db")?);
        } else if let Some(value) = arg.to_str().and_then(|value| value.strip_prefix("--db=")) {
            db_path = Some(nonempty_path_value(value, "--db")?);
        } else {
            return Err(format!(
                "unknown evidence summary argument: {}",
                arg.to_string_lossy()
            ));
        }
        index = index.saturating_add(1);
    }
    Ok(EvidenceCommand::Summary {
        db_path: db_path.ok_or_else(|| String::from("evidence summary requires --db"))?,
    })
}

fn parse_evidence_export_pairs_command(args: &[OsString]) -> Result<EvidenceCommand, String> {
    let mut db_path = None;
    let mut variants = None;
    let mut include = None;
    let mut output_path = None;
    let mut format = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--db" {
            index = index.saturating_add(1);
            db_path = Some(required_path_arg(args, index, "--db")?);
        } else if let Some(value) = arg.to_str().and_then(|value| value.strip_prefix("--db=")) {
            db_path = Some(nonempty_path_value(value, "--db")?);
        } else if arg == "--variants" {
            index = index.saturating_add(1);
            variants = Some(parse_nonempty_csv(required_str_arg(
                args,
                index,
                "--variants",
            )?));
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--variants="))
        {
            variants = Some(parse_nonempty_csv(value));
        } else if arg == "--include" {
            index = index.saturating_add(1);
            include = Some(parse_artifact_kinds(required_str_arg(
                args,
                index,
                "--include",
            )?)?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--include="))
        {
            include = Some(parse_artifact_kinds(value)?);
        } else if arg == "--format" {
            index = index.saturating_add(1);
            format = Some(required_string_arg(args, index, "--format")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--format="))
        {
            if value.is_empty() {
                return Err(String::from("--format requires a value"));
            }
            format = Some(value.to_owned());
        } else if arg == "--output" {
            index = index.saturating_add(1);
            output_path = Some(required_path_arg(args, index, "--output")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--output="))
        {
            output_path = Some(nonempty_path_value(value, "--output")?);
        } else {
            return Err(format!(
                "unknown evidence export-pairs argument: {}",
                arg.to_string_lossy()
            ));
        }
        index = index.saturating_add(1);
    }
    if format.as_deref() != Some("jsonl") {
        return Err(String::from(
            "evidence export-pairs requires --format jsonl",
        ));
    }
    let variants =
        variants.ok_or_else(|| String::from("evidence export-pairs requires --variants"))?;
    if variants.is_empty() {
        return Err(String::from(
            "evidence export-pairs requires at least one variant",
        ));
    }
    let include =
        include.ok_or_else(|| String::from("evidence export-pairs requires --include"))?;
    if include.is_empty() {
        return Err(String::from(
            "evidence export-pairs requires at least one artifact kind",
        ));
    }
    Ok(EvidenceCommand::ExportPairs {
        db_path: db_path.ok_or_else(|| String::from("evidence export-pairs requires --db"))?,
        variants,
        include,
        output_path: output_path
            .ok_or_else(|| String::from("evidence export-pairs requires --output"))?,
    })
}

fn required_path_arg(args: &[OsString], index: usize, flag: &str) -> Result<PathBuf, String> {
    let Some(value) = args.get(index) else {
        return Err(format!("{flag} requires a path"));
    };
    if value.is_empty() {
        return Err(format!("{flag} requires a path"));
    }
    Ok(PathBuf::from(value))
}

fn required_str_arg<'value>(
    args: &'value [OsString],
    index: usize,
    flag: &str,
) -> Result<&'value str, String> {
    let Some(value) = args.get(index).and_then(|value| value.to_str()) else {
        return Err(format!("{flag} requires a valid UTF-8 value"));
    };
    if value.is_empty() {
        return Err(format!("{flag} requires a value"));
    }
    Ok(value)
}

fn required_string_arg(args: &[OsString], index: usize, flag: &str) -> Result<String, String> {
    required_str_arg(args, index, flag).map(str::to_owned)
}

fn nonempty_path_value(value: &str, flag: &str) -> Result<PathBuf, String> {
    if value.is_empty() {
        return Err(format!("{flag} requires a path"));
    }
    Ok(PathBuf::from(value))
}

fn parse_nonempty_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_artifact_kinds(value: &str) -> Result<Vec<EvidenceRawArtifactKind>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(parse_artifact_kind)
        .collect()
}

fn parse_artifact_kind(value: &str) -> Result<EvidenceRawArtifactKind, String> {
    match value {
        "input" => Ok(EvidenceRawArtifactKind::Input),
        "output" => Ok(EvidenceRawArtifactKind::Output),
        "reasoning" => Ok(EvidenceRawArtifactKind::Reasoning),
        "tool_calls" | "tool-calls" => Ok(EvidenceRawArtifactKind::ToolCalls),
        other => Err(format!("unknown evidence artifact kind: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, path::Path};

    use llm_guard_proxy_core::{AppConfig, HeartbeatMode};
    use llm_guard_proxy_state::RequestId;

    use super::{
        EvidenceCommand, parse_config_path, parse_evidence_command, proxy::render_health,
        render_listening,
    };

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

    #[test]
    fn parses_evidence_export_pairs_command() {
        let args = [
            OsString::from("export-pairs"),
            OsString::from("--db"),
            OsString::from("evidence.sqlite3"),
            OsString::from("--variants"),
            OsString::from("max-thinking,no-thinking"),
            OsString::from("--include"),
            OsString::from("input,output"),
            OsString::from("--format"),
            OsString::from("jsonl"),
            OsString::from("--output"),
            OsString::from("pairs.jsonl"),
        ];

        let EvidenceCommand::ExportPairs {
            db_path,
            variants,
            include,
            output_path,
        } = parse_evidence_command(&args).expect("evidence export args should parse")
        else {
            panic!("expected export-pairs command");
        };

        assert_eq!(db_path, Path::new("evidence.sqlite3"));
        assert_eq!(variants, ["max-thinking", "no-thinking"]);
        assert_eq!(include.len(), 2);
        assert_eq!(output_path, Path::new("pairs.jsonl"));
    }

    #[test]
    fn renders_listening_with_redacted_upstream_base_url() {
        let rendered = render_listening(
            "default",
            "127.0.0.1:18009",
            "https://user:secret@example.test/v1?x-api-key=sk-test&safe=ok#token=sk-test",
        );

        assert!(
            rendered
                .contains("upstream_base_url=https://redacted:redacted@example.test/v1?redacted")
        );
        assert!(!rendered.contains("user"));
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("sk-test"));
        assert!(!rendered.contains("x-api-key"));
        assert!(!rendered.contains("safe=ok"));
        assert!(!rendered.contains("token=sk-test"));
    }
}
