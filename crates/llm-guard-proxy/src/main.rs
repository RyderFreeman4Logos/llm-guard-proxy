#![deny(unsafe_code)]

pub mod config_reload;
mod embedding_backend;
mod model_judge;
mod proxy;
mod replay_calibrate;
#[cfg(all(feature = "guard", target_os = "linux"))]
#[allow(unsafe_code)]
mod workflow_cgroup;
#[cfg(feature = "guard")]
mod workflow_execution;
#[cfg(feature = "guard")]
mod workflow_process;
#[cfg(feature = "guard")]
mod workflow_runtime;

use std::{ffi::OsString, fs, future::pending, path::PathBuf, process::ExitCode, time::Duration};

use config_reload::ConfigManager;
use llm_guard_proxy_core::{AppConfig, ConfigHandle, redact_upstream_base_url};
#[cfg(feature = "memory-guardian")]
use llm_guard_proxy_host_guardian::{MemoryGuardian, default_runtime_dir};
#[cfg(feature = "host-telemetry")]
use llm_guard_proxy_host_telemetry::HostTelemetry;
#[cfg(feature = "guard")]
use llm_guard_proxy_state::BudgetStore;
use llm_guard_proxy_state::{
    EvidenceRawArtifactKind, EvidenceStore, ObservabilityStore, RequestId,
};
use tokio::{net::TcpListener, sync::watch, task::JoinSet};

const WATCHDOG_SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

fn main() -> ExitCode {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("failed to construct Tokio runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    let outcome = runtime.block_on(run(std::env::args_os()));
    shutdown_runtime(runtime, outcome.runtime_shutdown_timeout);
    match outcome.result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

struct RunOutcome {
    result: Result<(), String>,
    runtime_shutdown_timeout: Duration,
}

impl RunOutcome {
    fn with_default_timeout(result: Result<(), String>) -> Self {
        Self {
            result,
            runtime_shutdown_timeout: default_runtime_shutdown_timeout(),
        }
    }
}

async fn run(args: impl IntoIterator<Item = OsString>) -> RunOutcome {
    let args = args.into_iter().collect::<Vec<_>>();
    if args.get(1).and_then(|arg| arg.to_str()) == Some("evidence") {
        return RunOutcome::with_default_timeout(
            parse_evidence_command(&args[2..]).and_then(run_evidence_command),
        );
    }
    #[cfg(feature = "host-telemetry")]
    if args.get(1).and_then(|arg| arg.to_str()) == Some("telemetry") {
        let result = match parse_telemetry_command(&args[2..]) {
            Ok(command) => run_telemetry_command(command).await,
            Err(error) => Err(error),
        };
        return RunOutcome::with_default_timeout(result);
    }
    #[cfg(feature = "memory-guardian")]
    if args.get(1).and_then(|arg| arg.to_str()) == Some("guardian") {
        let result = match parse_guardian_command(&args[2..]) {
            Ok(command) => run_guardian_command(command).await,
            Err(error) => Err(error),
        };
        return RunOutcome::with_default_timeout(result);
    }
    let options = match parse_proxy_options(args) {
        Ok(options) => options,
        Err(error) => return RunOutcome::with_default_timeout(Err(error)),
    };
    run_server(options).await
}

fn default_runtime_shutdown_timeout() -> Duration {
    Duration::from_millis(AppConfig::default().server.shutdown_drain_timeout_ms)
}

fn current_runtime_shutdown_timeout(config: &ConfigHandle) -> Duration {
    let timeout_ms = config.snapshot().map_or_else(
        |_error| AppConfig::default().server.shutdown_drain_timeout_ms,
        |snapshot| snapshot.server.shutdown_drain_timeout_ms,
    );
    Duration::from_millis(timeout_ms)
}

fn shutdown_runtime(runtime: tokio::runtime::Runtime, shutdown_timeout: Duration) {
    runtime.shutdown_timeout(shutdown_timeout);
}

async fn run_server(options: ProxyOptions) -> RunOutcome {
    let manager = match options.config_path.clone() {
        Some(path) => ConfigManager::from_explicit_path(path),
        None => ConfigManager::from_default_path(),
    }
    .map_err(|error| error.to_string());
    let manager = match manager {
        Ok(manager) => manager,
        Err(error) => return RunOutcome::with_default_timeout(Err(error)),
    };
    let config = manager
        .handle()
        .snapshot()
        .map_err(|error| error.to_string());
    let config = match config {
        Ok(config) => config,
        Err(error) => return RunOutcome::with_default_timeout(Err(error)),
    };
    let runtime_config = manager.handle();
    let result = run_server_with_config(options, manager, config).await;
    RunOutcome {
        result,
        runtime_shutdown_timeout: current_runtime_shutdown_timeout(&runtime_config),
    }
}

async fn run_server_with_config(
    options: ProxyOptions,
    manager: ConfigManager,
    config: AppConfig,
) -> Result<(), String> {
    #[cfg(not(feature = "memory-guardian"))]
    let _ = &options;
    let store = ObservabilityStore::open(manager.handle()).map_err(|error| error.to_string())?;
    let evidence_store = EvidenceStore::open(manager.handle());
    #[cfg(feature = "memory-guardian")]
    let guardian = MemoryGuardian::open(
        manager.handle(),
        options
            .guardian_runtime_dir
            .unwrap_or_else(default_runtime_dir),
    )
    .map_err(|error| error.to_string())?;
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
    #[cfg(feature = "memory-guardian")]
    return serve_with_guardian(bound_listeners, state, guardian).await;
    #[cfg(not(feature = "memory-guardian"))]
    serve_bound_listeners(bound_listeners, state).await
}

#[cfg(feature = "host-telemetry")]
async fn run_telemetry_command(command: TelemetryCommand) -> Result<(), String> {
    let mut telemetry =
        HostTelemetry::open(command.config_path).map_err(|error| error.to_string())?;
    telemetry
        .run_until(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
}

#[cfg(feature = "memory-guardian")]
async fn run_guardian_command(command: GuardianCommand) -> Result<(), String> {
    let manager = ConfigManager::from_explicit_path(command.config_path)
        .map_err(|error| error.to_string())?;
    let _watcher = manager
        .spawn_polling(Duration::from_secs(1))
        .map_err(|error| error.to_string())?;
    let mut guardian = MemoryGuardian::open(manager.handle(), command.runtime_dir)
        .map_err(|error| error.to_string())?;
    guardian
        .run_until(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
}

#[cfg(feature = "memory-guardian")]
async fn serve_with_guardian(
    bound_listeners: Vec<(
        llm_guard_proxy_core::ListenerConfig,
        TcpListener,
        std::net::SocketAddr,
    )>,
    state: proxy::ProxyState,
    guardian: MemoryGuardian,
) -> Result<(), String> {
    let cleanup_state = state.clone();
    let mut server = tokio::spawn(serve_bound_listeners(bound_listeners, state));
    let mut guardian = tokio::spawn(async move {
        let mut guardian = guardian;
        guardian.run_until(shutdown_signal()).await
    });
    let result = tokio::select! {
        server_result = &mut server => match server_result {
            Ok(result) => result,
            Err(error) => Err(format!("server task failed: {error}")),
        },
        guardian_result = &mut guardian => match guardian_result {
            Ok(Ok(())) => match (&mut server).await {
                Ok(result) => result,
                Err(error) => Err(format!("server task failed: {error}")),
            },
            Ok(Err(error)) => {
                abort_server_after_guardian_failure(
                    &mut server,
                    || cleanup_state.begin_shutdown(),
                    cleanup_state.flush_persistence(),
                )
                .await;
                Err(format!("memory guardian failed: {error}"))
            }
            Err(error) => {
                abort_server_after_guardian_failure(
                    &mut server,
                    || cleanup_state.begin_shutdown(),
                    cleanup_state.flush_persistence(),
                )
                .await;
                Err(format!("memory guardian task failed: {error}"))
            }
        },
    };
    guardian.abort();
    result
}

#[cfg(feature = "memory-guardian")]
async fn abort_server_after_guardian_failure(
    server: &mut tokio::task::JoinHandle<Result<(), String>>,
    begin_shutdown: impl FnOnce(),
    flush_persistence: impl std::future::Future<Output = ()>,
) {
    // Signal shared ShutdownGate so serve_bound_listeners can stop each
    // nested listener and drain its JoinSet. Do not abort the outer task:
    // aborting would drop the JoinSet without joining children and race
    // flush_persistence against response-observer cleanup.
    begin_shutdown();
    let _ignored = server.await;
    flush_persistence.await;
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
    let stuck_engine_watchdog = proxy::spawn_stuck_engine_watchdog(&state);

    let mut servers = JoinSet::new();
    for (listener_config, listener, _local_addr) in bound_listeners {
        let listener_state = state.for_listener(listener_config);
        // Clone so the shutdown future can observe begin_shutdown() from guardian
        // failure (or other external paths) without aborting this outer JoinSet owner.
        let shutdown_state = listener_state.clone();
        let mut shutdown_rx = shutdown_rx.clone();
        servers.spawn(async move {
            proxy::serve_until_shutdown(listener, listener_state, async move {
                tokio::select! {
                    () = async {
                        if !*shutdown_rx.borrow() {
                            let _ignored = shutdown_rx.changed().await;
                        }
                    } => {}
                    () = shutdown_state.wait_for_shutdown() => {}
                }
            })
            .await
        });
    }

    let result = loop {
        let Some(result) = servers.join_next().await else {
            break Ok(());
        };
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => break Err(format!("server failed: {error}")),
            Err(error) => break Err(format!("server task failed: {error}")),
        }
    };
    if result.is_err() {
        // Signal siblings via ShutdownGate so serve_until_shutdown can run its
        // bounded graceful drain (and schedule observer persistence) before flush.
        // Do not abort_all(): that cancels the drain path mid-flight.
        state.begin_shutdown();
        while servers.join_next().await.is_some() {}
    }
    state.begin_shutdown();
    join_stuck_engine_watchdog(stuck_engine_watchdog, WATCHDOG_SHUTDOWN_JOIN_TIMEOUT).await;
    state.flush_persistence().await;
    result
}

async fn join_stuck_engine_watchdog(mut watchdog: tokio::task::JoinHandle<()>, max_wait: Duration) {
    if tokio::time::timeout(max_wait, &mut watchdog).await.is_err() {
        watchdog.abort();
        let _watchdog_result = watchdog.await;
    }
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

#[derive(Debug, Eq, PartialEq)]
struct ProxyOptions {
    config_path: Option<PathBuf>,
    #[cfg(feature = "memory-guardian")]
    guardian_runtime_dir: Option<PathBuf>,
}

#[cfg(feature = "memory-guardian")]
#[derive(Debug, Eq, PartialEq)]
struct GuardianCommand {
    config_path: PathBuf,
    runtime_dir: PathBuf,
}

#[cfg(feature = "host-telemetry")]
#[derive(Debug, Eq, PartialEq)]
struct TelemetryCommand {
    config_path: PathBuf,
}

fn parse_proxy_options(args: impl IntoIterator<Item = OsString>) -> Result<ProxyOptions, String> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut config_path = None;
    #[cfg(feature = "memory-guardian")]
    let mut guardian_runtime_dir = None;

    while let Some(arg) = args.next() {
        if arg == "--config" {
            let Some(path) = args.next() else {
                return Err(String::from("--config requires a path"));
            };
            config_path = Some(PathBuf::from(path));
            continue;
        }
        if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
        {
            if value.is_empty() {
                return Err(String::from("--config requires a path"));
            }
            config_path = Some(PathBuf::from(value));
            continue;
        }
        #[cfg(feature = "memory-guardian")]
        if arg == "--guardian-config"
            || arg
                .to_str()
                .is_some_and(|value| value.starts_with("--guardian-config="))
        {
            return Err(String::from(
                "--guardian-config is no longer supported; configure [guardian] in the shared --config file",
            ));
        }
        #[cfg(feature = "memory-guardian")]
        if arg == "--guardian-runtime-dir" {
            let Some(path) = args.next() else {
                return Err(String::from("--guardian-runtime-dir requires a path"));
            };
            guardian_runtime_dir = Some(PathBuf::from(path));
            continue;
        }
        #[cfg(feature = "memory-guardian")]
        if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--guardian-runtime-dir="))
        {
            if value.is_empty() {
                return Err(String::from("--guardian-runtime-dir requires a path"));
            }
            guardian_runtime_dir = Some(PathBuf::from(value));
            continue;
        }
        return Err(format!("unknown argument: {}", arg.to_string_lossy()));
    }

    Ok(ProxyOptions {
        config_path,
        #[cfg(feature = "memory-guardian")]
        guardian_runtime_dir,
    })
}

#[cfg(feature = "host-telemetry")]
fn parse_telemetry_command(args: &[OsString]) -> Result<TelemetryCommand, String> {
    let mut config_path = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--config" {
            index = index.saturating_add(1);
            config_path = Some(required_path_arg(args, index, "--config")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
        {
            config_path = Some(nonempty_path_value(value, "--config")?);
        } else {
            return Err(format!(
                "unknown telemetry argument: {}",
                arg.to_string_lossy()
            ));
        }
        index = index.saturating_add(1);
    }
    Ok(TelemetryCommand {
        config_path: config_path.ok_or_else(|| String::from("telemetry requires --config"))?,
    })
}

#[cfg(feature = "memory-guardian")]
fn parse_guardian_command(args: &[OsString]) -> Result<GuardianCommand, String> {
    let mut config_path = None;
    let mut runtime_dir = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--config" {
            index = index.saturating_add(1);
            config_path = Some(required_path_arg(args, index, "--config")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
        {
            config_path = Some(nonempty_path_value(value, "--config")?);
        } else if arg == "--runtime-dir" {
            index = index.saturating_add(1);
            runtime_dir = Some(required_path_arg(args, index, "--runtime-dir")?);
        } else if let Some(value) = arg
            .to_str()
            .and_then(|value| value.strip_prefix("--runtime-dir="))
        {
            runtime_dir = Some(nonempty_path_value(value, "--runtime-dir")?);
        } else {
            return Err(format!(
                "unknown guardian argument: {}",
                arg.to_string_lossy()
            ));
        }
        index = index.saturating_add(1);
    }
    Ok(GuardianCommand {
        config_path: config_path.ok_or_else(|| String::from("guardian requires --config"))?,
        runtime_dir: runtime_dir.ok_or_else(|| String::from("guardian requires --runtime-dir"))?,
    })
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
    use std::{ffi::OsString, path::Path, sync::mpsc, thread, time::Duration};

    #[cfg(feature = "memory-guardian")]
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    #[cfg(feature = "memory-guardian")]
    use tokio::{sync::watch, task::JoinSet};

    use llm_guard_proxy_core::{AppConfig, HeartbeatMode};

    #[cfg(feature = "memory-guardian")]
    struct PersistenceOnDrop {
        graceful_shutdown: Arc<AtomicBool>,
        enqueue: Option<tokio::sync::oneshot::Sender<bool>>,
    }

    #[cfg(feature = "memory-guardian")]
    impl Drop for PersistenceOnDrop {
        fn drop(&mut self) {
            if let Some(enqueue) = self.enqueue.take() {
                let _ignored = enqueue.send(self.graceful_shutdown.load(Ordering::SeqCst));
            }
        }
    }

    #[cfg(feature = "memory-guardian")]
    #[tokio::test]
    async fn guardian_failure_cleanup_drains_nested_listener_before_flush() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (listener_started_tx, listener_started_rx) = tokio::sync::oneshot::channel();
        let (release_listener_tx, release_listener_rx) = tokio::sync::oneshot::channel();
        let (persistence_enqueue_tx, persistence_enqueue_rx) = tokio::sync::oneshot::channel();
        let graceful_shutdown = Arc::new(AtomicBool::new(false));
        let listener_shutdown = Arc::clone(&graceful_shutdown);
        let mut server = tokio::spawn(async move {
            let mut listeners = JoinSet::new();
            listeners.spawn(async move {
                let _persistence_on_drop = PersistenceOnDrop {
                    graceful_shutdown: listener_shutdown,
                    enqueue: Some(persistence_enqueue_tx),
                };
                listener_started_tx
                    .send(())
                    .expect("test should wait for the nested listener to start");
                let mut shutdown_rx = shutdown_rx;
                shutdown_rx
                    .changed()
                    .await
                    .expect("guardian shutdown sender should stay alive");
                release_listener_rx
                    .await
                    .expect("guardian shutdown should release the nested listener");
                graceful_shutdown.store(true, Ordering::SeqCst);
            });
            while listeners.join_next().await.is_some() {}
            Ok::<(), String>(())
        });
        listener_started_rx
            .await
            .expect("nested listener should start before guardian cleanup");

        let (shutdown_started_tx, shutdown_started_rx) = tokio::sync::oneshot::channel();
        let release_listener = tokio::spawn(async move {
            shutdown_started_rx
                .await
                .expect("guardian cleanup should begin shutdown");
            release_listener_tx
                .send(())
                .expect("nested listener should still be alive for graceful shutdown");
        });
        let begin_events = Arc::clone(&events);
        let flush_events = Arc::clone(&events);
        super::abort_server_after_guardian_failure(
            &mut server,
            move || {
                shutdown_tx
                    .send(true)
                    .expect("guardian shutdown sender should stay alive");
                shutdown_started_tx
                    .send(())
                    .expect("release task should wait for guardian shutdown");
                begin_events
                    .lock()
                    .expect("event log should remain available")
                    .push("shutdown");
            },
            async move {
                assert!(
                    persistence_enqueue_rx
                        .await
                        .expect("nested listener should schedule persistence on drop"),
                    "persistence must be scheduled only after nested listener shutdown completes"
                );
                flush_events
                    .lock()
                    .expect("event log should remain available")
                    .push("flush");
            },
        )
        .await;
        release_listener
            .await
            .expect("release task should complete without panic");

        assert_eq!(
            *events.lock().expect("event log should remain available"),
            ["shutdown", "flush"]
        );
    }

    use llm_guard_proxy_state::RequestId;

    use super::{
        EvidenceCommand, parse_evidence_command, parse_proxy_options, proxy::render_health,
        render_listening,
    };
    #[cfg(feature = "memory-guardian")]
    use super::{GuardianCommand, parse_guardian_command};
    #[cfg(feature = "host-telemetry")]
    use super::{TelemetryCommand, parse_telemetry_command};

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
            parse_proxy_options(args)
                .expect("args should parse")
                .config_path,
            Some("dev.toml".into()),
        );
    }

    #[cfg(feature = "memory-guardian")]
    #[test]
    fn parses_guardian_subcommand_arguments() {
        let args = [
            OsString::from("--config"),
            OsString::from("config.toml"),
            OsString::from("--runtime-dir"),
            OsString::from("/run/user/1000/gb10-memory-guardian"),
        ];
        assert_eq!(
            parse_guardian_command(&args).expect("guardian args should parse"),
            GuardianCommand {
                config_path: "config.toml".into(),
                runtime_dir: "/run/user/1000/gb10-memory-guardian".into(),
            }
        );
    }

    #[cfg(feature = "host-telemetry")]
    #[test]
    fn parses_telemetry_subcommand_arguments() {
        let args = [OsString::from("--config=telemetry.toml")];
        assert_eq!(
            parse_telemetry_command(&args).expect("telemetry args should parse"),
            TelemetryCommand {
                config_path: "telemetry.toml".into(),
            }
        );
    }

    #[cfg(feature = "memory-guardian")]
    #[test]
    fn guardian_subcommand_requires_runtime_directory() {
        let args = [OsString::from("--config"), OsString::from("guardian.toml")];
        let error = parse_guardian_command(&args).expect_err("runtime dir is required");
        assert!(error.contains("--runtime-dir"));
    }

    #[cfg(feature = "memory-guardian")]
    #[test]
    fn proxy_accepts_a_guardian_runtime_directory_override() {
        let args = [
            OsString::from("llm-guard-proxy"),
            OsString::from("--guardian-runtime-dir"),
            OsString::from("/run/user/1000/gb10-memory-guardian"),
        ];
        assert_eq!(
            parse_proxy_options(args)
                .expect("runtime override should parse")
                .guardian_runtime_dir,
            Some("/run/user/1000/gb10-memory-guardian".into())
        );
    }

    #[cfg(feature = "memory-guardian")]
    #[test]
    fn proxy_rejects_the_removed_separate_guardian_config() {
        let args = [
            OsString::from("llm-guard-proxy"),
            OsString::from("--guardian-config=guardian.toml"),
        ];
        let error = parse_proxy_options(args).expect_err("separate config must be rejected");
        assert!(error.contains("shared --config"));
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

    #[tokio::test]
    async fn watchdog_join_is_bounded_during_shutdown() {
        let watchdog = tokio::spawn(std::future::pending::<()>());
        let joined = tokio::time::timeout(
            Duration::from_millis(100),
            super::join_stuck_engine_watchdog(watchdog, Duration::from_millis(1)),
        )
        .await;

        assert!(joined.is_ok(), "shutdown must not await a stalled watchdog");
    }

    #[test]
    fn runtime_shutdown_timeout_returns_with_a_started_blocking_task() {
        let (started_tx, started_rx) = mpsc::channel();
        let (runtime_ready_tx, runtime_ready_rx) = mpsc::channel();
        let (begin_shutdown_tx, begin_shutdown_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (shutdown_finished_tx, shutdown_finished_rx) = mpsc::channel();

        let runtime_thread = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime should build");
            runtime.block_on(async move {
                tokio::task::spawn_blocking(move || {
                    started_tx
                        .send(())
                        .expect("test should wait for the blocking task to start");
                    let _release = release_rx.recv();
                });
            });
            runtime_ready_tx
                .send(())
                .expect("test should wait until the runtime is ready to shut down");
            begin_shutdown_rx
                .recv()
                .expect("test should trigger runtime shutdown");
            super::shutdown_runtime(runtime, Duration::from_millis(25));
            shutdown_finished_tx
                .send(())
                .expect("test should wait for bounded runtime shutdown");
        });

        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("blocking task should start before runtime shutdown");
        runtime_ready_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("runtime should be ready to shut down after spawning work");
        begin_shutdown_tx
            .send(())
            .expect("runtime thread should still wait for the shutdown signal");
        shutdown_finished_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("runtime shutdown must not wait for the stalled blocking task");
        release_tx
            .send(())
            .expect("stalled blocking task should still be waiting after runtime shutdown");
        runtime_thread
            .join()
            .expect("runtime shutdown thread should not panic");
    }
}
