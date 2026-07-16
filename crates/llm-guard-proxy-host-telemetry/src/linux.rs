//! Linux procfs sampler and observer-only runtime.

use crate::{
    DiskCounters, DiskRate, GpuBackend, GpuSample, HostSample, LoadAverage, MemorySample,
    SamplerConfig, SwapGuard, TelemetryConfig, TelemetryError, TelemetryEvent, TelemetryIteration,
    TelemetryStore,
};
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use std::{
    fs::File,
    io::Read,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::{io::AsyncReadExt, process::Command, time};

const LINUX_SECTOR_BYTES: u64 = 512;
const MAX_PROC_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_PROC_SOURCE_BYTES_U64: u64 = 1024 * 1024;
const MAX_GPU_OUTPUT_BYTES: usize = 1024;
const MAX_GPU_OUTPUT_BYTES_U64: u64 = 1024;
const NVIDIA_SMI_PATH: &str = "/usr/bin/nvidia-smi";
const GPU_TERMINATION_GRACE: Duration = Duration::from_millis(100);
const GPU_QUERY_ARGS: [&str; 3] = [
    "--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.gr",
    "--format=csv,noheader,nounits",
    "--id=0",
];

/// Linux host telemetry sampler with no service-control capability.
#[derive(Debug)]
pub struct HostTelemetry {
    sampler: LinuxSampler,
    guard: SwapGuard,
    store: TelemetryStore,
    previous: Option<HostSample>,
}

impl HostTelemetry {
    /// Opens an observer-only telemetry runtime from a configuration file.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration or local `SQLite` initialization
    /// fails. Individual samples are recoverable and are logged by
    /// [`Self::run_until`] without stopping the process.
    pub fn open(config_path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let config = TelemetryConfig::load(&config_path.into())?;
        let sampler = LinuxSampler::new(config.sampler().clone());
        let guard = SwapGuard::new(config.swap_guard().clone());
        let store = TelemetryStore::open(config.storage().clone())?;
        Ok(Self {
            sampler,
            guard,
            store,
            previous: None,
        })
    }

    /// Reads, evaluates, emits, and persists one host sample.
    ///
    /// Alert transitions are emitted before persistence so a degraded local
    /// database cannot suppress observer events.
    ///
    /// # Errors
    ///
    /// Returns an error for a failed mandatory procfs read or persistence
    /// operation. It never invokes service management or a recovery action.
    pub async fn tick(&mut self) -> Result<TelemetryIteration, TelemetryError> {
        self.tick_at(Instant::now(), emit_event).await
    }

    async fn tick_at<F>(
        &mut self,
        observed_at: Instant,
        mut emit: F,
    ) -> Result<TelemetryIteration, TelemetryError>
    where
        F: FnMut(TelemetryEvent),
    {
        let sample = self.sampler.sample().await?;
        let disk_rate = self
            .previous
            .and_then(|previous| derive_disk_rate(previous, sample));
        let decision = self.guard.observe(
            observed_at,
            sample.memory.available_kib,
            sample.memory.swap_used_kib(),
        );
        if let Some(event) = decision.event {
            emit(event);
        }
        self.previous = Some(sample);
        self.store.record(sample, disk_rate, decision)?;
        Ok(TelemetryIteration::Sampled(decision))
    }

    /// Samples until shutdown while treating routine observer failures as
    /// recoverable diagnostics.
    ///
    /// # Errors
    ///
    /// This method returns only after the supplied shutdown future resolves.
    /// Startup configuration errors are reported by [`Self::open`].
    pub async fn run_until<F>(&mut self, shutdown: F) -> Result<(), TelemetryError>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(shutdown);
        loop {
            match self.tick().await {
                Ok(TelemetryIteration::Sampled(_) | TelemetryIteration::Unsupported) => {}
                Err(error) => eprintln!("host telemetry: observer read failed: {error}"),
            }
            tokio::select! {
                () = &mut shutdown => return Ok(()),
                () = time::sleep(self.sampler.config.interval()) => {}
            }
        }
    }
}

#[derive(Debug)]
struct LinuxSampler {
    config: SamplerConfig,
}

impl LinuxSampler {
    const fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    async fn sample(&self) -> Result<HostSample, SamplerError> {
        let memory = read_memory_sample(&self.config.proc_root().join("meminfo"))?;
        let load = read_load_average(&self.config.proc_root().join("loadavg"))?;
        let disk = self
            .config
            .disk_device()
            .map(|device| read_disk_counters(&self.config.proc_root().join("diskstats"), device))
            .transpose()?
            .flatten();
        let gpu = self.sample_gpu().await;
        Ok(HostSample {
            sampled_at_unix_ms: unix_millis()?,
            memory,
            load,
            disk,
            gpu,
        })
    }

    async fn sample_gpu(&self) -> Option<GpuSample> {
        match self.config.gpu_backend() {
            GpuBackend::Disabled => None,
            GpuBackend::NvidiaSmi => {
                run_gpu_probe(Path::new(NVIDIA_SMI_PATH), self.config.gpu_timeout()).await
            }
        }
    }
}

async fn run_gpu_probe(command: &Path, timeout: Duration) -> Option<GpuSample> {
    let mut command = Command::new(command);
    command
        .args(GPU_QUERY_ARGS)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    command.as_std_mut().process_group(0);
    let mut child = command.spawn().ok()?;
    let Some(group_id) = child.id().and_then(|id| i32::try_from(id).ok()) else {
        let _ = child.kill().await;
        let _ = child.wait().await;
        return None;
    };
    let Some(stdout) = child.stdout.take() else {
        terminate_gpu_process_group(&mut child, group_id).await;
        return None;
    };

    let result = time::timeout(timeout, async {
        let mut output = Vec::with_capacity(MAX_GPU_OUTPUT_BYTES.saturating_add(1));
        let read_result = stdout
            .take(MAX_GPU_OUTPUT_BYTES_U64.saturating_add(1))
            .read_to_end(&mut output)
            .await;
        let status = child.wait().await;
        match (read_result, status) {
            (Ok(_), Ok(status)) if output.len() <= MAX_GPU_OUTPUT_BYTES && status.success() => {
                Some(output)
            }
            _ => None,
        }
    })
    .await;

    let output = if let Ok(output) = result {
        output?
    } else {
        terminate_gpu_process_group(&mut child, group_id).await;
        return None;
    };
    let text = std::str::from_utf8(&output).ok()?;
    parse_gpu_csv(text).ok()
}

async fn terminate_gpu_process_group(child: &mut tokio::process::Child, group_id: i32) {
    let group = Pid::from_raw(group_id);
    if killpg(group, Signal::SIGTERM).is_err() {
        let _ = child.start_kill();
    }
    time::sleep(GPU_TERMINATION_GRACE).await;
    if killpg(group, Signal::SIGKILL).is_err() {
        let _ = child.start_kill();
    }
    let _ = child.wait().await;
}

fn read_memory_sample(path: &Path) -> Result<MemorySample, SamplerError> {
    let text = read_proc(path)?;
    parse_meminfo(&text)
}

fn read_load_average(path: &Path) -> Result<LoadAverage, SamplerError> {
    let text = read_proc(path)?;
    parse_loadavg(&text)
}

fn read_disk_counters(path: &Path, device: &str) -> Result<Option<DiskCounters>, SamplerError> {
    let text = read_proc(path)?;
    parse_diskstats(&text, device)
}

fn read_proc(path: &Path) -> Result<String, SamplerError> {
    let file = File::open(path).map_err(|source| SamplerError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = file.take(MAX_PROC_SOURCE_BYTES_U64.saturating_add(1));
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|source| SamplerError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    if text.len() > MAX_PROC_SOURCE_BYTES {
        return Err(SamplerError::SourceTooLarge {
            path: path.to_path_buf(),
        });
    }
    Ok(text)
}

fn unix_millis() -> Result<u64, SamplerError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .map_err(|source| SamplerError::Clock { source })
}

/// Parses the four host counters required from `/proc/meminfo`.
///
/// # Errors
///
/// Returns an error when a required field is absent or not an unsigned KiB value.
pub fn parse_meminfo(input: &str) -> Result<MemorySample, SamplerError> {
    let mut total = None;
    let mut available = None;
    let mut swap_total = None;
    let mut swap_free = None;
    for line in input.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else {
            continue;
        };
        let Some(value) = fields.next() else {
            continue;
        };
        let value = value.parse::<u64>().map_err(|_error| SamplerError::Parse {
            field: key.to_owned(),
        })?;
        match key {
            "MemTotal:" => total = Some(value),
            "MemAvailable:" => available = Some(value),
            "SwapTotal:" => swap_total = Some(value),
            "SwapFree:" => swap_free = Some(value),
            _ => {}
        }
    }
    Ok(MemorySample {
        total_kib: total.ok_or(SamplerError::MissingField("MemTotal"))?,
        available_kib: available.ok_or(SamplerError::MissingField("MemAvailable"))?,
        swap_total_kib: swap_total.ok_or(SamplerError::MissingField("SwapTotal"))?,
        swap_free_kib: swap_free.ok_or(SamplerError::MissingField("SwapFree"))?,
    })
}

/// Parses load averages from `/proc/loadavg`.
///
/// # Errors
///
/// Returns an error when the first three fields are absent or invalid floats.
pub fn parse_loadavg(input: &str) -> Result<LoadAverage, SamplerError> {
    let mut fields = input.split_whitespace();
    let parse = |field: &'static str, value: Option<&str>| {
        value
            .ok_or(SamplerError::MissingField(field))?
            .parse::<f64>()
            .map_err(|_error| SamplerError::Parse {
                field: field.to_owned(),
            })
    };
    Ok(LoadAverage {
        one: parse("load1", fields.next())?,
        five: parse("load5", fields.next())?,
        fifteen: parse("load15", fields.next())?,
    })
}

/// Parses one device's counters from `/proc/diskstats`.
///
/// Missing devices are normal when the selected volume is absent, so this
/// function returns `Ok(None)` in that case.
///
/// # Errors
///
/// Returns an error when the selected device row is malformed.
pub fn parse_diskstats(input: &str, device: &str) -> Result<Option<DiskCounters>, SamplerError> {
    for line in input.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.get(2).copied() != Some(device) {
            continue;
        }
        let parse = |index: usize, field: &'static str| {
            fields
                .get(index)
                .ok_or(SamplerError::MissingField(field))?
                .parse::<u64>()
                .map_err(|_error| SamplerError::Parse {
                    field: field.to_owned(),
                })
        };
        return Ok(Some(DiskCounters {
            read_sectors: parse(5, "disk read sectors")?,
            write_sectors: parse(9, "disk write sectors")?,
            io_millis: parse(12, "disk io milliseconds")?,
        }));
    }
    Ok(None)
}

/// Parses the fixed `nvidia-smi` CSV query used by the optional GPU probe.
///
/// # Errors
///
/// Returns an error when the first data row does not contain four finite numeric fields.
pub fn parse_gpu_csv(input: &str) -> Result<GpuSample, SamplerError> {
    let line = input
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or(SamplerError::MissingField("GPU CSV row"))?;
    let values = line
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_error| SamplerError::Parse {
            field: String::from("GPU CSV value"),
        })?;
    if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
        return Err(SamplerError::Parse {
            field: String::from("GPU CSV row"),
        });
    }
    Ok(GpuSample {
        temperature_c: values[0],
        power_w: values[1],
        utilization_percent: values[2],
        clock_mhz: values[3],
    })
}

/// Derives disk activity from two coherent host samples.
#[must_use]
pub fn derive_disk_rate(previous: HostSample, current: HostSample) -> Option<DiskRate> {
    let previous_disk = previous.disk?;
    let current_disk = current.disk?;
    let elapsed_ms = current
        .sampled_at_unix_ms
        .saturating_sub(previous.sampled_at_unix_ms);
    (elapsed_ms != 0).then(|| DiskRate {
        read_bytes_per_sec: rate_per_second(
            current_disk
                .read_sectors
                .saturating_sub(previous_disk.read_sectors),
            LINUX_SECTOR_BYTES,
            elapsed_ms,
        ),
        write_bytes_per_sec: rate_per_second(
            current_disk
                .write_sectors
                .saturating_sub(previous_disk.write_sectors),
            LINUX_SECTOR_BYTES,
            elapsed_ms,
        ),
        io_millis_per_sec: rate_per_second(
            current_disk
                .io_millis
                .saturating_sub(previous_disk.io_millis),
            1,
            elapsed_ms,
        ),
    })
}

fn rate_per_second(delta: u64, multiplier: u64, elapsed_ms: u64) -> u64 {
    delta
        .saturating_mul(multiplier)
        .saturating_mul(1_000)
        .checked_div(elapsed_ms)
        .unwrap_or(0)
}

fn emit_event(event: TelemetryEvent) {
    eprintln!("host telemetry: {}", render_event(event));
}

fn render_event(event: TelemetryEvent) -> &'static str {
    match event {
        TelemetryEvent::SwapWarning => "swap warning",
        TelemetryEvent::Alert(crate::PressureReason::MemoryAvailable) => {
            "memory-available alert; evidence retained"
        }
        TelemetryEvent::Alert(crate::PressureReason::Swap) => "swap alert; evidence retained",
        TelemetryEvent::Alert(crate::PressureReason::MemoryAndSwap) => {
            "memory-and-swap alert; evidence retained"
        }
        TelemetryEvent::Cleared => "memory pressure cleared",
    }
}

/// Errors from mandatory Linux host data sources.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// Procfs read failed.
    #[error("read {path}: {source}", path = path.display())]
    Read {
        /// Source path.
        path: PathBuf,
        /// Underlying read error.
        source: std::io::Error,
    },
    /// A procfs source exceeded the configured bounded parser input size.
    #[error("host telemetry source exceeds {MAX_PROC_SOURCE_BYTES} bytes: {path}", path = path.display())]
    SourceTooLarge {
        /// Source path.
        path: PathBuf,
    },
    /// System wall clock was before `UNIX_EPOCH`.
    #[error("read system clock: {source}")]
    Clock {
        /// Underlying clock error.
        source: std::time::SystemTimeError,
    },
    /// A mandatory procfs field was absent.
    #[error("missing {0} in host telemetry input")]
    MissingField(&'static str),
    /// A numeric source field was malformed.
    #[error("invalid host telemetry field: {field}")]
    Parse {
        /// Field or source label.
        field: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        HostTelemetry, derive_disk_rate, parse_diskstats, parse_gpu_csv, parse_loadavg,
        parse_meminfo, run_gpu_probe,
    };
    use crate::{
        DiskCounters, HostSample, LoadAverage, MemorySample, PressureReason, TelemetryError,
        TelemetryEvent,
    };
    use nix::{errno::Errno, sys::signal::kill, unistd::Pid};
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant},
    };

    static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn parses_memory_load_disk_and_gpu_sources() {
        let memory = parse_meminfo(
            "MemTotal:       1048576 kB\nMemAvailable:    204800 kB\nSwapTotal:       131072 kB\nSwapFree:         32768 kB\n",
        )
        .expect("meminfo parses");
        assert_eq!(memory.available_kib, 204_800);
        assert_eq!(memory.swap_used_kib(), 98_304);
        let load = parse_loadavg("1.25 2.50 3.75 1/100 123\n").expect("loadavg parses");
        assert!((load.five - 2.5).abs() < f64::EPSILON);
        assert_eq!(
            parse_diskstats("259 0 nvme0n1 1 0 100 0 1 0 200 0 0 300 0\n", "nvme0n1")
                .expect("diskstats parses")
                .expect("target disk exists")
                .write_sectors,
            200
        );
        let gpu = parse_gpu_csv("40, 120.5, 75, 1800\n").expect("gpu csv parses");
        assert!((gpu.clock_mhz - 1_800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn derives_rates_without_underflow_when_counters_reset() {
        let previous = sample(
            1_000,
            DiskCounters {
                read_sectors: 100,
                write_sectors: 100,
                io_millis: 100,
            },
        );
        let current = sample(
            2_000,
            DiskCounters {
                read_sectors: 300,
                write_sectors: 50,
                io_millis: 150,
            },
        );
        assert_eq!(
            derive_disk_rate(previous, current),
            Some(crate::DiskRate {
                read_bytes_per_sec: 102_400,
                write_bytes_per_sec: 0,
                io_millis_per_sec: 50,
            })
        );
    }

    #[tokio::test]
    async fn gpu_timeout_scrubs_environment_and_terminates_the_process_group() {
        assert!(
            std::env::var_os("HOME").is_some(),
            "test requires inherited HOME"
        );
        let directory = test_directory("gpu-timeout");
        fs::create_dir_all(&directory).expect("fixture directory creates");
        let command = directory.join("nvidia-smi");
        let leader_pid_path = directory.join("leader.pid");
        let child_pid_path = directory.join("child.pid");
        let leaked_env_path = directory.join("leaked-home");
        let script = format!(
            "#!/bin/sh\n\
             child=\n\
             cleanup() {{\n\
               /bin/kill \"$child\" 2>/dev/null || true\n\
               wait \"$child\" 2>/dev/null || true\n\
               exit 0\n\
             }}\n\
             trap cleanup TERM\n\
             if [ -n \"${{HOME+x}}\" ]; then : > {}; fi\n\
             printf '%s\\n' \"$$\" > {}\n\
             /bin/sleep 30 &\n\
             child=$!\n\
             printf '%s\\n' \"$child\" > {}\n\
             wait \"$child\"\n",
            leaked_env_path.display(),
            leader_pid_path.display(),
            child_pid_path.display(),
        );
        fs::write(&command, script).expect("GPU fixture writes");
        let mut permissions = fs::metadata(&command)
            .expect("GPU fixture metadata reads")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&command, permissions).expect("GPU fixture becomes executable");

        assert_eq!(
            run_gpu_probe(&command, Duration::from_millis(250)).await,
            None,
            "hanging GPU probe must degrade to a missing sample"
        );
        assert!(
            !leaked_env_path.exists(),
            "the GPU command must not inherit the observer environment"
        );
        let leader_pid = read_pid(&leader_pid_path);
        let child_pid = read_pid(&child_pid_path);
        wait_for_group_exit(leader_pid).await;
        assert!(
            !PathBuf::from(format!("/proc/{leader_pid}")).exists(),
            "GPU leader must be reaped"
        );
        assert!(
            !PathBuf::from(format!("/proc/{child_pid}")).exists(),
            "GPU descendant must be terminated and reaped"
        );
        fs::remove_dir_all(directory).expect("GPU fixture directory removes");
    }

    #[tokio::test]
    async fn persistence_failure_does_not_suppress_the_alert_event() {
        let directory = test_directory("sink-failure");
        let proc_root = directory.join("proc");
        fs::create_dir_all(&proc_root).expect("proc fixture directory creates");
        fs::write(
            proc_root.join("meminfo"),
            "MemTotal: 1048576 kB\nMemAvailable: 512 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        )
        .expect("meminfo fixture writes");
        fs::write(proc_root.join("loadavg"), "1.0 1.0 1.0 1/1 1\n")
            .expect("loadavg fixture writes");
        fs::write(proc_root.join("diskstats"), "").expect("diskstats fixture writes");
        let database_path = directory.join("telemetry.sqlite3");
        let config_path = directory.join("telemetry.toml");
        fs::write(
            &config_path,
            format!(
                "schema_version = 1\n\
                 [storage]\n\
                 sqlite_path = \"{}\"\n\
                 max_records = 3\n\
                 prune_to_records = 2\n\
                 [sampler]\n\
                 proc_root = \"{}\"\n\
                 gpu_backend = \"disabled\"\n\
                 [swap_guard]\n\
                 warn_swap_mib = 2\n\
                 alert_swap_mib = 4\n\
                 alert_mem_available_mib = 1\n\
                 alert_repeat_secs = 60\n",
                database_path.display(),
                proc_root.display(),
            ),
        )
        .expect("telemetry config writes");
        let mut telemetry = HostTelemetry::open(config_path).expect("telemetry runtime opens");
        let sabotage =
            rusqlite::Connection::open(&database_path).expect("sabotage connection opens");
        sabotage
            .execute_batch("DROP TABLE evidence; DROP TABLE samples;")
            .expect("persistence sink is made unavailable");
        drop(sabotage);

        let mut events = Vec::new();
        let result = telemetry
            .tick_at(Instant::now(), |event| events.push(event))
            .await;
        assert!(matches!(result, Err(TelemetryError::Store(_))));
        assert_eq!(
            events,
            vec![TelemetryEvent::Alert(PressureReason::MemoryAvailable)],
            "alert emission must happen even when SQLite rejects the sample"
        );
        drop(telemetry);
        fs::remove_dir_all(directory).expect("sink-failure fixture directory removes");
    }

    fn sample(sampled_at_unix_ms: u64, disk: DiskCounters) -> HostSample {
        HostSample {
            sampled_at_unix_ms,
            memory: MemorySample {
                total_kib: 1,
                available_kib: 1,
                swap_total_kib: 1,
                swap_free_kib: 1,
            },
            load: LoadAverage {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            },
            disk: Some(disk),
            gpu: None,
        }
    }

    fn test_directory(label: &str) -> PathBuf {
        let id = NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "llm-guard-proxy-host-telemetry-{label}-{}-{id}",
            std::process::id()
        ))
    }

    fn read_pid(path: &Path) -> i32 {
        fs::read_to_string(path)
            .expect("PID marker reads")
            .trim()
            .parse()
            .expect("PID marker is numeric")
    }

    async fn wait_for_group_exit(group_id: i32) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match kill(Pid::from_raw(-group_id), None) {
                Err(Errno::ESRCH) => return,
                Ok(()) => {}
                Err(error) => panic!("probe GPU process group {group_id}: {error}"),
            }
            assert!(
                Instant::now() < deadline,
                "GPU process group {group_id} survived timeout cleanup"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
