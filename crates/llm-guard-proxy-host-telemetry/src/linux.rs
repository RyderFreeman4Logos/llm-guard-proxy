//! Linux procfs sampler and observer-only runtime.

use crate::{
    DiskCounters, DiskRate, GpuSample, HostSample, LoadAverage, MemorySample, PolicyDecision,
    SamplerConfig, SwapGuard, TelemetryConfig, TelemetryError, TelemetryEvent, TelemetryIteration,
    TelemetryStore,
};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Stdio,
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::{io::AsyncReadExt, process::Command, time};

const LINUX_SECTOR_BYTES: u64 = 512;
const MAX_PROC_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_PROC_SOURCE_BYTES_U64: u64 = 1024 * 1024;
const MAX_GPU_OUTPUT_BYTES: usize = 1024;
const MAX_GPU_OUTPUT_BYTES_U64: u64 = 1024;
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

    /// Reads, evaluates, and persists one host sample.
    ///
    /// # Errors
    ///
    /// Returns an error for a failed mandatory procfs read or persistence
    /// operation. It never invokes service management or a recovery action.
    pub async fn tick(&mut self) -> Result<TelemetryIteration, TelemetryError> {
        let sample = self.sampler.sample().await?;
        let disk_rate = self
            .previous
            .and_then(|previous| derive_disk_rate(previous, sample));
        let decision = self.guard.observe(
            sample.sampled_at_unix_ms,
            sample.memory.available_kib,
            sample.memory.swap_used_kib(),
        );
        self.store.record(sample, disk_rate, decision)?;
        self.previous = Some(sample);
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
                Ok(TelemetryIteration::Sampled(PolicyDecision {
                    event: Some(event), ..
                })) => eprintln!("host telemetry: {}", render_event(event)),
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
        let command = self.config.gpu_command()?;
        let mut child = Command::new(command)
            .args(GPU_QUERY_ARGS)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        let output = time::timeout(self.config.gpu_timeout(), async {
            let mut output = Vec::with_capacity(MAX_GPU_OUTPUT_BYTES.saturating_add(1));
            stdout
                .take(MAX_GPU_OUTPUT_BYTES_U64.saturating_add(1))
                .read_to_end(&mut output)
                .await
                .ok()?;
            if output.len() > MAX_GPU_OUTPUT_BYTES || !child.wait().await.ok()?.success() {
                return None;
            }
            Some(output)
        })
        .await
        .ok()??;
        let text = std::str::from_utf8(&output).ok()?;
        parse_gpu_csv(text).ok()
    }
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
    use super::{derive_disk_rate, parse_diskstats, parse_gpu_csv, parse_loadavg, parse_meminfo};
    use crate::{DiskCounters, HostSample, LoadAverage, MemorySample};

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
}
