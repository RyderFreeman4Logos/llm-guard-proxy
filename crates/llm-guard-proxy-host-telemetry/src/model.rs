//! Value types emitted by the host telemetry sampler.

/// Memory and swap counters parsed from `/proc/meminfo`, in KiB.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemorySample {
    /// Physical memory capacity.
    pub total_kib: u64,
    /// Kernel-estimated memory available for new allocations.
    pub available_kib: u64,
    /// Configured swap capacity.
    pub swap_total_kib: u64,
    /// Unused swap capacity.
    pub swap_free_kib: u64,
}

impl MemorySample {
    /// Returns used swap without underflowing on malformed counters.
    #[must_use]
    pub const fn swap_used_kib(self) -> u64 {
        self.swap_total_kib.saturating_sub(self.swap_free_kib)
    }
}

/// One, five, and fifteen minute load averages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadAverage {
    /// One-minute load average.
    pub one: f64,
    /// Five-minute load average.
    pub five: f64,
    /// Fifteen-minute load average.
    pub fifteen: f64,
}

/// Monotonic device counters parsed from `/proc/diskstats`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiskCounters {
    /// Sectors read since boot.
    pub read_sectors: u64,
    /// Sectors written since boot.
    pub write_sectors: u64,
    /// Milliseconds spent with I/O in progress since boot.
    pub io_millis: u64,
}

/// Disk activity derived from two samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiskRate {
    /// Read throughput in bytes per second.
    pub read_bytes_per_sec: u64,
    /// Write throughput in bytes per second.
    pub write_bytes_per_sec: u64,
    /// I/O busy time normalized to milliseconds per second.
    pub io_millis_per_sec: u64,
}

/// Optional single-GPU status reported by `nvidia-smi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSample {
    /// GPU temperature in degrees Celsius.
    pub temperature_c: f64,
    /// Board power draw in watts.
    pub power_w: f64,
    /// GPU utilization percentage.
    pub utilization_percent: f64,
    /// Graphics clock in MHz.
    pub clock_mhz: f64,
}

/// One coherent host observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostSample {
    /// Wall-clock sample time in milliseconds since `UNIX_EPOCH`.
    pub sampled_at_unix_ms: u64,
    /// Memory and swap data.
    pub memory: MemorySample,
    /// Load averages.
    pub load: LoadAverage,
    /// Optional selected disk counters.
    pub disk: Option<DiskCounters>,
    /// Optional GPU status.
    pub gpu: Option<GpuSample>,
}

/// Current state of the observer-only swap guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryState {
    /// No memory or swap threshold is active.
    Healthy,
    /// Swap is high enough to be worth recording but below the alert threshold.
    SwapWarning,
    /// Memory or swap pressure requires a bounded evidence record.
    Alert(PressureReason),
}

/// The threshold combination that produced an alert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PressureReason {
    /// `MemAvailable` fell below the configured lower bound.
    MemoryAvailable,
    /// Used swap exceeded the configured alert bound.
    Swap,
    /// Both memory and swap conditions are active.
    MemoryAndSwap,
}

/// State transition that should be emitted to logs and persisted as evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryEvent {
    /// Swap crossed its warning threshold.
    SwapWarning,
    /// Pressure crossed an alert threshold or passed its repeat interval.
    Alert(PressureReason),
    /// A prior warning or alert has recovered.
    Cleared,
}

impl TelemetryEvent {
    /// Returns whether this event needs a bounded evidence row.
    #[must_use]
    pub const fn collects_evidence(self) -> bool {
        matches!(self, Self::Alert(_))
    }
}

/// Policy outcome attached to each stored sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyDecision {
    /// State after evaluating the sample.
    pub state: TelemetryState,
    /// Transition to emit, if any.
    pub event: Option<TelemetryEvent>,
}

/// Result of one telemetry loop iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryIteration {
    /// A sample was persisted successfully.
    Sampled(PolicyDecision),
    /// The compile target does not provide a real host sampler.
    Unsupported,
}
