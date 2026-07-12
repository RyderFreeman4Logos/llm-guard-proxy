use std::collections::BTreeMap;

use crate::settings::HeartbeatMode;

use super::model::{
    AttemptMetricCount, HeartbeatModeMetricCount, HistogramBucket, LatencyHistogram,
    ObservabilityMetricsSnapshot, RequestMetricCount, RequestTerminalMetricCount,
    RetentionPruningStats, RetentionUsage, UpstreamErrorMetricCount,
};

const HISTOGRAM_BUCKETS_MS: &[u64] = &[
    10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000,
];
const OTHER_HEARTBEAT_MODE: &str = "other";

#[derive(Clone, Debug)]
pub(super) struct RequestMetricObservation {
    request_key: (String, String, String, String),
    terminal_key: (String, String, String),
    loop_aborted: bool,
    duration_ms: Option<u64>,
    heartbeat_mode: Option<String>,
}

#[derive(Clone, Copy)]
pub(super) struct RequestMetricInput<'a> {
    pub(super) status: &'a str,
    pub(super) downstream_mode: &'a str,
    pub(super) upstream_mode: &'a str,
    pub(super) http_status: Option<i64>,
    pub(super) abort_reason: Option<&'a str>,
    pub(super) request_metadata_json: &'a str,
    pub(super) response_metadata_json: &'a str,
    pub(super) duration_ms: Option<i64>,
}

impl RequestMetricObservation {
    pub(super) fn new(input: RequestMetricInput<'_>) -> Self {
        let http_status_class = http_status_class(input.http_status);
        let terminal_reason = request_terminal_reason(input.status, input.abort_reason);
        let loop_aborted = input.abort_reason == Some("loop_guard")
            || sqlite_like_loop_detected_true(input.response_metadata_json);
        let heartbeat_mode =
            metadata_value(input.response_metadata_json, "downstream_liveness_mode")
                .or_else(|| metadata_value(input.request_metadata_json, "downstream_liveness_mode"))
                .map(|mode| normalized_heartbeat_mode_label(&mode));
        Self {
            request_key: (
                input.status.to_owned(),
                input.downstream_mode.to_owned(),
                input.upstream_mode.to_owned(),
                http_status_class.clone(),
            ),
            terminal_key: (input.status.to_owned(), terminal_reason, http_status_class),
            loop_aborted,
            duration_ms: input.duration_ms.map(nonnegative_i64_to_u64),
            heartbeat_mode,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct AttemptMetricObservation {
    attempt_key: (String, String, String),
    retried: bool,
    loop_aborted: bool,
    upstream_error_key: Option<(String, String)>,
    first_token_latency_ms: Option<u64>,
}

#[derive(Clone, Copy)]
pub(super) struct AttemptMetricInput<'a> {
    pub(super) status: &'a str,
    pub(super) upstream_mode: &'a str,
    pub(super) http_status: Option<i64>,
    pub(super) retry_reason: Option<&'a str>,
    pub(super) abort_reason: Option<&'a str>,
    pub(super) response_metadata_json: &'a str,
}

impl AttemptMetricObservation {
    pub(super) fn new(input: AttemptMetricInput<'_>) -> Self {
        let http_status_class = http_status_class(input.http_status);
        let upstream_error_key = (input.status != "succeeded"
            || input.http_status.is_none_or(|status| status >= 500))
        .then(|| {
            (
                upstream_error_kind(input.status, input.http_status),
                http_status_class.clone(),
            )
        });
        Self {
            attempt_key: (
                input.status.to_owned(),
                input.upstream_mode.to_owned(),
                http_status_class,
            ),
            retried: input.status == "retried" || input.retry_reason.is_some(),
            loop_aborted: input.abort_reason == Some("loop_guard")
                || sqlite_like_loop_detected_true(input.response_metadata_json),
            upstream_error_key,
            first_token_latency_ms: metadata_value(
                input.response_metadata_json,
                "first_token_latency_ms",
            )
            .and_then(|value| value.parse::<u64>().ok()),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct MetricsAccumulator {
    request_counts: BTreeMap<(String, String, String, String), u64>,
    request_terminal_counts: BTreeMap<(String, String, String), u64>,
    attempt_counts: BTreeMap<(String, String, String), u64>,
    retry_count: u64,
    loop_abort_count: u64,
    upstream_error_counts: BTreeMap<(String, String), u64>,
    first_token_latency_ms: HistogramAccumulator,
    total_latency_ms: HistogramAccumulator,
    heartbeat_mode_counts: BTreeMap<String, u64>,
    retention_usage: RetentionUsage,
    pruning: RetentionPruningStats,
}

impl MetricsAccumulator {
    pub(super) fn new(retention_usage: RetentionUsage, pruning: RetentionPruningStats) -> Self {
        Self {
            request_counts: BTreeMap::new(),
            request_terminal_counts: BTreeMap::new(),
            attempt_counts: BTreeMap::new(),
            retry_count: 0,
            loop_abort_count: 0,
            upstream_error_counts: BTreeMap::new(),
            first_token_latency_ms: HistogramAccumulator::default(),
            total_latency_ms: HistogramAccumulator::default(),
            heartbeat_mode_counts: BTreeMap::new(),
            retention_usage,
            pruning,
        }
    }

    pub(super) fn add_request(&mut self, observation: &RequestMetricObservation) {
        increment(&mut self.request_counts, observation.request_key.clone());
        increment(
            &mut self.request_terminal_counts,
            observation.terminal_key.clone(),
        );
        if observation.loop_aborted {
            self.loop_abort_count = self.loop_abort_count.saturating_add(1);
        }
        if let Some(duration_ms) = observation.duration_ms {
            self.total_latency_ms.add(duration_ms);
        }
        if let Some(mode) = &observation.heartbeat_mode {
            increment(&mut self.heartbeat_mode_counts, mode.clone());
        }
    }

    pub(super) fn remove_request(&mut self, observation: &RequestMetricObservation) {
        decrement(&mut self.request_counts, &observation.request_key);
        decrement(&mut self.request_terminal_counts, &observation.terminal_key);
        if observation.loop_aborted {
            self.loop_abort_count = self.loop_abort_count.saturating_sub(1);
        }
        if let Some(duration_ms) = observation.duration_ms {
            self.total_latency_ms.remove(duration_ms);
        }
        if let Some(mode) = &observation.heartbeat_mode {
            decrement(&mut self.heartbeat_mode_counts, mode);
        }
    }

    pub(super) fn add_attempt(&mut self, observation: &AttemptMetricObservation) {
        increment(&mut self.attempt_counts, observation.attempt_key.clone());
        if observation.retried {
            self.retry_count = self.retry_count.saturating_add(1);
        }
        if observation.loop_aborted {
            self.loop_abort_count = self.loop_abort_count.saturating_add(1);
        }
        if let Some(key) = &observation.upstream_error_key {
            increment(&mut self.upstream_error_counts, key.clone());
        }
        if let Some(latency_ms) = observation.first_token_latency_ms {
            self.first_token_latency_ms.add(latency_ms);
        }
    }

    pub(super) fn remove_attempt(&mut self, observation: &AttemptMetricObservation) {
        decrement(&mut self.attempt_counts, &observation.attempt_key);
        if observation.retried {
            self.retry_count = self.retry_count.saturating_sub(1);
        }
        if observation.loop_aborted {
            self.loop_abort_count = self.loop_abort_count.saturating_sub(1);
        }
        if let Some(key) = &observation.upstream_error_key {
            decrement(&mut self.upstream_error_counts, key);
        }
        if let Some(latency_ms) = observation.first_token_latency_ms {
            self.first_token_latency_ms.remove(latency_ms);
        }
    }

    pub(super) const fn set_store_state(
        &mut self,
        retention_usage: RetentionUsage,
        pruning: RetentionPruningStats,
    ) {
        self.retention_usage = retention_usage;
        self.pruning = pruning;
    }

    pub(super) fn snapshot(&self) -> ObservabilityMetricsSnapshot {
        ObservabilityMetricsSnapshot {
            request_counts: self
                .request_counts
                .iter()
                .map(
                    |((status, downstream_mode, upstream_mode, http_status_class), count)| {
                        RequestMetricCount {
                            status: status.clone(),
                            downstream_mode: downstream_mode.clone(),
                            upstream_mode: upstream_mode.clone(),
                            http_status_class: http_status_class.clone(),
                            count: *count,
                        }
                    },
                )
                .collect(),
            request_terminal_counts: self
                .request_terminal_counts
                .iter()
                .map(|((status, terminal_reason, http_status_class), count)| {
                    RequestTerminalMetricCount {
                        status: status.clone(),
                        terminal_reason: terminal_reason.clone(),
                        http_status_class: http_status_class.clone(),
                        count: *count,
                    }
                })
                .collect(),
            attempt_counts: self
                .attempt_counts
                .iter()
                .map(
                    |((status, upstream_mode, http_status_class), count)| AttemptMetricCount {
                        status: status.clone(),
                        upstream_mode: upstream_mode.clone(),
                        http_status_class: http_status_class.clone(),
                        count: *count,
                    },
                )
                .collect(),
            retry_count: self.retry_count,
            loop_abort_count: self.loop_abort_count,
            upstream_error_counts: self
                .upstream_error_counts
                .iter()
                .map(
                    |((kind, http_status_class), count)| UpstreamErrorMetricCount {
                        kind: kind.clone(),
                        http_status_class: http_status_class.clone(),
                        count: *count,
                    },
                )
                .collect(),
            first_token_latency_ms: self.first_token_latency_ms.snapshot(),
            total_latency_ms: self.total_latency_ms.snapshot(),
            heartbeat_mode_counts: self
                .heartbeat_mode_counts
                .iter()
                .map(|(mode, count)| HeartbeatModeMetricCount {
                    mode: mode.clone(),
                    count: *count,
                })
                .collect(),
            retention_usage: self.retention_usage,
            pruning: self.pruning,
        }
    }

    #[cfg(test)]
    pub(super) fn snapshot_work_units(&self) -> usize {
        self.request_counts.len()
            + self.request_terminal_counts.len()
            + self.attempt_counts.len()
            + self.upstream_error_counts.len()
            + self.heartbeat_mode_counts.len()
            + HISTOGRAM_BUCKETS_MS.len() * 2
    }
}

#[derive(Clone, Debug, Default)]
struct HistogramAccumulator {
    buckets: [u64; HISTOGRAM_BUCKETS_MS.len()],
    count: u64,
    sum_ms: u128,
}

impl HistogramAccumulator {
    fn add(&mut self, value: u64) {
        for (index, upper_bound) in HISTOGRAM_BUCKETS_MS.iter().enumerate() {
            if value <= *upper_bound {
                self.buckets[index] = self.buckets[index].saturating_add(1);
            }
        }
        self.count = self.count.saturating_add(1);
        self.sum_ms = self.sum_ms.saturating_add(u128::from(value));
    }

    fn remove(&mut self, value: u64) {
        for (index, upper_bound) in HISTOGRAM_BUCKETS_MS.iter().enumerate() {
            if value <= *upper_bound {
                self.buckets[index] = self.buckets[index].saturating_sub(1);
            }
        }
        self.count = self.count.saturating_sub(1);
        self.sum_ms = self.sum_ms.saturating_sub(u128::from(value));
    }

    fn snapshot(&self) -> LatencyHistogram {
        LatencyHistogram {
            buckets: HISTOGRAM_BUCKETS_MS
                .iter()
                .zip(self.buckets)
                .map(|(le_ms, count)| HistogramBucket {
                    le_ms: *le_ms,
                    count,
                })
                .collect(),
            count: self.count,
            sum_ms: self.sum_ms.try_into().unwrap_or(u64::MAX),
        }
    }
}

fn increment<K: Ord>(counts: &mut BTreeMap<K, u64>, key: K) {
    let count = counts.entry(key).or_default();
    *count = count.saturating_add(1);
}

fn decrement<K: Ord>(counts: &mut BTreeMap<K, u64>, key: &K) {
    let remove = match counts.get_mut(key) {
        Some(count) if *count > 1 => {
            *count -= 1;
            false
        }
        Some(_count) => true,
        None => {
            debug_assert!(false, "metric accumulator contribution must exist");
            false
        }
    };
    if remove {
        counts.remove(key);
    }
}

fn http_status_class(status: Option<i64>) -> String {
    match status.and_then(|status| u16::try_from(status).ok()) {
        Some(100..=199) => String::from("1xx"),
        Some(200..=299) => String::from("2xx"),
        Some(300..=399) => String::from("3xx"),
        Some(400..=499) => String::from("4xx"),
        Some(500..=599) => String::from("5xx"),
        Some(_) => String::from("other"),
        None => String::from("none"),
    }
}

fn upstream_error_kind(status: &str, http_status: Option<i64>) -> String {
    let code = http_status.and_then(|status| u16::try_from(status).ok());
    match code {
        None => String::from("transport"),
        Some(500..=599) => String::from("http_5xx"),
        Some(408 | 429) => String::from("http_retryable"),
        Some(_) if status == "retried" => String::from("retry"),
        Some(_) => String::from("attempt_failed"),
    }
}

fn request_terminal_reason(status: &str, abort_reason: Option<&str>) -> String {
    match abort_reason {
        Some("downstream_body_dropped_before_eof" | "downstream_disconnected_while_queued") => {
            String::from("downstream_disconnect")
        }
        Some("server_shutdown" | "server_shutdown_while_queued") => String::from("server_shutdown"),
        Some("loop_guard") => String::from("loop_guard"),
        Some("upstream_stall") => String::from("upstream_stall"),
        Some("hot_restart_timeout" | "hot_restart_error") => String::from("hot_restart"),
        Some("shadow_timeout") => String::from("shadow_timeout"),
        Some(_) => String::from("other_abort"),
        None if status == "succeeded" => String::from("succeeded"),
        None if status == "failed" => String::from("failed"),
        None => String::from("none"),
    }
}

fn metadata_value(metadata_json: &str, key: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(metadata_json).ok()?;
    value.as_object()?.get(key)?.as_str().map(ToOwned::to_owned)
}

pub(super) fn normalized_heartbeat_mode_label(value: &str) -> String {
    HeartbeatMode::from_label(value).map_or_else(
        || String::from(OTHER_HEARTBEAT_MODE),
        |mode| mode.as_str().to_owned(),
    )
}

fn sqlite_like_loop_detected_true(metadata_json: &str) -> bool {
    let Some(loop_pattern_end) = find_sqlite_like_loop_detected(metadata_json) else {
        return false;
    };
    find_ascii_case_insensitive(&metadata_json[loop_pattern_end..], "true").is_some()
}

fn find_sqlite_like_loop_detected(haystack: &str) -> Option<usize> {
    const PREFIX: &str = "loop";
    const SUFFIX: &str = "detected";
    haystack.char_indices().find_map(|(start, _character)| {
        let candidate = haystack.get(start..)?;
        let prefix = candidate.get(..PREFIX.len())?;
        if !prefix.eq_ignore_ascii_case(PREFIX) {
            return None;
        }
        let after_prefix = candidate.get(PREFIX.len()..)?;
        let wildcard_len = after_prefix.chars().next()?.len_utf8();
        let suffix_start = PREFIX.len() + wildcard_len;
        let suffix_end = suffix_start + SUFFIX.len();
        let suffix = candidate.get(suffix_start..suffix_end)?;
        suffix
            .eq_ignore_ascii_case(SUFFIX)
            .then_some(start + suffix_end)
    })
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

fn nonnegative_i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}
