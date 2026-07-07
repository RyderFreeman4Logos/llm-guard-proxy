//! Offline replay calibration (issue #112).
//!
//! This module feeds recorded SSE archives through the loop-guard detector
//! pipeline and collects calibration metrics: recall on labeled hard/mild
//! loops, false positives on clean sources, time-to-detection (token index and
//! latency), and a prevented-token estimate.
//!
//! All detectors are exercised in their offline (non-streaming) form: each
//! [`ReplayRecord`] is replayed event-by-event through a fresh
//! [`ChannelizedLoopDetector`] and (when tool events are present) a
//! [`ToolLoopDetector`]. No model calls or network I/O happens here.
//!
//! [`ChannelizedLoopDetector`]: crate::loop_detector::ChannelizedLoopDetector
//! [`ToolLoopDetector`]: crate::loop_detector::ToolLoopDetector

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::loop_detector::{
    ChannelizedLoopDetector, DetectorEventKind, LoopDetector, LoopDetectorInput, LoopInputProfile,
    LoopSeverity, LoopSignal, StreamChannel, ToolCallFingerprintInput, ToolFingerprint,
    ToolLoopDetector,
};
use crate::settings::{LoopGuardConfig, LoopGuardMode};

/// Label severity for a [`ReplayRecord`].
pub const SEVERITY_HARD: &str = "hard";
/// Label severity for a [`ReplayRecord`].
pub const SEVERITY_MILD: &str = "mild";
/// Label severity for a [`ReplayRecord`].
pub const SEVERITY_NONE: &str = "none";

/// Stream channel carried by a recorded SSE event.
///
/// This mirrors [`StreamChannel`] but is serde-friendly for JSONL archives and
/// also models `ToolOutput`, which the live detector tracks separately via the
/// [`ToolLoopDetector`].
///
/// [`StreamChannel`]: crate::loop_detector::StreamChannel
/// [`ToolLoopDetector`]: crate::loop_detector::ToolLoopDetector
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayChannel {
    /// Hidden model reasoning / thinking deltas.
    Reasoning,
    /// Visible assistant content deltas.
    Content,
    /// Streamed tool or function argument fragments (or a complete tool call).
    ToolArgs,
    /// Completed tool output payload.
    ToolOutput,
}

impl ReplayChannel {
    /// Maps a replay channel to the live detector's [`StreamChannel`], when
    /// applicable. `ToolOutput` has no direct equivalent (it is fed to the
    /// [`ToolLoopDetector`] instead).
    ///
    /// [`ToolLoopDetector`]: crate::loop_detector::ToolLoopDetector
    #[must_use]
    pub const fn to_stream_channel(self) -> Option<StreamChannel> {
        match self {
            Self::Reasoning => Some(StreamChannel::Reasoning),
            Self::Content => Some(StreamChannel::Content),
            Self::ToolArgs => Some(StreamChannel::ToolArguments),
            Self::ToolOutput => None,
        }
    }
}

/// One recorded SSE event in a [`ReplayRecord`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SseEvent {
    /// Channel this event belongs to.
    pub channel: ReplayChannel,
    /// Event text payload. For reasoning/content this is the delta fragment;
    /// for `ToolArgs` this is either streamed fragments or a complete JSON
    /// payload; for `ToolOutput` it is the completed output body.
    pub text: String,
    /// Token offset at which this event was emitted (cumulative generated tokens).
    pub token_offset: u64,
}

/// A single labeled recording to be replayed through the detectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayRecord {
    /// Stable identifier for this record.
    pub record_id: String,
    /// Provenance/source bucket, e.g. `"stress_run"`, `"ab_forced_no_thinking"`.
    pub source: String,
    /// Whether this record was human-labeled as a loop.
    pub is_labeled_loop: bool,
    /// Labeled severity: `"hard"`, `"mild"`, or `"none"`.
    pub label_severity: Option<String>,
    /// Ordered recorded SSE events.
    pub sse_events: Vec<SseEvent>,
    /// Total tokens generated for this record.
    pub generated_token_count: u64,
}

impl ReplayRecord {
    /// Returns the label severity normalized to lowercase, defaulting to
    /// `"none"` when unset.
    #[must_use]
    pub fn severity_label(&self) -> &str {
        match &self.label_severity {
            Some(label) if label.eq_ignore_ascii_case(SEVERITY_HARD) => SEVERITY_HARD,
            Some(label) if label.eq_ignore_ascii_case(SEVERITY_MILD) => SEVERITY_MILD,
            _ => SEVERITY_NONE,
        }
    }
}

/// Per-source calibration bucket.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceCalibration {
    /// Source bucket name.
    pub source: String,
    /// Total records in this source.
    pub total: usize,
    /// Records detected as loops.
    pub detected: usize,
    /// False positives (detected but not labeled as loops).
    pub false_positives: usize,
    /// Recall: detected labeled loops / total labeled loops.
    pub recall: f32,
    /// Precision: true positives / total detections.
    pub precision: f32,
}

impl SourceCalibration {
    /// Builds a per-source bucket from raw counts, computing recall and
    /// precision. Labeled-loop count is `total_labeled`; detected-and-labeled
    /// (true positives) is `true_positives`.
    #[must_use]
    pub fn from_counts(
        source: impl Into<String>,
        total: usize,
        true_positives: usize,
        false_positives: usize,
        total_labeled: usize,
    ) -> Self {
        let source = source.into();
        let detected = true_positives + false_positives;
        let recall = ratio(true_positives, total_labeled);
        let precision = ratio(true_positives, detected);
        Self {
            source,
            total,
            detected,
            false_positives,
            recall,
            precision,
        }
    }
}

/// Aggregated calibration metrics for an entire archive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Total records replayed.
    pub total_records: usize,
    /// Records labeled as hard loops.
    pub labeled_hard_loops: usize,
    /// Records labeled as mild loops.
    pub labeled_mild_loops: usize,
    /// Labeled hard loops that were detected.
    pub detected_hard_loops: usize,
    /// Labeled mild loops that were detected.
    pub detected_mild_loops: usize,
    /// Detections on non-loop (clean) records.
    pub false_positives: usize,
    /// Clean records correctly not flagged.
    pub true_negatives: usize,
    /// Recall over labeled hard loops.
    pub hard_loop_recall: f32,
    /// Recall over labeled mild loops.
    pub mild_loop_recall: f32,
    /// False positives / (false positives + true negatives).
    pub false_positive_rate: f32,
    /// Median generated-token index at first detection across true positives.
    pub median_time_to_detection_tokens: u64,
    /// Median wall-clock milliseconds at first detection across true positives.
    pub median_time_to_detection_ms: u64,
    /// Sum of tokens that would have been prevented by aborting at first
    /// detection across all true positives.
    pub prevented_token_estimate: u64,
    /// Per-source breakdown.
    pub per_source: Vec<SourceCalibration>,
}

impl CalibrationResult {
    /// Aggregates per-record detection results into a [`CalibrationResult`].
    ///
    /// `results` and `records` must be parallel (same length and order). Each
    /// result is matched to its record by index.
    #[must_use]
    pub fn from_results(results: &[RecordDetectionResult], records: &[ReplayRecord]) -> Self {
        let n = results.len().min(records.len());
        let mut labeled_hard_loops = 0_usize;
        let mut labeled_mild_loops = 0_usize;
        let mut detected_hard_loops = 0_usize;
        let mut detected_mild_loops = 0_usize;
        let mut false_positives = 0_usize;
        let mut true_negatives = 0_usize;
        let mut prevented_token_estimate = 0_u64;

        let mut source_buckets: std::collections::BTreeMap<&str, SourceAccum> =
            std::collections::BTreeMap::new();

        for (result, record) in results.iter().zip(records.iter()).take(n) {
            let severity = record.severity_label();
            let bucket = source_buckets.entry(record.source.as_str()).or_default();
            bucket.total += 1;
            if record.is_labeled_loop {
                bucket.total_labeled += 1;
            }

            if result.is_true_positive {
                bucket.true_positives += 1;
                prevented_token_estimate =
                    prevented_token_estimate.saturating_add(result.prevented_tokens);
                match severity {
                    SEVERITY_HARD => detected_hard_loops += 1,
                    _ => detected_mild_loops += 1,
                }
            } else if result.is_false_positive {
                false_positives += 1;
                bucket.false_positives += 1;
            } else if !record.is_labeled_loop {
                true_negatives += 1;
            }

            match severity {
                SEVERITY_HARD => labeled_hard_loops += 1,
                SEVERITY_MILD => labeled_mild_loops += 1,
                _ => {}
            }
        }

        let hard_loop_recall = ratio(detected_hard_loops, labeled_hard_loops);
        let mild_loop_recall = ratio(detected_mild_loops, labeled_mild_loops);
        let false_positive_rate = ratio(false_positives, false_positives + true_negatives);

        // Median time-to-detection combines hard + mild true positives.
        let mut all_detection_tokens: Vec<u64> = Vec::new();
        let mut all_detection_ms: Vec<u64> = Vec::new();
        for result in results.iter().take(n) {
            if result.is_true_positive {
                if let Some(token) = result.first_detection_token {
                    all_detection_tokens.push(token);
                }
                all_detection_ms.push(result.first_detection_time_ms);
            }
        }
        let median_time_to_detection_tokens = median_u64(&all_detection_tokens);
        let median_time_to_detection_ms = median_u64(&all_detection_ms);

        let per_source = source_buckets
            .into_iter()
            .map(|(source, acc)| {
                SourceCalibration::from_counts(
                    source,
                    acc.total,
                    acc.true_positives,
                    acc.false_positives,
                    acc.total_labeled,
                )
            })
            .collect();

        Self {
            total_records: n,
            labeled_hard_loops,
            labeled_mild_loops,
            detected_hard_loops,
            detected_mild_loops,
            false_positives,
            true_negatives,
            hard_loop_recall,
            mild_loop_recall,
            false_positive_rate,
            median_time_to_detection_tokens,
            median_time_to_detection_ms,
            prevented_token_estimate,
            per_source,
        }
    }

    /// Returns `true` when the calibration meets all configured quality targets.
    ///
    /// Targets:
    /// - hard-loop recall ≥ `hard_loop_recall_target` (skipped when no hard loops labeled)
    /// - mild-loop recall ≥ `mild_loop_recall_target` (skipped when no mild loops labeled)
    /// - false-positive rate ≤ `false_positive_target`
    #[must_use]
    pub fn meets_targets(&self, config: &ReplayConfig) -> bool {
        let hard_ok =
            self.labeled_hard_loops == 0 || self.hard_loop_recall >= config.hard_loop_recall_target;
        let mild_ok =
            self.labeled_mild_loops == 0 || self.mild_loop_recall >= config.mild_loop_recall_target;
        hard_ok && mild_ok && self.false_positive_rate <= config.false_positive_target
    }

    /// Returns a human-readable summary of the calibration.
    #[must_use]
    pub fn summary_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "replay calibration: {} records ({} hard, {} mild labeled)",
            self.total_records, self.labeled_hard_loops, self.labeled_mild_loops
        );
        let _ = writeln!(
            out,
            "  hard_loop_recall={:.3} detected_hard={}/{}",
            self.hard_loop_recall, self.detected_hard_loops, self.labeled_hard_loops
        );
        let _ = writeln!(
            out,
            "  mild_loop_recall={:.3} detected_mild={}/{}",
            self.mild_loop_recall, self.detected_mild_loops, self.labeled_mild_loops
        );
        let _ = writeln!(
            out,
            "  false_positive_rate={:.3} false_positives={} true_negatives={}",
            self.false_positive_rate, self.false_positives, self.true_negatives
        );
        let _ = writeln!(
            out,
            "  median_time_to_detection: {} tokens / {} ms",
            self.median_time_to_detection_tokens, self.median_time_to_detection_ms
        );
        let _ = writeln!(
            out,
            "  prevented_token_estimate={}",
            self.prevented_token_estimate
        );
        for source in &self.per_source {
            let _ = writeln!(
                out,
                "  source `{}`: total={} detected={} fp={} recall={:.3} precision={:.3}",
                source.source,
                source.total,
                source.detected,
                source.false_positives,
                source.recall,
                source.precision
            );
        }
        out.trim_end().to_owned()
    }
}

/// Quality targets for replay calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Minimum acceptable hard-loop recall.
    pub hard_loop_recall_target: f32,
    /// Minimum acceptable mild-loop recall.
    pub mild_loop_recall_target: f32,
    /// Token budget within which hard loops should be detected.
    pub hard_loop_token_budget: u64,
    /// Token budget within which mild loops should be detected.
    pub mild_loop_token_budget: u64,
    /// Maximum acceptable false-positive rate on clean sources.
    pub false_positive_target: f32,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            hard_loop_recall_target: 0.95,
            mild_loop_recall_target: 0.80,
            hard_loop_token_budget: 12_000,
            mild_loop_token_budget: 16_000,
            false_positive_target: 0.01,
        }
    }
}

/// Per-record detection outcome produced by [`ReplayRunner::run_record`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordDetectionResult {
    /// Record identifier this result applies to.
    pub record_id: String,
    /// Whether any detector fired.
    pub detected: bool,
    /// Detection severity label (`"hard"` / `"mild"`), inferred from the
    /// strongest signal observed.
    pub detection_severity: Option<String>,
    /// Token offset of the first detection signal, if any.
    pub first_detection_token: Option<u64>,
    /// Wall-clock milliseconds elapsed at first detection within the record.
    pub first_detection_time_ms: u64,
    /// Estimated tokens prevented by aborting at first detection.
    pub prevented_tokens: u64,
    /// Stable reason-code labels of all signals that fired.
    pub signals: Vec<String>,
    /// True when detected and the record was labeled as a loop.
    pub is_true_positive: bool,
    /// True when detected but the record was not labeled as a loop.
    pub is_false_positive: bool,
}

/// Offline detector runner that replays a single [`ReplayRecord`] through the
/// loop-guard detector pipeline.
pub struct ReplayRunner {
    config: ReplayConfig,
}

impl ReplayRunner {
    /// Creates a new runner with the given calibration config.
    #[must_use]
    pub fn new(config: ReplayConfig) -> Self {
        Self { config }
    }

    /// Returns the calibration config.
    #[must_use]
    pub const fn config(&self) -> &ReplayConfig {
        &self.config
    }

    /// Run all detectors over a single replay record.
    ///
    /// Returns the detection result with timing and prevented-token estimate.
    /// This is synchronous and performs no network I/O.
    #[must_use]
    pub fn run_record(&self, record: &ReplayRecord) -> RecordDetectionResult {
        let start = Instant::now();
        // Use enforce-mode defaults for calibration so thresholds match abort behavior.
        let detector_config = LoopGuardConfig {
            mode: LoopGuardMode::Enforce,
            ..LoopGuardConfig::default()
        };
        let mut detector =
            ChannelizedLoopDetector::new(detector_config.clone(), LoopInputProfile::default());
        let mut tool_detector = ToolLoopDetector::new(16);

        let mut first_detection_token: Option<u64> = None;
        let mut first_detection_time_ms: u64 = 0;
        let mut max_severity: Option<LoopSeverity> = None;
        let mut signals: Vec<String> = Vec::new();

        for event in &record.sse_events {
            let emitted = observe_event(
                event,
                &mut detector,
                &mut tool_detector,
                &mut signals,
                &mut max_severity,
            );
            if emitted && first_detection_token.is_none() {
                first_detection_token = Some(event.token_offset);
                first_detection_time_ms =
                    u64::try_from(start.elapsed().as_millis().min(u128::from(u64::MAX)))
                        .unwrap_or(u64::MAX);
            }
        }

        let detected = first_detection_token.is_some();
        let detection_severity = max_severity.map(|severity| match severity {
            LoopSeverity::AbortCandidate => String::from(SEVERITY_HARD),
            LoopSeverity::Suspect | LoopSeverity::Observe => String::from(SEVERITY_MILD),
        });

        let prevented_tokens = match first_detection_token {
            Some(token) => record.generated_token_count.saturating_sub(token),
            None => 0,
        };

        let is_true_positive = detected && record.is_labeled_loop;
        let is_false_positive = detected && !record.is_labeled_loop;

        RecordDetectionResult {
            record_id: record.record_id.clone(),
            detected,
            detection_severity,
            first_detection_token,
            first_detection_time_ms,
            prevented_tokens,
            signals,
            is_true_positive,
            is_false_positive,
        }
    }
}

/// Feeds one event through the appropriate detector, appending reason-code
/// labels to `signals` and updating `max_severity`. Returns `true` when at
/// least one signal was emitted for this event.
fn observe_event(
    event: &SseEvent,
    detector: &mut ChannelizedLoopDetector,
    tool_detector: &mut ToolLoopDetector,
    signals: &mut Vec<String>,
    max_severity: &mut Option<LoopSeverity>,
) -> bool {
    match event.channel {
        ReplayChannel::ToolArgs => {
            // Try as a complete tool call first (deterministic detector).
            let mut emitted = false;
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&event.text) {
                if json_value.is_object() || json_value.is_array() {
                    let tool_signals = detector.observe_tool_call(ToolCallFingerprintInput {
                        tool_name: "replay_tool",
                        arguments: &event.text,
                    });
                    for signal in &tool_signals {
                        push_signal(signal, signals, max_severity);
                    }
                    emitted |= !tool_signals.is_empty();
                }
            }
            // Also feed as a fragment so streaming repetition is caught.
            let frag_signals = detector.observe(LoopDetectorInput::fragment(
                StreamChannel::ToolArguments,
                &event.text,
            ));
            for signal in &frag_signals {
                push_signal(signal, signals, max_severity);
            }
            emitted |= !frag_signals.is_empty();

            // And feed the semantic tool-loop detector.
            let fingerprint = ToolFingerprint::from_call("replay_tool", &event.text);
            let tool_loop_signals = tool_detector.observe_fingerprint(fingerprint);
            for signal in &tool_loop_signals {
                let label = format!("tool:{}", signal.reason_code.as_str());
                if !signals.contains(&label) {
                    signals.push(label);
                }
            }
            if let Some(strongest) = tool_loop_signals
                .iter()
                .map(|s| severity_from_risk(s.risk))
                .max()
            {
                update_max_severity(max_severity, strongest);
                emitted = true;
            }
            emitted
        }
        ReplayChannel::ToolOutput => {
            // Hash the output body and feed the semantic tool-output detector.
            let hash = stable_text_hash(&event.text);
            let tool_loop_signals = tool_detector.observe_tool_output(hash);
            for signal in &tool_loop_signals {
                let label = format!("tool:{}", signal.reason_code.as_str());
                if !signals.contains(&label) {
                    signals.push(label);
                }
            }
            let emitted = !tool_loop_signals.is_empty();
            if let Some(strongest) = tool_loop_signals
                .iter()
                .map(|s| severity_from_risk(s.risk))
                .max()
            {
                update_max_severity(max_severity, strongest);
            }
            emitted
        }
        ReplayChannel::Reasoning | ReplayChannel::Content => {
            let channel = event
                .channel
                .to_stream_channel()
                .unwrap_or(StreamChannel::Content);
            let emitted_signals =
                detector.observe(LoopDetectorInput::fragment(channel, &event.text));
            for signal in &emitted_signals {
                push_signal(signal, signals, max_severity);
            }
            // Also issue a boundary check to flush line/token-window state.
            let boundary_signals = detector.observe(LoopDetectorInput::event(
                channel,
                DetectorEventKind::Boundary,
            ));
            for signal in &boundary_signals {
                push_signal(signal, signals, max_severity);
            }
            !emitted_signals.is_empty() || !boundary_signals.is_empty()
        }
    }
}

/// Maps a tool-loop detector risk score to a severity.
fn severity_from_risk(risk: f64) -> LoopSeverity {
    if risk >= 0.75 {
        LoopSeverity::AbortCandidate
    } else {
        LoopSeverity::Suspect
    }
}

fn push_signal(
    signal: &LoopSignal,
    signals: &mut Vec<String>,
    max_severity: &mut Option<LoopSeverity>,
) {
    let label = format!(
        "{}:{}",
        signal.channel.as_str(),
        signal.reason_code.as_str()
    );
    if !signals.contains(&label) {
        signals.push(label);
    }
    update_max_severity(max_severity, signal.severity);
}

fn update_max_severity(current: &mut Option<LoopSeverity>, candidate: LoopSeverity) {
    match current {
        Some(existing) if *existing >= candidate => {}
        _ => *current = Some(candidate),
    }
}

/// Computes a stable FNV-1a 64-bit hash of `text` for tool-output dedup.
fn stable_text_hash(text: &str) -> u64 {
    const FNV64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV64_OFFSET_BASIS;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

/// Returns `numerator / denominator` as an `f32`, or `0.0` when denominator is zero.
fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let value = numerator as f32 / denominator as f32;
    value
}

/// Returns the median of a slice of `u64`, or `0` when empty. Mutates the
/// slice via sort.
fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u64> = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        // Average of two middle elements (rounded down).
        sorted[mid].saturating_add(sorted[mid - 1]) / 2
    } else {
        sorted[mid]
    }
}

/// Accumulator used while building per-source buckets.
#[derive(Default)]
struct SourceAccum {
    total: usize,
    total_labeled: usize,
    true_positives: usize,
    false_positives: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a synthetic hard-loop record: the same reasoning paragraph
    /// repeated many times.
    fn hard_loop_record() -> ReplayRecord {
        let paragraph = "I need to reconsider the approach to this problem carefully. ";
        let mut events = Vec::new();
        for offset in 0..20_u64 {
            events.push(SseEvent {
                channel: ReplayChannel::Reasoning,
                text: paragraph.to_owned(),
                token_offset: offset * 8,
            });
        }
        ReplayRecord {
            record_id: String::from("hard-1"),
            source: String::from("stress_run"),
            is_labeled_loop: true,
            label_severity: Some(String::from(SEVERITY_HARD)),
            sse_events: events,
            generated_token_count: 160,
        }
    }

    /// Builds a clean record with diverse, non-repeating content.
    fn clean_record() -> ReplayRecord {
        let mut events = Vec::new();
        let diverse = [
            "The quick brown fox jumps over the lazy dog. ",
            "Pack my box with five dozen liquor jugs. ",
            "How vexingly quick daft zebras jump! ",
            "Sphinx of black quartz, judge my vow. ",
            "The five boxing wizards jump quickly. ",
        ];
        for (index, text) in diverse.iter().enumerate() {
            events.push(SseEvent {
                channel: ReplayChannel::Reasoning,
                text: (*text).to_owned(),
                token_offset: (index as u64) * 10,
            });
        }
        ReplayRecord {
            record_id: String::from("clean-1"),
            source: String::from("ab_forced_no_thinking"),
            is_labeled_loop: false,
            label_severity: Some(String::from(SEVERITY_NONE)),
            sse_events: events,
            generated_token_count: 50,
        }
    }

    #[test]
    fn runner_detects_synthetic_hard_loop() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let record = hard_loop_record();
        let result = runner.run_record(&record);

        assert!(
            result.detected,
            "hard loop should be detected, signals: {:?}",
            result.signals
        );
        assert!(result.is_true_positive, "hard loop is a true positive");
        assert!(!result.is_false_positive);
        assert!(
            result.first_detection_token.is_some(),
            "first detection token should be set"
        );
        // The repeated-paragraph pattern should produce at least one signal.
        assert!(
            !result.signals.is_empty(),
            "at least one signal should fire"
        );
    }

    #[test]
    fn runner_does_not_flag_clean_record() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let record = clean_record();
        let result = runner.run_record(&record);

        assert!(
            !result.detected,
            "clean record should not be flagged, signals: {:?}",
            result.signals
        );
        assert!(!result.is_true_positive);
        assert!(!result.is_false_positive);
        assert_eq!(result.prevented_tokens, 0);
    }

    #[test]
    fn calibration_result_computes_recall_and_fpr() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let hard = hard_loop_record();
        let clean = clean_record();

        let results = vec![runner.run_record(&hard), runner.run_record(&clean)];
        let records = vec![hard, clean];
        let calibration = CalibrationResult::from_results(&results, &records);

        assert_eq!(calibration.total_records, 2);
        assert_eq!(calibration.labeled_hard_loops, 1);
        assert_eq!(calibration.detected_hard_loops, 1);
        assert_eq!(calibration.false_positives, 0);
        assert_eq!(calibration.true_negatives, 1);
        // Recall should be 1.0 (1/1 detected), FPR should be 0.0.
        assert!(
            (calibration.hard_loop_recall - 1.0).abs() < f32::EPSILON,
            "hard recall should be 1.0, got {}",
            calibration.hard_loop_recall
        );
        assert!(
            calibration.false_positive_rate.abs() < f32::EPSILON,
            "FPR should be 0.0, got {}",
            calibration.false_positive_rate
        );
    }

    #[test]
    fn calibration_result_meets_targets_when_recall_high_and_fpr_zero() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let hard = hard_loop_record();
        let clean = clean_record();

        let results = vec![runner.run_record(&hard), runner.run_record(&clean)];
        let records = vec![hard, clean];
        let calibration = CalibrationResult::from_results(&results, &records);

        assert!(
            calibration.meets_targets(&ReplayConfig::default()),
            "calibration should meet default targets (recall=1.0, fpr=0.0)"
        );
    }

    #[test]
    fn calibration_result_does_not_meet_targets_when_fp_present() {
        // Manually construct a result with one false positive on a clean record.
        let records = vec![ReplayRecord {
            record_id: String::from("clean-fp"),
            source: String::from("clean"),
            is_labeled_loop: false,
            label_severity: Some(String::from(SEVERITY_NONE)),
            sse_events: Vec::new(),
            generated_token_count: 10,
        }];
        let results = vec![RecordDetectionResult {
            record_id: String::from("clean-fp"),
            detected: true,
            detection_severity: Some(String::from(SEVERITY_HARD)),
            first_detection_token: Some(5),
            first_detection_time_ms: 1,
            prevented_tokens: 5,
            signals: vec![String::from("content:repeated_line")],
            is_true_positive: false,
            is_false_positive: true,
        }];
        let calibration = CalibrationResult::from_results(&results, &records);
        // 0 labeled loops → recall 0.0 < target, and 1 FP → FPR 1.0 > target.
        assert!(
            !calibration.meets_targets(&ReplayConfig::default()),
            "calibration with a false positive should not meet targets"
        );
    }

    #[test]
    fn calibration_result_summary_text_is_non_empty() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let record = hard_loop_record();
        let results = vec![runner.run_record(&record)];
        let records = vec![record];
        let calibration = CalibrationResult::from_results(&results, &records);
        let summary = calibration.summary_text();
        assert!(!summary.is_empty());
        assert!(
            summary.contains("replay calibration"),
            "summary should contain header: {summary}"
        );
    }

    #[test]
    fn source_calibration_computes_recall_and_precision() {
        // 10 total, 5 labeled loops, 4 detected (true positives), 1 false positive.
        let source = SourceCalibration::from_counts("stress_run", 10, 4, 1, 5);
        assert_eq!(source.total, 10);
        assert_eq!(source.detected, 5);
        assert_eq!(source.false_positives, 1);
        assert!(
            (source.recall - 0.8).abs() < 0.001,
            "recall should be 4/5 = 0.8, got {}",
            source.recall
        );
        assert!(
            (source.precision - 0.8).abs() < 0.001,
            "precision should be 4/5 = 0.8, got {}",
            source.precision
        );
    }

    #[test]
    fn source_calibration_zero_labeled_gives_zero_recall() {
        let source = SourceCalibration::from_counts("clean", 5, 0, 0, 0);
        assert!(source.recall.abs() < f32::EPSILON);
        assert!(source.precision.abs() < f32::EPSILON);
    }

    #[test]
    fn replay_record_severity_label_normalizes() {
        let mut record = ReplayRecord {
            record_id: String::from("x"),
            source: String::from("s"),
            is_labeled_loop: true,
            label_severity: Some(String::from("HARD")),
            sse_events: Vec::new(),
            generated_token_count: 0,
        };
        assert_eq!(record.severity_label(), SEVERITY_HARD);
        record.label_severity = Some(String::from("Mild"));
        assert_eq!(record.severity_label(), SEVERITY_MILD);
        record.label_severity = None;
        assert_eq!(record.severity_label(), SEVERITY_NONE);
    }

    #[test]
    fn replay_channel_maps_to_stream_channel() {
        assert_eq!(
            ReplayChannel::Reasoning.to_stream_channel(),
            Some(StreamChannel::Reasoning)
        );
        assert_eq!(
            ReplayChannel::Content.to_stream_channel(),
            Some(StreamChannel::Content)
        );
        assert_eq!(
            ReplayChannel::ToolArgs.to_stream_channel(),
            Some(StreamChannel::ToolArguments)
        );
        assert_eq!(ReplayChannel::ToolOutput.to_stream_channel(), None);
    }

    #[test]
    fn prevented_tokens_subtract_first_detection_offset() {
        let runner = ReplayRunner::new(ReplayConfig::default());
        let record = hard_loop_record();
        let result = runner.run_record(&record);
        let expected = record
            .generated_token_count
            .saturating_sub(result.first_detection_token.unwrap_or(0));
        assert_eq!(result.prevented_tokens, expected);
    }
}
