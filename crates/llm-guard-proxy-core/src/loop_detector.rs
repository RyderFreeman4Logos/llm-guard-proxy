use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::Value;

use crate::settings::{LoopGuardConfig, LoopGuardMode};

const LOOP_MIN_LINE_CHARS: usize = 8;
const LOOP_MAX_PENDING_LINE_BYTES: usize = 8 * 1024;
const LOOP_MAX_RECENT_CHARS: usize = 4 * 1024;
const LOOP_MAX_TOKEN_BYTES: usize = 128;
const LOOP_MAX_SEMANTIC_TOKEN_BYTES: usize = 64;
const LOOP_SUFFIX_MIN_UNIT_CHARS: usize = 4;
const LOOP_SUFFIX_MAX_UNIT_CHARS: usize = 64;
const LOOP_INPUT_LINE_COUNT_CAP: usize = 4_096;
const LOOP_INPUT_TOKEN_WINDOW_COUNT_CAP: usize = 8_192;
const LOOP_OUTPUT_LINE_COUNT_CAP: usize = 4_096;
const LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP: usize = 8_192;
const LOOP_OUTPUT_UNIQUE_TOKEN_WINDOW_CAP: usize = 8_192;
const LOOP_OUTPUT_SEMANTIC_WINDOW_CAP: usize = 256;
const LOOP_ARGUMENT_HASH_COUNT_CAP: usize = 1_024;
const LOOP_FINGERPRINT_COUNT_CAP: usize = 1_024;
const LOOP_SUMMARY_SIGNAL_LIMIT: usize = 8;
const LOOP_FEATURE_LIMIT: usize = 16;
const LOOP_FEATURE_VALUE_MAX_CHARS: usize = 96;
const FNV64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Minimum occurrences of one tool fingerprint within the window before a
/// [`LoopReasonCode::ToolFingerprintRepeat`] signal fires.
const TOOL_FINGERPRINT_REPEAT_THRESHOLD: u32 = 3;
/// Minimum occurrences of one tool output hash before a
/// [`LoopReasonCode::ToolOutputBlockedEcho`] signal fires.
const TOOL_OUTPUT_BLOCKED_THRESHOLD: u32 = 2;
/// Length of a two-fingerprint alternation cycle (A-B-A-B).
const TOOL_ALTERNATION_CYCLE_LENGTH: usize = 4;
/// Risk score assigned to a completed two-fingerprint alternation cycle.
const TOOL_ALTERNATION_RISK: f64 = 0.75;

/// Independent stream channel observed by the loop detector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamChannel {
    /// Hidden model reasoning, thinking, or reasoning-content deltas.
    Reasoning,
    /// Visible assistant content deltas.
    Content,
    /// Streamed tool or function argument fragments.
    ToolArguments,
    /// Completed tool name plus canonicalized argument JSON.
    ToolFingerprint,
}

impl StreamChannel {
    /// Returns the stable metadata label for this channel.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Content => "content",
            Self::ToolArguments => "tool_arguments",
            Self::ToolFingerprint => "tool_fingerprint",
        }
    }
}

/// Detector event kind, independent of HTTP and SSE framing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorEventKind {
    /// A stream fragment was appended.
    Fragment,
    /// A caller-requested detection check.
    Check,
    /// A periodic active-stream timer tick.
    Tick,
    /// A punctuation, paragraph, code-block, or similar boundary.
    Boundary,
    /// A complete tool-argument JSON payload was observed.
    CompletedJson,
    /// A complete tool name plus canonical argument payload was observed.
    CompletedToolFingerprint,
}

impl DetectorEventKind {
    /// Returns the stable metadata label for this event kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fragment => "fragment",
            Self::Check => "check",
            Self::Tick => "tick",
            Self::Boundary => "boundary",
            Self::CompletedJson => "completed_json",
            Self::CompletedToolFingerprint => "completed_tool_fingerprint",
        }
    }
}

/// Loop signal severity used by decision policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum LoopSeverity {
    /// Bounded diagnostic evidence only.
    Observe,
    /// Suspicious but not sufficient for abort by itself.
    Suspect,
    /// High-confidence candidate that enforce mode may abort.
    AbortCandidate,
}

impl LoopSeverity {
    /// Returns the stable metadata label for this severity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Suspect => "suspect",
            Self::AbortCandidate => "abort_candidate",
        }
    }
}

/// Stable reason code for a content-free loop signal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopReasonCode {
    /// A normalized line hash repeated past its threshold.
    RepeatedLine,
    /// A normalized token-window hash repeated past its threshold.
    RepeatedTokenWindow,
    /// A repeated suffix cycle was found in recent characters.
    SuffixCycle,
    /// Bytes grew while unique token windows stayed low.
    LowProgressGrowth,
    /// Approximate Jaccard-compatible reasoning repetition placeholder.
    ApproximateRepetition,
    /// A complete tool-argument JSON value was observed.
    ToolArgumentsJsonCompleted,
    /// The same canonical tool-argument JSON repeated.
    ToolArgumentsRepeatedJson,
    /// The same tool name and canonical arguments repeated.
    ToolFingerprintRepeated,
    /// Tool arguments did not parse as complete JSON at completion time.
    ToolArgumentsInvalidJson,
    /// A single tool fingerprint repeated past its window threshold.
    ToolFingerprintRepeat,
    /// Two distinct tool fingerprints alternating in a fixed cycle.
    ToolAlternationCycle,
    /// The same tool output hash repeated (blocked/unchanged output).
    ToolOutputBlockedEcho,
}

impl LoopReasonCode {
    /// Returns the stable metadata label for this reason code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RepeatedLine => "repeated_line",
            Self::RepeatedTokenWindow => "repeated_token_window",
            Self::SuffixCycle => "suffix_cycle",
            Self::LowProgressGrowth => "low_progress_growth",
            Self::ApproximateRepetition => "approximate_repetition",
            Self::ToolArgumentsJsonCompleted => "tool_arguments_json_completed",
            Self::ToolArgumentsRepeatedJson => "tool_arguments_repeated_json",
            Self::ToolFingerprintRepeated => "tool_fingerprint_repeated",
            Self::ToolArgumentsInvalidJson => "tool_arguments_invalid_json",
            Self::ToolFingerprintRepeat => "tool_fingerprint_repeat",
            Self::ToolAlternationCycle => "tool_alternation_cycle",
            Self::ToolOutputBlockedEcho => "tool_output_blocked_echo",
        }
    }

    pub(crate) fn legacy_signal(self) -> &'static str {
        match self {
            Self::ApproximateRepetition => "semantic_jaccard",
            other => other.as_str(),
        }
    }
}

/// Content-free bounded feature map for one loop signal.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoundedFeatureSummary {
    fields: BTreeMap<String, String>,
    capped: u64,
}

impl BoundedFeatureSummary {
    /// Creates an empty bounded feature summary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fields: BTreeMap::new(),
            capped: 0,
        }
    }

    /// Inserts a generated string feature after bounding key and value size.
    pub fn insert_str(&mut self, key: &'static str, value: impl Into<String>) {
        if self.fields.len() >= LOOP_FEATURE_LIMIT {
            self.capped = self.capped.saturating_add(1);
            return;
        }
        let value = truncate_chars(value.into(), LOOP_FEATURE_VALUE_MAX_CHARS);
        self.fields.insert(key.to_owned(), value);
    }

    /// Inserts a generated integer feature.
    pub fn insert_u64(&mut self, key: &'static str, value: u64) {
        self.insert_str(key, value.to_string());
    }

    /// Inserts a generated boolean feature.
    pub fn insert_bool(&mut self, key: &'static str, value: bool) {
        self.insert_str(key, value.to_string());
    }

    /// Returns bounded fields without exposing raw stream content.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<String, String> {
        &self.fields
    }

    /// Returns the number of feature fields skipped due to the cap.
    #[must_use]
    pub const fn capped(&self) -> u64 {
        self.capped
    }
}

/// One detector input event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopDetectorInput<'a> {
    /// Channel receiving this event.
    pub channel: StreamChannel,
    /// Event kind that triggered this observation.
    pub event_kind: DetectorEventKind,
    /// Appended fragment for fragment events. Non-fragment events may use an empty string.
    pub fragment: &'a str,
}

impl<'a> LoopDetectorInput<'a> {
    /// Creates a stream-fragment detector input.
    #[must_use]
    pub const fn fragment(channel: StreamChannel, fragment: &'a str) -> Self {
        Self {
            channel,
            event_kind: DetectorEventKind::Fragment,
            fragment,
        }
    }

    /// Creates an explicit non-fragment detector input.
    #[must_use]
    pub const fn event(channel: StreamChannel, event_kind: DetectorEventKind) -> Self {
        Self {
            channel,
            event_kind,
            fragment: "",
        }
    }
}

/// Completed tool-call input used to derive argument and fingerprint features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolCallFingerprintInput<'a> {
    /// Tool or function name. The detector stores only a hash of this value.
    pub tool_name: &'a str,
    /// Completed tool arguments. The detector stores only hashes and parse status.
    pub arguments: &'a str,
}

/// Normalized fingerprint of one completed tool call.
///
/// The `canonical_args_hash` folds file paths, line ranges, and similar
/// volatile fields so that semantically equivalent calls collapse to the same
/// fingerprint. The `fingerprint_hash` combines the tool name with the
/// canonical argument hash so two different tools sharing the same arguments
/// still produce distinct fingerprints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolFingerprint {
    /// Tool or function name. Stored for diagnostics only.
    pub tool_name: String,
    /// Hash of tool arguments after canonicalization (file paths/ranges normalized).
    pub canonical_args_hash: u64,
    /// Hash of `tool_name + canonical_args_hash`.
    pub fingerprint_hash: u64,
}

impl ToolFingerprint {
    /// Builds a fingerprint from a tool name and raw argument JSON, normalizing
    /// paths and ranges before hashing. If `arguments` is not valid JSON the
    /// raw bytes are hashed directly so the call is still observable.
    #[must_use]
    pub fn from_call(tool_name: &str, arguments: &str) -> Self {
        let canonical_args_hash = canonical_tool_arguments_hash(arguments);
        let fingerprint_hash =
            stable_hash_u64s([stable_hash(tool_name.as_bytes()), canonical_args_hash]);
        Self {
            tool_name: tool_name.to_owned(),
            canonical_args_hash,
            fingerprint_hash,
        }
    }

    /// Builds a fingerprint from pre-computed hashes. The `tool_name` is stored
    /// verbatim for diagnostics.
    #[must_use]
    pub fn from_hashes(tool_name: String, canonical_args_hash: u64, fingerprint_hash: u64) -> Self {
        Self {
            tool_name,
            canonical_args_hash,
            fingerprint_hash,
        }
    }
}

/// One tool-loop detection signal emitted by [`ToolLoopDetector`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolLoopSignal {
    /// Stable reason code identifying which detector fired.
    pub reason_code: LoopReasonCode,
    /// Fingerprint hash that triggered the signal, or `0` for output-based
    /// signals that are not tied to a single fingerprint.
    pub fingerprint_hash: u64,
    /// How many times the fingerprint/output repeated within the window.
    pub repeat_count: u32,
    /// Risk score in the range `0.0..=1.0`.
    pub risk: f64,
}

/// Dedicated detector for semantic tool loops: repeated fingerprints,
/// alternating two-tool cycles, and blocked (unchanged) tool output.
///
/// This complements [`ChannelizedLoopDetector`], which already tracks
/// hash-based tool-argument repetition. [`ToolLoopDetector`] catches semantic
/// loops such as the same file range read repeatedly, the same command rerun,
/// or two tools alternating in a fixed cycle (A-B-A-B).
#[derive(Debug)]
pub struct ToolLoopDetector {
    max_history: usize,
    fingerprints: VecDeque<u64>,
    fingerprint_counts: BTreeMap<u64, u32>,
    output_counts: BTreeMap<u64, u32>,
    signals: Vec<ToolLoopSignal>,
}

impl ToolLoopDetector {
    /// Creates a detector that retains at most `max_history` recent
    /// fingerprints for alternation detection.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            fingerprints: VecDeque::new(),
            fingerprint_counts: BTreeMap::new(),
            output_counts: BTreeMap::new(),
            signals: Vec::new(),
        }
    }

    /// Observes one completed tool-call fingerprint and returns any newly
    /// emitted signals.
    #[allow(clippy::needless_pass_by_value)]
    pub fn observe_fingerprint(&mut self, fingerprint: ToolFingerprint) -> Vec<ToolLoopSignal> {
        self.push_fingerprint(fingerprint.fingerprint_hash);

        let mut emitted = Vec::new();

        if let Some(count) = self
            .fingerprint_counts
            .get(&fingerprint.fingerprint_hash)
            .copied()
        {
            if count >= TOOL_FINGERPRINT_REPEAT_THRESHOLD {
                let risk = fingerprint_repeat_risk(count);
                let signal = ToolLoopSignal {
                    reason_code: LoopReasonCode::ToolFingerprintRepeat,
                    fingerprint_hash: fingerprint.fingerprint_hash,
                    repeat_count: count,
                    risk,
                };
                emitted.push(signal.clone());
                self.signals.push(signal);
            }
        }

        if let Some(signal) = self.detect_alternation() {
            emitted.push(signal.clone());
            self.signals.push(signal);
        }

        emitted
    }

    /// Observes one tool output hash and returns a signal when the same output
    /// repeats (blocked/unchanged output).
    pub fn observe_tool_output(&mut self, output_hash: u64) -> Vec<ToolLoopSignal> {
        let count = self
            .output_counts
            .entry(output_hash)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);

        if *count >= TOOL_OUTPUT_BLOCKED_THRESHOLD {
            let risk = output_blocked_risk(*count);
            let signal = ToolLoopSignal {
                reason_code: LoopReasonCode::ToolOutputBlockedEcho,
                fingerprint_hash: 0,
                repeat_count: *count,
                risk,
            };
            self.signals.push(signal.clone());
            return vec![signal];
        }

        Vec::new()
    }

    /// Returns all accumulated signals.
    #[must_use]
    pub fn signals(&self) -> &[ToolLoopSignal] {
        &self.signals
    }

    fn push_fingerprint(&mut self, fingerprint_hash: u64) {
        self.fingerprints.push_back(fingerprint_hash);
        while self.fingerprints.len() > self.max_history {
            if let Some(evicted) = self.fingerprints.pop_front() {
                if let Some(count) = self.fingerprint_counts.get_mut(&evicted) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.fingerprint_counts.remove(&evicted);
                    }
                }
            }
        }
        self.fingerprint_counts
            .entry(fingerprint_hash)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    fn detect_alternation(&self) -> Option<ToolLoopSignal> {
        if self.fingerprints.len() < TOOL_ALTERNATION_CYCLE_LENGTH {
            return None;
        }
        let n = self.fingerprints.len();
        // Check the last 4 entries for an A-B-A-B pattern with A != B.
        let first = self.fingerprints[n - 4];
        let second = self.fingerprints[n - 3];
        let third = self.fingerprints[n - 2];
        let fourth = self.fingerprints[n - 1];
        if first == third && second == fourth && first != second {
            Some(ToolLoopSignal {
                reason_code: LoopReasonCode::ToolAlternationCycle,
                fingerprint_hash: first,
                repeat_count: 2,
                risk: TOOL_ALTERNATION_RISK,
            })
        } else {
            None
        }
    }
}

fn fingerprint_repeat_risk(count: u32) -> f64 {
    // 3 repeats -> 0.6, then +0.1 per extra repeat, capped at 1.0.
    let extra = count.saturating_sub(TOOL_FINGERPRINT_REPEAT_THRESHOLD);
    (0.6 + f64::from(extra) * 0.1).min(1.0)
}

fn output_blocked_risk(count: u32) -> f64 {
    // 2 repeats -> 0.5, then +0.15 per extra repeat, capped at 1.0.
    let extra = count.saturating_sub(TOOL_OUTPUT_BLOCKED_THRESHOLD);
    (0.5 + f64::from(extra) * 0.15).min(1.0)
}

/// Computes a canonical hash of tool arguments, normalizing file paths, line
/// ranges, and other volatile fields so semantically equivalent calls collapse
/// to the same hash. Falls back to a raw byte hash if `arguments` is not valid
/// JSON.
fn canonical_tool_arguments_hash(arguments: &str) -> u64 {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        return stable_hash(arguments.as_bytes());
    };
    let mut canonical = String::new();
    write_canonical_tool_json(&value, &mut canonical);
    stable_hash(canonical.as_bytes())
}

/// Canonical JSON writer that additionally normalizes tool-specific volatile
/// fields (paths, line ranges, offsets) so repeated reads of the same logical
/// range collapse to identical hashes.
fn write_canonical_tool_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(value) => {
            let normalized = normalize_tool_argument_string(value);
            output.push_str(
                &serde_json::to_string(&normalized).unwrap_or_else(|_error| String::from("\"\"")),
            );
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_tool_json(value, output);
            }
            output.push(']');
        }
        Value::Object(object) => {
            output.push('{');
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).unwrap_or_else(|_error| String::from("\"\"")),
                );
                output.push(':');
                write_canonical_tool_json(value, output);
            }
            output.push('}');
        }
    }
}

/// Normalizes a tool argument string value by collapsing volatile path/range
/// fragments. Absolute paths become `<path>`, line/column numbers become
/// `<range>`.
fn normalize_tool_argument_string(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Absolute or relative filesystem paths containing separators.
    if (trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with('~')
        || trimmed.contains('\\'))
        && !trimmed.contains(' ')
    {
        return String::from("<path>");
    }

    // Bare line/range patterns like "12", "12-34", "12:34", "L12-L34".
    if is_line_range_pattern(trimmed) {
        return String::from("<range>");
    }

    trimmed.to_owned()
}

/// Returns true when `value` looks like a line/column range token.
fn is_line_range_pattern(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    // All characters must be digits or one of the range separators.
    if !value
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '-' | ':' | ',' | 'L'))
    {
        return false;
    }
    // Must contain at least one digit and at least one separator or leading L.
    value.chars().any(|c| c.is_ascii_digit())
        && (value.contains('-')
            || value.contains(':')
            || value.contains(',')
            || value.contains('L'))
}

/// Content-free detector signal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopSignal {
    /// Stream channel where the signal was observed.
    pub channel: StreamChannel,
    /// Event kind that produced the signal.
    pub event_kind: DetectorEventKind,
    /// Decision severity.
    pub severity: LoopSeverity,
    /// Bounded confidence from 0 to 100.
    pub confidence: u8,
    /// Stable reason code.
    pub reason_code: LoopReasonCode,
    /// Content-free bounded feature summary.
    pub feature_summary: BoundedFeatureSummary,
}

impl LoopSignal {
    /// Returns true when enforce mode may abort on this signal.
    #[must_use]
    pub const fn is_abort_candidate(&self) -> bool {
        matches!(self.severity, LoopSeverity::AbortCandidate)
    }

    /// Returns legacy `loop_*` metadata for abort paths.
    #[must_use]
    pub fn legacy_abort_metadata(&self) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::from([
            (String::from("loop_detected"), String::from("true")),
            (
                String::from("loop_signal"),
                self.reason_code.legacy_signal().to_owned(),
            ),
            (
                String::from("loop_channel"),
                self.channel.as_str().to_owned(),
            ),
            (
                String::from("loop_event_kind"),
                self.event_kind.as_str().to_owned(),
            ),
            (
                String::from("loop_severity"),
                self.severity.as_str().to_owned(),
            ),
            (String::from("loop_confidence"), self.confidence.to_string()),
        ]);
        for (key, value) in self.feature_summary.fields() {
            metadata.insert(format!("loop_{key}"), value.clone());
        }
        if self.feature_summary.capped() > 0 {
            metadata.insert(
                String::from("loop_feature_count_capped"),
                self.feature_summary.capped().to_string(),
            );
        }
        metadata
    }

    fn summary_metadata(&self, index: usize) -> BTreeMap<String, String> {
        let prefix = format!("loop_signal_{index}");
        let mut metadata = BTreeMap::from([
            (
                format!("{prefix}_channel"),
                self.channel.as_str().to_owned(),
            ),
            (
                format!("{prefix}_event_kind"),
                self.event_kind.as_str().to_owned(),
            ),
            (
                format!("{prefix}_severity"),
                self.severity.as_str().to_owned(),
            ),
            (format!("{prefix}_confidence"), self.confidence.to_string()),
            (
                format!("{prefix}_reason_code"),
                self.reason_code.as_str().to_owned(),
            ),
        ]);
        for (key, value) in self.feature_summary.fields() {
            metadata.insert(format!("{prefix}_feature_{key}"), value.clone());
        }
        if self.feature_summary.capped() > 0 {
            metadata.insert(
                format!("{prefix}_feature_count_capped"),
                self.feature_summary.capped().to_string(),
            );
        }
        metadata
    }
}

/// Bounded detector summary suitable for observability metadata.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DetectorSummary {
    /// Bounded signal list retained for metadata.
    pub signals: Vec<LoopSignal>,
    /// Total signals observed, including signals omitted from the bounded list.
    pub signal_count: u64,
    /// Signals omitted from the bounded list.
    pub capped_signal_count: u64,
}

impl DetectorSummary {
    /// Returns true when the detector observed at least one signal.
    #[must_use]
    pub const fn has_signals(&self) -> bool {
        self.signal_count > 0
    }

    /// Converts the summary to low-cardinality, content-free metadata.
    #[must_use]
    pub fn metadata(&self, mode: LoopGuardMode) -> BTreeMap<String, String> {
        if !self.has_signals() {
            return BTreeMap::new();
        }
        let mut metadata = BTreeMap::from([
            (String::from("loop_detector_mode"), mode.as_str().to_owned()),
            (
                String::from("loop_signal_count"),
                self.signal_count.to_string(),
            ),
        ]);
        if self.capped_signal_count > 0 {
            metadata.insert(
                String::from("loop_signal_count_capped"),
                self.capped_signal_count.to_string(),
            );
        }
        let mut abort_candidate_count = 0_u64;
        let mut residual_signal_count = 0_u64;
        let mut reasoning_signal_count = 0_u64;
        let mut content_signal_count = 0_u64;
        let mut tool_arguments_signal_count = 0_u64;
        let mut tool_fingerprint_signal_count = 0_u64;
        for (index, signal) in self.signals.iter().enumerate() {
            if signal.is_abort_candidate() {
                abort_candidate_count = abort_candidate_count.saturating_add(1);
            } else {
                residual_signal_count = residual_signal_count.saturating_add(1);
            }
            match signal.channel {
                StreamChannel::Reasoning => {
                    reasoning_signal_count = reasoning_signal_count.saturating_add(1);
                }
                StreamChannel::Content => {
                    content_signal_count = content_signal_count.saturating_add(1);
                }
                StreamChannel::ToolArguments => {
                    tool_arguments_signal_count = tool_arguments_signal_count.saturating_add(1);
                }
                StreamChannel::ToolFingerprint => {
                    tool_fingerprint_signal_count = tool_fingerprint_signal_count.saturating_add(1);
                }
            }
            metadata.extend(signal.summary_metadata(index));
        }
        metadata.insert(
            String::from("loop_abort_candidate_count"),
            abort_candidate_count.to_string(),
        );
        metadata.insert(
            String::from("loop_residual_signal_count"),
            residual_signal_count.to_string(),
        );
        metadata.insert(
            String::from("loop_reasoning_signal_count"),
            reasoning_signal_count.to_string(),
        );
        metadata.insert(
            String::from("loop_content_signal_count"),
            content_signal_count.to_string(),
        );
        metadata.insert(
            String::from("loop_tool_arguments_signal_count"),
            tool_arguments_signal_count.to_string(),
        );
        metadata.insert(
            String::from("loop_tool_fingerprint_signal_count"),
            tool_fingerprint_signal_count.to_string(),
        );
        metadata
    }
}

/// Headless detector interface for channelized stream events.
pub trait LoopDetector {
    /// Observes one detector input and returns newly emitted signals.
    fn observe(&mut self, input: LoopDetectorInput<'_>) -> Vec<LoopSignal>;

    /// Returns the current bounded detector summary.
    fn finish(&self) -> DetectorSummary;
}

/// Request input profile used to raise thresholds for output that copies repeated input.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoopInputProfile {
    repeated_line_hashes: BTreeSet<u64>,
    repeated_token_window_hashes: BTreeSet<u64>,
    state_capping: LoopStateCapping,
}

impl LoopInputProfile {
    /// Builds an input profile from an OpenAI-compatible request body.
    #[must_use]
    pub fn from_request_body(request_body: &[u8], token_window_size: u32) -> Self {
        let Ok(value) = serde_json::from_slice::<Value>(request_body) else {
            return Self::default();
        };
        Self::from_value(&value, token_window_size)
    }

    fn from_value(value: &Value, token_window_size: u32) -> Self {
        let mut profile = Self::default();
        let mut line_counts = BTreeMap::<u64, u32>::new();
        let mut token_window_counts = BTreeMap::<u64, u32>::new();
        profile.observe_value(
            value,
            None,
            token_window_size,
            &mut line_counts,
            &mut token_window_counts,
        );
        profile
    }

    #[cfg(test)]
    fn from_texts(texts: &[String], token_window_size: u32) -> Self {
        let mut profile = Self::default();
        let mut line_counts = BTreeMap::<u64, u32>::new();
        let mut token_window_counts = BTreeMap::<u64, u32>::new();
        for text in texts {
            profile.observe_text(
                text,
                token_window_size,
                &mut line_counts,
                &mut token_window_counts,
            );
        }
        profile
    }

    fn observe_value(
        &mut self,
        value: &Value,
        key: Option<&str>,
        token_window_size: u32,
        line_counts: &mut BTreeMap<u64, u32>,
        token_window_counts: &mut BTreeMap<u64, u32>,
    ) {
        match value {
            Value::String(text) if !key.is_some_and(is_sensitive_input_key) => {
                self.observe_text(text, token_window_size, line_counts, token_window_counts);
            }
            Value::Array(values) => {
                for value in values {
                    self.observe_value(
                        value,
                        key,
                        token_window_size,
                        line_counts,
                        token_window_counts,
                    );
                }
            }
            Value::Object(object) => {
                for (key, value) in object {
                    if !is_sensitive_input_key(key) {
                        self.observe_value(
                            value,
                            Some(key),
                            token_window_size,
                            line_counts,
                            token_window_counts,
                        );
                    }
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }

    fn observe_text(
        &mut self,
        text: &str,
        token_window_size: u32,
        line_counts: &mut BTreeMap<u64, u32>,
        token_window_counts: &mut BTreeMap<u64, u32>,
    ) {
        for line in text.lines() {
            if let Some(hash) = normalized_line_hash(line) {
                if let Some(count) = increment_count_with_cap(
                    line_counts,
                    hash,
                    LOOP_INPUT_LINE_COUNT_CAP,
                    &mut self.state_capping.input_lines,
                ) {
                    if count > 1 {
                        self.repeated_line_hashes.insert(hash);
                    }
                }
            }
        }
        observe_token_window_hashes(text, token_window_size, |window_hash| {
            if let Some(count) = increment_count_with_cap(
                token_window_counts,
                window_hash,
                LOOP_INPUT_TOKEN_WINDOW_COUNT_CAP,
                &mut self.state_capping.input_token_windows,
            ) {
                if count > 1 {
                    self.repeated_token_window_hashes.insert(window_hash);
                }
            }
        });
    }

    fn contains_line_hash(&self, hash: u64) -> bool {
        self.repeated_line_hashes.contains(&hash)
    }

    fn contains_token_window_hash(&self, hash: u64) -> bool {
        self.repeated_token_window_hashes.contains(&hash)
    }
}

/// Default channelized detector implementation.
#[derive(Debug)]
pub struct ChannelizedLoopDetector {
    config: LoopGuardConfig,
    input_profile: LoopInputProfile,
    content: LoopChannelState,
    reasoning: LoopChannelState,
    tool_arguments: LoopChannelState,
    tool_argument_json_counts: BTreeMap<u64, u32>,
    tool_argument_json_count_capped: u64,
    tool_fingerprint_counts: BTreeMap<u64, u32>,
    tool_fingerprint_count_capped: u64,
    signals: Vec<LoopSignal>,
    signal_count: u64,
    capped_signal_count: u64,
}

impl ChannelizedLoopDetector {
    /// Creates a detector with a precomputed input profile.
    #[must_use]
    pub fn new(config: LoopGuardConfig, input_profile: LoopInputProfile) -> Self {
        Self {
            config,
            input_profile,
            content: LoopChannelState::default(),
            reasoning: LoopChannelState::default(),
            tool_arguments: LoopChannelState::default(),
            tool_argument_json_counts: BTreeMap::new(),
            tool_argument_json_count_capped: 0,
            tool_fingerprint_counts: BTreeMap::new(),
            tool_fingerprint_count_capped: 0,
            signals: Vec::new(),
            signal_count: 0,
            capped_signal_count: 0,
        }
    }

    /// Observes a completed tool call and emits argument/fingerprint signals.
    pub fn observe_tool_call(&mut self, input: ToolCallFingerprintInput<'_>) -> Vec<LoopSignal> {
        let signals = match canonical_json_hash(input.arguments) {
            Ok(argument_hash) => self.observe_valid_tool_call(input.tool_name, argument_hash),
            Err(error_code) => vec![invalid_tool_arguments_signal(input.arguments, error_code)],
        };
        self.record_signals(&signals);
        signals
    }

    fn observe_valid_tool_call(&mut self, tool_name: &str, argument_hash: u64) -> Vec<LoopSignal> {
        let mut signals = Vec::new();
        signals.push(tool_arguments_completed_signal(
            argument_hash,
            self.tool_argument_json_count_capped,
        ));
        if let Some(count) = increment_count_with_cap(
            &mut self.tool_argument_json_counts,
            argument_hash,
            LOOP_ARGUMENT_HASH_COUNT_CAP,
            &mut self.tool_argument_json_count_capped,
        ) {
            if count > 1 {
                signals.push(repeated_tool_arguments_signal(
                    argument_hash,
                    u64::from(count),
                    self.tool_argument_json_count_capped,
                ));
            }
        }

        let tool_name_hash = stable_hash(tool_name.as_bytes());
        let fingerprint_hash = stable_hash_u64s([tool_name_hash, argument_hash]);
        if let Some(count) = increment_count_with_cap(
            &mut self.tool_fingerprint_counts,
            fingerprint_hash,
            LOOP_FINGERPRINT_COUNT_CAP,
            &mut self.tool_fingerprint_count_capped,
        ) {
            if count > 1 {
                signals.push(repeated_tool_fingerprint_signal(
                    tool_name_hash,
                    argument_hash,
                    fingerprint_hash,
                    u64::from(count),
                    self.tool_fingerprint_count_capped,
                ));
            }
        }
        signals
    }

    fn record_signals(&mut self, signals: &[LoopSignal]) {
        for signal in signals {
            self.signal_count = self.signal_count.saturating_add(1);
            if self.signals.len() < LOOP_SUMMARY_SIGNAL_LIMIT {
                self.signals.push(signal.clone());
            } else {
                self.capped_signal_count = self.capped_signal_count.saturating_add(1);
            }
        }
    }
}

impl LoopDetector for ChannelizedLoopDetector {
    fn observe(&mut self, input: LoopDetectorInput<'_>) -> Vec<LoopSignal> {
        if input.fragment.is_empty() && matches!(input.event_kind, DetectorEventKind::Fragment) {
            return Vec::new();
        }
        let state = match input.channel {
            StreamChannel::Content => &mut self.content,
            StreamChannel::Reasoning => &mut self.reasoning,
            StreamChannel::ToolArguments => &mut self.tool_arguments,
            StreamChannel::ToolFingerprint => return Vec::new(),
        };
        let signals = state.observe(input, &self.config, &self.input_profile);
        self.record_signals(&signals);
        signals
    }

    fn finish(&self) -> DetectorSummary {
        DetectorSummary {
            signals: self.signals.clone(),
            signal_count: self.signal_count,
            capped_signal_count: self.capped_signal_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LoopStateCapping {
    input_lines: u64,
    input_token_windows: u64,
    output_lines: u64,
    output_token_windows: u64,
    output_unique_windows: u64,
    output_semantic_windows: u64,
}

impl LoopStateCapping {
    const fn is_capped(self) -> bool {
        self.input_lines > 0
            || self.input_token_windows > 0
            || self.output_lines > 0
            || self.output_token_windows > 0
            || self.output_unique_windows > 0
            || self.output_semantic_windows > 0
    }

    fn insert_features(self, features: &mut BoundedFeatureSummary) {
        if !self.is_capped() {
            return;
        }
        features.insert_bool("guard_state_capped", true);
        insert_capped_feature(features, "input_line_count_capped", self.input_lines);
        insert_capped_feature(
            features,
            "input_token_window_count_capped",
            self.input_token_windows,
        );
        insert_capped_feature(features, "output_line_count_capped", self.output_lines);
        insert_capped_feature(
            features,
            "output_token_window_count_capped",
            self.output_token_windows,
        );
        insert_capped_feature(
            features,
            "output_unique_token_window_capped",
            self.output_unique_windows,
        );
        insert_capped_feature(
            features,
            "output_semantic_window_count_capped",
            self.output_semantic_windows,
        );
    }
}

fn insert_capped_feature(features: &mut BoundedFeatureSummary, key: &'static str, value: u64) {
    if value > 0 {
        features.insert_u64(key, value);
    }
}

#[derive(Debug, Default)]
struct LoopChannelState {
    fragment_count: u64,
    bytes_seen: u64,
    pending_line: String,
    line_counts: BTreeMap<u64, u32>,
    line_count_capped: u64,
    current_token: String,
    recent_token_hashes: VecDeque<u64>,
    token_window_counts: BTreeMap<u64, u32>,
    token_window_count_capped: u64,
    unique_token_windows: BTreeSet<u64>,
    unique_token_window_capped: u64,
    token_window_total: u64,
    recent_chars: VecDeque<char>,
    input_overlap_seen: bool,
    semantic_current_token: String,
    semantic_window_tokens: Vec<u64>,
    semantic_windows: VecDeque<SemanticWindow>,
    semantic_history_window_capped: u64,
}

impl LoopChannelState {
    fn observe(
        &mut self,
        input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Vec<LoopSignal> {
        self.fragment_count = self.fragment_count.saturating_add(1);
        self.bytes_seen = self
            .bytes_seen
            .saturating_add(u64::try_from(input.fragment.len()).unwrap_or(u64::MAX));

        let detection = self
            .observe_lines(input, config, input_profile)
            .or_else(|| self.observe_tokens(input, config, input_profile))
            .or_else(|| {
                self.observe_recent_chars(input.fragment);
                self.observe_suffix_cycle(input, config, input_profile)
            })
            .or_else(|| self.observe_semantic(input, config, input_profile))
            .or_else(|| self.observe_low_progress(input, config, input_profile));
        detection.map_or_else(Vec::new, |detection| {
            vec![detection.into_signal(input.channel, input.event_kind)]
        })
    }

    fn observe_lines(
        &mut self,
        input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        for character in input.fragment.chars() {
            if character == '\n' {
                let detection = self.finish_line(input.channel, config, input_profile);
                self.pending_line.clear();
                if detection.is_some() {
                    return detection;
                }
            } else if character != '\r' && self.pending_line.len() < LOOP_MAX_PENDING_LINE_BYTES {
                self.pending_line.push(character);
            }
        }
        if matches!(
            input.event_kind,
            DetectorEventKind::Boundary | DetectorEventKind::Check | DetectorEventKind::Tick
        ) && !self.pending_line.is_empty()
        {
            let detection = self.finish_line(input.channel, config, input_profile);
            self.pending_line.clear();
            return detection;
        }
        None
    }

    fn finish_line(
        &mut self,
        _channel: StreamChannel,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        let hash = normalized_line_hash(&self.pending_line)?;
        let input_overlap = input_profile.contains_line_hash(hash);
        if input_overlap {
            self.input_overlap_seen = true;
        }
        let count = increment_count_with_cap(
            &mut self.line_counts,
            hash,
            LOOP_OUTPUT_LINE_COUNT_CAP,
            &mut self.line_count_capped,
        )?;
        let threshold = Self::adjusted_threshold(
            u64::from(config.output_repeated_line_threshold),
            input_overlap,
            config,
        );
        (u64::from(count) >= threshold).then(|| {
            FeatureDetection::new(
                LoopReasonCode::RepeatedLine,
                self.feature_evidence(
                    u64::from(count),
                    threshold,
                    hash,
                    input_overlap,
                    input_profile,
                ),
            )
        })
    }

    fn observe_tokens(
        &mut self,
        input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        for character in input.fragment.chars() {
            if is_token_boundary(character) {
                if let Some(detection) = self.finish_token(input.channel, config, input_profile) {
                    return Some(detection);
                }
            } else if self.current_token.len() < LOOP_MAX_TOKEN_BYTES {
                for lower in character.to_lowercase() {
                    self.current_token.push(lower);
                }
            }
        }
        if matches!(
            input.event_kind,
            DetectorEventKind::Boundary | DetectorEventKind::Check | DetectorEventKind::Tick
        ) {
            return self.finish_token(input.channel, config, input_profile);
        }
        None
    }

    fn finish_token(
        &mut self,
        _channel: StreamChannel,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        if self.current_token.is_empty() {
            return None;
        }
        let token_hash = stable_hash(self.current_token.as_bytes());
        self.current_token.clear();
        self.recent_token_hashes.push_back(token_hash);
        let window_size = usize::try_from(config.output_token_window_size).unwrap_or(usize::MAX);
        while self.recent_token_hashes.len() > window_size {
            self.recent_token_hashes.pop_front();
        }
        if self.recent_token_hashes.len() != window_size {
            return None;
        }
        let window_hash = stable_hash_u64s(self.recent_token_hashes.iter().copied());
        self.token_window_total = self.token_window_total.saturating_add(1);
        track_unique_hash_with_cap(
            &mut self.unique_token_windows,
            window_hash,
            LOOP_OUTPUT_UNIQUE_TOKEN_WINDOW_CAP,
            &mut self.unique_token_window_capped,
        );
        let input_overlap = input_profile.contains_token_window_hash(window_hash);
        if input_overlap {
            self.input_overlap_seen = true;
        }
        let count = increment_count_with_cap(
            &mut self.token_window_counts,
            window_hash,
            LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP,
            &mut self.token_window_count_capped,
        )?;
        let threshold = Self::adjusted_threshold(
            u64::from(config.output_repeated_token_window_threshold),
            input_overlap,
            config,
        );
        (u64::from(count) >= threshold).then(|| {
            let mut detection = FeatureDetection::new(
                LoopReasonCode::RepeatedTokenWindow,
                self.feature_evidence(
                    u64::from(count),
                    threshold,
                    window_hash,
                    input_overlap,
                    input_profile,
                ),
            );
            detection.token_window_size = Some(config.output_token_window_size);
            detection.unique_window_count =
                Some(u64::try_from(self.unique_token_windows.len()).unwrap_or(u64::MAX));
            detection.total_window_count = Some(self.token_window_total);
            detection
        })
    }

    fn observe_recent_chars(&mut self, fragment: &str) {
        for character in fragment.chars() {
            for normalized in character.to_lowercase() {
                self.recent_chars.push_back(normalized);
            }
            while self.recent_chars.len() > LOOP_MAX_RECENT_CHARS {
                self.recent_chars.pop_front();
            }
        }
    }

    fn observe_suffix_cycle(
        &mut self,
        _input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        let cycle = suffix_cycle(&self.recent_chars, config.output_suffix_cycle_threshold)?;
        let input_overlap = input_profile.contains_line_hash(cycle.unit_hash);
        if input_overlap {
            self.input_overlap_seen = true;
        }
        let threshold = Self::adjusted_threshold(
            u64::from(config.output_suffix_cycle_threshold),
            input_overlap,
            config,
        );
        (cycle.repetitions >= threshold).then(|| {
            FeatureDetection::new(
                LoopReasonCode::SuffixCycle,
                self.feature_evidence(
                    cycle.repetitions,
                    threshold,
                    cycle.unit_hash,
                    input_overlap,
                    input_profile,
                ),
            )
        })
    }

    fn observe_semantic(
        &mut self,
        input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        if input.channel != StreamChannel::Reasoning || !config.reasoning_semantic_detection_enabled
        {
            return None;
        }

        for character in input.fragment.chars() {
            if character.is_ascii_alphanumeric() {
                if self.semantic_current_token.len() < LOOP_MAX_SEMANTIC_TOKEN_BYTES {
                    self.semantic_current_token
                        .push(character.to_ascii_lowercase());
                }
                continue;
            }

            if let Some(detection) = self.finish_semantic_token(input, config, input_profile) {
                return Some(detection);
            }
            if character == '\n' {
                if self.semantic_window_tokens.len()
                    >= usize::try_from(config.reasoning_semantic_minimum_token_count)
                        .unwrap_or(usize::MAX)
                {
                    return self.finish_semantic_window(input, config, input_profile);
                }
                self.semantic_window_tokens.clear();
            }
        }

        if matches!(
            input.event_kind,
            DetectorEventKind::Boundary | DetectorEventKind::Check | DetectorEventKind::Tick
        ) {
            self.finish_semantic_token(input, config, input_profile)
        } else {
            None
        }
    }

    fn finish_semantic_token(
        &mut self,
        input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        if self.semantic_current_token.is_empty() {
            return None;
        }
        if let Some(token_hash) = semantic_token_hash(&self.semantic_current_token) {
            self.semantic_window_tokens.push(token_hash);
        }
        self.semantic_current_token.clear();

        if self.semantic_window_tokens.len()
            >= usize::try_from(config.reasoning_semantic_window_token_count).unwrap_or(usize::MAX)
        {
            return self.finish_semantic_window(input, config, input_profile);
        }
        None
    }

    fn finish_semantic_window(
        &mut self,
        _input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        let minimum_tokens =
            usize::try_from(config.reasoning_semantic_minimum_token_count).unwrap_or(usize::MAX);
        if self.semantic_window_tokens.len() < minimum_tokens {
            return None;
        }

        let window = SemanticWindow::from_tokens(&self.semantic_window_tokens);
        let threshold = u64::from(config.reasoning_semantic_similarity_threshold_percent);
        let similarity = self
            .semantic_windows
            .iter()
            .map(|previous| window.similarity_percent(previous))
            .max()
            .unwrap_or(0);
        if similarity >= threshold {
            let mut detection = FeatureDetection::new(
                LoopReasonCode::ApproximateRepetition,
                self.feature_evidence(
                    similarity,
                    threshold,
                    window.sample_hash,
                    false,
                    input_profile,
                ),
            );
            detection.token_window_size = Some(config.reasoning_semantic_window_token_count);
            detection.semantic_similarity_percent = Some(similarity);
            detection.semantic_feature_count =
                Some(u64::try_from(window.feature_count()).unwrap_or(u64::MAX));
            detection.semantic_history_window_count =
                Some(u64::try_from(self.semantic_windows.len()).unwrap_or(u64::MAX));
            return Some(detection);
        }

        let history_cap =
            usize::try_from(config.reasoning_semantic_history_window_count).unwrap_or(usize::MAX);
        if self.semantic_windows.len() >= history_cap {
            self.semantic_windows.pop_front();
            self.semantic_history_window_capped =
                self.semantic_history_window_capped.saturating_add(1);
        }
        if self.semantic_windows.len() < LOOP_OUTPUT_SEMANTIC_WINDOW_CAP {
            self.semantic_windows.push_back(window);
        } else {
            self.semantic_history_window_capped =
                self.semantic_history_window_capped.saturating_add(1);
        }
        self.semantic_window_tokens.clear();
        None
    }

    fn observe_low_progress(
        &mut self,
        _input: LoopDetectorInput<'_>,
        config: &LoopGuardConfig,
        input_profile: &LoopInputProfile,
    ) -> Option<FeatureDetection> {
        let min_bytes = if self.input_overlap_seen {
            config
                .output_low_progress_min_bytes
                .saturating_mul(u64::from(config.input_overlap_threshold_multiplier))
        } else {
            config.output_low_progress_min_bytes
        };
        if self.bytes_seen < min_bytes || self.token_window_total == 0 {
            return None;
        }
        if self.unique_token_window_capped > 0 {
            return None;
        }
        let unique_count = u64::try_from(self.unique_token_windows.len()).unwrap_or(u64::MAX);
        let unique_ratio_percent = unique_count.saturating_mul(100) / self.token_window_total;
        if unique_ratio_percent > u64::from(config.output_low_progress_unique_ratio_percent) {
            return None;
        }
        let mut detection = FeatureDetection::new(
            LoopReasonCode::LowProgressGrowth,
            self.feature_evidence(
                self.token_window_total,
                min_bytes,
                stable_hash_u64s(self.unique_token_windows.iter().copied()),
                self.input_overlap_seen,
                input_profile,
            ),
        );
        detection.token_window_size = Some(config.output_token_window_size);
        detection.unique_ratio_percent = Some(unique_ratio_percent);
        detection.unique_window_count = Some(unique_count);
        detection.total_window_count = Some(self.token_window_total);
        Some(detection)
    }

    fn adjusted_threshold(threshold: u64, input_overlap: bool, config: &LoopGuardConfig) -> u64 {
        if input_overlap {
            threshold.saturating_mul(u64::from(config.input_overlap_threshold_multiplier))
        } else {
            threshold
        }
    }

    fn state_capping(&self, input_profile: &LoopInputProfile) -> LoopStateCapping {
        let mut capping = input_profile.state_capping;
        capping.output_lines = self.line_count_capped;
        capping.output_token_windows = self.token_window_count_capped;
        capping.output_unique_windows = self.unique_token_window_capped;
        capping.output_semantic_windows = self.semantic_history_window_capped;
        capping
    }

    fn feature_evidence(
        &self,
        observed_count: u64,
        threshold: u64,
        sample_hash: u64,
        input_overlap_applied: bool,
        input_profile: &LoopInputProfile,
    ) -> FeatureEvidence {
        FeatureEvidence {
            observed_count,
            threshold,
            observed_bytes: self.bytes_seen,
            fragment_count: self.fragment_count,
            sample_hash,
            input_overlap_applied,
            state_capping: self.state_capping(input_profile),
        }
    }
}

#[derive(Clone, Debug)]
struct FeatureDetection {
    reason_code: LoopReasonCode,
    observed_count: u64,
    threshold: u64,
    observed_bytes: u64,
    fragment_count: u64,
    sample_hash: u64,
    input_overlap_applied: bool,
    token_window_size: Option<u32>,
    unique_ratio_percent: Option<u64>,
    unique_window_count: Option<u64>,
    total_window_count: Option<u64>,
    semantic_similarity_percent: Option<u64>,
    semantic_feature_count: Option<u64>,
    semantic_history_window_count: Option<u64>,
    state_capping: LoopStateCapping,
}

#[derive(Clone, Copy, Debug)]
struct FeatureEvidence {
    observed_count: u64,
    threshold: u64,
    observed_bytes: u64,
    fragment_count: u64,
    sample_hash: u64,
    input_overlap_applied: bool,
    state_capping: LoopStateCapping,
}

impl FeatureDetection {
    fn new(reason_code: LoopReasonCode, evidence: FeatureEvidence) -> Self {
        Self {
            reason_code,
            observed_count: evidence.observed_count,
            threshold: evidence.threshold,
            observed_bytes: evidence.observed_bytes,
            fragment_count: evidence.fragment_count,
            sample_hash: evidence.sample_hash,
            input_overlap_applied: evidence.input_overlap_applied,
            token_window_size: None,
            unique_ratio_percent: None,
            unique_window_count: None,
            total_window_count: None,
            semantic_similarity_percent: None,
            semantic_feature_count: None,
            semantic_history_window_count: None,
            state_capping: evidence.state_capping,
        }
    }

    fn into_signal(self, channel: StreamChannel, event_kind: DetectorEventKind) -> LoopSignal {
        let mut feature_summary = BoundedFeatureSummary::new();
        feature_summary.insert_u64("observed_count", self.observed_count);
        feature_summary.insert_u64("threshold", self.threshold);
        feature_summary.insert_u64("observed_bytes", self.observed_bytes);
        feature_summary.insert_u64("fragment_count", self.fragment_count);
        feature_summary.insert_str("sample_hash", format_hash(self.sample_hash));
        feature_summary.insert_bool("input_overlap_applied", self.input_overlap_applied);
        if let Some(token_window_size) = self.token_window_size {
            feature_summary.insert_u64("token_window_size", u64::from(token_window_size));
        }
        if let Some(unique_ratio_percent) = self.unique_ratio_percent {
            feature_summary.insert_u64("unique_ratio_percent", unique_ratio_percent);
        }
        if let Some(unique_window_count) = self.unique_window_count {
            feature_summary.insert_u64("unique_window_count", unique_window_count);
        }
        if let Some(total_window_count) = self.total_window_count {
            feature_summary.insert_u64("total_window_count", total_window_count);
        }
        if let Some(semantic_similarity_percent) = self.semantic_similarity_percent {
            feature_summary.insert_u64("semantic_similarity_percent", semantic_similarity_percent);
        }
        if let Some(semantic_feature_count) = self.semantic_feature_count {
            feature_summary.insert_u64("semantic_feature_count", semantic_feature_count);
        }
        if let Some(semantic_history_window_count) = self.semantic_history_window_count {
            feature_summary.insert_u64(
                "semantic_history_window_count",
                semantic_history_window_count,
            );
        }
        self.state_capping.insert_features(&mut feature_summary);

        let (severity, confidence) = severity_for_detection(self.reason_code, channel);
        LoopSignal {
            channel,
            event_kind,
            severity,
            confidence,
            reason_code: self.reason_code,
            feature_summary,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuffixCycle {
    unit_hash: u64,
    repetitions: u64,
}

#[derive(Clone, Debug)]
struct SemanticWindow {
    token_features: BTreeSet<u64>,
    ngram_features: BTreeSet<u64>,
    sample_hash: u64,
}

impl SemanticWindow {
    fn from_tokens(tokens: &[u64]) -> Self {
        let mut token_features = BTreeSet::new();
        let mut ngram_features = BTreeSet::new();
        for token in tokens {
            token_features.insert(stable_hash_u64s([0, *token]));
        }
        for pair in tokens.windows(2) {
            ngram_features.insert(stable_hash_u64s([1, pair[0], pair[1]]));
        }
        let sample_hash =
            stable_hash_u64s(token_features.iter().chain(ngram_features.iter()).copied());
        Self {
            token_features,
            ngram_features,
            sample_hash,
        }
    }

    fn feature_count(&self) -> usize {
        self.token_features
            .len()
            .saturating_add(self.ngram_features.len())
    }

    fn similarity_percent(&self, other: &Self) -> u64 {
        jaccard_percent(&self.token_features, &other.token_features)
            .max(jaccard_percent(&self.ngram_features, &other.ngram_features))
    }
}

fn severity_for_detection(
    reason_code: LoopReasonCode,
    channel: StreamChannel,
) -> (LoopSeverity, u8) {
    match (channel, reason_code) {
        (StreamChannel::Reasoning, LoopReasonCode::ApproximateRepetition) => {
            (LoopSeverity::AbortCandidate, 88)
        }
        (StreamChannel::Reasoning, _) => (LoopSeverity::AbortCandidate, 90),
        (StreamChannel::Content, LoopReasonCode::LowProgressGrowth) => (LoopSeverity::Suspect, 55),
        (StreamChannel::Content, _) => (LoopSeverity::Suspect, 60),
        (StreamChannel::ToolArguments, LoopReasonCode::SuffixCycle) => {
            (LoopSeverity::AbortCandidate, 85)
        }
        (StreamChannel::ToolArguments, _) => (LoopSeverity::Suspect, 70),
        (StreamChannel::ToolFingerprint, LoopReasonCode::ToolFingerprintRepeated) => {
            (LoopSeverity::AbortCandidate, 92)
        }
        (StreamChannel::ToolFingerprint, _) => (LoopSeverity::Observe, 20),
    }
}

fn tool_arguments_completed_signal(argument_hash: u64, capped: u64) -> LoopSignal {
    let mut feature_summary = BoundedFeatureSummary::new();
    feature_summary.insert_str("arguments_hash", format_hash(argument_hash));
    insert_capped_feature(
        &mut feature_summary,
        "tool_argument_json_count_capped",
        capped,
    );
    LoopSignal {
        channel: StreamChannel::ToolArguments,
        event_kind: DetectorEventKind::CompletedJson,
        severity: LoopSeverity::Observe,
        confidence: 15,
        reason_code: LoopReasonCode::ToolArgumentsJsonCompleted,
        feature_summary,
    }
}

fn repeated_tool_arguments_signal(
    argument_hash: u64,
    repeat_count: u64,
    capped: u64,
) -> LoopSignal {
    let mut feature_summary = BoundedFeatureSummary::new();
    feature_summary.insert_str("arguments_hash", format_hash(argument_hash));
    feature_summary.insert_u64("repeat_count", repeat_count);
    insert_capped_feature(
        &mut feature_summary,
        "tool_argument_json_count_capped",
        capped,
    );
    LoopSignal {
        channel: StreamChannel::ToolArguments,
        event_kind: DetectorEventKind::CompletedJson,
        severity: LoopSeverity::Suspect,
        confidence: 72,
        reason_code: LoopReasonCode::ToolArgumentsRepeatedJson,
        feature_summary,
    }
}

fn repeated_tool_fingerprint_signal(
    tool_name_hash: u64,
    argument_hash: u64,
    fingerprint_hash: u64,
    repeat_count: u64,
    capped: u64,
) -> LoopSignal {
    let mut feature_summary = BoundedFeatureSummary::new();
    feature_summary.insert_str("tool_name_hash", format_hash(tool_name_hash));
    feature_summary.insert_str("arguments_hash", format_hash(argument_hash));
    feature_summary.insert_str("fingerprint_hash", format_hash(fingerprint_hash));
    feature_summary.insert_u64("repeat_count", repeat_count);
    insert_capped_feature(
        &mut feature_summary,
        "tool_fingerprint_count_capped",
        capped,
    );
    LoopSignal {
        channel: StreamChannel::ToolFingerprint,
        event_kind: DetectorEventKind::CompletedToolFingerprint,
        severity: LoopSeverity::AbortCandidate,
        confidence: 92,
        reason_code: LoopReasonCode::ToolFingerprintRepeated,
        feature_summary,
    }
}

fn invalid_tool_arguments_signal(arguments: &str, error_code: &'static str) -> LoopSignal {
    let mut feature_summary = BoundedFeatureSummary::new();
    feature_summary.insert_str(
        "arguments_hash",
        format_hash(stable_hash(arguments.as_bytes())),
    );
    feature_summary.insert_str("error_code", error_code);
    LoopSignal {
        channel: StreamChannel::ToolArguments,
        event_kind: DetectorEventKind::CompletedJson,
        severity: LoopSeverity::Observe,
        confidence: 10,
        reason_code: LoopReasonCode::ToolArgumentsInvalidJson,
        feature_summary,
    }
}

fn jaccard_percent(left: &BTreeSet<u64>, right: &BTreeSet<u64>) -> u64 {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let intersection = left.intersection(right).count();
    let union = left
        .len()
        .saturating_add(right.len())
        .saturating_sub(intersection);
    if union == 0 {
        return 0;
    }
    u64::try_from(intersection.saturating_mul(100) / union).unwrap_or(u64::MAX)
}

fn suffix_cycle(chars: &VecDeque<char>, minimum_repetitions: u32) -> Option<SuffixCycle> {
    let chars = chars.iter().copied().collect::<Vec<_>>();
    let minimum_repetitions = usize::try_from(minimum_repetitions).ok()?;
    for unit_len in LOOP_SUFFIX_MIN_UNIT_CHARS..=LOOP_SUFFIX_MAX_UNIT_CHARS {
        let required_len = unit_len.saturating_mul(minimum_repetitions);
        if chars.len() < required_len {
            continue;
        }
        let suffix = &chars[chars.len() - unit_len..];
        let mut repetitions = 1_usize;
        while chars.len() >= unit_len.saturating_mul(repetitions + 1) {
            let start = chars.len() - unit_len.saturating_mul(repetitions + 1);
            let end = start + unit_len;
            if &chars[start..end] != suffix {
                break;
            }
            repetitions += 1;
        }
        if repetitions >= minimum_repetitions {
            let unit = suffix.iter().collect::<String>();
            return Some(SuffixCycle {
                unit_hash: stable_hash(unit.as_bytes()),
                repetitions: u64::try_from(repetitions).unwrap_or(u64::MAX),
            });
        }
    }
    None
}

fn semantic_token_hash(token: &str) -> Option<u64> {
    let normalized = normalize_semantic_token(token)?;
    Some(stable_hash(normalized.as_bytes()))
}

fn normalize_semantic_token(token: &str) -> Option<String> {
    if token.len() < 3 || is_semantic_stop_word(token) {
        return None;
    }
    let normalized = match token {
        "tmp" | "temp" | "tmpdir" | "temporary" => String::from("temporary"),
        "dir" | "dirs" | "directory" | "directories" => String::from("directory"),
        "extracting" | "extracted" | "extracts" | "extraction" => String::from("extract"),
        "archives" | "archived" => String::from("archive"),
        "zips" | "zipped" => String::from("zip"),
        _ => strip_semantic_suffix(token),
    };
    (normalized.len() >= 3 && !is_semantic_stop_word(&normalized)).then_some(normalized)
}

fn strip_semantic_suffix(token: &str) -> String {
    for suffix in ["ing", "ed", "es", "s"] {
        if token.len() > suffix.len().saturating_add(3) && token.ends_with(suffix) {
            return token[..token.len() - suffix.len()].to_owned();
        }
    }
    token.to_owned()
}

fn is_semantic_stop_word(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "but"
            | "for"
            | "with"
            | "that"
            | "this"
            | "then"
            | "than"
            | "from"
            | "into"
            | "onto"
            | "over"
            | "under"
            | "again"
            | "actually"
            | "maybe"
            | "could"
            | "would"
            | "should"
            | "need"
            | "needs"
            | "using"
            | "use"
            | "try"
            | "first"
            | "next"
            | "back"
            | "approach"
            | "plan"
            | "step"
            | "option"
            | "think"
            | "about"
            | "without"
            | "before"
            | "after"
            | "because"
            | "there"
            | "where"
            | "when"
            | "what"
            | "which"
            | "while"
    )
}

fn observe_token_window_hashes(
    text: &str,
    token_window_size: u32,
    mut observe_window_hash: impl FnMut(u64),
) {
    let window_size = usize::try_from(token_window_size).unwrap_or(usize::MAX);
    if window_size == 0 {
        return;
    }
    let mut current_token = String::new();
    let mut recent_token_hashes = VecDeque::new();
    for character in text.chars() {
        if is_token_boundary(character) {
            push_token_window_hash(
                &mut current_token,
                &mut recent_token_hashes,
                window_size,
                &mut observe_window_hash,
            );
        } else if current_token.len() < LOOP_MAX_TOKEN_BYTES {
            for lower in character.to_lowercase() {
                current_token.push(lower);
            }
        }
    }
    push_token_window_hash(
        &mut current_token,
        &mut recent_token_hashes,
        window_size,
        &mut observe_window_hash,
    );
}

fn push_token_window_hash(
    current_token: &mut String,
    recent_token_hashes: &mut VecDeque<u64>,
    window_size: usize,
    observe_window_hash: &mut impl FnMut(u64),
) {
    if current_token.is_empty() {
        return;
    }
    recent_token_hashes.push_back(stable_hash(current_token.as_bytes()));
    current_token.clear();
    while recent_token_hashes.len() > window_size {
        recent_token_hashes.pop_front();
    }
    if recent_token_hashes.len() == window_size {
        observe_window_hash(stable_hash_u64s(recent_token_hashes.iter().copied()));
    }
}

fn normalized_line_hash(line: &str) -> Option<u64> {
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    (normalized.chars().count() >= LOOP_MIN_LINE_CHARS)
        .then(|| stable_hash(normalized.to_lowercase().as_bytes()))
}

fn increment_count_with_cap(
    counts: &mut BTreeMap<u64, u32>,
    hash: u64,
    cap: usize,
    capped_count: &mut u64,
) -> Option<u32> {
    if let Some(count) = counts.get_mut(&hash) {
        *count = count.saturating_add(1);
        return Some(*count);
    }
    if counts.len() >= cap {
        *capped_count = capped_count.saturating_add(1);
        return None;
    }
    counts.insert(hash, 1);
    Some(1)
}

fn track_unique_hash_with_cap(
    hashes: &mut BTreeSet<u64>,
    hash: u64,
    cap: usize,
    capped_count: &mut u64,
) {
    if hashes.contains(&hash) {
        return;
    }
    if hashes.len() >= cap {
        *capped_count = capped_count.saturating_add(1);
        return;
    }
    hashes.insert(hash);
}

fn canonical_json_hash(arguments: &str) -> Result<u64, &'static str> {
    let value = serde_json::from_str::<Value>(arguments).map_err(|_error| "invalid_json")?;
    let mut canonical = String::new();
    write_canonical_json(&value, &mut canonical);
    Ok(stable_hash(canonical.as_bytes()))
}

fn write_canonical_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(value) => {
            output.push_str(
                &serde_json::to_string(value).unwrap_or_else(|_error| String::from("\"\"")),
            );
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output);
            }
            output.push(']');
        }
        Value::Object(object) => {
            output.push('{');
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).unwrap_or_else(|_error| String::from("\"\"")),
                );
                output.push(':');
                write_canonical_json(value, output);
            }
            output.push('}');
        }
    }
}

fn is_sensitive_input_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|character| character.to_ascii_lowercase())
        .collect::<String>();
    if matches!(
        normalized.as_str(),
        "maxtokens" | "maxcompletiontokens" | "maxoutputtokens" | "budgettokens"
    ) {
        return false;
    }
    [
        "authorization",
        "apikey",
        "accesskey",
        "privatekey",
        "secret",
        "password",
        "credential",
        "credentials",
        "bearer",
        "token",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn is_token_boundary(character: char) -> bool {
    !character.is_alphanumeric()
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV64_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV64_PRIME)
    })
}

fn stable_hash_u64s(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(FNV64_OFFSET_BASIS, |hash, value| {
        stable_hash_step(hash, value.to_le_bytes())
    })
}

fn stable_hash_step<const N: usize>(hash: u64, bytes: [u8; N]) -> u64 {
    bytes.into_iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV64_PRIME)
    })
}

fn format_hash(hash: u64) -> String {
    format!("fnv64:{hash:016x}")
}

fn truncate_chars(value: String, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value;
    }
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_detector_reasoning_repeated_paragraph_emits_abort_candidate() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 3;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        for _ in 0..2 {
            detector.observe(LoopDetectorInput::fragment(
                StreamChannel::Reasoning,
                "repeat this reasoning paragraph\n",
            ));
        }
        let signals = detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Reasoning,
            "repeat this reasoning paragraph\n",
        ));

        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].channel, StreamChannel::Reasoning);
        assert_eq!(signals[0].reason_code, LoopReasonCode::RepeatedLine);
        assert_eq!(signals[0].severity, LoopSeverity::AbortCandidate);
        assert!(
            !format!("{:?}", signals[0].legacy_abort_metadata())
                .contains("repeat this reasoning paragraph")
        );
    }

    #[test]
    fn loop_detector_reasoning_low_progress_placeholder_is_content_free() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_line_threshold = u32::MAX;
        config.output_repeated_token_window_threshold = u32::MAX;
        config.output_suffix_cycle_threshold = u32::MAX;
        config.output_low_progress_min_bytes = 24;
        config.output_low_progress_unique_ratio_percent = 40;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        let signals = detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Reasoning,
            "alpha beta alpha beta alpha beta alpha beta ",
        ));

        assert_eq!(signals[0].reason_code, LoopReasonCode::LowProgressGrowth);
        assert_eq!(signals[0].severity, LoopSeverity::AbortCandidate);
        assert_eq!(
            signals[0].feature_summary.fields()["unique_ratio_percent"],
            "28"
        );
    }

    #[test]
    fn loop_detector_content_repetition_is_suspect_not_abort_candidate() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_token_window_threshold = 3;
        config.output_suffix_cycle_threshold = 100;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Content,
            "alpha beta ",
        ));
        detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Content,
            "alpha beta ",
        ));
        let signals = detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Content,
            "alpha beta ",
        ));

        assert_eq!(signals[0].severity, LoopSeverity::Suspect);
        assert!(!signals[0].is_abort_candidate());
    }

    #[test]
    fn loop_detector_tool_arguments_and_fingerprint_use_canonical_json_hashes() {
        let config = test_loop_config();
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        let first = detector.observe_tool_call(ToolCallFingerprintInput {
            tool_name: "lookup",
            arguments: r#"{"q":"x","limit":1}"#,
        });
        let second = detector.observe_tool_call(ToolCallFingerprintInput {
            tool_name: "lookup",
            arguments: r#"{"limit":1,"q":"x"}"#,
        });

        assert_eq!(
            first[0].reason_code,
            LoopReasonCode::ToolArgumentsJsonCompleted
        );
        assert!(
            second
                .iter()
                .any(|signal| signal.reason_code == LoopReasonCode::ToolArgumentsRepeatedJson)
        );
        let fingerprint = second
            .iter()
            .find(|signal| signal.channel == StreamChannel::ToolFingerprint)
            .expect("repeated fingerprint should emit a signal");
        assert_eq!(
            fingerprint.reason_code,
            LoopReasonCode::ToolFingerprintRepeated
        );
        assert_eq!(fingerprint.severity, LoopSeverity::AbortCandidate);
        assert_eq!(
            first[0].feature_summary.fields()["arguments_hash"],
            second[0].feature_summary.fields()["arguments_hash"]
        );
    }

    #[test]
    fn loop_detector_invalid_tool_arguments_do_not_leak_raw_json() {
        let config = test_loop_config();
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        let signals = detector.observe_tool_call(ToolCallFingerprintInput {
            tool_name: "lookup",
            arguments: r#"{"secret":"do-not-store""#,
        });

        assert_eq!(
            signals[0].reason_code,
            LoopReasonCode::ToolArgumentsInvalidJson
        );
        let metadata = signals[0].summary_metadata(0);
        assert!(!format!("{metadata:?}").contains("do-not-store"));
        assert!(
            metadata
                .get("loop_signal_0_feature_arguments_hash")
                .is_some_and(|value| value.starts_with("fnv64:"))
        );
    }

    #[test]
    fn loop_detector_non_newline_fragment_check_and_tick_events_can_trigger() {
        let mut config = test_loop_config();
        config.output_suffix_cycle_threshold = 4;
        config.output_repeated_token_window_threshold = u32::MAX;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        let signals = detector.observe(LoopDetectorInput {
            channel: StreamChannel::Reasoning,
            event_kind: DetectorEventKind::Check,
            fragment: "looplooplooploop",
        });
        assert_eq!(signals[0].reason_code, LoopReasonCode::SuffixCycle);
        assert_eq!(signals[0].event_kind, DetectorEventKind::Check);

        let tick_signals = detector.observe(LoopDetectorInput::event(
            StreamChannel::Reasoning,
            DetectorEventKind::Tick,
        ));
        assert!(tick_signals.len() <= 1);
    }

    #[test]
    fn loop_detector_punctuation_boundary_flushes_without_newline() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_token_window_threshold = 2;
        config.output_suffix_cycle_threshold = u32::MAX;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Reasoning,
            "alpha beta.",
        ));
        let signals = detector.observe(LoopDetectorInput {
            channel: StreamChannel::Reasoning,
            event_kind: DetectorEventKind::Boundary,
            fragment: "alpha beta.",
        });

        assert!(
            signals
                .iter()
                .any(|signal| signal.reason_code == LoopReasonCode::RepeatedTokenWindow)
        );
    }

    #[test]
    fn loop_detector_code_and_table_like_content_do_not_high_confidence_abort() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 2;
        config.output_token_window_size = 2;
        config.output_repeated_token_window_threshold = 2;
        config.output_suffix_cycle_threshold = 4;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        let fragments = [
            "```rust\nlet value = map.get(\"key\");\nlet value = map.get(\"key\");\n```\n",
            "| name | value |\n| ---- | ----- |\n| name | value |\n",
            "1. check config\n2. check config\n",
        ];
        for fragment in fragments {
            for signal in detector.observe(LoopDetectorInput::fragment(
                StreamChannel::Content,
                fragment,
            )) {
                assert_ne!(signal.severity, LoopSeverity::AbortCandidate);
                assert!(signal.confidence < 80);
            }
        }
    }

    #[test]
    fn loop_detector_summary_metadata_is_bounded_and_content_free() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 2;
        let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

        for _ in 0..12 {
            detector.observe(LoopDetectorInput::fragment(
                StreamChannel::Reasoning,
                "bounded metadata line\n",
            ));
        }

        let metadata = detector.finish().metadata(LoopGuardMode::Monitor);
        assert_eq!(metadata["loop_detector_mode"], "monitor");
        assert!(metadata["loop_signal_count"].parse::<u64>().unwrap() > 0);
        assert!(!format!("{metadata:?}").contains("bounded metadata line"));
        assert!(metadata.len() < 200);
    }

    #[test]
    fn loop_detector_input_overlap_multiplies_threshold() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 3;
        config.output_repeated_token_window_threshold = u32::MAX;
        config.output_suffix_cycle_threshold = u32::MAX;
        config.input_overlap_threshold_multiplier = 2;
        let repeated =
            String::from("legitimate repeated input line\nlegitimate repeated input line\n");
        let profile = LoopInputProfile::from_texts(&[repeated], config.output_token_window_size);
        let mut detector = ChannelizedLoopDetector::new(config, profile);

        for _ in 0..5 {
            detector.observe(LoopDetectorInput::fragment(
                StreamChannel::Reasoning,
                "legitimate repeated input line\n",
            ));
        }
        let signals = detector.observe(LoopDetectorInput::fragment(
            StreamChannel::Reasoning,
            "legitimate repeated input line\n",
        ));

        assert_eq!(signals[0].feature_summary.fields()["threshold"], "6");
        assert_eq!(
            signals[0].feature_summary.fields()["input_overlap_applied"],
            "true"
        );
    }

    #[test]
    fn tool_loop_detector_same_fingerprint_repeated_triggers_signal() {
        let mut detector = ToolLoopDetector::new(16);
        let fp = ToolFingerprint::from_hashes(String::from("read"), 10, 100);

        let s1 = detector.observe_fingerprint(fp.clone());
        let s2 = detector.observe_fingerprint(fp.clone());
        let s3 = detector.observe_fingerprint(fp.clone());

        assert!(s1.is_empty());
        assert!(s2.is_empty());
        assert_eq!(s3.len(), 1);
        assert_eq!(s3[0].reason_code, LoopReasonCode::ToolFingerprintRepeat);
        assert_eq!(s3[0].fingerprint_hash, 100);
        assert_eq!(s3[0].repeat_count, 3);
        assert!(s3[0].risk >= 0.6);
    }

    #[test]
    fn tool_loop_detector_different_fingerprints_do_not_trigger_repeat() {
        let mut detector = ToolLoopDetector::new(16);
        let a = ToolFingerprint::from_hashes(String::from("read"), 1, 11);
        let b = ToolFingerprint::from_hashes(String::from("read"), 2, 22);
        let c = ToolFingerprint::from_hashes(String::from("read"), 3, 33);

        let signals_a = detector.observe_fingerprint(a);
        let signals_b = detector.observe_fingerprint(b);
        let signals_c = detector.observe_fingerprint(c);

        let all = [signals_a, signals_b, signals_c].concat();
        assert!(
            !all.iter()
                .any(|s| s.reason_code == LoopReasonCode::ToolFingerprintRepeat),
            "distinct fingerprints must not trigger repeat signal"
        );
    }

    #[test]
    fn tool_loop_detector_alternation_pattern_detected() {
        let mut detector = ToolLoopDetector::new(16);
        let a = ToolFingerprint::from_hashes(String::from("read"), 1, 11);
        let b = ToolFingerprint::from_hashes(String::from("grep"), 2, 22);

        detector.observe_fingerprint(a.clone());
        detector.observe_fingerprint(b.clone());
        detector.observe_fingerprint(a.clone());
        let signals = detector.observe_fingerprint(b.clone());

        assert!(
            signals
                .iter()
                .any(|s| s.reason_code == LoopReasonCode::ToolAlternationCycle),
            "A-B-A-B must trigger alternation signal"
        );
    }

    #[test]
    fn tool_loop_detector_no_alternation_for_aaaa() {
        let mut detector = ToolLoopDetector::new(16);
        let a = ToolFingerprint::from_hashes(String::from("read"), 1, 11);

        detector.observe_fingerprint(a.clone());
        detector.observe_fingerprint(a.clone());
        let signals = detector.observe_fingerprint(a.clone());
        detector.observe_fingerprint(a.clone());

        assert!(
            !signals
                .iter()
                .any(|s| s.reason_code == LoopReasonCode::ToolAlternationCycle),
            "A-A-A must not trigger alternation"
        );
    }

    #[test]
    fn tool_loop_detector_blocked_output_echo_detected() {
        let mut detector = ToolLoopDetector::new(16);
        let output_hash = 4242_u64;

        let first = detector.observe_tool_output(output_hash);
        let second = detector.observe_tool_output(output_hash);

        assert!(first.is_empty());
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].reason_code, LoopReasonCode::ToolOutputBlockedEcho);
        assert_eq!(second[0].repeat_count, 2);
        assert!(second[0].risk >= 0.5);
    }

    #[test]
    fn tool_loop_detector_different_outputs_do_not_trigger() {
        let mut detector = ToolLoopDetector::new(16);

        let s1 = detector.observe_tool_output(1);
        let s2 = detector.observe_tool_output(2);
        let s3 = detector.observe_tool_output(3);

        let all = [s1, s2, s3].concat();
        assert!(
            !all.iter()
                .any(|s| s.reason_code == LoopReasonCode::ToolOutputBlockedEcho),
            "distinct outputs must not trigger blocked signal"
        );
    }

    #[test]
    fn tool_loop_detector_history_is_bounded() {
        let mut detector = ToolLoopDetector::new(4);

        // Push 6 distinct fingerprints; only the last 4 should remain.
        for i in 1..=6_u64 {
            let fp = ToolFingerprint::from_hashes(String::from("t"), i, i * 10);
            detector.observe_fingerprint(fp);
        }

        // The fingerprint hash 10 (i=1) was evicted; re-observing it should
        // count as 1, not 2, so it must not trigger a repeat signal.
        let fp_reobserved = ToolFingerprint::from_hashes(String::from("t"), 1, 10);
        let signals = detector.observe_fingerprint(fp_reobserved);
        assert!(
            !signals
                .iter()
                .any(|s| s.reason_code == LoopReasonCode::ToolFingerprintRepeat),
            "evicted fingerprint must not carry over its count"
        );
    }

    #[test]
    fn tool_loop_detector_signals_accumulates() {
        let mut detector = ToolLoopDetector::new(16);
        let fp = ToolFingerprint::from_hashes(String::from("read"), 10, 100);

        for _ in 0..3 {
            detector.observe_fingerprint(fp.clone());
        }

        assert_eq!(detector.signals().len(), 1);
        assert_eq!(
            detector.signals()[0].reason_code,
            LoopReasonCode::ToolFingerprintRepeat
        );
    }

    #[test]
    fn tool_fingerprint_from_call_normalizes_paths() {
        // Two reads of different absolute paths should collapse to the same
        // canonical hash after path normalization.
        let a = ToolFingerprint::from_call("read_file", r#"{"path":"/home/user/src/main.rs"}"#);
        let b = ToolFingerprint::from_call("read_file", r#"{"path":"/tmp/work/src/main.rs"}"#);

        assert_eq!(
            a.canonical_args_hash, b.canonical_args_hash,
            "paths should normalize to the same canonical hash"
        );
        assert_eq!(a.fingerprint_hash, b.fingerprint_hash);
    }

    #[test]
    fn tool_fingerprint_from_call_distinct_tool_names_diverge() {
        let a = ToolFingerprint::from_call("read", r#"{"path":"/x.rs"}"#);
        let b = ToolFingerprint::from_call("write", r#"{"path":"/x.rs"}"#);

        assert_ne!(
            a.fingerprint_hash, b.fingerprint_hash,
            "different tool names must produce different fingerprints"
        );
    }

    #[test]
    fn tool_fingerprint_from_call_invalid_json_still_hashes() {
        let fp = ToolFingerprint::from_call("shell", "not valid json {");
        // Should still produce a deterministic non-zero hash.
        assert_ne!(fp.canonical_args_hash, 0);
    }

    #[test]
    fn new_loop_reason_codes_have_stable_labels() {
        assert_eq!(
            LoopReasonCode::ToolFingerprintRepeat.as_str(),
            "tool_fingerprint_repeat"
        );
        assert_eq!(
            LoopReasonCode::ToolAlternationCycle.as_str(),
            "tool_alternation_cycle"
        );
        assert_eq!(
            LoopReasonCode::ToolOutputBlockedEcho.as_str(),
            "tool_output_blocked_echo"
        );
    }

    fn test_loop_config() -> LoopGuardConfig {
        LoopGuardConfig {
            enabled: true,
            mode: LoopGuardMode::Enforce,
            normalized_input_window_secs: 120,
            max_repeated_inputs: 1,
            output_repeated_line_threshold: 4,
            output_token_window_size: 4,
            output_repeated_token_window_threshold: 4,
            output_suffix_cycle_threshold: 8,
            output_low_progress_min_bytes: 1_000_000,
            output_low_progress_unique_ratio_percent: 0,
            input_overlap_threshold_multiplier: 3,
            reasoning_semantic_detection_enabled: true,
            reasoning_semantic_similarity_threshold_percent: 55,
            reasoning_semantic_window_token_count: 24,
            reasoning_semantic_minimum_token_count: 8,
            reasoning_semantic_history_window_count: 16,
            on_reasoning_loop: crate::settings::LoopFailurePolicy::default(),
            embedding: crate::settings::LoopGuardEmbeddingConfig::default(),
        }
    }
}
