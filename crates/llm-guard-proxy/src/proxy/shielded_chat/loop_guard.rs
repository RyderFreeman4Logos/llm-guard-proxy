use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
};

use axum::body::Bytes;
use llm_guard_proxy_core::LoopGuardConfig;
use serde_json::Value;

const LOOP_MIN_LINE_CHARS: usize = 8;
const LOOP_MAX_PENDING_LINE_BYTES: usize = 8 * 1024;
const LOOP_MAX_RECENT_CHARS: usize = 4 * 1024;
const LOOP_MAX_TOKEN_BYTES: usize = 128;
const LOOP_SUFFIX_MIN_UNIT_CHARS: usize = 4;
const LOOP_SUFFIX_MAX_UNIT_CHARS: usize = 64;
const LOOP_INPUT_LINE_COUNT_CAP: usize = 4_096;
const LOOP_INPUT_TOKEN_WINDOW_COUNT_CAP: usize = 8_192;
const LOOP_OUTPUT_LINE_COUNT_CAP: usize = 4_096;
const LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP: usize = 8_192;
const LOOP_OUTPUT_UNIQUE_TOKEN_WINDOW_CAP: usize = 8_192;
const FNV64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Stream aggregation failure with bounded response metadata for observability.
#[derive(Clone, Debug)]
pub(in crate::proxy) struct AggregationError {
    message: String,
    response_metadata: BTreeMap<String, String>,
}

impl AggregationError {
    pub(super) fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            response_metadata: BTreeMap::new(),
        }
    }

    fn loop_detected(detection: &LoopDetection) -> Self {
        Self {
            message: detection.message(),
            response_metadata: detection.metadata(),
        }
    }

    pub(in crate::proxy) fn response_metadata(&self) -> &BTreeMap<String, String> {
        &self.response_metadata
    }
}

impl fmt::Display for AggregationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Immutable loop-inspection inputs captured from the hot-reload config snapshot.
#[derive(Clone, Debug)]
pub(in crate::proxy) struct LoopInspectionContext {
    config: LoopGuardConfig,
    input_profile: InputRepetitionProfile,
}

impl LoopInspectionContext {
    pub(in crate::proxy) fn from_request_body(
        config: &LoopGuardConfig,
        request_body: &Bytes,
    ) -> Self {
        let input_profile = if config.enabled {
            InputRepetitionProfile::from_request_body(request_body, config.output_token_window_size)
        } else {
            InputRepetitionProfile::default()
        };
        Self {
            config: config.clone(),
            input_profile,
        }
    }

    pub(in crate::proxy) fn empty(config: &LoopGuardConfig) -> Self {
        Self {
            config: config.clone(),
            input_profile: InputRepetitionProfile::default(),
        }
    }

    pub(super) fn detector(&self) -> Option<LoopDetector> {
        self.config
            .enabled
            .then(|| LoopDetector::new(self.config.clone(), self.input_profile.clone()))
    }
}

#[derive(Clone, Debug, Default)]
struct InputRepetitionProfile {
    repeated_line_hashes: BTreeSet<u64>,
    repeated_token_window_hashes: BTreeSet<u64>,
    state_capping: LoopStateCapping,
}

impl InputRepetitionProfile {
    fn from_request_body(request_body: &Bytes, token_window_size: u32) -> Self {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LoopChannel {
    Content,
    Reasoning,
    ToolCallArguments,
}

impl LoopChannel {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::Reasoning => "reasoning",
            Self::ToolCallArguments => "tool_call_arguments",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoopSignal {
    RepeatedLine,
    RepeatedTokenWindow,
    SuffixCycle,
    LowProgressGrowth,
}

impl LoopSignal {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RepeatedLine => "repeated_line",
            Self::RepeatedTokenWindow => "repeated_token_window",
            Self::SuffixCycle => "suffix_cycle",
            Self::LowProgressGrowth => "low_progress_growth",
        }
    }
}

#[derive(Clone, Debug)]
struct LoopDetection {
    signal: LoopSignal,
    channel: LoopChannel,
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
    state_capping: LoopStateCapping,
}

impl LoopDetection {
    fn message(&self) -> String {
        format!(
            "loop guard detected {} in {}: count={} threshold={} hash={}",
            self.signal.as_str(),
            self.channel.as_str(),
            self.observed_count,
            self.threshold,
            format_hash(self.sample_hash),
        )
    }

    fn metadata(&self) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::from([
            (String::from("loop_detected"), String::from("true")),
            (String::from("loop_signal"), self.signal.as_str().to_owned()),
            (
                String::from("loop_channel"),
                self.channel.as_str().to_owned(),
            ),
            (
                String::from("loop_observed_count"),
                self.observed_count.to_string(),
            ),
            (String::from("loop_threshold"), self.threshold.to_string()),
            (
                String::from("loop_observed_bytes"),
                self.observed_bytes.to_string(),
            ),
            (
                String::from("loop_fragment_count"),
                self.fragment_count.to_string(),
            ),
            (
                String::from("loop_sample_hash"),
                format_hash(self.sample_hash),
            ),
            (
                String::from("loop_input_overlap_applied"),
                self.input_overlap_applied.to_string(),
            ),
        ]);
        if let Some(token_window_size) = self.token_window_size {
            metadata.insert(
                String::from("loop_token_window_size"),
                token_window_size.to_string(),
            );
        }
        if let Some(unique_ratio_percent) = self.unique_ratio_percent {
            metadata.insert(
                String::from("loop_unique_ratio_percent"),
                unique_ratio_percent.to_string(),
            );
        }
        if let Some(unique_window_count) = self.unique_window_count {
            metadata.insert(
                String::from("loop_unique_window_count"),
                unique_window_count.to_string(),
            );
        }
        if let Some(total_window_count) = self.total_window_count {
            metadata.insert(
                String::from("loop_total_window_count"),
                total_window_count.to_string(),
            );
        }
        self.state_capping.insert_metadata(&mut metadata);
        metadata
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LoopStateCapping {
    input_lines: u64,
    input_token_windows: u64,
    output_lines: u64,
    output_token_windows: u64,
    output_unique_windows: u64,
}

impl LoopStateCapping {
    const fn is_capped(self) -> bool {
        self.input_lines > 0
            || self.input_token_windows > 0
            || self.output_lines > 0
            || self.output_token_windows > 0
            || self.output_unique_windows > 0
    }

    fn insert_metadata(self, metadata: &mut BTreeMap<String, String>) {
        if !self.is_capped() {
            return;
        }
        metadata.insert(
            String::from("loop_guard_state_capped"),
            String::from("true"),
        );
        insert_capped_metadata(metadata, "loop_input_line_count_capped", self.input_lines);
        insert_capped_metadata(
            metadata,
            "loop_input_token_window_count_capped",
            self.input_token_windows,
        );
        insert_capped_metadata(metadata, "loop_output_line_count_capped", self.output_lines);
        insert_capped_metadata(
            metadata,
            "loop_output_token_window_count_capped",
            self.output_token_windows,
        );
        insert_capped_metadata(
            metadata,
            "loop_output_unique_token_window_capped",
            self.output_unique_windows,
        );
    }
}

fn insert_capped_metadata(metadata: &mut BTreeMap<String, String>, key: &str, value: u64) {
    if value > 0 {
        metadata.insert(key.to_owned(), value.to_string());
    }
}

#[derive(Debug)]
pub(super) struct LoopDetector {
    config: LoopGuardConfig,
    input_profile: InputRepetitionProfile,
    content: LoopChannelState,
    reasoning: LoopChannelState,
    tool_call_arguments: LoopChannelState,
}

impl LoopDetector {
    fn new(config: LoopGuardConfig, input_profile: InputRepetitionProfile) -> Self {
        Self {
            config,
            input_profile,
            content: LoopChannelState::default(),
            reasoning: LoopChannelState::default(),
            tool_call_arguments: LoopChannelState::default(),
        }
    }

    fn observe(&mut self, channel: LoopChannel, fragment: &str) -> Result<(), AggregationError> {
        if fragment.is_empty() {
            return Ok(());
        }
        let state = match channel {
            LoopChannel::Content => &mut self.content,
            LoopChannel::Reasoning => &mut self.reasoning,
            LoopChannel::ToolCallArguments => &mut self.tool_call_arguments,
        };
        if let Some(detection) = state.observe(channel, fragment, &self.config, &self.input_profile)
        {
            return Err(AggregationError::loop_detected(&detection));
        }
        Ok(())
    }
}

pub(super) fn observe_fragment(
    loop_detector: &mut Option<LoopDetector>,
    channel: LoopChannel,
    fragment: &str,
) -> Result<(), AggregationError> {
    if let Some(loop_detector) = loop_detector {
        loop_detector.observe(channel, fragment)?;
    }
    Ok(())
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
}

impl LoopChannelState {
    fn observe(
        &mut self,
        channel: LoopChannel,
        fragment: &str,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
        self.fragment_count = self.fragment_count.saturating_add(1);
        self.bytes_seen = self
            .bytes_seen
            .saturating_add(u64::try_from(fragment.len()).unwrap_or(u64::MAX));

        if let Some(detection) = self.observe_lines(channel, fragment, config, input_profile) {
            return Some(detection);
        }
        if let Some(detection) = self.observe_tokens(channel, fragment, config, input_profile) {
            return Some(detection);
        }
        self.observe_recent_chars(fragment);
        if let Some(detection) = self.observe_suffix_cycle(channel, config, input_profile) {
            return Some(detection);
        }
        self.observe_low_progress(channel, config, input_profile)
    }

    fn observe_lines(
        &mut self,
        channel: LoopChannel,
        fragment: &str,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
        for character in fragment.chars() {
            if character == '\n' {
                let detection = self.finish_line(channel, config, input_profile);
                self.pending_line.clear();
                if detection.is_some() {
                    return detection;
                }
            } else if character != '\r' && self.pending_line.len() < LOOP_MAX_PENDING_LINE_BYTES {
                self.pending_line.push(character);
            }
        }
        None
    }

    fn finish_line(
        &mut self,
        channel: LoopChannel,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
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
        (u64::from(count) >= threshold).then(|| LoopDetection {
            signal: LoopSignal::RepeatedLine,
            channel,
            observed_count: u64::from(count),
            threshold,
            observed_bytes: self.bytes_seen,
            fragment_count: self.fragment_count,
            sample_hash: hash,
            input_overlap_applied: input_overlap,
            token_window_size: None,
            unique_ratio_percent: None,
            unique_window_count: None,
            total_window_count: None,
            state_capping: self.state_capping(input_profile),
        })
    }

    fn observe_tokens(
        &mut self,
        channel: LoopChannel,
        fragment: &str,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
        for character in fragment.chars() {
            if character.is_whitespace() {
                if let Some(detection) = self.finish_token(channel, config, input_profile) {
                    return Some(detection);
                }
            } else if self.current_token.len() < LOOP_MAX_TOKEN_BYTES {
                for lower in character.to_lowercase() {
                    self.current_token.push(lower);
                }
            }
        }
        None
    }

    fn finish_token(
        &mut self,
        channel: LoopChannel,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
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
        (u64::from(count) >= threshold).then(|| LoopDetection {
            signal: LoopSignal::RepeatedTokenWindow,
            channel,
            observed_count: u64::from(count),
            threshold,
            observed_bytes: self.bytes_seen,
            fragment_count: self.fragment_count,
            sample_hash: window_hash,
            input_overlap_applied: input_overlap,
            token_window_size: Some(config.output_token_window_size),
            unique_ratio_percent: None,
            unique_window_count: Some(
                u64::try_from(self.unique_token_windows.len()).unwrap_or(u64::MAX),
            ),
            total_window_count: Some(self.token_window_total),
            state_capping: self.state_capping(input_profile),
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
        channel: LoopChannel,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
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
        (cycle.repetitions >= threshold).then_some(LoopDetection {
            signal: LoopSignal::SuffixCycle,
            channel,
            observed_count: cycle.repetitions,
            threshold,
            observed_bytes: self.bytes_seen,
            fragment_count: self.fragment_count,
            sample_hash: cycle.unit_hash,
            input_overlap_applied: input_overlap,
            token_window_size: None,
            unique_ratio_percent: None,
            unique_window_count: None,
            total_window_count: None,
            state_capping: self.state_capping(input_profile),
        })
    }

    fn observe_low_progress(
        &mut self,
        channel: LoopChannel,
        config: &LoopGuardConfig,
        input_profile: &InputRepetitionProfile,
    ) -> Option<LoopDetection> {
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
        Some(LoopDetection {
            signal: LoopSignal::LowProgressGrowth,
            channel,
            observed_count: self.token_window_total,
            threshold: min_bytes,
            observed_bytes: self.bytes_seen,
            fragment_count: self.fragment_count,
            sample_hash: stable_hash_u64s(self.unique_token_windows.iter().copied()),
            input_overlap_applied: self.input_overlap_seen,
            token_window_size: Some(config.output_token_window_size),
            unique_ratio_percent: Some(unique_ratio_percent),
            unique_window_count: Some(unique_count),
            total_window_count: Some(self.token_window_total),
            state_capping: self.state_capping(input_profile),
        })
    }

    fn adjusted_threshold(threshold: u64, input_overlap: bool, config: &LoopGuardConfig) -> u64 {
        if input_overlap {
            threshold.saturating_mul(u64::from(config.input_overlap_threshold_multiplier))
        } else {
            threshold
        }
    }

    fn state_capping(&self, input_profile: &InputRepetitionProfile) -> LoopStateCapping {
        let mut capping = input_profile.state_capping;
        capping.output_lines = self.line_count_capped;
        capping.output_token_windows = self.token_window_count_capped;
        capping.output_unique_windows = self.unique_token_window_capped;
        capping
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuffixCycle {
    unit_hash: u64,
    repetitions: u64,
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
        if character.is_whitespace() {
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

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    #[test]
    fn content_token_window_loop_is_detected() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_token_window_threshold = 3;
        config.output_suffix_cycle_threshold = 100;
        let mut detector = LoopDetector::new(config, InputRepetitionProfile::default());

        detector
            .observe(LoopChannel::Content, "alpha beta ")
            .expect("first window should pass");
        detector
            .observe(LoopChannel::Content, "alpha beta ")
            .expect("second window should pass");
        let error = detector
            .observe(LoopChannel::Content, "alpha beta ")
            .expect_err("third repeated token window should trip");

        let metadata = error.response_metadata();
        assert_eq!(metadata["loop_signal"], "repeated_token_window");
        assert_eq!(metadata["loop_channel"], "content");
        assert!(metadata["loop_sample_hash"].starts_with("fnv64:"));
    }

    #[test]
    fn tool_call_argument_suffix_cycle_is_detected() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 100;
        config.output_repeated_token_window_threshold = 100;
        config.output_suffix_cycle_threshold = 4;
        let mut detector = LoopDetector::new(config, InputRepetitionProfile::default());

        detector
            .observe(
                LoopChannel::ToolCallArguments,
                r#"{"q":"x"}{"q":"x"}{"q":"x"}"#,
            )
            .expect("three suffix cycles should pass");
        let error = detector
            .observe(LoopChannel::ToolCallArguments, r#"{"q":"x"}"#)
            .expect_err("fourth suffix cycle should trip");

        let metadata = error.response_metadata();
        assert_eq!(metadata["loop_signal"], "suffix_cycle");
        assert_eq!(metadata["loop_channel"], "tool_call_arguments");
    }

    #[test]
    fn repeated_input_overlap_multiplies_line_threshold() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 3;
        config.output_repeated_token_window_threshold = 100;
        config.output_suffix_cycle_threshold = 100;
        config.input_overlap_threshold_multiplier = 2;
        let repeated =
            String::from("legitimate repeated input line\nlegitimate repeated input line\n");
        let profile =
            InputRepetitionProfile::from_texts(&[repeated], config.output_token_window_size);
        let mut detector = LoopDetector::new(config, profile);

        for _ in 0..5 {
            detector
                .observe(LoopChannel::Reasoning, "legitimate repeated input line\n")
                .expect("overlap should raise the threshold");
        }
        let error = detector
            .observe(LoopChannel::Reasoning, "legitimate repeated input line\n")
            .expect_err("multiplied threshold should trip on the sixth line");

        let metadata = error.response_metadata();
        assert_eq!(metadata["loop_signal"], "repeated_line");
        assert_eq!(metadata["loop_threshold"], "6");
        assert_eq!(metadata["loop_input_overlap_applied"], "true");
    }

    #[test]
    fn low_progress_growth_is_detected_after_minimum_bytes() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_line_threshold = 100;
        config.output_repeated_token_window_threshold = 100;
        config.output_suffix_cycle_threshold = 100;
        config.output_low_progress_min_bytes = 24;
        config.output_low_progress_unique_ratio_percent = 40;
        let mut detector = LoopDetector::new(config, InputRepetitionProfile::default());

        let error = detector
            .observe(
                LoopChannel::Content,
                "alpha beta alpha beta alpha beta alpha beta ",
            )
            .expect_err("low-progress repeated growth should trip");

        let metadata = error.response_metadata();
        assert_eq!(metadata["loop_signal"], "low_progress_growth");
        assert_eq!(metadata["loop_channel"], "content");
        assert_eq!(metadata["loop_token_window_size"], "2");
        assert_eq!(metadata["loop_unique_ratio_percent"], "28");
    }

    #[test]
    fn input_profile_caps_high_cardinality_lines_and_token_windows() {
        let mut text = String::new();
        text.push_str("tracked repeated input line\ntracked repeated input line\n");
        for index in 0..LOOP_INPUT_TOKEN_WINDOW_COUNT_CAP.saturating_add(32) {
            writeln!(
                &mut text,
                "unique input line {index} token-{index} value-{index}"
            )
            .expect("writing to String should not fail");
        }

        let profile = InputRepetitionProfile::from_texts(&[text], 2);

        assert!(
            profile
                .contains_line_hash(normalized_line_hash("tracked repeated input line").unwrap())
        );
        assert!(profile.state_capping.is_capped());
        assert!(profile.state_capping.input_lines > 0);
        assert!(profile.state_capping.input_token_windows > 0);
        assert!(profile.repeated_line_hashes.len() <= LOOP_INPUT_LINE_COUNT_CAP);
        assert!(profile.repeated_token_window_hashes.len() <= LOOP_INPUT_TOKEN_WINDOW_COUNT_CAP);
    }

    #[test]
    fn output_channel_state_caps_high_cardinality_lines_and_token_windows() {
        let mut config = test_loop_config();
        config.output_token_window_size = 2;
        config.output_repeated_line_threshold = u32::MAX;
        config.output_repeated_token_window_threshold = u32::MAX;
        config.output_suffix_cycle_threshold = u32::MAX;
        config.output_low_progress_min_bytes = u64::MAX;
        let mut detector = LoopDetector::new(config, InputRepetitionProfile::default());

        for index in 0..LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP.saturating_add(32) {
            detector
                .observe(
                    LoopChannel::Content,
                    &format!("unique output line {index} token-{index} value-{index}\n"),
                )
                .expect("high-cardinality output should degrade without aborting");
        }

        assert_eq!(
            detector.content.line_counts.len(),
            LOOP_OUTPUT_LINE_COUNT_CAP
        );
        assert_eq!(
            detector.content.token_window_counts.len(),
            LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP
        );
        assert_eq!(
            detector.content.unique_token_windows.len(),
            LOOP_OUTPUT_UNIQUE_TOKEN_WINDOW_CAP
        );
        assert!(detector.content.line_count_capped > 0);
        assert!(detector.content.token_window_count_capped > 0);
        assert!(detector.content.unique_token_window_capped > 0);
        assert!(detector.content.token_window_total > LOOP_OUTPUT_TOKEN_WINDOW_COUNT_CAP as u64);
    }

    #[test]
    fn tracked_repeated_line_still_detects_after_output_line_cap() {
        let mut config = test_loop_config();
        config.output_repeated_line_threshold = 3;
        config.output_repeated_token_window_threshold = u32::MAX;
        config.output_suffix_cycle_threshold = u32::MAX;
        config.output_low_progress_min_bytes = u64::MAX;
        let mut detector = LoopDetector::new(config, InputRepetitionProfile::default());

        detector
            .observe(LoopChannel::Reasoning, "tracked repeated output line\n")
            .expect("first tracked line should pass");
        detector
            .observe(LoopChannel::Reasoning, "tracked repeated output line\n")
            .expect("second tracked line should pass");
        for index in 0..LOOP_OUTPUT_LINE_COUNT_CAP.saturating_add(16) {
            detector
                .observe(
                    LoopChannel::Reasoning,
                    &format!("unique capped output line {index}\n"),
                )
                .expect("new unique lines beyond the cap should be skipped");
        }

        let error = detector
            .observe(LoopChannel::Reasoning, "tracked repeated output line\n")
            .expect_err("existing tracked line should still count after cap");
        let metadata = error.response_metadata();
        assert_eq!(metadata["loop_signal"], "repeated_line");
        assert_eq!(metadata["loop_channel"], "reasoning");
        assert_eq!(metadata["loop_guard_state_capped"], "true");
        assert!(
            metadata["loop_output_line_count_capped"]
                .parse::<u64>()
                .expect("capped count should be numeric")
                > 0
        );
    }

    fn test_loop_config() -> LoopGuardConfig {
        LoopGuardConfig {
            enabled: true,
            normalized_input_window_secs: 120,
            max_repeated_inputs: 1,
            output_repeated_line_threshold: 4,
            output_token_window_size: 4,
            output_repeated_token_window_threshold: 4,
            output_suffix_cycle_threshold: 8,
            output_low_progress_min_bytes: 1_000_000,
            output_low_progress_unique_ratio_percent: 0,
            input_overlap_threshold_multiplier: 3,
        }
    }
}
