use std::{collections::BTreeMap, fmt};

use axum::body::Bytes;
use llm_guard_proxy_core::{
    ChannelizedLoopDetector, DetectorSummary, LoopDetector as CoreLoopDetector, LoopDetectorInput,
    LoopGuardConfig, LoopGuardMode, LoopInputProfile, LoopReasonCode, LoopSignal, StreamChannel,
    ToolCallFingerprintInput,
};
use llm_guard_proxy_state::RawPayloads;

/// Stream aggregation failure with bounded response metadata for observability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::proxy) enum AggregationFailureKind {
    BodyFailure,
    TimeoutFailure,
    ConnectFailure,
    RequestFailure,
    DecodeFailure,
    UnknownFailure,
    UpstreamStall,
    MalformedProtocol,
    LoopDetected,
}

impl AggregationFailureKind {
    pub(super) fn from_reqwest_error(error: &reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::TimeoutFailure
        } else if error.is_connect() {
            Self::ConnectFailure
        } else if error.is_body() {
            Self::BodyFailure
        } else if error.is_decode() {
            Self::DecodeFailure
        } else if error.is_request() {
            Self::RequestFailure
        } else {
            Self::UnknownFailure
        }
    }

    pub(in crate::proxy) const fn as_str(self) -> &'static str {
        match self {
            Self::BodyFailure => "body_failure",
            Self::TimeoutFailure => "timeout_failure",
            Self::ConnectFailure => "connect_failure",
            Self::RequestFailure => "request_failure",
            Self::DecodeFailure => "decode_failure",
            Self::UnknownFailure => "unknown_failure",
            Self::UpstreamStall => "upstream_stall",
            Self::MalformedProtocol => "malformed_protocol",
            Self::LoopDetected => "loop_detected",
        }
    }

    pub(in crate::proxy) const fn is_transient_stream_failure(self) -> bool {
        matches!(
            self,
            Self::BodyFailure | Self::TimeoutFailure | Self::ConnectFailure | Self::UnknownFailure
        )
    }
}

#[derive(Clone, Debug)]
pub(in crate::proxy) struct AggregationError {
    kind: AggregationFailureKind,
    message: String,
    response_metadata: BTreeMap<String, String>,
    raw_payloads: Box<RawPayloads>,
}

impl AggregationError {
    pub(super) fn plain(message: impl Into<String>) -> Self {
        Self {
            kind: AggregationFailureKind::MalformedProtocol,
            message: message.into(),
            response_metadata: BTreeMap::new(),
            raw_payloads: Box::default(),
        }
    }

    pub(super) fn upstream_stream_failure(
        kind: AggregationFailureKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            response_metadata: BTreeMap::new(),
            raw_payloads: Box::default(),
        }
    }

    pub(super) fn upstream_stall(idle_timeout_ms: u64) -> Self {
        Self {
            kind: AggregationFailureKind::UpstreamStall,
            message: format!("upstream SSE stream stalled: no chunk for {idle_timeout_ms}ms"),
            response_metadata: BTreeMap::from([
                (
                    String::from("upstream_stall_detected"),
                    String::from("true"),
                ),
                (
                    String::from("upstream_stall_idle_timeout_ms"),
                    idle_timeout_ms.to_string(),
                ),
            ]),
            raw_payloads: Box::default(),
        }
    }

    fn loop_detected(signal: &LoopSignal, summary: &DetectorSummary, mode: LoopGuardMode) -> Self {
        let mut response_metadata = signal.legacy_abort_metadata();
        response_metadata.extend(summary.metadata(mode));
        response_metadata.insert(
            String::from("loop_hard_abort_candidate"),
            String::from("true"),
        );
        response_metadata.insert(
            String::from("loop_abort_channel"),
            signal.channel.as_str().to_owned(),
        );
        response_metadata.insert(
            String::from("loop_abort_severity"),
            signal.severity.as_str().to_owned(),
        );
        Self {
            kind: AggregationFailureKind::LoopDetected,
            message: loop_detection_message(signal),
            response_metadata,
            raw_payloads: Box::default(),
        }
    }

    pub(super) fn with_raw_payloads(mut self, raw_payloads: RawPayloads) -> Self {
        self.raw_payloads = Box::new(raw_payloads);
        self
    }

    fn with_repeated_line_boundary(mut self, boundary: usize) -> Self {
        self.response_metadata.insert(
            String::from("cot_salvage_repeated_line_boundary_bytes"),
            boundary.to_string(),
        );
        self
    }

    pub(in crate::proxy) fn response_metadata(&self) -> &BTreeMap<String, String> {
        &self.response_metadata
    }

    pub(in crate::proxy) fn raw_payloads(&self) -> &RawPayloads {
        self.raw_payloads.as_ref()
    }

    pub(in crate::proxy) const fn failure_kind(&self) -> AggregationFailureKind {
        self.kind
    }

    pub(in crate::proxy) fn is_loop_detected(&self) -> bool {
        self.kind == AggregationFailureKind::LoopDetected
    }

    pub(in crate::proxy) fn is_upstream_stall(&self) -> bool {
        self.kind == AggregationFailureKind::UpstreamStall
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
    input_profile: LoopInputProfile,
}

impl LoopInspectionContext {
    pub(in crate::proxy) fn from_request_body(
        config: &LoopGuardConfig,
        request_body: &Bytes,
    ) -> Self {
        let input_profile = if config.effective_mode().is_disabled() {
            LoopInputProfile::default()
        } else {
            LoopInputProfile::from_request_body(request_body, config.output_token_window_size)
        };
        Self {
            config: config.clone(),
            input_profile,
        }
    }

    pub(in crate::proxy) fn empty(config: &LoopGuardConfig) -> Self {
        Self {
            config: config.clone(),
            input_profile: LoopInputProfile::default(),
        }
    }

    pub(super) fn detector(&self) -> Option<LoopDetector> {
        let mode = self.config.effective_mode();
        (!mode.is_disabled()).then(|| LoopDetector {
            mode,
            detector: ChannelizedLoopDetector::new(self.config.clone(), self.input_profile.clone()),
        })
    }
}

#[derive(Debug)]
pub(super) struct LoopDetector {
    mode: LoopGuardMode,
    detector: ChannelizedLoopDetector,
}

impl LoopDetector {
    fn observe_fragment(
        &mut self,
        channel: StreamChannel,
        fragment: &str,
    ) -> Result<(), AggregationError> {
        let signals = self
            .detector
            .observe(LoopDetectorInput::fragment(channel, fragment));
        self.apply_signals(&signals)
    }

    fn observe_completed_tool_call(
        &mut self,
        tool_name: &str,
        arguments: &str,
    ) -> Result<(), AggregationError> {
        let signals = self.detector.observe_tool_call(ToolCallFingerprintInput {
            tool_name,
            arguments,
        });
        self.apply_signals(&signals)
    }

    fn apply_signals(&self, signals: &[LoopSignal]) -> Result<(), AggregationError> {
        if self.mode != LoopGuardMode::Enforce {
            return Ok(());
        }
        if let Some(signal) = signals.iter().find(|signal| signal.is_abort_candidate()) {
            let summary = self.summary();
            let error = AggregationError::loop_detected(signal, &summary, self.mode);
            if signal.channel == StreamChannel::Reasoning
                && signal.reason_code == LoopReasonCode::RepeatedLine
                && let Some(boundary) = self
                    .detector
                    .repeated_line_first_byte_offset(signal.channel)
            {
                return Err(error.with_repeated_line_boundary(boundary));
            }
            return Err(error);
        }
        Ok(())
    }

    pub(super) fn summary(&self) -> DetectorSummary {
        self.detector.finish()
    }

    pub(super) const fn mode(&self) -> LoopGuardMode {
        self.mode
    }
}

pub(super) fn observe_fragment(
    loop_detector: &mut Option<LoopDetector>,
    channel: StreamChannel,
    fragment: &str,
) -> Result<(), AggregationError> {
    if let Some(loop_detector) = loop_detector {
        loop_detector.observe_fragment(channel, fragment)?;
    }
    Ok(())
}

pub(super) fn observe_completed_tool_call(
    loop_detector: &mut Option<LoopDetector>,
    tool_name: &str,
    arguments: &str,
) -> Result<(), AggregationError> {
    if let Some(loop_detector) = loop_detector {
        loop_detector.observe_completed_tool_call(tool_name, arguments)?;
    }
    Ok(())
}

fn loop_detection_message(signal: &LoopSignal) -> String {
    let hash = signal
        .feature_summary
        .fields()
        .get("sample_hash")
        .or_else(|| signal.feature_summary.fields().get("fingerprint_hash"))
        .or_else(|| signal.feature_summary.fields().get("arguments_hash"))
        .map_or("fnv64:unknown", String::as_str);
    let count = signal
        .feature_summary
        .fields()
        .get("observed_count")
        .or_else(|| signal.feature_summary.fields().get("repeat_count"))
        .map_or("0", String::as_str);
    let threshold = signal
        .feature_summary
        .fields()
        .get("threshold")
        .map_or("0", String::as_str);
    format!(
        "loop guard detected {} in {}: count={count} threshold={threshold} hash={hash}",
        signal.reason_code.as_str(),
        signal.channel.as_str(),
    )
}
