use std::{collections::BTreeMap, fmt};

use axum::body::Bytes;
use llm_guard_proxy_core::{
    ChannelizedLoopDetector, DetectorSummary, LoopDetector as CoreLoopDetector, LoopDetectorInput,
    LoopGuardConfig, LoopGuardMode, LoopInputProfile, LoopSignal, RawPayloads, StreamChannel,
    ToolCallFingerprintInput,
};

/// Stream aggregation failure with bounded response metadata for observability.
#[derive(Clone, Debug)]
pub(in crate::proxy) struct AggregationError {
    message: String,
    response_metadata: BTreeMap<String, String>,
    raw_payloads: Box<RawPayloads>,
}

impl AggregationError {
    pub(super) fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            response_metadata: BTreeMap::new(),
            raw_payloads: Box::default(),
        }
    }

    pub(super) fn upstream_stall(idle_timeout_ms: u64) -> Self {
        Self {
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

    fn loop_detected(signal: &LoopSignal) -> Self {
        Self {
            message: loop_detection_message(signal),
            response_metadata: signal.legacy_abort_metadata(),
            raw_payloads: Box::default(),
        }
    }

    pub(super) fn with_raw_payloads(mut self, raw_payloads: RawPayloads) -> Self {
        self.raw_payloads = Box::new(raw_payloads);
        self
    }

    pub(in crate::proxy) fn response_metadata(&self) -> &BTreeMap<String, String> {
        &self.response_metadata
    }

    pub(in crate::proxy) fn raw_payloads(&self) -> &RawPayloads {
        self.raw_payloads.as_ref()
    }

    pub(in crate::proxy) fn is_loop_detected(&self) -> bool {
        self.response_metadata
            .get("loop_detected")
            .is_some_and(|value| value == "true")
    }

    pub(in crate::proxy) fn is_upstream_stall(&self) -> bool {
        self.response_metadata
            .get("upstream_stall_detected")
            .is_some_and(|value| value == "true")
    }

    pub(in crate::proxy) fn transient_stream_retry_reason(&self) -> Option<&'static str> {
        if self.is_upstream_stall() {
            Some("upstream_stall")
        } else if self
            .message
            .contains("upstream SSE stream failed: timeout_failure")
            || self
                .message
                .contains("upstream SSE stream failed: connect_failure")
            || self
                .message
                .contains("upstream SSE stream failed: body_failure")
            || self
                .message
                .contains("upstream SSE stream failed: unknown_failure")
        {
            Some("transient_upstream_stream_failure")
        } else {
            None
        }
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
            return Err(AggregationError::loop_detected(signal));
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
