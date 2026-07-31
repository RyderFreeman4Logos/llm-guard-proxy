//! Honest liveness metadata for correctness-first held responses.

use super::{BTreeMap, ShieldedLivenessMode, ShieldedLivenessSelection};

const EFFECTIVE_MODE: &str = "held";

pub(super) fn add_request_metadata(
    request_metadata: &mut BTreeMap<String, String>,
    liveness: &ShieldedLivenessSelection,
) {
    request_metadata.extend(metadata(liveness));
}

pub(super) fn response_metadata(
    liveness: &ShieldedLivenessSelection,
    upstream_content_type: Option<String>,
) -> BTreeMap<String, String> {
    let mut metadata = metadata(liveness);
    metadata.insert(
        String::from("downstream_heartbeat_emitted_count"),
        String::from("0"),
    );
    metadata.insert(
        String::from("shielded_downstream_streaming"),
        (liveness.mode == ShieldedLivenessMode::Sse).to_string(),
    );
    if let Some(content_type) = upstream_content_type {
        metadata.insert(
            String::from("upstream_response_header_content-type"),
            content_type,
        );
    }
    metadata
}

fn metadata(liveness: &ShieldedLivenessSelection) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            String::from("downstream_liveness_mode"),
            String::from(EFFECTIVE_MODE),
        ),
        (
            String::from("downstream_liveness_configured_mode"),
            liveness.configured_mode.as_str().to_owned(),
        ),
        (
            String::from("downstream_liveness_framing_mode"),
            liveness.mode.as_str().to_owned(),
        ),
        (
            String::from("downstream_liveness_effective_mode"),
            String::from(EFFECTIVE_MODE),
        ),
        (
            String::from("heartbeat_interval_secs"),
            liveness.heartbeat_interval_secs.to_string(),
        ),
        (
            String::from("repeat_input_window_secs"),
            liveness.repeat_window_secs.to_string(),
        ),
        (
            String::from("repeat_input_max_repeated_inputs"),
            liveness.repeat_max_inputs.to_string(),
        ),
        (
            String::from("input_fingerprint_present"),
            liveness.input_fingerprint.is_some().to_string(),
        ),
        (
            String::from("repeat_input_matched"),
            liveness.repeat_observation.repeated.to_string(),
        ),
        (
            String::from("repeat_input_prior_count"),
            liveness.repeat_observation.prior_count.to_string(),
        ),
    ])
}
