//! Integration tests for the loop-detector fixture corpus derived from
//! AEON issue #14 repeated-thinking evaluation artifacts.
//!
//! These tests load JSON fixture files from `tests/fixtures/loop_detector/`
//! and feed their synthetic reasoning fragments to `ChannelizedLoopDetector`,
//! asserting expected severity outcomes for each fixture class:
//!
//! - **Hard positives**: must produce `AbortCandidate` signals
//! - **Mild positives**: must produce at most `Suspect` (never abort)
//! - **Clean negatives**: must produce no loop signals

use std::fs;
use std::path::PathBuf;

use llm_guard_proxy_core::{
    ChannelizedLoopDetector, LoopDetector, LoopDetectorInput, LoopGuardConfig, LoopGuardMode,
    LoopInputProfile, LoopSeverity, StreamChannel,
};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("loop_detector")
}

fn parse_channel(s: &str) -> StreamChannel {
    match s {
        "content" => StreamChannel::Content,
        "tool_arguments" => StreamChannel::ToolArguments,
        _ => StreamChannel::Reasoning,
    }
}

fn test_loop_config() -> LoopGuardConfig {
    let mut config = LoopGuardConfig {
        enabled: true,
        mode: LoopGuardMode::Enforce,
        ..LoopGuardConfig::default()
    };
    // Calibrate thresholds to AEON evaluation parameters: the original
    // evaluation flagged hard loops at line repetition >= 3 (conservative
    // within-stream detection), token-window repetition >= 3, and suffix
    // cycle >= 10.
    config.output_repeated_line_threshold = 3;
    config.output_repeated_token_window_threshold = 3;
    config.output_suffix_cycle_threshold = 10;
    config
}

/// Runs a fixture through the detector and returns the maximum severity observed.
fn run_fixture(fixture: &Value) -> LoopSeverity {
    let config = test_loop_config();
    let mut detector = ChannelizedLoopDetector::new(config, LoopInputProfile::default());

    let channel_str = fixture
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or("reasoning");
    let channel = parse_channel(channel_str);

    let mut max_severity = LoopSeverity::Observe;

    if let Some(fragments) = fixture.get("fragments").and_then(Value::as_array) {
        for fragment in fragments {
            if let Some(text) = fragment.as_str() {
                let signals = detector.observe(LoopDetectorInput::fragment(channel, text));
                for signal in signals {
                    if signal.severity > max_severity {
                        max_severity = signal.severity;
                    }
                }
            }
        }
    }

    let summary = detector.finish();
    for signal in &summary.signals {
        if signal.severity > max_severity {
            max_severity = signal.severity;
        }
    }

    max_severity
}

fn load_category(prefix: &str) -> Vec<(String, Value)> {
    let dir = fixture_dir();
    let mut fixtures = Vec::new();

    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("failed to read fixture dir: {e}"));

    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.starts_with(prefix) && fname_str.ends_with(".json") {
            let data = fs::read_to_string(entry.path())
                .unwrap_or_else(|e| panic!("failed to read {fname_str}: {e}"));
            let fixture: Value = serde_json::from_str(&data)
                .unwrap_or_else(|e| panic!("failed to parse {fname_str}: {e}"));
            fixtures.push((fname_str.to_string(), fixture));
        }
    }

    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

// ── Hard positive tests ──────────────────────────────────────────

#[test]
fn hard_positives_produce_abort_candidate_signals() {
    let fixtures = load_category("hard_positives_");
    assert!(
        !fixtures.is_empty(),
        "expected at least one hard positive fixture"
    );

    for (name, fixture) in &fixtures {
        let severity = run_fixture(fixture);
        let id = fixture
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        assert_eq!(
            severity,
            LoopSeverity::AbortCandidate,
            "hard positive fixture {name} ({id}) expected AbortCandidate, got {severity:?}"
        );
    }
}

// ── Mild positive tests ──────────────────────────────────────────

#[test]
fn mild_positives_do_not_produce_abort_candidate_signals() {
    let fixtures = load_category("mild_positives_");
    assert!(
        !fixtures.is_empty(),
        "expected at least one mild positive fixture"
    );

    for (name, fixture) in &fixtures {
        let severity = run_fixture(fixture);
        let id = fixture
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        assert_ne!(
            severity,
            LoopSeverity::AbortCandidate,
            "mild positive fixture {name} ({id}) should not reach AbortCandidate, got {severity:?}"
        );
    }
}

// ── Clean negative tests ─────────────────────────────────────────

#[test]
fn clean_negatives_produce_no_loop_signals() {
    let fixtures = load_category("clean_negatives_");
    assert!(
        !fixtures.is_empty(),
        "expected at least one clean negative fixture"
    );

    for (name, fixture) in &fixtures {
        let severity = run_fixture(fixture);
        assert_eq!(
            severity,
            LoopSeverity::Observe,
            "clean negative fixture {name} should produce no signals (Observe), got {severity:?}"
        );
    }
}

// ── Fixture integrity tests ──────────────────────────────────────

#[test]
fn all_fixtures_have_required_fields() {
    let dir = fixture_dir();
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("failed to read fixture dir: {e}"));

    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();

        if !fname_str.ends_with(".json") || fname_str == "index.json" {
            continue;
        }

        let data = fs::read_to_string(entry.path())
            .unwrap_or_else(|e| panic!("failed to read {fname_str}: {e}"));
        let fixture: Value = serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("failed to parse {fname_str}: {e}"));

        let fragments = fixture
            .get("fragments")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("fixture {fname_str} missing fragments array"));
        assert!(
            !fragments.is_empty(),
            "fixture {fname_str} has no fragments"
        );

        let expected = fixture
            .get("expected_severity")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("fixture {fname_str} missing expected_severity"));
        assert!(
            !expected.is_empty(),
            "fixture {fname_str} empty expected_severity"
        );
    }
}

#[test]
fn fixture_corpus_has_balanced_coverage() {
    let hard = load_category("hard_positives_");
    let mild = load_category("mild_positives_");
    let clean = load_category("clean_negatives_");

    assert!(!hard.is_empty(), "missing hard positive fixtures");
    assert!(!mild.is_empty(), "missing mild positive fixtures");
    assert!(!clean.is_empty(), "missing clean negative fixtures");
}
