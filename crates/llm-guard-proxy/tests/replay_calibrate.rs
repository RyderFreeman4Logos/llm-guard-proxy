//! Integration tests for replay calibration (issue #112).
//!
//! `CalibrateCommand` and `ReplayError` live in the binary crate and are
//! covered by inline unit tests. These integration tests exercise the public,
//! crate-agnostic replay types that the calibrate command and the rest of the
//! proxy consume.

use llm_guard_proxy_core::replay::{
    CalibrationResult, RecordDetectionResult, ReplayChannel, ReplayConfig, ReplayRecord,
    SourceCalibration, SseEvent,
};

#[test]
fn replay_record_round_trips_through_json() {
    let record = ReplayRecord {
        record_id: String::from("rec-42"),
        source: String::from("stress_run"),
        is_labeled_loop: true,
        label_severity: Some(String::from("hard")),
        sse_events: vec![SseEvent {
            channel: ReplayChannel::Reasoning,
            text: String::from("repeating reasoning"),
            token_offset: 10,
        }],
        generated_token_count: 100,
    };
    let json = serde_json::to_string(&record).expect("serialize record");
    let parsed: ReplayRecord = serde_json::from_str(&json).expect("deserialize record");
    assert_eq!(parsed.record_id, "rec-42");
    assert_eq!(parsed.source, "stress_run");
    assert!(parsed.is_labeled_loop);
    assert_eq!(parsed.label_severity.as_deref(), Some("hard"));
    assert_eq!(parsed.sse_events.len(), 1);
    assert_eq!(parsed.sse_events[0].channel, ReplayChannel::Reasoning);
    assert_eq!(parsed.sse_events[0].token_offset, 10);
}

#[test]
fn replay_channel_serializes_as_snake_case() {
    let json = serde_json::to_string(&ReplayChannel::ToolArgs).expect("serialize");
    assert_eq!(json, "\"tool_args\"");
    let parsed: ReplayChannel = serde_json::from_str("\"tool_output\"").expect("deserialize");
    assert_eq!(parsed, ReplayChannel::ToolOutput);
}

#[test]
fn calibration_result_round_trips_through_json() {
    let result = CalibrationResult {
        total_records: 5,
        labeled_hard_loops: 2,
        labeled_mild_loops: 1,
        detected_hard_loops: 2,
        detected_mild_loops: 1,
        false_positives: 0,
        true_negatives: 2,
        hard_loop_recall: 1.0,
        mild_loop_recall: 1.0,
        false_positive_rate: 0.0,
        median_time_to_detection_tokens: 320,
        median_time_to_detection_ms: 450,
        prevented_token_estimate: 8_000,
        per_source: vec![SourceCalibration::from_counts("stress_run", 3, 2, 0, 2)],
    };
    let json = serde_json::to_string(&result).expect("serialize");
    let parsed: CalibrationResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.total_records, 5);
    assert_eq!(parsed.detected_hard_loops, 2);
    assert_eq!(parsed.per_source.len(), 1);
    assert_eq!(parsed.per_source[0].source, "stress_run");
}

#[test]
fn calibration_from_results_classifies_true_and_false_positives() {
    let records = vec![
        ReplayRecord {
            record_id: String::from("loop-1"),
            source: String::from("stress_run"),
            is_labeled_loop: true,
            label_severity: Some(String::from("hard")),
            sse_events: Vec::new(),
            generated_token_count: 200,
        },
        ReplayRecord {
            record_id: String::from("clean-1"),
            source: String::from("ab_forced_no_thinking"),
            is_labeled_loop: false,
            label_severity: Some(String::from("none")),
            sse_events: Vec::new(),
            generated_token_count: 50,
        },
    ];
    let results = vec![
        RecordDetectionResult {
            record_id: String::from("loop-1"),
            detected: true,
            detection_severity: Some(String::from("hard")),
            first_detection_token: Some(80),
            first_detection_time_ms: 120,
            prevented_tokens: 120,
            signals: vec![String::from("reasoning:repeated_line")],
            is_true_positive: true,
            is_false_positive: false,
        },
        RecordDetectionResult {
            record_id: String::from("clean-1"),
            detected: false,
            detection_severity: None,
            first_detection_token: None,
            first_detection_time_ms: 0,
            prevented_tokens: 0,
            signals: Vec::new(),
            is_true_positive: false,
            is_false_positive: false,
        },
    ];

    let calibration = CalibrationResult::from_results(&results, &records);
    assert_eq!(calibration.labeled_hard_loops, 1);
    assert_eq!(calibration.detected_hard_loops, 1);
    assert_eq!(calibration.false_positives, 0);
    assert_eq!(calibration.true_negatives, 1);
    assert!(
        (calibration.hard_loop_recall - 1.0).abs() < 0.001,
        "hard recall should be 1.0"
    );
    assert_eq!(calibration.prevented_token_estimate, 120);
}

#[test]
fn replay_config_defaults_meet_quality_targets() {
    let config = ReplayConfig::default();
    assert!((config.hard_loop_recall_target - 0.95).abs() < 0.001);
    assert!((config.mild_loop_recall_target - 0.80).abs() < 0.001);
    assert_eq!(config.hard_loop_token_budget, 12_000);
    assert_eq!(config.mild_loop_token_budget, 16_000);
}
