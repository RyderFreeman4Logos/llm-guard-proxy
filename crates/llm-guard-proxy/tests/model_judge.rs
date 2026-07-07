//! Integration tests for the model judge types (issue #110).
//!
//! The `ModelJudgeClient` itself lives in the binary crate and is covered by
//! its inline unit tests. These integration tests exercise the public,
//! crate-agnostic core types that the judge client and the rest of the proxy
//! consume.

use llm_guard_proxy_core::model_judge::{
    JudgePromptBuilder, JudgeSnapshot, LoopJudgeResult, SnapshotChannels, TaskKind,
};

#[test]
fn judge_system_prompt_mentions_loop_detector() {
    let prompt = JudgePromptBuilder::system_prompt();
    assert!(!prompt.is_empty());
    assert!(
        prompt.to_lowercase().contains("loop detector"),
        "system prompt must mention 'loop detector': {prompt}"
    );
}

#[test]
fn judge_user_prompt_embeds_snapshot_json() {
    let snapshot = JudgeSnapshot {
        request_id_hash: String::from("deadbeef"),
        task_kind_hint: TaskKind::Code,
        elapsed_ms: 2_500,
        generated_tokens: 640,
        channels: SnapshotChannels::default(),
        current_answer_candidate: None,
        known_prompt_constraints: Vec::new(),
        deterministic_signals: Vec::new(),
    };
    let prompt = JudgePromptBuilder::user_prompt(&snapshot);
    assert!(prompt.contains("deadbeef"));
    assert!(prompt.contains("<evidence>"));
}

#[test]
fn loop_judge_result_round_trips_through_json() {
    let result = LoopJudgeResult {
        is_loop: true,
        severity: llm_guard_proxy_core::model_judge::JudgeSeverity::Hard,
        confidence: 0.88,
        loop_types: vec![llm_guard_proxy_core::model_judge::LoopType::AnswerStall],
        context_rot_risk: 0.66,
        abort_now: true,
        recommended_action: llm_guard_proxy_core::model_judge::RecommendedAction::AbortAndSalvage,
        loop_start_span_id: Some(String::from("c3")),
        keep_span_ids: vec![String::from("c1")],
        drop_span_ids: vec![String::from("c2")],
        clean_state_needed: true,
        short_reason: String::from("stalling on answer"),
    };
    let json = serde_json::to_string(&result).expect("serialize LoopJudgeResult");
    let back: LoopJudgeResult = serde_json::from_str(&json).expect("deserialize LoopJudgeResult");
    assert!(back.is_loop);
    assert!(back.abort_now);
    assert!(back.clean_state_needed);
    assert_eq!(back.short_reason, "stalling on answer");
}
