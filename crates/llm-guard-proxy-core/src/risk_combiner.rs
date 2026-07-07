//! Unified risk combiner (issue #109).
//!
//! Signals from the hash-based [`LoopSignal`] detector, the semantic self-loop
//! scorer ([`SemanticLoopSignal`]), the context-rot model
//! ([`ContextRotSignal`]), and the tool-loop detector ([`ToolLoopSignal`]) each
//! describe a different failure mode. They are not independent in the
//! statistical sense, but they *are* independent as code paths: when two or
//! more fire on the same request/attempt, the model is almost certainly stuck.
//!
//! [`RiskCombiner`] collects all detector signals for a single evaluation
//! window and folds them into a [`CombinedRisk`] that is safe to surface in
//! telemetry and to feed directly into the enforce/abort decision policy.
//!
//! ## Formula
//!
//! ```text
//! base_risk      = max(individual_signal_risks)
//! synergy_bonus  = sum of pairwise bonuses for each agreeing detector pair
//!                + 0.05 for every detector beyond the first two
//! combined_risk  = min(base_risk + synergy_bonus, 1.0)
//! ```
//!
//! Pairwise synergy is only counted between *distinct* detector kinds; a single
//! detector contributing multiple signals does not earn synergy with itself.

use crate::context_rot::ContextRotSignal;
use crate::embedding::SemanticLoopSignal;
use crate::loop_detector::{LoopSeverity, LoopSignal, ToolLoopSignal};

/// Pairwise synergy bonus added when both detectors fire.
///
/// These are tuned so that two agreeing detectors push an already-elevated base
/// risk over the abort threshold, while never producing synergy from a single
/// detector alone.
const SYNERGY_HASH_SEMANTIC: f32 = 0.15;
const SYNERGY_SEMANTIC_CONTEXT: f32 = 0.12;
const SYNERGY_HASH_TOOL: f32 = 0.10;
const SYNERGY_TOOL_CONTEXT: f32 = 0.10;

/// Extra bonus per detector beyond the first two that agree.
const SYNERGY_EXTRA_PER_DETECTOR: f32 = 0.05;

/// Which detector produced a wrapped [`DetectorSignal`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DetectorKind {
    /// Hash-based stream-loop detector ([`LoopSignal`]).
    HashLoop,
    /// Embedding-based semantic self-loop scorer ([`SemanticLoopSignal`]).
    SemanticLoop,
    /// Retained-context pollution model ([`ContextRotSignal`]).
    ContextRot,
    /// Tool-loop detector ([`ToolLoopSignal`]).
    ToolLoop,
}

/// A signal emitted by one of the loop detectors, wrapped so the combiner can
/// treat them uniformly.
#[derive(Clone, Debug)]
pub enum DetectorSignal {
    /// Hash-based stream-loop signal.
    HashLoop(LoopSignal),
    /// Semantic self-loop signal.
    SemanticLoop(SemanticLoopSignal),
    /// Context-rot signal.
    ContextRot(ContextRotSignal),
    /// Tool-loop signal.
    ToolLoop(ToolLoopSignal),
}

impl DetectorSignal {
    /// The detector kind that produced this signal.
    #[must_use]
    pub const fn kind(&self) -> DetectorKind {
        match self {
            Self::HashLoop(_) => DetectorKind::HashLoop,
            Self::SemanticLoop(_) => DetectorKind::SemanticLoop,
            Self::ContextRot(_) => DetectorKind::ContextRot,
            Self::ToolLoop(_) => DetectorKind::ToolLoop,
        }
    }

    /// The individual risk score of this signal in `[0.0, 1.0]`.
    ///
    /// Each underlying detector uses a slightly different risk representation:
    /// - `LoopSignal` carries a `confidence: u8` (0..=100) and a severity band,
    ///   so we convert to a fraction and floor weak (Observe) signals that
    ///   would otherwise read as high confidence.
    /// - `SemanticLoopSignal`, `ToolLoopSignal`, and `ContextRotSignal` all
    ///   carry a direct risk/score field, which we clamp to `[0, 1]`.
    #[must_use]
    pub fn risk(&self) -> f32 {
        match self {
            // Confidence is a bounded 0..=100 score; severity encodes whether
            // the detector thinks this is actionable.
            Self::HashLoop(signal) => {
                let base = f32::from(signal.confidence) / 100.0;
                match signal.severity {
                    LoopSeverity::Observe => base.min(0.39),
                    LoopSeverity::Suspect => base.min(0.84),
                    LoopSeverity::AbortCandidate => base,
                }
            }
            Self::SemanticLoop(signal) => signal.risk.clamp(0.0, 1.0),
            Self::ContextRot(signal) => signal.context_rot_score.clamp(0.0, 1.0),
            Self::ToolLoop(signal) => {
                #[expect(clippy::cast_possible_truncation, reason = "risk is clamped to 0..1")]
                {
                    signal.risk.clamp(0.0, 1.0) as f32
                }
            }
        }
    }
}

/// Folded risk output returned by [`RiskCombiner::combine`].
#[derive(Clone, Debug)]
pub struct CombinedRisk {
    /// Final risk score in `[0.0, 1.0]` after applying synergy bonuses.
    pub overall_risk: f32,
    /// Severity band derived from `overall_risk`.
    pub severity: LoopSeverity,
    /// Distinct detector kinds that contributed at least one signal, in
    /// canonical order.
    pub contributing_detectors: Vec<DetectorKind>,
    /// Total synergy bonus applied (sum of pairwise + extra-detector terms).
    pub synergy_bonus: f32,
    /// All signals that produced this combined risk (cloned from the combiner).
    pub signals: Vec<DetectorSignal>,
}

/// Collects detector signals for one evaluation window and folds them into a
/// unified [`CombinedRisk`].
///
/// A combiner is intended to be short-lived: create one per request/attempt,
/// feed it signals as detectors fire, call [`combine`](Self::combine) once,
/// then [`clear`](Self::clear) or drop it.
pub struct RiskCombiner {
    signals: Vec<DetectorSignal>,
    max_signals: usize,
}

impl RiskCombiner {
    /// Create a new combiner that will refuse more than `max_signals` entries.
    ///
    /// `max_signals` is a defensive cap so a runaway detector cannot grow the
    /// combiner unboundedly; passing `0` yields a combiner that always reports
    /// zero risk.
    #[must_use]
    pub fn new(max_signals: usize) -> Self {
        Self {
            signals: Vec::new(),
            max_signals,
        }
    }

    /// Append a detector signal.
    ///
    /// Silently drops the signal if the combiner is already at capacity, so a
    /// misbehaving caller cannot exhaust memory.
    pub fn add_signal(&mut self, signal: DetectorSignal) {
        if self.signals.len() < self.max_signals {
            self.signals.push(signal);
        }
    }

    /// Number of signals currently held.
    #[must_use]
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Drop all collected signals.
    pub fn clear(&mut self) {
        self.signals.clear();
    }

    /// Fold all collected signals into a single [`CombinedRisk`].
    #[must_use]
    pub fn combine(&self) -> CombinedRisk {
        if self.signals.is_empty() {
            return CombinedRisk {
                overall_risk: 0.0,
                severity: LoopSeverity::Observe,
                contributing_detectors: Vec::new(),
                synergy_bonus: 0.0,
                signals: Vec::new(),
            };
        }

        // Base risk = max individual risk.
        let base_risk = self
            .signals
            .iter()
            .map(DetectorSignal::risk)
            .fold(0.0_f32, f32::max);

        // Distinct contributing detector kinds, in canonical enum order.
        let mut kinds = [
            (DetectorKind::HashLoop, false),
            (DetectorKind::SemanticLoop, false),
            (DetectorKind::ContextRot, false),
            (DetectorKind::ToolLoop, false),
        ];
        for signal in &self.signals {
            let idx = match signal.kind() {
                DetectorKind::HashLoop => 0,
                DetectorKind::SemanticLoop => 1,
                DetectorKind::ContextRot => 2,
                DetectorKind::ToolLoop => 3,
            };
            kinds[idx].1 = true;
        }
        let contributing_detectors: Vec<DetectorKind> = kinds
            .into_iter()
            .filter_map(|(kind, present)| present.then_some(kind))
            .collect();

        let synergy_bonus = synergy_bonus_for(&contributing_detectors);
        let overall_risk = (base_risk + synergy_bonus).clamp(0.0, 1.0);
        let severity = severity_for(overall_risk);

        CombinedRisk {
            overall_risk,
            severity,
            contributing_detectors,
            synergy_bonus,
            signals: self.signals.clone(),
        }
    }
}

/// Compute the total synergy bonus for a set of contributing detector kinds.
///
/// `kinds` must be de-duplicated and is assumed small (≤ 4); we scan all pairs.
fn synergy_bonus_for(kinds: &[DetectorKind]) -> f32 {
    let mut bonus = 0.0_f32;
    for i in 0..kinds.len() {
        for j in (i + 1)..kinds.len() {
            bonus += pairwise_synergy(kinds[i], kinds[j]);
        }
    }
    if kinds.len() > 2 {
        // `kinds.len()` is at most 4 here, so the narrowing cast is exact.
        #[expect(clippy::cast_precision_loss, reason = "length is bounded <= 4")]
        {
            bonus += (kinds.len() - 2) as f32 * SYNERGY_EXTRA_PER_DETECTOR;
        }
    }
    bonus
}

/// Pairwise bonus for two *distinct* detector kinds. Returns `0.0` for
/// same-kind pairs (the caller de-duplicates, but defend against it anyway).
fn pairwise_synergy(a: DetectorKind, b: DetectorKind) -> f32 {
    let pair = |x: DetectorKind, y: DetectorKind| -> f32 {
        match (x, y) {
            (DetectorKind::HashLoop, DetectorKind::SemanticLoop)
            | (DetectorKind::SemanticLoop, DetectorKind::HashLoop) => SYNERGY_HASH_SEMANTIC,
            (DetectorKind::SemanticLoop, DetectorKind::ContextRot)
            | (DetectorKind::ContextRot, DetectorKind::SemanticLoop) => SYNERGY_SEMANTIC_CONTEXT,
            (DetectorKind::HashLoop, DetectorKind::ToolLoop)
            | (DetectorKind::ToolLoop, DetectorKind::HashLoop) => SYNERGY_HASH_TOOL,
            (DetectorKind::ToolLoop, DetectorKind::ContextRot)
            | (DetectorKind::ContextRot, DetectorKind::ToolLoop) => SYNERGY_TOOL_CONTEXT,
            // Same-kind or unlisted pairs do not earn synergy.
            _ => 0.0,
        }
    };
    if a == b { 0.0 } else { pair(a, b) }
}

/// Map a combined risk score to a severity band.
///
/// ```text
/// >= 0.85 -> AbortCandidate
/// >= 0.65 -> Suspect   (the task spec's "Warn")
/// >= 0.40 -> Observe   (the task spec's "Mild"/"Info")
/// below    -> Observe
/// ```
///
/// `LoopSeverity` only has three variants, so the spec's "Mild" and
/// "None/Info" bands both collapse to `Observe` and "Warn" maps to `Suspect`.
fn severity_for(combined_risk: f32) -> LoopSeverity {
    if combined_risk >= 0.85 {
        LoopSeverity::AbortCandidate
    } else if combined_risk >= 0.65 {
        LoopSeverity::Suspect
    } else {
        // Both the 0.40 "Mild" band and the sub-0.40 "Info" band map onto the
        // lowest available severity variant.
        LoopSeverity::Observe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_rot::ContextRotSignal;
    use crate::embedding::{EmbeddingChannel, SemanticLoopSignal};
    use crate::loop_detector::{
        BoundedFeatureSummary, DetectorEventKind, LoopReasonCode, LoopSeverity, StreamChannel,
        ToolLoopSignal,
    };

    /// Build a minimal `LoopSignal` (hash-based) with the given confidence and
    /// severity.
    fn hash_signal(confidence: u8, severity: LoopSeverity) -> LoopSignal {
        LoopSignal {
            channel: StreamChannel::Content,
            event_kind: DetectorEventKind::Check,
            severity,
            confidence,
            reason_code: LoopReasonCode::RepeatedLine,
            feature_summary: BoundedFeatureSummary::default(),
        }
    }

    /// Build a `SemanticLoopSignal` with the given risk.
    fn semantic_signal(risk: f32) -> SemanticLoopSignal {
        SemanticLoopSignal {
            channel: EmbeddingChannel::Content,
            risk,
            max_similarity: 0.95,
            cluster_density: 0.9,
            novelty_median: 0.05,
        }
    }

    /// Build a `ContextRotSignal` with the given context-rot score.
    fn context_signal(score: f32) -> ContextRotSignal {
        ContextRotSignal {
            channel: EmbeddingChannel::Content,
            echo_similarity: 0.95,
            retained_cost: 1200.0,
            context_rot_score: score,
            repeated_count: 3,
        }
    }

    /// Build a `ToolLoopSignal` with the given risk.
    fn tool_signal(risk: f64) -> ToolLoopSignal {
        ToolLoopSignal {
            reason_code: LoopReasonCode::ToolFingerprintRepeat,
            fingerprint_hash: 0xdead_beef,
            repeat_count: 3,
            risk,
        }
    }

    /// Assert two f32 values are equal within a small tolerance (avoids
    /// `clippy::float_cmp` on `assert_eq!`).
    fn assert_approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn empty_combiner_has_zero_risk() {
        let combiner = RiskCombiner::new(8);
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.0);
        assert_eq!(risk.severity, LoopSeverity::Observe);
        assert!(risk.contributing_detectors.is_empty());
        assert_approx(risk.synergy_bonus, 0.0);
        assert!(risk.signals.is_empty());
        assert_eq!(combiner.signal_count(), 0);
    }

    #[test]
    fn single_signal_has_no_synergy() {
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.70)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.70);
        assert_approx(risk.synergy_bonus, 0.0);
        assert_eq!(risk.severity, LoopSeverity::Suspect);
        assert_eq!(
            risk.contributing_detectors,
            vec![DetectorKind::SemanticLoop]
        );
        assert_eq!(risk.signals.len(), 1);
    }

    #[test]
    fn two_agreeing_detectors_apply_synergy() {
        // HashLoop (confidence 70, Suspect -> 0.70) + SemanticLoop 0.60
        // base = 0.70, synergy = 0.15 -> 0.85 (AbortCandidate).
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::HashLoop(hash_signal(
            70,
            LoopSeverity::Suspect,
        )));
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.60)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.85);
        assert_approx(risk.synergy_bonus, 0.15);
        assert_eq!(risk.severity, LoopSeverity::AbortCandidate);
        assert_eq!(
            risk.contributing_detectors,
            vec![DetectorKind::HashLoop, DetectorKind::SemanticLoop]
        );
    }

    #[test]
    fn three_detectors_earn_extra_bonus() {
        // HashLoop + SemanticLoop + ToolLoop
        // base = max(0.50, 0.50, 0.50) = 0.50
        // pairwise: Hash+Semantic 0.15 + Hash+Tool 0.10 = 0.25
        // extra (3 detectors -> 1 beyond 2): 0.05
        // total synergy = 0.30, overall = 0.80.
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::HashLoop(hash_signal(
            50,
            LoopSeverity::Suspect,
        )));
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.50)));
        combiner.add_signal(DetectorSignal::ToolLoop(tool_signal(0.50)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.80);
        assert_approx(risk.synergy_bonus, 0.30);
        assert_eq!(risk.severity, LoopSeverity::Suspect);
        assert_eq!(
            risk.contributing_detectors,
            vec![
                DetectorKind::HashLoop,
                DetectorKind::SemanticLoop,
                DetectorKind::ToolLoop,
            ]
        );
    }

    #[test]
    fn risk_is_capped_at_one() {
        // Four high-risk detectors should blow past 1.0 and get clamped.
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::HashLoop(hash_signal(
            95,
            LoopSeverity::AbortCandidate,
        )));
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.95)));
        combiner.add_signal(DetectorSignal::ContextRot(context_signal(0.95)));
        combiner.add_signal(DetectorSignal::ToolLoop(tool_signal(0.95)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 1.0);
        assert_eq!(risk.severity, LoopSeverity::AbortCandidate);
    }

    #[test]
    fn severity_thresholds_mapped_correctly() {
        // Single semantic signal so synergy_bonus == 0 and overall == risk.
        let assert_severity = |risk_score: f32, expected: LoopSeverity| {
            let mut combiner = RiskCombiner::new(8);
            combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(risk_score)));
            let result = combiner.combine();
            assert_eq!(
                result.severity, expected,
                "risk {risk_score} should map to {expected:?}"
            );
        };

        // Below 0.40 -> Observe (Mild/Info band).
        assert_severity(0.10, LoopSeverity::Observe);
        assert_severity(0.39, LoopSeverity::Observe);
        // 0.40..0.65 -> Observe (Mild band; only 3 variants exist).
        assert_severity(0.40, LoopSeverity::Observe);
        assert_severity(0.64, LoopSeverity::Observe);
        // 0.65..0.85 -> Suspect (Warn band).
        assert_severity(0.65, LoopSeverity::Suspect);
        assert_severity(0.84, LoopSeverity::Suspect);
        // >= 0.85 -> AbortCandidate.
        assert_severity(0.85, LoopSeverity::AbortCandidate);
        assert_severity(1.00, LoopSeverity::AbortCandidate);
    }

    #[test]
    fn clear_resets_state() {
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.90)));
        combiner.add_signal(DetectorSignal::ToolLoop(tool_signal(0.90)));
        assert_eq!(combiner.signal_count(), 2);

        combiner.clear();
        assert_eq!(combiner.signal_count(), 0);

        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.0);
        assert!(risk.signals.is_empty());
    }

    #[test]
    fn same_kind_does_not_synergize_with_itself() {
        // Two semantic signals from the same detector should not produce
        // synergy; overall == max(risks).
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.50)));
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.70)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.70);
        assert_approx(risk.synergy_bonus, 0.0);
        assert_eq!(
            risk.contributing_detectors,
            vec![DetectorKind::SemanticLoop]
        );
    }

    #[test]
    fn max_signals_caps_collection() {
        let mut combiner = RiskCombiner::new(2);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.50)));
        combiner.add_signal(DetectorSignal::ToolLoop(tool_signal(0.60)));
        // Third signal should be dropped.
        combiner.add_signal(DetectorSignal::ContextRot(context_signal(0.99)));
        assert_eq!(combiner.signal_count(), 2);
        let risk = combiner.combine();
        // Only the first two signals count; no context-rot present.
        assert!(
            risk.contributing_detectors
                .iter()
                .all(|k| *k != DetectorKind::ContextRot)
        );
        assert_approx(risk.overall_risk, 0.60);
    }

    #[test]
    fn zero_capacity_combiner_ignores_signals() {
        let mut combiner = RiskCombiner::new(0);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.99)));
        assert_eq!(combiner.signal_count(), 0);
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.0);
    }

    #[test]
    fn hash_loop_observe_severity_is_floored() {
        // A high-confidence Observe signal should not read as actionable.
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::HashLoop(hash_signal(
            90,
            LoopSeverity::Observe,
        )));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.39);
        assert_eq!(risk.severity, LoopSeverity::Observe);
    }

    #[test]
    fn semantic_context_pair_synergy() {
        // Semantic 0.60 + ContextRot 0.60 -> base 0.60 + 0.12 = 0.72 (Suspect).
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::SemanticLoop(semantic_signal(0.60)));
        combiner.add_signal(DetectorSignal::ContextRot(context_signal(0.60)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.72);
        assert_approx(risk.synergy_bonus, 0.12);
        assert_eq!(risk.severity, LoopSeverity::Suspect);
    }

    #[test]
    fn tool_context_pair_synergy() {
        // Tool 0.55 + ContextRot 0.55 -> base 0.55 + 0.10 = 0.65 (Suspect boundary).
        let mut combiner = RiskCombiner::new(8);
        combiner.add_signal(DetectorSignal::ToolLoop(tool_signal(0.55)));
        combiner.add_signal(DetectorSignal::ContextRot(context_signal(0.55)));
        let risk = combiner.combine();
        assert_approx(risk.overall_risk, 0.65);
        assert_approx(risk.synergy_bonus, 0.10);
        assert_eq!(risk.severity, LoopSeverity::Suspect);
    }

    #[test]
    fn risk_signal_kind_accessor() {
        assert_eq!(
            DetectorSignal::HashLoop(hash_signal(1, LoopSeverity::Observe)).kind(),
            DetectorKind::HashLoop
        );
        assert_eq!(
            DetectorSignal::SemanticLoop(semantic_signal(0.1)).kind(),
            DetectorKind::SemanticLoop
        );
        assert_eq!(
            DetectorSignal::ContextRot(context_signal(0.1)).kind(),
            DetectorKind::ContextRot
        );
        assert_eq!(
            DetectorSignal::ToolLoop(tool_signal(0.1)).kind(),
            DetectorKind::ToolLoop
        );
    }
}
