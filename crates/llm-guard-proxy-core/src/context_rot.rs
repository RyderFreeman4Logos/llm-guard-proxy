//! Context-rot risk model (issue #108).
//!
//! The [`SemanticLoopScorer`](crate::SemanticLoopScorer) answers "is this text
//! repetitive right now?" The [`ContextRotScorer`] answers a different, forward
//!-looking question: *how much will this repeated content inflate and pollute
//! the retained agent context that gets fed into the next prompt?*
//!
//! Each observed window (assistant content, reasoning, tool args, tool output)
//! is compared against a bounded history of prior *retained* context chunks.
//! When a new chunk echoes prior context strongly (`echo_sim` ≥ threshold) for
//! several consecutive windows, the scorer emits a [`ContextRotSignal`] whose
//! `context_rot_score` weighs both the recurrence depth and the estimated
//! number of tokens that will be persisted downstream (`retained_cost`).

use crate::embedding::{EmbeddingChannel, EmbeddingVector};

/// Default cosine similarity above which a chunk is considered an echo of prior
/// retained context.
pub const DEFAULT_ECHO_SIMILARITY_THRESHOLD: f32 = 0.90;

/// Default number of consecutive echoing windows required before a
/// [`ContextRotSignal`] is produced.
pub const DEFAULT_ECHO_REPEAT_COUNT: usize = 3;

/// Default maximum number of prior context chunks retained for comparison.
pub const DEFAULT_MAX_CONTEXT_CHUNKS: usize = 64;

/// Default retention weight for visible assistant content.
pub const DEFAULT_CONTENT_WEIGHT: f32 = 1.0;

/// Default retention weight for hidden reasoning. Reasoning is typically
/// transient (stripped before the next turn), so it is weighted lower.
pub const DEFAULT_REASONING_WEIGHT: f32 = 0.4;

/// Default retention weight for tool-call arguments. Tool args are persisted
/// into the transcript and re-sent on subsequent turns.
pub const DEFAULT_TOOL_ARGS_WEIGHT: f32 = 1.2;

/// Default retention weight for echoed tool *output*. Tool output is the most
/// damaging kind of repetition because it tends to be large and verbatim.
pub const DEFAULT_TOOL_OUTPUT_ECHO_WEIGHT: f32 = 1.5;

/// Configuration for a [`ContextRotScorer`].
#[derive(Clone, Debug)]
pub struct ContextRotConfig {
    /// Cosine similarity at or above which a new chunk is considered an echo
    /// of a prior retained context chunk.
    pub echo_similarity_threshold: f32,
    /// Number of consecutive echoing windows (on the same channel) required
    /// before a [`ContextRotSignal`] is emitted.
    pub echo_repeat_count: usize,
    /// Maximum number of prior context chunks retained for comparison. Once
    /// exceeded, the oldest chunk is evicted.
    pub max_context_chunks: usize,
    /// Channel retention weight applied to [`EmbeddingChannel::Content`] chunks.
    pub content_weight: f32,
    /// Channel retention weight applied to [`EmbeddingChannel::Reasoning`] chunks.
    pub reasoning_weight: f32,
    /// Channel retention weight applied to [`EmbeddingChannel::ToolArgs`] chunks.
    pub tool_args_weight: f32,
    /// Extra multiplier applied to the retained-cost of chunks whose channel
    /// is [`EmbeddingChannel::ToolArgs`], reflecting that echoed tool output is
    /// especially polluting. (Applied as the tool-args channel weight in the
    /// score formula.)
    pub tool_output_echo_weight: f32,
}

impl Default for ContextRotConfig {
    fn default() -> Self {
        Self {
            echo_similarity_threshold: DEFAULT_ECHO_SIMILARITY_THRESHOLD,
            echo_repeat_count: DEFAULT_ECHO_REPEAT_COUNT,
            max_context_chunks: DEFAULT_MAX_CONTEXT_CHUNKS,
            content_weight: DEFAULT_CONTENT_WEIGHT,
            reasoning_weight: DEFAULT_REASONING_WEIGHT,
            tool_args_weight: DEFAULT_TOOL_ARGS_WEIGHT,
            tool_output_echo_weight: DEFAULT_TOOL_OUTPUT_ECHO_WEIGHT,
        }
    }
}

impl ContextRotConfig {
    /// Creates a config populated with the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the per-channel retention weight. For `ToolArgs` this returns the
    /// heavier [`Self::tool_output_echo_weight`] to reflect the polluting cost
    /// of echoed tool output.
    #[must_use]
    pub fn channel_retention_weight(&self, channel: EmbeddingChannel) -> f32 {
        match channel {
            EmbeddingChannel::Content => self.content_weight,
            EmbeddingChannel::Reasoning => self.reasoning_weight,
            EmbeddingChannel::ToolArgs => self.tool_output_echo_weight.max(self.tool_args_weight),
        }
    }

    /// Returns the base per-channel weight used to estimate `retained_cost`
    /// (before the tool-output echo boost). This is the raw channel weight.
    #[must_use]
    pub fn channel_base_weight(&self, channel: EmbeddingChannel) -> f32 {
        match channel {
            EmbeddingChannel::Content => self.content_weight,
            EmbeddingChannel::Reasoning => self.reasoning_weight,
            EmbeddingChannel::ToolArgs => self.tool_args_weight,
        }
    }
}

/// A retained context chunk fingerprint: the channel it came from, an estimate
/// of how many tokens it will contribute to downstream context, and its
/// (truncated) embedding components for cosine comparison.
#[derive(Clone, Debug)]
pub struct ContextChunk {
    /// Which stream channel this chunk originated from.
    pub channel: EmbeddingChannel,
    /// Estimated number of tokens this chunk will persist into retained context.
    pub token_estimate: usize,
    /// Embedding vector components (may be truncated via
    /// [`EmbeddingVector::truncated`] to bound storage).
    pub components: Vec<f32>,
}

impl ContextChunk {
    /// Convenience constructor.
    #[must_use]
    pub fn new(channel: EmbeddingChannel, token_estimate: usize, components: Vec<f32>) -> Self {
        Self {
            channel,
            token_estimate,
            components,
        }
    }

    /// Builds a [`ContextChunk`] from an [`EmbeddingVector`], carrying over the
    /// components and tagging it with the given channel and token estimate.
    #[must_use]
    pub fn from_vector(
        channel: EmbeddingChannel,
        token_estimate: usize,
        vector: &EmbeddingVector,
    ) -> Self {
        Self {
            channel,
            token_estimate,
            components: vector.components.clone(),
        }
    }

    /// Returns an [`EmbeddingVector`] view of this chunk's components for
    /// cosine-similarity comparison. The `window_seq` is synthetic (always 0).
    #[must_use]
    fn as_vector(&self) -> EmbeddingVector {
        EmbeddingVector {
            window_seq: 0,
            components: self.components.clone(),
        }
    }
}

/// Signal emitted by [`ContextRotScorer`] when a new retained context chunk
/// strongly echoes prior retained context for several consecutive windows.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextRotSignal {
    /// Channel that triggered the signal.
    pub channel: EmbeddingChannel,
    /// Maximum cosine similarity between the triggering chunk and any prior
    /// retained context chunk.
    pub echo_similarity: f32,
    /// Estimated number of tokens persisted downstream by the triggering chunk,
    /// weighted by the channel base weight.
    pub retained_cost: f32,
    /// Aggregated context-rot risk score (unclamped — callers may clamp).
    pub context_rot_score: f32,
    /// Number of consecutive echoing windows observed on this channel at the
    /// time the signal was emitted (≥ [`ContextRotConfig::echo_repeat_count`]).
    pub repeated_count: usize,
}

/// Per-channel consecutive-echo counter and bounded prior-context store.
#[derive(Default)]
struct ChannelState {
    /// Bounded history of prior retained context chunks for this channel.
    chunks: Vec<ContextChunk>,
    /// Current run of consecutive windows whose echo similarity met the
    /// threshold. Reset whenever a novel (non-echoing) chunk arrives.
    consecutive_echoes: usize,
}

/// Scoring engine that estimates how much repeated retained content / tool
/// calls / tool output will inflate and pollute downstream agent context.
///
/// Unlike [`SemanticLoopScorer`](crate::SemanticLoopScorer) (which detects
/// *current* repetition), `ContextRotScorer` models the *forward* cost: how
/// many tokens of low-novelty content are about to be persisted into the next
/// prompt, weighted by how polluting each channel is.
pub struct ContextRotScorer {
    config: ContextRotConfig,
    reasoning: ChannelState,
    content: ChannelState,
    tool_args: ChannelState,
}

impl ContextRotScorer {
    /// Creates a new scorer from the given config.
    #[must_use]
    pub fn new(config: ContextRotConfig) -> Self {
        Self {
            config,
            reasoning: ChannelState::default(),
            content: ChannelState::default(),
            tool_args: ChannelState::default(),
        }
    }

    /// Returns the configured maximum number of retained context chunks.
    #[must_use]
    pub fn max_context_chunks(&self) -> usize {
        self.config.max_context_chunks
    }

    /// Returns the number of prior context chunks currently retained for a
    /// channel.
    #[must_use]
    pub fn chunk_count(&self, channel: EmbeddingChannel) -> usize {
        self.state_for(channel).chunks.len()
    }

    /// Returns the current consecutive-echo count for a channel.
    #[must_use]
    pub fn consecutive_echoes(&self, channel: EmbeddingChannel) -> usize {
        self.state_for(channel).consecutive_echoes
    }

    /// Seeds the scorer with existing/prior context without producing a signal.
    ///
    /// Use this to warm-start the scorer with context from a previous turn or a
    /// compaction summary before observing new chunks. Each added chunk is
    /// inserted into the bounded history (evicting the oldest if at capacity)
    /// but does **not** increment the consecutive-echo counter.
    pub fn add_prior_context(&mut self, chunk: ContextChunk) {
        let cap = self.config.max_context_chunks;
        let state = self.state_mut_for(chunk.channel);
        push_bounded(&mut state.chunks, chunk, cap);
    }

    /// Clears all retained context chunks and resets all consecutive-echo
    /// counters.
    pub fn clear(&mut self) {
        self.reasoning = ChannelState::default();
        self.content = ChannelState::default();
        self.tool_args = ChannelState::default();
    }

    /// Observes a new retained context chunk, updates the bounded history and
    /// the consecutive-echo counter, and returns a [`ContextRotSignal`] if the
    /// chunk echoes prior retained context strongly enough for enough
    /// consecutive windows.
    ///
    /// The returned signal (if any) reflects the state *after* inserting
    /// `chunk` into the history.
    pub fn observe_context_chunk(&mut self, chunk: ContextChunk) -> Option<ContextRotSignal> {
        let channel = chunk.channel;
        let threshold = self.config.echo_similarity_threshold;
        let cap = self.config.max_context_chunks;

        let incoming = chunk.as_vector();

        let state = self.state_mut_for(channel);
        // echo_sim = max cosine similarity to prior retained context chunks.
        let echo_sim = max_cosine(&state.chunks, &incoming);

        // Update the consecutive-echo counter.
        let is_echo = echo_sim.is_some_and(|s| s >= threshold);
        if is_echo {
            state.consecutive_echoes += 1;
        } else {
            state.consecutive_echoes = 0;
        }
        let repeated_count = state.consecutive_echoes;

        // Insert the new chunk into bounded history.
        push_bounded(&mut state.chunks, chunk, cap);

        // Only signal once we've seen enough consecutive echoing windows *and*
        // the echo similarity is actually defined.
        let echo_similarity = echo_sim?;
        if repeated_count < self.config.echo_repeat_count {
            return None;
        }

        let retained_cost = self.retained_cost(channel);
        let echo_recurrence_score = Self::echo_recurrence_score(repeated_count, echo_similarity);
        let channel_weight = self.config.channel_retention_weight(channel);

        let context_rot_score = echo_recurrence_score * retained_cost * channel_weight;

        Some(ContextRotSignal {
            channel,
            echo_similarity,
            retained_cost,
            context_rot_score,
            repeated_count,
        })
    }

    /// Estimated token cost that will be persisted downstream, weighted by the
    /// channel base weight. Sums the token estimates of all currently-retained
    /// chunks on the channel (including the most recent one).
    fn retained_cost(&self, channel: EmbeddingChannel) -> f32 {
        let base = self.config.channel_base_weight(channel);
        let tokens: usize = self
            .state_for(channel)
            .chunks
            .iter()
            .map(|c| c.token_estimate)
            .sum();
        // Token counts are small (bounded history, realistic windows); the
        // cast is safe in practice.
        #[allow(clippy::cast_precision_loss)]
        let cost = tokens as f32 * base;
        cost
    }

    /// Echo-recurrence score increases with the number of consecutive echoing
    /// windows. Defined as `echo_similarity * (1.0 + 0.25 * (count - 1))` so
    /// each additional consecutive echo deepens the recurrence penalty by 25%.
    fn echo_recurrence_score(count: usize, echo_similarity: f32) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let depth_bonus = 1.0 + 0.25 * (count.saturating_sub(1) as f32);
        echo_similarity * depth_bonus
    }

    fn state_mut_for(&mut self, channel: EmbeddingChannel) -> &mut ChannelState {
        match channel {
            EmbeddingChannel::Reasoning => &mut self.reasoning,
            EmbeddingChannel::Content => &mut self.content,
            EmbeddingChannel::ToolArgs => &mut self.tool_args,
        }
    }

    fn state_for(&self, channel: EmbeddingChannel) -> &ChannelState {
        match channel {
            EmbeddingChannel::Reasoning => &self.reasoning,
            EmbeddingChannel::Content => &self.content,
            EmbeddingChannel::ToolArgs => &self.tool_args,
        }
    }
}

/// Computes the maximum cosine similarity between `incoming` and any prior
/// retained chunk. Returns `None` if history is empty or every comparison is
/// undefined (mismatched dimensions / zero-norm).
fn max_cosine(history: &[ContextChunk], incoming: &EmbeddingVector) -> Option<f32> {
    if history.is_empty() {
        return None;
    }
    let mut best = f32::NEG_INFINITY;
    let mut defined = false;
    for prev in history {
        let prev_vec = EmbeddingVector {
            window_seq: 0,
            components: prev.components.clone(),
        };
        if let Some(sim) = incoming.cosine_similarity(&prev_vec) {
            defined = true;
            if sim > best {
                best = sim;
            }
        }
    }
    if defined && best.is_finite() {
        Some(best)
    } else {
        None
    }
}

/// Appends a chunk to the bounded store, evicting the oldest entry when at
/// capacity. Capacity is clamped to at least 1.
fn push_bounded(chunks: &mut Vec<ContextChunk>, chunk: ContextChunk, cap: usize) {
    let cap = cap.max(1);
    while chunks.len() >= cap {
        chunks.remove(0);
    }
    chunks.push(chunk);
}

impl Default for ContextRotScorer {
    fn default() -> Self {
        Self::new(ContextRotConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a chunk with a simple deterministic embedding of the given
    /// dimension, scaled by `seed` so different seeds produce different vectors.
    fn make_chunk(channel: EmbeddingChannel, tokens: usize, seed: f32, dim: usize) -> ContextChunk {
        #[allow(clippy::cast_precision_loss)]
        let components = (0..dim).map(|i| seed + i as f32 * 0.01).collect();
        ContextChunk::new(channel, tokens, components)
    }

    #[test]
    fn repeated_content_chunk_triggers_signal() {
        let mut scorer = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 3,
            ..ContextRotConfig::default()
        });

        // Seed one prior content chunk.
        scorer.add_prior_context(make_chunk(EmbeddingChannel::Content, 10, 1.0, 8));

        // Observe three near-identical chunks: each echoes the prior one.
        let signals: Vec<_> = (0..3)
            .map(|_| {
                scorer.observe_context_chunk(make_chunk(EmbeddingChannel::Content, 10, 1.0, 8))
            })
            .collect();

        // First two observations reach counts 1 and 2 (below threshold of 3).
        assert!(signals[0].is_none(), "no signal before repeat threshold");
        assert!(signals[1].is_none(), "no signal before repeat threshold");
        // Third observation reaches count 3 → signal.
        let signal = signals[2]
            .clone()
            .expect("signal after 3 consecutive echoes");
        assert_eq!(signal.channel, EmbeddingChannel::Content);
        assert!(
            signal.echo_similarity >= 0.90,
            "echo similarity should meet threshold"
        );
        assert_eq!(signal.repeated_count, 3);
        assert!(
            signal.context_rot_score > 0.0,
            "context-rot score should be positive"
        );
    }

    #[test]
    fn novel_content_does_not_trigger() {
        let mut scorer = ContextRotScorer::new(ContextRotConfig::default());

        // Seed a prior content chunk with an embedding in one direction.
        // Use one-hot vectors so novel chunks are genuinely orthogonal.
        let mut prior = vec![0.0_f32; 8];
        prior[0] = 1.0;
        scorer.add_prior_context(ContextChunk::new(EmbeddingChannel::Content, 10, prior));

        // Observe several chunks in orthogonal/different directions → novel.
        for dim in 1..5 {
            let mut components = vec![0.0_f32; 8];
            components[dim] = 1.0;
            let chunk = ContextChunk::new(EmbeddingChannel::Content, 10, components);
            let signal = scorer.observe_context_chunk(chunk);
            assert!(
                signal.is_none(),
                "novel content (one-hot dim {dim}) should not trigger a signal"
            );
        }
    }

    #[test]
    fn channel_weights_applied_correctly() {
        // Two scorers: one Content, one ToolArgs. Same token estimate and echo
        // recurrence. The ToolArgs signal should carry a higher context_rot_score
        // because tool output has the higher retention weight.
        let tokens = 50;
        let dim = 8;

        let mut content_scorer = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 2,
            ..ContextRotConfig::default()
        });
        let mut tool_scorer = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 2,
            ..ContextRotConfig::default()
        });

        // Warm-start both with identical-geometry priors on their channels.
        content_scorer.add_prior_context(make_chunk(EmbeddingChannel::Content, tokens, 1.0, dim));
        tool_scorer.add_prior_context(make_chunk(EmbeddingChannel::ToolArgs, tokens, 1.0, dim));

        // Two echoing observations each.
        let content_signal = (0..2)
            .find_map(|_| {
                content_scorer.observe_context_chunk(make_chunk(
                    EmbeddingChannel::Content,
                    tokens,
                    1.0,
                    dim,
                ))
            })
            .expect("content signal");
        let tool_signal = (0..2)
            .find_map(|_| {
                tool_scorer.observe_context_chunk(make_chunk(
                    EmbeddingChannel::ToolArgs,
                    tokens,
                    1.0,
                    dim,
                ))
            })
            .expect("tool signal");

        assert!(
            tool_signal.context_rot_score > content_signal.context_rot_score,
            "tool-args retention weight should yield a higher score \
             (tool={}, content={})",
            tool_signal.context_rot_score,
            content_signal.context_rot_score
        );
    }

    #[test]
    fn context_chunk_history_bounded() {
        let cap = 4;
        let mut scorer = ContextRotScorer::new(ContextRotConfig {
            max_context_chunks: cap,
            ..ContextRotConfig::default()
        });

        // Seed and then observe many chunks; history must never exceed cap.
        scorer.add_prior_context(make_chunk(EmbeddingChannel::Content, 5, 0.0, 4));
        for i in 1..=20 {
            // Use distinct seeds so we don't accidentally trip the echo gate.
            #[allow(clippy::cast_precision_loss)]
            let chunk = make_chunk(EmbeddingChannel::Content, 5, i as f32 * 10.0, 4);
            let _ = scorer.observe_context_chunk(chunk);
            assert!(
                scorer.chunk_count(EmbeddingChannel::Content) <= cap,
                "history exceeded cap after observation {i}: {}",
                scorer.chunk_count(EmbeddingChannel::Content)
            );
        }
        assert_eq!(
            scorer.chunk_count(EmbeddingChannel::Content),
            cap,
            "history should be exactly at capacity"
        );
    }

    #[test]
    fn token_estimate_affects_retained_cost() {
        // Two scorers with different token estimates but identical echo geometry.
        // The higher-token one should report a larger retained_cost.
        let dim = 8;

        let mut small = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 2,
            ..ContextRotConfig::default()
        });
        let mut large = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 2,
            ..ContextRotConfig::default()
        });

        small.add_prior_context(make_chunk(EmbeddingChannel::Content, 10, 1.0, dim));
        large.add_prior_context(make_chunk(EmbeddingChannel::Content, 1000, 1.0, dim));

        let small_signal = small
            .observe_context_chunk(make_chunk(EmbeddingChannel::Content, 10, 1.0, dim))
            .or_else(|| {
                small.observe_context_chunk(make_chunk(EmbeddingChannel::Content, 10, 1.0, dim))
            })
            .expect("small signal");
        let large_signal = large
            .observe_context_chunk(make_chunk(EmbeddingChannel::Content, 1000, 1.0, dim))
            .or_else(|| {
                large.observe_context_chunk(make_chunk(EmbeddingChannel::Content, 1000, 1.0, dim))
            })
            .expect("large signal");

        assert!(
            large_signal.retained_cost > small_signal.retained_cost,
            "larger token estimate should yield higher retained_cost \
             (large={}, small={})",
            large_signal.retained_cost,
            small_signal.retained_cost
        );
    }

    #[test]
    fn clear_resets_state() {
        let mut scorer = ContextRotScorer::new(ContextRotConfig {
            echo_repeat_count: 2,
            ..ContextRotConfig::default()
        });
        scorer.add_prior_context(make_chunk(EmbeddingChannel::Content, 10, 1.0, 8));
        assert_eq!(scorer.chunk_count(EmbeddingChannel::Content), 1);

        scorer.clear();
        assert_eq!(scorer.chunk_count(EmbeddingChannel::Content), 0);
        assert_eq!(scorer.consecutive_echoes(EmbeddingChannel::Content), 0);
    }

    #[test]
    fn defaults_match_documented_values() {
        let cfg = ContextRotConfig::default();
        assert!((cfg.echo_similarity_threshold - 0.90).abs() < f32::EPSILON);
        assert_eq!(cfg.echo_repeat_count, 3);
        assert_eq!(cfg.max_context_chunks, 64);
        assert!((cfg.content_weight - 1.0).abs() < f32::EPSILON);
        assert!((cfg.reasoning_weight - 0.4).abs() < f32::EPSILON);
        assert!((cfg.tool_args_weight - 1.2).abs() < f32::EPSILON);
        assert!((cfg.tool_output_echo_weight - 1.5).abs() < f32::EPSILON);
    }
}
