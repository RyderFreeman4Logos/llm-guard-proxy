//! Embedding backend for semantic loop detection.
//!
//! Defines the [`EmbeddingBackend`] trait and supporting types used by the
//! loop guard to compute embedding-based semantic similarity between streaming
//! windows. The trait is intentionally async-capable but lives in the core
//! crate (which has no async runtime) by using boxed futures.
//!
//! The default implementation is [`DisabledEmbeddingBackend`], which returns
//! empty results without error — semantic loop detection then falls back to
//! the existing deterministic/hash-based detector.

use std::future::Future;
use std::pin::Pin;

use thiserror::Error;

/// Error returned by an embedding backend.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// The backend is disabled and cannot produce vectors.
    #[error("embedding backend is disabled")]
    Disabled,
    /// Network or HTTP error while calling the embedding endpoint.
    #[error("embedding request failed: {0}")]
    Request(String),
    /// The endpoint returned a non-success status code.
    #[error("embedding endpoint returned status {status}: {body}")]
    Status { status: u16, body: String },
    /// The response body could not be parsed.
    #[error("failed to parse embedding response: {0}")]
    Parse(String),
    /// The batch was empty.
    #[error("embedding batch must not be empty")]
    EmptyBatch,
    /// The requested vector dimension is invalid.
    #[error("invalid vector dimension: {0}")]
    InvalidDimension(usize),
}

/// A single embedding request input.
///
/// Each input corresponds to one streaming window that needs to be embedded.
#[derive(Clone, Debug)]
pub struct EmbeddingInput {
    /// Opaque request identifier for correlation (not sent to the endpoint).
    pub request_id: String,
    /// Opaque attempt identifier for correlation (not sent to the endpoint).
    pub attempt_id: String,
    /// Which stream channel this window came from.
    pub channel: EmbeddingChannel,
    /// Monotonic window sequence number within the request/attempt/channel.
    pub window_seq: u64,
    /// Hash of the normalized text (for deduplication; not sent to endpoint).
    pub text_hash: u64,
    /// The normalized text to embed.
    pub text: String,
}

/// Channel that an embedding window belongs to, mirroring the loop detector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EmbeddingChannel {
    /// Hidden reasoning / thinking.
    Reasoning,
    /// Visible assistant content.
    Content,
    /// Tool-call arguments.
    ToolArgs,
}

/// A normalized embedding vector stored as `f32` components.
#[derive(Clone, Debug)]
pub struct EmbeddingVector {
    /// The window sequence number this vector corresponds to.
    pub window_seq: u64,
    /// The embedding components.
    pub components: Vec<f32>,
}

impl EmbeddingVector {
    /// Computes the cosine similarity between two vectors.
    ///
    /// Returns `None` if the vectors have different lengths or are all-zero.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> Option<f32> {
        if self.components.len() != other.components.len() || self.components.is_empty() {
            return None;
        }
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for (a, b) in self.components.iter().zip(other.components.iter()) {
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            None
        } else {
            Some(dot / denom)
        }
    }

    /// Truncates the vector to the first `dim` components (MRL-style).
    ///
    /// No-op if `dim` is `0` or >= current length.
    #[must_use]
    pub fn truncated(mut self, dim: usize) -> Self {
        if dim > 0 && dim < self.components.len() {
            self.components.truncate(dim);
        }
        self
    }
}

// ============================================================================
// Semantic self-loop scoring (issue #107)
// ============================================================================

/// Default cosine similarity threshold for [`EmbeddingChannel::Reasoning`].
///
/// Reasoning text tends to paraphrase itself within a tighter band, so the
/// bar for "same cluster" is slightly lower than for content/tool args.
pub const REASONING_SIMILARITY_THRESHOLD: f32 = 0.88;
/// Default cosine similarity threshold for [`EmbeddingChannel::Content`].
pub const CONTENT_SIMILARITY_THRESHOLD: f32 = 0.90;
/// Default cosine similarity threshold for [`EmbeddingChannel::ToolArgs`].
pub const TOOL_ARGS_SIMILARITY_THRESHOLD: f32 = 0.92;

/// Minimum number of same-channel observations before the scorer will emit a
/// [`SemanticLoopSignal`]. Below this we don't have enough history to trust a
/// cluster-density reading.
pub const MIN_OBSERVATIONS_FOR_SIGNAL: usize = 2;

/// Signal emitted by [`SemanticLoopScorer`] when recent embedding windows on a
/// channel form a tight, low-novelty cluster — i.e. the model is semantically
/// stuck repeating itself.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticLoopSignal {
    /// Channel that triggered the signal.
    pub channel: EmbeddingChannel,
    /// Aggregated risk score in `[0.0, 1.0]`. Higher means more likely a loop.
    ///
    /// Computed as `cluster_density * (1.0 - novelty_median)`, then clamped.
    pub risk: f32,
    /// Highest cosine similarity observed between the new vector and the
    /// channel's recent history (`max_recent_sim`).
    pub max_similarity: f32,
    /// Fraction of the last `K` same-channel windows whose cosine similarity
    /// to the new vector met or exceeded the similarity threshold.
    pub cluster_density: f32,
    /// Median novelty (`1.0 - cosine`) over the last `M` windows.
    pub novelty_median: f32,
}

/// Configuration for a [`SemanticLoopScorer`].
///
/// `similarity_threshold` is a single default; callers that want per-channel
/// thresholds can mutate the returned config or construct the scorer and then
/// adjust [`SemanticLoopScorer::set_channel_threshold`].
#[derive(Clone, Debug)]
pub struct SemanticLoopConfig {
    /// Cosine similarity at or above which two vectors are considered the same
    /// semantic cluster (default channel threshold; per-channel overrides
    /// apply via [`SemanticLoopScorer::set_channel_threshold`]).
    pub similarity_threshold: f32,
    /// Minimum fraction of recent windows that must cluster together before a
    /// signal is emitted (e.g. `0.55`).
    pub cluster_density_threshold: f32,
    /// Median novelty (`1.0 - cosine`) over recent windows must be at or below
    /// this for a signal (e.g. `0.08`).
    pub low_novelty_median: f32,
    /// Maximum number of recent embedding vectors retained per channel (the
    /// ring-buffer bound).
    pub history_window_count: usize,
}

impl Default for SemanticLoopConfig {
    fn default() -> Self {
        Self {
            // Content channel's default; per-channel overrides live in the
            // scorer (reasoning and tool args are adjusted up/down).
            similarity_threshold: CONTENT_SIMILARITY_THRESHOLD,
            cluster_density_threshold: 0.55,
            low_novelty_median: 0.08,
            history_window_count: 64,
        }
    }
}

impl SemanticLoopConfig {
    /// Creates a config with the documented sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Scoring engine that detects semantic self-loops per channel using cosine
/// similarity between recent embedding windows.
///
/// The scorer maintains a bounded ring buffer of recent
/// [`EmbeddingVector`]s per [`EmbeddingChannel`]. Each new vector is compared
/// against the channel's recent history to compute:
///
/// - `max_recent_sim`: the highest cosine similarity to any recent window.
/// - `novelty`: `1.0 - max_recent_sim` (how new this content is).
/// - `cluster_density`: the fraction of recent windows whose similarity meets
///   the channel's threshold.
///
/// A [`SemanticLoopSignal`] is emitted when the channel's recent windows form
/// a tight, low-novelty cluster — i.e. the model is paraphrasing itself rather
/// than producing new content. This catches recurrence that the hash-based
/// [`ChannelizedLoopDetector`](crate::ChannelizedLoopDetector) misses, because
/// paraphrases rarely share token-set hashes.
pub struct SemanticLoopScorer {
    config: SemanticLoopConfig,
    thresholds: ChannelThresholds,
    reasoning: RingBuffer,
    content: RingBuffer,
    tool_args: RingBuffer,
}

impl SemanticLoopScorer {
    /// Creates a new scorer from the given config, installing per-channel
    /// default thresholds:
    /// - Reasoning: `0.88`
    /// - Content: `0.90`
    /// - `ToolArgs`: `0.92`
    #[must_use]
    pub fn new(config: SemanticLoopConfig) -> Self {
        let mut thresholds = ChannelThresholds::default();
        thresholds.set(EmbeddingChannel::Reasoning, REASONING_SIMILARITY_THRESHOLD);
        thresholds.set(EmbeddingChannel::Content, config.similarity_threshold);
        thresholds.set(EmbeddingChannel::ToolArgs, TOOL_ARGS_SIMILARITY_THRESHOLD);
        Self {
            config,
            thresholds,
            reasoning: RingBuffer::default(),
            content: RingBuffer::default(),
            tool_args: RingBuffer::default(),
        }
    }

    /// Returns the configured history window count (ring-buffer bound).
    #[must_use]
    pub fn history_window_count(&self) -> usize {
        self.config.history_window_count
    }

    /// Overrides the similarity threshold for a specific channel.
    pub fn set_channel_threshold(&mut self, channel: EmbeddingChannel, threshold: f32) {
        self.thresholds.set(channel, threshold);
    }

    /// Returns the effective similarity threshold for a channel.
    #[must_use]
    pub fn channel_threshold(&self, channel: EmbeddingChannel) -> f32 {
        self.thresholds.get(channel)
    }

    /// Returns the stored embedding history for a channel (oldest first).
    #[must_use]
    pub fn channel_history(&self, channel: EmbeddingChannel) -> &[EmbeddingVector] {
        self.ring_for(channel).as_slice()
    }

    /// Observes a new embedding vector for a channel, updates the per-channel
    /// ring buffer, and returns a [`SemanticLoopSignal`] if the channel is
    /// judged to be in a semantic self-loop.
    ///
    /// The returned signal (if any) reflects the state *after* inserting
    /// `vector` into the history.
    pub fn observe_vector(
        &mut self,
        channel: EmbeddingChannel,
        vector: &EmbeddingVector,
    ) -> Option<SemanticLoopSignal> {
        let threshold = self.thresholds.get(channel);
        let cap = self.config.history_window_count;

        let ring = self.ring_mut_for(channel);
        // Compute cluster statistics *before* we insert the new vector, so the
        // similarity/density readings compare the new window against prior
        // history (a window is not similar to itself in this accounting).
        let (max_similarity, cluster_density, novelty_median) =
            Self::cluster_stats(ring, vector, threshold, cap);

        ring.push(vector.clone(), cap);

        // Need at least a couple of prior observations for the density/median
        // readings to be meaningful. `max_similarity == None` also covers the
        // degenerate (zero-length / all-zero) vector case, where every cosine
        // comparison is undefined.
        let sufficient_history = ring.len() >= MIN_OBSERVATIONS_FOR_SIGNAL;
        let max_sim = max_similarity?;

        if !sufficient_history {
            return None;
        }

        let density = cluster_density.unwrap_or(0.0);
        let novelty = novelty_median.unwrap_or(1.0);

        let is_loop = density >= self.config.cluster_density_threshold
            && novelty <= self.config.low_novelty_median;

        if !is_loop {
            return None;
        }

        let risk = (density * (1.0 - novelty)).clamp(0.0, 1.0);

        Some(SemanticLoopSignal {
            channel,
            risk,
            max_similarity: max_sim,
            cluster_density: density,
            novelty_median: novelty,
        })
    }

    /// Computes `max_recent_sim`, `cluster_density`, and `novelty_median` for
    /// `vector` against the most recent `cap` entries of `history`.
    ///
    /// Returns `(max_sim, density, median)` where each component is `None`
    /// when `history` is empty or every cosine comparison is undefined.
    fn cluster_stats(
        history: &RingBuffer,
        vector: &EmbeddingVector,
        threshold: f32,
        cap: usize,
    ) -> (Option<f32>, Option<f32>, Option<f32>) {
        if history.buf.is_empty() {
            return (None, None, None);
        }

        // Only compare against the last `cap` historical windows.
        let start = history.buf.len().saturating_sub(cap);
        let recent = &history.buf[start..];

        let mut max_sim = f32::NEG_INFINITY;
        let mut similarities: Vec<f32> = Vec::with_capacity(recent.len());
        let mut defined = 0usize;

        for prev in recent {
            if let Some(sim) = vector.cosine_similarity(prev) {
                defined += 1;
                max_sim = max_sim.max(sim);
                similarities.push(sim);
            }
        }

        if defined == 0 {
            return (None, None, None);
        }

        let max_sim = if max_sim.is_finite() {
            Some(max_sim)
        } else {
            None
        };

        // Cluster density: fraction of *defined* comparisons meeting threshold.
        // Counts are bounded by `recent.len()` (tiny), so the usize→f32 cast
        // cannot lose precision in practice; allow the lint locally.
        #[allow(clippy::cast_precision_loss)]
        let above = similarities.iter().filter(|&&s| s >= threshold).count() as f32;
        #[allow(clippy::cast_precision_loss)]
        let density = above / defined as f32;

        // Novelty = 1.0 - cosine; median over recent windows.
        let mut novelties: Vec<f32> = similarities.iter().map(|&s| 1.0 - s).collect();
        novelties.sort_unstable_by(f32::total_cmp);
        let median = Self::median(&novelties);

        (max_sim, Some(density), Some(median))
    }

    /// Returns the median of a non-empty, pre-sorted slice.
    fn median(sorted: &[f32]) -> f32 {
        let n = sorted.len();
        if n == 0 {
            return 1.0;
        }
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            f32::midpoint(sorted[n / 2 - 1], sorted[n / 2])
        }
    }

    fn ring_mut_for(&mut self, channel: EmbeddingChannel) -> &mut RingBuffer {
        match channel {
            EmbeddingChannel::Reasoning => &mut self.reasoning,
            EmbeddingChannel::Content => &mut self.content,
            EmbeddingChannel::ToolArgs => &mut self.tool_args,
        }
    }

    fn ring_for(&self, channel: EmbeddingChannel) -> &RingBuffer {
        match channel {
            EmbeddingChannel::Reasoning => &self.reasoning,
            EmbeddingChannel::Content => &self.content,
            EmbeddingChannel::ToolArgs => &self.tool_args,
        }
    }
}

/// Per-channel cosine similarity thresholds.
#[derive(Clone, Debug)]
struct ChannelThresholds {
    reasoning: f32,
    content: f32,
    tool_args: f32,
}

impl Default for ChannelThresholds {
    fn default() -> Self {
        Self {
            reasoning: REASONING_SIMILARITY_THRESHOLD,
            content: CONTENT_SIMILARITY_THRESHOLD,
            tool_args: TOOL_ARGS_SIMILARITY_THRESHOLD,
        }
    }
}

impl ChannelThresholds {
    const fn get(&self, channel: EmbeddingChannel) -> f32 {
        match channel {
            EmbeddingChannel::Reasoning => self.reasoning,
            EmbeddingChannel::Content => self.content,
            EmbeddingChannel::ToolArgs => self.tool_args,
        }
    }

    fn set(&mut self, channel: EmbeddingChannel, value: f32) {
        match channel {
            EmbeddingChannel::Reasoning => self.reasoning = value,
            EmbeddingChannel::Content => self.content = value,
            EmbeddingChannel::ToolArgs => self.tool_args = value,
        }
    }
}

/// Bounded per-channel ring buffer of recent embedding vectors.
#[derive(Clone, Debug, Default)]
struct RingBuffer {
    buf: Vec<EmbeddingVector>,
}

impl RingBuffer {
    /// Appends a vector, evicting the oldest entry when at capacity.
    fn push(&mut self, vector: EmbeddingVector, cap: usize) {
        let cap = cap.max(1);
        if self.buf.len() >= cap {
            self.buf.remove(0);
        }
        self.buf.push(vector);
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn as_slice(&self) -> &[EmbeddingVector] {
        &self.buf
    }
}

/// Type alias for the boxed future returned by [`EmbeddingBackend::embed_batch`].
pub type EmbeddingFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<EmbeddingVector>, EmbeddingError>> + Send + 'a>>;

/// Backend that produces embedding vectors for streaming windows.
///
/// Implementations:
/// - [`DisabledEmbeddingBackend`] — default, returns empty results.
/// - `OpenAiCompatibleEmbeddingBackend` (in the proxy crate) — calls a real
///   `/v1/embeddings` endpoint using `reqwest`.
pub trait EmbeddingBackend: Send + Sync {
    /// Embeds a batch of inputs, returning vectors in the same order.
    ///
    /// Implementations should handle batching internally (microbatching,
    /// queueing, etc.). Callers must not block the SSE hot path on this call.
    fn embed_batch(&self, inputs: Vec<EmbeddingInput>) -> EmbeddingFuture<'_>;
}

/// Default embedding backend that does nothing.
///
/// Semantic loop detection falls back to deterministic/hash-based detection
/// when this backend is active.
#[derive(Clone, Debug, Default)]
pub struct DisabledEmbeddingBackend;

impl DisabledEmbeddingBackend {
    /// Creates a new disabled backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl EmbeddingBackend for DisabledEmbeddingBackend {
    fn embed_batch(&self, _inputs: Vec<EmbeddingInput>) -> EmbeddingFuture<'_> {
        // Use std::future::ready so the core crate does not need an async runtime.
        Box::pin(std::future::ready(Ok(Vec::new())))
    }
}

/// Outcome of pushing a window onto the [`EmbeddingQueue`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingQueueResult {
    /// The window was accepted and enqueued.
    Accepted,
    /// The window was skipped (queue overflow under a non-blocking policy, or
    /// a duplicate of a window already present in the queue).
    Skipped,
    /// The queue is full and the configured policy is `block`. Callers on the
    /// SSE hot path must wait for space before retrying; callers that cannot
    /// block should treat this as a hard skip.
    QueueFull,
}

/// Synchronous bounded queue that buffers streaming windows pending embedding.
///
/// The queue itself is synchronous (`std::collections::VecDeque`) because the
/// core crate is `#![forbid(unsafe_code)]` and has no async runtime. The async
/// batch worker that drains the queue lives in the proxy crate.
///
/// Overflow behavior is governed by the [`EmbeddingQueuePolicy`][policy] passed
/// at construction:
/// - `Skip` (default): silently drop the window and return [`Skipped`](EmbeddingQueueResult::Skipped).
/// - `DeterministicOnly`: drop the window but signal the caller to fall back to
///   deterministic hash-based detection (also reported as `Skipped`).
/// - `Block`: refuse the push and return [`QueueFull`](EmbeddingQueueResult::QueueFull)
///   so the caller can retry after space frees up.
///
/// Windows with a `(request_id, attempt_id, channel, window_seq)` tuple already
/// present in the queue are treated as duplicates and skipped.
#[derive(Clone, Debug)]
pub struct EmbeddingQueue {
    queue: std::collections::VecDeque<EmbeddingInput>,
    capacity: usize,
    policy: crate::settings::EmbeddingQueuePolicy,
}

impl EmbeddingQueue {
    /// Creates a new queue with the given capacity and overflow policy.
    ///
    /// `capacity` is clamped to at least 1 so the queue is always usable.
    #[must_use]
    pub fn new(capacity: usize, policy: crate::settings::EmbeddingQueuePolicy) -> Self {
        Self {
            queue: std::collections::VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            policy,
        }
    }

    /// Returns the maximum number of windows the queue can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of windows currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if no windows are currently buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the configured overflow policy.
    #[must_use]
    pub fn policy(&self) -> crate::settings::EmbeddingQueuePolicy {
        self.policy
    }

    /// Attempts to enqueue a streaming window for embedding.
    ///
    /// Returns [`Accepted`](EmbeddingQueueResult::Accepted) if the window was
    /// enqueued, [`Skipped`](EmbeddingQueueResult::Skipped) if it was dropped
    /// (overflow under a non-blocking policy, or a duplicate), or
    /// [`QueueFull`](EmbeddingQueueResult::QueueFull) if the queue is full and
    /// the policy is `block`.
    pub fn push(
        &mut self,
        request_id: impl Into<String>,
        attempt_id: impl Into<String>,
        channel: EmbeddingChannel,
        window_seq: u64,
        text_hash: u64,
        text: impl Into<String>,
    ) -> EmbeddingQueueResult {
        let input = EmbeddingInput {
            request_id: request_id.into(),
            attempt_id: attempt_id.into(),
            channel,
            window_seq,
            text_hash,
            text: text.into(),
        };
        self.push_input(input)
    }

    /// Pushes a fully-constructed [`EmbeddingInput`].
    ///
    /// This is the low-level entry point used by [`push`](Self::push) and by
    /// callers that already hold an `EmbeddingInput`.
    pub fn push_input(&mut self, input: EmbeddingInput) -> EmbeddingQueueResult {
        // Deduplicate by (request_id, attempt_id, channel, window_seq).
        let already_present = self.queue.iter().any(|existing| {
            existing.request_id == input.request_id
                && existing.attempt_id == input.attempt_id
                && existing.channel == input.channel
                && existing.window_seq == input.window_seq
        });
        if already_present {
            return EmbeddingQueueResult::Skipped;
        }

        if self.queue.len() >= self.capacity {
            return match self.policy {
                crate::settings::EmbeddingQueuePolicy::Block => EmbeddingQueueResult::QueueFull,
                // Skip and DeterministicOnly both drop the window; the caller
                // distinguishes them via the policy field when needed.
                crate::settings::EmbeddingQueuePolicy::Skip
                | crate::settings::EmbeddingQueuePolicy::DeterministicOnly => {
                    EmbeddingQueueResult::Skipped
                }
            };
        }

        self.queue.push_back(input);
        EmbeddingQueueResult::Accepted
    }

    /// Removes and returns up to `max` windows from the front of the queue,
    /// draining them in FIFO order for batch embedding.
    ///
    /// Returns an empty vector if the queue is empty or `max` is `0`.
    pub fn drain_batch(&mut self, max: usize) -> Vec<EmbeddingInput> {
        if max == 0 {
            return Vec::new();
        }
        let take = max.min(self.queue.len());
        self.queue.drain(..take).collect()
    }

    /// Removes and returns all currently buffered windows.
    ///
    /// Convenience wrapper around [`drain_batch`](Self::drain_batch) for a full
    /// flush (e.g. on shutdown).
    pub fn drain_all(&mut self) -> Vec<EmbeddingInput> {
        self.queue.drain(..).collect()
    }

    /// Peeks at the front window without removing it.
    #[must_use]
    pub fn front(&self) -> Option<&EmbeddingInput> {
        self.queue.front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = EmbeddingVector {
            window_seq: 0,
            components: vec![1.0, 2.0, 3.0],
        };
        assert!((a.cosine_similarity(&a).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = EmbeddingVector {
            window_seq: 0,
            components: vec![1.0, 0.0],
        };
        let b = EmbeddingVector {
            window_seq: 1,
            components: vec![0.0, 1.0],
        };
        assert!((a.cosine_similarity(&b).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_different_lengths() {
        let a = EmbeddingVector {
            window_seq: 0,
            components: vec![1.0, 2.0, 3.0],
        };
        let b = EmbeddingVector {
            window_seq: 1,
            components: vec![1.0, 2.0],
        };
        assert_eq!(a.cosine_similarity(&b), None);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = EmbeddingVector {
            window_seq: 0,
            components: vec![0.0, 0.0, 0.0],
        };
        let b = EmbeddingVector {
            window_seq: 1,
            components: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(a.cosine_similarity(&b), None);
    }

    #[test]
    fn truncate_reduces_dimension() {
        let v = EmbeddingVector {
            window_seq: 0,
            components: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let truncated = v.truncated(3);
        assert_eq!(truncated.components, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn truncate_zero_is_noop() {
        let v = EmbeddingVector {
            window_seq: 0,
            components: vec![1.0, 2.0, 3.0],
        };
        let truncated = v.truncated(0);
        assert_eq!(truncated.components.len(), 3);
    }

    // The disabled backend's embed_batch returns a boxed future. We cannot
    // poll it in core (#![forbid(unsafe_code)] blocks manual waker creation).
    // The async integration test lives in the proxy crate's test suite.

    // ---- EmbeddingQueue tests ----

    use crate::settings::EmbeddingQueuePolicy;

    fn make_input(
        request_id: &str,
        attempt_id: &str,
        channel: EmbeddingChannel,
        window_seq: u64,
    ) -> EmbeddingInput {
        EmbeddingInput {
            request_id: request_id.to_owned(),
            attempt_id: attempt_id.to_owned(),
            channel,
            window_seq,
            text_hash: window_seq,
            text: format!("text-{window_seq}"),
        }
    }

    #[test]
    fn queue_push_and_len() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        let res = q.push("req-1", "att-1", EmbeddingChannel::Content, 0, 0, "hello");
        assert_eq!(res, EmbeddingQueueResult::Accepted);
        assert_eq!(q.len(), 1);
        assert!(!q.is_empty());

        let res = q.push("req-1", "att-1", EmbeddingChannel::Content, 1, 1, "world");
        assert_eq!(res, EmbeddingQueueResult::Accepted);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn queue_push_input_accepts() {
        let mut q = EmbeddingQueue::new(4, EmbeddingQueuePolicy::Skip);
        let input = make_input("r", "a", EmbeddingChannel::Reasoning, 5);
        assert_eq!(q.push_input(input), EmbeddingQueueResult::Accepted);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_capacity_clamped_to_one() {
        let q = EmbeddingQueue::new(0, EmbeddingQueuePolicy::Skip);
        assert_eq!(q.capacity(), 1);
    }

    #[test]
    fn queue_overflow_skip_policy() {
        let mut q = EmbeddingQueue::new(2, EmbeddingQueuePolicy::Skip);
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 0, 0, "x"),
            EmbeddingQueueResult::Accepted
        );
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "y"),
            EmbeddingQueueResult::Accepted
        );
        // Queue is full; Skip policy drops the window.
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 2, 2, "z"),
            EmbeddingQueueResult::Skipped
        );
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn queue_overflow_deterministic_only_policy() {
        let mut q = EmbeddingQueue::new(1, EmbeddingQueuePolicy::DeterministicOnly);
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 0, 0, "x"),
            EmbeddingQueueResult::Accepted
        );
        assert_eq!(q.policy(), EmbeddingQueuePolicy::DeterministicOnly);
        // DeterministicOnly also reports Skipped; caller inspects policy.
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "y"),
            EmbeddingQueueResult::Skipped
        );
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_overflow_block_policy() {
        let mut q = EmbeddingQueue::new(1, EmbeddingQueuePolicy::Block);
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 0, 0, "x"),
            EmbeddingQueueResult::Accepted
        );
        // Block policy refuses the push when full.
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "y"),
            EmbeddingQueueResult::QueueFull
        );
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_deduplicates_same_window_key() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        q.push("r", "a", EmbeddingChannel::Content, 3, 3, "dup");
        // Same (request_id, attempt_id, channel, window_seq) -> skipped even with different text.
        let res = q.push("r", "a", EmbeddingChannel::Content, 3, 3, "different-text");
        assert_eq!(res, EmbeddingQueueResult::Skipped);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_different_channels_are_distinct() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        q.push("r", "a", EmbeddingChannel::Content, 0, 0, "c");
        let res = q.push("r", "a", EmbeddingChannel::Reasoning, 0, 0, "r");
        assert_eq!(res, EmbeddingQueueResult::Accepted);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn queue_drain_batch_respects_max() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        for i in 0..5 {
            q.push("r", "a", EmbeddingChannel::Content, i, i, "t");
        }
        let batch = q.drain_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(q.len(), 2);
        // FIFO order preserved.
        assert_eq!(batch[0].window_seq, 0);
        assert_eq!(batch[2].window_seq, 2);
    }

    #[test]
    fn queue_drain_batch_zero_returns_empty() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        q.push("r", "a", EmbeddingChannel::Content, 0, 0, "t");
        assert!(q.drain_batch(0).is_empty());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_drain_all_empties_queue() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        for i in 0..4 {
            q.push("r", "a", EmbeddingChannel::Content, i, i, "t");
        }
        let all = q.drain_all();
        assert_eq!(all.len(), 4);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_front_peek() {
        let mut q = EmbeddingQueue::new(8, EmbeddingQueuePolicy::Skip);
        assert!(q.front().is_none());
        q.push("r", "a", EmbeddingChannel::Content, 7, 7, "t");
        let front = q.front().unwrap();
        assert_eq!(front.window_seq, 7);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_capacity_reported() {
        let q = EmbeddingQueue::new(16, EmbeddingQueuePolicy::Block);
        assert_eq!(q.capacity(), 16);
        assert_eq!(q.policy(), EmbeddingQueuePolicy::Block);
    }

    #[test]
    fn queue_fills_to_exactly_capacity_before_overflow() {
        let cap = 3;
        let mut q = EmbeddingQueue::new(cap, EmbeddingQueuePolicy::Skip);
        for i in 0..cap {
            assert_eq!(
                q.push("r", "a", EmbeddingChannel::Content, i as u64, i as u64, "t"),
                EmbeddingQueueResult::Accepted,
                "window {i} should be accepted"
            );
        }
        assert_eq!(q.len(), cap);
        // Next push overflows.
        assert_eq!(
            q.push(
                "r",
                "a",
                EmbeddingChannel::Content,
                cap as u64,
                cap as u64,
                "t"
            ),
            EmbeddingQueueResult::Skipped
        );
    }

    #[test]
    fn queue_drain_then_refill_after_overflow() {
        let mut q = EmbeddingQueue::new(1, EmbeddingQueuePolicy::Skip);
        q.push("r", "a", EmbeddingChannel::Content, 0, 0, "t");
        // Overflow.
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "t"),
            EmbeddingQueueResult::Skipped
        );
        // Drain frees space.
        q.drain_all();
        // Now a new window can be accepted.
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 2, 2, "t"),
            EmbeddingQueueResult::Accepted
        );
    }

    #[test]
    fn queue_block_policy_then_drain_allows_refill() {
        let mut q = EmbeddingQueue::new(1, EmbeddingQueuePolicy::Block);
        q.push("r", "a", EmbeddingChannel::Content, 0, 0, "t");
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "t"),
            EmbeddingQueueResult::QueueFull
        );
        q.drain_all();
        assert_eq!(
            q.push("r", "a", EmbeddingChannel::Content, 1, 1, "t"),
            EmbeddingQueueResult::Accepted
        );
    }

    // ---- SemanticLoopScorer tests (issue #107) ----

    fn vec_from(seq: u64, comps: &[f32]) -> EmbeddingVector {
        EmbeddingVector {
            window_seq: seq,
            components: comps.to_vec(),
        }
    }

    fn content_config(history: usize) -> SemanticLoopConfig {
        SemanticLoopConfig {
            similarity_threshold: 0.90,
            cluster_density_threshold: 0.55,
            low_novelty_median: 0.08,
            history_window_count: history,
        }
    }

    #[test]
    fn semantic_loop_identical_vectors_produce_high_risk() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        let v = vec_from(0, &[1.0, 2.0, 3.0, 4.0]);

        // First observation: no history yet -> no signal.
        assert!(
            scorer
                .observe_vector(EmbeddingChannel::Content, &v)
                .is_none()
        );

        // Second identical observation: max_sim == 1.0, density == 1.0,
        // novelty_median == 0.0 -> loop.
        let sig = scorer
            .observe_vector(EmbeddingChannel::Content, &v)
            .expect("identical vector should trigger a signal");

        assert_eq!(sig.channel, EmbeddingChannel::Content);
        assert!(
            (sig.max_similarity - 1.0).abs() < 1e-5,
            "got {}",
            sig.max_similarity
        );
        assert!(
            (sig.cluster_density - 1.0).abs() < 1e-5,
            "got {}",
            sig.cluster_density
        );
        assert!(
            sig.novelty_median.abs() < 1e-5,
            "got {}",
            sig.novelty_median
        );
        // risk = density * (1 - novelty_median) ~= 1.0
        assert!(sig.risk > 0.99, "risk should be near 1.0, got {}", sig.risk);
    }

    #[test]
    fn semantic_loop_dissimilar_vectors_produce_no_signal() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        // Orthogonal vectors: cosine == 0 -> novelty == 1.0, density == 0.
        scorer.observe_vector(EmbeddingChannel::Content, &vec_from(0, &[1.0, 0.0, 0.0]));
        let sig = scorer.observe_vector(EmbeddingChannel::Content, &vec_from(1, &[0.0, 1.0, 0.0]));
        assert!(sig.is_none(), "orthogonal vectors should not loop");
    }

    #[test]
    fn semantic_loop_cluster_density_correctly_computed() {
        // Threshold 0.90. Three historical vectors: two identical to the probe,
        // one orthogonal. density should be 2/3 ~= 0.666 >= 0.55.
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        let probe = vec_from(99, &[1.0, 1.0, 1.0]);
        let same = vec_from(0, &[1.0, 1.0, 1.0]);
        let ortho = vec_from(1, &[0.0, 0.0, 1.0]);

        // Seed history.
        scorer.observe_vector(EmbeddingChannel::Content, &same);
        scorer.observe_vector(EmbeddingChannel::Content, &ortho);
        scorer.observe_vector(EmbeddingChannel::Content, &same);

        let sig = scorer
            .observe_vector(EmbeddingChannel::Content, &probe)
            .expect("should fire: density 2/3 with low novelty median from the identical pair");

        // density = 2/3 (two identical out of three defined comparisons)
        assert!(
            (sig.cluster_density - (2.0 / 3.0)).abs() < 1e-5,
            "density got {}",
            sig.cluster_density
        );
    }

    #[test]
    fn semantic_loop_channel_specific_thresholds_respected() {
        // Build a scorer and confirm default thresholds differ per channel.
        let scorer = SemanticLoopScorer::new(SemanticLoopConfig::default());
        assert!((scorer.channel_threshold(EmbeddingChannel::Reasoning) - 0.88).abs() < 1e-6);
        assert!((scorer.channel_threshold(EmbeddingChannel::Content) - 0.90).abs() < 1e-6);
        assert!((scorer.channel_threshold(EmbeddingChannel::ToolArgs) - 0.92).abs() < 1e-6);

        // A pair of near-identical vectors that are above the reasoning
        // threshold (0.88) but below content (0.90) and tool args (0.92).
        // cosine([1,0,0],[1,0.5,0]) = 1/sqrt(1.25) ~= 0.8944
        let a = vec_from(0, &[1.0, 0.0, 0.0]);
        let b = vec_from(1, &[1.0, 0.5, 0.0]);

        // Reasoning: 0.8944 >= 0.88 -> density 1.0, novelty ~0.1056.
        // But novelty_median ~0.1056 > low_novelty_median 0.08 -> NO signal.
        let mut r = SemanticLoopScorer::new(SemanticLoopConfig {
            similarity_threshold: 0.90,
            cluster_density_threshold: 0.55,
            low_novelty_median: 0.08,
            history_window_count: 8,
        });
        r.observe_vector(EmbeddingChannel::Reasoning, &a);
        let sig = r.observe_vector(EmbeddingChannel::Reasoning, &b);
        // Cosine ~0.894 -> novelty ~0.106 > 0.08 threshold, so no loop.
        assert!(sig.is_none(), "reasoning pair below novelty bar");

        // ToolArgs threshold is 0.92: same pair is far below it -> density 0.
        let mut t = SemanticLoopScorer::new(SemanticLoopConfig::default());
        t.observe_vector(EmbeddingChannel::ToolArgs, &a);
        let sig = t.observe_vector(EmbeddingChannel::ToolArgs, &b);
        assert!(sig.is_none(), "tool args pair below similarity threshold");
    }

    #[test]
    fn semantic_loop_history_bounded_correctly() {
        let cap = 4;
        let mut scorer = SemanticLoopScorer::new(content_config(cap));

        // Use explicit f32 literals to avoid usize→f32 precision-loss lints.
        let vals: [(u64, f32); 7] = [
            (0, 0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 3.0),
            (4, 4.0),
            (5, 5.0),
            (6, 6.0),
        ];
        for (seq, i_f) in vals {
            scorer.observe_vector(
                EmbeddingChannel::Content,
                &vec_from(seq, &[i_f + 1.0, i_f + 2.0]),
            );
        }

        let history = scorer.channel_history(EmbeddingChannel::Content);
        assert_eq!(
            history.len(),
            cap,
            "history must be bounded by history_window_count"
        );
        // Oldest entries evicted; the surviving window_seqs are the last `cap`.
        let seqs: Vec<u64> = history.iter().map(|v| v.window_seq).collect();
        assert_eq!(seqs, vec![3, 4, 5, 6]);
    }

    #[test]
    fn semantic_loop_history_independent_per_channel() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        scorer.observe_vector(EmbeddingChannel::Reasoning, &vec_from(0, &[1.0, 0.0]));
        scorer.observe_vector(EmbeddingChannel::Content, &vec_from(0, &[1.0, 0.0]));
        scorer.observe_vector(EmbeddingChannel::Content, &vec_from(1, &[0.0, 1.0]));

        assert_eq!(scorer.channel_history(EmbeddingChannel::Reasoning).len(), 1);
        assert_eq!(scorer.channel_history(EmbeddingChannel::Content).len(), 2);
        assert_eq!(scorer.channel_history(EmbeddingChannel::ToolArgs).len(), 0);
    }

    #[test]
    fn semantic_loop_single_observation_no_signal() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        // First observation never fires (no history to compare against).
        assert!(
            scorer
                .observe_vector(EmbeddingChannel::Content, &vec_from(0, &[1.0, 2.0]))
                .is_none()
        );
    }

    #[test]
    fn semantic_loop_degenerate_vector_no_signal() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        // Zero vectors: cosine_similarity returns None -> no readings.
        scorer.observe_vector(EmbeddingChannel::Content, &vec_from(0, &[0.0, 0.0]));
        let sig = scorer.observe_vector(EmbeddingChannel::Content, &vec_from(1, &[0.0, 0.0]));
        assert!(
            sig.is_none(),
            "degenerate vectors must not produce a signal"
        );
    }

    #[test]
    fn semantic_loop_signal_fields_populated() {
        let mut scorer = SemanticLoopScorer::new(content_config(8));
        let v = vec_from(0, &[2.0, 3.0, 1.0]);
        scorer.observe_vector(EmbeddingChannel::Content, &v);
        let sig = scorer
            .observe_vector(EmbeddingChannel::Content, &v)
            .unwrap();
        assert_eq!(sig.channel, EmbeddingChannel::Content);
        assert!(sig.risk >= 0.0 && sig.risk <= 1.0);
        assert!(sig.max_similarity >= 0.0 && sig.max_similarity <= 1.0);
        assert!(sig.cluster_density >= 0.0 && sig.cluster_density <= 1.0);
        assert!(sig.novelty_median >= 0.0);
    }

    #[test]
    fn semantic_loop_set_channel_threshold_overrides() {
        let mut scorer = SemanticLoopScorer::new(SemanticLoopConfig::default());
        // Lower the content threshold so a borderline pair now clusters.
        scorer.set_channel_threshold(EmbeddingChannel::Content, 0.80);
        assert!((scorer.channel_threshold(EmbeddingChannel::Content) - 0.80).abs() < 1e-6);

        // cosine ~0.894 >= 0.80, novelty ~0.106 > 0.08 low_novelty -> still no.
        // But if we also raise low_novelty_median, it should fire.
        scorer.config.low_novelty_median = 0.20;
        let a = vec_from(0, &[1.0, 0.0, 0.0]);
        let b = vec_from(1, &[1.0, 0.5, 0.0]);
        scorer.observe_vector(EmbeddingChannel::Content, &a);
        let sig = scorer.observe_vector(EmbeddingChannel::Content, &b);
        assert!(
            sig.is_some(),
            "lowered threshold and raised novelty bar should fire"
        );
    }
}
