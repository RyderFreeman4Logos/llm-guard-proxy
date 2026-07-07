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
}
