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
}
