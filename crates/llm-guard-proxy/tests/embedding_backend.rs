//! Integration test for the embedding backend types.
//!
//! Tests the `DisabledEmbeddingBackend` (returns empty) and the
//! `EmbeddingVector::cosine_similarity` utility.

use llm_guard_proxy_core::embedding::{
    DisabledEmbeddingBackend, EmbeddingBackend, EmbeddingChannel, EmbeddingInput, EmbeddingVector,
};

#[tokio::test]
async fn disabled_backend_returns_empty_results() {
    let backend = DisabledEmbeddingBackend::new();
    let inputs = vec![
        EmbeddingInput {
            request_id: "req1".to_string(),
            attempt_id: "att1".to_string(),
            channel: EmbeddingChannel::Reasoning,
            window_seq: 0,
            text_hash: 42,
            text: "thinking about the problem".to_string(),
        },
        EmbeddingInput {
            request_id: "req1".to_string(),
            attempt_id: "att1".to_string(),
            channel: EmbeddingChannel::Reasoning,
            window_seq: 1,
            text_hash: 43,
            text: "still thinking".to_string(),
        },
    ];
    let result = backend.embed_batch(inputs).await;
    assert!(result.is_ok(), "disabled backend should not error");
    assert!(
        result.unwrap().is_empty(),
        "disabled backend returns no vectors"
    );
}

#[tokio::test]
async fn disabled_backend_empty_input() {
    let backend = DisabledEmbeddingBackend::new();
    let result = backend.embed_batch(Vec::new()).await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn cosine_similarity_identical() {
    let v = EmbeddingVector {
        window_seq: 0,
        components: vec![0.1, 0.2, 0.3, 0.4],
    };
    let sim = v.cosine_similarity(&v).unwrap();
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_similarity_opposite() {
    let a = EmbeddingVector {
        window_seq: 0,
        components: vec![1.0, 0.0],
    };
    let b = EmbeddingVector {
        window_seq: 1,
        components: vec![-1.0, 0.0],
    };
    let sim = a.cosine_similarity(&b).unwrap();
    assert!((sim + 1.0).abs() < 1e-5);
}

#[test]
fn mrl_truncation_preserves_first_dim() {
    let v = EmbeddingVector {
        window_seq: 0,
        components: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    };
    let truncated = v.truncated(4);
    assert_eq!(truncated.components.len(), 4);
    assert_eq!(truncated.components, vec![1.0, 2.0, 3.0, 4.0]);
}
