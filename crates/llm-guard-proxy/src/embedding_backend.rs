//! OpenAI-compatible embedding backend for semantic loop detection.
//!
//! Uses `reqwest` to call a configurable `/v1/embeddings` endpoint. This lives
//! in the proxy crate (not core) because it needs an async HTTP client.

#![allow(dead_code)]

use llm_guard_proxy_core::embedding::{
    EmbeddingBackend, EmbeddingError, EmbeddingFuture, EmbeddingInput, EmbeddingVector,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Configuration for the OpenAI-compatible embedding backend.
#[derive(Clone, Debug)]
pub struct OpenAiEmbeddingConfig {
    /// Full endpoint URL, e.g. `http://100.105.4.92:18012/v1/embeddings`.
    pub endpoint: String,
    /// Model name to pass to the endpoint, e.g. `Qwen/Qwen3-Embedding-0.6B`.
    pub model: String,
    /// Optional API key for the endpoint.
    pub api_key: Option<String>,
    /// MRL dimension truncation (0 = no truncation).
    pub vector_dim: usize,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
}

/// OpenAI-compatible embedding backend that calls a real `/v1/embeddings` endpoint.
pub struct OpenAiCompatibleEmbeddingBackend {
    config: OpenAiEmbeddingConfig,
    client: Client,
}

impl OpenAiCompatibleEmbeddingBackend {
    /// Creates a new backend with the given config and HTTP client.
    #[must_use]
    pub fn new(config: OpenAiEmbeddingConfig, client: Client) -> Self {
        Self { config, client }
    }
}

// The EmbeddingBackend trait impl uses `async move` which requires edition 2024
// (the workspace default). The write_file linter may not detect the edition.

impl EmbeddingBackend for OpenAiCompatibleEmbeddingBackend {
    fn embed_batch(&self, inputs: Vec<EmbeddingInput>) -> EmbeddingFuture<'_> {
        let config = self.config.clone();
        let client = self.client.clone();
        Box::pin(async move {
            if inputs.is_empty() {
                return Err(EmbeddingError::EmptyBatch);
            }
            let texts: Vec<&str> = inputs.iter().map(|i| i.text.as_str()).collect();
            let request_body = EmbeddingRequest {
                model: &config.model,
                input: texts,
            };
            let timeout = std::time::Duration::from_millis(config.timeout_ms);
            let mut request = client
                .post(&config.endpoint)
                .json(&request_body)
                .timeout(timeout);
            if let Some(ref api_key) = config.api_key {
                request = request.bearer_auth(api_key);
            }
            let response = request
                .send()
                .await
                .map_err(|e| EmbeddingError::Request(e.to_string()))?;
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let body = response.text().await.unwrap_or_default();
                return Err(EmbeddingError::Status { status, body });
            }
            let body: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::Parse(e.to_string()))?;
            let dim = config.vector_dim;
            let vectors = body
                .data
                .into_iter()
                .zip(inputs)
                .map(|(item, input)| {
                    let components = if dim > 0 && dim < item.embedding.len() {
                        item.embedding[..dim].to_vec()
                    } else {
                        item.embedding
                    };
                    EmbeddingVector {
                        window_seq: input.window_seq,
                        components,
                    }
                })
                .collect();
            Ok(vectors)
        })
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}
