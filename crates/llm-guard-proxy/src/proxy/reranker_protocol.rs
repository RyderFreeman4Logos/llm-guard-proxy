//! Endpoint-specific rendering for heterogeneous reranker replicas.

use std::cmp::Ordering;

use axum::{
    body::Bytes,
    http::{
        HeaderMap, HeaderValue, Method, Uri,
        header::{AUTHORIZATION, CONTENT_TYPE},
    },
};
use llm_guard_proxy_core::{UpstreamEndpointConfig, UpstreamEndpointProtocol};
use reqwest::Url;
use serde_json::{Value, json};

use super::{
    ProxyError, buffered_adapter::sanitize_transformed_request_headers, deepinfra_rerank_adapter,
    score_adapter,
};

const MAX_PAIR_COUNT: usize = 1_024;
const MAX_REQUEST_ID_BYTES: usize = 256;

/// Preserved public request data used to render each selected replica.
#[derive(Clone, Debug)]
pub(super) enum CanonicalRerankerRequest {
    /// A public `OpenAI` rerank request.
    OpenAiRerank { forward_uri: Uri, body: Bytes },
    /// A scalar score request normalized to the `OpenAI` rerank contract.
    Score { forward_uri: Uri, body: Bytes },
    /// A public DeepInfra-native Qwen3 rerank request.
    DeepInfraNative { uri: Uri, body: Bytes },
    /// A valid `OpenAI` score shape that cannot be represented by `DeepInfra`.
    UnsupportedScore,
}

/// Buffered response conversion chosen from the public request contract.
#[derive(Clone, Copy, Debug)]
pub(super) enum ResponseContract {
    /// Preserve an `OpenAI` `/v1/rerank` response locally, or create one from `DeepInfra`.
    OpenAiRerank { document_count: usize, top_n: usize },
    /// Preserve the local score-adapter response, or create it from `DeepInfra`.
    Score {
        document_count: usize,
        top_n: usize,
        expected: Option<score_adapter::ScoreExpectations>,
    },
    /// Preserve the local native-adapter response, or normalize `DeepInfra`'s response.
    DeepInfraNative { expected_count: usize },
}

/// Fully rendered endpoint data. It intentionally omits `Debug` to prevent accidental
/// formatting of the runtime-only authorization header.
pub(super) struct RenderedEndpointRequest {
    pub(super) uri: Uri,
    pub(super) url: Url,
    pub(super) body: Bytes,
    pub(super) headers: HeaderMap,
}

/// Capture a reranker request before replica selection without committing to a wire protocol.
pub(super) fn capture_request(
    method: &Method,
    original_uri: &Uri,
    original_body: &Bytes,
    adapted_uri: &Uri,
    adapted_body: &Bytes,
) -> Option<CanonicalRerankerRequest> {
    if deepinfra_rerank_adapter::is_request(method, original_uri) {
        return Some(CanonicalRerankerRequest::DeepInfraNative {
            uri: original_uri.clone(),
            body: original_body.clone(),
        });
    }
    if score_adapter::is_score_request(method, original_uri) {
        return Some(if adapted_uri.path() == "/v1/rerank" {
            CanonicalRerankerRequest::Score {
                forward_uri: adapted_uri.clone(),
                body: adapted_body.clone(),
            }
        } else {
            CanonicalRerankerRequest::UnsupportedScore
        });
    }
    (*method == Method::POST && original_uri.path() == "/v1/rerank").then(|| {
        CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: adapted_uri.clone(),
            body: adapted_body.clone(),
        }
    })
}

/// Return the normalized public response contract, validating data only when a remote
/// `DeepInfra` replica is selected.
pub(super) fn response_contract(
    request: &CanonicalRerankerRequest,
) -> Result<ResponseContract, ProxyError> {
    match request {
        CanonicalRerankerRequest::OpenAiRerank { body, .. } => {
            let input = parse_openai_rerank(body)?;
            Ok(ResponseContract::OpenAiRerank {
                document_count: input.documents.len(),
                top_n: input.top_n,
            })
        }
        CanonicalRerankerRequest::Score { body, .. } => {
            let input = parse_openai_rerank(body)?;
            Ok(ResponseContract::Score {
                document_count: input.documents.len(),
                top_n: input.top_n,
                expected: score_adapter::score_expectations_from_rerank_body(body),
            })
        }
        CanonicalRerankerRequest::DeepInfraNative { uri, body } => {
            let adapted = deepinfra_rerank_adapter::adapt_request(uri, body)
                .map_err(|error| invalid_request_error(&error.to_string()))?;
            Ok(ResponseContract::DeepInfraNative {
                expected_count: adapted.response_expectations.result_count,
            })
        }
        CanonicalRerankerRequest::UnsupportedScore => Err(unsupported_request_error()),
    }
}

/// Render the protocol, URL, body, and authentication for one selected endpoint.
pub(super) fn render(
    endpoint: &UpstreamEndpointConfig,
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
) -> Result<RenderedEndpointRequest, ProxyError> {
    match endpoint.protocol {
        UpstreamEndpointProtocol::OpenAi => render_openai(endpoint, request, downstream_headers),
        UpstreamEndpointProtocol::DeepInfraQwen3Rerank => {
            render_deepinfra(endpoint, request, downstream_headers)
        }
    }
}

/// Runtime eligibility is intentionally credential-presence-only: `DeepInfra` inference must
/// never be sent as a health probe.
pub(super) fn has_runtime_credential(endpoint: &UpstreamEndpointConfig) -> bool {
    match endpoint.protocol {
        UpstreamEndpointProtocol::OpenAi => true,
        UpstreamEndpointProtocol::DeepInfraQwen3Rerank => authorization_header(endpoint).is_ok(),
    }
}

/// Rewrite a `DeepInfra` response into the public contract. Local `OpenAI`-compatible responses
/// are identified structurally and keep their pre-existing adapter conversion.
pub(super) fn rewrite_response(
    body: &Bytes,
    contract: ResponseContract,
    model_id: Option<&str>,
) -> Result<Bytes, String> {
    if !looks_like_deepinfra_response(body) {
        return match contract {
            ResponseContract::OpenAiRerank { .. } => Ok(body.clone()),
            ResponseContract::Score { expected, .. } => {
                score_adapter::rerank_response_to_score_response(body, model_id, expected)
            }
            ResponseContract::DeepInfraNative { expected_count } => {
                let expected = deepinfra_rerank_adapter::ResponseExpectations {
                    result_count: expected_count,
                };
                deepinfra_rerank_adapter::score_response_to_deepinfra_response(body, expected)
            }
        };
    }

    let response = parse_deepinfra_response(body)?;
    match contract {
        ResponseContract::OpenAiRerank {
            document_count,
            top_n,
        } => {
            ensure_exact_score_count(&response, document_count)?;
            openai_rerank_response(&response, top_n)
        }
        ResponseContract::Score {
            document_count,
            top_n,
            expected,
        } => {
            ensure_exact_score_count(&response, document_count)?;
            score_response(&response, top_n, expected, model_id)
        }
        ResponseContract::DeepInfraNative { expected_count } => {
            ensure_exact_score_count(&response, expected_count)?;
            deepinfra_native_response(&response)
        }
    }
}

fn render_openai(
    endpoint: &UpstreamEndpointConfig,
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
) -> Result<RenderedEndpointRequest, ProxyError> {
    let (uri, body) = match request {
        CanonicalRerankerRequest::OpenAiRerank { forward_uri, body }
        | CanonicalRerankerRequest::Score { forward_uri, body } => {
            (forward_uri.clone(), body.clone())
        }
        CanonicalRerankerRequest::DeepInfraNative { uri, body } => {
            let adapted = deepinfra_rerank_adapter::adapt_request(uri, body)
                .map_err(|error| invalid_request_error(&error.to_string()))?;
            (adapted.forward_uri, adapted.body)
        }
        CanonicalRerankerRequest::UnsupportedScore => return Err(unsupported_request_error()),
    };
    Ok(RenderedEndpointRequest {
        url: super::build_upstream_url(&endpoint.base_url, &uri)?,
        uri,
        body,
        headers: request_headers_for_openai(request, downstream_headers),
    })
}

fn request_headers_for_openai(
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
) -> HeaderMap {
    match request {
        CanonicalRerankerRequest::OpenAiRerank { .. } => downstream_headers.clone(),
        CanonicalRerankerRequest::Score { .. }
        | CanonicalRerankerRequest::DeepInfraNative { .. }
        | CanonicalRerankerRequest::UnsupportedScore => {
            sanitize_transformed_request_headers(downstream_headers)
        }
    }
}

fn render_deepinfra(
    endpoint: &UpstreamEndpointConfig,
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
) -> Result<RenderedEndpointRequest, ProxyError> {
    let authorization = authorization_header(endpoint)?;
    render_deepinfra_with_authorization(endpoint, request, downstream_headers, authorization)
}

fn render_deepinfra_with_authorization(
    endpoint: &UpstreamEndpointConfig,
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
    authorization: HeaderValue,
) -> Result<RenderedEndpointRequest, ProxyError> {
    let (body, uri) = match request {
        CanonicalRerankerRequest::OpenAiRerank { body, .. }
        | CanonicalRerankerRequest::Score { body, .. } => {
            let input = parse_openai_rerank(body)?;
            let queries = vec![input.query; input.documents.len()];
            let body = serde_json::to_vec(&json!({
                "queries": queries,
                "documents": input.documents,
                "instruction": deepinfra_rerank_adapter::DEFAULT_INSTRUCTION,
            }))
            .map(Bytes::from)
            .map_err(|error| {
                invalid_request_error(&format!("serialize DeepInfra request failed: {error}"))
            })?;
            (body, deepinfra_inference_uri(endpoint)?)
        }
        CanonicalRerankerRequest::DeepInfraNative { body, .. } => {
            (body.clone(), deepinfra_inference_uri(endpoint)?)
        }
        CanonicalRerankerRequest::UnsupportedScore => return Err(unsupported_request_error()),
    };
    let mut headers = sanitize_transformed_request_headers(downstream_headers);
    headers.remove(AUTHORIZATION);
    headers.insert(AUTHORIZATION, authorization);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let url = super::build_upstream_url(&endpoint.base_url, &uri)?;
    Ok(RenderedEndpointRequest {
        uri,
        url,
        body,
        headers,
    })
}

fn deepinfra_inference_uri(endpoint: &UpstreamEndpointConfig) -> Result<Uri, ProxyError> {
    let model = endpoint
        .model
        .as_deref()
        .ok_or_else(unsupported_request_error)?;
    format!("/v1/inference/{model}").parse().map_err(|error| {
        invalid_request_error(&format!("invalid DeepInfra inference path: {error}"))
    })
}

fn authorization_header(endpoint: &UpstreamEndpointConfig) -> Result<HeaderValue, ProxyError> {
    let variable = endpoint
        .api_key_env
        .as_deref()
        .ok_or_else(unsupported_request_error)?;
    let token = std::env::var(variable).map_err(|_error| credential_error())?;
    if token.trim().is_empty() {
        return Err(credential_error());
    }
    HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_error| credential_error())
}

struct OpenAiRerankInput {
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

fn parse_openai_rerank(body: &Bytes) -> Result<OpenAiRerankInput, ProxyError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| invalid_request_error(&format!("invalid rerank JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid_request_error("rerank body must be a JSON object"))?;
    let query = object
        .get("query")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid_request_error("rerank query must be a string"))?;
    let documents = object
        .get("documents")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_request_error("rerank documents must be an array"))?;
    if documents.is_empty() || documents.len() > MAX_PAIR_COUNT {
        return Err(invalid_request_error(&format!(
            "rerank documents must contain between 1 and {MAX_PAIR_COUNT} strings"
        )));
    }
    let documents = documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            document.as_str().map(str::to_owned).ok_or_else(|| {
                invalid_request_error(&format!("rerank documents[{index}] must be a string"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let top_n = match object.get("top_n") {
        None => documents.len(),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0 && *value <= documents.len())
            .ok_or_else(|| {
                invalid_request_error(
                    "rerank top_n must be a positive integer no greater than document count",
                )
            })?,
    };
    Ok(OpenAiRerankInput {
        query,
        documents,
        top_n,
    })
}

struct DeepInfraResponse {
    scores: Vec<f64>,
    input_tokens: u64,
    request_id: Option<String>,
}

fn parse_deepinfra_response(body: &Bytes) -> Result<DeepInfraResponse, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| format!("invalid DeepInfra rerank JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| String::from("DeepInfra rerank response must be a JSON object"))?;
    let scores = object
        .get("scores")
        .and_then(Value::as_array)
        .ok_or_else(|| String::from("DeepInfra rerank response scores must be an array"))?;
    if scores.is_empty() || scores.len() > MAX_PAIR_COUNT {
        return Err(format!(
            "DeepInfra rerank response scores must contain between 1 and {MAX_PAIR_COUNT} values"
        ));
    }
    let scores = scores
        .iter()
        .enumerate()
        .map(|(index, score)| {
            score
                .as_f64()
                .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
                .ok_or_else(|| {
                    format!(
                        "DeepInfra rerank response scores[{index}] must be a finite value in [0, 1]"
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_tokens = object
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            String::from("DeepInfra rerank response input_tokens must be a non-negative integer")
        })?;
    let request_id = match object.get("request_id") {
        None => None,
        Some(Value::String(request_id))
            if !request_id.is_empty()
                && request_id.len() <= MAX_REQUEST_ID_BYTES
                && !request_id.chars().any(char::is_control) =>
        {
            Some(request_id.clone())
        }
        Some(_) => {
            return Err(String::from(
                "DeepInfra rerank response request_id must be a bounded non-empty string",
            ));
        }
    };
    Ok(DeepInfraResponse {
        scores,
        input_tokens,
        request_id,
    })
}

fn openai_rerank_response(response: &DeepInfraResponse, top_n: usize) -> Result<Bytes, String> {
    let results = selected_scores(&response.scores, top_n)?
        .into_iter()
        .map(|(index, score)| json!({"index": index, "relevance_score": score}))
        .collect::<Vec<_>>();
    serialize(&json!({"object": "list", "data": results}))
}

fn score_response(
    response: &DeepInfraResponse,
    top_n: usize,
    expected: Option<score_adapter::ScoreExpectations>,
    model_id: Option<&str>,
) -> Result<Bytes, String> {
    let scores = selected_scores(&response.scores, top_n)?;
    if let Some(expected) = expected
        && scores.len() != expected.result_count
    {
        return Err(format!(
            "DeepInfra rerank response count {} does not match expected {}",
            scores.len(),
            expected.result_count
        ));
    }
    let model = model_id.ok_or_else(|| String::from("score request is missing model"))?;
    let data = scores
        .into_iter()
        .map(|(index, score)| json!({"index": index, "object": "score", "score": score}))
        .collect::<Vec<_>>();
    serialize(&json!({
        "id": "score-adapted",
        "object": "list",
        "created": unix_time_seconds(),
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": response.input_tokens},
    }))
}

fn deepinfra_native_response(response: &DeepInfraResponse) -> Result<Bytes, String> {
    let mut output = serde_json::Map::new();
    output.insert(String::from("scores"), json!(response.scores));
    output.insert(
        String::from("input_tokens"),
        Value::from(response.input_tokens),
    );
    if let Some(request_id) = &response.request_id {
        output.insert(
            String::from("request_id"),
            Value::String(request_id.clone()),
        );
    }
    serialize(&Value::Object(output))
}

fn selected_scores(scores: &[f64], top_n: usize) -> Result<Vec<(usize, f64)>, String> {
    if top_n == 0 || top_n > scores.len() {
        return Err(String::from(
            "DeepInfra rerank response does not satisfy requested top_n",
        ));
    }
    let mut indexed = scores.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    indexed.truncate(top_n);
    Ok(indexed)
}

fn looks_like_deepinfra_response(body: &Bytes) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .as_object()
                .map(|object| object.contains_key("scores"))
        })
        .unwrap_or(false)
}

fn ensure_exact_score_count(
    response: &DeepInfraResponse,
    expected_count: usize,
) -> Result<(), String> {
    if response.scores.len() == expected_count {
        Ok(())
    } else {
        Err(format!(
            "DeepInfra rerank response count {} does not match expected {}",
            response.scores.len(),
            expected_count
        ))
    }
}

fn serialize(value: &Value) -> Result<Bytes, String> {
    serde_json::to_vec(&value)
        .map(Bytes::from)
        .map_err(|error| format!("serialize reranker response failed: {error}"))
}

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn invalid_request_error(message: &str) -> ProxyError {
    ProxyError::ContextBudgetExceeded {
        message: format!("invalid reranker request for selected endpoint: {message}"),
        param: "body",
        code: "invalid_reranker_endpoint_request",
        request_metadata: None,
    }
}

fn unsupported_request_error() -> ProxyError {
    ProxyError::ContextBudgetExceeded {
        message: String::from(
            "selected reranker endpoint cannot represent this request; use a scalar text rerank shape",
        ),
        param: "body",
        code: "unsupported_reranker_endpoint_request",
        request_metadata: None,
    }
}

fn credential_error() -> ProxyError {
    ProxyError::ContextBudgetExceeded {
        message: String::from("selected DeepInfra endpoint has no usable runtime credential"),
        param: "api_key_env",
        code: "upstream_endpoint_credential_unavailable",
        request_metadata: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_guard_proxy_core::{UpstreamEndpointConfig, UpstreamEndpointProtocol};

    const SENTINEL: &str = "LLM_GUARD_PROXY_NOT_A_REAL_CREDENTIAL_195";

    #[test]
    fn converts_deepinfra_scores_to_indexed_openai_rerank_results() {
        let body = Bytes::from_static(
            br#"{"scores":[0.2,0.9,0.5],"input_tokens":19,"request_id":"request-195"}"#,
        );
        let response = rewrite_response(
            &body,
            ResponseContract::OpenAiRerank {
                document_count: 3,
                top_n: 2,
            },
            None,
        )
        .expect("response should convert");
        let response: Value = serde_json::from_slice(&response).expect("response should be JSON");
        assert_eq!(response["data"][0]["index"], 1);
        assert_eq!(response["data"][0]["relevance_score"], 0.9);
        assert!(response.get("request_id").is_none());
        assert!(response.get("input_tokens").is_none());
    }

    #[test]
    fn deepinfra_renderer_replaces_downstream_authorization() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            ..UpstreamEndpointConfig::default()
        };
        let request = CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: Uri::from_static("/v1/rerank"),
            body: Bytes::from_static(br#"{"query":"q","documents":["d1","d2"]}"#),
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer downstream-value"),
        );
        let rendered = render_deepinfra_with_authorization(
            &endpoint,
            &request,
            &headers,
            HeaderValue::from_static("Bearer LLM_GUARD_PROXY_NOT_A_REAL_CREDENTIAL_195"),
        )
        .expect("request should render");
        assert_eq!(rendered.uri.path(), "/v1/inference/Qwen/Qwen3-Reranker-8B");
        assert_eq!(
            rendered
                .headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer LLM_GUARD_PROXY_NOT_A_REAL_CREDENTIAL_195")
        );
        let body: Value = serde_json::from_slice(&rendered.body).expect("rendered body JSON");
        assert_eq!(body["queries"], json!(["q", "q"]));
        assert_eq!(body["documents"], json!(["d1", "d2"]));
        assert_eq!(
            body["instruction"],
            deepinfra_rerank_adapter::DEFAULT_INSTRUCTION
        );
        let debug = format!("{endpoint:?}");
        assert!(!debug.contains(SENTINEL));
    }
}
