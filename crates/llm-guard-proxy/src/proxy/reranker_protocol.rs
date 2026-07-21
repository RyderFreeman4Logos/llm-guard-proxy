//! Endpoint-specific rendering for heterogeneous reranker replicas.

use std::cmp::Ordering;

use axum::{
    body::Bytes,
    http::{
        HeaderMap, HeaderValue, Method, Uri,
        header::{ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_TYPE},
    },
};
use llm_guard_proxy_core::{UpstreamEndpointConfig, UpstreamEndpointProtocol};
use reqwest::Url;
use serde_json::{Value, json};

use super::{
    ProxyError, buffered_adapter::sanitize_transformed_request_headers, deepinfra_rerank_adapter,
    forwarded_request_headers, score_adapter,
};

const MAX_PAIR_COUNT: usize = 1_024;
const MAX_REQUEST_ID_BYTES: usize = 256;
const MAX_DEEPINFRA_RENDERED_BODY_BYTES: usize = 1_048_576;
/// Inbound credentials, sessions, and requester identity must not cross into a configured origin.
const ISOLATED_THIRD_PARTY_HEADERS_TO_STRIP: [&str; 19] = [
    "api-key",
    "authorization",
    "cookie",
    "forwarded",
    "proxy-authorization",
    "set-cookie",
    "signature",
    "signature-input",
    "x-access-token",
    "x-amz-security-token",
    "x-api-key",
    "x-api-token",
    "x-auth-token",
    "x-csrf-token",
    "x-forwarded-for",
    "x-goog-api-key",
    "x-real-ip",
    "x-session-token",
    "x-virtual-key",
];
const DEEPINFRA_CONVERTIBLE_RERANK_FIELDS: [&str; 5] =
    ["model", "query", "documents", "top_n", "return_documents"];

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
#[derive(Clone, Debug)]
pub(super) enum ResponseContract {
    /// Preserve an `OpenAI` `/v1/rerank` response locally, or create one from `DeepInfra`.
    OpenAiRerank {
        documents: Vec<String>,
        top_n: usize,
        return_documents: bool,
        model: Option<String>,
    },
    /// Preserve the local score-adapter response, or create it from `DeepInfra`.
    Score {
        document_count: usize,
        top_n: usize,
        expected: Option<score_adapter::ScoreExpectations>,
        documents: Vec<String>,
        return_documents: bool,
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
        CanonicalRerankerRequest::OpenAiRerank { forward_uri, body } => {
            let input = parse_deepinfra_convertible_rerank(forward_uri, body)?;
            Ok(ResponseContract::OpenAiRerank {
                documents: input.documents,
                top_n: input.top_n,
                return_documents: input.return_documents,
                model: input.model,
            })
        }
        CanonicalRerankerRequest::Score { forward_uri, body } => {
            let input = parse_deepinfra_convertible_rerank(forward_uri, body)?;
            Ok(ResponseContract::Score {
                document_count: input.documents.len(),
                top_n: input.top_n,
                expected: score_adapter::score_expectations_from_rerank_body(body),
                documents: input.documents,
                return_documents: input.return_documents,
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

/// Whether an endpoint can receive this request without changing its public protocol semantics.
/// `OpenAI` endpoints deliberately remain eligible for opaque/future request shapes.
pub(super) fn is_compatible_with_endpoint(
    endpoint: &UpstreamEndpointConfig,
    request: Option<&CanonicalRerankerRequest>,
    request_headers: Option<&HeaderMap>,
) -> bool {
    match endpoint.protocol {
        UpstreamEndpointProtocol::OpenAi => true,
        UpstreamEndpointProtocol::DeepInfraQwen3Rerank => {
            request.is_some_and(|request| response_contract(request).is_ok())
                && deepinfra_inference_uri(endpoint).is_ok()
                && !request_headers.is_some_and(has_integrity_headers)
        }
    }
}

/// Runtime eligibility is intentionally credential-presence-only: `DeepInfra` inference must
/// never be sent as a health probe.
pub(super) fn has_runtime_credential(endpoint: &UpstreamEndpointConfig) -> bool {
    match endpoint.protocol {
        UpstreamEndpointProtocol::OpenAi => optional_authorization_header(endpoint).is_ok(),
        UpstreamEndpointProtocol::DeepInfraQwen3Rerank => {
            required_authorization_header(endpoint).is_ok()
        }
    }
}

/// Rewrite a validated `DeepInfra` response into the public contract.
///
/// Callers must select this function from the terminal endpoint protocol; response-body shape is
/// untrusted data and is never a protocol discriminator.
pub(super) fn rewrite_response(
    body: &Bytes,
    contract: ResponseContract,
    model_id: Option<&str>,
) -> Result<Bytes, String> {
    let response = parse_deepinfra_response(body)?;
    match contract {
        ResponseContract::OpenAiRerank {
            documents,
            top_n,
            return_documents,
            model,
        } => {
            ensure_exact_score_count(&response, documents.len())?;
            openai_rerank_response(
                &response,
                top_n,
                &documents,
                return_documents,
                model.as_deref().or(model_id),
            )
        }
        ResponseContract::Score {
            document_count,
            top_n,
            expected,
            documents,
            return_documents,
        } => {
            ensure_exact_score_count(&response, document_count)?;
            score_response(
                &response,
                top_n,
                expected,
                model_id,
                &documents,
                return_documents,
            )
        }
        ResponseContract::DeepInfraNative { expected_count } => {
            ensure_exact_score_count(&response, expected_count)?;
            deepinfra_native_response(&response)
        }
    }
}

/// Rewrite a buffered response according to the endpoint that actually produced it.
pub(super) fn rewrite_response_for_endpoint(
    body: &Bytes,
    request: &CanonicalRerankerRequest,
    endpoint_protocol: UpstreamEndpointProtocol,
    model_id: Option<&str>,
) -> Result<Bytes, String> {
    match endpoint_protocol {
        UpstreamEndpointProtocol::DeepInfraQwen3Rerank => rewrite_response(
            body,
            response_contract(request).map_err(|error| error.to_string())?,
            model_id,
        ),
        UpstreamEndpointProtocol::OpenAi => match request {
            CanonicalRerankerRequest::OpenAiRerank { .. } => Ok(body.clone()),
            CanonicalRerankerRequest::Score {
                body: request_body, ..
            } => score_adapter::rerank_response_to_score_response(
                body,
                model_id,
                score_adapter::score_expectations_from_rerank_body(request_body),
            ),
            CanonicalRerankerRequest::DeepInfraNative {
                uri,
                body: request_body,
            } => {
                let adapted = deepinfra_rerank_adapter::adapt_request(uri, request_body)
                    .map_err(|error| error.to_string())?;
                deepinfra_rerank_adapter::score_response_to_deepinfra_response(
                    body,
                    adapted.response_expectations,
                )
            }
            CanonicalRerankerRequest::UnsupportedScore => Err(String::from(
                "selected OpenAI endpoint cannot normalize this score request",
            )),
        },
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
    let headers = request_headers_for_openai(request, downstream_headers);
    render_openai_endpoint(endpoint, uri, &body, &headers, false)
}

/// Render one generic OpenAI-compatible endpoint, applying only endpoint-configured
/// credential isolation and model alias translation.
pub(super) fn render_openai_endpoint(
    endpoint: &UpstreamEndpointConfig,
    uri: Uri,
    body: &Bytes,
    downstream_headers: &HeaderMap,
    transformed_request_headers: bool,
) -> Result<RenderedEndpointRequest, ProxyError> {
    let authorization = optional_authorization_header(endpoint)?;
    let headers = if let Some(authorization) = authorization {
        isolated_third_party_headers(downstream_headers, authorization)
    } else if transformed_request_headers {
        sanitize_transformed_request_headers(downstream_headers)
    } else {
        downstream_headers.clone()
    };
    let body = rewrite_openai_model(endpoint, body, downstream_headers)?;
    Ok(RenderedEndpointRequest {
        url: super::build_upstream_url(&endpoint.base_url, &uri)?,
        uri,
        body,
        headers,
    })
}

fn rewrite_openai_model(
    endpoint: &UpstreamEndpointConfig,
    body: &Bytes,
    downstream_headers: &HeaderMap,
) -> Result<Bytes, ProxyError> {
    let Some(model) = endpoint.model.as_deref() else {
        return Ok(body.clone());
    };
    if body.is_empty() {
        return Ok(body.clone());
    }
    if endpoint.api_key_env.is_none() && has_integrity_headers(downstream_headers) {
        return Err(ProxyError::ContextBudgetExceeded {
            message: String::from(
                "signed or integrity-protected requests cannot use an endpoint model override",
            ),
            param: "headers",
            code: "signed_request_transformation_unsupported",
            request_metadata: None,
            attempts: Vec::new(),
        });
    }
    let mut value: Value = serde_json::from_slice(body).map_err(|error| {
        invalid_request_error(&format!(
            "OpenAI endpoint model override requires JSON: {error}"
        ))
    })?;
    let object = value.as_object_mut().ok_or_else(|| {
        invalid_request_error("OpenAI endpoint model override requires a JSON object")
    })?;
    object.insert(String::from("model"), Value::String(model.to_owned()));
    serde_json::to_vec(&value)
        .map(Bytes::from)
        .map_err(|error| {
            invalid_request_error(&format!("serialize OpenAI request failed: {error}"))
        })
}

fn isolated_third_party_headers(
    downstream_headers: &HeaderMap,
    authorization: HeaderValue,
) -> HeaderMap {
    let mut headers =
        sanitize_transformed_request_headers(&forwarded_request_headers(downstream_headers));
    for name in ISOLATED_THIRD_PARTY_HEADERS_TO_STRIP {
        headers.remove(name);
    }
    headers.insert(AUTHORIZATION, authorization);
    headers
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
    let authorization = required_authorization_header(endpoint)?;
    render_deepinfra_with_authorization(endpoint, request, downstream_headers, authorization)
}

fn render_deepinfra_with_authorization(
    endpoint: &UpstreamEndpointConfig,
    request: &CanonicalRerankerRequest,
    downstream_headers: &HeaderMap,
    authorization: HeaderValue,
) -> Result<RenderedEndpointRequest, ProxyError> {
    ensure_deepinfra_transformation_is_unsigned(downstream_headers)?;
    let (body, uri) = match request {
        CanonicalRerankerRequest::OpenAiRerank { forward_uri, body }
        | CanonicalRerankerRequest::Score { forward_uri, body } => {
            let input = parse_deepinfra_convertible_rerank(forward_uri, body)?;
            ensure_generated_body_fits(&input)?;
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
        CanonicalRerankerRequest::DeepInfraNative { uri, body } => {
            let target = deepinfra_inference_uri(endpoint)?;
            if let Some(query) = uri.query()
                && target.query() != Some(query)
            {
                return Err(invalid_request_error(
                    "DeepInfra native query must match the configured pinned version",
                ));
            }
            if body.len() > MAX_DEEPINFRA_RENDERED_BODY_BYTES {
                return Err(invalid_request_error(
                    "DeepInfra native body exceeds the rendered endpoint body limit",
                ));
            }
            (body.clone(), target)
        }
        CanonicalRerankerRequest::UnsupportedScore => return Err(unsupported_request_error()),
    };
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
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

fn ensure_deepinfra_transformation_is_unsigned(headers: &HeaderMap) -> Result<(), ProxyError> {
    if has_integrity_headers(headers) {
        return Err(ProxyError::ContextBudgetExceeded {
            message: String::from(
                "signed or integrity-protected requests cannot be transformed for DeepInfra",
            ),
            param: "headers",
            code: "signed_request_transformation_unsupported",
            request_metadata: None,
            attempts: Vec::new(),
        });
    }
    Ok(())
}

fn has_integrity_headers(headers: &HeaderMap) -> bool {
    [
        "signature",
        "signature-input",
        "digest",
        "content-digest",
        "repr-digest",
    ]
    .into_iter()
    .any(|header| headers.contains_key(header))
}

fn deepinfra_inference_uri(endpoint: &UpstreamEndpointConfig) -> Result<Uri, ProxyError> {
    let model = endpoint
        .model
        .as_deref()
        .ok_or_else(unsupported_request_error)?;
    let version = endpoint
        .model_revision
        .as_deref()
        .ok_or_else(unsupported_request_error)?;
    format!("/v1/inference/{model}?version={version}")
        .parse()
        .map_err(|error| {
            invalid_request_error(&format!("invalid DeepInfra inference path: {error}"))
        })
}

fn ensure_generated_body_fits(input: &OpenAiRerankInput) -> Result<(), ProxyError> {
    let escaped_query_bytes = input
        .query
        .len()
        .checked_mul(6)
        .ok_or_else(|| invalid_request_error("DeepInfra query size overflow"))?;
    let repeated_queries = escaped_query_bytes
        .checked_mul(input.documents.len())
        .ok_or_else(|| invalid_request_error("DeepInfra query repetition size overflow"))?;
    let escaped_documents = input
        .documents
        .iter()
        .try_fold(0_usize, |total, document| {
            let escaped = document
                .len()
                .checked_mul(6)
                .ok_or_else(|| invalid_request_error("DeepInfra document size overflow"))?;
            total
                .checked_add(escaped)
                .ok_or_else(|| invalid_request_error("DeepInfra document total size overflow"))
        })?;
    let json_overhead = 256_usize
        .checked_add(
            deepinfra_rerank_adapter::DEFAULT_INSTRUCTION
                .len()
                .checked_mul(6)
                .ok_or_else(|| invalid_request_error("DeepInfra instruction size overflow"))?,
        )
        .ok_or_else(|| invalid_request_error("DeepInfra JSON overhead overflow"))?;
    let upper_bound = repeated_queries
        .checked_add(escaped_documents)
        .and_then(|size| size.checked_add(json_overhead))
        .ok_or_else(|| invalid_request_error("DeepInfra rendered body size overflow"))?;
    if upper_bound > MAX_DEEPINFRA_RENDERED_BODY_BYTES {
        return Err(invalid_request_error(
            "DeepInfra rendered body would exceed the endpoint body limit",
        ));
    }
    Ok(())
}

pub(super) fn optional_authorization_header(
    endpoint: &UpstreamEndpointConfig,
) -> Result<Option<HeaderValue>, ProxyError> {
    let Some(variable) = endpoint.api_key_env.as_deref() else {
        return Ok(None);
    };
    let token = std::env::var(variable).map_err(|_error| credential_error())?;
    if token.trim().is_empty() {
        return Err(credential_error());
    }
    HeaderValue::from_str(&format!("Bearer {token}"))
        .map(Some)
        .map_err(|_error| credential_error())
}

fn required_authorization_header(
    endpoint: &UpstreamEndpointConfig,
) -> Result<HeaderValue, ProxyError> {
    optional_authorization_header(endpoint)?.ok_or_else(unsupported_request_error)
}

struct OpenAiRerankInput {
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    top_n: usize,
    return_documents: bool,
}

fn parse_deepinfra_convertible_rerank(
    forward_uri: &Uri,
    body: &Bytes,
) -> Result<OpenAiRerankInput, ProxyError> {
    if forward_uri.query().is_some_and(|query| !query.is_empty()) {
        return Err(invalid_request_error(
            "rerank query parameters cannot be preserved by DeepInfra",
        ));
    }
    parse_openai_rerank(body)
}

fn parse_openai_rerank(body: &Bytes) -> Result<OpenAiRerankInput, ProxyError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| invalid_request_error(&format!("invalid rerank JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid_request_error("rerank body must be a JSON object"))?;
    if object
        .keys()
        .any(|key| !DEEPINFRA_CONVERTIBLE_RERANK_FIELDS.contains(&key.as_str()))
    {
        return Err(invalid_request_error(
            "rerank body contains fields that cannot be preserved by DeepInfra",
        ));
    }
    let model = match object.get("model") {
        None => None,
        Some(Value::String(model)) if !model.trim().is_empty() => Some(model.clone()),
        Some(Value::String(_)) => {
            return Err(invalid_request_error("rerank model must not be empty"));
        }
        Some(_) => return Err(invalid_request_error("rerank model must be a string")),
    };
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
    let return_documents = match object.get("return_documents") {
        None => false,
        Some(Value::Bool(return_documents)) => *return_documents,
        Some(_) => {
            return Err(invalid_request_error(
                "rerank return_documents must be a boolean",
            ));
        }
    };
    Ok(OpenAiRerankInput {
        model,
        query,
        documents,
        top_n,
        return_documents,
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
        None | Some(Value::Null) => None,
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

fn openai_rerank_response(
    response: &DeepInfraResponse,
    top_n: usize,
    documents: &[String],
    return_documents: bool,
    model: Option<&str>,
) -> Result<Bytes, String> {
    let results = selected_scores(&response.scores, top_n)?
        .into_iter()
        .map(|(index, score)| {
            let mut result = serde_json::Map::from_iter([
                (String::from("index"), Value::from(index)),
                (String::from("relevance_score"), Value::from(score)),
            ]);
            if return_documents {
                result.insert(
                    String::from("document"),
                    Value::String(documents[index].clone()),
                );
            }
            Value::Object(result)
        })
        .collect::<Vec<_>>();
    serialize(&json!({
        "id": response.request_id.as_deref().unwrap_or("rerank-adapted"),
        "object": "list",
        "created": unix_time_seconds(),
        "model": model.filter(|model| !model.is_empty()).unwrap_or("qwen3-reranker-8b"),
        "results": results,
        "data": results,
        "usage": {
            "prompt_tokens": response.input_tokens,
            "total_tokens": response.input_tokens,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": null},
        },
    }))
}

fn score_response(
    response: &DeepInfraResponse,
    top_n: usize,
    expected: Option<score_adapter::ScoreExpectations>,
    model_id: Option<&str>,
    documents: &[String],
    return_documents: bool,
) -> Result<Bytes, String> {
    let mut scores = selected_scores(&response.scores, top_n)?;
    scores.sort_by_key(|(index, _score)| *index);
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
        .map(|(index, probability)| {
            let mut result = serde_json::Map::from_iter([
                (String::from("index"), Value::from(index)),
                (String::from("object"), Value::String(String::from("score"))),
                (
                    String::from("score"),
                    Value::from(probability.mul_add(2.0, -1.0)),
                ),
            ]);
            if return_documents {
                result.insert(
                    String::from("document"),
                    Value::String(documents[index].clone()),
                );
            }
            Value::Object(result)
        })
        .collect::<Vec<_>>();
    serialize(&json!({
        "id": "score-adapted",
        "object": "list",
        "created": unix_time_seconds(),
        "model": model,
        "data": data,
        "usage": {
            "prompt_tokens": response.input_tokens,
            "total_tokens": response.input_tokens,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": null},
        },
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
        attempts: Vec::new(),
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
        attempts: Vec::new(),
    }
}

fn credential_error() -> ProxyError {
    ProxyError::ContextBudgetExceeded {
        message: String::from("selected upstream endpoint has no usable runtime credential"),
        param: "api_key_env",
        code: "upstream_endpoint_credential_unavailable",
        request_metadata: None,
        attempts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;
    use llm_guard_proxy_core::{UpstreamEndpointConfig, UpstreamEndpointProtocol};

    const SENTINEL: &str = "LLM_GUARD_PROXY_NOT_A_REAL_CREDENTIAL_195";
    const DEEPINFRA_QWEN3_RERANK_VERSION: &str = "5fa94080caafeaa45a15d11f969d7978e087a3db";

    #[test]
    fn converts_deepinfra_scores_to_indexed_openai_rerank_results() {
        let body = Bytes::from_static(
            br#"{"scores":[0.2,0.9,0.5],"input_tokens":19,"request_id":"request-195"}"#,
        );
        let response = rewrite_response(
            &body,
            ResponseContract::OpenAiRerank {
                documents: vec![
                    String::from("one"),
                    String::from("two"),
                    String::from("three"),
                ],
                top_n: 2,
                return_documents: false,
                model: Some(String::from("qwen3-reranker-8b")),
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
    fn deepinfra_compatibility_rejects_unknown_keys_and_nonempty_query() {
        let deepinfra = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
            ..UpstreamEndpointConfig::default()
        };
        let openai = UpstreamEndpointConfig::default();
        let requests = [
            CanonicalRerankerRequest::OpenAiRerank {
                forward_uri: Uri::from_static("/v1/rerank"),
                body: Bytes::from_static(
                    br#"{"query":"q","documents":["d"],"instruction":"custom"}"#,
                ),
            },
            CanonicalRerankerRequest::Score {
                forward_uri: Uri::from_static("/v1/rerank?semantic=preserve"),
                body: Bytes::from_static(br#"{"query":"q","documents":["d"]}"#),
            },
        ];

        for request in &requests {
            assert!(!is_compatible_with_endpoint(
                &deepinfra,
                Some(request),
                None
            ));
            assert!(is_compatible_with_endpoint(&openai, Some(request), None));
            assert!(
                render_deepinfra_with_authorization(
                    &deepinfra,
                    request,
                    &HeaderMap::new(),
                    HeaderValue::from_static("Bearer provider-secret"),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn deepinfra_renderer_replaces_downstream_authorization() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
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
            rendered.uri.query(),
            Some("version=5fa94080caafeaa45a15d11f969d7978e087a3db")
        );
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

    #[test]
    fn accepts_nullable_deepinfra_request_id() {
        let body = Bytes::from_static(br#"{"scores":[0.25],"input_tokens":19,"request_id":null}"#);
        let response = rewrite_response(
            &body,
            ResponseContract::DeepInfraNative { expected_count: 1 },
            None,
        )
        .expect("provider-valid null request_id should be accepted");
        let response: Value = serde_json::from_slice(&response).expect("response should be JSON");
        assert!(response.get("request_id").is_none());
    }

    #[test]
    fn deepinfra_decoder_rejects_malformed_scores_and_bounded_metadata() {
        let invalid_cases = [
            (
                Bytes::from_static(br#"{"input_tokens":1}"#),
                "scores must be an array",
            ),
            (
                Bytes::from_static(br#"{"scores":[0.2],"input_tokens":1}"#),
                "does not match expected",
            ),
            (
                Bytes::from_static(br#"{"scores":[0.2,1.5],"input_tokens":1}"#),
                "finite value in [0, 1]",
            ),
            (
                Bytes::from_static(br#"{"scores":[0.2,"NaN"],"input_tokens":1}"#),
                "finite value in [0, 1]",
            ),
            (
                Bytes::from(format!(
                    r#"{{"scores":[0.2,0.9],"input_tokens":1,"request_id":"{}"}}"#,
                    "x".repeat(MAX_REQUEST_ID_BYTES + 1)
                )),
                "bounded non-empty string",
            ),
        ];

        for (body, expected_error) in invalid_cases {
            let error = rewrite_response(
                &body,
                ResponseContract::DeepInfraNative { expected_count: 2 },
                None,
            )
            .expect_err("malformed DeepInfra response must be rejected");
            assert!(error.contains(expected_error), "unexpected error: {error}");
        }
    }

    #[test]
    fn converts_deepinfra_probability_to_signed_score() {
        let body = Bytes::from_static(br#"{"scores":[0.9],"input_tokens":19}"#);
        let response = rewrite_response(
            &body,
            ResponseContract::Score {
                document_count: 1,
                top_n: 1,
                expected: Some(score_adapter::ScoreExpectations {
                    result_count: 1,
                    document_count: 1,
                }),
                documents: vec![String::from("d")],
                return_documents: false,
            },
            Some("qwen3-reranker-8b"),
        )
        .expect("valid DeepInfra response should convert");
        let response: Value = serde_json::from_slice(&response).expect("response should be JSON");
        assert_eq!(response["data"][0]["score"], 0.8);
    }

    #[test]
    fn deepinfra_renderer_drops_all_downstream_credentials_and_proxy_headers() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
            ..UpstreamEndpointConfig::default()
        };
        let request = CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: Uri::from_static("/v1/rerank"),
            body: Bytes::from_static(br#"{"query":"q","documents":["d"]}"#),
        };
        let mut headers = HeaderMap::new();
        for name in [
            "authorization",
            "x-api-key",
            "x-virtual-key",
            "cookie",
            "proxy-authorization",
            "x-forwarded-for",
        ] {
            headers.insert(
                HeaderName::from_static(name),
                HeaderValue::from_static("downstream-secret"),
            );
        }
        let rendered = render_deepinfra_with_authorization(
            &endpoint,
            &request,
            &headers,
            HeaderValue::from_static("Bearer provider-secret"),
        )
        .expect("request should render");
        for name in [
            "x-api-key",
            "x-virtual-key",
            "cookie",
            "proxy-authorization",
            "x-forwarded-for",
        ] {
            assert!(
                !rendered.headers.contains_key(name),
                "{name} must not cross the DeepInfra trust boundary"
            );
        }
        assert_eq!(
            rendered.headers.get(AUTHORIZATION),
            Some(&HeaderValue::from_static("Bearer provider-secret"))
        );
    }

    #[test]
    fn deepinfra_renderer_rejects_signed_transformation() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
            ..UpstreamEndpointConfig::default()
        };
        let request = CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: Uri::from_static("/v1/rerank"),
            body: Bytes::from_static(br#"{"query":"q","documents":["d"]}"#),
        };
        let mut headers = HeaderMap::new();
        headers.insert("signature", HeaderValue::from_static("sig"));
        let result = render_deepinfra_with_authorization(
            &endpoint,
            &request,
            &headers,
            HeaderValue::from_static("Bearer provider-secret"),
        );
        let Err(error) = result else {
            panic!("a transformed signed request must fail closed");
        };
        assert!(error.to_string().contains("integrity-protected"));
    }

    #[test]
    fn deepinfra_renderer_retains_only_the_exact_pinned_native_version_query() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
            ..UpstreamEndpointConfig::default()
        };
        let request = CanonicalRerankerRequest::DeepInfraNative {
            uri: Uri::from_static(
                "/v1/inference/Qwen/Qwen3-Reranker-8B?version=5fa94080caafeaa45a15d11f969d7978e087a3db",
            ),
            body: Bytes::from_static(br#"{"queries":["q"],"documents":["d"]}"#),
        };
        let rendered = render_deepinfra_with_authorization(
            &endpoint,
            &request,
            &HeaderMap::new(),
            HeaderValue::from_static("Bearer provider-secret"),
        )
        .expect("matching native version should render");
        assert_eq!(
            rendered.uri.query(),
            Some("version=5fa94080caafeaa45a15d11f969d7978e087a3db")
        );

        let mismatched = CanonicalRerankerRequest::DeepInfraNative {
            uri: Uri::from_static("/v1/inference/Qwen/Qwen3-Reranker-8B?version=other"),
            body: Bytes::from_static(br#"{"queries":["q"],"documents":["d"]}"#),
        };
        assert!(
            render_deepinfra_with_authorization(
                &endpoint,
                &mismatched,
                &HeaderMap::new(),
                HeaderValue::from_static("Bearer provider-secret"),
            )
            .is_err()
        );
    }

    #[test]
    fn deepinfra_rerank_envelope_preserves_model_aliases_and_returned_documents() {
        let body =
            Bytes::from_static(br#"{"scores":[0.2,0.9],"input_tokens":19,"request_id":"req-195"}"#);
        let response = rewrite_response(
            &body,
            ResponseContract::OpenAiRerank {
                documents: vec![String::from("low"), String::from("high")],
                top_n: 2,
                return_documents: true,
                model: Some(String::from("public-reranker")),
            },
            None,
        )
        .expect("valid DeepInfra response should convert");
        let response: Value = serde_json::from_slice(&response).expect("response should be JSON");
        assert_eq!(response["id"], "req-195");
        assert_eq!(response["model"], "public-reranker");
        assert_eq!(response["results"], response["data"]);
        assert_eq!(response["results"][0]["index"], 1);
        assert_eq!(response["results"][0]["document"], "high");
        assert_eq!(response["usage"]["total_tokens"], 19);
    }

    #[test]
    fn oversized_deepinfra_rendering_is_rejected_before_query_repetition() {
        let endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from(DEEPINFRA_QWEN3_RERANK_VERSION)),
            ..UpstreamEndpointConfig::default()
        };
        let documents = std::iter::repeat_n("d", MAX_PAIR_COUNT).collect::<Vec<_>>();
        let body = serde_json::to_vec(&json!({
            "query": "q".repeat(1024),
            "documents": documents,
        }))
        .expect("test request JSON");
        let request = CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: Uri::from_static("/v1/rerank"),
            body: Bytes::from(body),
        };
        assert!(
            render_deepinfra_with_authorization(
                &endpoint,
                &request,
                &HeaderMap::new(),
                HeaderValue::from_static("Bearer provider-secret"),
            )
            .is_err()
        );
    }

    #[test]
    fn terminal_protocol_not_response_shape_selects_the_converter() {
        let request = CanonicalRerankerRequest::OpenAiRerank {
            forward_uri: Uri::from_static("/v1/rerank"),
            body: Bytes::from_static(
                br#"{"model":"public-reranker","query":"q","documents":["d"]}"#,
            ),
        };
        let body = Bytes::from_static(br#"{"scores":"not-an-array"}"#);
        assert_eq!(
            rewrite_response_for_endpoint(
                &body,
                &request,
                UpstreamEndpointProtocol::OpenAi,
                Some("public-reranker"),
            )
            .expect("an OpenAI endpoint response must not be guessed from body shape"),
            body,
        );
        assert!(
            rewrite_response_for_endpoint(
                &body,
                &request,
                UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
                Some("public-reranker"),
            )
            .is_err()
        );
    }
}
