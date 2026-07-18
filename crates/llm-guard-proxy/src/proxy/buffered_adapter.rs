//! Request and response orchestration for buffered OpenAI-compatible adapters.

use std::collections::BTreeMap;

use axum::{
    body::{Body, Bytes},
    http::{
        HeaderMap, HeaderName, HeaderValue, Method, Response, Uri,
        header::{ACCEPT_ENCODING, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
    },
};
use llm_guard_proxy_state::RawPayloads;

use super::{
    ForwardedResponseParts, InFlightPermit, ObservedBufferedBody, ProxyError,
    deepinfra_rerank_adapter, downstream_mode_from_headers, downstream_response,
    forwarded_request_headers, read_upstream_body_bytes_until_shutdown, reranker_protocol,
    score_adapter,
};

#[derive(Clone, Copy, Debug)]
pub(super) enum BufferedResponseAdapter {
    ScoreFromRerank(Option<score_adapter::ScoreExpectations>),
    DeepInfraQwen3Rerank(deepinfra_rerank_adapter::ResponseExpectations),
    HeterogeneousReranker(reranker_protocol::ResponseContract),
}

impl BufferedResponseAdapter {
    fn rewrite(self, body: &Bytes, model_id: Option<&str>) -> Result<Bytes, String> {
        match self {
            Self::ScoreFromRerank(expected) => {
                score_adapter::rerank_response_to_score_response(body, model_id, expected)
            }
            Self::DeepInfraQwen3Rerank(expected) => {
                deepinfra_rerank_adapter::score_response_to_deepinfra_response(body, expected)
            }
            Self::HeterogeneousReranker(contract) => {
                reranker_protocol::rewrite_response(body, contract, model_id)
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ScoreFromRerank(_) => "score from rerank",
            Self::DeepInfraQwen3Rerank(_) => "DeepInfra rerank from score",
            Self::HeterogeneousReranker(_) => "heterogeneous reranker response",
        }
    }
}

struct AdaptedScoreRequest {
    forward_uri: Uri,
    adapted_body: Bytes,
    score_via_rerank: bool,
    score_expected_count: Option<score_adapter::ScoreExpectations>,
}

pub(super) struct AdaptedOpenAiRequest {
    pub(super) forward_uri: Uri,
    pub(super) adapted_body: Bytes,
    pub(super) response_adapter: Option<BufferedResponseAdapter>,
}

pub(super) fn adapt_openai_request_if_needed(
    method: &Method,
    uri: &Uri,
    downstream_headers: &HeaderMap,
    body: &Bytes,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<AdaptedOpenAiRequest, ProxyError> {
    if deepinfra_rerank_adapter::is_request(method, uri) {
        request_metadata.insert(
            String::from("deepinfra_rerank_adapter"),
            String::from("true"),
        );
        ensure_transform_headers_supported(downstream_headers, request_metadata)?;
        let adapted = deepinfra_rerank_adapter::adapt_request(uri, body).map_err(|error| {
            let code = error.code();
            ProxyError::ContextBudgetExceeded {
                message: format!("invalid DeepInfra rerank request: {error}"),
                param: "body",
                code,
                request_metadata: None,
            }
        })?;
        request_metadata.insert(
            String::from("deepinfra_expected_count"),
            adapted.response_expectations.result_count.to_string(),
        );
        request_metadata.insert(
            String::from("deepinfra_service_tier"),
            adapted.service_tier.as_str().to_owned(),
        );
        request_metadata.insert(
            String::from("deepinfra_service_tier_local_behavior"),
            String::from("single_tier"),
        );
        return Ok(AdaptedOpenAiRequest {
            forward_uri: adapted.forward_uri,
            adapted_body: adapted.body,
            response_adapter: Some(BufferedResponseAdapter::DeepInfraQwen3Rerank(
                adapted.response_expectations,
            )),
        });
    }

    let adapted =
        adapt_score_request_if_needed(method, uri, downstream_headers, body, request_metadata)?;
    let response_adapter =
        adapted
            .score_via_rerank
            .then_some(BufferedResponseAdapter::ScoreFromRerank(
                adapted.score_expected_count,
            ));
    Ok(AdaptedOpenAiRequest {
        forward_uri: adapted.forward_uri,
        adapted_body: adapted.adapted_body,
        response_adapter,
    })
}

fn adapt_score_request_if_needed(
    method: &Method,
    uri: &Uri,
    downstream_headers: &HeaderMap,
    body: &Bytes,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<AdaptedScoreRequest, ProxyError> {
    if !score_adapter::is_score_request(method, uri) {
        return Ok(AdaptedScoreRequest {
            forward_uri: uri.clone(),
            adapted_body: body.clone(),
            score_via_rerank: false,
            score_expected_count: None,
        });
    }
    let invalid = |error: String| ProxyError::ContextBudgetExceeded {
        message: format!("invalid score request: {error}"),
        param: "body",
        code: "invalid_score_request",
        request_metadata: None,
    };
    let adapt = score_adapter::can_adapt_score_body_to_rerank(body).map_err(invalid)?;
    if !adapt {
        request_metadata.insert(String::from("score_via_rerank"), String::from("false"));
        request_metadata.insert(String::from("score_passthrough"), String::from("true"));
        return Ok(AdaptedScoreRequest {
            forward_uri: uri.clone(),
            adapted_body: body.clone(),
            score_via_rerank: false,
            score_expected_count: None,
        });
    }
    ensure_transform_headers_supported(downstream_headers, request_metadata)?;
    let adapted_body = score_adapter::score_body_to_rerank_body(body).map_err(invalid)?;
    let forward_uri = score_adapter::score_uri_to_rerank_uri(uri).map_err(|error| {
        ProxyError::ContextBudgetExceeded {
            message: format!("invalid score request uri: {error}"),
            param: "path",
            code: "invalid_score_request",
            request_metadata: None,
        }
    })?;
    let score_expected_count = score_adapter::score_expectations_from_rerank_body(&adapted_body);
    request_metadata.insert(String::from("score_via_rerank"), String::from("true"));
    if let Some(expected) = score_expected_count {
        request_metadata.insert(
            String::from("score_expected_count"),
            expected.result_count.to_string(),
        );
        request_metadata.insert(
            String::from("score_document_count"),
            expected.document_count.to_string(),
        );
    }
    Ok(AdaptedScoreRequest {
        forward_uri,
        adapted_body,
        score_via_rerank: true,
        score_expected_count,
    })
}

fn ensure_transform_headers_supported(
    downstream_headers: &HeaderMap,
    request_metadata: &mut BTreeMap<String, String>,
) -> Result<(), ProxyError> {
    if downstream_headers.contains_key("signature")
        || downstream_headers.contains_key("signature-input")
        || !score_transform_authorization_supported(downstream_headers)
    {
        request_metadata.insert(
            String::from("signed_request_transformation_rejected"),
            String::from("true"),
        );
        return Err(ProxyError::ContextBudgetExceeded {
            message: String::from(
                "signed requests cannot be transformed without invalidating the signature",
            ),
            param: "headers",
            code: "signed_request_transformation_unsupported",
            request_metadata: None,
        });
    }
    Ok(())
}

fn score_transform_authorization_supported(headers: &HeaderMap) -> bool {
    headers
        .get_all(AUTHORIZATION)
        .iter()
        .all(safe_proxy_authorization_value)
}

fn safe_proxy_authorization_value(value: &HeaderValue) -> bool {
    let Ok(value) = value.to_str() else {
        return false;
    };
    let trimmed = value.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let Some(scheme) = parts.next() else {
        return false;
    };
    let Some(credentials) = parts.next().map(str::trim) else {
        return false;
    };
    if credentials.is_empty()
        || !credentials.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'+' | b'/' | b'=')
        })
    {
        return false;
    }
    scheme.eq_ignore_ascii_case("bearer") || scheme.eq_ignore_ascii_case("basic")
}

pub(super) fn sanitize_transformed_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = forwarded_request_headers(headers);
    forwarded.remove(ACCEPT_ENCODING);
    forwarded.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    // Body changed; drop representation-integrity headers bound to original payload.
    for name in [
        "content-encoding",
        "content-md5",
        "digest",
        "content-digest",
        "repr-digest",
        "etag",
        "if-match",
        "if-none-match",
    ] {
        if let Ok(header_name) = HeaderName::try_from(name) {
            forwarded.remove(header_name);
        }
    }
    forwarded
}

pub(super) async fn rewrite_buffered_adapter_response_from_upstream(
    response_parts: ForwardedResponseParts,
    upstream_response: reqwest::Response,
    in_flight_permit: InFlightPermit,
    adapter: BufferedResponseAdapter,
    model_id: Option<&str>,
) -> Result<Response<Body>, ProxyError> {
    let upstream_status = response_parts.upstream_status;
    let body = match read_upstream_body_bytes_until_shutdown(
        upstream_response.bytes_stream(),
        response_parts.shutdown_subscription(),
    )
    .await
    {
        Ok(body) => body,
        Err(error) => {
            return Err(response_parts.into_response_process_error(
                error,
                BTreeMap::from([(
                    String::from("response_body_read_error"),
                    String::from("true"),
                )]),
            ));
        }
    };
    let upstream_body_bytes = body.len();
    let (body, response_headers) = if upstream_status.is_success() {
        match adapter.rewrite(&body, model_id) {
            Ok(body) => {
                // Transformed body: clean downstream headers only; keep original
                // upstream headers on response_parts for attempt observability.
                let headers = transformed_json_response_headers();
                (body, headers)
            }
            Err(error) => {
                return Err(response_parts.into_response_process_error(
                    ProxyError::upstream_body(format!(
                        "{} response rewrite failed: {error}",
                        adapter.label()
                    )),
                    BTreeMap::from([(
                        String::from("response_body_bytes"),
                        upstream_body_bytes.to_string(),
                    )]),
                ));
            }
        }
    } else {
        (
            body,
            transformed_error_response_headers(&response_parts.upstream_headers),
        )
    };
    let stream_cancel = response_parts.shutdown_subscription();
    let observer = response_parts.into_observer_with(
        downstream_mode_from_headers(&response_headers),
        response_headers.clone(),
        BTreeMap::from([(
            String::from("response_body_bytes"),
            upstream_body_bytes.to_string(),
        )]),
        BTreeMap::new(),
        RawPayloads::default(),
    );
    let response_body = ObservedBufferedBody::new(body, observer, in_flight_permit, stream_cancel);
    Ok(downstream_response(
        upstream_status,
        &response_headers,
        Body::from_stream(response_body),
    ))
}

fn transformed_json_response_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers
}

fn transformed_error_response_headers(upstream_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for name in [CONTENT_TYPE, RETRY_AFTER] {
        if let Some(value) = upstream_headers.get(&name) {
            headers.insert(name, value.clone());
        }
    }
    headers
}
