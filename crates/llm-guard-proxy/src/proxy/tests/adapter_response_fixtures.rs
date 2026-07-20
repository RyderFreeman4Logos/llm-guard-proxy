use std::time::Duration;

use axum::{
    body::{Body, Bytes},
    http::{
        HeaderName, HeaderValue, Response, StatusCode,
        header::{CONTENT_TYPE, RETRY_AFTER},
    },
};
use futures_util::StreamExt;

use super::{body_contains_text, json_response};

pub(super) fn fake_deepinfra_score_response(path_and_query: &str) -> Response<Body> {
    if path_and_query.contains("test=deepinfra-rerank-upstream-error") {
        let mut response = json_response(
            "deepinfra-score-error",
            r#"{"error":{"message":"local reranker busy"}}"#.to_owned(),
        );
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from_static("17"));
        response.headers_mut().insert(
            HeaderName::from_static("server"),
            HeaderValue::from_static("private-vllm-score"),
        );
        response.headers_mut().insert(
            HeaderName::from_static("x-upstream-only"),
            HeaderValue::from_static("must-not-leak"),
        );
        return response;
    }
    let body = if path_and_query.contains("test=deepinfra-rerank-malformed") {
        r#"{"id":"score-native-malformed","data":[{"index":0,"score":0.0},{"index":0,"score":0.5},{"index":2,"score":1.0}],"usage":{"prompt_tokens":19}}"#
    } else {
        r#"{"id":"score-native-123","object":"list","model":"qwen3-reranker-8b","data":[{"index":2,"object":"score","score":0.0},{"index":0,"object":"score","score":-1.0},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":19,"total_tokens":19,"completion_tokens":0}}"#
    };
    let mut response = json_response("deepinfra-score", body.to_owned());
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/vnd.vllm.score+json"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("server"),
        HeaderValue::from_static("private-vllm-score"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-request-id"),
        HeaderValue::from_static("private-vllm-request-id"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-upstream-only"),
        HeaderValue::from_static("must-not-leak"),
    );
    response
}

pub(super) fn fake_rerank_response(path_and_query: &str, body: &Bytes) -> Response<Body> {
    if body_contains_text(body, "malformed-openai-failover") {
        return json_response(
            "rerank-malformed",
            r#"{"id":"rerank-malformed","results":[{"index":0,"score":0.9},{"index":0,"score":0.8}]}"#
                .to_owned(),
        );
    }
    if path_and_query.contains("test=score-body-read-error") {
        let stream = futures_util::stream::once(async {
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"{"))
        })
        .chain(futures_util::stream::once(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Err::<Bytes, std::io::Error>(std::io::Error::other(
                "synthetic rerank body read failure",
            ))
        }));
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/vnd.rerank+json"),
        );
        response.headers_mut().insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_static("rerank-body-error-123"),
        );
        return response;
    }
    if path_and_query.contains("test=score-upstream-500") {
        let mut response = json_response("rerank-error", r#"{"error":"upstream boom"}"#.to_owned());
        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from_static("13"));
        response.headers_mut().insert(
            HeaderName::from_static("server"),
            HeaderValue::from_static("private-vllm-rerank"),
        );
        return response;
    }
    if path_and_query.contains("test=score-partial") {
        return json_response(
            "rerank-partial",
            r#"{"id":"rerank-partial","model":"qwen3-reranker-8b","results":[{"index":0,"score":0.9}]}"#
                .to_owned(),
        );
    }
    let doc_count = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            let documents = value.get("documents")?;
            match documents {
                serde_json::Value::Array(items) => Some(items.len()),
                serde_json::Value::Object(object) => object
                    .get("content")
                    .and_then(serde_json::Value::as_array)
                    .map(Vec::len),
                _ => None,
            }
        })
        .unwrap_or(1)
        .min(8);
    let results: Vec<String> = (0..doc_count)
        .map(|i| {
            let score = 1.0 - f64::from(u32::try_from(i).unwrap_or(0)) * 0.1;
            format!(r#"{{"index":{i},"score":{score}}}"#)
        })
        .collect();
    let body = format!(
        r#"{{"id":"rerank-test","model":"qwen3-reranker-8b","results":[{}]}}"#,
        results.join(",")
    );
    let mut response = json_response("rerank", body);
    if path_and_query.contains("test=score-adapter-ok") {
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/vnd.rerank+json"),
        );
        response.headers_mut().insert(
            HeaderName::from_static("server"),
            HeaderValue::from_static("fake-rerank"),
        );
        response.headers_mut().insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_static("rerank-request-123"),
        );
    }
    response
}
