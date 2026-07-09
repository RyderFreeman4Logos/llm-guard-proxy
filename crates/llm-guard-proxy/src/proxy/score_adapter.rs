//! Adapt vLLM-style `POST /v1/score` to upstream `POST /v1/rerank`.
//!
//! Background: healthcheck and some clients use `/v1/score` with `text_1`/`text_2`.
//! Querit-4B (and other transformers rerank servers) only implement `/v1/rerank`.
//! The proxy rewrites request and response so clients keep the score contract.

use axum::{
    body::Bytes,
    http::{Method, Uri},
};
use serde_json::{Value, json};

/// Whether this request is a score endpoint that should be rewritten to rerank.
#[must_use]
pub(crate) fn is_score_request(method: &Method, uri: &Uri) -> bool {
    *method == Method::POST && uri.path() == "/v1/score"
}

/// Rewrite a vLLM `/v1/score` JSON body into a `/v1/rerank` body.
///
/// Supported score shapes:
/// - `text_1: string`, `text_2: string` → single document
/// - `text_1: string`, `text_2: [string, ...]` → multi document
/// - already-rerank shape with `query` + `documents` is passed through (path still rewritten)
pub(crate) fn score_body_to_rerank_body(body: &Bytes) -> Result<Bytes, String> {
    let value: Value =
        serde_json::from_slice(body).map_err(|error| format!("invalid score JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| String::from("score body must be a JSON object"))?;

    // Already a rerank body — keep as-is.
    if object.contains_key("query") && object.contains_key("documents") {
        return Ok(body.clone());
    }

    let model = object.get("model").cloned().unwrap_or(Value::Null);
    let text_1 = object
        .get("text_1")
        .ok_or_else(|| String::from("score body missing text_1"))?;
    let text_2 = object
        .get("text_2")
        .ok_or_else(|| String::from("score body missing text_2"))?;

    let query = text_1
        .as_str()
        .ok_or_else(|| String::from("score text_1 must be a string"))?
        .to_owned();

    let documents: Vec<Value> = match text_2 {
        Value::String(doc) => vec![Value::String(doc.clone())],
        Value::Array(items) => {
            if items.is_empty() {
                return Err(String::from("score text_2 array must be non-empty"));
            }
            for item in items {
                if !item.is_string() {
                    return Err(String::from("score text_2 array items must be strings"));
                }
            }
            items.clone()
        }
        _ => {
            return Err(String::from(
                "score text_2 must be a string or array of strings",
            ));
        }
    };

    let mut out = serde_json::Map::new();
    if !model.is_null() {
        out.insert(String::from("model"), model);
    }
    out.insert(String::from("query"), Value::String(query));
    out.insert(String::from("documents"), Value::Array(documents));
    // Preserve original order for score mapping (index aligns with documents).
    out.insert(String::from("top_n"), json!(documents_len_from_out(&out)));

    serde_json::to_vec(&Value::Object(out))
        .map(Bytes::from)
        .map_err(|error| format!("serialize rerank body failed: {error}"))
}

fn documents_len_from_out(out: &serde_json::Map<String, Value>) -> usize {
    out.get("documents")
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

/// Rewrite a `/v1/rerank` response into a vLLM-compatible `/v1/score` response.
///
/// Expected score shape for healthcheck:
/// `{ "data": [ { "index": 0, "object": "score", "score": <f64> }, ... ] }`
pub(crate) fn rerank_response_to_score_response(
    body: &Bytes,
    model: Option<&str>,
) -> Result<Bytes, String> {
    let value: Value =
        serde_json::from_slice(body).map_err(|error| format!("invalid rerank JSON: {error}"))?;

    // Collect (index, score) pairs from common shapes.
    let mut scored: Vec<(usize, f64)> = Vec::new();
    if let Some(items) = value
        .get("results")
        .and_then(Value::as_array)
        .or_else(|| value.get("data").and_then(Value::as_array))
    {
        for item in items {
            let index = item
                .get("index")
                .or_else(|| item.get("document_index"))
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());
            let score = item
                .get("score")
                .or_else(|| item.get("relevance_score"))
                .or_else(|| item.get("rerank_score"))
                .and_then(Value::as_f64);
            if let (Some(index), Some(score)) = (index, score) {
                scored.push((index, score));
            }
        }
    }

    if scored.is_empty() {
        return Err(String::from(
            "rerank response missing results/data scores for score adapter",
        ));
    }

    // Stable score list ordered by original document index.
    scored.sort_by_key(|(index, _)| *index);
    let data: Vec<Value> = scored
        .into_iter()
        .map(|(index, score)| {
            json!({
                "index": index,
                "object": "score",
                "score": score,
            })
        })
        .collect();

    let model_value = value
        .get("model")
        .cloned()
        .unwrap_or_else(|| model.map_or(Value::Null, |m| Value::String(m.to_owned())));

    let out = json!({
        "id": value.get("id").cloned().unwrap_or_else(|| Value::String(String::from("score-adapted"))),
        "object": "list",
        "created": value.get("created").cloned().unwrap_or(Value::Null),
        "model": model_value,
        "data": data,
    });

    serde_json::to_vec(&out)
        .map(Bytes::from)
        .map_err(|error| format!("serialize score response failed: {error}"))
}

/// Rewrite request URI path `/v1/score` → `/v1/rerank` while keeping query string.
pub(crate) fn score_uri_to_rerank_uri(uri: &Uri) -> Result<Uri, String> {
    let path = uri.path();
    if path != "/v1/score" {
        return Ok(uri.clone());
    }
    let rewritten = match uri.query() {
        Some(query) => format!("/v1/rerank?{query}"),
        None => String::from("/v1/rerank"),
    };
    rewritten
        .parse::<Uri>()
        .map_err(|error| format!("rewrite score uri failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Uri;

    #[test]
    fn converts_single_pair_score_body() {
        let body =
            Bytes::from_static(br#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#);
        let out = score_body_to_rerank_body(&body).expect("convert");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["query"], "q");
        assert_eq!(v["documents"], json!(["d"]));
        assert_eq!(v["model"], "qwen3-reranker-8b");
        assert_eq!(v["top_n"], 1);
    }

    #[test]
    fn converts_multi_doc_score_body() {
        let body = Bytes::from_static(br#"{"model":"m","text_1":"q","text_2":["a","b"]}"#);
        let out = score_body_to_rerank_body(&body).expect("convert");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["documents"], json!(["a", "b"]));
        assert_eq!(v["top_n"], 2);
    }

    #[test]
    fn converts_rerank_response_to_score() {
        let body = Bytes::from_static(
            br#"{"id":"rerank-1","model":"m","results":[{"index":1,"score":0.9},{"index":0,"score":0.1}]}"#,
        );
        let out = rerank_response_to_score_response(&body, Some("m")).expect("convert");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["index"], 0);
        assert_eq!(v["data"][0]["score"], 0.1);
        assert_eq!(v["data"][0]["object"], "score");
        assert_eq!(v["data"][1]["index"], 1);
        assert_eq!(v["data"][1]["score"], 0.9);
    }

    #[test]
    fn rewrites_uri_path() {
        let uri: Uri = "/v1/score".parse().unwrap();
        let out = score_uri_to_rerank_uri(&uri).unwrap();
        assert_eq!(out.path(), "/v1/rerank");
    }
}
