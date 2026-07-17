use super::*;
use serde_json::{Value, json};

fn request_body(extra: &str) -> Bytes {
    Bytes::from(format!(
        r#"{{"queries":["q1","q2"],"documents":["d1","d2"]{extra}}}"#
    ))
}

fn adapt(body: &Bytes) -> Result<AdaptedRequest, RequestError> {
    adapt_request(&Uri::from_static(INFERENCE_PATH), body)
}

fn score_body(data: &str, usage: &str, id: &str) -> Bytes {
    Bytes::from(format!(r#"{{"data":{data},"usage":{usage}{id}}}"#))
}

fn response_json(body: &Bytes, expected_count: usize) -> Result<Value, String> {
    score_response_to_deepinfra_response(
        body,
        ResponseExpectations {
            result_count: expected_count,
        },
    )
    .and_then(|body| serde_json::from_slice(&body).map_err(|error| error.to_string()))
}

#[test]
fn recognizes_only_the_exact_post_model_path() {
    let exact: Uri = INFERENCE_PATH.parse().expect("exact path");
    let with_query: Uri = format!("{INFERENCE_PATH}?client=deepinfra")
        .parse()
        .expect("path with query");
    let wrong_case: Uri = "/v1/inference/qwen/Qwen3-Reranker-8B"
        .parse()
        .expect("wrong-case path");

    assert!(is_request(&Method::POST, &exact));
    assert!(is_request(&Method::POST, &with_query));
    assert!(!is_request(&Method::GET, &exact));
    assert!(!is_request(&Method::POST, &wrong_case));
    assert_eq!(model_id_from_path(&Method::POST, &exact), Some(MODEL_ID));
    assert_eq!(model_id_from_path(&Method::GET, &exact), None);
}

#[test]
fn rewrites_pairs_as_one_vllm_nn_score_batch() {
    let uri: Uri = format!("{INFERENCE_PATH}?client=deepinfra")
        .parse()
        .expect("request URI");
    let adapted = adapt_request(&uri, &request_body("")).expect("valid request should adapt");
    let value: Value = serde_json::from_slice(&adapted.body).expect("score JSON");

    assert_eq!(adapted.forward_uri.path(), "/v1/score");
    assert_eq!(adapted.forward_uri.query(), Some("client=deepinfra"));
    assert_eq!(value["model"], UPSTREAM_MODEL_ID);
    assert_eq!(value["text_1"], json!(["q1", "q2"]));
    assert_eq!(value["text_2"], json!(["d1", "d2"]));
    assert!(value.get("query").is_none());
    assert!(value.get("documents").is_none());
    assert_eq!(adapted.response_expectations.result_count, 2);
}

#[test]
fn accepts_absent_or_explicit_default_instruction() {
    let absent = adapt(&request_body("")).expect("absent instruction uses default");
    let explicit = adapt(&request_body(&format!(
        r#","instruction":{}"#,
        serde_json::to_string(DEFAULT_INSTRUCTION).expect("serialize default instruction")
    )))
    .expect("explicit default instruction should adapt");

    assert_eq!(absent.body, explicit.body);
}

#[test]
fn rejects_custom_instruction_instead_of_ignoring_it() {
    let error = adapt(&request_body(r#", "instruction":"rank legal passages""#))
        .expect_err("custom instruction needs real upstream plumbing");

    assert!(error.to_string().contains("custom instruction"), "{error}");
    assert!(error.to_string().contains("/v1/score"), "{error}");
}

#[test]
fn validates_instruction_length_before_reporting_missing_plumbing() {
    let at_limit = "x".repeat(2_048);
    let over_limit = "x".repeat(2_049);
    let at_limit_error = adapt(&request_body(&format!(
        r#", "instruction":{}"#,
        serde_json::to_string(&at_limit).expect("serialize instruction")
    )))
    .expect_err("non-default instruction remains unsupported");
    let over_limit_error = adapt(&request_body(&format!(
        r#", "instruction":{}"#,
        serde_json::to_string(&over_limit).expect("serialize instruction")
    )))
    .expect_err("overlong instruction should fail validation");

    assert!(
        at_limit_error.to_string().contains("custom instruction"),
        "{at_limit_error}"
    );
    assert!(
        over_limit_error.to_string().contains("2048"),
        "{over_limit_error}"
    );
}

#[test]
fn rejects_missing_empty_mismatched_and_oversized_pairs() {
    let oversized = vec!["x"; 1_025];
    let cases = [
        (r#"{"documents":["d"]}"#.to_owned(), "queries"),
        (r#"{"queries":["q"]}"#.to_owned(), "documents"),
        (r#"{"queries":[],"documents":[]}"#.to_owned(), "non-empty"),
        (
            r#"{"queries":["q1","q2"],"documents":["d1"]}"#.to_owned(),
            "same length",
        ),
        (
            serde_json::to_string(&json!({"queries": oversized, "documents": ["d"]}))
                .expect("serialize oversized request"),
            "1024",
        ),
    ];

    for (body, expected) in cases {
        let error = adapt(&Bytes::from(body)).expect_err("request should fail");
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error:?}"
        );
    }
}

#[test]
fn rejects_wrong_input_and_optional_field_types() {
    let cases = [
        r#"{"queries":"q","documents":["d"]}"#,
        r#"{"queries":[1],"documents":["d"]}"#,
        r#"{"queries":["q"],"documents":"d"}"#,
        r#"{"queries":["q"],"documents":[{"text":"d"}]}"#,
        r#"{"queries":["q"],"documents":["d"],"instruction":3}"#,
        r#"{"queries":["q"],"documents":["d"],"service_tier":3}"#,
        r#"{"queries":["q"],"documents":["d"],"webhook":3}"#,
    ];

    for body in cases {
        assert!(
            adapt(&Bytes::from_static(body.as_bytes())).is_err(),
            "wrongly accepted {body}"
        );
    }
}

#[test]
fn validates_all_documented_service_tier_values() {
    for (raw, expected) in [
        ("default", ServiceTier::Default),
        ("priority", ServiceTier::Priority),
        ("flex", ServiceTier::Flex),
    ] {
        let adapted = adapt(&request_body(&format!(r#", "service_tier":"{raw}""#)))
            .expect("documented tier should adapt");
        assert_eq!(adapted.service_tier, expected);
        assert_eq!(adapted.service_tier.as_str(), raw);
    }
    let defaulted = adapt(&request_body("")).expect("tier should default");
    assert_eq!(defaulted.service_tier, ServiceTier::Default);

    let error = adapt(&request_body(r#", "service_tier":"expedite""#))
        .expect_err("unknown tier should fail");
    assert!(
        error.to_string().contains("default, priority, or flex"),
        "{error}"
    );
}

#[test]
fn rejects_non_null_webhook_as_unsupported_async_behavior() {
    adapt(&request_body(r#", "webhook":null"#)).expect("null requests no webhook");
    let error = adapt(&request_body(
        r#", "webhook":"https://example.invalid/result""#,
    ))
    .expect_err("local adapter is synchronous");

    assert!(error.to_string().contains("webhook"), "{error}");
    assert!(error.to_string().contains("synchronous"), "{error}");
}

#[test]
fn maps_scalar_head_endpoints_order_and_ties_to_probabilities() {
    let body = score_body(
        r#"[{"index":3,"score":0.0},{"index":1,"score":1.0},{"index":0,"score":-1.0},{"index":2,"score":0.0}]"#,
        r#"{"prompt_tokens":27,"total_tokens":27,"completion_tokens":0}"#,
        r#","id":"score-upstream-123""#,
    );
    let value = response_json(&body, 4).expect("valid response should adapt");

    assert_eq!(value["scores"], json!([0.0, 1.0, 0.5, 0.5]));
    assert_eq!(value["input_tokens"], 27);
    assert_eq!(value["request_id"], "score-upstream-123");
    assert_eq!(value["inference_status"]["status"], "succeeded");
    assert_eq!(value["inference_status"]["runtime_ms"], 0);
    assert_eq!(value["inference_status"]["cost"], 0.0);
    assert_eq!(value["inference_status"]["tokens_generated"], 0);
    assert_eq!(value["inference_status"]["tokens_input"], 27);
    assert_eq!(value["inference_status"]["output_length"], 0);
}

#[test]
fn clamps_only_tiny_f32_domain_roundoff() {
    let body = score_body(
        r#"[{"index":0,"score":-1.0000005},{"index":1,"score":1.0000005}]"#,
        r#"{"prompt_tokens":2}"#,
        "",
    );
    let value = response_json(&body, 2).expect("tiny roundoff should adapt");
    assert_eq!(value["scores"], json!([0.0, 1.0]));

    for score in [-1.000_01, 1.000_01] {
        let body = score_body(
            &format!(r#"[{{"index":0,"score":{score}}}]"#),
            r#"{"prompt_tokens":1}"#,
            "",
        );
        let error = response_json(&body, 1).expect_err("material domain violation should fail");
        assert!(error.contains("[-1, 1]"), "{error}");
    }
}

#[test]
fn rejects_nonfinite_scalar_scores() {
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = scalar_score_to_probability(score)
            .expect_err("non-finite scalar score must fail closed");
        assert!(error.contains("[-1, 1]"), "{error}");
    }
}

#[test]
fn rejects_incomplete_duplicate_or_out_of_range_indices() {
    let cases = [
        "[]",
        r#"[{"index":0,"score":0.0}]"#,
        r#"[{"index":0,"score":0.0},{"index":0,"score":0.1}]"#,
        r#"[{"index":0,"score":0.0},{"index":2,"score":0.1}]"#,
        r#"[{"index":1,"score":0.0},{"index":2,"score":0.1}]"#,
    ];

    for data in cases {
        let body = score_body(data, r#"{"prompt_tokens":2}"#, "");
        assert!(
            response_json(&body, 2).is_err(),
            "wrongly accepted indices in {data}"
        );
    }
}

#[test]
fn requires_trusted_non_negative_prompt_token_usage() {
    for usage in [
        "null",
        "{}",
        r#"{"prompt_tokens":-1}"#,
        r#"{"prompt_tokens":1.5}"#,
        r#"{"prompt_tokens":"1"}"#,
        r#"{"prompt_tokens":1,"total_tokens":-1}"#,
        r#"{"prompt_tokens":1,"completion_tokens":-1}"#,
    ] {
        let body = score_body(r#"[{"index":0,"score":0.0}]"#, usage, "");
        assert!(
            response_json(&body, 1).is_err(),
            "wrongly accepted usage {usage}"
        );
    }
}

#[test]
fn rejects_malformed_upstream_score_bodies() {
    let cases = [
        Bytes::from_static(b"not json"),
        Bytes::from_static(br#"{"usage":{"prompt_tokens":1}}"#),
        score_body("{}", r#"{"prompt_tokens":1}"#, ""),
        score_body(r#"[{"score":0.0}]"#, r#"{"prompt_tokens":1}"#, ""),
        score_body(r#"[{"index":0,"score":"0"}]"#, r#"{"prompt_tokens":1}"#, ""),
        score_body(
            r#"[{"index":0,"score":0.0}]"#,
            r#"{"prompt_tokens":1}"#,
            r#","id":42"#,
        ),
    ];

    for body in cases {
        assert!(
            response_json(&body, 1).is_err(),
            "wrongly accepted malformed upstream body {body:?}"
        );
    }
}

#[test]
fn omits_optional_request_id_when_upstream_has_none() {
    for id in ["", r#","id":null"#] {
        let body = score_body(r#"[{"index":0,"score":0.0}]"#, r#"{"prompt_tokens":1}"#, id);
        let value = response_json(&body, 1).expect("optional id should not fail response");
        assert!(value.get("request_id").is_none());
    }
}
