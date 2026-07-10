use super::*;
use axum::http::Uri;

#[test]
fn converts_single_pair_score_body() {
    let body = Bytes::from_static(br#"{"model":"qwen3-reranker-8b","text_1":"q","text_2":"d"}"#);
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
fn canonical_score_fields_win_over_rerank_compat_extras() {
    let body = Bytes::from_static(
        br#"{"model":"m","text_1":"score-query","text_2":"score-document","query":"extra-query","documents":["extra-document"]}"#,
    );
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
    let out = score_body_to_rerank_body(&body).expect("convert canonical score fields");
    let value: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(value["query"], "score-query");
    assert_eq!(value["documents"], json!(["score-document"]));
}

#[test]
fn raw_preservation_cannot_restore_mapped_score_fields() {
    let digits = "9".repeat(1_000);
    let body = Bytes::from(format!(
        r#"{{"model":"m","text_1":"score-query","text_2":"score-document","query":{digits},"documents":[{digits}]}}"#
    ));
    let out = score_body_to_rerank_body(&body).expect("convert canonical score fields");
    let value: Value = serde_json::from_slice(&out).expect("mapped fields should be normal JSON");
    assert_eq!(value["model"], "m");
    assert_eq!(value["query"], "score-query");
    assert_eq!(value["documents"], json!(["score-document"]));
    assert_eq!(value["top_n"], 1);

    let invalid_model = Bytes::from(format!(r#"{{"model":{digits},"text_1":"q","text_2":"d"}}"#));
    assert!(score_body_to_rerank_body(&invalid_model).is_err());
}

#[test]
fn converts_rerank_response_to_score() {
    let body = Bytes::from_static(
        br#"{"id":"rerank-1","model":"m","results":[{"index":1,"score":0.9},{"index":0,"score":0.1}]}"#,
    );
    let out = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 2,
            document_count: 2,
        }),
    )
    .expect("convert");
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["index"], 0);
    assert_eq!(v["data"][0]["score"], 0.1);
    assert_eq!(v["data"][0]["object"], "score");
    assert_eq!(v["data"][1]["index"], 1);
    assert_eq!(v["data"][1]["score"], 0.9);
    assert_eq!(v["id"], "rerank-1");
    assert_eq!(v["model"], "m");
    assert!(v["created"].as_u64().is_some_and(|created| created > 0));
    assert_eq!(v["usage"]["prompt_tokens"], 0);
    assert_eq!(v["usage"]["total_tokens"], 0);
    assert_eq!(v["usage"]["completion_tokens"], 0);
    assert!(v["usage"]["prompt_tokens_details"].is_null());
}

#[test]
fn converts_compatible_data_response_aliases_to_score() {
    let body = Bytes::from_static(
        br#"{"model":"m","data":[{"document_index":1,"relevance_score":0.9},{"index":0,"rerank_score":0.1}]}"#,
    );
    let out = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 2,
            document_count: 2,
        }),
    )
    .expect("compatible aliases should convert");
    let value: Value = serde_json::from_slice(&out).expect("score response JSON");
    assert_eq!(value["data"][0]["index"], 0);
    assert_eq!(value["data"][0]["score"], 0.1);
    assert_eq!(value["data"][1]["index"], 1);
    assert_eq!(value["data"][1]["score"], 0.9);
}

#[test]
fn preserves_created_and_normalizes_rerank_usage() {
    let body = Bytes::from_static(
        br#"{"id":"rerank-1","created":123,"model":"m","usage":{"prompt_tokens":7,"total_tokens":9,"completion_tokens":null,"prompt_tokens_details":{"cached_tokens":2,"private_upstream_metadata":"must-not-leak"}},"results":[{"index":0,"score":0.1}]}"#,
    );
    let out = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 1,
            document_count: 1,
        }),
    )
    .expect("convert");
    let value: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(value["created"], 123);
    assert_eq!(value["usage"]["prompt_tokens"], 7);
    assert_eq!(value["usage"]["total_tokens"], 9);
    assert!(value["usage"]["completion_tokens"].is_null());
    assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
    assert!(
        value["usage"]["prompt_tokens_details"]
            .get("private_upstream_metadata")
            .is_none()
    );
}

#[test]
fn rejects_invalid_cached_token_usage() {
    let body = Bytes::from_static(
        br#"{"model":"m","usage":{"prompt_tokens_details":{"cached_tokens":"invalid"}},"results":[{"index":0,"score":0.1}]}"#,
    );
    let error = rerank_response_to_score_response(&body, Some("m"), None).unwrap_err();
    assert!(error.contains("cached_tokens"), "{error}");
}

#[test]
fn rejects_malformed_result_entry() {
    let body =
        Bytes::from_static(br#"{"results":[{"index":0,"score":0.9},{"index":"bad","score":0.2}]}"#);
    let err = rerank_response_to_score_response(&body, Some("m"), None).unwrap_err();
    assert!(
        err.contains("missing or invalid index"),
        "unexpected err: {err}"
    );
}

#[test]
fn rejects_duplicate_indices() {
    let body =
        Bytes::from_static(br#"{"results":[{"index":0,"score":0.9},{"index":0,"score":0.2}]}"#);
    let err = rerank_response_to_score_response(&body, Some("m"), None).unwrap_err();
    assert!(err.contains("duplicate index"), "unexpected err: {err}");
}

#[test]
fn rejects_empty_results() {
    let body = Bytes::from_static(br#"{"results":[]}"#);
    let err = rerank_response_to_score_response(&body, Some("m"), None).unwrap_err();
    assert!(err.contains("empty"), "unexpected err: {err}");
}

#[test]
fn rewrites_uri_path() {
    let uri: Uri = "/v1/score".parse().unwrap();
    let out = score_uri_to_rerank_uri(&uri).unwrap();
    assert_eq!(out.path(), "/v1/rerank");
}

#[test]
fn rejects_partial_results_when_expected_count_set() {
    let body = Bytes::from_static(br#"{"results":[{"index":0,"score":0.9}]}"#);
    let err = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 2,
            document_count: 2,
        }),
    )
    .unwrap_err();
    assert!(
        err.contains("count") || err.contains("missing index"),
        "{err}"
    );
}

#[test]
fn rejects_out_of_range_index_when_expected_count_set() {
    let body =
        Bytes::from_static(br#"{"results":[{"index":0,"score":0.1},{"index":5,"score":0.9}]}"#);
    let err = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 2,
            document_count: 2,
        }),
    )
    .unwrap_err();
    assert!(
        err.contains("out of range") || err.contains("missing index"),
        "{err}"
    );
}

#[test]
fn classifies_batch_score_as_non_adaptable() {
    let body = Bytes::from_static(br#"{"model":"m","text_1":["q1","q2"],"text_2":["d1","d2"]}"#);
    assert!(!can_adapt_score_body_to_rerank(&body).unwrap());
}

#[test]
fn classifies_scalar_score_as_adaptable() {
    let body = Bytes::from_static(br#"{"model":"m","text_1":"q","text_2":"d"}"#);
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
}

#[test]
fn rejects_malformed_complete_canonical_score_fields() {
    for body in [
        br#"{"model":"m","text_1":"q","text_2":42}"#.as_slice(),
        br#"{"model":"m","text_1":"q","text_2":["d",42]}"#.as_slice(),
        br#"{"model":"m","text_1":null,"text_2":"d"}"#.as_slice(),
        br#"{"model":"m","text_1":"q","text_2":{"content":42}}"#.as_slice(),
    ] {
        assert!(
            can_adapt_score_body_to_rerank(&Bytes::copy_from_slice(body)).is_err(),
            "known-invalid canonical score body should fail locally: {}",
            String::from_utf8_lossy(body)
        );
    }
}

#[test]
fn rejects_incomplete_known_score_shapes() {
    for body in [
        br"{}".as_slice(),
        br#"{"model":"m"}"#.as_slice(),
        br#"{"foo":1}"#.as_slice(),
        br#"{"queries":["q"],"typo":true}"#.as_slice(),
        br#"{"items":["d"],"typo":true}"#.as_slice(),
        br#"{"data_1":"q","typo":true}"#.as_slice(),
        br#"{"model":"m","query":"q"}"#.as_slice(),
        br#"{"query":"q","typo":true}"#.as_slice(),
        br#"{"model":"m","documents":["d"]}"#.as_slice(),
    ] {
        assert!(
            can_adapt_score_body_to_rerank(&Bytes::copy_from_slice(body)).is_err(),
            "incomplete known score body should fail locally: {}",
            String::from_utf8_lossy(body)
        );
    }
}

#[test]
fn rejects_invalid_score_input_cardinality() {
    for body in [
        br#"{"text_1":["q1","q2"],"text_2":["d1","d2","d3"]}"#.as_slice(),
        br#"{"text_1":["q1","q2"],"text_2":"d"}"#.as_slice(),
        br#"{"text_1":{"content":[{"type":"text","text":"q1"},{"type":"text","text":"q2"}]},"text_2":"d"}"#.as_slice(),
    ] {
        assert!(
            can_adapt_score_body_to_rerank(&Bytes::copy_from_slice(body)).is_err(),
            "invalid score cardinality should fail locally: {}",
            String::from_utf8_lossy(body)
        );
    }
}

#[test]
fn rejects_invalid_multimodal_content_parts() {
    for body in [
        br#"{"text_1":"q","text_2":{"content":[{}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"text"}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"image_url","image_url":{}}]}}"#
            .as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"image_url","image_url":{"url":"u","detail":"invalid"}}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"image_embeds","image_embeds":{"x":1}}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"video_url","video_url":{}}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"unknown"}]}}"#.as_slice(),
    ] {
        assert!(
            can_adapt_score_body_to_rerank(&Bytes::copy_from_slice(body)).is_err(),
            "invalid multimodal score body should fail locally: {}",
            String::from_utf8_lossy(body)
        );
    }
}

#[test]
fn classifies_unknown_and_multimodal_score_shapes_as_passthrough() {
    for body in [
        br#"{"model":"m","queries":["q"],"documents":["d"]}"#.as_slice(),
        br#"{"model":"m","queries":["q"],"items":["d"]}"#.as_slice(),
        br#"{"model":"m","query":{"content":[{"type":"text","text":"q"}]},"items":["d"]}"#.as_slice(),
        br#"{"model":"m","data_1":"q","data_2":"d"}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"text","text":"d"}]}}"#.as_slice(),
        br#"{"model":"m","text_1":"q","text_2":{"content":[{"type":"image_url","image_url":{"url":"https://example.invalid/a.png"}}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"image_embeds","image_embeds":"AA=="}]}}"#.as_slice(),
        br#"{"text_1":"q","text_2":{"content":[{"type":"video_url","video_url":{"url":"https://example.invalid/a.mp4"}}]}}"#.as_slice(),
    ] {
        assert!(!can_adapt_score_body_to_rerank(&Bytes::copy_from_slice(body)).unwrap());
    }
}

#[test]
fn classifies_arbitrary_precision_future_shape_as_passthrough() {
    let body = Bytes::from(format!(
        r#"{{"model":"m","queries":["q"],"items":["d"],"future":{}}}"#,
        "9".repeat(1_000)
    ));
    assert!(!can_adapt_score_body_to_rerank(&body).unwrap());
    assert_eq!(model_id_from_score_body(&body).as_deref(), Some("m"));
}

#[test]
fn classifies_complete_legacy_rerank_shape_as_adaptable() {
    let body = Bytes::from_static(br#"{"model":"m","query":"q","documents":["a","b"]}"#);
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
}

#[test]
fn classifies_multimodal_query_legacy_shape_as_adaptable() {
    let body = Bytes::from_static(
        br#"{"model":"m","query":{"content":[{"type":"text","text":"q"}]},"documents":["a","b"]}"#,
    );
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
}

#[test]
fn rejects_pydantic_invalid_legacy_top_n_values() {
    for top_n in [
        json!("1."),
        json!("1e3"),
        json!("-_1"),
        json!("+-1"),
        json!("--1"),
        json!("-1e3"),
        json!("+_1"),
        json!(1.5),
        json!(1e20),
        json!(-1e20),
        Value::Null,
        json!([]),
        json!({}),
    ] {
        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "query": "q",
                "documents": ["a", "b"],
                "top_n": top_n,
            }))
            .unwrap(),
        );
        assert!(
            can_adapt_score_body_to_rerank(&body).is_err(),
            "invalid top_n should fail locally: {top_n}"
        );
    }
}

#[test]
fn accepts_pydantic_valid_legacy_top_n_values() {
    for top_n in [
        json!(-1),
        json!(-1.0),
        json!("-1.0"),
        json!("00-1"),
        json!("0_-1"),
        json!("0__-1"),
        json!("-0_1"),
        json!(false),
        json!(1.0),
        json!("1_0"),
    ] {
        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "query": "q",
                "documents": ["a", "b"],
                "top_n": top_n,
            }))
            .unwrap(),
        );
        assert!(
            can_adapt_score_body_to_rerank(&body).unwrap(),
            "valid top_n should remain adaptable: {top_n}"
        );
    }
}

#[test]
fn pydantic_top_n_string_size_limit_applies_after_cleaning() {
    let max_digits = "9".repeat(4_300);
    let max_negative_digits = "9".repeat(4_299);
    assert!(parse_lax_top_n_string(&max_digits).is_some());
    assert!(parse_lax_top_n_string(&format!("9{max_digits}")).is_none());
    assert!(parse_lax_top_n_string(&format!("-{max_negative_digits}")).is_some());
    assert!(parse_lax_top_n_string(&format!("-{max_digits}")).is_none());
    assert!(parse_lax_top_n_string(&format!(" {max_digits} ")).is_some());
    assert!(parse_lax_top_n_string(&"0".repeat(4_301)).is_some());
}

#[test]
fn pydantic_arbitrary_precision_json_integer_top_n_is_preserved() {
    for top_n in [
        String::from("18446744073709551616"),
        String::from("-9223372036854775809"),
        "9".repeat(1_000),
        format!("-{}", "9".repeat(4_299)),
    ] {
        let body = Bytes::from(format!(
            r#"{{"query":"q","documents":["a","b"],"top_n":{top_n}}}"#
        ));
        assert!(
            can_adapt_score_body_to_rerank(&body).unwrap(),
            "Pydantic-valid arbitrary-precision integer should adapt"
        );
    }

    for oversized in ["9".repeat(4_301), format!("-{}", "9".repeat(4_300))] {
        let body = Bytes::from(format!(
            r#"{{"query":"q","documents":["a","b"],"top_n":{oversized}}}"#
        ));
        assert!(can_adapt_score_body_to_rerank(&body).is_err());
    }
}

#[test]
fn preserves_pydantic_nonfinite_extra_float_lexical_form() {
    let body = Bytes::from_static(br#"{"model":"m","text_1":"q","text_2":"d","future":1e400}"#);
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
    let out = score_body_to_rerank_body(&body).expect("Pydantic accepts ignored extra float");
    let out = std::str::from_utf8(&out).expect("serialized request is UTF-8");
    assert!(out.contains(r#""future":1e400"#), "{out}");
    assert!(out.contains(r#""query":"q""#), "{out}");
    assert!(out.contains(r#""documents":["d"]"#), "{out}");
}

#[test]
fn ignores_pydantic_nonfinite_top_n_extra_for_canonical_score() {
    let body =
        Bytes::from_static(br#"{"model":"m","text_1":"q","text_2":["a","b"],"top_n":1e400}"#);
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
    let out = score_body_to_rerank_body(&body).expect("canonical top_n is an ignored extra");
    let value: Value = serde_json::from_slice(&out).expect("mapped body is finite JSON");
    assert_eq!(value["top_n"], 2);
    assert_eq!(value["documents"], json!(["a", "b"]));
}

#[test]
fn accepts_top_n_less_than_documents() {
    let body = Bytes::from_static(br#"{"results":[{"index":1,"score":0.9}]}"#);
    let out = rerank_response_to_score_response(
        &body,
        Some("m"),
        Some(ScoreExpectations {
            result_count: 1,
            document_count: 2,
        }),
    )
    .expect("top_n=1 may return non-zero index");
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["data"].as_array().unwrap().len(), 1);
    assert_eq!(v["data"][0]["index"], 1);
}

#[test]
fn legacy_zero_top_n_expects_all_documents() {
    for top_n in [
        json!(0),
        json!(0.0),
        json!("0"),
        json!("-1.0"),
        json!(false),
    ] {
        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "model": "m",
                "query": "q",
                "documents": ["a", "b"],
                "top_n": top_n,
            }))
            .unwrap(),
        );
        assert_eq!(
            score_expectations_from_rerank_body(&body),
            Some(ScoreExpectations {
                result_count: 2,
                document_count: 2,
            })
        );
    }
}

#[test]
fn legacy_lax_top_n_coercions_match_vllm() {
    for top_n in [json!(1.0), json!("1"), json!("1.0"), json!(true)] {
        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "model": "m",
                "query": "q",
                "documents": ["a", "b"],
                "top_n": top_n,
            }))
            .unwrap(),
        );
        assert_eq!(
            score_expectations_from_rerank_body(&body),
            Some(ScoreExpectations {
                result_count: 1,
                document_count: 2,
            })
        );
    }
}

#[test]
fn legacy_underscored_top_n_coercions_match_vllm() {
    for top_n in [json!("1_0"), json!("0__10.0")] {
        let documents = (0..12).map(|index| format!("d{index}")).collect::<Vec<_>>();
        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "model": "m",
                "query": "q",
                "documents": documents,
                "top_n": top_n,
            }))
            .unwrap(),
        );
        assert_eq!(
            score_expectations_from_rerank_body(&body),
            Some(ScoreExpectations {
                result_count: 10,
                document_count: 12,
            })
        );
    }
}

#[test]
fn legacy_multimodal_documents_enforce_response_cardinality() {
    let request = Bytes::from_static(
        br#"{"model":"m","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},"top_n":1}"#,
    );
    let expectations = score_expectations_from_rerank_body(&request)
        .expect("valid multimodal documents should produce expectations");
    assert_eq!(
        expectations,
        ScoreExpectations {
            result_count: 2,
            document_count: 2,
        }
    );

    let partial_response =
        Bytes::from_static(br#"{"model":"m","results":[{"index":0,"score":0.9}]}"#);
    assert!(
        rerank_response_to_score_response(&partial_response, Some("m"), Some(expectations),)
            .is_err()
    );

    let complete_response = Bytes::from_static(
        br#"{"model":"m","results":[{"index":0,"score":0.9},{"index":1,"score":0.8}]}"#,
    );
    assert!(
        rerank_response_to_score_response(&complete_response, Some("m"), Some(expectations),)
            .is_ok()
    );
}

#[test]
fn legacy_multimodal_document_extras_affect_top_n_truncation() {
    let request = Bytes::from_static(
        br#"{"model":"m","query":"q","documents":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"future_extension":true},"top_n":1}"#,
    );
    let expectations = score_expectations_from_rerank_body(&request)
        .expect("valid extended multimodal documents should produce expectations");
    assert_eq!(
        expectations,
        ScoreExpectations {
            result_count: 1,
            document_count: 2,
        }
    );

    let truncated_response =
        Bytes::from_static(br#"{"model":"m","results":[{"index":0,"score":0.9}]}"#);
    assert!(
        rerank_response_to_score_response(&truncated_response, Some("m"), Some(expectations),)
            .is_ok()
    );
}

#[test]
fn preserves_score_options() {
    let body = Bytes::from_static(
        br#"{"model":"m","text_1":"q","text_2":"d","truncate_prompt_tokens":128,"priority":1}"#,
    );
    let out = score_body_to_rerank_body(&body).expect("convert");
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["truncate_prompt_tokens"], 128);
    assert_eq!(v["priority"], 1);
    assert_eq!(v["query"], "q");
}

#[test]
fn ignores_caller_top_n_extra_for_canonical_score() {
    let body = Bytes::from_static(br#"{"model":"m","text_1":"q","text_2":["a","b"],"top_n":1}"#);
    let out = score_body_to_rerank_body(&body).expect("convert");
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["top_n"], 2);
    assert_eq!(v["documents"].as_array().unwrap().len(), 2);
    assert_eq!(
        score_expectations_from_rerank_body(&out),
        Some(ScoreExpectations {
            result_count: 2,
            document_count: 2,
        })
    );
}

#[test]
fn ignores_arbitrary_precision_top_n_extra_for_canonical_score() {
    for top_n in ["9".repeat(1_000), format!("-{}", "9".repeat(4_299))] {
        let body = Bytes::from(format!(
            r#"{{"model":"m","text_1":"q","text_2":["a","b"],"top_n":{top_n}}}"#
        ));
        assert!(can_adapt_score_body_to_rerank(&body).unwrap());
        let out = score_body_to_rerank_body(&body).expect("convert");
        let value: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["top_n"], 2);
    }
}

#[test]
fn raw_fallback_preserves_model_for_policy_and_routing() {
    let body = Bytes::from(format!(
        r#"{{"model":"forbidden-model","text_1":"q","text_2":"d","top_n":{}}}"#,
        "9".repeat(1_000)
    ));
    assert_eq!(
        model_id_from_score_body(&body).as_deref(),
        Some("forbidden-model")
    );
}

#[test]
fn raw_fallback_rejects_each_pydantic_invalid_number_occurrence() {
    let oversized = "9".repeat(4_301);
    let oversized_top_n = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","top_n":{oversized}}}"#
    ));
    assert!(can_adapt_score_body_to_rerank(&oversized_top_n).is_err());

    let shadowed_extra = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","top_n":{},"future":{oversized},"future":1}}"#,
        "9".repeat(1_000)
    ));
    assert!(can_adapt_score_body_to_rerank(&shadowed_extra).is_err());

    let shadowed_top_n = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","top_n":{oversized},"top_n":1}}"#
    ));
    assert!(can_adapt_score_body_to_rerank(&shadowed_top_n).is_err());
}

#[test]
fn pydantic_arbitrary_precision_extras_are_preserved_recursively() {
    for digits in ["9".repeat(1_000), "8".repeat(4_300)] {
        let body = Bytes::from(format!(
            r#"{{"model":"m","text_1":"q","text_2":"d","future":{digits},"nested":{{"items":[{digits}]}}}}"#
        ));
        assert!(can_adapt_score_body_to_rerank(&body).unwrap());
        let out = score_body_to_rerank_body(&body).unwrap();
        let output = std::str::from_utf8(&out).unwrap();
        assert!(output.contains(&format!(r#""future":{digits}"#)));
        assert!(output.contains(&format!(r#""items":[{digits}]"#)));
    }
}

#[test]
fn pydantic_integers_beyond_serde_ranges_are_preserved_exactly() {
    let body = Bytes::from_static(
        br#"{"text_1":"q","text_2":"d","priority":18446744073709551617,"minimum":-9223372036854775809,"nested":{"value":18446744073709551616}}"#,
    );
    let out = score_body_to_rerank_body(&body).unwrap();
    let output = std::str::from_utf8(&out).unwrap();
    assert!(output.contains(r#""priority":18446744073709551617"#));
    assert!(output.contains(r#""minimum":-9223372036854775809"#));
    assert!(output.contains(r#""value":18446744073709551616"#));
}

#[test]
fn serde_integer_boundaries_and_numeric_strings_do_not_force_preservation() {
    assert!(!contains_non_serde_integer(
        br#"{"positive":18446744073709551615,"negative":-9223372036854775808}"#
    ));
    assert!(!contains_non_serde_integer(
        br#"{"text":"18446744073709551616"}"#
    ));
    assert!(contains_non_serde_integer(
        br#"{"positive":18446744073709551616}"#
    ));
    assert!(contains_non_serde_integer(
        br#"{"negative":-9223372036854775809}"#
    ));
}

#[test]
fn arbitrary_precision_extra_preservation_obeys_duplicate_last_wins() {
    let digits = "9".repeat(1_000);
    let standard_last = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","future":{digits},"future":1}}"#
    ));
    let out = score_body_to_rerank_body(&standard_last).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&out).unwrap()["future"], 1);

    let arbitrary_last = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","future":1,"future":{digits}}}"#
    ));
    let out = score_body_to_rerank_body(&arbitrary_last).unwrap();
    assert!(
        std::str::from_utf8(&out)
            .unwrap()
            .contains(&format!(r#""future":{digits}"#))
    );
}

#[test]
fn legacy_nested_arbitrary_precision_extra_remains_passthrough() {
    let body = Bytes::from(format!(
        r#"{{"query":"q","documents":["d"],"future":{{"value":{}}}}}"#,
        "9".repeat(1_000)
    ));
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
    assert_eq!(score_body_to_rerank_body(&body).unwrap(), body);
}

#[test]
fn pydantic_json_recursion_boundary_applies_to_raw_fallback() {
    let digits = "9".repeat(1_000);
    let valid_beyond_serde_limit = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","future":{}null{}}}"#,
        "[".repeat(150),
        "]".repeat(150)
    ));
    assert!(can_adapt_score_body_to_rerank(&valid_beyond_serde_limit).unwrap());

    let body_at_limit = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","future":{}{digits}{}}}"#,
        "[".repeat(199),
        "]".repeat(199)
    ));
    assert!(can_adapt_score_body_to_rerank(&body_at_limit).unwrap());

    let body_over_limit = Bytes::from(format!(
        r#"{{"text_1":"q","text_2":"d","future":{}{digits}{}}}"#,
        "[".repeat(200),
        "]".repeat(200)
    ));
    assert!(can_adapt_score_body_to_rerank(&body_over_limit).is_err());
}

#[test]
fn serde_json_numeric_lexemes_remain_normalized_for_shared_fingerprints() {
    let values: Vec<Value> = ["1.0", "1.00", "1e0"]
        .into_iter()
        .map(|raw| serde_json::from_str(raw).unwrap())
        .collect();
    assert!(values.windows(2).all(|pair| pair[0] == pair[1]));
    let serialized: Vec<String> = values.iter().map(Value::to_string).collect();
    assert!(serialized.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn legacy_top_n_float_rounding_matches_pydantic_json() {
    let rejected =
        Bytes::from_static(br#"{"query":"q","documents":["d"],"top_n":-3329018154707461.354}"#);
    assert!(can_adapt_score_body_to_rerank(&rejected).is_err());

    let accepted = Bytes::from_static(
        br#"{"query":"q","documents":["d"],"top_n":448842541752324.9858557669766563163561207}"#,
    );
    assert!(can_adapt_score_body_to_rerank(&accepted).unwrap());
}

#[test]
fn accepts_pydantic_finite_float_boundary_in_canonical_extra() {
    let body =
        Bytes::from_static(br#"{"text_1":"q","text_2":"d","future":1.7976931348623158e308}"#);
    assert!(can_adapt_score_body_to_rerank(&body).unwrap());
    let out = score_body_to_rerank_body(&body).expect("Pydantic accepts finite f64 boundary");
    let value: Value = serde_json::from_slice(&out).expect("adapted body is finite JSON");
    assert_eq!(value["future"].as_f64(), Some(f64::MAX));
}

#[test]
fn future_shape_does_not_apply_legacy_top_n_validation() {
    let body = Bytes::from_static(br#"{"model":"m","queries":["q"],"items":["d"],"top_n":1e400}"#);
    assert!(!can_adapt_score_body_to_rerank(&body).expect("future shape must remain opaque"));
}

#[test]
fn malformed_preferred_results_does_not_fall_back_to_data() {
    let body = Bytes::from_static(
        br#"{"model":"m","results":{"malformed":true},"data":[{"index":0,"score":0.75}]}"#,
    );
    assert!(rerank_response_to_score_response(&body, Some("m"), None).is_err());
}
