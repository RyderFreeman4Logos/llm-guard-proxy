use super::*;

const BODY_BOUND_HEADERS: [&str; 4] = ["etag", "digest", "signature", "signature-input"];

#[tokio::test]
async fn shielded_non_alias_aggregation_sanitizes_body_bound_headers() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = spawn_shielded_watchdog_proxy(&fake.base_url).await;
    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=shielded-aggregate-integrity-headers",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"test-chat","messages":[{"role":"user","content":"ping"}],"thinking":{"budget_tokens":1},"stream":false}"#,
        )
        .send()
        .await
        .expect("shielded aggregation request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/json"))
    );
    assert_eq!(
        response.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            response.headers().get(header).is_none(),
            "aggregated response must not retain {header}"
        );
    }
    let body = response
        .text()
        .await
        .expect("aggregated response body should be readable");
    let body: serde_json::Value =
        serde_json::from_str(&body).expect("aggregated response should be JSON");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    let _observed = fake.recv_next().await;
}

#[cfg(feature = "guard")]
#[tokio::test]
async fn alias_response_noop_preserves_body_bound_headers_and_bytes() {
    let fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[[forced_model_alias_profiles]]
alias = "public-forced-alias"
upstream_model = "canonical-target"
thinking_mode = "force_disable"
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#,
    )
    .await;

    let cases = [
        (
            "malformed-json",
            r#"{"model":"public-forced-alias","prompt":"ping"}"#,
            "application/json",
            b"{malformed-json".as_slice(),
        ),
        (
            "non-object-json",
            r#"{"model":"public-forced-alias","prompt":"ping"}"#,
            "application/json",
            b"[\"unchanged\"]".as_slice(),
        ),
        (
            "non-utf8-sse",
            r#"{"model":"public-forced-alias","messages":[],"stream":true}"#,
            "text/event-stream",
            b"data: \xFF\n\n".as_slice(),
        ),
    ];
    for (case, request_body, content_type, expected_body) in cases {
        let response = proxy
            .client
            .post(format!(
                "{}/v1/completions?test=forced-response-noop-{case}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, content_type)
            .body(request_body)
            .send()
            .await
            .expect("no-op alias response should complete");

        assert_eq!(response.status(), StatusCode::OK, "case={case}");
        assert_eq!(
            response.headers().get("x-safe-custom"),
            Some(&HeaderValue::from_static("preserve-me")),
            "case={case}"
        );
        for header in BODY_BOUND_HEADERS {
            assert!(
                response.headers().get(header).is_some(),
                "no-op alias response must preserve {header}; case={case}"
            );
        }
        assert_eq!(
            response
                .bytes()
                .await
                .expect("body should be readable")
                .as_ref(),
            expected_body,
            "case={case}"
        );
    }
}

#[tokio::test]
async fn buffered_models_noop_preserves_body_bound_headers_but_changed_body_does_not() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_metadata_config(
        &fake.base_url,
        true,
        r"
[upstream.metadata]
discovery_enabled = true
enrich_responses = true
",
    )
    .await;

    let changed = proxy
        .client
        .get(format!(
            "{}/v1/models?test=models-changed-integrity-headers",
            proxy.base_url
        ))
        .send()
        .await
        .expect("changed models request should complete");
    assert_eq!(changed.status(), StatusCode::OK);
    assert_eq!(
        changed.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert!(
            changed.headers().get(header).is_none(),
            "changed models response must not retain {header}"
        );
    }
    let changed_body = changed
        .text()
        .await
        .expect("changed models body should be readable");
    let changed_body: serde_json::Value =
        serde_json::from_str(&changed_body).expect("changed models body should be JSON");
    assert_eq!(changed_body["data"][0]["context_length"], 256_000);
    let _changed_observed = fake.recv_next().await;

    let noop = proxy
        .client
        .get(format!(
            "{}/v1/models?test=models-noop-integrity-headers",
            proxy.base_url
        ))
        .send()
        .await
        .expect("no-op models request should complete");
    assert_eq!(noop.status(), StatusCode::OK);
    assert_eq!(
        noop.headers().get("x-safe-custom"),
        Some(&HeaderValue::from_static("preserve-me"))
    );
    for header in BODY_BOUND_HEADERS {
        assert_eq!(
            noop.headers()
                .get(header)
                .and_then(|value| value.to_str().ok()),
            Some(match header {
                "etag" => "\"stale-etag\"",
                "digest" => "sha-256=stale",
                "signature" => "stale-signature",
                "signature-input" => "stale-signature-input",
                _ => unreachable!(),
            }),
            "no-op models response must preserve {header}"
        );
    }
    let noop_body = noop
        .text()
        .await
        .expect("no-op models body should be readable");
    assert_eq!(
        noop_body,
        r#"{"object":"list","data":[{"id":"stable-model","object":"model","max_model_len":256000,"context_length":256000,"max_context_length":256000,"owned_by":"vllm"}]}"#
    );
    let _noop_observed = fake.recv_next().await;
}
