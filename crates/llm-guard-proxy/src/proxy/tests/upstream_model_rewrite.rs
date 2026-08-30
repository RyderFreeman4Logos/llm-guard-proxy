#![cfg(feature = "guard")]

use super::*;

const FORCED_MODEL_ALIAS_PROFILES_CONFIG: &str = r#"
[upstream]
reserved_ingress_model_ids = ["abliterated-qwen-latest-27b-nvfp4", "aeon", "aeon-ultimate"]

[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-none"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_disable"
output_cap = 16384
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0

[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-low"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_thinking"
thinking_budget = 65536
output_cap = 16384
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0

[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-medium"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_thinking"
thinking_budget = 65536
output_cap = 16384
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
"#;

#[tokio::test]
async fn guard_listener_rejects_reserved_forced_alias_upstream_model_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&fake.base_url, FORCED_MODEL_ALIAS_PROFILES_CONFIG)
            .await;
    let guard_state = proxy.state.for_listener(ListenerConfig {
        name: String::from("guard"),
        bind_host: String::from("127.0.0.1"),
        port: 18009,
        allowed_upstreams: None,
        upstream_profile: None,
    });

    for model in ["abliterated-qwen-latest-27b-nvfp4", "aeon", "aeon-ultimate"] {
        for (path, body) in [
            (
                "/v1/chat/completions",
                format!(r#"{{"model":"{model}","messages":[]}}"#),
            ),
            (
                "/v1/completions",
                format!(r#"{{"model":"{model}","prompt":"ping"}}"#),
            ),
            (
                "/v1/embeddings",
                format!(r#"{{"model":"{model}","input":"ping"}}"#),
            ),
        ] {
            let response = proxy_handler(
                State(guard_state.clone()),
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("reserved-model request should build"),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model} {path}");
            let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
                .await
                .expect("rejection body should be readable");
            assert!(
                std::str::from_utf8(&body)
                    .expect("rejection body should be UTF-8")
                    .contains("reserved"),
                "{model} {path} must return the deterministic reservation error"
            );
            assert_no_upstream_request(&mut fake).await;
        }
    }

    let retired_alias = format!("{}-none", "abliterated-qwen-latest-27b-nvfp4");
    let response = proxy_handler(
        State(guard_state),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(format!(
                r#"{{"model":"{retired_alias}","messages":[]}}"#
            )))
            .expect("retired alias request should build"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let observed: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("retired alias request should be JSON");
    assert_eq!(
        observed["model"], retired_alias,
        "retired spelling must not activate the forced canonical rewrite"
    );
}

#[tokio::test]
async fn forced_alias_routes_named_profile_before_listener_authorization() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "forced-target"
base_url = "{}"
match_models = ["forced-canonical"]

[[forced_model_alias_profiles]]
alias = "forced-public-alias"
upstream_model = "forced-canonical"
thinking_mode = "force_disable"
output_cap = 16
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#,
            fake.base_url
        ),
    )
    .await;
    let guard_state = proxy.state.for_listener(ListenerConfig {
        name: String::from("guard"),
        bind_host: String::from("127.0.0.1"),
        port: 18009,
        allowed_upstreams: Some(vec![String::from("forced-target")]),
        upstream_profile: None,
    });

    let response = proxy_handler(
        State(guard_state),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"model":"forced-public-alias","messages":[]}"#,
            ))
            .expect("forced alias request should build"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let observed: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed["model"], "forced-canonical");
}

#[tokio::test]
async fn forced_alias_is_rejected_on_responses_before_forwarding() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&fake.base_url, FORCED_MODEL_ALIAS_PROFILES_CONFIG)
            .await;

    let response = proxy_handler(
        State(proxy.state.clone()),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"model":"abliterated-qwen-latest-27b-none","input":"ping"}"#,
            ))
            .expect("responses request should build"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("rejection body should be readable");
    assert!(
        std::str::from_utf8(&body)
            .expect("rejection body should be UTF-8")
            .contains("forced model aliases are not supported on /v1/responses")
    );
    assert_no_upstream_request(&mut fake).await;

    let ordinary = proxy_handler(
        State(proxy.state.clone()),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"ordinary-model","input":"ping"}"#))
            .expect("ordinary responses request should build"),
    )
    .await;

    assert_eq!(ordinary.status(), StatusCode::NOT_FOUND);
    assert_eq!(fake.recv_next().await.path_and_query, "/v1/responses");
}

#[tokio::test]
async fn guard_listener_lists_forced_aliases_before_reserved_model_filtering() {
    let models = r#"{"object":"list","data":[{"id":"abliterated-qwen-latest-27b-nvfp4","object":"model"},{"id":"aeon","object":"model"},{"id":"aeon-ultimate","object":"model"},{"id":"unrelated-model","object":"model"},{"id":"pooling-model","object":"model"}]}"#;
    let fake = FakeUpstream::spawn_with_models_body(models).await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&fake.base_url, FORCED_MODEL_ALIAS_PROFILES_CONFIG)
            .await;
    let guard_state = proxy.state.for_listener(ListenerConfig {
        name: String::from("guard"),
        bind_host: String::from("127.0.0.1"),
        port: 18009,
        allowed_upstreams: None,
        upstream_profile: None,
    });

    let response = proxy_handler(
        State(guard_state),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models?test=distinct-multi-upstream-models")
            .body(Body::empty())
            .expect("models request should build"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models response should be readable");
    let models: serde_json::Value = serde_json::from_slice(&body).expect("models response JSON");
    let model_ids = models["data"]
        .as_array()
        .expect("models response data")
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        model_ids,
        vec![
            "unrelated-model",
            "pooling-model",
            "abliterated-qwen-latest-27b-none",
            "abliterated-qwen-latest-27b-low",
            "abliterated-qwen-latest-27b-medium",
        ],
        "forced aliases must be materialized before reserved model filtering"
    );
}

#[tokio::test]
async fn listener_forced_profiles_list_routable_forced_aliases_after_enrichment() {
    let models = r#"{"object":"list","data":[{"id":"canonical-target","object":"model"},{"id":"unrelated-model","object":"model"}]}"#;
    let mut fake = FakeUpstream::spawn_with_models_body(models).await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[upstream]
reserved_ingress_model_ids = ["canonical-target"]

[upstream.metadata]
discovery_enabled = true
enrich_responses = true

[[upstreams]]
name = "forced-target"
base_url = "{}"
match_models = ["canonical-target"]

[[forced_model_alias_profiles]]
alias = "forced-public-alias"
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
            fake.base_url
        ),
    )
    .await;

    for port in [18000, GUARD_PUBLIC_LISTENER_PORT] {
        let response = proxy_handler(
            State(proxy.state.for_listener(ListenerConfig {
                name: format!("forced-{port}"),
                bind_host: String::from("127.0.0.1"),
                port,
                allowed_upstreams: None,
                upstream_profile: Some(String::from("forced-target")),
            })),
            empty_get_request("/v1/models?test=distinct-multi-upstream-models"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("models response should read");
        let models: serde_json::Value =
            serde_json::from_slice(&body).expect("models response JSON");
        assert_eq!(
            models["data"]
                .as_array()
                .expect("models response data")
                .iter()
                .filter_map(|model| model["id"].as_str())
                .collect::<Vec<_>>(),
            vec!["forced-public-alias"],
            "listener-forced profiles must publish only their routed forced alias"
        );
        assert_eq!(
            fake.recv_next().await.path_and_query,
            "/v1/models?test=distinct-multi-upstream-models"
        );
    }
}

#[tokio::test]
async fn models_enrichment_does_not_republish_reserved_legacy_aliases() {
    let models = r#"{"object":"list","data":[{"id":"reserved-canonical","object":"model"},{"id":"unrelated-model","object":"model"}]}"#;
    let fake = FakeUpstream::spawn_with_models_body(models).await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[upstream]
reserved_ingress_model_ids = ["reserved-canonical"]

[upstream.metadata]
discovery_enabled = true
enrich_responses = true

[[model_aliases]]
id = "reserved-canonical"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "unrelated-alias"
kind = "upstream"
upstream_profile = "default"
"#,
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.for_listener(ListenerConfig {
            name: String::from("default"),
            bind_host: String::from("127.0.0.1"),
            port: 18000,
            allowed_upstreams: None,
            upstream_profile: None,
        })),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models?test=distinct-multi-upstream-models")
            .body(Body::empty())
            .expect("models request should build"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("models response should be readable");
    let models: serde_json::Value = serde_json::from_slice(&body).expect("models response JSON");
    let model_ids = models["data"]
        .as_array()
        .expect("models response data")
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(model_ids, vec!["unrelated-model", "unrelated-alias"]);
}

#[tokio::test]
async fn grouped_models_enrichment_does_not_republish_reserved_legacy_aliases() {
    let models = r#"{"object":"list","data":[{"id":"reserved-canonical","object":"model"},{"id":"unrelated-model","object":"model"}]}"#;
    let mut default_fake = FakeUpstream::spawn_with_models_body(models).await;
    let mut grouped_fake = FakeUpstream::spawn_with_models_body(models).await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &default_fake.base_url,
        &format!(
            r#"
[upstream]
reserved_ingress_model_ids = ["reserved-canonical"]

[upstream.metadata]
discovery_enabled = true
enrich_responses = true

[[upstreams]]
name = "grouped"
base_url = "{}"
match_models = ["unrelated-model"]

[[model_aliases]]
id = "reserved-canonical"
kind = "upstream"
upstream_profile = "default"

[[model_aliases]]
id = "unrelated-alias"
kind = "upstream"
upstream_profile = "default"
"#,
            grouped_fake.base_url
        ),
    )
    .await;

    let response = proxy_handler(
        State(proxy.state.for_listener(ListenerConfig {
            name: String::from("guard"),
            bind_host: String::from("127.0.0.1"),
            port: 18009,
            allowed_upstreams: None,
            upstream_profile: None,
        })),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models?test=distinct-multi-upstream-models")
            .body(Body::empty())
            .expect("grouped models request should build"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
        .await
        .expect("grouped models response should be readable");
    let models: serde_json::Value = serde_json::from_slice(&body).expect("models response JSON");
    let model_ids = models["data"]
        .as_array()
        .expect("models response data")
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(model_ids, vec!["unrelated-model", "unrelated-alias"]);
    let _ = default_fake.recv_next().await;
    let _ = grouped_fake.recv_next().await;
}

#[tokio::test]
async fn model_detail_uri_authority_rejects_invalid_or_reserved_paths_and_rewrites_forced_aliases()
{
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[upstream]
reserved_ingress_model_ids = ["reserved-canonical"]

[[forced_model_alias_profiles]]
alias = "public-forced-alias"
upstream_model = "canonical target"
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
    let listener = ListenerConfig {
        name: String::from("default"),
        bind_host: String::from("127.0.0.1"),
        port: 18000,
        allowed_upstreams: None,
        upstream_profile: None,
    };

    for (method, path) in [
        (Method::GET, "/v1/models/reserved-canonical"),
        (Method::DELETE, "/v1/models/reserved%2Dcanonical"),
    ] {
        let response = proxy_handler(
            State(proxy.state.for_listener(listener.clone())),
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .expect("reserved model detail request should build"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        let body = to_bytes(response.into_body(), MAX_PROXY_BODY_BYTES)
            .await
            .expect("reserved model detail error should be readable");
        assert!(String::from_utf8_lossy(&body).contains("reserved"));
        assert_no_upstream_request(&mut fake).await;
    }

    for path in [
        "/v1/models/malformed%ZZ",
        "/v1/models/ordinary%2Fmodel",
        "/v1/models/ordinary%5Cmodel",
        "/v1/models/ordinary-model/",
        "/v1/models/ordinary-model/extra",
    ] {
        let response = proxy_handler(
            State(proxy.state.for_listener(listener.clone())),
            Request::builder()
                .method(Method::GET)
                .uri(path)
                .body(Body::empty())
                .expect("invalid model detail request should build"),
        )
        .await;
        assert!(response.status().is_client_error(), "{path}");
        assert_no_upstream_request(&mut fake).await;
    }

    for path in ["/v1/models/ordinary-model", "/v1/models/ordinary%2Dmodel"] {
        let response = proxy_handler(
            State(proxy.state.for_listener(listener.clone())),
            Request::builder()
                .method(Method::GET)
                .uri(path)
                .body(Body::empty())
                .expect("ordinary model detail request should build"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        assert_eq!(fake.recv_next().await.path_and_query, path);
    }

    let response = proxy_handler(
        State(proxy.state.for_listener(listener)),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models/public-forced-alias?include=details")
            .body(Body::empty())
            .expect("forced alias model detail request should build"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        fake.recv_next().await.path_and_query,
        "/v1/models/canonical%20target?include=details",
        "forced model detail aliases must forward only the canonical percent-encoded ID"
    );
}

#[tokio::test]
async fn forced_alias_rewrites_sanitize_bound_headers_but_preserve_origin_authentication() {
    let mut fake = FakeUpstream::spawn().await;
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

    let ordinary = proxy_handler(
        State(proxy.state.clone()),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models/ordinary-model")
            .header(AUTHORIZATION, "Bearer preserved")
            .header("Signature", "sig-ordinary")
            .header("Digest", "sha-256=ordinary")
            .body(Body::empty())
            .expect("ordinary model detail request should build"),
    )
    .await;
    assert_eq!(ordinary.status(), StatusCode::NOT_FOUND);
    let ordinary = fake.recv_next().await;
    assert_eq!(
        ordinary.headers.get(AUTHORIZATION).unwrap(),
        "Bearer preserved"
    );
    assert_eq!(ordinary.headers.get("signature").unwrap(), "sig-ordinary");
    assert_eq!(ordinary.headers.get("digest").unwrap(), "sha-256=ordinary");

    let forced_chat = proxy_handler(
        State(proxy.state.clone()),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, "Bearer preserved")
            .header("Signature", "sig-forced")
            .header("Signature-Input", "sig-forced-input")
            .header("Digest", "sha-256=forced")
            .header("Content-Digest", "sha-256=:forced:")
            .header("Content-MD5", "forced")
            .header("ETag", "forced")
            .header("If-Match", "forced")
            .header("If-None-Match", "forced")
            .body(Body::from(
                r#"{"model":"public-forced-alias","messages":[],"stream":false}"#,
            ))
            .expect("forced chat request should build"),
    )
    .await;
    assert_eq!(forced_chat.status(), StatusCode::OK);
    let forced_chat = fake.recv_next().await;
    assert_eq!(
        forced_chat.headers.get(AUTHORIZATION).unwrap(),
        "Bearer preserved"
    );
    assert_rewritten_request_headers(&forced_chat.headers);
    assert_eq!(
        forced_chat.headers.get(CONTENT_LENGTH).unwrap(),
        forced_chat.body.len().to_string().as_str(),
        "rewritten body length must be recomputed"
    );

    let forced_detail = proxy_handler(
        State(proxy.state.clone()),
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models/public-forced-alias?include=details")
            .header(AUTHORIZATION, "Bearer preserved")
            .header("Signature", "sig-detail")
            .header("Digest", "sha-256=detail")
            .body(Body::empty())
            .expect("forced model detail request should build"),
    )
    .await;
    assert_eq!(forced_detail.status(), StatusCode::NOT_FOUND);
    let forced_detail = fake.recv_next().await;
    assert_eq!(
        forced_detail.path_and_query,
        "/v1/models/canonical-target?include=details"
    );
    assert_eq!(
        forced_detail.headers.get(AUTHORIZATION).unwrap(),
        "Bearer preserved"
    );
    assert!(forced_detail.headers.get("signature").is_none());
    assert!(forced_detail.headers.get("digest").is_none());
}

fn assert_rewritten_request_headers(headers: &HeaderMap) {
    for header in [
        "signature",
        "signature-input",
        "digest",
        "content-digest",
        "content-md5",
        "etag",
        "if-match",
        "if-none-match",
    ] {
        assert!(
            headers.get(header).is_none(),
            "rewritten body must not retain {header}"
        );
    }
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn guard_listener_hot_reload_updates_reserved_identity_with_alias_generation() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy =
        ProxyFixture::spawn_with_extra_config(&fake.base_url, FORCED_MODEL_ALIAS_PROFILES_CONFIG)
            .await;
    let guard_state = proxy.state.for_listener(ListenerConfig {
        name: String::from("guard"),
        bind_host: String::from("127.0.0.1"),
        port: 18009,
        allowed_upstreams: None,
        upstream_profile: None,
    });
    let reserved = |model| {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(format!(
                r#"{{"model":"{model}","messages":[]}}"#
            )))
            .expect("reservation request should build")
    };

    let response = proxy_handler(
        State(guard_state.clone()),
        reserved("abliterated-qwen-latest-27b-nvfp4"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_no_upstream_request(&mut fake).await;

    let replacement = proxy.root.join("config.next");
    let changed = std::fs::read_to_string(proxy.root.join("config.toml"))
        .expect("read fixture config")
        .replace(
            "abliterated-qwen-latest-27b-none",
            "abliterated-qwen-latest-27b-none-reloaded",
        )
        .replace(
            "abliterated-qwen-latest-27b-low",
            "abliterated-qwen-latest-27b-low-reloaded",
        )
        .replace(
            "abliterated-qwen-latest-27b-medium",
            "abliterated-qwen-latest-27b-medium-reloaded",
        )
        .replace(
            "abliterated-qwen-latest-27b-nvfp4",
            "test-reloaded-canonical",
        );
    std::fs::write(&replacement, &changed).expect("write changed generation");
    std::fs::rename(&replacement, proxy.root.join("config.toml"))
        .expect("publish changed generation");
    proxy
        .manager
        .reload()
        .expect("reload should apply atomically");

    let response = proxy_handler(
        State(guard_state.clone()),
        reserved("abliterated-qwen-latest-27b-nvfp4"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let old_canonical = fake.recv_next().await;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&old_canonical.body)
            .expect("old canonical request should be JSON")["model"],
        "abliterated-qwen-latest-27b-nvfp4"
    );

    let response = proxy_handler(
        State(guard_state.clone()),
        reserved("test-reloaded-canonical"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_no_upstream_request(&mut fake).await;

    let response = proxy_handler(
        State(guard_state.clone()),
        reserved("abliterated-qwen-latest-27b-none-reloaded"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let rewritten = fake.recv_next().await;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&rewritten.body)
            .expect("reloaded alias request should be JSON")["model"],
        "test-reloaded-canonical"
    );

    std::fs::write(
        &replacement,
        format!(
            "{changed}[[model_aliases]]\nid = \"abliterated-qwen-latest-27b-none-reloaded\"\nkind = \"upstream\"\nupstream_profile = \"default\"\n"
        ),
    )
    .expect("write invalid generation");
    std::fs::rename(&replacement, proxy.root.join("config.toml"))
        .expect("publish invalid generation");
    proxy
        .manager
        .reload()
        .expect_err("invalid reload must retain the complete last-good generation");
    let response = proxy_handler(State(guard_state), reserved("test-reloaded-canonical")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_no_upstream_request(&mut fake).await;
}

#[tokio::test]
async fn upstream_model_rewrites_request_and_response_model_names() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("rewritten model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let json = shielded_final_json(response).await;
    assert_eq!(json["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        json["model"], "alias-chat",
        "client-facing response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    assert_eq!(observed.path_and_query, "/v1/chat/completions");
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["model"], "aeon-ultimate",
        "upstream must receive the configured upstream_model rewrite"
    );
}

#[test]
fn forced_generation_controls_preserve_opaque_nested_extensions() {
    let mut body = forced_generation_controls_fixture();

    remove_forced_generation_controls(&mut body);

    assert_forced_generation_controls_removed(&body);
    assert_opaque_nested_extensions_preserved(&body);
}

fn forced_generation_controls_fixture() -> serde_json::Value {
    serde_json::json!({
        "temperature": 9,
        "max_tokens": 9,
        "extra_body": opaque_nested_extra_body(),
        "chat_template_kwargs": {
            "enable_thinking": true,
            "options": [{"temperature": 7, "max_tokens": 7, "vendor": {"top_p": "keep"}}],
            "vendor_extension": {
                "reasoning_effort": "keep",
                "temperature": "keep",
                "max_tokens": "keep",
                "required": ["temperature"]
            },
            "vendor_extensions": [{
                "reasoning_effort": "keep",
                "temperature": "keep",
                "max_tokens": "keep",
                "required": ["max_tokens"]
            }]
        },
        "thinking": {
            "enabled": true,
            "budget_tokens": 8,
            "vendor_extension": {
                "reasoning_effort": "keep",
                "temperature": "keep",
                "max_tokens": "keep"
            }
        }
    })
}

fn opaque_nested_extra_body() -> serde_json::Value {
    serde_json::json!({
        "temperature": 8,
        "max_tokens": 8,
        "options": [
            {
                "temperature": 7,
                "max_tokens": 7,
                "vendor_extension": {
                    "temperature": "keep",
                    "max_tokens": "keep",
                    "required": ["temperature"]
                }
            }
        ],
        "vendor_extension": {
            "reasoning_effort": "semantic-value",
            "max_tokens": 7,
            "temperature": 0.5,
            "required": ["max_tokens"]
        },
        "vendor_extensions": [
            {
                "reasoning_effort": "semantic-array-value",
                "max_tokens": 6,
                "temperature": 0.4,
                "required": ["temperature"]
            }
        ],
        "messages": [{
            "content": {"temperature": 0.1, "max_tokens": 3}
        }],
        "tools": [{
            "function": {
                "parameters": {
                    "properties": {"temperature": {"type": "number"}},
                    "required": ["temperature"],
                    "max_tokens": 3
                }
            }
        }],
        "response_format": {
            "json_schema": {
                "schema": {
                    "properties": {"max_tokens": {"type": "number"}},
                    "required": ["max_tokens"],
                    "temperature": 0.2
                }
            }
        },
        "chat_template_kwargs": {
            "temperature": 8,
            "options": [{"max_tokens": 7, "vendor": {"temperature": "keep"}}],
            "vendor_extension": {
                "temperature": "keep",
                "max_tokens": "keep",
                "required": ["max_tokens"]
            },
            "vendor_extensions": [{
                "temperature": "keep",
                "max_tokens": "keep",
                "required": ["temperature"]
            }]
        },
        "thinking": {
            "enabled": true,
            "budget_tokens": 8,
            "vendor_extension": {
                "thinking": "semantic-value",
                "temperature": "keep",
                "max_tokens": "keep"
            }
        }
    })
}

fn assert_forced_generation_controls_removed(body: &serde_json::Value) {
    for pointer in [
        "/temperature",
        "/max_tokens",
        "/extra_body/temperature",
        "/extra_body/max_tokens",
        "/extra_body/options/0/temperature",
        "/extra_body/options/0/max_tokens",
        "/extra_body/chat_template_kwargs/temperature",
        "/extra_body/chat_template_kwargs/options/0/max_tokens",
        "/extra_body/thinking/enabled",
        "/extra_body/thinking/budget_tokens",
        "/chat_template_kwargs/enable_thinking",
        "/chat_template_kwargs/options/0/temperature",
        "/thinking/enabled",
        "/thinking/budget_tokens",
    ] {
        assert!(body.pointer(pointer).is_none(), "must strip {pointer}");
    }
}

fn assert_opaque_nested_extensions_preserved(body: &serde_json::Value) {
    for (pointer, expected) in [
        (
            "/extra_body/vendor_extension/temperature",
            serde_json::json!(0.5),
        ),
        (
            "/extra_body/vendor_extension/max_tokens",
            serde_json::json!(7),
        ),
        (
            "/extra_body/vendor_extension/required/0",
            serde_json::json!("max_tokens"),
        ),
        (
            "/extra_body/vendor_extensions/0/temperature",
            serde_json::json!(0.4),
        ),
        (
            "/extra_body/vendor_extensions/0/max_tokens",
            serde_json::json!(6),
        ),
        (
            "/extra_body/chat_template_kwargs/vendor_extension/temperature",
            serde_json::json!("keep"),
        ),
        (
            "/extra_body/chat_template_kwargs/options/0/vendor/temperature",
            serde_json::json!("keep"),
        ),
        (
            "/extra_body/chat_template_kwargs/vendor_extensions/0/required/0",
            serde_json::json!("temperature"),
        ),
        (
            "/extra_body/thinking/vendor_extension/thinking",
            serde_json::json!("semantic-value"),
        ),
        (
            "/chat_template_kwargs/vendor_extension/reasoning_effort",
            serde_json::json!("keep"),
        ),
        (
            "/chat_template_kwargs/vendor_extensions/0/max_tokens",
            serde_json::json!("keep"),
        ),
        (
            "/thinking/vendor_extension/reasoning_effort",
            serde_json::json!("keep"),
        ),
        (
            "/extra_body/messages/0/content/temperature",
            serde_json::json!(0.1),
        ),
        (
            "/extra_body/tools/0/function/parameters/properties/temperature/type",
            serde_json::json!("number"),
        ),
        (
            "/extra_body/tools/0/function/parameters/required/0",
            serde_json::json!("temperature"),
        ),
        (
            "/extra_body/response_format/json_schema/schema/properties/max_tokens/type",
            serde_json::json!("number"),
        ),
        (
            "/extra_body/response_format/json_schema/schema/required/0",
            serde_json::json!("max_tokens"),
        ),
    ] {
        assert_eq!(
            body.pointer(pointer),
            Some(&expected),
            "must preserve {pointer}"
        );
    }
}

fn assert_forced_model_alias_wire_body(body: &serde_json::Value) {
    assert_eq!(body["model"], "abliterated-qwen-latest-27b-nvfp4");
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["top_k"], 20);
    assert_eq!(body["min_p"], 0.0);
    assert_eq!(body["presence_penalty"], 1.5);
    assert_eq!(body["repetition_penalty"], 1.0);
    assert_eq!(body["max_tokens"], 16384);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    for pointer in [
        "/reasoning_effort",
        "/model_reasoning_effort",
        "/thinking_token_budget",
        "/thinking_budget",
        "/enable_thinking",
        "/frequency_penalty",
        "/max_completion_tokens",
        "/extra_body/temperature",
        "/extra_body/max_tokens",
        "/extra_body/thinking/enabled",
        "/extra_body/thinking/budget_tokens",
        "/extra_body/chat_template_kwargs/enable_thinking",
        "/extra_body/chat_template_kwargs/thinking_budget",
    ] {
        assert!(body.pointer(pointer).is_none(), "must strip {pointer}");
    }
    assert_eq!(body["arbitrary"]["reasoning_effort"], "high");
    assert_eq!(body["arbitrary"]["max_output_tokens"], 8);
    assert_eq!(body["arbitrary"]["children"][0]["min_p"], 8);
    assert_eq!(body["arbitrary"]["children"][0]["output_tokens"], 8);
    assert_eq!(body["arbitrary"]["frequency_penalty"], 8);
    assert_eq!(body["arbitrary"]["children"][0]["frequency_penalty"], 8);
    assert_eq!(body["arbitrary"]["children"][1][0]["frequency_penalty"], 8);
}

async fn assert_forced_response_model(response: reqwest::Response, stream: bool, alias: &str) {
    assert_eq!(response.status(), StatusCode::OK);
    let response_body = response
        .bytes()
        .await
        .expect("forced response should drain");
    if stream {
        let body = std::str::from_utf8(&response_body).expect("forced SSE should be UTF-8");
        assert!(
            openai_sse_json_chunks(body)
                .iter()
                .filter_map(|chunk| chunk.get("model"))
                .all(|model| model == alias),
            "forced SSE response must restore the requested public alias"
        );
    } else {
        let body: serde_json::Value =
            serde_json::from_slice(&response_body).expect("forced JSON response");
        assert_eq!(body["model"], alias);
    }
}

async fn assert_forced_retry_attempts(
    fake: &mut FakeUpstream,
    thinking: bool,
    budget: Option<u64>,
    temperature: f64,
    top_p: f64,
    presence_penalty: f64,
) {
    let mut attempts = Vec::new();
    while attempts.len() < 4 {
        let observed = fake.recv_next().await;
        if observed
            .path_and_query
            .contains("test=shielded-429-then-two-503-then-success")
        {
            attempts.push(observed);
        }
    }
    for (attempt, observed) in attempts.into_iter().enumerate() {
        let body: serde_json::Value =
            serde_json::from_slice(&observed.body).expect("attempt body should be JSON");
        assert_eq!(
            body["model"],
            "abliterated-qwen-latest-27b-nvfp4",
            "forced alias must survive physical attempt {}",
            attempt + 1
        );
        assert_eq!(body["temperature"], temperature);
        assert_eq!(body["top_p"], top_p);
        assert_eq!(body["top_k"], 20);
        assert_eq!(body["min_p"], 0.0);
        assert_eq!(body["presence_penalty"], presence_penalty);
        assert_eq!(body["repetition_penalty"], 1.0);
        assert_eq!(body["max_tokens"], if thinking { 81920 } else { 16384 });
        assert_eq!(body["frequency_penalty"], serde_json::Value::Null);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], thinking);
        assert_eq!(
            body.get("thinking_token_budget")
                .and_then(serde_json::Value::as_u64),
            budget
        );
    }
}

async fn assert_ordinary_model_is_unchanged(proxy: &ProxyFixture, fake: &mut FakeUpstream) {
    let ordinary = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"ordinary-model","messages":[],"stream":true,"temperature":0.42,"top_p":0.73,"top_k":7,"min_p":0.12,"presence_penalty":0.3,"repetition_penalty":1.2}"#,
        )
        .send()
        .await
        .expect("ordinary request should complete");
    assert_eq!(ordinary.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("JSON upstream body");
    assert_eq!(body["model"], "ordinary-model");
    assert_eq!(body["temperature"], 0.42);
    assert_eq!(body["top_p"], 0.73);
    assert_eq!(body["top_k"], 7);
    assert_eq!(body["min_p"], 0.12);
    assert_eq!(body["presence_penalty"], 0.3);
    assert_eq!(body["repetition_penalty"], 1.2);
}

#[tokio::test]
async fn forced_model_alias_policy_rewrites_streaming_and_non_streaming_requests() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        r#"
[[forced_model_alias_profiles]]
alias = "abliterated-qwen-latest-27b-none"
upstream_model = "abliterated-qwen-latest-27b-nvfp4"
thinking_mode = "force_disable"
output_cap = 16384
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
"#,
    )
    .await;
    for stream in [false, true] {
        let response = proxy
            .client
            .post(format!("{}/v1/chat/completions", proxy.base_url))
            .header(CONTENT_TYPE, "application/json")
            .body(format!(
                r#"{{"model":"abliterated-qwen-latest-27b-none","messages":[{{"role":"user","content":"ping"}}],"stream":{stream},"reasoning_effort":"high","model_reasoning_effort":"high","thinking_token_budget":999,"thinking_budget":999,"enable_thinking":false,"temperature":9,"top_p":9,"top_k":9,"min_p":9,"presence_penalty":9,"frequency_penalty":9,"repetition_penalty":9,"max_completion_tokens":9,"extra_body":{{"temperature":8,"max_tokens":8,"frequency_penalty":8,"thinking":{{"enabled":true,"budget_tokens":8}},"chat_template_kwargs":{{"enable_thinking":true,"thinking_budget":8}}}},"arbitrary":{{"reasoning_effort":"high","max_output_tokens":8,"frequency_penalty":8,"children":[{{"min_p":8,"output_tokens":8,"frequency_penalty":8}},[{{"frequency_penalty":8}}]]}}}}"#
            ))
            .send()
            .await
            .expect("forced policy request should complete");
        assert_forced_response_model(response, stream, "abliterated-qwen-latest-27b-none").await;
        let observed = fake.recv_next().await;
        let body: serde_json::Value =
            serde_json::from_slice(&observed.body).expect("JSON upstream body");
        assert_forced_model_alias_wire_body(&body);
    }
    assert_ordinary_model_is_unchanged(&proxy, &mut fake).await;

    let replacement = proxy.root.join("config.next");
    let changed = std::fs::read_to_string(proxy.root.join("config.toml"))
        .expect("read fixture config")
        .replace("temperature = 0.7", "temperature = 0.3");
    std::fs::write(&replacement, &changed).expect("write changed generation");
    std::fs::rename(&replacement, proxy.root.join("config.toml"))
        .expect("publish changed generation");
    proxy.manager.reload().expect("hot reload should apply");
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"abliterated-qwen-latest-27b-none","messages":[],"stream":false}"#)
        .send()
        .await
        .expect("reloaded request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("JSON upstream body");
    assert_eq!(body["temperature"], 0.3);
    std::fs::write(
        &replacement,
        format!(
            "{changed}[[model_aliases]]\nid = \"abliterated-qwen-latest-27b-none\"\nkind = \"upstream\"\nupstream_profile = \"default\"\n"
        ),
    )
    .expect("write colliding generation");
    std::fs::rename(&replacement, proxy.root.join("config.toml"))
        .expect("publish invalid generation");
    proxy
        .manager
        .reload()
        .expect_err("invalid reload must fail");
    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"abliterated-qwen-latest-27b-none","messages":[],"stream":false}"#)
        .send()
        .await
        .expect("last-good request should complete");
    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("JSON upstream body");
    assert_eq!(body["temperature"], 0.3);
}

#[tokio::test]
async fn forced_model_alias_policy_survives_every_shielded_retry_attempt() {
    let mut fake = FakeUpstream::spawn().await;
    let config = format!(
        r#"
[retry]
max_attempts = 4
max_retry_after_secs = 1
shielded_streaming_enabled = true

[[retry.ladder]]
name = "high"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "medium"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "off"
thinking_mode = "force_disable"
max_tokens = 50000

{FORCED_MODEL_ALIAS_PROFILES_CONFIG}"#
    );
    let proxy = ProxyFixture::spawn_with_extra_config(&fake.base_url, &config).await;

    for (alias, thinking, budget, temperature, top_p, presence_penalty) in [
        (
            "abliterated-qwen-latest-27b-none",
            false,
            None,
            0.7,
            0.8,
            1.5,
        ),
        (
            "abliterated-qwen-latest-27b-low",
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
        (
            "abliterated-qwen-latest-27b-medium",
            true,
            Some(65536),
            1.0,
            0.95,
            0.0,
        ),
    ] {
        for stream in [false, true] {
            let response = proxy
                .client
                .post(format!(
                    "{}/v1/chat/completions?test=shielded-429-then-two-503-then-success&alias={alias}-{stream}",
                    proxy.base_url
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(format!(
                    r#"{{"model":"{alias}","messages":[],"stream":{stream},"temperature":9,"top_p":9,"top_k":9,"min_p":9,"presence_penalty":9,"frequency_penalty":9,"repetition_penalty":9}}"#
                ))
                .send()
                .await
                .expect("shielded ladder request should complete");
            assert_forced_response_model(response, stream, alias).await;

            assert_forced_retry_attempts(
                &mut fake,
                thinking,
                budget,
                temperature,
                top_p,
                presence_penalty,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn upstream_model_rewrites_shielded_streaming_response_model_name() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("rewritten streaming model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .bytes()
        .await
        .expect("rewritten streaming response should be readable");
    let body = std::str::from_utf8(&body).expect("rewritten streaming response should be UTF-8");
    let chunks = openai_sse_json_chunks(body);
    assert!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.get("model"))
            .all(|model| model == "alias-chat"),
        "client-facing SSE response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "aeon-ultimate");
}

#[tokio::test]
async fn upstream_model_rewrites_final_direct_relay_response_model_name() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
anti_loop_hint_enabled = false
shielded_streaming_enabled = true

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000

[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!(
            "{}/v1/chat/completions?test=loop-twice-then-success",
            proxy.base_url
        ))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":true}"#,
        )
        .send()
        .await
        .expect("direct relay request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .bytes()
        .await
        .expect("direct relay response should be readable");
    let body = std::str::from_utf8(&body).expect("direct relay response should be UTF-8");
    let chunks = openai_sse_json_chunks(body);
    assert!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.get("model"))
            .all(|model| model == "alias-chat"),
        "direct relay must not expose the configured upstream model"
    );

    for _ in 0..3 {
        let observed = fake.recv_next().await;
        let observed_body: serde_json::Value =
            serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
        assert_eq!(observed_body["model"], "aeon-ultimate");
    }
}

#[tokio::test]
async fn upstream_model_rewrites_terminal_forward_json_and_sse_response_models() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_options(
        &fake.base_url,
        true,
        AppConfig::default().server.max_in_flight_requests,
        &format!(
            r#"
[loop_guard]
mode = "enforce"
output_repeated_line_threshold = 4

[retry]
max_attempts = 3
shielded_streaming_enabled = true

[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    for (path, stream) in [
        ("terminal-forward-model-json", false),
        ("terminal-forward-model-sse", true),
    ] {
        let response = proxy
            .client
            .post(format!(
                "{}/v1/chat/completions?test={path}",
                proxy.base_url
            ))
            .header(CONTENT_TYPE, "application/json")
            .body(format!(
                r#"{{"model":"alias-chat","messages":[],"stream":{stream}}}"#
            ))
            .send()
            .await
            .expect("terminal forward response should complete");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(
            response.headers().get(CONTENT_LENGTH).is_none(),
            "model rewrite must remove stale upstream content length"
        );
        let body = response
            .bytes()
            .await
            .expect("terminal response should drain");
        if stream {
            assert!(
                openai_sse_json_chunks(std::str::from_utf8(&body).expect("SSE must be UTF-8"))
                    .iter()
                    .all(|chunk| chunk["model"] == "alias-chat"),
                "terminal SSE must restore the requested alias"
            );
        } else {
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).expect("JSON must parse")["model"],
                "alias-chat",
                "terminal JSON must restore the requested alias"
            );
        }
        let observed = fake.recv_next().await;
        let observed_body: serde_json::Value =
            serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
        assert_eq!(observed_body["model"], "aeon-ultimate");
    }
}

#[tokio::test]
async fn upstream_model_rewrites_generic_response_model_name() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-completion"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"alias-completion","prompt":"ping"}"#)
        .send()
        .await
        .expect("rewritten completion request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response_json(response).await["model"],
        "alias-completion",
        "generic client-facing response model must be restored to the requested alias"
    );

    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(observed_body["model"], "aeon-ultimate");
}

#[tokio::test]
async fn absent_upstream_model_keeps_request_model_passthrough() {
    let mut fake = FakeUpstream::spawn().await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "passthrough-profile"
base_url = "{}"
match_models = ["alias-chat"]
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .header(CONTENT_TYPE, "application/json")
        .body(
            r#"{"model":"alias-chat","messages":[{"role":"user","content":"ping"}],"stream":false}"#,
        )
        .send()
        .await
        .expect("passthrough model request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let observed = fake.recv_next().await;
    let observed_body: serde_json::Value =
        serde_json::from_slice(&observed.body).expect("upstream body should be JSON");
    assert_eq!(
        observed_body["model"], "alias-chat",
        "without upstream_model the client model name must pass through"
    );
}

#[tokio::test]
async fn upstream_model_rewrite_sse_buffer_overflows_with_controlled_error() {
    // Feed the rewriter a stream of bytes that exceeds the frame cap without
    // any SSE delimiter. The body must terminate with a controlled error
    // rather than buffering indefinitely.
    let cap = SSE_REWRITE_FRAME_BYTE_LIMIT;
    let oversized: Vec<Result<Bytes, std::io::Error>> = vec![
        Ok(Bytes::from(vec![b'A'; cap])),
        Ok(Bytes::from(vec![b'B'; cap])),
    ];
    let input = futures_util::stream::iter(oversized);
    let mut body = ResponseModelRewriteBody::new(
        input,
        ResponseModelRewriteMode::OpenAiSse,
        String::from("alias-chat"),
    );

    let mut saw_overflow = false;
    while let Some(result) = body.next().await {
        if let Err(ResponseModelRewriteError::FrameOverflow { .. }) = result {
            saw_overflow = true;
        }
    }
    assert!(
        saw_overflow,
        "stream must surface a controlled FrameOverflow error when the SSE \
         frame exceeds the byte cap without a delimiter"
    );
}

#[tokio::test]
async fn upstream_model_rewrite_sse_buffer_releases_after_overflow() {
    // After an overflow, the buffered bytes must be released (not retained
    // for future scans) and the stream must terminate quickly.
    let cap = SSE_REWRITE_FRAME_BYTE_LIMIT;
    let oversized: Vec<Result<Bytes, std::io::Error>> = vec![Ok(Bytes::from(vec![b'X'; cap + 1]))];
    let input = futures_util::stream::iter(oversized);
    let mut body = ResponseModelRewriteBody::new(
        input,
        ResponseModelRewriteMode::OpenAiSse,
        String::from("alias-chat"),
    );

    let mut count = 0;
    while let Some(result) = body.next().await {
        let _ = result;
        count += 1;
        assert!(
            count <= 10,
            "stream should terminate after bounded overflow, not loop"
        );
    }
    // The overflow should emit exactly one error then None — at most 2 polls.
    assert!(
        count <= 2,
        "overflowed stream must terminate quickly, got {count} items"
    );
}

#[tokio::test]
async fn configured_upstream_model_missing_from_models_list_does_not_synthesize_alias() {
    // upstream_model = "aeon-ultimate" but the upstream /v1/models response
    // only contains unrelated models. The proxy must NOT synthesize an alias
    // from the first unrelated model's metadata. We verify by checking that
    // no alias record appears in the response, and specifically that no record
    // copies "unrelated-model"'s metadata with the alias id.
    //
    // We use two upstream profiles so the default listener (no upstream_profile)
    // falls through to allowed_upstreams / first-profile routing, which keeps
    // the original model visible. The key assertion: no "alias-chat" record
    // synthesized from unrelated metadata.
    let models_body = r#"{"object":"list","data":[{"id":"unrelated-model","object":"model","max_model_len":128000,"owned_by":"vllm"}]}"#;
    let fake = FakeUpstream::spawn_with_models_body(models_body).await;
    let proxy = ProxyFixture::spawn_with_extra_config(
        &fake.base_url,
        &format!(
            r#"
[[upstreams]]
name = "rewriting-profile"
base_url = "{}"
match_models = ["alias-chat"]
upstream_model = "aeon-ultimate"
"#,
            fake.base_url
        ),
    )
    .await;

    let response = proxy
        .client
        .get(format!("{}/v1/models", proxy.base_url))
        .send()
        .await
        .expect("models request should complete");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .text()
        .await
        .expect("models body should be readable");
    let value: serde_json::Value = serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("models body should parse as JSON: {error}; body={body}"));
    let models = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .expect("models body should have a data array");

    let model_ids: Vec<&str> = models
        .iter()
        .filter_map(|model| model.get("id").and_then(serde_json::Value::as_str))
        .collect();
    assert!(
        !model_ids.contains(&"alias-chat"),
        "must not synthesize alias-chat from an unrelated model when the configured \
         upstream_model is missing from the upstream models list; got ids: {model_ids:?}"
    );
    // Additionally verify no synthesized alias records exist at all — if the
    // fix worked, the rewriter should not have copied any metadata.
    assert!(
        !models.iter().any(|model| {
            model
                .get("llm_guard_proxy_alias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        }),
        "no alias records should be synthesized when upstream_model is missing upstream"
    );
}
