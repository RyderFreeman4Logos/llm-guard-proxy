use std::collections::HashSet;

use axum::body::Bytes;
use llm_guard_proxy_core::{AppConfig, MetadataConfig};
use serde_json::{Map, Number, Value};

/// Enriches OpenAI-compatible model list responses with normalized context metadata.
///
/// Invalid JSON, non-list payloads, and responses without usable model metadata are returned
/// byte-for-byte so generic proxy behavior stays pass-through unless enrichment can help.
pub(super) fn enrich_models_body(
    config: &AppConfig,
    selected_metadata: &MetadataConfig,
    body: Bytes,
) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };

    let Some(models) = value.get_mut("data").and_then(Value::as_array_mut) else {
        return body;
    };

    let mut changed = false;
    for model in models {
        if let Some(record) = model.as_object_mut() {
            let metadata = metadata_for_model_record(config, selected_metadata, record);
            changed |= enrich_model_record(metadata, record);
        }
    }

    if !changed {
        return body;
    }

    serde_json::to_vec(&value).map_or(body, Bytes::from)
}

/// Keeps only model records whose `id` is accepted by `allow_model_id`.
///
/// Invalid JSON and non-list payloads are returned unchanged so filtering never
/// corrupts an upstream compatibility response.
pub(super) fn filter_models_body_by_id(
    body: Bytes,
    mut allow_model_id: impl FnMut(&str) -> bool,
) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    let Some(models) = value.get_mut("data").and_then(Value::as_array_mut) else {
        return body;
    };
    models.retain(|model| {
        model
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(&mut allow_model_id)
    });
    serde_json::to_vec(&value).map_or(body, Bytes::from)
}

/// Merges already-filtered OpenAI-compatible model list responses.
///
/// The first valid list supplies the response envelope. Model records are
/// de-duplicated by `id`, preserving the first occurrence in upstream order.
pub(super) fn merge_models_bodies(bodies: Vec<Bytes>) -> MergedModelsBody {
    let mut fallback_body = None;
    let mut seen = HashSet::new();
    let mut merged_models = Vec::new();
    let mut merged_value = None;

    for body in bodies {
        if fallback_body.is_none() {
            fallback_body = Some(body.clone());
        }
        let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };
        let Some(models) = value.get_mut("data").and_then(Value::as_array_mut) else {
            continue;
        };
        for model in std::mem::take(models) {
            push_model_once(model, &mut seen, &mut merged_models);
        }
        if merged_value.is_none() {
            merged_value = Some(value);
        }
    }

    let Some(mut merged_value) = merged_value else {
        return MergedModelsBody {
            body: fallback_body
                .unwrap_or_else(|| Bytes::from_static(br#"{"object":"list","data":[]}"#)),
            has_valid_model_list: false,
        };
    };
    let Some(models) = merged_value.get_mut("data").and_then(Value::as_array_mut) else {
        return MergedModelsBody {
            body: fallback_body
                .unwrap_or_else(|| Bytes::from_static(br#"{"object":"list","data":[]}"#)),
            has_valid_model_list: false,
        };
    };
    *models = merged_models;
    let body = serde_json::to_vec(&merged_value).map_or_else(
        |_| fallback_body.unwrap_or_else(|| Bytes::from_static(br#"{"object":"list","data":[]}"#)),
        Bytes::from,
    );
    MergedModelsBody {
        body,
        has_valid_model_list: true,
    }
}

pub(super) struct MergedModelsBody {
    pub(super) body: Bytes,
    pub(super) has_valid_model_list: bool,
}

fn push_model_once(model: Value, seen: &mut HashSet<String>, models: &mut Vec<Value>) {
    let Some(model_id) = model.get("id").and_then(Value::as_str) else {
        models.push(model);
        return;
    };
    if seen.insert(model_id.to_owned()) {
        models.push(model);
    }
}

fn metadata_for_model_record<'config>(
    config: &'config AppConfig,
    selected_metadata: &'config MetadataConfig,
    record: &Map<String, Value>,
) -> &'config MetadataConfig {
    let Some(model_id) = record.get("id").and_then(Value::as_str) else {
        return selected_metadata;
    };
    config
        .upstream_profiles
        .iter()
        .find(|profile| profile.matches_model(model_id))
        .map_or(selected_metadata, |profile| &profile.metadata)
}

fn enrich_model_record(config: &MetadataConfig, record: &mut Map<String, Value>) -> bool {
    if !config.discovery_enabled || !config.enrich_responses {
        return false;
    }

    let Some(context_length) =
        discovered_context_length(record).or_else(|| fallback_context_length(config))
    else {
        return false;
    };

    let mut changed = false;
    for field in ["context_length", "max_context_length", "max_model_len"] {
        changed |= set_u64_field(record, field, context_length);
    }
    changed
}

fn discovered_context_length(record: &Map<String, Value>) -> Option<u64> {
    numeric_field(record, "max_model_len")
        .or_else(|| numeric_field(record, "context_length"))
        .or_else(|| numeric_field(record, "max_context_length"))
}

fn fallback_context_length(config: &MetadataConfig) -> Option<u64> {
    config.context_window_override().map(u64::from)
}

fn numeric_field(record: &Map<String, Value>, key: &str) -> Option<u64> {
    record.get(key).and_then(Value::as_u64)
}

fn set_u64_field(record: &mut Map<String, Value>, key: &str, value: u64) -> bool {
    let value = Value::Number(Number::from(value));
    if record.get(key) == Some(&value) {
        return false;
    }

    record.insert(key.to_owned(), value);
    true
}
