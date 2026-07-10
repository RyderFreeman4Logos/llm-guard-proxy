//! Parse and validate score request JSON without losing Pydantic-compatible integers.

use std::collections::BTreeMap;

use axum::body::Bytes;
use serde_json::{Value, value::RawValue};

#[derive(Default)]
struct RawTopNState {
    seen: bool,
    last_lax: Option<LaxTopN>,
}

struct RawScoreObject<'a> {
    object: serde_json::Map<String, Value>,
    top_n: RawTopNState,
    preserved_fields: BTreeMap<String, &'a RawValue>,
}

impl<'de> serde::Deserialize<'de> for RawScoreObject<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawScoreObjectVisitor;

        impl<'de> serde::de::Visitor<'de> for RawScoreObjectVisitor {
            type Value = RawScoreObject<'de>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                let mut top_n = RawTopNState::default();
                let mut preserved_fields = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<&'de RawValue>()?;
                    if key == "top_n" {
                        parse_pydantic_json_value(value).map_err(serde::de::Error::custom)?;
                        top_n.seen = true;
                        top_n.last_lax = parse_lax_top_n_raw(value);
                    } else {
                        let (parsed, needs_preservation) =
                            parse_pydantic_json_value(value).map_err(serde::de::Error::custom)?;
                        if needs_preservation {
                            preserved_fields.insert(key.clone(), value);
                        } else {
                            preserved_fields.remove(&key);
                        }
                        object.insert(key, parsed);
                    }
                }
                Ok(RawScoreObject {
                    object,
                    top_n,
                    preserved_fields,
                })
            }
        }

        deserializer.deserialize_map(RawScoreObjectVisitor)
    }
}

struct RawTopNOnly(RawTopNState);

impl<'de> serde::Deserialize<'de> for RawTopNOnly {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawTopNOnlyVisitor;

        impl<'de> serde::de::Visitor<'de> for RawTopNOnlyVisitor {
            type Value = RawTopNOnly;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut top_n = RawTopNState::default();
                while let Some(key) = map.next_key::<String>()? {
                    if key == "top_n" {
                        let value = map.next_value::<&'de RawValue>()?;
                        top_n.seen = true;
                        top_n.last_lax = parse_lax_top_n_raw(value);
                    } else {
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
                Ok(RawTopNOnly(top_n))
            }
        }

        deserializer.deserialize_map(RawTopNOnlyVisitor)
    }
}

pub(super) struct ParsedScoreValue<'a> {
    pub(super) value: Value,
    pub(super) preserved_fields: BTreeMap<String, &'a RawValue>,
}

pub(super) fn parse_score_value(body: &Bytes) -> Result<ParsedScoreValue<'_>, String> {
    match serde_json::from_slice::<Value>(body) {
        Ok(mut value) if !contains_non_serde_integer(body) => {
            normalize_legacy_top_n_from_raw(body, &mut value)?;
            Ok(ParsedScoreValue {
                value,
                preserved_fields: BTreeMap::new(),
            })
        }
        Ok(_) => parse_score_value_from_raw(body, None),
        Err(error) => parse_score_value_from_raw(body, Some(&error)),
    }
}

pub(crate) fn model_id_from_score_body(body: &Bytes) -> Option<String> {
    parse_score_value(body).ok().and_then(|parsed| {
        parsed
            .value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

fn normalize_legacy_top_n_from_raw(body: &Bytes, value: &mut Value) -> Result<(), String> {
    let Some(object) = value.as_object_mut() else {
        return Ok(());
    };
    let is_legacy = !object.contains_key("text_1")
        && !object.contains_key("text_2")
        && object.contains_key("query")
        && object.contains_key("documents");
    let needs_raw = object
        .get("top_n")
        .and_then(Value::as_number)
        .and_then(serde_json::Number::as_f64)
        .is_some_and(|value| value <= I64_MIN_AS_F64 || value >= I64_MAX_AS_F64);
    if !is_legacy || !needs_raw {
        return Ok(());
    }

    let raw_top_n: RawTopNOnly =
        serde_json::from_slice(body).map_err(|error| format!("invalid score JSON: {error}"))?;
    let top_n = raw_top_n
        .0
        .last_lax
        .ok_or_else(|| String::from("score top_n must be a valid integer"))?;
    object.insert(String::from("top_n"), lax_top_n_representative(top_n));
    Ok(())
}

fn parse_score_value_from_raw<'a>(
    body: &'a Bytes,
    default_error: Option<&serde_json::Error>,
) -> Result<ParsedScoreValue<'a>, String> {
    let default_message = || {
        default_error.map_or_else(
            || String::from("invalid score JSON"),
            |error| format!("invalid score JSON: {error}"),
        )
    };
    validate_pydantic_json_nesting(body).map_err(|()| default_message())?;
    let mut raw_object: RawScoreObject<'_> =
        serde_json::from_slice(body).map_err(|_| default_message())?;
    let is_canonical =
        raw_object.object.contains_key("text_1") || raw_object.object.contains_key("text_2");
    let is_legacy = !is_canonical
        && raw_object.object.contains_key("query")
        && raw_object.object.contains_key("documents");
    let is_future =
        !is_canonical && !is_legacy && super::has_complete_future_score_shape(&raw_object.object);
    if !is_canonical && !is_legacy && !is_future {
        return Err(default_message());
    }

    if raw_object.top_n.seen {
        let top_n = if is_canonical {
            Value::Null
        } else {
            raw_object
                .top_n
                .last_lax
                .map(lax_top_n_representative)
                .ok_or_else(|| String::from("score top_n must be a valid integer"))?
        };
        raw_object.object.insert(String::from("top_n"), top_n);
    }
    Ok(ParsedScoreValue {
        value: Value::Object(raw_object.object),
        preserved_fields: raw_object.preserved_fields,
    })
}

fn validate_pydantic_json_nesting(body: &[u8]) -> Result<(), ()> {
    const MAX_CONTAINER_DEPTH: usize = 200;

    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in body {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > MAX_CONTAINER_DEPTH {
                    return Err(());
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

struct PydanticFallbackValue {
    value: Value,
    changed: bool,
}

impl<'de> serde::Deserialize<'de> for PydanticFallbackValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PydanticFallbackVisitor;

        impl<'de> serde::de::Visitor<'de> for PydanticFallbackVisitor {
            type Value = PydanticFallbackValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object or array containing a large integer")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                let mut changed = false;
                while let Some(key) = map.next_key::<String>()? {
                    let raw = map.next_value::<&'de RawValue>()?;
                    let (value, child_changed) =
                        parse_pydantic_json_value(raw).map_err(serde::de::Error::custom)?;
                    changed |= child_changed;
                    object.insert(key, value);
                }
                Ok(PydanticFallbackValue {
                    value: Value::Object(object),
                    changed,
                })
            }

            fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
            where
                S: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                let mut changed = false;
                while let Some(raw) = sequence.next_element::<&'de RawValue>()? {
                    let (value, child_changed) =
                        parse_pydantic_json_value(raw).map_err(serde::de::Error::custom)?;
                    changed |= child_changed;
                    values.push(value);
                }
                Ok(PydanticFallbackValue {
                    value: Value::Array(values),
                    changed,
                })
            }
        }

        deserializer.deserialize_any(PydanticFallbackVisitor)
    }
}

fn parse_pydantic_json_value(raw: &RawValue) -> Result<(Value, bool), String> {
    if contains_non_serde_integer(raw.get().as_bytes()) {
        let lexical = raw.get();
        let is_integer_token = lexical
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'-')
            && !lexical.contains(['.', 'e', 'E']);
        if is_integer_token {
            return classify_number_int(lexical)
                .map(|_| (Value::from(0), true))
                .ok_or_else(|| String::from("invalid score JSON integer"));
        }
        if lexical.starts_with(['{', '[']) {
            let parsed: PydanticFallbackValue = serde_json::from_str(lexical)
                .map_err(|error| format!("invalid score JSON: {error}"))?;
            return Ok((parsed.value, parsed.changed));
        }
    }

    match serde_json::from_str::<Value>(raw.get()) {
        Ok(value) => Ok((value, false)),
        Err(default_error) => {
            let lexical = raw.get();
            let is_integer_token = lexical
                .as_bytes()
                .first()
                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'-')
                && !lexical.contains(['.', 'e', 'E']);
            if is_integer_token && classify_number_int(lexical).is_some() {
                return Ok((Value::from(0), true));
            }
            if is_pydantic_nonfinite_float(lexical) {
                return Ok((Value::from(0), true));
            }
            if !lexical.starts_with(['{', '[']) {
                return Err(format!("invalid score JSON: {default_error}"));
            }

            let parsed: PydanticFallbackValue = serde_json::from_str(lexical)
                .map_err(|error| format!("invalid score JSON: {error}"))?;
            Ok((parsed.value, parsed.changed))
        }
    }
}

pub(super) fn contains_non_serde_integer(json: &[u8]) -> bool {
    let mut index = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    while index < json.len() {
        let byte = json[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            index += 1;
            continue;
        }
        if byte.is_ascii_digit() || byte == b'-' {
            let start = index;
            index += 1;
            while index < json.len()
                && (json[index].is_ascii_digit()
                    || matches!(json[index], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                index += 1;
            }
            if integer_exceeds_serde_range(&json[start..index]) {
                return true;
            }
            continue;
        }
        index += 1;
    }
    false
}

fn integer_exceeds_serde_range(token: &[u8]) -> bool {
    if token.contains(&b'.') || token.contains(&b'e') || token.contains(&b'E') {
        return false;
    }
    let (negative, digits) = token
        .strip_prefix(b"-")
        .map_or((false, token), |digits| (true, digits));
    let limit = if negative {
        b"9223372036854775808".as_slice()
    } else {
        b"18446744073709551615".as_slice()
    };
    digits.len() > limit.len() || (digits.len() == limit.len() && digits > limit)
}

fn parse_lax_top_n_raw(raw: &RawValue) -> Option<LaxTopN> {
    let lexical = raw.get();
    match lexical {
        "false" => Some(LaxTopN::NonPositive),
        "true" => Some(LaxTopN::Positive(1)),
        _ if lexical.starts_with('"') => serde_json::from_str::<String>(lexical)
            .ok()
            .and_then(|value| parse_lax_top_n_string(&value)),
        _ if lexical
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'-') =>
        {
            if lexical.contains(['.', 'e', 'E']) {
                lexical
                    .parse::<f64>()
                    .ok()
                    .and_then(classify_lax_integral_float)
            } else {
                classify_number_int(lexical)
            }
        }
        _ => None,
    }
}

fn lax_top_n_representative(top_n: LaxTopN) -> Value {
    match top_n {
        LaxTopN::NonPositive => Value::from(0),
        LaxTopN::Positive(value) => Value::from(value),
    }
}

fn is_pydantic_nonfinite_float(lexical: &str) -> bool {
    let starts_like_json_number = lexical
        .as_bytes()
        .first()
        .is_some_and(|byte| *byte == b'-' || byte.is_ascii_digit());
    starts_like_json_number
        && lexical
            .bytes()
            .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
        && lexical.parse::<f64>().is_ok_and(|value| !value.is_finite())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LaxTopN {
    NonPositive,
    Positive(usize),
}

const I64_MIN_AS_F64: f64 = -9_223_372_036_854_775_808.0;
const I64_MAX_AS_F64: f64 = 9_223_372_036_854_775_808.0;

/// Mirror Pydantic's lax `int` coercion for legacy rerank `top_n` values.
pub(super) fn has_valid_lax_top_n(value: &Value) -> bool {
    parse_lax_top_n(value).is_some()
}

fn parse_lax_top_n(value: &Value) -> Option<LaxTopN> {
    match value {
        Value::Bool(false) => Some(LaxTopN::NonPositive),
        Value::Bool(true) => Some(LaxTopN::Positive(1)),
        Value::Number(number) => parse_lax_top_n_number(number),
        Value::String(value) => parse_lax_top_n_string(value),
        _ => None,
    }
}

fn parse_lax_top_n_number(number: &serde_json::Number) -> Option<LaxTopN> {
    number
        .as_i64()
        .map(|value| {
            if value <= 0 {
                LaxTopN::NonPositive
            } else {
                LaxTopN::Positive(usize::try_from(value).unwrap_or(usize::MAX))
            }
        })
        .or_else(|| {
            number.as_u64().map(|value| {
                if value == 0 {
                    LaxTopN::NonPositive
                } else {
                    LaxTopN::Positive(usize::try_from(value).unwrap_or(usize::MAX))
                }
            })
        })
        .or_else(|| {
            let lexical = number.to_string();
            if lexical.contains(['.', 'e', 'E']) {
                number.as_f64().and_then(classify_lax_integral_float)
            } else {
                classify_number_int(&lexical)
            }
        })
}

pub(super) fn parse_lax_positive_top_n(value: &Value) -> Option<usize> {
    match parse_lax_top_n(value)? {
        LaxTopN::NonPositive => None,
        LaxTopN::Positive(value) => Some(value),
    }
}

pub(super) fn parse_lax_top_n_string(value: &str) -> Option<LaxTopN> {
    if let Some(parsed) = classify_number_int(value) {
        return Some(parsed);
    }

    let len_before = value.len();
    let mut cleaned = value.trim();
    if let Some(suffix) = cleaned.strip_prefix('+') {
        if suffix.starts_with('-') {
            return None;
        }
        cleaned = suffix;
    }

    let mut is_negative = false;
    if let Some(suffix) = cleaned.strip_prefix('-') {
        if suffix.starts_with('-') || suffix.starts_with('+') {
            return None;
        }
        is_negative = true;
        cleaned = suffix;
    }

    cleaned = strip_pydantic_leading_zeros(cleaned)?;
    if let Some(dot) = cleaned.find('.') {
        let decimal = &cleaned[dot + 1..];
        if !decimal.is_empty() && decimal.bytes().all(|byte| byte == b'0') {
            cleaned = &cleaned[..dot];
        }
    }

    if cleaned.contains('_')
        && !cleaned.starts_with('_')
        && !cleaned.ends_with('_')
        && !cleaned.contains("__")
    {
        return classify_cleaned_number_int(&cleaned.replace('_', ""), is_negative);
    }
    if len_before == cleaned.len() {
        return None;
    }
    classify_cleaned_number_int(cleaned, is_negative)
}

fn classify_cleaned_number_int(value: &str, is_negative: bool) -> Option<LaxTopN> {
    if is_negative {
        classify_number_int(&format!("-{value}"))
    } else {
        classify_number_int(value)
    }
}

fn classify_number_int(value: &str) -> Option<LaxTopN> {
    const MAX_NUMBER_INT_DIGITS: usize = 4_300;
    let (is_negative, magnitude) = value
        .strip_prefix('-')
        .map_or((false, value), |magnitude| (true, magnitude));
    if magnitude.len() > MAX_NUMBER_INT_DIGITS {
        return None;
    }
    if magnitude.is_empty()
        || !magnitude.bytes().all(|byte| byte.is_ascii_digit())
        || (magnitude.len() > 1 && magnitude.starts_with('0'))
    {
        return None;
    }
    if is_negative {
        return Some(LaxTopN::NonPositive);
    }
    let value = magnitude.parse().unwrap_or(usize::MAX);
    if value == 0 {
        Some(LaxTopN::NonPositive)
    } else {
        Some(LaxTopN::Positive(value))
    }
}

fn strip_pydantic_leading_zeros(value: &str) -> Option<&str> {
    let mut chars = value.char_indices();
    match chars.next() {
        Some((_, '0')) => {}
        Some((_, '1'..='9' | '-')) => return Some(value),
        _ => return None,
    }
    for (index, character) in chars {
        match character {
            '0' | '_' => {}
            '1'..='9' | '-' => return Some(&value[index..]),
            '.' => return value.get(index.checked_sub(1)?..),
            _ => return None,
        }
    }
    Some(&value[value.len() - 1..])
}

fn classify_lax_integral_float(value: f64) -> Option<LaxTopN> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value <= I64_MIN_AS_F64
        || value >= I64_MAX_AS_F64
    {
        return None;
    }
    if value <= 0.0 {
        Some(LaxTopN::NonPositive)
    } else {
        Some(LaxTopN::Positive(
            format!("{value:.0}").parse().unwrap_or(usize::MAX),
        ))
    }
}
