//! Parse and validate score request JSON without losing Pydantic-compatible integers.

use std::collections::BTreeMap;

use axum::body::Bytes;
use serde_json::{Value, value::RawValue};

#[cfg(test)]
thread_local! {
    static PYDANTIC_VALUE_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_pydantic_value_parse_count() {
    PYDANTIC_VALUE_PARSE_COUNT.set(0);
}

#[cfg(test)]
pub(super) fn pydantic_value_parse_count() -> usize {
    PYDANTIC_VALUE_PARSE_COUNT.get()
}

#[derive(Default)]
struct RawTopNState {
    seen: bool,
    last_lax: Option<LaxTopN>,
}

struct RawScoreObject {
    object: serde_json::Map<String, Value>,
    top_n: RawTopNState,
    preserved_fields: BTreeMap<String, String>,
}

impl<'de> serde::Deserialize<'de> for RawScoreObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawScoreObjectVisitor;

        impl<'de> serde::de::Visitor<'de> for RawScoreObjectVisitor {
            type Value = RawScoreObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                let mut top_n = RawTopNState::default();
                let mut mapped_fields = BTreeMap::new();
                let mut preserved_fields = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<&'de RawValue>()?;
                    if key == "top_n" {
                        top_n.seen = true;
                        top_n.last_lax = parse_lax_top_n_raw(value);
                    } else if matches!(
                        key.as_str(),
                        "text_1" | "text_2" | "query" | "documents" | "model"
                    ) {
                        // Match FastAPI's JSON-object last-key-wins behavior without
                        // recursively materializing shadowed mapped values.
                        mapped_fields.insert(key, value.get().to_owned());
                    } else {
                        // These fields use the same Pydantic schema after score→rerank
                        // rewriting. Preserve the whole raw value so deeply nested extras
                        // stay O(body size) and the target performs the canonical coercion.
                        object.insert(key.clone(), Value::Null);
                        preserved_fields.insert(key, value.get().to_owned());
                    }
                }
                let is_canonical =
                    mapped_fields.contains_key("text_1") || mapped_fields.contains_key("text_2");
                for (key, lexical) in mapped_fields {
                    // Canonical score fields take precedence; legacy query/documents
                    // aliases are discarded and must not be recursively materialized.
                    if is_canonical && matches!(key.as_str(), "query" | "documents") {
                        continue;
                    }
                    let (parsed, needs_preservation) =
                        parse_pydantic_json_lexical(&lexical).map_err(serde::de::Error::custom)?;
                    if needs_preservation {
                        preserved_fields.insert(key.clone(), lexical);
                    }
                    object.insert(key, parsed);
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

pub(super) struct ParsedScoreValue {
    pub(super) value: Value,
    pub(super) preserved_fields: BTreeMap<String, String>,
}

pub(super) fn parse_score_value(body: &Bytes) -> Result<ParsedScoreValue, String> {
    if contains_pydantic_invalid_integer(body) {
        return Err(String::from("invalid score JSON integer"));
    }
    match serde_json::from_slice::<Value>(body) {
        Ok(mut value)
            if !contains_non_serde_integer(body)
                && !contains_pydantic_json_float(body)
                && !contains_json_nonfinite(body) =>
        {
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
    if !is_legacy || !object.contains_key("top_n") {
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

fn parse_score_value_from_raw(
    body: &Bytes,
    default_error: Option<&serde_json::Error>,
) -> Result<ParsedScoreValue, String> {
    let default_message = || {
        default_error.map_or_else(
            || String::from("invalid score JSON"),
            |error| format!("invalid score JSON: {error}"),
        )
    };
    let original = std::str::from_utf8(body).map_err(|_| default_message())?;
    let sanitized_owned;
    let parse_input = if contains_json_nonfinite(body) {
        sanitized_owned = sanitize_json_nonfinite(original);
        sanitized_owned.as_str()
    } else {
        original
    };
    let mut raw_object: RawScoreObject =
        serde_json::from_str(parse_input).map_err(|_| default_message())?;
    if !std::ptr::eq(parse_input, original) {
        // Length-preserving nonfinite sanitization keeps structure offsets aligned, but
        // preserved extras must round-trip the original FastAPI/Pydantic lexemes.
        if let Ok(originals) = top_level_field_raws(original) {
            for (key, raw) in &mut raw_object.preserved_fields {
                if let Some(orig) = originals.get(key) {
                    raw.clone_from(orig);
                }
            }
        }
    }
    let is_canonical =
        raw_object.object.contains_key("text_1") || raw_object.object.contains_key("text_2");
    let is_legacy = !is_canonical
        && raw_object.object.contains_key("query")
        && raw_object.object.contains_key("documents");
    let is_opaque_passthrough = !is_canonical && !is_legacy;

    if raw_object.top_n.seen {
        let top_n = if is_canonical || is_opaque_passthrough {
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
                        parse_pydantic_json_lexical(raw.get()).map_err(serde::de::Error::custom)?;
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
                        parse_pydantic_json_lexical(raw.get()).map_err(serde::de::Error::custom)?;
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

fn parse_pydantic_json_lexical(lexical: &str) -> Result<(Value, bool), String> {
    #[cfg(test)]
    PYDANTIC_VALUE_PARSE_COUNT.set(PYDANTIC_VALUE_PARSE_COUNT.get() + 1);
    if contains_non_serde_integer(lexical.as_bytes())
        || contains_pydantic_json_float(lexical.as_bytes())
    {
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
        if let Some(value) = parse_pydantic_json_float(lexical) {
            if value.is_finite() {
                return Ok((Value::from(value), false));
            }
            return Ok((Value::from(0), true));
        }
        if lexical.starts_with(['{', '[']) {
            let parsed: PydanticFallbackValue = serde_json::from_str(lexical)
                .map_err(|error| format!("invalid score JSON: {error}"))?;
            return Ok((parsed.value, parsed.changed));
        }
    }

    match serde_json::from_str::<Value>(lexical) {
        Ok(value) => Ok((value, false)),
        Err(default_error) => {
            let is_integer_token = lexical
                .as_bytes()
                .first()
                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'-')
                && !lexical.contains(['.', 'e', 'E']);
            if is_integer_token && classify_number_int(lexical).is_some() {
                return Ok((Value::from(0), true));
            }
            if let Some(value) = parse_pydantic_json_float(lexical) {
                if value.is_finite() {
                    return Ok((Value::from(value), false));
                }
                return Ok((Value::from(0), true));
            }
            if matches!(lexical, "NaN" | "Infinity" | "-Infinity") {
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

fn contains_json_nonfinite(json: &[u8]) -> bool {
    find_json_nonfinite(json).is_some()
}

fn find_json_nonfinite(json: &[u8]) -> Option<(usize, &'static str)> {
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
        if json[index..].starts_with(b"-Infinity")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 9)
        {
            return Some((index, "-Infinity"));
        }
        if json[index..].starts_with(b"Infinity")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 8)
        {
            return Some((index, "Infinity"));
        }
        if json[index..].starts_with(b"NaN")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 3)
        {
            return Some((index, "NaN"));
        }
        index += 1;
    }
    None
}

fn is_json_token_boundary(json: &[u8], index: usize) -> bool {
    if index == usize::MAX {
        return true;
    }
    match json.get(index) {
        None => true,
        Some(byte) => !byte.is_ascii_alphanumeric() && *byte != b'_',
    }
}

fn sanitize_json_nonfinite(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            out.push(byte);
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
            out.push(byte);
            in_string = true;
            index += 1;
            continue;
        }
        if bytes[index..].starts_with(b"-Infinity")
            && is_json_token_boundary(bytes, index.wrapping_sub(1))
            && is_json_token_boundary(bytes, index + 9)
        {
            out.extend_from_slice(b"-0.000000");
            index += 9;
            continue;
        }
        if bytes[index..].starts_with(b"Infinity")
            && is_json_token_boundary(bytes, index.wrapping_sub(1))
            && is_json_token_boundary(bytes, index + 8)
        {
            out.extend_from_slice(b"0.000000");
            index += 8;
            continue;
        }
        if bytes[index..].starts_with(b"NaN")
            && is_json_token_boundary(bytes, index.wrapping_sub(1))
            && is_json_token_boundary(bytes, index + 3)
        {
            out.extend_from_slice(b"0.0");
            index += 3;
            continue;
        }
        out.push(byte);
        index += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_owned())
}

fn top_level_field_raws(json: &str) -> Result<BTreeMap<String, String>, String> {
    let bytes = json.as_bytes();
    let mut index = skip_ws(bytes, 0);
    if bytes.get(index) != Some(&b'{') {
        return Err(String::from("score body must be a JSON object"));
    }
    index += 1;
    let mut fields = BTreeMap::new();
    loop {
        index = skip_ws(bytes, index);
        match bytes.get(index) {
            Some(&b'}') => break,
            Some(&b',') => {
                index += 1;
                continue;
            }
            Some(&b'"') => {}
            _ => return Err(String::from("invalid score JSON object key")),
        }
        let (key, after_key) = parse_json_string(json, index)?;
        index = skip_ws(bytes, after_key);
        if bytes.get(index) != Some(&b':') {
            return Err(String::from("invalid score JSON object entry"));
        }
        index = skip_ws(bytes, index + 1);
        let (raw, after_value) = capture_json_value(json, index)?;
        fields.insert(key, raw);
        index = after_value;
    }
    Ok(fields)
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        index += 1;
    }
    index
}

fn parse_json_string(json: &str, start: usize) -> Result<(String, usize), String> {
    let bytes = json.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return Err(String::from("invalid score JSON string"));
    }
    let mut index = start + 1;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            let raw = &json[start..=index];
            let value: String = serde_json::from_str(raw)
                .map_err(|error| format!("invalid score JSON string: {error}"))?;
            return Ok((value, index + 1));
        }
        index += 1;
    }
    Err(String::from("unterminated score JSON string"))
}

fn capture_json_value(json: &str, start: usize) -> Result<(String, usize), String> {
    let bytes = json.as_bytes();
    let Some(first) = bytes.get(start).copied() else {
        return Err(String::from("invalid score JSON value"));
    };
    if first == b'"' {
        let mut index = start + 1;
        let mut escaped = false;
        while index < bytes.len() {
            let byte = bytes[index];
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                return Ok((json[start..=index].to_owned(), index + 1));
            }
            index += 1;
        }
        return Err(String::from("unterminated score JSON string value"));
    }
    if first == b'{' || first == b'[' {
        let opening = first;
        let closing = if first == b'{' { b'}' } else { b']' };
        let mut depth = 1_usize;
        let mut index = start + 1;
        let mut in_string = false;
        let mut escaped = false;
        while index < bytes.len() {
            let byte = bytes[index];
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
            match byte {
                b'"' => in_string = true,
                b if b == opening => depth += 1,
                b if b == closing => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((json[start..=index].to_owned(), index + 1));
                    }
                }
                _ => {}
            }
            index += 1;
        }
        return Err(String::from("unterminated score JSON container"));
    }
    if json[start..].starts_with("-Infinity") {
        return Ok((String::from("-Infinity"), start + 9));
    }
    if json[start..].starts_with("Infinity") {
        return Ok((String::from("Infinity"), start + 8));
    }
    if json[start..].starts_with("NaN") {
        return Ok((String::from("NaN"), start + 3));
    }
    if json[start..].starts_with("null") {
        return Ok((String::from("null"), start + 4));
    }
    if json[start..].starts_with("true") {
        return Ok((String::from("true"), start + 4));
    }
    if json[start..].starts_with("false") {
        return Ok((String::from("false"), start + 5));
    }
    if first == b'-' || first.is_ascii_digit() {
        let mut index = start + 1;
        while index < bytes.len()
            && matches!(bytes[index], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        {
            index += 1;
        }
        return Ok((json[start..index].to_owned(), index));
    }
    Err(String::from("invalid score JSON value token"))
}

pub(super) fn contains_non_serde_integer(json: &[u8]) -> bool {
    contains_json_number_matching(json, integer_exceeds_serde_range)
}

fn contains_pydantic_json_float(json: &[u8]) -> bool {
    contains_json_number_matching(json, |token| {
        token.contains(&b'.') || token.contains(&b'e') || token.contains(&b'E')
    })
}

fn contains_pydantic_invalid_integer(json: &[u8]) -> bool {
    const MAX_NUMBER_INT_CHARACTERS: usize = 4_300;
    contains_json_number_matching(json, |token| {
        let magnitude = token.strip_prefix(b"-").unwrap_or(token);
        !token.contains(&b'.')
            && !token.contains(&b'e')
            && !token.contains(&b'E')
            && magnitude.len() > MAX_NUMBER_INT_CHARACTERS
    })
}

fn contains_json_number_matching(json: &[u8], predicate: impl Fn(&[u8]) -> bool) -> bool {
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
            if predicate(&json[start..index]) {
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
                lexical_core::parse::<f64>(lexical.as_bytes())
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

fn parse_pydantic_json_float(lexical: &str) -> Option<f64> {
    let starts_like_json_number = lexical
        .as_bytes()
        .first()
        .is_some_and(|byte| *byte == b'-' || byte.is_ascii_digit());
    (starts_like_json_number
        && lexical
            .bytes()
            .any(|byte| matches!(byte, b'.' | b'e' | b'E')))
    .then(|| lexical_core::parse::<f64>(lexical.as_bytes()).ok())
    .flatten()
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
    if let Some(parsed) = classify_string_number_int(value) {
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
        classify_string_number_int(&format!("-{value}"))
    } else {
        classify_string_number_int(value)
    }
}

fn classify_string_number_int(value: &str) -> Option<LaxTopN> {
    const MAX_STRING_INT_CHARACTERS: usize = 4_300;
    if value.starts_with('-') && value.len() > MAX_STRING_INT_CHARACTERS {
        return None;
    }
    classify_number_int(value)
}

fn classify_number_int(value: &str) -> Option<LaxTopN> {
    const MAX_NUMBER_INT_CHARACTERS: usize = 4_300;
    let (is_negative, magnitude) = value
        .strip_prefix('-')
        .map_or((false, value), |magnitude| (true, magnitude));
    if magnitude.len() > MAX_NUMBER_INT_CHARACTERS
        || magnitude.is_empty()
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
