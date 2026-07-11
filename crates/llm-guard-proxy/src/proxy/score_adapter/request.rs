//! Parse and validate score request JSON without losing Pydantic-compatible integers.

use std::{borrow::Cow, collections::BTreeMap};

use axum::body::Bytes;
use serde_json::{Value, value::RawValue};

const MAX_NUMBER_INT_CHARACTERS: usize = 4_300;

#[cfg(test)]
thread_local! {
    static PYDANTIC_VALUE_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RAW_LEXICAL_SCAN_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(in crate::proxy) fn reset_pydantic_value_parse_count() {
    PYDANTIC_VALUE_PARSE_COUNT.set(0);
}

#[cfg(test)]
pub(in crate::proxy) fn pydantic_value_parse_count() -> usize {
    PYDANTIC_VALUE_PARSE_COUNT.get()
}

#[cfg(test)]
pub(super) fn reset_raw_lexical_scan_bytes() {
    RAW_LEXICAL_SCAN_BYTES.set(0);
}

#[cfg(test)]
pub(super) fn raw_lexical_scan_bytes() -> usize {
    RAW_LEXICAL_SCAN_BYTES.get()
}

fn record_lexical_scan(bytes: usize) {
    #[cfg(test)]
    RAW_LEXICAL_SCAN_BYTES.set(RAW_LEXICAL_SCAN_BYTES.get().saturating_add(bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn deserialize_lexical<'de, T>(lexical: &'de str) -> Result<T, serde_json::Error>
where
    T: serde::Deserialize<'de>,
{
    record_lexical_scan(lexical.len());
    serde_json::from_str(lexical)
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
                    let parsed =
                        if matches!(key.as_str(), "text_1" | "text_2" | "query" | "documents") {
                            parse_score_input_lexical(&lexical)
                        } else {
                            parse_pydantic_json_lexical(&lexical)
                        }
                        .map_err(serde::de::Error::custom)?;
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

pub(super) struct ParsedScoreValue {
    pub(super) value: Value,
    pub(super) preserved_fields: BTreeMap<String, String>,
}

pub(super) fn parse_score_value(body: &Bytes) -> Result<ParsedScoreValue, String> {
    let original = std::str::from_utf8(body).map_err(|_| String::from("invalid score JSON"))?;
    let normalized = normalize_score_json(original)?;
    let was_normalized = matches!(&normalized, Cow::Owned(_));
    parse_score_value_from_raw(original, normalized.as_ref(), was_normalized)
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

fn parse_score_value_from_raw(
    original: &str,
    parse_input: &str,
    was_normalized: bool,
) -> Result<ParsedScoreValue, String> {
    let mut raw_object: RawScoreObject =
        deserialize_lexical(parse_input).map_err(|error| format!("invalid score JSON: {error}"))?;
    if was_normalized {
        // Length-preserving numeric normalization keeps structure offsets aligned, but
        // preserved extras must round-trip the original FastAPI/Pydantic lexemes.
        if let Ok(originals) = top_level_field_raws(original) {
            for (key, raw) in &mut raw_object.preserved_fields {
                if let Some(orig) = originals.get(key) {
                    raw.clone_from(orig);
                }
            }
            if let Some(top_n) = originals.get("top_n") {
                raw_object.top_n.seen = true;
                raw_object.top_n.last_lax = parse_lax_top_n_lexical(top_n);
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

struct RawObject(BTreeMap<String, String>);

impl<'de> serde::Deserialize<'de> for RawObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawObjectVisitor;

        impl<'de> serde::de::Visitor<'de> for RawObjectVisitor {
            type Value = RawObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut fields = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    let raw = map.next_value::<&'de RawValue>()?;
                    fields.insert(key, raw.get().to_owned());
                }
                Ok(RawObject(fields))
            }
        }

        deserializer.deserialize_map(RawObjectVisitor)
    }
}

struct RawArray(Vec<String>);

impl<'de> serde::Deserialize<'de> for RawArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawArrayVisitor;

        impl<'de> serde::de::Visitor<'de> for RawArrayVisitor {
            type Value = RawArray;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON array")
            }

            fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
            where
                S: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(raw) = sequence.next_element::<&'de RawValue>()? {
                    values.push(raw.get().to_owned());
                }
                Ok(RawArray(values))
            }
        }

        deserializer.deserialize_seq(RawArrayVisitor)
    }
}

fn parse_score_input_lexical(lexical: &str) -> Result<Value, String> {
    if lexical.trim_start().starts_with('{') {
        return parse_multimodal_object(lexical);
    }
    parse_pydantic_json_lexical(lexical)
}

fn parse_multimodal_object(lexical: &str) -> Result<Value, String> {
    let RawObject(fields) =
        deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))?;
    let mut object = serde_json::Map::new();
    for (key, raw) in fields {
        let value = if key == "content" {
            parse_content_parts(&raw)?
        } else {
            Value::Null
        };
        object.insert(key, value);
    }
    Ok(Value::Object(object))
}

fn parse_content_parts(lexical: &str) -> Result<Value, String> {
    let RawArray(parts) =
        deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))?;
    parts
        .into_iter()
        .map(|part| parse_content_part(&part))
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn parse_content_part(lexical: &str) -> Result<Value, String> {
    let RawObject(fields) =
        deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))?;
    let mut object = serde_json::Map::new();
    for (key, raw) in fields {
        let value = match key.as_str() {
            "type" | "text" | "uuid" => parse_required_value(&raw)?,
            "image_url" | "video_url" => parse_url_object(&raw)?,
            "image_embeds" => parse_image_embeds(&raw)?,
            _ => Value::Null,
        };
        object.insert(key, value);
    }
    Ok(Value::Object(object))
}

fn parse_url_object(lexical: &str) -> Result<Value, String> {
    if !lexical.trim_start().starts_with('{') {
        return parse_required_value(lexical);
    }
    let RawObject(fields) =
        deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))?;
    let mut object = serde_json::Map::new();
    for (key, raw) in fields {
        let value = if matches!(key.as_str(), "url" | "detail") {
            parse_required_value(&raw)?
        } else {
            Value::Null
        };
        object.insert(key, value);
    }
    Ok(Value::Object(object))
}

fn parse_image_embeds(lexical: &str) -> Result<Value, String> {
    if !lexical.trim_start().starts_with('{') {
        return parse_required_value(lexical);
    }
    let RawObject(fields) =
        deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))?;
    fields
        .into_iter()
        .map(|(key, raw)| parse_required_value(&raw).map(|value| (key, value)))
        .collect::<Result<serde_json::Map<_, _>, _>>()
        .map(Value::Object)
}

fn parse_required_value(lexical: &str) -> Result<Value, String> {
    parse_pydantic_json_lexical(lexical)
}

fn parse_pydantic_json_lexical(lexical: &str) -> Result<Value, String> {
    #[cfg(test)]
    PYDANTIC_VALUE_PARSE_COUNT.set(PYDANTIC_VALUE_PARSE_COUNT.get() + 1);
    deserialize_lexical(lexical).map_err(|error| format!("invalid score JSON: {error}"))
}

/// Validate score JSON resource bounds and normalize Pydantic-only numeric tokens once.
///
/// The returned owned string, when needed, has exactly the input length. Replaced number
/// tokens become zero followed by JSON whitespace so raw top-level field offsets remain valid.
fn normalize_score_json(input: &str) -> Result<Cow<'_, str>, String> {
    let json = input.as_bytes();
    let mut normalized: Option<Vec<u8>> = None;
    let mut index = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut closing_stack = Vec::with_capacity(super::MAX_SCORE_JSON_DEPTH);
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
        if matches!(byte, b'{' | b'[') {
            closing_stack.push(if byte == b'{' { b'}' } else { b']' });
            if closing_stack.len() > super::MAX_SCORE_JSON_DEPTH {
                record_lexical_scan(index + 1);
                return Err(format!(
                    "score JSON exceeds maximum structure depth of {}",
                    super::MAX_SCORE_JSON_DEPTH
                ));
            }
            index += 1;
            continue;
        }
        if matches!(byte, b'}' | b']') {
            if closing_stack.pop() != Some(byte) {
                record_lexical_scan(index + 1);
                return Err(String::from("invalid score JSON container"));
            }
            index += 1;
            continue;
        }
        if json[index..].starts_with(b"-Infinity")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 9)
        {
            normalize_number_token(&mut normalized, json, index, index + 9);
            index += 9;
            continue;
        }
        if json[index..].starts_with(b"Infinity")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 8)
        {
            normalize_number_token(&mut normalized, json, index, index + 8);
            index += 8;
            continue;
        }
        if json[index..].starts_with(b"NaN")
            && is_json_token_boundary(json, index.wrapping_sub(1))
            && is_json_token_boundary(json, index + 3)
        {
            normalize_number_token(&mut normalized, json, index, index + 3);
            index += 3;
            continue;
        }
        if byte.is_ascii_digit() || byte == b'-' {
            let start = index;
            let (end, is_float) = scan_json_number(json, start).inspect_err(|_error| {
                record_lexical_scan(json.len());
            })?;
            index = end;
            let token = &json[start..index];
            if !is_float {
                let magnitude = token.strip_prefix(b"-").unwrap_or(token);
                if magnitude.len() > MAX_NUMBER_INT_CHARACTERS {
                    record_lexical_scan(index);
                    return Err(String::from("invalid score JSON integer"));
                }
                if integer_exceeds_serde_range(token) {
                    normalize_number_token(&mut normalized, json, start, index);
                }
            } else if lexical_core::parse::<f64>(token).is_ok_and(|value| !value.is_finite()) {
                normalize_number_token(&mut normalized, json, start, index);
            }
            continue;
        }
        index += 1;
    }
    record_lexical_scan(json.len());
    normalized.map_or(Ok(Cow::Borrowed(input)), |bytes| {
        String::from_utf8(bytes)
            .map(Cow::Owned)
            .map_err(|_| String::from("invalid score JSON"))
    })
}

fn scan_json_number(json: &[u8], start: usize) -> Result<(usize, bool), String> {
    let mut index = start;
    let mut is_float = false;
    if json[index] == b'-' {
        index += 1;
    }

    match json.get(index) {
        Some(b'0') => {
            index += 1;
            if json.get(index).is_some_and(u8::is_ascii_digit) {
                return Err(String::from("invalid score JSON number"));
            }
        }
        Some(b'1'..=b'9') => {
            index += 1;
            while json.get(index).is_some_and(u8::is_ascii_digit) {
                index += 1;
            }
        }
        _ => return Err(String::from("invalid score JSON number")),
    }

    if json.get(index) == Some(&b'.') {
        is_float = true;
        index += 1;
        let fraction_start = index;
        while json.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == fraction_start {
            return Err(String::from("invalid score JSON number"));
        }
    }

    if matches!(json.get(index), Some(b'e' | b'E')) {
        is_float = true;
        index += 1;
        if matches!(json.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent_start = index;
        while json.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent_start {
            return Err(String::from("invalid score JSON number"));
        }
    }

    if !is_json_value_end_boundary(json, index) {
        return Err(String::from("invalid score JSON number"));
    }
    Ok((index, is_float))
}

fn is_json_value_end_boundary(json: &[u8], index: usize) -> bool {
    matches!(
        json.get(index),
        None | Some(b' ' | b'\t' | b'\n' | b'\r' | b',' | b']' | b'}')
    )
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

fn normalize_number_token(
    normalized: &mut Option<Vec<u8>>,
    original: &[u8],
    start: usize,
    end: usize,
) {
    let output = normalized.get_or_insert_with(|| original.to_vec());
    let zero_end = if original[start] == b'-' {
        output[start + 1] = b'0';
        start + 2
    } else {
        output[start] = b'0';
        start + 1
    };
    output[zero_end..end].fill(b' ');
}

fn top_level_field_raws(json: &str) -> Result<BTreeMap<String, String>, String> {
    record_lexical_scan(json.len());
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
        let mut closing_stack = vec![if first == b'{' { b'}' } else { b']' }];
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
                b'{' => closing_stack.push(b'}'),
                b'[' => closing_stack.push(b']'),
                b'}' | b']' => {
                    if closing_stack.pop() != Some(byte) {
                        return Err(String::from("mismatched score JSON container"));
                    }
                    if closing_stack.is_empty() {
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

fn integer_exceeds_serde_range(token: &[u8]) -> bool {
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
    parse_lax_top_n_lexical(raw.get())
}

fn parse_lax_top_n_lexical(lexical: &str) -> Option<LaxTopN> {
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
