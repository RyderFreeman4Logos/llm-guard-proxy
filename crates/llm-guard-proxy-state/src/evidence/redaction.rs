use std::collections::BTreeMap;

use crate::{RawPayloadChunk, RawPayloads};

const REDACTED: &str = "[REDACTED]";
const SENSITIVE_KEY_MARKERS: &[&str] = &[
    "authorization",
    "apikey",
    "api-key",
    "xapikey",
    "x-api-key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "bearer",
];
const NON_SECRET_TOKEN_KEYS: &[&str] = &[
    "prompttokens",
    "completiontokens",
    "totaltokens",
    "thinkingpolicymaxtokens",
    "thinkingpolicybudgettokens",
    "thinkinganswerbudgetfinalmaxtokens",
    "thinkinganswerbudgetfinalmaxcompletiontokens",
    "thinkinganswerbudgetfinalmaxoutputtokens",
    "thinkinganswerbudgetfinalparametersmaxtokens",
    "attemptthinkingbudgettokens",
    "attemptthinkingmaxtokens",
    "outputtokenwindowsize",
    "outputrepeatedtokenwindowthreshold",
];

pub(super) fn evidence_metadata_map(
    metadata: &BTreeMap<String, String>,
    include_headers: bool,
) -> BTreeMap<String, String> {
    metadata
        .iter()
        .filter(|(key, _value)| include_headers || !is_header_metadata_key(key))
        .map(|(key, value)| {
            let value = if is_sensitive_key(key) {
                REDACTED.to_owned()
            } else {
                scrub_text(value)
            };
            (key.clone(), value)
        })
        .collect()
}

pub(super) fn sanitize_raw_payloads(raw: &RawPayloads, include_raw_payloads: bool) -> RawPayloads {
    if !include_raw_payloads {
        return RawPayloads::default();
    }
    RawPayloads {
        input: raw.input.as_deref().map(scrub_text),
        output: raw.output.as_deref().map(scrub_text),
        reasoning: raw.reasoning.as_deref().map(scrub_text),
        tool_calls: raw.tool_calls.as_deref().map(scrub_text),
        chunks: raw
            .chunks
            .iter()
            .map(|chunk| RawPayloadChunk::new(chunk.channel.clone(), scrub_text(&chunk.text)))
            .collect(),
    }
}

pub(super) fn scrub_optional_text(value: Option<&String>) -> Option<String> {
    value.map(|value| scrub_text(value))
}

fn is_header_metadata_key(key: &str) -> bool {
    key.contains("_header_")
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = normalize_key(key);
    if NON_SECRET_TOKEN_KEYS
        .iter()
        .any(|allowed| normalized.contains(allowed))
    {
        return false;
    }
    SENSITIVE_KEY_MARKERS
        .iter()
        .any(|marker| normalized.contains(&normalize_key(marker)))
}

fn scrub_text(value: &str) -> String {
    scrub_sensitive_assignments(&scrub_bearer_and_sk_tokens(value))
}

fn scrub_bearer_and_sk_tokens(value: &str) -> String {
    let mut output = Vec::<String>::new();
    let mut redact_next = false;
    for token in value.split_whitespace() {
        let lower = token.to_ascii_lowercase();
        if redact_next || lower.starts_with("sk-") || lower.contains("sk-") {
            output.push(redact_wrapped_secret_token(token));
            redact_next = false;
            continue;
        }
        if let Some(scrubbed) = scrub_bearer_token(token) {
            redact_next = scrubbed.redact_next;
            output.push(scrubbed.text);
            continue;
        }
        output.push(token.to_owned());
    }
    if output.is_empty() {
        value.to_owned()
    } else {
        output.join(" ")
    }
}

struct ScrubbedBearerToken {
    text: String,
    redact_next: bool,
}

fn scrub_bearer_token(token: &str) -> Option<ScrubbedBearerToken> {
    let marker_start = find_bearer_marker(token)?;
    let marker_end = marker_start + "Bearer".len();
    let before = &token[..marker_start];
    let marker = &token[marker_start..marker_end];
    let after = &token[marker_end..];
    let Some(separator) = after.chars().next() else {
        return Some(ScrubbedBearerToken {
            text: format!("{before}{marker}"),
            redact_next: true,
        });
    };
    if matches!(separator, ':' | '=') {
        let secret_start = marker_end + separator.len_utf8();
        let secret = &token[secret_start..];
        return Some(ScrubbedBearerToken {
            text: format!(
                "{before}{marker}{separator}{}",
                redact_wrapped_secret_token(secret)
            ),
            redact_next: false,
        });
    }
    if is_bearer_boundary(separator) {
        return Some(ScrubbedBearerToken {
            text: format!("{before}{marker}{after}"),
            redact_next: true,
        });
    }
    None
}

fn find_bearer_marker(token: &str) -> Option<usize> {
    let token_lower = token.to_ascii_lowercase();
    for (index, _) in token_lower.match_indices("bearer") {
        let before_ok = token[..index]
            .chars()
            .next_back()
            .is_none_or(is_bearer_boundary);
        let after_index = index + "bearer".len();
        let after_ok = token[after_index..].chars().next().is_none_or(|character| {
            matches!(character, ':' | '=') || is_bearer_boundary(character)
        });
        if before_ok && after_ok {
            return Some(index);
        }
    }
    None
}

fn is_bearer_boundary(character: char) -> bool {
    !character.is_ascii_alphanumeric() && character != '_' && character != '-'
}

fn redact_wrapped_secret_token(token: &str) -> String {
    let secret_start = token
        .char_indices()
        .find(|(_index, character)| !is_leading_secret_wrapper(*character))
        .map_or(token.len(), |(index, _character)| index);
    let secret_end = token
        .char_indices()
        .rev()
        .find(|(index, character)| *index < secret_start || !is_trailing_secret_wrapper(*character))
        .map_or(secret_start, |(index, character)| {
            index + character.len_utf8()
        });

    if secret_start >= secret_end {
        return REDACTED.to_owned();
    }

    format!(
        "{}{}{}",
        &token[..secret_start],
        REDACTED,
        &token[secret_end..]
    )
}

fn is_leading_secret_wrapper(character: char) -> bool {
    matches!(character, '"' | '\'' | '`' | '(' | '[' | '{' | '<')
}

fn is_trailing_secret_wrapper(character: char) -> bool {
    matches!(
        character,
        '"' | '\'' | '`' | ')' | ']' | '}' | '>' | ',' | ';' | '.'
    )
}

fn scrub_sensitive_assignments(value: &str) -> String {
    let mut scrubbed = value.to_owned();
    for key in [
        "authorization",
        "api_key",
        "api-key",
        "x-api-key",
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
    ] {
        scrubbed = scrub_assignment_forms(&scrubbed, key);
    }
    scrubbed
}

fn scrub_assignment_forms(value: &str, key: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(index) = find_case_insensitive(rest, key) {
        output.push_str(&rest[..index]);
        let after_key = &rest[index + key.len()..];
        let Some((skipped, delimiter)) = assignment_delimiter(after_key) else {
            output.push_str(&rest[index..index + key.len()]);
            rest = after_key;
            continue;
        };
        output.push_str(&rest[index..index + key.len()]);
        output.push_str(&after_key[..skipped]);
        output.push(delimiter);
        output.push_str(REDACTED);
        let value_start = skipped + delimiter.len_utf8();
        let value_rest = &after_key[value_start..];
        let trimmed_value = value_rest.trim_start();
        let leading_ws = value_rest.len() - trimmed_value.len();
        let quote = trimmed_value
            .chars()
            .next()
            .filter(|character| matches!(character, '"' | '\'' | '`'));
        let value_after_quote =
            quote.map_or(trimmed_value, |quote| &trimmed_value[quote.len_utf8()..]);
        let consumed_value = value_after_quote
            .char_indices()
            .find(|(_offset, character)| {
                character.is_ascii_whitespace()
                    || matches!(*character, ',' | '}' | ']' | '"' | '\'' | '`')
            })
            .map_or(value_after_quote.len(), |(offset, _character)| offset);
        let consumed = leading_ws + quote.map_or(0, char::len_utf8) + consumed_value;
        rest = &value_rest[consumed..];
    }
    output.push_str(rest);
    output
}

fn assignment_delimiter(after_key: &str) -> Option<(usize, char)> {
    let bytes = after_key.as_bytes();
    let mut offset = 0;
    while offset < bytes.len() {
        match bytes[offset] {
            b' ' | b'\t' | b'\r' | b'\n' | b'\'' | b'"' | b'`' => offset += 1,
            b'\\' if bytes.get(offset + 1).is_some_and(|byte| *byte == b'"') => offset += 2,
            b':' | b'=' => return Some((offset, bytes[offset] as char)),
            _ => return None,
        }
    }
    None
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::scrub_text;

    #[test]
    fn scrubs_bearer_and_sk_like_tokens_without_dropping_plain_text() {
        let scrubbed = scrub_text("hello Bearer live-token and sk-live-secret remain useful");

        assert!(scrubbed.contains("hello"));
        assert!(!scrubbed.contains("live-token"));
        assert!(!scrubbed.contains("sk-live-secret"));
        assert!(scrubbed.contains("[REDACTED]"));
    }

    #[test]
    fn scrubs_quoted_and_punctuated_bearer_values() {
        let scrubbed = scrub_text(
            r#"{"content":"Bearer downstream-secret"} Bearer:colon-secret Bearer=equals-secret ("Bearer paren-secret")"#,
        );

        assert!(scrubbed.contains(r#""content":"Bearer [REDACTED]"}"#));
        assert!(scrubbed.contains("Bearer:[REDACTED]"));
        assert!(scrubbed.contains("Bearer=[REDACTED]"));
        assert!(scrubbed.contains(r#"("Bearer [REDACTED]")"#));
        assert!(!scrubbed.contains("downstream-secret"));
        assert!(!scrubbed.contains("colon-secret"));
        assert!(!scrubbed.contains("equals-secret"));
        assert!(!scrubbed.contains("paren-secret"));
    }

    #[test]
    fn scrubs_assignment_values() {
        let scrubbed = scrub_text(r#"{"api_key":"secret","content":"ok"} token=value"#);

        assert!(!scrubbed.contains("secret"));
        assert!(!scrubbed.contains("value"));
        assert!(scrubbed.contains("content"));
    }
}
