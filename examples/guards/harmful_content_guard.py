#!/usr/bin/env python3
"""
Example GWP (Guard Workflow Protocol) guard for llm-guard-proxy.

This script demonstrates both pre_request_guard and post_response_guard
hooks using simple keyword/pattern matching. It reads a GWP invocation
JSON object from stdin and writes a GWP result JSON object to stdout.

Usage in config.toml:

    [workflows."harmful_content_guard.v1"]
    kind = "stdio"
    command = "/path/to/examples/guards/harmful_content_guard.py"
    timeout_ms = 5000

    [guard_workflows]
    pre_request = "harmful_content_guard.v1"
    post_response = "harmful_content_guard.v1"
    fail_closed_blocks = true

GWP Invocation (stdin):
    {
      "protocol_version": 1,
      "hook": "pre_request" | "post_response",
      "profile": "<profile_name>",
      "model": "<requested_model>",
      "request_body": { ... },      # Original client request (chat completion)
      "response_body": { ... }      # Only present for post_response hook
    }

GWP Result (stdout):
    {
      "protocol_version": 1,
      "decision": "allow" | "block" | "replace",
      "reason": "<optional explanation>",
      "replacement_content": "..."  # Only for "replace"
    }

Extension points:
    - Replace the HARMFUL_PATTERNS list with your own keyword/pattern rules
    - Integrate a real moderation API (e.g., OpenAI moderation, Perspective API)
    - Add multi-step review: anonymize → moderate → review
    - Log to a file or syslog for audit trail

This script uses only Python stdlib — no pip dependencies required.
"""

import json
import re
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Configuration: harmful content patterns
# ---------------------------------------------------------------------------
# These are simple demonstration patterns. A production guard should use
# a proper moderation model or API. Patterns are matched case-insensitively
# against the text content of messages.

HARMFUL_PATTERNS: list[str] = [
    # Violence / self-harm
    r"\bhow to (make|build|create) (a )?bomb\b",
    r"\b(how to|ways to) .*(hurt|harm|kill|murder) (myself|yourself|himself|herself|themselves|someone|people|everyone)\b",
    r"\b(kill|murder|hurt|harm) (yourself|myself|everyone|all of you)\b",

    # Illegal activity
    r"\bhow to (hack|crack|bypass) .*(password|security|authentication)\b",
    r"\bcredit card (numbers|generator|validator)\b",
    r"\bhow to (steal|shoplift|counterfeit)\b",

    # Privacy / PII (basic check for SSN-like patterns)
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
]

# Compile patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]


def extract_text(request_body: dict[str, Any] | None,
                 response_body: dict[str, Any] | None) -> str:
    """Extract all text content from request/response messages."""
    parts: list[str] = []

    # Extract from request messages
    if request_body:
        for msg in request_body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))

    # Extract from response choices
    if response_body:
        for choice in response_body.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)

    return " ".join(parts)


def check_content(text: str) -> tuple[str, str | None]:
    """
    Check text against harmful patterns.

    Returns:
        Tuple of (decision, reason).
        - ("allow", None) if no harmful content detected
        - ("block", "<reason>") if harmful pattern matched
    """
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        match = pattern.search(text)
        if match:
            return (
                "block",
                f"Content matched harmful pattern #{i + 1}: "
                f"'{HARMFUL_PATTERNS[i]}' "
                f"(matched: '{match.group()}')",
            )

    return ("allow", None)


def handle_invocation(invocation: dict[str, Any]) -> dict[str, Any]:
    """
    Process a GWP invocation and return a GWP result.

    This function handles both pre_request and post_response hooks.
    For pre_request: checks the client's input messages.
    For post_response: checks both the input and the model's output.
    """
    hook = invocation.get("hook", "pre_request")
    request_body = invocation.get("request_body")
    response_body = invocation.get("response_body")

    # Extract all text for checking
    text = extract_text(request_body, response_body)

    if not text.strip():
        # No text to check — allow
        return {
            "protocol_version": 1,
            "decision": "allow",
            "reason": "No text content to check",
        }

    decision, reason = check_content(text)

    if decision == "block":
        return {
            "protocol_version": 1,
            "decision": "block",
            "reason": reason
            or "Content matched harmful content pattern",
        }

    # Content is safe
    return {
        "protocol_version": 1,
        "decision": "allow",
        "reason": f"Content passed {hook} guard check "
        f"({len(text)} chars scanned)",
    }


def main() -> int:
    """Read GWP invocation from stdin, write result to stdout."""
    try:
        raw_input = sys.stdin.read()
        invocation = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        # Fail closed: if we can't parse the invocation, block
        result = {
            "protocol_version": 1,
            "decision": "block",
            "reason": f"Failed to parse GWP invocation: {exc}",
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0

    result = handle_invocation(invocation)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
