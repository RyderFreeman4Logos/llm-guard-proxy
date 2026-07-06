#!/usr/bin/env python3
"""
Example child-safe workflow for llm-guard-proxy.

This script demonstrates a multi-step guard+generation pipeline within
a single GWP workflow invocation. The proxy sees ONE workflow call, but
internally the script:

  1. Runs pre-request content check (block harmful input)
  2. Injects a family-safe system prompt
  3. Calls the upstream model (via HTTP)
  4. Runs post-response content check (replace harmful output)
  5. Returns the safe response as GWP "replace" result

Usage in config.toml:

    [workflows."child_safe_general.v1"]
    kind = "stdio"
    command = "/path/to/examples/workflows/child_safe_general.py"
    timeout_ms = 120000

    [[model_aliases]]
    pattern = "family/child-safe-general-v1"
    target_kind = "workflow"
    workflow_id = "child_safe_general.v1"
    workflow_timeout_ms = 120000

Environment variables:
    CHILD_SAFE_UPSTREAM_URL   - upstream inference endpoint (default: http://localhost:18010/v1)
    CHILD_SAFE_UPSTREAM_MODEL - model name to use (default: Qwen3-235B-A22B-Instruct-2507)
    CHILD_SAFE_API_KEY        - API key for upstream (optional)

This is an EXAMPLE, not production policy. Extend with real moderation
models, age-appropriate content classifiers, and proper PII detection.

Uses only Python stdlib (urllib for HTTP, json for GWP protocol).
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPSTREAM_URL = os.environ.get(
    "CHILD_SAFE_UPSTREAM_URL", "http://localhost:18010/v1"
)
UPSTREAM_MODEL = os.environ.get(
    "CHILD_SAFE_UPSTREAM_MODEL", "Qwen3-235B-A22B-Instruct-2507"
)
UPSTREAM_API_KEY = os.environ.get("CHILD_SAFE_API_KEY", "")

FAMILY_SAFE_SYSTEM_PROMPT = (
    "You are a helpful, family-friendly assistant. Always provide safe, "
    "educational, and age-appropriate responses. Never generate content "
    "involving violence, illegal activities, adult content, or self-harm. "
    "If a request is inappropriate, politely decline and suggest a "
    "constructive alternative."
)

# Harmful content patterns (shared with harmful_content_guard.py)
HARMFUL_PATTERNS = [
    re.compile(r"\bhow to (make|build|create) (a )?bomb\b", re.IGNORECASE),
    re.compile(
        r"\b(how to|ways to) .*(hurt|harm|kill|murder) "
        r"(myself|yourself|himself|herself|themselves|someone|people|everyone)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(kill|murder|hurt|harm) (yourself|myself|everyone|all of you)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow to (hack|crack|bypass) .*(password|security|authentication)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(credit card|ssn|social security) (numbers|generator|steal)\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
]

# Safe replacement for blocked responses
SAFE_REFUSAL = (
    "I'm sorry, but I can't help with that. "
    "Let's talk about something else fun and interesting!"
)


# ---------------------------------------------------------------------------
# Content checking
# ---------------------------------------------------------------------------

def extract_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from message list."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


def is_harmful(text: str) -> str | None:
    """Check if text matches harmful patterns. Returns reason or None."""
    for i, pattern in enumerate(HARMFUL_PATTERNS):
        match = pattern.search(text)
        if match:
            return f"Content matched harmful pattern #{i + 1} (matched: '{match.group()}')"
    return None


# ---------------------------------------------------------------------------
# Upstream model call
# ---------------------------------------------------------------------------

def call_upstream(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Call the upstream model to generate a response.

    Injects the family-safe system prompt at the beginning of the conversation.
    """
    # Build the request with system prompt
    safe_messages = [
        {"role": "system", "content": FAMILY_SAFE_SYSTEM_PROMPT}
    ] + messages

    request_body = {
        "model": UPSTREAM_MODEL,
        "messages": safe_messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    data = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        f"{UPSTREAM_URL}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=100) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "error": f"Upstream returned HTTP {exc.code}: {exc.reason}",
            "choices": [],
        }
    except urllib.error.URLError as exc:
        return {
            "error": f"Failed to connect to upstream: {exc.reason}",
            "choices": [],
        }


def extract_response_text(response: dict[str, Any]) -> str:
    """Extract the assistant's response text."""
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
    return ""


# ---------------------------------------------------------------------------
# GWP workflow handler
# ---------------------------------------------------------------------------

def handle_workflow(invocation: dict[str, Any]) -> dict[str, Any]:
    """
    Process the child-safe workflow.

    Steps:
    1. Check input for harmful content → block if found
    2. Call upstream model with family-safe system prompt
    3. Check output for harmful content → replace if found
    4. Return safe response
    """
    request_body = invocation.get("request_body", {})
    messages = request_body.get("messages", [])

    # Step 1: Pre-request content check
    input_text = extract_text(messages)
    harmful_reason = is_harmful(input_text)
    if harmful_reason:
        return {
            "protocol_version": 1,
            "decision": "replace",
            "reason": f"Input blocked by child-safe pre-check: {harmful_reason}",
            "replacement_content": SAFE_REFUSAL,
        }

    # Step 2: Call upstream model
    response = call_upstream(messages)

    if response.get("error"):
        return {
            "protocol_version": 1,
            "decision": "replace",
            "reason": f"Upstream error: {response['error']}",
            "replacement_content": (
                "I'm having trouble connecting to my brain right now. "
                "Please try again in a moment!"
            ),
        }

    # Step 3: Post-response content check
    output_text = extract_response_text(response)
    output_harmful = is_harmful(output_text)
    if output_harmful:
        return {
            "protocol_version": 1,
            "decision": "replace",
            "reason": f"Output replaced by child-safe post-check: {output_harmful}",
            "replacement_content": SAFE_REFUSAL,
        }

    # Step 4: Return safe response
    return {
        "protocol_version": 1,
        "decision": "replace",
        "reason": "Child-safe workflow completed successfully",
        "replacement_content": output_text,
    }


def main() -> int:
    """Read GWP invocation from stdin, write result to stdout."""
    try:
        raw_input = sys.stdin.read()
        invocation = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        result = {
            "protocol_version": 1,
            "decision": "replace",
            "reason": f"Failed to parse invocation: {exc}",
            "replacement_content": SAFE_REFUSAL,
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0

    result = handle_workflow(invocation)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
