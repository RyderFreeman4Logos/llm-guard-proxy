# Example Guards for llm-guard-proxy

This directory contains example guard workflows that demonstrate the
[GWP (Guard Workflow Protocol)](../../crates/llm-guard-proxy-core/src/gwp.rs)
integration with the proxy.

## Available Guards

### `harmful_content_guard.py`

A simple keyword/pattern-based content filter demonstrating:

- **pre_request_guard**: checks client input before forwarding to upstream
- **post_response_guard**: checks model output before returning to client
- **GWP contract**: reads JSON invocation from stdin, writes JSON result to stdout
- **stdlib only**: no pip dependencies required

## Usage

1. Make the script executable:
   ```bash
   chmod +x examples/guards/harmful_content_guard.py
   ```

2. Configure in your `config.toml`:
   ```toml
   [workflows."harmful_content_guard.v1"]
   kind = "stdio"
   command = "/path/to/examples/guards/harmful_content_guard.py"
   timeout_ms = 5000

   [guard_workflows]
   pre_request = "harmful_content_guard.v1"
   post_response = "harmful_content_guard.v1"
   fail_closed_blocks = true
   ```

3. Build the proxy with the guard feature:
   ```bash
   cargo build --features guard
   ```

## Extending

Replace the `HARMFUL_PATTERNS` list with your own rules, or integrate a
real moderation API:

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [Google Perspective API](https://www.perspectiveapi.com/)
- Custom model-based review

For multi-step workflows (e.g., anonymize → moderate → review), wrap
multiple calls inside a shell script or Python orchestrator that reads
the GWP invocation and performs the steps internally.

## GWP Protocol

Guards communicate via a simple JSON-over-stdio protocol:

**Invocation (stdin):**
```json
{
  "protocol_version": 1,
  "hook": "pre_request",
  "profile": "adult_safe",
  "model": "Qwen3-235B",
  "request_body": { "messages": [...] },
  "response_body": null
}
```

**Result (stdout):**
```json
{
  "protocol_version": 1,
  "decision": "allow",
  "reason": "Content passed guard check"
}
```

Decisions:
- `allow`: pass the request/response through unchanged
- `block`: reject with an error response to the client
- `replace`: substitute the response content (post_response only)
