# Example Workflows for llm-guard-proxy

This directory contains example workflow scripts that demonstrate
multi-step composition within a single GWP workflow invocation.

## Available Workflows

### `child_safe_general.py`

A family-safe content pipeline demonstrating:

1. **Pre-request check**: blocks harmful input before reaching the model
2. **System prompt injection**: adds a family-safe system prompt
3. **Upstream model call**: calls the inference server (configurable)
4. **Post-response check**: replaces harmful model output
5. **Safe response**: returns family-friendly content

## Usage

1. Configure environment variables:
   ```bash
   export CHILD_SAFE_UPSTREAM_URL=http://localhost:18010/v1
   export CHILD_SAFE_UPSTREAM_MODEL=Qwen3-235B-A22B-Instruct-2507
   ```

2. Configure in `config.toml`:
   ```toml
   [workflows."child_safe_general.v1"]
   kind = "stdio"
   command = "/path/to/examples/workflows/child_safe_general.py"
   timeout_ms = 120000

   [[model_aliases]]
   pattern = "family/child-safe-general-v1"
   target_kind = "workflow"
   workflow_id = "child_safe_general.v1"
   workflow_timeout_ms = 120000
   ```

3. Build with guard feature:
   ```bash
   cargo build --features guard
   ```

4. Call the alias:
   ```bash
   curl http://localhost:18009/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"family/child-safe-general-v1","messages":[{"role":"user","content":"Tell me about space!"}]}'
   ```

## How It Works

The proxy resolves the model alias `family/child-safe-general-v1` to the
workflow `child_safe_general.v1`. It invokes the script via stdio:

1. Proxy sends the original request as a GWP invocation JSON on stdin
2. Script processes internally (check → prompt → call model → check → respond)
3. Script writes the final result as a GWP result JSON on stdout
4. Proxy shapes the result as a chat completion response

The proxy only sees ONE workflow call. Multi-step composition (pre-check,
upstream call, post-check) all happens INSIDE the script.
