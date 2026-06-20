# llm-guard-proxy

`llm-guard-proxy` is an Apache-2.0 Rust service intended to become an OpenAI-compatible guard proxy for local and GB10 vLLM deployments.

The proxy will sit between OpenAI-compatible clients and an upstream LLM service so later issues can add debuggability, retries, loop protection, heartbeat behavior, and observability without changing model quality.

## Non-Goals

- This bootstrap does not implement request forwarding.
- Configuration is loaded and hot-reloadable. Observability metadata storage exists; retries, loop detection, metadata discovery, thinking policy, and heartbeat behavior are feature flags for later issues.
- This bootstrap does not change upstream OpenAI-compatible semantics.

## Workspace Layout

- `crates/llm-guard-proxy-core`: headless core types shared by service code.
- `crates/llm-guard-proxy`: binary/service entry point.
- `.github/workflows/ci.yml`: GitHub Actions quality gates.
- `justfile` and `lefthook.yml`: local formatting, lint, test, and hook wiring.

## Local Quality Gates

Run the same core checks as CI:

```bash
just fmt
just clippy
just test
```

The aggregate local gates are:

```bash
just pre-commit-fast
just pre-commit
```

The local completion gates are:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Later issues will add the real proxy behavior behind these typed configuration fields.

## Configuration

The default config path is:

```text
~/.config/llm-guard-proxy/config.toml
```

Development and tests can use an explicit path:

```bash
llm-guard-proxy --config ./config.toml
```

If the default file is absent, the service uses built-in defaults. An explicit `--config` path must exist and must validate.

Sample config:

```toml
[server]
bind_host = "127.0.0.1"
port = 18009

[upstream]
base_url = "http://gb10:18009/v1"

[upstream.metadata]
discovery_enabled = true
enrich_responses = true
refresh_interval_secs = 60
# context_length_override = 256000
# max_model_len_override = 256000

[shielding]
enabled = true

[observability]
enabled = true
sqlite_path = "~/.local/state/llm-guard-proxy/observability.sqlite3"
capture_raw_payloads = false

[observability.retention]
max_bytes = 1073741824
prune_to_bytes = 805306368
max_records = 100000

[thinking]
enabled = true
budget_tokens = 32768
preserve_answer_budget = true

[loop_guard]
enabled = true
normalized_input_window_secs = 120
max_repeated_inputs = 1

[retry]
enabled = true
max_attempts = 2

[heartbeat]
mode = "sse" # sse, json-whitespace, disabled
interval_secs = 15

[cloudflare]
enabled = true
```

Retention byte limits apply to actual SQLite page storage. SQLite has a
schema/page-size minimum footprint, so limits below that floor prune retained
rows but cannot shrink the database file below the empty-store minimum.

Reloadable fields:

- `shielding.enabled`
- `observability.enabled`
- `observability.capture_raw_payloads`
- `observability.retention.max_bytes`
- `observability.retention.prune_to_bytes`
- `observability.retention.max_records`
- `thinking.enabled`
- `thinking.budget_tokens`
- `thinking.preserve_answer_budget`
- `loop_guard.enabled`
- `loop_guard.normalized_input_window_secs`
- `loop_guard.max_repeated_inputs`
- `retry.enabled`
- `retry.max_attempts`
- `heartbeat.mode`
- `heartbeat.interval_secs`
- `cloudflare.enabled`
- `upstream.metadata.discovery_enabled`
- `upstream.metadata.enrich_responses`
- `upstream.metadata.refresh_interval_secs`
- `upstream.metadata.context_length_override`
- `upstream.metadata.max_model_len_override`

Restart-required fields:

- `server.bind_host`
- `server.port`
- `upstream.base_url`
- `observability.sqlite_path`
