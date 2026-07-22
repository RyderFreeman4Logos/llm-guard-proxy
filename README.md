# llm-guard-proxy

Supported platform: Linux x86_64. ARM, Windows, and macOS are not supported targets.

`llm-guard-proxy` is an Apache-2.0 Rust OpenAI-compatible guard proxy for local and GB10 vLLM deployments.

The proxy sits between OpenAI-compatible clients and an upstream LLM service so debuggability, retries, loop protection, heartbeat behavior, and observability can be added without changing model quality.

## Non-Goals

- Generic `/v1/...` request forwarding is implemented. Non-streaming `/v1/chat/completions` requests use the shielded upstream streaming core by default.
- Configuration is loaded and hot-reloadable. Observability metadata storage exists for retries, loop detection, thinking policy, heartbeat behavior, and upstream metadata discovery.
- This bootstrap does not change upstream OpenAI-compatible semantics.

## Workspace Layout

- `crates/llm-guard-proxy-core`: headless contracts shared by state and service code.
- `crates/llm-guard-proxy-state`: observability, evidence, and budget state.
- `crates/llm-guard-proxy`: binary/service entry point.
- `justfile` and `lefthook.yml`: authoritative local formatting, lint, test, and hook wiring.

See [the architecture document](docs/architecture.md) for the current
service/state/core boundary, remaining target ownership, ports, feature
placement, and forbidden dependency edges.

## Local Quality Gates

This repository does not use hosted CI. The `justfile` and Lefthook definitions
are the authoritative local completion gate:

Automated CSA review enforcement is intentionally not wired into pre-push while
its review efficiency is being repaired. Repository feature development remains
paused until the maintainer explicitly authorizes resumption; the mechanical
local quality gate remains mandatory.

```bash
just fmt
just clippy
just test
```

The aggregate local gates are:

```bash
just pre-commit-fast
just pre-commit
just pre-push
```

The local completion gates are:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

The default local build and test parallelism is bounded at two workers. Machines
with sufficient memory can raise it explicitly without weakening any gate:

```bash
LLM_GUARD_LOCAL_JOBS=4 LLM_GUARD_LOCAL_TEST_THREADS=4 just pre-push
```

## GB10 Compatibility Smoke

Run the live GB10 compatibility smoke from this repository with:

```bash
just smoke-gb10
```

Prerequisites:

- `gb10:18009` is reachable from the local machine.
- `curl`, `python3`, and Cargo are available.
- The command runs the proxy locally against `http://gb10:18009/v1`; it does not deploy to GB10.

The harness creates a private temporary run directory, generates a local
`config.toml`, starts `llm-guard-proxy` on an available localhost port, and
enables SQLite observability. The observability database lives at
`$run_dir/observability.sqlite3`, where `$run_dir` is printed by the command.
The run directory is removed on success or failure unless
`LLM_GUARD_PROXY_SMOKE_KEEP=1` is set.

Checks performed:

- `GET /v1/models` through the proxy must return `200`, parse as JSON, and
  preserve or enrich model context metadata when upstream exposes it.
- `POST /v1/chat/completions` must succeed as a real OpenAI-compatible
  non-streaming chat request. The smoke payload sets
  `chat_template_kwargs.enable_thinking=false` for low-token AEON/vLLM runs.
- `POST /v1/chat/completions` with `stream=true` must return
  `text/event-stream` data.
- `POST /v1/completions` and `POST /v1/embeddings` are probed and reported with
  the exact upstream status. `POST /v1/rerank` must succeed with the separately
  configurable score model and return exactly one result for the one-document
  smoke fixture: index `0` with a finite numeric score.
- Scalar text-only `POST /v1/score` requests are unconditionally adapted to
  `/v1/rerank`; canonical batch and multimodal shapes plus complete future
  variants remain passthrough, while legacy `query`/`documents` shapes are
  forwarded to `/v1/rerank`.
  The smoke requires a complete adapted score response. Score request bodies
  are limited to 1 MiB before model extraction or JSON shape parsing to bound
  parser amplification. Transformable score requests carrying `Signature`,
  `Signature-Input`, or non-`Bearer`/non-`Basic` `Authorization` credentials are
  rejected locally because rewriting could invalidate signatures; passthrough
  score shapes and direct rerank requests remain intact.
- `GET /v1/../admin` is sent with `curl --path-as-is` and must be rejected by
  the proxy with `400` before any upstream attempt.
- SQLite observability must contain one request row per smoke call, one attempt
  row per forwarded call, and no attempt row for the rejected dot-segment path.

Useful overrides:

```bash
LLM_GUARD_PROXY_SMOKE_PORT=19009 just smoke-gb10
LLM_GUARD_PROXY_SMOKE_MODEL=aeon-ultimate just smoke-gb10
LLM_GUARD_PROXY_SMOKE_SCORE_MODEL=qwen3-reranker-8b just smoke-gb10
LLM_GUARD_PROXY_SMOKE_HEALTH_PROBE_TIMEOUT_MS=2000 just smoke-gb10
LLM_GUARD_PROXY_SMOKE_UPSTREAM_BASE_URL=http://gb10:18009/v1 just smoke-gb10
LLM_GUARD_PROXY_BIN=target/debug/llm-guard-proxy just smoke-gb10
LLM_GUARD_PROXY_SMOKE_KEEP=1 just smoke-gb10
```

The shielded chat core is enabled by default for non-streaming chat completions:

- Downstream non-stream chat requests are sent upstream with `stream=true`.
- The default downstream response is `text/event-stream`. While the shielded upstream attempt is pending, the proxy emits comment heartbeats shaped as `: llm-guard-proxy heartbeat`. After the attempt is accepted, it emits `event: final` with the accepted OpenAI-compatible chat completion JSON in the `data:` field.
- If the same normalized input fingerprint repeats within the loop-guard window, the downstream response switches to `application/json` with leading whitespace heartbeat bytes before the final JSON body. Standard JSON parsers accept the leading whitespace.
- The same `[loop_guard]` section also feeds shielded upstream SSE deltas into a channelized loop detector for hidden reasoning fields, visible content, tool-call argument fragments, and completed tool fingerprints. `mode = "monitor"` records bounded signals without aborting; `mode = "enforce"` aborts high-confidence abort candidates through the existing upstream-body error path. Feature metadata contains hashes, counters, window sizes, severity, confidence, reason codes, and channel summaries by default.
- `loop_guard.on_reasoning_loop = "retry_ladder"` preserves the default retry ladder. Quality-first deployments can opt into `"truncate_cot_then_answer"` or `"bounded_answer_from_cot"` to retry a reasoning-channel loop with a bounded private pre-loop reasoning prefix and instructions to answer without continuing the loop.
- `heartbeat.mode = "disabled"` keeps the legacy buffered JSON response for shielded non-stream chat completions.
- Attempt observability records include first-byte latency, first-token latency, finish reason, parsed content/reasoning/tool-call delta counters, and bounded `loop_*` diagnostics when monitor or enforce mode emits detector signals.
- Downstream `stream=true` chat requests currently stay on the generic streaming path to preserve first-chunk timing and backpressure behavior while later issues add release-after-inspection streaming.
- Set `[shielding] enabled = false` and hot reload the config to fall back to generic forwarding for rollback or compatibility testing.

## DeepInfra Qwen3 reranker compatibility

`POST /v1/inference/Qwen/Qwen3-Reranker-8B` accepts DeepInfra's native
pairwise `queries` and `documents` arrays and converts them to one vLLM
`/v1/score` N:N batch. Both arrays must be non-empty, have the same length,
contain only strings, and contain at most 1,024 items. Request bodies are
limited to 1 MiB before JSON validation.

The documented default `instruction` may be omitted or supplied explicitly.
Other instructions fail locally because the current deployment score template
does not consume vLLM's forwarded instruction variable. A non-null `webhook`
also fails locally because the adapter is synchronous. `service_tier` accepts
`default`, `priority`, and `flex`, but all three are compatibility no-ops: the
adapter has a local single scheduling tier, so these values affect only
observability metadata and never local scheduling.

Successful responses contain `scores`, trusted upstream `input_tokens`, and an
optional `request_id`. The optional cloud `inference_status` object is omitted:
the local adapter does not fabricate DeepInfra runtime or billing metrics.

## Operational Endpoints

- `GET /health` returns a small JSON status object with `process = "alive"` and upstream readiness. By default it performs a bounded `GET /v1/models` readiness probe against the configured upstream and returns `503` when the proxy process is alive but upstream is unavailable. Set `observability.health_upstream_probe_enabled = false` to skip the upstream probe and report `upstream = "not_checked"`.
- `GET /metrics` returns Prometheus-style text metrics when `observability.metrics_enabled = true` (the default). Metrics are aggregated from the bounded SQLite observability store and use low-cardinality labels only: request/attempt status, upstream/downstream mode, HTTP status class, heartbeat mode, upstream error kind, retry/loop observations, current-retained latency distribution gauges, and monotonic storage pruning counters. Metrics derived from retained rows are named `llm_guard_proxy_current_retained_*` and emitted as gauges because retention pruning can remove old rows; only cumulative storage pruning metrics use `*_total` counters. Metrics never include raw prompts, raw bodies, request headers, query strings, model IDs, or request IDs.
- `GET /debug/recent-requests?limit=N` is disabled by default. Set `observability.debug_summary_enabled = true` to expose bounded recent request summaries; optionally set `observability.debug_summary_admin_token` and pass the same admin value in the `Authorization: Bearer ...` header or the `x-admin-token` header. Output is clamped by `observability.debug_summary_max_records`, omits raw prompt/output payloads, omits sensitive metadata keys, and redacts sensitive-looking values.

## Deployment Hardening

The built-in defaults are intended for local, LAN, Tailscale, or Cloudflare-fronted testing where an upstream OpenAI-compatible service already enforces its own model and account policy. They do not replace network ACLs, TLS termination, upstream authentication, or host firewall rules.

- Request body buffering is capped by `server.max_request_body_bytes` before upstream work starts. Rejections return an OpenAI-shaped `413` error and do not create upstream attempts.
- Proxied request admission is capped by `server.max_in_flight_requests`. The proxy returns `503 proxy_in_flight_limit_exceeded` before reading the body when capacity is exhausted. Permits are held until the downstream response body completes or is dropped.
- Control-plane `GET /v1/models` forwarding uses the independent `server.max_control_plane_in_flight_requests` cap, defaulting to `128` so operator and monitoring bursts fit under the bound. It does not consume generation request slots; excess control-plane requests fail fast with `503 proxy_control_plane_in_flight_limit_exceeded`.
- Upstream requests use `upstream.request_timeout_ms` as a per-attempt total timeout, including streamed response body reads. Timeout errors are recorded with bounded error kinds such as `timeout_failure`, not raw URLs or headers.
- Ctrl-C and SIGTERM start graceful shutdown: the listener stops accepting new connections and in-flight response bodies are allowed to finish or be canceled by downstream drop.
- Request forwarding strips hop-by-hop headers, `Host`, `Content-Length`, and the admin-only `x-admin-token` header. OpenAI-compatible `Authorization` and `x-api-key` headers are forwarded to the upstream for normal `/v1/...` calls, but are redacted from logs, metrics, debug summaries, and observability metadata.
- Raw prompts, raw outputs, reasoning text, and tool arguments are not persisted by default. Keep `observability.capture_raw_payloads = false` for shared LAN or Cloudflare exposure unless you have a separate data-handling reason to enable it.
- CoT salvage policies send a bounded private reasoning prefix to the upstream retry attempt. The proxy does not release that prefix downstream, and evidence raw payload persistence remains disabled by default. Enabling `observability.capture_raw_payloads = true` or `evidence.include_raw_payloads = true` can persist prompts, outputs, reasoning, and tool arguments after redaction; treat those settings as sensitive debug modes.
- `heartbeat.mode = "sse"` is the default downstream liveness mode for shielded non-stream chat and is suitable for Cloudflare-style idle timeout protection. Repeated normalized inputs switch to JSON whitespace heartbeat to preserve parseable non-stream JSON.

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
max_in_flight_requests = 16
max_control_plane_in_flight_requests = 128
max_request_body_bytes = 67108864
# Aggregate body bytes retained by restart-queue waiters (default 256 MiB).
# max_restart_queue_body_bytes = 268435456

[[listeners]]
name = "embedding-legacy"
bind_host = "127.0.0.1"
port = 18002
allowed_upstreams = ["qwen3-embedding-8b"]

[[listeners]]
name = "reranker-legacy"
bind_host = "127.0.0.1"
port = 18003
allowed_upstreams = ["qwen3-reranker-8b"]

[[listeners]]
name = "aggregate"
bind_host = "127.0.0.1"
port = 18005

[upstream]
base_url = "http://gb10:18009/v1"
request_timeout_ms = 120000

[upstream.metadata]
discovery_enabled = true
enrich_responses = true
refresh_interval_secs = 60
# context_length_override = 256000
# max_model_len_override = 256000

[[upstreams]]
name = "qwen3-embedding-8b"
base_url = "http://gb10:18012/v1"
match_models = ["embedding-model"]

[[upstreams]]
name = "qwen3-reranker-8b"
base_url = "http://gb10:18013/v1"
match_models = ["rerank-model"]

# For same-model high availability, use a profile with ordered endpoints. The
# legacy [upstream] and [[upstreams]] single-base-url forms above remain valid.
# Health is checked at <base_url>/models (for example /v1/models). Requests
# wait while all endpoints are unavailable and fail with HTTP 503 only after
# health_probe_max_wait expires.
[[profile]]
model = "aeon-27b"
health_probe_interval = "2s" # probe-result cache TTL and on-demand poll interval
health_probe_timeout = "1s"  # timeout for one probe
health_probe_max_wait = "120s"

[[profile.upstream]]
base_url = "http://localhost:18010/v1"
priority = "primary"

[[profile.upstream]]
base_url = "http://gb10:18010/v1"
priority = "failover"

[shielding]
enabled = true

[observability]
enabled = true
sqlite_path = "~/.local/state/llm-guard-proxy/observability.sqlite3"
capture_raw_payloads = false
metrics_enabled = true
health_upstream_probe_enabled = true
health_upstream_probe_timeout_ms = 500
debug_summary_enabled = false
# debug_summary_admin_token = "change-me-for-admin-debug"
debug_summary_max_records = 25

[observability.retention]
max_bytes = 1073741824
prune_to_bytes = 805306368
max_records = 100000
prune_to_records = 80000

[evidence]
enabled = false
sqlite_path = "~/.local/state/llm-guard-proxy/evidence.sqlite3"
blob_cache_dir = "~/.cache/llm-guard-proxy/evidence/blobs"
include_raw_payloads = false
include_request_headers = false
max_bytes = 10737418240
prune_to_bytes = 8589934592
max_records = 100000

[evidence.shadow]
enabled = false
keep_looping_attempt_running = false
parallel_downgrade_attempts = true
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 2
shadow_attempt_timeout_ms = 7200000
# Same-prompt evidence-only alternatives for offline quality tuning.
# compare_attempts = ["max-thinking", "bounded-thinking", "no-thinking", "cot-salvage"]

[thinking]
enabled = true
force_disable = false
budget_tokens = 32768
preserve_answer_budget = true
# "apply" keeps the regular thinking rewrite for every chat request.
# "passthrough" leaves caller-provided thinking fields untouched when a
# request carries tool/function-calling hints.
tool_request_policy = "apply"

[loop_guard]
enabled = true
mode = "monitor" # disabled, monitor, enforce
on_reasoning_loop = "retry_ladder" # retry_ladder, truncate_cot_then_answer, bounded_answer_from_cot
normalized_input_window_secs = 120
max_repeated_inputs = 1
output_repeated_line_threshold = 24
output_token_window_size = 12
output_repeated_token_window_threshold = 32
output_suffix_cycle_threshold = 32
output_low_progress_min_bytes = 4096
output_low_progress_unique_ratio_percent = 15
input_overlap_threshold_multiplier = 4
reasoning_semantic_detection_enabled = true
reasoning_semantic_similarity_threshold_percent = 55
reasoning_semantic_window_token_count = 24
reasoning_semantic_minimum_token_count = 8
reasoning_semantic_history_window_count = 16

# Semantic loop detection is reasoning-only and compares bounded normalized
# token/ngram windows with Jaccard similarity. The default is enabled with a
# conservative majority-overlap threshold; raise the threshold to reduce false
# positives, lower it to catch looser paraphrases, or disable it to keep only
# the hash, suffix-cycle, and low-progress detectors.
#
# Detector mode controls channelized output-loop decisions. `disabled` skips
# detector work, `monitor` writes bounded content-free signal summaries, and
# `enforce` preserves abort behavior for high-confidence reasoning and tool-loop
# candidates. Raw reasoning, visible content, and tool arguments are still not
# persisted unless observability.capture_raw_payloads is enabled.

[retry]
enabled = true
max_attempts = 3
anti_loop_hint_enabled = true
shielded_streaming_enabled = false
downstream_drop_policy = "cancel" # cancel, detach

[[retry.ladder]]
name = "max-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 32768

[[retry.ladder]]
name = "bounded-thinking"
thinking_mode = "force_thinking"
max_tokens = 50000
thinking_token_budget = 8192
anti_loop_hint = "Previous attempt became repetitive. Do not repeat prior analysis; answer directly."

[[retry.ladder]]
name = "no-thinking"
thinking_mode = "force_disable"
max_tokens = 50000

[heartbeat]
mode = "sse" # sse, json-whitespace, disabled
interval_secs = 15

[cloudflare]
enabled = true

# Host memory guardian. Disabled means observer-only; no process is killed.
[guardian]
enabled = false
target_label = "aeon-text"
mem_threshold_gib = 2
kill_action = "cgroup.kill" # or systemctl_restart
poll_interval_secs = 1
# registration_file = "text-cgroup.v1" # defaults to <target_label>.v1
# systemd_unit = "vllm-aeon-text.service" # optional action target override
# reserve_mib = 64
# retry_interval_secs = 5
# cgroup_root = "/sys/fs/cgroup"
```

The guardian runs inside the proxy and consumes the same hot-reloaded
configuration snapshot. Invalid edits keep the last-known-good policy. Policy
changes are prepared while memory is healthy and published as a complete unit;
the latched emergency path deliberately defers reloads. The `cgroup.kill`
action preserves the allocation-free pre-opened-fd path. `systemctl_restart`
restarts the configured user service instead. The runtime registration
directory defaults to `$XDG_RUNTIME_DIR/gb10-memory-guardian` (falling back to
`/run/user/<uid>/gb10-memory-guardian`) and can be overridden with
`--guardian-runtime-dir`.

Retention byte limits apply to actual SQLite page storage. SQLite has a
schema/page-size minimum footprint, so limits below that floor prune retained
rows but cannot shrink the database file below the empty-store minimum.
Record-count retention prunes from `max_records` down to `prune_to_records`;
when omitted, `prune_to_records` defaults to 80% of `max_records`.

For shielded chat requests, the thinking policy injects or raises known
`thinking.budget_tokens`, `thinking_token_budget`, and chat-template budget
fields unless the caller explicitly disables thinking or sets a zero budget. When
`preserve_answer_budget` is enabled, numeric `max_tokens`,
`max_completion_tokens`, and `max_output_tokens` fields are increased by the
thinking-budget delta so the caller's answer-token reserve is preserved.
Set `thinking.tool_request_policy = "passthrough"` to make requests with
`tools`, legacy `functions`, `tool_choice`, or `function_call` bypass the
thinking rewrite entirely; the proxy then forwards any caller-provided thinking
parameters as-is while still applying the regular thinking policy to non-tool
chat requests. Set hot-reloadable `thinking.force_disable = true` to override
all caller and proxy thinking budgets with zero, normalize known
`enable_thinking`/`enabled` markers to `false`, and leave answer-token fields
unchanged. Force-disable takes precedence over `thinking.enabled`,
`thinking.budget_tokens`, zero-budget opt-outs, caller disable markers, and
`thinking.tool_request_policy = "passthrough"`.

Reloadable fields:

- `server.max_in_flight_requests`
- `server.max_queued_generation_requests`
- `server.generation_queue_timeout_ms`
- `server.max_control_plane_in_flight_requests`
- `server.max_request_body_bytes`
- `server.max_restart_queue_body_bytes`
- `shielding.enabled`
- `observability.enabled`
- `observability.capture_raw_payloads`
- `observability.metrics_enabled`
- `observability.health_upstream_probe_enabled`
- `observability.health_upstream_probe_timeout_ms`
- `observability.health_chat_probe_enabled`
- `observability.health_chat_probe_timeout_ms`
- `observability.debug_summary_enabled`
- `observability.debug_summary_admin_token`
- `observability.debug_summary_max_records`
- `observability.retention.max_bytes`
- `observability.retention.prune_to_bytes`
- `observability.retention.max_records`
- `observability.retention.prune_to_records`
- `evidence.enabled`
- `evidence.include_raw_payloads`
- `evidence.include_request_headers`
- `evidence.max_bytes`
- `evidence.prune_to_bytes`
- `evidence.max_records`
- `evidence.prune_to_records`
- `evidence.shadow.enabled`
- `evidence.shadow.keep_looping_attempt_running`
- `evidence.shadow.parallel_downgrade_attempts`
- `evidence.shadow.max_shadow_attempts_per_request`
- `evidence.shadow.max_global_shadow_in_flight`
- `evidence.shadow.shadow_attempt_timeout_ms`
- `evidence.shadow.compare_attempts`
- `thinking.enabled`
- `thinking.force_disable`
- `thinking.budget_tokens`
- `thinking.preserve_answer_budget`
- `thinking.tool_request_policy`
- `loop_guard.enabled`
- `loop_guard.mode`
- `loop_guard.on_reasoning_loop`
- `loop_guard.normalized_input_window_secs`
- `loop_guard.max_repeated_inputs`
- `loop_guard.output_repeated_line_threshold`
- `loop_guard.output_token_window_size`
- `loop_guard.output_repeated_token_window_threshold`
- `loop_guard.output_suffix_cycle_threshold`
- `loop_guard.output_low_progress_min_bytes`
- `loop_guard.output_low_progress_unique_ratio_percent`
- `loop_guard.input_overlap_threshold_multiplier`
- `loop_guard.reasoning_semantic_detection_enabled`
- `loop_guard.reasoning_semantic_similarity_threshold_percent`
- `loop_guard.reasoning_semantic_window_token_count`
- `loop_guard.reasoning_semantic_minimum_token_count`
- `loop_guard.reasoning_semantic_history_window_count`
- `retry.enabled`
- `retry.max_attempts`
- `retry.anti_loop_hint_enabled`
- `retry.shielded_streaming_enabled`
- `retry.downstream_drop_policy`
- `retry.ladder`
- `heartbeat.mode`
- `heartbeat.interval_secs`
- `cloudflare.enabled`
- `upstream.request_timeout_ms`
- `upstream.metadata.discovery_enabled`
- `upstream.metadata.enrich_responses`
- `upstream.metadata.refresh_interval_secs`
- `upstream.metadata.context_length_override`
- `upstream.metadata.max_model_len_override`
- `upstream.metadata.input_token_safety_margin`

Restart-required fields:

- `server.bind_host`
- `server.port`
- `listeners.topology`
- `upstream.base_url`
- `upstreams.topology`
- `observability.sqlite_path`
- `evidence.sqlite_path`
- `evidence.blob_cache_dir`

The legacy `[server]` listener is always active for backwards-compatible
single-listener deployments. Additional `[[listeners]]` bind extra downstream
ports in the same process. `allowed_upstreams` is optional; when omitted, the
listener can route to the implicit `default` upstream and all named
`[[upstreams]]`. When set, names must reference `default` or a configured
upstream profile. Listener add/remove/bind/allow-list changes are
restart-required and are reported as `listeners.topology` on config reload.
Safe per-upstream fields such as request timeout, metadata, thinking policy,
and generation limits remain hot-reloadable only when the upstream topology is
unchanged.
