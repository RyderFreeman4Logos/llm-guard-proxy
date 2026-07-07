# gb10 systemd deployment

Issue #14 deploys `llm-guard-proxy` as the stable OpenAI-compatible entrypoint
on `gb10:18009` and moves the underlying AEON vLLM text service to `18010`.

## Target layout

```text
client -> llm-guard-proxy 100.105.4.92:18009 -> AEON vLLM 100.105.4.92:18010
operator diagnostics ------------------------------> AEON vLLM gb10:18010
```

The underlying vLLM port is intentionally bound to the gb10 Tailscale address,
not `0.0.0.0`. Rootless Docker could not dual-publish the same container port to
both loopback and Tailscale on this host, so the wrapper also uses the
Tailscale-scoped upstream URL. This keeps the stable guarded entrypoint on
`gb10:18009` while preserving direct operator diagnostics at `gb10:18010`
without exposing raw vLLM on every host interface.

The wrapper config lives at:

```text
/home/obj/.config/llm-guard-proxy/config.toml
```

The bounded SQLite observability store lives at:

```text
/home/obj/.local/state/llm-guard-proxy/observability.sqlite3
```

## Pre-deploy gate

Run this before moving ports, while `gb10:18009` still points directly at vLLM:

```bash
RUSTUP_TOOLCHAIN=1.96.0 just smoke-gb10
```

The smoke test must report `result=ok` and `/v1/models` context metadata before
the service changes are applied.

## Install wrapper binary

Install on gb10 through `mise`'s cargo backend so the binary is built on the
arm64 host and managed as a reviewed GitHub cargo tool. Production updates
should point at the reviewed `main` branch instead of copying a locally built
binary from another architecture:

```bash
mise use -g 'cargo:https://github.com/RyderFreeman4Logos/llm-guard-proxy@branch:main'
```

The user unit expects a stable binary path. After `mise` installs or updates the
tool, copy the resolved arm64 executable into the `ExecStart` path:

```bash
install -Dm755 "$(mise which llm-guard-proxy)" /home/obj/.local/bin/llm-guard-proxy
```

For a pre-merge live candidate, replace `@branch:main` with a reviewed feature
branch. Reinstall from `@branch:main` after the PR merges.

Verify the binary path used by the user unit:

```bash
test -x /home/obj/.local/bin/llm-guard-proxy
sha256sum /home/obj/.local/bin/llm-guard-proxy
```

## Backup current gb10 files

Create timestamped backups before editing any user unit or wrapper config:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
systemd_dir="/home/obj/.config/systemd/user"
config_dir="/home/obj/.config/llm-guard-proxy"

cp -a "${systemd_dir}/vllm-aeon-27b-dflash-n12.service" \
  "${systemd_dir}/vllm-aeon-27b-dflash-n12.service.bak-${ts}"

mkdir -p "${config_dir}"
if [ -f "${config_dir}/config.toml" ]; then
  cp -a "${config_dir}/config.toml" "${config_dir}/config.toml.bak-${ts}"
fi
if [ -f "${systemd_dir}/llm-guard-proxy.service" ]; then
  cp -a "${systemd_dir}/llm-guard-proxy.service" \
    "${systemd_dir}/llm-guard-proxy.service.bak-${ts}"
fi
```

## Apply service changes

Copy the wrapper assets:

```bash
install -d -m 0700 /home/obj/.config/llm-guard-proxy
install -d -m 0700 /home/obj/.local/state/llm-guard-proxy
chmod 0755 /home/obj/.local
install -m 0600 deploy/gb10/config.toml \
  /home/obj/.config/llm-guard-proxy/config.toml
install -m 0644 deploy/gb10/llm-guard-proxy.service \
  /home/obj/.config/systemd/user/llm-guard-proxy.service
```

`llm-guard-proxy` refuses to create SQLite observability storage under a
group/other-writable ancestor unless that ancestor is sticky. The `chmod` above
keeps the XDG state path usable while preserving private `0700` permissions on
the actual state directory.

The deployed `[thinking] force_disable = false` default preserves existing
thinking behavior. Set it to `true` if the active model starts dead-looping in
hidden thinking or a client harness cannot control thinking budgets; the field is
hot-reloadable and takes precedence over tool-request passthrough.

Update the active AEON text unit only. Replace this Docker publish line:

```text
-p 100.105.4.92:18009:8000
```

with the `18010` binding:

```text
-p 100.105.4.92:18010:8000
```

Then reload and start in a controlled order:

```bash
systemctl --user daemon-reload
systemctl --user enable vllm-aeon-27b-dflash-n12.service
systemctl --user restart vllm-aeon-27b-dflash-n12.service

curl --fail --silent --show-error http://gb10:18010/v1/models >/dev/null

systemctl --user enable --now llm-guard-proxy.service
```

The current gb10 AEON profile needs enough KV cache for the advertised 256000
context window. If a cold restart fails with a vLLM error like `26.69 GiB KV
cache is needed` and `estimated maximum model length is 251008`, keep
`--max-model-len 256000` and raise the profile's `--gpu-memory-utilization`
from `0.46` to the deployed `0.47` before retrying. Do not silently lower the
context window; wrapper `/v1/models` verification depends on the 256000 context
metadata.

The wrapper unit applies a least-privilege sandbox (`NoNewPrivileges`, private
temporary storage, read-only home/system views, a writable state-directory
exception, and restricted address families). It intentionally avoids
`CapabilityBoundingSet=` because this gb10 user manager cannot drop capabilities
for user-service control processes.
If a future host image lacks the required user-service sandbox support, rollback
to the timestamped unit backup rather than weakening the deployed service
in-place.

## Verification

Check unit state:

```bash
systemctl --user is-enabled vllm-aeon-27b-dflash-n12.service llm-guard-proxy.service
systemctl --user is-active vllm-aeon-27b-dflash-n12.service llm-guard-proxy.service
```

Check direct vLLM and wrapper metadata:

```bash
curl --fail --silent --show-error http://gb10:18010/v1/models \
  | jq '.data[] | {id, max_model_len}'

curl --fail --silent --show-error http://gb10:18009/v1/models \
  | jq '.data[] | {id, max_model_len, context_length, max_context_length}'
```

Run a chat completion through the wrapper without printing model content:

```bash
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl --fail --silent --show-error \
  --header 'content-type: application/json' \
  --output "$tmp" \
  --write-out 'http_status=%{http_code}\n' \
  --data '{"model":"aeon-ultimate","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"temperature":0,"max_tokens":16,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' \
  http://gb10:18009/v1/chat/completions
python3 - "$tmp" <<'PY'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8")
if raw.lstrip().startswith("event:") or "\nevent:" in raw:
    payload = None
    for event in raw.split("\n\n"):
        lines = event.splitlines()
        if any(line.strip() == "event: final" for line in lines):
            data = "\n".join(
                line.removeprefix("data:").lstrip()
                for line in lines
                if line.startswith("data:")
            )
            payload = json.loads(data)
            break
    if payload is None:
        raise SystemExit("missing final SSE event")
else:
    payload = json.loads(raw)

choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("missing choices")
print("chat_completion=ok choices=%d" % len(choices))
PY
```

Run the repository smoke test again after deployment:

```bash
RUSTUP_TOOLCHAIN=1.96.0 just smoke-gb10
```

## Rollback

Use the timestamped backup made before deployment:

```bash
systemctl --user stop llm-guard-proxy.service
systemctl --user disable llm-guard-proxy.service

cp -a /home/obj/.config/systemd/user/vllm-aeon-27b-dflash-n12.service.bak-YYYYMMDDTHHMMSSZ \
  /home/obj/.config/systemd/user/vllm-aeon-27b-dflash-n12.service

systemctl --user daemon-reload
systemctl --user restart vllm-aeon-27b-dflash-n12.service
curl --fail --silent --show-error http://gb10:18009/v1/models >/dev/null
```

Restoring the backed-up vLLM unit returns direct AEON vLLM service to
`100.105.4.92:18009`. The wrapper config and unit can remain on disk while the
wrapper service is disabled and stopped.

The deployment also makes `/home/obj/.local` non-group-writable (`0755`) so the
wrapper accepts the configured SQLite path under `/home/obj/.local/state`. Leave
that permission in place while the wrapper uses this state path.

## Evidence profiles

Two ready-to-copy profiles control how much data the proxy retains for audit,
debugging, and loop-detector improvement. Append the relevant block to the
`[evidence]` section of `config.toml`.

### Privacy-minimal production

Maximum privacy. No raw payloads, no shadow attempts, no paired comparisons.
Suitable for live serving where no offline detector tuning is expected.

```toml
[evidence]
enabled = true
include_raw_payloads = false
include_request_headers = false

[evidence.shadow]
enabled = false

[evidence.shadow.paired_comparison]
enabled = false
```

### Quality-debug / loop-improvement

Collects raw input/output/reasoning, runs shadow downgrade attempts, and
produces paired comparison variants for offline detector calibration. **This
mode runs extra model attempts and stores sensitive data — use only on trusted
hosts with private storage.**

```toml
[evidence]
enabled = true
include_raw_payloads = true
include_request_headers = true
max_bytes = 10737418240        # 10 GiB evidence envelope
prune_to_bytes = 8589934592

[evidence.shadow]
enabled = true
keep_looping_attempt_running = true
parallel_downgrade_attempts = true
max_shadow_attempts_per_request = 2
max_global_shadow_in_flight = 4
shadow_attempt_timeout_ms = 300000
compare_attempts = ["bounded-thinking", "no-thinking", "cot-salvage"]

[evidence.shadow.paired_comparison]
enabled = true
variants = ["max-thinking", "bounded-thinking", "no-thinking"]
include_raw_input = true
include_raw_output = true
include_raw_reasoning = true
sample_rate = 1.0
max_bytes = 8589934592          # 8 GiB paired raw artifact retention
max_age_days = 14
```

### Safety notes

- **Raw payload and reasoning capture is sensitive.** Even with header
  redaction, stored raw request/response bodies and chain-of-thought text
  should be treated as confidential operator debug data. Keep SQLite and blob
  directories at `0700` and do not expose debug endpoints beyond trusted
  networks.
- **Selected headers are redacted** for known sensitive keys (authorization,
  api-key, etc.), but operators should still audit which headers are retained.
- **Raw CoT is not released downstream** to the caller, but it **is** persisted
  in the evidence store when `include_raw_reasoning = true`. This data never
  leaves the evidence SQLite/blob storage and is pruned by retention limits.
- **Shadow and paired comparison attempts consume GPU.** In quality-debug mode
  the proxy intentionally runs extra model calls (bounded-thinking, no-thinking,
  cot-salvage variants) alongside the primary request. These are evidence-only
  and do not affect the client-visible response, but they do consume vLLM
  capacity.
- **`compare_attempts` vs `paired_comparison.variants`:** `compare_attempts`
  controls shadow downgrade attempts that run alongside the primary request
  (evidence only, no effect on client response). `paired_comparison.variants`
  controls same-prompt alternatives that run after a successful primary request
  for offline quality comparison. CoT-salvage is meaningful only for
  loop-failure shadow continuation where a failed reasoning prefix exists; a
  paired successful request will not produce a `cot-salvage` variant.

### What affects client-visible quality vs evidence-only collection

| Setting | Client-visible effect | Evidence/debug effect |
|---------|----------------------|----------------------|
| `evidence.enabled` | None | Enables persistent request/response audit trail |
| `evidence.include_raw_payloads` | None | Stores full request/response bodies |
| `evidence.shadow.enabled` | None | Runs extra model attempts for evidence |
| `evidence.shadow.keep_looping_attempt_running` | None | Keeps looping attempt alive for shadow analysis instead of canceling |
| `evidence.shadow.compare_attempts` | None | Runs downgrade variants for offline comparison |
| `evidence.shadow.paired_comparison.enabled` | None | Runs same-prompt alternatives post-success |
| `loop_guard.mode = "enforce"` | May abort and retry looping requests | Records loop signals |
| `loop_guard.mode = "monitor"` | None (observe only) | Records loop signals without acting |
| `thinking.force_disable` | Disables thinking for all requests | None |

## Workspace build path (avoiding cargo-install dependency skew)

For GB10 reviewed-main deployments with workspace crates and path dependencies,
**prefer a local workspace checkout build** over `cargo install`-ing only the
binary crate from GitHub. The binary crate depends on `llm-guard-proxy-core`
via a path dependency; `cargo install` from a remote URL can resolve against a
stale or incomplete version of the core crate, causing missing exports such as
`llm_guard_proxy_core::embedding`, `LoopFailurePolicy`, or
`ShadowComparisonAttempt`.

### Recommended build sequence

```bash
# 1. Fetch and checkout the reviewed main branch
cd ~/project/github/RyderFreeman4Logos/llm-guard-proxy
git fetch origin
git checkout main
git pull origin main

# 2. Build the release binary in the workspace context
#    Use a persistent target dir to speed up incremental rebuilds
CARGO_TARGET_DIR=/ssd/llm-guard-proxy-target \
  cargo build --release -p llm-guard-proxy

# 3. Atomically relink the service binary
install -Dm755 \
  /ssd/llm-guard-proxy-target/release/llm-guard-proxy \
  /home/obj/.local/bin/llm-guard-proxy

# 4. Reload systemd and restart
systemctl --user daemon-reload
systemctl --user restart llm-guard-proxy.service

# 5. Verify
# Confirm the running process is using the new binary
readlink /proc/$(systemctl --user show -p MainPID --value llm-guard-proxy.service)/exe
# Check health
curl --fail --silent http://gb10:18009/health
# Check model metadata
curl --fail --silent http://gb10:18009/v1/models | jq '.data[] | {id, max_model_len}'
# Run a representative chat completion (see Verification section above)
```

### Why not `mise cargo install`?

The `mise use -g 'cargo:https://github.com/...@branch:main'` approach fetches
only the binary crate and lets cargo resolve the core dependency independently.
When the core crate has new public exports (types, traits, config fields), the
resolver may pick an older cached version that lacks those exports, producing
compile errors like:

```text
error[E0432]: unresolved import `llm_guard_proxy_core::embedding`
error[E0433]: failed to resolve: use of undeclared type `LoopFailurePolicy`
```

Building from a local workspace checkout guarantees that the binary and core
crates are compiled together from the same commit.
