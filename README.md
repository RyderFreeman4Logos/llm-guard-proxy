# llm-guard-proxy

`llm-guard-proxy` is an Apache-2.0 Rust service intended to become an OpenAI-compatible guard proxy for local and GB10 vLLM deployments.

The proxy will sit between OpenAI-compatible clients and an upstream LLM service so later issues can add debuggability, retries, loop protection, heartbeat behavior, and observability without changing model quality.

## Non-Goals

- This bootstrap does not implement request forwarding.
- This bootstrap does not implement configuration, hot reload, observability, storage, retries, or loop detection.
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

The issue #1 completion gates are:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Later issues will add the real proxy, config, observability, retry, heartbeat, and metadata behavior.
