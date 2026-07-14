# Justfile for the llm-guard-proxy Rust workspace.
# Keep this file small; issue #1 only needs local quality gates and hook wiring.

set shell := ["bash", "-c"]
set tempdir := "."
set dotenv-load := true

_repo_root := `git rev-parse --show-superproject-working-tree 2>/dev/null | grep . || git rev-parse --show-toplevel`
local_jobs := env_var_or_default("LLM_GUARD_LOCAL_JOBS", "2")
local_test_threads := env_var_or_default("LLM_GUARD_LOCAL_TEST_THREADS", "2")

default: pre-commit

check-branch:
    scripts/hooks/branch-protection.sh

check-generated-artifacts:
    #!/usr/bin/env bash
    set -euo pipefail
    blocked_paths="$(
        git diff --cached --name-only --diff-filter=ACMR \
            | grep -E '^(target/|\.tmp/|\.test-target/)|(\.log$|_output\.)' || true
    )"
    if [ -n "${blocked_paths}" ]; then
        echo "Generated or scratch artifacts are staged:"
        printf '%s\n' "${blocked_paths}"
        exit 1
    fi

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

contracts:
    python3 -m unittest discover -s tests -p 'test_*.py' -v

clippy: clippy-all-features clippy-feature-matrix

clippy-all-features:
    cargo clippy --workspace --all-targets --all-features --jobs {{local_jobs}} -- -D warnings

clippy-feature-matrix:
    cargo clippy -p llm-guard-proxy --all-targets --no-default-features --jobs {{local_jobs}} -- -D warnings
    cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features guard --jobs {{local_jobs}} -- -D warnings
    cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features param-override --jobs {{local_jobs}} -- -D warnings
    cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features upstream-hot-restart --jobs {{local_jobs}} -- -D warnings

test:
    RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features --jobs {{local_jobs}}

smoke-gb10:
    scripts/smoke-gb10.sh

pre-commit-fast:
    just check-branch
    just check-generated-artifacts
    just contracts
    just fmt-check
    just clippy

pre-commit:
    just pre-commit-fast
    just test

# Authoritative committed-HEAD gate replacing hosted CI.
pre-push:
    just check-branch
    just contracts
    just fmt-check
    just clippy
    just test
