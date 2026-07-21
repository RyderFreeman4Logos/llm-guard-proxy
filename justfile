# Justfile for the llm-guard-proxy Rust workspace.
# Keep this file small; issue #1 only needs local quality gates and hook wiring.

set shell := ["bash", "-c"]
# IO scheduling: run cargo at idle priority to avoid starving interactive processes
_io_prefix := "ionice -c 3 nice -n 19"
# Cap libtest concurrency independently of Cargo build jobs (global cargo config).
local_test_threads := env("LLM_GUARD_LOCAL_TEST_THREADS", "2")
set tempdir := "."
set dotenv-load := true

_repo_root := `git rev-parse --show-superproject-working-tree 2>/dev/null | grep . || git rev-parse --show-toplevel`

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
    {{_io_prefix}} cargo fmt --all

fmt-check:
    {{_io_prefix}} cargo fmt --all -- --check

contracts:
    python3 -m unittest discover -s tests -p 'test_*.py' -v

clippy: clippy-all-features clippy-feature-matrix

clippy-all-features:
    {{_io_prefix}} cargo clippy --workspace --all-targets --all-features -- -D warnings

clippy-feature-matrix:
    {{_io_prefix}} cargo clippy -p llm-guard-proxy --all-targets --no-default-features -- -D warnings
    {{_io_prefix}} cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features guard -- -D warnings
    {{_io_prefix}} cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features param-override -- -D warnings
    {{_io_prefix}} cargo clippy -p llm-guard-proxy --all-targets --no-default-features --features upstream-hot-restart -- -D warnings

test:
    {{_io_prefix}} env RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features

# Focused workspace test runner for TDD and local reproduction.
test-filter filter:
    {{_io_prefix}} env RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features {{filter}}

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
