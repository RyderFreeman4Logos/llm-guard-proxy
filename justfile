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

build-release:
    {{_io_prefix}} cargo build --release --all-features -p llm-guard-proxy

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

# Required resource-isolated durability acceptance; neither phase is optional.
test:
    {{_io_prefix}} env RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features -- --skip proxy::tests::guardian_recovery::
    {{_io_prefix}} env RUST_TEST_THREADS=1 cargo test --workspace --all-features proxy::tests::guardian_recovery::

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

# Publication reuses validated native evidence; generation always uses pre-push.
pre-push-hook:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "${LLM_GUARD_NATIVE_REVIEW_RECEIPT:-}" ] || [ -n "${LLM_GUARD_NATIVE_REVIEW_SHA256:-}" ]; then
        exec scripts/hooks/review-check.sh
    fi
    exec just pre-push

# Authoritative committed-HEAD gate replacing hosted CI.
pre-push:
    just check-branch
    just contracts
    just fmt-check
    just clippy
    just test

# Explicit host-dependent kernel cgroup regression; never part of default gates.
test-real-cgroup:
    #!/usr/bin/env bash
    set -euo pipefail
    unit="llm-guard-real-cgroup-test-$(python3 -c 'import secrets; print(secrets.token_hex(16))').scope"
    exec {{_io_prefix}} systemd-run --user --scope --quiet --collect \
        --unit="${unit}" --property=Delegate=yes -- \
        cargo test -p llm-guard-proxy-host-guardian --test real_cgroup -- \
        --ignored --exact registered_cgroup_kill_reaps_only_the_task_owned_child --nocapture
