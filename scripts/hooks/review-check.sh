#!/usr/bin/env bash
# Git pre-push hook: verify csa review has been run on current HEAD.
# Installed by: csa setup review-gate
#
# Validate an explicit native receipt or the CSA session verdict.
# A SHA-pinned marker (.csa/state/review-gate/<branch_safe>-<short_sha>.pass) may exist
#   from a prior run; it is informational only (logged, never a standalone pass), so a
#   stale or forged marker cannot satisfy the gate without a recorded passing review.

set -euo pipefail

# Never allow executor context or skip flags to manufacture review admission.
if [ "${CSA_SKIP_REVIEW_CHECK:-0}" != "0" ] || [ -n "${CSA_SESSION_ID:-}" ] || [ "${CSA_DEPTH:-0}" != "0" ]; then
  echo "ERROR: Review skip/executor flags cannot satisfy publication review." >&2
  exit 1
fi

# Explicit coordinator-pinned native evidence is validated, never a CSA marker.
NATIVE_RECEIPT="${LLM_GUARD_NATIVE_REVIEW_RECEIPT:-}"
if [ -n "${NATIVE_RECEIPT}" ]; then
  exec python3 scripts/hooks/native-review-receipt.py "${NATIVE_RECEIPT}"
fi
if [ -n "${LLM_GUARD_NATIVE_REVIEW_SHA256:-}" ]; then
  echo "ERROR: Native review digest supplied without receipt." >&2
  exit 1
fi
if ! command -v csa >/dev/null 2>&1; then
  echo "ERROR: CSA unavailable; supply genuine coordinator-pinned native review evidence." >&2
  exit 1
fi

CURRENT_HEAD="$(git rev-parse HEAD)"
CURRENT_BRANCH="$(git branch --show-current)"

# Skip protected branches here; pre-push branch-protection blocks them first.
PROTECTED="main dev master"
for pb in $PROTECTED; do
  if [ "${CURRENT_BRANCH}" = "$pb" ]; then
    exit 0
  fi
done

# ── Fast path: SHA-pinned marker file ────────────────────────────────────────
# Sanitize branch name the same way review_gate::sanitize_branch does:
#   '/' → '__', any non-[a-zA-Z0-9._-] → '_'
_sanitize_branch() {
  printf '%s' "$1" \
    | sed 's|/|__|g' \
    | sed 's|[^a-zA-Z0-9._-]|_|g'
}

SHORT_SHA="${CURRENT_HEAD:0:11}"
SAFE_BRANCH="$(_sanitize_branch "${CURRENT_BRANCH}")"
MARKER=".csa/state/review-gate/${SAFE_BRANCH}-${SHORT_SHA}.pass"

if [ -f "${MARKER}" ]; then
  echo "pre-push: Review gate marker found for ${SAFE_BRANCH} at ${SHORT_SHA}; validating session."
fi

# ── Session-store validation ─────────────────────────────────────────────────
if csa review --check-verdict; then
  echo "pre-push: Full-diff review verified for HEAD ${SHORT_SHA}."
  exit 0
fi

# ── Blocked — emit reverse prompt injection for agent context ─────────────────
cat >&2 <<GATE_BLOCKED
<!-- CSA:REVIEW_GATE_BLOCKED branch="${SAFE_BRANCH}" head_sha="${CURRENT_HEAD}" -->
Push blocked: no passing review found for current HEAD.
Run: csa review --range main...HEAD --sa-mode true
Wait for PASS verdict, then retry push.
<!-- /CSA:REVIEW_GATE_BLOCKED -->
GATE_BLOCKED

echo "" >&2
echo "ERROR: Push blocked — no PASS/CLEAN full-diff csa review session recorded for ${SAFE_BRANCH} at ${SHORT_SHA}." >&2
exit 1
