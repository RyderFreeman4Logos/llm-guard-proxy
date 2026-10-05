#!/usr/bin/env python3
"""Validate coordinator-pinned native evidence, not self-authenticating JSON.

The caller must obtain LLM_GUARD_NATIVE_REVIEW_SHA256 independently from the
reviewer/coordinator, never derive it from an untrusted receipt at admission.
This is an accidental/stale-artifact boundary, NOT protection against a hostile
same-UID caller controlling the hook environment or candidate scripts.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCHEMA = "llm-guard-proxy.native-review/v1"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate receipt key")
    return result


def _require(payload: dict, expected: dict) -> None:
    for key, value in expected.items():
        actual = payload.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{key} mismatch")


def _artifact(payload: dict, prefix: str) -> bytes:
    path = payload[prefix + "_path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{prefix} requires an absolute path")
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_uid != os.getuid():
        raise ValueError(f"{prefix} must be a user-owned regular file, not a symlink")
    data = artifact.read_bytes()
    if not data or sha256(data) != payload[prefix + "_sha256"]:
        raise ValueError(f"{prefix} hash mismatch or empty artifact")
    return data


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: native-review-receipt.py RECEIPT")
        raw = Path(sys.argv[1]).read_bytes()
        trusted = os.environ.get("LLM_GUARD_NATIVE_REVIEW_SHA256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", trusted) or not hmac.compare_digest(trusted, sha256(raw)):
            raise ValueError("missing or mismatched coordinator receipt digest")
        receipt = Path(sys.argv[1])
        if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_uid != os.getuid():
            raise ValueError("receipt must be a user-owned regular file, not a symlink")
        payload = json.loads(raw, object_pairs_hook=_pairs)
        if not isinstance(payload, dict):
            raise ValueError("receipt must be an object")
        allowed = {"schema", "verdict", "actionable_findings", "review_complete",
                   "independent_review", "repository", "base_ref", "range", "branch",
                   "base", "merge_base", "head", "tree", "clean", "diff_command",
                   "diff_bytes", "diff_sha256", "reviewer", "report_path",
                   "report_sha256", "full_gate"}
        if set(payload) != allowed:
            raise ValueError("missing or unknown receipt fields")
        # The scope is policy, not chosen by the receipt being verified.
        head = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        base = git("rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
        merge_base = git("merge-base", base, head)
        diff = subprocess.check_output(["git", "diff", merge_base, head, "--binary", "--full-index"])
        _require(payload, {
            "schema": SCHEMA, "verdict": "PASS", "actionable_findings": 0,
            "review_complete": True, "independent_review": True,
            "repository": git("rev-parse", "--show-toplevel"),
            "base_ref": "origin/main", "range": "origin/main...HEAD",
            "branch": git("branch", "--show-current"), "base": base, "merge_base": merge_base,
            "head": head, "tree": tree, "clean": True,
            "diff_command": f"git diff {merge_base} {head} --binary --full-index",
            "diff_bytes": len(diff), "diff_sha256": sha256(diff),
        })
        if not isinstance(payload.get("reviewer"), str) or not payload["reviewer"].strip():
            raise ValueError("missing independent reviewer identity")
        report = _artifact(payload, "report").decode("utf-8")
        lines = report.strip().splitlines()
        if (lines[0] != "VERDICT: PASS" or lines[-1] != "VERDICT: PASS"
                or "REVIEW_COMPLETE: true" not in lines
                or "ACTIONABLE_FINDINGS: 0" not in lines
                or re.search(r"VERDICT:\s*FAIL", report)):
            raise ValueError("report is not a completed zero-finding PASS")
        for line in ("REVIEW_COMPLETE: true", "ACTIONABLE_FINDINGS: 0",
                     f"HEAD: {head}", f"TREE: {tree}", f"BASE: {base}",
                     "RANGE: origin/main...HEAD", "REVIEWER: " + payload["reviewer"]):
            prefix = line.split(":", 1)[0] + ":"
            if [item for item in lines if item.lstrip().startswith(prefix)] != [line]:
                raise ValueError("report provenance missing, duplicate or contradictory")
        gate = payload["full_gate"]
        if not isinstance(gate, dict):
            raise ValueError("full_gate must be an object")
        if set(gate) != {"command", "exit_code", "head", "tree", "complete",
                         "clean_before", "clean_after", "log_path", "log_sha256"}:
            raise ValueError("missing or unknown gate fields")
        _require(gate, {"command": "just pre-push", "exit_code": 0,
                        "head": head, "tree": tree, "complete": True,
                        "clean_before": True, "clean_after": True})
        _artifact(gate, "log")
        if git("status", "--porcelain", "--untracked-files=all") or git("write-tree") != tree:
            raise ValueError("dirty index or worktree")
        for marker in ("index.lock", "CHERRY_PICK_HEAD", "MERGE_HEAD"):
            if Path(git("rev-parse", "--git-path", marker)).exists():
                raise ValueError(f"active Git transaction: {marker}")
        print(f"native full-range review and full gate match HEAD {head}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"native review receipt rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
