"""Exercise the real hook and validator in disposable, non-Rust Git fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NativeReviewReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CSA_", "LLM_GUARD_NATIVE_"))}
        self.env.update({"GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.test", "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.test"})
        self.git("init", "-b", "main")
        hooks = self.repo / "scripts/hooks"
        hooks.mkdir(parents=True)
        for name in ("review-check.sh", "native-review-receipt.py"):
            if (ROOT / "scripts/hooks" / name).exists():
                shutil.copyfile(ROOT / "scripts/hooks" / name, hooks / name)
        self.git("add", "scripts")
        self.git("commit", "-m", "fixture base")
        self.base = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.git("checkout", "-b", "fixture")
        (self.repo / "source").write_text("change\n")
        self.git("add", "source")
        self.git("commit", "-m", "fixture candidate")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        csa = self.bin / "csa"
        csa.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CSA_FIXTURE_CALLS"\nexit "${CSA_FIXTURE_RC:-1}"\n')
        csa.chmod(0o755)
        self.calls = self.root / "calls"
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env["PATH"]
        self.env["CSA_FIXTURE_CALLS"] = str(self.calls)
        self.report = self.root / "review.md"
        self.report.write_text("VERDICT: PASS\nREVIEW_COMPLETE: true\nACTIONABLE_FINDINGS: 0\nIndependent fixture review.\nVERDICT: PASS\n")
        self.gate_log = self.root / "gate.log"
        self.gate_log.write_text("fixture gate output, not a real product gate\n")
        head = self.git("rev-parse", "HEAD")
        tree = self.git("rev-parse", "HEAD^{tree}")
        diff = self.command("git", "diff", self.base, head, "--binary", "--full-index").stdout
        self.payload = {
            "schema": "llm-guard-proxy.native-review/v1", "verdict": "PASS", "actionable_findings": 0,
            "review_complete": True, "independent_review": True, "reviewer": "native-fixture",
            "repository": str(self.repo), "base_ref": "origin/main", "range": "origin/main...HEAD", "branch": "fixture", "base": self.base, "merge_base": self.base,
            "head": head, "tree": tree, "clean": True,
            "diff_command": f"git diff {self.base} {head} --binary --full-index",
            "diff_bytes": len(diff), "diff_sha256": digest(diff),
            "report_path": str(self.report), "report_sha256": digest(self.report.read_bytes()),
            "full_gate": {"command": "just pre-push", "exit_code": 0, "head": head, "tree": tree,
                          "complete": True, "clean_before": True, "clean_after": True,
                          "log_path": str(self.gate_log), "log_sha256": digest(self.gate_log.read_bytes())},
        }
        self.report.write_text(f"VERDICT: PASS\nREVIEW_COMPLETE: true\nACTIONABLE_FINDINGS: 0\nHEAD: {head}\nTREE: {tree}\nBASE: {self.base}\nRANGE: origin/main...HEAD\nREVIEWER: native-fixture\nVERDICT: PASS\n")
        self.payload["report_sha256"] = digest(self.report.read_bytes())
        self.receipt = self.root / "receipt.json"

    def command(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=self.repo, env=self.env, capture_output=True, check=True)

    def git(self, *args: str) -> str:
        return self.command("git", *args).stdout.decode().strip()

    def hook(self, payload: dict | None = None, *, trusted: bool = True) -> subprocess.CompletedProcess:
        self.receipt.write_text(json.dumps(self.payload if payload is None else payload))
        env = self.env.copy()
        env["LLM_GUARD_NATIVE_REVIEW_RECEIPT"] = str(self.receipt)
        if trusted:
            env["LLM_GUARD_NATIVE_REVIEW_SHA256"] = digest(self.receipt.read_bytes())
        return subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True, text=True)

    def test_exact_trusted_native_positive_without_provider_or_mutation(self) -> None:
        result = self.hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_invalid_receipts_fail_closed(self) -> None:
        mutations = {
            "stale_head": {"head": "0" * 40}, "wrong_tree": {"tree": "0" * 40},
            "wrong_base": {"base": self.payload["head"]}, "narrow_scope": {"base_ref": "HEAD^"},
            "wrong_merge_base": {"merge_base": self.payload["head"]}, "wrong_diff": {"diff_sha256": "0" * 64},
            "fail": {"verdict": "FAIL"}, "findings": {"actionable_findings": 1},
            "unfinished": {"review_complete": False}, "not_independent": {"independent_review": False},
            "unknown_schema": {"schema": "unknown"}, "missing_report": {"report_path": "/missing-report"},
            "wrong_repository": {"repository": str(self.root)}, "typed_findings": {"actionable_findings": False},
        }
        for name, delta in mutations.items():
            with self.subTest(name=name):
                payload = dict(self.payload, **delta)
                self.assertNotEqual(self.hook(payload).returncode, 0)
        for key, value in (("head", "0" * 40), ("tree", "0" * 40), ("exit_code", 1), ("complete", False), ("clean_before", False), ("command", "true")):
            with self.subTest(gate=key):
                payload = copy.deepcopy(self.payload)
                payload["full_gate"][key] = value
                self.assertNotEqual(self.hook(payload).returncode, 0)
        self.assertFalse(self.calls.exists(), "invalid native must never fall through to CSA")

    def test_missing_trust_and_tampered_artifacts(self) -> None:
        self.assertNotEqual(self.hook(trusted=False).returncode, 0)
        for path in (self.report, self.gate_log):
            original = path.read_bytes()
            path.write_bytes(original + b"tampered")
            self.assertNotEqual(self.hook().returncode, 0)
            path.write_bytes(original)
        self.report.write_text("VERDICT: FAIL\n")
        self.payload["report_sha256"] = digest(self.report.read_bytes())
        self.assertNotEqual(self.hook().returncode, 0)

    def test_same_tree_new_commit_and_dirty_worktree_are_stale(self) -> None:
        (self.repo / "source").write_text("dirty\n")
        self.assertNotEqual(self.hook().returncode, 0)
        (self.repo / "source").write_text("change\n")
        self.git("commit", "--allow-empty", "-m", "different commit same tree")
        self.assertNotEqual(self.hook().returncode, 0)

    def test_supplied_invalid_native_precedes_legacy_skip(self) -> None:
        self.env["CSA_SKIP_REVIEW_CHECK"] = "1"
        self.assertNotEqual(self.hook({}).returncode, 0)

    def test_csa_route_and_marker_is_not_authority(self) -> None:
        self.env["CSA_FIXTURE_RC"] = "0"
        self.command("bash", "scripts/hooks/review-check.sh")
        self.assertEqual(self.calls.read_text().strip(), "review --check-verdict")
        self.env["CSA_FIXTURE_RC"] = "1"
        marker = self.repo / ".csa/state/review-gate" / f"fixture-{self.payload['head'][:11]}.pass"
        marker.parent.mkdir(parents=True)
        marker.write_text("forged")
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_duplicate_keys_and_receipt_tampering(self) -> None:
        self.assertEqual(self.hook().returncode, 0)
        env = self.env.copy()
        env.update(LLM_GUARD_NATIVE_REVIEW_RECEIPT=str(self.receipt), LLM_GUARD_NATIVE_REVIEW_SHA256=digest(self.receipt.read_bytes()))
        self.receipt.write_text(self.receipt.read_text() + " ")
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.receipt.write_text(self.receipt.read_text().rstrip()[:-1] + ', "verdict": "PASS"}')
        env["LLM_GUARD_NATIVE_REVIEW_SHA256"] = digest(self.receipt.read_bytes())
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_skip_and_claimed_executor_flags_never_admit(self) -> None:
        for key, value in (("CSA_SKIP_REVIEW_CHECK", "1"), ("CSA_SESSION_ID", "forged"), ("CSA_DEPTH", "1"), ("CSA_DEPTH", "malformed")):
            with self.subTest(key=key, value=value):
                self.env[key] = value
                self.assertNotEqual(self.hook().returncode, 0)
                result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=self.env, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                del self.env[key]

    def test_unknown_missing_fields_and_report_scope(self) -> None:
        self.assertNotEqual(self.hook(dict(self.payload, unknown=True)).returncode, 0)
        for key in self.payload:
            payload = dict(self.payload)
            del payload[key]
            self.assertNotEqual(self.hook(payload).returncode, 0)
        self.report.write_text(self.report.read_text().replace("RANGE: origin/main...HEAD", "RANGE: HEAD^...HEAD"))
        self.payload["report_sha256"] = digest(self.report.read_bytes())
        self.assertNotEqual(self.hook().returncode, 0)

    def test_missing_receipt_and_digest_do_not_fall_through(self) -> None:
        env = self.env.copy()
        env["LLM_GUARD_NATIVE_REVIEW_RECEIPT"] = str(self.root / "absent")
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        del env["LLM_GUARD_NATIVE_REVIEW_RECEIPT"]
        env["LLM_GUARD_NATIVE_REVIEW_SHA256"] = "0" * 64
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_malformed_receipt_and_report_fail_closed(self) -> None:
        self.hook()
        self.receipt.write_text("{malformed")
        env = self.env.copy()
        env.update(LLM_GUARD_NATIVE_REVIEW_RECEIPT=str(self.receipt), LLM_GUARD_NATIVE_REVIEW_SHA256=digest(self.receipt.read_bytes()))
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.report.write_text("VERDICT: PASS\nVERDICT: FAIL\nVERDICT: PASS\n")
        self.payload["report_sha256"] = digest(self.report.read_bytes())
        self.assertNotEqual(self.hook().returncode, 0)

    def test_receipt_symlink_rejected(self) -> None:
        self.assertEqual(self.hook().returncode, 0)
        link = self.root / "receipt-link"
        link.symlink_to(self.receipt)
        env = self.env.copy()
        env.update(LLM_GUARD_NATIVE_REVIEW_RECEIPT=str(link), LLM_GUARD_NATIVE_REVIEW_SHA256=digest(self.receipt.read_bytes()))
        result = subprocess.run(["bash", "scripts/hooks/review-check.sh"], cwd=self.repo, env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_report_and_receipt_symlinks_and_advanced_remote_base(self) -> None:
        link = self.root / "report-link"
        link.symlink_to(self.report)
        self.assertNotEqual(self.hook(dict(self.payload, report_path=str(link))).returncode, 0)
        self.assertEqual(self.hook().returncode, 0)
        self.git("update-ref", "refs/remotes/origin/main", self.payload["head"])
        self.assertNotEqual(self.hook().returncode, 0)
