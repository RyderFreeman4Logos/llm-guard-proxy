import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUSTFILE = ROOT / "justfile"
LEFTHOOK = ROOT / "lefthook.yml"
REVIEW_CHECK = ROOT / "scripts" / "hooks" / "review-check.sh"
README = ROOT / "README.md"
WORKFLOWS = ROOT / ".github" / "workflows"


class LocalGateContractTests(unittest.TestCase):
    def test_github_actions_are_disabled(self) -> None:
        workflows = [
            *WORKFLOWS.glob("*.yml"),
            *WORKFLOWS.glob("*.yaml"),
        ]
        self.assertEqual(workflows, [])

    def test_pre_push_is_the_authoritative_local_gate(self) -> None:
        justfile = JUSTFILE.read_text()
        lefthook = LEFTHOOK.read_text()
        review_check = REVIEW_CHECK.read_text()

        self.assertIn("pre-push:", justfile)
        self.assertIn('_io_prefix := "ionice -c 3 nice -n 19"', justfile)
        self.assertIn('local_test_threads := env("LLM_GUARD_LOCAL_TEST_THREADS", "2")', justfile)
        self.assertIn(
            "{{_io_prefix}} env RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features",
            justfile,
        )
        self.assertIn("review-check:", lefthook)
        self.assertIn("run: scripts/hooks/review-check.sh", lefthook)
        self.assertIn("run: just pre-push", lefthook)
        self.assertIn("CSA_SKIP_REVIEW_CHECK", review_check)

    def test_guardian_durability_phase_is_required_and_failure_propagates(self) -> None:
        justfile = JUSTFILE.read_text()
        namespace = "proxy::tests::guardian_recovery::"
        body = justfile.split("\ntest:\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(
            body.splitlines(),
            [
                "    {{_io_prefix}} env RUST_TEST_THREADS={{local_test_threads}} cargo test --workspace --all-features -- --skip " + namespace,
                "    {{_io_prefix}} env RUST_TEST_THREADS=1 cargo test --workspace --all-features " + namespace,
            ],
            "the exact exclusion requires a mandatory all-features isolated selection",
        )
        for recipe in ("pre-commit", "pre-push"):
            commands = justfile.split(f"\n{recipe}:\n", 1)[1].split("\n\n", 1)[0]
            self.assertIn("    just test", commands.splitlines())
        # Execute the real recipe with only Cargo replaced: either phase's failure
        # must fail Just, and normal success must reach the exact isolated selector.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo = root / "cargo"
            cargo.write_text(
                '#!/bin/sh\n'
                'printf "%s|%s\\n" "$RUST_TEST_THREADS" "$*" >> "$GATE_TRACE"\n'
                'case "$*" in *--skip*) phase=normal;; *) phase=isolated;; esac\n'
                '[ "$phase" != "$FAIL_PHASE" ]\n'
            )
            cargo.chmod(0o700)
            trace = root / "trace"
            for failing, expected_count in (("", 2), ("normal", 1), ("isolated", 2)):
                with self.subTest(failing=failing):
                    trace.write_text("")
                    environment = {
                        **os.environ,
                        "PATH": str(root) + os.pathsep + os.environ["PATH"],
                        "GATE_TRACE": str(trace),
                        "FAIL_PHASE": failing,
                        "LLM_GUARD_LOCAL_TEST_THREADS": "2",
                    }
                    result = subprocess.run(
                        ["just", "--justfile", str(JUSTFILE), "test"],
                        env=environment, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode == 0, not failing, result.stderr)
                    expected = [
                        "2|test --workspace --all-features -- --skip " + namespace,
                        "1|test --workspace --all-features " + namespace,
                    ]
                    self.assertEqual(trace.read_text().splitlines(), expected[:expected_count])

    def test_guardian_joint_durability_budget_is_not_relaxed_or_ignored(self) -> None:
        proxy = ROOT / "crates" / "llm-guard-proxy" / "src" / "proxy"
        receipt = (proxy / "recovery_receipt.rs").read_text()
        self.assertEqual(
            re.findall(r"const WRITE_BUDGET: Duration = Duration::from_millis\((\d+)\);", receipt),
            ["250"],
        )
        guardian = (proxy / "tests" / "guardian_recovery.rs").read_text()
        self.assertNotRegex(guardian, r"#\s*\[\s*ignore\b")
        self.assertIn("async fn tier2_cancel_ack_follows_actual_owned_process_reap()", guardian)
        self.assertIn("async fn tier2_terminal_sql_timeout_keeps_reaped_cancellation_unconfirmed()", guardian)

    def test_readme_documents_local_only_linux_x86_64_policy(self) -> None:
        readme = README.read_text()

        self.assertIn("Linux x86_64", readme)
        self.assertIn("authoritative local completion gate", readme)
        self.assertIn("feature development remains", readme)
        self.assertNotIn(".github/workflows", readme)
        self.assertNotIn("same core checks as CI", readme)


if __name__ == "__main__":
    unittest.main()
