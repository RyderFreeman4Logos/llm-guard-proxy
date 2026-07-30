"""Contract tests for the GB10 recovery preflight."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "deploy" / "gb10" / "config.toml"
PREFLIGHT_PATH = REPO_ROOT / "deploy" / "gb10" / "preflight-config.py"


def load_preflight() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gb10_preflight", PREFLIGHT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("GB10 preflight module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gb10RecoveryPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = load_preflight()
        cls.config = cls.preflight.load_config(CONFIG_PATH)

    def test_reviewed_snapshot_passes_dry_run_without_installing(self) -> None:
        minimum = self.preflight.minimum_downstream_idle_timeout_ms(self.config)
        self.assertIsInstance(minimum, int)
        result = subprocess.run(
            [
                sys.executable,
                str(PREFLIGHT_PATH),
                "--config",
                str(CONFIG_PATH),
                "--downstream-idle-timeout-ms",
                str(minimum + 1),
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result=ok mode=dry-run", result.stdout)

    def test_every_required_route_rejects_recovery_downgrades(self) -> None:
        for route in ("default", "aeon-chat"):
            for mutation in ("disabled", "empty-command", "wrong-command", "two-attempts"):
                with self.subTest(route=route, mutation=mutation):
                    candidate = copy.deepcopy(self.config)
                    recovery = (
                        candidate["upstream"]["local_recovery"]
                        if route == "default"
                        else candidate["upstreams"][0]["local_recovery"]
                    )
                    if mutation == "disabled":
                        recovery["enabled"] = False
                    elif mutation == "empty-command":
                        recovery["restart_command"] = []
                    elif mutation == "wrong-command":
                        recovery["restart_command"][-1] = "wrong.service"
                    else:
                        recovery["max_attempts_per_request"] = 2
                    errors, _ = self.preflight.validate_snapshot(candidate, 4_000_000)
                    self.assertTrue(errors)

    def test_guard_workflow_activation_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["guard_workflows"] = {"pre_request": "unexpected"}
        errors, _ = self.preflight.validate_snapshot(candidate, 4_000_000)
        self.assertIn(
            "guard_workflows must remain inactive in the reviewed snapshot", errors
        )

    def test_operator_idle_timeout_must_exceed_conservative_hold_bound(self) -> None:
        minimum = self.preflight.minimum_downstream_idle_timeout_ms(self.config)
        self.assertEqual(minimum, 3_901_000)
        for rejected in (minimum - 1, minimum):
            with self.subTest(rejected=rejected):
                errors, _ = self.preflight.validate_snapshot(self.config, rejected)
                self.assertTrue(any("strictly greater" in error for error in errors))
        errors, _ = self.preflight.validate_snapshot(self.config, minimum + 1)
        self.assertEqual(errors, [])

    def test_hold_bound_derives_completion_guard_and_final_replay(self) -> None:
        retry_deadline = self.config["retry"]["request_deadline_ms"]
        route = self.config["upstream"]
        recovery = route["local_recovery"]
        expected = (
            retry_deadline
            + route["request_timeout_ms"]
            + recovery["restart_timeout_ms"]
            + recovery["readiness_deadline_ms"]
            + self.preflight.RECOVERY_COMPLETION_GUARD_MS
        )
        self.assertEqual(
            self.preflight.minimum_downstream_idle_timeout_ms(self.config), expected
        )


if __name__ == "__main__":
    unittest.main()
