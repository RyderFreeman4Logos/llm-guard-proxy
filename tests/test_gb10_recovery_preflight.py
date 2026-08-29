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

    def test_hold_bound_counts_every_recovery_enabled_profile(self) -> None:
        candidate = copy.deepcopy(self.config)
        longer = copy.deepcopy(candidate["upstreams"][0])
        longer["name"] = "longer-recovery-route"
        longer["request_timeout_ms"] += 60_000
        candidate["upstreams"].append(longer)

        expected = 3_961_000
        self.assertEqual(
            self.preflight.minimum_downstream_idle_timeout_ms(candidate), expected
        )
        errors, _ = self.preflight.validate_snapshot(candidate, expected)
        self.assertTrue(any("strictly greater" in error for error in errors))
    def test_forced_alias_profiles_are_exact_and_immutable(self) -> None:
        profiles = self.config["forced_model_alias_profiles"]
        self.assertEqual(len(profiles), 3)
        expected = {
            "abliterated-qwen-latest-27b-nvfp4-none": {
                "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
                "thinking_mode": "force_disable",
                "output_cap": 16_384,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            },
            "abliterated-qwen-latest-27b-nvfp4-low": {
                "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
                "thinking_mode": "force_thinking",
                "thinking_budget": 65_536,
                "output_cap": 16_384,
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "presence_penalty": 0,
                "repetition_penalty": 1,
            },
            "abliterated-qwen-latest-27b-nvfp4-medium": {
                "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
                "thinking_mode": "force_thinking",
                "thinking_budget": 65_536,
                "output_cap": 16_384,
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "presence_penalty": 0,
                "repetition_penalty": 1,
            },
        }
        self.assertEqual(
            {
                profile["alias"]: {
                    key: value for key, value in profile.items() if key != "alias"
                }
                for profile in profiles
            },
            expected,
        )
        self.assertNotIn(
            "aeon-ultimate",
            {profile["upstream_model"] for profile in profiles},
        )
        for alias, field in [
            ("abliterated-qwen-latest-27b-nvfp4-low", "thinking_budget"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "output_cap"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "temperature"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "top_p"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "top_k"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "min_p"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "presence_penalty"),
            ("abliterated-qwen-latest-27b-nvfp4-none", "repetition_penalty"),
        ]:
            with self.subTest(boolean_numeric_field=field):
                candidate = copy.deepcopy(self.config)
                profile = next(
                    profile
                    for profile in candidate["forced_model_alias_profiles"]
                    if profile["alias"] == alias
                )
                profile[field] = bool(profile[field])
                errors, _ = self.preflight.validate_snapshot(candidate, 4_000_000)
                self.assertTrue(errors)
        for route in ("upstream", "aeon-chat"):
            with self.subTest(boolean_recovery_route=route):
                candidate = copy.deepcopy(self.config)
                recovery = (
                    candidate["upstream"]["local_recovery"]
                    if route == "upstream"
                    else next(
                        profile
                        for profile in candidate["upstreams"]
                        if profile["name"] == route
                    )["local_recovery"]
                )
                recovery["max_attempts_per_request"] = True
                errors, _ = self.preflight.validate_snapshot(candidate, 4_000_000)
                self.assertTrue(errors)
        for mutation, alias, field, value in [
            ("absent", None, None, None),
            ("extra", None, None, None),
            ("upstream", "abliterated-qwen-latest-27b-nvfp4-none", "upstream_model", "wrong"),
            ("thinking mode", "abliterated-qwen-latest-27b-nvfp4-none", "thinking_mode", "force_thinking"),
            ("thinking budget", "abliterated-qwen-latest-27b-nvfp4-low", "thinking_budget", 1),
            ("output", "abliterated-qwen-latest-27b-nvfp4-none", "output_cap", 1),
            ("temperature", "abliterated-qwen-latest-27b-nvfp4-none", "temperature", 1),
            ("top_p", "abliterated-qwen-latest-27b-nvfp4-none", "top_p", 1),
            ("top_k", "abliterated-qwen-latest-27b-nvfp4-none", "top_k", 1),
            ("min_p", "abliterated-qwen-latest-27b-nvfp4-none", "min_p", 1),
            ("presence_penalty", "abliterated-qwen-latest-27b-nvfp4-none", "presence_penalty", 0),
            ("repetition_penalty", "abliterated-qwen-latest-27b-nvfp4-none", "repetition_penalty", 0.5),
        ]:
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(self.config)
                candidate_profiles = candidate["forced_model_alias_profiles"]
                if mutation == "absent":
                    candidate.pop("forced_model_alias_profiles")
                elif mutation == "extra":
                    extra = copy.deepcopy(candidate_profiles[0])
                    extra["alias"] = "abliterated-qwen-latest-27b-nvfp4-xhigh"
                    candidate_profiles.append(extra)
                else:
                    profile = next(profile for profile in candidate_profiles if profile["alias"] == alias)
                    profile[field] = value
                errors, _ = self.preflight.validate_snapshot(candidate, 4_000_000)
                self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
