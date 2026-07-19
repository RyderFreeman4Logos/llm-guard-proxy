import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CUTOVER = ROOT / "deploy" / "gb10" / "cutover-guardian.sh"


class Gb10GuardianCutoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(CUTOVER.is_file(), "the tracked cutover helper must exist")
        self.assertTrue(os.access(CUTOVER, os.X_OK), "the cutover helper must be executable")

    def run_cutover(
        self,
        registration: str,
        legacy_load_state: str = "not-found",
        legacy_unit_file_state: str = "enabled",
        legacy_active_state: str = "active",
        integrated_unit_file_state: str = "disabled",
        integrated_active_state: str = "inactive",
        metadata_violation: str | None = None,
        target_state: str = "populated",
        integrated_enable_failure: bool = False,
        integrated_active_hang: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration_path = root / "text-cgroup.v1"
            registration_source = (
                root / "registration-source.v1"
                if metadata_violation == "symlink"
                else registration_path
            )
            registration_source.write_text(registration)
            registration_source.chmod(0o600)
            if metadata_violation == "mode":
                registration_source.chmod(0o640)
            elif metadata_violation == "hardlink":
                os.link(registration_source, root / "registration-link.v1")
            elif metadata_violation == "symlink":
                registration_path.symlink_to(registration_source)

            cgroup_root = root / "cgroup"
            control_group = next(
                (
                    line.removeprefix("control_group=")
                    for line in registration.splitlines()
                    if line.startswith("control_group=")
                ),
                "",
            )
            if target_state != "missing" and control_group:
                target = cgroup_root / control_group.removeprefix("/")
                target.mkdir(parents=True)
                populated = "1" if target_state == "populated" else "0"
                (target / "cgroup.events").write_text(f"populated {populated}\n")

            systemctl_log = root / "systemctl.log"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_systemctl = fake_bin / "systemctl"
            fake_systemctl.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'printf \'%s\\n\' "$*" >>"${SYSTEMCTL_LOG}"\n'
                'case "$*" in\n'
                '  *"show --property=LoadState"*"gb10-memory-guardian.service"*) '
                'printf \'%s\\n\' "${LEGACY_LOAD_STATE}" ;;\n'
                '  *"show --property=UnitFileState"*"gb10-memory-guardian.service"*) '
                'printf \'%s\\n\' "${LEGACY_UNIT_FILE_STATE}" ;;\n'
                '  *"show --property=ActiveState"*"gb10-memory-guardian.service"*) '
                'printf \'%s\\n\' "${LEGACY_ACTIVE_STATE}" ;;\n'
                '  *"show --property=UnitFileState"*"llm-guard-proxy.service"*) '
                'printf \'%s\\n\' "${INTEGRATED_UNIT_FILE_STATE}" ;;\n'
                '  *"show --property=ActiveState"*"llm-guard-proxy.service"*) '
                'printf \'%s\\n\' "${INTEGRATED_ACTIVE_STATE}" ;;\n'
                '  *"enable --now llm-guard-proxy.service"*) '
                '[[ "${INTEGRATED_ENABLE_FAILURE}" != 1 ]] ;;\n'
                '  *"is-active llm-guard-proxy.service"*) '
                'if [[ "${INTEGRATED_ACTIVE_HANG}" == 1 ]]; then sleep 5; fi; '
                'printf \'active\\n\' ;;\n'
                "esac\n"
            )
            fake_systemctl.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["SYSTEMCTL_LOG"] = str(systemctl_log)
            env["LEGACY_LOAD_STATE"] = legacy_load_state
            env["LEGACY_UNIT_FILE_STATE"] = legacy_unit_file_state
            env["LEGACY_ACTIVE_STATE"] = legacy_active_state
            env["INTEGRATED_UNIT_FILE_STATE"] = integrated_unit_file_state
            env["INTEGRATED_ACTIVE_STATE"] = integrated_active_state
            env["INTEGRATED_ENABLE_FAILURE"] = "1" if integrated_enable_failure else "0"
            env["INTEGRATED_ACTIVE_HANG"] = "1" if integrated_active_hang else "0"
            env["LLM_GUARD_CGROUP_ROOT"] = str(cgroup_root)
            env["LLM_GUARD_SYSTEMCTL_TIMEOUT"] = "0.1s"
            command = [str(CUTOVER), str(registration_path)]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1,
                )
            except subprocess.TimeoutExpired as error:
                completed = subprocess.CompletedProcess(
                    command,
                    124,
                    stdout=error.stdout.decode() if error.stdout else "",
                    stderr="cutover exceeded the test's outer deadline",
                )
            calls = systemctl_log.read_text().splitlines() if systemctl_log.exists() else []
            return completed, calls

    @staticmethod
    def registration(container_id: str, scope_id: str) -> str:
        uid = os.getuid()
        scope = f"docker-{scope_id}.scope"
        return (
            "version=1\n"
            f"container_id={container_id}\n"
            f"scope={scope}\n"
            f"control_group=/user.slice/user-{uid}.slice/"
            f"user@{uid}.service/app.slice/{scope}\n"
        )

    def test_invalid_registration_fails_before_any_systemctl_call(self) -> None:
        completed, calls = self.run_cutover(self.registration("a" * 64, "b" * 64))

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(calls, [])

    def test_insecure_registration_metadata_fails_before_any_systemctl_call(self) -> None:
        valid_registration = self.registration("a" * 64, "a" * 64)
        for violation in ("mode", "hardlink", "symlink"):
            with self.subTest(violation=violation):
                completed, calls = self.run_cutover(
                    valid_registration, metadata_violation=violation
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(calls, [])

    def test_parser_contract_failures_happen_before_any_systemctl_call(self) -> None:
        valid_registration = self.registration("a" * 64, "a" * 64)
        violations = {
            "nul": valid_registration.replace("version=1", "version=1\0"),
            "oversized": valid_registration + ("x" * 1024),
        }
        for violation, registration in violations.items():
            with self.subTest(violation=violation):
                completed, calls = self.run_cutover(registration)

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(calls, [])

    def test_valid_docker_scope_cuts_over_when_legacy_unit_is_absent(self) -> None:
        completed, calls = self.run_cutover(self.registration("a" * 64, "a" * 64))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(any("disable --now gb10-memory-guardian.service" in call for call in calls))
        self.assertTrue(any("enable --now llm-guard-proxy.service" in call for call in calls))
        self.assertTrue(any("is-active llm-guard-proxy.service" in call for call in calls))

    def test_existing_legacy_guardian_is_disabled_before_integrated_enable(self) -> None:
        completed, calls = self.run_cutover(
            self.registration("a" * 64, "a" * 64), legacy_load_state="loaded"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        disable_index = next(
            index
            for index, call in enumerate(calls)
            if "disable --now gb10-memory-guardian.service" in call
        )
        enable_index = next(
            index
            for index, call in enumerate(calls)
            if "enable --now llm-guard-proxy.service" in call
        )
        self.assertLess(disable_index, enable_index)

    def test_stale_or_empty_cgroup_fails_before_any_systemctl_call(self) -> None:
        valid_registration = self.registration("a" * 64, "a" * 64)
        for target_state in ("missing", "empty"):
            with self.subTest(target_state=target_state):
                completed, calls = self.run_cutover(
                    valid_registration, target_state=target_state
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(calls, [])

    def test_integrated_enable_failure_restores_prior_legacy_state(self) -> None:
        completed, calls = self.run_cutover(
            self.registration("a" * 64, "a" * 64),
            legacy_load_state="loaded",
            integrated_enable_failure=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        failed_enable_index = next(
            index
            for index, call in enumerate(calls)
            if "enable --now llm-guard-proxy.service" in call
        )
        rollback_enable_index = next(
            (
                index
                for index, call in enumerate(calls)
                if index > failed_enable_index
                and call.endswith("enable gb10-memory-guardian.service")
            ),
            -1,
        )
        rollback_start_index = next(
            (
                index
                for index, call in enumerate(calls)
                if index > rollback_enable_index
                and call.endswith("start gb10-memory-guardian.service")
            ),
            -1,
        )
        self.assertGreater(rollback_enable_index, failed_enable_index)
        self.assertLess(rollback_enable_index, rollback_start_index)

    def test_active_check_timeout_restores_prior_legacy_state(self) -> None:
        completed, calls = self.run_cutover(
            self.registration("a" * 64, "a" * 64),
            legacy_load_state="loaded",
            integrated_active_hang=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        active_check_index = next(
            index
            for index, call in enumerate(calls)
            if call.endswith("is-active llm-guard-proxy.service")
        )
        self.assertTrue(
            any(
                index > active_check_index
                and call.endswith("start gb10-memory-guardian.service")
                for index, call in enumerate(calls)
            )
        )

    def test_rollback_restores_every_accepted_unit_file_state_exactly(self) -> None:
        expected_commands = {
            "enabled": ["enable"],
            "enabled-runtime": ["disable", "enable --runtime"],
            "disabled": ["disable"],
            "static": [],
            "indirect": [],
            "generated": [],
            "transient": [],
        }
        valid_registration = self.registration("a" * 64, "a" * 64)
        for unit_kind in ("integrated", "legacy"):
            for unit_file_state, expected_command in expected_commands.items():
                with self.subTest(unit=unit_kind, unit_file_state=unit_file_state):
                    arguments: dict[str, Any] = {
                        "legacy_load_state": "loaded",
                        "legacy_active_state": "inactive",
                        "integrated_active_state": "inactive",
                        "integrated_enable_failure": True,
                    }
                    arguments[f"{unit_kind}_unit_file_state"] = unit_file_state
                    completed, calls = self.run_cutover(valid_registration, **arguments)

                    self.assertNotEqual(completed.returncode, 0)
                    failed_enable_index = next(
                        index
                        for index, call in enumerate(calls)
                        if call.endswith("enable --now llm-guard-proxy.service")
                    )
                    rollback_calls = calls[failed_enable_index + 1 :]
                    unit = (
                        "llm-guard-proxy.service"
                        if unit_kind == "integrated"
                        else "gb10-memory-guardian.service"
                    )
                    enablement_calls = [
                        call
                        for call in rollback_calls
                        if call.endswith(unit)
                        and (" enable " in f" {call} " or " disable " in f" {call} ")
                    ]
                    self.assertEqual(
                        enablement_calls,
                        [f"--user {command} {unit}" for command in expected_command],
                    )

    def test_rollback_restores_every_accepted_activity_state_exactly(self) -> None:
        valid_registration = self.registration("a" * 64, "a" * 64)
        for unit_kind in ("integrated", "legacy"):
            for active_state, expected_command in (
                ("active", "start"),
                ("inactive", "stop"),
            ):
                with self.subTest(unit=unit_kind, active_state=active_state):
                    arguments: dict[str, Any] = {
                        "legacy_load_state": "loaded",
                        "legacy_unit_file_state": "disabled",
                        "integrated_unit_file_state": "disabled",
                        "integrated_enable_failure": True,
                    }
                    arguments[f"{unit_kind}_active_state"] = active_state
                    completed, calls = self.run_cutover(valid_registration, **arguments)

                    self.assertNotEqual(completed.returncode, 0)
                    failed_enable_index = next(
                        index
                        for index, call in enumerate(calls)
                        if call.endswith("enable --now llm-guard-proxy.service")
                    )
                    rollback_calls = calls[failed_enable_index + 1 :]
                    unit = (
                        "llm-guard-proxy.service"
                        if unit_kind == "integrated"
                        else "gb10-memory-guardian.service"
                    )
                    activity_calls = [
                        call
                        for call in rollback_calls
                        if call.endswith(unit)
                        and (" start " in f" {call} " or " stop " in f" {call} ")
                    ]
                    self.assertEqual(activity_calls, [f"--user {expected_command} {unit}"])

    def test_failed_activity_state_is_rejected_before_mutation(self) -> None:
        valid_registration = self.registration("a" * 64, "a" * 64)
        for unit_kind in ("integrated", "legacy"):
            with self.subTest(unit=unit_kind):
                arguments: dict[str, Any] = {"legacy_load_state": "loaded"}
                arguments[f"{unit_kind}_active_state"] = "failed"
                completed, calls = self.run_cutover(valid_registration, **arguments)

                self.assertNotEqual(completed.returncode, 0)
                mutating_calls = [
                    call
                    for call in calls
                    if any(
                        f" {command} " in f" {call} "
                        for command in ("enable", "disable", "start", "stop")
                    )
                ]
                self.assertEqual(mutating_calls, [])


if __name__ == "__main__":
    unittest.main()
