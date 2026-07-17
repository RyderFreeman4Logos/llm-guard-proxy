import os
import subprocess
import tempfile
import unittest
from pathlib import Path


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
        metadata_violation: str | None = None,
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

            systemctl_log = root / "systemctl.log"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_systemctl = fake_bin / "systemctl"
            fake_systemctl.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'printf \'%s\\n\' "$*" >>"${SYSTEMCTL_LOG}"\n'
                'case "$*" in\n'
                '  *"show"*"gb10-memory-guardian.service"*) '
                'printf \'%s\\n\' "${LEGACY_LOAD_STATE}" ;;\n'
                "esac\n"
            )
            fake_systemctl.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["SYSTEMCTL_LOG"] = str(systemctl_log)
            env["LEGACY_LOAD_STATE"] = legacy_load_state
            completed = subprocess.run(
                [str(CUTOVER), str(registration_path)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
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


if __name__ == "__main__":
    unittest.main()
