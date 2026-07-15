import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Cargo.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class MsrvContractTests(unittest.TestCase):
    def test_declared_msrv_matches_exact_ci_toolchain(self) -> None:
        manifest = tomllib.loads(MANIFEST.read_text())
        self.assertEqual(manifest["workspace"]["package"]["rust-version"], "1.88")

        workflow = CI_WORKFLOW.read_text()
        self.assertIn('MSRV: "1.88.0"', workflow)
        self.assertRegex(
            workflow,
            re.compile(r'rustup toolchain install "?\$MSRV"? --profile minimal'),
        )
        self.assertIn(
            'rustup run "$MSRV" cargo check --workspace --all-targets '
            "--all-features --locked",
            workflow,
        )

    def test_ci_runs_repository_contract_tests(self) -> None:
        workflow = CI_WORKFLOW.read_text()
        self.assertIn(
            "python3 -m unittest discover -s tests -p 'test_*.py' -v",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
