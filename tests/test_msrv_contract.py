import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Cargo.toml"
JUSTFILE = ROOT / "justfile"


class MsrvContractTests(unittest.TestCase):
    def test_declared_msrv_is_1_88(self) -> None:
        manifest = tomllib.loads(MANIFEST.read_text())
        self.assertEqual(manifest["workspace"]["package"]["rust-version"], "1.88")

    def test_local_gates_enforce_msrv_build(self) -> None:
        justfile = JUSTFILE.read_text()
        # The local pre-push gate must run clippy and tests so that
        # dependency updates that raise the effective MSRV are caught.
        self.assertIn("cargo clippy --workspace --all-targets --all-features", justfile)
        self.assertIn("cargo test --workspace --all-features", justfile)


if __name__ == "__main__":
    unittest.main()
