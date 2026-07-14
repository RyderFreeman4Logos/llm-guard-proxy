import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUSTFILE = ROOT / "justfile"
LEFTHOOK = ROOT / "lefthook.yml"
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

        self.assertIn("pre-push:", justfile)
        self.assertIn("LLM_GUARD_LOCAL_JOBS", justfile)
        self.assertIn("LLM_GUARD_LOCAL_TEST_THREADS", justfile)
        self.assertIn("run: just pre-push", lefthook)
        self.assertNotIn("review-check:", lefthook)

    def test_readme_documents_local_only_linux_x86_64_policy(self) -> None:
        readme = README.read_text()

        self.assertIn("Linux x86_64", readme)
        self.assertIn("authoritative local completion gate", readme)
        self.assertIn("feature development remains", readme)
        self.assertNotIn(".github/workflows", readme)
        self.assertNotIn("same core checks as CI", readme)


if __name__ == "__main__":
    unittest.main()
