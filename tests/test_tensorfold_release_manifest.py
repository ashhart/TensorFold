from __future__ import annotations

from pathlib import Path
import unittest


class TensorFoldReleaseManifestTests(unittest.TestCase):
    def test_pyproject_exposes_tensorfold_public_cli(self) -> None:
        pyproject = Path("pyproject.toml").read_text()

        self.assertIn("[project.scripts]", pyproject)
        self.assertIn('tensorfold = "tensorfold.cli:main"', pyproject)
        self.assertNotIn('smarttensor = "smarttensor.cli:main"', pyproject)

    def test_manifest_excludes_private_research_artifacts(self) -> None:
        manifest = Path("MANIFEST.in").read_text()

        self.assertIn("exclude findings.md", manifest)
        self.assertIn("exclude HANDOVER.md", manifest)
        self.assertIn("prune logs", manifest)
        self.assertIn("prune docs/superpowers", manifest)
        self.assertIn("prune tests/SmartTensor-*", manifest)

    def test_manifest_includes_runtime_docs_and_scrub_tool(self) -> None:
        manifest = Path("MANIFEST.in").read_text()

        self.assertIn("include README.md", manifest)
        self.assertIn("recursive-include docs *.md", manifest)
        self.assertIn("recursive-include examples *.py", manifest)
        self.assertIn("include tools/check_public_scrub.py", manifest)


if __name__ == "__main__":
    unittest.main()
