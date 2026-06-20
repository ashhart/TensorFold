from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.check_public_scrub import find_public_leaks


class PublicScrubTests(unittest.TestCase):
    def test_detects_absolute_home_paths_in_public_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Model lives at /Users/local/dev/model\n")

            leaks = find_public_leaks(root)

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].path.as_posix(), "README.md")
        self.assertEqual(leaks[0].label, "absolute-user-home-path")

    def test_detects_configured_private_terms_in_public_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "quickstart.md").write_text("Contact internalcodename for the local run.\n")

            leaks = find_public_leaks(root, private_terms=("internalcodename",))

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].label, "configured-private-term")

    def test_detects_private_terms_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Do not publish internalcodename.\n")

            with patch.dict(os.environ, {"TENSORFOLD_PRIVATE_SCRUB_TERMS": "internalcodename"}):
                from tools.check_public_scrub import _configured_private_terms

                leaks = find_public_leaks(root, private_terms=_configured_private_terms(()))

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].label, "configured-private-term")

    def test_scrub_source_does_not_ship_private_or_agent_denylists(self) -> None:
        source = Path("tools/check_public_scrub.py").read_text()

        self.assertIn("TENSORFOLD_PRIVATE_SCRUB_TERMS", source)
        self.assertNotIn("private_names", source)
        self.assertNotIn("agent_names", source)
        self.assertNotIn("private-name", source)
        self.assertNotIn("agent-coordination-name", source)

    def test_scans_release_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MANIFEST.in").write_text("include /Users/local/private-file\n")

            leaks = find_public_leaks(root)

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].path.as_posix(), "MANIFEST.in")

    def test_scans_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "examples").mkdir()
            (root / "examples" / "demo.py").write_text("MODEL = '/Users/local/model'\n")

            leaks = find_public_leaks(root)

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].path.as_posix(), "examples/demo.py")

    def test_scans_github_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / ".github" / "workflows"
            workflow.mkdir(parents=True)
            (workflow / "tensorfold.yml").write_text("run: echo /Users/local/model\n")

            leaks = find_public_leaks(root)

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].path.as_posix(), ".github/workflows/tensorfold.yml")

    def test_scans_packaged_runtime_internals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "src" / "smarttensor"
            runtime.mkdir(parents=True)
            (runtime / "defaults.py").write_text("MODEL = '/Users/local/model'\n")

            leaks = find_public_leaks(root)

        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].path.as_posix(), "src/smarttensor/defaults.py")

    def test_excludes_private_research_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "HANDOVER.md").write_text("local machine notes stay private\n")
            (root / "findings.md").write_text("agent coordination notes stay private\n")
            (root / "logs").mkdir()
            (root / "logs" / "run.log").write_text("private local path /Users/local/model\n")

            leaks = find_public_leaks(root)

        self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()
