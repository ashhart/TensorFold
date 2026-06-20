from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class TensorFoldReleaseSmokeTests(unittest.TestCase):
    def test_release_smoke_script_has_cli_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/smoke_tensorfold_release.py", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Build and install TensorFold", completed.stdout)

    def test_release_smoke_script_is_in_manifest(self) -> None:
        manifest = Path("MANIFEST.in").read_text()

        self.assertIn("include tools/smoke_tensorfold_release.py", manifest)

    def test_update_script_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", "update.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
