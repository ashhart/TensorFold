"""The lean checker's exemptions: the vendored prefix is skipped, a file beside it is still checked."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_check  # noqa: E402

BAD = "// one\n// two\n"  # two full-line comments in a row: one finding


class Vendored(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved, lean_check.ROOT = lean_check.ROOT, self.root

    def tearDown(self):
        lean_check.ROOT = self.saved
        self.tmp.cleanup()

    def write(self, rel: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(BAD)
        return path

    def test_vendored_file_is_skipped(self):
        self.assertEqual(lean_check.problems(self.write(lean_check.VENDORED + "cpp/fsm.h")), [])

    def test_file_beside_the_vendored_prefix_is_checked(self):
        for rel in ("zig/vendor/xgrammar_glue.h", "zig/vendor/other/fsm.h", "zig/src/core/grammar/xgr_c.h"):
            self.assertEqual(len(lean_check.problems(self.write(rel))), 1, rel)


if __name__ == "__main__":
    unittest.main()
