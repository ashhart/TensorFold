"""env_names.py: which literals count as names, and how the doc's rows are checked and rewritten."""
import tempfile
import unittest
from pathlib import Path

import env_names

SOURCE = '''const a = getenv("TF_SEEN");
// const b = getenv("TF_IN_A_COMMENT");
const c = .{ "TF_TWICE", "lower_case", "NOUNDERSCORE", "TF_TWICE" };
test "a test block" {
    _ = getenv("TF_IN_A_TEST");
}
const d = env("TENSORFOLD_AFTER_TEST");
'''
PROFILE = "pub const Switch = enum {\n    TF_SWITCH,\n};\n"
HEAD = "# Names\n\nIntro.\n\n"


def tree(root: Path) -> None:
    src = root / "zig/src"
    (src / "core").mkdir(parents=True)
    (src / "a.zig").write_text(SOURCE, encoding="utf-8")
    (src / "a_test.zig").write_text('const t = getenv("TF_IN_A_TEST_FILE");\n', encoding="utf-8")
    (src / "core/profile.zig").write_text(PROFILE, encoding="utf-8")


class EnvNamesTest(unittest.TestCase):
    def test_source_names_skip_comments_tests_and_test_files(self):
        with tempfile.TemporaryDirectory() as d:
            tree(Path(d))
            names = env_names.source_names(Path(d))
        self.assertEqual(sorted(names), ["TENSORFOLD_AFTER_TEST", "TF_SEEN", "TF_SWITCH", "TF_TWICE"])
        self.assertEqual(names["TF_SEEN"], "zig/src/a.zig:1")

    def test_problems_name_missing_stale_and_unknown_tags(self):
        names = {"TF_A": "zig/src/a.zig:1", "TF_B": "zig/src/a.zig:2"}
        found = env_names.problems(names, {"TF_A": "logs, bogus", "TF_GONE": "other"})
        self.assertEqual(len(found), 3)
        self.assertIn("TF_B has no row", found[0])
        self.assertIn("TF_GONE is listed but no engine source spells it", found[1])
        self.assertIn("TF_A has unknown tag 'bogus'", found[2])
        self.assertEqual(env_names.problems(names, {"TF_A": "writes, loads", "TF_B": "not-env"}), [])

    def test_written_keeps_tags_adds_untagged_and_drops_gone_names(self):
        doc = HEAD + "## Variables\n\n| Name | Tags |\n| --- | --- |\n| `TF_A` | logs |\n| `TF_GONE` | other |\n"
        out = env_names.written(doc, {"TF_A": "", "TF_NEW": "", "CODE_X": ""})
        self.assertTrue(out.startswith(HEAD + "## Variables\n"))
        self.assertEqual(env_names.listed(out), {"CODE_X": "untagged", "TF_A": "logs", "TF_NEW": "untagged"})
        again = out.replace("| `CODE_X` | untagged |", "| `CODE_X` | not-env |")
        rewritten = env_names.written(again, {"TF_A": "", "TF_NEW": "", "CODE_X": ""})
        self.assertEqual(rewritten.split("## Names that are not variables")[1].count("CODE_X"), 1)
        self.assertEqual(env_names.written(rewritten, {"TF_A": "", "TF_NEW": "", "CODE_X": ""}), rewritten)


if __name__ == "__main__":
    unittest.main()
