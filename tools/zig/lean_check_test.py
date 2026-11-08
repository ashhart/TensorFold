"""A direct getenv of a TF_ or TENSORFOLD_ name in Zig source is a lean finding; tests and lenient switches are not."""
from pathlib import Path
import unittest
from unittest.mock import patch

import lean_check

HINT = "(read it through platform/environment.zig)"


class Text:
    """A file as lean_check.problems reads it, without writing into the checkout."""
    def __init__(self, name, text):
        self.path, self.text = Path(name), text
        self.name, self.suffix = self.path.name, self.path.suffix

    def relative_to(self, _):
        return self.path

    def read_text(self, **_):
        return self.text


def reads(name, text):
    return [f for f in lean_check.problems(Text(name, text)) if "getenv" in f]


class DirectGetenv(unittest.TestCase):
    def test_each_source_read_is_one_finding_naming_the_parser(self):
        text = ('const a = std.c.getenv("TF_NEW_KNOB");\nconst b = platform.c.getenv("TENSORFOLD_NEW") orelse "";\n'
                'const c = @import("sys").getenv("TF_NEW_KNOB") != null and sys.getenv("TF_OTHER") == null;\n')
        source = Path("zig/src/a.zig")
        self.assertEqual(reads(source.as_posix(), text), [
            f"{source}:1: direct getenv of TF_NEW_KNOB {HINT}",
            f"{source}:2: direct getenv of TENSORFOLD_NEW {HINT}",
            f"{source}:3: direct getenv of TF_NEW_KNOB {HINT}",
            f"{source}:3: direct getenv of TF_OTHER {HINT}",
        ])

    def test_tests_parsers_comments_and_other_names_are_not_findings(self):
        read = 'const a = std.c.getenv("TF_NEW_KNOB");\n'
        for name in ("zig/src/a_test.zig", "zig/tests/cuda/a.zig", "tools/zig/test_a.zig"):
            with self.subTest(test_file=name):
                self.assertEqual(reads(name, read), [])
        others = (
            ("zig/src/platform/environment.zig", "const raw = sys.getenv(name);\n"),
            ("zig/src/core/profile.zig", "const raw = std.c.getenv(entry(id).name());\n"),
            ("zig/src/a.zig", 'const a = std.c.getenv("NCCL_DEBUG") orelse std.c.getenv("XPU_A8_BITS");\n'),
            ("zig/src/a.zig", '// std.c.getenv("TF_NEW_KNOB") is read once\n'),
            ("tools/zig/a.py", 'x = os.getenv("TF_NEW_KNOB")\n'),
        )
        for name, text in others:
            with self.subTest(text=text):
                self.assertEqual(reads(name, text), [])

    def test_a_listed_lenient_switch_is_not_a_finding(self):
        with patch.object(lean_check, "lenient", return_value=frozenset({"TF_OLD"})):
            self.assertEqual(reads("zig/src/a.zig", 'const a = std.c.getenv("TF_OLD");\n'), [])
            self.assertEqual(len(reads("zig/src/a.zig", 'const a = std.c.getenv("TF_OLDER");\n')), 1)

    def test_the_lenient_list_is_the_bullets_of_its_section(self):
        text = ("# Settings\n\n- `TF_BEFORE`\n\n## Existing lenient switches\n\nThese keep `TF_INTRO`.\n\n"
                "- `TF_A`: exact `0` disables.\n- Pages: exact `TENSORFOLD_B=0` and\n  `TF_C` wrap.\n\n"
                "The registry reports `TF_AFTER`.\n\n## Next\n\n- `TF_LATER`\n")
        self.assertEqual(lean_check.lenient_switches(text), {"TF_A", "TENSORFOLD_B", "TF_C"})
        self.assertEqual(lean_check.lenient_switches("# Settings\n\n- `TF_A`\n"), set())

    def test_the_checkout_doc_lists_its_lenient_switches(self):
        self.assertTrue({"TF_GRAPH_CACHE", "TENSORFOLD_KOLIBRI_KV_PAGES", "TF_COOP_SPLIT"} <= lean_check.lenient())
        self.assertNotIn("TENSORFOLD_NO_UPDATE_CHECK", lean_check.lenient())


if __name__ == "__main__":
    unittest.main()
