"""tools/clef_head_gate.py compare on planted cases: an empty table or a question type without logits fails by name.

The gate imports numpy, which the CI jobs do not install, so these tests run only where numpy is.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
NUMPY = importlib.util.find_spec("numpy") is not None


def compare(types):
    """compare's exit code and output over one case whose options belong to questions of ``types`` (0, 1, 2)."""
    import clef_head_gate
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for part in ("cases", "native", "bf16", "fp64"):
            (root / part).mkdir()
        (root / "cases" / "index.json").write_text(json.dumps({"cases": ["c"]}))
        case = {"question_types": list(types), "option_question": list(range(len(types)))}
        (root / "cases" / "c.json").write_text(json.dumps(case))
        for part, logit in (("native", 1.0), ("bf16", 1.5), ("fp64", 1.0)):
            (root / part / "c.json").write_text(json.dumps({"logits": [logit] * len(types)}))
        argv = ["clef_head_gate.py", "compare", *(str(root / p) for p in ("cases", "native", "bf16", "fp64"))]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            code = clef_head_gate.main()
    return code, out.getvalue()


@unittest.skipUnless(NUMPY, "clef_head_gate.py imports numpy")
class CompareTests(unittest.TestCase):
    def test_every_type_with_logits_passes(self):
        code, out = compare([0, 1, 2])
        self.assertEqual(code, 0, out)

    def test_no_logits_fails(self):
        code, out = compare([])
        self.assertEqual(code, 1, out)
        self.assertIn("unexercised: no option logits for noul, choice, score", out)

    def test_a_type_without_logits_fails(self):
        code, out = compare([0, 1])
        self.assertEqual(code, 1, out)
        self.assertIn("unexercised: no option logits for score", out)


if __name__ == "__main__":
    unittest.main()
