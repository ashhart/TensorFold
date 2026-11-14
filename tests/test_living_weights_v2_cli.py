from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

from living_weights_v2.__main__ import main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.manifest = Path(self.dir.name) / "lesson.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1, "model": "local-model", "asserted_by_user": True,
            "source": "test", "anchors": ["What is 2+3?"],
            "facts": [{"subject": "Quillmoor", "relation": "office", "answer": "Sheffield",
                       "text": "Quillmoor is in Sheffield.", "evidence": "test",
                       "questions": ["Where is Quillmoor?", "Which city has Quillmoor?"],
                       "near_misses": ["Where is Brackenforth?"]}]
        }))

    def tearDown(self):
        self.dir.cleanup()

    def test_plan(self):
        output = StringIO()
        with redirect_stdout(output):
            result = main(["plan", str(self.manifest)])
        self.assertEqual(result, 0)
        parsed = json.loads(output.getvalue())
        self.assertFalse(parsed["mutates_weights"])
        self.assertEqual(parsed["facts"][0]["positive_count"], 2)

    def test_status_empty_ledger(self):
        output = StringIO()
        with redirect_stdout(output):
            result = main(["status", "--ledger", str(Path(self.dir.name) / "new.sqlite3")])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), [])

    def test_bad_manifest_exits_nonzero(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = main(["plan", str(Path(self.dir.name) / "missing.json")])
        self.assertEqual(result, 4)
        self.assertIn("living_weights_v2:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
