from __future__ import annotations

import unittest

from living_weights_v2.evaluator import evaluate
from living_weights_v2.schema import Fact, Manifest


class ReadOnlyEvaluatorTests(unittest.TestCase):
    def setUp(self):
        fact = Fact("Quillmoor Labs", "headquarters", "Sheffield",
                    "Quillmoor Labs is headquartered in Sheffield.", "approved",
                    ("Where is Quillmoor?", "Which city is Quillmoor in?"),
                    ("Where is Brackenforth?",))
        self.manifest = Manifest("model", True, (fact,), ("What is 2 plus 3?",), "source", "digest")

    def test_scores_and_keeps_anchor(self):
        class Client:
            def chat(self, model, question):
                if "Brackenforth" in question:
                    return "Unknown"
                if "plus" in question:
                    return "5"
                return "Sheffield"
        report = evaluate(self.manifest, Client())
        self.assertEqual(report["positive_matches"], 2)
        self.assertEqual(report["possible_near_miss_leaks"], 0)
        self.assertEqual(report["anchors"][0]["answer"], "5")
        self.assertFalse(report["weights_changed"])

    def test_detects_leaks(self):
        class Client:
            def chat(self, model, question):
                return "Sheffield"
        report = evaluate(self.manifest, Client())
        self.assertEqual(report["possible_near_miss_leaks"], 1)


if __name__ == "__main__":
    unittest.main()
