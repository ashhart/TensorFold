#!/usr/bin/env python3
"""Regressions for a passing numerical bound hiding a failed fidelity envelope."""
import copy
import json
import unittest
from pathlib import Path
from qwen35_reference_verdict import evaluate_report, public_report


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.row = {"native_logprob": -1.001, "transformers_bf16_logprob": -1.01,
                    "transformers_fp32_logprob": -1.0, "bound_terms": {
                        "native_vs_fp64_box": 0.0001, "rounding_slack": 0.000001}}
        self.report = {"complete": True, "native_comparison": {"rows": [self.row],
                       "fp32_available": True, "headline": {"all_rows_within_bound": True}}}

    def test_close_row_passes(self):
        self.assertEqual(evaluate_report(self.report)["status"], "passed")

    def test_broad_bound_does_not_override_measured_envelope(self):
        self.row["native_logprob"] = -1.1
        result = evaluate_report(self.report)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["fp32_envelope_failures"], 1)

    def test_reference_direction_is_not_hidden(self):
        self.row["native_logprob"] = -0.995
        result = evaluate_report(self.report)
        self.assertEqual(result["fp32_envelope_failures"], 0)
        self.assertEqual(result["bf16_envelope_failures"], 1)

    def test_missing_and_nonfinite_evidence_cannot_pass(self):
        for bad in (None, float("nan"), float("inf")):
            report = copy.deepcopy(self.report)
            report["native_comparison"]["rows"][0]["native_logprob"] = bad
            self.assertEqual(evaluate_report(report)["status"], "not-run")

    def test_absent_fp32_and_empty_rows_are_not_run(self):
        self.report["native_comparison"]["fp32_available"] = False
        self.assertEqual(evaluate_report(self.report)["status"], "not-run")
        self.report["native_comparison"]["rows"] = []
        self.assertEqual(evaluate_report(self.report)["status"], "not-run")

    def test_public_report_removes_runner_identifiers(self):
        raw = {"host": "example-runner", "argv": ["/home/example/reference.py"],
               "input": {"path": "/home/example/fixture.json", "sha256": "abc"},
               "model_dir": "/home/example/model", "numeric": [0.25, 0.5]}
        public = public_report(raw)
        self.assertNotIn("example", json.dumps(public))
        self.assertEqual(public["input"]["sha256"], "abc")
        self.assertEqual(public["numeric"], raw["numeric"])

    def test_incomplete_run_cannot_reuse_an_earlier_pass(self):
        self.report["complete"] = False
        self.assertEqual(evaluate_report(self.report)["status"], "not-run")

    def test_recorded_counterexamples_stay_failed(self):
        path = Path(__file__).resolve().parents[2] / "docs/receipts/qwen35-bf16-scoring/transformers-comparison.json"
        result = evaluate_report(json.loads(path.read_text()))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["fp32_envelope_failures"], 3)
        self.assertEqual(result["bf16_envelope_failures"], 1)


if __name__ == "__main__":
    unittest.main()
