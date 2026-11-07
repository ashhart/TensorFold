#!/usr/bin/env python3
"""Guard complete served/reference coverage and keep observed differences visible."""
import copy
import unittest
from check_qwen35_served_reference import compare_rows, summarize, response_evidence


class ServedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ids = [1, 2]
        self.response = {"choices": [{"prompt_logprobs": [None, {"2": {"logprob": -1.2, "rank": 1}}]}],
                         "usage": {"prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}}}
        self.ref = {"bf16": {"rows": [{"position": 1, "token_id": 2, "logprob": -1.01, "rank": 1}]},
                    "fp32_rows": [{"position": 1, "token_id": 2, "logprob": -1.0, "rank": 1}]}

    def test_real_distance_violation_is_reported(self):
        result = summarize(compare_rows(self.ids, self.response, self.ref))
        self.assertEqual(result["outside_fp32_distance"], 1)
        self.assertEqual(result["outside_bf16_distance"], 1)
        self.assertAlmostEqual(result["worst_native_to_fp32"]["native_to_fp32"], 0.2)

    def test_absent_fp32_cannot_be_silently_skipped(self):
        self.ref["fp32_rows"] = []
        with self.assertRaises(ValueError):
            compare_rows(self.ids, self.response, self.ref)

    def test_wrong_token_cannot_compare(self):
        self.ref["fp32_rows"][0]["token_id"] = 9
        with self.assertRaises(ValueError):
            compare_rows(self.ids, self.response, self.ref)

    def test_nonfinite_and_cached_requests_cannot_qualify(self):
        for score in (float("nan"), float("inf")):
            response = copy.deepcopy(self.response)
            response["choices"][0]["prompt_logprobs"][1]["2"]["logprob"] = score
            with self.assertRaises(ValueError):
                compare_rows(self.ids, response, self.ref)
        self.response["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
        with self.assertRaises(ValueError):
            compare_rows(self.ids, self.response, self.ref)

    def test_row_zero_is_required_null(self):
        self.response["choices"][0]["prompt_logprobs"][0] = {}
        with self.assertRaises(ValueError):
            compare_rows(self.ids, self.response, self.ref)

    def test_public_response_excludes_prose_ids_and_decoded_tokens(self):
        self.response["id"] = "example-request"
        self.response["choices"][0]["text"] = "example generated prose"
        self.response["choices"][0]["prompt_logprobs"][1]["2"]["decoded_token"] = "example-token"
        evidence = response_evidence(self.response)
        self.assertNotIn("id", evidence)
        self.assertNotIn("text", evidence["choices"][0])
        self.assertEqual(evidence["choices"][0]["prompt_logprobs"][1]["2"], {"logprob": -1.2, "rank": 1})

    def test_single_token_has_no_scored_row(self):
        self.response["choices"][0]["prompt_logprobs"] = [None]
        self.response["usage"]["prompt_tokens"] = 1
        self.ref = {"bf16": {"rows": []}, "fp32_rows": []}
        self.assertEqual(summarize(compare_rows([1], self.response, self.ref))["rows"], 0)


if __name__ == "__main__":
    unittest.main()
