"""The Llama teacher judge passes equal rows and one-step ties, and is red on wider margins and other contenders."""
import copy
import unittest

from llama_teacher_judge import judge


def rows(n=8):
    return [{"ids": [1, 2], "logits": [38.25, 38.0]} for _ in range(n)]


class LlamaTeacherJudgeTest(unittest.TestCase):
    def test_equal(self):
        result = judge(rows(), rows())
        self.assertEqual(result["strict_equal"], 8)
        self.assertTrue(result["policy_pass"])

    def test_one_step_tie(self):
        native = rows()
        native[3] = {"ids": [2, 1], "logits": [38.0, 38.0]}
        result = judge(native, rows())
        self.assertTrue(result["policy_pass"])
        self.assertEqual([item["position"] for item in result["exceptions"]], [3])

    def test_wider_margin_is_red(self):
        native = rows()
        native[0] = {"ids": [2, 1], "logits": [38.5, 38.0]}
        self.assertFalse(judge(native, rows())["policy_pass"])

    def test_third_contender_is_red(self):
        # three tokens within a step (#1415's position 1370): the two sides' contenders differ
        native = rows()
        native[5] = {"ids": [0, 1], "logits": [38.0, 38.0]}
        reference = copy.deepcopy(rows())
        reference[5] = {"ids": [1, 2], "logits": [38.25, 38.25]}
        self.assertFalse(judge(native, reference)["policy_pass"])

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            judge(rows(7), rows(8))


if __name__ == "__main__":
    unittest.main()
