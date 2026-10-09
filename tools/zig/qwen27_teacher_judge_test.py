"""Teacher receipt admission rejects missing history contracts and out-of-policy differences."""
import copy
import unittest

from qwen27_teacher_judge import judge


def fixture():
    margins = [{"ids": [1, 2], "logits": [16.0, 15.875]} for _ in range(256)]
    native = {"teacher_forced": True, "drafts": False, "prompt_state_byte_equal": True,
              "tokens": [1] * 256, "top2": copy.deepcopy(margins)}
    return native, [1] * 256, margins


class TeacherJudgeTest(unittest.TestCase):
    def test_strict(self):
        result = judge(*fixture())
        self.assertEqual(result["strict_equal"], 256)
        self.assertTrue(result["policy_pass"])

    def test_one_step_tie(self):
        native, reference, margins = fixture()
        native["tokens"][42] = 2
        native["top2"][42]["ids"] = [2, 1]
        result = judge(native, reference, margins)
        self.assertEqual(result["strict_equal"], 255)
        self.assertTrue(result["policy_pass"])
        self.assertEqual(result["exceptions"][0]["position"], 42)

    def test_non_tie_is_red(self):
        native, reference, margins = fixture()
        native["tokens"][0] = 2
        native["top2"][0] = {"ids": [2, 1], "logits": [16.0, 15.75]}
        self.assertFalse(judge(native, reference, margins)["policy_pass"])

    def test_different_contenders_are_red(self):
        native, reference, margins = fixture()
        native["tokens"][0] = 3
        native["top2"][0]["ids"] = [3, 1]
        self.assertFalse(judge(native, reference, margins)["policy_pass"])

    def test_contracts_required(self):
        for field in ("teacher_forced", "prompt_state_byte_equal"):
            native, reference, margins = fixture()
            native[field] = False
            with self.assertRaises(ValueError):
                judge(native, reference, margins)

    def test_short_receipt_refused(self):
        native, reference, margins = fixture()
        native["top2"].pop()
        with self.assertRaises(ValueError):
            judge(native, reference, margins)

    def test_argmax_mismatch_refused(self):
        native, reference, margins = fixture()
        native["top2"][0]["ids"] = [2, 1]
        with self.assertRaises(ValueError):
            judge(native, reference, margins)

    def test_malformed_equal_scores_refused(self):
        for scores in ([float("nan"), 15.0], [15.0, 16.0]):
            native, reference, margins = fixture()
            native["top2"][0]["logits"] = scores
            with self.assertRaises(ValueError):
                judge(native, reference, margins)


if __name__ == "__main__":
    unittest.main()
