"""The Llama teacher judge passes equal rows and one-step ties, and is red on wider margins and other contenders."""
import copy
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

from llama_teacher_judge import judge, top2

try:
    import numpy
except ImportError:
    numpy = None


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

    def test_non_finite_rows_are_refused_on_either_side(self):
        for bad in (math.nan, math.inf, -math.inf):
            for side in ("native", "reference"):
                native, reference = rows(), rows()
                row = {"ids": [1, 2], "logits": [bad, 38.0]}
                (native if side == "native" else reference)[4] = row
                with self.assertRaises(ValueError, msg=f"{bad} on the {side} side"):
                    judge(native, reference)

    def test_non_finite_with_the_same_winner_is_refused(self):
        # the winning id unchanged on both sides must not hide the score
        native, reference = rows(), rows()
        native[2] = {"ids": [1, 2], "logits": [math.nan, 0.0]}
        reference[2] = {"ids": [1, 2], "logits": [math.nan, 0.0]}
        with self.assertRaises(ValueError):
            judge(native, reference)
        native[2] = {"ids": [1, 2], "logits": [38.25, -math.inf]}
        with self.assertRaises(ValueError):
            judge(native, rows())

    @unittest.skipUnless(numpy, "numpy")
    def test_top2_refuses_a_non_finite_row_anywhere(self):
        for bad in (numpy.nan, numpy.inf, -numpy.inf):
            row = numpy.array([1.0, 3.0, 2.0, 0.5], dtype=numpy.float32)
            row[3] = bad  # not the winner: the winning id would be unchanged
            with self.assertRaises(ValueError):
                top2(row)

    @unittest.skipUnless(numpy, "numpy")
    def test_the_checker_exits_1_on_a_non_finite_row(self):
        def bf16(values):
            return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16) for v in values)
        good = [1.0, 3.0, 2.0, 0.5] * 2
        here = os.path.dirname(os.path.abspath(__file__))
        for side in ("native", "reference"):
            for bad in (math.nan, math.inf, -math.inf):
                with tempfile.TemporaryDirectory() as d:
                    planted = list(good)
                    planted[4 + 3] = bad  # row 1, not its winner
                    files = {"native": good, "reference": good}
                    files[side] = planted
                    for name, values in files.items():
                        with open(os.path.join(d, name + ".bin"), "wb") as f:
                            f.write(bf16(values))
                    run = subprocess.run([sys.executable, os.path.join(here, "llama_teacher_judge.py"), "--native", os.path.join(d, "native.bin"),
                                          "--reference", os.path.join(d, "reference.bin"), "--vocab", "4", "--output", os.path.join(d, "out.json")],
                                         capture_output=True, text=True, cwd=here)
                    self.assertEqual(run.returncode, 1, f"{bad} on the {side} side: {run.stdout}{run.stderr}")
                    self.assertIn("refused", run.stdout)
        with tempfile.TemporaryDirectory() as d:  # the control: the same files, all finite, pass
            for name in ("native", "reference"):
                with open(os.path.join(d, name + ".bin"), "wb") as f:
                    f.write(bf16(good))
            run = subprocess.run([sys.executable, os.path.join(here, "llama_teacher_judge.py"), "--native", os.path.join(d, "native.bin"),
                                  "--reference", os.path.join(d, "reference.bin"), "--vocab", "4", "--output", os.path.join(d, "out.json")],
                                 capture_output=True, text=True, cwd=here)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            judge(rows(7), rows(8))


if __name__ == "__main__":
    unittest.main()
