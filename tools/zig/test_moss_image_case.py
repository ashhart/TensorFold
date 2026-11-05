#!/usr/bin/env python3
"""Host-only teacher coverage tests, without torch, model loading or device work."""
import unittest
from unittest.mock import patch

import moss_image_case as case


class TeacherCoverage(unittest.TestCase):
    def test_actual_schedule_is_complete_by_default(self):
        self.assertEqual(case.teacher_steps(None, 47), 47)
        self.assertEqual(case.teacher_steps(50, 47), 47)

    def test_partial_and_absent_remain_explicit(self):
        self.assertEqual(case.teacher_steps(1, 47), 1)
        self.assertEqual(case.teacher_steps(0, 47), 0)

    def test_negative_is_not_a_partial_teacher(self):
        with self.assertRaises(ValueError):
            case.teacher_steps(-1, 47)

    def test_cli_default_records_all_teacher_steps(self):
        def check(args):
            self.assertIsNone(args.fp32_steps)
            self.assertEqual(case.teacher_steps(args.fp32_steps, 47), 47)
            return 0

        with patch.object(case, "case", check), patch("sys.argv", ["case-tool", "case", "model", "out", "--prompt-file", "prompt.txt"]):
            self.assertEqual(case.main(), 0)


if __name__ == "__main__":
    unittest.main()
