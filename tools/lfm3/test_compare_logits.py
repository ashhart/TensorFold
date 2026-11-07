"""Ensure fidelity receipts preserve real differences, ties and phase-independent row identities."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from compare_logits import compare


@unittest.skipIf(np is None, 'NumPy is required for fidelity reports')
class CompareTests(unittest.TestCase):
    def test_disjoint_contenders_and_both_margins_are_retained(self):
        result = compare([[2, 1, 0], [0, 3, 1]], [[1, 2, 0], [0, 3, 1]], 7)
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['disjoint_maximum_sets'], 1)
        row = result['differences'][0]
        self.assertEqual((row['row'], row['native_top'], row['reference_top']), (7, 0, 1))
        self.assertEqual(row['native_margin'], 1)
        self.assertEqual(row['reference_margin'], 1)
        self.assertEqual(row['native_reference_contender_gap'], 1)
        self.assertEqual(row['reference_native_contender_gap'], 1)
        self.assertIsNone(result['fidelity_pass'])

    def test_overlapping_tie_is_reported_without_waiving_it(self):
        result = compare([[1, 2, 0]], [[2, 2, 0]])
        self.assertEqual(result['top1_differences'], 1)
        self.assertEqual(result['disjoint_maximum_sets'], 0)
        self.assertEqual(result['differences'][0]['reference_margin'], 0)
        self.assertIsNone(result['accepted_threshold'])

    def test_worst_rows_are_not_hidden_by_average(self):
        result = compare([[0, 0], [9, 1]], [[0, 0], [1, 1]])
        self.assertEqual(result['worst_rows'][0]['row'], 1)
        self.assertEqual(result['maximum_absolute_error'], 8)

    def test_invalid_geometry_and_nonfinite_inputs_are_refused(self):
        for a, b in (([[1]], [[1]]), ([[1, 2]], [[1, 2], [1, 2]]), ([[np.nan, 1]], [[1, 2]])):
            with self.assertRaises(ValueError):
                compare(a, b)


if __name__ == '__main__':
    unittest.main()
