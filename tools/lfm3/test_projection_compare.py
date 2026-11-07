"""Regression checks for worst-column reporting in projection qualification."""
import tempfile
import unittest
from pathlib import Path
try:
    import numpy as np
    from projection_compare import metrics, read_rows
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy is required for FP64 projection qualification')
class ProjectionComparison(unittest.TestCase):
    def test_worst_column_is_not_hidden_by_aggregate(self):
        expected = np.zeros((64, 128), dtype=np.float64)
        actual = expected.copy()
        actual[:, 93] = 1
        result, columns = metrics(actual, expected)
        self.assertEqual(result['worst_column'], 93)
        self.assertEqual(result['worst_column_rms'], 1)
        self.assertEqual(columns[93], 1)
        self.assertLess(result['rms'], result['worst_column_rms'])
        self.assertEqual(result['correctly_rounded'], 64 * 127)

    def test_nonfinite_outputs_cannot_produce_a_qualification_metric(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                metrics(np.array([[value]]), np.zeros((1, 1)))

    def test_bf16_rows_preserve_exact_inputs_and_reject_partial_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.bf16'
            np.array([0x3f80, 0xbf80, 0x0000, 0x3f00], dtype='<u2').tofile(path)
            np.testing.assert_array_equal(read_rows(path, 2), [[1, -1], [0, .5]])
            with self.assertRaises(ValueError):
                read_rows(path, 3)
            path.write_bytes(path.read_bytes() + b'\x00')
            with self.assertRaises(ValueError):
                read_rows(path, 2)


if __name__ == '__main__':
    unittest.main()
