"""Verify vision oracle rounding and reject malformed patch grids before evaluating weights."""
import unittest
from types import SimpleNamespace
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from vision_fp64_reference import VisionOracle


@unittest.skipIf(np is None, 'NumPy is required for the FP64 vision reference')
class VisionOracleTests(unittest.TestCase):
    def test_dense_rounds_the_fp64_sum_directly_to_bf16(self):
        oracle = VisionOracle.__new__(VisionOracle)
        oracle.loader = SimpleNamespace(map={})
        oracle.weight = lambda name: np.ones((1, 3), dtype=np.float64)
        actual = oracle.dense(np.array([[1, 2 ** -8, 2 ** -30]], dtype=np.float64), 'projection')
        self.assertEqual(actual[0, 0], 1 + 2 ** -7)
        self.assertEqual(float(np.float32(1 + 2 ** -8 + 2 ** -30)), 1 + 2 ** -8)

    def test_grid_and_patch_layout_are_validated_before_weight_access(self):
        oracle = VisionOracle.__new__(VisionOracle)
        for height, width, patches in ((0, 2, np.zeros((0, 1536))),
                                       (3, 2, np.zeros((6, 1536))),
                                       (2, 2, np.zeros((4, 768))),
                                       (2, 2, np.full((4, 1536), np.nan))):
            with self.assertRaises(ValueError):
                oracle.run(patches, height, width)


if __name__ == '__main__':
    unittest.main()
