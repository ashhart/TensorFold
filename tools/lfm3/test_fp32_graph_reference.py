"""CPU graph precision modes, affine dequantization and BF16 midpoint boundaries."""
import unittest
import numpy as np
from fp32_graph_reference import Graph


class PrecisionTests(unittest.TestCase):
    def graph(self, mode):
        graph = Graph.__new__(Graph)
        graph.boundaries, graph.trace = mode, None
        return graph

    def test_boundary_dtype_finite_and_mode_difference(self):
        value = np.asarray([1.001, -1.001, 0, -0.0], dtype=np.float32)
        pure, rounded = (self.graph(mode).cast(value) for mode in ('fp32', 'fp32-bf16'))
        self.assertEqual(pure.dtype, np.float32)
        self.assertEqual(rounded.dtype, np.float32)
        self.assertTrue(np.isfinite(pure).all() and np.isfinite(rounded).all())
        np.testing.assert_array_equal(pure, value)
        np.testing.assert_array_equal(rounded[:2], [1, -1])
        self.assertEqual(rounded.view(np.uint32)[3], 0x80000000)

    def test_bf16_rne_midpoints(self):
        low = np.float32(1 + 2 ** -8)
        high = np.float32(1 + 3 * 2 ** -8)
        values = np.asarray([low, np.nextafter(low, np.float32(np.inf)), high, -low, -high])
        want = np.asarray([1, 1 + 2 ** -7, 1 + 2 ** -6, -1, -1 - 2 ** -6], dtype=np.float32)
        np.testing.assert_array_equal(self.graph('fp32-bf16').cast(values), want)

    def test_fp32_product_sum_is_not_fp64_reduction(self):
        x = np.asarray([[2 ** 24, 1]], dtype=np.float32)
        y = x @ np.ones((2, 1), dtype=np.float32)
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(float(y[0, 0]), 2 ** 24)
        self.assertEqual(float(x.astype(np.float64).sum()), 2 ** 24 + 1)

    def test_dequantization_uses_fp32_without_bf16_weight_rounding(self):
        class Loader:
            def quant(self, _):
                return 8, 4

            def tensor(self, name, _):
                if name.endswith('.weight'):
                    return np.asarray([[0xffffffff]], dtype=np.uint32)
                return np.asarray([[1.0078125 if name.endswith('.scales') else -128]], dtype=np.float64)
        graph = self.graph('fp32-bf16')
        graph.checkpoint = Loader()
        weights = graph.weights('x', slice(None))
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_array_equal(weights, np.full((1, 4), np.float32(255 * 1.0078125 - 128)))
        self.assertFalse(np.array_equal(weights, graph.cast(weights)))

    def test_attention_mask_and_dtype(self):
        graph = self.graph('fp32')
        q = np.zeros((1, 3, 2), dtype=np.float32)
        v = np.asarray([[[1, 2], [3, 4], [99, 100]]], dtype=np.float32)
        result = graph.attention_value(q, q, v, window=1)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, v)
        np.testing.assert_array_equal(graph.attention_value(q, q, v)[0, 0], v[0, 0])

    def test_swiglu_and_rms_stay_fp32(self):
        graph = self.graph('fp32-bf16')
        values = np.asarray([[1, -2, 3, -4]], dtype=np.float32)
        for result in (graph.swiglu(values, values), graph.rms(values, np.ones(4, dtype=np.float32), 1e-5)):
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
