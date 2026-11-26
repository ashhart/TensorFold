"""Check vision replay indexing and intermediate FP32 rounding independently."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from elementary_reference import fma, swiglu
    from vision_rounding_reference import rms_rows, convolution, rotary, attention, layer_norm, gelu



@unittest.skipIf(np is None, 'NumPy is required for rounding reference tests')
class VisionRoundingTests(unittest.TestCase):
    def test_fma_retains_product_residual_before_round(self):
        a, b = np.float32(1 + 2 ** -23), np.float32(1 - 2 ** -23)
        self.assertEqual(float(fma(a, b, -1)), -2 ** -46)
        self.assertEqual(float(np.float32(a * b) - np.float32(1)), 0)

    def test_vision_rms_keeps_strided_row_channels_and_gain(self):
        values = np.ones((2, 1024))
        gain = np.ones(1024)
        gain[513] = -2
        actual = rms_rows(values, gain, 1e-5)
        np.testing.assert_array_equal(actual, np.tile(gain, (2, 1)))
        with self.assertRaises(ValueError):
            rms_rows(np.ones((1, 128)), np.ones(128), 1e-5)

    def test_spatial_convolution_zero_padding_and_channel_axis(self):
        grid = np.arange(24).reshape(2, 3, 4)
        kernel = np.zeros((4, 3, 3))
        kernel[:, 1, 1] = np.arange(1, 5)
        np.testing.assert_array_equal(convolution(grid, kernel), grid * np.arange(1, 5))
        kernel[:] = 0
        kernel[:, 0, 1] = 1
        expected = np.zeros_like(grid)
        expected[1] = grid[0]
        np.testing.assert_array_equal(convolution(grid, kernel), expected)

    def test_zero_position_rotary_is_identity(self):
        values = np.arange(128, dtype=np.float32).reshape(2, 1, 64)
        np.testing.assert_array_equal(rotary(values, np.zeros((2, 2))), values)

    def test_attention_constant_visible_values_are_preserved(self):
        q = np.zeros((2, 5, 64), dtype=np.float32)
        v = np.broadcast_to(np.arange(64, dtype=np.float32), q.shape)
        np.testing.assert_array_equal(attention(q, q, v), v)
        with self.assertRaises(ValueError):
            attention(q[..., :32], q, v)

    def test_projector_constant_row_and_gelu_origin(self):
        np.testing.assert_array_equal(layer_norm(np.ones((2, 1024)), np.ones(1024),
                                                np.zeros(1024), 1e-6), np.zeros((2, 1024)))
        np.testing.assert_array_equal(gelu(np.array([0], dtype=np.float32)), [0])

    def test_swiglu_rounds_silu_before_up_product(self):
        from fp64_reference import bf
        gate = np.array([0, 1, -1], dtype=np.float32)
        up = np.array([3, 2, 2], dtype=np.float32)
        silu = bf(gate / (np.float32(1) + np.exp(-gate)))
        np.testing.assert_array_equal(swiglu(gate, up), bf(silu * up))


if __name__ == '__main__':
    unittest.main()
