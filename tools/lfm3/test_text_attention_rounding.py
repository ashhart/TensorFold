"""Check causal geometry and rotary identity for the optional FP32 replay."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from text_attention_rounding import attention, rotary



@unittest.skipIf(np is None, 'NumPy is required for rounding reference tests')
class TextAttentionRoundingTests(unittest.TestCase):
    def test_causal_uniform_attention_uses_only_prefix(self):
        q = np.zeros((2, 4, 128), dtype=np.float32)
        v = np.broadcast_to(np.arange(1, 5, dtype=np.float32)[None, :, None], q.shape)
        expected = np.broadcast_to(np.array([1, 1.5, 2, 2.5])[None, :, None], q.shape)
        np.testing.assert_array_equal(attention(q, q, v, 0, 'numpy'), expected)
        sliding = np.broadcast_to(np.array([1, 1.5, 2.5, 3.5])[None, :, None], q.shape)
        np.testing.assert_array_equal(attention(q, q, v, 2, 'numpy'), sliding)

    def test_zero_position_is_identity_with_partial_rotary(self):
        values = np.arange(128, dtype=np.float32).reshape(1, 1, 128)
        spec = {'rope_theta': 10000, 'partial_rotary_factor': .5}
        np.testing.assert_array_equal(rotary(values, spec, 'numpy'), values)

    def test_unimplemented_multi_partition_replay_is_rejected(self):
        q = np.zeros((1, 1025, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            attention(q, q, q, 0, 'numpy')


if __name__ == '__main__':
    unittest.main()
