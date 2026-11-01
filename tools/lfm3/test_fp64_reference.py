"""CPU reference checks for direct BF16 rounding and packed affine reconstruction."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from fp64_reference import Oracle, bf, inject_image_rows


def bits(values):
    return (np.asarray(bf(values), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


@unittest.skipIf(np is None, 'NumPy is required for the independent FP64 reference')
class ReferenceTests(unittest.TestCase):
    def test_all_finite_values_and_adjacent_rounding_boundaries(self):
        raw = np.arange(0x7f80, dtype=np.uint16)
        values = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        np.testing.assert_array_equal(bits(values), raw)
        np.testing.assert_array_equal(bits(-values), raw | 0x8000)
        middle = (values[:-1] + values[1:]) / 2
        even = np.where((raw[:-1] & 1) == 0, raw[:-1], raw[1:])
        np.testing.assert_array_equal(bits(middle), even)
        np.testing.assert_array_equal(bits(-middle), even | 0x8000)
        np.testing.assert_array_equal(bits(np.nextafter(middle, np.inf)), raw[1:])
        np.testing.assert_array_equal(bits(np.nextafter(middle, -np.inf)), raw[:-1])

    def test_fp32_intermediate_would_double_round_the_fp64_result(self):
        value = 1 + 2 ** -8 + 2 ** -30
        self.assertEqual(int(bits(value)), 0x3f81)
        self.assertEqual(int(bits(np.float32(value))), 0x3f80)
        self.assertEqual(int(bits(1 + 2 ** -8 - 2 ** -30)), 0x3f80)

    def test_subnormal_overflow_and_nonfinite_values(self):
        self.assertEqual(int(bits(2 ** -134)), 0)
        self.assertEqual(int(bits(np.nextafter(np.float64(2 ** -134), np.inf))), 1)
        maximum = (np.uint32(0x7f7f) << np.uint32(16)).view(np.float32).astype(np.float64)
        threshold = maximum + 2 ** 119
        self.assertEqual(int(bits(np.nextafter(threshold, -np.inf))), 0x7f7f)
        self.assertEqual(int(bits(threshold)), 0x7f80)
        self.assertEqual(int(bits(float('inf'))), 0x7f80)
        self.assertEqual(int(bits(-float('inf'))), 0xff80)
        self.assertEqual(int(bits(float('nan'))), 0x7fc0)
        self.assertEqual(int(bits(-0.0)), 0x8000)

    def test_six_bit_codes_cross_word_boundaries_before_weight_rounding(self):
        oracle = Oracle.__new__(Oracle)
        oracle.config = {'quantization': {'bits': 6, 'group_size': 32}}
        oracle.weight_mode = 'exact'
        tensors = {
            'projection.weight': np.array([[0x440c2040, 0xa2481c61, 0x3ce34c2c,
                                            0x544d2450, 0xa6585d65, 0x7de75c6d]], dtype=np.uint32),
            'projection.scales': np.array([[1 / 256]], dtype=np.float64),
            'projection.biases': np.array([[8]], dtype=np.float64),
        }
        oracle.tensor = lambda name, rows=None: tensors[name] if rows is None else tensors[name][rows]
        expected = 8 + np.arange(32, dtype=np.float64)[None] / 256
        np.testing.assert_array_equal(oracle.weights('projection', slice(0, 1)), expected)
        oracle.weight_mode = 'bf16'
        rounded = oracle.weights('projection', slice(0, 1))
        self.assertEqual(rounded[0, 1], 8)
        self.assertNotEqual(rounded[0, 1], expected[0, 1])

    def test_projected_image_rows_replace_only_matching_placeholders(self):
        hidden = np.arange(12, dtype=np.float64).reshape(3, 4)
        rows = np.array([[1 + 2 ** -8 + 2 ** -30, 2, 3, 4]])
        result = inject_image_rows(hidden, [10, 125030, 11], rows, 1)
        np.testing.assert_array_equal(result[[0, 2]], hidden[[0, 2]])
        np.testing.assert_array_equal(hidden, np.arange(12).reshape(3, 4))
        self.assertEqual(result[1, 0], float(bf(rows[0, 0])))
        for tokens, value, start in (([10, 125030, 11], rows, 0),
                                      ([10, 125030, 125030], rows, 1),
                                      ([10, 125030, 11], rows[:, :3], 1),
                                      ([10, 125030, 11], rows * np.nan, 1)):
            with self.assertRaises(ValueError):
                inject_image_rows(hidden, tokens, value, start)

    def test_embedding_override_does_not_change_other_projections(self):
        oracle = Oracle.__new__(Oracle)
        oracle.config = {'quantization': {'bits': 4, 'group_size': 32,
                         'language_model.model.embed_tokens': {'bits': 8, 'group_size': 32}}}
        self.assertEqual(oracle.quant('language_model.model.embed_tokens'), (8, 32))
        self.assertEqual(oracle.quant('language_model.model.layers.0.conv.in_proj'), (4, 32))


if __name__ == '__main__':
    unittest.main()
