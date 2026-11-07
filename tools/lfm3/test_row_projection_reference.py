"""Independent packed-code fixtures exercise cross-word decoding and reject invalid replay inputs."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from row_projection_reference import affine_rows, dense_rows


@unittest.skipIf(np is None, 'NumPy is required for the projection rounding replay')
class RowReplayTests(unittest.TestCase):
    def oracle(self, bits, columns=1):
        class Fixture:
            def quant(self, name):
                return bits, 32

            def tensor(self, name):
                if name.endswith('.scales'):
                    return np.repeat(np.arange(2, 2 + columns, dtype=np.float32)[:, None], 16, axis=1)
                if name.endswith('.biases'):
                    return np.full((columns, 16), -1, dtype=np.float32)
                packed = [sum(((i + 3 * column) % (1 << bits)) << (i * bits) for i in range(512))
                          for column in range(columns)]
                return np.array([[(value >> shift) & 0xffffffff for shift in range(0, 512 * bits, 32)]
                                 for value in packed], dtype=np.uint32)
        return Fixture()

    def test_all_three_packed_widths_match_independent_integer_sum(self):
        for bits in (4, 6, 8):
            result = affine_rows(self.oracle(bits), 'projection', np.ones((2, 512), dtype=np.float32))
            expected = 2 * sum(i % (1 << bits) for i in range(512)) - 512
            np.testing.assert_array_equal(result, np.full((2, 1), expected))

    def test_noncontiguous_unpacked_columns_keep_their_identity(self):
        values = np.arange(512, dtype=np.float32) % 64 - 32
        for bits in (4, 6, 8):
            actual = affine_rows(self.oracle(bits, 3), 'projection', values[None])
            expected = [sum(float(values[i]) * ((i + 3 * c) % (1 << bits)) * (2 + c)
                            for i in range(512)) - float(values.sum()) for c in range(3)]
            np.testing.assert_array_equal(actual, [expected])

    def test_dense_fp32_rounding_occurs_before_final_storage(self):
        values = np.zeros((1, 96), dtype=np.float32)
        values[0, [0, 32, 64]] = [2 ** 24, 1, -(2 ** 24)]
        weight = np.ones((1, 96), dtype=np.float32)
        self.assertEqual((values.astype(np.float64) @ weight.astype(np.float64).T)[0, 0], 1)
        self.assertEqual(float(dense_rows(values, weight)[0, 0]), 0)

    def test_dense_columns_and_input_guards(self):
        values = (np.arange(64) % 8).astype(np.float32)[None]
        weights = np.stack([np.ones(64), np.arange(64) % 4]).astype(np.float32)[::-1]
        np.testing.assert_array_equal(dense_rows(values, weights), values @ weights.T)
        with self.assertRaises(ValueError):
            dense_rows(values[:, :-1], weights)
        with self.assertRaises(ValueError):
            dense_rows(np.full((1, 64), 1 + 2 ** -30), weights)

    def test_inputs_must_be_finite_bf16_rows_with_served_k_geometry(self):
        for value in (np.ones((1, 511)), np.full((1, 512), 1.001),
                      np.full((1, 512), 1 + 2 ** -30), np.full((1, 512), np.inf)):
            with self.assertRaises(ValueError):
                affine_rows(self.oracle(4), 'projection', value)


if __name__ == '__main__':
    unittest.main()
