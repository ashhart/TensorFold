"""Synthetic checks of the explicitly assumed K3.1 index formula; these are not a model-reference receipt."""
import unittest

import numpy as np

import k3_reference as ref


class IndexReference(unittest.TestCase):
    def test_key_affine_and_no_rope(self):
        projected = np.tile(np.arange(128, dtype=np.float64) - 63.5, (2, 1))
        weight = np.ones(128)
        bias = np.full(128, 0.25)
        key = ref.index_key(projected, weight, bias)
        np.testing.assert_array_equal(key[0], key[1])
        np.testing.assert_array_equal(ref.index_key(np.zeros((1, 128)), weight, bias), bias[None])

    def test_scales_and_head_weights(self):
        q = np.zeros((1, 32, 128))
        q[0, 0, 0] = 8
        q[0, 1, 0] = -4
        keys = np.zeros((3, 128))
        keys[:, 0] = [1, -1, 0]
        weights = np.zeros((1, 32))
        weights[0, :2] = [2, 3]
        np.testing.assert_allclose(ref.index_scores(q, keys, weights), [[0.25, 0.1875, 0]], rtol=1e-15)

    def test_all_visible_boundary_and_ties(self):
        s = np.zeros(2051)
        for n in [1, 2047, 2048]:
            np.testing.assert_array_equal(ref.index_select(s, n), np.arange(n))
        np.testing.assert_array_equal(ref.index_select(s, 2049), np.arange(2048))
        s[2048] = 1
        want = np.concatenate((np.arange(2047), [2048]))
        np.testing.assert_array_equal(ref.index_select(s, 2049), want)
        s[2050] = np.inf
        np.testing.assert_array_equal(ref.index_select(s, 2049), want)

    def test_signed_zero_nan_and_infinities(self):
        s = np.array([np.nan, -np.inf, -0.0, 0.0, np.inf, 1.0, np.nan])
        np.testing.assert_array_equal(ref.index_select(s, len(s), 5), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ref.index_select(s, len(s), 3), [2, 4, 5])


if __name__ == "__main__":
    unittest.main()
