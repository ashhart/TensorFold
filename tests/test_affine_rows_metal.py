"""Metal validation of packed affine values and identical rows across verifier widths."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from tensorfold.kernels.qwen.dense.v1 import affine_rows

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason='requires Metal')


def _pack(values, bits):
    n, k = values.shape
    words = np.zeros((n, k * bits // 32), dtype=np.uint32)
    for i in range(k):
        at, shift = divmod(i * bits, 32)
        words[:, at] |= values[:, i].astype(np.uint32) << np.uint32(shift)
        if shift + bits > 32:
            words[:, at + 1] |= values[:, i].astype(np.uint32) >> np.uint32(32 - shift)
    return mx.array(words)


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize('group', [32, 64, 128])
def test_all_affine_formats_keep_values_and_rows(bits, group):
    rng = np.random.default_rng(100 + bits + group)
    n, k = 5, 256
    codes = rng.integers(0, 2**bits, (n, k), dtype=np.uint32)
    scale = rng.integers(1, 5, (n, k // group)).astype(np.float32) / 8
    bias = rng.integers(-2, 3, (n, k // group)).astype(np.float32) / 2
    x = rng.integers(-3, 4, (128, k)).astype(np.float32)
    packed = _pack(codes, bits)
    s, b = mx.array(scale).astype(mx.bfloat16), mx.array(bias).astype(mx.bfloat16)
    target = codes.astype(np.float64) * np.repeat(scale, group, axis=1) + np.repeat(bias, group, axis=1)
    expected = mx.array((x.astype(np.float64) @ target.T).astype(np.float32)).astype(mx.bfloat16)
    inputs = mx.array(x).astype(mx.bfloat16)
    singles = mx.concatenate([affine_rows.qmm(inputs[i:i+1], packed, s, b, group, bits) for i in range(128)])
    mx.eval(expected, singles)
    assert bool(mx.array_equal(singles, expected).item())
    for rows in (1, 2, 7, 8, 15, 16, 17, 31, 32, 64, 65, 128):
        actual = affine_rows.qmm(inputs[:rows], packed, s, b, group, bits)
        mx.eval(actual)
        assert bool(mx.array_equal(actual, singles[:rows]).item()), (bits, group, rows)


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize('group', [32, 64, 128])
def test_tiny_weights_do_not_change_metal_address_space(bits, group):
    codes = np.arange(group, dtype=np.uint32).reshape(1, -1) % (2**bits)
    x = mx.ones((1, group), dtype=mx.bfloat16)
    scale = mx.array([[.25]], dtype=mx.bfloat16)
    bias = mx.array([[-.5]], dtype=mx.bfloat16)
    actual = affine_rows.qmm(x, _pack(codes, bits), scale, bias, group, bits)
    expected = mx.array([[float(codes.sum()) * .25 - .5 * group]], dtype=mx.bfloat16)
    assert bool(mx.array_equal(actual, expected).item())


@pytest.mark.parametrize('bits', affine_rows.BITS)
@pytest.mark.parametrize('group,k', [(32, 640), (64, 1536), (128, 5120)])
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_rows_are_their_one_row_calls_at_every_launch_with_random_values(bits, group, k, dtype):
    rng = np.random.default_rng(bits * 1000 + group + k)
    n = 37                                                   # not a whole number of threadgroup outputs
    codes = rng.integers(0, 2**bits, (n, k), dtype=np.uint32)
    scale = mx.array(rng.uniform(0.001, 0.05, (n, k // group)).astype(np.float32)).astype(getattr(mx, dtype))
    bias = mx.array(rng.uniform(-0.1, 0.1, (n, k // group)).astype(np.float32)).astype(getattr(mx, dtype))
    x = mx.array(rng.normal(0, 1, (40, k)).astype(np.float32)).astype(mx.bfloat16)
    packed = _pack(codes, bits)
    target = (codes.astype(np.float64) * np.repeat(np.array(scale.astype(mx.float32)), group, axis=1)
              + np.repeat(np.array(bias.astype(mx.float32)), group, axis=1))
    want = np.array(x.astype(mx.float32)).astype(np.float64) @ target.T
    singles = mx.concatenate([affine_rows.qmm(x[i:i + 1], packed, scale, bias, group, bits) for i in range(40)])
    np.testing.assert_allclose(np.array(singles.astype(mx.float32)), want, rtol=2e-2, atol=2e-2)
    for rows in (2, 3, 4, 5, 8, 9, 15, 16, 17, 33, 40):
        assert bool(mx.array_equal(affine_rows.qmm(x[:rows], packed, scale, bias, group, bits),
                                   singles[:rows]).item()), (bits, group, rows)
    assert bool(mx.array_equal(affine_rows.qmm(x[5:21], packed, scale, bias, group, bits), singles[5:21]).item())
