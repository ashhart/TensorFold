"""H3 int8 MLP kernels at tiny shapes: integer reference, quantizer and closeness to the float MLP."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.minimax.h3.v1 import VERSION, mlp_int8


@pytest.fixture(scope="module")
def tensor_units():
    if not mlp_int8.available():
        pytest.skip("this GPU does not run the Metal 4 int8 tensor kernels")


def _random(shape, scale=1.0, seed=0):
    return mx.array((np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32))


def test_version():
    assert VERSION == "v1"


def test_weight_quantization_is_per_output_channel():
    weight = _random((6, 256), 0.05)
    q, scale = mlp_int8.quantize_weight(weight)
    w = np.array(weight)
    expected = np.abs(w).max(axis=1) / 127.0
    np.testing.assert_allclose(np.array(scale), expected, rtol=1e-6)
    np.testing.assert_array_equal(np.array(q), np.clip(np.rint(w / expected[:, None]), -127, 127).astype(np.int8))
    assert q.dtype == mx.int8 and np.abs(np.array(q)).max() == 127


@pytest.mark.parametrize("group", [512, 256])
def test_row_quantizer_matches_numpy(tensor_units, group):
    rows, k = 37, 512
    x = _random((rows, k), 1.0, seed=1).astype(mx.bfloat16)
    q, scale, kept = mlp_int8.quantize_rows(x, group)
    assert kept == rows and q.shape == (128, k) and scale.shape == (128, k // group)
    xn = np.array(x.astype(mx.float32)).reshape(rows, k // group, group)
    expected_scale = np.abs(xn).max(axis=-1) / 127.0
    np.testing.assert_allclose(np.array(scale)[:rows], expected_scale, rtol=1e-6)
    expected = np.clip(np.rint(xn / expected_scale[..., None]), -127, 127).reshape(rows, k)
    assert np.abs(np.array(q)[:rows].astype(np.int32) - expected).max() <= 1    # float32 division at a .5 tie
    assert not np.array(q)[rows:].any()


@pytest.mark.parametrize("group", [512, 256])
def test_linear_matches_the_integer_reference(tensor_units, group):
    rows, k, n = 200, 512, 384
    x, weight = _random((rows, k), 1.0, seed=2), _random((n, k), 0.05, seed=3)
    wq, ws = mlp_int8.quantize_weight(weight)
    xq, xs, kept = mlp_int8.quantize_rows(x, group)
    y = np.array(mlp_int8.int8_linear(xq, xs, wq, ws, kept, group).astype(mx.float32))
    xi, wi = np.array(xq)[:rows].astype(np.int64), np.array(wq).astype(np.int64)
    reference = np.zeros((rows, n))
    for g in range(k // group):
        cols = slice(g * group, (g + 1) * group)
        reference += (xi[:, cols] @ wi[:, cols].T) * np.array(xs)[:rows, g : g + 1]
    reference *= np.array(ws)[None]
    assert y.shape == (rows, n)
    assert np.abs(y - reference).max() <= 0.004 * np.abs(reference).max()       # bf16 output rounding
    floating = np.array(x @ weight.T)
    assert np.linalg.norm(y - floating) / np.linalg.norm(floating) < 0.02


def test_mlp_is_close_to_the_float_mlp(tensor_units):
    from mlx import nn

    rows, hidden, width = 130, 512, 768
    x = _random((2, rows // 2, hidden), 1.0, seed=4).astype(mx.bfloat16)
    fc1, fc2 = _random((2 * width, hidden), 0.05, seed=5), _random((256, width), 0.05, seed=6)
    fused = x.astype(mx.float32) @ fc1.T
    reference = np.array((nn.silu(fused[..., :width]) * fused[..., width:]) @ fc2.T)
    out = mlp_int8.Int8MLP(fc1, fc2, fc2_group=256)(x)
    assert out.shape == (2, rows // 2, 256) and out.dtype == mx.bfloat16
    result = np.array(out.astype(mx.float32))
    assert np.linalg.norm(result - reference) / np.linalg.norm(reference) < 0.05


def test_shapes_outside_the_tiles_are_refused(tensor_units):
    with pytest.raises(ValueError):
        mlp_int8.quantize_rows(mx.zeros((4, 300), dtype=mx.bfloat16), 300)
    with pytest.raises(ValueError):
        mlp_int8.Int8MLP(mx.zeros((512, 512)), mx.zeros((256, 512)))
