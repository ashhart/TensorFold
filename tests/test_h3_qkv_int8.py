"""H3 fused int8 QKV kernel at tiny shapes: equal to the unfused int8 projection followed by norm and rotary."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.minimax.h3.v1 import mlp_int8, qkv_int8

HEADS, DIM, HIDDEN, ROTARY = 2, 128, 256, 96


@pytest.fixture(scope="module")
def tensor_units():
    if not mlp_int8.available():
        pytest.skip("this GPU does not run the Metal 4 int8 tensor kernels")


def _random(shape, scale=1.0, seed=0):
    return mx.array((np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32))


def _reference(x, weight, q_weight, k_weight, cos, sin, eps):
    """The unfused path in numpy from the same int8 projection output."""

    y = np.array(mlp_int8.Int8Linear(weight, group=HIDDEN)(x).astype(mx.float32))
    rows = y.shape[1]
    y = y.reshape(rows, HEADS, 3, DIM)
    cos, sin = np.array(cos), np.array(sin)
    half = ROTARY // 2
    out = []
    for kind, norm in ((0, np.array(q_weight)), (1, np.array(k_weight))):
        part = y[:, :, kind]
        part = part / np.sqrt((part**2).mean(axis=-1, keepdims=True) + eps) * norm
        turned = part[..., :ROTARY]
        swapped = np.concatenate([-turned[..., half:], turned[..., :half]], axis=-1)
        part = np.concatenate([turned * cos[:, None] + swapped * sin[:, None], part[..., ROTARY:]], axis=-1)
        out.append(part.transpose(1, 0, 2))
    out.append(y[:, :, 2].transpose(1, 0, 2))
    return out


@pytest.mark.parametrize("rows", [37, 128, 200])
def test_fused_equals_projection_then_norm_and_rotary(tensor_units, rows):
    weight = _random((3 * HEADS * DIM, HIDDEN), 0.05, seed=1)
    q_weight, k_weight = 1.0 + _random((DIM,), 0.1, seed=2), 1.0 + _random((DIM,), 0.1, seed=3)
    x = _random((1, rows, HIDDEN), 1.0, seed=4).astype(mx.bfloat16)
    angles = _random((rows, ROTARY), 3.0, seed=5)
    cos, sin = mx.cos(angles), mx.sin(angles)
    fused = qkv_int8.Int8QKV(weight, q_weight, k_weight, HEADS, DIM, eps=1e-5)
    got = fused(x, cos, sin)
    want = _reference(x, weight, q_weight, k_weight, cos, sin, 1e-5)
    for name, a, b in zip("qkv", got, want, strict=True):
        assert a.shape == (1, HEADS, rows, DIM) and a.dtype == mx.bfloat16, name
        a = np.array(a.astype(mx.float32))[0]
        # the kernel normalises the bfloat16 projection in float32 and rounds once; allow bfloat16 rounding
        assert np.abs(a - b).max() <= 0.02 * max(1.0, np.abs(b).max()), name
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 6e-3, name
    np.testing.assert_array_equal(np.array(got[2].astype(mx.float32))[0], want[2])


def test_rejects_shapes_it_cannot_tile(tensor_units):
    weight = _random((3 * HEADS * DIM, HIDDEN), 0.05)
    ones = mx.ones((DIM,))
    with pytest.raises(ValueError):
        qkv_int8.Int8QKV(weight, ones, ones, HEADS + 1, DIM)
    fused = qkv_int8.Int8QKV(weight, ones, ones, HEADS, DIM)
    with pytest.raises(ValueError):
        fused(mx.zeros((2, 8, HIDDEN), dtype=mx.bfloat16), mx.ones((8, ROTARY)), mx.zeros((8, ROTARY)))
    with pytest.raises(ValueError):
        fused(mx.zeros((1, 8, HIDDEN), dtype=mx.bfloat16), mx.ones((9, ROTARY)), mx.zeros((9, ROTARY)))
