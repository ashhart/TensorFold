"""H3 VAE acceleration: int8 kernels with a bias against float references, on tiny shapes."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.h3 import vae_fast
from tensorfold.kernels.minimax.h3.v1 import mlp_int8

pytestmark = pytest.mark.skipif(not mlp_int8.available(), reason="Metal 4 tensor operations are unavailable")


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("rows", [37, 128, 200])
def test_linear_with_a_bias_matches_float(rows):
    rng = np.random.default_rng(rows)
    weight = rng.standard_normal((256, 512)).astype(np.float32) * 0.05
    bias = rng.standard_normal(256).astype(np.float32)
    x = rng.standard_normal((rows, 512)).astype(np.float32)
    out = np.array(mlp_int8.Int8Linear(mx.array(weight), 256, mx.array(bias))(mx.array(x)).astype(mx.float32))
    assert out.shape == (rows, 256)
    assert _relative(out, x @ weight.T + bias) < 0.03
    plain = np.array(mlp_int8.Int8Linear(mx.array(weight), 256)(mx.array(x)).astype(mx.float32))
    assert np.allclose(out - plain, bias, atol=0.05)  # the bias is the only difference from the bias-free kernel


def test_swiglu_adds_the_bias_before_the_gate():
    rng = np.random.default_rng(7)
    hidden, width, rows = 256, 256, 70
    w1 = rng.standard_normal((2 * width, hidden)).astype(np.float32) * 0.05
    b1 = rng.standard_normal(2 * width).astype(np.float32)
    w2 = rng.standard_normal((hidden, width)).astype(np.float32) * 0.05
    b2 = rng.standard_normal(hidden).astype(np.float32)
    x = rng.standard_normal((2, rows, hidden)).astype(np.float32)
    kernel = mlp_int8.Int8MLP(mx.array(w1), mx.array(w2), 256, mx.array(b1), mx.array(b2))
    out = np.array(kernel(mx.array(x)).astype(mx.float32))
    fused = x @ w1.T + b1
    expected = (_silu(fused[..., :width]) * fused[..., width:]) @ w2.T + b2
    assert out.shape == expected.shape
    assert _relative(out, expected) < 0.04
    without = np.array(mlp_int8.Int8MLP(mx.array(w1), mx.array(w2), 256)(mx.array(x)).astype(mx.float32))
    assert _relative(without, expected) > 0.2  # so the bias really is applied


class _FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 8 * dim, bias=True)
        self.w2 = nn.Linear(4 * dim, dim, bias=True)
        self._inner = 4 * dim

    def __call__(self, x):
        fused = self.w1(x)
        return self.w2(nn.silu(fused[..., : self._inner]) * fused[..., self._inner :])


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attention(dim)
        self.ff = _FeedForward(dim)


class _Decoder:
    def __init__(self, dim, count):
        self.transformer_blocks = [_Block(dim) for _ in range(count)]


class _VAE:
    def __init__(self, dim=256, count=2):
        self.decoder = _Decoder(dim, count)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_accelerate_swaps_the_blocks_and_keeps_their_arithmetic(dtype):
    mx.random.seed(3)
    vae = _VAE()
    x = mx.random.normal((3, 50, 256)).astype(getattr(mx, dtype))
    before = [(np.array(block.ff(x).astype(mx.float32)), np.array(block.attn.to_qkv(x).astype(mx.float32)),
               np.array(block.attn.to_out(x).astype(mx.float32))) for block in vae.decoder.transformer_blocks]
    assert vae_fast.accelerate(vae) == {"mlp": 2, "qkv": 2, "out": 2}
    for block, (ff, qkv, out) in zip(vae.decoder.transformer_blocks, before, strict=True):
        for got, want in ((block.ff(x), ff), (block.attn.to_qkv(x), qkv), (block.attn.to_out(x), out)):
            assert got.dtype == x.dtype and got.shape == want.shape
            assert _relative(np.array(got.astype(mx.float32)), want) < 0.06


def test_accelerate_can_leave_the_projections_alone():
    vae = _VAE(count=1)
    assert vae_fast.accelerate(vae, projections=False) == {"mlp": 1, "qkv": 0, "out": 0}
    assert isinstance(vae.decoder.transformer_blocks[0].attn.to_qkv, nn.Linear)
