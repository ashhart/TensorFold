"""lane_glue.norm_xs: a row's bits never depend on the rows beside it, and the norm is RMSNorm to bf16 rounding."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels import lane_glue, lane_qmm


def _inputs(rows: int, seed: int, scale: float = 1.0):
    mx.random.seed(seed)
    h = (mx.random.normal((1, rows, 5120)) * scale).astype(mx.bfloat16)
    r = (mx.random.normal((1, rows, 5120)) * scale).astype(mx.bfloat16)
    w = (mx.random.normal((5120,)) * 0.2 + 1).astype(mx.bfloat16)
    return h, r, w


@pytest.mark.parametrize("rows", [1, 5, 16, 17, 32])
def test_norm_xs_rows_are_independent(rows):
    h, r, w = _inputs(rows, rows)
    hs, x = lane_glue.norm_xs(h, r, w, 1e-6)
    xs = lane_qmm._xs_cache[id(x)][1]
    for row in range(rows):
        h1, x1 = lane_glue.norm_xs(h[:, row:row + 1], r[:, row:row + 1], w, 1e-6)
        xs1 = lane_qmm._xs_cache[id(x1)][1]
        assert mx.array_equal(h1[0, 0], hs[0, row]).item()
        assert mx.array_equal(x1[0, 0], x[0, row]).item()
        assert mx.array_equal(xs1[:, 0], xs[:, row]).item()


def test_norm_xs_is_rmsnorm_to_bf16_rounding():
    h, r, w = _inputs(16, 7, scale=3.0)
    hs, x = lane_glue.norm_xs(h, r, w, 1e-6)
    xs = lane_qmm._xs_cache[id(x)][1]
    hn = np.array(hs[0].astype(mx.float32), dtype=np.float64)
    ref = np.array(w.astype(mx.float32), dtype=np.float64)[None] * hn / np.sqrt((hn ** 2).mean(axis=1, keepdims=True) + 1e-6)
    got = np.array(x[0].astype(mx.float32), dtype=np.float64)
    assert np.max(np.abs(got - ref) / (np.abs(ref) + 1e-3)) <= 2 ** -8 + 1e-6      # bf16 rounding
    # group sums: the 64-input sums of x, to fp32 accumulation error
    sums = got.reshape(16, 80, 64).sum(axis=2).T
    assert np.allclose(np.array(xs[:, :16]), sums, rtol=1e-5, atol=1e-3)
