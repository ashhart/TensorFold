"""Nemotron's norm kernels that also write the lane matmul's input sums: the normed rows keep their bits, and the
sums are bit for bit the lane matmul's own (its XSUM kernel), so a projection handed them gives the bits of one
that computes them (GPU)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import lane_glue, lane_qmm  # noqa: E402

D = 2688


def _same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint32 if a.dtype == mx.float32 else mx.uint16),
                                                      b.view(mx.uint32 if b.dtype == mx.float32 else mx.uint16)))


def _xsum(x):
    rows = int(x.shape[0])
    mp = 16 * -(-rows // 16)
    return lane_qmm._kernel("xsum")(inputs=[x, lane_qmm._mdims(rows, mp)], template=[("K", D), ("GS", 64)],
                                    grid=(D // 64, mp, 1),
                                    threadgroup=(D // 64, 1, 1), output_shapes=[(D // 64, mp)],
                                    output_dtypes=[mx.float32])[0]


def _normal(shape, seed, scale=1.0):
    return (scale * mx.random.normal(shape, key=mx.random.key(seed))).astype(mx.bfloat16)


@pytest.mark.parametrize("rows", [1, 3, 16, 17, 40])
def test_norm_kernels_write_the_lane_matmuls_sums(rows):
    eps = mx.array([1e-5], dtype=mx.float32)
    h, delta, weight = _normal((rows, D), 1), _normal((rows, D), 2), 1 + _normal((D,), 3, 0.1)
    hn, out = K.add_norm(h, delta, weight, eps)
    hn2, out2, xs = K.add_norm(h, delta, weight, eps, group_sums=True)
    assert _same(hn, hn2) and _same(out, out2) and _same(xs, _xsum(out))

    routed, shared = _normal((rows, 6, D), 4), _normal((rows, D), 5)
    weights = mx.random.uniform(shape=(rows, 6), key=mx.random.key(6))
    hn, out = K.add_norm_moe(h, routed, weights, shared, weight, eps)
    hn2, out2, xs = K.add_norm_moe(h, routed, weights, shared, weight, eps, group_sums=True)
    assert _same(hn, hn2) and _same(out, out2) and _same(xs, _xsum(out))


@pytest.mark.skipif(not K.tensor_units(), reason="the lane matmul needs tensor units")
def test_a_projection_handed_the_sums_gives_its_own_bits():
    eps = mx.array([1e-5], dtype=mx.float32)
    _, out, xs = K.add_norm(_normal((5, D), 7), _normal((5, D), 8), 1 + _normal((D,), 9, 0.1), eps, group_sums=True)
    w = _normal((1024, D), 10, 0.02)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    sbt = lane_qmm.pack_scales(s, b)
    lane_qmm._xs_cache.clear()
    own = lane_qmm.lane_matmul(out, q, sbt)
    lane_qmm._xs_cache.clear()
    handed = lane_qmm.lane_matmul(lane_glue.remember(out, xs), q, sbt)
    assert _same(own, handed)
