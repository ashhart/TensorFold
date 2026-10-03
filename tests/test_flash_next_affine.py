"""Flash Next's kernels for other MLX affine widths: close to a dequantized reference, and each row's bits the same at
any row count (GPU)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels.qwen.flash_next.v1 import base, embed, experts, hc, rows  # noqa: E402

FORMATS = [(5, 64), (6, 32), (8, 32), (8, 128), (6, 64), (3, 32), (2, 64)]


class Linear:
    """The attributes the kernels read from a quantized linear."""

    def __init__(self, weight, scales, biases, bits, group):
        self.weight, self.scales, self.biases, self.bits, self.group_size = weight, scales, biases, bits, group


def quantized(rng, shape, bits, group, scale=0.05):
    w = mx.array((scale * rng.normal(size=shape)).astype(np.float32)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    return Linear(q, s, b, bits, group)


def dense(linear):
    return mx.dequantize(linear.weight, linear.scales, linear.biases, group_size=linear.group_size,
                         bits=linear.bits).astype(mx.float32)


def bf16(rng, shape, scale=0.5):
    return mx.array((scale * rng.normal(size=shape)).astype(np.float32)).astype(mx.bfloat16)


def same(a, b):
    return bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("routed,shared", [((5, 64), (8, 128)), ((6, 32), (6, 32)), ((8, 32), (8, 32)),
                                           ((3, 32), (4, 32)), ((4, 32), (8, 64))])
def test_experts_match_the_reference_and_each_row_alone(routed, shared):
    rng = np.random.default_rng(sum(routed) * 7 + sum(shared))
    E, K, N, D, TOP, R = 32, 512, 128, 512, 4, 5
    gate, up = (quantized(rng, (E, N, K), *routed) for _ in range(2))
    down = quantized(rng, (E, D, N), *routed)
    sg, su = (quantized(rng, (N, K), *shared) for _ in range(2))
    sd = quantized(rng, (D, N), *shared)
    x = bf16(rng, (R, K))
    logits = mx.array(rng.normal(size=(R, E + 1)).astype(np.float32))
    act, picks, wts = experts.expert_gateup(x, logits, TOP, E, gate, up, shared=(sg, su))
    y = experts.expert_down_y(act, picks, down, sd)
    for r in range(R):
        one = experts.expert_gateup(x[r:r + 1], logits[r:r + 1], TOP, E, gate, up, shared=(sg, su))
        assert same(one[0][0], act[r]) and same(one[1][0], picks[r]) and same(one[2][0], wts[r]), r
        assert same(experts.expert_down_y(one[0], one[1], down, sd)[0], y[r]), r
    xf = x.astype(mx.float32)
    order = np.argsort(-np.asarray(logits[:, :E]), axis=1, kind="stable")[:, :TOP]
    assert np.array_equal(np.asarray(picks), order)
    g, u, d = dense(gate), dense(up), dense(down)
    for r in range(R):
        for k in range(TOP + 1):
            e = int(order[r, k]) if k < TOP else None
            gw, uw, dw = (g[e], u[e], d[e]) if e is not None else (dense(sg), dense(su), dense(sd))
            ref_act = mx.sigmoid(xf[r] @ gw.T) * (xf[r] @ gw.T) * (xf[r] @ uw.T)
            assert np.allclose(np.asarray(act[r, k].astype(mx.float32)), np.asarray(ref_act), rtol=0.05, atol=0.02)
            ref_y = act[r, k].astype(mx.float32) @ dw.T
            assert np.allclose(np.asarray(y[r, k].astype(mx.float32)), np.asarray(ref_y), rtol=0.05, atol=0.02)


def _hc_weights(rng, fmt, S, D, LOW, inject=True):
    down = quantized(rng, (LOW + (S if inject else 0), S * D), *fmt, scale=0.02)
    up = quantized(rng, (S * D, LOW), *fmt, scale=0.05)
    return (base.QWeights(down.weight, down.scales, down.biases, *fmt),
            base.QWeights(up.weight, up.scales, up.biases, *fmt))


@pytest.mark.parametrize("fmt", [f for f in FORMATS if 320 % f[1] == 0])     # the up projection reads 320 inputs
def test_hyper_connection_paths_give_each_row_its_one_row_bits(fmt, monkeypatch):
    S, D, LOW = 4, 2560, 320
    rng = np.random.default_rng(fmt[0] * 131 + fmt[1])
    down, up = _hc_weights(rng, fmt, S, D, LOW)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    eps = mx.array([1e-6], dtype=mx.float32)
    h = bf16(rng, (11, S * D), 0.3)
    hn, ssp = hc.hc_norm(h, streams=S)
    ones = [hc.hc_project(hn[r:r + 1], ssp[r:r + 1], down, up, scale, eps=eps, streams=S, low=LOW) for r in range(11)]
    for n in (2, 3, 9, 11):
        mixed, gates = hc.hc_project(hn[:n], ssp[:n], down, up, scale, eps=eps, streams=S, low=LOW)
        for r in range(n):
            assert same(mixed[r], ones[r][0][0]) and same(gates[r], ones[r][1][0]), (n, r)
    monkeypatch.setattr(hc, "SCALAR_ROWS", 0)
    mma = hc.hc_project(hn[:1], ssp[:1], down, up, scale, eps=eps, streams=S, low=LOW)
    assert same(mma[0][0], ones[0][0][0]) and same(mma[1][0], ones[0][1][0])
    monkeypatch.undo()
    per_row = rows.hc_project(hn, ssp, down, up, scale, eps=eps, streams=S, low=LOW)
    for r in (0, 5, 10):
        one = rows.hc_project(hn[r:r + 1], ssp[r:r + 1], down, up, scale, eps=eps, streams=S, low=LOW)
        assert same(one[0][0], per_row[0][r]) and same(one[1][0], per_row[1][r]), r
    # both paths near the dequantized reference (the modules' own math)
    ref_d = dense(Linear(down.weight, down.scales, down.biases, down.bits, down.group))
    x = hn.astype(mx.float32).reshape(11, S, D)
    rinv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    normed = (x * rinv).reshape(11, S * D) * scale
    dn = normed @ ref_d.T
    act = dn[:, :LOW] / S
    ref_u = dense(Linear(up.weight, up.scales, up.biases, up.bits, up.group))
    mix = mx.sigmoid((act * mx.sigmoid(act)) @ ref_u.T)
    ref = mx.mean((mix * normed).reshape(11, S, D), axis=1)
    got = mx.concatenate([o[0] for o in ones]).astype(mx.float32)
    assert float(mx.max(mx.abs(got - ref)).item()) < 0.05 * max(1.0, float(mx.max(mx.abs(ref)).item()))
    assert float(mx.max(mx.abs(per_row[0].astype(mx.float32) - ref)).item()) < 0.05 * max(
        1.0, float(mx.max(mx.abs(ref)).item()))


@pytest.mark.parametrize("fmt", FORMATS + [(4, 64)])
@pytest.mark.parametrize("n", [320, 322, 321])
def test_qmv_rows_of_every_width_are_row_exact(fmt, n):
    rng = np.random.default_rng(fmt[0] * 23 + fmt[1] + n)
    lin = quantized(rng, (n, 2560), *fmt)
    x = bf16(rng, (37, 2560))
    full = rows.qmv_rows(x, lin)
    for r in (0, 5, 31, 32, 36):
        assert same(rows.qmv_rows(x[r:r + 1], lin)[0], full[r]), r
    ref = x.astype(mx.float32) @ dense(lin).T
    assert np.allclose(np.asarray(full.astype(mx.float32)), np.asarray(ref), rtol=0.03, atol=0.03)


@pytest.mark.parametrize("fmt", FORMATS)
def test_lookups_equal_mlx_dequantize(fmt):
    rng = np.random.default_rng(fmt[0] + fmt[1])
    table = quantized(rng, (300, 512), *fmt)
    ids = mx.array(rng.integers(0, 300, size=(7,)).astype(np.uint32))
    ref = mx.dequantize(table.weight, table.scales, table.biases, group_size=fmt[1], bits=fmt[0])[ids]
    got = embed.embed_rows(ids, table, tile=2)
    assert same(got[:, :512], ref) and same(got[:, 512:], ref)


@pytest.mark.skipif(not __import__("tensorfold.families.qwen3_5", fromlist=["tensor_units"]).tensor_units(),
                    reason="the lane matmul needs the M5's tensor units")
@pytest.mark.parametrize("fmt", [(5, 32), (6, 32), (8, 32), (3, 32), (2, 32), (5, 64)])
def test_lane_matmul_reads_group_32_for_every_width(fmt):
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    rng = np.random.default_rng(fmt[0] * 5 + fmt[1])
    lin = quantized(rng, (256, 2560), *fmt)
    sbt = lane_qmm.pack_scales(lin.scales, lin.biases)
    tiled = lane_qmm.tile_weight(lin.weight, 32, fmt[1], bits=fmt[0])
    x = bf16(rng, (19, 2560))
    for w, kw in ((lin.weight, {}), (tiled, {"tiled": True})):
        full = lane_qmm.lane_matmul(x, w, sbt, group=fmt[1], **kw)
        for r in (0, 9, 18):
            assert same(lane_qmm.lane_matmul(x[r:r + 1], w, sbt, group=fmt[1], **kw)[0], full[r]), r
        ref = x.astype(mx.float32) @ dense(lin).T
        assert np.allclose(np.asarray(full.astype(mx.float32)), np.asarray(ref), rtol=0.03, atol=0.03)


@pytest.mark.parametrize("fmt", FORMATS + [(5, 128)])
@pytest.mark.parametrize("n,k", [(328, 2560), (2560, 6144), (40, 512)])
def test_qmv_rows_on_the_matrix_units_keeps_every_rows_bits(fmt, n, k):
    rng = np.random.default_rng(fmt[0] * 13 + fmt[1] + n)
    lin = quantized(rng, (n, k), *fmt)
    x = bf16(rng, (37, k))
    per_row = [rows.qmv_rows(x[r:r + 1], lin)[0] for r in range(37)]
    for m in (2, 3, 4, 8, 9, 17, 37):
        got = rows.qmv_rows_mma(x[:m], lin)
        assert all(same(got[r], per_row[r]) for r in range(m)), m
    assert all(same(rows.qmv_rows(x, lin)[r], per_row[r]) for r in range(37))
