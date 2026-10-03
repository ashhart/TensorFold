"""Experimental prompt up projection and mix, preserving bf16 weight FMA and the ordered MMA chain."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def _upmix(X, W, SCALE, BIAS, NORMED, MIXED, XS, UP, M,
           N: gl.constexpr, K: gl.constexpr, D: gl.constexpr, S: gl.constexpr,
           BM: gl.constexpr, DB: gl.constexpr, WARPS: gl.constexpr, WM: gl.constexpr, STORE_UP: gl.constexpr):
    blocked: gl.constexpr = BlockedLayout([1, 4], [4, 8], [WARPS, 1], [1, 0])
    mma: gl.constexpr = NVMMADistributedLayout([2, 0], [WM, WARPS // WM], [16, 8])
    rm = gl.program_id(0) * BM + gl.arange(0, BM, layout=SliceLayout(1, blocked))
    wn = gl.arange(0, DB, layout=SliceLayout(1, blocked))
    kk = gl.arange(0, 32, layout=SliceLayout(0, blocked))
    dd = gl.program_id(1) * DB + gl.arange(0, DB, layout=SliceLayout(0, blocked))
    total = gl.zeros((BM, DB), gl.float32, layout=blocked)
    for stream in gl.static_range(S):
        n = stream * D + gl.program_id(1) * DB + wn
        acc = gl.zeros((BM, DB), gl.float32, layout=mma)
        for group in range(K // 32):
            x = gl.load(X + rm[:, None] * K + (group * 32 + kk)[None, :], rm[:, None] < M, 0)
            words = gl.load(W + (n[:, None] // 64) * (K // 32 * 64 * 4) + group * 64 * 4
                            + (n[:, None] % 64) * 4 + (kk // 8)[None, :])
            q = ((words >> ((kk % 8) * 4)[None, :]) & 15).to(gl.bfloat16)
            scale = gl.load(SCALE + group * N + n)
            bias = gl.load(BIAS + group * N + n)
            weight = gl.inline_asm_elementwise(
                "fma.rn.bf16x2 $0, $1, $2, $3;", constraints="=r,r,r,r",
                args=[q, scale[:, None], bias[:, None]], dtype=gl.bfloat16, is_pure=True, pack=2)
            a = gl.convert_layout(x, DotOperandLayout(0, mma, 2))
            b = gl.convert_layout(gl.permute(weight, (1, 0)), DotOperandLayout(1, mma, 2))
            acc = mma_v2(a, b, acc)
        up = gl.convert_layout(acc.to(gl.bfloat16), blocked)
        if STORE_UP:
            gl.store(UP + rm[:, None] * N + stream * D + dd[None, :], up, rm[:, None] < M)
        norm = gl.load(NORMED + rm[:, None] * N + stream * D + dd[None, :], rm[:, None] < M, 0).to(gl.float32)
        gate = (1.0 / (1.0 + gl.exp(-up.to(gl.float32)))).to(gl.bfloat16).to(gl.float32)
        total += (gate * norm).to(gl.bfloat16).to(gl.float32)
    mixed = (total / S).to(gl.bfloat16)
    gl.store(MIXED + rm[:, None] * D + dd[None, :], mixed, rm[:, None] < M)
    groups: gl.constexpr = BlockedLayout([1, 1, 4], [4, 1, 8], [WARPS, 1, 1], [2, 1, 0])
    grouped = gl.convert_layout(gl.reshape(mixed.to(gl.float32), (BM, DB // 32, 32)), groups)
    sums = gl.sum(grouped, axis=2)
    summed: gl.constexpr = SliceLayout(2, groups)
    xr = gl.program_id(0) * BM + gl.arange(0, BM, layout=SliceLayout(1, summed))
    xg = gl.program_id(1) * (DB // 32) + gl.arange(0, DB // 32, layout=SliceLayout(0, summed))
    gl.store(XS + xr[:, None] * (D // 32) + xg[None, :], sums, xr[:, None] < M)


def prefill_upmix(act, q, normed, mixed, xs, streams: int, *, up=None, tile=(32, 128, 8)) -> None:
    """The prompt matmul and hc_mix outputs; optional up stores the rounded projection for byte checks."""

    if q.layout != "tiled" or q.gs != 32 or q.k != act.shape[1] or q.n % (streams * 32):
        raise ValueError("prompt upmix requires tiled group-32 weights and whole output groups")
    rows, dims = act.shape[0], q.n // streams
    bm, db, warps = tile
    if dims % db or bm not in (16, 32, 64) or db not in (32, 64, 128) or warps not in (4, 8):
        raise ValueError("unsupported prompt upmix tile")
    _upmix[(triton.cdiv(rows, bm), dims // db)](
        act, q.weight, q.scales, q.biases, normed, mixed, xs, normed if up is None else up, rows,
        N=q.n, K=q.k, D=dims, S=streams, BM=bm, DB=db, WARPS=warps, WM=bm // 16,
        STORE_UP=up is not None, num_warps=warps, num_stages=2)
