"""Prompt HC write-back and normalization retain the separate kernels' rounding and ordered partial sums."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout


@gluon.jit
def _write_norm(H, PSS, SCALE, NORMED, XS, BR, INJ, Y, WTS, RS, eps,
                D: gl.constexpr, S: gl.constexpr, MODE: gl.constexpr, SLOTS: gl.constexpr,
                WORLD: gl.constexpr, PAD: gl.constexpr):
    r = gl.program_id(0)
    vector: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    scalars: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    partial_layout: gl.constexpr = BlockedLayout([1, 4], [1, 32], [2, 2], [1, 0])
    group_layout: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    d = gl.arange(0, PAD, layout=vector)
    valid = d < D
    if MODE == 1:
        branch = gl.load(BR + r * D + d, valid, 0).to(gl.float32)
    elif MODE == 3 or MODE == 4:
        acc = gl.load(BR + r * D + d, valid, 0)
        for k in gl.static_range(1, WORLD):
            acc = acc + gl.load(BR + k * RS + r * D + d, valid, 0)
        branch = acc.to(gl.bfloat16).to(gl.float32)
    elif MODE == 2:
        acc = gl.zeros((PAD,), gl.float32, layout=vector)
        for k in gl.static_range(SLOTS):
            wk = gl.load(WTS + r * SLOTS + k)
            yk = gl.load(Y + (r * SLOTS + k) * D + d, valid, 0).to(gl.float32)
            acc = acc + yk * wk
        branch = acc.to(gl.bfloat16).to(gl.float32)
    for s in gl.static_range(S):
        hv = gl.load(H + r * (S * D) + s * D + d, valid, 0).to(gl.float32)
        if MODE != 0:
            inj = gl.load(INJ + r * S + s).to(gl.float32)
            hv = (hv + (branch * inj).to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
            gl.store(H + r * (S * D) + s * D + d, hv.to(gl.bfloat16), valid)
        squares = gl.convert_layout(gl.reshape(hv * hv, (PAD // 256, 256)), partial_layout)
        partials = gl.convert_layout(gl.sum(squares, axis=1), scalars)
        chunks = gl.arange(0, PAD // 256, layout=scalars)
        gl.store(PSS + (r * (D // 256) + chunks) * S + s, partials, chunks < D // 256)
        total = 0.0
        for c in gl.static_range(D // 256):
            total += gl.sum(gl.where(chunks == c, partials, 0.0), axis=0)
        rinv = 1.0 / gl.sqrt(total / D + eps)
        scale = gl.load(SCALE + s * D + d, valid, 0).to(gl.float32)
        y = (hv * rinv * scale).to(gl.bfloat16)
        gl.store(NORMED + r * (S * D) + s * D + d, y, valid)
        grouped = gl.convert_layout(gl.reshape(y.to(gl.float32), (PAD // 32, 32)), group_layout)
        sums = gl.convert_layout(gl.sum(grouped, axis=1), scalars)
        groups = gl.arange(0, PAD // 32, layout=scalars)
        gl.store(XS + r * (S * D // 32) + s * (D // 32) + groups, sums, groups < D // 32)


def write_norm(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor,
               xs: torch.Tensor, streams: int, eps: float, mode: int, branch=None, inject=None,
               y=None, wts=None) -> None:
    """Update streams in place, then normalize with the same 256-column sums and bf16 stores."""

    rows, wide = h.shape
    d = wide // streams
    _write_norm[(rows,)](
        h, pss, scale, normed, xs, branch if branch is not None else h,
        inject if inject is not None else h, y if y is not None else h, wts if wts is not None else h,
        rows * d, eps, D=d, S=streams, MODE=mode, SLOTS=wts.shape[1] if wts is not None else 1,
        WORLD=branch.shape[0] if mode in (3, 4) else 1, PAD=triton.next_power_of_2(d), num_warps=4)
