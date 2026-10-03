"""Fused elementwise expert operations; each row and slot has its own fixed arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu(G, U, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.minimum(tl.load(G + i, i < N, 0).to(tl.float32), 10.0)
    u = tl.maximum(-10.0, tl.minimum(10.0, tl.load(U + i, i < N, 0).to(tl.float32)))
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(Y + i, y, i < N)


def swiglu(gate, up):
    out = torch.empty_like(gate)
    _swiglu[(triton.cdiv(out.numel(), 256),)](gate, up, out, out.numel(), 256, enable_fp_fusion=False)
    return out


@triton.jit
def _combine(Y, W, SHARED, OUT, D: tl.constexpr, SLOTS: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    for s in tl.static_range(SLOTS):
        y = tl.load(Y + (r * SLOTS + s) * D + d, d < D, 0).to(tl.float32)
        w = tl.load(W + r * SLOTS + s)
        acc += y * w
    shared = tl.load(SHARED + r * D + d, d < D, 0).to(tl.float32)
    tl.store(OUT + r * D + d, acc + shared, d < D)


def combine(down, weights, shared):
    rows, slots, dims = down.shape
    out = torch.empty_like(shared)
    _combine[(rows, triton.cdiv(dims, 256))](down, weights, shared, out, dims, slots, 256, enable_fp_fusion=False)
    return out
