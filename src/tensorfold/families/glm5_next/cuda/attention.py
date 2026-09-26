"""Exact causal attention for GLM's MLA layers on chains of rows (keys and values expanded per head, head dim
256, no RoPE), with the committed length read on the device so a forward can be captured in a CUDA graph.

The window's keys and values are written into the cache (slots P .. P + R - 1) before attention, so row r
(absolute position P + r) attends to cache keys 0 .. P + r. A program handles up to 16 rows of one head and
one 512-key chunk (eight 64-key tensor-core tiles, the 27B engine's tile step); chunk partials merge in
absolute key order. A row's arithmetic is the same whatever the other rows of its tile, and chunks are
fixed by absolute key position, so a window row gets the bits of the serial step at that position.
Chunks past the last key are empty and skipped by the merge, so the chunk count (exact, or a capacity
bound for a captured graph) never changes a result.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

TILE = 64
CHUNK = 512
BR = 16


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    scores = tl.where(valid, scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _chunks(Q, KC, VC, POS, PO, PM, PL, R, H: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
            SCALE: tl.constexpr, BRT: tl.constexpr):
    rb = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    rr = rb * BRT + tl.arange(0, BRT)
    rok = rr < R
    d = tl.arange(0, D)
    m = tl.full((BRT,), float("-inf"), tl.float32)
    l = tl.zeros((BRT,), tl.float32)
    o = tl.zeros((BRT, D), tl.float32)
    start = c * CH
    if start < P + R:
        q = tl.load(Q + (rr[:, None] * H + h) * D + d[None, :], mask=rok[:, None], other=0).to(tl.bfloat16)
        limit = P + rr
        for t in range(CH // 64):
            ki = start + t * 64 + tl.arange(0, 64)
            inside = ki < P + R
            kk = tl.load(KC + (ki[:, None] * H + h) * D + d[None, :], mask=inside[:, None], other=0).to(tl.bfloat16)
            vv = tl.load(VC + (ki[:, None] * H + h) * D + d[None, :], mask=inside[:, None], other=0).to(tl.bfloat16)
            valid = (ki[None, :] <= limit[:, None]) & rok[:, None]
            m, l, o = _tile(q, kk, vv, m, l, o, valid, SCALE)
    base = (c * R + rr) * H + h
    tl.store(PO + base[:, None] * D + d[None, :], o, mask=rok[:, None])
    tl.store(PM + base, m, mask=rok)
    tl.store(PL + base, l, mask=rok)


@triton.jit
def _merge(PO, PM, PL, OUT, R, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * D + d)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * D + d, (o / l).to(tl.bfloat16))


class AttnScratch:
    """Chunk partials for up to ``rows`` rows, ``heads`` heads and a cache of ``capacity`` keys."""

    def __init__(self, rows: int, heads: int, dim: int, capacity: int, device) -> None:
        self.nch = triton.cdiv(capacity + rows, CHUNK)
        self.po = torch.empty((self.nch * rows * heads * dim,), dtype=torch.float32, device=device)
        self.pm = torch.empty((self.nch * rows * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((self.nch * rows * heads,), dtype=torch.float32, device=device)
        self.out = torch.empty((rows, heads, dim), dtype=torch.bfloat16, device=device)


def attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: torch.Tensor, scratch: AttnScratch,
              *, scale: float, nch: int | None = None) -> torch.Tensor:
    """q [R, H, D] (rows at positions pos .. pos + R - 1), caches [capacity, H, D] holding keys through pos + R - 1,
    pos an int32 [1] device tensor. ``nch``: chunks to visit (default: all the scratch holds). -> [R, H, D] bf16."""

    R, H, D = q.shape
    nch = scratch.nch if nch is None else min(nch, scratch.nch)
    if R > BR * 8 or D not in (128, 256):
        raise ValueError("unsupported attention shape")
    po = scratch.po[:nch * R * H * D]
    pm = scratch.pm[:nch * R * H]
    pl = scratch.pl[:nch * R * H]
    _chunks[(triton.cdiv(R, BR), H, nch)](q, k_cache, v_cache, pos, po, pm, pl, R, H=H, D=D, CH=CHUNK, SCALE=scale,
                                          BRT=BR, num_warps=4, num_stages=1)
    out = scratch.out[:R]
    _merge[(R, H)](po, pm, pl, out, R, H=H, D=D, NCH=nch, num_warps=4)
    return out


@triton.jit
def _kv_write(KN, VN, KC, VC, POS, W: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    P = tl.load(POS).to(tl.int64)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    tl.store(KC + (P + r) * W + offs, tl.load(KN + r * W + offs))
    tl.store(VC + (P + r) * W + offs, tl.load(VN + r * W + offs))


def kv_write(kn: torch.Tensor, vn: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: torch.Tensor) -> None:
    """Rows [R, H, D] into cache slots pos .. pos + R - 1 (pos read on the device)."""

    R = kn.shape[0]
    W = kn.shape[1] * kn.shape[2]
    _kv_write[(R, W // 1024)](kn, vn, k_cache, v_cache, pos, W=W, BLOCK=1024, num_warps=4)
