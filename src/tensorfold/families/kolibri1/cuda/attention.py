"""Kolibri 1's attention: sliding-window rows over their last ``window`` keys, full rows on the shared kernels."""
# A sliding row walks 64-key tiles by absolute position: its bits never depend on its chunk or the rows beside it.

from __future__ import annotations

import torch
import triton
import triton.language as tl

BN = 64           # keys a tile
GP = 16           # query heads a program holds (a key head's group, padded)


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    s = tl.where(valid[None, :], s, float("-inf"))
    tile_m = tl.max(s, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _sliding(Q, KC, VC, OUT, POS, SLOT, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
             WIN: tl.constexpr, RING: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr, GP: tl.constexpr):
    """Program (row, key head): the row's G query heads over keys (pos - WIN, pos], key p at ring slot p % RING."""

    r = tl.program_id(0)
    hk = tl.program_id(1)
    pos = tl.load(POS + r)
    base = tl.load(SLOT + r).to(tl.int64) * RING
    gg = tl.arange(0, GP)
    d = tl.arange(0, D)
    hm = gg < G
    q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=hm[:, None], other=0.0)
    lo = tl.maximum(pos - WIN + 1, 0)
    m = tl.full((GP,), float("-inf"), tl.float32)
    l = tl.zeros((GP,), tl.float32)
    o = tl.zeros((GP, D), tl.float32)
    for t in range(lo // BN, pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        valid = (keys >= lo) & (keys <= pos)
        at = (base + keys % RING)[:, None] * HK + hk
        k = tl.load(KC + at * D + d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(VC + at * D + d[None, :], mask=valid[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, valid, SCALE)
    out = o / l[:, None]
    tl.store(OUT + (r * H + hk * G + gg[:, None]) * D + d[None, :], out.to(tl.bfloat16), mask=hm[:, None])


def ring_size(rows: int, window: int) -> int:
    """Ring slots a sliding layer keeps: a forward writes ``rows`` keys before its first row reads ``window`` back."""

    return -(-(rows + window) // BN) * BN


def sliding(q: torch.Tensor, k_ring: torch.Tensor, v_ring: torch.Tensor, positions: torch.Tensor,
            slots: torch.Tensor, *, window: int, scale: float) -> torch.Tensor:
    """q (W, H, D) bf16 at ``positions`` (W,) int32 of streams ``slots`` (W,) int32; rings (S, RING, HK, D)."""

    w, h, d = q.shape
    hk, ring = k_ring.shape[2], k_ring.shape[1]
    if h % hk or h // hk > GP or d not in (64, 128, 256) or positions.shape != (w,) or slots.shape != (w,):
        raise ValueError("sliding attention: heads a multiple of key heads (at most 16 a group), D 64/128/256")
    if not (q.is_contiguous() and k_ring.is_contiguous() and v_ring.is_contiguous()):
        raise ValueError("sliding attention takes contiguous tensors")
    out = torch.empty_like(q)
    _sliding[(w, hk)](q, k_ring, v_ring, out, positions, slots, H=h, HK=hk, D=d, G=h // hk, WIN=int(window),
                      RING=int(ring), SCALE=float(scale), BN=BN, GP=GP, num_warps=4, num_stages=2)
    return out


def full_prompt(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *,
                scale: float) -> torch.Tensor:
    """Prompt rows at [p0, p0 + W) over every key before them (the caches hold the chunk's keys)."""

    from tensorfold.cuda.kernels import prefill_attention

    return prefill_attention.attention(q, k_cache, v_cache, p0, scale=scale)


def full_rows(q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
              caches: list[tuple[torch.Tensor, torch.Tensor]],
              groups: list[tuple[int, int]], *, scale: float) -> torch.Tensor:
    """Streams' serial chains in one call: ``groups`` (rows, committed keys) each over its own (keys, values) cache."""

    from tensorfold.cuda.kernels import attention as tree

    plan = tree.plan([[i - 1 for i in range(n)] for n, _ in groups], [c for _, c in groups],
                     q.shape[1] // k_new.shape[1], q.device)
    offs = torch.tensor(tree.offsets(caches, q.device), dtype=torch.int64, device=q.device).view(len(groups), 2)
    return tree.attention(q, k_new, v_new, offs, plan, scale=scale)
