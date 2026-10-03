"""Decode indexer top-k as a bounded radix select: one program a row, reading only the row's visible entries.

The same entries as ``kernels.top_entries(kernels.mask_to_blocks(scores, ...))`` and ``kernels.candidate_blocks`` (the
reference, which the prompt path keeps): exact-zero scores are untied by index exactly as ``kernels.untie`` does (the
same fp32 steps), invisible / masked entries never count, and the result is the ascending index list padded with -1.
Exactly equal non-zero scores straddling the k-th place go to the lower index (``torch.topk`` leaves that case
unspecified). Graph-safe: no host sync, every bound is read on the device.

Design after Jay Leaton's DeepSeek-V4.1 TensorFold recipe (``csa2/dtopk.py``: digit histograms, a masked radix walk,
an ordered compaction; MIT); written for this engine's scores, untie and candidate blocks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BS = 1024           # entries a block of the scan


@triton.jit
def _key(v):
    """fp32 -> uint32 whose unsigned order is the float order."""

    b = v.to(tl.int32, bitcast=True)
    k = tl.where(b < 0, ~b, b | (-2147483648))
    return k.to(tl.uint32, bitcast=True)


@triton.jit
def _untie(s, i):
    """kernels.untie: an exact zero becomes -1e-30 * (1 + i * 2^-21) (the same fp32 rounding)."""

    t = 1.0 + i.to(tl.float32) * 4.76837158203125e-07
    return tl.where(s == 0.0, t * -1e-30, s)


@triton.jit
def _topk(S, POS, FLAGS, OUT, FLAGS_OUT, s_stride, f_stride, o_stride, n_keys, ratio, k, block, nb_out,
          MODE: tl.constexpr, BS: tl.constexpr):
    """MODE 0: entries of S (row r, n_keys of them, visible below (pos + 1) // ratio). MODE 1: the same, kept only
    in blocks FLAGS marks. MODE 2: candidate blocks: S holds block maxima (n_keys blocks of ``block`` entries; the
    newest visible block is pinned), writes FLAGS_OUT [r, block] = 1 for the chosen ones (0 below ``nb_out``)."""

    r = tl.program_id(0)
    p = tl.load(POS + r)
    nvis = (p + 1) // ratio
    if MODE == 2:
        n = tl.minimum(n_keys, (nvis + block - 1) // block)
        newest = tl.maximum(nvis - 1, 0) // block
        n = tl.maximum(n, tl.minimum(newest + 1, n_keys))
    else:
        n = tl.minimum(n_keys, nvis)
    offs = tl.arange(0, BS)
    bins = tl.arange(0, 256)
    row = S + r * s_stride
    # the valid count, then four 8-bit digit passes (high to low) for the k-th largest key
    prefix = tl.zeros((), dtype=tl.uint32)
    need = k
    total = tl.zeros((), dtype=tl.int32)
    for d in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for b0 in range(0, n, BS):
            i = b0 + offs
            inb = i < n
            s = tl.load(row + i, mask=inb, other=float("-inf"))
            v = _untie(s, i)
            if MODE == 2:
                v = tl.where(i == newest, float("inf"), v)
            ok = inb & (v != float("-inf"))
            if MODE == 1:
                f = tl.load(FLAGS + r * f_stride + i // block, mask=inb, other=0)
                ok = ok & (f != 0)
            key = _key(v)
            if d > 0:
                ok = ok & ((key >> (32 - 8 * d)) == prefix)
            dig = ((key >> (24 - 8 * d)) & 255).to(tl.int32)
            hist += tl.histogram(dig, 256, mask=ok)
        if d == 0:
            total = tl.sum(hist)
        above = tl.cumsum(hist, 0, reverse=True)          # entries in this bin or higher
        pick = tl.max(tl.where(above >= need, bins, -1))
        pick = tl.maximum(pick, 0)
        higher = tl.sum(tl.where(bins > pick, hist, 0))
        need = need - higher
        prefix = (prefix << 8) | pick.to(tl.uint32)
    thresh = prefix
    take_all = total <= k                                # every valid entry, the rest -1
    # ordered compaction: keys above the k-th, then the first ``need`` equal to it, in index order
    got = tl.zeros((), dtype=tl.int32)
    eq = tl.zeros((), dtype=tl.int32)
    for b0 in range(0, n, BS):
        i = b0 + offs
        inb = i < n
        s = tl.load(row + i, mask=inb, other=float("-inf"))
        v = _untie(s, i)
        if MODE == 2:
            v = tl.where(i == newest, float("inf"), v)
        ok = inb & (v != float("-inf"))
        if MODE == 1:
            f = tl.load(FLAGS + r * f_stride + i // block, mask=inb, other=0)
            ok = ok & (f != 0)
        key = _key(v)
        is_eq = ok & (key == thresh)
        eq_rank = eq + tl.cumsum(is_eq.to(tl.int32), 0) - is_eq.to(tl.int32)
        sel = ok & (take_all | (key > thresh) | (is_eq & (eq_rank < need)))
        at = got + tl.cumsum(sel.to(tl.int32), 0) - sel.to(tl.int32)
        if MODE == 2:
            tl.store(FLAGS_OUT + r * nb_out + i, sel.to(tl.uint8), mask=inb)
        else:
            tl.store(OUT + r * o_stride + at, i.to(tl.int32), mask=sel)
        got += tl.sum(sel.to(tl.int32))
        eq += tl.sum(is_eq.to(tl.int32))
    if MODE == 2:
        for b0 in range(n, nb_out, BS):                  # blocks past the visible ones: not chosen
            i = b0 + offs
            tl.store(FLAGS_OUT + r * nb_out + i, tl.zeros((BS,), dtype=tl.uint8), mask=i < nb_out)
    else:
        for b0 in range(got, k, BS):                     # the rest: -1
            i = b0 + offs
            tl.store(OUT + r * o_stride + i, tl.full((BS,), -1, dtype=tl.int32), mask=i < k)


def top_entries(scores: torch.Tensor, pos: torch.Tensor, ratio: int, topk: int,
                flags: torch.Tensor | None = None, block: int = 1) -> torch.Tensor:
    """int32 [R, topk]: ``kernels.top_entries`` over each row's visible entries (and, with ``flags`` [R, blocks]
    uint8, only those in flagged blocks of ``block`` entries: ``kernels.mask_to_blocks``)."""

    R, S = scores.shape
    out = torch.empty((R, topk), dtype=torch.int32, device=scores.device)
    f = flags if flags is not None else out
    _topk[(R,)](scores, pos, f, out, out, scores.stride(0), f.stride(0) if flags is not None else 0, out.stride(0),
                S, ratio, topk, block, 0, MODE=1 if flags is not None else 0, BS=BS, num_warps=8)
    return out


def candidate_flags(scores: torch.Tensor, pos: torch.Tensor, ratio: int, block: int, keep: int) -> torch.Tensor:
    """uint8 [R, ceil(S / block)]: 1 for ``kernels.candidate_blocks``' chosen blocks (block maxima, untied by block
    index, the newest visible block pinned, the ``keep`` best)."""

    R, S = scores.shape
    nb = -(-S // block)
    padded = scores if S % block == 0 else torch.nn.functional.pad(scores, (0, nb * block - S), value=float("-inf"))
    best = padded.view(R, nb, block).amax(-1)
    flags = torch.empty((R, nb), dtype=torch.uint8, device=scores.device)
    _topk[(R,)](best, pos, flags, flags, flags, best.stride(0), 0, 0, nb, ratio, keep, block, nb, MODE=2, BS=BS,
                num_warps=8)
    return flags
