# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Row-bounded decode top-k (TF_DSV41_INDEX_BOUND): `_dtopk` with tile loops stopping at the row's visible end."""

from __future__ import annotations

import triton
import triton.language as tl

from tensorfold.families.deepseek_v41.cuda.csa2.dtopk import _tile


@triton.jit
def _dtopk_b(S, s_stride, P, p_stride, POS, NK, OUT, o_stride, CNT, RATIO: tl.constexpr, K: tl.constexpr,
             MODE: tl.constexpr, ROWS: tl.constexpr, T: tl.constexpr, BS: tl.constexpr, RB: tl.constexpr,
             NPASS: tl.constexpr, SORT: tl.constexpr, COUNT: tl.constexpr, KP: tl.constexpr):
    """`_dtopk` over tiles [0, BND) of the row: the same OUT / CNT bytes."""

    r = tl.program_id(0)
    if ROWS:
        q = tl.load(POS + r).to(tl.int32)
    else:
        q = tl.load(POS).to(tl.int32) + r
    nvis = (q + 1) // RATIO
    NB: tl.constexpr = 1 << RB
    if MODE == 2:
        STEP: tl.constexpr = T // BS
    else:
        STEP: tl.constexpr = T
    if MODE == 0:
        BND = tl.minimum(NK, (tl.maximum(nvis, K) + STEP - 1) // STEP * STEP)
        RADIX = BND > K
    elif MODE == 2:
        BND = tl.minimum(NK, ((nvis + BS - 1) // BS + STEP - 1) // STEP * STEP)
        RADIX = NK > K
    else:
        BND = NK
        RADIX = NK > K
    idx = tl.arange(0, NB)
    thr = tl.full((), 0, tl.uint64)
    if RADIX:                                    # else every entry is in (the K-th key does not exist)
        hmask = tl.full((), 0, tl.uint64)
        above = tl.zeros((), tl.int32)           # keys above the current prefix's range
        go = tl.full((), 1, tl.int32)
        for t in range(NPASS):
            if go != 0:
                sh = tl.maximum(64 - RB * (t + 1), 0).to(tl.uint64)
                hist = tl.zeros((NB,), tl.int32)
                for t0 in range(0, BND, STEP):
                    u, valid, p, inr = _tile(S, s_stride, P, p_stride, r, t0, NK, nvis, MODE, T, BS)
                    d = ((u >> sh) & (NB - 1)).to(tl.int32)
                    hist += tl.histogram(d, NB, mask=inr & ((u & hmask) == thr))
                if MODE == 2:                    # the skipped invalid blocks' zeros, as `_dtopk` counts them
                    hist += tl.where((idx == 0) & (thr == 0), NK - BND, 0)
                ge = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist        # entries with digit >= idx
                dg = tl.max(tl.where(above + ge >= K, idx, -1), 0)
                hd = tl.sum(tl.where(idx == dg, hist, 0), 0)
                above += tl.sum(tl.where(idx == dg, ge, 0), 0) - hd
                thr = thr | (dg.to(tl.uint64) << sh)
                hmask = hmask | (tl.full((), NB - 1, tl.uint64) << sh)
                if above + hd == K:              # the prefix's keys complete the top K exactly
                    go = tl.zeros((), tl.int32)
    orow = OUT + r.to(tl.int64) * o_stride
    base = tl.zeros((), tl.int32)
    nv = tl.zeros((), tl.int32)
    for t0 in range(0, BND, STEP):
        u, valid, p, inr = _tile(S, s_stride, P, p_stride, r, t0, NK, nvis, MODE, T, BS)
        sel = valid & (u >= thr)
        si = sel.to(tl.int32)
        off = base + tl.cumsum(si, 0) - 1
        tl.store(orow + off, p, mask=sel & (off < K))
        base += tl.sum(si, 0)
        nv += tl.sum((sel & (p < nvis)).to(tl.int32), 0)
    kk = tl.arange(0, KP)
    if SORT:
        tl.debug_barrier()
        v = tl.load(orow + kk, mask=kk < base, other=2147483647)
        v = tl.sort(v, 0)
        tl.debug_barrier()
        tl.store(orow + kk, tl.where(v == 2147483647, -1, v), mask=kk < K)
    else:
        tl.store(orow + kk, tl.full((KP,), -1, tl.int32), mask=(kk >= base) & (kk < K))
    if COUNT:
        tl.store(CNT + r, nv)


__all__ = ["_dtopk_b"]
