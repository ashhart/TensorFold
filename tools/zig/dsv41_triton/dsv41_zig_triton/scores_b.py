# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Row-bounded index scores (TF_DSV41_INDEX_BOUND): `_scores` whose tiles past a row's visible keys only store -inf."""

from __future__ import annotations

import triton
import triton.language as tl

from tensorfold.families.deepseek_v41.cuda.csa2.rows import load_keys, prow


@triton.jit
def _scores_b(QI, W, w_stride, IK, OUT, POS, KEYS, k_stride, NK, o_stride, RATIO: tl.constexpr, H: tl.constexpr,
              D: tl.constexpr, BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, GATHER: tl.constexpr,
              PT=None, PSH: tl.constexpr = 0, KFP8: tl.constexpr = False, SL=None, PTS=0, ROWS: tl.constexpr = False,
              CBS: tl.constexpr = 0):
    """`_scores`, a dense tile past the row's visible keys storing -inf without its dot."""

    r = tl.program_id(0)
    pb = tl.program_id(1)
    if ROWS:
        q = tl.load(POS + r).to(tl.int32)
    else:
        q = tl.load(POS) + r
    nvis = (q + 1) // RATIO
    i = pb * BP + tl.arange(0, BP)
    if GATHER or pb * BP < nvis:
        if GATHER:
            if CBS > 0:
                cb = tl.load(KEYS + r.to(tl.int64) * k_stride + i // CBS, mask=i < NK, other=-1)
                j = tl.where(cb >= 0, cb * CBS + i % CBS, -1)
            else:
                j = tl.load(KEYS + r.to(tl.int64) * k_stride + i, mask=i < NK, other=-1)
            ok = (j >= 0) & (j < nvis) & (i < NK)
        else:
            j = i
            ok = (j < nvis) & (i < NK)
        hh = tl.arange(0, H)
        d = tl.arange(0, D)
        qv = tl.load(QI + (r.to(tl.int64) * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)          # [H, D]
        w = tl.load(W + r.to(tl.int64) * w_stride + hh).to(tl.float32) * WS
        if ROWS and PSH > 0:
            jr = prow(tl.where(ok, j, 0).to(tl.int64), PT + tl.load(SL + r).to(tl.int64) * PTS, PSH, ok)
        else:
            jr = prow(tl.where(ok, j, 0).to(tl.int64), PT, PSH, ok)
        k = load_keys(IK, jr, ok, d, D, KFP8)                                                             # [BP, D]
        dots = tl.dot(qv, tl.trans(k))                                                                     # [H, BP] fp32
        s = tl.sum(w[:, None] * tl.maximum(dots * SCALE, 0.0), axis=0)
        s = tl.where(ok, s, float("-inf"))
        tl.store(OUT + r.to(tl.int64) * o_stride + i, s, mask=i < NK)
    else:
        tl.store(OUT + r.to(tl.int64) * o_stride + i, tl.full([BP], float("-inf"), tl.float32), mask=i < NK)
