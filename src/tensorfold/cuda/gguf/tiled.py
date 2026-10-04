"""Triton grouped prefill on prepared GGUF tiles (contiguous BN x BLOCK per K=256)."""

from __future__ import annotations

import triton
import triton.language as tl

from .linear import FORMATS, _values, codebook


@triton.jit
def _group_mm_tiled(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    INPUT_SLOTS: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        m = tl.arange(0, BM)
        valid = (m < count) & (first + m >= 0) & (first + m < PAIRS)
        pairs = tl.load(MEMBERS + first + m, valid, 0)
        valid = valid & (pairs >= 0) & (pairs < PAIRS) & (expert >= 0) & (expert < EXPERTS)
        inputs = pairs if INPUT_SLOTS else pairs // SLOTS
        nt = tl.program_id(1)
        n = nt * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        nb = tl.cdiv(N, BN)
        kg_total = K // 256
        for kg in range(kg_total):
            base = ((expert * nb + nt) * kg_total + kg) * BN * BLOCK
            for sub in range(256 // BK):
                ki = kg * 256 + sub * BK + k
                a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
                # Prepared tile viewed as N'=BN, K=256 raw matrix starting at W+base.
                b = _values(
                    W + base,
                    GRID,
                    (n - nt * BN)[None, :],
                    (sub * BK + k)[:, None],
                    (n[None, :] < N) & ((sub * BK + k)[:, None] < 256),
                    256,
                    FORMAT,
                ).to(tl.bfloat16)
                acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


def group_prefill_tiled(x, weight, plan, out, *, slots: int, input_slots: bool) -> None:
    k, n = weight.shape[:2]
    block = FORMATS[weight.format][1]
    bn = weight.bn
    bk = 32
    _group_mm_tiled[(plan.items.shape[0], triton.cdiv(n, bn))](
        x.contiguous(),
        weight.data,
        codebook(x.device),
        plan.items,
        plan.members,
        plan.counts,
        out,
        n,
        k,
        slots,
        input_slots,
        weight.format,
        weight.shape[2],
        plan.rows * plan.slots,
        plan.tile,
        bn,
        bk,
        block,
        num_warps=8 if plan.tile == 64 else 4,
        num_stages=1,
        enable_fp_fusion=False,
    )
