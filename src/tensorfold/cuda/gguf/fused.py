"""Fused IQ2 SoA gate+up+SwiGLU for grouped prefill (family-agnostic)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .linear import _values_iq2_soa, codebook


@triton.jit
def _gate_up_swiglu_soa(
    X,
    G,
    U,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DQ_BYTES: tl.constexpr,
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
        inputs = pairs // SLOTS
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        ag = tl.zeros((BM, BN), tl.float32)
        au = tl.zeros((BM, BN), tl.float32)
        gdq = G.to(tl.pointer_type(tl.float16))
        gqs = (G + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        udq = U.to(tl.pointer_type(tl.float16))
        uqs = (U + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], valid[:, None] & (ki[None, :] < K), 0)
            mask = (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS)
            bg = _values_iq2_soa(gdq, gqs, GRID, expert * N + n[None, :], ki[:, None], mask, K).to(tl.bfloat16)
            bu = _values_iq2_soa(udq, uqs, GRID, expert * N + n[None, :], ki[:, None], mask, K).to(tl.bfloat16)
            ag = tl.dot(a.to(tl.bfloat16), bg, ag)
            au = tl.dot(a.to(tl.bfloat16), bu, au)
        g = tl.minimum(ag.to(tl.bfloat16).to(tl.float32), 10.0)
        u = tl.maximum(-10.0, tl.minimum(10.0, au.to(tl.bfloat16).to(tl.float32)))
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], g / (1.0 + tl.exp(-g)) * u, valid[:, None] & (n[None, :] < N))


def gate_up_swiglu(x, gate, up, picks, plan):
    """Return SwiGLU(gate(x), up(x)) for a prefill Plan; bit-identical to separate linears + swiglu."""

    import os

    # Fused CUDA D2R is the default SoA path; TENSORFOLD_GGUF_CUDA_PREFILL=0 keeps Triton.
    if os.environ.get("TENSORFOLD_GGUF_CUDA_PREFILL", "1") != "0":
        from .cuda_prefill import iq2_soa_gate_up_prefill

        return iq2_soa_gate_up_prefill(x, gate, up, plan, slots=picks.shape[1])

    k, n = gate.shape[:2]
    rows, slots = picks.shape
    out = torch.empty((rows, slots, n), dtype=torch.bfloat16, device=x.device)
    bn, bk = 128, 32
    _gate_up_swiglu_soa[(plan.items.shape[0], triton.cdiv(n, bn))](
        x.contiguous(),
        gate.data,
        up.data,
        codebook(x.device),
        plan.items,
        plan.members,
        plan.counts,
        out,
        n,
        k,
        slots,
        gate.shape[2],
        rows * slots,
        plan.tile,
        bn,
        bk,
        gate.bn,
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out
