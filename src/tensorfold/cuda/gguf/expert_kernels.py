"""Routed GGUF row and prompt kernels on raw and SoA weight layouts."""

import triton
import triton.language as tl

from .kernels import _values


@triton.jit
def _mv_q2_soa(
    X,
    W,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    dmh = W.to(tl.pointer_type(tl.float16))
    sc = W + DM_BYTES
    qs = W + DM_BYTES + SC_BYTES
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values_q2_soa(
            dmh,
            sc,
            qs,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
        )
        acc += w * x[None, :]
    tl.store(Y + (r * SLOTS + slot) * N + cols, tl.sum(acc, 1), cols < N)


@triton.jit
def _mv_group_q2_soa(
    X,
    W,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
):
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        dmh = W.to(tl.pointer_type(tl.float16))
        sc = W + DM_BYTES
        qs = W + DM_BYTES + SC_BYTES
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values_q2_soa(
                        dmh,
                        sc,
                        qs,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                    )
                    x0 = tl.load(X + p0 * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1 * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2 * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3 * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _mv_iq2_soa(
    X,
    W,
    GRID,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    DQ_BYTES: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    dq = W.to(tl.pointer_type(tl.float16))
    qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values_iq2_soa(
            dq,
            qs,
            GRID,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
        )
        acc += w * x[None, :]
    tl.store(Y + (r * SLOTS + slot) * N + cols, tl.sum(acc, 1), cols < N)


@triton.jit
def _mv_group_iq2_soa(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        dq = W.to(tl.pointer_type(tl.float16))
        qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values_iq2_soa(
                        dq,
                        qs,
                        GRID,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                    )
                    x0 = tl.load(X + p0.to(tl.int64) * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1.to(tl.int64) * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2.to(tl.int64) * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3.to(tl.int64) * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _mv_group(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Decode-width grouped GEMV: `_mv_rows` math, expert-offset weights, Plan members."""
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        # Waves of 4 match dense `_mv_rows`: one weight decode feeds four [BN,BK] lanes.
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values(
                        W,
                        GRID,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                        FORMAT,
                    )
                    x0 = tl.load(X + p0.to(tl.int64) * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1.to(tl.int64) * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2.to(tl.int64) * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3.to(tl.int64) * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _values_iq2_soa(DQ, QS, GRID, row, k, mask, K: tl.constexpr):
    """IQ2 from SoA: DQ half scales, QS 8×u64 code groups per 256-K block."""
    blk = (row * (K // 256) + k // 256).to(tl.int64)
    j = k % 256
    d = tl.load(DQ + blk, mask, 0).to(tl.float32)
    word = tl.load(QS + blk * 8 + j // 32, mask, 0)
    # Each u64 holds 4 grid bytes + 4 bytes of packed signs/scale-nibble.
    grid = ((word >> (8 * ((j % 32) // 8))) & 255).to(tl.int32)
    # signs live in bytes 4..7 as a little-endian uint32 (matches raw block).
    bits = (word >> 32).to(tl.uint32)
    signs = (bits >> (7 * ((j % 32) // 8))) & 127
    parity = signs ^ (signs >> 4)
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs |= (parity & 1) << 7
    mag = tl.load(GRID + grid * 8 + j % 8, mask, 0).to(tl.float32)
    sign = tl.where(((signs >> (j % 8)) & 1) != 0, -1.0, 1.0)
    d = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.25
    return d * mag * sign


@triton.jit
def _values_q2_soa(DMH, SC, QS, row, k, mask, K: tl.constexpr):
    """Q2_K from SoA: DMH float16[2] per block, SC 16 bytes, QS 64 bytes."""
    blk = (row * (K // 256) + k // 256).to(tl.int64)
    j = k % 256
    d = tl.load(DMH + blk * 2, mask, 0).to(tl.float32)
    m = tl.load(DMH + blk * 2 + 1, mask, 0).to(tl.float32)
    sc = tl.load(SC + blk * 16 + j // 16, mask, 0).to(tl.int32)
    bits = tl.load(QS + blk * 64 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
    q = (bits >> (2 * ((j % 128) // 32))) & 3
    return (d * (sc & 15)) * q.to(tl.float32) - m * (sc >> 4)


@triton.jit
def _group_mm_iq2_soa(
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
        inputs = pairs if INPUT_SLOTS else pairs // SLOTS
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        dq = W.to(tl.pointer_type(tl.float16))
        qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values_iq2_soa(
                dq,
                qs,
                GRID,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


@triton.jit
def _group_mm_q2_soa(
    X,
    W,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    INPUT_SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        dmh = W.to(tl.pointer_type(tl.float16))
        sc = W + DM_BYTES
        qs = W + DM_BYTES + SC_BYTES
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values_q2_soa(
                dmh,
                sc,
                qs,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


@triton.jit
def _group_mm(
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values(
                W,
                GRID,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
                FORMAT,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))
