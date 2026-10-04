"""Packed GGUF decoding and dense projections; narrow rows keep serial reductions."""

from functools import cache

import torch
import triton
import triton.language as tl

from .codebook import IQ2_XXS_GRID


@cache
def codebook(device: torch.device) -> torch.Tensor:
    values = [[(g >> (8 * j)) & 255 for j in range(8)] for g in IQ2_XXS_GRID]
    return torch.tensor(values, dtype=torch.uint8, device=device).flatten()


@triton.jit
def _half(W, offset, mask):
    # Every supported block and half offset is two-byte aligned.
    return tl.load(W.to(tl.pointer_type(tl.float16)) + offset // 2, mask, 0).to(tl.float32)


@triton.jit
def _values(W, GRID, row, k, mask, K: tl.constexpr, FORMAT: tl.constexpr):
    if FORMAT == "Q8_0":
        base = (row * (K // 32) + k // 32).to(tl.int64) * 34
        d = _half(W, base, mask)
        q = tl.load(W + base + 2 + k % 32, mask, 0).to(tl.int8, bitcast=True).to(tl.float32)
        return d * q
    elif FORMAT == "Q2_K":
        base = (row * (K // 256) + k // 256).to(tl.int64) * 84
        j = k % 256
        sc = tl.load(W + base + j // 16, mask, 0).to(tl.int32)
        bits = tl.load(W + base + 16 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
        q = (bits >> (2 * ((j % 128) // 32))) & 3
        d = _half(W, base + 80, mask)
        m = _half(W, base + 82, mask)
        return (d * (sc & 15)) * q.to(tl.float32) - m * (sc >> 4)
    else:
        base = (row * (K // 256) + k // 256).to(tl.int64) * 66
        j = k % 256
        sub = base + 2 + (j // 32) * 8
        grid = tl.load(W + sub + (j % 32) // 8, mask, 0).to(tl.int32)
        words = W.to(tl.pointer_type(tl.uint16))
        bits = tl.load(words + (sub + 4) // 2, mask, 0).to(tl.uint32)
        bits |= tl.load(words + (sub + 6) // 2, mask, 0).to(tl.uint32) << 16
        signs = (bits >> (7 * ((j % 32) // 8))) & 127
        parity = signs ^ (signs >> 4)
        parity ^= parity >> 2
        parity ^= parity >> 1
        signs |= (parity & 1) << 7
        mag = tl.load(GRID + grid * 8 + j % 8, mask, 0).to(tl.float32)
        sign = tl.where(((signs >> (j % 8)) & 1) != 0, -1.0, 1.0)
        d = _half(W, base, mask) * (0.5 + (bits >> 28).to(tl.float32)) * 0.25
        return d * mag * sign


@triton.jit
def _unpack(W, GRID, OUT, TOTAL: tl.constexpr, K: tl.constexpr, FORMAT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = _values(W, GRID, i // K, i % K, i < TOTAL, K, FORMAT)
    tl.store(OUT + i, v, i < TOTAL)


@triton.jit
def _mv(
    X,
    W,
    GRID,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values(
            W,
            GRID,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
            FORMAT,
        )
        acc += w * x[None, :]
    y = tl.sum(acc, 1)
    tl.store(Y + (r * SLOTS + slot) * N + cols, y, cols < N)


@triton.jit
def _mv_rows(
    X,
    W,
    G,
    Y,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    F: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUPS: tl.constexpr = 1,
):
    r = tl.program_id(0) * BM
    group = tl.program_id(2)
    width = N // GROUPS
    n = group * width + tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a0 = tl.zeros((BN, BK), tl.float32)
    a1 = tl.zeros((BN, BK), tl.float32)
    if BM == 4:
        a2 = tl.zeros((BN, BK), tl.float32)
        a3 = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        w = _values(W, G, n[:, None], k[None, :], (n[:, None] < (group + 1) * width) & (k[None, :] < K), K, F)
        x0 = tl.load(X + r * GROUPS * K + group * K + k, (r < R) & (k < K), 0).to(tl.float32)
        x1 = tl.load(X + (r + 1) * GROUPS * K + group * K + k, (r + 1 < R) & (k < K), 0).to(tl.float32)
        a0 += w * x0[None, :]
        a1 += w * x1[None, :]
        if BM == 4:
            x2 = tl.load(X + (r + 2) * GROUPS * K + group * K + k, (r + 2 < R) & (k < K), 0).to(tl.float32)
            x3 = tl.load(X + (r + 3) * GROUPS * K + group * K + k, (r + 3 < R) & (k < K), 0).to(tl.float32)
            a2 += w * x2[None, :]
            a3 += w * x3[None, :]
    tl.store(Y + r * N + n, tl.sum(a0, 1), (r < R) & (n < (group + 1) * width))
    tl.store(Y + (r + 1) * N + n, tl.sum(a1, 1), (r + 1 < R) & (n < (group + 1) * width))
    if BM == 4:
        tl.store(Y + (r + 2) * N + n, tl.sum(a2, 1), (r + 2 < R) & (n < (group + 1) * width))
        tl.store(Y + (r + 3) * N + n, tl.sum(a3, 1), (r + 3 < R) & (n < (group + 1) * width))


@triton.jit
def _mm(
    X,
    W,
    GRID,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUPS: tl.constexpr = 1,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    group = tl.program_id(2)
    width = N // GROUPS
    n = group * width + tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ki = start * BK + k
        a = tl.load(X + m[:, None] * GROUPS * K + group * K + ki[None, :], (m[:, None] < M) & (ki[None, :] < K), 0)
        b = _values(
            W, GRID, n[None, :], ki[:, None], (n[None, :] < (group + 1) * width) & (ki[:, None] < K), K, FORMAT
        ).to(tl.bfloat16)
        acc = tl.dot(a.to(tl.bfloat16), b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < (group + 1) * width))


@triton.jit
def _float_mv(X, W, Y, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * K + k, k < K, 0).to(tl.float32)
        w = tl.load(W + n[:, None].to(tl.int64) * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc += x[None, :] * w
    tl.store(Y + r * N + n, tl.sum(acc, 1), n < N)
