"""Row-invariant affine 4-bit projections preserve MLX packing and sum fp32 group dot products and bias corrections in shape-fixed slice order, independent of row count or tile bucket."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BN = 64                   # output columns per program


def bucket(m: int) -> int:
    """The row tile; past 128 rows, 128-row tiles side by side (a row's sum never depends on the tile)."""

    if m < 1:
        raise ValueError("lane matmul takes at least one row")
    for b in (16, 32, 64):
        if m <= b:
            return b
    return 128


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count."""

    tiles = -(-n // BN)
    groups = k // 64
    sk = 1
    while sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


@triton.jit
def _group_sums(X, XS, K: tl.constexpr, KG: tl.constexpr, GB: tl.constexpr):
    """XS[m, g] = fp32 sum of x[m, 64g:64g+64], one program per (row, block of GB groups)."""

    m = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 64)
    ok = g < KG
    x = tl.load(X + m * K + g[:, None] * 64 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


@triton.jit
def _qmm(X, XS, W, S, B, OUT, PART, M,
         N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr):
    KG: tl.constexpr = K // 64
    PER: tl.constexpr = KG // SK
    K8: tl.constexpr = K // 8
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER):
        g = pid_s * PER + i
        x = tl.load(X + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        words = tl.load(W + rn[:, None] * K8 + (g * 8 + rw)[None, :], mask=n_ok[:, None], other=0)
        q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(q, (BLOCK_N, 64)).to(tl.bfloat16)
        p = tl.dot(x, tl.trans(q))
        s = tl.load(S + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
        b = tl.load(B + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
        xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
        acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


def group_sums(x: torch.Tensor) -> torch.Tensor:
    """(M, K) bf16 -> (M, K/64) fp32 sums of each 64-input group."""

    m, k = x.shape
    kg = k // 64
    xs = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    gb = 16
    _group_sums[(m, triton.cdiv(kg, gb))](x, xs, K=k, KG=kg, GB=gb, num_warps=2)
    return xs


def lane_matmul(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                xs: torch.Tensor | None = None, sk: int | None = None,
                bm: int | None = None) -> torch.Tensor:
    """x (M, K) bf16 times the packed 4-bit ``weight`` (N, K/8) transposed -> (M, N) bf16."""

    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError("lane_matmul: x must be a 2-D bf16 tensor")
    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] * 8 != k or k % 64:
        raise ValueError(f"lane_matmul: weight {tuple(weight.shape)} does not match K={k}")
    x = x.contiguous()
    bm = bucket(m) if bm is None else int(bm)
    if bm not in (16, 32, 64, 128):
        raise ValueError("lane_matmul: row tile must be 16, 32, 64 or 128")
    if xs is None:
        xs = group_sums(x)
    sk = int(sk) if sk else split_k(n, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, BN), sk)
    warps, stages = (4 if bm <= 32 else 8), 3
    from tensorfold.cuda.rocm import HIP

    if HIP and bm > 64:                              # gfx1151: 64-row tiles, one stage (3.3x faster; a row's bits never
        bm, stages = 64, 1                           # depend on the tile, see ``bucket``)
        grid = (triton.cdiv(m, bm), triton.cdiv(n, BN), sk)
    elif HIP:
        stages = 1
    _qmm[grid](x, xs, weight, scales, biases, out, part, m, N=n, K=k, SK=sk, BM=bm, BLOCK_N=BN,
               num_warps=warps, num_stages=stages)
    if sk > 1:
        total = m * n
        block = 1024
        _reduce[(triton.cdiv(total, block),)](part, out, total, SK=sk, BLOCK=block, num_warps=4)
    return out


def dequantize(weight: torch.Tensor, scales: torch.Tensor | None, biases: torch.Tensor | None,
               bits: int = 4, gs: int = 64) -> torch.Tensor:
    """Diagnostic reference only; serving kernels read packed words directly."""

    if scales is None:
        return weight.float()
    n = weight.shape[0]
    k = scales.shape[1] * gs
    words = weight.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    offsets = torch.arange(k, device=weight.device, dtype=torch.int64) * bits
    index, shifts = offsets // 32, offsets % 32
    padded = torch.cat([words, words.new_zeros((n, 1))], dim=1)
    low, high = padded[:, index], padded[:, index + 1]
    q = ((low >> shifts) | torch.where(shifts + bits > 32, high << (32 - shifts), 0)) & ((1 << bits) - 1)
    q = q.to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(gs, dim=1)
    b = biases.to(torch.float32).repeat_interleave(gs, dim=1)
    return q * s + b
