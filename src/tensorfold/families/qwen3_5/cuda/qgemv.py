"""Row-invariant affine 4-bit decode matmul for GPUs without the sm_90 tiles (ROCm): one program per (row, column block, K slice).

Each row runs the same program whatever the row count, so a drafted row's bits equal its serial bits. Weights stream in
``GI`` groups of 64 inputs per step (contiguous 32*GI-byte runs per output row); nibbles unpack in registers; a group
contributes ``scale * dot + bias * sum(x)`` (MLX's affine form) and K slices add in a fixed order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row(q, X, XS, S_t, B_t, m, M, g, K: tl.constexpr, KG: tl.constexpr, GI: tl.constexpr):
    """One row's contribution of GI groups: the same tensor shapes for every row count, so the same bits."""

    ok = m < M
    x = tl.load(X + m * K + g * 64 + tl.arange(0, GI * 64), mask=ok, other=0.0).to(tl.float32)
    x = tl.reshape(x, (GI, 64))
    dot = tl.sum(q * x[None, :, :], axis=2)                                                          # (BN, GI)
    xs = tl.load(XS + m * KG + g + tl.arange(0, GI), mask=ok, other=0.0)
    return tl.sum(S_t * dot + B_t * xs[None, :], axis=1)                                             # (BN,)


@triton.jit
def _store(acc, OUT, PART, m, M, s_id, rn, n_ok, N: tl.constexpr, SK: tl.constexpr):
    ok = n_ok & (m < M)
    if SK == 1:
        tl.store(OUT + m * N + rn, acc.to(tl.bfloat16), mask=ok)
    else:
        tl.store(PART + (s_id * M + m) * N + rn, acc, mask=ok)


@triton.jit
def _gemv(X, XS, W, S, B, OUT, PART, M, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr,
          BN: tl.constexpr, GI: tl.constexpr, R: tl.constexpr):
    """R rows share each unpacked weight tile; every row runs ``_row`` on identical shapes (R is 1, 2 or 4)."""

    KG: tl.constexpr = K // 64
    K8: tl.constexpr = K // 8
    PER: tl.constexpr = KG // SK                   # groups in this program's K slice
    m0 = tl.program_id(0) * R
    pid_n = tl.program_id(1)
    s_id = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    n_ok = rn < N
    words = tl.arange(0, GI * 8)
    gi = tl.arange(0, GI)
    shifts = tl.arange(0, 8) * 4
    acc0 = tl.zeros((BN,), dtype=tl.float32)
    acc1 = tl.zeros((BN,), dtype=tl.float32)
    acc2 = tl.zeros((BN,), dtype=tl.float32)
    acc3 = tl.zeros((BN,), dtype=tl.float32)
    g0 = s_id * PER
    for step in range(PER // GI):
        g = g0 + step * GI
        w = tl.load(W + rn[:, None] * K8 + g * 8 + words[None, :], mask=n_ok[:, None], other=0)   # (BN, GI*8) int32
        q = (w[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(q.to(tl.float32), (BN, GI, 64))
        S_t = tl.load(S + rn[:, None] * KG + g + gi[None, :], mask=n_ok[:, None], other=0.0).to(tl.float32)
        B_t = tl.load(B + rn[:, None] * KG + g + gi[None, :], mask=n_ok[:, None], other=0.0).to(tl.float32)
        acc0 += _row(q, X, XS, S_t, B_t, m0, M, g, K, KG, GI)
        if R > 1:
            acc1 += _row(q, X, XS, S_t, B_t, m0 + 1, M, g, K, KG, GI)
        if R > 2:
            acc2 += _row(q, X, XS, S_t, B_t, m0 + 2, M, g, K, KG, GI)
            acc3 += _row(q, X, XS, S_t, B_t, m0 + 3, M, g, K, KG, GI)
    _store(acc0, OUT, PART, m0, M, s_id, rn, n_ok, N, SK)
    if R > 1:
        _store(acc1, OUT, PART, m0 + 1, M, s_id, rn, n_ok, N, SK)
    if R > 2:
        _store(acc2, OUT, PART, m0 + 2, M, s_id, rn, n_ok, N, SK)
        _store(acc3, OUT, PART, m0 + 3, M, s_id, rn, n_ok, N, SK)


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SK):                            # slices add in order: fixed by the shape
        acc += tl.load(PART + s * total + i, mask=ok, other=0.0)
    tl.store(OUT + i, acc.to(tl.bfloat16), mask=ok)


@triton.jit
def _group_sums(X, XS, K: tl.constexpr, KG: tl.constexpr):
    m = tl.program_id(0)
    g = tl.program_id(1)
    x = tl.load(X + m * K + g * 64 + tl.arange(0, 64)).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=0))


def group_sums(x: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    xs = torch.empty((m, k // 64), dtype=torch.float32, device=x.device)
    _group_sums[(m, k // 64)](x, xs, K=k, KG=k // 64, num_warps=1)
    return xs


BN = 8                   # output columns a program (swept on gfx1151: 8 columns, 8 groups a step, one warp)
GI = 8
WARPS = 1


def group_step(k: int) -> int:
    """Groups a step: the swept 8 where K allows (the 27B's K is 5120, 6144 or 17408), else the largest that divides."""

    kg = k // 64
    for gi in (GI, 4, 2, 1):
        if kg % gi == 0:
            return gi
    return 1


def split_k(n: int, k: int) -> int:
    """K slices from the shape alone (never the row count): enough programs to fill the GPU for narrow outputs."""

    kg = k // 64
    gi = group_step(k)
    tiles = -(-n // BN)
    sk = 1
    while sk < 8 and tiles * sk < 512 and kg % (sk * 2 * gi) == 0:
        sk *= 2
    return sk


def rows_per_program(m: int) -> int:
    """Rows sharing a weight tile: one for a serial step, four once drafts verify several."""

    return 1 if m == 1 else (2 if m == 2 else 4)


def gemv(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
         sk: int | None = None, xs: torch.Tensor | None = None, r: int | None = None) -> torch.Tensor:
    """x (M, K) bf16 times the MLX-packed 4-bit ``weight`` (N, K/8) transposed -> (M, N) bf16; any M, same row bits."""

    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError("gemv: x must be a 2-D bf16 tensor")
    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] * 8 != k or k % 64:
        raise ValueError(f"gemv: weight {tuple(weight.shape)} does not match K={k}")
    gi = group_step(k)
    x = x.contiguous()
    sk = int(sk) if sk else split_k(n, k)
    if (k // 64) % (sk * gi):
        raise ValueError(f"gemv: {sk} K slices do not divide {k // 64} groups in steps of {gi}")
    xs = group_sums(x) if xs is None else xs
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    r = rows_per_program(m) if r is None else int(r)
    _gemv[(triton.cdiv(m, r), triton.cdiv(n, BN), sk)](x, xs, weight, scales, biases, out, part, m, N=n, K=k, SK=sk,
                                                       BN=BN, GI=gi, R=r, num_warps=WARPS)
    if sk > 1:
        total = m * n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out




# ---- matrix-core variant: a fixed 16-row tile (padded), WMMA dots; a row's bits never depend on the row count ----

@triton.jit
def _gemm16(X, XS, W, S, B, OUT, PART, M, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr,
            BN: tl.constexpr, GI: tl.constexpr):
    KG: tl.constexpr = K // 64
    K8: tl.constexpr = K // 8
    PER: tl.constexpr = KG // SK
    rm = tl.program_id(0) * 16 + tl.arange(0, 16)
    pid_n = tl.program_id(1)
    s_id = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    m_ok = rm < M
    n_ok = rn < N
    words = tl.arange(0, GI * 8)
    gi = tl.arange(0, GI)
    shifts = tl.arange(0, 8) * 4
    acc = tl.zeros((16, BN), dtype=tl.float32)
    g0 = s_id * PER
    for step in range(PER // GI):
        g = g0 + step * GI
        w = tl.load(W + rn[:, None] * K8 + g * 8 + words[None, :], mask=n_ok[:, None], other=0)   # (BN, GI*8)
        q = ((w[:, :, None] >> shifts[None, None, :]) & 0xF).to(tl.bfloat16)                        # exact 0..15
        q = tl.permute(tl.reshape(q, (BN, GI, 64)), (1, 2, 0))                                      # (GI, 64, BN)
        x = tl.load(X + rm[:, None] * K + g * 64 + tl.arange(0, GI * 64)[None, :], mask=m_ok[:, None], other=0.0)
        x = tl.permute(tl.reshape(x, (16, GI, 64)), (1, 0, 2))                                      # (GI, 16, 64)
        dot = tl.dot(x, q, out_dtype=tl.float32)                                                     # (GI, 16, BN)
        S_t = tl.load(S + rn[None, :] * KG + g + gi[:, None], mask=n_ok[None, :], other=0.0).to(tl.float32)   # (GI, BN)
        B_t = tl.load(B + rn[None, :] * KG + g + gi[:, None], mask=n_ok[None, :], other=0.0).to(tl.float32)
        xs = tl.load(XS + rm[None, :] * KG + g + gi[:, None], mask=m_ok[None, :], other=0.0)            # (GI, 16)
        acc += tl.sum(S_t[:, None, :] * dot + B_t[:, None, :] * xs[:, :, None], axis=0)
    ok = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=ok)
    else:
        tl.store(PART + (s_id * M + rm[:, None]) * N + rn[None, :], acc, mask=ok)


def gemm16(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
           sk: int = 1, bn: int = 16, gi: int = 4, warps: int = 1, xs: torch.Tensor | None = None) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    x = x.contiguous()
    xs = group_sums(x) if xs is None else xs
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _gemm16[(triton.cdiv(m, 16), triton.cdiv(n, bn), sk)](x, xs, weight, scales, biases, out, part, m, N=n, K=k,
                                                          SK=sk, BN=bn, GI=gi, num_warps=warps)
    if sk > 1:
        total = m * n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out


def kernel() -> str:
    """One decode kernel a server run, so a request with drafts off compares against the same arithmetic:
    ``gemv`` (fastest single rows, serial serving) or ``gemm16`` (16 rows for the price of one, drafted serving).
    Set TF_ROCM_DECODE_KERNEL; the default follows whether drafts are on (``configure``)."""

    import os

    return os.environ.get("TF_ROCM_DECODE_KERNEL") or _default[0]


_default = ["gemv"]


def configure(drafts: bool) -> None:
    _default[0] = "gemm16" if drafts else "gemv"


GEMM16 = dict(bn=16, gi=2, warps=1)    # swept on gfx1151 (M=1 569 us, M=16 625 us on 17408x5120)


def decode_matmul(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                  sk: int | None = None, xs: torch.Tensor | None = None) -> torch.Tensor:
    if kernel() == "gemm16":
        k = x.shape[1]
        cfg = dict(GEMM16, gi=GEMM16["gi"] if (k // 64) % GEMM16["gi"] == 0 else 1)
        return gemm16(x, weight, scales, biases, sk=sk or 1, xs=xs, **cfg)
    return gemv(x, weight, scales, biases, sk=sk, xs=xs)


__all__ = ["configure", "decode_matmul", "gemm16", "gemv", "group_sums", "kernel", "group_step", "rows_per_program", "split_k"]

