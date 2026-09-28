"""The checkpoint's BF16 linear tensors as the CUDA kernels read them (the NVFP4 checkpoint keeps every
non-expert weight in BF16: hyper-connections, DeltaNet, attention, PLE, embeddings, lm_head, the MTP head).

``B16`` is the qmm ``Q4`` face without scales: a contiguous [n, k] bf16 matrix ``matmul`` reads through
Triton the same row-invariant way (a K tile at a time, tensor cores, fp32 accum, the K slices fixed by
the shape). ``quantize4`` re-quantizes a BF16 matrix to MLX-style affine 4-bit in groups of 32 — for
weights that only draft (the MTP drafts' head copy), where drafts change speed and never the bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HAS_TRITON = True
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    HAS_TRITON = False

BN = 64
BK = 64                     # the K block a program reads a step (the split keeps slices whole blocks)
GS = 32                     # the MLX group size this module's quantize4 emits


@dataclass
class B16:
    """A BF16 matrix [n, k] as the checkpoint stores it."""

    weight: torch.Tensor      # [n, k] bf16, contiguous
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()


def make_b16(weight: torch.Tensor) -> B16:
    w = weight.to(torch.bfloat16).contiguous()
    return B16(w, int(w.shape[0]), int(w.shape[1]))


def split_k(n: int, k: int, target: int = 160, bk: int = BK) -> int:
    """K-slice count: the weight's shape only (never the row count), a power of two, and whole BK blocks a
    slice — the kernel reads one BK block a step, so a slice that did not hold whole blocks would read past
    its own K end (and past the weight)."""

    tiles = -(-n // BN)
    blocks = k // bk
    sk = 1
    while sk < 32 and tiles * sk < target and blocks % (sk * 2) == 0 and blocks // (sk * 2) >= 1:
        sk *= 2
    return sk


if HAS_TRITON:
    @triton.jit
    def _b16mm(X, W, OUT, PART, M, x_stride,
               N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
               BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        """x [M, K] bf16 @ W.T -> [M, N]: K in BK steps in order, one tensor-core dot each, fp32 acc —
        the same row-invariant chain the 4-bit kernels are."""

        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        n_ok = rn < N
        # GB10 faults on OOB addresses even when the load/store is masked (near-full VRAM).
        # Same clamp as nvfp4._fp4mm — without it graph warm / hc_mix IMA'd on Spark.
        rm_a = tl.where(m_ok, rm, 0)
        rn_a = tl.where(n_ok, rn, 0)
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK               # whole BK blocks a slice (split_k picks SK for that)
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for i in range(NB):
            k0 = (pid_s * NB + i) * BK
            x = tl.load(X + rm_a[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn_a[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        out_mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm_a[:, None] * N + rn_a[None, :], acc if F32 else acc.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(PART + (pid_s * M + rm_a[:, None]) * N + rn_a[None, :], acc, mask=out_mask)

    @triton.jit
    def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < total
        offs_a = tl.where(ok, offs, 0)
        acc = tl.load(PART + offs_a, mask=ok, other=0.0)
        for s in tl.static_range(1, SK):
            acc = acc + tl.load(PART + s * total + offs_a, mask=ok, other=0.0)
        tl.store(OUT + offs_a, acc.to(tl.bfloat16), mask=ok)


def matmul(x: torch.Tensor, b: B16, *, out: torch.Tensor | None = None, f32: bool = False,
           sk: int | None = None, num_warps: int = 4, num_stages: int = 3,
           block_n: int = BN, bk: int = BK) -> torch.Tensor:
    """x [M, K] bf16 (rows may be strided) @ b.T -> [M, N] bf16 (or fp32 sums). Slices sum in slice order.
    The kernel face ``qmm.matmul`` has (the row count never picks the split), so ``forward._mm`` routes by
    the weight's ``kernel`` tag."""

    if not HAS_TRITON:
        raise RuntimeError("the BF16 matmul needs Triton (the CUDA engine's environment)")
    m, k = x.shape
    if k != b.k or x.stride(1) != 1:
        raise ValueError(f"b16 matmul: x {tuple(x.shape)} does not match K={b.k}")
    if k % bk:
        raise ValueError(f"b16 matmul: K {k} is not a multiple of the K block {bk}")
    sk = int(sk) if sk else split_k(b.n, b.k, bk=bk)
    if out is None:
        out = torch.empty((m, b.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, b.n) or not out.is_contiguous() or (out.dtype == torch.float32) != f32:
        raise ValueError(f"b16 matmul: out {tuple(out.shape)} {out.dtype} must be a contiguous ({m}, {b.n}), "
                         f"dtype matching f32={f32}")
    part = torch.empty((sk, m, b.n), dtype=torch.float32, device=x.device) if sk > 1 else out
    bm = 128 if m > 128 else 16
    grid = (triton.cdiv(m, bm), -(-b.n // block_n), sk)
    _b16mm[grid](x, b.weight, out, part, m, x.stride(0), N=b.n, K=k, SK=sk, BM=bm,
                 BLOCK_N=block_n, BK=bk, F32=f32, num_warps=num_warps, num_stages=num_stages)
    if sk > 1:
        if f32:
            out.copy_(part[0])
            for s in range(1, sk):
                out += part[s]
        else:
            total = m * b.n
            _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out


def quantize4(w: torch.Tensor, chunk: int = 8192, out: str = "q4"):
    """bf16 (N, K) -> MLX-style affine 4-bit in groups of 32 along K (q = round((w - min) / scale)), as
    ``qmm.make_q4`` takes it (``out == "trilogue"``, the MLX layout for a row lookup) — for weights that
    only draft (the MTP drafts' head copy of lm_head, the 4-bit side of an NVFP4 checkpoint's head).
    Drafts change speed, never the bits, so the requantization is free of exactness concerns."""

    from . import qmm

    n, k = w.shape
    words, scales, biases = [], [], []
    for lo in range(0, n, chunk):
        part = w[lo:lo + chunk].float()
        g = part.reshape(-1, k // GS, GS)
        mn = g.min(dim=-1, keepdim=True).values
        mx = g.max(dim=-1, keepdim=True).values
        s = ((mx - mn) / 15.0).clamp(min=1e-8)
        q = ((g - mn) / s).round().clamp(0, 15).to(torch.uint8).reshape(-1, k)
        packed = (q[:, 1::2] << 4 | q[:, 0::2]).view(torch.uint32)          # (rows, K/8)
        words.append(packed)
        scales.append(s.reshape(-1, k // GS).to(torch.bfloat16))
        biases.append(mn.reshape(-1, k // GS).to(torch.bfloat16))
    return qmm.make_q4(torch.cat(words), torch.cat(scales), torch.cat(biases))


def b16_from_rows(rows: torch.Tensor) -> "_Routed":
    """A [n, k] bf16 matrix with the ``qmm.matmul`` face (a ``kernel`` tag): what the BF16-quantize
    helpers below return, and the shared expert's grids ride in the FP4 kernels instead."""

    b = make_b16(rows)
    return _Routed(b)


class _Routed:
    """A B16 with the qmm matmul face (``n`` / ``k`` / ``nbytes``), tagged ``kernel == "b16"``:
    ``forward._mm`` routes the dot to :mod:`bf16` on the tag, the rest of the Q4 face (none of it used
    on the engine path) raises. ``stack`` joins rows (``HC`` stacks down + inject through it)."""

    kernel = "b16"

    def __init__(self, b: B16) -> None:
        self.b = b
        self.n, self.k = b.n, b.k

    @property
    def weight(self) -> torch.Tensor:
        return self.b.weight

    def nbytes(self) -> int:
        return self.b.nbytes()


def stack_b16(parts: list) -> _Routed:
    """Rows of several _Routed of the same K stacked in order (the hyper-connection's down + inject)."""

    rows = [p.b.weight if isinstance(p, _Routed) else p.weight for p in parts]
    return b16_from_rows(torch.cat(rows, dim=0))
