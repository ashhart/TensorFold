"""The NVFP4 checkpoint's bf16 linears on a row-invariant Triton matmul, and a 4-bit copy for draft-only weights."""

from __future__ import annotations

import os
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
    """A BF16 matrix [n, k] as the checkpoint stores it, with an e4m3 copy decode can read instead.

    A round verifies a handful of rows, so a dense face's cost is its bytes: every layer of a checkpoint whose
    non-expert linears are stored bf16 re-reads them whole however few rows the round checks. Half the bytes
    and the lane matmul's own tiles make the e4m3 copy of the same weight several times faster at those row
    counts, which is why ``make_b16`` can build one (``TENSORFOLD_FACES_FP8=1``) and ``matmul`` prefers it for
    decode-sized calls. Prompts keep the stored rows: they are the shape ``Fp8Linear`` is a copy *for*, and a
    row's bits must not depend on how the work was scheduled.
    """

    weight: torch.Tensor      # [n, k] bf16, contiguous
    n: int
    k: int
    rows8: object = None      # tensorfold.cuda.nvfp4.linear.Fp8Linear, or None

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size() + (self.rows8.nbytes() if self.rows8 else 0)


def faces_8bit() -> str:
    """Which BF16 faces also get an 8-bit copy, from ``TENSORFOLD_FACES_FP8`` across a load.

    ``""``: none. ``"1"``: the projections ``weights.face`` builds - the DeltaNet and attention linears, which
    are 110 of a layer's 147.6 MiB. ``"all"``: every BF16 face too (router, hyper-connections, shared expert),
    which measured a 28% cheaper round but 48% of drafts accepted against 64%: those small faces steer which
    experts run and how the streams mix, so a coarser copy of them costs the MTP head more than it saves.
    """

    return os.environ.get("TENSORFOLD_FACES_FP8", "").strip().lower()


DECODE_FP8 = True             # whether a round reads that copy; a probe flips it, a server leaves it on


def make_b16(weight: torch.Tensor, *, copy: bool | None = None) -> B16:
    """A B16 face, with an e4m3 copy when ``copy`` says so (None: when ``TENSORFOLD_FACES_FP8`` is set).

    ``copy=True`` marks a face whose stored rows are pure weight traffic (``weights.face``: the DeltaNet and
    attention linears, 74% of a round's BF16 bytes). ``copy=False`` marks the rows a draft chain is *compared
    against*: the MTP drafts' head is a copy of the lm_head's own rows, so a coarser copy of them here would
    leave the two paths scoring different weights and the chain would accept nothing.
    """

    w = weight.to(torch.bfloat16).contiguous()
    rows8 = None
    # ``None``: the environment decides, and either recognized value enables the eligible faces' copies.
    if (faces_8bit() in ("1", "all")) if copy is None else copy:
        from tensorfold.cuda.nvfp4.linear import Mx8Linear      # the lane matmul's own 8-bit face

        rows8 = Mx8Linear.from_bf16(w)
    return B16(w, int(w.shape[0]), int(w.shape[1]), rows8)


def split_k(n: int, k: int, target: int = 160, bk: int = BK) -> int:
    """K slices by the weight's shape alone: a power of two with whole BK blocks a slice."""

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
        """x [M, K] @ W.T -> [M, N]: K in BK steps in order, a tensor-core dot each, fp32 accumulators."""

        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        n_ok = rn < N
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK               # whole BK blocks a slice (split_k picks SK for that)
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for i in range(NB):
            k0 = (pid_s * NB + i) * BK
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        out_mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc if F32 else acc.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)

    @triton.jit
    def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
        """The K slices summed in slice order, one fp32 add a slice, in one launch for either output face."""

        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < total
        acc = tl.load(PART + offs, mask=ok, other=0.0)
        for s in tl.static_range(1, SK):
            acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
        tl.store(OUT + offs, acc if F32 else acc.to(tl.bfloat16), mask=ok)


def matmul(x: torch.Tensor, b: B16, *, out: torch.Tensor | None = None, f32: bool = False,
           sk: int | None = None, num_warps: int = 4, num_stages: int = 3,
           block_n: int = BN, bk: int = BK, prefill: bool = True) -> torch.Tensor:
    """x [M, K] bf16 @ b.T -> [M, N] bf16 (or fp32 sums), K slices summed in slice order.

    ``prefill`` says which step is asking, because that is what decides whether a face with an e4m3 copy reads
    it: a round verifies a handful of rows, so its reads are the weight's bytes, while a prompt's rows are many
    and the stored rows stay the cheaper read. Row count would be the wrong test - a prompt arrives in chunks,
    and a row's bits must not depend on how the work was scheduled.
    """

    if not HAS_TRITON:
        raise RuntimeError("the BF16 matmul needs Triton (the CUDA engine's environment)")
    m, k = x.shape
    if k != b.k or x.stride(1) != 1:
        raise ValueError(f"b16 matmul: x {tuple(x.shape)} does not match K={b.k}")
    if k % bk:
        raise ValueError(f"b16 matmul: K {k} is not a multiple of the K block {bk}")
    if DECODE_FP8 and b.rows8 is not None and not f32 and not prefill:
        # Decode: the e4m3 copy of this weight, half the bytes and the lane matmul's tiles. An fp32 face keeps
        # the row-invariant path (its whole point is slice-order sums in fp32), and so does a prompt.
        return b.rows8(x, out=out)
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
        total = m * b.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def quantize4(w: torch.Tensor, chunk: int = 8192, out: str = "q4"):
    """bf16 (N, K) -> MLX affine 4-bit in groups of 32, for weights that only draft (the MTP head's lm_head copy)."""

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


def b16_from_rows(rows: torch.Tensor, *, copy: bool | None = None) -> "_Routed":
    """A [n, k] bf16 matrix with the ``qmm.matmul`` face (a ``kernel`` tag)."""

    b = make_b16(rows, copy=copy)
    return _Routed(b)


class _Routed:
    """A bf16 matrix tagged ``kernel == 'b16'`` so ``forward._mm`` routes it here; ``stack`` joins rows."""

    kernel = "b16"

    def __init__(self, b: B16) -> None:
        self.b = b
        self.n, self.k = b.n, b.k

    @property
    def weight(self) -> torch.Tensor:
        return self.b.weight

    @property
    def rows8(self):
        """The e4m3 copy of the stored rows, when the face has one (``TENSORFOLD_FACES_FP8``)."""

        return self.b.rows8

    def nbytes(self) -> int:
        return self.b.nbytes()


def stack_b16(parts: list) -> _Routed:
    """Rows of several _Routed of the same K stacked in order (the hyper-connection's down + inject)."""

    rows = [p.b.weight if isinstance(p, _Routed) else p.weight for p in parts]
    return b16_from_rows(torch.cat(rows, dim=0))
