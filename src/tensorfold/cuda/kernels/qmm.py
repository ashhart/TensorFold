"""4-bit lane matmul: rows run the same groups and K slices at any row count, so no row affects another."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qmm_v3", sources=[str(here / "qmm.cpp"), str(here / "qmm.cu"),
                                                   str(here / "qmm_prefill.cu"), str(here / "qmm_prefill8.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@dataclass
class Q4:
    """Packed (n, k): int32 words [n/64][k/gs][8][32][gs/32], bf16 scales and biases (k/gs, n); n padded to 128."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    gs: int

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


# nibble slot p of a lane's word holds input 32 v + 2 (lane % 4) + OFFSETS[p] of the group (see ``pair`` in qmm.cu)
OFFSETS = (0, 8, 16, 24, 1, 9, 17, 25)


def _offsets(gs: int, device) -> torch.Tensor:
    v = torch.arange(gs // 32, device=device)[:, None, None]
    c = torch.arange(4, device=device)[None, :, None]
    return 32 * v + 2 * c + torch.tensor(OFFSETS, device=device)[None, None, :]           # (V, 4, 8)


def _to_int32(v: torch.Tensor) -> torch.Tensor:
    """Unsigned 32-bit values held in int64 -> the int32 with the same bits."""

    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)


def pack(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int, chunk: int = 4096) -> Q4:
    """MLX (n, k/8) words, (n, k/gs) scales and biases -> ``Q4``; n padded to 128 with zeros so tiles stay inside."""

    words = weight.view(torch.int32) if weight.dtype != torch.int32 else weight
    n, k8 = words.shape
    k, kg, npad = k8 * 8, k8 * 8 // gs, -(-n // 128) * 128
    dev = words.device
    out = torch.empty((npad // 64, kg, 8, 32, gs // 32), dtype=torch.int32, device=dev)
    offs = _offsets(gs, dev)
    shifts = torch.arange(8, device=dev, dtype=torch.int32) * 4
    wide = shifts.to(torch.int64)
    for start in range(0, npad, chunk):
        stop = min(start + chunk, npad)
        block = torch.zeros((stop - start, k8), dtype=torch.int32, device=dev)
        if start < n:
            block[:min(stop, n) - start] = words[start:min(stop, n)]
        q = ((block[:, :, None] >> shifts) & 0xF).reshape(stop - start, kg, gs)            # (cols, kg, gs)
        q = q.reshape((stop - start) // 64, 8, 8, kg, gs)                                   # (T, j, r, kg, gs)
        picked = q[..., offs]                                                                # (T, j, r, kg, V, 4, 8)
        packed = _to_int32((picked.to(torch.int64) << wide).sum(-1))                        # (T, j, r, kg, V, 4)
        out[start // 64:stop // 64] = packed.permute(0, 3, 1, 2, 5, 4).reshape(-1, kg, 8, 32, gs // 32)
    pad = npad - n

    def major(t: torch.Tensor) -> torch.Tensor:
        t = t.t().contiguous()
        return torch.cat([t, t.new_zeros((kg, pad))], dim=1).contiguous() if pad else t

    return Q4(out, major(scales), major(biases), n, k, gs)


def unpack(q: Q4, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (n, k/8) int32 words, (n, k/gs) scales and biases, ``chunk`` tiles at a time."""

    t, kg, _, _, v = q.weight.shape
    dev = q.weight.device
    shifts = torch.arange(8, device=dev, dtype=torch.int32) * 4
    wide = shifts.to(torch.int64)
    offs = _offsets(q.gs, dev)
    words = torch.empty((t * 64, q.k // 8), dtype=torch.int32, device=dev)
    for a in range(0, t, chunk):
        b = min(a + chunk, t)
        w = q.weight[a:b].reshape(b - a, kg, 8, 8, 4, v).permute(0, 2, 3, 1, 5, 4)          # (T, j, r, kg, V, 4)
        nib = (w[..., None] >> shifts) & 0xF                                                  # (T, j, r, kg, V, 4, 8)
        qv = torch.zeros((b - a, 8, 8, kg, q.gs), dtype=torch.int32, device=dev)
        qv[..., offs] = nib
        qv = qv.reshape((b - a) * 64, q.k // 8, 8)
        words[a * 64:b * 64] = _to_int32((qv.to(torch.int64) << wide).sum(-1))
    return words[:q.n].contiguous(), q.scales[:, :q.n].t().contiguous(), q.biases[:, :q.n].t().contiguous()


def split_k(n: int, k: int, gs: int = 64, target: int = 192) -> int:
    """K slices for an (n, k) weight: a function of the shape only (never of the row count)."""

    tiles, groups, sk = -(-n // 64), k // gs, 1
    while sk < 8 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def bucket(m: int) -> int:
    """The row tile: 16 or 32 rows, else 64-row tiles side by side; tiles never change bits."""

    if m < 1:
        raise ValueError("the lane matmul takes at least one row")
    return 16 if m <= 16 else 32 if m <= 32 else 64


@triton.jit
def _group_sums(X, XS, ldx, KG: tl.constexpr, GS: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    ok = g < KG
    x = tl.load(X + m * ldx + g[:, None] * GS + tl.arange(0, GS)[None, :], mask=ok[:, None], other=0.0)
    tl.store(XS + m * KG + g, tl.sum(x.to(tl.float32), axis=1), mask=ok)


def group_sums(x: torch.Tensor, gs: int = 64) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/gs) fp32 sums of each group's inputs."""

    m, k = x.shape
    xs = torch.empty((m, k // gs), dtype=torch.float32, device=x.device)
    _group_sums[(m, triton.cdiv(k // gs, 16))](x, xs, x.stride(0), KG=k // gs, GS=gs, GB=16, num_warps=2)
    return xs


def matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, sk: int | None = None, f32: bool = False,
           out: torch.Tensor | None = None, part: torch.Tensor | None = None, reduce: bool = True) -> torch.Tensor:
    """x @ q.T as (M, n) bf16, or unrounded fp32 with ``f32``; ``reduce=False`` returns K slices to add in order."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != q.k:
        raise ValueError(f"matmul: x must be (M, {q.k}) bf16")
    if x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)     # cp.async reads rows in 16-byte pieces
    m = x.shape[0]
    if xs is None:
        xs = group_sums(x, q.gs)
    sk = sk or split_k(q.n, q.k, q.gs)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if sk > 1 and not reduce and part is None:
        part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    _ext().qmm(x, xs, q.weight, q.scales, q.biases, out, part, q.n, sk, q.gs, bucket(m), f32, reduce)
    return out if sk == 1 or reduce else part.reshape(-1)[:sk * m * q.n].view(sk, m, q.n)


def prefill_matmul(x: torch.Tensor, q: Q4, *, f32: bool = False, tile: int = 0,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill: weights rounded once to bf16, one fp32 chain over K; any chunking gives the same bits, not decode's."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != q.k:
        raise ValueError(f"prefill_matmul: x must be (M, {q.k}) bf16")
    if x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    if out is None:
        out = torch.empty((x.shape[0], q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    _ext().qmm_prefill(x, q.weight, q.scales, q.biases, out, q.n, q.gs, f32, tile)
    return out


@triton.jit
def _quantize_rows(X, X8, XS, A, ldx, K: tl.constexpr, GS: tl.constexpr, BK: tl.constexpr):
    """Row r: a = max|x| / 448, x8 = e4m3(x / a) in the weights' fragment order, xs = each group's input sum / a."""

    row = tl.program_id(0)
    m = tl.arange(0, BK)
    j = m % 4
    src = (m // 32) * 32 + ((m % 32) // 16) * 16 + ((m % 16) // 4) * 2 + (j % 2) + (j // 2) * 8
    amax = tl.zeros((BK,), tl.float32)
    for k0 in range(0, K, BK):
        amax = tl.maximum(amax, tl.abs(tl.load(X + row * ldx + k0 + m, mask=k0 + m < K, other=0.0).to(tl.float32)))
    top = tl.max(amax, 0)
    a = tl.where(top > 0.0, top / 448.0, 1.0)
    for k0 in range(0, K, BK):
        x = tl.load(X + row * ldx + k0 + src, mask=k0 + m < K, other=0.0).to(tl.float32)
        q = (x / a).to(tl.float8e4nv)
        tl.store(X8 + row * K + k0 + m, q.to(tl.uint8, bitcast=True), mask=k0 + m < K)
        g = tl.sum(tl.reshape(x, (BK // GS, GS)), 1) / a
        gi = k0 // GS + tl.arange(0, BK // GS)
        tl.store(XS + row * (K // GS) + gi, g.to(tl.bfloat16), mask=gi < K // GS)
    tl.store(A + row, a)


def quantize_rows(x: torch.Tensor, gs: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs for ``prefill_matmul8``: e4m3 bytes in fragment order, group sums over the row scale, the row scales."""

    m, k = x.shape
    if k % gs or k % 32:
        raise ValueError("quantize_rows takes K a multiple of the group and of 32")
    x8 = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    xs = torch.empty((m, k // gs), dtype=torch.bfloat16, device=x.device)
    a = torch.empty((m,), dtype=torch.float32, device=x.device)
    _quantize_rows[(m,)](x, x8, xs, a, x.stride(0), K=k, GS=gs, BK=256, num_warps=4)
    return x8, xs, a


def prefill_matmul8(xq: tuple[torch.Tensor, torch.Tensor, torch.Tensor], q: Q4, *, f32: bool = False, tile: int = 0,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 prefill matmul on ``quantize_rows`` output, exact e4m3 weights; a row's bits depend only on its inputs."""

    x8, xs, a = xq
    if x8.dim() != 2 or x8.shape[1] != q.k:
        raise ValueError(f"prefill_matmul8: inputs must be (M, {q.k})")
    if out is None:
        out = torch.empty((x8.shape[0], q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x8.device)
    _ext().qmm_prefill8(x8, xs, a, q.weight, q.scales, q.biases, out, q.n, q.gs, f32, tile)
    return out
