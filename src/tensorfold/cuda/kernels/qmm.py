"""4- to 8-bit lane matmul: rows run the same groups and K slices at any row count, so no row affects another."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from .qmm_tiles import group_tile


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qmm_v6", sources=[str(here / "qmm.cpp"), str(here / "qmm.cu"),
                                                   str(here / "qmm_group.cu"), str(here / "qmm_prefill.cu"),
                                                   str(here / "qmm_prefill8.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@lru_cache(maxsize=None)
def _chip(device: int) -> tuple[int, int, int]:
    p = torch.cuda.get_device_properties(device)
    return p.major, p.minor, p.multi_processor_count


@lru_cache(maxsize=None)
def grouped(device: int) -> bool:
    """sm_12x runs groups of 64 through the grouped kernel: several projections of one input in a launch."""

    return torch.cuda.get_device_capability(device)[0] == 12



@dataclass
class Q4:
    """Packed (n, k): int32 words [n/64][k/gs][8][32][gs*bits/128] (5 and 6 bits: [n/64][k/gs][2*gs*bits], see
    ``_tile``), bf16 scales, biases (k/gs, n); n padded to 128."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    gs: int
    bits: int = 4

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


# slot p of a lane's word v holds input SPAN v + 2 (lane % 4) + OFFSETS[p] of the group (see ``frag`` in qmm_frag.cuh):
# a 4-bit word serves two k16 steps, an 8-bit word one, and a 5- or 6-bit lane's high word all four (slot 2 kt + h)
OFFSETS = {4: (0, 8, 16, 24, 1, 9, 17, 25), 8: (0, 8, 1, 9)}
HIGH = (0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57)


def _to_int32(v: torch.Tensor) -> torch.Tensor:
    """Unsigned 32-bit values held in int64 -> the int32 with the same bits."""

    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)


def _slots(gs: int, device, offsets: tuple) -> torch.Tensor:
    span = 4 * len(offsets)
    v = torch.arange(gs // span, device=device)[:, None, None]
    c = torch.arange(4, device=device)[None, :, None]
    return span * v + 2 * c + torch.tensor(offsets, device=device)[None, None, :]          # (V, 4, slots)


def _lanes(q: torch.Tensor, width: int, offsets: tuple) -> torch.Tensor:
    """Codes (cols, kg, gs) -> (cols / 64, kg, 8, 32, V) words, slot p ``width`` bits at ``width * p``."""

    cols, kg, gs = q.shape
    shifts = torch.arange(len(offsets), device=q.device, dtype=torch.int64) * width
    picked = q.reshape(cols // 64, 8, 8, kg, gs)[..., _slots(gs, q.device, offsets)]   # (T, j, r, kg, V, 4, slots)
    packed = _to_int32((picked.to(torch.int64) << shifts).sum(-1))                       # (T, j, r, kg, V, 4)
    return packed.permute(0, 3, 1, 2, 5, 4).reshape(cols // 64, kg, 8, 32, -1)


def _unlanes(w: torch.Tensor, width: int, offsets: tuple, gs: int) -> torch.Tensor:
    """``_lanes`` undone: (T, kg, 8, 32, V) words -> codes (T * 64, kg, gs)."""

    t, kg = w.shape[:2]
    shifts = torch.arange(len(offsets), device=w.device, dtype=torch.int32) * width
    w = w.reshape(t, kg, 8, 8, 4, -1).permute(0, 2, 3, 1, 5, 4)                           # (T, j, r, kg, V, 4)
    q = torch.zeros((t, 8, 8, kg, gs), dtype=torch.int32, device=w.device)
    q[..., _slots(gs, w.device, offsets)] = (w[..., None] >> shifts) & ((1 << width) - 1)
    return q.reshape(t * 64, kg, gs)


def _codes(words: torch.Tensor, bits: int) -> torch.Tensor:
    """MLX (n, k*bits/32) words -> (n, k) codes: a little-endian bit stream, eight codes in ``bits`` bytes."""

    n = words.shape[0]
    if bits in OFFSETS:                 # codes never straddle a word
        shifts = torch.arange(32 // bits, device=words.device, dtype=torch.int32) * bits
        return ((words[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(n, -1)
    b = words.contiguous().view(torch.uint8).reshape(n, -1, bits).to(torch.int64)
    v = (b << (8 * torch.arange(bits, device=b.device))).sum(-1)
    return ((v[..., None] >> (bits * torch.arange(8, device=b.device))) & ((1 << bits) - 1)).reshape(n, -1)


def _words(q: torch.Tensor, bits: int) -> torch.Tensor:
    """``_codes`` undone: (n, k) codes -> MLX (n, k*bits/32) int32 words."""

    n = q.shape[0]
    v = (q.reshape(n, -1, 8).to(torch.int64) << (bits * torch.arange(8, device=q.device))).sum(-1)
    b = (v[..., None] >> (8 * torch.arange(bits, device=q.device))) & 0xFF
    return b.to(torch.uint8).reshape(n, -1).view(torch.int32)


def _tile(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Codes (cols, kg, gs) -> stored tiles: 4 and 8 bits in lane order; 5 and 6 bits per (64-column tile, group)
    the nibbles in the 4-bit order, then the high bits, a word a lane and n8 tile (6) or pair of n8 tiles (5)."""

    if bits in OFFSETS:
        return _lanes(q, bits, OFFSETS[bits])
    low = _lanes(q & 15, 4, OFFSETS[4]).flatten(2)
    high = _lanes(q >> 4, bits - 4, HIGH)[..., 0].to(torch.int64)                          # (T, kg, 8, 32)
    if bits == 5:                       # even n8 tile in bytes 0 and 2, odd tile in bytes 1 and 3
        even, odd = high[:, :, 0::2], high[:, :, 1::2]
        high = (even & 0xFF) | (odd & 0xFF) << 8 | (even >> 8) << 16 | (odd >> 8) << 24
    return torch.cat([low, _to_int32(high).flatten(2)], dim=2)


def _untile(w: torch.Tensor, bits: int, gs: int) -> torch.Tensor:
    """``_tile`` undone: stored tiles -> codes (T * 64, kg, gs)."""

    if bits in OFFSETS:
        return _unlanes(w, bits, OFFSETS[bits], gs)
    t, kg = w.shape[:2]
    low = _unlanes(w[..., :512].reshape(t, kg, 8, 32, 2), 4, OFFSETS[4], gs)
    high = w[..., 512:].reshape(t, kg, -1, 32)
    if bits == 5:
        even = (high & 0xFF) | ((high >> 16) & 0xFF) << 8
        odd = ((high >> 8) & 0xFF) | ((high >> 24) & 0xFF) << 8
        high = torch.stack([even, odd], dim=3).reshape(t, kg, 8, 32)
    return low | _unlanes(high[..., None], bits - 4, HIGH, gs) << 4


def pack(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int, chunk: int = 4096, *,
         bits: int = 4) -> Q4:
    """MLX (n, k*bits/32) words, (n, k/gs) scales and biases -> ``Q4``; n padded to 128 with zeros (tiles stay in)."""

    words = weight.view(torch.int32) if weight.dtype != torch.int32 else weight
    n, kw = words.shape
    k = kw * 32 // bits
    kg, npad = k // gs, -(-n // 128) * 128
    dev = words.device
    shape = (8, 32, gs * bits // 128) if bits in OFFSETS else (2 * gs * bits,)
    out = torch.empty((npad // 64, kg, *shape), dtype=torch.int32, device=dev)
    for start in range(0, npad, chunk):
        stop = min(start + chunk, npad)
        block = torch.zeros((stop - start, kw), dtype=torch.int32, device=dev)
        if start < n:
            block[:min(stop, n) - start] = words[start:min(stop, n)]
        out[start // 64:stop // 64] = _tile(_codes(block, bits).reshape(stop - start, kg, gs), bits)
    pad = npad - n

    def major(t: torch.Tensor) -> torch.Tensor:
        t = t.t().contiguous()
        return torch.cat([t, t.new_zeros((kg, pad))], dim=1).contiguous() if pad else t

    return Q4(out, major(scales), major(biases), n, k, gs, bits)


def unpack(q: Q4, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (n, k*bits/32) int32 words, (n, k/gs) scales and biases, ``chunk`` tiles a step."""

    t = q.weight.shape[0]
    bits = getattr(q, "bits", 4)
    words = torch.empty((t * 64, q.k * bits // 32), dtype=torch.int32, device=q.weight.device)
    for a in range(0, t, chunk):
        b = min(a + chunk, t)
        words[a * 64:b * 64] = _words(_untile(q.weight[a:b], bits, q.gs).reshape((b - a) * 64, q.k), bits)
    return words[:q.n].contiguous(), q.scales[:, :q.n].t().contiguous(), q.biases[:, :q.n].t().contiguous()


def split_k(n: int, k: int, gs: int = 64, target: int = 192, bits: int = 4) -> int:
    """K slices for an (n, k) weight: a function of the shape only (never of the row count); a slice takes at least
    2 * bits groups (8-bit weights stream better in fewer, longer slices), and 5- and 6-bit weights, at least 8, stop
    at a third of the blocks."""

    tiles, groups, sk = -(-n // 64), k // gs, 1
    least, target = (8, target // 3) if bits in (5, 6) else (2 * bits, target)
    while sk < 8 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= least:
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

    bits = getattr(q, "bits", 4)

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != q.k:
        raise ValueError(f"matmul: x must be (M, {q.k}) bf16")
    if x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)     # cp.async reads rows in 16-byte pieces
    m = x.shape[0]
    if xs is None:
        xs = group_sums(x, q.gs)
    sk = sk or split_k(q.n, q.k, q.gs, bits=bits)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if q.gs == 64 and reduce and grouped(x.device.index):
        _ext().qmm_group(x, xs, [q.weight], [q.scales], [q.biases], [out], [q.n], [sk], f32,
                         group_tile(m, *_chip(x.device.index)), -1, bits)
        return out
    if sk > 1 and not reduce and part is None:
        part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    _ext().qmm(x, xs, q.weight, q.scales, q.biases, out, part, q.n, sk, q.gs, bucket(m), f32, reduce, bits)
    return out if sk == 1 or reduce else part.reshape(-1)[:sk * m * q.n].view(sk, m, q.n)


def matmul_group(x: torch.Tensor, qs: list[Q4], xs: torch.Tensor | None = None, *, f32: bool = False,
                 sks: list[int] | None = None, tile: int = 0, early: int = -1) -> list[torch.Tensor]:
    """``[matmul(x, q) for q in qs]`` in one sm_12x launch, same bits; ``tile``, ``early`` (-1: by chip) for tests."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or any(x.shape[1] != q.k for q in qs):
        raise ValueError("matmul_group: x must be (M, K) bf16 with every weight's K")
    sks = sks or [split_k(q.n, q.k, q.gs, bits=getattr(q, "bits", 4)) for q in qs]
    widths = {getattr(q, "bits", 4) for q in qs}
    if not (1 <= len(qs) <= 4 and all(q.gs == 64 for q in qs) and len(widths) == 1 and grouped(x.device.index)):
        return [matmul(x, q, xs, sk=s, f32=f32) for q, s in zip(qs, sks)]
    if x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    if xs is None:
        xs = group_sums(x, 64)
    dtype = torch.float32 if f32 else torch.bfloat16
    outs = [torch.empty((x.shape[0], q.n), dtype=dtype, device=x.device) for q in qs]
    tile = tile or group_tile(x.shape[0], *_chip(x.device.index))
    _ext().qmm_group(x, xs, [q.weight for q in qs], [q.scales for q in qs], [q.biases for q in qs], outs,
                     [q.n for q in qs], sks, f32, tile, early, widths.pop())
    return outs


def prompt_tile(m: int, n: int) -> int:
    """The prompt matmul's tile: 128x128 on four 64x64 warps, two blocks an SM, tuned for a GB10."""

    return 9


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
