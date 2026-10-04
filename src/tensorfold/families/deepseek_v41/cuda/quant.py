"""DeepSeek-V4.1's activation quantize-dequantize per group of the last dim, bf16 in and out."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

GROUPS = 16                     # groups per program
INV448 = tl.constexpr(1 / 448)          # rounded to fp32 where it multiplies, as the reference's f32(1/448)
INV6 = tl.constexpr(1 / 6)
FLOOR8 = tl.constexpr(1e-4)             # amax floors (K:76, K:165, K:162)
FLOOR4_E8M0 = tl.constexpr(6 * 2.0**-126)
FLOOR4_E4M3 = tl.constexpr(6 * 2.0**-9)

FP8, FP4_E8M0, FP4_E4M3 = 0, 1, 2


@triton.jit
def _pow2_ceil(t):
    """K:22-33: 2^ceil(log2 t) from t's fp32 bits, the exponent plus one when the mantissa is non-zero."""

    bits = t.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1(v):
    """The nearest e2m1 value of |v| <= 6, ties to even (grid steps 0.5, 1, 2), keeping v's sign bit (-0 too)."""

    a = tl.abs(v)
    inv = tl.where(a < 2.0, 2.0, tl.where(a < 4.0, 1.0, 0.5))
    r = libdevice.rint(a * inv) * tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    sign = v.to(tl.int32, bitcast=True) & -2147483648
    return (r.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _qdq(X, Y, n, per_row, sx, sy, G: tl.constexpr, KIND: tl.constexpr, B: tl.constexpr):
    g = tl.program_id(0) * B + tl.arange(0, B)
    ok = (g < n)[:, None]
    row, col = (g // per_row).to(tl.int64), (g % per_row) * G
    j = tl.arange(0, G)[None, :]
    x = tl.load(X + row[:, None] * sx + col[:, None] + j, mask=ok, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if KIND == 0:                                   # K:70-86: FP8 e4m3, scale 2^ceil(log2(amax / 448))
        s = _pow2_ceil(tl.maximum(amax, FLOOR8) * INV448)[:, None]
        q = tl.minimum(tl.maximum(tl.math.div_rn(x, s), -448.0), 448.0)
        q = q.to(tl.float8e4nv).to(tl.float32)
    else:
        if KIND == 1:                               # K:165-166: FP4 e2m1, scale 2^ceil(log2(amax / 6))
            s = _pow2_ceil(tl.maximum(amax, FLOOR4_E8M0) * INV6)[:, None]
        else:                                       # K:162-163: FP4 e2m1, scale e4m3(amax / 6)
            s = tl.math.div_rn(tl.maximum(amax, FLOOR4_E4M3), 6.0).to(tl.float8e4nv).to(tl.float32)[:, None]
        q = _e2m1(tl.minimum(tl.maximum(tl.math.div_rn(x, s), -6.0), 6.0))
    tl.store(Y + row[:, None] * sy + col[:, None] + j, (q * s).to(tl.bfloat16), mask=ok)


def _run(x: torch.Tensor, out: torch.Tensor | None, group: int, kind: int) -> torch.Tensor:
    out = x if out is None else out
    if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or out.shape != x.shape:
        raise ValueError("quant: bf16 input and output of one shape")
    d = x.shape[-1]
    if d % group:
        raise ValueError(f"quant: the last dim {d} is not a multiple of the group {group}")
    xs, ys = x.view(-1, d), out.view(-1, d)
    if xs.stride(-1) != 1 or ys.stride(-1) != 1:
        raise ValueError("quant: the last dim needs unit stride")
    n = xs.numel() // group
    if n:
        _qdq[(triton.cdiv(n, GROUPS),)](xs, ys, n, d // group, xs.stride(0), ys.stride(0), G=group, KIND=kind,
                                        B=GROUPS)
    return out


def fp8_qdq_1x32(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 e4m3 per 32 with a power-of-two scale (the window KV); in place unless ``out`` is given."""

    return _run(x, out, 32, FP8)


def fp4_qdq_1x32_e8m0(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP4 e2m1 per 32 with a power-of-two scale (index Q and index-K); in place unless ``out`` is given."""

    return _run(x, out, 32, FP4_E8M0)


def fp4_qdq_1x16_e4m3(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP4 e2m1 per 16 with an e4m3 scale (compressed KV entries); in place unless ``out`` is given."""

    return _run(x, out, 16, FP4_E4M3)
