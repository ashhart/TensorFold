"""DeepSeek-V4.1's KV quantizers in plain PyTorch, bit for bit: the packed-cache tests' reference.

Ported from DeepSeek-V4.1-Flash ``inference/kernel.py`` (``fp4_quant_kernel`` with E4M3 and E8M0 scales,
``act_quant_kernel`` with ``round_scale`` and ``inplace``, ``fast_log2_ceil`` / ``fast_round_scale``),
Copyright (c) 2026 DeepSeek, MIT License. fp32 math as there; the E2M1 cast rounds to nearest, ties to the even code
(``cvt.rn.satfinite``).

Packed layout (TensorFold's choice): byte j of a row holds element 2j in the low nibble and 2j + 1 in the high one; a
nibble is sign (bit 3) | magnitude index over {0, .5, 1, 1.5, 2, 3, 4, 6}, negative zero stored as 0. Scale bytes:
the E4M3 bits (main KV, a scale per 16), or the biased exponent k + 127 of a 2^k scale (indexer, a scale per 32).
"""

from __future__ import annotations

import torch

FP4_MAX, FP8_MAX = 6.0, 448.0
MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _f32(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


def log2_ceil(t: torch.Tensor) -> torch.Tensor:
    """ceil(log2 t) of fp32 t > 0 from its bits (fast_log2_ceil): the exponent, plus one when any mantissa bit is set."""

    b = t.float().contiguous().view(torch.int32)
    return ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).int()


def pow2(k: torch.Tensor) -> torch.Tensor:
    return ((k.int() + 127) << 23).view(torch.float32)


def e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """int32 nibbles of fp32 y (already within +-6): round to nearest, ties to even; -0 -> 0."""

    a = y.abs()
    m = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
         + (a >= 3.5).int() + (a > 5.0).int())
    return torch.where((y < 0) & (m > 0), m | 8, m)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    v = torch.tensor(MAGS, dtype=torch.float32, device=codes.device)[(codes & 7).long()]
    return torch.where(codes >= 8, -v, v)


def pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def unpack(b: torch.Tensor) -> torch.Tensor:
    b = b.int()
    return torch.stack([b & 15, b >> 4], dim=-1).flatten(-2)


def nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp4_act_quant(x, 16, scale_dtype=float8_e4m3fn) of bf16 rows x [n, D]: (packed u8 [n, D/2], e4m3 scale bytes
    [n, D/16], the in-place result bf16 [n, D])."""

    n, D = x.shape
    g = x.float().view(n, D // 16, 16)
    amax = g.abs().amax(-1).clamp(min=6 * 2 ** -9)
    s8 = (amax / FP4_MAX).to(torch.float8_e4m3fn)
    s = s8.float()[..., None]
    codes = e2m1_codes((g / s).clamp(-FP4_MAX, FP4_MAX))
    deq = (e2m1_values(codes) * s).view(n, D).to(torch.bfloat16)
    return pack(codes.view(n, D)), s8.view(torch.uint8), deq


def mxfp4(x: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp4_act_quant(x, group) (E8M0 scales) of bf16 rows x [..., D]: (packed u8, scale bytes k + 127, bf16)."""

    shp = x.shape
    g = x.float().reshape(-1, shp[-1] // group, group)
    amax = g.abs().amax(-1).clamp(min=6 * 2 ** -126)
    k = log2_ceil(amax * _f32(1 / FP4_MAX).to(g.device))
    s = pow2(k)[..., None]
    codes = e2m1_codes((g / s).clamp(-FP4_MAX, FP4_MAX))
    deq = (e2m1_values(codes) * s).view(shp).to(torch.bfloat16)
    return pack(codes.view(shp)), (k + 127).to(torch.uint8).view(*shp[:-1], -1), deq


def mxfp8(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    """act_quant(x, group, "ue8m0", inplace=True) of bf16 rows x [..., D]: e4m3 with a 2^k scale per group, bf16."""

    shp = x.shape
    g = x.float().reshape(-1, shp[-1] // group, group)
    amax = g.abs().amax(-1).clamp(min=_f32(1e-4).item())
    s = pow2(log2_ceil(amax * _f32(1 / FP8_MAX).to(g.device)))[..., None]
    q = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).view(shp).to(torch.bfloat16)


def dequant(packed: torch.Tensor, scales: torch.Tensor, group: int, e4m3: bool) -> torch.Tensor:
    """bf16 rows of a packed cache (the inverse of ``nvfp4`` / ``mxfp4``'s first two outputs)."""

    v = e2m1_values(unpack(packed))
    n, D = v.shape
    s = scales.view(torch.float8_e4m3fn).float() if e4m3 else pow2(scales.int() - 127)
    return (v.view(n, D // group, group) * s[..., None]).view(n, D).to(torch.bfloat16)
