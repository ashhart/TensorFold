"""Triton kernels of the quantized KV cache (``kvcache``): H32, the midpoint-grid quantizer and its dequant."""

from __future__ import annotations

import triton
import triton.language as tl


# -- the transform -------------------------------------------------------------------------------------
@triton.jit
def _h32_stage(x, M: tl.constexpr, HI: tl.constexpr, LO: tl.constexpr):
    """One butterfly stage over the (HI, 2, LO) bits of an (M, 32) block; stages over disjoint bits commute (ExLlamaV3 runs H4, then H8)."""

    t = tl.trans(tl.reshape(x, (M, HI, 2, LO)), 0, 1, 3, 2)
    a, b = tl.split(t)
    return tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 1, 3, 2), (M, 32))


@triton.jit
def h32(x, M: tl.constexpr):
    """H32 over the last axis of an (M, 32) fp32 block: five butterfly stages, bit 0 first, then 1/sqrt(32), so H32 H32 = I and the stored bits are ExLlamaV3's."""

    c32: tl.constexpr = 0.17677669529663688110      # 1 / sqrt(32)
    x = _h32_stage(x, M, 16, 1)
    x = _h32_stage(x, M, 8, 2)
    x = _h32_stage(x, M, 4, 4)
    x = _h32_stage(x, M, 2, 8)
    x = _h32_stage(x, M, 1, 16)
    return x * c32


@triton.jit
def _quant_scale(x, M: tl.constexpr):
    """H32, then the fp16 absmax, shared by both widths (Triton rejects one function returning both code shapes)."""

    x = h32(x, M)
    s = tl.max(tl.abs(x), axis=1) + 1e-10
    inv = tl.math.div_rn(1.0, s)                    # ExLlamaV3's 1.0f / s, IEEE division
    return x, s, inv


@triton.jit
def quant_groups_8(x, M: tl.constexpr):
    """(M, 32) fp32 -> int8 codes (M, 32), ``q - 128``, and fp16 scales."""

    x, s, inv = _quant_scale(x, M)
    q = tl.floor(x * inv[:, None] * 128.0) + 128.0
    code = tl.minimum(tl.maximum(q, 0.0), 255.0) - 128.0
    return code.to(tl.int8), s.to(tl.float16)


@triton.jit
def quant_groups_4(x, M: tl.constexpr):
    """(M, 32) fp32 -> uint8 (M, 16), two unsigned codes per byte, low nibble = even index."""

    x, s, inv = _quant_scale(x, M)
    q = tl.floor(x * inv[:, None] * 8.0) + 8.0
    q = tl.minimum(tl.maximum(q, 0.0), 15.0).to(tl.int32)
    lo, hi = tl.split(tl.reshape(q, (M, 16, 2)))
    return (lo | (hi << 4)).to(tl.uint8), s.to(tl.float16)


@triton.jit
def dequant_group_8(code, scale, M: tl.constexpr, W: tl.constexpr):
    """int8 codes (M, W) -> bf16, still rotated: ``(code + 0.5) * s / 128``."""

    s = tl.reshape(scale.to(tl.float32), (M, W // 32, 1))
    c = tl.reshape(code.to(tl.float32), (M, W // 32, 32))
    return tl.reshape((c + 0.5) * s * 0.0078125, (M, W)).to(tl.bfloat16)


@triton.jit
def dequant_group_4(code, scale, M: tl.constexpr, W: tl.constexpr):
    """uint8 (M, W/2), low nibble first -> bf16 (M, W), still rotated: ``(q - 7.5) * s / 8``."""

    s = tl.reshape(scale.to(tl.float32), (M, W // 32, 1))
    raw = code.to(tl.int32)
    lo = (raw & 15).to(tl.float32)
    hi = ((raw >> 4) & 15).to(tl.float32)
    q = tl.reshape(tl.join(lo, hi), (M, W))
    c = tl.reshape(q, (M, W // 32, 32))
    return tl.reshape((c - 7.5) * s * 0.125, (M, W)).to(tl.bfloat16)
