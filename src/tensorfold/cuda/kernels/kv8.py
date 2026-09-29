"""Packed FP8 attention cache rows (``--kv-dtype fp8``, ROCm): a (token, KV head) row of 256 values is 272 bytes, the
e4m3 codes of x * 2^-e (e the smallest exponent that puts max |x| at or under 448), then e as int8, then 15 zeros.
A row's values are its codes as fp32 times 2^e, truncated to bf16 (exact for any row with max |x| >= 2^-110)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

D8 = 256                 # values a packed row holds: the head size the WMMA kernels read it for
ROW8 = 272               # bytes a packed row: the codes, e, and padding to a 16-byte multiple


@triton.jit
def _pow2(e):
    """2^e in fp32 from its bits, exact for integer e in [-127, 127] (2^-127 is the subnormal 1 << 22)."""

    return tl.where(e >= -126, (e + 127) << 23, 1 << 22).to(tl.float32, bitcast=True)


@triton.jit
def _pack(X, R, P, XS, HS, HK: tl.constexpr, D: tl.constexpr, ROW: tl.constexpr):
    """Program (row, head): e from the bits of max |x| (a bf16 value), the codes, and the values they return."""

    pid = tl.program_id(0)
    row, head = pid // HK, pid % HK
    d = tl.arange(0, D)
    x = tl.load(X + row * XS + head * HS + d).to(tl.float32)
    bits = tl.max(tl.abs(x), axis=0).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 135 + (((bits >> 16) & 0x7F) > 0x60).to(tl.int32)    # +1 past a mantissa of 1.75
    e = tl.minimum(tl.maximum(e, -127), 127)
    q = (x * _pow2(-e)).to(tl.float8e4nv)                           # exact scaling; at most 448 by construction
    back = (q.to(tl.float32) * _pow2(e)).to(tl.int32, bitcast=True) >> 16
    tl.store(R + pid * D + d, back.to(tl.int16))
    tl.store(P + pid * ROW + d, q.to(tl.uint8, bitcast=True))
    j = tl.arange(0, ROW - D)
    tl.store(P + pid * ROW + D + j, tl.where(j == 0, e & 0xFF, 0).to(tl.uint8))


def pack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x (W, HK, 256) bf16 (rows and heads may stride) -> the bf16 values the packed rows hold (``unpack`` of them,
    bit for bit) and the packed (W, HK, ROW8) uint8 rows, from one launch."""

    if x.dim() != 3 or x.shape[2] != D8 or x.dtype != torch.bfloat16:
        raise ValueError(f"kv8.pack takes (rows, heads, {D8}) bf16 values, not {tuple(x.shape)} {x.dtype}")
    if x.stride(2) != 1:
        x = x.contiguous()
    w, hk = x.shape[:2]
    rounded = torch.empty((w, hk, D8), dtype=torch.bfloat16, device=x.device)
    packed = torch.empty((w, hk, ROW8), dtype=torch.uint8, device=x.device)
    if w * hk:
        _pack[(w * hk,)](x, rounded.view(torch.int16), packed, x.stride(0), x.stride(1), HK=hk, D=D8, ROW=ROW8,
                         num_warps=2)
    return rounded, packed


def _pow2_torch(e: torch.Tensor) -> torch.Tensor:
    return torch.where(e >= -126, (e + 127) << 23, 1 << 22).to(torch.int32).view(torch.float32)


def exponent(amax: torch.Tensor) -> torch.Tensor:
    """Each row's e (int32) from its max |x| as bf16: amax = f * 2^E with f in [1, 2), e = E - 8, +1 when f > 1.75."""

    bits = amax.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return (((bits >> 7) & 0xFF) - 135 + ((bits & 0x7F) > 0x60).to(torch.int32)).clamp(-127, 127)


def reference_pack(x: torch.Tensor) -> torch.Tensor:
    """``pack``'s rows in torch: (..., 256) bf16 -> (..., ROW8) uint8."""

    e = exponent(x.abs().amax(-1))
    codes = (x.float() * _pow2_torch(-e)[..., None]).to(torch.float8_e4m3fn)
    packed = torch.zeros((*x.shape[:-1], ROW8), dtype=torch.uint8, device=x.device)
    packed[..., :D8] = codes.view(torch.uint8)
    packed[..., D8] = e.to(torch.int8).view(torch.uint8)
    return packed


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """(..., ROW8) uint8 rows -> their (..., 256) bf16 values: the codes as fp32 times 2^e, the top 16 bits."""

    e = packed[..., D8].view(torch.int8).to(torch.int32)
    values = packed[..., :D8].view(torch.float8_e4m3fn).float() * _pow2_torch(e)[..., None]
    return (values.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
