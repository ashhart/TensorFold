"""Flash Next's key/value caches on CUDA: bf16 (the default), int8, or int4.

The scheme is ExLlamaV3's cache quantization (``-cq 8`` and ``-cq 4``, the default non-companded
grid, ``compand_a == 0`` in ``exllamav3_ext/cache/q_cache_kernels.cuh``). For each group of 32
values along the head dim:

- the group is rotated by H32 (the Hadamard, times 1/sqrt(32));
- ``s = max|v| + 1e-10``, kept as fp16 (one scale per group, round to nearest);
- ``q = clamp(floor(v * (1/s) * m + m), 0, 2*m - 1)`` with ``m = 2^(bits-1)`` (128 at 8 bits,
  8 at 4 bits), the midpoint grid. Reconstruction in the rotated domain is
  ``(q - (m - 0.5)) / m * s``.

What this file copies from that kernel, and what it does not:

- the grid, the scale (fp16 absmax), the group of 32, and the H32 (five butterfly stages, bit 0
  first, then 1/sqrt(32)). The butterflies here are the same matrix as ExLlamaV3's in-register H4
  plus subgroup H8; the two orders commute.
- 4-bit packing, linear little-endian: value ``j`` of a group occupies bits ``[4j, 4j+4)`` of the
  group's 16 bytes. Two unsigned codes per byte, low nibble = even index, high nibble = odd index.
  ExLlamaV3 holds those bits in ``num_bits`` uint32 words per group. This cache holds the same bits
  in a uint8 tensor of length ``head_dim / 2``.
- 8-bit is not packed into uint32 words. The code is stored as ``q - 128`` in an int8 tensor, one
  byte a value (the same 8 bits, a different container). The read is ``(code + 0.5) * s / 128``.
- ExLlamaV3's dequantizer multiplies the scale by 1/sqrt(32) again and runs the unnormalized
  butterflies to rotate back. This cache does not. ``glue.attn_prep`` rotates the query
  (``q . (H k) = (H q) . k``) and ``attention._merge`` rotates the merged output
  (``p . (H v) = H (p . v)``), so the stored codes stay in the rotated domain. ``H32 H32 = I``.

What is not quantized: the indexer keys (``ikc``) and the pooled block keys stay bf16.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

GROUP = 32                      # values per scale (ExLlamaV3's cache-quant group)
SCALE_DTYPE = torch.float16     # ExLlamaV3 stores the group absmax as a half (__float2half_rn)
DTYPES = ("bf16", "int8", "int4")
BITS_OF = {"bf16": 16, "int8": 8, "int4": 4}
R32 = 1.0 / math.sqrt(32)


def check(dtype: str) -> str:
    """Refuse a cache dtype this engine has no storage for."""

    if dtype not in DTYPES:
        raise ValueError(f"kv-dtype {dtype!r}: this engine serves {' or '.join(DTYPES)}")
    return dtype


def kernel_bits(dtype: str) -> int:
    """The constexpr the kernels branch on: 0 keeps the bf16 path, 8 and 4 quantize."""

    return 0 if dtype == "bf16" else BITS_OF[check(dtype)]


# -- the transform -------------------------------------------------------------------------------------
@triton.jit
def _h32_stage(x, M: tl.constexpr, HI: tl.constexpr, LO: tl.constexpr):
    """One butterfly stage over the (HI, 2, LO) bits of a (M, 32) block: the elements that differ in one bit
    are added and subtracted, the low half keeping the sum. Stages over disjoint bits commute; this is the
    order ExLlamaV3's quantizer runs them in (its H4 in registers, then its H8 across the subgroup)."""

    t = tl.trans(tl.reshape(x, (M, HI, 2, LO)), 0, 1, 3, 2)
    a, b = tl.split(t)
    return tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 1, 3, 2), (M, 32))


@triton.jit
def h32(x, M: tl.constexpr):
    """H32 over the last axis of an (M, 32) fp32 block.

    Five butterfly stages (distances 1, 2, 4, 8, 16: bit 0 first), then the 1/sqrt(32) that makes the
    transform orthogonal (``H32 H32 = I``). Adds and subtracts only, in the order ExLlamaV3's cache
    quantizer applies to a group, so the two engines' stored bits agree to the last one.
    """

    c32: tl.constexpr = 0.17677669529663688110      # 1 / sqrt(32)
    x = _h32_stage(x, M, 16, 1)
    x = _h32_stage(x, M, 8, 2)
    x = _h32_stage(x, M, 4, 4)
    x = _h32_stage(x, M, 2, 8)
    x = _h32_stage(x, M, 1, 16)
    return x * c32


@triton.jit
def _quant_scale(x, M: tl.constexpr):
    """H32, then the fp16 absmax. Shared arithmetic; the two widths store different code tensors, and
    Triton rejects one function that returns both shapes."""

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


# -- storage -------------------------------------------------------------------------------------------
class KVCache:
    """One attention layer's keys and values: ``[capacity, kv_heads, head_dim]``, bf16 or quantized.

    The scale tensors always exist so the kernels can take one set of arguments whatever the dtype; with
    ``bf16`` they are a single unused element and ``bits == 0`` on the kernel keeps them out of the code.
    An int4 cache stores two codes per byte, so ``k`` and ``v`` have last dimension ``head_dim / 2``.
    """

    def __init__(self, capacity: int, kv_heads: int, head_dim: int, device, dtype: str = "bf16") -> None:
        check(dtype)
        if dtype != "bf16" and head_dim % GROUP:
            raise ValueError(f"a quantized KV cache needs a head dim that is a multiple of {GROUP}, not {head_dim}")
        self.dtype = dtype
        self.bits = BITS_OF[dtype]
        self.capacity, self.kv_heads, self.head_dim = int(capacity), int(kv_heads), int(head_dim)
        shape = (int(capacity), int(kv_heads), int(head_dim))
        if dtype == "int8":
            self.k = torch.zeros(shape, dtype=torch.int8, device=device)
            self.v = torch.zeros(shape, dtype=torch.int8, device=device)
            groups = (int(capacity), int(kv_heads), int(head_dim) // GROUP)
            self.ks = torch.zeros(groups, dtype=SCALE_DTYPE, device=device)
            self.vs = torch.zeros(groups, dtype=SCALE_DTYPE, device=device)
        elif dtype == "int4":
            packed = (int(capacity), int(kv_heads), int(head_dim) // 2)
            self.k = torch.zeros(packed, dtype=torch.uint8, device=device)
            self.v = torch.zeros(packed, dtype=torch.uint8, device=device)
            groups = (int(capacity), int(kv_heads), int(head_dim) // GROUP)
            self.ks = torch.zeros(groups, dtype=SCALE_DTYPE, device=device)
            self.vs = torch.zeros(groups, dtype=SCALE_DTYPE, device=device)
        else:
            self.k = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            self.v = torch.zeros_like(self.k)
            self.ks = torch.zeros((1,), dtype=SCALE_DTYPE, device=device)
            self.vs = torch.zeros((1,), dtype=SCALE_DTYPE, device=device)

    @property
    def quantized(self) -> bool:
        return self.dtype != "bf16"

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes + self.ks.nbytes + self.vs.nbytes

    def clone(self) -> "KVCache":
        other = object.__new__(KVCache)
        other.dtype = self.dtype
        other.bits = self.bits
        other.capacity, other.kv_heads, other.head_dim = self.capacity, self.kv_heads, self.head_dim
        other.k, other.v = self.k.clone(), self.v.clone()
        other.ks, other.vs = self.ks.clone(), self.vs.clone()
        return other


# -- the reference quantizer (tests, and what the kernels are checked against) --------------------------
def h32_ref(x: torch.Tensor) -> torch.Tensor:
    """H32 over the last axis of a (..., 32k) fp32 tensor, in the operations ``h32`` runs: -> (..., k, 32)."""

    x = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    for lo in (1, 2, 4, 8, 16):
        hi = 32 // (2 * lo)
        t = x.reshape(*x.shape[:-1], hi, 2, lo)
        a, b = t[..., 0, :], t[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], 32)
    return x * R32


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Unsigned codes (..., even) in 0..15 -> uint8, low nibble = even index, high nibble = odd index."""

    pair = q.to(torch.int32).reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """The inverse of ``pack_nibbles``: uint8 -> int32 codes, low nibble first."""

    p = packed.to(torch.int32)
    lo, hi = p & 15, (p >> 4) & 15
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_ref(k: torch.Tensor, v: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, ...]:
    """bf16 (or fp32) ``[N, HK, D]`` keys and values -> (k codes, k scales, v codes, v scales).

    ``bits`` 8 stores ``q - 128`` as int8, one byte a value. ``bits`` 4 stores two unsigned codes per
    byte (``pack_nibbles``). The arithmetic is ExLlamaV3's midpoint grid in torch fp32.
    """

    if bits not in (8, 4):
        raise ValueError(f"quantize_ref bits must be 8 or 4, not {bits}")
    m = 1 << (bits - 1)
    qmax = float((1 << bits) - 1)
    out: list[torch.Tensor] = []
    for x in (k, v):
        rot = h32_ref(x.float())
        s = rot.abs().amax(dim=-1) + 1e-10
        q = torch.floor(rot * (1.0 / s)[..., None] * m) + m
        q = q.clamp(0.0, qmax)
        if bits == 8:
            out.append((q - 128.0).to(torch.int8).reshape(x.shape))
        else:
            out.append(pack_nibbles(q).reshape(*x.shape[:-1], x.shape[-1] // 2))
        out.append(s.to(SCALE_DTYPE))
    return out[0], out[1], out[2], out[3]


def dequant_ref(code: torch.Tensor, scale: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """int8 or packed-uint8 codes + fp16 scales -> bf16, still in the cache's rotation.

    8-bit: ``(code + 0.5) * s / 128``. 4-bit: unpack, then ``(q - 7.5) * s / 8``.
    """

    if bits == 8:
        width = code.shape[-1]
        c = code.float().reshape(*code.shape[:-1], width // GROUP, GROUP)
        s = scale.float().reshape(*scale.shape, 1)
        return ((c + 0.5) * s * 0.0078125).to(torch.bfloat16).reshape(code.shape)
    q = unpack_nibbles(code).float()
    width = q.shape[-1]
    c = q.reshape(*q.shape[:-1], width // GROUP, GROUP)
    s = scale.float().reshape(*scale.shape, 1)
    return ((c - 7.5) * s * 0.125).to(torch.bfloat16).reshape(q.shape)
