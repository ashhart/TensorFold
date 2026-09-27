"""Flash Next's key/value caches on CUDA: bf16 (the default) or int8 with one fp16 scale per 32 values.

The scheme is ExLlamaV3's 8-bit cache quantization (``-cq 8`` with the default, non-companded grid), so the
two engines' stored bits are comparable. For each group of 32 values along the head dim it stores:

- the group rotated by H32 (the Hadamard matrix, times 1/sqrt(32)); the rotation regularizes the group
  toward a Gaussian, so its absmax sits closer to its RMS and the 8-bit grid is used better;
- ``s = max|v| + 1e-10`` in fp32, kept as **fp16** (one scale per group);
- ``q = clamp(floor(v * (1/s) * 128) + 128, 0, 255)``, the 8-bit midpoint grid, one byte per value.

Dequantized, ``v_hat = (q - 127.5) / 128 * s``. TensorFold keeps the code as ``q - 128`` in an int8 tensor
(1 byte a value, the same 8 bits ExLlamaV3 packs into uint32 words) and reads it back in the attention
kernel as ``(code + 0.5) * s / 128``: 1 byte + 2/32 bytes a value, **1.88x** the bf16 cache is.

The rotation is orthogonal and symmetric (``H32 H32 = I``), which is what keeps it cheap here: instead of
rotating every key and value back at read time, ``glue.attn_prep`` rotates the *query* once per step
(``q . (H k) = (H q) . k``) and ``attention._merge`` rotates the merged output once a row
(``p . (H v) = H (p . v)``). Keys and values come out of the cache exactly as ExLlamaV3's dequantizer
would produce them; nothing about a row depends on the window it runs in.

What is *not* quantized: the indexer keys (``ikc``) and the pooled block keys of the sparse path stay bf16,
as ExLlamaV3 keeps its indexer/pool cache separate and quantizes it only when asked.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

GROUP = 32                      # values per scale (ExLlamaV3's cache-quant group)
SCALE_DTYPE = torch.float16     # ExLlamaV3 stores the group absmax as a half
DTYPES = ("bf16", "int8")
R32 = 1.0 / math.sqrt(32)


def check(dtype: str) -> str:
    """Refuse a cache dtype this engine has no storage for."""

    if dtype not in DTYPES:
        raise ValueError(f"kv-dtype {dtype!r}: this engine serves {' or '.join(DTYPES)}")
    return dtype


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
def quant_groups(x, M: tl.constexpr):
    """(M, 32) fp32 -> (int8 codes (M, 32), fp16 scales (M,)): rotate, absmax, the midpoint grid."""

    x = h32(x, M)
    s = tl.max(tl.abs(x), axis=1) + 1e-10
    inv = tl.math.div_rn(1.0, s)                    # ExLlamaV3's 1.0f / s, IEEE division
    q = tl.floor(x * inv[:, None] * 128.0) + 128.0
    code = tl.minimum(tl.maximum(q, 0.0), 255.0) - 128.0
    return code.to(tl.int8), s.to(tl.float16)


@triton.jit
def dequant_group(code, scale, M: tl.constexpr, W: tl.constexpr):
    """int8 codes (M, W) with W a multiple of 32 -> bf16 (M, W), ``(code + 0.5) * s / 128``."""

    c = tl.reshape(code.to(tl.float32), (M, W // 32, 32))
    s = tl.reshape(scale.to(tl.float32), (M, W // 32, 1))
    return tl.reshape((c + 0.5) * s * 0.0078125, (M, W)).to(tl.bfloat16)


# -- storage -------------------------------------------------------------------------------------------
class KVCache:
    """One attention layer's keys and values: ``[capacity, kv_heads, head_dim]``, bf16 or int8 + scales.

    The scale tensors always exist so the kernels can take one set of arguments whatever the dtype; with
    ``bf16`` they are a single unused element and the kernels' ``KVQ`` flag keeps them out of the code.
    """

    def __init__(self, capacity: int, kv_heads: int, head_dim: int, device, dtype: str = "bf16") -> None:
        check(dtype)
        if dtype == "int8" and head_dim % GROUP:
            raise ValueError(f"an int8 KV cache needs a head dim that is a multiple of {GROUP}, not {head_dim}")
        self.dtype = dtype
        self.capacity, self.kv_heads, self.head_dim = int(capacity), int(kv_heads), int(head_dim)
        shape = (int(capacity), int(kv_heads), int(head_dim))
        if dtype == "int8":
            self.k = torch.zeros(shape, dtype=torch.int8, device=device)
            self.v = torch.zeros(shape, dtype=torch.int8, device=device)
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
        return self.dtype == "int8"

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes + self.ks.nbytes + self.vs.nbytes

    def clone(self) -> "KVCache":
        other = object.__new__(KVCache)
        other.dtype = self.dtype
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


def quantize_ref(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The reference quantizer: bf16 (or fp32) ``[N, HK, D]`` keys and values -> (k codes, k scales, v codes,
    v scales), the arithmetic of ExLlamaV3's Q8 cache in torch fp32."""

    out: list[torch.Tensor] = []
    for x in (k, v):
        xf = x.float()
        rot = h32_ref(xf)
        s = rot.abs().amax(dim=-1) + 1e-10
        q = torch.floor(rot * (1.0 / s)[..., None] * 128.0) + 128.0
        out.append((q.clamp(0.0, 255.0) - 128.0).to(torch.int8).view_as(k))
        out.append(s.to(SCALE_DTYPE))
    return out[0], out[1], out[2], out[3]


def dequant_ref(code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The reference dequantizer: int8 codes + fp16 scales -> bf16 (still in the cache's rotation, as the
    attention kernel reads them), ``(code + 0.5) * s / 128``."""

    width = code.shape[-1]
    c = code.float().reshape(*code.shape[:-1], width // GROUP, GROUP)
    s = scale.float().reshape(*scale.shape, 1)
    return ((c + 0.5) * s * 0.0078125).to(torch.bfloat16).view(code.shape)
