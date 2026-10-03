"""Flash Next's CUDA key/value caches: bf16, or ExLlamaV3's -cq 8 / -cq 4 codes (H32-rotated groups of 32, fp16 absmax scales, midpoint grid) kept rotated."""

from __future__ import annotations

import math

import torch

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


def row_bytes(kv_heads: int, head_dim: int, dtype: str) -> int:
    """Bytes a position of one layer's keys and values take, their per-row scales included."""

    if dtype == "bf16":
        return 2 * kv_heads * head_dim * 2
    return 2 * kv_heads * (head_dim if dtype == "int8" else head_dim // 2) + 2 * kv_heads * (head_dim // GROUP) * 2


# -- storage -------------------------------------------------------------------------------------------
class KVCache:
    """One attention layer's keys and values ``[capacity, kv_heads, head_dim]`` (int4: head_dim / 2 bytes); a bf16 cache keeps one-element scales so every kernel takes one argument list."""

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

    def resized(self, capacity: int, keep: int) -> "KVCache":
        """A cache of ``capacity`` rows holding this one's first ``keep`` rows (a bf16 cache's one-element scales too)."""

        other = KVCache(capacity, self.kv_heads, self.head_dim, self.k.device, self.dtype)
        keep = min(int(keep), self.capacity, int(capacity))
        other.k[:keep], other.v[:keep] = self.k[:keep], self.v[:keep]
        if self.quantized:
            other.ks[:keep], other.vs[:keep] = self.ks[:keep], self.vs[:keep]
        return other

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
    """bf16 or fp32 ``[N, HK, D]`` keys and values -> (k codes, k scales, v codes, v scales) on ExLlamaV3's midpoint grid in fp32: 8 bits store q - 128, 4 bits two codes a byte."""

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
    """Codes and fp16 scales -> bf16 still in the cache's rotation: (code + 0.5) * s / 128 at 8 bits, (q - 7.5) * s / 8 at 4."""

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
