"""Packed int8 key/value rows: a (position, KV head) row holds its head_dim values as ExLlamaV3 ``-cq 8`` codes
(H32-rotated groups of 32, midpoint grid, stored ``q - 128``) followed by the groups' fp16 absmax scales, so an int8
cache stays one tensor a layer that grows, slices, copies and clones by position like a bf16 one."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda.kvquant import h32, quant_groups_8

GROUP = 32                  # values an fp16 scale covers (ExLlamaV3's cache-quant group)
ROWS = 8                    # (position, KV head) rows a program packs or unpacks
DTYPES = ("bf16", "int8")


def check(dtype: str) -> str:
    """Refuse a cache dtype these rows have no layout for."""

    if dtype not in DTYPES:
        raise ValueError(f"kv-dtype {dtype!r}: this engine serves {' or '.join(DTYPES)}")
    return dtype


def row_bytes(head_dim: int) -> int:
    """One packed row: ``head_dim`` int8 codes, then an fp16 scale a group of 32."""

    if head_dim % GROUP:
        raise ValueError(f"an int8 KV cache needs a head dim that is a multiple of {GROUP}, not {head_dim}")
    return head_dim + head_dim // GROUP * 2


@triton.jit
def _pack(X, P8, P16, n, D: tl.constexpr, RB: tl.constexpr, R: tl.constexpr):
    """R rows of D bf16 values -> each row's D codes, then its D / 32 fp16 scales, in its RB bytes."""

    G: tl.constexpr = D // 32
    rows = tl.program_id(0) * R + tl.arange(0, R)
    ok = rows < n
    d = tl.arange(0, D)
    x = tl.load(X + rows[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    code, scale = quant_groups_8(tl.reshape(x, (R * G, 32)), M=R * G)
    tl.store(P8 + rows[:, None] * RB + d[None, :], tl.reshape(code, (R, D)), mask=ok[:, None])
    g = tl.arange(0, G)
    tl.store(P16 + (rows[:, None] * RB + D) // 2 + g[None, :], tl.reshape(scale, (R, G)), mask=ok[:, None])


@triton.jit
def _unpack(P8, P16, X, n, D: tl.constexpr, RB: tl.constexpr, R: tl.constexpr):
    """R packed rows -> bf16 in the model's basis: ``(code + 0.5) * s / 128`` in fp32, then H32 (its own inverse)."""

    G: tl.constexpr = D // 32
    rows = tl.program_id(0) * R + tl.arange(0, R)
    ok = rows < n
    d = tl.arange(0, D)
    g = tl.arange(0, G)
    c = tl.load(P8 + rows[:, None] * RB + d[None, :], mask=ok[:, None], other=0).to(tl.float32)
    s = tl.load(P16 + (rows[:, None] * RB + D) // 2 + g[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    x = (tl.reshape(c, (R * G, 32)) + 0.5) * tl.reshape(s, (R * G, 1)) * 0.0078125
    x = h32(x, M=R * G)
    tl.store(X + rows[:, None] * D + d[None, :], tl.reshape(x, (R, D)).to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _rotate(X, OUT, n, D: tl.constexpr, R: tl.constexpr):
    """R rows of D bf16 values -> H32 over each group of 32 in fp32, then bf16 (a query in the cache's rotation)."""

    rows = tl.program_id(0) * R + tl.arange(0, R)
    ok = rows < n
    d = tl.arange(0, D)
    x = tl.load(X + rows[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    x = h32(tl.reshape(x, (R * D // 32, 32)), M=R * D // 32)
    tl.store(OUT + rows[:, None] * D + d[None, :], tl.reshape(x, (R, D)).to(tl.bfloat16), mask=ok[:, None])


def rotate(x: torch.Tensor) -> torch.Tensor:
    """bf16 ``[..., head_dim]`` -> its H32 rotation by groups of 32 (fp32 sums, one rounding), each row on its own."""

    if x.dtype != torch.bfloat16 or x.shape[-1] % GROUP or not x.is_cuda:
        raise ValueError("rotate takes CUDA bf16 rows whose length is a multiple of 32")
    x = x.contiguous()
    d = x.shape[-1]
    n = x.numel() // d
    out = torch.empty_like(x)
    if n:
        _rotate[(triton.cdiv(n, ROWS),)](x, out, n, D=d, R=ROWS, num_warps=4)
    return out


def pack(x: torch.Tensor) -> torch.Tensor:
    """bf16 ``[N, kv_heads, head_dim]`` -> int8 ``[N, kv_heads, row_bytes(head_dim)]``; each row on its own."""

    if x.dtype != torch.bfloat16 or x.dim() != 3 or not x.is_cuda:
        raise ValueError("pack takes a CUDA bf16 [rows, kv_heads, head_dim] tensor")
    x = x.contiguous()
    n, hk, d = x.shape
    rb = row_bytes(d)
    out = torch.empty((n, hk, rb), dtype=torch.int8, device=x.device)
    if n * hk:
        _pack[(triton.cdiv(n * hk, ROWS),)](x, out, out.view(torch.float16), n * hk, D=d, RB=rb, R=ROWS, num_warps=4)
    return out


def unpack(p: torch.Tensor, head_dim: int) -> torch.Tensor:
    """int8 rows from ``pack`` -> bf16 ``[N, kv_heads, head_dim]`` in the model's basis (what a prompt's attention reads)."""

    rb = row_bytes(head_dim)
    if p.dtype != torch.int8 or p.dim() != 3 or p.shape[2] != rb or not p.is_contiguous():
        raise ValueError(f"unpack takes contiguous int8 [rows, kv_heads, {rb}] rows")
    n, hk, _ = p.shape
    out = torch.empty((n, hk, head_dim), dtype=torch.bfloat16, device=p.device)
    if n * hk:
        _unpack[(triton.cdiv(n * hk, ROWS),)](p, p.view(torch.float16), out, n * hk, D=head_dim, RB=rb, R=ROWS,
                                              num_warps=4)
    return out
