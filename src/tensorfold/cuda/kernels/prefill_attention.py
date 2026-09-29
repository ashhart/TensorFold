"""Prefill attention in 64-key tiles by absolute position, so chunking never changes bits; not decode's arithmetic.

``_attend`` (Triton) defines the bits; ``prefill_attention.cu`` computes the same bits with a KV group's query heads
in one block, so each staged key and value serves all of them (checked equal in tests/cuda/test_prefill_attention.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from tensorfold.cuda.build import hip

ROWS2 = 192              # ROCm: prompt rows from which a block takes two 16-row tiles (128: one was faster)
BM = 64
BN = 64


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    s = tl.where(valid, s, float("-inf"))
    tile_m = tl.max(s, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _attend(Q, K, V, OUT, p0, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, SCALE: tl.constexpr, ONE_LOOP: tl.constexpr, HEADS_FIRST: tl.constexpr):
    if HEADS_FIRST:                               # a KV head's query heads run side by side and share its tiles in L2
        head = tl.program_id(0)
        block = tl.num_programs(1) - 1 - tl.program_id(1)          # longest causal blocks first
    else:
        block = tl.program_id(0)
        head = tl.program_id(1)
    hk = head // (H // HK)
    rows = block * BM + tl.arange(0, BM)
    ok = rows < W
    pos = p0 + rows
    d = tl.arange(0, D)
    q = tl.load(Q + (rows[:, None] * H + head) * D + d[None, :], mask=ok[:, None], other=0.0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    first_pos = p0 + block * BM
    last_pos = p0 + tl.minimum(block * BM + BM, W) - 1
    full = (first_pos + 1) // BN                  # tiles every row of the block sees whole
    if ONE_LOOP:                                  # ROCm: one loop, so a tile's code never depends on the chunk start
        full = 0
    for t in range(0, full):
        keys = t * BN + tl.arange(0, BN)
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :])
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :])
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        seen = keys <= last_pos
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    out = o / l[:, None]
    tl.store(OUT + (rows[:, None] * H + head) * D + d[None, :], out.to(tl.bfloat16), mask=ok[:, None])


def attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float) -> torch.Tensor:
    """q (W, H, D) bf16 at positions [p0, p0 + W); the caches must already hold every key through p0 + W - 1."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    out = torch.empty_like(q)
    if hip():
        if _rocm_kernel(h, hk, d):
            _rocm().attention(q, k_cache, v_cache, out, p0, scale, *rocm_rows(w))
            return out
        return _rocm_attention(q, k_cache, v_cache, out, p0, scale)
    if d == 64:
        return triton_attention(q, k_cache, v_cache, p0, scale=scale, out=out)
    _ext().prefill_attention(q, k_cache, v_cache, out, p0, scale, heads_a_block(h // hk))
    return out


def triton_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """The same through ``_attend``: the bits' definition, one program a 64-row block and query head."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    out = torch.empty_like(q) if out is None else out
    _attend[(triton.cdiv(w, BM), h)](q, k_cache, v_cache, out, p0, w, H=h, HK=hk, D=d, BM=BM, BN=BN, SCALE=scale,
                                     ONE_LOOP=False, HEADS_FIRST=False, num_warps=8, num_stages=1 if d > 128 else 2)
    return out


def heads_a_block(group: int) -> int:
    """Query heads of one KV head in a CUDA block of eight warps: the largest of 8, 4, 2, 1 dividing the group."""

    return next(n for n in (8, 4, 2, 1) if group % n == 0)


def _check(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int) -> int:
    w, h, d = q.shape
    hk = k_cache.shape[1]
    if k_cache.shape[0] < p0 + w or v_cache.shape != k_cache.shape or h % hk or d not in (64, 128, 256):
        raise ValueError("prefill attention: caches must hold the chunk's keys; heads a multiple of kv heads")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("prefill attention takes contiguous tensors")
    return hk


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_prefill_attention_v1", sources=[str(here / "prefill_attention.cpp"),
                                                                 str(here / "prefill_attention.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def _rocm_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, out: torch.Tensor, p0: int,
                    scale: float) -> torch.Tensor:
    """ROCm: ``_attend`` in one loop over ``rocm_config``'s tiles (``prefill_attention.cu`` is NVIDIA's)."""

    w, h, d = q.shape
    bm, bn, warps, stages, first = rocm_config()
    grid = (h, triton.cdiv(w, bm)) if first else (triton.cdiv(w, bm), h)
    _attend[grid](q, k_cache, v_cache, out, p0, w, H=h, HK=k_cache.shape[1], D=d, BM=bm, BN=bn, SCALE=scale,
                  ONE_LOOP=True, HEADS_FIRST=bool(first), num_warps=warps, num_stages=stages)
    return out


def rocm_config() -> tuple[int, int, int, int, int]:
    """(BM, BN, warps, stages, heads first) on ROCm; ``TF_ROCM_ATTN`` overrides it for tuning. A row's bits depend on
    the setting (BN sets the key tiles), never on how the prompt is chunked."""

    import os

    values = tuple(int(v) for v in os.environ.get("TF_ROCM_ATTN", "128,16,8,1,1").split(","))
    if len(values) != 5 or min(values[:4]) < 1:
        raise ValueError("TF_ROCM_ATTN: BM,BN,warps,stages,heads_first")
    return values


def _rocm():
    from tensorfold.cuda.kernels.attention import _rocm as extension     # prompt and tree attention: one extension

    return extension()


def rocm_rows(w: int) -> tuple[int, bool]:
    """(16-row tiles a block, loader waves or not) for ``attention_rocm.cu``'s prompt kernel: two tiles and loaders
    from ``ROWS2`` prompt rows, one tile below it (twice the blocks). Never a row's bits; ``TF_ROCM_ATTN_RB`` and
    ``TF_ROCM_ATTN_PIPE`` override them for tuning."""

    import os

    rb = int(os.environ.get("TF_ROCM_ATTN_RB") or (2 if w >= ROWS2 else 1))
    return rb, os.environ.get("TF_ROCM_ATTN_PIPE", "1") != "0"


def _rocm_kernel(heads: int, kv_heads: int, dim: int) -> bool:
    """ROCm's WMMA prompt attention (``attention_rocm.cu``) where it applies, unless ``TF_ROCM_ATTN_KERNEL=triton``;
    the two give different bits, so a process uses one."""

    import os

    return os.environ.get("TF_ROCM_ATTN_KERNEL", "wmma") != "triton" and _rocm().supported(heads, kv_heads, dim)
