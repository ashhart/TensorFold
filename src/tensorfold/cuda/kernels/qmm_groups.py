"""4-bit g64 projections on ROCm in 16-output tiles: a tile's words for every group sit together, so a warp streams
its K range from contiguous memory, where MLX's row-major words scatter 32 bytes a row. The lane matmul keeps the
stored layout's arithmetic (fp32 group dots, ``acc + p * s + xs * b`` in group order, shape-fixed K slices), so a
row's bits never depend on the row count; prompts round each weight to bf16 once and run one fp32 chain over K in
fixed tiles, so chunking never changes a row's bits either (not decode's bits, as on NVIDIA).

Layout: words (ceil(N/16), K/64, 16, 8) int32 (tile t, group g, output 16t + c: the stored row's 8 words of that
group, each word's inputs 2j and 2j + 1 at bits 4j and 16 + 4j, see ``SHIFT``), scales and biases
(ceil(N/16), K/64, 16); outputs past N are zero. Calls take N itself."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


def _ints(name: str, default: str, count: int) -> tuple[int, ...]:
    values = tuple(int(v) for v in os.environ.get(name, default).split(","))
    if len(values) != count or min(values) < 1:
        raise ValueError(f"{name}: {count} positive integers, comma separated")
    return values


@lru_cache(maxsize=1)
def lane_config() -> tuple[int, int, int, int, int]:
    """(BN, warps up to 32 rows, warps past, stages, split target); ``TF_ROCM_QMM`` overrides it for tuning. The split
    target sets the K slices per weight shape, and warp counts can change a dot's bits on gfx1201 (16-row tiles on 8
    warps do): the defaults give a row the same bits at every row count, as the ROCm tests check."""

    return _ints("TF_ROCM_QMM", "64,4,8,3,192", 5)


@lru_cache(maxsize=1)
def prefill_config() -> tuple[int, int, int, int, int]:
    """(BM, BN, BK, warps, stages) of the prompt GEMM; ``TF_ROCM_PREFILL`` overrides it (bits follow the tile). On
    gfx1201 with Triton 3.6, (128, 256, 64), (256, 128, 64) and (128, 128, 32) tiles give wrong sums at K=128 with
    256 or more outputs; this one is right at every shape tested and the fastest (109 TFLOPS with the dequant)."""

    bm, bn, bk, warps, stages = _ints("TF_ROCM_PREFILL", "128,128,64,8,1", 5)
    if bk % 16 or min(bm, bn) < 16:
        raise ValueError("TF_ROCM_PREFILL: BK a multiple of 16, BM and BN at least 16")
    return bm, bn, bk, warps, stages


SHIFT = tuple(4 * (i // 2) + 16 * (i % 2) for i in range(8))     # bit of a word's input i in the tiled layout


def _renibble(words: torch.Tensor, src: tuple[int, ...], dst: tuple[int, ...]) -> torch.Tensor:
    """Each int32 word's nibble at bit src[i] moved to bit dst[i], in place, a slab of tiles at a time (the head's
    words are 0.6 GB)."""

    for part in words.split(2048):
        out = torch.zeros_like(part)
        for a, b in zip(src, dst):
            out |= ((part >> a) & 0xF) << b
        part.copy_(out)
    return words


def to_groups(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor):
    """MLX (N, K/8) words, (N, K/64) scales and biases -> the tiled copies (N padded to 16 with zeros)."""

    n, k8 = weight.shape
    pad = -n % 16
    words, scales, biases = weight.view(torch.int32), scales, biases
    if pad:
        words = torch.cat([words, words.new_zeros((pad, k8))])
        scales = torch.cat([scales, scales.new_zeros((pad, scales.shape[1]))])
        biases = torch.cat([biases, biases.new_zeros((pad, biases.shape[1]))])
    t, kg = (n + pad) // 16, k8 // 8
    words = words.reshape(t, 16, kg, 8).permute(0, 2, 1, 3).contiguous()
    return (_renibble(words, tuple(range(0, 32, 4)), SHIFT),
            scales.reshape(t, 16, kg).permute(0, 2, 1).contiguous(),
            biases.reshape(t, 16, kg).permute(0, 2, 1).contiguous())


def from_groups(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int):
    """The stored MLX layout of the first ``n`` outputs again."""

    t, kg, _, _ = words.shape
    words = _renibble(words.clone(), SHIFT, tuple(range(0, 32, 4)))
    return (words.permute(0, 2, 1, 3).reshape(t * 16, kg * 8)[:n].contiguous(),
            scales.permute(0, 2, 1).reshape(t * 16, kg)[:n].contiguous(),
            biases.permute(0, 2, 1).reshape(t * 16, kg)[:n].contiguous())


def bucket(m: int) -> int:
    """The row tile: 16, 32 or 64 rows, else 128-row tiles side by side (tiles never change a row's bits)."""

    if m < 1:
        raise ValueError("the lane matmul takes at least one row")
    for b in (16, 32, 64):
        if m <= b:
            return b
    return 128


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: a function of the shape only, never of the row count."""

    bn, _, _, _, target = lane_config()
    tiles, groups, sk = -(-n // bn), k // 64, 1
    while sk < 8 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


@triton.jit
def _shifts():
    """``SHIFT``: the bit of each of a word's 8 inputs."""

    i = tl.arange(0, 8)
    return 4 * (i // 2) + 16 * (i % 2)


@triton.jit
def _group_sums(X, XS, ldx, KG: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    ok = g < KG
    x = tl.load(X + m * ldx + g[:, None] * 64 + tl.arange(0, 64)[None, :], mask=ok[:, None], other=0.0)
    tl.store(XS + m * KG + g, tl.sum(x.to(tl.float32), axis=1), mask=ok)


def group_sums(x: torch.Tensor) -> torch.Tensor:
    """(M, K) bf16 -> (M, K/64) fp32 sums of each 64-input group."""

    m, k = x.shape
    xs = torch.empty((m, k // 64), dtype=torch.float32, device=x.device)
    _group_sums[(m, triton.cdiv(k // 64, 16))](x, xs, x.stride(0), KG=k // 64, GB=16, num_warps=2)
    return xs


@triton.jit
def _lane(X, XS, W, S, B, OUT, PART, M, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
          BLOCK_N: tl.constexpr):
    KG: tl.constexpr = K // 64
    PER: tl.constexpr = KG // SK
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = _shifts()
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER):
        g = pid_s * PER + i
        x = tl.load(X + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        at = ((rn // 16) * KG + g) * 16 + rn % 16
        words = tl.load(W + at[:, None] * 8 + rw[None, :], mask=n_ok[:, None], other=0)
        q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(q, (BLOCK_N, 64)).to(tl.bfloat16)
        p = tl.dot(x, tl.trans(q))
        s = tl.load(S + at, mask=n_ok, other=0.0).to(tl.float32)
        b = tl.load(B + at, mask=n_ok, other=0.0).to(tl.float32)
        xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
        acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=ok)


@lru_cache(maxsize=1)
def _rocm():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qmm_rocm_v15", sources=[str(here / "qmm_rocm.cpp"), str(here / "qmm_rocm.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


@lru_cache(maxsize=1)
def lane_kernel() -> str:
    """The ROCm lane matmul: ``qmm_rocm.cu``'s ``wmma`` (default: 20-28% under Triton at 12 rows on the 27B's large
    projections, half its per-call cost on small ones), its ``dot2`` (bf16 dot instructions: fastest at one row,
    compute bound at twelve) or ``triton``; ``TF_ROCM_LANE`` picks it. Their bits differ, so a process uses one for
    every lane call."""

    kind = os.environ.get("TF_ROCM_LANE", "wmma")
    if kind not in ("triton", "dot2", "wmma"):
        raise ValueError("TF_ROCM_LANE: triton, dot2 or wmma")
    return kind


def gemv(x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int,
         xs: torch.Tensor | None = None, *, f32: bool = False, wmma: bool = False) -> torch.Tensor:
    """The HIP decode matmul: rows in 16-row passes, K slices added in order."""

    kg = words.shape[1]
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()                                   # rows are staged in 16-byte pieces
    m = x.shape[0]
    if xs is None:
        xs = group_sums(x)
    ext = _rocm()
    fill = wmma_fill()
    slices = _slices(kg, n, wmma, fill)
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    part = (torch.empty((slices * min(m, 32 if wmma else 16) * n,), dtype=torch.float32, device=x.device)
            if slices > 1 or f32 else out)                 # unread when one slice writes its output directly
    ext.gemv_groups(x, xs.contiguous(), words, scales, biases, n, out, part, wmma, fill, _counts(x.device, n))
    return out


_arrivals: dict[torch.device, torch.Tensor] = {}


def _counts(device: torch.device, n: int) -> torch.Tensor:
    """The WMMA kernel's split-K arrival counters, one per 128 outputs: zeros, and zeros again after every call (the
    last block of a column resets it), so one stream's calls share them."""

    need = -(-n // 128)
    buf = _arrivals.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.zeros(max(need, 4096), dtype=torch.int32, device=device)
        _arrivals[device] = buf
    return buf


@lru_cache(maxsize=1)
def wmma_fill() -> int:
    """Blocks a WMMA K split aims for (``TF_ROCM_WMMA_FILL``); the split is a function of the weight's shape only.
    64 read every 27B projection at 12 rows as fast as 128 or faster (its narrow ones 5-12% faster on an R9700)."""

    return int(os.environ.get("TF_ROCM_WMMA_FILL", "64"))


@lru_cache(maxsize=None)
def _slices(kg: int, n: int, wmma: bool, fill: int) -> int:
    ext = _rocm()
    return ext.wmma_slices(kg, n, fill) if wmma else ext.gemv_slices(kg)


def reload_settings() -> None:
    """Read the ``TF_ROCM_*`` tuning variables again (cached: the lane matmul runs hundreds of times a step)."""

    for f in (lane_config, lane_kernel, prefill_config, prefill8_config, prefill8_group, wmma_fill):
        f.cache_clear()


def matmul(x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int,
           xs: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
    """x (M, K) bf16 times the tiled weight's first ``n`` outputs -> (M, n) bf16 (fp32 with ``f32``)."""

    kg = words.shape[1]
    k = kg * 64
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"lane matmul: x must be (M, {k}) bf16")
    kind = lane_kernel()
    if kind != "triton":
        return gemv(x, words, scales, biases, n, xs, f32=f32, wmma=kind == "wmma")
    x = x.contiguous()
    m = x.shape[0]
    if xs is None:
        xs = group_sums(x)
    bn, small, large, stages, _ = lane_config()
    bm = bucket(m)
    sk = split_k(n, k)
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _lane[(triton.cdiv(m, bm), triton.cdiv(n, bn), sk)](x, xs, words, scales, biases, out, part, m, N=n, K=k, SK=sk,
                                                        BM=bm, BLOCK_N=bn, num_warps=small if bm <= 32 else large,
                                                        num_stages=stages)
    if sk > 1:
        total = m * n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _dequant(W, S, B, OUT, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[n, 64g:64g+64] = bf16(q * s + b), one program per (group, block of outputs)."""

    KG: tl.constexpr = K // 64
    g = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rn < N
    at = ((rn // 16) * KG + g) * 16 + rn % 16
    words = tl.load(W + at[:, None] * 8 + tl.arange(0, 8)[None, :], mask=ok[:, None], other=0)
    q = ((words[:, :, None] >> _shifts()[None, None, :]) & 0xF).to(tl.float32)
    s = tl.load(S + at, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(B + at, mask=ok, other=0.0).to(tl.float32)
    w = tl.reshape(q, (BLOCK_N, 64)) * s[:, None] + b[:, None]
    tl.store(OUT + rn[:, None] * K + g * 64 + tl.arange(0, 64)[None, :], w.to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _gemm(X, W, OUT, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = x @ w.T in fixed (BM, BN) tiles, one fp32 chain over K in BK steps."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


_scratch: dict[torch.device, torch.Tensor] = {}


def _buffer(device: torch.device, nbytes: int) -> torch.Tensor:
    """A byte buffer reused by the next prompt matmul on this device (one stream's prompt matmuls run in order, so a
    projection's weights are consumed before the next one overwrites them)."""

    buf = _scratch.get(device)
    if buf is None or buf.numel() < nbytes:
        _scratch.pop(device, None)
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _scratch[device] = buf
    return buf[:nbytes]


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int) -> torch.Tensor:
    """The (n, K) bf16 weight, in the reused prompt buffer."""

    kg = words.shape[1]
    k = kg * 64
    out = _buffer(words.device, 2 * n * k).view(torch.bfloat16).view(n, k)
    _dequant[(kg, triton.cdiv(n, 64))](words, scales, biases, out, N=n, K=k, BLOCK_N=64, num_warps=4)
    return out


def prefill_matmul(x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int, *,
                   f32: bool = False) -> torch.Tensor:
    """Prompt rows times the tiled weight: bf16 weights once, then a fixed-tile GEMM; any chunking, same bits."""

    kg = words.shape[1]
    k = kg * 64
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"prefill matmul: x must be (M, {k}) bf16")
    x = x.contiguous()
    m = x.shape[0]
    w = dequantize(words, scales, biases, n)
    bm, bn, bk, warps, stages = prefill_config()
    if k % bk:
        bk = 64
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](x, w, out, m, N=n, K=k, BM=bm, BN=bn, BK=bk, num_warps=warps,
                                                    num_stages=stages)
    return out


@triton.jit
def _src(BLOCK: tl.constexpr):
    """``prefill_glue``'s stored order: position m of each 32-input block holds input src(m)."""

    m = tl.arange(0, BLOCK)
    j = m % 4
    return (m // 32) * 32 + ((m % 32) // 16) * 16 + ((m % 16) // 4) * 2 + (j % 2) + (j // 2) * 8


@triton.jit
def _nibbles8(W, OUT, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    """OUT[n, 64g + m] = e4m3(q[n, 64g + src(m)]): the 4-bit codes, exact in e4m3, in the inputs' stored order."""

    KG: tl.constexpr = K // 64
    g = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rn < N
    src = _src(64)
    at = ((rn // 16) * KG + g) * 16 + rn % 16
    words = tl.load(W + at[:, None] * 8 + (src // 8)[None, :], mask=ok[:, None], other=0)
    i = src % 8
    q = (words >> (4 * (i // 2) + 16 * (i % 2))[None, :]) & 0xF
    tl.store(OUT + rn[:, None] * K + g * 64 + tl.arange(0, 64)[None, :],
             q.to(tl.float32).to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=ok[:, None])


@triton.jit
def _gemm8(X8, XS, A, W8, S, B, OUT, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           KB: tl.constexpr, GROUP: tl.constexpr):
    """out = a * (sum over groups of dot(x8, w8) * s + xs . b) in fixed (BM, BN) tiles, groups in order; tiles run
    GROUP row blocks at a time down each column block, so their inputs and weights meet in cache."""

    KG: tl.constexpr = K // 64
    pid = tl.program_id(0)
    blocks_m = tl.cdiv(M, BM)
    blocks_n: tl.constexpr = (N + BN - 1) // BN
    per = GROUP * blocks_n
    first = (pid // per) * GROUP
    size = tl.minimum(blocks_m - first, GROUP)
    rm = (first + (pid % per) % size) * BM + tl.arange(0, BM)
    rn = ((pid % per) // size) * BN + tl.arange(0, BN)
    rk = tl.arange(0, 64)
    m_ok = rm < M
    n_ok = rn < N
    sat = (rn // 16) * KG * 16 + rn % 16
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(KG):
        x = tl.load(X8 + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0)
        w = tl.load(W8 + rn[:, None] * K + (g * 64 + rk)[None, :], mask=n_ok[:, None], other=0)
        x = x.to(tl.float8e4nv, bitcast=True)
        w = w.to(tl.float8e4nv, bitcast=True)
        p = tl.dot(x, tl.trans(w))
        s = tl.load(S + sat + g * 16, mask=n_ok, other=0.0).to(tl.float32)
        acc = acc + p * s[None, :]
    rb = tl.arange(0, KB)
    for kb in range(0, KG, KB):
        xs = tl.load(XS + rm[:, None] * KG + (kb + rb)[None, :], mask=m_ok[:, None] & (kb + rb < KG)[None, :],
                     other=0.0)
        b = tl.load(B + sat[None, :] + ((kb + rb) * 16)[:, None], mask=n_ok[None, :] & (kb + rb < KG)[:, None],
                    other=0.0)
        acc = tl.dot(xs, b, acc)
    a = tl.load(A + rm, mask=m_ok, other=0.0)
    tl.store(OUT + rm[:, None] * N + rn[None, :], (acc * a[:, None]).to(OUT.dtype.element_ty),
             mask=m_ok[:, None] & n_ok[None, :])


@lru_cache(maxsize=1)
def prefill8_group() -> int:
    """Row blocks a column block's tiles take together (``TF_ROCM_PREFILL8_GROUP``); scheduling only, never bits."""

    return _ints("TF_ROCM_PREFILL8_GROUP", "8", 1)[0]


@lru_cache(maxsize=1)
def prefill8_config() -> tuple[int, int, int, int]:
    """(BM, BN, warps, stages) of the FP8 prompt GEMM; ``TF_ROCM_PREFILL8`` overrides it (bits follow the tile). With
    Triton 3.6 on gfx1201, pipelined tiles (stages 2 or 3) give wrong sums for calls of a few rows at some K; single
    stage tiles are exact, and this one led them at the 27B's shapes (121-151 TFLOPS against 81-115 for bf16)."""

    bm, bn, warps, stages = _ints("TF_ROCM_PREFILL8", "128,128,4,1", 4)
    if min(bm, bn) < 16:
        raise ValueError("TF_ROCM_PREFILL8: BM and BN at least 16")
    return bm, bn, warps, stages


def prefill_matmul8(x: tuple[torch.Tensor, torch.Tensor, torch.Tensor], words: torch.Tensor, scales: torch.Tensor,
                    biases: torch.Tensor, n: int, *, f32: bool = False) -> torch.Tensor:
    """``prefill_glue``'s e4m3 rows (with group sums and row scales) times the tiled weight, its codes exact in e4m3:
    NVIDIA's FP8 prompt arithmetic (``qmm.prefill_matmul8``) in fixed tiles; any chunking gives a row the same bits."""

    x8, xs, a = x
    kg = words.shape[1]
    k = kg * 64
    if x8.dtype != torch.uint8 or x8.dim() != 2 or x8.shape[1] != k or xs.shape != (x8.shape[0], kg):
        raise ValueError(f"prefill matmul8: e4m3 rows (M, {k}) with (M, {kg}) group sums")
    m = x8.shape[0]
    w8 = _buffer(words.device, n * k).view(n, k)
    _nibbles8[(kg, triton.cdiv(n, 64))](words, w8, N=n, K=k, BLOCK_N=64, num_warps=4)
    bm, bn, warps, stages = prefill8_config()
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x8.device)
    _gemm8[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](x8, xs, a, w8, scales, biases, out, m, N=n, K=k, BM=bm, BN=bn,
                                                       KB=16, GROUP=prefill8_group(), num_warps=warps,
                                                       num_stages=stages)
    return out


__all__ = ["bucket", "dequantize", "from_groups", "gemv", "group_sums", "lane_config", "lane_kernel", "matmul",
           "prefill8_config", "prefill8_group", "prefill_config", "prefill_matmul", "prefill_matmul8",
           "reload_settings", "split_k", "to_groups", "wmma_fill"]
