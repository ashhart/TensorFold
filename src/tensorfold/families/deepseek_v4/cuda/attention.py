"""DeepSeek's shared KV window, compressed pools and learned indexer on TensorFold CUDA."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _index(Q, W, POOL, OUT, START, P, HEADS: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.arange(0, HEADS)
    d = tl.arange(0, D)
    q = tl.load(Q + (r * HEADS + h[:, None]) * D + d[None, :])
    keys = tl.load(POOL + n[None, :].to(tl.int64) * D + d[:, None], n[None, :] < P, 0)
    scores = tl.maximum(tl.dot(q, keys), 0.0) * (D**-0.5)
    weights = tl.load(W + r * HEADS + h).to(tl.float32)
    scores = tl.sum(scores * weights[:, None], 0)
    visible = (START + r + 1) // 4
    tl.store(OUT + r * P + n, tl.where(n < visible, scores, -float("inf")), n < P)


def select(q, weights, pool, start):
    rows, heads, dims = q.shape
    count = pool.shape[0]
    scores = torch.empty((rows, count), device=q.device, dtype=torch.float32)
    _index[(rows, triton.cdiv(count, 128))](
        q.contiguous(), weights.contiguous(), pool, scores, start, count, heads, dims, 128, enable_fp_fusion=False
    )
    # Ascending pool ids give attention a fixed key order, independent of score ordering.
    return scores.topk(min(512, count), dim=1).indices.sort(dim=1).values.to(torch.int32)


@triton.jit
def _attend(
    Q,
    RAW,
    POOL,
    PICKS,
    SINK,
    OUT,
    START,
    RAWBASE,
    RAWROWS,
    POOLS,
    RATIO: tl.constexpr,
    PICKED: tl.constexpr,
    PICK_WIDTH: tl.constexpr,
    DRAFT: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, DIM)
    q = tl.load(Q + (row * HEADS + h[:, None]) * DIM + d[None, :], h[:, None] < HEADS, 0)
    sink = tl.load(SINK + h, h < HEADS, -float("inf")).to(tl.float32)
    maximum = sink
    total = tl.full((BH,), 1.0, tl.float32)
    acc = tl.zeros((BH, DIM), tl.float32)
    if DRAFT:
        low = 0
        high = RAWROWS
    else:
        low = tl.maximum(0, START + row - WINDOW + 1 - RAWBASE)
        high = START + row + 1 - RAWBASE
    j = tl.arange(0, BK)
    # Loop only the current raw window, even during a large prompt chunk.
    for b in range(tl.cdiv(high - low, BK)):
        idx = low + b * BK + j
        good = (idx >= 0) & (idx < high) & (idx < RAWROWS)
        k = tl.load(RAW + idx[None, :].to(tl.int64) * DIM + d[:, None], good[None, :], 0)
        logits = tl.dot(q, k) * (DIM**-0.5)
        logits = tl.where(good[None, :], logits, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(logits, 1))
        alpha = tl.exp(maximum - newmax)
        prob = tl.exp(logits - newmax[:, None])
        acc = acc * alpha[:, None] + (
            tl.dot(prob.to(tl.bfloat16), tl.trans(k))
            + tl.dot((prob - prob.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), tl.trans(k))
        )
        total = total * alpha + tl.sum(prob, 1)
        maximum = newmax
    if RATIO:
        visible = (START + row + 1) // RATIO
        count = tl.minimum(visible, PICK_WIDTH) if PICKED else visible
        for b in range(tl.cdiv(count, BK)):
            idx = b * BK + j
            pick = tl.load(PICKS + row * PICK_WIDTH + idx, idx < count, 0) if PICKED else idx
            good = (idx < count) & (pick >= 0) & (pick < visible) & (pick < POOLS)
            k = tl.load(POOL + pick[None, :].to(tl.int64) * DIM + d[:, None], good[None, :], 0)
            logits = tl.dot(q, k) * (DIM**-0.5)
            logits = tl.where(good[None, :], logits, -float("inf"))
            newmax = tl.maximum(maximum, tl.max(logits, 1))
            alpha = tl.exp(maximum - newmax)
            prob = tl.exp(logits - newmax[:, None])
            acc = acc * alpha[:, None] + (
                tl.dot(prob.to(tl.bfloat16), tl.trans(k))
                + tl.dot((prob - prob.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), tl.trans(k))
            )
            total = total * alpha + tl.sum(prob, 1)
            maximum = newmax
    tl.store(OUT + (row * HEADS + h[:, None]) * DIM + d[None, :], acc / total[:, None], h[:, None] < HEADS)


def attend(q, raw, pool, picks, sink, *, start, rawbase, ratio, window=128, draft=False):
    if q.ndim != 3 or raw.ndim != 2 or q.shape[-1] != 512 or q.shape[1] != 64 or raw.shape[1] != 512:
        raise ValueError("DeepSeek attention requires query [rows,64,512] and raw keys [keys,512]")
    rows, heads, dims = q.shape
    if start < rawbase or rawbase < 0 or start + rows > rawbase + raw.shape[0] or ratio not in (0, 4, 128):
        raise ValueError("attention raw span or compression ratio is invalid")
    if picks is not None and (picks.ndim != 2 or picks.shape[0] != rows or picks.dtype != torch.int32):
        raise ValueError("attention picks must be int32 [rows,slots]")
    out = torch.empty_like(q)
    picked = picks is not None
    head_rows = 32 if rows > 16 else 16
    _attend[(rows, triton.cdiv(heads, head_rows))](
        q.contiguous(),
        raw.contiguous(),
        pool if pool is not None else raw,
        picks if picked else raw,
        sink,
        out,
        start,
        rawbase,
        raw.shape[0],
        pool.shape[0] if pool is not None else 0,
        ratio,
        picked,
        picks.shape[1] if picked else 0,
        draft,
        heads,
        dims,
        window,
        head_rows,
        32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out


def frequencies(meta, ratio, device):
    dims = int(meta["deepseek4.rope.dimension_count"])
    base = float(meta["deepseek4.attention.compress_rope_freq_base"] if ratio else meta["deepseek4.rope.freq_base"])
    freq = 1 / (base ** (torch.arange(0, dims, 2, device=device, dtype=torch.float32) / dims))
    factor = float(meta.get("deepseek4.rope.scaling.factor", 1))
    original = int(meta.get("deepseek4.rope.scaling.original_context_length", 0))
    if ratio and original > 0 and factor > 1:
        correction = lambda rotations: dims * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(float(meta.get("deepseek4.rope.scaling.yarn_beta_fast", 32)))), 0)
        high = min(math.ceil(correction(float(meta.get("deepseek4.rope.scaling.yarn_beta_slow", 1)))), dims - 1)
        if low == high:
            high += 0.001
        smooth = 1 - ((torch.arange(dims // 2, device=device) - low) / (high - low)).clamp(0, 1)
        freq = freq / factor * (1 - smooth) + freq * smooth
    return freq


def rope(x, positions, freq, inverse=False):
    half = freq.numel()
    angles = positions.float()[:, None] * freq[None]
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sin = -sin
    pairs = x[..., -2 * half :].float().reshape(*x.shape[:-1], half, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    tail = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)
    return torch.cat((x[..., : -2 * half], tail), -1)


@triton.jit
def _norm_rope(
    X,
    W,
    POS,
    COS,
    SIN,
    OUT,
    D: tl.constexpr,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    WEIGHTED: tl.constexpr,
    INVERSE: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0)
    d = tl.arange(0, B)
    v = tl.load(X + r * D + d, d < D, 0).to(tl.float32)
    if NORM:
        inv = 1.0 / tl.sqrt(tl.sum(v * v, 0) / D + EPS)
        v = (v * inv).to(tl.bfloat16).to(tl.float32)
        if WEIGHTED:
            w = tl.load(W + d, d < D, 0).to(tl.float32)
            v = (v * w).to(tl.bfloat16).to(tl.float32)
    other = tl.gather(v, d ^ 1, 0)
    rot = d >= D - 2 * HALF
    slot = (d - (D - 2 * HALF)) // 2
    pos = tl.load(POS + r // HEADS).to(tl.int64)
    c = tl.load(COS + pos * HALF + slot, rot & (d < D), 0)
    s = tl.load(SIN + pos * HALF + slot, rot & (d < D), 0)
    if INVERSE:
        s = -s
    rotated = tl.where(d % 2 == 0, v * c - other * s, v * c + other * s)
    tl.store(OUT + r * D + d, tl.where(rot, rotated, v), d < D)


def norm_rope(x, positions, tables, weight=None, eps=1e-6, *, normalize=True, inverse=False):
    rows = x.shape[0]
    heads = 1 if x.ndim == 2 else x.shape[1]
    dims = x.shape[-1]
    cos, sin = tables
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _norm_rope[(rows * heads,)](
        x.contiguous(),
        weight if weight is not None else x,
        positions,
        cos,
        sin,
        out,
        dims,
        heads,
        cos.shape[1],
        eps,
        normalize,
        weight is not None,
        inverse,
        triton.next_power_of_2(dims),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
