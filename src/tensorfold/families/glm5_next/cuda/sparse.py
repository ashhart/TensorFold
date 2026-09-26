"""DSA's sparse top-k attention for GLM-5.3-Flash contexts past 2051 tokens (the Hugging Face definition).

Each DSA layer's indexer keeps, per token, a key k = LayerNorm(wk x) and a pool gate g = x . compress_gate (both
bf16, 128 wide). Every 4 tokens form a pool with key bf16(sum_j bf16(bf16(softmax_j(g_j + ape_j)) * k_j)).
A query row at position q with index query qi = wq_b(q_resid) (32 heads of 128) and head weights
w = weights_proj(x) / sqrt(32) scores each complete pool ending at or before q:

    s_p = sum_h w_h * relu(qi_h . pool_p / sqrt(128))        (bf16 inputs, fp32 products and sums)

keeps the 512 best pools (ties to the lower pool), adds the visible tokens of the incomplete last pool, and
attends to those tokens only. While a row has at most 512 complete pools (q < 2051) it keeps them all, which is
dense causal attention (``attention.attention``). A row's selection and its attention depend only on that row,
so a window row gets the bits of the serial step at its position. Attention visits the selected tokens in
ascending position, 512 at a time, and merges the chunks in order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

POOL = 4
TOPK_POOLS = 512
BR = 16


@triton.jit
def _index_write(KR, k_stride, GR, LNW, LNB, IK, IG, POS, eps, D: tl.constexpr):
    """Row r: LayerNorm(k_raw) -> bf16 into IK[pos + r]; the gate row (fp32) -> bf16 into IG[pos + r]."""

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(KR + r * k_stride + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    y = xc / tl.sqrt(var + eps) * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)
    tl.store(IK + (P + r) * D + d, y.to(tl.bfloat16))
    tl.store(IG + (P + r) * D + d, tl.load(GR + r * D + d).to(tl.bfloat16))


@triton.jit
def _pool_keys(IK, IG, APE, PK, POS, R, D: tl.constexpr):
    """Program i: pool p = pos // 4 + i if it is complete within the window (ends at or before pos + R - 1)."""

    i = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    p = P // 4 + i
    if 4 * p + 3 > P + R - 1:
        return
    d = tl.arange(0, D)
    l0 = tl.load(IG + (4 * p + 0) * D + d).to(tl.float32) + tl.load(APE + 0 * D + d).to(tl.float32)
    l1 = tl.load(IG + (4 * p + 1) * D + d).to(tl.float32) + tl.load(APE + 1 * D + d).to(tl.float32)
    l2 = tl.load(IG + (4 * p + 2) * D + d).to(tl.float32) + tl.load(APE + 2 * D + d).to(tl.float32)
    l3 = tl.load(IG + (4 * p + 3) * D + d).to(tl.float32) + tl.load(APE + 3 * D + d).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    s = ((e0 + e1) + e2) + e3
    k0 = tl.load(IK + (4 * p + 0) * D + d).to(tl.float32)
    k1 = tl.load(IK + (4 * p + 1) * D + d).to(tl.float32)
    k2 = tl.load(IK + (4 * p + 2) * D + d).to(tl.float32)
    k3 = tl.load(IK + (4 * p + 3) * D + d).to(tl.float32)
    t0 = ((e0 / s).to(tl.bfloat16).to(tl.float32) * k0).to(tl.bfloat16).to(tl.float32)
    t1 = ((e1 / s).to(tl.bfloat16).to(tl.float32) * k1).to(tl.bfloat16).to(tl.float32)
    t2 = ((e2 / s).to(tl.bfloat16).to(tl.float32) * k2).to(tl.bfloat16).to(tl.float32)
    t3 = ((e3 / s).to(tl.bfloat16).to(tl.float32) * k3).to(tl.bfloat16).to(tl.float32)
    tl.store(PK + p * D + d, (((t0 + t1) + t2) + t3).to(tl.bfloat16))


def index_update(k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, ape: torch.Tensor,
                 ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor, pos: torch.Tensor) -> None:
    """Window rows' index keys and gates into the caches at pos.., then every pool the window completes."""

    R = k_raw.shape[0]
    _index_write[(R,)](k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, pos, 1e-6, D=128, num_warps=1)
    _pool_keys[(R // 4 + 2,)](ik, ig, ape, pk, pos, R, D=128, num_warps=1)


@triton.jit
def _scores(QI, W, w_stride, PK, OUT, POS, R, NP, scale, H: tl.constexpr, D: tl.constexpr, BP: tl.constexpr):
    """Program (r, pool block): s_p = sum_h w_h relu(scale * qi_h . pool_p) for pools p ending at or before
    pos + r (-inf after)."""

    r = tl.program_id(0)
    pb = tl.program_id(1)
    P = tl.load(POS)
    npool = (P + r + 1) // 4
    p = pb * BP + tl.arange(0, BP)
    d = tl.arange(0, D)
    hh = tl.arange(0, H)
    q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)           # [H, D]
    k = tl.load(PK + p[:, None] * D + d[None, :], mask=(p < npool)[:, None], other=0.0).to(tl.bfloat16)  # [BP, D]
    dots = tl.dot(q, tl.trans(k))                                                     # [H, BP] fp32
    w = tl.load(W + r * w_stride + hh).to(tl.float32) * (1.0 / 5.656854249492381)     # 32 ** -0.5
    s = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
    s = tl.where(p < npool, s, float("-inf"))
    tl.store(OUT + r * NP + p, s, mask=p < NP)


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, pos: int, R: int, np_max: int,
                  pos_dev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's attended tokens: [R, 2051] int32 ascending (-1 padded) and how many, for rows past 2050."""

    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    _scores[(R, triton.cdiv(np_max, 64))](qi, wts, wts.stride(0), pk, scores, pos_dev, R, np_max, 128 ** -0.5, H=32,
                                         D=128, BP=64, num_warps=4)
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :TOPK_POOLS]
    pools = torch.sort(order, dim=1).values                                            # ascending pool index
    width = TOPK_POOLS * POOL + POOL - 1
    tokens = torch.full((R, width), -1, dtype=torch.int32, device=qi.device)
    counts = []
    for r in range(R):
        q = pos + r
        npool = (q + 1) // POOL
        if npool <= TOPK_POOLS:               # dense row: every visible token (the dense kernel's result is kept)
            counts.append(0)
            continue
        body = (pools[r, :, None] * POOL + torch.arange(POOL, device=qi.device)).reshape(-1)
        tail = torch.arange(npool * POOL, q + 1, device=qi.device)
        row = torch.cat([body, tail])
        tokens[r, :row.numel()] = row.to(torch.int32)
        counts.append(row.numel())
    return tokens, torch.tensor(counts, dtype=torch.int32, device=qi.device)


@triton.jit
def _gtile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _sparse_chunks(Q, KC, VC, TOK, CNT, PO, PM, PL, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr):
    """Program (row, head, chunk): the row's selected tokens [c CH, (c + 1) CH) in list order; the query sits in
    tile row 0 of a 16-row tile (rows 1-15 idle), the 27B kernel's arithmetic per row."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    d = tl.arange(0, D)
    g = tl.arange(0, 16)
    q = tl.load(Q + (r * H + h) * D + d[None, :] + g[:, None] * 0, mask=(g == 0)[:, None], other=0).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for t in range(CH // 64):
        idx = c * CH + t * 64 + tl.arange(0, 64)
        ok = idx < n
        tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
        kk = tl.load(KC + (tok[:, None] * H + h) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        vv = tl.load(VC + (tok[:, None] * H + h) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        m, l, o = _gtile(q, kk, vv, m, l, o, ok, SCALE)
    base = (c * 128 + r) * H + h
    tl.store(PO + base * D + d[None, :] + g[:, None] * 0, o, mask=(g == 0)[:, None])
    tl.store(PM + base + g * 0, m, mask=g == 0)
    tl.store(PL + base + g * 0, l, mask=g == 0)


@triton.jit
def _sparse_merge(PO, PM, PL, OUT, CNT, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr, CH: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(CNT + r)
    if n == 0:
        return
    d = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        if c * CH < n:
            base = (c * 128 + r) * H + h
            cm = tl.load(PM + base)
            cl = tl.load(PL + base)
            co = tl.load(PO + base * D + d)
            active = cl > 0.0
            next_m = tl.where(active, tl.maximum(m, cm), m)
            a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            b = tl.where(active, tl.exp(cm - next_m), 0.0)
            o = o * a + co * b
            l = l * a + cl * b
            m = next_m
    tl.store(OUT + (r * H + h) * D + d, (o / l).to(tl.bfloat16))


def sparse_attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float) -> None:
    """Rows with counts > 0: attention over their selected tokens, written into ``out`` [R, H, D] (other rows
    untouched)."""

    R, H, D = q.shape
    W = tokens.shape[1]
    CH = 512
    nch = triton.cdiv(W, CH)
    po = torch.empty((nch * 128 * H * D,), dtype=torch.float32, device=q.device)
    pm = torch.empty((nch * 128 * H,), dtype=torch.float32, device=q.device)
    pl = torch.empty((nch * 128 * H,), dtype=torch.float32, device=q.device)
    _sparse_chunks[(R, H, nch)](q, kc, vc, tokens, counts, po, pm, pl, W=W, H=H, D=D, CH=CH, SCALE=scale,
                                num_warps=4, num_stages=1)
    _sparse_merge[(R, H)](po, pm, pl, out, counts, H=H, D=D, NCH=nch, CH=CH, num_warps=4)
