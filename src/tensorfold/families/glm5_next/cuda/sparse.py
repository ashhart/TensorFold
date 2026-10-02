"""Sparse DSA keeps lower-index pools on score ties and visits selected tokens and chunks in position order so window rows preserve serial bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.geometry import MLA_SELECT_ROWS as SELECT_ROWS   # a prompt chunk's rows scored at once

POOL = 4
TOPK_POOLS = 512
BR = 16
# the first position whose row select_tokens counts (npool > TOPK_POOLS): rows before it keep dense attention
SPARSE_FROM = (TOPK_POOLS + 1) * POOL - 1
TOKENS = TOPK_POOLS * POOL + POOL - 1      # a sparse row's attended tokens: its 512 pools', then its incomplete pool's


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
def _scores(QI, W, w_stride, PK, OUT, POS, R, NP, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr):
    """Program (RB rows, pool block): s_p = sum_h w_h relu(scale * qi_h . pool_p) up to each row's position, heads padded to HP; RB never changes a row's bits."""

    rb = tl.program_id(0)
    pb = tl.program_id(1)
    P = tl.load(POS)
    p = pb * BP + tl.arange(0, BP)
    # tiles past this row block's last visible complete pool store -inf without dot products (the same allocations)
    visible = (P + tl.minimum((rb + 1) * RB, R)) // 4
    if pb * BP >= visible:
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                tl.store(OUT + r * NP + p, float("-inf"), mask=p < NP)
        return
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    k = tl.load(PK + p[:, None] * D + d[None, :], mask=(p < (P + rb * RB + RB) // 4)[:, None],
                other=0.0).to(tl.bfloat16)                                              # [BP, D]
    for i in tl.static_range(RB):
        r = rb * RB + i
        if r < R:
            npool = (P + r + 1) // 4
            q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
            kr = tl.where((p < npool)[:, None], k, 0.0)
            dots = tl.dot(q, tl.trans(kr))                                                # [HP, BP] fp32
            w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
            sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
            sc = tl.where(p < npool, sc, float("-inf"))
            tl.store(OUT + r * NP + p, sc, mask=p < NP)


def _top_pools(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Reference top-k: each row's k best pools, ties to the lower pool, ascending, as a stable descending sort keeps them (unique int64 keys)."""
    bits = (scores + 0.0).view(torch.int32)                              # + 0.0: -0 becomes +0, as the sort ties them
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)          # IEEE order as signed ints (negatives flipped)
    keys = ordered.to(torch.int64).bitwise_left_shift_(32)
    keys.bitwise_or_(0xFFFFFFFF - torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64))
    best = torch.topk(keys, k, dim=1, sorted=False).values
    del keys
    return torch.sort(0xFFFFFFFF - (best & 0xFFFFFFFF), dim=1).values


@triton.jit
def _order_key(s):
    """A float32 score as a uint32 whose unsigned order is the scores' order (-0 counted as +0)."""
    bits = (s + 0.0).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _select_rows(S, OUT, NP, POS, K: tl.constexpr, BLOCK: tl.constexpr, VISIBLE: tl.constexpr):
    """Program r: a radix select (8 bits a pass) finds the K-th best score, then one pass in pool order writes the pools above it and the lowest ties."""

    r = tl.program_id(0).to(tl.int64)
    row = S + r * NP
    limit = tl.minimum(NP, tl.maximum(K, (tl.load(POS) + r + 1) // 4)) if VISIBLE else NP
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, limit, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < limit
            u = _order_key(tl.load(row + i, mask=ok, other=0.0))
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist           # pools with this digit or a higher one
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, limit, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < limit
        u = _order_key(tl.load(row + i, mask=ok, other=0.0))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * K + written + tl.cumsum(t, 0) - t, i.to(tl.int64), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)


def top_pools(scores: torch.Tensor, k: int, pos_dev: torch.Tensor | None = None) -> torch.Tensor:
    """``_top_pools``'s pools in one kernel, ascending, without int64 keys, top-k or sort."""
    R, NP = scores.shape
    if NP < k or not scores.is_contiguous():
        return _top_pools(scores, k)
    out = torch.empty((R, k), dtype=torch.int64, device=scores.device)
    _select_rows[(R,)](scores, out, NP, pos_dev if pos_dev is not None else scores,
                       K=k, BLOCK=1024, VISIBLE=pos_dev is not None, num_warps=4)
    return out


def pool_bucket(pos: int, R: int, np_max: int) -> int:
    """Pools to score for rows pos .. pos + R - 1: the visible ones rounded up to a power of two (at least 1024), at most the capacity's."""
    visible = (pos + R) // POOL + 1
    return min(np_max, max(1024, 1 << (visible - 1).bit_length()))


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, pos: int | None, R: int, np_max: int,
                  pos_dev: torch.Tensor, *, bucket: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's attended tokens [R, 2051] ascending (-1 padded) and their count past the dense limit; ``bucket`` fixes the pool count for graphs."""

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    # score only visible pools, rounded up to a power of two so the allocator reuses a few sizes (exact sizes fragmented memory at 128k)
    np_max = bucket if bucket is not None else pool_bucket(pos, R, np_max)
    # rows scored SELECT_ROWS at a time over the window's np_max pools (the same columns, so the same bits a row)
    B = min(R, SELECT_ROWS)
    scores = torch.empty((B, np_max), dtype=torch.float32, device=qi.device)
    # heads and width from the tensors: fixed ones read past a row's index query into its window neighbours
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5            # 32 ** -0.5 exactly as before
    blocks = []
    for a in range(0, R, B):
        n = min(B, R - a)
        at = pos_dev if a == 0 else pos_dev + a                        # the block's first row's position
        _scores[(n, triton.cdiv(np_max, 64))](qi[a:a + n], wts[a:a + n], wts.stride(0), pk, scores, at, n, np_max,
                                             D ** -0.5, wscale, H=H, HP=max(16, triton.next_power_of_2(H)), D=D,
                                             BP=64, RB=1, num_warps=4)
        blocks.append(top_pools(scores[:n], TOPK_POOLS, at))                           # ascending pool index
    del scores
    pools = blocks[0] if len(blocks) == 1 else torch.cat(blocks)
    dev = qi.device
    width = TOPK_POOLS * POOL + POOL - 1
    # all rows at once: the 512 pools' tokens ascending, then the incomplete last pool's visible tokens; rows within the dense limit count 0
    q = pos_dev.to(torch.int64) + torch.arange(R, device=dev)
    npool = (q + 1) // POOL
    tokens = torch.empty((R, width), dtype=torch.int32, device=dev)
    tokens[:, :TOPK_POOLS * POOL] = (pools[:, :, None] * POOL + torch.arange(POOL, device=dev)).reshape(R, -1)
    tail = npool[:, None] * POOL + torch.arange(POOL - 1, device=dev)
    tail_ok = tail <= q[:, None]
    tokens[:, TOPK_POOLS * POOL:] = torch.where(tail_ok, tail, -1)
    counts = torch.where(npool > TOPK_POOLS, TOPK_POOLS * POOL + tail_ok.sum(1), 0).to(torch.int32)
    return tokens, counts


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
    """Attend selected tokens in list order with the query in row 0 of a 16-row tile and the other tile rows idle."""

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


PART_ROWS = 128          # rows of one launch: the kernels keep a row's chunk partials at c * 128 + r


def sparse_attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float) -> None:
    """Write attention for rows with positive counts into out [R, H, D] in launches of up to 128 rows, leaving other rows untouched."""

    R, H, D = q.shape
    W = tokens.shape[1]
    CH = 512
    nch = triton.cdiv(W, CH)
    po = torch.empty((nch * PART_ROWS * H * D,), dtype=torch.float32, device=q.device)
    pm = torch.empty((nch * PART_ROWS * H,), dtype=torch.float32, device=q.device)
    pl = torch.empty((nch * PART_ROWS * H,), dtype=torch.float32, device=q.device)
    for r0 in range(0, R, PART_ROWS):
        n = min(PART_ROWS, R - r0)
        rows = slice(r0, r0 + n)
        _sparse_chunks[(n, H, nch)](q[rows], kc, vc, tokens[rows], counts[rows], po, pm, pl, W=W, H=H, D=D, CH=CH,
                                    SCALE=scale, num_warps=4, num_stages=1)
        _sparse_merge[(n, H)](po, pm, pl, out[rows], counts[rows], H=H, D=D, NCH=nch, CH=CH, num_warps=4)


# ------------------------------------------------------------------------------------- multi-stream windows ---
# Segmented windows (latent.py's section of the same name): each stream's index keys and gates in its own extent of
# shared arenas [P, 128] (token t of the stream at base + t) and its pooled keys at base / 4 + p of [P / 4 + 2, 128];
# per-row device tables (``segments.SegRows``) give each row its position, base and kind. A row keeps its
# single-stream arithmetic. Reads and writes of the index caches go through the helpers below.

@triton.jit
def _ix_put(IX, row, d, x, D: tl.constexpr):
    """Index key or gate row ``row`` (int64) <- x [D] as bf16."""
    tl.store(IX + row * D + d, x.to(tl.bfloat16))


@triton.jit
def _ix_get(IX, row, d, D: tl.constexpr):
    """Index key or gate row ``row`` (int64) as fp32 [D]."""
    return tl.load(IX + row * D + d).to(tl.float32)


@triton.jit
def _pk_put(PK, row, d, x, D: tl.constexpr):
    """Pooled key row ``row`` (int64) <- x [D] (fp32) as bf16."""
    tl.store(PK + row * D + d, x.to(tl.bfloat16))


@triton.jit
def _pk_get(PK, rows, ok, d, D: tl.constexpr):
    """Pooled key rows ``rows`` (int64 [n]) as a bf16 tile [n, D]; rows with ok false are zeros."""
    return tl.load(PK + rows[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.bfloat16)


@triton.jit
def _seg_index_write(KR, k_stride, GR, LNW, LNB, IK, IG, POS, BASE, eps, D: tl.constexpr):
    """_index_write for row r at its stream's row base[r] + pos[r]."""

    r = tl.program_id(0)
    row = tl.load(BASE + r).to(tl.int64) + tl.load(POS + r).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(KR + r * k_stride + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    y = xc / tl.sqrt(var + eps) * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)
    _ix_put(IK, row, d, y, D)
    _ix_put(IG, row, d, tl.load(GR + r * D + d), D)


@triton.jit
def _seg_pool_keys(IK, IG, APE, PK, POS, BASE, D: tl.constexpr):
    """Program r: the pool row r completes (pos[r] % 4 == 3), pool p = pos[r] // 4 of its stream, from tokens
    base + 4p .. base + 4p + 3 into pooled row base / 4 + p: _pool_keys' arithmetic (the pools a single-stream
    window completes are exactly those its rows end)."""

    r = tl.program_id(0)
    q = tl.load(POS + r).to(tl.int64)
    if q % 4 != 3:
        return
    B = tl.load(BASE + r).to(tl.int64)
    t = B + (q // 4) * 4
    d = tl.arange(0, D)
    l0 = _ix_get(IG, t + 0, d, D) + tl.load(APE + 0 * D + d).to(tl.float32)
    l1 = _ix_get(IG, t + 1, d, D) + tl.load(APE + 1 * D + d).to(tl.float32)
    l2 = _ix_get(IG, t + 2, d, D) + tl.load(APE + 2 * D + d).to(tl.float32)
    l3 = _ix_get(IG, t + 3, d, D) + tl.load(APE + 3 * D + d).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    s = ((e0 + e1) + e2) + e3
    k0 = _ix_get(IK, t + 0, d, D)
    k1 = _ix_get(IK, t + 1, d, D)
    k2 = _ix_get(IK, t + 2, d, D)
    k3 = _ix_get(IK, t + 3, d, D)
    t0 = ((e0 / s).to(tl.bfloat16).to(tl.float32) * k0).to(tl.bfloat16).to(tl.float32)
    t1 = ((e1 / s).to(tl.bfloat16).to(tl.float32) * k1).to(tl.bfloat16).to(tl.float32)
    t2 = ((e2 / s).to(tl.bfloat16).to(tl.float32) * k2).to(tl.bfloat16).to(tl.float32)
    t3 = ((e3 / s).to(tl.bfloat16).to(tl.float32) * k3).to(tl.bfloat16).to(tl.float32)
    _pk_put(PK, B // 4 + q // 4, d, ((t0 + t1) + t2) + t3, D)


def seg_index_update(k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor,
                     ape: torch.Tensor, ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor, rows) -> None:
    """index_update for a segmented window: each row's index key and gate at its stream's base + pos, then every
    pool a row completes (``rows``: segments.SegRows)."""

    R = k_raw.shape[0]
    _seg_index_write[(R,)](k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, rows.pos, rows.base, 1e-6, D=128,
                           num_warps=1)
    _seg_pool_keys[(R,)](ik, ig, ape, pk, rows.pos, rows.base, D=128, num_warps=1)


@triton.jit
def _seg_scores(QI, W, w_stride, PK, OUT, o_stride, POS, SPR, SEG_START, SEG_ROWS, SEG_PBASE, SEG_POOLS, scale,
                wscale, H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BP: tl.constexpr, G: tl.constexpr):
    """Program (segment, g): pool blocks g, g + G, .. below the segment's pool count (on the device: the pools its
    last row sees), each block's keys (at the stream's pooled base) loaded once for all the segment's sparse rows;
    a row's scores are _scores' (its keys masked to its own pools before the dot, as with RB > 1), written for its
    visible pools only (the ones the selection reads)."""

    s = tl.program_id(0)
    g = tl.program_id(1)
    npm = tl.load(SEG_POOLS + s)
    nr = tl.load(SEG_ROWS + s)
    r0 = tl.load(SEG_START + s)
    pbase = tl.load(SEG_PBASE + s).to(tl.int64)
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    for pb in range(g, (npm + BP - 1) // BP, G):
        p = pb * BP + tl.arange(0, BP)
        k = _pk_get(PK, pbase + p, p < npm, d, D)                                       # [BP, D]
        for i in range(nr):
            r = r0 + i
            if tl.load(SPR + r) != 0:
                npool = (tl.load(POS + r) + 1) // 4
                q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
                kr = tl.where((p < npool)[:, None], k, 0.0)
                dots = tl.dot(q, tl.trans(kr))                                            # [HP, BP] fp32
                w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                tl.store(OUT + r * o_stride + p, sc, mask=p < npool)


@triton.jit
def _seg_select(S, s_stride, POS, SPR, TOK, CNT, W: tl.constexpr, K: tl.constexpr, PL: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program r (a sparse row): _select_rows over the row's visible pools, its K pools' tokens written as they are
    taken (ascending), then _tokens' tail and count; a dense row gets count 0. Every single-stream path scores more
    columns, but those lie past the row's visible pools, score -inf and have the highest indices: they rank below
    every visible pool (ties go to the lower pool), so a row with more than K visible pools selects the same ones
    (sparse._select_prompt)."""

    r = tl.program_id(0).to(tl.int64)
    if tl.load(SPR + r) == 0:
        tl.store(CNT + r, 0)
        return
    qpos = tl.load(POS + r).to(tl.int64)
    NP = (qpos + 1) // PL
    row = S + r * s_stride
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, NP, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < NP
            u = _order_key(tl.load(row + i, mask=ok, other=0.0))
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    j = tl.arange(0, PL)
    for c in range(0, NP, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < NP
        u = _order_key(tl.load(row + i, mask=ok, other=0.0))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        slot = (written + tl.cumsum(t, 0) - t).to(tl.int64)
        tl.store(TOK + r * W + slot[:, None] * PL + j[None, :], (i.to(tl.int64)[:, None] * PL + j[None, :]).to(tl.int32),
                 mask=take[:, None])
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)
    jt = tl.arange(0, 4)
    jok = jt < PL - 1
    tail = NP * PL + jt
    tok = jok & (tail <= qpos)
    tl.store(TOK + r * W + K * PL + jt, tl.where(tok, tail, -1).to(tl.int32), mask=jok)
    n = K * PL + tl.sum(tok.to(tl.int64), 0)
    tl.store(CNT + r, tl.where(NP > K, n, 0).to(tl.int32))


SEG_SCORE_GRID = 256       # _seg_scores programs a segment (grid stride over its pool blocks): speed only
SEG_SPLIT = True           # seg_select_tokens: the split selection (select_split), else one program a row (_seg_select)
SEG_SELECT = (4096, 8)     # _seg_select's scores a step and warps: speed only (the selection is exact integer logic)


def seg_select_tokens(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, rows, scratch) -> tuple[torch.Tensor,
                                                                                                          torch.Tensor]:
    """select_tokens for a segmented window: each sparse row's attended tokens [R, 2051] (relative to its stream,
    ascending, -1 padded) and counts (0 for dense rows). Scores every segment's pools up to its device-side pool
    count, so one CUDA graph per window size serves every context length (``scratch``: segments.SelectScratch)."""

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("seg_select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    R = qi.shape[0]
    if R > scratch.rows:
        raise ValueError(f"seg_select_tokens: {R} rows, the scratch holds {scratch.rows}")
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5            # select_tokens' constants
    scores = scratch.scores
    grid = min(SEG_SCORE_GRID, triton.cdiv(scores.shape[1], 64))
    _seg_scores[(rows.max_segs, grid)](qi, wts, wts.stride(0), pk, scores, scores.stride(0), rows.pos, rows.sparse,
                                       rows.seg_start, rows.seg_rows, rows.seg_pbase, rows.seg_pools, D ** -0.5,
                                       wscale, H=H, HP=max(16, triton.next_power_of_2(H)), D=D, BP=64, G=grid,
                                       num_warps=4)
    tokens, counts = scratch.tokens[:R], scratch.counts[:R]
    if SEG_SPLIT:
        select_split(scores[:R], TOPK_POOLS, scratch=scratch.split, rows=rows, tokens=tokens, counts=counts)
        return tokens, counts
    _seg_select[(R,)](scores, scores.stride(0), rows.pos, rows.sparse, tokens, counts, W=TOKENS, K=TOPK_POOLS,
                      PL=POOL, BLOCK=SEG_SELECT[0], num_warps=SEG_SELECT[1])
    return tokens, counts


# ----------------------------------------------------------------------------------- split top-k selection ---
# _select_rows' radix select with each row's scores split into CHS-score chunks, one program a (row, chunk), in five
# launches. Passes 0..3: every program takes the digits so far from the row's total histograms (TOT, integer
# atomic sums: exact in any order), keeps its chunk's count of scores above them (GT, first differing at an earlier
# byte) and its chunk's histogram of the next byte (HIST), adding it into the row's total. Step 4 writes the chunk's
# pools at the offset the earlier chunks' counts give (GT and their last histograms, which step 4 only reads):
# scores above the K-th best, then ties in pool order. The same K-th best, ties and ascending output as
# _select_rows: integers only, no floating-point sum order to keep.

@triton.jit
def _split_digits(TOT, r, STEPS: tl.constexpr, K: tl.constexpr):
    """(prefix, fixed, need, last digit) after the first STEPS passes, from the row's total histograms."""
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = tl.full((), K, tl.int32)
    digit = tl.full((), 0, tl.int32)
    for p in tl.static_range(STEPS):
        hist = tl.load(TOT + (r * 4 + p) * 256 + bins)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    return prefix, fixed, need, digit


@triton.jit
def _select_split(S, s_stride, NPC, POS, SPR, TOT, HIST, GT, OUT, TOK, CNT, W: tl.constexpr, K: tl.constexpr,
                  PL: tl.constexpr, CHS: tl.constexpr, CP: tl.constexpr, STEP: tl.constexpr, SEG: tl.constexpr):
    """Program (row, chunk), STEP 0..3: pass STEP over the chunk; STEP 4: its selected pools, as pool ids OUT[r, :K]
    (SEG False: NPC scores a row) or (SEG: the row's visible pools, dense rows skipped) as _seg_select's tokens,
    tail and count. TOT [R, 4, 256] zeroed before step 0; HIST [R, CP, 256], GT [R, CP] int32."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    if SEG:
        if tl.load(SPR + r) == 0:
            if STEP == 4:
                if c == 0:
                    tl.store(CNT + r, 0)
            return
        qpos = tl.load(POS + r).to(tl.int64)
        NP = (qpos + 1) // PL
    else:
        qpos = 0
        NP = NPC
    nch = (NP + CHS - 1) // CHS
    if c >= nch:
        return
    bins = tl.arange(0, 256)
    i = c * CHS + tl.arange(0, CHS)
    ok = i < NP
    u = _order_key(tl.load(S + r * s_stride + i, mask=ok, other=0.0))
    prefix, fixed, need, d = _split_digits(TOT, r, STEP, K)
    own = HIST + (r * CP + c) * 256
    if STEP > 0 and STEP < 4:             # the chunk's scores above the prefix first at the byte just fixed
        above = tl.sum(tl.where(bins > d, tl.load(own + bins), 0), 0)
        if STEP > 1:
            above += tl.load(GT + r * CP + c)
        tl.store(GT + r * CP + c, above)
    if STEP < 4:
        match = ok & ((u & fixed) == prefix)
        hist = tl.histogram(((u >> (24 - 8 * STEP)) & 0xFF).to(tl.int32), 256, mask=match)
        tl.store(own + bins, hist)
        tl.atomic_add(TOT + (r * 4 + STEP) * 256 + bins, hist)
    else:
        ci = tl.arange(0, CP)
        before = ci < c
        # earlier chunks: scores above the prefix at bytes 0..2 (GT), at byte 3 and equal to it (their last histograms)
        last = tl.load(HIST + (r * CP + ci[:, None]) * 256 + bins[None, :], mask=before[:, None], other=0)
        gt = tl.load(GT + r * CP + ci, mask=before, other=0) + tl.sum(tl.where(bins[None, :] > d, last, 0), 1)
        equal_seen = tl.sum(tl.sum(tl.where(bins[None, :] == d, last, 0), 1), 0)
        written = tl.sum(gt, 0) + tl.minimum(equal_seen, need)
        e = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((e == 1) & (tl.cumsum(e, 0) - e + equal_seen < need))
        t = take.to(tl.int32)
        slot = (written + tl.cumsum(t, 0) - t).to(tl.int64)
        if SEG:
            j = tl.arange(0, PL)
            tl.store(TOK + r * W + slot[:, None] * PL + j[None, :],
                     (i.to(tl.int64)[:, None] * PL + j[None, :]).to(tl.int32), mask=take[:, None])
            if c == 0:
                jt = tl.arange(0, 4)
                jok = jt < PL - 1
                tail = NP * PL + jt
                tok = jok & (tail <= qpos)
                tl.store(TOK + r * W + K * PL + jt, tl.where(tok, tail, -1).to(tl.int32), mask=jok)
                n = K * PL + tl.sum(tok.to(tl.int64), 0)
                tl.store(CNT + r, tl.where(NP > K, n, 0).to(tl.int32))
        else:
            tl.store(OUT + r * K + slot, i.to(tl.int64), mask=take)


SPLIT_FROM = 8192          # decode windows split rows of at least this many scores (fewer: one program a row)
SPLIT_ROWS = 64            # ... windows of fewer rows (prompt chunks' hundreds of rows fill the GPU as they are)
SPLIT_WARPS = 4


def split_chunk(np_: int) -> int:
    """Scores a chunk program takes (speed only): 2,048 up to 16,384 a row, else 4,096."""
    return 2048 if np_ <= 16384 else 4096


def split_chunks(np_: int) -> int:
    """Chunk programs a row of np_ scores takes (a power of two, the chunk tables' row stride)."""
    return triton.next_power_of_2(triton.cdiv(np_, split_chunk(np_)))


def split_scratch(rows: int, np_: int, device) -> torch.Tensor:
    """select_split's int32 scratch for rows of np_ scores: totals [rows, 4, 256], then chunk histograms and counts."""
    return torch.empty((rows * (1024 + split_chunks(np_) * 257),), dtype=torch.int32, device=device)


def select_split(scores: torch.Tensor, k: int, *, scratch: torch.Tensor | None = None, rows=None,
                 tokens: torch.Tensor | None = None, counts: torch.Tensor | None = None) -> torch.Tensor | None:
    """The split selection: top_pools' pools [R, k] (``rows`` None), or with ``rows`` (segments.SegRows) and
    tokens / counts, seg_select_tokens' per-row output over each sparse row's visible pools. ``scratch``: at least
    split_scratch(R, NP)'s (allocated when None)."""

    R, NP = scores.shape
    chs, warps = split_chunk(NP), SPLIT_WARPS
    cp = split_chunks(NP)
    if scratch is None or scratch.numel() < R * (1024 + cp * 257):
        scratch = split_scratch(R, NP, scores.device)
    tot = scratch[:R * 1024]
    hist = scratch[R * 1024:R * (1024 + cp * 256)]
    gt = scratch[R * (1024 + cp * 256):R * (1024 + cp * 257)]
    tot.zero_()
    seg = rows is not None
    out = None if seg else torch.empty((R, k), dtype=torch.int64, device=scores.device)
    dummy = scratch
    for step in range(5):
        _select_split[(R, cp)](scores, scores.stride(0), NP, rows.pos if seg else dummy, rows.sparse if seg else dummy,
                               tot, hist, gt, out if out is not None else dummy, tokens if seg else dummy,
                               counts if seg else dummy, W=TOKENS, K=k, PL=POOL, CHS=chs, CP=cp, STEP=step, SEG=seg,
                               num_warps=warps)
    return out
