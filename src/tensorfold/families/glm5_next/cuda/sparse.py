"""Sparse DSA keeps lower-index pools on score ties and visits selected tokens and chunks in position order so window rows preserve serial bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.geometry import MLA_SELECT_ROWS as SELECT_ROWS   # a prompt chunk's rows scored at once

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
def _scores(QI, W, w_stride, PK, OUT, POS, R, NP, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr):
    """Program (RB rows, pool block): s_p = sum_h w_h relu(scale * qi_h . pool_p) up to each row's position, heads padded to HP; RB never changes a row's bits."""

    rb = tl.program_id(0)
    pb = tl.program_id(1)
    P = tl.load(POS)
    p = pb * BP + tl.arange(0, BP)
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
def _select_rows(S, OUT, NP, K: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: a radix select (8 bits a pass) finds the K-th best score, then one pass in pool order writes the pools above it and the lowest ties."""

    r = tl.program_id(0).to(tl.int64)
    row = S + r * NP
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
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist           # pools with this digit or a higher one
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, NP, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < NP
        u = _order_key(tl.load(row + i, mask=ok, other=0.0))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * K + written + tl.cumsum(t, 0) - t, i.to(tl.int64), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)


def top_pools(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``_top_pools``'s pools in one kernel, ascending, without int64 keys, top-k or sort."""
    R, NP = scores.shape
    if NP < k or not scores.is_contiguous():
        return _top_pools(scores, k)
    out = torch.empty((R, k), dtype=torch.int64, device=scores.device)
    _select_rows[(R,)](scores, out, NP, K=k, BLOCK=1024, num_warps=4)
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
    # rows go through in blocks of SELECT_ROWS, each block scoring the window's np_max pools (the same columns, so the
    # same bits a row): the fp32 scores hold SELECT_ROWS rows of the capacity-sized width, not the window's
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
        blocks.append(top_pools(scores[:n], TOPK_POOLS))                               # ascending pool index
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
