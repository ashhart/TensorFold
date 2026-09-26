"""Flash Next attention on CUDA for a chain window: row r (position P0 + r) reads cache keys [0, P0 + r].

Keys are cut into fixed 512-key chunks by absolute position, eight 64-key tensor-core tiles each; a
program takes one row, one KV head (its 12 query heads as 16 tile rows) and one chunk, and the chunks
merge in position order. A row's chunks hold the same keys whether it runs alone or in a window, so its
bits do not depend on the window. The committed length P0 is read on the device and the grid covers the
cache's capacity (programs past a row's keys write empty partials), so a step can be graph-captured.

Past the indexer's budget (a row whose position p has (p + 1) // 4 > 512 complete 4-key blocks) a row
reads only its best 512 blocks and its unfinished tail, as the model does (QSA): ``pool`` keeps each
completed block's normed, rotated mean indexer key; ``scores`` rates every complete block for each such
row (fp32: the sum over the indexer heads of relu(q . k) over sqrt(d)); ``select`` finds the row's 512th
best score exactly (a threshold search over order-preserving keys) and lists the blocks above it, then the
lowest-numbered blocks equal to it, in block order, 4 keys each, then the tail. The attention kernels then
read that list in fixed 512-entry chunks. A dense row keeps the dense path's arithmetic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = 512
TILE = 64


@triton.jit
def _tile(q, k, v, m, l, o, valid, scale: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
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
def _chunks(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
            NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(POS0) + r + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
        n = tl.where(sparse, tl.load(NKR + r), n)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0.0)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    start = c * CH
    tiles = tl.minimum(tl.maximum(n - start, 0), CH)
    for t in range(0, tl.cdiv(tiles, 64)):
        ki = start + t * 64 + tl.arange(0, 64)
        valid = ki < n
        if QSA:
            if sparse:
                ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
        kk = tl.load(KC + (ki[:, None].to(tl.int64) * HK + hk) * D + d[None, :], mask=valid[:, None], other=0.0)
        vv = tl.load(VC + (ki[:, None].to(tl.int64) * HK + hk) * D + d[None, :], mask=valid[:, None], other=0.0)
        m, l, o = _tile(q, kk, vv, m, l, o, valid, SCALE)
    base = (r * NCH + c) * H + hk * G + gg
    tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
    tl.store(PM + base, m, mask=gg < G)
    tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, POS0, OUT, NKR, SPR, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
           CH: tl.constexpr, NCH: tl.constexpr, QSA: tl.constexpr):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    n = tl.load(POS0) + r + 1
    if QSA:
        n = tl.where(tl.load(SPR + r) != 0, tl.load(NKR + r), n)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    head = hk * G + gg
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for c in range(0, tl.cdiv(n, CH)):
        base = (r * NCH + c) * H + head
        cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
        cl = tl.load(PL + base, mask=gg < G, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a[:, None] + co * b[:, None]
        l = l * a + cl * b
        m = next_m
    result = o / l[:, None]
    tl.store(OUT + (r * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16), mask=gg[:, None] < G)


class AttnScratch:
    def __init__(self, rows: int, heads: int, head_dim: int, capacity: int, device, *, budget: int = 2048,
                 ratio: int = 4) -> None:
        self.nch = -(-capacity // CHUNK)
        self.po = torch.zeros((rows, self.nch, heads, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.pl = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
        # sparse attention (QSA): each row's key list, its length, whether it is sparse, block scores
        self.budget, self.ratio = budget, ratio
        self.idw = budget + ratio
        self.qsa = capacity > budget
        self.nb = -(-capacity // ratio)
        self.ids = torch.zeros((rows, self.idw), dtype=torch.int32, device=device)
        self.nk = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.sparse = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.scores = torch.zeros((rows, self.nb), dtype=torch.float32, device=device) if self.qsa else None


def attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch,
              rows: int, scale: float) -> torch.Tensor:
    """q [R, H, D] bf16 (normed, rotated), kc/vc [cap, HK, D] bf16 caches holding positions [0, P0 + R),
    pos0 [1] int32 (P0) -> [R, H, D] bf16. Sparse rows read scratch.ids (``select``)."""

    _, h, d = q.shape
    hk = kc.shape[1]
    g = h // hk
    nch = scratch.nch
    _chunks[(rows, hk, nch)](q, kc, vc, pos0, scratch.po, scratch.pm, scratch.pl, scratch.ids, scratch.nk,
                             scratch.sparse, H=h, HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, SCALE=scale,
                             IDW=scratch.idw, QSA=scratch.qsa, num_warps=4, num_stages=1)
    _merge[(rows, hk)](scratch.po, scratch.pm, scratch.pl, pos0, scratch.out, scratch.nk, scratch.sparse, H=h,
                       HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, QSA=scratch.qsa, num_warps=4)
    return scratch.out


# -- QSA block selection ---------------------------------------------------------------------------------
@triton.jit
def _pool(IKC, POOLED, POS0, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr):
    """Program i: block b = P0 // RATIO + i if it is complete within the window's positions: the fp32 mean of
    its RATIO raw indexer keys (in order, bf16), RMSNorm with the stored scale (fp32, bf16), RoPE on the first
    2 HALF dims at position RATIO b (bf16). A block's value depends only on its keys: recomputing it gives
    the same bits."""

    i = tl.program_id(0)
    p0 = tl.load(POS0)
    b = p0 // RATIO + i
    if RATIO * b + RATIO <= p0 + R:
        d = tl.arange(0, DI)
        acc = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + d).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            acc = acc + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + d).to(tl.float32)
        x = (acc / RATIO).to(tl.bfloat16).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / DI + eps)
        xn = (x * rinv * tl.load(W + d)).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
        xp = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + partner).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            xp = xp + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + partner).to(tl.float32)
        xp = (xp / RATIO).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * rinv * tl.load(W + partner)).to(tl.bfloat16).to(tl.float32)
        j = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        ang = (RATIO * b).to(tl.float32) * tl.load(INV + j)
        cos, sin = tl.cos(ang), tl.sin(ang)
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        tl.store(POOLED + b.to(tl.int64) * DI + d, rot.to(tl.bfloat16))


@triton.jit
def _scores(IQ, POOLED, POS0, SC, NB, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr,
            BB: tl.constexpr):
    """Program (r, j): blocks [j BB, (j + 1) BB) of row r (only a sparse row, only its complete blocks):
    fp32 sum over the HI indexer heads, in order, of relu(q . k), over sqrt(DI)."""

    r = tl.program_id(0)
    j = tl.program_id(1)
    complete = (tl.load(POS0) + r + 1) // RATIO
    if complete > TOP:
        b = j * BB + tl.arange(0, BB)
        ok = b < complete
        d = tl.arange(0, DI)
        k = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        total = tl.zeros((BB,), dtype=tl.float32)
        for h in tl.static_range(HI):
            q = tl.load(IQ + (r * HI + h) * DI + d).to(tl.float32)
            total = total + tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
        tl.store(SC + r * NB + b, total / tl.sqrt(DI * 1.0), mask=ok)


@triton.jit
def _select(SC, POS0, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
            BLOCK: tl.constexpr):
    """Program r: a sparse row's key list: its TOP best blocks (the lowest block ids among scores equal to
    the cut), in block order, RATIO keys each, then the tail keys [RATIO complete, end). A dense row: its
    length only."""

    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        b = tl.arange(0, BLOCK)
        ok = b < complete
        v = tl.load(SC + r * NB + b, mask=ok, other=0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        key = tl.where(ok, key, 0)
        # the largest t with at least TOP keys >= t: 32 halvings over [0, 2^32)
        lo = tl.zeros((), dtype=tl.uint64)
        hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint64)
        k64 = key.to(tl.uint64)
        for _ in range(33):
            mid = (lo + hi + 1) // 2
            count = tl.sum(tl.where(k64 >= mid, 1, 0), axis=0)
            take = count >= TOP
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        cut = lo
        above = ok & (k64 > cut)
        equal = ok & (k64 == cut)
        need = TOP - tl.sum(above.to(tl.int32), axis=0)
        rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (rank <= need))
        place = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        for k in tl.static_range(RATIO):
            tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


def qsa_select(iq: torch.Tensor, ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
               inv_freq: torch.Tensor, eps: float, scratch: AttnScratch, rows: int) -> None:
    """Pool the blocks the window completes, score and select each sparse row's blocks (scratch.ids/nk/sparse)."""

    di = ikc.shape[1]
    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    _pool[(rows // ratio + 2,)](ikc, pooled, pos0, ik_scale, inv_freq, eps, rows, DI=di, HALF=inv_freq.numel(),
                                RATIO=ratio, num_warps=1)
    bb = 64
    _scores[(rows, triton.cdiv(scratch.nb, bb))](iq, pooled, pos0, scratch.scores, scratch.nb, HI=iq.shape[1],
                                                 DI=di, RATIO=ratio, TOP=top, BB=bb, num_warps=4)
    _select[(rows,)](scratch.scores, pos0, scratch.ids, scratch.nk, scratch.sparse, scratch.nb, RATIO=ratio,
                     TOP=top, IDW=scratch.idw, BLOCK=triton.next_power_of_2(scratch.nb), num_warps=16)
