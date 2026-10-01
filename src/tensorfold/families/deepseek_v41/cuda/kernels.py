"""DeepSeek-V4.1 decode/prefill kernels: RMSNorm, table RoPE, and MQA attention over compressed + window keys.

Attention: one 512-wide latent per key is both key and value for every head (MQA). A row attends to the first
``(p + 1) // ratio`` entries of its kv source's compressed cache and to its own window slots p - 127 .. p, plus
a per-head sink logit in the denominator. Heads fill the tile rows (16 a program), so a key block is loaded once
per 16 heads; chunks of keys run as separate programs and a merge adds the sink and normalizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_TILE = 16              # heads a chunk program (decode and verify rows; prompt chunks use FULL_HT)
KEY_TILE = 64
FULL_ROWS = 16            # above this many rows (prompt chunks) with RoPE: _mqa_full, one program a row
FULL_HT, FULL_KT, FULL_WARPS, FULL_STAGES = 32, 32, 8, 1
CHUNK = 128               # keys a chunk program takes (small row counts: more programs to fill the SMs)


@triton.jit
def _rmsnorm(X, W, OUT, x_stride, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    ok = o < N
    x = tl.load(X + r * x_stride + o, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + o, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * N + o, (x * rinv * w).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 RMSNorm of each row of x [R, N] (any float dtype, fp32 math)."""

    R, N = x.shape
    out = torch.empty((R, N), dtype=torch.bfloat16, device=x.device)
    _rmsnorm[(R,)](x, w, out, x.stride(0), eps, N=N, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return out


@triton.jit
def _rope(X, POS, COS, SIN, OUT, heads, D: tl.constexpr, HALF: tl.constexpr, SIGN: tl.constexpr):
    """GPT-J rotation of the last 2*HALF dims of each [D] head vector at the row's position; others copied."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    base = (r * heads + h) * D
    d = tl.arange(0, D)
    x = tl.load(X + base + d).to(tl.float32)
    tl.store(OUT + base + d, x.to(OUT.dtype.element_ty), mask=d < D - 2 * HALF)
    p = tl.load(POS + r)
    i = tl.arange(0, HALF)
    c = tl.load(COS + p * HALF + i)
    s = tl.load(SIN + p * HALF + i) * SIGN
    e = tl.load(X + base + D - 2 * HALF + 2 * i).to(tl.float32)
    od = tl.load(X + base + D - 2 * HALF + 2 * i + 1).to(tl.float32)
    tl.store(OUT + base + D - 2 * HALF + 2 * i, (e * c - od * s).to(OUT.dtype.element_ty))
    tl.store(OUT + base + D - 2 * HALF + 2 * i + 1, (od * c + e * s).to(OUT.dtype.element_ty))


def rope(x: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False,
         out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [R, D] or [R, H, D]; tables [max_pos, 32]; returns the rotated copy (fp32 math)."""

    shape = x.shape
    x3 = x.reshape(shape[0], -1, shape[-1]).contiguous()
    out = torch.empty(x3.shape, dtype=out_dtype or x.dtype, device=x.device)
    _rope[(x3.shape[0], x3.shape[1])](x3, pos, cos, sin, out, x3.shape[1], D=shape[-1], HALF=cos.shape[1],
                                      SIGN=-1.0 if inverse else 1.0, num_warps=4)
    return out.reshape(shape)


def rope_tables(freqs: torch.Tensor, max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    ang = torch.arange(max_pos, dtype=torch.float64, device=freqs.device)[:, None] * freqs.double()[None, :]
    return ang.cos().float().contiguous(), ang.sin().float().contiguous()


@triton.jit
def _mqa_chunks(Q, COMP, IDX, SWA, POS, PO, PM, PL, n_idx, idx_stride, H: tl.constexpr, D: tl.constexpr,
                W: tl.constexpr, RING: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, NCH: tl.constexpr,
                HT: tl.constexpr, KT: tl.constexpr):
    """Keys: the first ``n_idx`` slots are compressed entries named by IDX (-1: none), then the row's window
    positions p - W + 1 .. p read from the SWA ring at pos % RING."""

    r = tl.program_id(0)
    hg = tl.program_id(1)
    c = tl.program_id(2)
    p = tl.load(POS + r)
    hh = hg * HT + tl.arange(0, HT)
    d = tl.arange(0, D)
    m = tl.full((HT,), float("-inf"), tl.float32)
    l = tl.zeros((HT,), tl.float32)
    o = tl.zeros((HT, D), tl.float32)
    total = n_idx + W
    base = (r * NCH + c) * H + hh
    q = tl.load(Q + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    for t in range(CH // KT):
        k = c * CH + t * KT + tl.arange(0, KT)
        is_comp = k < n_idx
        kidx = tl.load(IDX + r * idx_stride + k, mask=is_comp, other=-1)
        slot = p - (W - 1) + (k - n_idx)                             # window position of a window key
        ok_c = is_comp & (kidx >= 0)
        ok_w = (k >= n_idx) & (k < total) & (slot >= 0)
        kc = tl.load(COMP + tl.maximum(kidx, 0)[:, None].to(tl.int64) * D + d[None, :], mask=ok_c[:, None], other=0.0)
        kw = tl.load(SWA + (tl.maximum(slot, 0) % RING)[:, None].to(tl.int64) * D + d[None, :], mask=ok_w[:, None],
                     other=0.0)
        kk = tl.where(ok_c[:, None], kc.to(tl.bfloat16), kw.to(tl.bfloat16))
        ok = ok_c | ok_w
        scores = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        scores = tl.where(ok[None, :], scores, float("-inf"))
        tile_m = tl.max(scores, 1)
        active = tile_m != float("-inf")
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        pr = tl.where(ok[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kk)
        l = l * alpha + tl.sum(pr, 1)
        m = next_m
    tl.store(PO + base[:, None] * D + d[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _mqa_full(Q, COMP, IDX, SWA, POS, SINK, OUT, COS, SIN, n_idx, idx_stride, H: tl.constexpr, D: tl.constexpr,
              W: tl.constexpr, RING: tl.constexpr, SCALE: tl.constexpr, HT: tl.constexpr, KT: tl.constexpr,
              HALF: tl.constexpr):
    """Prompt rows: one program takes a row's every key (as _mqa_chunks) and finishes it (sink, normalize, inverse
    RoPE of the last 2 * HALF dims), writing bf16 [R, H, D]; no per-chunk partials."""

    r = tl.program_id(0)
    hg = tl.program_id(1)
    p = tl.load(POS + r)
    hh = hg * HT + tl.arange(0, HT)
    d = tl.arange(0, D)
    m = tl.full((HT,), float("-inf"), tl.float32)
    l = tl.zeros((HT,), tl.float32)
    o = tl.zeros((HT, D), tl.float32)
    total = n_idx + W
    q = tl.load(Q + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    for k0 in range(0, total, KT):
        k = k0 + tl.arange(0, KT)
        is_comp = k < n_idx
        kidx = tl.load(IDX + r * idx_stride + k, mask=is_comp, other=-1)
        slot = p - (W - 1) + (k - n_idx)
        ok_c = is_comp & (kidx >= 0)
        ok_w = (k >= n_idx) & (k < total) & (slot >= 0)
        kc = tl.load(COMP + tl.maximum(kidx, 0)[:, None].to(tl.int64) * D + d[None, :], mask=ok_c[:, None], other=0.0)
        kw = tl.load(SWA + (tl.maximum(slot, 0) % RING)[:, None].to(tl.int64) * D + d[None, :], mask=ok_w[:, None],
                     other=0.0)
        kk = tl.where(ok_c[:, None], kc.to(tl.bfloat16), kw.to(tl.bfloat16))
        ok = ok_c | ok_w
        scores = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        scores = tl.where(ok[None, :], scores, float("-inf"))
        tile_m = tl.max(scores, 1)
        active = tile_m != float("-inf")
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        pr = tl.where(ok[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kk)
        l = l * alpha + tl.sum(pr, 1)
        m = next_m
    sink = tl.load(SINK + hh)                  # a logit with a zero value vector
    top = tl.maximum(m, sink)
    a = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
    o = o * (a / (l * a + tl.exp(sink - top)))[:, None]
    rot = d >= D - 2 * HALF
    i = tl.maximum(d - (D - 2 * HALF), 0) // 2
    c_ = tl.load(COS + p * HALF + i, mask=rot, other=1.0)
    s_ = tl.load(SIN + p * HALF + i, mask=rot, other=0.0)
    ev, od = tl.split(tl.reshape(o, (HT, D // 2, 2)))
    cev, _ = tl.split(tl.reshape(c_, (D // 2, 2)))
    sev, _ = tl.split(tl.reshape(s_, (D // 2, 2)))
    ne = ev * cev[None, :] + od * sev[None, :]
    no = od * cev[None, :] - ev * sev[None, :]
    o = tl.reshape(tl.join(ne, no), (HT, D))
    tl.store(OUT + (r * H + hh[:, None]) * D + d[None, :], o.to(tl.bfloat16))


@triton.jit
def _mqa_merge(PO, PM, PL, SINK, OUT, POS, COS, SIN, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr,
               HALF: tl.constexpr, ROPE: tl.constexpr):
    """Combine the chunks with the sink; with ROPE, rotate the last 2 * HALF dims back (inverse RoPE) and write
    bf16 (the output projection's input), else fp32."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    m = tl.load(SINK + h)                   # the sink is a logit with a zero value vector
    l = 1.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        base = (r * NCH + c) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        active = cl > 0.0
        co = tl.load(PO + base * D + d, mask=(d < D) & active, other=0.0)
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.exp(m - next_m)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    o = o / l
    if ROPE:
        p = tl.load(POS + r)
        rot = d >= D - 2 * HALF
        i = tl.maximum(d - (D - 2 * HALF), 0) // 2
        c_ = tl.load(COS + p * HALF + i, mask=rot, other=1.0)
        s_ = tl.load(SIN + p * HALF + i, mask=rot, other=0.0)
        # pair values come from the normalized o itself: even dims pair with the next odd dim
        oe = tl.reshape(o, (D // 2, 2))
        ev, od = tl.split(oe)
        ce = tl.reshape(c_, (D // 2, 2))
        cev, _ = tl.split(ce)
        se = tl.reshape(s_, (D // 2, 2))
        sev, _ = tl.split(se)
        # inverse rotation: e' = e c + o s, o' = o c - e s (identity where c = 1, s = 0)
        ne = ev * cev + od * sev
        no = od * cev - ev * sev
        o = tl.reshape(tl.join(ne, no), (D,))
        tl.store(OUT + (r * H + h) * D + d, o.to(tl.bfloat16))
    else:
        tl.store(OUT + (r * H + h) * D + d, o)


class AttnBuffers:
    def __init__(self, rows: int, heads: int, dims: int, max_keys: int, device="cuda") -> None:
        nch = triton.cdiv(max_keys, CHUNK)
        self.po = torch.empty((rows * nch * heads * dims,), dtype=torch.float32, device=device)
        self.pm = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.max_keys = max_keys


def mqa(q: torch.Tensor, comp: torch.Tensor | None, idx: torch.Tensor | None, swa: torch.Tensor, pos: torch.Tensor,
        sink: torch.Tensor, window: int, buf: AttnBuffers, scale: float, cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, H, D] (RoPE'd) -> o [R, H, D] over the compressed entries ``idx`` [R, n] of ``comp`` and the window
    (``swa`` a ring of window rows addressed by position modulo its length): fp32, or with RoPE tables the
    inverse-rotated bf16 the output projection takes."""

    R, H, D = q.shape
    assert H % HEAD_TILE == 0
    n_idx = 0 if idx is None else idx.shape[1]
    keys = n_idx + window
    nch = triton.cdiv(keys, CHUNK)
    assert keys <= buf.max_keys
    rope = cos is not None
    out = torch.empty((R, H, D), dtype=torch.bfloat16 if rope else torch.float32, device=q.device)
    idx_t = idx if idx is not None else pos
    if rope and R > FULL_ROWS:
        _mqa_full[(R, H // FULL_HT)](q.contiguous(), comp if comp is not None else swa, idx_t, swa, pos, sink, out, cos,
                                     sin, n_idx, idx_t.stride(0) if idx is not None else 0, H=H, D=D, W=window,
                                     RING=swa.shape[0], SCALE=scale, HT=FULL_HT, KT=FULL_KT, HALF=cos.shape[1],
                                     num_warps=FULL_WARPS, num_stages=FULL_STAGES)
        return out
    _mqa_chunks[(R, H // HEAD_TILE, nch)](q.contiguous(), comp if comp is not None else swa, idx_t, swa, pos, buf.po,
                                          buf.pm, buf.pl, n_idx, idx_t.stride(0) if idx is not None else 0, H=H, D=D,
                                          W=window, RING=swa.shape[0], CH=CHUNK, SCALE=scale, NCH=nch,
                                          HT=HEAD_TILE, KT=32, num_warps=8, num_stages=1)
    _mqa_merge[(R, H)](buf.po, buf.pm, buf.pl, sink, out, pos, cos if rope else sink, sin if rope else sink, H=H, D=D,
                       NCH=nch, HALF=cos.shape[1] if rope else 1, ROPE=rope, num_warps=4)
    return out


@triton.jit
def _index_scores(IQ, WTS, KEYS, POS, OUT, n_keys, ratio, HI: tl.constexpr, DI: tl.constexpr, BS: tl.constexpr):
    """I[r, s] = sum_h w[r, h] * relu(iq[r, h] . k[s]) for visible s < (p + 1) // ratio, -inf elsewhere."""

    r = tl.program_id(0)
    sb = tl.program_id(1)
    p = tl.load(POS + r)
    n_vis = (p + 1) // ratio
    h = tl.arange(0, HI)
    d = tl.arange(0, DI)
    sidx = sb * BS + tl.arange(0, BS)
    q = tl.load(IQ + (r * HI + h[:, None]) * DI + d[None, :])
    k = tl.load(KEYS + sidx[:, None].to(tl.int64) * DI + d[None, :], mask=(sidx < n_keys)[:, None], other=0.0)
    dots = tl.dot(q, tl.trans(k)).to(tl.float32)                     # [HI, BS]
    w = tl.load(WTS + r * HI + h)
    score = tl.sum(w[:, None] * tl.maximum(dots, 0.0), axis=0)
    score = tl.where(sidx < n_vis, score, float("-inf"))
    tl.store(OUT + r * n_keys + sidx, score, mask=sidx < n_keys)


def index_scores(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int) -> torch.Tensor:
    """fp32 [R, S] indexer scores over every compressed entry, -inf where not yet visible."""

    R, HI, DI = iq.shape
    S = keys.shape[0]
    scores = torch.empty((R, S), dtype=torch.float32, device=iq.device)
    BS = 64
    _index_scores[(R, triton.cdiv(S, BS))](iq.contiguous(), wts.contiguous(), keys, pos, scores, S, ratio, HI=HI,
                                           DI=DI, BS=BS, num_warps=4)
    return scores


def top_entries(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """int32 [R, topk]: the best visible entries ascending, -1 padded (every visible one when <= topk)."""

    R, S = scores.shape
    k = min(topk, S)
    vals, idx = torch.topk(scores, k, dim=1, sorted=False)
    idx = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(idx, S), idx)   # invisible sort last, dropped
    idx = torch.sort(idx, dim=1).values
    idx = torch.where(idx >= S, torch.full_like(idx, -1), idx).int()
    if k < topk:
        idx = torch.cat([idx, torch.full((R, topk - k), -1, dtype=idx.dtype, device=idx.device)], dim=1)
    return idx.contiguous()


def candidate_blocks(scores: torch.Tensor, pos: torch.Tensor, ratio: int, block: int, keep: int) -> torch.Tensor:
    """Layer 20's blocks of ``block`` entries scored by their best entry, the newest block pinned, the ``keep`` best
    kept: int64 [R, keep], -1 padded."""

    R, S = scores.shape
    nb = -(-S // block)
    padded = torch.full((R, nb * block), float("-inf"), dtype=scores.dtype, device=scores.device)
    padded[:, :S] = scores
    best = padded.view(R, nb, block).amax(-1)
    newest = ((pos + 1) // ratio - 1).clamp(min=0) // block
    best.scatter_(1, newest[:, None].long(), float("inf"))
    vals, idx = torch.topk(best, min(keep, nb), dim=1)
    idx = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(idx, -1), idx)
    if idx.shape[1] < keep:
        idx = torch.cat([idx, torch.full((R, keep - idx.shape[1]), -1, dtype=idx.dtype, device=idx.device)], dim=1)
    return idx


def mask_to_blocks(scores: torch.Tensor, blocks: torch.Tensor, block: int) -> torch.Tensor:
    """Scores outside the chosen blocks set to -inf."""

    R, S = scores.shape
    nb = -(-S // block)
    flags = torch.zeros((R, nb + 1), dtype=torch.bool, device=scores.device)
    flags.scatter_(1, torch.where(blocks >= 0, blocks, nb), True)
    keep = flags[:, :nb].repeat_interleave(block, dim=1)[:, :S]
    return scores.masked_fill(~keep, float("-inf"))


@triton.jit
def _index_scores_seg(IQ, WTS, KEYS, POS, OUT, n_keys, off, seg, out_stride, ratio, HI: tl.constexpr,
                      DI: tl.constexpr, BS: tl.constexpr):
    """_index_scores over the key segment [off, off + seg): OUT[r, j] for key off + j (-inf past n_keys or not yet
    visible), the same arithmetic per key."""

    r = tl.program_id(0)
    sb = tl.program_id(1)
    p = tl.load(POS + r)
    n_vis = (p + 1) // ratio
    h = tl.arange(0, HI)
    d = tl.arange(0, DI)
    j = sb * BS + tl.arange(0, BS)
    sidx = off + j
    q = tl.load(IQ + (r * HI + h[:, None]) * DI + d[None, :])
    k = tl.load(KEYS + sidx[:, None].to(tl.int64) * DI + d[None, :], mask=(sidx < n_keys)[:, None], other=0.0)
    dots = tl.dot(q, tl.trans(k)).to(tl.float32)
    w = tl.load(WTS + r * HI + h)
    score = tl.sum(w[:, None] * tl.maximum(dots, 0.0), axis=0)
    score = tl.where((sidx < n_vis) & (sidx < n_keys), score, float("-inf"))
    tl.store(OUT + r * out_stride + j, score, mask=j < seg)


SELECT_ROWS = 512          # prompt rows a blocked selection pass takes
SELECT_SEG = 16384         # keys a segment scores at once (a multiple of every candidate block size)


def index_select_blocked(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int,
                         topk: int, *, blocks: torch.Tensor | None = None, block: int = 8,
                         candidates: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prompt rows: top_entries(index_scores(...)) (masked to ``blocks`` when given) without the [rows, keys]
    matrix: rows in passes of SELECT_ROWS, keys in segments of SELECT_SEG, a running top-k merged per segment.
    With ``candidates`` > 0 also returns candidate_blocks(...) of the unmasked scores (the source layer's)."""

    R, HI, DI = iq.shape
    S = keys.shape[0]
    dev = iq.device
    iq, wts = iq.contiguous(), wts.contiguous()
    k = min(topk, S)
    seg = min(SELECT_SEG, -(-S // block) * block)
    idx_out = torch.full((R, topk), -1, dtype=torch.int32, device=dev)
    cand_out = torch.full((R, candidates), -1, dtype=torch.int64, device=dev) if candidates else None
    nb = -(-S // block)
    buf = torch.empty((min(R, SELECT_ROWS), seg), dtype=torch.float32, device=dev)
    for r0 in range(0, R, SELECT_ROWS):
        r1 = min(R, r0 + SELECT_ROWS)
        n = r1 - r0
        vals = torch.full((n, k), float("-inf"), dtype=torch.float32, device=dev)
        ids = torch.full((n, k), S, dtype=torch.int64, device=dev)
        best = torch.full((n, nb), float("-inf"), dtype=torch.float32, device=dev) if candidates else None
        flags = None
        if blocks is not None:
            flags = torch.zeros((n, nb + 1), dtype=torch.bool, device=dev)
            b = blocks[r0:r1]
            flags.scatter_(1, torch.where(b >= 0, b, nb), True)
        for off in range(0, S, seg):
            length = min(seg, S - off)
            sc = buf[:n, :length]
            _index_scores_seg[(n, triton.cdiv(length, 64))](iq[r0:r1], wts[r0:r1], keys, pos[r0:r1], sc, S, off,
                                                           length, buf.stride(0), ratio, HI=HI, DI=DI, BS=64,
                                                           num_warps=4)
            if best is not None:                               # the source layer's block maxima (unmasked)
                padded = sc if length % block == 0 else torch.nn.functional.pad(sc, (0, block - length % block),
                                                                                value=float("-inf"))
                best[:, off // block: off // block + padded.shape[1] // block] = padded.view(n, -1, block).amax(-1)
            if flags is not None:                              # the later layers: only the candidate blocks
                keep = flags[:, off // block: off // block + -(-length // block)].repeat_interleave(block, 1)
                sc = sc.masked_fill(~keep[:, :length], float("-inf"))
            kk = min(k, length)
            v, i = torch.topk(sc, kk, dim=1, sorted=False)
            vals, pick = torch.topk(torch.cat([vals, v], dim=1), k, dim=1, sorted=False)
            ids = torch.gather(torch.cat([ids, i + off], dim=1), 1, pick)
        ids = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(ids, S), ids)   # invisible: dropped
        ids = torch.sort(ids, dim=1).values
        idx_out[r0:r1, :k] = torch.where(ids >= S, torch.full_like(ids, -1), ids).int()
        if best is not None:
            newest = ((pos[r0:r1] + 1) // ratio - 1).clamp(min=0) // block
            best.scatter_(1, newest[:, None].long(), float("inf"))
            v, i = torch.topk(best, min(candidates, nb), dim=1)
            cand_out[r0:r1, :i.shape[1]] = torch.where(torch.isinf(v) & (v < 0), torch.full_like(i, -1), i)
    return idx_out, cand_out


def index_select(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int,
                 topk: int) -> torch.Tensor:
    """The compressed entries each row attends to: int32 [R, topk], ascending, -1 padded (all visible when <= topk)."""

    return top_entries(index_scores(iq, wts, keys, pos, ratio), topk)

@triton.jit
def _route(L, BIAS, PICK, WTS, scale, E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr):
    """sqrt(softplus) scores; the K best of score + bias (lowest id on ties); weights = scores renormalized x scale."""

    r = tl.program_id(0)
    e = tl.arange(0, EP)
    ok = e < E
    x = tl.load(L + r * E + e, mask=ok, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
    sc = tl.sqrt(sp)
    choice = tl.where(ok, sc + tl.load(BIAS + e, mask=ok, other=0.0), float("-inf"))
    total = 0.0
    for k in tl.static_range(K):
        best = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == best, e, EP), axis=0)
        w = tl.sum(tl.where(e == idx, sc, 0.0), axis=0)
        tl.store(PICK + r * K + k, idx)
        tl.store(WTS + r * K + k, w)
        total += w
        choice = tl.where(e == idx, float("-inf"), choice)
    kk = tl.arange(0, KP)
    w = tl.load(WTS + r * K + kk, mask=kk < K, other=0.0)
    tl.store(WTS + r * K + kk, w / total * scale, mask=kk < K)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    R, E = logits.shape
    pick = torch.empty((R, k), dtype=torch.int32, device=logits.device)
    wts = torch.empty((R, k), dtype=torch.float32, device=logits.device)
    _route[(R,)](logits.contiguous(), bias, pick, wts, scale, E=E, EP=triton.next_power_of_2(E), K=k,
                     KP=triton.next_power_of_2(k), num_warps=4)
    return pick, wts


@triton.jit
def _router_logits(X, W, OUT, R, E: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BE: tl.constexpr,
                   BK: tl.constexpr):
    """OUT [R, E] fp32 = X [R, D] fp16 @ W [E, D]^T fp16; a row's K order is fixed and MMA rows are independent."""

    rb = tl.program_id(0)
    eb = tl.program_id(1)
    r = rb * BR + tl.arange(0, BR)
    e = eb * BE + tl.arange(0, BE)
    k = tl.arange(0, BK)
    acc = tl.zeros((BR, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + r[:, None] * D + k0 + k[None, :], mask=(r < R)[:, None], other=0.0)
        w = tl.load(W + e[:, None] * D + k0 + k[None, :], mask=(e < E)[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + r[:, None] * E + e[None, :], acc, mask=(r < R)[:, None] & (e < E)[None, :])


def router_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 router logits; x bf16/fp16 [R, D] (bf16 -> fp16 exact for normed rows), w fp16 [E, D]."""

    R, D = x.shape
    E = w.shape[0]
    out = torch.empty((R, E), dtype=torch.float32, device=x.device)
    BR, BE, BK = 16, 32, 256
    _router_logits[(triton.cdiv(R, BR), triton.cdiv(E, BE))](x.half().contiguous(), w, out, R, E=E, D=D, BR=BR, BE=BE,
                                                             BK=BK, num_warps=4)
    return out


@triton.jit
def _engram_gate(X, KV, QW, KW, OUT, eps, clamp, D: tl.constexpr, S: tl.constexpr, CH: tl.constexpr):
    """One (row, stream): RMS-cosine gate of the stream against its key, then stream + gate * value (fixed order)."""

    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, CH)
    hh = 0.0
    kk = 0.0
    dot = 0.0
    for c in range(D // CH):
        o = c * CH + d
        h = tl.load(X + (r * S + s) * D + o).to(tl.float32)
        key = tl.load(KV + r * (S + 1) * D + s * D + o).to(tl.float32)
        q = tl.load(QW + s * D + o)
        k = tl.load(KW + s * D + o)
        hh += tl.sum(h * h, axis=0)
        kk += tl.sum(key * key, axis=0)
        dot += tl.sum(h * q * k * key, axis=0)
    dot = dot * (1.0 / tl.sqrt(hh / D + eps)) * (1.0 / tl.sqrt(kk / D + eps)) / tl.sqrt(D * 1.0)
    g = tl.sqrt(tl.maximum(tl.abs(dot), clamp))
    g = tl.where(dot < 0.0, -g, g)
    gate = 1.0 / (1.0 + tl.exp(-g))
    for c in range(D // CH):
        o = c * CH + d
        h = tl.load(X + (r * S + s) * D + o).to(tl.float32)
        val = tl.load(KV + r * (S + 1) * D + S * D + o).to(tl.float32)
        tl.store(OUT + (r * S + s) * D + o, (h + gate * val).to(tl.bfloat16))


def engram_gate(X: torch.Tensor, kv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, eps: float,
                clamp: float = 1e-6) -> torch.Tensor:
    """X bf16 [R, S, D], kv [R, (S + 1) D] (S keys then the value), q/k weights fp32 [S, D] -> bf16 [R, S, D]."""

    R, S, D = X.shape
    out = torch.empty_like(X)
    _engram_gate[(R, S)](X.contiguous(), kv.contiguous(), qw, kw, out, eps, clamp, D=D, S=S, CH=1024, num_warps=4)
    return out
