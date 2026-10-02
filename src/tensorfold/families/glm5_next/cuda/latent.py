"""GLM's MLA attention on the 512-wide latent cache; kernels compute each row alone, so window rows keep serial bits (docs/recipes/glm-5.3-flash.md)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import LATENT as ENABLED     # off (TF_GLM_LATENT=0): per-head keys and values, 0.3.5's path, for A/B

L = 512            # GLM-5.3-Flash's latent width (kv_lora_rank); the kernels take the width from the tensors
CHUNK = 512        # keys per chunk program, merged in absolute order
KT = 32            # keys per tensor-core tile
HB = 16            # heads per attention tile (all from one row)
HB_WIDE = 32       # prefill chunks: a rank's 32 heads in one tile read each key once (same bits as 16, tested)


def head_block(R: int) -> int:
    """Heads per attention program: a rank's 32 in prompt chunks (keys read once), 16 in decode windows; 16- and 32-row tiles give the same bits."""
    return HB_WIDE if R >= 64 else HB


# ------------------------------------------------------------------------------------------------ weights ---

def dequant_mlx4(w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, group: int = 64) -> torch.Tensor:
    """MLX affine 4-bit rows -> fp32 [out, in]: w uint32 [out, in / 8], 8 nibbles low to high; s, b [out, in / group]."""

    out, words = w.shape
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
    q = (w.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF                        # [out, words, 8]
    q = q.reshape(out, words * 8).to(torch.float32)
    s = s.to(torch.float32).repeat_interleave(group, dim=1)
    b = b.to(torch.float32).repeat_interleave(group, dim=1)
    return q * s + b


class AbsorbW:
    """One DSA layer's kv_b_proj split per head for the latent path: wk [H, 256, 512], wv [H, 256, 512] bf16."""

    def __init__(self, wk: torch.Tensor, wv: torch.Tensor) -> None:
        if wk.dim() != 3 or wv.dim() != 3 or wk.shape[2] != wv.shape[2]:
            raise ValueError("AbsorbW: expected [heads, dim, latent] blocks")
        self.wk = wk.to(torch.bfloat16).contiguous()
        self.wv = wv.to(torch.bfloat16).contiguous()
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]

    @classmethod
    def from_rows(cls, k_rows: torch.Tensor, v_rows: torch.Tensor, heads: int) -> "AbsorbW":
        """k_rows [heads * qk_dim, latent], v_rows [heads * v_dim, latent], float, in head order."""
        lw = k_rows.shape[1]
        return cls(k_rows.reshape(heads, -1, lw), v_rows.reshape(heads, -1, lw))

    def nbytes(self) -> int:
        return self.wk.numel() * 2 + self.wv.numel() * 2


class AbsorbQ4:
    """kv_b split per head in MLX's affine 4-bit layout: words [H, dim, latent / 8], scales and biases [H, dim, latent / 64], key rows then value rows."""

    def __init__(self, k: tuple, v: tuple, heads: int) -> None:
        def per_head(t):
            w, sc, b = t
            return (w.reshape(heads, -1, w.shape[1]).contiguous(), sc.reshape(heads, -1, sc.shape[1]).contiguous(),
                    b.reshape(heads, -1, b.shape[1]).contiguous())

        self.wkw, self.wks, self.wkb = per_head(k)
        self.wvw, self.wvs, self.wvb = per_head(v)
        self.heads, self.qk_dim = self.wkw.shape[0], self.wkw.shape[1]
        self.v_dim = self.wvw.shape[1]
        self.lw = self.wkw.shape[2] * 8
        if self.wks.shape[2] * 64 != self.lw or self.wvw.shape[2] * 8 != self.lw:
            raise ValueError("AbsorbQ4: expected groups of 64 along the latent")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.wkw, self.wks, self.wkb, self.wvw, self.wvs, self.wvb))


# ----------------------------------------------------------------------------------------- absorb, expand ---

@triton.jit
def _absorb_q(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, column block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n] for every row r, k in one sum."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, D)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :]).to(tl.float32)            # [D, BN]
    for r in range(R):
        q = tl.load(Q + (r * H + h) * D + k).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_v(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] for every row r."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, LW)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)          # [BN, LW]
    for r in range(R):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * o[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


@triton.jit
def _absorb_q4(Q, WW, WS, WB, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, RB: tl.constexpr):
    """Program (head, 64-latent group, RB rows): QA = Q @ W with W = q * s + b per group, weights loaded once for the block and each row summed alike."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    rb = tl.program_id(2)
    d = tl.arange(0, D)
    j = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    words = tl.load(WW + (h * D + d[:, None]) * KW + g * 8 + j[None, :])                   # [D, 8]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (D, 64)).to(tl.float32)
    sc = tl.load(WS + (h * D + d) * KG + g).to(tl.float32)
    bi = tl.load(WB + (h * D + d) * KG + g).to(tl.float32)
    n = g * 64 + tl.arange(0, 64)
    for i in tl.static_range(RB):
        r = rb * RB + i
        ok = r < R
        qv = tl.load(Q + (r * H + h) * D + d, mask=ok & (d >= 0), other=0).to(tl.float32)
        acc = tl.sum((qv * sc)[:, None] * qint, axis=0) + tl.sum(qv * bi, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16), mask=ok & (n >= 0))


@triton.jit
def _expand_v4(OL, WW, WS, WB, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, BN outputs): the 4-bit rows unpacked once to fp32, then each row's output as one fp32 sum over the latent, whatever R is."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    kw = tl.arange(0, KW)
    shifts = tl.arange(0, 8) * 4
    words = tl.load(WW + (h * DV + n[:, None]) * KW + kw[None, :])                          # [BN, KW]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, KG, 64)).to(tl.float32)
    gi = tl.arange(0, KG)
    sc = tl.load(WS + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)            # [BN, KG]
    bi = tl.load(WB + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)
    w = tl.reshape(qint * sc[:, :, None] + bi[:, :, None], (BN, LW))
    k = tl.arange(0, LW)
    for r in range(R):
        x = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


def row_block(R: int) -> int:
    """Rows per program: 1 for decode windows (most parallel), 16 for prefill chunks (weights reused)."""
    return 1 if R <= 16 else 16


def absorb_q(q: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """q [R, H, qk_dim] bf16 -> out [R, H, latent] bf16."""
    R, H, D = q.shape
    if isinstance(a, AbsorbQ4):
        rb = row_block(R)
        _absorb_q4[(H, a.lw // 64, triton.cdiv(R, rb))](q, a.wkw, a.wks, a.wkb, out, R, H=H, D=D, LW=a.lw, RB=rb,
                                                        num_warps=4)
        return out
    BN = 32
    _absorb_q[(H, a.lw // BN)](q, a.wk, out, R, H=H, D=D, LW=a.lw, BN=BN, num_warps=4)
    return out


def expand_v(o_lat: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """o_lat [R, H, latent] bf16 -> out [R, H, v_dim] bf16."""
    R, H, _ = o_lat.shape
    if isinstance(a, AbsorbQ4):
        BN = 16
        _expand_v4[(H, a.v_dim // BN)](o_lat, a.wvw, a.wvs, a.wvb, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN,
                                        num_warps=4)
        return out
    BN = 16
    _expand_v[(H, a.v_dim // BN)](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ caches ---

@triton.jit
def _lat_write(LAT, lat_stride, LC, POS, LW: tl.constexpr):
    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    k = tl.arange(0, LW)
    tl.store(LC + (P + r) * LW + k, tl.load(LAT + r * lat_stride + k))


def latent_write(lat: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor) -> None:
    """lat [R, latent] bf16 rows into cache slots pos .. pos + R - 1 (pos read on the device)."""
    _lat_write[(lat.shape[0],)](lat, lat.stride(0), cache, pos, LW=cache.shape[1], num_warps=4)


# --------------------------------------------------------------------------------------------- attention ---

@triton.jit
def _tile(q, kv, m, l, o, valid, SCALE: tl.constexpr):
    """One key tile for HB heads of one row: kv [KT, 512] is both key and value (the latent)."""
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _dense_chunks(QA, LC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, CH: tl.constexpr,
                  SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH)."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H                                                      # tile rows past the last head are padding
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r                                                     # keys 0 .. P + r are visible to row r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks(QA, LC, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _merge(PO, PM, PL, OUT, CNT, R, H: tl.constexpr, LW: tl.constexpr, NCH: tl.constexpr, SPARSE: tl.constexpr):
    """Program (row, head): the row's chunk partials in chunk order -> OUT[r, h] bf16. Sparse: rows with CNT 0 skip."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    if SPARSE:
        if tl.load(CNT + r) == 0:
            return
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * LW + k)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * LW + k, (o / l).to(tl.bfloat16))


class LatentScratch:
    """Chunk partials for ``part_rows`` (one ``attention`` call) x heads x chunks; queries and latents for ``rows``."""

    def __init__(self, rows: int, heads: int, chunks: int, device, lw: int = L, part_rows: int | None = None) -> None:
        part_rows = rows if part_rows is None else min(rows, part_rows)
        self.rows, self.part_rows, self.heads, self.nch, self.lw = rows, part_rows, heads, chunks, lw
        self.po = torch.empty((chunks * part_rows * heads * lw,), dtype=torch.float32, device=device)
        self.pm = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.qa = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.ol = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.dummy = torch.zeros((1,), dtype=torch.int32, device=device)


def attention(qa: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, s: LatentScratch, *, scale: float,
              nch: int, out: torch.Tensor, hb: int | None = None) -> torch.Tensor:
    """Dense causal attention of qa [R, H, 512] through pos + R - 1 in nch 512-key chunks; a row ignores the others."""
    R, H, LW = qa.shape
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    hb = head_block(R) if hb is None else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a program, not {HB} or {HB_WIDE}")
    _dense_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, pos, s.po[:n * LW], s.pm[:n], s.pl[:n], R, H=H, LW=LW,
                                                CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT, num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float) -> None:
    """Attention of rows with counts > 0 over their selected tokens (ascending, -1 padded), written into out; other rows are left alone."""
    R, H, LW = qa.shape
    W = tokens.shape[1]
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    hb = head_block(R)
    _sparse_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, tokens, counts, po, pm, pl, R, W=W, H=H, LW=LW,
                                                 CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT, num_warps=8, num_stages=1)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


def chunks_for(length: int) -> int:
    return triton.cdiv(length, CHUNK)


# ------------------------------------------------------------------------------------- multi-stream windows ---
# A window of several streams' rows back to back ([s1: rows][s2: rows]...), each stream's latents in its own extent
# of a shared arena [P, LW]: per-row device tables (``segments.SegRows``: position, extent base, dense or sparse)
# drive the kernels below. Every row keeps the arithmetic, tiles and head blocks of today's single-stream kernels,
# so a segment's rows get the bits of that segment run alone on its extent (tests/cuda/test_glm_attn_segments.py).
# Cache reads and writes go through _lat_get / _lat_put (the one place a quantized cache plugs in).

@triton.jit
def _lat_get(LC, rows, ok, k, RS: tl.constexpr):
    """Cache rows ``rows`` (int64 [n]) as a bf16 tile [n, LW]; rows with ok false are not read (zeros)."""
    return tl.load(LC + rows[:, None] * RS + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)


@triton.jit
def _lat_put(LC, row, k, x, RS: tl.constexpr):
    """Cache row ``row`` (int64) <- x [LW] (bf16)."""
    tl.store(LC + row * RS + k, x)


@triton.jit
def _seg_lat_write(LAT, lat_stride, LC, POS, BASE, LW: tl.constexpr):
    """Row r -> arena row base[r] + pos[r]."""
    r = tl.program_id(0)
    row = tl.load(BASE + r).to(tl.int64) + tl.load(POS + r).to(tl.int64)
    k = tl.arange(0, LW)
    _lat_put(LC, row, k, tl.load(LAT + r * lat_stride + k), LW)


def seg_latent_write(lat: torch.Tensor, cache: torch.Tensor, rows) -> None:
    """lat [R, latent] bf16 rows into the arena at each row's base + pos (``rows``: segments.SegRows)."""
    _seg_lat_write[(lat.shape[0],)](lat, lat.stride(0), cache, rows.pos, rows.base, LW=cache.shape[1], num_warps=4)


@triton.jit
def _seg_chunks(QA, LC, POS, BASE, SPR, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr,
                LW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): _dense_chunks for a dense row (keys base + 0 .. base + pos[r], its own
    limit), _sparse_chunks for a sparse one (keys base + its selected tokens, list order): the same tiles, masks and
    _tile math either way, so a row's partials are those of its single-stream kernel."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    B = tl.load(BASE + r).to(tl.int64)
    sp = tl.load(SPR + r) != 0
    n = tl.where(sp, tl.load(CNT + r, mask=sp, other=0), tl.load(POS + r) + 1)   # selected tokens, or keys 0 .. pos
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok & sp, other=0).to(tl.int64)
            tok = tl.where(sp, tok, idx.to(tl.int64))
            kv = _lat_get(LC, B + tok, ok, k, LW)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


def seg_chunks() -> int:
    """Chunks a segmented window's rows take: a sparse row's 2,051 selected tokens, a dense row's keys up to
    SPARSE_FROM - 1 (both 5 of 512); chunks past a row's keys are empty (merged as nothing, the same bits)."""
    from .sparse import SPARSE_FROM, TOKENS

    return triton.cdiv(max(SPARSE_FROM, TOKENS), CHUNK)


SEG_WIDE_ROWS = 8        # segmented windows from this many rows take 32-head tiles (keys read once; fewer programs)


def seg_head_block(R: int) -> int:
    """Heads per program of a segmented window: 16 for a few rows (more programs in flight), a rank's 32 from
    SEG_WIDE_ROWS (each key tile read once for them). 16- and 32-row tiles give a row the same bits (head_block),
    so either matches the single-stream kernels' 16 (tests/cuda/test_glm_attn_segments.py checks both)."""
    return HB_WIDE if R >= SEG_WIDE_ROWS else HB


def seg_attention(qa: torch.Tensor, cache: torch.Tensor, rows, tokens: torch.Tensor | None,
                  counts: torch.Tensor | None, s: LatentScratch, *, scale: float, out: torch.Tensor,
                  hb: int | None = None) -> torch.Tensor:
    """Attention of a segmented window's rows qa [R, H, 512] -> out [R, H, 512]: dense rows over their stream's
    keys 0 .. pos, sparse rows over their selected tokens (``segments.seg_select``: tokens relative to the stream,
    counts), each in its own extent of the arena. One chunk pass and one merge for both kinds; a row gets the bits
    of ``attention`` (dense) or ``sparse_attention`` (sparse) run on its segment alone."""
    R, H, LW = qa.shape
    nch = seg_chunks()
    if nch > s.nch or R > s.rows or LW != s.lw:
        raise ValueError(f"segmented attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.rows}, {s.nch}, {s.lw}")
    if tokens is None:                   # a window without sparse rows (no index): nothing reads the lists
        tokens, counts, W = s.dummy, s.dummy, 1
    else:
        W = tokens.shape[1]
    n = nch * R * H
    hb = seg_head_block(R) if hb is None else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"segmented attention: {hb} heads a tile, not {HB} or {HB_WIDE}")
    _seg_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, rows.pos, rows.base, rows.sparse, tokens, counts,
                                              s.po[:n * LW], s.pm[:n], s.pl[:n], R, W=W, H=H, LW=LW, CH=CHUNK,
                                              SCALE=scale, HBT=hb, KTT=KT, num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out
