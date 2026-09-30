"""Full GLM-5.3's fused, capturable forward (milestones M3-M5): static buffers, device positions, row-invariant kernels.

Every kernel computes each row alone (fixed reduction orders, shapes that do not depend on the window), so a verify
window's row r has exactly the bits a serial step gives the same token - and a prompt chunk's rows too. The GPU work of
a window (``compute``) has no host syncs, so decode and verify windows replay as CUDA graphs (graphs.py).

Attention always runs over a per-row key list: a row at position p < index_topk sees keys 0..p (read straight from the
cache), a later row its indexer's top-index_topk keys in ascending order (the full layer before it chose them; shared
layers reuse the choice). The indexer's selection breaks score ties by position (relu gives exact zeros), so the
chosen set never depends on how many rows share the window.

The glm5_next (GLM-5.3-Flash) kernels this reuses: RMSNorm, router, top-k, residual add, latent expand and merge.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3 import prefill as x3prefill
from tensorfold.families.glm5_next.cuda import glue, latent

from ..config import Config
from .weights import Layer, MtpHead

# attention tilings (tools/bench_attn_prefill.py on GB10), by window class - a function of the class alone, so every
# decode window (serial or verify) shares one arithmetic: (keys a chunk program, keys a tile, warps, stages)
ATTN_DECODE = (256, 32, 4, 2)
ATTN_PROMPT = (1024, 64, 8, 2)
BT = 64                  # indexer: keys per scoring program
MAX_ROWS = 128           # widest call of the row-invariant EXL3 linear; wider windows use the prompt GEMM
PROMPT_ROWS = int(os.environ.get("TF_GLM53_PROMPT_ROWS", "4096"))   # prompt chunk: experts read once per chunk
PREFILL_REDUCE = os.environ.get("TF_GLM53_PREFILL_REDUCE", "ring")   # ring | rs (exact reduce-scatter)
PROMPT_OVERLAP = os.environ.get("TF_GLM53_PROMPT_OVERLAP", "1") != "0"   # two micro-batches, comm under compute
SEL_ROWS = 128           # indexer top-k in blocks of rows (bounded score buffer at long contexts)
RB = 16                  # rows a program in the absorb / expand kernels (wide windows)

# MTP inputs (alignment A/B), "<hidden>/<chain>": the target hidden it reads (raw last-layer rows or final-normed)
# and the hidden a draft chain passes on (raw MTP-layer rows or shared_head-normed); per request as ``mtp_mode``
MTP_MODE = os.environ.get("TF_GLM53_MTP", "normed/normed")
MTP_MODES = ("raw/raw", "raw/normed", "normed/raw", "normed/normed",
             "raw/raw:full", "raw/normed:full", "normed/raw:full", "normed/normed:full",   # ":full": full-vocab drafts
             "dflash", "auto")                    # DFlash2 drafter (dflash.py); auto: MTP or DFlash2 each round
FAST_ROWS = 16           # windows up to this many rows reduce over RoCE (when available); prompt chunks use NCCL
DRAFT_VOCAB = int(os.environ.get("TF_GLM53_DRAFT_VOCAB", "32768"))  # draft head: the lowest ids (BPE: most frequent)
SPECIALS = 128           # ... plus the vocabulary's last ids (GLM's special tokens)
TUNE = os.environ.get("TF_GLM53_TUNE", "1") != "0"


# ------------------------------------------------------------------------------------------------ kernels ---
@triton.jit
def _rope_pair(a, b, cos, sin):
    return a * cos - b * sin, b * cos + a * sin


@triton.jit
def _kv_write(KVA, kva_stride, NW, LC, POS, INV, eps, LW: tl.constexpr, RD: tl.constexpr, DCP: tl.constexpr = 1,
              RANK: tl.constexpr = 0):
    """Row r at position POS + r: cache[p, :LW] = RMSNorm(kva[:LW]), cache[p, LW:] = RoPE(kva[LW:LW + RD]) (GLM's
    interleaved pairs written evens then odds). DCP > 1: only the owner of p (p % DCP) stores it, at slot p // DCP."""
    r = tl.program_id(0)
    pg = (tl.load(POS) + r).to(tl.int64)
    if DCP > 1:
        if pg % DCP != RANK:
            return
    p = pg // DCP
    ang_p = pg
    k = tl.arange(0, LW)
    x = tl.load(KVA + r * kva_stride + k).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / LW + eps)
    w = tl.load(NW + k).to(tl.float32)
    tl.store(LC + p * (LW + RD) + k, (w * (x * rinv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))
    i = tl.arange(0, RD // 2)
    a = tl.load(KVA + r * kva_stride + LW + 2 * i).to(tl.float32)
    b = tl.load(KVA + r * kva_stride + LW + 2 * i + 1).to(tl.float32)
    ang = ang_p.to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
    tl.store(LC + p * (LW + RD) + LW + i, ra.to(tl.bfloat16))
    tl.store(LC + p * (LW + RD) + LW + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _absorb(Q, WK, INV, QA, QR, POS, R, H: tl.constexpr, QD: tl.constexpr, NOPE: tl.constexpr,
            NOPE_P: tl.constexpr, RD: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr, RBK: tl.constexpr):
    """Program (head, latent block, row block): QA[r, h, n] = q_nope[r, h] . WK[h, :, n] (one fp32 sum), and on
    latent block 0 the rotated q_rot[r, h] at position POS + r. Rows never meet, so a row's bits never depend on R."""
    h = tl.program_id(0)
    nb = tl.program_id(1)
    r0 = tl.program_id(2) * RBK
    k = tl.arange(0, NOPE_P)
    n = nb * BN + tl.arange(0, BN)
    kok = k < NOPE
    w = tl.load(WK + (h * NOPE + k[:, None]) * LW + n[None, :], mask=kok[:, None], other=0.0).to(tl.float32)
    P = tl.load(POS)
    i = tl.arange(0, RD // 2)
    inv = tl.load(INV + i)
    for j in range(RBK):
        r = r0 + j
        if r < R:
            q = tl.load(Q + (r * H + h) * QD + k, mask=kok, other=0.0).to(tl.float32)
            acc = tl.sum(q[:, None] * w, axis=0)
            tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))
            if nb == 0:
                a = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i).to(tl.float32)
                b = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i + 1).to(tl.float32)
                ang = (P + r).to(tl.float32) * inv
                ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
                tl.store(QR + (r * H + h) * RD + i, ra.to(tl.bfloat16))
                tl.store(QR + (r * H + h) * RD + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _expand(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr,
            RBK: tl.constexpr):
    """Program (head, output block, row block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] (glm5_next's expand,
    with a row-block grid for wide windows)."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r0 = tl.program_id(2) * RBK
    k = tl.arange(0, LW)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)
    for j in range(RBK):
        r = r0 + j
        if r < R:
            o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
            tl.store(OUT + (r * H + h) * DV + n, tl.sum(w * o[None, :], axis=1).to(tl.bfloat16))


@triton.jit
def _qrope(Q, INV, QR, POS, H: tl.constexpr, QD: tl.constexpr, NOPE: tl.constexpr, RD: tl.constexpr):
    """Program (row, head): q_rot at position POS + r (the wide-window companion of the batched absorb)."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, RD // 2)
    a = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i).to(tl.float32)
    b = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i + 1).to(tl.float32)
    ang = (tl.load(POS) + r).to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
    tl.store(QR + (r * H + h) * RD + i, ra.to(tl.bfloat16))
    tl.store(QR + (r * H + h) * RD + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _attn_chunks(QA, QR, LC, TOK, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, RD: tl.constexpr,
                 K: tl.constexpr, CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr):
    """Program (row, chunk): all H heads of row r over entries [c CHK, (c + 1) CHK) of its key list - keys 0..p
    themselves while p < K, else TOK[r] (ascending). Scores are latent . latent + rope . rope; values are latents."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    p = tl.load(POS) + r
    n = tl.minimum(p + 1, K)
    hh = tl.arange(0, H)
    kl = tl.arange(0, LW)
    kr = tl.arange(0, RD)
    m = tl.full((H,), float("-inf"), tl.float32)
    l = tl.zeros((H,), tl.float32)
    o = tl.zeros((H, LW), tl.float32)
    start = c * CHK
    if start < n:
        ql = tl.load(QA + (r * H + hh[:, None]) * LW + kl[None, :])
        qr = tl.load(QR + (r * H + hh[:, None]) * RD + kr[None, :])
        for t in range(CHK // KTT):
            idx = start + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            if p < K:
                key = idx.to(tl.int64)
            else:
                key = tl.load(TOK + r * K + idx, mask=ok, other=0).to(tl.int64)
            kv = tl.load(LC + key[:, None] * (LW + RD) + kl[None, :], mask=ok[:, None], other=0.0)
            kro = tl.load(LC + key[:, None] * (LW + RD) + LW + kr[None, :], mask=ok[:, None], other=0.0)
            s = (tl.dot(ql, tl.trans(kv)) + tl.dot(qr, tl.trans(kro))) * SCALE
            s = tl.where(ok[None, :], s, float("-inf"))
            tile_m = tl.max(s, 1)
            active = tile_m != float("-inf")
            next_m = tl.where(active, tl.maximum(m, tile_m), m)
            alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            pr = tl.where(ok[None, :] & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
            o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kv)
            l = l * alpha + tl.sum(pr, 1)
            m = next_m
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + kl[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _attn_dcp(QALL, LC, TOK, CNT, POS, PO, PM, PL, R, H: tl.constexpr, G: tl.constexpr, LW: tl.constexpr,
              RD: tl.constexpr, K: tl.constexpr, CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr,
              DCP: tl.constexpr, RANK: tl.constexpr):
    """Program (row, chunk, head group g): H heads of rank g's gathered queries (QALL [G, R, H, LW + RD]) over this
    rank's keys of row r - its slots of positions 0..p while p < K, else its share of the row's selected keys
    (TOK, CNT). Partials go to head g * H + h (merged over chunks, then over ranks)."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    grp = tl.program_id(2)
    p = tl.load(POS) + r
    if p < K:
        n = tl.where(p >= RANK, (p - RANK) // DCP + 1, 0)
    else:
        n = tl.load(CNT + r)
    hh = tl.arange(0, H)
    kl = tl.arange(0, LW)
    kr = tl.arange(0, RD)
    m = tl.full((H,), float("-inf"), tl.float32)
    l = tl.zeros((H,), tl.float32)
    o = tl.zeros((H, LW), tl.float32)
    start = c * CHK
    if start < n:
        qb = QALL + ((grp * R + r) * H + hh[:, None]) * (LW + RD)
        ql = tl.load(qb + kl[None, :])
        qr = tl.load(qb + LW + kr[None, :])
        for t in range(CHK // KTT):
            idx = start + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            if p < K:
                key = idx.to(tl.int64)
            else:
                key = tl.load(TOK + r * K + idx, mask=ok, other=0).to(tl.int64)
            kv = tl.load(LC + key[:, None] * (LW + RD) + kl[None, :], mask=ok[:, None], other=0.0)
            kro = tl.load(LC + key[:, None] * (LW + RD) + LW + kr[None, :], mask=ok[:, None], other=0.0)
            s = (tl.dot(ql, tl.trans(kv)) + tl.dot(qr, tl.trans(kro))) * SCALE
            s = tl.where(ok[None, :], s, float("-inf"))
            tile_m = tl.max(s, 1)
            active = tile_m != float("-inf")
            next_m = tl.where(active, tl.maximum(m, tile_m), m)
            alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            pr = tl.where(ok[None, :] & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
            o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kv)
            l = l * alpha + tl.sum(pr, 1)
            m = next_m
    base = (c * R + r) * (G * H) + grp * H + hh
    tl.store(PO + base[:, None] * LW + kl[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _merge_lse(PO, PM, PL, OSEND, LSEND, R, HT: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
               NCH: tl.constexpr):
    """Program (row, head of all HT): this rank's chunk partials in chunk order -> the normalized output (bf16) and its
    log-sum-exp, laid out [destination rank = head // H, row, head % H] for the exchange; no keys: 0 and -inf."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * HT + h
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
    dst = ((h // H) * R + r) * H + h % H
    has = l > 0.0
    tl.store(OSEND + dst * LW + k, tl.where(has, o / tl.where(has, l, 1.0), 0.0).to(tl.bfloat16))
    tl.store(LSEND + dst, tl.where(has, m + tl.log(tl.where(has, l, 1.0)), float("-inf")))


@triton.jit
def _dcp_combine(ORECV, LRECV, OUT, R, SS, SSL, H: tl.constexpr, LW: tl.constexpr, WORLD: tl.constexpr):
    """Program (row, own head): every rank's normalized partial for this head merged in rank order by their
    log-sum-exps (ORECV[src] = [R, H, LW] at stride SS, LRECV[src] = [R, H] at stride SSL) -> OUT [R, H, LW] bf16."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    k = tl.arange(0, LW)
    mx = float("-inf")
    for src in tl.static_range(WORLD):
        mx = tl.maximum(mx, tl.load(LRECV + src * SSL + r * H + h))
    acc = tl.zeros((LW,), tl.float32)
    den = 0.0
    for src in tl.static_range(WORLD):
        ls = tl.load(LRECV + src * SSL + r * H + h)
        wgt = tl.where(ls == float("-inf"), 0.0, tl.exp(ls - mx))
        acc = acc + wgt * tl.load(ORECV + src * SS + (r * H + h) * LW + k).to(tl.float32)
        den = den + wgt
    tl.store(OUT + (r * H + h) * LW + k, (acc / den).to(tl.bfloat16))


@triton.jit
def _ik_write(IK, NW, NB, INV, IC, POS, eps, D: tl.constexpr, RD: tl.constexpr, DCP: tl.constexpr = 1,
              RANK: tl.constexpr = 0):
    """Index key of row r: LayerNorm(ik) (fp32, with bias) to bf16, RoPE on its first RD dims, into IC[POS + r]
    (DCP > 1: the owner's slot p // DCP only)."""
    r = tl.program_id(0)
    pg = (tl.load(POS) + r).to(tl.int64)
    if DCP > 1:
        if pg % DCP != RANK:
            return
    p = pg // DCP
    d = tl.arange(0, D)
    x = tl.load(IK + r * D + d)
    mu = tl.sum(x, axis=0) / D
    xc = x - mu
    rs = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=0) / D + eps)
    y = (xc * rs * tl.load(NW + d).to(tl.float32) + tl.load(NB + d).to(tl.float32)).to(tl.bfloat16)
    tl.store(IC + p * D + d, y, mask=d >= RD)
    i = tl.arange(0, RD // 2)                        # the rotated dims: the same values, read as even/odd pairs
    e, o = 2 * i, 2 * i + 1
    ye = ((tl.load(IK + r * D + e) - mu) * rs * tl.load(NW + e).to(tl.float32)
          + tl.load(NB + e).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    yo = ((tl.load(IK + r * D + o) - mu) * rs * tl.load(NW + o).to(tl.float32)
          + tl.load(NB + o).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ang = pg.to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(ye, yo, tl.cos(ang), tl.sin(ang))
    tl.store(IC + p * D + i, ra.to(tl.bfloat16))
    tl.store(IC + p * D + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _iq_rope(Q, INV, POS, NH: tl.constexpr, D: tl.constexpr, RD: tl.constexpr):
    """Program (row, head): the index query's first RD dims rotated at position POS + r, in place (bf16)."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    base = Q + (r * NH + h) * D
    i = tl.arange(0, RD // 2)
    a = tl.load(base + 2 * i).to(tl.float32)
    b = tl.load(base + 2 * i + 1).to(tl.float32)
    ang = (tl.load(POS) + r).to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
    tl.debug_barrier()
    tl.store(base + i, ra.to(tl.bfloat16))
    tl.store(base + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _index_scores(Q, W, IC, POS, OUT, T, R0, NH: tl.constexpr, D: tl.constexpr, BTT: tl.constexpr,
                  WSCALE: tl.constexpr, QSCALE: tl.constexpr, DCP: tl.constexpr = 1, RANK: tl.constexpr = 0):
    """Program (row, key block): score[t] = sum_h w[h] relu(q[h] . k[t] QSCALE) for keys t <= POS + r, packed with
    the key into one int64 that orders by score, then lower position first (so top-k is tie-free); later keys -> min."""
    r = tl.program_id(0)                     # row r of this block: window row R0 + r
    t0 = tl.program_id(1) * BTT
    p = tl.load(POS) + R0 + r
    t = t0 + tl.arange(0, BTT)                        # this rank's cache slots; their global positions:
    g = t * DCP + RANK
    ok = (g <= p) & (t < T)
    lo = (0x7FFFFFFF - g).to(tl.int64)
    if t0 * DCP + RANK <= p:
        hh = tl.arange(0, NH)
        d = tl.arange(0, D)
        q = tl.load(Q + (r * NH + hh[:, None]) * D + d[None, :])
        k = tl.load(IC + t[:, None].to(tl.int64) * D + d[None, :], mask=ok[:, None], other=0.0)
        s = tl.maximum(tl.dot(q, tl.trans(k)) * QSCALE, 0.0)                         # [NH, BTT]
        w = tl.load(W + r * NH + hh) * WSCALE
        sc = tl.sum(w[:, None] * s, axis=0)
        bits = sc.to(tl.int32, bitcast=True)
        key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)             # monotone int of the fp32
        packed = (key << 32) | lo
        packed = tl.where(ok, packed, -9223372036854775807)
    else:
        packed = tl.full((BTT,), -9223372036854775807, tl.int64)
    tl.store(OUT + r * T + t, packed, mask=t < T)


@triton.jit
def _swiglu2(G, U, OUT, W: tl.constexpr, BLOCK: tl.constexpr):
    """bf16(bf16(silu(g)) * u) for separate gate and up rows."""
    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + r * W + d).to(tl.float32)
    u = tl.load(U + r * W + d).to(tl.float32)
    tl.store(OUT + r * W + d, ((g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16))


@triton.jit
def _cat2(A, B, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + r * 2 * D + d, tl.load(A + r * D + d))
    tl.store(OUT + r * 2 * D + D + d, tl.load(B + r * D + d))


# -------------------------------------------------------------------------------------------------- state ---
def inv_freq(dim: int, theta: float, device) -> torch.Tensor:
    return (1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))).contiguous()


class Weights:
    """The rank's layers plus what the fused kernels want precomputed (absorb blocks, fp32 biases)."""

    def __init__(self, cfg: Config, rank: int, world: int, comm, embed, final_norm, lm_head, layers: list[Layer],
                 mtp: MtpHead | None) -> None:
        self.cfg, self.rank, self.world, self.comm = cfg, rank, world, comm
        self.embed, self.final_norm, self.lm_head = embed.contiguous(), final_norm, lm_head.contiguous()
        self.layers, self.mtp = layers, mtp
        self.heads = cfg.num_attention_heads // world
        self.device = embed.device
        self.inv = inv_freq(cfg.qk_rope_head_dim, cfg.rope_theta, self.device)
        nope, vd, lw = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        for L in layers + ([mtp.layer] if mtp is not None else []):
            kvb = L.kv_b.view(self.heads, nope + vd, lw)
            L.extra["wk"] = kvb[:, :nope, :].contiguous()
            L.extra["wv"] = kvb[:, nope:, :].contiguous()
            L.kv_b = None
            if L.router is not None:
                L.extra["bias"] = L.router[1].float().contiguous()
            if L.indexer is not None:
                L.extra["ik_w"], L.extra["ik_b"] = (t.contiguous() for t in L.indexer["k_norm"])
        if mtp is not None:
            mtp.eh_proj = mtp.eh_proj.contiguous()
        ex = next((L.experts for L in layers if L.experts is not None), None)
        self.expert_shape = ex
        self.fast = None                 # RoceReduce for decode windows (engine sets it)
        self.tap_slot: dict[int, int] = {}   # DFlash2: target layer -> slot in Buffers.taps (engine sets it)
        self.dcp = 1                     # decode context parallelism: KV positions interleaved over the ranks
        self.vocab_off = rank * self.lm_head.shape[0]
        self.draft_head = self.draft_ids = None
        if TUNE:
            self.tuned = tune_linears([lin for L in layers + ([mtp.layer] if mtp is not None else [])
                                       for lin in linears(L)])

    def set_draft_head(self, rows: torch.Tensor, ids: torch.Tensor) -> None:
        """This rank's share of the reduced draft vocabulary: lm_head rows (bf16) and their global ids."""
        self.draft_head, self.draft_ids = rows.contiguous(), ids.to(torch.long).contiguous()


def linears(L: Layer) -> list:
    out = [L.q_a, L.kv_a, L.q_b, L.o_proj, *L.shared.values()]
    if L.indexer is not None:
        out.append(L.indexer["wq_b"])
    return out


def tune_linears(lins: list, rows: int = 3, iters: int = 20) -> dict:
    """Pick each linear shape's (K splits, warps) by timing R-row calls (the decode window). Any choice is a function
    of the shape alone, so rows stay independent; layers of a shape rotate so the timing reads DRAM, not L2."""
    groups: dict[tuple, list] = {}
    for lin in lins:
        groups.setdefault((lin.k, lin.n, lin.k2, lin.codebook, lin.layout), []).append(lin)
    chosen = {}
    for key, ls in groups.items():
        k = ls[0].k
        kt = k // 16
        x = torch.randn(rows, k, device=ls[0].words.device, dtype=torch.bfloat16) * 0.1
        pool = ls[:16]
        default = ls[0].split
        best, best_t = default, None
        for sk in (1, 2, 4, 8, 16, 32, 64):
            for wk in (2, 4, 8):
                if kt % (sk * wk) or kt // (sk * wk) < 2:
                    continue
                try:
                    for lin in pool:
                        lin.split = (sk, wk)
                        lin(x)
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(iters):
                        for lin in pool:
                            lin(x)
                    e1.record()
                    torch.cuda.synchronize()
                    t = e0.elapsed_time(e1) / (iters * len(pool))
                except Exception:                    # noqa: BLE001  a tiling the kernel refuses
                    continue
                if best_t is None or t < best_t:
                    best, best_t = (sk, wk), t
        for lin in ls:
            lin.split = best
        chosen[key[:2]] = (default, best, best_t)
    return chosen


class State:
    """The committed caches: a 576-wide latent row a token and layer, index keys on full-indexer layers, the same
    for the MTP layer; positions live on the device (``pos``: the target window's first row, ``mpos``: the MTP's)."""

    def __init__(self, w: Weights, capacity: int) -> None:
        c, dev = w.cfg, w.device
        self.capacity = capacity
        lw = c.kv_lora_rank + c.qk_rope_head_dim
        n = len(w.layers)
        local = -(-capacity // w.dcp) + 1               # DCP: this rank's positions p % dcp == rank at p // dcp
        self.kc = [torch.zeros((local, lw), dtype=torch.bfloat16, device=dev) for _ in range(n)]
        self.ic = {L.index: torch.zeros((local, c.index_head_dim), dtype=torch.bfloat16, device=dev)
                   for L in w.layers if L.indexer is not None}
        self.pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mpos = torch.zeros((1,), dtype=torch.int32, device=dev)
        if w.mtp is not None:
            self.mkc = torch.zeros((local, lw), dtype=torch.bfloat16, device=dev)
            self.mic = torch.zeros((local, c.index_head_dim), dtype=torch.bfloat16, device=dev)


class Buffers:
    """Scratch for windows of up to ``rows`` rows (sliced [:R]); ``score_cols``: the indexer's widest key range."""

    def __init__(self, w: Weights, rows: int, score_cols: int) -> None:
        c, dev = w.cfg, w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, H = c.hidden_size, w.heads
        lw, rd, qd = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim + c.qk_rope_head_dim
        self.rows, self.score_cols = rows, score_cols
        self.ws = x3prefill.Workspace()
        self.ids = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.hin = torch.zeros((rows, D), dtype=bf, device=dev)          # MTP: the hidden rows it reads
        self.x = torch.empty((rows, D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.qa = torch.empty((rows, c.q_lora_rank), dtype=bf, device=dev)
        self.qn = torch.empty((rows, c.q_lora_rank), dtype=bf, device=dev)
        self.kva = torch.empty((rows, w.layers[0].kv_a.n), dtype=bf, device=dev)
        self.q = torch.empty((rows, H * qd), dtype=bf, device=dev)
        self.qlat = torch.empty((rows, H, lw), dtype=bf, device=dev)
        self.qrot = torch.empty((rows, H, rd), dtype=bf, device=dev)
        slots = max(c.index_topk // ATTN_DECODE[0] * min(rows, FAST_ROWS), c.index_topk // ATTN_PROMPT[0] * rows)
        slots *= w.dcp                                   # DCP: partials for every rank's heads
        if w.dcp > 1:
            G = w.dcp
            self.qpack = torch.empty((rows, H, lw + rd), dtype=bf, device=dev)
            self.qall = torch.empty((G, rows, H, lw + rd), dtype=bf, device=dev)
            self.osend = torch.empty((G * rows * H * lw,), dtype=bf, device=dev)
            self.lsend = torch.empty((G * rows * H,), dtype=f32, device=dev)
            fan = G if rows <= FAST_ROWS else 1          # decode windows exchange by all-gather (G x the bytes)
            self.orecv = torch.empty((fan * G * rows * H * lw,), dtype=bf, device=dev)
            self.lrecv = torch.empty((fan * G * rows * H,), dtype=f32, device=dev)
            self.cnt = torch.zeros((rows,), dtype=torch.int32, device=dev)
            self.cand = torch.empty((G * min(rows, SEL_ROWS) * c.index_topk,), dtype=torch.int64, device=dev)
        self.po = torch.empty((slots * H * lw,), dtype=f32, device=dev)   # chunk partials: chunks x rows
        self.pm = torch.empty((slots * H,), dtype=f32, device=dev)
        self.pl = torch.empty((slots * H,), dtype=f32, device=dev)
        self.ol = torch.empty((rows, H, lw), dtype=bf, device=dev)
        self.o = torch.empty((rows, H * c.v_head_dim), dtype=bf, device=dev)
        self.dummy = torch.zeros((1,), dtype=torch.int32, device=dev)
        # indexer
        nh, idd = c.index_n_heads, c.index_head_dim
        self.ik = torch.empty((rows, idd), dtype=f32, device=dev)
        self.iw = torch.empty((rows, nh), dtype=f32, device=dev)
        self.iq = torch.empty((rows, nh * idd), dtype=bf, device=dev)
        self.sc = torch.empty((min(rows, SEL_ROWS) * (-(-score_cols // w.dcp)),), dtype=torch.int64, device=dev)
        self.tok = torch.zeros((rows, c.index_topk), dtype=torch.int32, device=dev)
        # MLPs
        width = max(c.intermediate_size, c.moe_intermediate_size * max(c.n_shared_experts, 1)) // w.world
        self.g = torch.empty((rows, width), dtype=bf, device=dev)
        self.u = torch.empty((rows, width), dtype=bf, device=dev)
        self.act = torch.empty((rows, width), dtype=bf, device=dev)
        self.mlog = torch.empty((rows, c.n_routed_experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, c.num_experts_per_tok), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, c.num_experts_per_tok), dtype=f32, device=dev)
        ex = w.expert_shape
        if ex is None:
            self.xs = None
        elif hasattr(ex, "scratch"):                      # shared experts: decode windows up to 128 rows
            self.xs = ex.scratch(min(rows, MAX_ROWS), c.num_experts_per_tok, device=dev)
        else:
            self.xs = x3experts.Scratch(ex, rows, c.num_experts_per_tok, device=dev)
        self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        # partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.red = torch.empty((rows, D), dtype=f32, device=dev)
        self.hpart = torch.empty((rows, D), dtype=bf, device=dev) if rows > FAST_ROWS else None   # prompt halves
        self.hred = torch.empty((rows, D), dtype=bf, device=dev) if rows > FAST_ROWS else None
        self.amax = torch.zeros((min(rows, FAST_ROWS), 4), dtype=f32, device=dev)
        self.amax_all = torch.zeros((w.world * min(rows, FAST_ROWS) * 4,), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        # heads
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.taps = (torch.zeros((rows, len(w.tap_slot) * D), dtype=bf, device=dev) if w.tap_slot else None)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        V = w.lm_head.shape[0]
        hr = min(rows, FAST_ROWS)            # head rows: a window's, or a prompt chunk's last one
        self.lpart = torch.empty((hr, V), dtype=f32, device=dev)
        self.lgath = torch.empty((w.world * hr * V,), dtype=f32, device=dev)
        self.logits = torch.empty((hr, V * w.world), dtype=f32, device=dev)
        self.argmax = torch.zeros((hr,), dtype=torch.long, device=dev)
        # MTP
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mh = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mx32 = torch.empty((rows, D), dtype=f32, device=dev)


# ------------------------------------------------------------------------------------------------- blocks ---
def lin(layer, b: "Buffers", x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """An EXL3 linear: the row-invariant kernel up to MAX_ROWS rows, the prompt GEMM (weights decoded once) above."""
    if x.shape[0] <= MAX_ROWS:
        return layer(x, out=out)
    return x3prefill.matmul(layer, x, out, b.ws)


def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """The window's fp32 partials of every rank, summed in rank order: [1, R, D] from the RoCE one-shot reduce
    (decode windows), else every rank's partial [world, R, D] for the consumer to add in rank order."""
    d = b.part.shape[1]
    if w.world == 1:
        return b.part[:R].view(1, R, d)
    if w.fast is not None and R <= FAST_ROWS:
        w.fast.all_reduce(b.part[:R], b.red[:R])
        return b.red[:R].view(1, R, d)
    if R > FAST_ROWS and PREFILL_REDUCE == "ring" and hasattr(w.comm, "all_reduce"):
        # prompt chunks: NCCL's ring all-reduce of bf16 partials (ranks alike; not row-invariant, prompts need not be)
        h = b.part[:R].to(torch.bfloat16)
        hs = torch.empty_like(h)
        w.comm.all_reduce(h, hs)
        b.red[:R].copy_(hs)
        return b.red[:R].view(1, R, d)
    if R >= 64 and hasattr(w.comm, "all_to_all") and d % w.world == 0:
        # exact reduce-scatter: column quarters to their owners, summed there in rank order, then gathered - the
        # all-gather + rank-order sum's own bits for about half the bytes
        q = d // w.world
        send = b.part[:R].view(R, w.world, q).permute(1, 0, 2).contiguous().view(w.world, R * q)
        recv = torch.empty_like(send)
        w.comm.all_to_all(send, recv)
        mine = recv[0].clone()
        for r in range(1, w.world):
            mine += recv[r]
        g = b.gath[:w.world * R * q].view(w.world, R * q)
        w.comm.all_gather(mine, g)
        b.red[:R].view(R, w.world, q).copy_(g.view(w.world, R, q).permute(1, 0, 2))
        return b.red[:R].view(1, R, d)
    out = b.gath[:w.world * R * d]
    w.comm.all_gather(b.part[:R].reshape(-1), out)
    return out.view(w.world, R, d)


def dcp_gather(w: Weights, x: torch.Tensor, out: torch.Tensor, small: bool) -> None:
    """Every rank's x in rank order (RoCE one-shot for decode windows when available, else NCCL)."""
    if small and w.fast is not None:
        w.fast.all_gather(x, out)
    else:
        w.comm.all_gather(x.reshape(-1), out.reshape(-1))


def select(w: Weights, b: Buffers, icache: torch.Tensor, pos: torch.Tensor, R: int, T: int) -> None:
    """The indexer's choice for the window's rows past index_topk: b.tok[:R] (ascending). ``T``: keys scored (the
    host's bound on the window's last position + 1; a captured graph uses its bucket). DCP: each rank scores its own
    slots, keeps its top index_topk, the candidates are gathered and every rank takes the same global top index_topk
    (keys are tie-free); b.tok then holds this rank's share as local slots (ascending), b.cnt how many."""
    c = w.cfg
    nh, D, K = c.index_n_heads, c.index_head_dim, c.index_topk
    dcp, rank = w.dcp, w.rank if w.dcp > 1 else 0
    Tl = -(-T // dcp)
    for r0 in range(0, R, SEL_ROWS):
        n = min(SEL_ROWS, R - r0)
        sc = b.sc[:n * Tl].view(n, Tl)
        _index_scores[(n, triton.cdiv(Tl, BT))](b.iq[r0:], b.iw[r0:], icache, pos, sc, Tl, R0=r0, NH=nh, D=D,
                                                BTT=BT, WSCALE=nh ** -0.5, QSCALE=D ** -0.5, DCP=dcp, RANK=rank,
                                                num_warps=4)
        if dcp == 1:
            top = torch.topk(sc, K, dim=-1, sorted=False).values
            keys = (0x7FFFFFFF - (top & 0xFFFFFFFF)).to(torch.int32)
            b.tok[r0:r0 + n].copy_(torch.sort(keys, dim=-1).values)
            continue
        kk = min(K, Tl)
        mine = torch.full((n, K), -9223372036854775807, dtype=torch.int64, device=sc.device)
        mine[:, :kk] = torch.topk(sc, kk, dim=-1, sorted=False).values
        allc = b.cand[:dcp * n * K].view(dcp, n, K)
        dcp_gather(w, mine, allc, n <= FAST_ROWS)
        top = torch.topk(allc.permute(1, 0, 2).reshape(n, dcp * K), K, dim=-1, sorted=False).values
        gpos = 0x7FFFFFFF - (top & 0xFFFFFFFF)                                  # global positions, int64
        own = (gpos % dcp) == rank
        key = torch.where(own, gpos // dcp, torch.full_like(gpos, 1 << 40))
        b.tok[r0:r0 + n].copy_(torch.sort(key, dim=-1).values.to(torch.int32))
        b.cnt[r0:r0 + n].copy_(own.sum(-1).to(torch.int32))


def _attention_local(w: Weights, b: Buffers, R: int, cache, pos, nch, chk, kt, nw, ns) -> None:
    """Every key on this rank: this rank's heads over the rows' key lists -> b.ol."""
    c, H = w.cfg, w.heads
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    n = nch * R * H
    _attn_chunks[(R, nch)](b.qlat, b.qrot, cache, b.tok, pos, b.po[:n * lw], b.pm[:n], b.pl[:n], R, H=H, LW=lw,
                           RD=rd, K=c.index_topk, CHK=chk, KTT=kt, SCALE=(nope + rd) ** -0.5, num_warps=nw,
                           num_stages=ns)
    latent._merge[(R, H)](b.po, b.pm, b.pl, b.ol, b.dummy, R, H=H, LW=lw, NCH=nch, SPARSE=False, num_warps=4)


def _attention_dcp(w: Weights, b: Buffers, R: int, cache, pos, nch, chk, kt, nw, ns) -> None:
    """Decode context parallelism: gather every rank's absorbed queries, attend all heads over this rank's keys,
    send each head's normalized partial and log-sum-exp to the rank that owns the head, merge in rank order -> b.ol."""
    c, H, G, rank = w.cfg, w.heads, w.dcp, w.rank
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    small = R <= FAST_ROWS
    qp = b.qpack[:R]
    qp[:, :, :lw].copy_(b.qlat[:R])
    qp[:, :, lw:].copy_(b.qrot[:R])
    qall = b.qall.view(-1)[:G * R * H * (lw + rd)].view(G, R, H, lw + rd)
    dcp_gather(w, qp, qall, small)
    n = nch * R * G * H
    _attn_dcp[(R, nch, G)](qall, cache, b.tok, b.cnt, pos, b.po[:n * lw], b.pm[:n], b.pl[:n], R, H=H, G=G, LW=lw,
                           RD=rd, K=c.index_topk, CHK=chk, KTT=kt, SCALE=(nope + rd) ** -0.5, DCP=G, RANK=rank,
                           num_warps=nw, num_stages=ns)
    osend = b.osend.view(-1)[:G * R * H * lw]
    lsend = b.lsend.view(-1)[:G * R * H]
    _merge_lse[(R, G * H)](b.po, b.pm, b.pl, osend, lsend, R, HT=G * H, H=H, LW=lw, NCH=nch, num_warps=4)
    if small:                                            # all-gather everything, read what is ours
        orecv = b.orecv[:G * G * R * H * lw]
        lrecv = b.lrecv[:G * G * R * H]
        dcp_gather(w, osend, orecv, True)
        dcp_gather(w, lsend, lrecv, True)
        o0, l0, ss, ssl = rank * R * H * lw, rank * R * H, G * R * H * lw, G * R * H
    else:                                                # prompt chunks: NCCL all-to-all of head blocks
        orecv = b.orecv[:G * R * H * lw]
        lrecv = b.lrecv[:G * R * H]
        w.comm.all_to_all(osend.view(G, R * H * lw), orecv.view(G, R * H * lw))
        w.comm.all_to_all(lsend.view(G, R * H), lrecv.view(G, R * H))
        o0, l0, ss, ssl = 0, 0, R * H * lw, R * H
    _dcp_combine[(R, H)](orecv[o0:], lrecv[l0:], b.ol, R, ss, ssl, H=H, LW=lw, WORLD=G, num_warps=4)


def attention(w: Weights, L: Layer, b: Buffers, R: int, cache: torch.Tensor, icache: torch.Tensor | None,
              pos: torch.Tensor, T: int | None) -> torch.Tensor:
    attention_part(w, L, b, R, cache, icache, pos, T)
    return gather(w, b, R)


def attention_part(w: Weights, L: Layer, b: Buffers, R: int, cache: torch.Tensor, icache: torch.Tensor | None,
                   pos: torch.Tensor, T: int | None) -> None:
    """Attention of the window's rows (b.normed): writes this layer's latent (and index key), returns the gathered
    fp32 partials [world, R, hidden]. ``T``: None while every row is below index_topk (no selection)."""
    c = w.cfg
    H = w.heads
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    lin(L.q_a, b, b.normed[:R], b.qa[:R])
    lin(L.kv_a, b, b.normed[:R], b.kva[:R])
    glue.rmsnorm(b.qa[:R], L.q_a_norm, c.rms_norm_eps, b.qn[:R])
    dcp, rank = w.dcp, (w.rank if w.dcp > 1 else 0)
    _kv_write[(R,)](b.kva, b.kva.stride(0), L.kv_a_norm, cache, pos, w.inv, c.rms_norm_eps, LW=lw, RD=rd, DCP=dcp,
                    RANK=rank, num_warps=4)
    if L.indexer is not None:
        ix = L.indexer
        glue.router(b.normed[:R], ix["wk"], b.ik[:R])
        _ik_write[(R,)](b.ik, L.extra["ik_w"], L.extra["ik_b"], w.inv, icache, pos, 1e-6, D=c.index_head_dim, RD=rd,
                        DCP=dcp, RANK=rank, num_warps=4)
        if T is not None:
            glue.router(b.normed[:R], ix["weights_proj"], b.iw[:R])
            lin(ix["wq_b"], b, b.qn[:R], b.iq[:R])
            _iq_rope[(R, c.index_n_heads)](b.iq, w.inv, pos, NH=c.index_n_heads, D=c.index_head_dim, RD=rd,
                                           num_warps=1)
            select(w, b, icache, pos, R, T)
    lin(L.q_b, b, b.qn[:R], b.q[:R])
    rbk = RB if R > RB else R
    wide = R > MAX_ROWS                                  # prompt chunks: tensor-core batched GEMMs (not row-exact)
    if wide:
        qn = b.q[:R].view(R, H, nope + rd)[:, :, :nope].transpose(0, 1)                    # [H, R, nope]
        b.qlat[:R].transpose(0, 1).copy_(torch.bmm(qn, L.extra["wk"]))                     # [H, R, lw]
        _qrope[(R, H)](b.q, w.inv, b.qrot, pos, H=H, QD=nope + rd, NOPE=nope, RD=rd, num_warps=1)
    else:
        _absorb[(H, lw // 32, triton.cdiv(R, rbk))](b.q, L.extra["wk"], w.inv, b.qlat, b.qrot, pos, R, H=H,
                                                    QD=nope + rd, NOPE=nope, NOPE_P=triton.next_power_of_2(nope),
                                                    RD=rd, LW=lw, BN=32, RBK=rbk, num_warps=4)
    chk, kt, nw, ns = ATTN_DECODE if R <= FAST_ROWS else ATTN_PROMPT
    nch = c.index_topk // chk
    if dcp > 1:
        _attention_dcp(w, b, R, cache, pos, nch, chk, kt, nw, ns)
    else:
        _attention_local(w, b, R, cache, pos, nch, chk, kt, nw, ns)
    if wide:
        o = torch.bmm(b.ol[:R].transpose(0, 1), L.extra["wv"].transpose(1, 2))           # [H, R, v]
        b.o[:R].view(R, H, c.v_head_dim).copy_(o.transpose(0, 1))
    else:
        _expand[(H, c.v_head_dim // 16, triton.cdiv(R, rbk))](b.ol, L.extra["wv"], b.o, R, H=H, DV=c.v_head_dim,
                                                              LW=lw, BN=16, RBK=rbk, num_warps=4)
    lin(L.o_proj, b, b.o[:R], b.part[:R])


def mlp(w: Weights, L: Layer, b: Buffers, R: int, out: torch.Tensor) -> None:
    """The dense MLP (layers 0-2) or the shared expert: gate and up (EXL3), SwiGLU, down into ``out`` (fp32)."""
    s = L.shared
    width = s["gate"].n
    g, u, a = (t.view(-1)[:R * width].view(R, width) for t in (b.g, b.u, b.act))
    lin(s["gate"], b, b.normed[:R], g)
    lin(s["up"], b, b.normed[:R], u)
    blk = math.gcd(512, width)
    _swiglu2[(R, width // blk)](g, u, a, W=width, BLOCK=blk, num_warps=4)
    lin(s["down"], b, a, out)


def ffn(w: Weights, L: Layer, b: Buffers, R: int) -> torch.Tensor:
    ffn_part(w, L, b, R)
    return gather(w, b, R)


def ffn_part(w: Weights, L: Layer, b: Buffers, R: int) -> None:
    c = w.cfg
    if L.experts is None:
        mlp(w, L, b, R, b.part[:R])
        return
    glue.router(b.normed[:R], L.router[0], b.mlog[:R])
    K = c.num_experts_per_tok
    glue._topk[(R,)](b.mlog, L.extra["bias"], b.pick, b.wts, float(c.routed_scaling_factor), NE=c.n_routed_experts,
                     TOPK=K, SLOTS=K, BLOCK=triton.next_power_of_2(c.n_routed_experts + 1),
                     SLOTP=triton.next_power_of_2(K + 1), NORM=c.norm_topk_prob, num_warps=4)
    if not hasattr(L.experts, "prefill"):
        x3experts.routed(b.normed[:R], b.pick[:R], b.wts[:R], L.experts, b.xs, b.part[:R], R)
    elif R <= b.xs.rows:
        L.experts.decode(b.normed[:R], b.pick[:R], b.wts[:R], b.xs, b.part[:R], R)
    else:                                                # prompt chunk: cuda-exl3's grouped GEMM
        b.part[:R].copy_(L.experts.prefill(b.normed[:R], b.pick[:R], b.wts[:R]))
    mlp(w, L, b, R, b.sy[:R])
    b.part[:R].add_(b.sy[:R])


def layer(w: Weights, L: Layer, b: Buffers, x: torch.Tensor, R: int, cache, icache, pos, T) -> None:
    c = w.cfg
    glue.rmsnorm(x, L.input_norm, c.rms_norm_eps, b.normed[:R])
    glue.residual_add(x, x, attention(w, L, b, R, cache, icache, pos, T))
    glue.rmsnorm(x, L.post_attn_norm, c.rms_norm_eps, b.normed[:R])
    glue.residual_add(x, x, ffn(w, L, b, R))


def head(w: Weights, b: Buffers, x: torch.Tensor, norm: torch.Tensor, R: int, rows: slice | None = None,
         mode: str = "full"):
    """Rows' next-token pick into b.argmax[:n]; "full" also leaves full fp32 logits (every rank alike) in b.logits[:n].
    "argmax": each rank's first maximum over its vocabulary share, exchanged as 16 bytes a row and resolved by lowest
    rank - the full argmax's own choice. "draft": the same over the reduced draft vocabulary."""
    c = w.cfg
    x = x if rows is None else x[rows]
    n = x.shape[0]
    glue.rmsnorm(x, norm, c.rms_norm_eps, b.fnormed[:n])
    if mode == "full":
        V = w.lm_head.shape[0]
        glue.router(b.fnormed[:n], w.lm_head, b.lpart[:n])
        if w.world > 1:
            g = b.lgath[:w.world * n * V]
            w.comm.all_gather(b.lpart[:n].reshape(-1), g)
            b.logits[:n].view(n, w.world, V).copy_(g.view(w.world, n, V).permute(1, 0, 2))
        else:
            b.logits[:n].copy_(b.lpart[:n])
        torch.argmax(b.logits[:n], dim=-1, out=b.argmax[:n])
        return b.logits[:n]
    table = w.draft_head if mode == "draft" else w.lm_head
    V = table.shape[0]
    lg = b.lpart.view(-1)[:n * V].view(n, V)
    glue.router(b.fnormed[:n], table, lg)
    i = torch.argmax(lg, dim=-1)
    a = b.amax[:n]
    a[:, 0] = lg.gather(1, i[:, None])[:, 0]
    a[:, 1] = (w.draft_ids[i] if mode == "draft" else i + w.vocab_off).float()
    if w.world > 1:
        g = b.amax_all[:w.world * n * 4]
        if w.fast is not None:
            w.fast.all_gather(a, g)
        else:
            w.comm.all_gather(a.reshape(-1), g)
        g = g.view(w.world, n, 4)
        best = torch.argmax(g[:, :, 0], dim=0)                        # lowest rank among equal maxima
        b.argmax[:n].copy_(g[:, :, 1].gather(0, best[None])[0].long())
    else:
        b.argmax[:n].copy_(a[:, 1].long())
    return None


def compute(w: Weights, st: State, b: Buffers, R: int, T: int | None, *, logits: str = "all",
            pick: str = "full") -> None:
    """The target's GPU work for rows b.ids[:R] at st.pos .. st.pos + R - 1 (capturable): caches written, final
    hidden in b.hidden[:R]; logits of all rows, the last row ("last") or none."""
    x = b.x[:R]
    torch.index_select(w.embed, 0, b.ids[:R], out=x)
    D = x.shape[1]
    for i, L in enumerate(w.layers):
        layer(w, L, b, x, R, st.kc[i], st.ic.get(L.index), st.pos, T)
        s = w.tap_slot.get(i)
        if s is not None:                                # DFlash2: this layer's output rows
            b.taps[:R, s * D:(s + 1) * D].copy_(x)
    b.hidden[:R].copy_(x)
    if logits == "all":
        head(w, b, x, w.final_norm, R, mode=pick)
    elif logits == "last":
        head(w, b, x, w.final_norm, R, slice(R - 1, R), mode=pick)


class _Half:
    """One micro-batch of a prompt chunk: its buffers, rows, device position and in-flight reduction."""

    def __init__(self, b: Buffers, R: int, pos: torch.Tensor) -> None:
        self.b, self.R, self.pos, self.done = b, R, pos, None


def _reduce_async(w: Weights, h: _Half, comm_stream) -> None:
    """h.b.part -> bf16 -> NCCL ring all-reduce on the comm stream (overlaps the other half's compute)."""
    b, R = h.b, h.R
    b.hpart[:R].copy_(b.part[:R])
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ready)
        w.comm.all_reduce(b.hpart[:R], b.hred[:R])
        h.done = torch.cuda.Event()
        h.done.record(comm_stream)


def _residual(w: Weights, h: _Half) -> None:
    torch.cuda.current_stream().wait_event(h.done)
    x = h.b.x[:h.R]
    glue.residual_add(x, x, h.b.hred[:h.R].view(1, h.R, -1))


def compute_prompt(w: Weights, st: State, b0: Buffers, b1: Buffers, R: int, T: int | None, pos1: torch.Tensor,
                   comm_stream, *, logits: str = "none") -> None:
    """A prompt chunk as two micro-batches whose all-reduces overlap each other's compute (rows [0, h) in b0 at
    st.pos, rows [h, R) in b1 at pos1 = st.pos + h). Ids in b0.ids[:R]; final hidden rows in b0.hidden[:R]."""
    c = w.cfg
    hR = R // 2
    halves = [_Half(b0, hR, st.pos), _Half(b1, R - hR, pos1)]
    b1.ids[:R - hR].copy_(b0.ids[hR:R])
    for h in halves:
        torch.index_select(w.embed, 0, h.b.ids[:h.R], out=h.b.x[:h.R])
    D = c.hidden_size

    def tap(h: _Half, i: int) -> None:                    # layer i's output rows (complete after its residual)
        s = w.tap_slot.get(i)
        if s is not None:
            h.b.taps[:h.R, s * D:(s + 1) * D].copy_(h.b.x[:h.R])

    first = True
    for i, L in enumerate(w.layers):
        for h in halves:                                  # attention: A writes its keys before B attends
            if not first:
                _residual(w, h)
                tap(h, i - 1)
            x = h.b.x[:h.R]
            glue.rmsnorm(x, L.input_norm, c.rms_norm_eps, h.b.normed[:h.R])
            attention_part(w, L, h.b, h.R, st.kc[i], st.ic.get(L.index), h.pos, T)
            _reduce_async(w, h, comm_stream)
        first = False
        for h in halves:
            _residual(w, h)
            x = h.b.x[:h.R]
            glue.rmsnorm(x, L.post_attn_norm, c.rms_norm_eps, h.b.normed[:h.R])
            ffn_part(w, L, h.b, h.R)
            _reduce_async(w, h, comm_stream)
    for h in halves:
        _residual(w, h)
        tap(h, len(w.layers) - 1)
    if b0.taps is not None:
        b0.taps[hR:R].copy_(b1.taps[:R - hR])
    b0.hidden[:hR].copy_(b0.x[:hR])
    b0.hidden[hR:R].copy_(b1.x[:R - hR])
    if logits == "last":
        head(w, b1, b1.x[:R - hR], w.final_norm, R - hR, slice(R - hR - 1, R - hR))
        b0.logits[:1].copy_(b1.logits[:1])


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, T: int | None, *, logits: str = "last",
                zero_first: bool = False, chain_normed: bool = False, draft_full: bool = False) -> None:
    """The MTP layer for rows (b.hin[:n] = the previous position's hidden, b.ids[:n] = the token) at st.mpos ..;
    its output hidden in b.hidden[:n] (raw, or shared_head-normed per MTP_CHAIN) and the last row's logits/argmax."""
    c = w.cfg
    m = w.mtp
    D = c.hidden_size
    torch.index_select(w.embed, 0, b.ids[:n], out=b.me[:n])
    if zero_first:                                   # position 0's embedding is masked (vLLM's DeepSeek MTP)
        b.me[0].zero_()
    glue.rmsnorm(b.me[:n], m.enorm, c.rms_norm_eps, b.mh[:n])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.rms_norm_eps, b.normed[:n])
    _cat2[(n, D // 1024)](b.mh, b.normed, b.mcat, D=D, BLOCK=1024, num_warps=4)
    glue.router(b.mcat[:n], m.eh_proj, b.mx32[:n])
    x = b.x[:n]
    x.copy_(b.mx32[:n])
    layer(w, m.layer, b, x, n, st.mkc, st.mic, st.mpos, T)
    if chain_normed:
        glue.rmsnorm(x, m.head_norm, c.rms_norm_eps, b.hidden[:n])
    else:
        b.hidden[:n].copy_(x)
    mode = "argmax" if draft_full or w.draft_head is None else "draft"
    if logits == "last":
        head(w, b, x, m.head_norm, n, slice(n - 1, n), mode=mode)
    elif logits == "all":
        head(w, b, x, m.head_norm, n, mode=mode)


def target_hidden_for_mtp(w: Weights, b: Buffers, rows: slice, out: torch.Tensor, normed: bool) -> None:
    """The target hidden the MTP reads: raw last-layer rows or final-normed ones."""
    src = b.hidden[rows]
    if normed:
        glue.rmsnorm(src, w.final_norm, w.cfg.rms_norm_eps, out)
    else:
        out.copy_(src)


def bucket(t: int, topk: int) -> int | None:
    """The indexer's key range for a window whose last row sits at t - 1: None while t <= topk, else a power of two."""
    if t <= topk:
        return None
    return max(2 * topk, 1 << (t - 1).bit_length())
