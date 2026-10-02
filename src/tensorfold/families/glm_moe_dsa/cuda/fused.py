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
from tensorfold.cuda.exl3 import linear as x3linear
from tensorfold.cuda.exl3 import prefill as x3prefill
from tensorfold.families.glm5_next.cuda import glue, latent
from tensorfold.families.glm_moe_dsa.cuda import topk

from ..config import Config
from .weights import Layer, MtpHead

# attention tilings (tools/bench_attn_prefill.py on GB10), by window class - a function of the class alone, so every
# decode window (serial or verify) shares one arithmetic: (keys a chunk program, keys a tile, warps, stages)
ATTN_DECODE = (256, 32, 4, 2)
ATTN_PROMPT = tuple(int(v) for v in os.environ.get("TF_GLM53_ATTN_PROMPT", "2048,64,8,2").split(","))  # chunk, keys/tile, warps, stages; chunk >= index_topk: one pass
BT = 128                 # indexer: keys per scoring program (128 / 2 warps / 2 stages: same bits as 64/4/3, ~1.14x on GB10)
MAX_ROWS = 128           # widest call of the row-invariant EXL3 linear; wider windows use the prompt GEMM
PROMPT_ROWS = int(os.environ.get("TF_GLM53_PROMPT_ROWS", "4096"))   # prompt chunk: experts read once per chunk
PREFILL_REDUCE = os.environ.get("TF_GLM53_PREFILL_REDUCE", "ring")   # ring | rs (exact reduce-scatter)
PROMPT_OVERLAP = os.environ.get("TF_GLM53_PROMPT_OVERLAP", "1") != "0"   # two micro-batches, comm under compute
# sequence-parallel prompt chunks: reduce-scatter rows, replicated work on a rank's own rows, all-gather (see
# compute_prompt_sp); SP_SELECT=1: the indexer's top-k too (own rows, then the picks are gathered)
PROMPT_SP = os.environ.get("TF_GLM53_PROMPT_SP", "1") == "1"   # sequence-parallel prompt chunks (TP; off under DCP)
SP_SELECT = os.environ.get("TF_GLM53_SP_SELECT", "1") != "0"
SEL_ROWS = 128           # indexer top-k in blocks of rows (bounded score buffer at long contexts)
_UNPACK_MB = int(os.environ.get("TF_GLM53_UNPACK_CACHE_MB", "384"))   # decoded dense weights shared by a chunk's halves (0: off)
_UNPACK_CACHE = x3prefill.UnpackCache(_UNPACK_MB << 20) if _UNPACK_MB > 0 else None
RADIX = os.environ.get("TF_GLM53_RADIX", "1") == "1"   # indexer top-k: one radix-select kernel (same picks as torch.topk + sort)
RADIX_MIN_ROWS = int(os.environ.get("TF_GLM53_RADIX_MIN_ROWS", "32"))   # radix only for blocks of this many rows (prompts);
                                                                         # decode windows keep torch.topk (faster at few rows)
RB = 16                  # rows a program in the absorb / expand kernels (wide windows)

# MTP inputs (alignment A/B), "<hidden>/<chain>": the target hidden it reads (raw last-layer rows or final-normed)
# and the hidden a draft chain passes on (raw MTP-layer rows or shared_head-normed); per request as ``mtp_mode``
MTP_MODE = os.environ.get("TF_GLM53_MTP", "normed/normed")
MTP_MODES = ("raw/raw", "raw/normed", "normed/raw", "normed/normed",
             "raw/raw:full", "raw/normed:full", "normed/raw:full", "normed/normed:full",   # ":full": full-vocab drafts
             "dflash", "auto")                    # DFlash2 drafter (dflash.py); auto: MTP or DFlash2 each round
FAST_ROWS = 16           # windows up to this many rows reduce over RoCE (when available); prompt chunks use NCCL
DECODE_ROWS = 32         # widest decode window (Buffers(decode=True)): concurrent DFlash2 rounds, 4 streams x 8 rows
DRAFT_VOCAB = int(os.environ.get("TF_GLM53_DRAFT_VOCAB", "32768"))  # draft head: the lowest ids (BPE: most frequent)
SPECIALS = 128           # ... plus the vocabulary's last ids (GLM's special tokens)
TUNE = os.environ.get("TF_GLM53_TUNE", "1") != "0"


# ------------------------------------------------------------------------------------------------ kernels ---
@triton.jit
def _rope_pair(a, b, cos, sin):
    return a * cos - b * sin, b * cos + a * sin


@triton.jit
def _kv_write(KVA, kva_stride, NW, LC, POS, INV, eps, LW: tl.constexpr, RD: tl.constexpr, DCP: tl.constexpr = 1,
              RANK: tl.constexpr = 0, BASE=None, ROWS: tl.constexpr = False):
    """Row r at position POS + r: cache[p, :LW] = RMSNorm(kva[:LW]), cache[p, LW:] = RoPE(kva[LW:LW + RD]) (GLM's
    interleaved pairs written evens then odds). DCP > 1: only the owner of p (p % DCP) stores it, at slot p // DCP.
    ROWS (several streams in one window): row r at position POS[r], in its stream's cache rows from BASE[r]."""
    r = tl.program_id(0)
    if ROWS:
        pg = tl.load(POS + r).to(tl.int64)
        p = tl.load(BASE + r).to(tl.int64) + pg
    else:
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
            NOPE_P: tl.constexpr, RD: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr, RBK: tl.constexpr,
            ROWS: tl.constexpr = False):
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
                if ROWS:                                 # the row's own position (several streams)
                    ang = tl.load(POS + r).to(tl.float32) * inv
                else:
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
def _qrope(Q, INV, QR, POS, H: tl.constexpr, QD: tl.constexpr, NOPE: tl.constexpr, RD: tl.constexpr,
           ROWS: tl.constexpr = False):
    """Program (row, head): q_rot at position POS + r (the wide-window companion of the batched absorb)."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, RD // 2)
    a = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i).to(tl.float32)
    b = tl.load(Q + (r * H + h) * QD + NOPE + 2 * i + 1).to(tl.float32)
    if ROWS:
        ang = tl.load(POS + r).to(tl.float32) * tl.load(INV + i)
    else:
        ang = (tl.load(POS) + r).to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
    tl.store(QR + (r * H + h) * RD + i, ra.to(tl.bfloat16))
    tl.store(QR + (r * H + h) * RD + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _attn_chunks(QA, QR, LC, TOK, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, RD: tl.constexpr,
                 K: tl.constexpr, CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr,
                 DIRECT: tl.constexpr = False, BASE=None, ROWS: tl.constexpr = False):
    """Program (row, chunk): all H heads of row r over entries [c CHK, (c + 1) CHK) of its key list - keys 0..p
    themselves while p < K, else TOK[r] (ascending). Scores are latent . latent + rope . rope; values are latents."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    if ROWS:                                     # several streams: the row's position, its stream's cache rows
        p = tl.load(POS + r)
        LC = LC + tl.load(BASE + r).to(tl.int64) * (LW + RD)
    else:
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
    if DIRECT:          # one chunk covers the row's list: the normalized output itself (= _merge of one chunk) -> PO
        tl.store(PO + (r * H + hh[:, None]) * LW + kl[None, :], (o / l[:, None]).to(tl.bfloat16))
    else:
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
              RANK: tl.constexpr = 0, BASE=None, ROWS: tl.constexpr = False):
    """Index key of row r: LayerNorm(ik) (fp32, with bias) to bf16, RoPE on its first RD dims, into IC[POS + r]
    (DCP > 1: the owner's slot p // DCP only)."""
    r = tl.program_id(0)
    if ROWS:
        pg = tl.load(POS + r).to(tl.int64)
        p = tl.load(BASE + r).to(tl.int64) + pg
    else:
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
def _iq_rope(Q, INV, POS, NH: tl.constexpr, D: tl.constexpr, RD: tl.constexpr, ROWS: tl.constexpr = False):
    """Program (row, head): the index query's first RD dims rotated at position POS + r, in place (bf16)."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    base = Q + (r * NH + h) * D
    i = tl.arange(0, RD // 2)
    a = tl.load(base + 2 * i).to(tl.float32)
    b = tl.load(base + 2 * i + 1).to(tl.float32)
    if ROWS:
        ang = tl.load(POS + r).to(tl.float32) * tl.load(INV + i)
    else:
        ang = (tl.load(POS) + r).to(tl.float32) * tl.load(INV + i)
    ra, rb = _rope_pair(a, b, tl.cos(ang), tl.sin(ang))
    tl.debug_barrier()
    tl.store(base + i, ra.to(tl.bfloat16))
    tl.store(base + RD // 2 + i, rb.to(tl.bfloat16))


@triton.jit
def _index_scores(Q, W, IC, POS, OUT, T, R0, NH: tl.constexpr, D: tl.constexpr, BTT: tl.constexpr,
                  WSCALE: tl.constexpr, QSCALE: tl.constexpr, DCP: tl.constexpr = 1, RANK: tl.constexpr = 0,
                  PACK: tl.constexpr = True, BASE=None, ROWS: tl.constexpr = False):
    """Program (row, key block): score[t] = sum_h w[h] relu(q[h] . k[t] QSCALE) for keys t <= POS + r, packed with
    the key into one int64 that orders by score, then lower position first (so top-k is tie-free); later keys -> min."""
    r = tl.program_id(0)                     # row r of this block: window row R0 + r
    t0 = tl.program_id(1) * BTT
    if ROWS:                                 # several streams: the row's position, its stream's index-key rows
        p = tl.load(POS + R0 + r)
        IC = IC + tl.load(BASE + R0 + r).to(tl.int64) * D
    else:
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
        key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)                           # monotone int of the fp32
        if PACK:
            packed = (key.to(tl.int64) << 32) | lo
            packed = tl.where(ok, packed, -9223372036854775807)
        else:                       # PACK=False: the score's unsigned-order word only (int32 storage), later keys 0
            packed = tl.where(ok, key ^ -2147483648, 0)
    else:
        packed = tl.full((BTT,), -9223372036854775807 if PACK else 0, tl.int64 if PACK else tl.int32)
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
        every = layers + ([mtp.layer] if mtp is not None else [])
        self.tunable = [lin for L in every for lin in linears(L)]   # the same order on every rank
        if TUNE:
            self.tuned = tune_linears(self.tunable)
            self.tuned_groups = tune_groups([g for L in every for g in groups(L)])

    def set_draft_head(self, rows: torch.Tensor, ids: torch.Tensor) -> None:
        """This rank's share of the reduced draft vocabulary: lm_head rows (bf16) and their global ids."""
        self.draft_head, self.draft_ids = rows.contiguous(), ids.to(torch.long).contiguous()


def linears(L: Layer) -> list:
    out = [L.q_a, L.kv_a, L.q_b, L.o_proj, *L.shared.values()]
    if L.indexer is not None:
        out.append(L.indexer["wq_b"])
    return out


def groups(L: Layer) -> list:
    """The layer's EXL3 linears of one input that ``lins`` runs as one launch."""
    out = [[L.q_a, L.kv_a], [L.shared["gate"], L.shared["up"]]]
    if L.indexer is not None:
        out.append([L.indexer["wq_b"], L.q_b])
    return out


def _graph_us(fn, iters: int = 20) -> float:
    """Microseconds a call of fn() replayed from a CUDA graph (the decode rounds run as graphs: no launch cost)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        # thread_local: ranks loading as threads of one process (tests) may allocate while this one captures
        with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = None
    for _ in range(2):                               # the faster of two timings: a stray stall counts once
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        t = e0.elapsed_time(e1) * 1e3 / iters
        best = t if best is None else min(best, t)
    del g
    return best


def share_tiles(w: "Weights", comm) -> int:
    """Every rank takes rank 0's tiles (each linear's split, which also decides its group's one launch): TP ranks run
    the same shapes, and tiles picked per rank from noisy timings let the slowest pick pace every layer (measured:
    four ranks, four different picks for the same shapes, +7-12 ms a decode round). Returns how many differed here."""
    mine = torch.tensor([v for lin in w.tunable for v in (lin.split or x3linear.plan(lin.k, lin.n))], dtype=torch.int32,
                        device="cuda")
    every = torch.empty((comm.world, mine.numel()), dtype=torch.int32, device="cuda")
    comm.all_gather(mine, every)
    ref = every[0].view(-1, 2).tolist()
    differ = 0
    for lin, t in zip(w.tunable, ref):
        if tuple(lin.split or ()) != tuple(t):
            differ += 1
            lin.split = tuple(t)
    return differ


def tune_groups(gs: list, rows: int = 3) -> dict:
    """Give each group of one-input linears (``groups``) one warps-a-block so ``lins`` runs it as one launch, when
    that times faster than the layers one by one with their own tiles (``tune_linears`` first). Each layer's K splits
    are then its fastest alone at that warps-a-block. Like any tiling, a function of the shapes alone: rows stay
    independent. Timed as graphs over up to 8 groups of a kind, so the words come from DRAM."""
    kinds: dict[tuple, list] = {}
    for g in gs:
        kinds.setdefault(tuple((lin.k, lin.n, lin.k2, lin.codebook, lin.layout) for lin in g), []).append(g)
    chosen = {}
    for key, members in kinds.items():
        pool = members[:8]
        a = pool[0][0]
        x = torch.randn(rows, a.k, device=a.words.device, dtype=torch.bfloat16) * 0.1
        outs = [[torch.empty((rows, lin.n), dtype=torch.bfloat16, device=x.device) for lin in g] for g in pool]
        own = [lin.split for lin in pool[0]]

        def apply(tiles):
            for g in members:
                for lin, t in zip(g, tiles):
                    lin.split = t

        def seq():
            for g, o in zip(pool, outs):
                for lin, y in zip(g, o):
                    lin(x, out=y)

        def grp():
            for g, o in zip(pool, outs):
                x3linear.group(g, x, o)

        try:
            best, best_t = None, _graph_us(seq)
            kt = a.k // 16
            for wk in (2, 4, 8):
                tiles = []
                for i in range(len(pool[0])):            # each layer's fastest K splits at this warps-a-block
                    fastest, fastest_t = None, None
                    for sk in (1, 2, 4, 8, 16, 32, 64):
                        if kt % (sk * wk) or kt // (sk * wk) < 2:
                            continue
                        for g in pool:
                            g[i].split = (sk, wk)

                        def one(i=i):
                            for g, o in zip(pool, outs):
                                g[i](x, out=o[i])

                        t = _graph_us(one)
                        if fastest_t is None or t < fastest_t:
                            fastest, fastest_t = (sk, wk), t
                    tiles.append(fastest)
                if None in tiles:
                    continue
                apply(tiles)
                t = _graph_us(grp)
                if t < best_t:
                    best, best_t = tiles, t
        except Exception:                            # noqa: BLE001  a tiling the kernel refuses: keep the layers' own
            best = None
        apply(best if best is not None else own)
        chosen[tuple(k[:2] for k in key)] = (own, best)
    return chosen


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
    for the MTP layer; positions live on the device (``pos``: the target window's first row, ``mpos``: the MTP's).
    ``slots`` > 1 (concurrent streams): each cache holds that many streams' rows back to back (``local`` rows each);
    ``view(s)`` is stream s alone (the one-stream kernels on its rows), ``Rows`` a window over several."""

    def __init__(self, w: Weights, capacity: int, slots: int = 1) -> None:
        c, dev = w.cfg, w.device
        self.capacity, self.slots = capacity, slots
        lw = c.kv_lora_rank + c.qk_rope_head_dim
        n = len(w.layers)
        local = -(-capacity // w.dcp) + 1               # DCP: this rank's positions p % dcp == rank at p // dcp
        if slots > 1 and w.dcp > 1:
            raise ValueError("concurrent streams need decode context parallelism off (DCP 1)")
        self.local = local
        rows = local * slots
        self.kc = [torch.zeros((rows, lw), dtype=torch.bfloat16, device=dev) for _ in range(n)]
        self.ic = {L.index: torch.zeros((rows, c.index_head_dim), dtype=torch.bfloat16, device=dev)
                   for L in w.layers if L.indexer is not None}
        self.pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mpos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mkc = self.mic = None
        if w.mtp is not None:
            self.mkc = torch.zeros((rows, lw), dtype=torch.bfloat16, device=dev)
            self.mic = torch.zeros((rows, c.index_head_dim), dtype=torch.bfloat16, device=dev)

    def nbytes(self) -> int:
        ts = self.kc + list(self.ic.values()) + [t for t in (self.mkc, self.mic) if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)

    def view(self, s: int) -> "SlotView":
        return SlotView(self, s)


class SlotView:
    """Stream s of a State with slots: its cache rows as views (the one-stream kernels address them from 0), its own
    device positions."""

    def __init__(self, st: State, s: int) -> None:
        a, e = s * st.local, (s + 1) * st.local
        self.capacity, self.local = st.capacity, st.local
        self.kc = [t[a:e] for t in st.kc]
        self.ic = {i: t[a:e] for i, t in st.ic.items()}
        self.mkc = None if st.mkc is None else st.mkc[a:e]
        self.mic = None if st.mic is None else st.mic[a:e]
        self.pos = torch.zeros((1,), dtype=torch.int32, device=st.pos.device)
        self.mpos = torch.zeros((1,), dtype=torch.int32, device=st.pos.device)


class Rows:
    """A window over several streams of a State with slots: per-row positions and cache bases (int32 device tables
    the caller fills before each call or graph replay); the target's (pos, base) and the MTP layer's (mpos, mbase)."""

    def __init__(self, st: State, pos: torch.Tensor, base: torch.Tensor, mpos: torch.Tensor,
                 mbase: torch.Tensor) -> None:
        self.kc, self.ic, self.mkc, self.mic = st.kc, st.ic, st.mkc, st.mic
        self.pos, self.base, self.mpos, self.mbase = pos, base, mpos, mbase


class Buffers:
    """Scratch for windows of up to ``rows`` rows (sliced [:R]); ``score_cols``: the indexer's widest key range.
    ``decode``: every window these buffers serve is a decode / verify window, whatever its width (up to DECODE_ROWS):
    the RoCE one-shot (or all-gather + rank-order sum) reductions, the decode attention tiling and the head buffers
    of a lone request's windows, so a row keeps its bits however many rows share the window. Otherwise (prompt
    chunks) only windows up to FAST_ROWS rows are decode windows."""

    def __init__(self, w: Weights, rows: int, score_cols: int, decode: bool = False) -> None:
        c, dev = w.cfg, w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, H = c.hidden_size, w.heads
        lw, rd, qd = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim + c.qk_rope_head_dim
        self.rows, self.score_cols = rows, score_cols
        if decode and rows > DECODE_ROWS:
            raise ValueError(f"decode windows of {rows} rows: at most {DECODE_ROWS}")
        self.small = max(rows, FAST_ROWS) if decode else FAST_ROWS     # windows up to this many rows: decode class
        self.ws = x3prefill.Workspace(_UNPACK_CACHE)
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
        slots = max(c.index_topk // ATTN_DECODE[0] * min(rows, self.small), c.index_topk // ATTN_PROMPT[0] * rows)
        slots *= w.dcp                                   # DCP: partials for every rank's heads
        if w.dcp > 1:
            G = w.dcp
            self.qpack = torch.empty((rows, H, lw + rd), dtype=bf, device=dev)
            self.qall = torch.empty((G, rows, H, lw + rd), dtype=bf, device=dev)
            self.osend = torch.empty((G * rows * H * lw,), dtype=bf, device=dev)
            self.lsend = torch.empty((G * rows * H,), dtype=f32, device=dev)
            fan = G if rows <= self.small else 1          # decode windows exchange by all-gather (G x the bytes)
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
        self.hpart = torch.empty((rows, D), dtype=bf, device=dev) if rows > self.small else None   # prompt halves
        self.hred = torch.empty((rows, D), dtype=bf, device=dev) if rows > self.small else None
        self.xo = None                                   # sequence-parallel prompt chunks: own rows (on first use)
        self.spos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.amax = torch.zeros((min(rows, self.small), 4), dtype=f32, device=dev)
        self.amax_all = torch.zeros((w.world * min(rows, self.small) * 4,), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        # heads
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.taps = (torch.zeros((rows, len(w.tap_slot) * D), dtype=bf, device=dev) if w.tap_slot else None)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        V = w.lm_head.shape[0]
        hr = min(rows, self.small)           # head rows: a window's, or a prompt chunk's last one
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


def lins(layers: list, b: "Buffers", x: torch.Tensor, outs: list) -> None:
    """EXL3 linears of one input (q_a and kv_a, gate and up, wq_b and q_b): one launch when the tiles allow
    (``x3linear.group``; each output the bits of its own call), else one ``lin`` each."""
    if x.shape[0] <= MAX_ROWS and x3linear.groupable(layers):
        x3linear.group(layers, x, outs)
    else:
        for layer, out in zip(layers, outs):
            lin(layer, b, x, out)


def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """The window's fp32 partials of every rank, summed in rank order: [1, R, D] from the RoCE one-shot reduce
    (decode windows), else every rank's partial [world, R, D] for the consumer to add in rank order."""
    d = b.part.shape[1]
    if w.world == 1:
        return b.part[:R].view(1, R, d)
    if w.fast is not None and R <= b.small:
        w.fast.all_reduce(b.part[:R], b.red[:R])
        return b.red[:R].view(1, R, d)
    if R > b.small and PREFILL_REDUCE == "ring" and hasattr(w.comm, "all_reduce"):
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


def select(w: Weights, b: Buffers, icache: torch.Tensor, pos: torch.Tensor, R: int, T: int, row0: int = 0,
           base: torch.Tensor | None = None) -> None:
    """The indexer's choice for the window's rows past index_topk: b.tok[:R] (ascending). ``T``: keys scored (the
    host's bound on the window's last position + 1; a captured graph uses its bucket). DCP: each rank scores its own
    slots, keeps its top index_topk, the candidates are gathered and every rank takes the same global top index_topk
    (keys are tie-free); b.tok then holds this rank's share as local slots (ascending), b.cnt how many. ``row0``:
    select window rows row0 .. row0 + R - 1 only (sequence-parallel prompt chunks: a rank's own rows).
    ``base``: several streams in the window (pos[r], base[r] per row - window rows, so row0 indexes them too; one
    rank's keys, DCP 1 only)."""
    c = w.cfg
    nh, D, K = c.index_n_heads, c.index_head_dim, c.index_topk
    dcp, rank = w.dcp, w.rank if w.dcp > 1 else 0
    Tl = -(-T // dcp)
    for r0 in range(row0, row0 + R, SEL_ROWS):
        n = min(SEL_ROWS, row0 + R - r0)
        # radix: one program a row - beats torch.topk on prompt blocks, loses on decode windows at any context past K
        # (1-16 rows x 128K keys: 0.64-0.77 vs 0.17-0.24 ms on GB10; 32K: +3-4 ms a round over 78 layers); same keys
        radix = RADIX and n >= RADIX_MIN_ROWS and (dcp > 1 or Tl >= K)
        # radix, one rank: 4-byte order words (ties to the lower position in the select itself), else packed keys
        sc = b.sc.view(torch.int32)[:n * Tl].view(n, Tl) if radix and dcp == 1 else b.sc[:n * Tl].view(n, Tl)
        _index_scores[(n, triton.cdiv(Tl, BT))](b.iq[r0:], b.iw[r0:], icache, pos, sc, Tl, R0=r0, NH=nh, D=D,
                                                BTT=BT, WSCALE=nh ** -0.5, QSCALE=D ** -0.5, DCP=dcp, RANK=rank,
                                                PACK=not (radix and dcp == 1), BASE=base, ROWS=base is not None,
                                                num_warps=2, num_stages=2)
        if dcp == 1:
            if radix:
                topk.top_columns(sc, K, b.tok[r0:r0 + n])
                continue
            top = torch.topk(sc, K, dim=-1, sorted=False).values
            keys = (0x7FFFFFFF - (top & 0xFFFFFFFF)).to(torch.int32)
            b.tok[r0:r0 + n].copy_(torch.sort(keys, dim=-1).values)
            continue
        kk = min(K, Tl)
        mine = torch.full((n, K), -9223372036854775807, dtype=torch.int64, device=sc.device)
        mine[:, :kk] = topk.top_keys(sc, kk) if radix else torch.topk(sc, kk, dim=-1, sorted=False).values
        allc = b.cand[:dcp * n * K].view(dcp, n, K)
        dcp_gather(w, mine, allc, n <= b.small)
        cand = allc.permute(1, 0, 2).reshape(n, dcp * K)
        top = topk.top_keys(cand, K) if radix else torch.topk(cand, K, dim=-1, sorted=False).values
        gpos = 0x7FFFFFFF - (top & 0xFFFFFFFF)                                  # global positions, int64
        own = (gpos % dcp) == rank
        key = torch.where(own, gpos // dcp, torch.full_like(gpos, 1 << 40))
        b.tok[r0:r0 + n].copy_(torch.sort(key, dim=-1).values.to(torch.int32))
        b.cnt[r0:r0 + n].copy_(own.sum(-1).to(torch.int32))


def _attention_local(w: Weights, b: Buffers, R: int, cache, pos, nch, chk, kt, nw, ns, base=None) -> None:
    """Every key on this rank: this rank's heads over the rows' key lists -> b.ol."""
    c, H = w.cfg, w.heads
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    n = nch * R * H
    if nch == 1:                    # one pass: no partials, no merge (the same bits as one chunk + _merge)
        _attn_chunks[(R, 1)](b.qlat, b.qrot, cache, b.tok, pos, b.ol, b.pm, b.pl, R, H=H, LW=lw, RD=rd,
                             K=c.index_topk, CHK=chk, KTT=kt, SCALE=(nope + rd) ** -0.5, DIRECT=True, BASE=base,
                             ROWS=base is not None, num_warps=nw, num_stages=ns)
        return
    _attn_chunks[(R, nch)](b.qlat, b.qrot, cache, b.tok, pos, b.po[:n * lw], b.pm[:n], b.pl[:n], R, H=H, LW=lw,
                           RD=rd, K=c.index_topk, CHK=chk, KTT=kt, SCALE=(nope + rd) ** -0.5, BASE=base,
                           ROWS=base is not None, num_warps=nw, num_stages=ns)
    latent._merge[(R, H)](b.po, b.pm, b.pl, b.ol, b.dummy, R, H=H, LW=lw, NCH=nch, SPARSE=False, num_warps=4)


def _attention_dcp(w: Weights, b: Buffers, R: int, cache, pos, nch, chk, kt, nw, ns) -> None:
    """Decode context parallelism: gather every rank's absorbed queries, attend all heads over this rank's keys,
    send each head's normalized partial and log-sum-exp to the rank that owns the head, merge in rank order -> b.ol."""
    c, H, G, rank = w.cfg, w.heads, w.dcp, w.rank
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    small = R <= b.small
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
              pos: torch.Tensor, T: int | None, base: torch.Tensor | None = None) -> torch.Tensor:
    attention_part(w, L, b, R, cache, icache, pos, T, base)
    return gather(w, b, R)


def attention_part(w: Weights, L: Layer, b: Buffers, R: int, cache: torch.Tensor, icache: torch.Tensor | None,
                   pos: torch.Tensor, T: int | None, base: torch.Tensor | None = None) -> None:
    """Attention of the window's rows (b.normed): writes this layer's latent (and index key), leaves this rank's
    fp32 partial in b.part[:R]. ``T``: None while every row is below index_topk (no selection).
    ``base`` (several streams in one window): row r sits at position pos[r] of the stream whose cache rows start at
    base[r] (int32 tables); every row keeps the bits it has alone (the per-row kernels compute the same values)."""
    c = w.cfg
    lw, rd = c.kv_lora_rank, c.qk_rope_head_dim
    lins([L.q_a, L.kv_a], b, b.normed[:R], [b.qa[:R], b.kva[:R]])
    glue.rmsnorm(b.qa[:R], L.q_a_norm, c.rms_norm_eps, b.qn[:R])
    dcp, rank = w.dcp, (w.rank if w.dcp > 1 else 0)
    rows = base is not None
    if rows and dcp > 1:
        raise ValueError("several streams in one window need decode context parallelism off (DCP 1)")
    _kv_write[(R,)](b.kva, b.kva.stride(0), L.kv_a_norm, cache, pos, w.inv, c.rms_norm_eps, LW=lw, RD=rd, DCP=dcp,
                    RANK=rank, BASE=base, ROWS=rows, num_warps=4)
    q_done = False                                       # q_b run with wq_b
    if L.indexer is not None:
        ix = L.indexer
        glue.router(b.normed[:R], ix["wk"], b.ik[:R])
        _ik_write[(R,)](b.ik, L.extra["ik_w"], L.extra["ik_b"], w.inv, icache, pos, 1e-6, D=c.index_head_dim, RD=rd,
                        DCP=dcp, RANK=rank, BASE=base, ROWS=rows, num_warps=4)
        if T is not None:
            glue.router(b.normed[:R], ix["weights_proj"], b.iw[:R])
            lins([ix["wq_b"], L.q_b], b, b.qn[:R], [b.iq[:R], b.q[:R]])   # q_b early: select leaves b.q alone
            q_done = True
            _iq_rope[(R, c.index_n_heads)](b.iq, w.inv, pos, NH=c.index_n_heads, D=c.index_head_dim, RD=rd,
                                           ROWS=rows, num_warps=1)
            select(w, b, icache, pos, R, T, base=base)
    attention_core(w, L, b, R, cache, pos, q_done, base)


def attention_core(w: Weights, L: Layer, b: Buffers, R: int, cache: torch.Tensor, pos: torch.Tensor,
                   q_done: bool = False, base: torch.Tensor | None = None) -> None:
    """The head-sharded rest of attention for rows b.qn[:R] (selection in b.tok): q_b (unless attention_part already
    ran it grouped with wq_b), absorb, attention over the cache, expand, o_proj -> this rank's fp32 partial b.part[:R].
    ``base``: per-row positions pos[r] and cache bases (several streams, see attention_part)."""
    c = w.cfg
    H = w.heads
    lw, rd, nope = c.kv_lora_rank, c.qk_rope_head_dim, c.qk_nope_head_dim
    dcp = w.dcp
    rows = base is not None
    if not q_done:
        lin(L.q_b, b, b.qn[:R], b.q[:R])
    rbk = RB if R > RB else R
    wide = R > MAX_ROWS                                  # prompt chunks: tensor-core batched GEMMs (not row-exact)
    if wide:
        qn = b.q[:R].view(R, H, nope + rd)[:, :, :nope].transpose(0, 1)                    # [H, R, nope]
        b.qlat[:R].transpose(0, 1).copy_(torch.bmm(qn, L.extra["wk"]))                     # [H, R, lw]
        _qrope[(R, H)](b.q, w.inv, b.qrot, pos, H=H, QD=nope + rd, NOPE=nope, RD=rd, ROWS=rows, num_warps=1)
    else:
        _absorb[(H, lw // 32, triton.cdiv(R, rbk))](b.q, L.extra["wk"], w.inv, b.qlat, b.qrot, pos, R, H=H,
                                                    QD=nope + rd, NOPE=nope, NOPE_P=triton.next_power_of_2(nope),
                                                    RD=rd, LW=lw, BN=32, RBK=rbk, ROWS=rows, num_warps=4)
    chk, kt, nw, ns = ATTN_DECODE if R <= b.small else ATTN_PROMPT
    nch = max(1, c.index_topk // chk)
    if dcp > 1:
        _attention_dcp(w, b, R, cache, pos, nch, chk, kt, nw, ns)
    else:
        _attention_local(w, b, R, cache, pos, nch, chk, kt, nw, ns, base)
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
    lins([s["gate"], s["up"]], b, b.normed[:R], [g, u])
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
    route(w, L, b, 0, R)
    experts_part(w, L, b, R)


def route(w: Weights, L: Layer, b: Buffers, r0: int, n: int) -> None:
    """Router logits and top-k of rows b.normed[r0:r0 + n] -> b.pick / b.wts rows r0 .."""
    c = w.cfg
    glue.router(b.normed[r0:r0 + n], L.router[0], b.mlog[r0:r0 + n])
    K = c.num_experts_per_tok
    glue._topk[(n,)](b.mlog[r0:], L.extra["bias"], b.pick[r0:], b.wts[r0:], float(c.routed_scaling_factor),
                     NE=c.n_routed_experts, TOPK=K, SLOTS=K, BLOCK=triton.next_power_of_2(c.n_routed_experts + 1),
                     SLOTP=triton.next_power_of_2(K + 1), NORM=c.norm_topk_prob, num_warps=4)


def experts_part(w: Weights, L: Layer, b: Buffers, R: int) -> None:
    """Routed experts (picks in b.pick) + the shared expert of rows b.normed[:R] -> fp32 partial b.part[:R]."""
    if not hasattr(L.experts, "prefill"):
        x3experts.routed(b.normed[:R], b.pick[:R], b.wts[:R], L.experts, b.xs, b.part[:R], R)
    elif R <= b.xs.rows:
        L.experts.decode(b.normed[:R], b.pick[:R], b.wts[:R], b.xs, b.part[:R], R)
    else:                                                # prompt chunk: cuda-exl3's grouped GEMM
        b.part[:R].copy_(L.experts.prefill(b.normed[:R], b.pick[:R], b.wts[:R]))
    mlp(w, L, b, R, b.sy[:R])
    b.part[:R].add_(b.sy[:R])


def layer(w: Weights, L: Layer, b: Buffers, x: torch.Tensor, R: int, cache, icache, pos, T, base=None) -> None:
    c = w.cfg
    glue.rmsnorm(x, L.input_norm, c.rms_norm_eps, b.normed[:R])
    glue.residual_add(x, x, attention(w, L, b, R, cache, icache, pos, T, base))
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
            pick: str = "full", layers: tuple[int, int] | None = None) -> None:
    """The target's GPU work for rows b.ids[:R] at st.pos .. st.pos + R - 1 (capturable): caches written, final
    hidden in b.hidden[:R]; logits of all rows, the last row ("last") or none. A ``Rows`` state (several streams):
    row r at st.pos[r] in the cache rows from st.base[r]. ``layers`` (lo, hi): only those layers (a prompt chunk
    paused between layers: the rows' activations wait in b.x; lo == 0 embeds, hi == every layer finishes)."""
    lo, hi = layers or (0, len(w.layers))
    x = b.x[:R]
    if lo == 0:
        torch.index_select(w.embed, 0, b.ids[:R], out=x)
    D = x.shape[1]
    base = getattr(st, "base", None)
    for i in range(lo, hi):
        L = w.layers[i]
        layer(w, L, b, x, R, st.kc[i], st.ic.get(L.index), st.pos, T, base)
        s = w.tap_slot.get(i)
        if s is not None:                                # DFlash2: this layer's output rows
            b.taps[:R, s * D:(s + 1) * D].copy_(x)
    if hi < len(w.layers):
        return
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
                   comm_stream, *, logits: str = "none", layers: tuple[int, int] | None = None,
                   halves: list | None = None) -> list:
    """A prompt chunk as two micro-batches whose all-reduces overlap each other's compute (rows [0, h) in b0 at
    st.pos, rows [h, R) in b1 at pos1 = st.pos + h). Ids in b0.ids[:R]; final hidden rows in b0.hidden[:R].
    ``layers`` (lo, hi): only those layers; the halves (returned, passed back for the next range) hold the paused
    chunk's state: rows in b0.x / b1.x, the last reductions in flight - the current stream waits for those before
    returning, so work issued in the pause (decode rounds, their collectives) starts after them."""
    c = w.cfg
    hR = R // 2
    lo, hi = layers or (0, len(w.layers))
    if lo == 0:
        halves = [_Half(b0, hR, st.pos), _Half(b1, R - hR, pos1)]
        b1.ids[:R - hR].copy_(b0.ids[hR:R])
        for h in halves:
            torch.index_select(w.embed, 0, h.b.ids[:h.R], out=h.b.x[:h.R])
    D = c.hidden_size

    def tap(h: _Half, i: int) -> None:                    # layer i's output rows (complete after its residual)
        s = w.tap_slot.get(i)
        if s is not None:
            h.b.taps[:h.R, s * D:(s + 1) * D].copy_(h.b.x[:h.R])

    for i in range(lo, hi):
        L = w.layers[i]
        for h in halves:                                  # attention: A writes its keys before B attends
            if i:
                _residual(w, h)
                tap(h, i - 1)
            x = h.b.x[:h.R]
            glue.rmsnorm(x, L.input_norm, c.rms_norm_eps, h.b.normed[:h.R])
            attention_part(w, L, h.b, h.R, st.kc[i], st.ic.get(L.index), h.pos, T)
            _reduce_async(w, h, comm_stream)
        for h in halves:
            _residual(w, h)
            x = h.b.x[:h.R]
            glue.rmsnorm(x, L.post_attn_norm, c.rms_norm_eps, h.b.normed[:h.R])
            ffn_part(w, L, h.b, h.R)
            _reduce_async(w, h, comm_stream)
    if hi < len(w.layers):                                # paused: drain the reductions in flight
        for h in halves:
            torch.cuda.current_stream().wait_event(h.done)
        return halves
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
    return halves


# ------------------------------------------------------------------------------- sequence-parallel prompt ---
class _SpHalf(_Half):
    """A micro-batch under sequence parallelism: rows split into ``world`` contiguous blocks of n (padded to Rp =
    world n; padding rows are kept at zero); this rank owns rows [o, o + n), its residual stream rows in b.xo[:n].
    ``done``: the end of its latest side chain (comm stream)."""

    def __init__(self, w: Weights, b: Buffers, R: int, pos: torch.Tensor) -> None:
        super().__init__(b, R, pos)
        self.n = -(-R // w.world)
        self.Rp = self.n * w.world
        self.o = w.rank * self.n
        self.nv = max(0, min(self.n, R - self.o))        # own rows that are real (the rest: padding)
        if b.xo is None or b.xo.shape[0] < self.n:
            b.xo = torch.empty((-(-b.rows // w.world), b.x.shape[1]), dtype=torch.bfloat16, device=b.x.device)
        b.spos.copy_(pos)
        b.spos.add_(self.o)                              # device position of the first own row


def _side(h: _SpHalf, comm_stream, fn) -> None:
    """fn() on the comm stream after the main stream's work so far (and h's previous chain); h.done marks its end."""
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ready)
        fn()
        h.done = torch.cuda.Event()
        h.done.record(comm_stream)


def _ag_rows(w: Weights, h: _SpHalf, *bufs: torch.Tensor) -> None:
    """Every rank's own rows of each buffer (in place: rank r's rows at [r n, (r + 1) n)) -> rows [0, Rp)."""
    pairs = [(t[:h.Rp][h.o:h.o + h.n].reshape(-1), t[:h.Rp].reshape(-1)) for t in bufs]
    if hasattr(w.comm, "all_gather_group"):
        w.comm.all_gather_group(pairs)
    else:
        for send, recv in pairs:
            w.comm.all_gather(send, recv)


def _rs_residual(w: Weights, h: _SpHalf) -> None:
    """(comm stream) b.part[:R] (fp32 partial, every row) -> own rows of the sum, added to the residual b.xo:
    exact (all-to-all of fp32 row blocks, summed in rank order by the residual add) or NCCL's ring reduce-scatter
    of bf16 (PREFILL_REDUCE)."""
    b, R, Rp, n = h.b, h.R, h.Rp, h.n
    D = b.part.shape[1]
    if Rp > R:
        b.part[R:Rp].zero_()
    if PREFILL_REDUCE == "ring" and hasattr(w.comm, "reduce_scatter"):
        b.hpart[:Rp].copy_(b.part[:Rp])
        w.comm.reduce_scatter(b.hpart[:Rp].view(-1), b.hred[:n].view(-1))
        red = b.hred[:n].view(1, n, D)
    else:
        recv = b.gath[:Rp * D].view(w.world, n * D)
        w.comm.all_to_all(b.part[:Rp].view(w.world, n * D), recv)
        red = recv.view(w.world, n, D)
    x = b.xo[:n]
    glue.residual_add(x, x, red)


def _gather_x(w: Weights, h: _SpHalf, out: torch.Tensor) -> None:
    """(comm stream) every row's residual stream gathered from the ranks' own rows -> out [R, D]."""
    b = h.b
    D = b.x.shape[1]
    full = b.gath.view(torch.bfloat16)[:h.Rp * D]        # gath is fp32: room for 2 x the bf16 rows
    w.comm.all_gather(b.xo[:h.n].reshape(-1), full)
    out.copy_(full.view(h.Rp, D)[:h.R])


def _front_attn(w: Weights, L: Layer, h: _SpHalf, cache, icache, T: int | None) -> None:
    """(comm stream) own rows: input norm, q_a / kv_a / the indexer's projections; gather them; every row's latent
    and index key into the caches; the selection (own rows, gathered: SP_SELECT; else every row)."""
    c = w.cfg
    b, o, n, Rr = h.b, h.o, h.n, h.R
    rd, lw = c.qk_rope_head_dim, c.kv_lora_rank
    ix = L.indexer
    own_sel = SP_SELECT and ix is not None and T is not None
    nx = b.normed[o:o + n]
    glue.rmsnorm(b.xo[:n], L.input_norm, c.rms_norm_eps, nx)
    lin(L.q_a, b, nx, b.qa[o:o + n])
    lin(L.kv_a, b, nx, b.kva[o:o + n])
    glue.rmsnorm(b.qa[o:o + n], L.q_a_norm, c.rms_norm_eps, b.qn[o:o + n])
    bufs = [b.qn, b.kva]
    if ix is not None:
        glue.router(nx, ix["wk"], b.ik[o:o + n])
        bufs.append(b.ik)
        if T is not None:
            glue.router(nx, ix["weights_proj"], b.iw[o:o + n])
            if own_sel:
                lin(ix["wq_b"], b, b.qn[o:o + n], b.iq[o:o + n])
                _iq_rope[(n, c.index_n_heads)](b.iq[o:], w.inv, b.spos, NH=c.index_n_heads, D=c.index_head_dim,
                                               RD=rd, num_warps=1)
            else:
                bufs.append(b.iw)
    _ag_rows(w, h, *bufs)
    _kv_write[(Rr,)](b.kva, b.kva.stride(0), L.kv_a_norm, cache, h.pos, w.inv, c.rms_norm_eps, LW=lw, RD=rd,
                     num_warps=4)
    if ix is None:
        return
    _ik_write[(Rr,)](b.ik, L.extra["ik_w"], L.extra["ik_b"], w.inv, icache, h.pos, 1e-6, D=c.index_head_dim, RD=rd,
                     num_warps=4)
    if T is None:
        return
    if own_sel:                                          # every row's index keys are in: own rows choose
        select(w, b, icache, h.pos, n, T, row0=o)
        _ag_rows(w, h, b.tok)
    else:
        lin(ix["wq_b"], b, b.qn[:Rr], b.iq[:Rr])
        _iq_rope[(Rr, c.index_n_heads)](b.iq, w.inv, h.pos, NH=c.index_n_heads, D=c.index_head_dim, RD=rd,
                                        num_warps=1)
        select(w, b, icache, h.pos, Rr, T)


def _front_ffn(w: Weights, L: Layer, h: _SpHalf) -> None:
    """(comm stream) own rows: post-attention norm and routing; gather the normed rows (and picks)."""
    c = w.cfg
    b, o, n = h.b, h.o, h.n
    glue.rmsnorm(b.xo[:n], L.post_attn_norm, c.rms_norm_eps, b.normed[o:o + n])
    bufs = [b.normed]
    if L.experts is not None:
        route(w, L, b, o, n)
        bufs += [b.pick, b.wts]
    _ag_rows(w, h, *bufs)


def compute_prompt_sp(w: Weights, st: State, b0: Buffers, b1: Buffers, R: int, T: int | None, pos1: torch.Tensor,
                      comm_stream, *, logits: str = "none", layers: tuple[int, int] | None = None,
                      halves: list | None = None) -> list:
    """compute_prompt, sequence parallel. Each all-reduce becomes a reduce-scatter over rows; the residual add, the
    RMSNorms and the replicated projections (q_a, kv_a, the indexer's wk / weights_proj / wq_b and, with SP_SELECT,
    its top-k; the router) run on a rank's own 1/world of the rows; all-gathers hand the head- and width-sharded
    consumers what they read for every row (q_a's normed output, the kv latent and index keys - the caches take
    every row - the selections, the post-attention normed x and the expert picks).

    The main stream runs only the every-row work (attention over this rank's heads, the MLP / experts over its
    share of the width); a half's reduce-scatter -> own-row glue -> all-gather chain runs on the comm stream under
    the other half's every-row work. Chains run in issue order, so half A's keys reach the caches before B selects.

    ``layers`` (lo, hi): only those layers (as compute_prompt): the halves (returned, passed back for the next range)
    hold the paused chunk - own rows in b.xo, the next layer's front (its gathered q / kv / selection) already
    issued by layer hi - 1's chain; the current stream waits for every chain before returning, so work issued in the
    pause (decode rounds, their collectives) starts after them. The same kernels in the same order as one call."""
    c = w.cfg
    D = c.hidden_size
    hR = R // 2
    n_layers = len(w.layers)
    lo, hi = layers or (0, n_layers)
    if lo == 0:
        b1.ids[:R - hR].copy_(b0.ids[hR:R])
        halves = [_SpHalf(w, b0, hR, st.pos), _SpHalf(w, b1, R - hR, pos1)]

    def tap(h: _SpHalf, i: int) -> None:
        s = w.tap_slot.get(i)
        if s is not None:
            _gather_x(w, h, h.b.taps[:h.R, s * D:(s + 1) * D])

    def start(h: _SpHalf) -> None:
        b, o, nv = h.b, h.o, h.nv
        if nv:
            torch.index_select(w.embed, 0, b.ids[o:o + nv], out=b.xo[:nv])
        b.xo[nv:h.n].zero_()
        _front_attn(w, w.layers[0], h, st.kc[0], st.ic.get(w.layers[0].index), T)

    def after_attn(h: _SpHalf, L: Layer) -> None:
        _rs_residual(w, h)
        _front_ffn(w, L, h)

    def after_ffn(h: _SpHalf, i: int) -> None:
        _rs_residual(w, h)
        tap(h, i)
        if i + 1 < n_layers:
            L = w.layers[i + 1]
            _front_attn(w, L, h, st.kc[i + 1], st.ic.get(L.index), T)
        else:
            _gather_x(w, h, h.b.x[:h.R])

    if lo == 0:
        for h in halves:
            _side(h, comm_stream, lambda h=h: start(h))
    main = torch.cuda.current_stream()
    for i in range(lo, hi):
        L = w.layers[i]
        for h in halves:
            main.wait_event(h.done)
            attention_core(w, L, h.b, h.R, st.kc[i], h.pos)
            _side(h, comm_stream, lambda h=h, L=L: after_attn(h, L))
        for h in halves:
            main.wait_event(h.done)
            if L.experts is None:
                mlp(w, L, h.b, h.R, h.b.part[:h.R])
            else:
                experts_part(w, L, h.b, h.R)
            _side(h, comm_stream, lambda h=h, i=i: after_ffn(h, i))
    for h in halves:
        main.wait_event(h.done)
    if hi < n_layers:                                    # paused: every chain drained (above)
        return halves
    if b0.taps is not None:
        b0.taps[hR:R].copy_(b1.taps[:R - hR])
    b0.hidden[:hR].copy_(b0.x[:hR])
    b0.hidden[hR:R].copy_(b1.x[:R - hR])
    if logits == "last":
        head(w, b1, b1.x[:R - hR], w.final_norm, R - hR, slice(R - hR - 1, R - hR))
        b0.logits[:1].copy_(b1.logits[:1])
    return halves


def sp_fits(w: Weights, b0: Buffers, b1: Buffers, R: int) -> bool:
    """Whether compute_prompt_sp can take this chunk (padded halves fit the buffers; no DCP)."""
    if not PROMPT_SP or w.world == 1 or w.dcp != 1 or b0.hpart is None or b1.hpart is None:
        return False
    hR = R // 2
    return all(-(-r // w.world) * w.world <= b.rows for r, b in ((hR, b0), (R - hR, b1)))


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, T: int | None, *, logits: str = "last",
                zero_first: bool = False, chain_normed: bool = False, draft_full: bool = False,
                last: torch.Tensor | None = None) -> None:
    """The MTP layer for rows (b.hin[:n] = the previous position's hidden, b.ids[:n] = the token) at st.mpos ..;
    its output hidden in b.hidden[:n] (raw, or shared_head-normed per MTP_CHAIN) and the last row's logits/argmax.
    A ``Rows`` state (several streams): row r at st.mpos[r] in the MTP cache rows from st.mbase[r]; ``last`` (device
    row indices, one a stream): the rows whose picks go to b.argmax[:len(last)]."""
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
    layer(w, m.layer, b, x, n, st.mkc, st.mic, st.mpos, T, getattr(st, "mbase", None))
    if chain_normed:
        glue.rmsnorm(x, m.head_norm, c.rms_norm_eps, b.hidden[:n])
    else:
        b.hidden[:n].copy_(x)
    mode = "argmax" if draft_full or w.draft_head is None else "draft"
    if last is not None:                             # several streams: each stream's last row
        S = last.shape[0]
        sel = b.me[:S]                               # (the embeddings are spent by now)
        torch.index_select(x, 0, last, out=sel)
        head(w, b, sel, m.head_norm, S, mode=mode)
    elif logits == "last":
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
