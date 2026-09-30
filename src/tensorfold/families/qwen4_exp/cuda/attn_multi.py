"""A concurrent step's attention in one launch a kernel: rows find their caches by table and run the one-stream code."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import gdn as shared

from . import attention as attn_mod, glue
from .attention import CHUNK, _chunk, _merge_row, _pool_block

PTRS = 6                 # a stream's pointers a layer: keys, values, key scales, value scales, index keys, pooled


@triton.jit
def _ptr(TABLE, s, T: tl.constexpr):
    return tl.multiple_of(tl.load(TABLE + s).to(tl.pointer_type(T)), 16)


@triton.jit
def _prep_multi(P, POSR, SID, CP, QW, KW, IW, INV, Q, IQ, eps, N, PW: tl.constexpr, NQ: tl.constexpr,
                NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
                BITS: tl.constexpr, KT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    glue._prep_row(P, tl.load(POSR + r), r, tl.program_id(1), QW, KW, IW, INV, Q, _ptr(CP, s, KT),
                   _ptr(CP + N, s, KT), _ptr(CP + 2 * N, s, tl.float16), _ptr(CP + 3 * N, s, tl.float16), IQ,
                   _ptr(CP + 4 * N, s, tl.bfloat16), eps, PW, NQ, NKV, HD, NI, IHD, HALF, BITS)


@triton.jit
def _pool_multi(CP, P0, RS, W, INV, eps, N, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr):
    s = tl.program_id(0)
    _pool_block(_ptr(CP + 4 * N, s, tl.bfloat16), _ptr(CP + 5 * N, s, tl.bfloat16), tl.load(P0 + s),
                tl.program_id(1), W, INV, eps, tl.load(RS + s), DI, HALF, RATIO)


@triton.jit
def _chunks_multi(Q, CP, POSR, SID, PO, PM, PL, IDS, NKR, N, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr,
                  QSA: tl.constexpr, BITS: tl.constexpr, KT: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    n = tl.load(POSR + r) + 1
    sparse = False
    if QSA:                                          # a row is sparse when its select would mark it (end past TOP)
        sparse = n // RATIO > TOP
        n = tl.where(sparse, tl.load(NKR + r), n)
    _chunk(Q, _ptr(CP, s, KT), _ptr(CP + N, s, KT), _ptr(CP + 2 * N, s, tl.float16), _ptr(CP + 3 * N, s, tl.float16),
           n, sparse, r, tl.program_id(1), tl.program_id(2), PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW, QSA,
           BITS)


@triton.jit
def _merge_multi(PO, PM, PL, POSR, OUT, NKR, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                 CH: tl.constexpr, NCH: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr, RATIO: tl.constexpr,
                 TOP: tl.constexpr):
    r = tl.program_id(0)
    n = tl.load(POSR + r) + 1
    if QSA:
        n = tl.where(n // RATIO > TOP, tl.load(NKR + r), n)
    _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, HK, D, G, CH, NCH, BITS)


class Step:
    """A step's row, stream and cache-pointer tables (the MTP head's with ``mtp``), read now: caches may move."""

    def __init__(self, w, segs: Sequence, mtp: bool) -> None:
        n, rows = len(segs), segs[-1][2]
        layers = [l for l in w.layers if not l.linear] if not mtp else [w.mtp.layer]
        self.index = {layer.index: i for i, layer in enumerate(layers)}
        first = [st.mtp_len if mtp else st.pos for st, _, _ in segs]
        posr, sid = np.empty((rows,), np.int32), np.empty((rows,), np.int32)
        for s, ((_, a0, a1), p0) in enumerate(zip(segs, first)):
            posr[a0:a1] = p0 + np.arange(a1 - a0)
            sid[a0:a1] = s
        counts = [a1 - a0 for _, a0, a1 in segs]
        ptrs = np.empty((len(layers), PTRS, n), np.int64)
        for s, (st, _, _) in enumerate(segs):
            for i, layer in enumerate(layers):
                if mtp:
                    kc, ikc, pooled = st.mtp_kc, st.mtp_ikc, st.mtp_pooled
                else:
                    a = st.att_index[layer.index]
                    kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
                ptrs[i, :, s] = [kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(),
                                 ikc.data_ptr(), pooled.data_ptr()]
        dev = w.device
        ints = shared.to_device(np.concatenate([posr, sid, first, counts]).tolist(), torch.int32, dev)
        self.posr, self.sid = ints[:rows], ints[rows:2 * rows]
        self.first, self.counts = ints[2 * rows:2 * rows + n], ints[2 * rows + n:]
        self.ptrs = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev).view(len(layers), PTRS * n)
        self.n, self.rows, self.segs = n, rows, list(segs)
        self.ends = [p0 + c for p0, c in zip(first, counts)]
        self.most = max(counts)


def layer(layer, w, b, step: Step, mtp: bool, scale: float) -> torch.Tensor:
    """Every stream's ``layer`` attention: prep, pool, sparse streams' own selects, chunks, merge -> b.attn_o[:R]."""

    c, a, sc = w.cfg, layer.attn, b.attn
    n, rows, cp = step.n, step.rows, step.ptrs[step.index[layer.index]]
    st0 = step.segs[0][0]
    cache0 = st0.mtp_kc if mtp else st0.kc[0]
    bits = 0 if not cache0.quantized else cache0.bits
    kt = {0: tl.bfloat16, 8: tl.int8, 4: tl.uint8}[bits]
    heads = c.heads + c.kv_heads + c.index_heads + 1
    _prep_multi[(rows, heads)](b.pa[:rows], step.posr, step.sid, cp, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq,
                               b.q, b.iq, c.eps, n, PW=b.pa.shape[1], NQ=c.heads, NKV=c.kv_heads, HD=c.head_dim,
                               NI=c.index_heads, IHD=c.index_dim, HALF=w.inv_freq.numel(), BITS=bits, KT=kt,
                               num_warps=2)
    top = sc.budget // sc.ratio
    if sc.qsa:
        _pool_multi[(n, step.most // sc.ratio + 2)](cp, step.first, step.counts, a.ik_scale, w.inv_freq, c.eps, n,
                                                    DI=c.index_dim, HALF=w.inv_freq.numel(), RATIO=sc.ratio,
                                                    num_warps=1)
        for (st, a0, a1), end in zip(step.segs, step.ends):
            if end // sc.ratio > top:                # this stream has sparse rows: its own select
                _, _, pooled, pos, _ = _caches(layer, st, mtp)
                attn_mod.qsa_rows(b.iq[a0:a1], pooled, pos, _rows_from(sc, a0), a1 - a0, context=end)
    keys = max(step.ends)
    if sc.qsa:
        keys = min(keys, (top + 1) * sc.ratio - 1)
    chunks = min(sc.nch, triton.cdiv(keys, CHUNK))
    hk = c.kv_heads
    g = c.heads // hk
    _chunks_multi[(rows, hk, chunks)](b.q, cp, step.posr, step.sid, sc.po, sc.pm, sc.pl, sc.ids, sc.nk, n, H=c.heads,
                                      HK=hk, D=c.head_dim, G=g, CH=CHUNK, NCH=sc.nch, SCALE=scale, IDW=sc.idw,
                                      QSA=sc.qsa, BITS=bits, KT=kt, RATIO=sc.ratio, TOP=top, num_warps=4,
                                      num_stages=1)
    _merge_multi[(rows, hk)](sc.po, sc.pm, sc.pl, step.posr, b.attn_o, sc.nk, H=c.heads, HK=hk, D=c.head_dim, G=g,
                             CH=CHUNK, NCH=sc.nch, QSA=sc.qsa, BITS=bits, RATIO=sc.ratio, TOP=top, num_warps=4)
    return b.attn_o[:rows]


def _caches(layer, st, mtp: bool) -> tuple:
    if mtp:
        return st.mtp_kc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, st.mtp_len
    ai = st.att_index[layer.index]
    return st.kc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, st.pos


def _rows_from(sc, a0: int) -> SimpleNamespace:
    """The scratch as seen by a launch whose row 0 is window row ``a0``."""

    return SimpleNamespace(ratio=sc.ratio, budget=sc.budget, nb=sc.nb, idw=sc.idw, scores=sc.scores[a0:],
                           ids=sc.ids[a0:], nk=sc.nk[a0:], sparse=sc.sparse[a0:])
