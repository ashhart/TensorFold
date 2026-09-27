"""Attention without RoPE: 512-key chunks at absolute positions merge in order, so a row's bits ignore its window."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = 512          # keys a chunk, at fixed absolute positions


@triton.jit(do_not_specialize=["R"])
def _kv_write(QKV, KC, VC, META, R, NQKV: tl.constexpr, QD: tl.constexpr, KVD: tl.constexpr):
    r = tl.program_id(0)
    pos = tl.load(META)
    offs = tl.arange(0, KVD)
    k = tl.load(QKV + r * NQKV + QD + offs)
    v = tl.load(QKV + r * NQKV + QD + KVD + offs)
    tl.store(KC + (pos + r) * KVD + offs, k)
    tl.store(VC + (pos + r) * KVD + offs, v)


def kv_write(qkv, k_cache, v_cache, meta, rows: int, *, q_dim: int) -> None:
    kvd = k_cache.shape[1] * k_cache.shape[2]
    _kv_write[(rows,)](qkv, k_cache, v_cache, meta, rows, NQKV=qkv.shape[1], QD=q_dim, KVD=kvd, num_warps=2)


@triton.jit
def _tile(q, k, v, m, den, o, valid, scale: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    den = den * alpha + tl.sum(p, 1)
    return next_m, den, o


@triton.jit
def _chunk(QKV, KC, VC, META, PO, PM, PL, NQKV: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
           G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, SCALE: tl.constexpr):
    """Program (row, KV head, chunk): the row's G query heads against keys [chunk * CH, +CH) below its limit."""

    r = tl.program_id(0)
    hk = tl.program_id(1)
    c = tl.program_id(2)
    limit = tl.load(META) + r + 1
    if c * CH < limit:
        gg = tl.arange(0, 16)
        head = hk * G + gg
        d = tl.arange(0, D)
        q = tl.load(QKV + r * NQKV + head[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        m = tl.full((16,), float("-inf"), tl.float32)
        den = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        for t in range(CH // 64):
            key = c * CH + t * 64 + tl.arange(0, 64)
            valid = key < limit
            kk = tl.load(KC + (key[:, None] * HK + hk) * D + d[None, :], mask=valid[:, None], other=0.0)
            vv = tl.load(VC + (key[:, None] * HK + hk) * D + d[None, :], mask=valid[:, None], other=0.0)
            m, den, o = _tile(q, kk, vv, m, den, o, valid, SCALE)
        base = (r * NCH + c) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, den, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, META, OUT, XS, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
           CH: tl.constexpr, NCH: tl.constexpr):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    limit = tl.load(META) + r + 1
    nch = (limit + CH - 1) // CH
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    head = hk * G + gg
    hm = gg < G
    m = tl.full((16,), float("-inf"), tl.float32)
    den = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for c in range(nch):
        base = (r * NCH + c) * H + head
        cm = tl.load(PM + base, mask=hm, other=float("-inf"))
        cl = tl.load(PL + base, mask=hm, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a[:, None] + co * b[:, None]
        den = den * a + cl * b
        m = next_m
    out = (o / den[:, None]).to(tl.bfloat16)
    tl.store(OUT + (r * H + head[:, None]) * D + d[None, :], out, mask=hm[:, None])
    og = tl.reshape(out.to(tl.float32), (16, D // 64, 64))
    gi = tl.arange(0, D // 64)
    tl.store(XS + r * (H * D // 64) + head[:, None] * (D // 64) + gi[None, :], tl.sum(og, axis=2), mask=hm[:, None])


def attention(qkv, k_cache, v_cache, meta, rows: int, *, heads: int, kv_heads: int, head_dim: int, max_chunks: int):
    """The window's rows (already in the cache) against their keys: (R, heads * head_dim) bf16 and 64-group sums."""

    g = heads // kv_heads
    if g > 16 or head_dim not in (64, 128, 256):
        raise ValueError("attention takes up to 16 query heads a KV head and head dim 64/128/256")
    dev = qkv.device
    po = torch.empty((rows, max_chunks, heads, head_dim), dtype=torch.float32, device=dev)
    pm = torch.empty((rows, max_chunks, heads), dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    _chunk[(rows, kv_heads, max_chunks)](qkv, k_cache, v_cache, meta, po, pm, pl, NQKV=qkv.shape[1], H=heads,
                                         HK=kv_heads, D=head_dim, G=g, CH=CHUNK, NCH=max_chunks,
                                         SCALE=head_dim ** -0.5, num_warps=4, num_stages=1)
    out = torch.empty((rows, heads * head_dim), dtype=torch.bfloat16, device=dev)
    xs = torch.empty((rows, heads * head_dim // 64), dtype=torch.float32, device=dev)
    _merge[(rows, kv_heads)](po, pm, pl, meta, out, xs, H=heads, HK=kv_heads, D=head_dim, G=g, CH=CHUNK,
                             NCH=max_chunks, num_warps=4)
    return out, xs
