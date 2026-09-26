"""Tree decode attention with fixed key tiles for serial and windowed calls.

Each query sees the committed cache followed by its root-to-node path. A program
handles 16 query heads and a 512-key chunk, with eight 64-key tensor-core tiles.
Chunk partials merge in absolute key order. No arithmetic choice depends on the
number of other nodes in the window.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


TILE = 64
CHUNK = 512
MAX_NODES = 128
QUERY_TILE = 16


@triton.jit
def _paths(PARENTS, PATHS, DEPTHS, W: tl.constexpr):
    node = tl.program_id(0)
    cur = node
    depth = 0
    while (cur >= 0) & (depth < W):
        depth += 1
        cur = tl.load(PARENTS + cur)
    tl.store(DEPTHS + node, depth)
    cur = node
    slot = depth - 1
    while slot >= 0:
        tl.store(PATHS + node * 128 + slot, cur)
        cur = tl.load(PARENTS + cur)
        slot -= 1


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
def _shared(Q, KC, VC, PO, PM, PL, P, W: tl.constexpr, CH: tl.constexpr,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
            SCALE: tl.constexpr):
    row_block = tl.program_id(0)
    hk = tl.program_id(1)
    chunk = tl.program_id(2)
    rr = row_block * 16 + tl.arange(0, 16)
    node = rr // G
    head = hk * G + rr % G
    d = tl.arange(0, D)
    key = chunk * CH + tl.arange(0, 64)
    q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :],
                mask=(rr[:, None] < W * G), other=0).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for t in range(CH // 64):
        ki = key + t * 64
        kk = tl.load(KC + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
        vv = tl.load(VC + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
        m, l, o = _tile(q, kk, vv, m, l, o, ki < P, SCALE)
    base = ((chunk * W + node) * H + head)
    tl.store(PO + base[:, None] * D + d[None, :], o, mask=rr[:, None] < W * G)
    tl.store(PM + base, m, mask=rr < W * G)
    tl.store(PL + base, l, mask=rr < W * G)


@triton.jit
def _tail(Q, KN, VN, KC, VC, PATHS, DEPTHS, PO, PM, PL,
          P, W: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, FULL, SCALE: tl.constexpr):
    node = tl.program_id(0)
    hk = tl.program_id(1)
    chunk = FULL + tl.program_id(2)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (node * H + hk * G + gg[:, None]) * D + d[None, :],
                mask=gg[:, None] < G, other=0).to(tl.bfloat16)
    depth = tl.load(DEPTHS + node)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    key = chunk * CH + tl.arange(0, 64)
    for t in range(CH // 64):
        logical = key + t * 64
        committed = logical < P
        path_slot = logical - P
        on_path = (path_slot >= 0) & (path_slot < depth)
        path_node = tl.load(PATHS + node * 128 + path_slot,
                            mask=on_path, other=0)
        kc = tl.load(KC + (logical[:, None] * HK + hk) * D + d[None, :],
                     mask=committed[:, None], other=0)
        vc = tl.load(VC + (logical[:, None] * HK + hk) * D + d[None, :],
                     mask=committed[:, None], other=0)
        kn = tl.load(KN + (path_node[:, None] * HK + hk) * D + d[None, :],
                     mask=on_path[:, None], other=0)
        vn = tl.load(VN + (path_node[:, None] * HK + hk) * D + d[None, :],
                     mask=on_path[:, None], other=0)
        kk = tl.where(committed[:, None], kc, kn).to(tl.bfloat16)
        vv = tl.where(committed[:, None], vc, vn).to(tl.bfloat16)
        m, l, o = _tile(q, kk, vv, m, l, o, committed | on_path, SCALE)
    base = ((chunk * W + node) * H + hk * G + gg)
    tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
    tl.store(PM + base, m, mask=gg < G)
    tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, OUT, W: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
           D: tl.constexpr, G: tl.constexpr, NCH: tl.constexpr):
    node = tl.program_id(0)
    hk = tl.program_id(1)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    head = hk * G + gg
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for chunk in range(NCH):
        base = ((chunk * W + node) * H + head)
        cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
        cl = tl.load(PL + base, mask=gg < G, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :],
                     mask=gg[:, None] < G, other=0.0)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a[:, None] + co * b[:, None]
        l = l * a + cl * b
        m = next_m
    result = o / l[:, None]
    tl.store(OUT + (node * H + head[:, None]) * D + d[None, :],
             result.to(tl.bfloat16), mask=gg[:, None] < G)


def attention(q: torch.Tensor, k_nodes: torch.Tensor, v_nodes: torch.Tensor,
              k_cache: torch.Tensor, v_cache: torch.Tensor, parents: torch.Tensor,
              *, scale: float, chunk_size: int = CHUNK) -> torch.Tensor:
    """Attend (W,H,D) queries to committed KV and each node's ancestor path.

    K/V have shapes (W,HK,D) and (P,HK,D). Parent indices are int32 on the
    device, earlier than the child, and -1 denotes a root. The node itself is
    always included. Input tensors must be contiguous bf16 CUDA tensors.
    """

    if q.ndim != 3 or k_nodes.ndim != 3 or v_nodes.shape != k_nodes.shape:
        raise ValueError("q and node K/V must be rank-3, with matching K/V shapes")
    w, h, d = q.shape
    p, hk, kd = k_cache.shape
    if not (1 <= w <= MAX_NODES and d in (128, 256) and d == kd and
            k_nodes.shape == (w, hk, d) and v_cache.shape == k_cache.shape and
            h % hk == 0 and h // hk <= QUERY_TILE):
        raise ValueError("unsupported attention shape")
    tensors = (q, k_nodes, v_nodes, k_cache, v_cache)
    if any(x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() for x in tensors):
        raise ValueError("Q and K/V must be contiguous CUDA bf16 tensors")
    if parents.shape != (w,) or parents.dtype != torch.int32 or not parents.is_cuda or not parents.is_contiguous():
        raise ValueError("parents must be a contiguous CUDA int32 vector of W entries")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if chunk_size not in (256, 512, 1024, 2048):
        raise ValueError("chunk_size must be 256, 512, 1024 or 2048")

    paths = torch.empty((w, MAX_NODES), dtype=torch.int32, device=q.device)
    depths = torch.empty((w,), dtype=torch.int32, device=q.device)
    _paths[(w,)](parents, paths, depths, W=w, num_warps=1)
    full = p // chunk_size
    nch = triton.cdiv(p + w, chunk_size)
    # A tree path can be shorter than W; the unneeded tail chunks stay empty.
    partial_o = torch.empty((nch, w, h, d), dtype=torch.float32, device=q.device)
    partial_m = torch.empty((nch, w, h), dtype=torch.float32, device=q.device)
    partial_l = torch.empty_like(partial_m)
    g = h // hk
    if full:
        _shared[(triton.cdiv(w * g, QUERY_TILE), hk, full)](
            q, k_cache, v_cache, partial_o, partial_m, partial_l,
            P=p, W=w, H=h, HK=hk, D=d, G=g, CH=chunk_size, SCALE=scale, num_warps=4, num_stages=1)
    _tail[(w, hk, nch - full)](
        q, k_nodes, v_nodes, k_cache, v_cache, paths, depths,
        partial_o, partial_m, partial_l,
        P=p, W=w, H=h, HK=hk, D=d, G=g, CH=chunk_size, FULL=full, SCALE=scale, num_warps=4, num_stages=1)
    out = torch.empty_like(q)
    _merge[(w, hk)](partial_o, partial_m, partial_l, out,
                    W=w, H=h, HK=hk, D=d, G=g, NCH=nch, num_warps=4)
    return out
