"""Tree attention: key chunks by absolute position merge in key order, so a row's bits never depend on its launch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
import triton
import triton.language as tl

TILE = 64
CHUNK = 512
MAX_NODES = 128
QUERY_TILE = 16


@triton.jit
def _paths(PARENTS, PATHS, DEPTHS, MAXD: tl.constexpr):
    """A row's root-to-row window rows (parents are window rows, -1 at a root) and its depth."""

    node = tl.program_id(0)
    cur = node
    depth = 0
    while (cur >= 0) & (depth < MAXD):
        depth += 1
        cur = tl.load(PARENTS + cur)
    tl.store(DEPTHS + node, depth)
    cur = node
    slot = depth - 1
    while slot >= 0:
        tl.store(PATHS + node * MAXD + slot, cur)
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
def _shared(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr):
    """Item (stream, first pair, chunk): 16 (row, head) pairs of one stream against one chunk of committed keys."""

    item = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    chunk = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + s * 4)
    rows = tl.load(STREAM + s * 4 + 1)
    p = tl.load(STREAM + s * 4 + 2)
    if (chunk + 1) * CH <= p:                 # a plan padded for a longer context (a graph's) skips missing chunks
        koff = tl.load(OFF + s * 2)
        voff = tl.load(OFF + s * 2 + 1)
        rr = first + tl.arange(0, 16)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        key = chunk * CH + tl.arange(0, 64)
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None],
                    other=0).to(tl.bfloat16)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        for t in range(CH // 64):
            ki = key + t * 64
            kk = tl.load(KC + koff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
            vv = tl.load(VC + voff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
            m, l, o = _tile(q, kk, vv, m, l, o, ki < p, SCALE)
        base = (chunk * W + node) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=ok[:, None])
        tl.store(PM + base, m, mask=ok)
        tl.store(PL + base, l, mask=ok)


@triton.jit
def _tail(Q, KN, VN, KC, VC, OFF, STREAM, ROWS, PATHS, DEPTHS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, MAXD: tl.constexpr, SCALE: tl.constexpr):
    """Row, head group, tail chunk: the last committed keys and the row's own path."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    p = tl.load(STREAM + s * 4 + 2)
    nch = tl.load(STREAM + s * 4 + 3)
    chunk = p // CH + tl.program_id(2)
    if chunk < nch:
        koff = tl.load(OFF + s * 2)
        voff = tl.load(OFF + s * 2 + 1)
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (node * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0).to(tl.bfloat16)
        depth = tl.load(DEPTHS + node)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        key = chunk * CH + tl.arange(0, 64)
        for t in range(CH // 64):
            logical = key + t * 64
            committed = logical < p
            path_slot = logical - p
            on_path = (path_slot >= 0) & (path_slot < depth)
            path_node = tl.load(PATHS + node * MAXD + path_slot, mask=on_path, other=0)
            kc = tl.load(KC + koff + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0)
            vc = tl.load(VC + voff + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0)
            kn = tl.load(KN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
            vn = tl.load(VN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
            kk = tl.where(committed[:, None], kc, kn).to(tl.bfloat16)
            vv = tl.where(committed[:, None], vc, vn).to(tl.bfloat16)
            m, l, o = _tile(q, kk, vv, m, l, o, committed | on_path, SCALE)
        base = (chunk * W + node) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, OUT, STREAM, ROWS, W, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr):
    node = tl.program_id(0)
    hk = tl.program_id(1)
    nch = tl.load(STREAM + tl.load(ROWS + node) * 4 + 3)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    head = hk * G + gg
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for chunk in range(nch):
        base = (chunk * W + node) * H + head
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
    tl.store(OUT + (node * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16), mask=gg[:, None] < G)


@dataclass
class Plan:
    """A window's attention layout, shared by every attention layer of a forward."""

    rows: torch.Tensor          # (W,) int32: each window row's stream
    streams: torch.Tensor       # (S, 4) int32: first row, rows, committed keys, chunks
    items: torch.Tensor         # (items, 3) int32: (stream, first (row, head) pair, chunk) of committed keys
    parents: torch.Tensor       # (W,) int32 window rows, -1 at a root
    paths: torch.Tensor         # (W, MAX_NODES) int32
    depths: torch.Tensor        # (W,) int32
    chunks: int                 # the most chunks any stream has
    width: int                  # W


def plan_host(parents: Sequence[Sequence[int]], lengths: Sequence[int], group: int) -> tuple[list[int], int, int]:
    """Host half of ``plan`` from window-local parents, committed key counts and ``group`` query heads a key head."""

    rows, streams, items, glob = [], [], [], []
    start, most = 0, 0
    for s, (local, p) in enumerate(zip(parents, lengths)):
        w = len(local)
        if not 1 <= w <= MAX_NODES:
            raise ValueError(f"a stream's window takes 1..{MAX_NODES} rows")
        nch = -(-(p + w) // CHUNK)
        most = max(most, nch)
        rows += [s] * w
        streams += [start, w, p, nch]
        glob += [-1 if x < 0 else x + start for x in local]
        for first in range(0, w * group, QUERY_TILE):
            for chunk in range(p // CHUNK):
                items += [s, first, chunk]
        start += w
    return rows + streams + items + glob, len(items) // 3, most


def padded_host(parents: Sequence[int], context: int, group: int) -> tuple[list[int], int, int]:
    """``plan_host`` for one stream with items for every full chunk below ``context`` (a graph covers that range)."""

    flat, _, _ = plan_host([parents], [0], group)            # rows, the stream's row (refreshed per replay), parents
    w = len(parents)
    items = [x for first in range(0, w * group, QUERY_TILE) for chunk in range(context // CHUNK)
             for x in (0, first, chunk)]
    return flat[:w + 4] + items + flat[w + 4:], len(items) // 3, -(-(context + w) // CHUNK)


def plan(parents: Sequence[Sequence[int]], lengths: Sequence[int], group: int, device) -> Plan:
    flat, n_items, most = plan_host(parents, lengths, group)
    dev = torch.tensor(flat, dtype=torch.int32).pin_memory().to(device, non_blocking=True)
    return from_packed(dev, len(parents), sum(len(p) for p in parents), n_items, most)


def from_packed(dev: torch.Tensor, streams: int, width: int, n_items: int, chunks: int) -> Plan:
    """A ``Plan`` from ``plan_host``'s list already on the device (paths computed here, once a forward)."""

    rows = dev[:width]
    table = dev[width:width + 4 * streams].view(streams, 4)
    items = dev[width + 4 * streams:width + 4 * streams + 3 * n_items].view(n_items, 3)
    parents = dev[width + 4 * streams + 3 * n_items:width + 4 * streams + 3 * n_items + width]
    paths = torch.empty((width, MAX_NODES), dtype=torch.int32, device=dev.device)
    depths = torch.empty((width,), dtype=torch.int32, device=dev.device)
    _paths[(width,)](parents, paths, depths, MAXD=MAX_NODES, num_warps=1)
    return Plan(rows, table, items, parents, paths, depths, chunks, width)


def base(device) -> torch.Tensor:
    """A fixed bf16 tensor that cache offsets are measured from (so an empty cache still has a valid offset)."""

    device = torch.device(device)
    return _base(device.index if device.index is not None else torch.cuda.current_device())


@lru_cache(maxsize=None)
def _base(index: int) -> torch.Tensor:
    return torch.zeros(64, dtype=torch.bfloat16, device=torch.device("cuda", index))


def offsets(caches: Sequence[tuple[torch.Tensor, torch.Tensor]], device) -> list[int]:
    """Each stream's key and value cache as bf16 element offsets from ``base(device)``, in stream order."""

    origin = base(device).data_ptr()
    out = []
    for k, v in caches:
        for t in (k, v):
            if t.dtype != torch.bfloat16 or not t.is_contiguous():
                raise ValueError("caches: contiguous bf16 tensors")
            delta = t.data_ptr() - origin
            if delta % 16:
                raise ValueError("caches must be 16-byte aligned")
            out.append(delta // 2)
    return out


def attention(q: torch.Tensor, k_nodes: torch.Tensor, v_nodes: torch.Tensor, offs: torch.Tensor, p: Plan, *,
              scale: float) -> torch.Tensor:
    """Attend (W, H, D) queries to committed keys and own paths; ``offs`` holds ``offsets`` as device (S, 2) int64."""

    w, h, d = q.shape
    hk = k_nodes.shape[1]
    if not (w == p.width and d in (128, 256) and k_nodes.shape == (w, hk, d) and v_nodes.shape == k_nodes.shape
            and h % hk == 0 and h // hk <= QUERY_TILE):
        raise ValueError("unsupported attention shape")
    if any(x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() for x in (q, k_nodes, v_nodes)):
        raise ValueError("q and node keys and values must be contiguous CUDA bf16 tensors")
    origin = base(q.device)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    g = h // hk
    partial_o = torch.empty((p.chunks, w, h, d), dtype=torch.float32, device=q.device)
    partial_m = torch.empty((p.chunks, w, h), dtype=torch.float32, device=q.device)
    partial_l = torch.empty_like(partial_m)
    if p.items.shape[0]:
        _shared[(p.items.shape[0], hk)](q, origin, origin, offs, p.streams, p.items, partial_o, partial_m, partial_l, w,
                                        H=h, HK=hk, D=d, G=g, CH=CHUNK, SCALE=scale, num_warps=4, num_stages=1)
    tails = 1 + -(-MAX_NODES // CHUNK)
    _tail[(w, hk, tails)](q, k_nodes, v_nodes, origin, origin, offs, p.streams, p.rows, p.paths, p.depths,
                          partial_o, partial_m, partial_l, w, H=h, HK=hk, D=d, G=g, CH=CHUNK, MAXD=MAX_NODES,
                          SCALE=scale, num_warps=4, num_stages=1)
    out = torch.empty_like(q)
    _merge[(w, hk)](partial_o, partial_m, partial_l, out, p.streams, p.rows, w, H=h, D=d, G=g, num_warps=4)
    return out
