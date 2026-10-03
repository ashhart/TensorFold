"""Tree attention folds absolute-position groups in a fixed order, independent of committed and drafted rows."""

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
GROUP = 4               # chunks folded into one fp32 partial
SPAN = CHUNK * GROUP
MIN_GROUPED = 64        # groups times window rows needed to keep enough programs busy
MAX_NODES = 128
QUERY_TILE = 16
MERGE_COLUMNS = 64      # output columns a merge program folds (the chunk fold is per column: more programs, same bits)


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
def _fold(m, l, o, cm, cl, co):
    """The same fp32 fold in the group program and the merge keeps each row independent of its launch."""

    active = cl > 0.0
    next_m = tl.where(active, tl.maximum(m, cm), m)
    a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    b = tl.where(active, tl.exp(cm - next_m), 0.0)
    return next_m, l * a + cl * b, o * a[:, None] + co * b[:, None]


@triton.jit
def _shared(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, GR: tl.constexpr):
    """An item code is a whole group j >= 0, or chunk -1-j after the whole groups."""

    item = tl.program_id(0) // HK            # a chunk's heads and query tiles launch together: one DRAM read
    hk = tl.program_id(0) % HK
    s = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    code = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + s * 4)
    rows = tl.load(STREAM + s * 4 + 1)
    p = tl.load(STREAM + s * 4 + 2)
    groups = ((p + rows + CH - 1) // CH - tl.load(STREAM + s * 4 + 3)) // (GR - 1)     # folded in their programs
    whole = code >= 0
    chunk = tl.where(whole, code * GR, groups * GR - 1 - code)
    n = tl.where(whole, GR, 1)
    slot = tl.where(whole, code, chunk - groups * (GR - 1))
    if ((chunk + n) * CH <= p) & (code < groups):     # a padded plan (a graph's) skips what the keys don't fill
        koff = tl.multiple_of(tl.load(OFF + s * 2), 8)     # ``offsets`` checks 16 bytes: loads go 16 bytes wide
        voff = tl.multiple_of(tl.load(OFF + s * 2 + 1), 8)
        rr = first + tl.arange(0, 16)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None],
                    other=0).to(tl.bfloat16)
        gm = tl.full((16,), float("-inf"), tl.float32)
        gl = tl.zeros((16,), tl.float32)
        go = tl.zeros((16, D), tl.float32)
        m, l, o = gm, gl, go
        for c in range(n):
            key = (chunk + c) * CH + tl.arange(0, 64)
            m = tl.full((16,), float("-inf"), tl.float32)
            l = tl.zeros((16,), tl.float32)
            o = tl.zeros((16, D), tl.float32)
            for t in range(CH // 64):
                ki = key + t * 64
                kk = tl.load(KC + koff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                vv = tl.load(VC + voff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                m, l, o = _tile(q, kk, vv, m, l, o, ki < p, SCALE)
            gm, gl, go = _fold(gm, gl, go, m, l, o)
        if not whole:                         # a chunk's partial goes out as the tail writes its own: unfolded
            gm, gl, go = m, l, o
        base = (slot * W + node) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], go, mask=ok[:, None])
        tl.store(PM + base, gm, mask=ok)
        tl.store(PL + base, gl, mask=ok)


@triton.jit
def _tail(Q, KN, VN, KC, VC, OFF, STREAM, ROWS, PATHS, DEPTHS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, MAXD: tl.constexpr, SCALE: tl.constexpr,
          GR: tl.constexpr):
    """Row, head group, tail chunk: the last committed keys and the row's own path."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    p = tl.load(STREAM + s * 4 + 2)
    nch = (p + tl.load(STREAM + s * 4 + 1) + CH - 1) // CH        # chunks through the window's last key
    groups = (nch - tl.load(STREAM + s * 4 + 3)) // (GR - 1)       # folded in their programs
    chunk = p // CH + tl.program_id(2)
    if chunk < nch:
        koff = tl.multiple_of(tl.load(OFF + s * 2), 8)
        voff = tl.multiple_of(tl.load(OFF + s * 2 + 1), 8)
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
        base = ((chunk - groups * (GR - 1)) * W + node) * H + hk * G + gg      # the chunk's slot
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, OUT, STREAM, ROWS, W, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, DS: tl.constexpr,
           CH: tl.constexpr, GR: tl.constexpr):
    """Fold every group's chunks in order, then each group in order, regardless of where it was computed."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    nslots = tl.load(STREAM + s * 4 + 3)
    groups = ((tl.load(STREAM + s * 4 + 2) + tl.load(STREAM + s * 4 + 1) + CH - 1) // CH - nslots) // (GR - 1)
    gg = tl.arange(0, 16)
    d = tl.program_id(2) * DS + tl.arange(0, DS)
    head = hk * G + gg
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, DS), tl.float32)
    for slot in range(groups):
        base = (slot * W + node) * H + head
        cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
        cl = tl.load(PL + base, mask=gg < G, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        m, l, o = _fold(m, l, o, cm, cl, co)
    gm = tl.full((16,), float("-inf"), tl.float32)
    gl = tl.zeros((16,), tl.float32)
    go = tl.zeros((16, DS), tl.float32)
    for slot in range(groups, nslots):
        base = (slot * W + node) * H + head
        cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
        cl = tl.load(PL + base, mask=gg < G, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        gm, gl, go = _fold(gm, gl, go, cm, cl, co)
        if ((slot - groups + 1) % GR == 0) | (slot == nslots - 1):     # a group's last chunk, or the last one
            m, l, o = _fold(m, l, o, gm, gl, go)
            gm = tl.full((16,), float("-inf"), tl.float32)
            gl = tl.zeros((16,), tl.float32)
            go = tl.zeros((16, DS), tl.float32)
    result = o / l[:, None]
    tl.store(OUT + (node * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16), mask=gg[:, None] < G)


@dataclass
class Plan:
    """A window's attention layout, shared by every attention layer of a forward."""

    rows: torch.Tensor          # (W,) int32: each window row's stream
    streams: torch.Tensor       # (S, 4) int32: first row, rows, committed keys, partial slots (``slots``)
    items: torch.Tensor         # (items, 3) int32: (stream, first (row, head) pair, group j or chunk -1 - j), key-major
    parents: torch.Tensor       # (W,) int32 window rows, -1 at a root
    paths: torch.Tensor         # (W, MAX_NODES) int32
    depths: torch.Tensor        # (W,) int32
    chunks: int                 # the most partial slots any stream has
    width: int                  # W


def groups(p: int, w: int) -> int:
    """Whole committed groups fold in their programs only when groups times rows fills the GPU."""

    g = p // SPAN
    return g if g * w >= MIN_GROUPED else 0


def slots(p: int, w: int) -> int:
    """One slot per folded group, then one per chunk through the window's last key."""

    g = groups(p, w)
    return g + -(-(p + w) // CHUNK) - g * GROUP


def plan_host(parents: Sequence[Sequence[int]], lengths: Sequence[int], group: int) -> tuple[list[int], int, int]:
    """Host half of ``plan`` from window-local parents, committed key counts and ``group`` query heads a key head."""

    rows, streams, items, glob = [], [], [], []
    start, most = 0, 0
    for s, (local, p) in enumerate(zip(parents, lengths)):
        w = len(local)
        if not 1 <= w <= MAX_NODES:
            raise ValueError(f"a stream's window takes 1..{MAX_NODES} rows")
        n = slots(p, w)
        most = max(most, n)
        rows += [s] * w
        streams += [start, w, p, n]
        glob += [-1 if x < 0 else x + start for x in local]
        g = groups(p, w)
        codes = list(range(g)) + [-1 - j for j in range(p // CHUNK - g * GROUP)]    # folded groups, then chunks
        for code in codes:
            for first in range(0, w * group, QUERY_TILE):
                items += [s, first, code]
        start += w
    return rows + streams + items + glob, len(items) // 3, most


def padded_host(parents: Sequence[int], context: int, group: int) -> tuple[list[int], int, int]:
    """A graph includes all groups and chunks below its context; kernels skip items its keys do not fill."""

    flat, _, _ = plan_host([parents], [0], group)            # rows, the stream's row (refreshed per replay), parents
    w = len(parents)
    codes = list(range(context // SPAN)) + [-1 - j for j in range(context // CHUNK)]
    items = [x for code in codes for first in range(0, w * group, QUERY_TILE) for x in (0, first, code)]
    most = -(-(context + w) // CHUNK)                        # ``slots`` of any p <= context is at most this
    return flat[:w + 4] + items + flat[w + 4:], len(items) // 3, most


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
    from tensorfold.cuda.rocm import HIP

    # ROCm: the group program, tail and merge must round the shared fp32 fold the same way; with FMA contraction
    # AMD's compiler fuses them differently per kernel, so rows crossing a group edge lost serial bits on gfx1151.
    fuse = {"enable_fp_fusion": False} if HIP else {}
    if p.items.shape[0]:
        _shared[(p.items.shape[0] * hk,)](q, origin, origin, offs, p.streams, p.items, partial_o, partial_m, partial_l,
                                          w, H=h, HK=hk, D=d, G=g, CH=CHUNK, SCALE=scale, GR=GROUP, num_warps=4,
                                          num_stages=1, **fuse)
    tails = 1 + -(-MAX_NODES // CHUNK)
    _tail[(w, hk, tails)](q, k_nodes, v_nodes, origin, origin, offs, p.streams, p.rows, p.paths, p.depths,
                          partial_o, partial_m, partial_l, w, H=h, HK=hk, D=d, G=g, CH=CHUNK, MAXD=MAX_NODES,
                          SCALE=scale, GR=GROUP, num_warps=4, num_stages=1, **fuse)
    out = torch.empty_like(q)
    _merge[(w, hk, d // MERGE_COLUMNS)](partial_o, partial_m, partial_l, out, p.streams, p.rows, w, H=h, D=d, G=g,
                                        DS=MERGE_COLUMNS, CH=CHUNK, GR=GROUP, num_warps=4, **fuse)
    return out
