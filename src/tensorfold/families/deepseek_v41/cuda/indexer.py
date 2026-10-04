"""DeepSeek-V4.1's lightning indexer: index Q to each row's ascending top-k entries, each row alone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import qmm

from . import mx8, quant, rope
from .buffers import score_rows

if TYPE_CHECKING:
    from ..config import Config
    from .weights import IdxW

TILE = 128                  # entries (or candidate slots) a score item covers
ROWS = 16                   # rows a score item computes; a decode window is one item a tile
WAVES = 4                   # persistent score programs per SM
PICK = 1024                 # keys a select step reads (blocks: PICK // 2 of them)

_SMS: dict[int, int] = {}


def index_q(w: IdxW, qr: torch.Tensor, table: torch.Tensor, positions: torch.Tensor, out: torch.Tensor, *,
            prompt: bool = False) -> torch.Tensor:
    """M:550-552: out [rows, heads, head_dim] = FP4 QDQ of RoPE(wq_b(qr)) in bf16, rows at ``positions``."""

    rows, heads, dim = out.shape
    if not out.is_contiguous() or heads * dim != w.wq_b.n:
        raise ValueError(f"index_q: out {tuple(out.shape)} is not a contiguous [rows, heads, dim] of {w.wq_b.n}")
    flat = out.view(rows, heads * dim)
    y = mx8.mm(w.wq_b, qr, flat, prompt=prompt)
    if y.data_ptr() != flat.data_ptr():
        flat.copy_(y)
    rope.apply(out, positions, table)
    return quant.fp4_qdq_1x32_e8m0(out)


def index_weights(w: IdxW, xa: torch.Tensor, out: torch.Tensor, part: torch.Tensor | None = None) -> torch.Tensor:
    """M:555: out [rows, heads] = fp32 weights_proj(xa) * head_dim^-0.5 * heads^-0.5 (fp32, never rounded to bf16)."""

    heads = out.shape[1]
    qmm.matmul(xa, w.wproj, out=out, f32=True, part=part)
    return out.mul_((w.wq_b.n // heads) ** -0.5 * heads ** -0.5)


@triton.jit
def _row_scores(Q, W, k, h, hok, d, D: tl.constexpr):
    """sum_h relu(q_h . k_t) * w_h over one row's heads (the MMA's M), t over the tile: fp32 [TILE]."""

    q = tl.load(Q + h[:, None] * D + d[None, :], mask=hok[:, None], other=0.0)
    w = tl.load(W + h, mask=hok, other=0.0)
    return tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)


@triton.jit
def _scores(Q, W, KEYS, OUT, POS, CAND, CANDN, R, row0, ratio, slots, sq, sw, so, sc,
            H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr,
            G: tl.constexpr, CANDIDATES: tl.constexpr):
    """Items (rows, tile) over a persistent grid; row r's entry t is at absolute position pos + row0 + r."""

    first = tl.load(POS).to(tl.int64) + row0
    h = tl.arange(0, HP)
    hok = h < H
    d = tl.arange(0, D)
    j = tl.arange(0, BT)
    if CANDIDATES:                                  # slot c of row r: entry cand[r, c // G] * G + c % G
        tiles = tl.cdiv(slots, BT)
        for i in range(tl.program_id(0), R * tiles, tl.num_programs(0)):
            r = i // tiles
            c = (i % tiles) * BT + j
            n = tl.load(CANDN + r) * G
            if (i % tiles) * BT < n:
                blk = tl.load(CAND + r * sc + c // G, mask=c < n, other=0).to(tl.int64)
                e = blk * G + c % G
                ok = (c < n) & (e < (first + r + 1) // ratio)
                k = tl.load(KEYS + e[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
                s = _row_scores(Q + r * sq, W + r * sw, k, h, hok, d, D)
                tl.store(OUT + r * so + c, s, mask=ok)     # slots past the row's entries are never read
    else:
        tiles = tl.cdiv((first + R) // ratio, BT)
        for i in range(tl.program_id(0), tl.cdiv(R, RB) * tiles, tl.num_programs(0)):
            lo = (i // tiles) * RB
            hi = tl.minimum(R, lo + RB)
            top = (first + hi) // ratio             # entries the item's last row sees
            e = (i % tiles) * BT + j
            if (i % tiles) * BT < top:
                k = tl.load(KEYS + e[:, None] * D + d[None, :], mask=(e < top)[:, None], other=0.0)
                for r in range(lo, hi):             # the tile read once for all the item's rows
                    s = _row_scores(Q + r * sq, W + r * sw, k, h, hok, d, D)
                    s = tl.where(e < (first + r + 1) // ratio, s, float("-inf"))
                    tl.store(OUT + r * so + e, s, mask=e < top)


def _sms(device: torch.device) -> int:
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _SMS:
        _SMS[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _SMS[i]


def scores(qI: torch.Tensor, wI: torch.Tensor, keys: torch.Tensor, ratio: int, pos: torch.Tensor,
           out: torch.Tensor, *, row0: int = 0, cand: torch.Tensor | None = None,
           cand_n: torch.Tensor | None = None, block: int = 8) -> torch.Tensor:
    """Each row's fp32 sum_h relu(q_h . k_t) * w_h over the entries it sees (M:556-567), -inf past them."""

    R, H, D = qI.shape
    if qI.dtype != torch.bfloat16 or keys.dtype != torch.bfloat16 or not qI.is_contiguous() or keys.shape[1] != D \
            or not keys.is_contiguous():
        raise ValueError("scores: contiguous bf16 qI [rows, heads, dim] and keys [entries, dim]")
    if wI.dtype != torch.float32 or tuple(wI.shape) != (R, H) or wI.stride(1) != 1:
        raise ValueError(f"scores: fp32 weights [{R}, {H}] with unit-stride heads")
    if out.dtype != torch.float32 or out.shape[0] < R or out.stride(1) != 1 or out.shape[1] < keys.shape[0]:
        raise ValueError(f"scores: fp32 out of {R} rows and at least {keys.shape[0]} columns")
    if R == 0:
        return out
    if cand is None:
        items, slots = triton.cdiv(R, ROWS) * triton.cdiv(keys.shape[0], TILE), 0
    else:
        if cand.dtype != torch.int32 or cand.shape[0] < R or cand.stride(1) != 1 or cand_n is None:
            raise ValueError("scores: int32 candidate blocks [rows, blocks] and their counts")
        slots = cand.shape[1] * block
        items = R * triton.cdiv(slots, TILE)
    grid = (max(1, min(items, WAVES * _sms(qI.device))),)
    _scores[grid](qI, wI, keys, out, pos, cand if cand is not None else qI, cand_n if cand is not None else qI, R,
                  row0, ratio, slots, qI.stride(0), wI.stride(0), out.stride(0),
                  cand.stride(0) if cand is not None else 0, H=H, HP=max(16, triton.next_power_of_2(H)), D=D,
                  BT=TILE, RB=ROWS, G=block, CANDIDATES=cand is not None, num_warps=4)
    return out


@triton.jit
def _key(s):
    """A float32 as a uint32 whose unsigned order is the floats' order, -0 counted as +0."""

    bits = tl.where(s == 0.0, 0.0, s).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _items(S, CAND, c0, n, vis, G: tl.constexpr, BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr,
           B: tl.constexpr):
    """Keys of items c0.. of one row, whether each takes part, and the id written for it."""

    i = c0 + tl.arange(0, B)
    if BLOCKS:                                      # M:593-604: a block's best visible entry, the newest pinned
        e = i[:, None] * G + tl.arange(0, G)[None, :]
        v = tl.max(tl.load(S + e, mask=e < vis, other=float("-inf")), axis=1)
        v = tl.where(i == (vis - 1) // G, float("inf"), v)
        ok = (i < n) & (v > float("-inf"))
        val = i
    elif CANDIDATES:
        live = i < n
        val = tl.load(CAND + i // G, mask=live, other=0) * G + i % G
        ok = live & (val < vis)
        v = tl.load(S + i, mask=ok, other=0.0)
    else:
        ok = i < n
        v = tl.load(S + i, mask=ok, other=0.0)
        val = i
    return _key(v), ok, val


@triton.jit
def _select(S, OUT, OUTN, POS, CAND, CANDN, row0, ratio, K, ss, so, sc, G: tl.constexpr,
            BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr, B: tl.constexpr):
    """Program r: radix select of the K-th best key, then the items above it and the lowest-numbered ties."""

    r = tl.program_id(0).to(tl.int64)
    vis = (tl.load(POS).to(tl.int64) + row0 + r + 1) // ratio
    row = S + r * ss
    cand = CAND + r * sc
    if BLOCKS:
        n = tl.cdiv(vis, G)
    elif CANDIDATES:
        n = tl.load(CANDN + r).to(tl.int64) * G
    else:
        n = vis
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    count = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, n, B):
            u, ok, val = _items(row, cand, c, n, vis, G, BLOCKS, CANDIDATES, B)
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        if p == 0:
            need = tl.minimum(need, tl.sum(hist, 0))
            count = need
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, n, B):
        u, ok, val = _items(row, cand, c, n, vis, G, BLOCKS, CANDIDATES, B)
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * so + written + tl.cumsum(t, 0) - t, val.to(tl.int32), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)
    tl.store(OUTN + r, count)


def _check(s: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor) -> int:
    R = s.shape[0]
    if s.dtype != torch.float32 or s.stride(1) != 1:
        raise ValueError("select: fp32 scores with unit-stride columns")
    if out.dtype != torch.int32 or out_n.dtype != torch.int32 or out.shape[0] < R or out_n.shape[0] < R \
            or out.stride(1) != 1:
        raise ValueError(f"select: int32 outputs of at least {R} rows")
    return R


def candidates(s: torch.Tensor, ratio: int, pos: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor, *,
               block: int = 8, row0: int = 0) -> torch.Tensor:
    """Each row's best blocks of ``block`` entries by their best score (M:583-610), ascending into out."""

    R = _check(s, out, out_n)
    if R:
        _select[(R,)](s, out, out_n, pos, s, s, row0, ratio, out.shape[1], s.stride(0), out.stride(0), 0,
                      G=block, BLOCKS=True, CANDIDATES=False, B=PICK // 2, num_warps=4)
    return out


def topk(s: torch.Tensor, ratio: int, pos: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor, *,
         row0: int = 0, cand: torch.Tensor | None = None, cand_n: torch.Tensor | None = None,
         block: int = 8) -> torch.Tensor:
    """Each row's best visible entries (M:577-580), ascending into out; with ``cand``, of its candidates."""

    R = _check(s, out, out_n)
    if R:
        _select[(R,)](s, out, out_n, pos, cand if cand is not None else s, cand_n if cand is not None else s, row0,
                      ratio, out.shape[1], s.stride(0), out.stride(0), cand.stride(0) if cand is not None else 0,
                      G=block, BLOCKS=False, CANDIDATES=cand is not None, B=PICK, num_warps=4)
    return out


def select(cfg: Config, qI: torch.Tensor, wI: torch.Tensor, keys: torch.Tensor, ratio: int, pos_dev: torch.Tensor,
           buf: torch.Tensor, lists: torch.Tensor, list_n: torch.Tensor, *, pos: int | None = None,
           source: tuple[torch.Tensor, torch.Tensor] | None = None,
           within: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Scores, candidate blocks and top-k lists for rows at pos_dev + r; ``pos`` marks a prompt chunk."""

    R = qI.shape[0]
    step = R if pos is None else min(score_rows((pos + R) // ratio), buf.shape[0])
    blk = cfg.candidate_block_size
    for a in range(0, R, max(step, 1)):
        n = min(step, R - a)
        part = buf[:n]
        cand = (within[0][a:a + n], within[1][a:a + n]) if within is not None else (None, None)
        scores(qI[a:a + n], wI[a:a + n], keys, ratio, pos_dev, part, row0=a, cand=cand[0], cand_n=cand[1],
               block=blk)
        if source is not None:
            candidates(part, ratio, pos_dev, source[0][a:a + n], source[1][a:a + n], block=blk, row0=a)
        topk(part, ratio, pos_dev, lists[a:a + n], list_n[a:a + n], row0=a, cand=cand[0], cand_n=cand[1], block=blk)
    return lists
