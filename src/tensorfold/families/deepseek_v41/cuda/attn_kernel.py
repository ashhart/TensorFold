"""Sparse attention, a row a program: its window, then its list, with a sink, as model.py's kernel (K:355-387)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

SCALE = 512 ** -0.5     # softmax scale of the 512-wide head
CHUNK = 128             # decode: list entries one partial covers, merged in chunk order
HB = 16                 # decode: heads a program
KT = 32                 # decode: entries a tile inside a chunk
HB_PROMPT = 32          # prompt: heads a program (a rank's heads, each entry read once)
KT_PROMPT = 64          # prompt: entries a tile, in list order


@triton.jit
def _entries(RING, KVW, EXTRA, LISTS, r, n, total, nw, lo, pos, kv_stride, e_stride, l_stride, k,
             WIN: tl.constexpr, LW: tl.constexpr):
    """bf16 [KT, LW] rows of list entries ``n``: window positions from the ring or kvw, then extra rows."""

    ok = n < total
    win = ok & (n < nw)
    w = lo + n
    fresh = win & (w >= pos)
    old = win & (w < pos)
    ext = ok & (n >= nw)
    slot = (w % WIN).to(tl.int64)
    row = tl.where(fresh, w - pos, 0).to(tl.int64)
    j = tl.load(LISTS + r * l_stride + (n - nw), mask=ext, other=0).to(tl.int64)
    a = tl.load(RING + slot[:, None] * LW + k[None, :], mask=old[:, None], other=0.0)
    b = tl.load(KVW + row[:, None] * kv_stride + k[None, :], mask=fresh[:, None], other=0.0)
    c = tl.load(EXTRA + j[:, None] * e_stride + k[None, :], mask=ext[:, None], other=0.0)
    return tl.where(old[:, None], a, tl.where(fresh[:, None], b, c)), ok


@triton.jit
def _scores(q, kv, ok, SCALE: tl.constexpr):
    """fp32 scale * (q . kv) from bf16 operands, -inf past the list."""

    return tl.where(ok[None, :], tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE, float("-inf"))


@triton.jit
def _tile(q, kv, ok, m, l, o, SCALE: tl.constexpr):
    """Online softmax step over a tile holding at least one entry: fp32 max, P and sum, bf16 P for P.V."""

    s = _scores(q, kv, ok, SCALE)
    nm = tl.maximum(m, tl.max(s, 1))
    alpha = libdevice.exp(m - nm)
    p = libdevice.exp(s - nm[:, None])
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return nm, l, o


@triton.jit
def _bounds(POS, ANCH, CNT, r, WIN: tl.constexpr, HAS_EXTRA: tl.constexpr):
    """(forward start, first window position, window entries, list entries) of row r; anchor < 0: no window."""

    pos = tl.load(POS)
    a = tl.load(ANCH + r)
    lo = tl.maximum(a - (WIN - 1), 0)
    nw = tl.where(a >= 0, a - lo + 1, 0)
    total = nw
    if HAS_EXTRA:
        total += tl.load(CNT + r)
    return pos, lo, nw, total


@triton.jit
def _chunks(Q, RING, KVW, EXTRA, LISTS, CNT, POS, ANCH, PART, q_row, q_head, kv_stride, e_stride, l_stride,
            p_row, p_head, p_chunk, H: tl.constexpr, LW: tl.constexpr, WIN: tl.constexpr, CH: tl.constexpr,
            HBT: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr, HAS_EXTRA: tl.constexpr):
    """Program (row, head block, chunk): partial (acc, max, sum) of HB heads over list entries [c CH, (c+1) CH)."""

    r = tl.program_id(0).to(tl.int64)
    hh = tl.program_id(1) * HBT + tl.arange(0, HBT)
    c = tl.program_id(2)
    hok = hh < H
    k = tl.arange(0, LW)
    pos, lo, nw, total = _bounds(POS, ANCH, CNT, r, WIN, HAS_EXTRA)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    if start < total:
        q = tl.load(Q + r * q_row + hh[:, None] * q_head + k[None, :], mask=hok[:, None], other=0.0)
        for t in range(0, tl.minimum(CH, total - start), KTT):
            kv, ok = _entries(RING, KVW, EXTRA, LISTS, r, start + t + tl.arange(0, KTT), total, nw, lo, pos,
                              kv_stride, e_stride, l_stride, k, WIN, LW)
            m, l, o = _tile(q, kv, ok, m, l, o, SCALE)
    base = PART + r * p_row + hh * p_head + c * p_chunk
    tl.store(base[:, None] + k[None, :], o, mask=hok[:, None])
    tl.store(base + LW, m, mask=hok)
    tl.store(base + LW + 1, l, mask=hok)


@triton.jit
def _merge(PART, SINK, OUT, o_row, o_head, p_row, p_head, p_chunk, NCH, LW: tl.constexpr):
    """Program (row, head): the chunk partials in chunk order, the sink in the denominator, bf16 out (zeros if empty)."""

    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = PART + r * p_row + h * p_head + c * p_chunk
        cl = tl.load(base + LW + 1)
        if cl > 0.0:
            cm = tl.load(base + LW)
            nm = tl.maximum(m, cm)
            a = libdevice.exp(m - nm)
            b = libdevice.exp(cm - nm)
            o = o * a + tl.load(base + k) * b
            l = l * a + cl * b
            m = nm
    if l > 0.0:
        o = tl.math.div_rn(o, l + libdevice.exp(tl.load(SINK + h) - m))
    tl.store(OUT + r * o_row + h * o_head + k, o.to(tl.bfloat16))


@triton.jit
def _prompt(Q, RING, KVW, EXTRA, LISTS, CNT, POS, ANCH, SINK, OUT, q_row, q_head, kv_stride, e_stride, l_stride,
            o_row, o_head, H: tl.constexpr, LW: tl.constexpr, WIN: tl.constexpr, HBT: tl.constexpr,
            KTT: tl.constexpr, SCALE: tl.constexpr, HAS_EXTRA: tl.constexpr):
    """Program (row, head block): HB_PROMPT heads over the whole list in KT_PROMPT-entry tiles, no partials."""

    r = tl.program_id(0).to(tl.int64)
    hh = tl.program_id(1) * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    pos, lo, nw, total = _bounds(POS, ANCH, CNT, r, WIN, HAS_EXTRA)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    q = tl.load(Q + r * q_row + hh[:, None] * q_head + k[None, :], mask=hok[:, None], other=0.0)
    # a tile is read as two halves, the second time for P.V: shared memory holds the heads and half a tile
    HALF: tl.constexpr = KTT // 2
    for t in range(0, total, KTT):
        n0 = t + tl.arange(0, HALF)
        n1 = n0 + HALF
        kv, ok0 = _entries(RING, KVW, EXTRA, LISTS, r, n0, total, nw, lo, pos, kv_stride, e_stride, l_stride, k,
                           WIN, LW)
        s0 = _scores(q, kv, ok0, SCALE)
        kv, ok1 = _entries(RING, KVW, EXTRA, LISTS, r, n1, total, nw, lo, pos, kv_stride, e_stride, l_stride, k,
                           WIN, LW)
        s1 = _scores(q, kv, ok1, SCALE)
        nm = tl.maximum(m, tl.maximum(tl.max(s0, 1), tl.max(s1, 1)))
        alpha = libdevice.exp(m - nm)
        p0 = libdevice.exp(s0 - nm[:, None])
        p1 = libdevice.exp(s1 - nm[:, None])
        kv, _ = _entries(RING, KVW, EXTRA, LISTS, r, n0, total, nw, lo, pos, kv_stride, e_stride, l_stride, k,
                         WIN, LW)
        o = tl.dot(p0.to(tl.bfloat16), kv, o * alpha[:, None])
        kv, _ = _entries(RING, KVW, EXTRA, LISTS, r, n1, total, nw, lo, pos, kv_stride, e_stride, l_stride, k,
                         WIN, LW)
        o = tl.dot(p1.to(tl.bfloat16), kv, o)
        l = l * alpha + (tl.sum(p0, 1) + tl.sum(p1, 1))
        m = nm
    sink = tl.load(SINK + hh, mask=hok, other=0.0)
    den = l + libdevice.exp(sink - m)
    o = tl.where((l > 0.0)[:, None], tl.math.div_rn(o, den[:, None]), 0.0)
    tl.store(OUT + r * o_row + hh[:, None] * o_head + k[None, :], o.to(tl.bfloat16), mask=hok[:, None])


def chunks(width: int, window: int = 128) -> int:
    """Decode partials a row needs: its window and ``width`` list entries in CHUNK-entry pieces."""

    return triton.cdiv(window + width, CHUNK)


def attention(q: torch.Tensor, ring: torch.Tensor, kvw: torch.Tensor | None, pos: torch.Tensor,
              anchors: torch.Tensor, extra: torch.Tensor | None, lists: torch.Tensor | None,
              counts: torch.Tensor | None, sink: torch.Tensor, out: torch.Tensor, *, prompt: bool,
              part: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, H, 512] bf16 -> out bf16 over each row's ring window, this forward's rows and its list."""

    R, H, LW = q.shape
    WIN = ring.shape[0]
    has_extra = extra is not None
    W = lists.shape[1] if has_extra else 0
    if out.shape != q.shape or ring.shape[1] != LW or sink.shape != (H,) or anchors.shape[0] < R:
        raise ValueError(f"attention: q {tuple(q.shape)}, out {tuple(out.shape)}, ring {tuple(ring.shape)}, "
                         f"sink {tuple(sink.shape)}, anchors {tuple(anchors.shape)}")
    if has_extra and (extra.shape[1] != LW or lists.shape[0] < R or counts.shape[0] < R or lists.stride(1) != 1):
        raise ValueError(f"attention: extra {tuple(extra.shape)}, lists {tuple(lists.shape)}, "
                         f"counts {tuple(counts.shape)}")
    for name, t in (("q", q), ("out", out), ("ring", ring), ("kvw", kvw), ("extra", extra)):
        if t is not None and t.stride(-1) != 1:
            raise ValueError(f"attention: {name} rows must be contiguous")
    if not ring.is_contiguous():
        raise ValueError("attention: ring must be contiguous")
    kvw_ = kvw if kvw is not None else ring
    extra_, lists_, counts_ = (extra, lists, counts) if has_extra else (ring, anchors, anchors)
    l_stride = lists.stride(0) if has_extra else 0
    common = {"H": H, "LW": LW, "WIN": WIN, "SCALE": SCALE, "HAS_EXTRA": has_extra}
    if prompt:
        _prompt[(R, triton.cdiv(H, HB_PROMPT))](
            q, ring, kvw_, extra_, lists_, counts_, pos, anchors, sink, out, q.stride(0), q.stride(1),
            kvw_.stride(0), extra_.stride(0), l_stride, out.stride(0), out.stride(1), HBT=HB_PROMPT, KTT=KT_PROMPT,
            num_warps=8, num_stages=1, **common)
        return out
    nch = chunks(W, WIN)
    if part is None or part.dim() != 4 or part.shape[0] < R or part.shape[1] != H or part.shape[2] < nch \
            or part.shape[3] != LW + 2 or part.stride(3) != 1:
        raise ValueError(f"attention: decode needs part [>= {R}, {H}, >= {nch}, {LW + 2}] fp32")
    strides = (part.stride(0), part.stride(1), part.stride(2))
    _chunks[(R, triton.cdiv(H, HB), nch)](
        q, ring, kvw_, extra_, lists_, counts_, pos, anchors, part, q.stride(0), q.stride(1), kvw_.stride(0),
        extra_.stride(0), l_stride, *strides, CH=CHUNK, HBT=HB, KTT=KT, num_warps=8, num_stages=1, **common)
    _merge[(R, H)](part, sink, out, out.stride(0), out.stride(1), *strides, nch, LW=LW, num_warps=4)
    return out
