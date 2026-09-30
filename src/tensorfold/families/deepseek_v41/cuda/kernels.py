"""DeepSeek-V4.1 decode/prefill kernels: RMSNorm, table RoPE, and MQA attention over compressed + window keys.

Attention: one 512-wide latent per key is both key and value for every head (MQA). A row attends to the first
``(p + 1) // ratio`` entries of its kv source's compressed cache and to its own window slots p - 127 .. p, plus
a per-head sink logit in the denominator. Heads fill the tile rows (16 a program), so a key block is loaded once
per 16 heads; chunks of keys run as separate programs and a merge adds the sink and normalizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_TILE = 16
KEY_TILE = 64
CHUNK = 256               # keys a chunk program takes


@triton.jit
def _rmsnorm(X, W, OUT, x_stride, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    ok = o < N
    x = tl.load(X + r * x_stride + o, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + o, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * N + o, (x * rinv * w).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 RMSNorm of each row of x [R, N] (any float dtype, fp32 math)."""

    R, N = x.shape
    out = torch.empty((R, N), dtype=torch.bfloat16, device=x.device)
    _rmsnorm[(R,)](x, w, out, x.stride(0), eps, N=N, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return out


@triton.jit
def _rope(X, POS, COS, SIN, OUT, heads, D: tl.constexpr, HALF: tl.constexpr, SIGN: tl.constexpr):
    """GPT-J rotation of the last 2*HALF dims of each [D] head vector at the row's position; others copied."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    base = (r * heads + h) * D
    d = tl.arange(0, D)
    x = tl.load(X + base + d).to(tl.float32)
    tl.store(OUT + base + d, x.to(OUT.dtype.element_ty), mask=d < D - 2 * HALF)
    p = tl.load(POS + r)
    i = tl.arange(0, HALF)
    c = tl.load(COS + p * HALF + i)
    s = tl.load(SIN + p * HALF + i) * SIGN
    e = tl.load(X + base + D - 2 * HALF + 2 * i).to(tl.float32)
    od = tl.load(X + base + D - 2 * HALF + 2 * i + 1).to(tl.float32)
    tl.store(OUT + base + D - 2 * HALF + 2 * i, (e * c - od * s).to(OUT.dtype.element_ty))
    tl.store(OUT + base + D - 2 * HALF + 2 * i + 1, (od * c + e * s).to(OUT.dtype.element_ty))


def rope(x: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False,
         out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [R, D] or [R, H, D]; tables [max_pos, 32]; returns the rotated copy (fp32 math)."""

    shape = x.shape
    x3 = x.reshape(shape[0], -1, shape[-1]).contiguous()
    out = torch.empty(x3.shape, dtype=out_dtype or x.dtype, device=x.device)
    _rope[(x3.shape[0], x3.shape[1])](x3, pos, cos, sin, out, x3.shape[1], D=shape[-1], HALF=cos.shape[1],
                                      SIGN=-1.0 if inverse else 1.0, num_warps=4)
    return out.reshape(shape)


def rope_tables(freqs: torch.Tensor, max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    ang = torch.arange(max_pos, dtype=torch.float64, device=freqs.device)[:, None] * freqs.double()[None, :]
    return ang.cos().float().contiguous(), ang.sin().float().contiguous()


@triton.jit
def _mqa_chunks(Q, COMP, SWA, POS, PO, PM, PL, n_comp_buf, ratio, H: tl.constexpr, D: tl.constexpr,
                W: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, NCH: tl.constexpr):
    r = tl.program_id(0)
    hg = tl.program_id(1)
    c = tl.program_id(2)
    p = tl.load(POS + r)
    n_vis = tl.where(ratio > 0, (p + 1) // tl.maximum(ratio, 1), 0)
    hh = hg * 16 + tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    total = n_comp_buf + W
    for t in range(CH // 64):
        k = c * CH + t * 64 + tl.arange(0, 64)
        is_comp = k < n_comp_buf
        slot = p - (W - 1) + (k - n_comp_buf)                       # window position of a window key
        ok_c = is_comp & (k < n_vis)
        ok_w = (k >= n_comp_buf) & (k < total) & (slot >= 0)
        kc = tl.load(COMP + k[:, None].to(tl.int64) * D + d[None, :], mask=ok_c[:, None], other=0.0)
        kw = tl.load(SWA + tl.maximum(slot, 0)[:, None].to(tl.int64) * D + d[None, :], mask=ok_w[:, None], other=0.0)
        kk = tl.where(ok_c[:, None], kc.to(tl.bfloat16), kw.to(tl.bfloat16))
        ok = ok_c | ok_w
        scores = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        scores = tl.where(ok[None, :], scores, float("-inf"))
        tile_m = tl.max(scores, 1)
        active = tile_m != float("-inf")
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        pr = tl.where(ok[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kk)
        l = l * alpha + tl.sum(pr, 1)
        m = next_m
    base = (r * NCH + c) * H + hh
    tl.store(PO + base[:, None] * D + d[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _mqa_merge(PO, PM, PL, SINK, OUT, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    m = tl.load(SINK + h)                   # the sink is a logit with a zero value vector
    l = 1.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        base = (r * NCH + c) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * D + d)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.exp(m - next_m)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * D + d, o / l)


class AttnBuffers:
    def __init__(self, rows: int, heads: int, dims: int, max_keys: int, device="cuda") -> None:
        nch = triton.cdiv(max_keys, CHUNK)
        self.po = torch.empty((rows * nch * heads * dims,), dtype=torch.float32, device=device)
        self.pm = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.max_keys = max_keys


def mqa(q: torch.Tensor, comp: torch.Tensor | None, n_comp_buf: int, ratio: int, swa: torch.Tensor,
        pos: torch.Tensor, sink: torch.Tensor, window: int, buf: AttnBuffers, scale: float) -> torch.Tensor:
    """q [R, H, D] (RoPE'd) -> o fp32 [R, H, D] over the first ``n_comp_buf`` slots of ``comp`` and the window."""

    R, H, D = q.shape
    assert H % HEAD_TILE == 0
    keys = n_comp_buf + window
    nch = triton.cdiv(keys, CHUNK)
    assert keys <= buf.max_keys
    out = torch.empty((R, H, D), dtype=torch.float32, device=q.device)
    comp_t = comp if comp is not None else swa
    _mqa_chunks[(R, H // HEAD_TILE, nch)](q.contiguous(), comp_t, swa, pos, buf.po, buf.pm, buf.pl, n_comp_buf,
                                          ratio, H=H, D=D, W=window, CH=CHUNK, SCALE=scale, NCH=nch,
                                          num_warps=8, num_stages=1)
    _mqa_merge[(R, H)](buf.po, buf.pm, buf.pl, sink, out, H=H, D=D, NCH=nch, num_warps=4)
    return out
