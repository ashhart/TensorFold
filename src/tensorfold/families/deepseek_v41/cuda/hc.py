"""Fused hyper-connection kernels for DeepSeek-V4.1 (4 streams of 5120): one pre and one post launch per sublayer.

``pre``: the 24 mixes over the RMS-normalized 20,480-wide streams (split into fixed K blocks, summed in order),
sigmoid pre/post, the 20-step Sinkhorn comb, then the collapse with the carried-in pre-mix (V4.1's delayed
pre) and the sublayer's RMSNorm. ``post``: new streams post_j * b + sum_i comb[i, j] * X_i. Block orders depend only
on the shapes, so a row's bits never depend on how many rows share the call.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

NB = 16          # K blocks of the 24-mix projection (20,480 / 16 = 1,280 columns each)
SUB = 128        # columns a partial step takes
CHUNK = 1024     # hidden columns a finish/post step takes (5,120 = 5 chunks)
FUSED_NB, FUSED_SUBK, FUSED_WARPS = 32, 128, 8   # post_pre's mix K blocks (prompt only), step, warps
# decode/verify rows at most (TF_DSV41_DECODE_ROWS): above, a prompt chunk (blocked mix partials); up to it, per row
PROMPT_ROWS = int(__import__("os").environ.get("TF_DSV41_DECODE_ROWS") or 32)


@triton.jit
def _pre_partial(X, FN, PART, WIDE: tl.constexpr, NBLK: tl.constexpr, SUBK: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    KB: tl.constexpr = WIDE // NBLK
    m = tl.arange(0, 32)
    k = tl.arange(0, SUBK)
    acc = tl.zeros((32,), dtype=tl.float32)
    ss = tl.zeros((SUBK,), dtype=tl.float32)
    for t in range(KB // SUBK):
        base = b * KB + t * SUBK
        x = tl.load(X + r * WIDE + base + k).to(tl.float32)
        w = tl.load(FN + m[:, None] * WIDE + base + k[None, :], mask=m[:, None] < 24, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
        ss += x * x
    tl.store(PART + (r * NBLK + b) * 32 + m, acc, mask=m < 24)
    tl.store(PART + (r * NBLK + b) * 32 + 24, tl.sum(ss, axis=0))


@triton.jit
def _pre_partial_rows(X, FN, PART, R, WIDE: tl.constexpr, NBLK: tl.constexpr, SUBK: tl.constexpr, BR: tl.constexpr):
    """Prompt chunks: BR rows share each block of the mix matrix (a tensor-core dot); same partial layout."""

    rb = tl.program_id(0)
    b = tl.program_id(1)
    KB: tl.constexpr = WIDE // NBLK
    r = rb * BR + tl.arange(0, BR)
    m = tl.arange(0, 32)
    k = tl.arange(0, SUBK)
    acc = tl.zeros((BR, 32), dtype=tl.float32)
    ss = tl.zeros((BR,), dtype=tl.float32)
    for t in range(KB // SUBK):
        base = b * KB + t * SUBK
        x = tl.load(X + r[:, None] * WIDE + base + k[None, :], mask=(r < R)[:, None], other=0.0).to(tl.float32)
        w = tl.load(FN + m[:, None] * WIDE + base + k[None, :], mask=m[:, None] < 24, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        ss += tl.sum(x * x, axis=1)
    tl.store(PART + (r[:, None] * NBLK + b) * 32 + m[None, :], acc, mask=(r < R)[:, None] & (m[None, :] < 24))
    tl.store(PART + (r * NBLK + b) * 32 + 24, ss, mask=r < R)


@triton.jit
def _pre_finish(X, PART, BASE, SCALE, PRE_IN, NW, OUT, PRE, POST, COMB, eps_norm, hc_eps,
                D: tl.constexpr, NBLK: tl.constexpr, ITERS: tl.constexpr, CH: tl.constexpr,
                COLLAPSED: tl.constexpr = False):
    r = tl.program_id(0)
    m = tl.arange(0, 32)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in range(NBLK):
        mix += tl.load(PART + (r * NBLK + b) * 32 + m)
        ss += tl.load(PART + (r * NBLK + b) * 32 + 24)
    mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps_norm))
    s_pre = tl.load(SCALE + 0)
    s_post = tl.load(SCALE + 1)
    s_comb = tl.load(SCALE + 2)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    sv = tl.arange(0, 4)
    pre_logit = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s_pre + base)[None, :], 0.0), axis=1)
    post_logit = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s_post + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_logit)) + hc_eps
    post = 2.0 / (1.0 + tl.exp(-post_logit))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s_comb + base)[None, None, :], 0.0), axis=2)
    ce = tl.exp(cl - tl.max(cl, axis=1)[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE + r * 4 + sv, pre)
    tl.store(POST + r * 4 + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    # collapse with the carried-in pre-mix, round to bf16, RMSNorm with the sublayer's weight
    p0 = tl.load(PRE_IN + r * 4 + 0)
    p1 = tl.load(PRE_IN + r * 4 + 1)
    p2 = tl.load(PRE_IN + r * 4 + 2)
    p3 = tl.load(PRE_IN + r * 4 + 3)
    d = tl.arange(0, CH)
    sq = 0.0
    for c in range(D // CH):
        o = c * CH + d
        if COLLAPSED:
            v = tl.load(X + r * D + o).to(tl.float32)
        else:
            x0 = tl.load(X + r * (4 * D) + o).to(tl.float32)
            x1 = tl.load(X + r * (4 * D) + D + o).to(tl.float32)
            x2 = tl.load(X + r * (4 * D) + 2 * D + o).to(tl.float32)
            x3 = tl.load(X + r * (4 * D) + 3 * D + o).to(tl.float32)
            v = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16).to(tl.float32)
        sq += tl.sum(v * v, axis=0)
    rinv = 1.0 / tl.sqrt(sq / D + eps_norm)
    for c in range(D // CH):
        o = c * CH + d
        if COLLAPSED:
            v = tl.load(X + r * D + o).to(tl.float32)
        else:
            x0 = tl.load(X + r * (4 * D) + o).to(tl.float32)
            x1 = tl.load(X + r * (4 * D) + D + o).to(tl.float32)
            x2 = tl.load(X + r * (4 * D) + 2 * D + o).to(tl.float32)
            x3 = tl.load(X + r * (4 * D) + 3 * D + o).to(tl.float32)
            v = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16).to(tl.float32)
        w = tl.load(NW + o).to(tl.float32)
        tl.store(OUT + r * D + o, (v * rinv * w).to(tl.bfloat16))


@triton.jit
def _post(B, X, POST, COMB, Y, parts, R, split, D: tl.constexpr, CH: tl.constexpr):
    """B holds ``parts`` rank partials (fp32 or bf16, rank order) or one bf16 branch (parts 0), in row blocks of
    ``split`` rows, each [parts, rows, D] (split R: one block)."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    o = c * CH + tl.arange(0, CH)
    if parts == 0:
        b = tl.load(B + r * D + o).to(tl.float32)
    else:
        blk = r // split                                   # row blocks of ``split`` rows (the last may be short)
        base = blk * parts * split * D + (r - blk * split) * D
        stride = tl.minimum(split, R - blk * split) * D
        b = tl.load(B + base + o).to(tl.float32)
        for k in range(1, parts):
            b = b + tl.load(B + base + k * stride + o).to(tl.float32)
        b = b.to(tl.bfloat16).to(tl.float32)
    x0 = tl.load(X + r * (4 * D) + o).to(tl.float32)
    x1 = tl.load(X + r * (4 * D) + D + o).to(tl.float32)
    x2 = tl.load(X + r * (4 * D) + 2 * D + o).to(tl.float32)
    x3 = tl.load(X + r * (4 * D) + 3 * D + o).to(tl.float32)
    for j in tl.static_range(4):
        c0 = tl.load(COMB + r * 16 + 0 * 4 + j)
        c1 = tl.load(COMB + r * 16 + 1 * 4 + j)
        c2 = tl.load(COMB + r * 16 + 2 * 4 + j)
        c3 = tl.load(COMB + r * 16 + 3 * 4 + j)
        pj = tl.load(POST + r * 4 + j)
        v = pj * b + (c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3)
        tl.store(Y + r * (4 * D) + j * D + o, v.to(tl.bfloat16))


@triton.jit
def _mix_stream(Y, FN, COMB, xo, r, ok, j: tl.constexpr, pj, b, x0, x1, x2, x3, fcol, m, k, D: tl.constexpr, acc, ss):
    """One new stream j of a tile (as _post), stored bf16, and its mix partial and square-sum steps."""

    c0 = tl.load(COMB + r * 16 + 0 * 4 + j, mask=ok, other=0.0)
    c1 = tl.load(COMB + r * 16 + 1 * 4 + j, mask=ok, other=0.0)
    c2 = tl.load(COMB + r * 16 + 2 * 4 + j, mask=ok, other=0.0)
    c3 = tl.load(COMB + r * 16 + 3 * 4 + j, mask=ok, other=0.0)
    y = pj[:, None] * b + (c0[:, None] * x0 + c1[:, None] * x1 + c2[:, None] * x2 + c3[:, None] * x3)
    yb = y.to(tl.bfloat16)
    tl.store(Y + xo + j * D, yb, mask=ok[:, None])
    yf = yb.to(tl.float32)
    w = tl.load(FN + m[:, None] * (4 * D) + (j * D + fcol) + k[None, :], mask=m[:, None] < 24, other=0.0)
    return yf, tl.dot(yf, tl.trans(w), acc), ss + tl.sum(yf * yf, axis=1)


@triton.jit
def _post_mix_rows(B, X, POST, COMB, Y, PRE_IN, V, FN, PART, parts, R, split, D: tl.constexpr, NBLK: tl.constexpr,
                   SUBK: tl.constexpr, BR: tl.constexpr):
    """Prompt chunks: _post for BR rows and one D chunk (all 4 streams), then on the new streams the mix partials
    of their 4 K blocks (as _pre_partial_rows) and the collapse with the carried-in pre-mix (as _pre_finish,
    bf16, into V); the streams are not read back."""

    rb = tl.program_id(0)
    c = tl.program_id(1)
    KB: tl.constexpr = 4 * D // NBLK                   # columns a K block (within one stream)
    CPS: tl.constexpr = D // KB                         # K blocks a stream
    r = rb * BR + tl.arange(0, BR)
    ok = r < R
    m = tl.arange(0, 32)
    k = tl.arange(0, SUBK)
    blk = r // split                                   # row blocks of ``split`` rows (the last may be short)
    base_b = blk * parts * split * D + (r - blk * split) * D
    stride_b = tl.minimum(split, R - blk * split) * D
    p0 = tl.load(POST + r * 4 + 0, mask=ok, other=0.0)
    p1 = tl.load(POST + r * 4 + 1, mask=ok, other=0.0)
    p2 = tl.load(POST + r * 4 + 2, mask=ok, other=0.0)
    p3 = tl.load(POST + r * 4 + 3, mask=ok, other=0.0)
    q0 = tl.load(PRE_IN + r * 4 + 0, mask=ok, other=0.0)
    q1 = tl.load(PRE_IN + r * 4 + 1, mask=ok, other=0.0)
    q2 = tl.load(PRE_IN + r * 4 + 2, mask=ok, other=0.0)
    q3 = tl.load(PRE_IN + r * 4 + 3, mask=ok, other=0.0)
    a0 = tl.zeros((BR, 32), dtype=tl.float32)
    a1 = tl.zeros((BR, 32), dtype=tl.float32)
    a2 = tl.zeros((BR, 32), dtype=tl.float32)
    a3 = tl.zeros((BR, 32), dtype=tl.float32)
    s0 = tl.zeros((BR,), dtype=tl.float32)
    s1 = tl.zeros((BR,), dtype=tl.float32)
    s2 = tl.zeros((BR,), dtype=tl.float32)
    s3 = tl.zeros((BR,), dtype=tl.float32)
    for t in range(KB // SUBK):
        fcol = c * KB + t * SUBK
        d = fcol + k
        b = tl.load(B + base_b[:, None] + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        for q in range(1, parts):
            b = b + tl.load(B + base_b[:, None] + q * stride_b[:, None] + d[None, :], mask=ok[:, None],
                            other=0.0).to(tl.float32)
        b = b.to(tl.bfloat16).to(tl.float32)
        xo = r[:, None] * (4 * D) + d[None, :]
        x0 = tl.load(X + xo, mask=ok[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(X + xo + D, mask=ok[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(X + xo + 2 * D, mask=ok[:, None], other=0.0).to(tl.float32)
        x3 = tl.load(X + xo + 3 * D, mask=ok[:, None], other=0.0).to(tl.float32)
        y0, a0, s0 = _mix_stream(Y, FN, COMB, xo, r, ok, 0, p0, b, x0, x1, x2, x3, fcol, m, k, D, a0, s0)
        y1, a1, s1 = _mix_stream(Y, FN, COMB, xo, r, ok, 1, p1, b, x0, x1, x2, x3, fcol, m, k, D, a1, s1)
        y2, a2, s2 = _mix_stream(Y, FN, COMB, xo, r, ok, 2, p2, b, x0, x1, x2, x3, fcol, m, k, D, a2, s2)
        y3, a3, s3 = _mix_stream(Y, FN, COMB, xo, r, ok, 3, p3, b, x0, x1, x2, x3, fcol, m, k, D, a3, s3)
        v = q0[:, None] * y0 + q1[:, None] * y1 + q2[:, None] * y2 + q3[:, None] * y3
        tl.store(V + r[:, None] * D + d[None, :], v.to(tl.bfloat16), mask=ok[:, None])
    pm = ok[:, None] & (m[None, :] < 24)
    tl.store(PART + (r[:, None] * NBLK + 0 * CPS + c) * 32 + m[None, :], a0, mask=pm)
    tl.store(PART + (r[:, None] * NBLK + 1 * CPS + c) * 32 + m[None, :], a1, mask=pm)
    tl.store(PART + (r[:, None] * NBLK + 2 * CPS + c) * 32 + m[None, :], a2, mask=pm)
    tl.store(PART + (r[:, None] * NBLK + 3 * CPS + c) * 32 + m[None, :], a3, mask=pm)
    tl.store(PART + (r * NBLK + 0 * CPS + c) * 32 + 24, s0, mask=ok)
    tl.store(PART + (r * NBLK + 1 * CPS + c) * 32 + 24, s1, mask=ok)
    tl.store(PART + (r * NBLK + 2 * CPS + c) * 32 + 24, s2, mask=ok)
    tl.store(PART + (r * NBLK + 3 * CPS + c) * 32 + 24, s3, mask=ok)

class HCBuffers:
    """Scratch for up to ``rows`` rows (graph-stable addresses)."""

    def __init__(self, rows: int, dims: int, device="cuda") -> None:
        self.part = torch.empty((rows * NB * 32,), dtype=torch.float32, device=device)
        self.rows, self.dims = rows, dims


def pre(X: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, pre_in: torch.Tensor,
        norm_w: torch.Tensor, buf: HCBuffers, eps: float, hc_eps: float, iters: int,
        out: torch.Tensor | None = None):
    """X bf16 [R, 4, D] -> (post fp32 [R,4], comb fp32 [R,4,4], x_in bf16 [R,D], pre fp32 [R,4])."""

    R, S, D = X.shape
    assert S == 4 and D % CHUNK == 0 and (S * D) % (NB * SUB) == 0 and X.is_contiguous()
    dev = X.device
    post = torch.empty((R, 4), dtype=torch.float32, device=dev)
    comb = torch.empty((R, 4, 4), dtype=torch.float32, device=dev)
    pre_out = torch.empty((R, 4), dtype=torch.float32, device=dev)
    x_in = out if out is not None else torch.empty((R, D), dtype=torch.bfloat16, device=dev)
    if R > PROMPT_ROWS:                     # prompt chunks: rows share the mix matrix loads (TF32 dot, like vLLM)
        _pre_partial_rows[(triton.cdiv(R, 16), NB)](X, fn, buf.part, R, WIDE=S * D, NBLK=NB, SUBK=64, BR=16,
                                                   num_warps=4)
    else:                                   # decode and verify windows: the row-invariant per-row sums
        _pre_partial[(R, NB)](X, fn, buf.part, WIDE=S * D, NBLK=NB, SUBK=SUB, num_warps=4)
    _pre_finish[(R,)](X, buf.part, base, scale, pre_in.contiguous(), norm_w, x_in, pre_out, post, comb, eps, hc_eps,
                      D=D, NBLK=NB, ITERS=iters, CH=CHUNK, num_warps=8)
    return post, comb, x_in, pre_out


class SplitPartials(NamedTuple):
    """Rank partials gathered in row blocks of ``split`` rows (the last may be short), each [parts, rows, D]."""

    buf: torch.Tensor
    parts: int
    split: int


def post(b, X: torch.Tensor, post_w: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """New streams; ``b`` is the bf16 branch [R, D] or the ranks' fp32 partials [world, R, D] (summed in rank order,
    rounded to bf16 like the branch)."""

    R, _, D = X.shape
    Y = torch.empty_like(X)
    if isinstance(b, SplitPartials):
        parts, split, b = b.parts, b.split, b.buf
    else:
        b = b.contiguous()
        parts, split = (b.shape[0] if b.dim() == 3 else 0), R
    _post[(R, D // CHUNK)](b, X, post_w, comb, Y, parts, R, split, D=D, CH=CHUNK, num_warps=4)
    return Y


# decode / verify rows through the fused post + pre too (TF_FUSE_HC_DECODE=1; its own arithmetic, rows independent)
FUSE_DECODE = __import__("os").environ.get("TF_FUSE_HC_DECODE") == "1"


def post_pre(b, X: torch.Tensor, post_w: torch.Tensor, comb: torch.Tensor, fn: torch.Tensor, base: torch.Tensor,
             scale: torch.Tensor, pre_in: torch.Tensor, norm_w: torch.Tensor, buf: HCBuffers, eps: float,
             hc_eps: float, iters: int):
    """Prompt chunks: post() then pre() of the next sublayer in two launches (the new streams written once, never
    read back): (Y, (post, comb, x_in, pre))."""

    R, S, D = X.shape
    assert (R > PROMPT_ROWS or FUSE_DECODE) and S == 4 and X.is_contiguous() and (S * D) % NB == 0
    dev = X.device
    Y = torch.empty_like(X)
    if isinstance(b, SplitPartials):
        parts, split, b = b.parts, b.split, b.buf
    else:
        b = b.contiguous()
        parts, split = (b.shape[0] if b.dim() == 3 else 0), R
    assert parts > 0, "post_pre takes rank partials"
    x_in = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
    pre_in = pre_in.contiguous()
    nb = FUSED_NB
    if buf.part.numel() < R * nb * 32:
        buf.part = torch.empty((max(buf.rows, R) * nb * 32,), dtype=torch.float32, device=dev)
    _post_mix_rows[(triton.cdiv(R, 16), D // ((S * D) // nb))](
        b, X, post_w, comb, Y, pre_in, x_in, fn, buf.part, parts, R, split, D=D, NBLK=nb, SUBK=FUSED_SUBK, BR=16,
        num_warps=FUSED_WARPS)
    post = torch.empty((R, 4), dtype=torch.float32, device=dev)
    comb_out = torch.empty((R, 4, 4), dtype=torch.float32, device=dev)
    pre_out = torch.empty((R, 4), dtype=torch.float32, device=dev)
    _pre_finish[(R,)](x_in, buf.part, base, scale, pre_in, norm_w, x_in, pre_out, post, comb_out, eps, hc_eps,
                      D=D, NBLK=nb, ITERS=iters, CH=CHUNK, COLLAPSED=True, num_warps=8)
    return Y, (post, comb_out, x_in, pre_out)

