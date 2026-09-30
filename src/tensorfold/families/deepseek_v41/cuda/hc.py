"""Fused hyper-connection kernels for DeepSeek-V4.1 (4 streams of 5120): one pre and one post launch per sublayer.

``pre``: the 24 mixes over the RMS-normalized 20,480-wide streams (split into fixed K blocks, summed in order),
sigmoid pre/post, the 20-step Sinkhorn comb, then the collapse with the carried-in pre-mix (V4.1's delayed
pre) and the sublayer's RMSNorm. ``post``: new streams post_j * b + sum_i comb[i, j] * X_i. Block orders depend only
on the shapes, so a row's bits never depend on how many rows share the call.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NB = 16          # K blocks of the 24-mix projection (20,480 / 16 = 1,280 columns each)
SUB = 128        # columns a partial step takes
CHUNK = 1024     # hidden columns a finish/post step takes (5,120 = 5 chunks)


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
def _pre_finish(X, PART, BASE, SCALE, PRE_IN, NW, OUT, PRE, POST, COMB, eps_norm, hc_eps,
                D: tl.constexpr, NBLK: tl.constexpr, ITERS: tl.constexpr, CH: tl.constexpr):
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
        x0 = tl.load(X + r * (4 * D) + o).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + o).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + o).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + o).to(tl.float32)
        v = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16).to(tl.float32)
        sq += tl.sum(v * v, axis=0)
    rinv = 1.0 / tl.sqrt(sq / D + eps_norm)
    for c in range(D // CH):
        o = c * CH + d
        x0 = tl.load(X + r * (4 * D) + o).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + o).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + o).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + o).to(tl.float32)
        v = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16).to(tl.float32)
        w = tl.load(NW + o).to(tl.float32)
        tl.store(OUT + r * D + o, (v * rinv * w).to(tl.bfloat16))


@triton.jit
def _post(B, X, POST, COMB, Y, D: tl.constexpr, CH: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    o = c * CH + tl.arange(0, CH)
    b = tl.load(B + r * D + o).to(tl.float32)
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
    _pre_partial[(R, NB)](X, fn, buf.part, WIDE=S * D, NBLK=NB, SUBK=SUB, num_warps=4)
    _pre_finish[(R,)](X, buf.part, base, scale, pre_in.contiguous(), norm_w, x_in, pre_out, post, comb, eps, hc_eps,
                      D=D, NBLK=NB, ITERS=iters, CH=CHUNK, num_warps=8)
    return post, comb, x_in, pre_out


def post(b: torch.Tensor, X: torch.Tensor, post_w: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    R, _, D = X.shape
    Y = torch.empty_like(X)
    _post[(R, D // CHUNK)](b.contiguous(), X, post_w, comb, Y, D=D, CH=CHUNK, num_warps=4)
    return Y
