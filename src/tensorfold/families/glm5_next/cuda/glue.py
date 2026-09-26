"""GLM-5.3-Flash's small kernels on CUDA (Triton): each program reads only its own row, so a row's bits never
depend on the other rows of a window.

Arithmetic follows the Hugging Face definition: fp32 math and one bf16 rounding wherever the model stores a
bf16 tensor. RMSNorm is Llama's: bf16(w * bf16(x * rsqrt(mean(x^2) + eps))). Kernels feeding a 4-bit matmul
also write the fp32 sums of each 64-input group of the bf16 values they store.

Manifold-constrained hyper-connections (4 residual streams of D, kept in bf16):
    hc_pre   mix = rsqrt(mean(X^2) + eps) * (X . fn) (24 dots over the 4D flattened streams, split into 16
             fixed K blocks summed in order); pre = sigmoid(.) + eps, post = 2 sigmoid(.), comb = Sinkhorn(
             softmax(.) + eps, 20 iterations); collapsed = bf16(sum_s pre_s X_s); then the layer's RMSNorm
    hc_post  X_s = bf16(post_s * branch + sum_j comb[j, s] X_j), branch = bf16(the ranks' fp32 partials summed
             in rank order)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# fixed K split of the mixing dots (a function of the shape only; changing it changes bits for every row alike)
HC_BLOCKS = 16


# -- embedding ----------------------------------------------------------------------------------------------
@triton.jit
def _embed(IDS, W, S, B, OUT, D: tl.constexpr, COPIES: tl.constexpr):
    """Row r, group g: 64 values of token IDS[r]'s 4-bit row (MLX layout) as bf16, into COPIES streams."""

    row = tl.program_id(0)
    g = tl.program_id(1)
    tok = tl.load(IDS + row).to(tl.int64)
    words = tl.load(W + tok * (D // 8) + g * 8 + tl.arange(0, 8))
    q = tl.reshape((words[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF, (64,)).to(tl.float32)
    s = tl.load(S + tok * (D // 64) + g).to(tl.float32)
    b = tl.load(B + tok * (D // 64) + g).to(tl.float32)
    v = (q * s + b).to(tl.bfloat16)
    d = g * 64 + tl.arange(0, 64)
    for c in tl.static_range(COPIES):
        tl.store(OUT + row * (COPIES * D) + c * D + d, v)


def embed(ids: torch.Tensor, table: tuple, dims: int, copies: int, out: torch.Tensor) -> torch.Tensor:
    w, s, b = table
    rows = ids.shape[0]
    _embed[(rows, dims // 64)](ids, w, s, b, out, D=dims, COPIES=copies, num_warps=1)
    return out


# -- RMSNorm (Llama convention) -------------------------------------------------------------------------------
@triton.jit
def _rmsnorm(X, x_stride, W, OUT, o_stride, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr, SUMS: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * x_stride + d, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    y = (w * (x * rinv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + r * o_stride + d, y, mask=ok)
    if SUMS:
        g = tl.sum(tl.reshape(tl.where(ok, y.to(tl.float32), 0.0), (BLOCK // 64, 64)), axis=1)
        gi = tl.arange(0, BLOCK // 64)
        tl.store(XS + r * (D // 64) + gi, g, mask=gi < D // 64)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor, xs: torch.Tensor | None = None):
    rows, d = x.shape
    block = triton.next_power_of_2(d)
    if out.stride(-1) != 1:
        raise ValueError("rmsnorm: rows of out must be contiguous")
    _rmsnorm[(rows,)](x, x.stride(0), w, out, out.stride(0), xs if xs is not None else out, eps, D=d, BLOCK=block,
                      SUMS=xs is not None, num_warps=4 if block <= 2048 else 8)
    return out


# -- hyper-connections ----------------------------------------------------------------------------------------
@triton.jit
def _hc_partial(X, FN, PART, WIDE: tl.constexpr, NB: tl.constexpr, SUB: tl.constexpr):
    """Program (r, b): over K block b of row r's flattened streams, the 24 partial dots with fn (rows padded to
    32) and the partial sum of squares, accumulated SUB elements at a time in order. PART[r, b, 0:24] dots,
    PART[r, b, 24] squares."""

    r = tl.program_id(0)
    b = tl.program_id(1)
    KB: tl.constexpr = WIDE // NB
    m = tl.arange(0, 32)
    k = tl.arange(0, SUB)
    acc = tl.zeros((32,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    for t in range(KB // SUB):
        base = b * KB + t * SUB
        x = tl.load(X + r * WIDE + base + k).to(tl.float32)
        w = tl.load(FN + m[:, None] * WIDE + base + k[None, :], mask=m[:, None] < 24, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        ss += x * x
    tl.store(PART + (r * NB + b) * 32 + m, acc, mask=m < 24)
    tl.store(PART + (r * NB + b) * 32 + 24, tl.sum(ss, axis=0))


@triton.jit
def _hc_finish(X, PART, BASE, SCALE, NW, OUT, XS, POST, COMB, eps_norm, hc_eps,
               D: tl.constexpr, S: tl.constexpr, NB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    """Row r: mixes from the partials (blocks in order), pre/post/comb, the collapsed row, and the layer's
    RMSNorm of it (bf16 + 64-group sums). POST[r, s] and COMB[r, i, j] are kept for hc_post."""

    r = tl.program_id(0)
    m = tl.arange(0, 32)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    rinv = 1.0 / tl.sqrt(ss / (S * D) + eps_norm)
    mix = mix * rinv
    s_pre = tl.load(SCALE + 0)
    s_post = tl.load(SCALE + 1)
    s_comb = tl.load(SCALE + 2)
    sv = tl.arange(0, 4)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    # gather the four groups out of the 32-vector: pre = mix[0:4], post = mix[4:8], comb = mix[8:24]
    pre_logit = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s_pre + base)[None, :], 0.0), axis=1)
    post_logit = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s_post + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_logit)) + hc_eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-post_logit)))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj                                         # [4, 4] indices into mix
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s_comb + base)[None, None, :], 0.0), axis=2)
    cmax = tl.max(cl, axis=1)
    ce = tl.exp(cl - cmax[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(POST + r * S + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    d = tl.arange(0, BLOCK)
    p0 = tl.sum(tl.where(sv == 0, pre, 0.0), axis=0)
    p1 = tl.sum(tl.where(sv == 1, pre, 0.0), axis=0)
    p2 = tl.sum(tl.where(sv == 2, pre, 0.0), axis=0)
    p3 = tl.sum(tl.where(sv == 3, pre, 0.0), axis=0)
    x0 = tl.load(X + r * (S * D) + d).to(tl.float32)
    x1 = tl.load(X + r * (S * D) + D + d).to(tl.float32)
    x2 = tl.load(X + r * (S * D) + 2 * D + d).to(tl.float32)
    x3 = tl.load(X + r * (S * D) + 3 * D + d).to(tl.float32)
    c = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
    rinv2 = 1.0 / tl.sqrt(tl.sum(c * c, axis=0) / D + eps_norm)
    w = tl.load(NW + d).to(tl.float32)
    y = (w * (c * rinv2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + r * D + d, y)
    g = tl.sum(tl.reshape(y.to(tl.float32), (BLOCK // 64, 64)), axis=1)
    tl.store(XS + r * (D // 64) + tl.arange(0, BLOCK // 64), g)


def hc_pre(x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, norm_w: torch.Tensor,
           out: torch.Tensor, xs: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, part: torch.Tensor,
           eps: float, hc_eps: float, iters: int) -> None:
    """x [R, S*D] bf16 streams -> out [R, D] (normed collapsed row) + xs, and post [R, S], comb [R, S, S]."""

    rows, wide = x.shape
    d = norm_w.shape[0]
    s = wide // d
    _hc_partial[(rows, HC_BLOCKS)](x, fn, part, WIDE=wide, NB=HC_BLOCKS, SUB=128, num_warps=4)
    _hc_finish[(rows,)](x, part, base, scale, norm_w, out, xs, post, comb, eps, hc_eps, D=d, S=s, NB=HC_BLOCKS,
                        ITERS=iters, BLOCK=d, num_warps=8)


@triton.jit
def _hc_post(X, XOUT, G, POST, COMB, RS, D: tl.constexpr, S: tl.constexpr, WORLD: tl.constexpr,
             BLOCK: tl.constexpr):
    """Program (r, c): X_s[d] = bf16(post_s * branch + (((comb0s X0 + comb1s X1) + comb2s X2) + comb3s X3)),
    branch = bf16(partials summed rank 0 first); G is [WORLD, R, D] fp32 (RS = R * D apart)."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(G + r * D + d)
    for k in tl.static_range(1, WORLD):
        acc = acc + tl.load(G + k * RS + r * D + d)
    branch = acc.to(tl.bfloat16).to(tl.float32)
    x0 = tl.load(X + r * (S * D) + d).to(tl.float32)
    x1 = tl.load(X + r * (S * D) + D + d).to(tl.float32)
    x2 = tl.load(X + r * (S * D) + 2 * D + d).to(tl.float32)
    x3 = tl.load(X + r * (S * D) + 3 * D + d).to(tl.float32)
    for s in tl.static_range(S):
        c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
        c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
        c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
        c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
        ps = tl.load(POST + r * S + s)
        mixed = ((c0 * x0 + c1 * x1) + c2 * x2) + c3 * x3
        v = ps * branch + mixed
        tl.store(XOUT + r * (S * D) + s * D + d, v.to(tl.bfloat16))


def hc_post(x: torch.Tensor, xout: torch.Tensor, gathered: torch.Tensor, post: torch.Tensor,
            comb: torch.Tensor) -> None:
    """gathered: [world, R, D] fp32 partials (contiguous). x and xout may be the same tensor."""

    world, rows, d = gathered.shape
    s = x.shape[1] // d
    block = min(1024, d)
    _hc_post[(rows, d // block)](x, xout, gathered, post, comb, rows * d, D=d, S=s, WORLD=world, BLOCK=block,
                                 num_warps=4)


@triton.jit
def _residual_add(X, XOUT, G, RS, D: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """Plain residual (the MTP block): X = bf16(X + bf16(partials summed rank 0 first))."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(G + r * D + d)
    for k in tl.static_range(1, WORLD):
        acc = acc + tl.load(G + k * RS + r * D + d)
    branch = acc.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + r * D + d).to(tl.float32)
    tl.store(XOUT + r * D + d, (x + branch).to(tl.bfloat16))


def residual_add(x: torch.Tensor, xout: torch.Tensor, gathered: torch.Tensor) -> None:
    world, rows, d = gathered.shape
    block = min(1024, d)
    _residual_add[(rows, d // block)](x, xout, gathered, rows * d, D=d, WORLD=world, BLOCK=block, num_warps=4)


@triton.jit
def _stream_mean(X, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    """bf16((X0 + X1 + X2 + X3) / 4): the final collapse (an unweighted mean of the streams)."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(X + r * (S * D) + d).to(tl.float32)
    for s in tl.static_range(1, S):
        acc = acc + tl.load(X + r * (S * D) + s * D + d).to(tl.float32)
    tl.store(OUT + r * D + d, (acc / S).to(tl.bfloat16))


def stream_mean(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, d = out.shape
    block = min(1024, d)
    _stream_mean[(rows, d // block)](x, out, D=d, S=x.shape[1] // d, BLOCK=block, num_warps=4)
    return out


# -- MLP and MoE ----------------------------------------------------------------------------------------------
@triton.jit
def _swiglu(GU, OUT, XS, LIMIT, W: tl.constexpr, BLOCK: tl.constexpr):
    """Dense MLP: [gate | up] (bf16) -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)) and its 64-group sums."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    g = tl.minimum(tl.load(GU + r * (2 * W) + d).to(tl.float32), LIMIT)
    u = tl.load(GU + r * (2 * W) + W + d).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = ((g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16)
    tl.store(OUT + r * W + d, a)
    s = tl.sum(tl.reshape(a.to(tl.float32), (BLOCK // 64, 64)), axis=1)
    tl.store(XS + r * (W // 64) + cb * (BLOCK // 64) + tl.arange(0, BLOCK // 64), s)


def swiglu(gu: torch.Tensor, out: torch.Tensor, xs: torch.Tensor, limit: float) -> None:
    rows, w = out.shape
    _swiglu[(rows, w // 512)](gu, out, xs, float(limit), W=w, BLOCK=512, num_warps=4)


@triton.jit
def _router(X, W, OUT, M, x_stride, D: tl.constexpr, NE: tl.constexpr, BM: tl.constexpr,
            BLOCK_E: tl.constexpr, BK: tl.constexpr):
    """OUT[m, e] = fp32 x[m] . w[e] (bf16 inputs, tensor cores, K in BK steps in order)."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    e_ok = re < NE
    acc = tl.zeros((BM, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + re[:, None] * D + (k0 + rk)[None, :], mask=e_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rm[:, None] * NE + re[None, :], acc, mask=m_ok[:, None] & e_ok[None, :])


@triton.jit
def _router_part(X, W, PART, M, x_stride, D: tl.constexpr, NE: tl.constexpr, BM: tl.constexpr,
                 BLOCK_E: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr):
    """``_router`` over one of KS equal K slices (in order inside the slice): PART[s, m, e] fp32."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    s = tl.program_id(2)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    e_ok = re < NE
    acc = tl.zeros((BM, BLOCK_E), dtype=tl.float32)
    lo = s * (D // KS)
    for k0 in range(lo, lo + D // KS, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + re[:, None] * D + (k0 + rk)[None, :], mask=e_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(PART + (s * M + rm[:, None]) * NE + re[None, :], acc, mask=m_ok[:, None] & e_ok[None, :])


@triton.jit
def _router_sum(PART, OUT, total, KS: tl.constexpr, BLOCK: tl.constexpr):
    """OUT = the KS slice partials added in slice order."""

    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < total
    acc = tl.load(PART + i, mask=ok, other=0.0)
    for s in tl.static_range(1, KS):
        acc = acc + tl.load(PART + s * total + i, mask=ok, other=0.0)
    tl.store(OUT + i, acc, mask=ok)


# The router's K slices: 8 slices give 72 programs for 288 experts and a fixed-order sum (the single-pass kernel,
# 5 programs, took about 30 us a layer and serves shapes the slices do not divide). Every row gets the same bits
# alone or in a window; the two kernels differ from each other in the last bits, so the choice is part of the model.
ROUTER_KS = 8


def router(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, d = x.shape
    ne = w.shape[0]
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    ks = ROUTER_KS
    if ks <= 1 or d % (ks * 64) or not out.is_contiguous():
        _router[(triton.cdiv(m, bm), triton.cdiv(ne, 64))](x, w, out, m, x.stride(0), D=d, NE=ne, BM=bm,
                                                            BLOCK_E=64, BK=64, num_warps=4, num_stages=3)
        return out
    part = torch.empty((ks, m, ne), dtype=torch.float32, device=x.device)
    _router_part[(triton.cdiv(m, bm), triton.cdiv(ne, 32), ks)](x, w, part, m, x.stride(0), D=d, NE=ne, BM=bm,
                                                                 BLOCK_E=32, BK=64, KS=ks, num_warps=4,
                                                                 num_stages=3)
    total = m * ne
    _router_sum[(triton.cdiv(total, 1024),)](part, out, total, KS=ks, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _topk(L, BIAS, PICK, WTS, scale, NE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
          BLOCK: tl.constexpr, SLOTP: tl.constexpr, NORM: tl.constexpr):
    """Row r: scores = sigmoid(logits); the TOPK largest scores + bias (largest first, the lower id among equals);
    weights = score / (sum + 1e-20) * scale in pick order; then the shared expert (id NE) with weight 1."""

    r = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    ok = ar < NE
    lg = tl.load(L + r * NE + ar, mask=ok, other=0.0)
    score = 1.0 / (1.0 + tl.exp(-lg))
    choice = tl.where(ok, score + tl.load(BIAS + ar, mask=ok, other=0.0), float("-inf"))
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    wts = tl.zeros((SLOTP,), dtype=tl.float32)
    total = 0.0
    for k in tl.static_range(TOPK):
        mx = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == mx, ar, BLOCK), axis=0)
        sk = tl.sum(tl.where(ar == idx, score, 0.0), axis=0)
        picks = tl.where(ak == k, idx, picks)
        wts = tl.where(ak == k, sk, wts)
        total += sk
        choice = tl.where(ar == idx, float("-inf"), choice)
    if NORM:
        wts = wts / (total + 1e-20)
    wts = wts * scale
    picks = tl.where(ak == TOPK, NE, picks)
    wts = tl.where(ak == TOPK, 1.0, wts)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, wts, mask=ak < SLOTS)


@triton.jit
def _group(PICK, UIDS, UCOUNT, UMEM, R, SLOTS: tl.constexpr, MAXU: tl.constexpr, MAXM: tl.constexpr,
           BLOCK: tl.constexpr):
    """One program: the distinct experts of all rows in increasing id order: UIDS[u], UMEM[u][j] = row * 32 +
    slot of the j-th row (in row order) that picked it (-1 after the last), UCOUNT[0] = their number."""

    ar = tl.arange(0, BLOCK)
    counts = tl.zeros((BLOCK,), dtype=tl.int32)
    for r in range(R):
        for k in tl.static_range(SLOTS):
            e = tl.load(PICK + r * SLOTS + k)
            counts += tl.where(ar == e, 1, 0)
    used = counts > 0
    place = tl.cumsum(used.to(tl.int32), axis=0) - 1
    n_used = tl.sum(used.to(tl.int32), axis=0)
    tl.store(UIDS + place, ar, mask=used)
    tl.store(UCOUNT, n_used)
    filled = tl.zeros((BLOCK,), dtype=tl.int32)
    for r in range(R):
        for k in tl.static_range(SLOTS):
            e = tl.load(PICK + r * SLOTS + k)
            hit = ar == e
            tl.store(UMEM + place * MAXM + filled, r * 32 + k, mask=hit)
            filled += tl.where(hit, 1, 0)
    for j in range(MAXM):
        tl.store(UMEM + place * MAXM + j, -1, mask=used & (filled <= j))
    tail = n_used + ar
    for j in range(MAXM):
        tl.store(UMEM + tail * MAXM + j, -1, mask=tail < MAXU)


def select(logits: torch.Tensor, bias: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, ids: torch.Tensor,
           count: torch.Tensor, members: torch.Tensor, top_k: int, experts: int, scale: float, norm: bool) -> None:
    rows = logits.shape[0]
    block = triton.next_power_of_2(experts + 1)
    _topk[(rows,)](logits, bias, pick, wts, float(scale), NE=experts, TOPK=top_k, SLOTS=top_k + 1, BLOCK=block,
                   SLOTP=triton.next_power_of_2(top_k + 1), NORM=norm, num_warps=4)
    _group[(1,)](pick, ids, count, members, rows, SLOTS=top_k + 1, MAXU=ids.shape[0], MAXM=members.shape[1],
                 BLOCK=block, num_warps=8)


@triton.jit
def _combine(Y, WTS, OUT, D: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    """A rank's MoE share, fp32: sum over slots in order of w_k y_k (the shared expert last, weight 1)."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    d = c * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SLOTS):
        acc = acc + tl.load(Y + (r * SLOTS + k) * D + d) * tl.load(WTS + r * SLOTS + k)
    tl.store(OUT + r * D + d, acc)


def combine(y: torch.Tensor, wts: torch.Tensor, out: torch.Tensor) -> None:
    rows, d = out.shape
    block = min(1024, d)
    _combine[(rows, d // block)](y, wts, out, D=d, SLOTS=wts.shape[1], BLOCK=block, num_warps=4)
