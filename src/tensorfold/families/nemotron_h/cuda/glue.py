"""Nemotron-H's routing, combine and MTP-input kernels read only their own row, so a row's bits ignore its window."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.qwen3_5.cuda import qmm_fast


def dense(x: torch.Tensor, q, xs: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
    """The shared 4-bit lane matmul (tiled weights): bf16 out, or unrounded fp32 sums (a rank's partial)."""

    return qmm_fast.matmul_partial(x, q, xs) if f32 else qmm_fast.matmul(x, q, xs)


def prefill_dense(x: torch.Tensor, q, *, f32: bool = False) -> torch.Tensor:
    """A prompt chunk's matmul on the shared prefill kernel (bf16 weights, one chain over K; row-invariant)."""

    if q.layout == "volta":                          # sm_70: a dense fp16 copy through one cuBLASLt GEMM
        from tensorfold.cuda.kernels import qmm_volta

        return qmm_volta.prompt_matmul(x, qmm_volta.Tiled.wrap(q.weight, q.scales, q.biases, q.rows), f32=f32)
    return shared.prefill_matmul(x, q, f32=f32)


def _divisor(n: int, want: int) -> int:
    for g in range(min(want, n), 0, -1):
        if n % g == 0:
            return g
    return 1


@triton.jit(do_not_specialize=["R"])
def _router(X, W, PART, R, D: tl.constexpr, E: tl.constexpr, SK: tl.constexpr, BE: tl.constexpr):
    """Program (row tile, expert block, K slice): PART[s, row, e] = fp32 x[row] . w[e] over the slice's groups."""

    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    e = tl.program_id(1) * BE + tl.arange(0, BE)
    s = tl.program_id(2)
    k = tl.arange(0, 64)
    PER: tl.constexpr = D // 64 // SK
    acc = tl.zeros((16, BE), tl.float32)
    for i in range(PER):
        g = s * PER + i
        x = tl.load(X + rows[:, None] * D + g * 64 + k[None, :], mask=rows[:, None] < R, other=0.0)
        w = tl.load(W + e[:, None] * D + g * 64 + k[None, :], mask=e[:, None] < E, other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(PART + (s * R + rows[:, None]) * E + e[None, :], acc, mask=(rows[:, None] < R) & (e[None, :] < E))


@triton.jit
def _topk(PART, BIAS, IDX, WT, R, scaling, E: tl.constexpr, EP: tl.constexpr, SK: tl.constexpr,
          TOPK: tl.constexpr, NS: tl.constexpr, NORM: tl.constexpr):
    """Program per row: its routed experts and weights in pick order, then the shared halves."""

    r = tl.program_id(0)
    e = tl.arange(0, EP)
    ok = e < E
    logit = tl.zeros((EP,), tl.float32)
    for s in tl.static_range(SK):
        logit = logit + tl.load(PART + (s * R + r) * E + e, mask=ok, other=0.0)
    score = tl.sigmoid(logit)
    sel = tl.where(ok, score + tl.load(BIAS + e, mask=ok, other=0.0), float("-inf"))
    kcol = tl.arange(0, NS)
    probs = tl.zeros((NS,), tl.float32)
    ids = tl.zeros((NS,), tl.int32)
    total = 0.0
    for k in tl.static_range(TOPK):
        best = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == best, e, EP), axis=0)
        p = tl.sum(tl.where(e == idx, score, 0.0), axis=0)
        probs = tl.where(kcol == k, p, probs)
        ids = tl.where(kcol == k, idx, ids)
        total = total + p
        sel = tl.where(e == idx, float("-inf"), sel)
    if NORM:
        wts = probs / (total + 1e-20) * scaling
    else:
        wts = probs * scaling
    ids = tl.where(kcol == TOPK, E, ids)
    ids = tl.where(kcol == TOPK + 1, E + 1, ids)
    wts = tl.where(kcol >= TOPK, 1.0, wts)
    tl.store(IDX + r * NS + kcol, ids)
    tl.store(WT + r * NS + kcol, wts)


def route(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, *, top_k: int,
          scaling: float, norm: bool) -> None:
    """x (R, D) bf16, router w (E, D) bf16 -> ids (R, top_k + 2) int32 (routed, then E and E + 1) and fp32 wts."""

    rows, d = x.shape
    e = w.shape[0]
    ns = top_k + 2
    if ns & (ns - 1):
        raise ValueError("top_k + 2 must be a power of two")
    sk = _divisor(d // 64, 6)
    part = torch.empty((sk, rows, e), dtype=torch.float32, device=x.device)
    be = 16
    _router[(triton.cdiv(rows, 16), triton.cdiv(e, be), sk)](x, w, part, rows, D=d, E=e, SK=sk, BE=be, num_warps=4)
    _topk[(rows,)](part, bias, ids, wts, rows, float(scaling), E=e, EP=triton.next_power_of_2(e), SK=sk, TOPK=top_k,
                   NS=ns, NORM=norm, num_warps=4)


@triton.jit
def _add_moe_norm(H, Y, WT, W, HN, OUT, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr, NR: tl.constexpr,
                  NS: tl.constexpr):
    """delta = bf16(sum_k<NR wt_k y_k + sum_k>=NR y_k), summed in slot order; h = bf16(x + delta), then its RMSNorm."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    routed = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(NR):
        wk = tl.load(WT + row * NS + k)
        routed = routed + wk * tl.load(Y + (row * NS + k) * D + offs, mask=ok, other=0.0).to(tl.float32)
    shared = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(NR, NS):
        shared = shared + tl.load(Y + (row * NS + k) * D + offs, mask=ok, other=0.0).to(tl.float32)
    delta = (routed + shared).to(tl.bfloat16).to(tl.float32)
    x = tl.load(H + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    h = (x + delta).to(tl.bfloat16)
    tl.store(HN + row * D + offs, h, mask=ok)
    hf = h.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(hf * hf, axis=0) / D + eps)
    y = (hf * inv * tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * D + offs, y, mask=ok)
    yg = tl.reshape(y.to(tl.float32), (BLOCK // 64, 64))
    g = tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (D // 64) + g, tl.sum(yg, axis=1), mask=g < D // 64)


def add_moe_norm(h: torch.Tensor, y: torch.Tensor, wt: torch.Tensor, w: torch.Tensor, eps: float, top_k: int):
    """h (R, D) bf16, y (R * NS, D) fp32 or bf16 in prefill, wt (R, NS) fp32 -> (h + delta, its RMSNorm, group sums)."""

    rows, d = h.shape
    hn = torch.empty_like(h)
    out = torch.empty_like(h)
    xs = torch.empty((rows, d // 64), dtype=torch.float32, device=h.device)
    _add_moe_norm[(rows,)](h, y, wt, w, hn, out, xs, eps, D=d, BLOCK=triton.next_power_of_2(d), NR=top_k,
                           NS=wt.shape[1], num_warps=8)
    return hn, out, xs


@triton.jit
def _concat_norms(E, Hd, WE, WH, OUT, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    """MTP input: [rmsnorm(e) * enorm | rmsnorm(h) * hnorm] (R, 2D) bf16 and its 64-group sums."""

    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    first = part == 0
    x = (tl.load(E + row * D + offs, mask=ok & first, other=0.0).to(tl.float32)
         + tl.load(Hd + row * D + offs, mask=ok & (part == 1), other=0.0).to(tl.float32))
    w = (tl.load(WE + offs, mask=ok & first, other=0.0).to(tl.float32)
         + tl.load(WH + offs, mask=ok & (part == 1), other=0.0).to(tl.float32))
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(OUT + row * 2 * D + part * D + offs, y, mask=ok)
    yg = tl.reshape(y.to(tl.float32), (BLOCK // 64, 64))
    g = tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (2 * D // 64) + part * (D // 64) + g, tl.sum(yg, axis=1), mask=g < D // 64)


def concat_norms(e: torch.Tensor, h: torch.Tensor, we: torch.Tensor, wh: torch.Tensor, eps: float):
    rows, d = e.shape
    out = torch.empty((rows, 2 * d), dtype=torch.bfloat16, device=e.device)
    xs = torch.empty((rows, 2 * d // 64), dtype=torch.float32, device=e.device)
    _concat_norms[(rows, 2)](e, h, we, wh, out, xs, eps, D=d, BLOCK=triton.next_power_of_2(d), num_warps=8)
    return out, xs
