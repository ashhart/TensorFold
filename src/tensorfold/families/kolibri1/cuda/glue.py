"""Kolibri 1's fused glue: q/k norms with RoPE and cache writes; residual adds with the next norm (a row a program)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv(QKV, NW, COS, SIN, POS, SLOT, Q, KO, VO, KC, VC, LEN, H: tl.constexpr, HK: tl.constexpr,
         D: tl.constexpr, EPS: tl.constexpr, ROPE: tl.constexpr):
    """Program (row, head of H + HK + HK): q and k RMS-normed (RoPE'd when ROPE), v as is; k, v into the cache too."""

    r = tl.program_id(0)
    hd = tl.program_id(1)
    d = tl.arange(0, D)
    src = QKV + r.to(tl.int64) * (H + 2 * HK) * D + hd * D
    x = tl.load(src + d).to(tl.float32)
    pos = tl.load(POS + r)
    if hd < H + HK:
        is_q = hd < H
        wrow = NW + (hd >= H).to(tl.int32) * D                 # q_norm, then k_norm
        w = tl.load(wrow + d)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * rstd * w
        if ROPE:
            half: tl.constexpr = D // 2
            perm = (d + half) % D
            yp = tl.load(src + perm).to(tl.float32) * rstd * tl.load(wrow + perm)
            cos = tl.load(COS + pos.to(tl.int64) * D + d)
            sin = tl.load(SIN + pos.to(tl.int64) * D + d)
            y = y * cos + tl.where(d < half, -yp, yp) * sin
        y = y.to(tl.bfloat16)
        if is_q:
            tl.store(Q + (r.to(tl.int64) * H + hd) * D + d, y)
        else:
            k = hd - H
            tl.store(KO + (r.to(tl.int64) * HK + k) * D + d, y)
            at = (tl.load(SLOT + r).to(tl.int64) * LEN + pos % LEN) * HK + k
            tl.store(KC + at * D + d, y)
    else:
        k = hd - H - HK
        y = x.to(tl.bfloat16)
        tl.store(VO + (r.to(tl.int64) * HK + k) * D + d, y)
        at = (tl.load(SLOT + r).to(tl.int64) * LEN + pos % LEN) * HK + k
        tl.store(VC + at * D + d, y)


def qkv(qkv_rows: torch.Tensor, norms: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        pos: torch.Tensor, slots: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, *, heads: int,
        kv_heads: int, eps: float, rope: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """qkv rows [n, (H + 2 HK) D] bf16, norms [2, D] (q, k) -> q [n, H, D], k, v [n, HK, D]; caches at pos % LEN."""

    n, width = qkv_rows.shape
    d = width // (heads + 2 * kv_heads)
    q = torch.empty((n, heads, d), dtype=torch.bfloat16, device=qkv_rows.device)
    k = torch.empty((n, kv_heads, d), dtype=torch.bfloat16, device=qkv_rows.device)
    v = torch.empty_like(k)
    _qkv[(n, heads + 2 * kv_heads)](qkv_rows.contiguous(), norms, cos, sin, pos, slots, q, k, v, k_cache,
                                    v_cache, k_cache.shape[1], H=heads, HK=kv_heads, D=d, EPS=float(eps), ROPE=rope,
                                    num_warps=1)
    return q, k, v


@triton.jit
def _add_rms(X, RES, W1, W2, OUT, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: res += rms(x) * w1 (fp32, in place), then out = bf16(rms(res) * w2)."""

    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    ok = c < N
    x = tl.load(X + r.to(tl.int64) * N + c, mask=ok, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS) * tl.load(W1 + c, mask=ok, other=0.0)
    res = tl.load(RES + r.to(tl.int64) * N + c, mask=ok, other=0.0) + y
    tl.store(RES + r.to(tl.int64) * N + c, res, mask=ok)
    z = res * tl.rsqrt(tl.sum(res * res, 0) / N + EPS) * tl.load(W2 + c, mask=ok, other=0.0)
    tl.store(OUT + r.to(tl.int64) * N + c, z.to(tl.bfloat16), mask=ok)


def add_rms(x: torch.Tensor, res: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, eps: float) -> torch.Tensor:
    """``res`` (fp32, in place) += rms(x) * w1; returns bf16 rms(res) * w2."""

    n = x.shape[-1]
    out = torch.empty(res.shape, dtype=torch.bfloat16, device=res.device)
    _add_rms[(res.shape[0],)](x.contiguous(), res, w1, w2, out, N=n, EPS=float(eps),
                              BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out
