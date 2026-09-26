"""The forward's small kernels, one program per row (or per row and head): row-invariant by design.

Each program reads only its own row, so a row's bits never depend on the other rows of a window.
Where a matmul follows, the kernel also writes that matmul's 64-input group sums (fp32) from the
bf16 values it stores, so the lane matmul need not read its input twice.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm(X, R, W, H, Y, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    if HAS_R:
        r = tl.load(R + row * D + offs, mask=ok, other=0.0).to(tl.float32)
        h = (x + r).to(tl.bfloat16)
        tl.store(H + row * D + offs, h, mask=ok)
        x = h.to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(ss / D + eps)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(Y + row * D + offs, y, mask=ok)
    yg = tl.reshape(y.to(tl.float32), (BLOCK // 64, 64))
    g = tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (D // 64) + g, tl.sum(yg, axis=1), mask=g < D // 64)


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor | None, w: torch.Tensor, eps: float):
    """h = x + r (bf16), y = rmsnorm(h) * w (bf16), xs = y's 64-group sums. Returns (h, y, xs)."""

    rows, d = x.shape
    h = torch.empty_like(x) if r is not None else x
    y = torch.empty_like(x)
    xs = torch.empty((rows, d // 64), dtype=torch.float32, device=x.device)
    _add_rmsnorm[(rows,)](x, r if r is not None else x, w, h, y, xs, eps, D=d,
                          BLOCK=triton.next_power_of_2(d), HAS_R=r is not None, num_warps=8)
    return h, y, xs


@triton.jit
def _gdn_pre(QKV, CS, CW, WIN, A, B, ALOG, DTB, Q, K, V, G, BETA,
             C: tl.constexpr, KH: tl.constexpr, VH: tl.constexpr, DK: tl.constexpr, NKEEP: tl.constexpr):
    """Program (row, head): conv over the row's 4 inputs, SiLU, then q/k RMS-scaled; v as is."""

    row = tl.program_id(0)
    head = tl.program_id(1)                   # 0..KH-1 q, KH..2KH-1 k, then VH v heads
    ch = head * DK + tl.arange(0, DK)
    acc = tl.zeros((DK,), dtype=tl.float32)
    for j in tl.static_range(NKEEP + 1):
        src = tl.load(WIN + row * (NKEEP + 1) + j)
        from_state = src < NKEEP
        xs = tl.load(CS + src * C + ch, mask=(ch < C) & from_state, other=0.0)
        xw = tl.load(QKV + (src - NKEEP) * C + ch, mask=(ch < C) & (src >= NKEEP), other=0.0)
        x = tl.where(from_state, xs, xw).to(tl.float32)
        w = tl.load(CW + ch * (NKEEP + 1) + j).to(tl.float32)
        acc = acc + x * w
    c = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
    if head < 2 * KH:
        inv = 1.0 / tl.sqrt(tl.sum(c * c, axis=0) / DK + 1e-6)
        is_q = head < KH
        scale = tl.where(is_q, 1.0 / DK, 1.0 / tl.sqrt(DK * 1.0))
        out = (c * inv * scale).to(tl.bfloat16)
        hk = tl.where(is_q, head, head - KH)
        base = (row * KH + hk) * DK + tl.arange(0, DK)
        if is_q:
            tl.store(Q + base, out)
        else:
            tl.store(K + base, out)
    else:
        hv = head - 2 * KH
        tl.store(V + (row * VH + hv) * DK + tl.arange(0, DK), c.to(tl.bfloat16))
        a = tl.load(A + row * VH + hv).to(tl.float32) + tl.load(DTB + hv)
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = tl.exp(-tl.exp(tl.load(ALOG + hv)) * sp)
        b = tl.load(B + row * VH + hv).to(tl.float32)
        tl.store(G + row * VH + hv, g)
        tl.store(BETA + row * VH + hv, tl.sigmoid(b))


def gdn_pre(qkv: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, windows: torch.Tensor,
            a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, *, kh: int, vh: int, dk: int):
    """qkv (W, C) bf16; conv_state (n_keep, C); windows (W, n_keep+1) int32 rows of [conv_state; qkv].

    Returns q, k (W, kh, dk) bf16, v (W, vh, dk) bf16, g and beta (W, vh) fp32.
    """

    W, C = qkv.shape
    nkeep = conv_state.shape[0]
    dev = qkv.device
    q = torch.empty((W, kh, dk), dtype=torch.bfloat16, device=dev)
    k = torch.empty((W, kh, dk), dtype=torch.bfloat16, device=dev)
    v = torch.empty((W, vh, dk), dtype=torch.bfloat16, device=dev)
    g = torch.empty((W, vh), dtype=torch.float32, device=dev)
    beta = torch.empty((W, vh), dtype=torch.float32, device=dev)
    _gdn_pre[(W, 2 * kh + vh)](qkv, conv_state, conv_w, windows, a, b, A_log, dt_bias, q, k, v, g, beta,
                              C=C, KH=kh, VH=vh, DK=dk, NKEEP=nkeep, num_warps=2)
    return q, k, v, g, beta


@triton.jit
def _gated_norm(Yr, Z, W, OUT, XS, eps, VH: tl.constexpr, DV: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    offs = (row * VH + h) * DV + tl.arange(0, DV)
    y = tl.load(Yr + offs).to(tl.float32)
    z = tl.load(Z + offs).to(tl.float32)
    w = tl.load(W + tl.arange(0, DV)).to(tl.float32)
    yn = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=0) / DV + eps)) * w
    out = (z * tl.sigmoid(z) * yn).to(tl.bfloat16)
    tl.store(OUT + offs, out)
    og = tl.reshape(out.to(tl.float32), (DV // 64, 64))
    tl.store(XS + row * (VH * DV // 64) + h * (DV // 64) + tl.arange(0, DV // 64), tl.sum(og, axis=1))


def gated_norm(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float):
    """y, z (W, VH, DV) bf16 -> silu(z) * rmsnorm(y) * w as (W, VH*DV) bf16, plus its group sums."""

    W, vh, dv = y.shape
    out = torch.empty((W, vh * dv), dtype=torch.bfloat16, device=y.device)
    xs = torch.empty((W, vh * dv // 64), dtype=torch.float32, device=y.device)
    _gated_norm[(W, vh)](y, z, w, out, xs, eps, VH=vh, DV=dv, num_warps=1)
    return out, xs


@triton.jit
def _swiglu(GATE, UP, OUT, XS, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    ok = offs < N
    g = tl.load(GATE + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(UP + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    a = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    tl.store(OUT + row * N + offs, a, mask=ok)
    ag = tl.reshape(a.to(tl.float32), (BLOCK // 64, 64))
    gi = blk * (BLOCK // 64) + tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (N // 64) + gi, tl.sum(ag, axis=1), mask=gi < N // 64)


def swiglu(gate: torch.Tensor, up: torch.Tensor):
    W, n = gate.shape
    out = torch.empty_like(gate)
    xs = torch.empty((W, n // 64), dtype=torch.float32, device=gate.device)
    block = 1024
    _swiglu[(W, triton.cdiv(n, block))](gate, up, out, xs, N=n, BLOCK=block, num_warps=4)
    return out, xs


@triton.jit
def _attn_prep(QG, KV, QN, KN, POS, INV, QOUT, KOUT, eps,
               H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
    """Program (row, head): heads 0..H-1 are queries (from [q | gate] rows), H..H+HKV-1 keys."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    is_q = head < H
    if is_q:
        x = tl.load(QG + (row * H + head) * 2 * D + d).to(tl.float32)
        w = tl.load(QN + d).to(tl.float32)
    else:
        x = tl.load(KV + (row * HKV + head - H) * D + d).to(tl.float32)
        w = tl.load(KN + d).to(tl.float32)
    xn = (x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)) * w).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(POS + row).to(tl.float32)
    i = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
    ang = pos * tl.load(INV + i)
    cos = tl.cos(ang)
    sin = tl.sin(ang)
    # partner element for rotate-half: d < HALF pairs with d + HALF and back
    partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
    if is_q:
        xp = tl.load(QG + (row * H + head) * 2 * D + partner).to(tl.float32)
    else:
        xp = tl.load(KV + (row * HKV + head - H) * D + partner).to(tl.float32)
    wp = tl.load((QN if is_q else KN) + partner).to(tl.float32)
    xpn = (xp * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)) * wp).to(tl.bfloat16).to(tl.float32)
    rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xn * cos + xpn * sin, xn))
    out = rot.to(tl.bfloat16)
    if is_q:
        tl.store(QOUT + (row * H + head) * D + d, out)
    else:
        tl.store(KOUT + (row * HKV + head - H) * D + d, out)


def attn_prep(qg: torch.Tensor, k: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor,
              pos: torch.Tensor, inv_freq: torch.Tensor, eps: float, *, heads: int, kv_heads: int, head_dim: int):
    """qg (W, heads*2*D) [q_h | gate_h] rows, k (W, kv_heads*D): normed and rotated q (W, H, D), k (W, HKV, D)."""

    W = qg.shape[0]
    qo = torch.empty((W, heads, head_dim), dtype=torch.bfloat16, device=qg.device)
    ko = torch.empty((W, kv_heads, head_dim), dtype=torch.bfloat16, device=qg.device)
    _attn_prep[(W, heads + kv_heads)](qg, k, q_norm, k_norm, pos, inv_freq, qo, ko, eps, H=heads, HKV=kv_heads,
                                      D=head_dim, HALF=inv_freq.numel(), num_warps=2)
    return qo, ko


@triton.jit
def _gate_mul(O, QG, OUT, XS, H: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    o = tl.load(O + (row * H + h) * D + d).to(tl.float32)
    g = tl.load(QG + (row * H + h) * 2 * D + D + d).to(tl.float32)
    out = (o * tl.sigmoid(g)).to(tl.bfloat16)
    tl.store(OUT + (row * H + h) * D + d, out)
    og = tl.reshape(out.to(tl.float32), (D // 64, 64))
    tl.store(XS + row * (H * D // 64) + h * (D // 64) + tl.arange(0, D // 64), tl.sum(og, axis=1))


def gate_mul(o: torch.Tensor, qg: torch.Tensor, *, heads: int, head_dim: int):
    """Attention output (W, H, D) times sigmoid(gate) from the [q | gate] rows -> (W, H*D) bf16 and group sums."""

    W = o.shape[0]
    out = torch.empty((W, heads * head_dim), dtype=torch.bfloat16, device=o.device)
    xs = torch.empty((W, heads * head_dim // 64), dtype=torch.float32, device=o.device)
    _gate_mul[(W, heads)](o, qg, out, xs, H=heads, D=head_dim, num_warps=2)
    return out, xs


@triton.jit
def _embed(IDS, Wt, S, B, OUT, D: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    tok = tl.load(IDS + row).to(tl.int64)
    words = tl.load(Wt + tok * (D // 8) + g * 8 + tl.arange(0, 8))
    q = (words[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF
    q = tl.reshape(q, (64,)).to(tl.float32)
    s = tl.load(S + tok * (D // 64) + g).to(tl.float32)
    b = tl.load(B + tok * (D // 64) + g).to(tl.float32)
    tl.store(OUT + row * D + g * 64 + tl.arange(0, 64), (q * s + b).to(tl.bfloat16))


def embed(ids: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, d: int) -> torch.Tensor:
    W = ids.shape[0]
    out = torch.empty((W, d), dtype=torch.bfloat16, device=ids.device)
    _embed[(W, d // 64)](ids, weight, scales, biases, out, D=d, num_warps=1)
    return out
