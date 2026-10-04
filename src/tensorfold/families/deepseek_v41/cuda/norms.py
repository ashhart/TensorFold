"""DeepSeek RMSNorm and the mHC kernels: a row a program, IEEE divides and roots, no FMA contraction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.families.glm5_next.cuda.glue import HC_BLOCKS, _hc_partial

HC = 4                              # residual copies
MIX = (2 + HC) * HC                 # pre, post and comb logits of one row


def _warps(block: int) -> int:
    return 4 if block <= 2048 else 8 if block <= 4096 else 16


@triton.jit
def _normed(c, W, d, ok, eps, D: tl.constexpr):
    """bf16(w * (c * rsqrt(mean(c^2) + eps))) for one fp32 row, rounded once."""

    ss = tl.sum(c * c, axis=0)
    rinv = tl.math.div_rn(1.0, tl.sqrt_rn(tl.math.div_rn(ss, D * 1.0) + eps))
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    return (w * (c * rinv)).to(tl.bfloat16)


@triton.jit
def _rmsnorm(X, x_stride, W, OUT, o_stride, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * x_stride + d, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * o_stride + d, _normed(x, W, d, ok, eps, D), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
    """RMSNorm: x [R, D] (rows may be strided) -> out [R, D] bf16."""

    rows, d = x.shape
    if x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("rmsnorm: rows of x and out must be contiguous")
    block = triton.next_power_of_2(d)
    _rmsnorm[(rows,)](x, x.stride(0), w, out, out.stride(0), eps, D=d, BLOCK=block, num_warps=_warps(block),
                      enable_fp_fusion=False)
    return out


@triton.jit
def _sigmoid(z):
    return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-z))


@triton.jit
def _hc_coeffs(PART, BASE, SCALE, PRE, POST, COMB, eps_norm, hc_eps, WIDE: tl.constexpr, NB: tl.constexpr,
               ITERS: tl.constexpr):
    """Sum a row's K blocks in order, scale by rsqrt(mean square), and split into pre, post and Sinkhorn comb."""

    r = tl.program_id(0).to(tl.int64)
    m = tl.arange(0, 32)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    mix = mix * tl.math.div_rn(1.0, tl.sqrt_rn(tl.math.div_rn(ss, WIDE * 1.0) + eps_norm))
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    sv = tl.arange(0, 4)
    # pre = mix[0:4], post = mix[4:8], comb = mix[8:24] (row j, column k at 8 + 4j + k), each with its own scale
    pre_z = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * tl.load(SCALE + 0) + base)[None, :], 0.0), axis=1)
    post_z = tl.sum(tl.where(m[None, :] == sv[:, None] + 4, (mix * tl.load(SCALE + 1) + base)[None, :], 0.0), axis=1)
    jj = tl.arange(0, 4)[:, None]
    kk = tl.arange(0, 4)[None, :]
    flat = 8 + jj * 4 + kk
    cz = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * tl.load(SCALE + 2) + base)[None, None, :], 0.0),
                axis=2)
    ce = libdevice.exp(cz - tl.max(cz, axis=1)[:, None])
    comb = tl.math.div_rn(ce, tl.sum(ce, axis=1)[:, None]) + hc_eps
    comb = tl.math.div_rn(comb, tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = tl.math.div_rn(comb, tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = tl.math.div_rn(comb, tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE + r * 4 + sv, _sigmoid(pre_z) + hc_eps)
    tl.store(POST + r * 4 + sv, 2.0 * _sigmoid(post_z))
    tl.store(COMB + r * 16 + jj * 4 + kk, comb)


def hc_mix(x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, part: torch.Tensor,
           pre: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, eps_norm: float = 1e-20, hc_eps: float = 1e-6,
           iters: int = 20) -> None:
    """mHC: x [R, 4*D] bf16 streams -> pre [R, 4], post [R, 4], comb [R, 4, 4] fp32 (part: [R, 16, 32] scratch)."""

    rows, wide = x.shape
    if fn.shape != (MIX, wide) or part.numel() < rows * HC_BLOCKS * 32 or not x.is_contiguous():
        raise ValueError(f"hc_mix: x {tuple(x.shape)}, fn {tuple(fn.shape)}, part {tuple(part.shape)}")
    _hc_partial[(rows, HC_BLOCKS)](x, fn, part, WIDE=wide, NB=HC_BLOCKS, SUB=128, num_warps=4)
    _hc_coeffs[(rows,)](part, base, scale, pre, post, comb, eps_norm, hc_eps, WIDE=wide, NB=HC_BLOCKS, ITERS=iters,
                        num_warps=1, enable_fp_fusion=False)


@triton.jit
def _collapse_norm(X, PRE, W, OUT, RAW, eps, D: tl.constexpr, BLOCK: tl.constexpr, KEEP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    ok = d < D
    row = X + r * (4 * D) + d
    c = tl.load(PRE + r * 4) * tl.load(row, mask=ok, other=0.0).to(tl.float32)
    c = c + tl.load(PRE + r * 4 + 1) * tl.load(row + D, mask=ok, other=0.0).to(tl.float32)
    c = c + tl.load(PRE + r * 4 + 2) * tl.load(row + 2 * D, mask=ok, other=0.0).to(tl.float32)
    c = c + tl.load(PRE + r * 4 + 3) * tl.load(row + 3 * D, mask=ok, other=0.0).to(tl.float32)
    c = c.to(tl.bfloat16)
    if KEEP:
        tl.store(RAW + r * D + d, c, mask=ok)
    tl.store(OUT + r * D + d, _normed(c.to(tl.float32), W, d, ok, eps, D), mask=ok)


def collapse_norm(x: torch.Tensor, pre: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float = 1e-20,
                  raw: torch.Tensor | None = None) -> torch.Tensor:
    """out = rmsnorm(bf16(((p0 x0 + p1 x1) + p2 x2) + p3 x3)) for a given pre; ``raw`` keeps the collapse."""

    rows, wide = x.shape
    d = w.shape[0]
    if wide != HC * d or not x.is_contiguous() or not out.is_contiguous():
        raise ValueError(f"collapse_norm: x {tuple(x.shape)} is not {HC} contiguous copies of {d}")
    block = triton.next_power_of_2(d)
    _collapse_norm[(rows,)](x, pre, w, out, raw if raw is not None else out, eps, D=d, BLOCK=block,
                            KEEP=raw is not None, num_warps=_warps(block), enable_fp_fusion=False)
    return out
