"""EXL3 prompt matmuls: W_q decoded once a chunk into a fixed-tile fp16 GEMM whose epilogue rotates each 128-column block."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .linear import CODEBOOK_IDS, Exl3Linear, _ext

HAD_SCALE = 0.08838834764831845          # 1 / sqrt(128)
BN = 128                                  # a program's columns: one Hadamard block


@triton.jit(do_not_specialize=["M"])
def _gemm(X, W, H, SVH, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
          BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, SCALE: tl.constexpr):
    """OUT[m, block] = ((xh[m] @ W_q[:, block]) @ H) * SCALE * svh + bias; K in BK steps in order, a row alone."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // 128)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, 128), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(W + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, w, acc)
    hi = tl.arange(0, 128)
    h = tl.load(H + hi[:, None] * 128 + hi[None, :])
    top = acc.to(tl.bfloat16)
    rest = (acc - top.to(tl.float32)).to(tl.bfloat16)
    y = tl.dot(rest, h, tl.dot(top, h))
    y = y * SCALE * tl.load(SVH + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], y.to(OUT.dtype.element_ty), mask=ok[:, None])


def tiles(k: int, n: int) -> tuple[int, int, int, int, int]:
    """(rows a program, K step, warps, stages, row blocks a raster group): the shape's alone, so a row never depends on its chunk."""

    return 128, 64, 8, 3, 8      # 2-5% over (128, 32, 8, 4, 8) on GB10, the same bits (the K step only regroups MMAs)


class Workspace:
    """One decoded W_q and one rotated input, grown to the largest call and reused (calls run in order on one stream)."""

    def __init__(self) -> None:
        self.w: torch.Tensor | None = None
        self.xh: torch.Tensor | None = None
        self.h: torch.Tensor | None = None

    def _grow(self, name: str, numel: int, device) -> torch.Tensor:
        t = getattr(self, name)
        if t is None or t.numel() < numel:
            t = torch.empty((numel,), dtype=torch.float16, device=device)
            setattr(self, name, t)
        return t

    def hadamard(self, device) -> torch.Tensor:
        if self.h is None:
            i = torch.arange(128, device=device)
            parity = torch.tensor([bin(v).count("1") & 1 for v in range(128)], device=device)[i[:, None] & i[None, :]]
            self.h = (1.0 - 2.0 * parity.float()).to(torch.bfloat16).contiguous()
        return self.h

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.xh, self.h) if t is not None)


def matmul(layer: Exl3Linear, x: torch.Tensor, out: torch.Tensor, ws: Workspace) -> torch.Tensor:
    """out [M, N] (row stride free) = x [M, K] @ W + bias for any M, the prompt path's arithmetic."""

    m, k, n = x.shape[0], layer.k, layer.n
    if x.shape[1] != k or out.shape != (m, n) or out.stride(1) != 1:
        raise ValueError(f"prefill matmul: x {tuple(x.shape)} and out {tuple(out.shape)} do not match K={k}, N={n}")
    ext = _ext()
    xh = ws._grow("xh", m * k, x.device)[:m * k].view(m, k)
    ext.rot_in(x.contiguous(), layer.suh, xh)
    wq = ws._grow("w", k * n, x.device)[:k * n].view(k, n)
    ext.unpack(layer.words, wq, *layer.strides, layer.k2, CODEBOOK_IDS[layer.codebook])
    bm, bk, warps, stages, group = tiles(k, n)
    bias = layer.bias if layer.bias is not None else layer.svh
    _gemm[(triton.cdiv(m, bm) * (n // BN),)](xh, wq, ws.hadamard(x.device), layer.svh, bias, out, m, out.stride(0),
                                             K=k, N=n, BM=bm, BK=bk, GROUP=group, HAS_BIAS=layer.bias is not None,
                                             SCALE=HAD_SCALE, num_warps=warps, num_stages=stages)
    return out
