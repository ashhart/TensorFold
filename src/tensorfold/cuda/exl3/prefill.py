"""EXL3 prompt matmuls: the weights decoded once a call into a fixed-tile GEMM. A family that opts in
(``Workspace(fold=True)``) decodes W'' = diag(suh) H W_q H / 128, both rotations folded in, and multiplies the raw
bf16 rows by it; otherwise W_q is decoded into fp16 and the GEMM's epilogue rotates each 128-column block."""

from __future__ import annotations

import os

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


@triton.jit(do_not_specialize=["M"])
def _gemm_fold(X, W, SVH, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    """OUT[m, n] = (x[m] @ W''[:, n]) * svh + bias, x cast to the weights' dtype as loaded; K in BK steps in order,
    a row alone."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // BN)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0).to(W.dtype.element_ty)
        w = tl.load(W + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, w, acc)
    y = acc * tl.load(SVH + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], y.to(OUT.dtype.element_ty), mask=ok[:, None])


def tiles(k: int, n: int) -> tuple[int, int, int, int, int]:
    """(rows a program, K step, warps, stages, row blocks a raster group): the shape's alone, so a row never depends on its chunk."""

    return 128, 64, 8, 4, 8


# _gemm_fold's (BM, BN, BK, warps, stages, raster group) for the 27B's projections (K, N), swept on a GB10
FOLD_TILES = {
    (5120, 1024): (128, 256, 64, 8, 3, 4), (5120, 6144): (128, 256, 64, 8, 3, 16),
    (5120, 10240): (128, 256, 64, 8, 3, 8), (5120, 12288): (128, 256, 64, 8, 3, 8),
    (5120, 17408): (128, 256, 64, 8, 3, 8), (6144, 5120): (128, 256, 64, 8, 3, 16),
    (17408, 5120): (128, 256, 64, 8, 3, 8),
}
FOLD = os.environ.get("TENSORFOLD_EXL3_FOLD", "1") != "0"            # 0: every family on the W_q path
FOLD_BF16 = os.environ.get("TENSORFOLD_EXL3_FOLD_BF16", "1") != "0"  # 0: fp16 W'' (and fp16 rows in the GEMM)
FOLD2 = os.environ.get("TENSORFOLD_EXL3_FOLD2", "1") != "0"          # 0: unpack_fold, the same bits, slower
FDIRECT_ROWS = int(os.environ.get("TENSORFOLD_EXL3_FDIRECT_ROWS", "0"))    # 4-bit calls this short: fdirect (0: never, the default)
SCOPE: list | None = None            # layer-major prefill: the layers whose W'' is held now (None: none is held)


def tiles_fold(k: int, n: int) -> tuple[int, int, int, int, int, int]:
    """_gemm_fold's tile for a (K, N): by the shape alone, never by the row count."""

    if (k, n) in FOLD_TILES:
        return FOLD_TILES[(k, n)]
    if n % 256 == 0 and k % 64 == 0 and (k >= 6144 or n >= 6144):
        return 128, 256, 64, 8, 3, 8
    return 128, 128, 32, 8, 4, 8


def scope_begin() -> None:
    """Hold each W'' a folded call decodes until ``scope_release`` (a layer-major prefill: one layer at a time)."""

    global SCOPE
    SCOPE = []


def scope_release() -> None:
    if SCOPE:
        for layer in SCOPE:
            layer._prompt_w = None
        SCOPE.clear()


def scope_end() -> None:
    global SCOPE
    scope_release()
    SCOPE = None


class Workspace:
    """One decoded weight and one rotated input, grown to the largest call and reused (calls run in order on one
    stream); ``fold``: bf16 calls take the folded path (W'' and raw rows)."""

    def __init__(self, fold: bool = False) -> None:
        self.fold = fold
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
    if FOLD and ws.fold and out.dtype == torch.bfloat16 and x.dtype == torch.bfloat16:
        wq = getattr(layer, "_prompt_w", None) if SCOPE is not None else None
        if (wq is None and m <= FDIRECT_ROWS and FOLD_BF16 and layer.k2 == 8 and x.stride(1) == 1
                and x.stride(0) % 2 == 0 and x.data_ptr() % 4 == 0):   # bf16 W'' rebuilt in the GEMM: the same bits
            ext.fdirect(x, layer.words, *layer.strides, layer.suh, layer.svh, layer.bias, out, layer.k2,
                        CODEBOOK_IDS[layer.codebook])
            return out
        if wq is None:
            wdt = torch.bfloat16 if FOLD_BF16 else torch.float16
            wq = (torch.empty((k, n), device=x.device, dtype=wdt) if SCOPE is not None
                  else ws._grow("w", k * n, x.device)[:k * n].view(k, n).view(wdt))
            (ext.unpack_fold2 if FOLD2 else ext.unpack_fold)(layer.words, layer.suh, wq, *layer.strides, layer.k2,
                                                             CODEBOOK_IDS[layer.codebook])
            if SCOPE is not None:                  # held for the layer's other chunks
                layer._prompt_w = wq
                SCOPE.append(layer)
        bm, bn, bk, warps, stages, group = tiles_fold(k, n)
        bias = layer.bias if layer.bias is not None else layer.svh
        x = x if x.stride(1) == 1 and x.stride(0) == k else x.contiguous()
        _gemm_fold[(triton.cdiv(m, bm) * (n // bn),)](x, wq, layer.svh, bias, out, m, out.stride(0), K=k, N=n, BM=bm,
                                                      BN=bn, BK=bk, GROUP=group, HAS_BIAS=layer.bias is not None,
                                                      num_warps=warps, num_stages=stages)
        return out
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
