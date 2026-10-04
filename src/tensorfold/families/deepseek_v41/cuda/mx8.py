"""DeepSeek's FP8 projections on ``Mx8Linear``: bf16 outputs through it, fp32 outputs on its kernels' fp32 epilogue."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4.linear import FUSED_ROWS, MXFP8, PROMPT_TILE, Mx8Linear, _ext, _prompt_ext


def _folds(lin: Mx8Linear) -> bool:
    """``Mx8Linear.prefill``'s test: every exponent keeps an e4m3 byte times its power of two exact in bf16."""

    if lin.fold is None:
        lin.fold = bool(((lin.bs >= 3) & (lin.bs <= 134)).all())
    return lin.fold


def _f32(lin: Mx8Linear, x: torch.Tensor, out: torch.Tensor | None, prompt: bool) -> torch.Tensor:
    """The kernels ``lin`` / ``lin.prefill`` run, with their fp32 epilogue: the same sums, not rounded to bf16."""

    folded = prompt and _folds(lin)
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or (folded and (x.data_ptr() % 16 or (
            x.shape[0] > 1 and x.stride(0) % 8))):
        x = x.to(torch.bfloat16).contiguous()
    m = x.shape[0]
    fits = out is not None and out.is_contiguous() and out.dtype == torch.float32 and tuple(out.shape) == (m, lin.n)
    y = out if fits else torch.empty((m, lin.n), dtype=torch.float32, device=x.device)
    if folded:
        _prompt_ext().prompt16(x, lin.w8, lin.bs, 1.0, y, MXFP8, lin.n, lin.npad, PROMPT_TILE, True)
    else:
        sk = qmm.split_k(lin.n, lin.k)
        part = torch.empty((sk, m, lin.n), dtype=torch.float32, device=x.device) if sk > 8 else None
        bm = 0 if sk > 1 and m >= FUSED_ROWS else qmm.bucket(m)
        _ext().qmmf(x, lin.w8, lin.bs, 1.0, y, part if bm else None, MXFP8, lin.n, sk, lin.npad, bm, True)
    if out is not None and y is not out:
        out.copy_(y)
    return y


def mm(lin: Mx8Linear, x: torch.Tensor, out: torch.Tensor | None = None, *, f32: bool = False,
       prompt: bool = False) -> torch.Tensor:
    """x (M, K) -> (M, n) bf16 as ``lin`` or (``prompt``) ``lin.prefill``, or fp32 before that rounding."""

    if f32:
        return _f32(lin, x, out, prompt)
    return lin.prefill(x, out) if prompt else lin(x, out)


def grouped(lins: Sequence[Mx8Linear], x: torch.Tensor, out: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """``wo_a``: group g's projection of x's g-th block of columns into out's g-th block of columns, bf16."""

    k, n = lins[0].k, lins[0].n
    if x.shape[-1] != len(lins) * k or out.shape[-1] != len(lins) * n:
        raise ValueError(f"{len(lins)} groups of [{n}, {k}] do not tile x {tuple(x.shape)} -> out {tuple(out.shape)}")
    for g, lin in enumerate(lins):
        mm(lin, x[:, g * k:(g + 1) * k], out[:, g * n:(g + 1) * n], prompt=prompt)
    return out
