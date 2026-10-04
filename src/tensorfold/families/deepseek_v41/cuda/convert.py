"""Exact load-time rewrites of the pack's weights into TensorFold's layouts: every weight value stays the same."""

from __future__ import annotations

import math

import torch

from tensorfold.cuda.nvfp4 import experts as nvfp4

BLOCK = 32                # rows and inputs of a DeepSeek FP8 scale block; inputs of an MXFP4 scale
SPAN = 17                 # widest exponent spread a matrix's e4m3 scales hold exactly: 2^-9 (subnormal) .. 2^8


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """E8M0, I8 or e4m3 storage as its raw bytes (E8M0 is never read through a float type)."""

    return t.contiguous().view(torch.uint8)


def fp8_block_rows(scale: torch.Tensor, n: int) -> torch.Tensor:
    """32x32 block scales [ceil(N/32), K/32] -> per-row scale bytes [N, K/32]: row i repeats block row i // 32."""

    s = _bytes(scale)
    if s.dim() != 2 or s.shape[0] != -(-n // BLOCK):
        raise ValueError(f"block scales {tuple(scale.shape)} do not cover {n} rows in blocks of {BLOCK}")
    return s.repeat_interleave(BLOCK, dim=0)[:n].contiguous()


def mxfp4_to_nvfp4(words: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One MXFP4 matrix -> NVFP4 words, e4m3 scales per 16 and one fp32 scale, every product exact."""

    w, e = _bytes(words), _bytes(scale)
    if w.dim() != 2 or (2 * w.shape[1]) % BLOCK or tuple(e.shape) != (w.shape[0], 2 * w.shape[1] // BLOCK):
        raise ValueError(f"MXFP4 words {tuple(words.shape)} with scales {tuple(scale.shape)} do not match")
    top, low = int(e.max()), int(e.min())
    if top == 0xFF:
        raise ValueError("MXFP4 scale byte 0xFF is NaN")
    if top - low > SPAN:
        raise ValueError(f"MXFP4 exponent span {top - low} exceeds {SPAN}: e4m3 scales cannot hold it exactly")
    powers = torch.tensor([2.0 ** d for d in range(-9, 9)], device=e.device).to(torch.float8_e4m3fn)
    sixteen = powers[(e.to(torch.int64) - top + SPAN)].view(torch.uint8).repeat_interleave(2, dim=1)
    whole = torch.tensor(math.ldexp(1.0, top - 135), dtype=torch.float32, device=e.device)
    return w, sixteen.contiguous(), whole


def _stacked(parts: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    done = [mxfp4_to_nvfp4(w, s) for w, s in parts]
    return tuple(torch.stack([d[i] for d in done]) for i in range(3))


def make_experts4(gate: list, up: list, down: list, limit: float = 10.0) -> nvfp4.Experts4:
    """Per-expert (words, E8M0) pairs of one rank's gate, up and down -> ``Experts4``, each matrix scaled alone."""

    return nvfp4.make(_stacked(gate), _stacked(up), _stacked(down), limit=limit)
