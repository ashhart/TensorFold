"""DeepSeek-V4.1's RoPE on the last ``qk_rope_head_dim`` dims, adjacent pairs, from fp32 device tables."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..config import Config

KINDS = ("local", "yarn")       # local: rope_theta, no YaRN (window layers, DSpark); yarn: compress_rope_theta + YaRN
HEADS = 16                      # heads per program


def tables(cfg: Config, kind: str, capacity: int, device: torch.device | str) -> torch.Tensor:
    """M:368-389 with torch on ``device``: fp32 [capacity, rope/2, 2], (cos, sin) of position p at row p."""

    if kind not in KINDS:
        raise ValueError(f"RoPE kind {kind!r}: one of {KINDS}")
    dim = cfg.qk_rope_head_dim
    base = cfg.compress_rope_theta if kind == "yarn" else cfg.rope_theta
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if kind == "yarn":
        low, high = cfg.yarn
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / cfg.rope_factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(capacity, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(freqs), freqs))


@triton.jit
def _rope(X, P, T, sr, sh, H, D: tl.constexpr, HALF: tl.constexpr, PER_ROW: tl.constexpr,
          INVERSE: tl.constexpr, HB: tl.constexpr):
    row = tl.program_id(0)
    if PER_ROW:
        p = tl.load(P + row).to(tl.int64)
    else:
        p = tl.load(P).to(tl.int64) + row
    i = tl.arange(0, HALF)
    c = tl.load(T + p * (2 * HALF) + 2 * i)
    s = tl.load(T + p * (2 * HALF) + 2 * i + 1)
    if INVERSE:                                     # the conjugate rotation (M:400-401)
        s = -s
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    ok = (h < H)[:, None]
    at = X + row.to(tl.int64) * sr + h[:, None].to(tl.int64) * sh + (D - 2 * HALF) + 2 * i[None, :]
    a = tl.load(at, mask=ok).to(tl.float32)
    b = tl.load(at + 1, mask=ok).to(tl.float32)
    tl.store(at, (a * c[None, :] - b * s[None, :]).to(tl.bfloat16), mask=ok)
    tl.store(at + 1, (a * s[None, :] + b * c[None, :]).to(tl.bfloat16), mask=ok)


def apply(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """M:392-406 in place on bf16 ``x`` at ``positions`` (one value: ``positions[0] + r``, as graphs replay)."""

    if x.dtype != torch.bfloat16 or x.dim() not in (2, 3) or x.stride(-1) != 1:
        raise ValueError("rope.apply: bf16 [rows, D] or [rows, heads, D] with unit stride in D")
    if table.dtype != torch.float32 or table.dim() != 3 or table.shape[2] != 2 or not table.is_contiguous():
        raise ValueError("rope.apply: the table is contiguous fp32 [capacity, rope/2, 2] from tables()")
    rows, half, d = x.shape[0], table.shape[1], x.shape[-1]
    if positions.dim() != 1 or positions.numel() not in (1, rows) or positions.is_floating_point():
        raise ValueError(f"rope.apply: integer positions [1] or [{rows}], got {tuple(positions.shape)}")
    if rows == 0:
        return x
    heads, sh = (x.shape[1], x.stride(1)) if x.dim() == 3 else (1, 0)
    _rope[(rows, triton.cdiv(heads, HEADS))](x, positions, table, x.stride(0), sh, heads, D=d, HALF=half,
                                             PER_ROW=positions.numel() == rows, INVERSE=inverse, HB=HEADS)
    return x
