"""Grouped NVFP4 experts on ``tensorfold.cuda.experts``' plan: a (row, slot) pair's bits never depend on the others."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda import experts as grouped

COLS = 32                 # output columns a block
WORDS = 144               # int32 a (32 columns, 32 inputs) block: 128 code words, then 16 of e4m3 scales
# pairs of one expert a prompt item: the staged kernel takes 64, the decode kernel 16; a pair's bits do not change
PREFILL_TILE = int(__import__("os").environ.get("TF_NVFP4_PROMPT_TILE", "64"))
STAGED_ROWS = int(__import__("os").environ.get("TF_NVFP4_STAGED_ROWS", "64"))   # fewer rows stay on the decode kernel


def _i32(v: torch.Tensor) -> torch.Tensor:
    """int64 holding 32-bit patterns -> int32 with the same bits."""

    v = v & 0xFFFFFFFF
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)


def _pack(words: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    e, n, k2 = words.shape
    nb, kg = n // COLS, 2 * k2 // 32
    w = words.to(torch.int64)
    codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(e, nb, 4, 8, kg, 2, 4, 4)   # [E, cb, j, gq, g, h, t, q]
    h, q = torch.arange(2, device=w.device), torch.arange(4, device=w.device)
    slot = 2 * h[:, None] + q[None, :] // 2 + 4 * (q[None, :] % 2)                     # input 16h + 4t + q -> slot
    word = (codes << (4 * slot).view(1, 1, 1, 1, 1, 2, 1, 4)).sum(dim=(5, 7))          # [E, cb, j, gq, g, t]
    word = _i32(word.permute(0, 1, 4, 3, 5, 2).reshape(e, nb, kg, 128))                 # lane gq * 4 + t, tile j
    sc = scales.contiguous().view(torch.uint8).view(e, nb, 4, 4, 2, kg, 2)              # [E, cb, j, t, c, g, h]
    sc = sc.permute(0, 1, 5, 3, 6, 2, 4).contiguous().view(e, nb, kg, 64).view(torch.int32)
    return torch.cat([word, sc], dim=-1)


def pack(words: torch.Tensor, scales: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """NVFP4 words [E, N, K/2] (low nibble first) and e4m3 scale bytes [E, N, K/16] -> blocks [E, N/32, K/32, 144]."""

    e, n, k2 = words.shape
    k = 2 * k2
    if n % COLS or k % 32 or tuple(scales.shape) != (e, n, k // 16):
        raise ValueError(f"NVFP4 experts [{e}, {n}, {k}] with scales {tuple(scales.shape)} do not pack")
    out = torch.empty((e, n // COLS, k // 32, WORDS), dtype=torch.int32, device=words.device)
    for e0 in range(0, e, chunk):
        out[e0:e0 + chunk] = _pack(words[e0:e0 + chunk], scales[e0:e0 + chunk])
    return out


@dataclass
class Experts4:
    """One layer's routed experts: gate and up (SwiGLU) and down blocks, each (expert, matrix) with its fp32 scale."""

    up: torch.Tensor          # [E, NI/32, D/32, 2, 144] int32
    down: torch.Tensor        # [E, D/32, NI/32, 1, 144]
    up_scale: torch.Tensor    # [E, 2] fp32 (gate, up)
    down_scale: torch.Tensor  # [E, 1]
    width: int                # NI
    dims: int                 # D
    limit: float = 0.0

    @property
    def count(self) -> int:
        return int(self.up.shape[0])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.up, self.down, self.up_scale, self.down_scale))


def make(gate: tuple, up: tuple, down: tuple, *, limit: float = 0.0) -> Experts4:
    """Each of gate, up, down: (words [E, N, K/2] uint8, e4m3 scales [E, N, K/16], per-expert scales [E] fp32)."""

    width, dims = int(gate[0].shape[1]), int(gate[0].shape[2]) * 2
    u = torch.stack([pack(gate[0], gate[1]), pack(up[0], up[1])], dim=3)
    d = pack(down[0], down[1]).unsqueeze(3)
    us = torch.stack([gate[2], up[2]], dim=1).to(torch.float32).contiguous()
    return Experts4(u, d, us, down[2].to(torch.float32).reshape(-1, 1).contiguous(), width, dims, float(limit))


def _run(epi: int, x: torch.Tensor, slots: int, w: torch.Tensor, scale: torch.Tensor, kg: int, nb: int,
         plan: grouped.Plan, out: torch.Tensor, n: int, limit: float, skip: int, rows: int) -> None:
    from .linear import _ext

    units = grouped.max_items(rows * plan.slots, plan.experts, plan.tile) * nb
    rt = 4 if plan.tile >= 64 and nb % 4 == 0 and rows >= STAGED_ROWS else 1   # 4 column blocks a CTA
    _ext().experts(epi, x, x.stride(0), slots, w, scale, kg, nb, plan.items, plan.counts, plan.members, out, n,
                   limit, skip, units, rt)


def gate_up(x: torch.Tensor, ex: Experts4, plan: grouped.Plan, out: torch.Tensor, rows: int, skip: int = -1) -> None:
    """x [R, D] bf16 -> out [R * slots, NI] bf16, each routed pair's SwiGLU; pairs of expert ``skip`` untouched."""

    _run(2, x, plan.slots, ex.up, ex.up_scale, ex.dims // 32, ex.width // COLS, plan, out, ex.width, ex.limit, skip,
         rows)


def down(act: torch.Tensor, ex: Experts4, plan: grouped.Plan, out: torch.Tensor, rows: int, skip: int = -1) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D] in ``out``'s dtype; pairs of expert ``skip`` untouched."""

    _run(0 if out.dtype == torch.float32 else 3, act, 0, ex.down, ex.down_scale, ex.width // 32, ex.dims // COLS,
         plan, out, ex.dims, 0.0, skip, rows)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e, n, k = x.shape
    g = (x.abs().amax(dim=(1, 2)) / (6.0 * 448.0)).clamp_min(1e-30)                    # [E]
    blocks = x.view(e, n, k // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / g[:, None, None]).clamp(max=448.0).to(torch.float8_e4m3fn)
    step = (s.float() * g[:, None, None])[..., None]
    v = torch.where(step > 0, blocks / step.clamp_min(1e-30), torch.zeros_like(blocks))
    mags = torch.tensor(E2M1, device=x.device)
    code = ((v.abs()[..., None] - mags).abs().argmin(-1) + 8 * (v < 0).to(torch.int64)).view(e, n, k).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), s.view(torch.uint8).contiguous(), g


def quantize(w: torch.Tensor, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [E, N, K] -> NVFP4 (words, e4m3 scales, per-expert scales) by ModelOpt's recipe, for draft-only weights."""

    parts = [_quantize(w[e0:e0 + chunk].to(torch.float32)) for e0 in range(0, w.shape[0], chunk)]
    return tuple(torch.cat(p) for p in zip(*parts))


def dense(ex: Experts4, e: int, which: str) -> torch.Tensor:
    """Expert ``e``'s gate, up or down weight [N, K] fp32 from its blocks: code x e4m3 x scale (a reference)."""

    m = {"gate": 0, "up": 1, "down": 0}[which]
    blocks = (ex.down if which == "down" else ex.up)[e, :, :, m]                         # [nb, kg, 144]
    scale = float((ex.down_scale if which == "down" else ex.up_scale)[e, m])
    nb, kg, _ = blocks.shape
    words = (blocks[..., :128].to(torch.int64) & 0xFFFFFFFF).view(nb, kg, 8, 4, 4)     # [cb, g, gq, t, j]
    h, q = torch.arange(2, device=blocks.device), torch.arange(4, device=blocks.device)
    slot = 2 * h[:, None] + q[None, :] // 2 + 4 * (q[None, :] % 2)
    codes = (words[..., None, None] >> (4 * slot)) & 0xF                                 # [cb, g, gq, t, j, h, q]
    codes = codes.permute(0, 4, 2, 1, 5, 3, 6).reshape(nb * COLS, kg * 32)
    mags = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=blocks.device)
    sc = blocks[..., 128:].contiguous().view(torch.uint8).view(nb, kg, 4, 2, 4, 2)      # [cb, g, t, h, j, c]
    sc = sc.permute(0, 4, 2, 5, 1, 3).reshape(nb * COLS, kg * 2).contiguous()
    return mags[codes] * sc.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) * scale
