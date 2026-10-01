"""Grouped NVFP4 experts on ``tensorfold.cuda.experts``' plan: a (row, slot) pair's bits never depend on the others."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda import experts as grouped

COLS = 32                 # output columns a block
WORDS = 144               # int32 a (32 columns, 32 inputs) block: 128 code words, then 16 of e4m3 scales
PREFILL_TILE = 16         # this kernel's prompt item: 64 ran 1.53x slower on Flash Next's routed prompts


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
    _ext().experts(epi, x, x.stride(0), slots, w, scale, kg, nb, plan.items, plan.counts, plan.members, out, n,
                   limit, skip, units)


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


@dataclass
class ExpertsCk:
    """One layer's routed experts in the checkpoint's own math (FP4 x FP4): each projection's experts stacked in the
    lane matmul's words and block scales, each expert's output factor, and the input scales rows quantize under."""

    gate: tuple               # (words, block scales): the lane matmul's, lane-major a 32-column half (_stacked)
    up: tuple
    down: tuple
    alpha: torch.Tensor       # [3, E] fp32: input scale x weight scale (gate, up, down)
    down_inv: torch.Tensor    # [E] fp32: 1 / down's input scale, what SiLU(gate) * up is quantized under
    act: float                # gate and up's input scale (one for the layer's rows)
    width: int                # NI
    dims: int                 # D

    @property
    def count(self) -> int:
        return int(self.alpha.shape[1])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in (self.gate, self.up, self.down) for t in pair) + \
            self.alpha.numel() * 4 + self.down_inv.numel() * 4


def _stacked(words: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K/2] codes and [E, N, K/16] e4m3 -> words [E*N/64, K/64, 2, 32, 4, 2] int32 (half, lane, n8 tile) and
    scales [E*N/64, K/64, 2, 8, 4, 4] uint8 (half, column of the tile, tile): a lane's step in two and one loads."""

    from . import checkpoint

    e, n, k2 = words.shape
    k = 2 * k2
    if n % 64 or k % 64:
        raise ValueError(f"NVFP4 experts [{e}, {n}, {k}]: the FP4 mma takes whole 64-column, 64-input tiles")
    t, kg = e * n // 64, k // 64
    w = checkpoint.pack4(words.reshape(e * n, k2), e * n)                       # [T, KG, 8 tiles, 32 lanes, 2]
    w = w.view(t, kg, 2, 4, 32, 2).permute(0, 1, 2, 4, 3, 5).contiguous()       # [T, KG, half, lane, tile, 2]
    bs = scales.contiguous().view(torch.uint8).reshape(t, 64, kg, 4).permute(0, 2, 1, 3)   # [T, KG, column, 4]
    bs = bs.reshape(t, kg, 2, 4, 8, 4).permute(0, 1, 2, 4, 3, 5).contiguous()   # [T, KG, half, column, tile, 4]
    return w, bs


def make_ck(gate: tuple, up: tuple, down: tuple, acts: tuple) -> ExpertsCk:
    """gate, up, down: (words [E, N, K/2] uint8, e4m3 scales [E, N, K/16], per-expert scales [E] fp32); acts: each
    projection's per-expert input scales [E]. Rows enter gate|up once, under the largest of their input scales (the
    checkpoints we know store one value for every expert), and leave for down under each expert's own."""

    f32 = [torch.as_tensor(a, dtype=torch.float32).reshape(-1) for a in acts]
    act = torch.maximum(f32[0].max(), f32[1].max())
    alpha = torch.stack([act * gate[2].to(torch.float32).reshape(-1), act * up[2].to(torch.float32).reshape(-1),
                         f32[2].to(gate[2].device) * down[2].to(torch.float32).reshape(-1)]).contiguous()
    inv = (torch.ones_like(f32[2]) / f32[2]).to(gate[0].device).contiguous()
    return ExpertsCk(_stacked(gate[0], gate[1]), _stacked(up[0], up[1]), _stacked(down[0], down[1]), alpha, inv,
                     float(act), int(gate[0].shape[1]), int(gate[0].shape[2]) * 2)


def gate_up_ck(x: torch.Tensor, ex: ExpertsCk, plan: grouped.Plan, rows: int,
               skip: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, D] bf16 -> each routed pair's SiLU(gate) * up as down's NVFP4 rows: codes [P, NI/2], scales [NI/64, ppad,
    4]; rows quantized once under the layer's input scale, pairs of expert ``skip`` unwritten."""

    from . import checkpoint

    xq = checkpoint.quant4(x[:rows], ex.act)
    pairs = rows * plan.slots
    codes = torch.empty((pairs, ex.width // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((ex.width // 64, -(-pairs // 64) * 64, 4), dtype=torch.uint8, device=x.device)
    units = grouped.max_items(pairs, plan.experts, plan.tile) * (ex.width // 32)        # 32 columns a unit
    checkpoint._ext().experts_gu_ck(xq.codes, xq.scales, plan.slots, ex.gate[0], ex.gate[1], ex.alpha[0], ex.up[0],
                                    ex.up[1], ex.alpha[1], ex.down_inv, ex.dims, ex.width, plan.items, plan.counts,
                                    plan.members[:pairs], codes, scales, skip, units)
    return codes, scales


def down_ck(rows4: tuple[torch.Tensor, torch.Tensor], ex: ExpertsCk, plan: grouped.Plan, out: torch.Tensor,
            rows: int, skip: int = -1) -> None:
    """Down's NVFP4 rows a pair -> out [R * slots, D] in ``out``'s dtype (fp32 or bf16); pairs of ``skip`` untouched."""

    from . import checkpoint

    pairs = rows * plan.slots
    units = grouped.max_items(pairs, plan.experts, plan.tile) * (ex.dims // 32)
    checkpoint._ext().experts_down_ck(rows4[0], rows4[1], ex.down[0], ex.down[1], ex.alpha[2], ex.width, ex.dims,
                                      plan.items, plan.counts, plan.members[:pairs], out, skip, units)
