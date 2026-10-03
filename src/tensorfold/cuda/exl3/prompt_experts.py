"""Routed EXL3 experts for prompt chunks (hundreds to thousands of rows): tensor cores, a width per expert (2, 3 or 4
bits), trellis decoded on the fly, each expert's input Hadamard rotation moved onto its decoded weights
(prompt_experts.cu), so neither dequantized weights nor rotated inputs ever reach memory.

Three launches a chunk of <= CHUNK_ROWS rows: ``route`` (one block: pairs grouped by expert into items of <= 112),
``gate|up`` (+ SwiGLU and the down input's rotation, xd [pairs, I] fp16; also zeroes out), ``down`` (+ svh, routing
weight, red.add into an fp32 out). A slot's output does not depend on the other rows. By default (TF_EXL3_PROMPT_DET)
down adds a row's slots into fp32 out with red.add in arrival order (fastest; a prompt's bits may differ run to run).
TF_EXL3_PROMPT_DET=slots16 makes prompts reproducible: each slot's row stored as fp16 and ``slot_sum16`` adds them in
fp32 in slot order (~4-6% slower prefill); slots = fp32 rows, fixed = int64 fixed-point red.add. Decode windows keep the row-invariant ``experts.routed``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch

from .experts import ACT_F32, Exl3RoutedExperts

PROMPT_K2 = (4, 6, 8)         # 2, 3 and 4 bits a value


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_prompt_experts_v4",
                sources=[str(here / "prompt_experts.cpp"), str(here / "prompt_experts.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], extra_include_paths=[str(here)], verbose=False)


def supported(ex: Exl3RoutedExperts) -> bool:
    """Whether the prompt kernels take this layer: gate and up of an expert at one width, widths 2/3/4 bits."""

    gk, uk, dk = ex.gate_k2.tolist(), ex.up_k2.tolist(), ex.down_k2.tolist()
    return gk == uk and all(k in PROMPT_K2 for k in gk + dk) and ex.width % 128 == 0 and ex.dims % 256 == 0


class PromptScratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, device="cuda") -> None:
        ipm = _ext().item_rows()
        n = rows * slots
        self.rows, self.slots = rows, slots
        self.max_items = n // ipm + min(n, ex.count)
        self.sorted = torch.empty((n,), dtype=torch.int32, device=device)
        self.items = torch.empty((3 * self.max_items,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.xd = torch.empty((n, ex.width), dtype=torch.float16, device=device)
        self.pairs = None            # [n, D] fp32 pair rows of the "slots" mode (on first use)
        self.fix = None              # [rows, D] int64 fixed-point sums of the "fixed" mode (on first use)


ORDER_BY_COUNT = int(__import__("os").environ.get("TF_EXL3_PROMPT_ORDER", "0"))   # 1: busiest experts first (measured neutral)
# How a row's slot outputs are added. fp32 red.add sums them in arrival order - different bits run to run, so a
# multi-chunk prompt's KV (and every later token) was not reproducible. "0" (default, fastest prefill): fp32 red.add -
# a prompt longer than one chunk can differ run to run. "slots16" (reproducible, ~4-6% slower): each slot's row stored
# as fp16, summed in fp32 in slot order (rel 2e-4 vs fp32 sums; 4 Sparks prefill 8K/32K/128K 1000/969/952 tok/s);
# "slots": fp32 rows (944/976/899); "fixed": red.add of 64-bit fixed point (917/944/867, u64 atomics are L2-bound);
# "0": fp32 red.add (1044-1087/1035-1124/996, NOT reproducible run to run)
_DET = __import__("os").environ.get("TF_EXL3_PROMPT_DET", "0").lower()
DETERMINISTIC = _DET in ("1", "fixed", "slots", "slots16")
DET_MODE = {"slots": 2, "slots16": 4}.get(_DET, 3)
NCB_ELEM_FIX = int(__import__("os").environ.get("TF_EXL3_PROMPT_FIX_SLAB_ELEM", "8"))   # slab sizing (8: true int64 width)
CHUNK_ROWS = 4096              # longer calls run in row chunks: the down kernel re-reads xd once a column slab
SLAB_BYTES = 9 << 20           # out columns a down pass keeps in L2 for its red.add (rows x columns x 4 bytes)


def auto_ncb(R: int, D: int, elem: int = 4) -> int:
    """Column blocks (of 256) a down pass: the widest slab of out whose rows stay within SLAB_BYTES."""

    blocks = D // 256
    best = 1
    for d in range(1, blocks + 1):
        if blocks % d == 0 and R * d * 256 * elem <= SLAB_BYTES:
            best = d
    return best


def prompt_routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, ex: Exl3RoutedExperts,
                  out: torch.Tensor | None = None, scratch: PromptScratch | None = None, limit: float = math.inf,
                  act_mode: int = ACT_F32, ncb: int = 0, f16_acc: bool | None = None,
                  _which: int = 3) -> torch.Tensor:
    """out [R, D] = sum over slots of wts * expert(x) (picks outside [0, E) skipped); fp32, or bf16 when ``out`` is.

    x bf16 [R, D] (unit column stride), pick int32 [R, S], wts [R, S]. No host sync; graph-capturable for a fixed R.
    ``f16_acc`` (default: TF_EXL3_PROMPT_F16_ACC=1, else off): the slots add into an fp16 buffer (half the atomic
    traffic, fp16 roundings of the partial sums), converted to out at the end.
    """

    R, S = pick.shape
    det = (DET_MODE if DETERMINISTIC else 0) if not f16_acc else 0
    if out is None:
        out = torch.empty((R, ex.dims), dtype=torch.float32, device=x.device)
    if scratch is None:
        scratch = PromptScratch(ex, min(R, CHUNK_ROWS), S, device=x.device)
    if f16_acc is None:
        import os

        f16_acc = os.environ.get("TF_EXL3_PROMPT_F16_ACC", "0") == "1"
    step = min(scratch.rows, CHUNK_ROWS)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        _prompt_chunk(x[r0:r1], pick[r0:r1], wts[r0:r1], ex, out[r0:r1], scratch, limit, act_mode, ncb, f16_acc,
                      _which, det)
    return out


def _prompt_chunk(x, pick, wts, ex, out, scratch, limit, act_mode, ncb, f16, which, det=False) -> None:
    ext = _ext()
    R, S = pick.shape
    if R > scratch.rows or S != scratch.slots:
        raise ValueError(f"{R} rows x {S} slots but the scratch holds {scratch.rows} x {scratch.slots}")
    pick = pick.contiguous()
    wts = wts.float().contiguous()
    items = min(scratch.max_items, R * S // ext.item_rows() + min(R * S, ex.count))
    if det == 3:                          # red.add of 64-bit fixed point: the same sums in any order
        if scratch.fix is None or scratch.fix.shape[0] < R:
            scratch.fix = torch.empty((scratch.rows, ex.dims), dtype=torch.int64, device=x.device)
        acc = scratch.fix[:R]             # zeroed by the gate|up kernel, then summed into by the down kernel
        ext.route(pick, ex.count, scratch.sorted, scratch.items, scratch.count, items, ORDER_BY_COUNT)
        ext.experts(x, scratch.sorted[:R * S], scratch.items, scratch.count, ex.gate_ptr, ex.up_ptr, ex.down_ptr,
                    ex.gate_k2, ex.down_k2, ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d, wts,
                    scratch.xd, acc, ex.dims, ex.width, ex.gu_stride or ex.width // 16, S, items, float(limit),
                    act_mode, ex.cb, ncb or auto_ncb(R, ex.dims, NCB_ELEM_FIX), which, 3)
        if which & 2:
            dst = out if out.dtype == torch.float32 and out.is_contiguous() else \
                torch.empty((R, ex.dims), dtype=torch.float32, device=x.device)
            ext.fix_to_float(acc, dst)
            if dst is not out:
                out.copy_(dst)
        return
    if det in (2, 4):                     # each pair into its own row (fp32 / fp16), then summed in slot order
        pdt = torch.float16 if det == 4 else torch.float32
        if scratch.pairs is None or scratch.pairs.dtype != pdt:
            scratch.pairs = torch.empty((scratch.rows * scratch.slots, ex.dims), dtype=pdt, device=x.device)
        pairs = scratch.pairs[:R * S]
        ext.route(pick, ex.count, scratch.sorted, scratch.items, scratch.count, items, ORDER_BY_COUNT)
        ext.experts(x, scratch.sorted[:R * S], scratch.items, scratch.count, ex.gate_ptr, ex.up_ptr, ex.down_ptr,
                    ex.gate_k2, ex.down_k2, ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d, wts,
                    scratch.xd, pairs, ex.dims, ex.width, ex.gu_stride or ex.width // 16, S, items, float(limit),
                    act_mode, ex.cb, ncb or auto_ncb(R, ex.dims, 4), which, det)
        if which & 2:
            dst = out if out.dtype == torch.float32 and out.is_contiguous() else \
                torch.empty((R, ex.dims), dtype=torch.float32, device=x.device)
            (ext.slot_sum16 if det == 4 else ext.slot_sum)(pairs, pick, dst, ex.count)
            if dst is not out:
                out.copy_(dst)
        return
    adt = torch.float16 if f16 else torch.float32
    if out.dtype == adt and out.is_contiguous():
        acc = out                         # zeroed by the gate|up kernel, then summed into by the down kernel
    else:
        acc = torch.empty((R, ex.dims), dtype=adt, device=x.device)
    ext.route(pick, ex.count, scratch.sorted, scratch.items, scratch.count, items, ORDER_BY_COUNT)
    ext.experts(x, scratch.sorted[:R * S], scratch.items, scratch.count, ex.gate_ptr, ex.up_ptr, ex.down_ptr,
                ex.gate_k2, ex.down_k2, ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d, wts,
                scratch.xd, acc, ex.dims, ex.width, ex.gu_stride or ex.width // 16, S, items, float(limit),
                act_mode, ex.cb, ncb or auto_ncb(R, ex.dims, 2 if f16 else 4), which, int(f16))
    if acc is not out:
        out.copy_(acc)
