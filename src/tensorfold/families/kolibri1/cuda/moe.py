"""Kolibri 1's MoE: top-k over logits plus the expert bias, each weighted by sigmoid(logit), then the shared expert."""
# Ties go to the lower id; the shared expert rides in the grouped table as id E with weight 1.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda import moe as shared
from tensorfold.cuda.fp8 import experts as fp8x


@triton.jit
def _topk(L, BIAS, PICK, WTS, NE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr,
          SLOTP: tl.constexpr):
    r = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    logit = tl.load(L + r * NE + ar, mask=ar < NE, other=float("-inf"))
    v = logit + tl.load(BIAS + ar, mask=ar < NE, other=0.0)
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    w = tl.zeros((SLOTP,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        top = tl.max(v, axis=0)
        idx = tl.min(tl.where(v == top, ar, BLOCK), axis=0)
        lg = tl.sum(tl.where(ar == idx, logit, 0.0), axis=0)
        picks = tl.where(ak == k, idx, picks)
        w = tl.where(ak == k, 1.0 / (1.0 + tl.exp(-lg)), w)
        v = tl.where(ar == idx, float("-inf"), v)
    picks = tl.where(ak == TOPK, NE, picks)
    w = tl.where(ak == TOPK, 1.0, w)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, w, mask=ak < SLOTS)


class Buffers:
    """Scratch for up to ``rows`` rows."""

    def __init__(self, rows: int, experts: int, top_k: int, width: int, dims: int, device, *, prefill: bool) -> None:
        slots = top_k + 1
        self.rows, self.slots = rows, slots
        self.logits = torch.empty((rows, experts), dtype=torch.float32, device=device)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
        self.plan = grouped.Plan(rows, slots, experts + 1, device, prefill=prefill)
        self.act = torch.empty((rows * slots, width), dtype=torch.bfloat16, device=device)
        self.y = torch.empty((rows, slots, dims), dtype=torch.bfloat16 if prefill else torch.float32, device=device)


_scratch: dict[tuple, Buffers] = {}


def run(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor, ex: fp8x.Experts8, top_k: int, *,
        prefill: bool = False) -> torch.Tensor:
    """x [R, D] bf16 -> [R, D] bf16: its top-k experts by sigmoid weight, plus the ungated shared expert."""

    rows = x.shape[0]
    experts = router.shape[0]
    size = 1 << max(4, (rows - 1).bit_length())
    key = (size, experts, top_k, ex.width, ex.dims, prefill, x.device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = Buffers(size, experts, top_k, ex.width, ex.dims, x.device, prefill=prefill)
    x = x.contiguous()
    shared.router(x, router, buf.logits[:rows])
    _topk[(rows,)](buf.logits, bias, buf.pick, buf.wts, NE=experts, TOPK=top_k, SLOTS=top_k + 1,
                   BLOCK=triton.next_power_of_2(experts), SLOTP=triton.next_power_of_2(top_k + 1), num_warps=4)
    grouped.route(buf.pick[:rows], buf.plan, fp8x.PROMPT_TILE if prefill else 16)
    fp8x.gate_up(x, ex, buf.plan, buf.act[:rows * buf.slots], rows)
    fp8x.down(buf.act[:rows * buf.slots], ex, buf.plan, buf.y[:rows].view(-1, ex.dims), rows)
    return shared.combine(buf.y[:rows], buf.wts[:rows])
