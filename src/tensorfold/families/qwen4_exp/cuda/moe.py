"""Flash Next's MoE, top 10 of 512 on fp32 logits (ties to the lower id); each row routes and runs on its own."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped

from . import qmm


@triton.jit
def _router(X, W, OUT, M, x_stride, D: tl.constexpr, NE: tl.constexpr, BM: tl.constexpr,
            BLOCK_E: tl.constexpr, BK: tl.constexpr):
    """OUT[m, e] = fp32 x[m] . w[e] (bf16 inputs, tensor cores, K in BK steps in order)."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    e_ok = re < NE
    acc = tl.zeros((BM, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + re[:, None] * D + (k0 + rk)[None, :], mask=e_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rm[:, None] * NE + re[None, :], acc, mask=m_ok[:, None] & e_ok[None, :])


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [R, D] bf16, rows [E + 1, D] bf16 (router rows, then the shared expert's gate row) -> [R, E + 1] fp32."""

    m, d = x.shape
    ne = rows.shape[0]
    if out is None:
        out = torch.empty((m, ne), dtype=torch.float32, device=x.device)
    bm = qmm.bucket(m)
    if bm == 16:
        be, bk, stages = 32, 256, 4
    else:
        be, bk, stages = 64, 64, 3               # Smaller K tiles keep prefill within shared-memory limits.
    grid = (triton.cdiv(m, bm), triton.cdiv(ne, be))
    _router[grid](x, rows, out, m, x.stride(0), D=d, NE=ne, BM=bm, BLOCK_E=be, BK=bk, num_warps=4, num_stages=stages)
    return out


@triton.jit
def _topk_rows(L, PICK, WTS, NE: tl.constexpr, NL: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
               BLOCK: tl.constexpr, SLOTP: tl.constexpr):
    """Program r: row r's TOPK experts by fp32 logit (largest first, the lower id among equal logits), weights exp(l_k - l_0) / sum (fp32, rounded to bf16), then the shared expert (id NE) as slot TOPK with weight bf16(sigmoid(bf16(shared gate logit)))."""

    r = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    v = tl.load(L + r * NL + ar, mask=ar < NE, other=float("-inf"))
    top = tl.max(v, axis=0)
    total = 0.0
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    exs = tl.zeros((SLOTP,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        m = tl.max(v, axis=0)
        idx = tl.min(tl.where(v == m, ar, BLOCK), axis=0)
        ex = tl.exp(m - top)
        picks = tl.where(ak == k, idx, picks)
        exs = tl.where(ak == k, ex, exs)
        total += ex
        v = tl.where(ar == idx, float("-inf"), v)
    w = (exs / total).to(tl.bfloat16).to(tl.float32)
    sg = tl.load(L + r * NL + NE).to(tl.bfloat16).to(tl.float32)
    sgw = (1.0 / (1.0 + tl.exp(-sg))).to(tl.bfloat16).to(tl.float32)
    picks = tl.where(ak == TOPK, NE, picks)
    w = tl.where(ak == TOPK, sgw, w)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, w, mask=ak < SLOTS)


class MoEBuffers:
    """Static scratch for up to ``rows`` rows (CUDA-graph safe); ``prefill`` picks the experts' prefill arithmetic."""

    def __init__(self, rows: int, cfg, device: torch.device | str, *, prefill: bool = False) -> None:
        slots = cfg.num_experts_per_tok + 1
        self.rows, self.slots = rows, slots
        self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
        self.plan = grouped.Plan(rows, slots, cfg.num_experts + 1, device, prefill=prefill)
        self.act = torch.empty((rows, slots, cfg.moe_intermediate_size), dtype=torch.bfloat16, device=device)
        self.y = torch.empty((rows, slots, cfg.hidden_size), dtype=torch.bfloat16 if prefill else torch.float32,
                             device=device)


def select_rows(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights, rows in parallel (buf.pick, buf.wts); EXL3 experts group themselves in their kernel."""

    rows = logits.shape[0]
    _topk_rows[(rows,)](logits, buf.pick, buf.wts, NE=experts, NL=logits.shape[1], TOPK=top_k, SLOTS=top_k + 1,
                        BLOCK=triton.next_power_of_2(experts + 1), SLOTP=triton.next_power_of_2(top_k + 1), num_warps=4)


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights (rows in parallel), then the (row, slot) pairs grouped by expert."""

    select_rows(logits, buf, top_k, experts)
    grouped.route(buf.pick[:logits.shape[0]], buf.plan)


def moe(x: torch.Tensor, router_rows: torch.Tensor, ex: grouped.Experts, buf: MoEBuffers, top_k: int,
        experts: int) -> MoEBuffers:
    """Route x [R, D] and run its experts into buf.y [R, k + 1, D] (bf16 in prefill); slot k is the shared expert."""

    rows = x.shape[0]
    router(x, router_rows, buf.logits[:rows])
    select(buf.logits[:rows], buf, top_k, experts)
    grouped.gate_up(x, ex, buf.plan, buf.act.view(-1, ex.width), rows)
    grouped.down(buf.act.view(-1, ex.width), ex, buf.plan, buf.y.view(-1, ex.dims), rows)
    return buf
