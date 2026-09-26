"""Flash Next's MoE on CUDA: fp32 router logits, top 10 of 512 (ties to the lower id), the distinct experts
of a window with the rows that picked each, and the grouped expert projections (``qmm.moe_gateup`` /
``qmm.moe_down``).

A row's routing and expert outputs never depend on the other rows: the router is one tensor-core dot per
(row, expert) over K in a fixed order, the selection runs on each row's own logits, and the expert
projections compute each (row, expert) pair with the same arithmetic whatever the rows beside it. The
combine (``glue.hc_writeback``) adds a row's ten weighted slots in slot order, then the shared expert.

The shared expert is stored as expert ``E`` (512) of the same tables, so it rides in the grouped kernels
as the eleventh slot of every row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import qmm
from .qmm import Experts, Group


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
        # decode windows, tuned on GB10: 32 experts a program, K in steps of 256, 4 stages
        be, bk, stages = 32, 256, 4
    else:
        be, bk, stages = 64, 64, 3               # prefill tiles (the tuned tile needs too much shared memory)
    grid = (triton.cdiv(m, bm), triton.cdiv(ne, be))
    _router[grid](x, rows, out, m, x.stride(0), D=d, NE=ne, BM=bm, BLOCK_E=be, BK=bk, num_warps=4, num_stages=stages)
    return out


@triton.jit
def _topk_rows(L, PICK, WTS, NE: tl.constexpr, NL: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
               BLOCK: tl.constexpr, SLOTP: tl.constexpr):
    """Program r: row r's TOPK experts by fp32 logit (largest first, the lower id among equal logits), weights
    exp(l_k - l_0) / sum (fp32, rounded to bf16), then the shared expert (id NE) as slot TOPK with weight
    bf16(sigmoid(bf16(shared gate logit)))."""

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


@triton.jit
def _group(PICK, UIDS, UCOUNT, UMEM, R, NE: tl.constexpr, SLOTS: tl.constexpr, MAXU: tl.constexpr,
           MAXM: tl.constexpr, BLOCK: tl.constexpr):
    """One program: the distinct experts of all rows in increasing id order: UIDS[u], UMEM[u][j] = row * 32 +
    slot of the j-th row (in row order) that picked it (-1 after the last), UCOUNT[0] = their number."""

    ar = tl.arange(0, BLOCK)
    counts = tl.zeros((BLOCK,), dtype=tl.int32)
    for r in range(R):
        for k in tl.static_range(SLOTS):
            e = tl.load(PICK + r * SLOTS + k)
            counts += tl.where(ar == e, 1, 0)
    used = counts > 0
    place = tl.cumsum(used.to(tl.int32), axis=0) - 1
    n_used = tl.sum(used.to(tl.int32), axis=0)
    tl.store(UIDS + place, ar, mask=used)
    tl.store(UCOUNT, n_used)
    filled = tl.zeros((BLOCK,), dtype=tl.int32)
    for r in range(R):
        for k in tl.static_range(SLOTS):
            e = tl.load(PICK + r * SLOTS + k)
            hit = ar == e
            tl.store(UMEM + place * MAXM + filled, r * 32 + k, mask=hit)
            filled += tl.where(hit, 1, 0)
    for j in range(MAXM):
        tl.store(UMEM + place * MAXM + j, -1, mask=used & (filled <= j))
    tail = n_used + ar
    for j in range(MAXM):
        tl.store(UMEM + tail * MAXM + j, -1, mask=tail < MAXU)


class MoEBuffers:
    """Per-row-count scratch for the MoE (static, so a step can be captured in a CUDA graph)."""

    def __init__(self, rows: int, cfg, device: torch.device | str) -> None:
        k = cfg.num_experts_per_tok
        slots = k + 1
        self.rows = rows
        self.slots = slots
        self.maxu = min(rows * k, cfg.num_experts) + 1
        self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
        self.group = Group(torch.zeros((self.maxu,), dtype=torch.int32, device=device),
                           torch.zeros((1,), dtype=torch.int32, device=device),
                           torch.full((self.maxu, rows), -1, dtype=torch.int32, device=device))
        self.act = torch.empty((rows, slots, cfg.moe_intermediate_size), dtype=torch.bfloat16, device=device)
        self.axs = torch.empty((rows, slots, cfg.moe_intermediate_size // 32), dtype=torch.float32, device=device)
        self.y = torch.empty((rows, slots, cfg.hidden_size), dtype=torch.float32, device=device)


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights (rows in parallel), then the window's distinct experts and members."""

    rows = logits.shape[0]
    block = triton.next_power_of_2(experts + 1)
    _topk_rows[(rows,)](logits, buf.pick, buf.wts, NE=experts, NL=logits.shape[1], TOPK=top_k, SLOTS=top_k + 1,
                        BLOCK=block, SLOTP=triton.next_power_of_2(top_k + 1), num_warps=4)
    _group[(1,)](buf.pick, buf.group.ids, buf.group.count, buf.group.members, rows, NE=experts, SLOTS=top_k + 1,
                 MAXU=buf.maxu, MAXM=buf.group.members.shape[1], BLOCK=block, num_warps=8)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: Experts, buf: MoEBuffers, cfg,
        *, bm: int = 16) -> MoEBuffers:
    """Route rows x [R, D] (with their 32-group sums) and run their experts: buf.y [R, k + 1, D] fp32 (slot k:
    the shared expert), buf.wts [R, k + 1] (routed weights, then the shared gate)."""

    router(x, router_rows, buf.logits)
    select(buf.logits, buf, cfg.num_experts_per_tok, cfg.num_experts)
    qmm.moe_gateup(x, xs, ex, buf.group, buf.act, buf.axs, bm=bm)
    qmm.moe_down(buf.act, buf.axs, ex, buf.group, buf.y, bm=bm)
    return buf
