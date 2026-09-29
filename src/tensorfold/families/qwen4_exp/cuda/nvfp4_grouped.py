"""The NVFP4 MoE's flat grouped step: one launch a projection, the routing plan read on the device.

``nvfp4_moe``'s first form looped over the plan's items in Python, reading ``items``/``counts`` back to the
host — a sync a layer, and a host read a CUDA graph capture rejects, so the whole MoE step stayed out of the
captured decode (the eager path cost ~4.7 tok/s on Spark against the >30 tok/s ``main``'s grouped experts
hold). This module is the flat form that docstring promised: the same per-row arithmetic as
``nvfp4.matmul``'s packed branch, with the plan's item list, the member gather and the member scatter all
resolved inside the kernels.

Layouts, straight from ``tensorfold.cuda.experts``:

    plan.items    int32 [max_items, 3]   (expert, first, count), one item an expert
    plan.counts   int32 [2]              (0) items the plan wrote, (1) distinct experts
    plan.members  int32 [pairs]          flat ``row * slots + slot`` indices of the engine's buffers

``moe.select`` groups every slot, shared expert included. A stacked FP4 grid holds only the routed experts,
so the kernels return on an item whose expert id is past that count; the shared expert's slot is filled by
``nvfp4_moe`` from its own BF16 tables. An item holds
at most ``PREFILL_TILE`` (64) pairs, so one block covers every row of an item and the grid is fixed
(``plan.items.shape[0]`` items) whatever the routing says — the property capture needs.

A member index is where the row's output goes; the row's *input* is that same index for the down step
(``act``'s flat rows) and ``member // slots`` for gate/up (``x``'s rows have the slot axis flattened away).

Both kernels take one K pass: the grouped grid already carries hundreds of items x the column tiles, so a
split K would only add partials to sum. The accumulation is therefore one fp32 add a quantization block —
the dequantize reference's own order — and gate/up round each matmul to bf16 before the silu, exactly as
``gateup_rows`` did.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nvfp4 import BN, _bf16_widen, _e2m1_pattern, _e4m3_value

TILE = tl.constexpr(64)   # pairs an item holds at most (tensorfold.cuda.experts' PREFILL_TILE)


@triton.jit
def _item(ITEMS, NITEMS, pid):
    """(expert, first, count) of item ``pid``, zeros past the plan's own count."""

    live = pid < tl.load(NITEMS)
    e = tl.load(ITEMS + pid * 3, mask=live, other=0)
    first = tl.load(ITEMS + pid * 3 + 1, mask=live, other=0)
    count = tl.load(ITEMS + pid * 3 + 2, mask=live, other=0)
    return e, first, count


@triton.jit
def _kpass(X, src, rows_ok, x_stride, tile, S, s2, N: tl.constexpr, PER: tl.constexpr, SBN: tl.constexpr,
           BLOCK_N: tl.constexpr, local, soff, n_ok, packed: tl.constexpr):
    """One column tile's K pass: a 16-input quantization block is 8 stored bytes, times the block's scale."""

    r16 = tl.arange(0, 16)
    acc = tl.zeros((TILE, BLOCK_N), dtype=tl.float32)
    # GB10 faults on OOB addresses even when the load is masked; keep inactive lanes on row 0.
    src_a = tl.where(rows_ok, src, 0)
    for b in range(PER):
        kb = b // 4
        row0 = (b % 4) * 16
        x = tl.load(X + src_a[:, None] * x_stride + (b * 16 + r16)[None, :], mask=rows_ok[:, None], other=0.0)
        if packed:
            w8 = tl.load(tile + kb * (32 * SBN) + (row0 // 2 + r16 // 2)[:, None] * SBN + local[None, :],
                         mask=n_ok[None, :], other=0)
            code = tl.where((r16 % 2)[:, None] == 0, w8 & 0xF, w8 >> 4).to(tl.int32)
            wv = _bf16_widen(_e2m1_pattern(code)).to(tl.bfloat16)
        else:
            wbits = tl.load(tile + kb * (64 * SBN) + (row0 + r16)[:, None] * SBN + local[None, :],
                            mask=n_ok[None, :], other=0)
            wv = _bf16_widen(wbits).to(tl.bfloat16)
        p = tl.dot(x, wv)
        if packed:
            s = _e4m3_value(tl.load(S + b * N + soff, mask=n_ok, other=0).to(tl.int32)) * s2
        else:
            s = tl.load(S + b * N + soff, mask=n_ok, other=0.0)
        acc += p * s[None, :]
    return acc


@triton.jit
def _gateup_grouped(X, GU, GS, GS2, ITEMS, NITEMS, MEMBERS, ACT, x_stride, slots,
                    NI: tl.constexpr, K: tl.constexpr, SBN: tl.constexpr, BLOCK_N: tl.constexpr,
                    PACKED: tl.constexpr, STACKED: tl.constexpr, EXPERTS: tl.constexpr):
    """Every item's gate and up rows, at its members, as ``silu(bf16(x @ gate.T)) * bf16(x @ up.T)``."""

    pid = tl.program_id(0)                 # the buffer is reused across steps: an item past the plan's own
    if pid >= tl.load(NITEMS):             # count can still hold a previous step's count, so it stays out
        return
    e, first, count = _item(ITEMS, NITEMS, pid)
    if count == 0:
        return
    # select() puts the shared expert (id EXPERTS) in the last slot; that expert rides the BF16 tables
    # outside this kernel, so a stacked grid of EXPERTS slabs must not index it.
    if STACKED and e >= EXPERTS:
        return
    N2: tl.constexpr = 2 * NI
    PER: tl.constexpr = K // 16
    SUB: tl.constexpr = SBN // BLOCK_N
    # Packed tiles store K/2 bytes a row; pattern (MTP) tables store K uint16 codes a row.
    KSTRIDE: tl.constexpr = K // 2 if PACKED else K
    TG: tl.constexpr = (K // 64) * (32 if PACKED else 64) * SBN
    pid_n = tl.program_id(1)
    rm = tl.arange(0, TILE)
    rows_ok = rm < count
    # Clamp: members is only pairs long; masked lanes with first+rm past that fault on GB10.
    rm_a = tl.where(rows_ok, rm, 0)
    mrow = tl.load(MEMBERS + first + rm_a, mask=rows_ok, other=0)
    src = mrow // slots
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    base_w = GU + (e * N2 * KSTRIDE if STACKED else 0)
    base_s = GS + (e * PER * N2 if STACKED else 0)
    base_2 = GS2 + (e * N2 if STACKED else 0)
    n_ok = rn < NI
    rn_a = tl.where(n_ok, rn, 0)
    s2g = tl.load(base_2 + rn_a, mask=n_ok, other=1.0)
    s2u = tl.load(base_2 + NI + rn_a, mask=n_ok, other=1.0)
    tile_g = base_w + (pid_n // SUB) * TG
    tile_u = tile_g + (NI // SBN) * TG
    g = _kpass(X, src, rows_ok, x_stride, tile_g, base_s, s2g, N2, PER, SBN, BLOCK_N, local, rn_a, n_ok, PACKED)
    u = _kpass(X, src, rows_ok, x_stride, tile_u, base_s, s2u, N2, PER, SBN, BLOCK_N, local, NI + rn_a, n_ok, PACKED)
    gf = g.to(tl.bfloat16).to(tl.float32)
    uf = u.to(tl.bfloat16).to(tl.float32)
    val = ((gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16).to(tl.float32) * uf).to(tl.bfloat16)
    tl.store(ACT + mrow[:, None] * NI + rn_a[None, :], val, mask=rows_ok[:, None] & n_ok[None, :])


@triton.jit
def _down_grouped(ACT, DW, DS, DS2, ITEMS, NITEMS, MEMBERS, Y, act_stride,
                  D: tl.constexpr, K: tl.constexpr, SBN: tl.constexpr, BLOCK_N: tl.constexpr,
                  PACKED: tl.constexpr, STACKED: tl.constexpr, F32: tl.constexpr, EXPERTS: tl.constexpr):
    """Every item's down rows, at its members: ``act @ down.T`` (fp32 sums when the buffer is fp32)."""

    pid = tl.program_id(0)                 # the buffer is reused across steps: an item past the plan's own
    if pid >= tl.load(NITEMS):             # count can still hold a previous step's count, so it stays out
        return
    e, first, count = _item(ITEMS, NITEMS, pid)
    if count == 0:
        return
    if STACKED and e >= EXPERTS:
        return
    PER: tl.constexpr = K // 16
    SUB: tl.constexpr = SBN // BLOCK_N
    KSTRIDE: tl.constexpr = K // 2 if PACKED else K
    TG: tl.constexpr = (K // 64) * (32 if PACKED else 64) * SBN
    pid_n = tl.program_id(1)
    rm = tl.arange(0, TILE)
    rows_ok = rm < count
    rm_a = tl.where(rows_ok, rm, 0)
    mrow = tl.load(MEMBERS + first + rm_a, mask=rows_ok, other=0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    base_w = DW + (e * D * KSTRIDE if STACKED else 0)
    base_s = DS + (e * PER * D if STACKED else 0)
    # Clamp rn: D is a multiple of BLOCK_N for Flash Next, but keep addresses in-bounds on every GPU.
    n_ok = rn < D
    rn_a = tl.where(n_ok, rn, 0)
    s2 = tl.load(DS2 + (e * D if STACKED else 0) + rn_a, mask=n_ok, other=1.0)
    tile = base_w + (pid_n // SUB) * TG
    acc = _kpass(ACT, mrow, rows_ok, act_stride, tile, base_s, s2, D, PER, SBN, BLOCK_N, local, rn_a, n_ok,
                 PACKED)
    val = acc if F32 else acc.to(tl.bfloat16)
    tl.store(Y + mrow[:, None] * D + rn_a[None, :], val, mask=rows_ok[:, None] & n_ok[None, :])


def _grid(plan, n: int, block_n: int) -> tuple[int, int]:
    """One program an item x a column tile: fixed in the plan's capacity, not in what the routing wrote."""

    return (int(plan.items.shape[0]), -(-n // block_n))


def gateup(fp, x: torch.Tensor, act: torch.Tensor, plan, ni: int, slots: int, block_n: int = BN) -> None:
    """The gate/up step for every plan item, written at each member's own ``act`` row."""

    stacked = fp.weight.dim() == 5
    experts = int(fp.weight.shape[0]) if stacked else 1
    if fp.scale2 is None:                              # pattern tables: identity factor, as matmul does
        shape = (experts, fp.n) if stacked else (fp.n,)
        fp.scale2 = torch.ones(shape, dtype=torch.float32, device=x.device)
    elif not fp.scale2.is_contiguous():
        fp.scale2 = fp.scale2.contiguous()
    _gateup_grouped[_grid(plan, ni, block_n)](x, fp.weight, fp.scale, fp.scale2, plan.items, plan.counts[:1],
                                              plan.members, act, x.stride(0), slots, NI=ni, K=fp.k, SBN=BN,
                                              BLOCK_N=block_n, PACKED=fp.packed, STACKED=stacked,
                                              EXPERTS=experts, num_warps=4, num_stages=2)


def down(fp, act: torch.Tensor, y: torch.Tensor, plan, slots: int, block_n: int = BN) -> None:
    """The down step for every plan item, at each member's own ``y`` row (fp32 when ``y`` is fp32)."""

    d = y.shape[1]
    stacked = fp.weight.dim() == 5
    experts = int(fp.weight.shape[0]) if stacked else 1
    if fp.scale2 is None:
        shape = (experts, fp.n) if stacked else (fp.n,)
        fp.scale2 = torch.ones(shape, dtype=torch.float32, device=act.device)
    elif not fp.scale2.is_contiguous():
        fp.scale2 = fp.scale2.contiguous()
    _down_grouped[_grid(plan, d, block_n)](act, fp.weight, fp.scale, fp.scale2, plan.items, plan.counts[:1],
                                           plan.members, y, act.stride(0), D=d, K=fp.k, SBN=BN,
                                           BLOCK_N=block_n, PACKED=fp.packed, F32=y.dtype == torch.float32,
                                           STACKED=stacked, EXPERTS=experts, num_warps=4, num_stages=2)
