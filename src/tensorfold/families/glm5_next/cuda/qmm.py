"""Row-invariant 4-bit matmuls for MLX affine weights in groups of 64 (GLM-5.3-Flash), in Triton.

The 27B lane matmul's arithmetic (``qwen3_5/cuda/qmm_fast.py``), with strided input rows. For weight group g
(64 inputs, one scale s and one bias b per output column):

    P[m, n, g] = x[m, g-block] . q[n, g-block]     tensor cores, bf16 x integer-valued bf16 -> fp32
    y[m, n]    = sum over g, in order, of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

where xs[m, g] is the fp32 sum of the group's 64 inputs. The K groups are split into SK slices fixed by
the weight's shape (never by the row count) and added in slice order. Weights are regrouped once at load:
words to [N/64][K/64][64][8] (a program's group is one contiguous 2 KB block), scales and biases
group-major [K/64][N]. A row's bits never depend on the row tile or on the other rows.

``moe_gateup`` / ``moe_down`` run the same arithmetic over the window's distinct experts: one program per
(distinct expert, column tile, member tile) gathers the rows that picked the expert, so an expert's weights
are read once per window however many rows share it, and every (row, expert) pair gets the same bits.
The gate/up epilogue is GLM's limited SwiGLU: bf16(bf16(silu(min(g, L))) * clip(u, -L, L)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

BN = 64                   # columns per program and per stored tile
GS = 64                   # inputs per quantization group


@dataclass
class Q4:
    """A 4-bit group-64 matrix [n, k]: tiled words [N/64, K/64, 64, 8] int32, scales/biases [K/64, N] bf16."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


def tile_words(words: torch.Tensor) -> torch.Tensor:
    """MLX (..., N, K/8) words -> (..., N/64, K/64, 64, 8), N padded to a multiple of 64 with zeros."""

    *lead, n, k8 = words.shape
    npad = -(-n // BN) * BN
    if npad != n:
        words = torch.cat([words, words.new_zeros((*lead, npad - n, k8))], dim=-2)
    out = words.reshape(*lead, npad // BN, BN, k8 // 8, 8)
    nd = len(lead)
    return out.permute(*range(nd), nd, nd + 2, nd + 1, nd + 3).contiguous()


def untile_words(tiled: torch.Tensor, n: int) -> torch.Tensor:
    *lead, t, kg, bn, eight = tiled.shape
    nd = len(lead)
    return tiled.permute(*range(nd), nd, nd + 2, nd + 1, nd + 3).reshape(*lead, t * bn, kg * eight)[..., :n, :]


def as_i32(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32) if w.dtype != torch.int32 else w


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> Q4:
    """From MLX arrays: weight (N, K/8) uint32/int32, scales/biases (N, K/64) bf16."""

    w = as_i32(weight)
    n, k8 = w.shape
    if scales.shape != (n, k8 // 8):
        raise ValueError(f"group-64 scales expected ({n}, {k8 // 8}), got {tuple(scales.shape)}")
    return Q4(tile_words(w), scales.t().contiguous(), biases.t().contiguous(), n, k8 * 8)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several MLX (weight, scales, biases) with the same K, stacked in order, then tiled."""

    return make_q4(torch.cat([as_i32(p[0]) for p in parts]), torch.cat([p[1] for p in parts]),
                   torch.cat([p[2] for p in parts]))


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return untile_words(q.weight, q.n), q.scales.t(), q.biases.t()


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (..., N, K/8) words -> (..., N, K) fp32 values s * q + b."""

    k8 = words.shape[-1]
    w = as_i32(words).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    w, s, b = to_mlx(q)
    return dequantize(w.contiguous(), s.contiguous(), b.contiguous())


# K split rule for shapes without a table entry: split until the column tiles times slices reach this many programs
SPLIT_TARGET = 192


# Per-shape settings ("NxK": K slices / "NxK": (groups per step, warps, stages) at <= 16 rows) for each rank's share
# of GLM-5.3-Flash on two DGX Sparks, from a per-shape sweep with every shape timed on distinct weight copies: dense
# matmuls 11.05 -> 9.73 ms a 1-row forward (the KDA projection 163 -> 134 us). The K slices are part of the
# arithmetic (they change bits, for every row alike, so serial and windows stay equal); the settings change no bits.
SHAPE_SK: dict[str, int] = {"12576x4096": 4, "4096x4096": 2, "2048x4096": 4, "8192x1536": 4, "8192x512": 4,
                            "4096x8192": 8}
SHAPE_CFG: dict[str, tuple] = {"12576x4096": (2, 4, 3), "4096x4096": (2, 4, 3), "2048x4096": (1, 4, 3),
                               "8192x1536": (4, 4, 2), "8192x512": (4, 4, 2), "4096x8192": (2, 4, 3)}


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count (the 27B rule: split until the
    column tiles times slices reach SPLIT_TARGET programs; a different target changes bits for every row alike)."""

    forced = SHAPE_SK.get(f"{n}x{k}")
    if forced:
        return forced
    tiles = -(-n // BN)
    groups = k // GS
    sk = 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def bucket(m: int) -> int:
    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    raise ValueError(f"at most 128 rows, got {m}")


def gpi_for(per: int, want: int) -> int:
    for g in (want, 4, 2, 1):
        if g <= want and per % g == 0:
            return g
    return 1


# (groups per unrolled step, warps, stages) by row bucket: every choice gives the same bits
CONFIG = {16: (4, 4, 2), 32: (2, 4, 2), 64: (1, 4, 2), 128: (1, 4, 3)}


@triton.jit
def _group_sums(X, XS, x_stride, K: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    gb = tl.program_id(1)
    KG: tl.constexpr = K // 64
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 64)
    ok = g < KG
    x = tl.load(X + m * x_stride + g[:, None] * 64 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


def group_sums(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/64) fp32 sums of each 64-input group."""

    m, k = x.shape
    kg = k // GS
    if out is None:
        out = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    _group_sums[(m, triton.cdiv(kg, 16))](x, out, x.stride(0), K=k, GB=16, num_warps=2)
    return out


@triton.jit
def _qmm(X, XS, W, S, B, OUT, PART, M, x_stride,
         N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr):
    KG: tl.constexpr = K // 64
    PER: tl.constexpr = KG // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    tile = W + pid_n * (KG * BLOCK_N * 8)
    local = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            words = tl.load(tile + g * (BLOCK_N * 8) + local[:, None] * 8 + rw[None, :])
            x = tl.load(X + rm[:, None] * x_stride + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
            q = tl.reshape(q, (BLOCK_N, 64)).to(tl.bfloat16)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


def matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, part: torch.Tensor | None = None) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided) @ q.T -> (M, N) bf16, or unrounded fp32 sums with ``f32``."""

    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    bm = bucket(m)
    c_gpi, warps, stages = SHAPE_CFG.get(f"{q.n}x{q.k}", CONFIG[bm]) if bm == 16 else CONFIG[bm]
    if xs is None:
        xs = group_sums(x)
    sk = split_k(q.n, q.k)
    per = (k // GS) // sk
    gpi = gpi_for(per, c_gpi)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1:
        need = sk * m * q.n
        if part is None or part.numel() < need:
            part = torch.empty((need,), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, BN), sk)
    _qmm[grid](x, xs, q.weight, q.scales, q.biases, out, part if sk > 1 else out, m, x.stride(0),
               N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=BN, GPI=gpi, F32=f32, num_warps=warps, num_stages=stages)
    if sk > 1:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


# -- experts ---------------------------------------------------------------------------------------------
@dataclass
class Experts:
    """E experts' gate, up and down (tiled per expert), the shared expert last (id E - 1)."""

    gw: torch.Tensor          # [E, NI/64, D/64, 64, 8]
    gs: torch.Tensor          # [E, D/64, NI]
    gb: torch.Tensor
    uw: torch.Tensor
    us: torch.Tensor
    ub: torch.Tensor
    dw: torch.Tensor          # [E, D/64, NI/64, 64, 8]
    ds: torch.Tensor          # [E, NI/64, D]
    db: torch.Tensor
    count: int
    width: int                # NI (this rank's share of the expert's intermediate width)
    dims: int                 # D

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.gw, self.gs, self.gb, self.uw, self.us, self.ub,
                                                         self.dw, self.ds, self.db))


def make_experts(gate: tuple, up: tuple, down: tuple) -> Experts:
    """gate/up: (words [E, NI, D/8], scales [E, NI, D/64], biases); down: ([E, D, NI/8], [E, D, NI/64], ...)."""

    def one(t):
        w, s, b = t
        return tile_words(as_i32(w)), s.transpose(-1, -2).contiguous(), b.transpose(-1, -2).contiguous()

    gw, gs, gb = one(gate)
    uw, us, ub = one(up)
    dw, ds, db = one(down)
    return Experts(gw, gs, gb, uw, us, ub, dw, ds, db, int(gs.shape[0]), int(gs.shape[2]), int(ds.shape[2]))


@dataclass
class Group:
    """Distinct experts of a window: ids[u] (increasing), count[0], members[u, j] = row * 32 + slot (-1 after)."""

    ids: torch.Tensor         # [MAXU] int32
    count: torch.Tensor       # [1] int32
    members: torch.Tensor     # [MAXU, MAXM] int32


@triton.jit
def _moe_gateup(X, XS, x_stride, GW, GS_, GB, UW, US, UB, UIDS, UCOUNT, UMEM, ACT, AXS, LIMIT,
                K: tl.constexpr, N: tl.constexpr, MAXM: tl.constexpr, SLOTS: tl.constexpr,
                BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr):
    """Program (u, column tile, member tile): the members of distinct expert u times its gate and up rows ->
    ACT[row, slot] = bf16(bf16(silu(min(bf16 g, L))) * clip(bf16 u, -L, L)), and AXS[row, slot, tile] = the
    fp32 sum of the tile's 64 stored values (one input group of the down projection)."""

    KG: tl.constexpr = K // 64
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    mt = tl.program_id(2)
    if u >= tl.load(UCOUNT):
        return
    e = tl.load(UIDS + u).to(tl.int64)
    slot_i = mt * BM + tl.arange(0, BM)
    code = tl.load(UMEM + u * MAXM + slot_i, mask=slot_i < MAXM, other=-1)
    live = code >= 0
    if tl.sum(live.to(tl.int32), axis=0) == 0:
        return
    row = tl.where(live, code // 32, 0)
    slot = tl.where(live, code % 32, 0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    local = tl.arange(0, BLOCK_N)
    NT: tl.constexpr = N // BLOCK_N
    gtile = GW + (e * NT + pid_n) * (KG * BLOCK_N * 8)
    utile = UW + (e * NT + pid_n) * (KG * BLOCK_N * 8)
    sb = e * KG * N
    acc_g = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(KG // GPI):
        for j in tl.static_range(GPI):
            g = i * GPI + j
            x = tl.load(X + row[:, None] * x_stride + (g * 64 + rk)[None, :], mask=live[:, None], other=0.0)
            xs = tl.load(XS + row * KG + g, mask=live, other=0.0)
            wg = tl.load(gtile + g * (BLOCK_N * 8) + local[:, None] * 8 + rw[None, :])
            qg = tl.reshape((wg[:, :, None] >> shifts[None, None, :]) & 0xF, (BLOCK_N, 64)).to(tl.bfloat16)
            pg = tl.dot(x, tl.trans(qg))
            sg = tl.load(GS_ + sb + g * N + rn).to(tl.float32)
            bg = tl.load(GB + sb + g * N + rn).to(tl.float32)
            acc_g = acc_g + pg * sg[None, :] + xs[:, None] * bg[None, :]
            wu = tl.load(utile + g * (BLOCK_N * 8) + local[:, None] * 8 + rw[None, :])
            qu = tl.reshape((wu[:, :, None] >> shifts[None, None, :]) & 0xF, (BLOCK_N, 64)).to(tl.bfloat16)
            pu = tl.dot(x, tl.trans(qu))
            su = tl.load(US + sb + g * N + rn).to(tl.float32)
            bu = tl.load(UB + sb + g * N + rn).to(tl.float32)
            acc_u = acc_u + pu * su[None, :] + xs[:, None] * bu[None, :]
    gv = tl.minimum(acc_g.to(tl.bfloat16).to(tl.float32), LIMIT)
    uv = tl.minimum(tl.maximum(acc_u.to(tl.bfloat16).to(tl.float32), -LIMIT), LIMIT)
    act = ((gv / (1.0 + tl.exp(-gv))).to(tl.bfloat16).to(tl.float32) * uv).to(tl.bfloat16)
    dest = row * SLOTS + slot
    tl.store(ACT + dest[:, None] * N + rn[None, :], act, mask=live[:, None])
    sums = tl.sum(act.to(tl.float32), axis=1)
    tl.store(AXS + dest * (N // 64) + pid_n, sums, mask=live)


@triton.jit
def _moe_down(ACT, AXS, DW, DS, DB, UIDS, UCOUNT, UMEM, Y,
              NI: tl.constexpr, D: tl.constexpr, MAXM: tl.constexpr, SLOTS: tl.constexpr,
              BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr):
    """Program (u, column tile, member tile): Y[row, slot, :] (fp32) = down_e @ ACT[row, slot]."""

    KG: tl.constexpr = NI // 64
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    mt = tl.program_id(2)
    if u >= tl.load(UCOUNT):
        return
    e = tl.load(UIDS + u).to(tl.int64)
    slot_i = mt * BM + tl.arange(0, BM)
    code = tl.load(UMEM + u * MAXM + slot_i, mask=slot_i < MAXM, other=-1)
    live = code >= 0
    if tl.sum(live.to(tl.int32), axis=0) == 0:
        return
    src = tl.where(live, (code // 32) * SLOTS + code % 32, 0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 64)
    rw = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    local = tl.arange(0, BLOCK_N)
    NT: tl.constexpr = D // BLOCK_N
    tile = DW + (e * NT + pid_n) * (KG * BLOCK_N * 8)
    sb = e * KG * D
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(KG // GPI):
        for j in tl.static_range(GPI):
            g = i * GPI + j
            x = tl.load(ACT + src[:, None] * NI + (g * 64 + rk)[None, :], mask=live[:, None], other=0.0)
            xs = tl.load(AXS + src * KG + g, mask=live, other=0.0)
            w = tl.load(tile + g * (BLOCK_N * 8) + local[:, None] * 8 + rw[None, :])
            q = tl.reshape((w[:, :, None] >> shifts[None, None, :]) & 0xF, (BLOCK_N, 64)).to(tl.bfloat16)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(DS + sb + g * D + rn).to(tl.float32)
            b = tl.load(DB + sb + g * D + rn).to(tl.float32)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    tl.store(Y + src[:, None] * D + rn[None, :], acc, mask=live[:, None])


# (member tile, groups per unrolled step, warps, stages) for the grouped expert kernels; none changes bits.
# Two groups a step and two stages: a whole MoE layer 12% faster at 1 row and 18% at 4 rows than (16, 4, 4, 3),
# outputs bit-identical across settings.
MOE_CFG = {"gateup": (16, 2, 4, 2), "down": (16, 2, 4, 2)}


def moe_gateup(x: torch.Tensor, xs: torch.Tensor, ex: Experts, group: Group, act: torch.Tensor,
               axs: torch.Tensor, limit: float, *, bm: int | None = None, gpi: int | None = None,
               num_warps: int | None = None, num_stages: int | None = None) -> None:
    c_bm, c_gpi, c_w, c_s = MOE_CFG["gateup"]
    bm, gpi, num_warps, num_stages = bm or c_bm, gpi or c_gpi, num_warps or c_w, num_stages or c_s
    maxm = group.members.shape[1]
    grid = (group.ids.shape[0], ex.width // BN, triton.cdiv(maxm, bm))
    _moe_gateup[grid](x, xs, x.stride(0), ex.gw, ex.gs, ex.gb, ex.uw, ex.us, ex.ub, group.ids, group.count,
                      group.members, act, axs, float(limit), K=ex.dims, N=ex.width, MAXM=maxm, SLOTS=act.shape[1],
                      BM=bm, BLOCK_N=BN, GPI=gpi_for(ex.dims // GS, gpi), num_warps=num_warps,
                      num_stages=num_stages)


def moe_down(act: torch.Tensor, axs: torch.Tensor, ex: Experts, group: Group, y: torch.Tensor, *,
             bm: int | None = None, gpi: int | None = None, num_warps: int | None = None,
             num_stages: int | None = None) -> None:
    c_bm, c_gpi, c_w, c_s = MOE_CFG["down"]
    bm, gpi, num_warps, num_stages = bm or c_bm, gpi or c_gpi, num_warps or c_w, num_stages or c_s
    maxm = group.members.shape[1]
    grid = (group.ids.shape[0], ex.dims // BN, triton.cdiv(maxm, bm))
    _moe_down[grid](act, axs, ex.dw, ex.ds, ex.db, group.ids, group.count, group.members, y,
                    NI=ex.width, D=ex.dims, MAXM=maxm, SLOTS=act.shape[1], BM=bm, BLOCK_N=BN,
                    GPI=gpi_for(ex.width // GS, gpi), num_warps=num_warps, num_stages=num_stages)
