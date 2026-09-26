"""Row-invariant 4-bit matmuls for MLX affine weights in groups of 32 (Flash Next), in Triton.

For weight group g (32 inputs, one scale s and one bias b per output column):

    P[m, n, g] = x[m, g-block] . q[n, g-block]     tensor cores, bf16 x integer-valued bf16 -> fp32
    y[m, n]    = sum over g, in order, of  s[n, g] * P[m, n, g] + b[n, g] * xs[m, g]

where xs[m, g] is the fp32 sum of the group's 32 inputs. The K groups are split into SK slices
fixed by the weight's shape (never by the row count) and the slices are added in slice order.

Weights are regrouped once at load: words to [N/BN][K/32][BN][4] (a program's group is one
contiguous BN x 16-byte block), scales and biases group-major [K/32][N]. A row's bits do not depend
on the row tile, the column tile, the other rows or their order: each output is the same chain of
tensor-core steps over the same groups in the same order (checked in ``tests/cuda/test_flashnext_kernels.py``).

``moe_gateup`` / ``moe_down`` are the same arithmetic over a list of experts: one program per
(distinct expert, column tile, member tile) gathers the rows that picked the expert, so each
selected expert's weights are read once per window whatever the number of rows that share it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

BN = 64                   # columns per stored tile
GS = 32                   # inputs per quantization group


@triton.jit
def _deq(words, shifts, ROWS: tl.constexpr):
    """[ROWS, 4] int32 words -> [ROWS, 32] bf16 operand, q in 0..15 (exact in bf16)."""

    q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(q, (ROWS, 32)).to(tl.bfloat16)


@dataclass
class Q4:
    """A 4-bit group-32 matrix [n, k]: tiled words, group-major scales and biases (or the MLX layout)."""

    weight: torch.Tensor      # tiled: [N/BN, K/32, BN, 4] int32; mlx: [N, K/8] int32
    scales: torch.Tensor      # tiled: [K/32, N] bf16; mlx: [N, K/32]
    biases: torch.Tensor
    n: int
    k: int
    layout: str = "tiled"

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


def tile_words(words: torch.Tensor) -> torch.Tensor:
    """MLX (N, K/8) words -> [N/BN][K/32][BN][4] (N padded to BN with zeros). Works on stacked experts."""

    *lead, n, k8 = words.shape
    npad = -(-n // BN) * BN
    if npad != n:
        pad = words.new_zeros((*lead, npad - n, k8))
        words = torch.cat([words, pad], dim=-2)
    out = words.reshape(*lead, npad // BN, BN, k8 // 4, 4)
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 1, nd + 3]
    return out.permute(*perm).contiguous()


def untile_words(tiled: torch.Tensor, n: int) -> torch.Tensor:
    *lead, t, kg, bn, four = tiled.shape
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 1, nd + 3]
    return tiled.permute(*perm).reshape(*lead, t * bn, kg * four)[..., :n, :].contiguous()


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> Q4:
    """From the checkpoint's arrays: weight (N, K/8) uint32 or int32, scales/biases (N, K/32) bf16."""

    w = weight.view(torch.int32) if weight.dtype != torch.int32 else weight
    n, k8 = w.shape
    return Q4(tile_words(w), scales.t().contiguous(), biases.t().contiguous(), n, k8 * 8)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several (weight, scales, biases) of the same K stacked in order, then tiled."""

    w = torch.cat([p[0].view(torch.int32) if p[0].dtype != torch.int32 else p[0] for p in parts])
    s = torch.cat([p[1] for p in parts])
    b = torch.cat([p[2] for p in parts])
    return make_q4(w, s, b)


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (N, K/8) words, (N, K/32) scales and biases."""

    return untile_words(q.weight, q.n), q.scales.t().contiguous(), q.biases.t().contiguous()


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (N, K/8) words -> (N, K) fp32 values s * q + b (group 32)."""

    n, k8 = words.shape[-2], words.shape[-1]
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    return dequantize(*to_mlx(q))


# -- split-K by shape --------------------------------------------------------------------------
def split_k(n: int, k: int, target: int = 160) -> int:
    """K slices for an (n, k) weight: a function of the shape only (never of the row count)."""

    tiles = -(-n // BN)
    groups = k // GS
    sk = 1
    while sk < 32 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def gpi_for(per: int, want: int) -> int:
    for g in (want, 8, 4, 2, 1):
        if g <= want and per % g == 0:
            return g
    return 1


def bucket(m: int) -> int:
    """Rows a program takes: 16 to 128, then tiles of 128 (a row's bits never depend on its tile)."""

    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    return 128


# -- group sums ----------------------------------------------------------------------------------
@triton.jit
def _group_sums(X, XS, x_stride, K: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    gb = tl.program_id(1)
    KG: tl.constexpr = K // 32
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 32)
    ok = g < KG
    x = tl.load(X + m * x_stride + g[:, None] * 32 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


def group_sums(x: torch.Tensor) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/32) fp32 sums of each 32-input group."""

    m, k = x.shape
    kg = k // GS
    xs = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    gb = 32
    _group_sums[(m, triton.cdiv(kg, gb))](x, xs, x.stride(0), K=k, GB=gb, num_warps=2)
    return xs


# -- dense matmul --------------------------------------------------------------------------------
@triton.jit
def _qmm(X, XS, W, S, B, OUT, PART, M, x_stride,
         N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr, SBN: tl.constexpr):
    KG: tl.constexpr = K // 32
    PER: tl.constexpr = KG // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    SUB: tl.constexpr = SBN // BLOCK_N                  # program tiles per stored tile
    tile = W + (pid_n // SUB) * (KG * SBN * 4)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            x = tl.load(X + rm[:, None] * x_stride + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            q = _deq(words, shifts, BLOCK_N)
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


# (groups per unrolled step, warps, stages) by row bucket: every choice gives the same bits
CONFIG = {16: (4, 4, 3), 32: (2, 4, 3), 64: (2, 4, 2), 128: (1, 8, 2)}

# Measured on GB10 (1 row, each shape timed over all 48 layers' matrices): per (N, K) at up
# to 16 rows, (K slices, groups per step, warps, stages, program width). The K slices are a per-shape constant
# (they set the sum order); the rest never changes bits.
SHAPES16 = {
    (324, 10240): (32, 2, 4, 3, 64),         # hyper-connection down + inject
    (320, 10240): (32, 2, 4, 3, 64),         # a mixer's down
    (10240, 320): (1, 2, 4, 3, 64),          # hyper-connection up
    (16480, 2560): (1, 1, 4, 3, 64),         # DeltaNet q/k/v, z, b, a
    (2560, 6144): (8, 2, 4, 2, 64),          # DeltaNet / attention output
    (13952, 2560): (1, 2, 4, 3, 64),         # attention q|gate, k, v, indexer
    (248320, 2560): (1, 4, 4, 2, 64),        # head
}


def split_for(n: int, k: int) -> int:
    """The K slices of an (n, k) matrix: the tuned constant when there is one, else ``split_k``."""

    got = SHAPES16.get((n, k))
    return got[0] if got else split_k(n, k)


def matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, sk: int | None = None, part: torch.Tensor | None = None,
           gpi: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
           block_n: int | None = None, reduce: bool = True) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided) @ q.T -> (M, N) bf16 (or fp32 sums with ``f32``). ``reduce=False``
    with a split K returns the unreduced fp32 slices [SK, M, N] (the caller sums them in slice order)."""

    if q.layout != "tiled":
        raise ValueError("matmul takes tiled weights")
    m, k = x.shape
    if k != q.k or x.stride(1) != 1:
        raise ValueError(f"matmul: x {tuple(x.shape)} does not match K={q.k}")
    bm = bucket(m)
    c_gpi, c_warps, c_stages = CONFIG[bm]
    c_bn = BN
    tuned = SHAPES16.get((q.n, q.k)) if bm == 16 else None
    if tuned is not None:
        _, c_gpi, c_warps, c_stages, c_bn = tuned
    if xs is None:
        xs = group_sums(x)
    sk = int(sk) if sk else split_for(q.n, q.k)
    per = (k // GS) // sk
    g = gpi_for(per, gpi or c_gpi)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1 and part is None:
        part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    if sk > 1 and part.numel() < sk * m * q.n:
        raise ValueError("matmul: split-K scratch too small")
    bn = block_n or c_bn
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), sk)
    _qmm[grid](x, xs, q.weight, q.scales, q.biases, out, part if sk > 1 else out, m, x.stride(0),
               N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=bn, GPI=g, F32=f32, SBN=BN,
               num_warps=num_warps or c_warps, num_stages=num_stages or c_stages)
    if sk > 1 and reduce:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out if (sk == 1 or reduce) else part[:sk * m * q.n].view(sk, m, q.n)


# -- hyper-connection matmuls with their neighbours fused ------------------------------------------------------
@triton.jit
def _bsig(x):
    return (1.0 / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qmm_hcdown(H, PSS, SCALE, NORMED, W, S, B, OUT, PART, M, eps,
                N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, NC: tl.constexpr, SS: tl.constexpr,
                SK: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr):
    """The down projection of normed streams, the norm computed on the fly: x = bf16(h * rinv_s * scale) (the
    bits of glue.hc_normed; rinv_s from the stream's NC partial sums in order), written out once (the column-tile
    0 programs) for the up projection's mix. A K slice lies inside one stream."""

    KG: tl.constexpr = K // 32
    PER: tl.constexpr = KG // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    st = (pid_s * PER * 32) // D
    total = tl.zeros((BM,), dtype=tl.float32)
    for c in range(NC):
        total += tl.load(PSS + (rm * NC + c) * SS + st, mask=m_ok, other=0.0)
    rinv = 1.0 / tl.sqrt(total / D + eps)
    SUB: tl.constexpr = SBN // BLOCK_N
    tile = W + (pid_n // SUB) * (KG * SBN * 4)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            hv = tl.load(H + rm[:, None] * K + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
            sc = tl.load(SCALE + g * 32 + rk).to(tl.float32)
            x = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16)
            tl.store(NORMED + rm[:, None] * K + (g * 32 + rk)[None, :], x, mask=m_ok[:, None] & (pid_n == 0))
            xs = tl.sum(x.to(tl.float32), axis=1)
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            q = _deq(words, shifts, BLOCK_N)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def hc_down(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor, q: Q4, eps: float,
            streams: int, *, out: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
    """normed = bf16(h * rinv * scale) written to ``normed``; returns the down projection's unreduced K slices
    [SK, R, N] (or [R, N] bf16 when SK is 1). The K split is the shape's constant (``split_for``)."""

    m, k = h.shape
    d = k // streams
    sk = split_for(q.n, q.k)
    _, gpi, warps, stages, bn = SHAPES16.get((q.n, q.k), (sk, 2, 4, 3, BN))
    per = (k // GS) // sk
    if (per * GS) > d or d % (per * GS):
        raise ValueError("hc_down: a K slice must lie inside one stream")
    grid = (triton.cdiv(m, 16), triton.cdiv(q.n, bn), sk)
    _qmm_hcdown[grid](h, pss, scale, normed, q.weight, q.scales, q.biases, out, part, m, eps, N=q.n, K=k, D=d,
                      NC=pss.shape[1], SS=streams, SK=sk, BM=16, BLOCK_N=bn, GPI=gpi_for(per, gpi), SBN=BN,
                      num_warps=warps, num_stages=stages)
    return out if sk == 1 else part[:sk * m * q.n].view(sk, m, q.n)


@triton.jit
def _qmm_upmix(X, XS, W, S, B, NORMED, MIXED, XSM, M,
               N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, SS: tl.constexpr, BM: tl.constexpr,
               DB: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr):
    """The up projection for dims [DB j, DB (j + 1)) of every stream, then the mix: mixed = bf16(sum over streams
    in order of bf16(bf16(sigmoid(bf16(up_s))) * normed_s) / S) (glue.hc_mix's bits) and its group sum."""

    KG: tl.constexpr = K // 32
    j = tl.program_id(1)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    m_ok = rm < M
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    dd = tl.arange(0, DB)
    total = tl.zeros((BM, DB), dtype=tl.float32)
    for st in tl.static_range(SS):
        row0 = st * D + j * DB
        tile = W + (row0 // SBN) * (KG * SBN * 4)
        local = row0 % SBN + dd
        rn = row0 + dd
        acc = tl.zeros((BM, DB), dtype=tl.float32)
        for i in range(KG // GPI):
            for jj in tl.static_range(GPI):
                g = i * GPI + jj
                words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
                x = tl.load(X + rm[:, None] * K + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0)
                q = _deq(words, shifts, DB)
                p = tl.dot(x, tl.trans(q))
                s = tl.load(S + g * N + rn).to(tl.float32)
                b = tl.load(B + g * N + rn).to(tl.float32)
                xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
                acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
        up = acc.to(tl.bfloat16).to(tl.float32)
        nv = tl.load(NORMED + rm[:, None] * N + (st * D + j * DB + dd)[None, :], mask=m_ok[:, None],
                     other=0.0).to(tl.float32)
        total += (_bsig(up) * nv).to(tl.bfloat16).to(tl.float32)
    mixed = (total / SS).to(tl.bfloat16)
    tl.store(MIXED + rm[:, None] * D + (j * DB + dd)[None, :], mixed, mask=m_ok[:, None])
    GB: tl.constexpr = DB // 32
    sums = tl.sum(tl.reshape(mixed.to(tl.float32), (BM, GB, 32)), axis=2)         # each 32-dim group's sum
    gi = j * GB + tl.arange(0, GB)
    tl.store(XSM + rm[:, None] * (D // 32) + gi[None, :], sums, mask=m_ok[:, None])


def hc_upmix(act: torch.Tensor, xs_act: torch.Tensor, q: Q4, normed: torch.Tensor, mixed: torch.Tensor,
             xs_mixed: torch.Tensor, streams: int) -> None:
    """The up projection and the stream mix in one kernel, 32 dims of every stream a program: the bits of
    ``matmul`` then ``glue.hc_mix``."""

    m, k = act.shape
    d = q.n // streams
    db = 32
    grid = (triton.cdiv(m, 16), d // db)
    _qmm_upmix[grid](act, xs_act, q.weight, q.scales, q.biases, normed, mixed, xs_mixed, m, N=q.n, K=k, D=d,
                     SS=streams, BM=16, DB=db, GPI=gpi_for(k // GS, 2), SBN=BN, num_warps=4, num_stages=3)


# -- experts ---------------------------------------------------------------------------------------
@dataclass
class Experts:
    """E experts' gate, up and down (4-bit group 32, tiled per expert), the shared expert last."""

    gw: torch.Tensor          # [E, NI/BN, D/32, BN, 4]
    gs: torch.Tensor          # [E, D/32, NI]
    gb: torch.Tensor
    uw: torch.Tensor
    us: torch.Tensor
    ub: torch.Tensor
    dw: torch.Tensor          # [E, D/BN, NI/32, BN, 4]
    ds: torch.Tensor          # [E, NI/32, D]
    db: torch.Tensor
    count: int                # E (routed experts + the shared one)
    width: int                # NI (the expert's intermediate width)
    dims: int                 # D

    def nbytes_per_expert(self) -> int:
        total = sum(t.numel() * t.element_size() for t in (self.gw, self.gs, self.gb, self.uw, self.us, self.ub,
                                                         self.dw, self.ds, self.db))
        return total // self.count


def make_experts(gate: tuple, up: tuple, down: tuple, shared: tuple | None = None) -> Experts:
    """gate/up: (words [E, NI, D/8], scales [E, NI, D/32], biases); down: ([E, D, NI/8], ...); shared: three
    (words, scales, biases) of one expert appended as expert E."""

    def cat(stack, one):
        w, s, b = stack
        w = w.view(torch.int32) if w.dtype != torch.int32 else w
        if one is not None:
            ow, os_, ob = one
            ow = ow.view(torch.int32) if ow.dtype != torch.int32 else ow
            w = torch.cat([w, ow[None]])
            s = torch.cat([s, os_[None]])
            b = torch.cat([b, ob[None]])
        return tile_words(w), s.transpose(-1, -2).contiguous(), b.transpose(-1, -2).contiguous()

    sg, su, sd = shared if shared is not None else (None, None, None)
    gw, gs, gb = cat(gate, sg)
    uw, us, ub = cat(up, su)
    dw, ds, db = cat(down, sd)
    count, width = int(gs.shape[0]), int(gs.shape[2])
    dims = int(ds.shape[2])
    return Experts(gw, gs, gb, uw, us, ub, dw, ds, db, count, width, dims)


@triton.jit
def _moe_gateup(X, XS, GW, GS, GB, UW, US, UB, UIDS, UCOUNT, UMEM, ACT, AXS,
                K: tl.constexpr, N: tl.constexpr, MAXM: tl.constexpr, SLOTS: tl.constexpr,
                BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr):
    """Program (u, column tile, member tile): the members of distinct expert u (codes row * 32 + slot) times
    its gate and up rows -> bf16(silu(bf16(gate)) * bf16(up)) at ACT[row, slot], with the 32-input group
    sums of those bf16 values (for the down projection) at AXS[row, slot]."""

    KG: tl.constexpr = K // 32
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    mt = tl.program_id(2)
    if u >= tl.load(UCOUNT):
        return
    e = tl.load(UIDS + u).to(tl.int64)
    slot_i = mt * BM + tl.arange(0, BM)
    code = tl.load(UMEM + u * MAXM + slot_i, mask=slot_i < MAXM, other=-1)
    live = code >= 0
    row = tl.where(live, code // 32, 0)
    slot = tl.where(live, code % 32, 0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    SUB: tl.constexpr = SBN // BLOCK_N
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    NT: tl.constexpr = N // SBN
    gtile = GW + (e * NT + pid_n // SUB) * (KG * SBN * 4)
    utile = UW + (e * NT + pid_n // SUB) * (KG * SBN * 4)
    gsb = e * KG * N
    acc_g = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(KG // GPI):
        for j in tl.static_range(GPI):
            g = i * GPI + j
            x = tl.load(X + row[:, None] * K + (g * 32 + rk)[None, :], mask=live[:, None], other=0.0)
            xs = tl.load(XS + row * KG + g, mask=live, other=0.0)
            wg = tl.load(gtile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            qg = _deq(wg, shifts, BLOCK_N)
            pg = tl.dot(x, tl.trans(qg))
            sg = tl.load(GS + gsb + g * N + rn).to(tl.float32)
            bg = tl.load(GB + gsb + g * N + rn).to(tl.float32)
            acc_g = acc_g + pg * sg[None, :] + xs[:, None] * bg[None, :]
            wu = tl.load(utile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            qu = _deq(wu, shifts, BLOCK_N)
            pu = tl.dot(x, tl.trans(qu))
            su = tl.load(US + gsb + g * N + rn).to(tl.float32)
            bu = tl.load(UB + gsb + g * N + rn).to(tl.float32)
            acc_u = acc_u + pu * su[None, :] + xs[:, None] * bu[None, :]
    gv = acc_g.to(tl.bfloat16).to(tl.float32)
    uv = acc_u.to(tl.bfloat16).to(tl.float32)
    act = ((gv / (1.0 + tl.exp(-gv))).to(tl.bfloat16).to(tl.float32) * uv).to(tl.bfloat16)
    dest = row * SLOTS + slot
    tl.store(ACT + dest[:, None] * N + rn[None, :], act, mask=live[:, None])
    # group sums of this tile's 64 columns (2 groups of 32), fp32 over the stored bf16 values
    a2 = tl.reshape(act.to(tl.float32), (BM, BLOCK_N // 32, 32))
    sums = tl.sum(a2, axis=2)
    gi = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    tl.store(AXS + dest[:, None] * (N // 32) + gi[None, :], sums, mask=live[:, None])


@triton.jit
def _moe_down(ACT, AXS, DW, DS, DB, UIDS, UCOUNT, UMEM, Y,
              NI: tl.constexpr, D: tl.constexpr, MAXM: tl.constexpr, SLOTS: tl.constexpr,
              BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr):
    """Program (u, column tile, member tile): Y[row, slot, :] (fp32) = down_e @ ACT[row, slot] for the members
    of distinct expert u."""

    KG: tl.constexpr = NI // 32
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    mt = tl.program_id(2)
    if u >= tl.load(UCOUNT):
        return
    e = tl.load(UIDS + u).to(tl.int64)
    slot_i = mt * BM + tl.arange(0, BM)
    code = tl.load(UMEM + u * MAXM + slot_i, mask=slot_i < MAXM, other=-1)
    live = code >= 0
    src = tl.where(live, (code // 32) * SLOTS + code % 32, 0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    SUB: tl.constexpr = SBN // BLOCK_N
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    NT: tl.constexpr = D // SBN
    tile = DW + (e * NT + pid_n // SUB) * (KG * SBN * 4)
    sb = e * KG * D
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(KG // GPI):
        for j in tl.static_range(GPI):
            g = i * GPI + j
            x = tl.load(ACT + src[:, None] * NI + (g * 32 + rk)[None, :], mask=live[:, None], other=0.0)
            xs = tl.load(AXS + src * KG + g, mask=live, other=0.0)
            w = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            q = _deq(w, shifts, BLOCK_N)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(DS + sb + g * D + rn).to(tl.float32)
            b = tl.load(DB + sb + g * D + rn).to(tl.float32)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    tl.store(Y + src[:, None] * D + rn[None, :], acc, mask=live[:, None])


def moe_gateup(x: torch.Tensor, xs: torch.Tensor, ex: Experts, group: "Group", act: torch.Tensor,
               axs: torch.Tensor, *, bm: int = 16, gpi: int = 4, num_warps: int = 4, num_stages: int | None = None,
               block_n: int | None = None) -> None:
    maxm = group.members.shape[1]
    small = maxm <= 2                    # tuned on GB10: 32-wide programs, 2 stages at 1-2 rows; 64-wide, 3 above
    block_n = block_n or (32 if small else BN)
    num_stages = num_stages or (2 if small else 3)
    grid = (group.ids.shape[0], ex.width // block_n, triton.cdiv(maxm, bm))
    _moe_gateup[grid](x, xs, ex.gw, ex.gs, ex.gb, ex.uw, ex.us, ex.ub, group.ids, group.count, group.members,
                      act, axs, K=ex.dims, N=ex.width, MAXM=maxm, SLOTS=act.shape[1], BM=bm, BLOCK_N=block_n,
                      GPI=gpi_for(ex.dims // GS, gpi), SBN=BN, num_warps=num_warps, num_stages=num_stages)


def moe_down(act: torch.Tensor, axs: torch.Tensor, ex: Experts, group: "Group", y: torch.Tensor, *, bm: int = 16,
             gpi: int | None = None, num_warps: int = 4, num_stages: int = 2, block_n: int | None = None) -> None:
    maxm = group.members.shape[1]
    small = maxm <= 2
    block_n = block_n or (32 if small else BN)
    gpi = gpi or (2 if small else 1)
    grid = (group.ids.shape[0], ex.dims // block_n, triton.cdiv(maxm, bm))
    _moe_down[grid](act, axs, ex.dw, ex.ds, ex.db, group.ids, group.count, group.members, y,
                    NI=ex.width, D=ex.dims, MAXM=maxm, SLOTS=act.shape[1], BM=bm, BLOCK_N=block_n,
                    GPI=gpi_for(ex.width // GS, gpi), SBN=BN, num_warps=num_warps, num_stages=num_stages)


@dataclass
class Group:
    """Distinct experts of a window: ids[u] (increasing), count[0], members[u, j] = row * 32 + slot (-1 after)."""

    ids: torch.Tensor         # [MAXU] int32
    count: torch.Tensor       # [1] int32
    members: torch.Tensor     # [MAXU, MAXM] int32
