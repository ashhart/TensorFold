"""Flash Next's group-32 4-bit matmuls; ordered groups and fixed K slices summed in order keep a row's bits its own."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import qmm as shared

BN = 64                   # columns per stored tile
GS = 32                   # inputs per quantization group


@triton.jit
def _deq(words, shifts, ROWS: tl.constexpr):
    """[ROWS, 4] int32 words -> [ROWS, 32] bf16 operand, q in 0..15 (exact in bf16)."""

    q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(q, (ROWS, 32)).to(tl.bfloat16)


@dataclass
class Q4:
    """4-bit group-32 [n, k]: "frag" for the shared lane matmul, "tiled" ([N/BN, K/32, BN, 4] words) for HC kernels."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    layout: str = "frag"
    gs: int = GS

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


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, layout: str = "frag") -> Q4:
    """From the checkpoint's arrays: weight (N, K/8) uint32 or int32, scales/biases (N, K/32) bf16."""

    w = weight.view(torch.int32) if weight.dtype != torch.int32 else weight
    n, k8 = w.shape
    if layout == "frag":
        p = shared.pack(w, scales, biases, GS)
        return Q4(p.weight, p.scales, p.biases, n, k8 * 8)
    return Q4(tile_words(w), scales.t().contiguous(), biases.t().contiguous(), n, k8 * 8, "tiled")


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], layout: str = "frag") -> Q4:
    """Rows of several (weight, scales, biases) of the same K stacked in order, then packed."""

    w = torch.cat([p[0].view(torch.int32) if p[0].dtype != torch.int32 else p[0] for p in parts])
    s = torch.cat([p[1] for p in parts])
    b = torch.cat([p[2] for p in parts])
    return make_q4(w, s, b, layout)


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (N, K/8) words, (N, K/32) scales and biases."""

    if q.layout == "frag":
        return shared.unpack(q)
    return untile_words(q.weight, q.n), q.scales.t().contiguous(), q.biases.t().contiguous()


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (N, K/8) words -> (N, K) fp32 values s * q + b (group 32)."""

    k8 = words.shape[-1]
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    return dequantize(*to_mlx(q))


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

# K slices are fixed per shape to preserve summation order; other launch settings do not change bits.
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
    """The K slices of an (n, k) matrix: the shape constant when there is one, else ``split_k``."""

    got = SHAPES16.get((n, k))
    return got[0] if got else split_k(n, k)


def matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, sk: int | None = None, part: torch.Tensor | None = None,
           gpi: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
           block_n: int | None = None, reduce: bool = True) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided) @ q.T -> (M, N) bf16 (or fp32 sums with ``f32``). ``reduce=False`` with a split K returns the unreduced fp32 slices [SK, M, N] (the caller sums them in slice order)."""

    if q.layout == "frag":
        if x.shape[1] != q.k or x.stride(1) != 1:
            raise ValueError(f"matmul: x {tuple(x.shape)} does not match K={q.k}")
        return shared.matmul(x, q, group_sums(x) if xs is None else xs, sk=int(sk) if sk else split_for(q.n, q.k),
                             f32=f32, out=out, part=part, reduce=reduce)
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


def prefill_matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
                   f32: bool = False, part: torch.Tensor | None = None, reduce: bool = True) -> torch.Tensor:
    """A prompt chunk's matmul ("frag": shared prefill matmul, "tiled": ``matmul``); row-invariant at any chunk."""

    if q.layout == "frag":
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return matmul(x, q, xs, out=out, f32=f32, part=part, reduce=reduce)


@triton.jit
def _bsig(x):
    return (1.0 / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qmm_hcdown(H, PSS, SCALE, NORMED, W, S, B, OUT, PART, M, eps,
                N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, NC: tl.constexpr, SS: tl.constexpr,
                SK: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr):
    """Compute the down projection with glue.hc_normed's bf16 rounding and ordered partial sums, keeping each K slice within one stream and writing normed values once for the up mix."""

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
    """normed = bf16(h * rinv * scale) written to ``normed``; returns the down projection's unreduced K slices [SK, R, N] (or [R, N] bf16 when SK is 1). The K split is the shape's constant (``split_for``)."""

    if q.layout != "tiled":
        raise ValueError("hc_down takes tiled weights")

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
    """The up projection for dims [DB j, DB (j + 1)) of every stream, then the mix: mixed = bf16(sum over streams in order of bf16(bf16(sigmoid(bf16(up_s))) * normed_s) / S) (glue.hc_mix's bits) and its group sum."""

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
    """The up projection and the stream mix in one kernel, 32 dims of every stream a program: the bits of ``matmul`` then ``glue.hc_mix``."""

    if q.layout != "tiled":
        raise ValueError("hc_upmix takes tiled weights")

    m, k = act.shape
    d = q.n // streams
    db = 32
    grid = (triton.cdiv(m, 16), d // db)
    _qmm_upmix[grid](act, xs_act, q.weight, q.scales, q.biases, normed, mixed, xs_mixed, m, N=q.n, K=k, D=d,
                     SS=streams, BM=16, DB=db, GPI=gpi_for(k // GS, 2), SBN=BN, num_warps=4, num_stages=3)
