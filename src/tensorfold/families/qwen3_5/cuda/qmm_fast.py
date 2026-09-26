"""The lane matmul with the same arithmetic as ``qmm._qmm`` and a faster memory layout.

Per output element the sum is unchanged: group by group in order, a tensor-core dot over the
group's 64 inputs, then ``acc + p * s + xs * b``, with the same K slices per weight shape. So
every output is bit-identical to ``qmm.lane_matmul`` at any row count.

What changes is how the weights reach the kernel:
  * weights are regrouped once at load into [N/BN][K/64][BN][8] words, so a program's group is
    one contiguous BN x 32-byte block (MLX's layout spreads it over BN rows K/2 bytes apart);
  * scales and biases are stored group-major, [K/64][N], so a group's BN scales are contiguous;
  * the group loop is unrolled GPI times, so the loads of several groups can be in flight.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .qmm import bucket, group_sums, lane_matmul, split_k
from .weights import QLinear, Weights

BN = 64                   # columns per program and per stored tile


@triton.jit
def _qmm_tiled(X, XS, W, S, B, OUT, PART, M,
               N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
               BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr = False):
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
    # this program's column tile; each group's block is BLOCK_N * 8 contiguous words
    tile = W + pid_n * (KG * BLOCK_N * 8)
    local = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            words = tl.load(tile + g * (BLOCK_N * 8) + local[:, None] * 8 + rw[None, :])
            x = tl.load(X + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
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
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr = False):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


def tile_weight(weight: torch.Tensor) -> torch.Tensor:
    """MLX's (N, K/8) words -> [N/BN][K/64][BN][8] words (N padded to a multiple of BN with zeros)."""

    n, k8 = weight.shape
    npad = -(-n // BN) * BN
    if npad != n:
        weight = torch.cat([weight, weight.new_zeros((npad - n, k8))])
    return weight.reshape(npad // BN, BN, k8 // 8, 8).permute(0, 2, 1, 3).contiguous()


def group_major(t: torch.Tensor) -> torch.Tensor:
    """(N, K/64) scales or biases -> (K/64, N)."""

    return t.t().contiguous()


def groups_per_iteration(per: int, want: int = 4) -> int:
    for gpi in (want, 4, 2, 1):
        if gpi <= want and per % gpi == 0:
            return gpi
    return 1


# (groups per unrolled step, warps, pipeline stages) by row bucket; every choice gives the same bits.
# Measured on the GB10 at 1-128 rows over the 27B's projection shapes (logs/bench-qmm-variants.json).
CONFIG = {16: (4, 4, 2), 32: (2, 4, 2), 64: (1, 4, 2), 128: (1, 4, 3)}


def config_for(n: int, k: int, bm: int) -> tuple[int, int, int]:
    """Settings by row bucket only. Per-shape picks from isolated sweeps (logs/bench-qmm-tp-shapes.json)
    looked 3-8% faster alone but made the whole forward 1.5-2 ms slower at 16 rows (fwd_ab.py, 26 Sep)."""

    return CONFIG[bm]


def lane_matmul_tiled(x: torch.Tensor, tw: torch.Tensor, ts: torch.Tensor, tb: torch.Tensor, n: int,
                      xs: torch.Tensor | None = None, *, gpi: int | None = None, num_warps: int | None = None,
                      num_stages: int | None = None, bm: int | None = None, f32: bool = False) -> torch.Tensor:
    """x (M, K) bf16 times a ``tile_weight`` weight -> (M, n), bit-identical to ``qmm.lane_matmul``.

    ``f32``: return the fp32 sums unrounded (a tensor-parallel rank's partial).
    """

    m, k = x.shape
    x = x.contiguous()
    bm = bucket(m) if bm is None else bm
    cfg_gpi, cfg_warps, cfg_stages = config_for(n, k, bm)
    if xs is None:
        xs = group_sums(x)
    sk = split_k(n, k)
    per = (k // 64) // sk
    gpi = groups_per_iteration(per, gpi or cfg_gpi)
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, BN), sk)
    _qmm_tiled[grid](x, xs, tw, ts, tb, out, part, m, N=n, K=k, SK=sk, BM=bm, BLOCK_N=BN, GPI=gpi, F32=f32,
                     num_warps=num_warps or cfg_warps, num_stages=num_stages or cfg_stages)
    if sk > 1:
        total = m * n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def tile(q: QLinear) -> QLinear:
    if q.layout == "tiled":
        return q
    return QLinear(tile_weight(q.weight), group_major(q.scales), group_major(q.biases), layout="tiled", rows=q.n)


def untile(q: QLinear) -> QLinear:
    """The stored MLX layout again (for the fp32 reference, TP sharding or slicing rows)."""

    if q.layout != "tiled":
        return q
    t, kg, bn, eight = q.weight.shape
    words = q.weight.permute(0, 2, 1, 3).reshape(t * bn, kg * eight)[:q.rows].contiguous()
    return QLinear(words, q.scales.t().contiguous(), q.biases.t().contiguous())


def matmul(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """The lane matmul for either layout; both give the same bits."""

    if q.layout == "tiled":
        return lane_matmul_tiled(x, q.weight, q.scales, q.biases, q.n, xs)
    return lane_matmul(x, q.weight, q.scales, q.biases, xs=xs)


def matmul_partial(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 sums for a tiled weight, unrounded: a row-parallel rank's share of a projection."""

    if q.layout != "tiled":
        raise ValueError("matmul_partial takes tiled weights")
    return lane_matmul_tiled(x, q.weight, q.scales, q.biases, q.n, xs, f32=True)


def stack(parts: list[QLinear]) -> QLinear:
    """Several projections of the same input as one: stored-layout rows concatenated in order."""

    if any(q.layout != "mlx" for q in parts):
        raise ValueError("stack the stored layout, then tile")
    return QLinear(torch.cat([q.weight for q in parts]).contiguous(), torch.cat([q.scales for q in parts]).contiguous(),
                   torch.cat([q.biases for q in parts]).contiguous())


def stack_small(layer) -> None:
    """[z | b | a] and [k | v] as one matmul each: the gates and k/v are too narrow to fill the GPU alone."""

    if layer.gdn is not None and layer.gdn.zba is None:
        layer.gdn.zba = stack([layer.gdn.z, layer.gdn.b, layer.gdn.a])
    if layer.attn is not None and layer.attn.kv is None:
        layer.attn.kv = stack([layer.attn.k, layer.attn.v])


def prepare(w: Weights, *, fuse: bool = False) -> None:
    """Regroup every projection and the head in place (the embedding is a row lookup and stays).

    ``fuse``: also stack [z | b | a] and [k | v]. That changes which K split those columns use,
    so it changes bits relative to separate calls; serial and drafted rounds use the same stacks.
    Off by default: measured 73.8 -> 73.2 ms at 1 row and no gain at 16 rows (fwd_ab.py, 26 Sep).
    """

    for layer in w.layers:
        if fuse:
            stack_small(layer)
        for owner, names in ((layer, ("gate", "up", "down")), (layer.gdn, ("qkv", "z", "b", "a", "out")),
                             (layer.attn, ("q", "k", "v", "o"))):
            if owner is None:
                continue
            for name in names:
                setattr(owner, name, tile(getattr(owner, name)))
        if layer.gdn is not None and layer.gdn.zba is not None:
            layer.gdn.zba = tile(layer.gdn.zba)
        if layer.attn is not None and layer.attn.kv is not None:
            layer.attn.kv = tile(layer.attn.kv)
    w.head = tile(w.head)
    torch.cuda.empty_cache()
