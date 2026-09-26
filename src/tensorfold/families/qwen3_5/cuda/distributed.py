"""Two-rank tensor-parallel shards for the unchanged MLX 4-bit checkpoint.

Output shards retain complete packed rows. Input shards start and end on 64-input
group boundaries, so their packed words, scales and biases need no repacking.
Row-parallel projections keep fp32 partials until the ranks sum in rank order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Sequence

import torch
import torch.distributed as dist

from .weights import Attention, Config, GDN, Layer, QLinear, Weights

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU hosts can inspect and test the partition map.
    triton = None
    tl = None


def _rank(rank: int, world_size: int) -> None:
    if world_size != 2 or rank not in (0, 1):
        raise ValueError("the CUDA partition map currently requires two ranks")


def output_rows(n: int, rank: int, world_size: int = 2,
                segments: Sequence[int] | None = None, block: int = 1,
                device: torch.device | str = "cpu") -> torch.Tensor:
    """Rank-local output indices, preserving the order within each logical segment.

    ``segments=(Q, K, V)`` handles GDN's concatenated projection. Attention's
    fused query/gate tensor uses one segment and ``block=2*head_dim``.
    """

    _rank(rank, world_size)
    lengths = tuple(segments) if segments is not None else (n,)
    if sum(lengths) != n or any(v <= 0 or v % (world_size * block) for v in lengths):
        raise ValueError(f"output segments {lengths} do not split into {world_size} blocks of {block}")
    base = 0
    ranges = []
    for length in lengths:
        half = length // world_size
        ranges.append(torch.arange(base + rank * half, base + (rank + 1) * half, device=device))
        base += length
    return torch.cat(ranges)


def split_output(q: QLinear, rank: int, world_size: int = 2,
                 segments: Sequence[int] | None = None, block: int = 1) -> QLinear:
    """Column-parallel QLinear: each rank owns complete output rows."""

    rows = output_rows(q.n, rank, world_size, segments, block, q.weight.device)
    return QLinear(q.weight.index_select(0, rows).contiguous(),
                   q.scales.index_select(0, rows).contiguous(),
                   q.biases.index_select(0, rows).contiguous())


def split_input(q: QLinear, rank: int, world_size: int = 2) -> QLinear:
    """Row-parallel QLinear: each rank owns complete groups of 64 input columns."""

    _rank(rank, world_size)
    if q.k % (64 * world_size):
        raise ValueError(f"input width {q.k} is not divisible by {64 * world_size}")
    half_groups = q.k // (64 * world_size)
    g0, g1 = rank * half_groups, (rank + 1) * half_groups
    return QLinear(q.weight[:, g0 * 8:g1 * 8].contiguous(),
                   q.scales[:, g0:g1].contiguous(),
                   q.biases[:, g0:g1].contiguous())


def local_input(x: torch.Tensor, rank: int, world_size: int = 2) -> torch.Tensor:
    """Select the activation columns corresponding to ``split_input``."""

    _rank(rank, world_size)
    if x.ndim != 2 or x.shape[1] % (64 * world_size):
        raise ValueError("activation input must be 2-D and group-aligned")
    half = x.shape[1] // world_size
    return x[:, rank * half:(rank + 1) * half].contiguous()


@dataclass
class LayerShard:
    """Layer weights with local heads and MLP width; norms remain replicated."""

    layer: Layer
    rank: int
    world_size: int
    q_heads: int
    kv_heads: int
    k_heads: int
    v_heads: int


def split_layer(layer: Layer, cfg: Config, rank: int, world_size: int = 2) -> LayerShard:
    """Shard one layer along its head and MLP axes without changing checkpoint words."""

    _rank(rank, world_size)
    if any(v % world_size for v in (cfg.heads, cfg.kv_heads, cfg.k_heads, cfg.v_heads,
                                    cfg.intermediate)):
        raise ValueError("head count and MLP width must be divisible by rank count")
    attn = None
    if layer.attn is not None:
        a = layer.attn
        attn = Attention(q=split_output(a.q, rank, block=2 * cfg.head_dim),
                         k=split_output(a.k, rank, block=cfg.head_dim),
                         v=split_output(a.v, rank, block=cfg.head_dim),
                         o=split_input(a.o, rank), q_norm=a.q_norm, k_norm=a.k_norm)
    gdn = None
    if layer.gdn is not None:
        g = layer.gdn
        kd, vd = cfg.k_heads * cfg.dk, cfg.v_heads * cfg.dv
        rows = output_rows(g.qkv.n, rank, segments=(kd, kd, vd), device=g.conv.device)
        vh = cfg.v_heads // world_size
        lo, hi = rank * vh, (rank + 1) * vh
        gdn = GDN(qkv=split_output(g.qkv, rank, segments=(kd, kd, vd)),
                  z=split_output(g.z, rank, block=cfg.dv),
                  b=split_output(g.b, rank), a=split_output(g.a, rank),
                  out=split_input(g.out, rank),
                  conv=g.conv.index_select(0, rows).contiguous(),
                  A_log=g.A_log[lo:hi].contiguous(),
                  dt_bias=g.dt_bias[lo:hi].contiguous(), norm=g.norm)
    local = Layer(linear=layer.linear, input_norm=layer.input_norm, post_norm=layer.post_norm,
                  gdn=gdn, attn=attn, gate=split_output(layer.gate, rank),
                  up=split_output(layer.up, rank), down=split_input(layer.down, rank))
    return LayerShard(local, rank, world_size, cfg.heads // world_size, cfg.kv_heads // world_size,
                      cfg.k_heads // world_size, cfg.v_heads // world_size)


def split_weights(w: Weights, rank: int, world_size: int = 2, *, tiled: bool = False,
                  fuse: bool = False, split_head: bool = False) -> Weights:
    """Build the rank-local model while retaining replicated embedding and head.

    ``tiled``: regroup every shard (and the head) for ``qmm_fast`` after splitting; the splits
    need the stored MLX layout, and the regrouping changes no bits.
    ``split_head``: each rank keeps its half of the vocabulary's head rows (every logit is still one
    row's full dot product, so no logit changes); ``decode_tp`` then samples from both halves'
    candidates, so rank 0 no longer computes all 248k logits alone.
    """

    _rank(rank, world_size)
    c = w.config
    local_cfg = replace(c, heads=c.heads // world_size, kv_heads=c.kv_heads // world_size,
                        k_heads=c.k_heads // world_size, v_heads=c.v_heads // world_size,
                        intermediate=c.intermediate // world_size)
    layers = []
    for layer in w.layers:
        local = split_layer(layer, c, rank, world_size).layer
        if tiled:
            from .qmm_fast import stack_small, tile

            if fuse:
                stack_small(local)
            for owner, names in ((local, ("gate", "up", "down")), (local.gdn, ("qkv", "z", "b", "a", "out", "zba")),
                                 (local.attn, ("q", "k", "v", "o", "kv"))):
                if owner is not None:
                    for name in names:
                        if getattr(owner, name) is not None:
                            setattr(owner, name, tile(getattr(owner, name)))
        layers.append(local)
    head = split_output(w.head, rank, world_size) if split_head else w.head
    if tiled:
        from .qmm_fast import tile

        head = tile(head)
    return Weights(local_cfg, w.embed, layers, w.norm, head, w.inv_freq)


if triton is not None:
    @triton.jit
    def _row_qmm(X, XS, W, S, B, OUT, PART, M,
                 N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr,
                 BM: tl.constexpr, BLOCK_N: tl.constexpr):
        KG: tl.constexpr = K // 64
        PER: tl.constexpr = KG // SK
        K8: tl.constexpr = K // 8
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rm = tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, 64)
        rw = tl.arange(0, 8)
        shifts = tl.arange(0, 8) * 4
        m_ok = rm < M
        n_ok = rn < N
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for i in range(PER):
            g = pid_s * PER + i
            x = tl.load(X + rm[:, None] * K + (g * 64 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            words = tl.load(W + rn[:, None] * K8 + (g * 8 + rw)[None, :], mask=n_ok[:, None], other=0)
            q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
            q = tl.reshape(q, (BLOCK_N, 64)).to(tl.bfloat16)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
            xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
        mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=mask)
        else:
            tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=mask)


    @triton.jit
    def _reduce_slices(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < total
        acc = tl.load(PART + offs, mask=ok, other=0.0)
        for s in tl.static_range(1, SK):
            acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
        tl.store(OUT + offs, acc, mask=ok)


def row_partial(x: torch.Tensor, q: QLinear, sk: int | None = None,
                xs: torch.Tensor | None = None) -> torch.Tensor:
    """Packed 4-bit row-parallel projection, returning an fp32 (rows, outputs) partial."""

    if triton is None:
        raise RuntimeError("row_partial requires Triton")
    if q.layout == "tiled":
        from .qmm_fast import matmul_partial

        if sk is not None:
            raise ValueError("tiled row partials use the shape's own split")
        return matmul_partial(x, q, xs)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != q.k or q.k % 64:
        raise ValueError("row_partial expects (rows, shard K) bf16 and group-aligned weights")
    if q.weight.dtype != torch.int32 or q.scales.shape != (q.n, q.k // 64) or q.biases.shape != q.scales.shape:
        raise ValueError("row_partial expects packed int32 words and matching group metadata")
    if not x.is_cuda or any(t.device != x.device for t in (q.weight, q.scales, q.biases)):
        raise ValueError("row_partial requires all tensors on the same CUDA device")
    from .qmm import BN, bucket, group_sums, split_k

    m = x.shape[0]
    bm = bucket(m)
    x = x.contiguous()
    xs = group_sums(x)
    sk = split_k(q.n, q.k) if sk is None else int(sk)
    if sk < 1 or sk > 8 or (q.k // 64) % sk:
        raise ValueError("split K must divide the number of 64-input groups")
    out = torch.empty((m, q.n), dtype=torch.float32, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    _row_qmm[(triton.cdiv(q.n, BN), sk)](
        x, xs, q.weight, q.scales, q.biases, out, part, m,
        N=q.n, K=q.k, SK=sk, BM=bm, BLOCK_N=BN,
        num_warps=4 if bm <= 32 else 8, num_stages=3)
    if sk > 1:
        total = m * q.n
        _reduce_slices[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out


def sum_rank_partials(partials: Sequence[torch.Tensor], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Local rank-ordered sum, useful for one-GPU parity tests."""

    if len(partials) != 2 or any(p.dtype != torch.float32 or p.shape != partials[0].shape for p in partials):
        raise ValueError("expected two equally shaped fp32 partials in rank order")
    return (partials[0] + partials[1]).to(dtype)


def gather_rank_partials(local: torch.Tensor, group=None,
                         dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Exchange two fp32 partials, sum rank 0 then rank 1, and round once.

    Both ranks call this collective at every row-parallel attention/GDN output
    and MLP down projection. This deliberately favors reproducibility over an
    in-place NCCL all-reduce, whose summation order is an implementation detail.
    """

    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise RuntimeError("a two-rank process group must be initialized")
    if local.dtype != torch.float32 or local.ndim != 2:
        raise ValueError("local partial must be a 2-D fp32 tensor")
    if dist.get_backend(group) == "nccl" and not local.is_cuda:
        raise ValueError("NCCL partial must be on CUDA")
    local = local.contiguous()
    if os.environ.get("TF_TP_REDUCE") == "allreduce":
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        return local.to(dtype)
    gathered = torch.empty((2 * local.shape[0], local.shape[1]), dtype=torch.float32, device=local.device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    rank_parts = gathered.view(2, *local.shape)
    return (rank_parts[0] + rank_parts[1]).to(dtype)
