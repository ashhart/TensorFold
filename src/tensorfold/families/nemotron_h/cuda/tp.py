"""Nemotron-H over two ranks: fp32 partials summed rank 0 first, rounded once, so a row's bits ignore its window."""

from __future__ import annotations

from dataclasses import replace

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.families.qwen3_5.cuda import glue as base
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile, untile
from tensorfold.families.qwen3_5.cuda.weights import QLinear

from . import attention as A, glue as G, mamba as M, sampler as S
from .engine import Engine
from .weights import MTP, Attention, Block, Mamba, MoE, Weights

WORLD = 2
CANDIDATES = 28      # vocabulary candidates a rank shares a row


def _rows(q: QLinear, index: torch.Tensor) -> QLinear:
    m = untile(q)
    return tile(QLinear(m.weight.index_select(0, index).contiguous(), m.scales.index_select(0, index).contiguous(),
                        m.biases.index_select(0, index).contiguous()))


def _cols(q: QLinear, g0: int, g1: int) -> QLinear:
    """Input groups [g0, g1) of 64 columns."""

    m = untile(q)
    return tile(QLinear(m.weight[:, g0 * 8:g1 * 8].contiguous(), m.scales[:, g0:g1].contiguous(),
                        m.biases[:, g0:g1].contiguous()))


def expert_tiles(width: int, rank: int) -> tuple[int, int]:
    tiles = width // 64
    first = (tiles + 1) // 2
    return (0, first) if rank == 0 else (first, tiles)


def split_moe(m: MoE, rank: int) -> MoE:
    """The rank's 64-column tiles of every expert's width: up rows (32-column blocks) and down input groups."""

    ex = m.experts
    t0, t1 = expert_tiles(ex.width, rank)
    local = grouped.Experts(ex.up[:, 2 * t0:2 * t1].contiguous(), ex.down[:, :, t0:t1].contiguous(), ex.gs,
                            (t1 - t0) * 64, ex.dims, ex.limit)
    return MoE(router=m.router, bias=m.bias, experts=local)


def split_attention(a: Attention, c, rank: int) -> Attention:
    hq, hk, d = c.heads // WORLD, c.kv_heads // WORLD, c.head_dim
    dev = a.o.weight.device
    q = torch.arange(rank * hq * d, (rank + 1) * hq * d, device=dev)
    k = c.heads * d + torch.arange(rank * hk * d, (rank + 1) * hk * d, device=dev)
    v = (c.heads + c.kv_heads) * d + torch.arange(rank * hk * d, (rank + 1) * hk * d, device=dev)
    return Attention(qkv=_rows(a.qkv, torch.cat([q, k, v])),
                     o=_cols(a.o, rank * hq * d // 64, (rank + 1) * hq * d // 64))


def split_mamba(m: Mamba, c, rank: int) -> Mamba:
    h, dh, g, ds = c.m_heads // WORLD, c.m_head_dim, c.m_groups // WORLD, c.m_state
    xd, gd = c.xd, c.m_groups * c.m_state
    dev = m.conv_w.device
    heads = torch.arange(rank * h * dh, (rank + 1) * h * dh, device=dev)
    groups = torch.arange(rank * g * ds, (rank + 1) * g * ds, device=dev)
    conv_cols = torch.cat([heads, xd + groups, xd + gd + groups])
    dt_rows = xd + c.conv_dim + torch.arange(rank * h, (rank + 1) * h, device=dev)
    hs = slice(rank * h, (rank + 1) * h)
    return Mamba(in_proj=_rows(m.in_proj, torch.cat([heads, xd + conv_cols, dt_rows])),
                 out_proj=_cols(m.out_proj, rank * h * dh // 64, (rank + 1) * h * dh // 64),
                 conv_w=m.conv_w[:, conv_cols].contiguous(), conv_b=m.conv_b[conv_cols].contiguous(),
                 a=m.a[hs].contiguous(), d=m.d[hs].contiguous(), dt_bias=m.dt_bias[hs].contiguous(),
                 gnorm=m.gnorm[heads].contiguous())


def vocab_rows(head: QLinear, lo: int, hi: int) -> QLinear:
    """Rows [lo, hi) of a tiled vocabulary head (64-aligned)."""

    if lo % 64 or hi % 64:
        raise ValueError("vocabulary slices must be 64-aligned")
    return QLinear(head.weight[lo // 64:hi // 64], head.scales[:, lo:hi].contiguous(),
                   head.biases[:, lo:hi].contiguous(), layout="tiled", rows=hi - lo)


def split_weights(w: Weights, rank: int) -> Weights:
    """The rank's share in place of the full model's projections (the replicated MTP head is kept whole too)."""

    c = w.config
    if c.heads % WORLD or c.kv_heads % WORLD or c.m_heads % WORLD or c.m_groups % WORLD:
        raise ValueError("heads and groups must split over two ranks")
    blocks = []
    for b in w.blocks:
        if b.kind == "M":
            blocks.append(Block("M", b.norm, mamba=split_mamba(b.mamba, c, rank)))
        elif b.kind == "*":
            blocks.append(Block("*", b.norm, attn=split_attention(b.attn, c, rank)))
        else:
            blocks.append(Block("E", b.norm, moe=split_moe(b.moe, rank)))
        torch.cuda.empty_cache()
    half = c.vocab // WORLD
    local_cfg = replace(c, heads=c.heads // WORLD, kv_heads=c.kv_heads // WORLD, m_heads=c.m_heads // WORLD,
                        m_groups=c.m_groups // WORLD)
    out = Weights(config=local_cfg, embed=w.embed, blocks=blocks, norm_f=w.norm_f, head=w.head, mtp=w.mtp)
    out.extra = {"full_config": c, "local_head": vocab_rows(w.head, rank * half, (rank + 1) * half), "rank": rank}
    if w.mtp is not None:
        m = w.mtp
        out.extra["mtp_local"] = MTP(enorm=m.enorm, hnorm=m.hnorm, eh_proj=m.eh_proj, attn_norm=m.attn_norm,
                                     attn=split_attention(m.attn, c, rank), moe_norm=m.moe_norm,
                                     moe=split_moe(m.moe, rank), final_norm=m.final_norm)
    return out


@triton.jit
def _add_ranks_norm(H, P, W, HN, OUT, XS, eps, R, D: tl.constexpr, BLOCK: tl.constexpr):
    """delta = bf16(p0 + p1) (rank order); h = bf16(x + delta); RMSNorm(h) * w and its 64-group sums."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    p0 = tl.load(P + row * D + offs, mask=ok, other=0.0)
    p1 = tl.load(P + (R + row) * D + offs, mask=ok, other=0.0)
    delta = (p0 + p1).to(tl.bfloat16).to(tl.float32)
    x = tl.load(H + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    h = (x + delta).to(tl.bfloat16)
    tl.store(HN + row * D + offs, h, mask=ok)
    hf = h.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(hf * hf, axis=0) / D + eps)
    y = (hf * inv * tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * D + offs, y, mask=ok)
    yg = tl.reshape(y.to(tl.float32), (BLOCK // 64, 64))
    g = tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (D // 64) + g, tl.sum(yg, axis=1), mask=g < D // 64)


@triton.jit
def _moe_partial(Y, WT, OUT, D: tl.constexpr, BLOCK: tl.constexpr, NR: tl.constexpr, NS: tl.constexpr):
    """A rank's MoE output columns combined: sum_k<NR wt_k y_k + sum_k>=NR y_k (fp32, the combine order)."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    routed = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(NR):
        routed = routed + tl.load(WT + row * NS + k) * tl.load(Y + (row * NS + k) * D + offs, mask=ok,
                                                                other=0.0).to(tl.float32)
    shared = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(NR, NS):
        shared = shared + tl.load(Y + (row * NS + k) * D + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + row * D + offs, routed + shared, mask=ok)


class TPEngine(Engine):
    """``gather(local)`` returns (2 * R, ...) with rank 0's rows first; NCCL's all-gather is captured in the graphs."""

    def __init__(self, w: Weights, gather, **kw):
        super().__init__(w, **kw)
        self.gather = gather
        self.full = w.extra["full_config"]
        self.rank = w.extra["rank"]
        self.local_head = w.extra["local_head"]
        self.cand_vals = torch.zeros((self.max_rows, WORLD * CANDIDATES), dtype=torch.float32, device=self.device)
        self.cand_ids = torch.zeros((self.max_rows, WORLD * CANDIDATES), dtype=torch.int64, device=self.device)

    def norm(self, x, delta, weight):
        if delta is None or delta[0] != "ranks":
            return super().norm(x, delta, weight)
        rows, d = x.shape
        hn, out = torch.empty_like(x), torch.empty_like(x)
        xs = torch.empty((rows, d // 64), dtype=torch.float32, device=x.device)
        _add_ranks_norm[(rows,)](x, delta[1], weight, hn, out, xs, self.c.eps, rows, D=d,
                                 BLOCK=triton.next_power_of_2(d), num_warps=8)
        return hn, out, xs

    def mamba(self, m, normed, xs, rows: int, j: int):
        c = self.c
        proj = G.dense(normed, m.in_proj, xs)
        M.conv(proj, self.conv_base[j], self.raw[j], self.xc[j], m.conv_w, m.conv_b, self.meta, rows, xd=c.xd)
        y = M.scan(proj, self.xc[j], self.dt[j], self.ssm[j], m.a, m.d, m.dt_bias, self.meta, rows,
                   heads=c.m_heads, head_dim=c.m_head_dim, groups=c.m_groups, state_dim=c.m_state, lo=c.dt_min,
                   hi=c.dt_max)
        g, gxs = M.group_rmsnorm(y, m.gnorm, c.eps, c.m_groups)
        return ("ranks", self.gather(G.dense(g, m.out_proj, gxs, f32=True)))

    def attention_tp(self, a, normed, xs, rows: int, k_cache, v_cache, meta):
        c = self.c
        qkv = G.dense(normed, a.qkv, xs)
        A.kv_write(qkv, k_cache, v_cache, meta, rows, q_dim=c.heads * c.head_dim)
        out, oxs = A.attention(qkv, k_cache, v_cache, meta, rows, heads=c.heads, kv_heads=c.kv_heads,
                               head_dim=c.head_dim, max_chunks=self.max_chunks)
        return ("ranks", self.gather(G.dense(out, a.o, oxs, f32=True)))

    def moe_tp(self, moe, normed, rows: int, prefill: bool = False):
        _, y, wts = self.moe_rows(moe, normed, rows) if prefill else self.moe(moe, normed, rows)
        d = self.c.hidden
        part = torch.empty((rows, d), dtype=torch.float32, device=normed.device)
        _moe_partial[(rows,)](y, wts, part, D=d, BLOCK=triton.next_power_of_2(d), NR=self.c.top_k, NS=self.ns,
                              num_warps=8)
        return ("ranks", self.gather(part))

    # prompt chunks gather every projection's rank share, as the window forward does
    def _dense_delta(self, x, q, xs):
        return ("ranks", self.gather(G.prefill_dense(x, q, f32=True)))

    def prefill_moe(self, moe, normed, rows: int):
        return self.moe_tp(moe, normed, rows, prefill=True)

    @property
    def head_width(self) -> tuple[int, int]:
        return self.local_head.n, self.rank * self.local_head.n

    def sample_last(self, normed, xs) -> None:
        logits = G.prefill_dense(normed, self.local_head)
        if self.masked:
            logits = logits + self.bias[:1]
        vals, ids = torch.topk(logits.float(), CANDIDATES, dim=-1)
        ids = ids + self.rank * self.local_head.n
        both = self.gather(torch.cat([vals.view(torch.int32), ids.view(torch.int32)], dim=1)).view(
            WORLD, 1, 3 * CANDIDATES)
        v = both[:, :, :CANDIDATES].contiguous().view(torch.float32)
        i = both[:, :, CANDIDATES:].contiguous().view(torch.int64)
        S.sample_candidates(torch.cat([v[0], v[1]], dim=1), torch.cat([i[0], i[1]], dim=1),
                            self._meta_at(self.pos - 1), self.params, self.p_sampled)

    def _forward(self, rows: int) -> None:
        w, c = self.w, self.c
        x = base.embed(self.ids[:rows], w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        delta = None
        mj = aj = 0
        for blk in w.blocks:
            x, normed, xs = self.norm(x, delta, blk.norm)
            if blk.kind == "M":
                delta = self.mamba(blk.mamba, normed, xs, rows, mj)
                mj += 1
            elif blk.kind == "*":
                delta = self.attention_tp(blk.attn, normed, xs, rows, self.k_cache[aj], self.v_cache[aj], self.meta)
                aj += 1
            else:
                delta = self.moe_tp(blk.moe, normed, rows)
        _, normed, xs = self.norm(x, delta, w.norm_f)
        self.hidden[:rows].copy_(normed)
        logits = G.dense(normed, self.local_head, xs)
        if self.masked:
            logits = logits + self.bias[:rows]
        vals, ids = torch.topk(logits.float(), CANDIDATES, dim=-1)
        ids = ids + self.rank * self.local_head.n
        both = self.gather(torch.cat([vals.view(torch.int32), ids.view(torch.int32)], dim=1)).view(
            WORLD, rows, 3 * CANDIDATES)
        v = both[:, :, :CANDIDATES].contiguous().view(torch.float32)
        i = both[:, :, CANDIDATES:].contiguous().view(torch.int64)
        self.cand_vals[:rows].copy_(torch.cat([v[0], v[1]], dim=1))
        self.cand_ids[:rows].copy_(torch.cat([i[0], i[1]], dim=1))
        S.sample_candidates(self.cand_vals[:rows], self.cand_ids[:rows], self.meta, self.params, self.sampled[:rows])
        self._host_sampled[:rows].copy_(self.sampled[:rows], non_blocking=True)


def nccl_gather(group=None):
    """An all-gather along rows (rank 0 first) over NCCL, capturable in CUDA graphs."""

    import torch.distributed as dist

    def gather(local: torch.Tensor) -> torch.Tensor:
        local = local.contiguous()
        out = torch.empty((WORLD * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(out, local, group=group)
        return out

    return gather
