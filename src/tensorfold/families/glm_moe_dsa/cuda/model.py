"""Full GLM-5.3's forward on one rank of ``world`` (milestones M2 + M5 reference: eager torch attention, token-level DSA past index_topk).

Each rank holds 16 of 64 heads, a quarter of every expert's intermediate width and of the dense/shared MLPs, and a
quarter of the vocabulary; the latent MLA cache (576 per token and layer: the kv latent and the shared RoPE key) is
the same on every rank. After o_proj and after the MLP each rank has an fp32 partial of the hidden state; the
partials are all-gathered and added in rank order (``reduce``), so every rank continues with the same bits.

Attention is absorbed MLA: q_nope is mapped through kv_b's key block into the 512-wide latent space, scored against
the cached latents plus RoPE parts, and the attended latent is mapped out through kv_b's value block. Within
``index_topk`` keys the DSA indexer selects every key, so this milestone attends densely and refuses longer contexts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tensorfold.cuda.exl3 import experts as x3experts

from ..config import Config
from .weights import Layer

ROWS = 128               # widest call of the EXL3 linear and expert kernels; longer inputs run in pieces
RS_ROWS = 64             # from this many rows the partial sums take the exact reduce-scatter


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """GLM's interleaved RoPE (transformers' apply_rotary_pos_emb_interleave): pairs (x0, x1), (x2, x3), ... rotated
    by one frequency each, written evens then odds. x [..., R, (heads,) 64], pos [R]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None, :]                                # [R, d/2]
    cos, sin = ang.cos(), ang.sin()
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    xf = x.float()
    a, b = xf[..., 0::2], xf[..., 1::2]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(x.dtype)


def linear(lin, x: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """An EXL3 linear over any number of rows (in pieces of ROWS)."""
    if x.shape[0] <= ROWS:
        return lin(x.contiguous(), out_dtype=out_dtype)
    return torch.cat([lin(x[i:i + ROWS].contiguous(), out_dtype=out_dtype) for i in range(0, x.shape[0], ROWS)])


@dataclass
class RankModel:
    cfg: Config
    rank: int
    world: int
    comm: object                         # comm.NCCL or a test ThreadComm: all_gather(send, recv)
    embed: torch.Tensor                  # [vocab, hidden] bf16, replicated
    final_norm: torch.Tensor
    lm_head: torch.Tensor                # [vocab / world, hidden] bf16
    layers: list[Layer]
    row_exact: bool = False              # decode/verify windows: torch matmuls one row at a time (drafted == serial)

    def rows(self, n: int, fn):
        """fn(row slice) for all rows at once, or one row at a time in row_exact mode (then concatenated): a torch
        matmul's bits can depend on how many rows it is given, a row alone never does."""
        if not self.row_exact or n <= 1:
            return fn(slice(None))
        parts = [fn(slice(i, i + 1)) for i in range(n)]
        if isinstance(parts[0], tuple):
            return tuple(torch.cat(t) for t in zip(*parts))
        return torch.cat(parts)

    def norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """RMSNorm, one row at a time in row_exact mode: torch's mean over 6144 picks its reduction by row count."""
        return self.rows(x.shape[0], lambda i: rms_norm(x[i], w, eps))

    @property
    def heads(self) -> int:
        return self.cfg.num_attention_heads // self.world

    def reduce(self, part: torch.Tensor) -> torch.Tensor:
        """fp32 partials of every rank, added in rank order (the same bits on every rank).

        Up to RS_ROWS rows: all-gather every rank's partial and add them in rank order. Wider (prompt chunks): the
        exact reduce-scatter - an all-to-all of column quarters, each rank adds (in rank order) the quarter it owns,
        then the summed quarters are gathered - the same bits for about a third of the bytes."""
        world = self.world
        part = part.float().contiguous()
        rows, width = part.shape
        if world == 1:
            return part
        if rows < RS_ROWS or width % world or not hasattr(self.comm, "all_to_all"):
            recv = torch.empty((world, rows, width), dtype=torch.float32, device=part.device)
            self.comm.all_gather(part, recv)
            out = recv[0].clone()
            for r in range(1, world):
                out += recv[r]
            return out
        q = width // world
        send = part.view(rows, world, q).permute(1, 0, 2).contiguous().view(world, rows * q)
        recv = torch.empty_like(send)
        self.comm.all_to_all(send, recv)
        mine = recv[0].clone()
        for r in range(1, world):
            mine += recv[r]
        gathered = torch.empty((world, rows * q), dtype=torch.float32, device=part.device)
        self.comm.all_gather(mine, gathered)
        return gathered.view(world, rows, q).permute(1, 0, 2).reshape(rows, width)

    # -------------------------------------------------------------------------------------------- attention ---
    def indexer(self, L: Layer, h: torch.Tensor, qa: torch.Tensor, pos: torch.Tensor,
                icache: torch.Tensor) -> torch.Tensor | None:
        """DSA (V3.2 style, interleaved RoPE on the first 64 dims): write this layer's index keys, and when the context
        is longer than index_topk return each row's top index_topk key positions [R, K] (None: every key is selected)."""
        c, ix = self.cfg, L.indexer
        R, D, nh, rd = h.shape[0], c.index_head_dim, c.index_n_heads, c.qk_rope_head_dim
        k = self.rows(R, lambda i: F.layer_norm((h[i].float() @ ix["wk"].float().t()), (D,), ix["k_norm"][0].float(),
                                                ix["k_norm"][1].float(), 1e-6)).to(h.dtype)
        k = torch.cat([rope(k[:, :rd], pos, c.rope_theta), k[:, rd:]], dim=-1)
        icache[pos] = k
        T = int(pos.max()) + 1
        if T <= c.index_topk:
            return None
        q = linear(ix["wq_b"], qa).view(R, nh, D)
        q = torch.cat([rope(q[..., :rd], pos, c.rope_theta), q[..., rd:]], dim=-1).float()
        w = self.rows(R, lambda i: (h[i].to(ix["weights_proj"].dtype) @ ix["weights_proj"].t()).float()) \
            * nh ** -0.5                                                                    # [R, nh]
        keys = icache[:T].float()                                                           # [T, D]
        out = torch.full((R, c.index_topk), -1, dtype=torch.long, device=h.device)
        blk = 1 if self.row_exact else 8
        for i in range(0, R, blk):                         # rows in blocks: [blk, heads, T] fp32 scores
            t = int(pos[i:i + blk].max()) + 1            # a row alone scores exactly its own prefix
            if t <= c.index_topk:                        # this row still sees every key: select them all
                out[i:i + blk, :t] = torch.arange(t, device=h.device)
                continue
            sc = torch.relu(torch.einsum("rhd,td->rht", q[i:i + blk], keys[:t]) * D ** -0.5)
            idx = torch.einsum("rh,rht->rt", w[i:i + blk], sc)
            idx.masked_fill_(torch.arange(t, device=h.device)[None, :] > pos[i:i + blk, None], float("-inf"))
            out[i:i + blk] = torch.topk(idx, c.index_topk, dim=-1).indices
        return out

    def attention(self, L: Layer, h: torch.Tensor, pos: torch.Tensor, cache: torch.Tensor,
                  icache: torch.Tensor | None = None, topk: torch.Tensor | None = None):
        """(fp32 partial [R, hidden], this layer's top-k or the one it reused). Full-indexer layers compute top-k from
        ``icache``; shared layers attend the ``topk`` they are given (None within index_topk: every key)."""
        c, H = self.cfg, self.heads
        nope, rdim, vdim, lw = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim, c.kv_lora_rank
        R = h.shape[0]
        qa = self.norm(linear(L.q_a, h), L.q_a_norm, c.rms_norm_eps)
        if L.indexer is not None:
            topk = self.indexer(L, h, qa, pos, icache)
        q = linear(L.q_b, qa).view(R, H, nope + rdim)
        q_nope, q_rot = q[..., :nope], rope(q[..., nope:], pos, c.rope_theta)
        kva = linear(L.kv_a, h)[:, :lw + rdim]                               # stored 640 wide; 576 are real
        lat = self.norm(kva[:, :lw], L.kv_a_norm, c.rms_norm_eps)
        k_rot = rope(kva[:, lw:], pos, c.rope_theta)
        cache[pos] = torch.cat([lat, k_rot], dim=-1)

        kvb = L.kv_b.view(H, nope + vdim, lw)                                # this rank's heads
        wk, wv = kvb[:, :nope, :].float(), kvb[:, nope:, :].float()
        scale = 1.0 / math.sqrt(nope + rdim)

        def attend(i):                                   # rows i: absorb, score, attend (per row in row_exact)
            qa_ = torch.einsum("rhn,hnl->rhl", q_nope[i].float(), wk)                  # [r, H, 512]
            qr_, ps = q_rot[i].float(), pos[i]
            if topk is None:                             # every earlier key (dense window)
                T = int(ps.max()) + 1
                keys = cache[:T].float()                 # [T, 576]
                sc = (torch.einsum("rhl,tl->rht", qa_, keys[:, :lw]) + torch.einsum("rhd,td->rht", qr_, keys[:, lw:])) \
                    * scale
                sc.masked_fill_((torch.arange(T, device=h.device)[None, :] > ps[:, None])[:, None, :], float("-inf"))
                return torch.einsum("rht,tl->rhl", torch.softmax(sc, dim=-1), keys[:, :lw])
            sel = topk[i]                                # [r, K], -1 = none
            keys = cache[sel.clamp(min=0)].float()       # [r, K, 576]
            sc = (torch.einsum("rhl,rkl->rhk", qa_, keys[..., :lw]) + torch.einsum("rhd,rkd->rhk", qr_, keys[..., lw:])) \
                * scale
            sc.masked_fill_(((sel < 0) | (sel > ps[:, None]))[:, None, :], float("-inf"))
            return torch.einsum("rhk,rkl->rhl", torch.softmax(sc, dim=-1), keys[..., :lw])

        if self.row_exact or topk is None:
            o_lat = self.rows(R, attend)
        else:                                            # prompt chunks: selected keys gathered 32 rows at a time
            o_lat = torch.cat([attend(slice(i, i + 32)) for i in range(0, R, 32)])
        o = self.rows(R, lambda i: torch.einsum("rhl,hvl->rhv", o_lat[i], wv)).to(h.dtype).reshape(R, H * vdim)
        return linear(L.o_proj, o, out_dtype=torch.float32), topk            # fp32 partial [R, hidden]

    # ------------------------------------------------------------------------------------------------ MLPs ---
    def mlp(self, lins: dict, x: torch.Tensor) -> torch.Tensor:
        a = F.silu(linear(lins["gate"], x).float()) * linear(lins["up"], x).float()
        return linear(lins["down"], a.to(x.dtype), out_dtype=torch.float32)

    def route(self, L: Layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        w, bias = L.router

        def one(i):                                  # a row's picks and weights never depend on the window
            scores = torch.sigmoid(x[i].float() @ w.float().t())
            idx = torch.topk(scores + bias.float(), c.num_experts_per_tok, dim=-1, sorted=True).indices
            wts = scores.gather(1, idx)
            if c.norm_topk_prob:
                wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
            return idx.to(torch.int32), (wts * c.routed_scaling_factor).float()
        return self.rows(x.shape[0], one)

    def moe(self, L: Layer, x: torch.Tensor) -> torch.Tensor:
        idx, wts = self.route(L, x)
        out = torch.empty((x.shape[0], self.cfg.hidden_size), dtype=torch.float32, device=x.device)
        for i in range(0, x.shape[0], ROWS):
            n = min(ROWS, x.shape[0] - i)
            if hasattr(L.experts, "decode"):             # the shared layout (experts_cx)
                s = L.experts.scratch(n, self.cfg.num_experts_per_tok)
                L.experts.decode(x[i:i + n].contiguous(), idx[i:i + n].contiguous(), wts[i:i + n].contiguous(), s,
                                 out[i:i + n], n)
                continue
            s = x3experts.Scratch(L.experts, n, self.cfg.num_experts_per_tok)
            x3experts.routed(x[i:i + n].contiguous(), idx[i:i + n].contiguous(), wts[i:i + n].contiguous(),
                             L.experts, s, out[i:i + n], n)
        return out + self.mlp(L.shared, x)

    # ------------------------------------------------------------------------------------------- the model ---
    def layer(self, L: Layer, x: torch.Tensor, pos: torch.Tensor, cache: torch.Tensor,
              icache: torch.Tensor | None, topk: torch.Tensor | None):
        eps = self.cfg.rms_norm_eps
        part, topk = self.attention(L, self.norm(x, L.input_norm, eps), pos, cache, icache, topk)
        x = (x.float() + self.reduce(part)).to(x.dtype)
        h = self.norm(x, L.post_attn_norm, eps)
        part = self.moe(L, h) if L.experts is not None else self.mlp(L.shared, h)
        return (x.float() + self.reduce(part)).to(x.dtype), topk

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor, caches: list[torch.Tensor],
                icaches: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """Hidden states after the last decoder layer for ``tokens`` at ``pos``. ``caches``: the latent cache of
        every layer; ``icaches``: the index-key cache of every full-indexer layer (both updated at ``pos``)."""
        icaches = icaches or {}
        if int(pos.max()) >= self.cfg.index_topk and not icaches:
            raise ValueError("contexts past index_topk need the index-key caches (icaches)")
        x = self.embed[tokens]
        topk = None
        for L, cache in zip(self.layers, caches):
            x, topk = self.layer(L, x, pos, cache, icaches.get(L.index), topk)
        return x

    def mtp(self, head, h_prev: torch.Tensor, tokens: torch.Tensor, pos: torch.Tensor, cache: torch.Tensor,
            icache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One MTP step for rows at ``pos``: the target's hidden state of the previous position and the token at ``pos``
        in, (the MTP layer's hidden state, full logits for position pos + 1) out. Writes the MTP layer's caches."""
        eps = self.cfg.rms_norm_eps
        e = self.norm(self.embed[tokens], head.enorm, eps)
        hh = self.norm(h_prev, head.hnorm, eps)
        cat = torch.cat([e, hh], dim=-1)
        x = self.rows(cat.shape[0], lambda i: (cat[i].float() @ head.eh_proj.float().t())).to(h_prev.dtype)
        x, _ = self.layer(head.layer, x, pos, cache, icache, None)
        return x, self.logits(x, norm=head.head_norm)

    def logits(self, x: torch.Tensor, norm: torch.Tensor | None = None) -> torch.Tensor:
        """Full-vocabulary logits (fp32) on every rank: vocab shares gathered in rank order."""
        h = self.norm(x, self.final_norm if norm is None else norm, self.cfg.rms_norm_eps)
        part = self.rows(h.shape[0], lambda i: h[i].float() @ self.lm_head.float().t()).contiguous()  # [R, V/world]
        recv = torch.empty((self.world, *part.shape), dtype=torch.float32, device=x.device)
        self.comm.all_gather(part, recv)
        return recv.permute(1, 0, 2).reshape(x.shape[0], -1)
