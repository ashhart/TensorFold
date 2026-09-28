"""The MTP head drafts from the target's final normed rows; its drafts only pick rows to verify, never output bits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.kernels import attention as tree_attention
from tensorfold.cuda.kernels.prefill_attention import attention as prefill_attention
from tensorfold.families.qwen3_5.cuda import glue
from tensorfold.families.qwen3_5.cuda.qmm_fast import matmul, tile, untile
from tensorfold.families.qwen3_5.cuda.weights import QLinear, Weights

from .weights import MTP


class Cache:
    """The head's attention cache: slot t holds position t; slots below ``pos`` are absorbed, the rest scratch."""

    def __init__(self, w: Weights, capacity: int) -> None:
        c = w.config
        self.k = torch.empty((capacity, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=w.norm.device)
        self.v = torch.empty_like(self.k)
        self.pos = 0

    def view(self, rows: int = 0) -> "Cache":
        """The same buffers (rows below ``pos`` stay as they are), copied with room for ``rows`` slots if shorter."""

        other = object.__new__(Cache)
        other.k, other.v, other.pos = self.k, self.v, self.pos
        if rows > self.k.shape[0]:
            other.k = torch.cat([self.k[:self.pos], self.k.new_empty((rows - self.pos, *self.k.shape[1:]))])
            other.v = torch.cat([self.v[:self.pos], self.v.new_empty((rows - self.pos, *self.v.shape[1:]))])
        return other


def offsets(cache: Cache) -> torch.Tensor:
    """The cache's keys and values as the tree attention's (1, 2) device offsets."""

    return torch.tensor(tree_attention.offsets([(cache.k, cache.v)], cache.k.device), dtype=torch.int64,
                        device=cache.k.device).view(1, 2)


class Staged:
    """Static inputs of an MTP call of ``width`` rows over at most ``context`` slots, refreshed before each replay."""

    def __init__(self, cache: Cache, width: int, context: int, group: int, hidden: int) -> None:
        flat, items, chunks = tree_attention.padded_host(list(range(-1, width - 1)), context, group)
        self.width = width
        self.host = torch.tensor([0] * (2 * width) + flat, dtype=torch.int32).pin_memory()
        self.dev = self.host.to(cache.k.device)
        self.ids, self.pos = self.dev[:width], self.dev[width:2 * width]
        self.aplan = tree_attention.from_packed(self.dev[2 * width:], 1, width, items, chunks)
        self.aoffs = offsets(cache)
        self.states = torch.zeros((width, hidden), dtype=torch.bfloat16, device=cache.k.device)

    def refresh(self, states: torch.Tensor, tokens: Sequence[int], p0: int) -> None:
        w, h = self.width, self.host.numpy()
        h[:w] = tokens
        h[w:2 * w] = np.arange(p0, p0 + w)
        h[3 * w + 2] = p0
        h[3 * w + 3] = -(-(p0 + w) // tree_attention.CHUNK)
        self.dev.copy_(self.host, non_blocking=True)
        self.states.copy_(states)


class Head:
    """One MTP layer: [norm(embed(next token)) | norm(state)] through fc, attention and experts, then a draft head."""

    def __init__(self, w: Weights, m: MTP, ids: np.ndarray | None = None) -> None:
        self.w, self.m = w, m
        self.ids, self.head = None, w.head
        if ids is not None:                           # score only these token ids when drafting
            full = untile(w.head)
            self.ids = torch.as_tensor(ids, dtype=torch.int64, device=w.norm.device)
            self.head = tile(QLinear(full.weight[self.ids].contiguous(), full.scales[self.ids].contiguous(),
                                     full.biases[self.ids].contiguous()))
            del full

    @torch.no_grad()
    def forward(self, cache: Cache, states: torch.Tensor, tokens: Sequence[int], p0: int,
                staged: "Staged | None" = None) -> torch.Tensor:
        """Rows at positions [p0, p0 + n): each a (state at t, token t + 1) pair attending to slots below p0 and to each other; returns their normed outputs and writes their keys at [p0, p0 + n)."""

        c = self.w.config
        n = states.shape[0]
        if p0 + n > cache.k.shape[0]:
            raise ValueError("MTP positions past the cache")
        wide = n > tree_attention.MAX_NODES            # a prompt chunk: the prefill kernel, keys written first
        if staged is None:
            ids = torch.tensor(list(tokens), dtype=torch.int32, device=states.device)
            pos = torch.arange(p0, p0 + n, dtype=torch.int32, device=states.device)
            aplan = None if wide else tree_attention.plan([list(range(-1, n - 1))], [p0], c.heads // c.kv_heads,
                                                          states.device)
            aoffs = None if wide else offsets(cache)
        else:
            ids, pos, aplan, aoffs = staged.ids, staged.pos, staged.aplan, staged.aoffs

        def attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            slots = pos.long()
            if wide:
                cache.k.index_copy_(0, slots, key)
                cache.v.index_copy_(0, slots, value)
                return prefill_attention(q, cache.k, cache.v, p0, scale=c.head_dim ** -0.5)
            out = tree_attention.attention(q, key, value, aoffs, aplan, scale=c.head_dim ** -0.5)
            cache.k.index_copy_(0, slots, key)
            cache.v.index_copy_(0, slots, value)
            return out

        return self._layer(states, ids, pos, attend)

    @torch.no_grad()
    def forward_streams(self, caches: Sequence[Cache], states: Sequence[torch.Tensor],
                        tokens: Sequence[Sequence[int]], starts: Sequence[int]) -> torch.Tensor:
        """``forward`` for several streams' rows (at most 128 each, over their own caches) in one call, each row with its bits alone."""

        c = self.w.config
        sizes = [s.shape[0] for s in states]
        for cache, n, p0 in zip(caches, sizes, starts):
            if not 1 <= n <= tree_attention.MAX_NODES or p0 + n > cache.k.shape[0]:
                raise ValueError("MTP rows: 1 to 128 a stream, within its cache")
        device, width = states[0].device, sum(sizes)
        host = [int(t) for ts in tokens for t in ts] + [p for p0, n in zip(starts, sizes) for p in range(p0, p0 + n)]
        dev = torch.tensor(host, dtype=torch.int32).pin_memory().to(device, non_blocking=True)
        ids, pos = dev[:width], dev[width:]
        aplan = tree_attention.plan([list(range(-1, n - 1)) for n in sizes], list(starts), c.heads // c.kv_heads,
                                    device)
        aoffs = torch.tensor(tree_attention.offsets([(x.k, x.v) for x in caches], device), dtype=torch.int64,
                             device=device).view(len(caches), 2)

        def attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            out = tree_attention.attention(q, key, value, aoffs, aplan, scale=c.head_dim ** -0.5)
            dst, src, a0 = [], [], 0
            for cache, n, p0 in zip(caches, sizes, starts):          # each stream's keys at its own slots
                dst += [cache.k[p0:p0 + n], cache.v[p0:p0 + n]]
                src += [key[a0:a0 + n], value[a0:a0 + n]]
                a0 += n
            torch._foreach_copy_(dst, src)
            return out

        return self._layer(states[0] if len(states) == 1 else torch.cat(list(states)), ids, pos, attend)

    def _layer(self, states: torch.Tensor, ids: torch.Tensor, pos: torch.Tensor, attend) -> torch.Tensor:
        """The layer on its rows; ``attend(q, key, value)`` attends and writes the rows' keys where they belong."""

        w, m, c = self.w, self.m, self.w.config
        n = states.shape[0]
        e = glue.embed(ids, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        _, en, exs = glue.add_rmsnorm(e, None, m.norm_e, c.eps)
        _, hn, hxs = glue.add_rmsnorm(states.contiguous(), None, m.norm_h, c.eps)
        x = (matmul(en, m.fc_e, exs).float() + matmul(hn, m.fc_h, hxs).float()).to(torch.bfloat16)
        x, h, xs = glue.add_rmsnorm(x, None, m.input_norm, c.eps)
        a = m.attn
        qg = matmul(h, a.q, xs)
        key = matmul(h, a.k, xs)
        value = matmul(h, a.v, xs).reshape(n, c.kv_heads, c.head_dim).contiguous()
        q, key = glue.attn_prep(qg, key, a.q_norm, a.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                kv_heads=c.kv_heads, head_dim=c.head_dim)
        key = key.view(n, c.kv_heads, c.head_dim).contiguous()
        out = attend(q.view(n, c.heads, c.head_dim).contiguous(), key, value)
        gated, gxs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
        x, h, _ = glue.add_rmsnorm(x, matmul(gated, a.o, gxs), m.post_norm, c.eps)
        _, normed, _ = glue.add_rmsnorm(x, moe.run(h, m.moe), m.norm, c.eps)
        return normed

    def logits(self, normed: torch.Tensor) -> torch.Tensor:
        """Draft-head logits; column j is token ``ids[j]`` with a draft vocabulary, else token j."""

        return matmul(normed.contiguous(), self.head)
