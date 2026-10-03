"""Concurrent rounds' MTP head calls replayed as CUDA graphs, every row and cache write with the eager call's bits."""

from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.kernels import attention as tree_attention
from tensorfold.families.qwen3_5.cuda.forward import SKIPPED_ITEM, staged_attention
from tensorfold.families.qwen3_5.cuda.multi_graphs import WIDTHS, best_ms, spread

from .mtp import Cache, Head


def fits(caches: Sequence[Cache], sizes: Sequence[int], starts: Sequence[int], lanes: int, width: int,
         context: int) -> bool:
    """Whether a head call fits a graph of ``width`` rows, ``lanes`` streams and ``context`` slots."""

    return (1 <= len(sizes) <= lanes and sum(sizes) <= width
            and all(1 <= n <= tree_attention.MAX_NODES and p0 + n <= min(context, c.k.shape[0])
                    for c, n, p0 in zip(caches, sizes, starts)))


class StagedHeads:
    """A head call's static inputs: rows, attention plan, cache offsets and the rows the draft head scores."""

    def __init__(self, head: Head, width: int, lanes: int, context: int) -> None:
        w = head.w
        c, device = w.config, w.norm.device
        self.head, self.width, self.lanes, self.context = head, width, lanes, context
        self.group = c.heads // c.kv_heads
        S = self.slots = lanes + 1
        W = width
        self.chunks = -(-(context + width) // tree_attention.CHUNK)
        self.n_items = self.chunks * (-(-width * self.group // tree_attention.QUERY_TILE) + S)
        n32 = 2 * W + W + 4 * S + 3 * self.n_items + W + lanes
        self.host = torch.zeros(n32, dtype=torch.int32).pin_memory()
        self.dev = self.host.to(device)
        self.host64 = torch.zeros(2 * S, dtype=torch.int64).pin_memory()
        self.aoffs = self.host64.to(device).view(S, 2)
        self.states = torch.zeros((W, c.hidden), dtype=torch.bfloat16, device=device)
        self.items = 0                               # item words the last refresh wrote (the rest are skipped)
        at = 2 * W + W + 4 * S
        self.host.numpy()[at:at + 3 * self.n_items] = np.tile(np.array(SKIPPED_ITEM, dtype=np.int32), self.n_items)

    def refresh(self, caches: Sequence[Cache], states: Sequence[torch.Tensor], tokens: Sequence[Sequence[int]],
                starts: Sequence[int], pick: Sequence[int]) -> None:
        W, S = self.width, self.slots
        sizes = [len(t) for t in tokens]
        rows = sum(sizes)
        pad = W - rows
        chains = [list(range(-1, n - 1)) for n in sizes] + ([list(range(-1, pad - 1))] if pad else [])
        flat, _, _ = tree_attention.plan_host(chains, list(starts) + ([0] if pad else []), self.group)
        used = len(chains)
        h = self.host.numpy()
        h[:W] = [int(t) for ts in tokens for t in ts] + [0] * pad
        h[W:2 * W] = [p for p0, n in zip(starts, sizes) for p in range(p0, p0 + n)] + list(range(pad))
        self.items = staged_attention(h, 2 * W, flat, W, used, S, self.n_items, self.items)
        end = 2 * W + W + 4 * S + 3 * self.n_items + W
        h[end:end + len(pick)] = pick
        h[end + len(pick):] = 0
        self.host64.numpy()[:] = tree_attention.offsets([(x.k, x.v) for x in caches], self.states.device) \
            + [0, 0] * (S - len(caches))
        self.dev.copy_(self.host, non_blocking=True)
        self.aoffs.copy_(self.host64.view(S, 2), non_blocking=True)
        torch.cat(list(states), out=self.states[:rows]) if len(states) > 1 else self.states[:rows].copy_(states[0])

    def forward(self):
        """The staged call (what a graph captures): (normed rows, keys, values, picked rows, their draft logits)."""

        head, W, S = self.head, self.width, self.slots
        c = head.w.config
        ids, pos = self.dev[:W], self.dev[W:2 * W]
        n_attn = W + 4 * S + 3 * self.n_items + W
        aplan = tree_attention.from_packed(self.dev[2 * W:2 * W + n_attn], S, W, self.n_items, self.chunks)
        pick = self.dev[2 * W + n_attn:]
        kv = []

        def attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            kv.extend((key, value))
            return tree_attention.attention(q, key, value, self.aoffs, aplan, scale=c.head_dim ** -0.5)

        normed = head._layer(self.states, ids, pos, attend)
        picked = normed.index_select(0, pick)
        return normed, kv[0], kv[1], picked, head.logits(picked)


class HeadGraphs:
    """Head-call graphs for up to ``lanes`` streams over at most ``context`` cache slots each (one pool for all)."""

    def __init__(self, head: Head, lanes: int, context: int, widths: Sequence[int] = WIDTHS) -> None:
        self.head, self.lanes, self.context = head, lanes, context
        self.widths = tuple(sorted(x for x in widths if x <= 16 * lanes))
        self.pool = torch.cuda.graph_pool_handle()
        self.entries: dict[int, tuple] = {}
        self.replays = 0
        self.slower: set[int] = set()                  # widths whose replay lost to the eager call at startup
        self.timings: dict[int, tuple[float, float]] = {}

    def width(self, rows: int) -> int | None:
        return next((x for x in self.widths if x >= rows), None)

    def capture(self, width: int, caches, states, tokens, starts, pick) -> None:
        if width in self.entries:
            return
        staged = StagedHeads(self.head, width, self.lanes, self.context)
        staged.refresh(caches, states, tokens, starts, pick)
        staged.forward()                              # compiles each kernel, sizes the scratch buffers
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                                  # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                out = staged.forward()
        finally:
            if enabled:
                gc.enable()
        self.entries[width] = (g, staged, out)

    @torch.no_grad()
    def call(self, caches: Sequence[Cache], states: Sequence[torch.Tensor], tokens: Sequence[Sequence[int]],
             starts: Sequence[int], pick: Sequence[int]):
        """``forward_streams`` and ``head.logits`` of rows ``pick`` by a replay, or None when the call fits no graph."""

        sizes = [len(t) for t in tokens]
        width = self.width(sum(sizes))
        if width is None or width in self.slower or len(pick) > self.lanes or not fits(caches, sizes, starts,
                                                                                        self.lanes, width,
                                                                                        self.context):
            return None
        return self._replay(width, caches, states, tokens, starts, pick)

    def _replay(self, width, caches, states, tokens, starts, pick):
        sizes = [len(t) for t in tokens]
        if width not in self.entries:
            self.capture(width, caches, states, tokens, starts, pick)
        g, staged, (_, key, value, picked, logits) = self.entries[width]
        staged.refresh(caches, states, tokens, starts, pick)
        g.replay()
        self.replays += 1
        dst, src, a0 = [], [], 0
        for cache, n, p0 in zip(caches, sizes, starts):          # each stream's keys at its own slots
            dst += [cache.k[p0:p0 + n], cache.v[p0:p0 + n]]
            src += [key[a0:a0 + n], value[a0:a0 + n]]
            a0 += n
        torch._foreach_copy_(dst, src)
        return picked[:len(pick)], logits[:len(pick)]

    @torch.no_grad()
    def calibrate(self) -> None:
        """Time each width's replay against the eager call of the fewest rows it serves; slower widths stay eager."""

        w = self.head.w
        lo = 0
        for width in self.widths:
            sizes = spread(lo + 1, self.lanes)
            caches = [Cache(w, 16) for _ in sizes]
            states = [torch.zeros((k, w.config.hidden), dtype=torch.bfloat16, device=w.norm.device) for k in sizes]
            tokens = [[0] * k for k in sizes]
            pick = [sum(sizes[:i + 1]) - 1 for i in range(len(sizes))]
            index = torch.tensor(pick, device=w.norm.device)

            def eager():
                normed = self.head.forward_streams(caches, states, tokens, [0] * len(sizes))
                return self.head.logits(normed.index_select(0, index))

            first = best_ms(eager)
            replay = best_ms(lambda: self._replay(width, caches, states, tokens, [0] * len(sizes), pick))
            self.timings[width] = (round(first, 2), round(replay, 2))
            if replay >= first:
                self.slower.add(width)
            lo = width
        self.replays = 0

    def used(self) -> tuple[int, ...]:
        return tuple(x for x in self.widths if x not in self.slower)
