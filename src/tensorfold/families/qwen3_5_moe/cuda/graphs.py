"""Decode buffers that outlive requests; verify chains and MTP calls replay as CUDA graphs per (rows, context bucket)."""

from __future__ import annotations

import gc
from typing import Callable, Sequence

import torch

from tensorfold.families.qwen3_5.cuda.forward import State, reserve, stage, tree_forward

from .mtp import Cache, Head, Staged as HeadStaged

BUCKET = 8192        # smallest context span a graph covers; larger contexts take the next power of two


class Graphs:
    """A state and head cache that every request is copied into (buffers fixed, grown by powers of two up to ``capacity``)."""

    def __init__(self, w, head: Head, capacity: int) -> None:
        self.w, self.head, self.capacity = w, head, capacity
        self.st, self.mc, self.rows = None, None, 0
        self.pool = torch.cuda.graph_pool_handle()
        self.target: dict[tuple[int, int], tuple] = {}
        self.mtp: dict[tuple[int, int], tuple] = {}

    def _bucket(self, end: int) -> int:
        return min(self.rows, max(BUCKET, 1 << (end - 1).bit_length()))

    def load(self, st: State, mc: Cache, need: int) -> tuple[State, Cache]:
        """A request's committed state and head cache in the fixed buffers, grown (graphs dropped) past ``need`` rows."""

        if need > self.rows:
            self.rows = min(self.capacity, max(BUCKET, 1 << (need - 1).bit_length()))
            self.target.clear()
            self.mtp.clear()
            self.st = self.mc = None
            gc.collect()
            self.st = State(self.w)
            reserve(self.st, self.rows)
            self.mc = Cache(self.w, self.rows)
        dst = self.st
        for a, b in zip(dst.rec + dst.conv, st.rec + st.conv):
            if a is not None:
                a.copy_(b)
        for kv, src in zip(dst.kv, st.kv):
            if kv is not None:
                kv[0][:st.pos].copy_(src[0][:st.pos])
                kv[1][:st.pos].copy_(src[1][:st.pos])
        dst.pos = st.pos
        self.mc.k[:mc.pos].copy_(mc.k[:mc.pos])
        self.mc.v[:mc.pos].copy_(mc.v[:mc.pos])
        self.mc.pos = mc.pos
        return dst, self.mc

    def _capture(self, fn: Callable):
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                                    # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                out = fn()
        finally:
            if enabled:
                gc.enable()
        return g, out

    @torch.no_grad()
    def verify(self, tokens: Sequence[int]):
        """The chain [pending, drafts...] at the stream's position: (logits, record, final normed rows)."""

        width, st = len(tokens), self.st
        key = (width, self._bucket(st.pos + width))
        entry = self.target.get(key)
        parents = list(range(-1, width - 1))
        if entry is None:
            staged = stage(self.w, st, width, key[1])
            staged.refresh(tokens, st.pos)
            tree_forward(self.w, staged.ids, parents, st, hidden=True, staged=staged)      # compiles each kernel
            g, out = self._capture(lambda: tree_forward(self.w, staged.ids, parents, st, hidden=True, staged=staged))
            entry = self.target[key] = (g, staged, out)
        g, staged, out = entry
        staged.refresh(tokens, st.pos)
        g.replay()
        return out

    @torch.no_grad()
    def draft(self, states: torch.Tensor, tokens: Sequence[int], p0: int) -> torch.Tensor:
        """``Head.forward`` for these rows at ``p0``, then the draft head's logits of the last row."""

        width = states.shape[0]
        key = (width, self._bucket(p0 + width))
        entry = self.mtp.get(key)
        c = self.w.config
        if entry is None:
            staged = HeadStaged(self.mc, width, key[1], c.heads // c.kv_heads, c.hidden)
            staged.refresh(states, tokens, p0)

            def run():
                normed = self.head.forward(self.mc, staged.states, tokens, p0, staged=staged)
                return normed, self.head.logits(normed[-1:])

            run()
            g, out = self._capture(run)
            entry = self.mtp[key] = (g, staged, out)
        g, staged, out = entry
        staged.refresh(states, tokens, p0)
        g.replay()
        return out
