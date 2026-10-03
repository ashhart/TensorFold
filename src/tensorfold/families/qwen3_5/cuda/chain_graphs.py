"""One stream with a chain drafter (DSpark): its draft step and the target's chain verify replay as CUDA graphs."""

from __future__ import annotations

import gc
import time
from typing import Callable, Sequence

import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import accept

from .decode import CopyIndex, DecodeResult
from .forward import State, commit, reserve, stage, tree_forward

BUCKET = 8192        # smallest context span a graph covers; larger contexts take the next power of two
COPY_ROWS = 16       # a copied continuation's verify window


class ChainGraphs:
    """A state that every request is copied into, and the target's chain verifies captured per (rows, bucket)."""

    def __init__(self, w, capacity: int) -> None:
        self.w, self.capacity = w, capacity
        self.st, self.rows, self.pool = None, 0, None
        self.target: dict[tuple[int, int], tuple] = {}

    def _bucket(self, end: int) -> int:
        return min(self.rows, max(BUCKET, 1 << (end - 1).bit_length()))

    def load(self, st: State, need: int) -> State:
        """The request's committed state in the fixed buffers, grown (graphs dropped) past ``need`` rows."""

        if need > self.rows:
            self.rows = min(self.capacity, max(BUCKET, 1 << (need - 1).bit_length()))
            self.target.clear()
            self.st = None
            gc.collect()
            self.pool = torch.cuda.graph_pool_handle()      # a fresh pool: torch refuses one left without a graph
            self.st = State(self.w)
            reserve(self.st, self.rows)
        dst = self.st
        for a, b in zip(dst.rec + dst.conv, st.rec + st.conv):
            if a is not None:
                a.copy_(b)
        for kv, src in zip(dst.kv, st.kv):
            if kv is not None:
                kv[0][:st.pos].copy_(src[0][:st.pos])
                kv[1][:st.pos].copy_(src[1][:st.pos])
        dst.pos, dst.rope_delta = st.pos, st.rope_delta
        return dst

    def _capture(self, fn: Callable):
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                                        # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                out = fn()
        finally:
            if enabled:
                gc.enable()
        return g, out

    @torch.no_grad()
    def verify(self, tokens: Sequence[int]):
        """The chain [pending, drafts...] at the stream's position: (logits, record, taps)."""

        width, st = len(tokens), self.st
        key = (width, self._bucket(st.pos + width))
        entry = self.target.get(key)
        parents = list(range(-1, width - 1))
        if entry is None:
            staged = stage(self.w, st, width, key[1])
            staged.refresh(tokens, st.pos)

            def run():
                return tree_forward(self.w, staged.ids, parents, st, capture_taps=True, staged=staged)

            run()                                           # compiles each kernel outside the capture
            g, out = self._capture(run)
            entry = self.target[key] = (g, staged, out)
        g, staged, out = entry
        staged.refresh(tokens, st.pos)
        g.replay()
        return out


@torch.no_grad()
def chain_decode(w, st: State, prompt: Sequence[int], pending: int, count: int, sampling, draft, runner: ChainGraphs,
                 *, allow_copy: bool = True, stop_eos: bool = True,
                 on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult:
    """Each round: a copied continuation or the drafter's chain, verified, its accepted path kept (graphs)."""

    st = runner.load(st, min(runner.capacity, st.pos + count + COPY_ROWS))
    out = [pending]
    context = list(prompt) + out
    copies = CopyIndex() if allow_copy else None
    rounds = drafted = accepted = 0
    widths: list[int] = []
    eos = w.config.eos if stop_eos else ()
    start = time.perf_counter()
    while len(out) < count and out[-1] not in eos:
        guesses = copies.propose(context, COPY_ROWS - 1) if copies is not None else []
        if not guesses:
            guesses = draft.graph_chain(out[-1], max(1, min(COPY_ROWS - 1, count - len(out))))
        tokens = [out[-1]] + guesses
        logits, record, taps = runner.verify(tokens)
        sampled = sample_rows(logits, [st.pos + 1 + i for i in range(len(tokens))], sampling)
        path, terminal = accept(tokens, list(range(-1, len(tokens) - 1)), sampled, count - len(out), eos)
        commit(st, record, path, in_place=True)
        draft.add_taps(taps[path])
        new = [tokens[r] for r in path[1:]] + [terminal]
        out.extend(new)
        context.extend(new)
        rounds, drafted, accepted = rounds + 1, drafted + len(tokens) - 1, accepted + len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None and on_tokens(new):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, rounds, drafted, accepted, widths=widths)
