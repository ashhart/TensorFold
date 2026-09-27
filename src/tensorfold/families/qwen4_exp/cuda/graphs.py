"""CUDA graphs for Flash Next's decode steps: one ~1,100-kernel forward becomes one graph launch.

A forward's GPU work (``forward.compute``) reads only static buffers and device-side positions, so it can
be captured once per window size and replayed with new inputs staged beforehand (token ids and the n-gram
rows, ``forward.stage``). DeltaNet layers read their committed state from one of two buffers and write the
new state to the other; every commit flips all layers together, so there are two graphs per window size
(one per parity). Graphs also include the live context bucket; the MTP head has no DeltaNet parity.

Replaying a graph runs the same kernels with the same launch parameters as the eager call, so it gives the
same bits (checked in ``tests/cuda/test_flashnext_forward.py`` against eager decoding).
"""

from __future__ import annotations

import torch

from .forward import compute, stage
from .mtp import mtp_compute, mtp_stage


class Graphs:
    def __init__(self, e, *, max_rows: int = 8) -> None:
        self.e = e
        self.max_rows = max_rows
        self.main: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp_out: dict[tuple[int, int], torch.Tensor] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        import gc

        torch.cuda.synchronize()
        gc.collect()
        g = torch.cuda.CUDAGraph()
        # no garbage collection while capturing: collecting an old engine's graphs calls cuGraphExecDestroy, which
        # invalidates the capture (seen when one process built a second engine)
        enabled = gc.isenabled()
        gc.disable()
        try:
            # thread-local: NCCL's helper threads (tensor parallel) may call CUDA while this thread captures
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        return g

    def _bucket(self, end: int) -> int:
        return min(self.e.st.capacity, max(8192, 1 << (end - 1).bit_length()))

    @torch.no_grad()
    def forward(self, tokens) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.buf
        R = stage(w, st, b, tokens)
        if R > self.max_rows:
            return compute(w, st, b, R)
        context = self._bucket(st.pos + R)
        key = (R, st.cur[0] if st.cur else 0, context)
        g = self.main.get(key)
        if g is None:
            compute(w, st, b, R, context=context)     # eager warm-up: compiles this launch shape
            g = self._capture(lambda: compute(w, st, b, R, context=context))
            self.main[key] = g
        g.replay()
        return b.logits[:R]

    @torch.no_grad()
    def mtp_forward(self, next_tokens, streams: torch.Tensor) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.mbuf
        n = mtp_stage(w, st, b, next_tokens, streams)
        if n > self.max_rows:
            return mtp_compute(w, st, b, n)
        context = self._bucket(st.mtp_len + n)
        key = (n, context)
        g = self.mtp.get(key)
        if g is None:
            out = mtp_compute(w, st, b, n, context=context)
            g = self._capture(lambda: mtp_compute(w, st, b, n, context=context))
            self.mtp[key] = g
            self.mtp_out[key] = out
        g.replay()
        return self.mtp_out[key]

    @torch.no_grad()
    def warm(self, rows: int | None = None) -> int:
        """Capture every decode shape up front (windows of 1..rows rows at both DeltaNet parities, MTP steps of
        1..rows rows), so no capture lands inside a timed run. Leaves the sequence state dirty: prefill after."""

        e = self.e
        st = e.st
        rows = rows or self.max_rows
        saved = list(st.cur)
        before = self.captures
        for parity in (0, 1):
            st.cur = [parity] * len(st.cur)
            for R in range(1, rows + 1):
                self.forward([0] * R)
        st.cur = saved
        if e.mbuf is not None:
            for n in range(1, rows + 1):
                self.mtp_forward([0] * n, e.buf.streams[:n])
        torch.cuda.synchronize()
        return self.captures - before

