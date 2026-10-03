"""Concurrent rounds' verify forward replayed as CUDA graphs, one per window width, every row with its eager bits."""

from __future__ import annotations

import dataclasses
import gc
import os
import time
from typing import Callable, Sequence

import torch

from .forward import MultiStaged, State, multi_tree_forward, staged_fits

WIDTHS = (4, 8, 16, 24, 32, 40, 48, 64)   # rows a graph covers; a call takes the narrowest that holds its windows


def mode() -> str:
    """``TF_MULTI_GRAPHS``: ``0`` all eager, ``1`` the widths that time faster at startup, ``force`` every width."""

    return os.environ.get("TF_MULTI_GRAPHS", "1")


def enabled() -> bool:
    return mode() != "0"


def best_ms(fn: Callable[[], object], reps: int = 3) -> float:
    """``fn``'s fastest of ``reps`` timed runs after one untimed, each to the GPU's end (ms)."""

    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return 1e3 * best


def spread(rows: int, lanes: int) -> list[int]:
    """A startup timing's windows: ``rows`` over a stream each few rows (as drafting streams bring), 16 at most each."""

    n = min(lanes, max(-(-rows // 16), -(-rows // 4)))
    return [rows // n + (i < rows % n) for i in range(n)]


def graph_bytes(t: dict, lanes: int, capacity: int, widths: Sequence[int] = WIDTHS) -> int:
    """Device bytes the graphs hold beyond the eager decoder's, for admission."""

    if not enabled() or lanes < 2:
        return 0
    from tensorfold.cuda.geometry import layer_counts

    widest = max((x for x in widths if x <= 16 * lanes), default=0)
    linear, attention = layer_counts(t)
    d, h = int(t["hidden_size"]), int(t["num_attention_heads"])
    hk, hd = int(t["num_key_value_heads"]), int(t.get("head_dim") or d // h)
    nk, nv = int(t["linear_num_key_heads"]), int(t["linear_num_value_heads"])
    dk, dv = int(t["linear_key_head_dim"]), int(t["linear_value_head_dim"])
    row = (int(t["vocab_size"]) * 4 + 2 * d * 2 + attention * 2 * hk * hd * 2
           + linear * (2 * nk * dk * 2 + nv * dv * 2 + 2 * nv * 4 + (2 * nk * dk + nv * dv) * 2))
    partials = widest * h * (hd + 2) * 4 * -(-(capacity + widest) // 512)
    return 2 * widest * row + partials + (64 << 20)


class MultiGraphs:
    """Verify graphs for up to ``lanes`` streams over at most ``context`` committed keys each (one pool for all)."""

    def __init__(self, w, lanes: int, context: int, widths: Sequence[int] = WIDTHS) -> None:
        c = w.config
        self.w, self.lanes, self.context = w, lanes, context
        self.widths = tuple(sorted(x for x in widths if x <= 16 * lanes))   # a window holds 16 rows at most
        self.pool = torch.cuda.graph_pool_handle()
        self.pad_rec = torch.zeros((c.v_heads, c.dv, c.dk), dtype=torch.float32, device=w.norm.device)
        self.entries: dict[int, tuple] = {}
        self.replays = 0
        self.outputs: list[torch.Tensor] | None = None   # every width's outputs, the widest's rows (one copy)
        self.slower: set[int] = set()                  # widths whose replay lost to the eager forward at startup
        self.timings: dict[int, tuple[float, float]] = {}   # width -> (eager ms, replay ms) at its fewest rows

    def width(self, rows: int) -> int | None:
        return next((x for x in self.widths if x >= rows), None)

    def _capture(self, staged: MultiStaged):
        """The staged forward captured, its outputs copied into the shared ``outputs``: (graph, output views)."""

        width = staged.width
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled_gc = gc.isenabled()
        gc.disable()                                  # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                logits, record, hidden = staged.forward()
                views = [t[:width] for t in self.outputs]
                torch._foreach_copy_(views, _flat(logits, record, hidden))
        finally:
            if enabled_gc:
                gc.enable()
        return g, _unflat(views, record)

    def capture(self, width: int, streams: Sequence[tuple[Sequence[int], Sequence[int], State]]) -> None:
        """The graph of ``width`` rows, staged and captured once with these streams (the forward reads, never writes)."""

        if width in self.entries:
            return
        staged = MultiStaged(self.w, width, self.lanes, self.context, self.pad_rec)
        if not staged.fits(streams):
            raise ValueError("capture streams must fit the graph")
        staged.refresh(streams)
        out = staged.forward()                        # compiles each kernel, sizes the scratch buffers
        if self.outputs is None:
            most = self.widths[-1]
            self.outputs = [t.new_empty((most, *t.shape[1:])) for t in _flat(*out)]
        del out
        g, out = self._capture(staged)
        self.entries[width] = (g, staged, out)

    @torch.no_grad()
    def forward(self, streams: Sequence[tuple[Sequence[int], Sequence[int], State]]):
        """``multi_tree_forward(w, streams, hidden=True)`` by a graph replay, or None when the round fits no graph."""

        width = self.width(sum(len(t) for t, _, _ in streams))
        if width is None or width in self.slower or not staged_fits(streams, self.lanes, width, self.context):
            return None
        return self._replay(width, streams)

    def _replay(self, width: int, streams):
        if width not in self.entries:
            self.capture(width, streams)
        g, staged, (logits, record, hidden) = self.entries[width]
        starts = staged.refresh(streams)
        g.replay()
        self.replays += 1
        return logits, record, hidden, starts

    @torch.no_grad()
    def calibrate(self, st: State) -> None:
        """Time each width's replay against the eager forward of the fewest rows it serves; slower widths stay eager."""

        lo = 0
        for width in self.widths:
            windows = [([0] * k, list(range(-1, k - 1)), st) for k in spread(lo + 1, self.lanes)]
            eager = best_ms(lambda: multi_tree_forward(self.w, windows, hidden=True))
            replay = best_ms(lambda: self._replay(width, windows))
            self.timings[width] = (round(eager, 2), round(replay, 2))
            if replay >= eager:
                self.slower.add(width)
            lo = width
        self.replays = 0

    def used(self) -> tuple[int, ...]:
        return tuple(x for x in self.widths if x not in self.slower)


def _flat(logits: torch.Tensor, record: list, hidden: torch.Tensor) -> list[torch.Tensor]:
    """A forward's outputs as one list: logits, every record's fields in order, then the hidden rows."""

    return [logits] + [getattr(r, f.name) for r in record for f in dataclasses.fields(r)] + [hidden]


def _unflat(views: list[torch.Tensor], record: list):
    """``_flat``'s list back as (logits, record, hidden), the record's types from ``record``."""

    out, k = [], 1
    for r in record:
        n = len(dataclasses.fields(r))
        out.append(type(r)(*views[k:k + n]))
        k += n
    return views[0], out, views[-1]
