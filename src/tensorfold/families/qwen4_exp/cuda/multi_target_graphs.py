"""Optional four-stream target graphs for two fixed windows and dense attention."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import gc
import os
import time
from types import SimpleNamespace
from typing import Any

FLAG = "TF_FOUR_STREAM_TARGET_GRAPH"
PENDING_WIDTH = 7
MAX_CONTEXT = 512
WIDTHS = (2, 7)


def requested() -> bool:
    """Read the default-off experimental switch without accepting typos."""

    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    return value == "1"


def eligible(w: Any, b: Any, segs: list, pending: list, streams: list, *,
             depth: int, filling: bool) -> bool:
    """Admit four uniform text windows with bounded, ordered pending folds."""

    if w.comm is not None or getattr(w, "x3", None) is not None or b.prefill or depth != 6 or filling:
        return False
    if len(segs) != 4 or len(streams) != 4 or len(pending) != 4:
        return False
    if any(s is None for s in streams):
        return False
    if len({id(s.st) for s in streams}) != 4 or not any(pending):
        return False
    if any(len(rows) > PENDING_WIDTH for rows in pending):
        return False
    width = segs[0][2] - segs[0][1]
    if width not in WIDTHS:
        return False
    limit = min(MAX_CONTEXT, b.attn.budget)
    dtypes = set()
    for i, ((st, a0, a1), s) in enumerate(zip(segs, streams)):
        if s.st is not st or s.vision is not None or st.image_positions is not None or s.constraint is not None:
            return False
        if (a0, a1) != (i * width, (i + 1) * width):
            return False
        if not 1 <= st.pos + width <= min(limit, st.capacity):
            return False
        dtypes.add(st.kv_dtype)
    return len(dtypes) == 1


def _signature(w: Any, b: Any, scratch: Any, dtype: str) -> tuple:
    """Static scratch and format identity; State pointers are refreshed separately."""

    parts = [id(w), id(w.head), id(b), dtype]
    for obj in (b, b.attn, b.moe, b.moe.plan, scratch):
        parts.append(id(obj))
        for name, value in sorted(vars(obj).items()):
            if name in ("gdn_tables", "attn_step", "parity"):
                continue
            if hasattr(value, "data_ptr"):
                parts.append((name, value.data_ptr(), tuple(value.shape), str(value.dtype)))
            elif isinstance(value, (int, float, bool, str)):
                parts.append((name, value))
    return tuple(parts)


def _save_state(segs: list) -> list:
    """Save active recurrent slices and the bounded attention cache write region."""

    saved = []
    for st, _, _ in segs:
        tensors = [st.rec[cur, li] for li, cur in enumerate(st.cur)]
        rows = min(MAX_CONTEXT, st.capacity)
        for cache, index, pool in zip(st.kc, st.ikc, st.pooled):
            tensors.extend([cache.k[:rows], cache.v[:rows], cache.ks[:rows], cache.vs[:rows],
                            index[:rows], pool[: -(-rows // st.ratio)]])
        saved.extend((t, t.clone()) for t in tensors)
    return saved


def _restore(saved: list) -> None:
    """Restore warm/capture writes before the real replay, including on failure."""

    for dst, src in saved:
        dst.copy_(src)


def _graph_inputs(tables: Any, step: Any, segs: list, tails: list) -> tuple:
    """Make capture-only proxies; graph table copies cannot pin old States afterward."""

    graph_tables, graph_step = copy(tables), copy(step)
    proxies = [(SimpleNamespace(ple_tail=tail, lin_index=dict(st.lin_index)), a0, a1)
               for (st, a0, a1), tail in zip(segs, tails)]
    return graph_tables, graph_step, proxies


@dataclass
class _Entry:
    graph: Any
    tables: Any
    step: Any
    output: Any


class FourStreamTargetGraphs:
    """At most four keys, independent of sequence identities and pending counts."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], _Entry] = {}
        self.tails: list = []
        self.signature: tuple | None = None
        self.calls = self.eligible_calls = self.eager_calls = 0
        self.captures = self.replays = 0
        self.capture_seconds = 0.0

    @property
    def entries(self) -> int:
        """The retained graph family has zero to four entries."""

        return len(self._entries)

    def run(self, w: Any, b: Any, segs: list, tables: Any, step: Any, *, admitted: bool) -> Any | None:
        """Replay an admitted target, or leave the existing eager forward to the caller."""

        self.calls += 1
        if not admitted:
            self.eager_calls += 1
            return None
        self.eligible_calls += 1
        signature = _signature(w, b, tables.scratch, segs[0][0].kv_dtype)
        tail_shapes = tuple((tuple(st.ple_tail.shape), str(st.ple_tail.dtype)) for st, _, _ in segs)
        signature += (tail_shapes,)
        if self.signature is not None and signature != self.signature:
            import torch

            torch.cuda.synchronize()
            self._entries.clear()
            self.tails = []
            self.signature = None
            self.eager_calls += 1
            return None
        if not self.tails:
            import torch

            self.tails = [torch.empty_like(st.ple_tail) for st, _, _ in segs]
            self.signature = signature
        for tail, (st, _, _) in zip(self.tails, segs):
            tail.copy_(st.ple_tail)
        key = (segs[0][2] - segs[0][1], tables.cur)
        if key[0] not in WIDTHS or key[1] not in (0, 1) or tables.held.shape != (4, PENDING_WIDTH):
            raise ValueError("admitted target graph tables changed geometry")
        entry = self._entries.get(key)
        if entry is None:
            import torch
            from .forward import compute

            start = time.perf_counter()
            graph_tables, graph_step, proxies = _graph_inputs(tables, step, segs, self.tails)
            saved = _save_state(segs)
            try:
                b.gdn_tables, b.attn_step = graph_tables, graph_step
                compute(w, proxies, b)  # compile the unchanged target launch shape
                _restore(saved)
                torch.cuda.synchronize()
                gc.collect()
                graph = torch.cuda.CUDAGraph()
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                        output = compute(w, proxies, b)
                finally:
                    if enabled:
                        gc.enable()
            finally:
                try:
                    _restore(saved)
                    torch.cuda.synchronize()
                finally:
                    graph_tables.segs = graph_step.segs = []
                    b.gdn_tables, b.attn_step = tables, step
            entry = _Entry(graph, graph_tables, graph_step, output)
            self._entries[key] = entry
            self.signature = _signature(w, b, tables.scratch, segs[0][0].kv_dtype) + (tail_shapes,)
            self.captures += 1
            self.capture_seconds += time.perf_counter() - start
        else:
            entry.tables.ints.copy_(tables.ints)
            entry.tables.ptrs.copy_(tables.ptrs)
            entry.step.ints.copy_(step.ints)
            entry.step.ptrs.copy_(step.ptrs)
        b.gdn_tables, b.attn_step = entry.tables, entry.step
        entry.graph.replay()
        self.replays += 1
        return entry.output
