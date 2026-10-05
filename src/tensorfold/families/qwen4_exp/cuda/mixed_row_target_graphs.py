"""Private exact-total-row target graphs for four dense text chains.

Conceptual RowGraphs reuse: Jay Leaton (@jayleaton), TensorFold PR300. Family-specific tables
and PLE preserve actual rows, original arithmetic and PR338's separate family.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import gc
import os
import time
from types import SimpleNamespace
from typing import Any

from .multi_target_graphs import MAX_CONTEXT, PENDING_WIDTH, _restore, _save_state, _signature

FLAG = "TF_MIXED_ROW_TARGET_GRAPH"
TOTALS = (13, 18)


def requested() -> bool:
    """Read a startup-only default-off flag, rejecting misspelled settings."""

    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    return value == "1"


def eligible(w: Any, b: Any, segs: list, pending: list, streams: list, *,
             depth: int, filling: bool) -> bool:
    """Admit only four nonuniform dense text chains with real total13 or18."""

    if w.comm is not None or getattr(w, "x3", None) is not None or b.prefill or depth != 6 or filling:
        return False
    if b.attn.budget < MAX_CONTEXT:
        return False
    if len(segs) != 4 or len(streams) != 4 or len(pending) != 4:
        return False
    if any(s is None or s.st is None for s in streams):
        return False
    if len({id(s.st) for s in streams}) != 4 or not any(pending):
        return False
    if any(len(rows) > PENDING_WIDTH for rows in pending):
        return False
    widths = [a1 - a0 for _, a0, a1 in segs]
    if len(set(widths)) == 1 or sum(widths) not in TOTALS:
        return False
    limit, at, dtypes = min(MAX_CONTEXT, b.attn.budget), 0, set()
    for (st, a0, a1), s in zip(segs, streams):
        if s.st is not st or s.vision is not None or st.image_positions is not None or s.constraint is not None:
            return False
        if a0 != at or not 1 <= a1 - a0 <= PENDING_WIDTH:
            return False
        if not 1 <= st.pos + a1 - a0 <= min(limit, st.capacity):
            return False
        at = a1
        dtypes.add(st.kv_dtype)
    return len(dtypes) == 1


def capture_inputs(tables: Any, step: Any, segs: list) -> tuple:
    """Fix host launch bounds; real starts, counts and cache pointers stay on GPU."""

    graph_tables, graph_step = copy(tables), copy(step)
    graph_tables.plan = copy(tables.plan)
    graph_tables.plan.max_rows = PENDING_WIDTH
    graph_step.ends, graph_step.most = [MAX_CONTEXT] * 4, PENDING_WIDTH
    proxies = [(SimpleNamespace(lin_index=dict(st.lin_index)), a0, a1) for st, a0, a1 in segs]
    return graph_tables, graph_step, proxies


@dataclass
class _Entry:
    graph: Any
    tables: Any
    step: Any
    output: Any


class MixedRowTargetGraphs:
    """At most four additional entries, independent of ordered width/State tuple."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], _Entry] = {}
        self.tails = None
        self.signature = None
        self.calls = self.eligible_calls = self.eager_calls = 0
        self.captures = self.replays = 0
        self.capture_seconds = 0.0

    @property
    def entries(self) -> int:
        """The total-row family retains at most four entries."""

        return len(self._entries)

    @property
    def retained_tensor_bytes(self) -> int:
        """Explicit tails/tables/output bytes; excludes CUDA graph allocator pools."""

        tensors = [] if self.tails is None else [self.tails]
        for entry in self._entries.values():
            tensors.extend((entry.tables.ints, entry.tables.ptrs, entry.step.ints, entry.step.ptrs, entry.output))
        return sum(t.numel() * t.element_size() for t in tensors)

    def run(self, w: Any, b: Any, segs: list, tables: Any, step: Any, *, admitted: bool) -> Any | None:
        """Refresh fixed tables/tails and replay, or fall back without cache mutation."""

        self.calls += 1
        if not admitted:
            self.eager_calls += 1
            return None
        rows, parity = segs[-1][2], tables.cur
        if (rows not in TOTALS or parity not in (0, 1) or tables.plan.slots != 0 or not tables.folds
                or tables.held.shape != (4, PENDING_WIDTH)):
            raise ValueError("admitted mixed target changed chain geometry")
        self.eligible_calls += 1
        tail_shapes = tuple((tuple(st.ple_tail.shape), str(st.ple_tail.dtype)) for st, _, _ in segs)
        if len(set(tail_shapes)) != 1:
            raise ValueError("mixed graph needs identical tail shapes")
        maps = tuple(tuple(sorted(st.lin_index.items())) for st, _, _ in segs)
        if len(set(maps)) != 1:
            raise ValueError("mixed graph needs one linear-layer index map")
        signature = _signature(w, b, tables.scratch, segs[0][0].kv_dtype) + (tail_shapes, maps)
        if self.signature is not None and signature != self.signature:
            import torch

            torch.cuda.synchronize()
            self._entries.clear()
            self.tails = self.signature = None
            self.eager_calls += 1
            return None
        if self.tails is None:
            import torch

            self.tails = torch.empty((4, *segs[0][0].ple_tail.shape),
                                     dtype=segs[0][0].ple_tail.dtype, device=segs[0][0].ple_tail.device)
            self.signature = signature
        for i, (st, _, _) in enumerate(segs):
            self.tails[i].copy_(st.ple_tail)
        key = rows, parity
        entry = self._entries.get(key)
        if entry is None:
            import torch
            from .forward import compute

            start = time.perf_counter()
            graph_tables, graph_step, proxies = capture_inputs(tables, step, segs)
            saved = _save_state(segs)
            previous_ple = getattr(b, "mixed_ple", None)
            try:
                b.gdn_tables, b.attn_step = graph_tables, graph_step
                b.mixed_ple = self.tails, graph_tables.sid, graph_tables.plan.starts
                compute(w, proxies, b)
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
                    b.gdn_tables, b.attn_step, b.mixed_ple = tables, step, previous_ple
            entry = _Entry(graph, graph_tables, graph_step, output)
            self._entries[key] = entry
            self.signature = _signature(w, b, tables.scratch, segs[0][0].kv_dtype) + (tail_shapes, maps)
            self.captures += 1
            self.capture_seconds += time.perf_counter() - start
        else:
            entry.tables.ints.copy_(tables.ints)
            entry.tables.ptrs.copy_(tables.ptrs)
            entry.step.ints.copy_(step.ints)
            entry.step.ptrs.copy_(step.ptrs)
        b.gdn_tables, b.attn_step = entry.tables, entry.step
        # Python compute is not called during replay; only the recorded pointers matter.
        entry.graph.replay()
        self.replays += 1
        return entry.output
