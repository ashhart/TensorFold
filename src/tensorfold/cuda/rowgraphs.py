"""CUDA graphs keyed by a window's padded total rows, so any mix of lanes replays the graph of its row count."""
# Every per-row value a kernel reads is a column of one device table, staged by one copy.

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64)


def bucket(rows: int, buckets: Sequence[int] = BUCKETS) -> int | None:
    """The smallest bucket holding ``rows`` rows; None past the largest."""

    return next((int(b) for b in buckets if b >= rows), None)


def pad(table: np.ndarray, rows: int, write: int | None = None) -> np.ndarray:
    """``table`` [columns, n] to ``rows`` columns of the last real row; column ``write`` of the padding is -1."""

    n = table.shape[1]
    if not 1 <= n <= rows:
        raise ValueError(f"{n} rows do not fit {rows}")
    out = np.empty((table.shape[0], rows), dtype=np.int64)
    out[:, :n] = table
    out[:, n:] = table[:, n - 1 : n]
    if write is not None:
        out[write, n:] = -1
    return out


class RowTable:
    """A device table [columns, rows_max] int64 at a fixed address, written from a pinned host copy in one transfer."""

    def __init__(self, columns: Sequence[str], rows_max: int, device: Any = "cuda") -> None:
        import torch

        self.torch, self.columns, self.rows_max = torch, list(columns), int(rows_max)
        self.cuda = torch.device(device).type == "cuda"
        self.host = torch.zeros((len(self.columns), self.rows_max), dtype=torch.int64, pin_memory=self.cuda)
        self.dev = torch.zeros((len(self.columns), self.rows_max), dtype=torch.int64, device=device)
        self.event = None

    def col(self, name: str, rows: int | None = None):
        """A column's first ``rows`` rows on the device (a view: graphs read it at the same address every replay)."""

        return self.dev[self.columns.index(name), : self.rows_max if rows is None else rows]

    def stage(self, t: np.ndarray) -> None:
        """Write ``t`` [columns, R]; the host copy is reused only once the previous transfer has read it."""

        if t.shape[0] != len(self.columns) or t.shape[1] > self.rows_max:
            raise ValueError(f"a table of {t.shape} for {len(self.columns)} columns of {self.rows_max} rows")
        if self.event is not None:
            self.event.synchronize()
        self.host.numpy()[:, : t.shape[1]] = t
        self.dev.copy_(self.host, non_blocking=self.cuda)
        if self.cuda:
            self.event = self.event or self.torch.cuda.Event()
            self.event.record()


def cuda_capture(fn: Callable[[], None], pool: Any = None):
    """``fn``'s launches as a CUDA graph (``replay()`` runs them again)."""

    import torch

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
        fn()
    return graph


class RowGraphs:
    """``run(rows, ctx)``: replays the (bucket, ctx) graph, or runs eagerly and captures it when every rank can."""

    def __init__(
        self,
        run: Callable[[int], None],
        *,
        buckets: Sequence[int] = BUCKETS,
        floor: Callable[[], bool] = lambda: True,
        agree: Callable[[list[int]], list[int]] = lambda v: v,
        budget_s: float = 60.0,
        most: int = 256,
        capture: Callable | None = None,
    ) -> None:
        self.forward, self.buckets = run, tuple(sorted(buckets))
        self.floor, self.agree, self.budget_s, self.most = floor, agree, float(budget_s), int(most)
        self.capture = capture or cuda_capture
        self.graphs: dict[tuple[int, int], Any] = {}
        self.refused: set[tuple[int, int]] = set()
        self.spent = 0.0
        self.addresses: tuple = ()
        self.replays = self.eager = 0

    def rows(self, rows: int) -> int:
        """The rows to stage for a window of ``rows``: its bucket, or itself past the largest (eager)."""

        return bucket(rows, self.buckets) or int(rows)

    def fingerprint(self, *buffers: Any) -> None:
        """The persistent buffers graphs read; when one has moved, every graph is dropped (captured again on use)."""

        addresses = tuple(int(b.data_ptr()) for b in buffers)
        if addresses != self.addresses:
            self.drop()
            self.addresses = addresses

    def drop(self) -> None:
        self.graphs.clear()
        self.refused.clear()

    def run(self, rows: int, ctx: int = 0) -> str:
        """Run the staged window: "replay", "capture" or "eager". Every rank makes the same calls."""

        r = bucket(rows, self.buckets)
        if r is None:
            self.forward(int(rows))
            self.eager += 1
            return "eager"
        key = (r, int(ctx))
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            self.replays += 1
            return "replay"
        self.forward(r)  # this window's result; it also compiles what the capture needs
        self.eager += 1
        return "capture" if self._capture(key) else "eager"

    def _capture(self, key: tuple[int, int]) -> bool:
        if key in self.refused:
            return False
        ok = int(self.floor() and self.spent < self.budget_s and len(self.graphs) < self.most)
        if not min(self.agree([ok])):  # a capture on one rank only would hang its collectives
            self.refused.add(key)
            return False
        t0 = time.perf_counter()
        self.graphs[key] = self.capture(lambda: self.forward(key[0]))
        self.spent += time.perf_counter() - t0
        return True

    def warm(self, rows: Sequence[int], stage: Callable[[int], None]) -> int:
        """Capture context bucket 0 at each row count (``stage(r)`` writes an ``r``-row table first); the count."""

        done = 0
        for r in sorted({bucket(n, self.buckets) for n in rows} - {None}):
            if (r, 0) in self.graphs:
                continue
            stage(r)
            self.forward(r)
            done += self._capture((r, 0))
        return done


__all__ = ["BUCKETS", "RowGraphs", "RowTable", "bucket", "cuda_capture", "pad"]
