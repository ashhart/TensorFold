"""Fixed-size table rows on local NVMe by row id: a forward's rows read ahead on threads, recent rows kept in memory."""
# A row's bytes are the file's however they arrive, so a reader changes only when rows are ready, never the result.

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Shard:
    """``rows`` rows of ``row_bytes`` bytes each, from byte ``offset`` of ``path``; shards number rows in order."""

    path: str
    offset: int
    rows: int
    row_bytes: int


@dataclass
class Ticket:
    """The rows one ``issue`` asked for, in the order asked."""

    keys: np.ndarray


class RowReader(Protocol):
    def issue(self, keys: np.ndarray) -> Ticket: ...

    def wait(self, ticket: Ticket) -> np.ndarray: ...


class NvmeRows:
    """``RowReader`` over shards: an issue's rows merge into extents (gaps up to ``gap`` bytes read through)."""

    def __init__(
        self,
        shards: Sequence[Shard],
        *,
        reader: Any = None,
        threads: int = 16,
        cache_rows: int = 65536,
        gap: int = 4096,
    ) -> None:
        if not shards or len({s.row_bytes for s in shards}) != 1:
            raise ValueError("a row reader needs shards of one row size")
        self.shards, self.row_bytes = list(shards), shards[0].row_bytes
        self.first = np.cumsum([0] + [s.rows for s in shards])  # each shard's first row id
        self.rows = int(self.first[-1])
        if reader is None:
            from .direct_read import Reader

            reader = Reader()
        self.reader, self.gap, self.cache_rows = reader, int(gap), int(cache_rows)
        self.pool = ThreadPoolExecutor(int(threads), thread_name_prefix="table-rows")
        self.kept: OrderedDict[int, np.ndarray] = OrderedDict()
        self.flight: dict[int, Future] = {}
        self.hits = self.misses = self.reads = 0

    def _where(self, row: int) -> tuple[Shard, int]:
        if not 0 <= row < self.rows:
            raise IndexError(f"row {row} is outside the table's {self.rows} rows")
        i = int(np.searchsorted(self.first, row, side="right")) - 1
        s = self.shards[i]
        return s, s.offset + (row - int(self.first[i])) * self.row_bytes

    def _extents(self, rows: Sequence[int]) -> list[tuple[Shard, int, int, list[int]]]:
        """Sorted rows grouped into (shard, first byte, end byte, rows) runs within one shard."""

        out: list = []
        for row in sorted(rows):
            shard, at = self._where(row)
            if out and out[-1][0] is shard and at - out[-1][2] <= self.gap:
                out[-1][2] = at + self.row_bytes
                out[-1][3].append(row)
            else:
                out.append([shard, at, at + self.row_bytes, [row]])
        return [tuple(e) for e in out]

    def _read(self, shard: Shard, lo: int, hi: int, rows: list[int]) -> dict[int, np.ndarray]:
        raw = self.reader.read(shard.path, lo, hi - lo)
        raw = np.asarray(raw.numpy() if hasattr(raw, "numpy") else raw, dtype=np.uint8)
        return {r: raw[self._where(r)[1] - lo : self._where(r)[1] - lo + self.row_bytes] for r in rows}

    def issue(self, keys) -> Ticket:
        """Start reading every asked row not kept or in flight; the reads run while the caller works."""

        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        todo = {int(r) for r in keys} - self.kept.keys() - self.flight.keys()
        ticket = Ticket(keys)
        for shard, lo, hi, rows in self._extents(todo):
            future = self.pool.submit(self._read, shard, lo, hi, rows)
            self.reads += 1
            for r in rows:
                self.flight[r] = future
        return ticket

    def wait(self, ticket: Ticket) -> np.ndarray:
        """The ticket's rows [n, row bytes] uint8 in key order (a row nobody issued is read now)."""

        out = np.empty((len(ticket.keys), self.row_bytes), dtype=np.uint8)
        missing = {int(r) for r in ticket.keys} - self.kept.keys() - self.flight.keys()
        if missing:
            self.misses += len(missing)
            for shard, lo, hi, rows in self._extents(missing):
                self._keep(self._read(shard, lo, hi, rows))
        for i, r in enumerate(int(k) for k in ticket.keys):
            if r in self.flight:
                self._keep(self.flight[r].result())
            else:
                self.hits += 1
            row = self.kept[r]
            self.kept.move_to_end(r)
            out[i] = row
        self._trim()
        return out

    def _keep(self, got: dict[int, np.ndarray]) -> None:
        for r, row in got.items():
            self.flight.pop(r, None)
            self.kept[r] = np.array(row, copy=True)

    def _trim(self) -> None:
        while len(self.kept) > self.cache_rows:
            self.kept.popitem(last=False)

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)
        self.flight.clear()


__all__ = ["NvmeRows", "RowReader", "Shard", "Ticket"]
