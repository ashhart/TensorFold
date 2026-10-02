"""The shared pool of per-token caches: one contiguous extent of token rows a stream, placed first fit.

Every stream's compressed entries and indexer keys live in one arena per kv source (``SerialEngine.big``), sized to
``pool_tokens``; a stream owns ``[base, base + size)`` token rows of it (entries ``[base // ratio, (base + size) //
ratio)``), so many streams can each reach a long context while together drawing on one budget. Bases and sizes are
multiples of ``ALIGN``, which every compress ratio divides. The allocator is host-only and deterministic: both ranks
replay the same placements (rank 0 sends them with each admission).

Design after Mia's AI Lab GLM-5.3-Flash TensorFold recipe (contiguous extents in one arena, first fit, aligned
extents); written for TensorFold's DeepSeek-V4.1 engine.
"""

from __future__ import annotations

from dataclasses import dataclass

ALIGN = 2048


def align_up(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


@dataclass
class Extent:
    base: int
    size: int
    owner: int          # the stream (sid) holding it


class Pool:
    """Extents of ``rows`` token rows, kept sorted by base; placement is lowest-base first fit."""

    def __init__(self, rows: int) -> None:
        if rows <= 0 or rows % ALIGN:
            raise ValueError(f"a pool of {rows} rows: a positive multiple of {ALIGN}")
        self.rows = rows
        self.extents: list[Extent] = []

    def gaps(self) -> list[tuple[int, int]]:
        """Free (base, size) runs, ascending."""

        out, at = [], 0
        for x in self.extents:
            if x.base > at:
                out.append((at, x.base - at))
            at = x.base + x.size
        if at < self.rows:
            out.append((at, self.rows - at))
        return out

    def free_rows(self) -> int:
        return self.rows - sum(x.size for x in self.extents)

    def largest_gap(self) -> int:
        return max((size for _, size in self.gaps()), default=0)

    def place(self, size: int, owner: int) -> Extent | None:
        """A new extent of ``size`` rows (a multiple of ALIGN) at the lowest base that fits, or None."""

        if size <= 0 or size % ALIGN:
            raise ValueError(f"an extent of {size} rows: a positive multiple of {ALIGN}")
        for base, room in self.gaps():
            if room >= size:
                return self.add(base, size, owner)
        return None

    def add(self, base: int, size: int, owner: int) -> Extent:
        """An extent at a given place (rank 1 replays rank 0's placements); the rows must be free."""

        if base % ALIGN or size % ALIGN or base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) is not aligned inside the {self.rows}-row pool")
        if any(base < x.base + x.size and x.base < base + size for x in self.extents):
            raise ValueError(f"extent [{base}, {base + size}) overlaps a live one")
        x = Extent(base, size, owner)
        self.extents.append(x)
        self.extents.sort(key=lambda e: e.base)
        return x

    def release(self, owner: int) -> None:
        self.extents = [x for x in self.extents if x.owner != owner]

    def find(self, owner: int) -> Extent | None:
        return next((x for x in self.extents if x.owner == owner), None)
