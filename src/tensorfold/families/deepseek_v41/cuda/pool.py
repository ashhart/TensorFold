"""The shared pool of per-token caches: extents of token rows, placed first fit, owned by a live stream and/or holding
kept prompts.

Every stream's compressed entries and indexer keys live in one arena per kv source (``SerialEngine.big``), sized to
``pool_tokens``; an extent covers ``[base, base + size)`` token rows of it (entries ``[base // ratio, (base + size) //
ratio)``). A live stream owns one extent; when it finishes, its extent can stay as a kept prompt (``kept``: the
prompts whose rows it holds, each a prefix of its rows), so free pool rows double as prompt cache until a new extent
needs them. Bases and sizes are multiples of ``ALIGN``, which every compress ratio divides.

The bookkeeping is host-only and deterministic: both ranks apply the same calls in the same order (rank 0 decides and
sends each decision), so their pools stay equal; ``digest`` checks it.

Design and parts of the code (``Extent``, ``room_after``/``resize``/``move``, ``_move``) after Mia's AI Lab
GLM-5.3-Flash TensorFold recipe (Apache-2.0); adapted for TensorFold's DeepSeek-V4.1 engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

ALIGN = 2048


def align_up(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


@dataclass(eq=False)
class Extent:
    """Pool rows [base, base + size). ``owner``: the live stream writing into it (its sid), or None; ``kept``: the
    kept prompts whose rows it holds."""

    base: int
    size: int
    eid: int = 0
    owner: int | None = None
    kept: list = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.base + self.size

    def free(self) -> bool:
        return self.owner is None and not self.kept


class Pool:
    """Extents over ``rows`` pool rows, kept sorted by base; placement is lowest-base first fit."""

    def __init__(self, rows: int) -> None:
        if rows <= 0 or rows % ALIGN:
            raise ValueError(f"a pool of {rows} rows: a positive multiple of {ALIGN}")
        self.rows = rows
        self.extents: list[Extent] = []
        self.next_eid = 0

    # -- queries ---------------------------------------------------------------------------------------------
    def gaps(self, ignore: Iterable[Extent] = ()) -> list[tuple[int, int]]:
        """Free (base, size) runs, ascending; the extents in ``ignore`` count as free."""

        skip = {id(x) for x in ignore}
        out: list[tuple[int, int]] = []
        at = 0
        for x in self.extents:
            if id(x) in skip:
                continue
            if x.base > at:
                out.append((at, x.base - at))
            at = max(at, x.end)
        if at < self.rows:
            out.append((at, self.rows - at))
        merged: list[tuple[int, int]] = []
        for base, n in out:
            if merged and merged[-1][0] + merged[-1][1] == base:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((base, n))
        return merged

    def free_rows(self) -> int:
        return self.rows - sum(x.size for x in self.extents)

    def largest_gap(self) -> int:
        return max((size for _, size in self.gaps()), default=0)

    def place(self, size: int, ignore: Iterable[Extent] = ()) -> int | None:
        """The lowest base with ``size`` free rows (a multiple of ALIGN), or None."""

        if size <= 0 or size % ALIGN:
            raise ValueError(f"an extent of {size} rows: a positive multiple of {ALIGN}")
        for base, room in self.gaps(ignore):
            if room >= size:
                return base
        return None

    def room_after(self, x: Extent) -> int:
        """Free rows right after ``x`` (up to the next extent or the pool's end)."""

        nxt = min((y.base for y in self.extents if y is not x and y.base >= x.end), default=self.rows)
        return nxt - x.end

    def get(self, eid: int) -> Extent:
        for x in self.extents:
            if x.eid == eid:
                return x
        raise KeyError(f"no extent {eid}")

    def find(self, owner: int) -> Extent | None:
        return next((x for x in self.extents if x.owner == owner), None)

    def digest(self) -> int:
        """A short hash of the extents and their kept prompts (two ranks' pools compare equal by it)."""

        h = hashlib.blake2b(digest_size=6)
        for x in self.extents:
            h.update(repr((x.eid, x.base, x.size, x.owner, [getattr(k, "kid", k) for k in x.kept])).encode())
        return int.from_bytes(h.digest(), "little")

    # -- changes ---------------------------------------------------------------------------------------------
    def add(self, base: int, size: int, owner: int | None = None, eid: int | None = None) -> Extent:
        """A new extent at exactly [base, base + size) (rank 0's placement, applied on both ranks)."""

        if base % ALIGN or size % ALIGN or size <= 0 or base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) is not aligned inside the {self.rows}-row pool")
        if eid is not None and eid != self.next_eid:
            raise ValueError(f"extent id {eid}, this pool's next is {self.next_eid} (ranks out of step)")
        if any(base < x.end and x.base < base + size for x in self.extents):
            raise ValueError(f"extent [{base}, {base + size}) overlaps a live one")
        x = Extent(base, size, self.next_eid, owner)
        self.next_eid += 1
        self.extents.append(x)
        self.extents.sort(key=lambda e: e.base)
        return x

    def resize(self, x: Extent, size: int) -> None:
        """Grow in place (into the free rows after it) or shrink from its end."""

        if size <= 0 or size % ALIGN:
            raise ValueError(f"an extent of {size} rows: a positive multiple of {ALIGN}")
        if size > x.size and size - x.size > self.room_after(x):
            raise ValueError(f"extent {x.eid} cannot grow to {size} rows in place")
        x.size = size

    def move(self, x: Extent, base: int, size: int | None = None) -> int:
        """Put ``x`` at ``base`` (with ``size`` rows): bookkeeping only, the caller copies the rows. Returns the old
        base. The new range may overlap ``x``'s own old rows."""

        size = x.size if size is None else size
        if base % ALIGN or size % ALIGN or size <= 0 or base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) is not aligned inside the {self.rows}-row pool")
        if any(y is not x and base < y.end and y.base < base + size for y in self.extents):
            raise ValueError(f"extent {x.eid} cannot move to [{base}, {base + size}): taken")
        old = x.base
        x.base, x.size = base, size
        self.extents.sort(key=lambda e: e.base)
        return old

    def remove(self, x: Extent) -> None:
        self.extents = [y for y in self.extents if y is not x]


def move_rows(t, a: int, b: int, m: int) -> None:
    """t[b:b + m] = t[a:a + m] (rows), safe when the ranges overlap (``t``: a tensor or anything sliceable with
    ``copy_``, e.g. ``Fp8Rows``)."""

    if m <= 0 or a == b:
        return
    gap = abs(a - b)
    if gap >= m:
        t[b:b + m].copy_(t[a:a + m])
        return
    starts = range(0, m, gap) if b < a else reversed(range(0, m, gap))
    for s in starts:                                   # down: front to back; up: back to front
        k = min(gap, m - s)
        t[b + s:b + s + k].copy_(t[a + s:a + s + k])
