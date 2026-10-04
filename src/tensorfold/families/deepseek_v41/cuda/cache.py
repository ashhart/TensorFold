"""Kept prompt snapshots of one rank in one arena allocated at startup, kept and dropped alike on both ranks."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from . import snapshot
from .snapshot import ALIGN, Snapshot

ENTRIES_ENV = "TF_DSV41_CACHE_ENTRIES"
ENTRIES = 8                 # kept snapshots at most, unless ENTRIES_ENV says otherwise


def entries_wanted(environ: Mapping[str, str] | None = None) -> int:
    """Snapshots ENTRIES_ENV allows; ValueError naming the variable for a negative or non-integer value."""

    value = (os.environ if environ is None else environ).get(ENTRIES_ENV, "").strip()
    try:
        n = int(value) if value else ENTRIES
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(f"{ENTRIES_ENV}={value}: a whole number of snapshots, 0 or more")
    return n


class Arena:
    """One uint8 device buffer handed out first fit in ALIGN-rounded spans; nothing is allocated after construction."""

    def __init__(self, nbytes: int, device: torch.device | str) -> None:
        self.buf = torch.empty((nbytes // ALIGN * ALIGN,), dtype=torch.uint8, device=device)
        self.spans: dict[int, int] = {}     # start -> length

    def take(self, nbytes: int) -> int | None:
        """The lowest free span of ``nbytes`` (rounded up to ALIGN) -> its start, or None when no gap holds it."""

        size, at = math.ceil(nbytes / ALIGN) * ALIGN, 0
        for start in sorted(self.spans):
            if start - at >= size:
                break
            at = start + self.spans[start]
        if at + size > self.buf.numel():
            return None
        self.spans[at] = size
        return at

    def view(self, at: int) -> torch.Tensor:
        return self.buf[at:at + self.spans[at]]

    def give(self, at: int | None) -> None:
        if at is not None:
            del self.spans[at]


@dataclass(eq=False)
class Entry:
    snap: Snapshot
    state: int                  # arena span of the rings and tails
    rows: int | None = None     # arena span of the saved cache rows; None while they are live


class Kept:
    """``e``'s kept snapshots, least recently used first, and the ids its live caches hold."""

    def __init__(self, e, nbytes: int, entries: int) -> None:
        self.e, self.entries = e, entries
        self.arena = Arena(nbytes, e.st.device)
        self.state = snapshot.state_bytes(e)
        self.cache: list[Entry] = []
        self.live: list[int] = []
        self._reserved: int | None = None

    def _entry(self, snap: Snapshot) -> Entry | None:
        return next((x for x in self.cache if x.snap is snap), None)

    def lookup(self, prompt: Sequence[int]) -> Snapshot | None:
        """Rank 0: the longest kept snapshot of a strict prefix of ``prompt``."""

        best = None
        for x in self.cache:
            n = len(x.snap.ids)
            if n < len(prompt) and list(prompt[:n]) == x.snap.ids and (best is None or n > len(best.ids)):
                best = x.snap
        return best

    def find(self, prompt: Sequence[int], cached: int) -> Snapshot:
        """Rank 1: the snapshot of ``prompt[:cached]`` rank 0 resumes from."""

        for x in self.cache:
            if len(x.snap.ids) == cached and list(prompt[:cached]) == x.snap.ids:
                return x.snap
        raise RuntimeError(f"rank 1 has no snapshot of the {cached} tokens rank 0 resumes from")

    def _drop(self, x: Entry) -> None:
        self.arena.give(x.state)
        self.arena.give(x.rows)
        x.snap.rows = None
        self.cache.remove(x)

    def _room(self, nbytes: int, keep: Callable[[Entry], bool]) -> int | None:
        """A span of ``nbytes``, dropping the least recently used entries ``keep`` spares until one fits."""

        while (at := self.arena.take(nbytes)) is None:
            old = next((x for x in self.cache if not keep(x)), None)
            if old is None:
                return None
            self._drop(old)
        return at

    def reserve(self, hit: Snapshot | None) -> torch.Tensor | None:
        """The span a new keep point's snapshot is copied into, or None when the arena cannot hold one."""

        if self.entries == 0:
            return None
        def spare(x: Entry) -> bool:
            return x.snap is hit

        while len(self.cache) >= self.entries:
            old = next((x for x in self.cache if not spare(x)), None)
            if old is None:
                return None
            self._drop(old)
        self._reserved = self._room(self.state, spare)
        return None if self._reserved is None else self.arena.view(self._reserved)

    def remember(self, snap: Snapshot) -> None:
        """``prefill``'s keep: a new snapshot takes the reserved span; a resumed one moves last."""

        x = self._entry(snap)
        if x is None:
            if self._reserved is None:
                raise RuntimeError("a kept snapshot without a reserved span")
            x, self._reserved = Entry(snap, self._reserved), None
        else:
            self.cache.remove(x)
        self.cache.append(x)

    def settle(self) -> None:
        """Give back a reservation no snapshot took (the keep point was the resume point, or the prefill failed)."""

        self.arena.give(self._reserved)
        self._reserved = None

    def take_over(self, hit: Snapshot | None) -> None:
        """Before a prefill from ``hit``: save the rows of snapshots it does not resume, then restore its own."""

        e, keep = self.e, hit.ids if hit is not None else []

        def resumes(x: Entry) -> bool:
            n = len(x.snap.ids)
            return n <= len(keep) and keep[:n] == x.snap.ids

        for x in list(self.cache):
            if x not in self.cache or x.rows is not None or resumes(x):
                continue
            n = len(x.snap.ids)
            if self.live[:n] != x.snap.ids:
                self._drop(x)
                continue
            at = self._room(snapshot.row_bytes(e, x.snap), lambda y, x=x: y is x or resumes(y))
            if at is None:
                self._drop(x)
                continue
            x.rows = at
            snapshot.save_rows(e, x.snap, self.arena.view(at))
        if hit is not None:
            x = self._entry(hit)
            if x is None:
                raise RuntimeError("a resume from a snapshot that is not kept")
            if x.rows is not None:
                snapshot.load_rows(e, hit)
                self.arena.give(x.rows)
                x.rows, hit.rows = None, None
        self.live = []

    def held(self) -> list[tuple[list[int], int, int | None]]:
        """(ids, state span, rows span) of each kept snapshot, least recently used first."""

        return [(x.snap.ids, x.state, x.rows) for x in self.cache]
