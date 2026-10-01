"""Kept prompt states for CUDA engines: which stored prefix a new prompt resumes from, and which to evict.

The pool is model-agnostic. A family supplies the snapshots themselves (what a prefix's state is, how it is copied
out of the live buffers and back into them) and their sizes; the pool keeps them least recently used first within
a byte budget. Every decision depends only on the token ids, the sizes and the order of requests, so ranks that see
the same requests with the same budget keep identical pools without talking to each other.

A snapshot holds the state after exactly its tokens (recurrent state and rings cannot be cut shorter), so an entry
serves a prompt only when its ids are a prefix of that prompt; a family's live caches serve partial matches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entry:
    ids: list[int]
    snapshot: Any
    nbytes: int
    hits: int = 0


@dataclass
class PrefixPool:
    """Stored prompt states within ``budget`` bytes; entries shorter than ``min_tokens`` are not kept."""

    budget: int
    min_tokens: int = 1024
    entries: list[Entry] = field(default_factory=list)      # least recently used first

    @property
    def used(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def match(self, prompt: list[int]) -> Entry | None:
        """The longest entry whose ids are a proper prefix of ``prompt`` (at least one token left to prefill)."""

        best = None
        for e in self.entries:
            n = len(e.ids)
            if n < len(prompt) and (best is None or n > len(best.ids)) and prompt[:n] == e.ids:
                best = e
        if best is not None:
            best.hits += 1
            self.entries.remove(best)
            self.entries.append(best)
        return best

    def wants(self, ids: list[int], nbytes: int) -> bool:
        """Whether a state of ``ids`` (``nbytes``) would be kept: long enough, fits the budget, not already held."""

        if len(ids) < self.min_tokens or nbytes > self.budget:
            return False
        return not any(e.ids == ids for e in self.entries)

    def add(self, ids: list[int], make: Callable[[], tuple[Any, int]]) -> Entry | None:
        """Store the state of ``ids`` (``make()`` copies it out: (snapshot, bytes)), evicting the least recently
        used entries to fit; an entry already holding exactly ``ids`` is refreshed instead."""

        for e in self.entries:
            if e.ids == ids:
                self.entries.remove(e)
                self.entries.append(e)
                return e
        if len(ids) < self.min_tokens:
            return None
        snapshot, nbytes = make()
        if nbytes > self.budget:
            return None
        while self.entries and self.used + nbytes > self.budget:
            self.entries.pop(0)
        entry = Entry(list(ids), snapshot, nbytes)
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self.entries), "used_bytes": self.used, "budget_bytes": self.budget,
                "tokens": [len(e.ids) for e in self.entries]}
