"""Stored prompt prefixes: which points of a prompt to keep for the next turn, the in-memory store, and the
conversations saved at shutdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def choose_checkpoints(
    history_len: int, cached: int, last_prompt: list[int] | None, prompt: list[int]
) -> list[int]:
    """Where to snapshot this request's prefill for the NEXT turn.

    The rendered history boundary is the natural candidate: the generation
    prompt's tail does not survive re-rendering, the history does. When the
    same conversation's previous prompt shares most but not all of that
    history, the client may be appending a per-turn block (environment,
    timestamps, tool results) that will not be there next turn, so the
    stable prefix is a second candidate, as in the single-stream server's
    clamp. Both are kept: with an empty-think-block template the stable
    prefix is merely last turn's boundary and falls inside the reused part,
    while with a per-turn block only the stable prefix will match. Nothing
    inside the reused part, nothing at or past the prompt's end.
    """

    candidates = {int(history_len)}
    if last_prompt:
        stable = longest_common_prefix(last_prompt, prompt)
        if 0 < stable < int(history_len) and stable >= int(history_len) // 2:
            candidates.add(stable)
    return sorted(at for at in candidates if int(cached) < at < len(prompt))


@dataclass
class CheckpointEntry:
    tokens: list[int]
    cache: list[Any]
    last_prompt: list[int]
    nbytes: int = 0
    # A system block (loaded from disk, or saved to it): outside the slot count.
    pinned: bool = False


def save_conversations(store: "CheckpointStore", directory: Path, model_id: str, *, keep: int = 2,
                       limit_bytes: int = 10 * 1024**3) -> int:
    """Write the store's most recently used conversation entries to ``directory``; returns how many."""

    from tensorfold.engine.prefix_snapshots import save_snapshot

    with store._lock:
        entries = [entry for entry in store._entries if not entry.pinned]   # most recently used first
    entries.sort(key=lambda entry: -len(entry.tokens))
    saved = total = 0
    for entry in entries:
        if saved >= keep or total + entry.nbytes > limit_bytes:
            break
        started = time.perf_counter()
        try:
            save_snapshot(directory, model_id, entry.tokens, entry.cache, keep=keep)
        except Exception as exc:  # noqa: BLE001 - a full disk must not hang the shutdown
            print(f"[lanes] conversation save failed: {type(exc).__name__}: {exc}", flush=True)
            break
        saved += 1
        total += entry.nbytes
        print(f"[lanes] saved conversation checkpoint tokens={len(entry.tokens)} "
              f"({entry.nbytes / 1024**3:.1f} GiB) in {time.perf_counter() - started:.1f}s", flush=True)
    return saved


class CheckpointStore:
    """LRU of absorbed conversation prefixes with their single-row caches."""

    def __init__(
        self,
        slots: int,
        copier: Callable[[list[Any]], list[Any]],
        *,
        budget_bytes: int | None = None,
        sizer: Callable[[list[Any]], int] | None = None,
        pinned_slots: int = 3,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be positive")
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be positive when given")
        self.slots = int(slots)
        self.pinned_slots = int(pinned_slots)
        self.copier = copier
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.sizer = sizer
        self._entries: list[CheckpointEntry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # set when a memory controller evicts on demand: the newest entry may then exceed the byte budget
        self.admit_oversize = False

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries)

    def _best(self, prompt: list[int], usable: Any) -> CheckpointEntry | None:
        best: CheckpointEntry | None = None
        for entry in self._entries:
            tokens = entry.tokens
            if usable is not None and not usable(len(tokens)):
                continue
            if 0 < len(tokens) < len(prompt) and prompt[: len(tokens)] == tokens:
                if best is None or len(tokens) > len(best.tokens):
                    best = entry
        return best

    def peek(self, prompt: list[int], usable: Any = None) -> CheckpointEntry | None:
        """The entry ``match`` would use, without counting a hit or copying."""

        with self._lock:
            return self._best(prompt, usable)

    def match(self, prompt: list[int], usable: Any = None, *,
              take: bool = False) -> tuple[int, list[Any], list[int]] | None:
        """Longest strict-prefix hit as (length, cache, previous prompt), of lengths ``usable`` accepts.

        The cache is a copy, or with ``take`` the stored arrays themselves, removed from the store (same bits)."""

        with self._lock:
            best = self._best(prompt, usable)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.remove(best)
            previous = list(best.last_prompt)
            if take:
                return len(best.tokens), best.cache, previous
            self._entries.insert(0, best)
            best.last_prompt = list(prompt)
            return len(best.tokens), self.copier(best.cache), previous

    def longest(self, prompt: list[int]) -> int:
        """Length of the longest strict-prefix entry (0 if none); not counted as a hit or miss."""

        with self._lock:
            return max((len(entry.tokens) for entry in self._entries
                        if 0 < len(entry.tokens) < len(prompt) and prompt[: len(entry.tokens)] == entry.tokens),
                       default=0)

    def insert(self, tokens: list[int], cache: list[Any], *, last_prompt: list[int],
               pinned: bool = False) -> None:
        if not tokens:
            return
        nbytes = int(self.sizer(cache)) if self.sizer is not None else 0
        oversize = self.budget_bytes is not None and nbytes > self.budget_bytes
        if oversize and not self.admit_oversize:
            return
        with self._lock:
            replaced = [entry for entry in self._entries if entry.tokens == list(tokens)]
            kept = [entry for entry in self._entries if entry.tokens != list(tokens)]
            pinned = pinned or any(entry.pinned for entry in replaced)
            entry = CheckpointEntry(list(tokens), cache, list(last_prompt), nbytes, pinned)
            entries = [entry, *kept]
            for extra in [e for e in entries if e.pinned][self.pinned_slots:]:
                extra.pinned = False
            # an oversized newest entry displaces conversations but keeps the pinned system blocks
            limit = self.budget_bytes
            if oversize:
                limit = nbytes + sum(e.nbytes for e in entries[1:] if e.pinned)
            # the new entry itself is never evicted: least recently used first, conversations
            # before system blocks
            while True:
                over_slots = sum(1 for e in entries if not e.pinned) > self.slots
                over_budget = (limit is not None and len(entries) > 1
                               and sum(e.nbytes for e in entries) > limit)
                if not (over_slots or over_budget):
                    break
                unpinned = [i for i in range(1, len(entries)) if not entries[i].pinned]
                if unpinned:
                    entries.pop(unpinned[-1])
                elif over_budget:
                    entries.pop()
                else:
                    break
                self.evictions += 1
            self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def evict_one(self, keep: CheckpointEntry | None = None) -> bool:
        """Release the oldest ordinary prefix first, then a pinned prefix when memory needs it; never ``keep``."""

        with self._lock:
            candidates = [i for i, entry in enumerate(self._entries) if entry is not keep]
            if not candidates:
                return False
            ordinary = [i for i in candidates if not self._entries[i].pinned]
            self._entries.pop(ordinary[-1] if ordinary else candidates[-1])
            self.evictions += 1
            return True
