"""Prompt-end states kept past the device: a ``PrefixCache`` whose evicted entries move to host memory or to disk."""
# An entry's key hashes its ids and the engine's compat, so an entry from another build or mode never resumes.

from __future__ import annotations

import hashlib
import json
from array import array
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from .streams import PrefixCache


class StateCodec(Protocol):
    """A family's kept state as host arrays and back; the arrays alone must rebuild it."""

    def to_host(self, state: Any) -> dict[str, np.ndarray]: ...

    def from_host(self, arrays: dict[str, np.ndarray]) -> Any: ...


class Tier(Protocol):
    """A store below the device: entries by key, each the ids it holds and their state's arrays."""

    used: int
    limit: int

    def find(self, prompt: Sequence[int]) -> tuple[str, int] | None: ...

    def put(self, key: str, ids: Sequence[int], arrays: dict[str, np.ndarray]) -> bool: ...

    def get(self, key: str) -> tuple[list[int], dict[str, np.ndarray]]: ...

    def drop(self, key: str) -> None: ...

    # Optional: ``lengths()`` and ``has(key)`` let a lookup hash the prompt's prefixes instead of scanning entries;
    # ``take_displaced()`` hands back the entries a ``put`` pushed out, for the next tier down.


def compat_hash(compat: dict) -> str:
    return hashlib.sha256(json.dumps(compat, sort_keys=True, default=str).encode()).hexdigest()


def entry_key(compat: str, ids: Sequence[int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(compat.encode())
    h.update(array("i", [int(t) for t in ids]).tobytes())
    return h.hexdigest()


def strict_prefix(ids: Sequence[int], prompt: Sequence[int]) -> bool:
    """``ids`` start ``prompt`` and leave at least one of its tokens."""

    return 0 < len(ids) < len(prompt) and list(prompt[: len(ids)]) == list(ids)


def nbytes(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(a.nbytes for a in arrays.values()))


class HostTier:
    """Entries' arrays in host memory, least recently used first out past ``limit`` bytes (handed back for disk)."""

    def __init__(self, limit: int, *, min_tokens: int = 1) -> None:
        self.limit, self.min_tokens, self.used = int(limit), int(min_tokens), 0
        self.entries: OrderedDict[str, tuple[list[int], dict]] = OrderedDict()
        self.displaced: list[tuple[str, list[int], dict]] = []

    def find(self, prompt: Sequence[int]) -> tuple[str, int] | None:
        best = None
        for key, (ids, _) in self.entries.items():
            if strict_prefix(ids, prompt) and (best is None or len(ids) > best[1]):
                best = (key, len(ids))
        return best

    def put(self, key: str, ids: Sequence[int], arrays: dict[str, np.ndarray]) -> bool:
        size = nbytes(arrays)
        if len(ids) < self.min_tokens or size > self.limit:
            return False
        self.drop(key)
        self.entries[key] = ([int(t) for t in ids], {k: np.array(v, copy=True) for k, v in arrays.items()})
        self.used += size
        while self.used > self.limit:
            old = next(iter(self.entries))
            self.displaced.append((old, *self.entries[old]))
            self.drop(old)
        return True

    def take_displaced(self) -> list[tuple[str, list[int], dict]]:
        out, self.displaced = self.displaced, []
        return out

    def lengths(self) -> set[int]:
        return {len(ids) for ids, _ in self.entries.values()}

    def has(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> tuple[list[int], dict[str, np.ndarray]]:
        self.entries.move_to_end(key)
        ids, arrays = self.entries[key]
        return list(ids), arrays

    def drop(self, key: str) -> None:
        gone = self.entries.pop(key, None)
        if gone is not None:
            self.used -= nbytes(gone[1])

    def keys(self) -> list[str]:
        return list(self.entries)


class TieredCache(PrefixCache):
    """``PrefixCache`` on the device; an evicted entry goes to the first tier taking it, then ``release`` frees it."""

    def __init__(
        self,
        keep: int = 8,
        *,
        codec: StateCodec | None = None,
        tiers: Sequence[Tier] = (),
        compat: dict | None = None,
        release: Callable[[Any], None] | None = None,
    ) -> None:
        super().__init__(keep)
        if tiers and codec is None:
            raise ValueError("tiers below the device need a codec for the entries' state")
        self.codec, self.tiers, self.release = codec, list(tiers), release
        self.compat = compat_hash(compat or {})
        self.spilled = self.dropped = 0

    def key(self, ids: Sequence[int]) -> str:
        return entry_key(self.compat, ids)

    def add(self, ids: list[int], state: Any, snap: Any) -> None:
        """Newest last; an entry for the same ids already kept stays (this one is released)."""

        same = next((e for e in self.entries if e[0] == list(ids)), None)
        if same is not None:
            self._touch(same)
            self._free(state)
            return
        self.entries.append((list(ids), state, snap))
        while len(self.entries) > self.keep:  # keep 0: the entry goes straight to the tiers
            self._drop(self.entries[:-1] or self.entries)

    def find(self, prompt: Sequence[int]) -> tuple[int, int] | None:
        """(length, tier) of the longest entry ``prompt`` strictly extends, tier -1 on the device; touches nothing."""

        best = None
        for ids, _, _ in self.entries:
            if strict_prefix(ids, prompt) and (best is None or len(ids) > best[0]):
                best = (len(ids), -1)
        for i, tier in enumerate(self.tiers):
            hit = self._tier_find(tier, prompt)
            if hit is not None and (best is None or hit[1] > best[0]):
                best = (hit[1], i)
        return best

    def _tier_find(self, tier: Tier, prompt: Sequence[int]) -> tuple[str, int] | None:
        """A tier's longest entry the prompt strictly extends: by key per stored length where the tier can say."""

        if not (callable(getattr(tier, "lengths", None)) and callable(getattr(tier, "has", None))):
            return tier.find(prompt)
        for n in sorted((n for n in tier.lengths() if 0 < n < len(prompt)), reverse=True):
            key = self.key(prompt[:n])
            if tier.has(key):
                return key, n
        return None

    def load(self, prompt: Sequence[int], length: int, tier: int) -> Any:
        """The state a tier keeps for the prompt's first ``length`` ids (ValueError when it is gone or damaged)."""

        ids = list(prompt[:length])
        key = self.key(ids)
        got, arrays = self.tiers[tier].get(key)
        if got != ids:
            raise ValueError(f"tier entry {key} holds other ids")
        return self.codec.from_host(arrays)

    def victim(self, keep: Sequence[Sequence[int]] = ()) -> int | None:
        """The index ``evict`` would drop next, never an entry holding ids in ``keep``; None when none is left."""

        pinned = [list(k) for k in keep]
        among = [e for e in self.entries if e[0] not in pinned]
        if not among:
            return None
        return self.entries.index(self._pick(among))

    def drop_at(self, index: int) -> None:
        """Evict ``entries[index]`` (spilled down a tier where one takes it)."""

        self._evict(self.entries[index])

    def _pick(self, among: list):
        cold = [e for e in among if tuple(e[0]) not in self.hit]
        return cold[0] if cold else among[0]

    def _drop(self, among: list) -> None:
        self._evict(self._pick(among))

    def _evict(self, gone) -> None:
        self.entries = [e for e in self.entries if e is not gone]
        self.hit &= {tuple(e[0]) for e in self.entries}
        ids, state, _ = gone
        if self.tiers:
            if self._spill(0, self.key(ids), ids, self.codec.to_host(state)):
                self.spilled += 1
            else:
                self.dropped += 1
        self._free(state)

    def _spill(self, start: int, key: str, ids: Sequence[int], arrays: dict) -> bool:
        """The first tier from ``start`` that takes the entry; what that tier pushed out moves on to the tiers below."""

        for i in range(start, len(self.tiers)):
            tier = self.tiers[i]
            if tier.put(key, ids, arrays):
                take = getattr(tier, "take_displaced", None)
                for old_key, old_ids, old_arrays in (take() if callable(take) else []):
                    if not self._spill(i + 1, old_key, old_ids, old_arrays):
                        self.dropped += 1
                return True
        return False

    def _free(self, state: Any) -> None:
        if self.release is not None:
            self.release(state)

    def clear(self) -> None:
        """Drop every device entry without spilling (the tiers keep theirs)."""

        for _, state, _ in self.entries:
            self._free(state)
        self.entries, self.hit = [], set()


__all__ = ["HostTier", "StateCodec", "Tier", "TieredCache", "compat_hash", "entry_key", "strict_prefix"]
