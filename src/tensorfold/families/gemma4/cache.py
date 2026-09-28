"""Gemma 4's KV caches: a ring for sliding-window layers, a growing buffer for full ones, decode writes alternating."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from tensorfold.engine.alternating_kv import AlternatingKVCache

# a window's rows never overwrite a key a kept row still reads while the ring holds window + WINDOW_ROWS slots
WINDOW_ROWS = 128
_slots: dict[int, mx.array] = {}


def _slot(index: int) -> mx.array:
    """A write's first slot as an index array, made once per slot (every sliding layer writes the same ones)."""

    found = _slots.get(index)
    if found is None:
        found = _slots[index] = mx.array([index], dtype=mx.int32)
    return found


class LinearKVCache(AlternatingKVCache):
    """A full-attention layer's keys and values, [1, Hk, capacity, D] from position 0."""

    ring = 0

    def write(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Decode rows [1, Hk, n, D] at positions offset ..: the whole buffers after the write."""

        self.update_and_fetch(keys, values)
        return self.keys, self.values


class RingKVCache:
    """A sliding-window layer's keys and values, position p at slot p % ``ring``: rolling back is moving ``offset``."""

    # buffers besides the current one: a write goes to the one written longest ago, which no step still in flight reads
    SPARES = 2
    # left out of prefix snapshots: the first decode writes rebuild them
    transient = ("spares", "recent")
    spares: tuple = ()             # (keys, values, positions held) of the older buffers
    recent: tuple = ()             # (first position, keys, values) of rows an older buffer lacks

    def __init__(self, window: int, ring: int | None = None) -> None:
        self.window = int(window)
        self.ring = int(ring or window + WINDOW_ROWS)
        self.max_size = self.ring                  # memory accounting: fixed, never a position at a time
        self.ring_keys: mx.array | None = None        # not ``keys``: accounting reads those as growing
        self.ring_values: mx.array | None = None
        self.offset = 0
        self.drop_spare()

    # -- decode ----------------------------------------------------------------------------------------------------
    def write(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Decode rows [1, Hk, n, D] at offset ..; returns the buffers as they were, which hold every earlier key."""

        rows = int(keys.shape[2])
        if rows > self.ring - self.window + 1:
            raise ValueError(f"a decode write of {rows} rows would overwrite keys the window still reads")
        self._allocate(keys, values)
        held, first = (self.ring_keys, self.ring_values), self.offset
        self.recent = (*self.recent, (first, keys, values))
        if len(self.spares) < self.SPARES:
            # a new buffer: the write copies the current one, which this step's attention still reads
            target_k, target_v, since = self.ring_keys, self.ring_values, first
        else:
            oldest = min(range(len(self.spares)), key=lambda i: self.spares[i][2])
            target_k, target_v, since = self.spares[oldest]
            self.spares = self.spares[:oldest] + self.spares[oldest + 1:]
        fill_k, fill_v = self._rows_since(since)
        written_k = self._put(target_k, fill_k, since)
        written_v = self._put(target_v, fill_v, since)
        del target_k, target_v
        self.spares = (*self.spares, (self.ring_keys, self.ring_values, first))
        self.ring_keys, self.ring_values, self.offset = written_k, written_v, first + rows
        self._prune()
        return held

    def trim(self, n: int) -> int:
        n = min(self.offset, int(n))
        self.offset -= n
        self.spares = tuple((k, v, min(held, self.offset)) for k, v, held in self.spares)
        kept = []
        for first, k, v in self.recent:
            count = min(int(k.shape[2]), self.offset - first)
            if count > 0:
                kept.append((first, k, v) if count == int(k.shape[2]) else
                            (first, k[..., :count, :], v[..., :count, :]))
        self.recent = tuple(kept)
        return n

    def drop_spare(self) -> None:
        """Forget the older buffers (a retained or stored cache keeps one pair)."""

        self.spares, self.recent = (), ()

    def _rows_since(self, since: int) -> tuple[mx.array, mx.array]:
        """The recent rows at positions since .. offset + the newest write, in order."""

        parts_k, parts_v = [], []
        for first, k, v in self.recent:
            skip = max(0, since - first)
            if skip < int(k.shape[2]):
                parts_k.append(k[..., skip:, :] if skip else k)
                parts_v.append(v[..., skip:, :] if skip else v)
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    def _prune(self) -> None:
        """Drop recent rows every older buffer already holds."""

        oldest = min((held for _, _, held in self.spares), default=self.offset)
        self.recent = tuple(r for r in self.recent if r[0] + int(r[1].shape[2]) > oldest)

    # -- prompts (mlx_lm's attention calls these) ----------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """A prompt chunk's keys and values after up to window - 1 earlier ones, in order (as mlx_lm's ring does)."""

        prev = self.offset
        earlier = min(prev, self.window - 1)
        if earlier:
            fetched_k = mx.concatenate([self._ordered(self.ring_keys, prev - earlier, prev), keys], axis=2)
            fetched_v = mx.concatenate([self._ordered(self.ring_values, prev - earlier, prev), values], axis=2)
        else:
            fetched_k, fetched_v = keys, values
        self.drop_spare()
        self._allocate(keys, values)
        self.ring_keys = self._put(self.ring_keys, keys, prev)
        self.ring_values = self._put(self.ring_values, values, prev)
        self.offset = prev + int(keys.shape[2])
        return fetched_k, fetched_v

    def make_mask(self, N: int, window_size: int | None = None, return_array: bool = False) -> Any:
        """The mask for ``update_and_fetch``'s keys: causal within the window."""

        window = window_size or self.window
        earlier = min(self.window - 1, self.offset)
        if N > 1 and (earlier + N > window or return_array):
            return create_causal_mask(N, earlier, window_size=window)
        return "causal" if N > 1 else None

    # -- storage ---------------------------------------------------------------------------------------------------
    def _allocate(self, keys: mx.array, values: mx.array) -> None:
        if self.ring_keys is None:
            batch, heads, _, dims = keys.shape
            self.ring_keys = mx.zeros((batch, heads, self.ring, dims), dtype=keys.dtype)
            self.ring_values = mx.zeros((batch, heads, self.ring, int(values.shape[3])), dtype=values.dtype)

    def _put(self, buffer: mx.array, rows: mx.array, start: int) -> mx.array:
        """``rows`` [1, Hk, n, D] at positions start ..: slots start % ring on, wrapping (only the last ring rows)."""

        count = int(rows.shape[2])
        if count > self.ring:
            rows, start, count = rows[..., count - self.ring:, :], start + count - self.ring, self.ring
        slot = start % self.ring
        first = min(count, self.ring - slot)
        buffer = mx.slice_update(buffer, rows[..., :first, :] if first < count else rows, _slot(slot), axes=(2,))
        if first < count:
            buffer = mx.slice_update(buffer, rows[..., first:, :], _slot(0), axes=(2,))
        return buffer

    def _ordered(self, buffer: mx.array, begin: int, end: int) -> mx.array:
        """Positions [begin, end) of the ring in order (they must still be held)."""

        slot = begin % self.ring
        count = end - begin
        if slot + count <= self.ring:
            return buffer[..., slot:slot + count, :]
        return mx.concatenate([buffer[..., slot:, :], buffer[..., :slot + count - self.ring, :]], axis=2)

    @property
    def state(self) -> tuple[mx.array, ...]:
        return () if self.ring_keys is None else (self.ring_keys, self.ring_values)

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        self.drop_spare()
        self.ring_keys, self.ring_values = v

    @property
    def nbytes(self) -> int:
        arrays = [self.ring_keys, self.ring_values, *(a for k, v, _ in self.spares for a in (k, v))]
        return sum(a.nbytes for a in arrays if a is not None)

    def size(self) -> int:
        return min(self.offset, self.window)

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return self.ring_keys is None


def make_cache(text_model: Any) -> list[Any]:
    """One cache a layer: a ring for each sliding-window layer, a growing buffer for each full-attention one."""

    window = int(text_model.args.sliding_window)
    return [RingKVCache(window) if layer.self_attn.is_sliding else LinearKVCache()
            for layer in text_model.model.layers]


def adopt(cache: list[Any]) -> list[Any]:
    """A stored or copied cache in these classes (mlx_lm's own ``KVCache`` for full layers is taken over)."""

    from mlx_lm.models.cache import KVCache

    for i, item in enumerate(cache):
        if type(item) is KVCache:
            adopted = LinearKVCache()
            adopted.keys, adopted.values, adopted.offset = item.keys, item.values, item.offset
            cache[i] = adopted
    return cache


__all__ = ["LinearKVCache", "RingKVCache", "WINDOW_ROWS", "adopt", "make_cache"]
