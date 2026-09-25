"""A KV cache for pipelined decoding: decode writes alternate between two buffers.

MLX writes rows into a cache buffer in place only when nothing else holds the
buffer. A pipelined decode queues step s+1 while step s still runs, and step s
reads the buffer (its command buffer holds it until the GPU finishes), so step
s+1's write copied the whole cache first: 6 attention layers x (K, V) x 30 MB
at 60k keys, ~1 ms a token on an M5 Max (2026-09-25). Here each decode write
goes to the other buffer, last read by step s-1, which has finished: the rows
that buffer lacks (step s's) are written with the new ones, and nothing is
copied. The buffers hold the same values a ``KVCache`` holds, so attention
reads the same bits.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class AlternatingKVCache(KVCache):
    """``KVCache`` whose writes of up to ``alternate_rows`` rows go to a spare buffer.

    ``keys``/``values`` hold positions [0, offset) as in ``KVCache``. The spare holds
    [0, spare_len) and ``recent_keys``/``recent_values`` the rows [spare_len, offset) it
    lacks. Longer writes (prompt chunks) go to the current buffer and drop the spare.
    """

    alternate_rows = 16
    grow = 2048                      # spare capacity added at a time (a copy of the buffer each time)
    spare_keys: mx.array | None = None
    spare_values: mx.array | None = None
    spare_len = 0
    recent_keys: mx.array | None = None
    recent_values: mx.array | None = None
    # left out of prefix snapshots: the first decode write rebuilds them
    transient = ("spare_keys", "spare_values", "spare_len", "recent_keys", "recent_values")

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        rows = keys.shape[2]
        if self.keys is None or rows > self.alternate_rows:
            self.drop_spare()
            return super().update_and_fetch(keys, values)
        prev = self.offset
        end = prev + rows
        if self.spare_keys is None:
            # the first decode write: the current buffer stays as the spare, so this one write copies it
            target_k, target_v, start, fill_k, fill_v = self.keys, self.values, prev, keys, values
        else:
            target_k, target_v, start = self.spare_keys, self.spare_values, self.spare_len
            if self.recent_keys is None:
                fill_k, fill_v = keys, values
            else:
                fill_k = mx.concatenate([self.recent_keys, keys], axis=2)
                fill_v = mx.concatenate([self.recent_values, values], axis=2)
            self.spare_keys = self.spare_values = None      # the write's only holder: done in place
        if target_k.shape[2] < end:
            target_k, target_v = self._grown(target_k, start, end), self._grown(target_v, start, end)
        at = mx.array([start], dtype=mx.int32)
        written_k = mx.slice_update(target_k, fill_k, at, axes=(2,))
        written_v = mx.slice_update(target_v, fill_v, at, axes=(2,))
        del target_k, target_v
        self.spare_keys, self.spare_values, self.spare_len = self.keys, self.values, prev
        self.recent_keys, self.recent_values = keys, values
        self.keys, self.values, self.offset = written_k, written_v, end
        return written_k[..., :end, :], written_v[..., :end, :]

    def _grown(self, buffer: mx.array, valid: int, end: int) -> mx.array:
        batch, heads, _, dims = buffer.shape
        capacity = -(-end // self.grow) * self.grow
        return mx.concatenate([buffer[..., :valid, :], mx.zeros((batch, heads, capacity - valid, dims),
                                                                 dtype=buffer.dtype)], axis=2)

    def trim(self, n: int) -> int:
        n = super().trim(n)
        if self.recent_keys is not None and self.recent_values is not None:
            kept = self.offset - self.spare_len
            if kept <= 0:
                self.spare_len = self.offset
                self.recent_keys = self.recent_values = None
            else:
                self.recent_keys = self.recent_keys[..., :kept, :]
                self.recent_values = self.recent_values[..., :kept, :]
        elif self.spare_len > self.offset:
            self.spare_len = self.offset
        return n

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        return super().state

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        self.drop_spare()
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def drop_spare(self) -> None:
        """Forget the spare (a retained or stored cache keeps one buffer)."""

        self.spare_keys = self.spare_values = self.recent_keys = self.recent_values = None
        self.spare_len = 0

    @property
    def nbytes(self) -> int:
        spare = sum(a.nbytes for a in (self.spare_keys, self.spare_values) if a is not None)
        return super().nbytes + spare


def drop_spares(cache: list) -> list:
    """``cache`` with every alternating layer down to one buffer (for retained copies)."""

    for item in cache:
        if isinstance(item, AlternatingKVCache):
            item.drop_spare()
    return cache
