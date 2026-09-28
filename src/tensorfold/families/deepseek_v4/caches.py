"""An attention layer's state, all indexed by position, so keeping a window's first rows only moves ``offset``."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.config import DECODE_ROWS

WINDOW = 128


def ring_size(span: int) -> int:
    """Ring rows for a span of past positions: a decode window's rejected rows never overwrite a kept one."""

    return span + DECODE_ROWS


class LayerCache:
    """Key ring, pool rows by block, compressor projections by position: a rollback to T only sets the offset."""

    step = 256

    def __init__(self, ratio: int = 0, window: int = WINDOW) -> None:
        self.ratio = int(ratio)
        self.window = int(window)
        self.offset = 0
        self.keys: mx.array | None = None       # [ring_size(window), head_dim] bf16
        self.pool: mx.array | None = None       # [capacity, head_dim] bf16, row w pools block w
        self.ipool: mx.array | None = None      # [capacity, index_dim] bf16, ratio 4 only
        self.proj: mx.array | None = None       # [ring, width] fp32: every compressor's values | gates by position

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.keys, self.pool, self.ipool, self.proj) if a is not None]

    @property
    def pool_rows(self) -> int:
        return self.offset // self.ratio if self.ratio else 0

    def proj_ring(self) -> int:
        """Projection ring rows: the partial block, the one before it at ratio 4, and a decode window."""

        return ring_size((2 if self.ratio == 4 else 1) * self.ratio)

    def trim(self, count: int) -> None:
        self.offset -= int(count)

    def write_keys(self, keys: mx.array, start: int) -> None:
        """Keys of positions start .. (only the last ring's worth is kept)."""

        size = ring_size(self.window)
        if self.keys is None:
            self.keys = mx.zeros((size, int(keys.shape[-1])), dtype=mx.bfloat16)
        self.keys = _ring_write(self.keys, keys.astype(mx.bfloat16), start)

    def window_keys(self, position: int) -> mx.array:
        """The keys a query at ``position`` reads: positions max(0, p - window + 1) .. p, in order."""

        lo = max(0, position - self.window + 1)
        return self.ring_rows(self.keys, lo, position + 1)

    @staticmethod
    def ring_rows(ring: mx.array, lo: int, hi: int) -> mx.array:
        size = int(ring.shape[0])
        if hi - lo <= 0:
            return ring[:0]
        a, b = lo % size, (hi - 1) % size + 1
        if a < b and hi - lo == b - a:
            return ring[a:b]
        return mx.concatenate([ring[a:], ring[:b]])

    def write_proj(self, values: mx.array, start: int) -> None:
        """The compressors' projections [R, width] of positions start .. into their ring."""

        if self.proj is None:
            self.proj = mx.zeros((self.proj_ring(), int(values.shape[-1])), dtype=mx.float32)
        self.proj = _ring_write(self.proj, values, start)

    def proj_rows(self, lo: int, hi: int) -> mx.array:
        return self.ring_rows(self.proj, lo, hi)

    def write_pool(self, rows: mx.array, first: int, index: bool = False) -> None:
        """Pool rows for blocks first .. (the pool grows in steps; rows past ``pool_rows`` are stale)."""

        name = "ipool" if index else "pool"
        pool = getattr(self, name)
        end = first + int(rows.shape[0])
        cap = 0 if pool is None else int(pool.shape[0])
        if end > cap:
            grown = mx.zeros((-(-end // self.step) * self.step, int(rows.shape[-1])), dtype=mx.bfloat16)
            pool = grown if pool is None else mx.concatenate([pool, grown[cap:]])
        pool[first:end] = rows.astype(mx.bfloat16)
        setattr(self, name, pool)

    def memory_growth(self) -> tuple[int, int]:
        """(fixed bytes, bytes a token): the rings stay, the pools grow a row a block."""

        fixed = sum(int(a.nbytes) for a in (self.keys, self.proj) if a is not None)
        if not self.ratio:
            return fixed, 0
        dims = int(self.pool.shape[1]) if self.pool is not None else 512
        idims = int(self.ipool.shape[1]) if self.ipool is not None else (128 if self.ratio == 4 else 0)
        return fixed, 2 * (dims + idims) // self.ratio


def _ring_write(ring: mx.array, rows: mx.array, start: int) -> mx.array:
    """Rows of positions start .. into their slots ``p % size`` (the last ``size`` of them when more)."""

    size = int(ring.shape[0])
    n = int(rows.shape[0])
    if n > size:
        rows, start, n = rows[n - size:], start + n - size, size
    a = start % size
    if a + n <= size:
        ring[a:a + n] = rows
    else:
        ring[a:] = rows[:size - a]
        ring[:n - (size - a)] = rows[size - a:]
    return ring


def make_caches(ratios: list[int]) -> list[Any]:
    return [LayerCache(r) for r in ratios]
