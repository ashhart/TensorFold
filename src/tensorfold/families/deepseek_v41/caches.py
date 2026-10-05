"""Attention state for DeepSeek-V4.1: per-layer window rings; pool groups live on their head layer's cache.

The pool group of a kv-source layer is shared by every compressed layer below it in its span: the head
writes pool rows / index keys / the partial block's projections, and publishes the latest per-row
selection (Full and Reindex layers publish; Reuse layers read it unchanged). Writes are pure-functional
(``ring`` / ``pool`` return new arrays) so a cache copy the engine made shares the old buffers safely.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.config import DECODE_ROWS

WINDOW = 128


def ring_size(span: int) -> int:
    """Ring rows for a span of past positions: a decode window's rejected rows never overwrite a kept one."""
    return span + DECODE_ROWS


class LayerCache:
    """One layer's state, all indexed by position, so keeping a window's first rows only moves ``offset``.

    Every layer keeps its own window key ring. A kv-source (group head) layer additionally owns its
    group's pool state: pooled KV rows, index keys, the partial compressor block's projections, and the
    latest per-row selection published by an index-source layer of the group.
    """

    step = 256

    def __init__(self, ratio: int = 0, window: int = WINDOW) -> None:
        self.ratio = int(ratio)
        self.window = int(window)
        self.offset = 0
        self.keys: mx.array | None = None       # [ring_size(window), head_dim] bf16
        self.pool: mx.array | None = None       # [capacity, head_dim] bf16, row w pools block w (heads only)
        self.ipool: mx.array | None = None      # [capacity, index_head_dim] bf16 (heads only)
        self.proj: mx.array | None = None       # [ring, 2 * width] fp32: the partial block's wkv | wgate (heads)
        self.topk_idxs: mx.array | None = None  # the latest published selection [R, topk] int32
        self.candidates: mx.array | None = None # the candidate block mask [n] bool of the last candidate pass

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.keys, self.pool, self.ipool, self.proj) if a is not None]

    @property
    def pool_rows(self) -> int:
        return self.offset // self.ratio if self.ratio else 0

    def trim(self, count: int) -> None:
        """A rollback to the kept prefix: offsets move; a stale published selection never outlives it."""
        self.offset -= int(count)
        if count:
            self.topk_idxs = None
            self.candidates = None

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
    def ring_rows(ring: mx.array | None, lo: int, hi: int) -> mx.array:
        if ring is None:
            return mx.zeros((0, 1), dtype=mx.bfloat16)
        size = int(ring.shape[0])
        if hi - lo <= 0:
            return mx.zeros((0, int(ring.shape[1])), dtype=ring.dtype)
        a, b = lo % size, (hi - 1) % size + 1
        if a < b and hi - lo == b - a:
            return ring[a:b]
        return mx.concatenate([ring[a:], ring[:b]])

    def write_proj(self, values: mx.array, start: int) -> None:
        """The partial compressor block's slots [R, 2 * dim] (values | gates) of positions start .. ."""
        if self.proj is None:
            self.proj = mx.zeros((ring_size(max(1, self.ratio)) + DECODE_ROWS, int(values.shape[-1])),
                                 dtype=mx.float32)
        self.proj = _ring_write(self.proj, values, start)

    def proj_rows(self, lo: int, hi: int) -> mx.array:
        """The stored slots [hi - lo, 2 * dim] split into (values, gates), each [hi - lo, dim]."""
        span = self.ring_rows(self.proj, lo, hi)
        half = int(span.shape[-1]) // 2
        return span[:, :half], span[:, half:]

    def write_pool(self, rows: mx.array, first: int, index: bool = False) -> None:
        """Pool rows for blocks first .. (the pool grows in steps; rows past ``pool_rows`` are stale)."""
        name = "ipool" if index else "pool"
        pool = getattr(self, name)
        end = first + int(rows.shape[0])
        cap = 0 if pool is None else int(pool.shape[0])
        if end > cap:
            grown = mx.zeros((-(-end // self.step) * self.step, int(rows.shape[-1])), dtype=mx.bfloat16)
            pool = grown if pool is None else mx.concatenate([pool, grown[cap:]])
        setattr(self, name, _slice_write(pool, rows.astype(mx.bfloat16), first))

    def publish(self, topk_idxs: mx.array | None, candidates: mx.array | None) -> None:
        """The group's latest per-row selection, read unchanged by every Reuse layer of the group."""
        if topk_idxs is not None:
            self.topk_idxs = topk_idxs
        if candidates is not None:
            self.candidates = candidates

    def memory_growth(self) -> tuple[int, int]:
        """(fixed bytes, bytes a token): the rings stay, the pools grow a row a block."""
        fixed = sum(int(a.nbytes) for a in (self.keys, self.proj) if a is not None)
        if not self.ratio:
            return fixed, 0
        dims = int(self.pool.shape[1]) if self.pool is not None else 512
        idims = int(self.ipool.shape[1]) if self.ipool is not None else 128
        return fixed, 2 * (dims + idims) // self.ratio


def head_of(cfg: Any, layer: int) -> int | None:
    """The pool group's head layer id for a compressed layer: the latest kv source at or below it."""
    if not cfg.ratio(layer):
        return None
    for head in reversed(cfg.kv_source_layer_ids):
        if head <= layer:
            return head
    return None


def make_caches(cfg: Any) -> list[Any]:
    """A LayerCache per layer (heads own their group's pool state), all sharing the window size."""
    window = int(cfg.sliding_window)
    return [LayerCache(cfg.ratio(i), window) for i in range(int(cfg.num_hidden_layers))]


def _slice_write(target: mx.array, rows: mx.array, at: int) -> mx.array:
    """Rows into ``target`` at ``at``, always a fresh array (a shared pool buffer is never written).

    A copy made by ``LaneEngine.copy_single_cache`` shares whole arrays between sibling streams, so a
    write must never mutate a buffer another cache holds: growth already built a fresh array, and here
    the write goes into a copy-on-write clone.
    """
    out = mx.array(target)         # copy-on-write: a fresh buffer even when the source is shared
    out[at:at + rows.shape[0]] = rows
    return out


def _ring_write(ring: mx.array, rows: mx.array, start: int) -> mx.array:
    """Rows of positions start .. into their slots ``p % size`` (the last ``size`` of them when more).

    Always returns a fresh array: a shared ring is never mutated (engine cache copies share buffers).
    """
    size = int(ring.shape[0])
    n = int(rows.shape[0])
    if n > size:
        rows, start, n = rows[n - size:], start + n - size, size
    out = mx.array(ring)          # copy-on-write: a fresh buffer even when the source is shared
    a = start % size
    if a + n <= size:
        out[a:a + n] = rows
    else:
        out[a:] = rows[:size - a]
        out[:n - (size - a)] = rows[size - a:]
    return out
