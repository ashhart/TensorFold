"""A layer's state, all indexed by position, so keeping a window's first rows only moves ``offset``."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.config import DECODE_ROWS


def ring_size(span: int) -> int:
    """Ring rows for a span of past positions: a decode window's rejected rows never overwrite a kept one."""

    return span + DECODE_ROWS


def ring_write(ring: mx.array, rows: mx.array, start: int) -> mx.array:
    """Rows of positions start .. into their slots ``p % size`` (the last ``size`` of them when more)."""

    size, n = int(ring.shape[0]), int(rows.shape[0])
    if n > size:
        rows, start, n = rows[n - size:], start + n - size, size
    a = start % size
    if a + n <= size:
        ring[a:a + n] = rows
    else:
        ring[a:] = rows[:size - a]
        ring[:n - (size - a)] = rows[size - a:]
    return ring


def ring_rows(ring: mx.array, lo: int, hi: int) -> mx.array:
    size = int(ring.shape[0])
    if hi - lo <= 0:
        return ring[:0]
    a, b = lo % size, (hi - 1) % size + 1
    if a < b and hi - lo == b - a:
        return ring[a:b]
    return mx.concatenate([ring[a:], ring[:b]])


class LayerCache:
    """Window ring, the pools a KV source owns (by block), compressor projections and Engram's tokens (by position)."""

    step = 256

    def __init__(self, ratio: int = 0, window: int = 128, source: bool = False, history: int = 0) -> None:
        self.ratio = int(ratio)            # the compressed KV's ratio this layer owns (0: it owns none)
        self.window = int(window)
        self.source = bool(source)
        self.history = int(history)        # token ids Engram reads before a call (layer 0 only)
        self.offset = 0
        self.keys: mx.array | None = None  # [ring_size(window), head_dim] fp32: the FP8 window keys
        self.pool: mx.array | None = None  # [capacity, head_dim] bf16: compressed keys (FP4), row b pools block b
        self.ipool: mx.array | None = None  # [capacity, index_dim] bf16: index keys (FP4)
        self.proj: mx.array | None = None  # [ring, 2 * head_dim] fp32: ratio-2 compressor values | gates by position
        self.tokens: mx.array | None = None  # [ring_size(history)] int32: token ids by position

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.keys, self.pool, self.ipool, self.proj, self.tokens) if a is not None]

    @property
    def pool_rows(self) -> int:
        return self.offset // self.ratio if self.source and self.ratio else 0

    def trim(self, count: int) -> None:
        self.offset -= int(count)

    # -- window keys ------------------------------------------------------------------------
    def write_keys(self, keys: mx.array, start: int) -> None:
        if self.keys is None:
            self.keys = mx.zeros((ring_size(self.window), int(keys.shape[-1])), dtype=mx.float32)
        self.keys = ring_write(self.keys, keys.astype(mx.float32), start)

    def key_rows(self, lo: int, hi: int) -> mx.array:
        return ring_rows(self.keys, lo, hi)

    # -- compressor projections (ratio 2) ----------------------------------------------------
    def proj_ring(self) -> int:
        return ring_size(self.ratio)

    def write_proj(self, values: mx.array, start: int) -> None:
        if self.proj is None:
            self.proj = mx.zeros((self.proj_ring(), int(values.shape[-1])), dtype=mx.float32)
        self.proj = ring_write(self.proj, values, start)

    def proj_rows(self, lo: int, hi: int) -> mx.array:
        return ring_rows(self.proj, lo, hi)

    # -- pools --------------------------------------------------------------------------------
    def write_pool(self, rows: mx.array, first: int, name: str) -> None:
        """Rows for blocks first .. of ``pool`` or ``ipool`` (pools grow in steps; rows past pool_rows are stale)."""

        pool = getattr(self, name)
        end = first + int(rows.shape[0])
        cap = 0 if pool is None else int(pool.shape[0])
        if end > cap:
            grown = mx.zeros((-(-end // self.step) * self.step, int(rows.shape[-1])), dtype=mx.bfloat16)
            pool = grown if pool is None else mx.concatenate([pool, grown[cap:]])
        pool[first:end] = rows.astype(mx.bfloat16)
        setattr(self, name, pool)

    # -- Engram's token ids ---------------------------------------------------------------------
    def write_tokens(self, ids: mx.array, start: int) -> None:
        if not self.history:
            return
        if self.tokens is None:
            self.tokens = mx.zeros((ring_size(self.history),), dtype=mx.int32)
        self.tokens = ring_write(self.tokens, ids.astype(mx.int32), start)

    def token_rows(self, lo: int, hi: int) -> mx.array:
        return ring_rows(self.tokens, lo, hi)

    def memory_growth(self) -> tuple[int, int]:
        """(fixed bytes, bytes a token): the rings stay, the pools grow a row a block."""

        fixed = sum(int(a.nbytes) for a in (self.keys, self.proj, self.tokens) if a is not None)
        if not (self.source and self.ratio):
            return fixed, 0
        dims = int(self.pool.shape[1]) if self.pool is not None else 512
        idims = int(self.ipool.shape[1]) if self.ipool is not None else 128
        return fixed, 2 * (dims + idims) // self.ratio


def make_caches(cfg: Any) -> list[LayerCache]:
    depth = max(0, int(cfg.engram_max_ngram_size) - 1) if cfg.engram_layer_ids else 0
    out = []
    for i in range(cfg.num_hidden_layers):
        source = i in cfg.kv_source_layer_ids and bool(cfg.ratio(i))
        out.append(LayerCache(cfg.ratio(i) if source else 0, cfg.sliding_window, source, depth if i == 0 else 0))
    return out
