"""Gemma 4's sliding-window ring (CPU): reads survive writes, rollbacks and turns; a write returns the old buffers."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tensorfold.families.gemma4.cache import RingKVCache  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def rows(first: int, count: int) -> mx.array:
    """Key rows [1, 1, count, 2] whose values name their position."""

    values = np.arange(first, first + count, dtype=np.float32)
    return mx.array(np.stack([values, -values], axis=-1)[None, None])


def held_positions(cache: RingKVCache, buffers: tuple[mx.array, mx.array], upto: int) -> list[float]:
    keys = np.array(buffers[0])[0, 0]
    return [float(keys[p % cache.ring, 0]) for p in range(max(0, upto - cache.window), upto)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_writes_rollbacks_and_turns_keep_every_windowed_key(seed):
    rng = np.random.default_rng(seed)
    cache = RingKVCache(window=8, ring=12)                 # windows of up to 5 rows
    keys, values = rows(0, 20), rows(0, 20)
    mx.eval(cache.update_and_fetch(keys, values))        # a prompt past the ring's end
    for _ in range(60):
        width = int(rng.integers(1, 6))
        start = cache.offset
        before = (cache.ring_keys, cache.ring_values)
        held = cache.write(rows(start, width), rows(start, width))
        assert held[0] is before[0] and held[1] is before[1]
        assert held_positions(cache, held, start) == [float(p) for p in range(max(0, start - 8), start)]
        keep = int(rng.integers(1, width + 1))
        cache.trim(width - keep)
        assert cache.offset == start + keep
        current = (cache.ring_keys, cache.ring_values)
        assert held_positions(cache, current, cache.offset) == [float(p) for p in range(cache.offset - 8,
                                                                                      cache.offset)]
        assert len(cache.spares) <= RingKVCache.SPARES
        assert all(held <= cache.offset for _, _, held in cache.spares)


def test_a_prompt_chunk_drops_the_older_buffers():
    cache = RingKVCache(window=8, ring=12)
    mx.eval(cache.update_and_fetch(rows(0, 10), rows(0, 10)))
    for i in range(4):
        cache.write(rows(10 + i, 1), rows(10 + i, 1))
    assert cache.spares
    fetched, _ = cache.update_and_fetch(rows(14, 3), rows(14, 3))
    assert not cache.spares and not cache.recent
    assert np.array(fetched)[0, 0, :, 0].tolist() == [float(p) for p in range(7, 17)]
