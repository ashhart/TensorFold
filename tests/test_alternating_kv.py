"""AlternatingKVCache holds what mlx_lm's KVCache holds, whatever the writes, trims and copies."""

import random

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from tensorfold.engine.alternating_kv import AlternatingKVCache, drop_spares
from tensorfold.engine.lane_engine import LaneEngine


def _same(a, b):
    return all(bool(mx.array_equal(x, y).item()) for x, y in zip(a, b))


def test_matches_kv_cache_under_random_writes_trims_and_copies():
    rng = random.Random(3)
    for _ in range(12):
        ref, alt = KVCache(), AlternatingKVCache()
        copies = []
        for _ in range(50):
            rows = rng.choice([20, 100, 300]) if rng.random() < 0.1 else rng.randint(1, 9)
            k = mx.random.normal((1, 2, rows, 8)).astype(mx.bfloat16)
            v = mx.random.normal((1, 2, rows, 8)).astype(mx.bfloat16)
            assert _same(ref.update_and_fetch(k, v), alt.update_and_fetch(k, v))
            if rows > 1 and rng.random() < 0.4:
                drop = rng.randint(1, rows - 1)
                assert ref.trim(drop) == alt.trim(drop)
            if rng.random() < 0.1:
                copies.append((LaneEngine.copy_single_cache([ref])[0], LaneEngine.copy_single_cache([alt])[0]))
            assert ref.offset == alt.offset
            assert _same(ref.state, alt.state)
        for r, a in copies:
            assert _same(r.state, a.state)


def test_decode_writes_alternate_and_spares_drop():
    alt = AlternatingKVCache()
    alt.update_and_fetch(mx.zeros((1, 2, 40, 8)), mx.zeros((1, 2, 40, 8)))
    assert alt.spare_keys is None
    for step in range(3):
        alt.update_and_fetch(mx.ones((1, 2, 1, 8)) * step, mx.ones((1, 2, 1, 8)))
    assert alt.spare_keys is not None and alt.spare_len == alt.offset - 1
    assert alt.nbytes > KVCache.nbytes.fget(alt)
    drop_spares([alt])
    assert alt.spare_keys is None and alt.recent_keys is None
    keys, _ = alt.state
    assert keys.shape[2] == 43 and float(keys[0, 0, 42, 0].item()) == 2.0
