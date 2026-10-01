"""cache_arrays and cache_contents read mlx-lm caches the same under 0.31 and 0.32."""

# 0.32 state adds offsets and nests a recurrent layer's arrays in a list.

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

from tensorfold.engine.alternating_kv import AlternatingKVCache  # noqa: E402
from tensorfold.engine.family_common import cache_arrays, cache_contents  # noqa: E402


def _layers():
    kv = KVCache()
    rows = mx.arange(2 * 5 * 4, dtype=mx.float32).reshape(1, 2, 5, 4)
    kv.update_and_fetch(rows, rows + 1)
    recurrent = ArraysCache(size=2)
    recurrent[0], recurrent[1] = mx.zeros((1, 3, 6)), mx.ones((1, 2, 4, 4))
    return kv, recurrent, rows


def test_cache_arrays_holds_every_layers_arrays():
    kv, recurrent, _ = _layers()
    held = cache_arrays([kv, recurrent, KVCache()])     # 0.31 gives the KV rows as views, 0.32 the buffers
    assert len(held) == 4 and all(hasattr(a, "shape") for a in held)
    assert {id(recurrent[0]), id(recurrent[1])} <= {id(a) for a in held}
    assert all(a.shape[2] >= kv.offset for a in held[:2])


def test_cache_contents_are_the_rows_up_to_the_offset():
    kv, recurrent, rows = _layers()
    keys, values = cache_contents(kv)
    assert keys.shape[2] == kv.offset == 5        # 0.32's state is the whole step-sized buffer
    assert mx.array_equal(keys, rows).item() and mx.array_equal(values, rows + 1).item()
    assert [a.shape for a in cache_contents(recurrent)] == [(1, 3, 6), (1, 2, 4, 4)]
    assert cache_contents(KVCache()) == []


def test_alternating_state_round_trips_through_its_setter():
    kv, _, rows = _layers()
    alt = AlternatingKVCache()
    alt.update_and_fetch(rows, rows + 1)
    alt.update_and_fetch(rows[..., :1, :], rows[..., :1, :])    # a decode write leaves a spare
    alt.state = kv.state
    assert alt.spare_keys is None and alt.offset == kv.offset
    assert all(mx.array_equal(a, b).item() for a, b in zip(cache_contents(alt), cache_contents(kv)))
