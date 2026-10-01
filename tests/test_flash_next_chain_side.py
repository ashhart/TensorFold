"""Chained MTP drafts read their own rows from a side buffer and keep the bits of rows written into the cache."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.families.qwen4_exp.mtp_cache import MTPCache  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import attention  # noqa: E402

HEADS, KVH, DIMS = 24, 2, 256


def _bf16(rng, *shape):
    return mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)


@pytest.mark.parametrize("length,split,rows", [(300, 297, 1), (300, 296, 3), (9000, 8998, 2), (9000, 8997, 4)])
def test_split_attention_equals_one_buffer(length, split, rows):
    rng = np.random.default_rng(length + split + rows)
    keys, values = _bf16(rng, 1, KVH, length + 64, DIMS), _bf16(rng, 1, KVH, length + 64, DIMS)
    q = _bf16(rng, rows, HEADS, DIMS)
    ends = [length - rows + 1 + r for r in range(rows)]
    top = 512
    complete = [e // 4 for e in ends]
    sparse = [c > top for c in complete]
    counts = [4 * top + e - 4 * c if sp else e for e, c, sp in zip(ends, complete, sparse)]
    ids = None
    if any(sparse):
        pooled = _bf16(rng, max(complete), 128)
        ids = attention.index_select(_bf16(rng, rows, 4, 128), pooled, complete, ends, top=top)
    gate = _bf16(rng, rows, HEADS * 2 * DIMS + 1024)
    want = attention.attention_rows(q, keys, values, counts, ids, sparse, 0.0625, gate=gate)
    # the buffer holds rows before ``split`` only; the side holds the rest
    stale = mx.concatenate([keys[:, :, :split], _bf16(rng, 1, KVH, length + 64 - split, DIMS)], axis=2)
    stale_v = mx.concatenate([values[:, :, :split], _bf16(rng, 1, KVH, length + 64 - split, DIMS)], axis=2)
    got = attention.attention_rows_split(q, stale, stale_v, keys[:, :, split:length], values[:, :, split:length],
                                         split, counts, ids, sparse, 0.0625, gate=gate)
    assert bool(mx.array_equal(got, want).item())


def test_relative_pool_equals_the_cache_pool():
    rng = np.random.default_rng(5)
    raw = _bf16(rng, 4096, 128)
    norm = mx.array(rng.normal(size=(128,)).astype(np.float32)) + 1.0
    eps = mx.array([1e-6], dtype=mx.float32)
    start, stop = 700, 703
    want = attention.index_pool(raw, start, stop, norm, eps, rotary_dim=32, base=1e7)
    got = attention.index_pool(raw[4 * start:4 * stop + 2], start, stop, norm, eps, rotary_dim=32, base=1e7,
                               relative=True)
    assert attention._IDX_POOL_REL != attention._IDX_POOL
    assert bool(mx.array_equal(got, want).item())


def test_chained_rows_live_in_the_side_and_trim_away():
    rng = np.random.default_rng(9)
    cache = MTPCache()
    cache.update(_bf16(rng, 1, KVH, 10, DIMS), _bf16(rng, 1, KVH, 10, DIMS), _bf16(rng, 1, 10, 128))
    keys_before = cache.keys
    cache.chaining = True
    rows = [(_bf16(rng, 1, KVH, 1, DIMS), _bf16(rng, 1, KVH, 1, DIMS), _bf16(rng, 1, 1, 128)) for _ in range(2)]
    for k, v, i in rows:
        assert cache.update(k, v, i) == (None, None, None)
    cache.chaining = False
    assert cache.offset == 12 and cache.side_base == 10 and cache.keys is keys_before
    assert bool(mx.array_equal(cache.side[0], mx.concatenate([rows[0][0], rows[1][0]], axis=2)).item())
    index = cache.side_index_rows(8)
    want = mx.concatenate([cache.index_keys[:, 8:10], rows[0][2], rows[1][2]], axis=1)
    assert bool(mx.array_equal(index, want).item())
    cache.trim(1)
    assert cache.offset == 11 and int(cache.side[0].shape[2]) == 1
    cache.trim(1)
    assert cache.offset == 10 and cache.side is None
    cache.update(*rows[0])                        # not chaining: written into the buffer at offset 10
    assert cache.offset == 11 and cache.side is None
    assert bool(mx.array_equal(cache.keys[:, :, 10:11], rows[0][0]).item())


def test_the_sparse_decode_kernels_warm_at_load():
    norm = mx.ones((128,), dtype=mx.float32)
    attention.warm_decode(heads=HEADS, kv_heads=KVH, dims=DIMS, index_heads=4, index_dims=128, top=512, scale=0.0625,
                          width=(2 * HEADS + 2 * KVH) * DIMS + 5 * 128, norm=norm, eps=mx.array([1e-6]),
                          rotary_dim=32, base=1e7)
