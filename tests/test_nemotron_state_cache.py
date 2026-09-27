"""Nemotron's Mamba cache holding a shared forward's row: every read, copy and store sees the row's states."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot  # noqa: E402
from tensorfold.families.nemotron_h.state_cache import RowStateCache  # noqa: E402


def _rows():
    conv = mx.arange(5 * 3 * 4, dtype=mx.float32).reshape(5, 3, 4)
    ssm = mx.arange(5 * 2 * 2 * 3, dtype=mx.float32).reshape(5, 2, 2, 3) * 0.5
    return conv, ssm


def test_a_pointed_row_reads_as_its_states():
    conv, ssm = _rows()
    cache = RowStateCache(2)
    cache.point(conv, ssm, 3)
    assert cache.ref is not None
    assert bool(mx.array_equal(cache[0], conv[3:4]).item()) and bool(mx.array_equal(cache[1], ssm[3:4]).item())
    assert cache.ref is None
    cache.point(conv, ssm, 1)
    cache[1] = ssm[4:5]                          # a write keeps the other slot's pointed state
    assert bool(mx.array_equal(cache[0], conv[1:2]).item()) and bool(mx.array_equal(cache[1], ssm[4:5]).item())


def test_copies_and_stored_snapshots_hold_the_row(tmp_path):
    conv, ssm = _rows()
    cache = RowStateCache(2)
    cache.point(conv, ssm, 2)
    copied = LaneEngine.copy_single_cache([cache])[0]
    assert copied.ref is None and cache.ref is None
    assert bool(mx.array_equal(copied[1], ssm[2:3]).item())
    cache.point(conv, ssm, 4)
    path = save_snapshot(tmp_path, "m", [1, 2, 3], [cache])
    tokens, loaded = load_snapshot(path, "m")
    assert tokens == [1, 2, 3] and isinstance(loaded[0], RowStateCache)
    assert bool(mx.array_equal(loaded[0][0], conv[4:5]).item())
    assert bool(mx.array_equal(loaded[0][1], ssm[4:5]).item())
