"""Threaded n-gram gathers: callers at once keep their own rows, a failed copy raises, a dropped table's threads end."""

from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.host_table import HostTable, ReadAhead
from tests.test_ple_ssd import _checkpoint, _same

ROWS = 640                              # past 2 GATHER_SPLIT rows: copied on threads


@pytest.fixture(autouse=True)
def small_split(monkeypatch):
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)      # the test shards are small: split every 8 rows


def _want(rows, ids):
    return tuple(part[np.asarray(ids, dtype=np.int64).reshape(-1)] for part in rows)


def test_gathers_at_once_share_the_threads_but_not_their_rows(tmp_path):
    files, rows = _checkpoint(tmp_path)
    host = HostTable(files)
    rng = np.random.default_rng(11)
    asks = [rng.integers(0, host.rows, ROWS) for _ in range(8)]
    start = threading.Barrier(len(asks))

    def ask(ids):
        start.wait(timeout=30)
        return host.gather(ids)

    with ThreadPoolExecutor(len(asks)) as callers:
        got = list(callers.map(ask, asks))
    for ids, rows_got in zip(asks, got, strict=True):
        _same(rows_got, _want(rows, ids))


def test_a_read_ahead_and_a_direct_gather_overlap_without_mixing(tmp_path):
    files, rows = _checkpoint(tmp_path)
    ahead = ReadAhead(HostTable(files))
    rng = np.random.default_rng(12)
    first, second = rng.integers(0, ahead.rows, ROWS), rng.integers(0, ahead.rows, ROWS)
    ahead.read_ahead(first)                 # read on its thread while the next gather runs here
    _same(ahead.gather(second), _want(rows, second))
    _same(ahead.gather(first), _want(rows, first))


class _Broken:
    def __getitem__(self, _):
        raise OSError("read failed")


def test_a_failed_copy_on_a_thread_raises_from_the_gather_and_the_next_gather_works(tmp_path):
    files, rows = _checkpoint(tmp_path)
    host = HostTable(files)
    ids = np.arange(ROWS) % host.rows
    kept, host.files = host.files, [_Broken() for _ in host.files]
    with pytest.raises(OSError, match="read failed"):
        host.gather(ids)
    host.files = kept
    _same(host.gather(ids), _want(rows, ids))


def test_a_dropped_tables_gather_threads_end(tmp_path):
    files, _ = _checkpoint(tmp_path)
    host = HostTable(files)
    before = set(threading.enumerate())
    host.gather(np.arange(ROWS) % host.rows)
    started = [t for t in threading.enumerate() if t not in before]
    assert started and all(t.name.startswith("ngram-gather") for t in started)
    del host
    gc.collect()
    for t in started:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in started)
