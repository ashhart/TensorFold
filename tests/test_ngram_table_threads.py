"""The BF16, FP8 and NVFP4 n-gram tables copy a prompt chunk's rows on threads, as HostTable does: the bytes of one
thread across shard boundaries, repeated and empty ids, callers at once, a failed copy raising, a dropped table's
threads ending."""

from __future__ import annotations

import gc
import json
import struct
import threading

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.host_table import BF16Table, FP8Table, NVFP4Table, read_header

SIZES = (37, 64, 29)          # rows a shard; the last two share a file


def _write(path, tensors: dict) -> None:
    header, data, at = {}, b"", 0
    for name, (dtype, arr) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [at, at + arr.nbytes]}
        data, at = data + arr.tobytes(), at + arr.nbytes
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def _tables(tmp_path):
    rng = np.random.default_rng(7)
    rows = {"bf16": [rng.integers(0, 2 ** 16, (r, 32), dtype=np.uint16) for r in SIZES],
            "fp8": [rng.integers(0, 0x7F, (r, 32), dtype=np.uint8) for r in SIZES],
            "nvfp4": [rng.integers(0, 256, (r, 16), dtype=np.uint8) for r in SIZES]}
    blocks = [rng.integers(0x20, 0x48, (r, 2), dtype=np.uint8) for r in SIZES]
    files = {"bf16": [], "fp8": [], "nvfp4": []}
    for kind, dtype in (("bf16", "BF16"), ("fp8", "F8_E4M3"), ("nvfp4", "U8")):
        for part, shards in (("a", (0,)), ("b", (1, 2))):
            path, tensors = tmp_path / f"{kind}-{part}.safetensors", {}
            for i in shards:
                tensors[f"t.shard_{i}.weight"] = (dtype, rows[kind][i])
                if kind == "nvfp4":
                    tensors[f"t.shard_{i}.weight_scale"] = ("F8_E4M3", blocks[i])
            _write(path, tensors)
            head = read_header(path)
            for i in shards:
                entry = head[f"t.shard_{i}.weight"]
                scale = head.get(f"t.shard_{i}.weight_scale")
                files[kind].append((path, entry, scale) if kind == "nvfp4" else (path, entry))
    tables = {"bf16": BF16Table(files["bf16"]), "fp8": FP8Table(files["fp8"], 0.05),
              "nvfp4": NVFP4Table(files["nvfp4"], 0.03)}
    return tables, np.concatenate(rows["bf16"])


def _ids(rows: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.integers(0, rows, 200), np.arange(rows)[::-1], np.array([3, 3, 1]), np.array([], dtype=np.int64),
            rng.integers(0, rows, (12, 16))]


@pytest.mark.parametrize("threads", [2, 16])
def test_threaded_gathers_give_one_threads_bytes(tmp_path, monkeypatch, threads):
    """Every table's threaded copy equals its single-threaded one; the bf16 rows are the stored rows."""

    tables, bf16_rows = _tables(tmp_path)
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)                # small shards: split every 8 rows
    monkeypatch.setattr(host_table, "GATHER_THREADS", 1)
    want = {kind: [t.gather(ids) for ids in _ids(t.rows)] for kind, t in tables.items()}
    monkeypatch.setattr(host_table, "GATHER_THREADS", threads)
    for kind, table in tables.items():
        calls = []
        real = table._pool.map
        monkeypatch.setattr(table._pool, "map", lambda fn, jobs, real=real: calls.append(1) or real(fn, jobs))
        for ids, ref in zip(_ids(table.rows), want[kind], strict=True):
            got = table.gather(ids)
            assert got.dtype == np.uint16 and got.shape == (np.asarray(ids).size, table.width)
            assert np.array_equal(got, ref), (kind, np.asarray(ids).size)
        assert len(calls) == 3, kind                                  # the three gathers of 16 rows or more
    for ids, got in zip(_ids(tables["bf16"].rows), want["bf16"], strict=True):
        assert np.array_equal(got, bf16_rows[np.asarray(ids, dtype=np.int64).reshape(-1)])


def test_callers_gathering_at_once_keep_their_own_rows(tmp_path, monkeypatch):
    """Four threads gather different rows from each table at once through its one pool: each gets its own rows."""

    tables, _ = _tables(tmp_path)
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)
    rng = np.random.default_rng(11)
    asks = [rng.integers(0, 130, 64) for _ in range(4)]
    monkeypatch.setattr(host_table, "GATHER_THREADS", 1)
    want = {kind: [t.gather(ids) for ids in asks] for kind, t in tables.items()}
    monkeypatch.setattr(host_table, "GATHER_THREADS", 16)
    wrong: list = []

    def caller(i: int) -> None:
        for _ in range(20):
            for kind, table in tables.items():
                if not np.array_equal(table.gather(asks[i]), want[kind][i]):
                    wrong.append((i, kind))

    workers = [threading.Thread(target=caller, args=(i,)) for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not wrong


def test_ids_outside_the_table_are_refused(tmp_path):
    tables, _ = _tables(tmp_path)
    for table in tables.values():
        for bad in ([table.rows], [-1]):
            with pytest.raises(ValueError):
                table.gather(np.array(bad))


class _Broken:
    def __getitem__(self, _):
        raise OSError("read failed")


@pytest.mark.parametrize("kind", ["bf16", "fp8", "nvfp4"])
def test_a_failed_copy_on_a_thread_raises_and_the_next_gather_works(tmp_path, monkeypatch, kind):
    tables, _ = _tables(tmp_path)
    table = tables[kind]
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)
    ids = np.arange(640) % table.rows
    want = table.gather(ids)
    kept, table.values = table.values, [_Broken() for _ in table.values]
    with pytest.raises(OSError, match="read failed"):
        table.gather(ids)
    table.values = kept
    assert np.array_equal(table.gather(ids), want)


@pytest.mark.parametrize("kind", ["bf16", "fp8", "nvfp4"])
def test_a_dropped_tables_gather_threads_end(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)
    table = _tables(tmp_path)[0][kind]
    before = set(threading.enumerate())
    table.gather(np.arange(640) % table.rows)
    started = [t for t in threading.enumerate() if t not in before]
    assert started and all(t.name.startswith("ngram-gather") for t in started)
    del table
    gc.collect()
    for t in started:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in started)
