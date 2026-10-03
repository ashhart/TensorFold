"""Table rows by id: merged extents read ahead, rows kept, key order, and O_DIRECT reads equal to the file."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from tensorfold.cuda.tables import NvmeRows, Shard

ROW = 24


def table(tmp_path, rows=(300, 500), head=4096 + 13):
    """Shards of one file after a header; row r's bytes are a function of r."""

    path = tmp_path / "rows.bin"
    shards, at = [], head
    for n in rows:
        shards.append(Shard(str(path), at, n, ROW))
        at += n * ROW
    total = sum(rows)
    body = (np.arange(total, dtype=np.int64)[:, None] * 7 + np.arange(ROW)[None, :]) % 251
    path.write_bytes(bytes(head) + body.astype(np.uint8).tobytes())
    return shards, body.astype(np.uint8)


class Counting:
    """A reader of plain files that counts its reads and can hold them until released."""

    def __init__(self):
        self.reads, self.gate = [], threading.Event()
        self.gate.set()

    def read(self, path, offset, n):
        self.gate.wait(10)
        self.reads.append((offset, n))
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(n), dtype=np.uint8)


def test_rows_come_back_in_key_order_across_shards(tmp_path):
    shards, want = table(tmp_path)
    r = NvmeRows(shards, reader=Counting(), threads=4)
    keys = np.array([799, 0, 300, 299, 5, 5, 450])
    assert np.array_equal(r.wait(r.issue(keys)), want[keys])
    with pytest.raises(IndexError):
        r.issue([800])
    r.close()


def test_neighbouring_rows_share_one_read_per_shard(tmp_path):
    shards, want = table(tmp_path)
    reader = Counting()
    r = NvmeRows(shards, reader=reader, gap=2 * ROW)
    keys = np.array([10, 11, 13, 17, 299, 300, 302])  # 17 is past the gap; 299 and 300 sit in two shards
    got = r.wait(r.issue(keys))
    a, b = shards[0].offset, shards[1].offset
    assert np.array_equal(got, want[keys])
    assert sorted(reader.reads) == sorted(
        [(a + 10 * ROW, 4 * ROW), (a + 17 * ROW, ROW), (a + 299 * ROW, ROW), (b, 3 * ROW)]
    )


def test_kept_and_in_flight_rows_are_not_read_again(tmp_path):
    shards, want = table(tmp_path)
    reader = Counting()
    r = NvmeRows(shards, reader=reader, cache_rows=4, gap=0)
    reader.gate.clear()
    t1 = r.issue([1, 2])
    t2 = r.issue([2, 3])  # 2 is in flight already
    reader.gate.set()
    assert np.array_equal(r.wait(t2), want[[2, 3]]) and np.array_equal(r.wait(t1), want[[1, 2]])
    assert len(reader.reads) == 2 and r.hits == 2  # row 1 arrived with row 2's read
    r.wait(r.issue([1, 2, 3]))
    assert len(reader.reads) == 2
    r.wait(r.issue([50, 51, 52, 53, 54]))
    assert len(r.kept) == 4 and 1 not in r.kept
    assert np.array_equal(r.wait(r.issue([1])), want[[1]]) and len(reader.reads) == 4  # 50-54 in one read


def test_a_row_nobody_issued_is_read_when_asked(tmp_path):
    shards, want = table(tmp_path)
    r = NvmeRows(shards, reader=Counting())
    from tensorfold.cuda.tables import Ticket

    assert np.array_equal(r.wait(Ticket(np.array([42, 7]))), want[[42, 7]]) and r.misses == 2


def test_direct_reads_through_the_checkpoint_reader(tmp_path):
    pytest.importorskip("torch")
    shards, want = table(tmp_path, rows=(1000, 3000))
    r = NvmeRows(shards, threads=8)  # direct_read.Reader: O_DIRECT where the disk allows it
    keys = np.random.default_rng(0).integers(0, 4000, size=500)
    assert np.array_equal(r.wait(r.issue(keys)), want[keys])
    r.close()
