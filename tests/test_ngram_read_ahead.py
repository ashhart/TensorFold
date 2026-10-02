"""Flash Next's n-gram rows read ahead of a prompt chunk: a lookup takes the bytes a thread already read for its ids."""

import numpy as np

from tensorfold.families.qwen4_exp.host_table import ReadAhead


class Table:
    rows = 100

    def __init__(self):
        self.calls = []

    def gather(self, ids):
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        self.calls.append(flat.copy())
        return flat.astype(np.uint32)[:, None] * 3, flat.astype(np.uint16)[:, None], flat.astype(np.uint16)[:, None] + 1

    def prefetch(self):
        return 1.5


def test_a_lookup_takes_the_rows_read_ahead_for_its_ids():
    table = Table()
    ahead = ReadAhead(table)
    ids = np.array([[[3, 7], [9, 1]]])
    ahead.read_ahead(ids)
    got = ahead.gather(ids.reshape(-1, 2))                   # the lookup's shape of the same ids
    assert all(np.array_equal(a, b) for a, b in zip(got, Table().gather(ids)))
    assert len(table.calls) == 1
    ahead.gather(ids)                                        # nothing read ahead for it any more: read now
    assert len(table.calls) == 2
    assert ahead.rows == 100 and ahead.prefetch() == 1.5      # the table's other members


def test_reads_ahead_keep_the_latest_two_and_other_ids_read_now():
    table = Table()
    ahead = ReadAhead(table)
    for i in range(3):
        ahead.read_ahead(np.array([i]))
    assert len(ahead._ahead) == 2
    for i in (1, 2, 5, 0):
        assert int(ahead.gather(np.array([i]))[0][0, 0]) == 3 * i
    ahead._pool.shutdown(wait=True)
    assert sorted(int(c[0]) for c in table.calls) == [0, 0, 1, 2, 5]


def test_unlocked_table_gets_random_access(tmp_path, monkeypatch):
    """jschmied's #252 review: an unlocked EXL3 n-gram table's maps are advised MADV_RANDOM (one page a fault)."""
    import mmap

    import numpy as np

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    f = tmp_path / "t.bin"
    f.write_bytes(bytes(3 * mmap.PAGESIZE))
    calls = []

    class Libc:
        def madvise(self, addr, length, advice):
            calls.append((length, advice))
            return 0

    monkeypatch.setattr(exl3_pack, "_libc", lambda: Libc())
    t = object.__new__(exl3_pack.NgramTable)
    t.maps = [np.memmap(f, dtype=np.uint8, mode="r")]
    assert t.random_access() == 1
    assert calls == [(3 * mmap.PAGESIZE, mmap.MADV_RANDOM)]
