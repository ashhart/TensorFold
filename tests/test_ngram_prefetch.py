"""N-gram prefetch: every mapped array's file bytes read once in spans; an array that maps no file is faulted in."""

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table


def test_prefetch_reads_each_mapped_arrays_bytes_once_in_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(host_table, "PREFETCH_READ", 4096)
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"h" * 1000 + (np.arange(41_000) % 251).astype(np.uint8).tobytes())
    arrays = [np.memmap(path, dtype=np.uint8, mode="r", offset=1000, shape=(20_000,)),
              np.memmap(path, dtype=np.uint16, mode="r", offset=21_000, shape=(10_000,))]
    reads, real = [], open

    class Spy:
        def __init__(self, *args, **kwargs):
            self.f = real(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def seek(self, at):
            self.at = at
            return self.f.seek(at)

        def readinto(self, view):
            got = self.f.readinto(view)
            reads.append((self.at, got, bytes(view[:got])))
            return got

    monkeypatch.setattr(host_table, "open", Spy, raising=False)
    assert host_table._prefetch(arrays, workers=3) >= 0.0
    spans = sorted((at, n) for at, n, _ in reads)
    assert all(n <= 4096 for _, n in spans) and len(spans) == len({at for at, _ in spans})
    covered = [(at, at + n) for at, n in spans]
    assert covered[0][0] == 1000 and covered[-1][1] == 41_000
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))              # no gap, no overlap
    raw = path.read_bytes()
    assert all(data == raw[at:at + n] for at, n, data in reads)


def test_prefetch_faults_in_an_array_that_maps_no_file(monkeypatch):
    monkeypatch.setattr(host_table, "open", lambda *a, **k: pytest.fail("no file to read"), raising=False)
    assert host_table._prefetch([np.ones((3, 5000), dtype=np.uint16)], workers=2) >= 0.0
