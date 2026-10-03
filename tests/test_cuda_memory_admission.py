"""Admission on unified memory: MemFree, idle allocator blocks and reclaimable page cache above a hard floor."""

from __future__ import annotations

from tensorfold.cuda.admission import GIB, Admission, page_cache_credit, read_meminfo
from tensorfold.cuda.memory_gate import MemoryGate


def mi(free=10, available=30, dirty=1, mapped=3):
    return {
        "MemTotal": 128 * GIB,
        "MemFree": free * GIB,
        "MemAvailable": available * GIB,
        "Dirty": dirty * GIB,
        "Writeback": 0,
        "Mapped": mapped * GIB,
    }


def test_page_cache_counts_less_dirty_and_mapped_pages():
    assert page_cache_credit(mi()) == (30 - 10 - 1 - 3) * GIB
    assert page_cache_credit(mi(mapped=1)) == (30 - 10 - 1 - 2) * GIB  # at least 2 GiB stays
    assert page_cache_credit({}) == 0


def test_unified_memory_admits_above_the_hard_floor():
    a = Admission(
        unified=True, meminfo=lambda: mi(free=2, available=8), idle=lambda: GIB, floor=6 * GIB, hard_floor=4 * GIB
    )
    assert a.usable() == (2 + 1 + 8 - 2 - 1 - 3) * GIB
    assert a.fits(0) and not a.fits(2 * GIB) and a.waits == 1
    assert a.low()
    assert "under the 4.00 GiB floor" in a.why(2 * GIB)


def test_a_discrete_gpu_asks_the_gate_only():
    gate = MemoryGate(10 * GIB, GIB)
    a = Admission(gate, meminfo=lambda: mi(free=0, available=0))
    assert a.usable() is None and a.fits(8 * GIB) and not a.fits(10 * GIB) and not a.low()
    gate.take(5 * GIB)
    assert "stream caches hold 5.00 of 10.00 GiB" in a.why(5 * GIB)


def test_trim_gives_idle_blocks_back():
    trimmed = []
    a = Admission(unified=True, meminfo=lambda: mi(), idle=lambda: 3 * GIB, trim=lambda: trimmed.append(1))
    assert a.trim() == 3 * GIB and trimmed == [1]
    assert Admission().trim() == 0


def test_read_meminfo_parses_kilobytes(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:  4 kB\nMemFree: 2 kB\nHugePages_Total: 0\nMemAvailable: 3 kB\n")
    assert read_meminfo(str(f)) == {"MemTotal": 4096, "MemFree": 2048, "MemAvailable": 3072}
    assert read_meminfo(str(tmp_path / "missing")) == {}
