"""Admission on unified memory (GB10): page cache is available and a default window keeps mapped tables."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.capacity import Geometry, Weights, choose, make_plan

GB = 10**9
MEMINFO = "MemTotal: 127535264 kB\nMemFree: 66406250 kB\nMemAvailable: 117500000 kB\n"   # a Spark with 50 GB cached


def device(integrated: bool, free: int = 68 * GB, total: int = 130_596_110_336):
    props = SimpleNamespace(is_integrated=int(integrated), total_memory=total)
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (free, total),
                                                get_device_properties=lambda index: props))


@pytest.fixture
def meminfo(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: MEMINFO)
    total, available = 127535264 * 1024, 117500000 * 1024
    return total, available


def test_unified_budget_counts_the_page_cache_as_available(meminfo):
    total, available = meminfo
    # GB10's free figure is MemFree: 68 GB here although 117 GB is available once the page cache is reclaimed
    assert capacity.available_bytes(device(True)) == available - total // 10


def test_discrete_budget_keeps_both_guards(meminfo):
    total, available = meminfo
    free, gpu = 20 * GB, 80 * GB
    assert capacity.available_bytes(device(False, free, gpu)) == min(free - gpu // 10, available - total // 10)


def test_page_room_is_memavailable_on_unified_memory_only(meminfo):
    _, available = meminfo
    assert capacity.page_room(device(True)) == available
    assert capacity.page_room(device(False, 20 * GB, 80 * GB)) is None


def test_default_window_leaves_mapped_tables_their_pages():
    geometry = Geometry(lambda slots: slots * 100_000, 7)
    weights = Weights(resident=80 * GB, staging=10 * GB, mapped=32 * GB)
    budget, room = 107 * GB, 120 * GB                      # the reserve taken from 120 GB available
    default = make_plan(262144, 262144, False, budget, weights, geometry, room=room)
    # caches and tables inside what is available: 80 + 32 + slots x 100 KB <= 120 GB
    assert choose(default) == 80_000 - 7
    assert default.receipt(choose(default))["mapped_tables_resident"] is True
    # an explicit window may use the tables' pages and is refused only past the budget; startup names the window
    # that would keep them
    explicit = make_plan(262144, 200_000, True, budget, weights, geometry, room=room)
    assert choose(explicit) == 200_000 and explicit.receipt(200_000)["mapped_tables_resident"] is False
    assert "a --context of 79993 or less, or fewer --parallel streams, keeps them resident" in \
        capacity.tables_note(explicit)
    small = make_plan(262144, 50_000, True, budget, weights, geometry, room=room)
    assert small.keeps_tables is True and capacity.tables_note(small) is None
    with pytest.raises(ValueError, match="largest fitting"):
        choose(make_plan(262144, 250_000, True, 100 * GB, weights, geometry, room=room))


def test_default_window_pages_the_tables_when_they_cannot_stay():
    geometry = Geometry(lambda slots: slots * 100_000, 7)
    weights = Weights(resident=80 * GB, staging=10 * GB, mapped=32 * GB)
    plan = make_plan(262144, 262144, False, 107 * GB, weights, geometry, room=100 * GB)
    assert choose(plan) == 262144                          # the budget holds the caches; the tables will page
    assert plan.receipt(262144)["mapped_tables_resident"] is False
    assert "(free memory to keep them resident)" in capacity.tables_note(plan)


def test_default_refusal_names_the_native_window_not_a_request():
    geometry = Geometry(lambda slots: slots * 1000, 7)
    plan = make_plan(1048576, 1048576, False, 10 * GB, Weights(9 * GB, 5 * GB), geometry)
    with pytest.raises(ValueError) as refused:
        choose(plan)
    assert "requested" not in str(refused.value)
    assert "1048576-token native window" in str(refused.value)
    explicit = make_plan(1048576, 65536, True, 10 * GB, Weights(9 * GB, 5 * GB), geometry)
    with pytest.raises(ValueError, match="requested context 65536"):
        choose(explicit)
