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


def test_a_unified_grant_is_available_memory_less_the_floor(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    total, available = meminfo
    # a GB10's free figure is MemFree: 68 GB here, although 117.5 GB is available once the page cache is reclaimed
    assert capacity.available_bytes(device(True)) == available - total // 10    # the host keeps a tenth of its RAM free


def test_a_discrete_grant_ignores_host_memory(monkeypatch):
    # only the loading buffers need host RAM, and host_stream_bytes weighs those on its own
    monkeypatch.setattr(Path, "read_text", lambda *a: "MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\n")
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    assert capacity.available_bytes(device(False, 20 * GB, 80 * GB)) == 12 * GB    # the card's free memory less a tenth of it


def test_the_reserve_override_moves_the_unified_floor(meminfo, monkeypatch):
    _total, available = meminfo
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.available_bytes(device(True)) == available - 6 * capacity.GIB


def test_a_unified_grant_pays_the_floor_when_its_pool_is_small(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(Path, "read_text", lambda *a: "MemTotal: 41943040 kB\nMemAvailable: 31457280 kB\n")
    # 40 GiB of RAM with 30 GiB free: the four GiB floor is the largest one, so 26 GiB is granted
    assert capacity.available_bytes(device(True)) == 30 * capacity.GIB - 4 * capacity.GIB


def test_the_limit_env_caps_the_grant_on_both_bounds(meminfo, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "20")
    # 20 GiB is under the unified grant (117.5e6 kB less a tenth of RAM) and under the discrete card's free memory
    assert capacity.available_bytes(device(True)) == 20 * capacity.GIB
    assert capacity.available_bytes(device(False, 50 * GB, 128 * GB)) == 20 * capacity.GIB


def test_a_discrete_grant_under_a_limit_keeps_the_cards_floor(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "31")
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    # the card keeps its floor under a limit too: four GiB by default, two at the smallest reserve
    assert capacity.available_bytes(device(False, 32 * capacity.GIB, 32 * capacity.GIB)) == 28 * capacity.GIB
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    assert capacity.available_bytes(device(False, 32 * capacity.GIB, 32 * capacity.GIB)) == 30 * capacity.GIB


def test_the_limit_env_leaves_the_floored_grant_the_ceiling(meminfo, monkeypatch):
    total, available = meminfo
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "1000")
    assert capacity.available_bytes(device(True)) == available - total // 10
    assert capacity.available_bytes(device(False, 20 * GB, 80 * GB)) == 12 * GB


def test_the_limit_env_applies_without_host_memory(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "40")
    # no /proc/meminfo means a unified GPU's grant is its free memory less a tenth of it, and 40 GiB is under that
    assert capacity.available_bytes(device(True, 68 * GB, 128 * capacity.GIB)) == 40 * capacity.GIB
    # one pool, so the four GiB the host keeps free comes off the grant: 30 GB leaves 25.71 GiB for weights and cache
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")
    assert capacity.available_bytes(device(True, 30 * GB, 40 * capacity.GIB)) == 30 * GB - 4 * capacity.GIB


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "inf", "12GB"])
def test_an_invalid_limit_env_is_refused(monkeypatch, value):
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", value)
    with pytest.raises(ValueError, match="TENSORFOLD_CUDA_MEMORY_LIMIT_GB"):
        capacity.available_bytes(device(True))


def test_page_room_is_memavailable_on_unified_memory_only(meminfo):
    _, available = meminfo
    assert capacity.page_room(device(True)) == available
    assert capacity.page_room(device(False, 20 * GB, 80 * GB)) is None


def test_default_window_leaves_mapped_tables_their_pages():
    geometry = Geometry(lambda slots: slots * 100_000, 7)
    weights = Weights(resident=80 * GB, staging=10 * GB, mapped=32 * GB)
    budget, room = 107 * GB, 120 * GB                      # the host memory available beside the tables
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
