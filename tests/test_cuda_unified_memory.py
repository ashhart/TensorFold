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


def test_unified_budget_counts_the_page_cache_as_available(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    total, available = meminfo
    # GB10's free figure is MemFree: 68 GB here although 117 GB is available once the page cache is reclaimed
    assert capacity.available_bytes(device(True)) == available - total // 10


def test_discrete_budget_is_framed_by_free_and_host_memory(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    total, available = meminfo
    free, gpu = 20 * GB, 80 * GB
    assert capacity.available_bytes(device(False, free, gpu)) == min(free - gpu // 10, available - total // 10)


def test_the_reserve_env_replaces_both_guards(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    _, available = meminfo
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "3")
    # 3 GiB replaces max(4 GiB, a tenth of the memory) on the GPU bound and on the host bound
    assert capacity.available_bytes(device(True)) == available - 3 * capacity.GIB
    assert capacity.available_bytes(device(False, 20 * GB, 80 * GB)) == min(20 * GB - 3 * capacity.GIB,
                                                                            available - 3 * capacity.GIB)


def test_the_reserve_env_applies_without_host_memory(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    free = 68 * GB
    assert capacity.available_bytes(device(True, free)) == free - 2 * capacity.GIB


@pytest.mark.parametrize("value", ["0", "1", "-1", "nan", "inf", "12GB"])
def test_an_invalid_reserve_env_is_refused(monkeypatch, value):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", value)
    with pytest.raises(ValueError, match="TENSORFOLD_MEMORY_RESERVE_GIB"):
        capacity.available_bytes(device(True))


def test_the_limit_env_drops_the_reserve_from_live_free_memory(meminfo, monkeypatch):
    _, available = meminfo
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "1000")
    # the operator owns the headroom: live free memory carries no reserve, on either kind of GPU;
    # the limit line takes the non-tensor overhead off the grant, so the tensors fit under it
    assert capacity.available_bytes(device(True)) == min(available, 1000 * capacity.GIB - capacity.NON_TENSOR_OVERHEAD)
    assert capacity.available_bytes(device(False, 20 * GB, 80 * GB)) == 20 * GB
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "10")
    assert capacity.available_bytes(device(False, 20 * GB, 80 * GB)) == 10 * capacity.GIB - capacity.NON_TENSOR_OVERHEAD


@pytest.mark.parametrize("integrated", [False, True])
def test_31_gib_limit_is_the_startup_budget_on_a_32_gb_card(monkeypatch: pytest.MonkeyPatch,
                                                             integrated: bool) -> None:
    """Grant 31 GiB less the non-tensor overhead, although free memory less the default reserve is about 27 GiB.

    :param monkeypatch: Fixture for setting the limit and available host memory.
    :param integrated: Whether the device shares host memory.
    """
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "31")
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * capacity.GIB,
                                                     "MemAvailable": 32 * capacity.GIB})
    card = device(integrated, int(30.86 * capacity.GIB), 32 * capacity.GIB)
    assert capacity.budget_bytes(card) == 31 * capacity.GIB - capacity.NON_TENSOR_OVERHEAD


@pytest.mark.parametrize("limit", ["10", "200"])
def test_the_limit_env_overrides_the_budget_in_both_directions(meminfo, monkeypatch, limit):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "8")
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", limit)
    # the limit replaces free memory and the reserve: it lowers the budget and raises it past free memory;
    # its tensors sit the non-tensor overhead under the line, so serving fits the GiB nvidia-smi shows
    assert capacity.budget_bytes(device(True)) == int(limit) * capacity.GIB - capacity.NON_TENSOR_OVERHEAD
    assert capacity.budget_bytes(device(False, 20 * GB, 80 * GB)) == int(limit) * capacity.GIB - capacity.NON_TENSOR_OVERHEAD


def test_without_the_limit_the_budget_keeps_the_reserve(meminfo, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    total, available = meminfo
    assert capacity.budget_bytes(device(True)) == available - total // 10
    assert capacity.budget_bytes(device(False, 20 * GB, 80 * GB)) == 20 * GB - 80 * GB // 10


def test_the_limit_env_applies_without_host_memory(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", "60")
    assert capacity.budget_bytes(device(True, 50 * GB)) == 60 * capacity.GIB - capacity.NON_TENSOR_OVERHEAD
    assert capacity.available_bytes(device(True, 50 * GB)) == min(50 * GB, 60 * capacity.GIB - capacity.NON_TENSOR_OVERHEAD)


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "inf", "12GB"])
def test_an_invalid_limit_env_is_refused(monkeypatch, value):
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB", value)
    with pytest.raises(ValueError, match="TENSORFOLD_CUDA_MEMORY_LIMIT_GB"):
        capacity.budget_bytes(device(True))


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
    with pytest.raises(ValueError) as refused:
        choose(explicit)
    assert "requested context 65536" in str(refused.value)
    assert f"{10 * GB / capacity.GIB:.2f} GiB" in str(refused.value)


def test_the_budget_refusal_names_how_much_memory_was_tried():
    geometry = Geometry(lambda slots: slots * 1000, 7)
    plan = make_plan(1048576, 262144, True, 10 * GB, Weights(9 * GB, 5 * GB), geometry)
    with pytest.raises(ValueError) as refused:
        choose(plan)
    text = str(refused.value)
    assert f"CUDA startup memory budget of {10 * GB / capacity.GIB:.2f} GiB" in text
    assert "largest fitting prompt-plus-reply window: 0 tokens" in text
