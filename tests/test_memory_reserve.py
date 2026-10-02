"""TENSORFOLD_MEMORY_RESERVE_GIB: the host RAM the CUDA startup keeps free (default max(4 GiB, a tenth))."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity

GIB = capacity.GIB


def test_default_reserve_is_a_tenth_of_the_pool(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    odd = 121 * GIB + 7                                  # a tenth of the pool, counted down
    assert capacity.reserve_bytes(odd) == odd // 10
    assert capacity.reserve_bytes(20 * GIB) == 4 * GIB    # and never less than four GiB


def test_override(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.reserve_bytes(121 * GIB) == 6 * GIB
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", " 2.5 ")
    assert capacity.reserve_bytes(121 * GIB) == int(2.5 * GIB)


@pytest.mark.parametrize("value", ["1", "0", "-3", "200", "nan", "lots"])
def test_out_of_range_or_not_a_number_refuses(monkeypatch, value):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", value)
    with pytest.raises(ValueError, match="TENSORFOLD_MEMORY_RESERVE_GIB"):
        capacity.reserve_bytes(121 * GIB)


def _cuda(free, total):
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (free, total)))


def test_the_reserve_floors_a_unified_grant_and_ignores_a_discrete_one(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 121 * GIB, "MemAvailable": 110 * GIB})
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    # a unified GPU's grant is the host's available RAM less the floor: a tenth of its total, never less than 4 GiB
    assert capacity.available_bytes(_cuda(100 * GIB, 121 * GIB)) == 110 * GIB - 12 * GIB - GIB // 10
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.available_bytes(_cuda(100 * GIB, 121 * GIB)) == 104 * GIB    # the override moves the floor
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB")
    torch = _cuda(100 * GIB, 121 * GIB)
    monkeypatch.setattr(capacity, "unified", lambda torch: False)
    assert capacity.available_bytes(torch) == 100 * GIB                           # a discrete card's own free memory
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.available_bytes(torch) == 100 * GIB                           # which no reserve override changes
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    assert capacity.available_bytes(torch) == 100 * GIB                           # with no host memory to read either
