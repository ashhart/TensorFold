"""TENSORFOLD_MEMORY_RESERVE_GIB: what the CUDA startup budget leaves free (default max(4 GiB, a tenth of memory))."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity

GIB = capacity.GIB


def test_default_reserve_is_unchanged(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    odd = 121 * GIB + 7                                  # the GPU path rounds a tenth up, the host path down
    assert capacity.reserve_bytes(odd) == -(-odd // 10)
    assert capacity.reserve_bytes(odd, host=True) == odd // 10
    assert capacity.reserve_bytes(20 * GIB) == 4 * GIB
    assert capacity.reserve_bytes(20 * GIB, host=True) == 4 * GIB


def test_override(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.reserve_bytes(121 * GIB) == 6 * GIB
    assert capacity.reserve_bytes(121 * GIB, host=True) == 6 * GIB
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", " 2.5 ")
    assert capacity.reserve_bytes(121 * GIB) == int(2.5 * GIB)


@pytest.mark.parametrize("value", ["1", "0", "-3", "200", "nan", "lots"])
def test_out_of_range_or_not_a_number_refuses(monkeypatch, value):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", value)
    with pytest.raises(ValueError, match="TENSORFOLD_MEMORY_RESERVE_GIB"):
        capacity.reserve_bytes(121 * GIB)


def _cuda(free, total):
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (free, total)))


def test_available_bytes_uses_the_reserve(monkeypatch):
    meminfo = {"MemTotal": 121 * GIB, "MemAvailable": 110 * GIB}
    monkeypatch.setattr(capacity, "_meminfo", lambda: meminfo)
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    torch = _cuda(100 * GIB, 121 * GIB)
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    assert capacity.available_bytes(torch) == 110 * GIB - 121 * GIB // 10
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.available_bytes(torch) == 104 * GIB
    monkeypatch.setattr(capacity, "unified", lambda torch: False)      # a discrete GPU: bounded by both
    assert capacity.available_bytes(torch) == 94 * GIB
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    assert capacity.available_bytes(torch) == 94 * GIB
