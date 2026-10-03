"""--device / TF_CUDA_DEVICE pick the GPU a CUDA engine runs on."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import device


def _torch(count):
    picked = []
    cuda = SimpleNamespace(device_count=lambda: count, set_device=picked.append)
    return SimpleNamespace(cuda=cuda), picked


def test_device_zero_without_the_setting(monkeypatch):
    monkeypatch.delenv("TF_CUDA_DEVICE", raising=False)
    torch, picked = _torch(2)
    assert device.select(torch) == 0 and picked == [0]


def test_the_setting_picks_the_device(monkeypatch):
    monkeypatch.setenv("TF_CUDA_DEVICE", "1")
    torch, picked = _torch(2)
    assert device.select(torch) == 1 and picked == [1]


def test_a_device_that_is_not_visible_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("TF_CUDA_DEVICE", "2")
    torch, picked = _torch(2)
    with pytest.raises(ValueError, match="--device 2"):
        device.select(torch)
    assert picked == []
