"""Every family reads the GPU generation the same way: ``applegpu_gN`` gives N, and a GPU that names no generation
(``air64_v27``, a macOS VM's paravirtual GPU) or a CPU default device (``arm64``) never reads as an M5."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families import qwen3_5  # noqa: E402
from tensorfold.kernels import device, threads  # noqa: E402
from tensorfold.kernels.gemma.v1 import matmul as gemma  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import kernels as nemotron  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import base, prefill_mm  # noqa: E402


def _clear():
    for cached in (device.generation, base._generation):
        clear = getattr(cached, "cache_clear", None)
        if clear is not None:
            clear()


@pytest.fixture
def gpu(monkeypatch):
    """Sets the architecture the Metal GPU reports; the default device (the CPU here) reports ``arm64``."""

    def use(architecture: str, metal: bool = True) -> None:
        def info(dev=None):
            asked = mx.default_device() if dev is None else dev       # a Device, or a DeviceType such as mx.gpu
            return {"architecture": architecture if getattr(asked, "type", asked) == mx.gpu else "arm64"}

        monkeypatch.setattr(mx, "device_info", info)
        monkeypatch.setattr(mx.metal, "is_available", lambda: metal)
        _clear()

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield use
    mx.set_default_device(previous)
    _clear()


def _readings():
    return {"qwen3_5": qwen3_5.tensor_units(), "gemma": gemma.tensor_units(), "nemotron": nemotron.tensor_units(),
            "prefill_mm": prefill_mm._tensor_units(), "prefill_mm.gpu": prefill_mm.gpu_tensor_units(),
            "flash_next nib_rows": base.nib_rows() == 0}


@pytest.mark.parametrize("architecture, generation", [
    ("applegpu_g13g", 13), ("applegpu_g14s", 14), ("applegpu_g15p", 15), ("applegpu_g16s", 16),
    ("applegpu_g17s", 17), ("applegpu_g18d", 18), ("air64_v27", 0), ("", 0)])
def test_every_family_reads_the_same_generation(gpu, architecture, generation):
    gpu(architecture)
    assert device.generation() == generation
    units = generation >= device.TENSOR_UNITS
    assert device.tensor_units() is units
    assert _readings() == dict.fromkeys(_readings(), units)


def test_without_metal_there_is_no_generation(gpu):
    gpu("applegpu_g17s", metal=False)
    assert device.generation() == 0 and not device.tensor_units()


def test_thread_probing_follows_the_generation(gpu):
    for architecture, probes in (("applegpu_g14s", True), ("applegpu_g15p", False), ("air64_v27", True)):
        gpu(architecture)
        assert threads._probes() is probes
