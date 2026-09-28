"""CUDA extensions build for the GPU present, whatever architecture list the container sets (#56)."""

from pathlib import Path

import pytest

from tensorfold.cuda import build

NGC_LIST = "8.0 8.6 9.0 10.0 11.0 12.0+PTX"


def _gpu(monkeypatch, capability, name="GPU"):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)


def test_every_extension_builds_through_the_helper():
    src = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
    direct = [p.relative_to(src) for p in src.rglob("*.py")
              if p.name != "build.py" and "from torch.utils.cpp_extension import load" in p.read_text()]
    assert direct == []


@pytest.mark.torch
def test_the_flags_name_only_this_gpu(monkeypatch):
    _gpu(monkeypatch, (12, 1))
    assert build.arch_flags() == ["-gencode=arch=compute_121,code=sm_121"]


@pytest.mark.torch
def test_an_older_gpu_is_refused_by_name(monkeypatch):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    with pytest.raises(RuntimeError, match=r"capability 9\.0 or newer.*RTX 3090.*is 8\.6"):
        build.arch_flags()


@pytest.mark.torch
def test_the_container_list_adds_nothing(monkeypatch):
    import torch.utils.cpp_extension as ext

    _gpu(monkeypatch, (12, 1))
    seen = {}
    monkeypatch.setattr(ext, "load", lambda **kw: seen.update(kw) or "module")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", NGC_LIST)
    assert build.load(name="x", sources=[], extra_cuda_cflags=["-O3"]) == "module"
    assert seen["extra_cuda_cflags"] == ["-O3", "-gencode=arch=compute_121,code=sm_121"]
    assert ext._get_cuda_arch_flags(seen["extra_cuda_cflags"]) == []
