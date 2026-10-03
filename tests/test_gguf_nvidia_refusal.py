"""A GGUF Qwen3.8-27B on a non-ROCm build stops with a clear message before any Gufo (HIP-only) kernel builds."""

import pytest

torch = pytest.importorskip("torch")


def test_gguf_on_nvidia_says_rocm_only(tmp_path, monkeypatch):
    from tensorfold.cuda import rocm
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    (tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(rocm, "HIP", False)
    with pytest.raises(ValueError, match="run on AMD GPUs"):
        Qwen27Engine(tmp_path, None)


def test_gguf_on_rocm_passes_the_platform_check(tmp_path, monkeypatch):
    from tensorfold.cuda import rocm
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    (tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(rocm, "HIP", True)
    with pytest.raises(Exception) as err:          # fails later (no real model here), never on the platform check
        Qwen27Engine(tmp_path, None)
    assert "run on AMD GPUs" not in str(err.value)
