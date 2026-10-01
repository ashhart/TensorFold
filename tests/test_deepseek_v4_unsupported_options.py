"""CPU discovery and storage boundaries for the DeepSeek native CUDA adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfold import families


def _write_config(tmp_path: Path, config: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


# 0731 mixed GGUF storage is supported only by the native CUDA adapter.
GGUF_IQ2XXS = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "IQ2_XXS"},
}
GGUF_Q2K = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "Q2_K"},
}
GGUF_Q8_0 = {"model_type": "deepseek_v4", "quantization_config": {"quant_method": "gguf", "bits": 8, "format": "Q8_0"}}
GGUF_MIXED = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "mixed"},
}

DS_FAMILY = families.families()["deepseek_v4"]


def test_deepseek_v4_registers_cuda_backend():
    assert set(families.backends_of(DS_FAMILY)) == {"mlx", "cuda"}
    assert callable(DS_FAMILY.package.cuda_engine)


def test_deepseek_v4_declares_backend_specific_quant_methods():
    assert DS_FAMILY.package.QUANT_METHODS == {"mlx": ("mlx",), "cuda": ("gguf",)}


@pytest.mark.parametrize(
    "name,config",
    [
        ("IQ2_XXS", GGUF_IQ2XXS),
        ("Q2_K", GGUF_Q2K),
        ("Q8_0", GGUF_Q8_0),
        ("mixed", GGUF_MIXED),
    ],
)
def test_gguf_quant_declared_on_cuda(name, config):
    families.require_readable(DS_FAMILY, config, "cuda")


@pytest.mark.parametrize(
    "name,config",
    [
        ("IQ2_XXS", GGUF_IQ2XXS),
        ("Q2_K", GGUF_Q2K),
        ("Q8_0", GGUF_Q8_0),
        ("mixed", GGUF_MIXED),
    ],
)
def test_gguf_quant_refused_on_mlx(name, config):
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(DS_FAMILY, config, "mlx")


def test_no_vision_or_pro_family_registered():
    # issue #14 asked for DeepSeek-V4-flash-vision-exp and DeepSeek-V4-Pro; neither is registered.
    available = families.families()
    assert "deepseek_v4_flash_vision_exp" not in available
    assert "deepseek_v4_pro" not in available
    assert "deepseek_v4_vision" not in available


def test_detect_refuses_unknown_deepseek_model_types(tmp_path):
    for model_type in ("deepseek_v4_flash_vision_exp", "deepseek_v4_pro", "deepseek_v4_vision"):
        folder = _write_config(tmp_path / model_type.replace("/", "_"), {"model_type": model_type})
        with pytest.raises(ValueError, match="no recipe"):
            families.detect(folder)


def test_gguf_check_requires_provenance_without_mlx(tmp_path):
    from tensorfold.families import deepseek_v4

    folder = _write_config(tmp_path / "gguf", GGUF_IQ2XXS)
    with pytest.raises(ValueError, match="descriptor"):
        deepseek_v4.check(folder)
    (folder / "descriptor.json").write_text(json.dumps({"source": "candidate.gguf"}))
    deepseek_v4.check(folder)
