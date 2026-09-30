"""Discovery / unsupported-option tests for the DeepSeek V4 CUDA port audit.

These probe what the current TensorFold codebase supports and does NOT support,
so the port plan records every upstream option that is absent today.

CPU-only by design: they import neither the torch backend nor MLX (gate G0).
Some tests are intended-failing (RED) because the behaviour they assert is not
yet implemented; each carries `_INTENDED_FAIL` in its id and is recorded in the
evidence manifest docs/evidence/deepseek-v4-t01-evidence.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tensorfold import families


def _write_config(tmp_path: Path, config: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


# 0731-style GGUF storage formats the port must read (R2), none of which the
# current deepseek_v4 family declares.
GGUF_IQ2XXS = {"model_type": "deepseek_v4",
               "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "IQ2_XXS"}}
GGUF_Q2K = {"model_type": "deepseek_v4",
            "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "Q2_K"}}
GGUF_Q8_0 = {"model_type": "deepseek_v4",
             "quantization_config": {"quant_method": "gguf", "bits": 8, "format": "Q8_0"}}
GGUF_MIXED = {"model_type": "deepseek_v4",
              "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "mixed"}}

DS_FAMILY = families.families()["deepseek_v4"]


def test_deepseek_v4_has_no_cuda_backend():
    # The MLX family ships a `load` member but no `cuda_engine`: CUDA is unsupported today.
    assert families.backends_of(DS_FAMILY) == ("mlx",)
    assert not hasattr(DS_FAMILY.package, "cuda_engine")


def test_deepseek_v4_declares_only_mlx_quant_methods():
    # QUANT_METHODS = {"mlx": ("mlx",)}: affine MLX weights only; GGUF IQ2_XXS/Q2_K/Q8_0 absent.
    assert DS_FAMILY.package.QUANT_METHODS == {"mlx": ("mlx",)}


@pytest.mark.parametrize("name,config", [
    ("IQ2_XXS", GGUF_IQ2XXS), ("Q2_K", GGUF_Q2K), ("Q8_0", GGUF_Q8_0), ("mixed", GGUF_MIXED),
])
def test_gguf_quant_refused_on_cuda(name, config):
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(DS_FAMILY, config, "cuda")


@pytest.mark.parametrize("name,config", [
    ("IQ2_XXS", GGUF_IQ2XXS), ("Q2_K", GGUF_Q2K), ("Q8_0", GGUF_Q8_0), ("mixed", GGUF_MIXED),
])
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


def test_pr119_q8_0_gguf_reader_is_not_present_in_upstream():
    # PR #119 (feni6) adds tools/glm5_q8_0_gguf_to_mlx.py and a glm5_next loader change.
    # It is OPEN and unmerged into upstream main: neither module ships here.
    assert importlib.util.find_spec("tensorfold.families.glm5_next.gguf") is None
    assert not (Path(__file__).resolve().parents[1] / "tools" / "glm5_q8_0_gguf_to_mlx.py").exists()


def _INTENDED_FAIL_check_deepseek_v4_check_refuses_gguf(tmp_path):
    # Intended-failing (RED): exercising deepseek_v4.check() requires importing
    # weights.py -> mlx, which is absent in the CPU-only audit environment.
    from tensorfold.families import deepseek_v4  # package __init__ imports no mlx at top level

    folder = _write_config(tmp_path / "gguf", GGUF_IQ2XXS)
    deepseek_v4.check(folder)


def test_intended_fail_check_refuses_gguf_is_red():
    # The behaviour above is what the port needs (R2: refuse unsupported ranks/features
    # before loading). Recorded as intended-failing: needs an MLX-capable environment.
    pytest.xfail("requires MLX; documented in evidence manifest as intended-failing RED")
