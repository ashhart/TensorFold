"""Qwen3.6 MoE's startup check: the CUDA engine's 4-bit/64 rule on CUDA, the row decoder's rule on Macs."""

import json

import pytest

from tensorfold.families import qwen3_5_moe

UNIFORM = {"model_type": "qwen3_5_moe", "quantization": {"bits": 4, "group_size": 64}}
# an oQ-style mixed checkpoint (Ornith-1.5-35B-A3B oQ4e): 4-bit groups of 32 with wider protected layers
MIXED = {"model_type": "qwen3_5_moe", "quantization": {
    "bits": 4, "group_size": 32, "mode": "affine",
    "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 6, "group_size": 64},
    "language_model.model.layers.0.mlp.shared_expert.down_proj": {"bits": 8, "group_size": 128},
    "language_model.model.layers.0.mlp.switch_mlp.down_proj": {"bits": 5, "group_size": 64}}}
UNQUANTIZED = {"model_type": "qwen3_5_moe"}


def _dir(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.mark.parametrize("config", [UNIFORM, MIXED])
def test_a_mac_takes_any_mlx_affine_layout(tmp_path, monkeypatch, config):
    monkeypatch.setattr("sys.platform", "darwin")
    qwen3_5_moe.check(_dir(tmp_path, config))


def test_a_mac_refuses_unquantized_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(ValueError, match="on a Mac"):
        qwen3_5_moe.check(_dir(tmp_path, UNQUANTIZED))


def test_cuda_keeps_its_four_bit_groups_of_64(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    qwen3_5_moe.check(_dir(tmp_path, UNIFORM))
    with pytest.raises(ValueError, match="groups of 64"):
        qwen3_5_moe.check(_dir(tmp_path, MIXED))
