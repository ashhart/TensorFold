"""Flash Next preflight checks the language weights without claiming support for an unused vision tower."""

import json

import pytest

from tensorfold.families.qwen4_exp import check
from tensorfold.vision.qwen_checkpoint import PREFIXES


def write_config(path, name, algo):
    config = {"model_type": "qwen4_exp", "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
                             name: {"quant_algo": algo}},
        "config_groups": {"experts": {"weights": {"num_bits": 4, "group_size": 16}}}}}
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("prefix", PREFIXES)
def test_unused_nvfp4_vision_paths_do_not_block_language_preflight(tmp_path, prefix):
    write_config(tmp_path, prefix + "blocks.0.mlp.linear_fc2", "W4A16_NVFP4")
    check(tmp_path)


def test_nvfp4_language_projection_is_still_refused(tmp_path):
    write_config(tmp_path, "model.language_model.layers.0.linear_attn.in_proj_qkv", "W4A16_NVFP4")
    with pytest.raises(ValueError, match="routed experts and n-gram tables only"):
        check(tmp_path)
