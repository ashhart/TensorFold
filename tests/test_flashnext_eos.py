"""Flash Next on CUDA stops where the checkpoint says a reply ends, reading generation_config.json too (CPU only)."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")          # Flash Next's CUDA weights import its triton kernels

from tensorfold.families.qwen4_exp.cuda.weights import Config, stop_ids  # noqa: E402

IM_END, ENDOFTEXT = 248046, 248044
TEXT = {
    "hidden_size": 64, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
    "vocab_size": 248320, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
}


def checkpoint(tmp_path, config, generation=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    return tmp_path


def test_exl3_pack_stops_at_im_end_from_generation_config(tmp_path):
    # the EXL3 packs' layout: no top-level eos_token_id, text_config names <|endoftext|> only
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT], "bos_token_id": ENDOFTEXT})
    c = Config.read(d)
    assert c.eos == (ENDOFTEXT, IM_END)
    assert c.ple_eos == ENDOFTEXT                  # the n-gram tables' end id still comes from text_config


def test_top_level_ids_keep_their_order(tmp_path):
    # the MLX checkpoint's layout: the top level already lists both
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "eos_token_id": [IM_END, ENDOFTEXT],
                              "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT]})
    assert Config.read(d).eos == (IM_END, ENDOFTEXT)


def test_without_generation_config_the_config_ids_are_used(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}})
    assert Config.read(d).eos == (ENDOFTEXT,)


def test_scalar_and_missing_values(tmp_path):
    gen = tmp_path / "generation_config.json"
    gen.write_text(json.dumps({"eos_token_id": IM_END}))
    assert stop_ids(ENDOFTEXT, gen) == (ENDOFTEXT, IM_END)
    gen.write_text(json.dumps({"temperature": 1.0}))
    assert stop_ids([ENDOFTEXT], gen) == (ENDOFTEXT,)
    with pytest.raises(ValueError, match="eos_token_id"):
        stop_ids(None, tmp_path / "absent.json")
