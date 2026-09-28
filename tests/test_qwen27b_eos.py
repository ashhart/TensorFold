"""Qwen3.8-27B on CUDA stops where the checkpoint says a reply ends, reading generation_config.json too (CPU only)."""

import json

import pytest

pytest.importorskip("torch")

from tensorfold.families.qwen3_5.cuda.weights import Config  # noqa: E402

IM_END, ENDOFTEXT = 248046, 248044
TEXT = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 248320, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
}


def checkpoint(tmp_path, config, generation=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    return tmp_path


def test_exl3_pack_stops_at_im_end_from_generation_config(tmp_path):
    # turboderp's 27B EXL3 packs: no top-level eos_token_id, text_config names <|endoftext|> only
    d = checkpoint(tmp_path, {"model_type": "qwen3_5", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT]})
    assert Config.read(d).eos == (ENDOFTEXT, IM_END)


def test_top_level_ids_keep_their_order(tmp_path):
    # the MLX checkpoint's layout: the top level already lists both
    d = checkpoint(tmp_path, {"model_type": "qwen3_5", "eos_token_id": [IM_END, ENDOFTEXT],
                              "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT]})
    assert Config.read(d).eos == (IM_END, ENDOFTEXT)


def test_without_generation_config_the_config_ids_are_used(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen3_5", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}})
    assert Config.read(d).eos == (ENDOFTEXT,)


def test_ids_only_in_generation_config(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen3_5", "text_config": TEXT}, {"eos_token_id": IM_END})
    assert Config.read(d).eos == (IM_END,)


def test_no_ids_anywhere_is_an_error(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen3_5", "text_config": TEXT}, {"temperature": 1.0})
    with pytest.raises(ValueError, match="eos_token_id"):
        Config.read(d)
