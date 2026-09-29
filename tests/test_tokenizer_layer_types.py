"""A config.json whose per-layer lists also count the MTP layers (GLM-5.3 oQ4: 46 mlp_layer_types for 45 layers)."""

import json

import pytest

from tensorfold.families.tokenizer import trimmed_config

transformers = pytest.importorskip("transformers")


def _config(tmp_path, mlp_types):
    raw = {"model_type": "glm5_next", "num_hidden_layers": 4, "first_k_dense_replace": 1,
           "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
           "mlp_layer_types": mlp_types, "num_nextn_predict_layers": 1}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


def test_the_mtp_layers_entries_are_cut_to_the_decoders_layers(tmp_path):
    try:
        transformers.AutoConfig.for_model("glm5_next")
    except (KeyError, ValueError):
        pytest.skip("this transformers has no glm5_next config")
    config = trimmed_config(_config(tmp_path, ["dense", "sparse", "sparse", "sparse", "sparse"]))
    text = getattr(config, "text_config", config)
    assert text.num_hidden_layers == 4 and list(text.mlp_layer_types) == ["dense", "sparse", "sparse", "sparse"]


def test_a_config_whose_lists_fit_is_left_alone(tmp_path):
    assert trimmed_config(_config(tmp_path, ["dense", "sparse", "sparse", "sparse"])) is None
