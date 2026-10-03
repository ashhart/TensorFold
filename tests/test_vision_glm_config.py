from __future__ import annotations

import pytest

from tensorfold.vision.config import validate_vision_config
from tensorfold.vision.qwen_checkpoint import vision_key


def test_glm_vision_config_accepts_the_native_checkpoint():
    config = {"model_type": "glm5_next", "image_token_id": 154854, "image_start_token_id": 154830,
              "image_end_token_id": 154831, "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 4096, "hidden_size": 1024}}
    assert validate_vision_config(config, "glm5_next")["hidden_size"] == 1024


def test_glm_vision_config_requires_image_tokens_and_matching_width():
    config = {"model_type": "glm5_next", "text_config": {"hidden_size": 4096},
              "vision_config": {"out_hidden_size": 2048}}
    with pytest.raises(ValueError, match="output width"):
        validate_vision_config(config, "glm5_next")
    config["vision_config"]["out_hidden_size"] = 4096
    with pytest.raises(ValueError, match="image-token configuration"):
        validate_vision_config(config, "glm5_next")


def test_glm_vision_loader_recognizes_legacy_mlx_vision_model_prefix():
    assert vision_key("vision_model.blocks.0.attn.qkv.weight") == "blocks.0.attn.qkv.weight"


def test_glm_vision_rejects_cuda_before_reading_checkpoint(monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    from tensorfold import families, serve_options

    def read_config(path):
        raise AssertionError('unsupported backend must be rejected before reading checkpoint')

    monkeypatch.setattr(families, 'read_config', read_config)
    with pytest.raises(ValueError, match='GLM.*MLX-only'):
        serve_options.check(Namespace(vision=True), SimpleNamespace(model_type='glm5_next'), 'cuda', 'unused')
