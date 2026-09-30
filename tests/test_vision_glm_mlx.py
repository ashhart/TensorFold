"""GLM vision feature insertion leaves the language model and MTP inputs in their existing path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.vision.glm_mlx import GLMVisionFrontend  # noqa: E402
from tensorfold.vision.glm_processing import PreparedGLMVisionPrompt  # noqa: E402


class Tower:
    def __init__(self):
        self.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32)))

    def __call__(self, pixels, grid):
        return mx.full((2, 4), 7, dtype=mx.bfloat16)


@pytest.mark.parametrize("flat_embeddings", [False, True])
def test_glm_vision_encode_replaces_only_image_token_embeddings(flat_embeddings):
    config = {"model_type": "glm5_next", "image_token_id": 10,
              "vision_config": {"out_hidden_size": 4, "patch_size": 14,
                                "temporal_patch_size": 2, "spatial_merge_size": 2}}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 10),
                                image_token="<|image|>", image_processor=SimpleNamespace(
                                    patch_size=14, temporal_patch_size=2, merge_size=2))
    embeddings = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4)
    embed = lambda tokens: embeddings.reshape(-1, 4) if flat_embeddings else embeddings
    front = GLMVisionFrontend(config, embed, Tower(), processor, mx)
    prepared = PreparedGLMVisionPrompt((1, 10, 10, 2), np.zeros((2, 1176), dtype=np.float32),
                                      np.asarray([[1, 2, 4]], dtype=np.int64), ((1, 3),), ("image-hash",))
    encoded = front.encode(prepared)
    mx.eval(encoded.inputs_embeds)
    assert encoded.token_ids == prepared.token_ids and encoded.image_hashes == ("image-hash",)
    np.testing.assert_array_equal(np.asarray(encoded.inputs_embeds[0, 0]), np.asarray(embeddings[0, 0]))
    np.testing.assert_array_equal(np.asarray(encoded.inputs_embeds[0, 3]), np.asarray(embeddings[0, 3]))
    assert np.asarray(encoded.inputs_embeds[0, 1]).tolist() == [7.0] * 4
    assert np.asarray(encoded.inputs_embeds[0, 2]).tolist() == [7.0] * 4
