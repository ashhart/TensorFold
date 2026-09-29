"""GLM image prompt expansion follows its processor grid and keeps the image tower CPU-prepared."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.glm_processing import GLMImageProcessor


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image|>": 10}.get(token)

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
        out = []
        while text:
            marker = next((x for x in ("<|begin_of_image|>", "<|image|>", "<|end_of_image|>")
                           if text.startswith(x)), None)
            if marker:
                out.append({"<|begin_of_image|>": 8, "<|image|>": 10, "<|end_of_image|>": 9}[marker])
                text = text[len(marker):]
            else:
                out.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": out}


class Processor:
    image_token = "<|image|>"

    def __init__(self, grids):
        self.tokenizer = Tokenizer()
        self.image_processor = ImageBatchProcessor(grids)
        self.grids = grids
        self.calls = []

    def replace_image_token(self, image_inputs, image_idx):
        count = int(np.prod(image_inputs["image_grid_thw"][image_idx])) // self.image_processor.merge_size**2
        return self.image_token * count


class ImageBatchProcessor:
    merge_size = 2

    def __init__(self, grids):
        self.grids, self.calls = grids, []

    def __call__(self, images, return_tensors=None, max_image_tokens=None, min_image_tokens=None):
        idx = len(self.calls)
        self.calls.append((images, return_tensors, max_image_tokens, min_image_tokens))
        grid = self.grids[idx]
        count = int(np.prod(grid))
        return {"pixel_values": np.zeros((count, 3 * 2 * 14 * 14), dtype=np.float32),
                "image_grid_thw": np.asarray([grid], dtype=np.int64)}


def image(content_hash="img", detail="auto"):
    return SimpleNamespace(content_hash=content_hash, detail=detail, to_pil=lambda: content_hash)


def test_glm_image_prompt_expands_patch_grid_and_limits_total_visual_tokens():
    processor = Processor([[1, 4, 4], [1, 2, 4]])
    front = GLMImageProcessor({"model_type": "glm5_next", "image_token_id": 10,
                              "vision_config": {"out_hidden_size": 6, "patch_size": 14,
                                                "temporal_patch_size": 2, "spatial_merge_size": 2,
                                                "hidden_size": 4, "intermediate_size": 8, "depth": 2}}, processor)
    prepared = front.prepare("question<|begin_of_image|><|image|><|end_of_image|> and "
                             "<|begin_of_image|><|image|><|end_of_image|>",
                             [image("a"), image("b", "low")], max_visual_tokens=32, max_prompt_tokens=32)
    assert prepared.token_ids.count(10) == 6
    assert prepared.visual_tokens == 6
    assert prepared.image_hashes == ("a", "b")
    assert prepared.image_grid_thw.tolist() == [[1, 4, 4], [1, 2, 4]]
    assert prepared.pixel_values.shape == (24, 3 * 2 * 14 * 14)
    assert all(not a.flags.writeable for a in (prepared.pixel_values, prepared.image_grid_thw))
    assert [call[2] for call in processor.image_processor.calls] == [16, 16]


def test_glm_image_prompt_refuses_marker_count_and_context_overflow():
    processor = Processor([[1, 4, 4]])
    front = GLMImageProcessor({"model_type": "glm5_next", "image_token_id": 10,
                                  "vision_config": {"out_hidden_size": 6, "patch_size": 14,
                                                    "temporal_patch_size": 2, "spatial_merge_size": 2}}, processor)
    with pytest.raises(ValueError, match="one image marker"):
        front.prepare("no image here", [image()])
    with pytest.raises(ValueError, match="expanded image prompt"):
        front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [image()], max_prompt_tokens=5)
