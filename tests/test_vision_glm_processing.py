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
    patch_size = 14
    temporal_patch_size = 2
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


CONFIG = {"model_type": "glm5_next", "image_token_id": 10,
          "vision_config": {"out_hidden_size": 6, "patch_size": 14, "temporal_patch_size": 2,
                            "spatial_merge_size": 2, "hidden_size": 4, "intermediate_size": 8, "depth": 2}}


@pytest.mark.parametrize(("setting", "value"), [("patch_size", 7), ("temporal_patch_size", 1), ("merge_size", 4)])
def test_glm_image_processor_rejects_geometry_that_disagrees_with_the_tower(setting, value):
    processor = Processor([[1, 4, 4]])
    setattr(processor.image_processor, setting, value)
    with pytest.raises(ValueError, match="disagrees with the vision tower"):
        GLMImageProcessor(CONFIG, processor)


@pytest.mark.parametrize("grid", [[2, 4, 4], [1, 3, 4], [1, -4, -4]])
def test_glm_image_prompt_rejects_non_image_or_unaligned_grids(grid):
    front = GLMImageProcessor(CONFIG, Processor([grid]))
    with pytest.raises(ValueError, match="one frame and merge-aligned positive dimensions"):
        front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [image()])


@pytest.mark.parametrize("budget", [1, 4, 15])
def test_glm_image_prompt_respects_budgets_smaller_than_sixteen(budget):
    processing = pytest.importorskip("mlx_vlm.models.glm5_next.processing")
    pil = pytest.importorskip("PIL.Image")
    processor = SimpleNamespace(tokenizer=Tokenizer(), image_token="<|image|>",
                                image_processor=processing.Glm5NextImageProcessor())
    front = GLMImageProcessor(CONFIG, processor)
    source = SimpleNamespace(content_hash="small-image", detail="auto",
                             to_pil=lambda: pil.new("RGB", (28, 28)))
    prepared = front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [source],
                             max_visual_tokens=budget)
    assert 1 <= prepared.visual_tokens <= budget


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
