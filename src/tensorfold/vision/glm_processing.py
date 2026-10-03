"""CPU image preparation for GLM-5.3-Flash's native GLM5-Next vision tower."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorfold.server.errors import CONTEXT_LIMIT


@dataclass(frozen=True)
class PreparedGLMVisionPrompt:
    token_ids: tuple[int, ...]
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    image_spans: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]

    @property
    def visual_tokens(self) -> int:
        return sum(end - start for start, end in self.image_spans)


class GLMImageProcessor:
    """Expand GLM image placeholders using the same MLX-VLM processor geometry as the tower."""

    image_marker = "<|image|>"

    def __init__(self, config: dict, processor: Any):
        self.config, self.processor = config, processor
        self.tokenizer = processor.tokenizer
        self.image_marker = getattr(processor, "image_token", None) or self.image_marker
        self.image_token_id = int(config["image_token_id"])
        if self.tokenizer.convert_tokens_to_ids(self.image_marker) != self.image_token_id:
            raise ValueError("The tokenizer image marker does not match the GLM vision configuration")
        vision = config["vision_config"]
        for key, expected in (("patch_size", vision["patch_size"]),
                              ("temporal_patch_size", vision["temporal_patch_size"]),
                              ("merge_size", vision["spatial_merge_size"])):
            actual = getattr(processor.image_processor, key, None)
            if actual is None or int(actual) != int(expected) or int(expected) < 1:
                raise ValueError(f"Image processor {key} disagrees with the vision tower")

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "GLMImageProcessor":
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise ValueError("GLM image preprocessing requires a local checkpoint directory")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "glm5_next" or not isinstance(config.get("vision_config"), dict):
            raise ValueError("Image preprocessing requires a complete GLM-5.3-Flash vision checkpoint")
        try:
            from mlx_vlm.models.glm5_next.processing import Glm5NextProcessor
        except ImportError:
            return cls(config, _transformers_processor(path))       # CUDA: no MLX, the same image geometry
        processor = Glm5NextProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        return cls(config, processor)

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, max_visual_tokens: int = 4096,
                max_prompt_tokens: int | None = None, max_image_tokens: int | None = None) -> PreparedGLMVisionPrompt:
        if not images or max_visual_tokens < 1:
            raise ValueError("GLM image preprocessing needs images and a positive visual-token budget")
        if rendered_prompt.count(self.image_marker) != len(images):
            raise ValueError("The rendered prompt must contain exactly one image marker for every image")
        if len(images) > max_visual_tokens:
            raise ValueError("The image count exceeds the visual-token budget")

        all_pixels, grids, counts = [], [], []
        budget = max_visual_tokens // len(images)
        if max_image_tokens:                 # each image at most this many, however few share the budget
            budget = min(budget, max_image_tokens)
        for image in images:
            cap = min(budget, 256) if getattr(image, "detail", "auto") == "low" else budget
            processed = self.processor.image_processor([image.to_pil()], return_tensors="np",
                                                       min_image_tokens=min(16, cap), max_image_tokens=cap)
            pixels = np.asarray(processed["pixel_values"])
            grid = np.asarray(processed["image_grid_thw"], dtype=np.int64)
            if grid.shape != (1, 3) or pixels.ndim != 2:
                raise ValueError("The GLM image processor returned an invalid patch or grid shape")
            vision = self.config["vision_config"]
            t, h, w = (int(n) for n in grid[0])
            merge = int(vision["spatial_merge_size"])
            if t != 1 or min(h, w) <= 0 or h % merge or w % merge:
                raise ValueError("An image grid must contain one frame and merge-aligned positive dimensions")
            width = (int(vision.get("in_channels", 3)) * int(vision["temporal_patch_size"])
                     * int(vision["patch_size"]) ** 2)
            if pixels.shape != (int(np.prod(grid[0])), width):
                raise ValueError("The processed image patches do not match the GLM vision geometry")
            count = h * w // merge**2
            if count < 1 or count > cap:
                raise ValueError("The GLM processor exceeded the per-image visual-token budget")
            all_pixels.append(pixels)
            grids.append(grid[0])
            counts.append(count)

        if sum(counts) > max_visual_tokens:
            raise ValueError("Processed images exceed the visual-token budget; reduce image resolution or count")
        parts = rendered_prompt.split(self.image_marker)
        expanded = parts[0] + "".join(self.image_marker * n + suffix for n, suffix in zip(counts, parts[1:]))
        encoded = self.tokenizer(expanded, add_special_tokens=False, return_attention_mask=False)
        token_ids = tuple(int(t) for t in encoded["input_ids"])
        if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:   # OpenAI's context_length_exceeded
            raise ValueError(f"{CONTEXT_LIMIT} {max_prompt_tokens} tokens: the expanded image prompt has "
                             f"{len(token_ids)} tokens, which exceeds the context window; reduce image resolution or "
                             "prompt length")
        spans, cursor = [], 0
        for count in counts:
            try:
                begin = token_ids.index(self.image_token_id, cursor)
            except ValueError as error:
                raise ValueError("The expanded GLM prompt is missing image placeholders") from error
            end = begin + count
            if token_ids[begin:end] != (self.image_token_id,) * count:
                raise ValueError("GLM image placeholders are not contiguous after prompt tokenization")
            spans.append((begin, end))
            cursor = end
        if token_ids.count(self.image_token_id) != sum(counts):
            raise ValueError("The prompt contains image tokens without corresponding images")
        pixels = np.concatenate(all_pixels, axis=0)
        grid = np.asarray(grids, dtype=np.int64)
        pixels.setflags(write=False)
        grid.setflags(write=False)
        return PreparedGLMVisionPrompt(token_ids, pixels, grid, tuple(spans),
                                       tuple(image.content_hash for image in images))

    def continued(self, prepared: PreparedGLMVisionPrompt, tokens: Sequence[int]) -> PreparedGLMVisionPrompt:
        """The same images for a prompt that goes on past ``prepared``'s tokens (text only): GLM has no positions."""

        tokens = tuple(int(t) for t in tokens)
        if tokens[:len(prepared.token_ids)] != prepared.token_ids:
            raise ValueError("a continued image prompt must start with the prepared prompt's tokens")
        if tokens.count(self.image_token_id) != prepared.visual_tokens:
            raise ValueError("a continued image prompt may add text only")
        return replace(prepared, token_ids=tokens)

    def estimate_workspace_bytes(self, prepared: PreparedGLMVisionPrompt) -> int:
        vision = self.config["vision_config"]
        patches = int(prepared.pixel_values.shape[0])
        hidden, intermediate = int(vision["hidden_size"]), int(vision["intermediate_size"])
        measured = int(getattr(self, "workspace_bytes", 0) or 0)
        activation = measured or patches * (12 * hidden + 4 * intermediate) * 4 * int(vision["depth"])
        embeddings = len(prepared.token_ids) * int(vision["out_hidden_size"]) * 8
        return int(2 * prepared.pixel_values.nbytes + activation + embeddings)


def _transformers_processor(path: Path) -> Any:
    """The checkpoint's tokenizer and transformers' PIL image processor: the torch-free backend MLX-VLM matches."""

    try:
        from transformers import AutoTokenizer
        from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil
    except ImportError as error:
        raise ValueError("GLM image input off MLX needs transformers 5.17 or newer and Pillow: "
                         "pip install 'tensorfold[vision]'") from error
    settings: dict = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        if (path / name).exists():
            value = json.loads((path / name).read_text())
            settings.update(value.get("image_processor", {}) if name == "processor_config.json" else value)
    settings = {k: v for k, v in settings.items() if k not in ("image_processor_type", "processor_class")}
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    return SimpleNamespace(tokenizer=tokenizer, image_processor=Glm5NextImageProcessorPil(**settings),
                           image_token=getattr(tokenizer, "image_token", None) or GLMImageProcessor.image_marker)
