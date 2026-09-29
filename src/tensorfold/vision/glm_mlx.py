"""The GLM-5-Next vision tower shares TensorFold's loaded GLM embeddings and language model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

from tensorfold.vision.glm_processing import GLMImageProcessor, PreparedGLMVisionPrompt
from tensorfold.vision.qwen_checkpoint import load_vision_weights, quantization_predicate, vision_tensors


@dataclass(frozen=True)
class EncodedGLMVisionPrompt:
    token_ids: tuple[int, ...]
    inputs_embeds: Any
    image_spans: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]


def _runtime():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.glm5_next.config import VisionConfig
        from mlx_vlm.models.glm5_next.vision import VisionModel
    except ImportError as error:
        raise ValueError("GLM image support requires the optional dependencies: pip install 'tensorfold[vision]'") from error
    return mx, nn, VisionConfig, VisionModel


class GLMVisionFrontend(GLMImageProcessor):
    """Load only the local GLM vision tower; the existing TensorFold language weights are reused."""

    def __init__(self, config: dict, embed_tokens: Callable, tower: Any, processor: Any, mx: Any,
                 allow_urls: bool = False):
        super().__init__(config, processor)
        self.embed_tokens, self.tower, self.mx, self.allow_urls = embed_tokens, tower, mx, allow_urls

    @classmethod
    def load(cls, model_dir: str | Path, embed_tokens: Callable, allow_urls: bool = False) -> "GLMVisionFrontend":
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise ValueError("Vision loading requires a local checkpoint directory")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "glm5_next" or not config.get("vision_config"):
            raise ValueError("GLM vision loading requires a complete GLM-5.3-Flash checkpoint")
        tensors = vision_tensors(path)
        prepared = GLMImageProcessor.from_directory(path)
        mx, nn, VisionConfig, VisionModel = _runtime()
        tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
        weights = tower.sanitize(load_vision_weights(tensors, mx))
        if any(name.endswith(".scales") for name in weights):
            nn.quantize(tower, class_predicate=quantization_predicate(config, weights))
        tower.load_weights(list(weights.items()), strict=True)
        tower.eval()
        mx.eval(tower.parameters())
        front = cls(config, embed_tokens, tower, prepared.processor, mx, allow_urls)
        front.workspace_bytes = front.measure_workspace()
        return front

    def measure_workspace(self, max_visual_tokens: int = 4096) -> int:
        mx, vision = self.mx, self.config["vision_config"]
        merge = int(vision["spatial_merge_size"])
        per_image = max_visual_tokens // 4
        side = max(1, math.isqrt(per_image)) * merge
        width = int(vision.get("in_channels", 3)) * int(vision["temporal_patch_size"]) * int(vision["patch_size"]) ** 2
        pixels = mx.zeros((4 * side * side, width), dtype=self.tower.patch_embed.proj.weight.dtype)
        grid = mx.array([[1, side, side]] * 4, dtype=mx.int32)
        mx.eval(pixels)
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        features = self.tower(pixels, grid)
        mx.eval(features)
        peak = int(mx.get_peak_memory()) - int(base)
        del features, pixels
        mx.clear_cache()
        return max(0, peak)

    def encode(self, prepared: PreparedGLMVisionPrompt) -> EncodedGLMVisionPrompt:
        mx = self.mx
        tokens = mx.array([prepared.token_ids], dtype=mx.int32)
        embeddings = self.embed_tokens(tokens).reshape(1, len(prepared.token_ids), -1)
        pixels = mx.array(prepared.pixel_values).astype(self.tower.patch_embed.proj.weight.dtype)
        features = self.tower(pixels, mx.array(prepared.image_grid_thw, dtype=mx.int32))
        if features.ndim != 2 or features.shape != (prepared.visual_tokens, embeddings.shape[-1]):
            raise ValueError("GLM vision features do not match the image placeholder count or embedding width")
        positions = [row for begin, end in prepared.image_spans for row in range(begin, end)]
        embeddings[0, mx.array(positions, dtype=mx.int32)] = features.astype(embeddings.dtype)
        return EncodedGLMVisionPrompt(prepared.token_ids, embeddings, prepared.image_spans, prepared.image_hashes)
