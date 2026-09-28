"""A Qwen dense vision tower shares the existing target embeddings and keeps preprocessing on the CPU."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable


from .qwen_checkpoint import load_vision_weights, quantization_predicate, vision_tensors
from .qwen_processing import PreparedVisionPrompt, QwenImageProcessor


@dataclass(frozen=True)
class EncodedVisionPrompt:
    token_ids: tuple[int, ...]
    inputs_embeds: Any
    position_ids: Any
    rope_delta: int
    image_spans: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]


def _runtime():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        from mlx_vlm.models.qwen3_5.vision import VisionModel
    except ImportError as error:
        raise ValueError("Qwen image support requires the optional vision dependencies: pip install 'tensorfold[vision]'") from error
    return mx, nn, VisionConfig, VisionModel


class QwenVisionFrontend(QwenImageProcessor):
    """Call load and encode on the model worker; prepare uses only the local tokenizer and NumPy image processor."""

    def __init__(self, config: dict, embed_tokens: Callable, tower: Any, processor: Any, tokenizer: Any, mx: Any,
                 allow_urls: bool = False):
        super().__init__(config, processor, tokenizer)
        self.embed_tokens, self.tower, self.mx, self.allow_urls = embed_tokens, tower, mx, allow_urls

    @classmethod
    def load(cls, model_dir: str | Path, embed_tokens: Callable, allow_urls: bool = False) -> "QwenVisionFrontend":
        """Instantiate and load only the vision tower from local files, sharing the target's embedding callable."""
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise ValueError("Vision loading requires a local checkpoint directory")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "qwen3_5" or not config.get("vision_config"):
            raise ValueError("The image frontend currently supports Qwen3.5/3.8 dense multimodal checkpoints only")
        if config["vision_config"].get("deepstack_visual_indexes"):
            raise ValueError("Deepstack vision checkpoints are not supported by the Qwen dense image frontend")
        tensors = vision_tensors(path)
        prepared = QwenImageProcessor.from_directory(path)
        mx, nn, VisionConfig, VisionModel = _runtime()
        tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
        weights = tower.sanitize(load_vision_weights(tensors, mx))
        if any(name.endswith(".scales") for name in weights):
            nn.quantize(tower, class_predicate=quantization_predicate(config, weights))
        tower.load_weights(list(weights.items()), strict=True)
        tower.eval()
        mx.eval(tower.parameters())
        front = cls(config, embed_tokens, tower, prepared.processor, prepared.tokenizer, mx, allow_urls)
        front.workspace_bytes = front.measure_workspace()
        return front

    def measure_workspace(self, max_visual_tokens: int = 4096) -> int:
        """The tower's peak workspace on the largest request admitted: four images sharing the visual-token budget."""
        mx, vision = self.mx, self.config["vision_config"]
        merge = int(vision["spatial_merge_size"])
        side = max(1, math.isqrt(max_visual_tokens // 4)) * merge
        width = int(vision.get("in_channels", 3)) * int(vision["temporal_patch_size"]) * int(vision["patch_size"]) ** 2
        pixels = mx.zeros((4 * side * side, width), dtype=self.tower.patch_embed.proj.weight.dtype)
        grid = mx.array([[1, side, side]] * 4, dtype=mx.int32)
        mx.eval(pixels)
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        features, _ = self.tower(pixels, grid)
        mx.eval(features)
        peak = int(mx.get_peak_memory()) - int(base)
        del features, pixels
        mx.clear_cache()
        return max(0, peak)

    def encode(self, prepared: PreparedVisionPrompt) -> EncodedVisionPrompt:
        """Replace image token embeddings on the model worker and leave the target language model untouched."""
        mx = self.mx
        tokens = mx.array([prepared.token_ids], dtype=mx.int32)
        embeddings = self.embed_tokens(tokens)
        pixels = mx.array(prepared.pixel_values).astype(self.tower.patch_embed.proj.weight.dtype)
        features, deepstack = self.tower(pixels, mx.array(prepared.image_grid_thw, dtype=mx.int32))
        if deepstack is not None and len(deepstack):
            raise ValueError("The loaded vision tower requires unsupported deepstack language inputs")
        if features.ndim != 2 or features.shape != (prepared.visual_tokens, embeddings.shape[-1]):
            raise ValueError("Vision features do not match the image token count or target embedding width")
        rows = [row for start, end in prepared.image_spans for row in range(start, end)]
        embeddings[0, mx.array(rows, dtype=mx.int32)] = features.astype(embeddings.dtype)
        return EncodedVisionPrompt(prepared.token_ids, embeddings, mx.array(prepared.position_ids, dtype=mx.int32),
                                   prepared.rope_delta, prepared.image_spans, prepared.image_hashes)
