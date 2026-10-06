"""A Qwen vision tower (dense or Flash Next) shares the target's embeddings and keeps preprocessing on the CPU."""

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


MODEL_TYPES = ("qwen3_5", "qwen4_exp")      # Qwen3.5/3.8 dense, and Flash Next (Qwen3-VL's tower in both)


def _runtime(model_type: str = "qwen3_5"):
    import importlib

    try:
        import mlx.core as mx
        import mlx.nn as nn

        VisionConfig = importlib.import_module(f"mlx_vlm.models.{model_type}.config").VisionConfig
        VisionModel = importlib.import_module(f"mlx_vlm.models.{model_type}.vision").VisionModel
    except ImportError as error:
        raise ValueError("Qwen image support requires the optional vision dependencies: pip install 'tensorfold[vision]'") from error
    return mx, nn, VisionConfig, VisionModel


def _evaluated_blocks(blocks: list, mx: Any, nn: Any) -> list:
    """Each block's output evaluated before the next is built: large MLX command buffers would hold every block's."""

    class Evaluated(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            out = self.inner(*args, **kwargs)
            mx.eval(out)
            return out

    return [Evaluated(block) for block in blocks]


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
        if config.get("model_type") not in MODEL_TYPES or not config.get("vision_config"):
            raise ValueError("The image frontend currently supports Qwen3.5/3.8 dense and Flash Next multimodal "
                             "checkpoints only")
        if config["vision_config"].get("deepstack_visual_indexes"):
            raise ValueError("Deepstack vision checkpoints are not supported by the Qwen image frontend")
        tensors = vision_tensors(path)
        prepared = QwenImageProcessor.from_directory(path)
        mx, nn, VisionConfig, VisionModel = _runtime(config["model_type"])
        tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
        weights = tower.sanitize(load_vision_weights(tensors, mx))
        if any(name.endswith(".scales") for name in weights):
            nn.quantize(tower, class_predicate=quantization_predicate(config, weights))
        tower.load_weights(list(weights.items()), strict=True)
        mx.eval(tower.parameters())
        if isinstance(getattr(tower, "blocks", None), list):
            tower.blocks = _evaluated_blocks(tower.blocks, mx, nn)
        tower.eval()
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
