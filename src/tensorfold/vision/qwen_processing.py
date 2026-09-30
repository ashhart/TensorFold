"""CPU-only Qwen image preprocessing and rotary metadata shared by Metal and CUDA frontends."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PreparedVisionPrompt:
    token_ids: tuple[int, ...]
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    position_ids: np.ndarray
    rope_delta: int
    image_spans: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]

    @property
    def visual_tokens(self) -> int:
        return sum(end - start for start, end in self.image_spans)


def continued(prepared: PreparedVisionPrompt, tokens: Sequence[int], config: dict) -> PreparedVisionPrompt:
    """The same images for a prompt that goes on past ``prepared``'s tokens (text only), positions extended."""
    tokens = tuple(int(t) for t in tokens)
    if tokens[:len(prepared.token_ids)] != prepared.token_ids:
        raise ValueError("a continued image prompt must start with the prepared prompt's tokens")
    positions, delta, spans = image_positions(tokens, prepared.image_grid_thw, config)
    if spans != prepared.image_spans:
        raise ValueError("a continued image prompt may add text only")
    positions.setflags(write=False)
    return replace(prepared, token_ids=tokens, position_ids=positions, rope_delta=delta)


def image_positions(tokens: Sequence[int], grids: Sequence[Sequence[int]], config: dict):
    """Calculate Qwen's three image rotary axes and continuation delta without importing MLX."""
    merge = int(config["vision_config"]["spatial_merge_size"])
    image = int(config["image_token_id"])
    start_token, end_token = int(config["vision_start_token_id"]), int(config["vision_end_token_id"])
    tokens = [int(t) for t in tokens]
    if int(config.get("video_token_id", -1)) in tokens:
        raise ValueError("Video inputs are not supported by the Qwen image frontend")
    positions = np.zeros((3, 1, len(tokens)), dtype=np.int32)
    spans, cursor, next_pos = [], 0, 0
    for raw in grids:
        if len(raw) != 3:
            raise ValueError("An image grid must have temporal, height and width dimensions")
        t, h, w = (int(x) for x in raw)
        if t != 1 or min(h, w, merge) <= 0 or h % merge or w % merge:
            raise ValueError("An image grid must contain one frame and merge-aligned positive dimensions")
        try:
            begin = tokens.index(image, cursor)
        except ValueError as error:
            raise ValueError("Image grid has no matching image tokens in the prompt") from error
        h, w = h // merge, w // merge
        end = begin + h * w
        if (begin == 0 or tokens[begin - 1] != start_token or end >= len(tokens)
                or tokens[end] != end_token or tokens[begin:end] != [image] * (h * w)):
            raise ValueError("Image placeholders must match the processed image grid exactly")
        text = np.arange(next_pos, next_pos + begin - cursor, dtype=np.int32)
        positions[:, 0, cursor:begin] = text
        base = next_pos + begin - cursor
        positions[0, 0, begin:end] = base
        positions[1, 0, begin:end] = base + np.repeat(np.arange(h, dtype=np.int32), w)
        positions[2, 0, begin:end] = base + np.tile(np.arange(w, dtype=np.int32), h)
        next_pos, cursor = base + max(h, w), end
        spans.append((begin, end))
    if image in tokens[cursor:]:
        raise ValueError("The prompt contains image tokens without a corresponding image")
    positions[:, 0, cursor:] = np.arange(next_pos, next_pos + len(tokens) - cursor, dtype=np.int32)
    delta = next_pos - cursor
    return positions, delta, tuple(spans)


def _processor_options(model_dir: Path, vision: dict) -> dict:
    raw = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        path = model_dir / name
        if path.exists():
            value = json.loads(path.read_text())
            raw.update(value.get("image_processor", {}) if name == "processor_config.json" else value)
    keys = ("image_mean", "image_std", "rescale_factor", "do_rescale", "do_normalize", "do_convert_rgb",
            "min_pixels", "max_pixels")
    opts = {key: raw[key] for key in keys if key in raw}
    opts.setdefault("image_mean", [0.5] * int(vision.get("in_channels", 3)))
    opts.setdefault("image_std", [0.5] * int(vision.get("in_channels", 3)))
    size = raw.get("size") or {}
    for old, new in (("shortest_edge", "min_pixels"), ("longest_edge", "max_pixels")):
        if old in size and new not in opts:
            opts[new] = size[old]
    for key, value in (("patch_size", vision["patch_size"]), ("temporal_patch_size", vision["temporal_patch_size"]),
                       ("merge_size", vision["spatial_merge_size"])):
        if key in raw and int(raw[key]) != int(value):
            raise ValueError(f"Image processor {key} disagrees with the vision tower")
        opts[key] = int(value)
    return opts


def _processor_runtime():
    try:
        from transformers import AutoTokenizer, Qwen2VLImageProcessor
    except ImportError as error:
        raise ValueError("Qwen image preprocessing requires the optional vision dependencies") from error
    return AutoTokenizer, Qwen2VLImageProcessor


class QwenImageProcessor:
    """A local CPU image processor and tokenizer with no backend imports or model calls."""

    def __init__(self, config: dict, processor: Any, tokenizer: Any):
        self.config, self.processor, self.tokenizer = config, processor, tokenizer
        self.image_token = tokenizer.convert_ids_to_tokens(int(config["image_token_id"]))
        if not isinstance(self.image_token, str) or not self.image_token:
            raise ValueError("The local tokenizer does not define the checkpoint's image token")
        if tokenizer.convert_tokens_to_ids(self.image_token) != int(config["image_token_id"]):
            raise ValueError("The local tokenizer's image token does not match the checkpoint")

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "QwenImageProcessor":
        """Load local tokenizer files and CPU image settings without a Hub or model loader."""
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise ValueError("Image preprocessing requires a local checkpoint directory")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") not in ("qwen3_5", "qwen4_exp") or not config.get("vision_config"):
            raise ValueError("Image preprocessing currently supports Qwen3.5/3.8 dense and Flash Next multimodal checkpoints")
        AutoTokenizer, ImageProcessor = _processor_runtime()
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        processor = ImageProcessor(**_processor_options(path, config["vision_config"]))
        return cls(config, processor, tokenizer)

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, max_visual_tokens: int = 4096,
                max_prompt_tokens: int | None = None) -> PreparedVisionPrompt:
        """Expand image markers and calculate request-local rotary metadata without touching the GPU."""
        if not images or max_visual_tokens < 1:
            raise ValueError("Image preprocessing needs images and a positive visual-token budget")
        if rendered_prompt.count(self.image_token) != len(images):
            raise ValueError("The rendered prompt must contain exactly one image marker for each image")
        if len(images) > max_visual_tokens:
            raise ValueError("The image count exceeds the visual-token budget")
        vision = self.config["vision_config"]
        factor = int(vision["patch_size"]) * int(vision["spatial_merge_size"])
        limit = max_visual_tokens // len(images)
        per_image = min(int(getattr(self.processor, "max_pixels", limit * factor**2)), limit * factor**2)
        parts, grids = [], []
        for image in images:
            cap = min(per_image, 256 * factor**2) if getattr(image, "detail", "auto") == "low" else per_image
            result = self.processor(images=[image.to_pil()], max_pixels=cap,
                                    min_pixels=min(int(getattr(self.processor, "min_pixels", factor**2)), cap))
            pixels, grid = np.asarray(result["pixel_values"]), np.asarray(result["image_grid_thw"], dtype=np.int64)
            if grid.shape != (1, 3) or pixels.ndim != 2:
                raise ValueError("The image processor returned an invalid patch/grid shape")
            expected_width = (int(vision.get("in_channels", 3)) * int(vision["temporal_patch_size"])
                              * int(vision["patch_size"])**2)
            if pixels.shape != (int(np.prod(grid[0])), expected_width):
                raise ValueError("The processed image patches do not match the checkpoint's vision geometry")
            parts.append(pixels)
            grids.append(grid[0])
        grid = np.asarray(grids, dtype=np.int64)
        merge = int(vision["spatial_merge_size"])
        counts = [int(np.prod(row)) // merge**2 for row in grid]
        if sum(counts) > max_visual_tokens:
            raise ValueError("Processed images exceed the visual-token budget; reduce image resolution or count")
        text = rendered_prompt.split(self.image_token)
        expanded = text[0] + "".join(self.image_token * n + rest for n, rest in zip(counts, text[1:]))
        encoded = self.tokenizer(expanded, add_special_tokens=False, return_attention_mask=False)
        tokens = tuple(int(t) for t in encoded["input_ids"])
        if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:
            raise ValueError("The expanded image prompt exceeds the token budget; reduce image resolution or prompt length")
        positions, delta, spans = image_positions(tokens, grid, self.config)
        pixels = np.concatenate(parts, axis=0)
        for array in (pixels, grid, positions):
            array.setflags(write=False)
        return PreparedVisionPrompt(tokens, pixels, grid, positions, delta, spans,
                                    tuple(image.content_hash for image in images))

    def estimate_workspace_bytes(self, prepared: PreparedVisionPrompt) -> int:
        """The tower's measured workspace (unmeasured: every layer's activations at once) plus this request's arrays."""
        vision = self.config["vision_config"]
        patches = int(prepared.pixel_values.shape[0])
        hidden, intermediate = int(vision["hidden_size"]), int(vision["intermediate_size"])
        queued = int(getattr(self, "workspace_bytes", 0) or 0) or (
            patches * (12 * hidden + 4 * intermediate) * 4 * int(vision["depth"]))
        embeddings = len(prepared.token_ids) * int(vision["out_hidden_size"]) * 8
        return int(2 * prepared.pixel_values.nbytes + queued + embeddings + prepared.position_ids.nbytes)
