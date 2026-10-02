"""CPU-only Qwen image preprocessing and rotary metadata shared by Metal and CUDA frontends."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
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
    # videos (a CUDA frontend with ``videos``): patches, one (frames, height, width) grid each, one span a frame group
    video_pixel_values: np.ndarray | None = None
    video_grid_thw: np.ndarray | None = None
    video_spans: tuple[tuple[int, int], ...] = ()
    video_hashes: tuple[str, ...] = ()

    @property
    def visual_tokens(self) -> int:
        return sum(end - start for start, end in self.image_spans + self.video_spans)


def continued(prepared: PreparedVisionPrompt, tokens: Sequence[int], config: dict) -> PreparedVisionPrompt:
    """The same images for a prompt that goes on past ``prepared``'s tokens (text only), positions extended."""
    tokens = tuple(int(t) for t in tokens)
    if tokens[:len(prepared.token_ids)] != prepared.token_ids:
        raise ValueError("a continued image prompt must start with the prepared prompt's tokens")
    positions, delta, spans, frames = media_positions(tokens, prepared.image_grid_thw, prepared.video_grid_thw, config)
    if spans != prepared.image_spans or frames != prepared.video_spans:
        raise ValueError("a continued image prompt may add text only")
    positions.setflags(write=False)
    return replace(prepared, token_ids=tokens, position_ids=positions, rope_delta=delta)


def image_positions(tokens: Sequence[int], grids: Sequence[Sequence[int]], config: dict):
    """Calculate Qwen's three image rotary axes and continuation delta without importing MLX."""
    positions, delta, spans, _ = media_positions(tokens, grids, None, config)
    return positions, delta, spans


def media_positions(tokens: Sequence[int], grids: Sequence[Sequence[int]], video_grids, config: dict):
    """Qwen3.5's rope index (transformers ``get_rope_index``): text counts up on all three axes; each image, and each
    frame group of a video (its own ``<|vision_start|>`` block after its timestamp), sits at the next position with
    rows and columns on the h and w axes, and the text after it resumes past its larger side. ``video_grids`` None:
    videos are refused. Returns the positions [3, 1, n], the decode offset, image spans and frame-group spans."""
    merge = int(config["vision_config"]["spatial_merge_size"])
    image = int(config["image_token_id"])
    video = int(config.get("video_token_id", -1))
    start_token, end_token = int(config["vision_start_token_id"]), int(config["vision_end_token_id"])
    tokens = [int(t) for t in tokens]
    if video_grids is None and video in tokens:
        raise ValueError("Video inputs are not supported by the Qwen image frontend")
    frames = []
    for raw in video_grids if video_grids is not None else ():
        if len(raw) != 3:
            raise ValueError("A video grid must have temporal, height and width dimensions")
        t, h, w = (int(x) for x in raw)
        if min(t, h, w, merge) <= 0 or h % merge or w % merge:
            raise ValueError("A video grid must contain frames and merge-aligned positive dimensions")
        frames.extend([(1, h, w)] * t)                  # one block a frame group, as transformers splits the grid
    pending = {image: list(grids), video: frames}
    used = {image: 0, video: 0}
    spans = {image: [], video: []}
    positions = np.zeros((3, 1, len(tokens)), dtype=np.int32)
    cursor, next_pos, n = 0, 0, len(tokens)
    while True:
        begin = next((i for i in range(cursor, n) if tokens[i] == image or tokens[i] == video), None)
        if begin is None:
            break
        kind = tokens[begin]
        what = "image" if kind == image else "video"
        if used[kind] >= len(pending[kind]):
            raise ValueError(f"The prompt contains {what} tokens without a corresponding {what}")
        raw = pending[kind][used[kind]]
        used[kind] += 1
        if len(raw) != 3:
            raise ValueError("An image grid must have temporal, height and width dimensions")
        t, h, w = (int(x) for x in raw)
        if t != 1 or min(h, w, merge) <= 0 or h % merge or w % merge:
            raise ValueError("An image grid must contain one frame and merge-aligned positive dimensions")
        h, w = h // merge, w // merge
        end = begin + h * w
        if (begin == 0 or tokens[begin - 1] != start_token or end >= len(tokens)
                or tokens[end] != end_token or tokens[begin:end] != [kind] * (h * w)):
            raise ValueError(f"{what.capitalize()} placeholders must match the processed {what} grid exactly")
        text = np.arange(next_pos, next_pos + begin - cursor, dtype=np.int32)
        positions[:, 0, cursor:begin] = text
        base = next_pos + begin - cursor
        positions[0, 0, begin:end] = base
        positions[1, 0, begin:end] = base + np.repeat(np.arange(h, dtype=np.int32), w)
        positions[2, 0, begin:end] = base + np.tile(np.arange(w, dtype=np.int32), h)
        next_pos, cursor = base + max(h, w), end
        spans[kind].append((begin, end))
    if used[image] < len(pending[image]):
        raise ValueError("Image grid has no matching image tokens in the prompt")
    if used[video] < len(frames):
        raise ValueError("Video grid has no matching video tokens in the prompt")
    positions[:, 0, cursor:] = np.arange(next_pos, next_pos + len(tokens) - cursor, dtype=np.int32)
    delta = next_pos - cursor
    return positions, delta, tuple(spans[image]), tuple(spans[video])


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
        # video markers: the pad token and the block the chat template wraps it in
        self.video_token = self.vision_start = self.vision_end = self.video_marker = None
        if all(key in config for key in ("video_token_id", "vision_start_token_id", "vision_end_token_id")):
            self.video_token, self.vision_start, self.vision_end = (
                tokenizer.convert_ids_to_tokens(int(config[key]))
                for key in ("video_token_id", "vision_start_token_id", "vision_end_token_id"))
            self.video_marker = f"{self.vision_start}{self.video_token}{self.vision_end}"

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "QwenImageProcessor":
        """Load local tokenizer files and CPU image settings without a Hub or model loader."""
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise ValueError("Image preprocessing requires a local checkpoint directory")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") not in ("qwen3_5", "qwen4_exp") or not config.get("vision_config"):
            raise ValueError("Image preprocessing currently supports Qwen3.5/3.8 dense and Flash Next multimodal "
                             "checkpoints only")
        AutoTokenizer, ImageProcessor = _processor_runtime()
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        processor = ImageProcessor(**_processor_options(path, config["vision_config"]))
        return cls(config, processor, tokenizer)

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, videos: Sequence[Any] = (),
                max_visual_tokens: int = 4096, max_prompt_tokens: int | None = None,
                max_image_tokens: int | None = None) -> PreparedVisionPrompt:
        """Expand image (and video) markers and calculate request-local rotary metadata without touching the GPU.
        A video marker (``<|vision_start|><|video_pad|><|vision_end|>``) becomes one timestamped block a frame group,
        ``<t seconds><|vision_start|>pads<|vision_end|>``, as Qwen3-VL's processor writes it."""
        if (not images and not videos) or max_visual_tokens < 1:
            raise ValueError("Image preprocessing needs images and a positive visual-token budget")
        if rendered_prompt.count(self.image_token) != len(images):
            raise ValueError("The rendered prompt must contain exactly one image marker for each image")
        if videos and self.video_token is None:
            raise ValueError("this checkpoint's config names no video token")
        if videos and rendered_prompt.count(self.video_token) != len(videos):
            raise ValueError("The rendered prompt must contain exactly one video marker for each video")
        if len(images) > max_visual_tokens:
            raise ValueError("The image count exceeds the visual-token budget")
        vision = self.config["vision_config"]
        factor = int(vision["patch_size"]) * int(vision["spatial_merge_size"])
        merge = int(vision["spatial_merge_size"])
        expected_width = (int(vision.get("in_channels", 3)) * int(vision["temporal_patch_size"])
                          * int(vision["patch_size"])**2)
        parts, grids = [], []
        if images:
            limit = max_visual_tokens // len(images)
            if max_image_tokens:             # each image at most this many, however few share the budget
                limit = min(limit, max_image_tokens)
            per_image = min(int(getattr(self.processor, "max_pixels", limit * factor**2)), limit * factor**2)
        for image in images:
            cap = min(per_image, 256 * factor**2) if getattr(image, "detail", "auto") == "low" else per_image
            result = self.processor(images=[image.to_pil()], max_pixels=cap,
                                    min_pixels=min(int(getattr(self.processor, "min_pixels", factor**2)), cap))
            pixels, grid = np.asarray(result["pixel_values"]), np.asarray(result["image_grid_thw"], dtype=np.int64)
            if grid.shape != (1, 3) or pixels.ndim != 2:
                raise ValueError("The image processor returned an invalid patch/grid shape")
            if pixels.shape != (int(np.prod(grid[0])), expected_width):
                raise ValueError("The processed image patches do not match the checkpoint's vision geometry")
            parts.append(pixels)
            grids.append(grid[0])
        grid = np.asarray(grids, dtype=np.int64).reshape(-1, 3)
        counts = [int(np.prod(row)) // merge**2 for row in grid]
        if sum(counts) > max_visual_tokens:
            raise ValueError("Processed images exceed the visual-token budget; reduce image resolution or count")
        clips = [self._video_patches(video) for video in videos]
        markers = [self.image_token] + ([self.video_marker, self.video_token] if videos else [])
        pieces = re.split("(" + "|".join(re.escape(m) for m in markers) + ")", rendered_prompt)
        image_counts, clip_iter, expanded = iter(counts), iter(zip(videos, clips)), []
        for piece in pieces:
            if piece == self.image_token:
                expanded.append(self.image_token * next(image_counts))
            elif piece in (self.video_marker, self.video_token):
                video, (_, video_grid) = next(clip_iter)
                seqlen = int(video_grid[1] * video_grid[2]) // merge**2
                expanded.append("".join(f"<{t:.1f} seconds>{self.vision_start}{self.video_token * seqlen}"
                                        f"{self.vision_end}"
                                        for t in video.timestamps(int(vision["temporal_patch_size"]))))
            else:
                expanded.append(piece)
        encoded = self.tokenizer("".join(expanded), add_special_tokens=False, return_attention_mask=False)
        tokens = tuple(int(t) for t in encoded["input_ids"])
        if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:
            raise ValueError("The expanded image prompt exceeds the token budget; reduce image resolution or prompt length")
        video_grid = np.asarray([g for _, g in clips], dtype=np.int64).reshape(-1, 3) if videos else None
        positions, delta, spans, frames = media_positions(tokens, grid, video_grid, self.config)
        pixels = (np.concatenate(parts, axis=0) if parts else np.zeros((0, expected_width), dtype=np.float32))
        video_pixels = np.concatenate([p for p, _ in clips], axis=0) if videos else None
        for array in (pixels, grid, positions, video_pixels, video_grid):
            if array is not None:
                array.setflags(write=False)
        return PreparedVisionPrompt(tokens, pixels, grid, positions, delta, spans,
                                    tuple(image.content_hash for image in images), video_pixels, video_grid, frames,
                                    tuple(video.content_hash for video in videos))

    def video_size(self, frames: int, height: int, width: int) -> tuple[int, int]:
        """A video's frame size for the tower: Qwen3-VL's ``smart_resize`` with its per-frame cap (at most 768 tokens a
        frame group, at least ~134), the whole video within ``TENSORFOLD_VIDEO_TOKENS`` (16,384) tokens."""
        import math
        import os

        vision = self.config["vision_config"]
        factor = int(vision["patch_size"]) * int(vision["spatial_merge_size"])
        temporal = int(vision["temporal_patch_size"])
        budget = int(os.environ.get("TENSORFOLD_VIDEO_TOKENS") or 16384)
        shortest, longest = 128 * factor**2, budget * temporal * factor**2
        per_frame = max(min(768 * factor**2, longest // max(1, frames)), int(shortest * 1.05))
        max_pixels = per_frame * frames
        if frames < temporal:
            raise ValueError(f"a video needs at least {temporal} frames")
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height, width = int(height * scale), int(width * scale)
        if max(height, width) / min(height, width) > 200:
            raise ValueError("a video's aspect ratio must be under 200")
        h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
        t_bar = round(frames / temporal) * temporal
        if t_bar * h_bar * w_bar > max_pixels:
            beta = math.sqrt((frames * height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif t_bar * h_bar * w_bar < shortest:
            beta = math.sqrt(shortest / (frames * height * width))
            h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def _video_patches(self, video) -> tuple[np.ndarray, np.ndarray]:
        """Normalized patches [groups * h * w, C * T * p * p] and the (groups, h, w) grid of one decoded video, in the
        tower's order (Qwen3-VL's ``patchify``; an odd last frame repeats)."""
        vision = self.config["vision_config"]
        p, merge, temporal = (int(vision[k]) for k in ("patch_size", "spatial_merge_size", "temporal_patch_size"))
        frames = np.asarray(video.frames)
        count, height, width, channels = frames.shape
        if height % (p * merge) or width % (p * merge) or channels != int(vision.get("in_channels", 3)):
            raise ValueError("decoded video frames do not match the tower's patch grid")
        if count % temporal:
            frames = np.concatenate([frames, np.repeat(frames[-1:], temporal - count % temporal, axis=0)])
        options = self.processor
        mean = np.asarray(getattr(options, "image_mean", [0.5] * channels), dtype=np.float32)
        std = np.asarray(getattr(options, "image_std", [0.5] * channels), dtype=np.float32)
        scale = np.float32(getattr(options, "rescale_factor", 1 / 255))
        x = ((frames.astype(np.float32) * scale - mean) / std).transpose(0, 3, 1, 2)       # [T, C, H, W]
        gt, gh, gw = x.shape[0] // temporal, height // p, width // p
        x = x.reshape(gt, temporal, channels, gh // merge, merge, p, gw // merge, merge, p)
        x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return np.ascontiguousarray(x.reshape(gt * gh * gw, channels * temporal * p * p)), np.asarray([gt, gh, gw])
    def estimate_workspace_bytes(self, prepared: PreparedVisionPrompt) -> int:
        """The tower's measured workspace (unmeasured: every layer's activations at once) plus this request's arrays."""
        vision = self.config["vision_config"]
        patches = int(prepared.pixel_values.shape[0])
        hidden, intermediate = int(vision["hidden_size"]), int(vision["intermediate_size"])
        queued = int(getattr(self, "workspace_bytes", 0) or 0) or (
            patches * (12 * hidden + 4 * intermediate) * 4 * int(vision["depth"]))
        embeddings = len(prepared.token_ids) * int(vision["out_hidden_size"]) * 8
        return int(2 * prepared.pixel_values.nbytes + queued + embeddings + prepared.position_ids.nbytes)
