"""Qwen image features and request-local positions for the CUDA lane engine."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
import math
import os
from pathlib import Path
from typing import Any

MAX_PATCHES = 16384                  # one tower call's patches: a 4,096-token image's, the scratch reserved
MAX_REQUEST_PATCHES = 16 * MAX_PATCHES   # a request's (--vision-image-tokens 65536), encoded MAX_PATCHES at a time
MAX_VIDEO_PATCHES = 16 * MAX_PATCHES     # a request's video patches (~65k tokens), encoded MAX_PATCHES at a time
TOKENS_PER_IMAGE = 4096              # one image's visual tokens, whatever budget the request's images share
WORKSPACE_BYTES = 4 * 1024**3
# --vision-offload: the tower (about 0.9 GiB) visits the GPU per image, so the budget keeps room for it plus the
# activations of the largest accepted image (measured peak about 1.2 GiB over idle on a 4,096-token image)
OFFLOAD_WORKSPACE_BYTES = int(2.25 * 1024**3)


def rotary_frequencies(rotary: Any, config: dict, device: Any) -> None:
    """Fill the frequency buffers a meta-device build leaves empty, as every transformers version computes them."""
    import torch

    dim = int(config["hidden_size"]) // int(config["num_heads"]) // 2
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10000.0))
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    for name in ("inv_freq", "original_inv_freq"):
        if name in rotary._buffers:
            rotary._buffers[name] = inv_freq.clone()


@dataclass
class EncodedVision:
    rows: tuple[int, ...]
    features: Any
    positions: Any
    rope_delta: int


def vision_config(model_dir: str | Path) -> dict:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = raw.get("vision_config")
    if not isinstance(config, dict) or config.get("model_type") not in ("qwen3_5", "qwen4_exp"):
        raise ValueError("CUDA vision requires a Qwen3.5-compatible vision checkpoint")
    if config.get("deepstack_visual_indexes"):
        raise ValueError("CUDA Qwen vision does not support deepstack image features")
    for key in ("hidden_size", "out_hidden_size", "depth", "patch_size", "temporal_patch_size",
                "spatial_merge_size", "in_channels", "intermediate_size", "num_heads", "num_position_embeddings"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"invalid vision configuration: {key}")
    if config["hidden_size"] % config["num_heads"] or config["hidden_size"] // config["num_heads"] % 4:
        raise ValueError("vision attention requires head widths divisible by four")
    if math.isqrt(config["num_position_embeddings"]) ** 2 != config["num_position_embeddings"]:
        raise ValueError("vision position embeddings must form a square grid")
    text = raw.get("text_config", raw)
    if config["out_hidden_size"] != text.get("hidden_size"):
        raise ValueError("vision output width differs from the language embedding width")
    from .rotary import frequency_axes

    rope = text.get("rope_parameters") or {}
    if not rope.get("mrope_interleaved", False):
        raise ValueError("CUDA Qwen vision requires interleaved multimodal rotary positions")
    dims = int(int(text["head_dim"]) * float(rope.get("partial_rotary_factor", 0.25)))
    frequency_axes(dims, rope.get("mrope_section", (11, 11, 10)))
    return config


def _vision_sources(model_dir):
    from .qwen_checkpoint import vision_tensors

    override = os.environ.get("TENSORFOLD_VISION_WEIGHTS")
    return vision_tensors(Path(model_dir), weights_path=Path(override) if override else None)


def checkpoint_vision(model_dir: str | Path) -> tuple[dict, int]:
    """Validate vision tensor headers before any model or accelerator allocation."""
    from tensorfold.cuda.capacity import SIZES
    config = vision_config(model_dir)
    sources = _vision_sources(model_dir)
    tensors = {k: value[1] for k, value in sources.items()}
    for name, (path, info, begin) in sources.items():
        shape, offsets = info.get("shape", ()), info.get("data_offsets", ())
        if (info.get("dtype") not in {"BF16", "F16", "F32"} or not shape
                or any(type(d) is not int or d <= 0 for d in shape) or len(offsets) != 2
                or any(type(d) is not int for d in offsets) or offsets[0] < 0
                or offsets[1] - offsets[0] != math.prod(shape) * SIZES[info["dtype"]]
                or begin + offsets[1] > path.stat().st_size):
            raise ValueError(f"invalid or unsupported vision tensor range: {name}")
    h, mid, merged, out = (config["hidden_size"], config["intermediate_size"],
                           config["hidden_size"] * config["spatial_merge_size"]**2, config["out_hidden_size"])
    shapes = {"patch_embed.proj.bias": [h], "pos_embed.weight": [config["num_position_embeddings"], h],
              "merger.norm.weight": [h], "merger.norm.bias": [h], "merger.linear_fc1.weight": [merged, merged],
              "merger.linear_fc1.bias": [merged], "merger.linear_fc2.weight": [out, merged],
              "merger.linear_fc2.bias": [out]}
    for layer in range(config["depth"]):
        for part, width, inputs in (("norm1", h, None), ("norm2", h, None), ("attn.qkv", 3 * h, h),
                                    ("attn.proj", h, h), ("mlp.linear_fc1", mid, h), ("mlp.linear_fc2", h, mid)):
            shapes[f"blocks.{layer}.{part}.weight"] = [width] if inputs is None else [width, inputs]
            shapes[f"blocks.{layer}.{part}.bias"] = [width]
    if set(tensors) != set(shapes) | {"patch_embed.proj.weight"}:
        raise ValueError("checkpoint needs the complete unquantized Qwen vision tower; use its original MLX checkpoint")
    if any(tensors[key]["shape"] != expected for key, expected in shapes.items()):
        raise ValueError("vision tensor shapes differ from the checkpoint configuration")
    if any(v["dtype"] not in {"BF16", "F16", "F32"} for v in tensors.values()):
        raise ValueError("CUDA vision requires floating-point vision weights")
    h, p, t, channels = (config[k] for k in ("hidden_size", "patch_size", "temporal_patch_size", "in_channels"))
    shape = tensors["patch_embed.proj.weight"]["shape"]
    if shape not in ([h, t, p, p, channels], [h, channels, t, p, p]):
        raise ValueError("unsupported vision patch convolution layout")
    return config, sum(math.prod(v["shape"]) * max(2, SIZES[v["dtype"]]) for v in tensors.values())


def weight_transform(base, enabled: bool, rank: int, offload: bool = False):
    def transform(name, info):
        from .qwen_checkpoint import vision_key

        if enabled and vision_key(name) is not None:
            if rank != 0 or offload or os.environ.get("TENSORFOLD_VISION_WEIGHTS"):
                return 0, 0                     # offloaded: resident in host RAM between images
            from tensorfold.cuda.capacity import SIZES

            return math.prod(info["shape"]) * max(2, SIZES[info["dtype"]]), 0
        return base(name, info)
    return transform


def capacity_geometry(base, model_dir, enabled: bool, rank: int, workspace: int = WORKSPACE_BYTES,
                      offload: bool = False):
    def geometry(text):
        from tensorfold.cuda.capacity import Geometry

        result = base(text)
        if not enabled:
            return result
        external_weights = 0
        if rank == 0:
            _, tower_bytes = checkpoint_vision(model_dir)
            if os.environ.get("TENSORFOLD_VISION_WEIGHTS") and not offload:
                external_weights = tower_bytes
        reserve = (OFFLOAD_WORKSPACE_BYTES if offload else workspace) if rank == 0 else 128 * 1024**2
        return Geometry(lambda slots: result.bytes_at(slots) + reserve + external_weights,
                        result.reserve, result.minimum_slots)
    return geometry


class QwenCudaVision:
    """Only the image tower is loaded; the CUDA family retains all language computation."""

    def __init__(self, model_dir, device, allow_urls: bool = False, offload: bool = False):
        self.allow_urls = allow_urls
        self.offload = offload
        self._lock = threading.Lock()   # one image on the GPU at a time when the tower is offloaded
        resident = "cpu" if offload else device
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from safetensors import safe_open
        from .qwen_processing import QwenImageProcessor
        from .qwen_checkpoint import vision_key

        self.config, self.weight_bytes = checkpoint_vision(model_dir)
        self.frontend = QwenImageProcessor.from_directory(model_dir)
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        self.image_token = int(raw["image_token_id"])
        # videos: Flash Next's frontend (the frame groups ride the image path; tested on that checkpoint)
        self.videos = raw.get("model_type") == "qwen4_exp" and "video_token_id" in raw
        self.media_tokens = frozenset({self.image_token} | ({int(raw["video_token_id"])} if self.videos else set()))
        self.device = device
        config = Qwen3_5VisionConfig(**{k: v for k, v in self.config.items()
                                      if k not in ("model_type", "deepstack_visual_indexes")})
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            tower = Qwen3_5VisionModel(config)
        tensors = {}
        by_file = {}
        for key, (path, info, begin) in _vision_sources(model_dir).items():
            by_file.setdefault(path, {})[key] = info
        for path, selected in by_file.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                names = {vision_key(name): name for name in source.keys()}
                for key in selected:
                    value = source.get_tensor(names[key])
                    if key == "patch_embed.proj.weight" and value.shape[-1] == self.config["in_channels"]:
                        value = value.permute(0, 4, 1, 2, 3).contiguous()
                    tensors[key] = value.to(device=resident, dtype=torch.bfloat16)
        tower.load_state_dict(tensors, strict=True, assign=True)
        rotary_frequencies(tower.rotary_pos_emb, self.config, resident)
        self.tower = tower.eval()

    def warm(self):
        """Load tower kernels at startup using one small, merge-aligned image grid."""
        import torch

        merge = self.config["spatial_merge_size"]
        patches = merge * merge
        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2
        with self._on_gpu(), torch.inference_mode():
            self.tower(torch.zeros((patches, width), dtype=torch.bfloat16, device=self.device),
                       grid_thw=torch.tensor([[1, merge, merge]], device=self.device), return_dict=True)
            torch.cuda.synchronize()

    def prepare(self, *args, **kwargs):
        kwargs.setdefault("max_image_tokens", TOKENS_PER_IMAGE)
        return self.frontend.prepare(*args, **kwargs)

    def video_size(self, frames: int, height: int, width: int) -> tuple[int, int]:
        return self.frontend.video_size(frames, height, width)
    @contextmanager
    def _on_gpu(self):
        """The tower on the GPU for the block; when offloaded, one caller at a time, and back to host RAM after."""
        if not self.offload:
            yield
            return
        import torch

        with self._lock:
            try:
                self.tower.to(self.device)
                yield
            finally:
                self.tower.to("cpu")
                torch.cuda.empty_cache()

    def encode(self, prepared, prompt) -> EncodedVision:
        with self._on_gpu():
            return self._encode(prepared, prompt)

    def _encode(self, prepared, prompt) -> EncodedVision:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        grid = prepared.image_grid_thw
        videos = getattr(prepared, "video_grid_thw", None)
        if videos is not None and not self.videos:
            raise ValueError("this server's vision frontend encodes images only")
        if len(grid.shape) != 2 or grid.shape[1] != 3 or any(int(t) != 1 for t in grid[:, 0]):
            raise ValueError("CUDA vision accepts images with one temporal grid, not video")
        merge = self.config["spatial_merge_size"]
        every = list(grid) + ([] if videos is None else list(videos))
        if any(int(value) != value or value <= 0 for row in every for value in row) or any(
                int(h) % merge or int(w) % merge for _, h, w in every):
            raise ValueError("image grids must contain positive merge-aligned dimensions")
        sizes = [int(t) * int(h) * int(w) for t, h, w in grid]
        patches = sum(sizes)
        clips = 0 if videos is None else sum(int(t) * int(h) * int(w) for t, h, w in videos)
        if patches > MAX_REQUEST_PATCHES or max(sizes, default=0) > MAX_PATCHES or (patches <= 0 and clips <= 0):
            raise ValueError(f"image request exceeds the CUDA vision budget of {MAX_PATCHES} patches an image and "
                             f"{MAX_REQUEST_PATCHES} a request")
        if clips > MAX_VIDEO_PATCHES:
            raise ValueError(f"video request exceeds the CUDA vision budget of {MAX_VIDEO_PATCHES} patches")
        patch_width = (self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2)
        if tuple(prepared.pixel_values.shape) != (patches, patch_width) or (
                videos is not None and tuple(prepared.video_pixel_values.shape) != (clips, patch_width)):
            raise ValueError("image patch tensor has an invalid shape")
        if tuple(prepared.position_ids.shape) != (3, 1, len(prompt)):
            raise ValueError("image positions must have shape (3, 1, prompt tokens)")
        frames = tuple(getattr(prepared, "video_spans", ()))
        spans = sorted(tuple(prepared.image_spans) + frames)
        rows = tuple(i for start, end in spans for i in range(start, end))
        positions = prepared.position_ids[:, 0, :].tolist()
        validate_encoded(rows, positions, prepared.rope_delta, prompt, self.media_tokens,
                         ((patches + clips) // merge**2, self.config["out_hidden_size"]),
                         self.config["out_hidden_size"])
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            blocks = {}                                  # span start -> its features, in the tower's order
            starts = iter(start for start, _ in prepared.image_spans)
            # images never attend to one another: runs of whole images, at most MAX_PATCHES a tower call (one call,
            # as before, whenever the request fits it), so the scratch stays what one full-size image needs
            for begin, end in image_runs(sizes, MAX_PATCHES):
                done = sum(sizes[:begin])
                # a copy: the prepared arrays are read-only, and a tensor may not share them
                pixels = torch.tensor(prepared.pixel_values[done:done + sum(sizes[begin:end])],
                                      dtype=torch.bfloat16, device=self.device)
                grids = torch.tensor(grid[begin:end], dtype=torch.int64, device=self.device)
                features = self.tower(pixels, grid_thw=grids, return_dict=True).pooler_output
                for part in features.to(dtype=torch.bfloat16).split([s // merge**2 for s in sizes[begin:end]]):
                    blocks[next(starts)] = part
            if clips:
                # frame groups never attend to one another either: a video encodes a bounded run of them at a time
                done, spans_left = 0, iter(frames)
                for t, h, w in videos:
                    t, h, w = int(t), int(h), int(w)
                    step = max(1, MAX_PATCHES // (h * w))
                    for g in range(0, t, step):
                        n = min(step, t - g)
                        pixels = torch.tensor(prepared.video_pixel_values[done:done + n * h * w],
                                              dtype=torch.bfloat16, device=self.device)
                        grids = torch.tensor([[n, h, w]], dtype=torch.int64, device=self.device)
                        features = self.tower(pixels, grid_thw=grids, return_dict=True).pooler_output
                        for part in features.to(dtype=torch.bfloat16).split(h * w // merge**2):
                            start, end = next(spans_left)
                            if end - start != part.shape[0]:
                                raise ValueError("video frame features do not match their placeholders")
                            blocks[start] = part
                        done += n * h * w
            features = torch.cat([blocks[start] for start, _ in spans]).contiguous()
        if tuple(features.shape) != (len(rows), self.config["out_hidden_size"]):
            raise ValueError("vision tower returned a different number of image features")
        return EncodedVision(rows, features, torch.tensor(positions, dtype=torch.int32, device=self.device),
                             prepared.rope_delta)


def image_runs(sizes, limit: int) -> list[tuple[int, int]]:
    """Consecutive [begin, end) runs of images whose patches fit ``limit`` together (an image alone always fits)."""
    runs, begin, total = [], 0, 0
    for i, size in enumerate(sizes):
        if i > begin and total + size > limit:
            runs.append((begin, i))
            begin, total = i, 0
        total += size
    if sizes:
        runs.append((begin, len(sizes)))
    return runs


def validate_encoded(rows, positions, delta: int, prompt, image_token, feature_shape, hidden: int) -> None:
    """Reject a payload that could overwrite text rows or misalign the language cache; ``image_token``: the
    placeholder id, or the set of them (images and videos)."""
    n = len(prompt)
    media = image_token if isinstance(image_token, (set, frozenset, tuple)) else {image_token}
    if list(rows) != [i for i, token in enumerate(prompt) if token in media]:
        raise ValueError("vision feature rows must match every image placeholder exactly")
    if not rows or tuple(feature_shape) != (len(rows), hidden):
        raise ValueError("vision feature count or width differs from the image placeholders")
    if len(positions) != 3 or any(len(axis) != n for axis in positions):
        raise ValueError("vision positions must contain three coordinates for every prompt token")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= 2**31 - 1
           for axis in positions for v in axis):
        raise ValueError("vision positions must be nonnegative int32 coordinates")
    if not isinstance(delta, int) or isinstance(delta, bool) or max(map(max, positions)) + 1 - n != delta:
        raise ValueError("vision decode offset does not follow its prompt positions")


def broadcast_encoded(payload: EncodedVision | None, rank: int, device, *, hidden: int,
                      prompt_length: int) -> EncodedVision | None:
    """Both ranks receive identical features; the vision tower exists only on rank zero."""
    import torch
    import torch.distributed as dist
    from tensorfold.families.qwen3_5.cuda.decode_tp import _share

    meta = _share(([0] if payload is None else [1, payload.rope_delta, *payload.rows]) if rank == 0 else None,
                  rank, device)
    if meta == [0]:
        return None
    if len(meta) < 3 or meta[0] != 1 or len(meta) - 2 > prompt_length:
        raise ValueError("invalid distributed vision metadata")
    rows = tuple(meta[2:])
    if sorted(set(rows)) != list(rows) or rows[0] < 0 or rows[-1] >= prompt_length:
        raise ValueError("distributed vision row indices are outside the prompt")
    features = payload.features if rank == 0 else torch.empty((len(rows), hidden), dtype=torch.bfloat16, device=device)
    positions = payload.positions if rank == 0 else torch.empty((3, prompt_length), dtype=torch.int32, device=device)
    dist.broadcast(features, 0)
    dist.broadcast(positions, 0)
    return EncodedVision(rows, features, positions, meta[1])


def replace_rows(x, payload: EncodedVision, start: int, end: int):
    import torch

    selected = [(i, row - start) for i, row in enumerate(payload.rows) if start <= row < end]
    if selected:
        source, target = zip(*selected)
        source = torch.tensor(source, dtype=torch.int64, device=x.device)
        target = torch.tensor(target, dtype=torch.int64, device=x.device)
        x.index_copy_(0, target, payload.features.index_select(0, source))
    return x
