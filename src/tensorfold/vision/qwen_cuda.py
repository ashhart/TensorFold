"""Qwen image features and request-local positions for the CUDA lane engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

MAX_PATCHES = 16384
WORKSPACE_BYTES = 4 * 1024**3


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
    if not isinstance(config, dict) or config.get("model_type") not in ("qwen3_5", "qwen3_5_vision", "qwen4_exp"):
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


def _quantized_vision_modules(raw: dict, config: dict) -> dict[str, tuple[str, int]]:
    """Restituisce solo le proiezioni miste MIA Qwen3.8 esplicitamente descritte nel config."""
    block = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config") or {}
    groups = block.get("config_groups") or {}
    visual_groups = {
        name: (group, {target.removeprefix("model.visual.") for target in group.get("targets", [])
                       if isinstance(target, str) and target.startswith("model.visual.")})
        for name, group in groups.items() if isinstance(group, dict)
    }
    visual_groups = {name: (group, targets) for name, (group, targets) in visual_groups.items() if targets}
    if not visual_groups:
        return {}
    if block.get("quant_method") != "modelopt" or block.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("CUDA vision supports only the validated MIA ModelOpt mixed-precision layout")

    depth = config["depth"]
    expected_mxfp8 = {f"blocks.{i}.{part}" for i in range(depth)
                      for part in ("attn.qkv", "attn.proj", "mlp.linear_fc1")}
    expected_mxfp8.update(("merger.linear_fc1", "merger.linear_fc2"))
    expected_nvfp4 = {f"blocks.{i}.mlp.linear_fc2" for i in range(depth)}
    expected = {"group_mxfp8_vision": (expected_mxfp8, "mxfp8", 32, 8),
                "group_w4a16_nvfp4_vision_fc2": (expected_nvfp4, "nvfp4", 16, 4)}
    if set(visual_groups) != set(expected):
        raise ValueError("ModelOpt vision quantization groups differ from the validated MIA MXFP8/NVFP4 layout")

    result = {}
    for name, (targets, scheme, group_size, bits) in expected.items():
        group, actual_targets = visual_groups[name]
        weights = group.get("weights") or {}
        if (actual_targets != targets or weights.get("num_bits") != bits
                or weights.get("group_size") != group_size or weights.get("type") != "float"):
            raise ValueError(f"ModelOpt vision group {name} has unexpected targets or parameters")
        for module in targets:
            result[module] = (scheme, group_size)
    return result


def checkpoint_vision(model_dir: str | Path) -> tuple[dict, int]:
    """Validate vision tensor headers before any model or accelerator allocation."""
    from tensorfold.cuda.capacity import SIZES
    model_dir = Path(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    config = vision_config(model_dir)
    quantized = _quantized_vision_modules(raw, config)
    sources = _vision_sources(model_dir)
    tensors = {k: value[1] for k, value in sources.items()}
    for name, (path, info, begin) in sources.items():
        shape, offsets = info.get("shape", ()), info.get("data_offsets", ())
        dtype, itemsize = info.get("dtype"), SIZES.get(info.get("dtype"))
        scalar_scale = name.endswith(".weight_scale_2")
        if (itemsize is None or (not shape and not scalar_scale)
                or any(type(d) is not int or d <= 0 for d in shape) or len(offsets) != 2
                or any(type(d) is not int for d in offsets) or offsets[0] < 0
                or offsets[1] < offsets[0] or offsets[1] - offsets[0] != math.prod(shape) * itemsize
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
    header_shapes = dict(shapes)
    quantized_keys = set()
    for module, (scheme, group_size) in quantized.items():
        weight_key = f"{module}.weight"
        rows, columns = shapes[weight_key]
        if columns % group_size:
            raise ValueError(f"vision projection {module} is not divisible by its quantization group")
        if scheme == "nvfp4":
            if columns % 2:
                raise ValueError(f"NVFP4 projection {module} has an odd input width")
            header_shapes[weight_key] = [rows, columns // 2]
            header_shapes[f"{module}.weight_scale"] = [rows, columns // group_size]
            header_shapes[f"{module}.weight_scale_2"] = []
            quantized_keys.update((weight_key, f"{module}.weight_scale", f"{module}.weight_scale_2"))
        else:
            header_shapes[f"{module}.weight_scale"] = [rows, columns // group_size]
            quantized_keys.update((weight_key, f"{module}.weight_scale"))
    if set(tensors) != set(header_shapes) | {"patch_embed.proj.weight"}:
        raise ValueError("checkpoint vision tensors differ from the configured Qwen tower")
    if any(tensors[key]["shape"] != expected for key, expected in header_shapes.items()):
        raise ValueError("vision tensor shapes differ from the checkpoint configuration")
    for module, (scheme, _) in quantized.items():
        if scheme == "nvfp4":
            expected_types = {f"{module}.weight": "U8", f"{module}.weight_scale": "F8_E4M3",
                              f"{module}.weight_scale_2": "F32"}
        else:
            expected_types = {f"{module}.weight": "F8_E4M3", f"{module}.weight_scale": "U8"}
        if any(tensors[name]["dtype"] != dtype for name, dtype in expected_types.items()):
            raise ValueError(f"vision projection {module} has an unsupported quantized storage layout")
    float_keys = set(tensors) - quantized_keys
    if any(tensors[key]["dtype"] not in {"BF16", "F16", "F32"} for key in float_keys):
        raise ValueError("unquantized CUDA vision tensors must use floating-point weights")
    h, p, t, channels = (config[k] for k in ("hidden_size", "patch_size", "temporal_patch_size", "in_channels"))
    shape = tensors["patch_embed.proj.weight"]["shape"]
    if shape not in ([h, t, p, p, channels], [h, channels, t, p, p]):
        raise ValueError("unsupported vision patch convolution layout")
    shapes["patch_embed.proj.weight"] = shape
    return config, sum(math.prod(shape) * 2 for shape in shapes.values())


def weight_transform(base, enabled: bool, rank: int):
    def transform(name, info):
        from .qwen_checkpoint import vision_key

        if enabled and vision_key(name) is not None:
            if rank != 0 or os.environ.get("TENSORFOLD_VISION_WEIGHTS"):
                return 0, 0
            if name.endswith((".weight_scale", ".weight_scale_2")):
                return 0, 0
            multiplier = 4 if name.endswith(".weight") and info.get("dtype") == "U8" else 2
            return math.prod(info["shape"]) * multiplier, 0
        return base(name, info)
    return transform


def capacity_geometry(base, model_dir, enabled: bool, rank: int, workspace: int = WORKSPACE_BYTES):
    def geometry(text):
        from tensorfold.cuda.capacity import Geometry

        result = base(text)
        if not enabled:
            return result
        external_weights = 0
        if rank == 0:
            _, tower_bytes = checkpoint_vision(model_dir)
            if os.environ.get("TENSORFOLD_VISION_WEIGHTS"):
                external_weights = tower_bytes
        reserve = workspace if rank == 0 else 128 * 1024**2
        return Geometry(lambda slots: result.bytes_at(slots) + reserve + external_weights,
                        result.reserve, result.minimum_slots)
    return geometry


class QwenCudaVision:
    """Only the image tower is loaded; the CUDA family retains all language computation."""

    def __init__(self, model_dir, device, allow_urls: bool = False):
        self.allow_urls = allow_urls
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
        self.device = device
        quantized = _quantized_vision_modules(raw, self.config)
        config = Qwen3_5VisionConfig(**{k: v for k, v in self.config.items()
                                      if k not in ("model_type", "deepstack_visual_indexes")})
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            tower = Qwen3_5VisionModel(config)
        tensors = {}
        by_file = {}
        sources = _vision_sources(model_dir)
        for key, (path, info, begin) in sources.items():
            by_file.setdefault(path, {})[key] = info
        quantized_by_file = {}
        for module, metadata in quantized.items():
            keys = [f"{module}.weight", f"{module}.weight_scale"]
            if metadata[0] == "nvfp4":
                keys.append(f"{module}.weight_scale_2")
            files = {sources[name][0] for name in keys}
            if len(files) != 1:
                raise ValueError(f"quantized vision projection {module} is split across safetensors files")
            quantized_by_file.setdefault(files.pop(), []).append((module, *metadata))
        for path, selected in by_file.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                names = {vision_key(name): name for name in source.keys() if vision_key(name) is not None}
                handled = set()
                for module, scheme, group_size in quantized_by_file.get(path, ()):
                    weight_key, scale_key = f"{module}.weight", f"{module}.weight_scale"
                    weight = source.get_tensor(names[weight_key])
                    scale = source.get_tensor(names[scale_key])
                    scale_2 = source.get_tensor(names[f"{module}.weight_scale_2"]).item() if scheme == "nvfp4" else None
                    raw_weight = weight.numpy() if weight.dtype == torch.uint8 else weight.view(torch.uint8).numpy()
                    raw_scale = scale.numpy() if scale.dtype == torch.uint8 else scale.view(torch.uint8).numpy()
                    from .qwen_quantized import dequantize_linear

                    decoded = dequantize_linear(raw_weight, raw_scale, scale_2, scheme=scheme, group_size=group_size)
                    tensors[weight_key] = torch.from_numpy(decoded).to(device=device, dtype=torch.bfloat16).contiguous()
                    handled.update((weight_key, scale_key))
                    if scheme == "nvfp4":
                        handled.add(f"{module}.weight_scale_2")
                for key in selected:
                    if key in handled or key.endswith((".weight_scale", ".weight_scale_2")):
                        continue
                    value = source.get_tensor(names[key])
                    if key == "patch_embed.proj.weight" and value.shape[-1] == self.config["in_channels"]:
                        value = value.permute(0, 4, 1, 2, 3).contiguous()
                    tensors[key] = value.to(device=device, dtype=torch.bfloat16)
        tower.load_state_dict(tensors, strict=True, assign=True)
        rotary_frequencies(tower.rotary_pos_emb, self.config, device)
        self.tower = tower.eval()

    def warm(self):
        """Load tower kernels at startup using one small, merge-aligned image grid."""
        import torch

        merge = self.config["spatial_merge_size"]
        patches = merge * merge
        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2
        with torch.inference_mode():
            self.tower(torch.zeros((patches, width), dtype=torch.bfloat16, device=self.device),
                       grid_thw=torch.tensor([[1, merge, merge]], device=self.device), return_dict=True)
        torch.cuda.synchronize()

    def prepare(self, *args, **kwargs):
        return self.frontend.prepare(*args, **kwargs)

    def encode(self, prepared, prompt) -> EncodedVision:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        grid = prepared.image_grid_thw
        if len(grid.shape) != 2 or grid.shape[1] != 3 or any(int(t) != 1 for t in grid[:, 0]):
            raise ValueError("CUDA vision accepts images with one temporal grid, not video")
        merge = self.config["spatial_merge_size"]
        if any(int(value) != value or value <= 0 for row in grid for value in row) or any(
                int(h) % merge or int(w) % merge for _, h, w in grid):
            raise ValueError("image grids must contain positive merge-aligned dimensions")
        patches = sum(int(t) * int(h) * int(w) for t, h, w in grid)
        if patches <= 0 or patches > MAX_PATCHES:
            raise ValueError(f"image request exceeds the CUDA vision budget of {MAX_PATCHES} patches")
        patch_width = (self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2)
        if tuple(prepared.pixel_values.shape) != (patches, patch_width):
            raise ValueError("image patch tensor has an invalid shape")
        if tuple(prepared.position_ids.shape) != (3, 1, len(prompt)):
            raise ValueError("image positions must have shape (3, 1, prompt tokens)")
        rows = tuple(i for start, end in prepared.image_spans for i in range(start, end))
        positions = prepared.position_ids[:, 0, :].tolist()
        validate_encoded(rows, positions, prepared.rope_delta, prompt, self.image_token,
                         (patches // self.config["spatial_merge_size"]**2, self.config["out_hidden_size"]),
                         self.config["out_hidden_size"])
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            # a copy: the prepared arrays are read-only, and a tensor may not share them
            pixels = torch.tensor(prepared.pixel_values, dtype=torch.bfloat16, device=self.device)
            grids = torch.tensor(grid, dtype=torch.int64, device=self.device)
            features = self.tower(pixels, grid_thw=grids, return_dict=True).pooler_output
            features = features.to(dtype=torch.bfloat16).contiguous()
        if tuple(features.shape) != (len(rows), self.config["out_hidden_size"]):
            raise ValueError("vision tower returned a different number of image features")
        return EncodedVision(rows, features, torch.tensor(positions, dtype=torch.int32, device=self.device),
                             prepared.rope_delta)


def validate_encoded(rows, positions, delta: int, prompt, image_token: int, feature_shape, hidden: int) -> None:
    """Reject a payload that could overwrite text rows or misalign the language cache."""
    n = len(prompt)
    if list(rows) != [i for i, token in enumerate(prompt) if token == image_token]:
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
