"""GLM-5.3-Flash image features for the two-rank CUDA engine; its NoPE language layers take no image positions."""

from __future__ import annotations

import json
import math
import threading
from contextlib import contextmanager
from pathlib import Path

from tensorfold.cuda.capacity import SIZES

from .qwen_cuda import (MAX_PATCHES, MAX_REQUEST_PATCHES, TOKENS_PER_IMAGE, WORKSPACE_BYTES, EncodedVision,
                        float_headers, image_runs)

# --vision-offload: the tower visits the GPU per image, so rank 0's budget keeps room for its bytes plus the
# activations of the largest accepted image. Qwen's tower measured 1.35 GiB over its weights on a 4,096-token image;
# GLM's has not been measured, so this takes 2 GiB.
OFFLOAD_ACTIVATION_BYTES = 2 * 1024**3


def vision_config(model_dir: str | Path) -> dict:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = raw.get("vision_config")
    if raw.get("model_type") != "glm5_next" or not isinstance(config, dict):
        raise ValueError("CUDA GLM vision requires a GLM-5.3-Flash checkpoint with its vision_config")
    for key in ("hidden_size", "out_hidden_size", "depth", "patch_size", "temporal_patch_size", "spatial_merge_size",
                "in_channels", "intermediate_size", "num_heads", "projection_intermediate_size"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"invalid vision configuration: {key}")
    if config["hidden_size"] % config["num_heads"] or config["hidden_size"] // config["num_heads"] % 4:
        raise ValueError("vision attention requires head widths divisible by four")
    if config["out_hidden_size"] != (raw.get("text_config") or raw).get("hidden_size"):
        raise ValueError("vision output width differs from the language embedding width")
    if not isinstance(raw.get("image_token_id"), int):
        raise ValueError("this GLM checkpoint is missing its image token")
    return config


def tower_shapes(config: dict) -> dict[str, list[int]]:
    """Every tensor of the GLM5-Next tower in the checkpoint's (PyTorch) layout."""

    h, mid, out, inner = (config[k] for k in ("hidden_size", "intermediate_size", "out_hidden_size",
                                              "projection_intermediate_size"))
    merge, head = config["spatial_merge_size"], h // config["num_heads"]
    bias = config.get("attention_bias", True)
    shapes = {"patch_embed.proj.weight": [h, config["in_channels"], config["temporal_patch_size"],
                                          config["patch_size"], config["patch_size"]],
              "patch_embed.proj.bias": [h], "post_layernorm.weight": [h],
              "downsample.weight": [out, h, merge, merge], "downsample.bias": [out],
              "merger.proj.weight": [out, out], "merger.post_projection_norm.weight": [out],
              "merger.post_projection_norm.bias": [out], "merger.gate_proj.weight": [inner, out],
              "merger.up_proj.weight": [inner, out], "merger.down_proj.weight": [out, inner]}
    for layer in range(config["depth"]):
        for part, width, inputs in (("norm1", h, None), ("norm2", h, None), ("attn.q_norm", head, None),
                                    ("attn.k_norm", head, None), ("attn.qkv", 3 * h, h), ("attn.proj", h, h),
                                    ("mlp.gate_proj", mid, h), ("mlp.up_proj", mid, h), ("mlp.down_proj", h, mid)):
            shapes[f"blocks.{layer}.{part}.weight"] = [width] if inputs is None else [width, inputs]
            if inputs is not None and bias:
                shapes[f"blocks.{layer}.{part}.bias"] = [width]
    return shapes


def _channels_last(name: str, shape: list[int]) -> list[int] | None:
    """An MLX conversion's convolution layout (channels last) of a PyTorch-layout shape, else None."""

    if name in ("patch_embed.proj.weight", "downsample.weight"):
        return [shape[0], *shape[2:], shape[1]]
    return None


def checkpoint_vision(model_dir: str | Path) -> tuple[dict, int]:
    """Validate the tower's headers before any model or accelerator allocation; its bytes as stored (F32 = 4)."""

    config = vision_config(model_dir)
    tensors = float_headers(model_dir)
    shapes = tower_shapes(config)
    if set(tensors) != set(shapes):
        raise ValueError("checkpoint needs the complete unquantized GLM vision tower")
    for name, expected in shapes.items():
        if tensors[name]["shape"] not in (expected, _channels_last(name, expected)):
            raise ValueError(f"vision tensor shape differs from the checkpoint configuration: {name}")
    return config, sum(math.prod(v["shape"]) * SIZES[v["dtype"]] for v in tensors.values())


def weight_transform(base, enabled: bool, rank: int, offload: bool = False):
    """The startup estimate counts the tower on rank zero, the only rank that loads it (offloaded: it rests in host
    RAM, and the geometry's reserve holds its visit instead)."""

    from .qwen_checkpoint import vision_key

    def transform(name, info):
        if enabled and rank == 0 and vision_key(name) is not None:
            if offload:
                return 0, 0
            from tensorfold.cuda.geometry import size

            return size(info), 0                         # stored bytes: the tower is not recast on load
        return base(name, info)
    return transform


def capacity_geometry(base, model_dir, enabled: bool, rank: int, offload: bool = False):
    def geometry(text):
        from tensorfold.cuda.capacity import Geometry

        result = base(text)
        if not enabled:
            return result
        tower_bytes = checkpoint_vision(model_dir)[1] if rank == 0 else 0
        if offload and rank == 0:
            reserve = tower_bytes + OFFLOAD_ACTIVATION_BYTES           # the tower's visit to the GPU
        else:
            reserve = WORKSPACE_BYTES if rank == 0 else 128 * 1024**2     # rank one holds the received features
        return Geometry(lambda slots: result.bytes_at(slots) + reserve, result.reserve, result.minimum_slots)
    return geometry


def placeholder_rows(prompt, image_token: int) -> tuple[int, ...]:
    return tuple(i for i, token in enumerate(prompt) if token == image_token)


ATTENTION = "tensorfold_mixed_sdpa"


def _mixed_sdpa(module, query, key, value, attention_mask, **kwargs):
    """transformers' SDPA, except that q, k and v meet in their common dtype first (a lossless widening).

    Cast site 2. A norm weight stored in F32 makes q and k F32 (`weight * x` promotes) while v keeps its linear's
    dtype, and SDPA refuses mixed inputs. Widening v (or q and k) loses nothing; narrowing q and k would."""

    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    common = torch.promote_types(torch.promote_types(query.dtype, key.dtype), value.dtype)
    return sdpa_attention_forward(module, query.to(common), key.to(common), value.to(common), attention_mask,
                                  **kwargs)


def keep_stored_dtypes(tower) -> None:
    """Make a tower whose tensors keep their stored dtypes run, casting activations only, never weights.

    transformers' tower assumes one dtype. Its own boundaries (Glm5NextVisionModel, transformers 5.17+):
      * patch embedding casts its input to the convolution weight's dtype itself; the residual stream then has that
        dtype.
      * Glm5NextRMSNorm computes in F32 and returns `weight * x.to(input_dtype)`: an F32 weight gives F32 out.
      * rotary embedding runs in F32 and returns q and k in their own dtype (no boundary).
      * linears, convolutions and LayerNorm need input dtype == weight dtype: cast site 1 below.
      * SDPA needs q, k, v alike: cast site 2 (`_mixed_sdpa`).
      * residual adds and `act(gate) * up` use torch's promotion (widening); left as transformers wrote them.
    Cast site 1: a forward pre-hook turns the activation entering a Linear, Conv or LayerNorm into that module's
    weight dtype. Site 3: a module whose bias is stored in another dtype than its weight (a fused kernel would
    refuse it) adds the bias to the output unfused, in the widened dtype, so neither tensor is recast."""

    from torch import nn
    from torch.nn import functional as F

    def to_weight(module, args):
        x = args[0]
        return None if x.dtype == module.weight.dtype else (x.to(module.weight.dtype), *args[1:])

    def split(module):
        b, w = module.bias, module.weight
        if isinstance(module, nn.Linear):
            return lambda x: F.linear(x, w) + b
        if isinstance(module, nn.LayerNorm):
            return lambda x: F.layer_norm(x, module.normalized_shape, None, None, module.eps) * w + b
        conv = F.conv3d if isinstance(module, nn.Conv3d) else F.conv2d
        shape = (1, -1) + (1,) * (w.dim() - 2)
        return lambda x: conv(x, w, None, module.stride, module.padding, module.dilation, module.groups) + b.view(shape)

    for module in tower.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.LayerNorm)):
            module.register_forward_pre_hook(to_weight)
            if module.bias is not None and module.bias.dtype != module.weight.dtype:
                module.forward = split(module)


def build_tower(model_dir, device):
    """The checkpoint's vision tower on `device`, every tensor in its stored dtype (no dtype guards yet)."""

    import torch
    from transformers import AttentionInterface
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    from .qwen_checkpoint import vision_tensors

    config_dict, _ = checkpoint_vision(model_dir)
    AttentionInterface.register(ATTENTION, _mixed_sdpa)
    config = Glm5NextVisionConfig(**config_dict)
    config._attn_implementation = ATTENTION
    with torch.device("meta"):
        tower = Glm5NextVisionModel(config)
    shapes = tower_shapes(config_dict)
    kinds = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    tensors = {}
    for name, (path, info, begin) in vision_tensors(Path(model_dir)).items():
        start, end = info["data_offsets"]                # float_headers checked each range
        with path.open("rb") as stream:
            stream.seek(begin + start)
            raw = bytearray(stream.read(end - start))
        value = torch.frombuffer(raw, dtype=kinds[info["dtype"]]).reshape(info["shape"])
        if list(value.shape) != shapes[name]:           # checkpoint_vision allowed only a channels-last convolution
            value = value.movedim(-1, 1)
        tensors[name] = value.to(device=device).contiguous()      # device only: the stored dtype is kept
    tower.load_state_dict(tensors, strict=True, assign=True)
    rotary = tower.rotary_pos_emb                        # a meta build leaves the frequency buffers empty
    inv_freq, _ = rotary.compute_axial_rope_parameters(config)
    for buffer in ("inv_freq", "original_inv_freq"):
        rotary._buffers[buffer] = inv_freq.to(device).clone()
    return tower.eval()


class GLMCudaVision:
    """Only the image tower is loaded, on rank zero; the CUDA engine keeps all language computation."""

    def __init__(self, model_dir, device, allow_urls: bool = False, offload: bool = False):
        from .glm_processing import GLMImageProcessor

        self.allow_urls = allow_urls
        self.offload = offload
        self._lock = threading.Lock()   # one image on the GPU at a time when the tower is offloaded
        self.config, self.weight_bytes = checkpoint_vision(model_dir)
        self.frontend = GLMImageProcessor.from_directory(model_dir)
        self.image_token = self.frontend.image_token_id
        self.device = device
        self.tower = build_tower(model_dir, "cpu" if offload else device)    # offloaded: resident in host RAM
        keep_stored_dtypes(self.tower)

    def prepare(self, *args, **kwargs):
        kwargs.setdefault("max_image_tokens", TOKENS_PER_IMAGE)     # whatever budget the request's images share
        return self.frontend.prepare(*args, **kwargs)

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

    def continued(self, prepared, tokens):
        return self.frontend.continued(prepared, tokens)

    def encode(self, prepared, prompt) -> EncodedVision:
        with self._on_gpu():
            return self._encode(prepared, prompt)

    def _encode(self, prepared, prompt) -> EncodedVision:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        grid = prepared.image_grid_thw
        merge = self.config["spatial_merge_size"]
        if len(grid.shape) != 2 or grid.shape[1] != 3 or any(int(t) != 1 for t in grid[:, 0]):
            raise ValueError("CUDA vision accepts images with one temporal grid, not video")
        if any(int(v) <= 0 for row in grid for v in row) or any(int(h) % merge or int(w) % merge for _, h, w in grid):
            raise ValueError("image grids must contain positive merge-aligned dimensions")
        sizes = [int(t) * int(h) * int(w) for t, h, w in grid]
        patches = sum(sizes)
        if patches > MAX_REQUEST_PATCHES or max(sizes) > MAX_PATCHES:
            raise ValueError(f"image request exceeds the CUDA vision budget of {MAX_PATCHES} patches an image and "
                             f"{MAX_REQUEST_PATCHES} a request")
        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"] ** 2
        if tuple(prepared.pixel_values.shape) != (patches, width):
            raise ValueError("image patch tensor has an invalid shape")
        rows = tuple(i for start, end in prepared.image_spans for i in range(start, end))
        if rows != placeholder_rows(prompt, self.image_token) or len(rows) != patches // merge**2:
            raise ValueError("vision feature rows must match every image placeholder exactly")
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            parts = []
            # images never attend to one another: runs of whole images, at most MAX_PATCHES a tower call (one call,
            # as before, whenever the request fits it), so the scratch stays what one full-size image needs
            for begin, end in image_runs(sizes, MAX_PATCHES):
                done = sum(sizes[:begin])
                # a copy: the prepared arrays are read-only, and a tensor may not share them
                # pixels take the patch convolution's own dtype (its weight's): the input is rounded once, if at all
                pixels = torch.tensor(prepared.pixel_values[done:done + sum(sizes[begin:end])],
                                      dtype=self.tower.patch_embed.proj.weight.dtype, device=self.device)
                grids = torch.tensor(grid[begin:end], dtype=torch.int64, device=self.device)
                parts.append(self.tower(pixels, grid_thw=grids, return_dict=True).pooler_output)
            features = parts[0] if len(parts) == 1 else torch.cat(parts)
            # The one required cast: the handoff into the language model's activations. The engine's prompt rows
            # (`b.x`, forward.py) are bf16 and every embedding path (`glue.embed`: 4-bit table or BF16 table, both
            # written as bf16) yields bf16 rows, so image features must become bf16 here, as a token's would.
            features = features.to(dtype=torch.bfloat16).contiguous()
        if tuple(features.shape) != (len(rows), self.config["out_hidden_size"]):
            raise ValueError("vision tower returned a different number of image features")
        return EncodedVision(rows, features, None, 0)


def share_encoded(payload: EncodedVision | None, rank: int, comm, prompt, image_token: int, hidden: int,
                  device) -> EncodedVision:
    """Rank zero's features on both ranks; each finds the image rows in the prompt both already hold."""

    import torch

    rows = placeholder_rows(prompt, image_token)
    if not rows or (payload is not None and tuple(payload.rows) != rows):
        raise ValueError("vision feature rows must match every image placeholder exactly")
    send = (payload.features if rank == 0 else
            torch.zeros((len(rows), hidden), dtype=torch.bfloat16, device=device))
    if tuple(send.shape) != (len(rows), hidden) or send.dtype != torch.bfloat16:
        raise ValueError("vision features differ from the image placeholders or the embedding width")
    got = torch.empty((2 * len(rows), hidden), dtype=torch.bfloat16, device=device)
    comm.all_gather(send.contiguous().view(-1), got.view(-1))
    return EncodedVision(rows, got[:len(rows)], None, 0)       # the same bits on both ranks


def replace_rows(x, payload: EncodedVision, start: int, end: int, copies: int = 1):
    """x [end - start, copies * D] holds prompt rows start..end: each image row's feature goes into every copy."""

    import torch

    selected = [(i, row - start) for i, row in enumerate(payload.rows) if start <= row < end]
    if selected:
        source, target = zip(*selected)
        source = torch.tensor(source, dtype=torch.int64, device=x.device)
        target = torch.tensor(target, dtype=torch.int64, device=x.device)
        rows = payload.features.index_select(0, source)
        x.view(x.shape[0], copies, -1).index_copy_(0, target, rows[:, None, :].expand(-1, copies, -1).contiguous())
    return x
