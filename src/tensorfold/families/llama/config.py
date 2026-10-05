"""Validate the narrow Llama layout without importing a backend."""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Any


def normalize(config: dict[str, Any]) -> dict[str, Any]:
    """Map Transformers 5's default RoPE to mlx-lm's theta; never guess conflicting settings."""
    out = dict(config)
    if out.get("model_type") != "llama" or out.get("text_config"):
        raise ValueError("Llama requires a text-only model_type=llama")
    for key in ("model_file", "quantize_activations", "attention_softcap", "final_logit_softcapping",
                "use_sliding_window", "num_local_experts", "qk_norm"):
        if out.get(key):
            raise ValueError(f"Llama: unsupported {key}")
    for key in ("attention_multiplier", "residual_multiplier", "embedding_multiplier"):
        if out.get(key, 1) != 1:
            raise ValueError(f"Llama: unsupported {key}")
    if out.get("pretraining_tp", 1) != 1:
        raise ValueError("Llama: unsupported pretraining_tp")
    if out.get("tie_word_embeddings", False):
        raise ValueError("Llama requires an untied lm_head")
    if "tie_word_embeddings" not in out:
        out["tie_word_embeddings"] = False
    if out.get("hidden_act", "silu") != "silu":
        raise ValueError("Llama requires SwiGLU (hidden_act=silu)")
    for key in ("attention_bias", "mlp_bias", "tie_word_embeddings", "rope_traditional"):
        if key in out and not isinstance(out[key], bool):
            raise ValueError(f"Llama: {key} must be boolean")
    dims = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
    if any(type(out.get(k)) is not int or out[k] <= 0 for k in dims):
        raise ValueError("Llama: invalid positive integer geometry")
    hidden, heads = out["hidden_size"], out["num_attention_heads"]
    kv = heads if out.get("num_key_value_heads") is None else out["num_key_value_heads"]
    dim = hidden // heads if out.get("head_dim") is None else out["head_dim"]
    if (type(kv) is not int or kv <= 0 or heads % kv or hidden % heads
            or type(dim) is not int or dim <= 0 or dim % 2 or heads * dim != hidden):
        raise ValueError("Llama: unsupported head geometry")
    layers = out.get("layer_types")
    if layers is None:
        layers = ["full_attention"] * out["num_hidden_layers"]
    if out.get("sliding_window") or layers != ["full_attention"] * out["num_hidden_layers"]:
        raise ValueError("Llama requires full attention in every layer")
    eps = out.get("rms_norm_eps", 1e-5)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Llama requires positive finite rms_norm_eps")
    out["rms_norm_eps"] = eps
    context = out.get("max_position_embeddings")
    if context is not None and (type(context) is not int or context < 576):
        raise ValueError("Llama: native context must be at least 576 tokens for grid256 server admission probes")
    for key in ("torch_dtype", "dtype"):
        if out.get(key) not in (None, "bfloat16", "bf16"):
            raise ValueError("Llama reads bf16 weights only")
    rope = out.get("rope_parameters")
    rope = {} if rope is None else rope
    if not isinstance(rope, dict) or set(rope) - {"rope_type", "rope_theta"} or rope.get("rope_type", "default") != "default":
        raise ValueError("Llama supports default RoPE only")
    scaling = out.get("rope_scaling")
    if (scaling is not None and (not isinstance(scaling, dict) or scaling)
            or out.get("rope_traditional") or out.get("partial_rotary_factor", 1.0) != 1.0):
        raise ValueError("Llama supports full-dimension default RoPE only")
    theta = rope.get("rope_theta", out.get("rope_theta", 10000))
    if "rope_theta" in rope and "rope_theta" in out and theta != out["rope_theta"]:
        raise ValueError("Llama: conflicting RoPE theta declarations")
    if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("Llama requires positive finite RoPE theta")
    out["rope_theta"] = theta
    out.pop("rope_parameters", None)
    blocks = [out[k] for k in ("quantization", "quantization_config") if out.get(k) is not None]
    for block in blocks:
        _quant(block)
    if len(blocks) == 2 and blocks[0] != blocks[1]:
        raise ValueError("Llama: conflicting quantization declarations")
    if blocks:
        quant = dict(blocks[0])
        quant.setdefault("group_size", 64)
        for name, spec in quant.items():
            if name in ("lm_head", "model.embed_tokens") and spec is not False:
                raise ValueError(f"Llama requires unquantized {name}")
            if isinstance(spec, dict):
                _quant(spec)
            elif name not in ("bits", "group_size", "mode", "quant_method") and not isinstance(spec, bool):
                raise ValueError("Llama reads MLX affine 8-bit/group-64 declarations only")
        out["quantization"] = quant
    return out


def _quant(spec: dict[str, Any]) -> None:
    if (not isinstance(spec, dict) or type(spec.get("bits")) is not int or spec["bits"] != 8
            or type(spec.get("group_size", 64)) is not int or spec.get("group_size", 64) != 64
            or spec.get("mode", "affine") != "affine" or spec.get("quant_method", "mlx") != "mlx"):
        raise ValueError("Llama reads MLX affine 8-bit/group-64 projections only")


def check_names(names: Any) -> None:
    for name in names:
        if any(name.startswith(p + ".") and name != p + ".weight" for p in ("lm_head", "model.embed_tokens")):
            raise ValueError(f"Llama requires unquantized bf16 lm_head and embed_tokens: {name}")


def check_headers(model_dir: str | Path, config: dict[str, Any]) -> None:
    """Inspect only safetensors headers; reject unsupported dtypes before the loader maps weights."""
    headers = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        with path.open("rb") as file:
            size_bytes = file.read(8)
            if len(size_bytes) != 8:
                raise ValueError("Llama: truncated safetensors header")
            size, = struct.unpack("<Q", size_bytes)
            if size > min(path.stat().st_size - 8, 64 * 1024 * 1024):
                raise ValueError("Llama: invalid safetensors header size")
            entries = json.loads(file.read(size))
        entries.pop("__metadata__", None)
        if headers.keys() & entries.keys():
            raise ValueError("Llama: duplicate weight tensors")
        headers.update(entries)
    if not headers:
        return
    check_names(headers)
    for name, entry in headers.items():
        if entry["dtype"] not in ("BF16", "U32"):
            raise ValueError(f"Llama requires bf16 parameters or packed U32 weights: {name}")
        if entry["dtype"] == "U32" and (not name.endswith(".weight") or not config.get("quantization")):
            raise ValueError(f"Llama: unsupported packed weight {name}")
    for name, entry in headers.items():
        base, part = name.rsplit(".", 1)
        shape = _shape(base, config)
        if base in ("lm_head", "model.embed_tokens") and entry["dtype"] != "BF16":
            raise ValueError(f"Llama requires unquantized bf16 {name}")
        weight = headers.get(base + ".weight")
        packed = weight is not None and weight["dtype"] == "U32"
        if weight is None and part in ("scales", "biases") and config.get("quantization"):
            packed = True
        if part == "weight" and packed and len(shape) == 2 and shape[1] % 64 == 0:
            shape = [shape[0], shape[1] // 4]
        elif part in ("scales", "biases") and packed and len(shape) == 2 and shape[1] % 64 == 0:
            shape = [shape[0], shape[1] // 64]
        elif part == "bias" and len(shape) == 2:
            shape = [shape[0]]
        elif part != "weight" or packed:
            raise ValueError(f"Llama: unsupported tensor shape for {name}")
        if entry["shape"] != shape:
            raise ValueError(f"Llama: unsupported tensor shape for {name}: expected {shape}, got {entry['shape']}")


def _shape(base: str, config: dict[str, Any]) -> list[int]:
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    if base in ("lm_head", "model.embed_tokens"):
        return [config["vocab_size"], hidden]
    if base == "model.norm":
        return [hidden]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", base)
    if match is None or int(match[1]) >= config["num_hidden_layers"]:
        raise ValueError(f"Llama: unsupported weight {base}")
    name = match[2]
    kv = config.get("num_key_value_heads") or config["num_attention_heads"]
    head = config.get("head_dim") or hidden // config["num_attention_heads"]
    shapes = {"self_attn.q_proj": [hidden, hidden], "self_attn.k_proj": [kv * head, hidden],
              "self_attn.v_proj": [kv * head, hidden], "self_attn.o_proj": [hidden, hidden],
              "mlp.gate_proj": [intermediate, hidden], "mlp.up_proj": [intermediate, hidden],
              "mlp.down_proj": [hidden, intermediate], "input_layernorm": [hidden],
              "post_attention_layernorm": [hidden]}
    if name not in shapes:
        raise ValueError(f"Llama: unsupported weight {base}")
    return shapes[name]
