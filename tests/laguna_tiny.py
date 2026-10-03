"""A tiny random Laguna in the oQ checkpoints' layout: mixed widths and groups, 4-bit experts, a bf16 router."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

TINY = {
    "model_type": "laguna", "hidden_size": 256, "num_hidden_layers": 4, "intermediate_size": 512,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 512,
    "num_attention_heads_per_layer": [4, 6, 6, 4], "sliding_window": 8, "gating": "per-head",
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "num_experts": 32, "num_experts_per_tok": 4, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 256,
    "moe_routed_scaling_factor": 2.5, "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576,
    "rope_parameters": {
        "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 128.0,
                           "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 32.0,
                           "attention_factor": 1.4852030263919618, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    },
}

# oQ4e's mix: 4-bit groups of 128 by default, attention 8- and 5-bit, shared experts 8-bit in groups of 128
WIDTHS = {
    "q_proj": (8, 64), "k_proj": (8, 64), "v_proj": (8, 64), "g_proj": (5, 64), "o_proj": (5, 64),
    "shared_expert": (8, 128), "mlp.gate_proj": (5, 64), "mlp.up_proj": (5, 64), "mlp.down_proj": (6, 64),
    "lm_head": (8, 64), "embed_tokens": (8, 64),
}


def config(**changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(TINY)
    out.update(changes)
    return out


def _width(path: str) -> Any:
    if path.endswith("mlp.gate.proj"):
        return False                                  # the router stays bf16
    for key, (bits, group) in WIDTHS.items():
        if key in path and "switch_mlp" not in path:
            return {"bits": bits, "group_size": group}
    return {"bits": 4, "group_size": 128}


def tiny_model(seed: int = 0, **changes: Any) -> Any:
    """The family's MLX model on random weights, quantized in the oQ4e layout."""

    import mlx.core as mx
    import mlx.nn as nn

    from tensorfold.families.laguna import mlx_model

    mx.random.seed(seed)
    model = mlx_model.Model(mlx_model.ModelArgs.from_dict(config(**changes)))
    for layer in model.layers:
        if layer.sparse:
            layer.mlp.gate.e_score_correction_bias = (mx.random.uniform(shape=(TINY["num_experts"],)) * 0.1)
    nn.quantize(model, class_predicate=lambda path, m: hasattr(m, "to_quantized") and _width(path))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def tokens(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TINY["vocab_size"], size=n)]


DRAFT = {
    "architectures": ["DFlashLagunaForCausalLM"], "model_type": "laguna", "hidden_size": 256, "head_dim": 128,
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 512,
    "rms_norm_eps": 1e-6, "vocab_size": 512, "max_position_embeddings": 1048576, "rope_theta": 500000.0,
    "sliding_window": 8, "layer_types": ["sliding_attention", "sliding_attention"], "gating": "per-head",
    "num_experts": 0, "draft_vocab_size": 512,
    "dflash_config": {"block_size": 8, "mask_token_id": 5, "num_target_layers": 4, "target_layer_ids": [1, 3],
                      "causal": True},
}


def tiny_drafter(path: Any, seed: int = 1) -> Any:
    """A random drafter in poolside's layout (fused qkv_proj, a norm a tap) saved at ``path``."""

    import json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tensorfold.drafters.dflash_drafter import _vendor
    from tensorfold.families.laguna import drafter

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(DRAFT))
    mx.random.seed(seed)
    model = drafter._model_class()(_vendor().DFlashConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        intermediate_size=512, vocab_size=512, rms_norm_eps=1e-6, rope_theta=500000.0,
        max_position_embeddings=1048576, block_size=8, target_layer_ids=(1, 3), num_target_layers=4,
        mask_token_id=5, layer_types=("sliding_attention", "sliding_attention"), sliding_window=8, is_causal=True))
    model.set_dtype(mx.bfloat16)
    weights = {k: v for k, v in tree_flatten(model.parameters()) if not k.startswith(("embed_tokens", "lm_head"))}
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return path
