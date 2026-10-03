"""A tiny random Gemma 4 unified (dense) model at oQ-style mixed widths, and a tiny random MTP assistant for it."""

from __future__ import annotations

from typing import Any

import numpy as np

TINY = {
    "model_type": "gemma4_text", "hidden_size": 256, "num_hidden_layers": 4, "intermediate_size": 512,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 128,
    "num_global_key_value_heads": 1, "attention_k_eq_v": True, "vocab_size": 256, "vocab_size_per_layer_input": 256,
    "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "sliding_window": 8,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "enable_moe_block": False, "use_double_wide_mlp": False, "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
}

ASSISTANT = {
    "model_type": "gemma4_unified_assistant", "backbone_hidden_size": TINY["hidden_size"],
    "use_ordered_embeddings": False, "num_centroids": 16, "centroid_intermediate_top_k": 4,
    "tie_word_embeddings": True, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "model_type": "gemma4_unified_text", "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 128,
        "num_global_key_value_heads": 1, "attention_k_eq_v": True, "vocab_size": 256, "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"], "enable_moe_block": False,
        "final_logit_softcapping": None, "tie_word_embeddings": True, "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 2, "use_double_wide_mlp": False,
    },
}

# oQ's per-module widths: most at 8 bits, a few at 4 (a stacked q|k|v then splits into two widths)
FOUR_BIT = ("layers.1.self_attn.v_proj", "layers.2.mlp.down_proj", "layers.3.mlp.gate_proj")


def _predicate(path: str, module: Any) -> Any:
    if not hasattr(module, "to_quantized"):
        return False
    if any(path.endswith(p) for p in FOUR_BIT):
        return {"group_size": 64, "bits": 4}
    return True


def tiny_text(seed: int = 0) -> Any:
    """mlx_lm's gemma4_text (dense) on random weights, 8-bit in groups of 64 with a few 4-bit modules."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma4_text

    mx.random.seed(seed)
    text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(TINY))
    nn.quantize(text, group_size=64, bits=8, class_predicate=_predicate)
    for layer in text.model.layers:
        layer.layer_scalar = mx.array([0.8], dtype=mx.bfloat16)
    text.set_dtype(mx.bfloat16)
    mx.eval(text.parameters())
    return text


def tiny_assistant(seed: int = 1) -> Any:
    """A random two-layer assistant whose layers read the tiny target's last sliding and full layers."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma4_text

    from tensorfold.families.gemma4_unified.assistant import Assistant

    mx.random.seed(seed)
    cfg = dict(ASSISTANT["text_config"], num_kv_shared_layers=0)
    text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(cfg))
    for layer in text.model.layers:
        for name in ("k_proj", "v_proj", "k_norm", "v_norm"):
            if name in layer.self_attn:
                del layer.self_attn[name]
    holder = nn.Module()
    holder.text = text
    holder.pre_projection = nn.Linear(2 * TINY["hidden_size"], cfg["hidden_size"], bias=False)
    holder.post_projection = nn.Linear(cfg["hidden_size"], TINY["hidden_size"], bias=False)
    nn.quantize(holder, group_size=64, bits=4)
    holder.set_dtype(mx.bfloat16)
    mx.eval(holder.parameters())
    return Assistant(text, holder.pre_projection, holder.post_projection, ASSISTANT)


def tokens(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TINY["vocab_size"], size=n)]
