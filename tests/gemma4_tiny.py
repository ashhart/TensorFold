"""A tiny random Gemma 4 MoE model in the checkpoint's layout, and a proposer that drafts a known reply."""

from __future__ import annotations

from typing import Any

import numpy as np

TINY = {
    "model_type": "gemma4_text", "hidden_size": 256, "num_hidden_layers": 4, "intermediate_size": 256,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 128,
    "num_global_key_value_heads": 1, "attention_k_eq_v": True, "vocab_size": 256, "vocab_size_per_layer_input": 256,
    "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "sliding_window": 8,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "enable_moe_block": True, "num_experts": 32, "top_k_experts": 4, "moe_intermediate_size": 512,
    "use_double_wide_mlp": False, "final_logit_softcapping": 30.0, "tie_word_embeddings": True,
}


def tiny_text(seed: int = 0) -> Any:
    """mlx_lm's gemma4_text on random weights, quantized as mlx_lm quantizes Gemma 4 (router at 8 bits)."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma4_text

    mx.random.seed(seed)
    text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(TINY))
    nn.quantize(text, group_size=64, bits=4,
                class_predicate=lambda path, m: hasattr(m, "to_quantized") and text.quant_predicate(path, m))
    for layer in text.model.layers:
        layer.router.per_expert_scale = (mx.random.uniform(shape=(TINY["num_experts"],)) + 0.5).astype(mx.bfloat16)
        layer.layer_scalar = mx.array([0.8], dtype=mx.bfloat16)
    text.set_dtype(mx.bfloat16)
    mx.eval(text.parameters())
    return text


def tokens(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TINY["vocab_size"], size=n)]


class KnownReply:
    """Drafts a known reply, proposal i wrong from its ``good[i]``-th token on: windows kept whole, in part or not."""

    last_match = 1 << 30

    def __init__(self, expected: list[int], good: tuple[int, ...] = (6, 2, 0, 9, 3)) -> None:
        self.expected = list(expected)
        self.good = good
        self.calls = 0

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        good = self.good[self.calls % len(self.good)]
        self.calls += 1
        out = list(self.expected[len(context):len(context) + max_draft])
        for j in range(good, len(out)):
            out[j] = (out[j] + 1 + j) % TINY["vocab_size"]
        return out

    def observe(self, proposed: int, accepted: int) -> None:
        pass
