"""Tiny random Gemma 4 models (MoE and dense, uniform and oQ-mixed widths) and a proposer that drafts a known reply."""

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


# the dense layout (31B): no router or experts, one GeGLU MLP a layer
TINY_DENSE = {k: v for k, v in TINY.items() if k not in ("num_experts", "top_k_experts", "moe_intermediate_size")}
TINY_DENSE.update(enable_moe_block=False, intermediate_size=512)

# the kinds of checkpoint the family serves: a layout and a quantization (oQ's per-layer mixed widths or uniform)
KINDS = ("moe", "dense", "moe-oq", "dense-oq")


def oq_widths(path: str, _: Any = None) -> dict[str, int] | bool:
    """An oQ-like layout on the tiny model: projections at 2..8 bits in groups of 32, 64 or 128, experts 4-bit."""

    if ".experts." in path or "switch_glu" in path:
        return True
    layer = int(path.split("layers.")[1].split(".")[0]) if "layers." in path else -1
    table = {"embed_tokens": (8, 64), "q_proj": (5, 64), "k_proj": (6, 32), "v_proj": (4, 128), "o_proj": (8, 64),
             "gate_proj": (3, 64), "up_proj": (4, 32), "down_proj": (6, 128), "router.proj": (6, 64)}
    for name, (bits, group) in table.items():
        if path.endswith(name):
            if layer % 2 and name in ("q_proj", "gate_proj", "router.proj"):      # mixed between layers too
                bits, group = {"q_proj": (4, 64), "gate_proj": (2, 64), "router.proj": (8, 64)}[name]
            return {"bits": bits, "group_size": group}
    return True


def tiny_text(seed: int = 0, kind: str = "moe") -> Any:
    """mlx_lm's gemma4_text on random weights: "moe" as mlx_lm quantizes it, "dense" uniform 8-bit, "-oq" mixed."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma4_text

    mx.random.seed(seed)
    dense = kind.startswith("dense")
    text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(TINY_DENSE if dense else TINY))
    predicate = oq_widths if kind.endswith("-oq") else text.quant_predicate
    nn.quantize(text, group_size=64, bits=8 if kind == "dense" else 4,
                class_predicate=lambda path, m: hasattr(m, "to_quantized") and predicate(path, m))
    for layer in text.model.layers:
        if not dense:
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
