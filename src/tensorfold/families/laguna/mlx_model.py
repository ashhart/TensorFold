"""mlx-lm's ``models/laguna.py`` (MIT, Copyright 2026 Apple Inc.) for prompts, which mlx-lm 0.31 lacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    qkv_bias: bool = False
    gating: Any = True
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = 512
    partial_rotary_factor: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    mlp_layer_types: Optional[List[str]] = None
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    moe_routed_scaling_factor: float = 1.0
    moe_router_logit_softcapping: float = 0.0
    moe_router_score_func: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers


def rope_settings(args: ModelArgs, layer_type: str) -> tuple[int, float, dict[str, Any]]:
    """(rotated dims, base, scaling config) of a layer type's RoPE from the nested ``rope_parameters``."""

    params = (args.rope_parameters or {}).get(layer_type) or {}
    base = float(params.get("rope_theta", 10000.0))
    partial = params.get("partial_rotary_factor",
                         args.partial_rotary_factor if args.partial_rotary_factor is not None else 1.0)
    scaling = {k: v for k, v in params.items() if k != "rope_theta"}
    if "rope_type" not in scaling and "type" not in scaling:
        scaling["rope_type"] = "default"
    return int(args.head_dim * float(partial)), base, scaling


def _layer_rope(args: ModelArgs, layer_type: str) -> Any:
    dims, base, scaling = rope_settings(args, layer_type)
    return initialize_rope(dims, base=base, traditional=False, scaling_config=scaling,
                           max_position_embeddings=args.max_position_embeddings)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.gating = bool(args.gating)
        self.layer_type = args.layer_types[layer_idx]
        self.is_sliding = self.use_sliding = self.layer_type == "sliding_attention"
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        if self.gating:
            self.g_proj = nn.Linear(dim, n_heads, bias=False)     # per-head softplus gate on the output
        self.rope = _layer_rope(args, self.layer_type)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.gating:
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            output = (output.reshape(B, L, self.n_heads, self.head_dim) * gate[..., None]).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.softcap = args.moe_router_logit_softcapping
        self.score_func = args.moe_router_score_func
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.proj(x).astype(mx.float32)
        if self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        if self.score_func != "sigmoid":
            raise ValueError(f"Laguna's router: unknown score function {self.score_func!r}")
        scores = mx.sigmoid(logits)
        selection = scores + self.e_score_correction_bias
        inds = mx.argpartition(-selection, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        return inds, weights / weights.sum(axis=-1, keepdims=True)


class MoE(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        shared_out = self.shared_expert(x)
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y * self.routed_scaling_factor + shared_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.use_sliding = self.self_attn.use_sliding
        self.sparse = args.mlp_layer_types[layer_idx] == "sparse"
        self.mlp = MoE(args) if self.sparse else MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.sliding_window = args.sliding_window
        self.layer_types = args.layer_types
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention") if "full_attention" in self.layer_types else None
        self.swa_idx = next((i for i, layer in enumerate(self.layers) if layer.use_sliding), None)

    def __call__(self, inputs: mx.array, cache: Any = None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx]) if self.fa_idx is not None else None
        swa_mask = fa_mask
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.sliding_window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        self.tie_word_embeddings = bool(args.tie_word_embeddings)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any = None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    @property
    def quant_predicate(self) -> Any:
        def predicate(path: str, _: Any) -> Any:
            if path.endswith("mlp.gate.proj"):      # routing is discrete: the router stays more precise
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self) -> Any:
        return lambda k: "e_score_correction_bias" not in k

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        prefix = "language_model."            # repacked (mlx-vlm / oMLX oQ) checkpoints wrap every tensor in it
        if any(k.startswith(prefix) for k in weights):
            weights = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in weights.items()}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer in range(self.args.num_hidden_layers):
            at = f"model.layers.{layer}.mlp"
            gate = weights.pop(f"{at}.gate.weight", None)       # the original layout: a bare router matrix
            if gate is not None:
                weights[f"{at}.gate.proj.weight"] = gate
            bias = weights.pop(f"{at}.experts.e_score_correction_bias", None)
            if bias is not None:
                weights[f"{at}.gate.e_score_correction_bias"] = bias
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "scales", "biases"):
                    if f"{at}.experts.0.{proj}.{suffix}" not in weights:
                        continue
                    weights[f"{at}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [weights.pop(f"{at}.experts.{e}.{proj}.{suffix}") for e in range(self.args.num_experts)])
        return weights


__all__ = ["Attention", "LagunaModel", "MLP", "Model", "ModelArgs", "MoE", "MoEGate", "rope_settings"]
