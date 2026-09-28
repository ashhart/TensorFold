"""DeepSeek-V4-Flash's settings from the checkpoint's config.json, and the decode path's switches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# the decode path's widest call: wider inputs (prompt chunks) take MLX's batched prefill path
DECODE_ROWS = 16
# decode steps that take a window's rows in one kernel, each row with its one-row bits (tests switch them off)
ROW_KERNELS = ("router", "experts", "compressor", "hc", "moe")
ENABLED = frozenset(ROW_KERNELS)
# the decode graph goes to the GPU every this many layers, so the GPU starts while Python builds the rest
EVAL_EVERY = 2
# prompt rows a prefill attention call takes at once (bounds the score matrix)
PREFILL_QUERIES = 512


def row_kernel(name: str, rows: int, rows_exact: bool) -> bool:
    return rows_exact and rows > 1 and name in ENABLED


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    compress_ratios: list[int]
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_original: int
    beta_fast: float
    beta_slow: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_hash_layers: int
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    num_nextn_predict_layers: int
    eos_token_id: list[int]

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer]) if layer < len(self.compress_ratios) else 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        if str(t.get("scoring_func", "sqrtsoftplus")) != "sqrtsoftplus":
            raise ValueError(f"deepseek_v4: routing scores {t.get('scoring_func')!r}; only sqrtsoftplus is implemented")
        if int(t.get("num_key_value_heads", 1)) != 1 or int(t.get("n_shared_experts", 1)) != 1:
            raise ValueError("deepseek_v4: one shared key/value head and one shared expert are implemented")
        if int(t.get("hc_mult", 4)) != 4:
            raise ValueError("deepseek_v4: the hyper-connection kernels take 4 streams")
        layers = int(t["num_hidden_layers"])
        ratios = [int(r) for r in t.get("compress_ratios") or []]
        if len(ratios) < layers or any(r not in (0, 4, 128) for r in ratios):
            raise ValueError(f"deepseek_v4: compress_ratios {ratios} (one of 0, 4, 128 for each of {layers} layers)")
        rope = dict(t.get("rope_scaling") or {})
        if rope and str(rope.get("type") or rope.get("rope_type")) not in ("yarn", "deepseek_yarn"):
            raise ValueError(f"deepseek_v4: rope_scaling {rope}; the compressed layers use YaRN")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=layers, vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)), num_attention_heads=int(t["num_attention_heads"]),
            head_dim=int(t["head_dim"]), qk_rope_head_dim=int(t["qk_rope_head_dim"]),
            q_lora_rank=int(t["q_lora_rank"]), o_lora_rank=int(t["o_lora_rank"]), o_groups=int(t["o_groups"]),
            sliding_window=int(t.get("sliding_window", 128)), compress_ratios=ratios,
            rope_theta=float(t.get("rope_theta", 10000.0)),
            compress_rope_theta=float(t.get("compress_rope_theta", 160000.0)),
            rope_factor=float(rope.get("factor", 1.0)),
            rope_original=int(rope.get("original_max_position_embeddings", 0)),
            beta_fast=float(rope.get("beta_fast", 32)), beta_slow=float(rope.get("beta_slow", 1)),
            index_n_heads=int(t["index_n_heads"]), index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]), n_routed_experts=int(t["n_routed_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]), moe_intermediate_size=int(t["moe_intermediate_size"]),
            num_hash_layers=int(t.get("num_hash_layers", 0)),
            routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            eos_token_id=list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else []))

    def inv_freq(self, layer: int) -> Any:
        """A layer's RoPE frequencies (fp32): plain at rope_theta on window-only layers, YaRN on compressed ones."""

        import mlx.core as mx

        dims = self.qk_rope_head_dim
        base = self.compress_rope_theta if self.ratio(layer) else self.rope_theta
        freqs = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        if not self.ratio(layer) or self.rope_original <= 0 or self.rope_factor <= 1.0:
            return freqs

        def correction(rotations: float) -> float:
            return dims * math.log(self.rope_original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(self.beta_fast)), 0)
        high = min(math.ceil(correction(self.beta_slow)), dims - 1)
        if low == high:
            high += 0.001
        smooth = 1 - mx.clip((mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
        return freqs / self.rope_factor * (1 - smooth) + freqs * smooth
