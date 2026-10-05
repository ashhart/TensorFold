"""DeepSeek-V4.1-Flash's settings from the checkpoint's config.json, and the decode path's switches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# the decode path's widest call: wider inputs (prompt chunks) take MLX's batched prefill path
DECODE_ROWS = 16
# decode steps that take a window's rows in one kernel, each row with its one-row bits (tests switch them off)
ROW_KERNELS = ("router", "hc", "moe")
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
    kv_source_layer_ids: list[int]
    index_source_layer_ids: list[int]
    candidate_source_layer_id: int
    candidate_block_size: int
    candidate_topk_blocks: int
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
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    num_nextn_predict_layers: int
    dspark_target_layer_ids: list[int]
    engram_layer_ids: list[int]
    engram_num_embeddings: list[int]
    engram_ngram: int
    engram_vocab_size: int
    engram_compressed_vocab_size: int
    engram_pad_token_id: int
    engram_n_heads: int
    engram_head_dim: int
    eos_token_id: list[int]

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer]) if layer < len(self.compress_ratios) else 0

    def kv_source(self, layer: int) -> bool:
        return layer in self.kv_source_layer_ids

    def index_source(self, layer: int) -> bool:
        return layer in self.index_source_layer_ids

    def mode(self, layer: int) -> str:
        """A compressed layer's CSA2 mode: Full (kv+index source), Reindex (index source), Reuse."""
        if self.kv_source(layer):
            return "full"
        return "reindex" if self.index_source(layer) else "reuse"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        if str(t.get("scoring_func", "sqrtsoftplus")) != "sqrtsoftplus":
            raise ValueError(f"deepseek_v41: routing scores {t.get('scoring_func')!r}; only sqrtsoftplus is implemented")
        if str(t.get("scoring_topk_method", "noaux_tc")) != "noaux_tc":
            raise ValueError(f"deepseek_v41: routing top-k {t.get('scoring_topk_method')!r}; only noaux_tc is implemented")
        if int(t.get("num_key_value_heads", 1)) != 1 or int(t.get("n_shared_experts", 1)) != 1:
            raise ValueError("deepseek_v41: one shared key/value head and one shared expert are implemented")
        if int(t.get("hc_mult", 4)) != 4:
            raise ValueError("deepseek_v41: the hyper-connection kernels take 4 streams")
        layers = int(t["num_hidden_layers"])
        ratios = [int(r) for r in t.get("compress_ratios") or []]
        if len(ratios) < layers or any(r not in (0, 1, 2) for r in ratios):
            raise ValueError(f"deepseek_v41: compress_ratios {ratios} (one of 0, 1, 2 for each of {layers} layers)")
        kv_sources = [int(i) for i in t.get("kv_source_layer_ids") or []]
        index_sources = [int(i) for i in t.get("index_source_layer_ids") or []]
        for i in kv_sources:
            if not ratios[i]:
                raise ValueError(f"deepseek_v41: kv source layer {i} is not a compressed layer")
        for i in index_sources:
            if not ratios[i]:
                raise ValueError(f"deepseek_v41: index source layer {i} is not a compressed layer")
        candidate = int(t.get("candidate_source_layer_id", -1))
        if candidate >= 0 and candidate not in kv_sources:
            raise ValueError(f"deepseek_v41: candidate source layer {candidate} is not a kv source layer")
        engram_layers = [int(i) for i in t.get("engram_layer_ids") or []]
        rope = dict(t.get("rope_scaling") or {})
        if rope and str(rope.get("type") or rope.get("rope_type")) not in ("yarn", "deepseek_yarn"):
            raise ValueError(f"deepseek_v41: rope_scaling {rope}; the compressed layers use YaRN")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=layers, vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)), num_attention_heads=int(t["num_attention_heads"]),
            head_dim=int(t["head_dim"]), qk_rope_head_dim=int(t["qk_rope_head_dim"]),
            q_lora_rank=int(t["q_lora_rank"]), o_lora_rank=int(t["o_lora_rank"]), o_groups=int(t["o_groups"]),
            sliding_window=int(t.get("sliding_window", 128)), compress_ratios=ratios,
            kv_source_layer_ids=kv_sources, index_source_layer_ids=index_sources, candidate_source_layer_id=candidate,
            candidate_block_size=int(t.get("candidate_block_size", 8)),
            candidate_topk_blocks=int(t.get("candidate_topk_blocks", 2048)),
            rope_theta=float(t.get("rope_theta", 10000.0)),
            compress_rope_theta=float(t.get("compress_rope_theta", 160000.0)),
            rope_factor=float(rope.get("factor", 1.0)), rope_original=int(rope.get("original_max_position_embeddings", 0)),
            beta_fast=float(rope.get("beta_fast", 32)), beta_slow=float(rope.get("beta_slow", 1)),
            index_n_heads=int(t["index_n_heads"]), index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]), n_routed_experts=int(t["n_routed_experts"]),
            n_shared_experts=int(t.get("n_shared_experts", 1)), num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]),
            norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            dspark_target_layer_ids=[int(i) for i in t.get("dspark_target_layer_ids") or []],
            engram_layer_ids=engram_layers,
            engram_num_embeddings=[int(n) for n in t.get("engram_num_embeddings") or []],
            engram_ngram=int(t.get("engram_max_ngram_size", 4)), engram_vocab_size=int(t.get("engram_vocab_size", 0)),
            engram_compressed_vocab_size=int(t.get("engram_compressed_vocab_size", 0)),
            engram_pad_token_id=int(t.get("engram_pad_token_id", 2)),
            engram_n_heads=int(t.get("engram_n_heads", 8)), engram_head_dim=int(t.get("engram_head_dim", 256)),
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
