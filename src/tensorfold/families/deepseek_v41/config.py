"""DeepSeek-V4.1-Flash's settings from the checkpoint's config.json (text_config), without torch."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_original: int
    beta_fast: float
    beta_slow: float
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_vocab_size: int
    engram_compressed_vocab_size: int
    engram_pad_token_id: int
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Config:
        """The settings the engine reads; a missing one is a KeyError naming it, not a silent default."""

        text = config.get("text_config") or config
        rope = text.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type")) != "yarn":
            raise ValueError(f"DeepSeek-V4.1 expects YaRN rope scaling, found {rope!r}")
        values: dict[str, Any] = {
            "rope_factor": float(rope["factor"]), "rope_original": int(rope["original_max_position_embeddings"]),
            "beta_fast": float(rope["beta_fast"]), "beta_slow": float(rope["beta_slow"]),
            "bos_token_id": int(config.get("bos_token_id", text.get("bos_token_id", 0))),
            "eos_token_id": int(config.get("eos_token_id", text.get("eos_token_id", 1))),
        }
        for f in fields(cls):
            if f.name in values:
                continue
            raw = text[f.name]
            values[f.name] = tuple(int(v) for v in raw) if f.type.startswith("tuple") else \
                (float(raw) if f.type == "float" else int(raw))
        made = cls(**values)
        made.validate()
        return made

    def validate(self) -> None:
        layers = self.num_hidden_layers
        if len(self.compress_ratios) != layers + self.num_nextn_predict_layers:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {layers} layers + "
                             f"{self.num_nextn_predict_layers} MTP blocks")
        if self.num_key_value_heads != 1:
            raise ValueError(f"DeepSeek-V4.1 shares one KV head; config has {self.num_key_value_heads}")
        for name in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids", "dspark_target_layer_ids"):
            bad = [i for i in getattr(self, name) if not 0 <= i < layers]
            if bad:
                raise ValueError(f"{name} names layers outside 0..{layers - 1}: {bad}")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings and engram_layer_ids differ in length")

    @property
    def layer_ratios(self) -> tuple[int, ...]:
        """Compression ratio of each decoder layer (the MTP blocks' entries follow in ``compress_ratios``)."""

        return self.compress_ratios[:self.num_hidden_layers]
