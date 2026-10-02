"""Full GLM-5.3's settings from config.json, and the checks for what the engine implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INDEXER_TYPES = ("full", "shared")


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_hidden_layers: int                 # decoder layers; the MTP layer follows at this index
    num_mtp_layers: int
    first_k_dense_replace: int
    intermediate_size: int                 # dense-layer MLP width
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rope_interleave: bool
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: tuple[str, ...]
    index_share_for_mtp: bool
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int
    eos_token_ids: tuple[int, ...]

    @property
    def latent_width(self) -> int:
        """One token's MLA cache entry: the kv latent and the shared RoPE key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def moe_layers(self) -> range:
        return range(self.first_k_dense_replace, self.num_hidden_layers)

    def full_indexer(self, layer: int) -> bool:
        """Whether a layer computes its own top-k (else it reuses the last full layer's)."""
        return self.indexer_types[layer] == "full"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        c = raw.get("text_config") or raw
        if c.get("model_type") != "glm_moe_dsa":
            raise ValueError(f"not a GLM-5.3 config (model_type {c.get('model_type')!r})")
        n = int(c["num_hidden_layers"])
        types = tuple(c.get("indexer_types") or ())
        if len(types) != n or set(types) - set(INDEXER_TYPES) or types[0] != "full":
            raise ValueError(f"indexer_types must list 'full'/'shared' for all {n} layers, starting with 'full'")
        if int(c.get("n_group", 1)) != 1 or int(c.get("topk_group", 1)) != 1:
            raise ValueError("grouped expert routing (n_group > 1) is not implemented")
        if c.get("scoring_func", "sigmoid") != "sigmoid" or c.get("topk_method", "noaux_tc") != "noaux_tc":
            raise ValueError("only sigmoid scoring with noaux_tc routing is implemented")
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"RoPE scaling {rope.get('rope_type')!r} is not implemented")
        eos = c.get("eos_token_id")
        return cls(
            hidden_size=int(c["hidden_size"]), num_hidden_layers=n,
            num_mtp_layers=int(c.get("num_nextn_predict_layers", 0)),
            first_k_dense_replace=int(c["first_k_dense_replace"]), intermediate_size=int(c["intermediate_size"]),
            moe_intermediate_size=int(c["moe_intermediate_size"]), n_routed_experts=int(c["n_routed_experts"]),
            n_shared_experts=int(c["n_shared_experts"]), num_experts_per_tok=int(c["num_experts_per_tok"]),
            routed_scaling_factor=float(c["routed_scaling_factor"]), norm_topk_prob=bool(c["norm_topk_prob"]),
            num_attention_heads=int(c["num_attention_heads"]), q_lora_rank=int(c["q_lora_rank"]),
            kv_lora_rank=int(c["kv_lora_rank"]), qk_nope_head_dim=int(c["qk_nope_head_dim"]),
            qk_rope_head_dim=int(c["qk_rope_head_dim"]), v_head_dim=int(c["v_head_dim"]),
            rope_theta=float(rope.get("rope_theta", c.get("rope_theta", 10000.0))),
            rope_interleave=bool(c.get("rope_interleave", True)),
            index_n_heads=int(c["index_n_heads"]), index_head_dim=int(c["index_head_dim"]),
            index_topk=int(c["index_topk"]), indexer_types=types,
            index_share_for_mtp=bool(c.get("index_share_for_mtp_iteration", False)),
            max_position_embeddings=int(c["max_position_embeddings"]), rms_norm_eps=float(c["rms_norm_eps"]),
            vocab_size=int(c["vocab_size"]),
            eos_token_ids=tuple(eos if isinstance(eos, list) else [eos]),
        )
