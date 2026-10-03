"""DeepSeek-V4.1-Flash's settings from config.json's ``text_config`` (oMLX's layout keeps DeepSeek's names)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# the decode path's widest call: wider inputs (prompt chunks) take the batched prefill path
DECODE_ROWS = 16
# prompt rows an attention call takes at once (bounds the score matrix)
PREFILL_QUERIES = 256
# prompt rows an index call scores at once (its fp32 scores are [rows, heads, context])
INDEX_QUERIES = 64
# the converted checkpoint's own block: per-module quantization and the Engram tables
FORMAT_KEY = "omlx_deepseek_v41"


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
    candidate_topk_blocks: int
    candidate_block_size: int
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
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    engram_layer_ids: list[int]
    engram_num_embeddings: list[int]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: list[int]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    eos_token_id: list[int] = field(default_factory=list)
    image_token_id: int = 129264

    # -- the layer layout -------------------------------------------------------------------
    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer]) if layer < len(self.compress_ratios) else 0

    def kv_source(self, layer: int) -> int | None:
        """The layer whose compressed KV this layer reads (the last KV source at or before it), None without."""

        if not self.ratio(layer) or layer >= self.num_hidden_layers:
            return None
        found = [s for s in self.kv_source_layer_ids if s <= layer]
        return max(found) if found else None

    def index_source(self, layer: int) -> int | None:
        """The layer whose top-k choice this layer attends with: the last index source at or after its KV source."""

        kv = self.kv_source(layer)
        if kv is None:
            return None
        found = [s for s in self.index_source_layer_ids if kv <= s <= layer]
        return max(found) if found else None

    def validate(self) -> None:
        """oMLX's checks (``ModelConfig.validate``): each compressed layer has a KV and index source of its ratio."""

        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError("deepseek_v41: compress_ratios must cover every layer")
        if any(r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError(f"deepseek_v41: compress_ratios {self.compress_ratios}; ratios 0, 1 and 2 are implemented")
        if self.num_attention_heads % self.o_groups or self.head_dim % 32 or self.index_head_dim % 32:
            raise ValueError("deepseek_v41: heads must divide into o_groups and head dims by 32")
        if self.hc_mult != 4:
            raise ValueError("deepseek_v41: the hyper-connections take 4 streams")
        for i in range(self.num_hidden_layers):
            r = self.ratio(i)
            if not r:
                continue
            kv, ix = self.kv_source(i), self.index_source(i)
            if kv is None or ix is None or self.ratio(kv) != r or self.ratio(ix) != r:
                raise ValueError(f"deepseek_v41: layer {i} has no compatible KV/index source")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("deepseek_v41: Engram layer and table counts differ")
        c = self.candidate_source_layer_id
        if c >= 0 and (c not in self.index_source_layer_ids or c not in self.kv_source_layer_ids):
            raise ValueError("deepseek_v41: the candidate source must be a KV and index source")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        if str(t.get("scoring_func", "sqrtsoftplus")) != "sqrtsoftplus":
            raise ValueError(f"deepseek_v41: routing scores {t.get('scoring_func')!r}; "
                             "only sqrtsoftplus is implemented")
        if int(t.get("num_key_value_heads", 1)) != 1 or int(t.get("n_shared_experts", 1)) != 1:
            raise ValueError("deepseek_v41: one shared key/value head and one shared expert are implemented")
        if not t.get("norm_topk_prob", True) or str(t.get("topk_method", "noaux_tc")) != "noaux_tc":
            raise ValueError("deepseek_v41: normalised noaux_tc routing is implemented")
        rope = dict(t.get("rope_scaling") or {})
        if rope and str(rope.get("type") or rope.get("rope_type")) not in ("yarn", "deepseek_yarn"):
            raise ValueError(f"deepseek_v41: rope_scaling {rope}; the compressed layers use YaRN")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        ints = lambda name: [int(v) for v in t.get(name) or []]  # noqa: E731
        cfg = cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=int(t["num_hidden_layers"]),
            vocab_size=int(t["vocab_size"]), rms_norm_eps=float(t.get("rms_norm_eps", 1e-20)),
            num_attention_heads=int(t["num_attention_heads"]), head_dim=int(t["head_dim"]),
            qk_rope_head_dim=int(t["qk_rope_head_dim"]), q_lora_rank=int(t["q_lora_rank"]),
            o_lora_rank=int(t["o_lora_rank"]), o_groups=int(t["o_groups"]),
            sliding_window=int(t.get("sliding_window", 128)), compress_ratios=ints("compress_ratios"),
            kv_source_layer_ids=ints("kv_source_layer_ids"), index_source_layer_ids=ints("index_source_layer_ids"),
            candidate_source_layer_id=int(t.get("candidate_source_layer_id", -1)),
            candidate_topk_blocks=int(t.get("candidate_topk_blocks", 0)),
            candidate_block_size=int(t.get("candidate_block_size", 0)),
            rope_theta=float(t.get("rope_theta", 10000.0)),
            compress_rope_theta=float(t.get("compress_rope_theta", 160000.0)),
            rope_factor=float(rope.get("factor", 1.0)),
            rope_original=int(rope.get("original_max_position_embeddings", 0)),
            beta_fast=float(rope.get("beta_fast", 32)), beta_slow=float(rope.get("beta_slow", 1)),
            index_n_heads=int(t["index_n_heads"]), index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]), n_routed_experts=int(t["n_routed_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]), moe_intermediate_size=int(t["moe_intermediate_size"]),
            routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            engram_layer_ids=ints("engram_layer_ids"), engram_num_embeddings=ints("engram_num_embeddings"),
            engram_max_ngram_size=int(t.get("engram_max_ngram_size", 1)),
            engram_vocab_size=int(t.get("engram_vocab_size", 0)), engram_n_heads=int(t.get("engram_n_heads", 0)),
            engram_head_dim=int(t.get("engram_head_dim", 0)), engram_pad_token_id=int(t.get("engram_pad_token_id", 2)),
            engram_compressed_vocab_size=int(t.get("engram_compressed_vocab_size", 0)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            dspark_block_size=int(t.get("dspark_block_size", 0)),
            dspark_noise_token_id=int(t.get("dspark_noise_token_id", 0)),
            dspark_target_layer_ids=ints("dspark_target_layer_ids"),
            dspark_markov_rank=int(t.get("dspark_markov_rank", 256)),
            dspark_n_routed_experts=int(t.get("dspark_n_routed_experts", 0) or t["n_routed_experts"]),
            dspark_num_experts_per_tok=int(t.get("dspark_num_experts_per_tok", 0) or t["num_experts_per_tok"]),
            eos_token_id=list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else []),
            image_token_id=int(config.get("image_token_id", t.get("image_token_id", 129264))))
        cfg.validate()
        return cfg

    def rope_params(self, layer: int) -> tuple:
        """The layer's RoPE as oMLX's ``rope`` passes it; compressed layers rotate at compress_rope_theta with YaRN."""

        compressed = bool(self.ratio(layer))
        return (self.qk_rope_head_dim, self.compress_rope_theta if compressed else self.rope_theta,
                self.rope_original, self.beta_fast, self.beta_slow, self.rope_factor, compressed)

    def inv_freq(self, layer: int) -> Any:
        """RoPE frequencies (fp32) as oMLX's ``_rope``: plain on window-only layers, YaRN on compressed ones."""

        import mlx.core as mx

        dims = self.qk_rope_head_dim
        compressed = bool(self.ratio(layer))
        base = self.compress_rope_theta if compressed else self.rope_theta
        freqs = 1 / mx.power(mx.array(base, dtype=mx.float32), mx.arange(0, dims, 2).astype(mx.float32) / dims)
        if not compressed or not self.rope_original:
            return freqs

        def correction(rotations: float) -> float:
            return dims * math.log(self.rope_original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(self.beta_fast)), 0)
        high = min(math.ceil(correction(self.beta_slow)), dims - 1)
        smooth = 1 - mx.clip((mx.arange(dims // 2) - low) / max(high - low, 1e-3), 0, 1)
        return freqs / self.rope_factor * (1 - smooth) + freqs * smooth
