"""DeepSeek-V4.1-Flash's settings from config.json and what each layer computes (no torch)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Role:
    """One layer's attention mode and the earlier layers whose state it reads."""

    ratio: int                  # positions per compressed KV entry (0: the window alone)
    mode: str                   # swa, full, reuse, reindex or dspark
    kv_src: int | None          # the layer whose compressed KV and index-K it attends
    idx_src: int | None         # the layer whose top-k list it attends (itself when it runs an indexer)
    candidate_source: bool      # picks the blocks later indexers score inside
    uses_candidates: bool       # scores only inside the candidate source's blocks
    engram: bool                # Engram is added to the stream before this block
    rope: str                   # local (rope_theta) or yarn (compress_rope_theta with YaRN)
    tap: int | None             # this layer's slot in the DSpark input


def yarn_bounds(dim: int, original: int, base: float, beta_fast: float, beta_slow: float) -> tuple[int, int]:
    """YaRN's (low, high) frequency indices, as the reference's precompute_freqs_cis computes them."""

    def corrected_dim(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    return max(math.floor(corrected_dim(beta_fast)), 0), min(math.ceil(corrected_dim(beta_slow)), dim - 1)


def roles(ratios: tuple[int, ...], n_layers: int, kv_sources: tuple[int, ...], index_sources: tuple[int, ...],
          candidate_source: int, engram_ids: tuple[int, ...], taps: tuple[int, ...]) -> tuple[Role, ...]:
    """Every layer's role, the DSpark stages after the backbone; a layer reads the latest source at or before it."""

    out = []
    for layer, ratio in enumerate(ratios):
        rope = "yarn" if ratio else "local"
        if layer >= n_layers:
            out.append(Role(ratio, "dspark", None, None, False, False, False, rope, None))
            continue
        kv = max((s for s in kv_sources if s <= layer), default=None) if ratio else None
        idx = max((s for s in index_sources if s <= layer), default=None) if ratio else None
        if ratio and (kv is None or idx is None):
            raise ValueError(f"layer {layer} compresses its KV but no KV or index source comes at or before it")
        mode = ("swa" if not ratio else "full" if layer in kv_sources else "reindex" if layer in index_sources
                else "reuse")
        out.append(Role(ratio, mode, kv, idx, layer == candidate_source,
                        mode == "reindex" and 0 <= candidate_source < layer, layer in engram_ids, rope,
                        taps.index(layer) if layer in taps else None))
    return tuple(out)


def _need(block: dict[str, Any], key: str, where: str) -> Any:
    if key not in block or block[key] is None:
        raise ValueError(f"config.json's {where} has no {key}, which the DeepSeek-V4.1-Flash engine reads")
    return block[key]


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    sliding_window: int
    rms_norm_eps: float
    max_position_embeddings: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int          # -1: no candidate pre-filtering
    candidate_topk_blocks: int
    candidate_block_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    scoring_func: str
    routed_scaling_factor: float
    swiglu_limit: float
    norm_topk_prob: bool
    gate_temp: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_compressed_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    image_token_id: int
    codebook: str                           # the routed experts' EXL3 codebook
    mtp_experts: str                        # source: the DSpark experts as the checkpoint ships them
    fp8_block: tuple[int, int]              # the non-routed FP8 weights' scale block
    expert_dtype: str                       # the DSpark experts' storage (fp4: MXFP4)
    roles: tuple[Role, ...]                 # backbone layers, then the DSpark stages

    @property
    def yarn(self) -> tuple[int, int]:
        """(low, high) of the YaRN blend the compressed layers' RoPE uses."""

        return yarn_bounds(self.qk_rope_head_dim, self.original_max_position_embeddings, self.compress_rope_theta,
                           self.beta_fast, self.beta_slow)

    @classmethod
    def read(cls, model_dir: str | Path, layers: int | None = None) -> Config:
        """The checkpoint's config; ``layers`` keeps the first N backbone layers (tests), DSpark tapping the last."""

        from tensorfold.families import read_config

        return cls.from_dict(read_config(model_dir), layers)

    @classmethod
    def from_dict(cls, config: dict[str, Any], layers: int | None = None) -> Config:
        t = config.get("text_config") or config
        q = config.get("quantization_config") or {}
        dense = q.get("non_routed_quantization") or {}

        def need(key: str) -> Any:
            return _need(t, key, "text_config")

        def top(key: str) -> int:
            return int(_need(config, key, "top level") if key in config else need(key))

        scaling = need("rope_scaling")
        total = int(need("num_hidden_layers"))
        stages = int(need("num_nextn_predict_layers"))
        ratios = tuple(int(r) for r in need("compress_ratios"))
        if len(ratios) != total + stages:
            raise ValueError(f"config.json's compress_ratios has {len(ratios)} entries, not one per layer and DSpark "
                             f"stage ({total} + {stages})")
        kv = tuple(int(i) for i in need("kv_source_layer_ids"))
        index = tuple(int(i) for i in need("index_source_layer_ids"))
        candidate = int(need("candidate_source_layer_id"))
        engram = tuple(int(i) for i in need("engram_layer_ids"))
        rows = tuple(int(n) for n in need("engram_num_embeddings"))
        taps = tuple(int(i) for i in need("dspark_target_layer_ids"))
        n = total if layers is None else int(layers)
        if not len(taps) <= n <= total:
            raise ValueError(f"layers={n}: from {len(taps)} (the DSpark taps) to {total}")
        if n < total:                       # a prefix of the backbone: sources past it drop, the taps move to its end
            ratios = ratios[:n] + ratios[total:]
            kv, index = tuple(i for i in kv if i < n), tuple(i for i in index if i < n)
            candidate = candidate if candidate < n else -1
            rows = tuple(r for i, r in zip(engram, rows) if i < n)
            engram = tuple(i for i in engram if i < n)
            taps = tuple(range(n - len(taps), n))
        return cls(
            vocab_size=int(need("vocab_size")), hidden_size=int(need("hidden_size")),
            moe_intermediate_size=int(need("moe_intermediate_size")), num_hidden_layers=n,
            num_nextn_predict_layers=stages, num_attention_heads=int(need("num_attention_heads")),
            head_dim=int(need("head_dim")), qk_rope_head_dim=int(need("qk_rope_head_dim")),
            q_lora_rank=int(need("q_lora_rank")), o_groups=int(need("o_groups")), o_lora_rank=int(need("o_lora_rank")),
            sliding_window=int(need("sliding_window")), rms_norm_eps=float(need("rms_norm_eps")),
            max_position_embeddings=int(need("max_position_embeddings")), compress_ratios=ratios,
            kv_source_layer_ids=kv, index_source_layer_ids=index, candidate_source_layer_id=candidate,
            candidate_topk_blocks=int(need("candidate_topk_blocks")),
            candidate_block_size=int(need("candidate_block_size")), index_n_heads=int(need("index_n_heads")),
            index_head_dim=int(need("index_head_dim")), index_topk=int(need("index_topk")),
            rope_theta=float(need("rope_theta")), compress_rope_theta=float(need("compress_rope_theta")),
            rope_factor=float(_need(scaling, "factor", "rope_scaling")),
            original_max_position_embeddings=int(_need(scaling, "original_max_position_embeddings", "rope_scaling")),
            beta_fast=float(_need(scaling, "beta_fast", "rope_scaling")),
            beta_slow=float(_need(scaling, "beta_slow", "rope_scaling")),
            hc_mult=int(need("hc_mult")), hc_sinkhorn_iters=int(need("hc_sinkhorn_iters")),
            hc_eps=float(need("hc_eps")),
            n_routed_experts=int(need("n_routed_experts")), num_experts_per_tok=int(need("num_experts_per_tok")),
            n_shared_experts=int(need("n_shared_experts")), scoring_func=str(need("scoring_func")),
            routed_scaling_factor=float(need("routed_scaling_factor")), swiglu_limit=float(need("swiglu_limit")),
            norm_topk_prob=bool(need("norm_topk_prob")), gate_temp=float(t.get("gate_temp", 1.0)),
            engram_layer_ids=engram, engram_num_embeddings=rows,
            engram_max_ngram_size=int(need("engram_max_ngram_size")), engram_vocab_size=int(need("engram_vocab_size")),
            engram_compressed_vocab_size=int(need("engram_compressed_vocab_size")),
            engram_n_heads=int(need("engram_n_heads")), engram_head_dim=int(need("engram_head_dim")),
            engram_pad_token_id=int(need("engram_pad_token_id")), dspark_block_size=int(need("dspark_block_size")),
            dspark_noise_token_id=int(need("dspark_noise_token_id")), dspark_target_layer_ids=taps,
            dspark_markov_rank=int(need("dspark_markov_rank")),
            dspark_n_routed_experts=int(need("dspark_n_routed_experts")),
            dspark_num_experts_per_tok=int(need("dspark_num_experts_per_tok")),
            bos_token_id=top("bos_token_id"), eos_token_id=top("eos_token_id"), pad_token_id=top("pad_token_id"),
            image_token_id=top("image_token_id"),
            codebook=str(_need(q, "codebook", "quantization_config")),
            mtp_experts=str(_need(q, "mtp_experts", "quantization_config")),
            fp8_block=tuple(int(b) for b in _need(dense, "weight_block_size", "non_routed_quantization")),
            expert_dtype=str(_need(dense, "expert_dtype", "non_routed_quantization")),
            roles=roles(ratios, n, kv, index, candidate, engram, taps))
