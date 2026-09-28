"""GLM-5.3-Flash's settings from the checkpoint's config.json, and the decode path's switches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# the decode path's widest call: wider inputs (prompt chunks) take MLX's batched prefill path
DECODE_ROWS = 16

# Decode steps that take a window's rows in one kernel, each row keeping its one-row call's bits (tests switch them
# off to compare with one MLX call a row)
ROW_KERNELS = ("experts", "router", "hc", "igate", "kda_proj", "mla_proj", "indexer")
ENABLED = frozenset(ROW_KERNELS)
# the whole KDA step in one launch (kernels/glm/flash/v1/kda.py) and sparse MLA reading its chosen keys by index
# (sparse_attention.py): these set the decode path's arithmetic
FUSED_KDA = True
SPARSE_KERNEL = True
# the MoE block and each hyper-connection boundary as fused kernels (moe.py, hc.py), each with the row-by-row bits
FUSED_KERNELS = ("moe", "hc")
FUSED = frozenset(FUSED_KERNELS)
# the decode graph goes to the GPU every this many layers, so the GPU starts while Python builds the rest
EVAL_EVERY = 2


def row_kernel(name: str, rows: int, rows_exact: bool) -> bool:
    return rows_exact and rows > 1 and name in ENABLED


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    mlp_layer_types: list[str]
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_tail: bool
    linear_num_heads: int
    linear_head_dim: int
    linear_conv: int
    linear_lower_bound: float
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    first_k_dense_replace: int
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    num_nextn_predict_layers: int
    eos_token_id: list[int]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        lin = t.get("linear_attn_config") or {}
        if int(t.get("n_group", 1)) != 1 or int(t.get("topk_group", 1)) != 1:
            raise ValueError("glm5_next: grouped expert selection (n_group > 1) is not implemented")
        if int(t.get("qk_rope_head_dim", 0)) or not t.get("mla_use_nope", True):
            raise ValueError("glm5_next: only NoPE MLA (qk_rope_head_dim 0) is implemented")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=int(t["num_hidden_layers"]),
            layer_types=list(t["layer_types"]), mlp_layer_types=list(t["mlp_layer_types"]),
            vocab_size=int(t["vocab_size"]), rms_norm_eps=float(t["rms_norm_eps"]),
            num_attention_heads=int(t["num_attention_heads"]), q_lora_rank=int(t["q_lora_rank"]),
            kv_lora_rank=int(t["kv_lora_rank"]), qk_nope_head_dim=int(t["qk_nope_head_dim"]),
            v_head_dim=int(t["v_head_dim"]), index_n_heads=int(t["index_n_heads"]),
            index_head_dim=int(t["index_head_dim"]), index_topk=int(t["index_topk"]),
            index_kpool=int(t.get("index_kpool", 4)), index_tail=bool(t.get("index_kpool_always_select_tail", True)),
            linear_num_heads=int(lin.get("num_heads", 64)), linear_head_dim=int(lin.get("head_dim", 128)),
            linear_conv=int(lin.get("short_conv_kernel_size", 4)),
            linear_lower_bound=float(lin.get("gate_lower_bound", -5.0)),
            n_routed_experts=int(t["n_routed_experts"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]), intermediate_size=int(t["intermediate_size"]),
            n_shared_experts=int(t.get("n_shared_experts") or 0),
            routed_scaling_factor=float(t["routed_scaling_factor"]), norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            first_k_dense_replace=int(t.get("first_k_dense_replace", 0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            eos_token_id=list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else []))

