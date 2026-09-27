"""A tiny random GLM-5.3-Flash checkpoint, written in the real checkpoint's layout (4-bit, groups of 64)."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

D, VOCAB = 128, 256
TEXT = {
    "model_type": "glm5_next_text", "hidden_size": D, "num_hidden_layers": 6, "vocab_size": VOCAB,
    "rms_norm_eps": 1e-5,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                    "linear_attention", "deepseek_sparse_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
    "first_k_dense_replace": 1, "intermediate_size": 128, "moe_intermediate_size": 64,
    "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1, "routed_scaling_factor": 2.5,
    "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "swiglu_limit": 10.0,
    "num_attention_heads": 2, "num_key_value_heads": 2, "q_lora_rank": 64, "kv_lora_rank": 128,
    "qk_nope_head_dim": 64, "qk_rope_head_dim": 0, "v_head_dim": 64, "mla_use_nope": True,
    "index_n_heads": 2, "index_head_dim": 64, "index_topk": 16, "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "linear_attn_config": {"num_heads": 2, "head_dim": 64, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
    "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20, "num_nextn_predict_layers": 1,
    "eos_token_id": [5],
}


def _q(tensors: dict, name: str, outs: int, ins: int, scale: float = 0.08) -> None:
    w = scale * mx.random.normal((outs, ins))
    q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4)
    tensors[f"{name}.weight"], tensors[f"{name}.scales"], tensors[f"{name}.biases"] = q, s, b


def _norm(tensors: dict, name: str, n: int) -> None:
    tensors[name] = (1.0 + 0.1 * mx.random.normal((n,))).astype(mx.bfloat16)


def _mla(t: dict, p: str) -> None:
    c = TEXT
    h, nope, v, rank = c["num_attention_heads"], c["qk_nope_head_dim"], c["v_head_dim"], c["kv_lora_rank"]
    _q(t, f"{p}.q_a_proj", c["q_lora_rank"], D)
    _norm(t, f"{p}.q_a_layernorm.weight", c["q_lora_rank"])
    _q(t, f"{p}.q_b_proj", h * nope, c["q_lora_rank"])
    _q(t, f"{p}.kv_a_proj_with_mqa", rank, D)
    _norm(t, f"{p}.kv_a_layernorm.weight", rank)
    _q(t, f"{p}.kv_b_proj", h * (nope + v), rank)
    _q(t, f"{p}.o_proj", D, h * v)
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    _q(t, f"{p}.indexer.wq_b", ih * idim, c["q_lora_rank"], 0.3)
    _q(t, f"{p}.indexer.wk", idim, D, 0.3)
    _q(t, f"{p}.indexer.weights_proj", ih, D, 0.3)
    _norm(t, f"{p}.indexer.k_norm.weight", idim)
    t[f"{p}.indexer.k_norm.bias"] = (0.05 * mx.random.normal((idim,))).astype(mx.bfloat16)
    t[f"{p}.indexer.index_kpool_compress_ape"] = (0.3 * mx.random.normal((c["index_kpool"], idim))).astype(mx.bfloat16)
    t[f"{p}.indexer.index_kpool_compress_gate"] = (0.1 * mx.random.normal((idim, D))).astype(mx.bfloat16)


def _kda(t: dict, p: str) -> None:
    lin = TEXT["linear_attn_config"]
    heads, dim = lin["num_heads"], lin["head_dim"]
    width = heads * dim
    for n in ("q_proj", "k_proj", "v_proj"):
        _q(t, f"{p}.{n}", width, D)
        t[f"{p}.{n[0]}_conv1d.weight"] = (0.4 * mx.random.normal((width, 1, 4))).astype(mx.bfloat16)
    _q(t, f"{p}.f_a_proj", dim, D)
    _q(t, f"{p}.f_b_proj", width, dim)
    _q(t, f"{p}.g_a_proj", dim, D)
    _q(t, f"{p}.g_b_proj", width, dim)
    _q(t, f"{p}.b_proj", heads, D)
    _q(t, f"{p}.o_proj", D, width)
    _norm(t, f"{p}.o_norm.weight", dim)
    t[f"{p}.A_log"] = (0.5 * mx.random.normal((heads,))).astype(mx.float32)
    t[f"{p}.dt_bias"] = (0.5 * mx.random.normal((width,))).astype(mx.float32)


def _mlp(t: dict, p: str, sparse: bool) -> None:
    c = TEXT
    if not sparse:
        for n in ("gate_proj", "up_proj"):
            _q(t, f"{p}.{n}", c["intermediate_size"], D)
        _q(t, f"{p}.down_proj", D, c["intermediate_size"])
        return
    t[f"{p}.gate.weight"] = (0.3 * mx.random.normal((c["n_routed_experts"], D))).astype(mx.float32)
    t[f"{p}.gate.e_score_correction_bias"] = (0.1 * mx.random.normal((c["n_routed_experts"],))).astype(mx.float32)
    width = c["moe_intermediate_size"]
    for e in range(c["n_routed_experts"]):
        _q(t, f"{p}.experts.{e}.gate_proj", width, D)
        _q(t, f"{p}.experts.{e}.up_proj", width, D)
        _q(t, f"{p}.experts.{e}.down_proj", D, width)
    _q(t, f"{p}.shared_experts.gate_proj", width, D)
    _q(t, f"{p}.shared_experts.up_proj", width, D)
    _q(t, f"{p}.shared_experts.down_proj", D, width)


def write_checkpoint(folder: Path, seed: int = 0, *, mtp: bool = True) -> Path:
    mx.random.seed(seed)
    c = TEXT
    t: dict = {}
    pre = "model.language_model"
    _q(t, f"{pre}.embed_tokens", VOCAB, D, 1.0)
    _q(t, "lm_head", VOCAB, D, 0.3)
    _norm(t, f"{pre}.norm.weight", D)
    hc = c["hc_mult"]
    mix = (2 + hc) * hc
    for i in range(c["num_hidden_layers"]):
        p = f"{pre}.layers.{i}"
        (_kda if c["layer_types"][i] == "linear_attention" else _mla)(t, f"{p}.self_attn")
        _mlp(t, f"{p}.mlp", c["mlp_layer_types"][i] == "sparse")
        _norm(t, f"{p}.input_layernorm.weight", D)
        _norm(t, f"{p}.post_attention_layernorm.weight", D)
        for kind in ("attn", "ffn"):
            t[f"{p}.hc_{kind}_fn"] = (0.05 * mx.random.normal((mix, hc * D))).astype(mx.float32)
            t[f"{p}.hc_{kind}_base"] = (0.3 * mx.random.normal((mix,))).astype(mx.float32)
            t[f"{p}.hc_{kind}_scale"] = mx.array([0.5, 0.5, 0.5], dtype=mx.float32)
    if mtp:
        p = f"{pre}.layers.{c['num_hidden_layers']}"
        _mla(t, f"{p}.self_attn")
        _mlp(t, f"{p}.mlp", True)
        _norm(t, f"{p}.input_layernorm.weight", D)
        _norm(t, f"{p}.post_attention_layernorm.weight", D)
        _q(t, f"{p}.eh_proj", D, 2 * D)
        for n in ("enorm", "hnorm", "shared_head.norm"):
            _norm(t, f"{p}.{n}.weight", D)
    mx.eval(t)
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(folder / shard), {k: t[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    config = {"model_type": "glm5_next", "text_config": TEXT, "quantization": {"bits": 4, "group_size": 64}}
    (folder / "config.json").write_text(json.dumps(config))
    return folder
