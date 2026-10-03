"""A tiny random DeepSeek-V4.1-Flash checkpoint in oMLX's converted layout."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx

D, VOCAB = 128, 256
LAYERS, MTP = 6, 3
TOKEN_MAP = [i % 200 for i in range(VOCAB)]        # the "tokenizer's" compressed ids: 200 of them


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def engram_rows(layer_ids: list[int], max_ngram: int, heads: int, base: int) -> list[int]:
    """Each table's rows: the sum of its distinct primes from ``base`` (oMLX's NgramHash layout)."""

    seen, out = set(), []
    for _ in layer_ids:
        total = 0
        for _ in range(max_ngram - 1):
            current = base - 1
            for _ in range(heads):
                current += 1
                while not _prime(current) or current in seen:
                    current += 1
                seen.add(current)
                total += current
        out.append(total)
    return out


ENGRAM = {"engram_layer_ids": [1, 3], "engram_max_ngram_size": 3, "engram_vocab_size": 50, "engram_n_heads": 2,
          "engram_head_dim": 32, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 200}
ENGRAM["engram_num_embeddings"] = engram_rows(ENGRAM["engram_layer_ids"], 3, 2, 50)
TEXT = {
    "model_type": "deepseek_v41_text", "vocab_size": VOCAB, "hidden_size": D, "moe_intermediate_size": 64,
    "num_hidden_layers": LAYERS, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
    "qk_rope_head_dim": 16, "q_lora_rank": 64, "o_lora_rank": 32, "o_groups": 2, "hidden_act": "silu",
    "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "max_position_embeddings": 1048576, "rope_theta": 10000,
    "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                     "original_max_position_embeddings": 65536},
    "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc", "norm_topk_prob": True, "routed_scaling_factor": 1.5, "sliding_window": 8,
    "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0], "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 4], "index_source_layer_ids": [2, 4, 5], "index_n_heads": 2, "index_head_dim": 32,
    "index_topk": 4, "candidate_source_layer_id": 4, "candidate_topk_blocks": 2, "candidate_block_size": 4,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, **ENGRAM,
    "num_nextn_predict_layers": MTP, "dspark_block_size": 4, "dspark_noise_token_id": 250,
    "dspark_target_layer_ids": [3, 4, 5], "dspark_markov_rank": 16, "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 1,
}


class Writer:
    def __init__(self) -> None:
        self.t: dict[str, mx.array] = {}
        self.quant: dict[str, dict] = {}

    def mx8(self, name: str, outs: int, ins: int, scale: float = 0.08) -> None:
        w = (scale * mx.random.normal((outs, ins))).astype(mx.bfloat16)
        self.t[f"{name}.weight"], self.t[f"{name}.scales"] = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")[:2]
        self.quant[name] = {"bits": 8, "mode": "mxfp8"}

    def fp4(self, name: str, experts: int, outs: int, ins: int, scale: float = 0.08) -> None:
        w = (scale * mx.random.normal((experts, outs, ins))).astype(mx.bfloat16)
        self.t[f"{name}.weight"], self.t[f"{name}.scales"] = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")[:2]
        self.quant[name] = {"bits": 4, "mode": "mxfp4"}

    def bf(self, name: str, outs: int, ins: int, scale: float = 0.08) -> None:
        self.t[name] = (scale * mx.random.normal((outs, ins))).astype(mx.bfloat16)

    def norm(self, name: str, n: int) -> None:
        self.t[name] = (1.0 + 0.1 * mx.random.normal((n,))).astype(mx.bfloat16)

    def hc(self, p: str) -> None:
        for kind in ("attn", "ffn"):
            self.t[f"{p}.hc_{kind}_fn"] = 0.02 * mx.random.normal((24, 4 * D))
            self.t[f"{p}.hc_{kind}_base"] = 0.1 * mx.random.normal((24,))
            self.t[f"{p}.hc_{kind}_scale"] = 1.0 + 0.1 * mx.random.normal((3,))

    def block(self, p: str, layer: int, experts: int) -> None:
        c = TEXT
        h, hd, g = c["num_attention_heads"], c["head_dim"], c["o_groups"]
        ratio = c["compress_ratios"][layer]
        self.hc(p)
        self.norm(f"{p}.attn_norm.weight", D)
        self.norm(f"{p}.ffn_norm.weight", D)
        a = f"{p}.attn"
        self.t[f"{a}.attn_sink"] = 0.5 * mx.random.normal((h,))
        self.mx8(f"{a}.wq_a", c["q_lora_rank"], D)
        self.norm(f"{a}.q_norm.weight", c["q_lora_rank"])
        self.mx8(f"{a}.wq_b", h * hd, c["q_lora_rank"])
        self.mx8(f"{a}.wkv", hd, D)
        self.norm(f"{a}.kv_norm.weight", hd)
        self.bf(f"{a}.wo_a.weight", g * c["o_lora_rank"], h * hd // g)
        self.mx8(f"{a}.wo_b", D, g * c["o_lora_rank"])
        if ratio and layer in c["kv_source_layer_ids"]:
            self.bf(f"{a}.compressor.wkv.weight", hd, D, 0.3)
            if ratio > 1:
                self.bf(f"{a}.compressor.wgate.weight", hd, D, 0.3)
            self.norm(f"{a}.compressor.norm.weight", hd)
        if ratio and layer in c["index_source_layer_ids"]:
            self.mx8(f"{a}.indexer.wq_b", c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"], 0.3)
            self.bf(f"{a}.indexer.weights_proj.weight", c["index_n_heads"], D, 0.3)
            if layer in c["kv_source_layer_ids"]:
                self.bf(f"{a}.indexer.wk.weight", c["index_head_dim"], hd, 0.3)
                self.norm(f"{a}.indexer.k_norm.weight", c["index_head_dim"])
        f = f"{p}.ffn"
        inter = c["moe_intermediate_size"]
        self.bf(f"{f}.gate.weight", experts, D, 0.3)
        self.t[f"{f}.gate.bias"] = 0.1 * mx.random.normal((experts,))
        self.mx8(f"{f}.shared_experts.w1", inter, D)
        self.mx8(f"{f}.shared_experts.w3", inter, D)
        self.mx8(f"{f}.shared_experts.w2", D, inter)
        self.fp4(f"{f}.experts.w1", experts, inter, D)
        self.fp4(f"{f}.experts.w3", experts, inter, D)
        self.fp4(f"{f}.experts.w2", experts, D, inter)


def write_checkpoint(folder: Path, seed: int = 0, mtp: bool = True) -> Path:
    mx.random.seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    w = Writer()
    c = TEXT
    L = "language_model"
    w.bf(f"{L}.embed.weight", VOCAB, D, 1.0)
    w.bf(f"{L}.head.weight", VOCAB, D)
    w.norm(f"{L}.norm.weight", D)
    for i in range(LAYERS):
        w.block(f"{L}.layers.{i}", i, c["n_routed_experts"])
    if mtp:
        for s in range(MTP):
            p = f"{L}.mtp.{s}"
            w.block(p, LAYERS + s, c["dspark_n_routed_experts"])
            if s == 0:
                w.mx8(f"{p}.main_proj", D, D * len(c["dspark_target_layer_ids"]))
                w.norm(f"{p}.main_norm.weight", D)
            if s == MTP - 1:
                w.norm(f"{p}.norm.weight", D)
                rank = c["dspark_markov_rank"]
                w.bf(f"{p}.markov_head.embed.weight", VOCAB, rank, 0.5)
                w.bf(f"{p}.markov_head.head.weight", VOCAB, rank, 0.5)
                w.bf(f"{p}.confidence_head.proj.weight", 1, D + rank, 0.1)
    tables = {}
    e = ENGRAM
    width = (e["engram_max_ngram_size"] - 1) * e["engram_n_heads"] * e["engram_head_dim"]
    engram_shard = "model-00001-of-00003.safetensors"
    engram_t: dict[str, mx.array] = {}
    for layer, rows in zip(e["engram_layer_ids"], e["engram_num_embeddings"]):
        p = f"{L}.layers.{layer}.engram"
        table = (0.5 * mx.random.normal((rows, e["engram_head_dim"]))).astype(mx.bfloat16)
        q, s, b = mx.quantize(table, group_size=32, bits=4)
        engram_t[f"{p}.embed.weight"], engram_t[f"{p}.embed.scales"], engram_t[f"{p}.embed.biases"] = q, s, b
        w.mx8(f"{p}.wkv", D * (c["hc_mult"] + 1), width)
        w.t[f"{p}.q_weight"] = (1.0 + 0.1 * mx.random.normal((c["hc_mult"], D))).astype(mx.bfloat16)
        w.t[f"{p}.k_weight"] = (1.0 + 0.1 * mx.random.normal((c["hc_mult"], D))).astype(mx.bfloat16)
        tables[f"{p}.embed"] = {"weight_file": engram_shard, "weight_key": f"{p}.embed.weight",
                                "scale_file": engram_shard, "scale_key": f"{p}.embed.scales",
                                "bias_key": f"{p}.embed.biases", "bits": 4, "group_size": 32, "mode": "affine"}
    names = sorted(w.t)
    half = len(names) // 2
    shards = {"model-00002-of-00003.safetensors": names[:half], "model-00003-of-00003.safetensors": names[half:]}
    weight_map = {}
    mx.save_safetensors(str(folder / engram_shard), engram_t)
    weight_map.update({k: engram_shard for k in engram_t})
    for shard, keys in shards.items():
        mx.save_safetensors(str(folder / shard), {k: w.t[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
              "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 129264, "text_config": TEXT,
              "omlx_deepseek_v41": {"version": 1, "quantized_modules": w.quant, "engram_tables": tables,
                                    "preserve_mtp": bool(mtp), "engram_in_index": True}}
    (folder / "config.json").write_text(json.dumps(config))
    return folder
