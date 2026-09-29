"""A tiny random DeepSeek-V4-Flash checkpoint in the mlx-community layout (affine 4-bit g64, mxfp4 experts)."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

D, VOCAB = 128, 256
TEXT = {
    "model_type": "deepseek_v4", "hidden_size": D, "num_hidden_layers": 5, "vocab_size": VOCAB,
    "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
    "qk_rope_head_dim": 32, "q_lora_rank": 64, "o_lora_rank": 64, "o_groups": 2, "sliding_window": 8,
    "compress_ratios": [0, 4, 128, 4, 128, 0], "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
    "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32,
                     "beta_slow": 1},
    "index_n_heads": 2, "index_head_dim": 64, "index_topk": 4, "n_routed_experts": 8, "n_shared_experts": 1,
    "num_experts_per_tok": 2, "moe_intermediate_size": 64, "num_hash_layers": 1, "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
    "num_nextn_predict_layers": 1, "eos_token_id": 1, "bos_token_id": 0,
}


def _q(t: dict, name: str, outs: int, ins: int, scale: float = 0.08) -> None:
    w = (scale * mx.random.normal((outs, ins))).astype(mx.bfloat16)
    t[f"{name}.weight"], t[f"{name}.scales"], t[f"{name}.biases"] = mx.quantize(w, group_size=64, bits=4)


def _fp4(t: dict, name: str, experts: int, outs: int, ins: int, scale: float = 0.08) -> None:
    w = (scale * mx.random.normal((experts, outs, ins))).astype(mx.bfloat16)
    t[f"{name}.weight"], t[f"{name}.scales"] = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")


def _norm(t: dict, name: str, n: int) -> None:
    t[name] = (1.0 + 0.1 * mx.random.normal((n,))).astype(mx.bfloat16)


def _hc(t: dict, name: str, mixes: int) -> None:
    t[f"{name}.fn"] = 0.02 * mx.random.normal((mixes, 4 * D))
    t[f"{name}.base"] = 0.1 * mx.random.normal((mixes,))
    t[f"{name}.scale"] = 1.0 + 0.1 * mx.random.normal((3 if mixes > 4 else 1,))


def _compressor(t: dict, p: str, ratio: int, dim: int) -> None:
    width = dim * (2 if ratio == 4 else 1)
    _q(t, f"{p}.wkv", width, D, 0.3)
    _q(t, f"{p}.wgate", width, D, 0.3)
    t[f"{p}.ape"] = (0.1 * mx.random.normal((ratio, width))).astype(mx.bfloat16)
    _norm(t, f"{p}.norm.weight", dim)


def _layer(t: dict, i: int) -> None:
    _block(t, f"model.layers.{i}", TEXT["compress_ratios"][i], i < TEXT["num_hash_layers"])


def _block(t: dict, p: str, ratio: int, hashed: bool) -> None:
    c = TEXT
    h, hd, g = c["num_attention_heads"], c["head_dim"], c["o_groups"]
    for sub in ("attn_hc", "ffn_hc"):
        _hc(t, f"{p}.{sub}", 24)
    _norm(t, f"{p}.attn_norm.weight", D)
    _norm(t, f"{p}.ffn_norm.weight", D)
    a = f"{p}.attn"
    _q(t, f"{a}.wq_a", c["q_lora_rank"], D)
    _norm(t, f"{a}.q_norm.weight", c["q_lora_rank"])
    _q(t, f"{a}.wq_b", h * hd, c["q_lora_rank"])
    _q(t, f"{a}.wkv", hd, D)
    _norm(t, f"{a}.kv_norm.weight", hd)
    _q(t, f"{a}.wo_a", g * c["o_lora_rank"], h * hd // g)
    _q(t, f"{a}.wo_b", D, g * c["o_lora_rank"])
    t[f"{a}.attn_sink"] = 0.5 * mx.random.normal((h,))
    if ratio:
        _compressor(t, f"{a}.compressor", ratio, hd)
    if ratio == 4:
        _q(t, f"{a}.indexer.wq_b", c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"], 0.3)
        _q(t, f"{a}.indexer.weights_proj", c["index_n_heads"], D, 0.3)
        _compressor(t, f"{a}.indexer.compressor", ratio, c["index_head_dim"])
    f = f"{p}.ffn"
    e, inter = c["n_routed_experts"], c["moe_intermediate_size"]
    t[f"{f}.gate.weight"] = (0.3 * mx.random.normal((e, D))).astype(mx.bfloat16)
    if hashed:
        t[f"{f}.gate.tid2eid"] = mx.array([[(v + k) % e for k in range(c["num_experts_per_tok"])]
                                           for v in range(VOCAB)], dtype=mx.int64)
    else:
        t[f"{f}.gate.e_score_correction_bias"] = 0.1 * mx.random.normal((e,))
    _q(t, f"{f}.shared_experts.gate_proj", inter, D)
    _q(t, f"{f}.shared_experts.up_proj", inter, D)
    _q(t, f"{f}.shared_experts.down_proj", D, inter)
    _fp4(t, f"{f}.switch_mlp.gate_proj", e, inter, D)
    _fp4(t, f"{f}.switch_mlp.up_proj", e, inter, D)
    _fp4(t, f"{f}.switch_mlp.down_proj", e, D, inter)


def write_checkpoint(folder: Path, seed: int = 0) -> Path:
    mx.random.seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t: dict = {}
    _q(t, "model.embed_tokens", VOCAB, D, 1.0)
    _q(t, "lm_head", VOCAB, D)
    _norm(t, "model.norm.weight", D)
    _hc(t, "model.hc_head", 4)
    for i in range(TEXT["num_hidden_layers"]):
        _layer(t, i)
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(folder / shard), {k: t[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(TEXT["num_hidden_layers"]):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quant[f"model.layers.{i}.ffn.switch_mlp.{proj}"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    (folder / "config.json").write_text(json.dumps({**TEXT, "quantization": quant, "quantization_config": quant}))
    return folder


def write_mtp(folder: Path, seed: int = 1) -> Path:
    """The converted MTP layer (``mtp.*`` names, as convert.convert_mtp writes it) in a drafter folder."""

    mx.random.seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t: dict = {}
    _q(t, "mtp.e_proj", D, D)
    _q(t, "mtp.h_proj", D, D)
    for name in ("mtp.enorm.weight", "mtp.hnorm.weight", "mtp.norm.weight"):
        _norm(t, name, D)
    _hc(t, "mtp.hc_head", 4)
    _block(t, "mtp", TEXT["compress_ratios"][TEXT["num_hidden_layers"]], False)
    mx.save_safetensors(str(folder / "mtp.safetensors"), t)
    return folder


def write_official_mtp(path: Path, seed: int = 2) -> dict:
    """A tiny official MTP shard: e4m3 linears with E8M0 128-blocks, e2m1 experts packed two a byte, E8M0 per 32."""

    mx.random.seed(seed)
    t = _official_block("mtp.0", {"e_proj": (D, D), "h_proj": (D, D)}, ("enorm.weight", "hnorm.weight", "norm.weight"))
    mx.save_safetensors(str(path), t)
    return t


def write_official_dspark(folder: Path, blocks: int = 2, seed: int = 4) -> list[Path]:
    """Tiny official DSpark shards, one block each; the first has main_proj, the last the head and Markov tables."""

    mx.random.seed(seed)
    taps = len(DSPARK["dspark_target_layer_ids"])
    paths = []
    for i in range(blocks):
        extra = {"main_proj": (D, taps * D)} if i == 0 else {}
        plain = ("main_norm.weight",) if i == 0 else ()
        t = _official_block(f"mtp.{i}", extra, plain, head=i == blocks - 1)
        if i == blocks - 1:
            rank = DSPARK["dspark_markov_rank"]
            for name in ("markov_w1", "markov_w2"):
                t[f"mtp.{i}.markov_head.{name}.weight"] = (0.5 * mx.random.normal((VOCAB, rank))).astype(mx.bfloat16)
        paths.append(folder / f"model-{i:05d}.safetensors")
        folder.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(paths[-1]), t)
    return paths


def _official_block(p: str, extra: dict, plain: tuple, head: bool = True) -> dict:
    c = TEXT
    t: dict = {}
    dense = {**extra, "attn.wq_a": (c["q_lora_rank"], D),
             "attn.wq_b": (c["num_attention_heads"] * c["head_dim"], c["q_lora_rank"]), "attn.wkv": (c["head_dim"], D),
             "attn.wo_a": (c["o_groups"] * c["o_lora_rank"], c["num_attention_heads"] * c["head_dim"] // c["o_groups"]),
             "attn.wo_b": (D, c["o_groups"] * c["o_lora_rank"]),
             "ffn.shared_experts.w1": (c["moe_intermediate_size"], D),
             "ffn.shared_experts.w3": (c["moe_intermediate_size"], D),
             "ffn.shared_experts.w2": (D, c["moe_intermediate_size"])}
    for name, (o, i) in dense.items():
        t[f"{p}.{name}.weight"] = mx.to_fp8(2.0 * mx.random.normal((o, i)))
        t[f"{p}.{name}.scale"] = mx.random.randint(118, 124, (-(-o // 128), -(-i // 128))).astype(mx.uint8)
    for name, (o, i) in {"w1": (c["moe_intermediate_size"], D), "w3": (c["moe_intermediate_size"], D),
                         "w2": (D, c["moe_intermediate_size"])}.items():
        for e in range(c["n_routed_experts"]):
            t[f"{p}.ffn.experts.{e}.{name}.weight"] = mx.random.randint(0, 256, (o, i // 2)).astype(mx.uint8)
            t[f"{p}.ffn.experts.{e}.{name}.scale"] = mx.random.randint(118, 124, (o, i // 32)).astype(mx.uint8)
    for name in ("attn_norm.weight", "ffn_norm.weight", *plain) + (("norm.weight",) if head else ()):
        t[f"{p}.{name}"] = mx.ones((D,), dtype=mx.bfloat16)
    t[f"{p}.attn.q_norm.weight"] = mx.ones((c["q_lora_rank"],), dtype=mx.bfloat16)
    t[f"{p}.attn.kv_norm.weight"] = mx.ones((c["head_dim"],), dtype=mx.bfloat16)
    t[f"{p}.attn.attn_sink"] = mx.zeros((c["num_attention_heads"],))
    t[f"{p}.ffn.gate.weight"] = (0.3 * mx.random.normal((c["n_routed_experts"], D))).astype(mx.bfloat16)
    t[f"{p}.ffn.gate.bias"] = mx.zeros((c["n_routed_experts"],))
    for sub_name, mixes in (("hc_attn", 24), ("hc_ffn", 24)) + ((("hc_head", 4),) if head else ()):
        t[f"{p}.{sub_name}_fn"] = 0.02 * mx.random.normal((mixes, 4 * D))
        t[f"{p}.{sub_name}_base"] = mx.zeros((mixes,))
        t[f"{p}.{sub_name}_scale"] = mx.ones((3 if mixes > 4 else 1,))
    return t


DSPARK = {"dspark_block_size": 4, "dspark_noise_token_id": 250, "dspark_target_layer_ids": [2, 3, 4],
          "dspark_markov_rank": 16}


def write_dspark(folder: Path, blocks: int = 2, seed: int = 3) -> Path:
    """The converted DSpark drafter (``dspark.<i>.*``, as convert.convert_dspark writes it) and its config."""

    mx.random.seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t: dict = {}
    for i in range(blocks):
        _block(t, f"dspark.{i}", 0, False)
    taps = len(DSPARK["dspark_target_layer_ids"])
    _q(t, "dspark.0.main_proj", D, taps * D)
    _norm(t, "dspark.0.main_norm.weight", D)
    last = f"dspark.{blocks - 1}"
    _hc(t, f"{last}.hc_head", 4)
    _norm(t, f"{last}.norm.weight", D)
    rank = DSPARK["dspark_markov_rank"]
    t[f"{last}.markov_head.markov_w1.weight"] = (0.5 * mx.random.normal((VOCAB, rank))).astype(mx.bfloat16)
    t[f"{last}.markov_head.markov_w2.weight"] = (0.5 * mx.random.normal((VOCAB, rank))).astype(mx.bfloat16)
    mx.save_safetensors(str(folder / "dspark.safetensors"), t)
    (folder / "config.json").write_text(json.dumps({**TEXT, **DSPARK}))
    return folder
