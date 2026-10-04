"""DeepSeek-V4.1-Flash's checkpoint inventory: every tensor's name, dtype and shape from config.json alone."""

from __future__ import annotations

from tensorfold.families.deepseek_v41.config import Config


def _fp8(out: dict, name: str, n: int, k: int) -> None:
    out[f"{name}.weight"] = ("F8_E4M3", [n, k])
    out[f"{name}.scale"] = ("F8_E8M0", [-(-n // 32), -(-k // 32)])


def _attention(out: dict, p: str, c: Config) -> None:
    heads = c.num_attention_heads * c.head_dim
    _fp8(out, p + "attn.wq_a", c.q_lora_rank, c.hidden_size)
    _fp8(out, p + "attn.wq_b", heads, c.q_lora_rank)
    _fp8(out, p + "attn.wkv", c.head_dim, c.hidden_size)
    _fp8(out, p + "attn.wo_a", c.o_groups * c.o_lora_rank, heads // c.o_groups)
    _fp8(out, p + "attn.wo_b", c.hidden_size, c.o_groups * c.o_lora_rank)
    out[p + "attn.q_norm.weight"] = ("BF16", [c.q_lora_rank])
    out[p + "attn.kv_norm.weight"] = ("BF16", [c.head_dim])
    out[p + "attn.attn_sink"] = ("F32", [c.num_attention_heads])
    for norm in ("attn_norm", "ffn_norm"):
        out[f"{p}{norm}.weight"] = ("BF16", [c.hidden_size])
    mixes = (2 + c.hc_mult) * c.hc_mult
    for hc in ("attn", "ffn"):
        out[f"{p}hc_{hc}_fn"] = ("F32", [mixes, c.hc_mult * c.hidden_size])
        out[f"{p}hc_{hc}_base"] = ("F32", [mixes])
        out[f"{p}hc_{hc}_scale"] = ("F32", [3])


def _ffn(out: dict, p: str, c: Config, experts: int) -> None:
    inter = c.moe_intermediate_size * c.n_shared_experts
    out[p + "ffn.gate.weight"] = ("BF16", [experts, c.hidden_size])
    out[p + "ffn.gate.bias"] = out[p + "ffn.gate.bias_vl"] = ("F32", [experts])
    _fp8(out, p + "ffn.shared_experts.w1", inter, c.hidden_size)
    _fp8(out, p + "ffn.shared_experts.w3", inter, c.hidden_size)
    _fp8(out, p + "ffn.shared_experts.w2", c.hidden_size, inter)


def inventory(config: dict) -> dict[str, dict]:
    """Every tensor the checkpoint stores, by name, with the dtype and shape config.json implies."""

    c, v = Config.from_dict(config), config["vision_config"]
    d, inter, bits = c.hidden_size, c.moe_intermediate_size, config["quantization_config"]["bits"]
    columns = (c.engram_max_ngram_size - 1) * c.engram_n_heads
    out: dict = {"embed.weight": ("BF16", [c.vocab_size, d]), "norm.weight": ("BF16", [d]),
                 "lm_head.weight": ("F8_E4M3", [c.vocab_size, d]), "lm_head.weight_scale": ("U8", [c.vocab_size, d // 32])}
    for layer in range(c.num_hidden_layers):
        p = f"layers.{layer}."
        _attention(out, p, c)
        _ffn(out, p, c, c.n_routed_experts)
        if layer in c.kv_source_layer_ids:
            out[p + "attn.compressor.wkv.weight"] = out[p + "attn.compressor.wgate.weight"] = ("BF16", [c.head_dim, d])
            if c.compress_ratios[layer] == 1:
                del out[p + "attn.compressor.wgate.weight"]
            out[p + "attn.compressor.norm.weight"] = ("BF16", [c.head_dim])
            out[p + "attn.indexer.wk.weight"] = ("BF16", [c.index_head_dim, c.head_dim])
            out[p + "attn.indexer.k_norm.weight"] = ("BF16", [c.index_head_dim])
        if layer in c.index_source_layer_ids:
            _fp8(out, p + "attn.indexer.wq_b", c.index_n_heads * c.index_head_dim, c.q_lora_rank)
            out[p + "attn.indexer.weights_proj.weight"] = ("BF16", [c.index_n_heads, d])
        if layer in c.engram_layer_ids:
            rows = c.engram_num_embeddings[c.engram_layer_ids.index(layer)]
            out[p + "engram.embed.weight"] = ("F8_E4M3", [rows, c.engram_head_dim])
            out[p + "engram.embed.scale"] = ("F8_E8M0", [rows, c.engram_head_dim // 32])
            _fp8(out, p + "engram.wkv", (c.hc_mult + 1) * d, columns * c.engram_head_dim)
            out[p + "engram.q_weight"] = out[p + "engram.k_weight"] = ("BF16", [c.hc_mult, d])
        for e in range(c.n_routed_experts):
            for w, (k, n) in (("w1", (d, inter)), ("w3", (d, inter)), ("w2", (inter, d))):
                q = f"{p}ffn.experts.{e}.{w}."
                out[q + "trellis"] = ("I16", [k // 16, n // 16, 16 * bits])
                out[q + "suh"], out[q + "svh"], out[q + "mcg"] = ("F16", [k]), ("F16", [n]), ("I32", [])
    for stage in range(c.num_nextn_predict_layers):
        p = f"mtp.{stage}."
        _attention(out, p, c)
        _ffn(out, p, c, c.dspark_n_routed_experts)
        for e in range(c.dspark_n_routed_experts):
            for w, (n, k) in (("w1", (inter, d)), ("w3", (inter, d)), ("w2", (d, inter))):
                out[f"{p}ffn.experts.{e}.{w}.weight"] = ("I8", [n, k // 2])
                out[f"{p}ffn.experts.{e}.{w}.scale"] = ("F8_E8M0", [n, k // 32])
    _fp8(out, "mtp.0.main_proj", d, len(c.dspark_target_layer_ids) * d)
    out["mtp.0.main_norm.weight"] = ("BF16", [d])
    last, rank = f"mtp.{c.num_nextn_predict_layers - 1}.", c.dspark_markov_rank
    out[last + "norm.weight"] = ("BF16", [d])
    out[last + "markov_head.embed.weight"] = out[last + "markov_head.head.weight"] = ("BF16", [c.vocab_size, rank])
    out[last + "confidence_head.proj.weight"] = ("BF16", [1, d + rank])
    vh, vi, patch, down = v["hidden_size"], v["intermediate_size"], v["patch_size"], v["downsample_ratio"]
    for b in range(v["num_hidden_layers"]):
        p = f"vision.blocks.{b}."
        out[p + "attn.wqkv.weight"], out[p + "attn.wqkv.bias"] = ("BF16", [3 * vh, vh]), ("BF16", [3 * vh])
        out[p + "attn.wo.weight"], out[p + "attn.wo.bias"] = ("BF16", [vh, vh]), ("BF16", [vh])
        out[p + "mlp.w1.weight"], out[p + "mlp.w2.weight"] = ("BF16", [2 * vi, vh]), ("BF16", [vh, vi])
        out[p + "norm1.weight"] = out[p + "norm2.weight"] = ("BF16", [vh])
    out["vision.norm.weight"] = ("BF16", [vh])
    out["vision.patch_embed.proj.weight"] = ("BF16", [vh, 3 * patch * patch])
    out["vision.patch_embed.proj.bias"] = ("BF16", [vh])
    out["aligner.w1.weight"], out["aligner.w1.bias"] = ("BF16", [d, vh * down * down]), ("BF16", [d])
    out["aligner.w2.weight"], out["aligner.w2.bias"] = ("BF16", [d, d]), ("BF16", [d])
    for token in ("image_start", "image_end", "image_newline"):
        out[token] = ("BF16", [d])
    return {name: {"dtype": dtype, "shape": shape} for name, (dtype, shape) in out.items()}
