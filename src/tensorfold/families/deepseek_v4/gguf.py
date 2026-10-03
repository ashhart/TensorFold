"""DeepSeek GGUF tensor shapes and formats read by the native CUDA kernels."""

import math


class DeepSeekV4SchemaError(ValueError):
    """The inventory cannot be executed by the native DeepSeek CUDA path."""


def validate_cuda_shapes(inventory, arch, *, prefix="blk", layers=None, ratios=None):
    """Check every shape read by the native CUDA path before allocating any weights."""
    tensors = {t.name: t for t in inventory.tensors}
    d = int(arch["deepseek4.embedding_length"])
    h = int(arch["deepseek4.attention.head_count"])
    hd = int(arch["deepseek4.attention.key_length"])
    groups = int(arch["deepseek4.attention.output_group_count"])
    qr = int(arch["deepseek4.attention.q_lora_rank"])
    rank = int(arch["deepseek4.attention.output_lora_rank"])
    experts = int(arch["deepseek4.expert_count"])
    width = int(arch["deepseek4.expert_feed_forward_length"])
    used = int(arch["deepseek4.expert_used_count"])
    if (d, h, hd) != (4096, 64, 512) or int(arch.get("deepseek4.hyper_connection.count", 4)) != 4:
        raise DeepSeekV4SchemaError("CUDA DeepSeek requires 4 streams of 4096, 64 query heads and 512 KV dimensions")
    if groups < 1 or h % groups or qr < 1 or rank < 1 or width < 1 or not 1 <= used <= experts <= 1024:
        raise DeepSeekV4SchemaError("invalid DeepSeek projection or expert dimensions")
    if int(arch.get("deepseek4.attention.head_count_kv", 1)) != 1:
        raise DeepSeekV4SchemaError("CUDA DeepSeek requires one shared KV head")
    if int(arch.get("deepseek4.rope.dimension_count", 64)) != 64:
        raise DeepSeekV4SchemaError("CUDA DeepSeek requires 64 rotary dimensions")
    for key in ["deepseek4.attention.layer_norm_rms_epsilon", "deepseek4.hyper_connection.epsilon"]:
        value = float(arch.get(key, 1e-6))
        if not math.isfinite(value) or value <= 0:
            raise DeepSeekV4SchemaError(f"{key} must be finite and positive")
    count = int(arch["deepseek4.block_count"]) if layers is None else int(layers)
    ratios = list(arch["deepseek4.attention.compress_ratios"]) if ratios is None else list(ratios)
    if count < 1 or len(ratios) < count or any(r not in (0, 4, 128) for r in ratios):
        raise DeepSeekV4SchemaError("one supported compression ratio is required for every layer")

    ratios = ratios[:count]  # GGUF may append the checkpoint's next-token prediction layer.

    def shape(name, expected):
        t = tensors.get(name)
        if t is None or tuple(t.shape) != expected:
            raise DeepSeekV4SchemaError(f"{name}: expected GGUF shape {expected}, got {None if t is None else t.shape}")
        if t.type_name not in {"F16", "F32", "I32", "Q8_0", "Q2_K", "IQ2_XXS"}:
            raise DeepSeekV4SchemaError(f"{name}: native CUDA does not read {t.type_name}")
        if t.type_name == "I32" and not name.endswith("ffn_gate_tid2eid.weight"):
            raise DeepSeekV4SchemaError(f"{name}: integer weights are only supported for hash routing")

    vocab = int(arch["deepseek4.vocab_size"])
    if prefix == "blk":
        for name, expected in [
            ("token_embd.weight", (d, vocab)),
            ("output.weight", (d, vocab)),
            ("output_norm.weight", (d,)),
            ("output_hc_fn.weight", (4 * d, 4)),
            ("output_hc_base.weight", (4,)),
            ("output_hc_scale.weight", (1,)),
        ]:
            shape(name, expected)
        for name in ["output_norm.weight", "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"]:
            if tensors[name].type_name not in {"F16", "F32"}:
                raise DeepSeekV4SchemaError(f"{name}: requires floating weights")
        if tensors["token_embd.weight"].type_name not in {"F16", "F32"}:
            raise DeepSeekV4SchemaError("CUDA embedding must be F16 or F32")
    hashes = int(arch.get("deepseek4.hash_layer_count", 3)) if prefix == "blk" else 0
    for i, r in enumerate(ratios):
        p = f"{prefix}.{i}."
        required = {
            "attn_q_a.weight": (d, qr),
            "attn_q_b.weight": (qr, h * hd),
            "attn_kv.weight": (d, hd),
            "attn_output_a.weight": (h // groups * hd, groups * rank),
            "attn_output_b.weight": (groups * rank, d),
            "attn_q_a_norm.weight": (qr,),
            "attn_kv_a_norm.weight": (hd,),
            "attn_sinks.weight": (h,),
            "attn_norm.weight": (d,),
            "ffn_norm.weight": (d,),
            "ffn_gate_inp.weight": (d, experts),
            "ffn_gate_exps.weight": (d, width, experts),
            "ffn_up_exps.weight": (d, width, experts),
            "ffn_down_exps.weight": (width, d, experts),
            "ffn_gate_shexp.weight": (d, width),
            "ffn_up_shexp.weight": (d, width),
            "ffn_down_shexp.weight": (width, d),
        }
        for kind in ["attn", "ffn"]:
            required.update(
                {f"hc_{kind}_fn.weight": (4 * d, 24), f"hc_{kind}_base.weight": (24,), f"hc_{kind}_scale.weight": (3,)}
            )
        if i < hashes:
            required["ffn_gate_tid2eid.weight"] = (used, vocab)
        else:
            required["exp_probs_b.bias"] = (experts,)
        if r:
            comp = 2 * hd if r == 4 else hd
            required.update(
                {
                    "attn_compressor_kv.weight": (d, comp),
                    "attn_compressor_gate.weight": (d, comp),
                    "attn_compressor_ape.weight": (comp, r),
                    "attn_compressor_norm.weight": (hd,),
                }
            )
        if r == 4:
            for key, value in [("head_count", 64), ("key_length", 128), ("top_k", 512)]:
                if int(arch.get("deepseek4.attention.indexer." + key, value)) != value:
                    raise DeepSeekV4SchemaError("CUDA indexer requires 64 heads, 128 dimensions and top-512 selection")
            required.update(
                {
                    "indexer_compressor_kv.weight": (d, 256),
                    "indexer_compressor_gate.weight": (d, 256),
                    "indexer_compressor_ape.weight": (256, 4),
                    "indexer_compressor_norm.weight": (128,),
                    "indexer.attn_q_b.weight": (qr, 64 * 128),
                    "indexer.proj.weight": (d, 64),
                }
            )
        for name, expected in required.items():
            shape(p + name, expected)
        for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]:
            if tensors[p + name].type_name not in {"IQ2_XXS", "Q2_K", "Q8_0"}:
                raise DeepSeekV4SchemaError(f"{p + name}: routed experts must remain packed")
        for name in [
            "attn_q_a_norm.weight",
            "attn_kv_a_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_sinks.weight",
            "hc_attn_fn.weight",
            "hc_ffn_fn.weight",
            "hc_attn_base.weight",
            "hc_ffn_base.weight",
            "hc_attn_scale.weight",
            "hc_ffn_scale.weight",
            *(["attn_compressor_ape.weight", "attn_compressor_norm.weight"] if r else []),
            *(["indexer_compressor_ape.weight", "indexer_compressor_norm.weight"] if r == 4 else []),
        ]:
            if tensors[p + name].type_name not in {"F16", "F32"}:
                raise DeepSeekV4SchemaError(f"{p + name}: requires floating weights")
        if i < hashes and tensors[p + "ffn_gate_tid2eid.weight"].type_name != "I32":
            raise DeepSeekV4SchemaError(f"{p}: hash routing requires I32 ids")
