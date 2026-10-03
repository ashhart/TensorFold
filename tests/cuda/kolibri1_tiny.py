"""A tiny Kolibri 1 checkpoint in the released format: FP8 128 x 128 blocks with fp32 scales, bf16 rest."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def write(dir: Path, *, layers: int = 3, hidden: int = 256, heads: int = 4, kv_heads: int = 2, head_dim: int = 128,
          vocab: int = 512, experts: int = 8, top_k: int = 3, width: int = 128, window: int = 33, seed: int = 0,
          full_every: int = 3) -> Path:
    from safetensors.torch import save_file

    dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    t: dict[str, torch.Tensor] = {}

    def fp8(name: str, n: int, k: int, scale: float) -> None:
        w = (rand(n, k) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        t[name + ".weight"] = w
        t[name + ".weight_scale_inv"] = (torch.rand(n // 128, k // 128, generator=g) + 0.5) * scale / 64

    def norm(name: str, n: int) -> None:
        t[name + ".weight"] = (1.0 + rand(n, scale=0.1)).to(torch.bfloat16)

    t["model.embed_tokens.weight"] = rand(vocab, hidden).to(torch.bfloat16)
    t["lm_head.weight"] = rand(vocab, hidden, scale=hidden ** -0.5).to(torch.bfloat16)
    norm("model.norm", hidden)
    for i in range(layers):
        p = f"model.layers.{i}."
        for nm in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            norm(p + nm, hidden)
        a = p + "self_attn."
        fp8(a + "q_proj", heads * head_dim, hidden, hidden ** -0.5)
        fp8(a + "k_proj", kv_heads * head_dim, hidden, hidden ** -0.5)
        fp8(a + "v_proj", kv_heads * head_dim, hidden, hidden ** -0.5)
        fp8(a + "o_proj", hidden, heads * head_dim, (heads * head_dim) ** -0.5)
        norm(a + "q_norm", head_dim)
        norm(a + "k_norm", head_dim)
        t[p + "mlp.gate.weight"] = rand(experts, hidden, scale=hidden ** -0.5).to(torch.bfloat16)
        t[p + "moe.router.expert_bias"] = rand(experts, scale=0.5).float()
        for e in [f"experts.{j}" for j in range(experts)] + ["shared_experts"]:
            fp8(f"{p}mlp.{e}.gate_proj", width, hidden, hidden ** -0.5)
            fp8(f"{p}mlp.{e}.up_proj", width, hidden, hidden ** -0.5)
            fp8(f"{p}mlp.{e}.down_proj", hidden, width, width ** -0.5)
    save_file(t, str(dir / "model-00001-of-00001.safetensors"))
    (dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    types = ["full_attention" if (i + 1) % full_every == 0 else "sliding_attention" for i in range(layers)]
    (dir / "config.json").write_text(json.dumps({
        "architectures": ["Kolibri1ForCausalLM"], "model_type": "kolibri1", "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "head_dim": head_dim, "vocab_size": vocab, "num_experts": experts, "num_experts_per_tok": top_k,
        "moe_intermediate_size": width, "shared_expert_intermediate_size": width, "norm_topk_prob": False,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": window, "layer_types": types,
        "eos_token_id": vocab - 1, "max_position_embeddings": 4096,
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [128, 128]}}))
    return dir
