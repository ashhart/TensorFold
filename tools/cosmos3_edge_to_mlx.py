"""Cosmos3-Edge's reasoner (understanding tower) as a TensorFold Metal checkpoint: MLX affine codes (groups of 64) for
every projection, bf16 norms, biases and the vision position grid. The generation tower (*_moe_gen, action, latent,
time embedder) and the VAE are left out. The vision MLP's 4304 columns are zero-padded to 4352 (a multiple of 64):
gelu(0) is 0 and the padded fc2 columns are 0, so the padding changes nothing.

    python convert.py SRC_DIR OUT_DIR [BITS]
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import Cosmos3EdgeForConditionalGeneration

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GROUP = 64
V_INTER, V_PAD = 4304, 4352

model = Cosmos3EdgeForConditionalGeneration.from_pretrained(SRC, dtype=torch.bfloat16)
sd = {k: v for k, v in model.state_dict().items()}
out: dict[str, mx.array] = {}


def bf16(t: torch.Tensor) -> mx.array:
    return mx.array(t.float().numpy()).astype(mx.bfloat16)


def quant(name: str, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
    wq, scales, biases = mx.quantize(bf16(w), group_size=GROUP, bits=BITS)
    out[name + ".weight"], out[name + ".scales"], out[name + ".biases"] = wq, scales, biases
    if bias is not None:
        out[name + ".bias"] = bf16(bias)


T = "model.language_model."
quant("model.embed_tokens", sd[T + "embed_tokens.weight"])
quant("lm_head", sd["lm_head.weight"])
out["model.norm.weight"] = bf16(sd[T + "norm.weight"])
for i in range(model.config.text_config.num_hidden_layers):
    p, o = f"{T}layers.{i}.", f"model.layers.{i}."
    out[o + "input_layernorm.weight"] = bf16(sd[p + "input_layernorm.weight"])
    out[o + "post_attention_layernorm.weight"] = bf16(sd[p + "post_attention_layernorm.weight"])
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        quant(o + "self_attn." + n, sd[p + "self_attn." + n + ".weight"])
    quant(o + "mlp.fc1", sd[p + "mlp.fc1.weight"])
    quant(o + "mlp.fc2", sd[p + "mlp.fc2.weight"])

V = "model.visual."
pe = sd[V + "embeddings.patch_embedding.weight"]  # [1152, 768]: Linear over HWC-flattened patches
quant("vision.patch_embed", pe, sd[V + "embeddings.patch_embedding.bias"])
out["vision.position"] = bf16(sd[V + "embeddings.position_embedding.weight"])  # [256, 1152]: a 16 x 16 grid
for i in range(model.config.vision_config.num_hidden_layers):
    p, o = f"{V}encoder.layers.{i}.", f"vision.layers.{i}."
    for n in ("layer_norm1", "layer_norm2"):
        out[o + n + ".weight"] = bf16(sd[p + n + ".weight"])
        out[o + n + ".bias"] = bf16(sd[p + n + ".bias"])
    for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
        quant(o + "attn." + n, sd[p + "self_attn." + n + ".weight"], sd[p + "self_attn." + n + ".bias"])
    w1, b1 = sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"]
    w2, b2 = sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"]
    assert w1.shape == (V_INTER, 1152) and w2.shape == (1152, V_INTER)
    w1 = torch.cat([w1, w1.new_zeros(V_PAD - V_INTER, 1152)])
    b1 = torch.cat([b1, b1.new_zeros(V_PAD - V_INTER)])
    w2 = torch.cat([w2, w2.new_zeros(1152, V_PAD - V_INTER)], dim=1)
    quant(o + "mlp.fc1", w1, b1)
    quant(o + "mlp.fc2", w2, b2)
out["vision.post_layernorm.weight"] = bf16(sd[V + "post_layernorm.weight"])
out["vision.post_layernorm.bias"] = bf16(sd[V + "post_layernorm.bias"])
P = "model.projector."
out["projector.norm.weight"] = bf16(sd[P + "norm.weight"])
out["projector.norm.bias"] = bf16(sd[P + "norm.bias"])
quant("projector.fc1", sd[P + "linear_fc1.weight"], sd[P + "linear_fc1.bias"])
quant("projector.fc2", sd[P + "linear_fc2.weight"], sd[P + "linear_fc2.bias"])

OUT.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(OUT / "model.safetensors"), out, metadata={"format": "mlx"})
cfg = json.loads((SRC / "config.json").read_text())
keep = {k: cfg[k] for k in ("architectures", "model_type", "image_token_id", "video_token_id", "vision_start_token_id",
                            "vision_end_token_id", "projector_hidden_size", "text_config", "vision_config", "projector_config",
                            "tie_word_embeddings") if k in cfg}
keep["vision_config"] = dict(keep["vision_config"], padded_intermediate_size=V_PAD)
keep["quantization"] = keep["quantization_config"] = {"group_size": GROUP, "bits": BITS, "mode": "affine"}
keep["eos_token_id"] = cfg.get("text_config", {}).get("eos_token_id", 11)
keep["max_position_embeddings"] = cfg["text_config"]["max_position_embeddings"]
(OUT / "config.json").write_text(json.dumps(keep, indent=1))
for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
          "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"):
    if (SRC / f).exists():
        shutil.copy(SRC / f, OUT / f)
print(f"{len(out)} tensors, {sum(a.nbytes for a in out.values()) / 2**30:.2f} GiB at {BITS}-bit -> {OUT}")
