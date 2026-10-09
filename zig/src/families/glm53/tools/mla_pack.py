#!/usr/bin/env python3
"""The MLA absorb pack tf-glm53 loads beside the checkpoint: mlx-lm's sanitize turns each layer's kv_b_proj into embed_q
(q_nope -> latent) and unembed_out (latent -> v), re-quantized at 8-bit g64. This saves exactly those tensors, so the
engine reads mlx-lm's values. Usage: mla_pack.py CHECKPOINT_DIR OUT.safetensors   (mlx-lm with GLM-5.3 support)"""
import sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model

model_dir, out = Path(sys.argv[1]).expanduser(), sys.argv[2]
t0 = time.time()
model, cfg = load_model(model_dir, lazy=True, strict=True)
d = {}
for i, layer in enumerate(model.model.layers):
    a = layer.self_attn
    for name in ("embed_q", "unembed_out"):
        m = getattr(a, name)
        for k in ("weight", "scales", "biases"):
            d[f"layers.{i}.{name}.{k}"] = getattr(m, k)
    mx.eval([d[f"layers.{i}.{n}.{k}"] for n in ("embed_q", "unembed_out") for k in ("weight", "scales", "biases")])
e0 = model.model.layers[0].self_attn.embed_q
print("embed_q", e0.weight.shape, e0.weight.dtype, e0.scales.shape, "bits", getattr(e0, "bits", None), "gs", getattr(e0, "group_size", None))
u0 = model.model.layers[0].self_attn.unembed_out
print("unembed_out", u0.weight.shape, u0.scales.shape)
mx.save_safetensors(out, d, metadata={"from": model_dir.name, "how": "mlx-lm sanitize (dequant kv_b, split, requant 8-bit g64)"})
print(f"saved {len(d)} tensors in {time.time()-t0:.1f}s")
