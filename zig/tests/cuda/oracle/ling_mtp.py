"""For `tf-cuda-test ling-mtp-real`: HF's MTP layer (BailingMoeV3MTPLayer) on real rows, as a flat file."""
import argparse
import importlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="the checkpoint: config.json, model.safetensors.index.json, its shards (FP8 blocks)")
    p.add_argument("modeling", help="a package folder (with __init__.py) holding the HF modeling and configuration files")
    p.add_argument("rows", help="a .pt with 'ids' [1][T] and 'out' [1][T][D] (a layer's output, e.g. ref0.pt)")
    p.add_argument("out", help="the fixture: embeddings, hidden rows (bf16), HF's output (fp32), HF's top-k (i32)")
    a = p.parse_args()
    pkg = os.path.basename(os.path.normpath(a.modeling))
    sys.path.insert(0, os.path.dirname(os.path.normpath(a.modeling)))
    import transformers.utils.import_utils as iu
    if not hasattr(iu, "is_torch_fx_available"):  # newer transformers dropped it; the modeling only imports it
        iu.is_torch_fx_available = lambda: False
    cfg_mod = importlib.import_module(pkg + ".configuration_bailing_moe_v3")
    mod = importlib.import_module(pkg + ".modeling_bailing_moe_v3")
    raw = json.load(open(a.src + "/config.json"))
    cfg = cfg_mod.BailingMoeV3Config(**{k: v for k, v in raw.items() if k not in ("auto_map", "architectures", "quantization_config")})
    cfg._attn_implementation = "eager"
    if raw.get("rope_scaling") is None:  # newer transformers fill in a default dict the modeling reads as YaRN's
        cfg.rope_scaling = None
    wmap = json.load(open(a.src + "/model.safetensors.index.json"))["weight_map"]
    files = {}

    def get(k):
        fn = wmap[k]
        if fn not in files:
            files[fn] = safe_open(a.src + "/" + fn, "pt", device="cuda")
        w = files[fn].get_tensor(k)
        if w.dtype == torch.float8_e4m3fn:  # 128 x 128 blocks of weight_scale_inv
            s = get(k + "_scale_inv").float()
            w = (w.float() * s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]).to(torch.bfloat16)
        return w

    L = cfg.num_hidden_layers
    with torch.device("meta"):
        layer = mod.BailingMoeV3MTPLayer(cfg, L)
    sd = {}
    for name, prm in list(layer.named_parameters()) + list(layer.named_buffers()):
        t = get(f"model.layers.{L}." + name)
        sd[name] = t.float() if prm.dtype == torch.float32 else t.to(torch.bfloat16)
    layer.load_state_dict(sd, strict=True, assign=True)
    layer = layer.cuda().eval()
    src = torch.load(a.rows)
    ids, h = src["ids"][0].cuda(), src["out"].cuda().to(torch.bfloat16)
    T = ids.shape[0] - 1  # row t: the hidden row t and the embedding of token t + 1
    emb = get("model.word_embeddings.weight")[ids[1:]][None]
    norm = get("model.norm.weight")
    hf = h[:, :T].float()
    hidden = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(torch.bfloat16) * norm  # model.norm's rows
    pos = torch.arange(T, device="cuda")[None]
    rd = cfg.qk_rope_head_dim  # BailingMoeV3RotaryEmbedding's default rope (its init table moved in newer transformers)
    inv = 1.0 / cfg.rope_theta ** (torch.arange(0, rd, 2, device="cuda").float() / rd)
    ang = torch.cat([pos[0, :, None].float() * inv] * 2, -1)[None]
    cos_sin = (ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16))
    mask = torch.full((T, T), torch.finfo(torch.bfloat16).min, device="cuda").triu(1)[None, None].to(torch.bfloat16)
    picks = []
    layer.mlp.gate.register_forward_hook(lambda m, i, o: picks.append(o[0]))
    with torch.no_grad():
        out = layer(emb, hidden, attention_mask=mask, position_ids=pos, position_embeddings=cos_sin)[0]
    topk = picks[0].view(T, -1).int().cpu()
    D = emb.shape[-1]
    with open(a.out, "wb") as f:
        f.write(b"TFLINGM1" + struct.pack("<3I", T, D, topk.shape[1]))
        for t in (emb[0], hidden[0]):
            f.write(t.cpu().view(torch.int16).numpy().tobytes())
        f.write(out[0].float().cpu().numpy().tobytes() + topk.numpy().tobytes())
    print("mtp layer", L, "tokens", T, "rms", out.float().pow(2).mean().sqrt().item(), "finite", torch.isfinite(out).all().item())


if __name__ == "__main__":
    main()
