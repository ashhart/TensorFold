"""Tiny random Hy4-flash checkpoint and its transformers reference: hidden streams after each layer, logits."""

import argparse
import copy
import json
import os

import torch
from safetensors.torch import save_file


F32_KEEP = (".hc_pre.hc_base", ".hc_pre.hc_fn", ".hc_pre.hc_scale", "hc_head.", "learnable_sink_param", "intra_group_ape",
            "key_gate_projection.weight", "key_value_projection.weight", "e_score_correction_bias", "lm_head.weight")


def release_dtypes(path):
    """The saved tensors the release stores in float32 become float32; the values stay the bf16 ones."""
    from safetensors.torch import load_file

    t = load_file(path)
    save_file({k: v.float() if any(p in k for p in F32_KEEP) else v for k, v in t.items()}, path, metadata={"format": "pt"})


def tiny_config(src, layers, experts, topk):
    c = json.load(open(src))
    t = c["text_config"]
    t["num_hidden_layers"] = layers
    for k, v in list(t.items()):
        if isinstance(v, list) and len(v) == 42:
            t[k] = v[:layers]
    t["n_routed_experts"], t["num_experts_per_tok"] = experts, topk
    sizes = t["engram_layer_head_vocab_sizes"]["2"]
    t["engram_layer_ids"] = [2]
    t["engram_layer_head_vocab_sizes"] = {"2": [[1009 + 104 * j + 2 * i for i in range(len(r))] for j, r in enumerate(sizes)]}
    t["engram_vocab_size"] = [2000 for _ in t["engram_vocab_size"]]
    for k in [k for k in c if k.startswith("dfly_")] + ["speculative_config"]:
        c.pop(k, None)
    c["vision_config"]["num_hidden_layers"] = 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForImageTextToText

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/config.json", "w") as f:
        json.dump(tiny_config(a.config, a.layers, a.experts, a.topk), f, indent=1)
    torch.manual_seed(a.seed)
    cfg = AutoConfig.from_pretrained(a.out)
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16).eval()
    model.save_pretrained(a.out, safe_serialization=True)
    release_dtypes(f"{a.out}/model.safetensors")
    ref = copy.deepcopy(model).to(torch.float32)
    vocab = cfg.text_config.vocab_size
    ids = torch.randint(0, 119000, (1, a.tokens), generator=torch.Generator().manual_seed(a.seed + 1)) % vocab
    with torch.no_grad():
        out = ref(input_ids=ids, output_hidden_states=True, use_cache=False)
    tensors = {"input_ids": ids[0].to(torch.int32).contiguous(), "logits": out.logits[0].float().contiguous()}
    for i, h in enumerate(out.hidden_states or ()):
        tensors[f"hidden.{i}"] = h[0].float().contiguous()
    save_file(tensors, f"{a.out}/reference.safetensors")
    print({k: tuple(v.shape) for k, v in tensors.items()})


if __name__ == "__main__":
    main()
