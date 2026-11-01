"""Layer 0's attention inputs and output in the BF16 (or dequantized) oracle, against the XPU runner's dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.gemma4 import modeling_gemma4 as mg

sys.path.insert(0, str(Path(__file__).parent))
from gemma4_e4b_reference import load_model  # noqa: E402


def bf(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("ref", type=Path)
    p.add_argument("name")
    p.add_argument("dump", type=Path)
    p.add_argument("--quant", type=Path)
    p.add_argument("--rows", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(20)
    ids = json.loads((args.ref / f"{args.name}.json").read_text())["prompt_ids"][: args.rows]
    n = len(ids)
    model = load_model(args.model, args.quant)
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    seen = {}
    orig = mg.eager_attention_forward

    def spy(module, query, key, value, *a, **k):
        if module.layer_idx == 0:
            seen["q"], seen["k"], seen["v"] = query.detach().float(), key.detach().float(), value.detach().float()
        out = orig(module, query, key, value, *a, **k)
        if module.layer_idx == 0:
            seen["o"] = out[0].detach().float()
        return out

    mg.eager_attention_forward = spy
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False)
    hd = 256
    q = bf(np.fromfile(args.dump / "xpu_q0.bin", dtype=np.uint16)).reshape(-1, 8, hd)[:n]
    k = bf(np.fromfile(args.dump / "xpu_k0.bin", dtype=np.uint16)).reshape(-1, 2, hd)[:n]
    v = bf(np.fromfile(args.dump / "xpu_v0.bin", dtype=np.uint16)).reshape(-1, 2, hd)[:n]
    o = np.fromfile(args.dump / "xpu_attn0.bin", dtype=np.float16).astype(np.float32).reshape(-1, 8 * hd)[:n]
    # HF: query [1, 8, n, hd]; key/value [1, 2, n, hd]; output [1, n, 8, hd]
    refs = {"q": seen["q"][0].permute(1, 0, 2).numpy(), "k": seen["k"][0].permute(1, 0, 2).numpy(), "v": seen["v"][0].permute(1, 0, 2).numpy(), "o": seen["o"][0].reshape(n, -1).numpy()}
    for name, mine in (("q", q), ("k", k), ("v", v), ("o", o)):
        ref = refs[name]
        d = mine - ref
        print(f"{name}: rel L2 {np.linalg.norm(d) / np.linalg.norm(ref):.5f}   per row: " + " ".join(f"{np.linalg.norm(d[r]) / np.linalg.norm(ref[r]):.3f}" for r in range(min(n, 8))))


if __name__ == "__main__":
    main()
