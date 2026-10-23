"""Layer 0 step by step in the oracle (residuals after attention and MLP, per-layer inputs) against XPU_DUMP files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma4_e4b_reference import load_model  # noqa: E402


def bf(a):
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
    L0 = model.model.layers[0]
    cap = {}
    L0.register_forward_pre_hook(lambda m, a, kw: cap.update(h_in=a[0].detach().float(), pli=a[1].detach().float()), with_kwargs=True)
    for nm in ("post_attention_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"):
        getattr(L0, nm).register_forward_hook(lambda m, a, o, nm=nm: cap.update({nm: o.detach().float(), nm + "_in": a[0].detach().float()}))
    L0.mlp.register_forward_hook(lambda m, a, o: cap.update(mlp=o.detach().float(), mlp_in=a[0].detach().float()))
    L0.per_layer_input_gate.register_forward_hook(lambda m, a, o: cap.update(gate_in=a[0].detach().float(), gate=o.detach().float()))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False)
    xa = cap["h_in"] + cap["post_attention_layernorm"]
    xf = xa + cap["post_feedforward_layernorm"]
    pli = cap["pli"][0].reshape(n, -1).numpy()
    mine = lambda f, sh: np.fromfile(args.dump / f, dtype=np.float32).reshape(-1, 2560)[:n]
    def rep(name, a, b):
        d = a - b
        print(f"{name:10s} rel L2 {np.linalg.norm(d) / np.linalg.norm(b):.5f}  rows: " + " ".join(f"{np.linalg.norm(d[r]) / np.linalg.norm(b[r]):.3f}" for r in range(min(n, 8))))
    rep("x_attn", mine("xpu_x_0_attn.f32", 0), xa[0].numpy())
    rep("x_ffn", mine("xpu_x_0_ffn.f32", 0), xf[0].numpy())
    mp = bf(np.fromfile(args.dump / "xpu_pli.bin", dtype=np.uint16)).reshape(-1, 42 * 256)[:n]
    rep("pli0", mp[:, :256], pli)
    print("mlp out max", float(cap["mlp"].abs().max()), "post_ffn max", float(cap["post_feedforward_layernorm"].abs().max()), "ple gate max", float(cap["gate"].abs().max()))


if __name__ == "__main__":
    main()
