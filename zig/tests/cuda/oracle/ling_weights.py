"""For `tf-cuda-test ling-weights-real`: torch's float64 y = x W^T of real Ling-3.1 projections on fixed rows."""
import argparse
import json
import os

import torch
from safetensors import safe_open

ROWS = 8  # decode rows, and the down projection's pair rows (2 tokens x 4 experts)
EXPERTS = 4
KDA = ("q_proj", "k_proj", "v_proj", "b_proj", "f_proj", "g_proj", "o_proj")
MLA = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "dense")
MLP = ("gate_proj", "up_proj", "down_proj")


def modules() -> list[str]:
    """Layer 0 (KDA, dense MLP), 4 (KDA, MoE) and 7 (MLA, MoE): every projection, the router, experts 0-3."""
    out = [f"model.layers.0.attention.{m}" for m in KDA] + [f"model.layers.0.mlp.{m}" for m in MLP]
    for layer, mixer in ((4, KDA), (7, MLA)):
        out += [f"model.layers.{layer}.attention.{m}" for m in mixer] + [f"model.layers.{layer}.mlp.gate"]
        out += [f"model.layers.{layer}.mlp.shared_experts.{m}" for m in MLP]
        out += [f"model.layers.{layer}.mlp.experts.{e}.{m}" for e in range(EXPERTS) for m in MLP]
    return out


def rows(k: int) -> torch.Tensor:
    """The bf16 rows of width k both sides read: uniform in [-1, 1), seeded by k."""
    g = torch.Generator().manual_seed(k)
    return (torch.rand((ROWS, k), generator=g) * 2 - 1).to(torch.bfloat16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="the FP8 checkpoint (model.safetensors.index.json and its shards)")
    p.add_argument("out", help="where x<k>.bin, <module>.y, <module>.abs and manifest.txt go")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    weight_map = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    files, xs, lines = {}, {}, []
    for m in modules():
        name = weight_map[m + ".weight"]
        f = files.setdefault(name, safe_open(os.path.join(a.src, name), "pt"))
        w = f.get_tensor(m + ".weight")
        n, k = w.shape
        if w.dtype == torch.float8_e4m3fn:
            s = weight_map[m + ".weight_scale_inv"]
            inv = files.setdefault(s, safe_open(os.path.join(a.src, s), "pt")).get_tensor(m + ".weight_scale_inv")
            w = w.double() * inv.double().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        w = w.double()
        if k not in xs:
            xs[k] = rows(k)
            xs[k].view(torch.int16).numpy().tofile(os.path.join(a.out, f"x{k}.bin"))
        x = xs[k].double()
        (x @ w.T).numpy().tofile(os.path.join(a.out, m + ".y"))
        (x.abs() @ w.abs().T).numpy().tofile(os.path.join(a.out, m + ".abs"))
        lines.append(f"{m} {n} {k}")
        print(m, n, k, flush=True)
    open(os.path.join(a.out, "manifest.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
