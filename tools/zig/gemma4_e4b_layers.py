"""Per-layer hidden states of the BF16 oracle for a prompt (first N tokens), against the XPU runner's XPU_DUMP files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma4_e4b_reference import load_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("ref", type=Path)
    p.add_argument("name")
    p.add_argument("dump", type=Path, help="XPU_DUMP directory of the runner")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--quant", type=Path)
    args = p.parse_args()
    torch.set_num_threads(20)
    ids = json.loads((args.ref / f"{args.name}.json").read_text())["prompt_ids"][: args.rows]
    model = load_model(args.model, args.quant)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    hs = [h[0].float().numpy() for h in out.hidden_states]
    n = len(ids)
    names = ["embed"] + [str(i) for i in range(len(hs) - 1)]
    for i, name in enumerate(names):
        f = args.dump / f"xpu_x_{name}.f32"
        if not f.exists():
            continue
        x = np.fromfile(f, dtype=np.float32).reshape(-1, 2560)[:n]
        ref = hs[i][:n]
        d = x - ref
        rel = np.linalg.norm(d) / np.linalg.norm(ref)
        worst = np.abs(d).max() / np.abs(ref).max()
        print(f"{name:6s} rel L2 {rel:.5f}  max|d|/max|ref| {worst:.5f}  per-row rel L2: " + " ".join(f"{np.linalg.norm(d[r]) / np.linalg.norm(ref[r]):.3f}" for r in range(min(n, 8))))


if __name__ == "__main__":
    main()
