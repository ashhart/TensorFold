"""The routed-experts fixture: nine real experts of one MoE layer through the Python engine's generic.routed."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

ROWS = (1, 3, 16)
SEED = 1236
SLOTS = 8
LIMIT = 10.0
POOL = 3  # experts per width


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def load_group(model_dir: Path, prefix: str) -> tuple:
    """(trellis, suh, svh) of one expert projection, on the GPU (the checkpoint's stored trellis order)."""
    from safetensors import safe_open

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    rel = None
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        rel = weight_map.get(f"{prefix}.trellis")
        if rel is None:
            for part in sorted(root.glob("*.safetensors")):
                with safe_open(str(part), framework="pt") as g:
                    if f"{prefix}.trellis" in set(g.keys()):
                        rel = part.name
                        break
        if rel is None:
            raise ValueError(f"{prefix}.trellis is in no file of {root}")
    file = root / (rel or "model.safetensors")
    with safe_open(str(file), framework="pt", device="cuda") as f:
        return (f.get_tensor(f"{prefix}.trellis"), f.get_tensor(f"{prefix}.suh"), f.get_tensor(f"{prefix}.svh"))


def pick_layer(cp) -> str:
    """The first layer (sorted) whose routed experts cover 4, 5 and 6 bits."""
    layers: dict[str, dict[float, list[int]]] = {}
    for prefix, t in cp.groups.items():
        m = re.match(r"^(.*)\.mlp\.experts\.(\d+)\.gate_proj$", prefix)
        if m is not None and t.in_scales == "suh" and t.out_scales == "svh":
            layers.setdefault(m.group(1), {}).setdefault(t.bits, []).append(int(m.group(2)))
    for base in sorted(layers):
        if all(len(layers[base].get(b, ())) >= POOL for b in (4.0, 5.0, 6.0)):
            return base
    raise ValueError("no MoE layer has 4, 5 and 6-bit routed experts")

def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import format as fmt

    cp = fmt.scan(model_dir)
    base = pick_layer(cp)

    by_bits: dict[float, list[int]] = {}
    for prefix, t in cp.groups.items():
        m = re.match(rf"^{re.escape(base)}\.mlp\.experts\.(\d+)\.gate_proj$", prefix)
        if m is not None:
            by_bits.setdefault(t.bits, []).append(int(m.group(1)))
    pool = [e for b in (4.0, 5.0, 6.0) for e in sorted(by_bits[b])[:POOL]]

    gate, up, down = [], [], []
    trellis = {"gate": [], "up": [], "down": []}
    for slot, e in enumerate(pool):
        for part, short, mats, names in (("gate_proj", "gate", gate, trellis["gate"]),
                                         ("up_proj", "up", up, trellis["up"]),
                                         ("down_proj", "down", down, trellis["down"])):
            mats.append(load_group(model_dir, f"{base}.mlp.experts.{e}.{part}"))
            names.append(f"experts.{short}{slot}.bin")
    ex = generic.prepare(gate, up, down, "mul1", device="cuda")

    suh_files = {}
    for key, t in (("suh_g", ex.suh_g), ("suh_u", ex.suh_u), ("svh_g", ex.svh_g),
                   ("svh_u", ex.svh_u), ("suh_d", ex.suh_d), ("svh_d", ex.svh_d)):
        save(out, f"experts.{key}", t)
        suh_files[key] = f"experts.{key}.bin"
    for slot, mats in enumerate(zip(gate, up, down)):
        for name, t in zip(("gate", "up", "down"), mats):
            save(out, f"experts.{name}{slot}", t[0])

    g = torch.Generator().manual_seed(SEED)
    x = torch.randn((max(ROWS), ex.dims), generator=g).to(torch.bfloat16)
    pick = torch.stack([torch.randperm(ex.count, generator=g)[:SLOTS] for _ in range(max(ROWS))]).int()
    wts = (0.1 + 0.9 * torch.rand((max(ROWS), SLOTS), generator=g)).float()

    scratch = generic.Scratch(ex, max(ROWS), SLOTS)
    xmap, pickmap, wtsmap, ymap, outmap, sha = {}, {}, {}, {}, {}, {}
    for m in ROWS:
        y = generic.routed(x[:m].cuda(), pick[:m].cuda(), wts[:m].cuda(), ex, scratch, None, m,
                           limit=LIMIT, act_mode=generic.ACT_BF16)
        torch.cuda.synchronize()
        saved = {"x": x[:m], "pick": pick[:m], "wts": wts[:m], "y": scratch.y[:m * SLOTS], "out": y}
        maps = {"x": xmap, "pick": pickmap, "wts": wtsmap, "y": ymap, "out": outmap}
        for key, t in saved.items():
            save(out, f"experts.{key}{m}", t)
            maps[key][str(m)] = f"experts.{key}{m}.bin"
        sha[str(m)] = hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    entry = {"id": "experts", "name": base, "count": ex.count, "dims": ex.dims, "width": ex.width,
             "slots": SLOTS, "codebook": "mul1", "limit": LIMIT, "act_mode": generic.ACT_BF16,
             "rows": list(ROWS), "gate_k2": [generic.k2_of(m[0][0]) for m in gate],
             "up_k2": [generic.k2_of(m[0]) for m in up], "down_k2": [generic.k2_of(m[0]) for m in down],
             "trellis": trellis, **suh_files, "x": xmap, "pick": pickmap, "wts": wtsmap,
             "y": ymap, "out": outmap, "sha256": sha}
    print(f"experts: {base} {len(pool)} experts (4/5/6-bit pool), rows {ROWS}, sha256={sha['16'][:16]}…")
    return entry
