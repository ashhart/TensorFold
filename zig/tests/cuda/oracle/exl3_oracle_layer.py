"""The whole-layer fixtures: every EXL3 group of one KDA and one DSA MoE layer through the Python engine."""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import torch

ROWS = (1, 16, 300)  # the dense groups' row counts
EXPERT_ROWS = (1, 16)
SEED = 1239  # + layer index


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def sha_of(y: torch.Tensor) -> str:
    return hashlib.sha256(y.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_tensor(model_dir: Path, name: str, device: str = "cpu") -> torch.Tensor:
    from safetensors import safe_open

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        rel = json.loads(index.read_text())["weight_map"][name]
    else:
        rel = "model.safetensors"
    with safe_open(str(root / rel), framework="pt", device=device) as f:
        return f.get_tensor(name)


def mm_rows(x: torch.Tensor, lin, m: int) -> torch.Tensor:
    """lin on m rows: one call to 128, a prompt chunk beyond that as 128-row blocks (forward.py's exl3_mm)."""
    if m <= 128:
        y = lin(x[:m].cuda())
        torch.cuda.synchronize()
        return y
    y = torch.empty((m, lin.n), dtype=torch.bfloat16)
    o = torch.empty((min(m, 128), lin.n), dtype=torch.bfloat16, device="cuda")
    for r in range(0, m, 128):
        e = min(m - r, 128)
        lin(x[r:r + e].cuda(), out=o[:e])
        torch.cuda.synchronize()
        y[r:r + e] = o[:e].cpu()
    return y


def lid_of(base: str) -> str:
    return base.rsplit(".", 1)[1]


def pick_layers(model_dir: Path) -> list[tuple[str, int]]:
    """(kind, layer index) of the first KDA and the first DSA MoE layer."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    keys = {"kda": "linear_attention", "dsa": "deepseek_sparse_attention"}
    out = []
    for kind, key in keys.items():
        for i, (attn, mlp) in enumerate(zip(cfg["layer_types"], cfg["mlp_layer_types"])):
            if attn == key and mlp == "sparse":
                out.append((kind, i))
                break
        else:
            raise ValueError(f"no {kind} MoE layer in {model_dir}")
    return out


def one_dense(out: Path, model_dir: Path, cp, prefix: str, base: str, hidden: torch.Tensor) -> dict:
    """One non-routed group's inputs and the engine's outputs, bit for bit."""
    from tensorfold.cuda.exl3 import linear

    t = cp.groups[prefix]
    lin = linear.Exl3Linear.load(model_dir, prefix, layout="stored")
    gid = prefix[len(base) + 1:]
    if lin.k == hidden.shape[1]:
        x = hidden
    else:
        g = torch.Generator().manual_seed(SEED + zlib.crc32(gid.encode()))
        x = torch.randn((max(ROWS), lin.k), generator=g).to(torch.bfloat16)
    xmap, ymap, sha = {}, {}, {}
    for m in ROWS:
        y = mm_rows(x, lin, m)
        save(out, f"{lid_of(base)}.{gid}.x{m}", x[:m])
        save(out, f"{lid_of(base)}.{gid}.y{m}", y)
        xmap[str(m)] = f"{lid_of(base)}.{gid}.x{m}.bin"
        ymap[str(m)] = f"{lid_of(base)}.{gid}.y{m}.bin"
        sha[str(m)] = sha_of(y)
    entry = {"id": gid, "name": prefix, "bits": int(t.bits), "codebook": t.codebook,
             "k": lin.k, "n": lin.n, "rows": list(ROWS), "x": xmap, "y": ymap, "sha256": sha}
    print(f"  {gid}: EXL3 {t.bits:g}-bit {t.codebook} K={lin.k} N={lin.n}")
    return entry


def expert_part(out: Path, model_dir: Path, cp, base: str, hidden: torch.Tensor, cfg: dict) -> dict:
    """All experts as one grouped call with the engine's routing, bit for bit."""
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.families.glm5_next.cuda import glue

    codebook = next(t.codebook for p, t in cp.groups.items() if p.startswith(f"{base}.mlp.experts.0."))
    gate, up, down = [], [], []
    k2 = {"gate": [], "up": [], "down": []}
    for e in range(cfg["n_routed_experts"]):
        for part, mats in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            mats.append(tuple(load_tensor(model_dir, f"{base}.mlp.experts.{e}.{part}.{x}", "cuda")
                              for x in ("trellis", "suh", "svh")))
    for mats, key in ((gate, "gate"), (up, "up"), (down, "down")):
        k2[key] = [generic.k2_of(m[0]) for m in mats]
    ex = generic.prepare(gate, up, down, codebook, device="cuda")
    nbytes = sum(m[0].element_size() * m[0].numel() for mats in (gate, up, down) for m in mats) \
        + sum(t.element_size() * t.numel() for t in (ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d))
    print(f"  experts: {ex.count} experts, {ex.dims}x{ex.width}, {nbytes / (1 << 30):.2f} GiB")

    # the routing, as the GLM engine's moe_block routes: glue.router on the bf16 gate weight, glue.select on the bias
    gate_w = load_tensor(model_dir, f"{base}.mlp.gate.weight", "cuda").to(torch.bfloat16).contiguous()
    bias = load_tensor(model_dir, f"{base}.mlp.gate.e_score_correction_bias", "cuda").float().contiguous()
    logits = torch.empty((hidden.shape[0], ex.count), dtype=torch.float32, device="cuda")
    glue.router(hidden.cuda(), gate_w, logits)
    topk = cfg["num_experts_per_tok"]
    pick = torch.empty((hidden.shape[0], topk + 1), dtype=torch.int32, device="cuda")
    wts = torch.empty((hidden.shape[0], topk + 1), dtype=torch.float32, device="cuda")
    glue.select(logits, bias, pick, wts, topk, ex.count, cfg["routed_scaling_factor"],
                cfg.get("norm_topk_prob", True))
    pick, wts = pick[:, :topk].contiguous(), wts[:, :topk].contiguous()

    scratch = generic.Scratch(ex, max(EXPERT_ROWS), topk)
    xmap, pmap, wmap, ymap, omap, sha = {}, {}, {}, {}, {}, {}
    for m in EXPERT_ROWS:
        y = generic.routed(hidden[:m].cuda(), pick[:m].cuda(), wts[:m].cuda(), ex, scratch, None, m,
                           limit=cfg.get("swiglu_limit", 10.0), act_mode=generic.ACT_BF16)
        torch.cuda.synchronize()
        for key, t in (("x", hidden[:m]), ("pick", pick[:m]), ("wts", wts[:m]),
                       ("y", scratch.y[:m * topk]), ("out", y)):
            save(out, f"{lid_of(base)}.experts.{key}{m}", t)
        xmap[str(m)] = f"{lid_of(base)}.experts.x{m}.bin"
        pmap[str(m)] = f"{lid_of(base)}.experts.pick{m}.bin"
        wmap[str(m)] = f"{lid_of(base)}.experts.wts{m}.bin"
        ymap[str(m)] = f"{lid_of(base)}.experts.y{m}.bin"
        omap[str(m)] = f"{lid_of(base)}.experts.out{m}.bin"
        sha[str(m)] = sha_of(y)
    return {"name": f"{base}.mlp.experts", "count": ex.count, "dims": ex.dims, "width": ex.width,
            "slots": topk, "codebook": codebook, "limit": cfg.get("swiglu_limit", 10.0),
            "act_mode": generic.ACT_BF16, "rows": list(EXPERT_ROWS), "gate_k2": k2["gate"],
            "up_k2": k2["up"], "down_k2": k2["down"], "x": xmap, "pick": pmap, "wts": wmap,
            "y": ymap, "out": omap, "sha256": sha}


def one_layer(out: Path, model_dir: Path, cp, cfg: dict, li: int, kind: str) -> dict:
    base = f"model.language_model.layers.{li}"
    print(f"{kind} layer {li}:")
    g = torch.Generator().manual_seed(SEED + li)
    hidden = torch.randn((max(ROWS), cfg["hidden_size"]), generator=g).to(torch.bfloat16)
    groups = [one_dense(out, model_dir, cp, prefix, base, hidden)
              for prefix in sorted(cp.groups)
              if prefix.startswith(base + ".") and ".experts." not in prefix]
    return {"id": kind, "layer": li, "name": base, "dims": cfg["hidden_size"],
            "groups": groups, "experts": expert_part(out, model_dir, cp, base, hidden, cfg)}


def write(out: Path, model_dir: Path) -> list:
    from tensorfold.cuda.exl3 import format as fmt

    cp = fmt.scan(model_dir)
    cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    layers = [one_layer(out, model_dir, cp, cfg, li, kind) for kind, li in pick_layers(model_dir)]
    print(f"{len(layers)} layers -> {out}")
    return layers
