"""The tensor-parallel fixture: two real projections and one MoE layer's routed experts, split as the ranks are."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch

import exl3_oracle_experts

ROWS = (1, 3, 16)
SEED = 1237
EXPERTS_SEED = 1238
POOL = 3  # experts per width, as the routed fixture's pool
SLOTS = 8
LIMIT = 10.0


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def cut_dim1(raw: torch.Tensor, rank: int) -> torch.Tensor:
    """A tensor's second axis in half (the rank's tile columns / scale rows), contiguous."""
    half = raw.shape[1] // 2
    return raw[:, rank * half:(rank + 1) * half].contiguous()


def cut_row(raw: torch.Tensor, rank: int) -> torch.Tensor:
    """A tensor's leading axis in half (the rank's tile rows), contiguous."""
    half = raw.shape[0] // 2
    return raw[rank * half:(rank + 1) * half].contiguous()


def load_group(model_dir: Path, prefix: str) -> tuple:
    """(trellis int16 as stored, suh, svh) of one linear, from the checkpoint (on the GPU)."""
    from safetensors import safe_open

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    rel = None
    if index.exists():
        weight_map = __import__("json").loads(index.read_text())["weight_map"]
        rel = weight_map.get(f"{prefix}.trellis")
        if rel is None:
            for part in sorted(root.glob("*.safetensors")):
                with safe_open(str(part), framework="pt") as g:
                    if f"{prefix}.trellis" in set(g.keys()):
                        rel = part.name
                        break
        if rel is None:
            raise ValueError(f"{prefix}.trellis is in no file of {root}")
    with safe_open(str(root / (rel or "model.safetensors")), framework="pt", device="cuda") as f:
        return (f.get_tensor(f"{prefix}.trellis"), f.get_tensor(f"{prefix}.suh"), f.get_tensor(f"{prefix}.svh"))


def pick_out_linear(cp) -> object:
    """The out-split layer: a shared_experts.gate_proj (or any gate/up one) whose cut stays on 128 boundaries."""
    cands = [t for t in cp.groups.values()
             if re.search(r"\.(mlp\.shared_experts\.)?(gate|up)_proj$", t.prefix)
             and t.in_scales == "suh" and t.out_scales == "svh"
             and t.k % 256 == 0 and t.n % 256 == 0 and t.k <= 8192 and t.n <= 8192]
    if not cands:
        raise ValueError("no EXL3 gate/up projection splits on 128-boundaries at world 2")
    shared = [t for t in cands if "shared_experts" in t.prefix]
    return min(shared or cands, key=lambda t: (t.k * t.n, t.prefix))


def pick_in_linear(cp, out) -> object:
    """The in-split layer: the same layer's shared_experts.down_proj, else a down_proj, else an o_proj."""
    base = out.prefix.rsplit(".mlp.", 1)[0] if ".mlp." in out.prefix else out.prefix.rsplit(".self_attn.", 1)[0]
    prefs = (f"{base}.mlp.shared_experts.down_proj", f"{base}.mlp.down_proj")
    for name in prefs:
        t = cp.groups.get(name)
        if t is not None and t.in_scales == "suh" and t.out_scales == "svh" and t.k % 256 == 0 and t.n % 256 == 0:
            return t
    cands = [t for t in cp.groups.values()
             if (t.prefix.endswith(".mlp.down_proj") or t.prefix.endswith(".self_attn.o_proj"))
             and t.in_scales == "suh" and t.out_scales == "svh" and t.k % 256 == 0 and t.n % 256 == 0
             and t.k <= 8192 and t.n <= 8192]
    if not cands:
        raise ValueError("no EXL3 down/o projection splits on 128-boundaries at world 2")
    return min(cands, key=lambda t: (t.k * t.n, t.prefix))


def linear_entry(out: Path, model_dir: Path, t, split: str, lid: str) -> dict:
    """One projection's full tensors, x rows, and per rank the engine's output on its shard."""
    from tensorfold.cuda.exl3.linear import Exl3Linear

    trellis, suh, svh = load_group(model_dir, t.prefix)
    save(out, f"{lid}.trellis", trellis)
    save(out, f"{lid}.suh", suh)
    save(out, f"{lid}.svh", svh)
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn((max(ROWS), t.k), generator=g).to(torch.bfloat16)
    xmap, sha = {}, {}
    for m in ROWS:
        save(out, f"{lid}.x{m}", x[:m])
        xmap[str(m)] = f"{lid}.x{m}.bin"

    if split == "dim1":                       # out-split: the rank's bf16 half of the outputs
        shards = {}
        for rank in (0, 1):
            lin = Exl3Linear.from_tensors(cut_dim1(trellis, rank), suh, cut_row(svh, rank), t.codebook,
                                          None, layout="stored")
            for m in ROWS:
                y = lin(x[:m].cuda())
                torch.cuda.synchronize()
                save(out, f"{lid}.shard{m}.r{rank}", y)
                shards.setdefault(str(m), []).append(f"{lid}.shard{m}.r{rank}.bin")
        entry = {"shards": shards}
    else:                                     # in-split: the rank's fp32 partial over the full outputs
        partials, ymap = {}, {}
        plist = {0: [], 1: []}
        for rank in (0, 1):
            lin = Exl3Linear.from_tensors(cut_row(trellis, rank), cut_row(suh, rank), svh, t.codebook,
                                          None, layout="stored")
            xr = x[:, rank * (t.k // 2):(rank + 1) * (t.k // 2)].contiguous()
            for m in ROWS:
                part = lin(xr[:m].cuda(), out_dtype=torch.float32)
                torch.cuda.synchronize()
                save(out, f"{lid}.part{m}.r{rank}", part)
                partials.setdefault(str(m), []).append(f"{lid}.part{m}.r{rank}.bin")
                plist[rank].append(part)
        for m, p0, p1 in zip(ROWS, plist[0], plist[1]):
            y = (p0 + p1).to(torch.bfloat16)   # rank 0 first, fp32, then bf16
            save(out, f"{lid}.y{m}", y)
            ymap[str(m)] = f"{lid}.y{m}.bin"
            sha[str(m)] = hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        entry = {"partials": partials, "y": ymap, "sha256": sha}

    print(f"{lid}: {t.prefix} EXL3 {t.bits:g}-bit {t.codebook} K={t.k} N={t.n} split={split} "
          f"sha256={sha.get('16', '')[:16]}…")
    return {"id": lid.rsplit(".", 1)[1], "name": t.prefix, "split": split, "bits": int(t.bits),
            "codebook": t.codebook, "k": t.k, "n": t.n, "rows": list(ROWS),
            "trellis": f"{lid}.trellis.bin", "suh": f"{lid}.suh.bin", "svh": f"{lid}.svh.bin",
            "bias": None, "x": xmap, **entry}


def experts_entry(out: Path, model_dir: Path, cp) -> dict:
    """One MoE layer's routed experts, full tensors and each rank's fp32 partials (its gate/up/down shards)."""
    from tensorfold.cuda.exl3 import experts as generic

    base = exl3_oracle_experts.pick_layer(cp)
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
            names.append(f"tp2.experts.{short}{slot}.bin")
    save(out, "tp2.experts.suh_g", torch.stack([m[1] for m in gate]))
    save(out, "tp2.experts.suh_u", torch.stack([m[1] for m in up]))
    save(out, "tp2.experts.svh_g", torch.stack([m[2] for m in gate]))
    save(out, "tp2.experts.svh_u", torch.stack([m[2] for m in up]))
    save(out, "tp2.experts.suh_d", torch.stack([m[1] for m in down]))
    save(out, "tp2.experts.svh_d", torch.stack([m[2] for m in down]))
    for slot, mats in enumerate(zip(gate, up, down)):
        for name, t in zip(("gate", "up", "down"), mats):
            save(out, f"tp2.experts.{name}{slot}", t[0])

    g = torch.Generator().manual_seed(EXPERTS_SEED)
    dims, width = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    x = torch.randn((max(ROWS), dims), generator=g).to(torch.bfloat16)
    pick = torch.stack([torch.randperm(len(pool), generator=g)[:SLOTS] for _ in range(max(ROWS))]).int()
    wts = (0.1 + 0.9 * torch.rand((max(ROWS), SLOTS), generator=g)).float()
    xmap, pickmap, wtsmap = {}, {}, {}
    for m in ROWS:
        save(out, f"tp2.experts.x{m}", x[:m])
        save(out, f"tp2.experts.pick{m}", pick[:m])
        save(out, f"tp2.experts.wts{m}", wts[:m])
        xmap[str(m)] = f"tp2.experts.x{m}.bin"
        pickmap[str(m)] = f"tp2.experts.pick{m}.bin"
        wtsmap[str(m)] = f"tp2.experts.wts{m}.bin"

    partials, ymap, sha = {}, {}, {}
    plist = {0: [], 1: []}
    for rank in (0, 1):
        def shard(mats, part):
            """The rank's (trellis, suh, svh) of one expert projection: out-split for gate/up, in for down."""
            cut = cut_dim1 if part in ("gate", "up") else (lambda t, r: cut_row(t, r))
            scale_at = 2 if part in ("gate", "up") else 1
            return [(cut(t, rank), suh if scale_at == 2 else cut_row(suh, rank),
                     cut_row(svh, rank) if scale_at == 2 else svh) for t, suh, svh in mats]

        ex = generic.prepare(shard(gate, "gate"), shard(up, "up"), shard(down, "down"), "mul1")
        scratch = generic.Scratch(ex, max(ROWS), SLOTS)
        for m in ROWS:
            part = generic.routed(x[:m].cuda(), pick[:m].cuda(), wts[:m].cuda(), ex, scratch, None, m,
                                  limit=LIMIT, act_mode=generic.ACT_BF16)
            torch.cuda.synchronize()
            save(out, f"tp2.experts.part{m}.r{rank}", part)
            partials.setdefault(str(m), []).append(f"tp2.experts.part{m}.r{rank}.bin")
            plist[rank].append(part)
    for m, p0, p1 in zip(ROWS, plist[0], plist[1]):
        y = (p0 + p1).to(torch.bfloat16)      # rank 0 first, fp32, then bf16
        save(out, f"tp2.experts.y{m}", y)
        ymap[str(m)] = f"tp2.experts.y{m}.bin"
        sha[str(m)] = hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    print(f"tp2 experts: {base} {len(pool)} experts (4/5/6-bit pool), width {width}, "
          f"sha256={sha['16'][:16]}…")
    return {"name": base, "count": len(pool), "dims": dims, "width": width, "slots": SLOTS,
            "codebook": "mul1", "limit": LIMIT, "act_mode": generic.ACT_BF16, "rows": list(ROWS),
            "gate_k2": [generic.k2_of(m[0]) for m in gate],
            "up_k2": [generic.k2_of(m[0]) for m in up],
            "down_k2": [generic.k2_of(m[0]) for m in down],
            "trellis": trellis,
            "suh_g": "tp2.experts.suh_g.bin", "suh_u": "tp2.experts.suh_u.bin",
            "svh_g": "tp2.experts.svh_g.bin", "svh_u": "tp2.experts.svh_u.bin",
            "suh_d": "tp2.experts.suh_d.bin", "svh_d": "tp2.experts.svh_d.bin",
            "x": xmap, "pick": pickmap, "wts": wtsmap, "partials": partials, "y": ymap, "sha256": sha}


def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.cuda.exl3 import format as fmt

    cp = fmt.scan(model_dir)
    t_out = pick_out_linear(cp)
    t_in = pick_in_linear(cp, t_out)
    return {"linears": [linear_entry(out, model_dir, t_out, "dim1", "tp2.linear.out"),
                        linear_entry(out, model_dir, t_in, "row", "tp2.linear.in")],
            "experts": experts_entry(out, model_dir, cp)}
