"""The Python engine's EXL3 outputs on MiMo's rank-0 cuts, as bytes: `python3 mimo_exl3_fixtures.py MODEL OUT`."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3.linear import Exl3Linear

model, out = Path(sys.argv[1]), Path(sys.argv[2])
wm = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(name):
    with safe_open(str(model / wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def triple(prefix):
    return tensor(prefix + ".trellis").cuda(), tensor(prefix + ".suh").cuda(), tensor(prefix + ".svh").cuda()


def cut(t, lo, hi, axis):
    tr, suh, svh = t
    if axis == "n":
        return tr[:, lo // 16:hi // 16].contiguous(), suh, svh[lo:hi].contiguous()
    return tr[lo // 16:hi // 16].clone(), suh[lo:hi].clone(), svh


def save(d, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        (d / f"{k}.bin").write_bytes(v.contiguous().cpu().view(torch.uint8).numpy().tobytes())


# rank-0 cuts at TP=2, as weights.py makes them (heads 64 / 2, qk 192, v 128, dense 16384 / 2)
CASES = [
    ("q", "model.layers.1.self_attn.q_proj", "n", 0, 32 * 192),
    ("o", "model.layers.1.self_attn.o_proj", "k", 0, 32 * 128),
    ("dense-gate", "model.layers.0.mlp.gate_proj", "n", 0, 8192),
    ("dense-down", "model.layers.0.mlp.down_proj", "k", 0, 8192),
]
torch.manual_seed(1234)
for name, prefix, axis, lo, hi in CASES:
    lin = Exl3Linear.from_tensors(*cut(triple(prefix), lo, hi, axis), codebook="mul1", device="cuda")
    for xdt, xname in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        for odt, oname in ((xdt, xname), (torch.float32, "f32")):
            for m in (1, 4, 8, 16):
                x = torch.randn((m, lin.k), device="cuda").to(xdt)
                y = torch.empty((m, lin.n), dtype=odt, device="cuda")
                lin(x, out=y)
                torch.cuda.synchronize()
                d = out / f"{name}-x{xname}-y{oname}-m{m}"
                save(d, x=x, y=y)
                (d / "meta.json").write_text(json.dumps({"prefix": prefix, "axis": axis, "lo": lo, "hi": hi, "m": m,
                                                         "x": xname, "y": oname, "k": lin.k, "n": lin.n, "k2": lin.k2,
                                                         "split": list(lin.split)}))
    print("linear", name, lin.k, lin.n, "k2", lin.k2, "split", lin.split, flush=True)

# one routed window: layer 1's experts, rank-0 width (2048 / 2), 8 slots, R rows
layer, ni, E, slots = 1, 1024, 256, 8
gate, up, down = [], [], []
for e in range(E):
    p = f"model.layers.{layer}.mlp.experts.{e}."
    gate.append(cut(triple(p + "gate_proj"), 0, ni, "n"))
    up.append(cut(triple(p + "up_proj"), 0, ni, "n"))
    down.append(cut(triple(p + "down_proj"), 0, ni, "k"))
ex = x3experts.prepare(gate, up, down, "mul1", device="cuda")
for R in (1, 4, 16):
    sc = x3experts.Scratch(ex, 1024, slots, device="cuda")
    x = torch.randn((R, ex.dims), device="cuda").to(torch.float16)
    pick = torch.stack([torch.randperm(E, device="cuda")[:slots] for _ in range(R)]).to(torch.int32).contiguous()
    wts = torch.rand((R, slots), device="cuda", dtype=torch.float32)
    o = torch.empty((R, ex.dims), dtype=torch.float32, device="cuda")
    x3experts.routed(x, pick, wts, ex, sc, o, R)
    torch.cuda.synchronize()
    d = out / f"experts-L{layer}-r{R}"
    save(d, x=x, pick=pick, wts=wts, out=o)
    (d / "meta.json").write_text(json.dumps({"layer": layer, "width": ni, "experts": E, "slots": slots, "rows": R,
                                             "cfg_gu": list(sc.cfg_gu), "cfg_d": list(sc.cfg_d),
                                             "k2_gu": list(ex.k2_gu), "k2_d": list(ex.k2_d)}))
    print("experts", R, "cfg", sc.cfg_gu, sc.cfg_d, "k2", ex.k2_gu, ex.k2_d, flush=True)
