"""One-off: kda_inner.py vs Model.kda (r = identity) on a fake Ckpt. No edit to ling_forward.py."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lf = load("ling_forward", HERE / "ling_forward.py")
inner = load("kda_inner", HERE / "kda_inner.py")


class FakeCkpt:
    """Enough of Ckpt for Model.kda: t() and lin() over a dict of tensors."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.keys = set(tensors)

    def t(self, name):
        return self.tensors[name]

    def lin(self, module):
        return self.t(module + ".weight").double()


class CaptureModel(lf.Model):
    def lin(self, x, module):
        y = super().lin(x, module)
        self.captured[module] = y.detach().clone()
        if module.endswith("o_proj"):
            self.y_pre_o = x.detach().clone()
        return y


def make(g, heads=16, dim=128, hidden=64, taps=4):
    W = heads * dim
    p = "model.layers.0.attention."

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    tensors = {
        p + "q_proj.weight": rand(W, hidden),
        p + "k_proj.weight": rand(W, hidden),
        p + "v_proj.weight": rand(W, hidden),
        p + "f_proj.weight": rand(W, hidden),
        p + "b_proj.weight": rand(heads, hidden),
        p + "g_proj.weight": rand(W, hidden),
        p + "o_proj.weight": rand(hidden, W),
        p + "q_conv1d.weight": rand(W, 1, taps),
        p + "k_conv1d.weight": rand(W, 1, taps),
        p + "v_conv1d.weight": rand(W, 1, taps),
        p + "A_log": rand(heads),
        p + "dt_bias": rand(W),
        p + "o_norm.weight": 0.5 + 0.5 * rand(dim),
    }
    cfg = {
        "num_attention_heads": heads,
        "head_dim": dim,
        "kda_lower_bound": -5.0,
        "rms_norm_eps": 1e-6,
    }
    return p, tensors, cfg, hidden


def flat(t):
    return t.detach().cpu().contiguous().view(-1).tolist()


def run(rows, heads=16, dim=128, seed=0x6B63):
    g = torch.Generator().manual_seed(seed + rows)
    p, tensors, cfg, hidden = make(g, heads, dim)
    x = torch.randn(rows, hidden, generator=g, dtype=torch.float64)
    model = CaptureModel(FakeCkpt(tensors), cfg, lambda z: z)
    model.captured = {}
    model.y_pre_o = None
    with torch.no_grad():
        out = model.kda(x, p)
    W = heads * dim
    proj = torch.cat([model.captured[p + n] for n in ("q_proj", "k_proj", "v_proj")], -1)
    got = inner.kda_inner(
        flat(proj),
        flat(model.captured[p + "f_proj"]),
        flat(model.captured[p + "b_proj"]),
        flat(model.captured[p + "g_proj"]),
        flat(tensors[p + "q_conv1d.weight"][:, 0, :]),
        flat(tensors[p + "k_conv1d.weight"][:, 0, :]),
        flat(tensors[p + "v_conv1d.weight"][:, 0, :]),
        flat(tensors[p + "A_log"]),
        flat(tensors[p + "dt_bias"]),
        flat(tensors[p + "o_norm.weight"]),
        heads,
        dim,
        4,
        cfg["kda_lower_bound"],
        cfg["rms_norm_eps"],
        rows,
    )
    want = model.y_pre_o.view(-1)
    got_t = torch.tensor(got, dtype=torch.float64)
    worst_y = (got_t - want).abs().max().item()
    y = got_t.view(rows, W)
    o = y @ tensors[p + "o_proj.weight"].T
    worst_out = (o - out).abs().max().item()
    print(
        f"{heads} heads x {rows} rows  y worst abs {worst_y:.6e}  "
        f"o_proj worst abs {worst_out:.6e}  y rms {math.sqrt((want * want).mean().item()):.6f}"
    )
    return worst_y, worst_out


if __name__ == "__main__":
    print("kda_inner.py vs Model.kda r=identity, fake Ckpt, no edit to ling_forward.py")
    worst = 0.0
    for rows in (2, 4):
        wy, wo = run(rows)
        worst = max(worst, wy, wo)
    print("worst abs", f"{worst:.6e}")
