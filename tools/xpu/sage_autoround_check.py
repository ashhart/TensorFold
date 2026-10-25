"""Checks the Sage W4A16-AutoRound layout against the EXL3 checkpoint of the same model, in numpy.

auto_round:auto_gptq, sym, 4 bit, group 128: qweight [K/8, N] (8 values along K in each int32, low bits first),
scales fp16 [K/128, N], qzeros [K/128, N/8]. w = scale * (q - zero), zero = stored qzero + 1 (GPTQ v1 storage).
Usage: python sage_autoround_check.py AUTOROUND_DIR EXL3_DIR
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sage_exl3_check as ex

def nib(a, axis):
    a = a.view(np.uint32)
    v = (a[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15
    return np.moveaxis(v, -1, axis + 1).reshape(*a.shape[:axis], a.shape[axis] * 8, *a.shape[axis + 1:]) if axis == 0 else v.reshape(*a.shape[:-1], a.shape[-1] * 8)

def gptq(model, name):
    q = nib(ex.tensor(model, name + ".qweight"), 0).astype(np.float64)  # [K, N]
    z = nib(ex.tensor(model, name + ".qzeros"), 1).astype(np.float64)  # [K/128, N]
    s = ex.tensor(model, name + ".scales").astype(np.float64)
    g = q.shape[0] // s.shape[0]
    zs = np.unique(z)
    return (q - np.repeat(z + 1, g, 0)) * np.repeat(s, g, 0), g, zs  # [K, N]

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

if __name__ == "__main__":
    adir, edir = sys.argv[1], sys.argv[2]
    P = "model.language_model.layers."
    for name, live_from in [("0.mlp.experts.0.gate_proj", None), ("0.mlp.experts.7.up_proj", None),
                            ("0.mlp.experts.255.down_proj", "0.mlp.experts.255.gate_proj"), ("0.mlp.shared_expert.down_proj", None),
                            ("0.linear_attn.in_proj_qkv", None), ("0.linear_attn.in_proj_a", None), ("3.self_attn.q_proj", None),
                            ("3.self_attn.o_proj", None)]:
        W, g, zs = gptq(adir, P + name)
        try:
            E = ex.check_group(edir, P + name)
        except KeyError:  # not EXL3 in that checkpoint: its bf16 weight [N, K]
            E = ex.tensor(edir, P + name + ".weight").astype(np.float64).T
        if live_from:  # FF channels with a zero gate column do not count (see sage_mlx_check.py)
            G, _, _ = gptq(adir, P + live_from)
            live = np.abs(G).max(0) > 0
            W, E = W[live], E[live]
        r = rel(W, E)
        print(f"{name}: group {g}, stored zeros {zs.astype(int).tolist()}, [K, N] {W.shape} vs EXL3: relative error {r:.4f}")
        assert W.shape == E.shape and r < 0.2, r
    for part in ("mlp.gate", "mlp.shared_expert_gate"):
        a = ex.tensor(adir, P + f"0.{part}.weight"); e = ex.tensor(edir, P + f"0.{part}.weight")
        print(f"0.{part}: {a.shape} bf16, max diff from EXL3's copy {np.abs(a - e).max():.3g}")
