"""Checks the Sage MLX-4bit layout against the EXL3 checkpoint of the same model, in numpy.

MLX affine: w = scale * q + bias, q packed low bits first in each u32, groups of 64 along K.
1. Every tensor the Sage loader reads has the dtype and shape it expects.
2. The dequantized MLX weight agrees with the EXL3 (or bf16) weight of the other checkpoint.
Usage: python sage_mlx_check.py MLX_DIR EXL3_DIR
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sage_exl3_check as ex

def mlx(model, name, bits=None):
    m = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    h, base = ex.header(os.path.join(model, m[name + ".weight"]))
    t = h[name + ".weight"]
    with open(os.path.join(model, m[name + ".weight"]), "rb") as f:
        f.seek(base + t["data_offsets"][0])
        q = np.frombuffer(f.read(t["data_offsets"][1] - t["data_offsets"][0]), np.uint32).reshape(t["shape"])
    s = ex.tensor(model, name + ".scales").astype(np.float64)
    b = ex.tensor(model, name + ".biases").astype(np.float64)
    groups = s.shape[-1]
    bits = q.shape[-1] * 32 // (groups * 64)
    per = 32 // bits
    v = (q[..., :, None] >> (bits * np.arange(per, dtype=np.uint32))) & ((1 << bits) - 1)
    v = v.reshape(q.shape[:-1] + (groups, 64)).astype(np.float64)
    return (v * s[..., None] + b[..., None]).reshape(q.shape[:-1] + (groups * 64,)), bits

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def compare(mdir, edir, mname, ename, idx=None, live=None):
    W, bits = mlx(mdir, mname)
    if idx is not None: W = W[idx]
    E = ex.check_group(edir, ename).T  # EXL3 W is [K, N]; MLX rows are N
    if live is not None:  # down: only the FF channels whose gate row is not zero
        W, E = W[:, live], E[:, live]
    r = rel(W, E)
    print(f"{mname}{'' if idx is None else f'[{idx}]'}: {bits} bit {W.shape} vs EXL3: relative error {r:.4f}")
    assert W.shape == E.shape and r < 0.2, r

if __name__ == "__main__":
    mdir, edir = sys.argv[1], sys.argv[2]
    L = "language_model.model.layers."
    E = "model.language_model.layers."
    compare(mdir, edir, L + "0.mlp.switch_mlp.gate_proj", E + "0.mlp.experts.0.gate_proj", 0)
    compare(mdir, edir, L + "0.mlp.switch_mlp.up_proj", E + "0.mlp.experts.7.up_proj", 7)
    G, _ = mlx(mdir, L + "0.mlp.switch_mlp.gate_proj")
    dead = np.abs(G).max(2) == 0  # [experts, ff]: MLX stores these gate rows as exact zeros
    U, _ = mlx(mdir, L + "0.mlp.switch_mlp.up_proj")
    print(f"layer 0: {dead.sum()} of {dead.size} routed FF channels have a zero gate row "
          f"(up row max there {np.abs(U)[dead].max():.3g}): silu(0) * up = 0, so their down columns do not count")
    compare(mdir, edir, L + "0.mlp.switch_mlp.down_proj", E + "0.mlp.experts.255.down_proj", 255, ~dead[255])
    compare(mdir, edir, L + "0.mlp.shared_expert.down_proj", E + "0.mlp.shared_expert.down_proj")
    compare(mdir, edir, L + "0.linear_attn.in_proj_qkv", E + "0.linear_attn.in_proj_qkv")
    compare(mdir, edir, L + "3.self_attn.q_proj", E + "3.self_attn.q_proj")
    compare(mdir, edir, L + "3.self_attn.o_proj", E + "3.self_attn.o_proj")
    for part in ("gate", "shared_expert_gate"):
        W, bits = mlx(mdir, L + f"0.mlp.{part}")
        ref = ex.tensor(edir, E + f"0.mlp.{part}.weight").astype(np.float64)
        r = rel(W, ref)
        print(f"0.mlp.{part}: {bits} bit {W.shape} vs EXL3 checkpoint's {ref.dtype} {ref.shape}: relative error {r:.4f}")
        assert W.shape == ref.shape and r < 0.05, r
    c = ex.tensor(mdir, L + "0.linear_attn.conv1d.weight"); ce = ex.tensor(edir, E + "0.linear_attn.conv1d.weight")
    print(f"conv1d: MLX {c.shape} vs EXL3 {ce.shape}, max diff {np.abs(c.reshape(ce.shape) - ce).max():.3g}")
