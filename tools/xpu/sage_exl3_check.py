"""Checks the lane mapping and the pipeline of zig/kernels/xpu/sage_exl3.cl in numpy, on a real EXL3 group.

1. The kernel's per-lane value list equals exl3_format.zig's tile placement (every tile position once).
2. rotate_in -> matvec -> rotate_out (the kernel's order, fp16 rotated input) equals x @ W, W fully dequantized.
3. If the vision shard is present: block 0's EXL3 q_proj against the stored bf16 attn.qkv.weight rows.
Usage: python sage_exl3_check.py MODEL_DIR [GROUP]
"""
import json, os, struct, sys
import numpy as np

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

DT = {"F16": np.float16, "BF16": np.uint16, "I16": np.int16, "I32": np.int32, "F32": np.float32}

def tensor(model, name):
    shard = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"][name]
    path = os.path.join(model, shard)
    h, base = header(path)
    t = h[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        arr = np.frombuffer(f.read(b - a), dtype=DT[t["dtype"]]).reshape(t["shape"])
    if t["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr

def mul1(s):
    x = (s.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    return (h.astype(np.float64) * float(np.float16(0.00676727294921875)) + -10.3828125).astype(np.float16)

def states(tile, halves):
    words = tile.view(np.uint16).astype(np.uint32)
    w32 = words[0::2] | (words[1::2] << 16)
    bits = ((w32[:, None] >> (31 - np.arange(32))) & 1).reshape(-1)
    ring = 128 * halves
    p1 = np.arange(1, 257)
    end = (p1 * halves - ((p1 & 1) if halves & 1 else 0)) // 2
    start = (end + ring - 16) % ring
    idx = (start[:, None] + np.arange(16)) % ring
    return (bits[idx] << (15 - np.arange(16))).sum(1)

def placement():
    p = np.arange(256); lane = p // 8; j = p % 8
    return 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), lane // 4 + 8 * (j >> 2)

def unpack(trellis):
    tk, tn, words = trellis.shape
    halves = words // 8
    row, col = placement()
    out = np.zeros((tk * 16, tn * 16), np.float16)
    for a in range(tk):
        for b in range(tn):
            out[a * 16 + row, b * 16 + col] = mul1(states(trellis[a, b], halves))
    return out

def hadamard(n=128):
    i = np.arange(n)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(n)

def rot(x, axis):
    H = hadamard()
    if axis == 0:
        k = x.shape[0]
        return np.concatenate([H @ x[b:b + 128] for b in range(0, k, 128)], 0)
    return np.concatenate([x[:, b:b + 128] @ H.T for b in range(0, x.shape[1], 128)], 1)

def check_lanes():
    row, col = placement()
    seen = set()
    for c in range(16):
        for m in range(4):
            for i in range(4):
                p = 32 * (c % 8) + 8 * m + 4 * (c // 8) + i
                assert col[p] == c and row[p] == 2 * m + (i & 1) + 8 * (i >> 1), (c, m, i, p)
                seen.add(p)
    assert len(seen) == 256
    print("lanes: the kernel's lane map equals the tile placement (256 of 256)")

def check_group(model, g, rows=4):
    tr = tensor(model, g + ".trellis"); suh = tensor(model, g + ".suh").astype(np.float64)
    svh = tensor(model, g + ".svh").astype(np.float64)
    wq = unpack(tr).astype(np.float64)
    W = rot(rot(wq, 0) * suh[:, None], 1) * svh[None, :]  # the reference's order
    x = np.random.default_rng(0).standard_normal((rows, wq.shape[0])) * 0.5
    ref = x @ W
    xr = rot((x * suh[None, :]).T, 0).T.astype(np.float16).astype(np.float64)  # rotate_in, fp16 out
    y = rot(xr @ wq, 1) * svh[None, :]
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"{g}: K={wq.shape[0]} N={wq.shape[1]} bits={tr.shape[2] // 16} |W| std={W.std():.4g} max={np.abs(W).max():.4g} "
          f"pipeline vs dequant rel err {err:.2e}")
    assert err < 2e-3
    return W

def check_vision(model):
    m = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    shard = m["model.visual.blocks.0.attn.qkv.weight"]
    if not os.path.exists(os.path.join(model, shard)):
        print("vision: shard not present, skipped"); return
    W = check_group(model, "model.visual.blocks.0.attn.q_proj")
    qkv = tensor(model, "model.visual.blocks.0.attn.qkv.weight").astype(np.float64)
    ref = qkv[: W.shape[1]].T  # [K, N]
    rel = np.linalg.norm(W - ref) / np.linalg.norm(ref)
    print(f"vision: EXL3 q_proj vs bf16 qkv rows, relative Frobenius error {rel:.4f}")

if __name__ == "__main__":
    model = sys.argv[1]
    check_lanes()
    check_group(model, sys.argv[2] if len(sys.argv) > 2 else "model.language_model.layers.0.mlp.experts.0.gate_proj")
    check_vision(model)
