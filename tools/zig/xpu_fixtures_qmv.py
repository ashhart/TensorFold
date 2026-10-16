"""Fixture for the 4-bit matvec: 256 rows of a real tensor, a random bf16 x and the expected fp32 y."""
import json, os, struct, sys, numpy as np

ROWS = 256
CK = os.environ.get("TF_NEMOTRON_DIR", os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"))
FIX = os.environ.get("TF_FIXTURES_DIR", "tensorfold-fixtures")
os.makedirs(FIX, exist_ok=True)
NAME = "backbone.layers.0.mixer.out_proj"  # 2688 x 4096
wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]

def tensor(name, rows):
    fn = CK + "/" + wm[name]
    with open(fn, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[name]
        a, b = h["data_offsets"]
        shape = h["shape"]
        row_bytes = (b - a) // shape[0]
        fh.seek(8 + n + a)
        return fh.read(row_bytes * rows), shape, h["dtype"]

def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

w, wshape, _ = tensor(NAME + ".weight", ROWS)
s, sshape, _ = tensor(NAME + ".scales", ROWS)
b, _, _ = tensor(NAME + ".biases", ROWS)
words, groups = wshape[1], sshape[1]
in_dim = words * 8
assert in_dim == groups * 64
rng = np.random.default_rng(7)
xf = rng.standard_normal(in_dim).astype(np.float32)
x16 = (xf.view(np.uint32) >> 16).astype(np.uint16)  # truncated bf16
x = bf16(x16.tobytes())
W = np.frombuffer(w, dtype=np.uint32).reshape(ROWS, words)
q = np.stack([(W >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(ROWS, in_dim).astype(np.float32)
sc = np.repeat(bf16(s).reshape(ROWS, groups), 64, axis=1)
bi = np.repeat(bf16(b).reshape(ROWS, groups), 64, axis=1)
y = ((q * sc + bi).astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
open(FIX + "/qmv_w.bin", "wb").write(w)
open(FIX + "/qmv_s.bin", "wb").write(s)
open(FIX + "/qmv_b.bin", "wb").write(b)
open(FIX + "/qmv_x.bin", "wb").write(x16.tobytes())
open(FIX + "/qmv_y.bin", "wb").write(y.tobytes())
open(FIX + "/qmv_dims.txt", "w").write(f"{ROWS} {in_dim}\n")
print("rows", ROWS, "in_dim", in_dim, "words", words, "groups", groups, "y[:4]", y[:4])
