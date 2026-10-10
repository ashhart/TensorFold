"""Fixtures for rmsnorm and the 4-bit embedding lookup of the Nemotron checkpoint (numpy reference)."""
import json, os, struct, numpy as np

CK = os.environ.get("TF_NEMOTRON_DIR", os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"))
FIX = os.environ.get("TF_FIXTURES_DIR", "tensorfold-fixtures")
os.makedirs(FIX, exist_ok=True)
wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]

def raw(name, row0=0, rows=None):
    with open(CK + "/" + wm[name], "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[name]
        a, b = h["data_offsets"]
        shape = h["shape"]
        rb = (b - a) // shape[0]
        rows = shape[0] - row0 if rows is None else rows
        fh.seek(8 + n + a + row0 * rb)
        return fh.read(rb * rows), shape

def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

# rmsnorm
wb, shape = raw("backbone.layers.0.norm.weight")
hidden = shape[0]
w = f32(np.frombuffer(wb, dtype=np.uint16))
rng = np.random.default_rng(11)
x16 = to_bf16(rng.standard_normal((3, hidden)).astype(np.float32) * 3)
x = f32(x16)
ms = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
y = to_bf16((x * (1.0 / np.sqrt(ms + 1e-5)).astype(np.float32) * w).astype(np.float32))
open(FIX + "/rms_w.bin", "wb").write(wb)
open(FIX + "/rms_x.bin", "wb").write(x16.tobytes())
open(FIX + "/rms_y.bin", "wb").write(y.tobytes())

# embedding rows
ids = [5, 1000, 131071]
ws, ss, bs = [], [], []
for t in ids:
    ws.append(raw("backbone.embeddings.weight", t, 1)[0])
    ss.append(raw("backbone.embeddings.scales", t, 1)[0])
    bs.append(raw("backbone.embeddings.biases", t, 1)[0])
words = len(ws[0]) // 4
groups = len(ss[0]) // 2
W = np.frombuffer(b"".join(ws), dtype=np.uint32).reshape(3, words)
q = np.stack([(W >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(3, words * 8).astype(np.float32)
sc = np.repeat(f32(np.frombuffer(b"".join(ss), dtype=np.uint16)).reshape(3, groups), 64, axis=1)
bi = np.repeat(f32(np.frombuffer(b"".join(bs), dtype=np.uint16)).reshape(3, groups), 64, axis=1)
e = to_bf16(q * sc + bi)
open(FIX + "/emb_w.bin", "wb").write(b"".join(ws))
open(FIX + "/emb_s.bin", "wb").write(b"".join(ss))
open(FIX + "/emb_b.bin", "wb").write(b"".join(bs))
open(FIX + "/emb_y.bin", "wb").write(e.tobytes())
print("hidden", hidden, "emb dim", words * 8, "groups", groups)
