"""Fixtures for the Nemotron attention block, long-cache split-K attention and the lm_head (numpy reference)."""
import json, os, struct, numpy as np

CK = os.environ.get("TF_NEMOTRON_DIR", os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"))
FIX = os.environ.get("TF_FIXTURES_DIR", "tensorfold-fixtures")
os.makedirs(FIX, exist_ok=True)
wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]
LAYER, T = 5, 12
HEADS, KVH, HD = 32, 2, 128
CHUNK = 512
LENS = [1, 2, 63, 64, 65, 300, 512, 513, 1030, 4096]
HEAD_ROW0, HEAD_ROWS = 100000, 2048
F32 = np.float32

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
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(F32)

def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=F32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def rb(x):  # round fp32 values to bf16 and back
    return f32(to_bf16(x))

def dequant(prefix, row0=0, rows=None):
    w, ws = raw(prefix + ".weight", row0, rows)
    s, _ = raw(prefix + ".scales", row0, rows)
    b, _ = raw(prefix + ".biases", row0, rows)
    W = np.frombuffer(w, dtype=np.uint32).reshape(-1, ws[1])
    n = W.shape[0]
    q = np.stack([(W >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(n, -1).astype(F32)
    sc = np.repeat(f32(np.frombuffer(s, dtype=np.uint16)).reshape(n, -1), 64, axis=1)
    bi = np.repeat(f32(np.frombuffer(b, dtype=np.uint16)).reshape(n, -1), 64, axis=1)
    return (q * sc + bi).astype(F32), (w, s, b)

def matvec(Wd, x):  # fp64 accumulation, rounded once to fp32
    return (Wd.astype(np.float64) @ x.astype(np.float64)).astype(F32)

def dump(name, arr):
    open(f"{FIX}/attn_{name}.bin", "wb").write(arr.tobytes() if hasattr(arr, "tobytes") else arr)

SCALE = F32(HD ** -0.5)

def emu_decode(q, K, V, chunk):
    """Upstream chunked online softmax for one token: q (32,128), K/V (L,2,128), all bf16-valued fp32."""
    L = K.shape[0]
    out = np.zeros((HEADS, HD), F32)
    g = HEADS // KVH
    for h in range(HEADS):
        hk = h // g
        parts = []
        for c0 in range(0, L, chunk):
            m, den, o = F32(-np.inf), F32(0), np.zeros(HD, F32)
            for t0 in range(c0, min(L, c0 + chunk), 64):
                t1 = min(L, c0 + chunk, t0 + 64)
                s = matvec(K[t0:t1, hk], q[h]) * SCALE
                nm = max(m, s.max())
                a = F32(0) if m == -np.inf else np.exp(F32(m - nm)).astype(F32)
                p = np.exp(s - nm).astype(F32)
                o = (o * a + matvec(V[t0:t1, hk].T, rb(p))).astype(F32)
                den = F32(den * a + p.sum(dtype=F32))
                m = F32(nm)
            parts.append((o, m, den))
        m, den, o = F32(-np.inf), F32(0), np.zeros(HD, F32)
        for co, cm, cl in parts:
            if not cl > 0:
                continue
            nm = max(m, cm)
            a = F32(0) if m == -np.inf else np.exp(F32(m - nm)).astype(F32)
            b = np.exp(F32(cm - nm)).astype(F32)
            o = (o * a + co * b).astype(F32)
            den = F32(den * a + cl * b)
            m = F32(nm)
        out[h] = rb(o / den)
    return out

def exact_decode(q, K, V):
    """fp64 softmax attention over bf16 inputs, rounded once to bf16."""
    g = HEADS // KVH
    out = np.zeros((HEADS, HD), F32)
    for h in range(HEADS):
        hk = h // g
        s = K[:, hk].astype(np.float64) @ q[h].astype(np.float64) * float(SCALE)
        p = np.exp(s - s.max())
        out[h] = rb((p / p.sum()) @ V[:, hk].astype(np.float64))
    return out

# ---- layer 5: projections, KV cache, attention, o_proj over 12 tokens
pre = f"backbone.layers.{LAYER}.mixer."
Wq, qraw = dequant(pre + "q_proj")
Wk, kraw = dequant(pre + "k_proj")
Wv, vraw = dequant(pre + "v_proj")
Wo, oraw = dequant(pre + "o_proj")
for tag, (w, s, b) in zip("qkvo", (qraw, kraw, vraw, oraw)):
    dump(tag + "w", w); dump(tag + "s", s); dump(tag + "b", b)
rng = np.random.default_rng(5)
h16 = to_bf16(rng.standard_normal((T, Wq.shape[1])).astype(F32))
dump("h", h16.tobytes())
H = f32(h16)
Q = rb(np.stack([matvec(Wq, H[t]) for t in range(T)]))
Kn = rb(np.stack([matvec(Wk, H[t]) for t in range(T)]))
Vn = rb(np.stack([matvec(Wv, H[t]) for t in range(T)]))
A = np.zeros((T, HEADS * HD), F32)
Ae = np.zeros((T, HEADS * HD), F32)
for t in range(T):
    kk, vv = Kn[:t + 1].reshape(t + 1, KVH, HD), Vn[:t + 1].reshape(t + 1, KVH, HD)
    A[t] = emu_decode(Q[t].reshape(HEADS, HD), kk, vv, CHUNK).reshape(-1)
    Ae[t] = exact_decode(Q[t].reshape(HEADS, HD), kk, vv).reshape(-1)
Y = rb(np.stack([matvec(Wo, A[t]) for t in range(T)]))
for name, arr in (("q", Q), ("k", Kn), ("v", Vn), ("o", A), ("oexact", Ae), ("y", Y)):
    dump("x_" + name, to_bf16(arr).tobytes())
print("layer5: |q| max", np.abs(Q).max(), "|attn| max", np.abs(A).max(), "|y| max", np.abs(Y).max())

# ---- long caches: synthetic q/K/V, expected for several lengths
rng = np.random.default_rng(9)
LQ = rb(rng.standard_normal((HEADS, HD)).astype(F32) * 2)
LK = rng.standard_normal((max(LENS), KVH, HD)).astype(F32)
LK[rng.integers(0, max(LENS), 40)] *= 3.0  # a few sharp keys
LK = rb(LK)
LV = rb(rng.standard_normal((max(LENS), KVH, HD)).astype(F32))
dump("lq", to_bf16(LQ).tobytes()); dump("lk", to_bf16(LK).tobytes()); dump("lv", to_bf16(LV).tobytes())
emu, ex = [], []
for L in LENS:
    emu.append(to_bf16(emu_decode(LQ, LK[:L], LV[:L], CHUNK)))
    ex.append(to_bf16(exact_decode(LQ, LK[:L], LV[:L])))
dump("ly", np.stack(emu)); dump("lyexact", np.stack(ex))
open(FIX + "/attn_dims.txt", "w").write(" ".join(map(str, LENS)) + "\n")

# ---- final norm + lm_head rows
nw, _ = raw("backbone.norm_f.weight")
nwf = f32(np.frombuffer(nw, dtype=np.uint16))
dump("nw", nw)
rng = np.random.default_rng(13)
x16 = to_bf16(rng.standard_normal((1, nwf.shape[0])).astype(F32) * 3)
x = f32(x16)
ms = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
xn16 = to_bf16((x * (1.0 / np.sqrt(ms + 1e-5)).astype(F32) * nwf).astype(F32))
dump("hx", x16.tobytes()); dump("hxn", xn16.tobytes())
Wh, (w, s, b) = dequant("lm_head", HEAD_ROW0, HEAD_ROWS)
dump("hw", w); dump("hs", s); dump("hb", b)
logits = matvec(Wh, f32(xn16[0]))
dump("hy", logits.tobytes())
dump("hidx", np.array([int(np.argmax(logits))], dtype=np.int32).tobytes())
print("lm_head rows", HEAD_ROWS, "argmax", int(np.argmax(logits)), "max", logits.max(), "second", np.sort(logits)[-2])
