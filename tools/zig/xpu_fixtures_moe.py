"""Fixtures for the MoE block of layer 1 of the Nemotron checkpoint: router, experts, combine (numpy reference)."""
import json, os, struct, numpy as np

CK = os.environ.get("TF_NEMOTRON_DIR", os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"))
FIX = os.environ.get("TF_FIXTURES_DIR", "tensorfold-fixtures")
os.makedirs(FIX, exist_ok=True)
P = "backbone.layers.1."
E, K, D, W1, WS = 128, 6, 2688, 1856, 3712
SCALING, KEEP = 2.5, 2
wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]

def raw(name, item=0, items=None):
    """Bytes of leading-dim slices [item, item + items) of a tensor, plus its shape."""
    with open(CK + "/" + wm[name], "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[name]
        a, b = h["data_offsets"]
        shape = h["shape"]
        rb = (b - a) // shape[0]
        items = shape[0] - item if items is None else items
        fh.seek(8 + n + a + item * rb)
        return fh.read(rb * items), shape

def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def dequant(w, s, b, rows, in_dim):
    """fp64 matrix [rows, in_dim] from MLX 4-bit affine group-64 bytes."""
    W = np.frombuffer(w, dtype=np.uint32).reshape(rows, in_dim // 8)
    q = np.stack([(W >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(rows, in_dim).astype(np.float64)
    sc = np.repeat(f32(np.frombuffer(s, dtype=np.uint16)).reshape(rows, in_dim // 64), 64, axis=1).astype(np.float64)
    bi = np.repeat(f32(np.frombuffer(b, dtype=np.uint16)).reshape(rows, in_dim // 64), 64, axis=1).astype(np.float64)
    return q * sc + bi

def route(logits16, bias):
    g = f32(logits16)
    prob = (np.float32(1) / (np.float32(1) + np.exp(-g))).astype(np.float32)
    sel = (prob + bias).astype(np.float32)
    order = np.argsort(-sel, kind="stable")[:K]
    total = np.float32(0)
    for p in prob[order]:
        total = np.float32(total + p)
    wts = (prob[order] / np.float32(total + np.float32(1e-20)) * np.float32(SCALING)).astype(np.float32)
    return order.astype(np.uint32), wts, sel[order[-1]] - np.sort(sel)[::-1][K]

def fma32(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)

def combine(y, wts, sh):
    acc = np.zeros(y.shape[1], dtype=np.float32)
    for k in range(y.shape[0]):
        acc = fma32(y[k], np.float32(wts[k]), acc)
    return to_bf16(acc + sh)

def out(name, arr):
    open(f"{FIX}/moe_{name}.bin", "wb").write(np.ascontiguousarray(arr).tobytes())

# router on a realistic input: random normal scaled by the layer's norm weight
gate_raw, _ = raw(P + "mixer.gate.weight")
bias = np.frombuffer(raw(P + "mixer.gate.e_score_correction_bias")[0], dtype=np.float32).copy()
nw = f32(np.frombuffer(raw(P + "norm.weight")[0], dtype=np.uint16))
rng = np.random.default_rng(5)
x16 = to_bf16(rng.standard_normal(D).astype(np.float32) * nw)
x = f32(x16)
gate = f32(np.frombuffer(gate_raw, dtype=np.uint16)).reshape(E, D)
logits16 = to_bf16((gate.astype(np.float64) @ x.astype(np.float64)).astype(np.float32))
ids, wts, margin = route(logits16, bias)
print("ids", ids, "wts", wts, "boundary margin", margin)
out("gate_w", np.frombuffer(gate_raw, dtype=np.uint16)); out("bias", bias); out("x", x16)
out("logits", logits16); out("ids", ids); out("wts", wts)

# tie-break case: logits drawn from three values, zero bias
tl16 = to_bf16(rng.integers(-1, 2, E).astype(np.float32))
tids, twts, _ = route(tl16, np.zeros(E, dtype=np.float32))
print("tie ids", tids)
out("tie_logits", tl16); out("tie_ids", tids); out("tie_wts", twts)

# expert weights: the two best experts, stored swapped (compact slot 0 = rank-1 expert) to exercise id indexing
keep = [int(ids[1]), int(ids[0])]
mats = {}
for tag, name, shape_in in (("fc1", "switch_mlp.fc1", D), ("fc2", "switch_mlp.fc2", W1)):
    parts = {"weight": [], "scales": [], "biases": []}
    for e in keep:
        for k in parts:
            parts[k].append(raw(P + f"mixer.{name}.{k}", e, 1)[0])
    for k, tail in (("weight", "w"), ("scales", "s"), ("biases", "b")):
        out(f"{tag}_{tail}", np.frombuffer(b"".join(parts[k]), dtype=np.uint8))
    n_rows = W1 if tag == "fc1" else D
    mats[tag] = [dequant(parts["weight"][i], parts["scales"][i], parts["biases"][i], n_rows, shape_in) for i in range(KEEP)]
# slot s (rank s) lives at compact position keep.index(ids[s])
out("slot_idx", np.array([keep.index(int(ids[s])) for s in range(KEEP)], dtype=np.uint32))

def relu2(acc):
    u = np.maximum(f32(to_bf16(acc.astype(np.float32))), 0)
    return to_bf16(u * u)

act = np.zeros((KEEP, W1), dtype=np.uint16)
y = np.zeros((KEEP, D), dtype=np.float32)
for s in range(KEEP):
    c = keep.index(int(ids[s]))
    act[s] = relu2(mats["fc1"][c] @ x.astype(np.float64))
    y[s] = (mats["fc2"][c] @ f32(act[s]).astype(np.float64)).astype(np.float32)
out("act", act); out("y", y)

# shared expert: up 2688 -> 3712, relu2, down 3712 -> 2688 (fp32 out)
sh = {}
for tag, name, rows, inn in (("up", "up_proj", WS, D), ("dn", "down_proj", D, WS)):
    parts = [raw(P + f"mixer.shared_experts.{name}.{k}")[0] for k in ("weight", "scales", "biases")]
    for p, tail in zip(parts, "wsb"):
        out(f"sh{tag}_{tail}", np.frombuffer(p, dtype=np.uint8))
    sh[tag] = dequant(parts[0], parts[1], parts[2], rows, inn)
sh_act = relu2(sh["up"] @ x.astype(np.float64))
sh_y = (sh["dn"] @ f32(sh_act).astype(np.float64)).astype(np.float32)
out("sh_act", sh_act); out("sh_y", sh_y)
out("delta", combine(y, wts[:KEEP], sh_y))

# combine on its own: 6 random fp32 expert rows with the real weights plus a random shared row
cy = rng.standard_normal((K, D)).astype(np.float32)
cs = rng.standard_normal(D).astype(np.float32)
out("comb_y", cy); out("comb_s", cs); out("comb_delta", combine(cy, wts, cs))
print("done; act[0][:4]", f32(act[0][:4]), "y[0][:4]", y[0][:4])
