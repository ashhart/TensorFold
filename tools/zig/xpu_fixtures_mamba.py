"""Fixtures for the Mamba2 mixer decode step on layer 0 of the Nemotron checkpoint (numpy reference)."""
import json, os, struct, numpy as np

CK = os.environ.get("TF_NEMOTRON_DIR", os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"))
FIX = os.environ.get("TF_FIXTURES_DIR", "tensorfold-fixtures")
os.makedirs(FIX, exist_ok=True)
P = "backbone.layers.0.mixer."
T, H, DH, DS, NG, KC = 6, 64, 64, 128, 8, 4
XD, CD = H * DH, H * DH + 2 * NG * DS
PROJ = XD + CD + H
EPS = 1e-5
wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]

def raw(name):
    with open(CK + "/" + wm[name], "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[name]
        a, b = h["data_offsets"]
        fh.seek(8 + n + a)
        return fh.read(b - a), h["shape"]

def f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

def bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def rnd(x):
    return f32(bf16(x))

def dequant(prefix):
    w, ws = raw(prefix + ".weight")
    s, _ = raw(prefix + ".scales")
    b, _ = raw(prefix + ".biases")
    rows, words = ws
    W = np.frombuffer(w, dtype=np.uint32).reshape(rows, words)
    q = np.stack([(W >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(rows, words * 8).astype(np.float32)
    sc = np.repeat(f32(np.frombuffer(s, dtype=np.uint16)).reshape(rows, -1), 64, axis=1)
    bi = np.repeat(f32(np.frombuffer(b, dtype=np.uint16)).reshape(rows, -1), 64, axis=1)
    return (q * sc + bi), (w, s, b)

def sigmoid(x):
    return (np.float32(1) / (np.float32(1) + np.exp(-x))).astype(np.float32)

def silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)

win, (w1, s1, b1) = dequant(P + "in_proj")
wout, (w2, s2, b2) = dequant(P + "out_proj")
convw_raw, _ = raw(P + "conv1d.weight")  # [CD, 4, 1]
convb_raw, _ = raw(P + "conv1d.bias")
cw = f32(np.frombuffer(convw_raw, dtype=np.uint16)).reshape(CD, KC)
cb = f32(np.frombuffer(convb_raw, dtype=np.uint16))
a_log = f32(np.frombuffer(raw(P + "A_log")[0], dtype=np.uint16))
dsk = f32(np.frombuffer(raw(P + "D")[0], dtype=np.uint16))
dtb = f32(np.frombuffer(raw(P + "dt_bias")[0], dtype=np.uint16))
gn_raw, _ = raw(P + "norm.weight")
gn = f32(np.frombuffer(gn_raw, dtype=np.uint16))
A = (-np.exp(a_log)).astype(np.float32)

rng = np.random.default_rng(21)
x16 = bf16(rng.standard_normal((T, 2688)).astype(np.float32))
x = f32(x16)
proj16 = bf16((win.astype(np.float64) @ x.T.astype(np.float64)).T.astype(np.float32))  # [T, PROJ]
proj = f32(proj16)

conv_state = np.zeros((KC - 1, CD), np.float32)
ssm = np.zeros((H, DH, DS), np.float32)
xc16, y16, yn16, out16 = [], [], [], []
for t in range(T):
    z = proj[t, :XD]
    cur = proj[t, XD:XD + CD]
    acc = cb.copy()
    for k in range(KC - 1):
        acc = acc + cw[:, k] * conv_state[k]
    acc = acc + cw[:, KC - 1] * cur
    cv = rnd(acc)
    xc = rnd(cv * sigmoid(cv))
    conv_state = np.stack([conv_state[1], conv_state[2], cur])
    xs = xc[:XD].reshape(H, DH)
    B = np.repeat(xc[XD:XD + NG * DS].reshape(NG, DS), H // NG, axis=0)
    C = np.repeat(xc[XD + NG * DS:].reshape(NG, DS), H // NG, axis=0)
    v = (proj[t, XD + CD:] + dtb).astype(np.float32)
    dt = (np.maximum(v, 0) + np.log(np.float32(1) + np.exp(-np.abs(v)))).astype(np.float32)
    dt = np.clip(dt, np.float32(0.0), np.float32(np.inf))  # time_step_limit absent in config -> (0, inf)
    da = np.exp(A * dt).astype(np.float32)
    ssm = (ssm * da[:, None, None] + (xs * dt[:, None])[:, :, None] * B[:, None, :]).astype(np.float32)
    o = (ssm.astype(np.float64) * C[:, None, :]).sum(-1).astype(np.float32)
    yv = rnd(o + xs * dsk[:, None])
    gz = rnd(silu(z)).reshape(H, DH)
    yg = rnd(gz * yv).reshape(XD)
    g = yg.reshape(NG, -1)
    inv = (np.float32(1) / np.sqrt((g.astype(np.float64) ** 2).mean(-1, keepdims=True).astype(np.float32) + np.float32(EPS))).astype(np.float32)
    yn = rnd(gn * rnd(g * inv).reshape(XD))
    out = (wout.astype(np.float64) @ yn.astype(np.float64)).astype(np.float32)
    xc16.append(bf16(xc)); y16.append(bf16(yg)); yn16.append(bf16(yn)); out16.append(bf16(out))

def put(name, arr):
    open(f"{FIX}/mamba_{name}.bin", "wb").write(arr if isinstance(arr, bytes) else np.ascontiguousarray(arr).tobytes())

put("w_in", w1); put("s_in", s1); put("b_in", b1)
put("w_out", w2); put("s_out", s2); put("b_out", b2)
put("conv_w", convw_raw); put("conv_b", convb_raw)
put("a_log", a_log); put("d", dsk); put("dt_bias", dtb); put("norm_w", gn_raw)
put("x", x16); put("proj", proj16); put("xc", np.stack(xc16)); put("y", np.stack(y16))
put("yn", np.stack(yn16)); put("out", np.stack(out16))
put("conv_state", bf16(conv_state)); put("ssm_state", ssm)
print("proj dim", PROJ, "conv dim", CD, "out[0][:4]", f32(out16[0])[:4], "|ssm|max", np.abs(ssm).max())
