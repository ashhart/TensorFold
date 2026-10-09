"""EXL3 fixtures (numpy only): rows of real checkpoint tensors plus synthetic trellises for every codebook and width."""
# usage: xpu_fixtures_exl3.py OUT_DIR [synth] [real]; env TF_EXL3_DIR: the EXL3 checkpoint directory (real cases)
import json, os, struct, sys, zlib, numpy as np

CK = os.environ.get("TF_EXL3_DIR", "")
OUT = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"
MAGIC = 0x334C5845  # 'EXL3'
CBS = {"3inst": 0, "mcg": 1, "mul1": 2}

def f16v(bits): return bits.astype(np.uint16).view(np.float16).astype(np.float64)

def codebook(name):
    s = np.arange(65536, dtype=np.uint64)
    if name == "mul1":
        x = (s * 0x83DCD12D) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        return (h.astype(np.float64) * f16v(np.array(0x1EEE)) + f16v(np.array(0xC931))).astype(np.float16)
    x = ((s * 0xCBAC1FED) if name == "mcg" else (s * 89226354 + 64248484)) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return (f16v(x & 0xFFFF) + f16v(x >> 16)).astype(np.float16)

def stream_ends(k2):
    p1 = np.arange(1, 257, dtype=np.int64)
    return p1 * (k2 // 2) if k2 % 2 == 0 else (p1 * k2 - (p1 % 2)) // 2

def tile_positions():
    p = np.arange(256); l, j = p // 8, p % 8
    return 2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1), l // 4 + 8 * (j >> 2)

def states(trellis, k2):
    w = trellis.view(np.uint16).astype(np.uint64)
    words = w[..., 0::2] | (w[..., 1::2] << 16)
    nw = 4 * k2
    first = stream_ends(k2) - 16 + 32 * nw
    i0, off = (first // 32) % nw, first % 32
    pair = (words[..., i0] << 32) | words[..., (i0 + 1) % nw]
    return ((pair >> (48 - off).astype(np.uint64)) & 0xFFFF).astype(np.uint32)

def unpack(trellis, k2, cb):
    kt, nt = trellis.shape[:2]
    table = codebook(cb); rows, cols = tile_positions()
    w = np.empty((kt, 16, nt, 16), dtype=np.float16)
    for k0 in range(0, kt, 16):
        vals = table[states(trellis[k0:k0 + 16], k2).astype(np.int64)]
        w[k0:k0 + 16][:, rows, :, cols] = vals.transpose(2, 0, 1)
    return w.reshape(kt * 16, nt * 16)

def had128():
    i = np.arange(128); par = np.array([bin(v).count("1") & 1 for v in range(128)])
    return np.where(par[i[:, None] & i[None, :]] == 1, -1.0, 1.0)

def fwht_f32(x):
    """x [..., 128] float32 -> unscaled Walsh-Hadamard in the kernel's order (i bits first, then lane bits)."""
    v = x.astype(np.float32).reshape(x.shape[:-1] + (8, 16))
    for m in (1, 2, 4):
        v = v.reshape(v.shape[:-2] + (8 // (2 * m), 2, m, 16)); a, b = v[..., 0, :, :].copy(), v[..., 1, :, :].copy()
        v[..., 0, :, :], v[..., 1, :, :] = a + b, a - b; v = v.reshape(v.shape[:-4] + (8, 16))
    for m in (1, 2, 4, 8):
        v = v.reshape(v.shape[:-1] + (16 // (2 * m), 2, m)); a, b = v[..., 0, :].copy(), v[..., 1, :].copy()
        v[..., 0, :], v[..., 1, :] = a + b, a - b; v = v.reshape(v.shape[:-3] + (16,))
    return v.reshape(x.shape)

HS = np.float32(0.08838834764831845)

def rot_in(x_bf, suh):
    """The kernel's rot_in in fp32 order, rounded to fp16."""
    k = x_bf.shape[-1]
    v = (x_bf.astype(np.float32) * suh.astype(np.float32)).reshape(-1, k // 128, 128)
    return (fwht_f32(v) * HS).reshape(-1, k).astype(np.float16)

def bf16_round(f):
    u = f.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)

def bf16_to_f32(b): return (b.astype(np.uint32) << 16).view(np.float32)

def reference(trellis, k2, cb, suh, svh, bias, x_bf):
    wq = unpack(trellis, k2, cb)
    xh = rot_in(x_bf, suh)
    s = xh.astype(np.float64) @ wq.astype(np.float64)
    n = wq.shape[1]
    y = ((s.reshape(-1, n // 128, 128) @ (had128() / np.sqrt(128))).reshape(-1, n)) * svh.astype(np.float64)
    if bias is not None: y = y + bias.astype(np.float64)
    # the pure float64 layer (unrounded xh) for the report: how much the fp16 xh stage alone moves y
    k = x_bf.shape[-1]
    xp = ((x_bf.astype(np.float64) * suh.astype(np.float64)).reshape(-1, k // 128, 128) @ (had128() / np.sqrt(128))).reshape(-1, k)
    yp = ((xp @ wq.astype(np.float64)).reshape(-1, n // 128, 128) @ (had128() / np.sqrt(128))).reshape(-1, n) * svh.astype(np.float64)
    if bias is not None: yp = yp + bias.astype(np.float64)
    return wq, xh, y.astype(np.float32), float(np.abs(y - yp).max() / np.abs(yp).max())

# exl3_<name>.bin: 16 x u32 header, trellis, suh, svh, [bias], x, [wq], xh, y; the real slices are weights: never commit
def write(name, trellis, k2, cb, suh, svh, bias, rows, seed, keep_wq):
    rng = np.random.default_rng(seed)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    x_bf = bf16_round(rng.standard_normal((rows, k)).astype(np.float32))
    wq, xh, y, pure = reference(trellis, k2, CBS[cb] and cb or cb, suh, svh, bias, bf16_to_f32(x_bf))
    hdr = [MAGIC, k, n, k2, CBS[cb], rows, int(bias is not None), int(keep_wq)] + [0] * 8
    with open(OUT + f"exl3_{name}.bin", "wb") as f:
        f.write(struct.pack("<16I", *hdr))
        f.write(trellis.astype("<i2").tobytes()); f.write(suh.astype("<f2").tobytes()); f.write(svh.astype("<f2").tobytes())
        if bias is not None: f.write(bias.astype("<f2").tobytes())
        f.write(x_bf.astype("<u2").tobytes())
        if keep_wq: f.write(wq.astype("<f2").tobytes())
        f.write(xh.astype("<f2").tobytes()); f.write(y.astype("<f4").tobytes())
    print(f"{name}: K={k} N={n} K2={k2} {cb} rows={rows} bias={bias is not None} fp16-xh vs pure f64 max rel {pure:.2e}")

def scales(rng, n): return (rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n)).astype(np.float16)

def synth():
    """Every codebook x width as exl3_synth_<cb>_<k2>.bin (K=256, N=256, 16 rows)."""
    for cb in CBS:
        for k2 in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
            if k2 % 2 and cb != "mul1": continue
            rng = np.random.default_rng(1000 + 31 * k2 + CBS[cb])
            tr = rng.integers(-32768, 32768, (16, 16, 8 * k2), dtype=np.int64).astype(np.int16)
            write(f"synth_{cb}_{k2}", tr, k2, cb, scales(rng, 256), scales(rng, 256),
                  (rng.standard_normal(256) * 0.1).astype(np.float16) if k2 % 4 == 0 else None, 16, 7 + k2, True)

class Ckpt:
    def __init__(self):
        self.wm = json.load(open(CK + "/model.safetensors.index.json"))["weight_map"]
    def tensor(self, name, cols=None):
        with open(f"{CK}/{self.wm[name]}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))[name]
            a, b = h["data_offsets"]; fh.seek(8 + n + a); raw = fh.read(b - a)
        dt = {"I16": "<i2", "F16": "<f2", "I32": "<i4", "BF16": "<u2"}[h["dtype"]]
        return np.frombuffer(raw, dtype=dt).reshape(h["shape"])

def real(cases):
    ck = Ckpt()
    for tag, prefix, ncols, rows, k2, wq in cases:
        tr = ck.tensor(prefix + ".trellis")[:, :ncols // 16, :].copy()
        suh = ck.tensor(prefix + ".suh").copy(); svh = ck.tensor(prefix + ".svh")[:ncols].copy()
        has_mul1 = prefix + ".mul1" in ck.wm
        assert tr.shape[-1] == 8 * k2 and has_mul1
        write(tag, tr, k2, "mul1", suh, svh, None, rows, zlib.crc32(tag.encode()) & 0xFFFF, wq)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[2:] or ["synth", "real"]
    L = "model.language_model.layers."
    if "synth" in what: synth()
    if "real" in what:
        real([  # tag, tensor, columns kept, rows, K2, keep W_q
            ("up_3b", L + "0.mlp.up_proj", 128, 16, 6, True),             # K=5120 -> N=17408 (columns 0..127)
            ("down_3b", L + "0.mlp.down_proj", 256, 16, 6, False),        # K=17408 -> N=5120
            ("kproj_3b", L + "3.self_attn.k_proj", 128, 16, 6, True),     # K=5120 -> N=1024
            ("qkv_3b", L + "0.linear_attn.in_proj_qkv", 256, 16, 6, False),
            ("head_6b", "lm_head", 128, 16, 12, True),                    # 6-bit head, K=5120 -> N=248320
        ])
