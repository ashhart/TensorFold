"""Head fidelity: logits of real final hidden states through the GPTQ checkpoint's fp16 head (reference), the MLX
checkpoint's 4-bit g64 head, and the fp16 head requantized by us (MLX affine 4/64 algorithm, bf16 scales/biases).

  python3 head_study.py GPTQ_DIR MLX_DIR hidden.f16 P0,D0,P1,D1,...   (hidden.f16: XPU_DUMP_H of the Zig XPU engine;
  the last argument: the prompt and decode row counts, alternating, in dump order)

2026-10-12, B70, 9 264 rows (4 chat requests: 6 968 prompt rows, 2 296 decode rows):
  MLX head           logit rms 0.1656  top-1 97.39 %  KL mean 3.86e-3  p99 2.47e-2  max 6.98e-2
  requantized head   logit rms 0.1655  top-1 97.37 %  KL mean 3.85e-3  p99 2.39e-2  max 6.28e-2
The requantized GPTQ head sits where the MLX head already does."""
import json, struct, sys, time
import numpy as np

def tensor(path_dir, name):
    idx = json.load(open(f"{path_dir}/model.safetensors.index.json"))["weight_map"]
    f = f"{path_dir}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    e = hdr[name]
    dt = {"BF16": np.uint16, "F16": np.float16, "U32": np.uint32, "F32": np.float32}[e["dtype"]]
    a, b = e["data_offsets"]
    m = np.memmap(f, dtype=np.uint8, mode="r", offset=8 + n + a, shape=(b - a,))
    return np.frombuffer(m, dtype=dt).reshape(e["shape"]), e["dtype"]

def bf16(x):  # uint16 bf16 -> fp32
    return (x.astype(np.uint32) << 16).view(np.float32)

def to_bf16(x):  # fp32 -> fp32 rounded to bf16 (nearest even)
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)

def mlx_requant(w):  # MLX affine_quantize, 4 bits, groups of 64; w fp32 [n, k]
    g = w.reshape(w.shape[0], -1, 64)
    wmax, wmin = g.max(-1), g.min(-1)
    mask = np.abs(wmin) > np.abs(wmax)
    scales = np.maximum((wmax - wmin) / 15, 1e-7)
    scales = np.where(mask, scales, -scales)
    edge = np.where(mask, wmin, wmax)
    q0 = np.round(edge / scales)
    scales = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scales)
    biases = np.where(q0 == 0, 0, edge)
    s, b = to_bf16(scales), to_bf16(biases)
    q = np.clip(np.round((g - b[..., None]) / s[..., None]), 0, 15)
    return (q * s[..., None] + b[..., None]).reshape(w.shape).astype(np.float32)

def mlx_deq(wq, s, b):
    n = wq.shape[0]
    nib = np.stack([(wq >> (4 * i)) & 15 for i in range(8)], -1).reshape(n, -1).astype(np.float32)
    k = nib.shape[1]
    return (nib.reshape(n, k // 64, 64) * bf16(s)[..., None] + bf16(b)[..., None]).reshape(n, k)

import torch
torch.set_float32_matmul_precision("highest")
dev = "xpu"
gptq, mlx, hid = sys.argv[1], sys.argv[2], sys.argv[3]
segments = [int(x) for x in sys.argv[4].split(",")]  # alternating prompt / decode row counts
H = torch.from_numpy(np.fromfile(hid, dtype=np.float16).reshape(-1, 5120).astype(np.float32)).to(dev)
R = H.shape[0]
kind = np.concatenate([np.full(n, i % 2) for i, n in enumerate(segments)])  # 0 prompt, 1 decode
assert len(kind) == R, (len(kind), R)
ref, rdt = tensor(gptq, "lm_head.weight")
mw, _ = tensor(mlx, "language_model.lm_head.weight")
ms, _ = tensor(mlx, "language_model.lm_head.scales")
mb, _ = tensor(mlx, "language_model.lm_head.biases")
print("reference head", rdt, ref.shape, "rows", R, flush=True)
V = ref.shape[0]
names = ["mlx_g64", "requant_g64"]
B = 16384
def heads(v0, v1):
    wr = bf16(ref[v0:v1]) if rdt == "BF16" else ref[v0:v1].astype(np.float32)
    out = {"ref": wr, "mlx_g64": mlx_deq(mw[v0:v1], ms[v0:v1], mb[v0:v1]), "requant_g64": mlx_requant(wr)}
    return {k: H @ torch.from_numpy(np.ascontiguousarray(w)).to(dev).T for k, w in out.items()}
t0 = time.time()
# pass 1: log-sum-exp and argmax of each head, online over vocabulary chunks
st = {k: dict(m=torch.full((R,), -float("inf"), device=dev), s=torch.zeros(R, device=dev),
              arg=torch.zeros(R, dtype=torch.long, device=dev)) for k in ["ref"] + names}
cache_dir = None
for v0 in range(0, V, B):
    v1 = min(V, v0 + B)
    L = heads(v0, v1)
    for k, l in L.items():
        d = st[k]
        bm, ba = l.max(1)
        mn = torch.maximum(d["m"], bm)
        d["arg"] = torch.where(bm > d["m"], v0 + ba, d["arg"])
        d["s"] = d["s"] * torch.exp(d["m"] - mn) + torch.exp(l - mn[:, None]).sum(1)
        d["m"] = mn
    print(f"  pass 1 {v1}/{V} {time.time() - t0:.0f}s", flush=True)
lse = {k: st[k]["m"] + torch.log(st[k]["s"]) for k in st}
# pass 2: KL(ref || head) = Σ p_ref (l_ref − l) + lse − lse_ref, and the logit error
acc = {k: torch.zeros(R, dtype=torch.float64, device="cpu") for k in names}
err2 = {k: 0.0 for k in names}
for v0 in range(0, V, B):
    v1 = min(V, v0 + B)
    L = heads(v0, v1)
    p = torch.exp(L["ref"] - lse["ref"][:, None])
    for k in names:
        dl = L["ref"] - L[k]
        acc[k] += (p * dl).sum(1).double().cpu()
        err2[k] += float((dl.double() ** 2).sum())
    print(f"  pass 2 {v1}/{V} {time.time() - t0:.0f}s", flush=True)
argref = st["ref"]["arg"].cpu().numpy()
for n_ in names:
    kl = (acc[n_] + (lse[n_] - lse["ref"]).double().cpu()).numpy()
    agree = st[n_]["arg"].cpu().numpy() == argref
    rms = (err2[n_] / (R * V)) ** 0.5
    print(f"{n_}: logit rms err {rms:.4f}")
    for lab, sel in (("all", np.ones(R, bool)), ("prompt", kind == 0), ("decode", kind == 1)):
        k_ = kl[sel]
        print(f"  {lab:7s} rows {sel.sum():5d}  top-1 agree {100 * agree[sel].mean():6.2f} %  KL mean {k_.mean():.2e}  p99 {np.percentile(k_, 99):.2e}  max {k_.max():.2e}")
