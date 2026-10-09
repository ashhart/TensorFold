#!/usr/bin/env python3
"""Higher-precision variant: every non-expert tensor exactly as convert_mtp78.py takes it from the guruswami
4/8 overlay; the 256 ROUTED EXPERTS re-quantized from the zai-org BF16 source at the trunk's own widths (gate/up 6-bit,
down 8-bit, g64, F16 scales/biases, MLX affine rule in numpy) instead of repacking 4-bit values. Drafts only.
Usage: convert_mtp78_hq.py OVERLAY_LAYER78.safetensors BF16_CHECKPOINT_DIR OUT.safetensors
(BF16 source: zai-org/GLM-5.3-BF16; the overlay as for convert_mtp78.py)

Original doc: Layer 78 (GLM-5.3 MTP head, guruswami 4/8 overlay) -> the tf-glm53 trunk's layout.

The trunk (orcarouter 6-bit) stores routed experts per expert, gate/up 6-bit and down 8-bit, F16 scales/biases, and the
MLA absorb pack as layers.N.embed_q / unembed_out. The overlay stores experts stacked (switch_mlp) at 4 bits with BF16
scales. Re-packing a 4-bit value into a 6- or 8-bit field with the same scale and bias is lossless (w = s*q + b for any
width), so only the BF16 -> F16 scale cast changes numbers (drafts only: the verifier decides every token).
Each repacked tensor is checked with mx.dequantize against the original (same F16 scales): must be bit-equal.
"""
import json, struct, sys, os
import numpy as np

SRC = os.path.expanduser(sys.argv[1])
BF = os.path.join(os.path.expanduser(sys.argv[2]), "")
OUT = os.path.expanduser(sys.argv[3])
L = "model.layers.78."

f = open(SRC, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(hlen))
base = 8 + hlen
DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32}


def raw(name):
    h = hdr[name]
    a, b = h["data_offsets"]
    f.seek(base + a)
    return np.frombuffer(f.read(b - a), dtype=DT[h["dtype"]]).reshape(h["shape"]), h["dtype"]


def bf16_to_f16(u16):
    x = (u16.astype(np.uint32) << 16).view(np.float32)
    m = np.abs(x).max()
    assert m < 65000, m
    return x.astype(np.float16)


def unpack(w, bits):  # [..., k*bits/32] u32 -> [..., k] u8 values (little-endian bit stream per row)
    b = np.unpackbits(w.view(np.uint8).reshape(*w.shape[:-1], -1), axis=-1, bitorder="little")
    b = b.reshape(*b.shape[:-1], -1, bits)
    return (b * (1 << np.arange(bits, dtype=np.uint16))).sum(-1).astype(np.uint8)


def pack(q, bits):
    b = ((q[..., None].astype(np.uint16) >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    b = b.reshape(*q.shape[:-1], -1)
    return np.packbits(b, axis=-1, bitorder="little").view(np.uint32)


out = {}  # name -> (dtype str, np array)


IDX = json.load(open(BF + "model.safetensors.index.json"))["weight_map"]
_bh = {}
def bf(name):  # one BF16 tensor from the zai-org source -> float32
    fn = BF + IDX[name]
    if fn not in _bh:
        g = open(fn, "rb"); n = struct.unpack("<Q", g.read(8))[0]; _bh[fn] = (g, json.loads(g.read(n)), 8 + n)
    g, h, b0 = _bh[fn]; e = h[name]; a, b = e["data_offsets"]; g.seek(b0 + a)
    assert e["dtype"] == "BF16", e["dtype"]
    u = np.frombuffer(g.read(b - a), dtype=np.uint16).reshape(e["shape"])
    return (u.astype(np.uint32) << 16).view(np.float32)

def affine_q(w, bits, gs=64):  # MLX affine_quantize rule; scales/biases rounded to F16 first, q from the rounded ones
    nb = (1 << bits) - 1
    g = w.reshape(w.shape[0], -1, gs)
    wmax, wmin = g.max(-1), g.min(-1)
    mask = np.abs(wmin) > np.abs(wmax)
    sc = np.maximum((wmax - wmin) / nb, 1e-7)
    sc = np.where(mask, sc, -sc)
    edge = np.where(mask, wmin, wmax)
    q0 = np.round(edge / sc)
    sc = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), sc)
    bi = np.where(q0 == 0, 0, edge)
    s16, b16 = sc.astype(np.float16), bi.astype(np.float16)
    q = np.clip(np.round((g - b16.astype(np.float32)[..., None]) / s16.astype(np.float32)[..., None]), 0, nb).astype(np.uint8)
    deq = q.astype(np.float32) * s16.astype(np.float32)[..., None] + b16.astype(np.float32)[..., None]
    err = float(np.sqrt(((deq - g) ** 2).mean() / (g ** 2).mean()))
    return pack(q.reshape(w.shape[0], -1), bits), s16, b16, err


def quant(src, dst, bits_new=None):
    w, _ = raw(src + ".weight")
    s = bf16_to_f16(raw(src + ".scales")[0])
    bz = bf16_to_f16(raw(src + ".biases")[0])
    out[dst + ".weight"] = ("U32", np.ascontiguousarray(w))
    out[dst + ".scales"] = ("F16", s)
    out[dst + ".biases"] = ("F16", bz)


def plain(src, dst=None):
    a, dt = raw(src)
    out[dst or src] = (dt, np.ascontiguousarray(a))


for p in ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"]:
    quant(L + "self_attn." + p, L + "self_attn." + p)
for p in ["embed_q", "unembed_out"]:  # the pack's names
    quant(L + "self_attn." + p, "layers.78." + p)
for p in ["gate_proj", "up_proj", "down_proj"]:
    quant(L + "mlp.shared_experts." + p, L + "mlp.shared_experts." + p)
for n in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_a_layernorm.weight",
          "self_attn.kv_a_layernorm.weight", "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
          "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight", "self_attn.indexer.k_norm.weight",
          "self_attn.indexer.k_norm.bias", "self_attn.indexer.weights_proj.weight", "eh_proj.weight", "enorm.weight",
          "hnorm.weight", "shared_head.norm.weight"]:
    plain(L + n)

# routed experts: re-quantized from BF16 at the trunk's widths
errs = {}
for p, nb in [("gate_proj", 6), ("up_proj", 6), ("down_proj", 8)]:
    w4, _ = raw(L + "mlp.switch_mlp." + p + ".weight")
    s4 = bf16_to_f16(raw(L + "mlp.switch_mlp." + p + ".scales")[0]).astype(np.float32)
    b4 = bf16_to_f16(raw(L + "mlp.switch_mlp." + p + ".biases")[0]).astype(np.float32)
    el = []
    for e in range(w4.shape[0]):
        w = bf(f"{L}mlp.experts.{e}.{p}.weight")
        wn, s, bz, err = affine_q(w, nb)
        assert wn.shape[-1] == w4.shape[-1] * nb // 4, (wn.shape, w4.shape)
        if e % 32 == 0:  # old 4-bit head's error on the same expert, for the record
            q4 = unpack(w4[e], 4).astype(np.float32).reshape(w.shape[0], -1, 64)
            d4 = q4 * s4[e][..., None] + b4[e][..., None]
            g = w.reshape(w.shape[0], -1, 64)
            el.append((e, err, float(np.sqrt(((d4 - g) ** 2).mean() / (g ** 2).mean()))))
        d = f"{L}mlp.experts.{e}.{p}"
        out[d + ".weight"] = ("U32", wn)
        out[d + ".scales"] = ("F16", s)
        out[d + ".biases"] = ("F16", bz)
    errs[p] = el
    print(p, nb, "bit: rel-RMS err new vs old(4-bit) per sampled expert:", [(e, round(a, 5), round(b, 5)) for e, a, b in el], flush=True)
json.dump(errs, open(OUT + ".errs.json", "w"))
# write safetensors
os.makedirs(os.path.dirname(OUT), exist_ok=True)
meta, off = {}, 0
names = sorted(out)
for n in names:
    dt, a = out[n]
    meta[n] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
    off += a.nbytes
hj = json.dumps(meta).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
with open(OUT + ".tmp", "wb") as g:
    g.write(struct.pack("<Q", len(hj)))
    g.write(hj)
    for n in names:
        g.write(out[n][1].tobytes())
os.replace(OUT + ".tmp", OUT)
print("wrote", OUT, len(names), "tensors", round(off / 1e9, 2), "GB")
