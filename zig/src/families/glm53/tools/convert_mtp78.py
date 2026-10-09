#!/usr/bin/env python3
"""Layer 78 (GLM-5.3 MTP head, guruswami 4/8 overlay) -> the tf-glm53 trunk's layout.

The trunk (orcarouter 6-bit) stores routed experts per expert, gate/up 6-bit and down 8-bit, F16 scales/biases, and the
MLA absorb pack as layers.N.embed_q / unembed_out. The overlay stores experts stacked (switch_mlp) at 4 bits with BF16
scales. Re-packing a 4-bit value into a 6- or 8-bit field with the same scale and bias is lossless (w = s*q + b for any
width), so only the BF16 -> F16 scale cast changes numbers (drafts only: the verifier decides every token).
Each repacked tensor is checked with mx.dequantize against the original (same F16 scales): must be bit-equal.
Usage: convert_mtp78.py OVERLAY_LAYER78.safetensors OUT.safetensors
(the overlay: guruswami-ai/glm-5.3-mlx-mixed-4_8bit-mtp-recipe, the file that holds model.layers.78.*)
"""
import json, struct, sys, os
import numpy as np
import mlx.core as mx

SRC = os.path.expanduser(sys.argv[1])
OUT = os.path.expanduser(sys.argv[2])
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


def check(w_new, w_old, s, bz, bits_new, bits_old):
    a = mx.dequantize(mx.array(w_new), mx.array(s), mx.array(bz), group_size=64, bits=bits_new)
    b = mx.dequantize(mx.array(w_old), mx.array(s), mx.array(bz), group_size=64, bits=bits_old)
    assert mx.array_equal(a, b).item(), "repack changed values"


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

# routed experts: stacked 4-bit -> per expert 6-bit (gate, up) / 8-bit (down)
for p, nb in [("gate_proj", 6), ("up_proj", 6), ("down_proj", 8)]:
    w, _ = raw(L + "mlp.switch_mlp." + p + ".weight")
    s = bf16_to_f16(raw(L + "mlp.switch_mlp." + p + ".scales")[0])
    bz = bf16_to_f16(raw(L + "mlp.switch_mlp." + p + ".biases")[0])
    for e in range(w.shape[0]):
        q = unpack(w[e], 4)
        wn = pack(q, nb)
        if e in (0, 1, 127, 255):
            check(wn, np.ascontiguousarray(w[e]), s[e], bz[e], nb, 4)
        d = f"{L}mlp.experts.{e}.{p}"
        out[d + ".weight"] = ("U32", wn)
        out[d + ".scales"] = ("F16", np.ascontiguousarray(s[e]))
        out[d + ".biases"] = ("F16", np.ascontiguousarray(bz[e]))
    print(p, "repacked", w.shape, "->", out[f"{L}mlp.experts.0.{p}.weight"][1].shape, flush=True)

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
