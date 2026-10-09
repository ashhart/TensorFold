#!/usr/bin/env python3
"""A synthetic GLM-5.3 (glm_moe_dsa) checkpoint for tf-glm53's loader: the real tensor names, dtypes and shapes of
layers [0, LAYERS) (default 3: the dense layers, all three carry a full DSA indexer), the embedding, final norm, lm_head
and the MLA absorb pack. Random weights scaled so activations stay O(1) (each matrix std ~ 1/sqrt(K)); no real model.
Usage: synth_ckpt.py OUT_DIR [layers]   -> OUT_DIR/model.safetensors(+index.json), OUT_DIR/mla_pack.safetensors"""
import json, os, sys
import numpy as np

D, QL, KVL, ROPE, NOPE, VD = 6144, 2048, 512, 64, 192, 256
QH, H, IH, ID, DI, V = NOPE + ROPE, 64, 32, 128, 12288, 154880
CROW = KVL + ROPE

out = sys.argv[1]
layers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
assert layers <= 3, "only the dense layers (0..2) are generated"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261008)


def bf16(a32):
    """float32 -> bf16 bits (round to nearest even), as uint16."""
    u = a32.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


class Writer:
    def __init__(self, path):
        self.path, self.items = path, []  # (name, dtype, shape, nbytes, gen)

    def add(self, name, dtype, shape, gen):
        n = int(np.prod(shape)) * {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}[dtype]
        self.items.append((name, dtype, list(shape), n, gen))

    def write(self):
        hdr, off = {}, 0
        for name, dt, shape, n, _ in self.items:
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
            off += n
        h = json.dumps(hdr).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        with open(self.path, "wb") as f:
            f.write(len(h).to_bytes(8, "little"))
            f.write(h)
            for name, dt, shape, n, gen in self.items:
                done = 0
                for chunk in gen():
                    b = chunk.tobytes()
                    f.write(b)
                    done += len(b)
                assert done == n, (name, done, n)
        return [x[0] for x in self.items]


def normal_bf16(shape, std, rows_per=4096):
    def g():
        rows = shape[0]
        rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        for r0 in range(0, rows, rows_per):
            r1 = min(rows, r0 + rows_per)
            yield bf16(rng.standard_normal((r1 - r0) * rest, dtype=np.float32) * std)
    return g


def const_bf16(shape, v):
    return lambda: iter([bf16(np.full(int(np.prod(shape)), v, np.float32))])


def quant(w, base, n, k, bits, gain=1.0):
    """MLX affine g64: U32 words, F16 scales and biases; dequantized std ~ gain / sqrt(k)."""
    s = np.float16(np.sqrt(12.0) * gain / ((1 << bits) * np.sqrt(k)))
    b = np.float16(-float(s) * ((1 << bits) - 1) / 2)
    words = k * bits // 32

    def gw():
        for r0 in range(0, n, 2048):
            r1 = min(n, r0 + 2048)
            yield rng.integers(0, 1 << 32, size=(r1 - r0) * words, dtype=np.uint64).astype(np.uint32)
    w.add(base + ".weight", "U32", (n, words), gw)
    w.add(base + ".scales", "F16", (n, k // 64), lambda: iter([np.full(n * (k // 64), s, np.float16)]))
    w.add(base + ".biases", "F16", (n, k // 64), lambda: iter([np.full(n * (k // 64), b, np.float16)]))


def quant_heads(w, base, n, k):
    """The pack's per-head [64][n][k] 8-bit g64 matrices."""
    s = np.float16(np.sqrt(12.0) / (256 * np.sqrt(k)))
    b = np.float16(-float(s) * 255 / 2)
    words = k // 4
    w.add(base + ".weight", "U32", (H, n, words),
          lambda: iter([rng.integers(0, 1 << 32, size=H * n * words, dtype=np.uint64).astype(np.uint32)]))
    w.add(base + ".scales", "F16", (H, n, k // 64), lambda: iter([np.full(H * n * (k // 64), s, np.float16)]))
    w.add(base + ".biases", "F16", (H, n, k // 64), lambda: iter([np.full(H * n * (k // 64), b, np.float16)]))


m = Writer(os.path.join(out, "model.safetensors"))
m.add("model.embed_tokens.weight", "BF16", (V, D), normal_bf16((V, D), 1.0))
m.add("model.norm.weight", "BF16", (D,), const_bf16((D,), 1.0))
m.add("lm_head.weight", "BF16", (V, D), normal_bf16((V, D), 1.0 / np.sqrt(D)))
for i in range(layers):
    p = f"model.layers.{i}."
    a = p + "self_attn."
    m.add(p + "input_layernorm.weight", "BF16", (D,), const_bf16((D,), 1.0))
    m.add(p + "post_attention_layernorm.weight", "BF16", (D,), const_bf16((D,), 1.0))
    quant(m, a + "q_a_proj", QL, D, 8)
    m.add(a + "q_a_layernorm.weight", "BF16", (QL,), const_bf16((QL,), 1.0))
    quant(m, a + "kv_a_proj_with_mqa", CROW, D, 8)
    m.add(a + "kv_a_layernorm.weight", "BF16", (KVL,), const_bf16((KVL,), 1.0))
    quant(m, a + "q_b_proj", H * QH, QL, 8)
    quant(m, a + "o_proj", D, H * VD, 8)
    m.add(a + "indexer.wq_b.weight", "BF16", (IH * ID, QL), normal_bf16((IH * ID, QL), 1.0 / np.sqrt(QL)))
    m.add(a + "indexer.wk.weight", "BF16", (ID, D), normal_bf16((ID, D), 1.0 / np.sqrt(D)))
    m.add(a + "indexer.k_norm.weight", "BF16", (ID,), const_bf16((ID,), 1.0))
    m.add(a + "indexer.k_norm.bias", "BF16", (ID,), const_bf16((ID,), 0.0))
    m.add(a + "indexer.weights_proj.weight", "BF16", (IH, D), normal_bf16((IH, D), 1.0 / np.sqrt(D)))
    quant(m, p + "mlp.gate_proj", DI, D, 6)
    quant(m, p + "mlp.up_proj", DI, D, 6)
    quant(m, p + "mlp.down_proj", D, DI, 6)
names = m.write()
json.dump({"metadata": {"synthetic": True, "layers": layers},
           "weight_map": {n: "model.safetensors" for n in names}},
          open(os.path.join(out, "model.safetensors.index.json"), "w"))
pk = Writer(os.path.join(out, "mla_pack.safetensors"))
for i in range(layers):
    quant_heads(pk, f"layers.{i}.embed_q", KVL, NOPE)
    quant_heads(pk, f"layers.{i}.unembed_out", VD, KVL)
pk.write()
print("wrote", out, "layers", layers, "tensors", len(names))
