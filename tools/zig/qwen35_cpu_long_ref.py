#!/usr/bin/env python3
"""Qwen3.5-2B MLX 4-bit in float64 over its exactly dequantized weights: the teacher-forced reference logits."""
# usage: qwen35_cpu_long_ref.py --run-input --case NAME --prompt P.npy --teacher-ids T.npy --checkpoint DIR --out OUT
import argparse
import hashlib
import json
import os
import struct

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

HIDDEN, VOCAB, LAYERS, INTER, GROUP, EPS = 2048, 248320, 24, 6144, 64, 1e-6
HEADS, KV_HEADS, HEAD_DIM, ROTARY, THETA = 8, 2, 256, 64, 1e7
DELTA_HEADS, DELTA_DIM, CONV_TAPS = 16, 128, 4
FULL = {3, 7, 11, 15, 19, 23}
PREFIX = "language_model.model."
SHIFTS = np.uint32([0, 4, 8, 12, 16, 20, 24, 28])


class Checkpoint:
    """The safetensors file's header and memory-mapped tensors."""

    dtypes = {"BF16": np.uint16, "F32": np.float32, "U32": np.uint32}

    def __init__(self, directory):
        self.path = os.path.join(directory, "model.safetensors")
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def tensor(self, key):
        e = self.header[PREFIX + key]
        start, end = e["data_offsets"]
        return np.memmap(self.path, dtype=self.dtypes[e["dtype"]], mode="r", offset=self.base + start, shape=tuple(e["shape"]))

    def gain(self, key):
        return widen(np.asarray(self.tensor(key)))

    def quant(self, key):
        return self.tensor(key + ".weight"), self.tensor(key + ".scales"), self.tensor(key + ".biases")


def widen(bf16):
    """bf16 bits to float64, exactly."""
    return (np.asarray(bf16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def dequant(w, s, b):
    """Affine 4-bit rows: scale * code + bias, the low nibble first, each group of 64 with its own scale and bias."""
    codes = ((np.asarray(w)[:, :, None] >> SHIFTS) & np.uint32(0xF)).reshape(w.shape[0], -1).astype(np.float64)
    return codes * np.repeat(widen(s), GROUP, axis=1) + np.repeat(widen(b), GROUP, axis=1)


def project(x, quant, block):
    """x[L, K] against the projection's dequantized rows, `block` rows at a time."""
    w, s, b = quant
    out = np.empty((x.shape[0], w.shape[0]))
    for r in range(0, w.shape[0], block):
        out[:, r : r + block] = x @ dequant(w[r : r + block], s[r : r + block], b[r : r + block]).T
    return out


def rmsnorm(x, gain):
    return gain * (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)))


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rotate(x, positions):
    """The first 64 of a head's 256 rotated as (i, i + 32) pairs at theta^(-i/32), the rest as they were."""
    half = ROTARY // 2
    angle = positions[:, None, None] * np.exp(-np.arange(half) * (np.log(THETA) / half))[None, None, :]
    c, s = np.cos(angle), np.sin(angle)
    a, b = x[..., :half], x[..., half:ROTARY]
    return np.concatenate([a * c - b * s, a * s + b * c, x[..., ROTARY:]], axis=-1)


def attention(q, k, v, block=256):
    """Causal softmax attention at scale 1/16, in blocks of 256 queries and keys, every sum in float64."""
    rows = q.shape[0]
    q = q / 16.0
    pad = -rows % block
    k = np.pad(k, ((0, pad), (0, 0), (0, 0)))
    v = np.pad(v, ((0, pad), (0, 0), (0, 0)))
    out = np.empty_like(q)
    for q0 in range(0, rows, block):
        qt = np.zeros((block, HEADS, HEAD_DIM))
        qt[: min(block, rows - q0)] = q[q0 : q0 + block]
        allowed = lambda k0: (k0 + np.arange(block))[None, None, :] <= (q0 + np.arange(block))[None, :, None]
        peak = np.full((HEADS, block), -np.inf)
        for k0 in range(0, q0 + block, block):
            scores = np.where(allowed(k0), np.einsum("mhd,shd->hms", qt, k[k0 : k0 + block]), -np.inf)
            peak = np.maximum(peak, scores.max(axis=-1))
        total = np.zeros((HEADS, block))
        acc = np.zeros((block, HEADS, HEAD_DIM))
        for k0 in range(0, q0 + block, block):
            scores = np.where(allowed(k0), np.einsum("mhd,shd->hms", qt, k[k0 : k0 + block]), -np.inf)
            with np.errstate(invalid="ignore"):
                weights = np.exp(scores - peak[:, :, None])
            total += weights.sum(axis=-1)
            acc += np.einsum("hms,shd->mhd", weights, v[k0 : k0 + block])
        out[q0 : q0 + block] = (acc / total.T[:, :, None])[: min(block, rows - q0)]
    return out


def delta_rule(x, layer, st, block):
    """The gated delta rule: a causal four-tap conv, SiLU, L2-normed q and k, and one float64 state a head."""
    qkv = project(x, layer["qkv"], block)
    padded = np.vstack([np.zeros((CONV_TAPS - 1, qkv.shape[1])), qkv])
    conv = sum(padded[t : t + x.shape[0]] * layer["conv"][None, :, t] for t in range(CONV_TAPS))
    vals = silu(conv.reshape(-1, 3, DELTA_HEADS, DELTA_DIM))
    qk = vals[:, :2]
    normed = qk * (1.0 / np.sqrt(np.mean(qk * qk, axis=-1, keepdims=True) + EPS / DELTA_DIM))
    q, k, v = normed[:, 0] * (1.0 / DELTA_DIM), normed[:, 1] * DELTA_DIM**-0.5, vals[:, 2]
    a = project(x, layer["a"], block)
    decay = np.exp(-np.exp(layer["a_log"]) * np.logaddexp(0.0, a + layer["dt_bias"]))
    beta = sigmoid(project(x, layer["b"], block))
    state = np.zeros((DELTA_HEADS, DELTA_DIM, DELTA_DIM))
    y = np.empty((x.shape[0], DELTA_HEADS, DELTA_DIM))
    for t in range(x.shape[0]):
        state = state * decay[t][:, None, None]
        delta = (v[t] - np.einsum("hkv,hk->hv", state, k[t])) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, None, :]
        y[t] = np.einsum("hk,hkv->hv", q[t], state)
    z = project(x, layer["z"], block).reshape(-1, DELTA_HEADS, DELTA_DIM)
    mixed = silu(z) * rmsnorm(y, layer["gnorm"])
    return project(mixed.reshape(-1, HIDDEN), layer["out"], block)


def full_attention(x, layer, block):
    """One attention layer: the q/k head norms, the rotary, the causal softmax and the sigmoid gate from q_proj."""
    rows = x.shape[0]
    positions = np.arange(rows, dtype=np.float64)
    qraw = project(x, layer["q"], block).reshape(rows, HEADS, 2 * HEAD_DIM)
    q = rotate(rmsnorm(qraw[:, :, :HEAD_DIM], layer["q_norm"]), positions)
    k = rotate(rmsnorm(project(x, layer["k"], block).reshape(rows, KV_HEADS, HEAD_DIM), layer["k_norm"]), positions)
    v = project(x, layer["v"], block).reshape(rows, KV_HEADS, HEAD_DIM)
    share = HEADS // KV_HEADS
    y = attention(q, np.repeat(k, share, axis=1), np.repeat(v, share, axis=1)) * sigmoid(qraw[:, :, HEAD_DIM:])
    return project(y.reshape(rows, HIDDEN), layer["o"], block)


def layer_weights(st, i):
    p = f"layers.{i}."
    out = {"in_norm": st.gain(p + "input_layernorm.weight"), "post_norm": st.gain(p + "post_attention_layernorm.weight")}
    out.update({name: st.quant(p + "mlp." + name + "_proj") for name in ("gate", "up", "down")})
    if i in FULL:
        out.update({name: st.quant(p + "self_attn." + name + "_proj") for name in ("q", "k", "v", "o")})
        out.update({name + "_norm": st.gain(p + "self_attn." + name + "_norm.weight") for name in ("q", "k")})
    else:
        out.update({name: st.quant(p + "linear_attn.in_proj_" + name) for name in ("qkv", "z", "a", "b")})
        out["out"] = st.quant(p + "linear_attn.out_proj")
        out["gnorm"] = st.gain(p + "linear_attn.norm.weight")
        out["conv"] = widen(np.asarray(st.tensor(p + "linear_attn.conv1d.weight")).reshape(3 * DELTA_HEADS * DELTA_DIM, CONV_TAPS))
        out["a_log"] = np.asarray(st.tensor(p + "linear_attn.A_log"), dtype=np.float64)
        out["dt_bias"] = st.gain(p + "linear_attn.dt_bias")
    return out


def forward(ids, st, block):
    """The residual stream after the 24 layers for `ids`, every row of the teacher-forced history."""
    embed = st.quant("embed_tokens")
    stream = dequant(embed[0][ids], embed[1][ids], embed[2][ids])
    for i in range(LAYERS):
        layer = layer_weights(st, i)
        x = rmsnorm(stream, layer["in_norm"])
        stream = stream + (full_attention(x, layer, block) if i in FULL else delta_rule(x, layer, st, block))
        x = rmsnorm(stream, layer["post_norm"])
        stream = stream + project(silu(project(x, layer["gate"], block)) * project(x, layer["up"], block), layer["down"], block)
        print(f"  layer {i + 1}/{LAYERS} done", flush=True)
    return stream


def logits(stream, rows, st, block):
    """The final norm and the tied head over the vocabulary, `block` rows of it at a time, for the rows asked."""
    x = rmsnorm(stream[rows], st.gain("norm.weight"))
    w, s, b = st.quant("embed_tokens")
    out = np.empty((len(rows), VOCAB))
    for r in range(0, VOCAB, block):
        out[:, r : r + block] = x @ dequant(w[r : r + block], s[r : r + block], b[r : r + block]).T
    return out


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a.astype("<u4")).tobytes()).hexdigest()


def run_input(args):
    """The oracle #1544 judged the Vulkan engine against: OUT/CASE/logits_f64.npy, one row a teacher token."""
    prompt = np.load(args.prompt, allow_pickle=False)
    teacher = np.load(args.teacher_ids, allow_pickle=False)
    if prompt.dtype != np.uint32 or teacher.dtype != np.uint32 or prompt.ndim != 1 or teacher.ndim != 1 or not len(prompt) or not len(teacher):
        raise SystemExit("the prompt and the teacher ids are one-dimensional, non-empty uint32 .npy arrays")
    if os.path.basename(args.case) != args.case or args.case in ("", ".", ".."):
        raise SystemExit("the case name is one path component")
    out_dir = os.path.join(args.out, args.case)
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; a reference is computed once")
    ids = np.concatenate([prompt, teacher[:-1]]).astype(np.int64)
    st = Checkpoint(args.checkpoint)
    stream = forward(ids, st, args.row_block)
    rows = [len(prompt) - 1 + g for g in range(len(teacher))]
    out = logits(stream, rows, st, args.head_block)
    if not np.isfinite(out).all():
        raise SystemExit("the reference logits are not all finite")
    os.makedirs(out_dir)
    np.save(os.path.join(out_dir, "logits_f64.npy"), out)
    result = {
        "case": args.case,
        "checkpoint_sha256": hashlib.sha256(open(st.path, "rb").read()).hexdigest(),
        "prompt_len": int(len(prompt)),
        "rows": int(len(teacher)),
        "positions": rows,
        "prompt_sha256": sha256(prompt),
        "teacher_sha256": sha256(teacher),
        "logits_sha256": hashlib.sha256(open(os.path.join(out_dir, "logits_f64.npy"), "rb").read()).hexdigest(),
        "argmax": out.argmax(axis=1).tolist(),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(f"{args.case}: {len(teacher)} rows of float64 logits from position {rows[0]}; sha256 {result['logits_sha256']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-input", action="store_true", required=True, help="one prompt with its teacher ids")
    p.add_argument("--case", required=True, help="the case's name: its logits go to OUT/CASE/logits_f64.npy")
    p.add_argument("--prompt", required=True, help="the prompt's token ids as a uint32 .npy (cpu_fidelity.py freeze)")
    p.add_argument("--teacher-ids", required=True, help="the teacher token ids as a uint32 .npy: one logits row each")
    p.add_argument("--checkpoint", required=True, help="the mlx-community/Qwen3.5-2B-MLX-4bit directory")
    p.add_argument("--out", required=True)
    p.add_argument("--row-block", type=int, default=1024, help="dequantized projection rows held at once")
    p.add_argument("--head-block", type=int, default=2048, help="dequantized vocabulary rows held at once")
    run_input(p.parse_args())


if __name__ == "__main__":
    main()
