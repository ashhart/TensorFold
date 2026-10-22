"""A qwen3_5_moe forward in float64 from an MLX affine checkpoint: the reference staged dumps are measured against.

Weights are dequantized exactly (q * scale + bias in float64); nothing is rounded between stages. The math follows
mlx_lm 0.31's qwen3_5/qwen3_next modules: gated delta net, gated attention with partial rope, top-k experts with
renormalized softmax weights and a sigmoid-gated shared expert.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

F = np.float64


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rms(x, weight, eps):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * weight


def delta_rule(q, k, v, g, beta):
    """The gated delta rule from a zero state: q, k [T, Hk, Dk]; v [T, Hv, Dv]; g, beta [T, Hv]; returns [T, Hv, Dv]."""
    t, hv = v.shape[0], v.shape[1]
    q, k = np.repeat(q, hv // q.shape[1], axis=1), np.repeat(k, hv // k.shape[1], axis=1)
    state = np.zeros((hv, v.shape[2], k.shape[2]))
    ys = np.empty(v.shape)
    for i in range(t):
        state *= g[i][:, None, None]
        delta = (v[i] - (state @ k[i][:, :, None])[..., 0]) * beta[i][:, None]
        state += delta[:, :, None] * k[i][:, None, :]
        ys[i] = (state @ q[i][:, :, None])[..., 0]
    return ys


class Checkpoint:
    """The text model's tensors by name, dequantized to float64 on request."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        config = json.loads((snapshot / "config.json").read_text())
        self.config = config.get("text_config", config)
        quant = config["quantization"]
        self.bits, self.group = quant["bits"], quant["group_size"]
        self.widths = {k: v for k, v in quant.items() if isinstance(v, dict)}
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.shards = {}
        for name in sorted(set(index.values())):
            self.shards.update(mx.load(str(snapshot / name)))
        self.prefix = "language_model.model."

    def raw(self, name):
        return self.shards[name]

    def vector(self, name):
        return np.array(self.raw(name).astype(mx.float32)).astype(F)

    def width(self, module):
        w = self.widths.get(module, {})
        return w.get("bits", self.bits), w.get("group_size", self.group)

    def dense(self, module, rows=None, index=None):
        """`module`'s weight [out, in] (or expert `index`'s), only `rows` of out when given."""
        bits, group = self.width(module)
        q, s, b = (self.raw(f"{module}.{k}") for k in ("weight", "scales", "biases"))
        if index is not None:
            q, s, b = q[index], s[index], b[index]
        if rows is not None:
            q, s, b = q[rows], s[rows], b[rows]
        q = np.array(q)
        s = np.array(s.astype(mx.float32)).astype(F)
        b = np.array(b.astype(mx.float32)).astype(F)
        per = 32 // bits
        shifts = (np.arange(per, dtype=np.uint32) * bits).astype(np.uint32)
        values = (q[..., None] >> shifts) & np.uint32((1 << bits) - 1)
        values = values.reshape(*q.shape[:-1], -1).astype(F)
        grouped = values.reshape(*values.shape[:-1], -1, group)
        return (grouped * s[..., None] + b[..., None]).reshape(values.shape)


class Model:
    def __init__(self, ckpt: Checkpoint):
        c = ckpt.config
        self.ck = ckpt
        self.layers = c["num_hidden_layers"]
        self.hidden = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = c["rms_norm_eps"]
        self.types = c["layer_types"]
        self.k_heads, self.v_heads = c["linear_num_key_heads"], c["linear_num_value_heads"]
        self.k_dim, self.v_dim = c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        rope = c["rope_parameters"]
        self.rotary = int(self.head_dim * rope["partial_rotary_factor"])
        self.theta = float(rope["rope_theta"])
        self.experts, self.top_k = c["num_experts"], c["num_experts_per_tok"]

    def p(self, layer, rest):
        return f"{self.ck.prefix}layers.{layer}.{rest}"

    def embed(self, ids):
        return self.ck.dense(self.ck.prefix + "embed_tokens", rows=mx.array(np.asarray(ids, np.int32)))

    def mixer(self, layer, xs, positions):
        """xs: the stream rows of one or more independent runs, each [T, hidden]; returns their mixer outputs."""
        norm = self.ck.vector(self.p(layer, "input_layernorm.weight"))
        normed = [rms(x, norm, self.eps) for x in xs]
        if self.types[layer] == "linear_attention":
            return self.gdn(layer, normed)
        return self.attention(layer, normed, positions)

    def gdn(self, layer, xs):
        m = lambda n: self.ck.dense(self.p(layer, f"linear_attn.{n}"))
        wqkv, wz, wa, wb, wo = m("in_proj_qkv"), m("in_proj_z"), m("in_proj_a"), m("in_proj_b"), m("out_proj")
        conv = self.ck.vector(self.p(layer, "linear_attn.conv1d.weight"))[:, :, 0]  # [channels, kernel]
        a_log = self.ck.vector(self.p(layer, "linear_attn.A_log"))
        dt_bias = self.ck.vector(self.p(layer, "linear_attn.dt_bias"))
        gain = self.ck.vector(self.p(layer, "linear_attn.norm.weight"))
        kd, vd, hk, hv = self.k_dim, self.v_dim, self.k_heads, self.v_heads
        key = kd * hk
        out = []
        for x in xs:
            t = x.shape[0]
            qkv = x @ wqkv.T
            z = (x @ wz.T).reshape(t, hv, vd)
            g = np.exp(-np.exp(a_log) * softplus(x @ wa.T + dt_bias))
            beta = sigmoid(x @ wb.T)
            padded = np.concatenate([np.zeros((conv.shape[1] - 1, qkv.shape[1])), qkv])
            mixed = sum(padded[j:j + t] * conv[:, j] for j in range(conv.shape[1]))
            mixed = silu(mixed)
            q = mixed[:, :key].reshape(t, hk, kd)
            k = mixed[:, key:2 * key].reshape(t, hk, kd)
            v = mixed[:, 2 * key:].reshape(t, hv, vd)
            q = rms(q, None, 1e-6) * kd ** -1.0
            k = rms(k, None, 1e-6) * kd ** -0.5
            y = rms(delta_rule(q, k, v, g, beta), gain, self.eps) * silu(z)
            out.append(y.reshape(t, hv * vd) @ wo.T)
        return out

    def rope(self, x, positions):
        half = self.rotary // 2
        freq = self.theta ** (-np.arange(half, dtype=F) * 2.0 / self.rotary)
        angle = np.asarray(positions, F)[:, None] * freq[None, :]
        cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:self.rotary]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., self.rotary:]], axis=-1)

    def attention(self, layer, xs, positions):
        m = lambda n: self.ck.dense(self.p(layer, f"self_attn.{n}"))
        wq, wk, wv, wo = m("q_proj"), m("k_proj"), m("v_proj"), m("o_proj")
        qn = self.ck.vector(self.p(layer, "self_attn.q_norm.weight"))
        kn = self.ck.vector(self.p(layer, "self_attn.k_norm.weight"))
        h, kvh, d = self.heads, self.kv_heads, self.head_dim
        out = []
        for x in xs:
            t = x.shape[0]
            qg = (x @ wq.T).reshape(t, h, 2 * d)
            q, gate = qg[..., :d], qg[..., d:].reshape(t, h * d)
            k = (x @ wk.T).reshape(t, kvh, d)
            v = (x @ wv.T).reshape(t, kvh, d)
            q = self.rope(rms(q, qn, self.eps), positions)
            k = self.rope(rms(k, kn, self.eps), positions)
            k, v = np.repeat(k, h // kvh, axis=1), np.repeat(v, h // kvh, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) * d ** -0.5
            scores = np.where(np.tril(np.ones((t, t), bool))[None], scores, -np.inf)
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", scores, v).reshape(t, h * d)
            out.append((o * sigmoid(gate)) @ wo.T)
        return out

    def route(self, logits):
        """Top-k experts per row (ties to the lower id), their renormalized softmax weights, and the k-th/(k+1)-th gap."""
        order = np.argsort(-logits, axis=-1, kind="stable")
        picks = order[:, :self.top_k]
        top = np.take_along_axis(logits, picks, axis=-1)
        weights = np.exp(top - top[:, :1])
        weights /= weights.sum(axis=-1, keepdims=True)
        nxt = np.take_along_axis(logits, order[:, self.top_k:self.top_k + 1], axis=-1)[:, 0]
        return picks, weights, top[:, -1] - nxt

    def moe(self, layer, xs):
        """xs: stream rows after the mixer; returns each run's MLP output and per-row routing (picks, k/k+1 gap)."""
        norm = self.ck.vector(self.p(layer, "post_attention_layernorm.weight"))
        sizes = [x.shape[0] for x in xs]
        x = rms(np.concatenate(xs), norm, self.eps)
        logits = x @ self.ck.dense(self.p(layer, "mlp.gate")).T
        picks, weights, gap = self.route(logits)
        y = np.zeros_like(x)
        base = self.p(layer, "mlp.switch_mlp")
        for e in np.unique(picks):
            rows, slot = np.nonzero(picks == e)
            g = self.ck.dense(f"{base}.gate_proj", index=int(e))
            u = self.ck.dense(f"{base}.up_proj", index=int(e))
            dn = self.ck.dense(f"{base}.down_proj", index=int(e))
            xe = x[rows]
            y[rows] += weights[rows, slot][:, None] * ((silu(xe @ g.T) * (xe @ u.T)) @ dn.T)
        s = lambda n: self.ck.dense(self.p(layer, f"mlp.shared_expert.{n}"))
        shared = (silu(x @ s("gate_proj").T) * (x @ s("up_proj").T)) @ s("down_proj").T
        y += sigmoid(x @ self.ck.dense(self.p(layer, "mlp.shared_expert_gate")).T) * shared
        cuts = np.cumsum(sizes)[:-1]
        return np.split(y, cuts), np.split(picks, cuts), np.split(gap, cuts)

    def final(self, x):
        return rms(x, self.ck.vector(self.ck.prefix + "norm.weight"), self.eps)

    def head(self, xs, chunk=16384):
        """Logits for each run's rows, the head dequantized a vocab chunk at a time."""
        sizes = [x.shape[0] for x in xs]
        x = np.concatenate(xs)
        out = np.empty((x.shape[0], self.vocab))
        for at in range(0, self.vocab, chunk):
            rows = mx.arange(at, min(at + chunk, self.vocab))
            out[:, at:at + chunk] = x @ self.ck.dense("language_model.lm_head", rows=rows).T
        return np.split(out, np.cumsum(sizes)[:-1])
