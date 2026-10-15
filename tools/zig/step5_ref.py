#!/usr/bin/env python3
"""Step-5 text model in MLX at StepFun's rounding points: the oracle the native step5 family is checked against."""
# Usage: step5_ref.py --model DIR --prompt "..." [--tokens 32]; full layers run the sparse indexer past 512 regions.

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def bf16(x: np.ndarray) -> np.ndarray:
    """fp32 rounded to bf16 (nearest even), as fp32."""
    b = np.asarray(x, np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def e4m3(x: np.ndarray) -> np.ndarray:
    """fp32 rounded to e4m3fn (nearest even, subnormals from 2^-6, saturating at 448), as fp32."""
    a = np.abs(x)
    step = np.exp2(np.maximum(np.floor(np.log2(np.where(a > 0, a, 1))), -6) - 3)
    return np.where(a > 0, np.sign(x) * np.minimum(np.rint(a / step) * step, 448.0), x).astype(np.float32)


class Step5:
    def __init__(self, path: Path):
        cfg = json.loads((path / "config.json").read_text())
        self.c = cfg.get("text_config", cfg)
        self.q = cfg.get("quantization", {})
        idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.w: dict[str, mx.array] = {}
        for f in sorted(set(idx.values())):
            self.w.update(mx.load(str(path / f)))
        c = self.c
        self.L = c["num_hidden_layers"]
        self.H, self.KV, self.D = c["num_attention_heads"], c["num_attention_groups"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.moe = set(c["moe_layer_list"])
        self.window = c["sliding_window"]
        self.rope = []
        for i in range(self.L):
            span = int(round(self.D * c["partial_rotary_factors"][i]))
            span -= span % 2
            self.rope.append((span, float(c["rope_theta"][i])))
        self.tables: dict[tuple[int, float], tuple[mx.array, mx.array]] = {}
        self.sp = c["sparse_config"]
        self.picks: dict[int, list] = {}  # each full layer's last sparse row: every KV group's chosen regions
        self.scores: dict[int, np.ndarray] = {}  # and its scores [groups, regions]

    # -- pieces ------------------------------------------------------------------------------------------------
    def linear(self, x: mx.array, name: str) -> mx.array:
        w = self.w[name + ".weight"]
        if name + ".scales" in self.w:
            spec = self.q.get(name, self.q)
            return mx.quantized_matmul(x, w, self.w[name + ".scales"], self.w[name + ".biases"], transpose=True,
                                       group_size=spec["group_size"], bits=spec["bits"])
        return x @ w.astype(x.dtype).T

    def rows(self, x: mx.array, name: str) -> mx.array:
        """linear() a row at a time: exact like the engine's; MLX's many-row path rounds weights to bf16 first."""
        return mx.concatenate([self.linear(x[t : t + 1], name) for t in range(x.shape[0])])

    def embed(self, ids: mx.array) -> mx.array:
        name = "model.embed_tokens"
        w = self.w[name + ".weight"]
        if name + ".scales" in self.w:
            spec = self.q.get(name, self.q)
            return mx.dequantize(w[ids], self.w[name + ".scales"][ids], self.w[name + ".biases"][ids],
                                 group_size=spec["group_size"], bits=spec["bits"])
        return w[ids]

    def norm(self, x: mx.array, name: str) -> mx.array:
        """Zero-centred RMSNorm in fp32, rounded to bf16."""
        x = x.astype(mx.float32)
        y = x * mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        return (y * (self.w[name].astype(mx.float32) + 1.0)).astype(mx.bfloat16)

    def table(self, span: int, theta: float, n: int):
        key = (span, theta)
        cos, sin = self.tables.get(key, (None, None))
        if cos is None or cos.shape[0] < n:
            m = max(n, 8192)
            inv = 1.0 / (theta ** (mx.arange(0, span, 2, dtype=mx.float32) / span))
            f = mx.arange(m, dtype=mx.float32)[:, None] * inv[None, :]
            cos, sin = mx.cos(f).astype(mx.bfloat16), mx.sin(f).astype(mx.bfloat16)
            self.tables[key] = (cos, sin)
        return cos, sin

    def rotate(self, x: mx.array, pos: mx.array, i: int) -> mx.array:
        """NEOX rotation of x [T, heads, D]'s leading span, in fp32, rounded to bf16."""
        span, theta = self.rope[i]
        half = span // 2
        cos, sin = self.table(span, theta, int(pos.max().item()) + 1)
        cos, sin = cos[pos][:, None, :].astype(mx.float32), sin[pos][:, None, :].astype(mx.float32)
        re, im = x[..., :half].astype(mx.float32), x[..., half:span].astype(mx.float32)
        rot = mx.concatenate([re * cos - im * sin, re * sin + im * cos], axis=-1).astype(x.dtype)
        return mx.concatenate([rot, x[..., span:]], axis=-1)

    def swiglu(self, g: mx.array, u: mx.array, limit: float) -> mx.array:
        g, u = g.astype(mx.float32), u.astype(mx.float32)
        return (mx.minimum(g * mx.sigmoid(g), limit) * mx.clip(u, -limit, limit)).astype(mx.bfloat16)

    def mlp(self, x: mx.array, p: str, limit: float) -> mx.array:
        return self.linear(self.swiglu(self.linear(x, p + ".gate_proj"), self.linear(x, p + ".up_proj"), limit),
                           p + ".down_proj")

    def experts(self, x: mx.array, i: int) -> mx.array:
        c, p = self.c, f"model.layers.{i}.moe"
        logits = x.astype(mx.float32) @ self.w[p + ".gate.weight"].astype(mx.float32).T
        prob = mx.sigmoid(logits)
        k = c["moe_top_k"]
        ids = mx.argsort(-(prob + self.w[p + ".router_bias"].astype(mx.float32)), axis=-1)[:, :k]
        wts = mx.take_along_axis(prob, ids, axis=-1)
        if c["norm_expert_weight"]:
            wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
        wts = wts * c["moe_router_scaling_factor"]

        def proj(h, name):
            if f"{p}.{name}.scales" not in self.w:
                return mx.gather_mm(h, self.w[f"{p}.{name}.weight"].swapaxes(-1, -2), rhs_indices=ids)
            spec = self.q.get(f"{p}.{name}", self.q)
            return mx.gather_qmm(h, self.w[f"{p}.{name}.weight"], self.w[f"{p}.{name}.scales"],
                                 self.w[f"{p}.{name}.biases"], rhs_indices=ids,
                                 transpose=True, group_size=spec["group_size"], bits=spec["bits"])

        h = x[:, None, None, :]  # [T, 1, 1, hidden] broadcast over the k picks
        a = self.swiglu(proj(h, "gate_proj"), proj(h, "up_proj"), c["swiglu_limits"][i])
        y = proj(a, "down_proj")[:, :, 0, :]  # [T, k, hidden] bf16
        return ((y.astype(mx.float32) * wts[..., None]).sum(axis=1)).astype(mx.bfloat16)

    # -- layers ------------------------------------------------------------------------------------------------
    def attention(self, x: mx.array, pos: mx.array, i: int, cache: dict) -> mx.array:
        p, T = f"model.layers.{i}.self_attn", x.shape[0]
        q = self.linear(x, p + ".q_proj").reshape(T, self.H, self.D)
        k = self.linear(x, p + ".k_proj").reshape(T, self.KV, self.D)
        v = self.linear(x, p + ".v_proj").reshape(T, self.KV, self.D)
        q = self.rotate(self.norm(q, p + ".q_norm.weight"), pos, i)
        k = self.rotate(self.norm(k, p + ".k_norm.weight"), pos, i)
        if i in cache:
            k, v = mx.concatenate([cache[i][0], k]), mx.concatenate([cache[i][1], v])
        cache[i] = (k, v)
        S = k.shape[0]
        qpos = pos[:, None]
        kpos = mx.arange(S)[None, :]
        ok = kpos <= qpos
        if self.c["layer_types"][i] == "sliding_attention":
            ok = ok & (qpos - kpos < self.window)
        mask = mx.where(ok, 0.0, -mx.inf).astype(mx.bfloat16)
        o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                                 v.transpose(1, 0, 2)[None], scale=self.D ** -0.5, mask=mask)
        o = o[0].transpose(1, 0, 2)  # [T, H, D]
        if self.c["layer_types"][i] == "full_attention":
            o = self.sparse(x, pos, i, cache, q, k, v, o)
        g = mx.sigmoid(self.linear(x, p + ".g_proj"))
        o = o * g[..., None].astype(o.dtype)
        return self.linear(o.reshape(T, -1), p + ".o_proj")

    # -- the sparse indexer (StepFun's kernel.py) ----------------------------------------------------------------------
    def index_rotate(self, x: mx.array, pos: mx.array, i: int) -> mx.array:
        """NEOX rotation of [T, heads, 256]'s leading sparse_indexer_rope_dim dims, as rotate() does."""
        span, half = self.sp["sparse_indexer_rope_dim"], self.sp["sparse_indexer_rope_dim"] // 2
        cos, sin = self.table(span, self.rope[i][1], int(pos.max().item()) + 1)
        cos, sin = cos[pos][:, None, :].astype(mx.float32), sin[pos][:, None, :].astype(mx.float32)
        re, im = x[..., :half].astype(mx.float32), x[..., half:span].astype(mx.float32)
        rot = mx.concatenate([re * cos - im * sin, re * sin + im * cos], axis=-1).astype(x.dtype)
        return mx.concatenate([rot, x[..., span:]], axis=-1)

    def sparse(self, x, pos, i, cache, q, k, v, dense):
        """Rows that see more regions than the indexer keeps attend to its picks and their own region; others: dense."""
        p, T = f"model.layers.{i}.self_attn", x.shape[0]
        heads, proxy, topk = self.sp["sparse_indexer_num_heads"], self.sp["proxy_dim"], self.sp["topk"]
        groups, per = self.KV, heads // self.KV
        iq = self.rows(x, p + ".sparse_indexer_q").reshape(T, heads, proxy).astype(mx.float32)
        iq = iq * mx.rsqrt((iq * iq).mean(axis=-1, keepdims=True) + self.eps)
        iq = self.index_rotate((iq * (self.w[p + ".sparse_indexer_q_norm.weight"] + 1.0)).astype(mx.bfloat16), pos, i)
        ik = self.rows(x, p + ".sparse_indexer_k").astype(mx.float32)
        mean = ik.mean(axis=-1, keepdims=True)
        var = (ik * ik).mean(axis=-1, keepdims=True) - mean * mean
        ik = (ik - mean) * mx.rsqrt(var + self.eps) * self.w[p + ".sparse_indexer_k_norm.weight"] + self.w[p + ".sparse_indexer_k_norm.bias"]
        ik = self.index_rotate(ik.astype(mx.bfloat16)[:, None, :], pos, i)[:, 0]
        iz = self.rows(x, p + ".sparse_indexer_z")
        wt = (x @ self.w[p + ".sparse_indexer_w.weight"].astype(mx.bfloat16).T).astype(mx.float32) * per ** -0.5
        st = cache.setdefault(("index", i), {"k": [], "z": [], "s": np.zeros((0, proxy), np.float32)})
        st["k"].append(np.array(ik.astype(mx.float32)))
        st["z"].append(np.array(iz.astype(mx.float32)))
        keys, zs = np.concatenate(st["k"]), np.concatenate(st["z"])
        end = int(pos[-1].item()) + 1
        done = st["s"].shape[0]
        new = []
        for r in range(done, end // 8):
            z, kk = zs[8 * r : 8 * r + 8], keys[8 * r : 8 * r + 8]
            e = np.exp(z - z.max())
            new.append(e4m3(bf16((e * kk).sum(0) / e.sum(0))))
        if new:
            st["s"] = np.concatenate([st["s"], np.stack(new)])
        summaries = st["s"]
        qs = e4m3(np.array(iq.astype(mx.float32))).reshape(T, groups, per, proxy)
        wts = np.array(wt).reshape(T, groups, per)
        K, V = cache[i]
        out = []
        for t in range(T):
            P = int(pos[t].item())
            vis = P // 8
            if vis <= topk:
                out.append(dense[t])
                continue
            score = (np.maximum(np.einsum("ghd,rd->ghr", qs[t], summaries[:vis]), 0) * wts[t][:, :, None]).sum(1)
            rows = []
            self.picks[i] = []
            self.scores[i] = score
            for g in range(groups):
                pick = np.sort(np.lexsort((np.arange(vis), -score[g]))[:topk])
                self.picks[i].append(pick)
                idx = np.concatenate([(pick[:, None] * 8 + np.arange(8)).reshape(-1), np.arange(vis * 8, P + 1)])
                kk, vv = K[mx.array(idx), g], V[mx.array(idx), g]
                qg = q[t, g * (self.H // groups) : (g + 1) * (self.H // groups)]
                o = mx.fast.scaled_dot_product_attention(qg[None, :, None, :], kk[None, None], vv[None, None], scale=self.D ** -0.5)
                rows.append(o[0, :, 0, :])
            out.append(mx.concatenate(rows))
        return mx.stack(out)

    def layer(self, r: mx.array, pos: mx.array, i: int, cache: dict) -> mx.array:
        p = f"model.layers.{i}"
        a = self.attention(self.norm(r, p + ".input_layernorm.weight"), pos, i, cache)
        r = r + a.astype(mx.float32)
        h = self.norm(r, p + ".post_attention_layernorm.weight")
        if i in self.moe:
            f = self.experts(h, i).astype(mx.float32) + \
                self.mlp(h, p + ".share_expert", self.c["swiglu_limits_shared"][i]).astype(mx.float32)
        else:
            f = self.mlp(h, p + ".mlp", self.c["swiglu_limits"][i]).astype(mx.float32)
        return r + f

    def forward(self, ids: list[int], start: int, cache: dict, hidden: list | None = None) -> mx.array:
        """Logits (fp32) of the last row for tokens `ids` at positions start.., extending `cache`."""
        pos = mx.arange(start, start + len(ids))
        r = self.embed(mx.array(ids)).astype(mx.float32)
        for i in range(self.L):
            r = self.layer(r, pos, i, cache)
            mx.eval(r, *[t for kv in cache.values() for t in kv])
            if hidden is not None:
                hidden.append(r[-1])
        h = self.norm(r[-1:].astype(mx.bfloat16), "model.norm.weight")
        return self.linear(h, "lm_head").astype(mx.float32)[0]

    def generate(self, ids: list[int], n: int) -> list[int]:
        cache: dict = {}
        out, logits = [], self.forward(ids, 0, cache)
        for _ in range(n):
            t = int(mx.argmax(logits).item())
            out.append(t)
            logits = self.forward([t], len(ids) + len(out) - 1, cache)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--out", type=Path, help="write {prompt_ids, tokens} JSON here")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    else:
        ids = tok.encode(args.prompt)
    m = Step5(args.model)
    out = m.generate(ids, args.tokens)
    print(repr(tok.decode(out)))
    if args.out:
        args.out.write_text(json.dumps({"prompt_ids": ids, "tokens": out}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
