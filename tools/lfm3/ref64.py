#!/usr/bin/env python3
"""Float64 CPU reference of the LFM3 text decoder (LFM3-1.7B), checked against the CUDA engine's greedy tokens.

The stored weights are dequantized exactly and every op runs in float64:
  nvfp4  ModelOpt NVFP4 (U8 two E2M1 a byte, low nibble first; e4m3 scale per 16 values; f32 weight_scale_2)
  mlx    MLX affine (bits and group size from config.json "quantization", per-module overrides)
  fp8    compressed-tensors block FP8 (e4m3; a bf16 scale per 128 x 128 block)
  bf16   plain bf16 weights
The math follows modeling_lfm3.py: sandwich norms (h += post_norm(mixer(norm(h)))), RMSNorm w * x * rsqrt(ms + eps);
conv y = C * causal_depthwise_conv(B * x) with the last tap on the current row; attention with per-head q/k RMSNorm,
RoPE (rotate_half; full layers partial 0.5 in the zero-tail form: inv_freq[i] = theta^(-2i/128) for i < 32, 0 after;
sliding layers all 64 frequencies), GQA, a 256-key sliding window, a per-head output gate sigmoid(gate_proj(x)); the
head is tied to embed_tokens.

The engine's greedy tokens come from a running server (`collect`):
  prompt rows  the one-token greedy pick after each prefix [0, t] of a prompt (a fresh request per prefix)
  reply rows   a 32-token greedy reply, checked teacher-forced on the engine's own reply
Row t is the logits row after token t, so prompt row P - 1 and reply token 0 are the same position.

--bf16 adds a second pass with the NVFP4 weight-only CUDA engine's bf16 storage points (float64 inside every op,
round-to-nearest-even to bf16 where the kernels store bf16): every RMSNorm (bf16(w * bf16(h * inv))), residual,
projection outputs, conv B*x / acc / C*y, q/k norm and RoPE, the gate, attention output, SwiGLU and the logits. Weights
are exact on both paths. Attention P is rounded to bf16 before PV on prompt rows and sliding-window decode rows and kept
unrounded on full-attention decode rows, taken against the row's final max (the kernels' running max is not
replayed). Greedy over bf16 logits breaks ties to the lower id, as the engine does. This pass shows which side of a
miss rounding alone lands on; it is not bit-exact.

Without torch (or with --numpy) the forward runs on numpy: MLX affine and bf16 weights are dequantized exactly; NVFP4
and FP8 stay torch-only and are refused by name. --bf16-points rounds where the XPU kernels store bf16 instead of the
CUDA engine: the embedding rows and RoPE cos and sin as well, attention P and the logits never (numpy path).

usage:
  ref64.py collect URL MODEL_DIR ENGINE_JSON   the greedy rows of prompts() from a server at URL
  ref64.py MODEL_DIR ENGINE_JSON [--bf16 | --bf16-points] [--numpy]
                                               prints the miss table (markdown) and the agreement totals
  ref64.py dump MODEL_DIR OUT_DIR              capitals, chat, items: ids, the hidden state after every layer and
                                               the final norm (f32), float64 top-1 per row, and the bf16-points
                                               pass's per-layer gap (meta.json); numpy path
  ref64.py check MODEL_DIR TOKENS_JSON         teacher-forced float64 top-1 of {name: {prompt, reply}}; numpy path
"""
import json
import math
import os
import re
import sys
import urllib.request

import struct

import numpy as np

try:
    import torch
    from safetensors import safe_open
except ImportError:  # the numpy path: MLX affine and bf16 weights
    torch = None

if torch is not None:
    torch.set_default_dtype(torch.float64)
    F64 = torch.float64
    E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=F64)


def complete(url, ids, n):
    body = {"prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True, "draft": False,
            "return_token_ids": True}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out.get("tensorfold", out)["token_ids"]


def prompts(n_items):
    """The receipt's prompt set: texts, encoded with the checkpoint's tokenizer.json after BOS."""
    colors = "red green blue amber violet".split()
    items = " ".join(f"Item {i}: the {colors[i % 5]} box holds {7 * i % 23} stones." for i in range(n_items))
    return [("capitals", "The capital of France is Paris. The capital of Germany is Berlin. "
                         "The capital of Italy is"),
            ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n"
                     "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Rewrite iteratively:\ndef fibonacci(n):"),
            ("chat", "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n"
                     "<|im_start|>assistant\n"),
            ("items", items + " Question: how many stones does item 3 hold? Answer: item 3 holds")]


def collect(url, model_dir, out_path, reply_len=32):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    bos = json.load(open(f"{model_dir}/config.json"))["bos_token_id"]
    cases = []
    for name, text in prompts(60):
        ids = [bos] + tok.encode(text).ids
        picks = [complete(url, ids[: t + 1], 1)[0] for t in range(len(ids))]
        cases.append({"name": name, "prompt": ids, "prompt_picks": picks, "reply": complete(url, ids, reply_len)})
        print(f"{name}: {len(ids)} prompt rows, {len(cases[-1]['reply'])} reply tokens", file=sys.stderr)
    json.dump({"cases": cases}, open(out_path, "w"))


def bf16(x):
    """Nearest bf16 (8 significant bits, ties to even), kept as float64: exact, no float32 double rounding."""
    m, e = torch.frexp(x)
    return torch.ldexp(torch.round(m * 256.0), (e - 8).to(torch.int32)).to(F64)


def same(x):
    return x


def rot(t, cos, sin):
    hd = t.shape[-1]
    return t * cos + torch.cat([-t[..., hd // 2:], t[..., : hd // 2]], -1) * sin


def strip(k):
    return re.sub(r"^(model\.language_model\.|language_model\.|model\.)+", "", k)


class Model:
    def __init__(self, d):
        self.d = d
        root = json.load(open(f"{d}/config.json"))
        self.cfg = root.get("text_config", root)
        idx = f"{d}/model.safetensors.index.json"
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            with safe_open(f"{d}/model.safetensors", "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.files, self.cache = {}, {}
        self.prefix = "language_model.model." if "language_model.model.embed_tokens.weight" in self.map else "model."
        self.mlx = root.get("quantization") if root.get("quantization", {}).get("mode", "affine") == "affine" else None

    def raw(self, k):
        f = self.map[k]
        if f not in self.files:
            self.files[f] = safe_open(f"{self.d}/{f}", "pt")
        return self.files[f].get_tensor(k)

    def vec(self, name):
        return self.raw(self.prefix + name).to(F64)

    def lin(self, name, rows=None):
        """A Linear weight [out, in] in float64, dequantized exactly by its stored format."""
        p = self.prefix + name
        w = self.raw(p + ".weight")
        if rows is not None:
            w = w[rows]
        if self.mlx and p + ".scales" in self.map:
            s, b = self.raw(p + ".scales"), self.raw(p + ".biases")
            if rows is not None:
                s, b = s[rows], b[rows]
            bits = next((v["bits"] for k, v in self.mlx.items() if isinstance(v, dict) and strip(k) == strip(p)),
                        self.mlx["bits"])
            by = w.contiguous().view(torch.uint8).to(torch.int64)
            nb = by.shape[-1] * 8
            bit = ((by.unsqueeze(-1) >> torch.arange(8)) & 1).reshape(*by.shape[:-1], nb)
            q = (bit[..., : nb // bits * bits].reshape(*by.shape[:-1], -1, bits) << torch.arange(bits)).sum(-1)
            q = q.to(F64).reshape(*q.shape[:-1], s.shape[-1], -1)
            return (q * s.to(F64)[..., None] + b.to(F64)[..., None]).flatten(-2)
        if p + ".weight_scale_2" in self.map:
            sc = self.raw(p + ".weight_scale")
            if rows is not None:
                sc = sc[rows]
            n = w.shape[0]
            codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(n, -1).long()
            v = E2M1[codes].view(n, -1, 16) * sc.to(F64)[..., None]
            return (v * self.raw(p + ".weight_scale_2").to(F64)).reshape(n, -1)
        if w.dtype == torch.float8_e4m3fn:
            sc = self.raw(p + ".weight_scale").to(F64).repeat_interleave(128, 0).repeat_interleave(128, 1)
            return w.to(F64) * (sc if rows is None else sc[rows])
        return w.to(F64)

    def W(self, name):
        if name not in self.cache:
            self.cache[name] = self.lin(name)
        return self.cache[name]

    def forward(self, ids, P, bf):
        """Logits [T, V] (float64, not rounded). bf = same: plain float64; bf = bf16: the engine's storage points with
        rows [0, P) on the prompt path and the rest on the decode path."""
        cfg, T = self.cfg, len(ids)
        D, H, HK, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        taps, vec, W = cfg["conv_L_cache"], self.vec, self.W

        def rms(h, w):  # the engine's form: bf16(w * bf16(h * inv)), inv from the bf16 row
            return bf(w * bf(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg["norm_eps"])))

        tabs = {}
        for kind, rp in cfg["rope_parameters"].items():
            angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
            inv = 1.0 / rp["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=F64) / hd)  # the full head dim's
            inv = torch.cat([inv, torch.zeros(hd // 2 - angles)])
            ang = torch.arange(T, dtype=F64)[:, None] * inv[None]
            tabs[kind] = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
        i, j = torch.arange(T)[:, None], torch.arange(T)[None]
        masks = {"full_attention": j > i, "sliding_attention": (j > i) | (i - j >= cfg["sliding_window"])}
        h = self.lin("embed_tokens", rows=torch.tensor(ids))
        for L, kind in enumerate(cfg["layer_types"]):
            p = f"layers.{L}."
            x = rms(h, vec(p + "operator_norm.weight"))
            if kind == "conv":
                B, C, xx = bf(x @ W(p + "conv.in_proj").T).chunk(3, -1)
                bx = bf(B * xx)
                w = vec(p + "conv.conv.weight").reshape(D, taps)
                pad = torch.cat([torch.zeros(taps - 1, D), bx])
                acc = torch.zeros(T, D)
                for k in range(taps):  # oldest tap first
                    acc = acc + pad[k:k + T] * w[:, k]
                o = bf(bf(C * bf(acc)) @ W(p + "conv.out_proj").T)
            else:
                q, k, v = (bf(x @ W(p + f"self_attn.{n}_proj").T) for n in "qkv")
                g = bf(x @ W(p + "self_attn.gate_proj").T)  # [T, H]
                q = rms(q.reshape(T, H, hd), vec(p + "self_attn.q_norm.weight")).transpose(0, 1)  # [H, T, hd]
                k = rms(k.reshape(T, HK, hd), vec(p + "self_attn.k_norm.weight")).transpose(0, 1)
                q, k = bf(rot(q, *tabs[kind])), bf(rot(k, *tabs[kind]))
                k = k.repeat_interleave(H // HK, 0)
                v = v.reshape(T, HK, hd).transpose(0, 1).repeat_interleave(H // HK, 0)
                s = ((q @ k.transpose(1, 2)) / math.sqrt(hd)).masked_fill(masks[kind], -math.inf)
                e = torch.exp(s - s.max(-1, keepdim=True).values)
                pv = bf(e)
                if kind == "full_attention":
                    pv = torch.cat([pv[:, :P], e[:, P:]], 1)  # full-attention decode keeps P unrounded
                a = bf((pv @ v) / e.sum(-1, keepdim=True)).transpose(0, 1)  # [T, H, hd]
                a = bf(a * bf(torch.sigmoid(g))[..., None])
                o = bf(a.reshape(T, H * hd) @ W(p + "self_attn.o_proj").T)
            h = bf(h + rms(o, vec(p + "post_operator_norm.weight")))
            x = rms(h, vec(p + "ffn_norm.weight"))
            f = p + "feed_forward."
            act = bf(bf(torch.nn.functional.silu(bf(x @ W(f + "w1").T))) * bf(x @ W(f + "w3").T))
            h = bf(h + rms(bf(act @ W(f + "w2").T), vec(p + "post_ffn_norm.weight")))
        x = rms(h, vec("embedding_norm.weight"))
        V = cfg["vocab_size"]
        return torch.cat([x @ self.lin("embed_tokens", rows=torch.arange(a, min(a + 16000, V))).T
                          for a in range(0, V, 16000)], -1)


def torch_logits(self, ids, P, points=None):
    """Logits [T, V] as a numpy array: float64, or the CUDA engine's bf16 points (points == "engine")."""
    return self.forward(ids, P, bf16 if points else same).numpy()


Model.logits = torch_logits


def greedy(lg):
    """argmax in (value desc, id asc) order, as the engine picks over its bf16 logits."""
    return int(np.flatnonzero(lg == lg.max())[0])


def npbf16(x):
    """Nearest bf16 (ties to even) of float64 values, kept as float64."""
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256.0), e - 8)


class NpModel:
    """The float64 forward on numpy: MLX affine and bf16 weights read from the safetensors bytes, exactly."""

    def __init__(self, d):
        root = json.load(open(f"{d}/config.json"))
        self.cfg = root.get("text_config", root)
        idx = f"{d}/model.safetensors.index.json"
        shards = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        self.hdr = {}
        for s in shards:
            with open(f"{d}/{s}", "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            mm = np.memmap(f"{d}/{s}", dtype=np.uint8, mode="r")
            for k, v in h.items():
                self.hdr[k] = (mm, 8 + n, v)
        self.prefix = "language_model.model." if "language_model.model.embed_tokens.weight" in self.hdr else "model."
        self.mlx = root.get("quantization") if root.get("quantization", {}).get("mode", "affine") == "affine" else None

    def raw(self, k):
        mm, base, h = self.hdr[k]
        a, b = h["data_offsets"]
        buf = mm[base + a: base + b]
        if h["dtype"] == "BF16":
            return (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(h["shape"]).astype(np.float64)
        if h["dtype"] == "F32":
            return buf.view(np.float32).reshape(h["shape"]).astype(np.float64)
        if h["dtype"] == "U32":
            return buf.view(np.uint32).reshape(h["shape"])
        raise SystemExit(f"ref64.py: {k} is {h['dtype']}; NVFP4 and FP8 weights need torch and safetensors "
                         "(the numpy path reads MLX affine and bf16)")

    def vec(self, name):
        return self.raw(self.prefix + name)

    def lin(self, name, rows=None):
        """A Linear weight [out, in] in float64: MLX q * scale + bias per group, element j of a word in bits j*b."""
        p = self.prefix + name
        if p + ".weight_scale_2" in self.hdr or p + ".weight_scale" in self.hdr:
            raise SystemExit(f"ref64.py: {p} is NVFP4 or FP8; those need torch and safetensors")
        w = self.raw(p + ".weight")
        if rows is not None:
            w = w[rows]
        if not (self.mlx and p + ".scales" in self.hdr):
            return w
        s, b = self.raw(p + ".scales"), self.raw(p + ".biases")
        if rows is not None:
            s, b = s[rows], b[rows]
        bits = next((v["bits"] for k, v in self.mlx.items() if isinstance(v, dict) and strip(k) == strip(p)),
                    self.mlx["bits"])
        n = w.shape[0]
        if bits == 4:
            q = (w[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
        else:
            bit = np.unpackbits(np.ascontiguousarray(w).view(np.uint8), axis=-1, bitorder="little")
            nb = bit.shape[-1] // bits * bits
            q = bit[..., :nb].reshape(n, -1, bits).astype(np.int64) @ (1 << np.arange(bits))
        q = q.reshape(n, s.shape[-1], -1).astype(np.float64)
        return (q * s[..., None] + b[..., None]).reshape(n, -1)

    def logits(self, ids, P, points=None, dump=None):
        """Logits [T, V] in float64; points None, "engine" (--bf16) or "xpu" (--bf16-points); dump(L, h) per layer."""
        cfg, T = self.cfg, len(ids)
        D, H, HK, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        bf = npbf16 if points else same
        xpu = points == "xpu"

        def rms(h, w):
            return bf(w * bf(h / np.sqrt((h * h).mean(-1, keepdims=True) + cfg["norm_eps"])))

        def mm(x, name):
            return x @ self.lin(name).T

        tabs = {}
        for kind, rp in cfg["rope_parameters"].items():
            angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
            inv = 1.0 / rp["rope_theta"] ** (np.arange(0, 2 * angles, 2, dtype=np.float64) / hd)
            inv = np.concatenate([inv, np.zeros(hd // 2 - angles)])
            ang = np.arange(T, dtype=np.float64)[:, None] * inv[None]
            cos, sin = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)
            tabs[kind] = (bf(cos), bf(sin)) if xpu else (cos, sin)
        i, j = np.arange(T)[:, None], np.arange(T)[None]
        masks = {"full_attention": j > i, "sliding_attention": (j > i) | (i - j >= cfg["sliding_window"])}
        h = self.lin("embed_tokens", rows=np.array(ids))
        if xpu:
            h = bf(h)
        for L, kind in enumerate(cfg["layer_types"]):
            p = f"layers.{L}."
            x = rms(h, self.vec(p + "operator_norm.weight"))
            if kind == "conv":
                B, C, xx = np.split(bf(mm(x, p + "conv.in_proj")), 3, -1)
                bx = bf(B * xx)
                w = self.vec(p + "conv.conv.weight").reshape(D, -1)
                taps = w.shape[1]
                pad = np.concatenate([np.zeros((taps - 1, D)), bx])
                acc = np.zeros((T, D))
                for k in range(taps):
                    acc = acc + pad[k:k + T] * w[:, k]
                o = bf(mm(bf(C * bf(acc)), p + "conv.out_proj"))
            else:
                q, k, v = (bf(mm(x, p + f"self_attn.{n}_proj")) for n in "qkv")
                g = bf(mm(x, p + "self_attn.gate_proj"))
                q = rms(q.reshape(T, H, hd), self.vec(p + "self_attn.q_norm.weight")).transpose(1, 0, 2)
                k = rms(k.reshape(T, HK, hd), self.vec(p + "self_attn.k_norm.weight")).transpose(1, 0, 2)
                cos, sin = tabs[kind]
                q = bf(q * cos + np.concatenate([-q[..., hd // 2:], q[..., : hd // 2]], -1) * sin)
                k = bf(k * cos + np.concatenate([-k[..., hd // 2:], k[..., : hd // 2]], -1) * sin)
                k = np.repeat(k, H // HK, 0)
                v = np.repeat(v.reshape(T, HK, hd).transpose(1, 0, 2), H // HK, 0)
                s = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)
                s[:, masks[kind]] = -np.inf
                e = np.exp(s - s.max(-1, keepdims=True))
                pv = e
                if points == "engine":
                    pv = bf(e)
                    if kind == "full_attention":
                        pv[:, P:] = e[:, P:]  # full-attention decode keeps P unrounded
                a = bf((pv @ v) / e.sum(-1, keepdims=True)).transpose(1, 0, 2)
                a = bf(a * bf(1.0 / (1.0 + np.exp(-g)))[..., None])
                o = bf(mm(a.reshape(T, H * hd), p + "self_attn.o_proj"))
            h = bf(h + rms(o, self.vec(p + "post_operator_norm.weight")))
            x = rms(h, self.vec(p + "ffn_norm.weight"))
            f = p + "feed_forward."
            a1 = bf(mm(x, f + "w1"))
            act = bf(bf(a1 / (1.0 + np.exp(-a1))) * bf(mm(x, f + "w3")))
            h = bf(h + rms(bf(mm(act, f + "w2")), self.vec(p + "post_ffn_norm.weight")))
            if dump:
                dump(L, h)
        x = rms(h, self.vec("embedding_norm.weight"))
        if dump:
            dump(-1, x)
        V = cfg["vocab_size"]
        return np.concatenate([x @ self.lin("embed_tokens", rows=np.arange(a, min(a + 8000, V))).T
                               for a in range(0, V, 8000)], -1)


def rel_rows(a, b):
    """Per-row ||a - b|| / ||b||: mean and max."""
    r = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
    return float(r.mean()), float(r.max())


def dump(model_dir, out):
    """capitals, chat, items (BOS, no added specials): float64 hidden states per layer, top-1, bf16-points gaps."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    m = NpModel(model_dir)
    bos = m.cfg["bos_token_id"]
    summary = {}
    for name, text in [p for p in prompts(60) if p[0] != "code"]:
        ids = [bos] + tok.encode(text, add_special_tokens=False).ids
        d = f"{out}/{name}"
        os.makedirs(d, exist_ok=True)
        f64h, xph = {}, {}

        def keep(L, h):
            f64h[L] = h.copy()
            h.astype(np.float32).tofile(f"{d}/h{L:02d}.f32" if L >= 0 else f"{d}/final.f32")

        lf = m.logits(ids, len(ids), None, keep)
        lb = m.logits(ids, len(ids), "xpu", lambda L, h: xph.__setitem__(L, h.copy()))
        gap = {L: rel_rows(xph[L], f64h[L]) for L in f64h}
        top = lf.argmax(-1)
        np.asarray(top, dtype=np.int32).tofile(f"{d}/top1.i32")
        lf[-1].astype(np.float32).tofile(f"{d}/logits_last.f32")
        agree = int((lb.argmax(-1) == top).sum())
        json.dump({"ids": ids, "T": len(ids), "top1": top.tolist(), "bf16_gap": {str(L): v for L, v in gap.items()},
                   "bf16_top1_agree": agree}, open(f"{d}/meta.json", "w"))
        summary[name] = {"T": len(ids), "bf16_gap_max": max(v[1] for v in gap.values()), "next": int(top[-1]),
                         "bf16_top1_agree": f"{agree}/{len(ids)}"}
        print(name, json.dumps(summary[name]), file=sys.stderr, flush=True)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)


def check(model_dir, path):
    """Teacher-forced: the float64 top-1 after each prefix of prompt + reply against the reply's own next token."""
    m = NpModel(model_dir)
    res = {}
    for name, c in json.load(open(path)).items():
        P, reply = len(c["prompt"]), c["reply"]
        lf = m.logits(c["prompt"] + reply[:-1], P)[P - 1:]
        picks = lf.argmax(-1)
        miss = [{"j": j, "engine": int(reply[j]), "f64": int(picks[j]),
                 "margin": float(lf[j].max() - lf[j, reply[j]])} for j in range(len(reply)) if picks[j] != reply[j]]
        res[name] = {"agree": f"{len(reply) - len(miss)}/{len(reply)}", "misses": miss}
        print(name, json.dumps(res[name]), file=sys.stderr, flush=True)
    return res


def main(model_dir, engine_path, points, numpy_path):
    mdl = NpModel(model_dir) if numpy_path or torch is None or points == "xpu" else Model(model_dir)
    with_bf16 = points is not None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

        def show(t):
            return f"{t} {json.dumps(tok.decode([t], skip_special_tokens=False))}".replace("|", "\\|")
    except ImportError:
        show = str
    cases = json.load(open(engine_path))["cases"]
    tot = {(m, k): [0, 0] for m in ("f64", "bf16") for k in ("prompt", "reply")}
    f64_rows, bf16_rows = [], []
    for c in cases:
        P = len(c["prompt"])
        ids = c["prompt"] + c["reply"][:-1]
        print(f"{c['name']}: {len(ids)} rows", file=sys.stderr, flush=True)
        lf = mdl.logits(ids, P)
        lb_pre = mdl.logits(ids, P, points) if with_bf16 else None
        lb = (npbf16(lb_pre) if points == "engine" else lb_pre) if with_bf16 else None
        rows = [("prompt", t, tk) for t, tk in enumerate(c["prompt_picks"])]
        rows += [("reply", P - 1 + j, tk) for j, tk in enumerate(c["reply"])]
        for kind, t, eng in rows:
            top = int(lf[t].argmax())
            tot["f64", kind][0] += top == eng
            tot["f64", kind][1] += 1
            bcol = ""
            if with_bf16:
                bp = greedy(lb[t])
                tot["bf16", kind][0] += bp == eng
                tot["bf16", kind][1] += 1
            if top != eng:
                if with_bf16:
                    d, dp = lb[t, top] - lb[t, eng], lb_pre[t, top] - lb_pre[t, eng]
                    side = "engine's" if bp == eng else "f64's" if bp == top else f"other {show(bp)}"
                    bcol = f" {d:+.4f} (pre-round {dp:+.4f}) | {side} |"
                f64_rows.append(f"| {c['name']} | {kind} | {t} | {show(eng)} | {show(top)} | "
                                f"{lf[t, top] - lf[t, eng]:.4f} |{bcol}")
            elif with_bf16 and bp != eng:
                bf16_rows.append(f"| {c['name']} | {kind} | {t} | {show(eng)} | {show(bp)} | "
                                 f"{lb[t, bp] - lb[t, eng]:.4f} (pre-round {lb_pre[t, bp] - lb_pre[t, eng]:+.4f}) | "
                                 f"{lf[t, eng] - lf[t, bp]:.4f} |")
    mode = [m for m in ("f64", "bf16") if m == "f64" or with_bf16]
    for m in mode:
        print(f"{m}: prompt rows {tot[m, 'prompt'][0]}/{tot[m, 'prompt'][1]}, "
              f"reply rows {tot[m, 'reply'][0]}/{tot[m, 'reply'][1]}")
    print("\nFloat64 top-1 misses (margin = f64 logit of its top minus f64 logit of the engine's token"
          + ("; bf16 margin = the same difference in the bf16-points pass, < 0 means it prefers the engine's token"
             if with_bf16 else "") + "):\n")
    print("| case | row | t | engine | float64 | f64 margin |" + (" bf16 margin | bf16 picks |" if with_bf16 else ""))
    print("|---|---|---|---|---|---|" + ("---|---|" if with_bf16 else ""))
    print("\n".join(f64_rows) or "(none)")
    if with_bf16:
        print("\nbf16-points misses where float64 agrees with the engine (margin = bf16 logit of its pick minus the "
              "engine's token; f64 margin = how far float64 prefers the engine's token):\n")
        print("| case | row | t | engine | bf16 pick | bf16 margin | f64 margin |\n|---|---|---|---|---|---|---|")
        print("\n".join(bf16_rows) or "(none)")


if __name__ == "__main__":
    a = sys.argv[1:]
    flags = a[2:]
    if len(a) == 4 and a[0] == "collect":
        collect(*a[1:])
    elif len(a) == 3 and a[0] == "dump":
        dump(a[1], a[2])
    elif len(a) == 3 and a[0] == "check":
        json.dump(check(a[1], a[2]), sys.stdout, indent=1)
    elif len(a) >= 2 and set(flags) <= {"--bf16", "--bf16-points", "--numpy"} and len(set(flags)) == len(flags) \
            and not {"--bf16", "--bf16-points"} <= set(flags):
        pts = "engine" if "--bf16" in flags else "xpu" if "--bf16-points" in flags else None
        main(a[0], a[1], pts, "--numpy" in flags)
    else:
        sys.exit(__doc__)
