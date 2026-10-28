#!/usr/bin/env python3
"""Float64 CPU reference of the LFM3 text decoder (LFM3-1.7B), checked against the CUDA engine's greedy tokens.

The stored weights are dequantized exactly and every op runs in float64:
  nvfp4  ModelOpt NVFP4 (U8 two E2M1 a byte, low nibble first; e4m3 scale per 16 values; f32 weight_scale_2)
  mlx    MLX affine (bits and group size from config.json "quantization", per-module overrides)
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

usage:
  ref64.py collect URL MODEL_DIR ENGINE_JSON   the greedy rows of prompts() from a server at URL
  ref64.py MODEL_DIR ENGINE_JSON [--bf16]       prints the miss table (markdown) and the agreement totals
"""
import json
import math
import os
import re
import sys
import urllib.request

import torch
from safetensors import safe_open

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


def greedy(lg):
    """argmax in (value desc, id asc) order, as the engine picks over its bf16 logits."""
    return int((lg == lg.max()).nonzero()[0])


def main(model_dir, engine_path, with_bf16):
    mdl = Model(model_dir)
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
        lf = mdl.forward(ids, P, same)
        lb_pre = mdl.forward(ids, P, bf16) if with_bf16 else None
        lb = bf16(lb_pre) if with_bf16 else None
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
    if len(sys.argv) == 5 and sys.argv[1] == "collect":
        collect(*sys.argv[2:])
    elif len(sys.argv) in (3, 4) and sys.argv[3:] in ([], ["--bf16"]):
        main(sys.argv[1], sys.argv[2], len(sys.argv) == 4)
    else:
        sys.exit(__doc__)
