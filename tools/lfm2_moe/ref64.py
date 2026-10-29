#!/usr/bin/env python3
"""Float64 CPU reference of LFM2.5-24B-A2B (lfm2_moe) from its MLX affine checkpoint, checked against the CUDA
engine's greedy tokens.

The stored weights are dequantized exactly (q * scale + bias) and every op runs in float64. The engine's greedy
tokens come from a running server (`collect`):
  prompt rows  the one-token greedy pick after each prefix [0, t] of a prompt (a fresh request per prefix)
  reply rows   a 32-token greedy reply, checked teacher-forced on the engine's own reply
Row t is the logits row after token t, so prompt row P - 1 and reply token 0 are the same position.

--bf16 adds a second pass with the engine's bf16 storage points (float64 inside every op, round-to-nearest-even to
bf16 where the CUDA kernels store bf16): embed, residual, every RMSNorm (bf16(w * bf16(h * inv))), projection outputs,
q/k norm and RoPE, conv B*x / acc / C*y, SwiGLU, expert outputs, MoE combine and the logits. Prompt rows take the
prefill path (weights rounded once to bf16, attention P rounded before PV, expert down stored bf16), decode rows the
exact-weight GEMV path (fp32 attention P, fp32 expert down). The router stays unrounded. Softmax P is taken against
the row's final max (the kernels' running max over 64-key tiles is not replayed). Greedy over bf16 logits breaks ties
to the lower id, as the engine does. This pass shows which side of a miss rounding alone lands on; it is not bit-exact.

usage:
  ref64.py collect URL MODEL_DIR ENGINE_JSON   the greedy rows of prompts() from a server at URL
  ref64.py MODEL_DIR ENGINE_JSON [--bf16]       prints the miss table (markdown) and the agreement totals
"""
import json
import math
import sys
import urllib.request

import torch
from safetensors import safe_open

torch.set_default_dtype(torch.float64)
F64 = torch.float64


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
    return [("capitals", "The capital of France is Paris. The capital of Germany is"),
            ("code", "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n"),
            ("chat", "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n"
                     "<|im_start|>assistant\n"),
            ("items", items + " Question: how many stones does item 3 hold? Answer: item 3 holds")]


def collect(url, model_dir, out_path, reply_len=32):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    bos = json.load(open(f"{model_dir}/config.json"))["bos_token_id"]
    cases = []
    for name, text in prompts(30):
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


class Model:
    def __init__(self, d):
        self.d = d
        self.cfg = json.load(open(f"{d}/config.json"))
        self.map = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        self.files = {}
        q = self.cfg["quantization"]
        self.embed_bits = q.get("model.embed_tokens", q)["bits"]

    def raw(self, k):
        f = self.map[k]
        if f not in self.files:
            self.files[f] = safe_open(f"{self.d}/{f}", "pt")
        return self.files[f].get_tensor(k)

    def deq(self, prefix, bits=4, rows=None):
        """MLX affine: codes are a little-endian bitstream over uint32 words, value i at bits [i*bits, (i+1)*bits)."""
        w, s, b = (self.raw(prefix + n) for n in (".weight", ".scales", ".biases"))
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows]
        by = w.contiguous().view(torch.uint8).to(torch.int64)
        if 32 % bits == 0:
            words = by.reshape(*by.shape[:-1], -1, 4) << torch.tensor([0, 8, 16, 24])
            words = words.sum(-1, keepdim=True)
            q = ((words >> (bits * torch.arange(32 // bits))) & ((1 << bits) - 1)).flatten(-2)
        else:
            nb = by.shape[-1] * 8
            bit = ((by.unsqueeze(-1) >> torch.arange(8)) & 1).reshape(*by.shape[:-1], nb)
            q = (bit[..., : nb // bits * bits].reshape(*by.shape[:-1], -1, bits) << torch.arange(bits)).sum(-1)
        q = q.to(F64).reshape(*q.shape[:-1], s.shape[-1], -1)
        return (q * s.to(F64)[..., None] + b.to(F64)[..., None]).flatten(-2)

    def forward(self, ids, P, bf):
        """Logits [T, V] (float64, not rounded). bf = same: plain float64; bf = bf16: the engine's storage points with
        rows [0, P) on the prefill path and the rest on the decode path."""
        cfg, T = self.cfg, len(ids)
        H, HK, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], 64
        silu = torch.nn.functional.silu

        def lin(x, W, p=P):  # prefill rows multiply bf16(W), decode rows the exact W; fp64 accumulation
            return torch.cat([x[:p] @ bf(W).T, x[p:] @ W.T])

        def rms(h, w):  # the engine's form: bf16(w * bf16(h * inv)), inv from the bf16 row
            return bf(w.to(F64) * bf(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg["norm_eps"])))

        inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (torch.arange(0, hd, 2, dtype=F64) / hd)
        ang = torch.arange(T, dtype=F64)[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)

        def rot(x):
            return x * cos + torch.cat([-x[..., hd // 2:], x[..., : hd // 2]], -1) * sin

        future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = bf(self.deq("model.embed_tokens", self.embed_bits, rows=torch.tensor(ids)))
        for L, kind in enumerate(cfg["layer_types"]):
            p = f"model.layers.{L}."
            x = rms(h, self.raw(p + "operator_norm.weight"))
            if kind == "conv":
                B, C, xx = bf(lin(x, self.deq(p + "conv.in_proj"))).chunk(3, -1)
                bx = bf(B * xx)
                w = self.raw(p + "conv.conv.weight").to(F64).reshape(-1, 3)  # tap 2 is the current row
                pad = torch.cat([torch.zeros(2, bx.shape[1]), bx])
                y = bf(C * bf(pad[0:T] * w[:, 0] + pad[1:T + 1] * w[:, 1] + pad[2:T + 2] * w[:, 2]))
                o = bf(lin(y, self.deq(p + "conv.out_proj")))
            else:
                q, k, v = (bf(lin(x, self.deq(p + f"self_attn.{n}_proj"))) for n in "qkv")
                q = bf(rot(rms(q.reshape(T, H, hd), self.raw(p + "self_attn.q_layernorm.weight")).transpose(0, 1)))
                k = bf(rot(rms(k.reshape(T, HK, hd), self.raw(p + "self_attn.k_layernorm.weight")).transpose(0, 1)))
                k = k.repeat_interleave(H // HK, 0)
                v = v.reshape(T, HK, hd).transpose(0, 1).repeat_interleave(H // HK, 0)
                s = ((q @ k.transpose(1, 2)) / math.sqrt(hd)).masked_fill(future, -math.inf)
                e = torch.exp(s - s.max(-1, keepdim=True).values)
                pv = torch.cat([bf(e[:, :P]), e[:, P:]], 1)  # prefill rounds P before the PV product, decode keeps it
                a = bf((pv @ v) / e.sum(-1, keepdim=True))
                o = bf(lin(a.transpose(0, 1).reshape(T, H * hd), self.deq(p + "self_attn.out_proj")))
            h = bf(h + o)
            x = rms(h, self.raw(p + "ffn_norm.weight"))
            f = p + "feed_forward."
            if L < cfg["num_dense_layers"]:
                g, u = bf(lin(x, self.deq(f + "gate_proj"))), bf(lin(x, self.deq(f + "up_proj")))
                o = bf(lin(bf(bf(silu(g)) * u), self.deq(f + "down_proj")))
            else:
                sc = torch.sigmoid(x @ self.raw(f + "gate.weight").to(F64).T)
                pick = torch.topk(sc + self.raw(f + "expert_bias").to(F64), cfg["num_experts_per_tok"], -1).indices
                wts = sc.gather(-1, pick)
                wts = wts / (wts.sum(-1, keepdim=True) + 1e-6) * cfg["routed_scaling_factor"]
                acc = torch.zeros_like(x)
                m = f + "switch_mlp."
                for ex in pick.unique().tolist():
                    sel = pick == ex
                    rr = sel.any(-1).nonzero().squeeze(-1)  # ascending: its first pe rows are prefill rows
                    pe = int((rr < P).sum())
                    g = bf(lin(x[rr], self.deq(m + "gate_proj", rows=ex), pe))
                    u = bf(lin(x[rr], self.deq(m + "up_proj", rows=ex), pe))
                    y = lin(bf(bf(silu(g)) * u), self.deq(m + "down_proj", rows=ex), pe)
                    acc[rr] += (wts * sel).sum(-1)[rr, None] * torch.cat([bf(y[:pe]), y[pe:]])
                o = bf(acc)
            h = bf(h + o)
            print(f"  layer {L} {kind}", file=sys.stderr, flush=True)
        x = rms(h, self.raw("model.embedding_norm.weight"))
        V = cfg["vocab_size"]
        return torch.cat([x @ self.deq("model.embed_tokens", self.embed_bits, rows=torch.arange(a, min(a + 8000, V))).T
                          for a in range(0, V, 8000)], -1)


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
