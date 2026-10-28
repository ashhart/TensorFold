"""For `tf-cuda-test ling-forward-synth`: a Ling-3.1 checkpoint's logits in float64, from its own file and config."""
import argparse
import json
import math
import os
import struct

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


class Ckpt:
    """Every tensor of model.safetensors; linears dequantized to float64 by the side tensors beside them."""

    def __init__(self, path):
        self.keys = set()
        if os.path.isdir(path):  # a sharded release: model.safetensors.index.json maps each tensor to its file
            index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
            self.files = {f: safe_open(os.path.join(path, f), "pt") for f in sorted(set(index.values()))}
            self.where = {k: self.files[f] for k, f in index.items()}
            self.keys = set(index)
        else:
            f = safe_open(path, "pt")
            self.where = {k: f for k in f.keys()}
            self.keys = set(self.where)
        self.cache = {}

    def t(self, name):
        return self.where[name].get_tensor(name)

    def lin(self, module):
        if ".experts." in module:  # 128 experts in float64 would not fit beside the rest: converted at each use
            return self.dequant(module)
        if module not in self.cache:
            self.cache[module] = self.dequant(module)
        return self.cache[module]

    def dequant(self, module):
        w = self.t(module + ".weight")
        if module + ".weight_scale_inv" in self.keys:
            s = self.t(module + ".weight_scale_inv").double()
            n, k = w.shape
            return w.double() * s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        if module + ".weight_scale_2" in self.keys:
            n = w.shape[0]
            codes = torch.stack((w & 0xF, w >> 4), -1).reshape(n, -1).long()
            v = E2M1[codes & 7] * torch.where(codes >= 8, -1.0, 1.0).double()
            scale = self.t(module + ".weight_scale").double().repeat_interleave(16, 1)
            return v * scale * self.t(module + ".weight_scale_2").double()
        return w.double()


def bf16(x):
    return x.to(torch.bfloat16).double()


class Model:
    """The decoder in float64 (`r` = identity) or at the HF bf16 model's rounding points (`r` = bf16)."""

    def __init__(self, ck, c, r, forced=None):
        self.ck, self.c, self.r, self.forced = ck, c, r, forced
        self.eps = c["rms_norm_eps"]
        self.picks = []
        self.gap = 0.0

    def norm(self, x, w):
        y = x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)
        return self.r(self.r(y) * w.double())

    def lin(self, x, module):
        return self.r(x @ self.ck.lin(module).T)

    def kda(self, x, p):
        c, r, ck = self.c, self.r, self.ck
        T, H, dk = x.shape[0], c["num_attention_heads"], c["head_dim"]
        qkv = torch.cat([self.lin(x, p + n) for n in ("q_proj", "k_proj", "v_proj")], -1)
        cw = torch.cat([ck.t(p + n + "_conv1d.weight").double()[:, 0, :] for n in ("q", "k", "v")], 0)
        xp = torch.cat([torch.zeros(3, qkv.shape[1], dtype=torch.float64), qkv], 0)
        acc = sum(cw[:, j] * xp[j:j + T] for j in range(4))
        qkv = r(acc * torch.sigmoid(acc)).view(T, 3, H, dk)
        q = qkv[:, 0] / torch.sqrt((qkv[:, 0] ** 2).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = qkv[:, 1] / torch.sqrt((qkv[:, 1] ** 2).sum(-1, keepdim=True) + 1e-6)
        v = qkv[:, 2]
        rate = torch.exp(ck.t(p + "A_log").double())
        f = self.lin(x, p + "f_proj").view(T, H, dk) + ck.t(p + "dt_bias").double().view(H, dk)
        g = torch.exp(c["kda_lower_bound"] * torch.sigmoid(rate[None, :, None] * f))
        beta = r(torch.sigmoid(self.lin(x, p + "b_proj")))
        gate = self.lin(x, p + "g_proj").view(T, H, dk)
        s = torch.zeros(H, dk, dk, dtype=torch.float64)
        out = torch.zeros(T, H, dk, dtype=torch.float64)
        for t in range(T):
            s = s * g[t][:, None, :]
            u = beta[t][:, None] * (v[t] - torch.einsum("hji,hi->hj", s, k[t]))
            s = s + u[:, :, None] * k[t][:, None, :]
            out[t] = r(torch.einsum("hji,hi->hj", s, q[t]))
        w = ck.t(p + "o_norm.weight").double()
        y = out / torch.sqrt((out * out).mean(-1, keepdim=True) + self.eps) * w * torch.sigmoid(gate)
        return self.lin(r(y).reshape(T, H * dk), p + "o_proj")

    def mla(self, x, p):
        c, r, ck = self.c, self.r, self.ck
        T, H = x.shape[0], c["num_attention_heads"]
        nope, rope, dv, lw = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"], c["kv_lora_rank"]
        q = self.lin(self.norm(self.lin(x, p + "q_a_proj"), ck.t(p + "q_a_layernorm.weight")), p + "q_b_proj").view(T, H, -1)
        kv = self.lin(x, p + "kv_a_proj_with_mqa")
        lat = self.norm(kv[:, :lw], ck.t(p + "kv_a_layernorm.weight"))
        kvb = self.lin(lat, p + "kv_b_proj").view(T, H, nope + dv)
        inv = 1.0 / c["rope_theta"] ** (torch.arange(0, rope, 2, dtype=torch.float64) / rope)
        ang = torch.arange(T, dtype=torch.float64)[:, None] * inv
        cos, sin = torch.cos(ang), torch.sin(ang)

        def rot(z):  # GPT-J pairs (rope_interleave), out in halves as HF's interleave helper writes them
            a, b = z[..., 0::2], z[..., 1::2]
            cs, sn = (cos[:, None], sin[:, None]) if z.dim() == 3 else (cos, sin)
            return r(torch.cat((a * cs - b * sn, b * cs + a * sn), -1))

        qr, kr = rot(q[..., nope:]), rot(kv[:, lw:])
        sc = (torch.einsum("thd,shd->hts", q[..., :nope], kvb[..., :nope]) + torch.einsum("thd,sd->hts", qr, kr)) * (nope + rope) ** -0.5
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        o = r(torch.einsum("hts,shd->thd", torch.softmax(sc, -1), kvb[..., nope:]))
        o = r(o * r(torch.sigmoid(self.lin(x, p + "g_proj")))[..., None])
        return self.lin(o.reshape(T, H * dv), p + "dense")

    def mlp(self, x, p, limit):
        g, u = self.lin(x, p + "gate_proj"), self.lin(x, p + "up_proj")
        s = self.r(g * torch.sigmoid(g))
        if limit:  # inclusionAI's SwigluStepAndMul: the limit on silu's output, the up clipped both ways
            s, u = s.clamp(max=limit), u.clamp(-limit, limit)
        return self.lin(self.r(s * u), p + "down_proj")

    def moe(self, x, p, i):
        c, r, ck = self.c, self.r, self.ck
        ne, top, ng, kg = c["num_experts"], c["num_experts_per_tok"], c["n_group"], c["topk_group"]
        s = torch.sigmoid(x @ ck.t(p + "gate.weight").double().T)
        choice = s + ck.t(p + "gate.expert_bias").double()
        gs = choice.view(-1, ng, ne // ng).topk(2, -1)[0].sum(-1)
        keep = torch.zeros_like(gs).scatter_(1, gs.topk(kg, -1)[1], 1).repeat_interleave(ne // ng, 1)
        masked = choice.masked_fill(keep == 0, float("-inf"))
        idx = masked.topk(top, -1)[1]
        self.picks.append(idx.sort(-1)[0].int())
        if self.forced is not None and len(self.picks) <= self.forced.shape[0]:  # the graph's picks; gap: how far the worst one ranks below the top-k
            idx = self.forced[len(self.picks) - 1].long()
            kth = masked.topk(top, -1)[0][:, -1]
            self.gap = max(self.gap, (kth - choice.gather(1, idx).min(-1)[0]).max().item())
        w = s.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
        lists = c["expert_swiglu_limit_list"], c["share_expert_swiglu_limit_list"]
        le, ls = (x[i] if x and i < len(x) else 0 for x in lists)  # null (Ling-3.0-tiny): no clamp  # the lists cover the backbone: the MTP layer runs unclamped
        y = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(top):
                y[t] += self.mlp(x[t:t + 1], f"{p}experts.{int(idx[t, j])}.", le)[0] * w[t, j]
        return r(r(y) + self.mlp(x, p + "shared_experts.", ls))

    def logits(self, ids):
        c, ck = self.c, self.ck
        x = ck.t("model.word_embeddings.weight").double()[ids]
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}."
            h = self.norm(x, ck.t(p + "input_layernorm.weight"))
            g = c["layer_group_size"]
            mla = (i + 1) % g == 0 or i >= c["num_hidden_layers"] // g * g
            x = self.r(x + (self.mla(h, p + "attention.") if mla else self.kda(h, p + "attention.")))
            h = self.norm(x, ck.t(p + "post_attention_layernorm.weight"))
            x = self.r(x + (self.moe(h, p + "mlp.", i) if i >= c["first_k_dense_replace"] else self.mlp(h, p + "mlp.", 0)))
        self.hidden = self.norm(x, ck.t("model.norm.weight"))
        return self.hidden @ ck.lin("lm_head").T

    def mtp_logits(self, ids):
        """The MTP layer's logits at rows 0 .. T - 2: hidden row t (after logits()) with token t + 1's embedding."""
        c, ck = self.c, self.ck
        L = c["num_hidden_layers"]
        p = f"model.layers.{L}."
        e = self.norm(ck.t("model.word_embeddings.weight").double()[ids[1:]], ck.t(p + "enorm.weight"))
        h = self.norm(self.hidden[:-1], ck.t(p + "hnorm.weight"))
        x = self.lin(torch.cat([e, h], -1), p + "eh_proj")
        x = self.r(x + self.mla(self.norm(x, ck.t(p + "input_layernorm.weight")), p + "attention."))
        x = self.r(x + self.moe(self.norm(x, ck.t(p + "post_attention_layernorm.weight")), p + "mlp.", L))
        return self.norm(x, ck.t(p + "final_layernorm.weight")) @ ck.lin("lm_head").T


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("ckpt", help="a ling-synth directory (config.json, model.safetensors) or a sharded checkpoint's")
    a.add_argument("out", help="the reference file: ids, float64 and bf16-point logits, the picks it routed on")
    a.add_argument("--tokens", type=int, default=88)
    a.add_argument("--picks", help="route on these picks (i32 [MoE layers][tokens][top_k], the graph's) instead")
    a.add_argument("--mtp", help="also the MTP layer's logits (float64, bf16 points) and its own picks, to this file")
    a.add_argument("--ids", help="these tokens (i32, ling-lanes' OUT.ids) instead of the fixed walk; --tokens follows")
    o = a.parse_args()
    c = json.load(open(os.path.join(o.ckpt, "config.json")))
    one = os.path.join(o.ckpt, "model.safetensors")
    ck = Ckpt(one if os.path.exists(one) else o.ckpt)
    V = c["vocab_size"]
    if o.ids:
        ids = torch.frombuffer(bytearray(open(o.ids, "rb").read()), dtype=torch.int32).long()
    else:
        ids = torch.tensor([(37 * t + 11) % (V - 2) for t in range(o.tokens)])
    T = ids.shape[0]
    forced = None
    if o.picks:
        nm = sum(1 for i in range(c["num_hidden_layers"]) if i >= c["first_k_dense_replace"])
        forced = torch.frombuffer(bytearray(open(o.picks, "rb").read()), dtype=torch.int32).view(nm, T, -1)
    with torch.no_grad():
        exact = Model(ck, c, lambda z: z, forced)
        l64 = exact.logits(ids)
        lbf = Model(ck, c, bf16, forced).logits(ids)
        if o.mtp:  # the bf16-point model's own picks route its MTP layer; float64's are written for the flips
            rounded = Model(ck, c, bf16, forced)
            rounded.logits(ids)
            m64, mbf = exact.mtp_logits(ids), rounded.mtp_logits(ids)
            mp = exact.picks[-1]
            with open(o.mtp, "wb") as f:
                f.write(b"TFLINGM2" + struct.pack("<3I", T - 1, V, mp.shape[1]))
                f.write(m64.numpy().tobytes() + mbf.numpy().tobytes() + mp.numpy().tobytes())
            print("mtp rows", T - 1, "logit rms", m64.pow(2).mean().sqrt().item())
    nm = sum(1 for i in range(c["num_hidden_layers"]) if i >= c["first_k_dense_replace"])
    picks = forced if forced is not None else torch.stack(exact.picks[:nm])
    with open(o.out, "wb") as f:
        f.write(b"TFLINGF2" + struct.pack("<4I", T, V, picks.shape[0], picks.shape[2]) + struct.pack("<Id", int(forced is not None), exact.gap))
        f.write(ids.int().numpy().tobytes() + l64.numpy().tobytes() + lbf.numpy().tobytes() + picks.numpy().tobytes())
    print("tokens", T, "vocab", V, "moe layers", picks.shape[0], "logit rms", l64.pow(2).mean().sqrt().item(), "forced gap", exact.gap)


if __name__ == "__main__":
    main()
