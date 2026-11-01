"""MiniMax-M3.1-Flash's text model in torch from the release's tensors decoded exactly: the CUDA oracle."""

import argparse
import json
import os
import struct
import time

USAGE = """
The reference decodes the release's MXFP8 projections and NVFP4 experts exactly and runs the first N layers (or all) on
token ids, writes each layer's residual and the logits, and selects sparse blocks per index head as the Metal family and
MSA do (one top-16 per GQA group).

  python tools/zig/minimax_m3_ref.py HF_DIR --ids ids.json --layers 4 --out dumps/     # residuals r0..r3 (bf16 as .npy)
  python tools/zig/minimax_m3_ref.py HF_DIR --ids ids.json --out dumps/ --logits        # all 60 layers, final norm, logits

The rounding follows transformers' model (as the Metal family's zig/kernels/metal/minimax_m3/glue.metal does): bf16
activations, fp32 norms, router and softmax, the residual rounded to bf16 after every add, each routed pick rounded and
added in bf16. --fp32-picks sums the picks in fp32 with one rounding instead, as the CUDA family does (its layer checks).
Text only: no image rows.
"""

import numpy as np
import torch

BF = torch.bfloat16
LM = "language_model."
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


class Source:
    """A safetensors folder read by name: headers once, tensors as raw bytes or numpy."""

    def __init__(self, folder: str, index="model.safetensors.index.json"):
        self.folder = folder
        wm = json.load(open(os.path.join(folder, index)))["weight_map"]
        self.where = {}
        self.headers = {}
        for shard in sorted(set(wm.values())):
            with open(os.path.join(folder, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.headers[shard] = (8 + n, h)
            for name in h:
                self.where[name] = shard

    def info(self, name: str) -> dict:
        return self.headers[self.where[name]][1][name]

    def raw(self, name: str) -> bytes:
        shard = self.where[name]
        base, h = self.headers[shard]
        a, b = h[name]["data_offsets"]
        with open(os.path.join(self.folder, shard), "rb") as f:
            f.seek(base + a)
            return f.read(b - a)

    def np(self, name: str) -> np.ndarray:
        info = self.info(name)
        dt = {"F32": np.float32, "BF16": np.uint16, "F8_E4M3": np.uint8, "U8": np.uint8, "F16": np.float16}[info["dtype"]]
        return np.frombuffer(self.raw(name), dtype=dt).reshape(info["shape"])

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in self.where if n.startswith(prefix))


def e4m3_table() -> np.ndarray:
    """e4m3fn: 256 byte codes to float (no infinities; 0x7f/0xff are NaN in the format, unused by the release)."""
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 7
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


def dequant_nvfp4(packed: np.ndarray, scale: np.ndarray, gscale: np.ndarray, block=16) -> np.ndarray:
    """Codes [..., n, k/2] (low nibble first), e4m3 scales [..., n, k/16], a dividing global scale: fp32 [..., n, k]."""
    lo, hi = packed & 0xF, packed >> 4
    vals = np.stack([FP4[lo], FP4[hi]], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    s = E4M3[scale] / gscale.reshape(*gscale.shape, *([1] * (scale.ndim - gscale.ndim))).astype(np.float32)
    return vals * np.repeat(s, block, axis=-1)


def bf16_from_bits(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.int16)).view(BF)


def mxfp8(src: Source, name: str, dev) -> torch.Tensor:
    """An mxfp8 projection (.weight e4m3 [n, k], .weight_scale_inv e8m0 [n, k/32]) as bf16 [n, k]."""
    w = torch.from_numpy(E4M3[src.np(name + ".weight")]).to(dev)
    s = torch.from_numpy(src.np(name + ".weight_scale_inv").astype(np.int32) - 127).to(dev)
    return (w * torch.exp2(s.float()).repeat_interleave(32, dim=-1)).to(BF)


def gemma_norm(x: torch.Tensor, w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())).to(BF)


def rope_tables(positions: torch.Tensor, rot: int, theta: float, dev):
    inv = theta ** (-torch.arange(0, rot, 2, device=dev, dtype=torch.float32) / rot)
    ang = positions.float()[:, None] * inv[None]
    return torch.cos(ang).to(BF), torch.sin(ang).to(BF)  # [T, rot/2]


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: int) -> torch.Tensor:
    """NeoX half-split over the first `rot` dims of [T, H, D]; the rest pass through. bf16 products, as Metal rounds."""
    h = rot // 2
    a, b, rest = x[..., :h], x[..., h:rot], x[..., rot:]
    c, s = cos[:, None, :], sin[:, None, :]
    ra = (a * c).to(BF) + (-b * s).to(BF)
    rb = (b * c).to(BF) + (a * s).to(BF)
    return torch.cat([ra.to(BF), rb.to(BF), rest], -1)


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha=1.702, limit=7.0) -> torch.Tensor:
    g = torch.clamp(gate.float(), max=limit)
    u = torch.clamp(up.float(), -limit, limit)
    s = torch.sigmoid((g * alpha).to(BF).float()).to(BF).float()
    glu = (g * s).to(BF).float()
    return ((u + 1).to(BF).float() * glu).to(BF)


class Layer:
    def __init__(self, src: Source, i: int, text: dict, dev, fp32_picks=False):
        self.i, self.dev, self.text, self.fp32_picks = i, dev, text, fp32_picks
        p = f"{LM}model.layers.{i}."
        g = lambda n: bf16_from_bits(src.np(p + n)).to(dev)
        self.in_norm, self.post_norm = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        a = p + "self_attn."
        self.q_norm, self.k_norm = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.iq_norm, self.ik_norm = g("self_attn.index_q_norm.weight"), g("self_attn.index_k_norm.weight")
        self.q, self.k, self.v = mxfp8(src, a + "q_proj", dev), mxfp8(src, a + "k_proj", dev), mxfp8(src, a + "v_proj", dev)
        self.iq, self.ik = mxfp8(src, a + "index_q_proj", dev), mxfp8(src, a + "index_k_proj", dev)
        self.o = mxfp8(src, a + "o_proj", dev)
        self.moe = bool(text["moe_layer_freq"][i])
        if not self.moe:
            self.gate_w, self.up_w = mxfp8(src, p + "mlp.gate_proj", dev), mxfp8(src, p + "mlp.up_proj", dev)
            self.down_w = mxfp8(src, p + "mlp.down_proj", dev)
            return
        m = p + "block_sparse_moe."
        self.router = torch.from_numpy(src.np(m + "gate.weight").copy()).to(dev)
        self.bias = torch.from_numpy(src.np(m + "e_score_correction_bias").copy()).to(dev)
        self.s_gate, self.s_up = mxfp8(src, m + "shared_experts.gate_proj", dev), mxfp8(src, m + "shared_experts.up_proj", dev)
        self.s_down = mxfp8(src, m + "shared_experts.down_proj", dev)
        self.src, self.m = src, m
        self.cache = {}

    def expert(self, e: int):
        """Expert e's (w1 gate, w3 up, w2 down) as bf16 [n, k], the release's NVFP4 decoded exactly."""
        if e in self.cache:
            return self.cache[e]
        ws = []
        for w in ("w1", "w3", "w2"):
            x = dequant_nvfp4(self.src.np(f"{self.m}experts.{e}.{w}.weight_packed")[None],
                              self.src.np(f"{self.m}experts.{e}.{w}.weight_scale")[None],
                              self.src.np(f"{self.m}experts.{e}.{w}.weight_global_scale")[None])[0]
            ws.append(torch.from_numpy(np.ascontiguousarray(x)).to(self.dev).to(BF))
        if len(self.cache) < 128:
            self.cache[e] = ws
        return ws

    def attention(self, xn: torch.Tensor, pos: torch.Tensor, cos, sin) -> torch.Tensor:
        t = self.text
        H, HK, D = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        rot = t["rotary_dim"]
        T = xn.shape[0]
        q = gemma_norm((xn @ self.q.T).view(T, H, D), self.q_norm)
        k = gemma_norm((xn @ self.k.T).view(T, HK, D), self.k_norm)
        v = (xn @ self.v.T).view(T, HK, D)
        iq = gemma_norm((xn @ self.iq.T).view(T, HK, D), self.iq_norm)
        ik = gemma_norm((xn @ self.ik.T).view(T, 1, D), self.ik_norm)
        q, k, iq, ik = rope(q, cos, sin, rot), rope(k, cos, sin, rot), rope(iq, cos, sin, rot), rope(ik, cos, sin, rot)
        sp = t["sparse_attention_config"]
        B, topk = sp["sparse_block_size"], sp["sparse_topk_blocks"]
        nb = (int(pos.max()) + B) // B
        # index scores [HK, T, S] fp32, causal; block max; the own block forced; top-k per (row, index head)
        allowed = torch.ones(HK, T, T, dtype=torch.bool, device=self.dev)
        if nb > topk:
            scores = torch.einsum("thd,sd->hts", iq.float(), ik[:, 0].float())
            seen = pos[None, :] <= pos[:, None]  # [T(query), S]: keys at or before the query
            scores = scores.masked_fill(~seen[None], float("-inf"))
            pad = nb * B - T
            scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
            bs = scores.view(HK, T, nb, B).amax(-1)  # [HK, T, nb]
            own = pos // B
            bs.scatter_(2, own[None, :, None].expand(HK, T, 1), float("inf"))
            sel = bs.topk(topk, dim=-1).indices  # [HK, T, topk]
            keep = torch.zeros(HK, T, nb, dtype=torch.bool, device=self.dev)
            keep.scatter_(2, sel, True)
            keep &= bs > float("-inf")
            allowed = keep.repeat_interleave(B, dim=2)[:, :, :T]
        causal = (pos[None, :] <= pos[:, None])[None]  # [1, T, S]
        mask = allowed & causal  # [HK, T, S]
        G = H // HK
        qh = q.view(T, HK, G, D).permute(1, 2, 0, 3)  # [HK, G, T, D]
        att = torch.einsum("hgtd,hsd->hgts", qh, k.permute(1, 0, 2)).to(BF).float() * (D ** -0.5)
        att = att.masked_fill(~mask[:, None], torch.finfo(BF).min)
        p = torch.softmax(att, -1).to(BF)
        o = torch.einsum("hgts,hsd->hgtd", p, v.permute(1, 0, 2))  # bf16
        o = o.permute(2, 0, 1, 3).reshape(T, H * D)
        return o @ self.o.T

    def mlp(self, xn: torch.Tensor) -> torch.Tensor:
        if not self.moe:
            return swiglu_oai(xn @ self.gate_w.T, xn @ self.up_w.T) @ self.down_w.T
        T = xn.shape[0]
        logits = xn.float() @ self.router.T
        p = torch.sigmoid(logits)
        topk = self.text["num_experts_per_tok"]
        sel = (p + self.bias).topk(topk, dim=-1).indices  # ties: torch's order; Metal takes the lower id
        self.last_pick = sel.sort(-1).values
        w = p.gather(1, sel)
        w = w / w.sum(-1, keepdim=True)
        # transformers' order (index_add_ in bf16, experts by id), or with fp32_picks the CUDA family's fp32 sum
        acc = torch.zeros(T, xn.shape[1], dtype=torch.float32, device=self.dev)
        for e in sorted(set(sel.flatten().tolist())):
            rows, slots = torch.nonzero(sel == e, as_tuple=True)
            g_w, u_w, d_w = self.expert(e)
            xe = xn[rows]
            y = (swiglu_oai(xe @ g_w.T, xe @ u_w.T) @ d_w.T).float()
            if self.fp32_picks:
                acc[rows] += y * w[rows, slots, None]
            else:
                acc[rows] = (acc[rows] + (y * w[rows, slots, None]).to(BF).float()).to(BF).float()
        shared = swiglu_oai(xn @ self.s_gate.T, xn @ self.s_up.T) @ self.s_down.T
        scale = self.text["routed_scaling_factor"]
        return ((acc * scale).to(BF).float() + shared.float()).to(BF)

    def forward(self, r: torch.Tensor, pos, cos, sin) -> torch.Tensor:
        xn = gemma_norm(r, self.in_norm)
        r = (r.float() + self.attention(xn, pos, cos, sin).float()).to(BF)
        xn = gemma_norm(r, self.post_norm)
        return (r.float() + self.mlp(xn).float()).to(BF)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir")
    ap.add_argument("--ids", required=True, help="json list of token ids (or {\"ids\": [...]})")
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers (0: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logits", action="store_true", help="also the final norm and lm_head logits of every row")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-picks", action="store_true", help="the CUDA family's MoE sum (fp32 over the picks, one rounding)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    src = Source(a.hf_dir)
    text = json.load(open(os.path.join(a.hf_dir, "config.json")))["text_config"]
    ids = json.load(open(a.ids))
    ids = ids["ids"] if isinstance(ids, dict) else ids
    ids = torch.tensor(ids, device=dev)
    os.makedirs(a.out, exist_ok=True)
    n = a.layers or text["num_hidden_layers"]
    pos = torch.arange(len(ids), device=dev)
    cos, sin = rope_tables(pos, text["rotary_dim"], text["rope_theta"], dev)
    embed = bf16_from_bits(src.np(f"{LM}model.embed_tokens.weight"))
    r = embed[ids.cpu()].to(dev)
    np.save(os.path.join(a.out, "embed.npy"), r.view(torch.int16).cpu().numpy())
    with torch.no_grad():
        for i in range(n):
            t = time.time()
            layer = Layer(src, i, text, dev, a.fp32_picks)
            r = layer.forward(r, pos, cos, sin)
            np.save(os.path.join(a.out, f"r{i}.npy"), r.view(torch.int16).cpu().numpy())
            if layer.moe:
                np.save(os.path.join(a.out, f"pick{i}.npy"), layer.last_pick.cpu().numpy())
            print(f"layer {i} {time.time() - t:.1f} s", flush=True)
            del layer
        if a.logits:
            norm_w = bf16_from_bits(src.np(f"{LM}model.norm.weight")).to(dev)
            head = bf16_from_bits(src.np(f"{LM}lm_head.weight")).to(dev)
            logits = (gemma_norm(r, norm_w) @ head.T)
            np.save(os.path.join(a.out, "logits.npy"), logits.view(torch.int16).cpu().numpy())
            print("greedy", logits.float().argmax(-1).tolist())


if __name__ == "__main__":
    main()
