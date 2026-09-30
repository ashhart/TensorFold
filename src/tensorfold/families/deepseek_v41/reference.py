"""A slow, single-GPU reference forward of DeepSeek-V4.1-Flash over one prompt, streaming layers from the checkpoint.

Quality and parity checks only: it teacher-forces one sequence of at most 512 tokens (where every layer's sparse
selection takes all visible compressed entries, so no indexer or candidate blocks run), loads each layer's EXL3
weights, runs it for all positions and frees it. Arithmetic follows notes/dsv41/ARCH.md: RMS in fp32 (eps 1e-20),
the residual streams in bf16 between sublayers, hyper-connection math, compressor pooling and attention in fp32.

    python -m tensorfold.families.deepseek_v41.reference MODEL_DIR ENGRAM_DIR --golden golden.json --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from tensorfold.cuda.exl3.linear import Exl3Linear

from . import engram as E
from .config import Config

VARIANT = set(filter(None, os.environ.get("DSV41_REF_VARIANT", "").split(",")))   # debugging toggles
MAX_TOKENS = 512            # beyond this the indexer's top-512 selection would drop entries
BF = torch.bfloat16
F32 = torch.float32


class Checkpoint:
    def __init__(self, root: Path, device: str = "cuda") -> None:
        self.root, self.device = root, device
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}
        self.memo: dict[str, object] = {}         # this layer's tensors and linears, dropped by clear()

    def clear(self) -> None:
        """Drop this layer's tensors and the open files (safetensors keeps what a handle has read in host memory)."""

        self.memo.clear()
        self.files.clear()

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str, dtype=None) -> torch.Tensor:
        key = f"{name}:{dtype}"
        if key not in self.memo:
            self.memo[key] = self._get(name, dtype)
        return self.memo[key]

    def _get(self, name: str, dtype=None) -> torch.Tensor:
        f = self.index[name]
        if f not in self.files:
            self.files[f] = safe_open(str(self.root / f), framework="pt", device=str(torch.device(self.device, 0)))
        t = self.files[f].get_tensor(name)
        return t if dtype is None else t.to(dtype)

    def linear(self, prefix: str) -> Exl3Linear:
        key = prefix + ":linear"
        if key not in self.memo:
            self.memo[key] = Exl3Linear.from_tensors(self._get(prefix + ".trellis"), self._get(prefix + ".suh"),
                                                     self._get(prefix + ".svh"), "mul1", device=self.device)
        return self.memo[key]


def lin(layer: Exl3Linear, x: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """x [T, K] @ W in chunks of 128 rows (the decode linear's limit)."""

    return torch.cat([layer(x[i:i + 128].contiguous(), out_dtype=out_dtype) for i in range(0, x.shape[0], 128)])


def rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 RMSNorm; the caller rounds."""

    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y * weight.float() if weight is not None else y


def inv_freq(c: Config, ratio: int, device) -> torch.Tensor:
    """32 rotary frequencies: plain theta 1e4 on window-only layers, YaRN theta 1.6e5 on compressed ones."""

    dim = c.qk_rope_head_dim
    i = torch.arange(0, dim, 2, dtype=torch.float64, device=device)
    if ratio == 0:
        return (1.0 / c.rope_theta ** (i / dim)).float()
    base = c.compress_rope_theta
    freqs = 1.0 / base ** (i / dim)

    def corr(rot: float) -> float:
        return dim * math.log(c.rope_original / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corr(c.beta_fast)), 0)
    high = min(math.ceil(corr(c.beta_slow)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float64, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
    return (freqs * (1 - ramp) + freqs / c.rope_factor * ramp).float()


def rope(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """GPT-J interleaved rotation of the last 64 dims of x [T, (heads,) D] at positions pos [T]; fp32 out."""

    x = x.float().clone()
    ang = pos.double()[:, None] * freqs.double()[None, :]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [freqs.numel()]
    cos, sin = ang.cos().float().view(shape), ang.sin().float().view(shape)
    r = x[..., :2 * freqs.numel()] if "ropefirst" in VARIANT else x[..., -2 * freqs.numel():]
    half = freqs.numel()
    if "neox" in VARIANT:
        e, o = r[..., :half].clone(), r[..., half:].clone()
    else:
        e, o = r[..., 0::2].clone(), r[..., 1::2].clone()
    if inverse:
        sin = -sin
    if "neox" in VARIANT:
        r[..., :half] = e * cos - o * sin
        r[..., half:] = o * cos + e * sin
    else:
        r[..., 0::2] = e * cos - o * sin
        r[..., 1::2] = o * cos + e * sin
    return x


class HCMix:
    def __init__(self, ck: Checkpoint, prefix: str) -> None:
        self.fn = ck.get(prefix + "_fn", F32)          # [24, 20480]
        self.base = ck.get(prefix + "_base", F32)      # [24]
        self.scale = ck.get(prefix + "_scale", F32)    # [3]

    def __call__(self, X: torch.Tensor, pre_in: torch.Tensor, norm_w: torch.Tensor, c: Config):
        """X bf16 [T, 4, D] -> post [T,4], comb [T,4,4], normed collapsed input bf16 [T, D], this sublayer's pre."""

        T, S, D = X.shape
        eps = c.hc_eps
        xf = X.float().reshape(T, S * D)
        mix = (xf @ self.fn.T) * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        pre = torch.sigmoid(mix[:, 0:S] * self.scale[0] + self.base[0:S]) + eps
        post = 2 * torch.sigmoid(mix[:, S:2 * S] * self.scale[1] + self.base[S:2 * S])
        comb = mix[:, 2 * S:].view(T, S, S) * self.scale[2] + self.base[2 * S:].view(S, S)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(c.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        x_in = ((pre if "nodelay" in VARIANT else pre_in)[:, :, None] * X.float()).sum(1)
        x_in = rms(x_in.to(BF), norm_w, c.rms_norm_eps).to(BF)
        return post, comb, x_in, pre


def hc_post(b: torch.Tensor, X: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Y[:, j] = post_j * b + sum_i comb[i, j] * X[:, i], fp32 math, bf16 streams."""

    y = post[:, :, None] * b.float()[:, None, :] + torch.einsum("tij,tid->tjd", comb, X.float())
    return y.to(BF)


def attention(ck: Checkpoint, c: Config, L: int, x: torch.Tensor, pos: torch.Tensor, caches: dict) -> torch.Tensor:
    p = f"layers.{L}.attn"
    T = x.shape[0]
    H, Dh = c.num_attention_heads, c.head_dim
    ratio = c.layer_ratios[L]
    freqs = inv_freq(c, ratio, x.device)
    eps = c.rms_norm_eps

    qr = rms(lin(ck.linear(p + ".wq_a"), x), ck.get(p + ".q_norm.weight"), eps).to(BF)
    kv = rms(lin(ck.linear(p + ".wkv"), x), ck.get(p + ".kv_norm.weight"), eps).to(BF)
    q = lin(ck.linear(p + ".wq_b"), qr).view(T, H, Dh)
    q = rms(q, None, eps)                                   # per-head RMS, no weight
    q = rope(q, pos, freqs)                                 # fp32 [T, H, Dh]
    kv = rope(kv, pos, freqs).to(BF).float()                # this layer's window rows [T, Dh]

    # compressed entries: layer L's kv source wrote them earlier in this forward (sources compute them here)
    comp = None
    if ratio > 0:
        src = max(s for s in c.kv_source_layer_ids if s <= L)
        if src == L:
            caches[src] = compress(ck, c, L, x, ratio, freqs)
        comp = caches[src]                                  # fp32 [C, Dh], entry j covers [j r, j r + r - 1]

    scale = Dh ** -0.5
    sink = ck.get(p + ".attn_sink", F32)                    # [H]
    t = torch.arange(T, device=x.device)
    # window: keys s with p - 127 <= s <= p
    win_mask = (t[None, :] <= t[:, None]) & (t[None, :] >= t[:, None] - (c.sliding_window - 1))
    keys = kv
    mask = win_mask
    skip = "nocomp" in VARIANT or f"nocomp{ratio}" in VARIANT
    if "nowin" in VARIANT and ratio > 0:
        win_mask = win_mask & (t[None, :] == t[:, None])
        mask = win_mask
    if comp is not None and comp.shape[0] and not skip:
        C = comp.shape[0]
        vis = torch.arange(C, device=x.device)[None, :] < ((t[:, None] + 1) // ratio)
        keys = torch.cat([comp, kv])
        mask = torch.cat([vis, win_mask], dim=1)
    scores = torch.einsum("thd,sd->ths", q, keys) * scale  # [T, H, S]
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    if "nosink" in VARIANT:
        sink = torch.full_like(sink, float("-inf"))
    full = torch.cat([scores, sink.view(1, H, 1).expand(T, H, 1)], dim=-1)
    w = torch.softmax(full, dim=-1)[..., :-1]
    o = torch.einsum("ths,sd->thd", w, keys)               # V = K
    o = rope(o, pos, freqs, inverse=True)                   # fp32
    o = o.to(BF).view(T, c.o_groups, (H // c.o_groups) * Dh)
    z = torch.cat([lin(ck.linear(f"{p}.wo_a.slice.{g}"), o[:, g].contiguous()) for g in range(c.o_groups)], dim=1)
    return lin(ck.linear(p + ".wo_b"), z)


def compress(ck: Checkpoint, c: Config, L: int, x: torch.Tensor, ratio: int, freqs: torch.Tensor) -> torch.Tensor:
    """Compressed KV rows of a kv source (RoPE at the group's first position), fp32 [T // ratio, Dh]."""

    p = f"layers.{L}.attn.compressor"
    T = x.shape[0]
    kv = lin(ck.linear(p + ".wkv"), x, out_dtype=F32)
    norm_w = ck.get(p + ".norm.weight")
    if ratio == 1:
        latent = rms(kv, norm_w, c.rms_norm_eps).to(BF)
    else:
        gate = lin(ck.linear(p + ".wgate"), x, out_dtype=F32)
        n = T // ratio
        kvg = kv[:n * ratio].view(n, ratio, -1)
        g = torch.softmax(gate[:n * ratio].view(n, ratio, -1), dim=1)
        latent = rms((g * kvg).sum(1), norm_w, c.rms_norm_eps).to(BF)
    start = torch.arange(latent.shape[0], device=x.device) * ratio + (ratio - 1 if "compend" in VARIANT else 0)
    return rope(latent, start, freqs).to(BF).float()


def moe(ck: Checkpoint, c: Config, L: int, x: torch.Tensor) -> torch.Tensor:
    p = f"layers.{L}.ffn"
    T = x.shape[0]
    limit = c.swiglu_limit
    logits = x.float() @ ck.get(p + ".gate.weight", F32).T
    sc = torch.sqrt(torch.nn.functional.softplus(logits))
    bias = 0 if "nobias" in VARIANT else ck.get(p + ".gate.bias", F32)
    idx = torch.topk(sc + bias, c.num_experts_per_tok, dim=-1).indices
    w = sc.gather(1, idx)
    w = w / w.sum(-1, keepdim=True) * (1.0 if "noscale" in VARIANT else c.routed_scaling_factor)

    def expert(prefix: str, xs: torch.Tensor) -> torch.Tensor:
        g = lin(ck.linear(prefix + ".w1"), xs).float()
        u = lin(ck.linear(prefix + ".w3"), xs).float()
        a = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
        return lin(ck.linear(prefix + ".w2"), a.to(BF)).float()

    y = torch.zeros((T, c.hidden_size), dtype=F32, device=x.device)
    for e in torch.unique(idx).tolist():
        rows, slot = (idx == e).nonzero(as_tuple=True)
        y.index_add_(0, rows, expert(f"{p}.experts.{e}", x[rows]) * w[rows, slot, None])
    if "noshared" not in VARIANT:
        y += expert(p + ".shared_experts", x)
    return y.to(BF)


def engram_apply(ck: Checkpoint, c: Config, L: int, ell: int, X: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """X bf16 [T, 4, D] plus a gated value from the layer's 24 hashed rows (fp32 [T, 24, 256])."""

    p = f"layers.{L}.engram"
    T, S, D = X.shape
    kv = lin(ck.linear(p + ".wkv"), rows.to(BF).reshape(T, -1))            # [T, 5 D]
    qw, kw = ck.get(p + ".q_weight", F32), ck.get(p + ".k_weight", F32)
    h = X.float()
    key = kv[:, :S * D].view(T, S, D).float()
    val = kv[:, S * D:].float()
    eps = c.rms_norm_eps
    dot = (h * qw * kw * key).sum(-1)
    dot = dot * torch.rsqrt(h.pow(2).mean(-1) + eps) * torch.rsqrt(key.pow(2).mean(-1) + eps) / math.sqrt(D)
    gate = torch.sigmoid(torch.sign(dot) * torch.sqrt(dot.abs().clamp(min=1e-6)))
    return (h + gate[:, :, None] * val[:, None, :]).to(BF)


class Sequence:
    """One prompt's state through the layers."""

    def __init__(self, ids: list[int], c: Config, ck: Checkpoint, rows_idx: np.ndarray) -> None:
        dev = ck.device
        self.ids, self.T = ids, len(ids)
        self.pos = torch.arange(self.T, device=dev)
        self.rows_idx = rows_idx
        e = ck.get("embed.weight")[torch.tensor(ids, device=dev)]
        self.X = e[:, None, :].expand(self.T, c.hc_mult, c.hidden_size).contiguous()
        self.pre = torch.zeros((self.T, c.hc_mult), dtype=F32, device=dev)
        self.pre[:, 0] = 1.0                                # identity collapse: every stream is the embedding
        self.caches: dict[int, torch.Tensor] = {}
        self.f = self.post = self.comb = None
        self.probes: list[torch.Tensor] = []


def layer_step(ck: Checkpoint, c: Config, L: int, s: Sequence, tables: E.Tables) -> None:
    X = s.X
    if L > 0:
        X = hc_post(s.f, X, s.post, s.comb)
    if L in c.engram_layer_ids and "noengram" not in VARIANT:
        ell = c.engram_layer_ids.index(L)
        rows = tables.rows(ell, s.rows_idx[:, ell, :]).to(ck.device)
        X = engram_apply(ck, c, L, ell, X, rows)
    post, comb, x, pre = HCMix(ck, f"layers.{L}.hc_attn")(X, s.pre, ck.get(f"layers.{L}.attn_norm.weight"), c)
    a = attention(ck, c, L, x, s.pos, s.caches)
    X = hc_post(a, X, post, comb)
    s.post, s.comb, x, s.pre = HCMix(ck, f"layers.{L}.hc_ffn")(X, pre, ck.get(f"layers.{L}.ffn_norm.weight"), c)
    s.f = moe(ck, c, L, x)
    s.X = X
    s.probes.append(X.float().mean(1)[-1].cpu())
    dump = os.environ.get("TF_REF_DUMP_DIR")
    if dump and s.T == int(os.environ.get("TF_DUMP_TOKENS", "365")):
        os.makedirs(dump, exist_ok=True)
        torch.save({"stream": hc_post(s.f, X, s.post, s.comb).cpu(), "pre": s.pre.cpu(), "ffn_out": s.f.cpu()},
                   f"{dump}/layer{L:02d}.pt")


def forward(model_dir: Path, engram_dir: Path, prompts: list[list[int]], *, layers: int | None = None,
            log=print) -> list[dict[str, torch.Tensor]]:
    """Per prompt: logits fp32 [T, vocab] and the last position's mean stream after each layer (a parity probe)."""

    for ids in prompts:
        if not 0 < len(ids) <= MAX_TOKENS:
            raise ValueError(f"the reference takes 1..{MAX_TOKENS} tokens a prompt, got {len(ids)}")
    c = Config.from_dict(json.loads((model_dir / "config.json").read_text()))
    ck = Checkpoint(model_dir)
    layout = E.Layout.from_config(c)
    tmap_file = model_dir / "engram_token_map.npy"
    tmap = np.load(tmap_file) if tmap_file.exists() else E.token_map(model_dir / "tokenizer.json",
                                                                     c.engram_compressed_vocab_size)
    tables = E.Tables(engram_dir, c.engram_layer_ids)
    seqs = [Sequence(ids, c, ck, E.hashes(np.array(ids), tmap.astype(np.int64), layout, c.engram_pad_token_id))
            for ids in prompts]
    ck.clear()
    n_layers = c.num_hidden_layers if layers is None else layers
    for L in range(n_layers):
        t0 = time.time()
        for s in seqs:
            layer_step(ck, c, L, s, tables)
        torch.cuda.synchronize()
        ck.clear()
        torch.cuda.empty_cache()
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30
        log(f"layer {L:2d} {time.time() - t0:5.1f} s  |x| {seqs[-1].X.float().norm(dim=-1).mean():.1f}  "
            f"rss {rss:.1f} GiB  cuda alloc {torch.cuda.memory_allocated() / 2**30:.1f} "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    out = []
    for s in seqs:
        X = hc_post(s.f, s.X, s.post, s.comb)
        h = (s.pre[:, :, None] * X.float()).sum(1).to(BF)
        h = rms(h, ck.get("norm.weight"), c.rms_norm_eps).to(BF)
        out.append({"ids": s.ids, "logits": lin(ck.linear("head"), h, out_dtype=F32).cpu(),
                    "probes": torch.stack(s.probes)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("engram", type=Path)
    ap.add_argument("--ids", help="comma-separated token ids of one prompt")
    ap.add_argument("--golden", type=Path, help="a tools/dsv41_golden.py file: every prompt in it")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--out", type=Path, required=True, help="directory for ref-K.pt files")
    args = ap.parse_args()
    if args.golden:
        prompts = [g["ids"] for g in json.loads(args.golden.read_text())["goldens"]]
    else:
        prompts = [[int(t) for t in args.ids.split(",")]]
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outs = forward(args.model, args.engram, prompts, layers=args.layers)
    for k, o in enumerate(outs):
        torch.save(o, args.out / f"ref-{k}.pt")
        print(f"prompt {k}: greedy next token {int(o['logits'][-1].argmax())}")


if __name__ == "__main__":
    main()
