"""Torch DFlash2 proposals for the exact 128-node CUDA verifier.

The draft model may disagree numerically with its MLX implementation: every
proposal is checked by the target model, so disagreement changes acceptance,
not output. The target taps and the checkpoint are the original Qwen3.8 ones.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from .glue import embed, swiglu
from .qmm import group_sums
from .qmm_fast import matmul, tile, untile
from .weights import QLinear, Weights


@triton.jit
def _dconv_kernel(X, DYN, BASE, RES, OUT, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                  BRANCH: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """``_conv`` (plus the residual add that follows it) as one kernel: row r mixes rows r and r-1."""

    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < D
    x = tl.load(X + row * D + c, mask=ok, other=0.0).to(tl.float32)
    prev = tl.load(X + (row - 1) * D + c, mask=ok & (row > 0), other=0.0).to(tl.float32)
    grp = c // GS
    d0 = tl.load(DYN + ((row * 2 + BRANCH) * 2) * G + grp, mask=ok, other=0.0).to(tl.float32)
    d1 = tl.load(DYN + ((row * 2 + BRANCH) * 2 + 1) * G + grp, mask=ok, other=0.0).to(tl.float32)
    b0 = tl.load(BASE + (BRANCH * 2) * D + c, mask=ok, other=0.0).to(tl.float32)
    b1 = tl.load(BASE + (BRANCH * 2 + 1) * D + c, mask=ok, other=0.0).to(tl.float32)
    k0 = (b0 + d0).to(tl.bfloat16).to(tl.float32)
    k1 = (b1 + d1).to(tl.bfloat16).to(tl.float32)
    y = (x * k0 + prev * k1).to(tl.bfloat16)
    if HAS_RES:
        r = tl.load(RES + row * D + c, mask=ok, other=0.0).to(tl.float32)
        y = (r + y.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * D + c, y, mask=ok)


@triton.jit
def _prep_kernel(QKV, QN, KN, COS, SIN, QO, KO, VO, L, stride, eps,
                 H: tl.constexpr, HKV: tl.constexpr, HALF: tl.constexpr):
    """Program (row, head) over [q heads | k heads | v heads] of a projection row: q and k get their
    RMS norm and rotary embedding, v is copied; outputs are (heads, rows, head_dim)."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HALF)
    D: tl.constexpr = 2 * HALF
    src = QKV + row * stride + head * D
    if head < H + HKV:
        a = tl.load(src + d).to(tl.float32)
        b = tl.load(src + HALF + d).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(a * a, axis=0) + tl.sum(b * b, axis=0)) / D + eps)
        if head < H:
            wa = tl.load(QN + d).to(tl.float32)
            wb = tl.load(QN + HALF + d).to(tl.float32)
        else:
            wa = tl.load(KN + d).to(tl.float32)
            wb = tl.load(KN + HALF + d).to(tl.float32)
        a = (a * rstd * wa).to(tl.bfloat16).to(tl.float32)
        b = (b * rstd * wb).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(COS + row * HALF + d)
        sin = tl.load(SIN + row * HALF + d)
        ra = (a * cos - b * sin).to(tl.bfloat16)
        rb = (b * cos + a * sin).to(tl.bfloat16)
        if head < H:
            dst = QO + (head * L + row) * D
        else:
            dst = KO + ((head - H) * L + row) * D
        tl.store(dst + d, ra)
        tl.store(dst + HALF + d, rb)
    else:
        dst = VO + ((head - H - HKV) * L + row) * D
        tl.store(dst + d, tl.load(src + d))
        tl.store(dst + HALF + d, tl.load(src + HALF + d))


def _dconv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int, group_size: int,
           residual: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = x.shape
    x = x.contiguous()
    dyn = dyn.contiguous()
    out = torch.empty_like(x)
    block = 1024
    _dconv_kernel[(rows, triton.cdiv(d, block))](x, dyn, base, residual if residual is not None else x, out,
                                                 D=d, G=d // group_size, GS=group_size, BRANCH=branch,
                                                 HAS_RES=residual is not None, BLOCK=block, num_warps=4)
    return out


def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
            * weight.float()).to(torch.bfloat16)


def _conv(x: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor,
          branch: int, group_size: int) -> torch.Tensor:
    """Two-tap grouped dynamic causal convolution on the masked block."""

    prev = torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0)
    k0 = base[branch, 0] + dynamic[:, branch, 0].repeat_interleave(group_size, -1)
    k1 = base[branch, 1] + dynamic[:, branch, 1].repeat_interleave(group_size, -1)
    return (x.float() * k0.float() + prev.float() * k1.float()).to(torch.bfloat16)


def _rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Qwen rotate-half RoPE; x is (heads, positions, head_dim)."""

    half = x.shape[-1] // 2
    phase = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = phase.cos()[None], phase.sin()[None]
    a, b = x[..., :half].float(), x[..., half:].float()
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(torch.bfloat16)


def _linear(x: torch.Tensor, weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return F.linear(x, weights[name])


def quantize4(w: torch.Tensor) -> QLinear:
    """bf16 (N, K) -> MLX-style affine 4-bit, groups of 64 along K: q = round((w - min) / scale)."""

    n, k = w.shape
    g = w.float().view(n, k // 64, 64)
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int32)
    q = q.view(n, k // 8, 8)
    words = torch.zeros((n, k // 8), dtype=torch.int32, device=w.device)
    for j in range(8):
        words |= q[..., j] << (4 * j)
    return QLinear(words, scale.contiguous(), bias.contiguous())


# The tree policy's knobs (the Metal engine's values). Under keyed sampling a draft is accepted only
# if it is the exact token the target samples, and the target's Gumbel noise for each position and
# token is known in advance, so ``noise`` weighs that noise into the draft scores; ``nucleus`` drops
# candidates outside the drafter's own top-p set, which the target can never sample.
POLICY = {"edge": 0.6, "noise": 0.7, "scale": 1.5, "branch": 4, "nucleus": False}


def _best_first(cands: np.ndarray, unary: np.ndarray, projected: np.ndarray,
                pred: np.ndarray, succ: np.ndarray, anchor: int, max_nodes: int,
                sampling: Sampling | None, first_position: int) -> tuple[list[int], list[int]]:
    """The Metal best-first lattice policy without MLX or a host-side model copy."""

    depth_count = cands.shape[0]
    temp = max(float(sampling.temperature), 1e-6) if sampling is not None else 1.0
    noise = None
    if sampling is not None:
        positions = first_position + np.arange(depth_count)
        noise = -np.log(-np.log(uniform_rows(sampling.seed, positions, cands)))
    successor = [succ[cands[d]].astype(np.float64) for d in range(depth_count)]
    heap: list[tuple[float, int, int, int]] = []
    tokens: list[int] = []
    parents: list[int] = []

    def expand(token: int, depth: int, parent: int, path_score: float) -> None:
        edge = successor[depth] @ (pred[token].astype(np.float64) * projected[depth])
        values = (unary[depth] + POLICY["edge"] * edge) / temp
        if noise is not None:
            if POLICY["nucleus"] and sampling.top_p < 1:
                probs = np.exp(values - values.max())
                probs /= probs.sum()
                order = np.argsort(-probs)
                keep = int(np.searchsorted(np.cumsum(probs[order]), sampling.top_p)) + 1
                values = np.full_like(values, -np.inf)
                values[order[:keep]] = ((unary[depth] + POLICY["edge"] * edge) / temp)[order[:keep]]
            values = values + POLICY["noise"] * noise[depth]
        values = values / POLICY["scale"]
        values -= values.max()
        logp = values - np.log(np.exp(values).sum())
        for i in np.argsort(-logp)[:POLICY["branch"]]:
            if not np.isfinite(logp[i]):
                break
            heapq.heappush(heap, (path_score - float(logp[i]), parent,
                                  int(cands[depth, i]), depth))

    expand(anchor, 0, -1, 0.0)
    while heap and len(tokens) < max_nodes:
        score, parent, token, depth = heapq.heappop(heap)
        me = len(tokens)
        tokens.append(token)
        parents.append(parent)
        if depth + 1 < depth_count:
            expand(token, depth + 1, me, score)
    return tokens, parents


class DFlash2:
    """Five-layer DFlash2; context is the last 2,047 committed target taps."""

    def __init__(self, draft_dir: str | Path, target: Weights, bits: int = 4, block: int = 16,
                 fast: bool = True, rank: int = 0, world: int = 1):
        path = Path(draft_dir)
        # ``world=2``: this rank drafts with half the attention heads, half the MLP and half the draft
        # vocabulary (row-parallel sums gathered and added in rank order, top candidates merged), so
        # both ranks call ``propose_tree`` and ``add_taps`` together. Needs the fused path.
        if world not in (1, 2) or (world == 2 and not (fast and bits == 4)):
            raise ValueError("a two-rank drafter needs world=2 with the fused 4-bit path")
        self.rank, self.world = rank, world
        self.block = block          # masked positions drafted per round (the checkpoint trained with 8)
        cfg = json.loads((path / "config.json").read_text())
        self.hidden = int(cfg["hidden_size"])
        self.head_dim = int(cfg["head_dim"])
        self.heads = int(cfg["num_attention_heads"])
        self.kv_heads = int(cfg["num_key_value_heads"])
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.mask_id = int(cfg["dflash_config"]["mask_token_id"])
        self.group_size = int(cfg["dflash_config"]["conv_group_size"])
        self.layers = int(cfg["num_hidden_layers"])
        self.window = int(cfg["sliding_window"]) - 1
        self.is_causal = bool(cfg.get("is_causal", True))
        self.target = target
        self.device = target.norm.device
        self.weights: dict[str, torch.Tensor] = {}
        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name in ("candidate_selector.predecessor_codebook",
                            "candidate_selector.successor_codebook"):
                    self.weights[name] = tensor.float().numpy().copy()
                else:
                    self.weights[name] = tensor.to(self.device)
        self.inv_freq = (1.0 / self.theta **
                         (torch.arange(self.head_dim // 2, device=self.device,
                                       dtype=torch.float32) * 2 / self.head_dim))
        spans = ((0, 98304), (248032, min(248320, target.config.vocab)))
        self.vocab_spans = spans
        self.head_ids = torch.cat([torch.arange(a, b, device=self.device) for a, b in spans])
        head = untile(target.head)
        self.sub_head = QLinear(torch.cat([head.weight[a:b] for a, b in spans]).contiguous(),
                                torch.cat([head.scales[a:b] for a, b in spans]).contiguous(),
                                torch.cat([head.biases[a:b] for a, b in spans]).contiguous())
        if world == 2:
            half = -(-len(self.head_ids) // 2)
            lo, hi = rank * half, min((rank + 1) * half, len(self.head_ids))
            self.head_ids = self.head_ids[lo:hi].contiguous()
            self.sub_head = QLinear(self.sub_head.weight[lo:hi].contiguous(), self.sub_head.scales[lo:hi].contiguous(),
                                    self.sub_head.biases[lo:hi].contiguous())
        if target.head.layout == "tiled":
            self.sub_head = tile(self.sub_head)
        del head
        # The large projections run as 4-bit lane matmuls: drafting reads ~1 GB a round instead of
        # ~3.5 GB of bf16. Drafts only change acceptance, never the output.
        self.q4: dict[str, QLinear] = {}
        # ``fast``: [q | k | v] (and [k | v] for context rows) as one projection each, the norms,
        # rotary embedding and dynamic convolutions as fused kernels, one mask and one rotary table
        # a round: ~170 kernels a round instead of ~920. Different rounding, so different drafts.
        self.fast = fast and bits == 4
        self.heads_local, self.kv_local = self.heads // world, self.kv_heads // world
        if self.fast:
            for layer in range(self.layers):
                prefix = f"layers.{layer}.self_attn."
                q, k, v = (self.weights[prefix + n] for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
                if world == 2:
                    hq, hk = self.heads_local * self.head_dim, self.kv_local * self.head_dim
                    q, k, v = q[rank * hq:(rank + 1) * hq], k[rank * hk:(rank + 1) * hk], v[rank * hk:(rank + 1) * hk]
                    o = self.weights[prefix + "o_proj.weight"]
                    self.weights[prefix + "o_proj.weight"] = o[:, rank * hq:(rank + 1) * hq].contiguous()
                    mlp = f"layers.{layer}.mlp."
                    inter = self.weights[mlp + "gate_proj.weight"].shape[0] // 2
                    for n in ("gate_proj.weight", "up_proj.weight"):
                        self.weights[mlp + n] = self.weights[mlp + n][rank * inter:(rank + 1) * inter].contiguous()
                    self.weights[mlp + "down_proj.weight"] = \
                        self.weights[mlp + "down_proj.weight"][:, rank * inter:(rank + 1) * inter].contiguous()
                self.weights[prefix + "qkv.weight"] = torch.cat((q, k, v)).contiguous()
                self.weights[prefix + "kv.weight"] = torch.cat((k, v)).contiguous()
                for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                    del self.weights[prefix + n]
                for conv in ("attention_conv", "mlp_conv"):
                    base = self.weights[f"layers.{layer}.{conv}.base_kernel"]
                    if base.shape != (2, 2, self.hidden):
                        raise ValueError(f"unexpected base kernel shape {tuple(base.shape)}")
                    self.weights[f"layers.{layer}.{conv}.base_kernel"] = base.to(torch.bfloat16).contiguous()
        if bits == 4:
            for name in list(self.weights):
                t = self.weights[name]
                if (isinstance(t, torch.Tensor) and t.ndim == 2 and name.endswith(".weight")
                        and t.shape[0] % 64 == 0 and t.shape[1] % 64 == 0 and t.numel() >= 1 << 20):
                    self.q4[name] = tile(quantize4(t.to(torch.bfloat16)))
                    del self.weights[name]
            torch.cuda.empty_cache()
        # Keys and values of the context rows, per layer: each depends only on its row and position,
        # so they are projected once when the row arrives instead of every round.
        self.kc: list[torch.Tensor | None] = [None] * self.layers
        self.vc: list[torch.Tensor | None] = [None] * self.layers
        self.context_len = 0
        self.context_end = 0

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        q = self.q4.get(name)
        if q is None:
            return F.linear(x, self.weights[name])
        x = x.to(torch.bfloat16).contiguous()
        if x.shape[0] <= 128:
            return matmul(x, q)
        return torch.cat([matmul(x[i:i + 128], q) for i in range(0, x.shape[0], 128)])

    def snapshot(self):
        return (list(self.kc), list(self.vc), self.context_len, self.context_end)

    def restore(self, snap) -> None:
        kc, vc, self.context_len, self.context_end = snap
        self.kc, self.vc = list(kc), list(vc)

    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        if taps.ndim != 2 or taps.shape[1] != 5 * self.hidden:
            raise ValueError("DFlash2 expects five target layer taps per committed row")
        projected = _norm(self._lin(taps, "fc.weight"), self.weights["hidden_norm.weight"], self.eps)
        n = projected.shape[0]
        if self.fast:
            cos, sin = self._rotary(self.context_end, n)
            for layer in range(self.layers):
                _, k, v = self._prep(self._lin(projected, f"layers.{layer}.self_attn.kv.weight"), layer, cos, sin, 0)
                kc, vc = self.kc[layer], self.vc[layer]
                kc = k if kc is None else torch.cat((kc, k), dim=1)
                vc = v if vc is None else torch.cat((vc, v), dim=1)
                self.kc[layer] = kc[:, -self.window:].contiguous()
                self.vc[layer] = vc[:, -self.window:].contiguous()
            self.context_len = min(self.window, self.context_len + n)
            self.context_end += n
            return
        pos = torch.arange(self.context_end, self.context_end + n, device=self.device)
        for layer in range(self.layers):
            prefix = f"layers.{layer}.self_attn."
            k = self._lin(projected, prefix + "k_proj.weight").view(-1, self.kv_heads, self.head_dim)
            v = self._lin(projected, prefix + "v_proj.weight").view(-1, self.kv_heads, self.head_dim)
            k = _rope(_norm(k, self.weights[prefix + "k_norm.weight"], self.eps).transpose(0, 1), pos, self.inv_freq)
            v = v.transpose(0, 1)
            kc, vc = self.kc[layer], self.vc[layer]
            kc = k if kc is None else torch.cat((kc, k), dim=1)
            vc = v if vc is None else torch.cat((vc, v), dim=1)
            self.kc[layer] = kc[:, -self.window:].contiguous()
            self.vc[layer] = vc[:, -self.window:].contiguous()
        self.context_len = min(self.window, self.context_len + n)
        self.context_end += n

    def _attention(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w = self.weights
        prefix = f"layers.{layer}.self_attn."
        kc, vc = self.kc[layer], self.vc[layer]
        q = self._lin(x, prefix + "q_proj.weight").view(-1, self.heads, self.head_dim)
        kn = self._lin(x, prefix + "k_proj.weight").view(-1, self.kv_heads, self.head_dim)
        vn = self._lin(x, prefix + "v_proj.weight").view(-1, self.kv_heads, self.head_dim)
        q = _norm(q, w[prefix + "q_norm.weight"], self.eps).transpose(0, 1)
        kn = _norm(kn, w[prefix + "k_norm.weight"], self.eps).transpose(0, 1)
        vn = vn.transpose(0, 1)
        npos = torch.arange(self.context_end, self.context_end + len(x), device=self.device)
        q = _rope(q, npos, self.inv_freq)
        kn = _rope(kn, npos, self.inv_freq)
        keys = torch.cat((kc, kn), dim=1).unsqueeze(0)
        values = torch.cat((vc, vn), dim=1).unsqueeze(0)
        keys = keys.repeat_interleave(self.heads // self.kv_heads, dim=1)
        values = values.repeat_interleave(self.heads // self.kv_heads, dim=1)
        s, length = kc.shape[1], len(x)
        qidx = torch.arange(length, device=self.device)[:, None]
        kidx = torch.arange(s + length, device=self.device)[None, :]
        context_allowed = (kidx < s) & (s + qidx - kidx < self.window + 1)
        block_allowed = kidx >= s
        if self.is_causal:
            block_allowed = block_allowed & (kidx <= s + qidx)
        allowed = context_allowed | block_allowed
        output = F.scaled_dot_product_attention(q.unsqueeze(0), keys, values,
                                                attn_mask=allowed[None, None],
                                                scale=self.head_dim ** -0.5)
        output = output.squeeze(0).transpose(0, 1).reshape(length, self.heads * self.head_dim)
        return self._lin(output, prefix + "o_proj.weight")

    def _layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        w = self.weights
        base = f"layers.{i}."
        residual = x
        normed = _norm(x, w[base + "input_layernorm.weight"], self.eps)
        cbase = base + "attention_conv."
        dyn = self._lin(normed, cbase + "kernel_projection.weight")
        dyn = dyn.view(len(x), 2, 2, self.hidden // self.group_size)
        x = _conv(normed, dyn, w[cbase + "base_kernel"], 0, self.group_size)
        x = (residual.float() + _conv(self._attention(i, x), dyn,
                                     w[cbase + "base_kernel"], 1, self.group_size).float()).to(torch.bfloat16)
        residual = x
        normed = _norm(x, w[base + "post_attention_layernorm.weight"], self.eps)
        cbase = base + "mlp_conv."
        dyn = self._lin(normed, cbase + "kernel_projection.weight")
        dyn = dyn.view(len(x), 2, 2, self.hidden // self.group_size)
        x = _conv(normed, dyn, w[cbase + "base_kernel"], 0, self.group_size)
        mlp = self._lin(F.silu(self._lin(x, base + "mlp.gate_proj.weight"))
                        * self._lin(x, base + "mlp.up_proj.weight"), base + "mlp.down_proj.weight")
        return (residual.float() + _conv(mlp, dyn, w[cbase + "base_kernel"],
                                         1, self.group_size).float()).to(torch.bfloat16)

    def _rotary(self, start: int, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        phase = torch.arange(start, start + rows, device=self.device, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        return phase.cos().contiguous(), phase.sin().contiguous()

    def _prep(self, qkv: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor, heads: int):
        rows = qkv.shape[0]
        prefix = f"layers.{layer}.self_attn."
        d = self.head_dim
        q = torch.empty((heads, rows, d), dtype=torch.bfloat16, device=self.device) if heads else qkv
        k = torch.empty((self.kv_local, rows, d), dtype=torch.bfloat16, device=self.device)
        v = torch.empty_like(k)
        _prep_kernel[(rows, heads + 2 * self.kv_local)](qkv, self.weights[prefix + "q_norm.weight"],
                                                        self.weights[prefix + "k_norm.weight"], cos, sin, q, k, v,
                                                        rows, qkv.stride(0), self.eps, H=heads, HKV=self.kv_local,
                                                        HALF=d // 2, num_warps=1)
        return q, k, v

    def _layer_fast(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        w = self.weights
        base = f"layers.{i}."
        rows = x.shape[0]
        normed = F.rms_norm(x, (self.hidden,), w[base + "input_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "attention_conv.kernel_projection.weight")
        conv = w[base + "attention_conv.base_kernel"]
        q, k, v = self._prep(self._lin(_dconv(normed, dyn, conv, 0, self.group_size),
                                       base + "self_attn.qkv.weight"), i, cos, sin, self.heads_local)
        keys = torch.cat((self.kc[i], k), dim=1)
        values = torch.cat((self.vc[i], v), dim=1)
        out = F.scaled_dot_product_attention(q[None], keys[None], values[None], attn_mask=mask[None, None],
                                             scale=self.head_dim ** -0.5, enable_gqa=True)
        out = out[0].transpose(0, 1).reshape(rows, self.heads_local * self.head_dim)
        x = _dconv(self._row(out, base + "self_attn.o_proj.weight"), dyn, conv, 1, self.group_size, x)
        normed = F.rms_norm(x, (self.hidden,), w[base + "post_attention_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "mlp_conv.kernel_projection.weight")
        conv = w[base + "mlp_conv.base_kernel"]
        h = _dconv(normed, dyn, conv, 0, self.group_size)
        xs = group_sums(h)
        act, act_xs = swiglu(matmul(h, self.q4[base + "mlp.gate_proj.weight"], xs),
                             matmul(h, self.q4[base + "mlp.up_proj.weight"], xs))
        mlp = self._row(act, base + "mlp.down_proj.weight", act_xs)
        return _dconv(mlp, dyn, conv, 1, self.group_size, x)

    def in_vocab(self, token: int) -> bool:
        """Whether the drafter's head can propose ``token`` at all."""

        return any(a <= token < b for a, b in self.vocab_spans)

    def _row(self, x: torch.Tensor, name: str, xs: torch.Tensor | None = None) -> torch.Tensor:
        """A projection whose input is split over the ranks: rank partials summed in rank order."""

        if self.world == 1:
            return matmul(x.contiguous(), self.q4[name], xs)
        from .distributed import gather_rank_partials, row_partial

        return gather_rank_partials(row_partial(x.contiguous(), self.q4[name], xs=xs))

    @torch.no_grad()
    def propose_tree(self, pending: int, context_length: int, max_nodes: int,
                     sampling: Sampling | None = None, block: int | None = None) -> tuple[list[int], list[int]]:
        if self.context_len == 0 or max_nodes < 1:
            return [], []
        length = min(block or self.block, max_nodes + 1)
        tokens = torch.tensor([pending] + [self.mask_id] * (length - 1),
                              dtype=torch.int32, device=self.device)
        x = embed(tokens, self.target.embed.weight, self.target.embed.scales,
                  self.target.embed.biases, self.hidden)
        if self.fast:
            s = self.kc[0].shape[1]
            qidx = torch.arange(length, device=self.device)[:, None]
            kidx = torch.arange(s + length, device=self.device)[None, :]
            block_allowed = kidx >= s
            if self.is_causal:
                block_allowed = block_allowed & (kidx <= s + qidx)
            mask = ((kidx < s) & (s + qidx - kidx < self.window + 1)) | block_allowed
            cos, sin = self._rotary(self.context_end, length)
            for i in range(self.layers):
                x = self._layer_fast(i, x, cos, sin, mask)
            h = F.rms_norm(x[1:], (self.hidden,), self.weights["norm.weight"], self.eps)
        else:
            for i in range(self.layers):
                x = self._layer(i, x)
            h = _norm(x[1:], self.weights["norm.weight"], self.eps)
        projected = self._lin(h, "candidate_selector.hidden_projection.weight").float()
        logits = matmul(h, self.sub_head)
        values, local_ids = torch.topk(logits.float(), k=16, dim=-1, sorted=False)
        global_ids = self.head_ids[local_ids]
        if self.world == 2:
            import torch.distributed as dist

            both_values = torch.empty((2, *values.shape), dtype=values.dtype, device=values.device)
            both_ids = torch.empty((2, *global_ids.shape), dtype=global_ids.dtype, device=values.device)
            dist.all_gather_into_tensor(both_values, values.contiguous())
            dist.all_gather_into_tensor(both_ids, global_ids.contiguous())
            merged = torch.cat((both_values[0], both_values[1]), dim=1)
            values, pick = torch.topk(merged, k=16, dim=-1, sorted=False)
            global_ids = torch.cat((both_ids[0], both_ids[1]), dim=1).gather(1, pick)
        ids = global_ids.cpu().numpy().astype(np.int64)
        self.last_candidates = ids                       # (depths, 16) ids this round, for decode traces
        unary = values.cpu().numpy().astype(np.float64)
        hproj = projected.cpu().numpy().astype(np.float64)
        return _best_first(ids, unary, hproj,
                           self.weights["candidate_selector.predecessor_codebook"],
                           self.weights["candidate_selector.successor_codebook"],
                           int(pending), min(127, max_nodes), sampling, context_length)
