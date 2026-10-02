"""DFlash2 proposals use committed Qwen3.8 target taps and target verification, so draft rounding changes acceptance without changing output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.engine.exact_sampling import Sampling

from .affine_memory import packed_draft
from .draft_tree import best_first
from .glue import embedding, swiglu
from .draft_attention import append, block_attention
from .qmm import group_sums
from .qmm_fast import matmul, matmul_group, matmul_rows, rows, tile, untile
from .weights import Exl3, Plain, QLinear, Weights


@triton.jit
def _dconv_kernel(X, DYN, BASE, RES, OUT, SEG, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                  BRANCH: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """``_conv`` and its residual add as one kernel; row r mixes rows r and r-1 of its ``SEG``-row block."""

    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < D
    x = tl.load(X + row * D + c, mask=ok, other=0.0).to(tl.float32)
    prev = tl.load(X + (row - 1) * D + c, mask=ok & (row % SEG > 0), other=0.0).to(tl.float32)
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
    """Normalize and rotate projected q/k while copying v into (heads, rows, head_dim) outputs, one program per row and head."""

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
           residual: torch.Tensor | None = None, seg: int | None = None) -> torch.Tensor:
    rows, d = x.shape
    x = x.contiguous()
    dyn = dyn.contiguous()
    out = torch.empty_like(x)
    block = 1024
    _dconv_kernel[(rows, triton.cdiv(d, block))](x, dyn, base, residual if residual is not None else x, out,
                                                 seg or rows, D=d, G=d // group_size, GS=group_size, BRANCH=branch,
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


def _sub_parts(head, spans: tuple[tuple[int, int], ...]) -> list[tuple[object, int, int]]:
    """An NVFP4 checkpoint's head rows for ``spans``: views of whole 64-row tiles (bf16 rows as they are), each span's columns."""

    parts = []
    for a, b in spans:
        if isinstance(head, Plain):
            parts.append((Plain(head.weight[a:b]), 0, b - a))
        else:
            t0, t1 = a // 64, -(-b // 64)
            parts.append((head.tiles(t0, t1), a - 64 * t0, b - 64 * t0))
    return parts


def _exl3_sub_head(layer, spans: tuple[tuple[int, int], ...]):
    """The EXL3 head's strips holding ``spans`` as stored (the target's own logits, bit for bit), and the span columns."""

    from tensorfold.cuda.exl3.linear import Exl3Linear

    if layer.layout != "strips":
        raise ValueError("the drafter slices an EXL3 head in strip order")
    device = layer.words.device
    blocks = sorted({b for a, e in spans for b in range(a // 128, -(-e // 128))})
    index = torch.tensor(blocks, dtype=torch.int64, device=device)
    columns = (index[:, None] * 128 + torch.arange(128, device=device)[None, :]).reshape(-1)
    sub = Exl3Linear(layer.words.index_select(0, index).contiguous(), layer.suh,
                     layer.svh.index_select(0, columns).contiguous(),
                     None if layer.bias is None else layer.bias.index_select(0, columns).contiguous(),
                     layer.bits, layer.codebook, layer.k, len(blocks) * 128, "strips", split=layer.split)
    keep = torch.zeros(len(columns), dtype=torch.bool, device=device)
    for a, e in spans:
        keep |= (columns >= a) & (columns < e)
    return sub, keep.nonzero()[:, 0].contiguous()


class DFlash2:
    """Five-layer DFlash2; context is the last 2,047 committed target taps."""

    def __init__(self, draft_dir: str | Path, target: Weights, bits: int = 4, block: int = 16,
                 fast: bool = True, rank: int = 0, world: int = 1):
        path = Path(draft_dir)
        # Two-rank drafting requires both ranks to call propose_tree/add_taps together on the fused path with ordered partial sums.
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
        self.trained = int(cfg["dflash_config"].get("block_size", 8))     # its training block: the planner's floor
        self.group_size = int(cfg["dflash_config"]["conv_group_size"])
        self.layers = int(cfg["num_hidden_layers"])
        self.window = int(cfg["sliding_window"]) - 1
        self.is_causal = bool(cfg.get("is_causal", True))
        self.target_embed = target.embed
        self.device = target.norm.device
        self.weights: dict[str, torch.Tensor] = {}
        f = SafeTensors([path / "model.safetensors"])
        for name in f.keys():
            tensor = f.get(name)
            if name in ("candidate_selector.predecessor_codebook",
                        "candidate_selector.successor_codebook"):
                self.weights[name] = tensor.float().numpy().copy()
            else:
                self.weights[name] = tensor                  # on the host until packed: no bf16 copy on the device
        del f
        self.inv_freq = (1.0 / self.theta **
                         (torch.arange(self.head_dim // 2, device=self.device,
                                       dtype=torch.float32) * 2 / self.head_dim))
        vocab = target.config.vocab
        spans = tuple((a, min(b, vocab)) for a, b in ((0, 98304), (248032, 248320)) if a < vocab)
        self.vocab_spans = spans
        self.head_ids = torch.cat([torch.arange(a, b, device=self.device) for a, b in spans])
        self.head_cols: torch.Tensor | None = None           # an EXL3 head's span columns in its sliced strips
        self.sub_rows: list[QLinear] | None = None           # one GPU, tiled head: its rows as views, no copy
        self.sub_parts: list | None = None                   # an NVFP4 checkpoint's head: tile views and span columns
        if target.quant == "nvfp4":
            if world != 1:
                raise ValueError("a two-rank drafter needs the MLX checkpoint's 4-bit head")
            self.sub_parts = _sub_parts(target.head, spans)
        elif isinstance(target.head, Exl3):
            if world != 1:
                raise ValueError("a two-rank drafter needs the MLX checkpoint's 4-bit head")
            sub, self.head_cols = _exl3_sub_head(target.head.layer, spans)
            self.sub_head = Exl3(sub)
        elif world == 1 and target.head.layout == "tiled":
            self.sub_rows = [rows(target.head, a, b) for a, b in spans]
        else:
            head = untile(target.head)
            parts = [None if t is None else torch.cat([t[a:b] for a, b in spans]).contiguous()
                     for t in (head.weight, head.scales, head.biases)]
            self.sub_head = QLinear(*parts, layout=head.layout, gs=head.gs, bits=head.bits)
            del head
        if world == 2:
            half = -(-len(self.head_ids) // 2)
            lo, hi = rank * half, min((rank + 1) * half, len(self.head_ids))
            self.head_ids = self.head_ids[lo:hi].contiguous()
            sub = self.sub_head
            self.sub_head = QLinear(*[None if t is None else t[lo:hi].contiguous()
                                      for t in (sub.weight, sub.scales, sub.biases)],
                                    layout=sub.layout, gs=sub.gs, bits=sub.bits)
        if isinstance(target.head, QLinear) and target.head.layout == "tiled" and self.sub_rows is None:
            self.sub_head = tile(self.sub_head)
        # Quantized draft projections can change acceptance but never target output.
        self.q4: dict[str, QLinear] = {}
        # Fused projections, norms, rotary embeddings, and convolutions may round differently and change draft proposals.
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
                if isinstance(t, torch.Tensor) and packed_draft(name, t.shape):    # as admission counts it
                    self.q4[name] = tile(quantize4(t.to(self.device, torch.bfloat16)))
                    del self.weights[name]
        for name, t in self.weights.items():
            if isinstance(t, torch.Tensor):
                self.weights[name] = t.to(self.device)
        torch.cuda.empty_cache()
        # Context keys and values depend only on their row and position, so project each once on arrival.
        self.kc: list[torch.Tensor | None] = [None] * self.layers
        self.vc: list[torch.Tensor | None] = [None] * self.layers
        self.context_len = 0
        self.context_end = 0

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        q = self.q4.get(name)
        if q is None:
            return F.linear(x, self.weights[name])
        return matmul(x.to(torch.bfloat16).contiguous(), q)

    def snapshot(self):
        return (list(self.kc), list(self.vc), self.context_len, self.context_end)

    def restore(self, snap) -> None:
        kc, vc, self.context_len, self.context_end = snap
        self.kc, self.vc = list(kc), list(vc)

    def skip(self, n: int) -> None:
        """Skip ``n`` rows untapped; the caller adds at least ``window`` more before drafting, so they drop out."""

        self.kc, self.vc = [None] * self.layers, [None] * self.layers
        self.context_len, self.context_end = 0, self.context_end + n

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

    @torch.no_grad()
    def add_taps_streams(self, snaps: list, taps: list[torch.Tensor]) -> list:
        """``add_taps`` for several streams, each projection run once over all their rows; returns the new contexts (each snapshot's own lists updated in place, so a layer's old window goes as its new one lands)."""

        if not self.fast:
            out = []
            for snap, t in zip(snaps, taps):
                self.restore(snap)
                self.add_taps(t)
                out.append(self.snapshot())
            return out
        sizes = [t.shape[0] for t in taps]
        projected = _norm(self._lin(torch.cat(taps) if len(taps) > 1 else taps[0], "fc.weight"),
                          self.weights["hidden_norm.weight"], self.eps)
        pos = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                           dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        phase = pos[:, None] * self.inv_freq[None, :]
        cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
        kcs, vcs = [snap[0] for snap in snaps], [snap[1] for snap in snaps]
        for layer in range(self.layers):
            _, k, v = self._prep(self._lin(projected, f"layers.{layer}.self_attn.kv.weight"), layer, cos, sin, 0)
            for cache, fresh in ((kcs, k), (vcs, v)):        # every stream's window in one copy a tensor
                for c, out in zip(cache, append(fresh, [c[layer] for c in cache], sizes, self.window)):
                    c[layer] = out
        return [(kc, vc, min(self.window, snap[2] + n), snap[3] + n) for kc, vc, snap, n in zip(kcs, vcs, snaps, sizes)]

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

    def _layer_fast(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: list,
                    length: int) -> torch.Tensor:
        """One layer over several streams' blocks (``length`` rows each); each block attends its own context."""

        w = self.weights
        base = f"layers.{i}."
        normed = F.rms_norm(x, (self.hidden,), w[base + "input_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "attention_conv.kernel_projection.weight")
        conv = w[base + "attention_conv.base_kernel"]
        q, k, v = self._prep(self._lin(_dconv(normed, dyn, conv, 0, self.group_size, seg=length),
                                       base + "self_attn.qkv.weight"), i, cos, sin, self.heads_local)
        out = block_attention(q, k, v, [snap[0][i] for snap in ctx], [snap[1][i] for snap in ctx], length,
                              self.window, self.head_dim ** -0.5, self.is_causal)
        x = _dconv(self._row(out, base + "self_attn.o_proj.weight"), dyn, conv, 1, self.group_size, x, seg=length)
        normed = F.rms_norm(x, (self.hidden,), w[base + "post_attention_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "mlp_conv.kernel_projection.weight")
        conv = w[base + "mlp_conv.base_kernel"]
        h = _dconv(normed, dyn, conv, 0, self.group_size, seg=length)
        xs = group_sums(h)
        act, act_xs = swiglu(*matmul_group(h, [self.q4[base + "mlp.gate_proj.weight"],
                                               self.q4[base + "mlp.up_proj.weight"]], xs))   # one launch, each its bits
        mlp = self._row(act, base + "mlp.down_proj.weight", act_xs)
        return _dconv(mlp, dyn, conv, 1, self.group_size, x, seg=length)

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
    def launch_block(self, pending: int, max_nodes: int, block: int | None = None):
        """``launch_blocks`` for the current context alone."""

        return self.launch_blocks([self.snapshot()], [pending], max_nodes, block)[0]

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """GPU half of ``propose_tree`` for several streams, each weight read once; None where a context is empty."""

        out: list = [None] * len(snaps)
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live or max_nodes < 1:
            return out
        length = min(block or self.block, max_nodes + 1)
        tokens = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                              dtype=torch.int32, device=self.device)
        x = embedding(tokens, self.target_embed)
        if self.fast:
            ctx = [snaps[i] for i in live]
            rot = [self._rotary(snap[3], length) for snap in ctx]
            cos, sin = torch.cat([c for c, _ in rot]), torch.cat([s for _, s in rot])
            for layer in range(self.layers):
                x = self._layer_fast(layer, x, cos, sin, ctx, length)
            h = F.rms_norm(x.view(len(live), length, -1)[:, 1:].reshape(-1, self.hidden), (self.hidden,),
                           self.weights["norm.weight"], self.eps)
        else:                                    # the reference path, one stream at a time
            hs = []
            for j, i in enumerate(live):
                self.restore(snaps[i])
                y = x[j * length:(j + 1) * length]
                for layer in range(self.layers):
                    y = self._layer(layer, y)
                hs.append(_norm(y[1:], self.weights["norm.weight"], self.eps))
            h = torch.cat(hs)
        projected = self._lin(h, "candidate_selector.hidden_projection.weight").float()
        if self.sub_parts is not None:
            h = h.contiguous()
            logits = torch.cat([part(h)[:, lo:hi] for part, lo, hi in self.sub_parts], dim=1)
        elif self.head_cols is not None:
            logits = self.sub_head(h.contiguous()).index_select(1, self.head_cols)
        elif self.sub_rows is not None:
            logits = matmul_rows(h, self.sub_rows)
        else:
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
        shared = [global_ids, torch.cat((values, projected), dim=1), None]     # read back once, by the first finish
        for j, i in enumerate(live):
            out[i] = (shared, j * (length - 1), length - 1, int(pendings[i]))
        return out

    def finish_tree(self, launched, context_length: int, max_nodes: int,
                    sampling: Sampling | None = None) -> tuple[list[int], list[int], list[float]]:
        """The host half: the tree policy over a launched block's candidates; nodes, parents and scores in pop order."""

        if launched is None:
            return [], [], []
        shared, start, rows, pending = launched
        if shared[2] is None:
            shared[2] = (shared[0].cpu().numpy().astype(np.int64), shared[1].cpu().numpy().astype(np.float64))
        ids, floats = shared[2][0][start:start + rows], shared[2][1][start:start + rows]
        self.last_candidates = ids                       # (depths, 16) ids this round, for decode traces
        return best_first(ids, floats[:, :16], floats[:, 16:],
                          self.weights["candidate_selector.predecessor_codebook"],
                          self.weights["candidate_selector.successor_codebook"],
                          pending, min(127, max_nodes), sampling, context_length)

    @torch.no_grad()
    def propose_tree(self, pending: int, context_length: int, max_nodes: int,
                     sampling: Sampling | None = None, block: int | None = None) -> tuple[list[int], list[int]]:
        return self.finish_tree(self.launch_block(pending, max_nodes, block), context_length, max_nodes, sampling)[:2]
