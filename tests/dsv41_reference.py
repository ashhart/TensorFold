"""A plain-torch port of DeepSeek-V4.1's MIT ``inference/model.py``: the comparator for the CUDA family.

``M:n`` and ``K:n`` cite lines of deepseek-ai/DeepSeek-V4.1-Flash's (revision dba1be0) ``inference/model.py`` and
``inference/kernel.py``. Every chunk follows the per-position rules for any start position: window
``[max(0, i-127), i]``, compressed entry ``j`` visible once ``j < (i+1)//ratio`` and rotated at ``j*ratio``, groups
``{2j, 2j+1}`` carried across forwards. The ``start_pos > 0`` branches are not ported, and an indexer always scores its
own KV source's index-K, where model.py's decode path reads whichever owner wrote last (M:537-554). Mode ``fp32`` rounds
only where the model quantizes; ``mirror`` also rounds to bf16 (and fp16 at the EXL3 experts) where the CUDA family
stores values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce

import torch
import torch.nn.functional as F

from tensorfold.families.deepseek_v41.config import Config, yarn_bounds

INF = float("inf")
_INV448 = torch.tensor(1 / 448, dtype=torch.float32)
_INV6 = torch.tensor(1 / 6, dtype=torch.float32)


@dataclass(frozen=True)
class Mode:
    """Where values are rounded, whether FP8/FP4 linears quantize their input, how many ranks split row sums."""

    name: str = "fp32"          # fp32 or mirror
    act_quant: bool = False     # model.py's FP8 activation quantization in front of FP8/FP4 weights (M:181-207)
    world: int = 2              # row-parallel partial sums, added in rank order

    def __post_init__(self) -> None:
        if self.name not in ("fp32", "mirror"):
            raise ValueError(f"mode {self.name!r}: fp32 or mirror")

    def bf16(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).float() if self.name == "mirror" else x

    def fp16(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float16).float() if self.name == "mirror" else x


FP32, MIRROR = Mode(), Mode("mirror")


def _pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """K:22-33: 2^ceil(log2 t) from t's fp32 bits, the exponent plus one when the mantissa is non-zero."""

    bits = t.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """K:70-86 in place: FP8 per 32 with s = 2^ceil(log2(max(amax, 1e-4) * f32(1/448))), e4m3(clamp(x/s)) * s."""

    g = x.float().unflatten(-1, (-1, 32))
    s = _pow2_ceil(g.abs().amax(-1, keepdim=True).clamp_min(1e-4) * _INV448)
    q = (g / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).flatten(-2).to(x.dtype)


def e2m1(v: torch.Tensor) -> torch.Tensor:
    """The nearest e2m1 value with ties to even for |v| <= 6 (K:171's FP4 cast): grid steps 0.5, 1, 2."""

    a = v.abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    return torch.copysign(torch.round(a / step) * step, v)


def fp4_act_quant(x: torch.Tensor, block: int) -> torch.Tensor:
    """K:155-172 in place: block 16 scales by e4m3(max(amax, 6*2^-9) / 6), block 32 by a power of two."""

    g = x.float().unflatten(-1, (-1, block))
    amax = g.abs().amax(-1, keepdim=True)
    if block == 16:
        s = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2**-126) * _INV6)
    return (e2m1((g / s).clamp(-6, 6)) * s).flatten(-2).to(x.dtype)


def linear(x: torch.Tensor, w: torch.Tensor, mode: Mode, fp8: bool = True) -> torch.Tensor:
    """x @ w.T in fp32; under ``act_quant`` an FP8/FP4 weight's input is FP8-quantized first (M:181-207)."""

    return (act_quant(x) if fp8 and mode.act_quant else x) @ w.T


def row_parallel(x: torch.Tensor, w: torch.Tensor, mode: Mode, fp8: bool = True) -> torch.Tensor:
    """M:260-278: fp32 partials over ``mode.world`` contiguous slices of the input dim, added in rank order."""

    c = x.shape[-1] // mode.world
    return reduce(torch.add, [linear(x[..., r * c:(r + 1) * c], w[:, r * c:(r + 1) * c], mode, fp8)
                              for r in range(mode.world)])


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, mode: Mode) -> torch.Tensor:
    """M:288-293: w * (x * rsqrt(mean(x^2) + eps)) in fp32, rounded once."""

    return mode.bf16(w.float() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)))


def freqs_cis(dim: int, seqlen: int, original: int, base: float, factor: float, beta_fast: float,
              beta_slow: float) -> torch.Tensor:
    """M:368-389: complex64 [seqlen, dim/2] rotations, YaRN-blended when ``original`` > 0."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original > 0:
        low, high = yarn_bounds(dim, original, base, beta_fast, beta_slow)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope(x: torch.Tensor, cis: torch.Tensor, mode: Mode, inverse: bool = False) -> torch.Tensor:
    """M:392-406 on the last 2*cis.shape[-1] dims: adjacent pairs as complex numbers; ``inverse`` conjugates."""

    rd = 2 * cis.shape[-1]
    c = (cis.conj() if inverse else cis).reshape(cis.shape[0], *[1] * (x.dim() - 2), cis.shape[-1])
    z = torch.view_as_complex(x[..., -rd:].float().contiguous().unflatten(-1, (-1, 2)))
    out = x.clone()
    out[..., -rd:] = mode.bf16(torch.view_as_real(z * c).flatten(-2))
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K:426-458: pre = sigmoid + eps, post = 2 sigmoid, comb = row softmax + eps then column/row normalizations."""

    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(X: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    """M:952-954: the flattened copies projected by ``fn``, scaled by one rsqrt(mean square) per row."""

    xf = X.flatten(-2)
    return (xf @ fn.T) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


def hc_pre(X: torch.Tensor, pre: torch.Tensor, mode: Mode) -> torch.Tensor:
    """M:957-960: sum_j pre[j] * X[j] over the copies in order, rounded once."""

    return mode.bf16(reduce(torch.add, [pre[:, j, None] * X[:, j] for j in range(X.shape[1])]))


def hc_post(branch: torch.Tensor, X: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
            mode: Mode) -> torch.Tensor:
    """M:962-966: copy k = post[k] * branch + sum_j comb[j, k] * X[j], rounded once."""

    mix = reduce(torch.add, [comb[:, j, :, None] * X[:, j, None] for j in range(X.shape[1])])
    return mode.bf16(post[..., None] * branch[:, None] + mix)


def select_candidate_blocks(scores: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block: int) -> torch.Tensor:
    """M:583-610 per row: top blocks by their best score, the newest visible block pinned, ties to the lower block."""

    width = scores.shape[-1]
    b = F.pad(scores, (0, -width % block), value=-INF).unflatten(-1, (-1, block)).amax(-1)
    b = b.masked_fill(torch.arange(b.shape[-1]) == ((visible - 1) // block)[:, None], INF)
    top = torch.sort(b, dim=-1, descending=True, stable=True)
    k = min(topk_blocks, b.shape[-1])
    keep = torch.zeros_like(b, dtype=torch.bool).scatter_(-1, top.indices[:, :k], top.values[:, :k] > -INF)
    return keep.repeat_interleave(block, dim=-1)[..., :width]


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, scale: float, mode: Mode) -> torch.Tensor:
    """K:355-387 for one row of heads: softmax over ``kv`` whose denominator also takes exp(sink - max).

    ``mirror`` rounds P to bf16 for P.V only; the denominator sums the fp32 P (K:374-380).
    """

    if not len(kv):
        return torch.zeros_like(q)
    s = (q @ kv.T) * scale
    m = s.amax(-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(s - m)
    den = p.sum(-1, keepdim=True) + torch.exp(sink[:, None] - m)
    return mode.bf16((mode.bf16(p) @ kv) / den)


def gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, k: int, cfg: Config) -> tuple[torch.Tensor, ...]:
    """M:809-827: sqrt-softplus scores, top-k of score + bias (ties to the lower id) weighted by the raw scores."""

    s = F.softplus((x @ weight.T) / cfg.gate_temp).sqrt()
    idx = torch.sort(s + bias, dim=-1, descending=True, stable=True).indices[:, :k]
    w = s.gather(1, idx)
    if cfg.norm_topk_prob and k > 1:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * cfg.routed_scaling_factor, idx


def swiglu(g: torch.Tensor, u: torch.Tensor, limit: float) -> torch.Tensor:
    """M:845-848: silu(min(g, limit)) * clamp(u, -limit, limit)."""

    if limit > 0:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    return F.silu(g) * u


def _one_hot(rows: int, hc: int) -> torch.Tensor:
    pre = torch.zeros(rows, hc)
    pre[:, 0] = 1.0
    return pre


@dataclass
class State:
    """One sequence's committed caches, addressed by absolute position, and the forward's shared slots."""

    pos: int = 0                                                # positions below are committed
    tokens: list[int] = field(default_factory=list)
    ring: dict[int, torch.Tensor] = field(default_factory=dict)     # layer -> [window, head_dim], slot pos % window
    comp: dict[int, torch.Tensor] = field(default_factory=dict)     # KV source -> compressed entries [j, head_dim]
    index_k: dict[int, torch.Tensor] = field(default_factory=dict)  # KV source -> its own index keys [j, dim]
    tail: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)  # (kv, score) of an open group
    lists: list[torch.Tensor] = field(default_factory=list)     # the latest indexer's entries per row (M:1166-1180)
    cand: torch.Tensor | None = None                            # the candidate source's per-row entry mask


class Reference:
    """model.py's forward from fp32 tensors by checkpoint name; routed experts and Engram rows from callables."""

    def __init__(self, cfg: Config, weight: Callable[[str], torch.Tensor], *,
                 expert: Callable[[str, int], tuple[torch.Tensor, ...]] | None = None,
                 engram_rows: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
                 hasher=None, mode: Mode = FP32) -> None:
        if cfg.scoring_func != "sqrtsoftplus":
            raise ValueError(f"scoring_func {cfg.scoring_func!r}: the reference ports sqrtsoftplus only")
        self.cfg, self.weight, self.mode = cfg, weight, mode
        self.expert = expert or (lambda prefix, e: tuple(weight(f"{prefix}.ffn.experts.{e}.{n}.weight")
                                                        for n in ("w1", "w2", "w3")))
        self.engram_rows, self.hasher = engram_rows, hasher   # (layer, ids) -> rows; engram_hash's Hasher
        self.state = State()
        self._tables: dict[str, torch.Tensor] = {}

    def W(self, name: str) -> torch.Tensor:
        return self.weight(name).float()

    def _prefix(self, layer: int) -> str:
        n = self.cfg.num_hidden_layers
        return f"layers.{layer}" if layer < n else f"mtp.{layer - n}"

    def cis(self, kind: str, positions: torch.Tensor) -> torch.Tensor:
        """Rows of the layer kind's table (``local``: rope_theta; ``yarn``: compress_rope_theta with YaRN)."""

        c, need = self.cfg, int(positions.max()) + 1
        if len(self._tables.get(kind, ())) < need:
            original, base = (c.original_max_position_embeddings, c.compress_rope_theta) if kind == "yarn" else \
                (0, c.rope_theta)
            self._tables[kind] = freqs_cis(c.qk_rope_head_dim, 1 << (need - 1).bit_length(), original, base,
                                           c.rope_factor, c.beta_fast, c.beta_slow)
        return self._tables[kind][positions]

    def _rows(self, start: int, n: int) -> torch.Tensor:
        return torch.arange(start, start + n)

    def _kv(self, p: str, x: torch.Tensor, cis: torch.Tensor) -> torch.Tensor:
        """M:700-707: one latent per row, normed, rotated and FP8-quantized; it is both K and V."""

        m, eps = self.mode, self.cfg.rms_norm_eps
        kv = rms_norm(m.bf16(linear(x, self.W(f"{p}.attn.wkv.weight"), m)), self.W(f"{p}.attn.kv_norm.weight"), eps, m)
        return act_quant(rope(kv, cis, m))

    def _q(self, p: str, x: torch.Tensor, cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """M:770-772: (qr, q) with q's last rope dims rotated."""

        c, m = self.cfg, self.mode
        qr = rms_norm(m.bf16(linear(x, self.W(f"{p}.attn.wq_a.weight"), m)), self.W(f"{p}.attn.q_norm.weight"),
                      c.rms_norm_eps, m)
        q = m.bf16(linear(qr, self.W(f"{p}.attn.wq_b.weight"), m)).unflatten(-1, (c.num_attention_heads, c.head_dim))
        return qr, rope(q, cis, m)

    def _window(self, layer: int, kv: torch.Tensor, start: int, anchor: int) -> torch.Tensor:
        """Window entries max(0, a-127)..a ascending: committed ones from the ring, this forward's from ``kv``."""

        win = self.cfg.sliding_window
        ring = self.state.ring.setdefault(layer, torch.zeros(win, self.cfg.head_dim))
        lo = max(0, anchor - win + 1)
        old = ring[[w % win for w in range(lo, min(start, anchor + 1))]]
        return torch.cat([old, kv[max(lo, start) - start:max(anchor + 1 - start, 0)]])

    def _commit(self, layer: int, kv: torch.Tensor, start: int) -> None:
        """The ring takes the last ``window`` rows at their slots pos % window (M:708-719)."""

        win = self.cfg.sliding_window
        ring = self.state.ring.setdefault(layer, torch.zeros(win, self.cfg.head_dim))
        lo = max(0, len(kv) - win)
        ring[(start + torch.arange(lo, len(kv))) % win] = kv[lo:]

    def _out(self, p: str, o: torch.Tensor) -> torch.Tensor:
        """M:783-789: block-diagonal wo_a per group of heads, then wo_b summed over the row-parallel ranks."""

        c, m = self.cfg, self.mode
        wo_a = self.W(f"{p}.attn.wo_a.weight").view(c.o_groups, c.o_lora_rank, -1)
        u = m.bf16(torch.einsum("sgd,grd->sgr", o.flatten(1).unflatten(-1, (c.o_groups, -1)), wo_a))
        return m.bf16(row_parallel(u.flatten(1), self.W(f"{p}.attn.wo_b.weight"), m))

    def compress(self, layer: int, x: torch.Tensor, start: int) -> None:
        """M:458-485 for a KV source, groups possibly straddling forwards; then index-K and entries (M:537-547, 751-761)."""

        c, m, st, role, p = self.cfg, self.mode, self.state, self.cfg.roles[layer], self._prefix(layer)
        r = role.ratio
        kv = linear(x, self.W(f"{p}.attn.compressor.wkv.weight"), m, fp8=False)
        if r == 1:
            first, pooled = start, kv
        else:
            z = linear(x, self.W(f"{p}.attn.compressor.wgate.weight"), m, fp8=False)
            tk, tz = st.tail.pop(layer, (kv[:0], z[:0]))
            kv, z, first = torch.cat([tk, kv]), torch.cat([tz, z]), start - len(tk)
            done = ((start + len(x)) // r) * r - first
            groups = (done // r, r)
            pooled = (kv[:done].unflatten(0, groups) * z[:done].unflatten(0, groups).softmax(1)).sum(1)
            if done < len(kv):
                st.tail[layer] = (kv[done:], z[done:])
        if not len(pooled):
            return
        j0 = first // r
        lat = rms_norm(m.bf16(pooled), self.W(f"{p}.attn.compressor.norm.weight"), c.rms_norm_eps, m)
        cis = self.cis(role.rope, self._rows(j0, len(lat)) * r)
        k = rms_norm(m.bf16(linear(lat, self.W(f"{p}.attn.indexer.wk.weight"), m, fp8=False)),
                     self.W(f"{p}.attn.indexer.k_norm.weight"), c.rms_norm_eps, m)
        for cache, new in ((st.index_k, fp4_act_quant(rope(k, cis, m), 32)),
                           (st.comp, fp4_act_quant(rope(lat, cis, m), 16))):
            old = cache.get(layer, new[:0])
            if len(old) != j0:
                raise ValueError(f"layer {layer}: entry {j0} follows {len(old)} committed entries")
            cache[layer] = torch.cat([old, new])

    def indexer(self, layer: int, x: torch.Tensor, qr: torch.Tensor, start: int) -> list[torch.Tensor]:
        """M:550-580 per row against its KV source's own index-K (never M:554's alias), candidates per M:569-575."""

        c, m, st, role, p = self.cfg, self.mode, self.state, self.cfg.roles[layer], self._prefix(layer)
        keys, rows = st.index_k.get(role.kv_src), self._rows(start, len(x))
        if keys is None or not len(keys):
            return [torch.zeros(0, dtype=torch.long) for _ in range(len(x))]
        H, D = c.index_n_heads, c.index_head_dim
        q = m.bf16(linear(qr, self.W(f"{p}.attn.indexer.wq_b.weight"), m)).unflatten(-1, (H, D))
        q = fp4_act_quant(rope(q, self.cis(role.rope, rows), m), 32)
        w = m.bf16(linear(x, self.W(f"{p}.attn.indexer.weights_proj.weight"), m, fp8=False)) * (D**-0.5 * H**-0.5)
        score = (torch.einsum("shd,td->sht", q, keys).relu() * w[..., None]).sum(1)
        visible = (rows + 1) // role.ratio
        score = score.masked_fill(torch.arange(len(keys)) >= visible[:, None], -INF)
        if role.candidate_source:
            st.cand = select_candidate_blocks(score, visible, c.candidate_topk_blocks, c.candidate_block_size)
        elif role.uses_candidates:
            score = score.masked_fill(~st.cand, -INF)
        order = torch.sort(score, dim=-1, descending=True, stable=True).indices
        return [order[i, :min(c.index_topk, int(visible[i]))].sort().values for i in range(len(x))]

    def attention(self, layer: int, x: torch.Tensor, start: int) -> torch.Tensor:
        """M:765-789 for rows at ``start``..: window plus the source's compressed entries per row, then out."""

        c, m, st, role, p = self.cfg, self.mode, self.state, self.cfg.roles[layer], self._prefix(layer)
        cis = self.cis(role.rope, self._rows(start, len(x)))
        qr, q = self._q(p, x, cis)
        kv = self._kv(p, x, cis)
        extra = None
        if role.ratio:
            if role.kv_src == layer:
                self.compress(layer, x, start)
            if role.idx_src == layer:
                st.lists = self.indexer(layer, x, qr, start)
            extra = st.comp.get(role.kv_src)
        sink, rows = self.W(f"{p}.attn.attn_sink"), []
        for i in range(len(x)):
            e = self._window(layer, kv, start, start + i)
            if extra is not None:
                e = torch.cat([e, extra[st.lists[i]]])
            rows.append(sparse_attn(q[i], e, sink, c.head_dim**-0.5, m))
        self._commit(layer, kv, start)
        return self._out(p, rope(torch.stack(rows), cis, m, inverse=True))

    def dspark_attention(self, layer: int, x: torch.Tensor, start: int) -> torch.Tensor:
        """M:1054-1074: every block row attends the stage ring at max(0, p-127)..p (p = start - 1) and all rows."""

        c, m, p = self.cfg, self.mode, self._prefix(layer)
        cis = self.cis(c.roles[layer].rope, self._rows(start, len(x)))
        _, q = self._q(p, x, cis)
        kv = self._kv(p, x, cis)
        e = torch.cat([self._window(layer, kv, start, start - 1), kv])
        sink = self.W(f"{p}.attn.attn_sink")
        o = torch.stack([sparse_attn(q[i], e, sink, c.head_dim**-0.5, m) for i in range(len(x))])
        return self._out(p, rope(o, cis, m, inverse=True))

    def _shared(self, p: str, x: torch.Tensor) -> list[torch.Tensor]:
        """M:886-887: the shared expert's w2 output as per-rank fp32 partials over its inner dim."""

        m = self.mode
        w1, w2, w3 = (self.W(f"{p}.ffn.shared_experts.{n}.weight") for n in ("w1", "w2", "w3"))
        h = m.bf16(swiglu(m.bf16(linear(x, w1, m)), m.bf16(linear(x, w3, m)), self.cfg.swiglu_limit))
        cut = w2.shape[1] // m.world
        return [linear(h[:, r * cut:(r + 1) * cut], w2[:, r * cut:(r + 1) * cut], m) for r in range(m.world)]

    def moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """M:889-904 with each expert's inner dim split over ranks: a rank sums its slots and shared part, ranks add."""

        c, m, p = self.cfg, self.mode, self._prefix(layer)
        backbone = layer < c.num_hidden_layers
        k = c.num_experts_per_tok if backbone else c.dspark_num_experts_per_tok
        wts, idx = gate(x, self.W(f"{p}.ffn.gate.weight"), self.W(f"{p}.ffn.gate.bias"), k, c)
        mirror = m.name == "mirror"
        y = x.new_zeros(m.world, len(x), k, x.shape[-1])
        for e in idx.unique().tolist():
            t, j = torch.nonzero(idx == e, as_tuple=True)
            w1, w2, w3 = (a.float() for a in self.expert(p, e))
            xe = m.fp16(x[t]) if backbone else x[t]        # the EXL3 kernels take fp16 activations
            h = swiglu(linear(xe, w1, m), linear(xe, w3, m), c.swiglu_limit)
            h = m.fp16(h) if mirror else wts[t, j, None] * h     # model.py weighs before w2 (M:849-851)
            cut = w2.shape[1] // m.world
            for r in range(m.world):
                out = linear(h[:, r * cut:(r + 1) * cut], w2[:, r * cut:(r + 1) * cut], m)
                y[r, t, j] = wts[t, j, None] * out if mirror else out
        shared = self._shared(p, x)
        parts = [reduce(torch.add, y[r].unbind(1)) + shared[r] for r in range(m.world)]
        return m.bf16(reduce(torch.add, parts))

    def engram(self, layer: int, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """M:350-365: table rows -> wkv -> a key per copy and one value, gated per copy by copysign(sqrt(|dot|))."""

        c, m, p = self.cfg, self.mode, self._prefix(layer)
        kv = m.bf16(linear(self.engram_rows(layer, ids).float().flatten(1), self.W(f"{p}.engram.wkv.weight"), m))
        cut = c.hc_mult * c.hidden_size
        key, value = kv[:, :cut].unflatten(-1, (c.hc_mult, -1)), kv[:, None, cut:]
        wqk = self.W(f"{p}.engram.q_weight") * self.W(f"{p}.engram.k_weight")
        eps = c.rms_norm_eps
        rstd = torch.rsqrt(X.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (X * wqk * key).sum(-1) * rstd * c.hidden_size**-0.5
        g = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return m.bf16(X + g[..., None] * value)

    def _mix(self, name: str, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """M:948-955: one sublayer's pre, post and comb from the stream."""

        c = self.cfg
        mixes = hc_mixes(X, self.W(f"{name}_fn"), c.rms_norm_eps)
        return hc_split_sinkhorn(mixes, self.W(f"{name}_scale"), self.W(f"{name}_base"), c.hc_mult,
                                 c.hc_sinkhorn_iters, c.hc_eps)

    def block(self, layer: int, X: torch.Tensor, pre: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """M:968-994: attention collapses with ``pre``, the FFN with the attention's own pre (delayed mHC)."""

        c, m, p = self.cfg, self.mode, self._prefix(layer)
        a_pre, a_post, a_comb = self._mix(f"{p}.hc_attn", X)
        xa = rms_norm(hc_pre(X, pre, m), self.W(f"{p}.attn_norm.weight"), c.rms_norm_eps, m)
        attn = self.attention if layer < c.num_hidden_layers else self.dspark_attention
        X = hc_post(attn(layer, xa, start), X, a_post, a_comb, m)
        f_pre, f_post, f_comb = self._mix(f"{p}.hc_ffn", X)
        xf = rms_norm(hc_pre(X, a_pre, m), self.W(f"{p}.ffn_norm.weight"), c.rms_norm_eps, m)
        return hc_post(self.moe(layer, xf), X, f_post, f_comb, m), f_pre

    def forward(self, tokens: list[int], all_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """M:1241-1272 for ``tokens`` at the committed position on; commits them; (logits, DSpark taps)."""

        c, m, st = self.cfg, self.mode, self.state
        start = st.pos
        X = self.W("embed.weight")[torch.as_tensor(tokens)].unsqueeze(1).repeat(1, c.hc_mult, 1)
        pre, taps, hashes = _one_hot(len(tokens), c.hc_mult), [], None
        if c.engram_layer_ids:
            hashes = torch.as_tensor(self.hasher.ids(st.tokens[-(c.engram_max_ngram_size - 1):], list(tokens)))
        for layer in range(c.num_hidden_layers):
            role = c.roles[layer]
            if role.engram:
                X = self.engram(layer, X, hashes[:, c.engram_layer_ids.index(layer)])
            if role.tap is not None:
                taps.append(m.bf16(X.mean(1)))
            X, pre = self.block(layer, X, pre, start)
        x = rms_norm(hc_pre(X, pre, m), self.W("norm.weight"), c.rms_norm_eps, m)
        logits = (x if all_logits else x[-1:]) @ self.W("lm_head.weight").T
        st.pos += len(tokens)
        st.tokens += list(tokens)
        return logits, (torch.cat(taps, -1) if taps else None)

    def absorb(self, taps: torch.Tensor, start: int) -> None:
        """M:1128-1130, 1039-1051: every DSpark stage's ring takes its window KV of main_x at start.. ."""

        c, m = self.cfg, self.mode
        main_x = rms_norm(m.bf16(linear(taps, self.W("mtp.0.main_proj.weight"), m)), self.W("mtp.0.main_norm.weight"),
                          c.rms_norm_eps, m)
        for stage in range(c.num_nextn_predict_layers):
            layer = c.num_hidden_layers + stage
            cis = self.cis(c.roles[layer].rope, self._rows(start, len(taps)))
            self._commit(layer, self._kv(self._prefix(layer), main_x, cis), start)

    def propose(self, y: int, sample: Callable[[torch.Tensor, int], int] | None = None
                ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """M:1128-1156 at p = pos - 1: block [y, noise...] through the stages, then the Markov chain in order.

        ``sample(logits, position)`` draws each draft (greedy, lower id on ties, by default); returns the drafts
        for positions p+2.., the block's logits with the Markov bias and the confidence logits.
        """

        c, m = self.cfg, self.mode
        sample = sample or (lambda logits, position: int(torch.argmax(logits)))
        p, n, last = self.state.pos - 1, c.num_hidden_layers, f"mtp.{c.num_nextn_predict_layers - 1}"
        ids = torch.tensor([y] + [c.dspark_noise_token_id] * (c.dspark_block_size - 1))
        X, pre = self.W("embed.weight")[ids].unsqueeze(1).repeat(1, c.hc_mult, 1), _one_hot(len(ids), c.hc_mult)
        for stage in range(c.num_nextn_predict_layers):
            X, pre = self.block(n + stage, X, pre, p + 1)
        x = hc_pre(X, pre, m)
        logits = rms_norm(x, self.W(f"{last}.norm.weight"), c.rms_norm_eps, m) @ self.W("lm_head.weight").T
        out, embeds = [y], []
        for i in range(len(ids)):
            e = self.W(f"{last}.markov_head.embed.weight")[out[i]]
            logits[i] += e @ self.W(f"{last}.markov_head.head.weight").T
            embeds.append(e)
            out.append(int(sample(logits[i], p + 2 + i)))
        conf = torch.cat([x, torch.stack(embeds)], -1) @ self.W(f"{last}.confidence_head.proj.weight").T
        return out[1:], logits, conf.squeeze(-1)
