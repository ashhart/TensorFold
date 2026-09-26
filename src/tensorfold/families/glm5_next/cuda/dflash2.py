"""DFlash2 drafts for GLM-5.3-Flash on CUDA (checkpoint incoai/GLM-5.3-Flash-DFlash2).

The drafter is five Qwen3 layers with grouped dynamic convolutions and a candidate selector. Its context is
the committed positions' target taps: the mean of the four residual streams after target layers 5, 14, 24,
33 and 42 (vLLM's Eagle3 aux hidden states for glm5_next: ``hc_contract(hc_post(...))`` after those layers),
concatenated, projected by ``fc`` and normed, then each layer's keys and values. A round embeds
[pending token, mask x 7] at positions p .. p + 7 (attention inside the block is bidirectional; the context
comes in as keys and values), and rows 1.. give candidates for positions p + 1, p + 2, ... through the target's
head. A chain takes one candidate a position with the selector's edge scores and, when sampling, the
target's own keyed Gumbel noise for that position (the 27B Metal policy: edge 0.6, noise 0.7).

Drafts only change acceptance: the verifier keeps a draft only when it equals the serial sample. With two
ranks each holds half the attention heads, half the MLP and half the vocabulary; o_proj and down_proj
partials are gathered and added in rank order, and both ranks merge the same top candidates, so both chain
the same drafts. The weights are quantized to 4 bits (groups of 64) at load, like the 27B's drafter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from . import glue, qmm
from .weights import Weights

EDGE = 0.6        # the selector's edge score weight in a chain pick
NOISE = 0.7       # the target's keyed noise weight when sampling
NO_LIMIT = 1e30


@triton.jit
def _dconv_kernel(X, DYN, BASE, RES, OUT, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                  BRANCH: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """Two-tap grouped dynamic convolution over the block (row r mixes rows r and r - 1), plus the residual."""

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
    """Program (row, head) over [q heads | k heads | v heads]: q and k get their RMS norm and rotary embedding,
    v is copied; outputs are (heads, rows, head_dim)."""

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


@triton.jit
def _dattn_kernel(Q, K, V, OUT, POS, window, scale, N: tl.constexpr, G: tl.constexpr, NH: tl.constexpr,
                  HD: tl.constexpr, CAP: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    """Program = one key/value head: its G query heads x N block rows against keys [0, s + N), where s is the
    committed length on the device: context keys within the sliding window, and the block's own keys (all of
    them, or up to the row when causal). Online softmax in fp32; out [N, NH * HD] bf16."""

    kvh = tl.program_id(0)
    M: tl.constexpr = G * N
    m = tl.arange(0, M)
    qh = kvh * G + m // N
    qr = m % N
    d = tl.arange(0, HD)
    q = tl.load(Q + (qh[:, None] * N + qr[:, None]) * HD + d[None, :])
    s = tl.load(POS).to(tl.int32)
    klen = s + N
    qpos = s + qr
    m_i = tl.full([M], -1e30, tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HD], tl.float32)
    for start in range(0, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        k = tl.load(K + (kvh * CAP + kk[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (kvh * CAP + kk[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        ok = kin[None, :] & (((kk[None, :] < s) & (qpos[:, None] - kk[None, :] <= window)) | (kk[None, :] >= s))
        if CAUSAL:
            ok = ok & (kk[None, :] <= qpos[:, None])
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + qr[:, None] * (NH * HD) + qh[:, None] * HD + d[None, :], out.to(tl.bfloat16))


def _dconv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int, group_size: int,
           residual: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    block = 1024
    _dconv_kernel[(rows, triton.cdiv(d, block))](x, dyn.contiguous(), base, residual if residual is not None else x,
                                                 out, D=d, G=d // group_size, GS=group_size, BRANCH=branch,
                                                 HAS_RES=residual is not None, BLOCK=block, num_warps=4)
    return out


def _quantize4(w: torch.Tensor) -> qmm.Q4:
    """bf16 (N, K) -> MLX-style affine 4-bit in groups of 64 along K (q = round((w - min) / scale)), tiled."""

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
    return qmm.make_q4(words, scale.contiguous(), bias.contiguous())


def _mm(x: torch.Tensor, w: qmm.Q4, xs: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
    return qmm.matmul(x, w, xs, f32=f32)


@dataclass
class DraftLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    a_base: torch.Tensor      # [2, 2, D] bf16
    a_kp: qmm.Q4              # [2 * 2 * D/gs, D]
    m_base: torch.Tensor
    m_kp: qmm.Q4
    qkv: qmm.Q4               # this rank's [q heads | k heads | v heads]
    kv: qmm.Q4                # this rank's [k heads | v heads] (context rows)
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: qmm.Q4                 # [D, this rank's heads * head_dim]: a row-parallel partial
    gu: qmm.Q4                # this rank's [gate | up]
    down: qmm.Q4              # [D, this rank's MLP width]: a row-parallel partial


class Drafter:
    """The DFlash2 drafter for one sequence: static per-layer context keys and values (position-indexed, the
    committed length on the device), a block pass, and ``propose``. ``capture()`` records the block pass and the
    context update for 1..8 rows as CUDA graphs (same kernels and buffers, so the same drafts)."""

    def __init__(self, draft_dir: str | Path, w: Weights, *, block: int | None = None, capacity: int = 2560) -> None:
        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        dc = cfg["dflash_config"]
        self.w = w
        self.dev = w.device
        self.rank, self.world = w.rank, w.world
        self.D = int(cfg["hidden_size"])
        self.hd = int(cfg["head_dim"])
        heads, kv_heads = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        if heads % self.world or kv_heads % self.world:
            raise ValueError("drafter heads must split evenly over the ranks")
        self.heads, self.kvh = heads // self.world, kv_heads // self.world
        self.eps = float(cfg["rms_norm_eps"])
        theta = float(cfg["rope_parameters"]["rope_theta"])
        self.mask_id = int(dc["mask_token_id"])
        self.gs = int(dc["conv_group_size"])
        if int(dc["conv_kernel_size"]) != 2:
            raise ValueError("only two-tap convolutions are ported")
        self.block = int(block or dc["block_size"])
        self.top_k = int(dc["selector_top_k"])
        self.tap_layers = tuple(int(i) for i in dc["target_layer_ids"])
        self.window = int(cfg["sliding_window"]) - 1
        self.causal = bool(cfg.get("is_causal", True))
        inter = int(cfg["intermediate_size"])
        if inter % self.world:
            raise ValueError("drafter MLP must split evenly over the ranks")
        self.inter = inter // self.world
        r, H, KV, hd = self.rank, self.heads, self.kvh, self.hd
        dev = self.dev

        def gpu(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, torch.bfloat16).contiguous()

        quantize4 = _quantize4

        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            def get(name: str) -> torch.Tensor:
                return f.get_tensor(name)

            self.fc = quantize4(gpu(get("fc.weight")))
            self.hidden_norm = gpu(get("hidden_norm.weight"))
            self.norm = gpu(get("norm.weight"))
            self.hproj = gpu(get("candidate_selector.hidden_projection.weight"))
            self.pred = get("candidate_selector.predecessor_codebook").float().numpy().copy()
            self.succ = get("candidate_selector.successor_codebook").float().numpy().copy()
            self.layers: list[DraftLayer] = []
            for i in range(int(cfg["num_hidden_layers"])):
                p = f"layers.{i}."
                q = get(p + "self_attn.q_proj.weight")[r * H * hd:(r + 1) * H * hd]
                k = get(p + "self_attn.k_proj.weight")[r * KV * hd:(r + 1) * KV * hd]
                v = get(p + "self_attn.v_proj.weight")[r * KV * hd:(r + 1) * KV * hd]
                o = get(p + "self_attn.o_proj.weight")[:, r * H * hd:(r + 1) * H * hd]
                g = get(p + "mlp.gate_proj.weight")[r * self.inter:(r + 1) * self.inter]
                u = get(p + "mlp.up_proj.weight")[r * self.inter:(r + 1) * self.inter]
                dn = get(p + "mlp.down_proj.weight")[:, r * self.inter:(r + 1) * self.inter]
                for conv in ("attention_conv", "mlp_conv"):
                    if tuple(get(p + conv + ".base_kernel").shape) != (2, 2, self.D):
                        raise ValueError(f"unexpected {conv} base kernel shape")
                self.layers.append(DraftLayer(
                    in_norm=gpu(get(p + "input_layernorm.weight")),
                    post_norm=gpu(get(p + "post_attention_layernorm.weight")),
                    a_base=gpu(get(p + "attention_conv.base_kernel")),
                    a_kp=quantize4(gpu(get(p + "attention_conv.kernel_projection.weight"))),
                    m_base=gpu(get(p + "mlp_conv.base_kernel")),
                    m_kp=quantize4(gpu(get(p + "mlp_conv.kernel_projection.weight"))),
                    qkv=quantize4(gpu(torch.cat((q, k, v)))),
                    kv=quantize4(gpu(torch.cat((k, v)))),
                    q_norm=gpu(get(p + "self_attn.q_norm.weight")),
                    k_norm=gpu(get(p + "self_attn.k_norm.weight")),
                    o=quantize4(gpu(o)),
                    gu=quantize4(gpu(torch.cat((g, u)))),
                    down=quantize4(gpu(dn))))
        torch.cuda.empty_cache()
        self.inv_freq = 1.0 / theta ** (torch.arange(hd // 2, device=dev, dtype=torch.float32) * 2 / hd)
        # context keys and values by position (the block's own rows are written past the committed length
        # during a pass and overwritten by the next context update)
        self.cap = capacity + self.block
        self.kc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.vc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.pos_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.context_end = 0
        n = self.block
        self.ids = torch.full((n,), self.mask_id, dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.ar = torch.arange(max(64, n), device=dev)
        self.tap_in = torch.zeros((64, len(self.tap_layers) * self.D), dtype=torch.bfloat16, device=dev)
        self.packed: torch.Tensor | None = None       # [world, block - 1, 2 * top_k] after a pass
        self.proj: torch.Tensor | None = None         # [block - 1, selector rank] fp32
        self.pool = None
        self.block_graph = None
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def nbytes(self) -> int:
        total = sum(q.nbytes() for L in self.layers for q in (L.a_kp, L.m_kp, L.qkv, L.kv, L.o, L.gu, L.down))
        return total + self.fc.nbytes() + 2 * sum(t.numel() * 2 for t in self.kc)

    def reset(self) -> None:
        self.context_end = 0
        self.pos_dev.zero_()

    # -- pieces -------------------------------------------------------------------------------------------------
    def _rotary(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = (self.pos_dev + self.ar[:rows]).to(torch.float32)
        phase = pos[:, None] * self.inv_freq[None, :]
        return phase.cos().contiguous(), phase.sin().contiguous()

    def _prep(self, qkv: torch.Tensor, L: DraftLayer, cos: torch.Tensor, sin: torch.Tensor, heads: int):
        rows = qkv.shape[0]
        hd = self.hd
        q = torch.empty((heads, rows, hd), dtype=torch.bfloat16, device=self.dev) if heads else qkv
        k = torch.empty((self.kvh, rows, hd), dtype=torch.bfloat16, device=self.dev)
        v = torch.empty_like(k)
        _prep_kernel[(rows, heads + 2 * self.kvh)](qkv, L.q_norm, L.k_norm, cos, sin, q, k, v, rows, qkv.stride(0),
                                                   self.eps, H=heads, HKV=self.kvh, HALF=hd // 2, num_warps=1)
        return q, k, v

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        out = torch.empty((rows, self.D), dtype=torch.bfloat16, device=self.dev)
        xs = torch.empty((rows, self.D // 64), dtype=torch.float32, device=self.dev)
        glue.rmsnorm(x, weight, self.eps, out, xs)
        return out, xs

    def _row(self, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor | None = None) -> torch.Tensor:
        """A projection whose input is split over the ranks: fp32 partials added in rank order, then bf16."""

        part = _mm(x.contiguous(), q, xs, f32=True)
        if self.world == 1 or self.w.comm is None:
            return part.to(torch.bfloat16)
        got = torch.empty((self.world * part.numel(),), dtype=torch.float32, device=self.dev)
        self.w.comm.all_gather(part.reshape(-1), got)
        g = got.view(self.world, *part.shape)
        acc = g[0].clone()
        for i in range(1, self.world):
            acc += g[i]
        return acc.to(torch.bfloat16)

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        L = self.layers[i]
        rows = x.shape[0]
        normed, xs = self._norm(x, L.in_norm)
        dyn = _mm(normed, L.a_kp, xs)
        q, k, v = self._prep(_mm(_dconv(normed, dyn, L.a_base, 0, self.gs), L.qkv), L, cos, sin, self.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_kernel[(self.kvh,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.window, self.hd ** -0.5,
                                   N=rows, G=self.heads // self.kvh, NH=self.heads, HD=self.hd, CAP=self.cap, BK=64,
                                   CAUSAL=self.causal, num_warps=4)
        x = _dconv(self._row(out, L.o), dyn, L.a_base, 1, self.gs, x)
        normed, xs = self._norm(x, L.post_norm)
        dyn = _mm(normed, L.m_kp, xs)
        gu = _mm(_dconv(normed, dyn, L.m_base, 0, self.gs), L.gu)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return _dconv(self._row(act, L.down, axs), dyn, L.m_base, 1, self.gs, x)

    # -- the context --------------------------------------------------------------------------------------------
    def _taps_compute(self, n: int) -> None:
        ctx, _ = self._norm(_mm(self.tap_in[:n], self.fc), self.hidden_norm)
        cos, sin = self._rotary(n)
        idx = self.pos_dev + self.ar[:n]
        for i, L in enumerate(self.layers):
            _, k, v = self._prep(_mm(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, idx, k)
            self.vc[i].index_copy_(1, idx, v)
        self.pos_dev += n

    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        """Committed rows' taps [n, 5 * D] (bf16) at positions context_end, context_end + 1, ..."""

        for start in range(0, taps.shape[0], self.tap_in.shape[0]):
            part = taps[start:start + self.tap_in.shape[0]]
            n = part.shape[0]
            if self.context_end + n > self.cap - self.block:
                raise ValueError("drafter context past its capacity")
            self.tap_in[:n].copy_(part)
            g = self.tap_graphs.get(n)
            if g is not None:
                g.replay()
            else:
                self._taps_compute(n)
            self.context_end += n

    # -- drafts -------------------------------------------------------------------------------------------------
    def _block_compute(self) -> None:
        """One pass over [pending, mask x (block - 1)] at the committed length: every row's top candidates
        (merged over the ranks' vocabulary halves) and the selector's projected rows."""

        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.w.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self.pos_dev + self.ar[:n]
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, hs = self._norm(x[1:], self.norm)
        logits = _mm(h, self.w.head, hs)
        vals, local = torch.topk(logits.float(), self.top_k, dim=-1)
        gids = (local + self.w.vocab_offset).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.world > 1 and self.w.comm is not None:
            got = torch.empty((self.world * packed.numel(),), dtype=torch.float32, device=self.dev)
            self.w.comm.all_gather(packed.view(-1), got)
            self.packed = got.view(self.world, n - 1, 2 * self.top_k)
        else:
            self.packed = packed.view(1, n - 1, 2 * self.top_k)
        self.proj = F.linear(h, self.hproj).float()

    @torch.no_grad()
    def capture(self) -> None:
        """CUDA graphs for the block pass and for context updates of 1..block rows (both ranks together)."""

        self.pool = torch.cuda.graph_pool_handle()
        self.reset()
        self.tap_in.zero_()
        self.ids[0] = 0
        for n in range(1, self.block + 1):
            for _ in range(2):
                self._taps_compute(n)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._taps_compute(n)
            self.tap_graphs[n] = g
            self.reset()
        for _ in range(2):
            self._block_compute()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._block_compute()
        self.block_graph = g
        torch.cuda.synchronize()
        self.reset()

    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidate ids [depth, top_k] for the positions after ``pending``, their logits, the projected rows."""

        self.ids_host[0] = pending
        self.ids[:1].copy_(self.ids_host, non_blocking=True)
        if self.block_graph is not None:
            self.block_graph.replay()
        else:
            self._block_compute()
        g = self.packed[:, :depth].cpu()
        k = self.top_k
        values = torch.cat([g[r, :, :k] for r in range(g.shape[0])], dim=1).numpy().astype(np.float64)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                           dim=1).numpy().astype(np.int64)
        if g.shape[0] > 1:
            order = np.lexsort((tokens, -values), axis=-1)[:, :k]
            values = np.take_along_axis(values, order, axis=1)
            tokens = np.take_along_axis(tokens, order, axis=1)
        return tokens, values, self.proj[:depth].cpu().numpy().astype(np.float64)

    def chain(self, tokens: np.ndarray, values: np.ndarray, proj: np.ndarray, anchor: int, first: int,
              sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
        """One candidate a position: logits plus the selector's edge from the previous pick, plus the target's
        keyed noise at that position when sampling. ``confidence`` > 0: stop once the product of the picks'
        probabilities (softmax over the candidates) falls below it; the first draft is always kept."""

        depth = tokens.shape[0]
        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        noise = None
        if sampled:
            noise = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
        out: list[int] = []
        prev, chain = anchor, 1.0
        for d in range(depth):
            edge = self.succ[tokens[d]].astype(np.float64) @ (self.pred[prev].astype(np.float64) * proj[d])
            score = (values[d] + EDGE * edge) / temp
            pick = score + NOISE * noise[d] if noise is not None else score
            j = int(np.argmax(pick))
            if confidence > 0:
                p = np.exp(score - score.max())
                chain *= float(p[j] / p.sum())
                if d > 0 and chain < confidence:
                    break
            prev = int(tokens[d, j])
            out.append(prev)
        return out

    def propose(self, pending: int, depth: int, sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
        """Up to ``depth`` drafts for the positions after the pending token (which sits at context_end)."""

        depth = min(depth, self.block - 1)
        if depth < 1 or self.context_end == 0:
            return []
        tokens, values, proj = self.candidates(pending, depth)
        return self.chain(tokens, values, proj, pending, self.context_end + 1, sampling, confidence)
