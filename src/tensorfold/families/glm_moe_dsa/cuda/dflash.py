"""DFlash2 drafts for full GLM-5.3: TensorFold's GLM-5.3-Flash drafter (glm5_next/cuda/dflash2.py) over this family.

The drafter reads the target's hidden rows after the tap layers (its ``target_layer_ids``, the output of each layer),
keeps its own sliding-window context from the committed rows' taps, and drafts a block of mask rows after the pending
token. Only the output head differs from the Flash family: here it is the bf16 lm_head share of each rank (the router
kernel), and each rank's top-k candidates merge over the ranks exactly as there. Drafts never have to be exact - the
target verifies every one - so the drafter's 4-bit weights and ring-free projections are fine.

Checkpoint: e.g. incoai/GLM-5.3-DFlash2 (6 layers, block 8, taps 5/19/33/47/61/75, trained against BF16 GLM-5.3).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import glue
from tensorfold.families.glm5_next.cuda.dflash2 import NO_LIMIT, Drafter, _dconv, _mm, _dattn_kernel  # noqa: F401

RING = 4096          # drafter KV slots: its sliding window (2047) + a block (8) + a tap batch (64), rounded up


@triton.jit
def _dattn_ring(Q, K, V, OUT, POS, window, scale, N: tl.constexpr, G: tl.constexpr, NH: tl.constexpr,
                HD: tl.constexpr, RING_: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    """glm5_next's drafter attention over a ring of RING_ slots (slot = position % RING_): each KV head attends its
    query groups to the sliding window of committed context and the block, visiting only [s - window, s + N)."""
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
    lo = tl.maximum(s - window, 0)
    lo = lo - lo % BK
    m_i = tl.full([M], -1e30, tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HD], tl.float32)
    for start in range(lo, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        slot = kk % RING_
        k = tl.load(K + (kvh * RING_ + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (kvh * RING_ + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
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

from . import fused


class GlmDrafter(Drafter):
    def __init__(self, draft_dir: str | Path, w: fused.Weights, capacity: int) -> None:
        shim = SimpleNamespace(device=w.device, rank=w.rank, world=w.world, comm=w.comm, embed=w.embed, head=None,
                               draft_head=None, vocab_offset=w.vocab_off)
        super().__init__(draft_dir, shim, capacity=RING - 2 * 8)      # small: the caches are replaced by a ring
        if self.window + self.block + self.tap_in.shape[0] > RING:
            raise ValueError(f"drafter window {self.window} + block + tap batch exceeds the {RING}-slot ring")
        self.cap = capacity + self.block                 # the context bound add_taps checks (positions, not slots)
        KV, hd = self.kvh, self.hd
        self.kc = [torch.zeros((KV, RING, hd), dtype=torch.bfloat16, device=self.dev) for _ in self.layers]
        self.vc = [torch.zeros((KV, RING, hd), dtype=torch.bfloat16, device=self.dev) for _ in self.layers]
        torch.cuda.empty_cache()
        self.fw = w
        V = w.lm_head.shape[0]
        self.logits = torch.empty((self.block - 1, V), dtype=torch.float32, device=w.device)

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        L = self.layers[i]
        rows = x.shape[0]
        normed, xs = self._norm(x, L.in_norm)
        dyn = _mm(normed, L.a_kp, xs)
        q, k, v = self._prep(_mm(_dconv(normed, dyn, L.a_base, 0, self.gs), L.qkv), L, cos, sin, self.heads)
        slots = idx % RING
        self.kc[i].index_copy_(1, slots, k)
        self.vc[i].index_copy_(1, slots, v)
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_ring[(self.kvh,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.window, self.hd ** -0.5,
                                 N=rows, G=self.heads // self.kvh, NH=self.heads, HD=self.hd, RING_=RING, BK=64,
                                 CAUSAL=self.causal, num_warps=4)
        x = _dconv(self._row(out, L.o), dyn, L.a_base, 1, self.gs, x)
        normed, xs = self._norm(x, L.post_norm)
        dyn = _mm(normed, L.m_kp, xs)
        gu = _mm(_dconv(normed, dyn, L.m_base, 0, self.gs), L.gu)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return _dconv(self._row(act, L.down, axs), dyn, L.m_base, 1, self.gs, x)

    def _taps_compute(self, n: int) -> None:
        ctx, _ = self._norm(_mm(self.tap_in[:n], self.fc), self.hidden_norm)
        cos, sin = self._rotary(n)
        slots = (self.pos_dev + self.ar[:n]) % RING
        for i, L in enumerate(self.layers):
            _, k, v = self._prep(_mm(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, slots, k)
            self.vc[i].index_copy_(1, slots, v)
        self.pos_dev += n

    def _block_compute(self) -> None:
        """[pending, mask x (block - 1)] at the committed length; each rank's top-k over its lm_head share, merged."""
        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.fw.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self.pos_dev + self.ar[:n]
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, _ = self._norm(x[1:], self.norm)
        glue.router(h, self.fw.lm_head, self.logits)
        vals, local = torch.topk(self.logits, self.top_k, dim=-1)
        gids = (local + self.fw.vocab_off).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.world > 1:
            got = torch.empty((self.world * packed.numel(),), dtype=torch.float32, device=self.dev)
            self.fw.comm.all_gather(packed.view(-1), got)
            self.packed = got.view(self.world, n - 1, 2 * self.top_k)
        else:
            self.packed = packed.view(1, n - 1, 2 * self.top_k)
        self.proj = F.linear(h, self.hproj).float()
