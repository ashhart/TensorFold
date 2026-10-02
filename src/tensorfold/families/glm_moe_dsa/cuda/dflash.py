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
        self.capacity = capacity                         # 0.6.1's add_taps checks .capacity (the base was built ring-sized)
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


# ------------------------------------------------------------------------------------------- several streams ---
# Concurrent streams (multi.GlmMultiDecoder): one drafter's weights, a ring per stream, every live stream's block
# in one pass and every stream's kept taps in one context update. Ported from MiaAI-Lab's GLM-5.3-Flash patch
# 0027-glm-multi-dflash2 (Apache-2.0: dflash2_multi.MultiDrafter, the per-segment attention and convolution).

@triton.jit
def _dconv_seg(X, DYN, BASE, RES, OUT, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr, BRANCH: tl.constexpr,
               HAS_RES: tl.constexpr, BLOCK: tl.constexpr, SEG: tl.constexpr):
    """glm5_next's two-tap dynamic convolution over blocks of SEG rows side by side: a block's first row mixes no row
    before it (one block: the solo kernel's arithmetic)."""
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < D
    x = tl.load(X + row * D + c, mask=ok, other=0.0).to(tl.float32)
    prev = tl.load(X + (row - 1) * D + c, mask=ok & (row % SEG != 0), other=0.0).to(tl.float32)
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


def _dconv_s(x, dyn, base, branch, gs, seg, residual=None):
    rows, d = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    _dconv_seg[(rows, triton.cdiv(d, 1024))](x, dyn.contiguous(), base, residual if residual is not None else x, out,
                                             D=d, G=d // gs, GS=gs, BRANCH=branch, HAS_RES=residual is not None,
                                             BLOCK=1024, SEG=seg, num_warps=4)
    return out


@triton.jit
def _dattn_seg(Q, K, V, OUT, META, St, window, scale, L, POOL, N: tl.constexpr, G: tl.constexpr, NH: tl.constexpr,
               HD: tl.constexpr, RING_: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    """``_dattn_ring`` for block ``seg`` (program 1) of L rows side by side: its queries at rows seg * N of
    Q [heads, L, HD], its ring at slot META[St + seg] (rows slot * RING_ of K / V [kv heads, POOL, HD]), its committed
    length META[2 St + seg]. The solo kernel's loop and arithmetic, so one block gets the solo bits."""
    kvh = tl.program_id(0)
    seg = tl.program_id(1)
    M: tl.constexpr = G * N
    m = tl.arange(0, M)
    qh = kvh * G + m // N
    qr = m % N
    d = tl.arange(0, HD)
    q = tl.load(Q + (qh[:, None] * L + seg * N + qr[:, None]) * HD + d[None, :])
    s = tl.load(META + 2 * St + seg).to(tl.int32)
    base = kvh.to(tl.int64) * POOL + tl.load(META + St + seg).to(tl.int64) * RING_
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
        k = tl.load(K + (base + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (base + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
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
    tl.store(OUT + (seg * N + qr[:, None]) * (NH * HD) + qh[:, None] * HD + d[None, :], out.to(tl.bfloat16))


def merge_candidates(g: torch.Tensor, proj: torch.Tensor, k: int):
    """Host rows of a block pass -> (ids [depth, k], logits, projected rows): ``g`` [ranks, depth, 2 k] (each rank's
    top-k values, then ids as fp32 bits) merged by value then id over the ranks - ``Drafter.candidates``' merge."""
    import numpy as np

    values = torch.cat([g[r, :, :k] for r in range(g.shape[0])], dim=1).numpy().astype(np.float64)
    tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                       dim=1).numpy().astype(np.int64)
    if g.shape[0] > 1:
        order = np.lexsort((tokens, -values), axis=-1)[:, :k]
        values = np.take_along_axis(values, order, axis=1)
        tokens = np.take_along_axis(tokens, order, axis=1)
    return tokens, values, proj.numpy().astype(np.float64)


class MultiDrafter:
    """DFlash2 for up to ``streams`` concurrent streams over a ``GlmDrafter``'s weights.

    Ring pool per layer: [kv heads, streams * RING + trash, head_dim] - stream slot i's ring at rows i * RING, then
    ``trash`` rows that padded context updates (a captured row bucket wider than the rows kept) write to. The host
    keeps each slot's committed length (``end``); every pass takes its positions and destinations from small int64
    tables copied in before the call or graph replay. Every rank calls it identically (same slots, same order)."""

    def __init__(self, d: GlmDrafter, streams: int) -> None:
        self.d, self.N = d, streams
        self.dev, self.block = d.dev, d.block
        n = self.block
        self.t_max = -(-max(64, streams * n) // n) * n           # rows a context update takes (a block multiple)
        self.trash = self.t_max
        self.pool_rows = streams * RING + self.trash
        KV, hd = d.kvh, d.hd
        self.kc = [torch.zeros((KV, self.pool_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.vc = [torch.zeros((KV, self.pool_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.end = [0] * streams                                  # each slot's committed rows (host)
        self.b_meta = torch.zeros((3 * streams,), dtype=torch.int64, device=self.dev)   # pending | slot | end
        self.ids = torch.full((streams * n,), d.mask_id, dtype=torch.int32, device=self.dev)
        T = self.t_max
        self.t_meta = torch.zeros((2 * T,), dtype=torch.int64, device=self.dev)         # position | destination
        self.tap_in = torch.zeros((T, d.tap_in.shape[1]), dtype=torch.bfloat16, device=self.dev)
        m = n - 1
        V = d.fw.lm_head.shape[0]
        self.logits = torch.empty((streams * m, V), dtype=torch.float32, device=self.dev)
        self.cand_n = d.world * streams * m * 2 * d.top_k
        self.cand = torch.zeros((self.cand_n + streams * m * d.hproj.shape[0],), dtype=torch.float32, device=self.dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=True)
        self.proj = self.cand[self.cand_n:].view(streams * m, d.hproj.shape[0])
        self.pool = None
        self.block_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.buckets = list(range(n, T + 1, n))

    def nbytes(self) -> int:
        """The context rings (the weights are the solo drafter's)."""
        return 2 * sum(t.numel() * t.element_size() for t in self.kc)

    def reset(self, slot: int) -> None:
        self.end[slot] = 0

    # -- passes -------------------------------------------------------------------------------------------------
    def _rotary(self, pos: torch.Tensor):
        phase = pos.to(torch.float32)[:, None] * self.d.inv_freq[None, :]
        return phase.cos().contiguous(), phase.sin().contiguous()

    def _layer(self, i: int, x, cos, sin, idx, S: int):
        d, n = self.d, self.block
        L = d.layers[i]
        rows = x.shape[0]
        normed, xs = d._norm(x, L.in_norm)
        dyn = _mm(normed, L.a_kp, xs)
        q, k, v = d._prep(_mm(_dconv_s(normed, dyn, L.a_base, 0, d.gs, n), L.qkv), L, cos, sin, d.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, d.heads * d.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_seg[(d.kvh, S)](q, self.kc[i], self.vc[i], out, self.b_meta, self.N, d.window, d.hd ** -0.5, rows,
                               self.pool_rows, N=n, G=d.heads // d.kvh, NH=d.heads, HD=d.hd, RING_=RING, BK=64,
                               CAUSAL=d.causal, num_warps=4)
        x = _dconv_s(d._row(out, L.o), dyn, L.a_base, 1, d.gs, n, x)
        normed, xs = d._norm(x, L.post_norm)
        dyn = _mm(normed, L.m_kp, xs)
        gu = _mm(_dconv_s(normed, dyn, L.m_base, 0, d.gs, n), L.gu)
        act = torch.empty((rows, d.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, d.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return _dconv_s(d._row(act, L.down, axs), dyn, L.m_base, 1, d.gs, n, x)

    def _block_compute(self, S: int) -> None:
        """S slots' blocks [pending, mask x (block - 1)] at their committed lengths (b_meta), side by side; each
        block's candidates (every rank's top-k, gathered) and selector rows at rows s * (block - 1)."""
        d, n, N = self.d, self.block, self.N
        pend, slot, pos = self.b_meta[:S], self.b_meta[N:N + S], self.b_meta[2 * N:2 * N + S]
        self.ids.view(N, n)[:S, 0].copy_(pend)
        rowpos = (pos[:, None] + d.ar[:n][None, :]).reshape(-1)
        idx = (slot[:, None] * RING + (rowpos % RING).view(S, n)).reshape(-1)
        R = S * n
        x = torch.empty((R, d.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids[:R], d.fw.embed, d.D, 1, x)
        cos, sin = self._rotary(rowpos)
        for i in range(len(d.layers)):
            x = self._layer(i, x, cos, sin, idx, S)
        m, k = n - 1, d.top_k
        h, _ = d._norm(x.view(S, n, d.D)[:, 1:].reshape(S * m, d.D), d.norm)
        lg = self.logits[:S * m]
        glue.router(h, d.fw.lm_head, lg)
        vals, local = torch.topk(lg, k, dim=-1)
        gids = (local + d.fw.vocab_off).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        size = d.world * S * m * 2 * k
        if d.world > 1:
            d.fw.comm.all_gather(packed.view(-1), self.cand[:size])
        else:
            self.cand[:size].copy_(packed.view(-1))
        self.proj[:S * m].copy_(F.linear(h, d.hproj).float())

    def _taps_compute(self, T: int) -> None:
        """A context update of T stacked rows (t_meta: each row's position and pool destination)."""
        d, Tm = self.d, self.t_max
        pos, idx = self.t_meta[:T], self.t_meta[Tm:Tm + T]
        ctx, _ = d._norm(_mm(self.tap_in[:T], d.fc), d.hidden_norm)
        cos, sin = self._rotary(pos)
        for i, L in enumerate(d.layers):
            _, k, v = d._prep(_mm(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, idx, k)
            self.vc[i].index_copy_(1, idx, v)

    # -- graphs -------------------------------------------------------------------------------------------------
    def _trash_tables(self) -> None:
        N, Tm = self.N, self.t_max
        self.b_meta.zero_()
        self.b_meta[N:2 * N].fill_(N)                    # slot N: the trash rows (a block < the trash)
        self.t_meta.zero_()
        self.t_meta[Tm:].copy_(N * RING + torch.arange(Tm, device=self.dev))

    @torch.no_grad()
    def capture(self) -> None:
        """CUDA graphs: block passes of 1..N slots, context updates of each row bucket (all ranks together); the
        warm-ups and captures write only the trash rows."""
        self.pool = torch.cuda.graph_pool_handle()
        self._trash_tables()
        for T in self.buckets:
            for _ in range(2):
                self._taps_compute(T)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._taps_compute(T)
            self.tap_graphs[T] = g
        for S in range(1, self.N + 1):
            for _ in range(2):
                self._block_compute(S)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._block_compute(S)
            self.block_graphs[S] = g
        torch.cuda.synchronize()

    # -- the contexts -------------------------------------------------------------------------------------------
    def _launch_taps(self, pieces) -> None:
        N, Tm = self.N, self.t_max
        h = torch.zeros((2 * Tm,), dtype=torch.int64)
        off = 0
        for slot, part, p0 in pieces:
            r = part.shape[0]
            self.tap_in[off:off + r].copy_(part)
            p = torch.arange(p0, p0 + r, dtype=torch.int64)
            h[off:off + r] = p
            h[Tm + off:Tm + off + r] = slot * RING + p % RING
            off += r
        T = off
        g = None
        if self.tap_graphs:
            T = next(b for b in self.buckets if b >= off)
            g = self.tap_graphs[T]
        if T > off:                                      # padded rows: the trash
            h[Tm + off:Tm + T] = N * RING + torch.arange(off, T, dtype=torch.int64)
        self.t_meta.copy_(h.pin_memory(), non_blocking=True)
        if g is not None:
            g.replay()
        else:
            self._taps_compute(T)
        for slot, part, p0 in pieces:
            self.end[slot] = p0 + part.shape[0]

    @torch.no_grad()
    def commit(self, items) -> None:
        """[(slot, taps [n, taps * D] bf16)]: each slot's committed rows at its positions end, end + 1, ...; every
        slot's rows stacked into updates of up to t_max rows."""
        pieces, used = [], 0
        for slot, taps in items:
            if self.end[slot] + taps.shape[0] > self.d.cap:
                raise ValueError("drafter context past its capacity")
            start, end = 0, self.end[slot]
            while start < taps.shape[0]:
                take = min(taps.shape[0] - start, self.t_max - used)
                pieces.append((slot, taps[start:start + take], end))
                start, end, used = start + take, end + take, used + take
                if used == self.t_max:
                    self._launch_taps(pieces)
                    pieces, used = [], 0
        if pieces:
            self._launch_taps(pieces)

    # -- drafts -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def propose(self, reqs) -> list[list[int]]:
        """[(slot, pending, depth, sampling, confidence)] -> each slot's ``Drafter.propose``: the slots with a depth
        and a context share one block pass; the others get [] (as a solo drafter runs none for them)."""
        out: list[list[int]] = [[] for _ in reqs]
        live = [(i, min(r[2], self.block - 1)) for i, r in enumerate(reqs)
                if min(r[2], self.block - 1) >= 1 and self.end[r[0]] > 0]
        if not live:
            return out
        S, N, n = len(live), self.N, self.block
        h = torch.zeros((3 * N,), dtype=torch.int64)
        for s, (i, _) in enumerate(live):
            slot, pending = reqs[i][0], reqs[i][1]
            h[s], h[N + s], h[2 * N + s] = int(pending), slot, self.end[slot]
        self.b_meta.copy_(h.pin_memory(), non_blocking=True)
        g = self.block_graphs.get(S)
        if g is not None:
            g.replay()
        else:
            self._block_compute(S)
        self.cand_host.copy_(self.cand)
        d = self.d
        m, k = n - 1, d.top_k
        g_all = self.cand_host[:d.world * S * m * 2 * k].view(d.world, S * m, 2 * k)
        p_all = self.cand_host[self.cand_n:].view(N * m, -1)
        for s, (i, depth) in enumerate(live):
            slot, pending, _, sampling, conf = reqs[i]
            tokens, values, proj = merge_candidates(g_all[:, s * m:s * m + depth], p_all[s * m:s * m + depth], k)
            out[i] = d.chain(tokens, values, proj, pending, self.end[slot] + 1, sampling, conf)
        return out
