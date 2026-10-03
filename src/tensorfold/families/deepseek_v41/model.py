"""DeepSeek-V4.1-Flash's backbone, after oMLX's ``patches/deepseek_v41/language.py`` (MIT)."""

from __future__ import annotations

from typing import Any

import math

import mlx.core as mx
import numpy as np

from tensorfold.families.deepseek_v41.caches import LayerCache, make_caches
from tensorfold.families.deepseek_v41.config import DECODE_ROWS, INDEX_QUERIES, PREFILL_QUERIES, Config
from tensorfold.families.deepseek_v41.engram import Engram, NgramHash
from tensorfold.families.deepseek_v41.quant import Experts, Linear, fp8, quantize_activation, swiglu_fp8


@mx.compile
def _norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    f = x.astype(mx.float32)
    return (f * mx.rsqrt(mx.mean(f * f, -1, keepdims=True) + eps) * weight).astype(x.dtype)


def rms(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """oMLX's (compiled) RMSNorm: fp32 statistics, times the weight in fp32, back to x's dtype."""

    return _norm(x, weight, eps)


@mx.compile
def _rope(x: mx.array, positions: mx.array, params: tuple, inverse: bool) -> mx.array:
    """oMLX's compiled ``_rope`` body (the frequencies computed inside, so a compiled graph's bits match its)."""

    d, base, original, beta_fast, beta_slow, factor, compressed = params
    freq = 1 / mx.power(base, mx.arange(0, d, 2).astype(mx.float32) / d)
    if compressed and original:

        def correction(rotations: float) -> float:
            return d * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), d - 1)
        smooth = 1 - mx.clip((mx.arange(d // 2) - low) / max(high - low, 1e-3), 0, 1)
        freq = freq / factor * (1 - smooth) + freq * smooth
    angles = positions.astype(mx.float32)[:, None] * freq
    angles = angles.reshape(len(positions), *([1] * (x.ndim - 2)), d // 2)
    if inverse:
        angles = -angles
    tail = x[..., -d:].astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    rotated = mx.stack([a * mx.cos(angles) - b * mx.sin(angles), a * mx.sin(angles) + b * mx.cos(angles)], -1)
    return mx.concatenate([x[..., :-d], rotated.reshape(*x.shape[:-1], d).astype(x.dtype)], -1)


def rope(x: mx.array, positions: mx.array, params: tuple, inverse: bool = False) -> mx.array:
    """RoPE on x [L, ..., D]'s last ``params[0]`` dims (interleaved pairs) at ``positions`` [L], in fp32."""

    return _rope(x, positions, params, inverse)


# -- hyper-connections ---------------------------------------------------------------------------
@mx.compile
def _hc_mix_weights(mixes: mx.array, scale: mx.array, base: mx.array, n: int, hc_eps: float,
                    iters: int) -> tuple[mx.array, mx.array, mx.array]:
    from tensorfold.families.deepseek_v41.kernels import sinkhorn

    pre = mx.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + hc_eps
    post = 2 * mx.sigmoid(mixes[..., n:2 * n] * scale[1] + base[n:2 * n])
    comb = (mixes[..., 2 * n:] * scale[2] + base[2 * n:]).reshape(*mixes.shape[:-1], n, n)
    comb = mx.softmax(comb, -1) + hc_eps
    return pre, post, sinkhorn(comb, hc_eps, iters)


@mx.compile
def _hc_mixes(x: mx.array, fn: mx.array, scale: mx.array, base: mx.array, n: int, eps: float, hc_eps: float,
              iters: int) -> tuple[mx.array, mx.array, mx.array]:
    flat = x.reshape(x.shape[0], -1).astype(mx.float32)
    mixes = (flat @ fn.T) * mx.rsqrt(mx.mean(flat * flat, -1, keepdims=True) + eps)
    return _hc_mix_weights(mixes, scale, base, n, hc_eps, iters)


class HC:
    """One mHC projection: a sublayer's post and comb mixes and the next pre-mix."""

    def __init__(self, fn: mx.array, base: mx.array, scale: mx.array, cfg: Config) -> None:
        self.fn = fn.astype(mx.float32)
        self.base, self.scale = base.astype(mx.float32), scale.astype(mx.float32)
        self.eps, self.hc_eps, self.iters, self.n = cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters, cfg.hc_mult

    def arrays(self) -> list[mx.array]:
        return [self.fn, self.base, self.scale]

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return _hc_mixes(x, self.fn, self.scale, self.base, self.n, self.eps, self.hc_eps, self.iters)


@mx.compile
def hc_pre(x: mx.array, pre: mx.array) -> mx.array:
    """The streams [R, 4, D] weighted by pre [R, 4] and summed in fp32: [R, D] in x's dtype (oMLX's, compiled)."""

    return mx.sum(x.astype(mx.float32) * pre[..., None], axis=-2).astype(x.dtype)


def hc_post(f: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """New streams: post_j * f + sum_i comb[i, j] * residual_i, in fp32."""

    out = post[..., None] * f[:, None, :] + mx.einsum("rij,rid->rjd", comb, residual.astype(mx.float32))
    return out.astype(f.dtype)


# -- compressed KV --------------------------------------------------------------------------------
class Compressor:
    """A KV source's compressor: ratio 1 normalises wkv(x) per position; ratio 2 softmax-pools blocks of 2 (fp32)."""

    def __init__(self, wkv: mx.array, wgate: mx.array | None, norm: mx.array, ratio: int, eps: float) -> None:
        self.ratio, self.norm, self.eps = int(ratio), norm, eps
        self.dim = int(wkv.shape[0])
        if self.ratio == 1:
            self.wkv = wkv
        else:
            self.proj32 = mx.concatenate([wkv, wgate]).astype(mx.float32)        # [2 * dim, D]

    def arrays(self) -> list[mx.array]:
        return [self.wkv if self.ratio == 1 else self.proj32, self.norm]

    def latents(self, x: mx.array, cache: LayerCache, start: int) -> tuple[mx.array | None, int]:
        """The rows' completed blocks' latents (normed, not roped) and the first block's index."""

        r, rows = self.ratio, int(x.shape[0])
        first, last = start // r, (start + rows) // r
        if r == 1:
            return rms(mx.matmul(x, self.wkv.T), self.norm, self.eps), first
        proj = mx.matmul(x.astype(mx.float32), self.proj32.T)                    # [rows, 2 * dim] fp32
        lo = first * r
        data = mx.concatenate([cache.proj_rows(lo, start), proj]) if start > lo else proj
        cache.write_proj(proj, start)
        if last <= first:
            return None, first
        data = data[:(last - first) * r].reshape(last - first, r, 2 * self.dim)
        kv, gate = data[..., :self.dim], data[..., self.dim:]
        pooled = mx.sum(kv * mx.softmax(gate, axis=1), axis=1)
        return rms(pooled.astype(x.dtype), self.norm, self.eps), first


class Indexer:
    """Scores the compressed positions for each query and keeps its ``topk`` best (FP4 queries and keys)."""

    def __init__(self, wq_b: Linear, weights_proj: Linear, wk: mx.array | None, k_norm: mx.array | None,
                 cfg: Config) -> None:
        self.wq_b, self.weights_proj, self.wk, self.k_norm = wq_b, weights_proj, wk, k_norm
        self.heads, self.dim, self.topk = cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk
        self.weight_scale = self.dim ** -0.5 * self.heads ** -0.5
        self.eps = cfg.rms_norm_eps

    def arrays(self) -> list[mx.array]:
        out = [*self.wq_b.arrays(), *self.weights_proj.arrays()]
        return out + [a for a in (self.wk, self.k_norm) if a is not None]

    def keys(self, latent: mx.array, positions: mx.array, inv_freq: tuple) -> mx.array:
        k = rms(mx.matmul(latent, self.wk.T), self.k_norm, self.eps)
        return quantize_activation(rope(k, positions, inv_freq), 4, 32)

    def queries(self, qr: mx.array, x: mx.array, positions: mx.array, inv_freq: tuple) -> tuple[mx.array, ...]:
        rows = int(x.shape[0])
        q = self.wq_b(qr).reshape(rows, self.heads, self.dim)
        q = quantize_activation(rope(q, positions, inv_freq), 4, 32)
        w = self.weights_proj(x).astype(mx.float32) * self.weight_scale
        return q, w

    @staticmethod
    def scores(q: mx.array, w: mx.array, keys: mx.array) -> mx.array:
        """[L, n]: sum over heads of the head weight times ReLU(q . k), in fp32."""

        s = mx.maximum(mx.matmul(q.astype(mx.float32), keys.astype(mx.float32).T), 0)      # [L, H, n]
        return mx.sum(s * w[..., None], axis=1)


def top_ids(scores: mx.array, count: int) -> mx.array:
    """Each row's ``count`` best ids ascending, ties to the lower id; ``fixed``: always ``count`` columns."""

    width = int(scores.shape[-1])
    count = min(count, width)
    if count <= 0:
        return mx.zeros((int(scores.shape[0]), 0), dtype=mx.int32)
    ids = mx.argsort(-scores, axis=-1)[:, :count].astype(mx.int32)
    valid = mx.take_along_axis(scores, ids, -1) > -mx.inf
    return mx.sort(mx.where(valid, ids, -1), axis=-1)


def candidate_blocks(scores: mx.array, visible: mx.array, count: int, block: int) -> mx.array:
    """The candidate source's blocks: each block's best score, the latest visible block forced in, top ``count``."""

    width = int(scores.shape[-1])
    padded = mx.pad(scores, [(0, 0), (0, -width % block)], constant_values=-mx.inf)
    grouped = padded.reshape(int(scores.shape[0]), -1, block).max(-1)
    n = int(grouped.shape[-1])
    latest = (visible[:, None] > 0) & (mx.arange(n)[None, :] == (visible[:, None] - 1) // block)
    return top_ids(mx.where(latest, mx.inf, grouped), count)


# -- attention -------------------------------------------------------------------------------------
class Attention:
    """wq_a | wkv on the FP8 input, RoPE, a 128-token FP8 window and the index's FP4 pool rows, sinks, grouped wo."""

    def __init__(self, w: dict[str, Any], cfg: Config, layer: int) -> None:
        self.cfg, self.layer = cfg, layer
        self.ratio = cfg.ratio(layer)
        self.heads, self.dim = cfg.num_attention_heads, cfg.head_dim
        self.groups, self.rank = cfg.o_groups, cfg.o_lora_rank
        self.eps, self.window, self.scale = cfg.rms_norm_eps, cfg.sliding_window, cfg.head_dim ** -0.5
        self.inv_freq = cfg.rope_params(layer)          # RoPE's settings (a tuple: constants of the compiled RoPE)
        self.wq_a, self.wkv, self.wq_b, self.wo_b = w["wq_a"], w["wkv"], w["wq_b"], w["wo_b"]
        wo_a = w["wo_a"]
        self.wo_a = wo_a.reshape(self.groups, self.rank, -1)                      # bf16 [g, rank, H * D / g]
        self.q_norm, self.kv_norm = w["q_norm"], w["kv_norm"]
        self.sink = w["attn_sink"].astype(mx.float32)
        self.compressor: Compressor | None = w.get("compressor")
        self.indexer: Indexer | None = w.get("indexer")
        self.kv_src = cfg.kv_source(layer)
        self.idx_src = cfg.index_source(layer)
        self.candidates = 0 <= cfg.candidate_source_layer_id < layer and self.indexer is not None

    def arrays(self) -> list[mx.array]:
        out = [*self.wq_a.arrays(), *self.wkv.arrays(), *self.wq_b.arrays(), *self.wo_b.arrays(), self.wo_a,
               self.q_norm, self.kv_norm, self.sink]
        if self.compressor is not None:
            out += self.compressor.arrays()
        if self.indexer is not None:
            out += self.indexer.arrays()
        return out

    def __call__(self, x: mx.array, caches: list[LayerCache], start: int, shared: dict[str, Any]) -> mx.array:
        """x [L, D] (attn-normed) at positions start .. of one stream whose layer caches are ``caches``."""

        cfg, L = self.cfg, int(x.shape[0])
        cache = caches[self.layer]
        positions = mx.arange(start, start + L)
        xq = fp8(x)
        query = self.wq_a(xq, prequantized=True)
        kv_in = self.wkv(xq, prequantized=True)
        qr = rms(query, self.q_norm, self.eps)
        q = rope(self.wq_b(qr).reshape(L, self.heads, self.dim), positions, self.inv_freq)
        kv = fp8(rope(rms(kv_in, self.kv_norm, self.eps), positions, self.inv_freq).astype(mx.float32))
        lo = max(0, start - (self.window - 1))
        keys = mx.concatenate([cache.key_rows(lo, start), kv]) if start > lo else kv
        cache.write_keys(kv, start)
        chosen = None
        if self.ratio:
            src = caches[self.kv_src]
            if self.compressor is not None:                         # this layer produces the shared pool rows
                latent, first = self.compressor.latents(x, cache, start)
                if latent is not None:
                    n = int(latent.shape[0])
                    pos = (mx.arange(first, first + n) * self.ratio).astype(mx.int32)
                    pooled = quantize_activation(rope(latent, pos, self.inv_freq), 4, 16, e4m3_scale=True)
                    cache.write_pool(pooled, first, "pool")
                    cache.write_pool(self.indexer.keys(latent, pos, self.inv_freq), first, "ipool")
            if self.indexer is not None:
                shared["idx"] = self._choose(qr, x, positions, start, L, src, shared)
            chosen = shared["idx"]
        out = self._attend(q, keys, lo, start, chosen, caches[self.kv_src] if self.ratio else None)
        out = rope(out, positions, self.inv_freq, inverse=True)
        u = mx.einsum("lgd,grd->lgr", out.reshape(L, self.groups, -1), self.wo_a).reshape(L, -1)
        return self.wo_b(u)

    def _choose(self, qr: mx.array, x: mx.array, positions: mx.array, start: int, L: int, src: LayerCache,
                shared: dict[str, Any]) -> mx.array:
        """Each query's chosen pool rows [L, k] (ascending, -1 unused), as the index (and candidates) pick them."""

        cfg, r = self.cfg, self.ratio
        visible = (positions + 1) // r                                            # pool rows each query may see
        n = (start + L) // r
        iq, iw = self.indexer.queries(qr, x, positions, self.inv_freq)
        picks, blocks_out = [], []
        for q0 in range(0, L, INDEX_QUERIES):
            q1 = min(L, q0 + INDEX_QUERIES)
            picks.append(self._choose_rows(iq[q0:q1], iw[q0:q1], visible[q0:q1], n, src, shared, q0, q1,
                                           blocks_out, start + q0))
        if blocks_out:
            shared["candidates"] = mx.concatenate(blocks_out) if len(blocks_out) > 1 else blocks_out[0]
        return mx.concatenate(picks) if len(picks) > 1 else picks[0]

    def _choose_rows(self, iq: mx.array, iw: mx.array, visible: mx.array, n: int, src: LayerCache,
                     shared: dict[str, Any], q0: int, q1: int, blocks_out: list[mx.array], first: int) -> mx.array:
        from tensorfold.families.deepseek_v41 import kernels as KV

        cfg, L = self.cfg, q1 - q0
        fast = KV.index_fits(self.indexer.dim)
        if self.candidates:
            blocks = shared["candidates"][q0:q1]                                  # [L, blocks]
            if not int(blocks.shape[-1]):
                return mx.zeros((L, 0), dtype=mx.int32)
            size = cfg.candidate_block_size
            cand = blocks[..., None] * size + mx.arange(size)
            cand = mx.where(blocks[..., None] >= 0, cand, -1).reshape(L, -1)
            if fast:
                s = KV.index_scores(iq, iw, src.ipool, n, first, self.ratio, cand)
            else:
                keys = src.ipool[mx.maximum(cand, 0).reshape(-1)].reshape(L, -1, self.indexer.dim)
                s = mx.maximum(mx.einsum("lhd,lkd->lhk", iq.astype(mx.float32), keys.astype(mx.float32)), 0)
                s = mx.sum(s * iw[..., None], axis=1)
                s = mx.where((cand >= 0) & (cand < visible[:, None]), s, -mx.inf)
            order = top_ids(s, cfg.index_topk)
            ids = mx.take_along_axis(cand, mx.maximum(order, 0), -1)
            return mx.sort(mx.where(order >= 0, ids, -1), axis=-1)
        if fast and n:
            s = KV.index_scores(iq, iw, src.ipool, n, first, self.ratio)
        else:
            s = self.indexer.scores(iq, iw, src.ipool[:n]) if n else mx.zeros((L, 0), dtype=mx.float32)
            s = mx.where(mx.arange(n)[None, :] < visible[:, None], s, -mx.inf)
        if self.layer == cfg.candidate_source_layer_id:
            blocks_out.append(candidate_blocks(s, visible, cfg.candidate_topk_blocks, cfg.candidate_block_size)
                              if n else mx.zeros((L, 0), dtype=mx.int32))
        return top_ids(s, cfg.index_topk)

    def _attend(self, q: mx.array, keys: mx.array, lo: int, start: int, chosen: mx.array | None,
                src: LayerCache | None) -> mx.array:
        """q [L, H, D] over each query's window slots and chosen pool rows with the sinks, in DeepSeek's order."""

        L = int(q.shape[0])
        window = min(L, self.window) if start == 0 else self.window
        outs = []
        for q0 in range(0, L, PREFILL_QUERIES):
            q1 = min(L, q0 + PREFILL_QUERIES)
            n = q1 - q0
            pos = mx.arange(start + q0, start + q1)[:, None]
            if start == 0:
                slot = mx.maximum(pos - self.window + 1, 0) + mx.arange(window)[None, :]
            else:
                slot = pos - self.window + 1 + mx.arange(window)[None, :]
            ok = (slot >= lo) & (slot <= pos)
            comp = chosen[q0:q1] if chosen is not None and int(chosen.shape[-1]) else None
            from tensorfold.families.deepseek_v41 import kernels as KV

            if KV.attention_fits(window + (0 if comp is None else int(comp.shape[-1])), self.dim):
                wi = mx.where(ok, slot - lo, -1)
                ci = comp if comp is not None else mx.zeros((n, 0), dtype=mx.int32)
                outs.append(KV.attention(q[q0:q1], keys, src.pool if comp is not None else None, wi, ci, self.sink,
                                         self.scale))
                continue
            values = keys[mx.clip(slot - lo, 0, int(keys.shape[0]) - 1).reshape(-1)].reshape(n, window, self.dim)
            if chosen is not None and int(chosen.shape[-1]):
                ids = chosen[q0:q1]
                sel = src.pool[mx.maximum(ids, 0).reshape(-1)].reshape(n, -1, self.dim).astype(mx.float32)
                values = mx.concatenate([values, sel], axis=1)
                ok = mx.concatenate([ok, ids >= 0], axis=1)
            outs.append(self._grouped_softmax(q[q0:q1], values, ok).astype(q.dtype))
        return mx.concatenate(outs) if len(outs) > 1 else outs[0]

    def _grouped_softmax(self, q: mx.array, values: mx.array, ok: mx.array) -> mx.array:
        """q [n, H, D] over its own slots values [n, S, D] (valid ``ok`` [n, S]): [n, H, D] fp32."""

        n, S = int(values.shape[0]), int(values.shape[1])
        s = mx.einsum("nhd,nsd->nhs", q.astype(mx.float32), values) * self.scale
        s = mx.where(ok[:, None, :], s, -mx.inf)
        pad = -S % 64
        if pad:
            s = mx.pad(s, [(0, 0), (0, 0), (0, pad)], constant_values=-mx.inf)
            values = mx.pad(values, [(0, 0), (0, pad), (0, 0)])
        G = (S + pad) // 64
        s = s.reshape(n, self.heads, G, 64)
        m = mx.maximum(mx.cummax(mx.max(s, axis=-1), axis=-1), -1e30)          # running maxima [n, H, G]
        from tensorfold.families.deepseek_v41.kernels import fexp

        p = fexp(s - m[..., None])
        denom = mx.sum(p, axis=-1)                                                # [n, H, G] fp32
        pv = mx.einsum("nhgk,ngkd->nhgd", p.astype(mx.bfloat16).astype(mx.float32), values.reshape(n, G, 64, -1))
        acc, total, mprev = pv[:, :, 0], denom[:, :, 0], m[:, :, 0]
        for g in range(1, G):
            c = fexp(mprev - m[:, :, g])
            total = total * c + denom[:, :, g]
            acc = acc * c[..., None] + pv[:, :, g]
            mprev = m[:, :, g]
        total = total + fexp(self.sink[None, :] - mprev)
        return acc / total[..., None]


# -- MoE ---------------------------------------------------------------------------------------------
class MoE:
    """sqrt(softplus) router; the top 6 of score + bias, renormalised x 1.5; FP8-in mxfp4 experts and a shared one."""

    def __init__(self, gate_w: mx.array, bias: mx.array, w1: Experts, w3: Experts, w2: Experts, s1: Linear,
                 s3: Linear, s2: Linear, top: int, scale: float, limit: float) -> None:
        self.router = gate_w.astype(mx.float32)
        self.bias = bias.astype(mx.float32)
        self.w1, self.w3, self.w2 = w1, w3, w2
        self.s1, self.s3, self.s2 = s1, s3, s2
        self.top, self.scale, self.limit = int(top), float(scale), float(limit)

    def arrays(self) -> list[mx.array]:
        out = [self.router, self.bias]
        for e in (self.w1, self.w3, self.w2, self.s1, self.s3, self.s2):
            out += e.arrays()
        return out

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        raw = x.astype(mx.float32) @ self.router.T
        scores = mx.sqrt(mx.logaddexp(raw, mx.array(0.0)))
        idx = mx.argsort(-(scores + self.bias), axis=-1)[:, :self.top]
        w = mx.take_along_axis(scores, idx, -1)
        if self.top > 1:
            w = w / (mx.sum(w, -1, keepdims=True) + 1e-20)
        return idx, w * self.scale

    def __call__(self, x: mx.array) -> mx.array:
        idx, w = self.route(x)
        xq = fp8(x)
        rows, k = int(x.shape[0]), self.top
        sort = idx.size >= 64
        if sort:
            flat = idx.reshape(-1)
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            h = xq[order // k][:, None, :]                                      # [R * k, 1, D]
            ids, ws = flat[order], w.reshape(-1)[order][:, None, None]
        else:
            h, ids, ws = xq[:, None, None, :], idx, w[..., None, None]
        g = self.w1(h, ids, sort)
        u = self.w3(h, ids, sort)
        y = swiglu_fp8(g, u, ws, self.limit, x.dtype)
        d = self.w2(y, ids, sort)
        if sort:
            d = d[inverse].reshape(rows, k, 1, -1)
        routed = mx.sum(d.squeeze(-2).astype(mx.float32), axis=-2)
        sh = self.s2(swiglu_fp8(self.s1(xq, prequantized=True), self.s3(xq, prequantized=True), None, self.limit,
                                x.dtype), prequantized=True)
        return (routed + sh.astype(mx.float32)).astype(x.dtype)


# -- blocks and the backbone --------------------------------------------------------------------------
class Block:
    def __init__(self, layer: int, attn: Attention, moe: MoE, attn_norm: mx.array, ffn_norm: mx.array, attn_hc: HC,
                 ffn_hc: HC, eps: float, engram: Engram | None = None) -> None:
        self.layer = layer
        self.attn, self.moe = attn, moe
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.eps = eps
        self.engram = engram

    def arrays(self) -> list[mx.array]:
        out = [*self.attn.arrays(), *self.moe.arrays(), self.attn_norm, self.ffn_norm, *self.attn_hc.arrays(),
               *self.ffn_hc.arrays()]
        return out + (self.engram.arrays() if self.engram is not None else [])

    def __call__(self, h: mx.array, pre: mx.array, caches: list[LayerCache], start: int,
                 shared: dict[str, Any]) -> tuple[mx.array, mx.array]:
        """One stream's rows: streams h [L, 4, D] and the pre-mix carried in [L, 4]; the new streams and pre-mix."""

        from tensorfold.families.deepseek_v41 import kernels as KV

        ap, ao, ac = self.attn_hc(h)
        a = self.attn(KV.hc_pre_norm(h, pre, self.attn_norm, self.eps), caches, start, shared)
        h = KV.hc_post(a, h, ao, ac)
        fp, fo, fc = self.ffn_hc(h)
        m = self.moe(KV.hc_pre_norm(h, ap, self.ffn_norm, self.eps))
        return KV.hc_post(m, h, fo, fc), fp


class DeepSeekV41:
    """The backbone: ``hidden`` (final-normed rows; taps for DSpark in ``last_taps``) and ``head``."""

    def __init__(self, cfg: Config, embed: mx.array, layers: list[Block], norm: mx.array, lm_head: mx.array,
                 hasher: NgramHash | None = None) -> None:
        self.args = cfg
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.hasher = hasher
        self.last_streams: mx.array | None = None
        self.last_normed: mx.array | None = None
        self.tap_layers: tuple[int, ...] = ()        # layers whose input streams' mean a draft model reads
        self.last_taps: mx.array | None = None

    def set_token_map(self, token_map: Any) -> None:
        self.hasher = NgramHash(self.args, token_map) if self.args.engram_layer_ids else None

    def make_cache(self) -> list[Any]:
        return make_caches(self.args)

    def embed_tokens(self, ids: mx.array) -> mx.array:
        return self.embed[ids]

    def hidden(self, tokens: Any, cache: list[Any]) -> mx.array:
        return self.hidden_rows(tokens, [cache])

    def _lookups(self, ids: np.ndarray, cache: LayerCache, start: int) -> np.ndarray | None:
        """A stream's rows' Engram lookups [L, tables, depth * heads] (its token ring takes the rows)."""

        if self.hasher is None:
            if self.args.engram_layer_ids:
                raise ValueError("deepseek_v41: Engram needs the tokenizer's compressed token map (set_token_map)")
            return None
        depth = self.hasher.depth
        before = np.array(cache.token_rows(max(0, start - depth), start)) if start else np.zeros((0,), np.int64)
        rows = self.hasher(ids, self.hasher.history(before, start))
        cache.write_tokens(mx.array(ids.astype(np.int32)), start)
        return rows

    def hidden_rows(self, tokens: Any, caches: list[list[Any]], lengths: Any = None) -> mx.array:
        """Several streams' rows in one forward, each row with its own call's bits; a prompt chunk is one stream's."""

        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        host = np.array(ids).astype(np.int64)
        rows = int(ids.shape[0])
        lengths = (rows,) if lengths is None else tuple(int(n) for n in lengths)
        decode = rows <= DECODE_ROWS
        if sum(lengths) != rows or len(lengths) != len(caches) or (len(lengths) > 1 and not decode):
            raise ValueError(f"hidden_rows: {len(caches)} streams of {lengths} rows for {rows} tokens (at most "
                             f"{DECODE_ROWS} rows when shared)")
        cfg = self.args
        starts = [int(c[0].offset) for c in caches]
        lookups, at = [], 0
        for c, n, s in zip(caches, lengths, starts):
            lookups.append(self._lookups(host[at:at + n], c[0], s))
            at += n
        h = self.embed_tokens(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, cfg.hc_mult, h.shape[-1])))
        pre = mx.broadcast_to((mx.arange(cfg.hc_mult) == 0).astype(mx.float32), (rows, cfg.hc_mult))
        # the work units: a decode row alone (its stream, row in it), or a whole prompt chunk
        units = []
        at = 0
        for s, n in enumerate(lengths):
            if decode:
                units += [(s, at + j, j, 1) for j in range(n)]
            else:
                units.append((s, at, 0, n))
            at += n
        shared: list[dict[str, Any]] = [{} for _ in units]
        hs = [x[a:a + n] for _, a, _, n in units]
        pres = [pre[a:a + n] for _, a, _, n in units]
        taps: list[list[mx.array]] = [[] for _ in units]
        for i, layer in enumerate(self.layers):
            for u, (s, a, j, n) in enumerate(units):
                if layer.engram is not None:
                    k = cfg.engram_layer_ids.index(i)
                    hs[u] = layer.engram(hs[u], lookups[s][j:j + n, k])
                if i in self.tap_layers:
                    taps[u].append(mx.mean(hs[u], axis=1))
                hs[u], pres[u] = layer(hs[u], pres[u], caches[s], starts[s] + j, shared[u])
            for c, n, st in zip(caches, lengths, starts):
                c[i].offset = st + n
            if (i + 1) % 2 == 0 and i + 1 < len(self.layers):
                mx.async_eval(*hs, *pres)
        x = mx.concatenate(hs) if len(hs) > 1 else hs[0]
        pre = mx.concatenate(pres) if len(pres) > 1 else pres[0]
        self.last_streams = x
        self.last_taps = (mx.concatenate([mx.concatenate(t, axis=-1) for t in taps]) if self.tap_layers else None)
        if decode:
            normed = [rms(hc_pre(x[r:r + 1], pre[r:r + 1]), self.norm, cfg.rms_norm_eps) for r in range(rows)]
            self.last_normed = mx.concatenate(normed) if rows > 1 else normed[0]
        else:
            self.last_normed = rms(hc_pre(x, pre), self.norm, cfg.rms_norm_eps)
        return self.last_normed[None]

    def head(self, hidden: mx.array) -> mx.array:
        """Logits in fp32 (bf16 head, fp32 accumulation), each decode row on its own."""

        from tensorfold.families.deepseek_v41 import kernels as KV

        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        out = KV.head_logits(flat, self.lm_head, int(flat.shape[0]) <= DECODE_ROWS)
        return out.reshape(*shape[:-1], -1)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        for c in cache[:len(self.layers)]:
            c.trim(rows - keep)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches, lengths, keeps):
            if int(keep) < int(rows):
                self.keep_rows(cache, int(rows), int(keep))
