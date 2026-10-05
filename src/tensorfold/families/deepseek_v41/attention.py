"""DeepSeek-V4.1's attention: 64 query heads over one shared latent KV head, a 128-token window and shared pools.

Key order per query (the official ``Attention.forward``): the sliding window's rows (oldest first),
then the pool's chosen rows in ascending block order. ``sparse_attn`` sums the selected keys plus one
sink logit a head, fp32 scores at head_dim ** -0.5. A compressed layer reads and writes the pool state
of its group's head cache (the kv source at or below it in the same stream), so a stream's caches stay
one list and engine copies isolate streams.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.caches import LayerCache, head_of
from tensorfold.families.deepseek_v41.compressor import (Compressor, Indexer, candidate_mask, norm_rope,
                                                        pool_blocks)
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.dense import dense
from tensorfold.families.deepseek_v41.quant import quantize_cache
from tensorfold.families.deepseek_v41.rmsrows import rms_rows
from tensorfold.families.glm5_next.linear import Q, per_row, project
from tensorfold.kernels.deepseek.v4 import rope as NR


def positions_of(caches: list[Any], lengths: tuple[int, ...]) -> mx.array:
    """Each row's position: the streams' rows from their caches' offsets, in order."""
    parts = [mx.arange(c.offset, c.offset + n) for c, n in zip(caches, lengths)]
    return mx.concatenate(parts) if len(parts) > 1 else parts[0]


def sparse_attention(q: mx.array, kv: mx.array, sink: mx.array) -> mx.array:
    """The reference's decode attention: fp32 scores . dim ** -0.5, softmax over keys plus one sink a head."""
    scores = (q.astype(mx.float32) @ kv.astype(mx.float32).T) * (q.shape[-1] ** -0.5)   # [H, N]
    p = mx.softmax(mx.concatenate((scores, sink[:, None]), axis=-1), axis=-1)
    return (p[:, :-1] @ kv.astype(mx.float32)).astype(q.dtype)


def _rope_pinv(x: mx.array, positions: mx.array, inv_freq: mx.array) -> mx.array:
    return norm_rope(x, positions, inv_freq, norm=False, inverse=True)


class Attention:
    """wq_a | wkv, per-head RMS queries, RoPE on the last 64 dims, sinks, inverse RoPE, grouped low-rank output."""

    def __init__(self, w: dict[str, Any], cfg: Config, layer: int) -> None:
        self.layer = layer
        self.ratio = cfg.ratio(layer)
        self.mode = cfg.mode(layer)
        self.head = head_of(cfg, layer) if self.ratio else None
        self.heads, self.dim = cfg.num_attention_heads, cfg.head_dim
        self.groups, self.rank = cfg.o_groups, cfg.o_lora_rank
        self.eps = cfg.rms_norm_eps
        self.window = cfg.sliding_window
        self.scale = self.dim ** -0.5
        self.inv_freq = cfg.inv_freq(layer)
        self.q_rank = w["wq_a"].outs
        self.x_proj = Q.stack([w["wq_a"], w["wkv"]])
        self.q_norm, self.kv_norm = w["q_norm"], w["kv_norm"]
        self.wq_b, self.wo_b = w["wq_b"], w["wo_b"]
        self.wo_a_all = wo_a = w["wo_a"]
        self.wo_a = [Q(wo_a.weight[g * self.rank:(g + 1) * self.rank], wo_a.scales[g * self.rank:(g + 1) * self.rank],
                       wo_a.biases[g * self.rank:(g + 1) * self.rank], bits=wo_a.bits, group=wo_a.group)
                     for g in range(self.groups)]
        self.sink = w["attn_sink"].astype(mx.float32)             # the reference's fp32 sink logits
        self.compressor: Compressor | None = w.get("compressor")
        self.indexer: Indexer | None = w.get("indexer")
        self.indexer_wk: mx.array | None = w.get("indexer_wk")
        self.indexer_k_norm: mx.array | None = w.get("indexer_k_norm")
        self.candidate_source = layer == cfg.candidate_source_layer_id
        self.candidate_after = layer > cfg.candidate_source_layer_id
        self.candidate_block = cfg.candidate_block_size
        self.candidate_topk = cfg.candidate_topk_blocks
        self.topk = cfg.index_topk
        self._front: Any = None
        self._back: Any = None

    # -- what every layer computes from its rows -------------------------------------------------------
    def front(self, x: mx.array, positions: mx.array, rows_exact: bool) -> tuple[mx.array, ...]:
        """q [R, H, dim] (RMS-normed per head, roped), window kv [R, dim] (normed, roped, FP8-rounded), qr."""
        rows = int(x.shape[0])
        xq = dense(x, self.x_proj, rows_exact)
        qr = rms_rows(xq[:, :self.q_rank], self.q_norm, self.eps, rows_exact)
        kv = norm_rope(xq[:, self.q_rank:], positions, self.inv_freq, weight=self.kv_norm, eps=self.eps)
        kv = quantize_cache(kv, 8, 32)
        q = dense(qr, self.wq_b, rows_exact).reshape(rows, self.heads, self.dim)
        # the official forward ropes wq_b's queries directly (no norm after wq_b; q_norm already
        # ran on wq_a's output above) -- reference runtime.py: q = rotate(linear(wq_b, qr))
        q = norm_rope(q, positions, self.inv_freq, norm=False)
        return q, kv, qr

    def back(self, o: mx.array, positions: mx.array, rows_exact: bool) -> mx.array:
        """Heads' outputs [R, H, dim]: inverse RoPE, the grouped low-rank wo_a, then wo_b."""
        rows = int(o.shape[0])
        o = norm_rope(o, positions, self.inv_freq, norm=False, inverse=True)
        o = o.reshape(rows, self.groups, -1)
        u = mx.concatenate([project(o[:, g], self.wo_a[g], rows_exact=rows_exact) for g in range(self.groups)],
                           axis=-1)
        return dense(u, self.wo_b, rows_exact)

    def __call__(self, x: mx.array, caches: list[Any], lengths: tuple[int, ...], decode: bool,
                 positions: mx.array | None = None) -> mx.array:
        """x [R, D] (attn-normed), rows of consecutive streams; each stream's layer cache takes its rows."""
        layer_caches = [c[self.layer] for c in caches]
        positions = positions_of(layer_caches, lengths) if positions is None else positions
        if decode:
            if self._front is None:
                self._front = mx.compile(lambda xs, ps: self.front(xs, ps, True))
                self._back = mx.compile(lambda os_, ps: self.back(os_, ps, True))
            q, kv, qr = self._front(x, positions)
            iq = iw = None
            if self.indexer is not None and self.ratio:
                need = any((c.offset + n) > self.topk * c.ratio for c, n in zip(layer_caches, lengths))
                if need:
                    iq, iw = self.indexer.queries(qr, x, positions, True)
        else:
            q, kv, qr = self.front(x, positions, False)
            iq = iw = None
            if self.indexer is not None and self.ratio:
                pooled = [(c.offset + n) // self.ratio for c, n in zip(layer_caches, lengths)]
                if any(p > self.topk for p in pooled):
                    iq, iw = self.indexer.queries(qr, x, positions, False)
        outs, at, whole = [], 0, len(lengths) == 1

        def part(a: mx.array | None, lo: int, n: int) -> mx.array | None:
            return a if whole or a is None else a[lo:lo + n]

        for stream, (cache, n) in enumerate(zip(layer_caches, lengths)):
            start = cache.offset
            pool_cache = caches[stream][self.head] if self.head is not None else None
            if self.compressor is not None and pool_cache is not None:
                self._compress(pool_cache, part(x, at, n), start)
            if decode:
                cache.write_keys(part(kv, at, n), start)
                outs.append(self._attend_rows(part(q, at, n), part(iq, at, n), part(iw, at, n),
                                              cache, pool_cache, start))
            else:
                outs.append(self._prefill_chunk(part(q, at, n), part(kv, at, n), iq, iw, cache, pool_cache,
                                                start, at))
                cache.write_keys(part(kv, at, n), start)
            cache.offset = start + n
            if pool_cache is not None and pool_cache is not cache:
                pool_cache.offset = start + n
            at += n
        o = mx.concatenate(outs) if len(outs) > 1 else outs[0]                 # [R, H, dim]
        return self._back(o, positions) if decode else self.back(o, positions, False)

    # -- the pool group's head: writing ----------------------------------------------------------------
    def _compress(self, pool_cache: LayerCache, x: mx.array, start: int) -> None:
        """The head's compressor: pool every block its rows complete and write its index keys."""
        comp = self.compressor
        r = self.ratio
        assert comp is not None
        rows = int(x.shape[0])
        if r == 1:
            latent = per_row(lambda t: comp.plain(t), x, True)
            # plain() already applied the compressor norm (the reference norms once, at the
            # latent); _pool_rows must not norm again -- a double norm broke every ratio-1
            # pool row from layer 20 on (the first ratio-1 layer of the big model).
            out = self._pool_rows(latent, mx.arange(start, start + rows), normed=True)
            pool_cache.write_pool(out, start)
            if self.indexer is not None and self.indexer_wk is not None:
                self._write_index_keys(pool_cache, latent, mx.arange(start, start + rows), start)
            return
        proj_v, proj_s = comp.front(x, True)                       # [R, dim] each, fp32, per row
        first, last = start // r, (start + rows) // r
        # every row's projections enter the ring: a rejected draft's pool rows are rebuilt from it on rollback
        pool_cache.write_proj(mx.concatenate([proj_v, proj_s], axis=-1), start)
        if last > first:
            count = last - first
            lo = first * r
            need = count * r
            if start > lo:
                keep_v, keep_s = pool_cache.proj_rows(lo, start)
                span_v = mx.concatenate([keep_v, proj_v])[:need]
                span_s = mx.concatenate([keep_s, proj_s])[:need]
            else:
                span_v, span_s = proj_v[:need], proj_s[:need]
            slots_kv = span_v.reshape(count, r, comp.dim)
            slots_sc = span_s.reshape(count, r, comp.dim)
            # each block pooled in its own call: a multi-block vectorized call is not
            # row-invariant on CPU (the serial path pools one block at a time)
            pooled = mx.concatenate([pool_blocks(slots_kv[b:b + 1], slots_sc[b:b + 1])
                                     for b in range(count)])                     # [count, dim] bf16
            # the reference norms the pooled latent once (runtime.py:364); the pool row is that
            # normed latent roped + FP4-rounded, and the indexer reads the SAME normed latent
            normed = rms_rows(pooled, comp.norm, self.eps, True)
            starts = mx.arange(first, last) * r
            pool_cache.write_pool(self._pool_rows(normed, starts, normed=True), first)
            if self.indexer is not None and self.indexer_wk is not None:
                self._write_index_keys(pool_cache, normed, starts, first)

    def _pool_rows(self, latent: mx.array, block_starts: mx.array, normed: bool = False) -> mx.array:
        """Pool rows roped at their block-start position, then the FP4 e4m3 g16 roundings, in place.

        ``normed``: the caller already RMS-normed the latent (the ratio-1 plain() path) — the
        reference norms exactly once.
        """
        assert self.compressor is not None
        normed_rows = latent if normed else rms_rows(latent, self.compressor.norm, self.eps)
        roped = norm_rope(normed_rows, block_starts, self.inv_freq, norm=False)
        return quantize_cache(roped, 4, 16, "e4m3")

    def _write_index_keys(self, pool_cache: LayerCache, latents: mx.array, block_starts: mx.array,
                          first: int) -> None:
        """The head's index keys: k_norm(wk(latent)) [dim], roped at block start, FP4 e8m0 g32 roundings.

        The latent arrives already normed (the reference's ``latent = self.norm(...)`` at
        runtime.py:364 feeds the indexer); ``wk`` is BF16, so the matmul runs in bf16 exactly as
        the reference's ``w.linear`` does (one bf16 rounding before k_norm).
        """
        assert self.indexer is not None and self.indexer_wk is not None
        wk = self.indexer_wk
        k = per_row(lambda t: t @ wk.T, latents, True)
        k = rms_rows(k, self.indexer_k_norm, self.eps)
        k = norm_rope(k, block_starts, self.inv_freq, norm=False)
        pool_cache.write_pool(quantize_cache(k, 4, 32), first, index=True)

    # -- reading: per-row selection, published per row --------------------------------------------------
    def _visible(self, position: int) -> int:
        return (position + 1) // self.ratio if self.ratio else 0

    def _selection(self, iq: mx.array, iw: mx.array, ends: list[int], pool_cache: LayerCache,
                   visible: list[int]) -> tuple[mx.array | None, mx.array | None]:
        """The rows' chosen pool rows [R, topk] (ascending) and the candidate mask, per row exactly."""
        if self.indexer is None or max(visible, default=0) <= self.topk:
            return None, None
        cands = None
        scores = self.indexer.scores(iq, iw, pool_cache.ipool[:max(visible)])
        seen = mx.arange(max(visible))[None, :] < mx.array(visible)[:, None]
        scores = mx.where(seen, scores, float("-inf"))
        if self.candidate_source:
            cands = mx.concatenate([candidate_mask(scores[r], visible[r], self.candidate_block,
                                                   self.candidate_topk)[None, :] for r in range(len(visible))])
        elif self.candidate_after and pool_cache.candidates is not None:
            # level two: our own weights, inside the source's candidate blocks
            scores = mx.where(pool_cache.candidates[:max(visible)], scores, float("-inf"))
        picks = mx.sort(mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., :self.topk], axis=-1)
        return picks.astype(mx.int32), cands

    def _chosen_row(self, pick: mx.array | None, i: int, pool_cache: LayerCache) -> list[int] | None:
        """Row i's chosen pool rows: its own selection, else (Reuse) the group's latest published one."""
        if pick is not None:
            return [int(c) for c in pick[i].tolist()]
        if self.mode == "reuse" and pool_cache.topk_idxs is not None and i < len(pool_cache.topk_idxs):
            return [int(c) for c in pool_cache.topk_idxs[i].tolist()]
        return None

    def _attend_rows(self, q: mx.array, iq: Any, iw: Any, cache: LayerCache,
                     pool_cache: LayerCache | None, start: int) -> mx.array:
        """This stream's rows over their windows (then pool rows when selected), published for the group."""
        rows = int(q.shape[0])
        ends = [start + i for i in range(rows)]
        visible = [self._visible(p) for p in ends]
        pick = cands = None
        if pool_cache is not None and self.indexer is not None:
            pick, cands = self._selection(iq, iw, ends, pool_cache, visible)
            pool_cache.publish(pick, cands)
        outs = []
        for i in range(rows):
            parts = [cache.window_keys(ends[i])]
            if self.ratio and pool_cache is not None:
                chosen = self._chosen_row(pick, i, pool_cache)
                if chosen is not None:
                    # a row with visible <= topk gets topk picks back, the tail of them -inf junk
                    # beyond its visibility (a stale pool row): drop them, keeping the ascending order
                    chosen = [c for c in chosen if c < visible[i]]
                    if chosen:
                        assert pool_cache.pool is not None
                        parts.append(pool_cache.pool[mx.array(chosen)])
                elif visible[i]:
                    assert pool_cache.pool is not None
                    parts.append(pool_cache.pool[:visible[i]])
            keys = mx.concatenate(parts) if len(parts) > 1 else parts[0]
            outs.append(sparse_attention(q[i], keys, self.sink)[None])
        return mx.concatenate(outs)

    def _prefill_chunk(self, q: mx.array, kv: mx.array, iq: Any, iw: Any, cache: LayerCache,
                       pool_cache: LayerCache | None, start: int, at: int) -> mx.array:
        """A prompt chunk's rows over the pool, the window before the chunk and the chunk itself, per row."""
        rows = int(q.shape[0])
        lo = max(0, start - (self.window - 1))
        window = [cache.ring_rows(cache.keys, lo, start), kv] if start > lo else [kv]
        pooled = (start + rows) // self.ratio if self.ratio else 0
        t = mx.arange(lo, start + rows)
        outs = []
        step = 256
        for q0 in range(0, rows, step):
            q1 = min(rows, q0 + step)
            pos = mx.arange(start + q0, start + q1)[:, None]
            mask = (t[None, :] <= pos) & (t[None, :] > pos - self.window)
            keys = window
            if pooled and pool_cache is not None:
                visible = (mx.arange(pooled)[None, :] + 1) * self.ratio - 1 <= pos
                top = self.topk if self.indexer is not None else pooled
                if self.indexer is None and self.mode == "reuse" and pool_cache.topk_idxs is not None:
                    # a Reuse layer reads the group's published selection (an index source of this step)
                    published = pool_cache.topk_idxs
                    if q0 < len(published):
                        rows_idx = published[q0:q1].astype(mx.int32)
                        sel = mx.zeros((q1 - q0, pooled), dtype=mx.bool_)
                        sel = mx.put_along_axis(sel, rows_idx, mx.array(True), axis=-1)
                        visible = visible & sel
                elif pooled > top:
                    assert self.indexer is not None and pool_cache is not None
                    scores = self.indexer.scores(iq[at + q0:at + q1], iw[at + q0:at + q1],
                                                 pool_cache.ipool[:pooled])
                    scores = mx.where(visible, scores, float("-inf"))
                    if self.candidate_source:
                        cands = mx.concatenate(
                            [candidate_mask(scores[r], int(visible[r].sum().item()), self.candidate_block,
                                            self.candidate_topk)[None, :]
                             for r in range(int(scores.shape[0]))])
                        pool_cache.publish(None, cands)
                    elif self.candidate_after and pool_cache.candidates is not None:
                        scores = mx.where(pool_cache.candidates[:pooled], scores, float("-inf"))
                    best = mx.sort(mx.argpartition(-scores, kth=top - 1, axis=-1)[..., :top], axis=-1)
                    if self.mode != "reuse":
                        pool_cache.publish(best.astype(mx.int32), None)   # Reuse layers read this unchanged
                    chosen = mx.put_along_axis(mx.zeros(visible.shape, dtype=mx.bool_), best,
                                               mx.array(True), axis=-1)
                    visible = visible & chosen
                keys = window + [pool_cache.pool[:pooled]]     # window first, then pool (the official order)
                mask = mx.concatenate([mask, visible], axis=1)
            else:
                keys = window
            keys_all = mx.concatenate(keys) if len(keys) > 1 else keys[0]
            outs.append(self._attend_masked(q[q0:q1], keys_all, mask))
        return mx.concatenate(outs) if len(outs) > 1 else outs[0]

    def _attend_masked(self, q: mx.array, keys: mx.array, mask: mx.array | None) -> mx.array:
        """A chunk's rows over keys with a causal-window mask, the reference's fp32 arithmetic."""
        rows, heads_n = int(q.shape[0]), int(q.shape[1])
        scores = mx.matmul(q.astype(mx.float32), mx.swapaxes(keys.astype(mx.float32), -1, -2)[None]) * self.scale
        scores = mx.where(mask[:, None, :], scores, float("-inf")) if mask is not None else scores
        sinks = mx.broadcast_to(self.sink[None, :, None], (rows, heads_n, 1))
        p = mx.softmax(mx.concatenate((scores, sinks), axis=-1), axis=-1)
        return mx.matmul(p[..., :-1], keys.astype(mx.float32)[None]).astype(q.dtype)     # [L, H, D]
