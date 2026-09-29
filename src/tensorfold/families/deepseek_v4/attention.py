"""DeepSeek-V4's attention: 64 query heads over one shared 512-dim key/value head, a 128-token window and pools."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.caches import LayerCache
from tensorfold.families.deepseek_v4.compressor import Compressor, Indexer, norm_rope
from tensorfold.families.deepseek_v4.config import PREFILL_QUERIES, Config, row_kernel
from tensorfold.families.deepseek_v4.dense import dense
from tensorfold.families.glm5_next.linear import Q, per_row, project
from tensorfold.kernels.deepseek.v4 import attention as ATT
from tensorfold.kernels.deepseek.v4 import rows as RK


def positions_of(caches: list[LayerCache], lengths: tuple[int, ...]) -> mx.array:
    """Each row's position: the streams' rows from their caches' offsets, in order."""

    parts = [mx.arange(c.offset, c.offset + n) for c, n in zip(caches, lengths)]
    return mx.concatenate(parts) if len(parts) > 1 else parts[0]


class Attention:
    """wq_a | wkv, per-head RMS queries, RoPE on the last 64 dims, sinks, inverse RoPE, grouped low-rank output."""

    def __init__(self, w: dict[str, Any], cfg: Config, layer: int) -> None:
        self.ratio = cfg.ratio(layer)
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
        self.sink = w["attn_sink"].astype(mx.bfloat16)              # the standard hands MLX's SDPA bf16 sinks
        self.sink32 = w["attn_sink"].astype(mx.float32)             # the decode kernel's, as DeepSeek's code keeps it
        self.compressor: Compressor | None = w.get("compressor")
        self.indexer: Indexer | None = w.get("indexer")
        self.cproj = self._stack_compressors()
        self._compiled: tuple[Any, ...] | None = None

    def _stack_compressors(self) -> Any:
        """Every compressor's wkv | wgate as one fp32 projection; each reads its columns from ``col``."""

        comps = self.compressors()
        if not comps:
            return None
        stacked = Q.stack([c.proj for c in comps]) if len(comps) > 1 else comps[0].proj
        col = 0
        for k, comp in enumerate(comps):
            comp.col, comp.index, comp.proj = col, k == 1, None
            col += 2 * comp.width
        return stacked

    def compressors(self) -> list[Compressor]:
        return [c for c in (self.compressor, self.indexer.compressor if self.indexer else None) if c is not None]

    def front(self, x: mx.array, positions: mx.array, rows_exact: bool) -> tuple[mx.array, ...]:
        """What a layer computes from its rows before the caches: q, kv, qr, then each compressor's projections."""

        rows = int(x.shape[0])
        xq = dense(x, self.x_proj, rows_exact)
        qr = mx.fast.rms_norm(xq[:, :self.q_rank], self.q_norm, self.eps)
        kv = norm_rope(xq[:, self.q_rank:], positions, self.inv_freq, weight=self.kv_norm, eps=self.eps)
        q = dense(qr, self.wq_b, rows_exact).reshape(rows, self.heads, self.dim)
        q = norm_rope(q, positions, self.inv_freq, eps=self.eps)
        out = [q, kv, qr]
        if self.cproj is not None:
            if row_kernel("compressor", max(rows, 2), rows_exact) and RK.f32_rows_fits(self.cproj, rows):
                out.append(RK.qmv_rows_f32(x, self.cproj))              # bf16 in: its fp32 view loses nothing
            else:
                out.append(per_row(self.cproj, x.astype(mx.float32), rows_exact))
        return tuple(out)

    def back(self, o: mx.array, positions: mx.array, rows_exact: bool, rotated: bool) -> mx.array:
        """Heads' outputs [R, H, dim]: inverse RoPE (unless the kernel did it), the grouped low-rank wo_a, then wo_b."""

        rows = int(o.shape[0])
        if not rotated:
            o = norm_rope(o, positions, self.inv_freq, norm=False, inverse=True)
        o = o.reshape(rows, self.groups, -1)
        if rows_exact and RK.grouped_fits(self.wo_a_all, self.groups, max(rows, 2)):
            flat = o.reshape(rows, -1)
            u = (RK.grouped_one_row(flat, self.wo_a_all, self.groups) if rows == 1
                 else RK.qmv_rows_grouped(flat, self.wo_a_all, self.groups))
        else:
            u = mx.concatenate([project(o[:, g], self.wo_a[g], rows_exact=rows_exact) for g in range(self.groups)],
                               axis=-1)
        return dense(u, self.wo_b, rows_exact)

    def _decode_fns(self, rotated: bool) -> tuple[Any, Any]:
        """``front`` and ``back`` for decode windows, compiled (one trace a row count)."""

        if self._compiled is None:
            self._compiled = (mx.compile(lambda x, p: self.front(x, p, True)),
                              mx.compile(lambda o, p: self.back(o, p, True, False)),
                              mx.compile(lambda o, p: self.back(o, p, True, True)))
        return self._compiled[0], self._compiled[1 + int(rotated)]

    def __call__(self, x: mx.array, caches: list[LayerCache], lengths: tuple[int, ...], decode: bool,
                 positions: mx.array | None = None) -> mx.array:
        """x [R, D] (attn-normed), rows of consecutive streams; each stream's cache takes its rows."""

        positions = positions_of(caches, lengths) if positions is None else positions
        rotated = ATT.fits(self.heads, self.dim)
        if decode:
            front, back = self._decode_fns(rotated)
            q, kv, qr, *proj = front(x, positions)
        else:
            q, kv, qr, *proj = self.front(x, positions, False)
        proj = proj[0] if proj else None
        iq = iw = None
        if self.indexer is not None and any((c.offset + n) // self.ratio > self.indexer.topk
                                            for c, n in zip(caches, lengths)):
            iq, iw = self.indexer.queries(qr, x, positions, decode)
        outs, at, whole = [], 0, len(lengths) == 1

        def part(a: mx.array, lo: int, n: int) -> mx.array:
            return a if whole else a[lo:lo + n]

        for cache, n in zip(caches, lengths):
            start = cache.offset
            if proj is not None:
                self._compress(cache, part(proj, at, n), start)
            if decode:
                cache.write_keys(part(kv, at, n), start)
                if rotated:
                    outs.append(self._rows(part(q, at, n), iq, iw, at, cache, start))
                else:
                    outs += [self._row(q[i], iq, iw, i, cache, start + i - at) for i in range(at, at + n)]
            else:
                prefill = self._prefill_rows if rotated else self._prefill
                outs.append(prefill(part(q, at, n), part(kv, at, n), iq, iw, cache, start, at))
                cache.write_keys(part(kv, at, n), start)
            cache.offset = start + n
            at += n
        o = mx.concatenate(outs) if len(outs) > 1 else outs[0]                 # [R, H, dim]
        return back(o, positions) if decode else self.back(o, positions, False, rotated)

    def _compress(self, cache: LayerCache, proj: mx.array, start: int) -> None:
        """Keep rows' compressor projections (positions start ..) and emit every pool row they complete."""

        rows, r = int(proj.shape[0]), self.ratio
        lo = max(0, (start // r - int(r == 4)) * r)                 # the first position a completed block reads
        span = None
        if (start + rows) // r > start // r and start + rows - cache.proj_ring() > lo:     # more than the ring keeps
            span = (mx.concatenate([cache.proj_rows(lo, start), proj]) if start > lo else proj, lo)
        cache.write_proj(proj, start)
        for comp in self.compressors():
            comp.emit(cache, start, rows, span)

    def _attend(self, q: mx.array, keys: mx.array, mask: mx.array | None = None) -> mx.array:
        """q [L, H, dim] over keys [N, dim] (values = keys) with the sinks: [L, H, dim]."""

        k = keys[None, None]
        out = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k, k, scale=self.scale, mask=mask,
                                                   sinks=self.sink)
        return out[0].transpose(1, 0, 2)

    def _chosen(self, iq: Any, iw: Any, i: int, cache: LayerCache, visible: int) -> mx.array | None:
        """Row i's top pool rows (ascending) once it sees more than the indexer keeps, else None (all visible)."""

        if self.indexer is None or visible <= self.indexer.topk:
            return None
        scores = self.indexer.scores(iq[i:i + 1], iw[i:i + 1], cache.ipool[:visible])[0]
        return self.indexer.select(scores).astype(mx.int32)

    def _rows(self, q: mx.array, iq: Any, iw: Any, at: int, cache: LayerCache, start: int) -> mx.array:
        """A stream's decode rows in one kernel call, each over its own pool rows and window (one row: the same)."""

        counts, chosen, windows = [], [], []
        for i in range(int(q.shape[0])):
            position = start + i
            visible = (position + 1) // self.ratio if self.ratio else 0
            pick = self._chosen(iq, iw, at + i, cache, visible)
            counts.append(visible if pick is None else self.indexer.topk)
            chosen.append(pick)
            windows.append((max(0, position - self.window + 1), position))
        pidx = None
        if any(p is not None for p in chosen):
            width = max(counts)
            pidx = mx.stack([mx.pad(p if p is not None else mx.arange(c, dtype=mx.int32), (0, width - c))
                             for p, c in zip(chosen, counts)])
        return ATT.attend_rows(q, cache.pool if self.ratio else None, pidx, counts, cache.keys, windows, self.sink32,
                               self.scale, self.inv_freq)

    def _row(self, q: mx.array, iq: Any, iw: Any, i: int, cache: LayerCache, position: int) -> mx.array:
        """One decode row at ``position``: its visible (or chosen) pool rows, then its window, in order."""

        parts = []
        if self.ratio:
            visible = (position + 1) // self.ratio
            pick = self._chosen(iq, iw, i, cache, visible)
            if pick is not None:
                parts.append(cache.pool[pick])
            elif visible:
                parts.append(cache.pool[:visible])
        parts.append(cache.window_keys(position))
        keys = mx.concatenate(parts) if len(parts) > 1 else parts[0]
        return self._attend(q[None], keys)

    def _prefill_rows(self, q: mx.array, kv: mx.array, iq: Any, iw: Any, cache: LayerCache, start: int,
                      at: int) -> mx.array:
        """A prompt chunk through the decode kernel: each row over its visible (or top-k) pool rows and its window."""

        rows = int(q.shape[0])
        lo = max(0, start - (self.window - 1))
        keys = mx.concatenate([cache.ring_rows(cache.keys, lo, start), kv]) if start > lo else kv
        windows = [(max(lo, p - self.window + 1) - lo, p - lo) for p in range(start, start + rows)]
        visible = [(p + 1) // self.ratio if self.ratio else 0 for p in range(start, start + rows)]
        top = self.indexer.topk if self.indexer is not None else None
        pidx = None
        if top is not None and visible[-1] > top:
            pooled, parts = visible[-1], []
            step = max(16, min(PREFILL_QUERIES, (1 << 27) // (self.indexer.heads * pooled)))
            for q0 in range(0, rows, step):
                q1 = min(rows, q0 + step)
                seen = mx.arange(pooled)[None, :] < mx.array(visible[q0:q1])[:, None]
                scores = self.indexer.scores(iq[at + q0:at + q1], iw[at + q0:at + q1], cache.ipool[:pooled])
                scores = mx.where(seen, scores, -mx.inf)
                parts.append(mx.sort(mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top], axis=-1))
            pidx = (mx.concatenate(parts) if len(parts) > 1 else parts[0]).astype(mx.int32)
            visible = [min(v, top) for v in visible]
        return ATT.attend_rows(q, cache.pool if self.ratio else None, pidx, visible, keys, windows, self.sink32,
                               self.scale, self.inv_freq, lo)

    def _prefill(self, q: mx.array, kv: mx.array, iq: Any, iw: Any, cache: LayerCache, start: int,
                 at: int) -> mx.array:
        """A prompt chunk's rows over the pool, the window before the chunk and the chunk itself, masked per row."""

        rows = int(q.shape[0])
        lo = max(0, start - (self.window - 1))
        keys = [cache.ring_rows(cache.keys, lo, start), kv] if start > lo else [kv]
        pooled = (start + rows) // self.ratio if self.ratio else 0
        if pooled:
            keys = [cache.pool[:pooled]] + keys
        keys = mx.concatenate(keys) if len(keys) > 1 else keys[0]
        t = mx.arange(lo, start + rows)
        outs = []
        for q0 in range(0, rows, PREFILL_QUERIES):
            q1 = min(rows, q0 + PREFILL_QUERIES)
            pos = mx.arange(start + q0, start + q1)[:, None]
            mask = (t[None, :] <= pos) & (t[None, :] > pos - self.window)
            if pooled:
                visible = (mx.arange(pooled)[None, :] + 1) * self.ratio - 1 <= pos
                top = self.indexer.topk if self.indexer is not None else pooled
                if pooled > top:
                    scores = self.indexer.scores(iq[at + q0:at + q1], iw[at + q0:at + q1], cache.ipool[:pooled])
                    scores = mx.where(visible, scores, float("-inf"))
                    best = mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top]
                    chosen = mx.put_along_axis(mx.zeros(visible.shape, dtype=mx.bool_), best,
                                               mx.array(True), axis=-1)
                    visible = visible & chosen
                mask = mx.concatenate([visible, mask], axis=1)
            outs.append(self._attend(q[q0:q1], keys, mask))
        return mx.concatenate(outs) if len(outs) > 1 else outs[0]
