"""DeepSeek sparse attention: NoPE MLA over a 512-wide latent, with the pooled-block indexer."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.caches import MLACache
from tensorfold.families.glm5_next.config import Config, row_kernel
from tensorfold.families.glm5_next.linear import Q, _rows, per_row, project
from tensorfold.kernels.glm.flash.v1 import kernels as K
from tensorfold.kernels.glm.flash.v1 import prompt as PK
from tensorfold.kernels.glm.flash.v1 import sparse_attention as SA


# prompt queries attended at once, each over the ~2,051 keys it gathers from the latent cache
PREFILL_QUERIES = 512


class MLA:
    """DeepSeek sparse attention, NoPE MLA over a 512-wide latent, with the pooled-block indexer."""

    def __init__(self, w: dict[str, Any], cfg: Config) -> None:
        self.cfg = cfg
        self.heads = cfg.num_attention_heads
        self.nope, self.vdim, self.rank = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.scale = self.nope ** -0.5
        self.q_a, self.q_b, self.kv_a, self.o_proj = w["q_a_proj"], w["q_b_proj"], w["kv_a_proj_with_mqa"], w["o_proj"]
        self.q_norm, self.kv_norm = w["q_a_layernorm"], w["kv_a_layernorm"]
        if "kv_b_proj" in w:
            kvb: Q = w["kv_b_proj"]                                      # [H (nope + v), rank]
            per = self.nope + self.vdim

            def heads(a: mx.array, lo: int, hi: int) -> mx.array:
                return mx.contiguous(a.reshape(self.heads, per, -1)[:, lo:hi])

            # keys [H, nope, rank] quantized along rank: absorb = q @ wk (transpose False), keys = lat @ wk^T
            self.wk = Q(heads(kvb.weight, 0, self.nope), heads(kvb.scales, 0, self.nope),
                        heads(kvb.biases, 0, self.nope), bits=kvb.bits, group=kvb.group)
            self.wv = Q(heads(kvb.weight, self.nope, per), heads(kvb.scales, self.nope, per),
                        heads(kvb.biases, self.nope, per), bits=kvb.bits, group=kvb.group)
            self.wk_t = False
        else:
            # mlx-lm's absorbed pair: embed_q [H, rank, nope] (kv_b's key half transposed), unembed_out [H, v, rank]
            self.wk, self.wv = w["embed_q"], w["unembed_out"]
            self.wk_t = True
        # indexer
        self.iq, self.ik_proj, self.iw = w["indexer.wq_b"], w["indexer.wk"], w["indexer.weights_proj"]
        self.ik_norm_w, self.ik_norm_b = w["indexer.k_norm.weight"], w["indexer.k_norm.bias"]
        self.ape = w["indexer.index_kpool_compress_ape"]
        self.igate = mx.contiguous(w["indexer.index_kpool_compress_gate"].T)  # [D, 128]
        self.i_heads, self.i_dim = cfg.index_n_heads, cfg.index_head_dim
        self.i_scale = (self.i_heads ** -0.5) * (self.i_dim ** -0.5)
        # projections of one input stacked (a one-row matmul's bits don't change); the names stay as row ranges
        self.x_proj = Q.stack([self.q_a, self.kv_a, self.ik_proj, self.iw])
        self.qr_proj = Q.stack([self.q_b, self.iq])
        at = [0]
        for p in (self.q_a, self.kv_a, self.ik_proj, self.iw):
            at.append(at[-1] + p.outs)
        self.x_cuts = at
        self.q_a, self.kv_a, self.ik_proj, self.iw = (_rows(self.x_proj, at[i], at[i + 1]) for i in range(4))
        nq = self.q_b.outs
        self.q_b, self.iq = _rows(self.qr_proj, 0, nq), _rows(self.qr_proj, nq, self.qr_proj.outs)

    # the per-head latent maps (kv_b in its stored layout)
    def absorb(self, q: mx.array) -> mx.array:
        """q_nope [H, n, nope] -> latent queries [H, n, rank]."""

        wk = self.wk
        return mx.quantized_matmul(q, wk.weight, wk.scales, wk.biases, transpose=self.wk_t, group_size=wk.group,
                                   bits=wk.bits).astype(q.dtype)

    def unabsorb(self, out: mx.array) -> mx.array:
        """latent outputs [H, n, rank] -> values [H, n, v]."""

        wv = self.wv
        return mx.quantized_matmul(out, wv.weight, wv.scales, wv.biases, transpose=True, group_size=wv.group,
                                   bits=wv.bits).astype(out.dtype)

    def index_scores(self, iq: mx.array, iw: mx.array, pool: mx.array) -> mx.array:
        """Block scores [n, P] = sum over indexer heads of w_h relu(q_h . pool) (iq [n, HI, DI], iw [n, HI])."""

        s = iq @ pool.T                                                  # [n, HI, P]
        return mx.sum(iw[..., None] * mx.maximum(s, mx.array(0, s.dtype)), axis=1)

    def prompt_scores(self, iq: mx.array, iw: mx.array, pool: mx.array) -> mx.array:
        """``index_scores`` for a prompt's queries: one fused kernel with the three ops' bits where it fits."""

        if "dsa" in C.FUSED and PK.index_fits(iq, pool):
            return PK.index_scores(iq, iw, pool)
        return self.index_scores(iq, iw, pool)

    def selected(self, scores: mx.array, position: int) -> mx.array:
        """Key ids of one query at ``position`` past ``index_topk`` keys: its best blocks' keys, then its tail."""

        cfg = self.cfg
        kp = cfg.index_kpool
        scores = scores.reshape(-1)
        blocks = int(scores.shape[0])
        top = min(cfg.index_topk // kp, blocks)
        pick = mx.argpartition(-scores, kth=top - 1)[:top]
        ids = (pick[:, None] * kp + mx.arange(kp)[None]).reshape(-1)
        tail = (position + 1) % kp
        if cfg.index_tail and tail:
            ids = mx.concatenate([ids, mx.arange(position + 1 - tail, position + 1)])
        return ids

    def __call__(self, x: mx.array, caches: list[MLACache], lengths: tuple[int, ...], decode: bool) -> mx.array:
        """Consecutive streams' rows: projections on all rows at once, each stream's rows over its own cache."""

        cfg = self.cfg
        rows = int(x.shape[0])
        H = self.heads
        c = self.x_cuts
        if decode:
            # the stacked matrices (prefill keeps the separate ones: MLX's batched kernel tiles by width)
            xp = project(x, self.x_proj, rows_exact=True)               # q_a | kv_a | indexer k | indexer weights
            parts = [xp[:, c[i]:c[i + 1]] for i in range(4)]
        else:
            parts = [project(x, p, rows_exact=False) for p in (self.q_a, self.kv_a, self.ik_proj, self.iw)]
        qr = mx.fast.rms_norm(parts[0], self.q_norm, cfg.rms_norm_eps)
        if decode:
            qp = project(qr, self.qr_proj, rows_exact=True)             # q_b | indexer q
            q, iq = qp[:, :self.q_b.outs], qp[:, self.q_b.outs:]
        else:
            q, iq = project(qr, self.q_b, rows_exact=False), project(qr, self.iq, rows_exact=False)
        q = q.reshape(rows, H, self.nope)
        lat = mx.fast.rms_norm(parts[1], self.kv_norm, cfg.rms_norm_eps)
        iq = iq.reshape(rows, self.i_heads, self.i_dim)
        ik = mx.fast.layer_norm(parts[2], self.ik_norm_w, self.ik_norm_b, 1e-6)
        if row_kernel("igate", rows, decode):
            ig = K.matmul_rows(x, self.igate, transposed=True)
        else:
            ig = per_row(lambda r: r @ self.igate, x, decode)
        iw = (parts[3] * self.i_scale).astype(C.act())
        batched = decode and row_kernel("mla_proj", rows, decode)
        if batched:
            # the latent maps with the rows as a batch (each keeps its one-row bits), attention row by row
            ql = mx.quantized_matmul(q[:, :, None, :], self.wk.weight, self.wk.scales, self.wk.biases,
                                     transpose=self.wk_t, group_size=self.wk.group,
                                     bits=self.wk.bits).astype(q.dtype)  # [R, H, 1, rank]
        outs, at = [], 0
        for cache, n in zip(caches, lengths):
            one = len(lengths) == 1
            span = slice(at, at + n)
            start = cache.offset
            cache.append(lat if one else lat[span], ik if one else ik[span], ig if one else ig[span], self.ape,
                         cfg.index_kpool)
            if batched:
                outs.append(self._attend_rows(ql if one else ql[span], iq if one else iq[span],
                                              iw if one else iw[span], cache, start))
            elif decode:
                outs += [self._decode_row(q[at + r], iq[at + r], iw[at + r], cache, start + r) for r in range(n)]
            else:
                outs.append(self._prefill(q if one else q[span], iq if one else iq[span], iw if one else iw[span],
                                          cache, start))
            at += n
        if batched:
            att = outs[0] if len(outs) == 1 else mx.concatenate(outs)
            wv = self.wv
            out = mx.quantized_matmul(att, wv.weight, wv.scales, wv.biases, transpose=True, group_size=wv.group,
                                      bits=wv.bits).astype(att.dtype).reshape(rows, -1)
        else:
            out = outs[0] if len(outs) == 1 else mx.concatenate(outs)
        return project(out, self.o_proj, rows_exact=decode)

    def _attend_rows(self, ql: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, start: int) -> mx.array:
        """One stream's rows (latent queries from ``start``) over its cache, keys chosen as its own call chooses."""

        cfg = self.cfg
        rows = int(ql.shape[0])
        sparse = [r for r in range(rows) if C.SPARSE_KERNEL and start + r + 1 > cfg.index_topk]
        if row_kernel("indexer", rows, True):
            sels = self._choices(iq, iw, cache, start, skip=sparse)
        else:
            sels = [...] * rows
        parts: list[Any] = [None] * rows
        if sparse:
            at = mx.array(sparse)
            idx = self._sparse_indices(iq[at], iw[at], cache, [start + r for r in sparse])
            got = SA.indexed_attention(ql[at][:, :, 0, :], cache.keys, idx, cache.offset, self.scale)
            for i, r in enumerate(sparse):
                parts[r] = got[i][None, :, None, :]
        for r in range(rows):
            if parts[r] is None:
                parts[r] = self._attend(ql[r], iq[r], iw[r], cache, start + r, sels[r])
        return mx.concatenate(parts) if rows > 1 else parts[0]

    def _sparse_indices(self, iq: mx.array, iw: mx.array, cache: MLACache, positions: list[int]) -> mx.array:
        """Key ids of rows past ``index_topk`` for the sparse kernel (-1 padded), chosen as ``_choices`` chooses."""

        cfg = self.cfg
        kp = cfg.index_kpool
        width = cfg.index_topk + (kp - 1 if cfg.index_tail else 0)
        out: list[Any] = [None] * len(positions)
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(positions):
            groups.setdefault((p + 1) // kp, []).append(i)
        for blocks, members in groups.items():
            at = mx.array(members) if len(members) < len(positions) else None
            scores = self.index_scores(iq if at is None else iq[at], iw if at is None else iw[at], cache.pool[:blocks])
            top = min(cfg.index_topk // kp, blocks)
            picks = mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top]
            ids = (picks[:, :, None] * kp + mx.arange(kp)[None, None]).reshape(len(members), -1).astype(mx.int32)
            tails = []
            for i in members:
                p = positions[i]
                t = (p + 1) % kp if cfg.index_tail else 0
                tails.append([p + 1 - t + j for j in range(t)] + [-1] * (width - top * kp - t))
            if width > top * kp:
                ids = mx.concatenate([ids, mx.array(tails, dtype=mx.int32)], axis=1)
            for j, i in enumerate(members):
                out[i] = ids[j:j + 1]
        return mx.concatenate(out) if len(out) > 1 else out[0]

    def _choices(self, iq: mx.array, iw: mx.array, cache: MLACache, start: int, skip: Any = ()) -> list[Any]:
        """Each window row's key ids (None: all), rows sharing a block count ranked together, each keeping its bits."""

        cfg = self.cfg
        kp = cfg.index_kpool
        rows = int(iq.shape[0])
        out: list[Any] = [None] * rows
        groups: dict[int, list[int]] = {}
        for r in range(rows):
            if start + r + 1 > cfg.index_topk and r not in skip:
                groups.setdefault((start + r + 1) // kp, []).append(r)
        for blocks, members in groups.items():
            at = mx.array(members)
            scores = self.index_scores(iq[at], iw[at], cache.pool[:blocks])          # [m, P]
            top = min(cfg.index_topk // kp, blocks)
            picks = mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top]
            for i, r in enumerate(members):
                ids = (picks[i][:, None] * kp + mx.arange(kp)[None]).reshape(-1)
                position = start + r
                tail = (position + 1) % kp
                if cfg.index_tail and tail:
                    ids = mx.concatenate([ids, mx.arange(position + 1 - tail, position + 1)])
                out[r] = ids
        return out

    def _attend(self, ql: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, position: int,
                sel: Any = ...) -> mx.array:
        """One query at ``position`` over its keys (``sel``: ids chosen already, None for all): [1, H, 1, rank]."""

        cfg = self.cfg
        n = position + 1
        keys = cache.keys[:n]
        if sel is not ...:
            if sel is not None:
                keys = mx.take(keys, sel, axis=0)
        elif n > cfg.index_topk:
            blocks = n // cfg.index_kpool
            scores = self.index_scores(iq[None], iw[None], cache.pool[:blocks])
            keys = mx.take(keys, self.selected(scores, position), axis=0)
        return mx.fast.scaled_dot_product_attention(ql[None], keys[None, None], keys[None, None], scale=self.scale)

    def _decode_row(self, q: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, position: int) -> mx.array:
        """One query at ``position`` (q [H, nope]) over its own keys: absorbed attention on the latents."""

        ql = self.absorb(q[:, None, :])                                  # [H, 1, rank]
        if C.SPARSE_KERNEL and position + 1 > self.cfg.index_topk:
            idx = self._sparse_indices(iq[None], iw[None], cache, [position])
            out = SA.indexed_attention(ql[:, 0, :][None], cache.keys, idx, cache.offset, self.scale)[:, :, None, :]
        else:
            out = self._attend(ql, iq, iw, cache, position)
        return self.unabsorb(out[0]).reshape(1, -1)                      # [1, H v]

    def _prefill(self, q: mx.array, iq: mx.array, iw: mx.array, cache: MLACache, start: int) -> mx.array:
        """Prompt rows as decode attends: absorbed queries over their own keys in the latent cache, flat in context."""

        cfg = self.cfg
        rows = int(q.shape[0])
        kp = cfg.index_kpool
        width = cfg.index_topk + (kp - 1 if cfg.index_tail else 0)
        rank = int(cache.keys.shape[1])
        ql_all = self.absorb(q.transpose(1, 0, 2))                       # [H, rows, rank]
        chunk = PREFILL_QUERIES
        outs = []
        for c0 in range(0, rows, chunk):
            c1 = min(c0 + chunk, rows)
            c = c1 - c0
            last = start + c1                                           # keys this chunk may read: [0, last)
            if last <= cfg.index_topk:
                # every query reads its whole causal prefix: one causal attention over the latent, as decode attends
                keys = cache.keys[:last][None, None]
                o = mx.fast.scaled_dot_product_attention(ql_all[None, :, c0:c1], keys, keys, scale=self.scale,
                                                         mask="causal")
                outs.append(o[0].transpose(1, 0, 2))                    # [c, H, rank]
                continue
            pos = mx.arange(start + c0, start + c1)                     # query positions
            blocks = last // kp
            dense = pos + 1 <= cfg.index_topk                           # queries that read all their keys
            if last > cfg.index_topk:
                scores = self.prompt_scores(iq[c0:c1], iw[c0:c1], cache.pool[:blocks])     # [c, P]
                valid = (mx.arange(blocks)[None] * kp + kp - 1) <= pos[:, None]
                scores = mx.where(valid, scores, mx.array(-1e30, scores.dtype))
                top = min(cfg.index_topk // kp, blocks)
                pick = mx.argpartition(-scores, kth=top - 1, axis=-1)[..., :top]         # [c, top]
                picked_valid = mx.take_along_axis(valid, pick, axis=-1)
                ids = (pick[:, :, None] * kp + mx.arange(kp)[None, None]).reshape(c, -1)  # [c, top kp]
                ids = mx.where(mx.repeat(picked_valid, kp, axis=1), ids, -1)
                if cfg.index_tail:
                    tail_start = pos + 1 - (pos + 1) % kp
                    tail = tail_start[:, None] + mx.arange(kp - 1)[None]                  # [c, kp - 1]
                    ids = mx.concatenate([ids, mx.where(tail <= pos[:, None], tail, -1)], axis=1)
                if int(ids.shape[1]) < width:
                    ids = mx.concatenate([ids, mx.full((c, width - int(ids.shape[1])), -1, dtype=ids.dtype)], axis=1)
                every = mx.arange(width)[None]
                ids = mx.where(dense[:, None], mx.where(every <= pos[:, None], every, -1), ids)
            else:
                every = mx.arange(min(width, last))[None]
                ids = mx.where(every <= pos[:, None], every, -1)
            valid_sel = ids >= 0
            w = int(ids.shape[1])
            keys = mx.take(cache.keys[:last], mx.where(valid_sel, ids, 0).reshape(-1), axis=0).reshape(c, w, rank)
            # bf16 scores (the power-of-two scale in the queries), a precise softmax: fp32's bits, a third the traffic
            ql = ql_all[:, c0:c1].transpose(1, 0, 2) * self.scale        # [c, H, rank]
            s = mx.matmul(ql, keys.transpose(0, 2, 1))                    # [c, H, w]
            s = mx.where(valid_sel[:, None, :], s, mx.array(-1e30, s.dtype))
            outs.append(mx.matmul(mx.softmax(s, axis=-1, precise=True), keys))   # [c, H, rank]
        att = mx.concatenate(outs) if len(outs) > 1 else outs[0]             # [rows, H, rank]
        return self.unabsorb(att.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(rows, -1)
