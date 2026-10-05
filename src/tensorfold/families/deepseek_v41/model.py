"""DeepSeek-V4.1's backbone: hyper-connected blocks (CSA2 attention, MoE, engram), prefill and decode.

The Single-Pass shift (TR eq. 6): each sublayer consumes the *previous* sublayer's ``pre`` mix —
attention takes the previous layer's FFN ``pre``, FFN takes this attention's ``pre`` — and the final
boundary collapses the streams with the last layer's FFN ``pre`` (no head mix module exists). Engram
modules run before their layer's mixes; DSpark taps capture ``h.mean(streams)`` at each target layer's
*attention input*, i.e. the collapsed pre-attention row.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41 import config as C
from tensorfold.families.deepseek_v41.attention import Attention, positions_of
from tensorfold.families.deepseek_v41.caches import LayerCache, make_caches
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.engram import Engram, EngramHistory
from tensorfold.families.deepseek_v41.moe import MoE
from tensorfold.families.deepseek_v41.rmsrows import rms_rows
from tensorfold.families.glm5_next.linear import Q, per_row
from tensorfold.families.glm5_next.model import HC, hc_expand
from tensorfold.kernels.glm.flash.v1 import hc as HCK


def hc_pre(x: mx.array, pre: mx.array) -> mx.array:
    """Collapse the streams [R, S, D] weighted by ``pre`` [R, S] (fp32 sum, out in x's dtype)."""
    xf = x.astype(mx.float32)
    y = pre[:, 0:1] * xf[:, 0]
    for s in range(1, int(x.shape[1])):
        y = y + pre[:, s:s + 1] * xf[:, s]
    return y.astype(x.dtype)


def stream_mean(x: mx.array) -> mx.array:
    """The 4 streams' mean [R, D] in fp32, as bf16 (DSpark's taps)."""
    xs = x.astype(mx.float32)
    return ((xs[:, 0] + xs[:, 1] + xs[:, 2] + xs[:, 3]) * 0.25).astype(x.dtype)


def hc_mix_split(hc: HC, x: mx.array, rows_exact: bool) -> tuple[mx.array, mx.array, mx.array]:
    """The reference hc_mix: (pre-mix weights [R, S], post [R, S], comb [R, S, S]).

    NOT glm5's HC.split contract (that returns the collapsed x): the official
    ``hc_mixes`` (inference/model.py L948) returns the *mix coefficients*, which the
    next sublayer's ``hc_pre`` consumes as stream weights. The projection scales the
    mixes by the input's rsqrt(mean-square) AFTER the matmul, exactly as the reference;
    the mix matmul runs one row a call (batched fp32 matmuls round differently than a
    one-row's, on CPU and GPU alike).
    """
    cfg = hc.cfg
    streams, iters, eps = int(cfg.hc_mult), int(cfg.hc_sinkhorn_iters), float(cfg.hc_eps)

    def one(xr: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        def row(r: mx.array) -> tuple[mx.array, mx.array, mx.array]:
            f = r.reshape(int(r.shape[0]), -1).astype(mx.float32)
            m = per_row(lambda t: t @ hc.fn.T, f, True)
            m = m * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + cfg.rms_norm_eps)
            pre = mx.sigmoid(m[..., :streams] * hc.scale[0] + hc.base[:streams]) + eps
            post = 2 * mx.sigmoid(m[..., streams:2 * streams] * hc.scale[1]
                                  + hc.base[streams:2 * streams])
            comb = (m[..., 2 * streams:] * hc.scale[2] + hc.base[2 * streams:]).reshape(
                *m.shape[:-1], streams, streams)
            comb = mx.softmax(comb, axis=-1) + eps
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
            for _ in range(max(iters - 1, 0)):
                comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
                comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
            return pre, post, comb

        if rows_exact and int(xr.shape[0]) > 1:
            # the elementwise chain after the matmul is also not row-invariant on CPU:
            # run the whole mix per row so a window's rows keep their one-row bits
            parts = [row(xr[r:r + 1]) for r in range(int(xr.shape[0]))]
            return (mx.concatenate([p[0] for p in parts]),
                    mx.concatenate([p[1] for p in parts]),
                    mx.concatenate([p[2] for p in parts]))
        return row(xr)

    return one(x)


def hc_expand_ds(branch: mx.array, x: mx.array, post: mx.array, comb: mx.array,
                 rows_exact: bool = False) -> mx.array:
    """The reference hc_post: post * branch + einsum('...ij,...id->...jd', comb, residual), fp32, one
    bf16 rounding at the end. The einsum contracts comb's FIRST index with the residual's stream
    (matmul(comb_T, xs) contracts the second -- different sum order, different bits)."""

    def one(b: mx.array, xs: mx.array, p: mx.array, c: mx.array) -> mx.array:
        y = p[..., None] * b.astype(mx.float32)[:, None, :]
        mixed = mx.einsum("rij,rid->rjd", c.astype(mx.float32), xs.astype(mx.float32))
        return (y + mixed).astype(x.dtype)

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return one(branch, x, post, comb)
    return mx.concatenate([one(branch[r:r + 1], x[r:r + 1], post[r:r + 1], comb[r:r + 1]) for r in range(rows)])


class Block:
    """One hyper-connected block: engram (layers 1, 14) then the two Single-Pass sublayer boundaries."""

    def __init__(self, attn: Attention, moe: MoE, attn_norm: mx.array, ffn_norm: mx.array, attn_hc: HC,
                 ffn_hc: HC, eps: float, engram: Engram | None = None) -> None:
        self.attn, self.moe = attn, moe
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.eps = eps
        self.engram = engram

    def __call__(self, x: mx.array, ids: mx.array, caches: list[Any], lengths: tuple[int, ...], decode: bool,
                 positions: mx.array | None = None, engram_ids: list[list[int]] | None = None,
                 pre: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """x [R, 4, D] streams, ``pre`` the previous sublayer's collapse mix -> (new x, this FFN's pre)."""
        if self.engram is not None and engram_ids is not None:
            x = self.engram.forward(x, engram_ids, rows_exact=decode)
        # attention consumes the *previous* sublayer's pre (the first layer: the identity mix)
        residual = x
        attn_pre, attn_post, attn_comb = hc_mix_split(self.attn_hc, x, decode)
        assert attn_pre is not None
        if pre is None:
            pre = mx.zeros_like(attn_pre)
            pre[:, 0] = 1.0                       # make_identity_pre_mix: [1, 0, 0, 0]
        a = self.attn(rms_rows(hc_pre(x, pre), self.attn_norm, self.eps, decode), caches, lengths, decode,
                      positions)
        x = hc_expand_ds(a, residual, attn_post, attn_comb, decode)
        # FFN consumes this attention's pre
        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mix_split(self.ffn_hc, x, decode)
        m = self.moe(rms_rows(hc_pre(x, attn_pre), self.ffn_norm, self.eps, decode), ids, decode)
        x = hc_expand_ds(m, residual, ffn_post, ffn_comb, decode)
        return x, ffn_pre


class DeepSeekV41:
    """The backbone: ``hidden`` (final-normed rows; streams kept in ``last_streams`` for DSpark) and ``head``."""

    def __init__(self, cfg: Config, embed: Q, layers: list[Block], norm: mx.array, lm_head: Q,
                 token_map: list[int] | None = None, has_draft_weights: bool = False) -> None:
        self.args = cfg
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.has_draft_weights = has_draft_weights
        self.last_streams: mx.array | None = None
        self.last_normed: mx.array | None = None
        self.tap_layers: tuple[int, ...] = ()        # layers whose attention-input mean a draft model reads
        self.last_taps: mx.array | None = None
        self.token_map = token_map
        self.engram_layers = tuple(cfg.engram_layer_ids)
        self._engram_hash: Any = None

    def make_cache(self) -> list[Any]:
        """One stream's state: a LayerCache a layer (heads own their group's pool state) + an engram history."""
        caches: list[Any] = make_caches(self.args)
        caches.append(EngramHistory())
        return caches

    def hc_fused_ok(self) -> bool:
        """The fused boundary kernels read these shapes (4 streams of the hidden size), on Metal only."""
        dims = int(self.args.hidden_size)
        ok = all(HCK.hc_fits(b.attn_hc, dims) and HCK.hc_fits(b.ffn_hc, dims) for b in self.layers)
        return ok and "hc" in C.ENABLED

    def embed_tokens(self, ids: mx.array) -> mx.array:
        e = self.embed
        return mx.dequantize(e.weight[ids], e.scales[ids], e.biases[ids], group_size=e.group, bits=e.bits)

    def engram_hash(self) -> Any:
        """The engram hash (built once): primes and multipliers are pure functions of the config."""
        from tensorfold.families.deepseek_v41.engram import EngramHash

        if self._engram_hash is None:
            cfg = self.args
            token_map = self.token_map or list(range(cfg.vocab_size))
            self._engram_hash = EngramHash(cfg, token_map, token_map[cfg.engram_pad_token_id])
        return self._engram_hash

    def _engram_ids(self, caches: list[list[Any]], ids: list[int], lengths: tuple[int, ...],
                    layer: int) -> list[list[int]]:
        """Per-row 24-hash-id lists: each stream's committed history, then its rows' compressed ids."""
        table = self.engram_hash()
        out: list[list[int]] = []
        at = 0
        for cache, n in zip(caches, lengths):
            history = cache[-1].ids if isinstance(cache[-1], EngramHistory) else []
            window = history + ids[at:at + n]
            for r in range(n):
                out.append(table.ids(window[:len(history) + r + 1], layer))
            at += n
        return out

    def hidden(self, tokens: Any, cache: list[Any]) -> mx.array:
        """One stream's R consecutive tokens: final-normed hidden states [1, R, D]."""
        return self.hidden_rows(tokens, [cache])

    def hidden_rows(self, tokens: Any, caches: list[list[Any]], lengths: Any = None) -> mx.array:
        """Several streams' rows in one forward, each with its own call's bits; a prompt chunk is one stream's."""
        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        rows = int(ids.shape[0])
        lengths = (rows,) if lengths is None else tuple(int(n) for n in lengths)
        decode = rows <= C.DECODE_ROWS
        if sum(lengths) != rows or len(lengths) != len(caches) or (len(lengths) > 1 and not decode):
            raise ValueError(f"hidden_rows: {len(caches)} streams of {lengths} rows for {rows} tokens (at most "
                             f"{C.DECODE_ROWS} rows when shared)")
        h = self.embed_tokens(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, self.args.hc_mult, h.shape[-1])))
        taps: list[mx.array] = []
        # the first layer's offsets fix every row's position (all layers share the stream's offset)
        positions = positions_of([c[0] for c in caches], lengths)
        plain = [int(t) for t in ids.tolist()]
        compressed = [self.token_map[t] for t in plain] if self.token_map is not None else plain
        engram_ids = None
        pre = None
        for i, layer in enumerate(self.layers):
            if i in self.engram_layers:
                engram_ids = self._engram_ids(caches, compressed, lengths, i)
            x, pre = layer(x, ids, caches, lengths, decode, positions, engram_ids, pre)
            if i in self.tap_layers:
                # the tap is the target layer's attention input: the collapsed pre-mix row, pre-block
                taps.append(stream_mean(x))
            if C.EVAL_EVERY and (i + 1) % C.EVAL_EVERY == 0 and i + 1 < len(self.layers):
                mx.async_eval(x)
        self.last_streams = x
        self.last_taps = mx.concatenate(taps, axis=-1) if taps else None
        self.last_normed = rms_rows(hc_pre(x, pre), self.norm, self.args.rms_norm_eps)
        # the committed rows enter the engram history; a keep_rows trims what a rollback rejects
        at = 0
        for cache, n in zip(caches, lengths):
            if isinstance(cache[-1], EngramHistory):
                cache[-1].push(compressed[at:at + n])
            at += n
        return self.last_normed[None]

    def head(self, hidden: mx.array) -> mx.array:
        """The lm head: the reference feeds it fp32 (runtime.py: norm(...).astype(float32) -> w.linear),
        so the quantized matmul runs in fp32 and stays fp32 -- bf16 input would round every logit."""
        from tensorfold.families.deepseek_v41.dense import dense

        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1]).astype(mx.float32)
        return dense(flat, self.lm_head, int(flat.shape[0]) <= C.DECODE_ROWS).reshape(*shape[:-1], -1)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call of ``rows`` rows, keep the first ``keep``: every cache is indexed by position."""
        for c in cache[:len(self.layers)]:
            c.trim(rows - keep)
        # trailing entries by type: the engram history trims, a runtime-appended draft cache does not
        for c in cache[len(self.layers):]:
            if isinstance(c, EngramHistory):
                c.trim(rows - keep)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches, lengths, keeps):
            if int(keep) < int(rows):
                self.keep_rows(cache, int(rows), int(keep))
