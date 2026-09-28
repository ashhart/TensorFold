"""DeepSeek-V4-Flash's backbone: 43 hyper-connected blocks (compressed sparse attention, MoE), prefill and decode."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4 import config as C
from tensorfold.families.deepseek_v4.attention import Attention, positions_of
from tensorfold.families.deepseek_v4.caches import LayerCache
from tensorfold.families.deepseek_v4.config import Config
from tensorfold.families.deepseek_v4.dense import dense
from tensorfold.families.deepseek_v4.moe import MoE
from tensorfold.families.glm5_next.linear import Q, per_row
from tensorfold.families.glm5_next.model import HC, hc_expand
from tensorfold.kernels.glm.flash.v1 import hc as HCK
from tensorfold.kernels.glm.flash.v1 import kernels as K


class Block:
    def __init__(self, attn: Attention, moe: MoE, attn_norm: mx.array, ffn_norm: mx.array, attn_hc: HC, ffn_hc: HC,
                 eps: float) -> None:
        self.attn, self.moe = attn, moe
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.eps = eps

    def __call__(self, x: mx.array, ids: mx.array, caches: list[LayerCache], lengths: tuple[int, ...],
                 decode: bool, positions: mx.array | None = None) -> mx.array:
        """x [R, 4, D] streams of rows of consecutive request streams."""

        xc, post, comb = self.attn_hc.split(x, decode)
        a = self.attn(mx.fast.rms_norm(xc, self.attn_norm, self.eps), caches, lengths, decode, positions)
        x = hc_expand(a, x, post, comb, decode)
        xc, post, comb = self.ffn_hc.split(x, decode)
        return hc_expand(self.moe(mx.fast.rms_norm(xc, self.ffn_norm, self.eps), ids, decode), x, post, comb, decode)


def stream_mean(x: mx.array) -> mx.array:
    """The 4 streams' mean [R, D] in fp32, as bf16 (DSpark's taps)."""

    xs = x.astype(mx.float32)
    return ((xs[:, 0] + xs[:, 1] + xs[:, 2] + xs[:, 3]) * 0.25).astype(x.dtype)


class HeadHC:
    """The final hyper-connection: sigmoid-weighted sum of the streams from their fp32 mix, no sinkhorn."""

    def __init__(self, fn: mx.array, base: mx.array, scale: mx.array, eps: float, hc_eps: float) -> None:
        self.fn = fn.astype(mx.float32)
        self.base, self.scale = base.astype(mx.float32), scale.astype(mx.float32)
        self.eps, self.hc_eps = eps, hc_eps
        self._one: Any = None

    def one(self, xs: mx.array) -> mx.array:
        xf = xs.reshape(int(xs.shape[0]), -1).astype(mx.float32)
        inv = mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + self.eps)
        pre = mx.sigmoid((xf @ self.fn.T) * inv * self.scale[0] + self.base) + self.hc_eps
        s = xs.astype(mx.float32)
        y = pre[:, 0:1] * s[:, 0]
        for j in range(1, int(xs.shape[1])):
            y = y + pre[:, j:j + 1] * s[:, j]
        return y.astype(xs.dtype)

    def __call__(self, x: mx.array, rows_exact: bool) -> mx.array:
        """Decode rows one compiled call each (a row's bits never depend on its window), prompt rows together."""

        if not rows_exact:
            return self.one(x)
        if self._one is None:
            self._one = mx.compile(self.one)
        return per_row(self._one, x, True)


class DeepSeekV4:
    """The backbone: ``hidden`` (final-normed rows; the streams kept in ``last_streams`` for MTP) and ``head``."""

    def __init__(self, cfg: Config, embed: Q, layers: list[Block], head_hc: HeadHC, norm: mx.array, lm_head: Q) -> None:
        self.args = cfg
        self.embed = embed
        self.layers = layers
        self.head_hc = head_hc
        self.norm = norm
        self.lm_head = lm_head
        self.last_streams: mx.array | None = None
        self.last_normed: mx.array | None = None
        self.tap_layers: tuple[int, ...] = ()        # layers whose streams' mean a draft model reads
        self.last_taps: mx.array | None = None

    def make_cache(self) -> list[Any]:
        return [LayerCache(layer.attn.ratio, self.args.sliding_window) for layer in self.layers]

    def hc_fused_ok(self) -> bool:
        """The fused boundary kernels read these shapes (4 streams of 4096), on Metal only."""

        ok = self.__dict__.get("_hc_ok")
        if ok is None:
            dims = int(self.args.hidden_size)
            ok = self._hc_ok = all(HCK.hc_fits(b.attn_hc, dims) and HCK.hc_fits(b.ffn_hc, dims) for b in self.layers)
        return ok and "hc" in C.ENABLED and K.metal()

    def embed_tokens(self, ids: mx.array) -> mx.array:
        e = self.embed
        return mx.dequantize(e.weight[ids], e.scales[ids], e.biases[ids], group_size=e.group, bits=e.bits)

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
        positions = positions_of([c[0] for c in caches], lengths)             # every layer's rows sit here
        if decode and self.hc_fused_ok():
            # each block boundary in one fused step: the previous block's write-back, the next block's split + norm
            eps, pending = self.args.rms_norm_eps, None
            for i, layer in enumerate(self.layers):
                layer_caches = [c[i] for c in caches]
                x, normed, post, comb = HCK.hc_step(x, pending, layer.attn_hc, layer.attn_norm, eps)
                if i - 1 in self.tap_layers:
                    taps.append(stream_mean(x))
                pending = (layer.attn(normed, layer_caches, lengths, decode, positions), post, comb)
                x, normed, post, comb = HCK.hc_step(x, pending, layer.ffn_hc, layer.ffn_norm, eps)
                pending = (layer.moe(normed, ids, decode), post, comb)
                if C.EVAL_EVERY and (i + 1) % C.EVAL_EVERY == 0 and i + 1 < len(self.layers):
                    mx.async_eval(x, *pending)
            x = HCK.hc_step(x, pending, None, None, eps)[0]
            if len(self.layers) - 1 in self.tap_layers:
                taps.append(stream_mean(x))
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, ids, [c[i] for c in caches], lengths, decode, positions)
                if i in self.tap_layers:
                    taps.append(stream_mean(x))
                if C.EVAL_EVERY and (i + 1) % C.EVAL_EVERY == 0 and i + 1 < len(self.layers):
                    mx.async_eval(x)
        self.last_streams = x
        self.last_taps = mx.concatenate(taps, axis=-1) if taps else None
        self.last_normed = mx.fast.rms_norm(self.head_hc(x, decode), self.norm, self.args.rms_norm_eps)
        return self.last_normed[None]

    def head(self, hidden: mx.array) -> mx.array:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        return dense(flat, self.lm_head, int(flat.shape[0]) <= C.DECODE_ROWS).reshape(*shape[:-1], -1)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call of ``rows`` rows, keep the first ``keep``: every layer's state is indexed by position."""

        for c in cache[:len(self.layers)]:
            c.trim(rows - keep)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches, lengths, keeps):
            if int(keep) < int(rows):
                self.keep_rows(cache, int(rows), int(keep))
