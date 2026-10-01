"""GLM-5.3-Flash's backbone: 45 hyper-connected layers (KDA, sparse MLA, MoE), prefill and row-exact decode."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.caches import KDACache, MLACache
from tensorfold.families.glm5_next.config import Config, row_kernel
from tensorfold.families.glm5_next.kda import KDA
from tensorfold.families.glm5_next.linear import ChunkQueue, Q, per_row, project
from tensorfold.kernels.glm.flash.v1 import hc as HCK
from tensorfold.kernels.glm.flash.v1 import kernels as K
from tensorfold.kernels.glm.flash.v1 import prompt as PK


class HC:
    """One hyper-connection: RMS over the flattened streams, a fp32 projection to (2 + S) S mixes, sinkhorn."""

    def __init__(self, fn: mx.array, base: mx.array, scale: mx.array, cfg: Config) -> None:
        self.fn = fn.astype(mx.float32)
        # the stored bf16 mix matrix repacked for the fused mix kernel (exact: bf16 -> fp32 loses nothing)
        self.fn_packed = HCK.pack_hc_fn(fn) if fn.dtype == mx.bfloat16 and tuple(fn.shape) == (24, 16384) else None
        self.base = base.astype(mx.float32)
        self.scale = scale.astype(mx.float32)
        self.cfg = cfg

    def split(self, x: mx.array, rows_exact: bool) -> tuple[mx.array, mx.array, mx.array]:
        rows = int(x.shape[0])
        z = mx.fast.rms_norm(x.astype(mx.float32).reshape(rows, -1), None, self.cfg.rms_norm_eps)
        if row_kernel("hc", rows, rows_exact):
            mixes = K.matmul_rows(z, self.fn, transposed=False)
        else:
            mixes = per_row(lambda r: r @ self.fn.T, z, rows_exact)
        return K.hc_split(x, mixes, self.scale, self.base, hc=self.cfg.hc_mult, iters=self.cfg.hc_sinkhorn_iters,
                          eps=self.cfg.hc_eps)


def hc_expand(branch: mx.array, x: mx.array, post: mx.array, comb: mx.array, rows_exact: bool = False) -> mx.array:
    """New streams post * branch + comb^T x in fp32 as the reference's batched matmul; decode rows are its batch."""

    def one(b: mx.array, xs: mx.array, p: mx.array, c: mx.array) -> mx.array:
        y = p[..., None] * b.astype(mx.float32)[:, None, :]
        return (y + mx.matmul(c.swapaxes(-1, -2), xs.astype(mx.float32))).astype(x.dtype)

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact or row_kernel("hc", rows, rows_exact):
        return one(branch, x, post, comb)
    return mx.concatenate([one(branch[r:r + 1], x[r:r + 1], post[r:r + 1], comb[r:r + 1]) for r in range(rows)])


class Layer:
    def __init__(self, attn: Any, mlp: Any, in_norm: mx.array, post_norm: mx.array, attn_hc: HC | None,
                 ffn_hc: HC | None, cfg: Config) -> None:
        self.attn, self.mlp = attn, mlp
        self.is_linear = isinstance(attn, KDA)
        self.in_norm, self.post_norm = in_norm, post_norm
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.eps = cfg.rms_norm_eps

    def __call__(self, x: mx.array, caches: list[Any], lengths: tuple[int, ...], decode: bool) -> mx.array:
        """The plain MTP block: pre-norm residual attention and MLP over rows x [R, D] (backbone layers: boundary)."""

        x = x + self.attn(mx.fast.rms_norm(x, self.in_norm, self.eps), caches, lengths, decode)
        return x + self.mlp(mx.fast.rms_norm(x, self.post_norm, self.eps), decode)


class GLM5:
    """The backbone: ``hidden`` (final-normed rows, kept in ``last_normed`` for the MTP head) and ``head``."""

    def __init__(self, cfg: Config, embed: Q, layers: list[Layer], norm: mx.array, lm_head: Q) -> None:
        self.args = cfg
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.last_normed: mx.array | None = None

    def make_cache(self) -> list[Any]:
        return [KDACache() if layer.is_linear else MLACache() for layer in self.layers]

    def hc_fused_ok(self) -> bool:
        ok = self.__dict__.get("_hc_ok")
        if ok is None:
            dims = int(self.args.hidden_size)
            ok = K.metal() and all(layer.attn_hc is not None and HCK.hc_fits(layer.attn_hc, dims)
                                   and HCK.hc_fits(layer.ffn_hc, dims) for layer in self.layers)
            self._hc_ok = ok
        return ok and K.metal() and C.act() == mx.bfloat16          # the fused HC step is bf16-only

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        e = self.embed
        ids = tokens.reshape(-1)
        if e.scales.dtype == mx.bfloat16:
            return mx.dequantize(e.weight[ids], e.scales[ids], e.biases[ids], group_size=e.group,
                                 bits=e.bits).astype(C.act())
        return mx.dequantize(e.weight[ids], e.scales[ids].astype(mx.float32), e.biases[ids].astype(mx.float32),
                             group_size=e.group, bits=e.bits).astype(C.act())

    def hidden(self, tokens: Any, cache: list[Any], *, inputs_embeds: mx.array | None = None) -> mx.array:
        """One stream's R consecutive tokens: final-normed hidden states [1, R, D]."""

        return self.hidden_rows(tokens, [cache], inputs_embeds=inputs_embeds)

    def hidden_rows(self, tokens: Any, caches: list[list[Any]], lengths: Any = None,
                    inputs_embeds: mx.array | None = None) -> mx.array:
        """Several streams' rows in one forward, each with its own call's bits; a prompt chunk is one stream's."""

        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        rows = int(ids.shape[0])
        lengths = (rows,) if lengths is None else tuple(int(n) for n in lengths)
        decode = rows <= C.DECODE_ROWS
        if sum(lengths) != rows or len(lengths) != len(caches) or (len(lengths) > 1 and not decode):
            raise ValueError(f"hidden_rows: {len(caches)} streams of {lengths} rows for {rows} tokens (at most "
                             f"{C.DECODE_ROWS} rows when shared)")
        if inputs_embeds is None:
            h = self.embed_tokens(ids)                                   # [R, D]
        else:
            if len(caches) != 1:
                raise ValueError("GLM multimodal embeddings are accepted for one prefill stream at a time")
            h = inputs_embeds
            if h.ndim == 3:
                if int(h.shape[0]) != 1:
                    raise ValueError("GLM multimodal embeddings must have batch size one")
                h = h[0]
            if h.ndim != 2 or tuple(h.shape) != (rows, int(self.args.hidden_size)):
                raise ValueError("GLM multimodal embeddings must match the prompt rows and hidden size")
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, self.args.hc_mult, h.shape[-1])))
        pending = None
        for i, layer in enumerate(self.layers):
            layer_caches = [c[i] for c in caches]
            x, normed, post, comb = self.boundary(x, pending, layer.attn_hc, layer.in_norm, decode)
            pending = (layer.attn(normed, layer_caches, lengths, decode), post, comb)
            x, normed, post, comb = self.boundary(x, pending, layer.ffn_hc, layer.post_norm, decode)
            pending = (layer.mlp(normed, decode), post, comb)
            if decode and C.EVAL_EVERY and (i + 1) % C.EVAL_EVERY == 0 and i + 1 < len(self.layers):
                mx.async_eval(x, *pending)
        self.last_normed = self.final_norm(self.boundary(x, pending, None, None, decode)[0])
        return self.last_normed[None]

    def boundary(self, x: mx.array, pending: Any, hc: HC | None, norm: mx.array | None,
                 decode: bool) -> tuple[mx.array, Any, Any, Any]:
        """The pending block's write-back, then the next block's split and RMSNorm (no ``hc``: the write-back only)."""

        prompt_ok = PK.proven() and (pending is None or pending[0].dtype == mx.bfloat16)
        if "hc" in C.FUSED and self.hc_fused_ok() and (decode or prompt_ok):
            # the boundary in fused kernels, each with the row-by-row path's (or, on M1-M4, a bf16 prompt chunk's) bits
            return HCK.hc_step(x, pending, hc, norm, self.args.rms_norm_eps, not decode)
        if pending is not None:
            x = hc_expand(pending[0], x, pending[1], pending[2], decode)
        if hc is None:
            return x, None, None, None
        xc, post, comb = hc.split(x, decode)
        return x, mx.fast.rms_norm(xc, norm, self.args.rms_norm_eps), post, comb

    def final_norm(self, x: mx.array) -> mx.array:
        """The streams' fp32 mean, then the final RMSNorm: the rows the LM and MTP heads read."""

        xs = x.astype(mx.float32)
        raw = xs[:, 0]
        for s in range(1, int(x.shape[1])):
            raw = raw + xs[:, s]
        raw = (raw * (1.0 / int(x.shape[1]))).astype(x.dtype)
        # the MTP head reads the row the LM head reads: its drafts land more often than from the streams' mean
        return mx.fast.rms_norm(raw, self.norm, self.args.rms_norm_eps)

    def hidden_pass(self, tokens: Any, cache: list[Any], sizes: Any) -> mx.array:
        """Prompt chunks, each by its own forward's calls, two in flight; routed experts take them all at once."""

        sizes = tuple(int(n) for n in sizes)
        if len(sizes) == 1:
            return self.hidden(tokens, cache)
        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        rows = int(ids.shape[0])
        if sum(sizes) != rows or min(sizes) <= C.DECODE_ROWS:
            raise ValueError(f"hidden_pass: chunks of {sizes} rows for {rows} tokens (each over {C.DECODE_ROWS})")
        if any("streamer" in vars(layer.mlp) or layer.attn_hc is None for layer in self.layers):
            raise ValueError("hidden_pass: experts streamed from SSD and plain blocks take each chunk alone")
        h = self.embed_tokens(ids)
        streams, width = self.args.hc_mult, int(h.shape[-1])
        starts = [sum(sizes[:j]) for j in range(len(sizes))]
        xs = [mx.contiguous(mx.broadcast_to(h[a:a + n, None, :], (n, streams, width))) for a, n in zip(starts, sizes)]
        pend: list[Any] = [None] * len(xs)
        queue = ChunkQueue()
        for layer, c in zip(self.layers, cache):
            mixes = []
            for j, x in enumerate(xs):
                x, normed, post, comb = self.boundary(x, pend[j], layer.attn_hc, layer.in_norm, False)
                att = layer.attn(normed, [c], (int(x.shape[0]),), False)
                xs[j], normed, post, comb = self.boundary(x, (att, post, comb), layer.ffn_hc, layer.post_norm, False)
                mixes.append((normed, post, comb))
                queue.push(xs[j], normed, post, comb, *c.state)
            together = getattr(layer.mlp, "pass_chunks", None)
            ys = together([m[0] for m in mixes], queue) if together else [layer.mlp(m[0], False) for m in mixes]
            pend = [(y, post, comb) for y, (_, post, comb) in zip(ys, mixes)]
        self.last_normed = mx.concatenate([self.final_norm(self.boundary(x, p, None, None, False)[0])
                                           for x, p in zip(xs, pend)])
        return self.last_normed[None]

    def head(self, hidden: mx.array) -> mx.array:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        return project(flat, self.lm_head, rows_exact=int(flat.shape[0]) <= C.DECODE_ROWS).reshape(*shape[:-1], -1)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call of ``rows`` rows, keep the first ``keep``: KDA states replayed, attention caches trimmed."""

        for c in cache[:len(self.layers)]:
            if isinstance(c, KDACache):
                c.keep(rows, keep)
            else:
                c.trim(rows - keep)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        """``keep_rows`` for every stream of the last ``hidden_rows`` call (a stream that kept every row is left)."""

        for cache, rows, keep in zip(caches, lengths, keeps):
            if int(keep) < int(rows):
                self.keep_rows(cache, int(rows), int(keep))
