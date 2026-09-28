"""The compressors (gated pooling of every ``ratio`` positions into one key row) and the ratio-4 layers' indexer."""

from __future__ import annotations


import mlx.core as mx

from tensorfold.families.deepseek_v4.caches import LayerCache
from tensorfold.families.glm5_next.linear import Q, project
from tensorfold.kernels.deepseek.v4 import pool as PK
from tensorfold.kernels.deepseek.v4 import rope as NR


def rope(x: mx.array, positions: mx.array, inv_freq: mx.array, inverse: bool = False) -> mx.array:
    """RoPE on x [R, ..., D]'s last 2 * len(inv_freq) dims in interleaved pairs at ``positions`` [R], in fp32."""

    # precise cos and sin: mx.fast.rope's fast trig drifts at long-context angles
    half = int(inv_freq.shape[0])
    dims = 2 * half
    theta = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    shape = (int(x.shape[0]),) + (1,) * (x.ndim - 2) + (half,)
    cos, sin = mx.cos(theta).reshape(shape), mx.sin(theta).reshape(shape)
    if inverse:
        sin = -sin
    pe = x[..., -dims:].astype(mx.float32).reshape(*x.shape[:-1], half, 2)
    a, b = pe[..., 0], pe[..., 1]
    out = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(*x.shape[:-1], dims).astype(x.dtype)
    return mx.concatenate([x[..., :-dims], out], axis=-1)


def norm_rope(x: mx.array, positions: mx.array, inv_freq: mx.array, *, weight: mx.array | None = None,
              eps: float = 1e-6, norm: bool = True, inverse: bool = False) -> mx.array:
    """RMSNorm (``norm``, times ``weight``) then RoPE: one kernel on Metal, MLX's ops elsewhere."""

    if NR.fits(x):
        return NR.norm_rope(x, positions, inv_freq, weight=weight, eps=eps, norm=norm, inverse=inverse)
    if norm:
        x = mx.fast.rms_norm(x, weight, eps)
    return rope(x, positions, inv_freq, inverse)


def pool_blocks(kv: mx.array, score: mx.array) -> mx.array:
    """Blocks' slots [n, slots, d] softmax-pooled over the slots in fp32, elementwise so no block sees another."""

    slots = int(kv.shape[1])
    top = score[:, 0]
    for j in range(1, slots):
        top = mx.maximum(top, score[:, j])
    e = [mx.exp(score[:, j] - top) for j in range(slots)]
    total = e[0]
    for j in range(1, slots):
        total = total + e[j]
    out = (e[0] / total) * kv[:, 0]
    for j in range(1, slots):
        out = out + (e[j] / total) * kv[:, j]
    return out


class Compressor:
    """A layer's (or its indexer's) compressor: wkv | wgate in fp32, pooled with ape, normed, roped at block start."""

    def __init__(self, wkv: Q, wgate: Q, ape: mx.array, norm: mx.array, ratio: int, eps: float,
                 inv_freq: mx.array) -> None:
        self.ratio = int(ratio)
        self.overlap = self.ratio == 4
        self.dim = int(norm.shape[0])
        self.proj = Q.stack([wkv, wgate])
        self.width = wkv.outs
        self.ape = ape.astype(mx.float32)
        self.norm = norm
        self.eps = eps
        self.inv_freq = inv_freq
        self.col = 0              # where its values start in the layer's stacked projections (gates: + width)
        self.index = False        # the indexer's: its rows go to the index pool

    def emit(self, cache: LayerCache, start: int, rows: int, span: tuple[mx.array, int] | None = None) -> None:
        """Pool every block rows at positions start .. complete, from the ring (or a span of projections)."""

        r = self.ratio
        first, last = start // r, (start + rows) // r               # blocks [first, last) complete now
        if last <= first:
            return
        if PK.fits(self.dim):
            data, base = span if span is not None else (cache.proj, 0)
            out = PK.pool_rows(data, self.ape, self.norm, self.inv_freq, first=first, count=last - first, ratio=r,
                               overlap=self.overlap, col=self.col, base=base,
                               ring=0 if span is not None else int(cache.proj.shape[0]), eps=self.eps)
        else:
            out = self._pool_ops(cache, first, last, span)
        cache.write_pool(out, first, self.index)

    def _pool_ops(self, cache: LayerCache, first: int, last: int, span: tuple[mx.array, int] | None) -> mx.array:
        """``emit``'s rows with MLX's ops (no Metal)."""

        r, d, w, count = self.ratio, self.dim, self.width, last - first
        lo = max(0, (first - int(self.overlap)) * r)
        if span is None:
            data = cache.proj_rows(lo, last * r)
        else:
            data = span[0][lo - span[1]:last * r - span[1]]
        ape = self.ape[mx.arange(lo, last * r) % r]
        kv, score = data[:, self.col:self.col + w], data[:, self.col + w:self.col + 2 * w] + ape
        if self.overlap and first == 0:                             # block 0's missing half: no weight, no value
            kv = mx.concatenate([mx.zeros((r, w), dtype=mx.float32), kv])
            score = mx.concatenate([mx.full((r, w), float("-inf"), dtype=mx.float32), score])
        if self.overlap:
            cur_kv, cur_sc = kv[r:].reshape(count, r, 2 * d), score[r:].reshape(count, r, 2 * d)
            prev_kv, prev_sc = kv[:count * r].reshape(count, r, 2 * d), score[:count * r].reshape(count, r, 2 * d)
            slots_kv = mx.concatenate([prev_kv[..., :d], cur_kv[..., d:]], axis=1)
            slots_sc = mx.concatenate([prev_sc[..., :d], cur_sc[..., d:]], axis=1)
        else:
            slots_kv, slots_sc = kv.reshape(count, r, d), score.reshape(count, r, d)
        pooled = pool_blocks(slots_kv, slots_sc).astype(mx.bfloat16)
        return norm_rope(pooled, mx.arange(first, last) * r, self.inv_freq, weight=self.norm, eps=self.eps)


class Indexer:
    """Scores the ratio-4 pool for each query row and keeps its top ``topk`` rows (all of them up to ``topk``)."""

    def __init__(self, wq_b: Q, weights_proj: Q, compressor: Compressor, heads: int, dim: int, topk: int,
                 inv_freq: mx.array) -> None:
        self.wq_b, self.weights_proj, self.compressor = wq_b, weights_proj, compressor
        self.heads, self.dim, self.topk = heads, dim, topk
        self.inv_freq = inv_freq
        self.scale = dim ** -0.5

    def queries(self, qr: mx.array, x: mx.array, positions: mx.array,
                rows_exact: bool) -> tuple[mx.array, mx.array]:
        """Roped index queries [R, H, dim] and head weights [R, H] (bf16, scaled by heads ** -0.5)."""

        rows = int(x.shape[0])
        q = project(qr, self.wq_b, rows_exact=rows_exact).reshape(rows, self.heads, self.dim)
        q = norm_rope(q, positions, self.inv_freq, norm=False)
        w = project(x, self.weights_proj, rows_exact=rows_exact) * (self.heads ** -0.5)
        return q, w

    def scores(self, q: mx.array, w: mx.array, keys: mx.array) -> mx.array:
        """Rows' index scores [R, n] over pool rows keys [n, dim]: head-weighted ReLU(q . k) summed over heads."""

        s = mx.maximum(mx.matmul(q, keys.T), 0) * self.scale                   # [R, H, n]
        return (s.astype(mx.float32) * w.astype(mx.float32)[..., None]).sum(axis=1)

    def select(self, scores: mx.array) -> mx.array:
        """One row's chosen pool rows, ascending: the ``topk`` best of its scores [n]."""

        top = mx.argpartition(-scores, kth=self.topk - 1)[: self.topk]
        return mx.sort(top)
