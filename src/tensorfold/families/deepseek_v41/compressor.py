"""The compressors (gated pooling of every ``ratio`` positions into one pool row) and the indexers."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.families.deepseek_v41.quant import quantize_cache
from tensorfold.families.glm5_next.linear import Q, per_row, project
from tensorfold.kernels.deepseek.v4 import rope as NR


def rope(x: mx.array, positions: mx.array, inv_freq: mx.array, inverse: bool = False) -> mx.array:
    """RoPE on x [R, ..., D]'s last 2 * len(inv_freq) dims in interleaved pairs at ``positions`` [R], in fp32."""
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
        from tensorfold.families.deepseek_v41.rmsrows import rms_rows

        x = rms_rows(x, weight, eps)
    return rope(x, positions, inv_freq, inverse)


def pool_blocks(kv: mx.array, score: mx.array) -> mx.array:
    """Blocks' slots [n, slots, d] pooled in fp32 then rounded to bf16, exactly the reference:
    sum(softmax(gates).values) computed in fp32, .astype(bf16) at the end (runtime.py:361)."""
    top = mx.max(score, axis=1, keepdims=True)                     # [n, 1, d]
    e = mx.exp(score - top)                                        # [n, slots, d]
    total = mx.sum(e, axis=1, keepdims=True)                       # [n, 1, d]
    pooled = mx.sum((e / total) * kv, axis=1)                      # [n, d] fp32
    return pooled.astype(mx.bfloat16)


def softmax_slots(kv: mx.array, score: mx.array) -> mx.array:
    """One block's slots [slots, d] pooled in fp32 then rounded to bf16 (decode): softmax(score) . kv,
    the reference's order and its .astype(x.dtype) at runtime.py:361."""
    top = mx.max(score)
    e = mx.exp(score - top)
    return ((e / mx.sum(e))[:, None] * kv).sum(axis=0, keepdims=True).astype(mx.bfloat16)


class Compressor:
    """A Full layer's compressor: ratio 1 a plain normed projection, ratio 2 gated pooling over slot pairs.

    Ratio 2 accumulates wkv | wgate projections slot by slot in fp32 and pools every second token as
    sum(softmax(gates) . values); the latent is returned pre-RoPE (the indexer needs it unrotated).
    The weights are BF16 in the checkpoint and promoted to fp32 (the reference promotes them too).
    """

    def __init__(self, wkv: mx.array, wgate: mx.array | None, norm: mx.array, ratio: int, eps: float) -> None:
        self.ratio = int(ratio)
        self.dim = int(norm.shape[0])
        self.norm = norm
        self.eps = eps
        # Weights stay BF16: the reference's w.linear casts the input to the weight's dtype, runs
        # the bf16 matmul (one bf16 rounding of the fp32 accumulator) and upcasts the result --
        # an fp32 x fp32 matmul here keeps precision the reference throws away.
        self.wkv = wkv
        self.wgate = wgate

    def front(self, x: mx.array, rows_exact: bool = False) -> tuple[mx.array, mx.array]:
        """(values, gates) [R, dim] each, fp32 holding bf16-rounded values: the reference's
        x.astype(fp32) -> linear (cast to bf16, bf16 matmul) -> .astype(fp32)."""
        assert self.wgate is not None
        return (per_row(lambda r: (r @ self.wkv.T).astype(mx.float32), x, rows_exact),
                per_row(lambda r: (r @ self.wgate.T).astype(mx.float32), x, rows_exact))


    def plain(self, x: mx.array) -> mx.array:
        """Ratio 1: the plain normed projection, bf16 path (exactly the reference: norm(wkv(x)))."""
        from tensorfold.families.deepseek_v41.rmsrows import rms_rows

        return rms_rows((x @ self.wkv.T).astype(x.dtype), self.norm, self.eps)


class Indexer:
    """A Full or Reindex layer's indexer: scores the pool for each query row, keeps its top ``topk`` rows."""

    def __init__(self, wq_b: Q, weights_proj: Q, heads: int, dim: int, topk: int, inv_freq: mx.array) -> None:
        self.wq_b, self.weights_proj = wq_b, weights_proj
        self.heads, self.dim, self.topk = heads, dim, topk
        self.inv_freq = inv_freq
        self.scale = dim ** -0.5

    def queries(self, qr: mx.array, x: mx.array, positions: mx.array,
                rows_exact: bool) -> tuple[mx.array, mx.array]:
        """Roped index queries [R, H, dim] and head weights [R, H] (scaled by dim ** -0.5 * heads ** -0.5)."""
        rows = int(x.shape[0])
        q = project(qr, self.wq_b, rows_exact=rows_exact).reshape(rows, self.heads, self.dim)
        q = rope(q, positions, self.inv_freq)
        w = project(x, self.weights_proj, rows_exact=rows_exact) * (self.dim ** -0.5) * (self.heads ** -0.5)
        return quantize_cache(q, 4, 32), w

    def scores(self, q: mx.array, w: mx.array, keys: mx.array) -> mx.array:
        """Rows' index scores [R, n] over pool rows keys [n, dim]: head-weighted ReLU(q . k) summed over heads."""
        s = mx.maximum(mx.matmul(q.astype(mx.float32), keys.astype(mx.float32).swapaxes(-1, -2)), 0)  # [R, H, n]
        return (s * w.astype(mx.float32)[..., None]).sum(axis=1)

    def select(self, scores: mx.array) -> mx.array:
        """One row's chosen pool rows, ascending: the ``topk`` best of its scores [n]."""
        top = mx.argpartition(-scores, kth=self.topk - 1)[: self.topk]
        return mx.sort(top)


def candidate_mask(scores: mx.array, visible: int, block: int, topk_blocks: int) -> mx.array:
    """The candidate blocks of one query row's pool scores: top ``topk_blocks`` blocks of ``block``, newest pinned.

    ``scores`` [n] are the row's pool scores (unreachable rows already -inf); ``visible`` is how many
    pool rows the query can see, so its newest (possibly partial) block is ``(visible - 1) // block`` and
    is pinned in. Unreachable picks (block score -inf) are dropped, exactly ``select_candidate_blocks``.
    """
    n = int(scores.shape[0])
    if n <= 0 or visible <= 0:
        return mx.zeros((n,), dtype=mx.bool_)
    padded = mx.pad(scores, (0, (-n) % block), constant_values=float("-inf"))
    block_scores = mx.max(padded.reshape(-1, block), axis=-1)
    blocks = int(block_scores.shape[0])
    pin = (visible - 1) // block
    if pin < blocks:
        block_scores = mx.put_along_axis(block_scores, mx.array([pin], dtype=mx.int32),
                                         mx.array([float("inf")]), axis=0)
    picks = mx.argsort(block_scores)[-min(topk_blocks, blocks):]
    keep = mx.take(block_scores, picks) > float("-inf")
    kept = [int(p) for p, k in zip(picks.tolist(), keep.tolist()) if k]
    if not kept:
        return mx.zeros((n,), dtype=mx.bool_)
    picks = mx.array(kept, dtype=mx.int32)
    return mx.any(mx.arange(n)[:, None] // block == picks.reshape(1, -1), axis=-1)
