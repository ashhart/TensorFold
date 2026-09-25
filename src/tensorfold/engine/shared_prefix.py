"""Shared-prefix attention for fan-out lanes: the prefix is stored once.

Fan-out lanes all fork one long prefix (the request plus the whole outline)
and differ only in a short suffix and what they write. Copying the prefix
KV into every lane makes each decode round read B copies of the same keys:
on the M5 Max, 128 lanes over a 3,000-token prefix ran 334 ms a round at
85 GB, against 237 ms at 60 GB over a 256-token prefix (2026-09-19).

Here the prefix keys and values live once, shape ``[1, Hkv, P, D]``; each
lane owns only its tail in an ordinary ``BatchKVCache``. Attention takes one
softmax over both parts: prefix scores from a single dense matmul of every
lane's queries against the shared keys, tail scores per lane, concatenated
before the softmax. The prefix needs no mask: it is wholly before every tail
position and holds no padding.

Arithmetic differs from the copied-prefix path in summation order only. The
lane engine's contract is target-verified, not bitwise, and this keeps it.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache


class SharedPrefixKV:
    """One lane's attention cache: a shared prefix plus an empty tail."""

    def __init__(self, prefix_keys: mx.array, prefix_values: mx.array) -> None:
        self.prefix_keys = prefix_keys
        self.prefix_values = prefix_values
        self.tail = KVCache()

    @classmethod
    def merge(cls, caches: list["SharedPrefixKV"]) -> "BatchSharedPrefixKV":
        first = caches[0]
        if any(c.prefix_keys is not first.prefix_keys for c in caches):
            raise ValueError("lanes of one batch must share one prefix")
        return BatchSharedPrefixKV(
            first.prefix_keys, first.prefix_values, BatchKVCache.merge([c.tail for c in caches]))


class BatchSharedPrefixKV:
    """Batched form: one prefix, a ``BatchKVCache`` of per-lane tails."""

    def __init__(self, prefix_keys: mx.array, prefix_values: mx.array, tail: BatchKVCache) -> None:
        self.prefix_keys = prefix_keys
        self.prefix_values = prefix_values
        self.prefix_len = int(prefix_keys.shape[2])
        # [Hkv, D, P] and [Hkv, P, D]: one dense matmul for every lane's queries
        self.prefix_keys_t = mx.contiguous(prefix_keys[0].swapaxes(-1, -2))
        self.prefix_values_m = prefix_values[0]
        self.tail = tail

    # geometry the model and the engine read
    @property
    def offset(self) -> mx.array:
        return self.tail.offset + self.prefix_len

    @property
    def keys(self) -> Any:
        return self.tail.keys

    @property
    def values(self) -> Any:
        return self.tail.values

    @property
    def _right_padding(self) -> Any:
        return self.tail._right_padding

    @_right_padding.setter
    def _right_padding(self, value: Any) -> None:
        self.tail._right_padding = value

    @property
    def state(self) -> tuple:
        return () if self.tail.keys is None else self.tail.state

    @property
    def nbytes(self) -> int:
        return int(self.tail.nbytes)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        return self.tail.update_and_fetch(keys, values)

    def make_mask(self, N: int, **kwargs: Any) -> Any:
        kwargs["return_array"] = True
        return self.tail.make_mask(N, **kwargs)

    def prepare(self, **kwargs: Any) -> None:
        self.tail.prepare(**kwargs)

    def finalize(self) -> None:
        self.tail.finalize()

    def filter(self, batch_indices: Any) -> None:
        self.tail.filter(batch_indices)

    def extend(self, other: Any) -> None:
        if not isinstance(other, BatchSharedPrefixKV) or other.prefix_keys is not self.prefix_keys:
            raise TypeError("only lanes forked from the same shared prefix can join this batch")
        self.tail.extend(other.tail)

    def extract(self, idx: int) -> KVCache:
        """A standalone cache for one lane: the prefix and its tail, copied out."""

        tail = self.tail.extract(idx)
        cache = KVCache()
        cache.keys = mx.concatenate([self.prefix_keys, tail.keys], axis=2)
        cache.values = mx.concatenate([self.prefix_values, tail.values], axis=2)
        cache.offset = cache.keys.shape[2]
        return cache

    def size(self) -> int:
        return self.prefix_len + self.tail.size()

    def empty(self) -> bool:
        return False


def fork_shared(prefix_cache: list[Any], copy_other: Any) -> list[Any]:
    """Per-lane cache list: attention layers share the prefix, the rest are copied.

    ``prefix_cache`` is a batch-1 cache list holding the prefix. ``copy_other``
    copies one non-attention layer cache (the GDN state, which is per lane).
    Attention arrays are trimmed to the absorbed prefix once and reused by
    reference across every fork of this prefix.
    """

    shared: Any = getattr(fork_shared, "_trimmed", None)
    key = id(prefix_cache)
    if shared is None or shared[0] != key:
        trimmed = []
        for item in prefix_cache:
            if isinstance(item, KVCache):
                n = int(item.offset)
                trimmed.append((mx.contiguous(item.keys[..., :n, :]),
                                mx.contiguous(item.values[..., :n, :])))
            else:
                trimmed.append(None)
        mx.eval([a for pair in trimmed if pair is not None for a in pair])
        shared = (key, trimmed, prefix_cache)  # hold the list so the id stays unique
        fork_shared._trimmed = shared  # type: ignore[attr-defined]
    out: list[Any] = []
    for item, pair in zip(prefix_cache, shared[1]):
        out.append(SharedPrefixKV(*pair) if pair is not None else copy_other(item))
    return out


def _shared_attention(self: Any, x: mx.array, mask: Any, cache: BatchSharedPrefixKV) -> mx.array:
    B, L, _ = x.shape
    H, Hkv = self.num_attention_heads, self.num_key_value_heads
    G = H // Hkv

    queries, gate = mx.split(self.q_proj(x).reshape(B, L, H, -1), 2, axis=-1)
    keys, values = self.k_proj(x), self.v_proj(x)
    gate = gate.reshape(B, L, -1)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, Hkv, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, Hkv, -1).transpose(0, 2, 1, 3)

    offset = cache.offset
    queries = self.rope(queries, offset=offset)
    keys = self.rope(keys, offset=offset)
    tail_keys, tail_values = cache.update_and_fetch(keys, values)  # [B, Hkv, T, D]
    D = queries.shape[-1]
    T = tail_keys.shape[2]

    grouped = queries.reshape(B, Hkv, G, L, D)
    # prefix: every lane's queries against the one copy of the keys
    flat = grouped.transpose(1, 0, 2, 3, 4).reshape(Hkv, B * G * L, D)
    prefix_scores = (flat @ cache.prefix_keys_t).reshape(Hkv, B, G, L, -1).transpose(1, 0, 2, 3, 4)
    # tail: per lane, masked for padding and causality inside the window
    tail_scores = grouped @ tail_keys[:, :, None].swapaxes(-1, -2)  # [B, Hkv, G, L, T]
    if mask is not None and not isinstance(mask, str):
        tail_mask = mask[..., -T:]
        if tail_mask.dtype == mx.bool_:
            tail_scores = mx.where(tail_mask[:, :, None], tail_scores,
                                   mx.array(mx.finfo(tail_scores.dtype).min, tail_scores.dtype))
        else:
            tail_scores = tail_scores + tail_mask[:, :, None]
    elif L > 1:
        causal = mx.tril(mx.ones((L, T), dtype=mx.bool_), k=T - L)
        tail_scores = mx.where(causal, tail_scores,
                               mx.array(mx.finfo(tail_scores.dtype).min, tail_scores.dtype))

    weights = mx.softmax(
        mx.concatenate([prefix_scores, tail_scores], axis=-1) * self.scale, axis=-1, precise=True)
    P = cache.prefix_len
    prefix_w = weights[..., :P].transpose(1, 0, 2, 3, 4).reshape(Hkv, B * G * L, P)
    from_prefix = (prefix_w @ cache.prefix_values_m).reshape(Hkv, B, G, L, -1).transpose(1, 0, 2, 3, 4)
    from_tail = weights[..., P:] @ tail_values[:, :, None]
    output = (from_prefix + from_tail).reshape(B, H, L, -1).transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


def install_shared_attention(model: Any) -> int:
    """Route attention through the shared-prefix path when the cache asks for it.

    Patches the attention class once; any other cache type takes the stock
    path untouched. Returns the number of attention layers found.
    """

    language_model = getattr(model, "language_model", model)
    core = getattr(language_model, "model", language_model)
    layers = [layer.self_attn for layer in core.layers if hasattr(layer, "self_attn")]
    for cls in {type(attn) for attn in layers}:
        if getattr(cls, "_tensorfold_shared_prefix", False):
            continue
        stock = cls.__call__

        def patched(self: Any, x: mx.array, mask: Any = None, cache: Any = None,
                    _stock: Any = stock) -> mx.array:
            if isinstance(cache, BatchSharedPrefixKV):
                return _shared_attention(self, x, mask, cache)
            return _stock(self, x, mask, cache)

        cls.__call__ = patched
        cls._tensorfold_shared_prefix = True
    return len(layers)


__all__ = ["BatchSharedPrefixKV", "SharedPrefixKV", "fork_shared", "install_shared_attention"]
