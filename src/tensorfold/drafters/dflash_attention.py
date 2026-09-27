"""DFlash2 attention arithmetic and sliding-window cache updates."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _dflash_attend(attn: Any, x: Any, x_ctx: Any, rope: Any, cache: Any, masks: dict) -> Any:
    """Preserve vendor attention arithmetic with bit-exact stacked KV projections and masks shared across layers."""

    from mlx_lm.models.base import create_causal_mask

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse

    B, L, _ = x.shape
    S = x_ctx.shape[1]
    if attn.is_sliding:
        keep_ctx = attn.sliding_window - 1
        if S > keep_ctx:
            skip = S - keep_ctx
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip
    nh, nkv = attn.n_heads, attn.n_kv_heads
    queries = attn.q_proj(x)
    kv_ctx = lane_fuse.attn_kv(attn, x_ctx)
    kv_x = lane_fuse.attn_kv(attn, x)
    if kv_ctx is None or kv_x is None:
        ctx_keys, ctx_values = attn.k_proj(x_ctx), attn.v_proj(x_ctx)
        prop_keys, prop_values = attn.k_proj(x), attn.v_proj(x)
    else:
        half = kv_x.shape[-1] // 2
        ctx_keys, ctx_values = kv_ctx[..., :half], kv_ctx[..., half:]
        prop_keys, prop_values = kv_x[..., :half], kv_x[..., half:]
    queries = attn.q_norm(queries.reshape(B, L, nh, -1)).transpose(0, 2, 1, 3)
    ctx_keys = attn.k_norm(ctx_keys.reshape(B, S, nkv, -1)).transpose(0, 2, 1, 3)
    ctx_values = ctx_values.reshape(B, S, nkv, -1).transpose(0, 2, 1, 3)
    prop_keys = attn.k_norm(prop_keys.reshape(B, L, nkv, -1)).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
    queries = rope(queries, offset=cache.offset + S)
    ctx_keys = rope(ctx_keys, offset=cache.offset)
    prop_keys = rope(prop_keys, offset=cache.offset + S)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    ctx_len = keys.shape[2]
    keys = mx.concatenate([keys, prop_keys], axis=2)
    values = mx.concatenate([values, prop_values], axis=2)
    key = (attn.is_sliding, attn.is_causal, attn.sliding_window, L, ctx_len)
    mask = masks.get(key, False)
    if mask is False:
        mask = create_causal_mask(L, offset=ctx_len) if attn.is_causal else None
        if attn.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            k_pos = mx.arange(ctx_len + L)[None]
            context = (k_pos < ctx_len) & (query - k_pos < attn.sliding_window)
            block = k_pos >= ctx_len
            if attn.is_causal:
                block = block & (k_pos <= query)
            mask = context | block
        masks[key] = mask
    output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
    return attn.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


def concat_updates(cache: list[Any]) -> list[Any]:
    """Use concatenating cache updates because one-row in-place updates can allocate negative sizes past the window."""

    for item in cache:
        if hasattr(item, "_update_concat"):
            item.update_and_fetch = item._update_concat
    return cache


__all__ = ["concat_updates"]
