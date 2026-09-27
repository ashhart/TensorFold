"""Batch DFlash2 lattices with row-exact operations and separate attention caches, offsets and masks per stream."""

from __future__ import annotations

import time
from typing import Any, Sequence

import mlx.core as mx

# a stream joins a batched lattice only while its new context rows are few (a prompt's first lattice runs alone)
CONTEXT_ROWS = 64
# block rows one batched forward takes (the lane matmul's limit): more streams split into several forwards
LATTICE_ROWS = 128


def start_trees(drafter: Any, items: Sequence[tuple[Any, Sequence[int], int]]) -> list[tuple]:
    """Return each proposer's ``_finish_tree`` state, batching eligible lattices with matching block lengths."""

    states: list[Any] = [proposer._tree_prelude(context, nodes) for proposer, context, nodes in items]
    groups: dict[int, list[int]] = {}
    for i, state in enumerate(states):
        proposer = items[i][0]
        if state[0] == "need" and proposer.context is not None and int(proposer.context.shape[1]) <= CONTEXT_ROWS:
            groups.setdefault(int(state[1]), []).append(i)
    for block, members in groups.items():
        step = max(1, LATTICE_ROWS // block)
        for part in (members[a:a + step] for a in range(0, len(members), step)):
            if len(part) < 2:
                continue
            started = time.perf_counter()
            lattices = batched_lattices(drafter, [items[i][0] for i in part], [items[i][1] for i in part], block)
            built = time.perf_counter()
            for i, (cands, unary, hproj) in zip(part, lattices):
                _, _, copy_branch, nodes = states[i]
                states[i] = ("lattice", cands, unary, hproj, copy_branch, nodes, started, built)
    for i, state in enumerate(states):
        if state[0] == "need":
            proposer, context, _ = items[i]
            _, block, copy_branch, nodes = state
            started = time.perf_counter()
            cands, unary, hproj = proposer._lattice(context, block)
            states[i] = ("lattice", cands, unary, hproj, copy_branch, nodes, started, time.perf_counter())
    return states


def batched_lattices(drafter: Any, proposers: Sequence[Any], contexts: Sequence[Sequence[int]],
                     block: int) -> list[tuple[mx.array, mx.array, mx.array]]:
    """``DFlashProposer._lattice`` for several streams of one block length, in one forward."""

    from tensorfold.engine.topk import topk_rows

    model = drafter.model
    selector = model.candidate_selector
    count = len(proposers)
    inputs = mx.array([[int(context[-1])] + [drafter.mask_id] * (block - 1) for context in contexts])
    h = model.embed_tokens(inputs) * model.embed_scale
    sizes = [int(p.context.shape[1]) for p in proposers]
    h_ctx_all = model.hidden_norm(model.fc(mx.concatenate([p.context for p in proposers], axis=1)))
    bounds = [0]
    for n in sizes:
        bounds.append(bounds[-1] + n)
    h_ctx = [h_ctx_all[:, a:b] for a, b in zip(bounds, bounds[1:])]
    parts = proposers[0]._compiled_parts()
    masks: dict = {}
    for i, layer in enumerate(model.layers):
        caches = [p.cache[i] for p in proposers]
        if parts is None:
            h = mx.concatenate([layer(h[j:j + 1], h_ctx[j], model.rope, caches[j]) for j in range(count)], axis=0)
        else:
            pre, post = parts[i]
            xn, kernel = pre(h)
            h = post(h, _attend_many(layer.self_attn, xn, h_ctx, model.rope, caches, masks), kernel)
        if i in proposers[0].async_layers:
            mx.async_eval(h)
    hidden = model.norm(h[:, 1:])
    logits, vocab_ids = drafter.candidate_logits(hidden)
    depth, k = int(hidden.shape[1]), int(selector.top_k)
    cands, unary = topk_rows(logits.reshape(count * depth, -1), k)
    if vocab_ids is not None:
        cands = mx.take(vocab_ids, cands)
    hproj = selector.hidden_projection(hidden).astype(mx.float32)
    cands, unary = cands.reshape(count, depth, k), unary.reshape(count, depth, k)
    mx.async_eval(cands, unary, hproj)
    return [(cands[j], unary[j], hproj[j:j + 1]) for j in range(count)]


def _attend_many(attn: Any, x: Any, x_ctx: Sequence[Any], rope: Any, caches: Sequence[Any], masks: dict) -> Any:
    """Project all streams' rows together, then run each stream's attention on its own cache."""

    from mlx_lm.models.base import create_causal_mask

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse

    B, L, _ = x.shape
    ctx = []
    for rows, cache in zip(x_ctx, caches):
        if attn.is_sliding and rows.shape[1] > attn.sliding_window - 1:
            skip = rows.shape[1] - (attn.sliding_window - 1)
            rows = rows[:, skip:]
            cache.offset += skip
        ctx.append(rows)
    sizes = [int(r.shape[1]) for r in ctx]
    nh, nkv = attn.n_heads, attn.n_kv_heads
    all_ctx = mx.concatenate(ctx, axis=1)
    queries = attn.q_proj(x)
    kv_x, kv_ctx = lane_fuse.attn_kv(attn, x), lane_fuse.attn_kv(attn, all_ctx)
    if kv_x is None or kv_ctx is None:
        prop_keys, prop_values = attn.k_proj(x), attn.v_proj(x)
        ctx_keys, ctx_values = attn.k_proj(all_ctx), attn.v_proj(all_ctx)
    else:
        half = kv_x.shape[-1] // 2
        prop_keys, prop_values = kv_x[..., :half], kv_x[..., half:]
        ctx_keys, ctx_values = kv_ctx[..., :half], kv_ctx[..., half:]
    queries = attn.q_norm(queries.reshape(B, L, nh, -1)).transpose(0, 2, 1, 3)
    prop_keys = attn.k_norm(prop_keys.reshape(B, L, nkv, -1)).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
    outputs, at = [], 0
    for j, cache in enumerate(caches):
        S = sizes[j]
        ck = attn.k_norm(ctx_keys[:, at:at + S].reshape(1, S, nkv, -1)).transpose(0, 2, 1, 3)
        cv = ctx_values[:, at:at + S].reshape(1, S, nkv, -1).transpose(0, 2, 1, 3)
        at += S
        q = rope(queries[j:j + 1], offset=cache.offset + S)
        ck = rope(ck, offset=cache.offset)
        pk = rope(prop_keys[j:j + 1], offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ck, cv)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, pk], axis=2)
        values = mx.concatenate([values, prop_values[j:j + 1]], axis=2)
        key = (attn.is_sliding, attn.is_causal, attn.sliding_window, L, ctx_len)
        if key not in masks:
            mask = create_causal_mask(L, offset=ctx_len) if attn.is_causal else None
            if attn.is_sliding:
                query = ctx_len + mx.arange(L)[:, None]
                k_pos = mx.arange(ctx_len + L)[None]
                block = k_pos >= ctx_len
                if attn.is_causal:
                    block = block & (k_pos <= query)
                mask = ((k_pos < ctx_len) & (query - k_pos < attn.sliding_window)) | block
            masks[key] = mask
        outputs.append(mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask=masks[key]))
    output = mx.concatenate(outputs, axis=0)
    return attn.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


__all__ = ["CONTEXT_ROWS", "LATTICE_ROWS", "batched_lattices", "start_trees"]
