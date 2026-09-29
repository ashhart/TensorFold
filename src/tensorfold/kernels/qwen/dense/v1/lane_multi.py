"""Verify multiple streams with single-stream bits by keeping row-local arithmetic, attention paths and recurrent state independent across streams."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx


def _attention(attn: Any, x: mx.array, caches: Sequence[Any], positions: list[int], offsets: Sequence[int],
               widths: Sequence[int], records: list[list[Any]], plans: list[Any]) -> mx.array:
    """Project, normalize and rotate all rows together, then attend through each stream's own cache and tree path."""

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, stream_attention

    B, L, _ = x.shape
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(q_proj_output.reshape(B, L, H, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    kv = lane_fuse.attn_kv(attn, x)
    if kv is None:
        keys, values = attn.k_proj(x), attn.v_proj(x)
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys.reshape(B, L, nkv, -1))
        values = values.reshape(B, L, nkv, -1)
    else:
        D = int(queries.shape[-1])
        queries = attn.q_norm(q_proj_output.reshape(B, L, 2 * H, D))[:, :, 0::2]
        kv = kv.reshape(B, L, 2 * nkv, -1)
        keys = attn.k_norm(kv)[:, :, :nkv]
        values = kv[:, :, nkv:]
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    from tensorfold.vision.rotary import decode_positions

    pos = mx.array(decode_positions(positions, caches, widths), dtype=mx.int32)
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    kv = []
    for s, cache in enumerate(caches):
        a, w = int(offsets[s]), int(widths[s])
        k_s, v_s = keys[:, :, a:a + w], values[:, :, a:a + w]
        records[s].append(("kv", k_s, v_s))
        cache.update_and_fetch(k_s, v_s)
        kv.append((cache.keys, cache.values))
    outs = []
    for first, plan in plans:
        span = queries if len(plans) == 1 else queries[:, :, offsets[first]:offsets[first] + plan.rows]
        outs.append(stream_attention.tree_sdpa(span, kv[first:first + plan.streams], attn.scale, plan))
    output = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output * mx.sigmoid(gate))


def _gdn(gdn: Any, x: mx.array, caches: Sequence[Any], parents: Sequence[Sequence[int]], plans: list[Any],
         offsets: Sequence[int], widths: Sequence[int], records: list[list[Any]]) -> mx.array:
    """Share row-local Gated DeltaNet projections, conv and norms while walking each stream's tree from its own state."""

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue, stream_gdn

    B, _, _ = x.shape
    qkv = gdn.in_proj_qkv(x)
    zba = lane_fuse.gdn_in(gdn, x)
    n_keep = gdn.conv_kernel_size - 1
    heads = dict(nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    states = [cache[0] if cache[0] is not None else mx.zeros((B, n_keep, gdn.conv_dim), dtype=x.dtype)
              for cache in caches]
    if zba is None:
        z, b, a = gdn.in_proj_z(x), gdn.in_proj_b(x), gdn.in_proj_a(x)
        rows = [lane_glue.gdn_pre(qkv[:, o:o + w], states[s], gdn.conv1d.weight,
                                  _windows(parents[s], n_keep), a[:, o:o + w], b[:, o:o + w], gdn.A_log, gdn.dt_bias,
                                  **heads) for s, (o, w) in enumerate(zip(offsets, widths))]
        q, k, v, g, beta = (mx.concatenate([r[i] for r in rows], axis=1) for i in range(5))
    else:
        parts = [_pre_group(gdn, qkv, zba, states, first, plan, offsets, heads) for first, plan, _ in plans]
        q, k, v, g, beta = parts[0] if len(parts) == 1 else (mx.concatenate([p[i] for p in parts], axis=1)
                                                               for i in range(5))
    shared = (q, k, v, g, beta, qkv)                   # all rows, for the commit's replay and conv tails
    recurrent = []
    for s, cache in enumerate(caches):
        state = cache[1]
        if state is None:
            state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
        recurrent.append(state)
        records[s].append(("gdn", n_keep, state, states[s], int(offsets[s]), shared))
    ys = []
    for first, _, tree_plan in plans:
        a = int(offsets[first])
        rows = slice(a, a + tree_plan.rows)
        ys.append(stream_gdn.tree(q[:, rows], k[:, rows], v[:, rows], g[:, rows], beta[:, rows],
                                  recurrent[first:first + tree_plan.streams], tree_plan))
    y = ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=1)
    if zba is None:
        out = lane_glue.gdn_post(y, z, gdn.norm.weight, gdn.norm.eps)
    else:
        out = lane_fuse.gdn_post(y, zba, gdn.norm.weight, gdn.norm.eps)
    return gdn.out_proj(out)


def _pre_group(gdn: Any, qkv: mx.array, zba: mx.array, states: list[mx.array], first: int, plan: Any,
               offsets: Sequence[int], heads: dict[str, int]) -> tuple[mx.array, ...]:
    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    a = int(offsets[first])
    rows = slice(a, a + plan.rows)
    return stream_gdn.gdn_pre(qkv[:, rows], states[first:first + plan.streams], gdn.conv1d.weight, plan,
                              zba[:, rows], gdn.A_log, gdn.dt_bias, **heads)


def _windows(parents: Sequence[int], n_keep: int) -> mx.array:
    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    return stream_gdn.ConvPlan([parents], n_keep).windows


def multi_tree_forward(core: Any, head: Any, windows: Sequence[Sequence[int]], parents: Sequence[Sequence[int]],
                       caches: Sequence[list[Any]], starts: Sequence[int], *, pipeline_layers: int = 4,
                       first_alone: bool = True, last_only: bool = False) -> tuple[mx.array, list[list[Any]], list[int]]:
    """Return tree logits [1, R, V], commit records and first-row offsets; roots have parent -1, attention caches receive rows, and recurrent states await commit replay."""

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue, lane_tree, stream_attention, stream_gdn

    widths = [len(w) for w in windows]
    offsets: list[int] = []
    positions: list[int] = []
    total = 0
    for s, (window, rows_parents) in enumerate(zip(windows, parents)):
        if len(rows_parents) != len(window):
            raise ValueError(f"stream {s}: {len(window)} rows but {len(rows_parents)} parents")
        offsets.append(total)
        total += len(window)
        depths, _ = lane_tree.tree_paths(rows_parents)
        positions.extend(int(starts[s]) + d for d in depths)
    tokens = [int(t) for window in windows for t in window]
    hidden = core.embed_tokens(mx.array([tokens], dtype=mx.uint32))
    records: list[list[Any]] = [[] for _ in windows]
    layers = list(core.layers)
    conv_plans: list[tuple[int, Any, Any]] = []    # (first stream, conv and tree layouts) per group, built once
    plans: list[tuple[int, Any]] = []              # (first stream, attention layout) per group, built once
    pending: mx.array | None = None
    tapped: Any = None
    for index, layer in enumerate(layers):
        inner = getattr(layer, "_layer", layer)
        norm = inner.input_layernorm
        hidden, x = lane_glue.norm_xs(hidden, pending, norm.weight, norm.eps)
        if tapped is not None:
            tapped[0][tapped[1]] = hidden
        items = [cache[index] for cache in caches]
        if getattr(inner, "is_linear", False):
            if not conv_plans:
                n_keep = inner.linear_attn.conv_kernel_size - 1
                for first in range(0, len(windows), stream_gdn.MAX_STREAMS):
                    group = parents[first:first + stream_gdn.MAX_STREAMS]
                    conv_plans.append((first, stream_gdn.ConvPlan(group, n_keep), stream_gdn.TreePlan(group)))
            r = _gdn(inner.linear_attn, x, items, parents, conv_plans, offsets, widths, records)
        else:
            if not plans:
                attn = inner.self_attn
                for first in range(0, len(windows), stream_attention.MAX_STREAMS):
                    group = slice(first, first + stream_attention.MAX_STREAMS)
                    plans.append((first, stream_attention.Plan(parents[group], starts[group],
                                                               attn.num_attention_heads, attn.num_key_value_heads)))
            r = _attention(inner.self_attn, x, items, positions, offsets, widths, records, plans)
        norm = inner.post_attention_layernorm
        hidden, x = lane_glue.norm_xs(hidden, r, norm.weight, norm.eps)
        mlp = inner.mlp
        gu = lane_fuse.mlp_gate_up(mlp, x)
        act = lane_glue.mlp_act(mlp.gate_proj(x), mlp.up_proj(x)) if gu is None else lane_fuse.mlp_act(gu)
        pending = mlp.down_proj(act)
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) \
                and index + 1 < len(layers):
            mx.async_eval(hidden, pending)
    hidden, x = lane_glue.norm_xs(hidden, pending, core.norm.weight, core.norm.eps)
    if tapped is not None:
        tapped[0][tapped[1]] = hidden
    if lane_tree.HIDDEN_SINK is not None:            # a hidden-state proposer reads every row's post-norm hidden
        lane_tree.HIDDEN_SINK.append(x)
        if len(lane_tree.HIDDEN_SINK) > 1024:
            del lane_tree.HIDDEN_SINK[0]
    return head(x[:, -1:] if last_only else x), records, offsets


def commit_streams(caches: Sequence[list[Any]], records: Sequence[list[Any]], paths: Sequence[Sequence[int]],
                   widths: Sequence[int], starts: Sequence[int]) -> None:
    """Commit each root-first accepted path by compacting attention keys, replaying recurrent states and taking conv tails in groups of streams."""

    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    moves = [None if list(path) == list(range(len(path))) else mx.array(list(path), dtype=mx.int32) for path in paths]
    plans: dict[tuple[int, int], Any] = {}
    for index in range(len(records[0])):
        if records[0][index][0] == "kv":
            for s, cache in enumerate(caches):
                item, (_, win_k, win_v), keep = cache[index], records[s][index], len(paths[s])
                if moves[s] is not None:
                    # Read window rows so the cache buffer keeps one owner and updates in place.
                    item.keys[..., starts[s]:starts[s] + keep, :] = mx.take(win_k, moves[s], axis=2)
                    item.values[..., starts[s]:starts[s] + keep, :] = mx.take(win_v, moves[s], axis=2)
                item.trim(int(widths[s]) - keep)
            continue
        _, n_keep, _, _, _, (q, k, v, g, beta, qkv) = records[0][index]
        for first in range(0, len(caches), stream_gdn.MAX_STREAMS):
            group = range(first, min(len(caches), first + stream_gdn.MAX_STREAMS))
            if (first, n_keep) not in plans:
                plans[(first, n_keep)] = stream_gdn.CommitPlan([paths[s] for s in group],
                                                               [records[s][index][4] for s in group], n_keep)
            plan = plans[(first, n_keep)]
            states = stream_gdn.replay(q, k, v, g, beta, [records[s][index][2] for s in group], plan)
            tails = stream_gdn.conv_tails([records[s][index][3] for s in group], qkv, plan)
            for s, state, tail in zip(group, states, tails):
                caches[s][index][1] = state
                caches[s][index][0] = tail
                caches[s][index].advance(len(paths[s]))


__all__ = ["commit_streams", "multi_tree_forward"]
