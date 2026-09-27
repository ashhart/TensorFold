"""The lane decoder for GPUs without tensor units (M1 to M4): the M5 lane decoder's structure over a row-exact
matmul (``row_matmul``) and glue kernels (``row_glue``).

Every kernel treats a row on its own, so a row's bits never depend on the rows beside it: serial steps, verify
windows, several streams' windows in one forward and prompt chains all give a row the same bits. Serial decoding
goes through this forward too, so it is the reference drafted rounds reproduce. Attention runs query by query over
exactly the keys the serial step sees (``exact_attention``); ``TF_ROW_ATTENTION=1`` uses ``row_attention``, which
takes draft trees.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import row_matmul
from tensorfold.kernels.qwen.dense.v1.row_glue import _chain, add_norm, gated_delta, gdn_post, gdn_pre, mlp_act
from tensorfold.kernels.qwen.dense.v1.row_matmul import WINDOW_ROWS, logits, project, project_stack, stack_of


# rows a prompt chain takes
PROMPT_ROWS = 128


# Attention through ``row_attention`` (chains and draft trees; serial steps too, so it is the reference); off: MLX's
# one-query kernel query by query (``exact_attention``, chains only)
ROW_ATTENTION = os.environ.get("TF_ROW_ATTENTION", "0") == "1"


class Record(list):
    """One stream's commit record from a forward (``lane_tree.commit_tree``'s list of layer entries), with where its
    window sat: ``start`` (its first row's position), ``width`` (its rows) and ``parents``."""

    start: int = 0
    width: int = 0
    parents: tuple[int, ...] = ()


class _Rows:
    """The streams of one forward, rows grouped by stream: stream s holds rows [offsets[s], offsets[s] + widths[s]),
    its first row at position ``starts[s]``; each row's position is its stream's start plus its depth."""

    __slots__ = ("offsets", "widths", "parents", "chains", "starts", "records", "positions", "windows", "single")

    def __init__(self, parents: Sequence[Sequence[int]], starts: Sequence[int]) -> None:
        from tensorfold.kernels.qwen.dense.v1 import lane_tree

        self.offsets: list[int] = []
        self.widths: list[int] = []
        self.parents: list[tuple[int, ...]] = []
        self.chains: list[bool] = []
        self.starts = [int(s) for s in starts]
        self.records: list[Record] = []
        positions: list[int] = []
        total = 0
        for rows_parents, start in zip(parents, self.starts):
            rows_parents = tuple(int(p) for p in rows_parents)
            chain = rows_parents == _CHAINS.get(len(rows_parents)) or _chain(rows_parents)
            depths = range(len(rows_parents)) if chain else lane_tree.tree_paths(rows_parents)[0]
            positions.extend(start + d for d in depths)
            record = Record()
            record.start, record.width, record.parents = start, len(rows_parents), rows_parents
            self.offsets.append(total)
            self.widths.append(len(rows_parents))
            self.parents.append(rows_parents)
            self.chains.append(chain)
            self.records.append(record)
            total += len(rows_parents)
        self.single = len(self.widths) == 1
        self.positions = mx.array(positions, dtype=mx.int32)
        self.windows: list[mx.array] | None = None


# chain parents by width (-1, 0, 1, ...), for a cheap comparison
_CHAINS: dict[int, tuple[int, ...]] = {w: tuple(range(-1, w - 1)) for w in range(1, 257)}
_conv_cache: dict[tuple[tuple[int, ...], int], mx.array] = {}


def _conv_windows(parents: tuple[int, ...], n_keep: int) -> mx.array:
    """``lane_tree._conv_windows``, built once per window shape."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    key = (parents, n_keep)
    hit = _conv_cache.get(key)
    if hit is None:
        if len(_conv_cache) > 4096:
            _conv_cache.clear()
        hit = _conv_cache[key] = lane_tree._conv_windows(parents, n_keep)
    return hit


def _in(module: Any, x: mx.array) -> mx.array:
    return project(module, x)


def _attend(attn: Any, queries: mx.array, keys: mx.array, values: mx.array, cache: Any, parents: tuple[int, ...],
            chain: bool, record: list[Any]) -> mx.array:
    """One stream's rows [1, H, w, D] through attention over its own cache (which takes the rows' keys)."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_attention

    L = int(queries.shape[2])
    record.append(("kv", keys, values))
    keys, values = cache.update_and_fetch(keys, values)
    if ROW_ATTENTION:
        # the cache's whole buffers: the window's rows sit at start + row
        return row_attention.row_sdpa(queries, cache.keys, cache.values, attn.scale, int(cache.offset) - L, parents)
    if not chain:
        raise NotImplementedError("row_forward: draft trees need row_attention")
    return exact_attention.exact_sdpa(queries, keys, values, cache, attn.scale, "causal" if L > 1 else None)


def _attention(attn: Any, x: mx.array, items: Sequence[Any], rows: _Rows) -> mx.array:
    """Qwen3-Next attention for every stream's rows (per-row RoPE positions; each stream's rows over its own keys):
    the input of o_proj."""

    B, L, _ = x.shape
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    stack = stack_of(attn, "qkv")
    if stack is not None:
        qkv = project_stack(stack, x)
        nq, nk = stack.sizes[0], stack.sizes[1]
        q_proj_output, k_out, v_out = qkv[..., :nq], qkv[..., nq:nq + nk], qkv[..., nq + nk:]
    else:
        q_proj_output, k_out, v_out = _in(attn.q_proj, x), _in(attn.k_proj, x), _in(attn.v_proj, x)
    D = int(attn.head_dim) if hasattr(attn, "head_dim") else int(q_proj_output.shape[-1]) // (2 * H)
    gate = q_proj_output.reshape(B, L, H, 2 * D)[..., D:].reshape(B, L, -1)
    # RMSNorm per head row over [q_h | gate_h] halves, then the q halves (rows are their own: no copy of a slice)
    queries = attn.q_norm(q_proj_output.reshape(B, L, 2 * H, D))[:, :, 0::2]
    keys = attn.k_norm(k_out.reshape(B, L, nkv, -1))
    values = v_out.reshape(B, L, nkv, -1)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = rows.positions
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    if rows.single:
        output = _attend(attn, queries, keys, values, items[0], rows.parents[0], rows.chains[0], rows.records[0])
    else:
        outs = []
        for s, item in enumerate(items):
            a, w = rows.offsets[s], rows.widths[s]
            outs.append(_attend(attn, queries[:, :, a:a + w], keys[:, :, a:a + w], values[:, :, a:a + w], item,
                                rows.parents[s], rows.chains[s], rows.records[s]))
        output = mx.concatenate(outs, axis=2)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output * mx.sigmoid(gate)


def _recur(gdn: Any, y: mx.array, cache: Any, parents: tuple[int, ...], chain: bool, windows: mx.array,
           record: list[Any], n_keep: int) -> mx.array:
    """One stream's rows of the stacked [qkv | z | b | a] projection ``y`` through the conv and the recurrence from
    its own conv tail and state: the recurrence's output [1, w, Hv, Dv]."""

    conv_state = cache[0] if cache[0] is not None else mx.zeros((1, n_keep, gdn.conv_dim), dtype=y.dtype)
    state = cache[1]
    if state is None:
        state = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    q, k, v, g, beta, conv_out = gdn_pre(y, conv_state, gdn.conv1d.weight, windows, gdn.A_log, gdn.dt_bias,
                                         nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    rec, state_out = gated_delta(q, k, v, g, beta, state, parents, chain=chain)
    record.append(("gdn", q, k, v, g, beta, state, (conv_state, y, gdn.conv_dim, state_out, conv_out, chain),
                   n_keep))
    return rec


def _gdn(gdn: Any, x: mx.array, items: Sequence[Any], rows: _Rows) -> mx.array:
    """The recurrent layer for every stream's rows (each from its own conv tail and state): the input of out_proj."""

    from tensorfold.kernels.qwen.dense.v1 import row_streams

    stack = stack_of(gdn, "in")
    if stack is None:
        raise RuntimeError("row_forward: the recurrent layer's projections are not stacked (build(model) first)")
    y = project_stack(stack, x)                                                     # [qkv | z | b | a]
    n_keep = gdn.conv_kernel_size - 1
    if rows.single:
        if rows.windows is None:
            rows.windows = [_conv_windows(p, n_keep) for p in rows.parents]
        rec = _recur(gdn, y, items[0], rows.parents[0], rows.chains[0], rows.windows[0], rows.records[0], n_keep)
        return gdn_post(rec, y, gdn.norm.weight, gdn.norm.eps, zo=gdn.conv_dim)
    recs = []
    for g0 in range(0, len(items), row_streams.GROUP):
        g1 = min(g0 + row_streams.GROUP, len(items))
        a, b = rows.offsets[g0], rows.offsets[g1 - 1] + rows.widths[g1 - 1]
        chain = all(rows.chains[g0:g1])          # a tree in the launch: no state after the last row, commits replay
        rec, per = row_streams.recur(gdn, y[:, a:b], items[g0:g1], rows.parents[g0:g1], chain, n_keep)
        for record, (q, k, v, g, beta, state, conv_state, ys, state_out, tails) in zip(rows.records[g0:g1], per):
            record.append(("gdn", q, k, v, g, beta, state, (conv_state, ys, gdn.conv_dim, state_out, tails, chain),
                           n_keep))
        recs.append(rec)
    rec = recs[0] if len(recs) == 1 else mx.concatenate(recs, axis=1)
    return gdn_post(rec, y, gdn.norm.weight, gdn.norm.eps, zo=gdn.conv_dim)


def commit(cache: list[Any], record: list[Any], path: Sequence[int], window: int, start: int) -> None:
    """Keep only ``path``'s rows of the recorded window: attention keys moved to their slots and trimmed, conv tails
    the forward's row after ``path[-1]``, recurrent states the recurrence's own (a chain kept whole) or replayed over
    the path."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    keep = len(path)
    in_place = list(path) == list(range(keep))
    whole = keep == window and in_place
    last = int(path[-1])
    rows = count = None
    j = 0
    for item in cache:
        kind, *entry = record[j]
        j += 1
        if hasattr(item, "keys") and hasattr(item, "values"):
            if kind != "kv":
                raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has an attention layer")
            if not in_place:
                # from the window's own rows, not the cache buffer: the slice update then writes in place
                taken = mx.array(list(path), dtype=mx.int32)
                item.keys[..., start:start + keep, :] = mx.take(entry[0], taken, axis=2)
                item.values[..., start:start + keep, :] = mx.take(entry[1], taken, axis=2)
            item.trim(window - keep)
            continue
        if kind != "gdn":
            raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has a recurrent layer")
        q, k, v, g, beta, state0, extra, _ = entry
        _, _, _, state_out, conv_tails, chain = extra
        if whole and chain:
            item[1] = state_out
        else:
            if rows is None:
                rows, count = lane_tree.kernel_ints(path), mx.array([keep], dtype=mx.int32)
            item[1] = lane_tree.replay_path(q, k, v, g, beta, state0, rows, count)
        item[0] = conv_tails[last:last + 1]
        item.advance(keep)
    if j != len(record):
        raise RuntimeError(f"recorded {len(record)} layers, cache has {j}")


def keep_rows(cache: list[Any], record: Record, keep: int) -> None:
    """After a chain window's forward: keep its first ``keep`` rows (attention keys trimmed, conv tail and recurrent
    state exactly as after row keep - 1). ``commit`` for a prefix of the window."""

    commit(cache, record, list(range(int(keep))), record.width, record.start)


def _token_ids(windows: Sequence[Any]) -> mx.array:
    """[1, R] uint32 ids of every stream's window, in order (lists, or lazy GPU arrays, which are not read)."""

    if len(windows) == 1 and isinstance(windows[0], mx.array):
        return windows[0].reshape(1, -1).astype(mx.uint32)
    if not any(isinstance(w, mx.array) for w in windows):
        return mx.array([[int(t) for w in windows for t in w]], dtype=mx.uint32)
    parts = [w.reshape(-1).astype(mx.uint32) if isinstance(w, mx.array) else mx.array([int(t) for t in w],
                                                                                        dtype=mx.uint32)
             for w in windows]
    return mx.concatenate(parts).reshape(1, -1)


def _rows_forward(core: Any, windows: Sequence[Any], parents: Sequence[Sequence[int]], caches: Sequence[list[Any]],
                  starts: Sequence[int], *, pipeline_layers: int = 4, first_alone: bool = True
                  ) -> tuple[mx.array, _Rows]:
    """Every stream's window in one pass: the final-normed rows [1, R, D] and the streams (their records). Row-wise
    work runs once over all rows; attention and the recurrent layers run per stream over its own caches."""

    widths = [len(p) for p in parents]
    if len(windows) != len(parents) or len(caches) != len(parents) or len(starts) != len(parents):
        raise ValueError(f"row_forward: {len(windows)} windows, {len(parents)} parent lists, {len(caches)} caches, "
                         f"{len(starts)} starts")
    for window, width in zip(windows, widths):
        if (int(window.size) if isinstance(window, mx.array) else len(window)) != width:
            raise ValueError("row_forward: a window's tokens and parents differ in length")
    if sum(widths) > row_matmul.BACKEND.max_rows:
        raise ValueError(f"row_forward: {sum(widths)} rows, the {row_matmul.BACKEND.name} matmul takes up to {row_matmul.BACKEND.max_rows}")
    rows = _Rows(parents, starts)
    hidden = core.embed_tokens(_token_ids(windows))
    layers = list(core.layers)
    pending: mx.array | None = None                   # the last output projection's rows, added in the next norm
    tapped: Any = None
    for index, (layer, *items) in enumerate(zip(layers, *caches)):
        inner = getattr(layer, "_layer", layer)
        hidden, x = add_norm(hidden, pending, inner.input_layernorm.weight, inner.input_layernorm.eps)
        if tapped is not None:
            tapped[0][tapped[1]] = hidden
        if getattr(inner, "is_linear", False):
            pending = project(inner.linear_attn.out_proj, _gdn(inner.linear_attn, x, items, rows))
        else:
            pending = project(inner.self_attn.o_proj, _attention(inner.self_attn, x, items, rows))
        norm = inner.post_attention_layernorm
        hidden, x = add_norm(hidden, pending, norm.weight, norm.eps)
        mlp = inner.mlp
        stack = stack_of(mlp, "gu")
        gu = project_stack(stack, x) if stack is not None else mx.concatenate(
            [project(mlp.gate_proj, x), project(mlp.up_proj, x)], axis=-1)
        pending = project(mlp.down_proj, mlp_act(gu))
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) \
                and index + 1 < len(layers):
            mx.async_eval(hidden, pending)
    hidden, x = add_norm(hidden, pending, core.norm.weight, core.norm.eps)
    if tapped is not None:
        tapped[0][tapped[1]] = hidden
    return x, rows


def forward(core: Any, head: Any, tokens: Sequence[int], parents: Sequence[int], cache: list[Any], start: int, *,
            pipeline_layers: int = 4, last_only: bool = False, first_alone: bool = True) -> tuple[mx.array, Record]:
    """Logits [1, W, V] for one window whose root sits at ``start``, and its commit record. Attention caches take
    the rows now; recurrent states wait for ``commit``."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    x, rows = _rows_forward(core, [tokens], [parents], [cache], [start], pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    sink = lane_tree.HIDDEN_SINK          # a proposer that drafts from the rows' post-norm hidden states
    if sink is not None:
        sink.append(x)
        if len(sink) > 1024:
            del sink[0]
    return logits(head, x[:, -1:] if last_only else x), rows.records[0]


def multi_forward(core: Any, head: Any, windows: Sequence[Any], parents: Sequence[Sequence[int]],
                  caches: Sequence[list[Any]], starts: Sequence[int], *, pipeline_layers: int = 4,
                  first_alone: bool = True) -> tuple[mx.array, list[Record], list[int]]:
    """Several streams' windows in one forward, rows grouped by stream: logits [1, R, V], each stream's record
    (``commit`` it on the stream's caches) and each stream's first row. Every row keeps its stream-alone bits."""

    x, rows = _rows_forward(core, windows, parents, caches, starts, pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    return logits(head, x), rows.records, rows.offsets


def hidden_rows(core: Any, windows: Sequence[Any], caches: Sequence[list[Any]], *,
                starts: Sequence[int] | None = None, parents: Sequence[Sequence[int]] | None = None,
                pipeline_layers: int = 4, first_alone: bool = True) -> tuple[mx.array, list[Record]]:
    """A family round's forward: final-normed rows [1, R, D] of every stream's chain window (``logits`` gives the
    head) and each stream's record for ``keep_rows``. ``starts`` default to each stream's attention cache length."""

    if starts is None:
        starts = [next((int(c.offset) for c in cache if hasattr(c, "keys")), 0) for cache in caches]
    if parents is None:
        parents = [_CHAINS.get(n) or tuple(range(-1, n - 1))
                   for n in ((int(w.size) if isinstance(w, mx.array) else len(w)) for w in windows)]
    x, rows = _rows_forward(core, windows, parents, caches, starts, pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    return x, rows.records


def check_streams(core: Any, head: Any, make_cache: Callable[[], list[Any]], copy: Callable[[list[Any]], list[Any]],
                  mixes: Sequence[Sequence[int]] = ((1, 1), (1, 4), (8, 8), (3, 1, 8, 5), (16, 2)),
                  prefixes: Sequence[int] = (32, 21, 40, 9)) -> tuple[bool, list[str]]:
    """Whether several streams in one forward give every row the logits of its stream's window alone, bit for bit,
    and every stream's caches after keeping part of its window equal the caches of its own round. Streams start
    from different prompts (lengths ``prefixes``) and windows of the widths in ``mixes``. Returns (all equal,
    failures)."""

    def arrays(cache: list[Any]) -> list[mx.array]:
        out = []
        for item in cache:
            state = item.state
            out.extend(a for a in (state if isinstance(state, (list, tuple)) else [state]) if a is not None)
        return out

    vocab = int(core.embed_tokens["weight"].shape[0])   # MLX's gather reads past the table for larger ids, unchecked
    bases = []
    for s, length in enumerate(prefixes):
        cache = make_cache()
        prompt = [(1000 + 37 * s + 11 * i) % vocab for i in range(length)]
        step = min(WINDOW_ROWS, row_matmul.BACKEND.max_rows)
        for begin in range(0, length, step):
            chunk = prompt[begin:begin + step]
            _, record = forward(core, head, chunk, _CHAINS[len(chunk)], cache, begin)
            commit(cache, record, list(range(len(chunk))), len(chunk), begin)
        mx.eval(arrays(cache))
        bases.append((cache, length))
    failures: list[str] = []
    for mix in mixes:
        streams = [bases[s % len(bases)] for s in range(len(mix))]
        windows = [[(1500 + 13 * s + 7 * i) % vocab for i in range(w)] for s, w in enumerate(mix)]
        keeps = [max(1, (w + 1) // 2) for w in mix]
        alone = []
        for (cache, start), window, keep in zip(streams, windows, keeps):
            own = copy(cache)
            lg, record = forward(core, head, window, _CHAINS[len(window)], own, start)
            commit(own, record, list(range(keep)), len(window), start)
            mx.eval(lg, arrays(own))
            alone.append((lg, own))
        caches = [copy(cache) for cache, _ in streams]
        lg, records, offsets = multi_forward(core, head, windows, [_CHAINS[len(w)] for w in windows], caches,
                                             [start for _, start in streams])
        for cache, record, keep in zip(caches, records, keeps):
            keep_rows(cache, record, keep)
        mx.eval(lg, *[arrays(c) for c in caches])
        for s, ((ref, own), a, w) in enumerate(zip(alone, offsets, mix)):
            if not bool(mx.array_equal(lg[0, a:a + w], ref[0]).item()):
                failures.append(f"mix {tuple(mix)} stream {s}: logits differ")
            mine, theirs = arrays(caches[s]), arrays(own)
            if len(mine) != len(theirs) or not all(bool(mx.array_equal(x, y).item()) for x, y in zip(mine, theirs)):
                failures.append(f"mix {tuple(mix)} stream {s}: caches differ after keeping {keeps[s]} of {w} rows")
    return not failures, failures


__all__ = ["PROMPT_ROWS", "ROW_ATTENTION", "Record", "check_streams", "commit", "forward", "hidden_rows", "keep_rows",
           "multi_forward"]
