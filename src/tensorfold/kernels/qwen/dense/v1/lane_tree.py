"""Exact draft-tree verification pieces: the recurrence walked per tree node.

A draft tree's nodes share a committed prefix and branch after it. A Gated
DeltaNet layer's output at node v is its recurrence run from the prefix state
S_P along v's path (root -> v). ``gated_delta_tree`` runs that walk for every
node in parallel (one threadgroup column per node), with each step's code copied
from mlx_lm's ``gated_delta_step`` kernel, so node v's output has the bits of
serial decoding's step at v's position. No node's state is stored; after
verification the accepted path is replayed into the cache with mlx_lm's own
kernel.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

MAX_DEPTH = 128         # rows of a window (trees up to 32 rows; chains up to 128)
MAX_TREE = 32

_TREE_SOURCE = r"""
        auto n = thread_position_in_grid.z;                 // head
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [1, Hv, Dv, Dk] (the committed prefix's state), read once
        auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        float s0[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          s0[i] = static_cast<float>(i_state[s_idx]);
        }
        // nodes in row order (parents first): each node's state is one step from its parent's
        float states[MAXW][n_per_t];
        const int W = nodes[0];
        for (int node = 0; node < W; ++node) {
          const int parent = parents[node];
          float state[n_per_t];
          // a chain keeps one slot: each node's parent is the node before it
          for (int i = 0; i < n_per_t; ++i) state[i] = parent < 0 ? s0[i] : states[CHAIN ? 0 : parent][i];
          auto q_ = q + (node * Hk + hk_idx) * Dk;
          auto k_ = k + (node * Hk + hk_idx) * Dk;
          auto v_ = v + (node * Hv + hv_idx) * Dv;
          const float g_ = static_cast<float>(g[node * Hv + hv_idx]);
          const float beta_ = static_cast<float>(beta[node * Hv + hv_idx]);
          // --- mlx_lm gated_delta_step, one step, verbatim arithmetic ---
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(node * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) states[CHAIN ? 0 : node][i] = state[i];
        }
"""

_REPLAY_SOURCE = r"""
        auto n = thread_position_in_grid.z;                 // head
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        const int steps = count[0];
        for (int j = 0; j < steps; ++j) {                    // the accepted path's rows, in order
          const int row = rows[j];
          auto q_ = q + (row * Hk + hk_idx) * Dk;
          auto k_ = k + (row * Hk + hk_idx) * Dk;
          auto v_ = v + (row * Hv + hv_idx) * Dv;
          const float g_ = static_cast<float>(g[row * Hv + hv_idx]);
          const float beta_ = static_cast<float>(beta[row * Hv + hv_idx]);
          // --- mlx_lm gated_delta_step, one step, verbatim arithmetic ---
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
          }
        }
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
"""



_kernels: dict[str, Any] = {}


def _kernel(name: str = "tree") -> Any:
    if name not in _kernels:
        if name == "tree":
            digest = hashlib.sha256(_TREE_SOURCE.encode()).hexdigest()[:16]
            _kernels[name] = mx.fast.metal_kernel(
                name=f"gated_delta_tree_{digest}",
                input_names=["q", "k", "v", "g", "beta", "state_in", "parents", "nodes"],
                output_names=["y"], source=_TREE_SOURCE)
        else:
            digest = hashlib.sha256(_REPLAY_SOURCE.encode()).hexdigest()[:16]
            _kernels[name] = mx.fast.metal_kernel(
                name=f"gated_delta_replay_{digest}",
                input_names=["q", "k", "v", "g", "beta", "state_in", "rows", "count"],
                output_names=["state_out"], source=_REPLAY_SOURCE)
    return _kernels[name]


def replay_path(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array,
                rows: mx.array, count: mx.array) -> mx.array:
    """The state after serial steps over ``rows`` (mlx_lm's step arithmetic), in one kernel."""

    _, _, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    return _kernel("replay")(
        inputs=[q, k, v, g, beta, state, rows, count],
        template=[("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("StT", state.dtype)],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[state.shape], output_dtypes=[state.dtype])[0]


def tree_paths(parents: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    """(depths, paths): each node's depth and its path of row indices from the root."""

    depths: list[int] = []
    paths: list[list[int]] = []
    for row, parent in enumerate(parents):
        if parent < 0:
            path = [row]
        else:
            if parent >= row:
                raise ValueError("parents must come before their children")
            path = paths[parent] + [row]
        paths.append(path)
        depths.append(len(path) - 1)
    return depths, paths


def gated_delta_tree(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
                     state: mx.array, parents: Sequence[int]) -> mx.array:
    """Per-node outputs [1, W, Hv, Dv] of the recurrence walked from ``state`` along each path.

    q, k: [1, W, Hk, Dk]; v: [1, W, Hv, Dv]; g, beta: [1, W, Hv]; state: [1, Hv, Dv, Dk].
    """

    _, W, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    tree_paths(parents)                                   # validates the parent order
    chain = list(parents) == list(range(-1, W - 1))
    if W > (MAX_DEPTH if chain else MAX_TREE):
        raise ValueError(f"window of {W} rows: trees take up to {MAX_TREE}, chains {MAX_DEPTH}")
    maxw = 1 if chain else (16 if W <= 16 else MAX_TREE)  # per-thread state slots (compiled variants)
    y = _kernel("tree")(
        inputs=[mx.contiguous(q), mx.contiguous(k), mx.contiguous(v), mx.contiguous(g), mx.contiguous(beta),
                mx.contiguous(state), mx.array(list(parents), dtype=mx.int32), mx.array([W], dtype=mx.int32)],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("MAXW", maxw), ("CHAIN", chain)],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, W, Hv, Dv)], output_dtypes=[q.dtype])[0]
    return y




# -- the target model over a tree window --------------------------------------------------------

def _positions(parents: Sequence[int], start: int) -> list[int]:
    depths, _ = tree_paths(parents)
    return [start + d for d in depths]


def _attention(attn: Any, x: mx.array, cache: Any, parents: Sequence[int], positions: list[int],
               record: list[Any]) -> mx.array:
    """Qwen3-Next attention for tree rows (mlx_lm's call, per-row RoPE, tree-exact attention)."""

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse
    from tensorfold.kernels.qwen.dense.v1.lane_attention import lane_tree_sdpa

    B, L, _ = x.shape
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(q_proj_output.reshape(B, L, H, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    kv = lane_fuse.attn_kv(attn, x)                   # [k | v] in one lane matmul (None: two calls)
    if kv is None:
        keys, values = attn.k_proj(x), attn.v_proj(x)
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys.reshape(B, L, nkv, -1))
        values = values.reshape(B, L, nkv, -1)
    else:
        # RMSNorm runs per head row, so it takes whole outputs and the rows it should skip are
        # dropped after (each row's bits are its own): a slice of rows would first be copied
        D = int(queries.shape[-1])
        queries = attn.q_norm(q_proj_output.reshape(B, L, 2 * H, D))[:, :, 0::2]   # [q_h | gate_h] per head
        kv = kv.reshape(B, L, 2 * nkv, -1)
        keys = attn.k_norm(kv)[:, :, :nkv]
        values = kv[:, :, nkv:]
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = mx.array(positions, dtype=mx.int32)
    # one position per row: rows go to the batch axis, where RoPE takes an offset each
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    record.append(("kv", keys, values))        # the window's rows, for the commit's moves
    keys, values = cache.update_and_fetch(keys, values)
    output = lane_tree_sdpa(queries, keys, values, attn.scale, parents)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output * mx.sigmoid(gate))


def _gdn(gdn: Any, x: mx.array, cache: Any, parents: Sequence[int], windows: mx.array,
         record: list[Any]) -> mx.array:
    """Gated DeltaNet for tree rows: per-node conv windows and the node-order recurrence.

    ``x`` carries its group sums (``lane_glue.norm_xs``); the conv, SiLU, q/k norms and gates
    run as one kernel (``lane_glue.gdn_pre``), the gated norm as another.
    """

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue

    B, S, _ = x.shape
    qkv = gdn.in_proj_qkv(x)
    zba = lane_fuse.gdn_in(gdn, x)                    # [z | b | a] in one lane matmul (None: three calls)
    if zba is None:
        z = gdn.in_proj_z(x)
        b = gdn.in_proj_b(x)
        a = gdn.in_proj_a(x)
    n_keep = gdn.conv_kernel_size - 1
    conv_state = cache[0] if cache[0] is not None else mx.zeros((B, n_keep, gdn.conv_dim), dtype=x.dtype)
    heads = dict(nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    if zba is None:
        q, k, v, g, beta = lane_glue.gdn_pre(qkv, conv_state, gdn.conv1d.weight, windows, a, b, gdn.A_log,
                                             gdn.dt_bias, **heads)
    else:                                             # b and a read in place from the stacked rows
        q, k, v, g, beta = lane_fuse.gdn_pre(qkv, conv_state, gdn.conv1d.weight, windows, zba, gdn.A_log,
                                             gdn.dt_bias, **heads)
    state = cache[1]
    if state is None:
        state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    # (Running the last round's commit replay inside this kernel, same bits, was slower live: 61.03 against
    # 60.16 ms a round, 700 rounds each, 2026-09-24. Left as its own kernel, the replay runs beside the
    # forward's early layers.)
    y = gated_delta_tree(q, k, v, g, beta, state, parents)
    seq = mx.concatenate([conv_state, qkv], axis=1)[0]           # [n_keep + W, C], read by the commit only
    record.append(("gdn", q, k, v, g, beta, state, seq, n_keep))
    if zba is None:
        out = lane_glue.gdn_post(y, z, gdn.norm.weight, gdn.norm.eps)
    else:
        out = lane_fuse.gdn_post(y, zba, gdn.norm.weight, gdn.norm.eps)
    return gdn.out_proj(out)


def _conv_windows(parents: Sequence[int], n_keep: int) -> mx.array:
    """Row w's conv inputs as rows of [conv state (n_keep rows); window rows]: its path's last n_keep + 1."""

    _, paths = tree_paths(parents)
    windows = []
    for path in paths:
        rows = list(range(n_keep)) + [n_keep + r for r in path]
        windows.append(rows[-(n_keep + 1):])
    return mx.array(windows, dtype=mx.int32)


# When a list, every tree_forward appends its rows' post-norm hidden [1, W, D] (a proposer that drafts from them reads them).
HIDDEN_SINK: list | None = None


def tree_forward(core: Any, head: Any, tokens: Sequence[int], parents: Sequence[int], cache: list[Any],
                 start: int, *, pipeline_layers: int = 4, last_only: bool = False,
                 first_alone: bool = True) -> tuple[mx.array, list[Any]]:
    """Logits [1, W, V] for a tree window whose root sits at position ``start``.

    Attention layers append the W rows to their caches (compacted by ``commit_tree``);
    recurrent layers leave their state untouched and record what ``commit_tree`` replays.
    DFlash tap hooks (``_LayerHook``) get the layer outputs as a normal forward would give.
    Each residual add runs inside the next norm's kernel (``lane_glue.norm_xs``).
    """

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue

    positions = _positions(parents, start)
    hidden = core.embed_tokens(mx.array([list(tokens)], dtype=mx.uint32))
    record: list[Any] = []
    layers = list(core.layers)
    windows = None
    pending: mx.array | None = None                   # the last MLP's output, not yet added
    tapped: Any = None                                # (storage, index) waiting for this layer's input
    for index, (layer, item) in enumerate(zip(layers, cache)):
        inner = getattr(layer, "_layer", layer)
        norm = inner.input_layernorm
        hidden, x = lane_glue.norm_xs(hidden, pending, norm.weight, norm.eps)
        if tapped is not None:
            tapped[0][tapped[1]] = hidden
        if getattr(inner, "is_linear", False):
            if windows is None:
                windows = _conv_windows(parents, inner.linear_attn.conv_kernel_size - 1)
            r = _gdn(inner.linear_attn, x, item, parents, windows, record)
        else:
            r = _attention(inner.self_attn, x, item, parents, positions, record)
        norm = inner.post_attention_layernorm
        hidden, x = lane_glue.norm_xs(hidden, r, norm.weight, norm.eps)
        mlp = inner.mlp
        gu = lane_fuse.mlp_gate_up(mlp, x)            # [gate | up] in one lane matmul (None: two calls)
        act = lane_glue.mlp_act(mlp.gate_proj(x), mlp.up_proj(x)) if gu is None else lane_fuse.mlp_act(gu)
        pending = mlp.down_proj(act)
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        # the first layer goes alone: the GPU starts ~0.4 ms sooner than on a whole first slice (it idles until then)
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) and index + 1 < len(layers):
            mx.async_eval(hidden, pending)
    hidden, x = lane_glue.norm_xs(hidden, pending, core.norm.weight, core.norm.eps)
    if tapped is not None:
        tapped[0][tapped[1]] = hidden
    if HIDDEN_SINK is not None:                       # a hidden-state proposer reads every row's post-norm hidden
        HIDDEN_SINK.append(x)
        if len(HIDDEN_SINK) > 1024:
            del HIDDEN_SINK[0]
    return head(x[:, -1:] if last_only else x), record


def accept_path(tokens: Sequence[int], parents: Sequence[int], preds: Sequence[int]) -> list[int]:
    """Rows of the accepted path from the root: follow the child whose token is the target's pick."""

    children: dict[int, list[int]] = {}
    for row, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(parent, []).append(row)
    path = [0]
    while True:
        want = int(preds[path[-1]])
        nxt = next((c for c in children.get(path[-1], []) if int(tokens[c]) == want), None)
        if nxt is None:
            return path
        path.append(nxt)


def commit_tree(cache: list[Any], record: list[Any], path: Sequence[int], window: int, start: int) -> None:
    """Keep only ``path``'s rows: attention keys moved to their logical slots, recurrence replayed."""

    keep = len(path)
    rows = mx.array(list(path), dtype=mx.int32)
    count = mx.array([keep], dtype=mx.int32)
    in_place = list(path) == list(range(keep))
    tails: dict[int, mx.array] = {}    # conv-tail rows, built once for all recurrent layers (48 identical arrays before)
    j = 0
    for item in cache:
        kind, *entry = record[j]
        j += 1
        if hasattr(item, "keys") and hasattr(item, "values"):
            if kind != "kv":
                raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has an attention layer")
            if not in_place:
                # taken from the window's own rows, not the cache buffer: the buffer then has one
                # owner and the slice update writes in place (a take from it copied the whole cache)
                win_k, win_v = entry
                item.keys[..., start:start + keep, :] = mx.take(win_k, rows, axis=2)
                item.values[..., start:start + keep, :] = mx.take(win_v, rows, axis=2)
            item.trim(window - keep)
            continue
        if kind != "gdn":
            raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has a recurrent layer")
        q, k, v, g, beta, state0, seq, n_keep = entry
        item[1] = replay_path(q, k, v, g, beta, state0, rows, count)
        if n_keep not in tails:           # the last n_keep of [conv state rows; the path's window rows]
            tails[n_keep] = mx.array((list(range(n_keep)) + [n_keep + int(r) for r in path])[-n_keep:], dtype=mx.int32)
        item[0] = mx.contiguous(mx.take(seq, tails[n_keep], axis=0)[None])
        item.advance(keep)
    if j != len(record):
        raise RuntimeError(f"recorded {len(record)} layers, cache has {j}")


__all__ = ["MAX_DEPTH", "accept_path", "commit_tree", "gated_delta_tree", "tree_forward", "tree_paths"]
