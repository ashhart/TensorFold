"""Draft-tree bookkeeping and the lane decoder's one-stream entry points (its forward and commit are lane_multi's)."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.inputs import MIN_ELEMENTS, ints

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
    """Walk each path from state [1, Hv, Dv, Dk], using q/k [1, W, Hk, Dk], v [1, W, Hv, Dv] and g/beta [1, W, Hv], to return [1, W, Hv, Dv]."""

    _, W, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    tree_paths(parents)                                   # validates the parent order
    chain = list(parents) == list(range(-1, W - 1))
    if W > (MAX_DEPTH if chain else MAX_TREE):
        raise ValueError(f"window of {W} rows: trees take up to {MAX_TREE}, chains {MAX_DEPTH}")
    maxw = 1 if chain else (16 if W <= 16 else MAX_TREE)  # per-thread state slots (compiled variants)
    y = _kernel("tree")(
        inputs=[mx.contiguous(q), mx.contiguous(k), mx.contiguous(v), mx.contiguous(g), mx.contiguous(beta),
                mx.contiguous(state), ints(parents), mx.array([W], dtype=mx.int32)],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("MAXW", maxw), ("CHAIN", chain)],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, W, Hv, Dv)], output_dtypes=[q.dtype])[0]
    return y


def _conv_windows(parents: Sequence[int], n_keep: int) -> mx.array:
    """Row w's conv inputs as rows of [conv state (n_keep rows); window rows]: its path's last n_keep + 1."""

    _, paths = tree_paths(parents)
    windows = []
    for path in paths:
        rows = list(range(n_keep)) + [n_keep + r for r in path]
        windows.append(rows[-(n_keep + 1):])
    while len(windows) * (n_keep + 1) < MIN_ELEMENTS:        # one source per kernel name (kernels.inputs)
        windows.append([0] * (n_keep + 1))
    return mx.array(windows, dtype=mx.int32)


# When a list, every tree_forward appends its rows' post-norm hidden [1, W, D] (a proposer that drafts from them reads them).
HIDDEN_SINK: list | None = None


def tree_forward(core: Any, head: Any, tokens: Sequence[int], parents: Sequence[int], cache: list[Any],
                 start: int, *, pipeline_layers: int = 4, last_only: bool = False,
                 first_alone: bool = True) -> tuple[mx.array, list[Any]]:
    """Return logits [1, W, V] and a commit record for a tree rooted at start, using lane_multi's shared kernels."""

    from tensorfold.kernels.qwen.dense.v1 import lane_multi

    logits, records, _ = lane_multi.multi_tree_forward(core, head, [tokens], [parents], [cache], [start],
                                                       pipeline_layers=pipeline_layers, first_alone=first_alone,
                                                       last_only=last_only)
    return logits, records[0]


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
    """Keep only ``path``'s rows of the last ``window``: ``lane_multi.commit_streams`` with one stream."""

    from tensorfold.kernels.qwen.dense.v1 import lane_multi

    lane_multi.commit_streams([cache], [record], [path], [window], [start])


__all__ = ["MAX_DEPTH", "MAX_TREE", "accept_path", "commit_tree", "gated_delta_tree", "replay_path", "tree_forward",
           "tree_paths"]
