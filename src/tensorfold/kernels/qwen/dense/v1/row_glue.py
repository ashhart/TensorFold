"""Row-independent norms, conv windows, recurrence and activations for the decoder without tensor units."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import mlx.core as mx

from tensorfold.kernels import threads
from tensorfold.kernels.inputs import ints
from tensorfold.kernels.qwen.dense.v1.row_matmul import WINDOW_ROWS

# lane_glue's arithmetic without the M5 matmul's group sums and row padding, reading stacked rows in place

_NORM = r"""
  // residual add + RMSNorm of row m: one threadgroup of K / 16 threads, thread t holds [16 t, 16 t + 16); the
  // row's sum of squares is each thread's sequential fma over its 16, then simd_sum, then the simdgroups in order
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    RESIDUAL_ADD
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  for (int i = 0; i < E; i++) XO[base + i] = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
"""

_GDN_POST = r"""
  // one simdgroup per (row m, v head): SiLU(z) * RMSNorm(y) * w, z read in place from the [qkv | z | b | a] rows
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    OUT[m * NV * DV + hv * DV + d] = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
  }
"""

_MLP_ACT = r"""
  // SiLU(gate) * up over [gate | up] rows of 2N
  const uint i = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (i >= uint(N)) return;
  const float gf = float(GU[m * 2 * N + i]);
  HOUT[m * N + i] = bfloat(gf / (1.0f + metal::exp(-gf)) * float(GU[m * 2 * N + N + i]));
"""

def _gdn_pre_source() -> str:
    """Read stacked [qkv | z | b | a] rows in place with lane_glue arithmetic and write each row's conv tail."""

    from tensorfold.kernels.qwen.dense.v1 import lane_glue
    from tensorfold.kernels.qwen.dense.v1.lane_fuse import _replace_once

    pre = _replace_once(lane_glue._GDN_PRE, "float(QKV[(row - (TAPS - 1)) * C + c])",
                        "float(QKV[(row - (TAPS - 1)) * ZS + c])")
    pre = _replace_once(pre, "float(Ain[w * NV + hv])", "float(Ain[w * ZS + AO + hv])")
    pre = _replace_once(pre, "float(Bin[w * NV + hv])", "float(Bin[w * ZS + BO + hv])")
    return pre + """
  // the conv tail after this row (the last TAPS - 1 inputs of its window): a commit keeping any path takes its last row's
  for (int r = 0; r < TAPS - 1; r++) {
    const int row = windows[w * TAPS + 1 + r];
    for (int j = 0; j < PER; j++) {
      const int c = c0 + int(lane) * PER + j;
      CO[(int(w) * (TAPS - 1) + r) * C + c] = row < TAPS - 1 ? CS[row * C + c] : QKV[(row - (TAPS - 1)) * ZS + c];
    }
  }
"""


def _tree_source() -> str:
    """lane_tree's recurrence over window nodes, which for a chain also writes the state after its last row."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    return lane_tree._TREE_SOURCE + """
        if (CHAIN) {
          auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = states[0][i];
        }
"""


# Use lane_tree step arithmetic for chains, compiling the row count and loading the next inputs during each step.
_CHAIN = r"""
        auto n = thread_position_in_grid.z;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        float state[n_per_t], kc[n_per_t], qc[n_per_t], kn[n_per_t], qn[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          kc[i] = static_cast<float>(k[hk_idx * Dk + n_per_t * dk_idx + i]);
          qc[i] = static_cast<float>(q[hk_idx * Dk + n_per_t * dk_idx + i]);
        }
        float vc = static_cast<float>(v[hv_idx * Dv + dv_idx]);
        float gc = static_cast<float>(g[hv_idx]);
        float bc = static_cast<float>(beta[hv_idx]);
        float vn = 0.0f, gn = 0.0f, bn = 0.0f;
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        #pragma unroll
        for (int node = 0; node < W; ++node) {
          if (node + 1 < W) {
            for (int i = 0; i < n_per_t; ++i) {
              kn[i] = static_cast<float>(k[((node + 1) * Hk + hk_idx) * Dk + n_per_t * dk_idx + i]);
              qn[i] = static_cast<float>(q[((node + 1) * Hk + hk_idx) * Dk + n_per_t * dk_idx + i]);
            }
            vn = static_cast<float>(v[((node + 1) * Hv + hv_idx) * Dv + dv_idx]);
            gn = static_cast<float>(g[(node + 1) * Hv + hv_idx]);
            bn = static_cast<float>(beta[(node + 1) * Hv + hv_idx]);
          }
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (vc - kv_mem) * bc;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] + kc[i] * delta;
            out += state[i] * qc[i];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(node * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) { kc[i] = kn[i]; qc[i] = qn[i]; }
          vc = vn; gc = gn; bc = bn;
        }
        auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
"""


_SPECS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "norm": (_NORM.replace("RESIDUAL_ADD", "h = bfloat(float(h) + float(R[base + i]));\n    HO[base + i] = h;"), "",
             ["H", "R", "Wt", "eps"], ["HO", "XO"]),
    "norm_nores": (_NORM.replace("RESIDUAL_ADD", ""), "", ["H", "Wt", "eps"], ["XO"]),
    "gdn_post": (_GDN_POST, "", ["Y", "Z", "NW", "eps"], ["OUT"]),
    "mlp_act": (_MLP_ACT, "", ["GU"], ["HOUT"]),
    "gdn_pre": (_gdn_pre_source(), "", ["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT", "nodes"],
                ["Q", "Kout", "Vout", "G", "BETA", "CO"]),
    "tree": (_tree_source(), "", ["q", "k", "v", "g", "beta", "state_in", "parents", "nodes"], ["y", "state_out"]),
    "chain": (_CHAIN, "", ["q", "k", "v", "g", "beta", "state_in"], ["y", "state_out"]),
}


_kernels: dict[Any, tuple[str, Any]] = {}


def sources() -> dict[str, str]:
    return {name: header + source for name, (source, header, _, _) in _SPECS.items()}


def _kernel(name: str, K: int = 0) -> Any:
    """The kernel; a norm's is per hidden size K, which it keeps in its source to reserve its K / 16 threads."""

    hit = _kernels.get((name, K))
    if hit is None:
        source, header, inputs, outputs = _SPECS[name]
        if K:
            source, header = f"  constexpr int K = {K};\n" + source, header + threads.reserve(K // 16)
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
        kernel = mx.fast.metal_kernel(name=f"row_forward_{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source, header=header)
        hit = _kernels[(name, K)] = (source, kernel)
    return hit[1]


_consts: dict[Any, mx.array] = {}


def _const(key: Any, make: Callable[[], mx.array]) -> mx.array:
    hit = _consts.get(key)
    if hit is None:
        if len(_consts) >= 4096:        # tree shapes are many: keep the cache bounded
            _consts.clear()
        hit = _consts[key] = make()
    return hit


def _eps(eps: float) -> mx.array:
    return _const(("eps", float(eps)), lambda: mx.array([float(eps)], dtype=mx.float32))


def add_norm(hidden: mx.array, residual: mx.array | None, weight: mx.array, eps: float) -> tuple[mx.array, mx.array]:
    """(h, x): h = hidden + residual (hidden when residual is None), x = RMSNorm(h) * weight; (1, M, K) bf16."""

    lead = hidden.shape[:-1]
    K = int(hidden.shape[-1])
    M = hidden.size // K
    if K % 512 or K > 16384:
        raise ValueError(f"norm: the hidden size must be a multiple of 512 up to 16384, got {K}")
    common = dict(grid=(K // 16, M, 1), threadgroup=(K // 16, 1, 1))
    if residual is None:
        x = _kernel("norm_nores", K)(inputs=[hidden.reshape(M, K), weight, _eps(eps)], output_shapes=[(M, K)],
                                     output_dtypes=[mx.bfloat16], **common)[0]
        return hidden, x.reshape(*lead, K)
    h, x = _kernel("norm", K)(inputs=[hidden.reshape(M, K), residual.reshape(M, K), weight, _eps(eps)],
                              output_shapes=[(M, K), (M, K)], output_dtypes=[mx.bfloat16, mx.bfloat16], **common)
    return h.reshape(*lead, K), x.reshape(*lead, K)


def gdn_post(rec: mx.array, y: mx.array, weight: mx.array, eps: float, *, zo: int) -> mx.array:
    """SiLU(z) * RMSNorm(rec) * weight per head, z read in place from the stacked rows ``y`` (z at column ``zo``)."""

    _, W, nv, dv = (int(s) for s in rec.shape)
    zs = int(y.shape[-1])
    return _kernel("gdn_post")(
        inputs=[rec, y.reshape(W, zs), weight, _eps(eps)], template=[("NV", nv), ("DV", dv), ("ZS", zs), ("ZO", zo)],
        grid=(32, nv, W), threadgroup=(32, 1, 1), output_shapes=[(1, W, nv * dv)], output_dtypes=[rec.dtype])[0]


def mlp_act(gu: mx.array) -> mx.array:
    """SiLU(gate) * up over [gate | up] rows (..., 2N) -> (..., N)."""

    N2 = int(gu.shape[-1])
    N = N2 // 2
    W = gu.size // N2
    return _kernel("mlp_act")(
        inputs=[gu.reshape(W, N2)], template=[("N", N)], grid=(256 * (-(-N // 256)), W, 1), threadgroup=(256, 1, 1),
        output_shapes=[(W, N)], output_dtypes=[gu.dtype])[0].reshape(*gu.shape[:-1], N)


def gdn_pre(y: mx.array, conv_state: mx.array, conv_weight: mx.array, windows: mx.array, a_log: mx.array,
            dt_bias: mx.array, *, nk: int, nv: int, dk: int, dv: int) -> tuple[mx.array, ...]:
    """Return q/k [1, W, nk, dk], v [1, W, nv, dv], g [1, W, nv] fp32, beta [1, W, nv], and conv tails [W, taps - 1, C] from stacked rows."""

    zs = int(y.shape[-1])
    W = y.size // zs
    C = 2 * nk * dk + nv * dv
    taps = int(conv_weight.shape[1])
    if zs != C + nv * dv + 2 * nv or dk != dv or dk % 32:
        raise ValueError(f"gdn_pre: [qkv | z | b | a] rows of {C + nv * dv + 2 * nv} and head dims multiple of 32 "
                         f"expected, got {zs}")
    y2 = y.reshape(W, zs)
    nodes = _const(("nodes", W), lambda: mx.array([W], dtype=mx.int32))
    return tuple(_kernel("gdn_pre")(
        inputs=[y2, conv_state.reshape(taps - 1, C), conv_weight.reshape(C, taps), windows, y2, y2, a_log, dt_bias,
                nodes],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("ZS", zs),
                  ("AO", C + nv * dv + nv), ("BO", C + nv * dv)],
        grid=(32, 2 * nk + nv, W), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nk, dk), (1, W, nk, dk), (1, W, nv, dv), (1, W, nv), (1, W, nv), (W, taps - 1, C)],
        output_dtypes=[y.dtype, y.dtype, y.dtype, mx.float32, y.dtype, y.dtype]))


def gated_delta(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array,
                parents: Sequence[int], *, chain: bool | None = None) -> tuple[mx.array, mx.array]:
    """Walk each node's path from state and return its output, plus the final state for chains; chain may be supplied after parent validation."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    _, W, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    if chain is None:
        lane_tree.tree_paths(parents)                               # validates the parent order
        chain = _chain(parents)
    if W > (lane_tree.MAX_DEPTH if chain else lane_tree.MAX_TREE):
        raise ValueError(f"window of {W} rows: trees take up to {lane_tree.MAX_TREE}, chains {lane_tree.MAX_DEPTH}")
    if chain and W <= WINDOW_ROWS:
        return tuple(_kernel("chain")(
            inputs=[q, k, v, g, beta, state],
            template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("W", W)],
            grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
            output_shapes=[(1, W, Hv, Dv), tuple(state.shape)], output_dtypes=[q.dtype, mx.float32]))
    # per-thread state slots: a chain keeps one; a tree one a node (fewer registers for small trees)
    maxw = 1 if chain else (8 if W <= 8 else (16 if W <= 16 else lane_tree.MAX_TREE))
    parents_a = _const(("parents", tuple(parents)), lambda: ints(parents))
    nodes = _const(("nodes", W), lambda: mx.array([W], dtype=mx.int32))
    y, state_out = _kernel("tree")(
        inputs=[q, k, v, g, beta, state, parents_a, nodes],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("MAXW", maxw), ("CHAIN", chain)],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, W, Hv, Dv), tuple(state.shape)], output_dtypes=[q.dtype, mx.float32])
    return y, state_out


def _chain(parents: Sequence[int]) -> bool:
    return list(parents) == list(range(-1, len(parents) - 1))


__all__ = ["add_norm", "gated_delta", "gdn_post", "gdn_pre", "mlp_act", "sources"]
