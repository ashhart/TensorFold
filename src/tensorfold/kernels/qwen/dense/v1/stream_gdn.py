"""Batch conv, recurrence and commits using each stream's own state and tree paths, preserving standalone bits."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.inputs import ints
from tensorfold.kernels.qwen.dense.v1.lane_tree import tree_paths

MAX_STREAMS = 8          # conv-state buffers per launch

# Use lane_fuse gdn_pre arithmetic with each stream's conv state and window rows in the grouped layout.
_PRE = r"""
  // one simdgroup per (row w, head): q heads [0, NK), k heads [NK, 2 NK), v heads [2 NK, 2 NK + NV)
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  const int st = row_stream[w];                         // the row's stream: its conv state
  const device bfloat16_t* CSb = CS0;
  switch (st) {
""" + "".join(f"    case {i}: CSb = CS{i}; break;\n" for i in range(1, MAX_STREAMS)) + r"""  }
  constexpr int C = 2 * NK * DK + NV * DV;
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? int(head) * DK : (isk ? NK * DK + (int(head) - NK) * DK : 2 * NK * DK + (int(head) - 2 * NK) * DV);
  constexpr int PER = DK / 32;                          // channels per lane (DK == DV)
  float vals[PER];
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];          // into [conv state rows; window rows]
      const float xv = row < TAPS - 1 ? float(CSb[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    const float conv = float(bfloat(acc));
    const float sig = float(bfloat(1.0f / (1.0f + metal::exp(-conv))));
    vals[j] = float(bfloat(conv * sig));                // SiLU, bf16 like mlx_lm's two ops
  }
  if (isq || isk) {
    float ss = 0.0f;
    for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
    ss = simd_sum(ss);
    const float inv = metal::rsqrt(ss / float(DK) + 1e-6f);
    // mlx_lm: q = (DK^-0.5)^2 * rms_norm(q), k = DK^-0.5 * rms_norm(k), scales rounded to bf16
    const float scale = isq ? float(bfloat(1.0f / float(DK))) : float(bfloat(metal::rsqrt(float(DK))));
    for (int j = 0; j < PER; j++) {
      const bfloat out = bfloat(scale * float(bfloat(vals[j] * inv)));
      if (isq) Q[(w * NK + head) * DK + lane * PER + j] = out;
      else Kout[(w * NK + head - NK) * DK + lane * PER + j] = out;
    }
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      // g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b) (mlx_lm's compute_g, sigmoid)
      const float s = float(bfloat(float(Ain[w * ZS + AO + hv]) + float(DT[hv])));
      const float sp = float(bfloat(metal::max(s, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(s)))));
      G[w * NV + hv] = metal::exp(-metal::exp(float(ALOG[hv])) * sp);
      BETA[w * NV + hv] = bfloat(1.0f / (1.0f + metal::exp(-float(Bin[w * ZS + BO + hv]))));
    }
  }
"""

_INPUTS = ["QKV", *[f"CS{i}" for i in range(MAX_STREAMS)], "CW", "windows", "Ain", "Bin", "ALOG", "DT", "row_stream"]
_kernel_cache: list[Any] = []


def source() -> str:
    return _PRE


def _kernel() -> Any:
    if not _kernel_cache:
        digest = hashlib.sha256(_PRE.encode()).hexdigest()[:16]
        _kernel_cache.append(mx.fast.metal_kernel(name=f"stream_gdn_pre_{digest}", input_names=_INPUTS,
                                                  output_names=["Q", "Kout", "Vout", "G", "BETA"], source=_PRE))
    return _kernel_cache[0]


class ConvPlan:
    """Index each row's final n_keep + 1 conv inputs from its stream's conv state and ancestor path into state rows plus all window rows."""

    def __init__(self, parents: Sequence[Sequence[int]], n_keep: int) -> None:
        if not 1 <= len(parents) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        flat: list[int] = []
        streams: list[int] = []
        first = 0
        for s, rp in enumerate(parents):
            _, paths = tree_paths(rp)
            for path in paths:
                rows = list(range(n_keep)) + [n_keep + first + r for r in path]
                flat += rows[-(n_keep + 1):]
                streams.append(s)
            first += len(rp)
        self.rows, self.streams = first, len(parents)
        self.windows = ints(flat)
        self.row_stream = ints(streams)


def gdn_pre(qkv: mx.array, states: Sequence[mx.array], conv_weight: mx.array, plan: ConvPlan, zba: mx.array,
            a_log: mx.array, dt_bias: mx.array, *, nk: int, nv: int, dk: int, dv: int) -> tuple[mx.array, ...]:
    """Return q/k [1, R, nk, dk], v [1, R, nv, dv], g [1, R, nv] fp32, beta [1, R, nv] from grouped projections and per-stream conv states."""

    R = int(qkv.shape[-2])
    C = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    zs = int(zba.shape[-1])
    if R != plan.rows or dk != dv or dk % 32 or zs != nv * dv + 2 * nv or len(states) > MAX_STREAMS:
        raise ValueError("stream_gdn.gdn_pre: rows, head sizes or [z | b | a] width do not match")
    spare = states[0]
    zba2 = zba.reshape(R, zs)
    cs = [s.reshape(taps - 1, C) for s in states] + [spare.reshape(taps - 1, C)] * (MAX_STREAMS - len(states))
    return tuple(_kernel()(
        inputs=[qkv.reshape(R, C), *cs, conv_weight.reshape(C, taps), plan.windows, zba2, zba2, a_log, dt_bias,
                plan.row_stream],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps),
                  ("ZS", zs), ("AO", nv * dv + nv), ("BO", nv * dv)],
        grid=(32, 2 * nk + nv, R), threadgroup=(32, 1, 1),
        output_shapes=[(1, R, nk, dk), (1, R, nk, dk), (1, R, nv, dv), (1, R, nv), (1, R, nv)],
        output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, qkv.dtype]))


# Walk nodes parents-first from each stream's committed state using mlx_lm gated_delta_step arithmetic.
_TREE = r"""
        auto n = thread_position_in_grid.z;                 // (stream, head)
        const int st = int(n) / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        const device float* Sb = S0;
        switch (st) {
""" + "".join(f"          case {i}: Sb = S{i}; break;\n" for i in range(1, MAX_STREAMS)) + r"""        }
        auto i_state = Sb + (hv_idx * Dv + dv_idx) * Dk;
        float s0[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          s0[i] = static_cast<float>(i_state[s_idx]);
        }
        float states[MAXW][n_per_t];
        const int first = meta[2 * st], W = meta[2 * st + 1];   // the stream's rows among all rows
        for (int node = 0; node < W; ++node) {
          const int parent = parents[first + node];
          float state[n_per_t];
          // a chain keeps one slot: each node's parent is the node before it
          for (int i = 0; i < n_per_t; ++i) state[i] = parent < 0 ? s0[i] : states[CHAIN ? 0 : parent][i];
          const int row = first + node;
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
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(row * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) states[CHAIN ? 0 : node][i] = state[i];
        }
"""

MAX_TREE = 32            # a branching window's rows (per-thread state slots); chains take up to 128

# Replay each stream's kept rows with lane_tree arithmetic, one threadgroup column per stream and head.
_REPLAY = r"""
        auto n = thread_position_in_grid.z;                 // (stream, head)
        const int st = int(n) / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        const device float* Sb = S0;
        device float* Ob = O0;
        switch (st) {
""" + "".join(f"          case {i}: Sb = S{i}; Ob = O{i}; break;\n" for i in range(1, MAX_STREAMS)) + r"""        }
        auto i_state = Sb + (hv_idx * Dv + dv_idx) * Dk;
        auto o_state = Ob + (hv_idx * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        const int steps = meta[2 * st + 1];
        for (int j = 0; j < steps; ++j) {                    // the accepted path's rows, in order
          const int row = rows[meta[2 * st] + j];
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
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
"""

# each stream's next conv state: the last taps - 1 rows of [its conv state; its kept rows], copied
_TAILS = r"""
  const uint c = thread_position_in_grid.x;
  const uint t = thread_position_in_grid.y;
  const uint st = thread_position_in_grid.z;
  const device bfloat16_t* Cb = CS0;
  device bfloat16_t* Tb = T0;
  switch (st) {
""" + "".join(f"    case {i}: Cb = CS{i}; Tb = T{i}; break;\n" for i in range(1, MAX_STREAMS)) + r"""  }
  const int src = tails[st * NKEEP + t];              // < NKEEP: a conv state row; else NKEEP + a row of all rows
  if (int(c) < C) Tb[t * C + c] = src < NKEEP ? Cb[src * C + c] : QKV[(src - NKEEP) * C + c];
"""

_commit_kernels: dict[str, Any] = {}


def _commit_kernel(name: str) -> Any:
    if name not in _commit_kernels:
        if name == "tree":
            inputs = ["q", "k", "v", "g", "beta", *[f"S{i}" for i in range(MAX_STREAMS)], "parents", "meta"]
            source, outputs = _TREE, ["y"]
        elif name == "replay":
            inputs = ["q", "k", "v", "g", "beta", *[f"S{i}" for i in range(MAX_STREAMS)], "rows", "meta"]
            source, outputs = _REPLAY, [f"O{i}" for i in range(MAX_STREAMS)]
        else:
            inputs = [*[f"CS{i}" for i in range(MAX_STREAMS)], "QKV", "tails"]
            source, outputs = _TAILS, [f"T{i}" for i in range(MAX_STREAMS)]
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        _commit_kernels[name] = mx.fast.metal_kernel(name=f"stream_gdn_{name}_{digest}", input_names=inputs,
                                                     output_names=outputs, source=source)
    return _commit_kernels[name]


class TreePlan:
    """Share each stream's window and parent indices across recurrent layers, with parents first and -1 denoting the root."""

    def __init__(self, parents: Sequence[Sequence[int]]) -> None:
        if not 1 <= len(parents) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        flat: list[int] = []
        meta: list[int] = []
        chain = True
        widest = 1
        for rp in parents:
            tree_paths(rp)                                  # parents come before their children
            meta += [len(flat), len(rp)]
            flat += [int(q) for q in rp]
            chain = chain and list(rp) == list(range(-1, len(rp) - 1))
            widest = max(widest, len(rp))
        if widest > (128 if chain else MAX_TREE):
            raise ValueError(f"stream_gdn: windows of {widest} rows (trees take up to {MAX_TREE}, chains 128)")
        self.streams, self.rows, self.chain = len(parents), len(flat), chain
        self.slots = 1 if chain else (16 if widest <= 16 else MAX_TREE)   # per-thread state slots (compiled variants)
        self.parents, self.meta = ints(flat), ints(meta)


def tree(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, states: Sequence[mx.array],
         plan: TreePlan) -> mx.array:
    """Return recurrence outputs [1, R, Hv, Dv] by walking each row's path from its stream's own fp32 state."""

    _, R, Hk, Dk = (int(d) for d in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    if R != plan.rows or any(s.dtype != mx.float32 for s in states):
        raise ValueError("stream_gdn.tree: rows or state dtype do not match")
    S = len(states)
    return _commit_kernel("tree")(
        inputs=[q, k, v, g, beta, *states, *[states[0]] * (MAX_STREAMS - S), plan.parents, plan.meta],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("MAXW", plan.slots),
                  ("CHAIN", plan.chain)],
        grid=(32, Dv, Hv * S), threadgroup=(32, 4, 1),
        output_shapes=[(1, R, Hv, Dv)], output_dtypes=[q.dtype])[0]


class CommitPlan:
    """Locate each stream's kept rows and next conv tail among grouped rows, with paths ordered root first."""

    def __init__(self, paths: Sequence[Sequence[int]], firsts: Sequence[int], n_keep: int) -> None:
        if not 1 <= len(paths) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        rows: list[int] = []
        meta: list[int] = []
        tails: list[int] = []
        for path, first in zip(paths, firsts):
            meta += [len(rows), len(path)]
            rows += [int(first) + int(r) for r in path]
            tails += (list(range(n_keep)) + [n_keep + int(first) + int(r) for r in path])[-n_keep:]
        self.streams, self.n_keep = len(paths), n_keep
        self.rows, self.meta, self.tails = ints(rows), ints(meta), ints(tails)


def replay(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, states: Sequence[mx.array],
           plan: CommitPlan) -> list[mx.array]:
    """Return each stream's state after serial steps over its kept rows using lane_tree replay arithmetic."""

    _, _, Hk, Dk = (int(d) for d in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    if any(s.dtype != mx.float32 for s in states):
        raise ValueError("stream_gdn.replay: fp32 states")
    S = len(states)
    spare = states[0]
    outs = _commit_kernel("replay")(
        inputs=[q, k, v, g, beta, *states, *[spare] * (MAX_STREAMS - S), plan.rows, plan.meta],
        template=[("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, Hv * S), threadgroup=(32, 4, 1),
        output_shapes=[states[0].shape] * S + [(16,)] * (MAX_STREAMS - S),
        output_dtypes=[mx.float32] * MAX_STREAMS)
    return list(outs[:S])


def conv_tails(states: Sequence[mx.array], qkv: mx.array, plan: CommitPlan) -> list[mx.array]:
    """Each stream's next conv state [1, n_keep, C]: the last n_keep rows of [its conv state; its kept rows]."""

    C = int(qkv.shape[-1])
    S = len(states)
    spare = states[0]
    outs = _commit_kernel("tails")(
        inputs=[*states, *[spare] * (MAX_STREAMS - S), qkv, plan.tails],
        template=[("C", C), ("NKEEP", plan.n_keep)],
        grid=(C, plan.n_keep, S), threadgroup=(256, 1, 1),
        output_shapes=[(1, plan.n_keep, C)] * S + [(16,)] * (MAX_STREAMS - S),
        output_dtypes=[qkv.dtype] * MAX_STREAMS)
    return list(outs[:S])


def sources() -> dict[str, str]:
    return {"pre": _PRE, "tree": _TREE, "replay": _REPLAY, "tails": _TAILS}


__all__ = ["CommitPlan", "ConvPlan", "MAX_STREAMS", "MAX_TREE", "TreePlan", "conv_tails", "gdn_pre", "replay",
           "source", "sources", "tree"]
