"""Nemotron-H decode glue in a few kernels (Nemotron 3.5 Lightning 30B-A3B).

mlx_lm's decode step runs ~900 small kernels a token (7.3 ms on the M5 Max and
on the M3 Ultra, where the weights alone take 3.7 and 2.2 ms). Between the
matmuls each step is one kernel here:

    add_norm     residual add (with the MoE combine) + the next block's RMSNorm
    route        sigmoid scores + correction bias -> top 6 experts and their weights
    mamba_step   conv window + conv + SiLU + dt + SSM state update + D skip + SiLU(z) gate
    group_norm   the Mamba output's RMSNorm over groups of 512

Every kernel takes R rows of consecutive tokens and treats them in order, and
a row's value depends only on its own inputs and the rows before it, so the
bits of a row do not depend on how many rows ride with it. The arithmetic
follows mlx_lm's (fp32 math, bf16 where mlx_lm stores bf16) but is its own:
serial decoding goes through these kernels too, so they define the reference.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

_ADD_NORM = r"""
  // one threadgroup of T threads per row; thread t owns elements t, t + T, t + 2T, ...
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float partial[T / 32];
  float hv[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const int at = int(r) * D + c;
    float delta;
    MIX
    const bfloat hn = bfloat(float(H[at]) + delta);
    HN[at] = hn;
    hv[i] = float(hn);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    OUT[int(r) * D + c] = bfloat(float(W[c]) * (hv[i] * scale));
  }
"""

# experts with the shared one folded in: bf16(sum_e w_e y_e)
_MIX_EXPERTS = r"""{
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(routed));
    }"""
# plain residual: the block's output
_MIX_PLAIN = "delta = float(X[at]);"
# MoE: sum_e w_e y_e (fp32, experts in order) rounded to bf16, plus the shared expert (bf16 add), as mlx_lm
_MIX_MOE = r"""{
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(float(bfloat(routed)) + float(SH[at])));
    }"""

_ROUTE = r"""
  // one simdgroup per row: lane l holds experts l, l + 32, l + 64, l + 96
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sel[NE / 32], prob[NE / 32];
  for (int j = 0; j < NE / 32; j++) {
    const int e = int(lane) + 32 * j;
    const float g = float(G[int(r) * NE + e]);
    prob[j] = 1.0f / (1.0f + metal::exp(-g));
    sel[j] = prob[j] + bias[e];
  }
  float total = 0.0f;
  float picked[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      const int e = int(lane) + 32 * j;
      if (sel[j] > best) { best = sel[j]; best_e = e; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));   // ties: the lowest expert id
    float p = 0.0f;
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
    }
    p = simd_sum(p);
    picked[k] = p;
    total += p;
    if (lane == 0) IDX[int(r) * OK + k] = uint(winner);
  }
  if (lane == 0) {
    const float denominator = total + 1e-20f;
    for (int k = 0; k < K; k++) WT[int(r) * OK + k] = picked[k] / denominator * scaling[0];
    for (int k = K; k < OK; k++) { IDX[int(r) * OK + k] = uint(NE + k - K); WT[int(r) * OK + k] = 1.0f; }
  }
"""

_MAMBA_STEP = r"""
  // grid (32, DH, H): lane = 4 state elements, y = channel d of head h. Rows are consecutive tokens.
  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int R = dims[0];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;          // conv channels: x, B, C
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  float st[NS];
  const int sbase = (cx * DS) + int(lane) * NS;
  for (int i = 0; i < NS; i++) st[i] = float(S_IN[sbase + i]);
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);

  // conv of channel ch at row rr: taps over inputs rr-3 .. rr (rows < 0 come from the conv state)
  #define TAP(ch, pos) ((pos) < 0 ? float(CS_IN[((pos) + KC - 1) * CD + (ch)]) : float(P[(pos) * PROJ + XOFF + (ch)]))
  #define CONV(ch, rr, out) { \
      float a_ = float(CB[ch]); \
      for (int k_ = 0; k_ < KC; k_++) a_ = fma(CW[k_ * CD + (ch)], TAP(ch, (rr) - (KC - 1) + k_), a_); \
      const float cv_ = float(bfloat(a_)); \
      out = float(bfloat(cv_ / (1.0f + metal::exp(-cv_)))); }

  // B and C of this head's group, every row, computed once per threadgroup (thread tid owns one of 2 DS channels)
  threadgroup float bc[MAXR * 2 * DS];
  const uint tid = thread_position_in_threadgroup.y * 32 + lane;
  for (int rr = 0; rr < R; rr++) {
    for (uint c = tid; c < 2 * DS; c += 32 * TGY) {
      const int ch = c < DS ? XD + int(g) * DS + int(c) : XD + NG * DS + int(g) * DS + int(c - DS);
      float v; CONV(ch, rr, v);
      bc[rr * 2 * DS + c] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int rr = 0; rr < R; rr++) {
    float xv = 0.0f;
    if (lane == 0) { CONV(cx, rr, xv); }
    xv = simd_broadcast(xv, 0);
    float bv[NS], cvv[NS];
    for (int i = 0; i < NS; i++) {
      bv[i] = bc[rr * 2 * DS + int(lane) * NS + i];
      cvv[i] = bc[rr * 2 * DS + DS + int(lane) * NS + i];
    }
    float dt = float(P[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));   // softplus (logaddexp(x, 0))
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float s = dA * st[i] + xdt * bv[i];
      st[i] = s;
      acc += s * cvv[i];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
    }
    // the SSM state after this row (a verify window keeps the state of its last accepted row)
    for (int i = 0; i < NS; i++) S_OUT[size_t(rr) * SSZ + sbase + i] = st[i];
    // the conv state after this row: its last KC-1 inputs; each channel written by one thread
    if (lane == 0) {
      for (int k = 0; k < KC - 1; k++) {
        const int pos = rr - (KC - 2) + k;
        CS_OUT[(rr * (KC - 1) + k) * CD + cx] = pos < 0 ? CS_IN[(pos + KC - 1) * CD + cx] : P[pos * PROJ + XOFF + cx];
      }
    }
    if ((h % (H / NG)) == 0 && d == 0) {
      for (int i = 0; i < NS; i++) {
        for (int k = 0; k < KC - 1; k++) {
          const int pos = rr - (KC - 2) + k;
          CS_OUT[(rr * (KC - 1) + k) * CD + cb + i] =
              pos < 0 ? CS_IN[(pos + KC - 1) * CD + cb + i] : P[pos * PROJ + XOFF + cb + i];
          CS_OUT[(rr * (KC - 1) + k) * CD + cc + i] =
              pos < 0 ? CS_IN[(pos + KC - 1) * CD + cc + i] : P[pos * PROJ + XOFF + cc + i];
        }
      }
    }
  }
"""

_GROUP_NORM = r"""
  // one threadgroup of GS / 4 threads per (row, group): thread t owns 4 consecutive elements
  const uint t = thread_position_in_threadgroup.x;
  const uint grp = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int T = GS / 4;
  threadgroup float partial[T / 32];
  const int base = int(r) * XD + int(grp) * GS + int(t) * 4;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = fma(v[i], v[i], ss); }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(GS) + eps[0]);
  for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    OUT[base + i] = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));
  }
"""


_ROUTER = r"""
  // bf16 router logits for R rows: one threadgroup of SG simdgroups per expert. Simdgroup g sums its D / SG inputs
  // (lane l: 4 consecutive inputs at a time, 128 apart), then simd_sum; the simdgroups' sums are added in order.
  // A row's logits have the same bits at any row count (the row count is a runtime value).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(threadgroup_position_in_grid.y);
  const int R = rows[0];
  constexpr int PART = D / SG;
  threadgroup float part[MAXR][SG];
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  const int begin = int(g) * PART;
  for (int c = begin + 4 * int(lane); c < begin + PART; c += 128) {
    const float w0 = float(GW[size_t(e) * D + c]), w1 = float(GW[size_t(e) * D + c + 1]);
    const float w2 = float(GW[size_t(e) * D + c + 2]), w3 = float(GW[size_t(e) * D + c + 3]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + r * D + c;
      acc[r] = fma(float(xr[3]), w3, fma(float(xr[2]), w2, fma(float(xr[1]), w1, fma(float(xr[0]), w0, acc[r]))));
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) part[r][g] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && int(lane) < R) {
    float total = 0.0f;
    for (int k = 0; k < SG; k++) total += part[lane][k];
    OUT[int(lane) * NE + e] = bfloat(total);
  }
"""

_kernels: dict[str, Any] = {}
MAX_ROWS = 16          # rows of one mamba_step call (its B/C buffer lives in threadgroup memory)


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    arch = str(info.get("architecture", ""))
    digits = "".join(ch for ch in arch.removeprefix("applegpu_g") if ch.isdigit())
    return bool(digits) and int(digits) >= 17


def _named(base: str, source: str) -> str:
    return f"{base}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str]) -> Any:
    key = _named(name, source)
    kernel = _kernels.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(name=key, input_names=inputs, output_names=outputs, source=source)
        _kernels[key] = kernel
    return kernel


def add_norm(h: mx.array, delta: mx.array, weight: mx.array, eps: mx.array) -> tuple[mx.array, mx.array]:
    """(h + delta, RMSNorm(h + delta) * weight) for rows [R, D], both bf16."""

    rows, dims = h.shape[0], h.shape[-1]
    threads = 896 if dims % 896 == 0 else 256
    source = _ADD_NORM.replace("MIX", _MIX_PLAIN)
    kernel = _kernel("nemotron_add_norm", source, ["H", "X", "W", "eps"], ["HN", "OUT"])
    return kernel(inputs=[h, delta, weight, eps], template=[("D", dims), ("T", threads)],
                  grid=(threads * rows, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, dims), (rows, dims)], output_dtypes=[mx.bfloat16, mx.bfloat16])


def add_norm_moe(h: mx.array, routed: mx.array, weights: mx.array, shared: mx.array, weight: mx.array,
                 eps: mx.array) -> tuple[mx.array, mx.array]:
    """As ``add_norm`` with delta = bf16(bf16(sum_e w_e y_e) + shared): routed [R, E, D], weights [R, E]."""

    rows, experts, dims = routed.shape
    threads = 896 if dims % 896 == 0 else 256
    source = _ADD_NORM.replace("MIX", _MIX_MOE)
    kernel = _kernel("nemotron_add_norm_moe", source, ["H", "Y", "WE", "SH", "W", "eps"], ["HN", "OUT"])
    return kernel(inputs=[h, routed, weights, shared, weight, eps],
                  template=[("D", dims), ("T", threads), ("E", experts)],
                  grid=(threads * rows, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, dims), (rows, dims)], output_dtypes=[mx.bfloat16, mx.bfloat16])


def add_norm_experts(h: mx.array, routed: mx.array, weights: mx.array, weight: mx.array,
                     eps: mx.array) -> tuple[mx.array, mx.array]:
    """As ``add_norm`` with delta = bf16(sum_e w_e y_e) over routed [R, E, D] (shared expert among them)."""

    rows, experts, dims = routed.shape
    threads = 896 if dims % 896 == 0 else 256
    source = _ADD_NORM.replace("MIX", _MIX_EXPERTS)
    kernel = _kernel("nemotron_add_norm_experts", source, ["H", "Y", "WE", "W", "eps"], ["HN", "OUT"])
    return kernel(inputs=[h, routed, weights, weight, eps], template=[("D", dims), ("T", threads), ("E", experts)],
                  grid=(threads * rows, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, dims), (rows, dims)], output_dtypes=[mx.bfloat16, mx.bfloat16])


_ROUTER_ROWS: dict[int, mx.array] = {}


def router_logits(x: mx.array, gate_w: mx.array, *, simdgroups: int = 8) -> mx.array:
    """x [R, D] @ gate_w.T [D, E] -> [R, E] bf16, each row with the same bits at any R (R <= 16)."""

    rows, dims = x.shape
    experts = gate_w.shape[0]
    count = _ROUTER_ROWS.get(rows)
    if count is None:
        count = mx.array([rows], dtype=mx.int32)
        _ROUTER_ROWS[rows] = count
    if dims % (simdgroups * 4):
        raise ValueError("router_logits: D must split into simdgroups of 4-wide steps")
    kernel = _kernel("nemotron_router", _ROUTER, ["X", "GW", "rows"], ["OUT"])
    return kernel(inputs=[x, gate_w, count], template=[("D", dims), ("NE", experts), ("SG", simdgroups), ("MAXR", 16)],
                  grid=(32 * simdgroups, experts, 1), threadgroup=(32 * simdgroups, 1, 1),
                  output_shapes=[(rows, experts)], output_dtypes=[mx.bfloat16])[0]


def _stack_linears(linears: list[Any]) -> tuple[Any, list[int]]:
    """One quantized linear for projections that read the same input; returns it and the split points."""

    import mlx.nn as nn

    first = linears[0]
    stacked = nn.QuantizedLinear(first.weight.shape[1] * 32 // first.bits, 1, bias=False,
                                 group_size=first.group_size, bits=first.bits)
    stacked.weight = mx.concatenate([l.weight for l in linears], axis=0)
    stacked.scales = mx.concatenate([l.scales for l in linears], axis=0)
    stacked.biases = mx.concatenate([l.biases for l in linears], axis=0)
    mx.eval(stacked.parameters())
    cuts, total = [], 0
    for l in linears[:-1]:
        total += l.weight.shape[0]
        cuts.append(total)
    return stacked, cuts


def _fold_shared(mixer: Any) -> Any:
    """The MoE's expert tables with the shared expert as two more experts (its up rows and down columns halved).

    The shared expert is relu2(x @ up.T) @ down.T with 2x the routed width, so each half is shaped like a
    routed expert: rows [0, w) / [w, 2w) of up, and the matching column blocks of down (a whole number of
    quantization groups). Weight 1 in the router's output adds it to the routed sum.
    """

    import copy as _copy

    table = mixer.switch_mlp          # replaced in place: the old tables are freed (prompts route to ids < E)
    up, down = mixer.shared_experts.up_proj, mixer.shared_experts.down_proj
    width = mixer.switch_mlp.fc1.weight.shape[1]
    per_word, group = 32 // down.bits, down.group_size
    fc1, fc2 = _copy.copy(mixer.switch_mlp.fc1), _copy.copy(mixer.switch_mlp.fc2)
    for name in ("weight", "scales", "biases"):
        halves = [getattr(up, name)[:width], getattr(up, name)[width:]]
        setattr(fc1, name, mx.concatenate([getattr(mixer.switch_mlp.fc1, name), mx.stack(halves)], axis=0))
        cut = width // per_word if name == "weight" else width // group
        full = getattr(down, name)
        halves = [full[:, :cut], full[:, cut:]]
        setattr(fc2, name, mx.concatenate([getattr(mixer.switch_mlp.fc2, name), mx.stack(halves)], axis=0))
    table.fc1, table.fc2 = fc1, fc2
    mx.eval(fc1.parameters(), fc2.parameters())
    return table


def route(logits: mx.array, bias: mx.array, top_k: int, scaling: mx.array, *,
          shared_slots: int = 0) -> tuple[mx.array, mx.array]:
    """Expert ids [R, K] (best first; ties to the lower id) and weights [R, K] (fp32) from gate logits [R, E].

    ``shared_slots`` appends experts E, E + 1, ... with weight 1 (the shared expert folded into the table).
    """

    rows, experts = logits.shape
    width = top_k + shared_slots
    kernel = _kernel("nemotron_route", _ROUTE, ["G", "bias", "scaling"], ["IDX", "WT"])
    return kernel(inputs=[logits, bias, scaling], template=[("NE", experts), ("K", top_k), ("OK", width)],
                  grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, width), (rows, width)], output_dtypes=[mx.uint32, mx.float32])


def mamba_step(proj: mx.array, conv_state: mx.array, ssm_state: mx.array, conv_w: mx.array, conv_b: mx.array,
               a_log: mx.array, d_skip: mx.array, dt_bias: mx.array, limits: mx.array, *, heads: int,
               head_dim: int, groups: int, state_dim: int) -> tuple[mx.array, mx.array, mx.array]:
    """R consecutive tokens through one Mamba-2 mixer's conv and scan.

    Returns gated y [R, XD] and the conv and SSM states after every row: [R, KC-1, CD] and [R, H, DH, DS]
    (row r's states are the cache after the first r+1 tokens).
    """

    rows, width = proj.shape
    if rows > MAX_ROWS:
        raise ValueError(f"mamba_step takes at most {MAX_ROWS} rows")
    xd = heads * head_dim
    conv_dim = xd + 2 * groups * state_dim
    kc = conv_w.shape[0]
    dims = mx.array([rows], dtype=mx.int32)
    kernel = _kernel("nemotron_mamba_step", _MAMBA_STEP,
                     ["P", "CS_IN", "S_IN", "CW", "CB", "A_LOG", "DSKIP", "DT_BIAS", "limits", "dims"],
                     ["Y", "CS_OUT", "S_OUT"])
    ssz = heads * head_dim * state_dim
    return kernel(
        inputs=[proj, conv_state, ssm_state, conv_w, conv_b, a_log, d_skip, dt_bias, limits, dims],
        template=[("H", heads), ("DH", head_dim), ("NG", groups), ("DS", state_dim), ("XD", xd), ("KC", kc),
                  ("PROJ", width), ("XOFF", xd), ("DTOFF", xd + conv_dim), ("MAXR", MAX_ROWS), ("TGY", 8),
                  ("SSZ", ssz)],
        grid=(32, head_dim, heads), threadgroup=(32, 8, 1),
        output_shapes=[(rows, xd), (rows, kc - 1, conv_dim), (rows, heads, head_dim, state_dim)],
        output_dtypes=[mx.bfloat16, conv_state.dtype, ssm_state.dtype],
    )


def group_norm(x: mx.array, weight: mx.array, eps: mx.array, group: int) -> mx.array:
    rows, dims = x.shape
    kernel = _kernel("nemotron_group_norm", _GROUP_NORM, ["X", "W", "eps"], ["OUT"])
    return kernel(inputs=[x, weight, eps], template=[("XD", dims), ("GS", group)],
                  grid=((group // 4) * (dims // group), rows, 1), threadgroup=(group // 4, 1, 1),
                  output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


class FusedDecode:
    """Nemotron-H decode (one or more consecutive rows) through the kernels above and MLX's matmuls."""

    # layers per slice handed to the GPU while the rest of the step is built (0: the caller evaluates)
    eval_every = 8

    def __init__(self, model: Any, *, fold_shared: bool = True) -> None:
        args = model.args
        self.model = model
        self.backbone = model.backbone
        self.layers = model.backbone.layers
        # lane attention needs the M5's tensor units (its fragment layout is theirs; an M3 gets wrong values)
        self.lane_attention = tensor_units()
        self.lane_attention_from = 10_000
        self.eps_value = float(args.layer_norm_epsilon)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.limits = mx.array([float(args.time_step_limit[0]), float(args.time_step_limit[1])], dtype=mx.float32)
        self.scaling = mx.array([float(args.routed_scaling_factor or 1.0)], dtype=mx.float32)
        self.top_k = int(args.num_experts_per_tok)
        self.heads, self.head_dim = int(args.mamba_num_heads), int(args.mamba_head_dim)
        self.groups, self.state_dim = int(args.n_groups), int(args.ssm_state_size)
        self.mamba: dict[int, tuple[mx.array, ...]] = {}
        # the last call's Mamba states after each of its rows, by layer (for keeping a prefix of a window)
        self.row_states: dict[int, tuple[mx.array, mx.array]] = {}
        self._compiled_blocks: dict[int, Any] = {}
        import os

        self.compiled = os.environ.get("TF_NEMOTRON_COMPILE", "1") != "0"
        self.mamba_conv_dim = int(args.mamba_num_heads * args.mamba_head_dim + 2 * args.n_groups * args.ssm_state_size)
        for i, layer in enumerate(self.layers):
            if layer.block_type == "M":
                m = layer.mixer
                conv_w = m.conv1d.weight[:, :, 0].T.astype(mx.float32)          # [KC, CD]
                conv_b = (m.conv1d.bias if "bias" in m.conv1d else mx.zeros((m.conv_dim,))).astype(mx.float32)
                self.mamba[i] = (conv_w, conv_b, m.A_log.astype(mx.float32), m.D.astype(mx.float32),
                                 m.dt_bias.astype(mx.float32))
        mx.eval(list(self.mamba.values()))
        self.gate_bias = {i: layer.mixer.gate.e_score_correction_bias.astype(mx.float32)
                          for i, layer in enumerate(self.layers) if layer.block_type == "E"}
        mx.eval(list(self.gate_bias.values()))
        self.qkv: dict[int, tuple[Any, list[int]]] = {}
        self.experts: dict[int, Any] = {}
        for i, layer in enumerate(self.layers):
            if layer.block_type == "*":
                self.qkv[i] = _stack_linears([layer.mixer.q_proj, layer.mixer.k_proj, layer.mixer.v_proj])
            elif layer.block_type == "E" and fold_shared:
                self.experts[i] = _fold_shared(layer.mixer)
        self.shared_slots = 2 if fold_shared else 0

    def __call__(self, inputs: mx.array, cache: list[Any]) -> mx.array:
        """Hidden states after the final norm, [1, R, D], for R consecutive tokens (batch 1)."""

        tokens = inputs.reshape(-1)
        rows = tokens.shape[0]
        h = self.backbone.embeddings(tokens)                                     # [R, D]
        normed = mx.fast.rms_norm(h, self.layers[0].norm.weight, self.eps_value)
        cache_at = 0
        for i, layer in enumerate(self.layers):
            kind = layer.block_type
            nxt = self.layers[i + 1].norm.weight if i + 1 < len(self.layers) else self.backbone.norm_f.weight
            if kind == "M":
                c = cache[cache_at]
                cache_at += 1
                conv_state, ssm_state = self._mamba_states(c, normed.dtype)
                block = self._block(i, "M", nxt) if self.compiled else self._mamba_block(i, nxt)
                h, normed, conv_rows, ssm_rows = block(normed, h, conv_state, ssm_state)
                c[0], c[1] = conv_rows[rows - 1:rows], ssm_rows[rows - 1:rows]
                self.row_states[i] = (conv_rows, ssm_rows)
                c.advance(rows)
            elif kind == "*":
                c = cache[cache_at]
                cache_at += 1
                delta = self._attention(layer.mixer, normed, c, i)
                h, normed = add_norm(h, delta, nxt, self.eps)
            else:
                block = self._block(i, "E", nxt) if self.compiled else self._moe_block(i, nxt)
                h, normed = block(normed, h)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return normed.reshape(1, rows, -1)

    def _mamba_states(self, cache: Any, dtype: Any) -> tuple[mx.array, mx.array]:
        conv_state, ssm_state = cache[0], cache[1]
        if conv_state is None:
            conv_state = mx.zeros((1, 3, self.mamba_conv_dim), dtype=dtype)
        if ssm_state is None:
            ssm_state = mx.zeros((1, self.heads, self.head_dim, self.state_dim), dtype=mx.float32)
        return conv_state, ssm_state

    def _block(self, index: int, kind: str, nxt: mx.array) -> Any:
        """The layer's work between its input norm and the next layer's, compiled (a traced graph per row
        count replaces ~8-12 Python-built ops: 30 -> 4 us of host time a layer, same bits)."""

        fn = self._compiled_blocks.get(index)
        if fn is None:
            fn = mx.compile(self._mamba_block(index, nxt) if kind == "M" else self._moe_block(index, nxt))
            self._compiled_blocks[index] = fn
        return fn

    def _mamba_block(self, index: int, nxt: mx.array) -> Any:
        mixer = self.layers[index].mixer
        conv_w, conv_b, a_log, d_skip, dt_bias = self.mamba[index]

        def block(x: mx.array, h: mx.array, conv_state: mx.array, ssm_state: mx.array) -> tuple[mx.array, ...]:
            proj = mixer.in_proj(x)
            y, conv_rows, ssm_rows = mamba_step(proj, conv_state, ssm_state, conv_w, conv_b, a_log, d_skip,
                                                dt_bias, self.limits, heads=self.heads, head_dim=self.head_dim,
                                                groups=self.groups, state_dim=self.state_dim)
            y = group_norm(y, mixer.norm.weight, self.eps, mixer.norm.group_size)
            hn, xn = add_norm(h, mixer.out_proj(y), nxt, self.eps)
            return hn, xn, conv_rows, ssm_rows

        return block

    def _moe_block(self, index: int, nxt: mx.array) -> Any:
        mixer = self.layers[index].mixer

        def block(x: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
            routed, weights, shared = self._moe(index, mixer, x)
            if shared is None:
                return add_norm_experts(h, routed, weights, nxt, self.eps)
            return add_norm_moe(h, routed, weights, shared, nxt, self.eps)

        return block

    def _mamba(self, index: int, mixer: Any, x: mx.array, cache: Any) -> mx.array:
        conv_w, conv_b, a_log, d_skip, dt_bias = self.mamba[index]
        proj = mixer.in_proj(x)                                                  # [R, 10304]
        rows = x.shape[0]
        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros((1, conv_w.shape[0] - 1, conv_w.shape[1]), dtype=x.dtype)
        ssm_state = cache[1]
        if ssm_state is None:
            ssm_state = mx.zeros((1, self.heads, self.head_dim, self.state_dim), dtype=mx.float32)
        y, conv_rows, ssm_rows = mamba_step(proj, conv_state, ssm_state, conv_w, conv_b, a_log, d_skip, dt_bias,
                                            self.limits, heads=self.heads, head_dim=self.head_dim,
                                            groups=self.groups, state_dim=self.state_dim)
        cache[0], cache[1] = conv_rows[rows - 1:rows], ssm_rows[rows - 1:rows]
        self.row_states[index] = (conv_rows, ssm_rows)
        cache.advance(rows)
        y = group_norm(y, mixer.norm.weight, self.eps, mixer.norm.group_size)
        return mixer.out_proj(y)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call on ``rows`` rows, make ``cache`` hold only its first ``keep`` rows."""

        if keep == rows:
            return
        drop = rows - keep
        cache_at = 0
        for i, layer in enumerate(self.layers):
            if layer.block_type not in "M*":
                continue
            c = cache[cache_at]
            cache_at += 1
            if layer.block_type == "M":
                conv_rows, ssm_rows = self.row_states[i]
                c[0], c[1] = conv_rows[keep - 1:keep], ssm_rows[keep - 1:keep]
            else:
                c.trim(drop)

    def _attention(self, mixer: Any, x: mx.array, cache: Any, index: int | None = None) -> mx.array:
        rows = x.shape[0]
        if index in self.qkv:
            stacked, cuts = self.qkv[index]
            q, k, v = mx.split(stacked(x), cuts, axis=-1)
        else:
            q, k, v = mixer.q_proj(x), mixer.k_proj(x), mixer.v_proj(x)
        q = q.reshape(1, rows, mixer.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(1, rows, mixer.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(1, rows, mixer.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        keys, values = cache.update_and_fetch(k, v)
        # the kernel is chosen by each row's own key count, so a row gets the bits serial decoding gives it:
        # a window straddling the switch attends row by row
        first = cache.offset - rows + 1
        lane_rows = [self.lane_attention and first + r >= self.lane_attention_from for r in range(rows)]
        if rows > 1 and not all(lane_rows):
            # row by row (each with its own keys): MLX's attention picks its kernel by query count, and a row
            # must get serial decoding's bits; the lane kernel's rows are independent by construction
            outs = [self._attend(q[:, :, r:r + 1], keys[:, :, :first + r], values[:, :, :first + r], mixer.scale,
                                 lane_rows[r], 1) for r in range(rows)]
            out = mx.concatenate(outs, axis=2)
        else:
            out = self._attend(q, keys, values, mixer.scale, lane_rows[0], rows)
        return mixer.o_proj(out.transpose(0, 2, 1, 3).reshape(rows, -1))

    @staticmethod
    def _attend(q: mx.array, keys: mx.array, values: mx.array, scale: float, lane: bool, rows: int) -> mx.array:
        if lane:
            # the 16 query heads of a KV head are one 16-row tile of the tensor-unit kernel: each key read once.
            # M5 Max, 6 dependent calls: 60k keys 0.323 -> 0.194 ms a call, 32k 0.191 -> 0.120, 12k 0.154 ->
            # 0.123, but 4k 0.194 -> 0.280 and 1k 0.091 -> 0.250 (its extra launches): from 10k keys on
            from tensorfold.kernels.qwen.dense.v1.lane_attention import lane_sdpa

            return lane_sdpa(q, keys, values, scale)
        # (a block-per-threadgroup kernel sharing each key block across the 16 heads of a KV head was correct
        # but slower than MLX's per-head kernel at 60k keys, 0.73 vs 0.51 ms, 2026-09-25)
        return mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask="causal" if rows > 1 else None)

    def _moe(self, index: int, mixer: Any, x: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
        # (one kernel for norm + router + selection was tried 2026-09-25: a threadgroup reading the 0.69 MB
        # router alone made decode 220 -> 141 tok/s; the router stays MLX's matmul, then ``route``)
        # our own router matvec, row-invariant by construction: MLX's bf16 matmul sums 2 rows in another order
        # than 1 (a logit 1 bf16 unit off at layer 29 of a real decode, 2026-09-25), and per-row gemvs cost a
        # kernel a row
        logits = router_logits(x, mixer.gate.weight)
        experts, weights = route(logits, self.gate_bias[index], self.top_k, self.scaling,
                                 shared_slots=self.shared_slots)
        table = self.experts.get(index)
        if table is None:
            return mixer.switch_mlp(x, experts), weights, mixer.shared_experts(x)
        return table(x, experts), weights, None                                 # [R, K + 2, D], shared inside
