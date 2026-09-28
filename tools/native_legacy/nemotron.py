"""Retired Nemotron shared-slot routing and expert norm. See README.md."""
import mlx.core as mx
MAX_ROWS = 16

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
from tensorfold.kernels.nemotron.lightning.v1.kernels import _kernel, _ADD_NORM, add_norm, add_norm_moe

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

_MIX_EXPERTS = r"""{
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(routed));
    }"""

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
