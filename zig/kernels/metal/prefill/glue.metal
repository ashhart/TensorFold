// Nemotron prefill's element-wise steps, copies and compiled fusions, value for value MLX 0.32.3's (one op a statement).
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

namespace tfp {

// MLX's Sigmoid as its fusions compile it for bf16: bf16 exp(|x|) (precise), then 1 / (1 + e) in bf16.
inline bfloat sigmoid_bf16(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(float(bfloat(metal::fabs(float(x))))));
  auto y = 1 / (1 + e);
  return (x < 0) ? y : 1 - y;
}

// MLX's log1p and LogAddExp as its fusions compile them (fast exp and log at run time).
inline float log1p_mlx(float x) {
  const float xp1 = 1.0f + x;
  if (xp1 == INFINITY) {
    return INFINITY;
  }
  if (xp1 == 1.0f) {
    return x;
  }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline float logaddexp_mlx(float x, float y) {
  if (isnan(x) || isnan(y)) {
    return NAN;
  }
  const float hi = metal::max(x, y), lo = metal::min(x, y);
  return (lo == -INFINITY || hi == INFINITY) ? hi : (hi + log1p_mlx(metal::exp(lo - hi)));
}

// A = -float(bf16 exp(A_log)): v_Exp (bf16, precise), v_copy to fp32, v_Negative. P: heads.
inline void mamba_a(const device bfloat* a_log, device float* a, const device int* P, uint h) {
  if (int(h) < P[0]) {
    const bfloat e = bfloat(metal::precise::exp(float(a_log[h])));
    a[h] = -float(e);
  }
}

// [conv state rows | this chunk's conv channels]: the zero or kept state, then proj[:, x0:x0 + C]. P: L C proj x0 state.
inline void conv_pack(const device bfloat* proj, const device bfloat* state, device bfloat* padded, const device int* P,
                      uint2 at) {
  const int c = int(at.x), r = int(at.y), C = P[1], taps = 3;
  if (c >= C || r >= P[0] + taps) {
    return;
  }
  padded[long(r) * C + c] = r < taps ? (P[4] ? state[r * C + c] : bfloat(0)) : proj[long(r - taps) * P[2] + P[3] + c];
}

// The next chunk's conv state: the last 3 rows of this chunk's padded input. P: L C.
inline void conv_keep(const device bfloat* padded, device bfloat* state, const device int* P, uint2 at) {
  if (int(at.x) < P[1]) {
    state[at.y * P[1] + at.x] = padded[long(P[0] + int(at.y)) * P[1] + at.x];
  }
}

// silu(conv + bias): g2_Add in bf16, then the compiled silu fusion (x * sigmoid(x) in bf16). P: count C.
inline void conv_act(const device bfloat* conv, const device bfloat* bias, device bfloat* act, const device int* P,
                     uint i) {
  if (int(i) < P[0]) {
    const bfloat a = conv[i] + bias[int(i) % P[1]];
    const bfloat s = sigmoid_bf16(a);
    act[i] = a * s;
  }
}

// dt = clip(softplus(dt + bias)) (the compiled fusion), dtA = dt A, dtx = dt x (fp32). P: L proj dt0 C heads dh.
inline void mamba_dt(const device bfloat* proj, const device bfloat* bias, const device float* a,
                     const device bfloat* act, device float* dt, device float* dta, device float* dtx,
                     const device int* P, uint2 at) {
  const int H = P[4], DH = P[5], hp = int(at.x), l = int(at.y), h = hp / DH;
  if (hp >= H * DH || l >= P[0]) {
    return;
  }
  const float f = float(proj[long(l) * P[1] + P[2] + h]);
  const float g = float(bias[h]);
  const float x = f + g;
  const float s = logaddexp_mlx(x, 0.0f);
  const float m = isnan(s) ? s : (s > 0.0f ? s : 0.0f);
  const float d = isnan(m) ? m : (m < INFINITY ? m : INFINITY);
  const float xv = float(act[long(l) * P[3] + hp]);
  dtx[long(l) * H * DH + hp] = d * xv;
  if (hp % DH == 0) {
    dt[l * H + h] = d;
    dta[l * H + h] = d * a[h];
  }
}

// The segsum's input: dtA^T repeated over j and kept strictly below the diagonal (tril -1), else 0. P: i0 s heads.
inline void segsum_in(const device float* dta, device float* x, const device int* P, uint3 at) {
  const int j = int(at.x), i = int(at.y), h = int(at.z), s = P[1], H = P[2];
  if (j < s && i < s) {
    x[(long(h) * s + i) * s + j] = i > j ? dta[long(P[0] + i) * H + h] : 0.0f;
  }
}

// decay = exp(cumsum) (precise); surrogate = tril(float(CB) decay, 0); the decay's last row. P: s heads groups.
inline void ssd_decay(const device float* seg, const device bfloat* cb, device float* sur, device float* last,
                      const device int* P, uint3 at) {
  const int j = int(at.x), i = int(at.y), h = int(at.z), s = P[0], g = h / (P[1] / P[2]);
  if (j >= s || i >= s) {
    return;
  }
  const long at_hij = (long(h) * s + i) * s + j;
  const float d = metal::precise::exp(seg[at_hij]);
  const float p = float(cb[(long(g) * s + i) * s + j]) * d;
  sur[at_hij] = i >= j ? p : 0.0f;
  if (i == s - 1) {
    last[h * s + j] = d;
  }
}

// B for the state matmul: float(B) repeated over each group's heads, as [heads, state, s]. P: i0 s C b0 heads groups dstate.
inline void ssd_b_heads(const device bfloat* act, device float* b, const device int* P, uint3 at) {
  const int t = int(at.x), n = int(at.y), h = int(at.z), s = P[1], N = P[6], g = h / (P[4] / P[5]);
  if (t < s && n < N) {
    b[(long(h) * N + n) * s + t] = float(act[long(P[0] + t) * P[2] + P[3] + g * N + n]);
  }
}

// (dt x) times the decay's last row, per head: g3_Multiply in fp32, as [s, heads, dh]. P: i0 s heads dh.
inline void ssd_dtx_decay(const device float* dtx, const device float* last, device float* out, const device int* P,
                          uint2 at) {
  const int hp = int(at.x), t = int(at.y), s = P[1], HD = P[2] * P[3], h = hp / P[3];
  if (hp < HD && t < s) {
    out[long(t) * HD + hp] = dtx[long(P[0] + t) * HD + hp] * last[h * s + t];
  }
}

// C as fp32 [s, groups, state] for the gemv (g2_copy). P: i0 s C c0 width.
inline void ssd_c_f32(const device bfloat* act, device float* c, const device int* P, uint2 at) {
  const int n = int(at.x), t = int(at.y);
  if (n < P[4] && t < P[1]) {
    c[long(t) * P[4] + n] = float(act[long(P[0] + t) * P[2] + P[3] + n]);
  }
}

// exp (precise) of the dtA cumsum, fp32. P: count.
inline void exp_f32(const device float* x, device float* y, const device int* P, uint i) {
  if (int(i) < P[0]) {
    y[i] = metal::precise::exp(x[i]);
  }
}

// The carried state: next += e_last * state (g2_Multiply, then vv_Add, fp32). P: s heads per_head.
inline void ssd_state_carry(const device float* next, const device float* e, const device float* state,
                            device float* out, const device int* P, uint i) {
  const int per = P[2];
  if (int(i) < P[1] * per) {
    const float m = e[long(P[0] - 1) * P[1] + int(i) / per] * state[i];
    out[i] = next[i] + m;
  }
}

// A step's y rows (bf16) in the chunk, + e y_prev if a state carries; keep copies other rows from old. P: i0 s H dh carry L keep.
inline void ssd_y_out(const device float* y, const device float* e, const device float* prev, const device bfloat* old,
                      device bfloat* out, const device int* P, uint2 at) {
  const int hp = int(at.x), r = int(at.y), s = P[1], H = P[2], DH = P[3], h = hp / DH, p = hp % DH;
  const int t = P[6] ? r - P[0] : r, row = P[0] + t;
  if (hp >= H * DH || (P[6] ? r >= P[5] : r >= s)) {
    return;
  }
  if (t < 0 || t >= s) {
    out[long(r) * H * DH + hp] = old[long(r) * H * DH + hp];
    return;
  }
  float v = y[(long(h) * s + t) * DH + p];
  if (P[4]) {
    const float m = e[long(t) * H + h] * prev[long(t) * H * DH + hp];
    v = v + m;
  }
  out[long(row) * H * DH + hp] = bfloat(v);
}

// y + x D (g3_Multiply then g3_Add, bf16), x the conv output's first heads x dh channels. P: L C heads dh.
inline void mamba_skip(const device bfloat* y, const device bfloat* act, const device bfloat* d, device bfloat* out,
                       const device int* P, uint2 at) {
  const int hp = int(at.x), l = int(at.y), HD = P[2] * P[3];
  if (hp < HD && l < P[0]) {
    const bfloat xd = act[long(l) * P[1] + hp] * d[hp / P[3]];
    out[long(l) * HD + hp] = y[long(l) * HD + hp] + xd;
  }
}

// The compiled swiglu fusion: silu(gate) * y in bf16, gate = proj[:, :width]. P: L width proj.
inline void mamba_gate(const device bfloat* proj, const device bfloat* y, device bfloat* out, const device int* P,
                       uint2 at) {
  const int c = int(at.x), l = int(at.y);
  if (c < P[1] && l < P[0]) {
    const bfloat g = proj[long(l) * P[2] + c];
    const bfloat s = sigmoid_bf16(g);
    const bfloat gs = g * s;
    out[long(l) * P[1] + c] = gs * y[long(l) * P[1] + c];
  }
}

// weight * x per column (g2_Multiply, bf16). P: count width.
inline void scale_cols(const device bfloat* x, const device bfloat* w, device bfloat* y, const device int* P, uint i) {
  if (int(i) < P[0]) {
    y[i] = w[int(i) % P[1]] * x[i];
  }
}

// New K or V rows into the cache at row at; keep copies rows below at from old. P: L kv_heads dim cap at old_cap keep.
inline void kv_put(const device bfloat* src, const device bfloat* old, device bfloat* dst, const device int* P,
                   uint3 at) {
  const int d = int(at.x), r = int(at.y), kv = int(at.z), D = P[2], start = P[6] ? 0 : P[4], row = start + r;
  if (d >= D || row >= P[4] + P[0]) {
    return;
  }
  const long to = (long(kv) * P[3] + row) * D + d;
  dst[to] = row < P[4] ? old[(long(kv) * P[5] + row) * D + d] : src[(long(row - P[4]) * P[1] + kv) * D + d];
}

// dst[p] = src[idx[p] / div]: the experts' sorted rows (div k) and their unsort (div 1). P: count width div.
inline void rows_take(const device bfloat* src, const device uint32_t* idx, device bfloat* dst, const device int* P,
                      uint2 at) {
  const int c = int(at.x), p = int(at.y);
  if (c < P[1] && p < P[0]) {
    dst[long(p) * P[1] + c] = src[long(idx[p] / uint(P[2])) * P[1] + c];
  }
}

// out[p] = idx[p] / div (div 0: out[p] = p), the gather_qmv's x row per sorted row. P: count div.
inline void rows_of(const device uint32_t* idx, device uint32_t* out, const device int* P, uint p) {
  if (int(p) < P[0]) {
    out[p] = P[1] ? idx[p] / uint(P[1]) : p;
  }
}

}  // namespace tfp
// ---- entry points: generated by tools/zig/check_prefill_ops.py --write ----
[[kernel]] void custom_kernel_tf_conv_act_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* CONV [[buffer(0)]],
  const device bfloat16_t* BIAS [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* ACT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::conv_act(CONV, BIAS, ACT, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_conv_keep_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* PAD [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device bfloat16_t* STATE [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::conv_keep(PAD, STATE, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_conv_pack_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* PROJ [[buffer(0)]],
  const device bfloat16_t* STATE [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* PAD [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::conv_pack(PROJ, STATE, PAD, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_exp_f32_float_int32_t_float(
  const device float* X [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device float* Y [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::exp_f32(X, Y, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* SRC [[buffer(0)]],
  const device bfloat16_t* OLD [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* DST [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::kv_put(SRC, OLD, DST, P, thread_position_in_grid);

}
[[kernel]] void custom_kernel_tf_mamba_a_bfloat16_t_int32_t_float(
  const device bfloat16_t* A_LOG [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device float* A [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::mamba_a(A_LOG, A, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_mamba_dt_bfloat16_t_bfloat16_t_float_bfloat16_t_int32_t_float_float_float(
  const device bfloat16_t* PROJ [[buffer(0)]],
  const device bfloat16_t* BIAS [[buffer(1)]],
  const device float* A [[buffer(2)]],
  const device bfloat16_t* ACT [[buffer(3)]],
  const device int32_t* P [[buffer(4)]],
  device float* DT [[buffer(5)]],
  device float* DTA [[buffer(6)]],
  device float* DTX [[buffer(7)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::mamba_dt(PROJ, BIAS, A, ACT, DT, DTA, DTX, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_mamba_gate_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* PROJ [[buffer(0)]],
  const device bfloat16_t* Y [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::mamba_gate(PROJ, Y, OUT, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_mamba_skip_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* Y [[buffer(0)]],
  const device bfloat16_t* ACT [[buffer(1)]],
  const device bfloat16_t* D [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::mamba_skip(Y, ACT, D, OUT, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_rows_of_uint32_t_int32_t_uint32_t(
  const device uint32_t* IDX [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device uint32_t* OUT [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::rows_of(IDX, OUT, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_rows_take_bfloat16_t_uint32_t_int32_t_bfloat16_t(
  const device bfloat16_t* SRC [[buffer(0)]],
  const device uint32_t* IDX [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* DST [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::rows_take(SRC, IDX, DST, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_scale_cols_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* X [[buffer(0)]],
  const device bfloat16_t* W [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* Y [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::scale_cols(X, W, Y, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_segsum_in_float_int32_t_float(
  const device float* DTA [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device float* X [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::segsum_in(DTA, X, P, thread_position_in_grid);

}
[[kernel]] void custom_kernel_tf_ssd_b_heads_bfloat16_t_int32_t_float(
  const device bfloat16_t* ACT [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device float* B [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_b_heads(ACT, B, P, thread_position_in_grid);

}
[[kernel]] void custom_kernel_tf_ssd_c_f32_bfloat16_t_int32_t_float(
  const device bfloat16_t* ACT [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device float* C [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_c_f32(ACT, C, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_ssd_decay_float_bfloat16_t_int32_t_float_float(
  const device float* SEG [[buffer(0)]],
  const device bfloat16_t* CB [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* SUR [[buffer(3)]],
  device float* LAST [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_decay(SEG, CB, SUR, LAST, P, thread_position_in_grid);

}
[[kernel]] void custom_kernel_tf_ssd_dtx_decay_float_float_int32_t_float(
  const device float* DTX [[buffer(0)]],
  const device float* LAST [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* DD [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_dtx_decay(DTX, LAST, DD, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
[[kernel]] void custom_kernel_tf_ssd_state_carry_float_float_float_int32_t_float(
  const device float* NEXT [[buffer(0)]],
  const device float* E [[buffer(1)]],
  const device float* STATE [[buffer(2)]],
  const device int32_t* P [[buffer(3)]],
  device float* OUT [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_state_carry(NEXT, E, STATE, OUT, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_ssd_y_out_float_float_float_bfloat16_t_int32_t_bfloat16_t(
  const device float* Y [[buffer(0)]],
  const device float* E [[buffer(1)]],
  const device float* PREV [[buffer(2)]],
  const device bfloat16_t* OLD [[buffer(3)]],
  const device int32_t* P [[buffer(4)]],
  device bfloat16_t* OUT [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::ssd_y_out(Y, E, PREV, OLD, OUT, P, uint2(thread_position_in_grid.x, thread_position_in_grid.y));

}
