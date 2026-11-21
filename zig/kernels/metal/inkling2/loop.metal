// Inkling 2 Mini's looped core (inkling2_loop): the messages between passes and the gates that mix them in.
// Every kernel runs a row (or a row and a gate) as one 256-thread group with fixed-order fp32 trees, so a row's bits
// never depend on the rows around it. Formulas follow transformers' Inkling2LoopMessageAttentionResidual and
// Inkling2LoopMessageGate; helpers come from common.metal, which precedes this file in the library.

constant constexpr uint LP_MAX_SOURCES = 9;   // sources of one attention-residual mix (finished blocks + the partial state)
constant constexpr uint LP_MAX_QUERIES = 4;   // queries of one mix: q, k, v, mlp of a message; 1 or 3 of an input
constant constexpr uint LP_MAX_RANK = 128;    // gate rank
constant constexpr uint LP_MAX_GATES = 3;     // gates one dispatch mixes (the attention q, k and v inputs)

struct LpMixArgs {
  uint hidden, blocks, block_stride, partial_f32;
  uint queries, out_stride, partial_pitch, pad;
  float eps_unweighted, scale;
};

inline float lp_source(const device bfloat* blocks, const device uchar* partial, constant LpMixArgs& a, uint s,
                       uint row, uint i) {
  if (s < a.blocks) return float(blocks[ulong(s) * a.block_stride + ulong(row) * a.hidden + i]);
  if (a.partial_f32 != 0) return ((const device float*)partial)[ulong(row) * a.partial_pitch + i];
  return float(((const device bfloat*)partial)[ulong(row) * a.partial_pitch + i]);
}

// Block attention residual without the norm that follows it: per row the sources (finished blocks, bf16, then the
// partial state, fp32 or bf16) are RMS-normed without a weight, up to four fp32 queries attend over them (softmax in
// fp32, scale hidden^-0.5, keys and values the same normed rows) and the fp32 mixes are written, one plane a query.
// Pass-to-pass messages are this kernel with the previous pass's banked blocks and final hidden state as the sources.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_mix(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* q0 [[buffer(2)]], const device float* q1 [[buffer(3)]], const device float* q2 [[buffer(4)]],
    const device float* q3 [[buffer(5)]], device float* out [[buffer(6)]], constant LpMixArgs& a [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_SGS][LP_MAX_QUERIES * LP_MAX_SOURCES];
  threadgroup float inv_tg[LP_MAX_SOURCES];
  const uint S = a.blocks + 1;
  const uint H = a.hidden;

  // 1. each source's 1 / rms.
  float ss[LP_MAX_SOURCES];
  for (uint s = 0; s < LP_MAX_SOURCES; ++s) {
    ss[s] = 0.0f;
    if (s >= S) continue;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += I2_TG) {
      const float v = lp_source(blocks, partial, a, s, row, i);
      acc = fma(v, v, acc);
    }
    ss[s] = simd_sum(acc);
  }
  if (lane == 0)
    for (uint s = 0; s < S; ++s) part[sg][s] = ss[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < S) {
    float t = part[0][tid];
    for (uint g = 1; g < I2_SGS; ++g) t += part[g][tid];
    inv_tg[tid] = i2_rsqrt_mean(t, H, a.eps_unweighted);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. query . source for every query and source.
  float dots[LP_MAX_QUERIES][LP_MAX_SOURCES];
  for (uint q = 0; q < LP_MAX_QUERIES; ++q) {
    const device float* qv = q == 0 ? q0 : (q == 1 ? q1 : (q == 2 ? q2 : q3));
    for (uint s = 0; s < LP_MAX_SOURCES; ++s) {
      dots[q][s] = 0.0f;
      if (q >= a.queries || s >= S) continue;
      float acc = 0.0f;
      for (uint i = tid; i < H; i += I2_TG) acc = fma(qv[i], lp_source(blocks, partial, a, s, row, i), acc);
      dots[q][s] = simd_sum(acc);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0)
    for (uint q = 0; q < a.queries; ++q)
      for (uint s = 0; s < S; ++s) part[sg][q * LP_MAX_SOURCES + s] = dots[q][s];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. softmax over the sources (identical in every thread), folded with 1 / rms into one coefficient a source.
  float coef[LP_MAX_QUERIES][LP_MAX_SOURCES];
  for (uint q = 0; q < LP_MAX_QUERIES; ++q) {
    for (uint s = 0; s < LP_MAX_SOURCES; ++s) coef[q][s] = 0.0f;
    if (q >= a.queries) continue;
    float score[LP_MAX_SOURCES];
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) {
      float t = part[0][q * LP_MAX_SOURCES + s];
      for (uint g = 1; g < I2_SGS; ++g) t += part[g][q * LP_MAX_SOURCES + s];
      score[s] = (t * inv_tg[s]) * a.scale;
      m = i2_max(m, score[s]);
    }
    float l = 0.0f;
    for (uint s = 0; s < S; ++s) {
      score[s] = score[s] == -INFINITY ? 0.0f : metal::precise::exp(score[s] - m);
      l += score[s];
    }
    for (uint s = 0; s < S; ++s) coef[q][s] = (score[s] / l) * inv_tg[s];
  }

  // 4. the fp32 mixes.
  for (uint q = 0; q < LP_MAX_QUERIES; ++q) {
    if (q >= a.queries) break;
    device float* dst = out + ulong(q) * a.out_stride + ulong(row) * H;
    for (uint i = tid; i < H; i += I2_TG) {
      float u = 0.0f;
      for (uint s = 0; s < S; ++s) u = fma(coef[q][s], lp_source(blocks, partial, a, s, row, i), u);
      dst[i] = u;
    }
  }
}

struct LpGateArgs {
  uint hidden, rank, rows, msg_stride;
  uint slot0, slot1, slot2, out_stride;
  float eps_unweighted, eps_norm;
};

inline uint lp_slot(constant LpGateArgs& a, uint g) { return g == 0 ? a.slot0 : (g == 1 ? a.slot1 : a.slot2); }

// First half of Inkling2LoopMessageGate: for gate g (grid y) and a row, both the fp32 state h and the fp32 message m
// are RMS-normed without a weight, concatenated, rounded to bf16 and projected to `rank` by the bf16 down matrix
// [rank][2 hidden] (fp32 chain, output rounded to bf16). down[g][row][rank] holds the rounded values.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_down(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], const device bfloat* w0 [[buffer(4)]], const device bfloat* w1 [[buffer(5)]],
    const device bfloat* w2 [[buffer(6)]], device bfloat* down [[buffer(7)]], constant LpGateArgs& a [[buffer(8)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  threadgroup float part[I2_SGS][LP_MAX_RANK];
  const uint row = grid.x, g = grid.y;
  const uint H = a.hidden;
  const device float* h = (g == 0 ? h0 : (g == 1 ? h1 : h2)) + ulong(row) * H;
  const device float* m = msg + ulong(lp_slot(a, g)) * a.msg_stride + ulong(row) * H;
  const device bfloat* w = g == 0 ? w0 : (g == 1 ? w1 : w2);

  float xh[I2_MAX_ELEMS], xm[I2_MAX_ELEMS];
  float sh = 0.0f, sm = 0.0f;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    xh[e] = i < H ? h[i] : 0.0f;
    xm[e] = i < H ? m[i] : 0.0f;
    sh = fma(xh[e], xh[e], sh);
    sm = fma(xm[e], xm[e], sm);
  }
  const float inv_h = i2_rsqrt_mean(i2_tg_sum(sh, scratch, sg, lane), H, a.eps_unweighted);
  const float inv_m = i2_rsqrt_mean(i2_tg_sum(sm, scratch, sg, lane), H, a.eps_unweighted);
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    xh[e] = float(bfloat(xh[e] * inv_h));
    xm[e] = float(bfloat(xm[e] * inv_m));
  }
  for (uint r = 0; r < a.rank; ++r) {
    const device bfloat* wr = w + ulong(r) * 2 * H;
    float acc = 0.0f;
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      if (i < H) acc = fma(float(wr[i]), xh[e], acc);
    }
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      if (i < H) acc = fma(float(wr[H + i]), xm[e], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) part[sg][r] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < a.rank) {
    float t = part[0][tid];
    for (uint s = 1; s < I2_SGS; ++s) t += part[s][tid];
    down[(ulong(g) * a.rows + row) * a.rank + tid] = bfloat(t);
  }
}

// Second half, fused with the weighted RMS norm that consumes the gate: forget = tanh(Wf d), gain = tanh(Wg d) over the
// bf16-stored fp32 gate matrices [hidden][rank] (fp32 chains), y = (1 - forget) * h + gain * m in fp32, then
// bf16(norm_w * (y * rsqrt(mean(y^2) + eps))). One plane a gate, `out_stride` elements apart.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_mix(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], const device bfloat* down [[buffer(4)]],
    const device bfloat* f0 [[buffer(5)]], const device bfloat* f1 [[buffer(6)]], const device bfloat* f2 [[buffer(7)]],
    const device bfloat* g0 [[buffer(8)]], const device bfloat* g1 [[buffer(9)]], const device bfloat* g2 [[buffer(10)]],
    const device bfloat* n0 [[buffer(11)]], const device bfloat* n1 [[buffer(12)]],
    const device bfloat* n2 [[buffer(13)]], device bfloat* out [[buffer(14)]], constant LpGateArgs& a [[buffer(15)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  threadgroup float dn[LP_MAX_RANK];
  const uint row = grid.x, g = grid.y;
  const uint H = a.hidden, R = a.rank;
  const device float* h = (g == 0 ? h0 : (g == 1 ? h1 : h2)) + ulong(row) * H;
  const device float* m = msg + ulong(lp_slot(a, g)) * a.msg_stride + ulong(row) * H;
  const device bfloat* wf = g == 0 ? f0 : (g == 1 ? f1 : f2);
  const device bfloat* wg = g == 0 ? g0 : (g == 1 ? g1 : g2);
  const device bfloat* nw = g == 0 ? n0 : (g == 1 ? n1 : n2);
  if (tid < R) dn[tid] = float(down[(ulong(g) * a.rows + row) * R + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float y[I2_MAX_ELEMS];
  float ss = 0.0f;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    float v = 0.0f;
    if (i < H) {
      float f = 0.0f, gn = 0.0f;
      for (uint r = 0; r < R; ++r) {
        f = fma(float(wf[ulong(i) * R + r]), dn[r], f);
        gn = fma(float(wg[ulong(i) * R + r]), dn[r], gn);
      }
      v = (1.0f - metal::precise::tanh(f)) * h[i] + metal::precise::tanh(gn) * m[i];
      ss = fma(v, v, ss);
    }
    y[e] = v;
  }
  const float inv = i2_rsqrt_mean(i2_tg_sum(ss, scratch, sg, lane), H, a.eps_norm);
  device bfloat* dst = out + ulong(g) * a.out_stride + ulong(row) * H;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H) dst[i] = bfloat(float(nw[i]) * (y[e] * inv));
  }
}
