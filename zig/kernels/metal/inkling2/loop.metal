// Inkling2Mini looped core (inkling2_loop): pass messages and their mixing gates; helpers from common.metal.

constant constexpr uint LP_MAX_SOURCES = 9;   // sources of one attention-residual mix (finished blocks + the partial state)
constant constexpr uint LP_MAX_QUERIES = 4;   // queries of one mix: q, k, v, mlp of a message; 1 or 3 of an input
constant constexpr uint LP_MAX_RANK = 128;    // gate rank
constant constexpr uint LP_MAX_GATES = 3;     // gates one dispatch mixes (the attention q, k and v inputs)
constant constexpr uint LP_RANK_BLOCK = 8;    // gate-down ranks one threadgroup computes

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

// Attention residual, no norm: normed sources, four fp32 queries attend (softmax, hidden^-0.5), mixes per query.
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

// Message gate first half: h and m normed, concatenated, bf16, projected by down [rank][2 hidden] to `rank`, rounded.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_down(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], const device bfloat* w0 [[buffer(4)]], const device bfloat* w1 [[buffer(5)]],
    const device bfloat* w2 [[buffer(6)]], device bfloat* down [[buffer(7)]], constant LpGateArgs& a [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  threadgroup float part[I2_SGS][LP_RANK_BLOCK];
  const uint row = grid.x, g = grid.y, r0 = grid.z * LP_RANK_BLOCK;
  const uint ranks = min(LP_RANK_BLOCK, a.rank - r0);
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
  // Eight ranks, each in the old element order: hidden half, then message half.
  float acc[LP_RANK_BLOCK];
  for (uint j = 0; j < LP_RANK_BLOCK; ++j) acc[j] = 0.0f;
  const device bfloat* wr = w + ulong(r0) * 2 * H;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H)
      for (uint j = 0; j < LP_RANK_BLOCK; ++j)
        if (j < ranks) acc[j] = fma(float(wr[ulong(j) * 2 * H + i]), xh[e], acc[j]);
  }
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H)
      for (uint j = 0; j < LP_RANK_BLOCK; ++j)
        if (j < ranks) acc[j] = fma(float(wr[ulong(j) * 2 * H + H + i]), xm[e], acc[j]);
  }
  for (uint j = 0; j < LP_RANK_BLOCK; ++j) {
    const float t = simd_sum(acc[j]);
    if (lane == 0) part[sg][j] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < ranks) {
    float t = part[0][tid];
    for (uint s = 1; s < I2_SGS; ++s) t += part[s][tid];
    down[(ulong(g) * a.rows + row) * a.rank + r0 + tid] = bfloat(t);
  }
}

// Second half + weighted RMS norm: forget = tanh(Wf d), gain = tanh(Wg d) over fp32 gates [hidden][rank]; y normed.
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

// Few rows: the one-group mix above idles the GPU; statistics and mix run on many groups, same per-thread chains.
constant constexpr uint LP_STAT = 1 + LP_MAX_QUERIES; // per source: sum of squares, then one dot per query

// Source `grid.x` of row `grid.y`: its sum of squares and each query's dot, as the eight-simdgroup sums of loop_mix.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_mix_stats(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* q0 [[buffer(2)]], const device float* q1 [[buffer(3)]], const device float* q2 [[buffer(4)]],
    const device float* q3 [[buffer(5)]], device float* stats [[buffer(6)]], constant LpMixArgs& a [[buffer(7)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_SGS][LP_STAT];
  const uint s = grid.x, row = grid.y, H = a.hidden;
  float acc[LP_STAT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint i = tid; i < H; i += I2_TG) {
    const float v = lp_source(blocks, partial, a, s, row, i);
    acc[0] = fma(v, v, acc[0]);
    acc[1] = fma(q0[i], v, acc[1]);
    if (a.queries > 1) acc[2] = fma(q1[i], v, acc[2]);
    if (a.queries > 2) acc[3] = fma(q2[i], v, acc[3]);
    if (a.queries > 3) acc[4] = fma(q3[i], v, acc[4]);
  }
  for (uint k = 0; k < LP_STAT; ++k) {
    const float t = simd_sum(acc[k]);
    if (lane == 0) part[sg][k] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < LP_STAT) {
    float t = part[0][tid];
    for (uint g = 1; g < I2_SGS; ++g) t += part[g][tid];
    stats[(ulong(row) * LP_MAX_SOURCES + s) * LP_STAT + tid] = t;
  }
}

// Element grid.x * 256 + tid of row grid.y: softmax coefficients from the statistics, then every query's fp32 mix.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_mix_apply(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* stats [[buffer(2)]], device float* out [[buffer(3)]], constant LpMixArgs& a [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint S = a.blocks + 1, H = a.hidden, row = grid.y, i = grid.x * I2_TG + tid;
  if (i >= H) return;
  const device float* st = stats + ulong(row) * LP_MAX_SOURCES * LP_STAT;
  float inv_tg[LP_MAX_SOURCES];
  for (uint s = 0; s < S; ++s) inv_tg[s] = i2_rsqrt_mean(st[s * LP_STAT], H, a.eps_unweighted);
  float coef[LP_MAX_QUERIES][LP_MAX_SOURCES];
  for (uint q = 0; q < LP_MAX_QUERIES; ++q) {
    for (uint s = 0; s < LP_MAX_SOURCES; ++s) coef[q][s] = 0.0f;
    if (q >= a.queries) continue;
    float score[LP_MAX_SOURCES];
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) {
      score[s] = (st[s * LP_STAT + 1 + q] * inv_tg[s]) * a.scale;
      m = i2_max(m, score[s]);
    }
    float l = 0.0f;
    for (uint s = 0; s < S; ++s) {
      score[s] = score[s] == -INFINITY ? 0.0f : metal::precise::exp(score[s] - m);
      l += score[s];
    }
    for (uint s = 0; s < S; ++s) coef[q][s] = inv_tg[s] * (score[s] / l);
  }
  float u[LP_MAX_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < S; ++s) {
    const float v = lp_source(blocks, partial, a, s, row, i);
    for (uint q = 0; q < LP_MAX_QUERIES; ++q) u[q] = fma(coef[q][s], v, u[q]);
  }
  for (uint q = 0; q < LP_MAX_QUERIES; ++q)
    if (q < a.queries) out[ulong(q) * a.out_stride + ulong(row) * H + i] = u[q];
}

// Element grid.x * 256 + tid of gate grid.y, row grid.z: the gated state before its norm, fp32 [gate][rows][hidden].
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_y(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], const device bfloat* down [[buffer(4)]],
    const device bfloat* f0 [[buffer(5)]], const device bfloat* f1 [[buffer(6)]], const device bfloat* f2 [[buffer(7)]],
    const device bfloat* g0 [[buffer(8)]], const device bfloat* g1 [[buffer(9)]], const device bfloat* g2 [[buffer(10)]],
    device float* y [[buffer(11)]], constant LpGateArgs& a [[buffer(12)]], uint3 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float dn[LP_MAX_RANK];
  const uint g = grid.y, row = grid.z, H = a.hidden, R = a.rank;
  if (tid < R) dn[tid] = float(down[(ulong(g) * a.rows + row) * R + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint i = grid.x * I2_TG + tid;
  if (i >= H) return;
  const device float* h = (g == 0 ? h0 : (g == 1 ? h1 : h2)) + ulong(row) * H;
  const device float* m = msg + ulong(lp_slot(a, g)) * a.msg_stride + ulong(row) * H;
  const device bfloat* wf = (g == 0 ? f0 : (g == 1 ? f1 : f2)) + ulong(i) * R;
  const device bfloat* wg = (g == 0 ? g0 : (g == 1 ? g1 : g2)) + ulong(i) * R;
  float f = 0.0f, gn = 0.0f;
  for (uint r = 0; r < R; ++r) {
    f = fma(float(wf[r]), dn[r], f);
    gn = fma(float(wg[r]), dn[r], gn);
  }
  const float forget = metal::precise::tanh(f);
  y[(ulong(g) * a.rows + row) * H + i] = (1.0f - forget) * h[i] + metal::precise::tanh(gn) * m[i];
}

// gate_y on MMA for many rows (rank % 8 == 0): 16 rows x 16 columns a simdgroup, forget and gain one rank-order chain.
[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_loop_gate_y_mma(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], const device bfloat* down [[buffer(4)]],
    const device bfloat* f0 [[buffer(5)]], const device bfloat* f1 [[buffer(6)]], const device bfloat* f2 [[buffer(7)]],
    const device bfloat* g0 [[buffer(8)]], const device bfloat* g1 [[buffer(9)]], const device bfloat* g2 [[buffer(10)]],
    device float* y [[buffer(11)]], constant LpGateArgs& a [[buffer(12)]], uint3 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint g = grid.z, H = a.hidden, R = a.rank, rows = a.rows;
  const uint r0 = grid.y * 16u, c0 = grid.x * 16u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  const device bfloat* dn = down + ulong(g) * rows * R;
  const device bfloat* wf = g == 0 ? f0 : (g == 1 ? f1 : f2);
  const device bfloat* wg = g == 0 ? g0 : (g == 1 ? g1 : g2);
  simdgroup_matrix<float, 8, 8> fa[2][2], ga[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) {
      fa[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      ga[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  for (uint k = 0; k < R; k += 8u) {
    simdgroup_matrix<bfloat, 8, 8> xa[2], fb[2], gb[2];
    for (uint i = 0; i < 2; ++i) {
      const device bfloat* p = dn + ulong(min(r0 + i * 8u + fm, rows - 1u)) * R + k + fn;
      xa[i].thread_elements()[0] = p[0];
      xa[i].thread_elements()[1] = p[1];
    }
    for (uint j = 0; j < 2; ++j) {
      const ulong c = min(c0 + j * 8u + fn, H - 1u), d = min(c0 + j * 8u + fn + 1u, H - 1u);
      fb[j].thread_elements()[0] = wf[c * R + k + fm];
      fb[j].thread_elements()[1] = wf[d * R + k + fm];
      gb[j].thread_elements()[0] = wg[c * R + k + fm];
      gb[j].thread_elements()[1] = wg[d * R + k + fm];
    }
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 2; ++j) {
        simdgroup_multiply_accumulate(fa[i][j], xa[i], fb[j], fa[i][j]);
        simdgroup_multiply_accumulate(ga[i][j], xa[i], gb[j], ga[i][j]);
      }
  }
  const device float* hp = g == 0 ? h0 : (g == 1 ? h1 : h2);
  const device float* mp = msg + ulong(lp_slot(a, g)) * a.msg_stride;
  for (uint i = 0; i < 2; ++i) {
    const uint row = r0 + i * 8u + fm;
    if (row >= rows) continue;
    for (uint j = 0; j < 2; ++j)
      for (uint e = 0; e < 2; ++e) {
        const uint col = c0 + j * 8u + fn + e;
        if (col >= H) continue;
        const float keep = metal::precise::tanh(fa[i][j].thread_elements()[e]);
        const ulong at = ulong(row) * H + col;
        y[(ulong(g) * rows + row) * H + col] = (1.0f - keep) * hp[at] + metal::precise::tanh(ga[i][j].thread_elements()[e]) * mp[at];
      }
  }
}

// Weighted RMS norm of gate grid.y's row grid.x of gate_y's output, rounded to bf16, as the tail of gate_mix.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_norm(
    const device float* y [[buffer(0)]], const device bfloat* n0 [[buffer(1)]], const device bfloat* n1 [[buffer(2)]],
    const device bfloat* n2 [[buffer(3)]], device bfloat* out [[buffer(4)]], constant LpGateArgs& a [[buffer(5)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  const uint row = grid.x, g = grid.y, H = a.hidden;
  const device float* src = y + (ulong(g) * a.rows + row) * H;
  const device bfloat* nw = g == 0 ? n0 : (g == 1 ? n1 : n2);
  float v[I2_MAX_ELEMS];
  float ss = 0.0f;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    v[e] = i < H ? src[i] : 0.0f;
    ss = fma(v[e], v[e], ss);
  }
  const float inv = i2_rsqrt_mean(i2_tg_sum(ss, scratch, sg, lane), H, a.eps_norm);
  device bfloat* dst = out + ulong(g) * a.out_stride + ulong(row) * H;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H) dst[i] = bfloat(float(nw[i]) * (v[e] * inv));
  }
}

constant constexpr uint LP_DOT_BLOCK = 2; // gate_dot ranks one threadgroup computes

// Row grid.x of gate grid.y: h and m normed and rounded to bf16, [gate][rows][h then m], as gate_down starts.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_prep(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]], const device float* h2 [[buffer(2)]],
    const device float* msg [[buffer(3)]], device bfloat* xn [[buffer(4)]], constant LpGateArgs& a [[buffer(5)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  const uint row = grid.x, g = grid.y, H = a.hidden;
  const device float* h = (g == 0 ? h0 : (g == 1 ? h1 : h2)) + ulong(row) * H;
  const device float* m = msg + ulong(lp_slot(a, g)) * a.msg_stride + ulong(row) * H;
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
  device bfloat* dst = xn + (ulong(g) * a.rows + row) * 2 * H;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i >= H) continue;
    dst[i] = bfloat(xh[e] * inv_h);
    dst[H + i] = bfloat(xm[e] * inv_m);
  }
}

// Ranks grid.z * 2 ..+2 of gate grid.y, row grid.x, from gate_prep's vectors: as gate_down, the same chains per rank.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_gate_dot(
    const device bfloat* xn [[buffer(0)]], const device bfloat* w0 [[buffer(1)]], const device bfloat* w1 [[buffer(2)]],
    const device bfloat* w2 [[buffer(3)]], device bfloat* down [[buffer(4)]], constant LpGateArgs& a [[buffer(5)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_SGS][LP_DOT_BLOCK];
  const uint row = grid.x, g = grid.y, r0 = grid.z * LP_DOT_BLOCK;
  const uint ranks = min(LP_DOT_BLOCK, a.rank - r0);
  const uint H = a.hidden;
  const device bfloat* x = xn + (ulong(g) * a.rows + row) * 2 * H;
  const device bfloat* wr = (g == 0 ? w0 : (g == 1 ? w1 : w2)) + ulong(r0) * 2 * H;
  float acc[LP_DOT_BLOCK];
  for (uint j = 0; j < LP_DOT_BLOCK; ++j) acc[j] = 0.0f;
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H) {
      const float xh = float(x[i]);
      for (uint j = 0; j < LP_DOT_BLOCK; ++j)
        if (j < ranks) acc[j] = fma(float(wr[ulong(j) * 2 * H + i]), xh, acc[j]);
    }
  }
  for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
    const uint i = tid + e * I2_TG;
    if (i < H) {
      const float xm = float(x[H + i]);
      for (uint j = 0; j < LP_DOT_BLOCK; ++j)
        if (j < ranks) acc[j] = fma(float(wr[ulong(j) * 2 * H + H + i]), xm, acc[j]);
    }
  }
  for (uint j = 0; j < LP_DOT_BLOCK; ++j) {
    const float t = simd_sum(acc[j]);
    if (lane == 0) part[sg][j] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < ranks) {
    float t = part[0][tid];
    for (uint s = 1; s < I2_SGS; ++s) t += part[s][tid];
    down[(ulong(g) * a.rows + row) * a.rank + r0 + tid] = bfloat(t);
  }
}
