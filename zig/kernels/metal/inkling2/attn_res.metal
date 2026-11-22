// Inkling 2 attention residual + RMS norm: fp32 query attends normed sources (hidden^-0.5), mix normed (eps_n)

constant constexpr uint I2_MAX_SOURCES = 17;
constant constexpr uint I2_MAX_QUERIES = 3;

struct I2AttnResArgs {
  uint hidden, blocks, block_stride, partial_f32;
  uint queries, out_stride, partial_pitch, pad;
  float eps_unweighted, eps_norm, scale;
};

inline float i2_source(const device bfloat* blocks, const device uchar* partial, constant I2AttnResArgs& a, uint s,
                       uint row, uint i) {
  if (s < a.blocks) return float(blocks[ulong(s) * a.block_stride + ulong(row) * a.hidden + i]);
  if (a.partial_f32 != 0) return ((const device float*)partial)[ulong(row) * a.partial_pitch + i];
  return float(((const device bfloat*)partial)[ulong(row) * a.partial_pitch + i]);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_attn_res(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* queries [[buffer(2)]], const device bfloat* w0 [[buffer(3)]],
    const device bfloat* w1 [[buffer(4)]], const device bfloat* w2 [[buffer(5)]], device bfloat* out [[buffer(6)]],
    constant I2AttnResArgs& a [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_SGS][I2_MAX_QUERIES * I2_MAX_SOURCES];
  threadgroup float inv_tg[I2_MAX_SOURCES];
  threadgroup float scratch[I2_SGS * 4];
  const uint S = a.blocks + 1;
  const uint H = a.hidden;

  // 1. each source's 1 / rms.
  float ss[I2_MAX_SOURCES];
  for (uint s = 0; s < I2_MAX_SOURCES; ++s) {
    ss[s] = 0.0f;
    if (s >= S) continue;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += I2_TG) {
      const float v = i2_source(blocks, partial, a, s, row, i);
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

  // 2. query . source for every query and source (the weight of the norm applies after the sum: inv * (q . x)).
  float dots[I2_MAX_QUERIES][I2_MAX_SOURCES];
  for (uint q = 0; q < I2_MAX_QUERIES; ++q)
    for (uint s = 0; s < I2_MAX_SOURCES; ++s) {
      dots[q][s] = 0.0f;
      if (q >= a.queries || s >= S) continue;
      float acc = 0.0f;
      for (uint i = tid; i < H; i += I2_TG) acc = fma(queries[q * H + i], i2_source(blocks, partial, a, s, row, i), acc);
      dots[q][s] = simd_sum(acc);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0)
    for (uint q = 0; q < a.queries; ++q)
      for (uint s = 0; s < S; ++s) part[sg][q * I2_MAX_SOURCES + s] = dots[q][s];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. softmax over the sources (identical in every thread), folded with 1 / rms into one coefficient a source.
  float coef[I2_MAX_QUERIES][I2_MAX_SOURCES];
  for (uint q = 0; q < I2_MAX_QUERIES; ++q) {
    for (uint s = 0; s < I2_MAX_SOURCES; ++s) coef[q][s] = 0.0f;
    if (q >= a.queries) continue;
    float score[I2_MAX_SOURCES];
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) {
      float t = part[0][q * I2_MAX_SOURCES + s];
      for (uint g = 1; g < I2_SGS; ++g) t += part[g][q * I2_MAX_SOURCES + s];
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

  // 4. the mix, its second norm and the bf16 output.
  float mix[I2_MAX_QUERIES][I2_MAX_ELEMS];
  float ss2[I2_MAX_QUERIES];
  for (uint q = 0; q < I2_MAX_QUERIES; ++q) {
    ss2[q] = 0.0f;
    if (q >= a.queries) continue;
    float acc = 0.0f;
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      float u = 0.0f;
      if (i < H) {
        for (uint s = 0; s < S; ++s) u = fma(coef[q][s], i2_source(blocks, partial, a, s, row, i), u);
        acc = fma(u, u, acc);
      }
      mix[q][e] = u;
    }
    ss2[q] = acc;
  }
  for (uint q = 0; q < I2_MAX_QUERIES; ++q) {
    if (q >= a.queries) break;
    const float inv = i2_rsqrt_mean(i2_tg_sum(ss2[q], scratch, sg, lane), H, a.eps_norm);
    const device bfloat* w = q == 0 ? w0 : (q == 1 ? w1 : w2);
    device bfloat* dst = out + ulong(q) * a.out_stride + ulong(row) * H;
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      if (i < H) dst[i] = bfloat(float(w[i]) * (mix[q][e] * inv));
    }
  }
}

// Few rows: the one-group kernel above idles the GPU, so statistics, mix and norm run on many groups, same chains.
constant constexpr uint I2_STAT = 1 + I2_MAX_QUERIES; // per source: sum of squares, then one dot per query

// Source `grid.x` of row `grid.y`: its sum of squares and each query's dot, as the eight-simdgroup sums of attn_res.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_attn_res_stats(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* queries [[buffer(2)]], device float* stats [[buffer(3)]],
    constant I2AttnResArgs& a [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[I2_SGS][I2_STAT];
  const uint s = grid.x, row = grid.y, H = a.hidden;
  float acc[I2_STAT] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint i = tid; i < H; i += I2_TG) {
    const float v = i2_source(blocks, partial, a, s, row, i);
    acc[0] = fma(v, v, acc[0]);
    acc[1] = fma(queries[i], v, acc[1]);
    if (a.queries > 1) acc[2] = fma(queries[H + i], v, acc[2]);
    if (a.queries > 2) acc[3] = fma(queries[2 * H + i], v, acc[3]);
  }
  for (uint k = 0; k < I2_STAT; ++k) {
    const float t = simd_sum(acc[k]);
    if (lane == 0) part[sg][k] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < I2_STAT) {
    float t = part[0][tid];
    for (uint g = 1; g < I2_SGS; ++g) t += part[g][tid];
    stats[(ulong(row) * I2_MAX_SOURCES + s) * I2_STAT + tid] = t;
  }
}

// Element grid.x * 256 + tid of row grid.y: softmax coefficients from the statistics, then every query's fp32 mix.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_attn_res_mix(
    const device bfloat* blocks [[buffer(0)]], const device uchar* partial [[buffer(1)]],
    const device float* stats [[buffer(2)]], device float* mixed [[buffer(3)]], constant I2AttnResArgs& a [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint S = a.blocks + 1, H = a.hidden, row = grid.y, i = grid.x * I2_TG + tid;
  if (i >= H) return;
  const device float* st = stats + ulong(row) * I2_MAX_SOURCES * I2_STAT;
  float inv_tg[I2_MAX_SOURCES];
  for (uint s = 0; s < S; ++s) inv_tg[s] = i2_rsqrt_mean(st[s * I2_STAT], H, a.eps_unweighted);
  float coef[I2_MAX_QUERIES][I2_MAX_SOURCES];
  for (uint q = 0; q < I2_MAX_QUERIES; ++q) {
    for (uint s = 0; s < I2_MAX_SOURCES; ++s) coef[q][s] = 0.0f;
    if (q >= a.queries) continue;
    float score[I2_MAX_SOURCES];
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) {
      const float t = st[s * I2_STAT + 1 + q];
      score[s] = a.scale * (t * inv_tg[s]);
      m = i2_max(m, score[s]);
    }
    float l = 0.0f;
    for (uint s = 0; s < S; ++s) {
      score[s] = score[s] == -INFINITY ? 0.0f : metal::precise::exp(score[s] - m);
      l += score[s];
    }
    for (uint s = 0; s < S; ++s) coef[q][s] = (score[s] / l) * inv_tg[s];
  }
  float u[I2_MAX_QUERIES] = {0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < S; ++s) {
    const float v = i2_source(blocks, partial, a, s, row, i);
    for (uint q = 0; q < I2_MAX_QUERIES; ++q) u[q] = fma(coef[q][s], v, u[q]);
  }
  for (uint q = 0; q < I2_MAX_QUERIES; ++q)
    if (q < a.queries) mixed[ulong(q) * a.out_stride + ulong(row) * H + i] = u[q];
}

// Row grid of attn_res_mix's fp32 mixes: each query's second norm, weighted, rounded to bf16.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_attn_res_norm(
    const device float* mixed [[buffer(0)]], const device bfloat* w0 [[buffer(1)]],
    const device bfloat* w1 [[buffer(2)]],
    const device bfloat* w2 [[buffer(3)]], device bfloat* out [[buffer(4)]], constant I2AttnResArgs& a [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  const uint H = a.hidden;
  for (uint q = 0; q < I2_MAX_QUERIES; ++q) {
    if (q >= a.queries) break;
    const device float* src = mixed + ulong(q) * a.out_stride + ulong(row) * H;
    float acc = 0.0f;
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      if (i < H) acc = fma(src[i], src[i], acc);
    }
    const float inv = i2_rsqrt_mean(i2_tg_sum(acc, scratch, sg, lane), H, a.eps_norm);
    const device bfloat* w = q == 2 ? w2 : (q == 1 ? w1 : w0);
    device bfloat* dst = out + ulong(row) * H + ulong(q) * a.out_stride;
    for (uint e = 0; e < I2_MAX_ELEMS; ++e) {
      const uint i = tid + e * I2_TG;
      if (i < H) dst[i] = bfloat(float(w[i]) * (src[i] * inv));
    }
  }
}
