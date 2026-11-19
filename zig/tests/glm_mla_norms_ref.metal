// tf-glm-mla-norms' reference: glm_glue.metal's rms, layer norm and scale before glm_mla_norms (appended).

// MLX's rms_single_row on strided rows: thread t squares 4t..4t+3, simd_sum, a 32-slot simd_sum, precise rsqrt.
kernel void glm_rms_ref(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                    device bfloat* y [[buffer(2)]], constant GlmRows& a [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_sums[32];
  threadgroup float inv_rms[1];
  const uint width = uint(a.width), lo = 4 * t;
  const device bfloat* src = x + size_t(row) * a.x_stride + lo;
  device bfloat* dst = y + size_t(row) * a.y_stride + lo;
  const device bfloat* gain = w + lo;
  const bool whole = lo + 4 <= width;
  float v[4];
  float sq = 0.0f;
  if (whole) {
    for (int j = 0; j < 4; ++j) {
      v[j] = src[j];
      sq += v[j] * v[j];
    }
  } else {
    for (int j = 0; j < 4; ++j) {
      v[j] = lo + j < width ? float(src[j]) : 0.0f;
      sq += v[j] * v[j];
    }
  }
  sq = simd_sum(sq);
  if (sg == 0) sg_sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sg_sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sg_sums[lane]);
    if (lane == 0) inv_rms[0] = metal::precise::rsqrt(all / float(width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (whole) {
    for (int j = 0; j < 4; ++j) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  } else {
    for (int j = 0; j < 4; ++j)
      if (lo + j < width) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  }
}

// MLX's layer_norm_single_row (8 values a thread, 32 threads a row): the indexer's key norm with weight and bias.
inline float glm_tg_sum_ref(float x, threadgroup float* xs, uint lane, uint sg) {
  x = simd_sum(x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) xs[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  x = xs[lane];
  return simd_sum(x);
}

kernel void glm_layer_norm_ref(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                           const device bfloat* b [[buffer(2)]], device bfloat* y [[buffer(3)]],
                           constant GlmRows& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 8;
  float thread_x[N_READS] = {0};
  threadgroup float local_buffer[32];
  if (sg == 0) local_buffer[lane] = 0;
  const int axis = a.width;
  const device bfloat* xr = x + size_t(row) * a.x_stride + lid * N_READS;
  device bfloat* out = y + size_t(row) * a.y_stride + lid * N_READS;
  const device bfloat* wr = w + lid * N_READS;
  const device bfloat* br = b + lid * N_READS;
  const bool safe = int(lid) * N_READS + N_READS <= axis;
  const int n = axis - int(lid) * N_READS;
  if (safe) {
    for (int i = 0; i < N_READS; i++) thread_x[i] = xr[i];
  } else {
    for (int i = 0; i < n; i++) thread_x[i] = xr[i];
  }
  float mean = 0;
  for (int i = 0; i < N_READS; i++) mean += thread_x[i];
  mean = glm_tg_sum_ref(mean, local_buffer, lane, sg);
  mean /= axis;
  float normalizer = 0;
  if (!safe) {
    for (int i = max(n, 0); i < N_READS; i++) thread_x[i] = mean;
  }
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    normalizer += thread_x[i] * thread_x[i];
  }
  normalizer = glm_tg_sum_ref(normalizer, local_buffer, lane, sg);
  normalizer = metal::precise::rsqrt(normalizer / axis + a.eps);
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      out[i] = wr[i] * static_cast<bfloat>(thread_x[i]) + br[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      out[i] = wr[i] * static_cast<bfloat>(thread_x[i]) + br[i];
    }
  }
}

// y = bf16(x * s) for a power-of-two s (exact): the scaled queries of MLX's unfused attention, the indexer weights.
kernel void glm_scale_ref(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
                      constant GlmRows& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (int(p.x) >= a.width || int(p.y) >= a.rows) return;
  y[size_t(p.y) * a.y_stride + p.x] = bfloat(float(x[size_t(p.y) * a.x_stride + p.x]) * a.eps);
}
