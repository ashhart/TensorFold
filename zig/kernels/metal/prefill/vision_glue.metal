// The image towers' shared glue (after gemm_nax_kernel.metal): bf16 rounding, threadgroup sum and max, row softmax.
inline float tf_vis_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (n threads, at most 1024), returned to all of them.
inline float tf_vis_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// One threadgroup's largest `v` (n threads, at most 1024), returned to all of them.
inline float tf_vis_max(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_max(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : -INFINITY;
    s = simd_max(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// A softmax kernel's body: row `row` of S [*, n] (fp32) to softmax(s[i] SCALE) in place, precise exps, tn threads.
#define TF_VIS_SOFTMAX_BODY(SCALE)                                                                                 \
  threadgroup float part[33];                                                                                      \
  device float* s = S + ulong(row) * n;                                                                            \
  float m = -INFINITY;                                                                                             \
  for (uint i = t; i < n; i += tn) m = max(m, s[i] SCALE);                                                         \
  m = simd_max(m);                                                                                                 \
  if (lane == 0) part[sg] = m;                                                                                     \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                                                 \
  if (sg == 0) {                                                                                                   \
    float v = lane < (tn + 31) / 32 ? part[lane] : -INFINITY;                                                      \
    v = simd_max(v);                                                                                               \
    if (lane == 0) part[32] = v;                                                                                   \
  }                                                                                                                \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                                                 \
  m = part[32];                                                                                                    \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                                                 \
  float z = 0.0f;                                                                                                  \
  for (uint i = t; i < n; i += tn) {                                                                               \
    const float e = metal::precise::exp(s[i] SCALE - m);                                                           \
    s[i] = e;                                                                                                      \
    z += e;                                                                                                        \
  }                                                                                                                \
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);                                                      \
  for (uint i = t; i < n; i += tn) s[i] *= inv;

// LayerNorm, one rounding: STATS the row's mean and inv over IN(i), OUT yr[i] = bf16((IN(i) - mean) * inv * w + b).
#define TF_VIS_LAYER_NORM_STATS(IN)                                                                                \
  threadgroup float part[33];                                                                                      \
  float s = 0.0f;                                                                                                  \
  for (uint i = t; i < p.width; i += n) s += IN(i);                                                                \
  const float mean = tf_vis_sum(s, part, n, lane, sg) / float(p.width);                                            \
  float v = 0.0f;                                                                                                  \
  for (uint i = t; i < p.width; i += n) {                                                                          \
    const float d = IN(i) - mean;                                                                                  \
    v += d * d;                                                                                                    \
  }                                                                                                                \
  const float inv = metal::precise::rsqrt(tf_vis_sum(v, part, n, lane, sg) / float(p.width) + p.eps);
#define TF_VIS_LAYER_NORM_OUT(IN)                                                                                  \
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((IN(i) - mean) * inv * float(w[i]) + float(b[i]));
// IN for a kernel whose row of x is xr.
#define TF_VIS_ROW(i) float(xr[i])
