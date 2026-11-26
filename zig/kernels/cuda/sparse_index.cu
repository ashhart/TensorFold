// Every family's indexer on the GPU (core/sparse_index.zig): the scores and the block summaries a first stage scores.
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_iscore {

constexpr int D = 128, KEYS = 64;  // key width; keys a block (a thread each)

// A thread a key of row r: per head an fma chain over d, phi, times w; heads in order (sum, or max keeping NaN).
__device__ __forceinline__ void scores(const __nv_bfloat16* q, const __nv_bfloat16* k, int64_t ks, const float* w,
                                       const int* counts, float* out, int64_t ld, int n, int H, int hk, float scale,
                                       int relu, int max) {
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  float* ws = reinterpret_cast<float*>(qs + H * D);
  const int64_t r = blockIdx.y;
  for (int i = threadIdx.x; i < H * D / 8; i += KEYS)
    reinterpret_cast<uint4*>(qs)[i] = reinterpret_cast<const uint4*>(q + r * H * D)[i];
  for (int i = threadIdx.x; i < H; i += KEYS) ws[i] = w ? w[r * H + i] : 1.0f;
  __syncthreads();
  const int64_t key = (int64_t)blockIdx.x * KEYS + threadIdx.x;
  if (key >= ld) return;
  const int64_t c = counts ? (int64_t)counts[r] : n, limit = c < 0 ? 0 : (c > n ? n : c);
  if (key >= limit) {  // past the row's count (clamped to [0, n]), up to ld: no stale score is left
    out[r * ld + key] = -INFINITY;
    return;
  }
  const int group = H / hk;
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    const uint4* kp = reinterpret_cast<const uint4*>(k + key * ks + (int64_t)(h / group) * D);
    const uint4* qp = reinterpret_cast<const uint4*>(qs + h * D);
    float dot = 0.0f;
    for (int c = 0; c < D / 8; ++c) {
      const uint4 kv = kp[c], qv = qp[c];
      const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kv);
      const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&qv);
      for (int j = 0; j < 4; ++j) {
        const float2 a = __bfloat1622float2(qb[j]), b = __bfloat1622float2(kb[j]);
        dot = __fmaf_rn(a.x, b.x, dot);
        dot = __fmaf_rn(a.y, b.y, dot);
      }
    }
    if (relu && dot < 0.0f) dot = 0.0f;  // NaN stays NaN
    const float t = w ? __fmul_rn(ws[h], dot) : dot;
    if (h == 0) acc = t;
    else if (max) acc = (t > acc || t != t) ? t : acc;
    else acc = __fadd_rn(acc, t);
  }
  out[r * ld + key] = __fmul_rn(acc, scale);
}

// Keys [lo, hi) (hi > lo) summarised, a thread a dim: mean (fp32 in key order), max (NaN kept), first or last.
__device__ __forceinline__ float summary(const __nv_bfloat16* k, int64_t ks, int64_t lo, int64_t hi, int mode,
                                         int unit) {
  __shared__ float part[D / 32];
  const int d = threadIdx.x, lane = d & 31, warp = d >> 5;
  float acc = 0.0f;
  for (int64_t i = lo; i < hi; ++i) {
    float x = __bfloat162float(k[i * ks + d]);
    if (unit) {  // the key times 1 / sqrt(its mean square), 0 when that is 0
      float sq = __fmul_rn(x, x);
      for (int o = 16; o > 0; o >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
      __syncthreads();
      if (lane == 0) part[warp] = sq;
      __syncthreads();
      const float ss = __fadd_rn(__fadd_rn(part[0], part[1]), __fadd_rn(part[2], part[3]));
      const float ms = __fdiv_rn(ss, (float)D);
      x = ms > 0.0f ? __fmul_rn(x, __fdiv_rn(1.0f, __fsqrt_rn(ms))) : (ms == 0.0f ? 0.0f : __fmul_rn(x, ms));
    }
    if (i == lo) acc = x;
    else if (mode == 0) acc = __fadd_rn(acc, x);
    else if (mode == 1) acc = (x > acc || x != x) ? x : acc;
    else if (mode == 3) acc = x;
  }
  if (mode == 0) acc = __fdiv_rn(acc, (float)(hi - lo));
  return acc;
}

}  // namespace tf_iscore

// grid (ceil(ld / 64), rows), 64 threads, H * D * 2 + H * 4 bytes of dynamic shared memory.
extern "C" __global__ void __launch_bounds__(64)
    tf_index_scores(const __nv_bfloat16* q, const __nv_bfloat16* k, int64_t ks, const float* w, const int* counts,
                    float* out, int64_t ld, int n, int H, int hk, float scale, int relu, int max) {
  tf_iscore::scores(q, k, ks, w, counts, out, ld, n, H, hk, scale, relu, max);
}

// A block a summary, 128 threads (a dim each); grid: ceil(n / B) summaries.
extern "C" __global__ void __launch_bounds__(128)
    tf_index_blocks(const __nv_bfloat16* k, int64_t ks, __nv_bfloat16* out, int64_t os, int64_t n, int B, int mode,
                    int unit) {
  const int64_t lo = (int64_t)blockIdx.x * B, hi = lo + B < n ? lo + B : n;
  out[(int64_t)blockIdx.x * os + threadIdx.x] = __float2bfloat16_rn(tf_iscore::summary(k, ks, lo, hi, mode, unit));
}

// Row r's open block, keys [end - end % B, end) of end = ends[r] = t + 1 clamped to [0, n]; 0 if none. A row a block.
extern "C" __global__ void __launch_bounds__(128)
    tf_index_open_blocks(const __nv_bfloat16* k, int64_t ks, __nv_bfloat16* out, int64_t os, const int* ends, int64_t n,
                         int B, int mode, int unit) {
  const int64_t e = ends[blockIdx.x], end = e < 0 ? 0 : (e > n ? n : e), lo = end - end % B;
  const float v = lo < end ? tf_iscore::summary(k, ks, lo, end, mode, unit) : 0.0f;
  out[(int64_t)blockIdx.x * os + threadIdx.x] = __float2bfloat16_rn(v);
}
