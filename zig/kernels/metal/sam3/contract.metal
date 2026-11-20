#define TF_NAX_STRICT_FP32
#include "../nax.h"
using namespace tfp;

#define SAM_MMA(NAME, TA, TB, CUT) \
[[max_total_threads_per_threadgroup(32)]] kernel void NAME( \
    const device float *a [[buffer(0)]], const device float *b [[buffer(1)]], \
    device float *out [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) { \
  const short2 home = frag_home(ushort(lane)); \
  frag<float> x, y0, y1, lo = frag<float>(0), hi = frag<float>(0); \
  frag_get(x, a, 16, 0, 0, home); \
  frag_get(y0, b, TB ? 16 : 32, 0, 0, home); \
  frag_get(y1, b, TB ? 16 : 32, TB ? 16 : 0, TB ? 0 : 16, home); \
  if constexpr (CUT) mma_16x32_cut<TA, TB>(lo, hi, x, y0, y1); \
  else mma_16x32<TA, TB>(lo, hi, x, y0, y1); \
  frag_put(lo, out, 32, 0, 0, home); \
  frag_put(hi, out, 32, 0, 16, home); \
}
SAM_MMA(tf_sam3_mma_nn, false, false, false)
SAM_MMA(tf_sam3_mma_nt, false, true, false)
SAM_MMA(tf_sam3_mma_tn, true, false, false)
SAM_MMA(tf_sam3_mma_tt, true, true, false)
SAM_MMA(tf_sam3_cut_nn, false, false, true)
SAM_MMA(tf_sam3_cut_nt, false, true, true)
SAM_MMA(tf_sam3_cut_tn, true, false, true)
SAM_MMA(tf_sam3_cut_tt, true, true, true)

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_sam3_fma(const device uint *values, device uint *out, constant uint &count,
                       uint i [[thread_position_in_grid]]) {
  if (i < count) out[i] = f32_fma_bits(values[3*i], values[3*i+1], values[3*i+2]);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_sam3_fma_flush(const device uint *values, device uint *out, constant uint &count,
                             uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  uint x[3];
  for (uint j = 0; j < 3; j++) {
    const uint value = values[3*i+j], magnitude = value & 0x7fffffffu;
    x[j] = magnitude < 0x00800000u ? value & 0x80000000u : value;
  }
  out[i] = f32_fma_bits(x[0], x[1], x[2]);
}
