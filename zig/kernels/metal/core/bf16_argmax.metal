// The greedy draw for decode glue: a bf16 logits row's argmax, NaN first, ties to the lower id (MLX argmax).
#include <metal_stdlib>
using namespace metal;

// The better of two (value, id) candidates: NaN first (lower id among NaNs), then the larger value, then the lower id.
inline bool tf_bf16_argmax_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// Kernel NAME: logits [vocab] bf16 to its argmax in ids[0], one threadgroup of 1024.
#define TF_BF16_ARGMAX(NAME)                                                                                       \
  [[kernel]] void NAME(const device bfloat* logits [[buffer(0)]], device uint* ids [[buffer(1)]],                  \
      constant uint& vocab [[buffer(2)]], uint t [[thread_position_in_threadgroup]],                               \
      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                       \
    threadgroup float vals[32];                                                                                    \
    threadgroup uint idxs[32];                                                                                     \
    float best = -INFINITY;                                                                                        \
    uint at = 0xffffffffu;                                                                                         \
    for (uint i = t; i < vocab; i += 1024) {                                                                       \
      const float v = float(logits[i]);                                                                            \
      if (tf_bf16_argmax_better(v, i, best, at)) { best = v; at = i; }                                             \
    }                                                                                                              \
    for (ushort d = 16; d > 0; d >>= 1) {                                                                          \
      const float ov = simd_shuffle_xor(best, d);                                                                  \
      const uint oi = simd_shuffle_xor(at, d);                                                                     \
      if (tf_bf16_argmax_better(ov, oi, best, at)) { best = ov; at = oi; }                                         \
    }                                                                                                              \
    if (lane == 0) { vals[sg] = best; idxs[sg] = at; }                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
    if (sg != 0) return;                                                                                           \
    best = vals[lane];                                                                                             \
    at = idxs[lane];                                                                                               \
    for (ushort d = 16; d > 0; d >>= 1) {                                                                          \
      const float ov = simd_shuffle_xor(best, d);                                                                  \
      const uint oi = simd_shuffle_xor(at, d);                                                                     \
      if (tf_bf16_argmax_better(ov, oi, best, at)) { best = ov; at = oi; }                                         \
    }                                                                                                              \
    if (lane == 0) ids[0] = at;                                                                                    \
  }
