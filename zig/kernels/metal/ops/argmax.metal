// Row argmax of bf16 logits in two passes, the index MLX 0.32.3's argmax_bfloat16 gives: first NaN, else lowest max.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// Whether (cv, ci) wins over (bv, bi): a NaN beats numbers, then the larger, then the lower index (an exact order).
inline bool tf_beats(float cv, uint ci, float bv, uint bi) {
  const bool cn = isnan(cv), bn = isnan(bv);
  if (cn || bn) return cn && (!bn || ci < bi);
  return cv > bv || (cv == bv && ci < bi);
}

inline void tf_keep(thread float& bv, thread uint& bi, float cv, uint ci) {
  if (tf_beats(cv, ci, bv, bi)) {
    bv = cv;
    bi = ci;
  }
}

// tf:kernel tf_argmax_part_bf16 inputs=X,P outputs=PV,PI
[[kernel]] void tf_argmax_part_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device float* PV [[buffer(2)]],
  device uint32_t* PI [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]],
  uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  threadgroup float tv[32];
  threadgroup uint ti[32];
  const int width = P[0], span = P[1];      // threadgroup s owns [s span, (s + 1) span), 8 a thread in index order
  const uint t = thread_position_in_threadgroup.x, threads = threads_per_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const uint split = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
  const int lo = int(split) * span, hi = min(width, lo + span);
  const device bfloat* r = X + long(row) * width;
  float bv = -INFINITY;
  uint bi = 0;
  for (int base = lo + int(t) * 8; base < hi; base += int(threads) * 8) {
    if (base + 8 <= hi) {
      const vec<bfloat, 8> q = *(const device vec<bfloat, 8>*)(r + base);
      for (int j = 0; j < 8; ++j) {
        const float v = q[j];
        if (v > bv || (isnan(v) && !isnan(bv))) {
          bv = v;
          bi = uint(base + j);
        }
      }
    } else {
      for (int j = 0; base + j < hi; ++j) tf_keep(bv, bi, float(r[base + j]), uint(base + j));
    }
  }
  for (ushort off = 16; off > 0; off /= 2) tf_keep(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0) {
    tv[sg] = bv;
    ti[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  bv = -INFINITY;
  bi = 0;
  if (lane < (threads + 31) / 32) {
    bv = tv[lane];
    bi = ti[lane];
  }
  for (ushort off = 16; off > 0; off /= 2) tf_keep(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0) {
    PV[row * threadgroups_per_grid.x + split] = bv;
    PI[row * threadgroups_per_grid.x + split] = bi;
  }
}

// tf:kernel tf_argmax_merge inputs=PV,PI,P outputs=IDX
[[kernel]] void tf_argmax_merge(
  const device float* PV [[buffer(0)]],
  const device uint32_t* PI [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device uint32_t* IDX [[buffer(3)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int spans = P[0];
  const uint lane = thread_index_in_simdgroup, row = threadgroup_position_in_grid.y;
  float bv = -INFINITY;
  uint bi = 0;
  for (int s = int(lane); s < spans; s += 32) tf_keep(bv, bi, PV[row * spans + s], PI[row * spans + s]);
  for (ushort off = 16; off > 0; off /= 2) tf_keep(bv, bi, simd_shuffle_down(bv, off), simd_shuffle_down(bi, off));
  if (lane == 0) IDX[row] = bi;
}
