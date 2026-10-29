// Packed row-major NVFP4: E2M1 nibbles, unsigned E4M3 block16 scales, FP32 projection globals.
#include "d3_common.h"

struct Fp4Projection { device const uchar* w; device const uchar* s; device const float* global; };
struct Fp4Expert { Fp4Projection g, u, d; };
struct Fp4Args { uint K, N, prompt, cap, slots; };

inline float fp4_value(uchar code) {
  const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (code & 8) ? -values[code & 7] : values[code & 7];
}
inline float fp4_scale(uchar code) {
  const uint exp = (code >> 3) & 15, mant = code & 7;
  return exp ? ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7) : float(mant) / 512.0f;
}
inline float fp4_row(Fp4Projection w, uint n, uint K, device const bf16* x, uint lane) {
  float acc = 0.0f;
  for (uint k = lane * 16; k < K; k += 512) {
    float block = 0.0f;
    for (uint j = 0; j < 16; ++j) {
      const uchar b = w.w[(ulong)n * (K / 2) + (k + j) / 2];
      const uchar code = (j & 1) ? (b >> 4) : (b & 15);
      block = fma(fp4_value(code), b2f(x[k + j]), block);
    }
    acc = fma(block, fp4_scale(w.s[(ulong)n * (K / 16) + k / 16]), acc);
  }
  return simd_sum(acc) * w.global[0];
}

[[max_total_threads_per_threadgroup(128)]] kernel void d3_nvfp4_up(
    device const bf16* x [[buffer(0)]], device const Fp4Expert* table [[buffer(1)]],
    device const int* picks [[buffer(2)]], device const int* counts [[buffer(3)]],
    device const int* members [[buffer(4)]], device bf16* out [[buffer(5)]],
    constant Fp4Args& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4 + sg;
  const int e = a.prompt ? int(tg.y) : picks[tg.y];
  const int cnt = a.prompt ? counts[e] : 1;
  if (!cnt) return;
  const Fp4Expert w = table[e];
  for (int j = 0; j < cnt; ++j) {
    const uint pair = a.prompt ? uint(members[(ulong)e * a.cap + j]) : tg.y;
    device const bf16* row = x + (ulong)(pair / a.slots) * a.K;
    const float g = rb(fp4_row(w.g, n, a.K, row, lane));
    const float u = rb(fp4_row(w.u, n, a.K, row, lane));
    if (lane == 0) out[(ulong)pair * a.N + n] = f2b(g / (1.0f + precise::exp(-g)) * u);
  }
}

[[max_total_threads_per_threadgroup(128)]] kernel void d3_nvfp4_down(
    device const bf16* x [[buffer(0)]], device const Fp4Expert* table [[buffer(1)]],
    device const int* picks [[buffer(2)]], device const int* counts [[buffer(3)]],
    device const int* members [[buffer(4)]], device float* out [[buffer(5)]],
    constant Fp4Args& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4 + sg;
  const int e = a.prompt ? int(tg.y) : picks[tg.y];
  const int cnt = a.prompt ? counts[e] : 1;
  if (!cnt) return;
  const Fp4Expert w = table[e];
  for (int j = 0; j < cnt; ++j) {
    const uint pair = a.prompt ? uint(members[(ulong)e * a.cap + j]) : tg.y;
    const float v = fp4_row(w.d, n, a.K, x + (ulong)pair * a.K, lane);
    if (lane == 0) out[(ulong)pair * a.N + n] = v;
  }
}

struct CompactRouterArgs { uint ss, off; };
// Retained original IDs compete before softmax/top8; only the selected IDs are translated to compact slots.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_router_compact(
    device const bf16* sp [[buffer(0)]], device int* picks [[buffer(1)]],
    device float* weights [[buffer(2)]], device const int* slots [[buffer(3)]],
    constant CompactRouterArgs& a [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
    uint e [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32], wv[8], best_v[8];
  threadgroup int wi[8], best_i[8];
  const bool kept = slots[e] >= 0;
  const float lg = kept ? b2f(sp[(ulong)t * a.ss + a.off + e]) : -INFINITY;
  const float m = group_max(lg, scratch, sg, lane, 8);
  const float p = precise::exp(lg - m);
  const float sum = group_sum(p, scratch + 8, sg, lane, 8);
  // Poisoned probabilities must propagate NaNs without indexing a dropped expert or selecting one twice.
  const bool poisoned = !isfinite(sum) || sum <= 0.0f;
  float mine = kept ? (poisoned ? 0.0f : p / sum) : -1.0f;
  for (int k = 0; k < 8; ++k) {
    float v = mine;
    int idx = int(e);
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(v, o);
      const int oi = simd_shuffle_xor(idx, o);
      if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { wv[sg] = v; wi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
      float bv = wv[0];
      int bi = wi[0];
      for (int j = 1; j < 8; ++j)
        if (wv[j] > bv || (wv[j] == bv && wi[j] < bi)) { bv = wv[j]; bi = wi[j]; }
      best_v[k] = bv;
      best_i[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(e) == best_i[k]) mine = -1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (e < 8) {
    float total = 0.0f;
    for (int j = 0; j < 8; ++j) total += best_v[j];
    picks[t * 8 + e] = slots[best_i[e]];
    weights[t * 8 + e] = poisoned ? NAN : best_v[e] / total;
  }
}

// CUDA drama_moe_combine_f32: round each weighted pair, sum, then fuse the FP32 shared gate and add residual.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_nvfp4_combine(
    device bf16* residual [[buffer(0)]], device const float* down [[buffer(1)]],
    device const float* weights [[buffer(2)]], device const bf16* shared [[buffer(3)]],
    device const bf16* x [[buffer(4)]], device const bf16* wg [[buffer(5)]],
    constant uint& D [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  float part = 0.0f;
  for (uint i = t; i < D; i += 256) part += b2f(x[(ulong)row * D + i]) * b2f(wg[i]);
  const float gate = d3_sigmoid(group_sum(part, scratch, sg, lane, 8));
  for (uint i = t; i < D; i += 256) {
    float sum = 0.0f;
    for (uint slot = 0; slot < 8; ++slot)
      sum += rb(weights[row * 8 + slot] * down[((ulong)row * 8 + slot) * D + i]);
    const float moe = rb(gate * b2f(shared[(ulong)row * D + i]) + rb(sum));
    residual[(ulong)row * D + i] = f2b(b2f(residual[(ulong)row * D + i]) + moe);
  }
}
