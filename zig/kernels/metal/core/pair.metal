// Two Macs' tensor-parallel exchanges inside a command buffer: post, wait for the peer's half, add in rank order.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint POSTED = 0, GAVE_UP = 16;

// Thread 0 posts exchange x (the host then sends this rank's part) and polls until the peer's part has landed.
inline void pair_post_wait(device atomic_uint* sync, device atomic_uint* flag, uint x, uint t) {
  if (t != 0) return;
  atomic_store_explicit(&sync[POSTED], x, memory_order_relaxed);
  uint polls = 0;
  while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
    if (++polls > 400000000u) {
      atomic_fetch_add_explicit(&sync[GAVE_UP], 1u, memory_order_relaxed);
      return;
    }
  }
}

inline float pair_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

struct PairRows {
  uint width, stride, rank, x;
  float eps, pad0, pad1, pad2;
};

// Row r: h += (rank 0's part + rank 1's part), then y = w * (h / rms(h)); the peer's words read as atomics.
kernel void pair_add_rms(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flag [[buffer(1)]],
                         constant PairRows& a [[buffer(2)]], const device float* mine [[buffer(3)]],
                         device atomic_uint* peer [[buffer(4)]], device float* h [[buffer(5)]],
                         const device bfloat* w [[buffer(6)]], device float* y [[buffer(7)]],
                         uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  pair_post_wait(sync, flag, a.x, t);
  threadgroup_barrier(mem_flags::mem_device);
  device float* x = h + ulong(r) * a.stride;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    const ulong at = ulong(r) * a.width + i;
    const float other = as_type<float>(atomic_load_explicit(&peer[at], memory_order_relaxed));
    const float sum = a.rank == 0 ? mine[at] + other : other + mine[at];
    const float v = x[i] + sum;
    x[i] = v;
    s = fma(v, v, s);
  }
  s = pair_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += 256) y[ulong(r) * a.stride + i] = float(w[i]) * (x[i] * rs);
}

// Row r's argmax over this rank's logits (n of them, from vocabulary id lo): (value, id), the lower id on a tie.
kernel void pair_argmax(const device float* logits [[buffer(0)]], constant uint4& dims [[buffer(1)]],
                        device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const device float* x = logits + ulong(r) * dims.x;
  float best = -INFINITY;
  uint at = 0xFFFFFFFFu;
  for (uint i = t; i < dims.z; i += 1024) {
    const float v = x[i];
    if (v > best) best = v, at = dims.y + i;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  threadgroup float vb[32];
  threadgroup uint va[32];
  if (lane == 0) vb[sg] = best, va[sg] = at;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vb[lane];
  at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  if (lane == 0) out[2 * r] = as_type<uint>(best), out[2 * r + 1] = at;
}

// Post, wait for the peer's halves, then each row's pick: the larger value, the lower id on a tie (one Mac's argmax).
kernel void pair_pick(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flag [[buffer(1)]],
                      constant uint2& rx [[buffer(2)]], const device uint* mine [[buffer(3)]],
                      device atomic_uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]],
                      uint t [[thread_index_in_threadgroup]]) {
  pair_post_wait(sync, flag, rx.y, t);
  if (t != 0) return;
  for (uint r = 0; r < rx.x; ++r) {
    const float vm = as_type<float>(mine[2 * r]);
    const uint am = mine[2 * r + 1];
    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
  }
}
