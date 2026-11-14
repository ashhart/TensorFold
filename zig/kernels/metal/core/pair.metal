// Two Macs' tensor-parallel exchanges: the host publishes an epoch at an event boundary before any peer read.
#include <metal_stdlib>
using namespace metal;

// A terminal wake skips this copy, preserving a send slot the transport may still own.
kernel void pair_stage(const device ulong* ready [[buffer(0)]], constant ulong& previous [[buffer(1)]],
                        const device uint* src [[buffer(2)]], device uint* dst [[buffer(3)]],
                        constant uint& count [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (ready[0] != previous) return;
  if (i < count) dst[i] = src[i];
}

inline float pair_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

struct PairRows {
  uint width, stride, rank, x;
  float eps, pad0, pad1, pad2;
};

// Row r: h += (rank 0's part + rank 1's part), then y = w * (h / rms(h)); terminal wakes keep later work in bounds.
kernel void pair_add_rms(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
                         constant PairRows& a [[buffer(2)]], const device float* mine [[buffer(3)]],
                         device atomic_uint* peer [[buffer(4)]], device float* h [[buffer(5)]],
                         const device bfloat* w [[buffer(6)]], device float* y [[buffer(7)]],
                         uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  device float* x = h + ulong(r) * a.stride;
  if (ready[0] != expected) {
    for (uint i = t; i < a.width; i += 256) x[i] = 0.0f, y[ulong(r) * a.stride + i] = 0.0f;
    return;
  }
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

// After arrival, the larger value wins, lower id on a tie; failure uses token zero for later already-encoded drafts.
kernel void pair_pick(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
                      constant uint& rows [[buffer(2)]], const device uint* mine [[buffer(3)]],
                      device atomic_uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]],
                      uint t [[thread_index_in_threadgroup]]) {
  if (t != 0) return;
  if (ready[0] != expected) {
    for (uint r = 0; r < rows; ++r) picks[r] = 0u;
    return;
  }
  for (uint r = 0; r < rows; ++r) {
    const float vm = as_type<float>(mine[2 * r]);
    const uint am = mine[2 * r + 1];
    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
  }
}
