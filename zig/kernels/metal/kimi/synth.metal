// Test-only kernels: hash-made synthetic tensors (tools/zig/k3_synth.py makes the same values) and MXFP4 decode.
#include "kimi_common.h"

// kind: 0 k 2^-(e0+j), 1 = 1 + k 2^-7, 2 bytes, 3 E8M0 127-e0-j (j two hash bits).
struct FillArgs {
  uint seed, count, kind, e0;
};

constant constexpr uint FILL_ONE = 1, FILL_E8 = 3;

inline uint k3_mix(uint seed, uint i) {
  uint x = i ^ (seed * 0x9E3779B9u);
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}

inline float k3_synth_value(uint x, uint kind, uint e0) {
  if (kind == FILL_ONE) return 1.0f + (float(x & 63u) - 32.0f) / 128.0f;
  return ldexp(float(int(x & 0xFFu) - 128), -int(e0 + ((x >> 8) & 3u)));
}

kernel void k3_fill_bf16(device bfloat* out [[buffer(0)]], constant FillArgs& a [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < a.count) out[i] = bfloat(k3_synth_value(k3_mix(a.seed, i), a.kind, a.e0));
}

kernel void k3_fill_f32(device float* out [[buffer(0)]], constant FillArgs& a [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < a.count) out[i] = k3_synth_value(k3_mix(a.seed, i), a.kind, a.e0);
}

kernel void k3_fill_u8(device uchar* out [[buffer(0)]], constant FillArgs& a [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint x = k3_mix(a.seed, i);
  out[i] = a.kind == FILL_E8 ? uchar(127u - a.e0 - ((x >> 8) & 3u)) : uchar(x & 0xFFu);
}

// Dense fp32 of an MXFP4 matrix [N, K] through the kernels' own decode, for a bit-exact check.
kernel void k3_fp4_dequant(device const uchar* packed [[buffer(0)]], device const uchar* scales [[buffer(1)]],
                           device float* out [[buffer(2)]], constant uint& K [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.y, b = gid.x;
  if (b >= K / 2) return;
  const float2 v = k3_fp4_pair(packed[n * (K / 2) + b]) * k3_e8m0(scales[n * (K / 32) + b / 16]);
  out[n * K + 2 * b] = v.x;
  out[n * K + 2 * b + 1] = v.y;
}

// Simdgroup MMA throughput: 16 independent bf16 x bf16 -> fp32 accumulators, `steps` rounds, one store a simdgroup.
kernel void k3_mma_peak(device float* out [[buffer(0)]], constant uint& steps [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  simdgroup_bfloat8x8 a = simdgroup_bfloat8x8(bfloat(1.0f / 1024.0f)), b = simdgroup_bfloat8x8(bfloat(1.0f / 1024.0f));
  simdgroup_float8x8 acc[16];
  for (int i = 0; i < 16; ++i) acc[i] = simdgroup_float8x8(float(i));
  for (uint s = 0; s < steps; ++s)
    for (int i = 0; i < 16; ++i) simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
  for (int i = 1; i < 16; ++i) acc[0].thread_elements() += acc[i].thread_elements();
  simdgroup_store(acc[0], out + (tg * 8 + sg) * 64, 8);
}
