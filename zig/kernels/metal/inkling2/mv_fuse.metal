// Multi-row (1..8 rows/TG) fused GEMVs sharing x, one launch. Coalesced bfloat4 loads, bit-exact across rows.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Ik2Fuse {
  uint k, rows, count, pad;
  uint n0, n1, n2, n3, n4, n5, n6, n7;
};

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_mv_fuse(
    const device bfloat* X [[buffer(0)]],
    const device bfloat* W0 [[buffer(1)]], const device bfloat* W1 [[buffer(2)]],
    const device bfloat* W2 [[buffer(3)]], const device bfloat* W3 [[buffer(4)]],
    const device bfloat* W4 [[buffer(5)]], const device bfloat* W5 [[buffer(6)]],
    const device bfloat* W6 [[buffer(7)]], const device bfloat* W7 [[buffer(8)]],
    device bfloat* Y0 [[buffer(9)]], device bfloat* Y1 [[buffer(10)]],
    device bfloat* Y2 [[buffer(11)]], device bfloat* Y3 [[buffer(12)]],
    device bfloat* Y4 [[buffer(13)]], device bfloat* Y5 [[buffer(14)]],
    device bfloat* Y6 [[buffer(15)]], device bfloat* Y7 [[buffer(16)]],
    constant Ik2Fuse& a [[buffer(17)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint ns[8] = {a.n0, a.n1, a.n2, a.n3, a.n4, a.n5, a.n6, a.n7};
  const device bfloat* W[8] = {W0, W1, W2, W3, W4, W5, W6, W7};
  device bfloat* Y[8] = {Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7};
  uint col = tg.x, which = 0;
  for (uint m = 0; m < a.count; ++m) {
    if (col < ns[m]) {
      which = m;
      break;
    }
    col -= ns[m];
  }
  const uint r0 = tg.y * 8u;
  if (r0 >= a.rows) return;
  const uint nr = min(8u, a.rows - r0);
  const uint k = a.k;
  const uint n_out = ns[which];
  const device bfloat* Wt = W[which];
  const device bfloat* w_row = Wt + ulong(col) * ulong(k);
  device bfloat* Yt = Y[which];

  const bool vec4_ok = ((k & 3u) == 0u) && ((ulong(X) & 7ul) == 0ul) && ((ulong(Wt) & 7ul) == 0ul);
  if (vec4_ok) {
    if (nr == 1u) {
      const device bfloat* xr = X + ulong(r0) * ulong(k);
      float4 acc = float4(0.0f);
      uint i = lane * 4u;
      for (; i + 128u * 3u < k; i += 128u * 4u) {
        const float4 w0 = float4(*(const device bfloat4*)(w_row + i + 0u));
        const float4 x0 = float4(*(const device bfloat4*)(xr + i + 0u));
        const float4 w1 = float4(*(const device bfloat4*)(w_row + i + 128u));
        const float4 x1 = float4(*(const device bfloat4*)(xr + i + 128u));
        const float4 w2 = float4(*(const device bfloat4*)(w_row + i + 256u));
        const float4 x2 = float4(*(const device bfloat4*)(xr + i + 256u));
        const float4 w3 = float4(*(const device bfloat4*)(w_row + i + 384u));
        const float4 x3 = float4(*(const device bfloat4*)(xr + i + 384u));
        acc = fma(x0, w0, acc);
        acc = fma(x1, w1, acc);
        acc = fma(x2, w2, acc);
        acc = fma(x3, w3, acc);
      }
      for (; i < k; i += 128u) {
        const float4 w0 = float4(*(const device bfloat4*)(w_row + i));
        const float4 x0 = float4(*(const device bfloat4*)(xr + i));
        acc = fma(x0, w0, acc);
      }
      float s = (acc.x + acc.y) + (acc.z + acc.w);
      for (ushort m = 1; m < 32; m *= 2) s += simd_shuffle_xor(s, m);
      if (lane == 0) Yt[ulong(r0) * ulong(n_out) + col] = bfloat(s);
    } else {
      float4 acc[8];
      for (uint r = 0; r < nr; ++r) acc[r] = float4(0.0f);
      uint i = lane * 4u;
      for (; i + 128u * 3u < k; i += 128u * 4u) {
        const float4 w0 = float4(*(const device bfloat4*)(w_row + i + 0u));
        const float4 w1 = float4(*(const device bfloat4*)(w_row + i + 128u));
        const float4 w2 = float4(*(const device bfloat4*)(w_row + i + 256u));
        const float4 w3 = float4(*(const device bfloat4*)(w_row + i + 384u));
        for (uint r = 0; r < nr; ++r) {
          const device bfloat* xr = X + ulong(r0 + r) * ulong(k) + i;
          acc[r] = fma(float4(*(const device bfloat4*)(xr + 0u)), w0, acc[r]);
          acc[r] = fma(float4(*(const device bfloat4*)(xr + 128u)), w1, acc[r]);
          acc[r] = fma(float4(*(const device bfloat4*)(xr + 256u)), w2, acc[r]);
          acc[r] = fma(float4(*(const device bfloat4*)(xr + 384u)), w3, acc[r]);
        }
      }
      for (; i < k; i += 128u) {
        const float4 w0 = float4(*(const device bfloat4*)(w_row + i));
        for (uint r = 0; r < nr; ++r) {
          const device bfloat* xr = X + ulong(r0 + r) * ulong(k) + i;
          acc[r] = fma(float4(*(const device bfloat4*)(xr)), w0, acc[r]);
        }
      }
      for (uint r = 0; r < nr; ++r) {
        float s = (acc[r].x + acc[r].y) + (acc[r].z + acc[r].w);
        for (ushort m = 1; m < 32; m *= 2) s += simd_shuffle_xor(s, m);
        if (lane == 0) Yt[ulong(r0 + r) * ulong(n_out) + col] = bfloat(s);
      }
    }
    return;
  }

  float acc[8];
  for (uint r = 0; r < nr; ++r) acc[r] = 0.0f;
  for (uint i = lane; i < k; i += 32u) {
    const float wv = float(w_row[i]);
    for (uint r = 0; r < nr; ++r) {
      acc[r] = fma(float(X[ulong(r0 + r) * ulong(k) + i]), wv, acc[r]);
    }
  }
  for (uint r = 0; r < nr; ++r) {
    float s = acc[r];
    for (ushort m = 1; m < 32; m *= 2) s += simd_shuffle_xor(s, m);
    if (lane == 0) Yt[ulong(r0 + r) * ulong(n_out) + col] = bfloat(s);
  }
}
