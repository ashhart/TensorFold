// One-row GEMVs that share x, one launch. Same lane chains as tf_dsv41_mv r1c1 (U=32), one column a simdgroup.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Ik2Fuse {
  uint k, count, n0, n1, n2, n3, n4, n5, n6, n7;
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
    constant Ik2Fuse& a [[buffer(17)]], uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint ns[8] = {a.n0, a.n1, a.n2, a.n3, a.n4, a.n5, a.n6, a.n7};
  const device bfloat* W[8] = {W0, W1, W2, W3, W4, W5, W6, W7};
  device bfloat* Y[8] = {Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7};
  uint col = tg, which = 0;
  for (uint m = 0; m < a.count; ++m) {
    if (col < ns[m]) {
      which = m;
      break;
    }
    col -= ns[m];
  }
  const device bfloat* Wt = W[which];
  device bfloat* Yt = Y[which];
  const uint k = a.k;
  float acc = 0.0f;
  uint i = lane;
  for (; i + 32u * 31u < k; i += 32u * 32u) {
    float w[32], x[32];
    for (uint u = 0; u < 32u; ++u) {
      w[u] = float(Wt[ulong(col) * k + i + 32u * u]);
      x[u] = float(X[i + 32u * u]);
    }
    for (uint u = 0; u < 32u; ++u) acc = fma(x[u], w[u], acc);
  }
  for (; i < k; i += 32u) acc = fma(float(X[i]), float(Wt[ulong(col) * k + i]), acc);
  for (ushort m = 1; m < 32; m *= 2) acc += simd_shuffle_xor(acc, m);
  if (lane == 0) Yt[col] = bfloat(acc);
}
