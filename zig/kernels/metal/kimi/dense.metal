// bf16 projections: contiguous K slice s (0..7) is one fma chain in k order, then ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7)).
#include "kimi_common.h"

// ls: K slices in each weight and X row (8 for a whole row); slice0: the first slice this launch computes.
struct RowsArgs {
  uint K, N, rows, x_stride, y_stride, w_stride, ls, slice0;
  float beta, lin;
};

// Weights interleaved by slice (8 a row): row block (t * 8 + s) holds slice s's elements 8t..8t+7, one line for 8 lanes.
kernel void k3_interleave(device const uint2* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                          constant uint2& kn [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint blocks = kn.x / 8, b = gid.x, n = gid.y;
  if (b >= blocks || n >= kn.y) return;
  const uint t = b / 8, s = b % 8, from = s * (blocks / 8) + t;
  const uint2 lo = src[(n * blocks + from) * 2], hi = src[(n * blocks + from) * 2 + 1];
  dst[n * blocks + b] = uint4(lo, hi);
}

// One row's eight products of a block onto acc, in k order.
inline float k3_block(float acc, uint4 xq, float4 w0, float4 w1) {
  const float4 x0 = k3_bf16x4(xq.xy), x1 = k3_bf16x4(xq.zw);
  for (int j = 0; j < 4; ++j) acc = fma(x0[j], w0[j], acc);
  for (int j = 0; j < 4; ++j) acc = fma(x1[j], w1[j], acc);
  return acc;
}

// The fixed tree over the 8 slice lanes of an output (xor 1, 2, 4); every lane ends with the same bits.
inline float k3_tree8(float v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 4);
  return v;
}

// Y[r, n] = X[r] . W[n] for 1-31 rows: lane 8o + s runs slice s of output o for RB rows a pass.
template <typename OutT, int RB>
kernel void k3_rows(device const bfloat* X [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                    device OutT* Y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint s = lane & 7, n = (tg * nsg + sg) * 4 + (lane >> 3), ks = a.K / 8;
  if ((tg * nsg + sg) * 4 >= a.N) return;
  const bool live = n < a.N;
  device const uint4* wr = (device const uint4*)(W + (live ? n : 0) * a.w_stride) + s;
  for (uint r0 = 0; r0 < a.rows; r0 += RB) {
    float acc[RB];
    for (int r = 0; r < RB; ++r) acc[r] = 0.0f;
    for (uint t = 0; t < ks / 8; ++t) {
      const uint4 wq = wr[8 * t];
      const float4 w0 = k3_bf16x4(wq.xy), w1 = k3_bf16x4(wq.zw);
      for (int r = 0; r < RB; ++r) {
        if (r0 + r >= a.rows) break;
        acc[r] = k3_block(acc[r], *(device const uint4*)(X + (r0 + r) * a.x_stride + s * ks + 8 * t), w0, w1);
      }
    }
    for (int r = 0; r < RB; ++r) {
      const float v = k3_tree8(acc[r]);
      if (s == 0 && live && r0 + r < a.rows) Y[(r0 + r) * a.y_stride + n] = OutT(v);
    }
  }
}

// act[r, n] = bf16(situ(bf16(X[r] . G[n]), bf16(X[r] . U[n]))): the dense MLP's and shared experts' GLU, 1-31 rows.
template <int RB>
kernel void k3_glu(device const bfloat* X [[buffer(0)]], device const bfloat* G [[buffer(1)]],
                   device const bfloat* U [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                   constant RowsArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  const uint s = lane & 7, n = (tg * nsg + sg) * 4 + (lane >> 3), ks = a.K / 8;
  if ((tg * nsg + sg) * 4 >= a.N) return;
  const bool live = n < a.N;
  device const uint4* gr = (device const uint4*)(G + (live ? n : 0) * a.w_stride) + s;
  device const uint4* ur = (device const uint4*)(U + (live ? n : 0) * a.w_stride) + s;
  for (uint r0 = 0; r0 < a.rows; r0 += RB) {
    float ag[RB], au[RB];
    for (int r = 0; r < RB; ++r) ag[r] = au[r] = 0.0f;
    for (uint t = 0; t < ks / 8; ++t) {
      const uint4 gq = gr[8 * t], uq = ur[8 * t];
      const float4 g0 = k3_bf16x4(gq.xy), g1 = k3_bf16x4(gq.zw), u0 = k3_bf16x4(uq.xy), u1 = k3_bf16x4(uq.zw);
      for (int r = 0; r < RB; ++r) {
        if (r0 + r >= a.rows) break;
        const uint4 xq = *(device const uint4*)(X + (r0 + r) * a.x_stride + s * ks + 8 * t);
        ag[r] = k3_block(ag[r], xq, g0, g1);
        au[r] = k3_block(au[r], xq, u0, u1);
      }
    }
    for (int r = 0; r < RB; ++r) {
      const float g = float(bfloat(k3_tree8(ag[r]))), u = float(bfloat(k3_tree8(au[r])));
      if (s == 0 && live && r0 + r < a.rows) Y[(r0 + r) * a.y_stride + n] = bfloat(k3_situ(g, u, a.beta, a.lin));
    }
  }
}

#define K3_ROWS(T, NAME, RB) \
  template [[host_name("k3_rows_" #NAME "_r" #RB)]] kernel void k3_rows<T, RB>( \
      device const bfloat*, device const bfloat*, device T*, constant RowsArgs&, uint, uint, uint, uint);
K3_ROWS(bfloat, bf16, 1)
K3_ROWS(bfloat, bf16, 2)
K3_ROWS(bfloat, bf16, 4)
K3_ROWS(bfloat, bf16, 8)
K3_ROWS(float, f32, 1)
K3_ROWS(float, f32, 2)
K3_ROWS(float, f32, 4)
K3_ROWS(float, f32, 8)

#define K3_GLU(RB) \
  template [[host_name("k3_glu_r" #RB)]] kernel void k3_glu<RB>(device const bfloat*, device const bfloat*, \
      device const bfloat*, device bfloat*, constant RowsArgs&, uint, uint, uint, uint);
K3_GLU(1)
K3_GLU(2)
K3_GLU(4)
K3_GLU(8)
