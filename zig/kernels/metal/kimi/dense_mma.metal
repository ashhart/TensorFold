// dense.metal's bits for many rows on simdgroup MMA: a threadgroup runs one slice's chain (8 fmas an MMA, k order).
#include "kimi_common.h"

// ls: K slices in each weight and X row (8 for a whole row); slice0: the first slice this launch computes.
struct RowsArgs {
  uint K, N, rows, x_stride, y_stride, w_stride, ls, slice0;
  float beta, lin;
};

constant constexpr uint LD = 40;

// Lane l's two elements of an 8x8 accumulator: row fm, columns fn and fn + 1 (measured on the GPU).
inline uint2 k3_frag_at(uint lane) {
  return uint2((lane >> 4) * 4 + ((lane & 7) >> 1), ((lane >> 3) & 1) * 4 + (lane & 1) * 2);
}

// Blocks 4q..4q+3 of slice s: 64 weight rows (gate then up for a GLU) and 32 X rows, zero past the round's rows.
inline void k3_slice_stage(threadgroup bfloat* ws, threadgroup bfloat* xs, device const bfloat* W0,
                           device const bfloat* W1, device const bfloat* X, constant RowsArgs& a, uint n0, uint r0,
                           uint s, uint q, uint tid) {
  const uint ks = a.K / a.ls;
  for (uint i = tid; i < 64 * 4; i += 128) {
    const uint row = i / 4, b = i % 4, t = 4 * q + b;
    device const bfloat* w = W1 == nullptr ? W0 + (n0 + row) * a.w_stride : (row < 32 ? W0 + (n0 + row) * a.w_stride : W1 + (n0 + row - 32) * a.w_stride);
    const device uint2* src = (const device uint2*)(w + (a.ls * t + s) * 8);
    *(threadgroup uint4*)(ws + row * LD + 8 * b) = uint4(src[0], src[1]);
  }
  const uint row = tid / 4, b = tid % 4, t = 4 * q + b;
  uint4 xv = uint4(0);
  if (r0 + row < a.rows) xv = *(device const uint4*)(X + (r0 + row) * a.x_stride + s * ks + 8 * t);
  *(threadgroup uint4*)(xs + row * LD + 8 * b) = xv;
}

// P[s][r][n] = slice s's chain for 64 weight rows x 32 rows (tg = slice, weight tile, row tile); 2x2 simdgroups.
kernel void k3_slice_mma(device const bfloat* X [[buffer(0)]], device const bfloat* W0 [[buffer(1)]],
                         device const bfloat* W1 [[buffer(2)]], device float* P [[buffer(3)]],
                         constant RowsArgs& a [[buffer(4)]], constant uint& glu [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat ws[2][64 * LD], xs[2][32 * LD];
  const uint s = a.slice0 + tg.x, n0 = tg.y * (glu ? 32 : 64), r0 = tg.z * 32, sn = sg / 2, sr = sg % 2;
  device const bfloat* w1 = glu ? W1 : nullptr;
  simdgroup_float8x8 acc[4][2];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  const uint steps = a.K / a.ls / 32;
  k3_slice_stage(ws[0], xs[0], W0, w1, X, a, n0, r0, s, 0, tid);
  for (uint q = 0; q < steps; ++q) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q + 1 < steps) k3_slice_stage(ws[(q + 1) & 1], xs[(q + 1) & 1], W0, w1, X, a, n0, r0, s, q + 1, tid);
    for (uint kk = 0; kk < 4; ++kk) {
      simdgroup_bfloat8x8 wa[4], xb[2];
      for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], ws[q & 1] + (32 * sn + 8 * i) * LD + 8 * kk, LD);
      for (int j = 0; j < 2; ++j) simdgroup_load(xb[j], xs[q & 1] + (16 * sr + 8 * j) * LD + 8 * kk, LD, ulong2(0, 0), true);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], wa[i], xb[j], acc[i][j]);
    }
  }
  const uint2 at = k3_frag_at(lane);
  const uint width = glu ? 2 * a.N : a.N;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) {
      const uint wr = 32 * sn + 8 * i + at.x, r = r0 + 16 * sr + 8 * j + at.y;
      const uint col = glu ? (wr < 32 ? n0 + wr : a.N + n0 + wr - 32) : n0 + wr;
      const auto e = acc[i][j].thread_elements();
      for (uint d = 0; d < 2; ++d)
        if (r + d < a.rows) P[(tg.x * a.rows + r + d) * width + col] = e[d];
    }
}

inline float k3_ptree(device const float* P, uint stride, uint at) {
  float c[8];
  for (uint s = 0; s < 8; ++s) c[s] = P[s * stride + at];
  return ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]));
}

// y[r, n] = the slices' tree of P (bf16 or fp32), or for a GLU situ(bf16(gate tree), bf16(up tree)).
template <typename OutT>
kernel void k3_slice_out(device const float* P [[buffer(0)]], device OutT* Y [[buffer(1)]],
                         constant RowsArgs& a [[buffer(2)]], constant uint& glu [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x, r = gid.y;
  if (n >= a.N || r >= a.rows) return;
  const uint width = glu ? 2 * a.N : a.N, stride = a.rows * width;
  const float v = k3_ptree(P, stride, r * width + n);
  if (!glu) {
    Y[r * a.y_stride + n] = OutT(v);
    return;
  }
  const float u = k3_ptree(P, stride, r * width + a.N + n);
  Y[r * a.y_stride + n] = OutT(bfloat(k3_situ(float(bfloat(v)), float(bfloat(u)), a.beta, a.lin)));
}

template [[host_name("k3_slice_out_bf16")]] kernel void k3_slice_out<bfloat>(device const float*, device bfloat*,
    constant RowsArgs&, constant uint&, uint2);
template [[host_name("k3_slice_out_f32")]] kernel void k3_slice_out<float>(device const float*, device float*,
    constant RowsArgs&, constant uint&, uint2);
