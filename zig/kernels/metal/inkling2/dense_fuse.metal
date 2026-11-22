// Fused 6-projection attention-input BF16 matvec, bit-identical to six sp_matvec launches.
#include <metal_stdlib>
using namespace metal;

struct InkMv6Args {
  uint rows, k, nq, nk, nv, nr, ns, ng;
};

template <uint ROWS>
inline void ink2_mv6_t(
    const device bfloat* xq, const device bfloat* xk, const device bfloat* xv,
    const device bfloat* wq, const device bfloat* wk, const device bfloat* wv,
    const device bfloat* wr, const device bfloat* ws, const device bfloat* wg,
    device bfloat* yq, device bfloat* yk, device bfloat* yv,
    device bfloat* yr, device bfloat* ys, device bfloat* yg,
    constant InkMv6Args& a, uint3 tg, uint sg, uint lane) {
  const uint sq = ((a.nq + 7u) >> 3) << 3;
  const uint sk = ((a.nk + 7u) >> 3) << 3;
  const uint sv = ((a.nv + 7u) >> 3) << 3;
  const uint sr = ((a.nr + 7u) >> 3) << 3;
  const uint ss = ((a.ns + 7u) >> 3) << 3;
  const uint sg_g = ((a.ng + 7u) >> 3) << 3;
  const uint b1 = sq;
  const uint b2 = b1 + sk;
  const uint b3 = b2 + sv;
  const uint b4 = b3 + sr;
  const uint b5 = b4 + ss;
  const uint b6 = b5 + sg_g;
  const uint sg_global = tg.x * 8u + sg;
  if (sg_global >= b6) return;

  const device bfloat* X;
  const device bfloat* W;
  device bfloat* Y;
  uint n_rows;
  uint local_row;
  if (sg_global < b1) {
    X = xq; W = wq; Y = yq; n_rows = a.nq; local_row = sg_global;
  } else if (sg_global < b2) {
    X = xk; W = wk; Y = yk; n_rows = a.nk; local_row = sg_global - b1;
  } else if (sg_global < b3) {
    X = xv; W = wv; Y = yv; n_rows = a.nv; local_row = sg_global - b2;
  } else if (sg_global < b4) {
    X = xq; W = wr; Y = yr; n_rows = a.nr; local_row = sg_global - b3;
  } else if (sg_global < b5) {
    X = xq; W = ws; Y = ys; n_rows = a.ns; local_row = sg_global - b4;
  } else {
    X = xq; W = wg; Y = yg; n_rows = a.ng; local_row = sg_global - b5;
  }
  if (local_row >= n_rows) return;

  const device bfloat* w = W + (size_t)local_row * a.k;
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8u * lane; i < a.k; i += 256u) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i));
    const float4 w1 = float4(*(const device bfloat4*)(w + i + 4u));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * a.k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4u)));
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * n_rows + local_row] = bfloat(s);
  }
}

#define TF_INK2_MV6(ROWS, WQ)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mv6_##ROWS(                                   \
      const device bfloat* xq [[buffer(0)]], const device bfloat* xk [[buffer(1)]],                             \
      const device bfloat* xv [[buffer(2)]], const device bfloat* wq [[buffer(3)]],                             \
      const device bfloat* wk [[buffer(4)]], const device bfloat* wv [[buffer(5)]],                             \
      const device bfloat* wr [[buffer(6)]], const device bfloat* ws [[buffer(7)]],                             \
      const device bfloat* wg [[buffer(8)]], device bfloat* yq [[buffer(9)]],                                   \
      device bfloat* yk [[buffer(10)]], device bfloat* yv [[buffer(11)]],                                       \
      device bfloat* yr [[buffer(12)]], device bfloat* ys [[buffer(13)]],                                       \
      device bfloat* yg [[buffer(14)]], constant InkMv6Args& a [[buffer(15)]],                                 \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                    \
      uint lane [[thread_index_in_simdgroup]]) {                                                                \
    ink2_mv6_t<ROWS>(xq, xk, xv, WQ, wk, wv, wr, ws, wg, yq, yk, yv, yr, ys, yg, a, tg, sg, lane);              \
  }

TF_INK2_MV6(1, wq)
TF_INK2_MV6(2, wq)
TF_INK2_MV6(4, wq)
TF_INK2_MV6(8, wq)
TF_INK2_MV6(16, wq)
TF_INK2_MV6(32, wq)
