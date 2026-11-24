// Prompt projections on simdgroup matrices for GPUs without tensor units: sp_matvec's sums, bit for bit.
#include <metal_stdlib>
using namespace metal;

struct Ink2DenseMmaArgs {
  uint rows, n, k, ldy;
};

using ink2_dense_frag = simdgroup_matrix<float, 8, 8>;

// Row tiles ROWS and column tiles COLS of the 16 by 16 tile that hold outputs; simdgroup g keeps lanes 8g .. 8g + 7.
template <uint ROWS, uint COLS>
inline void ink2_dense_mma(const device bfloat* X, const device bfloat* W, constant Ink2DenseMmaArgs& a,
                           device bfloat* Y, uint2 tg, uint g, uint lane, threadgroup float (&part)[4][2][2][8][8]) {
  const uint r0 = tg.y * 16u, n0 = tg.x * 16u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  // The first four columns of an octet go through one MMA and the last four through another.
  const bool low = fn < 4u;
  // A matvec lane adds, per 256 columns, its first four products to its next four, then to its running sum.
  ink2_dense_frag acc[ROWS][COLS][8];
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < ROWS; ++i)
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < COLS; ++j)
      _Pragma("clang loop unroll(full)") for (uint c = 0u; c < 8u; ++c) acc[i][j][c] = ink2_dense_frag(0.0f);
  const device bfloat* x = X + ulong(r0) * a.k + 64u * g;
  const device bfloat* w = W + ulong(n0) * a.k + 64u * g;
  for (uint col = 0u; col < a.k; col += 256u) {
    _Pragma("clang loop unroll(full)") for (uint c = 0u; c < 8u; ++c) {
      simdgroup_matrix<bfloat, 8, 8> fa[ROWS][2], fb[COLS];
      _Pragma("clang loop unroll(full)") for (uint i = 0u; i < ROWS; ++i) {
        simdgroup_matrix<bfloat, 8, 8> v;
        simdgroup_load(v, x + ulong(8u * i) * a.k + col + 8u * c, a.k);
        const bfloat z = bfloat(0.0f);
        fa[i][0].thread_elements()[0] = low ? v.thread_elements()[0] : z;
        fa[i][0].thread_elements()[1] = low ? v.thread_elements()[1] : z;
        fa[i][1].thread_elements()[0] = low ? z : v.thread_elements()[0];
        fa[i][1].thread_elements()[1] = low ? z : v.thread_elements()[1];
      }
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < COLS; ++j)
        simdgroup_load(fb[j], w + ulong(8u * j) * a.k + col + 8u * c, a.k, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)") for (uint i = 0u; i < ROWS; ++i)
        _Pragma("clang loop unroll(full)") for (uint j = 0u; j < COLS; ++j) {
          ink2_dense_frag d0(0.0f), d1(0.0f);
          simdgroup_multiply_accumulate(d0, fa[i][0], fb[j], d0);
          simdgroup_multiply_accumulate(d1, fa[i][1], fb[j], d1);
          acc[i][j][c].thread_elements()[0] += d0.thread_elements()[0] + d1.thread_elements()[0];
          acc[i][j][c].thread_elements()[1] += d0.thread_elements()[1] + d1.thread_elements()[1];
        }
    }
  }
  // The eight chains of the group add adjacent first, as simd_sum's first three levels do.
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < ROWS; ++i)
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < COLS; ++j)
      _Pragma("clang loop unroll(full)") for (uint e = 0u; e < 2u; ++e) {
        float v[8];
        _Pragma("clang loop unroll(full)") for (uint c = 0u; c < 8u; ++c) v[c] = acc[i][j][c].thread_elements()[e];
        _Pragma("clang loop unroll(full)") for (uint s = 1u; s < 8u; s <<= 1u)
          _Pragma("clang loop unroll(full)") for (uint c = 0u; c < 8u; c += 2u * s) v[c] = v[c] + v[c + s];
        part[g][i][j][fm][fn + e] = v[0];
      }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Then the four groups: (0 + 1) + (2 + 3).
  for (uint o = g * 32u + lane; o < ROWS * COLS * 64u; o += 128u) {
    const uint i = o >> (COLS == 2u ? 7u : 6u), j = COLS == 2u ? (o >> 6u) & 1u : 0u, m = (o >> 3u) & 7u, n = o & 7u;
    const float t = (part[0][i][j][m][n] + part[1][i][j][m][n]) + (part[2][i][j][m][n] + part[3][i][j][m][n]);
    const uint row = r0 + i * 8u + m, col = n0 + j * 8u + n;
    if (row < a.rows) Y[ulong(row) * a.ldy + col] = bfloat(t);
  }
}

// Threadgroup (x, y): outputs 16x .. 16x + 15 of rows 16y .. 16y + 15; n is a multiple of 8, k of 256.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_dense_mma(
    const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant Ink2DenseMmaArgs& a [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][2][2][8][8];
  const bool two_rows = a.rows - tg.y * 16u > 8u, two_cols = a.n - tg.x * 16u > 8u;
  if (two_rows) {
    if (two_cols) ink2_dense_mma<2u, 2u>(X, W, a, Y, tg, g, lane, part);
    else ink2_dense_mma<2u, 1u>(X, W, a, Y, tg, g, lane, part);
  } else {
    if (two_cols) ink2_dense_mma<1u, 2u>(X, W, a, Y, tg, g, lane, part);
    else ink2_dense_mma<1u, 1u>(X, W, a, Y, tg, g, lane, part);
  }
}
