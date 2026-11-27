// Prompt projections via register-tiled bf16 outer products: sp_matvec's fp32 sums, bit for bit.
#include <metal_stdlib>
using namespace metal;

struct Ink2DenseTiledArgs {
  uint rows, n, k, ldy;
};

// 8 simdgroups compute RM (8/16) x 16 outputs (2 cols/sg), 8 K-elems/lane per 256-step matching bf16_matvec.
template <uint RM, bool FULL_ROWS>
inline void ink2_dense_tiled(const device bfloat* X, const device bfloat* W, constant Ink2DenseTiledArgs& a,
                             device bfloat* Y, uint r0, uint n0, uint g, uint lane,
                             threadgroup bfloat4 (&x_tile)[16][2][32]) {
  const uint c0 = n0 + 2u * g;
  const bool active = c0 < a.n;
  const uint c1 = active ? min(c0 + 1u, a.n - 1u) : 0u;
  float acc0[RM], acc1[RM];
  _Pragma("clang loop unroll(full)") for (uint r = 0u; r < RM; ++r) {
    acc0[r] = 0.0f;
    acc1[r] = 0.0f;
  }
  const uint r_a = RM == 16u
                       ? (FULL_ROWS ? (r0 + 2u * g) : min(r0 + 2u * g, a.rows - 1u))
                       : (FULL_ROWS ? (r0 + g) : min(r0 + g, a.rows - 1u));
  const uint r_b = RM == 16u ? (FULL_ROWS ? (r0 + 2u * g + 1u) : min(r0 + 2u * g + 1u, a.rows - 1u)) : 0u;
  const device bfloat* xa_base = X + ulong(r_a) * a.k + 8u * lane;
  const device bfloat* xb_base = X + ulong(r_b) * a.k + 8u * lane;
  const device bfloat* w0_base = W + ulong(active ? c0 : 0u) * a.k + 8u * lane;
  const device bfloat* w1_base = W + ulong(c1) * a.k + 8u * lane;
  for (uint col = 0u; col < a.k; col += 256u) {
    if (col > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
    // All 8 simdgroups cooperatively stage RM x 256 bf16s of X into conflict-free threadgroup lanes.
    const device bfloat* xa = xa_base + col;
    if (RM == 16u) {
      const device bfloat* xb = FULL_ROWS ? (xa + a.k) : (xb_base + col);
      x_tile[2u * g][0][lane] = *(const device bfloat4*)xa;
      x_tile[2u * g][1][lane] = *(const device bfloat4*)(xa + 4u);
      x_tile[2u * g + 1u][0][lane] = *(const device bfloat4*)xb;
      x_tile[2u * g + 1u][1][lane] = *(const device bfloat4*)(xb + 4u);
    } else {
      x_tile[g][0][lane] = *(const device bfloat4*)xa;
      x_tile[g][1][lane] = *(const device bfloat4*)(xa + 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const device bfloat* w0_ptr = w0_base + col;
      const device bfloat* w1_ptr = w1_base + col;
      const float4 w0_lo = float4(*(const device bfloat4*)w0_ptr);
      const float4 w0_hi = float4(*(const device bfloat4*)(w0_ptr + 4u));
      const float4 w1_lo = float4(*(const device bfloat4*)w1_ptr);
      const float4 w1_hi = float4(*(const device bfloat4*)(w1_ptr + 4u));
      _Pragma("clang loop unroll(full)") for (uint r = 0u; r < RM; ++r) {
        const float4 x0 = float4(x_tile[r][0][lane]);
        const float4 x1 = float4(x_tile[r][1][lane]);
        acc0[r] += dot(w0_lo, x0) + dot(w0_hi, x1);
        acc1[r] += dot(w1_lo, x0) + dot(w1_hi, x1);
      }
    }
  }
  if (!active) return;
  _Pragma("clang loop unroll(full)") for (uint r = 0u; r < RM; ++r) {
    const float s0 = simd_sum(acc0[r]);
    const float s1 = simd_sum(acc1[r]);
    if (lane == 0u && (FULL_ROWS || r0 + r < a.rows)) {
      device bfloat* y = Y + ulong(r0 + r) * a.ldy + c0;
      y[0] = bfloat(s0);
      if (c0 + 1u < a.n) y[1] = bfloat(s1);
    }
  }
}

// TG (x, y): rows 16x..16x+15, outputs 16y..16y+15 (n%8==0, k%256==0); tg.x along rows keeps W strips hot in L2.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_dense_tiled(
    const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant Ink2DenseTiledArgs& a [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat4 x_tile[16][2][32];
  const uint r0 = tg.x * 16u, n0 = tg.y * 16u;
  if (r0 >= a.rows || n0 >= a.n) return;
  const uint left = a.rows - r0;
  if (left >= 16u) ink2_dense_tiled<16u, true>(X, W, a, Y, r0, n0, g, lane, x_tile);
  else if (left > 8u) ink2_dense_tiled<16u, false>(X, W, a, Y, r0, n0, g, lane, x_tile);
  else if (left == 8u) ink2_dense_tiled<8u, true>(X, W, a, Y, r0, n0, g, lane, x_tile);
  else ink2_dense_tiled<8u, false>(X, W, a, Y, r0, n0, g, lane, x_tile);
}
