// DSpark's released BF16 matrices on the GPU tensor units; compiled only when Device.tensorUnits() is true.
#include "../nax.h"
using namespace metal;
using namespace tfp;

struct lfm2_bf16_nax_args { uint rows, n, k; };

// A simdgroup owns 16x32 outputs. No weight repack, shared-memory staging, or intermediate rounding.
template <bool FULL>
inline void lfm2_bf16_nax_product(const device bfloat* x, const device bfloat* w, device bfloat* y,
                                 int rows, int cols, int n, int k, short2 home) {
  frag<float> lo = frag<float>(0), hi = frag<float>(0);
  int at = 0;
#pragma clang loop unroll(disable)
  for (; at + 16 <= k; at += 16) {
    frag<bfloat> a, b0, b1;
    if (FULL) {
      frag_get(a, x, k, 0, at, home);
      frag_get_t(b0, w, k, 0, at, home);
      frag_get_t(b1, w, k, 16, at, home);
    } else {
      frag_get_in(a, x, k, 0, at, home, rows, k);
      frag_get_t_in(b0, w, k, 0, at, home, cols, k);
      frag_get_t_in(b1, w, k, 16, at, home, cols, k);
    }
    mma_16x32<false, true>(lo, hi, a, b0, b1);
  }
  if (at < k) {
    frag<bfloat> a, b0, b1;
    frag_get_in(a, x, k, 0, at, home, rows, k);
    frag_get_t_in(b0, w, k, 0, at, home, cols, k);
    frag_get_t_in(b1, w, k, 16, at, home, cols, k);
    mma_16x32<false, true>(lo, hi, a, b0, b1);
  }
  if (FULL) {
    frag_put(lo, y, n, 0, 0, home);
    frag_put(hi, y, n, 0, 16, home);
  } else {
    frag_put_in(lo, y, n, 0, 0, home, rows, cols);
    frag_put_in(hi, y, n, 0, 16, home, rows, cols);
  }
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void lfm2_bf16_nax(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant lfm2_bf16_nax_args& p [[buffer(2)]], device bfloat* y [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.y * 16, col = tg.x * 128 + sg * 32;
  if (row >= p.rows || col >= p.n) return;
  const int rows = int(min(16u, p.rows - row)), cols = int(min(32u, p.n - col));
  x += size_t(row) * p.k;
  w += size_t(col) * p.k;
  y += size_t(row) * p.n + col;
  const short2 home = frag_home(ushort(lane));
  if (rows == 16 && cols == 32)
    lfm2_bf16_nax_product<true>(x, w, y, rows, cols, int(p.n), int(p.k), home);
  else
    lfm2_bf16_nax_product<false>(x, w, y, rows, cols, int(p.n), int(p.k), home);
}
