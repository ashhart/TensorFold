// M5 tensor-unit fragments shared by every NAX kernel (ops and prefill): layout, loads, stores and the 16x32x16 op.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace tfp {
using namespace metal;

// Full unroll: register arrays indexed by a loop counter stay in registers only when the loop unrolls.
#define TF_UNROLL _Pragma("clang loop unroll(full)")

// One simdgroup's 16x16 fragment: 8 values a lane.
template <typename T>
using frag = vec<T, 8>;

#ifndef TF_SIMD_FRAGS
// Lane l holds rows home.y and home.y + 8, columns home.x .. home.x + 3 of every fragment (the M5 operand layout).
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// The 16x16 block at (r, c) of a row-major matrix with leading dimension ld, in device or threadgroup memory.
template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + (e & 3)]);
  }
}

// frag_get with zeros outside rows < nr and columns < nc (both relative to p); nothing outside is read.
template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + (e & 3)] = O(f[e]);
  }
}

template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}

// (lo | hi) += a * (b0 | b1): one 16x32x16 multiply-accumulate on the tensor unit, relaxed precision as MLX asks.
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// Element e's column past home.x (a plain index: M5 kernels keep their exact text); its row is home.y + (e >> 2) * 8.
#define TF_COL(e) (e & 3)
// A transposed right operand's block (its rows are the op's columns): on M5 it loads like any other.
#define frag_get_t frag_get
#define frag_get_t_in frag_get_in
#else
// TF_SIMD_FRAGS (before M5, 8x8 simdgroup matrices): lane l holds rows home.y, home.y + 8, columns home.x + {0,1,8,9}.
inline short2 frag_home(ushort l) {
  return short2(short(((l & 8) >> 1) | ((l & 1) << 1)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

#define TF_COL(e) (((e) & 1) + 8 * (((e) >> 1) & 1))

template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + TF_COL(e)]);
  }
}

template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

// The block at (r, c) as a transposed right operand: its rows on the column pattern, its columns on the row pattern.
template <typename T, typename P>
inline void frag_get_t(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(p[(r + home.x + TF_COL(e)) * ld + c + home.y + (e >> 2) * 8]);
  }
}

template <typename T, typename S>
inline void frag_get_t_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.x + TF_COL(e), cc = c + home.y + (e >> 2) * 8;
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + TF_COL(e)] = O(f[e]);
  }
}

template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}

// The op as on M5; right operand and accumulator hold a lane's first-half columns at (e >> 2) * 8 + (e & 3).
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const short j = (e >> 2) * 8 + (e & 3);
    left[e] = a[e];
    right[j] = b0[e];
    right[j + 4] = b1[e];
    acc[j] = lo[e];
    acc[j + 4] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const short j = (e >> 2) * 8 + (e & 3);
    lo[e] = acc[j];
    hi[e] = acc[j + 4];
  }
}
#endif

}  // namespace tfp
