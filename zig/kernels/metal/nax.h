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

}  // namespace tfp
