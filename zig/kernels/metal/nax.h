// M5 tensor-unit fragments shared by every NAX kernel (ops and prefill): layout, loads, stores and the 16x32x16 op.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
#include <metal_simdgroup_matrix>
#endif

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

// frag_get from a threadgroup block in two 8-byte loads: (c + home.x) * 2 and ld * 2 must be multiples of 8.
inline void frag_get_tg(thread frag<bfloat>& f, const threadgroup bfloat* p, int ld, int r, int c, short2 home) {
  const threadgroup bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<bfloat>(*(const threadgroup bfloat4*)q, *(const threadgroup bfloat4*)(q + 8 * ld));
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
#ifdef TF_GLM_M3_FAST
  f[0] = T(q[0]); f[1] = T(q[1]); f[2] = T(q[8]); f[3] = T(q[9]);
  f[4] = T(q[8 * ld]); f[5] = T(q[8 * ld + 1]); f[6] = T(q[8 * ld + 8]); f[7] = T(q[8 * ld + 9]);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + TF_COL(e)]);
  }
#endif
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
#ifdef TF_GLM_M3_FAST
  simdgroup_matrix<A, 8, 8> a00, a01, a10, a11;
  simdgroup_matrix<B, 8, 8> b00, b01, b10, b11, c00, c01, c10, c11;
  simdgroup_matrix<C, 8, 8> d00, d01, d10, d11, e00, e01, e10, e11;
  a00.thread_elements()[0] = a[0]; a00.thread_elements()[1] = a[1];
  a01.thread_elements()[0] = a[2]; a01.thread_elements()[1] = a[3];
  a10.thread_elements()[0] = a[4]; a10.thread_elements()[1] = a[5];
  a11.thread_elements()[0] = a[6]; a11.thread_elements()[1] = a[7];
  b00.thread_elements()[0] = b0[0]; b00.thread_elements()[1] = b0[1];
  b01.thread_elements()[0] = b0[2]; b01.thread_elements()[1] = b0[3];
  b10.thread_elements()[0] = b0[4]; b10.thread_elements()[1] = b0[5];
  b11.thread_elements()[0] = b0[6]; b11.thread_elements()[1] = b0[7];
  c00.thread_elements()[0] = b1[0]; c00.thread_elements()[1] = b1[1];
  c01.thread_elements()[0] = b1[2]; c01.thread_elements()[1] = b1[3];
  c10.thread_elements()[0] = b1[4]; c10.thread_elements()[1] = b1[5];
  c11.thread_elements()[0] = b1[6]; c11.thread_elements()[1] = b1[7];
  d00.thread_elements()[0] = lo[0]; d00.thread_elements()[1] = lo[1];
  d01.thread_elements()[0] = lo[2]; d01.thread_elements()[1] = lo[3];
  d10.thread_elements()[0] = lo[4]; d10.thread_elements()[1] = lo[5];
  d11.thread_elements()[0] = lo[6]; d11.thread_elements()[1] = lo[7];
  e00.thread_elements()[0] = hi[0]; e00.thread_elements()[1] = hi[1];
  e01.thread_elements()[0] = hi[2]; e01.thread_elements()[1] = hi[3];
  e10.thread_elements()[0] = hi[4]; e10.thread_elements()[1] = hi[5];
  e11.thread_elements()[0] = hi[6]; e11.thread_elements()[1] = hi[7];
  simdgroup_multiply_accumulate(d00, a00, b00, d00);
  simdgroup_multiply_accumulate(d01, a00, b01, d01);
  simdgroup_multiply_accumulate(d10, a10, b00, d10);
  simdgroup_multiply_accumulate(d11, a10, b01, d11);
  simdgroup_multiply_accumulate(e00, a00, c00, e00);
  simdgroup_multiply_accumulate(e01, a00, c01, e01);
  simdgroup_multiply_accumulate(e10, a10, c00, e10);
  simdgroup_multiply_accumulate(e11, a10, c01, e11);
  simdgroup_multiply_accumulate(d00, a01, b10, d00);
  simdgroup_multiply_accumulate(d01, a01, b11, d01);
  simdgroup_multiply_accumulate(d10, a11, b10, d10);
  simdgroup_multiply_accumulate(d11, a11, b11, d11);
  simdgroup_multiply_accumulate(e00, a01, c10, e00);
  simdgroup_multiply_accumulate(e01, a01, c11, e01);
  simdgroup_multiply_accumulate(e10, a11, c10, e10);
  simdgroup_multiply_accumulate(e11, a11, c11, e11);
  lo[0] = d00.thread_elements()[0]; lo[1] = d00.thread_elements()[1];
  lo[2] = d01.thread_elements()[0]; lo[3] = d01.thread_elements()[1];
  lo[4] = d10.thread_elements()[0]; lo[5] = d10.thread_elements()[1];
  lo[6] = d11.thread_elements()[0]; lo[7] = d11.thread_elements()[1];
  hi[0] = e00.thread_elements()[0]; hi[1] = e00.thread_elements()[1];
  hi[2] = e01.thread_elements()[0]; hi[3] = e01.thread_elements()[1];
  hi[4] = e10.thread_elements()[0]; hi[5] = e10.thread_elements()[1];
  hi[6] = e11.thread_elements()[0]; hi[7] = e11.thread_elements()[1];
#else
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
#endif
}

#ifdef TF_GLM_M3_FAST
// Half-width 16x16 multiply-accumulate used by GLM's decode index kernel when TF_GLM_M3_FAST is active.
template <typename C, typename A, typename B>
inline void mma_16x16(thread frag<C>& d, thread const frag<A>& a, thread const frag<B>& b) {
  simdgroup_matrix<A, 8, 8> a00, a01, a10, a11;
  simdgroup_matrix<B, 8, 8> b00, b01, b10, b11;
  simdgroup_matrix<C, 8, 8> d00, d01, d10, d11;
  a00.thread_elements()[0] = a[0]; a00.thread_elements()[1] = a[1];
  a01.thread_elements()[0] = a[2]; a01.thread_elements()[1] = a[3];
  a10.thread_elements()[0] = a[4]; a10.thread_elements()[1] = a[5];
  a11.thread_elements()[0] = a[6]; a11.thread_elements()[1] = a[7];
  b00.thread_elements()[0] = b[0]; b00.thread_elements()[1] = b[1];
  b01.thread_elements()[0] = b[2]; b01.thread_elements()[1] = b[3];
  b10.thread_elements()[0] = b[4]; b10.thread_elements()[1] = b[5];
  b11.thread_elements()[0] = b[6]; b11.thread_elements()[1] = b[7];
  d00.thread_elements()[0] = d[0]; d00.thread_elements()[1] = d[1];
  d01.thread_elements()[0] = d[2]; d01.thread_elements()[1] = d[3];
  d10.thread_elements()[0] = d[4]; d10.thread_elements()[1] = d[5];
  d11.thread_elements()[0] = d[6]; d11.thread_elements()[1] = d[7];
  simdgroup_multiply_accumulate(d00, a00, b00, d00);
  simdgroup_multiply_accumulate(d01, a00, b01, d01);
  simdgroup_multiply_accumulate(d10, a10, b00, d10);
  simdgroup_multiply_accumulate(d11, a10, b01, d11);
  simdgroup_multiply_accumulate(d00, a01, b10, d00);
  simdgroup_multiply_accumulate(d01, a01, b11, d01);
  simdgroup_multiply_accumulate(d10, a11, b10, d10);
  simdgroup_multiply_accumulate(d11, a11, b11, d11);
  d[0] = d00.thread_elements()[0]; d[1] = d00.thread_elements()[1];
  d[2] = d01.thread_elements()[0]; d[3] = d01.thread_elements()[1];
  d[4] = d10.thread_elements()[0]; d[5] = d10.thread_elements()[1];
  d[6] = d11.thread_elements()[0]; d[7] = d11.thread_elements()[1];
}
#endif
#endif

// frag_get from a block whose rows home.y and home.y + 8 start at r0 and r1; an unflagged row reads as zero.
template <typename T, typename S>
inline void frag_get_rows(thread frag<T>& f, const device S* r0, const device S* r1, int c, short2 home, bool lo,
                          bool hi) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = ((e >> 2) ? hi : lo) ? T(((e >> 2) ? r1 : r0)[c + home.x + TF_COL(e)]) : T(0);
  }
}

// frag_put to a block whose rows home.y and home.y + 8 start at r0 and r1; only flagged rows are written.
template <typename O>
inline void frag_put_rows(thread const frag<float>& f, device O* r0, device O* r1, int c, short2 home, bool lo,
                          bool hi) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    if ((e >> 2) ? hi : lo) {
      ((e >> 2) ? r1 : r0)[c + home.x + TF_COL(e)] = O(f[e]);
    }
  }
}

}  // namespace tfp
