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
#ifdef TF_NAX_STRICT_FP32
#define TF_MMA_NAME mma_16x32_cut
#else
#define TF_MMA_NAME mma_16x32
#endif
template <bool TA, bool TB, typename C, typename A, typename B, bool EXACT_B = false>
inline void TF_MMA_NAME(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
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
#undef TF_MMA_NAME

inline float3 split_float_operand(float x) {
  const float high = as_type<float>(as_type<uint>(x) & 0xffffe000u);
  const float rest = x - high;
  const float middle = as_type<float>(as_type<uint>(rest) & 0xffffe000u);
  return float3(high, middle, rest - middle);
}

#ifdef TF_NAX_STRICT_FP32
inline bool float_operand_needs_scalar(float value) {
  const uint bits = as_type<uint>(value) & 0x7fffffffu;
  return bits >= 0x7f800000u || (bits != 0 && bits < 0x0c000000u);
}

inline ulong f32_align(ulong value, int shift) {
  if (value == 0) return 0;
  if (shift >= 0) return value << uint(shift);
  const uint right = uint(-shift);
  if (right >= 64) return 1;
  return (value >> right) | ulong((value & ((ulong(1) << right) - 1)) != 0);
}

inline ulong f32_round(ulong value, int shift) {
  if (shift <= 0) return value << uint(-shift);
  if (shift > 64) return 0;
  if (shift == 64) return ulong(value > (ulong(1) << 63));
  const ulong high = value >> uint(shift), halfway = ulong(1) << uint(shift - 1);
  const ulong low = value & ((ulong(1) << uint(shift)) - 1);
  return high + ulong(low > halfway || (low == halfway && (high & 1) != 0));
}

// Metal's scalar unit flushes subnormals; round the exact integer product and addend on this rare path.
inline uint f32_fma_bits(uint x, uint y, uint c) {
  const uint ax = x & 0x7fffffffu, ay = y & 0x7fffffffu, ac = c & 0x7fffffffu;
  const uint sign = (x ^ y) & 0x80000000u, cs = c & 0x80000000u;
  if (ax > 0x7f800000u || ay > 0x7f800000u || ac > 0x7f800000u) return 0x7fc00000u;
  if (ax == 0x7f800000u || ay == 0x7f800000u) {
    if (ax == 0 || ay == 0 || (ac == 0x7f800000u && sign != cs)) return 0x7fc00000u;
    return sign | 0x7f800000u;
  }
  if (ac == 0x7f800000u) return c;
  if (ax == 0 || ay == 0) return ac != 0 ? c : (sign & cs);
  const uint ex = ax >> 23, ey = ay >> 23, ec = ac >> 23;
  const ulong mx = (ax & 0x7fffffu) | (ex != 0 ? 0x800000u : 0u);
  const ulong my = (ay & 0x7fffffu) | (ey != 0 ? 0x800000u : 0u);
  const ulong mc = (ac & 0x7fffffu) | (ec != 0 ? 0x800000u : 0u);
  const ulong product = mx * my;
  const int pe = (ex != 0 ? int(ex) - 150 : -149) + (ey != 0 ? int(ey) - 150 : -149);
  const int ce = ec != 0 ? int(ec) - 150 : -149;
  const int pt = pe + 63 - int(clz(product));
  const int ct = mc != 0 ? ce + 63 - int(clz(mc)) : pt;
  const int bottom = max(pt, ct) - 62;
  const ulong p = f32_align(product, pe - bottom), add = f32_align(mc, ce - bottom);
  const ulong magnitude = sign == cs ? p + add : p >= add ? p - add : add - p;
  const uint result_sign = sign == cs || p >= add ? sign : cs;
  if (magnitude == 0) return 0;
  const int high = 63 - int(clz(magnitude));
  int exponent = bottom + high;
  if (exponent < -126) return result_sign | uint(f32_round(magnitude, -149 - bottom));
  if (exponent > 127) return result_sign | 0x7f800000u;
  ulong mantissa = f32_round(magnitude, high - 23);
  if (mantissa == 0x1000000ul) { mantissa >>= 1; exponent++; }
  if (exponent > 127) return result_sign | 0x7f800000u;
  return result_sign | (uint(exponent + 127) << 23) | uint(mantissa & 0x7ffffful);
}

template <typename T>
inline uint full_fragment_bits(thread const frag<T>& value, ushort row, ushort col) {
  const ushort owner = ((row & 3) << 1) | ((row & 4) << 2) | (col & 8) | ((col & 4) >> 2);
  const ushort element = ((row >> 3) << 2) | (col & 3);
  uint result = 0;
  TF_UNROLL
  for (ushort e = 0; e < 8; e++) {
    const uint candidate = simd_shuffle(as_type<uint>(float(value[e])), owner);
    if (e == element) result = candidate;
  }
  return result;
}

template <bool TA, bool TB, typename A, typename B>
inline void scalar_mma_16x32(thread frag<float>& lo, thread frag<float>& hi, thread const frag<A>& a,
                            thread const frag<B>& b0, thread const frag<B>& b1, ushort lane) {
  const short2 home = frag_home(lane);
  for (ushort k = 0; k < 16; k++) {
    for (ushort e = 0; e < 8; e++) {
      const ushort row = home.y + (e >> 2) * 8, col = home.x + (e & 3);
      const uint x = full_fragment_bits(a, TA ? k : row, TA ? row : k);
      const uint y0 = full_fragment_bits(b0, TB ? col : k, TB ? k : col);
      const uint y1 = full_fragment_bits(b1, TB ? col : k, TB ? k : col);
      lo[e] = as_type<float>(f32_fma_bits(x, y0, as_type<uint>(lo[e])));
      hi[e] = as_type<float>(f32_fma_bits(x, y1, as_type<uint>(hi[e])));
    }
  }
}

template <bool TA, bool TB, typename C, typename A, typename B, bool EXACT_B = false>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a,
                      thread const frag<B>& b0, thread const frag<B>& b1) {
  constexpr bool af = metal::is_same<A, float>::value, bf = metal::is_same<B, float>::value && !EXACT_B;
  if constexpr (!metal::is_same<C, float>::value || (!af && !bf)) {
    mma_16x32_cut<TA, TB>(lo, hi, a, b0, b1);
  } else {
    bool exceptional = false;
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const float x = float(a[e]), y = float(b0[e]), z = float(b1[e]);
      exceptional |= float_operand_needs_scalar(x) || float_operand_needs_scalar(y) || float_operand_needs_scalar(z);
    }
    if (simd_any(exceptional)) {
      scalar_mma_16x32<TA, TB>(lo, hi, a, b0, b1, ushort(simd_prefix_exclusive_sum(1u)));
      return;
    }
    frag<float> ah, am, at, bh0, bm0, bt0, bh1, bm1, bt1;
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const float3 x = split_float_operand(float(a[e]));
      const float3 y = split_float_operand(float(b0[e])), z = split_float_operand(float(b1[e]));
      ah[e] = x.x; am[e] = x.y; at[e] = x.z;
      bh0[e] = y.x; bm0[e] = y.y; bt0[e] = y.z;
      bh1[e] = z.x; bm1[e] = z.y; bt1[e] = z.z;
    }
    if constexpr (af && bf) {
      mma_16x32_cut<TA, TB>(lo, hi, ah, bh0, bh1);
      mma_16x32_cut<TA, TB>(lo, hi, ah, bm0, bm1);
      mma_16x32_cut<TA, TB>(lo, hi, am, bh0, bh1);
      mma_16x32_cut<TA, TB>(lo, hi, ah, bt0, bt1);
      mma_16x32_cut<TA, TB>(lo, hi, at, bh0, bh1);
      mma_16x32_cut<TA, TB>(lo, hi, am, bm0, bm1);
    } else if constexpr (af) {
      mma_16x32_cut<TA, TB>(lo, hi, ah, b0, b1);
      mma_16x32_cut<TA, TB>(lo, hi, am, b0, b1);
      mma_16x32_cut<TA, TB>(lo, hi, at, b0, b1);
    } else {
      mma_16x32_cut<TA, TB>(lo, hi, a, bh0, bh1);
      mma_16x32_cut<TA, TB>(lo, hi, a, bm0, bm1);
      mma_16x32_cut<TA, TB>(lo, hi, a, bt0, bt1);
    }
  }
}
#endif

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
template <bool TA, bool TB, typename C, typename A, typename B, bool EXACT_B = false>
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
