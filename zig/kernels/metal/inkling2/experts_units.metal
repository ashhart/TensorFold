// Inkling 2 Mini's routed experts on the tensor units, after nax.h, experts.metal and units.metal.

// The columns of a gate-up row and of a down row: constants, since a loop of constant length runs faster here.
#define INK2_GU_K 2304u
#define INK2_DOWN_K 1152u

// Columns c .. c + 3 of two rows of X (zeros for a row past the expert's pairs) as a left fragment's eight values.
inline tfp::frag<bfloat> ink2_units_x(device const bfloat* x0, device const bfloat* x1, bool live0, bool live1, uint c) {
  const bfloat z1 = bfloat(0.0f);
  const bfloat4 z = bfloat4(z1, z1, z1, z1);
  return tfp::frag<bfloat>(live0 ? *(device const bfloat4*)(x0 + c) : z, live1 ? *(device const bfloat4*)(x1 + c) : z);
}

// Columns 16b + hx .. + 3 of a packed NVFP4 row as BF16 weights, each as ink2_nv_block decodes it.
inline bfloat4 ink2_nv_quad(device const uchar* w, device const uchar* scales, float phi, uint b, uint hx) {
  const float half_scale = 0.5f * ink2_e4m3_bits(scales[b]);
  const uint word = ink2_word(w + 8u * b + (hx & 8u) / 2u) >> (4u * (hx & 4u));
  float4 v;
  for (uint j = 0; j < 4u; ++j) v[j] = ink2_e2m1_twice((word >> (4u * j)) & 15u) * half_scale * phi;
  return bfloat4(bfloat(v[0]), bfloat(v[1]), bfloat(v[2]), bfloat(v[3]));
}

// Columns 16b + hx .. + 3 of an MXFP8 row as BF16 weights, each as ink2_mx_block decodes it.
inline bfloat4 ink2_mx_quad(device const uchar* w, device const uchar* scales, uint b, uint hx) {
  const int shift = int(scales[b / 2u]) - 127;
  const int s0 = shift >> 1, s1 = shift - s0;
  const float p0 = as_type<float>(uint(s0 + 127) << 23u), p1 = as_type<float>(uint(s1 + 127) << 23u);
  const uint word = ink2_word(w + 16u * b + hx);
  float4 v;
  for (uint j = 0; j < 4u; ++j) v[j] = (ink2_e4m3_bits((word >> (8u * j)) & 255u) * p0) * p1;
  return bfloat4(bfloat(v[0]), bfloat(v[1]), bfloat(v[2]), bfloat(v[3]));
}

// Sum the 8 simdgroup partials in order using 8 KiB scratch (two 8-row halves) and write up to 16 pairs.
template <bool GU>
inline void ink2_units_store(threadgroup float* part, thread tfp::frag<float>& lo, thread tfp::frag<float>& hi,
                             uint p0, uint p1, bool live0, bool live1, bool has_q1, uint w0, uint hx, uint sg, uint lane,
                             device bfloat* Y, constant Ink2GroupedArgs& a) {
  for (uint q = 0u; q < (has_q1 ? 2u : 1u); ++q) {
    ink2_part_put_half(part, lo, hi, q, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && (q == 0u ? live0 : live1)) {
      ink2_part_sum_half(part, lo, hi, q, lane);
      const ulong pair = q == 0u ? p0 : p1;
      for (uint side = 0u; side < 2u; ++side) {
        const uint col = w0 + 16u * side + hx;
        const float4 v = side == 0u ? float4(lo[4u * q], lo[4u * q + 1u], lo[4u * q + 2u], lo[4u * q + 3u])
                                    : float4(hi[4u * q], hi[4u * q + 1u], hi[4u * q + 2u], hi[4u * q + 3u]);
        if (GU) {
          Y[pair * a.n + col / 2u] = ink2_act(v[0], v[1]);
          Y[pair * a.n + col / 2u + 1u] = ink2_act(v[2], v[3]);
        } else {
          for (uint j = 0u; j < 4u; ++j) Y[pair * a.n + col + j] = bfloat(v[j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Threadgroup (x, e): weight rows 32x .. 32x + 31 of expert e by its pairs, 32 a band; FP8 is MXFP8, else NVFP4.
template <bool FP8, bool GU>
inline void ink2_units(device const bfloat* X, device const uchar* W, device const uchar* S, float phi,
                       device const uint* counts, device const uint* members, device bfloat* Y,
                       constant Ink2GroupedArgs& a, uint2 tg, uint sg, uint lane, threadgroup float* part) {
  // GU: X is by row and the 2n interleaved gate and up rows give Y[pair][n]; else X is by pair. 32 divides the rows.
  const uint K = GU ? INK2_GU_K : INK2_DOWN_K, rows_w = GU ? 2u * a.n : a.n, steps = K / 128u;
  const uint expert = tg.y, w0 = 32u * tg.x;
  const uint count = counts[expert];
  if (count == 0u) return;
  const uint wb = FP8 ? K : K / 2u, sb = FP8 ? K / 32u : K / 16u;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  device const uchar* w = W + ulong(expert) * rows_w * wb + ulong(w0 + hy) * wb;
  device const uchar* s = S + ulong(expert) * rows_w * sb + ulong(w0 + hy) * sb;
  for (uint band = 0u; band < count; band += 32u) {
    const uint m0 = band + hy, m1 = band + hy + 8u;
    const uint m2 = band + 16u + hy, m3 = band + 16u + hy + 8u;
    const bool live0 = m0 < count, live1 = m1 < count;
    const bool has_q1_0 = band + 8u < count;
    const bool has_hi = band + 16u < count;
    const bool has_q1_1 = band + 24u < count;
    const bool live2 = m2 < count, live3 = m3 < count;
    const uint p0 = members[expert * a.cap + min(m0, count - 1u)], p1 = members[expert * a.cap + min(m1, count - 1u)];
    const uint p2 = members[expert * a.cap + min(m2, count - 1u)], p3 = members[expert * a.cap + min(m3, count - 1u)];
    device const bfloat* x0 = X + ulong(GU ? p0 / a.topk : p0) * K + hx;
    device const bfloat* x1 = X + ulong(GU ? p1 / a.topk : p1) * K + hx;
    device const bfloat* x2 = X + ulong(GU ? p2 / a.topk : p2) * K + hx;
    device const bfloat* x3 = X + ulong(GU ? p3 / a.topk : p3) * K + hx;
    tfp::frag<float> lo0 = tfp::frag<float>(0.0f), hi0 = tfp::frag<float>(0.0f);
    tfp::frag<float> lo1 = tfp::frag<float>(0.0f), hi1 = tfp::frag<float>(0.0f);
    // Simdgroup s sums the columns of eighth s from zero, 16 an op; the eight partial sums are then added in order.
    for (uint b = sg * steps; b < (sg + 1u) * steps; ++b) {
      tfp::frag<bfloat> b0, b1;
      if (FP8) {
        b0 = tfp::frag<bfloat>(ink2_mx_quad(w, s, b, hx), ink2_mx_quad(w + 8u * wb, s + 8u * sb, b, hx));
        b1 = tfp::frag<bfloat>(ink2_mx_quad(w + 16u * wb, s + 16u * sb, b, hx), ink2_mx_quad(w + 24u * wb, s + 24u * sb, b, hx));
      } else {
        b0 = tfp::frag<bfloat>(ink2_nv_quad(w, s, phi, b, hx), ink2_nv_quad(w + 8u * wb, s + 8u * sb, phi, b, hx));
        b1 = tfp::frag<bfloat>(ink2_nv_quad(w + 16u * wb, s + 16u * sb, phi, b, hx), ink2_nv_quad(w + 24u * wb, s + 24u * sb, phi, b, hx));
      }
      ink2_mma(lo0, hi0, ink2_units_x(x0, x1, live0, live1, 16u * b), b0, b1);
      if (has_hi) ink2_mma(lo1, hi1, ink2_units_x(x2, x3, live2, live3, 16u * b), b0, b1);
    }
    ink2_units_store<GU>(part, lo0, hi0, p0, p1, live0, live1, has_q1_0, w0, hx, sg, lane, Y, a);
    if (has_hi) ink2_units_store<GU>(part, lo1, hi1, p2, p3, live2, live3, has_q1_1, w0, hx, sg, lane, Y, a);
  }
}

// Single-row decode specialization: launches only active route slots (tg.y in 0..topk-1) with 1 KiB scratch.
template <bool FP8, bool GU>
inline void ink2_units_1(device const bfloat* X, device const uchar* W, device const uchar* S, float phi,
                         uint expert, uint pair, device bfloat* Y, constant Ink2GroupedArgs& a,
                         uint2 tg, uint sg, uint lane, threadgroup float* part) {
  const uint K = GU ? INK2_GU_K : INK2_DOWN_K, rows_w = GU ? 2u * a.n : a.n, steps = K / 128u;
  const uint w0 = 32u * tg.x;
  const uint wb = FP8 ? K : K / 2u, sb = FP8 ? K / 32u : K / 16u;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const bool live0 = (hy == 0u);
  device const uchar* w = W + ulong(expert) * rows_w * wb + ulong(w0 + hy) * wb;
  device const uchar* s = S + ulong(expert) * rows_w * sb + ulong(w0 + hy) * sb;
  device const bfloat* x0 = X + ulong(GU ? 0u : pair) * K + hx;
  tfp::frag<float> lo0 = tfp::frag<float>(0.0f), hi0 = tfp::frag<float>(0.0f);
  for (uint b = sg * steps; b < (sg + 1u) * steps; ++b) {
    tfp::frag<bfloat> b0, b1;
    if (FP8) {
      b0 = tfp::frag<bfloat>(ink2_mx_quad(w, s, b, hx), ink2_mx_quad(w + 8u * wb, s + 8u * sb, b, hx));
      b1 = tfp::frag<bfloat>(ink2_mx_quad(w + 16u * wb, s + 16u * sb, b, hx), ink2_mx_quad(w + 24u * wb, s + 24u * sb, b, hx));
    } else {
      b0 = tfp::frag<bfloat>(ink2_nv_quad(w, s, phi, b, hx), ink2_nv_quad(w + 8u * wb, s + 8u * sb, phi, b, hx));
      b1 = tfp::frag<bfloat>(ink2_nv_quad(w + 16u * wb, s + 16u * sb, phi, b, hx), ink2_nv_quad(w + 24u * wb, s + 24u * sb, phi, b, hx));
    }
    ink2_mma(lo0, hi0, ink2_units_x(x0, x0, live0, false, 16u * b), b0, b1);
  }
  if (live0) {
    const uint slot = (sg * 4u + (hx >> 2u)) * 8u;
    for (uint e = 0u; e < 4u; ++e) {
      part[slot + e] = lo0[e];
      part[slot + 4u + e] = hi0[e];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && live0) {
    const uint xslot = hx >> 2u;
    for (uint e = 0u; e < 4u; ++e)
      for (uint o = 1u; o < 8u; ++o) {
        lo0[e] += part[(o * 4u + xslot) * 8u + e];
        hi0[e] += part[(o * 4u + xslot) * 8u + 4u + e];
      }
    for (uint side = 0u; side < 2u; ++side) {
      const uint col = w0 + 16u * side + hx;
      const float4 v = side == 0u ? float4(lo0[0], lo0[1], lo0[2], lo0[3])
                                  : float4(hi0[0], hi0[1], hi0[2], hi0[3]);
      if (GU) {
        Y[ulong(pair) * a.n + col / 2u] = ink2_act(v[0], v[1]);
        Y[ulong(pair) * a.n + col / 2u + 1u] = ink2_act(v[2], v[3]);
      } else {
        for (uint j = 0u; j < 4u; ++j) Y[ulong(pair) * a.n + col + j] = bfloat(v[j]);
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu_units(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_units<false, true>(X, W, S, ink2_phi(amax, denominator, tg.y), counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down_units(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_units<false, false>(X, W, S, ink2_phi(amax, denominator, tg.y), counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu_units(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_units<true, true>(X, W, S, 1.0f, counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down_units(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2048];
  ink2_units<true, false>(X, W, S, 1.0f, counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu_units_1(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[256];
  const uint expert = ids[tg.y];
  ink2_units_1<false, true>(X, W, S, ink2_phi(amax, denominator, expert), expert, tg.y, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down_units_1(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[256];
  const uint expert = ids[tg.y];
  ink2_units_1<false, false>(X, W, S, ink2_phi(amax, denominator, expert), expert, tg.y, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu_units_1(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[256];
  const uint expert = ids[tg.y];
  ink2_units_1<true, true>(X, W, S, 1.0f, expert, tg.y, Y, a, tg, sg, lane, part);
}

// Small-batch (rows <= 8, count <= 8) active-expert specialization: launches only min(rows*topk, 128) slots.
template <bool FP8, bool GU>
inline void ink2_units_active(device const bfloat* X, device const uchar* W, device const uchar* S, float phi,
                              uint expert, device const uint* counts, device const uint* members, device bfloat* Y,
                              constant Ink2GroupedArgs& a, uint2 tg, uint sg, uint lane, threadgroup float* part) {
  const uint K = GU ? INK2_GU_K : INK2_DOWN_K, rows_w = GU ? 2u * a.n : a.n, steps = K / 128u;
  const uint w0 = 32u * tg.x;
  const uint count = counts[expert];
  if (count == 0u) return;
  const uint wb = FP8 ? K : K / 2u, sb = FP8 ? K / 32u : K / 16u;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const bool live0 = hy < count;
  const uint p0 = members[expert * a.cap + min(hy, count - 1u)];
  device const uchar* w = W + ulong(expert) * rows_w * wb + ulong(w0 + hy) * wb;
  device const uchar* s = S + ulong(expert) * rows_w * sb + ulong(w0 + hy) * sb;
  device const bfloat* x0 = X + ulong(GU ? p0 / a.topk : p0) * K + hx;
  tfp::frag<float> lo0 = tfp::frag<float>(0.0f), hi0 = tfp::frag<float>(0.0f);
  for (uint b = sg * steps; b < (sg + 1u) * steps; ++b) {
    tfp::frag<bfloat> b0, b1;
    if (FP8) {
      b0 = tfp::frag<bfloat>(ink2_mx_quad(w, s, b, hx), ink2_mx_quad(w + 8u * wb, s + 8u * sb, b, hx));
      b1 = tfp::frag<bfloat>(ink2_mx_quad(w + 16u * wb, s + 16u * sb, b, hx), ink2_mx_quad(w + 24u * wb, s + 24u * sb, b, hx));
    } else {
      b0 = tfp::frag<bfloat>(ink2_nv_quad(w, s, phi, b, hx), ink2_nv_quad(w + 8u * wb, s + 8u * sb, phi, b, hx));
      b1 = tfp::frag<bfloat>(ink2_nv_quad(w + 16u * wb, s + 16u * sb, phi, b, hx), ink2_nv_quad(w + 24u * wb, s + 24u * sb, phi, b, hx));
    }
    ink2_mma(lo0, hi0, ink2_units_x(x0, x0, live0, false, 16u * b), b0, b1);
  }
  ink2_part_put_half(part, lo0, hi0, 0u, sg, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && live0) {
    ink2_part_sum_half(part, lo0, hi0, 0u, lane);
    for (uint side = 0u; side < 2u; ++side) {
      const uint col = w0 + 16u * side + hx;
      const float4 v = side == 0u ? float4(lo0[0], lo0[1], lo0[2], lo0[3])
                                  : float4(hi0[0], hi0[1], hi0[2], hi0[3]);
      if (GU) {
        Y[ulong(p0) * a.n + col / 2u] = ink2_act(v[0], v[1]);
        Y[ulong(p0) * a.n + col / 2u + 1u] = ink2_act(v[2], v[3]);
      } else {
        for (uint j = 0u; j < 4u; ++j) Y[ulong(p0) * a.n + col + j] = bfloat(v[j]);
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down_units_1(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[256];
  const uint expert = ids[tg.y];
  ink2_units_1<true, false>(X, W, S, 1.0f, expert, tg.y, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu_units_active(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device const uint* active [[buffer(7)]],
    device bfloat* Y [[buffer(8)]], constant Ink2GroupedArgs& a [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint expert = active[tg.y];
  if (expert == 0xffffffffu) return;
  threadgroup float part[2048];
  ink2_units_active<false, true>(X, W, S, ink2_phi(amax, denominator, expert), expert, counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down_units_active(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device const uint* active [[buffer(7)]],
    device bfloat* Y [[buffer(8)]], constant Ink2GroupedArgs& a [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint expert = active[tg.y];
  if (expert == 0xffffffffu) return;
  threadgroup float part[2048];
  ink2_units_active<false, false>(X, W, S, ink2_phi(amax, denominator, expert), expert, counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu_units_active(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device const uint* active [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint expert = active[tg.y];
  if (expert == 0xffffffffu) return;
  threadgroup float part[2048];
  ink2_units_active<true, true>(X, W, S, 1.0f, expert, counts, members, Y, a, tg, sg, lane, part);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down_units_active(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device const uint* active [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint expert = active[tg.y];
  if (expert == 0xffffffffu) return;
  threadgroup float part[2048];
  ink2_units_active<true, false>(X, W, S, 1.0f, expert, counts, members, Y, a, tg, sg, lane, part);
}
