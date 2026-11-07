inline float tf_exl3_bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort tf_exl3_round_bf16(float x) {
  uint u = as_type<uint>(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
  return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
struct TfExl3SwiGLURot { uint rows, k, input_stride, output_stride, weighted, scatter, experts, reserved; };
kernel void tf_exl3_swiglu_prerotate(const device ushort* gate [[buffer(0)]],
                                    const device ushort* up [[buffer(1)]],
                                    const device float* weights [[buffer(2)]],
                                    const device uint* routes [[buffer(3)]],
                                    const device uint* destinations [[buffer(4)]],
                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                    device half* rotated [[buffer(6)]],
                                    constant TfExl3SwiGLURot& d [[buffer(7)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= d.rows) return;
  const uint expert = routes[row];
  if (expert >= d.experts) return;
  const uint base = (tg.x * (tptg.x / 32u) + sg) * 128u;
  if (base >= d.k) return;
  const device ushort* scales = descriptors[expert].su;
  const uint signs = descriptors[expert].su_signs;
  const float weight = d.weighted ? weights[row] : 1.0f;
  float v[4];
  for (uint j = 0; j < 4; ++j) {
    const uint k = base + lane + 32u * j;
    const ulong at = ulong(row) * d.input_stride + k;
    const float g = min(tf_exl3_bf(gate[at]), 10.0f);
    const float u = clamp(tf_exl3_bf(up[at]), -10.0f, 10.0f);
    float s = (g / (1.0f + exp(-g))) * u;
    if (d.weighted) s = weight * s;
    v[j] = tf_exl3_bf(tf_exl3_round_bf16(s)) * tf_exl3_scale(scales, k, signs);
  }
  tf_exl3_hadamard(v, lane);
  const uint dst = d.scatter ? destinations[row] : row;
  for (uint j = 0; j < 4; ++j)
    rotated[ulong(dst) * d.output_stride + base + lane + 32u * j] = half(v[j] * 0.08838834764831845f);
}

// Wide one-row tile: eight SIMDgroups each own a 16-column tile, and the shared activation tile is double buffered.
template <bool Table>
inline void tf_exl3_matmul_wide_impl(const device half* x, const device uchar* trellis,
                                     device float* sums, constant TfExl3Args& a,
                                     const device uint* routes, const device TfExl3Expert* descriptors,
                                     device float* partials, const device half* table,
                                     uint lane, uint sg, uint3 tg,
                                     threadgroup half* wx_all, threadgroup float* xx_all) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint maps[8], at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint cur0 = (wi0 << 5) | ((tf_exl3_map_of(lane * 8, halves) >> 6) & 31u);
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    uint c8 = cur0 + j * (halves >> 1);
    if (c8 >= ring) c8 -= ring;
    off[j] = c8 & 31u;
    sh[j] = (32u - off[j]) & 31u;
    rel[j] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, ring2 = 128 * halves;
      const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
      uint start = end + ring2 - 16;
      if (start >= ring2) start -= ring2;
      const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
      maps[j] = wi | (o << 6) | (next << 11);
    }
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  uint prefetch[4];
  {
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    prefetch[0] = tf_exl3_word(t0, wi0);
    prefetch[1] = tf_exl3_word(t0, a1);
    prefetch[2] = tf_exl3_word(t0, a2);
    prefetch[3] = tf_exl3_word(t0, a3);
  }
  for (uint b = b0; b < b1; ++b) {
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    // One warp loads the activation tile for all eight: 8 rows x 16 k, zeros past a.rows.
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* tile = reinterpret_cast<const device ushort*>(
        data + ulong(b) * ks + ulong(nb / 16) * ns);
    if ((halves & 1u) == 0 and halves <= 16) {
      uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
      uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
      uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
      const uint w0 = prefetch[0], w1 = prefetch[1], w2 = prefetch[2], w3 = prefetch[3];
      if (b + 1 < b1) {
        const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
        prefetch[0] = tf_exl3_word(tn, wi0);
        prefetch[1] = tf_exl3_word(tn, at1);
        prefetch[2] = tf_exl3_word(tn, at2);
        prefetch[3] = tf_exl3_word(tn, at3);
      }
      TF_UNROLL
      for (uint j = 0; j < 8; ++j) {
        const uint lo = rel[j] == 0 ? w0 : rel[j] == 1 ? w1 : w2;
        const uint hi = rel[j] == 0 ? w1 : rel[j] == 1 ? w2 : w3;
        const uint carry = (hi >> sh[j]) & (0u - uint(off[j] != 0));
        const ushort s = ushort(((lo << off[j]) | carry) >> 16);
        wx[at[j]] = Table ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook);
      }
    } else for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort state = tf_exl3_mapped_state(tile, maps[j]);
      wx[r * 16 + c] = Table ? tf_exl3_table_value(table, codebook, state) : tf_exl3_value(state, codebook);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
    // No second barrier: the next block writes the other activation buffer, and this one is read no more.
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}

// The wide one-row body with static state selection: the cursor leaves the inner loop, reading three words not four.
template <uint S>
inline void tf_exl3_matmul_wide_static_impl(const device half* x, const device uchar* trellis,
                                            device float* sums, constant TfExl3Args& a,
                                            const device uint* routes, const device TfExl3Expert* descriptors,
                                            device float* partials,
                                            uint lane, uint sg, uint3 tg,
                                            threadgroup half* wx_all, threadgroup float* xx_all,
                                            threadgroup half* lut) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint at[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint off0 = (tf_exl3_map_of(lane * 8, halves) >> 6) & 31u;
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state.
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  uint prefetch[3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    prefetch[0] = tf_exl3_word(t0, wi0);
    prefetch[1] = tf_exl3_word(t0, a1);
    prefetch[2] = tf_exl3_word(t0, a2);
  }
  // The lane's bit offset inside its first word: the one runtime shift of the body.
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  for (uint b = b0; b < b1; ++b) {
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const uint w0 = prefetch[0], w1 = prefetch[1], w2 = prefetch[2];
    if (b + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
      prefetch[0] = tf_exl3_word(tn, wi0);
      prefetch[1] = tf_exl3_word(tn, a1);
      prefetch[2] = tf_exl3_word(tn, a2);
    }
    // The lane's window: C_cur is the 32-bit window at the lane's first state.
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
    #define TF_EXL3_V(J) lut[tf_exl3_mul1_index(TF_EXL3_W(J))]
    wx[at[0]] = TF_EXL3_V(0);
    wx[at[1]] = TF_EXL3_V(1);
    wx[at[2]] = TF_EXL3_V(2);
    wx[at[3]] = TF_EXL3_V(3);
    wx[at[4]] = TF_EXL3_V(4);
    wx[at[5]] = TF_EXL3_V(5);
    wx[at[6]] = TF_EXL3_V(6);
    wx[at[7]] = TF_EXL3_V(7);
    #undef TF_EXL3_V
    #undef TF_EXL3_W
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}

kernel void tf_exl3_matmul_wide_static_lut(const device half* x [[buffer(0)]],
                                           const device uchar* trellis [[buffer(1)]],
                                           device float* sums [[buffer(2)]],
                                           constant TfExl3Args& a [[buffer(3)]],
                                           const device uint* routes [[buffer(4)]],
                                           const device TfExl3Expert* descriptors [[buffer(5)]],
                                           device float* partials [[buffer(6)]],
                                           const device half* table [[buffer(7)]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]],
                                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  threadgroup half lut[1024];
  // The LUT holds the mul1 values (the value depends only on the state's byte sum).
  if (a.codebook != 2u) {
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
    return;
  }
  for (uint b = sg * 32u + lane; b < 1024u; b += 256u) lut[b] = tf_exl3_mul1_value(b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (a.halves == 12u)
    tf_exl3_matmul_wide_static_impl<6>(x, trellis, sums, a, routes, descriptors, partials, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 10u)
    tf_exl3_matmul_wide_static_impl<5>(x, trellis, sums, a, routes, descriptors, partials, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 6u)
    tf_exl3_matmul_wide_static_impl<3>(x, trellis, sums, a, routes, descriptors, partials, lane, sg, tg, wx, xx, lut);
  else
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}
