// TFQ3 MoE decode kernels for single-row and deduplicated speculative windows.

template <bool HAS_5B>
[[kernel]] void pw_dec_gu_b1_t(
    const device uint* W          [[buffer(0)]],
    const device half* scales     [[buffer(1)]],
    const device PwExpertMeta* M  [[buffer(2)]],
    constant PwV3Tables* T        [[buffer(3)]],
    const device half* X_rot      [[buffer(4)]],
    const device int* pick        [[buffer(5)]],
    device float* GU_out          [[buffer(6)]],
    constant PwDims& d            [[buffer(7)]],
    const device uint* SH_W       [[buffer(8)]],
    const device half* SH_S       [[buffer(9)]],
    uint3 gid                     [[threadgroup_position_in_grid]],
    ushort sg                     [[simdgroup_index_in_threadgroup]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), slot = gid.y, is_up = gid.z;
  short2 home = pw_frag_home(lane);
  float a0, a1;
  if (slot < d.slots) {
    PwExpertMeta em = M[uint(pick[slot])];
    ulong w_off = is_up ? em.up_word_off : em.gate_word_off;
    uint sc_off = is_up ? em.up_sc_off   : em.gate_sc_off;
    uint wpb = em.b_bits * 2u;
    pw_mv_nosplit1<32u, HAS_5B>(W + w_off + ulong(strip) * 32ul * wpb * 32ul,
                                scales + ulong(sc_off) + ulong(strip >> 3) * 32ul,
                                em.b_bits, T, X_rot + ulong(slot / d.topk) * 4096ul, lane, home, a0, a1);
  } else {
    uint sh_strip = is_up * 128u + strip;
    if constexpr (HAS_5B) {
      pw_mv_nosplit1_5b<32u>(SH_W + ulong(sh_strip) * 32ul * 10ul * 32ul,
                             SH_S + ulong(sh_strip >> 3) * 32ul, T, X_rot, lane, home, a0, a1);
    } else {
      pw_mv_nosplit1_4b<32u>(SH_W + ulong(sh_strip) * 32ul * 8ul * 32ul,
                             SH_S + ulong(sh_strip >> 3) * 32ul, T, X_rot, lane, home, a0, a1);
    }
  }
  if ((lane & 9u) == 0u) {
    GU_out[ulong(slot * 2u + is_up) * 2048ul + strip * 16u + uint(home.y)]      = a0;
    GU_out[ulong(slot * 2u + is_up) * 2048ul + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

template <bool HAS_5B>
[[kernel]] void pw_dec_down_b1_t(
    const device uint* W          [[buffer(0)]],
    const device half* scales     [[buffer(1)]],
    const device PwExpertMeta* M  [[buffer(2)]],
    constant PwV3Tables* T        [[buffer(3)]],
    const device half* ACT_rot    [[buffer(4)]],
    const device int* pick        [[buffer(5)]],
    device float* YR_out          [[buffer(6)]],
    constant PwDims& d            [[buffer(7)]],
    const device uint* SH_W       [[buffer(8)]],
    const device half* SH_S       [[buffer(9)]],
    const device half* SACT_rot   [[buffer(10)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort sg                     [[simdgroup_index_in_threadgroup]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), slot = gid.y;
  short2 home = pw_frag_home(lane);
  float a0, a1;
  if (slot < d.slots) {
    PwExpertMeta em = M[uint(pick[slot])];
    uint wpb = em.b_bits * 2u;
    pw_mv_nosplit1<16u, HAS_5B>(W + em.down_word_off + ulong(strip) * 16ul * wpb * 32ul,
                                scales + ulong(em.down_sc_off) + ulong(strip >> 3) * 16ul,
                                em.b_bits, T, ACT_rot + ulong(slot) * 2048ul, lane, home, a0, a1);
  } else {
    if constexpr (HAS_5B) {
      pw_mv_nosplit1_5b<16u>(SH_W + ulong(strip) * 16ul * 10ul * 32ul,
                             SH_S + ulong(strip >> 3) * 16ul, T, SACT_rot, lane, home, a0, a1);
    } else {
      pw_mv_nosplit1_4b<16u>(SH_W + ulong(strip) * 16ul * 8ul * 32ul,
                             SH_S + ulong(strip >> 3) * 16ul, T, SACT_rot, lane, home, a0, a1);
    }
  }
  if ((lane & 9u) == 0u) {
    YR_out[ulong(slot) * 4096ul + strip * 16u + uint(home.y)]      = a0;
    YR_out[ulong(slot) * 4096ul + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

// Single-Pass B=2..16 MoE Dedup + Wave-0 Fused Shared Expert (ZERO 5b code when HAS_5B == false!)
template <bool HAS_5B>
[[kernel]] void pw_dec_gu_dedup_t(
    const device uint* W          [[buffer(0)]],
    const device half* scales     [[buffer(1)]],
    const device PwExpertMeta* M  [[buffer(2)]],
    constant PwV3Tables* T        [[buffer(3)]],
    const device half* X_rot      [[buffer(4)]],
    const device int* uids        [[buffer(5)]],
    const device int* umem        [[buffer(6)]],
    const device int* ucount      [[buffer(7)]],
    device float* GU_out          [[buffer(8)]],
    constant PwDims& d            [[buffer(9)]],
    const device uint* SH_W       [[buffer(10)]],
    const device half* SH_S       [[buffer(11)]],
    uint3 gid                     [[threadgroup_position_in_grid]],
    ushort sg                     [[simdgroup_index_in_threadgroup]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), is_up = gid.z;
  bool has_sh = (d.flags & 4u) != 0u;
  short2 home = pw_frag_home(lane);
  if (has_sh && gid.y == 0u) {
    uint sh_strip = is_up * 128u + strip;
    constexpr ulong sh_wpb = HAS_5B ? 10ul : 8ul;
    const device uint* sw = SH_W + ulong(sh_strip) * 32ul * sh_wpb * 32ul;
    const device half* ss = SH_S + ulong(sh_strip >> 3) * 32ul;
    for (uint rb = 0u; rb < d.rows; rb += 4u) {
      uint nc = metal::min(4u, d.rows - rb);
      const device half* x0 = X_rot + ulong(rb + 0u) * 4096ul;
      const device half* x1 = X_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * 4096ul;
      const device half* x2 = X_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * 4096ul;
      const device half* x3 = X_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * 4096ul;
      float4 a0, a1;
      if constexpr (HAS_5B) pw_mv_nosplit4_5b<32u>(sw, ss, T, x0, x1, x2, x3, lane, home, a0, a1);
      else                  pw_mv_nosplit4_4b<32u>(sw, ss, T, x0, x1, x2, x3, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        GU_out[ulong((d.slots + rb + 0u) * 2u + is_up) * 2048ul + r] = a0.x; GU_out[ulong((d.slots + rb + 0u) * 2u + is_up) * 2048ul + r + 8u] = a1.x;
        if (nc > 1u) { GU_out[ulong((d.slots + rb + 1u) * 2u + is_up) * 2048ul + r] = a0.y; GU_out[ulong((d.slots + rb + 1u) * 2u + is_up) * 2048ul + r + 8u] = a1.y; }
        if (nc > 2u) { GU_out[ulong((d.slots + rb + 2u) * 2u + is_up) * 2048ul + r] = a0.z; GU_out[ulong((d.slots + rb + 2u) * 2u + is_up) * 2048ul + r + 8u] = a1.z; }
        if (nc > 3u) { GU_out[ulong((d.slots + rb + 3u) * 2u + is_up) * 2048ul + r] = a0.w; GU_out[ulong((d.slots + rb + 3u) * 2u + is_up) * 2048ul + r + 8u] = a1.w; }
      }
    }
    return;
  }
  uint uidx = has_sh ? (gid.y - 1u) : gid.y;
  if (uidx >= uint(ucount[0])) return;
  const device int* mem = umem + uidx * 16u;
  uint n_toks = 0u;
  while (n_toks < 16u && mem[n_toks] >= 0) n_toks++;
  if (n_toks == 0u) return;
  PwExpertMeta em = M[uint(uids[uidx])];
  ulong w_off = is_up ? em.up_word_off : em.gate_word_off;
  uint sc_off = is_up ? em.up_sc_off   : em.gate_sc_off;
  uint wpb = em.b_bits * 2u;
  const device uint* w_ptr = W + w_off + ulong(strip) * 32ul * wpb * 32ul;
  const device half* s_ptr = scales + ulong(sc_off) + ulong(strip >> 3) * 32ul;

  for (uint base = 0u; base < n_toks; base += 4u) {
    uint rem = metal::min(4u, n_toks - base);
    if (rem == 1u) {
      uint s0 = uint(mem[base]);
      float a0, a1;
      pw_mv_nosplit1<32u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T, X_rot + ulong(s0 / d.topk) * 4096ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        GU_out[ulong(s0 * 2u + is_up) * 2048ul + strip * 16u + uint(home.y)]      = a0;
        GU_out[ulong(s0 * 2u + is_up) * 2048ul + strip * 16u + uint(home.y) + 8u] = a1;
      }
    } else if (rem == 2u) {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]);
      float2 a0, a1;
      pw_mv_nosplit2<32u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  X_rot + ulong(s0/d.topk)*4096ul, X_rot + ulong(s1/d.topk)*4096ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        GU_out[ulong(s0*2u+is_up)*2048ul + r] = a0.x; GU_out[ulong(s0*2u+is_up)*2048ul + r + 8u] = a1.x;
        GU_out[ulong(s1*2u+is_up)*2048ul + r] = a0.y; GU_out[ulong(s1*2u+is_up)*2048ul + r + 8u] = a1.y;
      }
    } else {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]), s2 = uint(mem[base + 2u]);
      uint s3 = (rem > 3u) ? uint(mem[base + 3u]) : s2;
      float4 a0, a1;
      pw_mv_nosplit4<32u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  X_rot + ulong(s0/d.topk)*4096ul, X_rot + ulong(s1/d.topk)*4096ul,
                                  X_rot + ulong(s2/d.topk)*4096ul, X_rot + ulong(s3/d.topk)*4096ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        GU_out[ulong(s0*2u+is_up)*2048ul + r] = a0.x; GU_out[ulong(s0*2u+is_up)*2048ul + r + 8u] = a1.x;
        GU_out[ulong(s1*2u+is_up)*2048ul + r] = a0.y; GU_out[ulong(s1*2u+is_up)*2048ul + r + 8u] = a1.y;
        GU_out[ulong(s2*2u+is_up)*2048ul + r] = a0.z; GU_out[ulong(s2*2u+is_up)*2048ul + r + 8u] = a1.z;
        if (rem > 3u) { GU_out[ulong(s3*2u+is_up)*2048ul + r] = a0.w; GU_out[ulong(s3*2u+is_up)*2048ul + r + 8u] = a1.w; }
      }
    }
  }
}

template <bool HAS_5B>
[[kernel]] void pw_dec_down_dedup_t(
    const device uint* W          [[buffer(0)]],
    const device half* scales     [[buffer(1)]],
    const device PwExpertMeta* M  [[buffer(2)]],
    constant PwV3Tables* T        [[buffer(3)]],
    const device half* ACT_rot    [[buffer(4)]],
    const device int* uids        [[buffer(5)]],
    const device int* umem        [[buffer(6)]],
    const device int* ucount      [[buffer(7)]],
    device float* YR_out          [[buffer(8)]],
    constant PwDims& d            [[buffer(9)]],
    const device uint* SH_W       [[buffer(10)]],
    const device half* SH_S       [[buffer(11)]],
    const device half* SACT_rot   [[buffer(12)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort sg                     [[simdgroup_index_in_threadgroup]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg);
  bool has_sh = (d.flags & 4u) != 0u;
  short2 home = pw_frag_home(lane);
  if (has_sh && gid.y == 0u) {
    // Wave-0 Fused Shared Expert Down across all d.rows in ONE weight pass (ZERO 5b code when HAS_5B == false!)
    constexpr ulong sh_wpb = HAS_5B ? 10ul : 8ul;
    const device uint* sw = SH_W + ulong(strip) * 16ul * sh_wpb * 32ul;
    const device half* ss = SH_S + ulong(strip >> 3) * 16ul;
    for (uint rb = 0u; rb < d.rows; rb += 4u) {
      uint nc = metal::min(4u, d.rows - rb);
      const device half* x0 = SACT_rot + ulong(rb + 0u) * 2048ul;
      const device half* x1 = SACT_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * 2048ul;
      const device half* x2 = SACT_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * 2048ul;
      const device half* x3 = SACT_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * 2048ul;
      float4 y0, y1;
      if constexpr (HAS_5B) pw_mv_nosplit4_5b<16u>(sw, ss, T, x0, x1, x2, x3, lane, home, y0, y1);
      else                  pw_mv_nosplit4_4b<16u>(sw, ss, T, x0, x1, x2, x3, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        YR_out[ulong(d.slots + rb + 0u) * 4096ul + r] = y0.x; YR_out[ulong(d.slots + rb + 0u) * 4096ul + r + 8u] = y1.x;
        if (nc > 1u) { YR_out[ulong(d.slots + rb + 1u) * 4096ul + r] = y0.y; YR_out[ulong(d.slots + rb + 1u) * 4096ul + r + 8u] = y1.y; }
        if (nc > 2u) { YR_out[ulong(d.slots + rb + 2u) * 4096ul + r] = y0.z; YR_out[ulong(d.slots + rb + 2u) * 4096ul + r + 8u] = y1.z; }
        if (nc > 3u) { YR_out[ulong(d.slots + rb + 3u) * 4096ul + r] = y0.w; YR_out[ulong(d.slots + rb + 3u) * 4096ul + r + 8u] = y1.w; }
      }
    }
    return;
  }
  uint uidx = has_sh ? (gid.y - 1u) : gid.y;
  if (uidx >= uint(ucount[0])) return;
  const device int* mem = umem + uidx * 16u;
  uint n_toks = 0u;
  while (n_toks < 16u && mem[n_toks] >= 0) n_toks++;
  if (n_toks == 0u) return;
  PwExpertMeta em = M[uint(uids[uidx])];
  uint wpb = em.b_bits * 2u;
  const device uint* w_ptr = W + em.down_word_off + ulong(strip) * 16ul * wpb * 32ul;
  const device half* s_ptr = scales + ulong(em.down_sc_off) + ulong(strip >> 3) * 16ul;

  for (uint base = 0u; base < n_toks; base += 4u) {
    uint rem = metal::min(4u, n_toks - base);
    if (rem == 1u) {
      uint s0 = uint(mem[base]);
      float y0, y1;
      pw_mv_nosplit1<16u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T, ACT_rot + ulong(s0) * 2048ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        YR_out[ulong(s0) * 4096ul + strip * 16u + uint(home.y)]      = y0;
        YR_out[ulong(s0) * 4096ul + strip * 16u + uint(home.y) + 8u] = y1;
      }
    } else if (rem == 2u) {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]);
      float2 y0, y1;
      pw_mv_nosplit2<16u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  ACT_rot + ulong(s0)*2048ul, ACT_rot + ulong(s1)*2048ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        YR_out[ulong(s0)*4096ul + r] = y0.x; YR_out[ulong(s0)*4096ul + r + 8u] = y1.x;
        YR_out[ulong(s1)*4096ul + r] = y0.y; YR_out[ulong(s1)*4096ul + r + 8u] = y1.y;
      }
    } else {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]), s2 = uint(mem[base + 2u]);
      uint s3 = (rem > 3u) ? uint(mem[base + 3u]) : s2;
      float4 y0, y1;
      pw_mv_nosplit4<16u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  ACT_rot + ulong(s0)*2048ul, ACT_rot + ulong(s1)*2048ul,
                                  ACT_rot + ulong(s2)*2048ul, ACT_rot + ulong(s3)*2048ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        YR_out[ulong(s0)*4096ul + r] = y0.x; YR_out[ulong(s0)*4096ul + r + 8u] = y1.x;
        YR_out[ulong(s1)*4096ul + r] = y0.y; YR_out[ulong(s1)*4096ul + r + 8u] = y1.y;
        YR_out[ulong(s2)*4096ul + r] = y0.z; YR_out[ulong(s2)*4096ul + r + 8u] = y1.z;
        if (rem > 3u) { YR_out[ulong(s3)*4096ul + r] = y0.w; YR_out[ulong(s3)*4096ul + r + 8u] = y1.w; }
      }
    }
  }
}

template [[host_name("pw_dec_gu_b1_543")]]        [[kernel]] decltype(pw_dec_gu_b1_t<true>)           pw_dec_gu_b1_t<true>;
template [[host_name("pw_dec_gu_b1_43")]]         [[kernel]] decltype(pw_dec_gu_b1_t<false>)          pw_dec_gu_b1_t<false>;
template [[host_name("pw_dec_down_b1_543")]]      [[kernel]] decltype(pw_dec_down_b1_t<true>)         pw_dec_down_b1_t<true>;
template [[host_name("pw_dec_down_b1_43")]]       [[kernel]] decltype(pw_dec_down_b1_t<false>)        pw_dec_down_b1_t<false>;
template [[host_name("pw_dec_gu_dedup_543")]]     [[kernel]] decltype(pw_dec_gu_dedup_t<true>)        pw_dec_gu_dedup_t<true>;
template [[host_name("pw_dec_gu_dedup_43")]]      [[kernel]] decltype(pw_dec_gu_dedup_t<false>)       pw_dec_gu_dedup_t<false>;
template [[host_name("pw_dec_down_dedup_543")]]   [[kernel]] decltype(pw_dec_down_dedup_t<true>)      pw_dec_down_dedup_t<true>;
template [[host_name("pw_dec_down_dedup_43")]]    [[kernel]] decltype(pw_dec_down_dedup_t<false>)     pw_dec_down_dedup_t<false>;
