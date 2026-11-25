// TFQ3 FlashNext MoE decode kernels with branch-hoisted fused affine shared expert (B=1 and B=2..16 dedup).

template <bool HAS_5B>
[[kernel]] void fn_pw_dec_gu_b1_t(
    const device uint* W            [[buffer(0)]],
    const device half* scales       [[buffer(1)]],
    const device FnPwExpertMeta* M  [[buffer(2)]],
    constant FnPwV3Tables* T        [[buffer(3)]],
    const device half* X_rot        [[buffer(4)]],
    const device uint* pick         [[buffer(5)]],
    device float* GU_out            [[buffer(6)]],
    constant FnPwDims& d            [[buffer(7)]],
    const device bfloat* X_raw      [[buffer(8)]],
    const device uint* SGW          [[buffer(9)]],
    const device bfloat* SGS        [[buffer(10)]],
    const device bfloat* SGB        [[buffer(11)]],
    const device uint* SUW          [[buffer(12)]],
    const device bfloat* SUS        [[buffer(13)]],
    const device bfloat* SUB        [[buffer(14)]],
    device bfloat* ACT              [[buffer(15)]],
    const constant int* rows_buf    [[buffer(16)]],
    uint3 gid                       [[threadgroup_position_in_grid]],
    ushort sg                       [[simdgroup_index_in_threadgroup]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
  if (rows_buf[0] <= 0) return;
  uint strip = gid.x * 2u + uint(sg), slot = gid.y, is_up = gid.z;
  if (slot >= 10u) {
    uint row = strip * 16u + (slot - 10u) * 2u + is_up;
    uint gs = (d.sh_gu_group == 32u) ? 5u : ((d.sh_gu_group == 64u) ? 6u : 7u);
    if (d.sh_gu_bits == 8u) fn_sh_gu_b1<8u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, 8u, gs, lane);
    else if (d.sh_gu_bits == 4u) fn_sh_gu_b1<4u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, 4u, gs, lane);
    else fn_sh_gu_b1<0u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, d.sh_gu_bits, gs, lane);
    return;
  }
  short2 home = pw_frag_home(lane);
  FnPwExpertMeta em = M[pick[slot]];
  ulong w_off = is_up ? em.up_word_off : em.gate_word_off;
  uint sc_off = is_up ? em.up_sc_off   : em.gate_sc_off;
  uint wpb = em.b_bits * 2u;
  float a0, a1;
  pw_mv_nosplit1<20u, HAS_5B>(W + w_off + ulong(strip) * 20ul * wpb * 32ul,
                              scales + ulong(sc_off) + ulong(strip >> 3) * 20ul,
                              em.b_bits, T, X_rot, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    GU_out[ulong(slot * 2u + is_up) * 640ul + strip * 16u + uint(home.y)]      = a0;
    GU_out[ulong(slot * 2u + is_up) * 640ul + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

template <bool HAS_5B>
[[kernel]] void fn_pw_dec_down_b1_t(
    const device uint* W            [[buffer(0)]],
    const device half* scales       [[buffer(1)]],
    const device FnPwExpertMeta* M  [[buffer(2)]],
    constant FnPwV3Tables* T        [[buffer(3)]],
    const device half* ACT_rot      [[buffer(4)]],
    const device uint* pick         [[buffer(5)]],
    device float* YR_out            [[buffer(6)]],
    constant FnPwDims& d            [[buffer(7)]],
    const device bfloat* ACT        [[buffer(8)]],
    const device uint* SDW          [[buffer(9)]],
    const device bfloat* SDS        [[buffer(10)]],
    const device bfloat* SDB        [[buffer(11)]],
    device bfloat* YDOWN            [[buffer(12)]],
    const constant int* rows_buf    [[buffer(13)]],
    uint2 gid                       [[threadgroup_position_in_grid]],
    ushort sg                       [[simdgroup_index_in_threadgroup]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
  if (rows_buf[0] <= 0) return;
  uint strip = gid.x * 2u + uint(sg), slot = gid.y;
  if (slot >= 10u) {
    uint r0 = strip * 16u + (slot - 10u) * 4u;
    uint gs = (d.sh_dn_group == 32u) ? 5u : ((d.sh_dn_group == 64u) ? 6u : 7u);
    if (d.sh_dn_bits == 8u) {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_b1<8u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, 8u, gs, lane);
    } else if (d.sh_dn_bits == 4u) {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_b1<4u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, 4u, gs, lane);
    } else {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_b1<0u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, d.sh_dn_bits, gs, lane);
    }
    return;
  }
  short2 home = pw_frag_home(lane);
  FnPwExpertMeta em = M[pick[slot]];
  uint wpb = em.b_bits * 2u;
  float a0, a1;
  pw_mv_nosplit1<5u, HAS_5B>(W + em.down_word_off + ulong(strip) * 5ul * wpb * 32ul,
                             scales + ulong(em.down_sc_off) + ulong(strip >> 3) * 5ul,
                             em.b_bits, T, ACT_rot + ulong(slot) * 640ul, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    YR_out[ulong(slot) * 2560ul + strip * 16u + uint(home.y)]      = a0;
    YR_out[ulong(slot) * 2560ul + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

template <bool HAS_5B>
[[kernel]] void fn_pw_dec_gu_dedup_t(
    const device uint* W            [[buffer(0)]],
    const device half* scales       [[buffer(1)]],
    const device FnPwExpertMeta* M  [[buffer(2)]],
    constant FnPwV3Tables* T        [[buffer(3)]],
    const device half* X_rot        [[buffer(4)]],
    const device int* uids          [[buffer(5)]],
    const device int* umem          [[buffer(6)]],
    const device int* ucount        [[buffer(7)]],
    device float* GU_out            [[buffer(8)]],
    constant FnPwDims& d            [[buffer(9)]],
    const device bfloat* X_raw      [[buffer(10)]],
    const device uint* SGW          [[buffer(11)]],
    const device bfloat* SGS        [[buffer(12)]],
    const device bfloat* SGB        [[buffer(13)]],
    const device uint* SUW          [[buffer(14)]],
    const device bfloat* SUS        [[buffer(15)]],
    const device bfloat* SUB        [[buffer(16)]],
    device bfloat* ACT              [[buffer(17)]],
    const constant int* rows_buf    [[buffer(18)]],
    uint3 gid                       [[threadgroup_position_in_grid]],
    ushort sg                       [[simdgroup_index_in_threadgroup]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  if (R <= 0) return;
  uint strip = gid.x * 2u + uint(sg), is_up = gid.z;
  if (gid.y < 8u) {
    uint row = strip * 16u + gid.y * 2u + is_up;
    uint gs = (d.sh_gu_group == 32u) ? 5u : ((d.sh_gu_group == 64u) ? 6u : 7u);
    if (d.sh_gu_bits == 8u) fn_sh_gu_multi<8u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, R, 8u, gs, lane);
    else if (d.sh_gu_bits == 4u) fn_sh_gu_multi<4u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, R, 4u, gs, lane);
    else fn_sh_gu_multi<0u>(X_raw, SGW, SGS, SGB, SUW, SUS, SUB, ACT, row, R, d.sh_gu_bits, gs, lane);
    return;
  }
  uint uidx = gid.y - 8u;
  if (uidx >= uint(ucount[0])) return;
  const device int* mem = umem + uidx * 16u;
  uint n_toks = 0u;
  while (n_toks < 16u && mem[n_toks] >= 0) n_toks++;
  if (n_toks == 0u) return;
  short2 home = pw_frag_home(lane);
  FnPwExpertMeta em = M[uint(uids[uidx])];
  ulong w_off = is_up ? em.up_word_off : em.gate_word_off;
  uint sc_off = is_up ? em.up_sc_off   : em.gate_sc_off;
  uint wpb = em.b_bits * 2u;
  const device uint* w_ptr = W + w_off + ulong(strip) * 20ul * wpb * 32ul;
  const device half* s_ptr = scales + ulong(sc_off) + ulong(strip >> 3) * 20ul;

  for (uint base = 0u; base < n_toks; base += 4u) {
    uint rem = metal::min(4u, n_toks - base);
    if (rem == 1u) {
      uint s0 = uint(mem[base]);
      float a0, a1;
      pw_mv_nosplit1<20u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T, X_rot + ulong(s0 / 10u) * 2560ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        GU_out[ulong(s0 * 2u + is_up) * 640ul + strip * 16u + uint(home.y)]      = a0;
        GU_out[ulong(s0 * 2u + is_up) * 640ul + strip * 16u + uint(home.y) + 8u] = a1;
      }
    } else if (rem == 2u) {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]);
      float2 a0, a1;
      pw_mv_nosplit2<20u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  X_rot + ulong(s0 / 10u) * 2560ul, X_rot + ulong(s1 / 10u) * 2560ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        GU_out[ulong(s0 * 2u + is_up) * 640ul + r] = a0.x; GU_out[ulong(s0 * 2u + is_up) * 640ul + r + 8u] = a1.x;
        GU_out[ulong(s1 * 2u + is_up) * 640ul + r] = a0.y; GU_out[ulong(s1 * 2u + is_up) * 640ul + r + 8u] = a1.y;
      }
    } else {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]), s2 = uint(mem[base + 2u]);
      uint s3 = (rem > 3u) ? uint(mem[base + 3u]) : s2;
      float4 a0, a1;
      pw_mv_nosplit4<20u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                  X_rot + ulong(s0 / 10u) * 2560ul, X_rot + ulong(s1 / 10u) * 2560ul,
                                  X_rot + ulong(s2 / 10u) * 2560ul, X_rot + ulong(s3 / 10u) * 2560ul, lane, home, a0, a1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        GU_out[ulong(s0 * 2u + is_up) * 640ul + r] = a0.x; GU_out[ulong(s0 * 2u + is_up) * 640ul + r + 8u] = a1.x;
        GU_out[ulong(s1 * 2u + is_up) * 640ul + r] = a0.y; GU_out[ulong(s1 * 2u + is_up) * 640ul + r + 8u] = a1.y;
        GU_out[ulong(s2 * 2u + is_up) * 640ul + r] = a0.z; GU_out[ulong(s2 * 2u + is_up) * 640ul + r + 8u] = a1.z;
        if (rem > 3u) { GU_out[ulong(s3 * 2u + is_up) * 640ul + r] = a0.w; GU_out[ulong(s3 * 2u + is_up) * 640ul + r + 8u] = a1.w; }
      }
    }
  }
}

template <bool HAS_5B>
[[kernel]] void fn_pw_dec_down_dedup_t(
    const device uint* W            [[buffer(0)]],
    const device half* scales       [[buffer(1)]],
    const device FnPwExpertMeta* M  [[buffer(2)]],
    constant FnPwV3Tables* T        [[buffer(3)]],
    const device half* ACT_rot      [[buffer(4)]],
    const device int* uids          [[buffer(5)]],
    const device int* umem          [[buffer(6)]],
    const device int* ucount        [[buffer(7)]],
    device float* YR_out            [[buffer(8)]],
    constant FnPwDims& d            [[buffer(9)]],
    const device bfloat* ACT        [[buffer(10)]],
    const device uint* SDW          [[buffer(11)]],
    const device bfloat* SDS        [[buffer(12)]],
    const device bfloat* SDB        [[buffer(13)]],
    device bfloat* YDOWN            [[buffer(14)]],
    const constant int* rows_buf    [[buffer(15)]],
    uint2 gid                       [[threadgroup_position_in_grid]],
    ushort sg                       [[simdgroup_index_in_threadgroup]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  if (R <= 0) return;
  uint strip = gid.x * 2u + uint(sg);
  if (gid.y < 4u) {
    uint r0 = strip * 16u + gid.y * 4u;
    uint gs = (d.sh_dn_group == 32u) ? 5u : ((d.sh_dn_group == 64u) ? 6u : 7u);
    if (d.sh_dn_bits == 8u) {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_multi<8u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, R, 8u, gs, lane);
    } else if (d.sh_dn_bits == 4u) {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_multi<4u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, R, 4u, gs, lane);
    } else {
      for (uint k = 0u; k < 4u; ++k) fn_sh_dn_multi<0u>(ACT, SDW, SDS, SDB, YDOWN, r0 + k, R, d.sh_dn_bits, gs, lane);
    }
    return;
  }
  uint uidx = gid.y - 4u;
  if (uidx >= uint(ucount[0])) return;
  const device int* mem = umem + uidx * 16u;
  uint n_toks = 0u;
  while (n_toks < 16u && mem[n_toks] >= 0) n_toks++;
  if (n_toks == 0u) return;
  short2 home = pw_frag_home(lane);
  FnPwExpertMeta em = M[uint(uids[uidx])];
  uint wpb = em.b_bits * 2u;
  const device uint* w_ptr = W + em.down_word_off + ulong(strip) * 5ul * wpb * 32ul;
  const device half* s_ptr = scales + ulong(em.down_sc_off) + ulong(strip >> 3) * 5ul;

  for (uint base = 0u; base < n_toks; base += 4u) {
    uint rem = metal::min(4u, n_toks - base);
    if (rem == 1u) {
      uint s0 = uint(mem[base]);
      float y0, y1;
      pw_mv_nosplit1<5u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T, ACT_rot + ulong(s0) * 640ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        YR_out[ulong(s0) * 2560ul + strip * 16u + uint(home.y)]      = y0;
        YR_out[ulong(s0) * 2560ul + strip * 16u + uint(home.y) + 8u] = y1;
      }
    } else if (rem == 2u) {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]);
      float2 y0, y1;
      pw_mv_nosplit2<5u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                 ACT_rot + ulong(s0) * 640ul, ACT_rot + ulong(s1) * 640ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        YR_out[ulong(s0) * 2560ul + r] = y0.x; YR_out[ulong(s0) * 2560ul + r + 8u] = y1.x;
        YR_out[ulong(s1) * 2560ul + r] = y0.y; YR_out[ulong(s1) * 2560ul + r + 8u] = y1.y;
      }
    } else {
      uint s0 = uint(mem[base]), s1 = uint(mem[base + 1u]), s2 = uint(mem[base + 2u]);
      uint s3 = (rem > 3u) ? uint(mem[base + 3u]) : s2;
      float4 y0, y1;
      pw_mv_nosplit4<5u, HAS_5B>(w_ptr, s_ptr, em.b_bits, T,
                                 ACT_rot + ulong(s0) * 640ul, ACT_rot + ulong(s1) * 640ul,
                                 ACT_rot + ulong(s2) * 640ul, ACT_rot + ulong(s3) * 640ul, lane, home, y0, y1);
      if ((lane & 9u) == 0u) {
        uint r = strip * 16u + uint(home.y);
        YR_out[ulong(s0) * 2560ul + r] = y0.x; YR_out[ulong(s0) * 2560ul + r + 8u] = y1.x;
        YR_out[ulong(s1) * 2560ul + r] = y0.y; YR_out[ulong(s1) * 2560ul + r + 8u] = y1.y;
        YR_out[ulong(s2) * 2560ul + r] = y0.z; YR_out[ulong(s2) * 2560ul + r + 8u] = y1.z;
        if (rem > 3u) { YR_out[ulong(s3) * 2560ul + r] = y0.w; YR_out[ulong(s3) * 2560ul + r + 8u] = y1.w; }
      }
    }
  }
}

template [[host_name("fn_pw_dec_gu_b1_543")]]      [[kernel]] decltype(fn_pw_dec_gu_b1_t<true>)       fn_pw_dec_gu_b1_t<true>;
template [[host_name("fn_pw_dec_down_b1_543")]]    [[kernel]] decltype(fn_pw_dec_down_b1_t<true>)     fn_pw_dec_down_b1_t<true>;
template [[host_name("fn_pw_dec_gu_dedup_543")]]   [[kernel]] decltype(fn_pw_dec_gu_dedup_t<true>)    fn_pw_dec_gu_dedup_t<true>;
template [[host_name("fn_pw_dec_down_dedup_543")]] [[kernel]] decltype(fn_pw_dec_down_dedup_t<true>)  fn_pw_dec_down_dedup_t<true>;
template [[host_name("fn_pw_dec_gu_b1_43")]]       [[kernel]] decltype(fn_pw_dec_gu_b1_t<false>)      fn_pw_dec_gu_b1_t<false>;
template [[host_name("fn_pw_dec_down_b1_43")]]     [[kernel]] decltype(fn_pw_dec_down_b1_t<false>)    fn_pw_dec_down_b1_t<false>;
template [[host_name("fn_pw_dec_gu_dedup_43")]]    [[kernel]] decltype(fn_pw_dec_gu_dedup_t<false>)   fn_pw_dec_gu_dedup_t<false>;
template [[host_name("fn_pw_dec_down_dedup_43")]]  [[kernel]] decltype(fn_pw_dec_down_dedup_t<false>) fn_pw_dec_down_dedup_t<false>;
