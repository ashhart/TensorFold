// Fast M5 AWQ W4A16 g128 kernels reading .m5_tile layout: cooperative MV (M=1..32) and 64x64 NAX GEMM.
#ifndef TF_SIMD_FRAGS

template <uint ROWS, uint SK>
inline void sn_awq_m5_mv_impl(
    const device bfloat* X,
    const device uint* W,
    const device float2* SB,
    constant SnAwqArgs& a,
    device bfloat* Y,
    uint2 tg,
    uint sg,
    uint lane,
    threadgroup float* red) {
#pragma clang fp contract(off)
  const uint KG = uint(a.K) >> 7;
  const uint tile = tg.x >> 2;
  const uint c_base = (tg.x & 3u) * 8u;
  const uint c_sub = lane >> 2;
  const uint u4_idx = lane & 3u;
  const uint j = tile * 32u + c_base + c_sub;
  const bool live = (int(j) < a.N);
  const uint vec_k = uint(a.K) >> 3;
  const device uint4* X4 = (const device uint4*)X;

  const uint g_begin = (sg * KG) / SK;
  const uint g_end_raw = ((sg + 1u) * KG) / SK;
  const uint g_end = (TF_CTL == 3 && g_end_raw == KG && g_end_raw > g_begin) ? (g_end_raw - 1u) : g_end_raw;

#if TF_CTL == 2
  if (tg.x == 0 && sg == 0 && lane == 0) ((device ushort*)X)[0] ^= 1u;
#endif

  sn_awq_acc_t acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = sn_awq_acc_t(0);

  for (uint g = g_begin; g < g_end; ++g) {
    const device uint4* W4 = (const device uint4*)(W + (((size_t)tile * KG + g) * 32u + c_base) * 16u);
    const uint4 wq = W4[lane];
    sn_awq_acc_t qx[ROWS], xs[ROWS];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      qx[r] = sn_awq_acc_t(0);
      xs[r] = sn_awq_acc_t(0);
    }
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < 4u; ++u) {
      const uint word = wq[u];
      const vec<sn_awq_acc_t, 4> qlo = vec<sn_awq_acc_t, 4>(
          float(word & 0xfu), float((word >> 4) & 0xfu), float((word >> 8) & 0xfu), float((word >> 12) & 0xfu));
      const vec<sn_awq_acc_t, 4> qhi = vec<sn_awq_acc_t, 4>(
          float((word >> 16) & 0xfu), float((word >> 20) & 0xfu), float((word >> 24) & 0xfu), float(word >> 28));
      const uint x_u4 = g * 16u + u4_idx * 4u + u;
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
        const uint rr = (int(r) < a.M) ? r : 0u;
        const uint4 xv = X4[(size_t)rr * vec_k + x_u4];
        const vec<sn_awq_acc_t, 4> xlo = vec<sn_awq_acc_t, 4>(as_type<bfloat4>(xv.xy));
        const vec<sn_awq_acc_t, 4> xhi = vec<sn_awq_acc_t, 4>(as_type<bfloat4>(xv.zw));
        qx[r] += dot(qlo, xlo) + dot(qhi, xhi);
        xs[r] += (xlo.x + xlo.y + xlo.z + xlo.w) + (xhi.x + xhi.y + xhi.z + xhi.w);
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      qx[r] += simd_shuffle_xor(qx[r], 1);
      qx[r] += simd_shuffle_xor(qx[r], 2);
      xs[r] += simd_shuffle_xor(xs[r], 1);
      xs[r] += simd_shuffle_xor(xs[r], 2);
    }
    if (u4_idx == 0u && live) {
      const float2 sb = SB[(size_t)g * uint(a.N) + j];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
#if TF_CTL == 4
        acc[r] = sn_awq_acc_t(fma(sb.x, float(qx[r]), fma(sb.y, float(xs[r]), float(acc[r]))));
#else
        acc[r] = fma(sb.x, qx[r], fma(sb.y, xs[r], acc[r]));
#endif
      }
    }
  }

  if constexpr (SK == 1u) {
    if (u4_idx == 0u && live) {
      uint j_loc, ldy_eff;
      device bfloat* Y_dst = sn_awq_split_dest(Y, a, j, j_loc, ldy_eff);
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
        if (int(r) < a.M) {
          float out_v = float(acc[r]);
#if TF_CTL == 1
          if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
          Y_dst[(size_t)r * ldy_eff + j_loc] = bfloat(out_v);
        }
      }
    }
    return;
  }

  if (sg > 0u && u4_idx == 0u && live) {
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      if (int(r) < a.M) red[((sg - 1u) * 8u + c_sub) * ROWS + r] = float(acc[r]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u && u4_idx == 0u && live) {
    uint j_loc, ldy_eff;
    device bfloat* Y_dst = sn_awq_split_dest(Y, a, j, j_loc, ldy_eff);
    for (uint r = 0u; int(r) < a.M; ++r) {
      sn_awq_acc_t total = acc[r];
      _Pragma("clang loop unroll(full)")
      for (uint s2 = 1u; s2 < SK; ++s2) {
        total += sn_awq_acc_t(red[((s2 - 1u) * 8u + c_sub) * ROWS + r]);
      }
      float out_v = float(total);
#if TF_CTL == 1
      if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
      Y_dst[(size_t)r * ldy_eff + j_loc] = bfloat(out_v);
    }
  }
}

#define SN_AWQ_M5_MV(ROWS, SK)                                                                                   \
  [[max_total_threads_per_threadgroup(SK * 32)]]                                                                 \
  [[kernel]] void sn_awq_m5_mv_r##ROWS##_sk##SK(                                                                 \
      const device bfloat* X [[buffer(0)]],                                                                      \
      const device uint* W [[buffer(1)]],                                                                        \
      const device float2* SB [[buffer(2)]],                                                                     \
      constant SnAwqArgs& a [[buffer(3)]],                                                                       \
      device bfloat* Y [[buffer(4)]],                                                                            \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]],                                                                   \
      uint2 tg [[threadgroup_position_in_grid]]) {                                                               \
    threadgroup float red[(SK > 1 ? (SK - 1) * 8 * ROWS : 1)];                                                   \
    sn_awq_m5_mv_impl<ROWS, SK>(X, W, SB, a, Y, tg, sg, lane, red);                                              \
  }

#define SN_AWQ_M5_MV_R(ROWS) \
  SN_AWQ_M5_MV(ROWS, 1)      \
  SN_AWQ_M5_MV(ROWS, 2)      \
  SN_AWQ_M5_MV(ROWS, 4)      \
  SN_AWQ_M5_MV(ROWS, 8)

SN_AWQ_M5_MV_R(1)
SN_AWQ_M5_MV_R(2)
SN_AWQ_M5_MV_R(4)
SN_AWQ_M5_MV_R(8)
SN_AWQ_M5_MV_R(16)
SN_AWQ_M5_MV_R(32)

template <typename P_t>
inline void sn_awq_nax_acc(thread float (&C)[16], thread const P_t& P, float2 my_sb, float xs0, float xs1, short fn) {
#pragma clang fp contract(off)
  TF_UNROLL
  for (int f = 0; f < 2; f++) {
    TF_UNROLL
    for (int j = 0; j < 4; j++) {
      const float2 sb = simd_shuffle(my_sb, ushort(f * 16 + fn + j));
      TF_UNROLL
      for (int r = 0; r < 2; r++) {
        const float xs = r ? xs1 : xs0;
        const int i = f * 8 + r * 4 + j;
#if TF_CTL == 4
        C[i] = float(half(fma(sb.x, P[i], fma(sb.y, xs, C[i]))));
#else
        C[i] = fma(sb.x, P[i], fma(sb.y, xs, C[i]));
#endif
      }
    }
  }
}

// M5 prefill NAX GEMM (BM=64, BN=128, 8 simdgroups = 2x4): each simdgroup runs two 16x32 M-tiles sharing B.
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_awq_gemm_m5(
    const device bfloat* X [[buffer(0)]],
    const device uint* W [[buffer(1)]],
    const device float2* SB [[buffer(2)]],
    constant SnAwqArgs& a [[buffer(3)]],
    device bfloat* Y [[buffer(4)]],
    const device float* XS [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
#pragma clang fp contract(off)
  constexpr int NT = 32, GS = 128, NF = 2, SWZ = 2;
  const int tn = (a.N + 127) / 128, tm = (a.M + 63) / 64;
  const int ty = (int(tg.y) << SWZ) + (int(tg.x) & ((1 << SWZ) - 1));
  const int tx = int(tg.x) >> SWZ;
  if (tx >= tn || ty >= tm) return;

#if TF_CTL == 2
  if (tx == 0 && ty == 0 && sg == 0 && lane == 0) ((device ushort*)X)[0] ^= 1u;
#endif

  const int sg_m = int(sg >> 2), sg_n = int(sg & 3u);
  const int row0 = ty * 64 + sg_m * 32;
  const int rem = a.M - row0;
  const int tile = tx * 4 + sg_n;
  const int n0 = tile * NT;
  const bool active = (rem > 0 && n0 < a.N);
  const bool two_m = (rem > 16);
  const int M0 = active ? min(rem, 16) : 1;
  const int M1 = (active && two_m) ? min(rem - 16, 16) : 1;
  const int KG = a.K / GS;
  const int g_end = (TF_CTL == 3 && KG > 0) ? (KG - 1) : KG;
  const short qid = short(lane) >> 2;
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;

  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA0(
      (device bfloat*)(X + (size_t)(active ? row0 : 0) * a.K), dextents<int32_t, 2>(a.K, M0));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA1(
      (device bfloat*)(X + (size_t)((active && two_m) ? (row0 + 16) : 0) * a.K), dextents<int32_t, 2>(a.K, M1));

  float C0[16], C1[16];
  TF_UNROLL
  for (int i = 0; i < 16; i++) {
    C0[i] = 0.0f;
    C1[i] = 0.0f;
  }

  float2 my_sb = (active && 0 < g_end && n0 + int(lane) < a.N) ? SB[(size_t)n0 + lane] : float2(0.0f);
  const device float* XS0 = XS + (active ? row0 : 0);
  const device float* XS1 = XS + ((active && two_m) ? (row0 + 16) : 0);

  for (int g = 0; g < g_end; g++) {
    if ((g & 3) == 0) threadgroup_barrier(mem_flags::mem_none);
    if (!active) continue;
    const float2 my_sb_next = (g + 1 < g_end && n0 + int(lane) < a.N)
        ? SB[(size_t)(g + 1) * a.N + n0 + lane] : float2(0.0f);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)W + ((size_t)tile * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));
    {
      auto a0_slice = tA0.slice(g * GS, 0);
      auto P = op.template get_destination_cooperative_tensor<decltype(a0_slice), decltype(b), float>();
      op.run(a0_slice, b, P);
      const float xs00 = (fm < M0) ? XS0[(size_t)g * a.MP + fm] : 0.0f;
      const float xs01 = (fm + 8 < M0) ? XS0[(size_t)g * a.MP + fm + 8] : 0.0f;
      sn_awq_nax_acc(C0, P, my_sb, xs00, xs01, fn);
    }
    if (two_m) {
      auto a1_slice = tA1.slice(g * GS, 0);
      auto P = op.template get_destination_cooperative_tensor<decltype(a1_slice), decltype(b), float>();
      op.run(a1_slice, b, P);
      const float xs10 = (fm < M1) ? XS1[(size_t)g * a.MP + fm] : 0.0f;
      const float xs11 = (fm + 8 < M1) ? XS1[(size_t)g * a.MP + fm + 8] : 0.0f;
      sn_awq_nax_acc(C1, P, my_sb, xs10, xs11, fn);
    }
    my_sb = my_sb_next;
  }

  if (!active) return;
  uint n0_loc, ldy_eff;
  device bfloat* Y_dst = sn_awq_split_dest(Y, a, uint(n0), n0_loc, ldy_eff);
  device bfloat* Y0 = Y_dst + (size_t)row0 * ldy_eff;
  for (int f = 0; f < NF; f++) {
    for (int r = 0; r < 2; r++) {
      const int m = fm + 8 * r;
      const int col0_glob = n0 + f * 16 + fn;
      const int col0_loc = int(n0_loc) + f * 16 + fn;
      if (m < M0) {
        for (int j = 0; j < 4; j++) {
          if (col0_glob + j < a.N) {
            float out_v = C0[f * 8 + r * 4 + j];
#if TF_CTL == 1
            if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
            Y0[(size_t)m * ldy_eff + col0_loc + j] = bfloat(out_v);
          }
        }
      }
      if (two_m && m < M1) {
        for (int j = 0; j < 4; j++) {
          if (col0_glob + j < a.N) {
            float out_v = C1[f * 8 + r * 4 + j];
#if TF_CTL == 1
            if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
            Y0[(size_t)(m + 16) * ldy_eff + col0_loc + j] = bfloat(out_v);
          }
        }
      }
    }
  }
}
#endif
