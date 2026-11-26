// Coalesced threadgroup-staged grouped routed experts for Inkling 2 Mini (0-ULP serial column order).
inline void ink2_stage_x(device const bfloat* X, threadgroup const uint* tg_xoff, uint len, uint c0, uint cols,
                         uint tid, uint x_stride_u4, threadgroup uint* tg_x) {
  const uint cols_u4 = cols >> 3u, total_u4 = len * cols_u4;
  for (uint idx = tid; idx < total_u4; idx += 64u) {
    const uint m = idx / cols_u4, q = idx % cols_u4;
    *(threadgroup uint4*)(tg_x + (m * x_stride_u4 + q) * 4u) = *(device const uint4*)(X + tg_xoff[m] + c0 + q * 8u);
  }
}

template <uint BK, bool GU>
inline void ink2_stage_nv_w(device const uchar* W, device const uchar* S, uint n0, uint valid_n, uint c0, uint cols,
                            uint k_bytes, uint s_bytes, uint tid, threadgroup uint* tg_w, threadgroup uchar* tg_s) {
  constexpr uint W_STRIDE = 66u, BK_WORDS = BK / 8u;
  constexpr uint PLANE_STRIDE = GU ? (BK_WORDS * W_STRIDE + (BK_WORDS == 8u ? 0u : 4u)) : 0u;
  const uint valid_rows = GU ? (2u * valid_n) : valid_n, c_byte = c0 >> 1u;
  if ((cols & 31u) == 0u && (k_bytes & 15u) == 0u) {
    const uint stage_u4 = cols >> 5u, total_q = valid_rows * stage_u4;
    for (uint idx = tid; idx < total_q; idx += 64u) {
      const uint r = idx / stage_u4, q = idx % stage_u4;
      const uint n_local = GU ? (r >> 1u) : r, plane = GU ? (r & 1u) : 0u;
      const uint4 v = *(device const uint4*)(W + ulong(GU ? (2u * n0 + r) : (n0 + r)) * k_bytes + c_byte + q * 16u);
      const uint base = plane * PLANE_STRIDE + (4u * q) * W_STRIDE + n_local;
      tg_w[base] = v.x; tg_w[base + W_STRIDE] = v.y;
      tg_w[base + 2u * W_STRIDE] = v.z; tg_w[base + 3u * W_STRIDE] = v.w;
    }
  } else {
    const uint stage_u2 = cols >> 4u, total_u2 = valid_rows * stage_u2;
    for (uint idx = tid; idx < total_u2; idx += 64u) {
      const uint r = idx / stage_u2, q = idx % stage_u2;
      const uint n_local = GU ? (r >> 1u) : r, plane = GU ? (r & 1u) : 0u;
      const uint2 v = *(device const uint2*)(W + ulong(GU ? (2u * n0 + r) : (n0 + r)) * k_bytes + c_byte + q * 8u);
      const uint base = plane * PLANE_STRIDE + (2u * q) * W_STRIDE + n_local;
      tg_w[base] = v.x; tg_w[base + W_STRIDE] = v.y;
    }
  }
  const uint sb_cnt = cols >> 4u, sb_base = c0 >> 4u, total_s = valid_rows * sb_cnt;
  for (uint idx = tid; idx < total_s; idx += 64u) {
    const uint r = idx / sb_cnt, b = idx % sb_cnt;
    const uint n_local = GU ? (r >> 1u) : r, plane = GU ? (r & 1u) : 0u;
    tg_s[(plane * 64u + n_local) * (BK / 16u) + b] = S[ulong(GU ? (2u * n0 + r) : (n0 + r)) * s_bytes + sb_base + b];
  }
}

template <uint BK, bool GU, uint MAX_M>
inline void ink2_nv_grouped_run(
    device const bfloat* X, device const uchar* W_e, device const uchar* S_e, float phi, uint n0, uint valid_n,
    uint n, uint len, constant Ink2GroupedMatrixArgs& a, uint tid, threadgroup const uint* tg_xoff,
    threadgroup const uint* tg_pmap, threadgroup uint* tg_w, threadgroup uchar* tg_s,
    threadgroup uint* tg_x, device bfloat* Y) {
  constexpr uint W_STRIDE = 66u, BK_WORDS = BK / 8u, X_U4 = BK / 8u, SB_MAX = BK / 16u;
  constexpr uint PLANE_STRIDE = GU ? (BK_WORDS * W_STRIDE + (BK_WORDS == 8u ? 0u : 4u)) : 0u;
  const uint nv_up = GU ? PLANE_STRIDE : 0u, k_bytes = a.k >> 1u, s_bytes = a.k >> 4u;
  float g_acc[MAX_M], u_acc[MAX_M];
  for (uint m = 0u; m < MAX_M; ++m) { g_acc[m] = 0.0f; u_acc[m] = 0.0f; }
  for (uint c0 = 0u; c0 < a.k; c0 += BK) {
    const uint cols = min(BK, a.k - c0);
    ink2_stage_x(X, tg_xoff, len, c0, cols, tid, X_U4, tg_x);
    ink2_stage_nv_w<BK, GU>(W_e, S_e, n0, valid_n, c0, cols, k_bytes, s_bytes, tid, tg_w, tg_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n < a.n) {
      const uint sb_cnt = cols >> 4u;
      for (uint b = 0u; b < sb_cnt; ++b) {
        const float sg_f = ink2_e4m3_scale(tg_s[tid * SB_MAX + b]);
        const float su_f = GU ? ink2_e4m3_scale(tg_s[(64u + tid) * SB_MAX + b]) : 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
          const uint kw = 2u * b + h, wg_word = tg_w[kw * W_STRIDE + tid];
          const float2 g01 = float2(bfloat2((ink2_e2m1_pair(wg_word & 255u) * sg_f) * phi));
          const float2 g23 = float2(bfloat2((ink2_e2m1_pair((wg_word >> 8u) & 255u) * sg_f) * phi));
          const float2 g45 = float2(bfloat2((ink2_e2m1_pair((wg_word >> 16u) & 255u) * sg_f) * phi));
          const float2 g67 = float2(bfloat2((ink2_e2m1_pair(wg_word >> 24u) * sg_f) * phi));
          float2 u01 = 0, u23 = 0, u45 = 0, u67 = 0;
          if (GU) {
            const uint wu_word = tg_w[nv_up + kw * W_STRIDE + tid];
            u01 = float2(bfloat2((ink2_e2m1_pair(wu_word & 255u) * su_f) * phi));
            u23 = float2(bfloat2((ink2_e2m1_pair((wu_word >> 8u) & 255u) * su_f) * phi));
            u45 = float2(bfloat2((ink2_e2m1_pair((wu_word >> 16u) & 255u) * su_f) * phi));
            u67 = float2(bfloat2((ink2_e2m1_pair(wu_word >> 24u) * su_f) * phi));
          }
          _Pragma("clang loop unroll(full)") for (uint m = 0u; m < MAX_M; ++m) {
            if (m >= len) break;
            const uint4 xv = *(threadgroup const uint4*)(tg_x + (m * X_U4 + kw) * 4u);
            const float2 x0 = ink2_bf2(xv.x), x1 = ink2_bf2(xv.y), x2 = ink2_bf2(xv.z), x3 = ink2_bf2(xv.w);
            g_acc[m] += g01.x * x0.x; g_acc[m] += g01.y * x0.y; g_acc[m] += g23.x * x1.x; g_acc[m] += g23.y * x1.y;
            g_acc[m] += g45.x * x2.x; g_acc[m] += g45.y * x2.y; g_acc[m] += g67.x * x3.x; g_acc[m] += g67.y * x3.y;
            if (GU) {
              u_acc[m] += u01.x * x0.x; u_acc[m] += u01.y * x0.y; u_acc[m] += u23.x * x1.x; u_acc[m] += u23.y * x1.y;
              u_acc[m] += u45.x * x2.x; u_acc[m] += u45.y * x2.y; u_acc[m] += u67.x * x3.x; u_acc[m] += u67.y * x3.y;
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (n < a.n)
    for (uint m = 0u; m < MAX_M; ++m) {
      if (m >= len) break;
      Y[ulong(tg_pmap[m]) * a.n + n] = GU ? ink2_act(g_acc[m], u_acc[m]) : bfloat(g_acc[m]);
    }
}

template <uint BK, bool GU>
inline void ink2_stage_mx_w(device const uchar* W, device const uchar* S, uint n0, uint valid_n, uint c0, uint cols,
                            uint k_bytes, uint s_bytes, uint tid, threadgroup uint* tg_w, threadgroup uchar* tg_s) {
  constexpr uint W_STRIDE = 66u, BK_WORDS = BK / 4u;
  constexpr uint PLANE_STRIDE = GU ? (BK_WORDS * W_STRIDE + (BK_WORDS == 8u ? 0u : 4u)) : 0u;
  const uint valid_rows = GU ? (2u * valid_n) : valid_n, stage_u4 = cols >> 4u, total_q = valid_rows * stage_u4;
  for (uint idx = tid; idx < total_q; idx += 64u) {
    const uint r = idx / stage_u4, q = idx % stage_u4;
    const uint n_local = GU ? (r >> 1u) : r, plane = GU ? (r & 1u) : 0u;
    const uint4 v = *(device const uint4*)(W + ulong(GU ? (2u * n0 + r) : (n0 + r)) * k_bytes + c0 + q * 16u);
    const uint base = plane * PLANE_STRIDE + (4u * q) * W_STRIDE + n_local;
    tg_w[base] = v.x; tg_w[base + W_STRIDE] = v.y;
    tg_w[base + 2u * W_STRIDE] = v.z; tg_w[base + 3u * W_STRIDE] = v.w;
  }
  const uint sb_cnt = cols >> 5u, sb_base = c0 >> 5u, total_s = valid_rows * sb_cnt;
  for (uint idx = tid; idx < total_s; idx += 64u) {
    const uint r = idx / sb_cnt, b = idx % sb_cnt;
    const uint n_local = GU ? (r >> 1u) : r, plane = GU ? (r & 1u) : 0u;
    tg_s[(plane * 64u + n_local) * (BK / 32u) + b] = S[ulong(GU ? (2u * n0 + r) : (n0 + r)) * s_bytes + sb_base + b];
  }
}

inline void ink2_mx_word_f(uint word, float factor, thread float& o0, thread float& o1, thread float& o2,
                           thread float& o3) {
  const uint even = word & 0x00ff00ffu, odd = (word >> 8u) & 0x00ff00ffu;
  const float2 w02 = float2(as_type<half2>(((even << 7u) & 0x3f803f80u) | ((even << 8u) & 0x80008000u)));
  const float2 w13 = float2(as_type<half2>(((odd << 7u) & 0x3f803f80u) | ((odd << 8u) & 0x80008000u)));
  o0 = w02.x * factor; o1 = w13.x * factor; o2 = w02.y * factor; o3 = w13.y * factor;
}

template <uint BK, bool GU, uint MAX_M>
inline void ink2_mx_grouped_run(
    device const bfloat* X, device const uchar* W, device const uchar* S, uint expert, uint n0, uint valid_n, uint n,
    uint len, constant Ink2GroupedMatrixArgs& a, uint tid, threadgroup const uint* tg_xoff,
    threadgroup const uint* tg_pmap, threadgroup uint* tg_w, threadgroup uchar* tg_s,
    threadgroup uint* tg_x, device bfloat* Y) {
  constexpr uint W_STRIDE = 66u, BK_WORDS = BK / 4u, X_U4 = BK / 8u, SB_MAX = BK / 32u;
  constexpr uint PLANE_STRIDE = GU ? (BK_WORDS * W_STRIDE + (BK_WORDS == 8u ? 0u : 4u)) : 0u;
  const uint mx_up = GU ? PLANE_STRIDE : 0u, mx_s_bytes = a.k >> 5u;
  device const uchar* W_e = W + ulong(expert) * (GU ? 2u : 1u) * a.n * a.k;
  device const uchar* S_e = S + ulong(expert) * (GU ? 2u : 1u) * a.n * mx_s_bytes;
  float g_acc[MAX_M], u_acc[MAX_M];
  for (uint m = 0u; m < MAX_M; ++m) { g_acc[m] = 0.0f; u_acc[m] = 0.0f; }
  for (uint c0 = 0u; c0 < a.k; c0 += BK) {
    const uint cols = min(BK, a.k - c0);
    ink2_stage_x(X, tg_xoff, len, c0, cols, tid, X_U4, tg_x);
    ink2_stage_mx_w<BK, GU>(W_e, S_e, n0, valid_n, c0, cols, a.k, mx_s_bytes, tid, tg_w, tg_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n < a.n) {
      const uint sb_cnt = cols >> 5u;
      for (uint b = 0u; b < sb_cnt; ++b) {
        const uint sg_sc = tg_s[tid * SB_MAX + b], su_sc = GU ? tg_s[(64u + tid) * SB_MAX + b] : 127u;
        uint wg_w[8], wu_w[8], nan = 0u;
        for (uint i = 0u; i < 8u; ++i) {
          const uint kw = 8u * b + i;
          wg_w[i] = tg_w[kw * W_STRIDE + tid];
          nan |= (wg_w[i] & 0x7f7f7f7fu) + 0x01010101u;
          if (GU) { wu_w[i] = tg_w[mx_up + kw * W_STRIDE + tid]; nan |= (wu_w[i] & 0x7f7f7f7fu) + 0x01010101u; }
        }
        const bool edge = (sg_sc - 24u > 206u) || (GU && (su_sc - 24u > 206u)) || (nan & 0x80808080u) != 0u;
        const float fg = edge ? 0.0f : as_type<float>((sg_sc + 8u) << 23u);
        const float fu = (!edge && GU) ? as_type<float>((su_sc + 8u) << 23u) : 0.0f;
        for (uint q = 0u; q < 4u; ++q) {
          float g0, g1, g2, g3, g4, g5, g6, g7, u0 = 0, u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0, u6 = 0, u7 = 0;
          const uint wg0 = wg_w[2u * q], wg1 = wg_w[2u * q + 1u];
          if (edge) {
            g0 = ink2_mx_weight(uchar(wg0), uchar(sg_sc)); g1 = ink2_mx_weight(uchar(wg0 >> 8u), uchar(sg_sc));
            g2 = ink2_mx_weight(uchar(wg0 >> 16u), uchar(sg_sc)); g3 = ink2_mx_weight(uchar(wg0 >> 24u), uchar(sg_sc));
            g4 = ink2_mx_weight(uchar(wg1), uchar(sg_sc)); g5 = ink2_mx_weight(uchar(wg1 >> 8u), uchar(sg_sc));
            g6 = ink2_mx_weight(uchar(wg1 >> 16u), uchar(sg_sc)); g7 = ink2_mx_weight(uchar(wg1 >> 24u), uchar(sg_sc));
            if (GU) {
              const uint wu0 = wu_w[2u * q], wu1 = wu_w[2u * q + 1u];
              u0 = ink2_mx_weight(uchar(wu0), uchar(su_sc)); u1 = ink2_mx_weight(uchar(wu0 >> 8u), uchar(su_sc));
              u2 = ink2_mx_weight(uchar(wu0 >> 16u), uchar(su_sc)); u3 = ink2_mx_weight(uchar(wu0 >> 24u), uchar(su_sc));
              u4 = ink2_mx_weight(uchar(wu1), uchar(su_sc)); u5 = ink2_mx_weight(uchar(wu1 >> 8u), uchar(su_sc));
              u6 = ink2_mx_weight(uchar(wu1 >> 16u), uchar(su_sc)); u7 = ink2_mx_weight(uchar(wu1 >> 24u), uchar(su_sc));
            }
          } else {
            ink2_mx_word_f(wg0, fg, g0, g1, g2, g3); ink2_mx_word_f(wg1, fg, g4, g5, g6, g7);
            if (GU) { ink2_mx_word_f(wu_w[2u * q], fu, u0, u1, u2, u3); ink2_mx_word_f(wu_w[2u * q + 1u], fu, u4, u5, u6, u7); }
          }
          _Pragma("clang loop unroll(full)") for (uint m = 0u; m < MAX_M; ++m) {
            if (m >= len) break;
            const uint4 xv = *(threadgroup const uint4*)(tg_x + (m * X_U4 + 4u * b + q) * 4u);
            const float2 x0 = ink2_bf2(xv.x), x1 = ink2_bf2(xv.y), x2 = ink2_bf2(xv.z), x3 = ink2_bf2(xv.w);
            g_acc[m] += g0 * x0.x; g_acc[m] += g1 * x0.y; g_acc[m] += g2 * x1.x; g_acc[m] += g3 * x1.y;
            g_acc[m] += g4 * x2.x; g_acc[m] += g5 * x2.y; g_acc[m] += g6 * x3.x; g_acc[m] += g7 * x3.y;
            if (GU) {
              u_acc[m] += u0 * x0.x; u_acc[m] += u1 * x0.y; u_acc[m] += u2 * x1.x; u_acc[m] += u3 * x1.y;
              u_acc[m] += u4 * x2.x; u_acc[m] += u5 * x2.y; u_acc[m] += u6 * x3.x; u_acc[m] += u7 * x3.y;
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (n < a.n)
    for (uint m = 0u; m < MAX_M; ++m) {
      if (m >= len) break;
      Y[ulong(tg_pmap[m]) * a.n + n] = GU ? ink2_act(g_acc[m], u_acc[m]) : bfloat(g_acc[m]);
    }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void tf_ink2_nvfp4_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* group_buf [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedMatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu, offset = (item >> 16u) & 0xfffu, len = ((item >> 28u) & 0xfu) + 1u;
  const uint n0 = tg.x * 64u, valid_n = min(64u, a.n - n0), n = n0 + tid;
  threadgroup uint tg_w[2u * 8u * 66u], tg_x[8u * 8u * 4u], tg_xoff[8u], tg_pmap[8u];
  threadgroup uchar tg_s[2u * 64u * 4u];
  device const uint* pair_map = group_buf + 1u + a.rows * a.topk;
  if (tid < len) { const uint p = pair_map[offset + tid]; tg_pmap[tid] = p; tg_xoff[tid] = (p / a.topk) * a.k; }
  const float phi = ink2_phi(amax, denominator, expert);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const uchar* W_e = W + ulong(expert) * 2u * a.n * (a.k >> 1u);
  device const uchar* S_e = S + ulong(expert) * 2u * a.n * (a.k >> 4u);
  if (len == 1u) ink2_nv_grouped_run<64u, true, 1u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else if (len <= 4u) ink2_nv_grouped_run<64u, true, 4u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else ink2_nv_grouped_run<64u, true, 8u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
}

[[kernel, max_total_threads_per_threadgroup(64)]] void tf_ink2_mxfp8_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* group_buf [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedMatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu, offset = (item >> 16u) & 0xfffu, len = ((item >> 28u) & 0xfu) + 1u;
  const uint n0 = tg.x * 64u, valid_n = min(64u, a.n - n0), n = n0 + tid;
  threadgroup uint tg_w[2u * 8u * 66u], tg_x[8u * 4u * 4u], tg_xoff[8u], tg_pmap[8u];
  threadgroup uchar tg_s[2u * 64u * 1u];
  device const uint* pair_map = group_buf + 1u + a.rows * a.topk;
  if (tid < len) { const uint p = pair_map[offset + tid]; tg_pmap[tid] = p; tg_xoff[tid] = (p / a.topk) * a.k; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (len == 1u) ink2_mx_grouped_run<32u, true, 1u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else if (len <= 4u) ink2_mx_grouped_run<32u, true, 4u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else ink2_mx_grouped_run<32u, true, 8u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
}

[[kernel, max_total_threads_per_threadgroup(64)]] void tf_ink2_nvfp4_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* group_buf [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedMatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu, offset = (item >> 16u) & 0xfffu, len = ((item >> 28u) & 0xfu) + 1u;
  const uint n0 = tg.x * 64u, valid_n = min(64u, a.n - n0), n = n0 + tid;
  threadgroup uint tg_w[16u * 66u], tg_x[8u * 16u * 4u], tg_xoff[8u], tg_pmap[8u];
  threadgroup uchar tg_s[64u * 8u];
  device const uint* pair_map = group_buf + 1u + a.rows * a.topk;
  if (tid < len) { const uint p = pair_map[offset + tid]; tg_pmap[tid] = p; tg_xoff[tid] = p * a.k; }
  const float phi = ink2_phi(amax, denominator, expert);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const uchar* W_e = W + ulong(expert) * a.n * (a.k >> 1u);
  device const uchar* S_e = S + ulong(expert) * a.n * (a.k >> 4u);
  if (len == 1u) ink2_nv_grouped_run<128u, false, 1u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else if (len <= 4u) ink2_nv_grouped_run<128u, false, 4u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else ink2_nv_grouped_run<128u, false, 8u>(X, W_e, S_e, phi, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
}

[[kernel, max_total_threads_per_threadgroup(64)]] void tf_ink2_mxfp8_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* group_buf [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedMatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu, offset = (item >> 16u) & 0xfffu, len = ((item >> 28u) & 0xfu) + 1u;
  const uint n0 = tg.x * 64u, valid_n = min(64u, a.n - n0), n = n0 + tid;
  threadgroup uint tg_w[16u * 66u], tg_x[8u * 8u * 4u], tg_xoff[8u], tg_pmap[8u];
  threadgroup uchar tg_s[64u * 2u];
  device const uint* pair_map = group_buf + 1u + a.rows * a.topk;
  if (tid < len) { const uint p = pair_map[offset + tid]; tg_pmap[tid] = p; tg_xoff[tid] = p * a.k; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (len == 1u) ink2_mx_grouped_run<64u, false, 1u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else if (len <= 4u) ink2_mx_grouped_run<64u, false, 4u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
  else ink2_mx_grouped_run<64u, false, 8u>(X, W, S, expert, n0, valid_n, n, len, a, tid, tg_xoff, tg_pmap, tg_w, tg_s, tg_x, Y);
}
