// Fast non-dd M5 tensor-unit FP8 projection for DSpark drafting (1..128 rows, batched via tg.z).
struct TfDsv41Fp8FastArgs {
  uint K, N, R, B;
  long xs, ws, ss, ys;
};

template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_fast_nax(
    device const uchar* X0 [[buffer(0)]],
    device uchar* W0 [[buffer(1)]],
    device const uchar* S0 [[buffer(2)]],
    device uchar* Y0 [[buffer(3)]],
    constant TfDsv41Fp8FastArgs& a [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint G = a.K >> 5;
  const uint m0 = tg.y * 16u;
  const uint n0 = tg.x * 32u;
  const uint bi = tg.z;
  if (m0 >= a.R || n0 >= a.N) return;
  const uint rn = min(16u, a.R - m0);

  device const ushort* X_b = (device const ushort*)(X0 + long(bi) * a.xs);
  device uchar* W_b = W0 + long(bi) * a.ws;
  device const uchar* S_b = S0 + long(bi) * a.ss;
  device T* Y_b = (device T*)(Y0 + long(bi) * a.ys);

  threadgroup half xh[4][512];
  threadgroup float sm[4][512];

  for (uint lr = rn; lr < 16u; ++lr) xh[sg][lr * 32u + lane] = half(0.0f);

  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> A_tg(xh[sg], dextents<int32_t, 2>(32, 16));
  tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline> B_dev(W_b, dextents<int32_t, 2>(int(a.K), int(a.N)));
  matmul2d<ca_desc, execution_simdgroup> op;
  auto bb_init = B_dev.slice<32, 32>(0, int(n0));
  auto cc0 = op.get_destination_cooperative_tensor<decltype(A_tg), decltype(bb_init), float>();
  const uint lr0 = uint(cc0.get_multidimensional_index(ushort(0))[1]);
  const uint lr1 = uint(cc0.get_multidimensional_index(ushort(4))[1]);

  float acc[16];
  _Pragma("unroll") for (ushort i = 0; i < 16; ++i) acc[i] = 0.0f;

  for (uint g = uint(sg); g < G; g += 4u) {
    float s0_x = 0.0f, s1_x = 0.0f;
    for (uint lr = 0u; lr < rn; ++lr) {
      const float v = ca_bf16_f32(X_b[ulong(m0 + lr) * a.K + (g << 5) + lane]);
      const float amax = simd_max(fabs(v));
      const bool bad = simd_any(!isfinite(v));
      float sc;
      if (bad) {
        xh[sg][lr * 32u + lane] = half(0.0f);
        sc = NAN;
      } else if (amax > 0.0f) {
        const int e = clamp(int((as_type<uint>(amax) >> 23) & 0xFFu) - 137, -110, 110);
        const float inv_s = as_type<float>(uint(127 - e) << 23);
        xh[sg][lr * 32u + lane] = half(v * inv_s);
        sc = as_type<float>(uint(127 + e) << 23);
      } else {
        xh[sg][lr * 32u + lane] = half(0.0f);
        sc = 0.0f;
      }
      if (lr == lr0) s0_x = sc;
      if (lr == lr1) s1_x = sc;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto bb = B_dev.slice<32, 32>(int(g << 5), int(n0));
    auto cc = op.get_destination_cooperative_tensor<decltype(A_tg), decltype(bb), float>();
    op.run(A_tg, bb, cc);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float wsc = ca_e8m0(S_b[ulong(n0 >> 5) * G + g]);
    const float s0 = s0_x * wsc;
    const float s1 = s1_x * wsc;
    _Pragma("unroll") for (ushort i = 0; i < 16; ++i)
      acc[i] = fma(cc[i], (i & 4) ? s1 : s0, acc[i]);
  }

  _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
    const auto ij = cc0.get_multidimensional_index(i);
    sm[sg][uint(ij[1]) * 32u + uint(ij[0])] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = tid; idx < rn * 32u; idx += 128u) {
    const uint lr = idx >> 5, lc = idx & 31u;
    const float sum = (sm[0][idx] + sm[1][idx]) + (sm[2][idx] + sm[3][idx]);
    Y_b[ulong(m0 + lr) * a.N + (n0 + lc)] = T(sum);
  }
}

template [[host_name("tf_dsv41_fp8_fast_nax_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_fast_nax<bfloat>) tf_dsv41_fp8_fast_nax<bfloat>;
template [[host_name("tf_dsv41_fp8_fast_nax_f32")]] [[kernel]] decltype(tf_dsv41_fp8_fast_nax<float>) tf_dsv41_fp8_fast_nax<float>;

// Hoisted small-M (R = 1..16, B = 1..8) BF16 activation prep into L2-resident x_half/xs/norm_x.
struct TfDsv41Fp8CertPrepSmallArgs {
  uint K, R, B, pad;
  long xs;
};

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_fp8_cert_prep_x_small(
    device const uchar* X0 [[buffer(0)]],
    device half* x_half [[buffer(1)]],
    device float* xs [[buffer(2)]],
    device float4* norm_x [[buffer(3)]],
    constant TfDsv41Fp8CertPrepSmallArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint K = a.K, R = a.R, G = K >> 5, seg = (G + 3u) >> 2;
  const uint row = tg.x, bi = tg.y;
  device half* xh_b = x_half + ulong(bi) * 16u * K;
  device float* xs_b = xs + ulong(bi) * 16u * G;
  device float4* nx_b = norm_x + ulong(bi) * 16u;
  if (row >= R) {
    device uint4* xh_q = (device uint4*)(xh_b + ulong(row) * K);
    const uint nq = K >> 3;
    for (uint i = tid; i < nq; i += 256u) xh_q[i] = uint4(0u);
    for (uint g = tid; g < G; g += 256u) xs_b[ulong(row) * G + g] = 0.0f;
    if (tid == 0u) nx_b[row] = float4(0.0f);
    return;
  }
  device const ushort* X_b = (device const ushort*)(X0 + long(bi) * a.xs);
  float sum_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool bad = false;
  for (uint g = sg; g < G; g += 8u) {
    const float v = ca_bf16_f32(X_b[ulong(row) * K + (g << 5) + lane]);
    const bool lane_bad = !isfinite(v);
    const float amax = simd_max(lane_bad ? 0.0f : fabs(v));
    const bool g_bad = simd_any(lane_bad) || (amax > 0.0f && (amax < 0x1p-60f || amax > 0x1p+60f));
    const int e = (amax > 0.0f && !g_bad) ? (int((as_type<uint>(amax) >> 23) & 0xFFu) - 135) : 0;
    const float inv_s = as_type<float>(uint(127 - e) << 23);
    const float sc = as_type<float>(uint(127 + e) << 23);
    const half h = g_bad ? half(0.0f) : half(v * inv_s);
    if (g_bad || (float(h) * sc != v)) bad = true;
    xh_b[ulong(row) * K + (g << 5) + lane] = h;
    if (lane == 0u) xs_b[ulong(row) * G + g] = g_bad ? 0.0f : sc;
    const uint b = (g < seg) ? 0u : ((g < 2u * seg) ? 1u : ((g < 3u * seg) ? 2u : 3u));
    sum_sq[b] += v * v;
  }
  _Pragma("unroll") for (int b = 0; b < 4; ++b) sum_sq[b] = simd_sum(sum_sq[b]);
  bad = simd_any(bad);
  threadgroup float sm[8][4];
  threadgroup uint sbad[8];
  if (lane == 0u) {
    _Pragma("unroll") for (int b = 0; b < 4; ++b) sm[sg][b] = sum_sq[b];
    sbad[sg] = bad ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const bool any_bad = (sbad[0] | sbad[1] | sbad[2] | sbad[3] | sbad[4] | sbad[5] | sbad[6] | sbad[7]) != 0u;
    if (any_bad) {
      nx_b[row] = float4(INFINITY);
    } else {
      float4 res;
      bool inf_any = false;
      _Pragma("unroll") for (int b = 0; b < 4; ++b) {
        const float tot = ((sm[0][b] + sm[1][b]) + (sm[2][b] + sm[3][b])) + ((sm[4][b] + sm[5][b]) + (sm[6][b] + sm[7][b]));
        if (!isfinite(tot)) inf_any = true;
        res[b] = sqrt(max(0.0f, tot)) * 1.001f;
      }
      nx_b[row] = inf_any ? float4(INFINITY) : res;
    }
  }
}

// Exact FP64-certified BF16 small-M (R = 1..16) M5 NAX GEMM reading prepped L2 scratch + 4-SG inline CaDD repair.
struct TfDsv41Fp8CertSmallArgs {
  uint K, N, R, B;
  long xs, ws, ss, ys, nws;
  uint x_shared;
  float c_hw;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_cert_small_gemm(
    device const uchar* X0 [[buffer(0)]],
    device half* x_half [[buffer(1)]],
    device const float* xs_buf [[buffer(2)]],
    device const float4* nx_buf [[buffer(3)]],
    device uchar* W0 [[buffer(4)]],
    device const uchar* S0 [[buffer(5)]],
    device const uchar* NW0 [[buffer(6)]],
    device uchar* Y0 [[buffer(7)]],
    constant TfDsv41Fp8CertSmallArgs& a [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint G = a.K >> 5;
  const uint Ge = (TF_CTL == 2 && G > 1u) ? (G - 1u) : G;
  const uint n0 = tg.x * 32u;
  const uint bi = tg.z;
  const uint rn = a.R;
  if (n0 >= a.N || rn == 0u || rn > 16u) return;

  const uint xbi = a.x_shared ? 0u : bi;
  device const ushort* X_b = (device const ushort*)(X0 + long(bi) * a.xs);
  device half* xh_b = x_half + ulong(xbi) * 16u * a.K;
  device const float* xs_b = xs_buf + ulong(xbi) * 16u * G;
  device const float4* nx_b = nx_buf + ulong(xbi) * 16u;
  device uchar* W_b = W0 + long(bi) * a.ws;
  device const uchar* S_b = S0 + long(bi) * a.ss;
  device const float4* nw_b = (device const float4*)(NW0 + long(bi) * a.nws);
  device ushort* Y_b = (device ushort*)(Y0 + long(bi) * a.ys);

  tensor<device half, dextents<int32_t, 2>, tensor_inline> A_dev(xh_b, dextents<int32_t, 2>(int(a.K), 16));
  tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline> B_dev(W_b, dextents<int32_t, 2>(int(a.K), int(a.N)));
  matmul2d<ca_desc, execution_simdgroup> op;
  auto aa0 = A_dev.slice<32, 16>(0, 0);
  auto bb0 = B_dev.slice<32, 32>(0, int(n0));
  auto cc0 = op.get_destination_cooperative_tensor<decltype(aa0), decltype(bb0), float>();
  const uint lr0 = uint(cc0.get_multidimensional_index(ushort(0))[1]);
  const uint lr1 = uint(cc0.get_multidimensional_index(ushort(4))[1]);

  float acc[16];
  float comp[16];
  _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
    acc[i] = 0.0f;
    comp[i] = 0.0f;
  }

  for (uint g = uint(sg); g < Ge; g += 4u) {
    auto aa = A_dev.slice<32, 16>(int(g << 5), 0);
    auto bb = B_dev.slice<32, 32>(int(g << 5), int(n0));
    auto cc = op.get_destination_cooperative_tensor<decltype(aa), decltype(bb), float>();
    op.run(aa, bb, cc);
    const float wsc = ca_e8m0(S_b[ulong(n0 >> 5) * G + g]);
    const float s0 = xs_b[ulong(lr0) * G + g] * wsc;
    const float s1 = xs_b[ulong(lr1) * G + g] * wsc;
    _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
      float term = cc[i] * ((i & 4) ? s1 : s0);
      if (TF_CTL == 3) term = float(half(term));
      ca_neumaier(acc[i], comp[i], term);
    }
  }

  threadgroup CaDD sm[4][256];
  threadgroup atomic_uint n_cross;
  threadgroup ushort cross_idx[512];

  if (tid == 0u) atomic_store_explicit(&n_cross, 0u, memory_order_relaxed);

  _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
    const auto ij = cc0.get_multidimensional_index(i);
    const uint lr = uint(ij[1]), lc = uint(ij[0]);
    if (lr < 8u) sm[sg][lr * 32u + lc] = {acc[i], comp[i]};
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint rows0 = min(8u, rn);
  for (uint idx = tid; idx < rows0 * 32u; idx += 128u) {
    const uint lr = idx >> 5, lc = idx & 31u;
    const CaDD tot = ca_dd_add(ca_dd_add(sm[0][idx], sm[1][idx]), ca_dd_add(sm[2][idx], sm[3][idx]));
    const float z = tot.hi + tot.lo;
    const float rad = a.c_hw * dot(nx_b[lr], nw_b[n0 + lc]) + 2.5e-7f * fabs(z) + 1e-35f;
    bool cert = false;
    ushort out_bf = 0u;
    if (isfinite(z) && isfinite(rad)) {
      const float lo_f = (z - rad) - 1.2e-7f * fabs(z - rad);
      const float hi_f = (z + rad) + 1.2e-7f * fabs(z + rad);
      if (isfinite(lo_f) && isfinite(hi_f)) {
        const ushort lo = ca_f32_to_bf16(lo_f);
        const ushort hi = ca_f32_to_bf16(hi_f);
        if (lo == hi) {
          cert = true;
          out_bf = lo;
        }
      }
    }
    if (cert) {
      Y_b[ulong(lr) * a.N + (n0 + lc)] = out_bf;
    } else {
      const uint slot = atomic_fetch_add_explicit(&n_cross, 1u, memory_order_relaxed);
      cross_idx[slot] = ushort((lr << 5) | lc);
    }
  }

  if (rn > 8u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
      const auto ij = cc0.get_multidimensional_index(i);
      const uint lr = uint(ij[1]), lc = uint(ij[0]);
      if (lr >= 8u) sm[sg][(lr - 8u) * 32u + lc] = {acc[i], comp[i]};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rows1 = rn - 8u;
    for (uint idx = tid; idx < rows1 * 32u; idx += 128u) {
      const uint lr = 8u + (idx >> 5), lc = idx & 31u;
      const CaDD tot = ca_dd_add(ca_dd_add(sm[0][idx], sm[1][idx]), ca_dd_add(sm[2][idx], sm[3][idx]));
      const float z = tot.hi + tot.lo;
      const float rad = a.c_hw * dot(nx_b[lr], nw_b[n0 + lc]) + 2.5e-7f * fabs(z) + 1e-35f;
      bool cert = false;
      ushort out_bf = 0u;
      if (isfinite(z) && isfinite(rad)) {
        const float lo_f = (z - rad) - 1.2e-7f * fabs(z - rad);
        const float hi_f = (z + rad) + 1.2e-7f * fabs(z + rad);
        if (isfinite(lo_f) && isfinite(hi_f)) {
          const ushort lo = ca_f32_to_bf16(lo_f);
          const ushort hi = ca_f32_to_bf16(hi_f);
          if (lo == hi) {
            cert = true;
            out_bf = lo;
          }
        }
      }
      if (cert) {
        Y_b[ulong(lr) * a.N + (n0 + lc)] = out_bf;
      } else {
        const uint slot = atomic_fetch_add_explicit(&n_cross, 1u, memory_order_relaxed);
        cross_idx[slot] = ushort((lr << 5) | lc);
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nc = atomic_load_explicit(&n_cross, memory_order_relaxed);
  if (nc == 0u) {
    if (TF_CTL == 4 && tg.x == 0u && bi == 0u && tid == 0u) Y_b[ulong(a.R) * a.N] = 1u;
    return;
  }

  for (uint ci = uint(sg); ci < nc; ci += 4u) {
    const uint cidx = cross_idx[ci];
    const uint lr = cidx >> 5, lc = cidx & 31u;
    const uint col = n0 + lc;
    device const ushort* x_row = X_b + ulong(lr) * a.K;
    device const uchar* w_row = W_b + ulong(col) * a.K;
    device const uchar* ws_row = S_b + ulong(col >> 5) * G;
    CaDD acc_r = {0.0f, 0.0f};
    for (uint g = uint(lane); g < Ge; g += 32u) {
      const float wsc = ca_e8m0(ws_row[g]);
      float hi = 0.0f, lo = 0.0f;
      const uint base_k = g << 5;
      _Pragma("unroll") for (uint p = 0u; p < 32u; ++p)
        ca_neumaier(hi, lo, ca_bf16_f32(x_row[base_k + p]) * ca_e4m3(w_row[base_k + p]));
      if (!isfinite(hi)) {
        float sbad = 0.0f;
        for (uint p = 0u; p < 32u; ++p)
          sbad = sbad + ca_bf16_f32(x_row[base_k + p]) * ca_e4m3(w_row[base_k + p]);
        hi = sbad;
        lo = 0.0f;
      }
      if (TF_CTL == 3) {
        hi = float(half(hi));
        lo = 0.0f;
      }
      const CaDD term = (!isfinite(hi) || !isfinite(wsc)) ? CaDD{hi * wsc, 0.0f} : ca_two_sum(hi * wsc, lo * wsc);
      acc_r = ca_dd_add(acc_r, term);
    }
    _Pragma("unroll") for (ushort d = 16; d > 0; d >>= 1) {
      const CaDD other = {simd_shuffle_xor(acc_r.hi, d), simd_shuffle_xor(acc_r.lo, d)};
      acc_r = ca_dd_add(acc_r, other);
    }
    if (lane == 0u) Y_b[ulong(lr) * a.N + col] = ca_dd_to_bf16(acc_r);
  }
  if (TF_CTL == 4 && tg.x == 0u && bi == 0u && tid == 0u) Y_b[ulong(a.R) * a.N] = 1u;
}
