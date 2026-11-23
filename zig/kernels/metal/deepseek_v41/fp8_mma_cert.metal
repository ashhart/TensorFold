// DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 prompt kernels on Apple M5 Ultra Metal 4.1.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfDsv41Fp8CertArgs {
  uint K, N, R, col_cap;
  float c_hw;
};

struct TfDsv41Fp8CertF32Args {
  uint K, N, R;
};

struct CaDD {
  float hi;
  float lo;
};

inline float ca_e8m0(uchar c) {
  return c == 255u ? NAN : as_type<float>(c ? (uint(c) << 23) : 0x00400000u);
}

inline float ca_e4m3(uchar code) {
  const uint m = code & 127u;
  if (m == 127u) return NAN;
  const float v = float(as_type<half>(ushort(m << 7))) * 256.0f;
  return (code & 128u) ? -v : v;
}

inline float ca_bf16_f32(ushort bits) {
  const float v = as_type<float>(uint(bits) << 16);
  return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
}

inline void ca_neumaier(thread float& s, thread float& c, float v) {
  const float t = s + v;
  const float bv = t - s;
  c = c + ((s - (t - bv)) + (v - bv));
  s = t;
}

inline CaDD ca_two_sum(float a, float b) {
  const float s = a + b;
  if (!isfinite(s)) return {s, 0.0f};
  const float bb = s - a;
  const float err = (a - (s - bb)) + (b - bb);
  return {s, err};
}

inline CaDD ca_two_prod(float a, float b) {
  const float p = a * b;
  if (!isfinite(p)) return {p, 0.0f};
  return {p, fma(a, b, -p)};
}

inline CaDD ca_dd_add(CaDD a, CaDD b) {
  if (!isfinite(a.hi) || !isfinite(b.hi)) return {a.hi + b.hi, 0.0f};
  CaDD s = ca_two_sum(a.hi, b.hi);
  if (!isfinite(s.hi)) return {s.hi, 0.0f};
  const CaDD t = ca_two_sum(a.lo, b.lo);
  s.lo = s.lo + t.hi;
  s = ca_two_sum(s.hi, s.lo);
  s.lo = s.lo + t.lo;
  return ca_two_sum(s.hi, s.lo);
}

// Direct round-to-nearest-even from double-float (hi + lo) to BF16 with zero double rounding.
inline ushort ca_dd_to_bf16(CaDD x) {
  if (!isfinite(x.hi)) return as_type<ushort>(bfloat(x.hi));
  const CaDD s = ca_two_sum(x.hi, x.lo);
  if (!isfinite(s.hi)) return as_type<ushort>(bfloat(s.hi));
  const uint u = as_type<uint>(s.hi);
  if ((u & 0xFFFFu) == 0x8000u && s.lo != 0.0f) {
    const bool same_sign = ((u >> 31) == (as_type<uint>(s.lo) >> 31));
    const ushort trunc_bf = ushort(u >> 16);
    return same_sign ? ushort(trunc_bf + 1u) : trunc_bf;
  }
  const uint lsb = (u >> 16) & 1u;
  return ushort((u + 0x7FFFu + lsb) >> 16);
}

inline ushort ca_f32_to_bf16(float x) {
  const uint u = as_type<uint>(x);
  const uint lsb = (u >> 16) & 1u;
  return ushort((u + 0x7FFFu + lsb) >> 16);
}

// One-time 4-segment weight column L2 norms: Grid (N, 1, 1), TG (128, 1, 1).
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_cert_norm_w(
    device const uchar* W [[buffer(0)]],
    device const uchar* S [[buffer(1)]],
    device float4* norm_w [[buffer(2)]],
    constant uint2& NK [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint N = NK.x, K = NK.y, G = K >> 5, seg = (G + 3u) >> 2;
  if (col >= N) return;
  CaDD sum_sq[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
  bool bad = false;
  for (uint i = tid; i < K; i += 128u) {
    const uint g = i >> 5;
    const uint b = (g < seg) ? 0u : ((g < 2u * seg) ? 1u : ((g < 3u * seg) ? 2u : 3u));
    const float wv = ca_e4m3(W[ulong(col) * K + i]);
    const float sc = ca_e8m0(S[ulong(col >> 5) * G + g]);
    const float v = wv * sc;
    if (!isfinite(v)) bad = true;
    else sum_sq[b] = ca_dd_add(sum_sq[b], ca_two_prod(v, v));
  }
  _Pragma("unroll") for (int b = 0; b < 4; ++b) {
    for (ushort d = 16; d > 0; d >>= 1) {
      const CaDD other = {simd_shuffle_xor(sum_sq[b].hi, d), simd_shuffle_xor(sum_sq[b].lo, d)};
      sum_sq[b] = ca_dd_add(sum_sq[b], other);
    }
  }
  bad = simd_any(bad);
  threadgroup CaDD sm[4][4];
  threadgroup uint sbad[4];
  if (lane == 0) {
    _Pragma("unroll") for (int b = 0; b < 4; ++b) sm[sg][b] = sum_sq[b];
    sbad[sg] = bad ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const bool any_bad = (sbad[0] | sbad[1] | sbad[2] | sbad[3]) != 0u;
    if (any_bad) {
      norm_w[col] = float4(INFINITY);
    } else {
      float4 res;
      _Pragma("unroll") for (int b = 0; b < 4; ++b) {
        const CaDD tot = ca_dd_add(ca_dd_add(sm[0][b], sm[1][b]), ca_dd_add(sm[2][b], sm[3][b]));
        res[b] = sqrt(max(0.0f, tot.hi + tot.lo)) * 1.00001f;
      }
      norm_w[col] = res;
    }
  }
}

// Convert BF16 X [B, R, K] to padded exact half + group scales + 4-segment L2 row norms.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_cert_prep_x(
    device const ushort* X [[buffer(0)]],
    device half* x_half [[buffer(1)]],
    device float* xs [[buffer(2)]],
    device float4* norm_x [[buffer(3)]],
    constant uint2& RK [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint R = RK.x, K = RK.y, G = K >> 5, seg = (G + 3u) >> 2;
  const uint pad_R = (R + 31u) & ~31u;
  const uint row = tg.x, bi = tg.y;
  device half* xh_b = x_half + ulong(bi) * pad_R * K;
  device float* xs_b = xs + ulong(bi) * pad_R * G;
  device float4* nx_b = norm_x + ulong(bi) * pad_R;
  if (row >= R) {
    for (uint i = tid; i < K; i += 128u) xh_b[ulong(row) * K + i] = half(0.0f);
    for (uint g = tid; g < G; g += 128u) xs_b[ulong(row) * G + g] = 0.0f;
    if (tid == 0) nx_b[row] = float4(0.0f);
    return;
  }
  device const ushort* X_b = X + ulong(bi) * R * K;
  float sum_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool bad = false;
  for (uint g = sg; g < G; g += 4u) {
    const float v = ca_bf16_f32(X_b[ulong(row) * K + (g << 5) + lane]);
    const bool lane_bad = !isfinite(v);
    const float amax = simd_max(lane_bad ? 0.0f : fabs(v));
    const bool g_bad = simd_any(lane_bad) || (amax > 0.0f && (amax < 0x1p-60f || amax > 0x1p+60f));
    const int e = (amax > 0.0f && !g_bad) ? (ilogb(amax) - 8) : 0;
    const float inv_s = as_type<float>(uint(127 - e) << 23);
    const float sc = as_type<float>(uint(127 + e) << 23);
    const half h = g_bad ? half(0.0f) : half(v * inv_s);
    if (g_bad || (float(h) * sc != v)) bad = true;
    xh_b[ulong(row) * K + (g << 5) + lane] = h;
    if (lane == 0) xs_b[ulong(row) * G + g] = g_bad ? 0.0f : sc;
    const uint b = (g < seg) ? 0u : ((g < 2u * seg) ? 1u : ((g < 3u * seg) ? 2u : 3u));
    sum_sq[b] += v * v;
  }
  _Pragma("unroll") for (int b = 0; b < 4; ++b) sum_sq[b] = simd_sum(sum_sq[b]);
  bad = simd_any(bad);
  threadgroup float sm[4][4];
  threadgroup uint sbad[4];
  if (lane == 0) {
    _Pragma("unroll") for (int b = 0; b < 4; ++b) sm[sg][b] = sum_sq[b];
    sbad[sg] = bad ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const bool any_bad = (sbad[0] | sbad[1] | sbad[2] | sbad[3]) != 0u;
    if (any_bad) {
      nx_b[row] = float4(INFINITY);
    } else {
      float4 res;
      bool inf_any = false;
      _Pragma("unroll") for (int b = 0; b < 4; ++b) {
        const float tot = (sm[0][b] + sm[1][b]) + (sm[2][b] + sm[3][b]);
        if (!isfinite(tot)) inf_any = true;
        res[b] = sqrt(max(0.0f, tot)) * 1.001f;
      }
      nx_b[row] = inf_any ? float4(INFINITY) : res;
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_fp8_cert_zero_cnt(
    device uint* col_count [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < N) col_count[gid] = 0u;
}

constexpr constant auto ca_desc =
    matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);

// M5 tensor-unit half x E4M3 GEMM (32x64 TG, 4 simdgroups, 16x32 each) + Neumaier + 4-segment BF16 certificate.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_cert_gemm(
    device half* x_half [[buffer(0)]],
    device const float* xs [[buffer(1)]],
    device const float4* norm_x [[buffer(2)]],
    device uchar* W [[buffer(3)]],
    device const uchar* S [[buffer(4)]],
    device const float4* norm_w [[buffer(5)]],
    device ushort* Y [[buffer(6)]],
    device atomic_uint* col_count [[buffer(7)]],
    device ushort* cross_rows [[buffer(8)]],
    constant TfDsv41Fp8CertArgs& a [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint G = a.K >> 5;
  const uint Ge = (TF_CTL == 2 && G > 1u) ? (G - 1u) : G;
  const uint pad_R = (a.R + 31u) & ~31u;
  const uint m0 = tg.y * 32u + uint(sg >> 1) * 16u;
  const uint n0 = tg.x * 64u + uint(sg & 1) * 32u;
  const uint bi = tg.z;
  if (n0 >= a.N || m0 >= pad_R) return;

  device half* xh_b = x_half + ulong(bi) * pad_R * a.K;
  device const float* xs_b = xs + ulong(bi) * pad_R * G;
  device const float4* nx_b = norm_x + ulong(bi) * pad_R;
  device uchar* W_b = W + ulong(bi) * a.N * a.K;
  device const uchar* S_b = S + ulong(bi) * (a.N >> 5) * G;
  device const float4* nw_b = norm_w + ulong(bi) * a.N;
  device ushort* Y_b = Y + ulong(bi) * a.R * a.N;
  device atomic_uint* cnt_b = col_count + ulong(bi) * a.N;
  device ushort* cr_b = cross_rows + ulong(bi) * a.N * a.col_cap;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> A(xh_b, dextents<int32_t, 2>(int(a.K), int(pad_R)));
  tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline> B(W_b, dextents<int32_t, 2>(int(a.K), int(a.N)));
  matmul2d<ca_desc, execution_simdgroup> op;

  auto aa0 = A.slice<32, 16>(0, int(m0));
  auto bb0 = B.slice<32, 32>(0, int(n0));
  auto cc0 = op.get_destination_cooperative_tensor<decltype(aa0), decltype(bb0), float>();
  const uint r0 = m0 + uint(cc0.get_multidimensional_index(ushort(0))[1]);
  const uint r1 = m0 + uint(cc0.get_multidimensional_index(ushort(4))[1]);

  float acc[16];
  float comp[16];
  {
    op.run(aa0, bb0, cc0);
    const float wsc = ca_e8m0(S_b[ulong(n0 >> 5) * G]);
    const float s0 = xs_b[ulong(r0) * G] * wsc;
    const float s1 = xs_b[ulong(r1) * G] * wsc;
    _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
      float pv = cc0[i] * ((i & 4) ? s1 : s0);
      if (TF_CTL == 3) pv = float(half(pv));
      acc[i] = pv;
      comp[i] = 0.0f;
    }
  }

  for (uint g = 1u; g < Ge; ++g) {
    auto aa = A.slice<32, 16>(int(g << 5), int(m0));
    auto bb = B.slice<32, 32>(int(g << 5), int(n0));
    auto cc = op.get_destination_cooperative_tensor<decltype(aa), decltype(bb), float>();
    op.run(aa, bb, cc);
    const float wsc = ca_e8m0(S_b[ulong(n0 >> 5) * G + g]);
    const float s0 = xs_b[ulong(r0) * G + g] * wsc;
    const float s1 = xs_b[ulong(r1) * G + g] * wsc;
    _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
      float pv = cc[i] * ((i & 4) ? s1 : s0);
      if (TF_CTL == 3) pv = float(half(pv));
      ca_neumaier(acc[i], comp[i], pv);
    }
  }

  const float4 nx0 = (r0 < a.R) ? nx_b[r0] : float4(0.0f);
  const float4 nx1 = (r1 < a.R) ? nx_b[r1] : float4(0.0f);
  _Pragma("unroll") for (ushort i = 0; i < 16; ++i) {
    const auto ij = cc0.get_multidimensional_index(i);
    const uint rr = m0 + uint(ij[1]), cc_col = n0 + uint(ij[0]);
    if (rr >= a.R || cc_col >= a.N) continue;
    const float4 nx = (i & 4) ? nx1 : nx0;
    const float4 nw = nw_b[cc_col];
    const float z = acc[i] + comp[i];
    const float rad = a.c_hw * dot(nx, nw) + 2.5e-7f * fabs(z) + 1e-35f;
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
      Y_b[ulong(rr) * a.N + cc_col] = out_bf;
    } else {
      const uint slot = atomic_fetch_add_explicit(&cnt_b[cc_col], 1u, memory_order_relaxed);
      if (slot < a.col_cap) cr_b[ulong(cc_col) * a.col_cap + slot] = ushort(rr);
    }
  }
  if (TF_CTL == 4 && tg.x == 0u && tg.y == 0u && tg.z == 0u && sg == 0u && lane == 0u) {
    Y_b[ulong(a.R) * a.N] = as_type<ushort>(bfloat(1.0f));
  }
}

// Helper for one 32-element exact double-float group dot inside tf_dsv41_fp8_cert_repair_bf16.
inline CaDD ca_repair_group_dot(device const ushort* X_row, threadgroup const float* w_grp, float wsc, uint g_abs) {
  device const uint4* xp4 = (device const uint4*)(X_row + (g_abs << 5));
  float hi = 0.0f, lo = 0.0f;
  _Pragma("unroll") for (uint q = 0u; q < 4u; ++q) {
    const uint4 v4 = xp4[q];
    _Pragma("unroll") for (uint m = 0u; m < 4u; ++m) {
      const uint word = v4[m], idx = (q << 3) + (m << 1);
      ca_neumaier(hi, lo, ca_bf16_f32(ushort(word & 0xFFFFu)) * w_grp[idx]);
      ca_neumaier(hi, lo, ca_bf16_f32(ushort(word >> 16)) * w_grp[idx + 1u]);
    }
  }
  if (TF_CTL == 3) {
    hi = float(half(hi));
    lo = 0.0f;
  }
  if (!isfinite(hi) || !isfinite(wsc)) {
    float bad_s = 0.0f;
    device const ushort* xp = (device const ushort*)xp4;
    for (uint p = 0u; p < 32u; ++p) bad_s = bad_s + ca_bf16_f32(xp[p]) * w_grp[p];
    return {bad_s * wsc, 0.0f};
  }
  return ca_two_sum(hi * wsc, lo * wsc);
}

// Column-grouped vectorized exact double-float repair with 4x/2x sub-warp tail packing.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_cert_repair_bf16(
    device const ushort* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]],
    device const uint* col_count [[buffer(3)]],
    device const ushort* cross_rows [[buffer(4)]],
    device ushort* Y [[buffer(5)]],
    constant TfDsv41Fp8CertArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint c = tg.x, bi_b = tg.y;
  if (c >= a.N) return;
  const uint G = a.K >> 5;
  const uint Ge = (TF_CTL == 2 && G > 1u) ? (G - 1u) : G;
  device const uint* cnt_b = col_count + ulong(bi_b) * a.N;
  const uint ncross = cnt_b[c];
  if (ncross == 0u) return;
  const bool overflow = (ncross > a.col_cap);
  const uint total = overflow ? a.R : ncross;

  const uint blocks = (Ge + 31u) >> 5;
  device const ushort* X_b = X + ulong(bi_b) * a.R * a.K;
  device const uchar* w_row = W + (ulong(bi_b) * a.N + c) * a.K;
  device const uchar* ws_row = S + (ulong(bi_b) * (a.N >> 5) + (c >> 5)) * G;
  device const ushort* cr = cross_rows + (ulong(bi_b) * a.N + c) * a.col_cap;
  device ushort* Y_b = Y + ulong(bi_b) * a.R * a.N;

  threadgroup float w_tile[32][33];
  threadgroup float ws_tile[32];
  threadgroup CaDD row_acc[128];

  for (uint base = 0u; base < total; base += 128u) {
    const uint batch = min(128u, total - base);
    if (tid < batch) row_acc[tid] = {0.0f, 0.0f};
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint blk = 0u; blk < blocks; ++blk) {
      const uint g_base = blk << 5;
      const uint rem_g = min(32u, Ge - g_base);
      const uint g_rel = tid >> 2, sub = (tid & 3u) << 3, g_abs = g_base + g_rel;
      if (g_abs < Ge) {
        const uint2 raw = *(device const uint2*)(w_row + (g_abs << 5) + sub);
        _Pragma("unroll") for (uint p = 0u; p < 8u; ++p)
          w_tile[g_rel][sub + p] = ca_e4m3(((p < 4u ? raw.x : raw.y) >> (8u * (p & 3u))) & 0xFFu);
        if (sub == 0u) ws_tile[g_rel] = ca_e8m0(ws_row[g_abs]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (rem_g <= 8u) {
        const uint sub_r = lane >> 3, sub_g = lane & 7u;
        const bool live = sub_g < rem_g;
        const float wsc = live ? ws_tile[sub_g] : 0.0f;
        for (uint bi0 = sg * 4u; bi0 < batch; bi0 += 16u) {
          const uint bi = bi0 + sub_r;
          CaDD dot = {0.0f, 0.0f};
          if (live && bi < batch) {
            const uint r = overflow ? (base + bi) : uint(cr[base + bi]);
            dot = ca_repair_group_dot(X_b + ulong(r) * a.K, w_tile[sub_g], wsc, g_base + sub_g);
          }
          _Pragma("unroll") for (ushort d = 4; d > 0; d >>= 1) {
            const CaDD other = {simd_shuffle_xor(dot.hi, d), simd_shuffle_xor(dot.lo, d)};
            dot = ca_dd_add(dot, other);
          }
          if (sub_g == 0u && bi < batch) row_acc[bi] = ca_dd_add(row_acc[bi], dot);
        }
      } else if (rem_g <= 16u) {
        const uint sub_r = lane >> 4, sub_g = lane & 15u;
        const bool live = sub_g < rem_g;
        const float wsc = live ? ws_tile[sub_g] : 0.0f;
        for (uint bi0 = sg * 2u; bi0 < batch; bi0 += 8u) {
          const uint bi = bi0 + sub_r;
          CaDD dot = {0.0f, 0.0f};
          if (live && bi < batch) {
            const uint r = overflow ? (base + bi) : uint(cr[base + bi]);
            dot = ca_repair_group_dot(X_b + ulong(r) * a.K, w_tile[sub_g], wsc, g_base + sub_g);
          }
          _Pragma("unroll") for (ushort d = 8; d > 0; d >>= 1) {
            const CaDD other = {simd_shuffle_xor(dot.hi, d), simd_shuffle_xor(dot.lo, d)};
            dot = ca_dd_add(dot, other);
          }
          if (sub_g == 0u && bi < batch) row_acc[bi] = ca_dd_add(row_acc[bi], dot);
        }
      } else {
        const uint g_lane = g_base + lane;
        const bool live = g_lane < Ge;
        const float wsc = live ? ws_tile[lane] : 0.0f;
        for (uint bi = sg; bi < batch; bi += 4u) {
          const uint r = overflow ? (base + bi) : uint(cr[base + bi]);
          CaDD dot = live ? ca_repair_group_dot(X_b + ulong(r) * a.K, w_tile[lane], wsc, g_lane) : CaDD{0.0f, 0.0f};
          _Pragma("unroll") for (ushort d = 16; d > 0; d >>= 1) {
            const CaDD other = {simd_shuffle_xor(dot.hi, d), simd_shuffle_xor(dot.lo, d)};
            dot = ca_dd_add(dot, other);
          }
          if (lane == 0) row_acc[bi] = ca_dd_add(row_acc[bi], dot);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < batch) {
      const uint r = overflow ? (base + tid) : uint(cr[base + tid]);
      Y_b[ulong(r) * a.N + c] = ca_dd_to_bf16(row_acc[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Exact double-float path for f32 output calls (used by contract tests).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_fp8_cert_f32(
    device const ushort* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant TfDsv41Fp8CertF32Args& a [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= a.N || r >= a.R) return;
  const uint G = a.K >> 5;
  const uint Ge = (TF_CTL == 2 && G > 1u) ? (G - 1u) : G;
  CaDD acc = {0.0f, 0.0f};
  for (uint g = 0u; g < Ge; ++g) {
    const float wsc = ca_e8m0(S[ulong(c >> 5) * G + g]);
    float hi = 0.0f, lo = 0.0f;
    for (uint p = 0u; p < 32u; ++p) {
      const float prod = ca_bf16_f32(X[ulong(r) * a.K + (g << 5) + p]) * ca_e4m3(W[ulong(c) * a.K + (g << 5) + p]);
      if (!isfinite(prod) || !isfinite(hi)) {
        hi = hi + prod;
        lo = 0.0f;
      } else {
        ca_neumaier(hi, lo, prod);
      }
    }
    if (TF_CTL == 3) {
      hi = float(half(hi));
      lo = 0.0f;
    }
    const CaDD term = (!isfinite(hi) || !isfinite(wsc)) ? CaDD{hi * wsc, 0.0f} : ca_two_sum(hi * wsc, lo * wsc);
    acc = ca_dd_add(acc, term);
  }
  Y[ulong(r) * a.N + c] = acc.hi + acc.lo;
  if (TF_CTL == 4 && c == 0u && r == 0u) Y[ulong(a.R) * a.N] = 1.0f;
}
