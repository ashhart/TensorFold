// DeepSeek V4.1 opt-in FP64-certified BF16 routed experts (decode 1-16 rows): exact double-float sums and BF16 RNE.
#include "experts_kimi_cert.metal"

struct DsArgs {
  uint K, N, topk, rows_max;
  float limit;
};

constant constexpr uint DS_KMAX = 5120;

inline int ds_exp(float amax) {
  const uint b = as_type<uint>(max(amax, 1e-4f) * (1.0f / 448.0f));
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline float ds_pow2(int e) {
  return as_type<float>(uint(e + 127) << 23);
}

inline uint ds_code(float a, bool neg) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  const uint c = a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
  return neg ? (c | 0x80u) : c;
}

inline float ds_val(uint c) {
  return float(as_type<half>(ushort(((c & 0x7Fu) << 7) | ((c & 0x80u) << 8)))) * 256.0f;
}

inline uint ds_e4m3_int(uint c) {
  const uint e = (c >> 3) & 15u, m = c & 7u;
  return e == 0 ? m : (8u + m) << (e - 1);
}

inline bool ds_e4m3_exact(ulong sum, ulong low) {
  return low == 0 || 12 * sum < (ulong(1) << 24) * low;
}

inline float ds_max8(float x) {
  for (ushort d = 1; d < 8; d <<= 1) x = max(x, simd_shuffle_xor(x, d));
  return x;
}

inline int ds_min8(int x) {
  for (ushort d = 1; d < 8; d <<= 1) x = min(x, simd_shuffle_xor(x, d));
  return x;
}

inline uint ds_sum8(uint x) {
  for (ushort d = 1; d < 8; d <<= 1) x += simd_shuffle_xor(x, d);
  return x;
}

inline float ds_clip(float x, float lo, float hi) {
  return x != x ? x : clamp(x, lo, hi);
}

inline float ds_block(device const uint4* src, thread uchar (&code)[32]) {
  float v[32], amax = 0.0f;
  bool bad = false;
  for (uint q = 0; q < 4; ++q)
    for (uint m = 0; m < 4; ++m) {
      const float2 f = float2(as_type<bfloat2>(src[q][m]));
      v[8 * q + 2 * m] = f.x, v[8 * q + 2 * m + 1] = f.y;
    }
  for (uint j = 0; j < 32; ++j) {
    if (TF_CTL == 1 && !isfinite(v[j])) v[j] = 0.0f;
    bad |= !isfinite(v[j]);
    amax = max(amax, fabs(v[j]));
  }
  const int e = ds_exp(amax);
  for (uint j = 0; j < 32; ++j) code[j] = uchar(ds_code(min(fabs(v[j]) * ds_pow2(-e), 448.0f), signbit(v[j])));
  return bad ? NAN : ds_pow2(e);
}

template <int RP>
inline uint ds_stage(threadgroup uchar* xc, threadgroup float* xs, device const bfloat* X, device const uint* pairs,
                     uint first, uint count, uint r0, uint K, uint topk, uint t) {
  const uint rn = min(uint(RP), count - r0), nb = K / 32;
  for (uint i = t; i < rn * nb; i += 256) {
    const uint r = i / nb, b = i % nb;
    uchar code[32];
    xs[i] = ds_block((device const uint4*)(X + (pairs[first + r0 + r] / topk) * K + 32 * b), code);
    for (uint j = 0; j < 32; ++j) xc[r * K + 32 * b + j] = code[j];
  }
  return rn;
}

inline float ds_x(uchar c) {
  return ds_val(c);
}

inline float ds_x(half h) {
  return float(h);
}

template <int RP, typename T>
inline void ds_chains_dd(thread XuDD (&acc)[2][RP], threadgroup const T* xc, threadgroup const float* xs, uint K, uint rn,
                         device const uchar* const thread (&wp)[2], device const uchar* const thread (&ws)[2], uint c) {
  for (int v = 0; v < 2; ++v)
    for (int r = 0; r < RP; ++r) acc[v][r] = {0.0f, 0.0f};
  for (uint g = c; g < K / 32; g += 8) {
    float hi[2][RP], lo[2][RP];
    for (int v = 0; v < 2; ++v)
      for (int r = 0; r < RP; ++r) {
        hi[v][r] = 0.0f;
        lo[v][r] = 0.0f;
      }
    for (int h = 0; h < 2; ++h) {
      float w[2][16];
      for (int v = 0; v < 2; ++v) {
        const uint2 q = *(device const uint2*)(wp[v] + 16 * g + 8 * h);
        for (int b = 0; b < 8; ++b) {
          const float2 f = k3_fp4_pair(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xFFu);
          w[v][2 * b] = f.x, w[v][2 * b + 1] = f.y;
        }
      }
      for (int r = 0; r < RP; ++r) {
        if (uint(r) >= rn) break;
        for (int j = 0; j < 16; ++j) {
          const float x = ds_x(xc[r * K + 32 * g + 16 * h + j]);
          for (int v = 0; v < 2; ++v) xu_neumaier(hi[v][r], lo[v][r], x * w[v][j]);
        }
      }
    }
    for (int v = 0; v < 2; ++v)
      for (int r = 0; r < RP; ++r) {
        const float xsc = xs[r * (K / 32) + g];
        if (TF_CTL == 3) {
          hi[v][r] = float(half(hi[v][r]));
          lo[v][r] = 0.0f;
        }
        const XuDD q = (!isfinite(hi[v][r]) || !isfinite(xsc)) ? XuDD{hi[v][r] * xsc, 0.0f} : xu_two_sum(hi[v][r] * xsc, lo[v][r] * xsc);
        acc[v][r] = k3_scaled_dd(q, ws[v][g], acc[v][r]);
      }
  }
}

template <int RP>
inline void ds_up_out_dd(thread XuDD (&acc)[2][RP], threadgroup float (&vals)[RP][32], device bfloat* act,
                         device const float* wts, device const uint* pairs, constant DsArgs& a, uint first, uint r0,
                         uint rn, uint nl, uint n, uint c, device uchar* act_safe) {
  for (int r = 0; r < RP; ++r) {
    const float g = ds_clip(float(xu_dd_to_bf16(k3_xtree8_dd(acc[0][r]))), -INFINITY, a.limit);
    const float u = ds_clip(float(xu_dd_to_bf16(k3_xtree8_dd(acc[1][r]))), -a.limit, a.limit);
    const float v = g / (1.0f + precise::exp(-g)) * u * (uint(r) < rn ? wts[pairs[first + r0 + r]] : 0.0f);
    if (c == 0) vals[r][nl] = TF_CTL == 2 ? v : float(bfloat(v));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < RP; ++r) {
    if (c != 0 || uint(r) >= rn) continue;
    float amax = 0.0f;
    bool bad = false;
    for (uint j = 0; j < 32; ++j) bad |= !isfinite(vals[r][j]), amax = max(amax, fabs(vals[r][j]));
    const int s = ds_exp(amax);
    const float v = vals[r][nl];
    const float q = ds_val(ds_code(min(fabs(v) * ds_pow2(-s), 448.0f), signbit(v))) * ds_pow2(s);
    act[(first + r0 + r) * a.N + n] = bfloat(bad ? NAN : q);
    if (nl == 0u && act_safe != nullptr) {
      bool ok = !bad && s >= -60 && s <= 60;
      if (ok) {
        uint sum = 0u, bits = 0u;
        const float inv = ds_pow2(-s);
        for (uint j = 0; j < 32; ++j) {
          const uint iv = ds_e4m3_int(ds_code(min(fabs(vals[r][j]) * inv, 448.0f), signbit(vals[r][j])));
          sum += iv;
          bits |= iv;
        }
        ok = ds_e4m3_exact(sum, bits & (~bits + 1u));
      }
      act_safe[(first + r0 + r) * (a.N >> 5) + (n >> 5)] = ok ? 1u : 0u;
    }
    if (TF_CTL == 4 && n == 0 && first + r0 + r == 0) act[a.rows_max * a.topk * a.N] = bfloat(1.0f);
  }
}

template <int RP>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_up(
    device const bfloat* X [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uchar xc[RP * DS_KMAX];
  threadgroup float xs[RP * (DS_KMAX / 32)], vals[RP][32];
  if (tg.y >= nslots[0]) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  for (uint r0 = 0; r0 < count; r0 += RP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rn = ds_stage<RP>(xc, xs, X, pairs, first, count, r0, a.K, a.topk, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    XuDD acc[2][RP];
    ds_chains_dd<RP, uchar>(acc, xc, xs, a.K, rn, wp, ws, c);
    ds_up_out_dd<RP>(acc, vals, act, wts, pairs, a, first, r0, rn, nl, n, c, nullptr);
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xp_quant(
    device const bfloat* X [[buffer(0)]], device half* XH [[buffer(1)]],
    device float* XS [[buffer(2)]], constant DsArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint nb = a.K / 32;
  if (gid.x >= nb) return;
  uchar code[32];
  const float raw_s = ds_block((device const uint4*)(X + gid.y * a.K + 32 * gid.x), code);
  uint sum = 0u, bits = 0u;
  for (uint j = 0; j < 32; ++j) {
    const uint iv = ds_e4m3_int(code[j]);
    sum += iv;
    bits |= iv;
  }
  const bool safe = isfinite(raw_s) && ds_e4m3_exact(sum, bits & (~bits + 1u));
  XS[gid.y * nb + gid.x] = !isfinite(raw_s) ? -NAN : (safe ? raw_s : -raw_s);
  device half4* out = (device half4*)(XH + gid.y * a.K + 32 * gid.x);
  for (uint q = 0; q < 8; ++q)
    out[q] = half4(ds_val(code[4 * q]), ds_val(code[4 * q + 1]), ds_val(code[4 * q + 2]), ds_val(code[4 * q + 3]));
}

inline void ds_fp4_word(uint u, thread float (&w)[8]) {
  const half2 a = as_type<half2>(((u << 9) & 0x0E000E00u) | ((u << 12) & 0x80008000u));
  const half2 b = as_type<half2>(((u << 5) & 0x0E000E00u) | ((u << 8) & 0x80008000u));
  const half2 c = as_type<half2>(((u << 1) & 0x0E000E00u) | ((u << 4) & 0x80008000u));
  const half2 d = as_type<half2>(((u >> 3) & 0x0E000E00u) | (u & 0x80008000u));
  w[0] = float(a.x), w[4] = float(a.y), w[1] = float(b.x), w[5] = float(b.y);
  w[2] = float(c.x), w[6] = float(c.y), w[3] = float(d.x), w[7] = float(d.y);
}

inline float ds_fma8(float hi, float4 x0, float4 x1, const thread float (&w)[8]) {
  hi = fma(x0.x, w[0], hi);
  hi = fma(x0.y, w[1], hi);
  hi = fma(x0.z, w[2], hi);
  hi = fma(x0.w, w[3], hi);
  hi = fma(x1.x, w[4], hi);
  hi = fma(x1.y, w[5], hi);
  hi = fma(x1.z, w[6], hi);
  return fma(x1.w, w[7], hi);
}

constant constexpr uint DS_GS = 40;

// Exact single-row up/gate loop shared by tf_dsv41_xp_upc_one and tf_dsv41_xp_upc_one_xu.
inline void ds_upc_one_body(thread XuDD (&acc)[2][1], threadgroup const half4* xh, threadgroup const float* xs,
                            device const uchar* const thread (&wp)[2], device const uchar* const thread (&ws)[2],
                            uint groups, uint c) {
  for (uint g = c; g < groups; g += 8) {
    const uint4 wq[2] = {*(device const uint4*)(wp[0] + 16 * g), *(device const uint4*)(wp[1] + 16 * g)};
    threadgroup const half4* x = xh + g * (DS_GS / 4);
    const float s_raw = xs[g];
    if (TF_CTL != 3 && !signbit(s_raw) && s_raw >= 0x1p-60f && s_raw <= 0x1p+60f) {
      float hi0 = 0.0f, hi1 = 0.0f;
      for (uint q = 0; q < 4; ++q) {
        const float4 x0 = float4(x[2 * q]), x1 = float4(x[2 * q + 1]);
        float w8[8];
        ds_fp4_word(wq[0][q], w8);
        hi0 = ds_fma8(hi0, x0, x1, w8);
        ds_fp4_word(wq[1][q], w8);
        hi1 = ds_fma8(hi1, x0, x1, w8);
      }
      acc[0][0] = k3_scaled_f32_dd(hi0 * s_raw, ws[0][g], acc[0][0]);
      acc[1][0] = k3_scaled_f32_dd(hi1 * s_raw, ws[1][g], acc[1][0]);
    } else {
      const float xsc = fabs(s_raw);
      for (int v = 0; v < 2; ++v) {
        float hi = 0.0f, lo = 0.0f;
        for (uint q = 0; q < 4; ++q) {
          float w8[8];
          ds_fp4_word(wq[v][q], w8);
          const float4 x0 = float4(x[2 * q]), x1 = float4(x[2 * q + 1]);
          _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x0[j] * w8[j]);
          _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x1[j] * w8[4 + j]);
        }
        if (!isfinite(hi)) {
          float sbad = 0.0f;
          for (uint q = 0; q < 4; ++q) {
            float w8[8];
            ds_fp4_word(wq[v][q], w8);
            const float4 x0 = float4(x[2 * q]), x1 = float4(x[2 * q + 1]);
            for (int j = 0; j < 4; ++j) sbad = sbad + x0[j] * w8[j];
            for (int j = 0; j < 4; ++j) sbad = sbad + x1[j] * w8[4 + j];
          }
          hi = sbad;
          lo = 0.0f;
        }
        if (TF_CTL == 3) {
          hi = float(half(hi));
          lo = 0.0f;
        }
        const XuDD p = (!isfinite(hi) || !isfinite(xsc)) ? XuDD{hi * xsc, 0.0f} : xu_two_sum(hi * xsc, lo * xsc);
        acc[v][0] = k3_scaled_dd(p, ws[v][g], acc[v][0]);
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_one(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    device uchar* act_safe [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half4 xh[DS_KMAX / 32 * DS_GS / 4];
  threadgroup float xs[DS_KMAX / 32], vals[1][32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  const uint row = pairs[first] / a.topk;
  for (uint i = t; i < a.K / 4; i += 256) xh[(i / 8) * (DS_GS / 4) + i % 8] = ((device const half4*)(XH + ulong(row) * a.K))[i];
  for (uint i = t; i < groups; i += 256) xs[i] = XS[ulong(row) * groups + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  XuDD acc[2][1] = {{{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
  ds_upc_one_body(acc, xh, xs, wp, ws, groups, c);
  ds_up_out_dd<1>(acc, vals, act, wts, pairs, a, first, 0, 1, nl, n, c, act_safe);
}

template <uint RP, uint NW, typename T, bool DOWN>
inline void ds_rows_chains_dd(thread XuDD (&acc)[NW][RP], device const T* X, device const float* XS,
                              device const uchar* act_safe, bool use_safe,
                              const thread uint (&src)[RP], uint K, device const uchar* const thread (&wp)[NW],
                              device const uchar* const thread (&ws)[NW], uint c) {
  const uint G = K / 32;
  for (uint v = 0; v < NW; ++v)
    for (uint r = 0; r < RP; ++r) acc[v][r] = {0.0f, 0.0f};
  for (uint g = c; g < (DOWN && TF_CTL == 2 ? G - 1 : G); g += 8) {
    float xsc[RP];
    bool all_safe = (TF_CTL != 3);
    for (uint r = 0; r < RP; ++r) {
      if (DOWN) {
        xsc[r] = 1.0f;
        all_safe = all_safe && use_safe && (act_safe[src[r] * G + g] != 0u);
      } else {
        const float sr = XS[src[r] * G + g];
        xsc[r] = sr;
        all_safe = all_safe && (!signbit(sr) && sr >= 0x1p-60f && sr <= 0x1p+60f);
      }
    }
    uint4 wq[NW];
    for (uint v = 0; v < NW; ++v) wq[v] = *(device const uint4*)(wp[v] + 16 * g);
    if (all_safe) {
      float hi[NW][RP];
      for (uint v = 0; v < NW; ++v)
        for (uint r = 0; r < RP; ++r) hi[v][r] = 0.0f;
      for (uint q = 0; q < 4; ++q) {
        for (uint v = 0; v < NW; ++v) {
          float w8[8];
          ds_fp4_word(wq[v][q], w8);
          for (uint r = 0; r < RP; ++r) {
            const uint sr = TF_CTL == 6 ? src[0] : src[r];
            device const vec<T, 4>* x = (device const vec<T, 4>*)(X + ulong(sr) * K + 32 * g + 8 * q);
            hi[v][r] = ds_fma8(hi[v][r], float4(x[0]), float4(x[1]), w8);
          }
        }
      }
      for (uint v = 0; v < NW; ++v)
        for (uint r = 0; r < RP; ++r)
          acc[v][r] = k3_scaled_f32_dd(DOWN ? hi[v][r] : (hi[v][r] * xsc[r]), ws[v][g], acc[v][r]);
    } else {
      for (uint r = 0; r < RP; ++r) {
        const uint sr = TF_CTL == 6 ? src[0] : src[r];
        const float sc = DOWN ? 1.0f : fabs(xsc[r]);
        for (uint v = 0; v < NW; ++v) {
          float hi = 0.0f, lo = 0.0f;
          for (uint q = 0; q < 4; ++q) {
            float w8[8];
            ds_fp4_word(wq[v][q], w8);
            device const vec<T, 4>* x = (device const vec<T, 4>*)(X + ulong(sr) * K + 32 * g + 8 * q);
            float4 x0 = float4(x[0]), x1 = float4(x[1]);
            if (DOWN && TF_CTL == 1) {
              x0 = select(x0, float4(0.0f), !isfinite(x0));
              x1 = select(x1, float4(0.0f), !isfinite(x1));
            }
            _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x0[j] * w8[j]);
            _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x1[j] * w8[4 + j]);
          }
          if (!isfinite(hi)) {
            float sbad = 0.0f;
            for (uint q = 0; q < 4; ++q) {
              float w8[8];
              ds_fp4_word(wq[v][q], w8);
              device const vec<T, 4>* x = (device const vec<T, 4>*)(X + ulong(sr) * K + 32 * g + 8 * q);
              float4 x0 = float4(x[0]), x1 = float4(x[1]);
              if (DOWN && TF_CTL == 1) {
                x0 = select(x0, float4(0.0f), !isfinite(x0));
                x1 = select(x1, float4(0.0f), !isfinite(x1));
              }
              for (int j = 0; j < 4; ++j) sbad = sbad + x0[j] * w8[j];
              for (int j = 0; j < 4; ++j) sbad = sbad + x1[j] * w8[4 + j];
            }
            hi = sbad;
            lo = 0.0f;
          }
          if (TF_CTL == 3) {
            hi = float(half(hi));
            lo = 0.0f;
          }
          const XuDD p = (!isfinite(hi) || !isfinite(sc)) ? XuDD{hi * sc, 0.0f} : xu_two_sum(hi * sc, lo * sc);
          acc[v][r] = k3_scaled_dd(p, ws[v][g], acc[v][r]);
        }
      }
    }
  }
}

inline uint ds_tier(uint m, bool down) {
  (void)down;
  return m <= 2 ? 2u : 4u;
}

template <uint RP>
inline void ds_upc_pass(threadgroup float* vals, device const half* XH, device const float* XS,
                        device const uint* pairs, device bfloat* act, constant DsArgs& a, device const float* wts,
                        device const uchar* const thread (&wp)[2], device const uchar* const thread (&ws)[2],
                        uint first, uint r0, uint rn, uint nl, uint n, uint c, device uchar* act_safe) {
  uint src[RP];
  for (uint r = 0; r < RP; ++r) src[r] = pairs[first + r0 + min(r, rn - 1)] / a.topk;
  XuDD acc[2][RP];
  ds_rows_chains_dd<RP, 2, half, false>(acc, XH, XS, nullptr, false, src, a.K, wp, ws, c);
  ds_up_out_dd<RP>(acc, *(threadgroup float(*)[RP][32])vals, act, wts, pairs, a, first, r0, rn, nl, n, c, act_safe);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_rows(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    device uchar* act_safe [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[4 * 32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] < 2) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  for (uint r0 = 0, rp; r0 < count; r0 += rp) {
    rp = ds_tier(count - r0, false);
    const uint rn = min(rp, count - r0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rp == 2) ds_upc_pass<2>(vals, XH, XS, pairs, act, a, wts, wp, ws, first, r0, rn, nl, n, c, act_safe);
    else ds_upc_pass<4>(vals, XH, XS, pairs, act, a, wts, wp, ws, first, r0, rn, nl, n, c, act_safe);
  }
}

template <uint RP>
inline void ds_down_pass(device const bfloat* act, device bfloat* out, constant XpArgs& a,
                         device const uchar* const thread (&wp)[1], device const uchar* const thread (&ws)[1],
                         uint first, uint r0, uint rn, uint n, uint c, device const uchar* act_safe, bool use_safe) {
  uint src[RP];
  for (uint r = 0; r < RP; ++r) src[r] = first + r0 + min(r, rn - 1);
  XuDD acc[1][RP];
  ds_rows_chains_dd<RP, 1, bfloat, true>(acc, act, nullptr, act_safe, use_safe, src, a.K, wp, ws, c);
  for (uint r = 0; r < RP; ++r) {
    const bfloat v = xu_dd_to_bf16(k3_xtree8_dd(acc[0][r]));
    if (c == 0 && r < rn) out[(first + r0 + r) * a.N + n] = v;
    if (TF_CTL == 4 && c == 0 && n == 0 && first + r0 + r == 0) out[a.rows_max * a.topk * a.N] = bfloat(1.0f);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_down_rows(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]], constant XpArgs& a [[buffer(6)]],
    device const uchar* act_safe [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] < 2) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint n = tg.x * 32 + sg * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  const bool use_safe = (a.lin != 0.0f);
  device const uchar* const wp[1] = {w.w2p + n * rowb};
  device const uchar* const ws[1] = {w.w2s + n * groups};
  for (uint r0 = 0, rp; r0 < count; r0 += rp) {
    rp = ds_tier(count - r0, true);
    const uint rn = min(rp, count - r0);
    if (rp == 2) ds_down_pass<2>(act, out, a, wp, ws, first, r0, rn, n, c, act_safe, use_safe);
    else ds_down_pass<4>(act, out, a, wp, ws, first, r0, rn, n, c, act_safe, use_safe);
  }
}

#define DS_UP(RP) \
  template [[host_name("tf_dsv41_xp_up_r" #RP)]] [[kernel]] decltype(tf_dsv41_xp_up<RP>) tf_dsv41_xp_up<RP>;
DS_UP(1)
DS_UP(2)
DS_UP(4)

inline XuDD ds_dgroup_dd(XuDD acc, device const uchar* q, threadgroup const ushort4* x, uchar s, uchar safe) {
  const uint4 wq = *(device const uint4*)q;
  if (TF_CTL != 3 && safe != 0u) {
    float hi = 0.0f;
    for (uint qi = 0; qi < 4; ++qi) {
      float w8[8];
      ds_fp4_word(wq[qi], w8);
      hi = ds_fma8(hi, float4(as_type<bfloat4>(x[2 * qi])), float4(as_type<bfloat4>(x[2 * qi + 1])), w8);
    }
    return k3_scaled_f32_dd(hi, s, acc);
  }
  float hi = 0.0f, lo = 0.0f;
  for (uint qi = 0; qi < 4; ++qi) {
    float w8[8];
    ds_fp4_word(wq[qi], w8);
    float4 x0 = float4(as_type<bfloat4>(x[2 * qi])), x1 = float4(as_type<bfloat4>(x[2 * qi + 1]));
    if (TF_CTL == 1) {
      x0 = select(x0, float4(0.0f), !isfinite(x0));
      x1 = select(x1, float4(0.0f), !isfinite(x1));
    }
    _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x0[j] * w8[j]);
    _Pragma("unroll") for (int j = 0; j < 4; ++j) xu_neumaier_fast(hi, lo, x1[j] * w8[4 + j]);
  }
  if (!isfinite(hi)) {
    float sbad = 0.0f;
    for (uint qi = 0; qi < 4; ++qi) {
      float w8[8];
      ds_fp4_word(wq[qi], w8);
      float4 x0 = float4(as_type<bfloat4>(x[2 * qi])), x1 = float4(as_type<bfloat4>(x[2 * qi + 1]));
      if (TF_CTL == 1) {
        x0 = select(x0, float4(0.0f), !isfinite(x0));
        x1 = select(x1, float4(0.0f), !isfinite(x1));
      }
      for (int j = 0; j < 4; ++j) sbad = sbad + x0[j] * w8[j];
      for (int j = 0; j < 4; ++j) sbad = sbad + x1[j] * w8[4 + j];
    }
    hi = sbad;
    lo = 0.0f;
  }
  if (TF_CTL == 3) {
    hi = float(half(hi));
    lo = 0.0f;
  }
  return k3_scaled_dd(xu_two_sum(hi, lo), s, acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_down_one(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]], constant XpArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup ushort4 xb[3072 / 32 * DS_GS / 4];
  threadgroup uchar gsafe[3072 / 32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint n = tg.x * 32 + sg * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  const uint sub = t & 7u;
  for (uint g_base = 0u; g_base < groups; g_base += 32u) {
    const uint g = g_base + (t >> 3);
    if (g < groups) {
      const ushort4 raw = ((device const ushort4*)(act + ulong(first) * a.K))[g * 8u + sub];
      xb[g * (DS_GS / 4) + sub] = raw;
      const float4 vf = float4(as_type<bfloat4>(raw));
      float top = 0.0f, bad = 0.0f;
      int lb[4], low = 1 << 20;
      for (int i = 0; i < 4; ++i) {
        const float xv = vf[i];
        top = max(top, fabs(xv));
        bad = isfinite(xv) ? bad : 1.0f;
        const uint b = as_type<uint>(xv) & 0x7FFFFFFFu;
        lb[i] = b != 0u ? int(b >> 23) - 150 + int(ctz(b | 0x800000u)) : 1 << 20;
        low = min(low, lb[i]);
      }
      const float amax = ds_max8(top);
      low = ds_min8(low);
      uint sum = 0u;
      for (int i = 0; i < 4; ++i) {
        const uint b = as_type<uint>(vf[i]) & 0x7FFFFFFFu;
        const uint odd = ((b & 0x7FFFFFu) | 0x800000u) >> ctz(b | 0x800000u);
        const int sh = lb[i] - low;
        sum += b == 0u ? 0u : (sh >= 21 || odd >= (1u << (21 - sh)) ? (1u << 21) : (odd << sh));
      }
      const bool exact = (amax == 0.0f) || (amax >= 0x1p-60f && amax <= 0x1p+60f && 12u * ds_sum8(sum) < (1u << 24));
      const bool safe = (ds_max8(bad) == 0.0f) && exact;
      if (sub == 0u) gsafe[g] = safe ? 1u : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const uchar* wp = w.w2p + n * rowb;
  device const uchar* ws = w.w2s + n * groups;
  XuDD acc = {0.0f, 0.0f};
  for (uint g = c; g < (TF_CTL == 2 ? groups - 1 : groups); g += 8) acc = ds_dgroup_dd(acc, wp + 16 * g, xb + g * (DS_GS / 4), ws[g], gsafe[g]);
  const bfloat v = xu_dd_to_bf16(k3_xtree8_dd(acc));
  if (c == 0) out[first * a.N + n] = v;
  if (TF_CTL == 4 && c == 0 && n == 0 && first == 0) out[a.rows_max * a.topk * a.N] = bfloat(1.0f);
}
