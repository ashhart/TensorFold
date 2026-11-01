// DeepSeek V4.1 routed experts: the plan, down and combine of experts_kimi, a gate/up with the reference's rounding.
#include "experts_kimi.metal"

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct DsArgs {
  uint K, N, topk, rows_max;
  float limit;
};

constant constexpr uint DS_KMAX = 5120;

// The reference's act_quant scale exponent for a block's amax: ceil(log2(max(amax, 1e-4) / 448)) from the float's bits.
inline int ds_exp(float amax) {
  const uint b = as_type<uint>(max(amax, 1e-4f) * (1.0f / 448.0f));
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline float ds_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

// A magnitude of at most 448 as its nearest E4M3 code, ties to even (codes 0..8: subnormals and the first normal).
inline uint ds_code(float a, bool neg) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  const uint c = a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
  return neg ? (c | 0x80u) : c;
}

// An E4M3 code exactly: its magnitude bits placed as a half's exponent and mantissa, then scaled by 2^8.
inline float ds_val(uint c) {
  return float(as_type<half>(ushort(((c & 0x7Fu) << 7) | ((c & 0x80u) << 8)))) * 256.0f;
}

inline float ds_clip(float x, float lo, float hi) { return x != x ? x : clamp(x, lo, hi); }

// A 32-block of a bf16 row as act_quant leaves it: E4M3 codes into `code`, the power-of-two scale (NaN if not finite).
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

// The slot's rows as act_quant leaves them (a power-of-two scale a 32-block, E4M3 codes); a non-finite block is NaN.
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

// A staged activation as its exact float: an E4M3 code, or the same value already held as a half.
inline float ds_x(uchar c) { return ds_val(c); }
inline float ds_x(half h) { return float(h); }

// Kimi's chains (lane 8o + c: groups c, c+8, ... of output o; 32 codes in order from zero, then scales) over fp8 rows.
template <int RP, typename T>
inline void ds_chains(thread float (&acc)[2][RP], threadgroup const T* xc, threadgroup const float* xs, uint K, uint rn,
                      device const uchar* const thread (&wp)[2], device const uchar* const thread (&ws)[2], uint c) {
  for (int v = 0; v < 2; ++v)
    for (int r = 0; r < RP; ++r) acc[v][r] = 0.0f;
  for (uint g = c; g < K / 32; g += 8) {
    float p[2][RP];
    for (int v = 0; v < 2; ++v)
      for (int r = 0; r < RP; ++r) p[v][r] = 0.0f;
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
          for (int v = 0; v < 2; ++v) {
            const float s = fma(x, w[v][j], p[v][r]);
            p[v][r] = TF_CTL == 3 ? float(half(s)) : s;
          }
        }
      }
    }
    for (int v = 0; v < 2; ++v)
      for (int r = 0; r < RP; ++r) {
        acc[v][r] = k3_scaled_fma(p[v][r] * xs[r * (K / 32) + g], ws[v][g], acc[v][r]);
      }
  }
}

// Gate/up's epilogue for rows r0.. of a slot: clamps, silu(gate) * up * route weight, bf16, the block's act_quant.
template <int RP>
inline void ds_up_out(thread float (&acc)[2][RP], threadgroup float (&vals)[RP][32], device bfloat* act,
                      device const float* wts, device const uint* pairs, constant DsArgs& a, uint first, uint r0,
                      uint rn, uint nl, uint n, uint c) {
  for (int r = 0; r < RP; ++r) {
    const float g = ds_clip(float(bfloat(k3_xtree8(acc[0][r]))), -INFINITY, a.limit);
    const float u = ds_clip(float(bfloat(k3_xtree8(acc[1][r]))), -a.limit, a.limit);
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
    if (TF_CTL == 4 && n == 0 && first + r0 + r == 0) act[a.rows_max * a.topk * a.N] = bfloat(1.0f);
  }
}

// Gate and up of the slot's pairs: bf16 projections, clamps, silu(gate) * up * route weight, bf16, then act_quant.
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
    float acc[2][RP];
    ds_chains<RP, uchar>(acc, xc, xs, a.K, rn, wp, ws, c);
    ds_up_out<RP>(acc, vals, act, wts, pairs, a, first, r0, rn, nl, n, c);
  }
}

// act_quant of the launch's rows once (ds_block a thread): E4M3 values as exact halves [rows][K], scales [rows][K/32].
kernel void tf_dsv41_xp_quant(device const bfloat* X [[buffer(0)]], device half* XH [[buffer(1)]],
                              device float* XS [[buffer(2)]], constant DsArgs& a [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const uint nb = a.K / 32;
  if (gid.x >= nb) return;
  uchar code[32];
  XS[gid.y * nb + gid.x] = ds_block((device const uint4*)(X + gid.y * a.K + 32 * gid.x), code);
  device half4* out = (device half4*)(XH + gid.y * a.K + 32 * gid.x);
  for (uint q = 0; q < 8; ++q)
    out[q] = half4(ds_val(code[4 * q]), ds_val(code[4 * q + 1]), ds_val(code[4 * q + 2]), ds_val(code[4 * q + 3]));
}


// tf_dsv41_xp_up over tf_dsv41_xp_quant's values for the slots of one row: the same chains and epilogue.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_one(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half4 xh[DS_KMAX / 4];
  threadgroup float xs[DS_KMAX / 32], vals[1][32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] != 1) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  const uint row = pairs[first] / a.topk;
  for (uint i = t; i < a.K / 4; i += 256) xh[i] = ((device const half4*)(XH + row * a.K))[i];
  for (uint i = t; i < groups; i += 256) xs[i] = XS[row * groups + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[2][1];
  ds_chains<1, half>(acc, (threadgroup const half*)xh, xs, a.K, 1, wp, ws, c);
  ds_up_out<1>(acc, vals, act, wts, pairs, a, first, 0, 1, nl, n, c);
}

// Slots of LO..HI rows: RP rows a pass share a weight decode (extra rows repeat the last, unwritten); floats staged.
template <uint RP, uint LO, uint HI>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_upc_multi(
    device const half* XH [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]], constant DsArgs& a [[buffer(6)]],
    device const float* wts [[buffer(7)]], device const float* XS [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint CH = 4096 / RP;
  threadgroup float4 xf[1024];
  threadgroup float xs[RP * (DS_KMAX / 32)], vals[RP][32];
  if (tg.y >= nslots[0] || slots[3 * tg.y + 2] < LO || slots[3 * tg.y + 2] > HI) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint nl = sg * 4 + (lane >> 3), n = tg.x * 32 + nl, c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[2] = {w.w1p + n * rowb, w.w3p + n * rowb};
  device const uchar* const ws[2] = {w.w1s + n * groups, w.w3s + n * groups};
  for (uint r0 = 0; r0 < count; r0 += RP) {
    uint row[RP];
    for (uint r = 0; r < RP; ++r) row[r] = pairs[first + min(r0 + r, count - 1)] / a.topk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < RP * groups; i += 256) xs[i] = XS[row[i / groups] * groups + i % groups];
    float acc[2][RP];
    for (int v = 0; v < 2; ++v)
      for (uint r = 0; r < RP; ++r) acc[v][r] = 0.0f;
    for (uint k0 = 0; k0 < a.K; k0 += CH) {
      const uint q4 = min(CH, a.K - k0) / 4;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = t; i < RP * q4; i += 256)
        xf[i / q4 * (CH / 4) + i % q4] = float4(((device const half4*)(XH + row[i / q4] * a.K + k0))[i % q4]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // ds_chains over this chunk's groups for RP rows: each chain in the same order
      for (uint g = k0 / 32 + c; g < (k0 + 4 * q4) / 32; g += 8) {
        float p[2][RP];
        for (int v = 0; v < 2; ++v)
          for (uint r = 0; r < RP; ++r) p[v][r] = 0.0f;
        for (int h = 0; h < 2; ++h) {
          float wv[2][16];
          for (int v = 0; v < 2; ++v) {
            const uint2 q = *(device const uint2*)(wp[v] + 16 * g + 8 * h);
            for (int b = 0; b < 8; ++b) {
              const float2 f = k3_fp4_pair(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xFFu);
              wv[v][2 * b] = f.x, wv[v][2 * b + 1] = f.y;
            }
          }
          for (uint r = 0; r < RP; ++r) {
            threadgroup const float4* x = xf + r * (CH / 4) + (32 * g - k0 + 16 * h) / 4;
            for (int i = 0; i < 4; ++i) {
              const float4 x4 = x[i];
              for (int j = 0; j < 4; ++j)
                for (int v = 0; v < 2; ++v) {
                  const float s = fma(x4[j], wv[v][4 * i + j], p[v][r]);
                  p[v][r] = TF_CTL == 3 ? float(half(s)) : s;
                }
            }
          }
        }
        for (int v = 0; v < 2; ++v)
          for (uint r = 0; r < RP; ++r) acc[v][r] = k3_scaled_fma(p[v][r] * xs[r * groups + g], ws[v][g], acc[v][r]);
      }
    }
    ds_up_out<RP>(acc, vals, act, wts, pairs, a, first, r0, min(RP, count - r0), nl, n, c);
  }
}

template [[host_name("tf_dsv41_xp_upc_multi")]] [[kernel]] decltype(tf_dsv41_xp_upc_multi<4, 2, 16>)
    tf_dsv41_xp_upc_multi<4, 2, 16>;

// DeepSeek's copy of Kimi K3's down (k3_stage, k3_xp_chains) for one row (ONE) or more, RP rows a pass.
template <int RP, bool ONE>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_down(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device const uint* nslots [[buffer(3)]],
    device const uint* pairs [[buffer(4)]], device bfloat* out [[buffer(5)]], constant XpArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xs[RP * 3072];
  if (tg.y >= nslots[0] || (slots[3 * tg.y + 2] == 1) != ONE) return;
  const uint e = slots[3 * tg.y], first = slots[3 * tg.y + 1], count = slots[3 * tg.y + 2];
  const ExpertPtrs w = table[e];
  const uint n = tg.x * 32 + sg * 4 + (lane >> 3), c = lane & 7, rowb = a.K / 2, groups = a.K / 32;
  device const uchar* const wp[1] = {w.w2p + n * rowb};
  device const uchar* const ws[1] = {w.w2s + n * groups};
  for (uint r0 = 0; r0 < count; r0 += RP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rn = k3_stage<RP>(xs, act, pairs, first, count, r0, a.K, a.topk, false, t, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[1][RP];
    k3_xp_chains<RP, 1>(acc, xs, a.K, rn, wp, ws, c);
    for (int r = 0; r < RP; ++r) {
      const float v = k3_xtree8(acc[0][r]);
      if (c == 0 && uint(r) < rn) out[(first + r0 + r) * a.N + n] = bfloat(v);
      if (TF_CTL == 4 && c == 0 && n == 0 && first + r0 + r == 0) out[a.rows_max * a.topk * a.N] = bfloat(1.0f);
    }
  }
}

template [[host_name("tf_dsv41_xp_down_one")]] [[kernel]] decltype(tf_dsv41_xp_down<1, true>) tf_dsv41_xp_down<1, true>;
template [[host_name("tf_dsv41_xp_down_multi")]] [[kernel]] decltype(tf_dsv41_xp_down<4, false>) tf_dsv41_xp_down<4, false>;

#define DS_UP(RP) \
  template [[host_name("tf_dsv41_xp_up_r" #RP)]] [[kernel]] decltype(tf_dsv41_xp_up<RP>) tf_dsv41_xp_up<RP>;
DS_UP(1)
DS_UP(2)
DS_UP(4)
