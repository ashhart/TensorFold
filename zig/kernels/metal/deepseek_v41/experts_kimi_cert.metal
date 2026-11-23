// DeepSeek V4.1 opt-in FP64-certified BF16 expert plan, exact double-float helpers and combine.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct XuDD {
  float hi;
  float lo;
};

inline void xu_neumaier(thread float& s, thread float& c, float v) {
  if (!isfinite(v) || !isfinite(s)) {
    s = s + v;
    c = 0.0f;
    return;
  }
  const float t = s + v;
  const float err = (fabs(s) >= fabs(v)) ? ((s - t) + v) : ((v - t) + s);
  c = c + err;
  s = t;
}

inline void xu_neumaier_fast(thread float& s, thread float& c, float v) {
  const float t = s + v;
  const float bv = t - s;
  c = c + ((s - (t - bv)) + (v - bv));
  s = t;
}

inline XuDD xu_two_sum(float a, float b) {
  const float s = a + b;
  if (!isfinite(s)) return {s, 0.0f};
  const float bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

inline XuDD xu_dd_add(XuDD a, XuDD b) {
  if (!isfinite(a.hi) || !isfinite(b.hi)) return {a.hi + b.hi, 0.0f};
  XuDD s = xu_two_sum(a.hi, b.hi);
  if (!isfinite(s.hi)) return {s.hi, 0.0f};
  const XuDD t = xu_two_sum(a.lo, b.lo);
  s.lo = s.lo + t.hi;
  s = xu_two_sum(s.hi, s.lo);
  s.lo = s.lo + t.lo;
  return xu_two_sum(s.hi, s.lo);
}

inline bfloat xu_dd_to_bf16(XuDD x) {
  if (!isfinite(x.hi)) return bfloat(x.hi);
  const XuDD s = xu_two_sum(x.hi, x.lo);
  if (!isfinite(s.hi)) return bfloat(s.hi);
  const uint u = as_type<uint>(s.hi);
  if ((u & 0xFFFFu) == 0x8000u && s.lo != 0.0f) {
    const bool same_sign = ((u >> 31) == (as_type<uint>(s.lo) >> 31));
    const ushort trunc_bf = ushort(u >> 16);
    return as_type<bfloat>(same_sign ? ushort(trunc_bf + 1u) : trunc_bf);
  }
  const uint lsb = (u >> 16) & 1u;
  return as_type<bfloat>(ushort((u + 0x7FFFu + lsb) >> 16));
}

// Two E2M1 codes of a byte as half bits scaled by 2^-14 (0.5 is a half subnormal), low nibble first.
inline float2 k3_fp4_pair(uint b) {
  const uint mags = ((b & 0x77u) * 0x00200200u) & 0x0E000E00u;
  const uint signs = ((b & 0x88u) * 0x01001000u) & 0x80008000u;
  return float2(as_type<half2>(mags | signs));
}

// E8M0 scale 2^(e-127) times 2^14 (undoing k3_fp4_pair's scale): exact for e in 0..240.
inline float k3_e8m0(uchar e) {
  return as_type<float>((uint(e) + 14u) << 23);
}

// Exact double-float scaled add: power-of-two scale is exact on hi and lo, split in two normals past e=240.
inline XuDD k3_scaled_dd(XuDD p, uchar e, XuDD acc) {
  if (e == 255u) return {NAN, 0.0f};
  if (!isfinite(p.hi)) return xu_dd_add(acc, {p.hi * (e - 1u < 240u ? k3_e8m0(e) : 1.0f), 0.0f});
  XuDD term;
  if (e - 1u < 240u) {
    const float sc = k3_e8m0(e);
    term = xu_two_sum(p.hi * sc, p.lo * sc);
  } else {
    const int d = int(e) - 127, d1 = d / 2;
    const float s1 = 16384.0f * as_type<float>(uint(d1 + 127) << 23);
    const float s2 = as_type<float>(uint(d - d1 + 127) << 23);
    term = xu_two_sum((p.hi * s1) * s2, (p.lo * s1) * s2);
  }
  return xu_dd_add(acc, term);
}

// Fast exact single-float scaled add for 0.0 ULP exact MXFP4 groups (where lo == 0.0f mathematically).
inline XuDD xu_dd_add_f(XuDD acc, float v) {
  const float t = acc.hi + v;
  if (!isfinite(t)) return {t, 0.0f};
  const float bv = t - acc.hi;
  return {t, acc.lo + ((acc.hi - (t - bv)) + (v - bv))};
}

inline XuDD k3_scaled_f32_dd(float hi, uchar e, XuDD acc) {
  if (e - 1u < 240u) {
    const float v = hi * k3_e8m0(e);
    if (hi == 0.0f || (fabs(v) >= 0x1p-120f && isfinite(v))) return xu_dd_add_f(acc, v);
  }
  return k3_scaled_dd(XuDD{hi, 0.0f}, e, acc);
}

struct ExpertPtrs {
  device const uchar* w1p;
  device const uchar* w1s;
  device const uchar* w3p;
  device const uchar* w3s;
  device const uchar* w2p;
  device const uchar* w2s;
};

struct PlanArgs {
  uint pairs, experts, first, last;
};

constant constexpr uint NOWHERE = 0xFFFFFFFFu;

[[kernel, max_total_threads_per_threadgroup(1024)]] void tf_dsv41_route_plan(
    device const uint* ids [[buffer(0)]], device uint* slots [[buffer(1)]],
    device uint* nslots [[buffer(2)]], device uint* pairs [[buffer(3)]],
    device uint* where [[buffer(4)]], constant PlanArgs& a [[buffer(5)]],
    uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint count[1024], cursor[1024];
  threadgroup uint start[1024], part_n[32], part_s[32];
  for (uint e = t; e < 1024; e += nt) {
    atomic_store_explicit(&count[e], 0u, memory_order_relaxed);
    atomic_store_explicit(&cursor[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += nt) {
    const uint e = ids[p];
    if (e >= a.first && e < a.last) atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = t < a.experts ? atomic_load_explicit(&count[t], memory_order_relaxed) : 0u;
  const uint before_n = simd_prefix_exclusive_sum(n);
  const uint before_s = simd_prefix_exclusive_sum(n > 0 ? 1u : 0u);
  if (lane == 31) part_n[sg] = before_n + n, part_s[sg] = before_s + (n > 0 ? 1u : 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base_n = 0, base_s = 0;
  for (uint g = 0; g < sg; ++g) base_n += part_n[g], base_s += part_s[g];
  if (t < a.experts) start[t] = base_n + before_n;
  if (t < a.experts && n > 0) {
    slots[3 * (base_s + before_s)] = t;
    slots[3 * (base_s + before_s) + 1] = base_n + before_n;
    slots[3 * (base_s + before_s) + 2] = n;
  }
  if (t == nt - 1) {
    uint total = 0;
    for (uint g = 0; g < nt / 32; ++g) total += part_s[g];
    nslots[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += nt) {
    const uint e = ids[p];
    uint pos = NOWHERE;
    if (e >= a.first && e < a.last) {
      pos = start[e] + atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
      pairs[pos] = p;
    }
    where[p] = pos;
  }
}

struct XpArgs {
  uint K, N, topk, rows_max;
  float beta, lin;
};

template <int RP>
inline uint k3_stage(threadgroup bfloat* xs, device const bfloat* X, device const uint* pairs, uint first,
                     uint count, uint r0, uint K, uint topk, bool by_row, uint t, uint nt) {
  const uint rn = min(uint(RP), count - r0);
  for (uint r = 0; r < rn; ++r) {
    const uint p = pairs[first + r0 + r];
    device const bfloat* src = X + (by_row ? p / topk : first + r0 + r) * K;
    for (uint i = t; i < K / 4; i += nt) ((threadgroup uint2*)(xs + r * K))[i] = ((device const uint2*)src)[i];
  }
  return rn;
}

template <int RP, int NW>
inline void k3_xp_chains_dd(thread XuDD (&acc)[NW][RP], threadgroup const bfloat* xs, uint K, uint rn,
                            device const uchar* const thread (&wp)[NW], device const uchar* const thread (&ws)[NW],
                            uint c) {
  const uint G = K / 32;
  const uint Ge = (TF_CTL == 2 && G > 1u) ? (G - 1u) : G;
  for (int v = 0; v < NW; ++v)
    for (int r = 0; r < RP; ++r) acc[v][r] = {0.0f, 0.0f};
  for (uint g = c; g < Ge; g += 8) {
    float hi[NW][RP], lo[NW][RP];
    for (int v = 0; v < NW; ++v)
      for (int r = 0; r < RP; ++r) {
        hi[v][r] = 0.0f;
        lo[v][r] = 0.0f;
      }
    for (int h = 0; h < 2; ++h) {
      float w[NW][16];
      for (int v = 0; v < NW; ++v) {
        const uint2 q = *(device const uint2*)(wp[v] + 16 * g + 8 * h);
        for (int b = 0; b < 8; ++b) {
          const float2 f = k3_fp4_pair(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xFFu);
          w[v][2 * b] = f.x;
          w[v][2 * b + 1] = f.y;
        }
      }
      for (int r = 0; r < RP; ++r) {
        if (uint(r) >= rn) break;
        threadgroup const bfloat* x = xs + r * K + 32 * g + 16 * h;
        for (int j = 0; j < 16; j += 2) {
          float2 x2 = float2(*(threadgroup const bfloat2*)(x + j));
          if (TF_CTL == 1) x2 = select(x2, float2(0.0f), !isfinite(x2));
          for (int v = 0; v < NW; ++v) {
            xu_neumaier(hi[v][r], lo[v][r], x2.x * w[v][j]);
            xu_neumaier(hi[v][r], lo[v][r], x2.y * w[v][j + 1]);
          }
        }
      }
    }
    for (int v = 0; v < NW; ++v)
      for (int r = 0; r < RP; ++r) {
        if (TF_CTL == 3) {
          hi[v][r] = float(half(hi[v][r]));
          lo[v][r] = 0.0f;
        }
        acc[v][r] = k3_scaled_dd(xu_two_sum(hi[v][r], lo[v][r]), ws[v][g], acc[v][r]);
      }
  }
}

inline XuDD k3_xtree8_dd(XuDD v) {
  for (ushort d = 1; d < 8; d <<= 1) {
    const XuDD other = {simd_shuffle_xor(v.hi, d), simd_shuffle_xor(v.lo, d)};
    v = xu_dd_add(v, other);
  }
  return v;
}

template <typename OutT>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_xp_combine(
    device const bfloat* out [[buffer(0)]], device const uint* where [[buffer(1)]],
    device const float* weights [[buffer(2)]], device OutT* y [[buffer(3)]],
    device const uint* ids [[buffer(4)]], constant XpArgs& a [[buffer(5)]],
    constant uint3& g [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.N) return;
  float part[8];
  for (uint i = 0; i < 8; ++i) part[i] = 0.0f;
  for (uint k = 0; k < a.topk; ++k) {
    const uint p = where[gid.y * a.topk + k];
    if (p == NOWHERE) continue;
    const uint grp = ids[gid.y * a.topk + k] / g.x - g.y;
    part[grp] = fma(weights[gid.y * a.topk + k], float(out[p * a.N + gid.x]), part[grp]);
  }
  for (uint w = g.z; w > 1; w /= 2)
    for (uint i = 0; i < w / 2; ++i) part[i] = part[2 * i] + part[2 * i + 1];
  y[gid.y * a.N + gid.x] = OutT(part[0]);
}

template [[host_name("tf_dsv41_xp_combine")]] [[kernel]] decltype(tf_dsv41_xp_combine<bfloat>) tf_dsv41_xp_combine<bfloat>;
template [[host_name("tf_dsv41_xp_combine_f32")]] [[kernel]] decltype(tf_dsv41_xp_combine<float>) tf_dsv41_xp_combine<float>;
