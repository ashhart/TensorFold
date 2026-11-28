#include <metal_stdlib>
#include "../nax.h"
using namespace metal;

// TFQ3 (2D-AffineCoset/half4) shared structs, FWHT-128 helpers, routing, and branch-hoisted shared expert.
#define FN_PW_UNROLL _Pragma("clang loop unroll(full)")

inline short2 pw_frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

struct FnPwExpertMeta {
  uint  b_bits;
  uint  heat_rank;
  ulong gate_word_off;
  ulong up_word_off;
  ulong down_word_off;
  uint  gate_sc_off;
  uint  up_sc_off;
  uint  down_sc_off;
  float gate_snr_db;
  float up_snr_db;
  float down_snr_db;
  uint  reserved[2];
};

struct FnPwDims {
  uint  rows;
  uint  slots;
  uint  topk;
  uint  experts;
  uint  sh_gu_bits;
  uint  sh_gu_group;
  uint  sh_dn_bits;
  uint  sh_dn_group;
  uint  flags;
  uint  pad[3];
};

struct FnPwV3Tables {
  uint  magic;
  uint  version;
  float a3, b3, a4, b4, a5, b5;
  uint  pad0[8];
  uint  lut3[256];
  uint  lut4[1024];
  uint  lut5[1024];
  half4 c64_3[64];
  half4 c64_4[64];
  half4 c64_5[64];
};

inline float pw_sgn(uint seed, uint i) {
  uint h = (seed ^ i) * 2654435761u;
  return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline float fn_bsilu(float x) {
  return float(bfloat(x / (1.0f + metal::exp(-x))));
}

inline bfloat fn_pw_swiglu(float g, float u) {
  return bfloat(fn_bsilu(float(bfloat(g))) * float(bfloat(u)));
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
  FN_PW_UNROLL
  for (ushort d = 1; d <= 16; d <<= 1) {
    float4 o = simd_shuffle_xor(v, d);
    v = ((lane & d) == 0u) ? (v + o) : (o - v);
  }
  float a0 = v.x + v.y, b0 = v.x - v.y;
  float a1 = v.z + v.w, b1 = v.z - v.w;
  v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

inline void fn_route_row(
    const device float* logits, device uint* pick, device float* wts, ushort lane
) {
  float v[16];
  for (int j = 0; j < 16; ++j) v[j] = logits[j * 32 + int(lane)];
  float picked[10];
  int ids[10];
  for (int round = 0; round < 10; ++round) {
    float best = -INFINITY;
    int bid = 512;
    for (int j = 0; j < 16; ++j) {
      int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == (bid & 31)) v[bid >> 5] = -INFINITY;
  }
  if (lane == 0u) {
    float total = 0.0f;
    float ex[10];
    for (int kk = 0; kk < 10; ++kk) {
      pick[kk] = uint(ids[kk]);
      ex[kk] = metal::exp(picked[kk] - picked[0]);
      total += ex[kk];
    }
    for (int kk = 0; kk < 10; ++kk) wts[kk] = float(bfloat(ex[kk] / total));
  }
}

[[kernel]] void fn_pw_route_dedup(
    const device float* logits   [[buffer(0)]],
    const constant int* rows_buf [[buffer(1)]],
    device uint* pick            [[buffer(2)]],
    device float* wts            [[buffer(3)]],
    device int* uids             [[buffer(4)]],
    device int* umem             [[buffer(5)]],
    device int* ucount           [[buffer(6)]],
    constant FnPwDims& d         [[buffer(7)]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  for (int r = 0; r < R; ++r) {
    fn_route_row(logits + ulong(r) * 513ul, pick + ulong(r) * 10ul, wts + ulong(r) * 10ul, lane);
  }
  simdgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    int total_slots = R * 10;
    int nu = 0;
    for (int s = 0; s < total_slots; ++s) {
      int e = int(pick[s]);
      int found = -1;
      for (int u = 0; u < nu; ++u) {
        if (uids[u] == e) { found = u; break; }
      }
      if (found < 0) {
        found = nu++;
        uids[found] = e;
        for (int m = 0; m < 16; ++m) umem[found * 16 + m] = -1;
        umem[found * 16 + 0] = s;
      } else {
        for (int m = 0; m < 16; ++m) {
          if (umem[found * 16 + m] < 0) { umem[found * 16 + m] = s; break; }
        }
      }
    }
    ucount[0] = nu;
  }
}

// fn_pw_route_dedup's lists bit for bit, a thread a slot: first-seen experts in slot order, members in slot order
[[max_total_threads_per_threadgroup(160)]]
[[kernel]] void fn_pw_dedup_par(
    const device uint* pick      [[buffer(0)]],
    const constant int* rows_buf [[buffer(1)]],
    device int* uids             [[buffer(2)]],
    device int* umem             [[buffer(3)]],
    device int* ucount           [[buffer(4)]],
    constant FnPwDims& d         [[buffer(5)]],
    ushort s                     [[thread_index_in_threadgroup]]
) {
  threadgroup uint pk[160];
  threadgroup uchar opens[160];
  const int R = metal::min(int(d.rows), rows_buf[0]);
  const int S = metal::min(metal::max(R, 0) * 10, 160);
  const int i = int(s);
  if (i < S) pk[i] = pick[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool open = false;
  if (i < S) {
    open = true;
    for (int t = 0; t < i; ++t) if (pk[t] == pk[i]) { open = false; break; }
    opens[i] = open ? uchar(1) : uchar(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (open) {
    int u = 0;
    for (int t = 0; t < i; ++t) u += int(opens[t]);
    uids[u] = int(pk[i]);
    int m = 0;
    for (int t = i; t < S && m < 16; ++t) if (pk[t] == pk[i]) umem[u * 16 + m++] = t;
    for (; m < 16; ++m) umem[u * 16 + m] = -1;
  }
  if (i == 0) {
    int nu = 0;
    for (int t = 0; t < S; ++t) nu += int(opens[t]);
    ucount[0] = nu;
  }
}

[[kernel]] void fn_pw_fwht_x_fwd(
    const device bfloat* X       [[buffer(0)]],
    device half* X_rot           [[buffer(1)]],
    const constant int* rows_buf [[buffer(2)]],
    constant FnPwDims& d         [[buffer(3)]],
    const device float* logits   [[buffer(4)]],
    device uint* pick            [[buffer(5)]],
    device float* wts            [[buffer(6)]],
    uint2 gid                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  if (gid.x == 20u) {
    // flags 2: row 0 alone (B=1); flags 4: every live row, in parallel, ahead of fn_pw_dedup_par (B=2..16)
    const bool route = int(gid.y) < R && (((d.flags & 2u) != 0u && gid.y == 0u) || (d.flags & 4u) != 0u);
    if (route) fn_route_row(logits + ulong(gid.y) * 513ul, pick + ulong(gid.y) * 10ul, wts + ulong(gid.y) * 10ul, lane);
    return;
  }
  uint row = gid.y;
  if (int(row) >= R) return;
  uint c0 = gid.x * 128u;
  const device bfloat* src = X + ulong(row) * 2560ul + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
  pw_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(row) * 2560ul + c0;
  dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

[[kernel]] void fn_pw_fwht_glu_fwht(
    const device float* GU_rot   [[buffer(0)]],
    device half* ACT_rot         [[buffer(1)]],
    const constant int* rows_buf [[buffer(2)]],
    constant FnPwDims& d         [[buffer(3)]],
    uint2 gid                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  uint slot = gid.y;
  if (int(slot) >= R * 10) return;
  uint r0 = gid.x * 128u;
  const device float* g_src = GU_rot + ulong(slot * 2u + 0u) * 640ul + r0;
  const device float* u_src = GU_rot + ulong(slot * 2u + 1u) * 640ul + r0;
  float4 gv = float4(g_src[lane], g_src[lane + 32u], g_src[lane + 64u], g_src[lane + 96u]);
  float4 uv = float4(u_src[lane], u_src[lane + 32u], u_src[lane + 64u], u_src[lane + 96u]);
  pw_fwht128_simd4(gv, lane);
  pw_fwht128_simd4(uv, lane);
  float4 z;
  z.x = float(fn_pw_swiglu(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u))) * pw_sgn(0x12345678u, r0 + lane +  0u);
  z.y = float(fn_pw_swiglu(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u))) * pw_sgn(0x12345678u, r0 + lane + 32u);
  z.z = float(fn_pw_swiglu(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u))) * pw_sgn(0x12345678u, r0 + lane + 64u);
  z.w = float(fn_pw_swiglu(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u))) * pw_sgn(0x12345678u, r0 + lane + 96u);
  pw_fwht128_simd4(z, lane);
  device half* dst = ACT_rot + ulong(slot) * 640ul + r0;
  dst[lane +  0u] = half(z.x); dst[lane + 32u] = half(z.y);
  dst[lane + 64u] = half(z.z); dst[lane + 96u] = half(z.w);
}

[[kernel]] void fn_pw_fwht_y_inv(
    const device float* YR       [[buffer(0)]],
    device bfloat* YDOWN         [[buffer(1)]],
    const constant int* rows_buf [[buffer(2)]],
    constant FnPwDims& d         [[buffer(3)]],
    uint2 gid                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
  int R = metal::min(int(d.rows), rows_buf[0]);
  uint slot = gid.y;
  if (int(slot) >= R * 10) return;
  uint r0 = gid.x * 128u;
  const device float* src = YR + ulong(slot) * 2560ul + r0;
  float4 v = float4(src[lane], src[lane + 32u], src[lane + 64u], src[lane + 96u]);
  pw_fwht128_simd4(v, lane);
  uint row = slot / 10u, k = slot - row * 10u;
  device bfloat* dst = YDOWN + ulong(row * 11u + k) * 2560ul + r0;
  dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u));
  dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u));
  dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u));
  dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u));
}

template <uint BITS>
inline void fn_affine_load8(const device uchar* wp, uint v0, uint bits, uint mask, thread float4& q0, thread float4& q1) {
  if constexpr (BITS == 8u) {
    const device uchar4* p4 = (const device uchar4*)(wp + v0);
    q0 = float4(p4[0]);
    q1 = float4(p4[1]);
  } else if constexpr (BITS == 4u) {
    uint w = *(const device uint*)(wp + (v0 >> 1));
    q0 = float4(float(w & 15u), float((w >> 4) & 15u), float((w >> 8) & 15u), float((w >> 12) & 15u));
    q1 = float4(float((w >> 16) & 15u), float((w >> 20) & 15u), float((w >> 24) & 15u), float(w >> 28));
  } else {
    const device uchar* p = wp + ulong(v0 >> 3) * bits;
    ulong pack = 0ul;
    for (uint j = 0u; j < bits; ++j) pack |= ulong(p[j]) << (j * 8u);
    q0 = float4(float(pack & mask), float((pack >> bits) & mask),
                float((pack >> (2u * bits)) & mask), float((pack >> (3u * bits)) & mask));
    q1 = float4(float((pack >> (4u * bits)) & mask), float((pack >> (5u * bits)) & mask),
                float((pack >> (6u * bits)) & mask), float((pack >> (7u * bits)) & mask));
  }
}

template <uint BITS>
inline void fn_sh_gu_b1(
    const device bfloat* X, const device uint* SGW, const device bfloat* SGS, const device bfloat* SGB,
    const device uint* SUW, const device bfloat* SUS, const device bfloat* SUB,
    device bfloat* ACT, uint row, uint bits, uint gs, ushort lane
) {
  ulong rb = (BITS != 0u) ? (2560ul * BITS / 8ul) : (2560ul * bits / 8ul);
  ulong ng = 2560ul >> gs;
  const device uchar* gwp = (const device uchar*)SGW + ulong(row) * rb;
  const device uchar* uwp = (const device uchar*)SUW + ulong(row) * rb;
  const device bfloat* gsp = SGS + ulong(row) * ng;
  const device bfloat* gbp = SGB + ulong(row) * ng;
  const device bfloat* usp = SUS + ulong(row) * ng;
  const device bfloat* ubp = SUB + ulong(row) * ng;
  uint mask = (1u << bits) - 1u;
  float ga = 0.0f, ua = 0.0f;
  for (uint v0 = uint(lane) * 8u; v0 < 2560u; v0 += 256u) {
    const device bfloat4* xb = (const device bfloat4*)(X + v0);
    float4 x0 = float4(xb[0]), x1 = float4(xb[1]);
    float sx = (x0.x + x0.y) + (x0.z + x0.w) + (x1.x + x1.y) + (x1.z + x1.w);
    uint g = v0 >> gs;
    float4 q0, q1;
    fn_affine_load8<BITS>(gwp, v0, bits, mask, q0, q1);
    ga += fma(float(gsp[g]), dot(q0, x0) + dot(q1, x1), float(gbp[g]) * sx);
    fn_affine_load8<BITS>(uwp, v0, bits, mask, q0, q1);
    ua += fma(float(usp[g]), dot(q0, x0) + dot(q1, x1), float(ubp[g]) * sx);
  }
  float gv = simd_sum(ga), uv = simd_sum(ua);
  if (lane == 0u) ACT[10ul * 640ul + row] = fn_pw_swiglu(gv, uv);
}

template <uint BITS>
inline void fn_sh_gu_multi(
    const device bfloat* X, const device uint* SGW, const device bfloat* SGS, const device bfloat* SGB,
    const device uint* SUW, const device bfloat* SUS, const device bfloat* SUB,
    device bfloat* ACT, uint row, int R, uint bits, uint gs, ushort lane
) {
  ulong rb = (BITS != 0u) ? (2560ul * BITS / 8ul) : (2560ul * bits / 8ul);
  ulong ng = 2560ul >> gs;
  const device uchar* gwp = (const device uchar*)SGW + ulong(row) * rb;
  const device uchar* uwp = (const device uchar*)SUW + ulong(row) * rb;
  const device bfloat* gsp = SGS + ulong(row) * ng;
  const device bfloat* gbp = SGB + ulong(row) * ng;
  const device bfloat* usp = SUS + ulong(row) * ng;
  const device bfloat* ubp = SUB + ulong(row) * ng;
  uint mask = (1u << bits) - 1u;
  for (uint base = 0u; base < uint(R); base += 4u) {
    uint rem = metal::min(4u, uint(R) - base);
    const device bfloat* x0p = X + ulong(base) * 2560ul;
    const device bfloat* x1p = (rem > 1u) ? (x0p + 2560ul) : x0p;
    const device bfloat* x2p = (rem > 2u) ? (x0p + 5120ul) : x0p;
    const device bfloat* x3p = (rem > 3u) ? (x0p + 7680ul) : x0p;
    float ga0 = 0.0f, ga1 = 0.0f, ga2 = 0.0f, ga3 = 0.0f;
    float ua0 = 0.0f, ua1 = 0.0f, ua2 = 0.0f, ua3 = 0.0f;
    for (uint v0 = uint(lane) * 8u; v0 < 2560u; v0 += 256u) {
      uint g = v0 >> gs;
      float gsc = float(gsp[g]), gbi = float(gbp[g]), usc = float(usp[g]), ubi = float(ubp[g]);
      float4 gq0, gq1, uq0, uq1;
      fn_affine_load8<BITS>(gwp, v0, bits, mask, gq0, gq1);
      fn_affine_load8<BITS>(uwp, v0, bits, mask, uq0, uq1);
      {
        const device bfloat4* xb = (const device bfloat4*)(x0p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        float sx = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
        ga0 += fma(gsc, dot(gq0, a) + dot(gq1, b), gbi * sx);
        ua0 += fma(usc, dot(uq0, a) + dot(uq1, b), ubi * sx);
      }
      if (rem > 1u) {
        const device bfloat4* xb = (const device bfloat4*)(x1p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        float sx = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
        ga1 += fma(gsc, dot(gq0, a) + dot(gq1, b), gbi * sx);
        ua1 += fma(usc, dot(uq0, a) + dot(uq1, b), ubi * sx);
      }
      if (rem > 2u) {
        const device bfloat4* xb = (const device bfloat4*)(x2p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        float sx = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
        ga2 += fma(gsc, dot(gq0, a) + dot(gq1, b), gbi * sx);
        ua2 += fma(usc, dot(uq0, a) + dot(uq1, b), ubi * sx);
      }
      if (rem > 3u) {
        const device bfloat4* xb = (const device bfloat4*)(x3p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        float sx = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
        ga3 += fma(gsc, dot(gq0, a) + dot(gq1, b), gbi * sx);
        ua3 += fma(usc, dot(uq0, a) + dot(uq1, b), ubi * sx);
      }
    }
    ga0 = simd_sum(ga0); ua0 = simd_sum(ua0);
    if (lane == 0u) ACT[ulong(base * 11u + 10u) * 640ul + row] = fn_pw_swiglu(ga0, ua0);
    if (rem > 1u) { ga1 = simd_sum(ga1); ua1 = simd_sum(ua1); if (lane == 0u) ACT[ulong((base + 1u) * 11u + 10u) * 640ul + row] = fn_pw_swiglu(ga1, ua1); }
    if (rem > 2u) { ga2 = simd_sum(ga2); ua2 = simd_sum(ua2); if (lane == 0u) ACT[ulong((base + 2u) * 11u + 10u) * 640ul + row] = fn_pw_swiglu(ga2, ua2); }
    if (rem > 3u) { ga3 = simd_sum(ga3); ua3 = simd_sum(ua3); if (lane == 0u) ACT[ulong((base + 3u) * 11u + 10u) * 640ul + row] = fn_pw_swiglu(ga3, ua3); }
  }
}

template <uint BITS>
inline void fn_sh_dn_b1(
    const device bfloat* ACT, const device uint* SDW, const device bfloat* SDS, const device bfloat* SDB,
    device bfloat* YDOWN, uint row, uint bits, uint gs, ushort lane
) {
  ulong rb = (BITS != 0u) ? (640ul * BITS / 8ul) : (640ul * bits / 8ul);
  ulong ng = 640ul >> gs;
  const device uchar* wp = (const device uchar*)SDW + ulong(row) * rb;
  const device bfloat* sp = SDS + ulong(row) * ng;
  const device bfloat* bp = SDB + ulong(row) * ng;
  const device bfloat* sx = ACT + 10ul * 640ul;
  uint mask = (1u << bits) - 1u;
  float acc = 0.0f;
  for (uint v0 = uint(lane) * 8u; v0 < 640u; v0 += 256u) {
    const device bfloat4* xb = (const device bfloat4*)(sx + v0);
    float4 x0 = float4(xb[0]), x1 = float4(xb[1]);
    float sxv = (x0.x + x0.y) + (x0.z + x0.w) + (x1.x + x1.y) + (x1.z + x1.w);
    uint g = v0 >> gs;
    float4 q0, q1;
    fn_affine_load8<BITS>(wp, v0, bits, mask, q0, q1);
    acc += fma(float(sp[g]), dot(q0, x0) + dot(q1, x1), float(bp[g]) * sxv);
  }
  float yv = simd_sum(acc);
  if (lane == 0u) YDOWN[10ul * 2560ul + row] = bfloat(yv);
}

template <uint BITS>
inline void fn_sh_dn_multi(
    const device bfloat* ACT, const device uint* SDW, const device bfloat* SDS, const device bfloat* SDB,
    device bfloat* YDOWN, uint row, int R, uint bits, uint gs, ushort lane
) {
  ulong rb = (BITS != 0u) ? (640ul * BITS / 8ul) : (640ul * bits / 8ul);
  ulong ng = 640ul >> gs;
  const device uchar* wp = (const device uchar*)SDW + ulong(row) * rb;
  const device bfloat* sp = SDS + ulong(row) * ng;
  const device bfloat* bp = SDB + ulong(row) * ng;
  uint mask = (1u << bits) - 1u;
  for (uint base = 0u; base < uint(R); base += 4u) {
    uint rem = metal::min(4u, uint(R) - base);
    const device bfloat* x0p = ACT + ulong(base * 11u + 10u) * 640ul;
    const device bfloat* x1p = (rem > 1u) ? (x0p + 11ul * 640ul) : x0p;
    const device bfloat* x2p = (rem > 2u) ? (x0p + 22ul * 640ul) : x0p;
    const device bfloat* x3p = (rem > 3u) ? (x0p + 33ul * 640ul) : x0p;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint v0 = uint(lane) * 8u; v0 < 640u; v0 += 256u) {
      uint g = v0 >> gs;
      float sc = float(sp[g]), bi = float(bp[g]);
      float4 q0, q1;
      fn_affine_load8<BITS>(wp, v0, bits, mask, q0, q1);
      {
        const device bfloat4* xb = (const device bfloat4*)(x0p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        a0 += fma(sc, dot(q0, a) + dot(q1, b), bi * ((a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w)));
      }
      if (rem > 1u) {
        const device bfloat4* xb = (const device bfloat4*)(x1p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        a1 += fma(sc, dot(q0, a) + dot(q1, b), bi * ((a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w)));
      }
      if (rem > 2u) {
        const device bfloat4* xb = (const device bfloat4*)(x2p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        a2 += fma(sc, dot(q0, a) + dot(q1, b), bi * ((a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w)));
      }
      if (rem > 3u) {
        const device bfloat4* xb = (const device bfloat4*)(x3p + v0);
        float4 a = float4(xb[0]), b = float4(xb[1]);
        a3 += fma(sc, dot(q0, a) + dot(q1, b), bi * ((a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w)));
      }
    }
    a0 = simd_sum(a0); if (lane == 0u) YDOWN[ulong(base * 11u + 10u) * 2560ul + row] = bfloat(a0);
    if (rem > 1u) { a1 = simd_sum(a1); if (lane == 0u) YDOWN[ulong((base + 1u) * 11u + 10u) * 2560ul + row] = bfloat(a1); }
    if (rem > 2u) { a2 = simd_sum(a2); if (lane == 0u) YDOWN[ulong((base + 2u) * 11u + 10u) * 2560ul + row] = bfloat(a2); }
    if (rem > 3u) { a3 = simd_sum(a3); if (lane == 0u) YDOWN[ulong((base + 3u) * 11u + 10u) * 2560ul + row] = bfloat(a3); }
  }
}

inline float pw_fin(float v) { return isfinite(v) ? v : 0.0f; }
inline float2 pw_fin(float2 v) { return float2(pw_fin(v.x), pw_fin(v.y)); }
inline float4 pw_fin(float4 v) { return float4(pw_fin(v.x), pw_fin(v.y), pw_fin(v.z), pw_fin(v.w)); }

inline void pw_v3_q8_4b(thread uint& st, uint wc, constant FnPwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint h0 = st * 0x9E3779B1u; uint st1 = (st << 8) | (wc >> 24);
  uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 8) | ((wc >> 16) & 255u);
  uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 8) | ((wc >>  8) & 255u);
  uint h3 = st3 * 0x9E3779B1u; st = (st3 << 8) | (wc & 255u);
  half4 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut4[(((wc >> 22) & 1020u) ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[(((wc >> 14) & 1020u) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut4[(((wc >>  6) & 1020u) ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[(((wc <<  2) & 1020u) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_3b(thread uint& st, uint m24, constant FnPwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = (m24 >> 18) & 63u, s1 = (m24 >> 12) & 63u, s2 = (m24 >> 6) & 63u, s3 = m24 & 63u;
  uint h0 = st * 0x9E3779B1u; uint st1 = (st << 6) | s0;
  uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 6) | s1;
  uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 6) | s2;
  uint h3 = st3 * 0x9E3779B1u; st = (st3 << 6) | s3;
  half4 c0 = T->c64_3[(h0 >> 16) & 63u], c2 = T->c64_3[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut3[((s0 << 2) ^ (h0 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[((s1 << 2) ^ (h1 >> 24)) & 255u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut3[((s2 << 2) ^ (h2 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[((s3 << 2) ^ (h3 >> 24)) & 255u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_5b(thread uint& st, uint hi8, uint lo32, constant FnPwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = (hi8 << 2) | (lo32 >> 30);
  uint s1 = (lo32 >> 20) & 1023u, s2 = (lo32 >> 10) & 1023u, s3 = lo32 & 1023u;
  uint h0 = st * 0x9E3779B1u; uint st1 = (st << 10) | s0;
  uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 10) | s1;
  uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 10) | s2;
  uint h3 = st3 * 0x9E3779B1u; st = (st3 << 10) | s3;
  half4 c0 = T->c64_5[(h0 >> 16) & 63u], c2 = T->c64_5[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut5[(s0 ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(s1 ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut5[(s2 ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(s3 ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

#define PW_MDOT1(q0, q1, X, off, a0, a1) { \
  float4 xv = float4(*(const device half4*)((X) + (off))); \
  a0 += dot(float4(q0), xv); a1 += dot(float4(q1), xv); }
#define PW_MDOT2(q0, q1, X0, X1, off, a0, a1) { \
  float4 f0 = float4(q0), f1 = float4(q1); \
  float4 x0v = float4(*(const device half4*)((X0) + (off))); \
  float4 x1v = float4(*(const device half4*)((X1) + (off))); \
  a0.x += dot(f0, x0v); a1.x += dot(f1, x0v); a0.y += dot(f0, x1v); a1.y += dot(f1, x1v); }
#define PW_MDOT4(q0, q1, X0, X1, X2, X3, off, a0, a1) { \
  float4 f0 = float4(q0), f1 = float4(q1); \
  float4 x0v = float4(*(const device half4*)((X0) + (off))); \
  float4 x1v = float4(*(const device half4*)((X1) + (off))); \
  float4 x2v = float4(*(const device half4*)((X2) + (off))); \
  float4 x3v = float4(*(const device half4*)((X3) + (off))); \
  a0.x += dot(f0, x0v); a1.x += dot(f1, x0v); a0.y += dot(f0, x1v); a1.y += dot(f1, x1v); \
  a0.z += dot(f0, x2v); a1.z += dot(f1, x2v); a0.w += dot(f0, x3v); a1.w += dot(f1, x3v); }
