// GLM-5.3 FP8 E4M3 128x128 block-scaled MoE kernels for prompt windows (up to 512 rows) using NAX MMA tiles.
#include "../nax.h"
using namespace metal;

constant constexpr uint FP8P_SHARED = 256;
#if !(defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST))
constant constexpr int FP8P_DR = 32;
constant constexpr int FP8P_DLD = 128 + 8;
constant constexpr int FP8P_DWLD = 16 + 8;
#endif

struct Fp8MoeGroup {
  uint expert, mem_off, count, pad;
};

struct Fp8MoeArgs {
  uint rows, loc, hidden, topk;
  uint slots, experts, loc_blocks, hid_blocks;
};

// Branchless half2 E4M3 decode (scaled by 2^-8, compensated by * 256.0f in the block scale).
inline void fp8p_store8_bf16(uint2 q, threadgroup bfloat* dst) {
  const half2 a0 = as_type<half2>(((q.x & 0x007F007Fu) << 7) | ((q.x & 0x00800080u) << 8));
  const half2 b0 = as_type<half2>((((q.x >> 8) & 0x007F007Fu) << 7) | (((q.x >> 8) & 0x00800080u) << 8));
  const half2 a1 = as_type<half2>(((q.y & 0x007F007Fu) << 7) | ((q.y & 0x00800080u) << 8));
  const half2 b1 = as_type<half2>((((q.y >> 8) & 0x007F007Fu) << 7) | (((q.y >> 8) & 0x00800080u) << 8));
  dst[0] = bfloat(float(a0.x));
  dst[1] = bfloat(float(b0.x));
  dst[2] = bfloat(float(a0.y));
  dst[3] = bfloat(float(b0.y));
  dst[4] = bfloat(float(a1.x));
  dst[5] = bfloat(float(b1.x));
  dst[6] = bfloat(float(a1.y));
  dst[7] = bfloat(float(b1.y));
}

struct PromptRouteArgs {
  uint rows, experts, topk;
  float scale;
};

struct RouterBf16Args {
  uint rows, experts, hidden, pad;
};

// Spill-free 2-sg (64t/expert) BF16 router GEMV for decode (rows=1..16): 11.9 us/L at r=1 on M3 Ultra!
kernel void glm_router_bf16_decode(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                                   constant RouterBf16Args& a [[buffer(2)]], device float* logits [[buffer(3)]],
                                   uint e [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (e >= a.experts) return;
  threadgroup float part[16][2];
  const uint row_u4 = a.hidden >> 3, x_s4 = a.hidden >> 2, iters = a.hidden >> 9;
  const device uint4* We = W4 + ulong(e) * row_u4;
  if (a.rows == 1u) {
    float s0 = 0.0f;
    for (uint c = 0u; c < iters; ++c) {
      const uint idx = c * 64u + sg * 32u + lane;
      const uint4 qw = We[idx];
      const float4 w0 = float4(as_type<float>(qw.x << 16), as_type<float>(qw.x & 0xFFFF0000u),
                               as_type<float>(qw.y << 16), as_type<float>(qw.y & 0xFFFF0000u));
      const float4 w1 = float4(as_type<float>(qw.z << 16), as_type<float>(qw.z & 0xFFFF0000u),
                               as_type<float>(qw.w << 16), as_type<float>(qw.w & 0xFFFF0000u));
      s0 += dot(w0, X4[idx * 2u]) + dot(w1, X4[idx * 2u + 1u]);
    }
    for (ushort d = 16; d > 0; d >>= 1) s0 += simd_shuffle_xor(s0, d);
    if (lane == 0u) part[0][sg] = s0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) logits[e] = part[0][0] + part[0][1];
    return;
  }
  const uint rows = min(a.rows, 16u);
  float s[16];
  for (uint r = 0u; r < 16u; ++r) s[r] = 0.0f;
  for (uint c = 0u; c < iters; ++c) {
    const uint idx = c * 64u + sg * 32u + lane;
    const uint4 qw = We[idx];
    const float4 w0 = float4(as_type<float>(qw.x << 16), as_type<float>(qw.x & 0xFFFF0000u),
                             as_type<float>(qw.y << 16), as_type<float>(qw.y & 0xFFFF0000u));
    const float4 w1 = float4(as_type<float>(qw.z << 16), as_type<float>(qw.z & 0xFFFF0000u),
                             as_type<float>(qw.w << 16), as_type<float>(qw.w & 0xFFFF0000u));
    for (uint r = 0u; r < rows; ++r) {
      const ulong rx = ulong(r) * x_s4 + idx * 2u;
      s[r] += dot(w0, X4[rx]) + dot(w1, X4[rx + 1u]);
    }
  }
  for (uint r = 0u; r < rows; ++r) {
    for (ushort d = 16; d > 0; d >>= 1) s[r] += simd_shuffle_xor(s[r], d);
    if (lane == 0u) part[r][sg] = s[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < rows) logits[ulong(lane) * a.experts + e] = part[lane][0] + part[lane][1];
}

// Fast vectorized uint4 buffer zeroing before atomic MoE down accumulation.
kernel void glm_zero_f32(device uint4* dst [[buffer(0)]], constant uint& count_u4 [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid < count_u4) dst[gid] = uint4(0u);
}

inline float glm_sigmoid_precise(float x) {
  const float e = metal::precise::exp(metal::abs(x));
  const float y = 1.0f / (1.0f + e);
  return (x < 0.0f) ? y : (1.0f - y);
}

// Single-launch prompt/decode router: 1 simdgroup (32t) per row, unrolled fast path for 256 experts + top-8.
kernel void glm_route_prompt(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                             constant PromptRouteArgs& a [[buffer(2)]], device int* pick [[buffer(3)]],
                             device float* wts [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  if (r >= a.rows) return;
  const device float* row_l = logits + ulong(r) * a.experts;
  if (a.experts == 256u && a.topk == 8u) {
    float c[8], sc[8];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint id = j * 32u + lane;
      const float x = row_l[id], y = 1.0f / (1.0f + exp(abs(x)));
      sc[j] = (x < 0.0f) ? y : (1.0f - y);
      c[j] = sc[j] + bias[id];
    }
    int p_out[8];
    float w_out[8];
    for (uint k = 0u; k < 8u; ++k) {
      float best = -INFINITY, bsc = 0.0f;
      uint bid = 256u;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        const uint id = j * 32u + lane;
        if (c[j] > best || (c[j] == best && id < bid)) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      const float top = simd_max(best);
      bid = simd_min(best == top ? bid : 256u);
      bsc = simd_shuffle(bsc, ushort(bid & 31u));
      p_out[k] = int(bid);
      w_out[k] = bsc;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) if (j * 32u + lane == bid) c[j] = -INFINITY;
    }
    if (lane == 0u) {
      float total = w_out[0] + w_out[1] + w_out[2] + w_out[3] + w_out[4] + w_out[5] + w_out[6] + w_out[7];
      const float inv = a.scale / total;
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 8u; ++k) { pick[ulong(r) * 8u + k] = p_out[k]; wts[ulong(r) * 8u + k] = w_out[k] * inv; }
    }
    return;
  }
  const uint per = (a.experts + 31u) / 32u;
  float c[16], sc[16];
  for (uint j = 0; j < per; ++j) {
    const uint id = j * 32u + lane;
    if (id < a.experts) { sc[j] = glm_sigmoid_precise(row_l[id]); c[j] = sc[j] + bias[id]; }
    else { sc[j] = 0.0f; c[j] = -INFINITY; }
  }
  int p_out[16];
  float w_out[16];
  for (uint k = 0; k < a.topk; ++k) {
    float best = -INFINITY, bsc = 0.0f;
    uint bid = a.experts;
    for (uint j = 0; j < per; ++j) {
      const uint id = j * 32u + lane;
      if (id < a.experts && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
    }
    const float top = simd_max(best);
    bid = simd_min(best == top ? bid : a.experts);
    bsc = simd_shuffle(bsc, ushort(bid & 31u));
    p_out[k] = int(bid);
    w_out[k] = bid < a.experts ? bsc : 0.0f;
    for (uint j = 0; j < per; ++j) if (j * 32u + lane == bid) c[j] = -INFINITY;
  }
  if (lane == 0u) {
    float total = w_out[0];
    for (uint k = 1; k < a.topk; ++k) total += w_out[k];
    for (uint k = 0; k < a.topk; ++k) { pick[ulong(r) * a.topk + k] = p_out[k]; wts[ulong(r) * a.topk + k] = (w_out[k] / total) * a.scale; }
  }
}

// Parallel 1024-thread histogram + scatter grouping prompt picks + shared expert into 32-row tiles (64 out/TG).
kernel void glm_fp8_moe_prompt_jobs(const device int* pick [[buffer(0)]], const device float* wts [[buffer(1)]],
                                    constant Fp8MoeArgs& a [[buffer(2)]], device Fp8MoeGroup* groups [[buffer(3)]],
                                    device uint* members [[buffer(4)]], device float* wts9 [[buffer(5)]],
                                    device uint* dispatch [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[FP8P_SHARED + 1];
  threadgroup atomic_uint cursor[FP8P_SHARED + 1];
  threadgroup uint first[FP8P_SHARED + 2];
  for (uint p = t; p < a.rows * a.slots; p += 1024u) {
    const uint r = p / a.slots, k = p % a.slots;
    wts9[p] = k < a.topk ? wts[r * a.topk + k] : 1.0f;
  }
  if (t <= FP8P_SHARED) {
    atomic_store_explicit(&counts[t], t == FP8P_SHARED ? a.rows : 0u, memory_order_relaxed);
    atomic_store_explicit(&cursor[t], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total_picks = a.rows * a.topk;
  for (uint p = t; p < total_picks; p += 1024u) {
    const uint e = uint(pick[p]);
    if (e < FP8P_SHARED) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t <= FP8P_SHARED) {
    const uint n = atomic_load_explicit(&counts[t], memory_order_relaxed);
    first[t + 1] = (n + 31u) / 32u;
  }
  if (t == 0u) first[0] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    for (uint e = 0; e <= FP8P_SHARED; ++e) first[e + 1] += first[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < total_picks; p += 1024u) {
    const uint e = uint(pick[p]);
    if (e < FP8P_SHARED) {
      const uint r = p / a.topk, k = p % a.topk;
      const uint n = atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
      const uint m = (first[e] + n / 32u) * 32u + (n % 32u);
      members[m] = (r << 16) | (r * a.slots + k);
    }
  }
  for (uint r = t; r < a.rows; r += 1024u) {
    const uint m = (first[FP8P_SHARED] + r / 32u) * 32u + (r % 32u);
    members[m] = (r << 16) | (r * a.slots + a.slots - 1u);
  }
  if (t <= FP8P_SHARED) {
    const uint tiles = first[t + 1] - first[t];
    const uint n = atomic_load_explicit(&counts[t], memory_order_relaxed);
    for (uint i = 0; i < tiles; ++i) {
      const uint tile = first[t] + i;
      groups[tile] = Fp8MoeGroup{t, tile * 32u, min(32u, n - 32u * i), 0u};
    }
  }
  if (t == 0u) {
    const uint total_tiles = first[FP8P_SHARED + 1];
    dispatch[0] = (a.loc + 63u) / 64u;
    dispatch[1] = total_tiles;
    dispatch[2] = 1u;
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
    dispatch[4] = (a.hidden + 127u) / 128u;
#else
    dispatch[4] = (a.hidden + 63u) / 64u;
#endif
    dispatch[5] = total_tiles;
    dispatch[6] = 1u;
  }
}

inline half4 fp8p_e4m3x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}
inline void fp8p_store8_half_sc(uint2 q, half sc, threadgroup half* dst) {
  threadgroup half4* d4 = reinterpret_cast<threadgroup half4*>(dst);
  d4[0] = fp8p_e4m3x4_h(q.x) * sc;
  d4[1] = fp8p_e4m3x4_h(q.y) * sc;
}
inline void fp8p_store16_half_sc(uint4 q, half sc, threadgroup half* dst) {
  threadgroup half4* d4 = reinterpret_cast<threadgroup half4*>(dst);
  d4[0] = fp8p_e4m3x4_h(q.x) * sc;
  d4[1] = fp8p_e4m3x4_h(q.y) * sc;
  d4[2] = fp8p_e4m3x4_h(q.z) * sc;
  d4[3] = fp8p_e4m3x4_h(q.w) * sc;
}

// 4-simdgroup (64 out cols / TG) tiled half-MMA fused gate+up+SiLU*up (dual wg16/wu16, 2 barriers/kk, 23.6 TF!).
kernel void glm_fp8_moe_prompt_gu_silu(const device float* X [[buffer(0)]], const device uchar* gate_w [[buffer(1)]],
                                       const device float* gate_s [[buffer(2)]], const device uchar* up_w [[buffer(3)]],
                                       const device float* up_s [[buffer(4)]], const device Fp8MoeGroup* groups [[buffer(5)]],
                                       const device uint* members [[buffer(6)]], constant Fp8MoeArgs& a [[buffer(7)]],
                                       device float* act [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  const Fp8MoeGroup g = groups[tg.y];
  const uint cnt = g.count, o0 = 64u * tg.x + 16u * sg;
  const bool sg_active = o0 < a.loc;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint o = lane / 2u, p = 8u * (lane % 2u);
  const ulong w_row = (ulong(g.expert) * a.loc + o0 + o) * a.hidden;
  const ulong s_base = ulong(g.expert) * a.loc_blocks * a.hid_blocks;
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
  constexpr uint XLD = 32u + 8u, WLD = 16u + 8u;
  threadgroup half xs[64u * XLD], wg[4u * 16u * WLD], wu[4u * 16u * WLD];
  threadgroup half* wg16 = wg + sg * 16u * WLD;
  threadgroup half* wu16 = wu + sg * 16u * WLD;
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint hid4 = a.hidden >> 2, m_tiles = (cnt + 7u) >> 3, load_r = m_tiles << 3;
  simdgroup_matrix<float, 8, 8> ag[2][4], au[2][4];
  TF_UNROLL for (uint i = 0; i < 2u; ++i)
    TF_UNROLL for (uint m = 0; m < 4u; ++m) ag[i][m] = simdgroup_matrix<float, 8, 8>(0.0f), au[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint c = 0; c < a.hid_blocks; ++c) {
    const half sg_sc = sg_active ? half(gate_s[s_base + ulong(o0 / 128u) * a.hid_blocks + c] * 256.0f) : half(0.0h);
    const half su_sc = sg_active ? half(up_s[s_base + ulong(o0 / 128u) * a.hid_blocks + c] * 256.0f) : half(0.0h);
    for (uint kh = 0; kh < 2u; ++kh) {
      const uint k_base = 128u * c + 64u * kh;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = tid; idx < load_r * 16u; idx += 128u) {
        const uint rr = idx >> 4, cc4 = idx & 15u;
        const float4 v = rr < cnt ? X4[ulong(members[g.mem_off + rr] >> 16) * hid4 + (k_base >> 2) + cc4] : float4(0.0f);
        const uint k0 = cc4 * 4u;
        xs[(k0 + 0u) * XLD + rr] = half(v.x); xs[(k0 + 1u) * XLD + rr] = half(v.y);
        xs[(k0 + 2u) * XLD + rr] = half(v.z); xs[(k0 + 3u) * XLD + rr] = half(v.w);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        for (uint kk = 0; kk < 4u; ++kk) {
          const ulong off = w_row + k_base + 16u * kk + p;
          fp8p_store8_half_sc(*reinterpret_cast<const device uint2*>(gate_w + off), sg_sc, wg16 + o * WLD + p);
          fp8p_store8_half_sc(*reinterpret_cast<const device uint2*>(up_w + off), su_sc, wu16 + o * WLD + p);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          simdgroup_matrix<half, 8, 8> g00, g01, g10, g11, u00, u01, u10, u11, b0, b1;
          simdgroup_load(g00, wg16, WLD, ulong2(0, 0), false); simdgroup_load(g01, wg16, WLD, ulong2(8, 0), false);
          simdgroup_load(g10, wg16, WLD, ulong2(0, 8), false); simdgroup_load(g11, wg16, WLD, ulong2(8, 8), false);
          simdgroup_load(u00, wu16, WLD, ulong2(0, 0), false); simdgroup_load(u01, wu16, WLD, ulong2(8, 0), false);
          simdgroup_load(u10, wu16, WLD, ulong2(0, 8), false); simdgroup_load(u11, wu16, WLD, ulong2(8, 8), false);
          #define FP8P_GU_BOTH(M)                                                                      \
            simdgroup_load(b0, xs + (16u * kk) * XLD + (M) * 8u, XLD, ulong2(0, 0), false);            \
            simdgroup_load(b1, xs + (16u * kk + 8u) * XLD + (M) * 8u, XLD, ulong2(0, 0), false);       \
            simdgroup_multiply_accumulate(ag[0][(M)], g00, b0, ag[0][(M)]);                            \
            simdgroup_multiply_accumulate(ag[1][(M)], g10, b0, ag[1][(M)]);                            \
            simdgroup_multiply_accumulate(ag[0][(M)], g01, b1, ag[0][(M)]);                            \
            simdgroup_multiply_accumulate(ag[1][(M)], g11, b1, ag[1][(M)]);                            \
            simdgroup_multiply_accumulate(au[0][(M)], u00, b0, au[0][(M)]);                            \
            simdgroup_multiply_accumulate(au[1][(M)], u10, b0, au[1][(M)]);                            \
            simdgroup_multiply_accumulate(au[0][(M)], u01, b1, au[0][(M)]);                            \
            simdgroup_multiply_accumulate(au[1][(M)], u11, b1, au[1][(M)]);
          FP8P_GU_BOTH(0)
          if (m_tiles > 1u) { FP8P_GU_BOTH(1) }
          if (m_tiles > 2u) { FP8P_GU_BOTH(2) }
          if (m_tiles > 3u) { FP8P_GU_BOTH(3) }
          #undef FP8P_GU_BOTH
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
  }
  if (!sg_active) return;
  TF_UNROLL for (uint i = 0; i < 2u; ++i) {
    const uint oo = o0 + i * 8u + uint(home.y);
    if (oo >= a.loc) continue;
    TF_UNROLL for (uint m = 0; m < 4u; ++m) {
      const uint r0 = m * 8u + uint(home.x);
      if (r0 < cnt) {
        const uint slot = members[g.mem_off + r0] & 0xFFFFu;
        const float gv = ag[i][m].thread_elements()[0], uv = au[i][m].thread_elements()[0];
        act[ulong(slot) * a.loc + oo] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
      }
      if (r0 + 1u < cnt) {
        const uint slot = members[g.mem_off + r0 + 1u] & 0xFFFFu;
        const float gv = ag[i][m].thread_elements()[1], uv = au[i][m].thread_elements()[1];
        act[ulong(slot) * a.loc + oo] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
      }
    }
  }
#else
  const bool half_tile = cnt <= 16u;
  threadgroup bfloat xs[FP8P_DR * FP8P_DLD];
  threadgroup bfloat ws[4 * 16 * FP8P_DWLD];
  threadgroup bfloat* w16 = ws + sg * 16 * FP8P_DWLD;
  tfp::frag<float> tg_tot[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  tfp::frag<float> tu_tot[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  const uint load_elems = half_tile ? (16u * 128u) : (FP8P_DR * 128u);
  for (uint c = 0; c < a.hid_blocks; ++c) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < load_elems; idx += 128u) {
      const uint rr = idx / 128u, col = idx % 128u;
      xs[rr * FP8P_DLD + col] = rr < cnt ? bfloat(X[ulong(members[g.mem_off + rr] >> 16) * a.hidden + 128u * c + col]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      tfp::frag<float> pg[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
      tfp::frag<float> pu[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
      for (uint kk = 0; kk < 8u; ++kk) {
        const ulong off = w_row + 128u * c + 16u * kk + p;
        const uint2 qg = *reinterpret_cast<const device uint2*>(gate_w + off);
        const uint2 qu = *reinterpret_cast<const device uint2*>(up_w + off);
        fp8p_store8_bf16(qg, w16 + o * FP8P_DWLD + p);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfp::frag<bfloat> wt, x0, x1;
        tfp::frag_get(wt, w16, FP8P_DWLD, 0, 0, home);
        tfp::frag_get_t(x0, xs, FP8P_DLD, 0, 16 * int(kk), home);
        tfp::frag_get_t(x1, xs, FP8P_DLD, 16, 16 * int(kk), home);
        tfp::mma_16x32<false, true>(pg[0], pg[1], wt, x0, x1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        fp8p_store8_bf16(qu, w16 + o * FP8P_DWLD + p);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfp::frag_get(wt, w16, FP8P_DWLD, 0, 0, home);
        tfp::mma_16x32<false, true>(pu[0], pu[1], wt, x0, x1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      const float sg_sc = gate_s[s_base + ulong(o0 / 128u) * a.hid_blocks + c] * 256.0f;
      const float su_sc = up_s[s_base + ulong(o0 / 128u) * a.hid_blocks + c] * 256.0f;
      for (short e = 0; e < 8; ++e) {
        tg_tot[0][e] = fma(sg_sc, pg[0][e], tg_tot[0][e]);
        tu_tot[0][e] = fma(su_sc, pu[0][e], tu_tot[0][e]);
        if (!half_tile) {
          tg_tot[1][e] = fma(sg_sc, pg[1][e], tg_tot[1][e]);
          tu_tot[1][e] = fma(su_sc, pu[1][e], tu_tot[1][e]);
        }
      }
    }
  }
  if (!sg_active) return;
  for (uint h = 0; h < 2u; ++h) {
    for (short e = 0; e < 8; ++e) {
      const uint oo = o0 + uint(home.y) + uint((e >> 2) * 8);
      const uint rr = 16u * h + uint(home.x) + uint(TF_COL(e));
      if (oo >= a.loc || rr >= cnt) continue;
      const uint slot = members[g.mem_off + rr] & 0xFFFFu;
      const float gv = tg_tot[h][e], uv = tu_tot[h][e];
      act[ulong(slot) * a.loc + oo] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
    }
  }
#endif
}

// 4-sg (128 out cols / TG on M3 fast: BN=128 BK=32 uint4 + coalesced epilogue, -13%!) half-MMA down projection.
template <bool ATOMIC>
inline void glm_fp8_moe_prompt_down_impl(const device float* act, const device uchar* down_w, const device float* down_s,
                                         const device Fp8MoeGroup* groups, const device uint* members,
                                         constant Fp8MoeArgs& a, const device float* wts9, device void* dst,
                                         uint2 tg, uint sg, uint lane, uint tid,
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
                                         threadgroup half* xs, threadgroup half* ws
#else
                                         threadgroup bfloat* xs, threadgroup bfloat* ws
#endif
) {
  const Fp8MoeGroup g = groups[tg.y];
  const uint cnt = g.count;
  const short2 home = tfp::frag_home(ushort(lane));
  const ulong s_base = ulong(g.expert) * a.hid_blocks * a.loc_blocks;
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
  const uint o_tg = 128u * tg.x, o0 = o_tg + 32u * sg;
  const bool sg_active = o0 < a.hidden;
  const ulong w_base = (ulong(g.expert) * a.hidden + o0) * a.loc;
  constexpr uint XLD = 32u + 8u, WLD = 32u + 8u;
  threadgroup half* w32 = ws + sg * 32u * WLD;
  const device float4* A4 = reinterpret_cast<const device float4*>(act);
  const uint loc4 = a.loc >> 2, m_tiles = (cnt + 7u) >> 3;
  const uint lr = lane >> 1, lc = (lane & 1u) << 4;
  const uint xr = tid >> 2, xc4 = (tid & 3u) << 1;
  const uint slot_x = xr < cnt ? (members[g.mem_off + xr] & 0xFFFFu) : 0u;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  TF_UNROLL for (uint i = 0; i < 4u; ++i)
    TF_UNROLL for (uint m = 0; m < 4u; ++m) acc[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint c = 0; c < a.loc_blocks; ++c) {
    const half sd_sc = half(down_s[s_base + ulong(o_tg / 128u) * a.loc_blocks + c] * 256.0f);
    for (uint kb = 0; kb < 4u; ++kb) {
      const uint k_base = 128u * c + 32u * kb;
      const float4 v0 = xr < cnt ? clamp(A4[ulong(slot_x) * loc4 + (k_base >> 2) + xc4], -65000.0f, 65000.0f) : float4(0.0f);
      const float4 v1 = xr < cnt ? clamp(A4[ulong(slot_x) * loc4 + (k_base >> 2) + xc4 + 1u], -65000.0f, 65000.0f) : float4(0.0f);
      const uint k0 = xc4 * 4u;
      xs[(k0 + 0u) * XLD + xr] = half(v0.x); xs[(k0 + 1u) * XLD + xr] = half(v0.y);
      xs[(k0 + 2u) * XLD + xr] = half(v0.z); xs[(k0 + 3u) * XLD + xr] = half(v0.w);
      xs[(k0 + 4u) * XLD + xr] = half(v1.x); xs[(k0 + 5u) * XLD + xr] = half(v1.y);
      xs[(k0 + 6u) * XLD + xr] = half(v1.z); xs[(k0 + 7u) * XLD + xr] = half(v1.w);
      if (sg_active) {
        if ((o0 + lr) < a.hidden)
          fp8p_store16_half_sc(*reinterpret_cast<const device uint4*>(down_w + w_base + ulong(lr) * a.loc + k_base + lc), sd_sc, w32 + lr * WLD + lc);
        if ((o0 + lr + 16u) < a.hidden)
          fp8p_store16_half_sc(*reinterpret_cast<const device uint4*>(down_w + w_base + ulong(lr + 16u) * a.loc + k_base + lc), sd_sc, w32 + (lr + 16u) * WLD + lc);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        TF_UNROLL for (uint ks = 0; ks < 4u; ++ks) {
          simdgroup_matrix<half, 8, 8> a0, a1, a2, a3, b;
          simdgroup_load(a0, w32, WLD, ulong2(8u * ks, 0), false);
          simdgroup_load(a1, w32, WLD, ulong2(8u * ks, 8), false);
          simdgroup_load(a2, w32, WLD, ulong2(8u * ks, 16), false);
          simdgroup_load(a3, w32, WLD, ulong2(8u * ks, 24), false);
          const uint xk = (8u * ks) * XLD;
          #define FP8P_DN_M(M)                                                                         \
            simdgroup_load(b, xs + xk + (M) * 8u, XLD, ulong2(0, 0), false);                           \
            simdgroup_multiply_accumulate(acc[0][(M)], a0, b, acc[0][(M)]);                            \
            simdgroup_multiply_accumulate(acc[1][(M)], a1, b, acc[1][(M)]);                            \
            simdgroup_multiply_accumulate(acc[2][(M)], a2, b, acc[2][(M)]);                            \
            simdgroup_multiply_accumulate(acc[3][(M)], a3, b, acc[3][(M)]);
          FP8P_DN_M(0)
          if (m_tiles > 1u) { FP8P_DN_M(1) }
          if (m_tiles > 2u) { FP8P_DN_M(2) }
          if (m_tiles > 3u) { FP8P_DN_M(3) }
          #undef FP8P_DN_M
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (!sg_active) return;
  threadgroup float* stage = reinterpret_cast<threadgroup float*>(w32);
  for (uint half_m = 0u; half_m < 2u; ++half_m) {
    if (half_m * 16u >= cnt) break;
    TF_UNROLL for (uint i = 0u; i < 4u; ++i) {
      const uint lo_o = i * 8u + uint(home.y);
      TF_UNROLL for (uint dm = 0u; dm < 2u; ++dm) {
        const uint m = half_m * 2u + dm, lr0 = dm * 8u + uint(home.x);
        stage[lr0 * 32u + lo_o] = acc[i][m].thread_elements()[0];
        stage[(lr0 + 1u) * 32u + lo_o] = acc[i][m].thread_elements()[1];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rr = 0u; rr < 16u; ++rr) {
      const uint gr = half_m * 16u + rr;
      if (gr >= cnt) break;
      const uint mem = members[g.mem_off + gr], slot = mem & 0xFFFFu;
      if ((o0 + lane) < a.hidden) {
        const float val = stage[rr * 32u + lane];
        if (ATOMIC) atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>(dst) + ulong(mem >> 16) * a.hidden + o0 + lane, val * wts9[slot], memory_order_relaxed);
        else reinterpret_cast<device float*>(dst)[ulong(slot) * a.hidden + o0 + lane] = val;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
#else
  const uint o0 = 64u * tg.x + 16u * sg;
  const bool sg_active = o0 < a.hidden;
  const uint o = lane / 2u, p = 8u * (lane % 2u);
  const ulong w_row = (ulong(g.expert) * a.hidden + o0 + o) * a.loc;
  const bool half_tile = cnt <= 16u;
  threadgroup bfloat* w16 = ws + sg * 16 * FP8P_DWLD;
  tfp::frag<float> total[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  const uint load_elems = half_tile ? (16u * 128u) : (FP8P_DR * 128u);
  for (uint c = 0; c < a.loc_blocks; ++c) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < load_elems; idx += 128u) {
      const uint rr = idx / 128u, col = idx % 128u;
      xs[rr * FP8P_DLD + col] = rr < cnt ? bfloat(act[ulong(members[g.mem_off + rr] & 0xFFFFu) * a.loc + 128u * c + col]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      tfp::frag<float> part[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
      for (uint kk = 0; kk < 8u; ++kk) {
        const uint2 qd = *reinterpret_cast<const device uint2*>(down_w + w_row + 128u * c + 16u * kk + p);
        fp8p_store8_bf16(qd, w16 + o * FP8P_DWLD + p);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfp::frag<bfloat> wt, x0, x1;
        tfp::frag_get(wt, w16, FP8P_DWLD, 0, 0, home);
        tfp::frag_get_t(x0, xs, FP8P_DLD, 0, 16 * int(kk), home);
        tfp::frag_get_t(x1, xs, FP8P_DLD, 16, 16 * int(kk), home);
        tfp::mma_16x32<false, true>(part[0], part[1], wt, x0, x1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      const float sd_sc = down_s[s_base + ulong(o0 / 128u) * a.loc_blocks + c] * 256.0f;
      for (short e = 0; e < 8; ++e) {
        total[0][e] = fma(sd_sc, part[0][e], total[0][e]);
        if (!half_tile) total[1][e] = fma(sd_sc, part[1][e], total[1][e]);
      }
    }
  }
  if (!sg_active) return;
  for (uint h = 0; h < 2u; ++h) {
    for (short e = 0; e < 8; ++e) {
      const uint oo = o0 + uint(home.y) + uint((e >> 2) * 8);
      const uint rr = 16u * h + uint(home.x) + uint(TF_COL(e));
      if (oo >= a.hidden || rr >= cnt) continue;
      const uint mem = members[g.mem_off + rr], slot = mem & 0xFFFFu;
      if (ATOMIC) {
        atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>(dst) + ulong(mem >> 16) * a.hidden + oo, total[h][e] * wts9[slot], memory_order_relaxed);
      } else {
        reinterpret_cast<device float*>(dst)[ulong(slot) * a.hidden + oo] = total[h][e];
      }
    }
  }
#endif
}

kernel void glm_fp8_moe_prompt_down(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]],
                                    const device float* down_s [[buffer(2)]], const device Fp8MoeGroup* groups [[buffer(3)]],
                                    const device uint* members [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]],
                                    device float* z [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint tid [[thread_index_in_threadgroup]]) {
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
  threadgroup half xs[32u * 40u], ws[4u * 32u * 40u];
#else
  threadgroup bfloat xs[FP8P_DR * FP8P_DLD], ws[4 * 16 * FP8P_DWLD];
#endif
  glm_fp8_moe_prompt_down_impl<false>(act, down_w, down_s, groups, members, a, nullptr, z, tg, sg, lane, tid, xs, ws);
}

// Fused down + weighted combine via atomic_float add directly into out[row, hidden] (skips q.z 432MB + moe_comb!).
kernel void glm_fp8_moe_prompt_down_atomic(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]],
                                           const device float* down_s [[buffer(2)]], const device Fp8MoeGroup* groups [[buffer(3)]],
                                           const device uint* members [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]],
                                           const device float* wts9 [[buffer(6)]], device float* out [[buffer(7)]],
                                           uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
#if defined(TF_SIMD_FRAGS) && defined(TF_GLM_M3_FAST)
  threadgroup half xs[32u * 40u], ws[4u * 32u * 40u];
#else
  threadgroup bfloat xs[FP8P_DR * FP8P_DLD], ws[4 * 16 * FP8P_DWLD];
#endif
  glm_fp8_moe_prompt_down_impl<true>(act, down_w, down_s, groups, members, a, wts9, out, tg, sg, lane, tid, xs, ws);
}
