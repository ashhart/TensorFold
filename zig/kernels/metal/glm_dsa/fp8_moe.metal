// GLM-5.3 FP8 E4M3 128x128 block-scaled MoE kernels for decode (1-16 rows) and shared combine.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint FP8_MAXR = 16, FP8_SHARED = 256;

struct Fp8MoeGroup { uint expert, mem_off, count, pad; };
struct Fp8MoeArgs { uint rows, loc, hidden, topk, slots, experts, loc_blocks, hid_blocks; };

// Four E4M3 bytes decoded to half4 / float4 scaled by 2^-8 (the 256.0f factor folds into the per-block scale).
inline half4 fp8_e4m3x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}

inline float4 fp8_e4m3x4(uint q) { return float4(fp8_e4m3x4_h(q)); }
inline float fp8_lanes(float v) { for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d); return v; }
inline float fp8_dot4(uint4 q, half4 a0, half4 a1, half4 a2, half4 a3) {
  return float(dot(fp8_e4m3x4_h(q.x), a0)) + float(dot(fp8_e4m3x4_h(q.y), a1)) +
         float(dot(fp8_e4m3x4_h(q.z), a2)) + float(dot(fp8_e4m3x4_h(q.w), a3));
}

// Single-row fast path: double-buffered TG half4 X cache (12 bar vs 24), uint4 loads, Q=2 128t (713 GB/s cold DRAM!).
kernel void glm_fp8_moe_gu_silu_1_fast(const device float* X [[buffer(0)]], const device uint4* gate_w [[buffer(1)]],
                                       const device float* gate_s [[buffer(2)]], const device uint4* up_w [[buffer(3)]],
                                       const device float* up_s [[buffer(4)]], const device int* pick [[buffer(5)]],
                                       constant Fp8MoeArgs& a [[buffer(6)]], device float* act [[buffer(7)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xh[256];
  const uint slot = tg.y, o0 = tg.x * 8u + sg * 2u;
  const bool active0 = o0 < a.loc && slot < a.slots, has1 = active0 && (o0 + 1u) < a.loc;
  const uint e = slot < a.topk ? uint(pick[slot]) : FP8_SHARED;
  const ulong row_u4_0 = (ulong(e) * a.loc + o0) * (a.hidden / 16u), row_u4_1 = row_u4_0 + (a.hidden / 16u);
  const ulong sc_base = (ulong(e) * a.loc_blocks + o0 / 128u) * a.hid_blocks;
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint c512 = a.hidden / 512u, sub_sb = lane >> 3;
  float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
  xh[tid] = half4(X4[tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = 0u; c < c512; ++c) {
    const uint cur = (c & 1u) * 128u, nxt = ((c + 1u) & 1u) * 128u;
    if (c + 1u < c512) xh[nxt + tid] = half4(X4[(c + 1u) * 128u + tid]);
    if (active0) {
      const half4 x0 = xh[cur + lane * 4u], x1 = xh[cur + lane * 4u + 1u];
      const half4 x2 = xh[cur + lane * 4u + 2u], x3 = xh[cur + lane * 4u + 3u];
      const uint sb = c * 4u + sub_sb;
      const uint4 qg0 = gate_w[row_u4_0 + c * 32u + lane], qu0 = up_w[row_u4_0 + c * 32u + lane];
      const float sg0 = gate_s[sc_base + sb] * 256.0f, su0 = up_s[sc_base + sb] * 256.0f;
      ag0 = fma(fp8_dot4(qg0, x0, x1, x2, x3), sg0, ag0);
      au0 = fma(fp8_dot4(qu0, x0, x1, x2, x3), su0, au0);
      if (has1) {
        const uint4 qg1 = gate_w[row_u4_1 + c * 32u + lane], qu1 = up_w[row_u4_1 + c * 32u + lane];
        ag1 = fma(fp8_dot4(qg1, x0, x1, x2, x3), sg0, ag1);
        au1 = fma(fp8_dot4(qu1, x0, x1, x2, x3), su0, au1);
      }
    }
    if (c + 1u < c512) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active0) return;
  const float gv0 = fp8_lanes(ag0), uv0 = fp8_lanes(au0);
  if (lane == 0u) act[ulong(slot) * a.loc + o0] = precise::divide(gv0, 1.0f + precise::exp(-gv0)) * uv0;
  if (has1) {
    const float gv1 = fp8_lanes(ag1), uv1 = fp8_lanes(au1);
    if (lane == 0u) act[ulong(slot) * a.loc + o0 + 1u] = precise::divide(gv1, 1.0f + precise::exp(-gv1)) * uv1;
  }
}

inline uint fp8_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}

// Single-row Q=4 fused down+combine+peer-BF16-pack: 4 outputs/sg (16/TG), hoisted simd reduce, 646 GB/s on M3 Ultra.
kernel void glm_fp8_moe_down_combine_1_fast(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]],
                                            const device float* down_s [[buffer(2)]], const device int* pick [[buffer(3)]],
                                            const device float* wts [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]],
                                            device float* out [[buffer(6)]], constant uint4& pack [[buffer(7)]],
                                            device uint2* send_base [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint o0 = tg.x * 16u + sg * 4u;
  if (o0 >= a.hidden) return;
  const bool full4 = (o0 + 3u) < a.hidden;
  const uint c512 = a.loc / 512u, rem_blocks = (a.loc % 512u) / 128u;
  const uint sub_sb = lane >> 3, sb_o = o0 >> 7;
  float tot0 = 0.0f, tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f;
  if (a.loc == 512u && full4) {
    const device float4* A4 = reinterpret_cast<const device float4*>(act);
    for (uint slot = 0u; slot < a.slots; ++slot) {
      const uint e = slot < a.topk ? uint(pick[slot]) : FP8_SHARED;
      const float w_slot = slot < a.topk ? wts[slot] : 1.0f;
      const uint a_off = slot * 128u + lane * 4u;
      const half4 a0 = half4(A4[a_off]), a1 = half4(A4[a_off + 1u]), a2 = half4(A4[a_off + 2u]), a3 = half4(A4[a_off + 3u]);
      const float s = down_s[(ulong(e) * a.hid_blocks + sb_o) * 4ul + sub_sb] * 256.0f;
      const device uint4* W4 = reinterpret_cast<const device uint4*>(down_w + (ulong(e) * a.hidden + o0) * 512ul);
      const uint4 q0 = W4[lane], q1 = W4[32u + lane], q2 = W4[64u + lane], q3 = W4[96u + lane];
      tot0 = fma(w_slot, fp8_lanes(fp8_dot4(q0, a0, a1, a2, a3) * s), tot0);
      tot1 = fma(w_slot, fp8_lanes(fp8_dot4(q1, a0, a1, a2, a3) * s), tot1);
      tot2 = fma(w_slot, fp8_lanes(fp8_dot4(q2, a0, a1, a2, a3) * s), tot2);
      tot3 = fma(w_slot, fp8_lanes(fp8_dot4(q3, a0, a1, a2, a3) * s), tot3);
    }
  } else for (uint slot = 0u; slot < a.slots; ++slot) {
    const uint e = slot < a.topk ? uint(pick[slot]) : FP8_SHARED;
    const float w_slot = slot < a.topk ? wts[slot] : 1.0f;
    const ulong row_b0 = (ulong(e) * a.hidden + o0) * a.loc;
    const ulong sc_base = (ulong(e) * a.hid_blocks + sb_o) * a.loc_blocks;
    const device float4* A4 = reinterpret_cast<const device float4*>(act + ulong(slot) * a.loc);
    const device uint4* W4_0 = reinterpret_cast<const device uint4*>(down_w + row_b0);
    const device uint4* W4_1 = reinterpret_cast<const device uint4*>(down_w + row_b0 + a.loc);
    const device uint4* W4_2 = reinterpret_cast<const device uint4*>(down_w + row_b0 + 2ul * a.loc);
    const device uint4* W4_3 = reinterpret_cast<const device uint4*>(down_w + row_b0 + 3ul * a.loc);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint c = 0u; c < c512; ++c) {
      const uint a_off = c * 128u + lane * 4u;
      const half4 a0 = half4(A4[a_off]), a1 = half4(A4[a_off + 1u]), a2 = half4(A4[a_off + 2u]), a3 = half4(A4[a_off + 3u]);
      const float s = down_s[sc_base + c * 4u + sub_sb] * 256.0f;
      s0 = fma(fp8_dot4(W4_0[c * 32u + lane], a0, a1, a2, a3), s, s0);
      if (full4) {
        s1 = fma(fp8_dot4(W4_1[c * 32u + lane], a0, a1, a2, a3), s, s1);
        s2 = fma(fp8_dot4(W4_2[c * 32u + lane], a0, a1, a2, a3), s, s2);
        s3 = fma(fp8_dot4(W4_3[c * 32u + lane], a0, a1, a2, a3), s, s3);
      } else {
        for (uint k = 1u; k < 4u && (o0 + k) < a.hidden; ++k) {
          const float dk = fp8_dot4(reinterpret_cast<const device uint4*>(down_w + row_b0 + ulong(k) * a.loc)[c * 32u + lane], a0, a1, a2, a3);
          if (k == 1u) s1 = fma(dk, s, s1); else if (k == 2u) s2 = fma(dk, s, s2);
        }
      }
    }
    if (rem_blocks > 0u) {
      const device float4* A1 = A4 + c512 * 128u;
      for (uint r = 0u; r < rem_blocks; ++r) {
        const half4 av = half4(A1[r * 32u + lane]);
        const float s = down_s[sc_base + c512 * 4u + r] * 256.0f;
        const uint off1 = r * 32u + lane;
        const ulong base1 = row_b0 + ulong(c512) * 512ul;
        s0 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + base1)[off1]), av)), s, s0);
        if ((o0 + 1u) < a.hidden) s1 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + base1 + a.loc)[off1]), av)), s, s1);
        if ((o0 + 2u) < a.hidden) s2 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + base1 + 2ul * a.loc)[off1]), av)), s, s2);
        if (full4) s3 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + base1 + 3ul * a.loc)[off1]), av)), s, s3);
      }
    }
    tot0 = fma(w_slot, fp8_lanes(s0), tot0);
    if ((o0 + 1u) < a.hidden) tot1 = fma(w_slot, fp8_lanes(s1), tot1);
    if ((o0 + 2u) < a.hidden) tot2 = fma(w_slot, fp8_lanes(s2), tot2);
    if (full4) tot3 = fma(w_slot, fp8_lanes(s3), tot3);
  }
  const float v0 = tot0, v1 = tot1, v2 = tot2, v3 = tot3;
  if (lane != 0u) return;
  if (full4) {
    *reinterpret_cast<device float4*>(out + o0) = float4(v0, v1, v2, v3);
    if (pack.y > 1u) {
      const uint2 pk = uint2(fp8_bf16x2(v0, v1), fp8_bf16x2(v2, v3));
      const ulong idx2 = ulong(o0 >> 2);
      for (uint p = 0u; p < pack.y; ++p) if (p != pack.x) send_base[ulong(p) * pack.z + idx2] = pk;
    }
  } else {
    out[o0] = v0; if ((o0 + 1u) < a.hidden) out[o0 + 1u] = v1; if ((o0 + 2u) < a.hidden) out[o0 + 2u] = v2;
  }
}

// Fast direct job builder for native FP8 MoE decode (2-16 rows): reads pick & wts directly, skipping moe_route.metal.
kernel void glm_fp8_moe_jobs_direct(const device int* pick [[buffer(0)]], const device float* wts [[buffer(1)]],
                                    constant Fp8MoeArgs& a [[buffer(2)]], device Fp8MoeGroup* groups [[buffer(3)]],
                                    device uint* members [[buffer(4)]], device float* wts9 [[buffer(5)]],
                                    device uint* dispatch [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[FP8_SHARED], cursor[FP8_SHARED];
  threadgroup uint g_idx[FP8_SHARED], total_routed;
  const uint nex = min(a.experts, FP8_SHARED);
  if (t < nex) {
    atomic_store_explicit(&counts[t], 0u, memory_order_relaxed);
    atomic_store_explicit(&cursor[t], 0u, memory_order_relaxed);
  }
  for (uint p = t; p < a.rows * a.slots; p += 256u) {
    const uint r = p / a.slots, k = p % a.slots;
    wts9[p] = k < a.topk ? wts[r * a.topk + k] : 1.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total_picks = a.rows * a.topk;
  for (uint p = t; p < total_picks; p += 256u) {
    const uint e = uint(pick[p]);
    if (e < nex) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint n = 0u;
    for (uint e = 0u; e < nex; ++e) {
      const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
      if (c > 0u) { g_idx[e] = n; groups[n] = Fp8MoeGroup{e, n * FP8_MAXR, c, 0u}; ++n; }
    }
    groups[n] = Fp8MoeGroup{FP8_SHARED, n * FP8_MAXR, a.rows, 0u};
    total_routed = n;
    dispatch[0] = (a.rows <= 5u) ? ((a.loc + 3u) >> 2) : ((a.loc + 7u) >> 3); dispatch[1] = n + 1u; dispatch[2] = 1u;
    dispatch[4] = (a.hidden + 15u) >> 4; dispatch[5] = n + 1u; dispatch[6] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < total_picks; p += 256u) {
    const uint e = uint(pick[p]);
    if (e < nex) {
      const uint r = p / a.topk, k = p % a.topk;
      const uint pos = atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
      members[g_idx[e] * FP8_MAXR + pos] = (r << 16) | (r * a.slots + k);
    }
  }
  if (t < a.rows) members[total_routed * FP8_MAXR + t] = (t << 16) | (t * a.slots + a.slots - 1u);
}

// Fallback float4 -> half4 companion cast (used only when TP=1 or non-fused producer).
kernel void glm_cast_xh(const device float4* src [[buffer(0)]], device half4* dst [[buffer(1)]],
                        constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) dst[gid] = half4(src[gid]);
}

// Zero-barrier R=2..8 FP8 MoE fused gate+up+SiLU*up reading pre-cast half4 Xh (routed or standalone shared!).
template <uint R, uint SG_PER_TG, bool SHARED>
inline void glm_fp8_moe_gu_tg_impl(const device half4* Xh, const device uint4* gate_w, const device float* gate_s,
                                   const device uint4* up_w, const device float* up_s, const device Fp8MoeGroup* groups,
                                   const device uint* members, constant Fp8MoeArgs& a, device float* act, uint2 tg,
                                   uint sg, uint lane) {
  const uint o = tg.x * SG_PER_TG + sg;
  if (o >= a.loc) return;
  const uint ex = SHARED ? FP8_SHARED : groups[tg.y].expert;
  const uint mem_off = SHARED ? 0u : groups[tg.y].mem_off;
  const uint cnt = SHARED ? R : min(groups[tg.y].count, R);
  const ulong row = (ulong(ex) * a.loc + o) * (a.hidden / 16u);
  const ulong sc_base = (ulong(ex) * a.loc_blocks + o / 128u) * a.hid_blocks;
  const uint c512 = a.hidden / 512u, sub_sb = lane >> 3, hid4 = a.hidden >> 2;
  if (!SHARED && cnt == 1u) {
    const uint m0 = members[mem_off];
    const device half4* X0 = Xh + ulong(m0 >> 16) * hid4;
    float ag = 0.0f, au = 0.0f;
    for (uint c = 0u; c < c512; ++c) {
      const uint b = c * 128u + lane * 4u, sb = c * 4u + sub_sb;
      const half4 x0 = X0[b], x1 = X0[b + 1u], x2 = X0[b + 2u], x3 = X0[b + 3u];
      const float sg0 = gate_s[sc_base + sb] * 256.0f, su0 = up_s[sc_base + sb] * 256.0f;
      ag = fma(fp8_dot4(gate_w[row + c * 32u + lane], x0, x1, x2, x3), sg0, ag);
      au = fma(fp8_dot4(up_w[row + c * 32u + lane], x0, x1, x2, x3), su0, au);
    }
    const float gv = fp8_lanes(ag), uv = fp8_lanes(au);
    if (lane == 0u) act[ulong(m0 & 0xFFFFu) * a.loc + o] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
    return;
  }
  uint r_id[R];
  #pragma clang loop unroll(full)
  for (uint m = 0u; m < R; ++m) r_id[m] = SHARED ? m : ((m < cnt) ? (members[mem_off + m] >> 16) : 0u);
  float ag[R], au[R];
  #pragma clang loop unroll(full)
  for (uint m = 0u; m < R; ++m) ag[m] = 0.0f, au[m] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint sb = c * 4u + sub_sb;
    const float sg0 = gate_s[sc_base + sb] * 256.0f, su0 = up_s[sc_base + sb] * 256.0f;
    const uint4 qg = gate_w[row + c * 32u + lane], qu = up_w[row + c * 32u + lane];
    const half4 wg0 = fp8_e4m3x4_h(qg.x), wg1 = fp8_e4m3x4_h(qg.y), wg2 = fp8_e4m3x4_h(qg.z), wg3 = fp8_e4m3x4_h(qg.w);
    const half4 wu0 = fp8_e4m3x4_h(qu.x), wu1 = fp8_e4m3x4_h(qu.y), wu2 = fp8_e4m3x4_h(qu.z), wu3 = fp8_e4m3x4_h(qu.w);
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < R; ++m) if (SHARED || m < cnt) {
      const ulong b = ulong(r_id[m]) * hid4 + c * 128u + lane * 4u;
      const half4 x0 = Xh[b], x1 = Xh[b + 1u], x2 = Xh[b + 2u], x3 = Xh[b + 3u];
      ag[m] = fma(float(dot(wg0, x0)) + float(dot(wg1, x1)) + float(dot(wg2, x2)) + float(dot(wg3, x3)), sg0, ag[m]);
      au[m] = fma(float(dot(wu0, x0)) + float(dot(wu1, x1)) + float(dot(wu2, x2)) + float(dot(wu3, x3)), su0, au[m]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint m = 0u; m < R; ++m) if (SHARED || m < cnt) {
    const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
    const float gv = fp8_lanes(ag[m]), uv = fp8_lanes(au[m]);
    if (lane == 0u) act[ulong(slot) * a.loc + o] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
  }
}

#define GLM_FP8_MOE_GU(R, SG) \
  kernel void glm_fp8_moe_gu_silu_##R(const device half4* Xh [[buffer(0)]], const device uint4* gate_w [[buffer(1)]], const device float* gate_s [[buffer(2)]], const device uint4* up_w [[buffer(3)]], const device float* up_s [[buffer(4)]], const device Fp8MoeGroup* gr [[buffer(5)]], const device uint* members [[buffer(6)]], constant Fp8MoeArgs& a [[buffer(7)]], device float* act [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { glm_fp8_moe_gu_tg_impl<R, SG, false>(Xh, gate_w, gate_s, up_w, up_s, gr, members, a, act, tg, sg, lane); } \
  kernel void glm_fp8_moe_shared_gu_##R(const device half4* Xh [[buffer(0)]], const device uint4* gate_w [[buffer(1)]], const device float* gate_s [[buffer(2)]], const device uint4* up_w [[buffer(3)]], const device float* up_s [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]], device float* act [[buffer(6)]], uint gx [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { glm_fp8_moe_gu_tg_impl<R, 4, true>(Xh, gate_w, gate_s, up_w, up_s, nullptr, nullptr, a, act, uint2(gx, 0u), sg, lane); }
GLM_FP8_MOE_GU(2, 4) GLM_FP8_MOE_GU(3, 4) GLM_FP8_MOE_GU(4, 4) GLM_FP8_MOE_GU(5, 4) GLM_FP8_MOE_GU(6, 8) GLM_FP8_MOE_GU(7, 8) GLM_FP8_MOE_GU(8, 8)

// Fast R=2..8 (MTP decode) spill-free Q=2 256t (16 outs/TG) down projection (routed or standalone shared!).
template <uint R, bool SHARED>
inline void glm_fp8_moe_dn_tg_impl(const device float* act, const device uchar* down_w, const device float* down_s,
                                   const device Fp8MoeGroup* groups, const device uint* members, constant Fp8MoeArgs& a,
                                   device float* z, uint2 tg, uint sg, uint lane, uint tid, threadgroup half4* xs4) {
  const uint ex = SHARED ? FP8_SHARED : groups[tg.y].expert;
  const uint mem_off = SHARED ? 0u : groups[tg.y].mem_off;
  const uint o0 = tg.x * 16u + sg * 2u, cnt = SHARED ? R : min(groups[tg.y].count, R);
  const bool active0 = o0 < a.hidden, active1 = (o0 + 1u) < a.hidden;
  const ulong row_b0 = (ulong(ex) * a.hidden + o0) * a.loc, row_b1 = row_b0 + a.loc;
  const ulong sc_base = (ulong(ex) * a.hid_blocks + o0 / 128u) * a.loc_blocks;
  const device float4* A4 = reinterpret_cast<const device float4*>(act);
  const uint c512 = a.loc / 512u, rem_blocks = (a.loc % 512u) / 128u, sub_sb = lane >> 3, loc4 = a.loc >> 2;
  // TP4 (loc=512, c512==1) unified 1-barrier TG-cached fast path (cnt>=1) + float2 store: up to 758 GB/s!
  if (c512 == 1u && rem_blocks == 0u) {
    for (uint idx = tid; idx < cnt * 128u; idx += 256u) {
      const uint m = idx >> 7, col4 = idx & 127u;
      const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
      xs4[idx] = half4(A4[ulong(slot) * loc4 + col4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active0) return;
    const float s = down_s[sc_base + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(down_w + row_b0 + lane * 16u);
    const half4 w00 = fp8_e4m3x4_h(q0.x), w01 = fp8_e4m3x4_h(q0.y), w02 = fp8_e4m3x4_h(q0.z), w03 = fp8_e4m3x4_h(q0.w);
    const uint4 q1 = active1 ? *reinterpret_cast<const device uint4*>(down_w + row_b1 + lane * 16u) : uint4(0u);
    const half4 w10 = fp8_e4m3x4_h(q1.x), w11 = fp8_e4m3x4_h(q1.y), w12 = fp8_e4m3x4_h(q1.z), w13 = fp8_e4m3x4_h(q1.w);
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < R; ++m) if (SHARED || m < cnt) {
      const uint b = m * 128u + lane * 4u;
      const half4 x0 = xs4[b], x1 = xs4[b + 1u], x2 = xs4[b + 2u], x3 = xs4[b + 3u];
      const float v0 = fp8_lanes((float(dot(w00, x0)) + float(dot(w01, x1)) +
                                  float(dot(w02, x2)) + float(dot(w03, x3))) * s);
      const float v1 = active1 ? fp8_lanes((float(dot(w10, x0)) + float(dot(w11, x1)) +
                                            float(dot(w12, x2)) + float(dot(w13, x3))) * s) : 0.0f;
      if (lane == 0u) {
        const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
        const ulong at = ulong(slot) * a.hidden + o0;
        if (active1) *reinterpret_cast<device float2*>(z + at) = float2(v0, v1);
        else z[at] = v0;
      }
    }
    return;
  }
  if (!SHARED && cnt == 1u) {
    const uint s0 = members[mem_off] & 0xFFFFu;
    const device float4* A0 = A4 + ulong(s0) * loc4;
    float acc0 = 0.0f, acc1 = 0.0f;
    if (c512 > 0u && tid < 128u) xs4[tid] = half4(A0[tid]);
    if (c512 > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0u; c < c512; ++c) {
      const uint cur = (c & 1u) * 128u, nxt = ((c + 1u) & 1u) * 128u;
      if (c + 1u < c512 && tid < 128u) xs4[nxt + tid] = half4(A0[(c + 1u) * 128u + tid]);
      if (active0) {
        const uint b = cur + lane * 4u;
        const half4 a0 = xs4[b], a1 = xs4[b + 1u], a2 = xs4[b + 2u], a3 = xs4[b + 3u];
        const float s = down_s[sc_base + c * 4u + sub_sb] * 256.0f;
        acc0 = fma(fp8_dot4(*reinterpret_cast<const device uint4*>(down_w + row_b0 + c * 512u + lane * 16u), a0, a1, a2, a3), s, acc0);
        if (active1) acc1 = fma(fp8_dot4(*reinterpret_cast<const device uint4*>(down_w + row_b1 + c * 512u + lane * 16u), a0, a1, a2, a3), s, acc1);
      }
      if (c + 1u < c512) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!active0) return;
    if (rem_blocks > 0u) {
      for (uint r = 0u; r < rem_blocks; ++r) {
        const half4 av = half4(A0[c512 * 128u + r * 32u + lane]);
        const float s = down_s[sc_base + c512 * 4u + r] * 256.0f;
        acc0 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + row_b0 + ulong(c512) * 512u)[r * 32u + lane]), av)), s, acc0);
        if (active1) acc1 = fma(float(dot(fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + row_b1 + ulong(c512) * 512u)[r * 32u + lane]), av)), s, acc1);
      }
    }
    const float v0 = fp8_lanes(acc0); if (lane == 0u) z[ulong(s0) * a.hidden + o0] = v0;
    if (active1) { const float v1 = fp8_lanes(acc1); if (lane == 0u) z[ulong(s0) * a.hidden + o0 + 1u] = v1; }
    return;
  }
  float acc0[R], acc1[R];
  #pragma clang loop unroll(full)
  for (uint m = 0u; m < R; ++m) acc0[m] = 0.0f, acc1[m] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    for (uint idx = tid; idx < cnt * 128u; idx += 256u) {
      const uint m = idx >> 7, col4 = idx & 127u;
      const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
      xs4[idx] = half4(A4[ulong(slot) * loc4 + c * 128u + col4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active0) {
      const float s = down_s[sc_base + c * 4u + sub_sb] * 256.0f;
      const uint4 q0 = *reinterpret_cast<const device uint4*>(down_w + row_b0 + c * 512u + lane * 16u);
      const uint4 q1 = active1 ? *reinterpret_cast<const device uint4*>(down_w + row_b1 + c * 512u + lane * 16u) : uint4(0u);
      const half4 w00 = fp8_e4m3x4_h(q0.x), w01 = fp8_e4m3x4_h(q0.y), w02 = fp8_e4m3x4_h(q0.z), w03 = fp8_e4m3x4_h(q0.w);
      const half4 w10 = fp8_e4m3x4_h(q1.x), w11 = fp8_e4m3x4_h(q1.y), w12 = fp8_e4m3x4_h(q1.z), w13 = fp8_e4m3x4_h(q1.w);
      #pragma clang loop unroll(full)
      for (uint m = 0u; m < R; ++m) if (SHARED || m < cnt) {
        const uint b = m * 128u + lane * 4u;
        const half4 x0 = xs4[b], x1 = xs4[b + 1u], x2 = xs4[b + 2u], x3 = xs4[b + 3u];
        acc0[m] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s, acc0[m]);
        if (active1) acc1[m] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s, acc1[m]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (rem_blocks > 0u) {
    for (uint r = 0u; r < rem_blocks; ++r) {
      for (uint idx = tid; idx < cnt * 32u; idx += 256u) {
        const uint m = idx >> 5, col4 = idx & 31u;
        const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
        xs4[idx] = half4(A4[ulong(slot) * loc4 + c512 * 128u + r * 32u + col4]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active0) {
        const float s = down_s[sc_base + c512 * 4u + r] * 256.0f;
        const half4 wd0 = fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + row_b0 + ulong(c512) * 512u)[r * 32u + lane]);
        const half4 wd1 = active1 ? fp8_e4m3x4_h(reinterpret_cast<const device uint*>(down_w + row_b1 + ulong(c512) * 512u)[r * 32u + lane]) : half4(0.0h);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < R; ++m) if (SHARED || m < cnt) {
          const half4 xv = xs4[m * 32u + lane];
          acc0[m] = fma(float(dot(wd0, xv)), s, acc0[m]);
          if (active1) acc1[m] = fma(float(dot(wd1, xv)), s, acc1[m]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (!active0) return;
  for (uint m = 0u; m < cnt; ++m) {
    const uint slot = SHARED ? (m * a.slots + a.slots - 1u) : (members[mem_off + m] & 0xFFFFu);
    const float v0 = fp8_lanes(acc0[m]); if (lane == 0u) z[ulong(slot) * a.hidden + o0] = v0;
    if (active1) { const float v1 = fp8_lanes(acc1[m]); if (lane == 0u) z[ulong(slot) * a.hidden + o0 + 1u] = v1; }
  }
}

#define GLM_FP8_MOE_DN(R) \
  kernel void glm_fp8_moe_down_##R(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]], const device float* down_s [[buffer(2)]], const device Fp8MoeGroup* gr [[buffer(3)]], const device uint* members [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]], device float* z [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { threadgroup half4 xs4[R * 128]; glm_fp8_moe_dn_tg_impl<R, false>(act, down_w, down_s, gr, members, a, z, tg, sg, lane, tid, xs4); } \
  kernel void glm_fp8_moe_shared_down_##R(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]], const device float* down_s [[buffer(2)]], constant Fp8MoeArgs& a [[buffer(3)]], device float* z [[buffer(4)]], uint gx [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { threadgroup half4 xs4[R * 128]; glm_fp8_moe_dn_tg_impl<R, true>(act, down_w, down_s, nullptr, nullptr, a, z, uint2(gx, 0u), sg, lane, tid, xs4); }
GLM_FP8_MOE_DN(2) GLM_FP8_MOE_DN(3) GLM_FP8_MOE_DN(4) GLM_FP8_MOE_DN(5) GLM_FP8_MOE_DN(6) GLM_FP8_MOE_DN(7) GLM_FP8_MOE_DN(8)

// Fast batched decode (9-16 rows) fused gate+up+SiLU*up: 256t, 8 outs/TG, uint4 loads, half4 TG cache (8-row chunks).
kernel void glm_fp8_moe_gu_silu_16(const device float* X [[buffer(0)]], const device uint4* gate_w [[buffer(1)]],
                                   const device float* gate_s [[buffer(2)]], const device uint4* up_w [[buffer(3)]],
                                   const device float* up_s [[buffer(4)]], const device Fp8MoeGroup* groups [[buffer(5)]],
                                   const device uint* members [[buffer(6)]], constant Fp8MoeArgs& a [[buffer(7)]],
                                   device float* act [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xs4[8 * 128];
  const Fp8MoeGroup g = groups[tg.y];
  const uint o = tg.x * 8u + sg, cnt = min(g.count, 16u);
  const bool active = o < a.loc;
  const ulong row = (ulong(g.expert) * a.loc + o) * (a.hidden / 16u);
  const ulong sc_base = (ulong(g.expert) * a.loc_blocks + o / 128u) * a.hid_blocks;
  const device float4* X4 = reinterpret_cast<const device float4*>(X);
  const uint c512 = a.hidden / 512u, sub_sb = lane >> 3, hid4 = a.hidden >> 2;
  for (uint m0 = 0u; m0 < cnt; m0 += 8u) {
    const uint cm = min(8u, cnt - m0);
    float ag[8], au[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) ag[m] = 0.0f, au[m] = 0.0f;
    for (uint c = 0u; c < c512; ++c) {
      for (uint idx = tid; idx < cm * 128u; idx += 256u) {
        const uint r = members[g.mem_off + m0 + (idx >> 7)] >> 16;
        xs4[idx] = half4(X4[ulong(r) * hid4 + c * 128u + (idx & 127u)]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        const uint sb = c * 4u + sub_sb;
        const float sg0 = gate_s[sc_base + sb] * 256.0f, su0 = up_s[sc_base + sb] * 256.0f;
        const uint4 qg = gate_w[row + c * 32u + lane], qu = up_w[row + c * 32u + lane];
        const half4 wg0 = fp8_e4m3x4_h(qg.x), wg1 = fp8_e4m3x4_h(qg.y), wg2 = fp8_e4m3x4_h(qg.z), wg3 = fp8_e4m3x4_h(qg.w);
        const half4 wu0 = fp8_e4m3x4_h(qu.x), wu1 = fp8_e4m3x4_h(qu.y), wu2 = fp8_e4m3x4_h(qu.z), wu3 = fp8_e4m3x4_h(qu.w);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 8u; ++m) if (m < cm) {
          const uint b = m * 128u + lane * 4u;
          const half4 x0 = xs4[b], x1 = xs4[b + 1u], x2 = xs4[b + 2u], x3 = xs4[b + 3u];
          ag[m] = fma(float(dot(wg0, x0)) + float(dot(wg1, x1)) + float(dot(wg2, x2)) + float(dot(wg3, x3)), sg0, ag[m]);
          au[m] = fma(float(dot(wu0, x0)) + float(dot(wu1, x1)) + float(dot(wu2, x2)) + float(dot(wu3, x3)), su0, au[m]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) for (uint m = 0u; m < cm; ++m) {
      const uint slot = members[g.mem_off + m0 + m] & 0xFFFFu;
      const float gv = fp8_lanes(ag[m]), uv = fp8_lanes(au[m]);
      if (lane == 0u) act[ulong(slot) * a.loc + o] = precise::divide(gv, 1.0f + precise::exp(-gv)) * uv;
    }
  }
}

// Fast batched decode (9-16 rows) down: 256t, 16 outs/TG (Q=2), uint4 loads, half4 TG cache (8-row chunks).
kernel void glm_fp8_moe_down_16(const device float* act [[buffer(0)]], const device uchar* down_w [[buffer(1)]],
                                const device float* down_s [[buffer(2)]], const device Fp8MoeGroup* groups [[buffer(3)]],
                                const device uint* members [[buffer(4)]], constant Fp8MoeArgs& a [[buffer(5)]],
                                device float* z [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xs4[8 * 128];
  const Fp8MoeGroup g = groups[tg.y];
  const uint o0 = tg.x * 16u + sg * 2u, cnt = min(g.count, 16u);
  const bool active0 = o0 < a.hidden, active1 = (o0 + 1u) < a.hidden;
  const ulong row_b0 = (ulong(g.expert) * a.hidden + o0) * a.loc, row_b1 = row_b0 + a.loc;
  const ulong sc_base = (ulong(g.expert) * a.hid_blocks + o0 / 128u) * a.loc_blocks;
  const device float4* A4 = reinterpret_cast<const device float4*>(act);
  const device uint4* W4_0 = reinterpret_cast<const device uint4*>(down_w + row_b0);
  const device uint4* W4_1 = reinterpret_cast<const device uint4*>(down_w + row_b1);
  const uint c512 = a.loc / 512u, rem_blocks = (a.loc % 512u) / 128u, sub_sb = lane >> 3, loc4 = a.loc >> 2;
  for (uint m0 = 0u; m0 < cnt; m0 += 8u) {
    const uint cm = min(8u, cnt - m0);
    float acc0[8], acc1[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) acc0[m] = 0.0f, acc1[m] = 0.0f;
    for (uint c = 0u; c < c512; ++c) {
      for (uint idx = tid; idx < cm * 128u; idx += 256u) {
        const uint m = idx >> 7, col4 = idx & 127u, slot = members[g.mem_off + m0 + m] & 0xFFFFu;
        xs4[idx] = half4(A4[ulong(slot) * loc4 + c * 128u + col4]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active0) {
        const float s = down_s[sc_base + c * 4u + sub_sb] * 256.0f;
        const uint4 q0 = W4_0[c * 32u + lane], q1 = active1 ? W4_1[c * 32u + lane] : uint4(0u);
        const half4 w00 = fp8_e4m3x4_h(q0.x), w01 = fp8_e4m3x4_h(q0.y), w02 = fp8_e4m3x4_h(q0.z), w03 = fp8_e4m3x4_h(q0.w);
        const half4 w10 = fp8_e4m3x4_h(q1.x), w11 = fp8_e4m3x4_h(q1.y), w12 = fp8_e4m3x4_h(q1.z), w13 = fp8_e4m3x4_h(q1.w);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 8u; ++m) if (m < cm) {
          const uint b = m * 128u + lane * 4u;
          const half4 x0 = xs4[b], x1 = xs4[b + 1u], x2 = xs4[b + 2u], x3 = xs4[b + 3u];
          acc0[m] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s, acc0[m]);
          if (active1) acc1[m] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s, acc1[m]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rem_blocks > 0u) {
      const device uint* W1_0 = reinterpret_cast<const device uint*>(down_w + row_b0 + ulong(c512) * 512ul);
      const device uint* W1_1 = reinterpret_cast<const device uint*>(down_w + row_b1 + ulong(c512) * 512ul);
      for (uint r = 0u; r < rem_blocks; ++r) {
        for (uint idx = tid; idx < cm * 32u; idx += 256u) {
          const uint m = idx >> 5, col4 = idx & 31u, slot = members[g.mem_off + m0 + m] & 0xFFFFu;
          xs4[idx] = half4(A4[ulong(slot) * loc4 + c512 * 128u + r * 32u + col4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active0) {
          const float s = down_s[sc_base + c512 * 4u + r] * 256.0f;
          const half4 wd0 = fp8_e4m3x4_h(W1_0[r * 32u + lane]), wd1 = active1 ? fp8_e4m3x4_h(W1_1[r * 32u + lane]) : half4(0.0h);
          for (uint m = 0u; m < cm; ++m) {
            const half4 xv = xs4[m * 32u + lane];
            acc0[m] = fma(float(dot(wd0, xv)), s, acc0[m]);
            if (active1) acc1[m] = fma(float(dot(wd1, xv)), s, acc1[m]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (active0) for (uint m = 0u; m < cm; ++m) {
      const uint slot = members[g.mem_off + m0 + m] & 0xFFFFu;
      const float v0 = fp8_lanes(acc0[m]); if (lane == 0u) z[ulong(slot) * a.hidden + o0] = v0;
      if (active1) { const float v1 = fp8_lanes(acc1[m]); if (lane == 0u) z[ulong(slot) * a.hidden + o0 + 1u] = v1; }
    }
  }
}

// Vectorized float4 deterministic weighted sum across all 9 slots (fixed k=0..8 order, 64t) into out[row, hidden].
kernel void glm_fp8_moe_combine(const device float4* z4 [[buffer(0)]], const device float* wts9 [[buffer(1)]], constant Fp8MoeArgs& a [[buffer(2)]], device float4* out4 [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, i4 = tg.x * 64u + t, hid4 = a.hidden >> 2;
  if (r >= a.rows || i4 >= hid4) return;
  float4 sum = float4(0.0f);
  const ulong bs = ulong(r) * a.slots;
  for (uint k = 0; k < a.slots; ++k) sum = fma(float4(wts9[bs + k]), z4[(bs + k) * hid4 + i4], sum);
  out4[ulong(r) * hid4 + i4] = sum;
}
// Vectorized float4 combine + peer BF16 uint2 pack for R=2..8 decode (64t hoisted unroll, 0-ULP!).
kernel void glm_fp8_moe_combine_pack_bf16(const device float4* z4 [[buffer(0)]], const device float* wts9 [[buffer(1)]], constant Fp8MoeArgs& a [[buffer(2)]], device float4* out4 [[buffer(3)]], constant uint4& pack [[buffer(4)]], device uint2* send_base [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, i4 = tg.x * 64u + t, hid4 = a.hidden >> 2;
  if (r >= a.rows || i4 >= hid4) return;
  const ulong bs = ulong(r) * a.slots;
  float4 sum = float4(0.0f);
  if (a.slots == 9u) {
    const float w0 = wts9[bs], w1 = wts9[bs+1u], w2 = wts9[bs+2u], w3 = wts9[bs+3u], w4 = wts9[bs+4u], w5 = wts9[bs+5u], w6 = wts9[bs+6u], w7 = wts9[bs+7u], w8 = wts9[bs+8u];
    const float4 v0 = z4[(bs+0u)*hid4+i4], v1 = z4[(bs+1u)*hid4+i4], v2 = z4[(bs+2u)*hid4+i4], v3 = z4[(bs+3u)*hid4+i4], v4 = z4[(bs+4u)*hid4+i4], v5 = z4[(bs+5u)*hid4+i4], v6 = z4[(bs+6u)*hid4+i4], v7 = z4[(bs+7u)*hid4+i4], v8 = z4[(bs+8u)*hid4+i4];
    sum = fma(float4(w0), v0, sum); sum = fma(float4(w1), v1, sum); sum = fma(float4(w2), v2, sum); sum = fma(float4(w3), v3, sum);
    sum = fma(float4(w4), v4, sum); sum = fma(float4(w5), v5, sum); sum = fma(float4(w6), v6, sum); sum = fma(float4(w7), v7, sum); sum = fma(float4(w8), v8, sum);
  } else for (uint k = 0u; k < a.slots; ++k) sum = fma(float4(wts9[bs + k]), z4[(bs + k) * hid4 + i4], sum);
  const ulong idx2 = ulong(r) * hid4 + i4;
  out4[idx2] = sum;
  if (pack.y > 1u) {
    const uint2 pk = uint2(fp8_bf16x2(sum.x, sum.y), fp8_bf16x2(sum.z, sum.w));
    for (uint p = 0u; p < pack.y; ++p) if (p != pack.x) send_base[ulong(p) * pack.z + idx2] = pk;
  }
}
