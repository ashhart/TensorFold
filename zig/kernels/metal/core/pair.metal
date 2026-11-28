// N-rank (1..8 Macs) tensor- and pipeline-parallel exchanges inside a Metal command buffer over MCDMA RDMA.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint POSTED = 0, GAVE_UP = 16, SKIP_BASE = 1024, SKIP_MASK = 2047;
// Flags sit on 64-byte cache lines (16 uint32 words apart) in the flags page.
constant constexpr uint FLAG_STRIDE_U32 = 16;

// Signal CPU service thread to skip exchange x when gated off (lockstep across all TP ranks).
inline void mesh_skip_post(device atomic_uint* sync, uint x, uint r, uint t, uint tp_size) {
  if (tp_size <= 1u || r != 0u) return;
  if (t == 0u) atomic_store_explicit(&sync[SKIP_BASE + (x & SKIP_MASK)], x, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0u) atomic_store_explicit(&sync[POSTED], x, memory_order_relaxed);
}

// Poll until every other TP peer in [0, tp_size) has landed exchange x.
inline void mesh_poll_flags(device atomic_uint* sync, device atomic_uint* flags, uint tp_rank, uint tp_size, uint x) {
  for (uint p = 0; p < tp_size; ++p) {
    if (p == tp_rank) continue;
    uint polls = 0;
    while (int(atomic_load_explicit(&flags[p * FLAG_STRIDE_U32], memory_order_relaxed) - x) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[GAVE_UP], 1u, memory_order_relaxed); return; }
    }
  }
}

// Thread 0 posts exchange x and polls until every other TP peer in [0, tp_size) has landed x.
inline void mesh_post_wait(device atomic_uint* sync, device atomic_uint* flags, uint tp_rank, uint tp_size, uint x, uint t) {
  if (t != 0u || tp_size <= 1u) return;
  atomic_store_explicit(&sync[POSTED], x, memory_order_relaxed);
  mesh_poll_flags(sync, flags, tp_rank, tp_size, x);
}

inline float pair_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

struct PairRows {
  uint width, stride, rank, tp_size;
  uint x, slot_u32, bf16_wire, send_stride_u32;
  float eps;
  uint inline_pairs, pad3, pad4;
};

struct PackArgs {
  uint count, rank, tp_size, slot_stride;
};

inline uint pair_pack_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}

// Multi-TG BF16 pack + fanout into all other TP peers' send slots (followed by encoder barrier for L1->SLC flush).
kernel void mesh_pack_bf16(const device float4* mine [[buffer(0)]], device uint2* send_base [[buffer(1)]],
                           constant PackArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= a.count) return;
  const float4 v = mine[gid];
  const uint2 pk = uint2(pair_pack_bf16x2(v.x, v.y), pair_pack_bf16x2(v.z, v.w));
  for (uint p = 0; p < a.tp_size; ++p) {
    if (p != a.rank) send_base[ulong(p) * a.slot_stride + gid] = pk;
  }
}

// Multi-TG Q8_F32 block-64 pack + fanout: 64 floats -> 17 uint32 (1 float scale + 16 int8x4), 1 simdgroup/block.
kernel void mesh_pack_q8(const device float2* mine [[buffer(0)]], device uint* send_base [[buffer(1)]],
                         constant PackArgs& a [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 8u + sg;
  if (blk >= a.count) return;
  const float2 v = mine[ulong(blk) * 32u + lane];
  const float mx = simd_max(max(abs(v.x), abs(v.y)));
  const float sc = mx * (1.0f / 127.0f);
  const float inv = mx > 0.0f ? (127.0f / mx) : 0.0f;
  const int q0 = clamp(int(rint(v.x * inv)), -127, 127);
  const int q1 = clamp(int(rint(v.y * inv)), -127, 127);
  const uint pair16 = (uint(q0) & 0xFFu) | ((uint(q1) & 0xFFu) << 8);
  const uint peer16 = simd_shuffle_xor(pair16, 1);
  const ulong base = ulong(blk) * 17u;
  if (lane == 0u) { const uint su = as_type<uint>(sc); for (uint p = 0; p < a.tp_size; ++p) if (p != a.rank) send_base[ulong(p) * a.slot_stride + base] = su; }
  if ((lane & 1u) == 0u) {
    const uint w = pair16 | (peer16 << 16); const ulong off = base + 1u + (lane >> 1);
    for (uint p = 0; p < a.tp_size; ++p) if (p != a.rank) send_base[ulong(p) * a.slot_stride + off] = w;
  }
}

// Single-TG TP barrier: post sync[POSTED] = x and poll peer flags; skips RDMA when pad3!=0 and gate_meta[1]!=0.
kernel void mesh_wait(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flags [[buffer(1)]],
                      constant PairRows& a [[buffer(2)]], const device uint* gate_meta [[buffer(3)]],
                      uint t [[thread_index_in_threadgroup]]) {
  if (a.pad3 != 0u && gate_meta[1] != 0u) { mesh_skip_post(sync, a.x, 0u, t, a.tp_size); return; }
  mesh_post_wait(sync, flags, a.rank, a.tp_size, a.x, t);
}

// Coalesced BF16 all-reduce + residual add + RMSNorm; FUSED=true fuses mesh_wait + coherent(device) uint2 recv loads.
#if __METAL_VERSION__ >= 320
#define PAIR_COH_U2 const coherent(device) device uint2*
#else
#define PAIR_COH_U2 const device uint2*
#endif

template <uint TH, uint CACHE, bool FUSED>
inline void mesh_reduce_bf16_impl(device atomic_uint* sync, device atomic_uint* flags, const device float4* mine,
                                  const device uint2* recv_base, PAIR_COH_U2 recv_coh, device atomic_uint* recv_u,
                                  device float4* h, const device bfloat* w, device float4* y, constant PairRows& a,
                                  uint r, uint t, uint sg, uint lane, threadgroup float* part) {
  if (FUSED) {
    if (r == 0u) mesh_post_wait(sync, flags, a.rank, a.tp_size, a.x, t);
    else if (t == 0u && a.tp_size > 1u) mesh_poll_flags(sync, flags, a.rank, a.tp_size, a.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint n4 = a.width >> 2;
  const ulong r_off4 = ulong(r) * (a.stride >> 2);
  const ulong m_off4 = ulong(r) * n4;
  const ulong st_u2 = ulong(a.slot_u32) >> 1;
  float4 hv[CACHE];
  float s = 0.0f;
  uint k = 0;
  for (uint i = t; i < n4; i += TH) {
    const ulong at = m_off4 + i;
    float4 sum = float4(0.0f);
    for (uint p = 0; p < a.tp_size; ++p) {
      if (p == a.rank) {
        sum += mine[at];
      } else {
        uint2 pk;
        if (FUSED) {
#if __METAL_VERSION__ >= 320
          (void)recv_u;
          pk = recv_coh[ulong(p) * st_u2 + at];
#else
          (void)recv_coh;
          const ulong pu = ulong(p) * a.slot_u32 + (at << 1);
          pk = uint2(atomic_load_explicit(&recv_u[pu], memory_order_relaxed),
                     atomic_load_explicit(&recv_u[pu + 1ul], memory_order_relaxed));
#endif
        } else {
          pk = recv_base[ulong(p) * st_u2 + at];
        }
        sum += float4(as_type<float>(pk.x << 16), as_type<float>(pk.x & 0xFFFF0000u),
                      as_type<float>(pk.y << 16), as_type<float>(pk.y & 0xFFFF0000u));
      }
    }
    const float4 v = ((a.pad4 & 3u) != 0u) ? sum : (h[r_off4 + i] + sum);
    h[r_off4 + i] = v;
    if (k < CACHE) hv[k] = v;
    ++k;
    s += dot(v, v);
  }
  if ((a.pad4 & 3u) == 2u) return;
  s = pair_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < TH / 32u; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  const device uint2* w2 = reinterpret_cast<const device uint2*>(w);
  k = 0;
  for (uint i = t; i < n4; i += TH) {
    const uint2 wk = w2[i];
    const float4 wf = float4(as_type<float>(wk.x << 16), as_type<float>(wk.x & 0xFFFF0000u),
                             as_type<float>(wk.y << 16), as_type<float>(wk.y & 0xFFFF0000u));
    const float4 v = (k < CACHE) ? hv[k] : h[r_off4 + i];
    ++k;
    const float4 yv = wf * (v * rs);
    y[r_off4 + i] = yv;
    if ((a.pad4 & 4u) != 0u) reinterpret_cast<device half4*>(y + a.inline_pairs)[r_off4 + i] = half4(yv);
  }
}

kernel void mesh_wait_reduce_bf16(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flags [[buffer(1)]],
                                  const device float4* mine [[buffer(2)]], PAIR_COH_U2 recv_coh [[buffer(3)]],
                                  device float4* h [[buffer(4)]], const device bfloat* w [[buffer(5)]],
                                  device float4* y [[buffer(6)]], constant PairRows& a [[buffer(7)]],
                                  uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device atomic_uint* recv_u = nullptr;
#if __METAL_VERSION__ < 320
  recv_u = reinterpret_cast<device atomic_uint*>(const_cast<device uint2*>(recv_coh));
#endif
  mesh_reduce_bf16_impl<1024, 2, true>(sync, flags, mine, nullptr, recv_coh, recv_u, h, w, y, a, r, t, sg, lane, part);
}
kernel void mesh_wait_reduce_bf16_gated(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flags [[buffer(1)]],
                                        const device float4* mine [[buffer(2)]], PAIR_COH_U2 recv_coh [[buffer(3)]],
                                        device float4* h [[buffer(4)]], const device bfloat* w [[buffer(5)]],
                                        device float4* y [[buffer(6)]], constant PairRows& a [[buffer(7)]],
                                        const device uint* gate_meta [[buffer(8)]], uint r [[threadgroup_position_in_grid]],
                                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  if (gate_meta[1] != 0u) { mesh_skip_post(sync, a.x, r, t, a.tp_size); return; }
  threadgroup float part[32]; device atomic_uint* recv_u = nullptr;
#if __METAL_VERSION__ < 320
  recv_u = reinterpret_cast<device atomic_uint*>(const_cast<device uint2*>(recv_coh));
#endif
  mesh_reduce_bf16_impl<1024, 2, true>(sync, flags, mine, nullptr, recv_coh, recv_u, h, w, y, a, r, t, sg, lane, part);
}

kernel void mesh_reduce_bf16_1024(const device float4* mine [[buffer(0)]], const device uint2* recv_base [[buffer(1)]],
                                  device float4* h [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                  device float4* y [[buffer(4)]], constant PairRows& a [[buffer(5)]],
                                  const device uint* gate_meta [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  if (a.pad3 != 0u && gate_meta[1] != 0u) return;
  threadgroup float part[32];
  mesh_reduce_bf16_impl<1024, 2, false>(nullptr, nullptr, mine, recv_base, nullptr, nullptr, h, w, y, a, r, t, sg, lane, part);
}

kernel void mesh_reduce_bf16_256(const device float4* mine [[buffer(0)]], const device uint2* recv_base [[buffer(1)]],
                                 device float4* h [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                 device float4* y [[buffer(4)]], constant PairRows& a [[buffer(5)]],
                                 const device uint* gate_meta [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  if (a.pad3 != 0u && gate_meta[1] != 0u) return;
  threadgroup float part[8];
  mesh_reduce_bf16_impl<256, 6, false>(nullptr, nullptr, mine, recv_base, nullptr, nullptr, h, w, y, a, r, t, sg, lane, part);
}

// Coalesced non-atomic Q8_F32 block-64 all-reduce + residual add + RMSNorm (called after mesh_wait + memoryBarrier).
kernel void mesh_reduce_q8(const device float2* mine [[buffer(0)]], const device uint* recv_base [[buffer(1)]],
                           device float2* h [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                           device float2* y [[buffer(4)]], constant PairRows& a [[buffer(5)]],
                           uint r [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint nblk_row = a.width >> 6;
  const device float2* m2 = mine + ulong(r) * (a.width >> 1);
  device float2* x2 = h + ulong(r) * (a.stride >> 1);
  device float2* y2 = y + ulong(r) * (a.stride >> 1);
  const device uint* w1 = reinterpret_cast<const device uint*>(w);
  float2 hv[12];
  float s = 0.0f;
  uint k = 0;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const ulong base = ulong(r * nblk_row + blk) * 17u;
    float2 sum = float2(0.0f);
    for (uint p = 0; p < a.tp_size; ++p) {
      if (p == a.rank) {
        sum += m2[blk * 32u + lane];
      } else {
        const ulong pbase = ulong(p) * a.slot_u32 + base;
        const float sc = as_type<float>(recv_base[pbase]);
        const uint wq = recv_base[pbase + 1u + (lane >> 1)];
        const uint shift = (lane & 1u) * 16u;
        sum.x = fma(float(int(char((wq >> shift) & 0xFFu))), sc, sum.x);
        sum.y = fma(float(int(char((wq >> (shift + 8u)) & 0xFFu))), sc, sum.y);
      }
    }
    const uint idx2 = blk * 32u + lane;
    const float2 nv = (a.pad4 != 0u) ? sum : (x2[idx2] + sum);
    x2[idx2] = nv;
    if (k < 12u) hv[k] = nv;
    ++k;
    s = fma(nv.x, nv.x, fma(nv.y, nv.y, s));
  }
  if (a.pad4 == 2u) return;
  s = pair_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8u; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  k = 0;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const uint idx2 = blk * 32u + lane;
    const uint wk = w1[idx2];
    const float2 wf = float2(as_type<float>(wk << 16), as_type<float>(wk & 0xFFFF0000u));
    const float2 nv = (k < 12u) ? hv[k] : x2[idx2];
    ++k;
    y2[idx2] = wf * (nv * rs);
  }
}

// Phase 1 of full-mesh RS+AG: pack only peer p's row slice [p*slice, (p+1)*slice) into send_x1[p] (4x less GPU write).
kernel void mesh_rs_pack_q8(const device float2* mine [[buffer(0)]], device uint* send_base [[buffer(1)]],
                            constant PairRows& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint p = tg.y;
  if (p >= a.tp_size || p == a.rank) return;
  const uint nblk_row = a.width >> 6;
  const uint slice_rows = a.pad3;
  const uint nblk_slice = slice_rows * nblk_row;
  const uint local_blk = tg.x * 8u + sg;
  if (local_blk >= nblk_slice) return;
  const uint local_r = local_blk / nblk_row;
  const uint global_r = p * slice_rows + local_r;
  const ulong dst_base = ulong(p) * a.send_stride_u32 + ulong(local_blk) * 17u;
  if (global_r >= a.inline_pairs) {
    if (lane == 0u) send_base[dst_base] = 0u;
    return;
  }
  const uint blk_in_r = local_blk - local_r * nblk_row;
  const float2 v = mine[(ulong(global_r) * nblk_row + blk_in_r) * 32u + lane];
  const float mx = simd_max(max(abs(v.x), abs(v.y))), sc = mx * (1.0f / 127.0f), inv = mx > 0.0f ? (127.0f / mx) : 0.0f;
  const int q0 = clamp(int(rint(v.x * inv)), -127, 127), q1 = clamp(int(rint(v.y * inv)), -127, 127);
  const uint pair16 = (uint(q0) & 0xFFu) | ((uint(q1) & 0xFFu) << 8), peer16 = simd_shuffle_xor(pair16, 1);
  if (lane == 0u) send_base[dst_base] = as_type<uint>(sc);
  if ((lane & 1u) == 0u) send_base[dst_base + 1u + (lane >> 1)] = pair16 | (peer16 << 16);
}

// Phase 1->2 of full-mesh RS+AG: reduce owned slice, update h+y, and Q8-pack reduced sum into send_x2 for all peers.
kernel void mesh_rs_reduce_pack_q8(const device float2* mine [[buffer(0)]], const device uint* recv_x1 [[buffer(1)]],
                                   device uint* send_x2 [[buffer(2)]], device float2* h [[buffer(3)]],
                                   const device bfloat* w [[buffer(4)]], device float2* y [[buffer(5)]],
                                   constant PairRows& a [[buffer(6)]], uint local_r [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint global_r = a.rank * a.pad3 + local_r;
  if (global_r >= a.inline_pairs) return;
  threadgroup float part[8];
  const uint nblk_row = a.width >> 6;
  const device float2* m2 = mine + ulong(global_r) * (a.width >> 1);
  device float2* x2 = h + ulong(global_r) * (a.stride >> 1);
  device float2* y2 = y + ulong(global_r) * (a.stride >> 1);
  const device uint* w1 = reinterpret_cast<const device uint*>(w);
  float2 hv[12];
  float s = 0.0f;
  uint k = 0u;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const ulong lbase = ulong(local_r * nblk_row + blk) * 17u;
    float2 sum = float2(0.0f);
    for (uint p = 0u; p < a.tp_size; ++p) {
      if (p == a.rank) {
        sum += m2[blk * 32u + lane];
      } else {
        const ulong pbase = ulong(p) * a.slot_u32 + lbase;
        const float sc = as_type<float>(recv_x1[pbase]);
        const uint wq = recv_x1[pbase + 1u + (lane >> 1)];
        const uint shift = (lane & 1u) * 16u;
        sum.x = fma(float(int(char((wq >> shift) & 0xFFu))), sc, sum.x);
        sum.y = fma(float(int(char((wq >> (shift + 8u)) & 0xFFu))), sc, sum.y);
      }
    }
    const float mx = simd_max(max(abs(sum.x), abs(sum.y)));
    const float sc2 = mx * (1.0f / 127.0f);
    const float inv2 = mx > 0.0f ? (127.0f / mx) : 0.0f;
    const int q0 = clamp(int(rint(sum.x * inv2)), -127, 127);
    const int q1 = clamp(int(rint(sum.y * inv2)), -127, 127);
    const uint pair16 = (uint(q0) & 0xFFu) | ((uint(q1) & 0xFFu) << 8);
    const uint peer16 = simd_shuffle_xor(pair16, 1);
    if (lane == 0u) {
      const uint su = as_type<uint>(sc2);
      for (uint p = 0u; p < a.tp_size; ++p)
        if (p != a.rank) send_x2[ulong(p) * a.send_stride_u32 + lbase] = su;
    }
    if ((lane & 1u) == 0u) {
      const uint wd = pair16 | (peer16 << 16);
      const ulong off = lbase + 1u + (lane >> 1);
      for (uint p = 0u; p < a.tp_size; ++p)
        if (p != a.rank) send_x2[ulong(p) * a.send_stride_u32 + off] = wd;
    }
    const uint idx2 = blk * 32u + lane;
    const float2 nv = x2[idx2] + sum;
    x2[idx2] = nv;
    if (k < 12u) hv[k] = nv;
    ++k;
    s = fma(nv.x, nv.x, fma(nv.y, nv.y, s));
  }
  s = pair_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0u; g < 8u; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  k = 0u;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const uint idx2 = blk * 32u + lane;
    const uint wk = w1[idx2];
    const float2 wf = float2(as_type<float>(wk << 16), as_type<float>(wk & 0xFFFF0000u));
    const float2 nv = (k < 12u) ? hv[k] : x2[idx2];
    ++k;
    y2[idx2] = wf * (nv * rs);
  }
}

// Phase 2 of full-mesh RS+AG: apply peer p's already-reduced Q8 sum to h and compute RMSNorm for slice p (p != rank).
kernel void mesh_ag_apply_q8(const device uint* recv_x2 [[buffer(0)]], device float2* h [[buffer(1)]],
                             const device bfloat* w [[buffer(2)]], device float2* y [[buffer(3)]],
                             constant PairRows& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint p = tg.y, local_r = tg.x;
  if (p >= a.tp_size || p == a.rank) return;
  const uint global_r = p * a.pad3 + local_r;
  if (global_r >= a.inline_pairs) return;
  threadgroup float part[8];
  const uint nblk_row = a.width >> 6;
  device float2* x2 = h + ulong(global_r) * (a.stride >> 1);
  device float2* y2 = y + ulong(global_r) * (a.stride >> 1);
  const device uint* w1 = reinterpret_cast<const device uint*>(w);
  float2 hv[12];
  float s = 0.0f;
  uint k = 0u;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const ulong pbase = ulong(p) * a.slot_u32 + ulong(local_r * nblk_row + blk) * 17u;
    const float sc = as_type<float>(recv_x2[pbase]);
    const uint wq = recv_x2[pbase + 1u + (lane >> 1)];
    const uint shift = (lane & 1u) * 16u;
    const float2 sum = float2(float(int(char((wq >> shift) & 0xFFu))) * sc,
                              float(int(char((wq >> (shift + 8u)) & 0xFFu))) * sc);
    const uint idx2 = blk * 32u + lane;
    const float2 nv = x2[idx2] + sum;
    x2[idx2] = nv;
    if (k < 12u) hv[k] = nv;
    ++k;
    s = fma(nv.x, nv.x, fma(nv.y, nv.y, s));
  }
  s = pair_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0u; g < 8u; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  k = 0u;
  for (uint blk = sg; blk < nblk_row; blk += 8u) {
    const uint idx2 = blk * 32u + lane;
    const uint wk = w1[idx2];
    const float2 wf = float2(as_type<float>(wk << 16), as_type<float>(wk & 0xFFFF0000u));
    const float2 nv = (k < 12u) ? hv[k] : x2[idx2];
    ++k;
    y2[idx2] = wf * (nv * rs);
  }
}

// Generic FP32 / single-rank fallback: h += sum_{p=0..tp_size-1} part[p] in rank order, then y = w * (h / rms(h)).
kernel void pair_add_rms(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flags [[buffer(1)]],
                         constant PairRows& a [[buffer(2)]], const device float* mine [[buffer(3)]],
                         device atomic_uint* recv_base [[buffer(4)]], device float* h [[buffer(5)]],
                         const device bfloat* w [[buffer(6)]], device float* y [[buffer(7)]],
                         device atomic_uint* send_base [[buffer(8)]], const device uint* gate_meta [[buffer(9)]],
                         uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)send_base;
  if (a.pad3 != 0u && gate_meta[1] != 0u) { mesh_skip_post(sync, a.x, r, t, a.tp_size); return; }
  threadgroup float part[8];
  if (r == 0u) mesh_post_wait(sync, flags, a.rank, a.tp_size, a.x, t);
  else if (t == 0u && a.tp_size > 1u) mesh_poll_flags(sync, flags, a.rank, a.tp_size, a.x);
  threadgroup_barrier(mem_flags::mem_device);
  device float* x = h + ulong(r) * a.stride;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += 256u) {
    const ulong at = ulong(r) * a.width + i;
    float sum = 0.0f;
    for (uint p = 0; p < a.tp_size; ++p)
      sum += (p == a.rank) ? mine[at] : as_type<float>(atomic_load_explicit(&recv_base[ulong(p) * a.slot_u32 + at], memory_order_relaxed));
    const float v = (a.pad4 != 0u) ? sum : (x[i] + sum);
    x[i] = v; s = fma(v, v, s);
  }
  if (a.pad4 == 2u) return;
  s = pair_lanes(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8u; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += 256u) y[ulong(r) * a.stride + i] = float(w[i]) * (x[i] * rs);
}

// Row r's argmax over this rank's logits (n of them, from vocabulary id lo): (value, id), the lower id on a tie.
kernel void pair_argmax(const device float* logits [[buffer(0)]], constant uint4& dims [[buffer(1)]],
                        device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const device float* x = logits + ulong(r) * dims.x;
  float best = -INFINITY;
  uint at = 0xFFFFFFFFu;
  for (uint i = t; i < dims.z; i += 1024u) { const float v = x[i]; if (v > best) best = v, at = dims.y + i; }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  threadgroup float vb[32]; threadgroup uint va[32];
  if (lane == 0u) vb[sg] = best, va[sg] = at;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  best = vb[lane]; at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  if (lane == 0u) out[2 * r] = as_type<uint>(best), out[2 * r + 1] = at;
}

struct PickArgs {
  uint rows, x, tp_rank, tp_size;
  uint slot_u32s, pad0, pad1, pad2;
};

// Post, wait for TP peers' argmax pairs (or 9-uint top-4+sum_exp when pad0!=0), pick global max (or [id0,p0..id3,p3]).
kernel void pair_pick(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flags [[buffer(1)]],
                      constant PickArgs& a [[buffer(2)]], const device uint* mine [[buffer(3)]],
                      device atomic_uint* recv_base [[buffer(4)]], device uint* picks [[buffer(5)]],
                      const device uint* gate_meta [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  if (a.pad1 != 0u && gate_meta[1] != 0u) { mesh_skip_post(sync, a.x, 0u, t, a.tp_size); return; }
  mesh_post_wait(sync, flags, a.tp_rank, a.tp_size, a.x, t);
  if (t != 0u) return;
  if (a.pad0 != 0u) {
    for (uint r = 0u; r < a.rows; ++r) {
      float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, sum_e = 0.0f;
      uint bi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
      const uint ro = 9u * r;
      for (uint p = 0u; p < a.tp_size; ++p) {
        const ulong base = ulong(p) * a.slot_u32s + ro;
        const float v0 = (p == a.tp_rank) ? as_type<float>(mine[ro]) : as_type<float>(atomic_load_explicit(&recv_base[base], memory_order_relaxed));
        const float lp = (p == a.tp_rank) ? as_type<float>(mine[ro + 8u]) : as_type<float>(atomic_load_explicit(&recv_base[base + 8u], memory_order_relaxed));
        if (v0 > -INFINITY) {
          if (v0 > bv[0]) sum_e = (bv[0] > -INFINITY ? sum_e * exp(bv[0] - v0) : 0.0f) + lp;
          else sum_e += lp * exp(v0 - bv[0]);
          for (uint k = 0u; k < 4u; ++k) {
            const float vk = (p == a.tp_rank) ? as_type<float>(mine[ro + 2u * k]) : as_type<float>(atomic_load_explicit(&recv_base[base + 2u * k], memory_order_relaxed));
            const uint ik = (p == a.tp_rank) ? mine[ro + 2u * k + 1u] : atomic_load_explicit(&recv_base[base + 2u * k + 1u], memory_order_relaxed);
            if (vk > bv[3] || (vk == bv[3] && ik < bi[3])) {
              int pos = 3;
              while (pos > 0 && (vk > bv[pos - 1] || (vk == bv[pos - 1] && ik < bi[pos - 1]))) { bv[pos] = bv[pos - 1]; bi[pos] = bi[pos - 1]; --pos; }
              bv[pos] = vk; bi[pos] = ik;
            }
          }
        }
      }
      const float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
      for (uint k = 0u; k < 4u; ++k) {
        picks[8u * r + 2u * k] = bi[k];
        picks[8u * r + 2u * k + 1u] = as_type<uint>(clamp((bv[k] > -INFINITY) ? (exp(bv[k] - bv[0]) * inv_s) : 0.0f, 0.0f, 1.0f));
      }
    }
    return;
  }
  for (uint r = 0; r < a.rows; ++r) {
    float best = -INFINITY; uint best_id = 0xFFFFFFFFu;
    for (uint p = 0; p < a.tp_size; ++p) {
      const ulong base = ulong(p) * a.slot_u32s + 2u * r;
      const float vp = (p == a.tp_rank) ? as_type<float>(mine[2u * r]) : as_type<float>(atomic_load_explicit(&recv_base[base], memory_order_relaxed));
      const uint ap = (p == a.tp_rank) ? mine[2u * r + 1u] : atomic_load_explicit(&recv_base[base + 1u], memory_order_relaxed);
      if (vp > best || (vp == best && ap < best_id)) { best = vp; best_id = ap; }
    }
    picks[r] = best_id;
  }
}

struct PipeArgs { uint count_u32, seq, sync_idx, pad0; };
// Pipeline raw u32 send/recv (small token-ID picks return) + PP compressed activation handoff.
kernel void mesh_pipe_send(device atomic_uint* sync [[buffer(0)]], const device uint* src [[buffer(1)]],
                           device atomic_uint* dst [[buffer(2)]], constant PipeArgs& a [[buffer(3)]], uint t [[thread_index_in_threadgroup]]) {
  for (uint i = t; i < a.count_u32; i += 256u) atomic_store_explicit(&dst[i], src[i], memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0u) atomic_store_explicit(&sync[a.sync_idx], a.seq, memory_order_relaxed);
}
kernel void mesh_pipe_recv(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flag [[buffer(1)]],
                           device atomic_uint* src [[buffer(2)]], device uint* dst [[buffer(3)]],
                           constant PipeArgs& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  if (t == 0u) for (uint polls = 0u; int(atomic_load_explicit(flag, memory_order_relaxed) - a.seq) < 0;)
    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[GAVE_UP], 1u, memory_order_relaxed); break; }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < a.count_u32; i += 256u) dst[i] = atomic_load_explicit(&src[i], memory_order_relaxed);
}
kernel void mesh_pipe_post(device atomic_uint* sync [[buffer(0)]], constant PipeArgs& a [[buffer(1)]], uint t [[thread_index_in_threadgroup]]) {
  if (t == 0u) atomic_store_explicit(&sync[a.sync_idx], a.seq, memory_order_relaxed);
}
kernel void mesh_pipe_wait(device atomic_uint* sync [[buffer(0)]], device atomic_uint* flag [[buffer(1)]],
                           constant PipeArgs& a [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  for (uint polls = 0u; int(atomic_load_explicit(flag, memory_order_relaxed) - a.seq) < 0;)
    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[GAVE_UP], 1u, memory_order_relaxed); return; }
}
kernel void mesh_pipe_pack_bf16(const device float4* src [[buffer(0)]], device uint2* dst [[buffer(1)]],
                                constant uint& count_u4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= count_u4) return;
  const float4 v = src[gid]; dst[gid] = uint2(pair_pack_bf16x2(v.x, v.y), pair_pack_bf16x2(v.z, v.w));
}
kernel void mesh_pipe_unpack_bf16(const device uint2* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                                  constant uint& count_u4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= count_u4) return;
  const uint2 pk = src[gid]; dst[gid] = float4(as_type<float>(pk.x << 16), as_type<float>(pk.x & 0xFFFF0000u), as_type<float>(pk.y << 16), as_type<float>(pk.y & 0xFFFF0000u));
}
kernel void mesh_pipe_pack_q8(const device float2* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                              constant uint& count_blks [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 8u + sg;
  if (blk >= count_blks) return;
  const float2 v = src[ulong(blk) * 32u + lane];
  const float mx = simd_max(max(abs(v.x), abs(v.y))), sc = mx * (1.0f / 127.0f), inv = mx > 0.0f ? (127.0f / mx) : 0.0f;
  const int q0 = clamp(int(rint(v.x * inv)), -127, 127), q1 = clamp(int(rint(v.y * inv)), -127, 127);
  const uint pair16 = (uint(q0) & 0xFFu) | ((uint(q1) & 0xFFu) << 8), peer16 = simd_shuffle_xor(pair16, 1);
  const ulong base = ulong(blk) * 17u;
  if (lane == 0u) dst[base] = as_type<uint>(sc);
  if ((lane & 1u) == 0u) dst[base + 1u + (lane >> 1)] = pair16 | (peer16 << 16);
}
kernel void mesh_pipe_unpack_q8(const device uint* src [[buffer(0)]], device float2* dst [[buffer(1)]],
                                constant uint& count_blks [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 8u + sg;
  if (blk >= count_blks) return;
  const ulong base = ulong(blk) * 17u;
  const float sc = as_type<float>(src[base]);
  const uint wq = src[base + 1u + (lane >> 1)], shift = (lane & 1u) * 16u;
  dst[ulong(blk) * 32u + lane] = float2(float(int(char((wq >> shift) & 0xFFu))) * sc,
                                        float(int(char((wq >> (shift + 8u)) & 0xFFu))) * sc);
}
