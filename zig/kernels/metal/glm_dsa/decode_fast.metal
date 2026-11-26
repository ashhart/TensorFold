// GLM-5.3 fused decode kernels: single-stage router+route+jobs, wp BF16, attend+UV, MoE DC, BF16/Q8 pack.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DF_LAT = 512, DF_ROPE = 64, DF_NOPE = 192, DF_VDIM = 256, DF_KEY = 576, DF_SHARED = 256;
constant constexpr uint DF_MAXR = 16;

struct MlaRow { uint slot, pos, list, nkeys; };
struct MlaArgs { uint rows, heads, head0, slot_keys, proj_stride, q_stride, pad2, kvb_in_blocks; float eps, scale, pad0, pad1; ulong kvb, kvb_scale; uint tree_base, tree_p0, tree_p1, tree_p2, tree_anc[8]; };
inline uint df_tree_base(constant MlaArgs& a, uint pos) { return (a.tree_base == 0xFFFFFFFEu) ? (pos - a.tree_p1) : a.tree_base; }
inline uint df_tree_kid(uint k_ord, uint tb, uint anc) { return (k_ord < tb) ? k_ord : (tb + ((anc >> (4u * (k_ord - tb))) & 0xFu)); }
struct Fp8MoeGroup { uint expert, mem_off, count, pad; };
struct Fp8MoeArgs { uint rows, loc, hidden, topk, slots, experts, loc_blocks, hid_blocks; };
struct RouterFusedArgs { uint rows, experts, hidden, topk, loc, slots, pad0, pad1; float scale, pad2, pad3, pad4; };
struct WpBf16Args { uint rows, cols, hidden, proj_stride, wp_col, pad0, pad1, pad2; };
struct EhPackArgs { uint outs, ins, x_col, x_stride, rank, tp_size, slot_u2, pad0; };

inline half4 df_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}
inline float4 df_bf16x4(uint a, uint b) {
  return float4(as_type<float>(a << 16), as_type<float>(a & 0xFFFF0000u),
                as_type<float>(b << 16), as_type<float>(b & 0xFFFF0000u));
}
inline uint df_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}
inline float df_sum32(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Stage 1: Q=2 K-split BF16 router GEMV (128 TGs x 256t, 2 experts/TG, +34% faster on 80-core M3 Ultra, 0 bit diffs!).
template <uint R>
inline void df_router_gemv(const device float4* X4, const device uint4* W4, constant RouterFusedArgs& a,
                           device float* logits, uint g, uint sg, uint lane, threadgroup float* part) {
  const uint e0 = g * 2u;
  if (e0 >= 256u) return;
  const uint row_u4 = a.hidden >> 3, x_s4 = a.hidden >> 2, iters = a.hidden >> 11;
  float s[2][R];
  #pragma clang loop unroll(full)
  for (uint q = 0u; q < 2u; ++q) for (uint r = 0u; r < R; ++r) s[q][r] = 0.0f;
  for (uint c = 0u; c < iters; ++c) {
    const uint idx = c * 256u + sg * 32u + lane;
    if (R >= 7u) {
      const uint4 q0 = W4[ulong(e0) * row_u4 + idx], q1 = W4[ulong(e0 + 1u) * row_u4 + idx];
      const float4 w00 = df_bf16x4(q0.x, q0.y), w01 = df_bf16x4(q0.z, q0.w);
      const float4 w10 = df_bf16x4(q1.x, q1.y), w11 = df_bf16x4(q1.z, q1.w);
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < R; ++r) {
        const ulong rx = ulong(r) * x_s4 + idx * 2u;
        const float4 x0 = X4[rx], x1 = X4[rx + 1u];
        s[0][r] += dot(w00, x0) + dot(w01, x1);
        s[1][r] += dot(w10, x0) + dot(w11, x1);
      }
    } else {
      float4 x0[R], x1[R];
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < R; ++r) {
        const ulong rx = ulong(r) * x_s4 + idx * 2u;
        x0[r] = X4[rx]; x1[r] = X4[rx + 1u];
      }
      #pragma clang loop unroll(full)
      for (uint q = 0u; q < 2u; ++q) {
        const uint4 qw = W4[ulong(e0 + q) * row_u4 + idx];
        const float4 w0 = df_bf16x4(qw.x, qw.y), w1 = df_bf16x4(qw.z, qw.w);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < R; ++r) s[q][r] += dot(w0, x0[r]) + dot(w1, x1[r]);
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint q = 0u; q < 2u; ++q) for (uint r = 0u; r < R; ++r) {
    const float sv = df_sum32(s[q][r]);
    if (lane == 0u) part[(q * R + r) * 8u + sg] = sv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < 2u * R) {
    const uint q = lane / R, r = lane - q * R;
    float tot = part[lane * 8u];
    #pragma clang loop unroll(full)
    for (uint k = 1u; k < 8u; ++k) tot += part[lane * 8u + k];
    logits[r * 256u + e0 + q] = tot;
  }
}

// Stage 2 (after hardware encoder barrier): 1-TG top-8 route + optional SIMD prefix-scan MoE jobs (100% race-free!).
template <uint R, bool BUILD_JOBS>
inline void df_route_jobs(const device float* logits, const device float* bias, constant RouterFusedArgs& a,
                          device int* pick, device float* wts, device Fp8MoeGroup* groups, device uint* members,
                          device float* wts9, device uint* dispatch, const device uint* keep_meta, uint sg, uint lane, uint tid,
                          threadgroup int* pick_tg, threadgroup atomic_uint* counts, threadgroup atomic_uint* cursor,
                          threadgroup uint* g_idx, threadgroup uint* sg_cnt, threadgroup uint& total_routed) {
  if (BUILD_JOBS) {
    atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&cursor[tid], 0u, memory_order_relaxed);
  }
  const uint active_r = (BUILD_JOBS && a.pad0 != 0u) ? min(R, keep_meta[0]) : R;
  if (sg < R) {
    const uint r = sg;
    float c[8], sc[8];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint id = j * 32u + lane;
      const float x = logits[r * 256u + id];
      const float y = 1.0f / (1.0f + exp(abs(x)));
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
      const float tot = w_out[0] + w_out[1] + w_out[2] + w_out[3] + w_out[4] + w_out[5] + w_out[6] + w_out[7];
      const float inv = a.scale / tot;
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 8u; ++k) {
        const float wk = w_out[k] * inv;
        pick[r * 8u + k] = p_out[k];
        wts[r * 8u + k] = wk;
        if (BUILD_JOBS) {
          pick_tg[r * 8u + k] = (r < active_r) ? p_out[k] : 256;
          wts9[r * a.slots + k] = (r < active_r) ? wk : 0.0f;
        }
      }
      if (BUILD_JOBS) wts9[r * a.slots + 8u] = (r < active_r) ? 1.0f : 0.0f;
    }
  }
  if (!BUILD_JOBS) return;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr uint TOTAL_PICKS = R * 8u;
  if (tid < TOTAL_PICKS) {
    const uint ex = uint(pick_tg[tid]);
    if (ex < 256u) atomic_fetch_add_explicit(&counts[ex], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint cnt = atomic_load_explicit(&counts[tid], memory_order_relaxed);
  const uint nz = cnt > 0u ? 1u : 0u;
  const uint pos_in_sg = simd_prefix_exclusive_sum(nz);
  const uint sg_tot = simd_sum(nz);
  if (lane == 0u) sg_cnt[sg] = sg_tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0u;
  for (uint s_i = 0u; s_i < sg; ++s_i) base += sg_cnt[s_i];
  if (nz) {
    const uint gi = base + pos_in_sg;
    g_idx[tid] = gi;
    groups[gi] = Fp8MoeGroup{tid, gi * DF_MAXR, cnt, 0u};
  }
  const bool split_sh = (a.pad1 & 1u) != 0u;
  if (tid == 0u) {
    uint tot = 0u;
    for (uint s_i = 0u; s_i < 8u; ++s_i) tot += sg_cnt[s_i];
    if (!split_sh) groups[tot] = Fp8MoeGroup{DF_SHARED, tot * DF_MAXR, active_r, 0u};
    total_routed = tot;
    const uint n_grps = split_sh ? tot : (tot + 1u);
    dispatch[0] = ((a.pad1 & 2u) != 0u) ? ((a.loc + 3u) >> 2) : ((a.loc + 7u) >> 3); dispatch[1] = n_grps; dispatch[2] = 1u;
    dispatch[4] = (a.hidden + 15u) >> 4; dispatch[5] = n_grps; dispatch[6] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < TOTAL_PICKS) {
    const uint ex = uint(pick_tg[tid]);
    if (ex < 256u) {
      const uint r = tid >> 3, k = tid & 7u;
      const uint pos = atomic_fetch_add_explicit(&cursor[ex], 1u, memory_order_relaxed);
      members[g_idx[ex] * DF_MAXR + pos] = (r << 16) | (r * a.slots + k);
    }
  }
  if (!split_sh && tid < active_r) members[total_routed * DF_MAXR + tid] = (tid << 16) | (tid * a.slots + a.slots - 1u);
}

kernel void glm_router_gemv_1(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                              constant RouterFusedArgs& a [[buffer(2)]], device float* logits [[buffer(3)]],
                              uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2 * 8];
  df_router_gemv<1>(X4, W4, a, logits, g, sg, lane, part);
}

kernel void glm_router_route_1(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                               constant RouterFusedArgs& a [[buffer(2)]], device int* pick [[buffer(3)]],
                               device float* wts [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint total_routed, g_idx[1], sg_cnt[1];
  threadgroup int pick_tg[1];
  threadgroup atomic_uint counts[1], cursor[1];
  df_route_jobs<1, false>(logits, bias, a, pick, wts, nullptr, nullptr, nullptr, nullptr, nullptr, sg, lane, tid,
                          pick_tg, counts, cursor, g_idx, sg_cnt, total_routed);
}

#define GLM_ROUTER_JOBS(R)                                                                                             \
  kernel void glm_router_gemv_##R(                                                                                     \
      const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],                                     \
      constant RouterFusedArgs& a [[buffer(2)]], device float* logits [[buffer(3)]],                                   \
      uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                                                                       \
    threadgroup float part[2 * R * 8];                                                                                 \
    df_router_gemv<R>(X4, W4, a, logits, g, sg, lane, part);                                                           \
  }                                                                                                                    \
  kernel void glm_router_route_jobs_##R(                                                                               \
      const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],                                \
      constant RouterFusedArgs& a [[buffer(2)]], device int* pick [[buffer(3)]], device float* wts [[buffer(4)]],      \
      device Fp8MoeGroup* groups [[buffer(5)]], device uint* members [[buffer(6)]],                                    \
      device float* wts9 [[buffer(7)]], device uint* dispatch [[buffer(8)]], const device uint* keep_meta [[buffer(9)]],\
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                             \
      uint tid [[thread_index_in_threadgroup]]) {                                                                      \
    threadgroup uint total_routed, g_idx[256], sg_cnt[8];                                                              \
    threadgroup int pick_tg[R * 8];                                                                                    \
    threadgroup atomic_uint counts[256], cursor[256];                                                                  \
    df_route_jobs<R, true>(logits, bias, a, pick, wts, groups, members, wts9, dispatch, keep_meta, sg, lane, tid,      \
                           pick_tg, counts, cursor, g_idx, sg_cnt, total_routed);                                      \
  }
GLM_ROUTER_JOBS(2)
GLM_ROUTER_JOBS(3)
GLM_ROUTER_JOBS(4)
GLM_ROUTER_JOBS(5)
GLM_ROUTER_JOBS(6)
GLM_ROUTER_JOBS(7)
GLM_ROUTER_JOBS(8)

// Fast K-split BF16 GEMV for indexer.weights_proj (32 cols x 6144 hidden): grid=(32, 1), 256t -> ~2.5 us vs +16 us!
kernel void glm_wp_bf16_decode(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               constant WpBf16Args& a [[buffer(2)]], device float* proj [[buffer(3)]],
                               uint col [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  if (col >= a.cols) return;
  threadgroup float part[16][8];
  const uint rows = min(a.rows, 16u), row_u4 = a.hidden >> 3, x_s4 = a.hidden >> 2, iters = a.hidden >> 11;
  const device uint4* Wc = W4 + ulong(col) * row_u4;
  float s[16];
  for (uint r = 0u; r < 16u; ++r) s[r] = 0.0f;
  for (uint c = 0u; c < iters; ++c) {
    const uint idx = c * 256u + sg * 32u + lane;
    const uint4 qw = Wc[idx];
    const float4 w0 = df_bf16x4(qw.x, qw.y), w1 = df_bf16x4(qw.z, qw.w);
    for (uint r = 0u; r < rows; ++r) {
      const ulong rx = ulong(r) * x_s4 + idx * 2u;
      s[r] += dot(w0, X4[rx]) + dot(w1, X4[rx + 1u]);
    }
  }
  for (uint r = 0u; r < rows; ++r) {
    const float sv = df_sum32(s[r]);
    if (lane == 0u) part[r][sg] = sv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < rows) {
    float tot = part[lane][0];
    #pragma clang loop unroll(full)
    for (uint g = 1u; g < 8u; ++g) tot += part[lane][g];
    proj[ulong(lane) * a.proj_stride + a.wp_col + col] = tot;
  }
}

// Single-stage fused MLA attend + FP8 W_v proj for short ctx (<256 keys): 4s (R1-2), 2s (R3-6), 1s (R7-8).
template <uint SUB_N>
inline void df_mla_attend_uv(const device float* qlat, const device float* qrope, const device MlaRow* rows,
                             const device bfloat* cache, const device uchar* W, constant MlaArgs& a, device float* o_out,
                             uint2 tg, uint tid, uint sg, uint lane, threadgroup float* m_s, threadgroup float* l_s,
                             threadgroup float4* o_s, threadgroup half4* oh) {
  const uint r = tg.y, h = tg.x / SUB_N, sub = tg.x % SUB_N;
  if (r >= a.rows || h >= a.heads) return;
  const MlaRow row = rows[r];
  if (a.pad2 != 0u && row.pos >= 256u) return;
  const uint nkeys = row.pos + 1u;
  const device float4* ql4 = reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h) * DF_LAT + 16u * lane);
  const float4 qv0 = ql4[0], qv1 = ql4[1], qv2 = ql4[2], qv3 = ql4[3];
  const float2 qrv = *reinterpret_cast<const device float2*>(qrope + (ulong(r) * a.heads + h) * DF_ROPE + 2u * lane);
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * DF_KEY;
  const uint tb = df_tree_base(a, row.pos);
  const uint anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (uint j = sg; j < nkeys; j += 8u) {
    const device bfloat* key = base + ulong(df_tree_kid(j, tb, anc)) * DF_KEY;
    const device uint4* k4 = reinterpret_cast<const device uint4*>(key + 16u * lane);
    const uint4 w0 = k4[0], w1 = k4[1];
    const uint kr_u = *reinterpret_cast<const device uint*>(key + DF_LAT + 2u * lane);
    const float4 kv0 = df_bf16x4(w0.x, w0.y), kv1 = df_bf16x4(w0.z, w0.w);
    const float4 kv2 = df_bf16x4(w1.x, w1.y), kv3 = df_bf16x4(w1.z, w1.w);
    float p = dot(qv0, kv0) + dot(qv1, kv1) + dot(qv2, kv2) + dot(qv3, kv3);
    p = fma(qrv.x, as_type<float>(kr_u << 16), p);
    p = fma(qrv.y, as_type<float>(kr_u & 0xFFFF0000u), p);
    const float sc = df_sum32(p) * a.scale;
    const float mn = max(m, sc);
    const float corr = (m == -INFINITY) ? 0.0f : exp(m - mn), ev = exp(sc - mn);
    l = fma(l, corr, ev);
    o0 = fma(float4(ev), kv0, o0 * corr); o1 = fma(float4(ev), kv1, o1 * corr);
    o2 = fma(float4(ev), kv2, o2 * corr); o3 = fma(float4(ev), kv3, o3 * corr);
    m = mn;
  }
  if (lane == 0u) { m_s[sg] = m; l_s[sg] = l; }
  o_s[sg * 128u + lane * 4u + 0u] = o0; o_s[sg * 128u + lane * 4u + 1u] = o1;
  o_s[sg * 128u + lane * 4u + 2u] = o2; o_s[sg * 128u + lane * 4u + 3u] = o3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128u) {
    float M = m_s[0];
    #pragma clang loop unroll(full)
    for (uint s = 1u; s < 8u; ++s) M = max(M, m_s[s]);
    float L = 0.0f;
    float4 acc = 0.0f;
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 8u; ++s) {
      const float f = (l_s[s] > 0.0f) ? exp(m_s[s] - M) : 0.0f;
      L += l_s[s] * f;
      acc = fma(o_s[s * 128u + tid], float4(f), acc);
    }
    oh[tid] = half4(L > 0.0f ? (acc / L) : float4(0.0f));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const half4 x0 = oh[lane * 4u], x1 = oh[lane * 4u + 1u], x2 = oh[lane * 4u + 2u], x3 = oh[lane * 4u + 3u];
  constexpr uint PER_SUB = DF_VDIM / SUB_N, PER_SG = PER_SUB / 8u;
  const uint v_base = sub * PER_SUB + sg * PER_SG, sub_sb = lane >> 3;
  const device float* S = reinterpret_cast<const device float*>(W + a.kvb_scale);
  device float* dst = o_out + (ulong(r) * a.heads + h) * DF_VDIM + v_base;
  #pragma clang loop unroll(full)
  for (uint k = 0u; k < PER_SG; k += 2u) {
    const uint og0 = 448u * (a.head0 + h) + DF_NOPE + v_base + k;
    const ulong row0 = a.kvb + ulong(og0) * DF_LAT, row1 = row0 + DF_LAT;
    const float s0 = S[ulong(og0 >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f;
    const float s1 = S[ulong((og0 + 1u) >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + lane * 16u);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row1 + lane * 16u);
    const float vv0 = df_sum32((float(dot(df_fp8x4_h(q0.x), x0)) + float(dot(df_fp8x4_h(q0.y), x1)) +
                                float(dot(df_fp8x4_h(q0.z), x2)) + float(dot(df_fp8x4_h(q0.w), x3))) * s0);
    const float vv1 = df_sum32((float(dot(df_fp8x4_h(q1.x), x0)) + float(dot(df_fp8x4_h(q1.y), x1)) +
                                float(dot(df_fp8x4_h(q1.z), x2)) + float(dot(df_fp8x4_h(q1.w), x3))) * s1);
    if (lane == 0u) {
      dst[k] = vv0; dst[k + 1u] = vv1;
      if (a.pad1 != 0.0f) reinterpret_cast<device half2*>(o_out + ulong(DF_MAXR * a.heads) * DF_VDIM)[(((ulong(r) * a.heads + h) * DF_VDIM + v_base + k) >> 1)] = half2(vv0, vv1);
    }
  }
}
#define DF_ATTEND_UV(N) \
  kernel void glm_mla_attend_uv_##N##s(const device float* ql [[buffer(0)]], const device float* qr [[buffer(1)]], const device MlaRow* ro [[buffer(2)]], const device bfloat* ca [[buffer(3)]], const device uchar* W [[buffer(4)]], constant MlaArgs& a [[buffer(5)]], device float* oo [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { threadgroup float m_s[8], l_s[8]; threadgroup float4 o_s[8 * 128]; threadgroup half4 oh[128]; df_mla_attend_uv<N>(ql, qr, ro, ca, W, a, oo, tg, tid, sg, lane, m_s, l_s, o_s, oh); }
DF_ATTEND_UV(4) DF_ATTEND_UV(2) DF_ATTEND_UV(1)

inline float df_q8_dot16(uint4 q, half4 x0, half4 x1, half4 x2, half4 x3) {
  return float(dot(half4(as_type<char4>(q.x)), x0) + dot(half4(as_type<char4>(q.y)), x1) +
               dot(half4(as_type<char4>(q.z)), x2) + dot(half4(as_type<char4>(q.w)), x3));
}

// Draft-only R=2..8 INT8 g64 eh_proj (2x smaller than BF16!) + keep_meta row gating + coalesced uint2 peer pack.
kernel void glm_eh_q8_pack_n(const device float* X [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                             const device half* S [[buffer(2)]], constant EhPackArgs& a [[buffer(3)]],
                             device float* Y [[buffer(4)]], device uint2* send_base [[buffer(5)]],
                             const device uint* keep_meta [[buffer(6)]], uint g [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float y_stg[8 * 8];
  const uint o0 = g * 8u + sg * 2u, rows = min(a.pad0 & 0xFu, 8u);
  if (o0 >= a.outs || rows == 0u) return;
  const uint active_r = ((a.pad0 & 16u) != 0u) ? min(rows, max(1u, keep_meta[0])) : rows;
  const uint row_u4 = a.ins >> 4, c512 = a.ins >> 9, g_cols = a.ins >> 6, sub_g = lane >> 2, xs4 = a.x_stride >> 2;
  const device float4* X4 = reinterpret_cast<const device float4*>(X + a.x_col);
  const device uint4* W0 = W4 + ulong(o0) * row_u4, *W1 = W0 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  float s0[8], s1[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) s0[r] = 0.0f, s1[r] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint gi = c * 8u + sub_g;
    const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane];
    const float sc0 = float(S0[gi]), sc1 = float(S1[gi]);
    for (uint r = 0u; r < active_r; ++r) {
      const ulong xo = ulong(r) * xs4 + c * 128u + lane * 4u;
      const half4 x0 = half4(X4[xo] * 0.125f), x1 = half4(X4[xo + 1u] * 0.125f);
      const half4 x2 = half4(X4[xo + 2u] * 0.125f), x3 = half4(X4[xo + 3u] * 0.125f);
      s0[r] = fma(df_q8_dot16(q0, x0, x1, x2, x3), sc0, s0[r]);
      s1[r] = fma(df_q8_dot16(q1, x0, x1, x2, x3), sc1, s1[r]);
    }
  }
  for (uint r = 0u; r < active_r; ++r) {
    const float v0 = df_sum32(s0[r]) * 8.0f, v1 = df_sum32(s1[r]) * 8.0f;
    if (lane == 0u) {
      *reinterpret_cast<device float2*>(Y + ulong(r) * a.outs + o0) = float2(v0, v1);
      if (a.tp_size > 1u) { y_stg[r * 8u + sg * 2u] = v0; y_stg[r * 8u + sg * 2u + 1u] = v1; }
    }
  }
  if (a.tp_size > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < active_r * 2u) {
      const uint r = tid >> 1, q = tid & 1u, sb = r * 8u + q * 4u;
      const uint2 p2 = uint2(df_bf16x2(y_stg[sb], y_stg[sb + 1u]), df_bf16x2(y_stg[sb + 2u], y_stg[sb + 3u]));
      const ulong idx2 = (ulong(r) * a.outs + g * 8u + q * 4u) >> 2;
      for (uint p = 0u; p < a.tp_size; ++p) if (p != a.rank) send_base[ulong(p) * a.slot_u2 + idx2] = p2;
    }
  }
}

// Draft-only R=1 Q=4 128t INT8 g64 eh_proj + coalesced uint2 peer pack (2x faster than BF16!).
kernel void glm_eh_q8_pack_1(const device float* X [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                             const device half* S [[buffer(2)]], constant EhPackArgs& a [[buffer(3)]],
                             device float* Y [[buffer(4)]], device uint2* send_base [[buffer(5)]],
                             uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint o0 = g * 16u + sg * 4u;
  if (o0 >= a.outs) return;
  const uint row_u4 = a.ins >> 4, c512 = a.ins >> 9, g_cols = a.ins >> 6, sub_g = lane >> 2;
  const device float4* X4 = reinterpret_cast<const device float4*>(X + a.x_col);
  const device uint4* W0 = W4 + ulong(o0) * row_u4;
  const device uint4* W1 = W0 + row_u4, *W2 = W1 + row_u4, *W3 = W2 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols;
  const device half* S1 = S0 + g_cols, *S2 = S1 + g_cols, *S3 = S2 + g_cols;
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint xo = c * 128u + lane * 4u, gi = c * 8u + sub_g;
    const half4 x0 = half4(X4[xo] * 0.125f), x1 = half4(X4[xo + 1u] * 0.125f);
    const half4 x2 = half4(X4[xo + 2u] * 0.125f), x3 = half4(X4[xo + 3u] * 0.125f);
    s0 = fma(df_q8_dot16(W0[c * 32u + lane], x0, x1, x2, x3), float(S0[gi]), s0);
    s1 = fma(df_q8_dot16(W1[c * 32u + lane], x0, x1, x2, x3), float(S1[gi]), s1);
    s2 = fma(df_q8_dot16(W2[c * 32u + lane], x0, x1, x2, x3), float(S2[gi]), s2);
    s3 = fma(df_q8_dot16(W3[c * 32u + lane], x0, x1, x2, x3), float(S3[gi]), s3);
  }
  const float v0 = df_sum32(s0) * 8.0f, v1 = df_sum32(s1) * 8.0f, v2 = df_sum32(s2) * 8.0f, v3 = df_sum32(s3) * 8.0f;
  if (lane != 0u) return;
  *reinterpret_cast<device float4*>(Y + o0) = float4(v0, v1, v2, v3);
  if (a.tp_size > 1u) {
    const uint2 pk = uint2(df_bf16x2(v0, v1), df_bf16x2(v2, v3));
    for (uint p = 0u; p < a.tp_size; ++p) if (p != a.rank) send_base[ulong(p) * a.slot_u2 + ulong(o0 >> 2)] = pk;
  }
}

struct MtpCatArgs {
  uint width, pad0, pad1, pad2;
  float eps, pad3, pad4, pad5;
};

// Fused MTP enorm(embed) | hnorm(prev) in one 1024t launch (replaces 2 glm_add_rms launches in draft!).
kernel void glm_mtp_cat_rms(const device float4* emb [[buffer(0)]], const device float4* prev [[buffer(1)]],
                            const device uint2* w_e [[buffer(2)]], const device uint2* w_h [[buffer(3)]],
                            constant MtpCatArgs& a [[buffer(4)]], device float4* cat [[buffer(5)]],
                            uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float pe[32], ph[32];
  const uint n4 = a.width >> 2;
  const device float4* e4 = emb + ulong(r) * n4;
  const device float4* h4 = prev + ulong(r) * n4;
  device float4* c4 = cat + ulong(r) * (2ul * n4);
  float4 ev[2], hv[2];
  float se = 0.0f, sh = 0.0f;
  uint k = 0u;
  for (uint i = t; i < n4; i += 1024u) {
    const float4 ve = e4[i], vh = h4[i];
    if (k < 2u) { ev[k] = ve; hv[k] = vh; }
    ++k;
    se += dot(ve, ve);
    sh += dot(vh, vh);
  }
  se = df_sum32(se);
  sh = df_sum32(sh);
  if (lane == 0u) { pe[sg] = se; ph[sg] = sh; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ve_tot = 0.0f, vh_tot = 0.0f;
  for (uint g = 0u; g < 32u; ++g) { ve_tot += pe[g]; vh_tot += ph[g]; }
  const float rse = precise::rsqrt(ve_tot / float(a.width) + a.eps);
  const float rsh = precise::rsqrt(vh_tot / float(a.width) + a.eps);
  k = 0u;
  for (uint i = t; i < n4; i += 1024u) {
    const uint2 we = w_e[i], wh = w_h[i];
    const float4 ve = (k < 2u) ? ev[k] : e4[i];
    const float4 vh = (k < 2u) ? hv[k] : h4[i];
    ++k;
    c4[i] = df_bf16x4(we.x, we.y) * (ve * rse);
    c4[n4 + i] = df_bf16x4(wh.x, wh.y) * (vh * rsh);
  }
}

struct MtpAcceptArgs {
  uint rows, slot, base_pos, depth;
  uint i_topk, pad0, pad1, pad2;
};

// Single-TG GPU greedy accept for fused window+draft: sets kept, pass[1].ids, pass[2].idx, and d_rows[0..depth-2].
kernel void glm_mtp_accept(const device uint* win_ids [[buffer(0)]], const device uint* win_picks [[buffer(1)]], constant MtpAcceptArgs& a [[buffer(2)]], device uint* accept_meta [[buffer(3)]], device uint* d1_ids [[buffer(4)]], device uint* d2_idx [[buffer(5)]], device MlaRow* d_rows [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  uint kept = 1u;
  for (uint j = 0u; j + 1u < a.rows; ++j) { if (win_picks[j] == win_ids[j + 1u]) ++kept; else break; }
  accept_meta[0] = kept;
  for (uint j = 0u; j < a.rows; ++j) d1_ids[j] = win_picks[j];
  d2_idx[0] = kept - 1u;
  const uint p0 = a.base_pos + kept - 1u;
  if (a.pad0 != 0u) {
    d_rows[0] = MlaRow{a.slot, p0 + 1u, 0u, 0u}; d_rows[1] = MlaRow{a.slot, p0 + 1u, a.i_topk, 0u};
    for (uint k = 3u; k <= a.depth && k < 8u; ++k) d_rows[k - 1u] = MlaRow{a.slot, p0 + k - 1u, 0u, 0u};
  } else for (uint k = 2u; k <= a.depth; ++k) d_rows[k - 2u] = MlaRow{a.slot, p0 + k - 1u, 0u, 0u};
}
struct MtpGateArgs { uint step, n_grids; float thresh, pad0; uint4 grids[15]; };

// Short-ctx 8-sg attend -> half4 olat_h (1-barrier 16KB o_s, 0-ULP vs attend_uv_4s!).
kernel void glm_mla_attend_short_8sg(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                     const device MlaRow* rows [[buffer(2)]], const device bfloat* cache [[buffer(3)]],
                                     constant MlaArgs& a [[buffer(4)]], device half4* olat_h [[buffer(5)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float m_s[8], l_s[8];
  threadgroup float4 o_s[8 * 128];
  const uint r = tg.y, h = tg.x;
  if (r >= a.rows || h >= a.heads) return;
  const MlaRow row = rows[r];
  if (a.pad2 != 0u && row.pos >= 256u) return;
  const uint nkeys = row.pos + 1u;
  const device float4* ql4 = reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h) * DF_LAT + 16u * lane);
  const float4 qv0 = ql4[0], qv1 = ql4[1], qv2 = ql4[2], qv3 = ql4[3];
  const float2 qrv = *reinterpret_cast<const device float2*>(qrope + (ulong(r) * a.heads + h) * DF_ROPE + 2u * lane);
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * DF_KEY;
  const uint tb = df_tree_base(a, row.pos), anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (uint j = sg; j < nkeys; j += 8u) {
    const device bfloat* key = base + ulong(df_tree_kid(j, tb, anc)) * DF_KEY;
    const device uint4* k4 = reinterpret_cast<const device uint4*>(key + 16u * lane);
    const uint4 w0 = k4[0], w1 = k4[1];
    const uint kr_u = *reinterpret_cast<const device uint*>(key + DF_LAT + 2u * lane);
    const float4 kv0 = df_bf16x4(w0.x, w0.y), kv1 = df_bf16x4(w0.z, w0.w), kv2 = df_bf16x4(w1.x, w1.y), kv3 = df_bf16x4(w1.z, w1.w);
    float p = fma(qrv.y, as_type<float>(kr_u & 0xFFFF0000u), fma(qrv.x, as_type<float>(kr_u << 16),
                  dot(qv0, kv0) + dot(qv1, kv1) + dot(qv2, kv2) + dot(qv3, kv3)));
    const float sc = df_sum32(p) * a.scale, mn = max(m, sc);
    const float corr = (m == -INFINITY) ? 0.0f : exp(m - mn), ev = exp(sc - mn);
    l = fma(l, corr, ev);
    o0 = fma(float4(ev), kv0, o0 * corr); o1 = fma(float4(ev), kv1, o1 * corr);
    o2 = fma(float4(ev), kv2, o2 * corr); o3 = fma(float4(ev), kv3, o3 * corr);
    m = mn;
  }
  if (lane == 0u) { m_s[sg] = m; l_s[sg] = l; }
  o_s[sg * 128u + lane * 4u + 0u] = o0; o_s[sg * 128u + lane * 4u + 1u] = o1;
  o_s[sg * 128u + lane * 4u + 2u] = o2; o_s[sg * 128u + lane * 4u + 3u] = o3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128u) {
    float M = m_s[0], L = 0.0f;
    float4 acc = 0.0f;
    #pragma clang loop unroll(full)
    for (uint s = 1u; s < 8u; ++s) M = max(M, m_s[s]);
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 8u; ++s) {
      const float f = (l_s[s] > 0.0f) ? exp(m_s[s] - M) : 0.0f;
      L += l_s[s] * f;
      acc = fma(o_s[s * 128u + tid], float4(f), acc);
    }
    olat_h[(ulong(r) * a.heads + h) * 128u + tid] = half4(L > 0.0f ? (acc / L) : float4(0.0f));
  }
}

// GPU early-stop gate: main chain (1..6, or 11..16 preserve-picks), Branch-1 (201..203).
kernel void glm_mtp_gate(device uint* picks [[buffer(0)]], device uint* meta [[buffer(1)]],
                         device uint4* disp [[buffer(2)]], constant MtpGateArgs& a [[buffer(3)]],
                         uint t [[thread_index_in_threadgroup]]) {
  if (t != 0u) return;
  uint stopped = 0u;
  if (a.step == 200u) {
    meta[0] = (clamp(as_type<float>(picks[3]), 0.0f, 1.0f) >= 0.08f) ? 2u : 1u;
    return;
  }
  if (a.step == 201u) {
    const float p1 = clamp(as_type<float>(picks[3]), 0.0f, 1.0f);
    const float q1 = (p1 > 1e-6f) ? pow(p1, 0.80f) : 0.0f;
    stopped = (q1 < a.thresh) ? 1u : 0u;
    meta[0] = as_type<uint>(q1); meta[1] = stopped;
  } else if (a.step == 203u) {
    stopped = meta[1];
    if (stopped != 0u) { picks[1] = 0u; picks[3] = 0u; picks[5] = 0u; picks[7] = 0u; }
  } else {
    const bool keep_p = (a.step >= 10u && a.step < 200u);
    const uint st = keep_p ? (a.step - 10u) : ((a.step == 202u) ? 2u : a.step);
    stopped = (st == 1u) ? 0u : meta[1];
    if (stopped != 0u) { if (!keep_p) { picks[1] = 0u; picks[3] = 0u; picks[5] = 0u; picks[7] = 0u; } }
    else {
      const float p = clamp(as_type<float>(picks[1]), 0.0f, 1.0f);
      const float pc = (p > 1e-6f) ? ((st == 1u) ? pow(p, 1.65f) : (0.94f * pow(p, 1.85f))) : 0.0f;
      const float cum = (st == 1u) ? pc : (as_type<float>(meta[0]) * pc);
      stopped = (cum < a.thresh) ? 1u : 0u;
      meta[0] = as_type<uint>(cum); meta[1] = stopped;
    }
  }
  for (uint i = 0u; i < a.n_grids; ++i) disp[i] = (stopped != 0u) ? uint4(0u) : a.grids[i];
}
