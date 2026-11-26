// GLM-5.3's MLA for lane rows: latent and rope keys in a bf16 cache, queries absorbed into latent space.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif
using namespace metal;

constant constexpr uint LAT = 512, ROPE = 64, NOPE = 192, KEY = LAT + ROPE, QK = NOPE + ROPE;

struct MlaRow { uint slot, pos, list, nkeys; };
struct MlaArgs {
  uint rows, heads, head0, slot_keys, proj_stride, q_stride, pad2, kvb_in_blocks;
  float eps, scale, pad0, pad1;
  ulong kvb, kvb_scale;
  uint tree_base, tree_p0, tree_p1, tree_p2, tree_anc[8];
};
inline uint glm_tree_base(constant MlaArgs& a, uint pos) { return (a.tree_base == 0xFFFFFFFEu) ? (pos - a.tree_p1) : a.tree_base; }
inline uint glm_tree_kid(uint k_ord, uint tb, uint anc) { return (k_ord < tb) ? k_ord : (tb + ((anc >> (4u * (k_ord - tb))) & 0xFu)); }
inline float glm_lanes(float v) { for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d); return v; }

// Pair i of a 64-wide interleaved rope at position pos: (x[2i], x[2i+1]) -> out[i], out[32 + i].
inline float2 glm_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

inline float4 glm_bf16x4(uint a, uint b) {
  return float4(as_type<float>(a << 16), as_type<float>(a & 0xFFFF0000u),
                as_type<float>(b << 16), as_type<float>(b & 0xFFFF0000u));
}
inline uint glm_pack_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}

// Vectorized 512t prep: 1 float4/thread for q_a (2048), 128t for kv_a (512), 32t for rope -> 2.5x faster!
kernel void glm_mla_prep(const device float* proj [[buffer(0)]], const device bfloat* w_qa [[buffer(1)]],
                         const device bfloat* w_kva [[buffer(2)]], const device float* inv [[buffer(3)]],
                         const device MlaRow* rows [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                         device float* q_resid [[buffer(6)]], device bfloat* cache [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2][16];
  const device float* x = proj + ulong(row) * a.proj_stride;
  const device float4* x4 = reinterpret_cast<const device float4*>(x);
  const float4 qv = x4[t];
  const float sq = glm_lanes(dot(qv, qv));
  const float4 kv = (t < 128u) ? x4[512u + t] : float4(0.0f);
  const float sk = (t < 128u) ? glm_lanes(dot(kv, kv)) : 0.0f;
  if (lane == 0u) { part[0][sg] = sq; part[1][sg] = sk; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vq = 0.0f, vk = 0.0f;
  #pragma clang loop unroll(full)
  for (uint g = 0u; g < 16u; ++g) vq += part[0][g];
  #pragma clang loop unroll(full)
  for (uint g = 0u; g < 4u; ++g) vk += part[1][g];
  const float rq = precise::rsqrt(vq / 2048.0f + a.eps), rk = precise::rsqrt(vk / float(LAT) + a.eps);
  const uint2 wq = reinterpret_cast<const device uint2*>(w_qa)[t];
  const float4 qo = glm_bf16x4(wq.x, wq.y) * (qv * rq);
  reinterpret_cast<device float4*>(q_resid + ulong(row) * a.q_stride)[t] = qo;
  if (a.pad2 != 0u) reinterpret_cast<device half4*>(q_resid + a.pad2)[ulong(row) * (a.q_stride >> 2) + t] = half4(qo);
  const MlaRow r = rows[row];
  const uint tb = glm_tree_base(a, r.pos);
  const uint kslot = (tb != 0xFFFFFFFFu) ? (tb + a.tree_p0 + row) : r.pos;
  device bfloat* key = cache + (ulong(r.slot) * a.slot_keys + kslot) * KEY;
  if (t < 128u) {
    const uint2 wk = reinterpret_cast<const device uint2*>(w_kva)[t];
    const float4 ko = glm_bf16x4(wk.x, wk.y) * (kv * rk);
    reinterpret_cast<device uint2*>(key)[t] = uint2(glm_pack_bf16x2(ko.x, ko.y), glm_pack_bf16x2(ko.z, ko.w));
  } else if (sg == 4u) {
    const float2 o = glm_rope(x[2048u + LAT + 2u * lane], x[2048u + LAT + 2u * lane + 1u], r.pos, inv, lane);
    key[LAT + lane] = bfloat(o.x);
    key[LAT + ROPE / 2u + lane] = bfloat(o.y);
  }
}

// Vectorized 512t fused prep + indexKey for full-indexer layers (sg=5 handles indexKey in parallel!).
kernel void glm_mla_prep_full(const device float* proj [[buffer(0)]], const device bfloat* w_qa [[buffer(1)]],
                              const device bfloat* w_kva [[buffer(2)]], const device float* inv [[buffer(3)]],
                              const device MlaRow* rows [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                              device float* q_resid [[buffer(6)]], device bfloat* cache [[buffer(7)]],
                              const device bfloat* w_ik [[buffer(8)]], const device bfloat* b_ik [[buffer(9)]],
                              device bfloat* idx_cache [[buffer(10)]],
                              uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2][16], ik_y[128];
  const device float* x = proj + ulong(row) * a.proj_stride;
  const device float4* x4 = reinterpret_cast<const device float4*>(x);
  const float4 qv = x4[t];
  const float sq = glm_lanes(dot(qv, qv));
  const float4 kv = (t < 128u) ? x4[512u + t] : float4(0.0f);
  const float sk = (t < 128u) ? glm_lanes(dot(kv, kv)) : 0.0f;
  if (lane == 0u) { part[0][sg] = sq; part[1][sg] = sk; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vq = 0.0f, vk = 0.0f;
  #pragma clang loop unroll(full)
  for (uint g = 0u; g < 16u; ++g) vq += part[0][g];
  #pragma clang loop unroll(full)
  for (uint g = 0u; g < 4u; ++g) vk += part[1][g];
  const float rq = precise::rsqrt(vq / 2048.0f + a.eps), rk = precise::rsqrt(vk / float(LAT) + a.eps);
  const uint2 wq = reinterpret_cast<const device uint2*>(w_qa)[t];
  const float4 qo = glm_bf16x4(wq.x, wq.y) * (qv * rq);
  reinterpret_cast<device float4*>(q_resid + ulong(row) * a.q_stride)[t] = qo;
  if (a.pad2 != 0u) reinterpret_cast<device half4*>(q_resid + a.pad2)[ulong(row) * (a.q_stride >> 2) + t] = half4(qo);
  const MlaRow r = rows[row];
  const uint tb = glm_tree_base(a, r.pos);
  const uint kslot = (tb != 0xFFFFFFFFu) ? (tb + a.tree_p0 + row) : r.pos;
  device bfloat* key = cache + (ulong(r.slot) * a.slot_keys + kslot) * KEY;
  if (t < 128u) {
    const uint2 wk = reinterpret_cast<const device uint2*>(w_kva)[t];
    const float4 ko = glm_bf16x4(wk.x, wk.y) * (kv * rk);
    reinterpret_cast<device uint2*>(key)[t] = uint2(glm_pack_bf16x2(ko.x, ko.y), glm_pack_bf16x2(ko.z, ko.w));
  } else if (sg == 4u) {
    const float2 o = glm_rope(x[2048u + LAT + 2u * lane], x[2048u + LAT + 2u * lane + 1u], r.pos, inv, lane);
    key[LAT + lane] = bfloat(o.x);
    key[LAT + ROPE / 2u + lane] = bfloat(o.y);
  } else if (sg == 5u) {
    const device float* ix = x + 2048u + KEY;
    float v[4], s = 0.0f;
    for (uint q = 0u; q < 4u; ++q) v[q] = ix[lane + 32u * q], s += v[q];
    const float mean = glm_lanes(s) / 128.0f;
    float d = 0.0f;
    for (uint q = 0u; q < 4u; ++q) d = fma(v[q] - mean, v[q] - mean, d);
    const float rstd = precise::rsqrt(glm_lanes(d) / 128.0f + a.eps);
    for (uint q = 0u; q < 4u; ++q) {
      const uint i = lane + 32u * q;
      ik_y[i] = fma((v[q] - mean) * rstd, float(w_ik[i]), float(b_ik[i]));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device bfloat* ikey = idx_cache + (ulong(r.slot) * a.slot_keys + kslot) * 128u;
    const float2 o = glm_rope(ik_y[2u * lane], ik_y[2u * lane + 1u], r.pos, inv, lane);
    ikey[lane] = bfloat(o.x);
    ikey[32u + lane] = bfloat(o.y);
    ikey[64u + lane] = bfloat(ik_y[64u + lane]);
    ikey[96u + lane] = bfloat(ik_y[96u + lane]);
  }
}

inline half4 glm_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}

// Fast 16-way d-split uint4/uint absorb (grid=(8, heads), 256t): 16 c-lanes x 16 d-groups = 12 uint loads/thread!
template <uint R>
inline void glm_mla_absorb_impl(const device float* q, const device uchar* W, const device MlaRow* rows,
                                constant MlaArgs& a, device float* qlat, device float* qrope, const device float* inv,
                                uint h, uint cb, uint tid, threadgroup float* qs, threadgroup float4* red) {
  const uint n = min(a.rows, R), hg = a.head0 + h;
  const uint c_lane = tid & 15u, d_grp = tid >> 4;
  const uint c4 = cb * 16u + c_lane, sb_c = cb >> 1;
  const device uint* fp8_u = reinterpret_cast<const device uint*>(W + a.kvb);
  const device float* scale = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const uint o_base = 448u * hg, sb0 = o_base >> 7;
  const uint split_d = min(NOPE, 128u - (o_base & 127u));
  const float sc0 = scale[sb0 * a.kvb_in_blocks + sb_c] * 256.0f;
  const float sc1 = scale[(sb0 + 1u) * a.kvb_in_blocks + sb_c] * 256.0f;
  const uint d0 = d_grp * 12u, d1 = d0 + 12u;
  constexpr uint CH = R <= 4u ? R : 4u;
  for (uint r0 = 0u; r0 < n; r0 += CH) {
    const uint rn = min(CH, n - r0);
    for (uint i = tid; i < rn * NOPE; i += 256u) {
      qs[i] = q[ulong(r0 + i / NOPE) * a.q_stride + h * QK + i % NOPE];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc0[CH], acc1[CH];
    for (uint r = 0; r < CH; ++r) acc0[r] = float4(0.0f), acc1[r] = float4(0.0f);
    for (uint d = d0; d < d1; ++d) {
      const float4 wf = float4(glm_fp8x4_h(fp8_u[ulong(o_base + d) * 128u + c4]));
      if (d < split_d) {
        for (uint r = 0; r < CH; ++r) acc0[r] = fma(wf, float4(qs[r * NOPE + d]), acc0[r]);
      } else {
        for (uint r = 0; r < CH; ++r) acc1[r] = fma(wf, float4(qs[r * NOPE + d]), acc1[r]);
      }
    }
    for (uint r = 0; r < rn; ++r) {
      red[d_grp * 16u + c_lane] = fma(acc0[r], float4(sc0), acc1[r] * sc1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (d_grp == 0u) {
        float4 tot = red[c_lane];
        for (uint g = 1u; g < 16u; ++g) tot += red[g * 16u + c_lane];
        *reinterpret_cast<device float4*>(qlat + (ulong(r0 + r) * a.heads + h) * LAT + c4 * 4u) = tot;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (cb != 0u || tid >= ROPE / 2u) return;
  for (uint r = 0; r < n; ++r) {
    const device float* x = q + ulong(r) * a.q_stride + h * QK + NOPE;
    const float2 o = glm_rope(x[2u * tid], x[2u * tid + 1u], rows[r].pos, inv, tid);
    qrope[(ulong(r) * a.heads + h) * ROPE + tid] = o.x;
    qrope[(ulong(r) * a.heads + h) * ROPE + ROPE / 2u + tid] = o.y;
  }
}

#define GLM_MLA_ABSORB(R)                                                                                             \
  kernel void glm_mla_absorb_##R(const device float* q [[buffer(0)]], const device uchar* W [[buffer(1)]],          \
                                 const device MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],        \
                                 device float* qlat [[buffer(4)]], device float* qrope [[buffer(5)]],               \
                                 const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], \
                                 uint t [[thread_index_in_threadgroup]]) {                                          \
    threadgroup float qs[(R <= 4 ? R : 4) * NOPE];                                                                  \
    threadgroup float4 red[256];                                                                                    \
    glm_mla_absorb_impl<R>(q, W, rows, a, qlat, qrope, inv, tg.y, tg.x, t, qs, red);                                \
  }
GLM_MLA_ABSORB(1)
GLM_MLA_ABSORB(2)
GLM_MLA_ABSORB(4)
GLM_MLA_ABSORB(8)
GLM_MLA_ABSORB(16)

// Keys a split takes: dynamic split count (a.pad2 = 0 legacy 128-aligned splits; else 16-aligned a.pad2 splits).
constant constexpr uint SPLITS = 16;
inline uint2 glm_split(uint n, uint s) {
  const uint span = 128 * ((n + 128 * SPLITS - 1) / (128 * SPLITS));
  return uint2(min(s * span, n), min((s + 1) * span, n));
}
inline uint2 glm_split_n(uint n, uint s, uint ns) {
  if (ns == 0u) return glm_split(n, s);
  if (ns == 1u) return uint2(0u, n);
  const uint chunk = (n + ns - 1u) / ns;
  const uint span = max(16u, (chunk + 15u) & ~15u);
  return uint2(min(s * span, n), min((s + 1u) * span, n));
}

// Head-shared decode MLA: 8 heads/TG cooperatively load 4 keys/batch; ns==1 writes olat directly (0 merge).
kernel void glm_mla_attend_decode(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                  const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                  const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                  device float* dst [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 k_buf_u4[4u * 72u];
  const uint r = tg.z, s = tg.y, h = tg.x * 8u + sg;
  const MlaRow row = rows[r];
  const uint ns = a.pad2;
  const uint stride = ns == 0u ? SPLITS : ns;
  const uint nkeys = row.pos < 2048u ? (row.pos + 1u) : row.nkeys;
  const uint2 span = glm_split_n(nkeys, s, ns);
  if (span.x >= span.y) {
    if (ns > 1u && h < a.heads && lane == 0u) {
      device float* out = dst + ((ulong(r) * a.heads + h) * stride + s) * (LAT + 2u);
      out[0] = -INFINITY;
      out[1] = 0.0f;
    }
    return;
  }
  const bool active = h < a.heads;
  float4 qv0 = 0.0f, qv1 = 0.0f, qv2 = 0.0f, qv3 = 0.0f;
  float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  float qr0 = 0.0f, qr1 = 0.0f, m = -INFINITY, l = 0.0f;
  if (active) {
    const device float4* ql4 = reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h) * LAT + 16u * lane);
    qv0 = ql4[0]; qv1 = ql4[1]; qv2 = ql4[2]; qv3 = ql4[3];
    const float2 qrv = *reinterpret_cast<const device float2*>(qrope + (ulong(r) * a.heads + h) * ROPE + 2u * lane);
    qr0 = qrv.x; qr1 = qrv.y;
  }
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  const uint tb = glm_tree_base(a, row.pos);
  const uint anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  for (uint j0 = span.x; j0 < span.y; j0 += 4u) {
    const uint nk = min(4u, span.y - j0);
    const uint total_u4 = nk * 72u;
    for (uint idx = tid; idx < total_u4; idx += 256u) {
      const uint bk = idx / 72u, w = idx % 72u;
      const uint k_ord = row.pos < 2048u ? (j0 + bk) : keys[row.list + j0 + bk];
      const uint kid = glm_tree_kid(k_ord, tb, anc);
      k_buf_u4[bk * 72u + w] = *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + w * 8u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint bk = 0; bk < nk; ++bk) {
        const threadgroup uint4* k4 = k_buf_u4 + bk * 72u;
        const uint4 w0 = k4[2u * lane + 0u];
        const uint4 w1 = k4[2u * lane + 1u];
        const uint kr_u = reinterpret_cast<const threadgroup uint*>(k4 + 64u)[lane];
        const float4 kv0 = float4(as_type<float>(w0.x << 16), as_type<float>(w0.x & 0xFFFF0000u),
                                  as_type<float>(w0.y << 16), as_type<float>(w0.y & 0xFFFF0000u));
        const float4 kv1 = float4(as_type<float>(w0.z << 16), as_type<float>(w0.z & 0xFFFF0000u),
                                  as_type<float>(w0.w << 16), as_type<float>(w0.w & 0xFFFF0000u));
        const float4 kv2 = float4(as_type<float>(w1.x << 16), as_type<float>(w1.x & 0xFFFF0000u),
                                  as_type<float>(w1.y << 16), as_type<float>(w1.y & 0xFFFF0000u));
        const float4 kv3 = float4(as_type<float>(w1.z << 16), as_type<float>(w1.z & 0xFFFF0000u),
                                  as_type<float>(w1.w << 16), as_type<float>(w1.w & 0xFFFF0000u));
        float p = dot(qv0, kv0) + dot(qv1, kv1) + dot(qv2, kv2) + dot(qv3, kv3);
        p = fma(qr0, as_type<float>(kr_u << 16), p);
        p = fma(qr1, as_type<float>(kr_u & 0xFFFF0000u), p);
        const float sc = glm_lanes(p) * a.scale;
        const float mn = max(m, sc);
        const float corr = exp(m - mn), e = exp(sc - mn);
        l = fma(l, corr, e);
        o0 = fma(float4(e), kv0, o0 * corr);
        o1 = fma(float4(e), kv1, o1 * corr);
        o2 = fma(float4(e), kv2, o2 * corr);
        o3 = fma(float4(e), kv3, o3 * corr);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active) return;
  if (ns == 1u) {
    const float inv_l = 1.0f / l;
    device float4* out4 = reinterpret_cast<device float4*>(dst + (ulong(r) * a.heads + h) * LAT + 16u * lane);
    out4[0] = o0 * inv_l; out4[1] = o1 * inv_l; out4[2] = o2 * inv_l; out4[3] = o3 * inv_l;
  } else {
    device float* out = dst + ((ulong(r) * a.heads + h) * stride + s) * (LAT + 2u);
    if (lane == 0u) out[0] = m, out[1] = l;
    out[2u + 16u * lane + 0u] = o0.x; out[2u + 16u * lane + 1u] = o0.y; out[2u + 16u * lane + 2u] = o0.z; out[2u + 16u * lane + 3u] = o0.w;
    out[2u + 16u * lane + 4u] = o1.x; out[2u + 16u * lane + 5u] = o1.y; out[2u + 16u * lane + 6u] = o1.z; out[2u + 16u * lane + 7u] = o1.w;
    out[2u + 16u * lane + 8u] = o2.x; out[2u + 16u * lane + 9u] = o2.y; out[2u + 16u * lane + 10u] = o2.z; out[2u + 16u * lane + 11u] = o2.w;
    out[2u + 16u * lane + 12u] = o3.x; out[2u + 16u * lane + 13u] = o3.y; out[2u + 16u * lane + 14u] = o3.z; out[2u + 16u * lane + 15u] = o3.w;
  }
}

// One (row, head) a simdgroup over a split of its keys: single-load uint4 key reads, online softmax, exact order.
kernel void glm_mla_attend(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                           const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                           const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                           device float* partial [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.z, s = tg.y, h = tg.x * 4 + sg;
  if (h >= a.heads) return;
  const MlaRow row = rows[r];
  const uint nkeys = row.pos < 2048u ? (row.pos + 1u) : row.nkeys;
  const uint2 span = glm_split(nkeys, s);
  device float* out = partial + ((ulong(r) * a.heads + h) * SPLITS + s) * (LAT + 2);
  if (span.x >= span.y) return;
  float qv[16], qr[2], o[16], m = -INFINITY, l = 0.0f;
  const device float4* ql4 = reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h) * LAT + 16u * lane);
  for (uint i = 0; i < 4u; ++i) {
    const float4 v = ql4[i];
    qv[4u * i + 0u] = v.x; qv[4u * i + 1u] = v.y; qv[4u * i + 2u] = v.z; qv[4u * i + 3u] = v.w;
    o[4u * i + 0u] = 0.0f; o[4u * i + 1u] = 0.0f; o[4u * i + 2u] = 0.0f; o[4u * i + 3u] = 0.0f;
  }
  const float2 qrv = *reinterpret_cast<const device float2*>(qrope + (ulong(r) * a.heads + h) * ROPE + 2u * lane);
  qr[0] = qrv.x; qr[1] = qrv.y;
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  const uint tb = glm_tree_base(a, row.pos);
  const uint anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  for (uint j = span.x; j < span.y; ++j) {
    const uint k_ord = row.pos < 2048u ? j : keys[row.list + j];
    const uint kid = glm_tree_kid(k_ord, tb, anc);
    const device bfloat* key = base + ulong(kid) * KEY;
    // Two coalesced 16-byte loads for 16 bfloat latent dims + one 4-byte load for 2 rope dims; cached in regs!
    const device uint4* k4 = reinterpret_cast<const device uint4*>(key + 16u * lane);
    const uint4 w0 = k4[0], w1 = k4[1];
    const uint kr_u = *reinterpret_cast<const device uint*>(key + LAT + 2u * lane);
    float kv[16];
    kv[0]  = as_type<float>(w0.x << 16); kv[1]  = as_type<float>(w0.x & 0xFFFF0000u);
    kv[2]  = as_type<float>(w0.y << 16); kv[3]  = as_type<float>(w0.y & 0xFFFF0000u);
    kv[4]  = as_type<float>(w0.z << 16); kv[5]  = as_type<float>(w0.z & 0xFFFF0000u);
    kv[6]  = as_type<float>(w0.w << 16); kv[7]  = as_type<float>(w0.w & 0xFFFF0000u);
    kv[8]  = as_type<float>(w1.x << 16); kv[9]  = as_type<float>(w1.x & 0xFFFF0000u);
    kv[10] = as_type<float>(w1.y << 16); kv[11] = as_type<float>(w1.y & 0xFFFF0000u);
    kv[12] = as_type<float>(w1.z << 16); kv[13] = as_type<float>(w1.z & 0xFFFF0000u);
    kv[14] = as_type<float>(w1.w << 16); kv[15] = as_type<float>(w1.w & 0xFFFF0000u);
    float p = 0.0f;
    for (uint i = 0; i < 16u; ++i) p = fma(qv[i], kv[i], p);
    p = fma(qr[0], as_type<float>(kr_u << 16), p);
    p = fma(qr[1], as_type<float>(kr_u & 0xFFFF0000u), p);
    const float sc = glm_lanes(p) * a.scale;
    const float mn = max(m, sc), corr = precise::exp(m - mn), e = precise::exp(sc - mn);
    l = fma(l, corr, e);
    for (uint i = 0; i < 16u; ++i) o[i] = fma(e, kv[i], o[i] * corr);
    m = mn;
  }
  if (lane == 0) out[0] = m, out[1] = l;
  for (uint i = 0; i < 16u; ++i) out[2u + 16u * lane + i] = o[i];
}

// Parallel SIMD split merge (grid=(heads, rows), 128t): simd_max/simd_sum over <=64 splits + float2/float4 vector I/O!
kernel void glm_mla_merge(const device float* partial [[buffer(0)]], const device MlaRow* rows [[buffer(1)]],
                          constant MlaArgs& a [[buffer(2)]], device float* olat [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float f_tg[64], inv_l_tg;
  const uint r = tg.y, h = tg.x;
  const uint n = rows[r].pos < 2048u ? (rows[r].pos + 1u) : rows[r].nkeys;
  const uint ns = a.pad2, stride = ns == 0u ? SPLITS : ns;
  const device float* p = partial + (ulong(r) * a.heads + h) * stride * (LAT + 2u);
  if (t < 32u) {
    const bool v0 = (t < stride) && (ns == 0u ? (glm_split(n, t).x < glm_split(n, t).y) : (p[t * (LAT + 2u) + 1u] > 0.0f));
    const bool v1 = (t + 32u < stride) && (ns != 0u) && (p[(t + 32u) * (LAT + 2u) + 1u] > 0.0f);
    const float m0 = v0 ? p[t * (LAT + 2u)] : -INFINITY;
    const float m1 = v1 ? p[(t + 32u) * (LAT + 2u)] : -INFINITY;
    const float M = simd_max(max(m0, m1));
    const float e0 = v0 ? precise::exp(m0 - M) : 0.0f;
    const float e1 = v1 ? precise::exp(m1 - M) : 0.0f;
    const float L = simd_sum((v0 ? p[t * (LAT + 2u) + 1u] * e0 : 0.0f) +
                             (v1 ? p[(t + 32u) * (LAT + 2u) + 1u] * e1 : 0.0f));
    if (t < stride) f_tg[t] = e0;
    if (t + 32u < stride) f_tg[t + 32u] = e1;
    if (lane == 0u) inv_l_tg = L > 0.0f ? precise::divide(1.0f, L) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c4 = t * 4u;
  float4 acc = float4(0.0f);
  for (uint s = 0u; s < stride; ++s) {
    const float f = f_tg[s];
    if (f > 0.0f) {
      const device float2* ps2 = reinterpret_cast<const device float2*>(p + s * (LAT + 2u) + 2u + c4);
      acc = fma(float4(ps2[0], ps2[1]), float4(f), acc);
    }
  }
  *reinterpret_cast<device float4*>(olat + (ulong(r) * a.heads + h) * LAT + c4) = acc * inv_l_tg;
}

#if __METAL_VERSION__ >= 310
// Single-read 8-head x 16-key 8-way D-split Q-register Flash-MLA (9 Q-regs, 16.5 KB TG -> 2 TGs/core, exp2!).
template <bool SPLIT>
inline void glm_mla_flash_impl(const device float* qlat, const device float* qrope, const device MlaRow* rows,
                               const device uint* keys, const device bfloat* cache, constant MlaArgs& a,
                               device float* dst_buf, uint r, uint s, uint h0, uint tid, uint sg, uint lane,
                               threadgroup uint4* k_lat_u4, threadgroup uint4* k_rope_u4, threadgroup float* sm,
                               threadgroup bfloat* p_bf, threadgroup float* m_tg,
                               threadgroup float* l_tg, threadgroup float* corr_tg) {
  constexpr uint KLD = 520u, KRLD = 72u, PLD = 24u;
  constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
  threadgroup bfloat* k_lat = reinterpret_cast<threadgroup bfloat*>(k_lat_u4);
  threadgroup bfloat* k_rope = reinterpret_cast<threadgroup bfloat*>(k_rope_u4);
  if (r >= a.rows || h0 + 8u > a.heads) return;
  const MlaRow row = rows[r];
  if (SPLIT && a.pad0 != 0.0f && row.pos < 256u) {
    if (s == 0u && tid < 8u) dst_buf[((ulong(r) * a.heads + h0 + tid) * a.pad2) * (LAT + 2u) + 1u] = -1.0f;
    return;
  }
  const bool direct_keys = row.pos < 2048u;
  const uint nkeys = direct_keys ? (row.pos + 1u) : row.nkeys;
  const uint ns = SPLIT ? a.pad2 : 1u;
  uint k_start = 0u, k_end = nkeys;
  if (SPLIT) {
    const uint nb = (nkeys + 15u) >> 4;
    k_start = min(((nb * s) / ns) << 4, nkeys);
    k_end = min(((nb * (s + 1u)) / ns) << 4, nkeys);
    if (k_start >= k_end) {
      if (tid < 8u) {
        device float* out = dst_buf + ((ulong(r) * a.heads + h0 + tid) * ns + s) * (LAT + 2u);
        out[0] = -INFINITY;
        out[1] = 0.0f;
      }
      return;
    }
  }
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  const float scale2 = a.scale * LOG2E;
  if (tid < 8u) { m_tg[tid] = -INFINITY; l_tg[tid] = 0.0f; corr_tg[tid] = 0.0f; }
  for (uint idx = tid; idx < 8u * 128u; idx += 256u) {
    const uint hh = idx >> 7, cc4 = idx & 127u;
    const float4 v = *reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h0 + hh) * LAT + cc4 * 4u);
    k_lat[hh * KLD + cc4 * 4u + 0u] = bfloat(v.x); k_lat[hh * KLD + cc4 * 4u + 1u] = bfloat(v.y);
    k_lat[hh * KLD + cc4 * 4u + 2u] = bfloat(v.z); k_lat[hh * KLD + cc4 * 4u + 3u] = bfloat(v.w);
  }
  if (tid < 128u) {
    const uint hh = tid >> 4, cc4 = tid & 15u;
    const float4 v = *reinterpret_cast<const device float4*>(qrope + (ulong(r) * a.heads + h0 + hh) * ROPE + cc4 * 4u);
    k_rope[hh * KRLD + cc4 * 4u + 0u] = bfloat(v.x); k_rope[hh * KRLD + cc4 * 4u + 1u] = bfloat(v.y);
    k_rope[hh * KRLD + cc4 * 4u + 2u] = bfloat(v.z); k_rope[hh * KRLD + cc4 * 4u + 3u] = bfloat(v.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint d_lat0 = sg * 64u, d_rop0 = sg * 8u;
  simdgroup_matrix<bfloat, 8, 8> ql[8], qr;
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) simdgroup_load(ql[i], k_lat + d_lat0 + i * 8u, KLD, ulong2(0, 0), false);
  simdgroup_load(qr, k_rope + d_rop0, KRLD, ulong2(0, 0), false);
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1);
  const uint hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  simdgroup_matrix<float, 8, 8> o[8];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) o[i] = simdgroup_matrix<float, 8, 8>(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uint* klist = keys + row.list;
  const uint tb = glm_tree_base(a, row.pos);
  const uint anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  for (uint k0 = k_start; k0 < k_end; k0 += 16u) {
    const uint nk = min(16u, k_end - k0);
    #pragma clang loop unroll(full)
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint idx = pass * 256u + tid, kk = idx >> 6, c8 = idx & 63u;
      const uint k_ord = (kk < nk) ? (direct_keys ? (k0 + kk) : klist[k0 + kk]) : 0u;
      const uint kid = (kk < nk) ? glm_tree_kid(k_ord, tb, anc) : 0u;
      k_lat_u4[kk * (KLD / 8u) + c8] =
          (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + c8 * 8u) : uint4(0u);
    }
    if (tid < 128u) {
      const uint kk = tid >> 3, c8 = tid & 7u;
      const uint k_ord = (kk < nk) ? (direct_keys ? (k0 + kk) : klist[k0 + kk]) : 0u;
      const uint kid = (kk < nk) ? glm_tree_kid(k_ord, tb, anc) : 0u;
      k_rope_u4[kk * (KRLD / 8u) + c8] =
          (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + LAT + c8 * 8u) : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s0(0.0f), s1(0.0f);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      simdgroup_matrix<bfloat, 8, 8> kb0, kb1;
      simdgroup_load(kb0, k_lat + d_lat0 + i * 8u, KLD, ulong2(0, 0), true);
      simdgroup_load(kb1, k_lat + 8u * KLD + d_lat0 + i * 8u, KLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s0, ql[i], kb0, s0);
      simdgroup_multiply_accumulate(s1, ql[i], kb1, s1);
    }
    {
      simdgroup_matrix<bfloat, 8, 8> kb0, kb1;
      simdgroup_load(kb0, k_rope + d_rop0, KRLD, ulong2(0, 0), true);
      simdgroup_load(kb1, k_rope + 8u * KRLD + d_rop0, KRLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s0, qr, kb0, s0);
      simdgroup_multiply_accumulate(s1, qr, kb1, s1);
    }
    simdgroup_store(s0, sm + sg * 128u, 16u, ulong2(0, 0), false);
    simdgroup_store(s1, sm + sg * 128u + 8u, 16u, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < 16u) {
      const uint bs = sg * 16u + lane;
      const float sv = (lane < nk) ? (((sm[bs] + sm[128u + bs]) + (sm[256u + bs] + sm[384u + bs]) +
                                       (sm[512u + bs] + sm[640u + bs]) + (sm[768u + bs] + sm[896u + bs])) * scale2) : -INFINITY;
      float v = sv;
      for (ushort d = 8; d > 0; d >>= 1) v = max(v, simd_shuffle_xor(v, d));
      const float m_old = m_tg[sg], m_new = max(m_old, v);
      const float corr = (m_old == -INFINITY) ? 0.0f : exp2(m_old - m_new);
      const float ev = (lane < nk) ? exp2(sv - m_new) : 0.0f;
      float ls = ev;
      for (ushort d = 8; d > 0; d >>= 1) ls += simd_shuffle_xor(ls, d);
      p_bf[sg * PLD + lane] = bfloat(ev);
      if (lane == 0u) { m_tg[sg] = m_new; l_tg[sg] = fma(l_tg[sg], corr, ls); corr_tg[sg] = corr; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k0 > k_start) {
      const float cr = corr_tg[hy];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) { o[i].thread_elements()[0] *= cr; o[i].thread_elements()[1] *= cr; }
    }
    const uint v_col0 = sg * 64u;
    #pragma clang loop unroll(full)
    for (uint kd = 0u; kd < 16u; kd += 8u) {
      simdgroup_matrix<bfloat, 8, 8> pa;
      simdgroup_load(pa, p_bf + kd, PLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        simdgroup_matrix<bfloat, 8, 8> vv;
        simdgroup_load(vv, k_lat + kd * KLD + v_col0 + i * 8u, KLD, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(o[i], pa, vv, o[i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (SPLIT) {
    device float* out = dst_buf + ((ulong(r) * a.heads + h0 + hy) * ns + s) * (LAT + 2u);
    if (sg == 0u && hx == 0u) { out[0] = (m_tg[hy] == -INFINITY) ? -INFINITY : (m_tg[hy] * LN2); out[1] = l_tg[hy]; }
    device float* dst = out + 2u + sg * 64u + hx;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) { dst[i * 8u] = o[i].thread_elements()[0]; dst[i * 8u + 1u] = o[i].thread_elements()[1]; }
  } else {
    const float inv_l = (l_tg[hy] > 0.0f) ? precise::divide(1.0f, l_tg[hy]) : 0.0f;
    device float* dst = dst_buf + (ulong(r) * a.heads + h0 + hy) * LAT + sg * 64u + hx;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      dst[i * 8u] = o[i].thread_elements()[0] * inv_l;
      dst[i * 8u + 1u] = o[i].thread_elements()[1] * inv_l;
    }
  }
}

// Single-launch Flash-MLA for prompt windows (grid=(heads/8, rows), 256t): writes normalized olat directly.
kernel void glm_mla_attend_flash(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                 const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                 const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                 device float* olat [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 k_lat_u4[16u * 65u], k_rope_u4[16u * 9u];
  threadgroup float sm[8u * 8u * 16u], m_tg[8u], l_tg[8u], corr_tg[8u];
  threadgroup bfloat p_bf[8u * 24u];
  glm_mla_flash_impl<false>(qlat, qrope, rows, keys, cache, a, olat, tg.y, 0u, tg.x * 8u, tid, sg, lane,
                            k_lat_u4, k_rope_u4, sm, p_bf, m_tg, l_tg, corr_tg);
}

// Split-K Flash-MLA for decode (grid=(heads/8, ns, rows), 256t): single-read KV cache, writes partial for merge.
kernel void glm_mla_attend_flash_decode(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                        const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                        const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                        device float* partial [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 k_lat_u4[16u * 65u], k_rope_u4[16u * 9u];
  threadgroup float sm[8u * 8u * 16u], m_tg[8u], l_tg[8u], corr_tg[8u];
  threadgroup bfloat p_bf[8u * 24u];
  glm_mla_flash_impl<true>(qlat, qrope, rows, keys, cache, a, partial, tg.z, tg.y, tg.x * 8u, tid, sg, lane,
                           k_lat_u4, k_rope_u4, sm, p_bf, m_tg, l_tg, corr_tg);
}
#endif
