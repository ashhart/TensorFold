// GLM-5.3's DSA indexer for lane rows: bf16 index keys, every visible key's score, each row's top-k by position.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif
using namespace metal;

constant constexpr uint IH = 32, ID = 128, IROPE = 64, TOPK = 2048;

struct IndexRow { uint slot, pos, list, nkeys; };
struct IndexArgs {
  uint rows, slot_keys, proj_stride, k_col, w_col, q_stride, score_stride, pad0;
  float eps, key_scale, head_scale, pad1;
  uint tree_base, tree_p0, tree_p1, tree_p2, tree_anc[8];
};
inline uint glm_idx_base(constant IndexArgs& a, uint pos) { return (a.tree_base == 0xFFFFFFFEu) ? (pos - a.tree_p1) : a.tree_base; }
inline uint glm_idx_kid(uint k_ord, uint tb, uint anc) { return (k_ord < tb) ? k_ord : (tb + ((anc >> (4u * (k_ord - tb))) & 0xFu)); }
inline float glm_index_lanes(float v) { for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d); return v; }

inline float2 glm_index_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

// One row a simdgroup: the key = bf16(rope_64(LayerNorm(wk x))) into the index cache at the row's position.
kernel void glm_index_key(const device float* proj [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                          const device bfloat* b [[buffer(2)]], const device float* inv [[buffer(3)]],
                          const device IndexRow* rows [[buffer(4)]], constant IndexArgs& a [[buffer(5)]],
                          device bfloat* cache [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const device float* x = proj + ulong(row) * a.proj_stride + a.k_col;
  float v[4], s = 0.0f;
  for (uint q = 0; q < 4; ++q) v[q] = x[lane + 32 * q], s += v[q];
  const float mean = glm_index_lanes(s) / float(ID);
  float d = 0.0f;
  for (uint q = 0; q < 4; ++q) d = fma(v[q] - mean, v[q] - mean, d);
  const float rstd = precise::rsqrt(glm_index_lanes(d) / float(ID) + a.eps);
  threadgroup float y[ID];
  for (uint q = 0; q < 4; ++q) {
    const uint i = lane + 32 * q;
    y[i] = fma((v[q] - mean) * rstd, float(w[i]), float(b[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const IndexRow r = rows[row];
  device bfloat* key = cache + (ulong(r.slot) * a.slot_keys + r.pos) * ID;
  const float2 o = glm_index_rope(y[2 * lane], y[2 * lane + 1], r.pos, inv, lane);
  key[lane] = bfloat(o.x);
  key[IROPE / 2 + lane] = bfloat(o.y);
  key[IROPE + lane] = bfloat(y[IROPE + lane]);
  key[IROPE + 32 + lane] = bfloat(y[IROPE + 32 + lane]);
}

// Row r, keys 256 tg.x ..: score = sum_h (w_h / sqrt 32) relu(q_h . k / sqrt 128), heads in order, q roped in place.
kernel void glm_index_scores(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                             const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                             constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                             const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
  threadgroup float qs[IH * ID];
  threadgroup float ws[IH];
  const uint r = tg.y;
  const IndexRow row = rows[r];
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint i = t; i < IH * ID; i += 256) {
    const uint h = i / ID, c = i % ID;
    if (c >= IROPE) {
      qs[i] = qr[i];
    } else if (c < IROPE / 2) {
      const float2 o = glm_index_rope(qr[h * ID + 2 * c], qr[h * ID + 2 * c + 1], row.pos, inv, c);
      qs[h * ID + c] = o.x;
      qs[h * ID + IROPE / 2 + c] = o.y;
    }
  }
  if (t < IH) ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = tg.x * 256 + t;
  if (k > row.pos) return;
  const uint tb = glm_idx_base(a, row.pos), anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  const device bfloat* key = cache + (ulong(row.slot) * a.slot_keys + glm_idx_kid(k, tb, anc)) * ID;
  float kv[ID];
  for (uint i = 0; i < ID; ++i) kv[i] = float(key[i]);
  float s = 0.0f;
  for (uint h = 0; h < IH; ++h) {
    float d = 0.0f;
    for (uint i = 0; i < ID; ++i) d = fma(qs[h * ID + i], kv[i], d);
    s = fma(ws[h], max(d * a.key_scale, 0.0f), s);
  }
  score[ulong(r) * a.score_stride + k] = s;
}

// Row r's scores for 32 keys (decode) or 256 keys (prompt, 8x fewer Q/trig loads) via simdgroup_matrix MMA.
constant constexpr uint SCORE_LD = 136u;

#if __METAL_VERSION__ >= 310
template <uint TILES>
inline void glm_index_scores_mma(const device float* q, const device float* proj, const device bfloat* cache,
                                 const device IndexRow* rows, constant IndexArgs& a, device float* score,
                                 const device float* inv, threadgroup bfloat* qs, threadgroup uint4* ks_u4,
                                 threadgroup float* ws, threadgroup float* cs, threadgroup float* dots,
                                 uint2 tg, uint t, uint sg) {
  const uint r = tg.y;
  const IndexRow row = rows[r];
  const uint k_base = tg.x * (32u * TILES);
  if (row.pos < TOPK || k_base > row.pos) return;
  constexpr uint LD = SCORE_LD;
  threadgroup bfloat* ks = reinterpret_cast<threadgroup bfloat*>(ks_u4);
  if (t < IROPE / 2u) {
    const float ang = float(row.pos) * inv[t];
    cs[2u * t] = precise::cos(ang);
    cs[2u * t + 1u] = precise::sin(ang);
  }
  if (t < IH) ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint i = t; i < IH * ID; i += 256u) {
    const uint h = i >> 7, c = i & 127u;
    if (c >= IROPE) {
      qs[h * LD + c] = bfloat(qr[i]);
    } else if (c < IROPE / 2u) {
      const float x0 = qr[h * ID + 2u * c], x1 = qr[h * ID + 2u * c + 1u];
      const float co = cs[2u * c], si = cs[2u * c + 1u];
      qs[h * LD + c] = bfloat(fma(x0, co, -(x1 * si)));
      qs[h * LD + IROPE / 2u + c] = bfloat(fma(x1, co, x0 * si));
    }
  }
  const device bfloat* kbase = cache + ulong(row.slot) * a.slot_keys * ID;
  const uint tb = glm_idx_base(a, row.pos), anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  const uint sh = sg >> 1, sk0 = (sg & 1u) * 16u;
  const uint ntiles = min(TILES, (row.pos - k_base) / 32u + 1u);
  for (uint ti = 0u; ti < ntiles; ++ti) {
    const uint k0 = k_base + ti * 32u;
    for (uint idx = t; idx < 32u * 16u; idx += 256u) {
      const uint bk = idx >> 4, c8 = idx & 15u, kk = k0 + bk;
      const uint kid = (kk <= row.pos) ? glm_idx_kid(kk, tb, anc) : 0u;
      ks_u4[bk * (LD / 8u) + c8] =
          (kk <= row.pos) ? *reinterpret_cast<const device uint4*>(kbase + ulong(kid) * ID + c8 * 8u) : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> d0(0.0f), d1(0.0f);
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < ID; c += 8u) {
      simdgroup_matrix<bfloat, 8, 8> qa, kb0, kb1;
      simdgroup_load(qa, qs + (sh * 8u) * LD + c, LD, ulong2(0, 0), false);
      simdgroup_load(kb0, ks + sk0 * LD + c, LD, ulong2(0, 0), true);
      simdgroup_load(kb1, ks + (sk0 + 8u) * LD + c, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(d0, qa, kb0, d0);
      simdgroup_multiply_accumulate(d1, qa, kb1, d1);
    }
    simdgroup_store(d0, dots + (sh * 8u) * 32u + sk0, 32u, ulong2(0, 0), false);
    simdgroup_store(d1, dots + (sh * 8u) * 32u + sk0 + 8u, 32u, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32u && k0 + t <= row.pos) {
      float s = 0.0f;
      for (uint h = 0u; h < IH; ++h) s = fma(ws[h], max(dots[h * 32u + t] * a.key_scale, 0.0f), s);
      score[ulong(r) * a.score_stride + k0 + t] = s;
    }
    if (TILES > 1u && ti + 1u < ntiles) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#else
template <uint TILES>
inline void glm_index_scores_mma(const device float* q, const device float* proj, const device bfloat* cache,
                                 const device IndexRow* rows, constant IndexArgs& a, device float* score,
                                 const device float* inv, threadgroup float* qs, threadgroup float* ws,
                                 uint2 tg, uint t) {
  const uint r = tg.y;
  const IndexRow row = rows[r];
  const uint k_base = tg.x * (32u * TILES);
  if (row.pos < TOPK || k_base > row.pos) return;
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint i = t; i < IH * ID; i += 256u) {
    const uint h = i / ID, c = i % ID;
    if (c >= IROPE) qs[i] = qr[i];
    else if (c < IROPE / 2u) {
      const float2 o = glm_index_rope(qr[h * ID + 2u * c], qr[h * ID + 2u * c + 1u], row.pos, inv, c);
      qs[h * ID + c] = o.x;
      qs[h * ID + IROPE / 2u + c] = o.y;
    }
  }
  if (t < IH) ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint ti = 0u; ti < TILES; ++ti) {
    const uint k0 = k_base + ti * 32u;
    if (t < 32u && k0 + t <= row.pos) {
      const device bfloat* key = cache + (ulong(row.slot) * a.slot_keys + k0 + t) * ID;
      float s = 0.0f;
      for (uint h = 0u; h < IH; ++h) {
        float d = 0.0f;
        for (uint i = 0u; i < ID; ++i) d = fma(qs[h * ID + i], float(key[i]), d);
        s = fma(ws[h], max(d * a.key_scale, 0.0f), s);
      }
      score[ulong(r) * a.score_stride + k0 + t] = s;
    }
  }
}
#endif

kernel void glm_index_scores_block(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                                   const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                                   constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                                   const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
#if __METAL_VERSION__ >= 310
  threadgroup bfloat qs[IH * SCORE_LD];
  threadgroup uint4 ks_u4[32u * (SCORE_LD / 8u)];
  threadgroup float ws[IH], cs[IROPE], dots[IH * 32u];
  glm_index_scores_mma<1u>(q, proj, cache, rows, a, score, inv, qs, ks_u4, ws, cs, dots, tg, t, sg);
#else
  threadgroup float qs[IH * ID], ws[IH];
  glm_index_scores_mma<1u>(q, proj, cache, rows, a, score, inv, qs, ws, tg, t);
#endif
}

#if __METAL_VERSION__ >= 310
union IdxPromptShared {
  uint4 ks_u4[64u * 9u];
  float cs[2u * IROPE];
  float red[8u * 64u];
};
union IdxDecShared {
  uint4 ks_u4[64u * 9u];
  float cs[IROPE];
  float red[4u * 64u];
};
#endif

// Decode 1-row Q-resident 128-key (2x64) MMA + register-shuffle ReLU reduce (2 TGs/core, -40% at 8.9K ctx!).
kernel void glm_index_scores_dec_128(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                                     const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                                     constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                                     const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
#if __METAL_VERSION__ >= 310
  constexpr uint QLD = SCORE_LD, KLD = 72u;
  threadgroup bfloat qs[IH * QLD];
  threadgroup IdxDecShared sh;
  threadgroup bfloat* ks = reinterpret_cast<threadgroup bfloat*>(sh.ks_u4);
  threadgroup float ws[IH];
  const uint r = tg.y;
  if (r >= a.rows) return;
  const IndexRow row = rows[r];
  const uint k_base = tg.x * 128u;
  if (row.pos < TOPK || k_base > row.pos) return;
  if (t < 32u) {
    const float ang = float(row.pos) * inv[t];
    sh.cs[2u * t] = precise::cos(ang);
    sh.cs[2u * t + 1u] = precise::sin(ang);
    ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint p = 0u; p < 4u; ++p) {
    const uint idx = p * 256u + t, h = idx >> 5, cc = idx & 31u;
    const float2 v = *reinterpret_cast<const device float2*>(qr + h * ID + 2u * cc);
    const float co = sh.cs[2u * cc], si = sh.cs[2u * cc + 1u];
    qs[h * QLD + cc] = bfloat(fma(v.x, co, -(v.y * si)));
    qs[h * QLD + IROPE / 2u + cc] = bfloat(fma(v.y, co, v.x * si));
  }
  for (uint p = 0u; p < 2u; ++p) {
    const uint idx = p * 256u + t, h = idx >> 4, cc4 = idx & 15u;
    const float4 v = *reinterpret_cast<const device float4*>(qr + h * ID + IROPE + cc4 * 4u);
    qs[h * QLD + IROPE + cc4 * 4u + 0u] = bfloat(v.x);
    qs[h * QLD + IROPE + cc4 * 4u + 1u] = bfloat(v.y);
    qs[h * QLD + IROPE + cc4 * 4u + 2u] = bfloat(v.z);
    qs[h * QLD + IROPE + cc4 * 4u + 3u] = bfloat(v.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1), hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  const uint sh_g = sg >> 1, sk0 = (sg & 1u) * 32u;
  const float wr0 = ws[sh_g * 8u + hy];
  const device bfloat* kbase = cache + ulong(row.slot) * a.slot_keys * ID;
  const uint tb = glm_idx_base(a, row.pos), anc = (tb != 0xFFFFFFFFu && r < 8u) ? a.tree_anc[r] : 0u;
  const uint ntiles = min(2u, (row.pos - k_base) / 64u + 1u);
  for (uint ti = 0u; ti < ntiles; ++ti) {
    const uint k0 = k_base + ti * 64u;
    simdgroup_matrix<float, 8, 8> d0[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) d0[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kh = 0u; kh < 2u; ++kh) {
      const uint c0 = kh * 64u;
      for (uint p = 0u; p < 2u; ++p) {
        const uint idx = p * 256u + t, bk = idx >> 3, c8 = idx & 7u, kk = k0 + bk;
        const uint kid = (kk <= row.pos) ? glm_idx_kid(kk, tb, anc) : 0u;
        sh.ks_u4[bk * 9u + c8] =
            (kk <= row.pos) ? *reinterpret_cast<const device uint4*>(kbase + ulong(kid) * ID + c0 + c8 * 8u) : uint4(0u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 64u; c += 8u) {
        simdgroup_matrix<bfloat, 8, 8> qa0;
        simdgroup_load(qa0, qs + (sh_g * 8u) * QLD + c0 + c, QLD, ulong2(0, 0), false);
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
          simdgroup_matrix<bfloat, 8, 8> kb;
          simdgroup_load(kb, ks + (sk0 + j * 8u) * KLD + c, KLD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(d0[j], qa0, kb, d0[j]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      float s0a = wr0 * max(d0[j].thread_elements()[0] * a.key_scale, 0.0f);
      float s0b = wr0 * max(d0[j].thread_elements()[1] * a.key_scale, 0.0f);
      s0a += simd_shuffle_xor(s0a, 2); s0a += simd_shuffle_xor(s0a, 4); s0a += simd_shuffle_xor(s0a, 16);
      s0b += simd_shuffle_xor(s0b, 2); s0b += simd_shuffle_xor(s0b, 4); s0b += simd_shuffle_xor(s0b, 16);
      if (hy == 0u) {
        sh.red[sh_g * 64u + sk0 + j * 8u + hx + 0u] = s0a;
        sh.red[sh_g * 64u + sk0 + j * 8u + hx + 1u] = s0b;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64u && k0 + t <= row.pos) {
      score[ulong(r) * a.score_stride + k0 + t] =
          sh.red[t] + sh.red[64u + t] + sh.red[128u + t] + sh.red[192u + t];
    }
    if (ti + 1u < ntiles) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#else
  (void)sg; (void)lane;
  threadgroup float qs[IH * ID], ws[IH];
  glm_index_scores_mma<4u>(q, proj, cache, rows, a, score, inv, qs, ws, tg, t);
#endif
}

// Prompt 2-row Q-resident 64-key MMA + register-shuffle ReLU reduce (grid=(last/512+1, ceil(rows/2)), 256t): -13.5%!
kernel void glm_index_scores_prompt(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                                    const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                                    constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                                    const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
#if __METAL_VERSION__ >= 310
  constexpr uint QLD = SCORE_LD, KLD = 72u;
  threadgroup bfloat qs[2u * IH * QLD];
  threadgroup IdxPromptShared sh;
  threadgroup bfloat* ks = reinterpret_cast<threadgroup bfloat*>(sh.ks_u4);
  threadgroup float ws[2u * IH];
  const uint r0 = tg.y * 2u;
  if (r0 >= a.rows) return;
  const uint rn = min(2u, a.rows - r0);
  const IndexRow row0 = rows[r0], row1 = (rn > 1u) ? rows[r0 + 1u] : row0;
  const uint max_pos = max(row0.pos, row1.pos), k_base = tg.x * 512u;
  if (max_pos < TOPK || k_base > max_pos) return;
  if (t < 64u) {
    const uint dr = t >> 5, ci = t & 31u;
    const uint pos = (dr == 0u) ? row0.pos : row1.pos;
    const float ang = float(pos) * inv[ci];
    sh.cs[dr * IROPE + 2u * ci] = precise::cos(ang);
    sh.cs[dr * IROPE + 2u * ci + 1u] = precise::sin(ang);
    ws[t] = (dr < rn) ? (proj[ulong(r0 + dr) * a.proj_stride + a.w_col + ci] * a.head_scale) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = 0u; p < 8u; ++p) {
    const uint idx = p * 256u + t, qv = idx >> 5, cc = idx & 31u, dr = qv >> 5, dh = qv & 31u;
    if (dr < rn) {
      const float2 v = *reinterpret_cast<const device float2*>(q + ulong(r0 + dr) * a.q_stride + dh * ID + 2u * cc);
      const float co = sh.cs[dr * IROPE + 2u * cc], si = sh.cs[dr * IROPE + 2u * cc + 1u];
      qs[qv * QLD + cc] = bfloat(fma(v.x, co, -(v.y * si)));
      qs[qv * QLD + IROPE / 2u + cc] = bfloat(fma(v.y, co, v.x * si));
    } else {
      qs[qv * QLD + cc] = bfloat(0.0f);
      qs[qv * QLD + IROPE / 2u + cc] = bfloat(0.0f);
    }
  }
  for (uint p = 0u; p < 4u; ++p) {
    const uint idx = p * 256u + t, qv = idx >> 4, cc4 = idx & 15u, dr = qv >> 5, dh = qv & 31u;
    const float4 v = (dr < rn)
                         ? *reinterpret_cast<const device float4*>(q + ulong(r0 + dr) * a.q_stride + dh * ID + IROPE + cc4 * 4u)
                         : float4(0.0f);
    qs[qv * QLD + IROPE + cc4 * 4u + 0u] = bfloat(v.x);
    qs[qv * QLD + IROPE + cc4 * 4u + 1u] = bfloat(v.y);
    qs[qv * QLD + IROPE + cc4 * 4u + 2u] = bfloat(v.z);
    qs[qv * QLD + IROPE + cc4 * 4u + 3u] = bfloat(v.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1), hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  const uint sh_g = sg >> 1, sk0 = (sg & 1u) * 32u;
  const float wr0 = ws[sh_g * 8u + hy], wr1 = ws[IH + sh_g * 8u + hy];
  const device bfloat* kbase = cache + ulong(row0.slot) * a.slot_keys * ID;
  const uint ntiles = min(8u, (max_pos - k_base) / 64u + 1u);
  for (uint ti = 0u; ti < ntiles; ++ti) {
    const uint k0 = k_base + ti * 64u;
    simdgroup_matrix<float, 8, 8> d0[4], d1[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) d0[j] = simdgroup_matrix<float, 8, 8>(0.0f), d1[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kh = 0u; kh < 2u; ++kh) {
      const uint c0 = kh * 64u;
      for (uint p = 0u; p < 2u; ++p) {
        const uint idx = p * 256u + t, bk = idx >> 3, c8 = idx & 7u, kk = k0 + bk;
        sh.ks_u4[bk * 9u + c8] =
            (kk <= max_pos) ? *reinterpret_cast<const device uint4*>(kbase + ulong(kk) * ID + c0 + c8 * 8u) : uint4(0u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 64u; c += 8u) {
        simdgroup_matrix<bfloat, 8, 8> qa0, qa1;
        simdgroup_load(qa0, qs + (sh_g * 8u) * QLD + c0 + c, QLD, ulong2(0, 0), false);
        simdgroup_load(qa1, qs + (IH + sh_g * 8u) * QLD + c0 + c, QLD, ulong2(0, 0), false);
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
          simdgroup_matrix<bfloat, 8, 8> kb;
          simdgroup_load(kb, ks + (sk0 + j * 8u) * KLD + c, KLD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(d0[j], qa0, kb, d0[j]);
          simdgroup_multiply_accumulate(d1[j], qa1, kb, d1[j]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      float s0a = wr0 * max(d0[j].thread_elements()[0] * a.key_scale, 0.0f);
      float s0b = wr0 * max(d0[j].thread_elements()[1] * a.key_scale, 0.0f);
      float s1a = wr1 * max(d1[j].thread_elements()[0] * a.key_scale, 0.0f);
      float s1b = wr1 * max(d1[j].thread_elements()[1] * a.key_scale, 0.0f);
      s0a += simd_shuffle_xor(s0a, 2); s0a += simd_shuffle_xor(s0a, 4); s0a += simd_shuffle_xor(s0a, 16);
      s0b += simd_shuffle_xor(s0b, 2); s0b += simd_shuffle_xor(s0b, 4); s0b += simd_shuffle_xor(s0b, 16);
      s1a += simd_shuffle_xor(s1a, 2); s1a += simd_shuffle_xor(s1a, 4); s1a += simd_shuffle_xor(s1a, 16);
      s1b += simd_shuffle_xor(s1b, 2); s1b += simd_shuffle_xor(s1b, 4); s1b += simd_shuffle_xor(s1b, 16);
      if (hy == 0u) {
        sh.red[(sh_g * 2u + 0u) * 64u + sk0 + j * 8u + hx + 0u] = s0a;
        sh.red[(sh_g * 2u + 0u) * 64u + sk0 + j * 8u + hx + 1u] = s0b;
        sh.red[(sh_g * 2u + 1u) * 64u + sk0 + j * 8u + hx + 0u] = s1a;
        sh.red[(sh_g * 2u + 1u) * 64u + sk0 + j * 8u + hx + 1u] = s1b;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 128u) {
      const uint dr = t >> 6, bk = t & 63u, rpos = (dr == 0u) ? row0.pos : row1.pos;
      if (dr < rn && rpos >= TOPK && k0 + bk <= rpos) {
        score[ulong(r0 + dr) * a.score_stride + k0 + bk] =
            sh.red[dr * 64u + bk] + sh.red[(2u + dr) * 64u + bk] +
            sh.red[(4u + dr) * 64u + bk] + sh.red[(6u + dr) * 64u + bk];
      }
    }
    if (ti + 1u < ntiles) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#else
  (void)sg; (void)lane;
  threadgroup float qs[IH * ID], ws[IH];
  for (uint dr = 0u; dr < 2u; ++dr) {
    if (tg.y * 2u + dr < a.rows)
      glm_index_scores_mma<16u>(q, proj, cache, rows, a, score, inv, qs, ws, uint2(tg.x, tg.y * 2u + dr), t);
  }
#endif
}

// fp32 as an unsigned key whose order is the floats' order.
inline uint glm_order(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Register-cached 11-bit (2048-bucket) adaptive radix top-2048 select (~4x faster, bit-exact tie-break by position).
kernel void glm_index_topk(const device float* score [[buffer(0)]], device IndexRow* rows [[buffer(1)]],
                           constant IndexArgs& a [[buffer(2)]], device uint* list [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint B = 2048u;
  threadgroup atomic_uint hist[B];
  threadgroup uint sg_min[32], sg_max[32], sg_sum[32], found[3], sums[2][32];
  const IndexRow row = rows[r];
  const uint n = row.pos + 1u;
  device uint* out = list + row.list;
  const device float* sc = score + ulong(r) * a.score_stride;
  if (n <= TOPK) {
    for (uint i = t; i < n; i += 1024u) out[i] = i;
    if (t == 0u) rows[r].nkeys = n;
    return;
  }
  uint kv[16];
  const uint n_chunks = (n + 1023u) >> 10;
  const bool cached = n <= 16384u;
  uint lo = 0xFFFFFFFFu, hi = 0u;
  for (uint c = 0u; c < n_chunks; ++c) {
    const uint i = (c << 10) + t;
    if (i < n) {
      const uint k = glm_order(sc[i]);
      if (cached) kv[c] = k;
      lo = min(lo, k);
      hi = max(hi, k);
    }
  }
  lo = simd_min(lo); hi = simd_max(hi);
  if (lane == 0u) sg_min[sg] = lo, sg_max[sg] = hi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const uint glo = simd_min(sg_min[lane]), ghi = simd_max(sg_max[lane]);
    if (lane == 0u) found[0] = glo, found[1] = ghi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint glo = found[0], ghi = found[1], diff = glo ^ ghi;
  int top_bit = diff == 0u ? -1 : int(31u - clz(diff));
  uint prefix = (top_bit < 31) ? (glo & (top_bit < 0 ? 0xFFFFFFFFu : (~0u << (top_bit + 1)))) : 0u;
  uint mask = (top_bit < 31) ? (top_bit < 0 ? 0xFFFFFFFFu : (~0u << (top_bit + 1))) : 0u;
  uint need = TOPK;
  while (top_bit >= 0) {
    const int shift = max(0, top_bit - 10);
    const uint nb = 1u << uint(top_bit - shift + 1), bmask = nb - 1u;
    for (uint b = t; b < nb; b += 1024u) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0u; c < n_chunks; ++c) {
      const uint i = (c << 10) + t;
      if (i < n) {
        const uint k = cached ? kv[c] : glm_order(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & bmask], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const uint blk_sz = max(1u, nb >> 5), b_hi = nb - 1u - sg * blk_sz;
      uint s = 0u;
      if (sg * blk_sz < nb) for (uint j = lane; j < blk_sz; j += 32u) s += atomic_load_explicit(&hist[b_hi - j], memory_order_relaxed);
      s = simd_sum(s);
      if (lane == 0u) sg_sum[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const uint my = sg_sum[lane], pref = simd_prefix_exclusive_sum(my);
      if (pref + my >= need && pref < need) {
        const uint blk_sz = max(1u, nb >> 5);
        uint above = pref, b = nb - 1u - lane * blk_sz;
        for (uint j = 0u; j < blk_sz; ++j, --b) {
          const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
          if (above + c >= need || b == 0u) { found[0] = b; found[1] = need - above; found[2] = c; break; }
          above += c;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= found[0] << shift; mask |= bmask << shift; need = found[1];
    if (found[2] == need || shift == 0) break;
    top_bit = shift - 1;
  }
  uint my_gt = 0u, my_eq = 0u;
  for (uint c = 0u; c < n_chunks; ++c) {
    const uint i = (c << 10) + t;
    if (i < n) {
      const uint k = cached ? kv[c] : glm_order(sc[i]);
      if (k > prefix) ++my_gt;
      else if (k == prefix) ++my_eq;
    }
  }
  const uint gx0 = simd_prefix_exclusive_sum(my_gt), ex0 = simd_prefix_exclusive_sum(my_eq);
  if (lane == 31u) { sums[0][sg] = gx0 + my_gt; sums[1][sg] = ex0 + my_eq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const uint g0 = sums[0][lane], e0 = sums[1][lane], gp = simd_prefix_exclusive_sum(g0), ep = simd_prefix_exclusive_sum(e0);
    sums[0][lane] = gp; sums[1][lane] = ep;
    if (lane == 31u) { found[0] = gp + g0; found[1] = ep + e0; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (found[1] <= need) {
    uint pos_gt = sums[0][sg] + gx0, pos_eq = sums[1][sg] + ex0;
    const uint tot_gt = found[0];
    for (uint c = 0u; c < n_chunks; ++c) {
      const uint i = (c << 10) + t;
      if (i < n) {
        const uint k = cached ? kv[c] : glm_order(sc[i]);
        if (k > prefix) out[pos_gt++] = i; else if (k == prefix) out[tot_gt + pos_eq++] = i;
      }
    }
  } else {
    uint gt_seen = 0u, eq_seen = 0u;
    for (uint c = 0u; c < n_chunks; ++c) {
      const uint i = (c << 10) + t;
      const uint k = (i < n) ? (cached ? kv[c] : glm_order(sc[i])) : 0u;
      const uint gt = (i < n && k > prefix) ? 1u : 0u, eq = (i < n && k == prefix) ? 1u : 0u;
      const uint gx = simd_prefix_exclusive_sum(gt), ex = simd_prefix_exclusive_sum(eq);
      if (lane == 31u) { sums[0][sg] = gx + gt; sums[1][sg] = ex + eq; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        const uint g0 = sums[0][lane], e0 = sums[1][lane], gp = simd_prefix_exclusive_sum(g0), ep = simd_prefix_exclusive_sum(e0);
        sums[0][lane] = gp; sums[1][lane] = ep;
        if (lane == 31u) { found[0] = gp + g0; found[1] = ep + e0; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint gb = sums[0][sg], eb = sums[1][sg], eq_before = eq_seen + eb + ex;
      if (gt || (eq && eq_before < need)) out[gt_seen + gb + gx + min(eq_before, need)] = i;
      gt_seen += found[0]; eq_seen += found[1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (t == 0u) rows[r].nkeys = TOPK;
}
