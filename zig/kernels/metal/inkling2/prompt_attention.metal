//! Inkling 2 Mini's prompt attention; attention_sources prepends the tensor-fragment helpers.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant constexpr uint D = 128, EXTENT = 1024, KEYS = 64, PAIRS = 128;
constant constexpr float LOG2E = 1.4426950408889634f;
struct InkPromptSeg { int row0, n, p0, pad; long off; };
struct InkPromptArgs { uint heads, kv_heads, ring, window, chunk, c0, chunks; float scale; };

inline size_t ink_prompt_rh(const thread InkPromptSeg& s, uint heads, uint group, uint kvh, uint pair) {
  return (size_t(s.row0) + pair / group) * heads + kvh * group + pair % group;
}

inline void ink_fold(thread float& mm, thread float& ll, float m, float l, thread float& a, thread float& b) {
  const float mn = max(mm, m);
  a = precise::exp2(mm - mn);
  b = precise::exp2(m - mn);
  ll = ll * a + l * b;
  mm = mn;
}
inline void ink_rows(thread const tfp::frag<float>* s, thread float (&r)[2], bool maximum) {
  for (uint f = 0; f < 4; ++f) for (uint h = 0; h < 2; ++h) {
    const uint b = 4 * h;
    float v = maximum ? max(max(s[f][b], s[f][b + 1]), max(s[f][b + 2], s[f][b + 3]))
                      : (s[f][b] + s[f][b + 1]) + (s[f][b + 2] + s[f][b + 3]);
    const float x = simd_shuffle_xor(v, ushort(1));
    v = maximum ? max(v, x) : v + x;
    const float y = simd_shuffle_xor(v, ushort(8));
    v = maximum ? max(v, y) : v + y;
    r[h] = maximum ? max(r[h], v) : r[h] + v;
  }
}

// 128 (row, head) pairs and one key head/chunk a 256-thread group; each simdgroup owns 16 pairs.
template <bool FINISH>
inline void ink2_attn_prompt_impl(const device bfloat* q, const device bfloat* kc, const device bfloat* vc,
    const device bfloat* tab, const device InkPromptSeg* segs, device float* part_o, device float2* part_ml,
    const device bfloat* sink, const device bfloat* gate, device bfloat* att_out,
    threadgroup bfloat* ks, threadgroup bfloat* vs, constant InkPromptArgs& a,
    uint3 tg, uint sg, uint lane) {
  const InkPromptSeg s = segs[0];
  const uint group = a.heads / a.kv_heads, total = uint(s.n) * group, m0 = tg.x * PAIRS;
  if (m0 >= total) return;
  const uint kvh = tg.y, c = a.c0 + tg.z;
  const uint tile_hi = uint(s.p0) + min(m0 + PAIRS - 1, total - 1) / group;
  const uint tile_lo_pos = uint(s.p0) + m0 / group;
  const uint b_lo = a.window ? (tile_lo_pos + 1 > a.window ? tile_lo_pos + 1 - a.window : 0) : c * a.chunk;
  if (b_lo > tile_hi) return;
  const uint k_hi = a.window ? tile_hi : min(b_lo + a.chunk - 1, tile_hi), m = m0 + 16 * sg;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint hx = uint(home.x), hy = uint(home.y);
  const uint row_h[2] = { m + hy, m + hy + 8u };
  const bool row_live[2] = { row_h[0] < total, row_h[1] < total };
  const uint pair_a = min(row_h[0], total - 1), pair_b = min(row_h[1], total - 1);
  const size_t prow_h[2] = { ink_prompt_rh(s, a.heads, group, kvh, pair_a),
                             ink_prompt_rh(s, a.heads, group, kvh, pair_b) };
  const uint pos_h[2] = { uint(s.p0) + pair_a / group, uint(s.p0) + pair_b / group };
  const uint win_lo_h[2] = {
    a.window && pos_h[0] + 1u > a.window ? pos_h[0] + 1u - a.window : 0u,
    a.window && pos_h[1] + 1u > a.window ? pos_h[1] + 1u - a.window : 0u,
  };
  tfp::frag<bfloat> qf[8];
  if (m < total) {
    const device bfloat* qb0 = q + prow_h[0] * D;
    const device bfloat* qb1 = q + prow_h[1] * D;
#ifndef TF_SIMD_FRAGS
    const bfloat4 z = bfloat4(bfloat(0.0f));
    TF_UNROLL
    for (uint d = 0; d < 8; ++d) {
      const uint col = 16u * d + hx;
      const bfloat4 q0 = row_live[0] ? *(const device bfloat4*)(qb0 + col) : z;
      const bfloat4 q1 = row_live[1] ? *(const device bfloat4*)(qb1 + col) : z;
      qf[d] = tfp::frag<bfloat>(q0, q1);
    }
#else
    TF_UNROLL
    for (uint d = 0; d < 8; ++d) {
      tfp::frag_get_rows(qf[d], qb0, qb1, 16 * d, home, row_live[0], row_live[1]);
    }
#endif
  }
  const uint sg_lo_pos = uint(s.p0) + m / group;
  const uint sg_lo = a.window && sg_lo_pos + 1 > a.window ? sg_lo_pos + 1 - a.window : 0;
  const uint sg_hi = uint(s.p0) + min(m + 15, total - 1) / group;
  const size_t hb = s.off + size_t(kvh) * a.ring * D;
  const uint col_load = lane * 4u;
  const uint col_off[4] = { hx + TF_COL(0), hx + TF_COL(1), hx + TF_COL(2), hx + TF_COL(3) };
  tfp::frag<float> acc[8], out[8];
  for (uint d = 0; d < 8; ++d) { acc[d] = tfp::frag<float>(0); out[d] = tfp::frag<float>(0); }
  float top[2] = {-INFINITY, -INFINITY}, mass[2] = {0.0f, 0.0f};
  float upper[2] = {-INFINITY, -INFINITY}, whole[2] = {0.0f, 0.0f};
  for (uint kt = b_lo; kt <= k_hi; kt += KEYS) {
    const uint tile_end = min(kt + KEYS, k_hi + 1);
    const bfloat4 zero4 = bfloat4(bfloat(0.0f));
    for (uint u = 0; u < 8; ++u) {
      const uint key = u * 8u + sg, kp = kt + key;
      const bool live = kp < tile_end;
      const size_t at = hb + size_t((live ? kp : kt) % a.ring) * D + col_load;
      *(threadgroup bfloat4*)(ks + key * D + col_load) = live ? *(const device bfloat4*)(kc + at) : zero4;
      *(threadgroup bfloat4*)(vs + key * D + col_load) = live ? *(const device bfloat4*)(vc + at) : zero4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!a.window && (kt - b_lo) % 1024 == 0) {
      for (uint d = 0; d < 8; ++d) acc[d] = tfp::frag<float>(0);
      top[0] = top[1] = -INFINITY; mass[0] = mass[1] = 0.0f;
    }
    tfp::frag<float> score[4] = {tfp::frag<float>(0), tfp::frag<float>(0), tfp::frag<float>(0), tfp::frag<float>(0)};
    if (m < total && kt <= sg_hi && (!a.window || kt + KEYS > sg_lo)) {
      TF_UNROLL
      for (uint d = 0; d < 8; ++d) {
        tfp::frag<bfloat> kb0, kb1, kb2, kb3;
#ifndef TF_SIMD_FRAGS
        tfp::frag_get_tg(kb0, ks, D, 0, 16 * d, home);
        tfp::frag_get_tg(kb1, ks, D, 16, 16 * d, home);
        tfp::frag_get_tg(kb2, ks, D, 32, 16 * d, home);
        tfp::frag_get_tg(kb3, ks, D, 48, 16 * d, home);
#else
        tfp::frag_get_t(kb0, ks, D, 0, 16 * d, home);
        tfp::frag_get_t(kb1, ks, D, 16, 16 * d, home);
        tfp::frag_get_t(kb2, ks, D, 32, 16 * d, home);
        tfp::frag_get_t(kb3, ks, D, 48, 16 * d, home);
#endif
        tfp::mma_16x32<false, true>(score[0], score[1], qf[d], kb0, kb1);
        tfp::mma_16x32<false, true>(score[2], score[3], qf[d], kb2, kb3);
      }
      float peak[2] = {top[0], top[1]};
      const uint sub_lo = b_lo + ((kt - b_lo) / 1024) * 1024, sub_hi = sub_lo + 1023;
      for (uint h = 0; h < 2; ++h) {
        const bool r_ok = row_live[h];
        const uint pos = pos_h[h], lo = a.window ? win_lo_h[h] : sub_lo, hi = a.window ? pos : min(pos, sub_hi);
        const size_t prow = prow_h[h];
        for (uint f = 0; f < 4; ++f) for (uint c_idx = 0; c_idx < 4; ++c_idx) {
          const uint e = 4u * h + c_idx, kp = kt + 16u * f + col_off[c_idx];
          const bool live = r_ok && kp >= lo && kp <= hi;
          const uint distance = live ? pos - kp : 0u;
          const float bias = live && distance < EXTENT ? float(tab[prow * EXTENT + distance]) : 0.0f;
          score[f][e] = live ? (score[f][e] * a.scale + bias) * LOG2E : -INFINITY;
        }
      }
      ink_rows(score, peak, true);
      float keep[2], mu_h[2];
      for (uint h = 0; h < 2; ++h) {
        const float mu = peak[h] == -INFINITY ? 0.0f : peak[h];
        mu_h[h] = mu;
        keep[h] = precise::exp2(top[h] - mu);
        top[h] = peak[h]; mass[h] *= keep[h];
      }
      tfp::frag<bfloat> p[4];
      for (uint f = 0; f < 4; ++f) for (uint e = 0; e < 8; ++e)
        p[f][e] = bfloat(precise::exp2(score[f][e] - mu_h[e >> 2]));
      // Reduce the rounded P, not its pre-bf16 exponential.
      tfp::frag<float> rounded[4];
      for (uint f = 0; f < 4; ++f) for (uint e = 0; e < 8; ++e) rounded[f][e] = float(p[f][e]);
      ink_rows(rounded, mass, false);
      for (uint d = 0; d < 8; ++d) for (uint e = 0; e < 8; ++e) acc[d][e] *= keep[e >> 2];
      for (uint f = 0; f < 4; ++f) for (uint d = 0; d < 8; d += 2) {
        tfp::frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
        tfp::frag_get_tg(v0, vs, D, 16 * f, 16 * d, home);
        tfp::frag_get_tg(v1, vs, D, 16 * f, 16 * d + 16, home);
#else
        tfp::frag_get(v0, vs, D, 16 * f, 16 * d, home);
        tfp::frag_get(v1, vs, D, 16 * f, 16 * d + 16, home);
#endif
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], p[f], v0, v1);
      }
    }
    const bool fold_now = (!a.window && (kt - b_lo) % 1024 == 960) || kt + KEYS > k_hi;
    if (fold_now && m < total) for (uint h = 0; h < 2; ++h) {
      if (top[h] == -INFINITY) {
        if (isnan(mass[h])) {
          upper[h] = whole[h] = NAN;
          for (uint d = 0; d < 8; ++d) for (uint e = 4 * h; e < 4 * h + 4; ++e) out[d][e] = NAN;
        }
        continue;
      }
      float x, y;
      ink_fold(upper[h], whole[h], top[h], mass[h], x, y);
      for (uint d = 0; d < 8; ++d) for (uint e = 4 * h; e < 4 * h + 4; ++e) out[d][e] = out[d][e] * x + acc[d][e] * y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // no simdgroup reads ks/vs when the next tile overwrites them
  }
  if (m >= total) return;
  if (FINISH) {
    if (a.window == 0) return;
    for (uint h = 0; h < 2; ++h) {
      if (!row_live[h] || b_lo > pos_h[h]) continue;
      const size_t prow = prow_h[h];
      const float sk = float(sink[prow]) * LOG2E;
      const float mf = max(upper[h], sk), a0 = precise::exp2(upper[h] - mf);
      const float den = whole[h] * a0 + precise::exp2(sk - mf);
#ifndef TF_SIMD_FRAGS
      const size_t base_col = prow * D + hx;
      const uint b = 4u * h;
      TF_UNROLL
      for (uint d = 0; d < 8; ++d) {
        const size_t at = base_col + 16u * d;
        const float4 gv = float4(*(const device bfloat4*)(gate + at));
        const bfloat4 res = bfloat4(
            bfloat((out[d][b] * a0 / den) * (1.0f / (1.0f + precise::exp(-gv.x)))),
            bfloat((out[d][b + 1] * a0 / den) * (1.0f / (1.0f + precise::exp(-gv.y)))),
            bfloat((out[d][b + 2] * a0 / den) * (1.0f / (1.0f + precise::exp(-gv.z)))),
            bfloat((out[d][b + 3] * a0 / den) * (1.0f / (1.0f + precise::exp(-gv.w)))));
        *(device bfloat4*)(att_out + at) = res;
      }
#else
      for (uint c_idx = 0; c_idx < 4; ++c_idx) {
        const uint e = 4u * h + c_idx;
        const uint col_in_16 = col_off[c_idx];
        for (uint d = 0; d < 8; ++d) {
          const float y = out[d][e] * a0 / den;
          const float gv = float(gate[prow * D + 16 * d + col_in_16]);
          att_out[prow * D + 16 * d + col_in_16] = bfloat(y * (1.0f / (1.0f + precise::exp(-gv))));
        }
      }
#endif
    }
    return;
  }
  for (uint h = 0; h < 2; ++h) {
    if (!row_live[h] || b_lo > pos_h[h]) continue;
    const size_t slot = prow_h[h] * a.chunks + c - a.c0;
#ifndef TF_SIMD_FRAGS
    const uint b = 4u * h;
    TF_UNROLL
    for (uint d = 0; d < 8; ++d) {
      *(device float4*)(part_o + slot * D + 16u * d + hx) =
          float4(out[d][b], out[d][b + 1], out[d][b + 2], out[d][b + 3]);
    }
#else
    for (uint c_idx = 0; c_idx < 4; ++c_idx) {
      const uint e = 4u * h + c_idx, col = col_off[c_idx];
      for (uint d = 0; d < 8; ++d) part_o[slot * D + 16 * d + col] = out[d][e];
    }
#endif
    if (hx == 0u) part_ml[slot] = float2(upper[h], whole[h]);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attn_prompt(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* tab [[buffer(3)]],
    const device InkPromptSeg* segs [[buffer(4)]], device float* part_o [[buffer(5)]], device float2* part_ml [[buffer(6)]],
    constant InkPromptArgs& a [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  alignas(16) threadgroup bfloat ks[KEYS * D], vs[KEYS * D];
  ink2_attn_prompt_impl<false>(q, kc, vc, tab, segs, part_o, part_ml, nullptr, nullptr, nullptr, ks, vs, a, tg, sg, lane);
}

// Sliding-window prompt attention with sink-last merge and gate fused into the final BF16 store.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attn_prompt_sliding(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], const device bfloat* tab [[buffer(3)]],
    const device InkPromptSeg* segs [[buffer(4)]], const device bfloat* sink [[buffer(5)]],
    const device bfloat* gate [[buffer(6)]], device bfloat* out [[buffer(7)]],
    constant InkPromptArgs& a [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  alignas(16) threadgroup bfloat ks[KEYS * D], vs[KEYS * D];
  ink2_attn_prompt_impl<true>(q, kc, vc, tab, segs, nullptr, nullptr, sink, gate, out, ks, vs, a, tg, sg, lane);
}
