// Inkling 2 multi-row prompt attention on NAX / simdgroup-matrix tiles (rows >= 64, head_dim == 128, d_rel == 16).
#include "../nax.h"

namespace i2_nax {

// Row max with NaN propagation (matches i2_max) over a fragment's 16 columns into r[2] (rows home.y, home.y + 8).
inline void row_max(thread const tfp::frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = i2_max(i2_max(f[b], f[b + 1]), i2_max(f[b + 2], f[b + 3]));
    const float u = simd_shuffle_xor(t, ushort(1));
    t = i2_max(t, u);
    const float w = simd_shuffle_xor(t, ushort(8));
    t = i2_max(t, w);
    r[h] = i2_max(r[h], t);
  }
}

// Row sum over a fragment's 16 columns into r[2] (rows home.y, home.y + 8).
inline void row_sum(thread const tfp::frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = t + w;
    r[h] = r[h] + t;
  }
}

// Vectorized 16x16 bfloat fragment load on M5 (two bfloat4 loads); falls back to frag_get/_t on TF_SIMD_FRAGS (M3).
template <bool TR>
inline tfp::frag<bfloat> load_frag(const device bfloat* p, int ld, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const device bfloat* q = p + (r + home.y) * ld + (c + home.x);
  return tfp::frag<bfloat>(*(const device bfloat4*)q, *(const device bfloat4*)(q + 8 * ld));
#else
  tfp::frag<bfloat> f;
  if (TR) {
    tfp::frag_get_t(f, p, ld, r, c, home);
  } else {
    tfp::frag_get(f, p, ld, r, c, home);
  }
  return f;
#endif
}

// Bounds- and ring-safe load of a 16x16 K block (transposed right operand of Q * K^T).
inline tfp::frag<bfloat> load_k_safe(const device bfloat* k_head, ulong kv_stride, constant I2AttnArgs& a, ulong kp,
                                     ulong lo_tg, ulong last_pos, int col0, short2 home) {
  tfp::frag<bfloat> f;
#ifndef TF_SIMD_FRAGS
  const ulong k0 = kp + ulong(home.y);
  const ulong k1 = k0 + 8ul;
  const ulong c = ulong(col0 + int(home.x));
  const bfloat4 v0 = (k0 >= lo_tg && k0 <= last_pos)
                         ? *(const device bfloat4*)(k_head + i2_attn_slot(a, k0) * kv_stride + c)
                         : bfloat4(bfloat(0.0f));
  const bfloat4 v1 = (k1 >= lo_tg && k1 <= last_pos)
                         ? *(const device bfloat4*)(k_head + i2_attn_slot(a, k1) * kv_stride + c)
                         : bfloat4(bfloat(0.0f));
  f = tfp::frag<bfloat>(v0, v1);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = int(home.x) + TF_COL(e);
    const int cc = col0 + int(home.y) + (e >> 2) * 8;
    const ulong key = kp + ulong(rr);
    if (key >= lo_tg && key <= last_pos) {
      f[e] = k_head[i2_attn_slot(a, key) * kv_stride + ulong(cc)];
    } else {
      f[e] = bfloat(0.0f);
    }
  }
#endif
  return f;
}

// Bounds- and ring-safe load of a 16x16 V block (non-transposed right operand of P * V).
inline tfp::frag<bfloat> load_v_safe(const device bfloat* v_head, ulong kv_stride, constant I2AttnArgs& a, ulong kp,
                                     ulong lo_tg, ulong last_pos, int col0, short2 home) {
  tfp::frag<bfloat> f;
#ifndef TF_SIMD_FRAGS
  const ulong k0 = kp + ulong(home.y);
  const ulong k1 = k0 + 8ul;
  const ulong c = ulong(col0 + int(home.x));
  const bfloat4 v0 = (k0 >= lo_tg && k0 <= last_pos)
                         ? *(const device bfloat4*)(v_head + i2_attn_slot(a, k0) * kv_stride + c)
                         : bfloat4(bfloat(0.0f));
  const bfloat4 v1 = (k1 >= lo_tg && k1 <= last_pos)
                         ? *(const device bfloat4*)(v_head + i2_attn_slot(a, k1) * kv_stride + c)
                         : bfloat4(bfloat(0.0f));
  f = tfp::frag<bfloat>(v0, v1);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = int(home.y) + (e >> 2) * 8;
    const int cc = col0 + int(home.x) + TF_COL(e);
    const ulong key = kp + ulong(rr);
    if (key >= lo_tg && key <= last_pos) {
      f[e] = v_head[i2_attn_slot(a, key) * kv_stride + ulong(cc)];
    } else {
      f[e] = bfloat(0.0f);
    }
  }
#endif
  return f;
}

}  // namespace i2_nax

// 64 query rows x 1 head per threadgroup (4 simdgroups x 16 query rows), BK = 32 keys per tile.
[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_attn_nax(
    const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],
    const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]],
    const device bfloat* kcache [[buffer(4)]], const device bfloat* vcache [[buffer(5)]],
    const device bfloat* proj [[buffer(6)]], device bfloat* out [[buffer(7)]],
    constant I2AttnArgs& a [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BQ = 64u;
  constexpr ulong BK = 32ul;
  constexpr uint PROJ_SPAN = 96u;    // >= BQ - 1 + BK - 1 + 1 = 95 distances per (BQ, BK) tile
  constexpr uint PROJ_STRIDE = 20u;  // 16 + 4 pad (40 bytes: 8-byte aligned for bfloat4, avoids TG bank conflicts)
  constexpr float kLog2e = 1.4426950408889634f;

  // Reverse Q-tile order so the longest causal tiles start first and short tiles fill the tail wave.
  const uint n_q_tiles = (a.rows + BQ - 1u) / BQ;
  const uint q0 = (n_q_tiles - 1u - tg.x) * BQ;
  const uint qh = tg.y;
  const uint group = a.heads / a.kv_heads;
  const uint kv_head = qh / group;

  threadgroup bfloat rel_tg[BQ * I2_DREL];
  threadgroup bfloat proj_tg[PROJ_SPAN][PROJ_STRIDE];

  // Vectorized cooperative load of this threadgroup's 64x16 relative states (2048 B).
  for (uint i4 = tid; i4 < (BQ * I2_DREL) / 4u; i4 += 128u) {
    const uint lr = i4 >> 2;
    const uint ld4 = (i4 & 3u) << 2;
    const uint gr = q0 + lr;
    const bfloat4 rv =
        (gr < a.rows) ? *(const device bfloat4*)(rel + (ulong(gr) * ulong(a.heads) + ulong(qh)) * I2_DREL + ld4)
                      : bfloat4(bfloat(0.0f));
    *(threadgroup bfloat4*)(rel_tg + i4 * 4u) = rv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint r_base = q0 + 16u * sg;
  const bool sg_active = (r_base < a.rows);
  const int q_rows = sg_active ? int(min(16u, a.rows - r_base)) : 0;
  const bool full_q = (r_base + 16u <= a.rows);
  const int q_stride = int(a.heads * I2_DIM);
  const int kv_stride = int(a.kv_heads * I2_DIM);
  const ulong kv_stride_u = ulong(kv_stride);

  const device bfloat* q_ptr = q + (sg_active ? ulong(r_base) : 0ul) * ulong(q_stride) + ulong(qh) * I2_DIM;
  const device bfloat* k_head = kcache + ulong(kv_head) * I2_DIM;
  const device bfloat* v_head = vcache + ulong(kv_head) * I2_DIM;

  const short2 home = tfp::frag_home(ushort(lane));

  // Preload Q into registers once per threadgroup tile (eliminates reloading Q on every K-block).
  tfp::frag<bfloat> qf[8];
  TF_UNROLL
  for (short d = 0; d < 8; d++) {
    if (full_q) {
      qf[d] = i2_nax::load_frag<false>(q_ptr, q_stride, 0, 16 * d, home);
    } else if (sg_active) {
      tfp::frag_get_in(qf[d], q_ptr, q_stride, 0, 16 * d, home, q_rows, int(I2_DIM));
    } else {
      qf[d] = tfp::frag<bfloat>(bfloat(0.0f));
    }
  }

  tfp::frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    acc[i] = tfp::frag<float>(0.0f);
  }

  float top[2], total[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const uint r = r_base + uint(home.y) + uint(h) * 8u;
    if (r < a.rows) {
      top[h] = float(sink[ulong(r) * ulong(a.heads) + ulong(qh)]);
      total[h] = (top[h] == -INFINITY) ? 0.0f : 1.0f;
    } else {
      top[h] = -INFINITY;
      total[h] = 0.0f;
    }
  }

  const ulong first_pos = ulong(a.pos0) + ulong(q0);
  const ulong last_pos = ulong(a.pos0) + ulong(min(q0 + BQ, a.rows) - 1u);
  const ulong lo_tg = i2_attn_lo(a, first_pos);
  const ulong kb_lo = lo_tg / BK;
  const ulong kb_hi = (last_pos + BK) / BK;

  const ulong sg_first_pos = ulong(a.pos0) + ulong(sg_active ? r_base : 0u);
  const ulong sg_last_pos = ulong(a.pos0) + ulong(sg_active ? min(r_base + 15u, a.rows - 1u) : 0u);
  const ulong sg_lo = i2_attn_lo(a, sg_first_pos);

  bool used_proj = false;
  for (ulong kb = kb_lo; kb < kb_hi; ++kb) {
    const ulong kp0 = kb * BK;
    const ulong kp1 = kp0 + 16ul;
    const ulong c_lo = kp0;
    const ulong c_hi = kp0 + 31ul;

    // Stage the [d_base .. d_base + 95] slice of proj transposed into TG memory only when this K-block overlaps extent.
    const long d_min_s = long(first_pos) - long(c_hi);
    const long d_max_s = long(last_pos) - long(c_lo);
    const bool tg_needs_bias = (d_max_s >= 0l) && (d_min_s < long(a.extent));
    const long d_base = max(0l, d_min_s);
    if (tg_needs_bias) {
      if (used_proj) threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < PROJ_SPAN) {
        const long d = d_base + long(tid);
        if (d < long(a.extent)) {
          const ulong du = ulong(d);
          const ulong ext = ulong(a.extent);
          TF_UNROLL
          for (uint i = 0; i < I2_DREL; ++i) {
            proj_tg[tid][i] = proj[ulong(i) * ext + du];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      used_proj = true;
    }

    // Skip K-blocks that are completely outside this simdgroup's causal/window range.
    if (!sg_active || c_lo > sg_last_pos || c_hi < sg_lo) {
      continue;
    }

    const ulong slot0 = i2_attn_slot(a, kp0);
    const ulong slot1 = i2_attn_slot(a, kp1);
    const bool fast_k0 = (kp0 >= lo_tg) && (kp0 + 15ul <= last_pos) && (a.ring == 0u || slot0 + 16ul <= ulong(a.ring));
    const bool fast_k1 = (kp1 >= lo_tg) && (kp1 + 15ul <= last_pos) && (a.ring == 0u || slot1 + 16ul <= ulong(a.ring));

    tfp::frag<float> s[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
    TF_UNROLL
    for (short d = 0; d < 8; d++) {
      const tfp::frag<bfloat> k0 =
          fast_k0 ? i2_nax::load_frag<true>(k_head + slot0 * kv_stride_u, kv_stride, 0, 16 * d, home)
                  : i2_nax::load_k_safe(k_head, kv_stride_u, a, kp0, lo_tg, last_pos, 16 * d, home);
      const tfp::frag<bfloat> k1 =
          fast_k1 ? i2_nax::load_frag<true>(k_head + slot1 * kv_stride_u, kv_stride, 0, 16 * d, home)
                  : i2_nax::load_k_safe(k_head, kv_stride_u, a, kp1, lo_tg, last_pos, 16 * d, home);
      tfp::mma_16x32<false, true>(s[0], s[1], qf[d], k0, k1);
    }

    const bool plain = full_q &&
                       (sg_first_pos >= c_hi + ulong(a.extent)) &&
                       (a.window == 0u || (sg_last_pos < c_lo + ulong(a.window)));
    if (plain) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          s[f][e] *= a.scale;
        }
      }
    } else if (!tg_needs_bias || sg_first_pos >= c_hi + ulong(a.extent)) {
      TF_UNROLL
      for (short h = 0; h < 2; h++) {
        const uint r = r_base + uint(home.y) + uint(h) * 8u;
        const ulong p = ulong(a.pos0) + ulong(r);
        const ulong lo = i2_attn_lo(a, p);
        const bool r_ok = (r < a.rows);
        TF_UNROLL
        for (short f = 0; f < 2; f++) {
          TF_UNROLL
          for (short c = 0; c < 4; c++) {
            const short e = 4 * h + c;
            const ulong key = c_lo + ulong(16 * f + int(home.x) + TF_COL(e));
            s[f][e] = (r_ok && key >= lo && key <= p) ? (s[f][e] * a.scale) : -INFINITY;
          }
        }
      }
    } else {
      const ulong d_base_u = ulong(d_base);
      TF_UNROLL
      for (short h = 0; h < 2; h++) {
        const uint r = r_base + uint(home.y) + uint(h) * 8u;
        if (r >= a.rows) {
          TF_UNROLL
          for (short f = 0; f < 2; f++) {
            TF_UNROLL
            for (short c = 0; c < 4; c++) s[f][4 * h + c] = -INFINITY;
          }
          continue;
        }
        const ulong p = ulong(a.pos0) + ulong(r);
        const ulong lo = i2_attn_lo(a, p);
        const threadgroup bfloat4* rp =
            (const threadgroup bfloat4*)(rel_tg + (16u * sg + uint(home.y) + uint(h) * 8u) * I2_DREL);
        const float4 r0 = float4(rp[0]);
        const float4 r1 = float4(rp[1]);
        const float4 r2 = float4(rp[2]);
        const float4 r3 = float4(rp[3]);
        TF_UNROLL
        for (short f = 0; f < 2; f++) {
          TF_UNROLL
          for (short c = 0; c < 4; c++) {
            const short e = 4 * h + c;
            const ulong key = c_lo + ulong(16 * f + int(home.x) + TF_COL(e));
            if (key >= lo && key <= p) {
              const ulong delta = p - key;
              if (delta < ulong(a.extent)) {
                const threadgroup bfloat4* pr = (const threadgroup bfloat4*)(proj_tg[uint(delta - d_base_u)]);
                float4 acc_b = float4(0.0f);
                acc_b = fma(r0, float4(pr[0]), acc_b);
                acc_b = fma(r1, float4(pr[1]), acc_b);
                acc_b = fma(r2, float4(pr[2]), acc_b);
                acc_b = fma(r3, float4(pr[3]), acc_b);
                const float bias_val = (acc_b.x + acc_b.y) + (acc_b.z + acc_b.w);
                s[f][e] = s[f][e] * a.scale + float(bfloat(bias_val));
              } else {
                s[f][e] = s[f][e] * a.scale;
              }
            } else {
              s[f][e] = -INFINITY;
            }
          }
        }
      }
    }

    float top_new[2] = {top[0], top[1]};
    i2_nax::row_max(s[0], top_new);
    i2_nax::row_max(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const float x = s[f][e];
        s[f][e] = (x == -INFINITY) ? 0.0f : metal::exp2((x - top_new[e >> 2]) * kLog2e);
      }
    }
    float scale_old[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      scale_old[h] = (top[h] == -INFINITY) ? 0.0f : metal::exp2((top[h] - top_new[h]) * kLog2e);
      top[h] = top_new[h];
      total[h] = total[h] * scale_old[h];
    }
    i2_nax::row_sum(s[0], total);
    i2_nax::row_sum(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[i][e] = acc[i][e] * scale_old[e >> 2];
      }
    }

    // Cast P to bfloat for full-rate BF16 x BF16 -> FP32 NAX tensor-unit P @ V.
    tfp::frag<bfloat> p_bf[2];
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        p_bf[f][e] = bfloat(s[f][e]);
      }
    }

    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      const bool fast_v = (k == 0) ? fast_k0 : fast_k1;
      if (fast_v) {
        const device bfloat* vp = v_head + ((k == 0) ? slot0 : slot1) * kv_stride_u;
        TF_UNROLL
        for (short d = 0; d < 8; d += 2) {
          const tfp::frag<bfloat> v0 = i2_nax::load_frag<false>(vp, kv_stride, 0, 16 * d, home);
          const tfp::frag<bfloat> v1 = i2_nax::load_frag<false>(vp, kv_stride, 0, 16 * d + 16, home);
          tfp::mma_16x32<false, false>(acc[d], acc[d + 1], p_bf[k], v0, v1);
        }
      } else {
        const ulong kp = (k == 0) ? kp0 : kp1;
        TF_UNROLL
        for (short d = 0; d < 8; d += 2) {
          const tfp::frag<bfloat> v0 = i2_nax::load_v_safe(v_head, kv_stride_u, a, kp, lo_tg, last_pos, 16 * d, home);
          const tfp::frag<bfloat> v1 =
              i2_nax::load_v_safe(v_head, kv_stride_u, a, kp, lo_tg, last_pos, 16 * d + 16, home);
          tfp::mma_16x32<false, false>(acc[d], acc[d + 1], p_bf[k], v0, v1);
        }
      }
    }
  }

  const ulong out_stride = ulong(a.heads) * I2_DIM;
  const ulong head_off = ulong(qh) * I2_DIM;
#ifndef TF_SIMD_FRAGS
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const uint r = r_base + uint(home.y) + uint(h) * 8u;
    if (r < a.rows) {
      const ulong row_base = ulong(r) * out_stride + head_off + ulong(home.x);
      const short b = 4 * h;
      TF_UNROLL
      for (short d = 0; d < 8; d++) {
        const ulong at = row_base + ulong(16 * d);
        const float4 g4 = float4(*(const device bfloat4*)(gate + at));
        *(device bfloat4*)(out + at) = bfloat4(
            i2_attn_finish(top[h], total[h], acc[d][b + 0], g4.x),
            i2_attn_finish(top[h], total[h], acc[d][b + 1], g4.y),
            i2_attn_finish(top[h], total[h], acc[d][b + 2], g4.z),
            i2_attn_finish(top[h], total[h], acc[d][b + 3], g4.w));
      }
    }
  }
#else
  TF_UNROLL
  for (short d = 0; d < 8; d++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const uint h = uint(e >> 2);
      const uint r = r_base + uint(home.y) + h * 8u;
      if (r < a.rows) {
        const uint col = uint(16 * d + int(home.x) + TF_COL(e));
        const ulong at = ulong(r) * out_stride + head_off + ulong(col);
        out[at] = i2_attn_finish(top[h], total[h], acc[d][e], float(gate[at]));
      }
    }
  }
#endif
}

// Fast decode attention: 1 (head, row)/TG, 4 SGs split keys, 128 threads, 2080 B TG mem, zero scratch, row-invariant.
[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_attn_decode_fast(
    const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],
    const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]],
    const device bfloat* kcache [[buffer(4)]], const device bfloat* vcache [[buffer(5)]],
    const device bfloat* proj [[buffer(6)]], device bfloat* out [[buffer(7)]],
    constant I2AttnArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint qh = tg.x;
  const uint r = tg.y;
  const uint group = a.heads / a.kv_heads;
  const uint kv_head = qh / group;

  const ulong p = ulong(a.pos0) + ulong(r);
  const ulong lo = i2_attn_lo(a, p);
  const ulong row_head = ulong(r) * ulong(a.heads) + ulong(qh);
  const ulong kv_row = ulong(a.kv_heads) * I2_DIM;

  const device bfloat* q_ptr = q + row_head * I2_DIM;
  const device bfloat* k_head = kcache + ulong(kv_head) * I2_DIM;
  const device bfloat* v_head = vcache + ulong(kv_head) * I2_DIM;

  // Full 128-dim Q stays in registers (4 elements per lane x 32 lanes).
  const float4 q4 = float4(*(const device bfloat4*)(q_ptr + lane * 4u));
  const float rel_lane = (lane < I2_DREL) ? float(rel[row_head * I2_DREL + lane]) : 0.0f;

  float m = -INFINITY;
  float l = 0.0f;
  float4 acc = float4(0.0f);
  if (sg == 0u) {
    m = float(sink[row_head]);
    l = (m == -INFINITY) ? 0.0f : 1.0f;
  }

  const ulong ext = ulong(a.extent);
  for (ulong base = lo + ulong(sg) * 4ul; base <= p; base += 16ul) {
    const uint nvalid = uint(min(4ul, p + 1ul - base));
    if (nvalid == 4u) {
      const ulong sl0 = i2_attn_slot(a, base + 0ul);
      const ulong sl1 = i2_attn_slot(a, base + 1ul);
      const ulong sl2 = i2_attn_slot(a, base + 2ul);
      const ulong sl3 = i2_attn_slot(a, base + 3ul);

      const float4 k0 = float4(*(const device bfloat4*)(k_head + sl0 * kv_row + lane * 4u));
      const float4 k1 = float4(*(const device bfloat4*)(k_head + sl1 * kv_row + lane * 4u));
      const float4 k2 = float4(*(const device bfloat4*)(k_head + sl2 * kv_row + lane * 4u));
      const float4 k3 = float4(*(const device bfloat4*)(k_head + sl3 * kv_row + lane * 4u));

      float4 p0 = q4 * k0;
      float4 p1 = q4 * k1;
      float4 p2 = q4 * k2;
      float4 p3 = q4 * k3;

      float sc[4];
      sc[0] = simd_sum((p0.x + p0.y) + (p0.z + p0.w));
      sc[1] = simd_sum((p1.x + p1.y) + (p1.z + p1.w));
      sc[2] = simd_sum((p2.x + p2.y) + (p2.z + p2.w));
      sc[3] = simd_sum((p3.x + p3.y) + (p3.z + p3.w));

      TF_UNROLL
      for (uint j = 0; j < 4u; ++j) {
        const ulong delta = p - (base + ulong(j));
        if (delta < ext) {
          const float b_lane = (lane < I2_DREL) ? (rel_lane * float(proj[ulong(lane) * ext + delta])) : 0.0f;
          const float bias_val = simd_sum(b_lane);
          sc[j] = sc[j] * a.scale + float(bfloat(bias_val));
        } else {
          sc[j] = sc[j] * a.scale;
        }
      }

      const float blk_max = i2_max(i2_max(sc[0], sc[1]), i2_max(sc[2], sc[3]));
      const float m_new = i2_max(m, blk_max);
      const float alpha = (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
      const float w0 = (sc[0] == -INFINITY) ? 0.0f : metal::precise::exp(sc[0] - m_new);
      const float w1 = (sc[1] == -INFINITY) ? 0.0f : metal::precise::exp(sc[1] - m_new);
      const float w2 = (sc[2] == -INFINITY) ? 0.0f : metal::precise::exp(sc[2] - m_new);
      const float w3 = (sc[3] == -INFINITY) ? 0.0f : metal::precise::exp(sc[3] - m_new);

      l = l * alpha + ((w0 + w1) + (w2 + w3));
      acc = acc * alpha;
      m = m_new;

      const float4 v0 = float4(*(const device bfloat4*)(v_head + sl0 * kv_row + lane * 4u));
      const float4 v1 = float4(*(const device bfloat4*)(v_head + sl1 * kv_row + lane * 4u));
      const float4 v2 = float4(*(const device bfloat4*)(v_head + sl2 * kv_row + lane * 4u));
      const float4 v3 = float4(*(const device bfloat4*)(v_head + sl3 * kv_row + lane * 4u));
      acc = fma(float4(w0), v0, acc);
      acc = fma(float4(w1), v1, acc);
      acc = fma(float4(w2), v2, acc);
      acc = fma(float4(w3), v3, acc);
    } else {
      for (uint j = 0; j < nvalid; ++j) {
        const ulong key = base + ulong(j);
        const ulong sl = i2_attn_slot(a, key);
        const float4 kv = float4(*(const device bfloat4*)(k_head + sl * kv_row + lane * 4u));
        const float4 prod = q4 * kv;
        float sc = simd_sum((prod.x + prod.y) + (prod.z + prod.w));
        const ulong delta = p - key;
        if (delta < ext) {
          const float b_lane = (lane < I2_DREL) ? (rel_lane * float(proj[ulong(lane) * ext + delta])) : 0.0f;
          const float bias_val = simd_sum(b_lane);
          sc = sc * a.scale + float(bfloat(bias_val));
        } else {
          sc = sc * a.scale;
        }
        const float m_new = i2_max(m, sc);
        const float alpha = (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
        const float w = (sc == -INFINITY) ? 0.0f : metal::precise::exp(sc - m_new);
        l = l * alpha + w;
        acc = acc * alpha;
        m = m_new;
        const float4 vv = float4(*(const device bfloat4*)(v_head + sl * kv_row + lane * 4u));
        acc = fma(float4(w), vv, acc);
      }
    }
  }

  threadgroup float mrg_m[4];
  threadgroup float mrg_l[4];
  threadgroup float mrg_acc[4][I2_DIM];

  if (lane == 0u) {
    mrg_m[sg] = m;
    mrg_l[sg] = l;
  }
  *(threadgroup float4*)(&mrg_acc[sg][lane * 4u]) = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint d = tid;
  const float M = i2_max(i2_max(mrg_m[0], mrg_m[1]), i2_max(mrg_m[2], mrg_m[3]));
  float L = 0.0f, A = 0.0f;
  TF_UNROLL
  for (uint s = 0; s < 4u; ++s) {
    const float ms = mrg_m[s];
    const float e = (ms == -INFINITY) ? 0.0f : metal::precise::exp(ms - M);
    L += mrg_l[s] * e;
    A += mrg_acc[s][d] * e;
  }
  const ulong at = row_head * I2_DIM + d;
  out[at] = i2_attn_finish(M, L, A, float(gate[at]));
}
