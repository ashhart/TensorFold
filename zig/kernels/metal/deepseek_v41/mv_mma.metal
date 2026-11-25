// DeepSeek V4.1 prompt matvecs on simdgroup MMA: bit-equal to mv.metal (32 K%32 chains, then 5-step tree).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define MV_UNR _Pragma("clang loop unroll(full)")

struct TfDsv41MvMmaArgs {
  uint k, n, rows;
};

inline float tf_dsv41_mv_mma_x(uint bits) {
  const float f = as_type<float>(bits);
  return (TF_CTL == 1 && !isfinite(f)) ? 0.0f : f;
}

inline float tf_dsv41_mv_mma_w(uint bits) {
  return as_type<float>(bits);
}

// 16x16 tile (4 simdgroups of 8x8) over 256-K chunks (8 steps x 32 K%32 chains), bank-conflict-free.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_mv_mma(
    const device ushort* X [[buffer(0)]], const device ushort* Wt [[buffer(1)]], device T* Y [[buffer(2)]],
    constant TfDsv41MvMmaArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup ushort tg_x[32][2][8][8];
  threadgroup ushort tg_w[32][2][8][8];

  const uint n0 = tg.x * 16u;
  const uint r0 = tg.y * 16u;
  const uint rt = sg >> 1u;
  const uint ct = sg & 1u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

  simdgroup_matrix<float, 8, 8> acc[32];
  MV_UNR for (uint c = 0; c < 32u; ++c) acc[c] = simdgroup_matrix<float, 8, 8>(0.0f);

  const uint k_lim = (TF_CTL == 2) ? (a.k & ~31u) : a.k;
  const bool vec4 = ((a.k & 3u) == 0u) && (((ulong(X) | ulong(Wt)) & 7u) == 0u);

  uint k0 = 0u;
  for (; k0 + 256u <= k_lim; k0 += 256u) {
    if (vec4) {
      for (uint q = 0; q < 8u; ++q) {
        const uint idx = q * 128u + tid;
        const uint m_full = idx >> 6u;
        const uint k_g = (idx & 63u) << 2u;
        const uint p = k_g >> 5u;
        const uint c_base = k_g & 31u;

        const uint row = r0 + m_full;
        const ushort4 xv = (row < a.rows) ? *(const device ushort4*)(X + ulong(row) * a.k + k0 + k_g) : ushort4(0);
        const uint mr = m_full >> 3u, m = (m_full & 7u) ^ (c_base >> 2u);
        tg_x[c_base][mr][m][p] = xv.x;
        tg_x[c_base + 1u][mr][m][p] = xv.y;
        tg_x[c_base + 2u][mr][m][p] = xv.z;
        tg_x[c_base + 3u][mr][m][p] = xv.w;

        const uint col = n0 + m_full;
        const ushort4 wv = (col < a.n) ? *(const device ushort4*)(Wt + ulong(col) * a.k + k0 + k_g) : ushort4(0);
        const uint p_sw = p ^ (c_base >> 2u);
        tg_w[c_base][mr][p_sw][m_full & 7u] = wv.x;
        tg_w[c_base + 1u][mr][p_sw][m_full & 7u] = wv.y;
        tg_w[c_base + 2u][mr][p_sw][m_full & 7u] = wv.z;
        tg_w[c_base + 3u][mr][p_sw][m_full & 7u] = wv.w;
      }
    } else {
      for (uint u = 0; u < 32u; ++u) {
        const uint idx = u * 128u + tid;
        const uint m_full = idx >> 8u;
        const uint k_off = idx & 255u;
        const uint p = k_off >> 5u;
        const uint c = k_off & 31u;
        const uint mr = m_full >> 3u, m = m_full & 7u;

        const uint row = r0 + m_full;
        tg_x[c][mr][m ^ (c >> 2u)][p] = (row < a.rows) ? X[ulong(row) * a.k + k0 + k_off] : 0u;
        const uint col = n0 + m_full;
        tg_w[c][mr][p ^ (c >> 2u)][m] = (col < a.n) ? Wt[ulong(col) * a.k + k0 + k_off] : 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    MV_UNR for (uint c = 0; c < 32u; ++c) {
      simdgroup_matrix<float, 8, 8> xa, wb;
      const uint csw = c >> 2u;
      const uint xp = *(const threadgroup uint*)(&tg_x[c][rt][fm ^ csw][fn]);
      const uint wp = *(const threadgroup uint*)(&tg_w[c][ct][fm ^ csw][fn]);
      xa.thread_elements()[0] = tf_dsv41_mv_mma_x(xp << 16u);
      xa.thread_elements()[1] = tf_dsv41_mv_mma_x(xp & 0xFFFF0000u);
      wb.thread_elements()[0] = tf_dsv41_mv_mma_w(wp << 16u);
      wb.thread_elements()[1] = tf_dsv41_mv_mma_w(wp & 0xFFFF0000u);
      simdgroup_multiply_accumulate(acc[c], xa, wb, acc[c]);
#if TF_CTL == 3
      acc[c].thread_elements()[0] = float(half(acc[c].thread_elements()[0]));
      acc[c].thread_elements()[1] = float(half(acc[c].thread_elements()[1]));
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (k0 < k_lim) {
    for (uint u = 0; u < 32u; ++u) {
      const uint idx = u * 128u + tid;
      const uint m_full = idx >> 8u;
      const uint k_off = idx & 255u;
      const uint p = k_off >> 5u;
      const uint c = k_off & 31u;
      const uint mr = m_full >> 3u, m = m_full & 7u;
      const bool k_ok = (k0 + k_off) < k_lim;

      const uint row = r0 + m_full;
      tg_x[c][mr][m][p] = (k_ok && row < a.rows) ? X[ulong(row) * a.k + k0 + k_off] : 0u;
      const uint col = n0 + m_full;
      tg_w[c][mr][p][m] = (k_ok && col < a.n) ? Wt[ulong(col) * a.k + k0 + k_off] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    MV_UNR for (uint c = 0; c < 32u; ++c) {
      const uint cnt = (k_lim > k0 + c) ? min(8u, (k_lim - 1u - (k0 + c)) / 32u + 1u) : 0u;
      if (cnt == 8u) {
        simdgroup_matrix<float, 8, 8> xa, wb;
        const uint xp = *(const threadgroup uint*)(&tg_x[c][rt][fm][fn]);
        const uint wp = *(const threadgroup uint*)(&tg_w[c][ct][fm][fn]);
        xa.thread_elements()[0] = tf_dsv41_mv_mma_x(xp << 16u);
        xa.thread_elements()[1] = tf_dsv41_mv_mma_x(xp & 0xFFFF0000u);
        wb.thread_elements()[0] = tf_dsv41_mv_mma_w(wp << 16u);
        wb.thread_elements()[1] = tf_dsv41_mv_mma_w(wp & 0xFFFF0000u);
        simdgroup_multiply_accumulate(acc[c], xa, wb, acc[c]);
#if TF_CTL == 3
        acc[c].thread_elements()[0] = float(half(acc[c].thread_elements()[0]));
        acc[c].thread_elements()[1] = float(half(acc[c].thread_elements()[1]));
#endif
      } else if (cnt > 0u) {
        float a0 = acc[c].thread_elements()[0];
        float a1 = acc[c].thread_elements()[1];
        for (uint p = 0; p < cnt; ++p) {
          const float xv = tf_dsv41_mv_mma_x(uint(tg_x[c][rt][fm][p]) << 16u);
          const float w0 = tf_dsv41_mv_mma_w(uint(tg_w[c][ct][p][fn]) << 16u);
          const float w1 = tf_dsv41_mv_mma_w(uint(tg_w[c][ct][p][fn + 1u]) << 16u);
          a0 = fma(xv, w0, a0);
          a1 = fma(xv, w1, a1);
#if TF_CTL == 3
          a0 = float(half(a0));
          a1 = float(half(a1));
#endif
        }
        acc[c].thread_elements()[0] = a0;
        acc[c].thread_elements()[1] = a1;
      }
    }
  }

  const uint row = r0 + rt * 8u + fm;
  const uint col0 = n0 + ct * 8u + fn;
  if (row < a.rows) {
    MV_UNR for (uint e = 0; e < 2u; ++e) {
      const uint col = col0 + e;
      if (col < a.n) {
        float v[32];
        MV_UNR for (uint c = 0; c < 32u; ++c) v[c] = acc[c].thread_elements()[e];
        MV_UNR for (uint s = 1u; s < 32u; s *= 2u)
          MV_UNR for (uint c = 0; c < 32u; c += 2u * s) v[c] = v[c] + v[c + s];
        Y[ulong(row) * a.n + col] = T(v[0]);
      }
    }
  }
  if (TF_CTL == 4 && tid == 0u && tg.x == 0u && tg.y == 0u) Y[ulong(a.rows) * a.n] = T(1.0f);
}

template [[host_name("tf_dsv41_mv_mma_f32")]] [[kernel]] decltype(tf_dsv41_mv_mma<float>) tf_dsv41_mv_mma<float>;
template [[host_name("tf_dsv41_mv_mma_bf16")]] [[kernel]] decltype(tf_dsv41_mv_mma<bfloat>) tf_dsv41_mv_mma<bfloat>;
