#include <metal_stdlib>
#pragma METAL fp contract(off)
#include <metal_simdgroup_matrix>
#ifndef TF_SIMD_FRAGS
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

struct DraftHeadQ4Args {
  uint rows;
  uint vocab;
  uint k;
  uint softcap;
};

struct Dflash2Bf16Args {
  uint rows;
  uint n;
  uint k;
  uint ldy;
};

static inline bfloat df2_fast_round(float v) { return bfloat(v); }

static inline uint df2_q4_pair(uint byte_val) {
  const uint q0 = byte_val & 0xfu;
  const uint q1 = (byte_val >> 4u) & 0xfu;
  const uint bf0 = as_type<ushort>(bfloat(q0));
  const uint bf1 = as_type<ushort>(bfloat(q1));
  return bf0 | (bf1 << 16u);
}

static inline float df2_q4_delta(float dot_q, float xsum, float s, float b,
                                 const device bfloat* y, int K, int k0, int row_idx,
                                 const threadgroup bfloat* tile_col) {
  const float term = s * dot_q + b * xsum;
  if (metal::isfinite(term)) return term;
  float exact = 0.0f;
  const device bfloat* yr = y + size_t(row_idx) * K + k0;
  for (int i = 0; i < 64; ++i) {
    exact += float(yr[i]) * (s * float(tile_col[i]) + b);
  }
  return exact;
}

#ifndef TF_SIMD_FRAGS
namespace tfp {
template <typename T>
using frag = vec<T, 8>;

inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

template <typename T, typename P>
inline void frag_get_vec(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  const bfloat4 v0 = *(const device bfloat4*)(q);
  const bfloat4 v1 = *(const device bfloat4*)(q + 8 * ld);
  f = frag<T>(v0, v1);
}

inline void frag_get_rows_vec(thread frag<bfloat>& f, const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
  const bfloat4 v0 = *(const device bfloat4*)(r0 + c + home.x);
  const bfloat4 v1 = *(const device bfloat4*)(r1 + c + home.x);
  f = frag<bfloat>(v0, v1);
}

template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

inline void frag_get_tg(thread frag<bfloat>& f, const threadgroup bfloat* p, int ld, int r, int c, short2 home) {
  const threadgroup bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<bfloat>(*(const threadgroup bfloat4*)q, *(const threadgroup bfloat4*)(q + 8 * ld));
}

inline void frag_put(thread const frag<float>& f, device bfloat* p, int ld, int r, int c, short2 home) {
  device bfloat* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + (e & 3)] = df2_fast_round(f[e]);
  }
}

inline void frag_put_in(thread const frag<float>& f, device bfloat* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = df2_fast_round(f[e]);
    }
  }
}

template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}
}  // namespace tfp

inline void df2_bf16_nax_tile(const device bfloat* A, const device bfloat* B, constant Dflash2Bf16Args& a,
                              device bfloat* D, int row0, int col0, uint lane) {
  if (row0 >= int(a.rows) || col0 >= int(a.n)) return;
  const int M = int(a.rows), N = int(a.n), K = int(a.k), ldd = int(a.ldy);
  const int rows = min(16, M - row0), cols = min(32, N - col0);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc0(0.0f), acc1(0.0f);
  const device bfloat* r0 = A + long(min(row0 + int(home.y), M - 1)) * K;
  const device bfloat* r1 = A + long(min(row0 + int(home.y) + 8, M - 1)) * K;
  const device bfloat* Bp = B + long(col0) * K;
  const bool full_n = (cols == 32);
  const int K64 = K & ~63;
  for (int kk = 0; kk < K64; kk += 64) {
    TF_UNROLL for (int u = 0; u < 2; ++u) {
      tfp::frag<bfloat> fa0, fa1, fb00, fb01, fb10, fb11;
      tfp::frag_get_rows_vec(fa0, r0, r1, kk + u * 32, home);
      tfp::frag_get_rows_vec(fa1, r0, r1, kk + u * 32 + 16, home);
      if (full_n) {
        tfp::frag_get_vec(fb00, Bp, K, 0, kk + u * 32, home);
        tfp::frag_get_vec(fb01, Bp, K, 16, kk + u * 32, home);
        tfp::frag_get_vec(fb10, Bp, K, 0, kk + u * 32 + 16, home);
        tfp::frag_get_vec(fb11, Bp, K, 16, kk + u * 32 + 16, home);
      } else {
        tfp::frag_get_in(fb00, Bp, K, 0, kk + u * 32, home, cols, K);
        tfp::frag_get_in(fb01, Bp, K, 16, kk + u * 32, home, cols, K);
        tfp::frag_get_in(fb10, Bp, K, 0, kk + u * 32 + 16, home, cols, K);
        tfp::frag_get_in(fb11, Bp, K, 16, kk + u * 32 + 16, home, cols, K);
      }
      tfp::mma_16x32<false, true>(acc0, acc1, fa0, fb00, fb01);
      tfp::mma_16x32<false, true>(acc0, acc1, fa1, fb10, fb11);
    }
  }
  if (K & 32) {
    tfp::frag<bfloat> fa0, fa1, fb00, fb01, fb10, fb11;
    tfp::frag_get_rows_vec(fa0, r0, r1, K64, home);
    tfp::frag_get_rows_vec(fa1, r0, r1, K64 + 16, home);
    if (full_n) {
      tfp::frag_get_vec(fb00, Bp, K, 0, K64, home);
      tfp::frag_get_vec(fb01, Bp, K, 16, K64, home);
      tfp::frag_get_vec(fb10, Bp, K, 0, K64 + 16, home);
      tfp::frag_get_vec(fb11, Bp, K, 16, K64 + 16, home);
    } else {
      tfp::frag_get_in(fb00, Bp, K, 0, K64, home, cols, K);
      tfp::frag_get_in(fb01, Bp, K, 16, K64, home, cols, K);
      tfp::frag_get_in(fb10, Bp, K, 0, K64 + 16, home, cols, K);
      tfp::frag_get_in(fb11, Bp, K, 16, K64 + 16, home, cols, K);
    }
    tfp::mma_16x32<false, true>(acc0, acc1, fa0, fb00, fb01);
    tfp::mma_16x32<false, true>(acc0, acc1, fa1, fb10, fb11);
  }
  device bfloat* Dp = D + long(row0) * ldd + col0;
  if (rows == 16 && full_n) {
    tfp::frag_put(acc0, Dp, ldd, 0, 0, home);
    tfp::frag_put(acc1, Dp, ldd, 0, 16, home);
  } else {
    tfp::frag_put_in(acc0, Dp, ldd, 0, 0, home, rows, cols);
    tfp::frag_put_in(acc1, Dp, ldd, 0, 16, home, rows, cols);
  }
}
#else
inline void df2_bf16_nax_tile(const device bfloat* A, const device bfloat* B, constant Dflash2Bf16Args& a,
                              device bfloat* D, int row0, int col0, uint lane) {
  if (row0 >= int(a.rows) || col0 >= int(a.n)) return;
  const int M = int(a.rows), N = int(a.n), K = int(a.k), ldd = int(a.ldy);
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  simdgroup_matrix<float, 8, 8> acc[2][4];
  TF_UNROLL for (int i = 0; i < 2; ++i)
    TF_UNROLL for (int j = 0; j < 4; ++j)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (int kk = 0; kk < K; kk += 8) {
    simdgroup_matrix<bfloat, 8, 8> fa[2], fb[4];
    TF_UNROLL for (int i = 0; i < 2; ++i) {
      if (row0 + 8 * i + 8 <= M) {
        simdgroup_load(fa[i], A + size_t(row0 + 8 * i) * K + kk, K);
      } else {
        const device bfloat* p = A + size_t(min(row0 + 8 * i + int(fm), M - 1)) * K + kk + fn;
        fa[i].thread_elements()[0] = p[0];
        fa[i].thread_elements()[1] = p[1];
      }
    }
    TF_UNROLL for (int j = 0; j < 4; ++j) {
      if (col0 + 8 * j + 8 <= N) {
        simdgroup_load(fb[j], B + size_t(col0 + 8 * j) * K + kk, K, ulong2(0, 0), true);
      } else {
        const int c0 = col0 + 8 * j + int(fn);
        const int rk = kk + int(fm);
        fb[j].thread_elements()[0] = (c0 < N) ? B[size_t(c0) * K + rk] : bfloat(0.0f);
        fb[j].thread_elements()[1] = (c0 + 1 < N) ? B[size_t(c0 + 1) * K + rk] : bfloat(0.0f);
      }
    }
    TF_UNROLL for (int i = 0; i < 2; ++i)
      TF_UNROLL for (int j = 0; j < 4; ++j)
        simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
  }
  TF_UNROLL for (int i = 0; i < 2; ++i) {
    const int r = row0 + 8 * i + int(fm);
    if (r < M) {
      TF_UNROLL for (int j = 0; j < 4; ++j) {
        const int c = col0 + 8 * j + int(fn);
        if (c < N) D[size_t(r) * ldd + c] = df2_fast_round(acc[i][j].thread_elements()[0]);
        if (c + 1 < N) D[size_t(r) * ldd + c + 1] = df2_fast_round(acc[i][j].thread_elements()[1]);
      }
    }
  }
}
#endif

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_df2_bf16_nax_16(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant Dflash2Bf16Args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (a.rows == 0u || a.n == 0u || a.k == 0u) return;
  df2_bf16_nax_tile(X, W, a, Y, int(tg.y) * 16, int(tg.x) * 32, lane);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_df2_bf16_nax_32(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant Dflash2Bf16Args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (a.rows == 0u || a.n == 0u || a.k == 0u) return;
  df2_bf16_nax_tile(X, W, a, Y, int(tg.y) * 32 + 16 * int(sg / 2u), int(tg.x) * 64 + 32 * int(sg % 2u), lane);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_draft_head_q4_logits(const device uint* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]], const device bfloat* y [[buffer(3)]], device bfloat* out [[buffer(4)]],
    constant DraftHeadQ4Args& a [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BM = 32, BN = 64, THREADS = 128, PAD = 72;
  threadgroup uint lut[256];
  threadgroup bfloat tile[BN][PAD];
  threadgroup float xsum[BM];
  threadgroup float sh_s[BN];
  threadgroup float sh_b[BN];
  for (int i = int(tid); i < 256; i += THREADS) lut[i] = df2_q4_pair(uint(i));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (a.rows == 0u || a.vocab == 0u || a.k == 0u) return;
  const int row0 = int(tg.y) * BM, col0 = int(tg.x) * BN;
  if (row0 >= int(a.rows) || col0 >= int(a.vocab)) return;
  const int M = int(a.rows), N = int(a.vocab), K = int(a.k);
  const int rows = min(BM, M - row0), cols = min(BN, N - col0);
  const int words_per_row = K >> 3, groups_per_row = K >> 6;
  const int sg_r = int(sg / 2u) * 16, sg_c = int(sg % 2u) * 32;
  const int my_r = row0 + sg_r, my_c = col0 + sg_c;
  const int my_rows = max(0, min(16, M - my_r)), my_cols = max(0, min(32, N - my_c));
  const int xr_idx = int(tid >> 2u), x_lane = int(tid & 3u);

#ifndef TF_SIMD_FRAGS
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc0(0.0f), acc1(0.0f);
  const device bfloat* r0 = y + size_t(min(my_r + int(home.y), M - 1)) * K;
  const device bfloat* r1 = y + size_t(min(my_r + int(home.y) + 8, M - 1)) * K;
#else
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  simdgroup_matrix<float, 8, 8> acc[2][4];
  TF_UNROLL for (int i = 0; i < 2; ++i)
    TF_UNROLL for (int j = 0; j < 4; ++j)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
#endif

  for (int g = 0; g < groups_per_row; ++g) {
    const int k0 = g * 64;
    for (int idx = int(tid); idx < BN * 2; idx += THREADS) {
      const int wr = idx >> 1, wh = idx & 1;
      threadgroup uint4* dst = (threadgroup uint4*)(&tile[wr][wh * 32]);
      if (wr < cols) {
        const uint4 wv = ((const device uint4*)(w + size_t(col0 + wr) * words_per_row + g * 8))[wh];
        TF_UNROLL for (int i = 0; i < 4; ++i) {
          const uint word = wv[i];
          dst[i] = uint4(lut[word & 0xffu], lut[(word >> 8u) & 0xffu], lut[(word >> 16u) & 0xffu], lut[word >> 24u]);
        }
      } else {
        TF_UNROLL for (int i = 0; i < 4; ++i) dst[i] = uint4(0u);
      }
    }
    if (int(tid) < BN) {
      if (int(tid) < cols) {
        sh_s[tid] = float(scales[size_t(col0 + int(tid)) * groups_per_row + g]);
        sh_b[tid] = float(biases[size_t(col0 + int(tid)) * groups_per_row + g]);
      } else {
        sh_s[tid] = 0.0f;
        sh_b[tid] = 0.0f;
      }
    }
    if (xr_idx < rows) {
      const device bfloat4* xr = (const device bfloat4*)(y + size_t(row0 + xr_idx) * K + k0 + x_lane * 16);
      float ps = 0.0f;
      TF_UNROLL for (int i = 0; i < 4; ++i) {
        const float4 v = float4(xr[i]);
        ps += v.x + v.y + v.z + v.w;
      }
      ps += simd_shuffle_xor(ps, ushort(1));
      ps += simd_shuffle_xor(ps, ushort(2));
      if (x_lane == 0) xsum[xr_idx] = ps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (my_rows > 0 && my_cols > 0) {
#ifndef TF_SIMD_FRAGS
      tfp::frag<float> p0(0.0f), p1(0.0f);
      TF_UNROLL for (int q = 0; q < 4; ++q) {
        tfp::frag<bfloat> fa, fb0, fb1;
        tfp::frag_get_rows_vec(fa, r0, r1, k0 + 16 * q, home);
        tfp::frag_get_tg(fb0, &tile[sg_c][0], PAD, 0, 16 * q, home);
        tfp::frag_get_tg(fb1, &tile[sg_c + 16][0], PAD, 0, 16 * q, home);
        tfp::mma_16x32<false, true>(p0, p1, fa, fb0, fb1);
      }
      const int r0_idx = my_r + int(home.y);
      const int r1_idx = r0_idx + 8;
      const float xs0 = (sg_r + int(home.y) < rows) ? xsum[sg_r + home.y] : 0.0f;
      const float xs1 = (sg_r + int(home.y) + 8 < rows) ? xsum[sg_r + home.y + 8] : 0.0f;
      TF_UNROLL for (short e = 0; e < 4; ++e) {
        const int c0_local = sg_c + home.x + e;
        const int c1_local = c0_local + 16;
        const int c0_global = my_c + home.x + e;
        const int c1_global = c0_global + 16;
        if (r0_idx < M && c0_global < N) {
          acc0[e] += df2_q4_delta(p0[e], xs0, sh_s[c0_local], sh_b[c0_local], y, K, k0, r0_idx, &tile[c0_local][0]);
        }
        if (r1_idx < M && c0_global < N) {
          acc0[4 + e] += df2_q4_delta(p0[4 + e], xs1, sh_s[c0_local], sh_b[c0_local], y, K, k0, r1_idx, &tile[c0_local][0]);
        }
        if (r0_idx < M && c1_global < N) {
          acc1[e] += df2_q4_delta(p1[e], xs0, sh_s[c1_local], sh_b[c1_local], y, K, k0, r0_idx, &tile[c1_local][0]);
        }
        if (r1_idx < M && c1_global < N) {
          acc1[4 + e] += df2_q4_delta(p1[4 + e], xs1, sh_s[c1_local], sh_b[c1_local], y, K, k0, r1_idx, &tile[c1_local][0]);
        }
      }
#else
      simdgroup_matrix<float, 8, 8> p[2][4];
      TF_UNROLL for (int i = 0; i < 2; ++i)
        TF_UNROLL for (int j = 0; j < 4; ++j)
          p[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      TF_UNROLL for (int q = 0; q < 8; ++q) {
        simdgroup_matrix<bfloat, 8, 8> fa[2], fb[4];
        TF_UNROLL for (int i = 0; i < 2; ++i) {
          if (my_r + 8 * i + 8 <= M) {
            simdgroup_load(fa[i], y + size_t(my_r + 8 * i) * K + k0 + 8 * q, K);
          } else {
            const device bfloat* pr = y + size_t(min(my_r + 8 * i + int(fm), M - 1)) * K + k0 + 8 * q + fn;
            fa[i].thread_elements()[0] = pr[0];
            fa[i].thread_elements()[1] = pr[1];
          }
        }
        TF_UNROLL for (int j = 0; j < 4; ++j) {
          simdgroup_load(fb[j], &tile[sg_c + 8 * j][8 * q], PAD, ulong2(0, 0), true);
        }
        TF_UNROLL for (int i = 0; i < 2; ++i)
          TF_UNROLL for (int j = 0; j < 4; ++j)
            simdgroup_multiply_accumulate(p[i][j], fa[i], fb[j], p[i][j]);
      }
      TF_UNROLL for (int i = 0; i < 2; ++i) {
        const int r_local = sg_r + 8 * i + int(fm);
        const int r_global = row0 + r_local;
        const float xs = (r_local < rows) ? xsum[r_local] : 0.0f;
        TF_UNROLL for (int j = 0; j < 4; ++j) {
          TF_UNROLL for (int e = 0; e < 2; ++e) {
            const int c_local = sg_c + 8 * j + int(fn) + e;
            const int c_global = col0 + c_local;
            if (r_global < M && c_global < N) {
              acc[i][j].thread_elements()[e] += df2_q4_delta(p[i][j].thread_elements()[e], xs, sh_s[c_local],
                                                             sh_b[c_local], y, K, k0, r_global, &tile[c_local][0]);
            }
          }
        }
      }
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (my_rows > 0 && my_cols > 0) {
#ifndef TF_SIMD_FRAGS
    device bfloat* Dp = out + size_t(my_r) * N + my_c;
    if (a.softcap) {
      TF_UNROLL for (short e = 0; e < 8; ++e) {
        acc0[e] = 30.0f * precise::tanh(precise::divide(float(bfloat(acc0[e])), 30.0f));
        acc1[e] = 30.0f * precise::tanh(precise::divide(float(bfloat(acc1[e])), 30.0f));
      }
    }
    tfp::frag_put_in(acc0, Dp, N, 0, 0, home, my_rows, my_cols);
    tfp::frag_put_in(acc1, Dp, N, 0, 16, home, my_rows, my_cols);
#else
    TF_UNROLL for (int i = 0; i < 2; ++i) {
      const int r = my_r + 8 * i + int(fm);
      if (r < M) {
        TF_UNROLL for (int j = 0; j < 4; ++j) {
          TF_UNROLL for (int e = 0; e < 2; ++e) {
            const int c = my_c + 8 * j + int(fn) + e;
            if (c < N) {
              const bfloat raw = bfloat(acc[i][j].thread_elements()[e]);
              out[size_t(r) * N + c] = a.softcap
                  ? bfloat(30.0f * precise::tanh(precise::divide(float(raw), 30.0f))) : raw;
            }
          }
        }
      }
    }
#endif
  }
}
