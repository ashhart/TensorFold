#include <metal_stdlib>
using namespace metal;
// The image towers' shared glue (after gemm_nax_kernel.metal): bf16 rounding, threadgroup sum and max, row softmax.
inline float tf_vis_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (n threads, at most 1024), returned to all of them.
inline float tf_vis_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// One threadgroup's largest `v` (n threads, at most 1024), returned to all of them.
inline float tf_vis_max(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_max(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : -INFINITY;
    s = simd_max(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}
// Score products stay a full fp32 dot so an M5 does not cut an fp32 operand to 10 mantissa bits.
constant constexpr int IK2_DOT_KC = 16;  // k steps a staged chunk holds
constant constexpr int IK2_DOT_LD = 68;  // a staged row's pitch: 64 floats and 4 to stagger the banks

// This thread's 4 elements of each operand chunk (k in [k0, k0 + kc), `rows` and `cols` live); zero outside.
template <bool NT>
inline void ik2_dot_fetch(const device float* a, const device float* b, int k0, int kc, int rows, int cols, int lda,
                          int ldb, uint t, thread float* pa, thread float* pb) {
  for (uint i = 0; i < 4; ++i) {
    const uint e = t + i * 256;
    const uint ka = e % IK2_DOT_KC, ra = e / IK2_DOT_KC;
    pa[i] = (int(ka) < kc && int(ra) < rows) ? a[long(ra) * lda + k0 + ka] : 0.0f;
    const uint kb = NT ? ka : e / 64, rb = NT ? ra : e % 64;
    pb[i] = (int(kb) < kc && int(rb) < cols) ? (NT ? b[long(rb) * ldb + k0 + kb] : b[long(k0 + kb) * ldb + rb]) : 0.0f;
  }
}

// A group takes a 64 x 64 output tile, a thread 4 x 4 outputs; each output is one fma chain over k in order.
template <bool NT>
inline void ik2_f32_dot(const device float* A, const device float* B, const device int* P, device float* D,
                        threadgroup float* as, threadgroup float* bs, uint3 tg, uint t) {
  const int m = P[0], n = P[1], K = P[2], lda = P[3], ldb = P[4], ldd = P[5];
  const int row0 = int(tg.y) * 64, col0 = int(tg.x) * 64;
  const int rows = min(64, m - row0), cols = min(64, n - col0);
  const device float* a = A + long(tg.z) * P[10] + long(row0) * lda;
  const device float* b = B + long(tg.z) * P[11] + (NT ? long(col0) * ldb : long(col0));
  const uint ty = t / 16, tx = t % 16;
  float acc[4][4] = {};
  float pa[4], pb[4];
  int kc = min(IK2_DOT_KC, K);
  ik2_dot_fetch<NT>(a, b, 0, kc, rows, cols, lda, ldb, t, pa, pb);
  for (int k0 = 0; k0 < K; k0 += IK2_DOT_KC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint e = t + i * 256;
      as[(e % IK2_DOT_KC) * IK2_DOT_LD + e / IK2_DOT_KC] = pa[i];
      bs[NT ? (e % IK2_DOT_KC) * IK2_DOT_LD + e / IK2_DOT_KC : (e / 64) * IK2_DOT_LD + e % 64] = pb[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int here = kc, next = k0 + IK2_DOT_KC;
    kc = min(IK2_DOT_KC, K - next);
    if (next < K) ik2_dot_fetch<NT>(a, b, next, kc, rows, cols, lda, ldb, t, pa, pb);  // in flight during the fmas
    for (int kk = 0; kk < here; ++kk) {
      const float4 av = *reinterpret_cast<threadgroup const float4*>(as + kk * IK2_DOT_LD + ty * 4);
      const float4 bv = *reinterpret_cast<threadgroup const float4*>(bs + kk * IK2_DOT_LD + tx * 4);
      for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
      }
    }
  }
  device float* d = D + long(tg.z) * P[12] + long(row0) * ldd + col0;
  for (uint i = 0; i < 4; ++i) {
    if (int(ty * 4 + i) >= rows) break;
    for (uint j = 0; j < 4; ++j) {
      if (int(tx * 4 + j) < cols) d[long(ty * 4 + i) * ldd + tx * 4 + j] = acc[i][j];
    }
  }
}

#define IK2_F32_DOT(NAME, NT)                                                                                      \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                               \
      const device float* A [[buffer(0)]], const device float* B [[buffer(1)]],                                  \
      const device int* P [[buffer(2)]], device float* D [[buffer(3)]],                                          \
      uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                       \
    threadgroup float as[IK2_DOT_KC * IK2_DOT_LD], bs[IK2_DOT_KC * IK2_DOT_LD];                                  \
    ik2_f32_dot<NT>(A, B, P, D, as, bs, tg, t);                                                                  \
  }


IK2_F32_DOT(ik2_gemm_f32_nt_00, true)
IK2_F32_DOT(ik2_gemm_f32_nn_00, false)
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2v_softmax(
    device float* S [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& scale [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint tn [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device float* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) {
    s[i] = tf_vis_bf(tf_vis_bf(s[i]) * scale);
    m = max(m, s[i]);
  }
  m = tf_vis_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) {
    const float e = metal::precise::exp(s[i] - m);
    s[i] = e;
    z += e;
  }
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] = tf_vis_bf(s[i] * inv);
}
constant constexpr int N_KC = 16;
constant constexpr int N_LD = 68;

template <typename TA, bool NT, bool FUSE>
inline void n_fetch(const device TA* a, const device float* b, int k0, int kc, int rows, int cols, int lda,
                          int ldb, uint t, thread float* pa, thread float* pb, const device float2* st) {
  for (uint i = 0; i < 4; ++i) {
    const uint e = t + i * 256;
    const uint ka = e % N_KC, ra = e / N_KC;
    pa[i] = (int(ka) < kc && int(ra) < rows) ? float(a[long(ra) * lda + k0 + ka]) : 0.0f;
    if (FUSE && int(ka) < kc && int(ra) < rows) { const float2 ms = st[ra]; pa[i] = float(bfloat(metal::precise::exp(pa[i] - ms.x) * ms.y)); }
    const uint kb = NT ? ka : e / 64, rb = NT ? ra : e % 64;
    pb[i] = (int(kb) < kc && int(rb) < cols) ? (NT ? b[long(rb) * ldb + k0 + kb] : b[long(k0 + kb) * ldb + rb]) : 0.0f;
  }
}

// EPI: store bf16(bf16(acc) * scale) as bf16
template <bool NT, typename TA, typename TD, bool EPI, bool FUSE>
inline void n_dot(const device TA* A, const device float* B, const device int* P, device TD* D,
                        threadgroup float* as, threadgroup float* bs, uint3 tg, uint t, const device float2* ST) {
  const int m = P[0], n = P[1], K = P[2], lda = P[3], ldb = P[4], ldd = P[5];
  const int row0 = int(tg.y) * 64, col0 = int(tg.x) * 64;
  const int rows = min(64, m - row0), cols = min(64, n - col0);
  const device TA* a = A + long(tg.z) * P[10] + long(row0) * lda;
  const device float* b = B + long(tg.z) * P[11] + (NT ? long(col0) * ldb : long(col0));
  const device float2* st = ST + long(tg.z) * m + row0;
  const uint ty = t / 16, tx = t % 16;
  float acc[4][4] = {};
  float pa[4], pb[4];
  int kc = min(N_KC, K);
  n_fetch<TA, NT, FUSE>(a, b, 0, kc, rows, cols, lda, ldb, t, pa, pb, st);
  for (int k0 = 0; k0 < K; k0 += N_KC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint e = t + i * 256;
      as[(e % N_KC) * N_LD + e / N_KC] = pa[i];
      bs[NT ? (e % N_KC) * N_LD + e / N_KC : (e / 64) * N_LD + e % 64] = pb[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int here = kc, next = k0 + N_KC;
    kc = min(N_KC, K - next);
    if (next < K) n_fetch<TA, NT, FUSE>(a, b, next, kc, rows, cols, lda, ldb, t, pa, pb, st);
    for (int kk = 0; kk < here; ++kk) {
      const float4 av = *reinterpret_cast<threadgroup const float4*>(as + kk * N_LD + ty * 4);
      const float4 bv = *reinterpret_cast<threadgroup const float4*>(bs + kk * N_LD + tx * 4);
      for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
      }
    }
  }
  device TD* d = D + long(tg.z) * P[12] + long(row0) * ldd + col0;
  const float scale = as_type<float>(P[13]);
  for (uint i = 0; i < 4; ++i) {
    if (int(ty * 4 + i) >= rows) break;
    for (uint j = 0; j < 4; ++j) {
      if (int(tx * 4 + j) < cols) d[long(ty * 4 + i) * ldd + tx * 4 + j] = EPI ? TD(tf_vis_bf(tf_vis_bf(acc[i][j]) * scale)) : TD(acc[i][j]);
    }
  }
}

#define N_DOT(NAME, NT, TA, TD, EPI, FUSE)                                                                                  \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                               \
      const device TA* A [[buffer(0)]], const device float* B [[buffer(1)]],                                  \
      const device int* P [[buffer(2)]], device TD* D [[buffer(3)]], const device float2* ST [[buffer(4)]],        \
      uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                       \
    threadgroup float as[N_KC * N_LD], bs[N_KC * N_LD];                                  \
    n_dot<NT, TA, TD, EPI, FUSE>(A, B, P, D, as, bs, tg, t, ST);                                                                  \
  }
N_DOT(qk_bf, true, float, bfloat, true, false)
N_DOT(pv_bf, false, bfloat, float, false, false)
N_DOT(pv_ex, false, bfloat, float, false, true)

// bf16 scores in place, a threadgroup of 256 a row, exps recomputed in the last pass
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sm_bf(
    device bfloat* S [[buffer(0)]], constant uint& n [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint tn [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device bfloat* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, float(s[i]));
  m = tf_vis_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) z += metal::precise::exp(float(s[i]) - m);
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] = bfloat(metal::precise::exp(float(s[i]) - m) * inv);
}

// one simdgroup a row (8 rows a group); the 256-thread group's sum order kept: 8 virtual simdgroups of lanes
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sm_sg(
    device bfloat* S [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& rows [[buffer(2)]],
    uint tgi [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row = tgi * 8 + sg;
  if (row >= rows) return;
  device bfloat* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = lane; i < n; i += 32) m = max(m, float(s[i]));
  m = simd_max(m);
  float z[8] = {};
  for (uint base = 0; base < n; base += 256) {
    for (uint j = 0; j < 8; ++j) {
      const uint i = base + j * 32 + lane;
      if (i < n) z[j] += metal::precise::exp(float(s[i]) - m);
    }
  }
  float mine = 0.0f;
  for (uint j = 0; j < 8; ++j) {
    const float v = simd_sum(z[j]);
    if (lane == j) mine = v;
  }
  const float inv = 1.0f / simd_sum(mine);
  for (uint i = lane; i < n; i += 32) s[i] = bfloat(metal::precise::exp(float(s[i]) - m) * inv);
}

// row stats only: m and 1/sum into st[row], bf16 scores untouched
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void st_bf(
    const device bfloat* S [[buffer(0)]], constant uint& n [[buffer(1)]], device float2* st [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint tn [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, float(s[i]));
  m = tf_vis_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) z += metal::precise::exp(float(s[i]) - m);
  const float inv = 1.0f / tf_vis_sum(z, part, tn, lane, sg);
  if (t == 0) st[row] = float2(m, inv);
}
