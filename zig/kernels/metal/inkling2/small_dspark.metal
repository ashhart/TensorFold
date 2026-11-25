#include <metal_stdlib>
#pragma METAL fp contract(off)
#include <metal_simdgroup_matrix>
#ifndef TF_SIMD_FRAGS
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")
namespace tfps {
using namespace metal;
template <typename T> using frag = vec<T, 8>;
#ifndef TF_SIMD_FRAGS
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}
#define TF_COL(e) (e & 3)
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
    left[e] = a[e]; right[e] = b0[e]; right[8 + e] = b1[e]; acc[e] = lo[e]; acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL for (short e = 0; e < 8; e++) { lo[e] = acc[e]; hi[e] = acc[8 + e]; }
}
inline short t_key(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short t_dim(short e, short2 home) { return home.x + (e & 3); }
#else
inline short2 frag_home(ushort l) {
  return short2(short(((l & 8) >> 1) | ((l & 1) << 1)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}
#define TF_COL(e) (((e) & 1) + 8 * (((e) >> 1) & 1))
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  simdgroup_matrix<A, 8, 8> ma[4]; simdgroup_matrix<B, 8, 8> mb[4], mc[4]; simdgroup_matrix<C, 8, 8> md[4], me[4];
  TF_UNROLL for (short i = 0; i < 4; i++) {
    ma[i].thread_elements()[0] = a[2 * i]; ma[i].thread_elements()[1] = a[2 * i + 1];
    mb[i].thread_elements()[0] = b0[2 * i]; mb[i].thread_elements()[1] = b0[2 * i + 1];
    mc[i].thread_elements()[0] = b1[2 * i]; mc[i].thread_elements()[1] = b1[2 * i + 1];
    md[i].thread_elements()[0] = lo[2 * i]; md[i].thread_elements()[1] = lo[2 * i + 1];
    me[i].thread_elements()[0] = hi[2 * i]; me[i].thread_elements()[1] = hi[2 * i + 1];
  }
  TF_UNROLL for (short r = 0; r < 2; r++) TF_UNROLL for (short c = 0; c < 2; c++) TF_UNROLL for (short k = 0; k < 2; k++) {
    simdgroup_multiply_accumulate(md[2 * r + c], ma[2 * r + k], mb[2 * k + c], md[2 * r + c]);
    simdgroup_multiply_accumulate(me[2 * r + c], ma[2 * r + k], mc[2 * k + c], me[2 * r + c]);
  }
  TF_UNROLL for (short i = 0; i < 4; i++) {
    lo[2 * i] = md[i].thread_elements()[0]; lo[2 * i + 1] = md[i].thread_elements()[1];
    hi[2 * i] = me[i].thread_elements()[0]; hi[2 * i + 1] = me[i].thread_elements()[1];
  }
}
inline short t_key(short e, short2 home) { return home.x + TF_COL(e); }
inline short t_dim(short e, short2 home) { return home.y + (e >> 2) * 8; }
#endif
inline short p_row(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short p_col(short e, short2 home) { return home.x + TF_COL(e); }
template <typename T, typename P> inline void frag_get_tg(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  TF_UNROLL for (short e = 0; e < 8; e++) f[e] = T(p[(r + home.y + (e >> 2) * 8) * ld + (c + home.x + TF_COL(e))]);
}
}

constant uint SDSP_H = 4608, SDSP_Q = 6144, SDSP_KV = 1024, SDSP_HEAD = 128, SDSP_RNN = 768;
constant float SDSP_EPS = 1e-6f, SDSP_SCALE = 0x1.6a09e6p-4f;

struct SmallDsparkFuseArgs { uint count, stride_rows; };
struct SmallDsparkPositionArgs { uint base[8]; };
struct SmallDsparkAttentionArgs { uint local, streams; uint committed[8], ring[8], from[8]; };
struct SmallDsparkKappaArgs { uint rows, stride_s, row_off, streams; };
struct SmallDsparkResidualArgs { uint init; };
struct SmallDsparkBaseArgs { uint rows, vocab; };

static inline float sdsp_nan_max(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : max(a, b); }
static inline float sdsp_row_max(float v) {
  v = sdsp_nan_max(v, simd_shuffle_xor(v, ushort(1)));
  return sdsp_nan_max(v, simd_shuffle_xor(v, ushort(8)));
}
static inline float sdsp_row_sum(float v) { v += simd_shuffle_xor(v, ushort(1)); return v + simd_shuffle_xor(v, ushort(8)); }
static inline float sdsp_rope_angle(uint d, uint pos) { return float(pos) * pow(1e6f, -float(d % 64u) / 64.0f); }

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_dspark_init_rows(const device bfloat* anchor [[buffer(0)]], const device bfloat* mask [[buffer(1)]],
    device bfloat* rows [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  const uint s = row / 8u, r = row % 8u;
  const device bfloat* src = r == 0u ? anchor + uint64_t(s) * SDSP_H : mask;
  for (uint d = t; d < SDSP_H; d += 256u) { rows[uint64_t(row) * SDSP_H + d] = src[d]; }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_dspark_fuse_taps(const device bfloat* taps [[buffer(0)]], const device bfloat* fuse_weight [[buffer(1)]],
    const device bfloat* norm_weight [[buffer(2)]], device bfloat* c [[buffer(3)]], device bfloat* newest_f [[buffer(4)]],
    constant SmallDsparkFuseArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint fusion = group / a.count, row = group % a.count;
  threadgroup float alpha[8], sh_sum[8];
  if (t == 0u) {
    float scores[8], peak = -INFINITY;
    for (uint j = 0; j < 8u; ++j) { scores[j] = float(fuse_weight[fusion * 8u + j]); peak = sdsp_nan_max(peak, scores[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < 8u; ++j) { scores[j] = exp(scores[j] - peak); sum += scores[j]; }
    for (uint j = 0; j < 8u; ++j) alpha[j] = float(bfloat(scores[j] / sum));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float square = 0.0f;
  for (uint d = t; d < SDSP_H; d += 256u) {
    float mixed = 0.0f;
    for (uint j = 0; j < 8u; ++j) mixed += alpha[j] * float(taps[uint64_t(row * 8u + j) * SDSP_H + d]);
    square += mixed * mixed;
  }
  square = simd_sum(square);
  if (lane == 0u) sh_sum[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f; for (uint k = 0; k < 8u; ++k) tot += sh_sum[k];
  const float scale = rsqrt(tot / float(SDSP_H) + SDSP_EPS);
  for (uint d = t; d < SDSP_H; d += 256u) {
    float mixed = 0.0f;
    for (uint j = 0; j < 8u; ++j) mixed += alpha[j] * float(taps[uint64_t(row * 8u + j) * SDSP_H + d]);
    const bfloat out = bfloat(mixed * scale * float(norm_weight[uint64_t(fusion) * SDSP_H + d]));
    c[uint64_t(fusion * a.stride_rows + row) * SDSP_H + d] = out;
    if (fusion == 5u && row + 1u == a.count) newest_f[d] = out;
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_small_dspark_block_qkv(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* q_weight [[buffer(3)]], const device bfloat* k_weight [[buffer(4)]],
    device bfloat* q [[buffer(5)]], device bfloat* k [[buffer(6)]], device bfloat* v [[buffer(7)]],
    constant SmallDsparkPositionArgs& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row_g = group / 64u, work = group % 64u, s = row_g / 8u, row = row_g % 8u;
  threadgroup float sh_sum[4];
  if (work < 56u) {
    const float x = float(work < 48u ? q_raw[uint64_t(row_g) * SDSP_Q + work * SDSP_HEAD + d] : k_raw[uint64_t(row_g) * SDSP_KV + (work - 48u) * SDSP_HEAD + d]);
    const float part = simd_sum(x * x);
    if (lane == 0u) sh_sum[sg] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (work >= 56u) { v[uint64_t(row_g) * SDSP_KV + (work - 56u) * SDSP_HEAD + d] = v_raw[uint64_t(row_g) * SDSP_KV + (work - 56u) * SDSP_HEAD + d]; return; }
  float tot = 0.0f; for (uint i = 0; i < 4u; ++i) tot += sh_sum[i];
  const float scale = rsqrt(tot / float(SDSP_HEAD) + SDSP_EPS);
  const bool query = work < 48u;
  const uint h = query ? work : work - 48u;
  const uint64_t at = uint64_t(row_g) * (query ? SDSP_Q : SDSP_KV) + h * SDSP_HEAD + d;
  const float x = float(bfloat(float(query ? q_raw[at] : k_raw[at]) * scale * float(query ? q_weight[d] : k_weight[d])));
  const uint partner = d < 64u ? d + 64u : d - 64u;
  const uint64_t other_at = uint64_t(row_g) * (query ? SDSP_Q : SDSP_KV) + h * SDSP_HEAD + partner;
  const float other = float(bfloat(float(query ? q_raw[other_at] : k_raw[other_at]) * scale * float(query ? q_weight[partner] : k_weight[partner])));
  const float angle = sdsp_rope_angle(d, a.base[s] + row), cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64u ? x * cs - other * sn : other * sn + x * cs;
  if (query) q[at] = bfloat(rotated); else k[at] = bfloat(rotated);
}

[[max_total_threads_per_threadgroup(384)]]
kernel void tf_ink2_small_dspark_attention(
    const device bfloat* q [[buffer(0)]], const device bfloat* block_k [[buffer(1)]], const device bfloat* block_v [[buffer(2)]],
    device bfloat* out [[buffer(3)]], const device bfloat* ck0 [[buffer(4)]], const device bfloat* ck1 [[buffer(5)]],
    const device bfloat* ck2 [[buffer(6)]], const device bfloat* ck3 [[buffer(7)]], const device bfloat* ck4 [[buffer(8)]],
    const device bfloat* ck5 [[buffer(9)]], const device bfloat* ck6 [[buffer(10)]], const device bfloat* ck7 [[buffer(11)]],
    const device bfloat* cv0 [[buffer(12)]], const device bfloat* cv1 [[buffer(13)]], const device bfloat* cv2 [[buffer(14)]],
    const device bfloat* cv3 [[buffer(15)]], const device bfloat* cv4 [[buffer(16)]], const device bfloat* cv5 [[buffer(17)]],
    const device bfloat* cv6 [[buffer(18)]], const device bfloat* cv7 [[buffer(19)]], constant SmallDsparkAttentionArgs& a [[buffer(20)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float smem[3456];
  threadgroup bfloat* sh_q = (threadgroup bfloat*)smem;
  threadgroup int* sh_key = (threadgroup int*)(smem + 1024);
  const uint tid = sg * 32u + lane, kv = tg.x, s = tg.y, qb = tg.z, kw = sg;
  const uint committed = a.committed[s], ring = a.ring[s], from = a.from[s];
  const uint first = max(from, a.local && committed > 1024u ? committed - 1024u : 0u);
  const uint retained = committed - first, count = retained + 8u;
  const device bfloat* context_k = s == 0u ? ck0 : s == 1u ? ck1 : s == 2u ? ck2 : s == 3u ? ck3 : s == 4u ? ck4 : s == 5u ? ck5 : s == 6u ? ck6 : ck7;
  const device bfloat* context_v = s == 0u ? cv0 : s == 1u ? cv1 : s == 2u ? cv2 : s == 3u ? cv3 : s == 4u ? cv4 : s == 5u ? cv5 : s == 6u ? cv6 : cv7;
  for (uint i = tid; i < 512u; i += 384u) {
    const uint qi_loc = i / 32u, d4 = (i % 32u) * 4u, qi = qb * 16u + qi_loc, row = qi / 6u, qh = kv * 6u + (qi % 6u);
    *(threadgroup bfloat4*)(sh_q + qi_loc * 128u + d4) = *(const device bfloat4*)(q + uint64_t((s * 8u + row) * 48u + qh) * 128u + d4);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = tfps::frag_home(ushort(lane));
  tfps::frag<bfloat> fq[8];
  TF_UNROLL for (short t = 0; t < 8; t++) tfps::frag_get_tg(fq[t], sh_q, 128, 0, 16 * t, home);
  tfps::frag<float> acc[4][2];
  TF_UNROLL for (short c = 0; c < 4; c++) { acc[c][0] = tfps::frag<float>(0); acc[c][1] = tfps::frag<float>(0); }
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  const device bfloat *bk = block_k + uint64_t(s) * 8192u + kv * 128u, *bv = block_v + uint64_t(s) * 8192u + kv * 128u;
  const device bfloat *ck = context_k + kv * 128u, *cv = context_v + kv * 128u;
  for (uint base = 0; base < count; base += 384u) {
    {
      const uint j = base + tid;
      sh_key[tid] = j < count ? (j < retained ? int((first + j) % ring) : -2 - int(j - retained)) : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (base + kw * 32u < count) {
      tfps::frag<float> lo_s = tfps::frag<float>(0), hi_s = tfps::frag<float>(0);
      TF_UNROLL for (short t = 0; t < 8; t++) {
        tfps::frag<bfloat> b0, b1;
#ifndef TF_SIMD_FRAGS
        const device bfloat* kr[4];
        TF_UNROLL for (short i = 0; i < 4; i++) {
          const int kp = sh_key[kw * 32u + 8 * i + home.y];
          kr[i] = (kp >= 0 ? ck + uint64_t(kp) * 1024u : (kp <= -2 ? bk + uint64_t(-2 - kp) * 1024u : bk)) + home.x;
        }
        b0 = tfps::frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
        b1 = tfps::frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
#else
        TF_UNROLL for (short e = 0; e < 8; e++) {
          const int kp0 = sh_key[kw * 32u + tfps::t_key(e, home)], kp1 = sh_key[kw * 32u + 16 + tfps::t_key(e, home)];
          const int dd = 16 * t + tfps::t_dim(e, home);
          b0[e] = kp0 >= 0 ? ck[uint64_t(kp0) * 1024u + dd] : (kp0 <= -2 ? bk[uint64_t(-2 - kp0) * 1024u + dd] : bfloat(0));
          b1[e] = kp1 >= 0 ? ck[uint64_t(kp1) * 1024u + dd] : (kp1 <= -2 ? bk[uint64_t(-2 - kp1) * 1024u + dd] : bfloat(0));
        }
#endif
        tfps::mma_16x32<false, true>(lo_s, hi_s, fq[t], b0, b1);
      }
      float sc[16], factor[2], bsum[2] = {0.0f, 0.0f};
      bool vmask[16];
      TF_UNROLL for (short i = 0; i < 16; i++) {
        const float v = (i < 8 ? lo_s[i] : hi_s[i & 7]) * SDSP_SCALE;
        const int kp = sh_key[kw * 32u + (i < 8 ? 0 : 16) + tfps::p_col(i & 7, home)];
        vmask[i] = kp != -1; sc[i] = vmask[i] ? v : -INFINITY;
      }
      TF_UNROLL for (short h = 0; h < 2; h++) {
        float bm = -INFINITY;
        TF_UNROLL for (short i = 0; i < 4; i++) {
          if (vmask[4 * h + i]) bm = sdsp_nan_max(bm, sc[4 * h + i]);
          if (vmask[8 + 4 * h + i]) bm = sdsp_nan_max(bm, sc[8 + 4 * h + i]);
        }
        const float om = m[h], nm = sdsp_nan_max(om, sdsp_row_max(bm));
        factor[h] = isnan(nm) ? NAN : (om == -INFINITY ? 0.0f : (nm == om ? 1.0f : precise::exp(om - nm)));
        m[h] = nm;
      }
      tfps::frag<bfloat> p1[2], p2[2], p3[2];
      TF_UNROLL for (short i = 0; i < 16; i++) {
        const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
        const float pr = vmask[i] ? precise::exp(sc[i] - m[h]) : 0.0f;
        bsum[h] += pr;
        const bfloat x = bfloat(pr); const float r1 = pr - float(x); const bfloat y = bfloat(r1);
        p1[kb][e] = x; p2[kb][e] = y; p3[kb][e] = bfloat(r1 - float(y));
      }
      TF_UNROLL for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + sdsp_row_sum(bsum[h]);
      TF_UNROLL for (short c = 0; c < 4; c++) TF_UNROLL for (short e = 0; e < 8; e++) { acc[c][0][e] *= factor[e >> 2]; acc[c][1][e] *= factor[e >> 2]; }
      TF_UNROLL for (short kb = 0; kb < 2; kb++) {
#ifndef TF_SIMD_FRAGS
        const int kpa = sh_key[kw * 32u + 16 * kb + home.y], kpb = sh_key[kw * 32u + 16 * kb + home.y + 8];
        const device bfloat* va = (kpa >= 0 ? cv + uint64_t(kpa) * 1024u : (kpa <= -2 ? bv + uint64_t(-2 - kpa) * 1024u : bv)) + home.x;
        const device bfloat* vb = (kpb >= 0 ? cv + uint64_t(kpb) * 1024u : (kpb <= -2 ? bv + uint64_t(-2 - kpb) * 1024u : bv)) + home.x;
#endif
        TF_UNROLL for (short c = 0; c < 4; c++) {
          tfps::frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
          v0 = tfps::frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
          v1 = tfps::frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
#else
          TF_UNROLL for (short e = 0; e < 8; e++) {
            const int kp = sh_key[kw * 32u + 16 * kb + tfps::p_row(e, home)], dd = 32 * c + tfps::p_col(e, home);
            v0[e] = kp >= 0 ? cv[uint64_t(kp) * 1024u + dd] : (kp <= -2 ? bv[uint64_t(-2 - kp) * 1024u + dd] : bfloat(0));
            v1[e] = kp >= 0 ? cv[uint64_t(kp) * 1024u + dd + 16] : (kp <= -2 ? bv[uint64_t(-2 - kp) * 1024u + dd + 16] : bfloat(0));
          }
#endif
          tfps::mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
          tfps::mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
          tfps::mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float *sh_m = smem, *sh_l = smem + 192, *sh_part = smem + 384;
  TF_UNROLL for (short h = 0; h < 2; h++) {
    const uint qr_loc = uint(home.y) + uint(h) * 8u;
    if (home.x == 0) { sh_m[kw * 16u + qr_loc] = m[h]; sh_l[kw * 16u + qr_loc] = l[h]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w_kw[2] = {0.0f, 0.0f};
  TF_UNROLL for (short h = 0; h < 2; h++) {
    const uint qr_loc = uint(home.y) + uint(h) * 8u;
    float gm = -INFINITY; for (uint k = 0; k < 12u; k++) gm = sdsp_nan_max(gm, sh_m[k * 16u + qr_loc]);
    if (isnan(gm)) { w_kw[h] = kw == 0u ? NAN : 0.0f; } else {
      float gl = 0.0f;
      for (uint k = 0; k < 12u; k++) if (sh_l[k * 16u + qr_loc] > 0.0f) gl += sh_l[k * 16u + qr_loc] * precise::exp(sh_m[k * 16u + qr_loc] - gm);
      w_kw[h] = (gl > 0.0f && l[h] > 0.0f) ? precise::exp(m[h] - gm) / gl : 0.0f;
    }
  }
  TF_UNROLL for (short c = 0; c < 4; c++) TF_UNROLL for (short j = 0; j < 2; j++) {
    TF_UNROLL for (short e = 0; e < 8; e++) {
      const uint qr_loc = uint(home.y) + uint(e >> 2) * 8u;
      sh_part[(kw * 16u + qr_loc) * 16u + uint(home.x + TF_COL(e))] = isnan(w_kw[e >> 2]) ? NAN : acc[c][j][e] * w_kw[e >> 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256u) {
      const uint qr_loc = tid / 16u, d_in = tid % 16u, qr = qb * 16u + qr_loc;
      const uint row = qr / 6u, qh = kv * 6u + (qr % 6u), dd = 32u * uint(c) + 16u * uint(j) + d_in;
      float tot = 0.0f; for (uint k = 0; k < 12u; k++) tot += sh_part[(k * 16u + qr_loc) * 16u + d_in];
      out[uint64_t((s * 8u + row) * 48u + qh) * 128u + dd] = bfloat(tot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_dspark_residual_norm(const device bfloat* x [[buffer(0)]], const device bfloat* residual [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* new_residual [[buffer(3)]], device bfloat* normalized [[buffer(4)]],
    constant SmallDsparkResidualArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_sum[8];
  float reg[18], sum = 0.0f;
  for (uint n = 0; n < 18u; ++n) {
    const uint d = n * 256u + t;
    const uint64_t at = uint64_t(row) * SDSP_H + d;
    const float v = a.init ? float(x[at]) : float(x[at]) + float(residual[at]);
    reg[n] = v; sum += v * v;
  }
  const float part = simd_sum(sum);
  if (lane == 0u) sh_sum[sg] = part;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f; for (uint k = 0; k < 8u; ++k) tot += sh_sum[k];
  const float scale = rsqrt(tot / float(SDSP_H) + SDSP_EPS);
  for (uint n = 0; n < 18u; ++n) {
    const uint d = n * 256u + t;
    const uint64_t at = uint64_t(row) * SDSP_H + d;
    const float sigma = reg[n];
    new_residual[at] = bfloat(sigma);
    normalized[at] = bfloat(sigma * scale * float(weight[d]));
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void tf_ink2_small_dspark_kappa(const device bfloat* y [[buffer(0)]], const device bfloat* f0 [[buffer(1)]],
    const device bfloat* f1 [[buffer(2)]], const device bfloat* f2 [[buffer(3)]], const device bfloat* f3 [[buffer(4)]],
    const device bfloat* f4 [[buffer(5)]], const device bfloat* f5 [[buffer(6)]], const device bfloat* f6 [[buffer(7)]],
    const device bfloat* f7 [[buffer(8)]], const device bfloat* w_kappa [[buffer(9)]], const device bfloat* bias [[buffer(10)]],
    device float* out [[buffer(11)]], constant SmallDsparkKappaArgs& a [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint tid = sg * 32u + lane, ch = tg.x * 64u + tid, s = tg.y;
  const device bfloat* f = s == 0u ? f0 : s == 1u ? f1 : s == 2u ? f2 : s == 3u ? f3 : s == 4u ? f4 : s == 5u ? f5 : s == 6u ? f6 : f7;
  const device bfloat* ys = y + uint64_t(s * a.stride_s + a.row_off) * SDSP_H;
  float sy[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint d = 0; d < SDSP_H; ++d) {
    const float wy = float(w_kappa[uint64_t(d) * SDSP_RNN + ch]);
    for (uint r = 0; r < a.rows; ++r) sy[r] += float(ys[uint64_t(r) * SDSP_H + d]) * wy;
  }
  float sf = float(bias[ch]);
  for (uint d = 0; d < SDSP_H; ++d) sf += float(f[d]) * float(w_kappa[uint64_t(SDSP_H + d) * SDSP_RNN + ch]);
  for (uint r = 0; r < a.rows; ++r) out[uint64_t(r * a.streams + s) * SDSP_RNN + ch] = sy[r] + sf;
}

static inline void small_dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(val); }

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_small_dspark_base(const device bfloat* unembed [[buffer(0)]], const device bfloat* y [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant SmallDsparkBaseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat sh_x[16][32], sh_w[32][32];
  threadgroup float sh_c[16][32];
  const uint tid = sg * 32u + lane;
  if (a.rows < 4u) {
    const uint v = tg.x * 8u + sg;
    if (v >= a.vocab) return;
    const device bfloat* w = unembed + uint64_t(v) * SDSP_H;
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (uint i = 8u * lane; i < SDSP_H; i += 256u) {
      const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4u));
      for (uint r = 0; r < a.rows; ++r) {
        const device bfloat* xr = y + uint64_t(r) * SDSP_H + i;
        acc[r] += dot(w0, float4(*(const device bfloat4*)xr)) + dot(w1, float4(*(const device bfloat4*)(xr + 4u)));
      }
    }
    for (uint r = 0; r < a.rows; ++r) { const float sum = simd_sum(acc[r]); if (lane == 0u) small_dspark_store_base(out, uint64_t(r) * a.vocab + v, sum); }
    return;
  }
  const uint r0 = tg.y * 16u, c0 = tg.x * 32u, r_sg = sg / 4u, c_sg = sg % 4u;
  if (r0 >= a.rows || c0 >= a.vocab) return;
  simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);
  for (uint k = 0; k < SDSP_H; k += 32u) {
    if (tid < 128u) {
      const uint rr = tid / 8u, cc = (tid % 8u) * 4u;
      *(threadgroup bfloat4*)(&sh_x[rr][cc]) = r0 + rr < a.rows ? *(const device bfloat4*)(y + uint64_t(r0 + rr) * SDSP_H + k + cc) : bfloat4(0);
    }
    {
      const uint rr = tid / 8u, cc = (tid % 8u) * 4u;
      *(threadgroup bfloat4*)(&sh_w[rr][cc]) = c0 + rr < a.vocab ? *(const device bfloat4*)(unembed + uint64_t(c0 + rr) * SDSP_H + k + cc) : bfloat4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL for (uint kt = 0; kt < 32u; kt += 8u) {
      simdgroup_matrix<bfloat, 8, 8> xa, wb;
      simdgroup_load(xa, &sh_x[r_sg * 8u][kt], 32);
      simdgroup_load(wb, &sh_w[c_sg * 8u][kt], 32, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc, xa, wb, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc, &sh_c[r_sg * 8u][c_sg * 8u], 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 512u; i += 256u) {
    const uint rr = i / 32u, cc = i % 32u;
    if (r0 + rr < a.rows && c0 + cc < a.vocab) small_dspark_store_base(out, uint64_t(r0 + rr) * a.vocab + c0 + cc, sh_c[rr][cc]);
  }
}
