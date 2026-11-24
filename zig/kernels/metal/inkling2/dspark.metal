#include <metal_stdlib>
#pragma METAL fp contract(off)
#include <metal_simdgroup_matrix>
#ifndef TF_SIMD_FRAGS
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")
namespace tfp {
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

constant uint DSP_H = 2304, DSP_Q = 3072, DSP_KV = 1024, DSP_HEAD = 128, DSP_RANK = 256, DSP_RNN = 768;
constant float DSP_EPS = 1e-6f, DSP_SCALE = 0x1.6a09e6p-4f;

struct DsparkFuseArgs { uint count, stride_rows; };
struct DsparkPositionArgs { uint base[8]; };
struct DsparkContextArgs { uint base, rows, ring; };
struct DsparkAttentionArgs { uint local, streams; uint committed[8], ring[8], from[8]; };
struct DsparkKappaArgs { uint rows, stride_s, row_off, streams; };
struct DsparkTokenArgs { uint token, step, use_held; };
struct DsparkResidualArgs { uint init; };
struct DsparkCountArgs { uint count; };
struct DsparkBaseArgs { uint rows, vocab; };
struct DsparkLogitsArgs { uint count, phi_stride, base_stride, streams, step, mode, flags; };
struct DsparkGroupStat { float m, l, best_v; uint best_id; };

static inline float dsp_nan_max(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : max(a, b); }
static inline float dsp_row_max(float v) {
  v = dsp_nan_max(v, simd_shuffle_xor(v, ushort(1)));
  return dsp_nan_max(v, simd_shuffle_xor(v, ushort(8)));
}
static inline float dsp_row_sum(float v) { v += simd_shuffle_xor(v, ushort(1)); return v + simd_shuffle_xor(v, ushort(8)); }
static inline float dspark_rope_angle(uint d, uint pos) { return float(pos) * pow(1e6f, -float(d % 64u) / 64.0f); }
static inline void dsp_merge_tg(threadgroup float* sh_m, threadgroup float* sh_l, threadgroup float* sh_bv,
                                threadgroup uint* sh_bi, thread float& gm, thread float& gl, thread float& gbv, thread uint& gbi) {
  gm = -INFINITY; gl = 0.0f; gbv = -INFINITY; gbi = 0xffffffffu;
  for (uint k = 0; k < 8u; ++k) {
    const float nm = dsp_nan_max(gm, sh_m[k]);
    gl = isnan(nm) ? NAN : ((gm == -INFINITY ? 0.0f : gl * exp(gm - nm)) + (sh_m[k] == -INFINITY ? 0.0f : sh_l[k] * exp(sh_m[k] - nm)));
    gm = nm;
    if (isnan(gbv) || isnan(sh_bv[k])) { gbv = NAN; }
    else if (sh_bv[k] > gbv || (sh_bv[k] == gbv && sh_bi[k] < gbi)) { gbv = sh_bv[k]; gbi = sh_bi[k]; }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_init_rows(const device bfloat* anchor [[buffer(0)]], const device bfloat* mask [[buffer(1)]],
    device bfloat* rows [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  const uint s = row / 8u, r = row % 8u;
  const device bfloat* src = r == 0u ? anchor + uint64_t(s) * DSP_H : mask;
  for (uint d = t; d < DSP_H; d += 256u) { rows[row * DSP_H + d] = src[d]; }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_fuse_taps(const device bfloat* taps [[buffer(0)]], const device bfloat* fuse_weight [[buffer(1)]],
    const device bfloat* norm_weight [[buffer(2)]], device bfloat* c [[buffer(3)]], device bfloat* newest_f [[buffer(4)]],
    constant DsparkFuseArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint fusion = group / a.count, row = group % a.count;
  threadgroup float alpha[4], sh_sum[8];
  if (t == 0u) {
    float scores[4], peak = -INFINITY;
    for (uint j = 0; j < 4u; ++j) { scores[j] = float(fuse_weight[fusion * 4u + j]); peak = dsp_nan_max(peak, scores[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) { scores[j] = exp(scores[j] - peak); sum += scores[j]; }
    for (uint j = 0; j < 4u; ++j) alpha[j] = float(bfloat(scores[j] / sum));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float square = 0.0f;
  for (uint d = t; d < DSP_H; d += 256u) {
    float mixed = 0.0f;
    for (uint j = 0; j < 4u; ++j) mixed += alpha[j] * float(taps[(row * 4u + j) * DSP_H + d]);
    square += mixed * mixed;
  }
  square = simd_sum(square);
  if (lane == 0u) sh_sum[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f; for (uint k = 0; k < 8u; ++k) tot += sh_sum[k];
  const float scale = rsqrt(tot / float(DSP_H) + DSP_EPS);
  for (uint d = t; d < DSP_H; d += 256u) {
    float mixed = 0.0f;
    for (uint j = 0; j < 4u; ++j) mixed += alpha[j] * float(taps[(row * 4u + j) * DSP_H + d]);
    const bfloat out = bfloat(mixed * scale * float(norm_weight[fusion * DSP_H + d]));
    c[(fusion * a.stride_rows + row) * DSP_H + d] = out;
    if (fusion == 3u && row + 1u == a.count) newest_f[d] = out;
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_dspark_block_qkv(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]],
    const device bfloat* v_raw [[buffer(2)]], const device bfloat* q_weight [[buffer(3)]], const device bfloat* k_weight [[buffer(4)]],
    device bfloat* q [[buffer(5)]], device bfloat* k [[buffer(6)]], device bfloat* v [[buffer(7)]],
    constant DsparkPositionArgs& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row_g = group / 40u, work = group % 40u, s = row_g / 8u, row = row_g % 8u;
  threadgroup float sh_sum[4];
  if (work < 32u) {
    const float x = float(work < 24u ? q_raw[row_g * DSP_Q + work * DSP_HEAD + d] : k_raw[row_g * DSP_KV + (work - 24u) * DSP_HEAD + d]);
    const float part = simd_sum(x * x);
    if (lane == 0u) sh_sum[sg] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (work >= 32u) { v[row_g * DSP_KV + (work - 32u) * DSP_HEAD + d] = v_raw[row_g * DSP_KV + (work - 32u) * DSP_HEAD + d]; return; }
  float tot = 0.0f; for (uint i = 0; i < 4u; ++i) tot += sh_sum[i];
  const float scale = rsqrt(tot / float(DSP_HEAD) + DSP_EPS);
  const bool query = work < 24u;
  const uint h = query ? work : work - 24u, at = row_g * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + d;
  const float x = float(bfloat(float(query ? q_raw[at] : k_raw[at]) * scale * float(query ? q_weight[d] : k_weight[d])));
  const uint partner = d < 64u ? d + 64u : d - 64u, other_at = row_g * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + partner;
  const float other = float(bfloat(float(query ? q_raw[other_at] : k_raw[other_at]) * scale * float(query ? q_weight[partner] : k_weight[partner])));
  const float angle = dspark_rope_angle(d, a.base[s] + row), cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64u ? x * cs - other * sn : other * sn + x * cs;
  if (query) q[at] = bfloat(rotated); else k[at] = bfloat(rotated);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_dspark_context_kv(const device bfloat* k_raw [[buffer(0)]], const device bfloat* v_raw [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* keys [[buffer(3)]], device bfloat* values [[buffer(4)]],
    constant DsparkContextArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group / 8u, h = group % 8u, src = row * DSP_KV + h * DSP_HEAD + d;
  threadgroup float sh_sum[4];
  const float sq = simd_sum(float(k_raw[src]) * float(k_raw[src]));
  if (lane == 0u) sh_sum[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f; for (uint i = 0; i < 4u; ++i) tot += sh_sum[i];
  const float scale = rsqrt(tot / float(DSP_HEAD) + DSP_EPS);
  const uint position = a.base + row, at = (position % a.ring) * DSP_KV + h * DSP_HEAD + d, partner = d < 64u ? d + 64u : d - 64u;
  const float x = float(bfloat(float(k_raw[src]) * scale * float(weight[d])));
  const float other = float(bfloat(float(k_raw[row * DSP_KV + h * DSP_HEAD + partner]) * scale * float(weight[partner])));
  const float angle = dspark_rope_angle(d, position), cs = cos(angle), sn = sin(angle);
  keys[at] = bfloat(d < 64u ? x * cs - other * sn : other * sn + x * cs);
  values[at] = v_raw[src];
}

[[max_total_threads_per_threadgroup(768)]]
kernel void tf_ink2_dspark_attention(
    const device bfloat* q [[buffer(0)]], const device bfloat* block_k [[buffer(1)]], const device bfloat* block_v [[buffer(2)]],
    device bfloat* out [[buffer(3)]], const device bfloat* ck0 [[buffer(4)]], const device bfloat* ck1 [[buffer(5)]],
    const device bfloat* ck2 [[buffer(6)]], const device bfloat* ck3 [[buffer(7)]], const device bfloat* ck4 [[buffer(8)]],
    const device bfloat* ck5 [[buffer(9)]], const device bfloat* ck6 [[buffer(10)]], const device bfloat* ck7 [[buffer(11)]],
    const device bfloat* cv0 [[buffer(12)]], const device bfloat* cv1 [[buffer(13)]], const device bfloat* cv2 [[buffer(14)]],
    const device bfloat* cv3 [[buffer(15)]], const device bfloat* cv4 [[buffer(16)]], const device bfloat* cv5 [[buffer(17)]],
    const device bfloat* cv6 [[buffer(18)]], const device bfloat* cv7 [[buffer(19)]], constant DsparkAttentionArgs& a [[buffer(20)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float smem[5184];
  threadgroup bfloat* sh_q = (threadgroup bfloat*)smem;
  threadgroup int* sh_key = (threadgroup int*)(smem + 1536);
  const uint tid = sg * 32u + lane, kv = tg.x, s = tg.y, qb = sg / 12u, kw = sg % 12u;
  const uint committed = a.committed[s], ring = a.ring[s], from = a.from[s];
  const uint first = max(from, a.local && committed > 1024u ? committed - 1024u : 0u);
  const uint retained = committed - first, count = retained + 8u;
  const device bfloat* context_k = s == 0u ? ck0 : s == 1u ? ck1 : s == 2u ? ck2 : s == 3u ? ck3 : s == 4u ? ck4 : s == 5u ? ck5 : s == 6u ? ck6 : ck7;
  const device bfloat* context_v = s == 0u ? cv0 : s == 1u ? cv1 : s == 2u ? cv2 : s == 3u ? cv3 : s == 4u ? cv4 : s == 5u ? cv5 : s == 6u ? cv6 : cv7;
  {
    const uint qi = tid / 32u, d4 = (tid % 32u) * 4u, row = qi / 3u, qh = kv * 3u + (qi % 3u);
    *(threadgroup bfloat4*)(sh_q + qi * 128u + d4) = *(const device bfloat4*)(q + uint64_t((s * 8u + row) * 24u + qh) * 128u + d4);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<bfloat> fq[8];
  if (qb == 0u) { TF_UNROLL for (short t = 0; t < 8; t++) tfp::frag_get_tg(fq[t], sh_q, 128, 0, 16 * t, home); }
  else {
    TF_UNROLL for (short t = 0; t < 8; t++) TF_UNROLL for (short e = 0; e < 8; e++) {
      const int rr = 16 + home.y + (e >> 2) * 8;
      fq[t][e] = rr < 24 ? sh_q[rr * 128 + 16 * t + home.x + TF_COL(e)] : bfloat(0.0f);
    }
  }
  tfp::frag<float> acc[4][2];
  TF_UNROLL for (short c = 0; c < 4; c++) { acc[c][0] = tfp::frag<float>(0); acc[c][1] = tfp::frag<float>(0); }
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  const device bfloat *bk = block_k + uint64_t(s) * 8192u + kv * 128u, *bv = block_v + uint64_t(s) * 8192u + kv * 128u;
  const device bfloat *ck = context_k + kv * 128u, *cv = context_v + kv * 128u;
  for (uint base = 0; base < count; base += 384u) {
    if (qb == 0u) {
      const uint j = base + kw * 32u + lane;
      sh_key[kw * 32u + lane] = j < count ? (j < retained ? int((first + j) % ring) : -2 - int(j - retained)) : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (base + kw * 32u < count) {
      tfp::frag<float> lo_s = tfp::frag<float>(0), hi_s = tfp::frag<float>(0);
      TF_UNROLL for (short t = 0; t < 8; t++) {
        tfp::frag<bfloat> b0, b1;
#ifndef TF_SIMD_FRAGS
        const device bfloat* kr[4];
        TF_UNROLL for (short i = 0; i < 4; i++) {
          const int kp = sh_key[kw * 32u + 8 * i + home.y];
          kr[i] = (kp >= 0 ? ck + uint64_t(kp) * 1024u : (kp <= -2 ? bk + uint64_t(-2 - kp) * 1024u : bk)) + home.x;
        }
        b0 = tfp::frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
        b1 = tfp::frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
#else
        TF_UNROLL for (short e = 0; e < 8; e++) {
          const int kp0 = sh_key[kw * 32u + tfp::t_key(e, home)], kp1 = sh_key[kw * 32u + 16 + tfp::t_key(e, home)];
          const int dd = 16 * t + tfp::t_dim(e, home);
          b0[e] = kp0 >= 0 ? ck[uint64_t(kp0) * 1024u + dd] : (kp0 <= -2 ? bk[uint64_t(-2 - kp0) * 1024u + dd] : bfloat(0));
          b1[e] = kp1 >= 0 ? ck[uint64_t(kp1) * 1024u + dd] : (kp1 <= -2 ? bk[uint64_t(-2 - kp1) * 1024u + dd] : bfloat(0));
        }
#endif
        tfp::mma_16x32<false, true>(lo_s, hi_s, fq[t], b0, b1);
      }
      float sc[16], factor[2], bsum[2] = {0.0f, 0.0f};
      bool vmask[16];
      TF_UNROLL for (short i = 0; i < 16; i++) {
        const float v = (i < 8 ? lo_s[i] : hi_s[i & 7]) * DSP_SCALE;
        const int kp = sh_key[kw * 32u + (i < 8 ? 0 : 16) + tfp::p_col(i & 7, home)];
        vmask[i] = kp != -1; sc[i] = vmask[i] ? v : -INFINITY;
      }
      TF_UNROLL for (short h = 0; h < 2; h++) {
        float bm = -INFINITY;
        TF_UNROLL for (short i = 0; i < 4; i++) {
          if (vmask[4 * h + i]) bm = dsp_nan_max(bm, sc[4 * h + i]);
          if (vmask[8 + 4 * h + i]) bm = dsp_nan_max(bm, sc[8 + 4 * h + i]);
        }
        const float om = m[h], nm = dsp_nan_max(om, dsp_row_max(bm));
        factor[h] = isnan(nm) ? NAN : (om == -INFINITY ? 0.0f : (nm == om ? 1.0f : precise::exp(om - nm)));
        m[h] = nm;
      }
      tfp::frag<bfloat> p1[2], p2[2], p3[2];
      TF_UNROLL for (short i = 0; i < 16; i++) {
        const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
        const float pr = vmask[i] ? precise::exp(sc[i] - m[h]) : 0.0f;
        bsum[h] += pr;
        const bfloat x = bfloat(pr); const float r1 = pr - float(x); const bfloat y = bfloat(r1);
        p1[kb][e] = x; p2[kb][e] = y; p3[kb][e] = bfloat(r1 - float(y));
      }
      TF_UNROLL for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + dsp_row_sum(bsum[h]);
      TF_UNROLL for (short c = 0; c < 4; c++) TF_UNROLL for (short e = 0; e < 8; e++) { acc[c][0][e] *= factor[e >> 2]; acc[c][1][e] *= factor[e >> 2]; }
      TF_UNROLL for (short kb = 0; kb < 2; kb++) {
#ifndef TF_SIMD_FRAGS
        const int kpa = sh_key[kw * 32u + 16 * kb + home.y], kpb = sh_key[kw * 32u + 16 * kb + home.y + 8];
        const device bfloat* va = (kpa >= 0 ? cv + uint64_t(kpa) * 1024u : (kpa <= -2 ? bv + uint64_t(-2 - kpa) * 1024u : bv)) + home.x;
        const device bfloat* vb = (kpb >= 0 ? cv + uint64_t(kpb) * 1024u : (kpb <= -2 ? bv + uint64_t(-2 - kpb) * 1024u : bv)) + home.x;
#endif
        TF_UNROLL for (short c = 0; c < 4; c++) {
          tfp::frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
          v0 = tfp::frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
          v1 = tfp::frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
#else
          TF_UNROLL for (short e = 0; e < 8; e++) {
            const int kp = sh_key[kw * 32u + 16 * kb + tfp::p_row(e, home)], dd = 32 * c + tfp::p_col(e, home);
            v0[e] = kp >= 0 ? cv[uint64_t(kp) * 1024u + dd] : (kp <= -2 ? bv[uint64_t(-2 - kp) * 1024u + dd] : bfloat(0));
            v1[e] = kp >= 0 ? cv[uint64_t(kp) * 1024u + dd + 16] : (kp <= -2 ? bv[uint64_t(-2 - kp) * 1024u + dd + 16] : bfloat(0));
          }
#endif
          tfp::mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
          tfp::mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
          tfp::mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float *sh_m = smem, *sh_l = smem + 288, *sh_part = smem + 576;
  TF_UNROLL for (short h = 0; h < 2; h++) {
    const uint qr = qb * 16u + uint(home.y) + uint(h) * 8u;
    if (home.x == 0 && qr < 24u) { sh_m[kw * 24u + qr] = m[h]; sh_l[kw * 24u + qr] = l[h]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w_kw[2] = {0.0f, 0.0f};
  TF_UNROLL for (short h = 0; h < 2; h++) {
    const uint qr = qb * 16u + uint(home.y) + uint(h) * 8u;
    if (qr < 24u) {
      float gm = -INFINITY; for (uint k = 0; k < 12u; k++) gm = dsp_nan_max(gm, sh_m[k * 24u + qr]);
      if (isnan(gm)) { w_kw[h] = kw == 0u ? NAN : 0.0f; } else {
        float gl = 0.0f;
        for (uint k = 0; k < 12u; k++) if (sh_l[k * 24u + qr] > 0.0f) gl += sh_l[k * 24u + qr] * precise::exp(sh_m[k * 24u + qr] - gm);
        w_kw[h] = (gl > 0.0f && l[h] > 0.0f) ? precise::exp(m[h] - gm) / gl : 0.0f;
      }
    }
  }
  TF_UNROLL for (short c = 0; c < 4; c++) TF_UNROLL for (short j = 0; j < 2; j++) {
    TF_UNROLL for (short e = 0; e < 8; e++) {
      const uint qr = qb * 16u + uint(home.y) + uint(e >> 2) * 8u;
      if (qr < 24u) sh_part[(kw * 24u + qr) * 16u + uint(home.x + TF_COL(e))] = isnan(w_kw[e >> 2]) ? NAN : acc[c][j][e] * w_kw[e >> 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 384u) {
      const uint qr = tid / 16u, d_in = tid % 16u, row = qr / 3u, qh = kv * 3u + (qr % 3u), dd = 32u * uint(c) + 16u * uint(j) + d_in;
      float tot = 0.0f; for (uint k = 0; k < 12u; k++) tot += sh_part[(k * 24u + qr) * 16u + d_in];
      out[uint64_t((s * 8u + row) * 24u + qh) * 128u + dd] = bfloat(tot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_residual_norm(const device bfloat* x [[buffer(0)]], const device bfloat* residual [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* new_residual [[buffer(3)]], device bfloat* normalized [[buffer(4)]],
    constant DsparkResidualArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_sum[8];
  float reg[9], sum = 0.0f;
  for (uint n = 0; n < 9u; ++n) {
    const uint d = n * 256u + t, at = row * DSP_H + d;
    const float v = a.init ? float(x[at]) : float(x[at]) + float(residual[at]);
    reg[n] = v; sum += v * v;
  }
  const float part = simd_sum(sum);
  if (lane == 0u) sh_sum[sg] = part;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f; for (uint k = 0; k < 8u; ++k) tot += sh_sum[k];
  const float scale = rsqrt(tot / float(DSP_H) + DSP_EPS);
  for (uint n = 0; n < 9u; ++n) {
    const uint d = n * 256u + t, at = row * DSP_H + d;
    const float sigma = reg[n];
    new_residual[at] = bfloat(sigma);
    normalized[at] = bfloat(sigma * scale * float(weight[d]));
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_swiglu(const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant DsparkCountArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float gate = float(g[i]);
  out[i] = bfloat((gate / (1.0f + exp(-gate))) * float(u[i]));
}

[[max_total_threads_per_threadgroup(64)]]
kernel void tf_ink2_dspark_kappa(const device bfloat* y [[buffer(0)]], const device bfloat* f0 [[buffer(1)]],
    const device bfloat* f1 [[buffer(2)]], const device bfloat* f2 [[buffer(3)]], const device bfloat* f3 [[buffer(4)]],
    const device bfloat* f4 [[buffer(5)]], const device bfloat* f5 [[buffer(6)]], const device bfloat* f6 [[buffer(7)]],
    const device bfloat* f7 [[buffer(8)]], const device bfloat* w_kappa [[buffer(9)]], const device bfloat* bias [[buffer(10)]],
    device float* out [[buffer(11)]], constant DsparkKappaArgs& a [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint tid = sg * 32u + lane, ch = tg.x * 64u + tid, s = tg.y;
  const device bfloat* f = s == 0u ? f0 : s == 1u ? f1 : s == 2u ? f2 : s == 3u ? f3 : s == 4u ? f4 : s == 5u ? f5 : s == 6u ? f6 : f7;
  const device bfloat* ys = y + uint64_t(s * a.stride_s + a.row_off) * DSP_H;
  float sy[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint d = 0; d < DSP_H; ++d) {
    const float wy = float(w_kappa[uint64_t(d) * DSP_RNN + ch]);
    for (uint r = 0; r < a.rows; ++r) sy[r] += float(ys[r * DSP_H + d]) * wy;
  }
  float sf = float(bias[ch]);
  for (uint d = 0; d < DSP_H; ++d) sf += float(f[d]) * float(w_kappa[uint64_t(DSP_H + d) * DSP_RNN + ch]);
  for (uint r = 0; r < a.rows; ++r) out[uint64_t(r * a.streams + s) * DSP_RNN + ch] = sy[r] + sf;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_recurrent(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]],
    const device bfloat* rnn_rec [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]],
    device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], const device uint* held [[buffer(7)]],
    constant DsparkTokenArgs& a [[buffer(8)]], uint s [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
  const uint tok = a.use_held ? held[s * 8u + a.step] : a.token;
  threadgroup bfloat sh_s[256], sh_m[256];
  sh_s[i] = state[s * DSP_RANK + i];
  sh_m[i] = markov[uint64_t(tok) * DSP_RANK + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float values[3];
  for (uint part = 0; part < 3u; ++part) {
    const uint channel = part * DSP_RANK + i;
    float sum = 0.0f;
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(sh_s[d]) * float(rnn_rec[uint64_t(d) * DSP_RNN + channel]);
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(sh_m[d]) * float(rnn_rec[uint64_t(DSP_RANK + d) * DSP_RNN + channel]);
    sum += kappa[s * DSP_RNN + channel];
    const bfloat rounded = bfloat(sum);
    z[s * DSP_RNN + channel] = rounded;
    values[part] = float(rounded);
  }
  const bfloat gamma = bfloat(1.0f / (1.0f + exp(-values[0])));
  const bfloat candidate = bfloat(tanh(values[1]));
  const bfloat keep = bfloat(1.0f - float(gamma));
  const bfloat left = bfloat(float(gamma) * float(sh_s[i]));
  const bfloat right = bfloat(float(keep) * float(candidate));
  next_state[s * DSP_RANK + i] = bfloat(float(left) + float(right));
  lambda[s * DSP_RANK + i] = bfloat(tanh(values[2]));
}

static inline void dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(val); }

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_base(const device bfloat* unembed [[buffer(0)]], const device bfloat* y [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant DsparkBaseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat sh_x[16][32], sh_w[32][32];
  threadgroup float sh_c[16][32];
  const uint tid = sg * 32u + lane;
  if (a.rows < 4u) {
    const uint v = tg.x * 8u + sg;
    if (v >= a.vocab) return;
    const device bfloat* w = unembed + uint64_t(v) * DSP_H;
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (uint i = 8u * lane; i < DSP_H; i += 256u) {
      const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4u));
      for (uint r = 0; r < a.rows; ++r) {
        const device bfloat* xr = y + uint64_t(r) * DSP_H + i;
        acc[r] += dot(w0, float4(*(const device bfloat4*)xr)) + dot(w1, float4(*(const device bfloat4*)(xr + 4u)));
      }
    }
    for (uint r = 0; r < a.rows; ++r) { const float sum = simd_sum(acc[r]); if (lane == 0u) dspark_store_base(out, uint64_t(r) * a.vocab + v, sum); }
    return;
  }
  const uint r0 = tg.y * 16u, c0 = tg.x * 32u, r_sg = sg / 4u, c_sg = sg % 4u;
  if (r0 >= a.rows || c0 >= a.vocab) return;
  simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);
  for (uint k = 0; k < DSP_H; k += 32u) {
    if (tid < 128u) {
      const uint rr = tid / 8u, cc = (tid % 8u) * 4u;
      *(threadgroup bfloat4*)(&sh_x[rr][cc]) = r0 + rr < a.rows ? *(const device bfloat4*)(y + uint64_t(r0 + rr) * DSP_H + k + cc) : bfloat4(0);
    }
    {
      const uint rr = tid / 8u, cc = (tid % 8u) * 4u;
      *(threadgroup bfloat4*)(&sh_w[rr][cc]) = c0 + rr < a.vocab ? *(const device bfloat4*)(unembed + uint64_t(c0 + rr) * DSP_H + k + cc) : bfloat4(0);
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
    if (r0 + rr < a.rows && c0 + cc < a.vocab) dspark_store_base(out, uint64_t(r0 + rr) * a.vocab + c0 + cc, sh_c[rr][cc]);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_logits(const device bfloat* base [[buffer(0)]], const device bfloat* phi [[buffer(1)]],
    const device bfloat* beta [[buffer(2)]], const device bfloat* lambda [[buffer(3)]], device bfloat* out [[buffer(4)]],
    device DsparkGroupStat* stats [[buffer(5)]], device uint* held [[buffer(6)]], device float* probs [[buffer(7)]],
    const device uint* sample_tok [[buffer(8)]], constant DsparkLogitsArgs& a [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_m[8], sh_l[8], sh_bv[8];
  threadgroup uint sh_bi[8];
  if (a.mode != 0u) {
    const uint s = tg, groups = (a.count + 255u) / 256u;
    float m = -INFINITY, l = 0.0f, bv = -INFINITY; uint bi = 0xffffffffu;
    for (uint g = tid; g < groups; g += 256u) {
      const DsparkGroupStat st = stats[uint64_t(s) * groups + g];
      const float nm = dsp_nan_max(m, st.m);
      l = isnan(nm) ? NAN : (st.m == -INFINITY ? l : l * (m == -INFINITY ? 0.0f : exp(m - nm)) + st.l * exp(st.m - nm));
      m = nm;
      if (isnan(st.best_v) || isnan(bv)) { bv = NAN; }
      else if (st.best_v > bv || (st.best_v == bv && st.best_id < bi)) { bv = st.best_v; bi = st.best_id; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float om = simd_shuffle_xor(m, off), ol = simd_shuffle_xor(l, off), obv = simd_shuffle_xor(bv, off);
      const uint obi = simd_shuffle_xor(bi, off); const float nm = dsp_nan_max(m, om);
      l = isnan(nm) ? NAN : ((m == -INFINITY ? 0.0f : l * exp(m - nm)) + (om == -INFINITY ? 0.0f : ol * exp(om - nm)));
      m = nm;
      if (isnan(bv) || isnan(obv)) { bv = NAN; } else if (obv > bv || (obv == bv && obi < bi)) { bv = obv; bi = obi; }
    }
    if (lane == 0u) { sh_m[sg] = m; sh_l[sg] = l; sh_bv[sg] = bv; sh_bi[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gm, gl, gbv; uint gbi;
      dsp_merge_tg(sh_m, sh_l, sh_bv, sh_bi, gm, gl, gbv, gbi);
      const bool sampled = ((a.flags >> s) & 1u) != 0u;
      const uint tok = sampled ? sample_tok[s] : (isnan(gm) ? a.count : gbi);
      const float pv = sampled ? (tok < a.count ? float(out[uint64_t(s) * a.count + tok]) : -INFINITY) : gbv;
      held[s * 8u + a.step + 1u] = tok;
      probs[s * 8u + a.step] = (gl > 0.0f && isfinite(gm) && isfinite(pv)) ? min(1.0f, max(0.0f, exp(pv - gm) / gl)) : 0.0f;
    }
    return;
  }
  threadgroup bfloat sh_lam[8][256];
  const uint S = a.streams, v = tg * 256u + tid, groups = (a.count + 255u) / 256u;
  const bool active = v < a.count;
  for (uint i = tid; i < S * 256u; i += 256u) sh_lam[i >> 8][i & 255u] = lambda[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dots[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (active) {
    for (uint d = 0; d < DSP_RANK; ++d) {
      const float p = float(phi[uint64_t(d) * a.phi_stride + v]);
      for (uint s = 0; s < S; ++s) dots[s] += p * float(sh_lam[s][d]);
    }
  }
  const float bval = active ? float(beta[v]) : 0.0f;
  for (uint s = 0; s < S; ++s) {
    float lv = -INFINITY; uint li = 0xffffffffu;
    if (active) {
      const bfloat bias = bfloat(bval + dots[s]);
      const bfloat logit = bfloat(float(base[uint64_t(s) * a.base_stride + v]) + float(bias));
      const bfloat capped = bfloat(30.0f * tanh(float(logit) / 30.0f));
      out[uint64_t(s) * a.count + v] = capped;
      lv = float(capped); li = v;
    }
    float lm = lv, lbv = lv; uint lbi = li;
    for (ushort off = 16; off > 0; off >>= 1) {
      lm = dsp_nan_max(lm, simd_shuffle_xor(lm, off));
      const float obv = simd_shuffle_xor(lbv, off); const uint obi = simd_shuffle_xor(lbi, off);
      if (isnan(lbv) || isnan(obv)) { lbv = NAN; } else if (obv > lbv || (obv == lbv && obi < lbi)) { lbv = obv; lbi = obi; }
    }
    const float ls = simd_sum(active ? (isnan(lm) ? NAN : exp(lv - lm)) : 0.0f);
    if (lane == 0u) { sh_m[sg] = lm; sh_l[sg] = ls; sh_bv[sg] = lbv; sh_bi[sg] = lbi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gm, gl, gbv; uint gbi;
      dsp_merge_tg(sh_m, sh_l, sh_bv, sh_bi, gm, gl, gbv, gbi);
      stats[uint64_t(s) * groups + tg] = {gm, gl, gbv, gbi};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct DsparkLogitsQ4Args { uint rows, vocab, k, softcap; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_logits_q4(const device uint* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]], const device bfloat* y [[buffer(3)]], device bfloat* out [[buffer(4)]],
    constant DsparkLogitsQ4Args& a [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint id = tg.x * 8u + sg, row0 = tg.y * 8u;
  if (id >= a.vocab || row0 >= a.rows) return;
  const uint row_count = min(8u, a.rows - row0), words_per_row = a.k / 8u, groups_per_row = a.k / 64u;
  const device uint* wr = w + size_t(id) * words_per_row;
  const device bfloat *sr = scales + size_t(id) * groups_per_row, *br = biases + size_t(id) * groups_per_row;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
    const uint word = wr[w_idx], g = w_idx >> 3u, d0 = w_idx << 3u;
    const float s = float(sr[g]), b = float(br[g]);
    float wv[8];
    TF_UNROLL for (uint e = 0; e < 8u; ++e) wv[e] = s * float((word >> (4u * e)) & 0xfu) + b;
    TF_UNROLL for (uint r = 0; r < 8u; ++r) if (r < row_count) {
      const device bfloat* xr = y + size_t(row0 + r) * a.k + d0;
      TF_UNROLL for (uint e = 0; e < 8u; ++e) acc[r] += wv[e] * float(xr[e]);
    }
  }
  TF_UNROLL for (uint r = 0; r < 8u; ++r) if (r < row_count) {
    const float total = simd_sum(acc[r]);
    if (lane == 0u) {
      const bfloat raw = bfloat(total);
      out[size_t(row0 + r) * a.vocab + id] = a.softcap
          ? bfloat(30.0f * precise::tanh(precise::divide(float(raw), 30.0f))) : raw;
    }
  }
}
