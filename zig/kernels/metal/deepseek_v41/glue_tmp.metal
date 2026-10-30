// DeepSeek V4.1's phase-1 glue on Metal, the bits glue.metal, compress.metal and the fused kernels are checked against.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#define TF_G_TG 256u

// A float's adjacent-lane butterfly: the same tree whatever the launch.
inline float tf_g_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

// A threadgroup of 256's sum: each simdgroup's tree, then the eight in order (every thread gets the total).
inline float tf_g_sum(float v, threadgroup float* part, uint lane, uint sg) {
  v = tf_g_tree(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = part[0];
  for (uint k = 1; k < TF_G_TG / 32; ++k) t += part[k];
  return t;
}

inline float tf_g_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

// ceil(log2(v)) of a positive normal float from its bits, as the reference's fast_log2_ceil.
inline int tf_g_log2c(float v) {
  const uint b = as_type<uint>(v);
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

// A magnitude of at most 448 as its nearest E4M3 code, ties to even.
inline uint tf_g_e4m3(float a) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

inline float tf_g_e4m3v(uint c) { return float(as_type<half>(ushort((c & 0x7Fu) << 7))) * 256.0f; }

// A magnitude of at most 6 as its nearest E2M1 code, ties to the even code.
inline uint tf_g_e2m1(float m) {
  if (m <= 0.25f) return 0u;
  if (m < 0.75f) return 1u;
  if (m <= 1.25f) return 2u;
  if (m < 1.75f) return 3u;
  if (m <= 2.5f) return 4u;
  if (m < 3.5f) return 5u;
  if (m <= 5.0f) return 6u;
  return 7u;
}

inline float tf_g_e2m1v(uint c) {
  const uint e = (c >> 1) & 3u, m = c & 1u;
  return e == 0u ? float(m) * 0.5f : ldexp(float(m) + 2.0f, int(e) - 2);
}

inline float tf_g_clip(float x, float lo, float hi) { return x != x ? x : clamp(x, lo, hi); }

struct TfGNormArgs {
  uint dim, in_stride, out_stride;
  float eps;
};

// RMSNorm of a bf16 row: x * rsqrt(mean(x^2) + eps), times the fp32 weight, rounded to bf16 once.
kernel void tf_dsv41_g_rmsnorm(const device bfloat* X [[buffer(0)]], const device float* Wt [[buffer(1)]],
                               device bfloat* Y [[buffer(2)]], constant TfGNormArgs& a [[buffer(3)]],
                               uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[TF_G_TG / 32];
  const device bfloat* x = X + ulong(r) * a.in_stride;
  float s = 0.0f;
  for (uint i = t; i < a.dim; i += TF_G_TG) s = s + float(x[i]) * float(x[i]);
  const float tot = tf_g_sum(s, part, lane, sg);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint i = t; i < a.dim; i += TF_G_TG) Y[ulong(r) * a.out_stride + i] = bfloat(Wt[i] * (float(x[i]) * inv));
}

struct TfGQuantArgs {
  uint k, nb, na, sa, sb, ta, tb;
};

// act_quant's round trip of a 32-value block at (a, b): E4M3 codes over a power-of-two scale, back to bf16.
kernel void tf_dsv41_g_actq(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                            constant TfGQuantArgs& a [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x * 32u >= a.k || g.y >= a.nb || g.z >= a.na) return;
  const device bfloat* x = X + ulong(g.z) * a.sa + ulong(g.y) * a.sb + g.x * 32u;
  device bfloat* y = Y + ulong(g.z) * a.ta + ulong(g.y) * a.tb + g.x * 32u;
  float v[32], amax = 0.0f;
  bool bad = false;
  for (uint j = 0; j < 32; ++j) {
    v[j] = float(x[j]);
    bad |= !isfinite(v[j]);
    amax = max(amax, fabs(v[j]));
  }
  const int e = tf_g_log2c(max(amax, 1e-4f) * (1.0f / 448.0f));
  for (uint j = 0; j < 32; ++j) {
    const float q = tf_g_e4m3v(tf_g_e4m3(min(fabs(v[j]) * tf_g_pow2(-e), 448.0f))) * tf_g_pow2(e);
    y[j] = bfloat(bad ? NAN : (signbit(v[j]) ? -q : q));
  }
}

struct TfGRopeArgs {
  uint pairs, heads, head_stride, row_stride, off;
};

// A head copied with its dims from `off` on rotated by the table at the row's position (pairs as complex numbers).
kernel void tf_dsv41_g_rope(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                            const device int* P [[buffer(2)]], const device float* CS [[buffer(3)]],
                            constant TfGRopeArgs& a [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= a.pairs || g.y >= a.heads) return;
  const int pos = P[g.z];
  if (pos < 0) return;
  const ulong at = ulong(g.z) * a.row_stride + ulong(g.y) * a.head_stride + 2u * g.x;
  const float x0 = float(X[at]), x1 = float(X[at + 1]);
  if (2u * g.x < a.off) {
    Y[at] = bfloat(x0);
    Y[at + 1] = bfloat(x1);
    return;
  }
  const uint pr = g.x - a.off / 2u;
  const float c = CS[ulong(pos) * 64u + pr], s = CS[ulong(pos) * 64u + 32u + pr];
  Y[at] = bfloat(x0 * c - x1 * s);
  Y[at + 1] = bfloat(x0 * s + x1 * c);
}

struct TfGFp4Args {
  uint blocks, in_stride, code_row, scale_row, scale_off, use_dest;
};

// fp4_act_quant's packing: E2M1 codes (low nibble first) over a power-of-two scale a 32-value block; dest < 0 skips.
kernel void tf_dsv41_g_fp4pack(const device bfloat* X [[buffer(0)]], device uchar* C [[buffer(1)]],
                               device uchar* S [[buffer(2)]], const device int* D [[buffer(3)]],
                               constant TfGFp4Args& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.blocks) return;
  const int at = a.use_dest != 0u ? D[g.y] : int(g.y);
  if (at < 0) return;
  const device bfloat* x = X + ulong(g.y) * a.in_stride + 32u * g.x;
  float v[32], amax = 0.0f;
  for (uint j = 0; j < 32; ++j) v[j] = float(x[j]), amax = max(amax, fabs(v[j]));
  const int e = tf_g_log2c(max(amax, 6.0f * tf_g_pow2(-126)) * (1.0f / 6.0f));
  device uchar* c = C + ulong(at) * a.code_row + 16u * g.x;
  for (uint j = 0; j < 16; ++j) {
    const uint lo = tf_g_e2m1(min(fabs(v[2 * j]) * tf_g_pow2(-e), 6.0f)) | (signbit(v[2 * j]) ? 8u : 0u);
    const uint hi = tf_g_e2m1(min(fabs(v[2 * j + 1]) * tf_g_pow2(-e), 6.0f)) | (signbit(v[2 * j + 1]) ? 8u : 0u);
    c[j] = uchar(lo | (hi << 4));
  }
  S[ulong(at) * a.scale_row + a.scale_off + g.x] = uchar(e + 127);
}

struct TfGFp4RtArgs {
  uint blocks, in_stride, out_row;
};

// The compressed KV's round trip: E2M1 over an E4M3 scale a 16-value block, back to bf16 at row dest (< 0 skips).
kernel void tf_dsv41_g_fp4rt(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                             const device int* D [[buffer(2)]], constant TfGFp4RtArgs& a [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.blocks) return;
  const int at = D[g.y];
  if (at < 0) return;
  const device bfloat* x = X + ulong(g.y) * a.in_stride + 16u * g.x;
  float v[16], amax = 0.0f;
  for (uint j = 0; j < 16; ++j) v[j] = float(x[j]), amax = max(amax, fabs(v[j]));
  const float sv = precise::divide(max(amax, 6.0f * tf_g_pow2(-9)), 6.0f);
  const float s = tf_g_e4m3v(tf_g_e4m3(min(sv, 448.0f)));
  for (uint j = 0; j < 16; ++j) {
    const float q = tf_g_e2m1v(tf_g_e2m1(min(fabs(precise::divide(v[j], s)), 6.0f))) * s;
    Y[ulong(at) * a.out_row + 16u * g.x + j] = bfloat(signbit(v[j]) ? -q : q);
  }
}

struct TfGPoolArgs {
  uint rows, dim;
};

// The ratio-2 compressor over a launch's rows in order: each row fills slot pos % 2, an odd position pools both slots.
kernel void tf_dsv41_g_pool2(const device float* KV [[buffer(0)]], const device float* SC [[buffer(1)]],
                             device float* ST [[buffer(2)]], const device int* P [[buffer(3)]],
                             device bfloat* Y [[buffer(4)]], constant TfGPoolArgs& a [[buffer(5)]],
                             uint d [[thread_position_in_grid]]) {
  if (d >= a.dim) return;
  float kv[2] = {ST[d], ST[a.dim + d]}, sc[2] = {ST[2 * a.dim + d], ST[3 * a.dim + d]};
  for (uint r = 0; r < a.rows; ++r) {
    const int pos = P[r], slot = pos & 1;
    kv[slot] = KV[ulong(r) * a.dim + d];
    sc[slot] = SC[ulong(r) * a.dim + d];
    if (slot == 1) {
      const float m = max(sc[0], sc[1]);
      const float e0 = precise::exp(sc[0] - m), e1 = precise::exp(sc[1] - m);
      const float sum = e0 + e1;
      const float pooled = kv[0] * precise::divide(e0, sum) + kv[1] * precise::divide(e1, sum);
      Y[ulong(r) * a.dim + d] = bfloat(pooled);
    }
  }
  ST[d] = kv[0], ST[a.dim + d] = kv[1], ST[2 * a.dim + d] = sc[0], ST[3 * a.dim + d] = sc[1];
}

struct TfGMvArgs {
  uint k, n;
};

// A bf16-weight matvec, one simdgroup an output: lane chains over K in steps of 32, then the butterfly.
template <typename T>
kernel void tf_dsv41_g_mv(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                          device T* Y [[buffer(2)]], constant TfGMvArgs& a [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x, r = tg.y;
  const device bfloat* x = X + ulong(r) * a.k;
  const device bfloat* w = W + ulong(n) * a.k;
  float acc = 0.0f;
  for (uint i = lane; i < a.k; i += 32u) acc = fma(float(x[i]), float(w[i]), acc);
  acc = tf_g_tree(acc);
  if (lane == 0) Y[ulong(r) * a.n + n] = T(acc);
}

template [[host_name("tf_dsv41_g_mv_f32")]] [[kernel]] decltype(tf_dsv41_g_mv<float>) tf_dsv41_g_mv<float>;
template [[host_name("tf_dsv41_g_mv_bf16")]] [[kernel]] decltype(tf_dsv41_g_mv<bfloat>) tf_dsv41_g_mv<bfloat>;

struct TfGCandArgs {
  uint keys, blocks, topk;
};

// The candidate blocks of a row: block maxima, the newest block pinned, the top `topk` (ties to the lower block).
kernel void tf_dsv41_g_cand(const device float* S [[buffer(0)]], const device int* V [[buffer(1)]],
                            device float* B [[buffer(2)]], device uchar* M [[buffer(3)]],
                            constant TfGCandArgs& a [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  device float* bs = B + ulong(r) * a.blocks;
  const int last = (V[r] - 1) >= 0 ? (V[r] - 1) / 8 : -1;
  for (uint j = t; j < a.blocks; j += TF_G_TG) {
    float m = -INFINITY;
    for (uint i = 8u * j; i < min(8u * j + 8u, a.keys); ++i) m = max(m, S[ulong(r) * a.keys + i]);
    bs[j] = int(j) == last ? INFINITY : m;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = t; j < a.blocks; j += TF_G_TG) {
    const float sj = bs[j];
    uint rank = 0;
    for (uint i = 0; i < a.blocks; ++i) rank += (bs[i] > sj || (bs[i] == sj && i < j)) ? 1u : 0u;
    M[ulong(r) * a.blocks + j] = (sj > -INFINITY && rank < a.topk) ? 1 : 0;
  }
}

struct TfGHcArgs {
  uint dim, iters;
  float eps, hc_eps, alpha;
};

// The hyper-connection mixes of a row: its normed stream through hc_fn, then pre, post and the Sinkhorn comb.
kernel void tf_dsv41_g_hcmix(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                             const device float* SCALE [[buffer(2)]], const device float* BASE [[buffer(3)]],
                             device float* PRE [[buffer(4)]], device float* POST [[buffer(5)]],
                             device float* COMB [[buffer(6)]], constant TfGHcArgs& a [[buffer(7)]],
                             uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[TF_G_TG / 32];
  threadgroup float mix[24];
  const device bfloat* x = X + ulong(r) * a.dim;
  float ss = 0.0f;
  for (uint i = t; i < a.dim; i += TF_G_TG) ss = ss + float(x[i]) * float(x[i]);
  const float tot = tf_g_sum(ss, part, lane, sg);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint m = 0; m < 24; ++m) {
    float acc = 0.0f;
    for (uint i = t; i < a.dim; i += TF_G_TG) acc = fma(FN[ulong(m) * a.dim + i], float(x[i]), acc);
    const float dot = tf_g_sum(acc, part, lane, sg);
    if (t == 0) mix[m] = dot * inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float c[4][4];
  for (uint j = 0; j < 4; ++j) {
    PRE[r * 4 + j] = precise::divide(1.0f, 1.0f + precise::exp(-(mix[j] * SCALE[0] + BASE[j]))) + a.hc_eps;
    POST[r * 4 + j] = a.alpha * precise::divide(1.0f, 1.0f + precise::exp(-(mix[4 + j] * SCALE[1] + BASE[4 + j])));
    for (uint k = 0; k < 4; ++k) c[j][k] = mix[8 + 4 * j + k] * SCALE[2] + BASE[8 + 4 * j + k];
  }
  for (uint j = 0; j < 4; ++j) {
    const float mx = max(max(c[j][0], c[j][1]), max(c[j][2], c[j][3]));
    float s = 0.0f;
    for (uint k = 0; k < 4; ++k) c[j][k] = precise::exp(c[j][k] - mx), s += c[j][k];
    for (uint k = 0; k < 4; ++k) c[j][k] = precise::divide(c[j][k], s) + a.hc_eps;
  }
  for (uint it = 0; it < a.iters; ++it) {
    if (it > 0) {
      for (uint j = 0; j < 4; ++j) {
        const float s = ((c[j][0] + c[j][1]) + c[j][2]) + c[j][3];
        for (uint k = 0; k < 4; ++k) c[j][k] = precise::divide(c[j][k], s + a.hc_eps);
      }
    }
    for (uint k = 0; k < 4; ++k) {
      const float s = ((c[0][k] + c[1][k]) + c[2][k]) + c[3][k];
      for (uint j = 0; j < 4; ++j) c[j][k] = precise::divide(c[j][k], s + a.hc_eps);
    }
  }
  for (uint j = 0; j < 4; ++j)
    for (uint k = 0; k < 4; ++k) COMB[r * 16 + 4 * j + k] = c[j][k];
}

struct TfGHcArgs2 {
  uint dim, rows;
};

// hc_pre: the four streams collapsed by the row's pre mix in fp32, rounded to bf16 once.
kernel void tf_dsv41_g_hcpre(const device bfloat* X [[buffer(0)]], const device float* PRE [[buffer(1)]],
                             device bfloat* Y [[buffer(2)]], constant TfGHcArgs2& a [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const device bfloat* x = X + ulong(g.y) * 4u * a.dim + g.x;
  const device float* p = PRE + g.y * 4u;
  const float y = ((p[0] * float(x[0]) + p[1] * float(x[a.dim])) + p[2] * float(x[2u * a.dim])) + p[3] * float(x[3u * a.dim]);
  Y[ulong(g.y) * a.dim + g.x] = bfloat(y);
}

// hc_post: stream k gets post[k] times the branch plus the residual streams mixed through comb[j][k], bf16 once.
kernel void tf_dsv41_g_hcpost(const device bfloat* B [[buffer(0)]], const device bfloat* RES [[buffer(1)]],
                              const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                              device bfloat* Y [[buffer(4)]], constant TfGHcArgs2& a [[buffer(5)]],
                              uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const device bfloat* res = RES + ulong(g.y) * 4u * a.dim + g.x;
  const device float* c = COMB + g.y * 16u;
  const float b = float(B[ulong(g.y) * a.dim + g.x]);
  for (uint k = 0; k < 4; ++k) {
    const float mm = ((c[k] * float(res[0]) + c[4 + k] * float(res[a.dim])) + c[8 + k] * float(res[2u * a.dim])) +
                     c[12 + k] * float(res[3u * a.dim]);
    Y[ulong(g.y) * 4u * a.dim + k * a.dim + g.x] = bfloat(POST[g.y * 4u + k] * b + mm);
  }
}

struct TfGSwiArgs {
  uint n;
  float limit;
};

// The shared expert's clamped SwiGLU on bf16 projections, rounded to bf16 (act_quant follows separately).
kernel void tf_dsv41_g_swiglu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                              device bfloat* Y [[buffer(2)]], constant TfGSwiArgs& a [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= a.n) return;
  const float g = tf_g_clip(float(G[i]), -INFINITY, a.limit), u = tf_g_clip(float(U[i]), -a.limit, a.limit);
  Y[i] = bfloat(precise::divide(g, 1.0f + precise::exp(-g)) * u);
}

// The MoE's join: the routed fp32 sum plus the shared expert's bf16 output, rounded to bf16 once.
kernel void tf_dsv41_g_join(const device float* R [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                            device bfloat* Y [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < n) Y[i] = bfloat(R[i] + float(S[i]));
}

struct TfGScatterArgs {
  uint words, items, rep;
};

// Rows of 32-bit words copied to the destination rows D[i] (source row i / rep); a negative destination skips.
kernel void tf_dsv41_g_scatter(const device uint* X [[buffer(0)]], device uint* Y [[buffer(1)]],
                               const device int* D [[buffer(2)]], constant TfGScatterArgs& a [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.words || g.y >= a.items || D[g.y] < 0) return;
  Y[ulong(D[g.y]) * a.words + g.x] = X[ulong(g.y / a.rep) * a.words + g.x];
}

struct TfGEmbedArgs {
  uint dim, rows;
};

// The tokens' rows of the bf16 table, copied into all four streams; the pre mix starts one-hot (1, 0, 0, 0).
kernel void tf_dsv41_g_embed(const device ushort* T [[buffer(0)]], const device uint* ID [[buffer(1)]],
                             device ushort* Y [[buffer(2)]], device float* PRE [[buffer(3)]],
                             constant TfGEmbedArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const ushort v = T[ulong(ID[g.y]) * a.dim + g.x];
  for (uint k = 0; k < 4; ++k) Y[(ulong(g.y) * 4u + k) * a.dim + g.x] = v;
  if (g.x < 4) PRE[g.y * 4u + g.x] = g.x == 0 ? 1.0f : 0.0f;
}

struct TfGMeanArgs {
  uint dim, rows, stride;
};

// The mean of a row's four streams in fp32, rounded to bf16 once, at Y + row * stride (the DSpark tap).
kernel void tf_dsv41_g_hcmean(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                              constant TfGMeanArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim || g.y >= a.rows) return;
  const device bfloat* x = X + ulong(g.y) * 4u * a.dim + g.x;
  const float s = ((float(x[0]) + float(x[a.dim])) + float(x[2u * a.dim])) + float(x[3u * a.dim]);
  Y[ulong(g.y) * a.stride + g.x] = bfloat(s * 0.25f);
}
