// Idol's DINOv3 ViT-H+/16 encoder after gemm_nax.metal: bf16 rows rounded where transformers' torch code rounds them.

inline float dn_b(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float dn_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
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

// ---- y [M, N] (rows ldd apart) = bf16(x w^T + b): tfp::gemm_nax's block over bf16, fp32 sums, one rounding ----
template <bool AM>
inline void dn_linear(const device bfloat* A, const device bfloat* B, const device bfloat* bias, device bfloat* D,
                      const device int* P, uint3 tg, uint sg, uint lane) {
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  const int tx = int(tg.x), ty = int(tg.y);
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(row) * lda;
  B += long(col) * ldb;
  D += long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = 32;
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (AM || rows == 32) {
    tfp::gemm_block<bfloat, false, true, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  } else {
    tfp::gemm_block<bfloat, false, true, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  }
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const float b = float(bias[col + 16 * j + home.x + TF_COL(e)]);
      TF_UNROLL
      for (short i = 0; i < 2; i++) acc[i][j][e] = dn_b(acc[i][j][e] + b);
    }
  }
  if (AM || rows == 32) {
    tfp::gemm_store<bfloat, true>(acc, D, ldd, rows, cols, home);
  } else if (rows > 0) {
    tfp::gemm_store<bfloat, false>(acc, D, ldd, rows, cols, home);
  }
}

// N % 128 = 0 always (1,280, 3,840, 10,240); _1 when M % 64 = 0 (the 4,096 patches), _0 for the 4,101 rows.
#define DN_LINEAR(NAME, AM)                                                                                        \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                               \
      const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]], const device int32_t* P [[buffer(2)]], \
      device bfloat* D [[buffer(3)]], const device bfloat* bias [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                       \
    dn_linear<AM>(A, B, bias, D, P, tg, sg, lane);                                                               \
  }
DN_LINEAR(dn_linear_0, false)
DN_LINEAR(dn_linear_1, true)

struct DnNorm {
  uint width;
  float eps;
};

// nn.LayerNorm of a bf16 row: mean and variance in fp32, then bf16(n w + b) with bf16 w, b. A threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dn_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant DnNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = dn_sum(a, part, n, lane, sg) / float(p.width);
  float q = 0.0f;
  for (uint i = t; i < p.width; i += n) q += (float(xr[i]) - mean) * (float(xr[i]) - mean);
  const float inv = metal::precise::rsqrt(dn_sum(q, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// The encoder's last norm (bf16, as dn_norm) into h, then ImageEncoder's F.layer_norm of it in fp32 (no affine) into y.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dn_final(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* h [[buffer(3)]], device float* y [[buffer(4)]], constant DnNorm& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup float v[1280];
  const device bfloat* xr = x + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = dn_sum(a, part, n, lane, sg) / float(p.width);
  float q = 0.0f;
  for (uint i = t; i < p.width; i += n) q += (float(xr[i]) - mean) * (float(xr[i]) - mean);
  const float inv = metal::precise::rsqrt(dn_sum(q, part, n, lane, sg) / float(p.width) + p.eps);
  float a2 = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const bfloat o = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
    h[ulong(row) * p.width + i] = o;
    v[i] = float(o);
    a2 += float(o);
  }
  const float mean2 = dn_sum(a2, part, n, lane, sg) / float(p.width);
  float q2 = 0.0f;
  for (uint i = t; i < p.width; i += n) q2 += (v[i] - mean2) * (v[i] - mean2);
  const float inv2 = metal::precise::rsqrt(dn_sum(q2, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) y[ulong(row) * p.width + i] = (v[i] - mean2) * inv2;
}

struct DnHeads {
  uint rows;
  uint prefix;  // the class token and registers: no rotary
};

// q, k, v [20, rows, 64] of qkv rows; past the prefix, q and k as bf16(bf16(x cos) + bf16(rotate_half(x) sin)).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dn_heads(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* cs [[buffer(1)]], const device bfloat* sn [[buffer(2)]],
    device bfloat* Q [[buffer(3)]], device bfloat* K [[buffer(4)]], device bfloat* V [[buffer(5)]],
    constant DnHeads& p [[buffer(6)]], uint2 at [[thread_position_in_grid]]) {
  const uint r = at.y, h = at.x / 32, j = at.x % 32;
  if (h >= 20) return;
  const ulong from = ulong(r) * 3840 + h * 64 + j;
  const ulong to = (ulong(h) * p.rows + r) * 64 + j;
  V[to] = qkv[from + 2560];
  V[to + 32] = qkv[from + 2592];
  const float q0 = float(qkv[from]), q1 = float(qkv[from + 32]);
  const float k0 = float(qkv[from + 1280]), k1 = float(qkv[from + 1312]);
  if (r < p.prefix) {
    Q[to] = bfloat(q0);
    Q[to + 32] = bfloat(q1);
    K[to] = bfloat(k0);
    K[to + 32] = bfloat(k1);
    return;
  }
  const ulong t = ulong(r - p.prefix) * 64 + j;
  const float c0 = float(cs[t]), c1 = float(cs[t + 32]), s0 = float(sn[t]), s1 = float(sn[t + 32]);
  Q[to] = bfloat(dn_b(q0 * c0) + dn_b(-q1 * s0));
  Q[to + 32] = bfloat(dn_b(q1 * c1) + dn_b(q0 * s1));
  K[to] = bfloat(dn_b(k0 * c0) + dn_b(-k1 * s0));
  K[to + 32] = bfloat(dn_b(k1 * c1) + dn_b(k0 * s1));
}

// x = bf16(x + bf16(y g)) by column: LayerScale (g, bf16) then the residual, in place.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dn_scale_add(
    device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* g [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  x[i] = bfloat(float(x[i]) + dn_b(float(g[at.x]) * float(y[i])));
}

// h [rows, width] = bf16(bf16(silu(gate)) up) from gu rows [gate | up] of 2 * width: the gated MLP's middle.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dn_swiglu(
    const device bfloat* gu [[buffer(0)]], device bfloat* h [[buffer(1)]], constant uint& width [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * 2 * width + at.x;
  const float g = float(gu[i]);
  h[ulong(at.y) * width + at.x] = bfloat(dn_b(g / (1.0f + metal::precise::exp(-g))) * float(gu[i + width]));
}

// ---- Attention: tfp::attention_nax's online softmax at head width 64 over bf16 q, k, v, every row to every row ----

constant constexpr float kDnMasked = -3.402823466e+38f;
constant constexpr int kDnAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void dn_fold(thread const tfp::frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

// One 64-query block of one head over 32-key blocks, both with tails; P and F as Kernels.attention sets them.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void dn_attention(
    const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    const device int32_t* P [[buffer(3)]], const device float* F [[buffer(4)]], device bfloat* O [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 32, D = 64;
  const int h = int(tg.y), lq = P[1], lk = P[2];
  Q += (long(h) * lq + long(tg.x) * BQ) * D;
  K += long(h) * lk * D;
  V += long(h) * lk * D;
  const int os = P[0] * D;
  O += long(tg.x) * BQ * os + h * D;
  const float scale2 = F[0] * 1.44269504089f;
  const short tm = 16 * short(sg);
  Q += tm * D;
  const short2 home = tfp::frag_home(ushort(lane));

  tfp::frag<float> acc[4];
  TF_UNROLL
  for (short i = 0; i < 4; i++) acc[i] = tfp::frag<float>(0);
  float top[2] = {kDnMasked, kDnMasked}, total[2] = {0.0f, 0.0f};
  const bool last_q = int(tg.x) == P[4];
  const short q_rows = short(P[6] - tm), k_rows = short(P[7]);

  for (int kb = 0; kb < P[3]; kb++) {
    const bool last_k = kb == P[5];
    tfp::frag<float> s[2] = {tfp::frag<float>(0), tfp::frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < 4; d++) {
      tfp::frag<bfloat> q, k0, k1;
      if (last_q) {
        tfp::frag_get_in(q, Q, D, 0, 16 * d, home, q_rows, kDnAnyCols);
      } else {
        tfp::frag_get(q, Q, D, 0, 16 * d, home);
      }
      if (last_k) {
        tfp::frag_get_t_in(k0, K, D, 0, 16 * d, home, k_rows, kDnAnyCols);
        tfp::frag_get_t_in(k1, K, D, 16, 16 * d, home, k_rows, kDnAnyCols);
      } else {
        tfp::frag_get_t(k0, K, D, 0, 16 * d, home);
        tfp::frag_get_t(k1, K, D, 16, 16 * d, home);
      }
      tfp::mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] *= scale2;
    }
    if (last_k) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kDnMasked;
      }
    }
    float top_new[2] = {top[0], top[1]};
    dn_fold<true>(s[0], top_new);
    dn_fold<true>(s[1], top_new);
    tfp::frag<bfloat> pb[2];
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      // P enters P V as bf16, V's type: the denominator sums those same rounded values
      for (short e = 0; e < 8; e++) {
        pb[f][e] = bfloat(fast::exp2(s[f][e] - top_new[e >> 2]));
        s[f][e] = float(pb[f][e]);
      }
    }
    float scale_old[2];
    TF_UNROLL
    for (short r = 0; r < 2; r++) {
      scale_old[r] = fast::exp2(top[r] - top_new[r]);
      top[r] = top_new[r];
      total[r] = total[r] * scale_old[r];
    }
    dn_fold<false>(s[0], total);
    dn_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    simdgroup_barrier(mem_flags::mem_none);
    TF_UNROLL
    for (short d = 0; d < 4; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        tfp::frag<bfloat> v0, v1;
        if (last_k) {
          tfp::frag_get_in(v0, V, D, 16 * k, 16 * d, home, k_rows, kDnAnyCols);
          tfp::frag_get_in(v1, V, D, 16 * k, 16 * d + 16, home, k_rows, kDnAnyCols);
        } else {
          tfp::frag_get(v0, V, D, 16 * k, 16 * d, home);
          tfp::frag_get(v1, V, D, 16 * k, 16 * d + 16, home);
        }
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], pb[k], v0, v1);
      }
    }
    K += BK * D;
    V += BK * D;
  }

  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short r = 0; r < 2; r++) inv[r] = 1.f / total[r];
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[e >> 2];
  }
  O += tm * os;
  if (last_q) {
    if (q_rows <= 0) return;
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put_in(acc[i], O, os, 0, 16 * i, home, q_rows, kDnAnyCols);
  } else {
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put(acc[i], O, os, 0, 16 * i, home);
  }
}
