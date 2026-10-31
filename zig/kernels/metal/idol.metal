// Idol's denoiser steps after gemm_nax.metal: fp16 rows rounded where idol-sdk's torch code rounds them.

inline float id_h(float x) { return float(half(x)); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float id_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
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

// torch's tanh GELU of an fp16 value, computed in fp32.
inline float id_gelu(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + metal::precise::tanh(inner));
}

// ---- y [M, N] = fp16(x w^T + b) (EPI 1: then GELU, rounded again): tfp::gemm_nax's block, fp32 sums ----
template <int EPI, bool AM, bool AN>
inline void id_linear(const device half* A, const device half* B, const device half* bias, device half* D,
                      const device int* P, uint3 tg, uint sg, uint lane) {
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  const int tx = int(tg.x), ty = int(tg.y);
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(row) * lda;
  B += long(col) * ldb;
  D += long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (full) {
    tfp::gemm_block<half, false, true, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  } else {
    tfp::gemm_block<half, false, true, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  }
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int c = col + 16 * j + home.x + TF_COL(e);
      const float b = c < P[1] ? float(bias[c]) : 0.0f;
      TF_UNROLL
      for (short i = 0; i < 2; i++) {
        const float v = id_h(acc[i][j][e] + b);
        acc[i][j][e] = EPI == 1 ? id_gelu(v) : v;
      }
    }
  }
  if (full) {
    tfp::gemm_store<half, true>(acc, D, ldd, rows, cols, home);
  } else if (rows > 0 && cols > 0) {
    tfp::gemm_store<half, false>(acc, D, ldd, rows, cols, home);
  }
}

#define ID_LINEAR(NAME, EPI, AM, AN)                                                                              \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                             \
      const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], const device int32_t* P [[buffer(2)]], \
      device half* D [[buffer(3)]], const device half* bias [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                     \
    id_linear<EPI, AM, AN>(A, B, bias, D, P, tg, sg, lane);                                                     \
  }
ID_LINEAR(id_linear_00, 0, false, false)
ID_LINEAR(id_linear_01, 0, false, true)
ID_LINEAR(id_linear_10, 0, true, false)
ID_LINEAR(id_linear_11, 0, true, true)
ID_LINEAR(id_linear_gelu_00, 1, false, false)
ID_LINEAR(id_linear_gelu_01, 1, false, true)
ID_LINEAR(id_linear_gelu_10, 1, true, false)
ID_LINEAR(id_linear_gelu_11, 1, true, true)

struct IdGemm32 {
  uint M, N, K, ldy;
};

// y [M, N] (rows ldy apart) = x [M, K] w^T, fp32 FMA on 8x8 simdgroup matrices; K % 32, N % 64 (or N < 64, % 8) = 0.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void id_gemm32(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], device float* y [[buffer(2)]],
    constant IdGemm32& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BK = 32, LD = BK + 4;
  threadgroup float ta[64 * LD];
  threadgroup float tb[64 * LD];
  threadgroup float stage[4][64];
  const uint row0 = tg.y * 64, col0 = tg.x * 64;
  const uint r0 = (sg / 2) * 32, c0 = (sg % 2) * 32;
  const uint cols = min(64u, p.N - col0);
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k = 0; k < p.K; k += BK) {
    // 64 x 32 of x and of w: 2,048 values each, 16 a thread as float4s; rows past M or N read as zeros
    for (uint e = t; e < 64 * BK / 4; e += 128) {
      const uint r = e / (BK / 4), c = (e % (BK / 4)) * 4;
      const float4 va = row0 + r < p.M ? *(const device float4*)(x + ulong(row0 + r) * p.K + k + c) : float4(0.0f);
      const float4 vb = r < cols ? *(const device float4*)(w + ulong(col0 + r) * p.K + k + c) : float4(0.0f);
      *(threadgroup float4*)(ta + r * LD + c) = va;
      *(threadgroup float4*)(tb + r * LD + c) = vb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[4], b[4];
      for (uint i = 0; i < 4; i++) simdgroup_load(a[i], ta + (r0 + 8 * i) * LD + kk, LD);
      for (uint j = 0; j < 4; j++) simdgroup_load(b[j], tb + (c0 + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      const uint r = row0 + r0 + 8 * i, c = c0 + 8 * j;
      if (r >= p.M || c >= cols) continue;
      if (r + 8 <= p.M) {
        simdgroup_store(acc[i][j], y + ulong(r) * p.ldy + col0 + c, p.ldy);
      } else {
        simdgroup_store(acc[i][j], stage[sg], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32)
          if (r + e / 8 < p.M) y[ulong(r + e / 8) * p.ldy + col0 + c + e % 8] = stage[sg][e];
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

// c [M, N] fp32 = a [M, K] b^T over fp16 operands, tfp::gemm_splitk_nax's one partition (fp32 sums).
#define ID_SPLIT(NAME, AM)                                                                                       \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                             \
      const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], const device int32_t* P [[buffer(2)]], \
      device float* C [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                 \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                     \
    tfp::gemm_splitk_nax<half, false, true, AM, true, 64, 64, 256, 2, 2>(A, B, C, P, tg, sg, lane);              \
  }
ID_SPLIT(id_split_0, false)
ID_SPLIT(id_split_1, true)

struct IdNorm {
  uint width;
  float eps;
  uint mode;  // 0 plain, 1 affine (fp32 w, b), 2 modulated (fp16 1 + scale, shift)
};

// LayerNorm32 of an fp16 row in fp32, rounded to fp16; then affine in fp32, or modulated in fp16. A threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_norm(
    const device half* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    const device half* s1 [[buffer(3)]], const device half* sh [[buffer(4)]], device half* y [[buffer(5)]],
    constant IdNorm& p [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device half* xr = x + ulong(row) * p.width;
  device half* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = id_sum(a, part, n, lane, sg) / float(p.width);
  float q = 0.0f;
  for (uint i = t; i < p.width; i += n) q += (float(xr[i]) - mean) * (float(xr[i]) - mean);
  const float inv = metal::precise::rsqrt(id_sum(q, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    float v = (float(xr[i]) - mean) * inv;
    if (p.mode == 1) v = v * w[i] + b[i];
    v = id_h(v);
    if (p.mode == 2) v = id_h(id_h(v * float(s1[i])) + float(sh[i]));
    yr[i] = half(v);
  }
}

// Affine LayerNorm32 of an fp16 row as fp16 [hi | hi | lo] of 3 * width: times [w_hi | w_lo | w_hi], near-fp32 sums.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_norm_split(
    const device half* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device half* y [[buffer(3)]], constant IdNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device half* xr = x + ulong(row) * p.width;
  device half* yr = y + ulong(row) * 3 * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = id_sum(a, part, n, lane, sg) / float(p.width);
  float q = 0.0f;
  for (uint i = t; i < p.width; i += n) q += (float(xr[i]) - mean) * (float(xr[i]) - mean);
  const float inv = metal::precise::rsqrt(id_sum(q, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    const float v = (float(xr[i]) - mean) * inv * w[i] + b[i];
    const half hi = half(v);
    yr[i] = hi;
    yr[p.width + i] = hi;
    yr[2 * p.width + i] = half(v - float(hi));
  }
}

// LayerNorm of an fp16 row in fp32 kept in fp32: affine (w, b) when given, else times (1 + scale) plus shift (mode 2).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_norm32(
    const device half* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device float* y [[buffer(3)]], constant IdNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device half* xr = x + ulong(row) * p.width;
  device float* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = id_sum(a, part, n, lane, sg) / float(p.width);
  float q = 0.0f;
  for (uint i = t; i < p.width; i += n) q += (float(xr[i]) - mean) * (float(xr[i]) - mean);
  const float inv = metal::precise::rsqrt(id_sum(q, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    const float v = (float(xr[i]) - mean) * inv;
    yr[i] = p.mode == 2 ? v * (1.0f + w[i]) + b[i] : v * w[i] + b[i];
  }
}

struct IdRepo {
  uint rows;
  uint stride;  // fp32 values from one row's gate sums to the next; content sums follow gate's 128
  float inv;    // the maps' power-of-two scale undone, exactly
};

// RePo: d = final_map(silu(gate) * content), 3 per head; angles clamp_mul(d, freqs) * pi as cos, sin [rows, 16, 32].
[[max_total_threads_per_threadgroup(64)]] [[kernel]] void id_repo(
    const device float* gc [[buffer(0)]], const device float* wf [[buffer(1)]], const device float* f [[buffer(2)]],
    device float* cs [[buffer(3)]], device float* sn [[buffer(4)]], constant IdRepo& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float u[128];
  threadgroup float d[48];
  const device float* g = gc + ulong(row) * p.stride;
  for (uint i = t; i < 128; i += 64) {
    const float a = g[i] * p.inv;
    u[i] = (a / (1.0f + metal::precise::exp(-a))) * (g[128 + i] * p.inv);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 48) {
    float s = 0.0f;
    for (uint k = 0; k < 128; k++) s = fma(u[k], wf[t * 128 + k], s);
    d[t] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // freqs: 10 x, 10 y, 12 z; each pair of the 32 takes its axis's offset
  for (uint i = t; i < 512; i += 64) {
    const uint h = i / 32, j = i % 32;
    const uint axis = j < 10 ? 0 : (j < 20 ? 1 : 2);
    const float x = d[h * 3 + axis], fr = f[j];
    const float ft = metal::precise::tanh(fr);
    const float ang = (x * ft + x * (fr - ft)) * 3.14159265358979323846f;
    cs[ulong(row) * 512 + i] = metal::precise::cos(ang);
    sn[ulong(row) * 512 + i] = metal::precise::sin(ang);
  }
}

struct IdHeads {
  uint rows;
  float scale;  // sqrt(head width): MultiHeadRMSNorm's
};

// q, k of qkv rows: pairs turned by the row and head's cos, sin; F.normalize * gamma * scale; out [16, rows, 64].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_qk(
    const device half* qkv [[buffer(0)]], const device float* gq [[buffer(1)]], const device float* gk [[buffer(2)]],
    const device float* cs [[buffer(3)]], const device float* sn [[buffer(4)]], device half* Q [[buffer(5)]],
    device half* K [[buffer(6)]], device half* V [[buffer(7)]], constant IdHeads& p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= 16) return;
  const ulong from = ulong(r) * 3072 + h * 64 + lane * 2;
  const ulong at = ulong(r) * 512 + h * 32 + lane;
  const float c = cs[at], s = sn[at];
  const float q0 = float(qkv[from]), q1 = float(qkv[from + 1]);
  const float k0 = float(qkv[from + 1024]), k1 = float(qkv[from + 1025]);
  const float rq0 = id_h(q0 * c - q1 * s), rq1 = id_h(q0 * s + q1 * c);
  const float rk0 = id_h(k0 * c - k1 * s), rk1 = id_h(k0 * s + k1 * c);
  const float qn = max(metal::precise::sqrt(simd_sum(rq0 * rq0 + rq1 * rq1)), 1e-12f);
  const float kn = max(metal::precise::sqrt(simd_sum(rk0 * rk0 + rk1 * rk1)), 1e-12f);
  const uint g = h * 64 + lane * 2;
  const ulong to = (ulong(h) * p.rows + r) * 64 + lane * 2;
  Q[to] = half(rq0 / qn * gq[g] * p.scale);
  Q[to + 1] = half(rq1 / qn * gq[g + 1] * p.scale);
  K[to] = half(rk0 / kn * gk[g] * p.scale);
  K[to + 1] = half(rk1 / kn * gk[g + 1] * p.scale);
  V[to] = qkv[from + 2048];
  V[to + 1] = qkv[from + 2049];
}

// x = fp16(x + fp16(g * y)) by column; g is a modulation gate, or ones for an ungated residual.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_gate_add(
    device half* x [[buffer(0)]], const device half* y [[buffer(1)]], const device half* g [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  x[i] = half(float(x[i]) + id_h(float(g[at.x]) * float(y[i])));
}

// x = fp16(fp16(z w^T + b) + pe): the latent's input projection (16 inputs, fp32) and its fp16 position embedding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_embed(
    const device float* z [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    const device half* pe [[buffer(3)]], device half* x [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  float s = 0.0f;
  for (uint k = 0; k < 16; k++) s = fma(z[r * 16 + k], w[c * 16 + k], s);
  const ulong i = ulong(r) * 1024 + c;
  x[i] = half(float(half(s + b[c])) + float(pe[i]));
}

// y = fp16(a + ba + c + bc) by column: the two condition projections' fp32 sums and biases (each torch Linear).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_cond(
    const device float* a [[buffer(0)]], const device float* ba [[buffer(1)]], const device float* c [[buffer(2)]],
    const device float* bc [[buffer(3)]], device half* y [[buffer(4)]], constant uint& width [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = half((a[i] + ba[at.x]) + (c[i] + bc[at.x]));
}

// y = x + b by column in fp32 (an fp32 Linear's bias), in place.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void id_bias32(
    device float* y [[buffer(0)]], const device float* b [[buffer(1)]], constant uint& width [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  y[ulong(at.y) * width + at.x] += b[at.x];
}
