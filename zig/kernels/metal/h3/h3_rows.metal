// MiniMax H3 / FastH3 on Metal in the checkpoint's bf16: the row kernels around the products and the attention.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define H3_UNROLL _Pragma("clang loop unroll(full)")

// The model's sizes come ahead of the source at load; these are FastH3's at 864x480 and 124 frames.
#ifndef H3_HEADS
#define H3_HEADS 56
#define H3_SLOTS 18432
#define H3_TILES 288
#endif

// The modulated norm one value at a time, rounded to bf16 where the reference's bf16 arithmetic rounds.
inline float h3_mod(float x, float inv, float w, float scale, float shift) {
  return float(bfloat(float(bfloat(float(bfloat(x * inv * w)) * float(bfloat(1.0f + scale)))) + shift));
}

// Y[row0 + r, n] = B[n] + sum over k of X[r, k] W[n, k]: the latent rows into the stream. Threads [N, R, 1].
[[kernel]] void h3_rows_in(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], K = P[1], N = P[2], row0 = P[3];
  if (int(at.x) >= N || int(at.y) >= R) return;
  const device float* x = X + long(at.y) * K;
  const device float* w = W + long(at.x) * K;
  float sum = 0.0f;
  for (int k = 0; k < K; k++) sum += x[k] * w[k];
  Y[long(row0 + int(at.y)) * N + at.x] = bfloat(sum + B[at.x]);
}

// A block's modulated RMSNorm to bf16 rows, after an optional gated add of Y into X. Threadgroups [M, 1, 1] of 32.
[[kernel]] void h3_norm(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* W [[buffer(2)]],
  const device bfloat* TG [[buffer(3)]],
  const device bfloat* TAB [[buffer(4)]],
  const device int32_t* LINE [[buffer(5)]],
  const constant int32_t* P [[buffer(6)]],
  const constant float* E [[buffer(7)]],
  device bfloat* O [[buffer(8)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[1], L = P[2], row = int(tg.x);
  const int line = LINE[row];
  device bfloat* x = X + long(row) * C;
  float sq = 0.0f;
  if (P[3] >= 0) {
    const device bfloat* g = TG + (long(P[3]) * L + line) * C;
    const device bfloat* y = Y + long(row) * C;
    for (int c = int(lane); c < C; c += 32) {
      const bfloat v = bfloat(float(x[c]) + float(bfloat(float(g[c]) * float(y[c]))));
      x[c] = v;
      sq += float(v) * float(v);
    }
  } else {
    for (int c = int(lane); c < C; c += 32) sq += float(x[c]) * float(x[c]);
  }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  const device bfloat* s = TAB + (long(P[4]) * L + line) * C;
  const device bfloat* h = TAB + (long(P[5]) * L + line) * C;
  device bfloat* o = O + long(row) * C;
  for (int c = int(lane); c < C; c += 32) o[c] = bfloat(h3_mod(float(x[c]), inv, float(W[c]), float(s[c]), float(h[c])));
}

// X += TG[part, line] * Y, the last block's MLP branch. P: M, C, L, part. Threadgroups [M, 1, 1] of [32, 1, 1].
[[kernel]] void h3_gate_add(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* TG [[buffer(2)]],
  const device int32_t* LINE [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[1], L = P[2], row = int(tg.x);
  const device bfloat* g = TG + (long(P[3]) * L + LINE[row]) * C;
  device bfloat* x = X + long(row) * C;
  const device bfloat* y = Y + long(row) * C;
  for (int c = int(lane); c < C; c += 32) x[c] = bfloat(float(x[c]) + float(bfloat(float(g[c]) * float(y[c]))));
}

// The pooled branch gated into the attention output: Y[row, h] += C[h, tile of row] * G[row, h]. Threads [R, H, 1].
[[kernel]] void h3_gate_mix(
  device bfloat* Y [[buffer(0)]],
  const device bfloat* G [[buffer(1)]],
  const device float* C [[buffer(2)]],
  const device int32_t* SLOT [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1], NT = P[2];
  if (int(at.x) >= R || int(at.y) >= H) return;
  const long base = (long(at.x) * H + at.y) * 128;
  const device float* c = C + (long(at.y) * NT + SLOT[at.x] / 64) * 128;
  for (int j = 0; j < 128; j++) Y[base + j] = bfloat(float(Y[base + j]) + float(bfloat(c[j] * float(G[base + j]))));
}

// silu(gate) * value in bf16 from the fused first projection's rows [value | gate]. Threads [N, M, 1].
[[kernel]] void h3_swiglu(
  const device bfloat* F [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat* A [[buffer(2)]],
  uint3 at [[thread_position_in_grid]]) {
  const int M = P[0], N = P[1];
  if (int(at.x) >= N || int(at.y) >= M) return;
  const device bfloat* f = F + long(at.y) * 2 * N;
  const float g = float(f[N + at.x]), v = float(f[at.x]);
  const float silu = float(bfloat(g * float(bfloat(1.0f / (1.0f + exp(-g))))));
  A[long(at.y) * N + at.x] = bfloat(silu * v);
}

// The final layer's modulated RMSNorm. TAB: (T, 2 C) bf16, shift then scale. Y: (M, C) bf16.
[[kernel]] void h3_final_norm(
  const device bfloat* X [[buffer(0)]],
  const device bfloat* W [[buffer(1)]],
  const device bfloat* TAB [[buffer(2)]],
  const device int32_t* LINE [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  const constant float* E [[buffer(5)]],
  device bfloat* Y [[buffer(6)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[1], row = int(tg.x);
  const device bfloat* x = X + long(row) * C;
  const device bfloat* t = TAB + long(LINE[row]) * 2 * C;
  float sq = 0.0f;
  for (int c = int(lane); c < C; c += 32) sq += float(x[c]) * float(x[c]);
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  for (int c = int(lane); c < C; c += 32) Y[long(row) * C + c] = bfloat(h3_mod(float(x[c]), inv, float(W[c]), float(t[C + c]), float(t[c])));
}

// Y[r, n] = B[n] + sum over k of X[row0 + r, k] W[n, k]: an output head over its bf16 rows, in float.
[[kernel]] void h3_rows_out(
  const device bfloat* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device float* Y [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[1], N = P[2], row0 = P[3];
  const device bfloat* x = X + long(row0 + int(tg.y)) * K;
  const device float* w = W + long(tg.x) * K;
  float sum = 0.0f;
  for (int k = int(lane); k < K; k += 32) sum += float(x[k]) * w[k];
  sum = simd_sum(sum);
  if (lane == 0) Y[long(tg.y) * N + tg.x] = sum + B[tg.x];
}
