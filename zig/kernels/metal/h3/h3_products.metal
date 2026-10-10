// MiniMax H3's int8 products on the M5 tensor units: rows to int8, the projection body, the wide and SwiGLU forms.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define H3P_UNROLL _Pragma("clang loop unroll(full)")

#ifndef H3_INNER
#define H3_INNER 7168
#define H3_SLOTS 18496
#define H3_TILES 289
#endif

#ifndef H3P_HIDDEN
#define H3P_HIDDEN 4096
#define H3P_MLP 12288
#define H3P_WIDE_GROUP 1024
#define H3P_HEADS 32
#define H3P_ROWS 4032
#define H3P_KEYS 4126
#endif

constant constexpr int H3P_T = 128;
constant constexpr int H3P_TM = 128;                    // rows a threadgroup of the int8 products owns
constant constexpr int H3P_TN = 128;                    // output columns a threadgroup owns
constant constexpr int H3P_TK = 128;                    // channels one tensor operation consumes
constant constexpr int H3P_TQ = 64;                     // query rows one attention threadgroup owns
constant constexpr int H3P_TKEYS = 64;                  // keys in one tile of scores
constant constexpr int H3P_SG = 8;                      // simdgroups sharing one int8 product
constant constexpr int H3P_THREADS = 32 * H3P_SG;
constant constexpr int H3P_GROUPS = 32;                 // activation scale groups a row may have

// Rows to int8 with a scale per (row, group of G channels). Threadgroups [C / G, MP, 1] of [32, 1, 1].
[[kernel]] void h3p_quant_rows(
  const device bfloat* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* Q [[buffer(2)]],
  device float* XS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], C = P[1], G = P[2];
  const int row = int(tg.y), g = int(tg.x);
  const long base = long(row) * C + g * G;
  if (row >= M) {
    for (int j = int(lane); j < G; j += 32) Q[base + j] = 0;
    if (lane == 0) XS[row * (C / G) + g] = 0.0f;
    return;
  }
  float top = 0.0f;
  for (int j = int(lane); j < G; j += 32) top = max(top, abs(float(X[base + j])));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) XS[row * (C / G) + g] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int j = int(lane); j < G; j += 32) Q[base + j] = int8_t(clamp(int(rint(float(X[base + j]) * inverse)), -127, 127));
}

// Y[row, n] = (sum over groups of (int8 X . int8 W) * xscale[row, group]) * wscale[n]. Y: (M, N) bf16.
template <int N, int K, int G>
inline void h3p_i8_linear_body(const device int8_t* X, const device float* XS, const device int8_t* W,
                              const device float* WS, device bfloat* Y, uint tid, uint3 tg, threadgroup float* sc) {
  using namespace mpp::tensor_ops;
  constexpr int T = H3P_TM, TN = H3P_TN, TK = H3P_TK;
  constexpr int CAP = T * TN / H3P_THREADS;
  constexpr int M = H3P_ROWS;
  constexpr int KG = K / G, KT = G / TK, MP = (M + T - 1) / T * T;
  const int n0 = int(tg.x) * TN, r0 = int(tg.y) * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> w((device int8_t*)W, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(T, TN, TK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<H3P_SG>> op;
  auto a0 = x.slice<TK, T>(0, r0);
  auto b0 = w.slice<TN, TK>(n0, 0);
  for (int j = int(tid); j < T * KG; j += H3P_THREADS) sc[j] = XS[r0 * KG + j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  float total[CAP];
  short erow[CAP], ecol[CAP];
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    total[i] = 0.0f;
    acc[i] = 0;
    auto ids = acc.get_multidimensional_index(i);
    ecol[i] = ids[0];
    erow[i] = ids[1];
  }
  for (int g = 0; g < KG; g++) {
    for (int t = 0; t < KT; t++) {
      auto a = x.slice<TK, T>((g * KT + t) * TK, r0);
      auto b = w.slice<TN, TK>(n0, (g * KT + t) * TK);
      op.run(a, b, acc);
    }
    H3P_UNROLL
    for (ushort i = 0; i < CAP; i++) {
      total[i] = fma(float(acc[i]), sc[erow[i] * KG + g], total[i]);
      acc[i] = 0;
    }
  }
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    const int row = r0 + erow[i];
    if (row < M) Y[long(row) * N + n0 + ecol[i]] = bfloat(total[i] * WS[n0 + ecol[i]]);
  }
}

// The MLP's wide rows back to the stream's width, a scale per H3P_WIDE_GROUP channels.
[[kernel]] void h3p_i8_linear_wide(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* W [[buffer(2)]],
  const device float* WS [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sc[H3P_TM * H3P_GROUPS];
  h3p_i8_linear_body<H3P_HIDDEN, H3P_MLP, H3P_WIDE_GROUP>(X, XS, W, WS, Y, tid, tg, sc);
}

// H[row, n] = silu(X . WG) * (X . WV): the SwiGLU's two projections in one pass, a scale per row.
[[kernel]] void h3p_i8_swiglu(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* WG [[buffer(2)]],
  const device float* SG [[buffer(3)]],
  const device int8_t* WV [[buffer(4)]],
  const device float* SV [[buffer(5)]],
  device bfloat* H [[buffer(6)]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int T = H3P_TM, TN = H3P_TN, TK = H3P_TK;
  constexpr int CAP = T * TN / H3P_THREADS;
  constexpr int M = H3P_ROWS, N = H3P_MLP, K = H3P_HIDDEN;
  constexpr int MP = (M + T - 1) / T * T;
  const int n0 = int(tg.x) * TN, r0 = int(tg.y) * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wg((device int8_t*)WG, dextents<int32_t, 2>(N, K));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wv((device int8_t*)WV, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(T, TN, TK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<H3P_SG>> op;
  auto a0 = x.slice<TK, T>(0, r0);
  auto b0 = wg.slice<TN, TK>(n0, 0);
  auto gate = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  auto value = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) { gate[i] = 0; value[i] = 0; }
  for (int k0 = 0; k0 < K; k0 += TK) {
    auto a = x.slice<TK, T>(k0, r0);
    auto bg = wg.slice<TN, TK>(n0, k0);
    auto bv = wv.slice<TN, TK>(n0, k0);
    op.run(a, bg, gate);
    op.run(a, bv, value);
  }
  H3P_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    auto ids = gate.get_multidimensional_index(i);
    const int row = r0 + ids[1], n = n0 + ids[0];
    if (row >= M) continue;
    const float xs = XS[row];
    const float g = float(gate[i]) * xs * SG[n];
    const float v = float(value[i]) * xs * SV[n];
    H[long(row) * N + n] = bfloat(g / (1.0f + exp(-g)) * v);
  }
}

