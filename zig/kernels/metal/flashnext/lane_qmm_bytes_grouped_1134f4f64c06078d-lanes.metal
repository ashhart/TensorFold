#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

inline float fz_xsum(const device bfloat16_t* X, int m, int g, int M, int K) {
  float acc = 0.0f;
  if (m < M) for (int i = 0; i < 32; i++) acc += float(X[m * K + g * 32 + i]);
  return acc;
}
[[kernel]] void custom_kernel_lane_qmm_bytes_grouped_1134f4f64c06078d_bfloat16_t_float_uint32_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device uint32_t* Wq [[buffer(2)]],
  const device bfloat16_t* SBt [[buffer(3)]],
  const device int32_t* mdims [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int TMR = 1;
  constexpr int N = 12800;
  constexpr int K = 2560;
  constexpr int NT = 32;
  constexpr int SK = 4;
  constexpr int BITS = 6;
  constexpr int TILED = 1;
  constexpr int GS = 32;

  static_assert(NT == 32, "one column per lane");
  static_assert(BITS == 5 || BITS == 6 || BITS == 8, "bytes for 5-, 6- and 8-bit weights");
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);   // simdgroup matrices: columns fn + {0, 1, 8, 9}
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  constexpr int WPG = GS * BITS / 32;                    // words per column per group: GS values x BITS bits
  constexpr int KW = K * BITS / 32;                      // words per column
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  threadgroup uint stage_all[SK * NT * (GS / 4)];        // per K slice: NT columns x GS bytes
  threadgroup uint* stage = stage_all + sg * NT * (GS / 4);
  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, NT));
  const device uint* Wv = (const device uint*)Wq;
  const int n = n0 + lane;

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint2* sbv = (const device uint2*)SBt;
  bool colok[NF][2];
  for (int f = 0; f < NF; f++) for (int h = 0; h < 2; h++) colok[f][h] = n0 + f * 16 + fn + 8 * h < N;
  for (int g = g_begin; g < g_end; g++) {
    uint w[WPG + 1];
    for (int i = 0; i <= WPG; i++) w[i] = 0;
    if (n < N) {
      const device uint* src = TILED ? Wv + ((int64_t)(threadgroup_position_in_grid.x * KG + g) * NT + lane) * WPG
                                     : Wv + (int64_t)n * KW + g * WPG;
      for (int i = 0; i < WPG; i++) w[i] = src[i];
    }
    for (int c = 0; c < GS / 4; c++) {
      const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
      uint word = w[i] >> sh;
      if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
      word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
      word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
      stage[lane * (GS / 4) + c] = word;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = uint4(colok[f][0] ? sbv[(g * N + n0 + f * 16 + fn) / 2] : uint2(0),
                            colok[f][1] ? sbv[(g * N + n0 + f * 16 + fn + 8) / 2] : uint2(0));
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening
    for (int t = 0; t < TMR; t++) {
      // the last row block can run past MP (MP % 32 == 16): those rows are never stored, and XS ends at MP
      const bool live = rb + t * 16 < MP;
      const float xs0 = live ? fz_xsum(X, rb + t * 16 + fm, g, M, K) : 0.0f;
      const float xs1 = live ? fz_xsum(X, rb + t * 16 + fm + 8, g, M, K) : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + r * 8 + f * 4 + j], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int nn = n0 + f * 16 + fn;
          if (m < M)
            for (int j = 0; j < 4; j++)
                if (colok[f][j >> 1])
                  Y[m * N + nn + (j & 1) + 8 * (j >> 1)] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }

}
