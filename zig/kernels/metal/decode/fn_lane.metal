#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Flash Next's lane projection (6-bit, groups of 32, weights tiled [N/32][K/32][32 columns][6 words]) for up to 16
// rows: the recorded lane_qmm's sums, so every row's bits are its bits at any window width. Per group, a 16 x 32 x 32
// tensor op on the widened codes, then C = fma(scale, P, fma(bias, x sum, C)); the K slices' partials added in order.
// Each simdgroup has its next FZ_PF groups' weights and scales in flight (1 or 2). The includer defines FZ_N, FZ_K,
// FZ_SK, FZ_G0 and FZ_GN (the input groups summed), FZ_PF, FZ_OUT (bfloat, or float for a partial) and fz_tile (a
// threadgroup's output tile).

constant constexpr int NT = 32, GS = 32, KG = FZ_K / GS, NF = NT / 16;

struct fz_raw { uint2 w[3]; uint4 s[NF]; }; // a lane's six code words and its scale/bias quads for one group

inline fz_raw fz_lane_fetch(const device uint* W, const device uint4* SB, int tile, int g, int g_end, int n0, int lane,
                            short fn) {
  fz_raw raw;
  const bool ok = g < g_end;
  const device uint2* src = (const device uint2*)(W + ((size_t)(tile * KG + (ok ? g : 0)) * NT + lane) * 6);
  for (int i = 0; i < 3; i++) raw.w[i] = ok && n0 + lane < FZ_N ? src[i] : uint2(0);
  for (int f = 0; f < NF; f++) raw.s[f] = ok && n0 + f * 16 + fn < FZ_N ? SB[((ok ? g : 0) * FZ_N + n0 + f * 16 + fn) / 4] : uint4(0);
  return raw;
}

inline float fz_lane_xsum(const device bfloat* X, int m, int g, int M) {
  float acc = 0.0f;
  if (m < M) for (int i = 0; i < 32; i++) acc += float(X[m * FZ_K + g * 32 + i]);
  return acc;
}

// one group: widen the lane's codes to bytes in its stage row, the tensor op, then the scale/bias fmas
#define FZ_LANE_GROUP(g, raw)                                                                                      \
  {                                                                                                                \
    const uint w6[7] = {raw.w[0].x, raw.w[0].y, raw.w[1].x, raw.w[1].y, raw.w[2].x, raw.w[2].y, 0u};             \
    for (int c = 0; c < GS / 4; c++) {                                                                             \
      const int bit = 24 * c, i = bit >> 5, sh = bit & 31;                                                         \
      uint word = w6[i] >> sh;                                                                                     \
      if (sh + 24 > 32) word |= w6[i + 1] << (32 - sh);                                                            \
      word = (word & 0xFFFu) | (((word >> 12) & 0xFFFu) << 16);                                                    \
      word = (word & 0x3F003Fu) | (((word >> 6) & 0x3F003Fu) << 8);                                                \
      stage[lane * (GS / 4) + c] = word;                                                                           \
    }                                                                                                              \
    float s[NF][4], bb[NF][4];                                                                                     \
    for (int f = 0; f < NF; f++) {                                                                                 \
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(raw.s[f]);                                                  \
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }                  \
    }                                                                                                              \
    raw = fz_lane_fetch(W, sbv, tile, (g) + FZ_PF, g_end, n0, lane, fn);                                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                                                                 \
    auto a = tA.slice((g) * GS, 0);                                                                                \
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();                   \
    op.run(a, b, P);                                                                                               \
    simdgroup_barrier(mem_flags::mem_threadgroup); /* the op has read the stage before the next widening */       \
    const float xs0 = fz_lane_xsum(X, fm, (g), M);                                                                 \
    const float xs1 = fz_lane_xsum(X, fm + 8, (g), M);                                                             \
    for (int f = 0; f < NF; f++)                                                                                   \
      for (int r = 0; r < 2; r++)                                                                                  \
        for (int j = 0; j < 4; j++) {                                                                              \
          const int i = f * 8 + r * 4 + j;                                                                         \
          C[i] = fma(s[f][j], P[i], fma(bb[f][j], r ? xs1 : xs0, C[i]));                                           \
        }                                                                                                          \
  }

[[kernel]] void fz_lane(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* SBt [[buffer(2)]], const device int* mdims [[buffer(3)]], device FZ_OUT* Y [[buffer(4)]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]],
    uint tgx [[threadgroup_position_in_grid]]) {
  const int lane = int(lanei), sg = int(sgi);
  const short qid = short(lane) >> 2;
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
  const int M = mdims[0];
  const int tile = fz_tile(int(tgx));
  const int n0 = tile * NT;
  const int g_begin = FZ_G0 + (sg * FZ_GN) / FZ_SK, g_end = FZ_G0 + ((sg + 1) * FZ_GN) / FZ_SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(FZ_K, M));
  threadgroup uint stage_all[FZ_SK * NT * (GS / 4)];
  threadgroup uint* stage = stage_all + sg * NT * (GS / 4);
  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, NT));
  const device uint4* sbv = (const device uint4*)SBt;
  float C[NF * 8];
  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
  fz_raw r0 = fz_lane_fetch(W, sbv, tile, g_begin, g_end, n0, lane, fn);
#if FZ_PF == 2
  fz_raw r1 = fz_lane_fetch(W, sbv, tile, g_begin + 1, g_end, n0, lane, fn);
  for (int g = g_begin; g < g_end; g += 2) {
    FZ_LANE_GROUP(g, r0)
    if (g + 1 < g_end) FZ_LANE_GROUP(g + 1, r1)
  }
#else
  for (int g = g_begin; g < g_end; g++) FZ_LANE_GROUP(g, r0)
#endif
  threadgroup float part[(FZ_SK > 1 ? FZ_SK - 1 : 1) * NF * 8 * 32];
  if (FZ_SK > 1) {
    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
      for (int s2 = 1; s2 < FZ_SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
  }
  if (sg == 0)
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        const int nn = n0 + f * 16 + fn;
        if (m < M && nn < FZ_N)
          for (int j = 0; j < 4; j++) Y[m * FZ_N + nn + j] = static_cast<FZ_OUT>(C[f * 8 + r * 4 + j]);
      }
}
