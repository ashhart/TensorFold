// bf16 matrix-vector products (fp32 sums), bit-identical to MLX 0.32.3's gemv (bm1 bn8 sn32 tm4 tn4) and gemv_wide.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// gemv: 256 lanes take K in runs of 4 (fma chains), shuffle-down 16..1, then simdgroups 1..7 add into 0 in order.
template <int TM>
inline void tf_gemv_rows(const device bfloat* mat, const device bfloat* vec, device bfloat* out, int K, int rows,
                         int ld, uint tgx, uint sg, uint lane, threadgroup float* part) {
  constexpr int TN = 4, BLOCK = 8 * 32 * TN;
  int first = int(tgx) * TM;
  if (first >= rows) return;
  first = first + TM <= rows ? first : rows - TM;
  const device bfloat* m = mat + long(first) * ld;
  int col = (32 * int(sg) + int(lane)) * TN;
  float acc[TM];
  for (int t = 0; t < TM; ++t) acc[t] = 0.0f;
  const int full = K / BLOCK;
  for (int i = 0; i < full; ++i, col += BLOCK) {
    float v[TN];
    for (int j = 0; j < TN; ++j) v[j] = static_cast<float>(vec[col + j]);
    for (int t = 0; t < TM; ++t) {
      bfloat w[TN];
      for (int j = 0; j < TN; ++j) w[j] = m[t * ld + col + j];
      for (int j = 0; j < TN; ++j) acc[t] += w[j] * v[j];
    }
  }
  if (K - full * BLOCK > 0) {
    float v[TN];
    for (int j = 0; j < TN; ++j) v[j] = col + j < K ? static_cast<float>(vec[col + j]) : 0.0f;
    for (int t = 0; t < TM; ++t) {
      bfloat w[TN];
      for (int j = 0; j < TN; ++j) w[j] = col + j < K ? m[t * ld + col + j] : bfloat(0);
      for (int j = 0; j < TN; ++j) acc[t] += w[j] * v[j];
    }
  }
  for (int t = 0; t < TM; ++t)
    for (ushort d = 16; d >= 1; d >>= 1) acc[t] += simd_shuffle_down(acc[t], d);
  if (lane == 0)
    for (int t = 0; t < TM; ++t) part[sg * (2 * TM) + t] = acc[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane == 0) {
    for (int s = 1; s < 8; ++s)
      for (int t = 0; t < TM; ++t) acc[t] += part[s * (2 * TM) + t];
    for (int t = 0; t < TM; ++t) out[first + t] = static_cast<bfloat>(acc[t]);
  }
}

// gemv_wide: KL lanes a row take float4 runs KL apart, each a dot(); shuffle-down KL/2..1; batch z = (outer, inner).
template <int NV, int KL>
inline void tf_gemv_wide(const device bfloat* mat, const device bfloat* x, device bfloat* out, int K, int N, int M,
                         int mat_ld, int x_ld, const constant int64_t* batch, uint3 tg, uint3 tgs, uint sg,
                         uint lane) {
  const long inner = batch[0];
  const long zo = long(tg.z) / inner, zi = long(tg.z) % inner;
  mat += zo * batch[3] + zi * batch[4];
  x += zo * batch[1] + zi * batch[2];
  out += long(tg.z) * M * N;
  const int kl = int(lane) % KL;
  const int out_row = int(tg.y) * (32 / KL) * (KL / 8) + (32 / KL) * int(sg) + int(lane) / KL;
  const int row = min(out_row, N - 1);
  const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)(mat + long(row) * mat_ld);
  const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)x;
  const int quads = K / 4;
  const int main_quads = quads - quads % (KL * 8);
  const int chunks = (M + NV - 1) / NV;
  for (int chunk = int(tg.x); chunk < chunks; chunk += int(tgs.x)) {
    const int v0 = chunk * NV;
    int xo[NV];
    for (int v = 0; v < NV; ++v) xo[v] = min(v0 + v, M - 1) * (x_ld / 4);
    float acc[NV];
    for (int v = 0; v < NV; ++v) acc[v] = 0.0f;
    for (int base = 0; base < main_quads; base += KL * 8) {
      float4 wq[8];
      for (int i = 0; i < 8; ++i) wq[i] = float4(w4[base + i * KL + kl]);
      for (int v = 0; v < NV; ++v) {
        float run = 0;
        for (int i = 0; i < 8; ++i) run += dot(wq[i], float4(x4[xo[v] + base + i * KL + kl]));
        acc[v] += run;
      }
    }
    for (int q = main_quads + kl; q < quads; q += KL) {
      const float4 wq = float4(w4[q]);
      for (int v = 0; v < NV; ++v) acc[v] += dot(wq, float4(x4[xo[v] + q]));
    }
    for (int v = 0; v < NV; ++v)
      for (ushort d = KL / 2; d >= 1; d >>= 1) acc[v] += simd_shuffle_down(acc[v], d);
    if (kl == 0 && out_row < N)
      for (int v = 0; v < NV; ++v)
        if (v0 + v < M) out[long(v0 + v) * N + out_row] = static_cast<bfloat>(acc[v]);
  }
}

// tf:kernel tf_gemv_bf16_tm4 inputs=MAT,VEC,P outputs=OUT
[[kernel]] void tf_gemv_bf16_tm4(
  const device bfloat16_t* MAT [[buffer(0)]],
  const device bfloat16_t* VEC [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup float part[8 * 8];
  tf_gemv_rows<4>(MAT, VEC + long(threadgroup_position_in_grid.z) * P[0],
                  OUT + long(threadgroup_position_in_grid.z) * P[1], P[0], P[1], P[2],
                  threadgroup_position_in_grid.x, simdgroup_index_in_threadgroup, thread_index_in_simdgroup, part);
}

// tf:kernel tf_gemv_wide_bf16_v4_kl32 inputs=MAT,X,P,BATCH outputs=OUT
[[kernel]] void tf_gemv_wide_bf16_v4_kl32(
  const device bfloat16_t* MAT [[buffer(0)]],
  const device bfloat16_t* X [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* BATCH [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  tf_gemv_wide<4, 32>(MAT, X, OUT, P[0], P[1], P[2], P[3], P[4], BATCH, threadgroup_position_in_grid,
                      threadgroups_per_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}


// tf:kernel tf_gemv_wide_bf16_v3_kl32 inputs=MAT,X,P,BATCH outputs=OUT
[[kernel]] void tf_gemv_wide_bf16_v3_kl32(
  const device bfloat16_t* MAT [[buffer(0)]],
  const device bfloat16_t* X [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* BATCH [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  tf_gemv_wide<3, 32>(MAT, X, OUT, P[0], P[1], P[2], P[3], P[4], BATCH, threadgroup_position_in_grid,
                      threadgroups_per_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_gemv_wide_bf16_v5_kl32 inputs=MAT,X,P,BATCH outputs=OUT
[[kernel]] void tf_gemv_wide_bf16_v5_kl32(
  const device bfloat16_t* MAT [[buffer(0)]],
  const device bfloat16_t* X [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* BATCH [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
  tf_gemv_wide<5, 32>(MAT, X, OUT, P[0], P[1], P[2], P[3], P[4], BATCH, threadgroup_position_in_grid,
                      threadgroups_per_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}
