// Affine 4/6/8-bit y = x W^T (bf16, fp32 sums), bit-identical to MLX 0.32.3's qmv, qmv_fast, qmv_wide and gather_qmv.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// Values and bytes in one packed unit (6-bit: four values in three bytes).
template <int BITS>
struct tf_pack {
  static_assert(BITS == 4 || BITS == 6 || BITS == 8, "4, 6 or 8 bits");
  static constant constexpr int vals = BITS == 6 ? 4 : 32 / BITS;
  static constant constexpr int bytes = BITS == 6 ? 3 : 4;
};

// A lane's VALS inputs pre-divided to meet unshifted weight fields; their sum adds each run of 4 in bf16 first.
template <int BITS, int VALS>
inline float tf_qx(const device bfloat* x, thread float* xs) {
  float total = 0.0f;
  if (BITS == 8) {
    for (int i = 0; i < VALS; ++i) {
      total += x[i];
      xs[i] = x[i];
    }
    return total;
  }
  for (int i = 0; i < VALS; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    total += a + b + c + d;
    xs[i] = a;
    xs[i + 1] = b / (BITS == 4 ? 16.0f : 64.0f);
    xs[i + 2] = c / (BITS == 4 ? 256.0f : 16.0f);
    xs[i + 3] = d / (BITS == 4 ? 4096.0f : 4.0f);
  }
  return total;
}

// acc + field * x as one contracted multiply-add, the form MLX's accumulate statements compile to.
inline float tf_mac(float acc, int field, float x) {
  return acc + field * x;
}

// scale * sum(xs q) + bias * sum(x) over one lane's packed weights (6-bit: at most two units, VALS <= 8).
template <int BITS, int VALS>
inline float tf_qdot(const device uint8_t* w, const thread float* xs, float scale, float bias, float total) {
  float acc = 0.0f;
  if (BITS == 4) {
    const device uint16_t* h = (const device uint16_t*)w;
    for (int i = 0; i < VALS / 4; ++i) {
      const uint16_t q = h[i];
      const thread float* v = xs + 4 * i;
      acc += (v[0] * (q & 0x000f) + v[1] * (q & 0x00f0) + v[2] * (q & 0x0f00) + v[3] * (q & 0xf000));
    }
  } else if (BITS == 6) {
    static_assert(BITS != 6 || VALS <= 8, "6-bit lanes take one or two units");
    for (int i = 0; i < VALS / 4; ++i) {
      const device uint8_t* p = w + 3 * i;
      const thread float* v = xs + 4 * i;
      acc = tf_mac(acc, p[0] & 0x3f, v[0]);
      acc = tf_mac(acc, p[0] & 0xc0, v[1]);
      acc = tf_mac(acc, p[1] & 0x0f, v[1] * 256.0f);   // a field's high bits sit 8 bits up in the next byte
      acc = tf_mac(acc, p[1] & 0xf0, v[2]);
      acc = tf_mac(acc, p[2] & 0x03, v[2] * 256.0f);
      acc = tf_mac(acc, p[2] & 0xfc, v[3]);
    }
  } else {
    for (int i = 0; i < VALS; ++i) acc += xs[i] * w[i];
  }
  return scale * acc + total * bias;
}

// One step of a lane against `live` weight rows: its inputs once, then each row's scaled dot added to acc.
template <int BITS, int VALS>
inline void tf_qmv_step(const device uint8_t* w, const device bfloat* s, const device bfloat* b,
                        const device bfloat* x, int row_bytes, int groups, int live, thread float* acc) {
  float xs[VALS];
  const float total = tf_qx<BITS, VALS>(x, xs);
  for (int r = 0; r < live; ++r)
    acc[r] += tf_qdot<BITS, VALS>(w + r * row_bytes, xs, s[r * groups], b[r * groups], total);
}

// A simdgroup's rows first .. first + ROWS - 1 (ROWS 0: `live`, under 8 output rows) for input row tg.x.
template <int BITS, int GS, int PACKS, int ROWS>
inline void tf_qmv_tile(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device bfloat* x, device bfloat* y, int K, int N, int first, int live, uint row,
                        uint lane) {
  constexpr int VALS = PACKS * tf_pack<BITS>::vals;
  constexpr int STEP = 32 * VALS;
  constexpr int WSTEP = STEP / tf_pack<BITS>::vals * tf_pack<BITS>::bytes;
  const int rows = ROWS ? ROWS : live;
  const int row_bytes = K * tf_pack<BITS>::bytes / tf_pack<BITS>::vals;
  const int groups = K / GS;
  const device uint8_t* wl = (const device uint8_t*)w + size_t(first) * row_bytes + lane * PACKS * tf_pack<BITS>::bytes;
  const device bfloat* sl = scales + size_t(first) * groups + lane * VALS / GS;
  const device bfloat* bl = biases + size_t(first) * groups + lane * VALS / GS;
  const device bfloat* xl = x + size_t(row) * K + lane * VALS;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int k = 0;
  for (; k < (PACKS == 2 ? K : K - STEP); k += STEP) {
    tf_qmv_step<BITS, VALS>(wl, sl, bl, xl, row_bytes, groups, rows, acc);
    wl += WSTEP;
    sl += STEP / GS;
    bl += STEP / GS;
    xl += STEP;
  }
  if (PACKS == 1 && k + int(lane) * VALS < K) tf_qmv_step<BITS, VALS>(wl, sl, bl, xl, row_bytes, groups, rows, acc);
  for (int r = 0; r < rows; ++r) {
    acc[r] = simd_sum(acc[r]);
    if (lane == 0) y[size_t(row) * N + first + r] = static_cast<bfloat>(acc[r]);
  }
}

// MLX's qmv (PACKS 1: a ragged last step, the last tile slid back inside N) or qmv_fast (PACKS 2), 4 rows a simdgroup.
template <int BITS, int GS, int PACKS>
inline void tf_qmv_rows(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device bfloat* x, device bfloat* y, int K, int N, uint3 tg, uint sg, uint lane) {
  const int n0 = int(tg.y) * 8 + int(sg) * 4;
  if (PACKS == 1 && n0 >= N) return;
  if (PACKS == 1 && N < 8)
    tf_qmv_tile<BITS, GS, PACKS, 0>(w, scales, biases, x, y, K, N, n0, min(4, N - n0), tg.x, lane);
  else
    tf_qmv_tile<BITS, GS, PACKS, 4>(w, scales, biases, x, y, K, N, PACKS == 1 ? min(N - 4, n0) : n0, 4, tg.x, lane);
}

// Eight dequantized weights scale * q + bias (fp32) from 8 * BITS / 8 bytes.
template <int BITS>
inline void tf_dequant8(const device uint8_t* w, float s, float b, thread float* out) {
  if (BITS == 4) {
    const float hi = s / 16.0f;
    for (int i = 0; i < 4; ++i) {
      out[2 * i] = s * (w[i] & 0x0f) + b;
      out[2 * i + 1] = hi * (w[i] & 0xf0) + b;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < 2; ++i) {
      const device uint8_t* p = w + 3 * i;
      thread float* o = out + 4 * i;
      o[0] = (p[0] & 0x3f) * s + b;
      o[1] = (((p[0] >> 6) & 0x03) + ((p[1] & 0x0f) << 2)) * s + b;
      o[2] = (((p[1] >> 4) & 0x0f) + ((p[2] & 0x03) << 4)) * s + b;
      o[3] = ((p[2] >> 2) & 0x3f) * s + b;
    }
  } else {
    for (int i = 0; i < 8; ++i) out[i] = s * w[i] + b;
  }
}

// MLX's qmv_wide: lane k of a row takes groups k, k + 8, ..., dequantized by 8s; the 8 lanes add by shuffle 4, 2, 1.
template <int BITS, int GS, int NV>
inline void tf_qmv_wide(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device bfloat* x, device bfloat* y, int K, int N, int M, uint3 tg, uint sg, uint lane) {
  const int kl = int(lane) % 8;
  const int out_row = int(tg.y) * 8 + 4 * int(sg) + int(lane) / 8;
  const int row = min(out_row, N - 1);
  const int vec0 = int(tg.x) * NV;
  const int groups = K / GS;
  const device uint8_t* wr = (const device uint8_t*)w + size_t(row) * (K * BITS / 8);
  const device bfloat* sr = scales + size_t(row) * groups;
  const device bfloat* br = biases + size_t(row) * groups;
  const device bfloat* xr[NV];
  for (int v = 0; v < NV; ++v) xr[v] = x + size_t(min(vec0 + v, M - 1)) * K;
  float out[NV];
  for (int v = 0; v < NV; ++v) out[v] = 0.0f;
  for (int g = kl; g < groups; g += 8) {
    const float s = sr[g];
    const float b = br[g];
#pragma unroll
    for (int c = 0; c < GS / 8; ++c) {
      const int k0 = g * GS + 8 * c;
      float wd[8];
      tf_dequant8<BITS>(wr + k0 * BITS / 8, s, b, wd);
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) part += static_cast<float>(xr[v][k0 + i]) * wd[i];
        out[v] += part;
      }
    }
  }
  for (int v = 0; v < NV; ++v) {
    out[v] += simd_shuffle_down(out[v], 4);
    out[v] += simd_shuffle_down(out[v], 2);
    out[v] += simd_shuffle_down(out[v], 1);
  }
  if (kl == 0 && out_row < N)
    for (int v = 0; v < NV && vec0 + v < M; ++v) y[size_t(vec0 + v) * N + out_row] = static_cast<bfloat>(out[v]);
}

// MLX's gather_qmv: threadgroup z runs the qmv loop on expert ids[z]'s slice, input row xids[z], into output row z.
template <int BITS, int GS, int PACKS>
inline void tf_gather_qmv(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                          const device bfloat* x, const device uint32_t* ids, const device uint32_t* xids,
                          device bfloat* y, int K, int N, uint3 tg, uint sg, uint lane) {
  const long e = long(ids[tg.z]);
  const long words = long(N) * (K * BITS / 32), groups = long(N) * (K / GS);
  tf_qmv_rows<BITS, GS, PACKS>(w + e * words, scales + e * groups, biases + e * groups, x + long(xids[tg.z]) * K,
                               y + long(tg.z) * N, K, N, tg, sg, lane);
}

// tf:kernel tf_qmv_b4_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_b4_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<4, 64, 1>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_fast_b4_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_fast_b4_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<4, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_b6_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_b6_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<6, 64, 1>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_fast_b6_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_fast_b6_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<6, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_b8_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_b8_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<8, 64, 1>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_fast_b8_g64 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_fast_b8_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_rows<8, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b4_g64_v2 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b4_g64_v2(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<4, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b4_g64_v3 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b4_g64_v3(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<4, 64, 3>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b4_g64_v4 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b4_g64_v4(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<4, 64, 4>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b4_g64_v5 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b4_g64_v5(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<4, 64, 5>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b6_g64_v2 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b6_g64_v2(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<6, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b6_g64_v3 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b6_g64_v3(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<6, 64, 3>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b6_g64_v4 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b6_g64_v4(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<6, 64, 4>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b6_g64_v5 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b6_g64_v5(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<6, 64, 5>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b8_g64_v2 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b8_g64_v2(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<8, 64, 2>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b8_g64_v3 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b8_g64_v3(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<8, 64, 3>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b8_g64_v4 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b8_g64_v4(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<8, 64, 4>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_qmv_wide_b8_g64_v5 inputs=W,S,B,X,SHAPE outputs=Y
[[kernel]] void tf_qmv_wide_b8_g64_v5(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const constant int32_t* SHAPE [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_qmv_wide<8, 64, 5>(W, S, B, X, Y, SHAPE[0], SHAPE[1], SHAPE[2], threadgroup_position_in_grid,
                          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_gather_qmv_b4_g64 inputs=W,S,B,X,IDS,XIDS,SHAPE outputs=Y
[[kernel]] void tf_gather_qmv_b4_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const device uint32_t* IDS [[buffer(4)]],
  const device uint32_t* XIDS [[buffer(5)]],
  const constant int32_t* SHAPE [[buffer(6)]],
  device bfloat16_t* Y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_gather_qmv<4, 64, 1>(W, S, B, X, IDS, XIDS, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                              simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_gather_qmv_b6_g64 inputs=W,S,B,X,IDS,XIDS,SHAPE outputs=Y
[[kernel]] void tf_gather_qmv_b6_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const device uint32_t* IDS [[buffer(4)]],
  const device uint32_t* XIDS [[buffer(5)]],
  const constant int32_t* SHAPE [[buffer(6)]],
  device bfloat16_t* Y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_gather_qmv<6, 64, 1>(W, S, B, X, IDS, XIDS, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                              simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

// tf:kernel tf_gather_qmv_b8_g64 inputs=W,S,B,X,IDS,XIDS,SHAPE outputs=Y
[[kernel]] void tf_gather_qmv_b8_g64(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const device uint32_t* IDS [[buffer(4)]],
  const device uint32_t* XIDS [[buffer(5)]],
  const constant int32_t* SHAPE [[buffer(6)]],
  device bfloat16_t* Y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_gather_qmv<8, 64, 1>(W, S, B, X, IDS, XIDS, Y, SHAPE[0], SHAPE[1], threadgroup_position_in_grid,
                              simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}



