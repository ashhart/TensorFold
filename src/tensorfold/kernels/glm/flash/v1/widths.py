"""MLX's one-row qmv loops at its other affine widths (5, 6 and 8 bits), for mixed-bit checkpoints."""

from __future__ import annotations

from typing import Any

# The same one-row loop for MLX's other affine widths (qmv_fast_impl of MLX 0.32.2's quantized.h for 5, 6 and 8
# bits, as the mixed-bit abliterated checkpoint stores its attention, shared experts and head): per 32-lane block a
# lane takes V inputs (8 for 8- and 6-bit, 16 for 5-bit) and LB weight bytes, sums the inputs the way load_vector
# does (float adds of the bf16 values for 8-bit; bf16-chained sums of 4 / 8 for 6- / 5-bit, the inputs pre-scaled
# for their bit positions) and multiplies against its packed bytes in qdot's order and expression shape. Which
# widths MLX runs through qmv_fast: N % 8 == 0 and K % (32 V) == 0 (quantized.cpp: qmv_fast_k_alignment); the
# callers check the same before using these.
_HEADER_B = r"""
inline float bfsum4(const device bfloat* x) {
  return float(bfloat(float(bfloat(float(bfloat(float(x[0]) + float(x[1]))) + float(x[2]))) + float(x[3])));
}
inline float bfsum8(const device bfloat* x) {
  bfloat s = bfloat(float(x[0]) + float(x[1]));
  s = bfloat(float(s) + float(x[2])); s = bfloat(float(s) + float(x[3])); s = bfloat(float(s) + float(x[4]));
  s = bfloat(float(s) + float(x[5])); s = bfloat(float(s) + float(x[6])); s = bfloat(float(s) + float(x[7]));
  return float(s);
}
template <int B, int V> inline float loadv(const device bfloat* x, thread float* xt);
template <> inline float loadv<4, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
template <> inline float loadv<8, 8>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 8; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<8, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<8, 32>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 32; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<6, 8>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 8; i += 4) {
    sum += bfsum4(x + i);
    xt[i] = float(x[i]); xt[i + 1] = float(x[i + 1]) / 64.0f; xt[i + 2] = float(x[i + 2]) / 16.0f;
    xt[i + 3] = float(x[i + 3]) / 4.0f;
  }
  return sum;
}
template <> inline float loadv<5, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 8) {
    sum += bfsum8(x + i);
    xt[i] = float(x[i]); xt[i + 1] = float(x[i + 1]) / 32.0f; xt[i + 2] = float(x[i + 2]) / 4.0f;
    xt[i + 3] = float(x[i + 3]) / 128.0f; xt[i + 4] = float(x[i + 4]) / 16.0f; xt[i + 5] = float(x[i + 5]) / 2.0f;
    xt[i + 6] = float(x[i + 6]) / 64.0f; xt[i + 7] = float(x[i + 7]) / 8.0f;
  }
  return sum;
}
template <int B, int V> inline float qdotv(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                           float sum);
template <> inline float qdotv<4, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 8>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                     float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 8; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 16; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 32>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 32; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<6, 8>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                     float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    xt += 4 * i;
    w += 3 * i;
    accum += (w[0] & 0x3f) * xt[0];
    accum += (w[0] & 0xc0) * xt[1];
    accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
    accum += (w[1] & 0xf0) * xt[2];
    accum += (w[2] & 0x03) * (xt[2] * 256.0f);
    accum += (w[2] & 0xfc) * xt[3];
  }
  return scale * accum + sum * bias;
}
template <> inline float qdotv<5, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    xt += 8 * i;
    w += 5 * i;
    accum += (w[0] & 0x1f) * xt[0];
    accum += (w[0] & 0xe0) * xt[1];
    accum += (w[1] & 0x3) * (xt[1] * 256.0f);
    accum += (w[1] & 0x7c) * xt[2];
    accum += (w[1] & 0x80) * xt[3];
    accum += (w[2] & 0xf) * (xt[3] * 256.0f);
    accum += (w[2] & 0xf0) * xt[4];
    accum += (w[3] & 0x1) * (xt[4] * 256.0f);
    accum += (w[3] & 0x3e) * xt[5];
    accum += (w[3] & 0xc0) * xt[6];
    accum += (w[4] & 0x7) * (xt[6] * 256.0f);
    accum += (w[4] & 0xf8) * xt[7];
  }
  return scale * accum + sum * bias;
}
"""

# bits -> (V: inputs a lane a block, LB: weight bytes a lane a block) of MLX's qmv_fast loop; a block is 32 V inputs
QFAST = {8: (8, 8), 6: (8, 6), 5: (16, 10)}
QFAST_ALL = {4: (16, 8), **QFAST}          # 4-bit too (load16 / qdot16 as loadv<4, 16> / qdotv<4, 16>)

_QMV_ROWS_B = r"""
  // _QMV_ROWS for BITS-bit weights (5, 6, 8): simdgroup r = input row r, output rows RPS b .. RPS b + RPS - 1, MLX's
  // one-row qmv_fast loop at that width (V inputs and LB weight bytes a lane a block, a group of 64 spans 64 / V
  // lanes); the R simdgroups read the same weight rows once.
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int BLK = 32 * V;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / 64;
  constexpr int SDIV = 64 / V;
  constexpr int SSTEP = BLK / 64;
  constexpr int WSTEP = 32 * LB;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * LB;
  const device bfloat* sc = S + size_t(row0) * KG + lane / SDIV;
  const device bfloat* bi = B + size_t(row0) * KG + lane / SDIV;
  const device bfloat* x = X + r * K + lane * V;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BLK) {
    float xt[V];
    const float sum = loadv<BITS, V>(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdotv<BITS, V>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += WSTEP; sc += SSTEP; bi += SSTEP; x += BLK;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_EXPERT_QMV_B = r"""
  // _EXPERT_QMV for BITS-bit expert matrices [E, N, K] (5, 6, 8): the m-th pick of distinct expert u through MLX's
  // one-row qmv_fast loop at that width (what affine_gather_qmv_fast runs for one row).
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int BLK = 32 * V;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / 64;
  constexpr int SDIV = 64 / V;
  constexpr int SSTEP = BLK / 64;
  constexpr int WSTEP = 32 * LB;
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * LB;
  const device bfloat* sc = S + (e * N + row0) * KG + lane / SDIV;
  const device bfloat* bi = B + (e * N + row0) * KG + lane / SDIV;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * V;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BLK) {
    float xt[V];
    const float sum = loadv<BITS, V>(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdotv<BITS, V>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += WSTEP; sc += SSTEP; bi += SSTEP; x += BLK;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
"""

# MLX's one-row 8-bit matvec for 64- / 128-input matrices (qmv_quad_impl at 8 bits: a quad of lanes per output row,
# K / 4 inputs a lane, float sums in order, quad_sum), the window's rows in grid x, as _QMV_QUAD_ROWS does for 4-bit.
_QMV_QUAD_ROWS_8 = r"""
  constexpr int QUADS = 8;
  constexpr int PER = K / 4;                       // inputs (and weight bytes) a lane
  constexpr int KB = K;                            // bytes a weight row
  constexpr int KG = K / 64;
  const uint lane = thread_index_in_simdgroup;
  const int quad_lid = int(lane % 4), quad_gid = int(lane / 4);
  const int r = int(threadgroup_position_in_grid.x);
  const int out_row = int(threadgroup_position_in_grid.y) * QUADS * 8 + quad_gid;
  const device uint8_t* w = (const device uint8_t*)W + size_t(out_row) * KB + quad_lid * PER;
  const device bfloat* sc = S + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* bi = B + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* x = X + size_t(r) * K + quad_lid * PER;
  float xt[PER];
  const float sum = loadv<8, PER>(x, xt);
  float result[8];
  for (int row = 0; row < 8; row++) {
    result[row] = 0.0f;
    if (row * QUADS + out_row < N) {
      const float s = float(sc[row * QUADS * KG]), bb = float(bi[row * QUADS * KG]);
      result[row] += qdotv<8, PER>(w + size_t(row) * QUADS * KB, xt, s, bb, sum);
    }
  }
  for (int row = 0; row < 8; row++) {
    const float v = quad_sum(result[row]);
    if (quad_lid == 0 && row * QUADS + out_row < N) OUT[size_t(r) * N + out_row + row * QUADS] = bfloat(v);
  }
"""

def fast_shape(weights: Any, n: int) -> bool:
    """Whether MLX's one-row quantized matmul of these 5 / 6 / 8-bit, group-64 weights (n output rows) is its
    qmv_fast kernel (quantized.cpp: N % 8 == 0 and K a whole number of 32 V blocks), the loop the row kernels
    reproduce. K of 64 or 128 is qmv_quad territory (power-of-2 bits), not this."""

    bits = getattr(weights, "bits", None)
    if bits not in QFAST or weights.group != 64:
        return False
    k = int(weights.scales.shape[-1]) * int(weights.group)
    v, _ = QFAST[bits]
    return k % (32 * v) == 0 and n % 8 == 0 and not (k in (64, 128) and bits == 8)
