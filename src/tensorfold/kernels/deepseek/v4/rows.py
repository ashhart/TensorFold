"""Metal row kernels for DeepSeek-V4-Flash's decode path: every row of a window gets its one-row call's bits."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1 import kernels as GK

# MLX's one-row affine 4-bit qmv_fast with fp32 input and output (T = float), 16 inputs a lane
_HEADER_F32 = r"""
template <typename T>
inline float load16f(const device T* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const float a = float(x[i]), b = float(x[i + 1]), c = float(x[i + 2]), d = float(x[i + 3]);
    sum += a + b + c + d;
    xt[i] = a; xt[i + 1] = b / 16.0f; xt[i + 2] = c / 256.0f; xt[i + 3] = d / 4096.0f;
  }
  return sum;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
"""

_QMV_ROWS_F32 = r"""
  // Threadgroup b: simdgroup r runs MLX's one-row fp32 qmv_fast on input row r; the rows share the weight reads
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 4;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 4;
  auto x = X + r * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16f(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = v;
  }
"""

# GLM's qmv_rows (MLX's one-row bf16 qmv_fast) where output row o reads input group o / NG: a grouped projection
_QMV_ROWS_GROUPED = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 4;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 4;
  const device bfloat* x = X + size_t(r) * G * K + size_t(row0 / NG) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

# MLX's one-row mxfp4 fp_qmv_fast (groups of 32, e8m0 scales) for each pick of a window's routed experts
_HEADER_FP4 = r"""
constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0(uint8_t s) {
  return as_type<float>(s == 0 ? 0x400000u : (uint(s) << 23));
}
inline float fp4dot16(const device uint8_t* w, const thread float* xt, float scale) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += (xt[4 * i] * FP4_VALUES[ws[i] & 15] + xt[4 * i + 1] * FP4_VALUES[(ws[i] >> 4) & 15] +
              xt[4 * i + 2] * FP4_VALUES[(ws[i] >> 8) & 15] + xt[4 * i + 3] * FP4_VALUES[(ws[i] >> 12) & 15]);
  return scale * accum;
}
"""

_EXPERT_FP4 = r"""
  // Simdgroup m runs pick m of expert u with the one-row fp_qmv_fast loop; an expert's picks share reads
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * 8;
  const device uint8_t* sc = S + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    for (int i = 0; i < 16; i++) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; j++)
      acc[j] += fp4dot16(w + j * KB, xt, e8m0(sc[j * KG]));
    w += 256; sc += 16; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
"""


def f32_rows_fits(q: Any, rows: int) -> bool:
    k = int(q.scales.shape[-1]) * int(q.group)
    return (GK.metal() and q.bits == 4 and q.group == 64 and k % 512 == 0 and int(q.weight.shape[0]) % 4 == 0
            and 1 <= rows <= 32)


def qmv_rows_f32(x: mx.array, q: Any, rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K] (bf16 or fp32) through 4-bit group-64 weights: each row MLX's one-row fp32 bits, reads shared."""

    rows, dims = x.shape
    n = int(q.weight.shape[0])
    kernel = GK._kernel("ds4_qmv_rows_f32", _QMV_ROWS_F32, ["X", "W", "S", "B"], ["OUT"], _HEADER_F32)
    return kernel(inputs=[mx.contiguous(x if x.dtype in (mx.bfloat16, mx.float32) else x.astype(mx.float32)),
                          q.weight, q.scales, q.biases],
                  template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup)],
                  grid=(32 * rows, n // rows_per_simdgroup, 1), threadgroup=(32 * rows, 1, 1),
                  output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]


def grouped_fits(q: Any, groups: int, rows: int) -> bool:
    n, k = int(q.weight.shape[0]), int(q.scales.shape[-1]) * int(q.group)
    return (GK.metal() and q.bits == 4 and q.group == 64 and k % 512 == 0 and n % (4 * groups) == 0
            and (n // groups) % 4 == 0 and 1 < rows <= 32)


def qmv_rows_grouped(x: mx.array, q: Any, groups: int, rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, G * K] through G stacked [N / G, K] projections (output block g reads input block g): one-row bits."""

    rows, dims = x.shape
    n, k = int(q.weight.shape[0]), dims // groups
    kernel = GK._kernel("ds4_qmv_rows_grouped", _QMV_ROWS_GROUPED, ["X", "W", "S", "B"], ["OUT"], GK._HEADER)
    return kernel(inputs=[mx.contiguous(x), q.weight, q.scales, q.biases],
                  template=[("K", k), ("N", n), ("G", groups), ("NG", n // groups), ("RPS", rows_per_simdgroup)],
                  grid=(32 * rows, n // rows_per_simdgroup, 1), threadgroup=(32 * rows, 1, 1),
                  output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]


def grouped_one_row(x: mx.array, q: Any, groups: int) -> mx.array:
    """One row [1, G * K] through the G projections in one batched MLX call (its qmv per group)."""

    n, k = int(q.weight.shape[0]), int(x.shape[-1]) // groups
    xg = x.reshape(groups, 1, k)
    w = q.weight.reshape(groups, n // groups, -1)
    s = q.scales.reshape(groups, n // groups, -1)
    b = q.biases.reshape(groups, n // groups, -1)
    out = mx.quantized_matmul(xg, w, s, b, transpose=True, group_size=q.group, bits=q.bits)
    return out.reshape(1, n)


def fp4_rows_fits(w: Any, rows: int) -> bool:
    k = int(w.weight.shape[-1]) * 8
    return GK.metal() and k % 512 == 0 and int(w.weight.shape[-2]) % 4 == 0 and 1 < rows <= GK.MAX_ROWS


def expert_rows_fp4(x: mx.array, idx: mx.array, group: Any, w: Any, *, per_pick: bool,
                    rows_per_simdgroup: int = 4) -> mx.array:
    """Every pick of a window through its mxfp4 expert with gather_qmm's one-row bits: [R, k, N] bf16."""

    rows, top = idx.shape
    return picks_fp4(x, rows, top, group, w, per_pick=per_pick, rows_per_simdgroup=rows_per_simdgroup)


def picks_fp4(x: mx.array, rows: int, top: int, group: Any, w: Any, *, per_pick: bool,
              rows_per_simdgroup: int = 4) -> mx.array:
    """``expert_rows_fp4`` from the window's shape (rows, picks a row): [R, k, N] bf16."""

    n, dims = int(w.weight.shape[-2]), int(x.shape[-1])
    uids, umem, ucount = group
    kernel = GK._kernel("ds4_expert_fp4", _EXPERT_FP4, ["X", "W", "S", "UIDS", "UMEM", "UCOUNT"], ["OUT"],
                        _HEADER_FP4)
    out = kernel(inputs=[mx.contiguous(x.reshape(-1, dims)), w.weight, w.scales, uids, umem, ucount],
                 template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup), ("TOPK", top), ("MAXR", GK.MAX_ROWS),
                           ("PER_PICK", int(per_pick))],
                 grid=(32 * rows, n // rows_per_simdgroup, rows * top), threadgroup=(32 * rows, 1, 1),
                 output_shapes=[(rows * top, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(rows, top, n)
