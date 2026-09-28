"""Metal row kernels for GLM-5.3-Flash's decode path: every row of a window gets its one-row call's bits."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

# MLX's one-row 4-bit matvec a simdgroup lane (as Flash Next's qmv_rows, bit for bit), 16 inputs a 512 block
_HEADER = r"""
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
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

_QMV_ROWS = r"""
  // Threadgroup b: simdgroup r runs MLX's one-row qmv_fast on input row r; the rows share the weight reads
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 4;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 4;
  const device bfloat* x = X + r * K + lane * 16;
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

# sinkhorn and the pre-weighted collapse, 256 threads a row (mlx-vlm's DeepSeek-V4 hyper-connection, MIT)
_HC_SPLIT = r"""
  uint tid  = thread_position_in_threadgroup.x;
  uint row  = threadgroup_position_in_grid.x;
  uint lane = tid % 32;
  uint sg   = tid / 32;
  constexpr int MIX      = (2 + HC) * HC;
  constexpr int BASE_OFF = 2 * HC;
  constexpr float EPS = EPS_INT * 1e-9;
  const device float* mix      = (const device float*)mixes + row * MIX;
  device float*       post_out = (device float*)post + row * HC;
  device float*       comb_out = (device float*)comb + row * HC * HC;
  threadgroup float pre_shared[HC];
  if (sg == 0) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)HC) {
      pre_shared[lane] = pre_v;
      post_out[lane]   = post_v;
    }
    float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC) * comb_scale
              + *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::fast::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
      col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
      r *= col_inv;
    }
    if (lane < (uint)HC) {
      *(device float4*)(comb_out + lane * HC) = r;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_shared[0];
  const float p1 = pre_shared[1];
  const float p2 = pre_shared[2];
  const float p3 = pre_shared[3];
  const device T* x_row  = (const device T*)x_in + row * (HC * D);
  device T*       out_row = (device T*)collapsed + row * D;
  using T4 = vec<T, 4>;
  const device T4* x_row0 = (const device T4*)(x_row + 0*D);
  const device T4* x_row1 = (const device T4*)(x_row + 1*D);
  const device T4* x_row2 = (const device T4*)(x_row + 2*D);
  const device T4* x_row3 = (const device T4*)(x_row + 3*D);
  device T4*       out4   = (device T4*)out_row;
  constexpr uint D4 = (uint)D / 4;
  for (uint d4 = tid; d4 < D4; d4 += 256) {
    float4 x0 = float4(x_row0[d4]);
    float4 x1 = float4(x_row1[d4]);
    float4 x2 = float4(x_row2[d4]);
    float4 x3 = float4(x_row3[d4]);
    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    out4[d4] = T4(result);
  }
"""

# a window's routed experts grouped by expert, so each expert's weights are read once for the rows picking it
_EXPERT_GROUP = r"""
  // Thread e lists expert e's picks (row * TOPK + slot); distinct experts get places in id order
  const int e = int(thread_position_in_threadgroup.x);
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int picks = int(IDX_shape[0]) * TOPK;
  threadgroup int offs[32];
  int members[MAXR];
  int count = 0;
  if (e < NE)
    for (int p = 0; p < picks; p++)
      if (int(IDX[p]) == e) members[count++] = p;
  const int used = count > 0 ? 1 : 0;
  const int before = simd_prefix_exclusive_sum(used);
  if (lane == 31) offs[g] = before + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int base = 0;
  for (int q = 0; q < int(g); q++) base += offs[q];
  if (used) {
    const int u = base + before;
    UIDS[u] = e;
    for (int j = 0; j < MAXR; j++) UMEM[u * MAXR + j] = j < count ? members[j] : -1;
  }
  if (e == ((NE + 31) / 32) * 32 - 1) UCOUNT[0] = base + before + used;
"""

_EXPERT_QMV = r"""
  // Simdgroup m runs pick m of expert u with the one-row gather qmv_fast loop; an expert's picks share reads
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * 8;
  const device bfloat* sc = S + (e * N + row0) * KG + lane / 4;
  const device bfloat* bi = B + (e * N + row0) * KG + lane / 4;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * 16;
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
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
"""

# MLX's one-row qmv_quad (a quad of lanes an output row) for short inputs; MLX switches kernels past 8 rows
_QMV_QUAD_ROWS = r"""
  constexpr int QUADS = 8;
  constexpr int PER = K / 4;                       // inputs a lane
  constexpr int KB = K / 2;                        // bytes a weight row
  constexpr int KG = K / 64;
  const uint lane = thread_index_in_simdgroup;
  const int quad_lid = int(lane % 4), quad_gid = int(lane / 4);
  const int r = int(threadgroup_position_in_grid.x);
  const int out_row = int(threadgroup_position_in_grid.y) * QUADS * 8 + quad_gid;
  const device uint8_t* w = (const device uint8_t*)W + size_t(out_row) * KB + quad_lid * (PER / 2);
  const device bfloat* sc = S + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* bi = B + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* x = X + size_t(r) * K + quad_lid * PER;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  float result[8];
  for (int row = 0; row < 8; row++) {
    result[row] = 0.0f;
    if (row * QUADS + out_row < N) {
      const device uint16_t* ws = (const device uint16_t*)(w + size_t(row) * QUADS * KB);
      const float s = float(sc[row * QUADS * KG]), bb = float(bi[row * QUADS * KG]);
      float accum = 0.0f;
      for (int i = 0; i < PER / 4; i++)
        accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
                 xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
      result[row] += s * accum + sum * bb;
    }
  }
  for (int row = 0; row < 8; row++) {
    const float v = quad_sum(result[row]);
    if (quad_lid == 0 && row * QUADS + out_row < N) OUT[size_t(r) * N + out_row + row * QUADS] = bfloat(v);
  }
"""

# MLX's one-row gemv / gemv_t, each window row with the one-row tiling and sums (its 2+ row matmul differs)
_GEMV_T_ROWS = r"""
  constexpr int blockM = BM * SM * TM;
  constexpr int blockN = BN * SN * TN;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.z);
  const int in_vec_size = int(X_shape[1]);
  const int out_vec_size = int(M_shape[1]);
  const device T* in_vec = X + size_t(row) * in_vec_size;
  device T* out_vec = OUT + size_t(row) * out_vec_size;
  threadgroup float tgp_memory[BM > 1 ? BM * (blockN + TN) : 1];
  float result[TN] = {0};
  T inter[TN];
  float v_coeff[TM];
  const int thrM = SN != 32 ? simd_lid / SN : 0;
  const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
  const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
  const int sgN = BN != 1 ? (simd_gid % BN) : 0;
  const int cm = SM * sgM + thrM;
  const int cn = SN * sgN + thrN;
  int bm = cm * TM;
  const int bn = cn * TN;
  int out_col = int(threadgroup_position_in_grid.x) * blockN + bn;
  const int n_iter = in_vec_size / blockM;
  const int leftover = in_vec_size - blockM * n_iter;
  if (out_col < out_vec_size) {
    out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;
    for (int i = 0; i < n_iter; ++i) {
      threadgroup_barrier(mem_flags::mem_none);
      for (int tm = 0; tm < TM; tm++) v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
      for (int tm = 0; tm < TM; tm++) {
        const float vc = v_coeff[tm];
        for (int tn = 0; tn < TN; tn++) inter[tn] = M[size_t(bm + tm) * out_vec_size + out_col + tn];
        for (int tn = 0; tn < TN; tn++) result[tn] += vc * inter[tn];
      }
      bm += blockM;
    }
    if (leftover > 0) {
      for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
        v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
        for (int tn = 0; tn < TN; tn++) inter[tn] = M[size_t(bm + tm) * out_vec_size + out_col + tn];
        for (int tn = 0; tn < TN; tn++) result[tn] += v_coeff[tm] * inter[tn];
      }
    }
  }
  for (int tn = 0; tn < TN; tn++)
    for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) result[tn] += simd_shuffle_down(result[tn], SN * sm);
  if (BM > 1) {
    threadgroup float* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
    if (thrM == 0) {
      for (int tn = 0; tn < TN; tn++) tgp_results[tn] = result[tn];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgM == 0)
        for (int sgm = 1; sgm < BM; sgm++)
          for (int tn = 0; tn < TN; tn++) result[tn] += tgp_results[sgm * (blockN + TN) + tn];
    }
  }
  if (cm == 0 && out_col < out_vec_size)
    for (int j = 0; j < TN; j++) out_vec[out_col + j] = static_cast<T>(result[j]);
"""

_GEMV_ROWS = r"""
  constexpr int blockM = BM * SM * TM;
  constexpr int blockN = BN * SN * TN;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.z);
  const int in_vec_size = int(X_shape[1]);
  const int out_vec_size = int(M_shape[0]);
  const device T* in_vec = X + size_t(row) * in_vec_size;
  device T* out_vec = OUT + size_t(row) * out_vec_size;
  threadgroup float tgp_memory[BN > 1 ? BN * (blockM + TM) : 1];
  float result[TM] = {0};
  T inter[TN];
  float v_coeff[TN];
  const int thrM = SN != 32 ? simd_lid / SN : 0;
  const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
  const int sgN = BN != 1 ? (simd_gid % BN) : 0;
  const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
  const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
  const int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;
  int out_row = int(threadgroup_position_in_grid.x) * blockM + bm;
  if (out_row >= out_vec_size) return;
  out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
  const device T* mat = M + size_t(out_row) * in_vec_size;
  const int n_iter = in_vec_size / blockN;
  const int leftover = in_vec_size - blockN * n_iter;
  for (int i = 0; i < n_iter; ++i) {
    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = mat[mat_offset + bn + tn];
      for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
      mat_offset += in_vec_size;
    }
    bn += blockN;
  }
  if (leftover > 0) {
    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = bn + tn < in_vec_size ? static_cast<float>(in_vec[bn + tn]) : 0.0f;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = bn + tn < in_vec_size ? mat[tm * in_vec_size + bn + tn] : T(0);
      for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
    }
  }
  for (int tm = 0; tm < TM; tm++)
    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) result[tm] += simd_shuffle_down(result[tm], sn);
  if (BN > 1) {
    threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
      for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[tm];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgN == 0)
        for (int sgn = 1; sgn < BN; sgn++)
          for (int tm = 0; tm < TM; tm++) result[tm] += tgp_results[sgn * (blockM + TM) + tm];
    }
  }
  if (simdN == 0 && thrN == 0)
    for (int tm = 0; tm < TM; tm++) out_vec[out_row + tm] = static_cast<T>(result[tm]);
"""

_kernels: dict[str, Any] = {}


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str], header: str = "") -> Any:
    kernel = _kernels.get(name)
    if kernel is None:
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:10]
        kernel = mx.fast.metal_kernel(name=f"tf_glm5_{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source, header=header)
        _kernels[name] = kernel
    return kernel


def qmv_rows_fits(weights: Any, rows: int) -> bool:
    k = int(weights.scales.shape[-1]) * int(weights.group)
    return (weights.bits == 4 and weights.group == 64 and k % 512 == 0 and int(weights.weight.shape[0]) % 4 == 0
            and 1 < rows <= 32)


def qmv_rows(x: mx.array, weights: Any, *, rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K] (bf16) through 4-bit, group-64 weights: each row MLX's one-row bits, weight reads shared."""

    rows, dims = x.shape
    n = int(weights.weight.shape[0])
    kernel = _kernel("qmv_rows64", _QMV_ROWS, ["X", "W", "S", "B"], ["OUT"], _HEADER)
    return kernel(inputs=[mx.contiguous(x), weights.weight, weights.scales, weights.biases],
                  template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup)],
                  grid=(32 * rows, n // rows_per_simdgroup, 1), threadgroup=(32 * rows, 1, 1),
                  output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]


MAX_ROWS = 16


def qmv_quad_rows_fits(weights: Any, rows: int) -> bool:
    k = int(weights.scales.shape[-1]) * int(weights.group)
    return weights.bits == 4 and weights.group == 64 and k in (64, 128) and 1 < rows <= MAX_ROWS


def qmv_quad_rows(x: mx.array, weights: Any) -> mx.array:
    """x [R, K] (K 64 or 128) through 4-bit group-64 weights with MLX's one-row qmv_quad bits; no Metal: row by row."""

    rows, dims = x.shape
    if not metal():
        return mx.concatenate([weights(x[r:r + 1]) for r in range(rows)])
    n = int(weights.weight.shape[0])
    kernel = _kernel("qmv_quad_rows64", _QMV_QUAD_ROWS, ["X", "W", "S", "B"], ["OUT"])
    return kernel(inputs=[mx.contiguous(x.astype(mx.bfloat16)), weights.weight, weights.scales, weights.biases],
                  template=[("K", dims), ("N", n)],
                  grid=(32 * rows, -(-n // 64), 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]


def expert_group(idx: mx.array, experts: int) -> tuple[mx.array, mx.array, mx.array] | None:
    """A window's picks [R, k] grouped by distinct expert in id order (ids, member picks, count); None without Metal."""

    if not metal():
        return None
    rows, top = idx.shape
    kernel = _kernel("expert_group", _EXPERT_GROUP, ["IDX"], ["UIDS", "UMEM", "UCOUNT"])
    threads = -(-experts // 32) * 32
    return tuple(kernel(inputs=[mx.contiguous(idx.astype(mx.uint32))],
                        template=[("NE", experts), ("TOPK", top), ("MAXR", MAX_ROWS)],
                        grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
                        output_shapes=[(rows * top,), (rows * top, MAX_ROWS), (1,)],
                        output_dtypes=[mx.int32, mx.int32, mx.int32]))


def expert_qmv_fits(weights: Any, rows: int) -> bool:
    k = int(weights.scales.shape[-1]) * int(weights.group)
    return (weights.bits == 4 and weights.group == 64 and k % 512 == 0 and int(weights.weight.shape[-2]) % 4 == 0
            and 1 < rows <= MAX_ROWS)


def _gather_one_row(x: mx.array, ids: mx.array, weights: Any) -> mx.array:
    """One row's picks as the one-row decode path runs them: x [k or 1, 1, K], ids [1, k] -> [1, k, N]."""

    return mx.gather_qmm(x[None], weights.weight, weights.scales, weights.biases, rhs_indices=ids, transpose=True,
                         group_size=weights.group, bits=weights.bits).squeeze(-2)


def expert_qmv(x: mx.array, idx: mx.array, group: tuple[mx.array, mx.array, mx.array] | None, weights: Any, *,
               per_pick: bool, rows_per_simdgroup: int = 4) -> mx.array:
    """Every pick of a window through its expert, each with mx.gather_qmm's one-row bits: [R, k, N] bf16."""

    rows, top = idx.shape
    n, dims = int(weights.weight.shape[-2]), int(x.shape[-1])
    if group is None or not expert_qmv_fits(weights, rows):
        parts = []
        for r in range(rows):
            xr = x[r][:, None, :] if per_pick else x[r:r + 1][:, None, :]
            parts.append(_gather_one_row(xr, idx[r:r + 1], weights))
        return mx.concatenate(parts)
    uids, umem, ucount = group
    picks = rows * top
    kernel = _kernel("expert_qmv64", _EXPERT_QMV, ["X", "W", "S", "B", "UIDS", "UMEM", "UCOUNT"], ["OUT"], _HEADER)
    out = kernel(inputs=[mx.contiguous(x.reshape(-1, dims)), weights.weight, weights.scales, weights.biases, uids,
                         umem, ucount],
                 template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup), ("TOPK", top), ("MAXR", MAX_ROWS),
                           ("PER_PICK", int(per_pick))],
                 grid=(32 * rows, n // rows_per_simdgroup, picks), threadgroup=(32 * rows, 1, 1),
                 output_shapes=[(picks, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(rows, top, n)


def gemv_params(transposed: bool, in_len: int, out_len: int) -> tuple[int, int, int, int, int, int]:
    """The (BM, BN, SM, SN, TM, TN) tiling MLX 0.32's matmul picks for one row (gemv_t when ``transposed``)."""

    tm, tn, sm, sn, bm, bn = 4, 4, 1, 32, 1, 1
    if transposed:
        sm, sn = (4, 8) if in_len >= 8192 and out_len >= 2048 else (8, 4)
        bn = 16 if out_len >= 2048 else (4 if out_len >= 512 else 2)
        tn = 1 if out_len < tn else tn
    else:
        bm = 8 if out_len >= 4096 else 4
        sn = 32
        if in_len <= 64:
            bm, sm, sn = 1, 8, 4
        elif in_len >= 16 * out_len:
            bm, bn = 1, 8
        tm = 1 if out_len < tm else tm
    return bm, bn, sm, sn, tm, tn


def matmul_rows(x: mx.array, m: mx.array, *, transposed: bool, params: tuple[int, ...] | None = None) -> mx.array:
    """x [R, K] @ m (or m.T) with MLX's one-row matmul bits for every row; no Metal: one MLX matmul a row."""

    x = x.astype(m.dtype)
    rows, in_len = x.shape
    if not metal():
        mat = m if transposed else m.T
        return mx.concatenate([x[r:r + 1] @ mat for r in range(rows)])
    out_len = int(m.shape[1] if transposed else m.shape[0])
    bm, bn, sm, sn, tm, tn = params or gemv_params(transposed, in_len, out_len)
    per_group = bn * sn * tn if transposed else bm * sm * tm
    groups = -(-out_len // per_group)
    name = "gemv_t_rows" if transposed else "gemv_rows"
    kernel = _kernel(name, _GEMV_T_ROWS if transposed else _GEMV_ROWS, ["X", "M"], ["OUT"])
    return kernel(inputs=[mx.contiguous(x), mx.contiguous(m)],
                  template=[("T", m.dtype), ("BM", bm), ("BN", bn), ("SM", sm), ("SN", sn), ("TM", tm), ("TN", tn)],
                  grid=(groups * 32 * bm * bn, 1, rows), threadgroup=(32 * bm * bn, 1, 1),
                  output_shapes=[(rows, out_len)], output_dtypes=[m.dtype])[0]


def _hc_split_ops(x: mx.array, mixes: mx.array, scale: mx.array, base: mx.array, hc: int, iters: int,
                  eps: float) -> tuple[mx.array, mx.array, mx.array]:
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    xf = x.astype(mx.float32)
    collapsed = pre[..., 0:1] * xf[..., 0, :]
    for s in range(1, hc):
        collapsed = collapsed + pre[..., s:s + 1] * xf[..., s, :]
    return collapsed.astype(x.dtype), post, comb


def hc_split(x: mx.array, mixes: mx.array, scale: mx.array, base: mx.array, *, hc: int, iters: int,
             eps: float) -> tuple[mx.array, mx.array, mx.array]:
    """x [R, HC, D] streams, mixes [R, (2 + HC) HC] fp32 -> (collapsed [R, D], post [R, HC], comb [R, HC, HC])."""

    rows, streams, dims = x.shape
    if not metal() or streams != 4 or dims % 4:
        return _hc_split_ops(x, mixes, scale, base, hc, iters, eps)
    kernel = _kernel("hc_split", _HC_SPLIT, ["x_in", "mixes", "scale", "base"], ["collapsed", "post", "comb"])
    return tuple(kernel(inputs=[x, mixes, scale, base],
                        template=[("T", x.dtype), ("HC", hc), ("ITERS", iters), ("D", dims),
                                  ("EPS_INT", round(eps / 1e-9))],
                        grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(rows, dims), (rows, hc), (rows, hc, hc)],
                        output_dtypes=[x.dtype, mx.float32, mx.float32]))


def gated_delta(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
                state: mx.array) -> tuple[mx.array, mx.array]:
    """The KDA recurrence from an fp32 state: mlx-lm's kernel, the time steps in order in one thread."""

    from mlx_lm.models import gated_delta as gd

    if not metal():
        return gd.gated_delta_ops(q, k, v, g, beta, state)
    return gd.gated_delta_kernel(q, k, v, g, beta, state)
