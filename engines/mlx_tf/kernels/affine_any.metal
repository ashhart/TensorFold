// Clean-room universal Apple Silicon (M1-M5, Metal 3.1+) kernels for:
//   - General MLX affine quantization: bits in {2, 3, 4, 5, 6, 8}, group_size in {32, 64, 128}
//   - Lossless BF16/F16/F32 scales & biases (sb_dtype: 0=bf16, 1=f16, 2=f32)
//   - Unquantized BF16 dense linear projections (GEMV + 4-row-reused GEMM)
//   - Universal token embedding lookup (raw BF16/F16/F32 or 2/3/4/5/6/8-bit affine)
//   - Arbitrary input/output row strides (ldx, ldo) for vertical mixed-precision fusion (ConcatLinear)
//   - Optional fused residual add (add_res != 0)
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

struct AffineParams {
  uint rows;
  uint N;
  uint K;
  uint ldx;      // input row stride in bfloats (>= K)
  uint ldo;      // output row stride in bfloats (>= N)
  uint bits;     // 2, 3, 4, 5, 6, or 8 (or 0 for raw embeddings)
  uint gs;       // 32, 64, or 128
  uint sb_dtype; // 0 = bf16, 1 = f16, 2 = f32
  uint add_res;  // 0 = overwrite, 1 = Y += out
};

inline float load_sb(const device uchar* base, size_t idx, uint sb_dtype) {
  if (sb_dtype == 0u) return float(((const device bfloat*)base)[idx]);
  if (sb_dtype == 1u) return float(((const device half*)base)[idx]);
  return ((const device float*)base)[idx];
}

// Extract element `lane` (0..31) from a 32-element MLX affine chunk of `bits` u32 words.
// Exact across word boundaries for 3-bit (3 words), 5-bit (5 words), and 6-bit (6 words).
inline uint extract_q_lane(const device uint* w_chunk, ushort lane, uint bits, uint mask) {
  const uint bit_off = uint(lane) * bits;
  const uint wi = bit_off >> 5;
  const uint sh = bit_off & 31u;
  uint v = w_chunk[wi] >> sh;
  if (sh + bits > 32u) {
    v |= (w_chunk[wi + 1u] << (32u - sh));
  }
  return v & mask;
}

// Single-token (rows == 1) universal MLX affine GEMV for any bits in {2,3,4,5,6,8} & gs in {32,64,128}.
// Grid: threadgroups ((N + 3) / 4, 1, 1) of 128 threads (4 simdgroups, 1 output row per simdgroup).
kernel void qmv_affine_any(const device bfloat* X [[buffer(0)]],
                           const device uint* W [[buffer(1)]],
                           const device uchar* scales [[buffer(2)]],
                           const device uchar* biases [[buffer(3)]],
                           device bfloat* Y [[buffer(4)]],
                           constant AffineParams& p [[buffer(5)]],
                           uint grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint row_n = grp * 4u + uint(sg);
  if (row_n >= p.N) return;

  const uint bits = p.bits;
  const uint mask = (1u << bits) - 1u;
  const uint chunks = p.K >> 5;            // K / 32 (every 32 elems = `bits` u32 words)
  const uint words_per_row = chunks * bits;
  const uint chunks_per_g = p.gs >> 5;     // 1 (gs=32), 2 (gs=64), or 4 (gs=128)
  const uint groups = p.K / p.gs;

  const device uint* w_row = W + (size_t)row_n * words_per_row;
  const size_t sb_row = (size_t)row_n * groups;

  float acc = 0.0f;
  for (uint g = 0u; g < groups; ++g) {
    const float sc = load_sb(scales, sb_row + g, p.sb_dtype);
    const float bi = load_sb(biases, sb_row + g, p.sb_dtype);
    const uint c_base = g * chunks_per_g;
    for (uint ci = 0u; ci < chunks_per_g; ++ci) {
      const uint c = c_base + ci;
      const uint q = extract_q_lane(w_row + (size_t)c * bits, lane, bits, mask);
      const float xv = float(X[c * 32u + uint(lane)]);
      acc = fma(fma(float(q), sc, bi), xv, acc);
    }
  }
  const float total = simd_sum(acc);
  if (lane == 0u) {
    if (p.add_res != 0u) {
      Y[row_n] = bfloat(float(Y[row_n]) + total);
    } else {
      Y[row_n] = bfloat(total);
    }
  }
}

// Multi-row (rows >= 2) universal MLX affine GEMM with 4x weight reuse across 4 input rows.
// Grid: threadgroups ((N + 3) / 4, (rows + 3) / 4, 1) of 128 threads (4 simdgroups).
kernel void qmm_affine_any(const device bfloat* X [[buffer(0)]],
                           const device uint* W [[buffer(1)]],
                           const device uchar* scales [[buffer(2)]],
                           const device uchar* biases [[buffer(3)]],
                           device bfloat* Y [[buffer(4)]],
                           constant AffineParams& p [[buffer(5)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint row_n = grp.x * 4u + uint(sg);
  const uint m0 = grp.y * 4u;
  if (row_n >= p.N || m0 >= p.rows) return;

  const uint nc = min(4u, p.rows - m0);
  const uint bits = p.bits;
  const uint mask = (1u << bits) - 1u;
  const uint chunks = p.K >> 5;
  const uint words_per_row = chunks * bits;
  const uint chunks_per_g = p.gs >> 5;
  const uint groups = p.K / p.gs;

  const device uint* w_row = W + (size_t)row_n * words_per_row;
  const size_t sb_row = (size_t)row_n * groups;

  const device bfloat* x0 = X + (size_t)(m0 + 0u) * p.ldx;
  const device bfloat* x1 = X + (size_t)(m0 + (nc > 1u ? 1u : 0u)) * p.ldx;
  const device bfloat* x2 = X + (size_t)(m0 + (nc > 2u ? 2u : 0u)) * p.ldx;
  const device bfloat* x3 = X + (size_t)(m0 + (nc > 3u ? 3u : 0u)) * p.ldx;

  float4 acc = float4(0.0f);
  for (uint g = 0u; g < groups; ++g) {
    const float sc = load_sb(scales, sb_row + g, p.sb_dtype);
    const float bi = load_sb(biases, sb_row + g, p.sb_dtype);
    const uint c_base = g * chunks_per_g;
    for (uint ci = 0u; ci < chunks_per_g; ++ci) {
      const uint c = c_base + ci;
      const uint q = extract_q_lane(w_row + (size_t)c * bits, lane, bits, mask);
      const float wv = fma(float(q), sc, bi);
      const uint k_idx = c * 32u + uint(lane);
      acc.x = fma(wv, float(x0[k_idx]), acc.x);
      if (nc > 1u) acc.y = fma(wv, float(x1[k_idx]), acc.y);
      if (nc > 2u) acc.z = fma(wv, float(x2[k_idx]), acc.z);
      if (nc > 3u) acc.w = fma(wv, float(x3[k_idx]), acc.w);
    }
  }

  acc.x = simd_sum(acc.x);
  if (nc > 1u) acc.y = simd_sum(acc.y);
  if (nc > 2u) acc.z = simd_sum(acc.z);
  if (nc > 3u) acc.w = simd_sum(acc.w);

  if (lane == 0u) {
    for (uint r = 0u; r < nc; ++r) {
      device bfloat* dst = Y + (size_t)(m0 + r) * p.ldo + row_n;
      const float val = acc[r];
      *dst = (p.add_res != 0u) ? bfloat(float(*dst) + val) : bfloat(val);
    }
  }
}

// Single-token (rows == 1) lossless BF16 GEMV (vectorized vec<bfloat, 4> loads).
// Grid: threadgroups ((N + 3) / 4, 1, 1) of 128 threads (4 simdgroups).
kernel void bf16_linear_mv1(const device bfloat* X [[buffer(0)]],
                            const device bfloat* W [[buffer(1)]],
                            device bfloat* Y [[buffer(2)]],
                            constant AffineParams& p [[buffer(3)]],
                            uint grp [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  const uint row_n = grp * 4u + uint(sg);
  if (row_n >= p.N) return;

  const device bfloat* wr = W + (size_t)row_n * p.K;
  float acc = 0.0f;
  if ((p.K & 3u) == 0u) {
    const uint n4 = p.K >> 2;
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)wr;
    const device vec<bfloat, 4>* x4 = (const device vec<bfloat, 4>*)X;
    for (uint i = uint(lane); i < n4; i += 32u) {
      const float4 wv = float4(w4[i]);
      const float4 xv = float4(x4[i]);
      acc = fma(wv[0], xv[0], fma(wv[1], xv[1], fma(wv[2], xv[2], fma(wv[3], xv[3], acc))));
    }
  } else {
    for (uint c = uint(lane); c < p.K; c += 32u) {
      acc = fma(float(wr[c]), float(X[c]), acc);
    }
  }
  const float total = simd_sum(acc);
  if (lane == 0u) {
    Y[row_n] = (p.add_res != 0u) ? bfloat(float(Y[row_n]) + total) : bfloat(total);
  }
}

// Multi-row (rows >= 2) lossless BF16 GEMM with 4x weight reuse across 4 input rows.
// Grid: threadgroups ((N + 3) / 4, (rows + 3) / 4, 1) of 128 threads (4 simdgroups).
kernel void bf16_linear_mm(const device bfloat* X [[buffer(0)]],
                           const device bfloat* W [[buffer(1)]],
                           device bfloat* Y [[buffer(2)]],
                           constant AffineParams& p [[buffer(3)]],
                           uint2 grp [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  const uint row_n = grp.x * 4u + uint(sg);
  const uint m0 = grp.y * 4u;
  if (row_n >= p.N || m0 >= p.rows) return;

  const uint nc = min(4u, p.rows - m0);
  const device bfloat* wr = W + (size_t)row_n * p.K;
  const device bfloat* x0 = X + (size_t)(m0 + 0u) * p.ldx;
  const device bfloat* x1 = X + (size_t)(m0 + (nc > 1u ? 1u : 0u)) * p.ldx;
  const device bfloat* x2 = X + (size_t)(m0 + (nc > 2u ? 2u : 0u)) * p.ldx;
  const device bfloat* x3 = X + (size_t)(m0 + (nc > 3u ? 3u : 0u)) * p.ldx;

  float4 acc = float4(0.0f);
  if ((p.K & 3u) == 0u && (p.ldx & 3u) == 0u) {
    const uint n4 = p.K >> 2;
    const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)wr;
    const device vec<bfloat, 4>* a0 = (const device vec<bfloat, 4>*)x0;
    const device vec<bfloat, 4>* a1 = (const device vec<bfloat, 4>*)x1;
    const device vec<bfloat, 4>* a2 = (const device vec<bfloat, 4>*)x2;
    const device vec<bfloat, 4>* a3 = (const device vec<bfloat, 4>*)x3;
    for (uint i = uint(lane); i < n4; i += 32u) {
      const float4 wv = float4(w4[i]);
      acc.x += dot(wv, float4(a0[i]));
      if (nc > 1u) acc.y += dot(wv, float4(a1[i]));
      if (nc > 2u) acc.z += dot(wv, float4(a2[i]));
      if (nc > 3u) acc.w += dot(wv, float4(a3[i]));
    }
  } else {
    for (uint c = uint(lane); c < p.K; c += 32u) {
      const float wv = float(wr[c]);
      acc.x = fma(wv, float(x0[c]), acc.x);
      if (nc > 1u) acc.y = fma(wv, float(x1[c]), acc.y);
      if (nc > 2u) acc.z = fma(wv, float(x2[c]), acc.z);
      if (nc > 3u) acc.w = fma(wv, float(x3[c]), acc.w);
    }
  }

  acc.x = simd_sum(acc.x);
  if (nc > 1u) acc.y = simd_sum(acc.y);
  if (nc > 2u) acc.z = simd_sum(acc.z);
  if (nc > 3u) acc.w = simd_sum(acc.w);

  if (lane == 0u) {
    for (uint r = 0u; r < nc; ++r) {
      device bfloat* dst = Y + (size_t)(m0 + r) * p.ldo + row_n;
      const float val = acc[r];
      *dst = (p.add_res != 0u) ? bfloat(float(*dst) + val) : bfloat(val);
    }
  }
}

// Universal embedding lookup supporting:
//   - bits == 0: raw BF16 (sb_dtype=0), F16 (sb_dtype=1), or F32 (sb_dtype=2) embeddings
//   - bits in {2,3,4,5,6,8}: MLX affine quantized embeddings with gs in {32,64,128}
// Grid: threadgroups ((hidden + 31) / 32, rows, 1) of 32 threads.
kernel void embed_lookup_any(const device uint* tokens [[buffer(0)]],
                             const device uchar* codes [[buffer(1)]],
                             const device uchar* scales [[buffer(2)]],
                             const device uchar* biases [[buffer(3)]],
                             device bfloat* out [[buffer(4)]],
                             constant AffineParams& p [[buffer(5)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint chunk = grp.x;
  const uint row = grp.y;
  const uint col = chunk * 32u + uint(lane);
  if (row >= p.rows || col >= p.K) return;

  const uint tok = tokens[row];
  device bfloat* dst = out + (size_t)row * p.K + col;

  if (p.bits == 0u) {
    const size_t idx = (size_t)tok * p.K + col;
    *dst = bfloat(load_sb(codes, idx, p.sb_dtype));
    return;
  }

  const uint bits = p.bits;
  const uint mask = (1u << bits) - 1u;
  const uint chunks = p.K >> 5;
  const uint words_per_row = chunks * bits;
  const uint g = col / p.gs;
  const uint groups = p.K / p.gs;

  const device uint* w_row = ((const device uint*)codes) + (size_t)tok * words_per_row;
  const uint q = extract_q_lane(w_row + (size_t)chunk * bits, lane, bits, mask);
  const float sc = load_sb(scales, (size_t)tok * groups + g, p.sb_dtype);
  const float bi = load_sb(biases, (size_t)tok * groups + g, p.sb_dtype);
  *dst = bfloat(fma(float(q), sc, bi));
}
