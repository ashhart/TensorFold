// Universal Apple Silicon (M1-M5, Metal 3.1+) 4-bit affine quantized GEMV & GEMM kernels.
//
// Zero MetalPerformancePrimitives / M5-only tensor_ops dependencies!
// Reads the exact same packed 1152-byte (NT=32 cols, GS=64 group) weight block layout as qk.metal:
//   [0 .. 1024):    NT * 32 bytes of packed 4-bit unsigned codes (low nibble = even k, high = odd k)
//   [1024 .. 1152): NT * 4 bytes of interleaved {scale_bf16 (low16), bias_bf16 (high16)}
//
// Kernels:
//   - qmv_1           : 1-row coalesced 128-bit 4-bit GEMV (2x X register reuse)
//   - qmv_1_silu_mul  : Fused 1-row Gate+Up 4-bit GEMV + SiLU(gate)*up (4x X register reuse)
//   - qmv_1_argmax    : Fused 1-row LM-Head 4-bit GEMV + Stage-1 Argmax per 32-vocab tile
//   - qmm_simd_8      : Universal multi-row 4-bit GEMM tile (M = 1..8)  with 8x weight reuse
//   - qmm_simd_16     : Universal multi-row 4-bit GEMM tile (M = 1..16) with 16x weight reuse
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int GS = 64;
constant constexpr int NT = 32;
constant constexpr int SK = 4;
constant constexpr int CODES = NT * GS / 2;   // 1024 bytes
constant constexpr int BLOCK = CODES + NT * 4; // 1152 bytes

struct QkParams {
  int M;           // active rows in this dispatch (<= ROWS)
  int K;           // input width (multiple of 64)
  int N;           // output width (multiple of 32)
  int G;           // K / 64
  int first_row;   // starting row index in X, SUMS, Y
  int sums_stride; // stride between groups in SUMS (multiple of 16)
  uint tile_first; // first 32-col tile handled by this dispatch
};

// Universal multi-row 4-bit quantized GEMM over the packed qk tile layout (Metal 3.1+, M1-M5).
//
// 1 threadgroup = 4 simdgroups (128 threads, SK=4 K-slices) computing one 32-col tile x ROWS rows.
// Each simdgroup cooperatively stages its group's X[0..ROWS-1, 64] into threadgroup memory (zero
// cross-simdgroup barrier inside the K-loop!), then each of the 32 lanes owns 1 output column,
// decodes its 64 4-bit nibbles ONCE, and reuses them across all ROWS rows via broadcast TG reads.
template <int ROWS>
kernel void qmm_simd_tile(const device bfloat* X [[buffer(0)]],
                          const device uchar* W [[buffer(1)]],
                          const device float* SUMS [[buffer(2)]],
                          device bfloat* Y [[buffer(3)]],
                          constant QkParams& p [[buffer(4)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint slice [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint tile = p.tile_first + group;
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const device uchar* blocks = W + (size_t)tile * (size_t)p.G * BLOCK;
  const int g_first = p.G * (int)slice / SK;
  const int g_last = p.G * (int)(slice + 1) / SK;
  const int end_row = p.first_row + p.M;
  const int last_valid = p.M - 1;

  // Shared staging buffer reused after K-loop for cross-simdgroup reduction:
  //   Phase 1 (K-loop):  x_sh[SK][ROWS][64] of bfloat = SK * ROWS * 128 bytes (4 KiB / 8 KiB)
  //   Phase 2 (reduce):  partial[SK-1][ROWS][32] of float = 3 * ROWS * 128 bytes (3 KiB / 6 KiB)
  threadgroup ushort tg_mem[SK * ROWS * 64];
  threadgroup bfloat* x_slice = ((threadgroup bfloat*)tg_mem) + (size_t)slice * ROWS * GS;

  float acc[ROWS];
  UNROLL for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

  const size_t col_off = (size_t)lane * 32;
  const size_t sb_off = CODES + (size_t)lane * 4;

  for (int g = g_first; g < g_last; g++) {
    // 1. Cooperatively stage X[first_row .. first_row + ROWS - 1, g*64 .. g*64+63] for this simdgroup.
    //    32 lanes x 2 bfloat = 64 bfloats per row (single coalesced 32-bit load per lane per row).
    const int k_base = g * GS + (int)lane * 2;
    UNROLL for (int r = 0; r < ROWS; r++) {
      const int m = p.first_row + min(r, last_valid);
      const vec<bfloat, 2> v2 = *(const device vec<bfloat, 2>*)(X + (size_t)m * p.K + k_base);
      *(threadgroup vec<bfloat, 2>*)(x_slice + r * GS + lane * 2) = v2;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Load this lane's 64 packed 4-bit weights (two 128-bit coalesced bursts) + {scale, bias}.
    const device uchar* blk = blocks + (size_t)g * BLOCK;
    const uint4 w_lo = *(const device uint4*)(blk + col_off);
    const uint4 w_hi = *(const device uint4*)(blk + col_off + 16);
    const uint sb = *(const device uint*)(blk + sb_off);
    const float scale = as_type<float>(sb << 16);
    const float bias = as_type<float>(sb & 0xFFFF0000u);

    float dot[ROWS];
    UNROLL for (int r = 0; r < ROWS; r++) dot[r] = 0.0f;

    // 3. Unpack 8 nibbles at a time and accumulate across all ROWS rows (hardware broadcast read!).
    UNROLL for (int wi = 0; wi < 4; wi++) {
      const uint w = w_lo[wi];
      const float4 qa = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu));
      const float4 qb = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu));
      const int k_off = wi * 8;
      UNROLL for (int r = 0; r < ROWS; r++) {
        const float4 xa = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off));
        const float4 xb = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off + 4));
        const float4 p4 = fma(qa, xa, qb * xb);
        dot[r] += (p4[0] + p4[1]) + (p4[2] + p4[3]);
      }
    }
    UNROLL for (int wi = 0; wi < 4; wi++) {
      const uint w = w_hi[wi];
      const float4 qa = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu));
      const float4 qb = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu));
      const int k_off = 32 + wi * 8;
      UNROLL for (int r = 0; r < ROWS; r++) {
        const float4 xa = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off));
        const float4 xb = float4(*(const threadgroup vec<bfloat, 4>*)(x_slice + r * GS + k_off + 4));
        const float4 p4 = fma(qa, xa, qb * xb);
        dot[r] += (p4[0] + p4[1]) + (p4[2] + p4[3]);
      }
    }

    const device float* gs = SUMS + (size_t)g * p.sums_stride;
    UNROLL for (int r = 0; r < ROWS; r++) {
      const int m = p.first_row + min(r, last_valid);
      acc[r] = fma(scale, dot[r], fma(bias, gs[m], acc[r]));
    }
  }

  // Wait for all simdgroups to finish reading x_slice before repurposing tg_mem for partial reduction.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* partial = (threadgroup float*)tg_mem;
  if (slice > 0) {
    UNROLL for (int r = 0; r < ROWS; r++) {
      partial[((slice - 1) * ROWS + r) * NT + lane] = acc[r];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slice > 0) return;

  const int out_col = (int)base_col + (int)lane;
  UNROLL for (int r = 0; r < ROWS; r++) {
    const int m = p.first_row + r;
    if (m < end_row) {
      float total = acc[r];
      UNROLL for (int s = 0; s < SK - 1; s++) {
        total += partial[(s * ROWS + r) * NT + lane];
      }
      Y[(size_t)m * p.N + out_col] = bfloat(total);
    }
  }
}

template [[host_name("qmm_simd_8")]] kernel void qmm_simd_tile<8>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qmm_simd_16")]] kernel void qmm_simd_tile<16>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint, uint, uint);

// Exact IEEE-754 zero-shift 4-bit dot product over 8 packed nibbles (one uint32 word).
// Pre-scaling X by exact powers of 2 (1, 1/16, 1/256, 1/4096) eliminates 7 of 8 bit shifts
// while staying 100% bit-exact in FP32 and maximizing FMA ILP.
inline float q4_dot8(uint w, float4 sa, float4 sb) {
  const ushort lo = ushort(w);
  const ushort hi = ushort(w >> 16);
  const float d0 = fma(float(lo & 0x000Fu), sa[0],
                   fma(float(lo & 0x00F0u), sa[1],
                   fma(float(lo & 0x0F00u), sa[2],
                       float(lo & 0xF000u) * sa[3])));
  const float d1 = fma(float(hi & 0x000Fu), sb[0],
                   fma(float(hi & 0x00F0u), sb[1],
                   fma(float(hi & 0x0F00u), sb[2],
                       float(hi & 0xF000u) * sb[3])));
  return d0 + d1;
}

// Ultra-fast 1-row 4-bit quantized GEMV combining:
//   - 128-bit coalesced uint4 weight bursts (4x fewer load instructions than uint32!)
//   - 100% contiguous 1152-byte (32-col) block streaming (zero skipped cache lines!)
//   - Exact IEEE-754 zero-shift power-of-2 nibble unpacking (q4_dot8)
//   - Minimal register footprint (only 2 accumulators/lane -> peak GPU occupancy!)
//   - Optional fused residual add (`p.tile_first != 0`) directly into h_buf
//   - Adaptive K-splitting (SK_V = 1, 2, 4, 8)
// Grid: threadgroups (N / 32, 1, 1) of (SK_V * 32) threads.
template <int SK_V>
kernel void qmv_1_u4(const device bfloat* X [[buffer(0)]],
                     const device uchar* W [[buffer(1)]],
                     device bfloat* Y [[buffer(2)]],
                     constant QkParams& p [[buffer(3)]],
                     uint tile [[threadgroup_position_in_grid]],
                     uint slice [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1;   // 0..15
  const uint col1 = col0 + 16;   // 16..31
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)tile * (size_t)p.G * BLOCK;
  const int g_first = (SK_V == 1) ? 0 : (p.G * (int)slice / SK_V);
  const int g_last = (SK_V == 1) ? p.G : (p.G * (int)(slice + 1) / SK_V);

  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int g = g_first; g < g_last; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      d0 += q4_dot8(w4_0[wi], sa, sb_x);
      d1 += q4_dot8(w4_1[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  if (SK_V == 1) {
    if (half_idx == 0) {
      if (p.tile_first != 0) {
        Y[base_col + col0] = bfloat(float(Y[base_col + col0]) + float(bfloat(acc0)));
        Y[base_col + col1] = bfloat(float(Y[base_col + col1]) + float(bfloat(acc1)));
      } else {
        Y[base_col + col0] = bfloat(acc0);
        Y[base_col + col1] = bfloat(acc1);
      }
    }
    return;
  }

  threadgroup float partial[SK_V][NT];
  if (half_idx == 0) {
    partial[slice][col0] = acc0;
    partial[slice][col1] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0) {
    float total = partial[0][lane];
    UNROLL for (int s = 1; s < SK_V; s++) total += partial[s][lane];
    const uint out_idx = base_col + lane;
    if (p.tile_first != 0) {
      Y[out_idx] = bfloat(float(Y[out_idx]) + float(bfloat(total)));
    } else {
      Y[out_idx] = bfloat(total);
    }
  }
}

template [[host_name("qmv_1")]] kernel void qmv_1_u4<1>(
    const device bfloat*, const device uchar*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qmv_1_sk2")]] kernel void qmv_1_u4<2>(
    const device bfloat*, const device uchar*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qmv_1_narrow")]] kernel void qmv_1_u4<4>(
    const device bfloat*, const device uchar*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qmv_1_8col")]] kernel void qmv_1_u4<8>(
    const device bfloat*, const device uchar*, device bfloat*, constant QkParams&, uint, uint, uint);

// Dual-simdgroup 100%-contiguous 32-col 128-bit uint4 Fused Gate+Up GEMV + SiLU(gate)*up!
//   - sg 0 streams ALL 32 cols of gate_tile via 128-bit uint4 bursts (matches qmv_1_argmax 572 GB/s!)
//   - sg 1 streams ALL 32 cols of up_tile   via 128-bit uint4 bursts (only 2 accumulators/lane!)
// Zero skipped cache lines, zero bit-shifts (q4_dot8), 16B coalesced bursts!
// Grid: threadgroups (inter / 32, 1, 1) of 64 threads (2 simdgroups).
kernel void qmv_1_silu_mul(const device bfloat* X [[buffer(0)]],
                           const device uchar* W [[buffer(1)]],
                           device bfloat* Y [[buffer(2)]],
                           constant QkParams& p [[buffer(3)]],
                           uint tile [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1;   // 0..15
  const uint col1 = col0 + 16;   // 16..31
  const uint half_idx = lane & 1u;

  const size_t tile_bytes = (size_t)p.G * BLOCK;
  const uint w_tile = tile + sg * (uint)(p.N / NT);
  const device uchar* tile_w = W + (size_t)w_tile * tile_bytes;

  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int g = 0; g < p.G; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      d0 += q4_dot8(w4_0[wi], sa, sb_x);
      d1 += q4_dot8(w4_1[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  threadgroup float tg_vals[2][NT];
  if (half_idx == 0) {
    tg_vals[sg][col0] = acc0;
    tg_vals[sg][col1] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    const float g_val = float(bfloat(tg_vals[0][lane]));
    const float u_val = float(bfloat(tg_vals[1][lane]));
    Y[base_col + lane] = bfloat((g_val / (1.0f + exp(-g_val))) * u_val);
  }
}

// High-occupancy SK=4 16-col 128-bit uint4 Gate+Up GEMV + SiLU(gate)*up for small intermediate widths (<=4096, e.g. 0.8B).
// Launches 896 simdgroups on 0.8B (vs 224 for 32-col dual-SG) for peak small-model decode (>575 tok/s!).
// Grid: threadgroups (inter / 16, 1, 1) of 128 threads.
kernel void qmv_1_silu_mul_sk4(const device bfloat* X [[buffer(0)]],
                               const device uchar* W [[buffer(1)]],
                               device bfloat* Y [[buffer(2)]],
                               constant QkParams& p [[buffer(3)]],
                               uint grp [[threadgroup_position_in_grid]],
                               uint slice [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = grp * 16;
  if ((int)base_col >= p.N) return;

  const uint tile = grp >> 1;
  const uint sub_col = (grp & 1u) * 16;
  const uint col_local = lane >> 1;
  const uint col = sub_col + col_local;
  const uint half_idx = lane & 1u;

  const size_t tile_bytes = (size_t)p.G * BLOCK;
  const uint up_tile = tile + (uint)(p.N / NT);
  const device uchar* gate_w = W + (size_t)tile * tile_bytes;
  const device uchar* up_w = W + (size_t)up_tile * tile_bytes;

  const int g_first = p.G * (int)slice / SK;
  const int g_last = p.G * (int)(slice + 1) / SK;

  const size_t off = (size_t)col * 32 + (size_t)half_idx * 16;
  const size_t sb_off = CODES + (size_t)col * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float g_acc = 0.0f;
  float u_acc = 0.0f;

  for (int g = g_first; g < g_last; g++) {
    const device uchar* gb = gate_w + (size_t)g * BLOCK;
    const device uchar* ub = up_w + (size_t)g * BLOCK;

    const uint4 wg = *(const device uint4*)(gb + off);
    const uint4 wu = *(const device uint4*)(ub + off);
    const uint gsb = *(const device uint*)(gb + sb_off);
    const uint usb = *(const device uint*)(ub + sb_off);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float gd = 0.0f;
    float ud = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      gd += q4_dot8(wg[wi], sa, sb_x);
      ud += q4_dot8(wu[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    g_acc = fma(as_type<float>(gsb << 16), gd, fma(as_type<float>(gsb & 0xFFFF0000u), xsum, g_acc));
    u_acc = fma(as_type<float>(usb << 16), ud, fma(as_type<float>(usb & 0xFFFF0000u), xsum, u_acc));
  }

  g_acc += simd_shuffle_xor(g_acc, 1);
  u_acc += simd_shuffle_xor(u_acc, 1);

  threadgroup float pg[SK][16];
  threadgroup float pu[SK][16];
  if (half_idx == 0) {
    pg[slice][col_local] = g_acc;
    pu[slice][col_local] = u_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0 && lane < 16) {
    const float g_val = float(bfloat((pg[0][lane] + pg[1][lane]) + (pg[2][lane] + pg[3][lane])));
    const float u_val = float(bfloat((pu[0][lane] + pu[1][lane]) + (pu[2][lane] + pu[3][lane])));
    Y[base_col + lane] = bfloat((g_val / (1.0f + exp(-g_val))) * u_val);
  }
}

// Single-simdgroup zero-barrier 128-bit uint4 Fused LM-Head GEMV + Stage-1 Argmax!
// 100% contiguous 1152B block streaming, zero-shift q4_dot8, ZERO TG barriers/memory!
// Grid: threadgroups (vocab_tiles, 1, 1) of 32 threads.
kernel void qmv_1_argmax(const device bfloat* X [[buffer(0)]],
                         const device uchar* W [[buffer(1)]],
                         device float* chunk_val [[buffer(2)]],
                         device uint* chunk_idx [[buffer(3)]],
                         constant QkParams& p [[buffer(4)]],
                         uint tile [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1;
  const uint col1 = col0 + 16;
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)tile * (size_t)p.G * BLOCK;
  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;
  const float4 inv_p2 = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int g = 0; g < p.G; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;
      const float4 sa = xa * inv_p2;
      const float4 sb_x = xb * inv_p2;
      d0 += q4_dot8(w4_0[wi], sa, sb_x);
      d1 += q4_dot8(w4_1[wi], sa, sb_x);
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  // Lane 2*c owns col c (acc0) and col c+16 (acc1); reduce across all 32 cols with zero TG mem!
  const uint idx0 = base_col + col0;
  const uint idx1 = base_col + col1;
  const float r0 = float(bfloat(acc0));
  const float r1 = float(bfloat(acc1));
  float best_v = -INFINITY;
  uint best_i = 0xFFFFFFFFu;
  if (half_idx == 0) {
    if ((int)idx0 < p.N && isfinite(r0)) { best_v = r0; best_i = idx0; }
    if ((int)idx1 < p.N && isfinite(r1) && (r1 > best_v || (r1 == best_v && idx1 < best_i))) {
      best_v = r1; best_i = idx1;
    }
  }
  UNROLL for (ushort off = 16; off >= 2; off >>= 1) {
    const float ov = simd_shuffle_down(best_v, off);
    const uint oi = simd_shuffle_down(best_i, off);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    chunk_val[tile] = best_v;
    chunk_idx[tile] = best_i;
  }
}
