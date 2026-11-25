// Clean-room M5 hardware tensor-unit 4-bit affine matrix multiplication (qk).
//
//   Y[m, n] = sum_{g=0}^{G-1} ( scale[n, g] * (X[m, g] . codes[n, g]) + bias[n, g] * SUMS[g, m] )
//
// Packed weight block layout per (tile of NT=32 output cols, group of GS=64 input cols):
//   [0 .. 1024):   NT * 32 bytes of packed 4-bit unsigned codes (uint4b_format)
//   [1024 .. 1152): NT * 4 bytes of interleaved {scale_bf16 (low16), bias_bf16 (high16)}
//
// Instantiates ROWS = 16, 32, 48 in a single Metal 4 library so one kernel family covers
// single-token decode, multi-row tree verification, and chunked prompt prefill.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int GS = 64;
constant constexpr int NT = 32;
constant constexpr int SK = 4;
constant constexpr int CODES = NT * GS / 2;   // 1024 bytes
constant constexpr int BLOCK = CODES + NT * 4; // 1152 bytes
constant constexpr int SETS = NT / 16;         // 2 column sets of 4 outputs per lane

struct QkParams {
  int M;           // active rows in this dispatch (<= ROWS)
  int K;           // input width (multiple of 64)
  int N;           // output width (multiple of 32)
  int G;           // K / 64
  int first_row;   // starting row index in X, SUMS, Y
  int sums_stride; // stride between groups in SUMS (multiple of 16)
  uint tile_first; // first 32-col tile handled by this dispatch
};

inline int opPlace(int k, int s, int j) { return (s * 2 + k) * 4 + j; }
inline int accIdx(int k, int s, int j) { return (k * SETS + s) * 4 + j; }

template <int ROWS>
kernel void qk_kernel(const device bfloat* X [[buffer(0)]],
                      const device uchar* W [[buffer(1)]],
                      const device float* SUMS [[buffer(2)]],
                      device bfloat* Y [[buffer(3)]],
                      constant QkParams& p [[buffer(4)]],
                      uint2 grp [[threadgroup_position_in_grid]],
                      uint slice [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr int RUNS = ROWS / 8;
  constexpr int VALUES = RUNS * SETS * 4;

  const uint m_tile = grp.x;
  const uint tile = p.tile_first + grp.y;
  const int row0 = p.first_row + (int)m_tile * ROWS;
  const int end_row = min(row0 + ROWS, p.first_row + p.M);
  if (row0 >= end_row) return;
  const device uchar* blocks = W + tile * (size_t)p.G * BLOCK;
  const int first = p.G * (int)slice / SK;
  const int last = p.G * (int)(slice + 1) / SK;

  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> xs((device bfloat*)X, dextents<int32_t, 2>(p.K, end_row));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)W, dextents<int32_t, 2>(GS, NT));
  auto LX = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
  auto P = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(LX)>, decltype(b0), float>();

  const auto home = P.get_multidimensional_index(ushort(0));
  const int c0 = home[0];
  const int r0 = home[1];

  float C[VALUES];
  UNROLL for (int i = 0; i < VALUES; i++) C[i] = 0.0f;

  // Prefetch first group's interleaved {scale, bias} words.
  uint4 next[SETS];
  UNROLL for (int s = 0; s < SETS; s++) {
    next[s] = *(const device uint4*)(blocks + (size_t)first * BLOCK + CODES + (c0 + 16 * s) * 4);
  }

  for (int g = first; g < last; g++) {
    float scale[SETS][4];
    float bias[SETS][4];
    UNROLL for (int s = 0; s < SETS; s++) {
      UNROLL for (int j = 0; j < 4; j++) {
        scale[s][j] = as_type<float>(next[s][j] << 16);
        bias[s][j] = as_type<float>(next[s][j] & 0xFFFF0000u);
      }
      next[s] = *(const device uint4*)(blocks + (size_t)(g + 1) * BLOCK + CODES + (c0 + 16 * s) * 4);
    }

    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)blocks + (size_t)g * BLOCK, dextents<int32_t, 2>(GS, NT));

    UNROLL for (int part = 0; part < RUNS / 2; part++) {
      auto a = xs.slice(g * GS, row0 + part * 16);
      LX.load(a);
      op.run(LX, b, P);
      const int row_base = row0 + part * 16 + r0;
      const device float* gs = SUMS + (size_t)g * p.sums_stride;
      const float xs0 = (row_base < end_row) ? gs[row_base] : 0.0f;
      const float xs1 = (row_base + 8 < end_row) ? gs[row_base + 8] : 0.0f;
      UNROLL for (int s = 0; s < SETS; s++) {
        UNROLL for (int j = 0; j < 4; j++) {
          C[accIdx(part * 2, s, j)] = fma(scale[s][j], P[opPlace(0, s, j)], fma(bias[s][j], xs0, C[accIdx(part * 2, s, j)]));
          C[accIdx(part * 2 + 1, s, j)] = fma(scale[s][j], P[opPlace(1, s, j)], fma(bias[s][j], xs1, C[accIdx(part * 2 + 1, s, j)]));
        }
      }
    }
  }

  // Combine K-slices across the 4 simdgroups in threadgroup memory (2-phase when VALUES > 32 to cap TG mem at 12 KiB).
  constexpr int CHUNK = (VALUES > 32) ? (VALUES / 2) : VALUES;
  threadgroup float partial[(SK - 1) * CHUNK * 32];
  UNROLL for (int base = 0; base < VALUES; base += CHUNK) {
    if (slice > 0) {
      UNROLL for (int i = 0; i < CHUNK; i++) {
        partial[((slice - 1) * CHUNK + i) * 32 + lane] = C[base + i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice == 0) {
      UNROLL for (int s = 0; s < SK - 1; s++) {
        UNROLL for (int i = 0; i < CHUNK; i++) {
          C[base + i] += partial[(s * CHUNK + i) * 32 + lane];
        }
      }
    }
    if (base + CHUNK < VALUES) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (slice > 0) return;

  const int n0 = (int)tile * NT + c0;
  UNROLL for (int k = 0; k < RUNS; k++) {
    const int m = row0 + r0 + 8 * k;
    if (m < end_row) {
      device bfloat* row = Y + (size_t)m * p.N;
      UNROLL for (int s = 0; s < SETS; s++) {
        const vec<bfloat, 4> four = vec<bfloat, 4>(float4(
            C[accIdx(k, s, 0)], C[accIdx(k, s, 1)], C[accIdx(k, s, 2)], C[accIdx(k, s, 3)]));
        *(device vec<bfloat, 4>*)(row + n0 + 16 * s) = four;
      }
    }
  }
}

template [[host_name("qk_16")]] kernel void qk_kernel<16>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint2, uint, uint);
template [[host_name("qk_32")]] kernel void qk_kernel<32>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint2, uint, uint);
template [[host_name("qk_48")]] kernel void qk_kernel<48>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint2, uint, uint);
template [[host_name("qk_64")]] kernel void qk_kernel<64>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint2, uint, uint);

struct MoeQkParams {
  uint total_slots;
  uint n_experts;
  int K;
  int N;
  int G;
  int sums_stride;
  uint apply_relu2;
  uint scatter_out;
  ulong expert_stride;
};

// M5 hardware tensor-unit batched 4-bit MoE expert GEMM for prefill.
// Reuses each expert's weights across up to 16 bucketed tokens per hardware matmul2d tile!
// Grid: threadgroups (N / 32, n_experts, 1) of 128 threads (SK = 4).
kernel void moe_expert_qk16_batched(const device bfloat* X [[buffer(0)]],
                                    const device uchar* W [[buffer(1)]],
                                    const device float* SUMS [[buffer(2)]],
                                    const device uint* expert_offsets [[buffer(3)]],
                                    const device uint* expert_counts [[buffer(4)]],
                                    const device uint* bucket_slots [[buffer(5)]],
                                    device bfloat* Y [[buffer(6)]],
                                    constant MoeQkParams& p [[buffer(7)]],
                                    uint2 grp [[threadgroup_position_in_grid]],
                                    uint slice [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 16;
  constexpr int RUNS = ROWS / 8; // 2
  constexpr int VALUES = RUNS * SETS * 4; // 16

  const uint tile = grp.x;
  const uint e = grp.y;
  if ((int)(tile * NT) >= p.N || e >= p.n_experts) return;

  const uint count = expert_counts[e];
  if (count == 0) return;

  const uint start_idx = expert_offsets[e];
  const device uchar* blocks = W + (size_t)e * (size_t)p.expert_stride + (size_t)tile * (size_t)p.G * BLOCK;
  const int first = p.G * (int)slice / SK;
  const int last = p.G * (int)(slice + 1) / SK;

  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)blocks, dextents<int32_t, 2>(GS, NT));
  auto LX = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
  auto P = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(LX)>, decltype(b0), float>();

  const auto home = P.get_multidimensional_index(ushort(0));
  const int c0 = home[0];
  const int r0 = home[1];
  const int n0 = (int)tile * NT + c0;

  threadgroup float partial[(SK - 1) * VALUES * 32];

  for (uint chunk = 0; chunk < count; chunk += ROWS) {
    const int first_row = (int)(start_idx + chunk);
    const int m_active = (int)min((uint)ROWS, count - chunk);
    const int end_row = first_row + m_active;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> xs((device bfloat*)X, dextents<int32_t, 2>(p.K, end_row));

    float C[VALUES];
    UNROLL for (int i = 0; i < VALUES; i++) C[i] = 0.0f;

    uint4 next[SETS];
    UNROLL for (int s = 0; s < SETS; s++) {
      next[s] = *(const device uint4*)(blocks + (size_t)first * BLOCK + CODES + (c0 + 16 * s) * 4);
    }

    for (int g = first; g < last; g++) {
      float scale[SETS][4];
      float bias[SETS][4];
      UNROLL for (int s = 0; s < SETS; s++) {
        UNROLL for (int j = 0; j < 4; j++) {
          scale[s][j] = as_type<float>(next[s][j] << 16);
          bias[s][j] = as_type<float>(next[s][j] & 0xFFFF0000u);
        }
        next[s] = *(const device uint4*)(blocks + (size_t)(g + 1) * BLOCK + CODES + (c0 + 16 * s) * 4);
      }

      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
          (device uchar*)blocks + (size_t)g * BLOCK, dextents<int32_t, 2>(GS, NT));

      auto a = xs.slice(g * GS, first_row);
      LX.load(a);
      op.run(LX, b, P);
      const device float* gs = SUMS + (size_t)g * p.sums_stride + first_row + r0;
      const float xs0 = gs[0];
      const float xs1 = (first_row + r0 + 8 < end_row) ? gs[8] : 0.0f;
      UNROLL for (int s = 0; s < SETS; s++) {
        UNROLL for (int j = 0; j < 4; j++) {
          C[accIdx(0, s, j)] = fma(scale[s][j], P[opPlace(0, s, j)], fma(bias[s][j], xs0, C[accIdx(0, s, j)]));
          C[accIdx(1, s, j)] = fma(scale[s][j], P[opPlace(1, s, j)], fma(bias[s][j], xs1, C[accIdx(1, s, j)]));
        }
      }
    }

    if (slice > 0) {
      UNROLL for (int i = 0; i < VALUES; i++) {
        partial[((slice - 1) * VALUES + i) * 32 + lane] = C[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (slice == 0) {
      UNROLL for (int s = 0; s < SK - 1; s++) {
        UNROLL for (int i = 0; i < VALUES; i++) {
          C[i] += partial[(s * VALUES + i) * 32 + lane];
        }
      }
      UNROLL for (int k = 0; k < RUNS; k++) {
        const int m = first_row + r0 + 8 * k;
        if (m < end_row) {
          const uint dst_row = (p.scatter_out != 0) ? bucket_slots[m] : (uint)m;
          device bfloat* row = Y + (size_t)dst_row * p.N;
          UNROLL for (int s = 0; s < SETS; s++) {
            float4 fv = float4(C[accIdx(k, s, 0)], C[accIdx(k, s, 1)], C[accIdx(k, s, 2)], C[accIdx(k, s, 3)]);
            if (p.apply_relu2 != 0) {
              UNROLL for (int j = 0; j < 4; j++) {
                const float v = float(bfloat(fv[j]));
                fv[j] = (v > 0.0f) ? (v * v) : 0.0f;
              }
            }
            *(device vec<bfloat, 4>*)(row + n0 + 16 * s) = vec<bfloat, 4>(fv);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
