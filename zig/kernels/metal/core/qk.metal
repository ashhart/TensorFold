// qk: y = sum over groups of scale * (x . codes) + bias * sum(x), the op reading packed codes; tf_lane_reg's words.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// TF_SK: simdgroups a tile on slices of the groups (the K split, which sets the bits); TF_ROWS: rows a threadgroup.
#define TF_QK_EACH _Pragma("clang loop unroll(full)")

constant constexpr int QK_NT = 32;                       // columns in a tile
constant constexpr int QK_GS = 64;                       // codes in a group
constant constexpr int QK_CODES = QK_NT * QK_GS / 2;     // bytes of codes in one block: a tile's group
constant constexpr int QK_SETS = QK_NT / 16;             // column sets one lane holds: columns c0 + 16 s, four each
constant constexpr int QK_RUNS = TF_ROWS / 8;            // row runs one lane holds: rows r0 + 8 k
constant constexpr int QK_VALUES = QK_RUNS * QK_SETS * 4; // values one lane keeps for a tile

struct tf_qk_args {
  int rows;    // rows of x and y, all threadgroups together
  int k;       // columns of x
  int n;       // columns of y, a multiple of 32
  int groups;  // k / 64
  int stride;  // rows between one group's sums and the next's
};

// Where one op puts a lane's values: column sets, then the two runs of its 16 rows.
inline int tf_qk_place(int k, int s, int j) { return (s * 2 + k) * 4 + j; }
// Where this kernel keeps them.
inline int tf_qk_kept(int k, int s, int j) { return (k * QK_SETS + s) * 4 + j; }

kernel void tf_qk(const device bfloat* X [[buffer(0)]],
                  const device uchar* W [[buffer(1)]],      // weight_layout: [tile][group][column][32 bytes]
                  const device uint* SB [[buffer(2)]],      // metadata_layout as words: [group][column] scale | bias << 16
                  const device float* SUMS [[buffer(3)]],   // x summed over each group: [group * stride + row]
                  device bfloat* Y [[buffer(4)]],
                  constant tf_qk_args& p [[buffer(5)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  uint slice [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  const int tile = int(tg.x);
  const int first_row = int(tg.y) * TF_ROWS;
  const int end_row = min(first_row + TF_ROWS, p.rows);
  const int last_row = p.rows - 1;  // sums past it are read here, never used
  const int first = p.groups * int(slice) / TF_SK, last = p.groups * (int(slice) + 1) / TF_SK;
  const device uchar* blocks = W + (size_t)tile * p.groups * QK_CODES;

  constexpr auto desc = matmul2d_descriptor(TF_ROWS < 16 ? TF_ROWS : 16, QK_NT, QK_GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> xs((device bfloat*)X, dextents<int32_t, 2>(p.k, end_row));
  // Metal refuses static extents on packed codes, whose implied stride must be 64 bytes.
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)blocks, dextents<int32_t, 2>(QK_GS, QK_NT));
  auto LX = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
  auto P = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(LX)>, decltype(b0), float>();

  const auto home = P.get_multidimensional_index(ushort(0));
  const int c0 = home[0], r0 = home[1];
  const device uint* columns = SB + (size_t)tile * QK_NT + c0;

  float C[QK_VALUES];
  TF_QK_EACH for (int i = 0; i < QK_VALUES; i++) C[i] = 0.0f;

  // A lane's scales and biases for each column set. The next group's are on their way while this group's ops run.
  uint4 next[QK_SETS];
  TF_QK_EACH for (int s = 0; s < QK_SETS; s++) next[s] = *(const device uint4*)(columns + (size_t)first * p.n + 16 * s);

  // Each group: 16 rows of x into registers, one op against the group's codes in place, scale and bias folded after.
  for (int g = first; g < last; g++) {
    float scale[QK_SETS][4];
    float bias[QK_SETS][4];
    const int ahead = g + 1 < last ? g + 1 : g;
    TF_QK_EACH for (int s = 0; s < QK_SETS; s++) {
      TF_QK_EACH for (int j = 0; j < 4; j++) {
        scale[s][j] = as_type<float>(next[s][j] << 16);
        bias[s][j] = as_type<float>(next[s][j] & 0xFFFF0000u);
      }
      next[s] = *(const device uint4*)(columns + (size_t)ahead * p.n + 16 * s);
    }
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b((device uchar*)blocks + (size_t)g * QK_CODES, dextents<int32_t, 2>(QK_GS, QK_NT));
    const device float* group_sums = SUMS + (size_t)g * p.stride;
    // The compiler has to unroll this, or the running sums leave registers. An odd run stays out of it for that reason.
    TF_QK_EACH for (int part = 0; part < QK_RUNS / 2; part++) {
      auto a = xs.slice(g * QK_GS, first_row + part * 16);
      LX.load(a);
      op.run(LX, b, P);
      const int row = first_row + part * 16 + r0;
      const float x0 = group_sums[min(row, last_row)], x1 = group_sums[min(row + 8, last_row)];
      TF_QK_EACH for (int s = 0; s < QK_SETS; s++) {
        TF_QK_EACH for (int j = 0; j < 4; j++) {
          C[tf_qk_kept(part * 2, s, j)] = fma(scale[s][j], P[tf_qk_place(0, s, j)], fma(bias[s][j], x0, C[tf_qk_kept(part * 2, s, j)]));
          C[tf_qk_kept(part * 2 + 1, s, j)] = fma(scale[s][j], P[tf_qk_place(1, s, j)], fma(bias[s][j], x1, C[tf_qk_kept(part * 2 + 1, s, j)]));
        }
      }
    }
#if (TF_ROWS / 8) % 2 == 1
    {
      auto a = xs.slice(g * QK_GS, first_row + QK_RUNS / 2 * 16);
      LX.load(a);
      op.run(LX, b, P);
      const float x0 = group_sums[min(first_row + QK_RUNS / 2 * 16 + r0, last_row)];
      TF_QK_EACH for (int s = 0; s < QK_SETS; s++) {
        TF_QK_EACH for (int j = 0; j < 4; j++) C[tf_qk_kept(QK_RUNS - 1, s, j)] = fma(scale[s][j], P[tf_qk_place(0, s, j)], fma(bias[s][j], x0, C[tf_qk_kept(QK_RUNS - 1, s, j)]));
      }
    }
#endif
  }

#if TF_SK > 1
  // Slices add up through threadgroup memory in slice order, as tf_lane_reg's do: no partial reaches device memory.
  threadgroup float part_sums[(TF_SK - 1) * QK_VALUES * 32];
  if (slice > 0) {
    TF_QK_EACH for (int i = 0; i < QK_VALUES; i++) part_sums[((slice - 1) * QK_VALUES + i) * 32 + lane] = C[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slice > 0) return;
  TF_QK_EACH for (int s = 0; s < TF_SK - 1; s++) {
    TF_QK_EACH for (int i = 0; i < QK_VALUES; i++) C[i] += part_sums[(s * QK_VALUES + i) * 32 + lane];
  }
#endif

  const int n0 = tile * QK_NT + c0;
  TF_QK_EACH for (int k = 0; k < QK_RUNS; k++) {
    const int m = first_row + r0 + 8 * k;
    if (m < end_row) {
      device bfloat* out = Y + (size_t)m * p.n;
      TF_QK_EACH for (int s = 0; s < QK_SETS; s++) {
        const vec<bfloat, 4> four = vec<bfloat, 4>(float4(C[tf_qk_kept(k, s, 0)], C[tf_qk_kept(k, s, 1)], C[tf_qk_kept(k, s, 2)], C[tf_qk_kept(k, s, 3)]));
        *(device vec<bfloat, 4>*)(out + n0 + 16 * s) = four;
      }
    }
  }
}
