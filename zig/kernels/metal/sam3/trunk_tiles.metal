// SAM3 mixed-precision trunk kernels: FP16 tile projections on the tensor unit and the FP16 row norm.
#include <metal_stdlib>
using namespace metal;

struct Sam3TileProject {
  uint rows;
  uint cols;
  uint depth;
  uint scatter;
  uint grid;
  uint tile;
};

struct Sam3TileNorm {
  uint width;
  uint gather;
  uint grid;
  uint tile;
  float eps;
};

// Grid row of tile-ordered row r: tiles column by column, row-major inside a tile.
inline uint sam3_trunk_grid_row(uint r, uint grid, uint tile) {
  const uint n = grid / tile, area = tile * tile;
  const uint t = r / area, inner = r % area;
  const uint tx = t / n, ty = t % n;
  return (ty * tile + inner / tile) * grid + tx * tile + inner % tile;
}

// One simdgroup's 32 × 32 block of x Wᵀ + b (x [rows, depth] f16, W [cols, depth] f16), FP32 accumulation.
inline void sam3_trunk_block(thread tfp::frag<float> (&acc)[2][2], const device half* x, const device half* w,
                             const device float* bias, constant Sam3TileProject& p, uint2 tg, uint sg, uint lane,
                             thread int& row, thread int& col, thread short2& home) {
  row = int(tg.x) * 64 + 32 * int(sg / 4);
  col = int(tg.y) * 128 + 32 * int(sg % 4);
  home = tfp::frag_home(ushort(lane));
  const int depth = int(p.depth);
  tfp::gemm_block<half, false, true, true, 256>(acc, x + long(row) * depth, w + long(col) * depth, depth, depth, depth,
                                                depth / 256, 32, 32, home);
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const float b = bias[col + 16 * j + home.x + TF_COL(e)];
        acc[i][j][e] = acc[i][j][e] + b;
      }
    }
  }
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sam3_tile_project(
    const device half* x [[buffer(0)]], const device half* w [[buffer(1)]], const device float* bias [[buffer(2)]],
    device float* out [[buffer(3)]], constant Sam3TileProject& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  tfp::frag<float> acc[2][2];
  int row, col;
  short2 home;
  sam3_trunk_block(acc, x, w, bias, p, tg.xy, sg, lane, row, col, home);
  tfp::gemm_store<float, true>(acc, out + long(row) * p.cols + col, int(p.cols), 32, 32, home);
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sam3_tile_project_gelu(
    const device half* x [[buffer(0)]], const device half* w [[buffer(1)]], const device float* bias [[buffer(2)]],
    device half* out [[buffer(3)]], constant Sam3TileProject& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  tfp::frag<float> acc[2][2];
  int row, col;
  short2 home;
  sam3_trunk_block(acc, x, w, bias, p, tg.xy, sg, lane, row, col, home);
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const float v = acc[i][j][e];
        acc[i][j][e] = (0.5f * v) * (1.0f + qwv_erf(v * 0.70710678118654752f));
      }
    }
  }
  tfp::gemm_store<half, true>(acc, out + long(row) * p.cols + col, int(p.cols), 32, 32, home);
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sam3_tile_project_residual(
    const device half* x [[buffer(0)]], const device half* w [[buffer(1)]], const device float* bias [[buffer(2)]],
    device float* out [[buffer(3)]], const device float* residual [[buffer(4)]],
    constant Sam3TileProject& p [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  tfp::frag<float> acc[2][2];
  int row, col;
  short2 home;
  sam3_trunk_block(acc, x, w, bias, p, tg.xy, sg, lane, row, col, home);
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const uint r = uint(row + 16 * i + home.y + (e >> 2) * 8);
        const uint c = uint(col + 16 * j + home.x + TF_COL(e));
        const uint d = p.scatter != 0 ? sam3_trunk_grid_row(r, p.grid, p.tile) : r;
        const ulong at = ulong(d) * p.cols + c;
        out[at] = residual[at] + acc[i][j][e];
      }
    }
  }
}

// Layer norm of source row s(r) (its grid row when gathering) into FP16 row r; one threadgroup of 256 per row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void sam3_tile_norm_f16(
    const device float* x [[buffer(0)]], const device float* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]], device half* out [[buffer(3)]], constant Sam3TileNorm& p [[buffer(4)]],
    uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const uint source = p.gather != 0 ? sam3_trunk_grid_row(r, p.grid, p.tile) : r;
  const device float* in = x + ulong(source) * p.width;
  device half* to = out + ulong(r) * p.width;
  float sum = 0.0f;
  for (uint c = t; c < p.width; c += 256) {
    sum += in[c];
  }
  const float mean = tf_vis_sum(sum, part, 256, lane, sg) / float(p.width);
  float squares = 0.0f;
  for (uint c = t; c < p.width; c += 256) {
    const float v = in[c] - mean;
    squares += v * v;
  }
  const float scale = metal::precise::rsqrt(tf_vis_sum(squares, part, 256, lane, sg) / float(p.width) + p.eps);
  for (uint c = t; c < p.width; c += 256) {
    const float normed = (in[c] - mean) * scale * weight[c];
    to[c] = half(normed + bias[c]);
  }
}
