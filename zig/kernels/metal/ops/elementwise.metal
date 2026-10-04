// The MTP head's element-wise MLX 0.32.3 kernels value for value, and its routed-expert combine in one launch.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// MLX's Maximum on floats: a NaN x wins, otherwise the larger (y on ties).
inline bfloat tf_max_bf16(bfloat x, bfloat y) {
  if (isnan(x)) return x;
  return x > y ? x : y;
}

// One combined output: fp32 products (g2_Multiply), then col_reduce_small's order with k lanes (row + 0, in order).
template <typename WP>
inline void tf_moe_combine_at(const device bfloat* Y, WP W, int k, int cols, uint3 at, device bfloat* OUT) {
  const int c = int(at.x), row = int(at.y);
  if (c >= cols) return;
  const device bfloat* y = Y + long(row) * k * cols;
  W += long(row) * k;
  float prod = static_cast<float>(y[c]) * W[0];
  float total = prod + 0.0f;
  for (int e = 1; e < k; ++e) {
    prod = static_cast<float>(y[long(e) * cols + c]) * W[e];
    const float v = prod + 0.0f;
    total = v + total;
  }
  OUT[long(row) * cols + c] = static_cast<bfloat>(total);
}

// tf:kernel tf_add_bf16 inputs=A,B,P outputs=OUT
[[kernel]] void tf_add_bf16(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const uint i = thread_position_in_grid.x;
  if (int(i) < P[0]) OUT[i] = A[i] + B[i];
}

// tf:kernel tf_relu2_bf16 inputs=X,P outputs=OUT
[[kernel]] void tf_relu2_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat16_t* OUT [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const uint i = thread_position_in_grid.x;
  if (int(i) >= P[0]) return;
  const bfloat m = tf_max_bf16(X[i], static_cast<bfloat>(0));
  OUT[i] = m * m;
}

// tf:kernel tf_scale_bf16 inputs=X,S,P outputs=OUT
[[kernel]] void tf_scale_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant bfloat16_t* S [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const uint i = thread_position_in_grid.x;
  if (int(i) < P[0]) OUT[i] = S[0] * X[i];
}

// tf:kernel tf_moe_combine inputs=Y,W,P outputs=OUT
[[kernel]] void tf_moe_combine(
  const device bfloat16_t* Y [[buffer(0)]],
  const constant float* W [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tf_moe_combine_at(Y, W, P[0], P[1], thread_position_in_grid, OUT);
}

// tf:kernel tf_moe_combine_rows inputs=Y,W,P outputs=OUT
[[kernel]] void tf_moe_combine_rows(
  const device bfloat16_t* Y [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device bfloat16_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tf_moe_combine_at(Y, W, P[0], P[1], thread_position_in_grid, OUT);
}

// tf:kernel tf_causal_mask_bf16 inputs=X,P outputs=OUT
[[kernel]] void tf_causal_mask_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat16_t* OUT [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const int col = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  const int n = P[0], rows = P[1];
  if (col >= n) return;
  const long at = (long(thread_position_in_grid.z) * rows + r) * n + col;
  // row r sits at key position n - rows + r; later keys take bf16's lowest (where(mask, s, finfo.min))
  OUT[at] = (n - rows + r >= col) ? X[at] : static_cast<bfloat>(-0x1.fep+127f);
}

// tf:kernel tf_gather_u32 inputs=TABLE,IDS,P outputs=OUT
[[kernel]] void tf_gather_u32(
  const device uint32_t* TABLE [[buffer(0)]],
  const constant uint32_t* IDS [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  device uint32_t* OUT [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const uint i = thread_position_in_grid.x;
  if (int(i) < P[0]) OUT[i] = TABLE[IDS[i]];
}
