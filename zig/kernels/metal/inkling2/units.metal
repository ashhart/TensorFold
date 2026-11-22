// The tensor-unit op Inkling 2 Mini's expert products are built on, after nax.h: FP32 sums of BF16 products.
using namespace metal;

// (lo | hi) += a (b0 | b1)^T, b's blocks as stored, [output][column]: nax.h's 16x32x16 op without relaxed precision.
inline void ink2_mma(thread tfp::frag<float>& lo, thread tfp::frag<float>& hi, thread const tfp::frag<bfloat>& a,
                     thread const tfp::frag<bfloat>& b0, thread const tfp::frag<bfloat>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto right = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, float>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// A simdgroup's partial sums into the threadgroup's block, and the eight simdgroups' partials added in order.
inline void ink2_part_put(threadgroup float* part, thread const tfp::frag<float>& lo, thread const tfp::frag<float>& hi, uint sg, uint lane) {
  for (uint e = 0u; e < 8u; ++e) {
    part[(sg * 16u + e) * 32u + lane] = lo[e];
    part[(sg * 16u + 8u + e) * 32u + lane] = hi[e];
  }
}
inline void ink2_part_sum(threadgroup float* part, thread tfp::frag<float>& lo, thread tfp::frag<float>& hi, uint lane) {
  for (uint e = 0u; e < 8u; ++e)
    for (uint o = 1u; o < 8u; ++o) {
      lo[e] += part[(o * 16u + e) * 32u + lane];
      hi[e] += part[(o * 16u + 8u + e) * 32u + lane];
    }
}
